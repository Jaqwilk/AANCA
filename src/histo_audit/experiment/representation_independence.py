"""Strict schema-v2 representation-independence evidence production.

The builder operates only on the preselected development audit partition.  It
binds every matrix decision to exact feature bytes, sample/group identities, and
the semantic implementation/weight/preprocessing hashes already published by
the corresponding stage-eligible cache sidecar.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from histo_audit.corruption.controlled import (
    FeatureIndependenceEvidence,
    FeatureSpaceEvidence,
    canonical_sha256,
)
from histo_audit.pannuke.exceptions import PanNukeSemanticsError
from histo_audit.pannuke.io import ensure_derived_output_outside_raw
from histo_audit.pannuke.models import PanNukeValidationResult
from histo_audit.pannuke.publication import (
    ExclusiveBundlePublicationLock,
    PublishedPath,
    assert_mutable_publication_destination,
    publish_file_no_overwrite,
    rollback_owned_publications,
)
from histo_audit.representations.cache_provenance import (
    ordered_sample_ids_sha256,
    primary_cache_provenance_record,
    verify_frozen_cache_sidecar,
)
from histo_audit.representations.imagenet import ResNet18EmbeddingConfig
from histo_audit.representations.pannuke import (
    PanNukeCropConfig,
    PanNukeRepresentationArtifacts,
    _manifest_frame,
    build_pannuke_representation_cache,
    require_full_manifest_cache_disk_space,
)
from histo_audit.utils.run_tracking import atomic_write_json, sha256_file

from .reference_groups import deterministic_group_greedy_class_distribution_v1

_SHA256_FIELDS = (
    "encoder_implementation_sha256",
    "weights_sha256",
    "preprocessing_sha256",
    "sample_order_sha256",
    "dataset_manifest_sha256",
    "cache_recipe_sha256",
    "cache_file_sha256",
)
_PRIMARY_PROVENANCE_FIELDS = {
    "status",
    "encoder_id",
    *_SHA256_FIELDS,
}
_FEATURE_SPACE_FIELDS = {
    "representation_name",
    "feature_artifact_hash",
    "family",
    "implementation_hash",
    "weights_hash",
    "preprocessing_hash",
    "fitted_data_hash",
}
_ENTRY_FIELDS = {
    "schema_version",
    "matrix_version",
    "matrix_decision",
    "matrix_reason",
    "generator",
    "auditor",
    "independence_matrix_hash",
}


@dataclass(frozen=True, slots=True)
class IndependenceAuditorInput:
    """One exact audit-partition feature matrix and its frozen cache semantics."""

    representation_id: str
    family: str
    features: NDArray[np.generic]
    cache_provenance: Mapping[str, Any]
    matrix_decision: str
    matrix_reason: str


@dataclass(frozen=True, slots=True)
class RepresentationIndependenceArtifact:
    """One atomically published and read-back-validated schema-v2 matrix."""

    path: Path
    file_sha256: str
    entry_hashes: Mapping[str, str]
    audit_sample_count: int
    audit_group_count: int
    publication_record: PublishedPath


@dataclass(frozen=True, slots=True)
class PanNukeRepresentationIndependenceBundle:
    """Five aligned caches plus their same-process audit-slice evidence."""

    representations: PanNukeRepresentationArtifacts
    independence: RepresentationIndependenceArtifact


@dataclass(frozen=True, slots=True)
class _AnalysisManifestAuthority:
    canonical_manifest_sha256: str
    analysis_eligible_sample_order_sha256: str
    analysis_eligible_sample_count: int

    @classmethod
    def validated(
        cls,
        *,
        canonical_manifest_sha256: str,
        analysis_eligible_sample_order_sha256: str,
        analysis_eligible_sample_count: int,
    ) -> _AnalysisManifestAuthority:
        if not _is_sha256(canonical_manifest_sha256):
            raise ValueError(
                "analysis manifest authority canonical_manifest_sha256 must be a lowercase SHA-256"
            )
        if not _is_sha256(analysis_eligible_sample_order_sha256):
            raise ValueError(
                "analysis manifest authority analysis_eligible_sample_order_sha256 must "
                "be a lowercase SHA-256"
            )
        if type(analysis_eligible_sample_count) is not int or analysis_eligible_sample_count <= 0:
            raise ValueError(
                "analysis manifest authority analysis_eligible_sample_count must be a "
                "positive integer"
            )
        return cls(
            canonical_manifest_sha256=canonical_manifest_sha256,
            analysis_eligible_sample_order_sha256=analysis_eligible_sample_order_sha256,
            analysis_eligible_sample_count=analysis_eligible_sample_count,
        )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_source_analysis_manifest_authority(
    manifest_path: str | Path,
    authority: _AnalysisManifestAuthority,
) -> Path:
    """Authenticate the complete canonical analysis view before cache construction."""

    source = Path(manifest_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PanNuke manifest does not exist: {source}")
    frame, provenance, actual_manifest_sha256 = _manifest_frame(
        source,
        None,
        eligibility_scope="analysis",
    )
    identifiers = np.asarray(frame["sample_id"].astype(str).tolist(), dtype=np.str_)
    actual_order_sha256 = ordered_sample_ids_sha256(identifiers)
    actual_count = len(identifiers)
    mismatches: list[str] = []
    if actual_manifest_sha256 != authority.canonical_manifest_sha256:
        mismatches.append(
            "canonical_manifest_sha256="
            f"{actual_manifest_sha256} (expected {authority.canonical_manifest_sha256})"
        )
    if actual_order_sha256 != authority.analysis_eligible_sample_order_sha256:
        mismatches.append(
            "analysis_eligible_sample_order_sha256="
            f"{actual_order_sha256} "
            f"(expected {authority.analysis_eligible_sample_order_sha256})"
        )
    if actual_count != authority.analysis_eligible_sample_count:
        mismatches.append(
            "analysis_eligible_sample_count="
            f"{actual_count} (expected {authority.analysis_eligible_sample_count})"
        )
    provenance_expectations = {
        "manifest_eligible_instance_count": authority.analysis_eligible_sample_count,
        "output_sample_count": authority.analysis_eligible_sample_count,
        "manifest_eligible_sample_ids_sha256": (authority.analysis_eligible_sample_order_sha256),
        "output_sample_ids_sha256": authority.analysis_eligible_sample_order_sha256,
    }
    for field, expected in provenance_expectations.items():
        if provenance.get(field) != expected:
            mismatches.append(f"eligibility_provenance.{field}={provenance.get(field)!r}")
    if mismatches:
        raise ValueError(
            "analysis manifest authority mismatch; cache construction requires the exact "
            "canonical complete eligible view: " + "; ".join(mismatches)
        )
    return source


def _read_stable_cache_sidecar(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"published cache sidecar is missing: {path}")
    before = sha256_file(path)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"published cache sidecar is invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"published cache sidecar must be a mapping: {path}")
    if sha256_file(path) != before:
        raise RuntimeError(f"published cache sidecar changed during authority readback: {path}")
    return value


def _verify_published_analysis_manifest_authority(
    artifacts: PanNukeRepresentationArtifacts,
    manifest_path: Path,
    authority: _AnalysisManifestAuthority,
) -> None:
    """Reconcile every returned cache and sidecar with the frozen manifest authority."""

    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path) != authority.canonical_manifest_sha256
    ):
        raise RuntimeError("canonical analysis manifest changed during cache construction")
    identifiers = np.asarray(artifacts.crops.sample_ids, dtype=np.str_)
    if len(identifiers) != authority.analysis_eligible_sample_count:
        raise RuntimeError("published cache sample count differs from analysis manifest authority")
    if ordered_sample_ids_sha256(identifiers) != authority.analysis_eligible_sample_order_sha256:
        raise RuntimeError("published cache sample order differs from analysis manifest authority")
    crop_metadata = artifacts.crops.metadata
    if crop_metadata.get("manifest_sha256") != authority.canonical_manifest_sha256:
        raise RuntimeError(
            "published crop metadata manifest differs from analysis manifest authority"
        )
    recorded_manifest_path = crop_metadata.get("manifest_path")
    if (
        not isinstance(recorded_manifest_path, str)
        or Path(recorded_manifest_path).resolve() != manifest_path
    ):
        raise RuntimeError("published crop metadata points to a different analysis manifest")
    eligibility = crop_metadata.get("analysis_eligibility")
    if not isinstance(eligibility, Mapping):
        raise RuntimeError("published crop metadata lacks analysis eligibility provenance")
    eligibility_expectations = {
        "manifest_eligible_instance_count": authority.analysis_eligible_sample_count,
        "output_sample_count": authority.analysis_eligible_sample_count,
        "manifest_eligible_sample_ids_sha256": (authority.analysis_eligible_sample_order_sha256),
        "output_sample_ids_sha256": authority.analysis_eligible_sample_order_sha256,
    }
    if any(
        eligibility.get(field) != expected for field, expected in eligibility_expectations.items()
    ):
        raise RuntimeError(
            "published cache eligibility provenance differs from analysis manifest authority"
        )

    if artifacts.context_embeddings is None or artifacts.context_morphometrics is None:
        raise RuntimeError("complete authority verification requires all five cache families")
    aligned_identifiers = {
        "target-highlighted embeddings": artifacts.embeddings.sample_ids,
        "context embeddings": artifacts.context_embeddings.sample_ids,
        "context plus target morphometrics": artifacts.context_morphometrics.sample_ids,
    }
    for role, values in aligned_identifiers.items():
        if not np.array_equal(identifiers, np.asarray(values, dtype=np.str_)):
            raise RuntimeError(f"{role} sample order differs from analysis manifest authority")
    if artifacts.engineered.values.shape[0] != authority.analysis_eligible_sample_count:
        raise RuntimeError("engineered feature count differs from analysis manifest authority")

    cache_contracts = {
        "crop": (
            artifacts.crop_cache_path,
            artifacts.crop_metadata_path,
            "pannuke_component_covering_target_crops",
            (
                "context_rgb_plus_component_covering_projected_binary_target_mask_and_"
                "raw_instance_identity"
            ),
        ),
        "engineered": (
            artifacts.engineered_cache_path,
            artifacts.engineered_metadata_path,
            "engineered_target_features",
            "context_rgb_plus_binary_target_mask",
        ),
        "target-highlighted": (
            artifacts.embeddings.cache_path,
            artifacts.embeddings.metadata_path,
            "imagenet_target_highlighted_embeddings",
            "target_highlighted_rgb",
        ),
        "context": (
            artifacts.context_embeddings.cache_path,
            artifacts.context_embeddings.metadata_path,
            "imagenet_resnet18_context_embeddings",
            "context_rgb",
        ),
        "context plus target morphometrics": (
            artifacts.context_morphometrics.cache_path,
            artifacts.context_morphometrics.metadata_path,
            "imagenet_context_embeddings_plus_target_morphometrics",
            "context_rgb_plus_target_morphometrics",
        ),
    }
    for role, (
        cache_path,
        sidecar_path,
        representation_id,
        input_variant,
    ) in cache_contracts.items():
        if cache_path is None or sidecar_path is None:
            raise RuntimeError(f"published {role} cache lacks an NPZ/sidecar path")
        verification = verify_frozen_cache_sidecar(
            Path(cache_path).resolve(),
            expected_manifest_sha256=authority.canonical_manifest_sha256,
            expected_representation_id=representation_id,
            expected_input_variant=input_variant,
        )
        if verification.sidecar_path != Path(sidecar_path).resolve():
            raise RuntimeError(f"published {role} cache points to a different sidecar")
        metadata = verification.metadata
        record = metadata.get("primary_cache_provenance")
        if not isinstance(record, Mapping) or dict(record) != primary_cache_provenance_record(
            metadata
        ):
            raise RuntimeError(f"published {role} cache lacks exact primary_cache_provenance")
        if (
            record.get("dataset_manifest_sha256") != authority.canonical_manifest_sha256
            or record.get("sample_order_sha256") != authority.analysis_eligible_sample_order_sha256
            or metadata.get("sample_count") != authority.analysis_eligible_sample_count
        ):
            raise RuntimeError(f"published {role} cache differs from analysis manifest authority")


def _feature_matrix(value: NDArray[np.generic], *, location: str) -> NDArray[np.generic]:
    array = np.asarray(value)
    if (
        array.ndim != 2
        or array.shape[0] == 0
        or array.shape[1] == 0
        or array.dtype.hasobject
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{location} must be a finite non-empty numeric matrix")
    return array


def _identity_vector(
    value: Sequence[str] | NDArray[np.str_], *, location: str, unique: bool
) -> NDArray[np.str_]:
    array = np.asarray(value, dtype=np.str_)
    if array.ndim != 1 or len(array) == 0 or any(not item for item in array.tolist()):
        raise ValueError(f"{location} must be a non-empty one-dimensional string vector")
    if unique and len(set(array.tolist())) != len(array):
        raise ValueError(f"{location} must contain unique identities")
    return array


def _available_cache_provenance(value: Mapping[str, Any], *, location: str) -> dict[str, Any]:
    record = dict(value)
    if set(record) != _PRIMARY_PROVENANCE_FIELDS or record.get("status") != "available":
        raise ValueError(f"{location} must be exact available primary_cache_provenance")
    if not isinstance(record.get("encoder_id"), str) or not record["encoder_id"].strip():
        raise ValueError(f"{location}.encoder_id must be explicit")
    if any(not _is_sha256(record.get(field)) for field in _SHA256_FIELDS):
        raise ValueError(f"{location} contains an invalid or unavailable SHA-256 binding")
    return record


def _parse_feature_space(value: object, *, location: str) -> FeatureSpaceEvidence:
    if not isinstance(value, Mapping) or set(value) != _FEATURE_SPACE_FIELDS:
        raise ValueError(f"{location} fields are invalid")
    evidence = FeatureSpaceEvidence(
        representation_name=str(value["representation_name"]),
        feature_artifact_hash=str(value["feature_artifact_hash"]),
        family=str(value["family"]),
        implementation_hash=str(value["implementation_hash"]),
        weights_hash=str(value["weights_hash"]),
        preprocessing_hash=str(value["preprocessing_hash"]),
        fitted_data_hash=str(value["fitted_data_hash"]),
    )
    evidence.validate()
    return evidence


def _parse_entry(value: object, *, location: str) -> FeatureIndependenceEvidence:
    if (
        not isinstance(value, Mapping)
        or set(value) != _ENTRY_FIELDS
        or value.get("schema_version") != 1
    ):
        raise ValueError(f"{location} schema/fields are invalid")
    evidence = FeatureIndependenceEvidence(
        matrix_version=str(value["matrix_version"]),
        matrix_decision=str(value["matrix_decision"]),
        matrix_reason=str(value["matrix_reason"]),
        generator=_parse_feature_space(value["generator"], location=f"{location}.generator"),
        auditor=_parse_feature_space(value["auditor"], location=f"{location}.auditor"),
        independence_matrix_hash=str(value["independence_matrix_hash"]),
    )
    evidence.validate()
    return evidence


def validate_representation_independence_payload(
    payload: object,
    *,
    expected_representation_ids: Sequence[str] | None = None,
) -> dict[str, FeatureIndependenceEvidence]:
    """Validate the exact schema-v2 matrix and its cross-entry semantics."""

    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"schema_version", "entries"}
        or payload.get("schema_version") != 2
    ):
        raise ValueError("representation-independence matrix must use strict schema_version=2")
    entries = payload.get("entries")
    if not isinstance(entries, Mapping) or not entries:
        raise ValueError("representation-independence entries must be a non-empty mapping")
    parsed: dict[str, FeatureIndependenceEvidence] = {}
    for raw_identifier, raw_entry in entries.items():
        identifier = str(raw_identifier)
        if not identifier.strip() or identifier in parsed:
            raise ValueError("representation-independence entry identifiers are invalid")
        evidence = _parse_entry(raw_entry, location=f"entries.{identifier}")
        if evidence.auditor.representation_name != identifier:
            raise ValueError(f"matrix key and auditor identity differ for {identifier}")
        parsed[identifier] = evidence
    if expected_representation_ids is not None:
        expected = {str(value) for value in expected_representation_ids}
        if not expected or len(expected) != len(tuple(expected_representation_ids)):
            raise ValueError("expected representation IDs must be unique and non-empty")
        if set(parsed) != expected:
            raise ValueError(
                "representation-independence entries differ from expected available "
                f"representations: expected={sorted(expected)}, actual={sorted(parsed)}"
            )
    generators = {evidence.generator for evidence in parsed.values()}
    versions = {evidence.matrix_version for evidence in parsed.values()}
    fitted_hashes = {
        (evidence.generator.fitted_data_hash, evidence.auditor.fitted_data_hash)
        for evidence in parsed.values()
    }
    if len(generators) != 1:
        raise ValueError("all independence entries must bind one exact generator space")
    generator = next(iter(generators))
    if generator.representation_name != "morphology_only_v1" or generator.family != "morphology":
        raise ValueError("independence generator must be morphology_only_v1/morphology")
    if len(versions) != 1:
        raise ValueError("all independence entries must use one matrix_version")
    if fitted_hashes != {(generator.fitted_data_hash, generator.fitted_data_hash)}:
        raise ValueError("generator and auditors must bind one audit sample/group set")
    for identifier, evidence in parsed.items():
        if evidence.auditor.family == "engineered" and (
            evidence.matrix_decision != "not_independent"
        ):
            raise ValueError(
                f"engineered auditor {identifier} overlaps morphology_only_v1 and must be "
                "not_independent"
            )
        if evidence.matrix_decision == "verified_independent" and (
            evidence.auditor.space_signature() == generator.space_signature()
            or evidence.auditor.feature_artifact_hash == generator.feature_artifact_hash
        ):
            raise ValueError(
                f"identical feature-space signatures or bytes cannot certify {identifier}"
            )
    return parsed


def validate_representation_independence_file(
    path: str | Path,
    *,
    expected_representation_ids: Sequence[str] | None = None,
) -> dict[str, FeatureIndependenceEvidence]:
    """Read and validate one stable strict-v2 JSON artifact without mutation."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"representation-independence artifact is missing: {source}")
    before = sha256_file(source)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"representation-independence artifact is invalid JSON: {source}"
        ) from error
    parsed = validate_representation_independence_payload(
        payload,
        expected_representation_ids=expected_representation_ids,
    )
    if sha256_file(source) != before:
        raise RuntimeError("representation-independence artifact changed during validation")
    return parsed


def build_representation_independence_payload(
    *,
    generator_features: NDArray[np.generic],
    audit_sample_ids: Sequence[str] | NDArray[np.str_],
    audit_group_ids: Sequence[str] | NDArray[np.str_],
    generator_cache_provenance: Mapping[str, Any],
    auditors: Sequence[IndependenceAuditorInput],
    matrix_version: str = "pannuke_primary_representation_independence_v2",
) -> dict[str, Any]:
    """Build deterministic strict-v2 evidence from audit-only feature matrices."""

    if not matrix_version.strip():
        raise ValueError("matrix_version must be explicit")
    sample_ids = _identity_vector(audit_sample_ids, location="audit_sample_ids", unique=True)
    group_ids = _identity_vector(audit_group_ids, location="audit_group_ids", unique=False)
    generator_values = _feature_matrix(generator_features, location="generator_features")
    if len(sample_ids) != len(group_ids) or len(sample_ids) != len(generator_values):
        raise ValueError("generator features and audit sample/group identities must align")
    generator_provenance = _available_cache_provenance(
        generator_cache_provenance,
        location="generator_cache_provenance",
    )
    fitted_data_hash = canonical_sha256(
        {"sample_ids": sample_ids.tolist(), "group_ids": group_ids.tolist()}
    )
    generator = FeatureSpaceEvidence.from_array(
        generator_values,
        representation_name="morphology_only_v1",
        family="morphology",
        implementation_hash=str(generator_provenance["encoder_implementation_sha256"]),
        weights_hash=str(generator_provenance["weights_sha256"]),
        preprocessing_hash=str(generator_provenance["preprocessing_sha256"]),
        fitted_data_hash=fitted_data_hash,
    )
    if not auditors:
        raise ValueError("at least one available auditor representation is required")
    entries: dict[str, Any] = {}
    for auditor_input in auditors:
        identifier = str(auditor_input.representation_id)
        family = str(auditor_input.family)
        if not identifier.strip() or identifier in entries or not family.strip():
            raise ValueError(
                "auditor representation identities/families must be unique and explicit"
            )
        values = _feature_matrix(
            auditor_input.features,
            location=f"auditor {identifier} features",
        )
        if len(values) != len(sample_ids):
            raise ValueError(f"auditor {identifier} does not align with the audit partition")
        provenance = _available_cache_provenance(
            auditor_input.cache_provenance,
            location=f"auditor {identifier} cache provenance",
        )
        if auditor_input.matrix_decision not in {"verified_independent", "not_independent"}:
            raise ValueError(f"auditor {identifier} has an invalid matrix decision")
        if not auditor_input.matrix_reason.strip():
            raise ValueError(f"auditor {identifier} requires an explicit matrix reason")
        if family == "engineered" and auditor_input.matrix_decision != "not_independent":
            raise ValueError(
                "engineered target features contain morphology_only_v1 and cannot be certified "
                "independent"
            )
        auditor = FeatureSpaceEvidence.from_array(
            values,
            representation_name=identifier,
            family=family,
            implementation_hash=str(provenance["encoder_implementation_sha256"]),
            weights_hash=str(provenance["weights_sha256"]),
            preprocessing_hash=str(provenance["preprocessing_sha256"]),
            fitted_data_hash=fitted_data_hash,
        )
        evidence = FeatureIndependenceEvidence.create(
            matrix_version=matrix_version,
            matrix_decision=auditor_input.matrix_decision,
            matrix_reason=auditor_input.matrix_reason,
            generator=generator,
            auditor=auditor,
        )
        entries[identifier] = evidence.as_dict()
    payload = {
        "schema_version": 2,
        "entries": {identifier: entries[identifier] for identifier in sorted(entries)},
    }
    validate_representation_independence_payload(
        payload,
        expected_representation_ids=tuple(entries),
    )
    return payload


def publish_representation_independence_artifact(
    payload: Mapping[str, Any],
    output_path: str | Path,
    *,
    expected_representation_ids: Sequence[str] | None = None,
    audit_sample_count: int,
    audit_group_count: int,
) -> RepresentationIndependenceArtifact:
    """Validate, atomically publish without overwrite, and read back one matrix."""

    parsed = validate_representation_independence_payload(
        payload,
        expected_representation_ids=expected_representation_ids,
    )
    destination = assert_mutable_publication_destination(
        output_path,
        role="representation-independence artifact",
    )
    if destination.suffix.casefold() != ".json":
        raise ValueError("representation-independence artifact must end in .json")
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite independence artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = assert_mutable_publication_destination(
        destination,
        role="representation-independence artifact",
    )
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temporary_directory:
        staged = Path(temporary_directory) / destination.name
        atomic_write_json(staged, dict(payload))
        validate_representation_independence_file(
            staged,
            expected_representation_ids=expected_representation_ids,
        )
        publication = None
        try:
            publication = publish_file_no_overwrite(staged, destination)
            readback = validate_representation_independence_file(
                destination,
                expected_representation_ids=expected_representation_ids,
            )
            if readback != parsed:
                raise RuntimeError("independence readback semantics differ from staged payload")
            result = RepresentationIndependenceArtifact(
                path=destination,
                file_sha256=sha256_file(destination),
                entry_hashes={
                    identifier: evidence.independence_matrix_hash
                    for identifier, evidence in sorted(parsed.items())
                },
                audit_sample_count=audit_sample_count,
                audit_group_count=audit_group_count,
                publication_record=publication,
            )
        except BaseException as error:
            if publication is not None:
                try:
                    rollback_owned_publications([publication])
                except RuntimeError as rollback_error:
                    raise RuntimeError(
                        "independence post-publication verification failed and "
                        "ownership-safe rollback was incomplete: "
                        f"{rollback_error}"
                    ) from error
            raise
    return result


def _cache_primary_provenance(cache_path: Path) -> dict[str, Any]:
    verification = verify_frozen_cache_sidecar(cache_path)
    metadata = verification.metadata
    record = metadata.get("primary_cache_provenance")
    if not isinstance(record, Mapping) or dict(record) != primary_cache_provenance_record(metadata):
        raise ValueError(f"cache lacks exact producer primary_cache_provenance: {cache_path}")
    return _available_cache_provenance(record, location=str(cache_path))


def build_pannuke_representation_independence_artifact(
    artifacts: PanNukeRepresentationArtifacts,
    output_path: str | Path,
    *,
    class_order: tuple[int, ...],
    development_official_folds: tuple[int, ...],
    final_test_fold: int,
    reference_validation_fraction_groups: float,
    split_seed: int,
    matrix_version: str = "pannuke_primary_representation_independence_v2",
) -> RepresentationIndependenceArtifact:
    """Produce the real PanNuke audit-partition matrix without using final-fold outcomes."""

    crops = artifacts.crops
    if artifacts.context_embeddings is None:
        raise ValueError("context embeddings are required for the primary independence matrix")
    if not np.array_equal(crops.sample_ids, artifacts.embeddings.sample_ids) or not np.array_equal(
        crops.sample_ids, artifacts.context_embeddings.sample_ids
    ):
        raise ValueError("representation sample order differs from the crop cache")
    if artifacts.engineered.values.shape[0] != len(crops.sample_ids):
        raise ValueError("engineered features do not align with the crop cache")
    development_mask = np.isin(crops.official_folds, development_official_folds)
    final_mask = crops.official_folds == final_test_fold
    if not development_mask.any() or not final_mask.any() or np.any(development_mask & final_mask):
        raise ValueError("development/final fold policy is invalid or empty")
    if set(np.unique(crops.official_folds).tolist()) != {
        *development_official_folds,
        final_test_fold,
    }:
        raise ValueError("crop folds differ from the frozen development/final fold policy")
    for group in np.unique(crops.group_ids):
        if len(np.unique(crops.official_folds[crops.group_ids == group])) != 1:
            raise ValueError(f"source group spans official folds: {group}")
    validation_groups = deterministic_group_greedy_class_distribution_v1(
        crops.pre_corruption_labels[development_mask],
        crops.group_ids[development_mask],
        class_order=class_order,
        fraction=reference_validation_fraction_groups,
        seed=split_seed,
    )
    audit_mask = development_mask & ~np.isin(crops.group_ids, validation_groups)
    audit_indices = np.flatnonzero(audit_mask)
    if len(audit_indices) == 0 or set(crops.pre_corruption_labels[audit_indices].tolist()) != set(
        class_order
    ):
        raise ValueError("development audit partition is empty or lacks a frozen class")
    morphology_columns = tuple(
        index
        for index, name in enumerate(artifacts.engineered.names)
        if name.startswith("morphology.")
    )
    if not morphology_columns:
        raise ValueError("engineered cache contains no morphology_only_v1 columns")
    morphology_audit = np.asarray(
        artifacts.engineered.values[np.ix_(audit_indices, morphology_columns)],
        dtype=np.float64,
    )
    engineered_audit = np.asarray(artifacts.engineered.values[audit_indices])
    context_audit = np.asarray(artifacts.context_embeddings.embeddings[audit_indices])
    highlighted_audit = np.asarray(artifacts.embeddings.embeddings[audit_indices])
    engineered_provenance = _cache_primary_provenance(artifacts.engineered_cache_path)
    context_cache = artifacts.context_embeddings.cache_path
    highlighted_cache = artifacts.embeddings.cache_path
    if context_cache is None or highlighted_cache is None:
        raise ValueError("ImageNet cache paths are required for independence provenance")
    context_provenance = _cache_primary_provenance(context_cache)
    highlighted_provenance = _cache_primary_provenance(highlighted_cache)
    payload = build_representation_independence_payload(
        generator_features=morphology_audit,
        audit_sample_ids=crops.sample_ids[audit_indices],
        audit_group_ids=crops.group_ids[audit_indices],
        generator_cache_provenance=engineered_provenance,
        auditors=(
            IndependenceAuditorInput(
                representation_id="engineered_target_features",
                family="engineered",
                features=engineered_audit,
                cache_provenance=engineered_provenance,
                matrix_decision="not_independent",
                matrix_reason=(
                    "The engineered auditor contains the exact morphology_only_v1 generator "
                    "columns; instance-dependent results are circularity_risk."
                ),
            ),
            IndependenceAuditorInput(
                representation_id="imagenet_resnet18_context",
                family="imagenet",
                features=context_audit,
                cache_provenance=context_provenance,
                matrix_decision="verified_independent",
                matrix_reason=(
                    "The frozen context-RGB ResNet representation does not consume the "
                    "engineered morphology_only_v1 feature vector used by the generator."
                ),
            ),
            IndependenceAuditorInput(
                representation_id="imagenet_resnet18_highlighted",
                family="imagenet",
                features=highlighted_audit,
                cache_provenance=highlighted_provenance,
                matrix_decision="verified_independent",
                matrix_reason=(
                    "The frozen target-highlighted ResNet representation does not consume the "
                    "engineered morphology_only_v1 feature vector used by the generator."
                ),
            ),
        ),
        matrix_version=matrix_version,
    )
    return publish_representation_independence_artifact(
        payload,
        output_path,
        expected_representation_ids=(
            "engineered_target_features",
            "imagenet_resnet18_context",
            "imagenet_resnet18_highlighted",
        ),
        audit_sample_count=len(audit_indices),
        audit_group_count=len(np.unique(crops.group_ids[audit_indices])),
    )


def _rollback_representation_publication(
    artifacts: PanNukeRepresentationArtifacts,
    output: Path,
) -> None:
    from histo_audit.representations.pannuke_chunked import (
        rollback_pannuke_chunked_publication,
    )

    if rollback_pannuke_chunked_publication(artifacts):
        return
    publications = list(artifacts.publication_records)
    if not publications or publications[0].path != output:
        raise RuntimeError(
            "representation publication has no exact ownership records; foreign destination "
            f"preserved: {output}"
        )
    rollback_owned_publications(publications)


def build_pannuke_representation_cache_with_independence(
    validation: PanNukeValidationResult,
    manifest_path: str | Path,
    output_dir: str | Path,
    independence_output_path: str | Path,
    *,
    class_order: tuple[int, ...],
    development_official_folds: tuple[int, ...],
    final_test_fold: int,
    reference_validation_fraction_groups: float,
    split_seed: int,
    expected_canonical_manifest_sha256: str,
    expected_analysis_eligible_sample_order_sha256: str,
    expected_analysis_eligible_sample_count: int,
    sample_ids: tuple[str, ...] | None = None,
    crop_config: PanNukeCropConfig | None = None,
    resnet_config: ResNet18EmbeddingConfig | None = None,
    chunk_size: int | None = None,
    matrix_version: str = "pannuke_primary_representation_independence_v2",
) -> PanNukeRepresentationIndependenceBundle:
    """Publish all five caches and real audit-slice evidence in one process.

    The feature-space matrix is built while chunked extraction memmaps are still
    leased.  If evidence production fails, an owned chunked output is moved back
    into its hidden resumable workspace (or a small-path output is removed), so a
    failed command never leaves a public five-cache directory without its required
    independence artifact.
    """

    if sample_ids is not None:
        raise ValueError(
            "same-process primary independence publication requires the complete "
            "analysis-eligible manifest; sample_ids subsets are forbidden"
        )
    authority = _AnalysisManifestAuthority.validated(
        canonical_manifest_sha256=expected_canonical_manifest_sha256,
        analysis_eligible_sample_order_sha256=(expected_analysis_eligible_sample_order_sha256),
        analysis_eligible_sample_count=expected_analysis_eligible_sample_count,
    )

    output = ensure_derived_output_outside_raw(
        output_dir,
        validation.root,
        purpose="PanNuke representation output directory",
    )
    independence_output = ensure_derived_output_outside_raw(
        independence_output_path,
        validation.root,
        purpose="PanNuke representation-independence artifact",
    )
    try:
        output = assert_mutable_publication_destination(
            output_dir,
            role="PanNuke representation output directory",
        )
        independence_output = assert_mutable_publication_destination(
            independence_output_path,
            role="PanNuke representation-independence artifact",
        )
    except (NotADirectoryError, PermissionError, RuntimeError) as error:
        raise PanNukeSemanticsError(str(error)) from error
    if independence_output.suffix.casefold() != ".json":
        raise ValueError("representation-independence artifact must end in .json")
    if (
        independence_output == output
        or output in independence_output.parents
        or independence_output in output.parents
    ):
        raise ValueError(
            "representation-independence artifact and cache output directory must not "
            "contain or overlap one another"
        )
    require_full_manifest_cache_disk_space(
        manifest_path,
        output,
        sample_ids=None,
    )
    source_manifest = _verify_source_analysis_manifest_authority(
        manifest_path,
        authority,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    independence_output.parent.mkdir(parents=True, exist_ok=True)
    with ExclusiveBundlePublicationLock(
        (output, independence_output),
        role="PanNuke cache plus representation-independence bundle",
    ):
        if os.path.lexists(output):
            raise FileExistsError(f"representation output directory already exists: {output}")
        if os.path.lexists(independence_output):
            raise FileExistsError(
                f"representation-independence artifact already exists: {independence_output}"
            )
        artifacts = build_pannuke_representation_cache(
            validation,
            source_manifest,
            output,
            sample_ids=None,
            crop_config=crop_config,
            resnet_config=resnet_config,
            include_context_embeddings=True,
            chunk_size=chunk_size,
        )
        independence: RepresentationIndependenceArtifact | None = None
        try:
            _verify_published_analysis_manifest_authority(
                artifacts,
                source_manifest,
                authority,
            )
            independence = build_pannuke_representation_independence_artifact(
                artifacts,
                independence_output,
                class_order=class_order,
                development_official_folds=development_official_folds,
                final_test_fold=final_test_fold,
                reference_validation_fraction_groups=reference_validation_fraction_groups,
                split_seed=split_seed,
                matrix_version=matrix_version,
            )
            _verify_source_analysis_manifest_authority(
                source_manifest,
                authority,
            )
            _verify_published_analysis_manifest_authority(
                artifacts,
                source_manifest,
                authority,
            )
        except BaseException as evidence_error:
            rollback_errors: list[str] = []
            if independence is not None:
                try:
                    rollback_owned_publications([independence.publication_record])
                except BaseException as rollback_error:
                    rollback_errors.append(f"independence artifact: {rollback_error}")
            try:
                _rollback_representation_publication(
                    artifacts,
                    output,
                )
            except BaseException as rollback_error:
                rollback_errors.append(f"cache bundle: {rollback_error}")
            if rollback_errors:
                raise RuntimeError(
                    "representation-independence production failed and ownership-safe cache "
                    "bundle rollback was incomplete: " + "; ".join(rollback_errors)
                ) from evidence_error
            raise
    assert independence is not None
    return PanNukeRepresentationIndependenceBundle(
        representations=artifacts,
        independence=independence,
    )


__all__ = [
    "IndependenceAuditorInput",
    "PanNukeRepresentationIndependenceBundle",
    "RepresentationIndependenceArtifact",
    "build_pannuke_representation_cache_with_independence",
    "build_pannuke_representation_independence_artifact",
    "build_representation_independence_payload",
    "publish_representation_independence_artifact",
    "validate_representation_independence_file",
    "validate_representation_independence_payload",
]
