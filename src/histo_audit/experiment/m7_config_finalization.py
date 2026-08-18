"""Deterministically finalise the M7 primary and confirmatory study configs.

The finaliser is intentionally separate from the general CLI.  It consumes the
five immutable full-PanNuke representation caches and the already frozen M6/M7
evidence, projects their producer provenance into both study schemas, validates
the two configs together, and only then publishes explicit YAML outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from histo_audit.config import config_sha256, load_config, resolve_config
from histo_audit.corruption.controlled import array_artifact_sha256
from histo_audit.experiment import pannuke_primary_inputs as primary_inputs
from histo_audit.experiment import pilot_derived_parameters as pilot_parameters_module
from histo_audit.experiment.reference_groups import (
    deterministic_group_greedy_class_distribution_v1,
)
from histo_audit.experiment.study_contracts import (
    ConfirmatoryMatrixPlan,
    PrimaryMatrixPlan,
    build_confirmatory_matrix_plan,
    build_primary_matrix_plan,
    validate_primary_confirmatory_cross_config,
)
from histo_audit.models import cnn as cnn_module
from histo_audit.representations import pathology as pathology_module
from histo_audit.representations.cache_provenance import (
    FrozenCacheVerification,
    canonical_sha256,
    confirmatory_cache_provenance_record,
    primary_cache_provenance_record,
    verify_frozen_cache_sidecar,
)
from histo_audit.representations.pathology import (
    unavailable_optional_pathology_cache_provenance,
)
from histo_audit.utils.run_tracking import (
    atomic_write_bytes,
    require_run_stage_eligible,
    sha256_file,
    verify_run_integrity,
)
from histo_audit.workflows import preregistration as preregistration_workflow

_SHA256_LENGTH = 64
_PILOT_REPORT_RELATIVE_PATH = Path("reports/pilot_derived_primary_parameters.json")
_PRIMARY_REPRESENTATION_BY_ROLE = {
    "engineered": "engineered_target_features",
    "context": "imagenet_resnet18_context",
    "highlighted": "imagenet_resnet18_highlighted",
}
_CACHE_REPRESENTATION_BY_ROLE = {
    "crop": "pannuke_component_covering_target_crops",
    "engineered": "engineered_target_features",
    "highlighted": "imagenet_target_highlighted_embeddings",
    "context": "imagenet_resnet18_context_embeddings",
    "context_morphometrics": "imagenet_context_embeddings_plus_target_morphometrics",
}
_CONFIRMATORY_RECORD_BY_ROLE = {
    "context": "imagenet_context_embedding_cache",
    "highlighted": "imagenet_target_highlighted_embedding_cache",
    "context_morphometrics": "imagenet_context_morphometrics_cache",
}
_CNN_RECORD_SPECS = (
    {
        "record_id": "cnn_context_rgb_cache",
        "representation_id": "cnn_context_rgb_pixels",
        "encoder_identifier": "resnet18_imagenet1k_v1",
        "preprocessing_identifier": "confirmatory_context_rgb_224_v1",
        "input_variant": "context_rgb",
        "input_channels": 3,
        "required_arrays": ("context_rgb",),
    },
    {
        "record_id": "cnn_context_target_mask_cache",
        "representation_id": "cnn_context_target_mask_pixels",
        "encoder_identifier": "resnet18_imagenet1k_v1_zero_init_fourth_channel",
        "preprocessing_identifier": "confirmatory_context_rgb_plus_target_mask_224_v1",
        "input_variant": "context_rgb_plus_binary_target_mask",
        "input_channels": 4,
        "required_arrays": ("context_rgb", "target_masks"),
    },
)
_PILOT_SOURCE_ARTIFACTS = {
    "immutable_marker_sha256": Path(".immutable.json"),
    "metrics_sha256": Path("metrics.json"),
    "selection_record_sha256": Path("selected_groups_and_samples.json"),
    "crop_cache_sha256": Path("representations/pannuke_crops.npz"),
    "crop_sidecar_sha256": Path("representations/pannuke_crops.npz.metadata.json"),
    "highlighted_embedding_cache_sha256": Path(
        "representations/pannuke_resnet18_target_highlighted_embeddings.npz"
    ),
    "highlighted_embedding_sidecar_sha256": Path(
        "representations/pannuke_resnet18_target_highlighted_embeddings.npz.metadata.json"
    ),
}
_FINAL_REFERENCE_PRIVACY_ARTIFACT = Path("final_reference_privacy_reconciliation.json")


class M7ConfigFinalizationError(ValueError):
    """Raised when M7 evidence cannot safely produce frozen study configs."""


@dataclass(frozen=True, slots=True)
class M7CachePaths:
    """The exact five full-PanNuke immutable cache artifacts."""

    crop: Path
    engineered: Path
    highlighted: Path
    context: Path
    context_morphometrics: Path


@dataclass(frozen=True, slots=True)
class M7ConfigFinalizationResult:
    """Published config identities and their validated expansion plans."""

    primary_output_path: Path
    confirmatory_output_path: Path
    primary_file_sha256: str
    confirmatory_file_sha256: str
    primary_config_sha256: str
    confirmatory_config_sha256: str
    primary_plan: PrimaryMatrixPlan
    confirmatory_plan: ConfirmatoryMatrixPlan
    cache_file_sha256_by_role: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary_output_path": str(self.primary_output_path),
            "confirmatory_output_path": str(self.confirmatory_output_path),
            "primary_file_sha256": self.primary_file_sha256,
            "confirmatory_file_sha256": self.confirmatory_file_sha256,
            "primary_config_sha256": self.primary_config_sha256,
            "confirmatory_config_sha256": self.confirmatory_config_sha256,
            "primary_plan_semantic_sha256": canonical_sha256(self.primary_plan.as_dict()),
            "confirmatory_plan_semantic_sha256": canonical_sha256(self.confirmatory_plan.as_dict()),
            "cache_file_sha256_by_role": dict(self.cache_file_sha256_by_role),
        }


@dataclass(frozen=True, slots=True)
class _CanonicalPilotRun:
    path: Path
    run_id: str
    artifact_root_sha256: str

    @property
    def source_artifact_paths(self) -> tuple[Path, ...]:
        return tuple(self.path / relative for relative in _PILOT_SOURCE_ARTIFACTS.values())


def _load_json_object(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise M7ConfigFinalizationError(f"{role} is missing: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise M7ConfigFinalizationError(f"cannot read strict {role}: {exc}") from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise M7ConfigFinalizationError(f"{role} must be a JSON object")
    return cast(dict[str, Any], value)


def _require_sha256(value: object, role: str) -> str:
    digest = str(value).casefold()
    if len(digest) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise M7ConfigFinalizationError(f"{role} must be a lowercase SHA-256")
    return digest


def _assert_canonical_pilot_run_current(pilot: _CanonicalPilotRun) -> None:
    integrity = verify_run_integrity(pilot.path)
    if not integrity.valid:
        details = "; ".join(integrity.errors) or "artifact identity mismatch"
        raise M7ConfigFinalizationError(
            f"canonical M6 pilot run failed sealed integrity: {details}"
        )
    if (
        integrity.run_id != pilot.run_id
        or integrity.expected_root_sha256 != pilot.artifact_root_sha256
        or integrity.actual_root_sha256 != pilot.artifact_root_sha256
        or not integrity.registry_record_present
    ):
        raise M7ConfigFinalizationError(
            "canonical M6 pilot run differs from its frozen run/root authority"
        )
    try:
        require_run_stage_eligible(pilot.path, integrity=integrity)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise M7ConfigFinalizationError(
            f"canonical M6 pilot run is not stage eligible: {exc}"
        ) from exc
    marker = _load_json_object(pilot.path / ".immutable.json", "canonical M6 pilot marker")
    if marker.get("status") != "completed" or marker.get("run_id") != pilot.run_id:
        raise M7ConfigFinalizationError(
            "canonical M6 pilot must be a completed sealed run with the matching run_id"
        )
    try:
        resolved = load_config(pilot.path / "resolved_config.yaml")
    except (FileNotFoundError, OSError, ValueError, yaml.YAMLError) as exc:
        raise M7ConfigFinalizationError(f"cannot load canonical M6 pilot config: {exc}") from exc
    if resolved.get("experiment_name") != "pannuke_pilot":
        raise M7ConfigFinalizationError("canonical M6 authority must identify a pannuke_pilot run")
    selection = _load_json_object(
        pilot.path / "selected_groups_and_samples.json",
        "canonical M6 pilot selection evidence",
    )
    if selection.get("development_official_folds") != [1, 2]:
        raise M7ConfigFinalizationError(
            "canonical M6 pilot selection is not restricted to official folds 1/2"
        )
    for field_name in (
        "final_reference_class_labels_read",
        "final_reference_sample_ids_read",
        "final_reference_representations_extracted",
        "final_reference_outcomes_used",
    ):
        if selection.get(field_name) is not False:
            raise M7ConfigFinalizationError(
                f"canonical M6 pilot selection does not prove {field_name}=false"
            )
    privacy = _load_json_object(
        pilot.path / _FINAL_REFERENCE_PRIVACY_ARTIFACT,
        "canonical M6 final-reference privacy evidence",
    )
    if (
        privacy.get("status") != "passed"
        or privacy.get("policy") != "final_reference_identity_and_outcome_nonpublication_v1"
        or privacy.get("final_reference_official_fold") != 3
    ):
        raise M7ConfigFinalizationError(
            "canonical M6 pilot lacks the required passed final-reference privacy policy"
        )


def _canonical_pilot_run_from_draft(
    primary_draft: Mapping[str, Any],
    project_root: Path,
) -> _CanonicalPilotRun:
    binding = primary_draft.get("pilot_derived_parameters")
    if not isinstance(binding, Mapping):
        raise M7ConfigFinalizationError("primary draft lacks its canonical M6 pilot authority")
    raw_run_id = binding.get("source_pilot_run_id")
    if not isinstance(raw_run_id, str) or not raw_run_id.strip():
        raise M7ConfigFinalizationError("canonical M6 pilot run_id is absent")
    run_id = raw_run_id.strip()
    if (
        Path(run_id).is_absolute()
        or Path(run_id).name != run_id
        or run_id in {".", ".."}
        or "/" in run_id
        or "\\" in run_id
    ):
        raise M7ConfigFinalizationError("canonical M6 pilot run_id is not a safe basename")
    artifact_root = _require_sha256(
        binding.get("source_pilot_artifact_root_sha256"),
        "canonical M6 pilot artifact root",
    )
    runs_root = (project_root / "artifacts" / "runs").resolve()
    run_path = (runs_root / run_id).resolve()
    if run_path.parent != runs_root or run_path.name != run_id:
        raise M7ConfigFinalizationError(
            "canonical M6 pilot run must resolve directly under artifacts/runs"
        )
    pilot = _CanonicalPilotRun(run_path, run_id, artifact_root)
    _assert_canonical_pilot_run_current(pilot)
    return pilot


def _validate_pilot_source_artifacts(
    report_path: Path,
    pilot: _CanonicalPilotRun,
) -> None:
    payload = _load_json_object(report_path, "pilot-derived parameter report")
    source = payload.get("source_pilot")
    if not isinstance(source, Mapping):
        raise M7ConfigFinalizationError(
            "pilot-derived parameter report lacks source_pilot evidence"
        )
    for field_name, relative_path in _PILOT_SOURCE_ARTIFACTS.items():
        expected = _require_sha256(
            source.get(field_name),
            f"pilot-derived source_pilot.{field_name}",
        )
        source_path = pilot.path / relative_path
        if not source_path.is_file() or sha256_file(source_path) != expected:
            raise M7ConfigFinalizationError(
                "pilot-derived parameter report differs from canonical M6 source artifact "
                f"{relative_path.as_posix()}"
            )


def _verified_caches(paths: M7CachePaths) -> dict[str, FrozenCacheVerification]:
    output: dict[str, FrozenCacheVerification] = {}
    for role in _CACHE_REPRESENTATION_BY_ROLE:
        path = Path(getattr(paths, role)).expanduser().resolve()
        try:
            verification = verify_frozen_cache_sidecar(
                path,
                expected_representation_id=_CACHE_REPRESENTATION_BY_ROLE[role],
            )
            projected = primary_cache_provenance_record(verification.metadata)
        except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
            raise M7ConfigFinalizationError(
                f"{role} cache/sidecar failed frozen verification: {exc}"
            ) from exc
        if verification.metadata.get("primary_cache_provenance") != projected:
            raise M7ConfigFinalizationError(
                f"{role} sidecar primary_cache_provenance differs from its producer fields"
            )
        output[role] = verification

    common_fields = ("sample_order_sha256", "manifest_sha256", "raw_inventory_sha256")
    for field_name in common_fields:
        values = {str(item.metadata.get(field_name)) for item in output.values()}
        if len(values) != 1:
            raise M7ConfigFinalizationError(f"full PanNuke caches disagree on {field_name}")
        _require_sha256(next(iter(values)), f"cache {field_name}")
    weight_bindings = {
        (
            str(output[role].metadata.get("weight_identifier")),
            str(output[role].metadata.get("weights_sha256")),
        )
        for role in ("highlighted", "context", "context_morphometrics")
    }
    if len(weight_bindings) != 1:
        raise M7ConfigFinalizationError(
            "context, highlighted, and context+morphometrics caches use different weights"
        )
    _validate_cache_lineage(output)
    return output


def _validate_analysis_manifest_authority(
    primary_draft: Mapping[str, Any],
    caches: Mapping[str, FrozenCacheVerification],
) -> None:
    data = primary_draft.get("data")
    authority = data.get("analysis_manifest_authority") if isinstance(data, Mapping) else None
    expected_fields = {
        "canonical_manifest_sha256",
        "analysis_eligible_sample_order_sha256",
        "analysis_eligible_sample_count",
    }
    if not isinstance(authority, Mapping) or set(authority) != expected_fields:
        raise M7ConfigFinalizationError(
            "primary draft data.analysis_manifest_authority must contain exactly its "
            "manifest, sample-order, and sample-count authorities"
        )
    manifest_sha256 = _require_sha256(
        authority["canonical_manifest_sha256"],
        "analysis manifest authority canonical manifest",
    )
    sample_order_sha256 = _require_sha256(
        authority["analysis_eligible_sample_order_sha256"],
        "analysis manifest authority eligible sample order",
    )
    raw_count = authority["analysis_eligible_sample_count"]
    if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count <= 0:
        raise M7ConfigFinalizationError(
            "analysis manifest authority eligible sample count must be a positive integer"
        )
    for role, cache in caches.items():
        if (
            cache.metadata.get("manifest_sha256") != manifest_sha256
            or cache.metadata.get("sample_order_sha256") != sample_order_sha256
            or cache.metadata.get("sample_count") != raw_count
        ):
            raise M7ConfigFinalizationError(
                f"{role} cache differs from primary data.analysis_manifest_authority"
            )


def _validate_cache_lineage(caches: Mapping[str, FrozenCacheVerification]) -> None:
    crop = caches["crop"]
    crop_metadata = crop.metadata
    crop_arrays = crop_metadata.get("cache_array_sha256_by_name")
    if not isinstance(crop_arrays, Mapping) or not {
        "context_rgb",
        "target_masks",
        "target_highlighted_rgb",
    }.issubset(crop_arrays):
        raise M7ConfigFinalizationError(
            "crop sidecar lacks exact context, target-mask, or highlighted array hashes"
        )
    common_expected = {
        "crop_cache_file_sha256": crop.cache_file_sha256,
        "crop_cache_sidecar_file_sha256": crop.sidecar_file_sha256,
        "crop_cache_content_sha256": crop_metadata.get("cache_content_sha256"),
        "crop_manifest_sha256": crop_metadata.get("manifest_sha256"),
        "raw_inventory_sha256": crop_metadata.get("raw_inventory_sha256"),
        "sample_order_sha256": crop_metadata.get("sample_order_sha256"),
    }
    for role in ("engineered", "highlighted", "context"):
        binding = caches[role].metadata.get("source_crop_cache_binding")
        if not isinstance(binding, Mapping) or any(
            binding.get(field_name) != expected for field_name, expected in common_expected.items()
        ):
            raise M7ConfigFinalizationError(
                f"{role} cache is not bound to the exact full crop cache"
            )
    morph_binding = caches["context_morphometrics"].metadata.get(
        "component_engineered_cache_binding"
    )
    engineered = caches["engineered"]
    if not isinstance(morph_binding, Mapping) or (
        morph_binding.get("engineered_cache_file_sha256") != engineered.cache_file_sha256
        or morph_binding.get("engineered_cache_sidecar_file_sha256")
        != engineered.sidecar_file_sha256
        or morph_binding.get("engineered_cache_content_sha256")
        != engineered.metadata.get("cache_content_sha256")
    ):
        raise M7ConfigFinalizationError(
            "context+morphometrics cache is not bound to the exact engineered cache"
        )
    context = caches["context"]
    morph_metadata = caches["context_morphometrics"].metadata
    morph_encoder = morph_metadata.get("encoder_metadata")
    morph_preprocessing = morph_metadata.get("preprocessing")
    morph_recipe = morph_metadata.get("cache_recipe")
    if (
        not isinstance(morph_encoder, Mapping)
        or morph_encoder.get("component_context_cache_file_sha256") != context.cache_file_sha256
        or morph_encoder.get("component_context_cache_content_sha256")
        != context.metadata.get("cache_content_sha256")
        or morph_encoder.get("component_context_encoder_metadata_sha256")
        != context.metadata.get("encoder_metadata_sha256")
        or not isinstance(morph_preprocessing, Mapping)
        or morph_preprocessing.get("context_embedding_preprocessing_identifier")
        != context.metadata.get("preprocessing_identifier")
        or morph_preprocessing.get("context_embedding_preprocessing_sha256")
        != context.metadata.get("preprocessing_sha256")
        or not isinstance(morph_recipe, Mapping)
        or morph_recipe.get("context_cache_recipe_sha256")
        != context.metadata.get("cache_recipe_sha256")
        or morph_recipe.get("context_sample_order_sha256")
        != context.metadata.get("sample_order_sha256")
    ):
        raise M7ConfigFinalizationError(
            "context+morphometrics cache is not bound to the exact context embedding cache"
        )


def derive_confirmatory_cnn_logical_provenance(
    crop: FrozenCacheVerification,
    *,
    weight_identifier: str,
    weights_sha256: str,
    input_size: int,
) -> dict[str, dict[str, Any]]:
    """Derive both CNN view records from one verified crop cache and runtime recipe."""

    if input_size < 32:
        raise M7ConfigFinalizationError("confirmatory CNN input_size must be at least 32")
    weights_digest = _require_sha256(weights_sha256, "confirmatory CNN weights_sha256")
    crop_arrays = crop.metadata.get("cache_array_sha256_by_name")
    if not isinstance(crop_arrays, Mapping):
        raise M7ConfigFinalizationError("crop sidecar lacks per-array hashes")
    source_file = Path(cnn_module.__file__).resolve()
    implementation_sha256 = sha256_file(source_file)
    output: dict[str, dict[str, Any]] = {}
    for raw_spec in _CNN_RECORD_SPECS:
        spec = cast(Mapping[str, Any], raw_spec)
        required_arrays = tuple(str(value) for value in spec["required_arrays"])
        if any(name not in crop_arrays for name in required_arrays):
            raise M7ConfigFinalizationError(
                f"crop sidecar lacks arrays required by {spec['record_id']}"
            )
        input_binding = {
            "schema_version": 1,
            "binding_type": "confirmatory_cnn_logical_crop_view_v1",
            "crop_cache_file_sha256": crop.cache_file_sha256,
            "crop_sidecar_semantic_sha256": crop.sidecar_semantic_sha256,
            "crop_cache_content_sha256": crop.metadata.get("cache_content_sha256"),
            "sample_order_sha256": crop.metadata.get("sample_order_sha256"),
            "manifest_sha256": crop.metadata.get("manifest_sha256"),
            "input_array_sha256_by_name": {
                name: str(crop_arrays[name]) for name in required_arrays
            },
        }
        encoder_metadata = {
            "schema_version": 1,
            "identifier": "confirmatory_resnet18_five_class_encoder_v1",
            "implementation_module": "histo_audit.models.cnn",
            "implementation_source_sha256": implementation_sha256,
            "architecture": "torchvision.resnet18",
            "class_order": [0, 1, 2, 3, 4],
            "input_variant": spec["input_variant"],
            "input_channels": spec["input_channels"],
            "output_classes": 5,
            "weight_identifier": weight_identifier,
            "weights_sha256": weights_digest,
            "fourth_channel_initialisation": (
                "zeros" if int(spec["input_channels"]) == 4 else None
            ),
        }
        preprocessing = {
            "schema_version": 1,
            "identifier": spec["preprocessing_identifier"],
            "implementation_module": "histo_audit.models.cnn._batch_tensor",
            "implementation_source_sha256": implementation_sha256,
            "input_size": input_size,
            "rgb_dtype": "uint8",
            "rgb_scale": "divide_by_255_to_float32",
            "rgb_resize": "bilinear_align_corners_false_antialias_true",
            "rgb_mean": [0.485, 0.456, 0.406],
            "rgb_standard_deviation": [0.229, 0.224, 0.225],
            "target_mask_resize": (
                "nearest_binary_unnormalised" if int(spec["input_channels"]) == 4 else None
            ),
            "logical_input_binding": input_binding,
        }
        record_id = str(spec["record_id"])
        output[record_id] = {
            "id": record_id,
            "representation_id": str(spec["representation_id"]),
            "status": "available",
            "cache_file_sha256": None,
            "sidecar_semantic_sha256": crop.sidecar_semantic_sha256,
            "sample_order_sha256": str(crop.metadata["sample_order_sha256"]),
            "manifest_sha256": str(crop.metadata["manifest_sha256"]),
            "encoder_identifier": str(spec["encoder_identifier"]),
            "encoder_metadata_sha256": canonical_sha256(encoder_metadata),
            "weight_identifier": weight_identifier,
            "weights_sha256": weights_digest,
            "preprocessing_identifier": str(spec["preprocessing_identifier"]),
            "preprocessing_sha256": canonical_sha256(preprocessing),
            "input_variant": str(spec["input_variant"]),
        }
    return output


def _validate_independence_matrix(
    path: Path,
    caches: Mapping[str, FrozenCacheVerification],
) -> str:
    try:
        entries = primary_inputs._load_independence_evidence_matrix(path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise M7ConfigFinalizationError(
            f"strict feature-independence matrix is invalid: {exc}"
        ) from exc
    expected = set(_PRIMARY_REPRESENTATION_BY_ROLE.values())
    if set(entries) != expected:
        raise M7ConfigFinalizationError(
            "feature-independence matrix must contain exactly every available primary representation"
        )
    decisions = {
        "engineered_target_features": "not_independent",
        "imagenet_resnet18_context": "verified_independent",
        "imagenet_resnet18_highlighted": "verified_independent",
    }
    expected_families = {
        "engineered_target_features": "engineered",
        "imagenet_resnet18_context": "imagenet",
        "imagenet_resnet18_highlighted": "imagenet",
    }
    cache_role_by_representation = {
        "engineered_target_features": "engineered",
        "imagenet_resnet18_context": "context",
        "imagenet_resnet18_highlighted": "highlighted",
    }

    try:
        with np.load(caches["crop"].cache_path, allow_pickle=False) as crop_payload:
            sample_ids = np.asarray(crop_payload["sample_ids"], dtype=np.str_)
            group_ids = np.asarray(crop_payload["group_ids"], dtype=np.str_)
            labels = np.asarray(crop_payload["pre_corruption_labels"], dtype=np.int64)
            official_folds = np.asarray(crop_payload["official_folds"], dtype=np.int64)
    except (OSError, ValueError, KeyError) as exc:
        raise M7ConfigFinalizationError(
            f"cannot resolve the independence audit partition from the crop cache: {exc}"
        ) from exc
    n = len(sample_ids)
    if (
        n == 0
        or group_ids.shape != (n,)
        or labels.shape != (n,)
        or official_folds.shape != (n,)
        or len(np.unique(sample_ids)) != n
    ):
        raise M7ConfigFinalizationError(
            "crop cache identity vectors are malformed for independence verification"
        )
    development_mask = np.isin(official_folds, (1, 2))
    try:
        reference_groups = deterministic_group_greedy_class_distribution_v1(
            labels[development_mask],
            group_ids[development_mask],
            class_order=(0, 1, 2, 3, 4),
            fraction=0.10,
            seed=223,
        )
    except ValueError as exc:
        raise M7ConfigFinalizationError(
            f"cannot reproduce the frozen independence audit partition: {exc}"
        ) from exc
    audit_indices = np.flatnonzero(development_mask & ~np.isin(group_ids, reference_groups)).astype(
        np.int64, copy=False
    )
    if not len(audit_indices):
        raise M7ConfigFinalizationError("independence audit partition is empty")
    audit_fitted_data_hash = canonical_sha256(
        {
            "sample_ids": sample_ids[audit_indices].tolist(),
            "group_ids": group_ids[audit_indices].tolist(),
        }
    )

    feature_cache_by_representation = {
        "engineered_target_features": caches["engineered"],
        "imagenet_resnet18_context": caches["context"],
        "imagenet_resnet18_highlighted": caches["highlighted"],
    }
    audit_feature_hashes: dict[str, str] = {}
    morphology_audit_hash: str | None = None
    for representation_id, verification in feature_cache_by_representation.items():
        matrix_key = str(verification.metadata.get("matrix_key"))
        try:
            with np.load(verification.cache_path, allow_pickle=False) as payload:
                cache_sample_ids = np.asarray(payload["sample_ids"], dtype=np.str_)
                matrix = np.asarray(payload[matrix_key])
                names = (
                    np.asarray(payload["names"], dtype=np.str_)
                    if representation_id == "engineered_target_features"
                    else None
                )
        except (OSError, ValueError, KeyError) as exc:
            raise M7ConfigFinalizationError(
                f"cannot verify independence features for {representation_id}: {exc}"
            ) from exc
        if (
            not np.array_equal(cache_sample_ids, sample_ids)
            or matrix.ndim != 2
            or matrix.shape[0] != n
            or not np.issubdtype(matrix.dtype, np.floating)
            or not np.isfinite(matrix).all()
        ):
            raise M7ConfigFinalizationError(
                f"feature cache order/matrix is invalid for {representation_id}"
            )
        audit_feature_hashes[representation_id] = array_artifact_sha256(
            np.asarray(matrix[audit_indices])
        )
        if names is not None:
            morphology_columns = tuple(
                index for index, name in enumerate(names) if str(name).startswith("morphology.")
            )
            if not morphology_columns or names.shape != (matrix.shape[1],):
                raise M7ConfigFinalizationError(
                    "engineered cache cannot resolve morphology_only_v1 columns"
                )
            morphology_audit_hash = array_artifact_sha256(
                np.asarray(matrix[:, morphology_columns], dtype=np.float64)[audit_indices]
            )

    generator = None
    for representation_id, evidence in entries.items():
        if evidence.matrix_decision != decisions[representation_id]:
            raise M7ConfigFinalizationError(
                f"feature-independence decision is invalid for {representation_id}"
            )
        if evidence.generator.representation_name != "morphology_only_v1":
            raise M7ConfigFinalizationError(
                "feature-independence generator must be morphology_only_v1"
            )
        engineered_metadata = caches["engineered"].metadata
        if (
            evidence.generator.family != "morphology"
            or evidence.generator.implementation_hash
            != engineered_metadata.get("encoder_implementation_sha256")
            or evidence.generator.weights_hash != engineered_metadata.get("weights_sha256")
            or evidence.generator.preprocessing_hash
            != engineered_metadata.get("preprocessing_sha256")
        ):
            raise M7ConfigFinalizationError(
                "feature-independence generator semantics differ from the engineered cache"
            )
        if evidence.auditor.representation_name != representation_id:
            raise M7ConfigFinalizationError(
                f"feature-independence auditor identity differs for {representation_id}"
            )
        if evidence.auditor.family != expected_families[representation_id]:
            raise M7ConfigFinalizationError(
                f"feature-independence auditor family differs for {representation_id}"
            )
        if generator is None:
            generator = evidence.generator
        elif evidence.generator != generator:
            raise M7ConfigFinalizationError(
                "feature-independence entries do not share one generator/audit assignment"
            )
        metadata = caches[cache_role_by_representation[representation_id]].metadata
        if (
            evidence.generator.feature_artifact_hash != morphology_audit_hash
            or evidence.generator.fitted_data_hash != audit_fitted_data_hash
            or evidence.auditor.feature_artifact_hash != audit_feature_hashes[representation_id]
            or evidence.auditor.fitted_data_hash != audit_fitted_data_hash
            or evidence.auditor.implementation_hash != metadata.get("encoder_implementation_sha256")
            or evidence.auditor.weights_hash != metadata.get("weights_sha256")
            or evidence.auditor.preprocessing_hash != metadata.get("preprocessing_sha256")
        ):
            raise M7ConfigFinalizationError(
                f"feature-independence audit slice/provenance differs for {representation_id}"
            )
    return sha256_file(path)


def _relative_evidence_path(path: Path, project_root: Path, role: str) -> str:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(project_root)
    except ValueError as exc:
        raise M7ConfigFinalizationError(f"{role} must remain inside project_root") from exc
    return relative.as_posix()


def _pilot_report_bindings(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    report = _load_json_object(path, "pilot-derived parameter report")
    source = report.get("source_pilot")
    confusion = report.get("confusion_targeted_corruption")
    group = report.get("group_conditional_corruption")
    if (
        report.get("schema_version") != 1
        or report.get("producer_id") != "pilot_derived_primary_parameters_v1"
        or not isinstance(source, Mapping)
        or not isinstance(confusion, Mapping)
        or not isinstance(group, Mapping)
    ):
        raise M7ConfigFinalizationError(
            "pilot-derived parameter report schema/producer/sections are invalid"
        )
    run_id = source.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise M7ConfigFinalizationError("pilot-derived report source run_id is absent")
    root_sha = _require_sha256(
        source.get("artifact_root_sha256"),
        "pilot-derived source artifact root",
    )
    pilot_binding = {
        "schema_version": 1,
        "producer_id": "pilot_derived_primary_parameters_v1",
        "path": _PILOT_REPORT_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(path),
        "source_pilot_run_id": run_id,
        "source_pilot_artifact_root_sha256": root_sha,
    }
    confusion_parameters = {"transition_matrix": confusion.get("transition_matrix")}
    group_parameters = {
        field_name: group.get(field_name)
        for field_name in ("grouping_field", "weights_by_value", "default_weight")
    }
    return pilot_binding, confusion_parameters, group_parameters


def _pathology_blocker(
    path: Path,
    *,
    sample_order_sha256: str,
    manifest_sha256: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    try:
        payload = preregistration_workflow.validate_pathology_encoder_audit(path)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise M7ConfigFinalizationError(
            f"pathology availability audit fails the freeze semantic gate: {exc}"
        ) from exc
    if (
        payload.get("status") != "blocked"
        or payload.get("selected_encoder") is not None
        or not str(payload.get("blocker", "")).strip()
    ):
        raise M7ConfigFinalizationError(
            "pathology availability audit must explicitly record a blocked, unselected encoder"
        )
    audit_sha256 = sha256_file(path)
    expected_primary_record = unavailable_optional_pathology_cache_provenance(
        sample_order_sha256=sample_order_sha256,
        dataset_manifest_sha256=manifest_sha256,
    )
    if payload.get("primary_cache_provenance") != expected_primary_record:
        raise M7ConfigFinalizationError(
            "pathology audit primary_cache_provenance differs from the exact producer policy"
        )
    primary_record = dict(expected_primary_record)
    confirmatory_record = {
        "id": "pathology_context_embedding_cache",
        "representation_id": "pathology_context_embeddings",
        "status": "unavailable_with_frozen_blocker",
        "sample_order_sha256": sample_order_sha256,
        "manifest_sha256": manifest_sha256,
        "encoder_identifier": "availability_selected_pathology_encoder",
        "input_variant": "context_rgb",
        "blocker_evidence_sha256": audit_sha256,
    }
    return audit_sha256, primary_record, confirmatory_record


def _finalised_configs(
    primary_draft: Mapping[str, Any],
    confirmatory_draft: Mapping[str, Any],
    *,
    project_root: Path,
    caches: Mapping[str, FrozenCacheVerification],
    pilot_report_path: Path,
    pathology_audit_path: Path,
    independence_matrix_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary = deepcopy(resolve_config(primary_draft))
    confirmatory = deepcopy(resolve_config(confirmatory_draft))
    primary["status"] = "READY_FOR_FREEZE"
    confirmatory["status"] = "READY_FOR_FREEZE"
    if confirmatory.get("data", {}).get("analysis_manifest_authority") != primary.get(
        "data", {}
    ).get("analysis_manifest_authority"):
        raise M7ConfigFinalizationError(
            "primary/confirmatory data.analysis_manifest_authority bindings differ"
        )
    confirmatory["data"]["analysis_manifest_authority"] = deepcopy(
        primary["data"]["analysis_manifest_authority"]
    )

    expected_pilot_path = (project_root / _PILOT_REPORT_RELATIVE_PATH).resolve()
    if pilot_report_path.expanduser().resolve() != expected_pilot_path:
        raise M7ConfigFinalizationError(
            "pilot-derived report must use reports/pilot_derived_primary_parameters.json"
        )
    pilot_binding, confusion_parameters, group_parameters = _pilot_report_bindings(
        expected_pilot_path
    )
    primary["pilot_derived_parameters"] = pilot_binding
    primary["corruption"]["mechanisms"]["confusion_targeted_corruption"] = confusion_parameters
    primary["corruption"]["mechanisms"]["group_conditional_corruption"] = group_parameters

    sample_order_sha256 = str(caches["crop"].metadata["sample_order_sha256"])
    manifest_sha256 = str(caches["crop"].metadata["manifest_sha256"])
    audit_sha256, pathology_primary, pathology_confirmatory = _pathology_blocker(
        pathology_audit_path,
        sample_order_sha256=sample_order_sha256,
        manifest_sha256=manifest_sha256,
    )
    independence_sha256 = _validate_independence_matrix(independence_matrix_path, caches)
    independence_relative = _relative_evidence_path(
        independence_matrix_path,
        project_root,
        "feature-independence matrix",
    )
    instance_parameters = primary["corruption"]["mechanisms"]["instance_dependent_corruption"]
    instance_parameters["independence_matrix_path"] = independence_relative
    instance_parameters["independence_matrix_sha256"] = independence_sha256

    primary_provenance_by_role = {
        role: primary_cache_provenance_record(caches[role].metadata)
        for role in _PRIMARY_REPRESENTATION_BY_ROLE
    }
    for representation in primary["representations"]:
        identifier = str(representation["id"])
        matching_roles = [
            role
            for role, expected_identifier in _PRIMARY_REPRESENTATION_BY_ROLE.items()
            if expected_identifier == identifier
        ]
        if matching_roles:
            representation["cache_provenance"] = primary_provenance_by_role[matching_roles[0]]
            representation["generator_independence"]["independence_matrix_sha256"] = (
                independence_sha256
            )
        elif identifier == "pathology_encoder_optional":
            representation["availability_audit_sha256"] = audit_sha256
            representation["cache_provenance"] = pathology_primary
            representation["generator_independence"] = {
                "status": "unavailable_optional",
                "independence_matrix_sha256": independence_sha256,
            }
        else:
            raise M7ConfigFinalizationError(
                f"primary draft contains an unsupported representation: {identifier}"
            )

    weight_identifier = str(caches["context"].metadata["weight_identifier"])
    weights_sha256 = str(caches["context"].metadata["weights_sha256"])
    cnn_records = derive_confirmatory_cnn_logical_provenance(
        caches["crop"],
        weight_identifier=weight_identifier,
        weights_sha256=weights_sha256,
        input_size=int(confirmatory["training"]["input_size"]),
    )
    frozen_records = {
        _CONFIRMATORY_RECORD_BY_ROLE[role]: confirmatory_cache_provenance_record(
            caches[role].metadata,
            record_id=_CONFIRMATORY_RECORD_BY_ROLE[role],
            bind_sidecar_semantics=True,
        )
        for role in _CONFIRMATORY_RECORD_BY_ROLE
    }
    records_by_id = {**cnn_records, **frozen_records}
    records_by_id["pathology_context_embedding_cache"] = pathology_confirmatory
    expected_order = [str(record["id"]) for record in confirmatory["cache_provenance"]]
    if set(expected_order) != set(records_by_id):
        raise M7ConfigFinalizationError(
            "confirmatory draft cache records differ from the finalizer's fixed scenario set"
        )
    confirmatory["cache_provenance"] = [records_by_id[value] for value in expected_order]
    for scenario in confirmatory["scenarios"]:
        if scenario["id"] == "pathology_frozen_logistic":
            scenario["availability_audit_sha256"] = audit_sha256

    confirmatory["corruption"]["cells"] = [
        {
            "id": "clean_reference_cell",
            "mechanism": "symmetric_random_corruption",
            "rate": 0.0,
            "seed": 404,
            "parameters": {},
        },
        {
            "id": "confusion_targeted_ten_percent",
            "mechanism": "confusion_targeted_corruption",
            "rate": 0.10,
            "seed": 404,
            "parameters": confusion_parameters,
        },
    ]
    return primary, confirmatory


def _stable_yaml(config: Mapping[str, Any]) -> str:
    return yaml.safe_dump(
        resolve_config(config),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    )


def _stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat(follow_symlinks=False)
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _ownership_identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat(follow_symlinks=False)
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    path: Path
    identity: tuple[int, int, int, int, int]
    sha256: str

    @classmethod
    def capture(cls, path: Path, role: str) -> _FileSnapshot:
        source = path.expanduser().resolve()
        if not source.is_file():
            raise M7ConfigFinalizationError(f"{role} is missing: {source}")
        return cls(source, _stat_identity(source), sha256_file(source))

    def assert_current(self) -> None:
        try:
            current = _stat_identity(self.path)
        except OSError as exc:
            raise M7ConfigFinalizationError(
                f"M7 source disappeared during finalisation: {self.path}"
            ) from exc
        if current != self.identity or sha256_file(self.path) != self.sha256:
            raise M7ConfigFinalizationError(f"M7 source changed during finalisation: {self.path}")


def _cache_fingerprint(
    values: Mapping[str, FrozenCacheVerification],
) -> dict[str, tuple[str, str, str, str]]:
    return {
        role: (
            verification.cache_file_sha256,
            verification.sidecar_file_sha256,
            verification.sidecar_semantic_sha256,
            canonical_sha256(verification.metadata),
        )
        for role, verification in values.items()
    }


def _source_snapshots(
    cache_paths: M7CachePaths,
    evidence_paths: Sequence[Path],
) -> tuple[_FileSnapshot, ...]:
    snapshots: list[_FileSnapshot] = []
    for role in _CACHE_REPRESENTATION_BY_ROLE:
        cache = Path(getattr(cache_paths, role)).expanduser().resolve()
        sidecar = cache.with_suffix(f"{cache.suffix}.metadata.json")
        snapshots.append(_FileSnapshot.capture(cache, f"{role} cache"))
        snapshots.append(_FileSnapshot.capture(sidecar, f"{role} cache sidecar"))
    for index, path in enumerate(evidence_paths):
        snapshots.append(_FileSnapshot.capture(path, f"M7 evidence[{index}]"))
    return tuple(snapshots)


def _assert_sources_fresh(
    cache_paths: M7CachePaths,
    expected_cache_fingerprint: Mapping[str, tuple[str, str, str, str]],
    snapshots: Sequence[_FileSnapshot],
    canonical_pilot: _CanonicalPilotRun,
) -> None:
    _assert_canonical_pilot_run_current(canonical_pilot)
    for snapshot in snapshots:
        snapshot.assert_current()
    fresh_caches = _verified_caches(cache_paths)
    if _cache_fingerprint(fresh_caches) != dict(expected_cache_fingerprint):
        raise M7ConfigFinalizationError("M7 cache provenance changed during config finalisation")
    for snapshot in snapshots:
        snapshot.assert_current()
    _assert_canonical_pilot_run_current(canonical_pilot)


@dataclass(frozen=True, slots=True)
class _OutputOriginal:
    path: Path
    content: bytes | None
    sha256: str | None


@dataclass(frozen=True, slots=True)
class _OwnedOutput:
    path: Path
    identity: tuple[int, int, int, int]
    sha256: str

    @classmethod
    def from_staged(cls, path: Path, staged_path: Path) -> _OwnedOutput:
        return cls(path, _ownership_identity(staged_path), sha256_file(staged_path))

    def still_owned(self) -> bool:
        try:
            return (
                _ownership_identity(self.path) == self.identity
                and sha256_file(self.path) == self.sha256
            )
        except OSError:
            return False


def _capture_output_originals(
    outputs: Sequence[tuple[Path, str | None]],
    *,
    replace_draft: bool,
) -> tuple[_OutputOriginal, ...]:
    originals: list[_OutputOriginal] = []
    for raw_path, expected_sha256 in outputs:
        path = raw_path.expanduser().resolve()
        if replace_draft:
            if expected_sha256 is None:
                raise M7ConfigFinalizationError(
                    "--replace-draft requires an exact precondition SHA-256 for both outputs"
                )
            expected = _require_sha256(expected_sha256, f"{path} precondition")
            if not path.is_file():
                raise M7ConfigFinalizationError(
                    f"replace-draft precondition target is missing: {path}"
                )
            identity_before = _stat_identity(path)
            content = path.read_bytes()
            identity_after = _stat_identity(path)
            if identity_before != identity_after or hashlib.sha256(content).hexdigest() != expected:
                raise M7ConfigFinalizationError(
                    f"replace-draft precondition changed or differs for {path}"
                )
            originals.append(_OutputOriginal(path, content, expected))
        elif expected_sha256 is not None:
            raise M7ConfigFinalizationError("output precondition hashes require replace_draft=True")
        elif os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite finalised config: {path}")
        else:
            originals.append(_OutputOriginal(path, None, None))
    return tuple(originals)


def _stage_output(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _commit_staged_output(
    temporary: Path,
    original: _OutputOriginal,
    *,
    replace_draft: bool,
) -> _OwnedOutput:
    destination = original.path
    publication = _OwnedOutput.from_staged(destination, temporary)
    try:
        if replace_draft:
            if original.sha256 is None or sha256_file(destination) != original.sha256:
                raise M7ConfigFinalizationError(
                    f"replace-draft precondition changed before publication: {destination}"
                )
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    except BaseException as commit_error:
        if publication.still_owned():
            _rollback_outputs((publication,), (original,))
        elif replace_draft:
            original_still_present = (
                original.sha256 is not None
                and destination.is_file()
                and sha256_file(destination) == original.sha256
            )
            if not original_still_present:
                raise RuntimeError(
                    "M7 config commit failed after destination ownership became uncertain: "
                    f"{destination}"
                ) from commit_error
        elif isinstance(commit_error, FileExistsError):
            raise FileExistsError(
                f"refusing to overwrite finalised config: {destination}"
            ) from commit_error
        elif os.path.lexists(destination):
            raise RuntimeError(
                f"M7 config no-overwrite commit failed with an unowned destination: {destination}"
            ) from commit_error
        raise
    return publication


def _rollback_outputs(
    publications: Sequence[_OwnedOutput],
    originals: Sequence[_OutputOriginal],
) -> None:
    original_by_path = {item.path: item for item in originals}
    for publication in reversed(publications):
        if not publication.still_owned():
            raise RuntimeError(
                f"cannot rollback M7 config pair after output ownership changed: {publication.path}"
            )
        original = original_by_path[publication.path]
        if original.content is None:
            publication.path.unlink()
        else:
            atomic_write_bytes(publication.path, original.content)
            if sha256_file(publication.path) != original.sha256:
                raise RuntimeError(f"M7 config rollback readback failed: {publication.path}")


def _publish_config_pair(
    originals: Sequence[_OutputOriginal],
    contents: Sequence[str],
    *,
    replace_draft: bool,
    source_freshness_check: Callable[[], None],
    readback_check: Callable[[], None],
) -> None:
    if len(originals) != 2 or len(contents) != 2:
        raise AssertionError("M7 publication requires exactly two configs")
    staged: list[Path] = []
    try:
        for original, content in zip(originals, contents, strict=True):
            staged.append(_stage_output(original.path, content))
    except BaseException:
        try:
            for temporary in staged:
                temporary.unlink(missing_ok=True)
        except OSError as cleanup_error:
            raise RuntimeError(
                "M7 config staging failed and partial staging cleanup also failed"
            ) from cleanup_error
        raise
    publications: list[_OwnedOutput] = []
    try:
        source_freshness_check()
        for temporary, original in zip(staged, originals, strict=True):
            publication = _commit_staged_output(
                temporary,
                original,
                replace_draft=replace_draft,
            )
            publications.append(publication)
            temporary.unlink(missing_ok=True)
        source_freshness_check()
        readback_check()
        source_freshness_check()
    except BaseException:
        try:
            _rollback_outputs(publications, originals)
        except BaseException as rollback_error:
            raise RuntimeError(
                "M7 pair publication failed and ownership-safe rollback also failed"
            ) from rollback_error
        raise
    finally:
        for temporary in staged:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def finalise_m7_configs(
    *,
    project_root: str | Path,
    primary_draft_path: str | Path,
    confirmatory_draft_path: str | Path,
    cache_paths: M7CachePaths,
    pathology_audit_path: str | Path,
    independence_matrix_path: str | Path,
    pilot_report_path: str | Path,
    primary_output_path: str | Path,
    confirmatory_output_path: str | Path,
    replace_draft: bool = False,
    expected_primary_output_sha256: str | None = None,
    expected_confirmatory_output_sha256: str | None = None,
) -> M7ConfigFinalizationResult:
    """Build, cross-check, and atomically publish both final M7 configurations."""

    root = Path(project_root).expanduser().resolve()
    primary_output = Path(primary_output_path).expanduser().resolve()
    confirmatory_output = Path(confirmatory_output_path).expanduser().resolve()
    if primary_output == confirmatory_output:
        raise M7ConfigFinalizationError("primary and confirmatory output paths must differ")
    try:
        primary_draft = load_config(primary_draft_path)
        confirmatory_draft = load_config(confirmatory_draft_path)
    except (FileNotFoundError, OSError, ValueError, yaml.YAMLError) as exc:
        raise M7ConfigFinalizationError(f"cannot load M7 draft configs: {exc}") from exc

    pilot_evidence = Path(pilot_report_path).expanduser().resolve()
    pathology_evidence = Path(pathology_audit_path).expanduser().resolve()
    independence_evidence = Path(independence_matrix_path).expanduser().resolve()
    canonical_pilot = _canonical_pilot_run_from_draft(primary_draft, root)
    caches = _verified_caches(cache_paths)
    _validate_analysis_manifest_authority(primary_draft, caches)
    expected_cache_fingerprint = _cache_fingerprint(caches)
    source_code_paths = (
        Path(__file__).resolve(),
        Path(cnn_module.__file__).resolve(),
        Path(pilot_parameters_module.__file__).resolve(),
        Path(pathology_module.__file__).resolve(),
        Path(preregistration_workflow.__file__).resolve(),
    )
    source_snapshots = _source_snapshots(
        cache_paths,
        (
            pilot_evidence,
            pathology_evidence,
            independence_evidence,
            *source_code_paths,
            *canonical_pilot.source_artifact_paths,
            canonical_pilot.path / "artifact_manifest.json",
            canonical_pilot.path / _FINAL_REFERENCE_PRIVACY_ARTIFACT,
        ),
    )
    primary, confirmatory = _finalised_configs(
        primary_draft,
        confirmatory_draft,
        project_root=root,
        caches=caches,
        pilot_report_path=pilot_evidence,
        pathology_audit_path=pathology_evidence,
        independence_matrix_path=independence_evidence,
    )
    primary, confirmatory = validate_primary_confirmatory_cross_config(primary, confirmatory)
    _validate_pilot_source_artifacts(pilot_evidence, canonical_pilot)
    try:
        preregistration_workflow.validate_pilot_derived_parameters(
            primary,
            project_root=root,
            pilot_evidence={
                "run_id": canonical_pilot.run_id,
                "artifact_root_sha256": canonical_pilot.artifact_root_sha256,
            },
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise M7ConfigFinalizationError(
            f"pilot-derived parameters fail the freeze semantic gate: {exc}"
        ) from exc
    primary_inputs.verify_pilot_derived_parameters_binding(primary, root)
    primary_plan = build_primary_matrix_plan(primary)
    confirmatory_plan = build_confirmatory_matrix_plan(confirmatory)
    primary_yaml = _stable_yaml(primary)
    confirmatory_yaml = _stable_yaml(confirmatory)
    outputs = (
        (primary_output, expected_primary_output_sha256),
        (confirmatory_output, expected_confirmatory_output_sha256),
    )
    originals = _capture_output_originals(outputs, replace_draft=replace_draft)

    def source_freshness_check() -> None:
        _assert_sources_fresh(
            cache_paths,
            expected_cache_fingerprint,
            source_snapshots,
            canonical_pilot,
        )

    def readback_check() -> None:
        published_primary = load_config(primary_output)
        published_confirmatory = load_config(confirmatory_output)
        validate_primary_confirmatory_cross_config(
            published_primary,
            published_confirmatory,
        )
        if config_sha256(published_primary) != config_sha256(primary) or config_sha256(
            published_confirmatory
        ) != config_sha256(confirmatory):
            raise RuntimeError("published M7 configs differ from their validated staging values")

    _publish_config_pair(
        originals,
        (primary_yaml, confirmatory_yaml),
        replace_draft=replace_draft,
        source_freshness_check=source_freshness_check,
        readback_check=readback_check,
    )
    return M7ConfigFinalizationResult(
        primary_output_path=primary_output,
        confirmatory_output_path=confirmatory_output,
        primary_file_sha256=sha256_file(primary_output),
        confirmatory_file_sha256=sha256_file(confirmatory_output),
        primary_config_sha256=config_sha256(primary),
        confirmatory_config_sha256=config_sha256(confirmatory),
        primary_plan=primary_plan,
        confirmatory_plan=confirmatory_plan,
        cache_file_sha256_by_role={
            role: verification.cache_file_sha256 for role, verification in caches.items()
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--primary-draft", required=True)
    parser.add_argument("--confirmatory-draft", required=True)
    parser.add_argument("--crop-cache", required=True)
    parser.add_argument("--engineered-cache", required=True)
    parser.add_argument("--highlighted-cache", required=True)
    parser.add_argument("--context-cache", required=True)
    parser.add_argument("--context-morphometrics-cache", required=True)
    parser.add_argument("--pathology-audit", required=True)
    parser.add_argument("--independence-matrix", required=True)
    parser.add_argument("--pilot-report", required=True)
    parser.add_argument("--primary-output", required=True)
    parser.add_argument("--confirmatory-output", required=True)
    parser.add_argument("--replace-draft", action="store_true")
    parser.add_argument("--expected-primary-output-sha256")
    parser.add_argument("--expected-confirmatory-output-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone deterministic M7 config finaliser."""

    arguments = _parser().parse_args(argv)
    result = finalise_m7_configs(
        project_root=arguments.project_root,
        primary_draft_path=arguments.primary_draft,
        confirmatory_draft_path=arguments.confirmatory_draft,
        cache_paths=M7CachePaths(
            crop=Path(arguments.crop_cache),
            engineered=Path(arguments.engineered_cache),
            highlighted=Path(arguments.highlighted_cache),
            context=Path(arguments.context_cache),
            context_morphometrics=Path(arguments.context_morphometrics_cache),
        ),
        pathology_audit_path=arguments.pathology_audit,
        independence_matrix_path=arguments.independence_matrix,
        pilot_report_path=arguments.pilot_report,
        primary_output_path=arguments.primary_output,
        confirmatory_output_path=arguments.confirmatory_output,
        replace_draft=arguments.replace_draft,
        expected_primary_output_sha256=arguments.expected_primary_output_sha256,
        expected_confirmatory_output_sha256=arguments.expected_confirmatory_output_sha256,
    )
    print(json.dumps(result.as_dict(), allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
