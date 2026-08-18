"""Fail-closed, primary-outcome-blind input adapter for the production confirmatory CLI.

The public CLI deliberately exposes no cache-selection knobs.  Crop identity and
raw-inventory identity come from the sealed primary run's input bindings, while
the frozen-feature cache selection is recomputed from the verified confirmatory
configuration and cache sidecars.  Primary scientific outcome artifacts are never read.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from histo_audit.config import config_sha256, load_config
from histo_audit.corruption.controlled import apply_controlled_corruption
from histo_audit.experiment.confirmatory_core import (
    confirmatory_execution_controls_from_frozen_config,
)
from histo_audit.experiment.pannuke_confirmatory_inputs import (
    ConfirmatoryFrozenFeatureCacheSpec,
    ConfirmatoryObservedLabelSet,
)
from histo_audit.experiment.reference_groups import (
    deterministic_group_greedy_class_distribution_v1,
)
from histo_audit.experiment.study_contracts import (
    CLASS_ORDER,
    validate_frozen_confirmatory_config,
    validate_resource_bounded_confirmatory_config,
)
from histo_audit.representations.cache_provenance import (
    FrozenCacheVerification,
    confirmatory_cache_provenance_record,
    verify_frozen_cache_sidecar,
)
from histo_audit.workflows.study_gates import (
    ConfirmatoryExecutionGateEvidence,
    ResourceBoundedExecutionGateEvidence,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PRIMARY_INPUT_BINDINGS_FILENAME = "primary_input_bindings.json"
_CONTEXT_MORPHOMETRICS_CACHE_FILENAME = "pannuke_resnet18_context_plus_target_morphometrics.npz"
_FROZEN_FEATURE_FAMILIES = frozenset({"imagenet_frozen", "pathology_frozen"})
_BOUND_CACHE_PATH_TO_HASH = {
    "crop_cache_path": "crop_cache_sha256",
    "engineered_cache_path": "engineered_cache_sha256",
    "context_embedding_cache_path": "context_embedding_cache_sha256",
    "highlighted_embedding_cache_path": "highlighted_embedding_cache_sha256",
    "pathology_embedding_cache_path": "pathology_embedding_cache_sha256",
}


class ConfirmatoryCLIInputError(ValueError):
    """The verified authority cannot produce one exact production input set."""


@dataclass(frozen=True, slots=True)
class ConfirmatoryCLIInputs:
    """Exact executor arguments derived after lifecycle and scientific gating."""

    crop_cache_path: Path
    expected_crop_cache_sha256: str
    expected_crop_metadata_sha256: str
    expected_raw_inventory_sha256: str
    frozen_feature_caches: tuple[ConfirmatoryFrozenFeatureCacheSpec, ...]
    observed_label_sets: tuple[ConfirmatoryObservedLabelSet, ...]
    primary_input_bindings_sha256: str
    confirmatory_config_sha256: str
    confirmatory_config_semantic_sha256: str

    def executor_kwargs(self) -> dict[str, Any]:
        """Return only the additional arguments accepted by the real runner."""

        return {
            "crop_cache_path": self.crop_cache_path,
            "expected_crop_cache_sha256": self.expected_crop_cache_sha256,
            "expected_crop_metadata_sha256": self.expected_crop_metadata_sha256,
            "expected_raw_inventory_sha256": self.expected_raw_inventory_sha256,
            "frozen_feature_caches": self.frozen_feature_caches,
            "observed_label_sets": self.observed_label_sets,
        }


@dataclass(frozen=True, slots=True)
class _CropIdentity:
    sample_ids: tuple[str, ...]
    group_ids: NDArray[np.str_]
    official_folds: NDArray[np.int64]
    pre_corruption_labels: NDArray[np.int64]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfirmatoryCLIInputError(f"{role} must be a mapping")
    return value


def _sequence(value: object, role: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ConfirmatoryCLIInputError(f"{role} must be a sequence")
    return value


def _sha(value: object, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ConfirmatoryCLIInputError(f"{role} must be a lowercase SHA-256 digest")
    return value


def _read_json_object(path: Path, role: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ConfirmatoryCLIInputError(f"{role} is unavailable or invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ConfirmatoryCLIInputError(f"{role} must be a JSON object: {path}")
    return payload


def _is_link_or_reparse(path: Path, value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    return path.is_symlink() or bool(
        attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    )


def _regular_bound_file(value: object, role: str) -> Path:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConfirmatoryCLIInputError(f"{role} path must be an explicit string")
    lexical = Path(value).expanduser()
    if not lexical.is_absolute():
        raise ConfirmatoryCLIInputError(f"{role} path must be absolute")
    try:
        observed = lexical.stat(follow_symlinks=False)
    except OSError as exc:
        raise ConfirmatoryCLIInputError(f"{role} is unavailable: {lexical}") from exc
    if not stat.S_ISREG(observed.st_mode) or _is_link_or_reparse(lexical, observed):
        raise ConfirmatoryCLIInputError(f"{role} must be a regular non-link file: {lexical}")
    return lexical.resolve()


def _optional_bound_file(value: object, role: str) -> Path | None:
    return None if value is None else _regular_bound_file(value, role)


def _verified_primary_cache_paths(
    bindings: Mapping[str, Any],
) -> tuple[dict[str, Path | None], Mapping[str, Any], Mapping[str, Any]]:
    paths = _mapping(bindings.get("cache_paths"), "primary cache paths")
    expected = _mapping(bindings.get("expected_hashes"), "primary expected cache hashes")
    verified = _mapping(bindings.get("verified_hashes"), "primary verified cache hashes")
    resolved: dict[str, Path | None] = {}
    for path_field, hash_field in _BOUND_CACHE_PATH_TO_HASH.items():
        path = _optional_bound_file(paths.get(path_field), f"primary {path_field}")
        expected_sha = expected.get(hash_field)
        verified_sha = verified.get(hash_field)
        if path is None:
            if expected_sha is not None or verified_sha is not None:
                raise ConfirmatoryCLIInputError(
                    f"primary {path_field}/{hash_field} optional path/hash binding is incomplete"
                )
        else:
            actual = _sha256_file(path)
            if (
                _sha(expected_sha, f"primary expected {hash_field}") != actual
                or _sha(verified_sha, f"primary verified {hash_field}") != actual
            ):
                raise ConfirmatoryCLIInputError(
                    f"primary {path_field} differs from expected/verified hashes"
                )
        resolved[path_field] = path
    return resolved, expected, verified


def _candidate_cache_paths(bound_paths: Mapping[str, Path | None]) -> tuple[Path, ...]:
    context = bound_paths.get("context_embedding_cache_path")
    highlighted = bound_paths.get("highlighted_embedding_cache_path")
    if context is None or highlighted is None:
        raise ConfirmatoryCLIInputError(
            "primary input bindings lack required confirmatory embedding caches"
        )
    morphometrics = _regular_bound_file(
        str(context.parent / _CONTEXT_MORPHOMETRICS_CACHE_FILENAME),
        "canonical context-plus-morphometrics cache",
    )
    candidates = {context, highlighted, morphometrics}
    pathology = bound_paths.get("pathology_embedding_cache_path")
    if pathology is not None:
        candidates.add(pathology)
    return tuple(sorted(candidates, key=str))


def _verify_candidate_caches(
    candidates: Sequence[Path],
    *,
    manifest_sha256: str,
) -> tuple[FrozenCacheVerification, ...]:
    verified: list[FrozenCacheVerification] = []
    for path in candidates:
        try:
            item = verify_frozen_cache_sidecar(
                path,
                expected_manifest_sha256=manifest_sha256,
            )
        except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
            raise ConfirmatoryCLIInputError(
                f"candidate cache in the primary-bound cache directory failed verification: {path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if item.cache_path != path:
            raise ConfirmatoryCLIInputError("cache verifier resolved a different candidate path")
        verified.append(item)
    return tuple(verified)


def _enforce_pathology_authority(
    config: Mapping[str, Any],
    *,
    bound_paths: Mapping[str, Path | None],
    expected_hashes: Mapping[str, Any],
    verified_hashes: Mapping[str, Any],
) -> None:
    raw_records = _sequence(config.get("cache_provenance"), "confirmatory cache provenance")
    records = {
        str(_mapping(value, "confirmatory cache provenance record").get("id")): _mapping(
            value, "confirmatory cache provenance record"
        )
        for value in raw_records
    }
    for raw in _sequence(config.get("scenarios"), "confirmatory scenarios"):
        scenario = _mapping(raw, "confirmatory scenario")
        if scenario.get("family") != "pathology_frozen":
            continue
        scenario_id = str(scenario.get("id", ""))
        record_id = str(scenario.get("cache_provenance_id", ""))
        record = records.get(record_id)
        if record is None:
            raise ConfirmatoryCLIInputError(
                f"pathology scenario {scenario_id!r} lacks frozen cache authority"
            )
        status = record.get("status")
        path = bound_paths.get("pathology_embedding_cache_path")
        expected_sha = expected_hashes.get("pathology_embedding_cache_sha256")
        verified_sha = verified_hashes.get("pathology_embedding_cache_sha256")
        if status == "unavailable_with_frozen_blocker":
            if scenario.get("required") is not False:
                raise ConfirmatoryCLIInputError(
                    f"required pathology scenario is frozen as unavailable: {scenario_id}"
                )
            if path is not None or expected_sha is not None or verified_sha is not None:
                raise ConfirmatoryCLIInputError(
                    "primary bindings supply pathology data despite the frozen unavailable blocker"
                )
            blocker_sha = record.get("blocker_evidence_sha256")
            if blocker_sha != scenario.get("availability_audit_sha256"):
                raise ConfirmatoryCLIInputError(
                    "pathology scenario and frozen blocker evidence differ"
                )
        elif status == "available":
            if path is None:
                raise ConfirmatoryCLIInputError(
                    f"available pathology scenario lacks a primary-bound cache: {scenario_id}"
                )
        else:
            raise ConfirmatoryCLIInputError(
                f"pathology scenario has unsupported frozen status: {scenario_id}: {status!r}"
            )


def _readonly(values: NDArray[Any], dtype: Any) -> NDArray[Any]:
    output = np.asarray(values, dtype=dtype).copy()
    output.setflags(write=False)
    return output


def _load_crop_identity(
    crop_path: Path,
    *,
    official_folds: tuple[int, ...],
) -> _CropIdentity:
    required = {"sample_ids", "group_ids", "official_folds", "pre_corruption_labels"}
    try:
        with np.load(crop_path, allow_pickle=False) as payload:
            if missing := sorted(required.difference(payload.files)):
                raise ConfirmatoryCLIInputError(
                    f"crop cache lacks confirmatory identity arrays: {missing}"
                )
            sample_raw = payload["sample_ids"]
            group_raw = payload["group_ids"]
            fold_raw = payload["official_folds"]
            label_raw = payload["pre_corruption_labels"]
    except ConfirmatoryCLIInputError:
        raise
    except (OSError, KeyError, ValueError) as exc:
        raise ConfirmatoryCLIInputError(
            "crop identity arrays cannot be loaded without pickle"
        ) from exc
    if sample_raw.dtype.kind not in {"U", "S"} or group_raw.dtype.kind not in {"U", "S"}:
        raise ConfirmatoryCLIInputError("crop identity strings must not require pickle")
    sample_values = tuple(str(value) for value in np.asarray(sample_raw, dtype=np.str_))
    groups = np.asarray(group_raw, dtype=np.str_)
    folds = np.asarray(fold_raw)
    labels = np.asarray(label_raw)
    n = len(sample_values)
    if not n or len(set(sample_values)) != n or any(not value for value in sample_values):
        raise ConfirmatoryCLIInputError("crop sample IDs must be non-empty and unique")
    if groups.shape != (n,) or any(not str(value) for value in groups):
        raise ConfirmatoryCLIInputError("crop group IDs must be aligned and non-empty")
    if folds.shape != (n,) or not np.issubdtype(folds.dtype, np.integer):
        raise ConfirmatoryCLIInputError("crop official folds must be an aligned integer vector")
    if labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer):
        raise ConfirmatoryCLIInputError(
            "crop pre-corruption labels must be an aligned integer vector"
        )
    typed_folds = np.asarray(folds, dtype=np.int64)
    typed_labels = np.asarray(labels, dtype=np.int64)
    if set(int(value) for value in typed_folds) != set(official_folds):
        raise ConfirmatoryCLIInputError("crop official folds differ from frozen controls")
    if set(int(value) for value in typed_labels) != set(CLASS_ORDER):
        raise ConfirmatoryCLIInputError("crop labels differ from the frozen class order")
    for group in np.unique(groups):
        if len(np.unique(typed_folds[groups == group])) != 1:
            raise ConfirmatoryCLIInputError(f"crop group spans official folds: {group}")
    return _CropIdentity(
        sample_ids=sample_values,
        group_ids=_readonly(groups, np.str_),
        official_folds=_readonly(typed_folds, np.int64),
        pre_corruption_labels=_readonly(typed_labels, np.int64),
    )


def _materialize_observed_label_sets(
    crop_path: Path,
    *,
    config: Mapping[str, Any],
    config_semantic_sha256: str,
) -> tuple[ConfirmatoryObservedLabelSet, ...]:
    try:
        controls = confirmatory_execution_controls_from_frozen_config(config)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfirmatoryCLIInputError(
            "confirmatory execution controls cannot be derived from frozen config"
        ) from exc
    if controls.config_semantic_sha256 != config_semantic_sha256:
        raise ConfirmatoryCLIInputError("confirmatory control/config semantic hashes differ")
    nonzero = tuple(value for value in controls.corruption_specs if value.rate > 0.0)
    if len(nonzero) != 1:
        raise ConfirmatoryCLIInputError(
            "production label materialization requires exactly one nonzero frozen corruption cell"
        )
    corruption = nonzero[0]
    if corruption.mechanism == "instance_dependent_corruption":
        raise ConfirmatoryCLIInputError(
            "instance-dependent confirmatory corruption requires a separately frozen, "
            "independence-bound materializer"
        )
    if corruption.mechanism not in {
        "symmetric_random_corruption",
        "confusion_targeted_corruption",
        "group_conditional_corruption",
    }:
        raise ConfirmatoryCLIInputError(
            f"unsupported frozen confirmatory corruption: {corruption.mechanism}"
        )
    data = _mapping(config.get("data"), "confirmatory data controls")
    if data.get("reference_group_selection_algorithm") != (
        "deterministic_group_greedy_class_distribution_v1"
    ):
        raise ConfirmatoryCLIInputError("unsupported frozen reference-group selector")
    fraction = float(data["reference_validation_fraction_groups"])
    identity = _load_crop_identity(crop_path, official_folds=controls.official_folds)
    all_indices = np.arange(len(identity.sample_ids), dtype=np.int64)
    output: list[ConfirmatoryObservedLabelSet] = []
    for outer_fold in controls.official_folds:
        final_mask = identity.official_folds == outer_fold
        development_mask = ~final_mask
        reference_groups = deterministic_group_greedy_class_distribution_v1(
            identity.pre_corruption_labels[development_mask],
            identity.group_ids[development_mask],
            class_order=CLASS_ORDER,
            fraction=fraction,
            seed=controls.split_seed,
        )
        reference_mask = development_mask & np.isin(identity.group_ids, reference_groups)
        audit_mask = development_mask & ~reference_mask
        audit_indices = all_indices[audit_mask]
        replay_kwargs: dict[str, Any] = {}
        if corruption.mechanism == "confusion_targeted_corruption":
            replay_kwargs["transition_matrix"] = np.asarray(
                corruption.parameters["transition_matrix"], dtype=np.float64
            )
        elif corruption.mechanism == "group_conditional_corruption":
            grouping_field = str(corruption.parameters["grouping_field"])
            if grouping_field != controls.statistical_group_unit:
                raise ConfirmatoryCLIInputError(
                    "frozen group-conditional field differs from the statistical group unit"
                )
            raw_weights = _mapping(
                corruption.parameters["weights_by_value"],
                "group-conditional corruption weights",
            )
            default_weight = float(corruption.parameters["default_weight"])
            replay_kwargs["group_weights"] = {
                str(group): float(raw_weights.get(str(group), default_weight))
                for group in set(identity.group_ids[audit_mask].tolist())
            }
        try:
            replay = apply_controlled_corruption(
                identity.pre_corruption_labels[audit_mask],
                sample_ids=tuple(identity.sample_ids[index] for index in audit_indices),
                group_ids=tuple(str(value) for value in identity.group_ids[audit_mask]),
                rate=corruption.rate,
                mechanism=corruption.mechanism,
                seed=corruption.seed,
                n_classes=len(CLASS_ORDER),
                **replay_kwargs,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfirmatoryCLIInputError(
                f"frozen corruption cannot be replayed for outer fold {outer_fold}"
            ) from exc
        observed = np.asarray(identity.pre_corruption_labels, dtype=np.int64).copy()
        injected = np.zeros(len(identity.sample_ids), dtype=bool)
        observed[audit_mask] = replay.observed_labels
        injected[audit_mask] = replay.is_injected_corruption
        corruption_types = ["none"] * len(identity.sample_ids)
        for index, is_injected in zip(
            audit_indices.tolist(), replay.is_injected_corruption.tolist(), strict=True
        ):
            if is_injected:
                corruption_types[int(index)] = corruption.mechanism
        if np.any(injected[reference_mask]) or np.any(injected[final_mask]):
            raise RuntimeError("confirmatory materializer corrupted a protected partition")
        output.append(
            ConfirmatoryObservedLabelSet(
                outer_fold=outer_fold,
                sample_ids=identity.sample_ids,
                pre_corruption_labels=_readonly(identity.pre_corruption_labels, np.int64),
                observed_labels=_readonly(observed, np.int64),
                is_injected_corruption=_readonly(injected, np.bool_),
                corruption_types=tuple(corruption_types),
                configuration_sha256=config_semantic_sha256,
            )
        )
    return tuple(output)


def _frozen_feature_specs(
    config: Mapping[str, Any],
    candidates: Sequence[FrozenCacheVerification],
) -> tuple[ConfirmatoryFrozenFeatureCacheSpec, ...]:
    raw_records = _sequence(config.get("cache_provenance"), "confirmatory cache provenance")
    records: dict[str, Mapping[str, Any]] = {}
    for raw in raw_records:
        record = _mapping(raw, "confirmatory cache provenance record")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id or record_id in records:
            raise ConfirmatoryCLIInputError("confirmatory cache provenance IDs are invalid")
        records[record_id] = record

    raw_scenarios = _sequence(config.get("scenarios"), "confirmatory scenarios")
    specs: list[ConfirmatoryFrozenFeatureCacheSpec] = []
    selected_paths: dict[Path, str] = {}
    for raw in raw_scenarios:
        scenario = _mapping(raw, "confirmatory scenario")
        family = str(scenario.get("family", ""))
        if family not in _FROZEN_FEATURE_FAMILIES:
            continue
        scenario_id = str(scenario.get("id", ""))
        record_id = str(scenario.get("cache_provenance_id", ""))
        selected_record = records.get(record_id)
        if not scenario_id or selected_record is None:
            raise ConfirmatoryCLIInputError(
                f"frozen scenario {scenario_id!r} lacks one cache provenance record"
            )
        status = selected_record.get("status")
        if status == "unavailable_with_frozen_blocker":
            if scenario.get("required") is not False:
                raise ConfirmatoryCLIInputError(
                    f"required frozen scenario is unavailable: {scenario_id}"
                )
            continue
        if status != "available":
            raise ConfirmatoryCLIInputError(
                f"frozen scenario has unsupported cache status: {scenario_id}: {status!r}"
            )

        direct_sha = selected_record.get("cache_file_sha256")
        semantic_sha = selected_record.get("sidecar_semantic_sha256")
        if (direct_sha is None) == (semantic_sha is None):
            raise ConfirmatoryCLIInputError(
                f"frozen scenario does not have exactly one cache authority: {scenario_id}"
            )
        bind_semantics = semantic_sha is not None
        matches: list[FrozenCacheVerification] = []
        for candidate in candidates:
            try:
                projected = confirmatory_cache_provenance_record(
                    candidate.metadata,
                    record_id=record_id,
                    bind_sidecar_semantics=bind_semantics,
                )
            except (KeyError, TypeError, ValueError):
                continue
            if projected == dict(selected_record):
                matches.append(candidate)
        if len(matches) != 1:
            raise ConfirmatoryCLIInputError(
                "frozen scenario must resolve to exactly one verified physical cache: "
                f"{scenario_id}; matches={len(matches)}"
            )
        selected = matches[0]
        previous = selected_paths.get(selected.cache_path)
        if previous is not None and previous != record_id:
            raise ConfirmatoryCLIInputError(
                "one physical cache matched multiple distinct frozen records: "
                f"{previous}, {record_id}"
            )
        selected_paths[selected.cache_path] = record_id
        specs.append(
            ConfirmatoryFrozenFeatureCacheSpec(
                scenario_id=scenario_id,
                cache_path=selected.cache_path,
                expected_cache_sha256=selected.cache_file_sha256,
                expected_metadata_sha256=selected.sidecar_file_sha256,
                expected_weight_sha256=_sha(
                    selected_record.get("weights_sha256"),
                    f"frozen scenario {scenario_id} weight hash",
                ),
            )
        )
    if not specs:
        raise ConfirmatoryCLIInputError("confirmatory authority yields no available frozen caches")
    return tuple(specs)


def _resolve_bound_confirmatory_inputs(
    *,
    primary_run: Path,
    config_path: Path,
    manifest: Path,
    config: Mapping[str, Any],
    config_file_sha: str,
    semantic_sha: str,
    manifest_sha: str,
) -> ConfirmatoryCLIInputs:
    """Resolve cache/input bindings shared by original and amended executions."""

    bindings_path = primary_run / _PRIMARY_INPUT_BINDINGS_FILENAME
    bindings_file_sha = _sha256_file(bindings_path)
    bindings = _read_json_object(bindings_path, "sealed primary input bindings")
    if bindings.get("schema_version") != 1:
        raise ConfirmatoryCLIInputError("primary input bindings schema is unsupported")
    bound_paths, expected, verified = _verified_primary_cache_paths(bindings)
    raw_manifest_path = _mapping(bindings.get("cache_paths"), "primary cache paths").get(
        "dataset_manifest_path"
    )
    if raw_manifest_path is None:
        raise ConfirmatoryCLIInputError("primary input bindings lack dataset_manifest_path")
    bound_manifest = _regular_bound_file(raw_manifest_path, "primary dataset manifest")
    if bound_manifest != manifest:
        raise ConfirmatoryCLIInputError("primary-bound manifest path differs from gated manifest")
    for source in (expected, verified):
        field = "dataset_manifest_sha256"
        if _sha(source.get(field), f"primary {field}") != manifest_sha:
            raise ConfirmatoryCLIInputError("primary manifest hash differs from gated manifest")

    crop_path = bound_paths.get("crop_cache_path")
    if crop_path is None:
        raise ConfirmatoryCLIInputError("primary input bindings lack the required crop cache")
    crop = verify_frozen_cache_sidecar(
        crop_path,
        expected_manifest_sha256=manifest_sha,
        expected_representation_id="pannuke_component_covering_target_crops",
    )
    raw_inventory = _sha(
        expected.get("raw_inventory_sha256"), "primary expected raw inventory hash"
    )
    if (
        _sha(verified.get("raw_inventory_sha256"), "primary verified raw inventory hash")
        != raw_inventory
        or crop.metadata.get("raw_inventory_sha256") != raw_inventory
    ):
        raise ConfirmatoryCLIInputError(
            "crop cache raw-inventory binding differs from primary input evidence"
        )

    _enforce_pathology_authority(
        config,
        bound_paths=bound_paths,
        expected_hashes=expected,
        verified_hashes=verified,
    )
    candidates = _candidate_cache_paths(bound_paths)
    verified_candidates = _verify_candidate_caches(candidates, manifest_sha256=manifest_sha)
    specs = _frozen_feature_specs(config, verified_candidates)
    observed_label_sets = _materialize_observed_label_sets(
        crop.cache_path,
        config=config,
        config_semantic_sha256=semantic_sha,
    )

    source_hashes = {
        bindings_path: bindings_file_sha,
        config_path: config_file_sha,
        manifest: manifest_sha,
        crop.cache_path: crop.cache_file_sha256,
        crop.sidecar_path: crop.sidecar_file_sha256,
    }
    for spec in specs:
        path = Path(spec.cache_path).resolve()
        source_hashes[path] = spec.expected_cache_sha256
        source_hashes[path.with_suffix(f"{path.suffix}.metadata.json")] = (
            spec.expected_metadata_sha256
        )
    for path, digest in source_hashes.items():
        if _sha256_file(path) != digest:
            raise ConfirmatoryCLIInputError(
                f"confirmatory CLI input changed during derivation: {path}"
            )

    return ConfirmatoryCLIInputs(
        crop_cache_path=crop.cache_path,
        expected_crop_cache_sha256=crop.cache_file_sha256,
        expected_crop_metadata_sha256=crop.sidecar_file_sha256,
        expected_raw_inventory_sha256=raw_inventory,
        frozen_feature_caches=specs,
        observed_label_sets=observed_label_sets,
        primary_input_bindings_sha256=bindings_file_sha,
        confirmatory_config_sha256=config_file_sha,
        confirmatory_config_semantic_sha256=semantic_sha,
    )


def resolve_confirmatory_cli_inputs(
    *,
    gate_evidence: ConfirmatoryExecutionGateEvidence,
    primary_run_directory: str | Path,
    frozen_confirmatory_config_path: str | Path,
    manifest_path: str | Path,
) -> ConfirmatoryCLIInputs:
    """Derive exact runner inputs after lifecycle and confirmatory gate validation."""

    if not isinstance(gate_evidence, ConfirmatoryExecutionGateEvidence):
        raise TypeError("gate_evidence must be verified ConfirmatoryExecutionGateEvidence")
    primary_run = Path(primary_run_directory).resolve()
    config_path = Path(frozen_confirmatory_config_path).resolve()
    manifest = Path(manifest_path).resolve()
    if gate_evidence.primary_run_directory.resolve() != primary_run:
        raise ConfirmatoryCLIInputError("CLI primary run differs from confirmatory gate evidence")
    if config_path != gate_evidence.primary_gate.freeze_directory / "confirmatory_frozen.yaml":
        raise ConfirmatoryCLIInputError(
            "CLI confirmatory config is not the canonical file in the verified authority"
        )
    config_file_sha = _sha256_file(config_path)
    if config_file_sha != gate_evidence.primary_gate.frozen_confirmatory_config_sha256:
        raise ConfirmatoryCLIInputError("confirmatory config bytes differ from gate evidence")
    config = validate_frozen_confirmatory_config(load_config(config_path))
    semantic_sha = config_sha256(config)
    if semantic_sha != gate_evidence.primary_gate.confirmatory_config_semantic_sha256:
        raise ConfirmatoryCLIInputError("confirmatory config semantics differ from gate evidence")
    manifest_sha = _sha256_file(manifest)
    if manifest_sha != gate_evidence.primary_gate.manifest_sha256:
        raise ConfirmatoryCLIInputError("manifest bytes differ from gate evidence")
    return _resolve_bound_confirmatory_inputs(
        primary_run=primary_run,
        config_path=config_path,
        manifest=manifest,
        config=config,
        config_file_sha=config_file_sha,
        semantic_sha=semantic_sha,
        manifest_sha=manifest_sha,
    )


def resolve_resource_bounded_cli_inputs(
    *,
    gate_evidence: ResourceBoundedExecutionGateEvidence,
    primary_run_directory: str | Path,
    resource_confirmatory_config_path: str | Path,
    manifest_path: str | Path,
) -> ConfirmatoryCLIInputs:
    """Derive inputs from historical P data and the current C resource profile."""

    if not isinstance(gate_evidence, ResourceBoundedExecutionGateEvidence):
        raise TypeError("gate_evidence must be verified ResourceBoundedExecutionGateEvidence")
    if (
        gate_evidence.analysis_disposition != "amended_or_exploratory"
        or gate_evidence.study_outcome_eligible is not False
        or gate_evidence.completion_stage is not None
        or gate_evidence.original_confirmatory_claim_allowed is not False
    ):
        raise ConfirmatoryCLIInputError(
            "resource-bounded gate does not preserve its permanent ineligible disposition"
        )
    primary_run = Path(primary_run_directory).resolve()
    config_path = Path(resource_confirmatory_config_path).resolve()
    manifest = Path(manifest_path).resolve()
    historical = gate_evidence.historical_primary
    authority = gate_evidence.execution_authority
    if historical.primary_run_directory.resolve() != primary_run:
        raise ConfirmatoryCLIInputError("CLI primary run differs from historical P evidence")
    if config_path != authority.authority_directory / "confirmatory_frozen.yaml":
        raise ConfirmatoryCLIInputError(
            "resource config is not the canonical snapshot in execution authority C"
        )
    config_file_sha = _sha256_file(config_path)
    if config_file_sha != authority.resource_confirmatory_config_file_sha256:
        raise ConfirmatoryCLIInputError("resource config bytes differ from authority C")
    config = validate_resource_bounded_confirmatory_config(load_config(config_path))
    semantic_sha = config_sha256(config)
    if semantic_sha != authority.resource_confirmatory_config_semantic_sha256:
        raise ConfirmatoryCLIInputError("resource config semantics differ from authority C")
    manifest_sha = _sha256_file(manifest)
    if manifest_sha != historical.primary_gate.manifest_sha256:
        raise ConfirmatoryCLIInputError("manifest bytes differ from historical P")
    return _resolve_bound_confirmatory_inputs(
        primary_run=primary_run,
        config_path=config_path,
        manifest=manifest,
        config=config,
        config_file_sha=config_file_sha,
        semantic_sha=semantic_sha,
        manifest_sha=manifest_sha,
    )


__all__ = [
    "ConfirmatoryCLIInputError",
    "ConfirmatoryCLIInputs",
    "resolve_confirmatory_cli_inputs",
    "resolve_resource_bounded_cli_inputs",
]
