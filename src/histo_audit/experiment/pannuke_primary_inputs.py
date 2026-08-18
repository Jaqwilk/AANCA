"""Strict, hash-bound PanNuke cache adapter for the primary matrix.

This module is deliberately an adapter rather than an experiment runner.  It accepts
only a complete schema-v2 primary configuration, verifies every immutable cache and
freeze binding available to it, creates one deterministic source-group partition, and
returns :class:`~histo_audit.experiment.primary_core.PrimaryMatrixInputs`.

The final official fold is read only as identity/reference data.  No observed label,
corruption flag, or outcome-dependent choice is accepted by the crop-cache schema.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage  # type: ignore[import-untyped]

from histo_audit.config import config_sha256
from histo_audit.corruption.controlled import (
    FeatureIndependenceEvidence,
    FeatureSpaceEvidence,
    array_artifact_sha256,
    canonical_sha256,
)
from histo_audit.data.targets import highlight_target
from histo_audit.experiment.primary_core import PrimaryMatrixInputs
from histo_audit.experiment.reference_groups import (
    ReferenceGroupSelectionError,
    deterministic_group_greedy_class_distribution_v1,
)
from histo_audit.experiment.study_contracts import (
    PrimaryMatrixPlan,
    build_primary_matrix_plan,
    validate_frozen_primary_config,
)
from histo_audit.representations.cache_provenance import (
    array_artifact_sha256 as cache_array_artifact_sha256,
)
from histo_audit.representations.eligibility import validate_analysis_eligibility_provenance
from histo_audit.representations.imagenet import EmbeddingResult, load_embedding_cache
from histo_audit.utils.run_tracking import sha256_file
from histo_audit.workflows.preregistration import BASE_FREEZE_EVIDENCE_SCHEMA_VERSION

StringArray = NDArray[np.str_]
IntegerArray = NDArray[np.integer[Any]]
FeatureArray = NDArray[np.floating[Any]]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INSTANCE_CONNECTIVITY_4 = ndimage.generate_binary_structure(2, 1)
_CROP_KEYS = frozenset(
    {
        "sample_ids",
        "context_rgb",
        "target_highlighted_rgb",
        "target_masks",
        "target_contour_masks",
        "raw_component_counts",
        "disconnected_instance_flags",
        "projected_union_component_counts",
        "projection_fallback_component_counts",
        "projection_collision_pixel_counts",
        "projection_collision_excess_counts",
        "projection_adjacency_pair_counts",
        "projection_topology_changed",
        "projected_component_pixel_counts",
        "projected_component_unique_pixel_counts",
        "baseline_projected_component_counts",
        "projection_fallback_component_flags",
        "projected_component_offsets",
        "source_crop_boxes",
        "source_target_boxes",
        "official_folds",
        "source_patch_indices",
        "instance_channel_indices",
        "instance_ids",
        "pre_corruption_labels",
        "group_ids",
        "tissue_types",
        "source_contour_xy",
        "source_contour_offsets",
        "identity_verified",
        "primary_eligible",
        "confirmatory_eligible",
    }
)
_ENGINEERED_KEYS = frozenset({"values", "names", "sample_ids"})
_PRIMARY_CACHE_PROVENANCE_FIELDS = frozenset(
    {
        "status",
        "encoder_id",
        "encoder_implementation_sha256",
        "weights_sha256",
        "preprocessing_sha256",
        "sample_order_sha256",
        "dataset_manifest_sha256",
        "cache_recipe_sha256",
        "cache_file_sha256",
    }
)


class PanNukePrimaryInputError(ValueError):
    """A frozen binding, cache, partition, or feature-space contract is invalid."""


@dataclass(frozen=True, slots=True)
class PanNukePrimaryCachePaths:
    """Immutable cache and evidence locations consumed by the adapter."""

    crop_cache_path: Path
    engineered_cache_path: Path
    context_embedding_cache_path: Path | None
    highlighted_embedding_cache_path: Path | None
    pathology_embedding_cache_path: Path | None = None
    pathology_availability_audit_path: Path | None = None
    dataset_evidence_path: Path | None = None
    dataset_manifest_path: Path | None = None
    freeze_record_path: Path | None = None


@dataclass(frozen=True, slots=True)
class PanNukePrimaryHashExpectations:
    """Optional outer freeze/checksum values verified in addition to sidecars."""

    dataset_evidence_sha256: str | None = None
    dataset_manifest_sha256: str | None = None
    raw_inventory_sha256: str | None = None
    crop_cache_sha256: str | None = None
    engineered_cache_sha256: str | None = None
    context_embedding_cache_sha256: str | None = None
    highlighted_embedding_cache_sha256: str | None = None
    pathology_embedding_cache_sha256: str | None = None
    freeze_record_sha256: str | None = None

    def validate(self) -> None:
        """Reject malformed expected digests before touching study inputs."""

        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if value is not None:
                _require_sha256(str(value), field_name)


@dataclass(frozen=True, slots=True)
class RepresentationAvailability:
    """One frozen representation's exact adapter availability state."""

    representation_id: str
    family: str
    required: bool
    status: Literal["available", "unavailable_optional"]
    blocker: str | None
    cache_sha256: str | None


@dataclass(frozen=True, slots=True)
class PanNukePrimaryInputsResult:
    """Primary inputs plus cryptographic and partition provenance."""

    inputs: PrimaryMatrixInputs
    plan: PrimaryMatrixPlan
    config_sha256: str
    plan_semantic_sha256: str
    sample_order_sha256: str
    partition_assignment_sha256: str
    selected_reference_validation_groups: tuple[str, ...]
    morphology_feature_names: tuple[str, ...]
    cache_provenance_by_representation: Mapping[str, Mapping[str, Any]]
    independence_evidence_by_representation: Mapping[str, FeatureIndependenceEvidence]
    representation_availability: tuple[RepresentationAvailability, ...]
    verified_hashes: Mapping[str, str]
    eligibility_provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _CropCache:
    sample_ids: StringArray
    group_ids: StringArray
    tissue_types: StringArray
    official_folds: NDArray[np.int64]
    pre_corruption_labels: NDArray[np.int64]
    metadata: Mapping[str, Any]
    cache_sha256: str
    sidecar_sha256: str
    cache_content_sha256: str
    array_sha256_by_name: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _FeatureCache:
    values: FeatureArray
    sample_ids: StringArray
    names: tuple[str, ...] | None
    metadata: Mapping[str, Any]
    cache_sha256: str


def _require_sha256(value: str, field_name: str) -> str:
    normalised = value.casefold()
    if _SHA256.fullmatch(normalised) is None:
        raise PanNukePrimaryInputError(f"{field_name} must be a 64-character SHA-256")
    return normalised


def _resolved_file(path: str | Path, field_name: str) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{field_name} does not exist: {source}")
    return source


def _metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(f"{cache_path.suffix}.metadata.json")


def _load_json_mapping(path: Path, field_name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PanNukePrimaryInputError(f"cannot read {field_name}: {path}: {exc}") from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise PanNukePrimaryInputError(f"{field_name} must contain a JSON object")
    return cast(dict[str, Any], value)


def verify_pilot_derived_parameters_binding(
    frozen_config: Mapping[str, Any],
    project_root: str | Path,
) -> str:
    """Verify the pilot-derived report and its exact frozen scientific values.

    This preflight deliberately runs before any model-facing cache is loaded.  The
    configuration freezes both the report identity and the subset of its output that
    controls primary corruption, so neither a replaced report nor hand-copied values
    can silently enter the study.
    """

    pilot_binding = frozen_config.get("pilot_derived_parameters")
    corruption = frozen_config.get("corruption")
    if not isinstance(pilot_binding, Mapping) or not isinstance(corruption, Mapping):
        raise PanNukePrimaryInputError(
            "frozen primary config lacks the pilot-derived parameter binding"
        )
    mechanisms = corruption.get("mechanisms")
    if not isinstance(mechanisms, Mapping):
        raise PanNukePrimaryInputError("frozen primary corruption mechanisms are absent")

    root = Path(project_root).expanduser().resolve()
    report_path = (root / str(pilot_binding.get("path", ""))).resolve()
    try:
        report_path.relative_to(root)
    except ValueError as exc:
        raise PanNukePrimaryInputError(
            "pilot-derived parameter report must remain inside project_root"
        ) from exc
    if not report_path.is_file():
        raise PanNukePrimaryInputError(f"pilot-derived parameter report is missing: {report_path}")

    expected_sha256 = _require_sha256(
        str(pilot_binding.get("sha256", "")),
        "pilot_derived_parameters.sha256",
    )
    report_sha256 = sha256_file(report_path)
    if report_sha256 != expected_sha256:
        raise PanNukePrimaryInputError(
            "pilot-derived parameter report SHA-256 differs from the frozen config"
        )
    report = _load_json_mapping(report_path, "pilot-derived parameter report")
    if sha256_file(report_path) != report_sha256:
        raise PanNukePrimaryInputError("pilot-derived parameter report changed during preflight")

    if report.get("schema_version") != pilot_binding.get("schema_version"):
        raise PanNukePrimaryInputError(
            "pilot-derived parameter report schema_version differs from the frozen config"
        )
    if report.get("producer_id") != pilot_binding.get("producer_id"):
        raise PanNukePrimaryInputError(
            "pilot-derived parameter report producer_id differs from the frozen config"
        )
    source_pilot = report.get("source_pilot")
    if not isinstance(source_pilot, Mapping):
        raise PanNukePrimaryInputError(
            "pilot-derived parameter report lacks source_pilot provenance"
        )
    if source_pilot.get("run_id") != pilot_binding.get("source_pilot_run_id"):
        raise PanNukePrimaryInputError(
            "pilot-derived parameter report source run differs from the frozen config"
        )
    if source_pilot.get("artifact_root_sha256") != pilot_binding.get(
        "source_pilot_artifact_root_sha256"
    ):
        raise PanNukePrimaryInputError(
            "pilot-derived parameter report source artifact root differs from the frozen config"
        )

    report_confusion = report.get("confusion_targeted_corruption")
    frozen_confusion = mechanisms.get("confusion_targeted_corruption")
    if not isinstance(report_confusion, Mapping) or not isinstance(frozen_confusion, Mapping):
        raise PanNukePrimaryInputError("pilot-derived confusion-targeted parameters are absent")
    if report_confusion.get("transition_matrix") != frozen_confusion.get("transition_matrix"):
        raise PanNukePrimaryInputError(
            "pilot-derived transition matrix differs from the frozen primary config"
        )

    report_group = report.get("group_conditional_corruption")
    frozen_group = mechanisms.get("group_conditional_corruption")
    if not isinstance(report_group, Mapping) or not isinstance(frozen_group, Mapping):
        raise PanNukePrimaryInputError("pilot-derived group-conditional parameters are absent")
    for field_name in ("grouping_field", "weights_by_value", "default_weight"):
        if report_group.get(field_name) != frozen_group.get(field_name):
            raise PanNukePrimaryInputError(
                "pilot-derived group-conditional "
                f"{field_name} differs from the frozen primary config"
            )
    return report_sha256


def _load_sidecar(cache_path: Path, field_name: str) -> tuple[dict[str, Any], str, str]:
    sidecar = _resolved_file(_metadata_path(cache_path), f"{field_name} metadata sidecar")
    sidecar_sha256 = sha256_file(sidecar)
    metadata = _load_json_mapping(sidecar, f"{field_name} metadata sidecar")
    if sha256_file(sidecar) != sidecar_sha256:
        raise PanNukePrimaryInputError(f"{field_name} sidecar changed during loading")
    expected = _require_sha256(str(metadata.get("cache_npz_sha256", "")), "cache_npz_sha256")
    actual = sha256_file(cache_path)
    if actual != expected:
        raise PanNukePrimaryInputError(f"{field_name} checksum does not match its sidecar")
    return metadata, actual, sidecar_sha256


def _safe_npz_arrays(
    path: Path,
    *,
    expected_keys: frozenset[str],
    field_name: str,
) -> dict[str, NDArray[np.generic]]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            keys = frozenset(payload.files)
            if keys != expected_keys:
                missing = sorted(expected_keys.difference(keys))
                unexpected = sorted(keys.difference(expected_keys))
                raise PanNukePrimaryInputError(
                    f"{field_name} schema differs (missing={missing}, unexpected={unexpected})"
                )
            arrays: dict[str, NDArray[np.generic]] = {}
            for key in sorted(keys):
                array = np.asarray(payload[key])
                if array.dtype.hasobject:
                    raise PanNukePrimaryInputError(
                        f"{field_name}/{key} has unsafe object/pickle-dependent dtype"
                    )
                arrays[key] = array
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, PanNukePrimaryInputError):
            raise
        raise PanNukePrimaryInputError(f"cannot safely load {field_name}: {exc}") from exc
    return arrays


def _string_vector(value: NDArray[np.generic], name: str, *, n: int | None = None) -> StringArray:
    if value.ndim != 1 or value.dtype.kind not in {"U", "S"}:
        raise PanNukePrimaryInputError(f"{name} must be a one-dimensional string array")
    result = np.asarray(value.astype(str), dtype=np.str_)
    if n is not None and result.shape != (n,):
        raise PanNukePrimaryInputError(f"{name} does not align with sample_ids")
    if any(not str(item).strip() for item in result):
        raise PanNukePrimaryInputError(f"{name} contains an empty value")
    return result


def _integer_vector(value: NDArray[np.generic], name: str, *, n: int) -> NDArray[np.int64]:
    if value.shape != (n,) or not np.issubdtype(value.dtype, np.integer):
        raise PanNukePrimaryInputError(f"{name} must be an aligned integer vector")
    return np.asarray(value, dtype=np.int64)


def _load_crop_cache(path: str | Path, class_order: tuple[int, ...]) -> _CropCache:
    source = _resolved_file(path, "PanNuke crop cache")
    if source.suffix.casefold() != ".npz":
        raise PanNukePrimaryInputError("PanNuke crop cache must be an .npz file")
    metadata, checksum, sidecar_checksum = _load_sidecar(source, "PanNuke crop cache")
    arrays = _safe_npz_arrays(
        source,
        expected_keys=_CROP_KEYS,
        field_name="PanNuke crop cache",
    )
    array_sha256_by_name = {
        name: cache_array_artifact_sha256(value) for name, value in sorted(arrays.items())
    }
    cache_content_sha256 = canonical_sha256(array_sha256_by_name)
    if (
        metadata.get("cache_array_sha256_by_name") != array_sha256_by_name
        or metadata.get("cache_content_sha256") != cache_content_sha256
        or metadata.get("cache_file_sha256", metadata.get("cache_npz_sha256")) != checksum
        or sha256_file(source) != checksum
        or sha256_file(_metadata_path(source)) != sidecar_checksum
    ):
        raise PanNukePrimaryInputError(
            "PanNuke crop cache content or sidecar changed during loading"
        )
    sample_ids = _string_vector(arrays["sample_ids"], "crop sample_ids")
    n = len(sample_ids)
    if n == 0 or len(set(sample_ids.tolist())) != n:
        raise PanNukePrimaryInputError("crop sample_ids must be non-empty and unique")
    group_ids = _string_vector(arrays["group_ids"], "crop group_ids", n=n)
    tissue_types = _string_vector(arrays["tissue_types"], "crop tissue_types", n=n)
    folds = _integer_vector(arrays["official_folds"], "crop official_folds", n=n)
    labels = _integer_vector(arrays["pre_corruption_labels"], "crop pre_corruption_labels", n=n)
    if not set(labels.tolist()).issubset(class_order):
        raise PanNukePrimaryInputError("crop reference labels lie outside frozen class_order")

    context = arrays["context_rgb"]
    highlighted = arrays["target_highlighted_rgb"]
    if (
        context.ndim != 4
        or context.shape[0] != n
        or context.shape[1] <= 0
        or context.shape[1] != context.shape[2]
        or context.shape[3] != 3
        or context.dtype != np.uint8
    ):
        raise PanNukePrimaryInputError("context_rgb must be non-empty square uint8 RGB crops")
    if highlighted.shape != context.shape or highlighted.dtype != np.uint8:
        raise PanNukePrimaryInputError("highlighted RGB crops must align with context RGB")
    masks = cast(NDArray[np.bool_], arrays["target_masks"])
    contours = cast(NDArray[np.bool_], arrays["target_contour_masks"])
    expected_mask_shape = context.shape[:3]
    if masks.shape != expected_mask_shape or masks.dtype != np.bool_:
        raise PanNukePrimaryInputError("target_masks must be aligned boolean masks")
    if contours.shape != expected_mask_shape or contours.dtype != np.bool_:
        raise PanNukePrimaryInputError("target_contour_masks must be aligned boolean masks")
    if not np.all(masks.reshape(n, -1).any(axis=1)):
        raise PanNukePrimaryInputError(
            "every cached crop must retain its component-covering projected target"
        )
    if not np.all(contours.reshape(n, -1).any(axis=1)) or np.any(contours & ~masks):
        raise PanNukePrimaryInputError("target contours must be non-empty target-mask subsets")
    crop_configuration = metadata.get("crop_configuration")
    if not isinstance(crop_configuration, Mapping):
        raise PanNukePrimaryInputError("crop configuration provenance is absent")
    context_brightness = crop_configuration.get("context_brightness")
    if (
        isinstance(context_brightness, bool)
        or not isinstance(context_brightness, (int, float))
        or not 0.0 <= float(context_brightness) <= 1.0
        or any(
            not np.array_equal(
                highlighted_image,
                highlight_target(
                    context_image,
                    mask,
                    context_brightness=float(context_brightness),
                ),
            )
            for context_image, mask, highlighted_image in zip(
                context,
                masks,
                highlighted,
                strict=True,
            )
        )
    ):
        raise PanNukePrimaryInputError(
            "highlighted crop content differs from context/mask/provenance"
        )

    raw_component_counts = _integer_vector(
        arrays["raw_component_counts"],
        "crop raw_component_counts",
        n=n,
    )
    projected_union_counts = _integer_vector(
        arrays["projected_union_component_counts"],
        "crop projected_union_component_counts",
        n=n,
    )
    fallback_counts = _integer_vector(
        arrays["projection_fallback_component_counts"],
        "crop projection_fallback_component_counts",
        n=n,
    )
    collision_counts = _integer_vector(
        arrays["projection_collision_pixel_counts"],
        "crop projection_collision_pixel_counts",
        n=n,
    )
    collision_excess = _integer_vector(
        arrays["projection_collision_excess_counts"],
        "crop projection_collision_excess_counts",
        n=n,
    )
    adjacency_counts = _integer_vector(
        arrays["projection_adjacency_pair_counts"],
        "crop projection_adjacency_pair_counts",
        n=n,
    )
    disconnected_flags = arrays["disconnected_instance_flags"]
    topology_changed = arrays["projection_topology_changed"]
    recomputed_union_counts = np.asarray(
        [int(ndimage.label(mask, structure=_INSTANCE_CONNECTIVITY_4)[1]) for mask in masks],
        dtype=np.int64,
    )
    maximum_adjacency_pairs = raw_component_counts * (raw_component_counts - 1) // 2
    if (
        disconnected_flags.shape != (n,)
        or disconnected_flags.dtype != np.bool_
        or topology_changed.shape != (n,)
        or topology_changed.dtype != np.bool_
        or np.any(raw_component_counts < 1)
        or np.any(projected_union_counts < 1)
        or np.any(projected_union_counts > raw_component_counts)
        or not np.array_equal(projected_union_counts, recomputed_union_counts)
        or not np.array_equal(disconnected_flags, raw_component_counts > 1)
        or np.any(fallback_counts < 0)
        or np.any(fallback_counts > raw_component_counts)
        or np.any(collision_counts < 0)
        or np.any(collision_excess < 0)
        or np.any(collision_excess < collision_counts)
        or np.any((collision_counts == 0) != (collision_excess == 0))
        or np.any(adjacency_counts < 0)
        or np.any(adjacency_counts > maximum_adjacency_pairs)
        or not np.array_equal(topology_changed, projected_union_counts != raw_component_counts)
    ):
        raise PanNukePrimaryInputError("crop component-projection vectors are inconsistent")
    component_pixels = arrays["projected_component_pixel_counts"]
    component_unique = arrays["projected_component_unique_pixel_counts"]
    baseline_component_counts = arrays["baseline_projected_component_counts"]
    fallback_component_flags = arrays["projection_fallback_component_flags"]
    component_offsets = arrays["projected_component_offsets"]
    if (
        component_pixels.ndim != 1
        or component_unique.shape != component_pixels.shape
        or not np.issubdtype(component_pixels.dtype, np.integer)
        or not np.issubdtype(component_unique.dtype, np.integer)
        or baseline_component_counts.shape != component_pixels.shape
        or not np.issubdtype(baseline_component_counts.dtype, np.integer)
        or fallback_component_flags.shape != component_pixels.shape
        or fallback_component_flags.dtype != np.bool_
        or component_offsets.shape != (n + 1,)
        or not np.issubdtype(component_offsets.dtype, np.integer)
    ):
        raise PanNukePrimaryInputError("crop per-component projection ledger is malformed")
    component_pixels_int = np.asarray(component_pixels, dtype=np.int64)
    component_unique_int = np.asarray(component_unique, dtype=np.int64)
    baseline_component_counts_int = np.asarray(baseline_component_counts, dtype=np.int64)
    if (
        np.any(component_pixels_int <= 0)
        or np.any(component_unique_int <= 0)
        or np.any(component_unique_int > component_pixels_int)
        or np.any(baseline_component_counts_int < 0)
        or not np.array_equal(
            fallback_component_flags,
            baseline_component_counts_int != 1,
        )
    ):
        raise PanNukePrimaryInputError("crop per-component projection ledger is malformed")
    offsets_projection = np.asarray(component_offsets, dtype=np.int64)
    if (
        offsets_projection[0] != 0
        or offsets_projection[-1] != len(component_pixels)
        or not np.array_equal(np.diff(offsets_projection), raw_component_counts)
    ):
        raise PanNukePrimaryInputError("crop component offsets do not match raw counts")
    for index in range(n):
        start = int(offsets_projection[index])
        stop = int(offsets_projection[index + 1])
        if int(masks[index].sum()) != (
            int(np.sum(component_pixels_int[start:stop])) - int(collision_excess[index])
        ) or int(np.count_nonzero(fallback_component_flags[start:stop])) != int(
            fallback_counts[index]
        ):
            raise PanNukePrimaryInputError("crop projected-mask pixel accounting differs")

    projection_raw = metadata.get("target_mask_projection")
    if not isinstance(projection_raw, Mapping):
        raise PanNukePrimaryInputError("crop target-mask projection provenance is absent")
    projection = dict(projection_raw)
    semantic_sha256 = projection.pop("semantic_sha256", None)
    records = projection.get("disconnected_instances")
    expected_fallback_ids = [
        str(sample_id)
        for sample_id, count in zip(
            sample_ids.tolist(),
            fallback_counts.tolist(),
            strict=True,
        )
        if count > 0
    ]
    expected_collision_ids = [
        str(sample_id)
        for sample_id, count in zip(
            sample_ids.tolist(),
            collision_counts.tolist(),
            strict=True,
        )
        if count > 0
    ]
    expected_adjacency_ids = [
        str(sample_id)
        for sample_id, count in zip(
            sample_ids.tolist(),
            adjacency_counts.tolist(),
            strict=True,
        )
        if count > 0
    ]
    expected_topology_ids = [
        str(sample_id)
        for sample_id, changed in zip(
            sample_ids.tolist(),
            topology_changed.tolist(),
            strict=True,
        )
        if changed
    ]
    if (
        projection.get("schema_version") != 1
        or projection.get("identifier") != "nearest_per_component_with_forward_fallback_v1"
        or projection.get("raw_component_connectivity") != "4-connected"
        or projection.get("quality_flag") != "disconnected_instance_id"
        or projection.get("raw_identity_action")
        != "retain_one_raw_identity_without_split_merge_repair_or_relabel"
        or projection.get("manifest_and_validation_must_agree_for_disconnected_identity")
        is not True
        or projection.get("all_raw_components_must_contribute") is not True
        or projection.get("projected_union_topology_is_exact") is not False
        or projection.get("source_annotations_modified") is not False
        or projection.get("sample_count") != n
        or projection.get("raw_component_count") != int(raw_component_counts.sum())
        or projection.get("zero_covered_component_count") != 0
        or projection.get("disconnected_instance_count")
        != int(np.count_nonzero(disconnected_flags))
        or projection.get("fallback_component_count") != int(fallback_counts.sum())
        or projection.get("fallback_instance_ids") != expected_fallback_ids
        or projection.get("collision_instance_count") != int(np.count_nonzero(collision_counts))
        or projection.get("collision_instance_ids") != expected_collision_ids
        or projection.get("adjacency_instance_count") != int(np.count_nonzero(adjacency_counts))
        or projection.get("adjacency_instance_ids") != expected_adjacency_ids
        or projection.get("topology_changed_instance_count")
        != int(np.count_nonzero(topology_changed))
        or projection.get("topology_changed_instance_ids") != expected_topology_ids
        or semantic_sha256 != canonical_sha256(projection)
        or not isinstance(records, list)
    ):
        raise PanNukePrimaryInputError("crop target-mask projection provenance is invalid")
    expected_record_ids = tuple(
        str(sample_id)
        for sample_id, flagged in zip(
            sample_ids.tolist(),
            disconnected_flags.tolist(),
            strict=True,
        )
        if flagged
    )
    record_ids = tuple(
        str(record.get("sample_id")) if isinstance(record, Mapping) else "" for record in records
    )
    if record_ids != expected_record_ids:
        raise PanNukePrimaryInputError("crop disconnected-instance ledger is not sample-aligned")
    index_by_id = {str(sample_id): index for index, sample_id in enumerate(sample_ids)}
    for record in records:
        if not isinstance(record, Mapping):
            raise PanNukePrimaryInputError("crop disconnected-instance record is invalid")
        index = index_by_id[str(record["sample_id"])]
        start = int(offsets_projection[index])
        stop = int(offsets_projection[index + 1])
        components = record.get("components")
        if (
            record.get("component_count") != int(raw_component_counts[index])
            or record.get("projected_union_component_count") != int(projected_union_counts[index])
            or record.get("fallback_component_count") != int(fallback_counts[index])
            or record.get("collision_pixel_count") != int(collision_counts[index])
            or record.get("collision_excess_count") != int(collision_excess[index])
            or record.get("adjacency_pair_count") != int(adjacency_counts[index])
            or record.get("topology_changed") is not bool(topology_changed[index])
            or record.get("projected_component_pixel_counts")
            != component_pixels_int[start:stop].tolist()
            or record.get("projected_component_unique_pixel_counts")
            != component_unique_int[start:stop].tolist()
            or record.get("projected_target_mask_sha256")
            != cache_array_artifact_sha256(masks[index])
            or not isinstance(components, list)
            or len(components) != int(raw_component_counts[index])
        ):
            raise PanNukePrimaryInputError(
                "crop disconnected-instance record differs from aligned vectors"
            )
        for component_index, component in enumerate(components, start=1):
            if (
                not isinstance(component, Mapping)
                or component.get("component_index") != component_index
                or component.get("projected_pixel_count")
                != int(component_pixels_int[start + component_index - 1])
                or component.get("projected_unique_pixel_count")
                != int(component_unique_int[start + component_index - 1])
                or component.get("baseline_projected_component_count")
                != int(baseline_component_counts_int[start + component_index - 1])
                or component.get("fallback_used")
                is not bool(fallback_component_flags[start + component_index - 1])
            ):
                raise PanNukePrimaryInputError(
                    "crop disconnected component ledger differs from aligned vectors"
                )

    for key in ("source_crop_boxes", "source_target_boxes"):
        value = arrays[key]
        if value.shape != (n, 4) or not np.issubdtype(value.dtype, np.integer):
            raise PanNukePrimaryInputError(f"{key} must be an aligned integer (n, 4) matrix")
    for key in (
        "source_patch_indices",
        "instance_channel_indices",
        "instance_ids",
    ):
        _integer_vector(arrays[key], f"crop {key}", n=n)
    identity = arrays["identity_verified"]
    if identity.shape != (n,) or identity.dtype != np.bool_ or not bool(identity.all()):
        raise PanNukePrimaryInputError("every crop must retain verified raw-instance identity")
    try:
        validate_analysis_eligibility_provenance(
            metadata,
            sample_ids,
            primary_eligible=arrays["primary_eligible"],
            confirmatory_eligible=arrays["confirmatory_eligible"],
        )
    except ValueError as exc:
        raise PanNukePrimaryInputError(f"crop analysis eligibility is invalid: {exc}") from exc
    contour_xy = arrays["source_contour_xy"]
    contour_offsets = arrays["source_contour_offsets"]
    if (
        contour_xy.ndim != 2
        or contour_xy.shape[1] != 2
        or not np.issubdtype(contour_xy.dtype, np.integer)
        or contour_offsets.shape != (n + 1,)
        or not np.issubdtype(contour_offsets.dtype, np.integer)
    ):
        raise PanNukePrimaryInputError("source contour coordinate/offset arrays are malformed")
    offsets = np.asarray(contour_offsets, dtype=np.int64)
    if offsets[0] != 0 or offsets[-1] != len(contour_xy) or np.any(np.diff(offsets) <= 0):
        raise PanNukePrimaryInputError("every source contour must be non-empty and aligned")
    if metadata.get("schema_version") != 1 or int(metadata.get("sample_count", -1)) != n:
        raise PanNukePrimaryInputError("crop sidecar schema/sample count is invalid")
    for key in ("manifest_sha256", "raw_inventory_sha256"):
        _require_sha256(str(metadata.get(key, "")), f"crop metadata {key}")
    return _CropCache(
        sample_ids,
        group_ids,
        tissue_types,
        folds,
        labels,
        metadata,
        checksum,
        sidecar_checksum,
        cache_content_sha256,
        array_sha256_by_name,
    )


def _load_engineered_cache(path: str | Path, expected_ids: StringArray) -> _FeatureCache:
    source = _resolved_file(path, "engineered feature cache")
    metadata, checksum, sidecar_checksum = _load_sidecar(source, "engineered feature cache")
    arrays = _safe_npz_arrays(
        source,
        expected_keys=_ENGINEERED_KEYS,
        field_name="engineered feature cache",
    )
    array_hashes = {
        name: cache_array_artifact_sha256(value) for name, value in sorted(arrays.items())
    }
    if (
        metadata.get("cache_array_sha256_by_name") != array_hashes
        or metadata.get("cache_content_sha256") != canonical_sha256(array_hashes)
        or sha256_file(source) != checksum
        or sha256_file(_metadata_path(source)) != sidecar_checksum
    ):
        raise PanNukePrimaryInputError("engineered feature cache changed during loading")
    ids = _string_vector(arrays["sample_ids"], "engineered sample_ids")
    if not np.array_equal(ids, expected_ids):
        raise PanNukePrimaryInputError("engineered cache sample IDs/order differ from crop cache")
    names_array = _string_vector(arrays["names"], "engineered feature names")
    names = tuple(str(value) for value in names_array)
    if not names or len(set(names)) != len(names):
        raise PanNukePrimaryInputError("engineered feature names must be non-empty and unique")
    values = arrays["values"]
    if (
        values.ndim != 2
        or values.shape != (len(ids), len(names))
        or not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(values).all()
    ):
        raise PanNukePrimaryInputError("engineered feature matrix is malformed or non-finite")
    if metadata.get("schema_version") != 1:
        raise PanNukePrimaryInputError("engineered sidecar schema_version must be 1")
    if int(metadata.get("sample_count", -1)) != len(ids) or int(
        metadata.get("feature_count", -1)
    ) != len(names):
        raise PanNukePrimaryInputError("engineered sidecar dimensions differ from its cache")
    _require_sha256(
        str(metadata.get("crop_manifest_sha256", "")),
        "engineered crop_manifest_sha256",
    )
    return _FeatureCache(
        values=cast(FeatureArray, values),
        sample_ids=ids,
        names=names,
        metadata=metadata,
        cache_sha256=checksum,
    )


def _verified_imagenet_cache(
    path: str | Path,
    *,
    expected_ids: StringArray,
    expected_variant: str,
    expected_crop: _CropCache,
) -> tuple[EmbeddingResult, str]:
    source = _resolved_file(path, f"ImageNet {expected_variant} embedding cache")
    try:
        result = load_embedding_cache(source)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise PanNukePrimaryInputError(
            f"invalid ImageNet {expected_variant} embedding cache: {exc}"
        ) from exc
    if not np.array_equal(result.sample_ids, expected_ids):
        raise PanNukePrimaryInputError(
            f"ImageNet {expected_variant} sample IDs/order differ from crop cache"
        )
    if result.metadata.get("input_variant") != expected_variant:
        raise PanNukePrimaryInputError(
            f"ImageNet cache declares {result.metadata.get('input_variant')!r}, "
            f"expected {expected_variant!r}"
        )
    _require_sha256(str(result.metadata.get("weight_sha256", "")), "ImageNet weight_sha256")
    canonical_variant = "context_rgb" if expected_variant == "rgb" else expected_variant
    input_array_key = (
        "context_rgb" if canonical_variant == "context_rgb" else "target_highlighted_rgb"
    )
    projection = expected_crop.metadata.get("target_mask_projection")
    if not isinstance(projection, Mapping):
        raise PanNukePrimaryInputError("crop target-mask projection provenance is absent")
    expected_binding = {
        "schema_version": 1,
        "binding_type": "pannuke_component_covering_crop_cache_v1",
        "crop_cache_file_sha256": expected_crop.cache_sha256,
        "crop_cache_sidecar_file_sha256": expected_crop.sidecar_sha256,
        "crop_cache_content_sha256": expected_crop.cache_content_sha256,
        "crop_manifest_sha256": str(expected_crop.metadata["manifest_sha256"]),
        "raw_inventory_sha256": str(expected_crop.metadata["raw_inventory_sha256"]),
        "sample_order_sha256": canonical_sha256(expected_crop.sample_ids.tolist()),
        "target_mask_projection_semantic_sha256": str(projection.get("semantic_sha256")),
        "input_variant": canonical_variant,
        "input_array_key": input_array_key,
        "input_array_sha256": expected_crop.array_sha256_by_name[input_array_key],
    }
    binding = result.metadata.get("source_crop_cache_binding")
    if binding != expected_binding or result.metadata.get(
        "source_crop_cache_binding_sha256"
    ) != canonical_sha256(expected_binding):
        raise PanNukePrimaryInputError(
            f"ImageNet {expected_variant} embedding is not bound to the exact crop cache"
        )
    final_cache_sha256 = sha256_file(source)
    if final_cache_sha256 != result.metadata.get("cache_file_sha256"):
        raise PanNukePrimaryInputError(
            f"ImageNet {expected_variant} embedding cache changed during loading"
        )
    return result, final_cache_sha256


def _load_generic_pathology_cache(path: str | Path, expected_ids: StringArray) -> _FeatureCache:
    source = _resolved_file(path, "pathology embedding cache")
    metadata, checksum, sidecar_checksum = _load_sidecar(source, "pathology embedding cache")
    try:
        with np.load(source, allow_pickle=False) as payload:
            keys = frozenset(payload.files)
            if not {"embeddings", "sample_ids"}.issubset(keys):
                raise PanNukePrimaryInputError(
                    "pathology cache must contain embeddings and sample_ids"
                )
            unexpected = keys.difference({"embeddings", "sample_ids", "metadata_json"})
            if unexpected:
                raise PanNukePrimaryInputError(
                    f"pathology cache contains unexpected arrays: {sorted(unexpected)}"
                )
            embeddings = np.asarray(payload["embeddings"])
            ids_raw = np.asarray(payload["sample_ids"])
            if embeddings.dtype.hasobject or ids_raw.dtype.hasobject:
                raise PanNukePrimaryInputError("pathology cache contains object dtype")
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, PanNukePrimaryInputError):
            raise
        raise PanNukePrimaryInputError(f"cannot safely load pathology cache: {exc}") from exc
    if sha256_file(source) != checksum or sha256_file(_metadata_path(source)) != sidecar_checksum:
        raise PanNukePrimaryInputError("pathology embedding cache changed during loading")
    ids = _string_vector(ids_raw, "pathology sample_ids")
    if not np.array_equal(ids, expected_ids):
        raise PanNukePrimaryInputError("pathology cache sample IDs/order differ from crop cache")
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != len(ids)
        or embeddings.shape[1] == 0
        or not np.issubdtype(embeddings.dtype, np.floating)
        or not np.isfinite(embeddings).all()
    ):
        raise PanNukePrimaryInputError("pathology embedding matrix is malformed or non-finite")
    return _FeatureCache(
        values=cast(FeatureArray, embeddings),
        sample_ids=ids,
        names=None,
        metadata=metadata,
        cache_sha256=checksum,
    )


def select_stratified_reference_validation_groups(
    pre_corruption_labels: Sequence[int] | NDArray[np.integer[Any]],
    group_ids: Sequence[str] | StringArray,
    *,
    class_order: tuple[int, ...],
    fraction: float,
    seed: int,
) -> tuple[str, ...]:
    """Compatibility wrapper around the frozen shared reference-group selector."""

    try:
        return deterministic_group_greedy_class_distribution_v1(
            pre_corruption_labels,
            group_ids,
            class_order=class_order,
            fraction=fraction,
            seed=seed,
        )
    except ReferenceGroupSelectionError as exc:
        raise PanNukePrimaryInputError(str(exc)) from exc


def _parse_feature_space(value: Any, location: str) -> FeatureSpaceEvidence:
    if not isinstance(value, Mapping):
        raise PanNukePrimaryInputError(f"{location} must be a mapping")
    expected = {
        "representation_name",
        "feature_artifact_hash",
        "family",
        "implementation_hash",
        "weights_hash",
        "preprocessing_hash",
        "fitted_data_hash",
    }
    if set(value) != expected:
        raise PanNukePrimaryInputError(f"{location} fields do not match FeatureSpaceEvidence")
    evidence = FeatureSpaceEvidence(**{key: str(value[key]) for key in expected})
    try:
        evidence.validate()
    except ValueError as exc:
        raise PanNukePrimaryInputError(f"invalid {location}: {exc}") from exc
    return evidence


def _parse_independence_evidence(raw: Any, *, location: str) -> FeatureIndependenceEvidence:
    if not isinstance(raw, Mapping):
        raise PanNukePrimaryInputError(f"{location} must be a mapping")
    expected = {
        "schema_version",
        "matrix_version",
        "matrix_decision",
        "matrix_reason",
        "generator",
        "auditor",
        "independence_matrix_hash",
    }
    if set(raw) != expected or raw.get("schema_version") != 1:
        raise PanNukePrimaryInputError(f"{location} schema/fields are invalid")
    evidence = FeatureIndependenceEvidence(
        matrix_version=str(raw["matrix_version"]),
        matrix_decision=str(raw["matrix_decision"]),
        matrix_reason=str(raw["matrix_reason"]),
        generator=_parse_feature_space(raw["generator"], "independence generator"),
        auditor=_parse_feature_space(raw["auditor"], "independence auditor"),
        independence_matrix_hash=str(raw["independence_matrix_hash"]),
    )
    try:
        evidence.validate()
    except ValueError as exc:
        raise PanNukePrimaryInputError(f"invalid {location}: {exc}") from exc
    return evidence


def _load_independence_evidence_matrix(
    path: Path,
) -> dict[str, FeatureIndependenceEvidence]:
    payload = _load_json_mapping(path, "feature-space independence matrix")
    if set(payload) != {"schema_version", "entries"} or payload.get("schema_version") != 2:
        raise PanNukePrimaryInputError(
            "feature-space independence matrix must use strict schema_version=2"
        )
    entries = payload.get("entries")
    if not isinstance(entries, Mapping) or not entries:
        raise PanNukePrimaryInputError("independence matrix entries must be a non-empty mapping")
    result: dict[str, FeatureIndependenceEvidence] = {}
    for raw_identifier, raw_evidence in entries.items():
        identifier = str(raw_identifier)
        if not identifier.strip() or identifier in result:
            raise PanNukePrimaryInputError("independence matrix representation IDs are invalid")
        result[identifier] = _parse_independence_evidence(
            raw_evidence,
            location=f"independence matrix entry {identifier}",
        )
    return result


def _check_expected(actual: str, expected: str | None, field_name: str) -> None:
    if expected is not None and actual != expected:
        raise PanNukePrimaryInputError(f"{field_name} differs from its frozen expected SHA-256")


def _verify_primary_cache_provenance(
    representation: Mapping[str, Any],
    evidence_source: Mapping[str, Any],
    *,
    available: bool,
    actual_cache_sha256: str | None,
    sample_order_sha256: str,
    dataset_manifest_sha256: str,
) -> dict[str, Any]:
    """Bind one frozen representation to exact sidecar/audit semantic evidence."""

    identifier = str(representation["id"])
    frozen_raw = representation.get("cache_provenance")
    sidecar_raw = evidence_source.get("primary_cache_provenance")
    if not isinstance(frozen_raw, Mapping) or set(frozen_raw) != _PRIMARY_CACHE_PROVENANCE_FIELDS:
        raise PanNukePrimaryInputError(
            f"frozen cache provenance fields are invalid for {identifier}"
        )
    if not isinstance(sidecar_raw, Mapping) or set(sidecar_raw) != (
        _PRIMARY_CACHE_PROVENANCE_FIELDS
    ):
        raise PanNukePrimaryInputError(
            f"cache/audit sidecar lacks exact primary_cache_provenance for {identifier}"
        )
    frozen = dict(frozen_raw)
    sidecar = dict(sidecar_raw)
    if sidecar != frozen:
        raise PanNukePrimaryInputError(
            f"cache/audit sidecar provenance differs from frozen config for {identifier}"
        )
    expected_status = "available" if available else "unavailable_optional"
    if frozen["status"] != expected_status:
        raise PanNukePrimaryInputError(
            f"cache availability differs from frozen provenance for {identifier}"
        )
    if frozen["sample_order_sha256"] != sample_order_sha256:
        raise PanNukePrimaryInputError(
            f"sample order differs from frozen cache provenance for {identifier}"
        )
    if frozen["dataset_manifest_sha256"] != dataset_manifest_sha256:
        raise PanNukePrimaryInputError(
            f"dataset manifest differs from frozen cache provenance for {identifier}"
        )
    if available:
        if actual_cache_sha256 is None or frozen["cache_file_sha256"] != actual_cache_sha256:
            raise PanNukePrimaryInputError(
                f"cache file differs from frozen cache provenance for {identifier}"
            )
        if evidence_source.get("cache_npz_sha256") != actual_cache_sha256:
            raise PanNukePrimaryInputError(
                f"cache sidecar top-level checksum differs for {identifier}"
            )
    elif actual_cache_sha256 is not None or any(
        frozen[field] is not None
        for field in (
            "encoder_implementation_sha256",
            "weights_sha256",
            "preprocessing_sha256",
            "cache_file_sha256",
        )
    ):
        raise PanNukePrimaryInputError(
            f"unavailable optional provenance claims an artifact for {identifier}"
        )
    if (
        available
        and representation["family"] == "imagenet"
        and frozen["weights_sha256"] != evidence_source.get("weight_sha256")
    ):
        raise PanNukePrimaryInputError(
            f"ImageNet weight hash differs from frozen cache provenance for {identifier}"
        )
    return frozen


def _partition_features(
    features: Mapping[str, FeatureArray], indices: NDArray[np.int64]
) -> dict[str, NDArray[np.float64]]:
    # PrimaryMatrixInputs historically annotates float64, while immutable frozen
    # encoder caches are normally float32/float16.  Keeping their native dtype here
    # avoids doubling multi-gigabyte caches; the estimator core performs its explicit
    # numeric conversion per cell.
    return {
        identifier: cast(NDArray[np.float64], np.asarray(values[indices]))
        for identifier, values in features.items()
    }


def build_pannuke_primary_inputs(
    frozen_config: Mapping[str, Any],
    cache_paths: PanNukePrimaryCachePaths,
    *,
    expected_config_sha256: str,
    expected_plan_semantic_sha256: str,
    project_root: str | Path,
    expected_hashes: PanNukePrimaryHashExpectations | None = None,
) -> PanNukePrimaryInputsResult:
    """Validate immutable PanNuke caches and build strict primary-matrix inputs."""

    resolved = validate_frozen_primary_config(frozen_config)
    config_digest = config_sha256(resolved)
    _require_sha256(expected_config_sha256, "expected_config_sha256")
    if config_digest != expected_config_sha256:
        raise PanNukePrimaryInputError("frozen primary config semantic SHA-256 differs")
    plan = build_primary_matrix_plan(resolved)
    plan_digest = canonical_sha256(plan.as_dict())
    _require_sha256(expected_plan_semantic_sha256, "expected_plan_semantic_sha256")
    if plan_digest != expected_plan_semantic_sha256:
        raise PanNukePrimaryInputError("expanded primary plan semantic SHA-256 differs")
    if plan.config_sha256 != config_digest:
        raise PanNukePrimaryInputError("expanded plan is not bound to the frozen config")
    pilot_derived_parameters_sha256 = verify_pilot_derived_parameters_binding(
        resolved,
        project_root,
    )
    expectations = expected_hashes or PanNukePrimaryHashExpectations()
    expectations.validate()

    data = cast(Mapping[str, Any], resolved["data"])
    class_order = tuple(int(value) for value in cast(Sequence[Any], data["class_order"]))
    if str(data["group_unit"]) != "source_patch_id":
        raise PanNukePrimaryInputError(
            "current PanNuke crop cache proves source_patch_id grouping only"
        )
    crop = _load_crop_cache(cache_paths.crop_cache_path, class_order)
    eligibility_provenance = validate_analysis_eligibility_provenance(
        crop.metadata,
        crop.sample_ids,
    )
    engineered = _load_engineered_cache(cache_paths.engineered_cache_path, crop.sample_ids)
    sample_order_sha = canonical_sha256(crop.sample_ids.tolist())
    dataset_manifest_sha = str(crop.metadata["manifest_sha256"])
    if engineered.metadata["crop_manifest_sha256"] != crop.metadata["manifest_sha256"]:
        raise PanNukePrimaryInputError("engineered and crop caches name different manifests")
    crop_projection = crop.metadata.get("target_mask_projection")
    if not isinstance(crop_projection, Mapping):  # already checked in the crop loader
        raise PanNukePrimaryInputError("crop target-mask projection provenance is absent")
    if engineered.metadata.get("target_mask_projection_sha256") != canonical_sha256(
        dict(crop_projection)
    ) or engineered.metadata.get("target_mask_projection_semantic_sha256") != crop_projection.get(
        "semantic_sha256"
    ):
        raise PanNukePrimaryInputError(
            "engineered cache is not bound to the crop target-mask projection"
        )
    expected_engineered_crop_binding = {
        "schema_version": 1,
        "binding_type": "pannuke_component_covering_crop_cache_engineered_v1",
        "crop_cache_file_sha256": crop.cache_sha256,
        "crop_cache_sidecar_file_sha256": crop.sidecar_sha256,
        "crop_cache_content_sha256": crop.cache_content_sha256,
        "crop_manifest_sha256": str(crop.metadata["manifest_sha256"]),
        "raw_inventory_sha256": str(crop.metadata["raw_inventory_sha256"]),
        "sample_order_sha256": canonical_sha256(crop.sample_ids.tolist()),
        "target_mask_projection_semantic_sha256": str(crop_projection.get("semantic_sha256")),
        "input_variant": "context_rgb_plus_component_covering_target_masks",
        "input_array_sha256_by_name": {
            "context_rgb": crop.array_sha256_by_name["context_rgb"],
            "target_masks": crop.array_sha256_by_name["target_masks"],
        },
    }
    if engineered.metadata.get(
        "source_crop_cache_binding"
    ) != expected_engineered_crop_binding or engineered.metadata.get(
        "source_crop_cache_binding_sha256"
    ) != canonical_sha256(expected_engineered_crop_binding):
        raise PanNukePrimaryInputError(
            "engineered cache is not bound to the exact crop cache content"
        )

    verified_hashes: dict[str, str] = {
        "config_semantic_sha256": config_digest,
        "plan_semantic_sha256": plan_digest,
        "pilot_derived_parameters_sha256": pilot_derived_parameters_sha256,
        "crop_cache_sha256": crop.cache_sha256,
        "engineered_cache_sha256": engineered.cache_sha256,
        "dataset_manifest_sha256": str(crop.metadata["manifest_sha256"]),
        "raw_inventory_sha256": str(crop.metadata["raw_inventory_sha256"]),
        "analysis_eligibility_semantic_sha256": str(eligibility_provenance["semantic_sha256"]),
    }
    _check_expected(crop.cache_sha256, expectations.crop_cache_sha256, "crop cache")
    _check_expected(
        engineered.cache_sha256,
        expectations.engineered_cache_sha256,
        "engineered cache",
    )
    _check_expected(
        str(crop.metadata["manifest_sha256"]),
        expectations.dataset_manifest_sha256,
        "dataset manifest",
    )
    _check_expected(
        str(crop.metadata["raw_inventory_sha256"]),
        expectations.raw_inventory_sha256,
        "raw inventory",
    )
    if cache_paths.dataset_manifest_path is not None:
        manifest = _resolved_file(cache_paths.dataset_manifest_path, "dataset manifest")
        manifest_sha = sha256_file(manifest)
        if manifest_sha != crop.metadata["manifest_sha256"]:
            raise PanNukePrimaryInputError("dataset manifest file differs from crop provenance")
        verified_hashes["dataset_manifest_file_sha256"] = manifest_sha
    if (cache_paths.dataset_evidence_path is None) != (
        expectations.dataset_evidence_sha256 is None
    ):
        raise PanNukePrimaryInputError(
            "dataset_evidence_path and dataset_evidence_sha256 must be supplied together"
        )
    if cache_paths.dataset_evidence_path is not None:
        dataset_evidence = _resolved_file(cache_paths.dataset_evidence_path, "dataset evidence")
        dataset_evidence_sha = sha256_file(dataset_evidence)
        _check_expected(
            dataset_evidence_sha,
            expectations.dataset_evidence_sha256,
            "dataset evidence",
        )
        verified_hashes["dataset_evidence_sha256"] = dataset_evidence_sha
    if (cache_paths.freeze_record_path is None) != (expectations.freeze_record_sha256 is None):
        raise PanNukePrimaryInputError(
            "freeze_record_path and freeze_record_sha256 must be supplied together"
        )
    if cache_paths.freeze_record_path is not None:
        freeze = _resolved_file(cache_paths.freeze_record_path, "freeze record")
        freeze_sha = sha256_file(freeze)
        _check_expected(freeze_sha, expectations.freeze_record_sha256, "freeze record")
        freeze_payload = _load_json_mapping(freeze, "freeze record")
        primary_record = freeze_payload.get("primary_config")
        dataset_record = freeze_payload.get("dataset")
        manifest_record = freeze_payload.get("manifest")
        if freeze_payload.get("schema_version") != BASE_FREEZE_EVIDENCE_SCHEMA_VERSION:
            raise PanNukePrimaryInputError(
                f"freeze record schema_version must be {BASE_FREEZE_EVIDENCE_SCHEMA_VERSION}"
            )
        if (
            freeze_payload.get("completion_stage_enabled") != "PRE_REGISTRATION_FROZEN"
            or not isinstance(primary_record, Mapping)
            or primary_record.get("semantic_sha256") != config_digest
            or (
                expectations.dataset_evidence_sha256 is not None
                and (
                    not isinstance(dataset_record, Mapping)
                    or dataset_record.get("sha256") != expectations.dataset_evidence_sha256
                )
            )
            or not isinstance(manifest_record, Mapping)
            or manifest_record.get("sha256") != crop.metadata["manifest_sha256"]
        ):
            raise PanNukePrimaryInputError(
                "freeze record is not bound to this frozen primary config and manifest"
            )
        verified_hashes["freeze_record_sha256"] = freeze_sha

    development_folds = tuple(
        int(value) for value in cast(Sequence[Any], data["development_official_folds"])
    )
    final_fold = int(data["final_test_fold"])
    expected_folds = set(development_folds) | {final_fold}
    actual_folds = set(int(value) for value in np.unique(crop.official_folds))
    if actual_folds != expected_folds:
        raise PanNukePrimaryInputError(
            f"crop official folds {sorted(actual_folds)} differ from frozen folds "
            f"{sorted(expected_folds)}"
        )
    for group in np.unique(crop.group_ids):
        if len(np.unique(crop.official_folds[crop.group_ids == group])) != 1:
            raise PanNukePrimaryInputError(f"source group {group!s} spans official folds")
    final_mask = crop.official_folds == final_fold
    development_mask = np.isin(crop.official_folds, development_folds)
    if not bool(final_mask.any()) or not bool(development_mask.any()):
        raise PanNukePrimaryInputError("frozen final/development partitions must be non-empty")
    selected_validation_groups = select_stratified_reference_validation_groups(
        crop.pre_corruption_labels[development_mask],
        crop.group_ids[development_mask],
        class_order=class_order,
        fraction=float(data["reference_validation_fraction_groups"]),
        seed=int(data["split_seed"]),
    )
    validation_mask = development_mask & np.isin(crop.group_ids, selected_validation_groups)
    audit_mask = development_mask & ~validation_mask
    indices = np.arange(len(crop.sample_ids), dtype=np.int64)
    audit_indices = indices[audit_mask]
    validation_indices = indices[validation_mask]
    final_indices = indices[final_mask]
    if not len(audit_indices) or not len(validation_indices) or not len(final_indices):
        raise PanNukePrimaryInputError("audit/reference/final partitions must all be non-empty")
    if set(crop.pre_corruption_labels[audit_indices].tolist()) != set(class_order):
        raise PanNukePrimaryInputError(
            "audit pool must retain every frozen class for controlled corruption"
        )
    frozen_oof = cast(Mapping[str, Any], resolved["oof"])
    if len(np.unique(crop.group_ids[audit_indices])) < int(frozen_oof["n_splits"]):
        raise PanNukePrimaryInputError(
            "audit pool has fewer source groups than the frozen OOF fold count"
        )

    context_cache: tuple[EmbeddingResult, str] | None = None
    highlighted_cache: tuple[EmbeddingResult, str] | None = None
    pathology_cache: _FeatureCache | None = None
    raw_representations = cast(Sequence[Mapping[str, Any]], resolved["representations"])
    needs_context = any(
        item["family"] == "imagenet" and item["input_variant"] == "context_rgb"
        for item in raw_representations
    )
    needs_highlighted = any(
        item["family"] == "imagenet" and item["input_variant"] == "target_highlighted_rgb"
        for item in raw_representations
    )
    if needs_context:
        if cache_paths.context_embedding_cache_path is None:
            raise PanNukePrimaryInputError("required ImageNet context cache is unavailable")
        context_cache = _verified_imagenet_cache(
            cache_paths.context_embedding_cache_path,
            expected_ids=crop.sample_ids,
            expected_variant="rgb",
            expected_crop=crop,
        )
        verified_hashes["context_embedding_cache_sha256"] = context_cache[1]
        _check_expected(
            context_cache[1],
            expectations.context_embedding_cache_sha256,
            "context embedding cache",
        )
    if needs_highlighted:
        if cache_paths.highlighted_embedding_cache_path is None:
            raise PanNukePrimaryInputError("required ImageNet highlighted cache is unavailable")
        highlighted_cache = _verified_imagenet_cache(
            cache_paths.highlighted_embedding_cache_path,
            expected_ids=crop.sample_ids,
            expected_variant="target_highlighted_rgb",
            expected_crop=crop,
        )
        verified_hashes["highlighted_embedding_cache_sha256"] = highlighted_cache[1]
        _check_expected(
            highlighted_cache[1],
            expectations.highlighted_embedding_cache_sha256,
            "highlighted embedding cache",
        )
    if (
        context_cache is not None
        and highlighted_cache is not None
        and context_cache[0].metadata["weight_sha256"]
        != highlighted_cache[0].metadata["weight_sha256"]
    ):
        raise PanNukePrimaryInputError("ImageNet variants use different frozen weights")

    pathology_specs = [item for item in raw_representations if item["family"] == "pathology"]
    pathology_audit: Mapping[str, Any] | None = None
    if not pathology_specs and (
        cache_paths.pathology_embedding_cache_path is not None
        or cache_paths.pathology_availability_audit_path is not None
        or expectations.pathology_embedding_cache_sha256 is not None
    ):
        raise PanNukePrimaryInputError(
            "pathology cache/audit evidence was supplied without a frozen pathology representation"
        )
    if pathology_specs:
        frozen_audit_hashes = {str(item["availability_audit_sha256"]) for item in pathology_specs}
        if len(frozen_audit_hashes) != 1:
            raise PanNukePrimaryInputError("pathology representations name different audits")
        frozen_audit_hash = next(iter(frozen_audit_hashes))
        if cache_paths.pathology_availability_audit_path is None:
            raise PanNukePrimaryInputError("frozen pathology availability audit is unavailable")
        audit_path = _resolved_file(
            cache_paths.pathology_availability_audit_path,
            "pathology availability audit",
        )
        if sha256_file(audit_path) != frozen_audit_hash:
            raise PanNukePrimaryInputError("pathology availability audit differs from frozen SHA")
        pathology_audit = _load_json_mapping(audit_path, "pathology availability audit")
        verified_hashes["pathology_availability_audit_sha256"] = frozen_audit_hash
        if cache_paths.pathology_embedding_cache_path is not None:
            if pathology_audit.get("status") != "available":
                raise PanNukePrimaryInputError(
                    "a pathology cache cannot be used when the frozen audit is blocked"
                )
            pathology_cache = _load_generic_pathology_cache(
                cache_paths.pathology_embedding_cache_path,
                crop.sample_ids,
            )
            if pathology_cache.metadata.get("availability_audit_sha256") != frozen_audit_hash:
                raise PanNukePrimaryInputError(
                    "pathology cache is not bound to the frozen availability audit"
                )
            verified_hashes["pathology_embedding_cache_sha256"] = pathology_cache.cache_sha256
            _check_expected(
                pathology_cache.cache_sha256,
                expectations.pathology_embedding_cache_sha256,
                "pathology embedding cache",
            )
        elif pathology_audit.get("status") != "blocked":
            raise PanNukePrimaryInputError(
                "missing pathology cache requires a frozen audit with status=blocked"
            )
        elif expectations.pathology_embedding_cache_sha256 is not None:
            raise PanNukePrimaryInputError(
                "an expected pathology cache SHA was supplied but the optional cache is unavailable"
            )

    all_features: dict[str, FeatureArray] = {}
    availability: list[RepresentationAvailability] = []
    provenance_by_representation: dict[str, Mapping[str, Any]] = {}
    for representation in raw_representations:
        identifier = str(representation["id"])
        family = str(representation["family"])
        variant = str(representation["input_variant"])
        required = bool(representation["required"])
        values: FeatureArray | None
        checksum: str | None
        provenance_source: Mapping[str, Any]
        blocker: str | None = None
        if family == "engineered" and variant == "context_rgb":
            values, checksum = engineered.values, engineered.cache_sha256
            provenance_source = engineered.metadata
        elif family == "imagenet" and variant == "context_rgb" and context_cache is not None:
            values = cast(FeatureArray, context_cache[0].embeddings)
            checksum = context_cache[1]
            provenance_source = context_cache[0].metadata
        elif (
            family == "imagenet"
            and variant == "target_highlighted_rgb"
            and highlighted_cache is not None
        ):
            values = cast(FeatureArray, highlighted_cache[0].embeddings)
            checksum = highlighted_cache[1]
            provenance_source = highlighted_cache[0].metadata
        elif family == "pathology" and variant == "context_rgb" and pathology_cache is not None:
            values, checksum = pathology_cache.values, pathology_cache.cache_sha256
            provenance_source = pathology_cache.metadata
        elif family == "pathology" and not required:
            values, checksum = None, None
            if pathology_audit is None:
                raise PanNukePrimaryInputError(
                    "optional pathology provenance requires its frozen availability audit"
                )
            provenance_source = pathology_audit
            audit_blocker = pathology_audit.get("blocker") if pathology_audit is not None else None
            blocker = str(audit_blocker or "frozen optional pathology cache is unavailable")
        else:
            raise PanNukePrimaryInputError(
                f"unsupported or unavailable required representation {identifier}: "
                f"family={family}, input_variant={variant}"
            )
        if values is not None:
            try:
                feature_eligibility = validate_analysis_eligibility_provenance(
                    provenance_source,
                    crop.sample_ids,
                )
            except ValueError as exc:
                raise PanNukePrimaryInputError(
                    f"{identifier} analysis eligibility is invalid: {exc}"
                ) from exc
            if feature_eligibility["semantic_sha256"] != eligibility_provenance["semantic_sha256"]:
                raise PanNukePrimaryInputError(
                    f"{identifier} eligibility provenance differs from the crop cache"
                )
        provenance = _verify_primary_cache_provenance(
            representation,
            provenance_source,
            available=values is not None,
            actual_cache_sha256=checksum,
            sample_order_sha256=sample_order_sha,
            dataset_manifest_sha256=dataset_manifest_sha,
        )
        provenance_by_representation[identifier] = provenance
        verified_hashes[f"{identifier}_cache_provenance_sha256"] = canonical_sha256(provenance)
        if values is not None:
            all_features[identifier] = values
            availability.append(
                RepresentationAvailability(
                    identifier, family, required, "available", None, checksum
                )
            )
        else:
            availability.append(
                RepresentationAvailability(
                    identifier,
                    family,
                    required,
                    "unavailable_optional",
                    blocker,
                    None,
                )
            )

    morphology_columns = tuple(
        index for index, name in enumerate(engineered.names or ()) if name.startswith("morphology.")
    )
    morphology_names = tuple((engineered.names or ())[index] for index in morphology_columns)
    if not morphology_columns or any(
        not name.startswith("morphology.") for name in morphology_names
    ):
        raise PanNukePrimaryInputError("morphology_only_v1 columns cannot be resolved exactly")
    morphology_full = np.asarray(engineered.values[:, morphology_columns], dtype=np.float64)
    morphology_audit = morphology_full[audit_indices]

    instance = cast(
        Mapping[str, Any],
        cast(Mapping[str, Any], resolved["corruption"])["mechanisms"][
            "instance_dependent_corruption"
        ],
    )
    evidence_path = (
        Path(project_root).resolve() / str(instance["independence_matrix_path"])
    ).resolve()
    evidence_path = _resolved_file(evidence_path, "frozen feature independence matrix")
    evidence_file_sha = sha256_file(evidence_path)
    if evidence_file_sha != str(instance["independence_matrix_sha256"]):
        raise PanNukePrimaryInputError("independence matrix file differs from frozen SHA")
    evidence_by_representation = _load_independence_evidence_matrix(evidence_path)
    generator_name = str(instance["generator_representation"])
    representation_by_id = {str(item["id"]): item for item in raw_representations}
    allowed_families = {
        str(value) for value in cast(Sequence[Any], instance["auditor_representation_families"])
    }
    if set(evidence_by_representation) != set(all_features):
        missing = sorted(set(all_features).difference(evidence_by_representation))
        unexpected = sorted(set(evidence_by_representation).difference(all_features))
        raise PanNukePrimaryInputError(
            "independence matrix must contain exactly every available frozen representation "
            f"(missing={missing}, unexpected={unexpected})"
        )
    generator_hash = array_artifact_sha256(morphology_audit)
    audit_fitted_data_hash = canonical_sha256(
        {
            "sample_ids": crop.sample_ids[audit_indices].tolist(),
            "group_ids": crop.group_ids[audit_indices].tolist(),
        }
    )
    canonical_generator = next(iter(evidence_by_representation.values())).generator
    engineered_generator_provenance = provenance_by_representation.get("engineered_target_features")
    if engineered_generator_provenance is None:
        raise PanNukePrimaryInputError(
            "morphology generator lacks engineered-cache semantic provenance"
        )
    for auditor_id, evidence in evidence_by_representation.items():
        if evidence.generator != canonical_generator:
            raise PanNukePrimaryInputError(
                "all independence entries must bind the same concrete generator space"
            )
        if evidence.generator.representation_name != generator_name:
            raise PanNukePrimaryInputError(
                f"independence evidence for {auditor_id} names a different generator"
            )
        if evidence.generator.family != "morphology":
            raise PanNukePrimaryInputError("generator evidence family must be exactly morphology")
        if (
            evidence.generator.implementation_hash
            != engineered_generator_provenance["encoder_implementation_sha256"]
            or evidence.generator.weights_hash != engineered_generator_provenance["weights_sha256"]
            or evidence.generator.preprocessing_hash
            != engineered_generator_provenance["preprocessing_sha256"]
        ):
            raise PanNukePrimaryInputError(
                "morphology generator semantics differ from engineered-cache provenance"
            )
        if evidence.generator.feature_artifact_hash != generator_hash:
            raise PanNukePrimaryInputError(
                "morphology-only audit features differ from generator evidence hash"
            )
        if evidence.generator.fitted_data_hash != audit_fitted_data_hash:
            raise PanNukePrimaryInputError(
                "morphology generator evidence names a different audit sample/group set"
            )
        auditor_spec = representation_by_id[auditor_id]
        auditor_family = str(auditor_spec["family"])
        auditor_provenance = provenance_by_representation[auditor_id]
        frozen_independence = cast(Mapping[str, Any], auditor_spec["generator_independence"])
        if auditor_family == "engineered" and evidence.matrix_decision != "not_independent":
            raise PanNukePrimaryInputError(
                "engineered auditor overlaps morphology_only_v1 and must remain circularity_risk"
            )
        expected_independence_status = (
            "circularity_risk"
            if evidence.matrix_decision == "not_independent"
            else "verified_independent"
        )
        if (
            frozen_independence["status"] != expected_independence_status
            or frozen_independence["independence_matrix_sha256"] != evidence_file_sha
        ):
            raise PanNukePrimaryInputError(
                f"frozen generator-independence binding differs for {auditor_id}; "
                f"expected status={expected_independence_status} and the exact matrix SHA"
            )
        if auditor_family not in allowed_families:
            raise PanNukePrimaryInputError(
                f"independence auditor family for {auditor_id} is not frozen as eligible"
            )
        if evidence.auditor.representation_name != auditor_id:
            raise PanNukePrimaryInputError(
                f"independence matrix key/auditor identity differ for {auditor_id}"
            )
        if evidence.auditor.family != auditor_family:
            raise PanNukePrimaryInputError(
                f"auditor evidence family differs from frozen representation {auditor_id}"
            )
        if (
            evidence.auditor.implementation_hash
            != auditor_provenance["encoder_implementation_sha256"]
            or evidence.auditor.weights_hash != auditor_provenance["weights_sha256"]
            or evidence.auditor.preprocessing_hash != auditor_provenance["preprocessing_sha256"]
            or evidence.auditor.fitted_data_hash != audit_fitted_data_hash
        ):
            raise PanNukePrimaryInputError(
                f"auditor semantic provenance differs from its cache/evidence for {auditor_id}"
            )
        if auditor_family == "imagenet":
            input_variant = str(auditor_spec["input_variant"])
            embedding_result = (
                context_cache[0]
                if input_variant == "context_rgb" and context_cache is not None
                else highlighted_cache[0]
                if input_variant == "target_highlighted_rgb" and highlighted_cache is not None
                else None
            )
            if embedding_result is None or evidence.auditor.weights_hash != str(
                embedding_result.metadata["weight_sha256"]
            ):
                raise PanNukePrimaryInputError(
                    f"auditor weight hash differs from the cache for {auditor_id}"
                )
        auditor_audit = np.asarray(all_features[auditor_id][audit_indices])
        if evidence.auditor.feature_artifact_hash != array_artifact_sha256(auditor_audit):
            raise PanNukePrimaryInputError(
                f"auditor cache differs from independence evidence hash for {auditor_id}"
            )
        if auditor_family == "engineered":
            if evidence.matrix_decision != "not_independent":
                raise PanNukePrimaryInputError(
                    "engineered auditor contains the morphology generator columns and must "
                    "be frozen as not_independent/circularity_risk"
                )
        elif evidence.matrix_decision != "verified_independent":
            raise PanNukePrimaryInputError(
                f"{auditor_id} lacks verified-independent hash-bound evidence"
            )
    verified_hashes["independence_matrix_file_sha256"] = evidence_file_sha
    verified_hashes["independence_matrix_entries_sha256"] = canonical_sha256(
        {
            identifier: evidence.independence_matrix_hash
            for identifier, evidence in sorted(evidence_by_representation.items())
        }
    )

    audit_ids = tuple(str(value) for value in crop.sample_ids[audit_indices])
    validation_ids = tuple(str(value) for value in crop.sample_ids[validation_indices])
    final_ids = tuple(str(value) for value in crop.sample_ids[final_indices])
    group_conditional = cast(
        Mapping[str, Any],
        cast(Mapping[str, Any], resolved["corruption"])["mechanisms"][
            "group_conditional_corruption"
        ],
    )
    grouping_field = str(group_conditional["grouping_field"])
    if grouping_field == "tissue_type":
        for group in np.unique(crop.group_ids[audit_indices]):
            members = audit_indices[crop.group_ids[audit_indices] == group]
            if len(np.unique(crop.tissue_types[members])) != 1:
                raise PanNukePrimaryInputError(
                    f"source group {group!s} spans multiple tissue_type values"
                )
        grouping_values = tuple(str(value) for value in crop.tissue_types[audit_indices])
    elif grouping_field in {"source_patch_id", "group_id"}:
        grouping_values = tuple(str(value) for value in crop.group_ids[audit_indices])
    else:
        raise PanNukePrimaryInputError(
            f"crop cache cannot prove frozen group-conditional field {grouping_field!r}"
        )
    inputs = PrimaryMatrixInputs(
        audit_sample_ids=audit_ids,
        audit_group_ids=tuple(str(value) for value in crop.group_ids[audit_indices]),
        audit_pre_corruption_labels=crop.pre_corruption_labels[audit_indices],
        audit_features=_partition_features(all_features, audit_indices),
        reference_validation_sample_ids=validation_ids,
        reference_validation_group_ids=tuple(
            str(value) for value in crop.group_ids[validation_indices]
        ),
        reference_validation_labels=crop.pre_corruption_labels[validation_indices],
        reference_validation_features=_partition_features(all_features, validation_indices),
        final_test_sample_ids=final_ids,
        final_test_group_ids=tuple(str(value) for value in crop.group_ids[final_indices]),
        final_test_labels=crop.pre_corruption_labels[final_indices],
        final_test_features=_partition_features(all_features, final_indices),
        corruption_generator_features=morphology_audit,
        corruption_generator_representation=generator_name,
        corruption_auditor_representation=None,
        independence_evidence=None,
        dataset_seed=None,
        class_order=class_order,
        audit_grouping_values={grouping_field: grouping_values},
        independence_evidence_by_representation=evidence_by_representation,
        independence_matrix_artifact_sha256_by_representation={
            identifier: evidence_file_sha for identifier in evidence_by_representation
        },
        corruption_auditor_family_by_representation={
            identifier: str(representation_by_id[identifier]["family"])
            for identifier in evidence_by_representation
        },
    )
    try:
        inputs.validate()
    except ValueError as exc:
        raise PanNukePrimaryInputError(f"constructed primary inputs are invalid: {exc}") from exc
    partition_sha = canonical_sha256(
        {
            "assignment_label_source": "pre_corruption_label",
            "split_algorithm": "deterministic_group_greedy_class_distribution_v1",
            "split_seed": int(data["split_seed"]),
            "audit_sample_ids": audit_ids,
            "reference_validation_sample_ids": validation_ids,
            "final_test_sample_ids": final_ids,
        }
    )
    return PanNukePrimaryInputsResult(
        inputs=inputs,
        plan=plan,
        config_sha256=config_digest,
        plan_semantic_sha256=plan_digest,
        sample_order_sha256=sample_order_sha,
        partition_assignment_sha256=partition_sha,
        selected_reference_validation_groups=selected_validation_groups,
        morphology_feature_names=morphology_names,
        cache_provenance_by_representation=provenance_by_representation,
        independence_evidence_by_representation=evidence_by_representation,
        representation_availability=tuple(availability),
        verified_hashes=verified_hashes,
        eligibility_provenance=eligibility_provenance,
    )


__all__ = [
    "PanNukePrimaryCachePaths",
    "PanNukePrimaryHashExpectations",
    "PanNukePrimaryInputError",
    "PanNukePrimaryInputsResult",
    "RepresentationAvailability",
    "build_pannuke_primary_inputs",
    "select_stratified_reference_validation_groups",
]
