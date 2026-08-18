"""Streaming, immutable nucleus-level manifest construction for PanNuke."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from scipy import ndimage  # type: ignore[import-untyped]
from skimage.measure import perimeter as measure_perimeter

from .discovery import discover_pannuke_release
from .exceptions import PanNukeSemanticsError
from .io import ensure_output_capacity, open_npy_mmap, sha256_file
from .models import (
    SOURCE_PATCH_INDEPENDENCE_STATEMENT,
    ManifestArtifacts,
    MaskInstanceQC,
    PanNukeValidationResult,
    PatchMaskQC,
    ValidationArtifacts,
    VerifiedClassMapping,
)
from .publication import (
    ExclusiveBundlePublicationLock,
    PublishedPath,
    publish_file_no_overwrite,
    rollback_owned_publications,
)
from .validation import (
    resolve_class_mapping,
    validate_discovered_release,
    verify_raw_inventory_unchanged,
)

OVERLAP_EXCLUSION_REASON = "touches_cross_class_overlap"
PATCH_OVERLAP_FLAG = "patch_contains_cross_class_overlap"
PATCH_VOID_FLAG = "patch_contains_unlabeled_void"
PATCH_POSITIVE_BACKGROUND_FLAG = "patch_contains_positive_and_background"

MANIFEST_REQUIRED_COLUMNS = (
    "sample_id",
    "official_fold",
    "patch_id",
    "group_id",
    "grouping_unit",
    "patient_id",
    "wsi_id",
    "tissue_type",
    "source_image_path",
    "source_mask_path",
    "source_tissue_path",
    "source_patch_index",
    "patch_index",
    "nucleus_index_in_patch",
    "nucleus_class_index",
    "nucleus_class_name",
    "instance_channel_index",
    "instance_id",
    "raw_instance_identity",
    "bbox",
    "bbox_x_min",
    "bbox_y_min",
    "bbox_x_max",
    "bbox_y_max",
    "centroid",
    "centroid_x",
    "centroid_y",
    "area",
    "perimeter",
    "border_touch",
    "quality_flags",
    "patch_height",
    "patch_width",
    "patch_pixel_count",
    "patch_has_overlap",
    "patch_has_void",
    "patch_has_positive_and_background",
    "patch_overlap_pixel_count",
    "patch_overlap_pixel_rate",
    "patch_positive_occupied_pixel_count",
    "patch_positive_occupied_pixel_rate",
    "patch_has_supplied_background_channel",
    "patch_supplied_background_channel_index",
    "patch_supplied_background_pixel_count",
    "patch_supplied_background_pixel_rate",
    "patch_void_pixel_count",
    "patch_void_pixel_rate",
    "patch_positive_background_conflict_pixel_count",
    "patch_positive_background_conflict_pixel_rate",
    "patch_anomaly_union_pixel_count",
    "patch_anomaly_union_pixel_rate",
    "patch_qc_affected_instance_count",
    "cross_class_overlap_touching",
    "overlap_pixel_count_for_instance",
    "overlap_class_channel_indices",
    "overlap_instance_channel_indices",
    "overlap_instance_ids",
    "overlap_instance_pixel_counts",
    "positive_background_touching",
    "positive_background_pixel_count_for_instance",
    "qc_exclusion_reason",
    "primary_eligible",
    "confirmatory_eligible",
    "crop_generated",
    "crop_path",
    "pre_corruption_label",
    "observed_label",
    "is_injected_corruption",
    "corruption_type",
    "original_class",
    "replacement_class",
    "corruption_seed",
    "corruption_rate",
    "corruption_representation",
    "auditor_representation",
    "feature_space_independent",
    "circularity_risk",
    "configuration_hash",
    "corruption_timestamp_utc",
)


def _manifest_configuration_hash(validation: PanNukeValidationResult) -> str:
    """Bind every source-manifest row to the exact validated release and policy."""

    payload = {
        "schema_version": 2,
        "dataset": "PanNuke",
        "raw_file_inventory": [item.as_dict() for item in validation.inventory],
        "class_mapping": validation.mapping.as_dict(),
        "mask_qc_policy": validation.qc_policy.as_dict(),
        "global_mask_qc": validation.global_mask_qc.as_dict(),
        "fold_ids": [fold.fold_id for fold in validation.fold_validation],
        "grouping_unit": validation.grouping_unit,
        "patient_id_available": validation.patient_id_available,
        "wsi_id_available": validation.wsi_id_available,
        "source_annotations_modified": False,
        "crop_generated": False,
        "corruption_type": "none",
        "cross_class_overlap_policy": (
            "preserve_raw_channel_instance_identity;exclude_every_touching_instance;"
            "never_resolve_overlap_pixels"
        ),
        "cross_class_overlap_exclusion_reason": OVERLAP_EXCLUSION_REASON,
        "eligibility_policy": "one_identical_primary_and_confirmatory_instance_mask",
        "void_pixel_policy": "retain_as_unlabeled_void; never_assign_background_or_positive_class",
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _raw_inventory_payload(validation: PanNukeValidationResult) -> bytes:
    return json.dumps(
        [item.as_dict() for item in validation.inventory],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _manifest_schema(validation: PanNukeValidationResult) -> pa.Schema:
    raw_inventory = _raw_inventory_payload(validation)
    schema = pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("official_fold", pa.int16(), nullable=False),
            pa.field("patch_id", pa.string(), nullable=False),
            pa.field("group_id", pa.string(), nullable=False),
            pa.field("grouping_unit", pa.string(), nullable=False),
            pa.field("patient_id", pa.string()),
            pa.field("wsi_id", pa.string()),
            pa.field("tissue_type", pa.string(), nullable=False),
            pa.field("source_image_path", pa.string(), nullable=False),
            pa.field("source_mask_path", pa.string(), nullable=False),
            pa.field("source_tissue_path", pa.string(), nullable=False),
            pa.field("source_patch_index", pa.int32(), nullable=False),
            pa.field("patch_index", pa.int32(), nullable=False),
            pa.field("nucleus_index_in_patch", pa.int32(), nullable=False),
            pa.field("nucleus_class_index", pa.int16(), nullable=False),
            pa.field("nucleus_class_name", pa.string(), nullable=False),
            pa.field("instance_channel_index", pa.int16(), nullable=False),
            pa.field("instance_id", pa.int64(), nullable=False),
            pa.field("raw_instance_identity", pa.string(), nullable=False),
            pa.field("bbox", pa.list_(pa.int32(), 4), nullable=False),
            pa.field("bbox_x_min", pa.int32(), nullable=False),
            pa.field("bbox_y_min", pa.int32(), nullable=False),
            pa.field("bbox_x_max", pa.int32(), nullable=False),
            pa.field("bbox_y_max", pa.int32(), nullable=False),
            pa.field("centroid", pa.list_(pa.float64(), 2), nullable=False),
            pa.field("centroid_x", pa.float64(), nullable=False),
            pa.field("centroid_y", pa.float64(), nullable=False),
            pa.field("area", pa.int32(), nullable=False),
            pa.field("perimeter", pa.float64(), nullable=False),
            pa.field("border_touch", pa.bool_(), nullable=False),
            pa.field("quality_flags", pa.list_(pa.string()), nullable=False),
            pa.field("patch_height", pa.int32(), nullable=False),
            pa.field("patch_width", pa.int32(), nullable=False),
            pa.field("patch_pixel_count", pa.int64(), nullable=False),
            pa.field("patch_has_overlap", pa.bool_(), nullable=False),
            pa.field("patch_has_void", pa.bool_(), nullable=False),
            pa.field("patch_has_positive_and_background", pa.bool_(), nullable=False),
            pa.field("patch_overlap_pixel_count", pa.int64(), nullable=False),
            pa.field("patch_overlap_pixel_rate", pa.float64(), nullable=False),
            pa.field("patch_positive_occupied_pixel_count", pa.int64(), nullable=False),
            pa.field("patch_positive_occupied_pixel_rate", pa.float64(), nullable=False),
            pa.field("patch_has_supplied_background_channel", pa.bool_(), nullable=False),
            pa.field("patch_supplied_background_channel_index", pa.int16()),
            pa.field("patch_supplied_background_pixel_count", pa.int64(), nullable=False),
            pa.field("patch_supplied_background_pixel_rate", pa.float64(), nullable=False),
            pa.field("patch_void_pixel_count", pa.int64(), nullable=False),
            pa.field("patch_void_pixel_rate", pa.float64(), nullable=False),
            pa.field("patch_positive_background_conflict_pixel_count", pa.int64(), nullable=False),
            pa.field("patch_positive_background_conflict_pixel_rate", pa.float64(), nullable=False),
            pa.field("patch_anomaly_union_pixel_count", pa.int64(), nullable=False),
            pa.field("patch_anomaly_union_pixel_rate", pa.float64(), nullable=False),
            pa.field("patch_qc_affected_instance_count", pa.int64(), nullable=False),
            pa.field("cross_class_overlap_touching", pa.bool_(), nullable=False),
            pa.field("overlap_pixel_count_for_instance", pa.int64(), nullable=False),
            pa.field("overlap_class_channel_indices", pa.list_(pa.int16()), nullable=False),
            pa.field("overlap_instance_channel_indices", pa.list_(pa.int16()), nullable=False),
            pa.field("overlap_instance_ids", pa.list_(pa.int64()), nullable=False),
            pa.field("overlap_instance_pixel_counts", pa.list_(pa.int64()), nullable=False),
            pa.field("positive_background_touching", pa.bool_(), nullable=False),
            pa.field("positive_background_pixel_count_for_instance", pa.int64(), nullable=False),
            pa.field("qc_exclusion_reason", pa.string()),
            pa.field("primary_eligible", pa.bool_(), nullable=False),
            pa.field("confirmatory_eligible", pa.bool_(), nullable=False),
            pa.field("crop_generated", pa.bool_(), nullable=False),
            pa.field("crop_path", pa.string()),
            pa.field("pre_corruption_label", pa.int16(), nullable=False),
            pa.field("observed_label", pa.int16(), nullable=False),
            pa.field("is_injected_corruption", pa.bool_(), nullable=False),
            pa.field("corruption_type", pa.string(), nullable=False),
            pa.field("original_class", pa.int16(), nullable=False),
            pa.field("replacement_class", pa.int16()),
            pa.field("corruption_seed", pa.int64()),
            pa.field("corruption_rate", pa.float64(), nullable=False),
            pa.field("corruption_representation", pa.string()),
            pa.field("auditor_representation", pa.string()),
            pa.field("feature_space_independent", pa.bool_()),
            pa.field("circularity_risk", pa.bool_(), nullable=False),
            pa.field("configuration_hash", pa.string(), nullable=False),
            pa.field("corruption_timestamp_utc", pa.string()),
        ]
    )
    metadata = {
        b"dataset": b"PanNuke",
        b"bbox_convention": b"half-open [x_min, y_min, x_max, y_max]",
        b"grouping_unit": b"source_patch",
        b"independence_statement": validation.independence_statement.encode("utf-8"),
        b"class_mapping": json.dumps(validation.mapping.as_dict(), sort_keys=True).encode("utf-8"),
        b"source_annotations_modified": b"false",
        b"raw_file_inventory": raw_inventory,
        b"raw_file_inventory_sha256": hashlib.sha256(raw_inventory).hexdigest().encode("ascii"),
        b"tiny_crops_generated": b"false",
        b"cross_class_overlap_policy": (
            b"preserve_raw_channel_instance_identity;exclude_every_touching_instance;"
            b"never_resolve_overlap_pixels"
        ),
        b"cross_class_overlap_exclusion_reason": OVERLAP_EXCLUSION_REASON.encode("ascii"),
        b"eligibility_policy": b"one_identical_primary_and_confirmatory_instance_mask",
        b"void_pixel_policy": b"retain_as_unlabeled_void;never_assign_background_or_positive_class",
        b"representation_cache_binding_policy": (
            b"downstream_cache_must_bind_manifest_sha256_and_crop_configuration"
        ),
        b"manifest_configuration_hash": _manifest_configuration_hash(validation).encode("ascii"),
    }
    return schema.with_metadata(metadata)


def _channel_last_patch(array: np.ndarray, index: int, channel_axis: int) -> np.ndarray:
    patch = np.asarray(array[index])
    patch_axis = channel_axis - 1
    return np.moveaxis(patch, patch_axis, -1) if patch_axis != patch.ndim - 1 else patch


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _validate_full_patch(
    mask: np.ndarray,
    *,
    fold_id: int,
    patch_index: int,
    positive_channels: tuple[int, ...],
    background_channel: int | None,
) -> None:
    if (
        mask.ndim != 3
        or not positive_channels
        or any(channel < 0 or channel >= mask.shape[-1] for channel in positive_channels)
    ):
        raise PanNukeSemanticsError(
            f"fold {fold_id} patch {patch_index}: positive-channel indices are invalid"
        )
    if not np.isfinite(mask).all() or np.any(mask < 0):
        raise PanNukeSemanticsError(
            f"fold {fold_id} patch {patch_index}: mask has non-finite or negative IDs"
        )
    if not np.equal(mask, np.rint(mask)).all():
        raise PanNukeSemanticsError(
            f"fold {fold_id} patch {patch_index}: mask IDs are not integer-like"
        )
    if background_channel is not None:
        if not 0 <= background_channel < mask.shape[-1]:
            raise PanNukeSemanticsError(
                f"fold {fold_id} patch {patch_index}: background-channel index is invalid"
            )
        background = mask[..., background_channel]
        if not set(np.unique(background).tolist()).issubset({0, 1}):
            raise PanNukeSemanticsError(
                f"fold {fold_id} patch {patch_index}: background is not binary"
            )


def _pixel_rate(count: int, pixel_count: int) -> float:
    if pixel_count <= 0:
        raise PanNukeSemanticsError("source mask patch must contain at least one pixel")
    return float(count / pixel_count)


def _patch_qc_lookup(validation: PanNukeValidationResult) -> dict[tuple[int, int], PatchMaskQC]:
    """Index the validator's complete per-patch QC evidence and reject gaps."""

    policy = validation.qc_policy
    if (
        policy.analysis_instance_exclusion_reason != OVERLAP_EXCLUSION_REASON
        or not policy.applies_identically_to_primary_and_confirmatory
        or not policy.no_class_arbitration
        or policy.source_masks_modified
        or policy.supplied_background_is_exact_complement_required
        or policy.release_annotation_anomalies_are_fatal
        or not policy.structural_invalidity_is_fatal
    ):
        raise PanNukeSemanticsError(
            "validated mask-QC policy differs from the fixed anomaly-safe manifest policy"
        )
    lookup: dict[tuple[int, int], PatchMaskQC] = {}
    expected_keys: set[tuple[int, int]] = set()
    for facts in validation.fold_validation:
        expected_keys.update((facts.fold_id, index) for index in range(facts.n_patches))
        fold_qc = facts.mask_qc
        if fold_qc.fold_id != facts.fold_id or fold_qc.patch_count != facts.n_patches:
            raise PanNukeSemanticsError(
                f"fold {facts.fold_id}: mask-QC fold identity/coverage is inconsistent"
            )
        if len(fold_qc.patches) != facts.n_patches:
            raise PanNukeSemanticsError(
                f"fold {facts.fold_id}: mask-QC does not contain every source patch"
            )
        for patch_qc in fold_qc.patches:
            key = (patch_qc.fold_id, patch_qc.patch_index)
            if key in lookup:
                raise PanNukeSemanticsError(f"duplicate mask-QC patch record: {key}")
            if (
                patch_qc.fold_id != facts.fold_id
                or not 0 <= patch_qc.patch_index < facts.n_patches
                or patch_qc.height != facts.height
                or patch_qc.width != facts.width
                or patch_qc.total_pixel_count != facts.height * facts.width
            ):
                raise PanNukeSemanticsError(
                    f"fold {facts.fold_id}: malformed mask-QC patch record {key}"
                )
            if patch_qc.affected_instance_count != len(patch_qc.affected_instances):
                raise PanNukeSemanticsError(
                    f"fold {facts.fold_id} patch {patch_qc.patch_index}: "
                    "mask-QC affected-instance count is inconsistent"
                )
            lookup[key] = patch_qc
    if set(lookup) != expected_keys:
        missing = sorted(expected_keys.difference(lookup))
        unexpected = sorted(set(lookup).difference(expected_keys))
        raise PanNukeSemanticsError(
            "mask-QC patch coverage differs from the validated release: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    return lookup


def _affected_instance_lookup(
    patch_qc: PatchMaskQC,
    *,
    mapping: VerifiedClassMapping,
    positive_channels: tuple[int, ...],
) -> dict[tuple[int, int], MaskInstanceQC]:
    """Index anomaly-affected raw instances without selecting a winning class."""

    lookup: dict[tuple[int, int], MaskInstanceQC] = {}
    for instance_qc in patch_qc.affected_instances:
        key = (instance_qc.channel_index, instance_qc.instance_id)
        if key in lookup:
            raise PanNukeSemanticsError(
                f"fold {patch_qc.fold_id} patch {patch_qc.patch_index}: duplicate "
                f"affected-instance QC identity {key}"
            )
        if (
            instance_qc.fold_id != patch_qc.fold_id
            or instance_qc.patch_index != patch_qc.patch_index
            or not 0 <= instance_qc.class_index < len(mapping.class_names)
            or instance_qc.class_name != mapping.class_names[instance_qc.class_index]
            or instance_qc.channel_index != positive_channels[instance_qc.class_index]
            or instance_qc.instance_id <= 0
            or instance_qc.total_pixel_count <= 0
            or not 0 <= instance_qc.overlap_pixel_count <= instance_qc.total_pixel_count
            or not 0 <= instance_qc.positive_background_pixel_count <= instance_qc.total_pixel_count
            or not (instance_qc.overlap_pixel_count or instance_qc.positive_background_pixel_count)
        ):
            raise PanNukeSemanticsError(
                f"fold {patch_qc.fold_id} patch {patch_qc.patch_index}: malformed "
                f"affected-instance QC record {key}"
            )
        lookup[key] = instance_qc
    return lookup


def _verify_patch_qc_against_mask(
    mask: np.ndarray,
    *,
    patch_qc: PatchMaskQC,
    positive_channels: tuple[int, ...],
    background_channel: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconcile saved QC counts with the unchanged source patch."""

    positive_occupancy = mask[..., list(positive_channels)] > 0
    positive_count = np.count_nonzero(positive_occupancy, axis=-1)
    positive_any = positive_count > 0
    overlap = positive_count > 1
    if background_channel is None:
        supplied_background = np.zeros(mask.shape[:2], dtype=bool)
    else:
        supplied_background = mask[..., background_channel] > 0
    void = ~positive_any & ~supplied_background
    positive_and_background = positive_any & supplied_background
    anomaly_union = overlap | void | positive_and_background
    actual = {
        "positive_any_pixel_count": int(np.count_nonzero(positive_any)),
        "background_pixel_count": int(np.count_nonzero(supplied_background)),
        "void_pixel_count": int(np.count_nonzero(void)),
        "cross_class_overlap_pixel_count": int(np.count_nonzero(overlap)),
        "positive_and_background_pixel_count": int(np.count_nonzero(positive_and_background)),
        "anomaly_union_pixel_count": int(np.count_nonzero(anomaly_union)),
    }
    for field, count in actual.items():
        if int(getattr(patch_qc, field)) != count:
            raise PanNukeSemanticsError(
                f"fold {patch_qc.fold_id} patch {patch_qc.patch_index}: saved mask-QC "
                f"{field} differs from the immutable raw mask"
            )
    expected_flags = (
        (patch_qc.has_void, actual["void_pixel_count"] > 0, "has_void"),
        (
            patch_qc.has_cross_class_overlap,
            actual["cross_class_overlap_pixel_count"] > 0,
            "has_cross_class_overlap",
        ),
        (
            patch_qc.has_positive_and_background,
            actual["positive_and_background_pixel_count"] > 0,
            "has_positive_and_background",
        ),
    )
    for observed, expected, field in expected_flags:
        if observed is not expected:
            raise PanNukeSemanticsError(
                f"fold {patch_qc.fold_id} patch {patch_qc.patch_index}: saved mask-QC "
                f"{field} differs from its exact count"
            )
    return overlap, positive_and_background


def _overlap_instance_evidence(
    mask: np.ndarray,
    *,
    target_overlap_mask: np.ndarray,
    positive_channels: tuple[int, ...],
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return deterministic raw channel/instance evidence without resolving overlap."""

    if not np.any(target_overlap_mask):
        return [], [], [], []
    evidence: list[tuple[int, int, int]] = []
    for channel_index in positive_channels:
        channel = mask[..., channel_index]
        values = channel[target_overlap_mask]
        values = values[values > 0]
        if values.size == 0:
            continue
        instance_ids, counts = np.unique(values.astype(np.int64, copy=False), return_counts=True)
        evidence.extend(
            (channel_index, int(instance_id), int(count))
            for instance_id, count in zip(instance_ids, counts, strict=True)
        )
    evidence.sort(key=lambda value: (value[0], value[1]))
    channels = [channel for channel, _, _ in evidence]
    return (
        sorted(set(channels)),
        channels,
        [instance_id for _, instance_id, _ in evidence],
        [count for _, _, count in evidence],
    )


def _iter_manifest_rows(
    validation: PanNukeValidationResult,
    *,
    summary_counts: dict[tuple[int, str, str], int],
    summary_patch_counts: dict[tuple[int, str, str], int],
) -> Iterator[dict[str, Any]]:
    root = validation.root
    configuration_hash = _manifest_configuration_hash(validation)
    facts_by_fold = {item.fold_id: item for item in validation.fold_validation}
    qc_by_patch = _patch_qc_lookup(validation)
    for fold in validation.folds:
        facts = facts_by_fold[fold.fold_id]
        masks = open_npy_mmap(fold.mask_path)
        tissues = open_npy_mmap(fold.tissue_path)
        source_image = _relative(fold.image_path, root)
        source_mask = _relative(fold.mask_path, root)
        source_tissue = _relative(fold.tissue_path, root)
        for patch_index in range(facts.n_patches):
            mask = _channel_last_patch(masks, patch_index, fold.mask_channel_axis)
            _validate_full_patch(
                mask,
                fold_id=fold.fold_id,
                patch_index=patch_index,
                positive_channels=facts.positive_channel_indices,
                background_channel=facts.background_channel_index,
            )
            patch_qc = qc_by_patch[(fold.fold_id, patch_index)]
            overlap_mask, positive_background_mask = _verify_patch_qc_against_mask(
                mask,
                patch_qc=patch_qc,
                positive_channels=facts.positive_channel_indices,
                background_channel=facts.background_channel_index,
            )
            affected_instances = _affected_instance_lookup(
                patch_qc,
                mapping=validation.mapping,
                positive_channels=facts.positive_channel_indices,
            )
            tissue = str(tissues[patch_index])
            patch_id = f"pannuke-fold-{fold.fold_id}-patch-{patch_index:06d}"
            nucleus_index = 0
            classes_seen: set[str] = set()
            for class_index, (class_name, channel_index) in enumerate(
                zip(
                    validation.mapping.class_names,
                    facts.positive_channel_indices,
                    strict=True,
                )
            ):
                channel = mask[..., channel_index]
                instance_ids = np.unique(channel)
                instance_ids = instance_ids[instance_ids > 0].astype(np.int64, copy=False)
                for instance_id_value in instance_ids:
                    instance_id = int(instance_id_value)
                    binary = channel == instance_id_value
                    y_values, x_values = np.nonzero(binary)
                    if not len(x_values):
                        continue
                    x_min = int(x_values.min())
                    y_min = int(y_values.min())
                    x_max = int(x_values.max()) + 1
                    y_max = int(y_values.max()) + 1
                    area = int(binary.sum())
                    centroid_x = float(x_values.mean())
                    centroid_y = float(y_values.mean())
                    _, component_count = ndimage.label(binary)
                    flags: list[str] = []
                    if component_count != 1:
                        flags.append("disconnected_instance_id")
                    border_touch = bool(
                        x_min == 0 or y_min == 0 or x_max == facts.width or y_max == facts.height
                    )
                    if border_touch:
                        flags.append("touches_source_patch_border")
                    if patch_qc.has_cross_class_overlap:
                        flags.append(PATCH_OVERLAP_FLAG)
                    if patch_qc.has_void:
                        flags.append(PATCH_VOID_FLAG)
                    if patch_qc.has_positive_and_background:
                        flags.append(PATCH_POSITIVE_BACKGROUND_FLAG)
                    instance_qc = affected_instances.get((channel_index, instance_id))
                    raw_overlap_count = int(np.count_nonzero(binary & overlap_mask))
                    raw_positive_background_count = int(
                        np.count_nonzero(binary & positive_background_mask)
                    )
                    if instance_qc is None:
                        if raw_overlap_count or raw_positive_background_count:
                            raise PanNukeSemanticsError(
                                f"fold {fold.fold_id} patch {patch_index}: raw instance "
                                f"({channel_index}, {instance_id}) touches an anomaly but is "
                                "missing from validator QC"
                            )
                    elif (
                        instance_qc.total_pixel_count != area
                        or instance_qc.overlap_pixel_count != raw_overlap_count
                        or instance_qc.positive_background_pixel_count
                        != raw_positive_background_count
                    ):
                        raise PanNukeSemanticsError(
                            f"fold {fold.fold_id} patch {patch_index}: affected-instance QC "
                            f"differs from raw instance ({channel_index}, {instance_id})"
                        )
                    touching_overlap = raw_overlap_count > 0
                    touching_positive_background = raw_positive_background_count > 0
                    if touching_overlap:
                        flags.append(OVERLAP_EXCLUSION_REASON)
                    if touching_positive_background:
                        flags.append("touches_positive_and_background")
                    (
                        overlap_class_channels,
                        overlap_instance_channels,
                        overlap_instance_ids,
                        overlap_instance_counts,
                    ) = _overlap_instance_evidence(
                        mask,
                        target_overlap_mask=binary & overlap_mask,
                        positive_channels=facts.positive_channel_indices,
                    )
                    if touching_overlap:
                        if instance_qc is None:
                            raise PanNukeSemanticsError(
                                f"fold {fold.fold_id} patch {patch_index}: overlap-touching "
                                "instance lacks saved QC evidence"
                            )
                        class_index_by_channel = {
                            value: index
                            for index, value in enumerate(facts.positive_channel_indices)
                        }
                        other_evidence = [
                            (evidence_channel, evidence_id)
                            for evidence_channel, evidence_id in zip(
                                overlap_instance_channels,
                                overlap_instance_ids,
                                strict=True,
                            )
                            if evidence_channel != channel_index
                        ]
                        other_class_indices = tuple(
                            sorted(
                                {
                                    class_index_by_channel[evidence_channel]
                                    for evidence_channel, _ in other_evidence
                                }
                            )
                        )
                        other_ids = tuple(sorted({value for _, value in other_evidence}))
                        other_ids_by_class = {
                            validation.mapping.class_names[class_index]: tuple(
                                sorted(
                                    {
                                        value
                                        for evidence_channel, value in other_evidence
                                        if class_index_by_channel[evidence_channel] == class_index
                                    }
                                )
                            )
                            for class_index in other_class_indices
                        }
                        if (
                            tuple(instance_qc.overlapping_class_indices) != other_class_indices
                            or tuple(instance_qc.overlapping_instance_ids) != other_ids
                            or instance_qc.overlapping_instance_ids_by_class != other_ids_by_class
                        ):
                            raise PanNukeSemanticsError(
                                f"fold {fold.fold_id} patch {patch_index}: saved cross-class "
                                f"identity evidence differs for ({channel_index}, {instance_id})"
                            )
                    sample_id = (
                        f"pannuke-f{fold.fold_id}-p{patch_index:06d}-c{class_index}-i{instance_id}"
                    )
                    summary_key = (fold.fold_id, class_name, tissue)
                    summary_counts[summary_key] += 1
                    classes_seen.add(class_name)
                    yield {
                        "sample_id": sample_id,
                        "official_fold": fold.fold_id,
                        "patch_id": patch_id,
                        "group_id": patch_id,
                        "grouping_unit": "source_patch",
                        "patient_id": None,
                        "wsi_id": None,
                        "tissue_type": tissue,
                        "source_image_path": source_image,
                        "source_mask_path": source_mask,
                        "source_tissue_path": source_tissue,
                        "source_patch_index": patch_index,
                        "patch_index": patch_index,
                        "nucleus_index_in_patch": nucleus_index,
                        "nucleus_class_index": class_index,
                        "nucleus_class_name": class_name,
                        "instance_channel_index": channel_index,
                        "instance_id": instance_id,
                        "raw_instance_identity": (
                            f"pannuke/fold={fold.fold_id}/patch={patch_index}/"
                            f"channel={channel_index}/instance={instance_id}"
                        ),
                        "bbox": [x_min, y_min, x_max, y_max],
                        "bbox_x_min": x_min,
                        "bbox_y_min": y_min,
                        "bbox_x_max": x_max,
                        "bbox_y_max": y_max,
                        "centroid": [centroid_x, centroid_y],
                        "centroid_x": centroid_x,
                        "centroid_y": centroid_y,
                        "area": area,
                        "perimeter": float(measure_perimeter(binary, neighborhood=8)),
                        "border_touch": border_touch,
                        "quality_flags": flags,
                        "patch_height": patch_qc.height,
                        "patch_width": patch_qc.width,
                        "patch_pixel_count": patch_qc.total_pixel_count,
                        "patch_has_overlap": patch_qc.has_cross_class_overlap,
                        "patch_has_void": patch_qc.has_void,
                        "patch_has_positive_and_background": (patch_qc.has_positive_and_background),
                        "patch_overlap_pixel_count": (patch_qc.cross_class_overlap_pixel_count),
                        "patch_overlap_pixel_rate": _pixel_rate(
                            patch_qc.cross_class_overlap_pixel_count,
                            patch_qc.total_pixel_count,
                        ),
                        "patch_positive_occupied_pixel_count": (patch_qc.positive_any_pixel_count),
                        "patch_positive_occupied_pixel_rate": _pixel_rate(
                            patch_qc.positive_any_pixel_count,
                            patch_qc.total_pixel_count,
                        ),
                        "patch_has_supplied_background_channel": (
                            facts.background_channel_index is not None
                        ),
                        "patch_supplied_background_channel_index": (facts.background_channel_index),
                        "patch_supplied_background_pixel_count": (patch_qc.background_pixel_count),
                        "patch_supplied_background_pixel_rate": _pixel_rate(
                            patch_qc.background_pixel_count,
                            patch_qc.total_pixel_count,
                        ),
                        "patch_void_pixel_count": patch_qc.void_pixel_count,
                        "patch_void_pixel_rate": _pixel_rate(
                            patch_qc.void_pixel_count,
                            patch_qc.total_pixel_count,
                        ),
                        "patch_positive_background_conflict_pixel_count": (
                            patch_qc.positive_and_background_pixel_count
                        ),
                        "patch_positive_background_conflict_pixel_rate": _pixel_rate(
                            patch_qc.positive_and_background_pixel_count,
                            patch_qc.total_pixel_count,
                        ),
                        "patch_anomaly_union_pixel_count": patch_qc.anomaly_union_pixel_count,
                        "patch_anomaly_union_pixel_rate": _pixel_rate(
                            patch_qc.anomaly_union_pixel_count,
                            patch_qc.total_pixel_count,
                        ),
                        "patch_qc_affected_instance_count": patch_qc.affected_instance_count,
                        "cross_class_overlap_touching": touching_overlap,
                        "overlap_pixel_count_for_instance": raw_overlap_count,
                        "overlap_class_channel_indices": overlap_class_channels,
                        "overlap_instance_channel_indices": overlap_instance_channels,
                        "overlap_instance_ids": overlap_instance_ids,
                        "overlap_instance_pixel_counts": overlap_instance_counts,
                        "positive_background_touching": touching_positive_background,
                        "positive_background_pixel_count_for_instance": (
                            raw_positive_background_count
                        ),
                        "qc_exclusion_reason": (
                            OVERLAP_EXCLUSION_REASON if touching_overlap else None
                        ),
                        "primary_eligible": not touching_overlap,
                        "confirmatory_eligible": not touching_overlap,
                        "crop_generated": False,
                        "crop_path": None,
                        "pre_corruption_label": class_index,
                        "observed_label": class_index,
                        "is_injected_corruption": False,
                        "corruption_type": "none",
                        "original_class": class_index,
                        "replacement_class": None,
                        "corruption_seed": None,
                        "corruption_rate": 0.0,
                        "corruption_representation": None,
                        "auditor_representation": None,
                        "feature_space_independent": None,
                        "circularity_risk": False,
                        "configuration_hash": configuration_hash,
                        "corruption_timestamp_utc": None,
                    }
                    nucleus_index += 1
            for class_name in classes_seen:
                summary_patch_counts[(fold.fold_id, class_name, tissue)] += 1


def _coerce_validation(
    source: PanNukeValidationResult | ValidationArtifacts | str | Path,
    *,
    output_dir: Path,
    class_mapping: VerifiedClassMapping | None,
    use_documented_default_mapping: bool,
    explicit_background_channel: int | None,
) -> PanNukeValidationResult:
    if isinstance(source, ValidationArtifacts):
        return source.result
    if isinstance(source, PanNukeValidationResult):
        return source
    mapping = resolve_class_mapping(
        class_mapping, use_documented_default=use_documented_default_mapping
    )
    discovery = discover_pannuke_release(source, positive_class_count=len(mapping.class_names))
    return validate_discovered_release(
        discovery,
        class_mapping=mapping,
        explicit_background_channel=explicit_background_channel,
        inventory_exclude_paths=(output_dir,),
    )


def _verify_inventory_unchanged(validation: PanNukeValidationResult) -> None:
    """Fail if raw files changed after the validation evidence was produced."""

    if not validation.inventory:
        raise PanNukeSemanticsError("validated raw-file inventory is empty")
    try:
        verify_raw_inventory_unchanged(validation)
    except PanNukeSemanticsError as error:
        raise PanNukeSemanticsError(
            "validated raw file changed before manifest construction because the complete "
            f"raw inventory changed; rerun the PanNuke validation gate: {error}"
        ) from error


def _write_parquet_streaming(
    destination: Path,
    validation: PanNukeValidationResult,
    *,
    batch_rows: int,
) -> tuple[int, int, dict[tuple[int, str, str], int], dict[tuple[int, str, str], int]]:
    schema = _manifest_schema(validation)
    summary_counts: dict[tuple[int, str, str], int] = defaultdict(int)
    summary_patch_counts: dict[tuple[int, str, str], int] = defaultdict(int)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=destination.suffix, dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    row_count = 0
    patch_ids: set[str] = set()
    buffer: list[dict[str, Any]] = []
    try:
        with pq.ParquetWriter(temporary, schema=schema, compression="zstd") as writer:
            for row in _iter_manifest_rows(
                validation,
                summary_counts=summary_counts,
                summary_patch_counts=summary_patch_counts,
            ):
                buffer.append(row)
                row_count += 1
                patch_ids.add(str(row["patch_id"]))
                if len(buffer) >= batch_rows:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                    buffer.clear()
            if buffer:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                buffer.clear()
        if row_count == 0:
            raise PanNukeSemanticsError("no positive nucleus instances were found")
        validate_manifest_invariants(pq.read_table(temporary), validation=validation)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return row_count, len(patch_ids), dict(summary_counts), dict(summary_patch_counts)


def _write_summary_csv(
    destination: Path,
    *,
    nucleus_counts: Mapping[tuple[int, str, str], int],
    patch_counts: Mapping[tuple[int, str, str], int],
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=destination.suffix, dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "official_fold",
                    "nucleus_class_name",
                    "tissue_type",
                    "nucleus_count",
                    "source_patch_count",
                    "grouping_unit",
                    "independence_note",
                ),
            )
            writer.writeheader()
            for key in sorted(nucleus_counts):
                fold_id, class_name, tissue = key
                writer.writerow(
                    {
                        "official_fold": fold_id,
                        "nucleus_class_name": class_name,
                        "tissue_type": tissue,
                        "nucleus_count": nucleus_counts[key],
                        "source_patch_count": patch_counts.get(key, 0),
                        "grouping_unit": "source_patch",
                        "independence_note": SOURCE_PATCH_INDEPENDENCE_STATEMENT,
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _ensure_output_outside_raw(output: Path, raw_root: Path) -> None:
    """Reject derived-output paths located inside the immutable raw release."""

    resolved_output = output.resolve()
    resolved_raw = raw_root.resolve()
    if resolved_output == resolved_raw or resolved_raw in resolved_output.parents:
        raise PanNukeSemanticsError(
            "nucleus-manifest output must be outside the immutable PanNuke raw root"
        )


def _verify_manifest_publication_destinations(
    destinations: tuple[Path, ...],
    *,
    validation: PanNukeValidationResult,
) -> None:
    """Require distinct final paths that cannot resolve into the raw release.

    The check is repeated while the bundle lock is held and immediately around
    publication.  It deliberately uses both resolved paths and existing-file
    identities: distinct names must not alias one another, and an existing output
    hard link must not alias an immutable raw input.
    """

    if len(destinations) < 2:
        raise PanNukeSemanticsError("PanNuke manifest publication requires a complete bundle")

    raw_root = validation.root.resolve()
    canonical_keys: set[str] = set()
    for destination in destinations:
        resolved = destination.resolve(strict=False)
        key = os.path.normcase(str(resolved))
        if key in canonical_keys:
            raise PanNukeSemanticsError(
                "PanNuke manifest publication destinations must be distinct and non-colliding"
            )
        canonical_keys.add(key)
        if resolved == raw_root or raw_root in resolved.parents:
            raise PanNukeSemanticsError(
                "nucleus-manifest publication paths must be outside the immutable PanNuke raw root"
            )

    existing = tuple(path for path in destinations if os.path.lexists(path))
    for index, first in enumerate(existing):
        for second in existing[index + 1 :]:
            try:
                collision = os.path.samefile(first, second)
            except OSError:
                collision = False
            if collision:
                raise PanNukeSemanticsError(
                    "PanNuke manifest publication destinations must not alias the same file"
                )

    raw_paths = tuple(validation.root / item.relative_path for item in validation.inventory)
    for destination in existing:
        for raw_path in raw_paths:
            try:
                aliases_raw = os.path.samefile(destination, raw_path)
            except OSError:
                aliases_raw = False
            if aliases_raw:
                raise PanNukeSemanticsError(
                    "nucleus-manifest publication path aliases an immutable PanNuke raw file"
                )


def _same_file_bytes(first: Path, second: Path) -> bool:
    return first.stat().st_size == second.stat().st_size and sha256_file(first) == sha256_file(
        second
    )


def _promote_staged_manifest_file(staged: Path, destination: Path) -> PublishedPath:
    """Atomic create-if-absent seam for coordinated manifest publication."""

    return publish_file_no_overwrite(staged, destination)


def _publish_manifest_pair(
    *,
    staged_parquet: Path,
    staged_summary: Path,
    parquet_path: Path,
    summary_path: Path,
    raw_inventory_verifier: Callable[[], object],
    destination_verifier: Callable[[], object],
    lock_verifier: Callable[[], object],
) -> list[PublishedPath]:
    """Publish both artifacts together, preserving existing outputs on conflict."""

    # The Parquet manifest is the canonical success marker and is published last.
    destinations = ((staged_summary, summary_path), (staged_parquet, parquet_path))
    lock_verifier()
    destination_verifier()
    raw_inventory_verifier()
    existing = tuple(os.path.lexists(destination) for _, destination in destinations)
    if any(existing):
        if not all(existing):
            raise FileExistsError(
                "refusing to publish over an incomplete existing PanNuke manifest artifact set"
            )
        if not all(_same_file_bytes(staged, destination) for staged, destination in destinations):
            raise FileExistsError(
                "refusing to overwrite non-identical existing PanNuke manifest artifacts"
            )
        raw_inventory_verifier()
        destination_verifier()
        lock_verifier()
        if not all(
            os.path.lexists(destination) and _same_file_bytes(staged, destination)
            for staged, destination in destinations
        ):
            raise RuntimeError(
                "existing PanNuke manifest artifacts changed during idempotent publication"
            )
        return []

    published: list[PublishedPath] = []
    try:
        for staged, destination in destinations:
            lock_verifier()
            destination_verifier()
            published.append(_promote_staged_manifest_file(staged, destination))
        # Close the publish-time race: a raw mutation during either promotion
        # invalidates the derived bundle and rolls both newly published files back.
        raw_inventory_verifier()
        destination_verifier()
        lock_verifier()
        if not all(publication.still_owned() for publication in published):
            raise RuntimeError(
                "PanNuke manifest publication lost ownership before final bundle readback"
            )
        if not all(_same_file_bytes(staged, destination) for staged, destination in destinations):
            raise RuntimeError("PanNuke manifest bundle failed final byte-for-byte readback")
    except BaseException as error:
        try:
            rollback_owned_publications(published)
        except RuntimeError:
            raise RuntimeError(
                "PanNuke manifest publication failed and ownership-safe rollback was incomplete"
            ) from error
        raise
    return published


def build_nucleus_manifest(
    source: PanNukeValidationResult | ValidationArtifacts | str | Path,
    output_dir: str | Path,
    *,
    class_mapping: VerifiedClassMapping | None = None,
    use_documented_default_mapping: bool = True,
    explicit_background_channel: int | None = None,
    batch_rows: int = 4096,
) -> ManifestArtifacts:
    """Build an atomic Parquet manifest and CSV summary without writing crops."""

    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    output = Path(output_dir).resolve()
    if isinstance(source, (str, Path)):
        _ensure_output_outside_raw(output, Path(source))
    validation = _coerce_validation(
        source,
        output_dir=output,
        class_mapping=class_mapping,
        use_documented_default_mapping=use_documented_default_mapping,
        explicit_background_channel=explicit_background_channel,
    )
    _ensure_output_outside_raw(output, validation.root)
    parquet_path = output / "pannuke_nucleus_manifest.parquet"
    summary_path = output / "pannuke_manifest_summary.csv"
    final_paths = (parquet_path, summary_path)

    def destination_verifier() -> None:
        _verify_manifest_publication_destinations(final_paths, validation=validation)

    published: list[PublishedPath] = []
    try:
        with ExclusiveBundlePublicationLock(final_paths, role="PanNuke manifest") as bundle_lock:
            destination_verifier()
            output.mkdir(parents=True, exist_ok=True)
            destination_verifier()
            _verify_inventory_unchanged(validation)
            total_mask_bytes = sum(fold.mask_path.stat().st_size for fold in validation.folds)
            ensure_output_capacity(
                output,
                estimated_bytes=min(4 * 1024**3, max(16 * 1024**2, total_mask_bytes // 8)),
            )
            with tempfile.TemporaryDirectory(
                prefix=".pannuke-manifest-stage-", dir=output
            ) as stage_name:
                stage = Path(stage_name)
                staged_parquet = stage / parquet_path.name
                staged_summary = stage / summary_path.name
                row_count, patch_count, nucleus_counts, patch_counts = _write_parquet_streaming(
                    staged_parquet, validation, batch_rows=batch_rows
                )
                _write_summary_csv(
                    staged_summary,
                    nucleus_counts=nucleus_counts,
                    patch_counts=patch_counts,
                )
                validate_manifest_invariants(pq.read_table(staged_parquet), validation=validation)
                _verify_inventory_unchanged(validation)
                published = _publish_manifest_pair(
                    staged_parquet=staged_parquet,
                    staged_summary=staged_summary,
                    parquet_path=parquet_path,
                    summary_path=summary_path,
                    raw_inventory_verifier=lambda: _verify_inventory_unchanged(validation),
                    destination_verifier=destination_verifier,
                    lock_verifier=bundle_lock.assert_owned,
                )
            if published and not all(publication.still_owned() for publication in published):
                raise RuntimeError(
                    "PanNuke manifest publication lost ownership after staging cleanup"
                )
            artifacts = ManifestArtifacts(
                parquet_path=parquet_path,
                summary_csv_path=summary_path,
                row_count=row_count,
                patch_count=patch_count,
                sha256=sha256_file(parquet_path),
            )
            if published and not all(publication.still_owned() for publication in published):
                raise RuntimeError(
                    "PanNuke manifest publication lost ownership before transaction completion"
                )
            bundle_lock.assert_owned()
        return artifacts
    except BaseException as error:
        if published:
            try:
                rollback_owned_publications(published)
            except RuntimeError:
                raise RuntimeError(
                    "PanNuke manifest transaction failed and ownership-safe rollback was incomplete"
                ) from error
        raise


def _is_missing_scalar(value: Any) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and bool(np.isnan(value)))


def _list_values(value: Any, *, field: str, sample_id: str) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ValueError(f"{sample_id}: manifest {field} must be a list")


def _require_sha256(value: str, *, field: str) -> None:
    if (
        len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")


def _validate_manifest_metadata(
    table: pa.Table,
    frame: Any,
    *,
    validation: PanNukeValidationResult | None,
) -> tuple[str, ...]:
    metadata = table.schema.metadata or {}
    exact_metadata = {
        b"dataset": b"PanNuke",
        b"bbox_convention": b"half-open [x_min, y_min, x_max, y_max]",
        b"grouping_unit": b"source_patch",
        b"source_annotations_modified": b"false",
        b"tiny_crops_generated": b"false",
        b"cross_class_overlap_policy": (
            b"preserve_raw_channel_instance_identity;exclude_every_touching_instance;"
            b"never_resolve_overlap_pixels"
        ),
        b"cross_class_overlap_exclusion_reason": OVERLAP_EXCLUSION_REASON.encode("ascii"),
        b"eligibility_policy": b"one_identical_primary_and_confirmatory_instance_mask",
        b"void_pixel_policy": b"retain_as_unlabeled_void;never_assign_background_or_positive_class",
        b"representation_cache_binding_policy": (
            b"downstream_cache_must_bind_manifest_sha256_and_crop_configuration"
        ),
    }
    for key, expected in exact_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"manifest metadata {key.decode()} is missing or inconsistent")

    configuration_hashes = frame["configuration_hash"].astype(str)
    if configuration_hashes.nunique() != 1:
        raise ValueError("source manifest must have one immutable configuration hash")
    configuration_hash = str(configuration_hashes.iloc[0])
    _require_sha256(configuration_hash, field="manifest configuration_hash")
    if metadata.get(b"manifest_configuration_hash") != configuration_hash.encode("ascii"):
        raise ValueError("manifest configuration hash differs between rows and schema metadata")

    inventory_payload = metadata.get(b"raw_file_inventory")
    inventory_digest = metadata.get(b"raw_file_inventory_sha256")
    if inventory_payload is None or inventory_digest is None:
        raise ValueError("manifest lacks immutable raw-file inventory metadata")
    digest_text = inventory_digest.decode("ascii", errors="strict")
    _require_sha256(digest_text, field="raw_file_inventory_sha256")
    if hashlib.sha256(inventory_payload).hexdigest() != digest_text:
        raise ValueError("manifest raw-file inventory digest is inconsistent")
    try:
        inventory = json.loads(inventory_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("manifest raw-file inventory is not valid canonical JSON") from error
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("manifest raw-file inventory must be a non-empty list")
    paths: set[str] = set()
    for item in inventory:
        if not isinstance(item, dict):
            raise ValueError("manifest raw-file inventory records must be objects")
        relative_path = item.get("relative_path")
        size_bytes = item.get("size_bytes")
        sha256 = item.get("sha256")
        if not isinstance(relative_path, str) or not relative_path or relative_path in paths:
            raise ValueError("manifest raw-file inventory paths must be non-empty and unique")
        paths.add(relative_path)
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise ValueError("manifest raw-file inventory size_bytes is invalid")
        if not isinstance(sha256, str):
            raise ValueError("manifest raw-file inventory SHA-256 is missing")
        _require_sha256(sha256, field=f"raw file {relative_path} SHA-256")

    try:
        mapping = json.loads(metadata[b"class_mapping"])
        class_names_value = mapping["class_names"]
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("manifest class-mapping metadata is invalid") from error
    if (
        not isinstance(class_names_value, list)
        or not class_names_value
        or any(not isinstance(value, str) or not value for value in class_names_value)
        or len(set(class_names_value)) != len(class_names_value)
    ):
        raise ValueError("manifest class-mapping names are invalid")

    if validation is not None:
        _verify_inventory_unchanged(validation)
        if inventory_payload != _raw_inventory_payload(validation):
            raise ValueError("manifest raw-file inventory differs from validated release")
        if configuration_hash != _manifest_configuration_hash(validation):
            raise ValueError("manifest configuration hash differs from validated release")
        if tuple(class_names_value) != validation.mapping.class_names:
            raise ValueError("manifest class mapping differs from validated release")
    return tuple(class_names_value)


def _validate_patch_qc_rows(patch: Any) -> None:
    patch_id = str(patch["patch_id"].iloc[0])
    constant_fields = (
        "patch_height",
        "patch_width",
        "patch_pixel_count",
        "patch_has_overlap",
        "patch_has_void",
        "patch_has_positive_and_background",
        "patch_overlap_pixel_count",
        "patch_overlap_pixel_rate",
        "patch_positive_occupied_pixel_count",
        "patch_positive_occupied_pixel_rate",
        "patch_has_supplied_background_channel",
        "patch_supplied_background_channel_index",
        "patch_supplied_background_pixel_count",
        "patch_supplied_background_pixel_rate",
        "patch_void_pixel_count",
        "patch_void_pixel_rate",
        "patch_positive_background_conflict_pixel_count",
        "patch_positive_background_conflict_pixel_rate",
        "patch_anomaly_union_pixel_count",
        "patch_anomaly_union_pixel_rate",
        "patch_qc_affected_instance_count",
    )
    for field in constant_fields:
        values = patch[field]
        if values.nunique(dropna=False) != 1:
            raise ValueError(f"{patch_id}: patch-level QC field {field} is not constant")

    row = patch.iloc[0]
    height = int(row["patch_height"])
    width = int(row["patch_width"])
    pixel_count = int(row["patch_pixel_count"])
    if height <= 0 or width <= 0 or pixel_count != height * width:
        raise ValueError(f"{patch_id}: invalid patch dimensions or pixel count")
    count_rate_fields = (
        ("patch_overlap_pixel_count", "patch_overlap_pixel_rate"),
        ("patch_positive_occupied_pixel_count", "patch_positive_occupied_pixel_rate"),
        ("patch_supplied_background_pixel_count", "patch_supplied_background_pixel_rate"),
        ("patch_void_pixel_count", "patch_void_pixel_rate"),
        (
            "patch_positive_background_conflict_pixel_count",
            "patch_positive_background_conflict_pixel_rate",
        ),
        ("patch_anomaly_union_pixel_count", "patch_anomaly_union_pixel_rate"),
    )
    counts: dict[str, int] = {}
    for count_field, rate_field in count_rate_fields:
        count = int(row[count_field])
        rate = float(row[rate_field])
        if not 0 <= count <= pixel_count or not np.isfinite(rate):
            raise ValueError(f"{patch_id}: invalid {count_field}/{rate_field}")
        if not np.isclose(rate, count / pixel_count, rtol=0.0, atol=1e-15):
            raise ValueError(f"{patch_id}: {rate_field} does not match its exact count")
        counts[count_field] = count
    overlap_count = counts["patch_overlap_pixel_count"]
    if bool(row["patch_has_overlap"]) != (overlap_count > 0):
        raise ValueError(f"{patch_id}: patch_has_overlap does not match overlap count")
    if overlap_count > counts["patch_positive_occupied_pixel_count"]:
        raise ValueError(f"{patch_id}: overlap count exceeds positive occupancy")
    conflict_count = counts["patch_positive_background_conflict_pixel_count"]
    positive_count = counts["patch_positive_occupied_pixel_count"]
    background_count = counts["patch_supplied_background_pixel_count"]
    void_count = counts["patch_void_pixel_count"]
    anomaly_union_count = counts["patch_anomaly_union_pixel_count"]
    if positive_count + background_count + void_count - conflict_count != pixel_count:
        raise ValueError(f"{patch_id}: positive/background/void QC counts do not partition pixels")
    if bool(row["patch_has_void"]) != (void_count > 0):
        raise ValueError(f"{patch_id}: patch_has_void does not match void count")
    if bool(row["patch_has_positive_and_background"]) != (conflict_count > 0):
        raise ValueError(f"{patch_id}: patch_has_positive_and_background does not match its count")
    if anomaly_union_count < max(overlap_count, void_count, conflict_count):
        raise ValueError(f"{patch_id}: anomaly-union count is smaller than a component")
    if anomaly_union_count > overlap_count + void_count + conflict_count:
        raise ValueError(f"{patch_id}: anomaly-union count exceeds component sum")
    has_background = bool(row["patch_has_supplied_background_channel"])
    background_channel = row["patch_supplied_background_channel_index"]
    if has_background == _is_missing_scalar(background_channel):
        raise ValueError(f"{patch_id}: supplied-background channel flag/index disagree")
    if not has_background and (background_count != 0 or conflict_count != 0):
        raise ValueError(f"{patch_id}: background metrics exist without a supplied channel")

    touching = patch["cross_class_overlap_touching"].astype(bool)
    if bool(row["patch_has_overlap"]) != bool(touching.any()):
        raise ValueError(f"{patch_id}: overlap flag lacks matching affected raw instances")
    touched_sum = int(patch.loc[touching, "overlap_pixel_count_for_instance"].sum())
    if overlap_count > 0 and touched_sum < 2 * overlap_count:
        raise ValueError(
            f"{patch_id}: affected-instance overlap counts under-evidence patch overlap"
        )
    positive_background_touching = patch["positive_background_touching"].astype(bool)
    if bool(row["patch_has_positive_and_background"]) != bool(positive_background_touching.any()):
        raise ValueError(
            f"{patch_id}: positive/background flag lacks matching affected raw instances"
        )
    affected = touching | positive_background_touching
    if int(row["patch_qc_affected_instance_count"]) != int(affected.sum()):
        raise ValueError(f"{patch_id}: patch affected-instance count is inconsistent")


def validate_manifest_invariants(
    table: pa.Table,
    *,
    validation: PanNukeValidationResult | None = None,
) -> None:
    """Fail closed on raw identity, QC evidence, grouping, geometry, and immutability."""

    missing = set(MANIFEST_REQUIRED_COLUMNS).difference(table.column_names)
    if missing:
        raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
    frame = table.select(list(MANIFEST_REQUIRED_COLUMNS)).to_pandas()
    if frame.empty:
        raise ValueError("manifest must contain at least one nucleus")
    class_names = _validate_manifest_metadata(table, frame, validation=validation)

    if frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique")
    raw_key_fields = (
        "official_fold",
        "source_patch_index",
        "instance_channel_index",
        "instance_id",
    )
    if frame.duplicated(subset=list(raw_key_fields)).any():
        raise ValueError("raw fold/patch/channel/instance identity must be unique")
    if not (frame["patch_id"] == frame["group_id"]).all():
        raise ValueError("source patch must be the group_id when stronger metadata is absent")
    if not (frame["grouping_unit"] == "source_patch").all():
        raise ValueError("unexpected grouping unit")
    if not (frame["source_patch_index"] == frame["patch_index"]).all():
        raise ValueError("source and working patch indices must remain identical")
    if frame["patient_id"].notna().any() or frame["wsi_id"].notna().any():
        raise ValueError("unverified patient/WSI identifiers cannot appear in the manifest")

    if not (
        (frame["pre_corruption_label"] == frame["observed_label"])
        & (frame["pre_corruption_label"] == frame["nucleus_class_index"])
        & (frame["original_class"] == frame["nucleus_class_index"])
    ).all():
        raise ValueError("source manifest immutable labels must match the raw class identity")
    if frame["is_injected_corruption"].any():
        raise ValueError("source manifest cannot contain injected corruption")
    if not (frame["corruption_type"] == "none").all() or not np.allclose(
        frame["corruption_rate"].astype(float), 0.0, rtol=0.0, atol=0.0
    ):
        raise ValueError("source manifest corruption metadata must remain inactive")
    nullable_corruption_fields = (
        "replacement_class",
        "corruption_seed",
        "corruption_representation",
        "auditor_representation",
        "feature_space_independent",
        "corruption_timestamp_utc",
    )
    if any(frame[field].notna().any() for field in nullable_corruption_fields):
        raise ValueError("source manifest contains non-null corruption metadata")
    if frame["circularity_risk"].any():
        raise ValueError("source manifest cannot assert circularity risk")
    if frame["crop_generated"].any() or frame["crop_path"].notna().any():
        raise ValueError("manifest construction must not generate tiny crops")

    for _, row in frame.iterrows():
        sample_id = str(row["sample_id"])
        fold_id = int(row["official_fold"])
        patch_index = int(row["source_patch_index"])
        class_index = int(row["nucleus_class_index"])
        channel_index = int(row["instance_channel_index"])
        instance_id = int(row["instance_id"])
        if fold_id <= 0 or patch_index < 0 or channel_index < 0 or instance_id <= 0:
            raise ValueError(f"{sample_id}: raw instance identity has invalid indices")
        if not 0 <= class_index < len(class_names):
            raise ValueError(f"{sample_id}: nucleus class index is outside verified mapping")
        if str(row["nucleus_class_name"]) != class_names[class_index]:
            raise ValueError(f"{sample_id}: nucleus class name differs from verified mapping")
        expected_patch_id = f"pannuke-fold-{fold_id}-patch-{patch_index:06d}"
        expected_sample_id = f"pannuke-f{fold_id}-p{patch_index:06d}-c{class_index}-i{instance_id}"
        expected_raw_identity = (
            f"pannuke/fold={fold_id}/patch={patch_index}/channel={channel_index}/"
            f"instance={instance_id}"
        )
        if str(row["patch_id"]) != expected_patch_id or sample_id != expected_sample_id:
            raise ValueError(f"{sample_id}: canonical patch/sample identity is inconsistent")
        if str(row["raw_instance_identity"]) != expected_raw_identity:
            raise ValueError(f"{sample_id}: canonical raw instance identity is inconsistent")
        if any(
            not str(row[field])
            for field in ("source_image_path", "source_mask_path", "source_tissue_path")
        ):
            raise ValueError(f"{sample_id}: raw source paths must be non-empty")

        x_min = int(row["bbox_x_min"])
        y_min = int(row["bbox_y_min"])
        x_max = int(row["bbox_x_max"])
        y_max = int(row["bbox_y_max"])
        bbox = [
            int(value) for value in _list_values(row["bbox"], field="bbox", sample_id=sample_id)
        ]
        if bbox != [x_min, y_min, x_max, y_max]:
            raise ValueError(f"{sample_id}: packed and scalar bounding boxes disagree")
        height = int(row["patch_height"])
        width = int(row["patch_width"])
        area = int(row["area"])
        if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height and area > 0):
            raise ValueError(f"{sample_id}: manifest contains invalid nucleus geometry")
        if area > (x_max - x_min) * (y_max - y_min):
            raise ValueError(f"{sample_id}: area exceeds its bounding box")
        centroid_x = float(row["centroid_x"])
        centroid_y = float(row["centroid_y"])
        centroid = [
            float(value)
            for value in _list_values(row["centroid"], field="centroid", sample_id=sample_id)
        ]
        if len(centroid) != 2 or not np.allclose(
            centroid, [centroid_x, centroid_y], rtol=0.0, atol=0.0
        ):
            raise ValueError(f"{sample_id}: packed and scalar centroids disagree")
        if not (
            np.isfinite(centroid_x)
            and np.isfinite(centroid_y)
            and x_min <= centroid_x < x_max
            and y_min <= centroid_y < y_max
            and np.isfinite(float(row["perimeter"]))
            and float(row["perimeter"]) >= 0.0
        ):
            raise ValueError(f"{sample_id}: centroid or perimeter is invalid")

        flags = [
            str(value)
            for value in _list_values(
                row["quality_flags"], field="quality_flags", sample_id=sample_id
            )
        ]
        if len(flags) != len(set(flags)):
            raise ValueError(f"{sample_id}: quality flags must be unique")
        expected_patch_flags = {
            PATCH_OVERLAP_FLAG: bool(row["patch_has_overlap"]),
            PATCH_VOID_FLAG: bool(row["patch_has_void"]),
            PATCH_POSITIVE_BACKGROUND_FLAG: bool(row["patch_has_positive_and_background"]),
        }
        for flag, expected in expected_patch_flags.items():
            if (flag in flags) is not expected:
                raise ValueError(f"{sample_id}: patch QC flag {flag} is inconsistent")
        touching = bool(row["cross_class_overlap_touching"])
        overlap_count = int(row["overlap_pixel_count_for_instance"])
        if not 0 <= overlap_count <= area:
            raise ValueError(f"{sample_id}: instance overlap count is outside its raw area")
        class_channels = [
            int(value)
            for value in _list_values(
                row["overlap_class_channel_indices"],
                field="overlap_class_channel_indices",
                sample_id=sample_id,
            )
        ]
        evidence_channels = [
            int(value)
            for value in _list_values(
                row["overlap_instance_channel_indices"],
                field="overlap_instance_channel_indices",
                sample_id=sample_id,
            )
        ]
        evidence_ids = [
            int(value)
            for value in _list_values(
                row["overlap_instance_ids"], field="overlap_instance_ids", sample_id=sample_id
            )
        ]
        evidence_counts = [
            int(value)
            for value in _list_values(
                row["overlap_instance_pixel_counts"],
                field="overlap_instance_pixel_counts",
                sample_id=sample_id,
            )
        ]
        evidence = list(zip(evidence_channels, evidence_ids, evidence_counts, strict=True))
        if not (
            len(evidence_channels) == len(evidence_ids) == len(evidence_counts)
            and evidence == sorted(evidence, key=lambda value: (value[0], value[1]))
            and len({(channel, identity) for channel, identity, _ in evidence}) == len(evidence)
            and all(
                channel >= 0 and identity > 0 and count > 0 for channel, identity, count in evidence
            )
            and class_channels == sorted(set(evidence_channels))
        ):
            raise ValueError(f"{sample_id}: overlap class/instance evidence is invalid")
        positive_background_touching = bool(row["positive_background_touching"])
        positive_background_count = int(row["positive_background_pixel_count_for_instance"])
        if (
            not 0 <= positive_background_count <= area
            or positive_background_touching != (positive_background_count > 0)
            or ("touches_positive_and_background" in flags) is not positive_background_touching
        ):
            raise ValueError(f"{sample_id}: positive/background instance evidence is inconsistent")
        reason = row["qc_exclusion_reason"]
        primary_eligible = bool(row["primary_eligible"])
        confirmatory_eligible = bool(row["confirmatory_eligible"])
        if primary_eligible is not confirmatory_eligible:
            raise ValueError(
                f"{sample_id}: primary and confirmatory eligibility masks must be identical"
            )
        overlap_flag_present = OVERLAP_EXCLUSION_REASON in flags
        if touching:
            target_counts = [
                count
                for channel, identity, count in evidence
                if channel == channel_index and identity == instance_id
            ]
            if (
                overlap_count <= 0
                or reason != OVERLAP_EXCLUSION_REASON
                or primary_eligible
                or confirmatory_eligible
                or not overlap_flag_present
                or len(class_channels) < 2
                or len(target_counts) != 1
                or target_counts[0] != overlap_count
                or not any(channel != channel_index for channel in class_channels)
                or not bool(row["patch_has_overlap"])
                or overlap_count > int(row["patch_overlap_pixel_count"])
            ):
                raise ValueError(f"{sample_id}: overlap exclusion evidence is inconsistent")
        elif (
            overlap_count != 0
            or evidence
            or class_channels
            or not _is_missing_scalar(reason)
            or not primary_eligible
            or not confirmatory_eligible
            or overlap_flag_present
        ):
            raise ValueError(f"{sample_id}: non-touching instance carries overlap exclusion")

    for patch_id, patch in frame.groupby("patch_id", sort=False):
        indices = sorted(int(value) for value in patch["nucleus_index_in_patch"].tolist())
        if indices != list(range(len(patch))):
            raise ValueError(f"{patch_id}: nucleus indices must be contiguous and unique")
        _validate_patch_qc_rows(patch)


# CLI-friendly synonym.
build_manifest = build_nucleus_manifest


__all__ = [
    "MANIFEST_REQUIRED_COLUMNS",
    "build_manifest",
    "build_nucleus_manifest",
    "validate_manifest_invariants",
]
