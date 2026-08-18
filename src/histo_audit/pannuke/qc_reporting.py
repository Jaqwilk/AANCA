"""Immutable, reconciled PanNuke mask-QC reports and anomaly overlays.

The reporter consumes a successful :class:`PanNukeValidationResult`.  It never
changes a source array and never assigns an overlapping pixel to one of its raw
positive classes.  Cross-class overlap is rendered with one neutral colour.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch as LegendPatch
from numpy.typing import NDArray

from .exceptions import PanNukeSemanticsError
from .io import open_npy_mmap, sha256_file
from .models import (
    AnomalyOverlaySelection,
    FoldMaskQC,
    MaskInstanceQC,
    PanNukeValidationResult,
    PatchMaskQC,
)
from .publication import (
    ExclusiveBundlePublicationLock,
    PublishedPath,
    publish_flat_directory_no_overwrite,
    rollback_owned_publications,
)
from .validation import verify_raw_inventory_unchanged

QC_REPORT_SCHEMA_VERSION = 2
OVERLAP_COLOUR_HEX = "#d9d9d9"
VOID_COLOUR_HEX = "#f0a830"
OVERLAP_ALPHA = 0.82
VOID_ALPHA = 0.42


def _verify_report_raw_inventory(result: PanNukeValidationResult) -> None:
    """Rehash when validation carries canonical inventory; allow isolated unit fixtures."""

    if result.inventory:
        verify_raw_inventory_unchanged(result)


_REPORT_JSON = "pannuke_mask_qc.json"
_PATCH_CSV = "pannuke_mask_qc_patches.csv"
_INSTANCE_CSV = "pannuke_mask_qc_instances.csv"
_MARKDOWN = "pannuke_mask_qc.md"
_OVERLAY_PNG = "pannuke_mask_qc_overlays.png"
_SELECTION_JSON = "pannuke_mask_qc_overlay_selection.json"
_ARTIFACT_MANIFEST_JSON = "artifact_manifest.json"
_CONTENT_NAMES = (
    _REPORT_JSON,
    _PATCH_CSV,
    _INSTANCE_CSV,
    _MARKDOWN,
    _OVERLAY_PNG,
    _SELECTION_JSON,
)
_REQUIRED_NAMES = (*_CONTENT_NAMES, _ARTIFACT_MANIFEST_JSON)

_SUM_FIELDS = (
    "patch_count",
    "total_pixel_count",
    "positive_any_pixel_count",
    "background_pixel_count",
    "void_pixel_count",
    "cross_class_overlap_pixel_count",
    "positive_and_background_pixel_count",
    "anomaly_union_pixel_count",
    "void_patch_count",
    "cross_class_overlap_patch_count",
    "positive_and_background_patch_count",
    "anomaly_union_patch_count",
    "normal_patch_count",
    "affected_instance_count",
    "overlap_touching_instance_count",
    "positive_background_touching_instance_count",
)

_PATCH_FIELDS = (
    "patch_key",
    "fold_id",
    "patch_index",
    "height",
    "width",
    "total_pixel_count",
    "positive_any_pixel_count",
    "background_pixel_count",
    "void_pixel_count",
    "cross_class_overlap_pixel_count",
    "positive_and_background_pixel_count",
    "anomaly_union_pixel_count",
    "affected_instance_count",
    "affected_class_indices",
    "affected_class_names",
    "has_void",
    "has_cross_class_overlap",
    "has_positive_and_background",
    "mask_sha256_by_kind",
)

_INSTANCE_FIELDS = (
    "patch_key",
    "fold_id",
    "patch_index",
    "class_index",
    "class_name",
    "channel_index",
    "instance_id",
    "total_pixel_count",
    "overlap_pixel_count",
    "positive_background_pixel_count",
    "overlapping_class_indices",
    "overlapping_instance_ids",
    "overlapping_instance_ids_by_class",
    "touches_cross_class_overlap",
    "analysis_eligible",
    "primary_eligible",
    "confirmatory_eligible",
    "analysis_exclusion_reason",
)


class MaskQCReportError(PanNukeSemanticsError):
    """The QC result or immutable report bundle failed reconciliation."""


@dataclass(frozen=True, slots=True)
class MaskQCReportArtifacts:
    """Paths and hashes for one immutable mask-QC report bundle."""

    bundle_dir: Path
    json_path: Path
    patch_csv_path: Path
    instance_csv_path: Path
    markdown_path: Path
    overlay_path: Path
    overlay_selection_path: Path
    artifact_manifest_path: Path
    selection_sha256: str
    overlay_sha256: str
    patch_row_count: int
    instance_row_count: int


@dataclass(frozen=True, slots=True)
class _SelectedPatchEvidence:
    patch_key: str
    fold_id: int
    patch_index: int
    categories: tuple[str, ...]
    image_patch_sha256: str
    raw_mask_patch_sha256: str
    overlap_mask_sha256: str
    void_mask_sha256: str
    positive_and_background_mask_sha256: str
    overlap_pixel_count: int
    void_pixel_count: int
    positive_and_background_pixel_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "patch_key": self.patch_key,
            "fold_id": self.fold_id,
            "patch_index": self.patch_index,
            "categories": list(self.categories),
            "image_patch_sha256": self.image_patch_sha256,
            "raw_mask_patch_sha256": self.raw_mask_patch_sha256,
            "overlap_mask_sha256": self.overlap_mask_sha256,
            "void_mask_sha256": self.void_mask_sha256,
            "positive_and_background_mask_sha256": (self.positive_and_background_mask_sha256),
            "overlap_pixel_count": self.overlap_pixel_count,
            "void_pixel_count": self.void_pixel_count,
            "positive_and_background_pixel_count": (self.positive_and_background_pixel_count),
        }


def _canonical_json_bytes(value: Any, *, pretty: bool) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value, pretty=False)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value: NDArray[np.generic]) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(b"histo-audit-array-v1\0")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _qc_flag_mask_sha256(kind: str, value: NDArray[np.bool_]) -> str:
    """Independently reproduce the validator's versioned binary-mask digest."""

    contiguous = np.ascontiguousarray(value, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(b"pannuke-mask-qc-v1\0")
    digest.update(kind.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def patch_key(fold_id: int, patch_index: int) -> str:
    """Return the stable source-patch identifier used in QC report artifacts."""

    if fold_id <= 0 or patch_index < 0:
        raise ValueError("fold_id must be positive and patch_index non-negative")
    return f"fold_{fold_id}:patch_{patch_index}"


def _parse_patch_key(value: str) -> tuple[int, int]:
    patterns = (
        r"^fold_(\d+):patch_(\d+)$",
        r"^fold-(\d+)/patch-(\d+)$",
        r"^fold_(\d+)_patch_(\d+)$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, value)
        if match is not None:
            return int(match.group(1)), int(match.group(2))
    raise MaskQCReportError(f"invalid anomaly-overlay patch key: {value!r}")


def _channel_last_patch(
    array: NDArray[np.generic], patch_index: int, channel_axis: int
) -> NDArray[np.generic]:
    patch = np.asarray(array[patch_index])
    patch_axis = channel_axis - 1
    if patch_axis < 0 or patch_axis >= patch.ndim:
        raise MaskQCReportError(
            f"invalid channel axis {channel_axis} for patch shape {patch.shape}"
        )
    return np.moveaxis(patch, patch_axis, -1) if patch_axis != patch.ndim - 1 else patch


def _normalise_rgb(image: NDArray[np.generic]) -> NDArray[np.float32]:
    if image.ndim != 3 or image.shape[-1] < 3:
        raise MaskQCReportError(f"overlay image is not channel-last RGB: shape={image.shape}")
    value = image[..., :3].astype(np.float32, copy=False)
    if not np.isfinite(value).all():
        raise MaskQCReportError("overlay image contains non-finite values")
    minimum = float(value.min(initial=0.0))
    maximum = float(value.max(initial=0.0))
    if minimum >= 0.0 and maximum <= 1.0:
        return np.clip(value, 0.0, 1.0)
    if minimum >= 0.0 and maximum <= 255.0:
        return np.clip(value / 255.0, 0.0, 1.0)
    scale = maximum - minimum
    if scale <= 0.0:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip((value - minimum) / scale, 0.0, 1.0)


def anomaly_overlay_rgba(
    overlap_mask: NDArray[np.bool_],
    void_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Build a class-neutral RGBA anomaly layer.

    Void is drawn first.  Cross-class overlap is drawn last in the same neutral
    grey regardless of which positive channels contributed to the overlap.
    """

    overlap = np.asarray(overlap_mask, dtype=bool)
    void = np.asarray(void_mask, dtype=bool)
    if overlap.shape != void.shape or overlap.ndim != 2:
        raise ValueError("overlap_mask and void_mask must be same-shape 2D arrays")
    rgba = np.zeros((*overlap.shape, 4), dtype=np.float32)
    void_rgb = np.asarray((0xF0, 0xA8, 0x30), dtype=np.float32) / 255.0
    overlap_rgb = np.asarray((0xD9, 0xD9, 0xD9), dtype=np.float32) / 255.0
    rgba[void, :3] = void_rgb
    rgba[void, 3] = VOID_ALPHA
    rgba[overlap, :3] = overlap_rgb
    rgba[overlap, 3] = OVERLAP_ALPHA
    return rgba


def _patches(result: PanNukeValidationResult) -> tuple[PatchMaskQC, ...]:
    return tuple(
        sorted(
            (patch for fold in result.fold_validation for patch in fold.mask_qc.patches),
            key=lambda value: (value.fold_id, value.patch_index),
        )
    )


def _instances(patches: Sequence[PatchMaskQC]) -> tuple[MaskInstanceQC, ...]:
    return tuple(instance for patch in patches for instance in patch.affected_instances)


def _validate_patch(patch: PatchMaskQC) -> None:
    key = patch_key(patch.fold_id, patch.patch_index)
    if patch.total_pixel_count != patch.height * patch.width:
        raise MaskQCReportError(f"{key}: total_pixel_count does not equal height*width")
    if (
        patch.void_pixel_count
        + patch.positive_any_pixel_count
        + patch.background_pixel_count
        - patch.positive_and_background_pixel_count
        != patch.total_pixel_count
    ):
        raise MaskQCReportError(f"{key}: positive/background/void counts do not reconcile")
    if patch.has_void != (patch.void_pixel_count > 0):
        raise MaskQCReportError(f"{key}: has_void flag does not match its count")
    if patch.has_cross_class_overlap != (patch.cross_class_overlap_pixel_count > 0):
        raise MaskQCReportError(f"{key}: overlap flag does not match its count")
    if patch.has_positive_and_background != (patch.positive_and_background_pixel_count > 0):
        raise MaskQCReportError(f"{key}: positive+background flag does not match its count")
    anomaly_flags = (
        patch.has_void,
        patch.has_cross_class_overlap,
        patch.has_positive_and_background,
    )
    if (patch.anomaly_union_pixel_count > 0) != any(anomaly_flags):
        raise MaskQCReportError(f"{key}: anomaly-union count does not match anomaly flags")
    if patch.affected_instance_count != len(patch.affected_instances):
        raise MaskQCReportError(f"{key}: affected-instance count does not match records")
    identities: set[tuple[int, int]] = set()
    for instance in patch.affected_instances:
        identity = (instance.class_index, instance.instance_id)
        if identity in identities:
            raise MaskQCReportError(f"{key}: duplicate affected instance identity {identity}")
        identities.add(identity)
        if instance.fold_id != patch.fold_id or instance.patch_index != patch.patch_index:
            raise MaskQCReportError(f"{key}: affected instance points at another source patch")
        if instance.total_pixel_count <= 0:
            raise MaskQCReportError(f"{key}: affected instance has no source pixels")
        if instance.overlap_pixel_count <= 0 and instance.positive_background_pixel_count <= 0:
            raise MaskQCReportError(f"{key}: affected instance does not touch a recorded anomaly")
    expected_indices = tuple(sorted({item.class_index for item in patch.affected_instances}))
    expected_names = tuple(
        sorted({item.class_name for item in patch.affected_instances}, key=str.casefold)
    )
    if patch.affected_class_indices != expected_indices:
        raise MaskQCReportError(f"{key}: affected class indices do not reconcile")
    if tuple(sorted(patch.affected_class_names, key=str.casefold)) != expected_names:
        raise MaskQCReportError(f"{key}: affected class names do not reconcile")


def _validate_fold(fold: FoldMaskQC) -> None:
    patches = tuple(sorted(fold.patches, key=lambda value: value.patch_index))
    if len(patches) != fold.patch_count:
        raise MaskQCReportError(f"fold {fold.fold_id}: patch count does not match records")
    if tuple(patch.patch_index for patch in patches) != tuple(range(fold.patch_count)):
        raise MaskQCReportError(
            f"fold {fold.fold_id}: patch indices are not complete and contiguous"
        )
    if any(patch.fold_id != fold.fold_id for patch in patches):
        raise MaskQCReportError(f"fold {fold.fold_id}: patch record belongs to another fold")
    for patch in patches:
        _validate_patch(patch)
    patch_sum_fields = (
        "total_pixel_count",
        "positive_any_pixel_count",
        "background_pixel_count",
        "void_pixel_count",
        "cross_class_overlap_pixel_count",
        "positive_and_background_pixel_count",
        "anomaly_union_pixel_count",
        "affected_instance_count",
    )
    for field in patch_sum_fields:
        actual = sum(int(getattr(patch, field)) for patch in patches)
        if actual != int(getattr(fold, field)):
            raise MaskQCReportError(f"fold {fold.fold_id}: {field} does not reconcile")
    flag_fields = (
        ("void_patch_count", "has_void"),
        ("cross_class_overlap_patch_count", "has_cross_class_overlap"),
        ("positive_and_background_patch_count", "has_positive_and_background"),
    )
    for count_field, flag_field in flag_fields:
        actual = sum(bool(getattr(patch, flag_field)) for patch in patches)
        if actual != int(getattr(fold, count_field)):
            raise MaskQCReportError(f"fold {fold.fold_id}: {count_field} does not reconcile")
    union_count = sum(patch.anomaly_union_pixel_count > 0 for patch in patches)
    normal_count = sum(patch.anomaly_union_pixel_count == 0 for patch in patches)
    if union_count != fold.anomaly_union_patch_count or normal_count != fold.normal_patch_count:
        raise MaskQCReportError(
            f"fold {fold.fold_id}: anomaly/normal patch totals do not reconcile"
        )
    if union_count + normal_count != fold.patch_count:
        raise MaskQCReportError(f"fold {fold.fold_id}: patch categories do not cover the fold")
    expected_index_sets = (
        ("void_patch_indices", "has_void"),
        ("cross_class_overlap_patch_indices", "has_cross_class_overlap"),
        ("positive_and_background_patch_indices", "has_positive_and_background"),
    )
    for index_field, flag_field in expected_index_sets:
        expected = tuple(patch.patch_index for patch in patches if getattr(patch, flag_field))
        if tuple(getattr(fold, index_field)) != expected:
            raise MaskQCReportError(f"fold {fold.fold_id}: {index_field} does not reconcile")
    expected_union = tuple(
        patch.patch_index for patch in patches if patch.anomaly_union_pixel_count > 0
    )
    if fold.anomaly_union_patch_indices != expected_union:
        raise MaskQCReportError(f"fold {fold.fold_id}: anomaly_union_patch_indices differ")


def reconcile_mask_qc_result(result: PanNukeValidationResult) -> None:
    """Fail closed unless patch, fold, global, policy, and selection facts agree."""

    fold_qc = tuple(value.mask_qc for value in result.fold_validation)
    if not fold_qc:
        raise MaskQCReportError("mask-QC result contains no official folds")
    fold_ids = tuple(value.fold_id for value in fold_qc)
    if len(set(fold_ids)) != len(fold_ids):
        raise MaskQCReportError("mask-QC result contains duplicate fold IDs")
    discovered_ids = tuple(value.fold_id for value in result.folds)
    if fold_ids != discovered_ids:
        raise MaskQCReportError("discovered folds and fold-QC records are not aligned")
    for facts, qc in zip(result.fold_validation, fold_qc, strict=True):
        if facts.fold_id != qc.fold_id or facts.n_patches != qc.patch_count:
            raise MaskQCReportError(f"fold {facts.fold_id}: array and QC patch counts differ")
        if facts.mask_qc is not qc:
            raise MaskQCReportError(f"fold {facts.fold_id}: inconsistent mask-QC object binding")
        if (
            facts.disconnected_instance_count_full_scan < 0
            or facts.disconnected_patch_count_full_scan < 0
            or facts.disconnected_instance_count_sampled < 0
            or facts.disconnected_instance_count_full_scan > facts.full_scan_instance_count
            or facts.disconnected_patch_count_full_scan > facts.n_patches
        ):
            raise MaskQCReportError(
                f"fold {facts.fold_id}: disconnected-instance QC counts are invalid"
            )
        _validate_fold(qc)
    global_qc = result.global_mask_qc
    if global_qc.fold_ids != fold_ids or global_qc.fold_count != len(fold_qc):
        raise MaskQCReportError("global mask-QC fold coverage does not reconcile")
    for field in _SUM_FIELDS:
        expected = sum(int(getattr(fold, field)) for fold in fold_qc)
        if int(getattr(global_qc, field)) != expected:
            raise MaskQCReportError(f"global mask-QC {field} does not reconcile")
    expected_class_indices = tuple(
        sorted({i for fold in fold_qc for i in fold.affected_class_indices})
    )
    expected_class_names = tuple(
        sorted({name for fold in fold_qc for name in fold.affected_class_names}, key=str.casefold)
    )
    if global_qc.affected_class_indices != expected_class_indices:
        raise MaskQCReportError("global affected class indices do not reconcile")
    if tuple(sorted(global_qc.affected_class_names, key=str.casefold)) != expected_class_names:
        raise MaskQCReportError("global affected class names do not reconcile")
    policy = result.qc_policy
    if policy.supplied_background_is_exact_complement_required:
        raise MaskQCReportError(
            "QC policy incorrectly requires supplied background complementarity"
        )
    if not policy.no_class_arbitration or policy.source_masks_modified:
        raise MaskQCReportError("QC policy permits class arbitration or source-mask modification")
    if policy.release_annotation_anomalies_are_fatal or not policy.structural_invalidity_is_fatal:
        raise MaskQCReportError("QC policy does not distinguish anomalies from invalid structure")
    if (
        policy.disconnected_instance_ids_are_fatal
        or not policy.disconnected_instance_definition.strip()
        or not policy.disconnected_instance_action.strip()
    ):
        raise MaskQCReportError("QC policy does not preserve disconnected raw instance IDs")
    if not policy.applies_identically_to_primary_and_confirmatory:
        raise MaskQCReportError("QC exclusion policy differs between analysis families")
    if not policy.analysis_instance_exclusion_reason.strip():
        raise MaskQCReportError("QC policy lacks the fixed shared analysis exclusion reason")
    _validate_selection(result.anomaly_overlay_selection, _patches(result))


def _validate_selection(selection: AnomalyOverlaySelection, patches: Sequence[PatchMaskQC]) -> None:
    if selection.requested_max_patches <= 0:
        raise MaskQCReportError("overlay selection requested a non-positive maximum")
    if len(selection.selected_patch_keys) > selection.requested_max_patches:
        raise MaskQCReportError("overlay selection exceeds its declared maximum")
    if len(set(selection.selected_patch_keys)) != len(selection.selected_patch_keys):
        raise MaskQCReportError("overlay selection contains duplicate patch keys")
    lookup = {(item.fold_id, item.patch_index): item for item in patches}
    selected = set(selection.selected_patch_keys)
    for key in selection.selected_patch_keys:
        identity = _parse_patch_key(key)
        if identity not in lookup:
            raise MaskQCReportError(f"overlay selection references an unknown patch: {key}")
    category_flags = {
        "cross_class_overlap": "has_cross_class_overlap",
        "void_unlabelled": "has_void",
        "positive_and_background": "has_positive_and_background",
    }
    category_union: set[str] = set()
    for category, keys in selection.selected_by_category.items():
        if category not in {*category_flags, "normal"}:
            raise MaskQCReportError(f"overlay selection has unknown category: {category}")
        if len(set(keys)) != len(keys):
            raise MaskQCReportError(f"overlay selection category {category} has duplicates")
        for key in keys:
            if key not in selected:
                raise MaskQCReportError(f"category {category} contains an unselected patch: {key}")
            patch = lookup[_parse_patch_key(key)]
            exhibits_category = (
                patch.anomaly_union_pixel_count == 0
                if category == "normal"
                else bool(getattr(patch, category_flags[category]))
            )
            if not exhibits_category:
                raise MaskQCReportError(f"selected patch {key} does not exhibit {category}")
        category_union.update(keys)
        candidate_count = sum(
            item.anomaly_union_pixel_count == 0
            if category == "normal"
            else bool(getattr(item, category_flags[category]))
            for item in patches
        )
        if selection.category_candidate_counts.get(category) != candidate_count:
            raise MaskQCReportError(f"overlay candidate count differs for {category}")
    if category_union != selected:
        raise MaskQCReportError("selected patches are not exactly covered by anomaly categories")


def _selection_categories(selection: AnomalyOverlaySelection, key: str) -> tuple[str, ...]:
    return tuple(
        category
        for category, values in sorted(selection.selected_by_category.items())
        if key in values
    )


def _raw_selected_patches(
    result: PanNukeValidationResult,
) -> tuple[
    tuple[_SelectedPatchEvidence, NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_]],
    ...,
]:
    patch_lookup = {(item.fold_id, item.patch_index): item for item in _patches(result)}
    fold_lookup = {item.fold_id: item for item in result.folds}
    background_by_fold = result.qc_policy.background_channel_index_by_fold
    rows: list[
        tuple[_SelectedPatchEvidence, NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_]]
    ] = []
    opened: dict[Path, NDArray[np.generic]] = {}

    def mmap(path: Path) -> NDArray[np.generic]:
        if path not in opened:
            opened[path] = open_npy_mmap(path)
        return opened[path]

    for key in result.anomaly_overlay_selection.selected_patch_keys:
        fold_id, patch_index = _parse_patch_key(key)
        fold = fold_lookup[fold_id]
        patch_qc = patch_lookup[(fold_id, patch_index)]
        image = _channel_last_patch(mmap(fold.image_path), patch_index, fold.image_channel_axis)
        raw_mask = _channel_last_patch(mmap(fold.mask_path), patch_index, fold.mask_channel_axis)
        positive_indices = result.qc_policy.positive_channel_indices
        if not positive_indices or max(positive_indices) >= raw_mask.shape[-1]:
            raise MaskQCReportError(f"{key}: positive-channel policy exceeds the raw mask shape")
        positive = np.greater(raw_mask[..., list(positive_indices)], 0)
        positive_any = np.any(positive, axis=-1)
        overlap = np.sum(positive, axis=-1, dtype=np.int16) > 1
        background_index = background_by_fold.get(str(fold_id))
        if background_index is None:
            background = np.zeros(positive_any.shape, dtype=bool)
        else:
            if background_index < 0 or background_index >= raw_mask.shape[-1]:
                raise MaskQCReportError(f"{key}: background-channel policy is out of bounds")
            background = np.greater(raw_mask[..., background_index], 0)
        void = np.logical_not(np.logical_or(positive_any, background))
        positive_background = np.logical_and(positive_any, background)
        observed = (
            int(np.count_nonzero(overlap)),
            int(np.count_nonzero(void)),
            int(np.count_nonzero(positive_background)),
        )
        expected = (
            patch_qc.cross_class_overlap_pixel_count,
            patch_qc.void_pixel_count,
            patch_qc.positive_and_background_pixel_count,
        )
        if observed != expected:
            raise MaskQCReportError(
                f"{key}: raw anomaly masks differ from validated QC counts; "
                f"observed={observed}, expected={expected}"
            )
        anomaly_union = np.logical_or.reduce((void, overlap, positive_background))
        flag_masks = {
            "positive_any": positive_any,
            "supplied_background": background,
            "void_unlabelled": void,
            "cross_class_overlap": overlap,
            "positive_and_background": positive_background,
            "anomaly_union": anomaly_union,
        }
        recomputed_qc_hashes = {
            kind: _qc_flag_mask_sha256(kind, mask) for kind, mask in flag_masks.items()
        }
        if set(patch_qc.mask_sha256_by_kind) != set(recomputed_qc_hashes):
            raise MaskQCReportError(f"{key}: stored QC mask-hash kinds differ from policy")
        for kind, expected_hash in patch_qc.mask_sha256_by_kind.items():
            if expected_hash != recomputed_qc_hashes[kind]:
                raise MaskQCReportError(f"{key}: stored {kind} mask hash differs from raw data")
        evidence = _SelectedPatchEvidence(
            patch_key=key,
            fold_id=fold_id,
            patch_index=patch_index,
            categories=_selection_categories(result.anomaly_overlay_selection, key),
            image_patch_sha256=_array_sha256(image),
            raw_mask_patch_sha256=_array_sha256(raw_mask),
            overlap_mask_sha256=_array_sha256(overlap),
            void_mask_sha256=_array_sha256(void),
            positive_and_background_mask_sha256=_array_sha256(positive_background),
            overlap_pixel_count=observed[0],
            void_pixel_count=observed[1],
            positive_and_background_pixel_count=observed[2],
        )
        rows.append((evidence, _normalise_rgb(image), overlap, void))
    return tuple(rows)


def _render_overlay_png(
    selected: Sequence[
        tuple[_SelectedPatchEvidence, NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_]]
    ],
) -> bytes:
    count = max(1, len(selected))
    columns = min(3, count)
    rows = (count + columns - 1) // columns
    figure = Figure(figsize=(4.2 * columns, 4.0 * rows), dpi=120)
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False)
    for axis in axes.flat:
        axis.set_axis_off()
    if not selected:
        axes.flat[0].text(
            0.5,
            0.5,
            "No overlap/void patch selected",
            ha="center",
            va="center",
            transform=axes.flat[0].transAxes,
        )
    for axis, (evidence, image, overlap, void) in zip(axes.flat, selected, strict=False):
        axis.imshow(image, interpolation="nearest")
        axis.imshow(anomaly_overlay_rgba(overlap, void), interpolation="nearest")
        axis.set_title(
            f"{evidence.patch_key}\noverlap={evidence.overlap_pixel_count:,}; "
            f"void={evidence.void_pixel_count:,}",
            fontsize=8,
        )
        axis.set_axis_off()
    figure.legend(
        handles=(
            LegendPatch(
                facecolor=OVERLAP_COLOUR_HEX,
                edgecolor="#555555",
                label="cross-class overlap (neutral; no class arbitration)",
            ),
            LegendPatch(facecolor=VOID_COLOUR_HEX, edgecolor="#8a5b00", label="unlabeled / void"),
        ),
        loc="lower center",
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    figure.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.10, wspace=0.04, hspace=0.16)
    buffer = io.BytesIO()
    try:
        figure.savefig(
            buffer,
            format="png",
            dpi=120,
            metadata={"Software": "histo-audit PanNuke mask-QC reporter"},
        )
        return buffer.getvalue()
    finally:
        figure.clear()


def _csv_bytes(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _patch_rows(patches: Sequence[PatchMaskQC]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for patch in patches:
        rows.append(
            {
                "patch_key": patch_key(patch.fold_id, patch.patch_index),
                "fold_id": patch.fold_id,
                "patch_index": patch.patch_index,
                "height": patch.height,
                "width": patch.width,
                "total_pixel_count": patch.total_pixel_count,
                "positive_any_pixel_count": patch.positive_any_pixel_count,
                "background_pixel_count": patch.background_pixel_count,
                "void_pixel_count": patch.void_pixel_count,
                "cross_class_overlap_pixel_count": patch.cross_class_overlap_pixel_count,
                "positive_and_background_pixel_count": (patch.positive_and_background_pixel_count),
                "anomaly_union_pixel_count": patch.anomaly_union_pixel_count,
                "affected_instance_count": patch.affected_instance_count,
                "affected_class_indices": json.dumps(list(patch.affected_class_indices)),
                "affected_class_names": json.dumps(
                    list(patch.affected_class_names), ensure_ascii=False
                ),
                "has_void": str(patch.has_void).lower(),
                "has_cross_class_overlap": str(patch.has_cross_class_overlap).lower(),
                "has_positive_and_background": str(patch.has_positive_and_background).lower(),
                "mask_sha256_by_kind": json.dumps(
                    patch.mask_sha256_by_kind, ensure_ascii=False, sort_keys=True
                ),
            }
        )
    return rows


def _instance_rows(
    instances: Sequence[MaskInstanceQC], exclusion_reason: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in instances:
        touches_overlap = item.overlap_pixel_count > 0
        rows.append(
            {
                "patch_key": patch_key(item.fold_id, item.patch_index),
                "fold_id": item.fold_id,
                "patch_index": item.patch_index,
                "class_index": item.class_index,
                "class_name": item.class_name,
                "channel_index": item.channel_index,
                "instance_id": item.instance_id,
                "total_pixel_count": item.total_pixel_count,
                "overlap_pixel_count": item.overlap_pixel_count,
                "positive_background_pixel_count": item.positive_background_pixel_count,
                "overlapping_class_indices": json.dumps(list(item.overlapping_class_indices)),
                "overlapping_instance_ids": json.dumps(list(item.overlapping_instance_ids)),
                "overlapping_instance_ids_by_class": json.dumps(
                    item.overlapping_instance_ids_by_class,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "touches_cross_class_overlap": str(touches_overlap).lower(),
                "analysis_eligible": str(not touches_overlap).lower(),
                "primary_eligible": str(not touches_overlap).lower(),
                "confirmatory_eligible": str(not touches_overlap).lower(),
                "analysis_exclusion_reason": exclusion_reason if touches_overlap else "",
            }
        )
    return rows


def _markdown_report(
    result: PanNukeValidationResult,
    *,
    selection_sha256: str,
    selected: Sequence[_SelectedPatchEvidence],
) -> str:
    qc = result.global_mask_qc
    lines = [
        "# PanNuke mask QC",
        "",
        "**QC status:** release-level annotation anomalies recorded; structural gate valid.",
        "",
        "This read-only report identifies potentially inconsistent annotations and patches "
        "recommended for expert review. It does not adjudicate a class, modify a source mask, "
        "or treat model disagreement as proof of annotation error.",
        "",
        "## Fixed interpretation",
        "",
        "- Positive channels are evaluated independently from the supplied background channel.",
        "- Pixels with neither a positive assignment nor supplied background remain unlabeled/void.",
        "- Cross-class overlap is retained and rendered in one neutral colour; no class arbitration "
        "is performed.",
        f"- Shared primary/confirmatory overlap-touching exclusion reason: "
        f"`{result.qc_policy.analysis_instance_exclusion_reason}`.",
        "- The same eligibility mask applies to primary and confirmatory analyses.",
        "- Disconnected raw instance IDs are retained and quality-flagged without splitting or "
        "repair; their primary/confirmatory eligibility is frozen only after the pilot and "
        "without final-reference outcomes.",
        "- Raw source masks modified: `false`.",
        "",
        "## Release totals",
        "",
        f"- Folds / source patches / pixels: {qc.fold_count:,} / {qc.patch_count:,} / "
        f"{qc.total_pixel_count:,}",
        f"- Cross-class-overlap pixels / patches: {qc.cross_class_overlap_pixel_count:,} / "
        f"{qc.cross_class_overlap_patch_count:,}",
        f"- Unlabeled/void pixels / patches: {qc.void_pixel_count:,} / {qc.void_patch_count:,}",
        f"- Positive-and-supplied-background pixels / patches: "
        f"{qc.positive_and_background_pixel_count:,} / "
        f"{qc.positive_and_background_patch_count:,}",
        f"- Overlap-touching instances excluded from primary and confirmatory analyses: "
        f"{qc.overlap_touching_instance_count:,}",
        f"- Disconnected raw instance IDs / affected patches: "
        f"{sum(value.disconnected_instance_count_full_scan for value in result.fold_validation):,} / "
        f"{sum(value.disconnected_patch_count_full_scan for value in result.fold_validation):,}",
        "",
        "## Fold reconciliation",
        "",
        "| Fold | Patches | Overlap pixels | Overlap patches | Void pixels | Void patches | "
        "Overlap-touching instances | Disconnected IDs | Disconnected patches |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in result.fold_validation:
        value = fold.mask_qc
        lines.append(
            f"| {value.fold_id} | {value.patch_count} | "
            f"{value.cross_class_overlap_pixel_count} | "
            f"{value.cross_class_overlap_patch_count} | {value.void_pixel_count} | "
            f"{value.void_patch_count} | {value.overlap_touching_instance_count} | "
            f"{fold.disconnected_instance_count_full_scan} | "
            f"{fold.disconnected_patch_count_full_scan} |"
        )
    lines.extend(
        [
            "",
            "## Deterministic anomaly overlay",
            "",
            f"- Selection strategy: `{result.anomaly_overlay_selection.strategy}`",
            f"- Selected unique source patches: {len(selected)}",
            f"- Selection SHA-256: `{selection_sha256}`",
            f"- Patch IDs: {', '.join(f'`{item.patch_key}`' for item in selected) or 'none'}",
            "- Grey pixels denote cross-class overlap without selecting a winning class; amber "
            "pixels denote unlabeled/void regions.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_record(content: bytes) -> dict[str, Any]:
    return {"size_bytes": len(content), "sha256": _bytes_sha256(content)}


def _build_bundle_content(
    result: PanNukeValidationResult,
    *,
    max_overlay_patches: int,
) -> tuple[dict[str, bytes], str, str, int, int]:
    reconcile_mask_qc_result(result)
    selection = result.anomaly_overlay_selection
    if max_overlay_patches != selection.requested_max_patches:
        raise MaskQCReportError(
            "report max_overlay_patches differs from the validated deterministic selection: "
            f"report={max_overlay_patches}, selection={selection.requested_max_patches}"
        )
    patches = _patches(result)
    instances = _instances(patches)
    raw_selected = _raw_selected_patches(result)
    selected_evidence = tuple(item[0] for item in raw_selected)
    selection_payload = selection.as_dict()
    selection_sha256 = _canonical_sha256(selection_payload)
    overlay_bytes = _render_overlay_png(raw_selected)
    overlay_sha256 = _bytes_sha256(overlay_bytes)
    selection_document = {
        "schema_version": QC_REPORT_SCHEMA_VERSION,
        "selection": selection_payload,
        "selection_sha256": selection_sha256,
        "selected_patch_evidence": [item.as_dict() for item in selected_evidence],
        "rendering": {
            "cross_class_overlap_colour": OVERLAP_COLOUR_HEX,
            "cross_class_overlap_alpha": OVERLAP_ALPHA,
            "cross_class_overlap_semantics": "neutral_overlay_no_class_arbitration",
            "void_colour": VOID_COLOUR_HEX,
            "void_alpha": VOID_ALPHA,
            "void_semantics": "unlabeled_not_background",
            "positive_class_winner_encoded": False,
        },
        "overlay_sha256": overlay_sha256,
    }
    patch_rows = _patch_rows(patches)
    instance_rows = _instance_rows(instances, result.qc_policy.analysis_instance_exclusion_reason)
    content: dict[str, bytes] = {
        _PATCH_CSV: _csv_bytes(_PATCH_FIELDS, patch_rows),
        _INSTANCE_CSV: _csv_bytes(_INSTANCE_FIELDS, instance_rows),
        _MARKDOWN: _markdown_report(
            result,
            selection_sha256=selection_sha256,
            selected=selected_evidence,
        ).encode("utf-8"),
        _OVERLAY_PNG: overlay_bytes,
        _SELECTION_JSON: _canonical_json_bytes(selection_document, pretty=True),
    }
    report_payload = {
        "schema_version": QC_REPORT_SCHEMA_VERSION,
        "report_kind": "pannuke_mask_qc",
        "status": "structurally_valid_release_anomalies_reported",
        "source_root": str(result.root),
        "source_masks_modified": False,
        "class_mapping": result.mapping.as_dict(),
        "qc_policy": result.qc_policy.as_dict(),
        "global_mask_qc": result.global_mask_qc.as_dict(),
        "fold_mask_qc": [value.mask_qc.as_dict() for value in result.fold_validation],
        "disconnected_instance_qc": {
            "connectivity": "4_connected",
            "source_masks_modified": False,
            "instances_split_or_repaired": False,
            "analysis_eligibility_status": "to_be_frozen_after_pilot_without_final_outcomes",
            "instance_count": sum(
                value.disconnected_instance_count_full_scan for value in result.fold_validation
            ),
            "affected_patch_count": sum(
                value.disconnected_patch_count_full_scan for value in result.fold_validation
            ),
            "by_fold": [
                {
                    "fold_id": value.fold_id,
                    "instance_count": value.disconnected_instance_count_full_scan,
                    "affected_patch_count": value.disconnected_patch_count_full_scan,
                }
                for value in result.fold_validation
            ],
        },
        "anomaly_overlay_selection": selection_payload,
        "selection_sha256": selection_sha256,
        "overlay_sha256": overlay_sha256,
        "patch_csv_row_count": len(patch_rows),
        "instance_csv_row_count": len(instance_rows),
        "artifacts": {name: _artifact_record(value) for name, value in sorted(content.items())},
    }
    content[_REPORT_JSON] = _canonical_json_bytes(report_payload, pretty=True)
    manifest_payload = {
        "schema_version": QC_REPORT_SCHEMA_VERSION,
        "bundle_kind": "pannuke_mask_qc",
        "files": {name: _artifact_record(content[name]) for name in _CONTENT_NAMES},
        "selection_sha256": selection_sha256,
        "overlay_sha256": overlay_sha256,
    }
    content[_ARTIFACT_MANIFEST_JSON] = _canonical_json_bytes(manifest_payload, pretty=True)
    return content, selection_sha256, overlay_sha256, len(patch_rows), len(instance_rows)


def _existing_names(destination: Path) -> set[str]:
    if not destination.is_dir():
        raise FileExistsError(f"QC report destination exists and is not a directory: {destination}")
    return {item.name for item in destination.iterdir()}


def _artifacts_from_bundle(destination: Path) -> MaskQCReportArtifacts:
    report = json.loads((destination / _REPORT_JSON).read_text(encoding="utf-8"))
    return MaskQCReportArtifacts(
        bundle_dir=destination,
        json_path=destination / _REPORT_JSON,
        patch_csv_path=destination / _PATCH_CSV,
        instance_csv_path=destination / _INSTANCE_CSV,
        markdown_path=destination / _MARKDOWN,
        overlay_path=destination / _OVERLAY_PNG,
        overlay_selection_path=destination / _SELECTION_JSON,
        artifact_manifest_path=destination / _ARTIFACT_MANIFEST_JSON,
        selection_sha256=str(report["selection_sha256"]),
        overlay_sha256=str(report["overlay_sha256"]),
        patch_row_count=int(report["patch_csv_row_count"]),
        instance_row_count=int(report["instance_csv_row_count"]),
    )


def _persisted_csv_rows(path: Path, expected_fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(expected_fields):
            raise MaskQCReportError(f"mask-QC CSV schema differs for {path.name}")
        return list(reader)


def _validate_persisted_reconciliation(
    bundle: Path,
    *,
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    selection_document: Mapping[str, Any],
) -> tuple[int, int]:
    """Reconcile persisted JSON and CSV semantics, not just their byte hashes."""

    if report.get("report_kind") != "pannuke_mask_qc" or report.get("status") != (
        "structurally_valid_release_anomalies_reported"
    ):
        raise MaskQCReportError("mask-QC report kind/status differs")
    policy = report.get("qc_policy")
    if not isinstance(policy, dict) or (
        policy.get("source_masks_modified") is not False
        or policy.get("no_class_arbitration") is not True
        or policy.get("supplied_background_is_exact_complement_required") is not False
        or policy.get("release_annotation_anomalies_are_fatal") is not False
        or policy.get("structural_invalidity_is_fatal") is not True
        or policy.get("disconnected_instance_ids_are_fatal") is not False
        or not isinstance(policy.get("disconnected_instance_definition"), str)
        or not str(policy.get("disconnected_instance_definition")).strip()
        or not isinstance(policy.get("disconnected_instance_action"), str)
        or not str(policy.get("disconnected_instance_action")).strip()
        or policy.get("applies_identically_to_primary_and_confirmatory") is not True
    ):
        raise MaskQCReportError("persisted mask-QC policy differs from the fixed policy")
    folds = report.get("fold_mask_qc")
    global_qc = report.get("global_mask_qc")
    if not isinstance(folds, list) or not folds or not isinstance(global_qc, dict):
        raise MaskQCReportError("persisted fold/global mask-QC records are missing")
    if len(folds) != global_qc.get("fold_count"):
        raise MaskQCReportError("persisted global fold count differs")
    fold_ids = [fold.get("fold_id") for fold in folds if isinstance(fold, dict)]
    if len(fold_ids) != len(folds) or fold_ids != global_qc.get("fold_ids"):
        raise MaskQCReportError("persisted global fold IDs differ")
    disconnected = report.get("disconnected_instance_qc")
    if (
        not isinstance(disconnected, dict)
        or disconnected.get("connectivity") != "4_connected"
        or disconnected.get("source_masks_modified") is not False
        or disconnected.get("instances_split_or_repaired") is not False
        or disconnected.get("analysis_eligibility_status")
        != "to_be_frozen_after_pilot_without_final_outcomes"
        or not isinstance(disconnected.get("by_fold"), list)
    ):
        raise MaskQCReportError("persisted disconnected-instance QC evidence is malformed")
    disconnected_by_fold = disconnected["by_fold"]
    if [
        item.get("fold_id") for item in disconnected_by_fold if isinstance(item, dict)
    ] != fold_ids or len(disconnected_by_fold) != len(fold_ids):
        raise MaskQCReportError("persisted disconnected-instance fold coverage differs")
    try:
        disconnected_instances = sum(int(item["instance_count"]) for item in disconnected_by_fold)
        disconnected_patches = sum(
            int(item["affected_patch_count"]) for item in disconnected_by_fold
        )
    except (KeyError, TypeError, ValueError) as error:
        raise MaskQCReportError(
            "persisted disconnected-instance fold counts are malformed"
        ) from error
    if (
        disconnected_instances < 0
        or disconnected_patches < 0
        or disconnected.get("instance_count") != disconnected_instances
        or disconnected.get("affected_patch_count") != disconnected_patches
    ):
        raise MaskQCReportError("persisted disconnected-instance totals do not reconcile")
    for field in _SUM_FIELDS:
        try:
            expected = sum(int(fold[field]) for fold in folds)
            observed = int(global_qc[field])
        except (KeyError, TypeError, ValueError) as error:
            raise MaskQCReportError(f"persisted mask-QC field is malformed: {field}") from error
        if observed != expected:
            raise MaskQCReportError(f"persisted global mask-QC {field} does not reconcile")

    persisted_patches: dict[str, Mapping[str, Any]] = {}
    persisted_instances: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for fold in folds:
        if not isinstance(fold, dict) or not isinstance(fold.get("patches"), list):
            raise MaskQCReportError("persisted fold mask-QC patch records are malformed")
        fold_id = int(fold["fold_id"])
        patch_values = fold["patches"]
        if len(patch_values) != int(fold["patch_count"]):
            raise MaskQCReportError(f"persisted fold {fold_id} patch count differs")
        for expected_index, patch in enumerate(patch_values):
            if not isinstance(patch, dict):
                raise MaskQCReportError("persisted patch mask-QC record is malformed")
            if (
                int(patch.get("fold_id", -1)) != fold_id
                or int(patch.get("patch_index", -1)) != expected_index
            ):
                raise MaskQCReportError(f"persisted fold {fold_id} patch order differs")
            key = patch_key(fold_id, expected_index)
            if key in persisted_patches:
                raise MaskQCReportError(f"persisted duplicate patch key: {key}")
            persisted_patches[key] = patch
            instances = patch.get("affected_instances")
            if not isinstance(instances, list) or len(instances) != int(
                patch.get("affected_instance_count", -1)
            ):
                raise MaskQCReportError(f"persisted affected-instance records differ for {key}")
            for instance in instances:
                if not isinstance(instance, dict):
                    raise MaskQCReportError("persisted affected-instance record is malformed")
                identity = (key, int(instance["class_index"]), int(instance["instance_id"]))
                if identity in persisted_instances:
                    raise MaskQCReportError(f"persisted duplicate affected instance: {identity}")
                persisted_instances[identity] = instance
    if len(persisted_patches) != int(global_qc["patch_count"]):
        raise MaskQCReportError("persisted patch records do not cover the release")

    patch_rows = _persisted_csv_rows(bundle / _PATCH_CSV, _PATCH_FIELDS)
    if len(patch_rows) != len(persisted_patches):
        raise MaskQCReportError("mask-QC patch CSV coverage differs from JSON")
    numeric_patch_fields = (
        "fold_id",
        "patch_index",
        "height",
        "width",
        "total_pixel_count",
        "positive_any_pixel_count",
        "background_pixel_count",
        "void_pixel_count",
        "cross_class_overlap_pixel_count",
        "positive_and_background_pixel_count",
        "anomaly_union_pixel_count",
        "affected_instance_count",
    )
    for row in patch_rows:
        key = row["patch_key"]
        patch = persisted_patches.get(key)
        if patch is None:
            raise MaskQCReportError(f"mask-QC patch CSV has an unknown key: {key}")
        if any(int(row[field]) != int(patch[field]) for field in numeric_patch_fields):
            raise MaskQCReportError(f"mask-QC patch CSV values differ for {key}")
        boolean_fields = (
            "has_void",
            "has_cross_class_overlap",
            "has_positive_and_background",
        )
        if any(row[field] != str(bool(patch[field])).lower() for field in boolean_fields):
            raise MaskQCReportError(f"mask-QC patch CSV flags differ for {key}")

    instance_rows = _persisted_csv_rows(bundle / _INSTANCE_CSV, _INSTANCE_FIELDS)
    if len(instance_rows) != len(persisted_instances):
        raise MaskQCReportError("mask-QC instance CSV coverage differs from JSON")
    exclusion_reason = str(policy.get("analysis_instance_exclusion_reason", ""))
    if not exclusion_reason:
        raise MaskQCReportError("persisted shared analysis exclusion reason is missing")
    for row in instance_rows:
        identity = (row["patch_key"], int(row["class_index"]), int(row["instance_id"]))
        instance = persisted_instances.get(identity)
        if instance is None:
            raise MaskQCReportError(f"mask-QC instance CSV has an unknown identity: {identity}")
        touches_overlap = int(instance["overlap_pixel_count"]) > 0
        eligibility = str(not touches_overlap).lower()
        if (
            row["touches_cross_class_overlap"] != str(touches_overlap).lower()
            or row["analysis_eligible"] != eligibility
            or row["primary_eligible"] != eligibility
            or row["confirmatory_eligible"] != eligibility
            or row["analysis_exclusion_reason"] != (exclusion_reason if touches_overlap else "")
        ):
            raise MaskQCReportError(f"mask-QC shared eligibility differs for {identity}")

    selection_payload = selection_document.get("selection")
    if report.get("anomaly_overlay_selection") != selection_payload:
        raise MaskQCReportError("mask-QC report and overlay selection JSON differ")
    if not isinstance(selection_payload, dict) or not isinstance(
        selection_payload.get("selected_patch_keys"), list
    ):
        raise MaskQCReportError("persisted overlay selection is malformed")
    selected_keys = selection_payload["selected_patch_keys"]
    evidence = selection_document.get("selected_patch_evidence")
    if (
        not isinstance(evidence, list)
        or [item.get("patch_key") for item in evidence] != selected_keys
    ):
        raise MaskQCReportError("persisted overlay evidence does not cover selection in order")
    for item in evidence:
        if not isinstance(item, dict) or item.get("patch_key") not in persisted_patches:
            raise MaskQCReportError("persisted overlay evidence references an unknown patch")
        for name in (
            "image_patch_sha256",
            "raw_mask_patch_sha256",
            "overlap_mask_sha256",
            "void_mask_sha256",
            "positive_and_background_mask_sha256",
        ):
            value = item.get(name)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise MaskQCReportError(f"persisted overlay evidence has invalid {name}")

    files = manifest.get("files")
    report_artifacts = report.get("artifacts")
    if not isinstance(files, dict) or not isinstance(report_artifacts, dict):
        raise MaskQCReportError("persisted mask-QC artifact bindings are missing")
    expected_report_artifacts = {
        name: files[name] for name in _CONTENT_NAMES if name != _REPORT_JSON
    }
    if report_artifacts != expected_report_artifacts:
        raise MaskQCReportError("mask-QC report artifact bindings differ from manifest")
    return len(patch_rows), len(instance_rows)


def validate_mask_qc_report_bundle(destination: str | Path) -> MaskQCReportArtifacts:
    """Verify exact bundle membership, every file hash, and cross-file bindings."""

    supplied = Path(destination).expanduser()
    if not supplied.is_absolute():
        supplied = Path.cwd() / supplied
    bundle = supplied.parent.resolve() / supplied.name
    if not os.path.lexists(bundle) or not bundle.is_dir() or bundle.is_symlink():
        raise MaskQCReportError(f"mask-QC report bundle does not exist: {bundle}")
    names = _existing_names(bundle)
    expected = set(_REQUIRED_NAMES)
    if names != expected:
        raise MaskQCReportError(
            "mask-QC report bundle is partial or has unexpected files: "
            f"missing={sorted(expected - names)}, unexpected={sorted(names - expected)}"
        )
    try:
        manifest = json.loads((bundle / _ARTIFACT_MANIFEST_JSON).read_text(encoding="utf-8"))
        report = json.loads((bundle / _REPORT_JSON).read_text(encoding="utf-8"))
        selection = json.loads((bundle / _SELECTION_JSON).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise MaskQCReportError("mask-QC report JSON is unreadable or malformed") from error
    if manifest.get("schema_version") != QC_REPORT_SCHEMA_VERSION:
        raise MaskQCReportError("mask-QC artifact manifest schema version differs")
    file_records = manifest.get("files")
    if not isinstance(file_records, dict) or set(file_records) != set(_CONTENT_NAMES):
        raise MaskQCReportError("mask-QC artifact manifest file coverage differs")
    for name in _CONTENT_NAMES:
        record = file_records[name]
        path = bundle / name
        if not isinstance(record, dict):
            raise MaskQCReportError(f"invalid artifact record for {name}")
        if record.get("size_bytes") != path.stat().st_size or record.get("sha256") != sha256_file(
            path
        ):
            raise MaskQCReportError(f"mask-QC artifact hash/size differs for {name}")
    if report.get("schema_version") != QC_REPORT_SCHEMA_VERSION:
        raise MaskQCReportError("mask-QC report schema version differs")
    if report.get("source_masks_modified") is not False:
        raise MaskQCReportError("mask-QC report claims source masks were modified")
    selection_payload = selection.get("selection")
    if not isinstance(selection_payload, dict):
        raise MaskQCReportError("mask-QC selection payload is missing")
    selection_sha = _canonical_sha256(selection_payload)
    if not (
        selection.get("selection_sha256")
        == report.get("selection_sha256")
        == manifest.get("selection_sha256")
        == selection_sha
    ):
        raise MaskQCReportError("mask-QC selection SHA-256 bindings differ")
    overlay_sha = sha256_file(bundle / _OVERLAY_PNG)
    if not (
        selection.get("overlay_sha256")
        == report.get("overlay_sha256")
        == manifest.get("overlay_sha256")
        == overlay_sha
    ):
        raise MaskQCReportError("mask-QC overlay SHA-256 bindings differ")
    rendering = selection.get("rendering")
    if not isinstance(rendering, dict) or (
        rendering.get("cross_class_overlap_semantics") != "neutral_overlay_no_class_arbitration"
        or rendering.get("positive_class_winner_encoded") is not False
    ):
        raise MaskQCReportError("mask-QC overlap rendering permits class arbitration")
    patch_count, instance_count = _validate_persisted_reconciliation(
        bundle,
        manifest=manifest,
        report=report,
        selection_document=selection,
    )
    if patch_count != report.get("patch_csv_row_count"):
        raise MaskQCReportError("mask-QC patch CSV row count differs")
    if instance_count != report.get("instance_csv_row_count"):
        raise MaskQCReportError("mask-QC instance CSV row count differs")
    return _artifacts_from_bundle(bundle)


def _write_staged_bundle(destination: Path, content: Mapping[str, bytes]) -> list[PublishedPath]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    )
    try:
        for name in _REQUIRED_NAMES:
            path = staging / name
            with path.open("wb") as handle:
                handle.write(content[name])
                handle.flush()
                os.fsync(handle.fileno())
        return publish_flat_directory_no_overwrite(
            staging,
            destination,
            success_marker_name=_ARTIFACT_MANIFEST_JSON,
        )
    finally:
        for path in staging.iterdir() if staging.exists() else ():
            path.unlink(missing_ok=True)
        if staging.exists():
            staging.rmdir()


def _guard_destination_outside_raw_release(
    result: PanNukeValidationResult, destination: str | Path
) -> Path:
    """Resolve traversal/symlinks and reject every destination within raw data."""

    raw_root = Path(result.root).resolve()
    supplied = Path(destination).expanduser()
    if not supplied.is_absolute():
        supplied = Path.cwd() / supplied
    fully_resolved = supplied.resolve()
    bundle = supplied.parent.resolve() / supplied.name
    inside_raw = fully_resolved == raw_root or raw_root in fully_resolved.parents
    contains_raw = fully_resolved == raw_root or fully_resolved in raw_root.parents
    if inside_raw or contains_raw:
        raise MaskQCReportError(
            "mask-QC report destination must be outside the immutable raw release and disjoint "
            "from it: "
            f"destination={bundle}, raw_root={raw_root}"
        )
    return bundle


def _require_exact_qc_bundle_content(bundle: Path, content: Mapping[str, bytes]) -> None:
    """Require one regular, non-symlink, byte-exact immutable QC bundle."""

    if not os.path.lexists(bundle) or not bundle.is_dir() or bundle.is_symlink():
        raise MaskQCReportError(f"mask-QC report bundle does not exist: {bundle}")
    names = _existing_names(bundle)
    expected = set(_REQUIRED_NAMES)
    if names != expected:
        raise MaskQCReportError(
            "mask-QC report bundle is partial or has unexpected files: "
            f"missing={sorted(expected - names)}, unexpected={sorted(names - expected)}"
        )
    for name in _REQUIRED_NAMES:
        path = bundle / name
        if path.is_symlink() or not path.is_file():
            raise MaskQCReportError(f"mask-QC artifact is not a regular file: {path}")
        if path.read_bytes() != content[name]:
            raise FileExistsError(
                f"refusing to overwrite a differing immutable mask-QC artifact: {path}"
            )


def write_mask_qc_report_bundle(
    result: PanNukeValidationResult,
    destination: str | Path,
    *,
    max_overlay_patches: int = 24,
) -> MaskQCReportArtifacts:
    """Atomically create or idempotently verify an immutable mask-QC bundle.

    A complete byte-identical bundle is accepted on rerun.  A partial bundle or
    any differing existing byte fails closed and is never overwritten.
    """

    if max_overlay_patches <= 0:
        raise ValueError("max_overlay_patches must be positive")
    # This guard intentionally precedes report construction and every mkdir/write.
    bundle = _guard_destination_outside_raw_release(result, destination)
    with ExclusiveBundlePublicationLock((bundle.parent, bundle), role="PanNuke mask-QC bundle"):
        _verify_report_raw_inventory(result)
        content, selection_sha, overlay_sha, patch_count, instance_count = _build_bundle_content(
            result, max_overlay_patches=max_overlay_patches
        )
        if os.path.lexists(bundle):
            validate_mask_qc_report_bundle(bundle)
            _require_exact_qc_bundle_content(bundle, content)
            _verify_report_raw_inventory(result)
            validate_mask_qc_report_bundle(bundle)
            _require_exact_qc_bundle_content(bundle, content)
            return _artifacts_from_bundle(bundle)
        publications: list[PublishedPath] = []
        try:
            publications = _write_staged_bundle(bundle, content)
            _verify_report_raw_inventory(result)
            artifacts = validate_mask_qc_report_bundle(bundle)
            _require_exact_qc_bundle_content(bundle, content)
            if (
                artifacts.selection_sha256 != selection_sha
                or artifacts.overlay_sha256 != overlay_sha
                or artifacts.patch_row_count != patch_count
                or artifacts.instance_row_count != instance_count
            ):
                raise MaskQCReportError(
                    "new mask-QC bundle did not reconcile after atomic publication"
                )
            return artifacts
        except BaseException as publish_error:
            if publications:
                try:
                    rollback_owned_publications(publications)
                except RuntimeError:
                    raise RuntimeError(
                        "mask-QC publication failed and ownership-safe rollback was incomplete"
                    ) from publish_error
            raise


__all__ = [
    "OVERLAP_ALPHA",
    "OVERLAP_COLOUR_HEX",
    "QC_REPORT_SCHEMA_VERSION",
    "VOID_ALPHA",
    "VOID_COLOUR_HEX",
    "MaskQCReportArtifacts",
    "MaskQCReportError",
    "anomaly_overlay_rgba",
    "patch_key",
    "reconcile_mask_qc_result",
    "validate_mask_qc_report_bundle",
    "write_mask_qc_report_bundle",
]
