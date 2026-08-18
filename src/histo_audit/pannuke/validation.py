"""Alignment, semantic, provenance, and visual validation for PanNuke arrays."""

from __future__ import annotations

import csv
import inspect
import io
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from contextlib import suppress
from functools import wraps
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy import ndimage  # type: ignore[import-untyped]

from .discovery import discover_pannuke_release, infer_fold_id
from .exceptions import PanNukeSemanticsError
from .io import (
    atomic_replace_via_temp,
    atomic_write_json,
    atomic_write_text,
    deterministic_sample_indices,
    ensure_output_capacity,
    inventory_raw_files,
    open_npy_mmap,
)
from .models import (
    OFFICIAL_METRICS_CLASS_MAPPING,
    OFFICIAL_RELEASE_FOLD_IDS,
    AnomalyOverlaySelection,
    DiscoveredFold,
    FoldMaskQC,
    FoldValidation,
    GlobalMaskQC,
    MaskQCPolicy,
    PanNukeValidationResult,
    PatchMaskQC,
    RawFileRecord,
    ReleaseDiscovery,
    ValidationArtifacts,
    VerifiedClassMapping,
)
from .publication import (
    ExclusiveBundlePublicationLock,
    PublishedPath,
    publish_file_no_overwrite,
    rollback_owned_publications,
)
from .qc import (
    analyse_patch_mask_qc,
    select_anomaly_overlay_patches,
    summarise_fold_mask_qc,
    summarise_global_mask_qc,
)

_OVERLAY_COLOURS = ("#ff3366", "#00c853", "#2979ff", "#ff9100", "#aa00ff", "#00b8d4")
_INSTANCE_CONNECTIVITY_4 = ndimage.generate_binary_structure(2, 1)


def _require_derived_destination_outside_raw(
    raw_root: Path,
    destination: Path,
    role: str,
    *,
    directory: bool = False,
) -> Path:
    """Resolve a derived destination and reject raw containment before any write."""

    raw = raw_root.resolve()
    fully_resolved = destination.resolve()
    resolved = (
        fully_resolved
        if directory
        else destination.expanduser().absolute().parent.resolve() / destination.name
    )
    inside_raw = fully_resolved == raw or raw in fully_resolved.parents
    contains_raw = directory and (fully_resolved == raw or fully_resolved in raw.parents)
    if inside_raw or contains_raw:
        raise PanNukeSemanticsError(
            f"{role} overlaps the immutable PanNuke raw root: destination={resolved}, raw={raw}"
        )
    return resolved


def _require_distinct_artifact_paths(paths: Sequence[tuple[str, Path]]) -> None:
    by_path: dict[Path, str] = {}
    for role, path in paths:
        resolved = path.resolve()
        previous = by_path.get(resolved)
        if previous is not None:
            raise PanNukeSemanticsError(
                f"derived PanNuke artifact paths alias after resolution: "
                f"{previous} and {role}: {resolved}"
            )
        by_path[resolved] = role


def _require_disjoint_file_artifact_paths(paths: Sequence[tuple[str, Path]]) -> None:
    """Reject file targets that equal, contain, or are contained by another file target."""

    resolved = tuple((role, path.resolve()) for role, path in paths)
    for index, (left_role, left) in enumerate(resolved):
        for right_role, right in resolved[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise PanNukeSemanticsError(
                    "derived PanNuke file artifact paths collide after resolution: "
                    f"{left_role} and {right_role}: {left}, {right}"
                )


def _require_artifact_suffix(path: Path, suffix: str, role: str) -> None:
    if path.suffix.lower() != suffix:
        raise PanNukeSemanticsError(f"{role} must use suffix {suffix}: {path}")


def _resolve_validation_artifact_paths(
    raw_root: Path,
    output_dir: str | Path,
    *,
    raw_inventory_csv_path: str | Path | None,
    overlay_path: str | Path | None,
) -> tuple[Path, Path, Path, Path, Path]:
    """Resolve and preflight every base-validation destination before any build/write."""

    output = _require_derived_destination_outside_raw(
        raw_root,
        Path(output_dir),
        "validation output directory",
        directory=True,
    )
    json_path = output / "pannuke_validation.json"
    markdown_path = output / "pannuke_validation.md"
    resolved_overlay_path = _require_derived_destination_outside_raw(
        raw_root,
        Path(overlay_path) if overlay_path is not None else output / "pannuke_overlay_grid.png",
        "validation overlay",
    )
    resolved_inventory_path = _require_derived_destination_outside_raw(
        raw_root,
        (
            Path(raw_inventory_csv_path)
            if raw_inventory_csv_path is not None
            else output / "raw_files_sha256.csv"
        ),
        "raw inventory CSV",
    )
    artifact_files = (
        ("validation JSON", json_path),
        ("validation Markdown", markdown_path),
        ("validation overlay", resolved_overlay_path),
        ("raw inventory CSV", resolved_inventory_path),
    )
    _require_distinct_artifact_paths((("validation output directory", output), *artifact_files))
    _require_disjoint_file_artifact_paths(artifact_files)
    _require_artifact_suffix(json_path, ".json", "validation JSON")
    _require_artifact_suffix(markdown_path, ".md", "validation Markdown")
    _require_artifact_suffix(resolved_overlay_path, ".png", "validation overlay")
    _require_artifact_suffix(resolved_inventory_path, ".csv", "raw inventory CSV")
    return output, json_path, markdown_path, resolved_overlay_path, resolved_inventory_path


def resolve_class_mapping(
    mapping: VerifiedClassMapping | None = None,
    *,
    use_documented_default: bool = True,
) -> VerifiedClassMapping:
    """Require explicit verified semantics or use the documented official default."""

    if mapping is None:
        if use_documented_default:
            return OFFICIAL_METRICS_CLASS_MAPPING
        raise PanNukeSemanticsError(
            "positive mask-channel semantics cannot be inferred from pixel values; supply "
            "a VerifiedClassMapping with precise provenance or enable the documented "
            "official metrics-repository mapping"
        )
    if not isinstance(mapping, VerifiedClassMapping) or not mapping.verified:
        raise PanNukeSemanticsError(
            "class mapping must be a verified, provenance-bearing VerifiedClassMapping"
        )
    return mapping


def _channel_last_patch(array: np.ndarray, patch_index: int, channel_axis: int) -> np.ndarray:
    patch = np.asarray(array[patch_index])
    patch_axis = channel_axis - 1
    return np.moveaxis(patch, patch_axis, -1) if patch_axis != patch.ndim - 1 else patch


def _channel_last_shape(shape: tuple[int, ...], channel_axis: int) -> tuple[int, int, int, int]:
    axes = [0, 1, 2, 3]
    channel = axes.pop(channel_axis)
    axes.append(channel)
    reordered = tuple(shape[index] for index in axes)
    return reordered  # type: ignore[return-value]


def _bounded_range(
    array: np.ndarray,
    indices: Sequence[int],
    *,
    role: str,
) -> tuple[float, float]:
    minimum = float("inf")
    maximum = float("-inf")
    for index in indices:
        patch = np.asarray(array[index])
        if not np.issubdtype(patch.dtype, np.number):
            raise PanNukeSemanticsError(f"{role} array must be numeric")
        if not np.isfinite(patch).all():
            raise PanNukeSemanticsError(f"{role} array contains non-finite sampled values")
        minimum = min(minimum, float(patch.min()))
        maximum = max(maximum, float(patch.max()))
    if minimum == float("inf"):
        raise PanNukeSemanticsError(f"{role} array has no patches")
    return minimum, maximum


def _instance_ids_and_disconnected_components(
    channel: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    """Find instance IDs and exact 4-connected component counts using dense-label crops.

    PanNuke IDs are raw identifiers and need not be small or contiguous.  We
    therefore remap only the IDs present in this patch to dense temporary labels,
    obtain one bounding box per dense label, and run the original 4-connected
    ``ndimage.label`` check only inside that instance's box. The raw identity is
    never split or repaired; callers decide whether the observed release property
    is fatal or report-only.
    """

    value = np.asarray(channel)
    positive = value > 0
    ids = np.unique(value[positive]).astype(np.int64, copy=False)
    if not len(ids):
        return ids, {}
    dense = np.zeros(value.shape, dtype=np.int32)
    dense[positive] = np.searchsorted(ids, value[positive]).astype(np.int32) + 1
    boxes = ndimage.find_objects(dense, max_label=len(ids))
    if len(boxes) != len(ids):  # pragma: no cover - scipy contract guard
        raise AssertionError("dense instance bounding-box count is inconsistent")
    disconnected: dict[int, int] = {}
    for dense_id, (instance_id, bounding_box) in enumerate(zip(ids, boxes, strict=True), start=1):
        if bounding_box is None:  # pragma: no cover - every dense label is present
            raise AssertionError("present dense instance label lacks a bounding box")
        _, component_count = ndimage.label(
            dense[bounding_box] == dense_id,
            structure=_INSTANCE_CONNECTIVITY_4,
        )
        if component_count != 1:
            disconnected[int(instance_id)] = int(component_count)
    return ids, disconnected


def _background_candidates(
    sampled_masks: Sequence[np.ndarray],
    *,
    extra_channels: tuple[int, ...],
) -> tuple[int, ...]:
    """Return extra channels that are structurally valid supplied-background masks.

    A supplied background channel must be binary.  It is deliberately *not*
    required to equal the complement of positive occupancy, because released
    masks can contain unlabeled/void pixels.
    """

    candidates: list[int] = []
    for channel_index in extra_channels:
        is_candidate = True
        for mask in sampled_masks:
            background = mask[..., channel_index]
            unique = np.unique(background)
            if not set(unique.tolist()).issubset({0, 1}):
                is_candidate = False
                break
        if is_candidate:
            candidates.append(channel_index)
    return tuple(candidates)


def _resolve_background_channel(
    *,
    channel_count: int,
    positive_channels: tuple[int, ...],
    candidates: tuple[int, ...],
    explicit_background_channel: int | None,
    fold_id: int,
) -> int | None:
    extras = tuple(index for index in range(channel_count) if index not in positive_channels)
    if not extras:
        if explicit_background_channel is not None:
            raise PanNukeSemanticsError(
                f"fold {fold_id}: explicit background channel {explicit_background_channel} "
                "does not exist outside the positive channels"
            )
        return None
    if explicit_background_channel is not None:
        if explicit_background_channel not in extras:
            raise PanNukeSemanticsError(
                f"fold {fold_id}: background channel must be one of extra channels {extras}"
            )
        if explicit_background_channel not in candidates:
            raise PanNukeSemanticsError(
                f"fold {fold_id}: supplied background channel "
                f"{explicit_background_channel} is not binary"
            )
        unresolved = tuple(index for index in extras if index != explicit_background_channel)
        if unresolved:
            raise PanNukeSemanticsError(
                f"fold {fold_id}: unexplained extra mask channels remain: {unresolved}"
            )
        return explicit_background_channel
    if len(extras) == 1 and candidates == extras:
        return extras[0]
    raise PanNukeSemanticsError(
        f"fold {fold_id}: mask-channel semantics are ambiguous; positive channels are "
        f"{positive_channels}, extra channels are {extras}, background candidates are "
        f"{candidates}. Supply verified semantics rather than guessing."
    )


def _full_semantic_scan(
    images: np.ndarray,
    masks: np.ndarray,
    *,
    fold: DiscoveredFold,
    mapping: VerifiedClassMapping,
    positive_channels: tuple[int, ...],
    background_channel: int | None,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    dict[str, set[int]],
    int,
    int,
    int,
    FoldMaskQC,
]:
    """Stream every patch before a release can receive global valid status."""

    image_minimum = float("inf")
    image_maximum = float("-inf")
    mask_minimum = float("inf")
    mask_maximum = float("-inf")
    ids_by_class: dict[str, set[int]] = {name: set() for name in mapping.class_names}
    instance_count = 0
    disconnected_instance_count = 0
    disconnected_patch_indices: set[int] = set()
    patch_qc: list[PatchMaskQC] = []
    for patch_index in range(len(images)):
        image = _channel_last_patch(images, patch_index, fold.image_channel_axis)
        mask = _channel_last_patch(masks, patch_index, fold.mask_channel_axis)
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise PanNukeSemanticsError(
                f"fold {fold.fold_id} patch {patch_index}: image values must be finite numeric"
            )
        if not np.issubdtype(mask.dtype, np.number) or not np.isfinite(mask).all():
            raise PanNukeSemanticsError(
                f"fold {fold.fold_id} patch {patch_index}: mask values must be finite numeric"
            )
        image_minimum = min(image_minimum, float(image.min()))
        image_maximum = max(image_maximum, float(image.max()))
        mask_minimum = min(mask_minimum, float(mask.min()))
        mask_maximum = max(mask_maximum, float(mask.max()))
        if image_minimum < 0:
            raise PanNukeSemanticsError(
                f"fold {fold.fold_id} patch {patch_index}: image values are negative"
            )
        if mask_minimum < 0:
            raise PanNukeSemanticsError(
                f"fold {fold.fold_id} patch {patch_index}: instance IDs are negative"
            )
        if not np.equal(mask, np.rint(mask)).all():
            raise PanNukeSemanticsError(
                f"fold {fold.fold_id} patch {patch_index}: mask IDs are not integer-like"
            )
        patch_qc.append(
            analyse_patch_mask_qc(
                mask,
                fold_id=fold.fold_id,
                patch_index=patch_index,
                class_names=mapping.class_names,
                positive_channel_indices=positive_channels,
                background_channel_index=background_channel,
            )
        )
        for class_index, class_name in enumerate(mapping.class_names):
            channel = mask[..., class_index]
            ids, disconnected = _instance_ids_and_disconnected_components(channel)
            ids_by_class[class_name].update(int(value) for value in ids)
            instance_count += len(ids)
            if disconnected:
                disconnected_instance_count += len(disconnected)
                disconnected_patch_indices.add(patch_index)
    if image_minimum == float("inf") or mask_minimum == float("inf"):
        raise PanNukeSemanticsError(f"fold {fold.fold_id}: release contains no patches")
    return (
        (image_minimum, image_maximum),
        (mask_minimum, mask_maximum),
        ids_by_class,
        instance_count,
        disconnected_instance_count,
        len(disconnected_patch_indices),
        summarise_fold_mask_qc(fold.fold_id, patch_qc),
    )


def _validate_fold(
    fold: DiscoveredFold,
    *,
    mapping: VerifiedClassMapping,
    max_samples: int,
    memory_budget_bytes: int,
    explicit_background_channel: int | None,
) -> FoldValidation:
    images = open_npy_mmap(fold.image_path)
    masks = open_npy_mmap(fold.mask_path)
    tissues = open_npy_mmap(fold.tissue_path)
    if images.ndim != 4 or masks.ndim != 4 or tissues.ndim != 1:
        raise PanNukeSemanticsError(
            f"fold {fold.fold_id}: expected image/mask/tissue ranks 4/4/1, found "
            f"{images.ndim}/{masks.ndim}/{tissues.ndim}"
        )
    image_shape = tuple(int(value) for value in images.shape)
    mask_shape = tuple(int(value) for value in masks.shape)
    canonical_image_shape = _channel_last_shape(image_shape, fold.image_channel_axis)
    canonical_mask_shape = _channel_last_shape(mask_shape, fold.mask_channel_axis)
    n_patches, height, width, image_channels = canonical_image_shape
    mask_count, mask_height, mask_width, mask_channels = canonical_mask_shape
    if image_channels not in (3, 4):
        raise PanNukeSemanticsError(
            f"fold {fold.fold_id}: image array must have 3 or 4 channels, found {image_channels}"
        )
    if (mask_count, mask_height, mask_width) != (n_patches, height, width):
        raise PanNukeSemanticsError(
            f"fold {fold.fold_id}: image/mask alignment mismatch after channel-axis "
            f"normalisation: image={canonical_image_shape}, mask={canonical_mask_shape}"
        )
    if len(tissues) != n_patches:
        raise PanNukeSemanticsError(
            f"fold {fold.fold_id}: tissue length {len(tissues)} does not align with "
            f"{n_patches} image patches"
        )
    if not (np.issubdtype(tissues.dtype, np.str_) or np.issubdtype(tissues.dtype, np.bytes_)):
        raise PanNukeSemanticsError(
            f"fold {fold.fold_id}: tissue array dtype {tissues.dtype} lacks explicit names; "
            "numeric tissue semantics require a separate verified mapping"
        )
    if mask_channels < len(mapping.class_names):
        raise PanNukeSemanticsError(
            f"fold {fold.fold_id}: mask has {mask_channels} channels but verified mapping "
            f"contains {len(mapping.class_names)} positive classes"
        )
    positive_channels = tuple(range(len(mapping.class_names)))
    bytes_per_pair = (
        int(np.prod(canonical_image_shape[1:], dtype=np.int64)) * images.dtype.itemsize
        + int(np.prod(canonical_mask_shape[1:], dtype=np.int64)) * masks.dtype.itemsize
    )
    sampled_indices = deterministic_sample_indices(
        n_patches,
        max_samples=max_samples,
        bytes_per_sample=bytes_per_pair,
        memory_budget_bytes=memory_budget_bytes,
    )
    image_range = _bounded_range(images, sampled_indices, role=f"fold {fold.fold_id} image")
    mask_range = _bounded_range(masks, sampled_indices, role=f"fold {fold.fold_id} mask")
    if image_range[0] < 0:
        raise PanNukeSemanticsError(f"fold {fold.fold_id}: sampled image values are negative")
    if mask_range[0] < 0:
        raise PanNukeSemanticsError(f"fold {fold.fold_id}: sampled instance IDs are negative")

    sampled_masks: list[np.ndarray] = []
    ids_by_class: dict[str, set[int]] = {name: set() for name in mapping.class_names}
    overlap_pixels = 0
    sampled_disconnected_instances = 0
    for patch_index in sampled_indices:
        mask = _channel_last_patch(masks, patch_index, fold.mask_channel_axis)
        if not np.equal(mask, np.rint(mask)).all():
            raise PanNukeSemanticsError(
                f"fold {fold.fold_id} patch {patch_index}: mask IDs are not integer-like"
            )
        sampled_masks.append(mask)
        occupancy = mask[..., list(positive_channels)] > 0
        overlap_pixels += int(np.count_nonzero(np.count_nonzero(occupancy, axis=-1) > 1))
        for class_index, class_name in enumerate(mapping.class_names):
            channel = mask[..., class_index]
            ids, disconnected = _instance_ids_and_disconnected_components(channel)
            ids_by_class[class_name].update(int(value) for value in ids)
            sampled_disconnected_instances += len(disconnected)
    extras = tuple(index for index in range(mask_channels) if index not in positive_channels)
    background_candidates = _background_candidates(
        sampled_masks,
        extra_channels=extras,
    )
    background = _resolve_background_channel(
        channel_count=mask_channels,
        positive_channels=positive_channels,
        candidates=background_candidates,
        explicit_background_channel=explicit_background_channel,
        fold_id=fold.fold_id,
    )
    tissue_values = tuple(sorted({str(value) for value in tissues.tolist()}))
    if not tissue_values or any(not value.strip() for value in tissue_values):
        raise PanNukeSemanticsError(f"fold {fold.fold_id}: empty tissue values are invalid")
    (
        full_image_range,
        full_mask_range,
        full_ids_by_class,
        full_instance_count,
        full_disconnected_instance_count,
        full_disconnected_patch_count,
        mask_qc,
    ) = _full_semantic_scan(
        images,
        masks,
        fold=fold,
        mapping=mapping,
        positive_channels=positive_channels,
        background_channel=background,
    )
    return FoldValidation(
        fold_id=fold.fold_id,
        n_patches=n_patches,
        height=height,
        width=width,
        image_shape=image_shape,
        image_dtype=images.dtype.str,
        image_range=full_image_range,
        mask_shape=mask_shape,
        mask_dtype=masks.dtype.str,
        mask_range=full_mask_range,
        tissue_shape=tuple(int(value) for value in tissues.shape),
        tissue_dtype=tissues.dtype.str,
        tissue_values=tissue_values,
        positive_channel_indices=positive_channels,
        background_channel_index=background,
        background_channel_candidates=background_candidates,
        validation_scope="full_semantic_scan",
        full_scan_patch_count=n_patches,
        full_scan_instance_count=full_instance_count,
        sampled_patch_indices=sampled_indices,
        sampled_instance_ids_by_class={
            name: tuple(sorted(values)[:64]) for name, values in full_ids_by_class.items()
        },
        overlap_pixel_count_sampled=overlap_pixels,
        malformed_instance_count_sampled=0,
        mask_qc=mask_qc,
        disconnected_instance_count_full_scan=full_disconnected_instance_count,
        disconnected_patch_count_full_scan=full_disconnected_patch_count,
        disconnected_instance_count_sampled=sampled_disconnected_instances,
    )


def _assert_inventory_exclusions_do_not_overlap_raw(
    root: Path, exclude_paths: Sequence[str | Path]
) -> None:
    """Forbid compatibility exclusions that could hide any raw-release path."""

    resolved_root = root.resolve()
    for value in exclude_paths:
        excluded = Path(value).resolve()
        if (
            excluded == resolved_root
            or excluded in resolved_root.parents
            or resolved_root in excluded.parents
        ):
            raise PanNukeSemanticsError(
                "raw inventory exclusions may not overlap or contain the PanNuke root: "
                f"root={resolved_root}, exclusion={excluded}"
            )


def _snapshot_full_raw_inventory(root: Path) -> tuple[RawFileRecord, ...]:
    """Hash every file under the raw root, including temporary-looking names."""

    inventory = inventory_raw_files(
        root,
        fold_resolver=infer_fold_id,
        include_temporary_files=True,
    )
    if not inventory:
        raise PanNukeSemanticsError("raw PanNuke inventory is empty")
    return inventory


def _require_identical_raw_inventory(
    expected: Sequence[RawFileRecord],
    observed: Sequence[RawFileRecord],
    *,
    context: str,
) -> None:
    """Fail with a bounded path-level diff when immutable raw evidence changed."""

    if tuple(expected) == tuple(observed):
        return
    expected_by_path = {item.relative_path: item for item in expected}
    observed_by_path = {item.relative_path: item for item in observed}
    expected_paths = set(expected_by_path)
    observed_paths = set(observed_by_path)
    added = sorted(observed_paths - expected_paths)
    removed = sorted(expected_paths - observed_paths)
    changed = sorted(
        path
        for path in expected_paths & observed_paths
        if expected_by_path[path] != observed_by_path[path]
    )

    def bounded(values: Sequence[str]) -> list[str]:
        return list(values[:10])

    raise PanNukeSemanticsError(
        f"raw inventory changed {context}; added={bounded(added)}, "
        f"removed={bounded(removed)}, changed={bounded(changed)}, "
        f"counts=({len(added)}/{len(removed)}/{len(changed)})"
    )


def verify_raw_inventory_unchanged(
    result: PanNukeValidationResult,
) -> tuple[RawFileRecord, ...]:
    """Rehash the complete raw root and require the validated snapshot exactly."""

    observed = _snapshot_full_raw_inventory(result.root)
    _require_identical_raw_inventory(
        result.inventory,
        observed,
        context="after semantic validation",
    )
    return observed


def validate_discovered_release(
    discovery: ReleaseDiscovery,
    *,
    class_mapping: VerifiedClassMapping | None = None,
    use_documented_default_mapping: bool = True,
    explicit_background_channel: int | None = None,
    max_samples_per_fold: int = 32,
    memory_budget_bytes: int = 256 * 1024 * 1024,
    inventory_exclude_paths: Sequence[str | Path] = (),
    expected_fold_ids: Sequence[int] = OFFICIAL_RELEASE_FOLD_IDS,
    max_qc_overlay_patches: int = 24,
) -> PanNukeValidationResult:
    """Run the semantic gate and return evidence without writing artifacts."""

    mapping = resolve_class_mapping(
        class_mapping, use_documented_default=use_documented_default_mapping
    )
    if not discovery.folds:
        raise PanNukeSemanticsError("no extracted folds were resolved")
    expected = tuple(int(value) for value in expected_fold_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_fold_ids must contain unique fold IDs")
    if max_qc_overlay_patches <= 0:
        raise ValueError("max_qc_overlay_patches must be positive")
    discovered = tuple(sorted(fold.fold_id for fold in discovery.folds))
    if discovered != tuple(sorted(expected)):
        missing = sorted(set(expected).difference(discovered))
        unexpected = sorted(set(discovered).difference(expected))
        raise PanNukeSemanticsError(
            "release fold inventory is incomplete or unexpected: "
            f"expected={sorted(expected)}, discovered={list(discovered)}, "
            f"missing={missing}, unexpected={unexpected}. A subset cannot receive "
            "complete-release valid status."
        )
    _assert_inventory_exclusions_do_not_overlap_raw(discovery.root, inventory_exclude_paths)
    # ``inventory_exclude_paths`` remains in the API for compatibility with
    # existing manifest/duplicate callers, but exclusions outside the raw root
    # cannot affect this deliberately complete snapshot and are not applied.
    inventory_before = _snapshot_full_raw_inventory(discovery.root)
    fold_validation = tuple(
        _validate_fold(
            fold,
            mapping=mapping,
            max_samples=max_samples_per_fold,
            memory_budget_bytes=memory_budget_bytes,
            explicit_background_channel=explicit_background_channel,
        )
        for fold in discovery.folds
    )
    inventory_after = _snapshot_full_raw_inventory(discovery.root)
    _require_identical_raw_inventory(
        inventory_before,
        inventory_after,
        context="during semantic validation",
    )
    global_mask_qc: GlobalMaskQC = summarise_global_mask_qc(
        tuple(item.mask_qc for item in fold_validation)
    )
    anomaly_overlay_selection: AnomalyOverlaySelection = select_anomaly_overlay_patches(
        tuple(item.mask_qc for item in fold_validation), max_patches=max_qc_overlay_patches
    )
    policy = MaskQCPolicy(
        policy_version="pannuke-mask-qc-v2",
        positive_channel_indices=tuple(range(len(mapping.class_names))),
        background_channel_index_by_fold={
            str(item.fold_id): item.background_channel_index for item in fold_validation
        },
        supplied_background_is_exact_complement_required=False,
        void_definition="no positive-class occupancy and no supplied-background occupancy",
        cross_class_overlap_definition="positive occupancy in more than one class channel",
        positive_and_background_definition=(
            "positive-class occupancy and supplied-background occupancy at the same pixel"
        ),
        cross_class_overlap_action=(
            "retain raw identities; flag pixel, instance, and patch; never arbitrate a class"
        ),
        analysis_instance_exclusion_reason="touches_cross_class_overlap",
        applies_identically_to_primary_and_confirmatory=True,
        no_class_arbitration=True,
        source_masks_modified=False,
        release_annotation_anomalies_are_fatal=False,
        structural_invalidity_is_fatal=True,
        disconnected_instance_definition=(
            "one raw channel/instance ID occupies more than one 4-connected component"
        ),
        disconnected_instance_action=(
            "retain raw identity and flag quality; freeze analysis eligibility after the pilot "
            "without final-reference outcomes"
        ),
        disconnected_instance_ids_are_fatal=False,
    )
    return PanNukeValidationResult(
        root=discovery.root,
        mapping=mapping,
        folds=discovery.folds,
        fold_validation=fold_validation,
        inventory=inventory_before,
        archive_paths=tuple(
            path.relative_to(discovery.root).as_posix() for path in discovery.archives
        ),
        global_mask_qc=global_mask_qc,
        qc_policy=policy,
        anomaly_overlay_selection=anomaly_overlay_selection,
        expected_fold_ids=tuple(sorted(expected)),
        release_complete=True,
    )


def _markdown_report(result: PanNukeValidationResult) -> str:
    lines = [
        "# PanNuke local-release validation",
        "",
        "**Gate status:** valid for manifest construction after a full streaming semantic scan.",
        "",
        "This report inventories the local release and recommends potentially inconsistent "
        "annotations for later expert review; it does not modify source annotations.",
        "",
        "## Semantic provenance",
        "",
        f"- Positive channel order: `{', '.join(result.mapping.class_names)}`",
        f"- Mapping source: {result.mapping.source}",
        f"- Mapping source revision: `{result.mapping.source_revision}`",
        f"- Verification note: {result.mapping.source_note}",
        "",
        "## Fold arrays",
        "",
        "| Fold | Patches | Image shape / dtype / full range | Mask shape / dtype / full range | Supplied background | Overlap pixels / patches | Void pixels / patches | Positive+background pixels / patches | Overlap-touching instances | Disconnected IDs / patches | Tissues |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in result.fold_validation:
        lines.append(
            f"| {fold.fold_id} | {fold.n_patches} | `{fold.image_shape}` / "
            f"`{fold.image_dtype}` / `{fold.image_range}` | `{fold.mask_shape}` / "
            f"`{fold.mask_dtype}` / `{fold.mask_range}` | "
            f"{fold.background_channel_index if fold.background_channel_index is not None else 'none'} | "
            f"{fold.mask_qc.cross_class_overlap_pixel_count} / "
            f"{fold.mask_qc.cross_class_overlap_patch_count} | "
            f"{fold.mask_qc.void_pixel_count} / {fold.mask_qc.void_patch_count} | "
            f"{fold.mask_qc.positive_and_background_pixel_count} / "
            f"{fold.mask_qc.positive_and_background_patch_count} | "
            f"{fold.mask_qc.overlap_touching_instance_count} | "
            f"{fold.disconnected_instance_count_full_scan} / "
            f"{fold.disconnected_patch_count_full_scan} | "
            f"{len(fold.tissue_values)} |"
        )
    lines.extend(
        [
            "",
            "## Raw provenance",
            "",
            f"- Hashed raw files (SHA-256): {len(result.inventory)}",
            "- Semantic scan scope: every patch in every resolved fold",
            f"- Released fold IDs discovered: {', '.join(str(item.fold_id) for item in result.fold_validation)}",
            f"- Expected official fold IDs: {', '.join(str(value) for value in result.expected_fold_ids)}",
            f"- Complete release inventory: {str(result.release_complete).lower()}",
            f"- Archives retained and never auto-extracted/deleted: {len(result.archive_paths)}",
            "",
            "## Fixed anomaly-safe mask policy",
            "",
            "- The supplied background channel is recorded as supplied; it is not required "
            "to be the exact complement of positive-class occupancy.",
            "- Pixels with neither positive nor supplied-background occupancy remain "
            "unlabeled (`void`).",
            "- Cross-class-overlap pixels retain every raw channel/instance identity; "
            "never arbitrate a class or repair the raw mask.",
            "- Instances touching cross-class overlap are flagged with the shared primary/"
            "confirmatory analysis exclusion reason "
            f"`{result.qc_policy.analysis_instance_exclusion_reason}`.",
            "- Raw instance IDs occupying multiple 4-connected components are retained as one "
            "raw identity, counted, and flagged; they are never split or repaired. Their "
            "primary/confirmatory eligibility must be frozen after the pilot without final-"
            "reference outcomes.",
            "- Invalid array shapes, non-finite values, negative IDs, and non-integer-like IDs "
            "remain fatal structural errors.",
            f"- Release totals: {result.global_mask_qc.cross_class_overlap_pixel_count} "
            "cross-class-overlap pixels, "
            f"{result.global_mask_qc.void_pixel_count} void pixels, "
            f"{result.global_mask_qc.positive_and_background_pixel_count} positive+background "
            "pixels, and "
            f"{result.global_mask_qc.overlap_touching_instance_count} overlap-touching instances.",
            "",
            "## Independence limitation",
            "",
            result.independence_statement,
            "Source patch is therefore the mandatory `group_id`; stronger patient/WSI "
            "separation must not be claimed without separately verified metadata.",
            "",
        ]
    )
    return "\n".join(lines)


def _normalise_rgb(image: np.ndarray) -> np.ndarray:
    value = image[..., :3].astype(np.float32, copy=False)
    if not np.isfinite(value).all():
        raise PanNukeSemanticsError("cannot render non-finite image values")
    maximum = float(value.max(initial=0.0))
    minimum = float(value.min(initial=0.0))
    if maximum <= 1.0 and minimum >= 0.0:
        return np.clip(value, 0.0, 1.0)
    if maximum <= 255.0 and minimum >= 0.0:
        return np.clip(value / 255.0, 0.0, 1.0)
    scale = maximum - minimum
    return np.clip((value - minimum) / scale, 0.0, 1.0) if scale else np.zeros_like(value)


def write_overlay_grid(
    result: PanNukeValidationResult,
    destination: str | Path,
    *,
    max_patches: int = 6,
) -> Path:
    """Write a small full-patch contour overlay; no nucleus crops are generated."""

    if max_patches <= 0:
        raise ValueError("max_patches must be positive")
    target = _require_derived_destination_outside_raw(
        result.root,
        Path(destination),
        "overlay destination",
    )
    _require_artifact_suffix(target, ".png", "overlay destination")
    per_fold_choices: list[list[tuple[DiscoveredFold, int]]] = []
    per_fold_limit = max(1, (max_patches + len(result.folds) - 1) // len(result.folds))
    for fold, facts in zip(result.folds, result.fold_validation, strict=True):
        indices = deterministic_sample_indices(facts.n_patches, max_samples=per_fold_limit)
        per_fold_choices.append([(fold, index) for index in indices])
    choices: list[tuple[DiscoveredFold, int]] = []
    for offset in range(per_fold_limit):
        for fold_choices in per_fold_choices:
            if offset < len(fold_choices):
                choices.append(fold_choices[offset])
                if len(choices) == max_patches:
                    break
        if len(choices) == max_patches:
            break
    if not choices:
        raise PanNukeSemanticsError("cannot render an overlay for an empty release")
    columns = min(3, len(choices))
    rows = (len(choices) + columns - 1) // columns
    figure = Figure(figsize=(4.0 * columns, 4.0 * rows), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False)
    for axis in axes.flat:
        axis.set_axis_off()
    for axis, (fold, patch_index) in zip(axes.flat, choices, strict=False):
        images = open_npy_mmap(fold.image_path)
        masks = open_npy_mmap(fold.mask_path)
        image = _channel_last_patch(images, patch_index, fold.image_channel_axis)
        mask = _channel_last_patch(masks, patch_index, fold.mask_channel_axis)
        axis.imshow(_normalise_rgb(image))
        for class_index, _class_name in enumerate(result.mapping.class_names):
            occupancy = mask[..., class_index] > 0
            if occupancy.any():
                axis.contour(
                    occupancy.astype(np.uint8),
                    levels=[0.5],
                    colors=[_OVERLAY_COLOURS[class_index % len(_OVERLAY_COLOURS)]],
                    linewidths=0.9,
                )
        axis.set_title(f"Fold {fold.fold_id}, source patch {patch_index}", fontsize=9)
        axis.set_axis_off()

    def save(path: Path) -> None:
        figure.savefig(path, dpi=120, format=target.suffix.lstrip(".") or "png")

    try:
        return atomic_replace_via_temp(target, save)
    finally:
        figure.clear()


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _create_missing_parents(path: Path) -> list[Path]:
    missing: list[Path] = []
    current = path
    while not _path_exists(current):
        missing.append(current)
        current = current.parent
    if not current.is_dir():
        raise NotADirectoryError(f"validation artifact parent is not a directory: {current}")
    created: list[Path] = []
    for directory in reversed(missing):
        directory.mkdir()
        created.append(directory)
    return created


def _promote_without_overwrite(source: Path, destination: Path) -> PublishedPath:
    return publish_file_no_overwrite(source, destination)


def _publish_immutable_validation_files(
    ancillary_files: Sequence[tuple[Path, Path]],
    *,
    success_marker: tuple[Path, Path],
    raw_inventory_verifier: Callable[[], object],
) -> None:
    """Publish four staged base artifacts with the validation JSON last."""

    staged_marker, final_marker = success_marker
    pairs = tuple(
        (source.resolve(), target.parent.resolve() / target.name)
        for source, target in ancillary_files
    )
    staged_marker = staged_marker.resolve()
    final_marker = final_marker.parent.resolve() / final_marker.name
    staged_all = (*pairs, (staged_marker, final_marker))
    if any(not source.is_file() for source, _target in staged_all):
        raise FileNotFoundError("staged validation artifact set is incomplete")

    def require_exact_final_set() -> None:
        present = tuple(_path_exists(target) for _source, target in staged_all)
        if not all(present) or any(
            target.is_symlink() or not target.is_file() for _source, target in staged_all
        ):
            raise FileExistsError(
                "existing PanNuke base-validation artifact set is partial; refusing mutation"
            )
        differing = [
            str(target)
            for source, target in staged_all
            if source.read_bytes() != target.read_bytes()
        ]
        if differing:
            raise FileExistsError(
                "existing immutable PanNuke base-validation artifacts differ: "
                + ", ".join(differing)
            )

    raw_inventory_verifier()
    present = tuple(_path_exists(target) for _source, target in staged_all)
    if any(present):
        require_exact_final_set()
        raw_inventory_verifier()
        # Re-read under the held per-output lock after the final raw rehash.  This
        # catches a non-cooperating replacement instead of returning a mixed set.
        require_exact_final_set()
        return

    created_directories: list[Path] = []
    for parent in {target.parent for _source, target in staged_all}:
        created_directories.extend(_create_missing_parents(parent))
    published: list[PublishedPath] = []
    try:
        for source, target in pairs:
            published.append(_promote_without_overwrite(source, target))
        published.append(_promote_without_overwrite(staged_marker, final_marker))
        # A raw mutation racing any promotion invalidates this complete derived
        # set; the rollback path restores every staged artifact and removes finals.
        raw_inventory_verifier()
        require_exact_final_set()
    except BaseException as publication_error:
        del created_directories
        try:
            rollback_owned_publications(published)
        except RuntimeError:
            raise RuntimeError(
                "base-validation publication failed and ownership-safe rollback was incomplete"
            ) from publication_error
        raise


def _with_validation_publication_lock(
    function: Callable[..., ValidationArtifacts],
) -> Callable[..., ValidationArtifacts]:
    """Serialize every final target across the complete library validation build."""

    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args: object, **kwargs: object) -> ValidationArtifacts:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        raw_root = Path(bound.arguments["root"]).resolve()
        output, json_path, markdown_path, resolved_overlay_path, resolved_inventory_path = (
            _resolve_validation_artifact_paths(
                raw_root,
                bound.arguments["output_dir"],
                raw_inventory_csv_path=bound.arguments["raw_inventory_csv_path"],
                overlay_path=bound.arguments["overlay_path"],
            )
        )
        lock_targets = (
            output,
            json_path,
            markdown_path,
            resolved_overlay_path,
            resolved_inventory_path,
        )
        with ExclusiveBundlePublicationLock(lock_targets, role="PanNuke base validation"):
            return function(*args, **kwargs)

    return wrapped


@_with_validation_publication_lock
def validate_pannuke(
    root: str | Path,
    output_dir: str | Path,
    *,
    class_mapping: VerifiedClassMapping | None = None,
    use_documented_default_mapping: bool = True,
    explicit_background_channel: int | None = None,
    max_samples_per_fold: int = 32,
    max_overlay_patches: int = 6,
    memory_budget_bytes: int = 256 * 1024 * 1024,
    expected_fold_ids: Sequence[int] = OFFICIAL_RELEASE_FOLD_IDS,
    raw_inventory_csv_path: str | Path | None = None,
    overlay_path: str | Path | None = None,
) -> ValidationArtifacts:
    """Discover, validate, hash, and atomically report a local PanNuke release."""

    raw_root = Path(root).resolve()
    output, json_path, markdown_path, resolved_overlay_path, resolved_inventory_path = (
        _resolve_validation_artifact_paths(
            raw_root,
            output_dir,
            raw_inventory_csv_path=raw_inventory_csv_path,
            overlay_path=overlay_path,
        )
    )
    staging_parent = _require_derived_destination_outside_raw(
        raw_root,
        output.parent / f".{output.name}.pannuke-validation-staging",
        "validation staging directory",
        directory=True,
    )

    mapping = resolve_class_mapping(
        class_mapping, use_documented_default=use_documented_default_mapping
    )
    discovery = discover_pannuke_release(
        root,
        positive_class_count=len(mapping.class_names),
        max_inspection_samples=min(max_samples_per_fold, 8),
        memory_budget_bytes=memory_budget_bytes,
    )
    result = validate_discovered_release(
        discovery,
        class_mapping=mapping,
        explicit_background_channel=explicit_background_channel,
        max_samples_per_fold=max_samples_per_fold,
        memory_budget_bytes=memory_budget_bytes,
        inventory_exclude_paths=(output,),
        expected_fold_ids=expected_fold_ids,
        max_qc_overlay_patches=max_overlay_patches,
    )
    staging_parent_created = False
    staging_root: Path | None = None
    try:
        if not staging_parent.exists():
            staging_parent.mkdir(parents=True)
            staging_parent_created = True
        staging_root = Path(tempfile.mkdtemp(prefix="run-", dir=staging_parent))
        staged_output = ensure_output_capacity(
            staging_root / "base", estimated_bytes=8 * 1024 * 1024
        )
        staged_json_path = atomic_write_json(
            staged_output / "pannuke_validation.json", result.as_dict()
        )
        staged_markdown_path = atomic_write_text(
            staged_output / "pannuke_validation.md", _markdown_report(result)
        )
        staged_overlay_path = write_overlay_grid(
            result,
            staging_root / "pannuke_overlay_grid.png",
            max_patches=max_overlay_patches,
        )
        inventory_buffer = io.StringIO(newline="")
        inventory_writer = csv.DictWriter(
            inventory_buffer,
            fieldnames=("relative_path", "size_bytes", "sha256", "fold_id", "file_kind"),
        )
        inventory_writer.writeheader()
        for item in result.inventory:
            inventory_writer.writerow(item.as_dict())
        staged_inventory_path = atomic_write_text(
            staging_root / "raw_files_sha256.csv",
            inventory_buffer.getvalue(),
        )
        verify_raw_inventory_unchanged(result)
        _publish_immutable_validation_files(
            (
                (staged_markdown_path, markdown_path),
                (staged_overlay_path, resolved_overlay_path),
                (staged_inventory_path, resolved_inventory_path),
            ),
            success_marker=(staged_json_path, json_path),
            raw_inventory_verifier=lambda: verify_raw_inventory_unchanged(result),
        )
    finally:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)
        if staging_parent_created:
            with suppress(OSError):
                staging_parent.rmdir()
    return ValidationArtifacts(
        result=result,
        json_path=json_path,
        markdown_path=markdown_path,
        overlay_path=resolved_overlay_path,
        raw_inventory_csv_path=resolved_inventory_path,
    )


# Verb-oriented aliases make CLI wiring explicit.
run_validation_gate = validate_pannuke


__all__ = [
    "resolve_class_mapping",
    "run_validation_gate",
    "validate_discovered_release",
    "validate_pannuke",
    "verify_raw_inventory_unchanged",
    "write_overlay_grid",
]
