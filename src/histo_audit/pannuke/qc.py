"""Anomaly-safe, non-mutating quality control for released PanNuke masks.

The functions in this module deliberately keep three pixel states separate:

* a pixel occupied by more than one positive class (cross-class overlap),
* a pixel occupied by a positive class and the supplied background channel, and
* a pixel occupied by neither a positive class nor supplied background (void).

None of these states is resolved into a preferred class.  The returned records
retain raw channel/instance identities and hashes of every derived binary flag
mask so downstream artifacts can be reconciled without changing source arrays.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from .exceptions import PanNukeSemanticsError
from .models import (
    AnomalyOverlaySelection,
    FoldMaskQC,
    GlobalMaskQC,
    MaskInstanceQC,
    PatchMaskQC,
)

_MASK_HASH_VERSION = b"pannuke-mask-qc-v1\0"
_MASK_KINDS = (
    "positive_any",
    "supplied_background",
    "void_unlabelled",
    "cross_class_overlap",
    "positive_and_background",
    "anomaly_union",
)


def _flag_mask_sha256(kind: str, flag_mask: np.ndarray) -> str:
    """Hash one exact derived flag mask, including kind and array shape."""

    if kind not in _MASK_KINDS:
        raise ValueError(f"unknown QC mask kind: {kind}")
    value = np.ascontiguousarray(flag_mask, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(_MASK_HASH_VERSION)
    digest.update(kind.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _validate_patch_inputs(
    mask: np.ndarray,
    *,
    fold_id: int,
    patch_index: int,
    class_names: Sequence[str],
    positive_channel_indices: Sequence[int],
    background_channel_index: int | None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[int, ...]]:
    value = np.asarray(mask)
    if value.ndim != 3 or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise PanNukeSemanticsError(
            f"fold {fold_id} patch {patch_index}: source mask patch must be non-empty "
            "channel-last rank 3"
        )
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise PanNukeSemanticsError(
            f"fold {fold_id} patch {patch_index}: mask values must be finite numeric"
        )
    if np.any(value < 0):
        raise PanNukeSemanticsError(
            f"fold {fold_id} patch {patch_index}: instance IDs are negative"
        )
    if not np.equal(value, np.rint(value)).all():
        raise PanNukeSemanticsError(
            f"fold {fold_id} patch {patch_index}: mask IDs are not integer-like"
        )
    names = tuple(str(name) for name in class_names)
    channels = tuple(int(index) for index in positive_channel_indices)
    if not names or len(names) != len(channels):
        raise PanNukeSemanticsError(
            f"fold {fold_id} patch {patch_index}: positive class names/channels do not align"
        )
    if len(set(names)) != len(names) or any(not name.strip() for name in names):
        raise PanNukeSemanticsError(
            f"fold {fold_id} patch {patch_index}: positive class names must be unique and non-empty"
        )
    if len(set(channels)) != len(channels) or any(
        index < 0 or index >= value.shape[-1] for index in channels
    ):
        raise PanNukeSemanticsError(
            f"fold {fold_id} patch {patch_index}: positive channel indices are invalid"
        )
    if background_channel_index is not None:
        background_channel_index = int(background_channel_index)
        if (
            background_channel_index < 0
            or background_channel_index >= value.shape[-1]
            or background_channel_index in channels
        ):
            raise PanNukeSemanticsError(
                f"fold {fold_id} patch {patch_index}: supplied background channel is invalid"
            )
        background_values = value[..., background_channel_index]
        if not np.isin(background_values, (0, 1)).all():
            raise PanNukeSemanticsError(
                f"fold {fold_id} patch {patch_index}: supplied background channel is not binary"
            )
    return value, names, channels


def analyse_patch_mask_qc(
    mask: np.ndarray,
    *,
    fold_id: int,
    patch_index: int,
    class_names: Sequence[str],
    positive_channel_indices: Sequence[int],
    background_channel_index: int | None,
) -> PatchMaskQC:
    """Return complete QC facts for a channel-last mask without modifying it."""

    value, names, channels = _validate_patch_inputs(
        mask,
        fold_id=fold_id,
        patch_index=patch_index,
        class_names=class_names,
        positive_channel_indices=positive_channel_indices,
        background_channel_index=background_channel_index,
    )
    positive_occupancy = value[..., list(channels)] > 0
    positive_any = np.any(positive_occupancy, axis=-1)
    if background_channel_index is None:
        supplied_background = np.zeros(value.shape[:2], dtype=bool)
    else:
        supplied_background = value[..., int(background_channel_index)] > 0
    void_unlabelled = ~positive_any & ~supplied_background
    cross_class_overlap = np.count_nonzero(positive_occupancy, axis=-1) > 1
    positive_and_background = positive_any & supplied_background
    anomaly_union = void_unlabelled | cross_class_overlap | positive_and_background

    affected_instances: list[MaskInstanceQC] = []
    affected_pixel_mask = cross_class_overlap | positive_and_background
    for class_index, (class_name, channel_index) in enumerate(zip(names, channels, strict=True)):
        channel = value[..., channel_index]
        # Normal instances never need a full-size equality mask.  Restrict the
        # candidate IDs first to pixels carrying one of the two instance-touching
        # anomalies; void pixels cannot belong to an instance by definition.
        candidate_values = channel[affected_pixel_mask & (channel > 0)]
        instance_ids = np.unique(candidate_values).astype(np.int64, copy=False)
        for instance_id_value in instance_ids:
            instance_id = int(instance_id_value)
            instance_mask = channel == instance_id_value
            target_overlap = instance_mask & cross_class_overlap
            target_positive_background = instance_mask & positive_and_background
            overlap_pixel_count = int(np.count_nonzero(target_overlap))
            positive_background_pixel_count = int(np.count_nonzero(target_positive_background))
            if overlap_pixel_count == 0 and positive_background_pixel_count == 0:
                continue

            by_class: dict[str, tuple[int, ...]] = {}
            overlapping_class_indices: list[int] = []
            overlapping_ids: set[int] = set()
            if overlap_pixel_count:
                for other_class_index, (other_name, other_channel_index) in enumerate(
                    zip(names, channels, strict=True)
                ):
                    if other_class_index == class_index:
                        continue
                    other_values = value[..., other_channel_index][target_overlap]
                    other_ids = tuple(
                        int(item)
                        for item in np.unique(other_values[other_values > 0]).astype(
                            np.int64, copy=False
                        )
                    )
                    if other_ids:
                        overlapping_class_indices.append(other_class_index)
                        by_class[other_name] = other_ids
                        overlapping_ids.update(other_ids)
            affected_instances.append(
                MaskInstanceQC(
                    fold_id=int(fold_id),
                    patch_index=int(patch_index),
                    class_index=class_index,
                    class_name=class_name,
                    channel_index=channel_index,
                    instance_id=instance_id,
                    total_pixel_count=int(np.count_nonzero(instance_mask)),
                    overlap_pixel_count=overlap_pixel_count,
                    positive_background_pixel_count=positive_background_pixel_count,
                    overlapping_class_indices=tuple(overlapping_class_indices),
                    overlapping_instance_ids=tuple(sorted(overlapping_ids)),
                    overlapping_instance_ids_by_class=by_class,
                )
            )

    affected_instances.sort(key=lambda item: (item.channel_index, item.instance_id))
    affected_class_indices = tuple(sorted({item.class_index for item in affected_instances}))
    affected_class_names = tuple(names[index] for index in affected_class_indices)
    flag_masks = {
        "positive_any": positive_any,
        "supplied_background": supplied_background,
        "void_unlabelled": void_unlabelled,
        "cross_class_overlap": cross_class_overlap,
        "positive_and_background": positive_and_background,
        "anomaly_union": anomaly_union,
    }
    total_pixel_count = int(np.prod(value.shape[:2], dtype=np.int64))
    positive_any_pixel_count = int(np.count_nonzero(positive_any))
    background_pixel_count = int(np.count_nonzero(supplied_background))
    void_pixel_count = int(np.count_nonzero(void_unlabelled))
    positive_and_background_pixel_count = int(np.count_nonzero(positive_and_background))
    if (
        positive_any_pixel_count
        + background_pixel_count
        + void_pixel_count
        - positive_and_background_pixel_count
        != total_pixel_count
    ):
        raise AssertionError("positive/background/void QC masks do not partition the patch")
    return PatchMaskQC(
        fold_id=int(fold_id),
        patch_index=int(patch_index),
        height=int(value.shape[0]),
        width=int(value.shape[1]),
        total_pixel_count=total_pixel_count,
        positive_any_pixel_count=positive_any_pixel_count,
        background_pixel_count=background_pixel_count,
        void_pixel_count=void_pixel_count,
        cross_class_overlap_pixel_count=int(np.count_nonzero(cross_class_overlap)),
        positive_and_background_pixel_count=positive_and_background_pixel_count,
        anomaly_union_pixel_count=int(np.count_nonzero(anomaly_union)),
        affected_instance_count=len(affected_instances),
        affected_class_indices=affected_class_indices,
        affected_class_names=affected_class_names,
        affected_instances=tuple(affected_instances),
        has_void=bool(np.any(void_unlabelled)),
        has_cross_class_overlap=bool(np.any(cross_class_overlap)),
        has_positive_and_background=bool(np.any(positive_and_background)),
        mask_sha256_by_kind={
            kind: _flag_mask_sha256(kind, flag_masks[kind]) for kind in _MASK_KINDS
        },
    )


def summarise_fold_mask_qc(fold_id: int, patches: Sequence[PatchMaskQC]) -> FoldMaskQC:
    """Reconcile an exhaustive sequence of patch records for one fold."""

    ordered = tuple(sorted(patches, key=lambda item: item.patch_index))
    if not ordered:
        raise ValueError(f"fold {fold_id}: at least one patch QC record is required")
    if any(item.fold_id != fold_id for item in ordered):
        raise ValueError(f"fold {fold_id}: patch QC contains a different fold ID")
    indices = tuple(item.patch_index for item in ordered)
    if indices != tuple(range(len(ordered))):
        raise ValueError(
            f"fold {fold_id}: patch QC must contain each zero-based patch index exactly once"
        )
    dimensions = {(item.height, item.width) for item in ordered}
    if len(dimensions) != 1:
        raise ValueError(f"fold {fold_id}: patch QC dimensions are inconsistent")

    def total(field: str) -> int:
        return sum(int(getattr(item, field)) for item in ordered)

    void_indices = tuple(item.patch_index for item in ordered if item.has_void)
    overlap_indices = tuple(item.patch_index for item in ordered if item.has_cross_class_overlap)
    positive_background_indices = tuple(
        item.patch_index for item in ordered if item.has_positive_and_background
    )
    anomaly_indices = tuple(
        item.patch_index for item in ordered if item.anomaly_union_pixel_count > 0
    )
    affected = tuple(instance for item in ordered for instance in item.affected_instances)
    affected_class_indices = tuple(sorted({item.class_index for item in affected}))
    class_name_by_index: dict[int, str] = {}
    for item in affected:
        previous = class_name_by_index.setdefault(item.class_index, item.class_name)
        if previous != item.class_name:
            raise ValueError(f"fold {fold_id}: inconsistent class name in affected-instance QC")
    return FoldMaskQC(
        fold_id=int(fold_id),
        patch_count=len(ordered),
        total_pixel_count=total("total_pixel_count"),
        positive_any_pixel_count=total("positive_any_pixel_count"),
        background_pixel_count=total("background_pixel_count"),
        void_pixel_count=total("void_pixel_count"),
        cross_class_overlap_pixel_count=total("cross_class_overlap_pixel_count"),
        positive_and_background_pixel_count=total("positive_and_background_pixel_count"),
        anomaly_union_pixel_count=total("anomaly_union_pixel_count"),
        void_patch_count=len(void_indices),
        cross_class_overlap_patch_count=len(overlap_indices),
        positive_and_background_patch_count=len(positive_background_indices),
        anomaly_union_patch_count=len(anomaly_indices),
        normal_patch_count=len(ordered) - len(anomaly_indices),
        affected_instance_count=len(affected),
        overlap_touching_instance_count=sum(item.overlap_pixel_count > 0 for item in affected),
        positive_background_touching_instance_count=sum(
            item.positive_background_pixel_count > 0 for item in affected
        ),
        affected_class_indices=affected_class_indices,
        affected_class_names=tuple(class_name_by_index[index] for index in affected_class_indices),
        void_patch_indices=void_indices,
        cross_class_overlap_patch_indices=overlap_indices,
        positive_and_background_patch_indices=positive_background_indices,
        anomaly_union_patch_indices=anomaly_indices,
        patches=ordered,
    )


def summarise_global_mask_qc(folds: Sequence[FoldMaskQC]) -> GlobalMaskQC:
    """Reconcile fold summaries into release-wide QC totals."""

    ordered = tuple(sorted(folds, key=lambda item: item.fold_id))
    fold_ids = tuple(item.fold_id for item in ordered)
    if not ordered or len(fold_ids) != len(set(fold_ids)):
        raise ValueError("global mask QC requires unique, non-empty fold summaries")

    def total(field: str) -> int:
        return sum(int(getattr(item, field)) for item in ordered)

    class_name_by_index: dict[int, str] = {}
    for fold in ordered:
        for index, name in zip(fold.affected_class_indices, fold.affected_class_names, strict=True):
            previous = class_name_by_index.setdefault(index, name)
            if previous != name:
                raise ValueError("affected class names disagree across fold QC summaries")
    affected_class_indices = tuple(sorted(class_name_by_index))
    return GlobalMaskQC(
        fold_ids=fold_ids,
        fold_count=len(ordered),
        patch_count=total("patch_count"),
        total_pixel_count=total("total_pixel_count"),
        positive_any_pixel_count=total("positive_any_pixel_count"),
        background_pixel_count=total("background_pixel_count"),
        void_pixel_count=total("void_pixel_count"),
        cross_class_overlap_pixel_count=total("cross_class_overlap_pixel_count"),
        positive_and_background_pixel_count=total("positive_and_background_pixel_count"),
        anomaly_union_pixel_count=total("anomaly_union_pixel_count"),
        void_patch_count=total("void_patch_count"),
        cross_class_overlap_patch_count=total("cross_class_overlap_patch_count"),
        positive_and_background_patch_count=total("positive_and_background_patch_count"),
        anomaly_union_patch_count=total("anomaly_union_patch_count"),
        normal_patch_count=total("normal_patch_count"),
        affected_instance_count=total("affected_instance_count"),
        overlap_touching_instance_count=total("overlap_touching_instance_count"),
        positive_background_touching_instance_count=total(
            "positive_background_touching_instance_count"
        ),
        affected_class_indices=affected_class_indices,
        affected_class_names=tuple(class_name_by_index[index] for index in affected_class_indices),
    )


def _interleaved_category_candidates(folds: Sequence[FoldMaskQC], category: str) -> tuple[str, ...]:
    per_fold: list[tuple[int, ...]] = []
    for fold in sorted(folds, key=lambda item: item.fold_id):
        if category == "cross_class_overlap":
            indices = fold.cross_class_overlap_patch_indices
        elif category == "positive_and_background":
            indices = fold.positive_and_background_patch_indices
        elif category == "void_unlabelled":
            indices = fold.void_patch_indices
        elif category == "normal":
            anomaly = set(fold.anomaly_union_patch_indices)
            indices = tuple(index for index in range(fold.patch_count) if index not in anomaly)
        else:  # pragma: no cover - private caller fixes the categories
            raise ValueError(f"unknown overlay category: {category}")
        per_fold.append(indices)
    keys: list[str] = []
    maximum = max((len(indices) for indices in per_fold), default=0)
    ordered_folds = tuple(sorted(folds, key=lambda item: item.fold_id))
    for offset in range(maximum):
        for fold, indices in zip(ordered_folds, per_fold, strict=True):
            if offset < len(indices):
                keys.append(f"fold_{fold.fold_id}:patch_{indices[offset]}")
    return tuple(keys)


def select_anomaly_overlay_patches(
    folds: Sequence[FoldMaskQC], *, max_patches: int = 24
) -> AnomalyOverlaySelection:
    """Select deterministic, fold-interleaved QC examples without outcome tuning."""

    if max_patches <= 0:
        raise ValueError("max_patches must be positive")
    if not folds:
        raise ValueError("at least one fold QC summary is required")
    categories = (
        "cross_class_overlap",
        "positive_and_background",
        "void_unlabelled",
        "normal",
    )
    candidates = {
        category: _interleaved_category_candidates(folds, category) for category in categories
    }
    cursors = {category: 0 for category in categories}
    selected: list[str] = []
    selected_set: set[str] = set()
    selected_by_category: dict[str, list[str]] = {category: [] for category in categories}
    while len(selected) < max_patches:
        progressed = False
        for category in categories:
            category_candidates = candidates[category]
            while (
                cursors[category] < len(category_candidates)
                and category_candidates[cursors[category]] in selected_set
            ):
                cursors[category] += 1
            if cursors[category] >= len(category_candidates):
                continue
            key = category_candidates[cursors[category]]
            cursors[category] += 1
            selected.append(key)
            selected_set.add(key)
            selected_by_category[category].append(key)
            progressed = True
            if len(selected) == max_patches:
                break
        if not progressed:
            break
    return AnomalyOverlaySelection(
        strategy=(
            "fixed category round-robin (cross-class overlap, positive+background, void, "
            "normal), interleaved by fold then ascending source patch index"
        ),
        requested_max_patches=int(max_patches),
        selected_patch_keys=tuple(selected),
        selected_by_category={key: tuple(value) for key, value in selected_by_category.items()},
        category_candidate_counts={key: len(value) for key, value in candidates.items()},
    )


__all__ = [
    "analyse_patch_mask_qc",
    "select_anomaly_overlay_patches",
    "summarise_fold_mask_qc",
    "summarise_global_mask_qc",
]
