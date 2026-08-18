"""Manifest-driven PanNuke crops with exact raw identity and projected target masks.

The manifest is an index, not a trusted substitute for the raw instance arrays.
Every requested row is resolved back to its validated read-only memory map and
its class channel/instance ID is checked before a crop or feature is emitted.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from dataclasses import field as dataclass_field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from numpy.typing import NDArray
from PIL import Image
from scipy import ndimage  # type: ignore[import-untyped]

from histo_audit.data.targets import TargetCrop, highlight_target, mask_bbox
from histo_audit.pannuke.exceptions import PanNukeSemanticsError
from histo_audit.pannuke.io import (
    ensure_derived_output_outside_raw,
    open_npy_mmap,
    sha256_file,
)
from histo_audit.pannuke.manifest import validate_manifest_invariants
from histo_audit.pannuke.models import PanNukeValidationResult
from histo_audit.pannuke.publication import (
    PublishedPath,
    assert_mutable_publication_destination,
    publish_flat_directory_no_overwrite,
    rollback_owned_publications,
)
from histo_audit.pannuke.validation import verify_raw_inventory_unchanged

from .cache_provenance import (
    array_artifact_sha256,
    atomic_save_npz_with_sidecar,
    build_frozen_cache_metadata,
    canonical_sha256,
    explicit_unlearned_weights_sha256,
    ordered_sample_ids_sha256,
    verify_frozen_cache_sidecar,
)
from .eligibility import (
    select_manifest_rows,
    validate_analysis_eligibility_provenance,
)
from .engineered import (
    EngineeredFeatureSet,
    build_engineered_feature_set,
    save_engineered_feature_cache,
    select_target_morphometrics,
)
from .imagenet import EmbeddingResult, ResNet18EmbeddingConfig, extract_resnet18_embeddings

UInt8Images = NDArray[np.uint8]
BoolMasks = NDArray[np.bool_]
_DISCONNECTED_INSTANCE_FLAG = "disconnected_instance_id"
_INSTANCE_CONNECTIVITY_4 = ndimage.generate_binary_structure(2, 1)
_COMPONENT_PROJECTION_ID = "nearest_per_component_with_forward_fallback_v1"
_CHUNKED_AUTO_THRESHOLD = 10_000
_DEFAULT_CHUNK_SIZE = 4_096
FULL_MANIFEST_CACHE_MIN_FREE_BYTES = 35 * 1024**3


@dataclass(frozen=True, slots=True)
class FullManifestCacheDiskCheck:
    """Destination-volume capacity evidence for a full-manifest cache build."""

    manifest_row_count: int
    target_output_parent: Path
    check_required: bool
    disk_usage_path: Path | None
    free_bytes: int | None
    required_free_bytes: int


class InsufficientFullManifestCacheDiskSpaceError(RuntimeError):
    """The fixed M7 destination-volume start gate is not satisfied."""

    def __init__(self, check: FullManifestCacheDiskCheck) -> None:
        self.check = check
        super().__init__(
            "full-manifest PanNuke cache start gate failed: "
            f"free_bytes={check.free_bytes}, "
            f"required_free_bytes={check.required_free_bytes}, "
            f"target_output_parent={check.target_output_parent}, "
            f"disk_usage_path={check.disk_usage_path}, "
            f"manifest_row_count={check.manifest_row_count}"
        )


def _existing_destination_volume_path(target_output_parent: Path) -> Path:
    candidate = target_output_parent
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise OSError(
                f"cannot resolve an existing destination-volume path for {target_output_parent}"
            )
        candidate = parent
    if not candidate.is_dir():
        raise NotADirectoryError(
            f"destination output parent resolves through a non-directory: {candidate}"
        )
    return candidate


def require_full_manifest_cache_disk_space(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    sample_ids: tuple[str, ...] | None = None,
) -> FullManifestCacheDiskCheck:
    """Enforce the fixed 35-GiB M7 start gate on the destination volume.

    Explicit sample-ID extractions and small manifests of at most 10,000 rows are
    smoke/test paths and do not require the production full-release capacity gate.
    Larger explicit selections are production-scale and cannot bypass the gate.  No
    directory is created by this check, and there is deliberately no caller-supplied
    threshold override.
    """

    manifest_rows = (
        pq.ParquetFile(Path(manifest_path).resolve()).metadata.num_rows
        if sample_ids is None
        else len(sample_ids)
    )
    target_output_parent = Path(output_dir).resolve().parent
    if manifest_rows <= _CHUNKED_AUTO_THRESHOLD:
        return FullManifestCacheDiskCheck(
            manifest_row_count=manifest_rows,
            target_output_parent=target_output_parent,
            check_required=False,
            disk_usage_path=None,
            free_bytes=None,
            required_free_bytes=FULL_MANIFEST_CACHE_MIN_FREE_BYTES,
        )

    disk_usage_path = _existing_destination_volume_path(target_output_parent)
    free_bytes = int(shutil.disk_usage(disk_usage_path).free)
    check = FullManifestCacheDiskCheck(
        manifest_row_count=manifest_rows,
        target_output_parent=target_output_parent,
        check_required=True,
        disk_usage_path=disk_usage_path,
        free_bytes=free_bytes,
        required_free_bytes=FULL_MANIFEST_CACHE_MIN_FREE_BYTES,
    )
    if free_bytes < FULL_MANIFEST_CACHE_MIN_FREE_BYTES:
        raise InsufficientFullManifestCacheDiskSpaceError(check)
    return check


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


@dataclass(frozen=True, slots=True)
class _ComponentProjection:
    """One model-facing mask plus source-component contribution evidence."""

    mask: NDArray[np.bool_]
    raw_component_count: int
    projected_union_component_count: int
    raw_component_pixel_counts: tuple[int, ...]
    projected_component_pixel_counts: tuple[int, ...]
    projected_component_unique_pixel_counts: tuple[int, ...]
    baseline_projected_component_counts: tuple[int, ...]
    fallback_used: tuple[bool, ...]
    collision_pixel_count: int
    collision_excess_count: int
    adjacency_pair_count: int


def _resize_binary_nearest(mask: NDArray[np.bool_], output_size: int) -> NDArray[np.bool_]:
    return (
        np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (output_size, output_size),
                resample=Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        )
        > 0
    )


def _forward_component_footprint(
    component: NDArray[np.bool_], output_size: int
) -> NDArray[np.bool_]:
    """Project every source pixel deterministically when inverse NN loses topology."""

    height, width = component.shape
    output = np.zeros((output_size, output_size), dtype=bool)
    source_y, source_x = np.nonzero(component)
    if not len(source_x):  # pragma: no cover - caller labels present components
        raise AssertionError("source component is empty")
    if height > output_size and width > output_size:
        output_y = ((2 * source_y + 1) * output_size) // (2 * height)
        output_x = ((2 * source_x + 1) * output_size) // (2 * width)
        output[output_y, output_x] = True
        return output

    for y, x in zip(source_y.tolist(), source_x.tolist(), strict=True):
        if height > output_size:
            projected_y = ((2 * y + 1) * output_size) // (2 * height)
            output_y = range(projected_y, projected_y + 1)
        else:
            start_y = (y * output_size) // height
            stop_y = ((y + 1) * output_size) // height
            output_y = range(start_y, max(start_y + 1, stop_y))
        if width > output_size:
            projected_x = ((2 * x + 1) * output_size) // (2 * width)
            output_x = range(projected_x, projected_x + 1)
        else:
            start_x = (x * output_size) // width
            stop_x = ((x + 1) * output_size) // width
            output_x = range(start_x, max(start_x + 1, stop_x))
        for projected_y in output_y:
            for projected_x in output_x:
                output[projected_y, projected_x] = True
    return output


def _component_covering_projection(
    source_mask: NDArray[np.bool_], output_size: int
) -> _ComponentProjection:
    """Project every raw 4-component without claiming exact union topology.

    Standard nearest-neighbour output is retained byte-for-byte for every
    component it represents as one non-empty 4-connected footprint.  Only a
    component lost or split by nearest-neighbour uses the deterministic forward
    footprint.  No source pixel, component, class, or instance identity is
    changed.
    """

    labels, raw_component_count = ndimage.label(
        source_mask,
        structure=_INSTANCE_CONNECTIVITY_4,
    )
    if raw_component_count < 1:
        raise PanNukeSemanticsError("component projection received an empty source mask")
    projected_components: list[NDArray[np.bool_]] = []
    raw_pixel_counts: list[int] = []
    projected_pixel_counts: list[int] = []
    baseline_counts: list[int] = []
    fallback_used: list[bool] = []
    for component_index in range(1, raw_component_count + 1):
        component = np.asarray(labels == component_index, dtype=bool)
        raw_pixel_counts.append(int(component.sum()))
        baseline = _resize_binary_nearest(component, output_size)
        _, baseline_count = ndimage.label(
            baseline,
            structure=_INSTANCE_CONNECTIVITY_4,
        )
        baseline_counts.append(int(baseline_count))
        use_fallback = baseline_count != 1
        projected = (
            _forward_component_footprint(component, output_size) if use_fallback else baseline
        )
        _, projected_count = ndimage.label(
            projected,
            structure=_INSTANCE_CONNECTIVITY_4,
        )
        if not projected.any() or projected_count != 1:
            raise PanNukeSemanticsError(
                "component-covering projection failed to emit one connected footprint"
            )
        projected_components.append(projected)
        projected_pixel_counts.append(int(projected.sum()))
        fallback_used.append(use_fallback)

    contribution_count = np.sum(
        np.stack(projected_components, axis=0),
        axis=0,
        dtype=np.int32,
    )
    union = contribution_count > 0
    unique_counts = tuple(
        int(np.count_nonzero(component & (contribution_count == 1)))
        for component in projected_components
    )
    if any(count <= 0 for count in unique_counts):
        raise PanNukeSemanticsError(
            "component-covering projection lost a unique raw-component contribution"
        )
    collision_pixels = int(np.count_nonzero(contribution_count > 1))
    collision_excess = int(np.sum(np.maximum(contribution_count - 1, 0)))
    adjacency_pairs = 0
    for left in range(raw_component_count):
        expanded = ndimage.binary_dilation(
            projected_components[left],
            structure=_INSTANCE_CONNECTIVITY_4,
        )
        for right in range(left + 1, raw_component_count):
            if np.any(expanded & projected_components[right]):
                adjacency_pairs += 1
    _, union_component_count = ndimage.label(
        union,
        structure=_INSTANCE_CONNECTIVITY_4,
    )
    if int(union.sum()) != sum(projected_pixel_counts) - collision_excess:
        raise PanNukeSemanticsError("component projection pixel accounting is inconsistent")
    return _ComponentProjection(
        mask=np.asarray(union, dtype=bool),
        raw_component_count=int(raw_component_count),
        projected_union_component_count=int(union_component_count),
        raw_component_pixel_counts=tuple(raw_pixel_counts),
        projected_component_pixel_counts=tuple(projected_pixel_counts),
        projected_component_unique_pixel_counts=unique_counts,
        baseline_projected_component_counts=tuple(baseline_counts),
        fallback_used=tuple(fallback_used),
        collision_pixel_count=collision_pixels,
        collision_excess_count=collision_excess,
        adjacency_pair_count=adjacency_pairs,
    )


def _component_covering_target_crop(
    image: NDArray[np.uint8],
    target_mask: NDArray[np.bool_],
    *,
    output_size: int,
    padding: int,
) -> tuple[TargetCrop, _ComponentProjection, NDArray[np.bool_]]:
    """Crop RGB and project components before any resized-union emptiness check.

    The generic crop helper quite reasonably rejects an empty nearest-neighbour
    resized union.  PanNuke's fixed component-covering policy is stronger: even
    when *all* tiny raw components disappear under that baseline resize, each one
    must receive a deterministic forward-projected footprint.  Reproduce the
    generic helper's source-box and bilinear RGB policy here, then run the
    component projection before constructing the model-facing crop.
    """

    if image.ndim != 3 or image.shape[2] != 3 or target_mask.shape != image.shape[:2]:
        raise ValueError("image/target mask shapes do not align")
    if image.dtype != np.uint8 or target_mask.dtype != np.bool_ or not target_mask.any():
        raise ValueError("component-covering crop requires uint8 RGB and a non-empty bool mask")
    if output_size <= 0 or padding < 0:
        raise ValueError("output_size must be positive and padding non-negative")

    height, width = target_mask.shape
    x0, y0, x1, y1 = mask_bbox(target_mask)
    side = max(x1 - x0, y1 - y0) + 2 * padding
    centre_x = (x0 + x1) / 2.0
    centre_y = (y0 + y1) / 2.0
    crop_x0 = max(0, int(np.floor(centre_x - side / 2.0)))
    crop_y0 = max(0, int(np.floor(centre_y - side / 2.0)))
    crop_x1 = min(width, crop_x0 + side)
    crop_y1 = min(height, crop_y0 + side)
    crop_x0 = max(0, crop_x1 - side)
    crop_y0 = max(0, crop_y1 - side)
    source_box = (crop_x0, crop_y0, crop_x1, crop_y1)
    source_mask = np.asarray(
        target_mask[crop_y0:crop_y1, crop_x0:crop_x1],
        dtype=bool,
    )
    projection = _component_covering_projection(source_mask, output_size)
    resized_image = np.asarray(
        Image.fromarray(image[crop_y0:crop_y1, crop_x0:crop_x1]).resize(
            (output_size, output_size),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )
    return (
        TargetCrop(
            image=resized_image,
            target_mask=projection.mask,
            source_box=source_box,
        ),
        projection,
        source_mask,
    )


@dataclass(frozen=True, slots=True)
class PanNukeCropConfig:
    """Dynamic square-crop and target-highlighting policy."""

    output_size: int = 64
    padding: int = 8
    context_brightness: float = 0.45

    def validate(self) -> None:
        if self.output_size <= 0:
            raise ValueError("output_size must be positive")
        if self.padding < 0:
            raise ValueError("padding must be non-negative")
        if not 0.0 <= self.context_brightness <= 1.0:
            raise ValueError("context_brightness must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class PanNukeCropBatch:
    """Aligned component-covering crops with exact raw-instance provenance."""

    sample_ids: NDArray[np.str_]
    context_rgb: UInt8Images
    target_highlighted_rgb: UInt8Images
    target_masks: BoolMasks
    target_contour_masks: BoolMasks
    raw_component_counts: NDArray[np.int32]
    disconnected_instance_flags: NDArray[np.bool_]
    projected_union_component_counts: NDArray[np.int32]
    projection_fallback_component_counts: NDArray[np.int32]
    projection_collision_pixel_counts: NDArray[np.int32]
    projection_collision_excess_counts: NDArray[np.int32]
    projection_adjacency_pair_counts: NDArray[np.int32]
    projection_topology_changed: NDArray[np.bool_]
    projected_component_pixel_counts: NDArray[np.int32]
    projected_component_unique_pixel_counts: NDArray[np.int32]
    baseline_projected_component_counts: NDArray[np.int32]
    projection_fallback_component_flags: NDArray[np.bool_]
    projected_component_offsets: NDArray[np.int64]
    source_crop_boxes: NDArray[np.int32]
    source_target_boxes: NDArray[np.int32]
    official_folds: NDArray[np.int16]
    source_patch_indices: NDArray[np.int32]
    instance_channel_indices: NDArray[np.int16]
    instance_ids: NDArray[np.int64]
    pre_corruption_labels: NDArray[np.int64]
    group_ids: NDArray[np.str_]
    tissue_types: NDArray[np.str_]
    source_contours_xy: tuple[NDArray[np.int32], ...]
    identity_verified: NDArray[np.bool_]
    primary_eligible: NDArray[np.bool_]
    confirmatory_eligible: NDArray[np.bool_]
    metadata: dict[str, Any]
    validation_binding: PanNukeValidationResult = dataclass_field(repr=False, compare=False)

    def validate(self) -> None:
        n = len(self.sample_ids)
        size = self.context_rgb.shape[1] if self.context_rgb.ndim == 4 else -1
        if n == 0 or self.context_rgb.shape != (n, size, size, 3) or size <= 0:
            raise ValueError("context crops must have non-empty square RGB shape")
        if self.context_rgb.dtype != np.uint8:
            raise ValueError("context crops must be uint8")
        if self.target_highlighted_rgb.shape != self.context_rgb.shape:
            raise ValueError("highlighted and context RGB crops must align")
        if self.target_highlighted_rgb.dtype != np.uint8:
            raise ValueError("highlighted crops must be uint8")
        expected_mask_shape = (n, size, size)
        if self.target_masks.shape != expected_mask_shape:
            raise ValueError("target masks do not align with crops")
        if self.target_contour_masks.shape != expected_mask_shape:
            raise ValueError("target contour masks do not align with crops")
        if not np.all(self.target_masks.reshape(n, -1).any(axis=1)):
            raise ValueError("every crop must retain its exact target")
        if np.any(self.target_contour_masks & ~self.target_masks):
            raise ValueError("target contour pixels must belong to the target mask")
        if not np.all(self.target_contour_masks.reshape(n, -1).any(axis=1)):
            raise ValueError("every target must have a non-empty pixel contour")
        crop_configuration = self.metadata.get("crop_configuration")
        if not isinstance(crop_configuration, dict):
            raise ValueError("crop batch lacks its exact crop configuration")
        context_brightness = crop_configuration.get("context_brightness")
        if (
            isinstance(context_brightness, bool)
            or not isinstance(context_brightness, (int, float))
            or not 0.0 <= float(context_brightness) <= 1.0
            or any(
                not np.array_equal(
                    highlighted,
                    highlight_target(
                        context,
                        mask,
                        context_brightness=float(context_brightness),
                    ),
                )
                for context, mask, highlighted in zip(
                    self.context_rgb,
                    self.target_masks,
                    self.target_highlighted_rgb,
                    strict=True,
                )
            )
        ):
            raise ValueError(
                "highlighted crops differ from context/mask/context-brightness provenance"
            )
        aligned_vectors = (
            self.official_folds,
            self.source_patch_indices,
            self.instance_channel_indices,
            self.instance_ids,
            self.pre_corruption_labels,
            self.group_ids,
            self.tissue_types,
            self.identity_verified,
            self.primary_eligible,
            self.confirmatory_eligible,
            self.raw_component_counts,
            self.disconnected_instance_flags,
            self.projected_union_component_counts,
            self.projection_fallback_component_counts,
            self.projection_collision_pixel_counts,
            self.projection_collision_excess_counts,
            self.projection_adjacency_pair_counts,
            self.projection_topology_changed,
        )
        if any(value.shape != (n,) for value in aligned_vectors):
            raise ValueError("crop identity vectors do not align")
        if self.source_crop_boxes.shape != (n, 4) or self.source_target_boxes.shape != (n, 4):
            raise ValueError("source boxes must have shape (n, 4)")
        if len(set(self.sample_ids.tolist())) != n or any(not value for value in self.sample_ids):
            raise ValueError("crop sample IDs must be unique and non-empty")
        if not self.identity_verified.all():
            raise ValueError("an emitted crop lacks raw-instance identity verification")
        if not np.array_equal(self.primary_eligible, self.confirmatory_eligible):
            raise ValueError("crop primary/confirmatory eligibility masks differ")
        integer_projection_vectors = (
            self.raw_component_counts,
            self.projected_union_component_counts,
            self.projection_fallback_component_counts,
            self.projection_collision_pixel_counts,
            self.projection_collision_excess_counts,
            self.projection_adjacency_pair_counts,
        )
        if any(value.dtype != np.int32 for value in integer_projection_vectors):
            raise ValueError("crop component-projection vectors must be int32")
        recomputed_union_component_counts = np.asarray(
            [
                int(
                    ndimage.label(
                        mask,
                        structure=_INSTANCE_CONNECTIVITY_4,
                    )[1]
                )
                for mask in self.target_masks
            ],
            dtype=np.int32,
        )
        maximum_adjacency_pairs = (
            self.raw_component_counts.astype(np.int64)
            * (self.raw_component_counts.astype(np.int64) - 1)
            // 2
        )
        if (
            self.disconnected_instance_flags.dtype != np.bool_
            or self.projection_topology_changed.dtype != np.bool_
            or not np.array_equal(
                self.disconnected_instance_flags,
                self.raw_component_counts > 1,
            )
            or np.any(self.raw_component_counts < 1)
            or np.any(self.projected_union_component_counts < 1)
            or np.any(self.projected_union_component_counts > self.raw_component_counts)
            or not np.array_equal(
                self.projected_union_component_counts,
                recomputed_union_component_counts,
            )
            or np.any(self.projection_fallback_component_counts < 0)
            or np.any(self.projection_fallback_component_counts > self.raw_component_counts)
            or np.any(self.projection_collision_pixel_counts < 0)
            or np.any(self.projection_collision_excess_counts < 0)
            or np.any(
                self.projection_collision_excess_counts < self.projection_collision_pixel_counts
            )
            or np.any(
                (self.projection_collision_pixel_counts == 0)
                != (self.projection_collision_excess_counts == 0)
            )
            or np.any(self.projection_adjacency_pair_counts < 0)
            or np.any(
                self.projection_adjacency_pair_counts.astype(np.int64) > maximum_adjacency_pairs
            )
            or not np.array_equal(
                self.projection_topology_changed,
                self.projected_union_component_counts != self.raw_component_counts,
            )
        ):
            raise ValueError("crop component-projection vectors are inconsistent")
        if (
            self.projected_component_offsets.shape != (n + 1,)
            or self.projected_component_offsets.dtype != np.int64
            or self.projected_component_offsets[0] != 0
            or self.projected_component_offsets[-1] != len(self.projected_component_pixel_counts)
            or not np.array_equal(
                np.diff(self.projected_component_offsets),
                self.raw_component_counts,
            )
            or self.projected_component_pixel_counts.dtype != np.int32
            or self.projected_component_unique_pixel_counts.dtype != np.int32
            or self.projected_component_unique_pixel_counts.shape
            != self.projected_component_pixel_counts.shape
            or self.baseline_projected_component_counts.shape
            != self.projected_component_pixel_counts.shape
            or self.baseline_projected_component_counts.dtype != np.int32
            or self.projection_fallback_component_flags.shape
            != self.projected_component_pixel_counts.shape
            or self.projection_fallback_component_flags.dtype != np.bool_
            or np.any(self.projected_component_pixel_counts <= 0)
            or np.any(self.projected_component_unique_pixel_counts <= 0)
            or np.any(self.baseline_projected_component_counts < 0)
            or not np.array_equal(
                self.projection_fallback_component_flags,
                self.baseline_projected_component_counts != 1,
            )
            or np.any(
                self.projected_component_unique_pixel_counts > self.projected_component_pixel_counts
            )
        ):
            raise ValueError("crop per-component projection ledger is inconsistent")
        for index in range(n):
            start = int(self.projected_component_offsets[index])
            stop = int(self.projected_component_offsets[index + 1])
            projected_sum = int(self.projected_component_pixel_counts[start:stop].sum())
            if int(self.target_masks[index].sum()) != projected_sum - int(
                self.projection_collision_excess_counts[index]
            ) or int(np.count_nonzero(self.projection_fallback_component_flags[start:stop])) != int(
                self.projection_fallback_component_counts[index]
            ):
                raise ValueError("crop projected-mask pixel accounting is inconsistent")
        if len(self.source_contours_xy) != n:
            raise ValueError("source contours do not align with crops")
        if any(
            contour.ndim != 2 or contour.shape[1] != 2 or not len(contour)
            for contour in self.source_contours_xy
        ):
            raise ValueError("source contours must be non-empty (x, y) coordinate matrices")
        eligibility = self.metadata.get("analysis_eligibility")
        if not isinstance(eligibility, dict):
            raise ValueError("crop batch lacks eligibility provenance")
        if eligibility.get("selection_scope") == "analysis":
            validate_analysis_eligibility_provenance(
                self.metadata,
                self.sample_ids,
                primary_eligible=self.primary_eligible,
                confirmatory_eligible=self.confirmatory_eligible,
            )
        elif eligibility.get("selection_scope") != "review_only":
            raise ValueError("crop batch has an unknown eligibility scope")
        if (
            Path(str(self.metadata.get("raw_root", ""))).resolve()
            != self.validation_binding.root.resolve()
        ):
            raise ValueError("crop batch raw-root binding differs from validation evidence")
        projection = self.metadata.get("target_mask_projection")
        if not isinstance(projection, dict):
            raise ValueError("crop batch lacks target-mask projection provenance")
        semantic_sha256 = projection.get("semantic_sha256")
        semantic_payload = dict(projection)
        semantic_payload.pop("semantic_sha256", None)
        records = projection.get("disconnected_instances")
        sample_ids = self.sample_ids.tolist()
        expected_fallback_ids = [
            str(sample_id)
            for sample_id, count in zip(
                sample_ids,
                self.projection_fallback_component_counts.tolist(),
                strict=True,
            )
            if count > 0
        ]
        expected_collision_ids = [
            str(sample_id)
            for sample_id, count in zip(
                sample_ids,
                self.projection_collision_pixel_counts.tolist(),
                strict=True,
            )
            if count > 0
        ]
        expected_adjacency_ids = [
            str(sample_id)
            for sample_id, count in zip(
                sample_ids,
                self.projection_adjacency_pair_counts.tolist(),
                strict=True,
            )
            if count > 0
        ]
        expected_topology_ids = [
            str(sample_id)
            for sample_id, changed in zip(
                sample_ids,
                self.projection_topology_changed.tolist(),
                strict=True,
            )
            if changed
        ]
        if (
            projection.get("schema_version") != 1
            or projection.get("identifier") != _COMPONENT_PROJECTION_ID
            or projection.get("raw_component_connectivity") != "4-connected"
            or projection.get("quality_flag") != _DISCONNECTED_INSTANCE_FLAG
            or projection.get("raw_identity_action")
            != "retain_one_raw_identity_without_split_merge_repair_or_relabel"
            or projection.get("manifest_and_validation_must_agree_for_disconnected_identity")
            is not True
            or projection.get("all_raw_components_must_contribute") is not True
            or projection.get("projected_union_topology_is_exact") is not False
            or projection.get("source_annotations_modified") is not False
            or projection.get("zero_covered_component_count") != 0
            or projection.get("sample_count") != n
            or projection.get("raw_component_count") != int(self.raw_component_counts.sum())
            or projection.get("disconnected_instance_count")
            != int(self.disconnected_instance_flags.sum())
            or projection.get("fallback_component_count")
            != int(self.projection_fallback_component_counts.sum())
            or projection.get("fallback_instance_ids") != expected_fallback_ids
            or projection.get("collision_instance_count")
            != int(np.count_nonzero(self.projection_collision_pixel_counts))
            or projection.get("collision_instance_ids") != expected_collision_ids
            or projection.get("adjacency_instance_count")
            != int(np.count_nonzero(self.projection_adjacency_pair_counts))
            or projection.get("adjacency_instance_ids") != expected_adjacency_ids
            or projection.get("topology_changed_instance_count")
            != int(self.projection_topology_changed.sum())
            or projection.get("topology_changed_instance_ids") != expected_topology_ids
            or semantic_sha256 != canonical_sha256(semantic_payload)
            or not isinstance(records, list)
        ):
            raise ValueError("crop batch target-mask projection policy is invalid")
        expected_record_ids = tuple(
            str(sample_id)
            for sample_id, flagged in zip(
                self.sample_ids.tolist(),
                self.disconnected_instance_flags.tolist(),
                strict=True,
            )
            if flagged
        )
        record_ids = tuple(
            str(record.get("sample_id")) if isinstance(record, dict) else "" for record in records
        )
        if record_ids != expected_record_ids:
            raise ValueError("crop disconnected-instance ledger is not sample-aligned")
        index_by_id = {
            str(sample_id): index for index, sample_id in enumerate(self.sample_ids.tolist())
        }
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("crop batch disconnected-instance record is invalid")
            sample_id = str(record.get("sample_id"))
            index = index_by_id[sample_id]
            start = int(self.projected_component_offsets[index])
            stop = int(self.projected_component_offsets[index + 1])
            mask_sha256 = record.get("raw_target_mask_sha256")
            projected_mask_sha256 = record.get("projected_target_mask_sha256")
            components = record.get("components")
            if (
                type(record.get("component_count")) is not int
                or record["component_count"] != int(self.raw_component_counts[index])
                or type(record.get("area")) is not int
                or record["area"] <= 0
                or not isinstance(record.get("bbox"), list)
                or len(record["bbox"]) != 4
                or not isinstance(mask_sha256, str)
                or len(mask_sha256) != 64
                or any(character not in "0123456789abcdef" for character in mask_sha256)
                or not isinstance(projected_mask_sha256, str)
                or projected_mask_sha256 != array_artifact_sha256(self.target_masks[index])
                or type(record.get("primary_eligible")) is not bool
                or type(record.get("confirmatory_eligible")) is not bool
                or record["primary_eligible"] is not record["confirmatory_eligible"]
                or record["primary_eligible"] is not bool(self.primary_eligible[index])
                or record.get("projected_union_component_count")
                != int(self.projected_union_component_counts[index])
                or record.get("fallback_component_count")
                != int(self.projection_fallback_component_counts[index])
                or record.get("collision_pixel_count")
                != int(self.projection_collision_pixel_counts[index])
                or record.get("collision_excess_count")
                != int(self.projection_collision_excess_counts[index])
                or record.get("adjacency_pair_count")
                != int(self.projection_adjacency_pair_counts[index])
                or record.get("topology_changed")
                is not bool(self.projection_topology_changed[index])
                or record.get("projected_component_pixel_counts")
                != self.projected_component_pixel_counts[start:stop].tolist()
                or record.get("projected_component_unique_pixel_counts")
                != self.projected_component_unique_pixel_counts[start:stop].tolist()
                or not isinstance(components, list)
                or len(components) != int(self.raw_component_counts[index])
            ):
                raise ValueError("crop batch disconnected-instance record is invalid")
            raw_area = 0
            fallback_count = 0
            for component_index, component in enumerate(components, start=1):
                component_hash = (
                    component.get("raw_component_mask_sha256")
                    if isinstance(component, dict)
                    else None
                )
                if (
                    not isinstance(component, dict)
                    or component.get("component_index") != component_index
                    or type(component.get("raw_pixel_count")) is not int
                    or component["raw_pixel_count"] <= 0
                    or not isinstance(component_hash, str)
                    or len(component_hash) != 64
                    or any(character not in "0123456789abcdef" for character in component_hash)
                    or type(component.get("baseline_projected_component_count")) is not int
                    or component["baseline_projected_component_count"]
                    != int(self.baseline_projected_component_counts[start + component_index - 1])
                    or type(component.get("fallback_used")) is not bool
                    or component["fallback_used"]
                    is not bool(
                        self.projection_fallback_component_flags[start + component_index - 1]
                    )
                    or component.get("projected_pixel_count")
                    != int(self.projected_component_pixel_counts[start + component_index - 1])
                    or component.get("projected_unique_pixel_count")
                    != int(
                        self.projected_component_unique_pixel_counts[start + component_index - 1]
                    )
                ):
                    raise ValueError("crop disconnected component ledger is invalid")
                raw_area += int(component["raw_pixel_count"])
                fallback_count += int(component["fallback_used"])
            if raw_area != record["area"] or fallback_count != record["fallback_component_count"]:
                raise ValueError("crop disconnected component ledger does not reconcile")


@dataclass(frozen=True, slots=True)
class PanNukeRepresentationArtifacts:
    """Paths and in-memory outputs from one aligned representation extraction."""

    crops: PanNukeCropBatch
    engineered: EngineeredFeatureSet
    embeddings: EmbeddingResult
    crop_cache_path: Path
    crop_metadata_path: Path
    engineered_cache_path: Path
    engineered_metadata_path: Path
    context_embeddings: EmbeddingResult | None = None
    context_morphometrics: ContextMorphometricsCache | None = None
    publication_records: tuple[PublishedPath, ...] = ()


@dataclass(frozen=True, slots=True)
class ContextMorphometricsCache:
    """Explicit aligned context-embedding plus target-morphometrics ablation."""

    values: NDArray[np.float32]
    names: tuple[str, ...]
    sample_ids: NDArray[np.str_]
    metadata: dict[str, Any]
    cache_path: Path
    metadata_path: Path

    def validate(self) -> None:
        if self.values.ndim != 2 or self.values.shape[0] != len(self.sample_ids):
            raise ValueError("context+morphometrics matrix must align with sample IDs")
        if self.values.shape[1] != len(self.names) or not self.names:
            raise ValueError("context+morphometrics names must align with columns")
        if len(set(self.names)) != len(self.names):
            raise ValueError("context+morphometrics names must be unique")
        if len(set(self.sample_ids.tolist())) != len(self.sample_ids):
            raise ValueError("context+morphometrics sample IDs must be unique")
        if self.values.dtype != np.float32 or not np.isfinite(self.values).all():
            raise ValueError("context+morphometrics values must be finite float32")


def _channel_last_patch(
    array: NDArray[np.generic], index: int, channel_axis: int
) -> NDArray[np.generic]:
    patch = np.asarray(array[index])
    patch_axis = channel_axis - 1
    return np.moveaxis(patch, patch_axis, -1) if patch_axis != patch.ndim - 1 else patch


def _uint8_patch(image: NDArray[np.generic], *, sample_id: str) -> NDArray[np.uint8]:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise PanNukeSemanticsError(f"{sample_id}: source image is not channel-last RGB")
    if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
        raise PanNukeSemanticsError(f"{sample_id}: source image is not finite numeric RGB")
    converted = image.astype(np.float64)
    if float(converted.min()) < 0.0 or float(converted.max()) > 255.0:
        raise PanNukeSemanticsError(f"{sample_id}: source RGB values lie outside [0, 255]")
    return np.rint(converted).astype(np.uint8)


def _pixel_boundary(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    structure = ndimage.generate_binary_structure(2, 1)
    return mask & ~ndimage.binary_erosion(mask, structure=structure, border_value=0)


def _verify_validation_binding(
    validation: PanNukeValidationResult,
    *,
    verify_inventory: bool = True,
) -> None:
    if not validation.release_complete or not validation.inventory:
        raise PanNukeSemanticsError("a complete validated PanNuke release is required")
    if any(facts.validation_scope != "full_semantic_scan" for facts in validation.fold_validation):
        raise PanNukeSemanticsError("all PanNuke folds require a full semantic scan")
    if verify_inventory:
        verify_raw_inventory_unchanged(validation)


def _manifest_frame(
    manifest_path: Path,
    sample_ids: tuple[str, ...] | None,
    *,
    eligibility_scope: Literal["analysis", "review_only"],
) -> tuple[Any, dict[str, Any], str]:
    manifest_sha256 = sha256_file(manifest_path)
    table = pq.read_table(manifest_path)
    validate_manifest_invariants(table)
    selection = select_manifest_rows(
        table,
        sample_ids=sample_ids,
        scope=eligibility_scope,
    )
    if sha256_file(manifest_path) != manifest_sha256:
        raise PanNukeSemanticsError("PanNuke manifest changed during parsing")
    return selection.frame, selection.provenance, manifest_sha256


def _row_int(row: Any, field: str) -> int:
    value = row[field]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise PanNukeSemanticsError(f"manifest field {field} is missing")
    return int(value)


def _row_quality_flags(row: Any, *, sample_id: str) -> tuple[str, ...]:
    value = row["quality_flags"]
    if isinstance(value, np.ndarray):
        raw = value.tolist()
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raise PanNukeSemanticsError(f"{sample_id}: manifest quality_flags is not a list")
    flags = tuple(str(item) for item in raw)
    if any(not flag for flag in flags) or len(flags) != len(set(flags)):
        raise PanNukeSemanticsError(f"{sample_id}: manifest quality_flags is invalid")
    return flags


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise PanNukeSemanticsError(
            f"validated source path lies outside release root: {path}"
        ) from error


def extract_pannuke_crop_batch(
    validation: PanNukeValidationResult,
    manifest_path: str | Path,
    *,
    sample_ids: tuple[str, ...] | None = None,
    config: PanNukeCropConfig | None = None,
    eligibility_scope: Literal["analysis", "review_only"] = "analysis",
    _selection_override: tuple[Any, dict[str, Any], str] | None = None,
    _verify_raw_sources: bool = True,
) -> PanNukeCropBatch:
    """Resolve selected manifest rows back to exact raw PanNuke instances."""

    settings = config or PanNukeCropConfig()
    settings.validate()
    _verify_validation_binding(validation, verify_inventory=_verify_raw_sources)
    source_manifest = Path(manifest_path).resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(f"PanNuke manifest does not exist: {source_manifest}")
    if _selection_override is None:
        frame, eligibility_provenance, manifest_sha256 = _manifest_frame(
            source_manifest,
            sample_ids,
            eligibility_scope=eligibility_scope,
        )
    else:
        if sample_ids is not None:
            raise ValueError("internal manifest selection cannot be combined with sample_ids")
        frame, eligibility_provenance, manifest_sha256 = _selection_override
        if sha256_file(source_manifest) != manifest_sha256:
            raise PanNukeSemanticsError("PanNuke manifest changed before crop extraction")
    folds = {fold.fold_id: fold for fold in validation.folds}
    facts = {item.fold_id: item for item in validation.fold_validation}
    images_by_fold = {fold_id: open_npy_mmap(fold.image_path) for fold_id, fold in folds.items()}
    masks_by_fold = {fold_id: open_npy_mmap(fold.mask_path) for fold_id, fold in folds.items()}

    context_images: list[NDArray[np.uint8]] = []
    highlighted_images: list[NDArray[np.uint8]] = []
    resized_masks: list[NDArray[np.bool_]] = []
    resized_contours: list[NDArray[np.bool_]] = []
    crop_boxes: list[tuple[int, int, int, int]] = []
    target_boxes: list[tuple[int, int, int, int]] = []
    contours: list[NDArray[np.int32]] = []
    verified: list[bool] = []
    disconnected_instances: list[dict[str, Any]] = []
    raw_component_counts: list[int] = []
    disconnected_flags: list[bool] = []
    projected_union_component_counts: list[int] = []
    fallback_component_counts: list[int] = []
    collision_pixel_counts: list[int] = []
    collision_excess_counts: list[int] = []
    adjacency_pair_counts: list[int] = []
    topology_changed: list[bool] = []
    projected_component_pixel_counts: list[int] = []
    projected_component_unique_pixel_counts: list[int] = []
    baseline_projected_component_counts: list[int] = []
    projection_fallback_component_flags: list[bool] = []
    projected_component_offsets: list[int] = [0]
    for _, row in frame.iterrows():
        sample_id = str(row["sample_id"])
        fold_id = _row_int(row, "official_fold")
        if fold_id not in folds or fold_id not in facts:
            raise PanNukeSemanticsError(f"{sample_id}: manifest references an unvalidated fold")
        fold = folds[fold_id]
        fold_facts = facts[fold_id]
        patch_index = _row_int(row, "source_patch_index")
        if _row_int(row, "patch_index") != patch_index:
            raise PanNukeSemanticsError(f"{sample_id}: duplicate manifest patch indices disagree")
        if not 0 <= patch_index < fold_facts.n_patches:
            raise PanNukeSemanticsError(f"{sample_id}: source patch index is out of range")
        expected_paths = {
            "source_image_path": _relative_to_root(fold.image_path, validation.root),
            "source_mask_path": _relative_to_root(fold.mask_path, validation.root),
            "source_tissue_path": _relative_to_root(fold.tissue_path, validation.root),
        }
        for field, expected_path in expected_paths.items():
            if str(row[field]).replace("\\", "/") != expected_path:
                raise PanNukeSemanticsError(
                    f"{sample_id}: manifest {field} differs from validated fold source"
                )
        class_index = _row_int(row, "nucleus_class_index")
        if not 0 <= class_index < len(validation.mapping.class_names):
            raise PanNukeSemanticsError(f"{sample_id}: class index is out of range")
        channel_index = _row_int(row, "instance_channel_index")
        expected_channel = fold_facts.positive_channel_indices[class_index]
        if channel_index != expected_channel:
            raise PanNukeSemanticsError(
                f"{sample_id}: class/channel identity differs from validated mapping"
            )
        if str(row["nucleus_class_name"]) != validation.mapping.class_names[class_index]:
            raise PanNukeSemanticsError(f"{sample_id}: class name differs from validated mapping")
        instance_id = _row_int(row, "instance_id")
        mask_patch = _channel_last_patch(
            masks_by_fold[fold_id], patch_index, fold.mask_channel_axis
        )
        if channel_index >= mask_patch.shape[-1]:
            raise PanNukeSemanticsError(f"{sample_id}: instance channel is out of range")
        target_mask = np.asarray(mask_patch[..., channel_index] == instance_id, dtype=bool)
        if not target_mask.any():
            raise PanNukeSemanticsError(
                f"{sample_id}: instance ID {instance_id} is absent from its raw class channel"
            )
        quality_flags = _row_quality_flags(row, sample_id=sample_id)
        raw_component_labels, component_count = ndimage.label(
            target_mask,
            structure=_INSTANCE_CONNECTIVITY_4,
        )
        flagged_disconnected = _DISCONNECTED_INSTANCE_FLAG in quality_flags
        raw_disconnected = component_count > 1
        if raw_disconnected != flagged_disconnected:
            if raw_disconnected:
                raise PanNukeSemanticsError(
                    f"{sample_id}: raw target is disconnected but manifest lacks "
                    f"{_DISCONNECTED_INSTANCE_FLAG}"
                )
            raise PanNukeSemanticsError(
                f"{sample_id}: manifest {_DISCONNECTED_INSTANCE_FLAG} differs from connected "
                "raw target"
            )
        if raw_disconnected and (
            validation.qc_policy.disconnected_instance_ids_are_fatal
            or validation.qc_policy.source_masks_modified
            or fold_facts.disconnected_instance_count_full_scan <= 0
            or fold_facts.disconnected_patch_count_full_scan <= 0
        ):
            raise PanNukeSemanticsError(
                f"{sample_id}: disconnected raw target lacks compatible validated metadata"
            )
        raw_box = mask_bbox(target_mask)
        manifest_box = tuple(
            _row_int(row, field)
            for field in ("bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max")
        )
        if raw_box != manifest_box or int(target_mask.sum()) != _row_int(row, "area"):
            raise PanNukeSemanticsError(f"{sample_id}: raw target geometry differs from manifest")
        raw_y, raw_x = np.nonzero(target_mask)
        if not np.isclose(
            float(raw_x.mean()), float(row["centroid_x"]), atol=1e-9
        ) or not np.isclose(float(raw_y.mean()), float(row["centroid_y"]), atol=1e-9):
            raise PanNukeSemanticsError(f"{sample_id}: raw target centroid differs from manifest")
        if _row_int(row, "pre_corruption_label") != class_index:
            raise PanNukeSemanticsError(
                f"{sample_id}: immutable reference label differs from class"
            )
        image_patch = _uint8_patch(
            _channel_last_patch(images_by_fold[fold_id], patch_index, fold.image_channel_axis),
            sample_id=sample_id,
        )
        if image_patch.shape[:2] != target_mask.shape:
            raise PanNukeSemanticsError(f"{sample_id}: image/mask patch alignment changed")
        target_crop, projection, source_crop_mask = _component_covering_target_crop(
            image_patch,
            target_mask,
            output_size=settings.output_size,
            padding=settings.padding,
        )
        if projection.raw_component_count != component_count:
            raise PanNukeSemanticsError(
                f"{sample_id}: crop projection changed the raw component inventory"
            )
        fallback_count = int(sum(projection.fallback_used))
        if fallback_count == 0 and not np.array_equal(
            projection.mask,
            _resize_binary_nearest(source_crop_mask, settings.output_size),
        ):
            raise PanNukeSemanticsError(
                f"{sample_id}: component projection drifted from valid nearest-neighbour mask"
            )
        raw_component_counts.append(int(component_count))
        disconnected_flags.append(raw_disconnected)
        projected_union_component_counts.append(projection.projected_union_component_count)
        fallback_component_counts.append(fallback_count)
        collision_pixel_counts.append(projection.collision_pixel_count)
        collision_excess_counts.append(projection.collision_excess_count)
        adjacency_pair_counts.append(projection.adjacency_pair_count)
        topology_changed.append(projection.projected_union_component_count != component_count)
        projected_component_pixel_counts.extend(projection.projected_component_pixel_counts)
        projected_component_unique_pixel_counts.extend(
            projection.projected_component_unique_pixel_counts
        )
        baseline_projected_component_counts.extend(projection.baseline_projected_component_counts)
        projection_fallback_component_flags.extend(projection.fallback_used)
        projected_component_offsets.append(projected_component_offsets[-1] + component_count)
        if raw_disconnected:
            disconnected_instances.append(
                {
                    "sample_id": sample_id,
                    "raw_instance_identity": str(row["raw_instance_identity"]),
                    "component_count": int(component_count),
                    "area": int(target_mask.sum()),
                    "bbox": list(raw_box),
                    "centroid": [float(raw_x.mean()), float(raw_y.mean())],
                    "raw_target_mask_sha256": array_artifact_sha256(target_mask),
                    "projected_target_mask_sha256": array_artifact_sha256(projection.mask),
                    "projected_union_component_count": (projection.projected_union_component_count),
                    "fallback_component_count": fallback_count,
                    "collision_pixel_count": projection.collision_pixel_count,
                    "collision_excess_count": projection.collision_excess_count,
                    "adjacency_pair_count": projection.adjacency_pair_count,
                    "topology_changed": (
                        projection.projected_union_component_count != component_count
                    ),
                    "projected_component_pixel_counts": list(
                        projection.projected_component_pixel_counts
                    ),
                    "projected_component_unique_pixel_counts": list(
                        projection.projected_component_unique_pixel_counts
                    ),
                    "components": [
                        {
                            "component_index": component_index,
                            "raw_pixel_count": projection.raw_component_pixel_counts[
                                component_index - 1
                            ],
                            "raw_component_mask_sha256": array_artifact_sha256(
                                np.asarray(
                                    raw_component_labels == component_index,
                                    dtype=bool,
                                )
                            ),
                            "baseline_projected_component_count": (
                                projection.baseline_projected_component_counts[component_index - 1]
                            ),
                            "fallback_used": projection.fallback_used[component_index - 1],
                            "projected_pixel_count": (
                                projection.projected_component_pixel_counts[component_index - 1]
                            ),
                            "projected_unique_pixel_count": (
                                projection.projected_component_unique_pixel_counts[
                                    component_index - 1
                                ]
                            ),
                        }
                        for component_index in range(1, component_count + 1)
                    ],
                    "primary_eligible": bool(row["primary_eligible"]),
                    "confirmatory_eligible": bool(row["confirmatory_eligible"]),
                }
            )
        context_images.append(target_crop.image)
        resized_masks.append(projection.mask)
        resized_contours.append(_pixel_boundary(projection.mask))
        highlighted_images.append(
            highlight_target(
                target_crop.image,
                projection.mask,
                context_brightness=settings.context_brightness,
            )
        )
        crop_boxes.append(target_crop.source_box)
        target_boxes.append(raw_box)
        source_boundary = _pixel_boundary(target_mask)
        source_y, source_x = np.nonzero(source_boundary)
        contours.append(np.column_stack((source_x, source_y)).astype(np.int32, copy=False))
        verified.append(True)

    selected_sample_ids = frame["sample_id"].astype(str).tolist()
    fallback_instance_ids = [
        sample_id
        for sample_id, count in zip(
            selected_sample_ids,
            fallback_component_counts,
            strict=True,
        )
        if count > 0
    ]
    collision_instance_ids = [
        sample_id
        for sample_id, count in zip(
            selected_sample_ids,
            collision_pixel_counts,
            strict=True,
        )
        if count > 0
    ]
    adjacency_instance_ids = [
        sample_id
        for sample_id, count in zip(
            selected_sample_ids,
            adjacency_pair_counts,
            strict=True,
        )
        if count > 0
    ]
    topology_changed_instance_ids = [
        sample_id
        for sample_id, changed in zip(
            selected_sample_ids,
            topology_changed,
            strict=True,
        )
        if changed
    ]
    projection_policy: dict[str, Any] = {
        "schema_version": 1,
        "identifier": _COMPONENT_PROJECTION_ID,
        "raw_component_connectivity": "4-connected",
        "quality_flag": _DISCONNECTED_INSTANCE_FLAG,
        "raw_identity_action": ("retain_one_raw_identity_without_split_merge_repair_or_relabel"),
        "manifest_and_validation_must_agree_for_disconnected_identity": True,
        "model_facing_mask_definition": (
            "binary union of independently projected raw 4-connected components"
        ),
        "nearest_policy": (
            "retain per-component PIL nearest-neighbour output byte-for-byte when it is "
            "non-empty and exactly one 4-connected footprint"
        ),
        "fallback_policy": (
            "replace only a lost or split nearest-neighbour component with its "
            "deterministic source-pixel forward footprint"
        ),
        "all_raw_components_must_contribute": True,
        "all_projected_component_footprints_are_4_connected": True,
        "projected_union_topology_is_exact": False,
        "projected_morphology_semantics": (
            "model features are computed on the derived component-covering projected "
            "binary union, not on exact source-resolution topology"
        ),
        "sample_count": len(frame),
        "raw_component_count": int(sum(raw_component_counts)),
        "zero_covered_component_count": 0,
        "disconnected_instance_count": int(sum(disconnected_flags)),
        "fallback_component_count": int(sum(fallback_component_counts)),
        "fallback_instance_ids": fallback_instance_ids,
        "collision_instance_count": int(sum(value > 0 for value in collision_pixel_counts)),
        "collision_instance_ids": collision_instance_ids,
        "adjacency_instance_count": int(sum(value > 0 for value in adjacency_pair_counts)),
        "adjacency_instance_ids": adjacency_instance_ids,
        "topology_changed_instance_count": int(sum(topology_changed)),
        "topology_changed_instance_ids": topology_changed_instance_ids,
        "disconnected_instances": disconnected_instances,
        "source_annotations_modified": False,
    }
    projection_policy["semantic_sha256"] = canonical_sha256(projection_policy)
    metadata = {
        "schema_version": 1,
        "dataset": "PanNuke",
        "manifest_path": str(source_manifest),
        "manifest_sha256": manifest_sha256,
        "raw_root": str(validation.root.resolve()),
        "raw_inventory_sha256": hashlib.sha256(
            json.dumps(
                [record.as_dict() for record in validation.inventory],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "class_mapping": validation.mapping.as_dict(),
        "crop_configuration": asdict(settings),
        "target_identity_policy": (
            "raw class channel plus instance ID remains authoritative; source-resolution "
            "union bbox, area, centroid, component inventory and immutable reference class "
            "are revalidated; the model-facing mask is a separately provenance-bound "
            "component-covering projection and does not claim exact union topology"
        ),
        "target_mask_projection": projection_policy,
        "contour_policy": "four-connected boundary pixels in source-patch (x, y) coordinates",
        "source_annotations_modified": False,
        "sample_count": len(frame),
        "analysis_eligibility": eligibility_provenance,
    }
    batch = PanNukeCropBatch(
        sample_ids=np.asarray(frame["sample_id"].astype(str).tolist(), dtype=np.str_),
        context_rgb=np.stack(context_images).astype(np.uint8, copy=False),
        target_highlighted_rgb=np.stack(highlighted_images).astype(np.uint8, copy=False),
        target_masks=np.stack(resized_masks).astype(bool, copy=False),
        target_contour_masks=np.stack(resized_contours).astype(bool, copy=False),
        raw_component_counts=np.asarray(raw_component_counts, dtype=np.int32),
        disconnected_instance_flags=np.asarray(disconnected_flags, dtype=bool),
        projected_union_component_counts=np.asarray(
            projected_union_component_counts,
            dtype=np.int32,
        ),
        projection_fallback_component_counts=np.asarray(
            fallback_component_counts,
            dtype=np.int32,
        ),
        projection_collision_pixel_counts=np.asarray(
            collision_pixel_counts,
            dtype=np.int32,
        ),
        projection_collision_excess_counts=np.asarray(
            collision_excess_counts,
            dtype=np.int32,
        ),
        projection_adjacency_pair_counts=np.asarray(
            adjacency_pair_counts,
            dtype=np.int32,
        ),
        projection_topology_changed=np.asarray(topology_changed, dtype=bool),
        projected_component_pixel_counts=np.asarray(
            projected_component_pixel_counts,
            dtype=np.int32,
        ),
        projected_component_unique_pixel_counts=np.asarray(
            projected_component_unique_pixel_counts,
            dtype=np.int32,
        ),
        baseline_projected_component_counts=np.asarray(
            baseline_projected_component_counts,
            dtype=np.int32,
        ),
        projection_fallback_component_flags=np.asarray(
            projection_fallback_component_flags,
            dtype=bool,
        ),
        projected_component_offsets=np.asarray(
            projected_component_offsets,
            dtype=np.int64,
        ),
        source_crop_boxes=np.asarray(crop_boxes, dtype=np.int32),
        source_target_boxes=np.asarray(target_boxes, dtype=np.int32),
        official_folds=np.asarray(frame["official_fold"], dtype=np.int16),
        source_patch_indices=np.asarray(frame["source_patch_index"], dtype=np.int32),
        instance_channel_indices=np.asarray(frame["instance_channel_index"], dtype=np.int16),
        instance_ids=np.asarray(frame["instance_id"], dtype=np.int64),
        pre_corruption_labels=np.asarray(frame["pre_corruption_label"], dtype=np.int64),
        group_ids=np.asarray(frame["group_id"].astype(str).tolist(), dtype=np.str_),
        tissue_types=np.asarray(frame["tissue_type"].astype(str).tolist(), dtype=np.str_),
        source_contours_xy=tuple(contours),
        identity_verified=np.asarray(verified, dtype=bool),
        primary_eligible=np.asarray(frame["primary_eligible"], dtype=bool),
        confirmatory_eligible=np.asarray(frame["confirmatory_eligible"], dtype=bool),
        metadata=metadata,
        validation_binding=validation,
    )
    batch.validate()
    if sha256_file(source_manifest) != manifest_sha256:
        raise PanNukeSemanticsError("PanNuke manifest changed during crop extraction")
    if _verify_raw_sources:
        verify_raw_inventory_unchanged(validation)
    return batch


def _contour_arrays(
    contours: tuple[NDArray[np.int32], ...],
) -> tuple[NDArray[np.int32], NDArray[np.int64]]:
    offsets = np.zeros(len(contours) + 1, dtype=np.int64)
    for index, contour in enumerate(contours):
        offsets[index + 1] = offsets[index] + len(contour)
    coordinates = np.concatenate(contours, axis=0).astype(np.int32, copy=False)
    return coordinates, offsets


def save_pannuke_crop_cache(
    crops: PanNukeCropBatch,
    destination: str | Path,
    *,
    _verify_raw_sources: bool = True,
) -> tuple[Path, Path]:
    """Persist component-covering crop arrays with immutable raw provenance."""

    crops.validate()
    validate_analysis_eligibility_provenance(
        crops.metadata,
        crops.sample_ids,
        primary_eligible=crops.primary_eligible,
        confirmatory_eligible=crops.confirmatory_eligible,
    )
    target = ensure_derived_output_outside_raw(
        destination,
        crops.validation_binding.root,
        purpose="PanNuke crop cache destination",
    )
    if target.suffix.lower() != ".npz":
        raise ValueError("crop cache must end in .npz")
    source_manifest = Path(str(crops.metadata["manifest_path"])).resolve()
    expected_manifest_sha256 = str(crops.metadata["manifest_sha256"])

    def verify_fresh_source_bindings() -> None:
        ensure_derived_output_outside_raw(
            target,
            crops.validation_binding.root,
            purpose="PanNuke crop cache destination",
        )
        if (
            not source_manifest.is_file()
            or sha256_file(source_manifest) != expected_manifest_sha256
        ):
            raise PanNukeSemanticsError("PanNuke manifest changed before crop-cache publication")
        if _verify_raw_sources:
            verify_raw_inventory_unchanged(crops.validation_binding)

    verify_fresh_source_bindings()
    contour_xy, contour_offsets = _contour_arrays(crops.source_contours_xy)
    arrays = {
        "sample_ids": crops.sample_ids,
        "context_rgb": crops.context_rgb,
        "target_highlighted_rgb": crops.target_highlighted_rgb,
        "target_masks": crops.target_masks,
        "target_contour_masks": crops.target_contour_masks,
        "raw_component_counts": crops.raw_component_counts,
        "disconnected_instance_flags": crops.disconnected_instance_flags,
        "projected_union_component_counts": crops.projected_union_component_counts,
        "projection_fallback_component_counts": (crops.projection_fallback_component_counts),
        "projection_collision_pixel_counts": crops.projection_collision_pixel_counts,
        "projection_collision_excess_counts": crops.projection_collision_excess_counts,
        "projection_adjacency_pair_counts": crops.projection_adjacency_pair_counts,
        "projection_topology_changed": crops.projection_topology_changed,
        "projected_component_pixel_counts": crops.projected_component_pixel_counts,
        "projected_component_unique_pixel_counts": (crops.projected_component_unique_pixel_counts),
        "baseline_projected_component_counts": crops.baseline_projected_component_counts,
        "projection_fallback_component_flags": crops.projection_fallback_component_flags,
        "projected_component_offsets": crops.projected_component_offsets,
        "source_crop_boxes": crops.source_crop_boxes,
        "source_target_boxes": crops.source_target_boxes,
        "official_folds": crops.official_folds,
        "source_patch_indices": crops.source_patch_indices,
        "instance_channel_indices": crops.instance_channel_indices,
        "instance_ids": crops.instance_ids,
        "pre_corruption_labels": crops.pre_corruption_labels,
        "group_ids": crops.group_ids,
        "tissue_types": crops.tissue_types,
        "source_contour_xy": contour_xy,
        "source_contour_offsets": contour_offsets,
        "identity_verified": crops.identity_verified,
        "primary_eligible": crops.primary_eligible,
        "confirmatory_eligible": crops.confirmatory_eligible,
    }
    weight_identifier = "unlearned:no_learned_weights_pannuke_component_covering_target_crop_v2"
    preprocessing = {
        "identifier": "pannuke_component_covering_dynamic_square_crop_v2",
        "crop_configuration": crops.metadata["crop_configuration"],
        "target_identity_policy": crops.metadata["target_identity_policy"],
        "target_mask_projection": crops.metadata["target_mask_projection"],
        "contour_policy": crops.metadata["contour_policy"],
        "source_annotations_modified": False,
    }
    encoder_metadata = {
        "class_mapping": crops.metadata["class_mapping"],
        "identity_verified_for_all_samples": bool(crops.identity_verified.all()),
        "output_shape": list(crops.context_rgb.shape[1:]),
        "target_identity_policy": crops.metadata["target_identity_policy"],
        "target_mask_projection_semantic_sha256": crops.metadata["target_mask_projection"][
            "semantic_sha256"
        ],
    }
    encoder_implementation = {
        "module": "histo_audit.representations.pannuke",
        "entrypoint": "extract_pannuke_crop_batch",
        "source_file_sha256": sha256_file(Path(__file__)),
    }
    cache_recipe = {
        "identifier": "pannuke_component_covering_target_crop_npz_v3",
        "array_keys": sorted(arrays),
        "contour_encoding": "flat int32 xy plus int64 exclusive offsets",
        "component_encoding": (
            "aligned raw/union vectors plus flat int32 projected-component counts and "
            "int64 exclusive offsets"
        ),
        "sample_alignment": "exact ordered sample_ids axis 0",
        "pickle_allowed": False,
    }
    metadata = build_frozen_cache_metadata(
        base_metadata={
            **crops.metadata,
            "cache_publication_binding_policy": (
                "manifest SHA-256 and full raw inventory reverified immediately before "
                "atomic publication"
            ),
        },
        sample_ids=crops.sample_ids,
        manifest_sha256=str(crops.metadata["manifest_sha256"]),
        raw_inventory_sha256=str(crops.metadata["raw_inventory_sha256"]),
        representation_id="pannuke_component_covering_target_crops",
        input_variant=(
            "context_rgb_plus_component_covering_projected_binary_target_mask_and_"
            "raw_instance_identity"
        ),
        encoder_identifier="pannuke_component_covering_target_crop_v2",
        encoder_metadata=encoder_metadata,
        encoder_implementation=encoder_implementation,
        weight_identifier=weight_identifier,
        weights_sha256=explicit_unlearned_weights_sha256(weight_identifier),
        preprocessing_identifier="pannuke_component_covering_dynamic_square_crop_v2",
        preprocessing=preprocessing,
        cache_recipe=cache_recipe,
        dtype=str(crops.context_rgb.dtype),
        feature_dimension=[int(value) for value in crops.context_rgb.shape[1:]],
        package_versions={
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "pyarrow": _package_version("pyarrow"),
            "scipy": _package_version("scipy"),
        },
        matrix_key="context_rgb",
        provenance_scope="stage_eligible",
    )
    cache, metadata_path, _ = atomic_save_npz_with_sidecar(
        target,
        arrays=arrays,
        metadata=metadata,
        pre_publish_check=verify_fresh_source_bindings,
        post_publish_check=verify_fresh_source_bindings,
    )
    return cache, metadata_path


def save_context_morphometrics_cache(
    context_embeddings: EmbeddingResult,
    engineered: EngineeredFeatureSet,
    sample_ids: NDArray[np.str_],
    destination: str | Path,
    *,
    manifest_sha256: str,
    raw_inventory_sha256: str,
    analysis_eligibility: dict[str, Any] | None = None,
    engineered_cache_binding: dict[str, Any] | None = None,
) -> ContextMorphometricsCache:
    """Persist the explicit §21 context-embedding + target-morphometrics ablation."""

    context_embeddings.validate()
    identifiers = np.asarray(sample_ids, dtype=np.str_)
    if not np.array_equal(context_embeddings.sample_ids, identifiers):
        raise ValueError("context embeddings and morphometrics sample order differs")
    engineered.validate(expected_samples=len(identifiers))
    context_metadata = context_embeddings.metadata
    required_context_fields = (
        "weights_sha256",
        "weight_identifier",
        "encoder_metadata_sha256",
        "encoder_implementation_sha256",
        "preprocessing_sha256",
        "cache_recipe_sha256",
        "sample_order_sha256",
        "manifest_sha256",
        "raw_inventory_sha256",
        "provenance_scope",
    )
    missing = [field for field in required_context_fields if field not in context_metadata]
    if missing:
        raise ValueError(f"context embedding provenance fields are absent: {missing}")
    if context_metadata["provenance_scope"] != "stage_eligible":
        raise ValueError("context+morphometrics requires a stage-eligible context cache")
    if context_metadata["manifest_sha256"] != manifest_sha256:
        raise ValueError("context embedding and concat manifest SHA-256 differs")
    if context_metadata["raw_inventory_sha256"] != raw_inventory_sha256:
        raise ValueError("context embedding and concat raw inventory SHA-256 differs")
    context_variant = context_metadata.get(
        "contract_input_variant", context_metadata.get("input_variant")
    )
    if context_variant != "context_rgb":
        raise ValueError("context+morphometrics requires context_rgb embeddings")
    exact_engineered_binding = (
        dict(engineered_cache_binding) if engineered_cache_binding is not None else None
    )
    context_crop_binding = context_metadata.get("source_crop_cache_binding")
    if exact_engineered_binding is not None:
        expected_binding_keys = {
            "schema_version",
            "binding_type",
            "engineered_cache_file_sha256",
            "engineered_cache_sidecar_file_sha256",
            "engineered_cache_content_sha256",
            "manifest_sha256",
            "raw_inventory_sha256",
            "sample_order_sha256",
            "cache_array_sha256_by_name",
            "source_crop_cache_binding",
            "source_crop_cache_binding_sha256",
        }
        array_hashes = exact_engineered_binding.get("cache_array_sha256_by_name")
        source_crop_binding = exact_engineered_binding.get("source_crop_cache_binding")
        expected_array_hashes = {
            "names": array_artifact_sha256(np.asarray(engineered.names, dtype=np.str_)),
            "sample_ids": array_artifact_sha256(identifiers),
            "values": array_artifact_sha256(engineered.values),
        }
        sha256_fields = (
            "engineered_cache_file_sha256",
            "engineered_cache_sidecar_file_sha256",
            "engineered_cache_content_sha256",
            "manifest_sha256",
            "raw_inventory_sha256",
            "sample_order_sha256",
            "source_crop_cache_binding_sha256",
        )
        common_crop_fields = (
            "crop_cache_file_sha256",
            "crop_cache_sidecar_file_sha256",
            "crop_cache_content_sha256",
            "crop_manifest_sha256",
            "raw_inventory_sha256",
            "sample_order_sha256",
            "target_mask_projection_semantic_sha256",
        )
        if (
            set(exact_engineered_binding) != expected_binding_keys
            or exact_engineered_binding.get("schema_version") != 1
            or exact_engineered_binding.get("binding_type") != "pannuke_engineered_feature_cache_v1"
            or not isinstance(array_hashes, dict)
            or array_hashes != expected_array_hashes
            or exact_engineered_binding.get("engineered_cache_content_sha256")
            != canonical_sha256(expected_array_hashes)
            or exact_engineered_binding.get("manifest_sha256") != manifest_sha256
            or exact_engineered_binding.get("raw_inventory_sha256") != raw_inventory_sha256
            or exact_engineered_binding.get("sample_order_sha256")
            != ordered_sample_ids_sha256(identifiers)
            or not isinstance(source_crop_binding, dict)
            or exact_engineered_binding.get("source_crop_cache_binding_sha256")
            != canonical_sha256(source_crop_binding)
            or any(
                not isinstance(exact_engineered_binding.get(field), str)
                or len(str(exact_engineered_binding[field])) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in str(exact_engineered_binding[field]).casefold()
                )
                for field in sha256_fields
            )
            or (
                isinstance(context_crop_binding, dict)
                and any(
                    context_crop_binding.get(field) != source_crop_binding.get(field)
                    for field in common_crop_fields
                )
            )
        ):
            raise ValueError(
                "engineered cache binding differs from in-memory features or context lineage"
            )
    elif context_crop_binding is not None:
        raise ValueError("stage-bound context embeddings require an exact engineered cache binding")
    engineered_binding_sha256 = (
        canonical_sha256(exact_engineered_binding) if exact_engineered_binding is not None else None
    )
    morphometrics = select_target_morphometrics(engineered)
    context_values = np.asarray(context_embeddings.embeddings, dtype=np.float32)
    morphology_values = np.asarray(morphometrics.values, dtype=np.float32)
    values = np.ascontiguousarray(
        np.concatenate((context_values, morphology_values), axis=1), dtype=np.float32
    )
    names = tuple(
        [f"imagenet.context_embedding_{index:04d}" for index in range(context_values.shape[1])]
        + list(morphometrics.names)
    )
    preprocessing = {
        "context_embedding_preprocessing_identifier": context_metadata["preprocessing_identifier"],
        "context_embedding_preprocessing_sha256": context_metadata["preprocessing_sha256"],
        "target_morphometrics": {
            "column_order": list(morphometrics.names),
            "float64_to_float32_cast": "numpy astype float32 before concatenation",
            "selection": "ordered engineered columns with morphology. prefix",
        },
    }
    encoder_metadata = {
        "component_context_encoder_metadata_sha256": context_metadata["encoder_metadata_sha256"],
        "component_context_cache_content_sha256": context_metadata.get("cache_content_sha256"),
        "component_context_cache_file_sha256": context_metadata.get("cache_file_sha256"),
        "component_context_dimension": int(context_values.shape[1]),
        "component_morphometrics_dimension": int(morphology_values.shape[1]),
        "component_morphometrics_names_sha256": canonical_sha256(list(morphometrics.names)),
        "component_morphometrics_values_sha256": array_artifact_sha256(morphology_values),
        "output_dimension": int(values.shape[1]),
        **(
            {
                "component_engineered_cache_binding": exact_engineered_binding,
                "component_engineered_cache_binding_sha256": engineered_binding_sha256,
            }
            if exact_engineered_binding is not None
            else {}
        ),
    }
    encoder_implementation = {
        "module": "histo_audit.representations.pannuke",
        "entrypoint": "save_context_morphometrics_cache",
        "source_file_sha256": sha256_file(Path(__file__)),
        "context_encoder_implementation_sha256": context_metadata["encoder_implementation_sha256"],
    }
    cache_recipe = {
        "identifier": "imagenet_context_plus_target_morphometrics_npz_v1",
        "array_keys": ["names", "sample_ids", "values"],
        "column_order_sha256": canonical_sha256(list(names)),
        "context_cache_recipe_sha256": context_metadata["cache_recipe_sha256"],
        "context_sample_order_sha256": context_metadata["sample_order_sha256"],
        "morphometric_selection": "morphology. prefix in engineered cache order",
        "output_dtype": "float32",
        "pickle_allowed": False,
        **(
            {"component_engineered_cache_binding_sha256": engineered_binding_sha256}
            if exact_engineered_binding is not None
            else {}
        ),
    }
    metadata = build_frozen_cache_metadata(
        base_metadata={
            "schema_version": 1,
            "crop_manifest_sha256": manifest_sha256,
            "feature_count": len(names),
            "feature_names_sha256": canonical_sha256(list(names)),
            "source_annotations_modified": False,
            "lineage_binding_status": (
                "verified_exact_engineered_cache"
                if exact_engineered_binding is not None
                else "absent_non_stage_fixture"
            ),
            **(
                {
                    "component_engineered_cache_binding": exact_engineered_binding,
                    "component_engineered_cache_binding_sha256": engineered_binding_sha256,
                }
                if exact_engineered_binding is not None
                else {}
            ),
            **(
                {"analysis_eligibility": dict(analysis_eligibility)}
                if analysis_eligibility is not None
                else {}
            ),
        },
        sample_ids=identifiers,
        manifest_sha256=manifest_sha256,
        raw_inventory_sha256=raw_inventory_sha256,
        representation_id="imagenet_context_embeddings_plus_target_morphometrics",
        input_variant="context_rgb_plus_target_morphometrics",
        encoder_identifier="resnet18_imagenet1k_v1_plus_target_morphometrics_v1",
        encoder_metadata=encoder_metadata,
        encoder_implementation=encoder_implementation,
        weight_identifier=str(context_metadata["weight_identifier"]),
        weights_sha256=str(context_metadata["weights_sha256"]),
        preprocessing_identifier=(
            "torchvision_resnet18_imagenet1k_v1_official_plus_target_morphometrics_v1"
        ),
        preprocessing=preprocessing,
        cache_recipe=cache_recipe,
        dtype="float32",
        feature_dimension=len(names),
        package_versions={
            str(key): str(value)
            for key, value in dict(context_metadata.get("package_versions", {})).items()
        },
        matrix_key="values",
        provenance_scope=(
            "stage_eligible" if exact_engineered_binding is not None else "non_stage_fixture"
        ),
    )
    cache, sidecar, complete = atomic_save_npz_with_sidecar(
        destination,
        arrays={
            "values": values,
            "names": np.asarray(names, dtype=np.str_),
            "sample_ids": identifiers,
        },
        metadata=metadata,
    )
    result = ContextMorphometricsCache(
        values=values,
        names=names,
        sample_ids=identifiers,
        metadata=complete,
        cache_path=cache,
        metadata_path=sidecar,
    )
    result.validate()
    return result


def _embedding_crop_cache_binding(
    crops: PanNukeCropBatch,
    crop_cache_path: Path,
    *,
    input_variant: Literal["context_rgb", "target_highlighted_rgb"],
) -> dict[str, Any]:
    """Bind an ImageNet embedding to one exact verified crop artifact."""

    verification = verify_frozen_cache_sidecar(crop_cache_path)
    metadata = verification.metadata
    input_array_key = "context_rgb" if input_variant == "context_rgb" else "target_highlighted_rgb"
    input_array = (
        crops.context_rgb if input_variant == "context_rgb" else crops.target_highlighted_rgb
    )
    input_array_sha256 = array_artifact_sha256(input_array)
    declared_array_hashes = metadata.get("cache_array_sha256_by_name")
    projection = crops.metadata.get("target_mask_projection")
    if (
        not isinstance(declared_array_hashes, dict)
        or declared_array_hashes.get(input_array_key) != input_array_sha256
        or not isinstance(projection, dict)
        or metadata.get("target_mask_projection") != projection
        or metadata.get("manifest_sha256") != crops.metadata.get("manifest_sha256")
        or metadata.get("raw_inventory_sha256") != crops.metadata.get("raw_inventory_sha256")
    ):
        raise RuntimeError("crop cache content/projection differs from the extracted batch")
    return {
        "schema_version": 1,
        "binding_type": "pannuke_component_covering_crop_cache_v1",
        "crop_cache_file_sha256": verification.cache_file_sha256,
        "crop_cache_sidecar_file_sha256": verification.sidecar_file_sha256,
        "crop_cache_content_sha256": str(metadata["cache_content_sha256"]),
        "crop_manifest_sha256": str(crops.metadata["manifest_sha256"]),
        "raw_inventory_sha256": str(crops.metadata["raw_inventory_sha256"]),
        "sample_order_sha256": str(metadata["sample_order_sha256"]),
        "target_mask_projection_semantic_sha256": str(projection["semantic_sha256"]),
        "input_variant": input_variant,
        "input_array_key": input_array_key,
        "input_array_sha256": input_array_sha256,
    }


def _engineered_crop_cache_binding(
    crops: PanNukeCropBatch,
    crop_cache_path: Path,
) -> dict[str, Any]:
    """Bind engineered features to exact RGB and projected-mask crop arrays."""

    verification = verify_frozen_cache_sidecar(crop_cache_path)
    metadata = verification.metadata
    declared_array_hashes = metadata.get("cache_array_sha256_by_name")
    projection = crops.metadata.get("target_mask_projection")
    if not isinstance(declared_array_hashes, dict) or not isinstance(projection, dict):
        raise RuntimeError("crop cache lacks engineered-input content provenance")
    input_hashes = {
        "context_rgb": array_artifact_sha256(crops.context_rgb),
        "target_masks": array_artifact_sha256(crops.target_masks),
    }
    if any(declared_array_hashes.get(key) != value for key, value in input_hashes.items()):
        raise RuntimeError("engineered input arrays differ from the exact crop cache")
    return {
        "schema_version": 1,
        "binding_type": "pannuke_component_covering_crop_cache_engineered_v1",
        "crop_cache_file_sha256": verification.cache_file_sha256,
        "crop_cache_sidecar_file_sha256": verification.sidecar_file_sha256,
        "crop_cache_content_sha256": str(metadata["cache_content_sha256"]),
        "crop_manifest_sha256": str(crops.metadata["manifest_sha256"]),
        "raw_inventory_sha256": str(crops.metadata["raw_inventory_sha256"]),
        "sample_order_sha256": str(metadata["sample_order_sha256"]),
        "target_mask_projection_semantic_sha256": str(projection["semantic_sha256"]),
        "input_variant": "context_rgb_plus_component_covering_target_masks",
        "input_array_sha256_by_name": input_hashes,
    }


def _engineered_cache_binding(engineered_cache_path: Path) -> dict[str, Any]:
    """Return an exact downstream binding for one verified engineered cache."""

    verification = verify_frozen_cache_sidecar(engineered_cache_path)
    metadata = verification.metadata
    source_crop_binding = metadata.get("source_crop_cache_binding")
    array_hashes = metadata.get("cache_array_sha256_by_name")
    if (
        not isinstance(source_crop_binding, dict)
        or not isinstance(array_hashes, dict)
        or set(array_hashes) != {"names", "sample_ids", "values"}
    ):
        raise RuntimeError("engineered cache lacks its exact crop-cache binding")
    return {
        "schema_version": 1,
        "binding_type": "pannuke_engineered_feature_cache_v1",
        "engineered_cache_file_sha256": verification.cache_file_sha256,
        "engineered_cache_sidecar_file_sha256": verification.sidecar_file_sha256,
        "engineered_cache_content_sha256": str(metadata["cache_content_sha256"]),
        "manifest_sha256": str(metadata["manifest_sha256"]),
        "raw_inventory_sha256": str(metadata["raw_inventory_sha256"]),
        "sample_order_sha256": str(metadata["sample_order_sha256"]),
        "cache_array_sha256_by_name": dict(array_hashes),
        "source_crop_cache_binding": source_crop_binding,
        "source_crop_cache_binding_sha256": canonical_sha256(source_crop_binding),
    }


def _build_pannuke_representation_cache_staged(
    validation: PanNukeValidationResult,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    sample_ids: tuple[str, ...] | None = None,
    crop_config: PanNukeCropConfig | None = None,
    resnet_config: ResNet18EmbeddingConfig | None = None,
    include_context_embeddings: bool = False,
) -> PanNukeRepresentationArtifacts:
    """Build one representation bundle inside an unpublished staging directory.

    The pilot requires the target-highlighted cache.  Setting
    ``include_context_embeddings=True`` also emits context RGB embeddings and the
    separate context-embedding + target-morphometrics confirmatory ablation, all
    in the same immutable crop/sample order.
    """

    output = ensure_derived_output_outside_raw(
        output_dir,
        validation.root,
        purpose="PanNuke representation output directory",
    )
    output.mkdir(parents=True, exist_ok=True)
    crop_settings = crop_config or PanNukeCropConfig()
    encoder_settings = resnet_config or ResNet18EmbeddingConfig(
        input_variant="target_highlighted_rgb",
        context_brightness=crop_settings.context_brightness,
    )
    if encoder_settings.input_variant != "target_highlighted_rgb":
        raise ValueError("the declared PanNuke pilot requires target_highlighted_rgb")
    if encoder_settings.context_brightness != crop_settings.context_brightness:
        raise ValueError("crop and encoder context-brightness policies differ")
    crops = extract_pannuke_crop_batch(
        validation,
        manifest_path,
        sample_ids=sample_ids,
        config=crop_settings,
    )
    crop_cache_destination = ensure_derived_output_outside_raw(
        output / "pannuke_crops.npz",
        validation.root,
        purpose="PanNuke crop cache destination",
    )
    crop_cache, crop_metadata = save_pannuke_crop_cache(crops, crop_cache_destination)
    highlighted_crop_binding = _embedding_crop_cache_binding(
        crops,
        crop_cache,
        input_variant="target_highlighted_rgb",
    )
    engineered_crop_binding = _engineered_crop_cache_binding(crops, crop_cache)
    engineered = build_engineered_feature_set(crops.context_rgb, crops.target_masks)
    engineered_destination = ensure_derived_output_outside_raw(
        output / "pannuke_engineered_features.npz",
        validation.root,
        purpose="PanNuke engineered cache destination",
    )
    engineered_cache, engineered_metadata, _ = save_engineered_feature_cache(
        engineered,
        crops.sample_ids,
        engineered_destination,
        manifest_sha256=str(crops.metadata["manifest_sha256"]),
        raw_inventory_sha256=str(crops.metadata["raw_inventory_sha256"]),
        analysis_eligibility=crops.metadata["analysis_eligibility"],
        target_mask_projection=crops.metadata["target_mask_projection"],
        source_crop_cache_binding=engineered_crop_binding,
    )
    engineered_exact_binding = _engineered_cache_binding(engineered_cache)
    highlighted_destination = ensure_derived_output_outside_raw(
        output / "pannuke_resnet18_target_highlighted_embeddings.npz",
        validation.root,
        purpose="PanNuke highlighted embedding cache destination",
    )
    embeddings = extract_resnet18_embeddings(
        crops.context_rgb,
        crops.sample_ids,
        target_masks=crops.target_masks,
        config=encoder_settings,
        cache_path=highlighted_destination,
        manifest_sha256=str(crops.metadata["manifest_sha256"]),
        raw_inventory_sha256=str(crops.metadata["raw_inventory_sha256"]),
        representation_id="imagenet_target_highlighted_embeddings",
        analysis_eligibility=crops.metadata["analysis_eligibility"],
        source_crop_cache_binding=highlighted_crop_binding,
    )
    context_embeddings = None
    context_morphometrics = None
    if include_context_embeddings:
        context_crop_binding = _embedding_crop_cache_binding(
            crops,
            crop_cache,
            input_variant="context_rgb",
        )
        context_destination = ensure_derived_output_outside_raw(
            output / "pannuke_resnet18_context_rgb_embeddings.npz",
            validation.root,
            purpose="PanNuke context embedding cache destination",
        )
        context_embeddings = extract_resnet18_embeddings(
            crops.context_rgb,
            crops.sample_ids,
            config=replace(encoder_settings, input_variant="rgb"),
            cache_path=context_destination,
            manifest_sha256=str(crops.metadata["manifest_sha256"]),
            raw_inventory_sha256=str(crops.metadata["raw_inventory_sha256"]),
            representation_id="imagenet_resnet18_context_embeddings",
            analysis_eligibility=crops.metadata["analysis_eligibility"],
            source_crop_cache_binding=context_crop_binding,
        )
        if not np.array_equal(context_embeddings.sample_ids, embeddings.sample_ids):
            raise RuntimeError("context and target-highlighted embedding sample order differs")
        if context_embeddings.metadata.get("weight_sha256") != embeddings.metadata.get(
            "weight_sha256"
        ):
            raise RuntimeError("context and target-highlighted embeddings use different weights")
        context_morphometrics_destination = ensure_derived_output_outside_raw(
            output / "pannuke_resnet18_context_plus_target_morphometrics.npz",
            validation.root,
            purpose="PanNuke context+morphometrics cache destination",
        )
        context_morphometrics = save_context_morphometrics_cache(
            context_embeddings,
            engineered,
            crops.sample_ids,
            context_morphometrics_destination,
            manifest_sha256=str(crops.metadata["manifest_sha256"]),
            raw_inventory_sha256=str(crops.metadata["raw_inventory_sha256"]),
            analysis_eligibility=crops.metadata["analysis_eligibility"],
            engineered_cache_binding=engineered_exact_binding,
        )
    return PanNukeRepresentationArtifacts(
        crops=crops,
        engineered=engineered,
        embeddings=embeddings,
        crop_cache_path=crop_cache,
        crop_metadata_path=crop_metadata,
        engineered_cache_path=engineered_cache,
        engineered_metadata_path=engineered_metadata,
        context_embeddings=context_embeddings,
        context_morphometrics=context_morphometrics,
    )


def build_pannuke_representation_cache(
    validation: PanNukeValidationResult,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    sample_ids: tuple[str, ...] | None = None,
    crop_config: PanNukeCropConfig | None = None,
    resnet_config: ResNet18EmbeddingConfig | None = None,
    include_context_embeddings: bool = False,
    chunk_size: int | None = None,
) -> PanNukeRepresentationArtifacts:
    """Atomically publish one source-fresh PanNuke representation bundle."""

    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    output = ensure_derived_output_outside_raw(
        output_dir,
        validation.root,
        purpose="PanNuke representation output directory",
    )
    try:
        output = assert_mutable_publication_destination(
            output_dir,
            role="PanNuke representation output directory",
        )
    except (NotADirectoryError, PermissionError, RuntimeError) as error:
        raise PanNukeSemanticsError(str(error)) from error
    disk_check = require_full_manifest_cache_disk_space(
        manifest_path,
        output,
        sample_ids=sample_ids,
    )
    manifest_row_count = disk_check.manifest_row_count
    if chunk_size is not None or manifest_row_count > _CHUNKED_AUTO_THRESHOLD:
        from .pannuke_chunked import build_pannuke_representation_cache_chunked

        return build_pannuke_representation_cache_chunked(
            validation,
            manifest_path,
            output_dir,
            sample_ids=sample_ids,
            crop_config=crop_config,
            resnet_config=resnet_config,
            include_context_embeddings=include_context_embeddings,
            chunk_size=chunk_size or _DEFAULT_CHUNK_SIZE,
        )

    if os.path.lexists(output):
        raise FileExistsError(f"representation output directory already exists: {output}")
    try:
        assert_mutable_publication_destination(
            output.parent / f".{output.name}.staging",
            role="PanNuke representation staging workspace",
        )
    except (NotADirectoryError, PermissionError, RuntimeError) as error:
        raise PanNukeSemanticsError(str(error)) from error
    output.parent.mkdir(parents=True, exist_ok=True)
    ensure_derived_output_outside_raw(
        output,
        validation.root,
        purpose="PanNuke representation output directory",
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    publications: list[PublishedPath] = []
    try:
        artifacts = _build_pannuke_representation_cache_staged(
            validation,
            manifest_path,
            staging,
            sample_ids=sample_ids,
            crop_config=crop_config,
            resnet_config=resnet_config,
            include_context_embeddings=include_context_embeddings,
        )
        source_manifest = Path(str(artifacts.crops.metadata["manifest_path"])).resolve()
        expected_manifest_sha256 = str(artifacts.crops.metadata["manifest_sha256"])

        def verify_fresh_sources_and_destination() -> None:
            ensure_derived_output_outside_raw(
                output,
                validation.root,
                purpose="PanNuke representation output directory",
            )
            if (
                not source_manifest.is_file()
                or sha256_file(source_manifest) != expected_manifest_sha256
            ):
                raise PanNukeSemanticsError(
                    "PanNuke manifest changed during representation extraction"
                )
            verify_raw_inventory_unchanged(validation)

        verify_fresh_sources_and_destination()
        if os.path.lexists(output):
            raise FileExistsError(f"representation output directory already exists: {output}")
        publications = publish_flat_directory_no_overwrite(staging, output)
        verify_fresh_sources_and_destination()

        def rebase(path: Path) -> Path:
            return output / path.relative_to(staging)

        embeddings = replace(
            artifacts.embeddings,
            cache_path=(
                rebase(artifacts.embeddings.cache_path)
                if artifacts.embeddings.cache_path is not None
                else None
            ),
            metadata_path=(
                rebase(artifacts.embeddings.metadata_path)
                if artifacts.embeddings.metadata_path is not None
                else None
            ),
        )
        context_embeddings = (
            replace(
                artifacts.context_embeddings,
                cache_path=(
                    rebase(artifacts.context_embeddings.cache_path)
                    if artifacts.context_embeddings.cache_path is not None
                    else None
                ),
                metadata_path=(
                    rebase(artifacts.context_embeddings.metadata_path)
                    if artifacts.context_embeddings.metadata_path is not None
                    else None
                ),
            )
            if artifacts.context_embeddings is not None
            else None
        )
        context_morphometrics = (
            replace(
                artifacts.context_morphometrics,
                cache_path=rebase(artifacts.context_morphometrics.cache_path),
                metadata_path=rebase(artifacts.context_morphometrics.metadata_path),
            )
            if artifacts.context_morphometrics is not None
            else None
        )
        return replace(
            artifacts,
            embeddings=embeddings,
            crop_cache_path=rebase(artifacts.crop_cache_path),
            crop_metadata_path=rebase(artifacts.crop_metadata_path),
            engineered_cache_path=rebase(artifacts.engineered_cache_path),
            engineered_metadata_path=rebase(artifacts.engineered_metadata_path),
            context_embeddings=context_embeddings,
            context_morphometrics=context_morphometrics,
            publication_records=tuple(publications),
        )
    except BaseException as error:
        if publications:
            try:
                rollback_owned_publications(publications)
            except RuntimeError as rollback_error:
                raise RuntimeError(
                    "representation publication failed and ownership-safe rollback was "
                    f"incomplete: {rollback_error}"
                ) from error
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "FULL_MANIFEST_CACHE_MIN_FREE_BYTES",
    "ContextMorphometricsCache",
    "FullManifestCacheDiskCheck",
    "InsufficientFullManifestCacheDiskSpaceError",
    "PanNukeCropBatch",
    "PanNukeCropConfig",
    "PanNukeRepresentationArtifacts",
    "build_pannuke_representation_cache",
    "extract_pannuke_crop_batch",
    "require_full_manifest_cache_disk_space",
    "save_context_morphometrics_cache",
    "save_pannuke_crop_cache",
]
