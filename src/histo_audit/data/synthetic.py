"""Deterministic five-class patch/nucleus-like software-validation data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .manifest import NucleusRecord, validate_manifest
from .targets import (
    extract_morphology_features,
    extract_target_colour_features,
    mask_bbox,
    mask_perimeter,
)

CLASS_NAMES: tuple[str, ...] = (
    "neoplastic",
    "inflammatory",
    "connective_soft_tissue",
    "dead",
    "non_neoplastic_epithelial",
)

SYNTHETIC_GENERATOR_SCHEMA_VERSION = 2


def synthetic_generator_code_sha256() -> str:
    """Hash every local implementation file that defines generated arrays/features."""

    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("manifest.py", "synthetic.py", "targets.py"):
        path = package_dir / name
        payload = path.read_bytes()
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "little"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """In-memory synthetic validation dataset with explicit target identities."""

    images: NDArray[np.uint8]
    target_masks: NDArray[np.bool_]
    records: tuple[NucleusRecord, ...]
    audit_features: NDArray[np.float64]
    corruption_features: NDArray[np.float64]
    class_names: tuple[str, ...] = CLASS_NAMES

    @property
    def pre_corruption_labels(self) -> NDArray[np.int64]:
        return np.asarray([row.pre_corruption_label for row in self.records], dtype=np.int64)

    @property
    def observed_labels(self) -> NDArray[np.int64]:
        return np.asarray([row.observed_label for row in self.records], dtype=np.int64)

    @property
    def group_ids(self) -> NDArray[np.str_]:
        return np.asarray([row.group_id for row in self.records], dtype=np.str_)

    @property
    def sample_ids(self) -> NDArray[np.str_]:
        return np.asarray([row.sample_id for row in self.records], dtype=np.str_)

    @property
    def official_folds(self) -> NDArray[np.int64]:
        return np.asarray([row.official_fold for row in self.records], dtype=np.int64)

    def validate(self) -> None:
        """Validate alignment, mask identity, geometry, and manifest invariants."""

        n = len(self.records)
        if self.images.shape[0] != n or self.target_masks.shape[0] != n:
            raise ValueError("image, mask, and manifest lengths differ")
        if self.images.ndim != 4 or self.images.shape[-1] != 3:
            raise ValueError("images must have shape (n, height, width, 3)")
        if self.target_masks.shape != self.images.shape[:3]:
            raise ValueError("target masks must align with images")
        if not np.all(self.target_masks.reshape(n, -1).any(axis=1)):
            raise ValueError("every target mask must be non-empty")
        if self.audit_features.shape[0] != n or self.corruption_features.shape[0] != n:
            raise ValueError("feature arrays must align with manifest")
        validate_manifest(
            self.records, n_classes=len(self.class_names), image_shape=self.images.shape[1:3]
        )
        for row, mask in zip(self.records, self.target_masks, strict=True):
            if row.bbox != mask_bbox(mask):
                raise ValueError(f"target bbox does not match mask for {row.sample_id}")
            if row.area != int(mask.sum()):
                raise ValueError(f"target area does not match mask for {row.sample_id}")


def _ellipse_mask(
    patch_size: int, center_x: float, center_y: float, radius_x: float, radius_y: float
) -> NDArray[np.bool_]:
    yy, xx = np.mgrid[:patch_size, :patch_size]
    return (((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2) <= 1.0


def _layout(n_instances: int, patch_size: int) -> list[tuple[float, float]]:
    columns = int(np.ceil(np.sqrt(n_instances)))
    rows = int(np.ceil(n_instances / columns))
    x_positions = np.linspace(patch_size * 0.18, patch_size * 0.82, columns)
    y_positions = np.linspace(patch_size * 0.2, patch_size * 0.8, rows)
    return [(float(x), float(y)) for y in y_positions for x in x_positions][:n_instances]


def generate_synthetic_dataset(
    *,
    n_groups: int = 30,
    instances_per_group: int = 7,
    patch_size: int = 64,
    seed: int = 2027,
) -> SyntheticDataset:
    """Generate grouped, imbalanced, overlapping five-class synthetic nuclei.

    Each source patch is a group and contributes multiple target instances.
    Labels follow a deliberately imbalanced distribution while a deterministic
    coverage guard places every class in each synthetic outer fold.
    """

    if n_groups < 3:
        raise ValueError("n_groups must be at least three")
    if instances_per_group < len(CLASS_NAMES):
        raise ValueError("instances_per_group must be at least five")
    if patch_size < 32:
        raise ValueError("patch_size must be at least 32 pixels")
    rng = np.random.default_rng(seed)
    base_colours = np.asarray(
        [
            [121.0, 67.0, 131.0],
            [91.0, 73.0, 143.0],
            [151.0, 87.0, 126.0],
            [102.0, 62.0, 103.0],
            [172.0, 105.0, 146.0],
        ]
    )
    class_probabilities = np.asarray([0.39, 0.25, 0.17, 0.11, 0.08])
    geometry = np.asarray(
        [
            [4.9, 4.3],
            [3.8, 3.5],
            [5.6, 3.8],
            [4.2, 3.1],
            [5.4, 4.8],
        ]
    )
    config_hash = hashlib.sha256(
        json.dumps(
            {
                "n_groups": n_groups,
                "instances_per_group": instances_per_group,
                "patch_size": patch_size,
                "seed": seed,
                "generator_schema_version": SYNTHETIC_GENERATOR_SCHEMA_VERSION,
                "generator_code_sha256": synthetic_generator_code_sha256(),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    images: list[NDArray[np.uint8]] = []
    target_masks: list[NDArray[np.bool_]] = []
    records: list[NucleusRecord] = []
    positions = _layout(instances_per_group, patch_size)
    label_grid = rng.choice(
        len(CLASS_NAMES),
        size=(n_groups, instances_per_group),
        p=class_probabilities,
    )
    for outer_fold in range(3):
        fold_groups = np.flatnonzero(np.arange(n_groups) % 3 == outer_fold)
        if len(fold_groups):
            coverage_slots = [
                (int(group), instance)
                for group in fold_groups
                for instance in range(instances_per_group)
            ]
            for class_index, (group, instance) in enumerate(coverage_slots[: len(CLASS_NAMES)]):
                label_grid[group, instance] = class_index

    for group_index in range(n_groups):
        group_rng = np.random.default_rng(rng.integers(0, np.iinfo(np.int64).max))
        background_mean = np.asarray([218.0, 166.0, 190.0]) + group_rng.normal(0.0, 7.0, 3)
        patch = group_rng.normal(background_mean, 8.0, (patch_size, patch_size, 3))
        labels = label_grid[group_index].tolist()
        masks: list[NDArray[np.bool_]] = []
        for _instance_index, (label, position) in enumerate(zip(labels, positions, strict=True)):
            cx = position[0] + float(group_rng.normal(0.0, 0.9))
            cy = position[1] + float(group_rng.normal(0.0, 0.9))
            rx, ry = geometry[label] + group_rng.normal(0.0, 0.65, 2)
            rx = float(max(rx, 2.5))
            ry = float(max(ry, 2.5))
            mask = _ellipse_mask(patch_size, cx, cy, rx, ry)
            masks.append(mask)
            colour = base_colours[label] + group_rng.normal(0.0, 12.0, 3)
            texture = group_rng.normal(0.0, 10.0, (int(mask.sum()), 3))
            patch[mask] = 0.25 * patch[mask] + 0.75 * (colour + texture)
            # Add a class-independent pale centre to keep appearances overlapping.
            centre = _ellipse_mask(patch_size, cx, cy, max(rx * 0.28, 1.0), max(ry * 0.28, 1.0))
            patch[centre] = 0.6 * patch[centre] + 0.4 * background_mean

        patch_uint8 = np.clip(np.rint(patch), 0, 255).astype(np.uint8)
        patch_id = f"synthetic_patch_{group_index:04d}"
        group_id = patch_id
        official_fold = group_index % 3
        for instance_index, (label, mask) in enumerate(zip(labels, masks, strict=True), start=1):
            sample_id = f"{patch_id}_instance_{instance_index:02d}"
            x0, y0, x1, y1 = mask_bbox(mask)
            ys, xs = np.nonzero(mask)
            border_touch = bool(x0 == 0 or y0 == 0 or x1 == patch_size or y1 == patch_size)
            record = NucleusRecord(
                sample_id=sample_id,
                official_fold=official_fold,
                patch_id=patch_id,
                group_id=group_id,
                tissue_type=f"synthetic_tissue_{group_index % 4}",
                source_image_path=f"synthetic://images/{patch_id}",
                source_mask_path=f"synthetic://masks/{patch_id}",
                patch_index=group_index,
                nucleus_class_index=int(label),
                nucleus_class_name=CLASS_NAMES[int(label)],
                instance_id=instance_index,
                bbox=(x0, y0, x1, y1),
                centroid=(float(xs.mean()), float(ys.mean())),
                area=int(mask.sum()),
                perimeter=mask_perimeter(mask),
                border_touch=border_touch,
                crop_padding=6,
                quality_flags=(),
                pre_corruption_label=int(label),
                observed_label=int(label),
                original_class=int(label),
                dataset_seed=seed,
                configuration_hash=config_hash,
            )
            images.append(patch_uint8.copy())
            target_masks.append(mask.copy())
            records.append(record)

    image_array = np.stack(images).astype(np.uint8, copy=False)
    mask_array = np.stack(target_masks).astype(bool, copy=False)
    dataset = SyntheticDataset(
        images=image_array,
        target_masks=mask_array,
        records=tuple(records),
        audit_features=extract_target_colour_features(image_array, mask_array),
        corruption_features=extract_morphology_features(mask_array),
    )
    dataset.validate()
    return dataset
