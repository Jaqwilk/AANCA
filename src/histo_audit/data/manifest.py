"""Typed nucleus-manifest records and invariant validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NucleusRecord:
    """One immutable nucleus-level manifest row.

    Bounding boxes use half-open ``(x_min, y_min, x_max, y_max)`` coordinates.
    ``pre_corruption_label`` is a controlled reference annotation, not a claim of
    biological truth.
    """

    sample_id: str
    official_fold: int
    patch_id: str
    group_id: str
    tissue_type: str
    source_image_path: str
    source_mask_path: str
    patch_index: int
    nucleus_class_index: int
    nucleus_class_name: str
    instance_id: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: int
    perimeter: float
    border_touch: bool
    crop_padding: int
    quality_flags: tuple[str, ...]
    pre_corruption_label: int
    observed_label: int
    is_injected_corruption: bool = False
    corruption_type: str = "none"
    original_class: int | None = None
    replacement_class: int | None = None
    corruption_seed: int | None = None
    corruption_rate: float = 0.0
    corruption_representation: str | None = None
    auditor_representation: str | None = None
    feature_space_independent: bool | None = None
    circularity_risk: bool = False
    dataset_seed: int = 0
    configuration_hash: str = ""
    corruption_timestamp_utc: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly row."""

        return asdict(self)


def _field(record: NucleusRecord | Mapping[str, Any], name: str) -> Any:
    return getattr(record, name) if isinstance(record, NucleusRecord) else record[name]


def validate_manifest(
    records: Sequence[NucleusRecord | Mapping[str, Any]],
    *,
    n_classes: int = 5,
    image_shape: tuple[int, int] | None = None,
) -> None:
    """Raise ``ValueError`` if required identity, geometry, or label rules fail."""

    if not records:
        raise ValueError("manifest must contain at least one record")
    sample_ids: set[str] = set()
    patch_instances: set[tuple[str, int]] = set()
    for record in records:
        sample_id = str(_field(record, "sample_id"))
        patch_id = str(_field(record, "patch_id"))
        instance_id = int(_field(record, "instance_id"))
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"duplicate or empty sample_id: {sample_id!r}")
        sample_ids.add(sample_id)
        key = (patch_id, instance_id)
        if key in patch_instances:
            raise ValueError(f"duplicate patch-instance key: {key!r}")
        patch_instances.add(key)

        area = int(_field(record, "area"))
        if area <= 0:
            raise ValueError(f"non-positive target area for {sample_id}")
        x0, y0, x1, y1 = (int(v) for v in _field(record, "bbox"))
        if not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise ValueError(f"invalid bounding box for {sample_id}: {(x0, y0, x1, y1)}")
        if image_shape is not None:
            height, width = image_shape
            if x1 > width or y1 > height:
                raise ValueError(f"bounding box exceeds image for {sample_id}")

        pre = int(_field(record, "pre_corruption_label"))
        observed = int(_field(record, "observed_label"))
        if pre not in range(n_classes) or observed not in range(n_classes):
            raise ValueError(f"invalid class label for {sample_id}")
        injected = bool(_field(record, "is_injected_corruption"))
        replacement = _field(record, "replacement_class")
        if injected:
            if pre == observed:
                raise ValueError(f"flagged corruption does not change label: {sample_id}")
            if replacement is None or int(replacement) != observed:
                raise ValueError(f"replacement metadata mismatch for {sample_id}")
        elif pre != observed:
            raise ValueError(f"unflagged row changes observed label: {sample_id}")


def assert_group_partitions_disjoint(partitions: Iterable[Iterable[str]]) -> None:
    """Ensure no source group occurs in more than one partition."""

    seen: set[str] = set()
    for partition_number, partition in enumerate(partitions):
        groups = {str(group) for group in partition}
        overlap = seen.intersection(groups)
        if overlap:
            raise ValueError(f"group leakage in partition {partition_number}: {sorted(overlap)!r}")
        seen.update(groups)
