"""Typed, JSON-friendly records for PanNuke acquisition and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

SOURCE_PATCH_INDEPENDENCE_STATEMENT = (
    "Separation was performed at source-patch level. Patient- and WSI-level "
    "independence could not be guaranteed from the released metadata."
)

OFFICIAL_RELEASE_FOLD_IDS = (1, 2, 3)


@dataclass(frozen=True, slots=True)
class VerifiedClassMapping:
    """Positive mask-channel order backed by an identified source."""

    class_names: tuple[str, ...]
    source: str
    source_revision: str
    verified: bool = True
    source_note: str = ""

    def __post_init__(self) -> None:
        if not self.class_names or any(not name.strip() for name in self.class_names):
            raise ValueError("class_names must contain non-empty positive-class names")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("positive class names must be unique")
        if not self.source.strip():
            raise ValueError("a precise mapping source is required")
        if not self.source_revision.strip():
            raise ValueError("a pinned mapping-source revision is required")
        if not self.verified:
            raise ValueError("unverified class mappings cannot pass the PanNuke gate")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# This order is documented by the public official PanNuke evaluation/metrics code.
# It is a semantic assertion with provenance, not an inference from array contents.
OFFICIAL_METRICS_CLASS_MAPPING = VerifiedClassMapping(
    class_names=(
        "neoplastic",
        "inflammatory",
        "connective_soft_tissue",
        "dead",
        "non_neoplastic_epithelial",
    ),
    source=(
        "https://github.com/TissueImageAnalytics/PanNuke-metrics/tree/"
        "c00014d766ca1be142b81bea19d9ef4315cde65a"
    ),
    source_revision="c00014d766ca1be142b81bea19d9ef4315cde65a",
    source_note=(
        "Default positive-channel order documented by the official PanNuke metrics "
        "repository README at pinned archived commit c00014d766ca1be142b81bea19d9ef4315cde65a; "
        "re-verify this evidence before a frozen study."
    ),
)


@dataclass(frozen=True, slots=True)
class RawFileRecord:
    """One immutable raw-file inventory entry."""

    relative_path: str
    size_bytes: int
    sha256: str
    fold_id: int | None
    file_kind: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ArrayInspection:
    """Bounded content/header inspection for one memory-mapped NumPy array."""

    relative_path: str
    shape: tuple[int, ...]
    dtype: str
    role_scores: dict[str, float]
    sample_min: float | str | None
    sample_max: float | str | None
    finite: bool | None
    integer_like: bool | None
    zero_fraction: float | None
    sampled_patch_indices: tuple[int, ...]
    channel_axis_candidates: tuple[int, ...]
    load_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DiscoveredFold:
    """Conservatively resolved image, mask, and tissue arrays for one fold."""

    fold_id: int
    image_path: Path
    mask_path: Path
    tissue_path: Path
    image_channel_axis: int
    mask_channel_axis: int

    def as_dict(self, *, root: Path | None = None) -> dict[str, Any]:
        def render(path: Path) -> str:
            if root is None:
                return str(path)
            try:
                return path.relative_to(root).as_posix()
            except ValueError:
                return str(path)

        return {
            "fold_id": self.fold_id,
            "image_path": render(self.image_path),
            "mask_path": render(self.mask_path),
            "tissue_path": render(self.tissue_path),
            "image_channel_axis": self.image_channel_axis,
            "mask_channel_axis": self.mask_channel_axis,
        }


@dataclass(frozen=True, slots=True)
class ReleaseDiscovery:
    """Raw discovery evidence before full semantic validation."""

    root: Path
    fold_ids: tuple[int, ...]
    npy_files: tuple[Path, ...]
    archives: tuple[Path, ...]
    inspections: tuple[ArrayInspection, ...]
    folds: tuple[DiscoveredFold, ...]


@dataclass(frozen=True, slots=True)
class MaskInstanceQC:
    """One raw nucleus instance touched by a pixel-level mask anomaly.

    The record preserves the source class/channel/instance identity.  In
    particular, ``overlapping_instance_ids`` is evidence about the other raw
    channels at cross-class-overlap pixels; it is never an adjudicated label.
    """

    fold_id: int
    patch_index: int
    class_index: int
    class_name: str
    channel_index: int
    instance_id: int
    total_pixel_count: int
    overlap_pixel_count: int
    positive_background_pixel_count: int
    overlapping_class_indices: tuple[int, ...]
    overlapping_instance_ids: tuple[int, ...]
    overlapping_instance_ids_by_class: dict[str, tuple[int, ...]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PatchMaskQC:
    """Complete anomaly-safe QC facts for one source mask patch."""

    fold_id: int
    patch_index: int
    height: int
    width: int
    total_pixel_count: int
    positive_any_pixel_count: int
    background_pixel_count: int
    void_pixel_count: int
    cross_class_overlap_pixel_count: int
    positive_and_background_pixel_count: int
    anomaly_union_pixel_count: int
    affected_instance_count: int
    affected_class_indices: tuple[int, ...]
    affected_class_names: tuple[str, ...]
    affected_instances: tuple[MaskInstanceQC, ...]
    has_void: bool
    has_cross_class_overlap: bool
    has_positive_and_background: bool
    mask_sha256_by_kind: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FoldMaskQC:
    """Full-patch QC aggregation for one official fold."""

    fold_id: int
    patch_count: int
    total_pixel_count: int
    positive_any_pixel_count: int
    background_pixel_count: int
    void_pixel_count: int
    cross_class_overlap_pixel_count: int
    positive_and_background_pixel_count: int
    anomaly_union_pixel_count: int
    void_patch_count: int
    cross_class_overlap_patch_count: int
    positive_and_background_patch_count: int
    anomaly_union_patch_count: int
    normal_patch_count: int
    affected_instance_count: int
    overlap_touching_instance_count: int
    positive_background_touching_instance_count: int
    affected_class_indices: tuple[int, ...]
    affected_class_names: tuple[str, ...]
    void_patch_indices: tuple[int, ...]
    cross_class_overlap_patch_indices: tuple[int, ...]
    positive_and_background_patch_indices: tuple[int, ...]
    anomaly_union_patch_indices: tuple[int, ...]
    patches: tuple[PatchMaskQC, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GlobalMaskQC:
    """Release-wide sums reconciled from all fold-level QC records."""

    fold_ids: tuple[int, ...]
    fold_count: int
    patch_count: int
    total_pixel_count: int
    positive_any_pixel_count: int
    background_pixel_count: int
    void_pixel_count: int
    cross_class_overlap_pixel_count: int
    positive_and_background_pixel_count: int
    anomaly_union_pixel_count: int
    void_patch_count: int
    cross_class_overlap_patch_count: int
    positive_and_background_patch_count: int
    anomaly_union_patch_count: int
    normal_patch_count: int
    affected_instance_count: int
    overlap_touching_instance_count: int
    positive_background_touching_instance_count: int
    affected_class_indices: tuple[int, ...]
    affected_class_names: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MaskQCPolicy:
    """Machine-readable, outcome-independent interpretation of PanNuke masks."""

    policy_version: str
    positive_channel_indices: tuple[int, ...]
    background_channel_index_by_fold: dict[str, int | None]
    supplied_background_is_exact_complement_required: bool
    void_definition: str
    cross_class_overlap_definition: str
    positive_and_background_definition: str
    cross_class_overlap_action: str
    analysis_instance_exclusion_reason: str
    applies_identically_to_primary_and_confirmatory: bool
    no_class_arbitration: bool
    source_masks_modified: bool
    release_annotation_anomalies_are_fatal: bool
    structural_invalidity_is_fatal: bool
    disconnected_instance_definition: str = (
        "one raw channel/instance ID occupies more than one 4-connected component"
    )
    disconnected_instance_action: str = (
        "retain raw identity and flag quality; freeze analysis eligibility after the pilot "
        "without final-reference outcomes"
    )
    disconnected_instance_ids_are_fatal: bool = False

    @property
    def confirmatory_instance_exclusion_reason(self) -> str:
        """Compatibility alias; the rule is shared by both analysis families."""

        return self.analysis_instance_exclusion_reason

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnomalyOverlaySelection:
    """Deterministic source-patch selection for downstream QC overlays."""

    strategy: str
    requested_max_patches: int
    selected_patch_keys: tuple[str, ...]
    selected_by_category: dict[str, tuple[str, ...]]
    category_candidate_counts: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FoldValidation:
    """Validated array facts for a released fold."""

    fold_id: int
    n_patches: int
    height: int
    width: int
    image_shape: tuple[int, ...]
    image_dtype: str
    image_range: tuple[float, float]
    mask_shape: tuple[int, ...]
    mask_dtype: str
    mask_range: tuple[float, float]
    tissue_shape: tuple[int, ...]
    tissue_dtype: str
    tissue_values: tuple[str, ...]
    positive_channel_indices: tuple[int, ...]
    background_channel_index: int | None
    background_channel_candidates: tuple[int, ...]
    validation_scope: Literal["full_semantic_scan"]
    full_scan_patch_count: int
    full_scan_instance_count: int
    sampled_patch_indices: tuple[int, ...]
    sampled_instance_ids_by_class: dict[str, tuple[int, ...]]
    overlap_pixel_count_sampled: int
    malformed_instance_count_sampled: int
    mask_qc: FoldMaskQC
    disconnected_instance_count_full_scan: int = 0
    disconnected_patch_count_full_scan: int = 0
    disconnected_instance_count_sampled: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PanNukeValidationResult:
    """Successful result of the non-mutating PanNuke semantic gate."""

    root: Path
    mapping: VerifiedClassMapping
    folds: tuple[DiscoveredFold, ...]
    fold_validation: tuple[FoldValidation, ...]
    inventory: tuple[RawFileRecord, ...]
    archive_paths: tuple[str, ...]
    global_mask_qc: GlobalMaskQC
    qc_policy: MaskQCPolicy
    anomaly_overlay_selection: AnomalyOverlaySelection
    expected_fold_ids: tuple[int, ...]
    release_complete: bool
    grouping_unit: Literal["source_patch"] = "source_patch"
    patient_id_available: bool = False
    wsi_id_available: bool = False
    independence_statement: str = SOURCE_PATCH_INDEPENDENCE_STATEMENT

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "valid",
            "validation_scope": "full_semantic_scan",
            "root": str(self.root),
            "class_mapping": self.mapping.as_dict(),
            "folds": [fold.as_dict(root=self.root) for fold in self.folds],
            "fold_validation": [value.as_dict() for value in self.fold_validation],
            "expected_fold_ids": list(self.expected_fold_ids),
            "release_complete": self.release_complete,
            "raw_file_inventory": [item.as_dict() for item in self.inventory],
            "archives": list(self.archive_paths),
            "global_mask_qc": self.global_mask_qc.as_dict(),
            "qc_policy": self.qc_policy.as_dict(),
            "anomaly_overlay_selection": self.anomaly_overlay_selection.as_dict(),
            "grouping_unit": self.grouping_unit,
            "patient_id_available": self.patient_id_available,
            "wsi_id_available": self.wsi_id_available,
            "independence_statement": self.independence_statement,
            "automatic_source_annotation_modification": False,
        }


@dataclass(frozen=True, slots=True)
class ValidationArtifacts:
    """Paths emitted by :func:`validate_pannuke`."""

    result: PanNukeValidationResult
    json_path: Path
    markdown_path: Path
    overlay_path: Path
    raw_inventory_csv_path: Path


@dataclass(frozen=True, slots=True)
class ManifestArtifacts:
    """Nucleus-manifest outputs; images are referenced, never cropped here."""

    parquet_path: Path
    summary_csv_path: Path
    row_count: int
    patch_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DuplicateAuditArtifacts:
    """Read-only duplicate candidate report outputs."""

    json_path: Path
    csv_path: Path
    markdown_path: Path
    visual_grid_path: Path
    hash_provenance_csv_path: Path
    embedding_cache_path: Path | None
    exact_pair_count: int
    perceptual_pair_count: int
    embedding_pair_count: int
    sampled_patch_count: int
    embedding_sampled_patch_count: int
    embedding_status: Literal["passed", "blocked", "failed", "not_requested"]
    required_two_signal_gate_complete: bool
