"""Dataset, manifest, target-representation, and duplicate helpers."""

from .duplicates import (
    DuplicateCandidate,
    canonical_array_sha256,
    find_embedding_duplicate_candidates,
    find_exact_duplicate_pairs,
    find_perceptual_duplicate_candidates,
    perceptual_hash,
)
from .manifest import NucleusRecord, validate_manifest
from .splitting import (
    OuterAuditSplit,
    make_fractional_outer_audit_split,
    make_outer_audit_split,
)
from .synthetic import CLASS_NAMES, SyntheticDataset, generate_synthetic_dataset
from .targets import extract_target_colour_features, extract_target_crop, highlight_target

__all__ = [
    "CLASS_NAMES",
    "DuplicateCandidate",
    "NucleusRecord",
    "OuterAuditSplit",
    "SyntheticDataset",
    "canonical_array_sha256",
    "extract_target_colour_features",
    "extract_target_crop",
    "find_embedding_duplicate_candidates",
    "find_exact_duplicate_pairs",
    "find_perceptual_duplicate_candidates",
    "generate_synthetic_dataset",
    "highlight_target",
    "make_fractional_outer_audit_split",
    "make_outer_audit_split",
    "perceptual_hash",
    "validate_manifest",
]
