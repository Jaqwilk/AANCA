"""Controlled label-corruption mechanisms."""

from .controlled import (
    CorruptionResult,
    FeatureIndependenceEvidence,
    FeatureSpaceEvidence,
    apply_controlled_corruption,
    apply_corruption_to_records,
    array_artifact_sha256,
    canonical_sha256,
    exact_corruption_count,
    normalise_rate,
    semantic_sha256,
)

__all__ = [
    "CorruptionResult",
    "FeatureIndependenceEvidence",
    "FeatureSpaceEvidence",
    "apply_controlled_corruption",
    "apply_corruption_to_records",
    "array_artifact_sha256",
    "canonical_sha256",
    "exact_corruption_count",
    "normalise_rate",
    "semantic_sha256",
]
