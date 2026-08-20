"""Composable annotation-risk strategies built from group-safe evidence.

The probability vector supplied here must already be out of fold.  Neighbour-based
scores additionally receive the exact fold assignment and training-group provenance,
so a scored nucleus and its complete source group remain unavailable to its reference
neighbourhood.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .neighbours import NeighbourDisagreementResult, fold_safe_neighbour_disagreement
from .scores import fixed_hybrid_score, score_annotations

_PROBABILITY_METHODS = {
    "self_confidence": "self_confidence",
    "self-confidence": "self_confidence",
    "one_minus_probability_of_observed_label": "self_confidence",
    "negative_log_likelihood": "negative_log_likelihood",
    "nll": "negative_log_likelihood",
    "prediction_margin": "prediction_margin",
    "margin": "prediction_margin",
    "predictive_entropy": "predictive_entropy",
    "entropy": "predictive_entropy",
}
_NEIGHBOUR_METHODS = {
    "nearest_neighbour_disagreement",
    "fold_safe_neighbour_disagreement",
    "neighbour_disagreement",
}
_HYBRID_METHODS = {
    "fixed_hybrid",
    "fixed_hybrid_self_confidence_neighbour",
}


@dataclass(frozen=True, slots=True)
class GroupSafeAuditScoreResult:
    """One risk vector and the auditable components used to construct it."""

    method: str
    risk_scores: NDArray[np.float64]
    component_scores: Mapping[str, NDArray[np.float64]]
    neighbour_evidence: NeighbourDisagreementResult | None
    neighbour_k: int | None
    neighbour_metric: str | None
    hybrid_weights: tuple[float, float] | None

    def as_dict(self) -> dict[str, object]:
        """Return compact JSON-compatible strategy metadata, never sample outcomes."""

        return {
            "method": self.method,
            "components": list(self.component_scores),
            "neighbour_k": self.neighbour_k,
            "neighbour_metric": self.neighbour_metric,
            "hybrid_weights": list(self.hybrid_weights) if self.hybrid_weights else None,
            "risk_direction": "larger means more suspicious, not confirmed incorrect",
        }


def group_safe_audit_scores(
    features: NDArray[np.generic],
    observed_labels: Sequence[int] | NDArray[np.integer],
    probabilities: NDArray[np.generic],
    group_ids: Sequence[str],
    fold_ids: Sequence[int] | NDArray[np.integer],
    training_groups_by_fold: Mapping[int, Sequence[str]],
    *,
    sample_ids: Sequence[str],
    method: str,
    class_order: Sequence[int],
    neighbour_k: int = 7,
    neighbour_metric: str = "cosine",
    hybrid_weights: tuple[float, float] = (0.5, 0.5),
) -> GroupSafeAuditScoreResult:
    """Build a probability, neighbour, or fixed-hybrid audit score.

    This function never sees a hidden reference label or an outcome flag.  Switching
    strategies after inspecting those outcomes must therefore be recorded separately
    as exploratory method development.
    """

    requested = str(method).strip()
    probability_method = _PROBABILITY_METHODS.get(requested)
    if probability_method is not None:
        risk = score_annotations(
            observed_labels,
            probabilities,
            method=probability_method,
            class_order=class_order,
        )
        return GroupSafeAuditScoreResult(
            method=probability_method,
            risk_scores=risk,
            component_scores={probability_method: risk},
            neighbour_evidence=None,
            neighbour_k=None,
            neighbour_metric=None,
            hybrid_weights=None,
        )

    if requested not in _NEIGHBOUR_METHODS | _HYBRID_METHODS:
        raise ValueError(f"unsupported group-safe audit strategy: {method!r}")

    neighbour = fold_safe_neighbour_disagreement(
        features,
        observed_labels,
        group_ids,
        fold_ids,
        training_groups_by_fold,
        sample_ids=sample_ids,
        class_order=class_order,
        k=neighbour_k,
        metric=neighbour_metric,
    )
    neighbour_risk = np.asarray(neighbour.risk_scores, dtype=np.float64)
    components: dict[str, NDArray[np.float64]] = {"nearest_neighbour_disagreement": neighbour_risk}
    if requested in _NEIGHBOUR_METHODS:
        return GroupSafeAuditScoreResult(
            method="nearest_neighbour_disagreement",
            risk_scores=neighbour_risk,
            component_scores=components,
            neighbour_evidence=neighbour,
            neighbour_k=neighbour_k,
            neighbour_metric=neighbour_metric,
            hybrid_weights=None,
        )

    if (
        len(hybrid_weights) != 2
        or not np.isfinite(hybrid_weights).all()
        or any(value < 0.0 for value in hybrid_weights)
        or sum(hybrid_weights) <= 0.0
    ):
        raise ValueError("hybrid_weights must contain two finite non-negative values")
    self_confidence = score_annotations(
        observed_labels,
        probabilities,
        method="self_confidence",
        class_order=class_order,
    )
    components = {
        "self_confidence": self_confidence,
        "nearest_neighbour_disagreement": neighbour_risk,
    }
    risk = fixed_hybrid_score(
        components,
        components=("self_confidence", "nearest_neighbour_disagreement"),
        weights=hybrid_weights,
    )
    total = float(sum(hybrid_weights))
    normalised_weights = (
        float(hybrid_weights[0] / total),
        float(hybrid_weights[1] / total),
    )
    return GroupSafeAuditScoreResult(
        method="fixed_hybrid",
        risk_scores=risk,
        component_scores=components,
        neighbour_evidence=neighbour,
        neighbour_k=neighbour_k,
        neighbour_metric=neighbour_metric,
        hybrid_weights=normalised_weights,
    )


__all__ = ["GroupSafeAuditScoreResult", "group_safe_audit_scores"]
