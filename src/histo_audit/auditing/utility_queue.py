"""Conservative model-improvement queue from independently measured utility."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .scores import percentile_normalise
from .two_queue import (
    CROSS_FITTED_UTILITY_EVIDENCE,
    GROUP_SAFE_OOF_EVIDENCE,
    BalancedReviewQueue,
    QueueConstraints,
    _cosine_ready,
    _select_queue,
    _unavailable_queue,
    _values,
)

UTILITY_PRODUCT_PRIORITY = (
    "percentile_annotation_inconsistency_score_times_positive_utility_lower_bound"
)


@dataclass(frozen=True, slots=True)
class MeasuredUtilityQueueResult:
    """One fail-closed queue and the exact factors used to order it."""

    model_improvement: BalancedReviewQueue
    combined_priority_scores: NDArray[np.float64]
    annotation_risk_percentiles: NDArray[np.float64]
    expected_downstream_gain: NDArray[np.float64]
    downstream_gain_lower_bound: NDArray[np.float64]
    priority_method: str = UTILITY_PRODUCT_PRIORITY
    annotation_evidence_role: str = GROUP_SAFE_OOF_EVIDENCE
    utility_evidence_role: str = CROSS_FITTED_UTILITY_EVIDENCE
    source_annotations_modified: bool = False


def build_measured_utility_queue(
    annotation_inconsistency_scores: Sequence[float] | NDArray[np.floating],
    expected_downstream_gain: Sequence[float] | NDArray[np.floating],
    downstream_gain_lower_bound: Sequence[float] | NDArray[np.floating],
    group_ids: Sequence[str],
    observed_labels: Sequence[int],
    sample_ids: Sequence[str],
    *,
    constraints: QueueConstraints,
    annotation_evidence_role: str,
    utility_evidence_role: str,
    proposed_labels: Sequence[int] | None = None,
    tissue_types: Sequence[str] | None = None,
    embeddings: NDArray[np.generic] | None = None,
    minimum_annotation_score: float = 0.0,
    minimum_downstream_gain: float = 0.0,
) -> MeasuredUtilityQueueResult:
    """Rank by audit-risk percentile times a positive conservative utility bound.

    The audit score is deliberately treated as a rank factor, not as ``P(error)``.
    Utility must be measured on development interventions and predicted with the
    nested group-cross-fitted estimator. Invalid utility provenance makes the queue
    unavailable instead of silently falling back to annotation risk alone.
    """

    risks = np.asarray(annotation_inconsistency_scores, dtype=np.float64)
    expected = np.asarray(expected_downstream_gain, dtype=np.float64)
    lower = np.asarray(downstream_gain_lower_bound, dtype=np.float64)
    n_samples = len(risks)
    identifiers = tuple(str(value) for value in sample_ids)
    groups = _values(group_ids, n_samples, role="group")
    classes = _values(observed_labels, n_samples, role="observed-label")
    tissues = _values(tissue_types, n_samples, role="tissue")
    proposals = _values(proposed_labels, n_samples, role="proposed-label")
    if (
        risks.ndim != 1
        or not n_samples
        or expected.shape != risks.shape
        or lower.shape != risks.shape
        or not np.isfinite(risks).all()
        or not np.isfinite(expected).all()
        or not np.isfinite(lower).all()
        or np.any(lower > expected + 1.0e-12)
        or len(identifiers) != n_samples
        or len(set(identifiers)) != n_samples
        or any(not value for value in identifiers)
    ):
        raise ValueError("risk, utility and unique sample IDs must be finite and aligned")
    if groups is None or classes is None:
        raise RuntimeError("required utility-queue values were not materialised")
    if annotation_evidence_role != GROUP_SAFE_OOF_EVIDENCE:
        raise ValueError("model-improvement ranking requires group-safe OOF audit evidence")
    if (
        not np.isfinite(minimum_annotation_score)
        or not np.isfinite(minimum_downstream_gain)
        or minimum_downstream_gain < 0.0
    ):
        raise ValueError("utility-queue thresholds must be finite and non-negative")

    risk_percentiles = percentile_normalise(risks)
    positive_lower = np.maximum(lower, 0.0)
    combined = risk_percentiles * positive_lower
    if utility_evidence_role != CROSS_FITTED_UTILITY_EVIDENCE:
        queue = _unavailable_queue(
            "model_improvement",
            constraints,
            "utility evidence is not a cross-fitted measured development estimate",
        )
    else:
        transitions = (
            np.asarray(
                [
                    f"{observed}->{proposed}"
                    for observed, proposed in zip(classes, proposals, strict=True)
                ],
                dtype=np.str_,
            )
            if proposals is not None
            else None
        )
        normalised_embeddings = _cosine_ready(
            embeddings, n_samples, constraints.minimum_cosine_distance
        )
        eligible = (risks >= minimum_annotation_score) & (lower > minimum_downstream_gain)
        queue = _select_queue(
            "model_improvement",
            combined,
            eligible,
            identifiers,
            groups,
            classes,
            tissues,
            transitions,
            normalised_embeddings,
            constraints,
        )
    return MeasuredUtilityQueueResult(
        model_improvement=queue,
        combined_priority_scores=combined,
        annotation_risk_percentiles=risk_percentiles,
        expected_downstream_gain=expected,
        downstream_gain_lower_bound=lower,
        annotation_evidence_role=annotation_evidence_role,
        utility_evidence_role=utility_evidence_role,
    )


__all__ = [
    "UTILITY_PRODUCT_PRIORITY",
    "MeasuredUtilityQueueResult",
    "build_measured_utility_queue",
]
