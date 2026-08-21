"""Development-only comparison of reviewed-label training interventions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from histo_audit.cross_validation.oof import MultinomialLogisticRegression

from .restoration import ClassificationMetrics, classification_metrics
from .retraining_guard import (
    INDEPENDENT_GROUP_VALIDATION,
    MulticriteriaRetrainingGuardDecision,
    evaluate_multicriteria_retraining_guard,
)
from .review_interventions import ReviewInterventionResult
from .review_training import SoftTargetMultinomialLogisticRegression


@dataclass(frozen=True, slots=True)
class TrainingStrategyEvidence:
    """One fixed training strategy evaluated on disjoint development groups."""

    strategy: str
    available: bool
    unavailable_reason: str | None
    metrics: ClassificationMetrics | None
    probabilities: NDArray[np.float64] | None
    adoption_guard: MulticriteriaRetrainingGuardDecision | None
    uses_soft_targets: bool
    uses_sample_weights: bool
    excluded_training_samples: int

    def as_dict(self) -> dict[str, object]:
        """Return compact evidence without duplicating the saved probability matrix."""

        return {
            "strategy": self.strategy,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "metrics": self.metrics.as_dict() if self.metrics is not None else None,
            "adoption_guard": (
                self.adoption_guard.as_dict() if self.adoption_guard is not None else None
            ),
            "uses_soft_targets": self.uses_soft_targets,
            "uses_sample_weights": self.uses_sample_weights,
            "excluded_training_samples": self.excluded_training_samples,
        }


@dataclass(frozen=True, slots=True)
class ReviewTrainingComparison:
    """Fail-closed strategy selection performed before any final external test."""

    uncorrected: TrainingStrategyEvidence
    candidates: tuple[TrainingStrategyEvidence, ...]
    selected_strategy: str
    apply_review_intervention: bool
    evidence_role: str
    training_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    final_external_test_used_for_selection: bool = False
    source_annotations_modified: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "uncorrected": self.uncorrected.as_dict(),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "selected_strategy": self.selected_strategy,
            "apply_review_intervention": self.apply_review_intervention,
            "evidence_role": self.evidence_role,
            "training_groups": list(self.training_groups),
            "validation_groups": list(self.validation_groups),
            "final_external_test_used_for_selection": self.final_external_test_used_for_selection,
            "source_annotations_modified": self.source_annotations_modified,
        }


def _one_hot(labels: NDArray[np.int64], classes: tuple[int, ...]) -> NDArray[np.float64]:
    lookup = {label: column for column, label in enumerate(classes)}
    if any(int(value) not in lookup for value in labels):
        raise ValueError("training label is absent from class_order")
    output = np.zeros((len(labels), len(classes)), dtype=np.float64)
    output[np.arange(len(labels)), [lookup[int(value)] for value in labels]] = 1.0
    return output


def _fit_hard(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    validation: NDArray[np.float64],
    classes: tuple[int, ...],
    *,
    l2: float,
    max_iter: int,
) -> NDArray[np.float64]:
    model = MultinomialLogisticRegression(
        class_order=classes,
        l2=l2,
        max_iter=max_iter,
        class_weight_balanced=True,
    )
    model.fit(features, labels)
    return np.asarray(model.predict_proba(validation), dtype=np.float64)


def _fit_soft(
    features: NDArray[np.float64],
    targets: NDArray[np.float64],
    weights: NDArray[np.float64],
    validation: NDArray[np.float64],
    classes: tuple[int, ...],
    *,
    l2: float,
    max_iter: int,
) -> NDArray[np.float64]:
    model = SoftTargetMultinomialLogisticRegression(
        class_order=classes,
        l2=l2,
        max_iter=max_iter,
        class_weight_balanced=True,
    )
    model.fit_soft_targets(features, targets, sample_weight=weights)
    return np.asarray(model.predict_proba(validation), dtype=np.float64)


def compare_review_training_strategies(
    training_features: NDArray[np.generic],
    observed_training_labels: Sequence[int] | NDArray[np.integer],
    intervention: ReviewInterventionResult,
    training_group_ids: Sequence[str],
    validation_features: NDArray[np.generic],
    validation_reference_labels: Sequence[int] | NDArray[np.integer],
    validation_group_ids: Sequence[str],
    *,
    class_order: Sequence[int],
    evidence_role: str,
    important_classes: Sequence[int] | None = None,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 26082071,
    minimum_macro_f1_effect: float = 0.0,
    minimum_per_class_recall_effect: float = -0.01,
    minimum_valid_iteration_fraction: float = 0.95,
    l2: float = 1.0e-2,
    max_iter: int = 400,
) -> ReviewTrainingComparison:
    """Compare reviewed-label policies on independent development groups only.

    This function deliberately has no final-test input.  Strategy adoption requires
    the multicriteria guard; adverse, uncertain, unavailable, or non-independent
    evidence retains the model trained on unchanged observed labels.
    """

    if evidence_role != INDEPENDENT_GROUP_VALIDATION:
        raise ValueError("strategy comparison requires independent group validation")
    train_x = np.asarray(training_features, dtype=np.float64)
    train_y = np.asarray(observed_training_labels, dtype=np.int64)
    validation_x = np.asarray(validation_features, dtype=np.float64)
    validation_y = np.asarray(validation_reference_labels, dtype=np.int64)
    train_groups_array = np.asarray(
        tuple(str(value) for value in training_group_ids), dtype=np.str_
    )
    validation_groups_array = np.asarray(
        tuple(str(value) for value in validation_group_ids), dtype=np.str_
    )
    classes = tuple(int(value) for value in class_order)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("class_order must contain at least two unique values")
    if (
        train_x.ndim != 2
        or validation_x.ndim != 2
        or not len(train_x)
        or not len(validation_x)
        or train_x.shape[1] != validation_x.shape[1]
        or train_y.shape != (len(train_x),)
        or validation_y.shape != (len(validation_x),)
        or train_groups_array.shape != train_y.shape
        or validation_groups_array.shape != validation_y.shape
        or not np.isfinite(train_x).all()
        or not np.isfinite(validation_x).all()
        or any(not value for value in train_groups_array)
        or any(not value for value in validation_groups_array)
    ):
        raise ValueError("training and validation data must be finite, aligned, and non-empty")
    overlap = sorted(set(train_groups_array).intersection(validation_groups_array))
    if overlap:
        raise ValueError(f"training and validation groups overlap: {overlap}")
    if not np.array_equal(intervention.source_observed_labels, train_y):
        raise ValueError("intervention source labels differ from observed training labels")
    if (
        intervention.soft_targets.shape != (len(train_y), len(classes))
        or intervention.class_order != classes
    ):
        raise ValueError("intervention targets do not follow the fixed class_order")

    baseline_probabilities = _fit_hard(
        train_x,
        train_y,
        validation_x,
        classes,
        l2=l2,
        max_iter=max_iter,
    )
    baseline_metrics = classification_metrics(
        validation_y, baseline_probabilities, class_order=classes
    )
    baseline = TrainingStrategyEvidence(
        strategy="uncorrected_observed_labels",
        available=True,
        unavailable_reason=None,
        metrics=baseline_metrics,
        probabilities=baseline_probabilities,
        adoption_guard=None,
        uses_soft_targets=False,
        uses_sample_weights=False,
        excluded_training_samples=0,
    )

    non_excluded = intervention.training_weights > 0.0
    one_hot_hard = _one_hot(intervention.derived_hard_labels, classes)
    specifications = (
        (
            "gated_hard_changes",
            False,
            False,
            lambda: _fit_hard(
                train_x[non_excluded],
                intervention.derived_hard_labels[non_excluded],
                validation_x,
                classes,
                l2=l2,
                max_iter=max_iter,
            ),
        ),
        (
            "soft_labels",
            True,
            False,
            lambda: _fit_soft(
                train_x,
                intervention.soft_targets,
                non_excluded.astype(np.float64),
                validation_x,
                classes,
                l2=l2,
                max_iter=max_iter,
            ),
        ),
        (
            "downweighted_hard_targets",
            False,
            True,
            lambda: _fit_soft(
                train_x,
                one_hot_hard,
                intervention.training_weights,
                validation_x,
                classes,
                l2=l2,
                max_iter=max_iter,
            ),
        ),
        (
            "soft_labels_with_abstention_weights",
            True,
            True,
            lambda: _fit_soft(
                train_x,
                intervention.soft_targets,
                intervention.training_weights,
                validation_x,
                classes,
                l2=l2,
                max_iter=max_iter,
            ),
        ),
    )
    candidates: list[TrainingStrategyEvidence] = []
    for name, uses_soft, uses_weights, fit in specifications:
        try:
            probabilities = fit()
            metrics = classification_metrics(validation_y, probabilities, class_order=classes)
            guard = evaluate_multicriteria_retraining_guard(
                validation_y,
                baseline_probabilities,
                probabilities,
                validation_groups_array.tolist(),
                class_order=classes,
                evidence_role=evidence_role,
                important_classes=important_classes,
                n_iterations=bootstrap_iterations,
                seed=bootstrap_seed,
                minimum_macro_f1_effect=minimum_macro_f1_effect,
                minimum_per_class_recall_effect=minimum_per_class_recall_effect,
                minimum_valid_iteration_fraction=minimum_valid_iteration_fraction,
            )
        except (RuntimeError, ValueError) as error:
            candidates.append(
                TrainingStrategyEvidence(
                    strategy=name,
                    available=False,
                    unavailable_reason=f"{type(error).__name__}: {error}",
                    metrics=None,
                    probabilities=None,
                    adoption_guard=None,
                    uses_soft_targets=uses_soft,
                    uses_sample_weights=uses_weights,
                    excluded_training_samples=int((~non_excluded).sum()),
                )
            )
            continue
        candidates.append(
            TrainingStrategyEvidence(
                strategy=name,
                available=True,
                unavailable_reason=None,
                metrics=metrics,
                probabilities=probabilities,
                adoption_guard=guard,
                uses_soft_targets=uses_soft,
                uses_sample_weights=uses_weights,
                excluded_training_samples=int((~non_excluded).sum()),
            )
        )

    passing = [
        candidate
        for candidate in candidates
        if candidate.adoption_guard is not None and candidate.adoption_guard.apply_candidate
    ]
    if passing:

        def passing_key(candidate: TrainingStrategyEvidence) -> tuple[float, float, str]:
            guard = candidate.adoption_guard
            if guard is None or guard.macro_f1.interval_95 is None:
                raise RuntimeError("passing candidate lacks complete adoption evidence")
            return (
                -guard.macro_f1.interval_95[0],
                -guard.macro_f1.candidate_minus_uncorrected_macro_f1,
                candidate.strategy,
            )

        passing.sort(key=passing_key)
        selected = passing[0].strategy
    else:
        selected = baseline.strategy
    return ReviewTrainingComparison(
        uncorrected=baseline,
        candidates=tuple(candidates),
        selected_strategy=selected,
        apply_review_intervention=selected != baseline.strategy,
        evidence_role=evidence_role,
        training_groups=tuple(sorted(set(str(value) for value in train_groups_array))),
        validation_groups=tuple(sorted(set(str(value) for value in validation_groups_array))),
    )


__all__ = [
    "ReviewTrainingComparison",
    "TrainingStrategyEvidence",
    "compare_review_training_strategies",
]
