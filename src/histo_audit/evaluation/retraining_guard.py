"""Fail-closed guard for applying reviewed-label retraining candidates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from .restoration import (
    classification_metrics,
    macro_f1_from_confusion,
    per_class_recall_from_confusion,
)

INDEPENDENT_GROUP_VALIDATION = "independent_group_validation"


@dataclass(frozen=True, slots=True)
class RetrainingGuardDecision:
    """Decision based on a paired whole-group bootstrap of macro F1."""

    apply_candidate: bool
    action: str
    candidate_minus_uncorrected_macro_f1: float
    interval_95: tuple[float, float] | None
    minimum_effect: float
    group_count: int
    requested_iterations: int
    valid_iterations: int
    evidence_role: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible decision evidence."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class MulticriteriaRetrainingGuardDecision:
    """Adoption decision requiring macro-F1 benefit and class-level non-degradation."""

    apply_candidate: bool
    action: str
    macro_f1: RetrainingGuardDecision
    important_classes: tuple[int, ...]
    candidate_minus_uncorrected_recall: dict[int, float]
    per_class_recall_intervals_95: dict[int, tuple[float, float] | None]
    per_class_valid_iterations: dict[int, int]
    minimum_per_class_recall_effect: float
    minimum_valid_iteration_fraction: float
    important_classes_pass: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible evidence with explicit string class keys."""

        return {
            "apply_candidate": self.apply_candidate,
            "action": self.action,
            "macro_f1": self.macro_f1.as_dict(),
            "important_classes": list(self.important_classes),
            "candidate_minus_uncorrected_recall": {
                str(key): value for key, value in self.candidate_minus_uncorrected_recall.items()
            },
            "per_class_recall_intervals_95": {
                str(key): list(value) if value is not None else None
                for key, value in self.per_class_recall_intervals_95.items()
            },
            "per_class_valid_iterations": {
                str(key): value for key, value in self.per_class_valid_iterations.items()
            },
            "minimum_per_class_recall_effect": self.minimum_per_class_recall_effect,
            "minimum_valid_iteration_fraction": self.minimum_valid_iteration_fraction,
            "important_classes_pass": self.important_classes_pass,
            "reason": self.reason,
        }


def _group_confusions(
    reference: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    groups: NDArray[np.str_],
    unique_groups: tuple[str, ...],
    classes: tuple[int, ...],
) -> NDArray[np.int64]:
    lookup = {label: index for index, label in enumerate(classes)}
    predictions = np.asarray(classes, dtype=np.int64)[np.argmax(probabilities, axis=1)]
    group_lookup = {group: index for index, group in enumerate(unique_groups)}
    output = np.zeros((len(unique_groups), len(classes), len(classes)), dtype=np.int64)
    for truth, prediction, group in zip(reference, predictions, groups, strict=True):
        output[
            group_lookup[str(group)],
            lookup[int(truth)],
            lookup[int(prediction)],
        ] += 1
    return output


def evaluate_retraining_guard(
    reference_labels: Sequence[int] | NDArray[np.integer],
    uncorrected_probabilities: NDArray[np.generic],
    candidate_probabilities: NDArray[np.generic],
    group_ids: Sequence[str],
    *,
    class_order: Sequence[int],
    evidence_role: str,
    n_iterations: int = 2000,
    seed: int = 26082071,
    minimum_effect: float = 0.0,
) -> RetrainingGuardDecision:
    """Permit a retraining candidate only after positive independent group evidence.

    The guard is intentionally asymmetric: uncertain, neutral, adverse, in-sample,
    or too-small evidence all retain the uncorrected model.  It does not certify
    clinical utility and never changes source annotations.
    """

    reference = np.asarray(reference_labels, dtype=np.int64)
    baseline = np.asarray(uncorrected_probabilities, dtype=np.float64)
    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    groups = np.asarray(tuple(str(value) for value in group_ids), dtype=np.str_)
    classes = tuple(int(value) for value in class_order)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("class_order must contain at least two unique values")
    expected_shape = (len(reference), len(classes))
    if not len(reference) or baseline.shape != expected_shape or candidate.shape != expected_shape:
        raise ValueError("reference labels and both probability matrices must align")
    if groups.shape != reference.shape or any(not value for value in groups):
        raise ValueError("group IDs must be non-empty and align with labels")
    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive")
    if not np.isfinite(minimum_effect):
        raise ValueError("minimum_effect must be finite")
    baseline_metrics = classification_metrics(reference, baseline, class_order=classes)
    candidate_metrics = classification_metrics(reference, candidate, class_order=classes)
    estimate = float(candidate_metrics.macro_f1 - baseline_metrics.macro_f1)
    unique_groups = tuple(sorted(set(str(value) for value in groups)))

    if len(unique_groups) < 2:
        return RetrainingGuardDecision(
            apply_candidate=False,
            action="retain_uncorrected",
            candidate_minus_uncorrected_macro_f1=estimate,
            interval_95=None,
            minimum_effect=float(minimum_effect),
            group_count=len(unique_groups),
            requested_iterations=n_iterations,
            valid_iterations=0,
            evidence_role=evidence_role,
            reason="fewer than two independent validation groups; guard failed closed",
        )

    baseline_confusions = _group_confusions(reference, baseline, groups, unique_groups, classes)
    candidate_confusions = _group_confusions(reference, candidate, groups, unique_groups, classes)
    rng = np.random.default_rng(seed)
    differences = np.empty(n_iterations, dtype=np.float64)
    for index in range(n_iterations):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        differences[index] = macro_f1_from_confusion(
            candidate_confusions[sampled].sum(axis=0)
        ) - macro_f1_from_confusion(baseline_confusions[sampled].sum(axis=0))
    interval = (
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975)),
    )
    independent = evidence_role == INDEPENDENT_GROUP_VALIDATION
    apply_candidate = independent and interval[0] > minimum_effect
    if not independent:
        reason = "evidence is not an independent group-held-out validation; guard failed closed"
    elif apply_candidate:
        reason = "lower 95% whole-group bootstrap bound exceeds the minimum effect"
    else:
        reason = "lower 95% whole-group bootstrap bound does not exceed the minimum effect"
    return RetrainingGuardDecision(
        apply_candidate=apply_candidate,
        action="apply_candidate" if apply_candidate else "retain_uncorrected",
        candidate_minus_uncorrected_macro_f1=estimate,
        interval_95=interval,
        minimum_effect=float(minimum_effect),
        group_count=len(unique_groups),
        requested_iterations=n_iterations,
        valid_iterations=len(differences),
        evidence_role=evidence_role,
        reason=reason,
    )


def evaluate_multicriteria_retraining_guard(
    reference_labels: Sequence[int] | NDArray[np.integer],
    uncorrected_probabilities: NDArray[np.generic],
    candidate_probabilities: NDArray[np.generic],
    group_ids: Sequence[str],
    *,
    class_order: Sequence[int],
    evidence_role: str,
    important_classes: Sequence[int] | None = None,
    n_iterations: int = 2000,
    seed: int = 26082071,
    minimum_macro_f1_effect: float = 0.0,
    minimum_per_class_recall_effect: float = -0.01,
    minimum_valid_iteration_fraction: float = 0.95,
) -> MulticriteriaRetrainingGuardDecision:
    """Require both macro-F1 benefit and bounded recall loss for important classes.

    Per-class intervals use the same whole-group bootstrap as the macro decision.
    A resample that contains no reference example for a class is undefined for that
    class and is omitted.  If any important class has no valid interval, the guard
    fails closed.  The final external test must not be used to choose these limits.
    """

    classes = tuple(int(value) for value in class_order)
    selected_classes = (
        classes if important_classes is None else tuple(int(value) for value in important_classes)
    )
    if not selected_classes or len(set(selected_classes)) != len(selected_classes):
        raise ValueError("important_classes must contain unique class labels")
    unknown = sorted(set(selected_classes).difference(classes))
    if unknown:
        raise ValueError(f"important_classes are absent from class_order: {unknown}")
    if not np.isfinite(minimum_per_class_recall_effect):
        raise ValueError("minimum_per_class_recall_effect must be finite")
    if not 0.0 < minimum_valid_iteration_fraction <= 1.0:
        raise ValueError("minimum_valid_iteration_fraction must lie in (0, 1]")

    macro = evaluate_retraining_guard(
        reference_labels,
        uncorrected_probabilities,
        candidate_probabilities,
        group_ids,
        class_order=classes,
        evidence_role=evidence_role,
        n_iterations=n_iterations,
        seed=seed,
        minimum_effect=minimum_macro_f1_effect,
    )
    reference = np.asarray(reference_labels, dtype=np.int64)
    baseline = np.asarray(uncorrected_probabilities, dtype=np.float64)
    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    groups = np.asarray(tuple(str(value) for value in group_ids), dtype=np.str_)
    unique_groups = tuple(sorted(set(str(value) for value in groups)))
    class_columns = {label: index for index, label in enumerate(classes)}
    point_baseline_metrics = classification_metrics(reference, baseline, class_order=classes)
    point_candidate_metrics = classification_metrics(reference, candidate, class_order=classes)
    point_differences = {
        label: float(
            point_candidate_metrics.per_class_recall[class_columns[label]]
            - point_baseline_metrics.per_class_recall[class_columns[label]]
        )
        for label in selected_classes
    }

    if len(unique_groups) < 2:
        intervals = {label: None for label in selected_classes}
        valid_iterations = {label: 0 for label in selected_classes}
    else:
        baseline_confusions = _group_confusions(reference, baseline, groups, unique_groups, classes)
        candidate_confusions = _group_confusions(
            reference, candidate, groups, unique_groups, classes
        )
        samples: dict[int, list[float]] = {label: [] for label in selected_classes}
        rng = np.random.default_rng(seed)
        for _ in range(n_iterations):
            sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
            baseline_recall = per_class_recall_from_confusion(
                baseline_confusions[sampled].sum(axis=0)
            )
            candidate_recall = per_class_recall_from_confusion(
                candidate_confusions[sampled].sum(axis=0)
            )
            for label in selected_classes:
                column = class_columns[label]
                difference = candidate_recall[column] - baseline_recall[column]
                if np.isfinite(difference):
                    samples[label].append(float(difference))
        intervals = {
            label: (
                (
                    float(np.quantile(values, 0.025)),
                    float(np.quantile(values, 0.975)),
                )
                if values
                else None
            )
            for label, values in samples.items()
        }
        valid_iterations = {label: len(values) for label, values in samples.items()}

    independent = evidence_role == INDEPENDENT_GROUP_VALIDATION
    minimum_valid_iterations = int(np.ceil(n_iterations * minimum_valid_iteration_fraction))
    class_pass = independent and all(
        intervals[label] is not None
        and intervals[label][0] >= minimum_per_class_recall_effect
        and valid_iterations[label] >= minimum_valid_iterations
        for label in selected_classes
    )
    apply_candidate = macro.apply_candidate and class_pass
    if not independent:
        reason = "evidence is not independent group-held-out validation; guard failed closed"
    elif not macro.apply_candidate:
        reason = "macro-F1 benefit gate failed; retain the uncorrected model"
    elif not class_pass:
        reason = "at least one important class failed the registered recall non-degradation gate"
    else:
        reason = "macro-F1 and every important-class recall gate passed"
    return MulticriteriaRetrainingGuardDecision(
        apply_candidate=apply_candidate,
        action="apply_candidate" if apply_candidate else "retain_uncorrected",
        macro_f1=macro,
        important_classes=selected_classes,
        candidate_minus_uncorrected_recall=point_differences,
        per_class_recall_intervals_95=intervals,
        per_class_valid_iterations=valid_iterations,
        minimum_per_class_recall_effect=float(minimum_per_class_recall_effect),
        minimum_valid_iteration_fraction=float(minimum_valid_iteration_fraction),
        important_classes_pass=class_pass,
        reason=reason,
    )


__all__ = [
    "INDEPENDENT_GROUP_VALIDATION",
    "MulticriteriaRetrainingGuardDecision",
    "RetrainingGuardDecision",
    "evaluate_multicriteria_retraining_guard",
    "evaluate_retraining_guard",
]
