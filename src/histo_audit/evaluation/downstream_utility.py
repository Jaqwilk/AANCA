"""Nested group-cross-fitted prediction of measured downstream intervention utility."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

MEASURED_DEVELOPMENT_INTERVENTION_UTILITY = "measured_development_intervention_utility"
CROSS_FITTED_UTILITY_EVIDENCE = "cross_fitted_development_estimate"


def _group_folds(
    groups: NDArray[np.str_], n_splits: int, seed: int
) -> tuple[tuple[NDArray[np.int64], NDArray[np.int64]], ...]:
    unique_groups = np.unique(groups)
    if n_splits < 2 or n_splits > len(unique_groups):
        raise ValueError("utility folds must be between two and the number of groups")
    from sklearn.model_selection import GroupKFold  # type: ignore[import-untyped]

    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros((len(groups), 1), dtype=np.float64)
    return tuple(
        (np.asarray(train, dtype=np.int64), np.asarray(holdout, dtype=np.int64))
        for train, holdout in splitter.split(dummy, groups=groups)
    )


def _fit_ridge(
    features: NDArray[np.float64], targets: NDArray[np.float64], l2: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    design = np.column_stack([(features - mean) / scale, np.ones(len(features))])
    penalty = np.eye(design.shape[1], dtype=np.float64) * l2
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ targets
    return mean, scale, np.asarray(coefficients, dtype=np.float64)


def _predict_ridge(
    features: NDArray[np.float64],
    mean: NDArray[np.float64],
    scale: NDArray[np.float64],
    coefficients: NDArray[np.float64],
) -> NDArray[np.float64]:
    design = np.column_stack([(features - mean) / scale, np.ones(len(features))])
    return np.asarray(design @ coefficients, dtype=np.float64)


def _conformal_radius(residuals: NDArray[np.float64], alpha: float) -> float:
    if residuals.ndim != 1 or not len(residuals) or not np.isfinite(residuals).all():
        raise ValueError("conformal residuals must be a finite non-empty vector")
    quantile = min(1.0, np.ceil((len(residuals) + 1) * (1.0 - alpha)) / len(residuals))
    return float(np.quantile(residuals, quantile, method="higher"))


@dataclass(frozen=True, slots=True)
class FrozenDownstreamUtilityEstimator:
    """Ridge utility model and one-sided development residual radius."""

    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    lower_bound_radius: float
    l2: float
    alpha: float
    feature_count: int
    target_evidence_role: str
    utility_evidence_role: str = CROSS_FITTED_UTILITY_EVIDENCE

    def predict(
        self, features: NDArray[np.generic]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        matrix = np.asarray(features, dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != self.feature_count
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("utility features differ from the frozen finite feature schema")
        estimate = _predict_ridge(matrix, self.mean, self.scale, self.coefficients)
        return estimate, np.asarray(estimate - self.lower_bound_radius, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class CrossFittedUtilityResult:
    """Per-sample development predictions and a frozen future-data estimator."""

    expected_downstream_gain: NDArray[np.float64]
    downstream_gain_lower_bound: NDArray[np.float64]
    absolute_residuals: NDArray[np.float64]
    fold_id: NDArray[np.int64]
    fold_lower_bound_radii: tuple[float, ...]
    frozen_estimator: FrozenDownstreamUtilityEstimator
    sample_ids: tuple[str, ...]
    target_evidence_role: str
    utility_evidence_role: str = CROSS_FITTED_UTILITY_EVIDENCE
    nested_group_cross_fitted: bool = True
    final_external_test_used: bool = False


def estimate_cross_fitted_downstream_utility(
    features: NDArray[np.generic],
    measured_downstream_gain: Sequence[float] | NDArray[np.floating],
    group_ids: Sequence[str],
    sample_ids: Sequence[str],
    *,
    target_evidence_role: str,
    outer_splits: int = 5,
    inner_splits: int = 3,
    split_seed: int = 26082091,
    l2: float = 1.0,
    alpha: float = 0.05,
) -> CrossFittedUtilityResult:
    """Learn utility only from measured development interventions, never NuCLS final.

    Every outer prediction excludes the target source group.  Its lower-bound radius
    is estimated from an inner group-cross-fitted residual set that also excludes the
    outer outcome.  The routine cannot create the measured gain targets; they must
    come from a prospectively defined development intervention experiment.
    """

    if target_evidence_role != MEASURED_DEVELOPMENT_INTERVENTION_UTILITY:
        raise ValueError("utility targets must be measured development interventions")
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(measured_downstream_gain, dtype=np.float64)
    groups = np.asarray(tuple(str(value) for value in group_ids), dtype=np.str_)
    identifiers = tuple(str(value) for value in sample_ids)
    if (
        matrix.ndim != 2
        or not len(matrix)
        or target.shape != (len(matrix),)
        or groups.shape != target.shape
        or not np.isfinite(matrix).all()
        or not np.isfinite(target).all()
        or any(not value for value in groups)
        or len(identifiers) != len(target)
        or len(set(identifiers)) != len(identifiers)
        or any(not value for value in identifiers)
    ):
        raise ValueError("utility features, targets, groups, and unique IDs must align")
    if not np.isfinite(l2) or l2 < 0.0:
        raise ValueError("utility ridge l2 must be finite and non-negative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("utility lower-bound alpha must lie in (0, 1)")

    outer = _group_folds(groups, outer_splits, split_seed)
    predictions = np.zeros(len(target), dtype=np.float64)
    lower_bounds = np.zeros(len(target), dtype=np.float64)
    fold_id = np.full(len(target), -1, dtype=np.int64)
    coverage = np.zeros(len(target), dtype=np.int64)
    fold_radii: list[float] = []
    for outer_fold_id, (outer_train, outer_holdout) in enumerate(outer):
        outer_train_groups = groups[outer_train]
        if inner_splits > len(np.unique(outer_train_groups)):
            raise ValueError("inner utility split count exceeds outer-training group count")
        inner_predictions = np.zeros(len(outer_train), dtype=np.float64)
        inner_coverage = np.zeros(len(outer_train), dtype=np.int64)
        for inner_train, inner_holdout in _group_folds(
            outer_train_groups, inner_splits, split_seed + outer_fold_id + 1
        ):
            mean, scale, coefficients = _fit_ridge(
                matrix[outer_train[inner_train]], target[outer_train[inner_train]], l2
            )
            inner_predictions[inner_holdout] = _predict_ridge(
                matrix[outer_train[inner_holdout]], mean, scale, coefficients
            )
            inner_coverage[inner_holdout] += 1
        if not np.array_equal(inner_coverage, np.ones(len(outer_train), dtype=np.int64)):
            raise RuntimeError("inner utility cross-fitting did not cover each row exactly once")
        radius = _conformal_radius(np.abs(target[outer_train] - inner_predictions), alpha)
        mean, scale, coefficients = _fit_ridge(matrix[outer_train], target[outer_train], l2)
        outer_prediction = _predict_ridge(matrix[outer_holdout], mean, scale, coefficients)
        predictions[outer_holdout] = outer_prediction
        lower_bounds[outer_holdout] = outer_prediction - radius
        fold_id[outer_holdout] = outer_fold_id
        coverage[outer_holdout] += 1
        fold_radii.append(radius)
    if not np.array_equal(coverage, np.ones(len(target), dtype=np.int64)):
        raise RuntimeError("outer utility cross-fitting did not cover each row exactly once")

    residuals = np.abs(target - predictions)
    frozen_radius = _conformal_radius(residuals, alpha)
    mean, scale, coefficients = _fit_ridge(matrix, target, l2)
    frozen = FrozenDownstreamUtilityEstimator(
        mean=mean,
        scale=scale,
        coefficients=coefficients,
        lower_bound_radius=frozen_radius,
        l2=float(l2),
        alpha=float(alpha),
        feature_count=matrix.shape[1],
        target_evidence_role=target_evidence_role,
    )
    return CrossFittedUtilityResult(
        expected_downstream_gain=predictions,
        downstream_gain_lower_bound=lower_bounds,
        absolute_residuals=residuals,
        fold_id=fold_id,
        fold_lower_bound_radii=tuple(fold_radii),
        frozen_estimator=frozen,
        sample_ids=identifiers,
        target_evidence_role=target_evidence_role,
    )


__all__ = [
    "CROSS_FITTED_UTILITY_EVIDENCE",
    "MEASURED_DEVELOPMENT_INTERVENTION_UTILITY",
    "CrossFittedUtilityResult",
    "FrozenDownstreamUtilityEstimator",
    "estimate_cross_fitted_downstream_utility",
]
