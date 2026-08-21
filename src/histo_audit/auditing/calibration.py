"""Group-cross-fitted temperature calibration for fresh expert development data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]

from histo_audit.cross_validation.oof import make_group_stratified_fold_plan
from histo_audit.evaluation.restoration import ClassificationMetrics, classification_metrics

from .two_queue import GROUP_SAFE_OOF_EVIDENCE

NEW_EXPERT_DEVELOPMENT_LABELS = "new_expert_development_labels"


def _validate_probabilities(
    probabilities: NDArray[np.generic],
    labels: Sequence[int] | NDArray[np.integer],
    class_order: Sequence[int],
) -> tuple[NDArray[np.float64], NDArray[np.int64], tuple[int, ...]]:
    matrix = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    classes = tuple(int(value) for value in class_order)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("class_order must contain at least two unique values")
    if matrix.shape != (len(target), len(classes)) or not len(target):
        raise ValueError("probabilities and labels must be non-empty and aligned")
    if (
        not np.isfinite(matrix).all()
        or np.any(matrix < 0.0)
        or np.any(matrix > 1.0)
        or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8)
    ):
        raise ValueError("probability rows must be finite values in [0, 1] summing to one")
    if any(int(value) not in classes for value in target):
        raise ValueError("calibration label is absent from class_order")
    return matrix, target, classes


def _temperature_transform(
    probabilities: NDArray[np.float64], temperature: float
) -> NDArray[np.float64]:
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    logits = np.log(np.clip(probabilities, 1e-15, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exponent = np.exp(logits)
    return np.asarray(exponent / exponent.sum(axis=1, keepdims=True), dtype=np.float64)


def _negative_log_likelihood(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
    classes: tuple[int, ...],
) -> float:
    lookup = {label: column for column, label in enumerate(classes)}
    columns = np.asarray([lookup[int(value)] for value in labels], dtype=np.int64)
    return float(
        -np.mean(np.log(np.clip(probabilities[np.arange(len(labels)), columns], 1e-15, 1.0)))
    )


def _fit_temperature(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
    classes: tuple[int, ...],
    bounds: tuple[float, float],
) -> float:
    lower, upper = bounds
    if not 0.0 < lower < upper:
        raise ValueError("temperature_bounds must be positive and increasing")

    def objective(temperature: float) -> float:
        return _negative_log_likelihood(
            _temperature_transform(probabilities, temperature), labels, classes
        )

    result = minimize_scalar(
        objective,
        bounds=(float(lower), float(upper)),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError("temperature optimisation did not converge")
    return float(result.x)


@dataclass(frozen=True, slots=True)
class FrozenTemperatureScaler:
    """One scalar fitted only after cross-fitted development evidence passes."""

    temperature: float
    class_order: tuple[int, ...]
    evidence_role: str

    def transform(self, probabilities: NDArray[np.generic]) -> NDArray[np.float64]:
        matrix = np.asarray(probabilities, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.class_order):
            raise ValueError("probability columns differ from the frozen class order")
        if (
            not np.isfinite(matrix).all()
            or np.any(matrix < 0.0)
            or np.any(matrix > 1.0)
            or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8)
        ):
            raise ValueError("probability rows are invalid")
        return _temperature_transform(matrix, self.temperature)


@dataclass(frozen=True, slots=True)
class CrossFittedCalibrationResult:
    """Calibration evidence that never fits a scored group's own labels."""

    raw_probabilities: NDArray[np.float64]
    cross_fitted_probabilities: NDArray[np.float64]
    selected_probabilities: NDArray[np.float64]
    fold_id: NDArray[np.int64]
    fold_temperatures: tuple[float, ...]
    frozen_scaler: FrozenTemperatureScaler | None
    uncalibrated_metrics: ClassificationMetrics
    calibrated_metrics: ClassificationMetrics
    uncalibrated_negative_log_likelihood: float
    calibrated_negative_log_likelihood: float
    calibration_adopted: bool
    adoption_reason: str
    evidence_role: str
    probability_evidence_role: str
    sample_ids: tuple[str, ...]
    group_safe_cross_fitted: bool = True
    final_external_test_used: bool = False


def cross_fitted_temperature_calibration(
    probabilities: NDArray[np.generic],
    expert_reference_labels: Sequence[int] | NDArray[np.integer],
    group_ids: Sequence[str],
    sample_ids: Sequence[str],
    *,
    class_order: Sequence[int],
    evidence_role: str,
    probability_evidence_role: str,
    n_splits: int = 5,
    split_seed: int = 26082081,
    temperature_bounds: tuple[float, float] = (0.05, 20.0),
    maximum_ece_increase: float = 0.0,
) -> CrossFittedCalibrationResult:
    """Evaluate and freeze temperature scaling without using NuCLS or a final test.

    Adoption requires lower cross-fitted negative log-likelihood and no ECE increase
    beyond the predeclared tolerance.  Otherwise raw probabilities remain selected.
    """

    if evidence_role != NEW_EXPERT_DEVELOPMENT_LABELS:
        raise ValueError("calibration requires newly collected expert development labels")
    if probability_evidence_role != GROUP_SAFE_OOF_EVIDENCE:
        raise ValueError("calibration requires group-safe OOF input probabilities")
    matrix, target, classes = _validate_probabilities(
        probabilities, expert_reference_labels, class_order
    )
    groups = np.asarray(tuple(str(value) for value in group_ids), dtype=np.str_)
    identifiers = tuple(str(value) for value in sample_ids)
    if (
        groups.shape != target.shape
        or any(not value for value in groups)
        or len(identifiers) != len(target)
        or len(set(identifiers)) != len(identifiers)
        or any(not value for value in identifiers)
    ):
        raise ValueError("group and unique sample IDs must align with calibration rows")
    if not np.isfinite(maximum_ece_increase) or maximum_ece_increase < 0.0:
        raise ValueError("maximum_ece_increase must be finite and non-negative")
    plan = make_group_stratified_fold_plan(
        target,
        groups.tolist(),
        n_splits=n_splits,
        class_order=classes,
        seed=split_seed,
    )
    calibrated = np.zeros_like(matrix)
    coverage = np.zeros(len(target), dtype=np.int64)
    fold_id = np.full(len(target), -1, dtype=np.int64)
    temperatures: list[float] = []
    for fold in plan.folds:
        if set(fold.training_groups).intersection(fold.held_out_groups):
            raise RuntimeError("calibration fold leaks a held-out source group")
        temperature = _fit_temperature(
            matrix[fold.train_indices],
            target[fold.train_indices],
            classes,
            temperature_bounds,
        )
        calibrated[fold.holdout_indices] = _temperature_transform(
            matrix[fold.holdout_indices], temperature
        )
        coverage[fold.holdout_indices] += 1
        fold_id[fold.holdout_indices] = fold.fold_id
        temperatures.append(temperature)
    if not np.array_equal(coverage, np.ones(len(target), dtype=np.int64)):
        raise RuntimeError("cross-fitted calibration did not cover every row exactly once")

    raw_metrics = classification_metrics(target, matrix, class_order=classes)
    calibrated_metrics = classification_metrics(target, calibrated, class_order=classes)
    raw_nll = _negative_log_likelihood(matrix, target, classes)
    calibrated_nll = _negative_log_likelihood(calibrated, target, classes)
    improves_nll = calibrated_nll < raw_nll
    preserves_ece = (
        calibrated_metrics.expected_calibration_error
        <= raw_metrics.expected_calibration_error + maximum_ece_increase
    )
    adopted = improves_nll and preserves_ece
    if adopted:
        frozen_temperature = _fit_temperature(matrix, target, classes, temperature_bounds)
        frozen_scaler = FrozenTemperatureScaler(
            temperature=frozen_temperature,
            class_order=classes,
            evidence_role=evidence_role,
        )
        reason = "cross-fitted NLL improved and ECE stayed within the registered tolerance"
    else:
        frozen_scaler = None
        reason = "calibration gate failed; retain raw probabilities"
    return CrossFittedCalibrationResult(
        raw_probabilities=matrix,
        cross_fitted_probabilities=calibrated,
        selected_probabilities=calibrated if adopted else matrix,
        fold_id=fold_id,
        fold_temperatures=tuple(temperatures),
        frozen_scaler=frozen_scaler,
        uncalibrated_metrics=raw_metrics,
        calibrated_metrics=calibrated_metrics,
        uncalibrated_negative_log_likelihood=raw_nll,
        calibrated_negative_log_likelihood=calibrated_nll,
        calibration_adopted=adopted,
        adoption_reason=reason,
        evidence_role=evidence_role,
        probability_evidence_role=probability_evidence_role,
        sample_ids=identifiers,
    )


__all__ = [
    "NEW_EXPERT_DEVELOPMENT_LABELS",
    "CrossFittedCalibrationResult",
    "FrozenTemperatureScaler",
    "cross_fitted_temperature_calibration",
]
