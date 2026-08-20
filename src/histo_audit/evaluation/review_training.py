"""Experimental weighted/soft-target training for reviewed annotations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from histo_audit.cross_validation.oof import MultinomialLogisticRegression

from .review_interventions import ReviewInterventionResult


class SoftTargetMultinomialLogisticRegression(MultinomialLogisticRegression):
    """The existing deterministic classifier with an explicit soft-target fit path.

    The inherited hard-label ``fit`` method is intentionally untouched. This method
    is development-only until a strategy is selected and frozen on independent data.
    """

    def fit_soft_targets(
        self,
        features: NDArray[np.generic],
        target_probabilities: NDArray[np.generic],
        *,
        sample_weight: Sequence[float] | NDArray[np.floating] | None = None,
    ) -> SoftTargetMultinomialLogisticRegression:
        matrix = np.asarray(features, dtype=np.float64)
        targets = np.asarray(target_probabilities, dtype=np.float64)
        weights = (
            np.ones(len(matrix), dtype=np.float64)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        expected_shape = (len(matrix), len(self.class_order))
        if matrix.ndim != 2 or not len(matrix) or targets.shape != expected_shape:
            raise ValueError("features and soft targets must be non-empty and aligned")
        if weights.shape != (len(matrix),):
            raise ValueError("sample weights must align with features")
        if (
            not np.isfinite(matrix).all()
            or not np.isfinite(targets).all()
            or not np.isfinite(weights).all()
            or np.any(targets < 0.0)
            or np.any(weights < 0.0)
            or not np.allclose(targets.sum(axis=1), 1.0, atol=1e-8)
            or float(weights.sum()) <= 0.0
        ):
            raise ValueError("soft targets and weights must be finite valid distributions")

        weight_sum = float(weights.sum())
        self.mean_ = np.average(matrix, axis=0, weights=weights)
        variance = np.average((matrix - self.mean_) ** 2, axis=0, weights=weights)
        self.scale_ = np.sqrt(variance)
        self.scale_[self.scale_ < 1e-12] = 1.0
        standardised = (matrix - self.mean_) / self.scale_
        design = np.column_stack([standardised, np.ones(len(standardised))])

        if self.class_weight_balanced:
            effective_counts = (weights[:, None] * targets).sum(axis=0)
            if np.any(effective_counts <= 0.0):
                raise ValueError("soft targets must retain positive mass for every fixed class")
            class_weights = weight_sum / (len(self.class_order) * effective_counts)
        else:
            class_weights = np.ones(len(self.class_order), dtype=np.float64)
        weighted_targets = targets * class_weights[None, :]
        per_sample_mass = weighted_targets.sum(axis=1)
        objective_weight = float(np.sum(weights * per_sample_mass))
        shape = (design.shape[1], len(self.class_order))

        def objective(flat: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            coefficients = flat.reshape(shape)
            probabilities = self._softmax(design @ coefficients)
            log_likelihood = np.log(np.clip(probabilities, 1e-15, 1.0))
            loss = (
                -float(np.sum(weights[:, None] * weighted_targets * log_likelihood))
                / objective_weight
            )
            loss += 0.5 * self.l2 * float(np.sum(coefficients[:-1] ** 2))
            residual = (
                (probabilities * per_sample_mass[:, None] - weighted_targets)
                * weights[:, None]
                / objective_weight
            )
            gradient = design.T @ residual
            gradient[:-1] += self.l2 * coefficients[:-1]
            return loss, gradient.ravel()

        initial = np.zeros(shape, dtype=np.float64).ravel()
        try:
            from scipy.optimize import minimize  # type: ignore[import-untyped]

            optimisation = minimize(
                objective,
                initial,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": self.max_iter, "ftol": 1e-11},
            )
            flat_result = np.asarray(optimisation.x, dtype=np.float64)
            self.converged_ = bool(optimisation.success)
        except ImportError:
            flat_result = initial
            first_moment = np.zeros_like(flat_result)
            second_moment = np.zeros_like(flat_result)
            for iteration in range(1, self.max_iter + 1):
                _, gradient = objective(flat_result)
                first_moment = 0.9 * first_moment + 0.1 * gradient
                second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
                corrected_first = first_moment / (1.0 - 0.9**iteration)
                corrected_second = second_moment / (1.0 - 0.999**iteration)
                flat_result -= 0.03 * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            self.converged_ = True
        self.coef_ = flat_result.reshape(shape)
        return self


def fit_review_intervention_model(
    features: NDArray[np.generic],
    intervention: ReviewInterventionResult,
    *,
    class_order: Sequence[int],
    l2: float = 1.0e-2,
    max_iter: int = 400,
) -> SoftTargetMultinomialLogisticRegression:
    """Fit one derived model; source labels remain outside the mutation path."""

    classes = tuple(int(value) for value in class_order)
    if intervention.class_order != classes:
        raise ValueError("intervention targets differ from the requested class_order")
    model = SoftTargetMultinomialLogisticRegression(
        class_order=classes,
        l2=l2,
        max_iter=max_iter,
        class_weight_balanced=True,
    )
    return model.fit_soft_targets(
        features,
        intervention.soft_targets,
        sample_weight=intervention.training_weights,
    )


__all__ = ["SoftTargetMultinomialLogisticRegression", "fit_review_intervention_model"]
