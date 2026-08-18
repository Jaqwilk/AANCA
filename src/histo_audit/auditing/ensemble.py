"""Deterministic ensemble-disagreement risks for annotation auditing.

All public score vectors in this module use the project-wide direction convention:
larger values mean a more potentially inconsistent annotation and therefore a higher
priority for expert review.  The functions consume already generated probability
matrices; they never fit models or inspect corruption outcomes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

PrimaryEnsembleRisk = Literal[
    "predictive_entropy_of_mean",
    "mean_pairwise_js_divergence",
    "variation_ratio",
    "observed_label_probability_variance",
    "predicted_class_disagreement",
]


@dataclass(frozen=True, slots=True)
class EnsembleDisagreementResult:
    """Auditable disagreement summaries from a fixed set of model predictions.

    Hard predictions use the first maximum-probability column.  Consequently, ties
    are resolved deterministically by the explicitly supplied ``class_order``.
    ``predicted_class_disagreement`` is the fraction of unordered model pairs whose
    hard predictions differ, whereas ``variation_ratio`` is one minus the modal vote
    fraction.  They are related but not identical when there are more than two models.
    """

    averaged_probabilities: NDArray[np.float64]
    entropy_of_mean: NDArray[np.float64]
    variation_ratio: NDArray[np.float64]
    observed_label_probability_variance: NDArray[np.float64]
    predicted_class_disagreement: NDArray[np.float64]
    mean_pairwise_js_divergence: NDArray[np.float64]
    class_order: tuple[int, ...]
    model_count: int

    @property
    def observed_probability_variance(self) -> NDArray[np.float64]:
        """Backward-compatible name for observed-label probability variance."""

        return self.observed_label_probability_variance


def _validated_class_order(class_order: Sequence[int], n_classes: int) -> tuple[int, ...]:
    values = tuple(class_order)
    if len(values) != n_classes:
        raise ValueError("class_order must contain one entry per probability column")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise ValueError("class_order entries must be integer class labels")
    classes = tuple(int(value) for value in values)
    if len(set(classes)) != len(classes):
        raise ValueError("class_order entries must be unique")
    return classes


def _validated_observed_columns(
    observed_labels: Sequence[int] | NDArray[np.integer],
    *,
    n_samples: int,
    class_order: tuple[int, ...],
) -> NDArray[np.int64]:
    raw = np.asarray(observed_labels)
    if raw.shape != (n_samples,):
        raise ValueError("observed_labels must have shape (n_samples,)")
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError("observed_labels must contain integer class labels")
    labels = np.asarray(raw, dtype=np.int64)
    lookup = {label: column for column, label in enumerate(class_order)}
    missing = sorted({int(label) for label in labels if int(label) not in lookup})
    if missing:
        raise ValueError(f"observed_labels contain classes absent from class_order: {missing}")
    return np.asarray([lookup[int(label)] for label in labels], dtype=np.int64)


def _validated_probability_tensor(
    model_probabilities: Sequence[NDArray[np.generic]] | NDArray[np.generic],
) -> NDArray[np.float64]:
    try:
        tensor = np.asarray(model_probabilities, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "model_probabilities must contain identically shaped numeric matrices"
        ) from error
    if tensor.ndim != 3:
        raise ValueError("model_probabilities must have shape (n_models, n_samples, n_classes)")
    n_models, n_samples, n_classes = tensor.shape
    if n_models < 2:
        raise ValueError("model_probabilities must contain at least two models")
    if n_samples < 1 or n_classes < 2:
        raise ValueError("probability matrices require at least one sample and two classes")
    if not np.isfinite(tensor).all():
        raise ValueError("model_probabilities contain non-finite values")
    if np.any(tensor < 0.0) or np.any(tensor > 1.0):
        raise ValueError("model_probabilities must lie within [0, 1]")
    if not np.allclose(tensor.sum(axis=2), 1.0, rtol=0.0, atol=1.0e-8):
        raise ValueError("every model probability row must sum to one")
    return tensor


def _entropy(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0.0
    terms[positive] = -probabilities[positive] * np.log(probabilities[positive])
    return np.asarray(terms.sum(axis=-1), dtype=np.float64)


def _mean_pairwise_js_divergence(
    tensor: NDArray[np.float64],
) -> NDArray[np.float64]:
    total = np.zeros(tensor.shape[1], dtype=np.float64)
    pair_count = 0
    for left_index in range(tensor.shape[0] - 1):
        left = tensor[left_index]
        for right_index in range(left_index + 1, tensor.shape[0]):
            right = tensor[right_index]
            midpoint = (left + right) / 2.0
            # H((P+Q)/2) - (H(P)+H(Q))/2 is stable at zero probability.
            total += _entropy(midpoint) - (_entropy(left) + _entropy(right)) / 2.0
            pair_count += 1
    result = total / pair_count
    # Cancellation can produce tiny excursions beyond the analytical [0, ln(2)] bound.
    return np.clip(result, 0.0, np.log(2.0)).astype(np.float64, copy=False)


def _readonly(array: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.asarray(array, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def ensemble_disagreement(
    model_probabilities: Sequence[NDArray[np.generic]] | NDArray[np.generic],
    *,
    observed_labels: Sequence[int] | NDArray[np.integer],
    class_order: Sequence[int],
) -> EnsembleDisagreementResult:
    """Compute fixed ensemble risks from two or more probability matrices.

    ``model_probabilities`` may be a sequence of matrices or one tensor with shape
    ``(n_models, n_samples, n_classes)``.  Model order does not affect any result.
    Population variance (``ddof=0``) is used for observed-label probabilities.
    Natural logarithms are used for entropy and Jensen-Shannon divergence.
    """

    tensor = _validated_probability_tensor(model_probabilities)
    classes = _validated_class_order(class_order, tensor.shape[2])
    observed_columns = _validated_observed_columns(
        observed_labels,
        n_samples=tensor.shape[1],
        class_order=classes,
    )

    averaged = tensor.mean(axis=0)
    entropy_of_mean = _entropy(averaged)

    # np.argmax deliberately implements the documented first-class tie rule.
    votes = np.argmax(tensor, axis=2)
    vote_counts = np.stack(
        [(votes == column).sum(axis=0) for column in range(tensor.shape[2])],
        axis=1,
    )
    variation_ratio = 1.0 - vote_counts.max(axis=1) / tensor.shape[0]

    agreeing_ordered_pairs = np.sum(vote_counts * (vote_counts - 1), axis=1)
    all_ordered_pairs = tensor.shape[0] * (tensor.shape[0] - 1)
    predicted_class_disagreement = 1.0 - agreeing_ordered_pairs / all_ordered_pairs

    sample_indices = np.arange(tensor.shape[1])
    observed_probabilities = tensor[:, sample_indices, observed_columns]
    observed_variance = observed_probabilities.var(axis=0, ddof=0)
    observed_variance[np.ptp(observed_probabilities, axis=0) == 0.0] = 0.0

    return EnsembleDisagreementResult(
        averaged_probabilities=_readonly(averaged),
        entropy_of_mean=_readonly(entropy_of_mean),
        variation_ratio=_readonly(variation_ratio),
        observed_label_probability_variance=_readonly(observed_variance),
        predicted_class_disagreement=_readonly(predicted_class_disagreement),
        mean_pairwise_js_divergence=_readonly(_mean_pairwise_js_divergence(tensor)),
        class_order=classes,
        model_count=tensor.shape[0],
    )


def predeclared_ensemble_risk(
    result: EnsembleDisagreementResult,
    *,
    primary_risk: PrimaryEnsembleRisk,
) -> NDArray[np.float64]:
    """Return one risk selected in the frozen design, without outcome inspection.

    This deliberately accepts no labels indicating injected corruption, metrics, or
    validation outcomes.  It is only a deterministic mapping from a predeclared name
    to a previously computed risk vector and therefore cannot tune or select a method.
    """

    risks: dict[str, NDArray[np.float64]] = {
        "predictive_entropy_of_mean": result.entropy_of_mean,
        "mean_pairwise_js_divergence": result.mean_pairwise_js_divergence,
        "variation_ratio": result.variation_ratio,
        "observed_label_probability_variance": result.observed_label_probability_variance,
        "predicted_class_disagreement": result.predicted_class_disagreement,
    }
    try:
        return _readonly(risks[primary_risk])
    except KeyError as error:
        raise ValueError(f"unsupported predeclared ensemble risk: {primary_risk!r}") from error
