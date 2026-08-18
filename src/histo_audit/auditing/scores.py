"""Consistent larger-is-more-suspicious annotation-risk scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from .ensemble import (
    EnsembleDisagreementResult as EnsembleDisagreementResult,
)
from .ensemble import (
    ensemble_disagreement as ensemble_disagreement,
)


@dataclass(frozen=True, slots=True)
class CleanlabScoreResult:
    """Cleanlab output or an explicit optional-dependency blocker."""

    available: bool
    quality_scores: NDArray[np.float64] | None
    risk_scores: NDArray[np.float64] | None
    issue_mask: NDArray[np.bool_] | None
    suggested_class: NDArray[np.int64] | None
    package_version: str | None
    api_path: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class FixedHybridDropOneResult:
    """Full frozen hybrid and one deterministic ablation per component.

    ``drop_one_scores`` is keyed by the omitted component and preserves the declared
    component order.  Remaining weights retain their frozen ratios and are renormalised
    to sum to one; no outcomes or performance metrics enter the calculation.
    """

    full_score: NDArray[np.float64]
    drop_one_scores: Mapping[str, NDArray[np.float64]]
    components: tuple[str, ...]
    normalised_weights: tuple[float, ...]


_METHOD_ALIASES = {
    "self_confidence": "self_confidence",
    "self-confidence": "self_confidence",
    "nll": "negative_log_likelihood",
    "negative_log_likelihood": "negative_log_likelihood",
    "margin": "prediction_margin",
    "prediction_margin": "prediction_margin",
    "entropy": "predictive_entropy",
    "predictive_entropy": "predictive_entropy",
}


def _validate_probabilities(probabilities: NDArray[np.generic]) -> NDArray[np.float64]:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2 or not matrix.shape[0]:
        raise ValueError("probabilities must have shape (n_samples, n_classes>=2)")
    if not np.isfinite(matrix).all():
        raise ValueError("probabilities contain non-finite values")
    if np.any(matrix < -1e-12) or np.any(matrix > 1.0 + 1e-12):
        raise ValueError("probabilities lie outside [0, 1]")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-7):
        raise ValueError("probability rows must sum to one")
    return matrix


def _observed_columns(
    observed_labels: Sequence[int] | NDArray[np.integer],
    n_samples: int,
    n_classes: int,
    class_order: Sequence[int] | None,
) -> NDArray[np.int64]:
    labels = np.asarray(observed_labels, dtype=np.int64)
    if labels.shape != (n_samples,):
        raise ValueError("observed labels must align with probabilities")
    classes = (
        tuple(int(value) for value in class_order)
        if class_order is not None
        else tuple(range(n_classes))
    )
    if len(classes) != n_classes or len(set(classes)) != n_classes:
        raise ValueError("class_order must align with unique probability columns")
    lookup = {label: column for column, label in enumerate(classes)}
    if any(int(label) not in lookup for label in labels):
        raise ValueError("observed label absent from class_order")
    return np.asarray([lookup[int(label)] for label in labels], dtype=np.int64)


def score_annotations(
    observed_labels: Sequence[int] | NDArray[np.integer],
    probabilities: NDArray[np.generic],
    *,
    method: str,
    class_order: Sequence[int] | None = None,
    epsilon: float = 1.0e-12,
) -> NDArray[np.float64]:
    """Score annotations; every supported method has larger=suspicious direction."""

    matrix = _validate_probabilities(probabilities)
    columns = _observed_columns(observed_labels, matrix.shape[0], matrix.shape[1], class_order)
    canonical = _METHOD_ALIASES.get(method)
    if canonical is None:
        raise ValueError(f"unsupported annotation-risk method: {method!r}")
    observed_probability = matrix[np.arange(len(matrix)), columns]
    if canonical == "self_confidence":
        risk = 1.0 - observed_probability
    elif canonical == "negative_log_likelihood":
        if not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must lie in (0, 1)")
        risk = -np.log(np.clip(observed_probability, epsilon, 1.0))
    elif canonical == "prediction_margin":
        alternatives = matrix.copy()
        alternatives[np.arange(len(matrix)), columns] = -np.inf
        risk = alternatives.max(axis=1) - observed_probability
    else:
        clipped = np.clip(matrix, epsilon, 1.0)
        risk = -np.sum(clipped * np.log(clipped), axis=1)
    risk = np.asarray(risk, dtype=np.float64)
    if not np.isfinite(risk).all():
        raise RuntimeError(f"{canonical} produced non-finite risk scores")
    return risk


def percentile_normalise(values: NDArray[np.generic]) -> NDArray[np.float64]:
    """Average-tie empirical percentile ranks in ``[0, 1]``."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("component scores must be a finite non-empty vector")
    if len(array) == 1:
        return np.asarray([0.5], dtype=np.float64)
    order = np.argsort(array, kind="stable")
    sorted_values = array[order]
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks / (len(array) - 1)


def fixed_hybrid_score(
    component_scores: Mapping[str, NDArray[np.generic]],
    *,
    components: Sequence[str] = (
        "self_confidence",
        "prediction_margin",
        "neighbour_disagreement",
    ),
    weights: Sequence[float] | None = None,
) -> NDArray[np.float64]:
    """Combine frozen complementary components after percentile normalisation."""

    names = tuple(components)
    if not names or len(set(names)) != len(names):
        raise ValueError("hybrid components must be a non-empty unique sequence")
    missing = [name for name in names if name not in component_scores]
    if missing:
        raise ValueError(f"missing hybrid score components: {missing}")
    normalised = [percentile_normalise(component_scores[name]) for name in names]
    lengths = {len(values) for values in normalised}
    if len(lengths) != 1:
        raise ValueError("hybrid score components have different lengths")
    if weights is None:
        weight_array = np.ones(len(names), dtype=np.float64)
    else:
        weight_array = np.asarray(tuple(weights), dtype=np.float64)
        if weight_array.shape != (len(names),):
            raise ValueError("hybrid weights must align with components")
    if not np.isfinite(weight_array).all() or np.any(weight_array < 0) or weight_array.sum() <= 0:
        raise ValueError("hybrid weights must be finite, non-negative, and not all zero")
    weight_array /= weight_array.sum()
    hybrid = np.average(np.stack(normalised), axis=0, weights=weight_array)
    if not np.isfinite(hybrid).all():
        raise RuntimeError("fixed hybrid produced non-finite values")
    return np.asarray(hybrid, dtype=np.float64)


def fixed_hybrid_drop_one_ablations(
    component_scores: Mapping[str, NDArray[np.generic]],
    *,
    components: Sequence[str],
    weights: Sequence[float],
) -> FixedHybridDropOneResult:
    """Build a frozen hybrid plus exactly one drop-one-component ablation each."""

    names = tuple(components)
    if len(names) < 2:
        raise ValueError("drop-one ablations require at least two hybrid components")
    if len(set(names)) != len(names):
        raise ValueError("hybrid components must be unique")
    missing = [name for name in names if name not in component_scores]
    if missing:
        raise ValueError(f"missing hybrid score components: {missing}")

    weight_array = np.asarray(tuple(weights), dtype=np.float64)
    if weight_array.shape != (len(names),):
        raise ValueError("hybrid weights must align with components")
    if not np.isfinite(weight_array).all() or np.any(weight_array < 0.0):
        raise ValueError("hybrid weights must be finite and non-negative")
    total_weight = float(weight_array.sum())
    if total_weight <= 0.0:
        raise ValueError("hybrid weights must not all be zero")

    remaining_totals = total_weight - weight_array
    if np.any(remaining_totals <= 0.0):
        dropped = names[int(np.flatnonzero(remaining_totals <= 0.0)[0])]
        raise ValueError(f"dropping component {dropped!r} leaves no positive frozen hybrid weight")

    full_score = fixed_hybrid_score(
        component_scores,
        components=names,
        weights=tuple(float(value) for value in weight_array),
    )
    full_score.setflags(write=False)

    ablations: dict[str, NDArray[np.float64]] = {}
    for dropped_index, dropped_name in enumerate(names):
        remaining_names = names[:dropped_index] + names[dropped_index + 1 :]
        remaining_weights = np.delete(weight_array, dropped_index)
        score = fixed_hybrid_score(
            component_scores,
            components=remaining_names,
            weights=tuple(float(value) for value in remaining_weights),
        )
        score.setflags(write=False)
        ablations[dropped_name] = score

    return FixedHybridDropOneResult(
        full_score=full_score,
        drop_one_scores=MappingProxyType(ablations),
        components=names,
        normalised_weights=tuple(float(value) for value in weight_array / total_weight),
    )


def cleanlab_scores(
    observed_labels: Sequence[int] | NDArray[np.integer],
    probabilities: NDArray[np.generic],
) -> CleanlabScoreResult:
    """Use stable Cleanlab APIs when installed, otherwise return an explicit blocker."""

    matrix = _validate_probabilities(probabilities)
    labels = np.asarray(observed_labels, dtype=np.int64)
    if labels.shape != (len(matrix),):
        raise ValueError("observed labels must align with probabilities")
    suggested = np.argmax(matrix, axis=1).astype(np.int64)
    try:
        from cleanlab.filter import find_label_issues  # type: ignore[import-untyped]
        from cleanlab.rank import get_label_quality_scores  # type: ignore[import-untyped]
    except (ImportError, ModuleNotFoundError) as error:
        return CleanlabScoreResult(
            available=False,
            quality_scores=None,
            risk_scores=None,
            issue_mask=None,
            suggested_class=None,
            package_version=None,
            api_path=None,
            error=f"Cleanlab unavailable: {error}",
        )
    try:
        quality = np.asarray(
            get_label_quality_scores(labels=labels, pred_probs=matrix), dtype=np.float64
        )
        issues = np.asarray(find_label_issues(labels=labels, pred_probs=matrix), dtype=bool)
        if quality.shape != labels.shape or issues.shape != labels.shape:
            raise RuntimeError("Cleanlab returned arrays with unexpected shapes")
        version = metadata.version("cleanlab")
        return CleanlabScoreResult(
            available=True,
            quality_scores=quality,
            risk_scores=1.0 - quality,
            issue_mask=issues,
            suggested_class=suggested,
            package_version=version,
            api_path="cleanlab.rank.get_label_quality_scores + cleanlab.filter.find_label_issues",
            error=None,
        )
    except Exception as error:  # Cleanlab API/version incompatibilities must stay explicit.
        return CleanlabScoreResult(
            available=False,
            quality_scores=None,
            risk_scores=None,
            issue_mask=None,
            suggested_class=None,
            package_version=(
                metadata.version("cleanlab")
                if metadata.packages_distributions().get("cleanlab")
                else None
            ),
            api_path="cleanlab.rank.get_label_quality_scores + cleanlab.filter.find_label_issues",
            error=f"Cleanlab integration failed: {type(error).__name__}: {error}",
        )
