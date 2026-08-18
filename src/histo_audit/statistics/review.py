"""Fixed review budgets, random baselines, and paired group bootstrap."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from histo_audit.corruption.controlled import normalise_rate

MetricFunction = Callable[[NDArray[np.bool_], NDArray[np.float64]], float | None]


@dataclass(frozen=True, slots=True)
class ReviewBudgetResult:
    """Operational ranking result at one exact annotation-review budget."""

    budget_fraction: float
    total_examples: int
    reviewed_count: int
    injected_total: int
    injected_reviewed: int
    precision: float | None
    recall: float | None
    expected_random_recall: float | None
    lift_over_random: float | None
    average_precision: float | None
    reviewed_indices: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class RandomReviewSummary:
    """Repeated deterministic random review at the same exact budget."""

    budget_fraction: float
    reviewed_count: int
    seeds: tuple[int, ...]
    injected_reviewed: NDArray[np.int64]
    mean_precision: float | None
    mean_recall: float | None
    recall_interval_95: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    """Paired method difference under identical source-group resamples."""

    metric_name: str
    metric_a: NDArray[np.float64]
    metric_b: NDArray[np.float64]
    differences: NDArray[np.float64]
    mean_difference: float | None
    interval_95: tuple[float, float] | None
    probability_positive: float | None
    requested_iterations: int
    valid_iterations: int
    seed: int


@dataclass(frozen=True, slots=True)
class SubgroupRankingResult:
    """Count-gated AP for one explicitly named subgroup."""

    subgroup: str
    total_examples: int
    injected_corruptions: int
    average_precision: float | None
    status: str
    reason: str | None


def budget_count(n_samples: int, budget: float | int) -> int:
    """Use a ceiling so a positive operational budget reviews at least one sample."""

    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    fraction = normalise_rate(budget)
    if n_samples == 0 or fraction == 0.0:
        return 0
    return min(n_samples, int(np.ceil(n_samples * fraction)))


def _validate_ranking_inputs(
    is_injected_corruption: Sequence[bool] | NDArray[np.bool_],
    risk_scores: Sequence[float] | NDArray[np.floating],
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    injected = np.asarray(is_injected_corruption, dtype=bool)
    scores = np.asarray(risk_scores, dtype=np.float64)
    if injected.ndim != 1 or scores.shape != injected.shape or not len(scores):
        raise ValueError("injected flags and risk scores must be non-empty aligned vectors")
    if not np.isfinite(scores).all():
        raise ValueError("risk scores contain non-finite values")
    return injected, scores


def average_precision(
    is_injected_corruption: Sequence[bool] | NDArray[np.bool_],
    risk_scores: Sequence[float] | NDArray[np.floating],
) -> float | None:
    """Non-interpolated AP with score ties evaluated at a common threshold."""

    injected, scores = _validate_ranking_inputs(is_injected_corruption, risk_scores)
    positives = int(injected.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_injected = injected[order]
    cumulative_true = np.cumsum(sorted_injected)
    threshold_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    previous_recall = 0.0
    area = 0.0
    for end in threshold_ends:
        true_positive = int(cumulative_true[end])
        recall = true_positive / positives
        precision = true_positive / (int(end) + 1)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return float(area)


def binary_auroc(
    is_injected_corruption: Sequence[bool] | NDArray[np.bool_],
    risk_scores: Sequence[float] | NDArray[np.floating],
) -> float | None:
    """Tie-aware AUROC; return ``None`` when either binary class is absent."""

    injected, scores = _validate_ranking_inputs(is_injected_corruption, risk_scores)
    positives = int(injected.sum())
    negatives = len(injected) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[injected].sum())
    statistic = positive_rank_sum - positives * (positives + 1) / 2.0
    return statistic / (positives * negatives)


def subgroup_average_precision(
    is_injected_corruption: Sequence[bool] | NDArray[np.bool_],
    risk_scores: Sequence[float] | NDArray[np.floating],
    subgroup_values: Sequence[str | int],
    *,
    min_samples: int = 100,
    min_injected_corruptions: int = 10,
) -> tuple[SubgroupRankingResult, ...]:
    """Calculate AP only for subgroups satisfying the frozen support gate."""

    injected, scores = _validate_ranking_inputs(is_injected_corruption, risk_scores)
    values = np.asarray(tuple(str(value) for value in subgroup_values), dtype=np.str_)
    if values.shape != injected.shape or any(not value for value in values):
        raise ValueError("subgroup values must be non-empty and align with ranking arrays")
    results: list[SubgroupRankingResult] = []
    for subgroup in sorted(np.unique(values).tolist()):
        members = values == subgroup
        total = int(members.sum())
        positives = int(injected[members].sum())
        reportable = subgroup_is_reportable(
            total,
            positives,
            min_samples=min_samples,
            min_injected_corruptions=min_injected_corruptions,
        )
        results.append(
            SubgroupRankingResult(
                subgroup=str(subgroup),
                total_examples=total,
                injected_corruptions=positives,
                average_precision=(
                    average_precision(injected[members], scores[members]) if reportable else None
                ),
                status="reported" if reportable else "insufficient_support",
                reason=(
                    None
                    if reportable
                    else (
                        f"requires at least {min_samples} samples and "
                        f"{min_injected_corruptions} injected corruptions"
                    )
                ),
            )
        )
    return tuple(results)


def holm_adjust(p_values: Sequence[float]) -> NDArray[np.float64]:
    """Return Holm step-down family-wise-error adjusted p-values."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("p_values must be a finite non-empty vector")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("p_values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    scaled = (len(values) - np.arange(len(values))) * values[order]
    monotone = np.minimum(1.0, np.maximum.accumulate(scaled))
    adjusted = np.empty_like(monotone)
    adjusted[order] = monotone
    return adjusted


def rank_indices(
    risk_scores: Sequence[float] | NDArray[np.floating],
    *,
    tie_break_ids: Sequence[str] | None = None,
) -> NDArray[np.int64]:
    """Rank descending risk with a deterministic identifier/index tie break."""

    scores = np.asarray(risk_scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("risk scores must be a finite non-empty vector")
    if tie_break_ids is None:
        tie_values: NDArray[np.generic] = np.arange(len(scores), dtype=np.int64)
    else:
        if len(tie_break_ids) != len(scores):
            raise ValueError("tie-break IDs must align with scores")
        tie_values = np.asarray(tuple(str(value) for value in tie_break_ids), dtype=np.str_)
    return np.lexsort((tie_values, -scores)).astype(np.int64)


def evaluate_review_budget(
    is_injected_corruption: Sequence[bool] | NDArray[np.bool_],
    risk_scores: Sequence[float] | NDArray[np.floating],
    *,
    budget: float | int,
    tie_break_ids: Sequence[str] | None = None,
) -> ReviewBudgetResult:
    """Evaluate the top-ranked examples at a fixed operational review budget."""

    injected, scores = _validate_ranking_inputs(is_injected_corruption, risk_scores)
    fraction = normalise_rate(budget)
    reviewed_count = budget_count(len(scores), fraction)
    reviewed_indices = rank_indices(scores, tie_break_ids=tie_break_ids)[:reviewed_count]
    injected_total = int(injected.sum())
    found = int(injected[reviewed_indices].sum())
    precision = found / reviewed_count if reviewed_count else None
    if injected_total:
        recall = found / injected_total
        expected_random_recall = reviewed_count / len(scores)
        lift = recall / expected_random_recall if expected_random_recall else None
    else:
        recall = None
        expected_random_recall = None
        lift = None
    return ReviewBudgetResult(
        budget_fraction=fraction,
        total_examples=len(scores),
        reviewed_count=reviewed_count,
        injected_total=injected_total,
        injected_reviewed=found,
        precision=precision,
        recall=recall,
        expected_random_recall=expected_random_recall,
        lift_over_random=lift,
        average_precision=average_precision(injected, scores),
        reviewed_indices=reviewed_indices,
    )


def random_review_baseline(
    is_injected_corruption: Sequence[bool] | NDArray[np.bool_],
    *,
    budget: float | int,
    repeats: int = 100,
    seed: int = 101,
) -> RandomReviewSummary:
    """Repeat random review with the identical number reviewed and retained seeds."""

    injected = np.asarray(is_injected_corruption, dtype=bool)
    if injected.ndim != 1 or not len(injected):
        raise ValueError("injected flags must be a non-empty vector")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    fraction = normalise_rate(budget)
    reviewed_count = budget_count(len(injected), fraction)
    seeds = tuple(int(seed + repeat) for repeat in range(repeats))
    found = np.empty(repeats, dtype=np.int64)
    for repeat, repeat_seed in enumerate(seeds):
        rng = np.random.default_rng(repeat_seed)
        selected = rng.choice(len(injected), size=reviewed_count, replace=False)
        found[repeat] = int(injected[selected].sum())
    precision_values = found / reviewed_count if reviewed_count else None
    positives = int(injected.sum())
    recall_values = found / positives if positives else None
    return RandomReviewSummary(
        budget_fraction=fraction,
        reviewed_count=reviewed_count,
        seeds=seeds,
        injected_reviewed=found,
        mean_precision=(float(np.mean(precision_values)) if precision_values is not None else None),
        mean_recall=float(np.mean(recall_values)) if recall_values is not None else None,
        recall_interval_95=(
            (float(np.quantile(recall_values, 0.025)), float(np.quantile(recall_values, 0.975)))
            if recall_values is not None
            else None
        ),
    )


def draw_group_bootstrap_indices(
    group_ids: Sequence[str], *, n_iterations: int = 2_000, seed: int = 211
) -> tuple[NDArray[np.int64], ...]:
    """Draw groups with replacement and expand every draw to member indices."""

    groups = np.asarray(group_ids, dtype=np.str_)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("group IDs must be a non-empty vector")
    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive")
    unique_groups = np.unique(groups)
    members = {str(group): np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    draws: list[NDArray[np.int64]] = []
    for _ in range(n_iterations):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        draws.append(
            np.concatenate([members[str(group)] for group in sampled_groups]).astype(np.int64)
        )
    return tuple(draws)


def paired_group_bootstrap(
    is_injected_corruption: Sequence[bool] | NDArray[np.bool_],
    scores_a: Sequence[float] | NDArray[np.floating],
    scores_b: Sequence[float] | NDArray[np.floating],
    group_ids: Sequence[str],
    *,
    metric: MetricFunction = average_precision,
    metric_name: str = "average_precision",
    n_iterations: int = 2_000,
    seed: int = 211,
    bootstrap_indices: Sequence[NDArray[np.integer]] | None = None,
) -> PairedBootstrapResult:
    """Compare two methods on the exact same group-resampled observations."""

    injected, first = _validate_ranking_inputs(is_injected_corruption, scores_a)
    _, second = _validate_ranking_inputs(is_injected_corruption, scores_b)
    groups = np.asarray(group_ids, dtype=np.str_)
    if groups.shape != injected.shape:
        raise ValueError("group IDs must align with ranking arrays")
    draws = (
        tuple(np.asarray(indices, dtype=np.int64) for indices in bootstrap_indices)
        if bootstrap_indices is not None
        else draw_group_bootstrap_indices(
            tuple(str(value) for value in groups),
            n_iterations=n_iterations,
            seed=seed,
        )
    )
    if bootstrap_indices is not None:
        n_iterations = len(draws)
        if not n_iterations:
            raise ValueError("bootstrap_indices must not be empty")
    metric_a: list[float] = []
    metric_b: list[float] = []
    for indices in draws:
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(injected)):
            raise ValueError("invalid bootstrap observation indices")
        value_a = metric(injected[indices], first[indices])
        value_b = metric(injected[indices], second[indices])
        if value_a is None or value_b is None:
            continue
        if np.isfinite(value_a) and np.isfinite(value_b):
            metric_a.append(float(value_a))
            metric_b.append(float(value_b))
    first_values = np.asarray(metric_a, dtype=np.float64)
    second_values = np.asarray(metric_b, dtype=np.float64)
    differences = first_values - second_values
    if len(differences):
        summary_mean = float(differences.mean())
        interval = (
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        )
        probability = float(np.mean(differences > 0.0) + 0.5 * np.mean(differences == 0.0))
    else:
        summary_mean = None
        interval = None
        probability = None
    return PairedBootstrapResult(
        metric_name=metric_name,
        metric_a=first_values,
        metric_b=second_values,
        differences=differences,
        mean_difference=summary_mean,
        interval_95=interval,
        probability_positive=probability,
        requested_iterations=n_iterations,
        valid_iterations=len(differences),
        seed=seed,
    )


def subgroup_is_reportable(
    n_samples: int,
    n_injected_corruptions: int,
    *,
    min_samples: int = 100,
    min_injected_corruptions: int = 10,
) -> bool:
    """Apply the frozen minimum-count gate for subgroup average precision."""

    if min_samples < 0 or min_injected_corruptions < 0:
        raise ValueError("subgroup thresholds must be non-negative")
    return n_samples >= min_samples and n_injected_corruptions >= min_injected_corruptions
