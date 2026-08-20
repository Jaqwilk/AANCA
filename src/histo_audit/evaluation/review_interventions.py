"""Derived training interventions from genuine multi-reviewer responses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

AmbiguousPolicy = Literal["soft_label", "downweight", "exclude"]
InsufficientContextPolicy = Literal["downweight", "exclude"]


@dataclass(frozen=True, slots=True)
class ReviewInterventionResult:
    """Immutable source labels plus derived hard, soft, weighted training views."""

    source_observed_labels: NDArray[np.int64]
    derived_hard_labels: NDArray[np.int64]
    soft_targets: NDArray[np.float64]
    training_weights: NDArray[np.float64]
    actions: tuple[str, ...]
    valid_vote_counts: NDArray[np.int64]
    majority_fractions: NDArray[np.float64]
    class_order: tuple[int, ...]
    hard_change_count: int
    soft_label_count: int
    downweight_count: int
    excluded_count: int
    source_annotations_modified: bool = False


def derive_review_interventions(
    observed_labels: Sequence[int] | NDArray[np.integer],
    reviewer_labels: NDArray[np.generic],
    reviewed_mask: Sequence[bool] | NDArray[np.bool_],
    *,
    class_order: Sequence[int],
    ambiguous_mask: Sequence[bool] | NDArray[np.bool_] | None = None,
    insufficient_context_mask: Sequence[bool] | NDArray[np.bool_] | None = None,
    technical_exclusion_mask: Sequence[bool] | NDArray[np.bool_] | None = None,
    missing_vote_value: int = -1,
    allow_hard_change: bool = False,
    minimum_hard_change_votes: int = 2,
    minimum_hard_change_fraction: float = 2.0 / 3.0,
    ambiguous_policy: AmbiguousPolicy = "soft_label",
    insufficient_context_policy: InsufficientContextPolicy = "exclude",
    uncertain_weight: float = 0.5,
) -> ReviewInterventionResult:
    """Create derived training targets without ever mutating observed labels.

    Hard changes require explicit opt-in plus the declared independent-vote gate.
    Disagreement defaults to a soft vote distribution; ambiguity and insufficient
    context may instead downweight or exclude the derived training row.
    """

    observed = np.asarray(observed_labels, dtype=np.int64).copy()
    votes = np.asarray(reviewer_labels)
    reviewed = np.asarray(reviewed_mask, dtype=bool)
    classes = tuple(int(value) for value in class_order)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("class_order must contain at least two unique labels")
    if observed.ndim != 1 or not len(observed) or reviewed.shape != observed.shape:
        raise ValueError("observed labels and reviewed mask must be non-empty and aligned")
    if votes.ndim != 2 or votes.shape[1] != len(observed) or votes.shape[0] < 1:
        raise ValueError("reviewer_labels must have shape (n_reviewers, n_samples)")
    if votes.dtype.kind not in {"i", "u"}:
        raise ValueError("reviewer labels must be integers")
    votes = votes.astype(np.int64, copy=True)
    allowed_votes = set(classes) | {int(missing_vote_value)}
    if any(int(value) not in allowed_votes for value in votes.ravel()):
        raise ValueError("reviewer label is outside class_order and missing-vote value")
    if any(int(value) not in classes for value in observed):
        raise ValueError("observed label is outside class_order")
    if np.any(votes[:, ~reviewed] != missing_vote_value):
        raise ValueError("unreviewed samples cannot contain reviewer label votes")
    if minimum_hard_change_votes < 2:
        raise ValueError("hard changes require at least two independent votes")
    if not 0.5 < minimum_hard_change_fraction <= 1.0:
        raise ValueError("minimum_hard_change_fraction must lie in (0.5, 1]")
    if not 0.0 <= uncertain_weight <= 1.0:
        raise ValueError("uncertain_weight must lie in [0, 1]")

    def flag(values: Sequence[bool] | NDArray[np.bool_] | None, role: str) -> NDArray[np.bool_]:
        result = (
            np.zeros(len(observed), dtype=bool)
            if values is None
            else np.asarray(values, dtype=bool)
        )
        if result.shape != observed.shape:
            raise ValueError(f"{role} mask must align with observed labels")
        if np.any(result & ~reviewed):
            raise ValueError(f"{role} flags cannot be attached to unreviewed samples")
        return result

    ambiguous = flag(ambiguous_mask, "ambiguous")
    insufficient = flag(insufficient_context_mask, "insufficient-context")
    technical = flag(technical_exclusion_mask, "technical-exclusion")
    lookup = {label: column for column, label in enumerate(classes)}
    hard = observed.copy()
    soft = np.zeros((len(observed), len(classes)), dtype=np.float64)
    soft[np.arange(len(observed)), [lookup[int(value)] for value in observed]] = 1.0
    weights = np.ones(len(observed), dtype=np.float64)
    actions = ["keep"] * len(observed)
    vote_counts = np.zeros(len(observed), dtype=np.int64)
    majority_fractions = np.zeros(len(observed), dtype=np.float64)

    for index in np.flatnonzero(reviewed):
        valid = votes[:, index]
        valid = valid[valid != missing_vote_value]
        vote_counts[index] = len(valid)
        distribution = np.zeros(len(classes), dtype=np.float64)
        if len(valid):
            for value in valid:
                distribution[lookup[int(value)]] += 1.0
            distribution /= distribution.sum()
            # Class identifiers are semantic values, not necessarily contiguous or
            # non-negative array offsets.  The vote distribution is already indexed
            # through ``class_order``, so derive the majority from that representation.
            majority_fractions[index] = float(distribution.max())

        if technical[index]:
            actions[index] = "exclude"
            weights[index] = 0.0
            continue
        if insufficient[index]:
            actions[index] = insufficient_context_policy
            weights[index] = 0.0 if insufficient_context_policy == "exclude" else uncertain_weight
            continue
        if ambiguous[index] or not len(valid):
            actions[index] = ambiguous_policy
            if ambiguous_policy == "exclude":
                weights[index] = 0.0
            elif ambiguous_policy == "downweight":
                weights[index] = uncertain_weight
            else:
                if len(valid):
                    soft[index] = distribution
                weights[index] = uncertain_weight
            continue

        top_columns = np.flatnonzero(distribution == distribution.max())
        has_unique_majority = len(top_columns) == 1
        majority_label = classes[int(top_columns[0])] if has_unique_majority else None
        unanimous = bool(distribution.max() == 1.0)
        hard_gate = (
            allow_hard_change
            and has_unique_majority
            and majority_label != int(observed[index])
            and len(valid) >= minimum_hard_change_votes
            and majority_fractions[index] >= minimum_hard_change_fraction
        )
        if hard_gate:
            assert majority_label is not None
            actions[index] = "hard_change"
            hard[index] = majority_label
            soft[index] = 0.0
            soft[index, lookup[majority_label]] = 1.0
        elif unanimous and majority_label == int(observed[index]):
            actions[index] = "keep"
        else:
            actions[index] = "soft_label"
            soft[index] = distribution
            weights[index] = uncertain_weight

    if not np.array_equal(observed, np.asarray(observed_labels, dtype=np.int64)):
        raise RuntimeError("source observed labels changed during intervention derivation")
    for array in (observed, hard, soft, weights, vote_counts, majority_fractions):
        array.setflags(write=False)
    return ReviewInterventionResult(
        source_observed_labels=observed,
        derived_hard_labels=hard,
        soft_targets=soft,
        training_weights=weights,
        actions=tuple(actions),
        valid_vote_counts=vote_counts,
        majority_fractions=majority_fractions,
        class_order=classes,
        hard_change_count=actions.count("hard_change"),
        soft_label_count=actions.count("soft_label"),
        downweight_count=actions.count("downweight"),
        excluded_count=actions.count("exclude"),
    )


__all__ = ["ReviewInterventionResult", "derive_review_interventions"]
