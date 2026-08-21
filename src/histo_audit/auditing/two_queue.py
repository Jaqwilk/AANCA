"""Separate annotation-quality review from downstream-improvement selection."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .scores import percentile_normalise

GROUP_SAFE_OOF_EVIDENCE = "group_safe_oof"
CROSS_FITTED_UTILITY_EVIDENCE = "cross_fitted_development_estimate"


@dataclass(frozen=True, slots=True)
class QueueConstraints:
    """Predeclared capacity, balance, and optional diversity constraints."""

    requested_count: int
    max_per_group: int | None = None
    max_per_class: int | None = None
    max_per_tissue: int | None = None
    max_per_transition: int | None = None
    minimum_cosine_distance: float | None = None

    def validate(self) -> None:
        if self.requested_count <= 0:
            raise ValueError("requested_count must be positive")
        limits = (
            self.max_per_group,
            self.max_per_class,
            self.max_per_tissue,
            self.max_per_transition,
        )
        if any(value is not None and value <= 0 for value in limits):
            raise ValueError("queue quota limits must be positive when supplied")
        if (
            self.minimum_cosine_distance is not None
            and not 0.0 <= self.minimum_cosine_distance <= 2.0
        ):
            raise ValueError("minimum_cosine_distance must lie in [0, 2]")


@dataclass(frozen=True, slots=True)
class BalancedReviewQueue:
    """One deterministic queue with explicit underfill and rejection evidence."""

    name: str
    available: bool
    unavailable_reason: str | None
    requested_count: int
    eligible_count: int
    selected_indices: NDArray[np.int64]
    selected_sample_ids: tuple[str, ...]
    selected_priority_scores: NDArray[np.float64]
    underfilled: bool
    rejection_counts: dict[str, int]
    group_counts: dict[str, int]
    class_counts: dict[str, int]
    tissue_counts: dict[str, int]
    transition_counts: dict[str, int]

    @property
    def selected_count(self) -> int:
        return len(self.selected_indices)


@dataclass(frozen=True, slots=True)
class TwoReviewQueues:
    """Independent quality-control and model-improvement review queues."""

    quality_control: BalancedReviewQueue
    model_improvement: BalancedReviewQueue
    annotation_inconsistency_scores: NDArray[np.float64]
    expected_downstream_gain: NDArray[np.float64] | None
    downstream_gain_lower_bound: NDArray[np.float64] | None
    annotation_evidence_role: str
    utility_evidence_role: str | None
    source_annotations_modified: bool = False


@dataclass(frozen=True, slots=True)
class MatchedRandomComparator:
    """One-to-one random comparator matched on predeclared exact strata."""

    available: bool
    unavailable_reason: str | None
    top_indices: NDArray[np.int64]
    top_sample_ids: tuple[str, ...]
    top_match_strata: tuple[str, ...]
    comparator_indices: NDArray[np.int64]
    comparator_sample_ids: tuple[str, ...]
    comparator_match_strata: tuple[str, ...]
    match_fields: tuple[str, ...]
    stratum_counts: dict[str, int]
    seed: int
    source_annotations_modified: bool = False

    def selection_plan_records(self) -> tuple[dict[str, str], ...]:
        """Return private plan records accepted by the blinded package builder."""

        if not self.available:
            raise RuntimeError("matched comparator is unavailable")
        records = [
            {
                "sample_id": sample_id,
                "selection_source": "top_ranked",
                "match_stratum": stratum,
            }
            for sample_id, stratum in zip(self.top_sample_ids, self.top_match_strata, strict=True)
        ]
        records.extend(
            {
                "sample_id": sample_id,
                "selection_source": "random",
                "match_stratum": stratum,
            }
            for sample_id, stratum in zip(
                self.comparator_sample_ids,
                self.comparator_match_strata,
                strict=True,
            )
        )
        return tuple(records)


def _values(
    supplied: Sequence[str | int] | None,
    n_samples: int,
    *,
    role: str,
) -> NDArray[np.str_] | None:
    if supplied is None:
        return None
    values = np.asarray(tuple(str(value) for value in supplied), dtype=np.str_)
    if values.shape != (n_samples,) or any(not value for value in values):
        raise ValueError(f"{role} values must be non-empty and align with samples")
    return values


def _cosine_ready(
    embeddings: NDArray[np.generic] | None,
    n_samples: int,
    threshold: float | None,
) -> NDArray[np.float64] | None:
    if threshold is None:
        return None
    if embeddings is None:
        raise ValueError("minimum_cosine_distance requires aligned embeddings")
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != n_samples or not np.isfinite(matrix).all():
        raise ValueError("queue embeddings must be a finite aligned matrix")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("queue diversity cannot use zero-norm embeddings")
    return matrix / norms[:, None]


def _unavailable_queue(
    name: str, constraints: QueueConstraints, reason: str
) -> BalancedReviewQueue:
    return BalancedReviewQueue(
        name=name,
        available=False,
        unavailable_reason=reason,
        requested_count=constraints.requested_count,
        eligible_count=0,
        selected_indices=np.empty(0, dtype=np.int64),
        selected_sample_ids=(),
        selected_priority_scores=np.empty(0, dtype=np.float64),
        underfilled=True,
        rejection_counts={},
        group_counts={},
        class_counts={},
        tissue_counts={},
        transition_counts={},
    )


def _select_queue(
    name: str,
    priority: NDArray[np.float64],
    eligible: NDArray[np.bool_],
    identifiers: tuple[str, ...],
    groups: NDArray[np.str_],
    classes: NDArray[np.str_],
    tissues: NDArray[np.str_] | None,
    transitions: NDArray[np.str_] | None,
    embeddings: NDArray[np.float64] | None,
    constraints: QueueConstraints,
) -> BalancedReviewQueue:
    constraints.validate()
    if constraints.max_per_tissue is not None and tissues is None:
        raise ValueError("max_per_tissue requires tissue values")
    if constraints.max_per_transition is not None and transitions is None:
        raise ValueError("max_per_transition requires proposed labels")
    order = np.lexsort((np.asarray(identifiers, dtype=np.str_), -priority))
    group_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    tissue_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    selected: list[int] = []
    for raw_index in order:
        index = int(raw_index)
        if not eligible[index]:
            continue
        group = str(groups[index])
        observed_class = str(classes[index])
        tissue = str(tissues[index]) if tissues is not None else ""
        transition = str(transitions[index]) if transitions is not None else ""
        if (
            constraints.max_per_group is not None
            and group_counts[group] >= constraints.max_per_group
        ):
            rejections["group_quota"] += 1
            continue
        if (
            constraints.max_per_class is not None
            and class_counts[observed_class] >= constraints.max_per_class
        ):
            rejections["class_quota"] += 1
            continue
        if (
            constraints.max_per_tissue is not None
            and tissue_counts[tissue] >= constraints.max_per_tissue
        ):
            rejections["tissue_quota"] += 1
            continue
        if (
            constraints.max_per_transition is not None
            and transition_counts[transition] >= constraints.max_per_transition
        ):
            rejections["transition_quota"] += 1
            continue
        if embeddings is not None and selected:
            distances = 1.0 - embeddings[selected] @ embeddings[index]
            assert constraints.minimum_cosine_distance is not None
            if np.any(distances < constraints.minimum_cosine_distance):
                rejections["embedding_similarity"] += 1
                continue
        selected.append(index)
        group_counts[group] += 1
        class_counts[observed_class] += 1
        if tissues is not None:
            tissue_counts[tissue] += 1
        if transitions is not None:
            transition_counts[transition] += 1
        if len(selected) == constraints.requested_count:
            break
    selected_array = np.asarray(selected, dtype=np.int64)
    return BalancedReviewQueue(
        name=name,
        available=True,
        unavailable_reason=None,
        requested_count=constraints.requested_count,
        eligible_count=int(eligible.sum()),
        selected_indices=selected_array,
        selected_sample_ids=tuple(identifiers[index] for index in selected),
        selected_priority_scores=np.asarray(priority[selected_array], dtype=np.float64),
        underfilled=len(selected) < constraints.requested_count,
        rejection_counts=dict(sorted(rejections.items())),
        group_counts=dict(sorted(group_counts.items())),
        class_counts=dict(sorted(class_counts.items())),
        tissue_counts=dict(sorted(tissue_counts.items())),
        transition_counts=dict(sorted(transition_counts.items())),
    )


def build_two_review_queues(
    annotation_inconsistency_scores: Sequence[float] | NDArray[np.floating],
    group_ids: Sequence[str],
    observed_labels: Sequence[int],
    sample_ids: Sequence[str],
    *,
    quality_constraints: QueueConstraints,
    model_constraints: QueueConstraints,
    annotation_evidence_role: str,
    expected_downstream_gain: Sequence[float] | NDArray[np.floating] | None = None,
    downstream_gain_lower_bound: Sequence[float] | NDArray[np.floating] | None = None,
    utility_evidence_role: str | None = None,
    proposed_labels: Sequence[int] | None = None,
    tissue_types: Sequence[str] | None = None,
    embeddings: NDArray[np.generic] | None = None,
    minimum_annotation_score: float = 0.0,
    minimum_downstream_gain: float = 0.0,
) -> TwoReviewQueues:
    """Construct two queues without treating inconsistency as downstream utility.

    The model-improvement queue fails closed unless per-sample utility estimates come
    from a declared cross-fitted development procedure. This function does not infer
    those effects from the final test and never edits a source label.
    """

    risks = np.asarray(annotation_inconsistency_scores, dtype=np.float64)
    n_samples = len(risks)
    identifiers = tuple(str(value) for value in sample_ids)
    groups = _values(group_ids, n_samples, role="group")
    classes = _values(observed_labels, n_samples, role="observed-label")
    tissues = _values(tissue_types, n_samples, role="tissue")
    proposals = _values(proposed_labels, n_samples, role="proposed-label")
    if (
        risks.ndim != 1
        or not n_samples
        or not np.isfinite(risks).all()
        or len(identifiers) != n_samples
        or len(set(identifiers)) != n_samples
        or any(not value for value in identifiers)
    ):
        raise ValueError("risk scores and unique sample IDs must be finite and aligned")
    if groups is None or classes is None:
        raise RuntimeError("required queue values were not materialised")
    if annotation_evidence_role != GROUP_SAFE_OOF_EVIDENCE:
        raise ValueError("quality-control ranking requires group-safe OOF evidence")
    if not np.isfinite(minimum_annotation_score) or not np.isfinite(minimum_downstream_gain):
        raise ValueError("queue thresholds must be finite")
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
    quality_embeddings = _cosine_ready(
        embeddings, n_samples, quality_constraints.minimum_cosine_distance
    )
    model_embeddings = _cosine_ready(
        embeddings, n_samples, model_constraints.minimum_cosine_distance
    )
    quality_priority = percentile_normalise(risks)
    quality = _select_queue(
        "annotation_quality_control",
        quality_priority,
        risks >= minimum_annotation_score,
        identifiers,
        groups,
        classes,
        tissues,
        transitions,
        quality_embeddings,
        quality_constraints,
    )

    supplied_utility = (
        expected_downstream_gain is not None or downstream_gain_lower_bound is not None
    )
    if supplied_utility and (
        expected_downstream_gain is None or downstream_gain_lower_bound is None
    ):
        raise ValueError("expected gain and its lower bound must be supplied together")
    if not supplied_utility:
        model = _unavailable_queue(
            "model_improvement",
            model_constraints,
            "no independently estimated per-sample downstream utility is available",
        )
        gain = None
        lower = None
    else:
        gain = np.asarray(expected_downstream_gain, dtype=np.float64)
        lower = np.asarray(downstream_gain_lower_bound, dtype=np.float64)
        if (
            gain.shape != risks.shape
            or lower.shape != risks.shape
            or not np.isfinite(gain).all()
            or not np.isfinite(lower).all()
            or np.any(lower > gain + 1e-12)
        ):
            raise ValueError(
                "downstream gain estimates and lower bounds must be finite and aligned"
            )
        if utility_evidence_role != CROSS_FITTED_UTILITY_EVIDENCE:
            model = _unavailable_queue(
                "model_improvement",
                model_constraints,
                "utility evidence is not a cross-fitted development estimate",
            )
        else:
            model_priority = 0.5 * percentile_normalise(risks) + 0.5 * percentile_normalise(lower)
            model_eligible = (risks >= minimum_annotation_score) & (lower > minimum_downstream_gain)
            model = _select_queue(
                "model_improvement",
                model_priority,
                model_eligible,
                identifiers,
                groups,
                classes,
                tissues,
                transitions,
                model_embeddings,
                model_constraints,
            )
    return TwoReviewQueues(
        quality_control=quality,
        model_improvement=model,
        annotation_inconsistency_scores=risks,
        expected_downstream_gain=gain,
        downstream_gain_lower_bound=lower,
        annotation_evidence_role=annotation_evidence_role,
        utility_evidence_role=utility_evidence_role,
    )


def draw_matched_random_comparator(
    top_indices: Sequence[int] | NDArray[np.integer],
    eligible_mask: Sequence[bool] | NDArray[np.bool_],
    sample_ids: Sequence[str],
    match_values: Mapping[str, Sequence[str | int]],
    *,
    seed: int,
) -> MatchedRandomComparator:
    """Draw an exact-stratum comparator without reusing a top-ranked case.

    Matching fields may include observed class, tissue, patient/WSI/source group, or
    another prospectively registered covariate.  If any stratum lacks enough cases,
    no partial comparator is returned; the design must change before review starts.
    """

    identifiers = tuple(str(value) for value in sample_ids)
    eligible = np.asarray(eligible_mask, dtype=bool)
    selected = np.asarray(top_indices, dtype=np.int64)
    n_samples = len(identifiers)
    if (
        not n_samples
        or eligible.shape != (n_samples,)
        or len(set(identifiers)) != n_samples
        or any(not value for value in identifiers)
        or selected.ndim != 1
        or not len(selected)
        or len(set(int(value) for value in selected)) != len(selected)
        or np.any(selected < 0)
        or np.any(selected >= n_samples)
        or not np.all(eligible[selected])
    ):
        raise ValueError("top indices, eligibility, and unique sample IDs must be valid")
    if not match_values:
        raise ValueError("at least one predeclared matching field is required")
    fields = tuple(sorted(str(name) for name in match_values))
    if len(set(fields)) != len(fields) or any(not field for field in fields):
        raise ValueError("matching field names must be unique and non-empty")
    values: dict[str, NDArray[np.str_]] = {}
    for field in fields:
        if field not in match_values:
            raise RuntimeError("normalised matching field cannot be resolved")
        vector = _values(match_values[field], n_samples, role=f"match-{field}")
        if vector is None:
            raise RuntimeError("matching vector was not materialised")
        values[field] = vector

    def stratum(index: int) -> tuple[str, ...]:
        return tuple(str(values[field][index]) for field in fields)

    def rendered_stratum(index: int) -> str:
        return json.dumps(stratum(index), ensure_ascii=True, separators=(",", ":"))

    top_by_stratum: dict[tuple[str, ...], list[int]] = {}
    for index in selected:
        top_by_stratum.setdefault(stratum(int(index)), []).append(int(index))
    selected_set = set(int(value) for value in selected)
    pool_by_stratum: dict[tuple[str, ...], list[int]] = {}
    for index in np.flatnonzero(eligible):
        integer = int(index)
        if integer not in selected_set:
            pool_by_stratum.setdefault(stratum(integer), []).append(integer)
    missing = {
        key: len(top_members) - len(pool_by_stratum.get(key, ()))
        for key, top_members in top_by_stratum.items()
        if len(pool_by_stratum.get(key, ())) < len(top_members)
    }
    rendered_counts = {
        json.dumps(key, ensure_ascii=True, separators=(",", ":")): len(members)
        for key, members in sorted(top_by_stratum.items())
    }
    if missing:
        reason = "; ".join(
            f"{json.dumps(key, ensure_ascii=True, separators=(',', ':'))} lacks {count}"
            for key, count in sorted(missing.items())
        )
        return MatchedRandomComparator(
            available=False,
            unavailable_reason=f"insufficient exact-stratum comparator pool: {reason}",
            top_indices=selected.copy(),
            top_sample_ids=tuple(identifiers[int(index)] for index in selected),
            top_match_strata=tuple(rendered_stratum(int(index)) for index in selected),
            comparator_indices=np.empty(0, dtype=np.int64),
            comparator_sample_ids=(),
            comparator_match_strata=(),
            match_fields=fields,
            stratum_counts=rendered_counts,
            seed=seed,
        )

    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for key, top_members in sorted(top_by_stratum.items()):
        pool = np.asarray(sorted(pool_by_stratum[key]), dtype=np.int64)
        picked = rng.choice(pool, size=len(top_members), replace=False)
        chosen.extend(int(value) for value in np.sort(picked))
    comparator = np.asarray(chosen, dtype=np.int64)
    return MatchedRandomComparator(
        available=True,
        unavailable_reason=None,
        top_indices=selected.copy(),
        top_sample_ids=tuple(identifiers[int(index)] for index in selected),
        top_match_strata=tuple(rendered_stratum(int(index)) for index in selected),
        comparator_indices=comparator,
        comparator_sample_ids=tuple(identifiers[index] for index in comparator),
        comparator_match_strata=tuple(rendered_stratum(int(index)) for index in comparator),
        match_fields=fields,
        stratum_counts=rendered_counts,
        seed=seed,
    )


__all__ = [
    "CROSS_FITTED_UTILITY_EVIDENCE",
    "GROUP_SAFE_OOF_EVIDENCE",
    "BalancedReviewQueue",
    "MatchedRandomComparator",
    "QueueConstraints",
    "TwoReviewQueues",
    "build_two_review_queues",
    "draw_matched_random_comparator",
]
