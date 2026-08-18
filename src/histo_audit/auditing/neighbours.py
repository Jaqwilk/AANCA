"""Fold-safe nearest-neighbour label disagreement."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class NeighbourDisagreementResult:
    """Weighted disagreement with auditable neighbour identities and distances."""

    risk_scores: NDArray[np.float64]
    alternative_class_support: NDArray[np.float64]
    suggested_class: NDArray[np.int64]
    neighbour_ids: tuple[tuple[str, ...], ...]
    neighbour_distances: tuple[tuple[float, ...], ...]
    neighbour_groups: tuple[tuple[str, ...], ...]
    k: int
    metric: str

    def validate(self, sample_ids: Sequence[str], group_ids: Sequence[str]) -> None:
        n = len(sample_ids)
        if self.risk_scores.shape != (n,) or self.alternative_class_support.shape != (n,):
            raise ValueError("neighbour score arrays do not align with samples")
        if (
            not np.isfinite(self.risk_scores).all()
            or not np.isfinite(self.alternative_class_support).all()
        ):
            raise ValueError("neighbour disagreement contains non-finite scores")
        id_to_group = {
            str(sample_id): str(group)
            for sample_id, group in zip(sample_ids, group_ids, strict=True)
        }
        for sample_id, group, neighbours, neighbour_groups in zip(
            sample_ids,
            group_ids,
            self.neighbour_ids,
            self.neighbour_groups,
            strict=True,
        ):
            if str(sample_id) in neighbours:
                raise ValueError("nearest-neighbour output contains the query itself")
            if str(group) in neighbour_groups:
                raise ValueError("nearest-neighbour output contains the query source group")
            if any(
                id_to_group[neighbour] != neighbour_group
                for neighbour, neighbour_group in zip(neighbours, neighbour_groups, strict=True)
            ):
                raise ValueError("neighbour group provenance mismatch")


def _distance(
    query: NDArray[np.float64], references: NDArray[np.float64], metric: str
) -> NDArray[np.float64]:
    if metric == "euclidean":
        return np.linalg.norm(references - query, axis=1)
    if metric == "cosine":
        query_norm = max(float(np.linalg.norm(query)), 1e-12)
        reference_norm = np.maximum(np.linalg.norm(references, axis=1), 1e-12)
        similarity = (references @ query) / (reference_norm * query_norm)
        return np.clip(1.0 - similarity, 0.0, 2.0)
    raise ValueError("metric must be 'euclidean' or 'cosine'")


def _fold_neighbours(
    reference_matrix: NDArray[np.float64],
    query_matrix: NDArray[np.float64],
    reference_indices: NDArray[np.int64],
    query_indices: NDArray[np.int64],
    groups: NDArray[np.str_],
    identifiers: tuple[str, ...],
    *,
    k: int,
    metric: str,
) -> tuple[tuple[NDArray[np.int64], NDArray[np.float64]], ...]:
    """Query one exact fold-level index while resolving boundary ties canonically."""

    reference_groups = groups[reference_indices]
    reference_group_counts = Counter(str(value) for value in reference_groups)
    eligible_counts = np.asarray(
        [
            len(reference_indices) - reference_group_counts.get(str(groups[index]), 0)
            for index in query_indices
        ],
        dtype=np.int64,
    )
    empty = np.flatnonzero(eligible_counts <= 0)
    if len(empty):
        query_index = int(query_indices[int(empty[0])])
        raise ValueError(f"no fold-safe neighbours remain for sample {identifiers[query_index]}")

    # At most all members of the query group can be filtered from a returned
    # row. Asking for one extra eligible item lets us prove that the kth
    # boundary is not tied in the usual case. Exact ties are expanded below.
    maximum_excluded = max(
        reference_group_counts.get(str(groups[index]), 0) for index in query_indices
    )
    requested = min(len(reference_indices), k + maximum_excluded + 1)
    index = NearestNeighbors(metric=metric, algorithm="brute", n_jobs=1)
    index.fit(reference_matrix)

    selected: list[tuple[NDArray[np.int64], NDArray[np.float64]] | None] = [None] * len(
        query_indices
    )
    pending = np.arange(len(query_indices), dtype=np.int64)
    while len(pending):
        _, local_candidate_rows = index.kneighbors(
            query_matrix[pending],
            n_neighbors=requested,
            return_distance=True,
        )
        retry: list[int] = []
        for result_row, query_position_value in enumerate(pending):
            query_position = int(query_position_value)
            query_index = int(query_indices[query_position])
            candidate_positions = np.asarray(local_candidate_rows[result_row], dtype=np.int64)
            candidate_indices = reference_indices[candidate_positions]
            eligible = (candidate_indices != query_index) & (
                groups[candidate_indices] != groups[query_index]
            )
            candidate_positions = candidate_positions[eligible]
            candidate_indices = candidate_indices[eligible]
            target_count = min(k, int(eligible_counts[query_position]))
            if len(candidate_indices) < target_count:
                if requested == len(reference_indices):
                    raise RuntimeError("nearest-neighbour index omitted eligible references")
                retry.append(query_position)
                continue

            exact_distances = _distance(
                query_matrix[query_position],
                reference_matrix[candidate_positions],
                metric,
            )
            stable_order = np.lexsort(
                (
                    np.asarray(
                        [identifiers[int(value)] for value in candidate_indices],
                        dtype=np.str_,
                    ),
                    exact_distances,
                )
            )
            candidate_indices = candidate_indices[stable_order]
            exact_distances = exact_distances[stable_order]

            boundary_resolved = requested == len(reference_indices)
            if len(candidate_indices) > target_count:
                boundary = float(exact_distances[target_count - 1])
                next_distance = float(exact_distances[target_count])
                boundary_resolved = boundary_resolved or (
                    next_distance > boundary
                    and not np.isclose(next_distance, boundary, rtol=1e-12, atol=1e-15)
                )
            if boundary_resolved:
                selected[query_position] = (
                    np.asarray(candidate_indices[:target_count], dtype=np.int64),
                    np.asarray(exact_distances[:target_count], dtype=np.float64),
                )
            else:
                retry.append(query_position)

        if not retry:
            break
        if requested == len(reference_indices):
            raise RuntimeError("nearest-neighbour boundary could not be resolved")
        requested = min(len(reference_indices), max(requested + 1, requested * 2))
        pending = np.asarray(retry, dtype=np.int64)

    if any(value is None for value in selected):
        raise RuntimeError("nearest-neighbour index did not cover every fold query")
    return tuple(value for value in selected if value is not None)


def fold_safe_neighbour_disagreement(
    embeddings: NDArray[np.generic],
    observed_labels: Sequence[int] | NDArray[np.integer],
    group_ids: Sequence[str],
    fold_ids: Sequence[int] | NDArray[np.integer],
    training_groups_by_fold: Mapping[int, Sequence[str]],
    *,
    sample_ids: Sequence[str] | None = None,
    class_order: Sequence[int] = (0, 1, 2, 3, 4),
    k: int = 7,
    metric: str = "euclidean",
) -> NeighbourDisagreementResult:
    """Score each held-out sample using only its fold's labelled training groups."""

    matrix = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(observed_labels, dtype=np.int64)
    groups = np.asarray(group_ids, dtype=np.str_)
    folds = np.asarray(fold_ids, dtype=np.int64)
    if matrix.ndim != 2 or labels.shape != (len(matrix),) or groups.shape != labels.shape:
        raise ValueError("embeddings, labels, and groups must be aligned")
    if folds.shape != labels.shape or not np.isfinite(matrix).all():
        raise ValueError("fold IDs must align and embeddings must be finite")
    if k <= 0:
        raise ValueError("k must be positive")
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be 'euclidean' or 'cosine'")
    identifiers = (
        tuple(str(value) for value in sample_ids)
        if sample_ids is not None
        else tuple(f"sample_{index:08d}" for index in range(len(labels)))
    )
    if len(identifiers) != len(labels) or len(set(identifiers)) != len(labels):
        raise ValueError("sample IDs must be aligned and unique")
    classes = tuple(int(value) for value in class_order)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("class_order must contain at least two unique values")
    lookup = {value: index for index, value in enumerate(classes)}
    if any(int(label) not in lookup for label in labels):
        raise ValueError("observed label absent from class_order")

    risks = np.empty(len(labels), dtype=np.float64)
    alternatives = np.empty(len(labels), dtype=np.float64)
    suggested = np.empty(len(labels), dtype=np.int64)
    neighbour_ids: list[tuple[str, ...]] = [()] * len(labels)
    neighbour_distances: list[tuple[float, ...]] = [()] * len(labels)
    neighbour_groups: list[tuple[str, ...]] = [()] * len(labels)
    for fold in np.unique(folds):
        if int(fold) not in training_groups_by_fold:
            raise ValueError(f"missing training-group provenance for fold {int(fold)}")
        allowed_groups = {str(value) for value in training_groups_by_fold[int(fold)]}
        reference_indices = np.flatnonzero(np.isin(groups, tuple(allowed_groups)))
        query_indices = np.flatnonzero(folds == fold)
        if not len(reference_indices):
            raise ValueError(f"fold {int(fold)} has no valid reference samples")
        reference_mean = matrix[reference_indices].mean(axis=0)
        reference_scale = matrix[reference_indices].std(axis=0)
        reference_scale[reference_scale < 1e-12] = 1.0
        standardised_references = (matrix[reference_indices] - reference_mean) / reference_scale
        standardised_queries = (matrix[query_indices] - reference_mean) / reference_scale
        fold_results = _fold_neighbours(
            standardised_references,
            standardised_queries,
            np.asarray(reference_indices, dtype=np.int64),
            np.asarray(query_indices, dtype=np.int64),
            groups,
            identifiers,
            k=k,
            metric=metric,
        )
        for query_index_value, (chosen, chosen_distances) in zip(
            query_indices, fold_results, strict=True
        ):
            query_index = int(query_index_value)
            weights = 1.0 / np.maximum(chosen_distances, 1e-8)
            support = np.zeros(len(classes), dtype=np.float64)
            for neighbour_index, weight in zip(chosen, weights, strict=True):
                support[lookup[int(labels[neighbour_index])]] += float(weight)
            support /= support.sum()
            observed_column = lookup[int(labels[query_index])]
            risks[query_index] = 1.0 - support[observed_column]
            alternative_support = support.copy()
            alternative_support[observed_column] = -np.inf
            alternatives[query_index] = float(alternative_support.max())
            suggested[query_index] = classes[int(np.argmax(support))]
            neighbour_ids[query_index] = tuple(identifiers[index] for index in chosen)
            neighbour_distances[query_index] = tuple(float(value) for value in chosen_distances)
            neighbour_groups[query_index] = tuple(str(groups[index]) for index in chosen)
    result = NeighbourDisagreementResult(
        risk_scores=risks,
        alternative_class_support=alternatives,
        suggested_class=suggested,
        neighbour_ids=tuple(neighbour_ids),
        neighbour_distances=tuple(neighbour_distances),
        neighbour_groups=tuple(neighbour_groups),
        k=k,
        metric=metric,
    )
    result.validate(identifiers, tuple(str(value) for value in groups))
    return result
