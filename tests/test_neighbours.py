from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors as SklearnNearestNeighbors

import histo_audit.auditing.neighbours as neighbour_module
from histo_audit.auditing.neighbours import (
    NeighbourDisagreementResult,
    fold_safe_neighbour_disagreement,
)


def _oracle_distance(
    query: NDArray[np.float64], references: NDArray[np.float64], metric: str
) -> NDArray[np.float64]:
    if metric == "euclidean":
        return np.linalg.norm(references - query, axis=1)
    query_norm = max(float(np.linalg.norm(query)), 1e-12)
    reference_norm = np.maximum(np.linalg.norm(references, axis=1), 1e-12)
    similarity = (references @ query) / (reference_norm * query_norm)
    return np.clip(1.0 - similarity, 0.0, 2.0)


def _brute_force_oracle(
    embeddings: NDArray[np.float64],
    observed_labels: NDArray[np.int64],
    group_ids: Sequence[str],
    fold_ids: NDArray[np.int64],
    training_groups_by_fold: Mapping[int, Sequence[str]],
    sample_ids: Sequence[str],
    *,
    class_order: Sequence[int],
    k: int,
    metric: str,
) -> NeighbourDisagreementResult:
    matrix = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(observed_labels, dtype=np.int64)
    groups = np.asarray(group_ids, dtype=np.str_)
    identifiers = tuple(str(value) for value in sample_ids)
    classes = tuple(int(value) for value in class_order)
    lookup = {value: index for index, value in enumerate(classes)}
    risks = np.empty(len(labels), dtype=np.float64)
    alternatives = np.empty(len(labels), dtype=np.float64)
    suggested = np.empty(len(labels), dtype=np.int64)
    neighbour_ids: list[tuple[str, ...]] = [()] * len(labels)
    neighbour_distances: list[tuple[float, ...]] = [()] * len(labels)
    neighbour_groups: list[tuple[str, ...]] = [()] * len(labels)

    for fold in np.unique(fold_ids):
        allowed = tuple(str(value) for value in training_groups_by_fold[int(fold)])
        reference_indices = np.flatnonzero(np.isin(groups, allowed))
        query_indices = np.flatnonzero(fold_ids == fold)
        mean = matrix[reference_indices].mean(axis=0)
        scale = matrix[reference_indices].std(axis=0)
        scale[scale < 1e-12] = 1.0
        standardised = (matrix - mean) / scale
        for query_index in query_indices:
            candidates = reference_indices[
                (reference_indices != query_index)
                & (groups[reference_indices] != groups[query_index])
            ]
            distances = _oracle_distance(
                standardised[query_index], standardised[candidates], metric
            )
            order = np.lexsort(
                (np.asarray([identifiers[index] for index in candidates]), distances)
            )
            chosen = candidates[order[: min(k, len(candidates))]]
            chosen_distances = distances[order[: min(k, len(candidates))]]
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
    return NeighbourDisagreementResult(
        risk_scores=risks,
        alternative_class_support=alternatives,
        suggested_class=suggested,
        neighbour_ids=tuple(neighbour_ids),
        neighbour_distances=tuple(neighbour_distances),
        neighbour_groups=tuple(neighbour_groups),
        k=k,
        metric=metric,
    )


def _assert_same_result(
    actual: NeighbourDisagreementResult, expected: NeighbourDisagreementResult
) -> None:
    np.testing.assert_allclose(actual.risk_scores, expected.risk_scores, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(
        actual.alternative_class_support,
        expected.alternative_class_support,
        rtol=1e-13,
        atol=1e-15,
    )
    np.testing.assert_array_equal(actual.suggested_class, expected.suggested_class)
    assert actual.neighbour_ids == expected.neighbour_ids
    assert actual.neighbour_groups == expected.neighbour_groups
    for actual_row, expected_row in zip(
        actual.neighbour_distances, expected.neighbour_distances, strict=True
    ):
        np.testing.assert_allclose(actual_row, expected_row, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_fold_index_matches_brute_force_oracle_without_group_leakage(metric: str) -> None:
    rng = np.random.default_rng(912)
    group_ids = np.repeat(np.asarray([f"group_{index}" for index in range(9)]), 3)
    group_fold = {f"group_{index}": index % 3 for index in range(9)}
    fold_ids = np.asarray([group_fold[str(group)] for group in group_ids], dtype=np.int64)
    embeddings = rng.normal(size=(len(group_ids), 7))
    labels = np.asarray([index % 3 for index in range(len(group_ids))], dtype=np.int64)
    sample_ids = tuple(f"sample_{value:03d}" for value in rng.permutation(len(group_ids)))
    training_groups = {
        fold: tuple(group for group, held_out_fold in group_fold.items() if held_out_fold != fold)
        for fold in range(3)
    }
    arguments = {
        "sample_ids": sample_ids,
        "class_order": (0, 1, 2),
        "k": 5,
        "metric": metric,
    }

    actual = fold_safe_neighbour_disagreement(
        embeddings,
        labels,
        group_ids,
        fold_ids,
        training_groups,
        **arguments,
    )
    expected = _brute_force_oracle(
        embeddings,
        labels,
        group_ids,
        fold_ids,
        training_groups,
        **arguments,
    )
    _assert_same_result(actual, expected)

    id_to_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    for query_index, row in enumerate(actual.neighbour_ids):
        allowed = set(training_groups[int(fold_ids[query_index])])
        for neighbour_id in row:
            neighbour_index = id_to_index[neighbour_id]
            assert str(group_ids[neighbour_index]) in allowed
            assert int(fold_ids[neighbour_index]) != int(fold_ids[query_index])
            assert str(group_ids[neighbour_index]) != str(group_ids[query_index])


@pytest.mark.parametrize("metric, expected_distance", [("euclidean", 0.0), ("cosine", 1.0)])
def test_boundary_ties_use_sample_id_and_exclude_entire_query_group(
    metric: str, expected_distance: float
) -> None:
    group_ids = np.repeat(np.asarray([f"group_{index}" for index in range(6)]), 2)
    fold_ids = np.asarray([index % 2 for index in range(len(group_ids))], dtype=np.int64)
    embeddings = np.zeros((len(group_ids), 4), dtype=np.float64)
    labels = np.asarray([index % 3 for index in range(len(group_ids))], dtype=np.int64)
    sample_ids = tuple(f"sample_{value:02d}" for value in range(len(group_ids), 0, -1))
    all_groups = tuple(sorted(set(str(value) for value in group_ids)))
    training_groups = {0: all_groups, 1: all_groups}
    arguments = {
        "sample_ids": sample_ids,
        "class_order": (0, 1, 2),
        "k": 4,
        "metric": metric,
    }

    actual = fold_safe_neighbour_disagreement(
        embeddings,
        labels,
        group_ids,
        fold_ids,
        training_groups,
        **arguments,
    )
    expected = _brute_force_oracle(
        embeddings,
        labels,
        group_ids,
        fold_ids,
        training_groups,
        **arguments,
    )
    _assert_same_result(actual, expected)
    for query_group, ids, groups, distances in zip(
        group_ids,
        actual.neighbour_ids,
        actual.neighbour_groups,
        actual.neighbour_distances,
        strict=True,
    ):
        assert tuple(ids) == tuple(sorted(ids))
        assert str(query_group) not in groups
        np.testing.assert_allclose(distances, expected_distance, rtol=0.0, atol=0.0)


def test_fold_standardisation_does_not_fit_on_held_out_queries() -> None:
    embeddings = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [3.0, 1.0],
            [4.0, 0.0],
            [5.0, 1.0],
        ],
        dtype=np.float64,
    )
    groups = np.asarray(["held_a", "held_b", "train_a", "train_b", "train_c", "train_d"])
    folds = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int64)
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    sample_ids = tuple(f"sample_{index}" for index in range(len(labels)))
    training_groups = {
        0: ("train_a", "train_b", "train_c", "train_d"),
        1: ("held_a", "held_b"),
    }
    baseline = fold_safe_neighbour_disagreement(
        embeddings,
        labels,
        groups,
        folds,
        training_groups,
        sample_ids=sample_ids,
        class_order=(0, 1),
        k=2,
    )
    changed = embeddings.copy()
    changed[1] = np.asarray([1.0e12, -1.0e12])
    perturbed = fold_safe_neighbour_disagreement(
        changed,
        labels,
        groups,
        folds,
        training_groups,
        sample_ids=sample_ids,
        class_order=(0, 1),
        k=2,
    )
    assert perturbed.neighbour_ids[0] == baseline.neighbour_ids[0]
    np.testing.assert_allclose(
        perturbed.neighbour_distances[0], baseline.neighbour_distances[0], rtol=0.0, atol=0.0
    )


def test_search_fits_one_index_per_fold_and_batches_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_sizes: list[int] = []
    query_sizes: list[int] = []

    class RecordingNearestNeighbours:
        def __init__(self, **kwargs: object) -> None:
            self._index = SklearnNearestNeighbors(**kwargs)

        def fit(self, matrix: NDArray[np.float64]) -> RecordingNearestNeighbours:
            fit_sizes.append(len(matrix))
            self._index.fit(matrix)
            return self

        def kneighbors(
            self,
            matrix: NDArray[np.float64],
            *,
            n_neighbors: int,
            return_distance: bool,
        ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
            query_sizes.append(len(matrix))
            return self._index.kneighbors(
                matrix,
                n_neighbors=n_neighbors,
                return_distance=return_distance,
            )

    monkeypatch.setattr(neighbour_module, "NearestNeighbors", RecordingNearestNeighbours)
    embeddings = np.arange(36, dtype=np.float64).reshape(12, 3)
    groups = np.asarray([f"group_{index}" for index in range(12)])
    folds = np.asarray([index % 3 for index in range(12)], dtype=np.int64)
    training_groups = {
        fold: tuple(
            str(group) for group, item_fold in zip(groups, folds, strict=True) if item_fold != fold
        )
        for fold in range(3)
    }
    fold_safe_neighbour_disagreement(
        embeddings,
        np.asarray([index % 2 for index in range(12)], dtype=np.int64),
        groups,
        folds,
        training_groups,
        class_order=(0, 1),
        k=2,
    )
    assert fit_sizes == [8, 8, 8]
    assert query_sizes == [4, 4, 4]


def test_no_eligible_reference_fails_closed() -> None:
    with pytest.raises(ValueError, match="no fold-safe neighbours remain"):
        fold_safe_neighbour_disagreement(
            np.zeros((2, 2), dtype=np.float64),
            np.asarray([0, 1], dtype=np.int64),
            ("only_group", "only_group"),
            np.asarray([0, 0], dtype=np.int64),
            {0: ("only_group",)},
            sample_ids=("sample_a", "sample_b"),
            class_order=(0, 1),
        )
