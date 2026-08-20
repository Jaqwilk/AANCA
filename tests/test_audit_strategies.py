from __future__ import annotations

import numpy as np

from histo_audit.auditing.strategies import group_safe_audit_scores


def _inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
    dict[int, tuple[str, ...]],
    list[str],
]:
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [1.1, 0.0],
            [1.0, 0.1],
            [0.0, 1.1],
            [0.1, 1.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray(
        [[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.4, 0.6]] * 2,
        dtype=np.float64,
    )
    groups = ["g0", "g0", "g1", "g1", "g2", "g2", "g3", "g3"]
    folds = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    training = {0: ("g2", "g3"), 1: ("g0", "g1")}
    sample_ids = [f"sample-{index}" for index in range(len(labels))]
    return features, labels, probabilities, groups, folds, training, sample_ids


def test_probability_alias_preserves_existing_self_confidence() -> None:
    features, labels, probabilities, groups, folds, training, sample_ids = _inputs()

    result = group_safe_audit_scores(
        features,
        labels,
        probabilities,
        groups,
        folds,
        training,
        sample_ids=sample_ids,
        method="one_minus_probability_of_observed_label",
        class_order=(0, 1),
    )

    expected = 1.0 - probabilities[np.arange(len(labels)), labels]
    assert result.method == "self_confidence"
    assert np.array_equal(result.risk_scores, expected)
    assert result.neighbour_evidence is None


def test_neighbour_and_hybrid_strategies_keep_query_groups_out() -> None:
    features, labels, probabilities, groups, folds, training, sample_ids = _inputs()

    neighbour = group_safe_audit_scores(
        features,
        labels,
        probabilities,
        groups,
        folds,
        training,
        sample_ids=sample_ids,
        method="nearest_neighbour_disagreement",
        class_order=(0, 1),
        neighbour_k=2,
        neighbour_metric="cosine",
    )
    hybrid = group_safe_audit_scores(
        features,
        labels,
        probabilities,
        groups,
        folds,
        training,
        sample_ids=sample_ids,
        method="fixed_hybrid",
        class_order=(0, 1),
        neighbour_k=2,
        neighbour_metric="cosine",
    )

    assert neighbour.neighbour_evidence is not None
    for query_group, neighbour_groups in zip(
        groups, neighbour.neighbour_evidence.neighbour_groups, strict=True
    ):
        assert query_group not in neighbour_groups
    assert np.isfinite(neighbour.risk_scores).all()
    assert hybrid.method == "fixed_hybrid"
    assert hybrid.hybrid_weights == (0.5, 0.5)
    assert set(hybrid.component_scores) == {
        "self_confidence",
        "nearest_neighbour_disagreement",
    }
