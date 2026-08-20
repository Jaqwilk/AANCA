from __future__ import annotations

import numpy as np
import pytest

from histo_audit.auditing.two_queue import (
    CROSS_FITTED_UTILITY_EVIDENCE,
    GROUP_SAFE_OOF_EVIDENCE,
    QueueConstraints,
    build_two_review_queues,
    draw_matched_random_comparator,
)


def _inputs() -> dict[str, object]:
    return {
        "annotation_inconsistency_scores": np.asarray(
            [0.99, 0.98, 0.97, 0.80, 0.70, 0.60, 0.50, 0.40]
        ),
        "group_ids": ["p1", "p1", "p1", "p2", "p3", "p4", "p5", "p6"],
        "observed_labels": [0, 0, 0, 1, 1, 2, 2, 0],
        "sample_ids": [f"s{index}" for index in range(8)],
        "proposed_labels": [1, 1, 1, 0, 2, 0, 1, 2],
        "tissue_types": ["a", "a", "a", "a", "b", "b", "c", "c"],
        "embeddings": np.asarray(
            [
                [1.0, 0.0],
                [0.999, 0.001],
                [0.998, 0.002],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
                [0.7, 0.7],
                [-0.7, 0.7],
            ]
        ),
    }


def test_quality_queue_is_balanced_and_model_queue_fails_closed_without_utility() -> None:
    result = build_two_review_queues(
        **_inputs(),
        quality_constraints=QueueConstraints(
            requested_count=4,
            max_per_group=1,
            max_per_class=2,
            max_per_tissue=2,
            max_per_transition=1,
            minimum_cosine_distance=0.05,
        ),
        model_constraints=QueueConstraints(requested_count=3, max_per_group=1),
        annotation_evidence_role=GROUP_SAFE_OOF_EVIDENCE,
    )

    quality = result.quality_control
    assert quality.available is True
    assert quality.selected_count == 4
    assert max(quality.group_counts.values()) == 1
    assert max(quality.class_counts.values()) <= 2
    assert max(quality.tissue_counts.values()) <= 2
    assert max(quality.transition_counts.values()) == 1
    assert quality.rejection_counts["group_quota"] >= 1
    assert result.model_improvement.available is False
    assert "no independently estimated" in str(result.model_improvement.unavailable_reason)
    assert result.source_annotations_modified is False


def test_model_queue_requires_cross_fitted_positive_lower_bound() -> None:
    common = {
        **_inputs(),
        "quality_constraints": QueueConstraints(requested_count=2, max_per_group=1),
        "model_constraints": QueueConstraints(requested_count=3, max_per_group=1),
        "annotation_evidence_role": GROUP_SAFE_OOF_EVIDENCE,
        "expected_downstream_gain": np.asarray([0.04, 0.03, 0.02, 0.05, 0.04, 0.03, 0.02, 0.01]),
        "downstream_gain_lower_bound": np.asarray(
            [0.01, 0.01, -0.01, 0.03, 0.02, -0.02, 0.01, -0.01]
        ),
    }
    unavailable = build_two_review_queues(
        **common,
        utility_evidence_role="same_sample_estimate",
    )
    available = build_two_review_queues(
        **common,
        utility_evidence_role=CROSS_FITTED_UTILITY_EVIDENCE,
    )

    assert unavailable.model_improvement.available is False
    assert "not a cross-fitted" in str(unavailable.model_improvement.unavailable_reason)
    assert available.model_improvement.available is True
    assert available.model_improvement.selected_count == 3
    assert all(
        common["downstream_gain_lower_bound"][index] > 0.0  # type: ignore[index]
        for index in available.model_improvement.selected_indices
    )
    assert max(available.model_improvement.group_counts.values()) == 1


def test_quality_queue_rejects_non_group_safe_scores() -> None:
    with pytest.raises(ValueError, match="group-safe OOF"):
        build_two_review_queues(
            **_inputs(),
            quality_constraints=QueueConstraints(requested_count=2),
            model_constraints=QueueConstraints(requested_count=2),
            annotation_evidence_role="in_sample",
        )


def test_random_comparator_matches_every_predeclared_stratum_exactly() -> None:
    result = draw_matched_random_comparator(
        [0, 2, 4],
        [True] * 8,
        [f"s{index}" for index in range(8)],
        {
            "class": [0, 0, 1, 1, 0, 0, 1, 1],
            "tissue": ["a", "a", "a", "a", "b", "b", "b", "b"],
        },
        seed=31,
    )

    assert result.available is True
    assert len(result.comparator_indices) == 3
    assert set(result.top_indices).isdisjoint(result.comparator_indices)
    assert result.match_fields == ("class", "tissue")
    assert result.stratum_counts == {
        '["0","a"]': 1,
        '["0","b"]': 1,
        '["1","a"]': 1,
    }
    assert result.source_annotations_modified is False
    records = result.selection_plan_records()
    assert {record["selection_source"] for record in records} == {"top_ranked", "random"}
    assert len(records) == 6


def test_random_comparator_fails_closed_when_a_stratum_has_no_match() -> None:
    result = draw_matched_random_comparator(
        [0, 1],
        [True, True, True],
        ["s0", "s1", "s2"],
        {"class": [0, 0, 1]},
        seed=31,
    )

    assert result.available is False
    assert not len(result.comparator_indices)
    assert "insufficient exact-stratum" in str(result.unavailable_reason)
    with pytest.raises(RuntimeError, match="unavailable"):
        result.selection_plan_records()
