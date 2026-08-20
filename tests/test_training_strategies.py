from __future__ import annotations

import numpy as np
import pytest

from histo_audit.evaluation.retraining_guard import INDEPENDENT_GROUP_VALIDATION
from histo_audit.evaluation.review_interventions import derive_review_interventions
from histo_audit.evaluation.training_strategies import compare_review_training_strategies


def _comparison_inputs() -> dict[str, object]:
    train_x = np.asarray(
        [[-3.0], [-2.5], [-2.0], [-1.5], [-1.0], [-0.5], [0.5], [1.0], [1.5], [2.0], [2.5], [3.0]]
    )
    observed = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int64)
    votes = np.full((3, len(observed)), -1, dtype=np.int64)
    votes[:, 6:10] = 1
    intervention = derive_review_interventions(
        observed,
        votes,
        [False] * 6 + [True] * 4 + [False] * 2,
        class_order=(0, 1),
        allow_hard_change=True,
    )
    validation_x = np.asarray([[-2.0], [0.5], [-1.5], [1.0], [-1.0], [1.5], [-0.5], [2.0]])
    validation_y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    return {
        "training_features": train_x,
        "observed_training_labels": observed,
        "intervention": intervention,
        "training_group_ids": [f"train-{index // 2}" for index in range(len(observed))],
        "validation_features": validation_x,
        "validation_reference_labels": validation_y,
        "validation_group_ids": [group for group in ("v1", "v2", "v3", "v4") for _ in range(2)],
        "class_order": (0, 1),
        "evidence_role": INDEPENDENT_GROUP_VALIDATION,
        "bootstrap_iterations": 100,
        "bootstrap_seed": 13,
    }


def test_strategy_comparison_uses_disjoint_development_validation_and_can_adopt() -> None:
    result = compare_review_training_strategies(**_comparison_inputs())

    assert result.final_external_test_used_for_selection is False
    assert result.source_annotations_modified is False
    assert len(result.candidates) == 4
    assert all(candidate.available for candidate in result.candidates)
    assert result.apply_review_intervention is True
    assert result.selected_strategy != "uncorrected_observed_labels"
    assert set(result.training_groups).isdisjoint(result.validation_groups)
    assert all(
        candidate.metrics is not None and 0.0 <= candidate.metrics.expected_calibration_error <= 1.0
        for candidate in result.candidates
    )


def test_strategy_comparison_rejects_group_overlap_and_nonindependent_evidence() -> None:
    inputs = _comparison_inputs()
    inputs["validation_group_ids"] = ["train-0"] * 2 + ["v2"] * 2 + ["v3"] * 2 + ["v4"] * 2
    with pytest.raises(ValueError, match="groups overlap"):
        compare_review_training_strategies(**inputs)

    inputs = _comparison_inputs()
    inputs["evidence_role"] = "final_test"
    with pytest.raises(ValueError, match="independent group validation"):
        compare_review_training_strategies(**inputs)
