from __future__ import annotations

import numpy as np
import pytest

from histo_audit.auditing.two_queue import (
    CROSS_FITTED_UTILITY_EVIDENCE,
    GROUP_SAFE_OOF_EVIDENCE,
    QueueConstraints,
    build_two_review_queues,
)
from histo_audit.evaluation.downstream_utility import (
    MEASURED_DEVELOPMENT_INTERVENTION_UTILITY,
    estimate_cross_fitted_downstream_utility,
)


def _utility_inputs() -> dict[str, object]:
    values = np.linspace(-1.0, 1.0, 24)
    return {
        "features": np.column_stack([values, values**2]),
        "measured_downstream_gain": 0.04 + 0.08 * values,
        "group_ids": [f"p{index // 3}" for index in range(24)],
        "sample_ids": [f"s{index}" for index in range(24)],
        "target_evidence_role": MEASURED_DEVELOPMENT_INTERVENTION_UTILITY,
        "outer_splits": 4,
        "inner_splits": 3,
        "split_seed": 23,
        "alpha": 0.1,
    }


def test_utility_estimator_is_nested_group_cross_fitted_and_feeds_second_queue() -> None:
    inputs = _utility_inputs()
    result = estimate_cross_fitted_downstream_utility(**inputs)

    assert result.nested_group_cross_fitted is True
    assert result.final_external_test_used is False
    assert result.utility_evidence_role == CROSS_FITTED_UTILITY_EVIDENCE
    assert np.all(result.downstream_gain_lower_bound <= result.expected_downstream_gain)
    assert np.array_equal(np.sort(np.unique(result.fold_id)), np.arange(4))
    future, future_lower = result.frozen_estimator.predict(np.asarray([[1.2, 1.44]]))
    assert future.shape == future_lower.shape == (1,)
    assert future_lower[0] <= future[0]

    queues = build_two_review_queues(
        np.linspace(0.2, 0.9, 24),
        inputs["group_ids"],  # type: ignore[arg-type]
        [index % 2 for index in range(24)],
        inputs["sample_ids"],  # type: ignore[arg-type]
        quality_constraints=QueueConstraints(requested_count=4, max_per_group=1),
        model_constraints=QueueConstraints(requested_count=4, max_per_group=1),
        annotation_evidence_role=GROUP_SAFE_OOF_EVIDENCE,
        expected_downstream_gain=result.expected_downstream_gain,
        downstream_gain_lower_bound=result.downstream_gain_lower_bound,
        utility_evidence_role=result.utility_evidence_role,
    )
    assert queues.model_improvement.available is True
    assert all(
        result.downstream_gain_lower_bound[index] > 0.0
        for index in queues.model_improvement.selected_indices
    )


def test_utility_estimator_rejects_unmeasured_targets_and_infeasible_groups() -> None:
    inputs = _utility_inputs()
    inputs["target_evidence_role"] = "derived_from_final_test"
    with pytest.raises(ValueError, match="measured development interventions"):
        estimate_cross_fitted_downstream_utility(**inputs)

    inputs = _utility_inputs()
    inputs["group_ids"] = ["one"] * 24
    with pytest.raises(ValueError, match="number of groups"):
        estimate_cross_fitted_downstream_utility(**inputs)
