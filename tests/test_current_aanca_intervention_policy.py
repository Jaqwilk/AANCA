from __future__ import annotations

from pathlib import Path

import yaml

from histo_audit.auditing import (
    build_two_review_queues,
    cross_fitted_temperature_calibration,
    draw_matched_random_comparator,
    persistent_group_safe_risk,
)
from histo_audit.evaluation import (
    compare_review_training_strategies,
    derive_review_interventions,
    estimate_cross_fitted_downstream_utility,
    evaluate_multicriteria_retraining_guard,
)


def test_current_policy_is_one_project_fail_closed_and_excludes_nucls_tuning() -> None:
    root = Path(__file__).resolve().parents[1]
    policy = yaml.safe_load(
        (root / "configs/current_aanca_intervention_policy.yaml").read_text(encoding="utf-8")
    )

    assert policy["project"] == "AANCA"
    assert policy["replacement_project_or_v2"] is False
    assert policy["frozen_nucls_result"]["allowed_for_method_or_threshold_selection"] is False
    assert (
        policy["claim_boundary"]["call_score_probability_of_error_before_new_expert_calibration"]
        is False
    )
    assert (
        policy["review_queues"]["model_improvement"]["unavailable_without_measured_utility"] is True
    )
    assert policy["expert_review"]["hard_change"]["default_enabled"] is False
    assert (
        policy["development_training_comparison"]["final_external_test_used_for_selection"] is False
    )
    assert policy["new_final_test"]["current_execution_status"] == "not_executed"
    assert policy["new_final_test"]["success_requires_all"] == [
        "top_k_beats_exact_matched_random_control",
        "intervention_model_beats_uncorrected_model",
        "no_important_class_breaches_registered_recall_non_degradation_limit",
    ]

    assert all(
        callable(value)
        for value in (
            build_two_review_queues,
            cross_fitted_temperature_calibration,
            draw_matched_random_comparator,
            persistent_group_safe_risk,
            compare_review_training_strategies,
            derive_review_interventions,
            estimate_cross_fitted_downstream_utility,
            evaluate_multicriteria_retraining_guard,
        )
    )
