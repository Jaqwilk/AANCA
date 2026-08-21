from __future__ import annotations

from pathlib import Path

import yaml

from histo_audit.auditing import (
    UTILITY_PRODUCT_PRIORITY,
    build_measured_utility_queue,
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
    assert policy["new_final_test"]["current_execution_status"] == (
        "controlled_new_source_complete"
    )
    assert policy["new_final_test"]["controlled_success_rule_met"] is True
    assert policy["new_final_test"]["natural_case_confirmation_status"] == "not_executed"
    assert policy["new_final_test"]["executable_action_on_natural_data"] == ("retain_uncorrected")
    assert policy["expanded_controlled_development"]["selected_candidate_sha256"] == (
        "78547a73ef239dab11aee66e8b9b787e84508b82f6ace7bb81dc725f38803ffe"
    )
    assert policy["expanded_controlled_development"]["selected_candidate_record_sha256"] == (
        "229bc293b3ba7c3909423178552f5f3789f00411223c2f87b5185eee1542487d"
    )
    assert policy["expanded_controlled_development"]["selected_policy"] == {
        "representation": "resnet18_multiscale_64_128",
        "audit_l2": 0.1,
        "audit_class_weight_balanced": False,
        "risk": "fixed_hybrid",
        "self_confidence_weight": 0.6,
        "neighbour_disagreement_weight": 0.4,
        "neighbour_k": 31,
        "queue": "balanced_relaxed",
        "review_budget": 0.05,
        "training_intervention": "flag_exclude",
        "downstream_l2": 0.01,
        "downstream_class_weight_balanced": True,
    }
    assert policy["review_queues"]["model_improvement"]["priority"] == (UTILITY_PRODUCT_PRIORITY)
    assert (
        policy["expanded_controlled_development"]["selected_result"][
            "development_intervals_are_not_independent_post_selection_confirmation"
        ]
        is True
    )
    assert policy["expanded_controlled_development"]["convergence_verification"] == {
        "artifact": "artifacts/autoresearch/monusac_aanca_expanded_convergence_v1.json",
        "artifact_sha256": ("d10fbcb3179abe6058ae43231663f3aeefc7d754c40bac2bfa6fdea1a4abae38"),
        "fit_count": 220,
        "hard_label_fit_count": 100,
        "weighted_fit_count": 120,
        "all_models_converged": True,
        "frozen_metrics_reproduced_exactly": True,
    }
    assert policy["puma_controlled_external_confirmation"]["all_seven_frozen_gates_passed"] is True
    assert (
        policy["puma_controlled_external_confirmation"]["controlled_noise_transfer_supported"]
        is True
    )
    assert (
        policy["puma_controlled_external_confirmation"]["natural_error_detection_evaluated"]
        is False
    )
    assert policy["puma_post_confirmation_stress"] == {
        "study_id": "puma_realism_stress_v1",
        "role": "post_confirmation_exploratory_stress_only",
        "candidate_changed": False,
        "scenario_count": 9,
        "scenarios_with_positive_macro_f1_lower_bound_vs_unchanged": 9,
        "scenarios_with_positive_macro_f1_lower_bound_vs_matched_random": 9,
        "scenarios_passing_every_registered_gate": 1,
        "passing_scenario": "group_conditional_10pct",
        "class_recall_safety_failures": 8,
        "clean_label_other_recall_effect": -0.013732833957552981,
        "clean_label_other_recall_interval_95": [
            -0.025389778615167228,
            -0.0027894707822914014,
        ],
        "result_sha256": ("0091c14075e7304e0a6effef5a398c32c692f40aaefe7ee7d0b126a101cf7892"),
        "evidence_arrays_sha256": (
            "186861575266985b4e1071190a957fd45769752e69e138633e94fa3be6eb4d39"
        ),
        "natural_data_policy_consequence": "retain_uncorrected_until_expert_review",
    }
    assert policy["puma_audit_time_label_sensitivity"] == {
        "study_id": "puma_audit_time_label_sensitivity_v1",
        "role": "post_confirmation_exploratory_sensitivity_only",
        "fold_assignment_label_source": "observed_label",
        "pre_corruption_label_used_for_fold_assignment": False,
        "candidate_changed": False,
        "retrieval_precision": 0.5381861575178998,
        "retrieval_precision_difference_vs_exact_matched_random": 0.3230310262529833,
        "retrieval_precision_difference_interval_95": [
            0.25973397585007,
            0.38131225104714983,
        ],
        "candidate_minus_unchanged_macro_f1": 0.00667944296150233,
        "candidate_minus_unchanged_interval_95": [
            0.004141475700661824,
            0.009505712132344529,
        ],
        "candidate_minus_exact_matched_random_macro_f1": 0.009069355900155174,
        "candidate_minus_exact_matched_random_interval_95": [
            0.005855100059258797,
            0.01246062687290453,
        ],
        "all_seven_sensitivity_gates_passed": True,
        "config_sha256": ("ed6fd1e85d15604efc331b634a0d7604ca2675ba58345aa31386c266781e661f"),
        "results_sha256": ("8f524b236995a495048a0955ebf930e14e732a1214d3856d3711571af13fd5cd"),
        "evidence_arrays_sha256": (
            "aad24975c29f004e6f5b44575ea5b3d97daa1122457277cca902d69f36e64903"
        ),
        "independent_confirmation": False,
        "natural_error_detection_evaluated": False,
    }
    assert policy["new_final_test"]["success_requires_all"] == [
        "top_k_beats_exact_matched_random_control",
        "intervention_model_beats_uncorrected_model",
        "no_important_class_breaches_registered_recall_non_degradation_limit",
    ]

    assert all(
        callable(value)
        for value in (
            build_two_review_queues,
            build_measured_utility_queue,
            cross_fitted_temperature_calibration,
            draw_matched_random_comparator,
            persistent_group_safe_risk,
            compare_review_training_strategies,
            derive_review_interventions,
            estimate_cross_fitted_downstream_utility,
            evaluate_multicriteria_retraining_guard,
        )
    )
