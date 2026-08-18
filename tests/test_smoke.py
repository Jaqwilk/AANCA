from __future__ import annotations

import json

import numpy as np
import pytest

from histo_audit.experiment.smoke import run_synthetic_smoke
from histo_audit.reporting.builder import build_synthetic_report


def test_synthetic_smoke_writes_machine_sourced_core_artifacts(tmp_path) -> None:
    result = run_synthetic_smoke(
        project_root=tmp_path,
        config={
            "n_groups": 12,
            "patch_size": 40,
            "oof_splits": 3,
            "random_review_repeats": 5,
            "downstream_random_repeats": 2,
            "bootstrap_iterations": 10,
        },
    )
    assert result.success
    assert result.status == "completed"
    assert result.run_dir is not None and result.run_dir.is_dir()
    for path in (
        result.metrics_path,
        result.predictions_path,
        result.rankings_path,
        result.corruption_manifest_path,
        result.oof_provenance_path,
        result.representation_example_path,
        result.neighbour_evidence_path,
        result.restoration_evidence_path,
        result.bootstrap_evidence_path,
        result.dataset_evidence_path,
        result.source_manifest_path,
        result.source_manifest_csv_path,
        result.report_inputs_path,
    ):
        assert path is not None and path.is_file()
    assert result.representation_example_path is not None
    with np.load(result.representation_example_path, allow_pickle=False) as example:
        assert example["full_patch"].shape[-1] == 3
        assert example["target_crop"].shape == (48, 48, 3)
        assert example["full_target_mask"].any()
        assert example["crop_target_mask"].any()
        assert str(example["sample_id"].item())
        assert int(example["target_instance_id"].item()) > 0
    saved_metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert saved_metrics["artifact_scope"] == "synthetic_software_validation"
    assert saved_metrics["oof"]["complete_once_coverage"] is True
    assert saved_metrics["corruption"]["exact_count"] > 0
    independence = saved_metrics["corruption"]["feature_space_independence"]
    assert independence == {
        "status": "not_applicable",
        "independent": None,
        "reason": "Feature-space independence applies only to instance-dependent corruption.",
    }
    assert saved_metrics["corruption"]["circularity_risk"] is False
    assert set(saved_metrics["random_review_by_budget"]) == {
        str(value) for value in saved_metrics["resolved_core_config"]["review_budgets"]
    }
    assert all(
        summary["repeats"] == 5 for summary in saved_metrics["random_review_by_budget"].values()
    )
    assert "feature_space_independent" not in saved_metrics["corruption"]
    if saved_metrics["cleanlab"]["available"]:
        assert "cleanlab" in saved_metrics["ranking"]
        cleanlab_budgets = saved_metrics["ranking"]["cleanlab"]["review_budgets"]
        assert cleanlab_budgets
        assert all(result["average_precision"] is not None for result in cleanlab_budgets.values())
        ranking_header = result.rankings_path.read_text(encoding="utf-8").splitlines()[0]
        assert ",cleanlab," in f",{ranking_header},"
    else:
        assert "cleanlab" not in saved_metrics["ranking"]
        assert saved_metrics["cleanlab"]["error"]
    provenance = json.loads(result.oof_provenance_path.read_text(encoding="utf-8"))
    assert saved_metrics["resolved_core_config"]["primary_representation"] == (
        "target_colour_statistics"
    )
    assert saved_metrics["resolved_core_config"]["auditor_representation"] == (
        "target_colour_statistics"
    )
    assert provenance["representation"] == "target_colour_statistics"
    assert provenance["folds"]
    assert all(not fold["group_overlap"] for fold in provenance["folds"])
    assert all(fold["training_groups"] for fold in provenance["folds"])
    with np.load(result.predictions_path) as predictions:
        assert predictions["probabilities"].shape[1] == 5
        np.testing.assert_allclose(predictions["probabilities"].sum(axis=1), 1.0)
        if saved_metrics["cleanlab"]["available"]:
            n = predictions["probabilities"].shape[0]
            assert predictions["cleanlab_quality_score"].shape == (n,)
            assert predictions["cleanlab_risk_score"].shape == (n,)
            assert predictions["cleanlab_issue_flag"].shape == (n,)
            assert predictions["cleanlab_suggested_class"].shape == (n,)
            assert saved_metrics["cleanlab"]["issue_count"] == int(
                predictions["cleanlab_issue_flag"].sum()
            )
        else:
            assert predictions["cleanlab_quality_score"].size == 0
            assert saved_metrics["cleanlab"]["issue_count"] is None
    assert result.neighbour_evidence_path is not None
    with np.load(result.neighbour_evidence_path, allow_pickle=False) as evidence:
        n = saved_metrics["sample_counts"]["audit_pool"]
        assert evidence["sample_ids"].shape == (n,)
        assert evidence["neighbour_ids"].shape[0] == n
        assert evidence["neighbour_groups"].shape == evidence["neighbour_ids"].shape
        assert evidence["neighbour_distances"].shape == evidence["neighbour_ids"].shape
        assert int(evidence["k"].item()) == 7
    assert result.restoration_evidence_path is not None
    with np.load(result.restoration_evidence_path, allow_pickle=False) as evidence:
        repeats = saved_metrics["resolved_core_config"]["downstream_random_repeats"]
        budget = saved_metrics["downstream_restoration"]["review_budget_count"]
        assert evidence["random_reviewed_indices"].shape == (repeats, budget)
        assert evidence["random_restored_label"].shape == (repeats, n)
        assert evidence["audit_guided_restored_label"].shape == (n,)
        random_probabilities = evidence["random_review_restoration_final_test_probabilities"]
        np.testing.assert_allclose(random_probabilities.sum(axis=-1), 1.0)
    assert result.bootstrap_evidence_path is not None
    with np.load(result.bootstrap_evidence_path, allow_pickle=False) as evidence:
        methods = tuple(str(value) for value in evidence["comparison_methods"])
        valid = int(evidence["valid_iterations"].item())
        assert str(evidence["status"].item()) == "reported"
        assert methods == tuple(saved_metrics["paired_method_differences"]["comparison_order"])
        assert evidence["metric_a"].shape == (len(methods), valid)
        assert evidence["metric_b"].shape == (len(methods), valid)
        assert evidence["differences"].shape == (len(methods), valid)
        np.testing.assert_allclose(
            evidence["differences"], evidence["metric_a"] - evidence["metric_b"]
        )
    assert result.dataset_evidence_path is not None
    with np.load(result.dataset_evidence_path, allow_pickle=False) as evidence:
        total = saved_metrics["sample_counts"]["total"]
        assert evidence["images"].shape[0] == total
        assert evidence["target_masks"].shape == evidence["images"].shape[:3]
        assert evidence["audit_features"].shape[0] == total
        assert evidence["corruption_features"].shape[0] == total
        final = evidence["split_partition"] == "final_reference_test"
        assert not evidence["is_injected_corruption"][final].any()
        np.testing.assert_array_equal(
            evidence["observed_label"][final], evidence["pre_corruption_label"][final]
        )


def test_nested_smoke_config_is_flattened_and_honoured() -> None:
    result = run_synthetic_smoke(
        config={
            "seed": {
                "dataset": 901,
                "split": 902,
                "model": 903,
                "corruption": 904,
                "random_review": 905,
                "bootstrap": 906,
            },
            "data": {
                "classes": 5,
                "groups": 15,
                "samples_per_group": 5,
                "image_size": 36,
                "final_test_fraction_groups": 0.2,
                "reference_validation_fraction_groups": 0.1,
            },
            "corruption": {"mechanism": "instance_dependent_corruption", "rate": 0.2},
            "model": {"oof_splits": 3},
            "audit": {"nearest_neighbours": 4},
            "evaluation": {
                "review_budgets": [0.05, 0.1],
                "random_review_repeats": 3,
                "bootstrap_iterations": 5,
            },
            "restoration": {"review_budget": 0.1},
        }
    )
    resolved = result.metrics["resolved_core_config"]
    assert resolved["dataset_seed"] == 901
    assert resolved["n_groups"] == 15
    assert resolved["instances_per_group"] == 5
    assert resolved["patch_size"] == 36
    assert resolved["final_test_fold"] == {
        "status": "not_applicable",
        "value": None,
        "reason": (
            "The synthetic final-reference partition is selected by a source-group fraction, "
            "not by an official fold identifier."
        ),
    }
    assert resolved["corruption_rate"] == 0.2
    assert resolved["oof_splits"] == 3
    assert resolved["neighbour_k"] == 4
    assert result.metrics["sample_counts"]["total"] == 75
    assert result.metrics["random_review"]["repeats"] == 3
    independence = result.metrics["corruption"]["feature_space_independence"]
    assert independence["status"] == "verified_independent"
    assert independence["independent"] is True
    assert len(independence["evidence"]["independence_matrix_hash"]) == 64
    assert independence["evidence"]["generator"]["feature_artifact_hash"]
    assert independence["evidence"]["auditor"]["implementation_hash"]
    assert independence["evidence"]["generator"]["family"] == "engineered_target_morphology"
    assert independence["evidence"]["auditor"]["family"] == "engineered_target_colour_only"
    assert "no morphology features are reused" in independence["reason"]
    assert result.metrics["corruption"]["circularity_risk"] is False
    assert result.metrics["paired_group_bootstrap_hybrid_minus_self_confidence"]["iterations"] == 5


def test_synthetic_smoke_rejects_misdeclared_representation() -> None:
    with pytest.raises(ValueError, match="supports only 'target_colour_statistics'"):
        run_synthetic_smoke(
            config={
                "representation": {
                    "primary": "engineered_target_features",
                    "auditor": "target_colour_statistics",
                }
            }
        )


def test_zero_corruption_smoke_is_reportable_without_fabricated_detection_metrics(
    tmp_path,
) -> None:
    result = run_synthetic_smoke(
        project_root=tmp_path,
        config={
            "n_groups": 12,
            "patch_size": 40,
            "oof_splits": 3,
            "corruption_rate": 0,
            "random_review_repeats": 3,
            "downstream_random_repeats": 2,
            "bootstrap_iterations": 5,
        },
    )
    assert result.metrics_path is not None
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["corruption"]["exact_count"] == 0
    for method in metrics["ranking"].values():
        distribution = method["score_distribution"]
        assert distribution["count"] == metrics["sample_counts"]["audit_pool"]
        for budget_result in method["review_budgets"].values():
            assert budget_result["status"] == "not_applicable"
            assert budget_result["average_precision"]["status"] == "not_applicable"
            assert budget_result["recall"]["value"] is None
            assert budget_result["lift_over_random"]["value"] is None
            assert budget_result["false_alert_count"] == budget_result["reviewed_count"]
    assert metrics["random_review"]["status"] == "not_applicable"
    assert metrics["random_review"]["mean_recall"]["value"] is None
    bootstrap = metrics["paired_group_bootstrap_hybrid_minus_self_confidence"]
    assert bootstrap["status"] == "not_applicable"
    assert bootstrap["mean_difference"]["value"] is None
    paired = metrics["paired_method_differences"]
    assert paired["status"] == "not_applicable"
    assert all(
        comparison["status"] == "not_applicable" for comparison in paired["comparisons"].values()
    )
    assert result.bootstrap_evidence_path is not None
    with np.load(result.bootstrap_evidence_path, allow_pickle=False) as evidence:
        assert str(evidence["status"].item()) == "not_applicable"
        assert str(evidence["reason"].item())
        assert int(evidence["valid_iterations"].item()) == 0
        assert evidence["draw_indices"].size == 0
        assert evidence["valid_draw_indices"].size == 0
        assert evidence["metric_a"].shape[1] == 0
        assert evidence["metric_b"].shape[1] == 0
        assert evidence["differences"].shape[1] == 0
    report = build_synthetic_report(
        result.metrics_path,
        output_directory=tmp_path / "zero_report",
        predictions_path=result.predictions_path,
        generate_figures=False,
    )
    assert report.markdown_path.is_file()
    assert report.html_path.is_file()
