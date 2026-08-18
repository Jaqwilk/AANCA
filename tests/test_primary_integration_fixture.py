"""End-to-end integration coverage for the non-eligible primary matrix fixture."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from histo_audit.experiment import primary_core as primary_core_module
from histo_audit.experiment.primary_completion import read_primary_filesystem_evidence
from histo_audit.experiment.primary_core import run_synthetic_primary_integration_fixture
from histo_audit.utils.run_tracking import verify_run_integrity


def test_synthetic_primary_fixture_executes_matrix_oof_bootstrap_and_restoration(
    tmp_path: Path,
) -> None:
    result = run_synthetic_primary_integration_fixture(project_root=tmp_path)

    integrity = verify_run_integrity(result.run_directory)
    assert integrity.valid
    assert integrity.registry_record_present
    assert result.matrix_cell_count == 8
    readback = read_primary_filesystem_evidence(
        primary_core_module._synthetic_fixture_plan(), result.run_directory
    )
    assert readback.reconciliation.passed
    assert readback.circularity_excluded_cell_ids == ()

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    completion = json.loads(result.completion_evidence_path.read_text(encoding="utf-8"))
    reconciliation = json.loads(result.reconciliation_path.read_text(encoding="utf-8"))
    assert metrics["artifact_scope"] == "synthetic_primary_orchestrator_integration_test"
    assert metrics["study_outcome_eligible"] is False
    assert metrics["completion_stage"] is None
    assert completion["completion_stage"] is None
    assert completion["study_outcome_eligible"] is False
    assert completion["circularity_excluded_cell_count"] == 0
    assert reconciliation["status"] == "passed"
    assert reconciliation["completed_required_cell_count"] == 8
    assert reconciliation["failed_cell_count"] == 0
    assert "not PanNuke evidence" in result.report_path.read_text(encoding="utf-8")

    with (result.run_directory / "cell_index.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert {row["status"] for row in rows} == {"completed"}
    assert all(row["representation_id"] and row["classifier_id"] for row in rows)
    execution_controls = json.loads(
        (result.run_directory / "execution_controls.json").read_text(encoding="utf-8")
    )
    assert execution_controls["source"] == "sealed_synthetic_primary_fixture_v1"
    assert execution_controls["frozen_config_schema_version"] == 1
    assert execution_controls["plan_schema_version"] == 1
    hashes_by_scenario: dict[str, set[str]] = {}
    fold_assignment_hashes: set[str] = set()
    for row in rows:
        scenario_id = row["scenario_id"]
        hashes_by_scenario.setdefault(scenario_id, set()).add(row["corruption_configuration_hash"])
        cell_directory = result.run_directory / "cells" / row["cell_id"]
        with np.load(cell_directory / "oof_predictions.npz", allow_pickle=False) as evidence:
            np.testing.assert_array_equal(
                evidence["coverage_count"], np.ones(len(evidence["sample_ids"]), dtype=np.int64)
            )
            np.testing.assert_allclose(evidence["probabilities"].sum(axis=1), 1.0, atol=1e-7)
            np.testing.assert_array_equal(
                evidence["fold_assignment_labels"], evidence["pre_corruption_label"]
            )
            assert evidence["fold_assignment_label_source"].tolist() == ["pre_corruption_label"]
            fold_assignment_hashes.add(str(evidence["fold_assignment_labels_sha256"][0]))
        provenance = json.loads(
            (cell_directory / "oof_provenance.json").read_text(encoding="utf-8")
        )
        assert provenance["fold_assignment_label_source"] == "pre_corruption_label"
        assert provenance["fold_assignment_labels_sha256"] in fold_assignment_hashes
        cell_metrics = json.loads((cell_directory / "metrics.json").read_text(encoding="utf-8"))
        assert cell_metrics["oof_group_overlap_count"] == 0
        assert cell_metrics["final_reference_group_overlap_count"] == 0
        assert cell_metrics["random_review"]["repeats"] == 100
        assert set(cell_metrics["random_review_by_budget"]) == {"0.05", "0.01", "0.1", "0.2"}
        assert cell_metrics["primary_ranking_method"] == "self_confidence"
        assert cell_metrics["primary_confirmatory_eligible"] is True
        manifest = json.loads(
            (cell_directory / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        assert {item["path"] for item in manifest["artifacts"]} == {
            "oof_predictions.npz",
            "risk_scores.npz",
            "ranking.csv",
            "metrics.json",
            "bootstrap_evidence.npz",
            "oof_provenance.json",
            "corruption_manifest.json",
            "neighbour_evidence.npz",
            "cleanlab_evidence.npz",
            "cleanlab_evidence.json",
            "independence_evidence.json",
        }
        corruption_manifest = json.loads(
            (cell_directory / "corruption_manifest.json").read_text(encoding="utf-8")
        )
        assert (
            corruption_manifest["shared_scenario_corruption_hash"]
            == row["corruption_configuration_hash"]
            == cell_metrics["corruption_configuration_hash"]
        )
        independence = json.loads(
            (cell_directory / "independence_evidence.json").read_text(encoding="utf-8")
        )
        assert independence["primary_confirmatory_eligible"] is True
        bootstrap = cell_metrics["paired_group_bootstrap_hybrid_minus_self_confidence"]
        assert bootstrap["iterations"] == 2000
        with np.load(cell_directory / "bootstrap_evidence.npz", allow_pickle=False) as evidence:
            assert len(evidence["draw_offsets"]) == 2001
            assert evidence["comparison_ids"].tolist() == ["hybrid_vs_self_confidence"]
            np.testing.assert_allclose(
                evidence["metric_hybrid"] - evidence["metric_self_confidence"],
                evidence["differences"],
            )
    assert all(len(values) == 1 for values in hashes_by_scenario.values())
    assert len(fold_assignment_hashes) == 1

    restoration = json.loads(
        (result.run_directory / "restoration.json").read_text(encoding="utf-8")
    )
    assert restoration["schema_version"] == 1
    assert restoration["execution_controls_binding_sha256"] == execution_controls["binding_sha256"]
    assert restoration["required_experiments"] == [
        "uncorrupted_reference_baseline",
        "corrupted_observed_baseline",
        "random_review_restoration",
        "audit_guided_restoration",
    ]
    downstream_comparison = restoration["downstream_comparisons"][0]
    assert downstream_comparison["comparison_id"] == "audit_guided_minus_random_macro_f1"
    assert downstream_comparison["direction"] == "method_a_minus_method_b"
    assert downstream_comparison["random_repetitions"] == 100
    assert downstream_comparison["status"] == "reported"
    partition = restoration["evaluation"]["partition_evidence"]
    assert partition["group_overlap_count"] == 0
    assert partition["final_test_uncorrupted_verified"] is True
    assert partition["reference_validation_in_training"] is True
    with np.load(result.run_directory / "restoration_evidence.npz", allow_pickle=False) as evidence:
        assert evidence["random_final_probabilities"].shape[0] == 100
        assert evidence["random_reviewed_masks"].shape[0] == 100
        assert evidence["random_restored_masks"].shape[0] == 100
        assert evidence["random_restored_labels"].shape[0] == 100
        assert evidence["random_final_predicted_class"].shape[0] == 100
        np.testing.assert_array_equal(
            evidence["audit_pre_corruption_labels"] != evidence["audit_observed_labels"],
            evidence["audit_is_injected_corruption"],
        )
        np.testing.assert_array_equal(
            evidence["guided_restored_mask"],
            evidence["guided_reviewed_mask"] & evidence["audit_is_injected_corruption"],
        )
        assert evidence["downstream_comparison_ids"].tolist() == [
            "audit_guided_minus_random_macro_f1"
        ]
        np.testing.assert_allclose(
            evidence["downstream_comparison_000_metric_a"]
            - evidence["downstream_comparison_000_metric_b"],
            evidence["downstream_comparison_000_differences"],
        )
        assert int(evidence["guided_restored_mask"].sum()) <= int(
            evidence["guided_reviewed_mask"].sum()
        )
    restoration_manifest = json.loads(
        (result.run_directory / "restoration_manifest.json").read_text(encoding="utf-8")
    )
    assert {item["path"] for item in restoration_manifest["artifacts"]} == {
        "restoration.json",
        "restoration_evidence.npz",
    }
