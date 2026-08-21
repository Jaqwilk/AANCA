from __future__ import annotations

import json
from pathlib import Path

import yaml

from histo_audit.utils.run_tracking import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG_SHA256 = "ed6fd1e85d15604efc331b634a0d7604ca2675ba58345aa31386c266781e661f"
RESULT_SHA256 = "8f524b236995a495048a0955ebf930e14e732a1214d3856d3711571af13fd5cd"
ARRAYS_SHA256 = "aad24975c29f004e6f5b44575ea5b3d97daa1122457277cca902d69f36e64903"


def _result() -> dict[str, object]:
    value = json.loads(
        (ROOT / "artifacts/puma_audit_time_label_sensitivity/results.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(value, dict)
    return value


def test_audit_time_label_sensitivity_was_frozen_after_primary_without_tuning() -> None:
    path = ROOT / "configs/puma_audit_time_label_sensitivity.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert sha256_file(path) == CONFIG_SHA256
    assert config["replacement_project_or_v2"] is False
    assert config["primary_puma_result_opened_before_freeze"] is True
    assert config["candidate_selection_or_change_permitted"] is False
    assert config["fold_assignment"]["label_source"] == "observed_label"
    assert config["fold_assignment"]["pre_corruption_label_available"] is False
    assert not any(config["claim_boundary"].values())


def test_observed_label_sensitivity_artifacts_and_success_are_locked() -> None:
    result_path = ROOT / "artifacts/puma_audit_time_label_sensitivity/results.json"
    arrays_path = ROOT / "artifacts/puma_audit_time_label_sensitivity/evidence_arrays.npz"
    result = _result()

    assert sha256_file(result_path) == RESULT_SHA256
    assert sha256_file(arrays_path) == ARRAYS_SHA256
    assert result["candidate_sha256"] == (
        "78547a73ef239dab11aee66e8b9b787e84508b82f6ace7bb81dc725f38803ffe"
    )
    assert result["candidate_changed"] is False
    assert result["fold_assignment_label_source"] == "observed_label"
    assert result["pre_corruption_label_used_for_fold_assignment"] is False
    assert result["all_sensitivity_gates_passed"] is True
    assert all(result["success_conditions"].values())
    assert result["natural_error_detection_evaluated"] is False
    assert result["pathologist_error_detection_proven"] is False
    assert result["source_annotations_modified"] is False


def test_observed_label_folds_materially_change_without_erasing_the_effect() -> None:
    result = _result()
    per_seed = result["per_seed"]
    downstream = result["downstream"]
    retrieval = result["retrieval"]

    assert len(per_seed) == 4
    assert all(item["fold_assignment_label_source"] == "observed_label" for item in per_seed)
    assert max(item["fraction_matching_primary_reference_label_folds"] for item in per_seed) < 0.34
    assert retrieval["candidate_precision"] == 0.5381861575178998
    assert retrieval["interval_95"][0] > 0.0
    assert downstream["candidate_minus_uncorrected_macro_f1"] == 0.00667944296150233
    assert downstream["candidate_minus_uncorrected_interval_95"] == [
        0.004141475700661824,
        0.009505712132344529,
    ]
    assert downstream["candidate_minus_matched_random_macro_f1"] == 0.009069355900155174
    assert downstream["candidate_minus_matched_random_interval_95"][0] > 0.0
    assert all(
        interval[0] >= -0.01
        for interval in downstream["candidate_minus_uncorrected_recall_intervals_95"].values()
    )
