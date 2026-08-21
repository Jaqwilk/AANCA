from __future__ import annotations

import json
from pathlib import Path

import yaml

from histo_audit.utils.run_tracking import sha256_file

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CONFIG_SHA256 = "00214940ee2cc1faf51202e23fe800a7f42a4f9b2936dae26bbfe20e3ee2555a"
EXPECTED_RESULTS_SHA256 = "0091c14075e7304e0a6effef5a398c32c692f40aaefe7ee7d0b126a101cf7892"
EXPECTED_ARRAYS_SHA256 = "186861575266985b4e1071190a957fd45769752e69e138633e94fa3be6eb4d39"
EXPECTED_SCENARIOS = (
    "clean_labels",
    "symmetric_1pct",
    "symmetric_2_5pct",
    "symmetric_5pct",
    "targeted_5pct",
    "targeted_10pct",
    "group_conditional_5pct",
    "group_conditional_10pct",
    "instance_geometry_5pct",
)


def _result() -> dict[str, object]:
    value = json.loads(
        (ROOT / "artifacts/puma_realism_stress/results.json").read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value


def test_realism_stress_configuration_is_frozen_and_cannot_select_candidate() -> None:
    path = ROOT / "configs/puma_realism_stress.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert sha256_file(path) == EXPECTED_CONFIG_SHA256
    assert config["replacement_project_or_v2"] is False
    assert config["primary_puma_result_opened_before_freeze"] is True
    assert config["candidate_selection_or_change_permitted"] is False
    assert tuple(item["name"] for item in config["scenarios"]) == EXPECTED_SCENARIOS
    assert config["claim_boundary"] == {
        "independent_confirmation": False,
        "natural_error_detection_proven": False,
        "pathologist_error_detection_proven": False,
        "clinical_utility_proven": False,
        "automatic_annotation_change_permitted": False,
    }


def test_realism_stress_artifacts_are_immutable_and_candidate_is_unchanged() -> None:
    result_path = ROOT / "artifacts/puma_realism_stress/results.json"
    arrays_path = ROOT / "artifacts/puma_realism_stress/evidence_arrays.npz"
    result = _result()

    assert sha256_file(result_path) == EXPECTED_RESULTS_SHA256
    assert sha256_file(arrays_path) == EXPECTED_ARRAYS_SHA256
    assert result["candidate_sha256"] == (
        "78547a73ef239dab11aee66e8b9b787e84508b82f6ace7bb81dc725f38803ffe"
    )
    assert result["candidate_changed"] is False
    assert result["replacement_project_or_v2"] is False
    assert result["primary_result_was_open_before_freeze"] is True
    assert result["source_annotations_modified"] is False
    assert result["natural_error_detection_evaluated"] is False
    assert result["pathologist_error_detection_proven"] is False


def test_realism_stress_preserves_positive_average_effect_and_adverse_class_result() -> None:
    result = _result()
    scenarios = result["scenarios"]
    assert isinstance(scenarios, list)
    assert tuple(item["name"] for item in scenarios) == EXPECTED_SCENARIOS

    passing = [item["name"] for item in scenarios if item["all_scenario_gates_passed"]]
    assert passing == ["group_conditional_10pct"]
    for item in scenarios:
        downstream = item["downstream"]
        gates = item["gates"]
        assert downstream["candidate_minus_uncorrected_macro_f1"] > 0.0
        assert downstream["candidate_minus_uncorrected_interval_95"][0] > 0.0
        assert downstream["candidate_minus_matched_random_interval_95"][0] > 0.0
        assert gates["all_models_converged"] is True
        failed = {name for name, value in gates.items() if value is False}
        if item["name"] == "group_conditional_10pct":
            assert not failed
        else:
            assert failed == {"every_class_recall_lower_bound_gte_minus_0_01"}

    clean = scenarios[0]
    assert clean["downstream"]["candidate_minus_uncorrected_recall"]["other"] == (
        -0.013732833957552981
    )
    assert clean["downstream"]["candidate_minus_uncorrected_recall_intervals_95"]["other"] == [
        -0.025389778615167228,
        -0.0027894707822914014,
    ]


def test_geometry_stress_records_independent_feature_spaces() -> None:
    result = _result()
    independence = result["geometry_auditor_independence"]
    geometry = result["scenarios"][-1]

    assert independence["matrix_decision"] == "verified_independent"
    assert independence["generator"]["family"] == "released_annotation_geometry"
    assert independence["auditor"]["family"] == "frozen_imagenet_resnet18_pixels"
    assert all(item["circularity_risk"] is False for item in geometry["per_seed"])
    assert all(
        item["independence_status"] == "verified_independent" for item in geometry["per_seed"]
    )
