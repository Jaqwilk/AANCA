from __future__ import annotations

from pathlib import Path

import yaml

from histo_audit.auditing import UTILITY_PRODUCT_PRIORITY
from histo_audit.utils.run_tracking import sha256_file


def _load(name: str) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    value = yaml.safe_load((root / "configs" / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_measured_utility_protocol_is_nested_product_ranked_and_fail_closed() -> None:
    config = _load("aanca_measured_utility_development.yaml")

    assert config["project"] == "AANCA"
    assert config["replacement_project_or_v2"] is False
    assert config["execution_status"] == "INITIALISED"
    assert config["evidence"]["synthetic_corruption_unlocks_natural_case_queue"] is False  # type: ignore[index]
    assert config["nested_design"]["outer_evaluation_unavailable_to_all_selection"] is True  # type: ignore[index]
    assert config["model_improvement_queue"]["priority"] == UTILITY_PRODUCT_PRIORITY  # type: ignore[index]
    assert config["model_improvement_queue"]["implementation"] == (  # type: ignore[index]
        "histo_audit.auditing.utility_queue.build_measured_utility_queue"
    )
    assert config["adoption_guard"]["failure_action"] == "retain_uncorrected"  # type: ignore[index]
    assert not any(config["claim_boundary"].values())  # type: ignore[union-attr]


def test_new_external_protocol_records_controlled_success_without_natural_claims() -> None:
    config = _load("aanca_new_external_confirmation.yaml")
    root = Path(__file__).resolve().parents[1]

    assert config["execution_status"] == "EXTERNAL_VALIDATION_COMPLETE"
    assert config["candidate_frozen"] is True
    assert config["candidate_sha256"] == (
        "78547a73ef239dab11aee66e8b9b787e84508b82f6ace7bb81dc725f38803ffe"
    )
    assert config["candidate_record_sha256"] == sha256_file(  # type: ignore[index]
        root / str(config["candidate_record"])
    )
    assert config["data"]["authorised_archive_present"] is True  # type: ignore[index]
    assert config["data"]["final_source_cohort_locked_before_outcomes"] is True  # type: ignore[index]
    assert config["expert_reference"]["required_for_natural_inconsistency_claim"] is True  # type: ignore[index]
    assert config["controlled_execution"] == {  # type: ignore[index]
        "study_id": "puma_new_data_confirmation_v1",
        "result": "supported",
        "final_case_groups": 62,
        "final_nuclei": 30397,
        "all_frozen_gates_passed": True,
        "independently_verified": True,
        "report": "reports/puma_new_data_confirmation_results.md",
        "natural_expert_reference_available": False,
        "prospective_real_workflow_executed": False,
    }
    claims = config["current_claims"]
    assert claims["controlled_noise_transfer_supported"] is True  # type: ignore[index]
    assert claims["executable_action"] == "retain_uncorrected"  # type: ignore[index]
    assert not any(
        value
        for name, value in claims.items()  # type: ignore[union-attr]
        if name.endswith("_proven")
    )
