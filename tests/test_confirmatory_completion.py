"""Completion-stage safety tests for the frozen confirmatory matrix."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_study_contracts import complete_confirmatory_config

from histo_audit.experiment.confirmatory_completion import (
    REAL_CONFIRMATORY_ARTIFACT_SCOPE,
    SYNTHETIC_CONFIRMATORY_ARTIFACT_SCOPE,
    build_confirmatory_completion_evidence,
    reconcile_confirmatory_cell_outcomes,
)
from histo_audit.experiment.study_contracts import build_confirmatory_matrix_plan
from histo_audit.workflows.study_gates import (
    ConfirmatoryExecutionGateEvidence,
    PrimaryExecutionGateEvidence,
)


def _successful_outcomes() -> tuple[Any, list[dict[str, Any]]]:
    plan = build_confirmatory_matrix_plan(complete_confirmatory_config())
    outcomes: list[dict[str, Any]] = []
    for cell in plan.cells:
        if cell.required:
            outcomes.append(
                {
                    "cell_id": cell.cell_id,
                    "required": True,
                    "status": "completed",
                    "outer_fold": cell.outer_fold,
                    "model_seed": cell.model_seed,
                    "scenario_id": cell.scenario_id,
                    "corruption_cell_id": cell.corruption_cell_id,
                    "artifact_manifest_sha256": "a" * 64,
                    "metrics_sha256": "b" * 64,
                }
            )
        else:
            outcomes.append(
                {
                    "cell_id": cell.cell_id,
                    "required": False,
                    "status": "skipped_with_frozen_blocker",
                    "frozen_unavailability": True,
                    "blocker": "pathology encoder unavailable under frozen priority rule",
                }
            )
    return plan, outcomes


def _gate(plan: Any) -> ConfirmatoryExecutionGateEvidence:
    primary_gate = PrimaryExecutionGateEvidence(
        freeze_directory=Path("freeze"),
        base_freeze_directory=Path("freeze"),
        freeze_artifact_root_sha256="1" * 64,
        freeze_manifest_sha256="2" * 64,
        preregistration_sha256="3" * 64,
        frozen_primary_config_sha256="4" * 64,
        frozen_confirmatory_config_sha256="5" * 64,
        primary_config_semantic_sha256="6" * 64,
        confirmatory_config_semantic_sha256=plan.config_sha256,
        primary_matrix_cell_count=216,
        primary_required_cell_count=180,
        confirmatory_matrix_cell_count=len(plan.cells),
        pilot_run_id="real-pannuke-pilot",
        pilot_artifact_root_sha256="7" * 64,
        dataset_sha256="8" * 64,
        manifest_sha256="9" * 64,
        duplicate_audit_sha256="a" * 64,
        pathology_encoder_audit_sha256="b" * 64,
        source_tree_root_sha256="c" * 64,
    )
    return ConfirmatoryExecutionGateEvidence(
        primary_gate=primary_gate,
        primary_run_directory=Path("primary-run"),
        primary_run_id="real-pannuke-primary",
        primary_artifact_root_sha256="d" * 64,
        primary_completion_evidence_sha256="e" * 64,
        primary_reconciliation_sha256="f" * 64,
        completed_required_cell_count=180,
    )


def test_real_confirmatory_stage_requires_filesystem_readback() -> None:
    plan, outcomes = _successful_outcomes()
    reconciliation = reconcile_confirmatory_cell_outcomes(plan, outcomes)
    assert reconciliation.passed
    assert reconciliation.fold_rotation_complete
    assert reconciliation.completed_outer_folds == (1, 2, 3)
    assert reconciliation.completed_required_cell_count == 90
    assert reconciliation.skipped_optional_cell_count == 18
    with pytest.raises(ValueError, match="filesystem-backed run directory"):
        build_confirmatory_completion_evidence(
            plan=plan,
            reconciliation=reconciliation,
            artifact_scope=REAL_CONFIRMATORY_ARTIFACT_SCOPE,
            study_outcome_eligible=True,
            gate_evidence=_gate(plan),
        )


def test_synthetic_fixture_never_enables_confirmatory_completion() -> None:
    plan, outcomes = _successful_outcomes()
    reconciliation = reconcile_confirmatory_cell_outcomes(plan, outcomes)
    evidence = build_confirmatory_completion_evidence(
        plan=plan,
        reconciliation=reconciliation,
        artifact_scope=SYNTHETIC_CONFIRMATORY_ARTIFACT_SCOPE,
        study_outcome_eligible=False,
    )

    assert evidence["completion_stage"] is None
    assert evidence["study_outcome_eligible"] is False
    with pytest.raises(ValueError, match="synthetic confirmatory fixtures"):
        build_confirmatory_completion_evidence(
            plan=plan,
            reconciliation=reconciliation,
            artifact_scope=SYNTHETIC_CONFIRMATORY_ARTIFACT_SCOPE,
            study_outcome_eligible=True,
            gate_evidence=_gate(plan),
        )


def test_missing_rotation_cells_separately_marks_fold_rotation_incomplete() -> None:
    plan, outcomes = _successful_outcomes()
    outcomes = [item for item in outcomes if item.get("outer_fold") != 3]
    # Optional skipped outcomes do not repeat identity, so remove them by their plan cell.
    fold_three_ids = {cell.cell_id for cell in plan.cells if cell.outer_fold == 3}
    outcomes = [item for item in outcomes if item["cell_id"] not in fold_three_ids]

    reconciliation = reconcile_confirmatory_cell_outcomes(plan, outcomes)

    assert not reconciliation.passed
    assert not reconciliation.fold_rotation_complete
    assert reconciliation.completed_outer_folds == (1, 2)
    assert reconciliation.incomplete_outer_folds == (3,)
    with pytest.raises(ValueError, match="passed matrix reconciliation"):
        build_confirmatory_completion_evidence(
            plan=plan,
            reconciliation=reconciliation,
            artifact_scope=REAL_CONFIRMATORY_ARTIFACT_SCOPE,
            study_outcome_eligible=True,
            gate_evidence=_gate(plan),
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("outer_fold", 99),
        ("model_seed", 999),
        ("scenario_id", "wrong_scenario"),
        ("corruption_cell_id", "wrong_corruption"),
        ("metrics_sha256", "not-a-sha"),
        ("artifact_manifest_sha256", None),
    ],
)
def test_completed_cell_requires_exact_plan_identity_and_hashes(
    field: str, invalid_value: Any
) -> None:
    plan, outcomes = _successful_outcomes()
    completed = next(item for item in outcomes if item["status"] == "completed")
    completed[field] = invalid_value

    reconciliation = reconcile_confirmatory_cell_outcomes(plan, outcomes)

    assert not reconciliation.passed
    assert not reconciliation.fold_rotation_complete
    assert completed["cell_id"] in reconciliation.invalid_cell_ids


def test_required_skip_optional_unfrozen_skip_and_failure_block_completion() -> None:
    plan, outcomes = _successful_outcomes()
    required = next(item for item in outcomes if item["required"])
    required.update(
        status="skipped_with_frozen_blocker",
        frozen_unavailability=True,
        blocker="required scenarios cannot be skipped",
    )
    optional = next(item for item in outcomes if not item["required"])
    optional.update(frozen_unavailability=False)
    failed = next(
        item for item in outcomes if item["required"] and item["cell_id"] != required["cell_id"]
    )
    failed.update(status="failed", error="CUDA OOM at minimum frozen batch size")

    reconciliation = reconcile_confirmatory_cell_outcomes(plan, outcomes)

    assert not reconciliation.passed
    assert reconciliation.failed_cell_count == 1
    assert required["cell_id"] in reconciliation.invalid_cell_ids
    assert optional["cell_id"] in reconciliation.invalid_cell_ids


def test_duplicate_extra_and_missing_outcomes_fail_exact_set_reconciliation() -> None:
    plan, outcomes = _successful_outcomes()
    missing = outcomes.pop()
    outcomes.append(dict(outcomes[0]))
    outcomes.append(
        {
            "cell_id": "confirmatory_unplanned_cell",
            "required": True,
            "status": "completed",
        }
    )

    reconciliation = reconcile_confirmatory_cell_outcomes(plan, outcomes)

    assert not reconciliation.passed
    assert reconciliation.missing_cell_ids == (missing["cell_id"],)
    assert reconciliation.duplicate_cell_ids == (outcomes[0]["cell_id"],)
    assert reconciliation.extra_cell_ids == ("confirmatory_unplanned_cell",)


def test_serialised_gate_is_rejected_for_outcome_eligible_completion() -> None:
    plan, outcomes = _successful_outcomes()
    reconciliation = reconcile_confirmatory_cell_outcomes(plan, outcomes)
    gate = _gate(plan).as_dict()

    with pytest.raises(ValueError, match="typed ConfirmatoryExecutionGateEvidence"):
        build_confirmatory_completion_evidence(
            plan=plan,
            reconciliation=reconciliation,
            artifact_scope=REAL_CONFIRMATORY_ARTIFACT_SCOPE,
            study_outcome_eligible=True,
            gate_evidence=gate,
            run_directory=Path("."),
        )


def test_frozen_hash_or_matrix_binding_mismatch_blocks_completion() -> None:
    plan, outcomes = _successful_outcomes()
    reconciliation = reconcile_confirmatory_cell_outcomes(plan, outcomes)
    gate = _gate(plan)
    gate = replace(
        gate,
        primary_gate=replace(
            gate.primary_gate,
            confirmatory_config_semantic_sha256="0" * 64,
        ),
    )

    with pytest.raises(ValueError, match="semantic hash differs"):
        build_confirmatory_completion_evidence(
            plan=plan,
            reconciliation=reconciliation,
            artifact_scope=REAL_CONFIRMATORY_ARTIFACT_SCOPE,
            study_outcome_eligible=True,
            gate_evidence=gate,
            run_directory=Path("."),
        )


def test_missing_frozen_sha_binding_blocks_completion() -> None:
    plan, outcomes = _successful_outcomes()
    reconciliation = reconcile_confirmatory_cell_outcomes(plan, outcomes)
    gate = _gate(plan)
    gate = replace(
        gate,
        primary_gate=replace(
            gate.primary_gate,
            dataset_sha256="invalid",
        ),
    )

    with pytest.raises(ValueError, match="dataset_sha256 is not a SHA-256"):
        build_confirmatory_completion_evidence(
            plan=plan,
            reconciliation=reconciliation,
            artifact_scope=REAL_CONFIRMATORY_ARTIFACT_SCOPE,
            study_outcome_eligible=True,
            gate_evidence=gate,
            run_directory=Path("."),
        )
