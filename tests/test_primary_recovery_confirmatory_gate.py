"""Downstream identity tests for the bounded primary orphan recovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import histo_audit.experiment.primary_completion as completion_module
import histo_audit.experiment.primary_statistics as statistics_module
import histo_audit.utils.run_tracking as run_tracking
import histo_audit.workflows.preregistration_amendment as amendment_module
import histo_audit.workflows.study_gates as study_gates
from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.experiment.primary_completion import (
    PrimaryFilesystemReadbackEvidence,
    PrimaryMatrixReconciliation,
    PrimaryRestorationReadbackEvidence,
)
from histo_audit.experiment.primary_recovery import RECOVERY_COPY_POLICY
from histo_audit.experiment.primary_statistics import InheritedPrimaryStatisticsVerification
from histo_audit.utils.run_tracking import (
    IntegrityVerification,
    RunStageEligibilityReceipt,
    RunTracker,
    attest_primary_run_stage_eligibility,
)
from histo_audit.workflows.study_gates import (
    PRIMARY_RECOVERY_EXPERIMENT_NAME,
    PrimaryExecutionGateEvidence,
    validate_confirmatory_execution_gate,
)

_SOURCE_RUN_ID = "interrupted-primary"
_RECOVERY_RUN_ID = "sealed-recovery"
_EVIDENCE_POLICY = "interrupted_unsealed_primary_recovery_v1"
_COPY_POLICY = RECOVERY_COPY_POLICY
_VERIFICATION_MODE = "inherited_prior_numeric_verification_v1"
_DISPOSITION = "amended_or_exploratory"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _gate(
    tmp_path: Path,
    *,
    source_tree_root_sha256: str = "d" * 64,
    frozen_primary_config: str | None = None,
) -> PrimaryExecutionGateEvidence:
    authority = (tmp_path / "recovery-authority").resolve()
    base = (tmp_path / "base-freeze").resolve()
    authority.mkdir()
    base.mkdir()
    frozen_primary_config_sha256 = "4" * 64
    if frozen_primary_config is not None:
        config_path = authority / "primary_frozen.yaml"
        config_path.write_text(frozen_primary_config, encoding="utf-8")
        frozen_primary_config_sha256 = study_gates.sha256_file(config_path)
    return PrimaryExecutionGateEvidence(
        freeze_directory=authority,
        freeze_artifact_root_sha256="1" * 64,
        freeze_manifest_sha256="2" * 64,
        preregistration_sha256="3" * 64,
        frozen_primary_config_sha256=frozen_primary_config_sha256,
        frozen_confirmatory_config_sha256="5" * 64,
        primary_config_semantic_sha256="6" * 64,
        confirmatory_config_semantic_sha256="7" * 64,
        primary_matrix_cell_count=222,
        primary_required_cell_count=185,
        confirmatory_matrix_cell_count=3,
        pilot_run_id="pilot",
        pilot_artifact_root_sha256="8" * 64,
        dataset_sha256="9" * 64,
        manifest_sha256="a" * 64,
        duplicate_audit_sha256="b" * 64,
        pathology_encoder_audit_sha256="c" * 64,
        source_tree_root_sha256=source_tree_root_sha256,
        base_freeze_directory=base,
        registration_authority_kind="preregistration_amendment",
        registration_status=_DISPOSITION,
        registration_authority_chain_depth=1,
        original_unamended_primary_claim_allowed=False,
        amended_primary_claim_allowed=False,
    )


def _authorization(tmp_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy": _EVIDENCE_POLICY,
        "source_run_id": _SOURCE_RUN_ID,
        "source_run_directory": str((tmp_path / "runs" / _SOURCE_RUN_ID).resolve()),
        "interruption_evidence": {
            "kind": "host_reboot",
            "observed_at_utc": "2026-07-27T11:07:04.422516Z",
            "last_boot_at_utc": "2026-07-27T10:37:04.500000Z",
            "event_id": 12,
            "source_process_id": 20792,
            "process_checked_at_utc": "2026-07-27T11:07:04.422516Z",
            "process_active": False,
            "receipt_path": str((tmp_path / "boot-receipt.json").resolve()),
            "receipt_sha256": "e" * 64,
        },
        "outcomes_inspected": True,
        "outcome_inspection_at_utc": "2026-07-27T10:57:07.000000Z",
        "analysis_disposition": _DISPOSITION,
        "scientific_method_changes": [],
        "expected_status_sha256": "f" * 64,
        "expected_primary_execution_gate_sha256": "0" * 64,
        "expected_source_tree_manifest_sha256": "1" * 64,
        "expected_source_tree_root_sha256": "2" * 64,
        "expected_source_snapshot_root_sha256": "3" * 64,
        "expected_source_filesystem_readback_root_sha256": "4" * 64,
        "expected_restoration_readback_root_sha256": "5" * 64,
        "expected_statistics_manifest_sha256": "6" * 64,
        "trust_assumption": "trusted read-only checksum-bound orphan",
        "limitation": "post-outcome recovery is exploratory",
        "reason": "Recover an interrupted run without changing the scientific method.",
    }


@dataclass(slots=True)
class _RecoveryTree:
    run: Path
    gate: PrimaryExecutionGateEvidence
    authorization: dict[str, Any]
    evidence: dict[str, Any]
    completion: dict[str, Any]
    metrics: dict[str, Any]
    receipt: RunStageEligibilityReceipt


def _receipt(
    run: Path,
    *,
    completion_sha256: str,
    evidence_sha256: str,
    authorization_sha256: str,
) -> RunStageEligibilityReceipt:
    verification = {
        "schema_version": 2,
        "policy": "primary_orphan_recovery_postseal_attestation_v1",
        "experiment_name": PRIMARY_RECOVERY_EXPERIMENT_NAME,
        "run_id": run.name,
        "run_path": str(run),
        "completion_stage": "PRIMARY_STUDY_COMPLETE",
        "retry_of_run_id": _SOURCE_RUN_ID,
        "lineage_binding_sha256": evidence_sha256,
        "authorization_binding_sha256": authorization_sha256,
        "semantic_verification_status": "passed",
    }
    record = {
        "schema_version": 1,
        "sequence": 1,
        "event_type": "postseal_primary_stage_attested",
        "recorded_at_utc": "2026-07-27T12:00:00.000000Z",
        "run_id": run.name,
        "run_path": str(run),
        "terminal_status": "completed",
        "scientific_stage_eligible": True,
        "completion_stage": "PRIMARY_STUDY_COMPLETE",
        "artifact_root_sha256": "7" * 64,
        "artifact_manifest_sha256": "8" * 64,
        "completion_evidence_sha256": completion_sha256,
        "verification": verification,
        "verification_sha256": canonical_sha256(verification),
        "previous_record_sha256": None,
        "record_sha256": "9" * 64,
    }
    receipt = RunStageEligibilityReceipt(
        run_directory=run,
        run_id=run.name,
        completion_stage="PRIMARY_STUDY_COMPLETE",
        record_sha256="9" * 64,
        verification_sha256=canonical_sha256(verification),
        _canonical_record_json=json.dumps(record, sort_keys=True),
    )
    object.__setattr__(
        receipt,
        "_attestation",
        run_tracking._RUN_STAGE_ELIGIBILITY_RECEIPT_TOKEN,
    )
    return receipt


def _recovery_tree(tmp_path: Path) -> _RecoveryTree:
    gate = _gate(tmp_path)
    authorization = _authorization(tmp_path)
    authorization_sha256 = canonical_sha256(authorization)
    run = (tmp_path / "runs" / _RECOVERY_RUN_ID).resolve()
    run.mkdir(parents=True)
    source_snapshot_root = authorization["expected_source_snapshot_root_sha256"]
    proof_sha256 = "a" * 64
    evidence = {
        "schema_version": 1,
        "policy": _EVIDENCE_POLICY,
        "experiment_name": PRIMARY_RECOVERY_EXPERIMENT_NAME,
        "source_run_id": _SOURCE_RUN_ID,
        "destination_run_id": run.name,
        "recovery_authorization_sha256": authorization_sha256,
        "source_snapshot_root_sha256": source_snapshot_root,
        "destination_snapshot_root_sha256": source_snapshot_root,
        "reused_required_cell_count": 185,
        "skipped_optional_cell_count": 37,
        "retrained_cell_count": 0,
        "copy_policy": _COPY_POLICY,
        "copied_artifact_count": 2_270,
        "copied_total_bytes": 46_291_340_622,
        "verification_mode": _VERIFICATION_MODE,
        "prior_numeric_verification_proof_sha256": proof_sha256,
        "analysis_disposition": _DISPOSITION,
        "outcomes_inspected": True,
        "training_invoked": False,
        "matrix_executor_invoked": False,
        "fallback_invoked": False,
        "automatic_retry_allowed": False,
    }
    evidence_path = run / "primary_recovery_evidence.json"
    _write_json(evidence_path, evidence)
    evidence_sha256 = study_gates.sha256_file(evidence_path)
    completion = {
        "schema_version": 2,
        "completion_stage": "PRIMARY_STUDY_COMPLETE",
        "study_outcome_eligible": True,
        "artifact_scope": "real_pannuke_primary_study",
        "run_id": run.name,
        "recovery_only": True,
        "recovery_policy": _EVIDENCE_POLICY,
        "primary_recovery_evidence_sha256": evidence_sha256,
        "recovery_evidence_sha256": evidence_sha256,
        "recovery_authorization_sha256": authorization_sha256,
        "recovery_source_snapshot_root_sha256": source_snapshot_root,
        "retry_of_run_id": _SOURCE_RUN_ID,
        "retry_predecessor_binding_sha256": evidence_sha256,
        "analysis_disposition": _DISPOSITION,
        "outcomes_inspected": True,
        "outcome_inspection_at_utc": authorization["outcome_inspection_at_utc"],
        "verification_mode": _VERIFICATION_MODE,
        "prior_numeric_verification_proof_sha256": proof_sha256,
        "planned_cell_count": 222,
        "required_cell_count": 185,
        "completed_required_cell_count": 185,
        "reused_required_cell_count": 185,
        "skipped_optional_cell_count": 37,
        "failed_required_cell_count": 0,
        "retrained_cell_count": 0,
        "physical_copy_verified": True,
        "copy_policy": _COPY_POLICY,
        "copied_artifact_count": 2_270,
        "copied_total_bytes": 46_291_340_622,
        "training_invoked": False,
        "matrix_executor_invoked": False,
        "fallback_invoked": False,
        "automatic_retry_allowed": False,
    }
    completion.update(
        {
            field: getattr(gate, field)
            for field in (
                "freeze_artifact_root_sha256",
                "frozen_primary_config_sha256",
                "frozen_confirmatory_config_sha256",
                "dataset_sha256",
                "manifest_sha256",
                "duplicate_audit_sha256",
                "pathology_encoder_audit_sha256",
                "source_tree_root_sha256",
            )
        }
    )
    completion_path = run / "completion_evidence.json"
    _write_json(completion_path, completion)
    metrics = {
        "schema_version": 1,
        "artifact_scope": "real_pannuke_primary_study",
        "completion_stage": "PRIMARY_STUDY_COMPLETE",
        "study_outcome_eligible": True,
        "recovery_only": True,
        "recovery_policy": _EVIDENCE_POLICY,
        "retry_of_run_id": _SOURCE_RUN_ID,
        "retry_predecessor_binding_sha256": evidence_sha256,
        "analysis_disposition": _DISPOSITION,
        "outcomes_inspected": True,
        "required_cell_count": 185,
        "completed_required_cell_count": 185,
        "skipped_optional_cell_count": 37,
        "physical_copy_verified": True,
        "copy_policy": _COPY_POLICY,
        "copied_artifact_count": 2_270,
        "copied_total_bytes": 46_291_340_622,
        "training_invoked": False,
        "matrix_executor_invoked": False,
        "fallback_invoked": False,
        "automatic_retry_allowed": False,
    }
    provenance = {
        key: value
        for key, value in completion.items()
        if key not in {"required_cell_count", "completed_required_cell_count"}
    }
    _write_json(run / "metrics.json", metrics)
    _write_json(run / "run_provenance.json", provenance)
    _write_json(
        run / "status.json",
        {
            "schema_version": 1,
            "run_id": run.name,
            "experiment_name": PRIMARY_RECOVERY_EXPERIMENT_NAME,
            "status": "completed",
        },
    )
    _write_json(run / "reconciliation.json", {"status": "passed"})
    receipt = _receipt(
        run,
        completion_sha256=study_gates.sha256_file(completion_path),
        evidence_sha256=evidence_sha256,
        authorization_sha256=authorization_sha256,
    )
    return _RecoveryTree(
        run=run,
        gate=gate,
        authorization=authorization,
        evidence=evidence,
        completion=completion,
        metrics=metrics,
        receipt=receipt,
    )


def _patch_authority(
    monkeypatch: pytest.MonkeyPatch,
    tree: _RecoveryTree,
) -> None:
    for module in (study_gates, amendment_module):
        monkeypatch.setattr(
            module,
            "require_primary_recovery_authorization",
            lambda _directory: dict(tree.authorization),
        )


@dataclass(slots=True)
class _SealedRecoveryE2E:
    run: Path
    gate: PrimaryExecutionGateEvidence
    controls: Any
    filesystem: PrimaryFilesystemReadbackEvidence
    restoration: PrimaryRestorationReadbackEvidence
    authorization: dict[str, Any]


def _sealed_recovery_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _SealedRecoveryE2E:
    project_root = (tmp_path / "project").resolve()
    project_root.mkdir()
    tracker = RunTracker.start(
        experiment_name=PRIMARY_RECOVERY_EXPERIMENT_NAME,
        config={"experiment_name": PRIMARY_RECOVERY_EXPERIMENT_NAME, "seed": {}},
        project_root=project_root,
        runs_root=tmp_path / "runs",
        environment={},
    )
    gate = _gate(
        tmp_path,
        source_tree_root_sha256=str(tracker.source_tree["root_sha256"]),
        frozen_primary_config="{}\n",
    )
    run = tracker.run_directory.resolve()

    matrix_plan_path = tracker.write_text("matrix_plan.json", '{"fixture":"plan"}\n')
    execution_controls_path = tracker.write_text(
        "execution_controls.json",
        '{"fixture":"controls"}\n',
    )
    cell_index_path = tracker.write_text("cell_index.csv", "cell_id,status\n")
    tracker.write_json("primary_execution_gate.json", gate.as_dict())
    controls = SimpleNamespace(
        config_semantic_sha256=gate.primary_config_semantic_sha256,
        plan=SimpleNamespace(cells=tuple(range(222)), required_cell_count=185),
        plan_sha256=study_gates.sha256_file(matrix_plan_path),
        binding_sha256="e" * 64,
        within_cell_comparisons=("within",),
        method_vs_random_comparisons=("random",),
        cross_cell_comparisons=("cross",),
    )
    tracker.write_json(
        "primary_input_bindings.json",
        {
            "config_semantic_sha256": controls.config_semantic_sha256,
            "plan_semantic_sha256": controls.plan_sha256,
            "execution_controls_binding_sha256": controls.binding_sha256,
        },
    )

    reconciliation = PrimaryMatrixReconciliation(
        status="passed",
        planned_cell_count=222,
        planned_required_cell_count=185,
        completed_cell_count=185,
        completed_required_cell_count=185,
        skipped_optional_cell_count=37,
        failed_cell_count=0,
        missing_cell_ids=(),
        extra_cell_ids=(),
        duplicate_cell_ids=(),
        invalid_cell_ids=(),
        errors=(),
    )
    tracker.write_json("reconciliation.json", reconciliation.as_dict())
    filesystem = PrimaryFilesystemReadbackEvidence(
        run_directory=run,
        status="passed",
        matrix_plan_sha256=study_gates.sha256_file(matrix_plan_path),
        execution_controls_sha256=study_gates.sha256_file(execution_controls_path),
        execution_controls_binding_sha256=controls.binding_sha256,
        cell_index_sha256=study_gates.sha256_file(cell_index_path),
        readback_root_sha256="f" * 64,
        planned_cell_count=222,
        completed_cell_count=185,
        completed_required_cell_count=185,
        skipped_optional_cell_count=37,
        circularity_excluded_cell_ids=(),
        cell_artifact_manifest_sha256=(),
        scenario_artifact_sha256=(),
        scenario_corruption_sha256=(),
        reconciliation=reconciliation,
    )
    object.__setattr__(
        filesystem,
        "_attestation",
        completion_module._FILESYSTEM_READBACK_ATTESTATION,
    )

    restoration_index_path = tracker.write_text("restoration_index.json", "{}\n")
    restoration = PrimaryRestorationReadbackEvidence(
        run_directory=run,
        status="passed",
        restoration_index_sha256=study_gates.sha256_file(restoration_index_path),
        readback_root_sha256="0" * 64,
        source_readback_root_sha256=filesystem.readback_root_sha256,
        restoration_cell_count=0,
        downstream_comparison_count=0,
        cell_json_sha256=(),
        cell_evidence_sha256=(),
        cell_manifest_sha256=(),
    )
    object.__setattr__(
        restoration,
        "_attestation",
        completion_module._RESTORATION_READBACK_ATTESTATION,
    )

    statistics_path = tracker.write_text("primary_statistics.json", '{"status":"passed"}\n')
    bootstrap_path = tracker.write_text("primary_bootstrap_evidence.npz", "bootstrap\n")
    subgroups_path = tracker.write_text("primary_subgroups.csv", "cell_id,status\n")
    statistics_records = []
    for path in (statistics_path, bootstrap_path, subgroups_path):
        statistics_records.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": study_gates.sha256_file(path),
            }
        )
    statistics_manifest_path = tracker.write_json(
        "primary_statistics_manifest.json",
        {
            "schema_version": 1,
            "execution_controls_binding_sha256": controls.binding_sha256,
            "source_filesystem_readback_root_sha256": filesystem.readback_root_sha256,
            "source_cell_artifact_manifest_sha256": {},
            "primary_input_bindings_sha256": study_gates.sha256_file(
                run / "primary_input_bindings.json"
            ),
            "crop_cache_sha256": "1" * 64,
            "artifacts": statistics_records,
            "statistics_payload_sha256": "2" * 64,
            "subgroup_rows_sha256": "3" * 64,
        },
    )
    comparison_count = 3

    authorization = _authorization(tmp_path)
    authorization.update(
        {
            "expected_source_filesystem_readback_root_sha256": (filesystem.readback_root_sha256),
            "expected_restoration_readback_root_sha256": restoration.readback_root_sha256,
            "expected_statistics_manifest_sha256": study_gates.sha256_file(
                statistics_manifest_path
            ),
        }
    )
    authorization_sha256 = canonical_sha256(authorization)
    source_snapshot_root = str(authorization["expected_source_snapshot_root_sha256"])
    proof_sha256 = "4" * 64
    evidence = {
        "schema_version": 1,
        "policy": _EVIDENCE_POLICY,
        "experiment_name": PRIMARY_RECOVERY_EXPERIMENT_NAME,
        "source_run_id": _SOURCE_RUN_ID,
        "destination_run_id": tracker.run_id,
        "recovery_authorization_sha256": authorization_sha256,
        "source_snapshot_root_sha256": source_snapshot_root,
        "destination_snapshot_root_sha256": source_snapshot_root,
        "reused_required_cell_count": 185,
        "skipped_optional_cell_count": 37,
        "retrained_cell_count": 0,
        "copy_policy": _COPY_POLICY,
        "copied_artifact_count": 2_270,
        "copied_total_bytes": 46_291_340_622,
        "verification_mode": _VERIFICATION_MODE,
        "prior_numeric_verification_proof_sha256": proof_sha256,
        "analysis_disposition": _DISPOSITION,
        "outcomes_inspected": True,
        "training_invoked": False,
        "matrix_executor_invoked": False,
        "fallback_invoked": False,
        "automatic_retry_allowed": False,
    }
    recovery_evidence_path = tracker.write_json("primary_recovery_evidence.json", evidence)
    evidence_sha256 = study_gates.sha256_file(recovery_evidence_path)
    recovery_input_path = tracker.write_json(
        "primary_recovery_input_bindings.json",
        {
            "schema_version": 1,
            "policy": _EVIDENCE_POLICY,
            "experiment_name": PRIMARY_RECOVERY_EXPERIMENT_NAME,
            "source_run_id": _SOURCE_RUN_ID,
            "recovery_authorization_sha256": authorization_sha256,
            "authority_directory": str(gate.freeze_directory),
            "authority_artifact_root_sha256": gate.freeze_artifact_root_sha256,
            "authority_manifest_sha256": gate.freeze_manifest_sha256,
            "source_snapshot_root_sha256": source_snapshot_root,
            "matrix_plan_sha256": controls.plan_sha256,
            "execution_controls_binding_sha256": controls.binding_sha256,
            "primary_gate_sha256": canonical_sha256(gate.as_dict()),
            "analysis_disposition": _DISPOSITION,
            "outcomes_inspected": True,
            "training_invoked": False,
            "matrix_executor_invoked": False,
            "fallback_invoked": False,
            "automatic_retry_allowed": False,
            "source_annotations_modified": False,
            "planned_cell_count": 222,
            "required_cell_count": 185,
        },
    )
    statistics_sha256 = study_gates.sha256_file(statistics_path)
    bootstrap_sha256 = study_gates.sha256_file(bootstrap_path)
    subgroups_sha256 = study_gates.sha256_file(subgroups_path)
    statistics_manifest_sha256 = study_gates.sha256_file(statistics_manifest_path)
    completion = {
        "schema_version": 2,
        "run_id": tracker.run_id,
        "completion_stage": "PRIMARY_STUDY_COMPLETE",
        "study_outcome_eligible": True,
        "artifact_scope": "real_pannuke_primary_study",
        "post_seal_attestation_required": True,
        "completion_stage_enabled_only_after_run_seal_and_integrity_verification": True,
        "post_seal_integrity_verification_required": True,
        "matrix_config_sha256": controls.config_semantic_sha256,
        "planned_cell_count": 222,
        "required_cell_count": 185,
        "completed_required_cell_count": 185,
        "skipped_optional_cell_count": 37,
        "failed_required_cell_count": 0,
        "reconciliation_status": "passed",
        "filesystem_run_directory": str(run),
        "filesystem_matrix_plan_sha256": filesystem.matrix_plan_sha256,
        "filesystem_execution_controls_sha256": filesystem.execution_controls_sha256,
        "filesystem_execution_controls_binding_sha256": (
            filesystem.execution_controls_binding_sha256
        ),
        "execution_controls_binding_sha256": controls.binding_sha256,
        "filesystem_cell_index_sha256": filesystem.cell_index_sha256,
        "filesystem_readback_root_sha256": filesystem.readback_root_sha256,
        "filesystem_completed_cell_count": filesystem.completed_cell_count,
        "circularity_excluded_cell_count": 0,
        "circularity_excluded_cell_ids": [],
        "primary_confirmatory_claims_require_exclusion_of_these_cells": False,
        "filesystem_scenario_corruption_sha256": {},
        "primary_statistics_verification_status": "passed",
        "primary_statistics_sha256": statistics_sha256,
        "primary_bootstrap_evidence_sha256": bootstrap_sha256,
        "primary_subgroups_sha256": subgroups_sha256,
        "primary_statistics_manifest_sha256": statistics_manifest_sha256,
        "primary_statistics_comparison_count": comparison_count,
        "primary_statistics_source_readback_root_sha256": filesystem.readback_root_sha256,
        "primary_restoration_verification_status": "passed",
        "primary_restoration_index_sha256": restoration.restoration_index_sha256,
        "primary_restoration_readback_root_sha256": restoration.readback_root_sha256,
        "primary_restoration_source_readback_root_sha256": (
            restoration.source_readback_root_sha256
        ),
        "primary_restoration_cell_count": restoration.restoration_cell_count,
        "primary_restoration_downstream_comparison_count": (
            restoration.downstream_comparison_count
        ),
        "recovery_only": True,
        "recovery_policy": _EVIDENCE_POLICY,
        "primary_recovery_evidence_sha256": evidence_sha256,
        "recovery_evidence_sha256": evidence_sha256,
        "recovery_authorization_sha256": authorization_sha256,
        "recovery_source_snapshot_root_sha256": source_snapshot_root,
        "retry_of_run_id": _SOURCE_RUN_ID,
        "retry_predecessor_binding_sha256": evidence_sha256,
        "analysis_disposition": _DISPOSITION,
        "outcomes_inspected": True,
        "outcome_inspection_at_utc": authorization["outcome_inspection_at_utc"],
        "verification_mode": _VERIFICATION_MODE,
        "prior_numeric_verification_proof_sha256": proof_sha256,
        "reused_required_cell_count": 185,
        "retrained_cell_count": 0,
        "physical_copy_verified": True,
        "copy_policy": _COPY_POLICY,
        "copied_artifact_count": 2_270,
        "copied_total_bytes": 46_291_340_622,
        "training_invoked": False,
        "matrix_executor_invoked": False,
        "fallback_invoked": False,
        "automatic_retry_allowed": False,
        "freeze_artifact_root_sha256": gate.freeze_artifact_root_sha256,
        "frozen_primary_config_sha256": gate.frozen_primary_config_sha256,
        "frozen_confirmatory_config_sha256": gate.frozen_confirmatory_config_sha256,
        "dataset_sha256": gate.dataset_sha256,
        "manifest_sha256": gate.manifest_sha256,
        "duplicate_audit_sha256": gate.duplicate_audit_sha256,
        "pathology_encoder_audit_sha256": gate.pathology_encoder_audit_sha256,
        "source_tree_root_sha256": gate.source_tree_root_sha256,
    }
    completion_path = tracker.write_json("completion_evidence.json", completion)
    metrics = {
        "schema_version": 1,
        "run_id": tracker.run_id,
        "artifact_scope": "real_pannuke_primary_study",
        "completion_stage": "PRIMARY_STUDY_COMPLETE",
        "study_outcome_eligible": True,
        "matrix_config_sha256": controls.config_semantic_sha256,
        "matrix_plan_sha256": controls.plan_sha256,
        "execution_controls_binding_sha256": controls.binding_sha256,
        "filesystem_readback_root_sha256": filesystem.readback_root_sha256,
        "primary_statistics_manifest_sha256": statistics_manifest_sha256,
        "primary_statistics_comparison_count": comparison_count,
        "completion_evidence_sha256": study_gates.sha256_file(completion_path),
        "recovery_only": True,
        "recovery_policy": _EVIDENCE_POLICY,
        "retry_of_run_id": _SOURCE_RUN_ID,
        "retry_predecessor_binding_sha256": evidence_sha256,
        "analysis_disposition": _DISPOSITION,
        "outcomes_inspected": True,
        "required_cell_count": 185,
        "completed_required_cell_count": 185,
        "skipped_optional_cell_count": 37,
        "physical_copy_verified": True,
        "copy_policy": _COPY_POLICY,
        "copied_artifact_count": 2_270,
        "copied_total_bytes": 46_291_340_622,
        "training_invoked": False,
        "matrix_executor_invoked": False,
        "fallback_invoked": False,
        "automatic_retry_allowed": False,
    }
    tracker.write_json("metrics.json", metrics)
    provenance = {
        key: value
        for key, value in completion.items()
        if key not in {"required_cell_count", "completed_required_cell_count"}
    }
    provenance.update(
        {
            "primary_execution_gate_sha256": study_gates.sha256_file(
                run / "primary_execution_gate.json"
            ),
            "primary_recovery_input_bindings_sha256": study_gates.sha256_file(recovery_input_path),
        }
    )
    tracker.write_provenance(**provenance)
    tracker.write_text("report.md", "# Recovery fixture\n")
    tracker.complete()

    tree = _RecoveryTree(
        run=run,
        gate=gate,
        authorization=authorization,
        evidence=evidence,
        completion=completion,
        metrics=metrics,
        receipt=_receipt(
            run,
            completion_sha256=study_gates.sha256_file(completion_path),
            evidence_sha256=evidence_sha256,
            authorization_sha256=authorization_sha256,
        ),
    )
    _patch_authority(monkeypatch, tree)
    integrity = run_tracking.verify_run_integrity(run)
    assert integrity.valid and integrity.expected_root_sha256 is not None
    statistics_verification = InheritedPrimaryStatisticsVerification(
        status="passed",
        output_directory=run,
        statistics_sha256=statistics_sha256,
        bootstrap_evidence_sha256=bootstrap_sha256,
        subgroups_sha256=subgroups_sha256,
        manifest_sha256=statistics_manifest_sha256,
        source_readback_root_sha256=filesystem.readback_root_sha256,
        comparison_count=comparison_count,
        verification_mode=_VERIFICATION_MODE,
        prior_numeric_verification_proof_sha256=proof_sha256,
    )
    object.__setattr__(
        statistics_verification,
        "_attestation",
        statistics_module._INHERITED_STATISTICS_VERIFICATION_ATTESTATION,
    )
    object.__setattr__(
        statistics_verification,
        "_authorization_kind",
        "orphan_recovery",
    )
    verification = run_tracking._build_primary_stage_attestation_verification(
        run,
        integrity=integrity,
        completion=completion,
        filesystem_readback=filesystem,
        statistics_verification=statistics_verification,
        restoration_readback=restoration,
        lineage_verification=evidence,
    )
    attest_primary_run_stage_eligibility(run, verification=verification)
    return _SealedRecoveryE2E(
        run=run,
        gate=gate,
        controls=controls,
        filesystem=filesystem,
        restoration=restoration,
        authorization=authorization,
    )


def test_exact_recovery_lineage_is_admitted_as_exploratory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _recovery_tree(tmp_path)
    _patch_authority(monkeypatch, tree)

    identity = study_gates._validate_primary_recovery_downstream_identity(
        run_directory=tree.run,
        primary_gate=tree.gate,
        completion=tree.completion,
        metrics=tree.metrics,
        primary_stage_eligibility_receipt=tree.receipt,
        recovery_experiment=True,
    )

    assert identity is not None
    assert identity.source_run_id == _SOURCE_RUN_ID
    assert identity.analysis_disposition == _DISPOSITION
    assert identity.evidence_sha256 == study_gates.sha256_file(
        tree.run / "primary_recovery_evidence.json"
    )


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("completion", "reused_required_cell_count", 184),
        ("completion", "skipped_optional_cell_count", 36),
        ("completion", "retrained_cell_count", 1),
        ("completion", "training_invoked", True),
        ("completion", "fallback_invoked", True),
        ("completion", "automatic_retry_allowed", True),
        ("completion", "analysis_disposition", "original_unamended"),
        ("evidence", "reused_required_cell_count", 184),
        ("evidence", "matrix_executor_invoked", True),
    ],
)
def test_recovery_lineage_tamper_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    field: str,
    value: Any,
) -> None:
    tree = _recovery_tree(tmp_path)
    _patch_authority(monkeypatch, tree)
    payload = getattr(tree, target)
    payload[field] = value
    if target == "evidence":
        _write_json(tree.run / "primary_recovery_evidence.json", payload)
    else:
        _write_json(tree.run / "completion_evidence.json", payload)

    with pytest.raises(ValueError, match=r"orphan.?recovery"):
        study_gates._validate_primary_recovery_downstream_identity(
            run_directory=tree.run,
            primary_gate=tree.gate,
            completion=tree.completion,
            metrics=tree.metrics,
            primary_stage_eligibility_receipt=tree.receipt,
            recovery_experiment=True,
        )


def test_ordinary_primary_keeps_the_existing_downstream_shape(tmp_path: Path) -> None:
    run = (tmp_path / "ordinary-primary").resolve()
    run.mkdir()
    gate = _gate(tmp_path)

    identity = study_gates._validate_primary_recovery_downstream_identity(
        run_directory=run,
        primary_gate=gate,
        completion={"completion_stage": "PRIMARY_STUDY_COMPLETE"},
        metrics={},
        primary_stage_eligibility_receipt=SimpleNamespace(),
        recovery_experiment=False,
    )
    ordinary_gate = study_gates.ConfirmatoryExecutionGateEvidence(
        primary_gate=gate,
        primary_run_directory=run,
        primary_run_id=run.name,
        primary_artifact_root_sha256="1" * 64,
        primary_completion_evidence_sha256="2" * 64,
        primary_reconciliation_sha256="3" * 64,
        completed_required_cell_count=185,
    )

    assert identity is None
    assert "primary_orphan_recovery" not in ordinary_gate.as_dict()
    assert "primary_recovery_evidence_sha256" not in ordinary_gate.as_dict()


def test_confirmatory_gate_selects_recovery_authority_from_sealed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _recovery_tree(tmp_path)
    _patch_authority(monkeypatch, tree)
    for relative in (
        ".immutable.json",
        "artifact_manifest.json",
        "source_tree_manifest.json",
        "primary_execution_gate.json",
        "primary_input_bindings.json",
        "matrix_plan.json",
        "execution_controls.json",
        "cell_index.csv",
        "primary_statistics.json",
        "primary_bootstrap_evidence.npz",
        "primary_subgroups.csv",
        "primary_statistics_manifest.json",
        "restoration_index.json",
        "report.md",
    ):
        path = tree.run / relative
        if not path.exists():
            path.write_bytes(b"fixture")
    captured: dict[str, Any] = {}

    def fake_primary_gate(**kwargs: Any) -> PrimaryExecutionGateEvidence:
        captured.update(kwargs)
        return tree.gate

    integrity = IntegrityVerification(
        valid=True,
        run_id=tree.run.name,
        expected_root_sha256="7" * 64,
        actual_root_sha256="7" * 64,
        missing_paths=(),
        added_paths=(),
        changed_paths=(),
        registry_record_present=True,
        errors=(),
    )
    statistics = SimpleNamespace(
        manifest_sha256="1" * 64,
        statistics_sha256="2" * 64,
        bootstrap_evidence_sha256="3" * 64,
        subgroups_sha256="4" * 64,
        source_readback_root_sha256="5" * 64,
        comparison_count=7,
        stage_attestation_record_sha256=tree.receipt.record_sha256,
        stage_attestation_verification_sha256=tree.receipt.verification_sha256,
    )
    restoration = SimpleNamespace(readback_root_sha256="6" * 64)
    monkeypatch.setattr(study_gates, "verify_run_integrity", lambda _run: integrity)
    monkeypatch.setattr(study_gates, "validate_primary_execution_gate", fake_primary_gate)
    monkeypatch.setattr(
        study_gates,
        "require_run_stage_eligibility_receipt",
        lambda *_args, **_kwargs: tree.receipt,
    )
    monkeypatch.setattr(
        study_gates,
        "_validate_primary_completion_attestations",
        lambda **_kwargs: (SimpleNamespace(), statistics, restoration),
    )
    monkeypatch.setattr(
        study_gates,
        "require_confirmatory_storage_policy",
        lambda _directory: {"policy": "single_canonical_checkpoint_copy_v1"},
    )

    result = validate_confirmatory_execution_gate(
        primary_run_directory=tree.run,
        project_root=tmp_path,
        freeze_directory=tree.gate.freeze_directory,
        dataset_path=tmp_path / "dataset",
        manifest_path=tmp_path / "manifest",
        duplicate_audit_path=tmp_path / "duplicates",
        pathology_encoder_audit_path=tmp_path / "pathology",
    )

    assert captured["experiment_name"] == PRIMARY_RECOVERY_EXPERIMENT_NAME
    assert result.primary_orphan_recovery is True
    assert result.primary_recovery_source_run_id == _SOURCE_RUN_ID
    assert result.primary_recovery_analysis_disposition == _DISPOSITION
    assert result.as_dict()["primary_recovery_evidence_sha256"] == (
        study_gates.sha256_file(tree.run / "primary_recovery_evidence.json")
    )


def test_actual_sealed_recovery_receipt_passes_full_confirmatory_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _sealed_recovery_e2e(tmp_path, monkeypatch)
    captured: dict[str, Any] = {}

    def fake_primary_gate(**kwargs: Any) -> PrimaryExecutionGateEvidence:
        captured.update(kwargs)
        return tree.gate

    monkeypatch.setattr(study_gates, "validate_primary_execution_gate", fake_primary_gate)
    monkeypatch.setattr(study_gates, "load_config", lambda _path: {})
    monkeypatch.setattr(
        study_gates,
        "primary_execution_controls_from_frozen_config",
        lambda _config: tree.controls,
    )
    monkeypatch.setattr(
        study_gates,
        "read_primary_filesystem_evidence",
        lambda _plan, _run_directory: tree.filesystem,
    )
    monkeypatch.setattr(
        study_gates,
        "read_primary_restoration_evidence",
        lambda _run_directory, _controls: tree.restoration,
    )
    monkeypatch.setattr(
        study_gates,
        "require_confirmatory_storage_policy",
        lambda _directory: {"policy": "single_canonical_checkpoint_copy_v1"},
    )

    integrity = run_tracking.verify_run_integrity(tree.run)
    receipt = run_tracking.require_run_stage_eligibility_receipt(
        tree.run,
        integrity=integrity,
    )
    assert integrity.valid
    assert receipt is not None and receipt.valid
    result = validate_confirmatory_execution_gate(
        primary_run_directory=tree.run,
        project_root=tmp_path / "project",
        freeze_directory=tree.gate.freeze_directory,
        dataset_path=tmp_path / "dataset",
        manifest_path=tmp_path / "manifest",
        duplicate_audit_path=tmp_path / "duplicates",
        pathology_encoder_audit_path=tmp_path / "pathology",
    )

    assert captured["experiment_name"] == PRIMARY_RECOVERY_EXPERIMENT_NAME
    assert result.primary_orphan_recovery is True
    assert result.primary_recovery_source_run_id == _SOURCE_RUN_ID
    assert result.primary_recovery_analysis_disposition == _DISPOSITION
    assert result.primary_stage_attestation_record_sha256 == receipt.record_sha256
    assert result.primary_stage_attestation_verification_sha256 == (receipt.verification_sha256)
    assert result.completed_required_cell_count == 185
