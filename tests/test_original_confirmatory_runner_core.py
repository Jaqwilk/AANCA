"""Closed-union tests for the original-confirmatory checkpoint data plane."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest
from test_study_contracts import complete_confirmatory_config

from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.cross_validation.image_oof import (
    ConfirmatoryCheckpointContractError,
    ConfirmatoryCheckpointFileIdentity,
    ConfirmatoryCheckpointPhysicalIdentity,
    _register_checkpoint_execution_contract,
    grouped_oof_confirmatory_cnn,
)
from histo_audit.experiment.confirmatory_completion import (
    REAL_CONFIRMATORY_ARTIFACT_SCOPE,
)
from histo_audit.experiment.confirmatory_core import (
    ConfirmatoryCellRequest,
    ConfirmatoryFrozenBlocker,
    _synthetic_frozen_runner,
    _synthetic_rotation,
    confirmatory_execution_controls_from_frozen_config,
    execute_confirmatory_matrix,
)
from histo_audit.experiment.original_confirmatory_resume import (
    ORIGINAL_CONFIRMATORY_CONFIG_SEMANTIC_SHA256,
    ORIGINAL_CONFIRMATORY_CONTROLS_BINDING_SHA256,
    ORIGINAL_CONFIRMATORY_PLAN_SEMANTIC_SHA256,
    OriginalConfirmatoryResumeContract,
)
from histo_audit.experiment.original_confirmatory_runner_core import (
    ORIGINAL_CONFIRMATORY_FIT_DIRECTIVE_POLICY,
    ORIGINAL_CONFIRMATORY_SUCCESSOR_COPY_POLICY,
    ORIGINAL_CONFIRMATORY_SUCCESSOR_PRECOPY_POLICY,
    OriginalConfirmatoryCanonicalECheckpointIdentity,
    OriginalConfirmatoryCanonicalECheckpointProjection,
    OriginalConfirmatoryCanonicalEDirectoryIdentity,
    OriginalConfirmatoryCapsuleExecutionRequest,
    OriginalConfirmatorySuccessorPrecopyCheckpointProjection,
    OriginalConfirmatorySuccessorPrecopyFitDirective,
    _run_original_confirmatory_capsule_request,
    build_original_confirmatory_canonical_e_checkpoint_projection,
    build_original_confirmatory_capsule_request_from_authority,
    build_original_confirmatory_fresh_checkpoint_execution_contract,
    build_original_confirmatory_successor_checkpoint_execution_contract,
    fit_directive_authority_records,
    fit_directives_root_sha256,
    materialize_original_confirmatory_successor_checkpoint_execution,
    prepare_original_confirmatory_successor_checkpoint_execution,
    require_original_confirmatory_canonical_e_projection,
)
from histo_audit.experiment.resource_bounded_resume import (
    ResumeCheckpointExpectation,
)
from histo_audit.models.cnn import ConfirmatoryCNNConfig
from histo_audit.workflows.study_gates import (
    ConfirmatoryExecutionGateEvidence,
    PrimaryExecutionGateEvidence,
)

_DATA_KEYS = (
    "training_data_sha256",
    "reference_validation_data_sha256",
    "training_split_sha256",
    "reference_validation_split_sha256",
)


def _confirmatory_gate(plan: Any) -> ConfirmatoryExecutionGateEvidence:
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
        pilot_run_id="pilot",
        pilot_artifact_root_sha256="7" * 64,
        dataset_sha256="8" * 64,
        manifest_sha256="9" * 64,
        duplicate_audit_sha256="a" * 64,
        pathology_encoder_audit_sha256="b" * 64,
        source_tree_root_sha256="c" * 64,
    )
    return ConfirmatoryExecutionGateEvidence(
        primary_gate=primary_gate,
        primary_run_directory=Path("primary"),
        primary_run_id="primary",
        primary_artifact_root_sha256="d" * 64,
        primary_completion_evidence_sha256="e" * 64,
        primary_reconciliation_sha256="f" * 64,
        completed_required_cell_count=180,
    )


def _successor_draft(tmp_path: Path) -> Any:
    fresh = build_original_confirmatory_fresh_checkpoint_execution_contract(_original_contract())
    directives = [replace(value, execution_mode="successor_resume") for value in fresh.directives]
    first = directives[0]
    relative_import = f"cells/{first.cell_id}/checkpoints/fold_{first.fold_id:02d}.pt"
    source = (tmp_path / "predecessor" / relative_import).resolve()
    destination = (tmp_path / "successor" / relative_import).resolve()
    directives[0] = replace(
        first,
        action="resume_incomplete_fit",
        checkpoint_sha256="c" * 64,
        checkpoint_size_bytes=123,
        source_predecessor_checkpoint=ConfirmatoryCheckpointFileIdentity(
            path=source,
            physical_identity=ConfirmatoryCheckpointPhysicalIdentity(
                device=1,
                inode=1,
                size_bytes=123,
                mode=stat.S_IFREG | 0o600,
                link_count=1,
                modified_time_ns=1,
                changed_time_ns=1,
            ),
            size_bytes=123,
            sha256="c" * 64,
        ),
        destination_imported_checkpoint=ConfirmatoryCheckpointFileIdentity(
            path=destination,
            physical_identity=ConfirmatoryCheckpointPhysicalIdentity(
                device=2,
                inode=2,
                size_bytes=123,
                mode=stat.S_IFREG | 0o600,
                link_count=1,
                modified_time_ns=2,
                changed_time_ns=2,
            ),
            size_bytes=123,
            sha256="c" * 64,
        ),
        completed_epochs_before_fit=1,
        stopped_early_before_fit=False,
        next_epoch_index=1,
    )
    values = tuple(directives)
    draft = replace(
        fresh,
        execution_mode="successor_resume",
        retry_of_run_id="synthetic_predecessor",
        directives=values,
        directives_sha256=canonical_sha256([value.as_dict() for value in values]),
        predecessor_checkpoint_read_performed=True,
        predecessor_checkpoint_copy_performed=True,
        predecessor_snapshot_sha256="d" * 64,
        predecessor_copy_receipt_sha256="e" * 64,
    )
    return _register_checkpoint_execution_contract(
        draft,
        expected_directive_count=180,
    )


def _original_contract(
    *,
    all_cell_ids: tuple[str, ...] | None = None,
    cnn_cell_ids: tuple[str, ...] | None = None,
    maximum_epochs: int = 3,
) -> OriginalConfirmatoryResumeContract:
    cnn_ids = cnn_cell_ids or tuple(f"cnn_cell_{index:03d}" for index in range(36))
    all_ids = all_cell_ids or tuple(
        sorted((*cnn_ids, *(f"other_cell_{index:03d}" for index in range(72))))
    )
    expectations: list[ResumeCheckpointExpectation] = []
    for cell_id in cnn_ids:
        for fold_id in range(5):
            expectations.append(
                ResumeCheckpointExpectation(
                    relative_path=(f"cells/{cell_id}/checkpoints/fold_{fold_id:02d}.pt"),
                    cell_id=cell_id,
                    fold_id=fold_id,
                    expected_configuration={
                        "epochs": maximum_epochs,
                        "seed": 303 + fold_id,
                        "test_fixture": True,
                    },
                    expected_model_metadata={
                        "architecture": "fixture_resnet18",
                        "cell_id": cell_id,
                    },
                    expected_data_and_split_sha256={
                        key: canonical_sha256(
                            {
                                "cell_id": cell_id,
                                "fold_id": fold_id,
                                "role": key,
                            }
                        )
                        for key in _DATA_KEYS
                    },
                )
            )
    payload = [
        {
            "relative_path": value.relative_path,
            "cell_id": value.cell_id,
            "fold_id": value.fold_id,
            "expected_configuration": dict(value.expected_configuration),
            "expected_model_metadata": dict(value.expected_model_metadata),
            "expected_data_and_split_sha256": dict(value.expected_data_and_split_sha256),
        }
        for value in expectations
    ]
    return OriginalConfirmatoryResumeContract(
        plan_semantic_sha256=ORIGINAL_CONFIRMATORY_PLAN_SEMANTIC_SHA256,
        config_semantic_sha256=ORIGINAL_CONFIRMATORY_CONFIG_SEMANTIC_SHA256,
        controls_binding_sha256=ORIGINAL_CONFIRMATORY_CONTROLS_BINDING_SHA256,
        all_cell_ids=tuple(sorted(all_ids)),
        cnn_cell_ids=tuple(sorted(cnn_ids)),
        checkpoint_expectations=tuple(expectations),
        checkpoint_allowlist_sha256=canonical_sha256(payload),
    )


def _blockers(
    controls: Any,
) -> dict[str, ConfirmatoryFrozenBlocker]:
    return {
        scenario.scenario_id: ConfirmatoryFrozenBlocker(
            scenario_id=scenario.scenario_id,
            config_semantic_sha256=controls.config_semantic_sha256,
            availability_audit_sha256=str(scenario.availability_audit_sha256),
            blocker="frozen optional encoder unavailable",
        )
        for scenario in controls.scenario_specs
        if not scenario.required
    }


def _capsule_request(
    output_directory: Path,
    *,
    draft: Any | None = None,
    projection: OriginalConfirmatoryCanonicalECheckpointProjection | None = None,
) -> OriginalConfirmatoryCapsuleExecutionRequest:
    controls = confirmatory_execution_controls_from_frozen_config(complete_confirmatory_config())
    if draft is None:
        all_ids = tuple(sorted(cell.cell_id for cell in controls.plan.cells))
        cnn_ids = tuple(
            sorted(
                cell.cell_id
                for cell in controls.plan.cells
                if controls.scenarios_by_id[cell.scenario_id].family == "cnn"
            )
        )
        draft = build_original_confirmatory_fresh_checkpoint_execution_contract(
            _original_contract(
                all_cell_ids=all_ids,
                cnn_cell_ids=cnn_ids,
                maximum_epochs=controls.max_epochs,
            )
        )
    if projection is None:
        projection = build_original_confirmatory_canonical_e_checkpoint_projection(draft)
    expected_run_directory = output_directory.resolve()
    runs_root = expected_run_directory.parent
    project_root = runs_root.parent.resolve()
    published_t0 = _published_t0_lifecycle_binding(
        project_root,
        artifact_root_sha256="8" * 64,
        technical_authorization_sha256="9" * 64,
    )
    technical = published_t0["technical_authority"]
    return OriginalConfirmatoryCapsuleExecutionRequest(
        gate_evidence=_confirmatory_gate(controls.plan),
        primary_run_directory=project_root / "primary",
        project_root=project_root,
        freeze_directory=project_root / "freeze",
        technical_authority_directory=Path(technical["authority_directory"]),
        technical_authority_artifact_root_sha256="8" * 64,
        technical_authorization_sha256="9" * 64,
        technical_authority_namespace_directory=Path(published_t0["namespace_directory"]),
        technical_authority_namespace_claim_sha256=str(published_t0["namespace_claim_sha256"]),
        published_technical_authority_lifecycle_binding_sha256=str(published_t0["binding_sha256"]),
        dataset_path=project_root / "dataset.parquet",
        manifest_path=project_root / "manifest.parquet",
        duplicate_audit_path=project_root / "duplicate-audit.npz",
        pathology_encoder_audit_path=project_root / "pathology-audit.json",
        frozen_primary_config_path=project_root / "primary.yaml",
        frozen_confirmatory_config_path=project_root / "confirmatory.yaml",
        crop_cache_path=project_root / "crop-cache",
        expected_crop_cache_sha256="a" * 64,
        expected_crop_metadata_sha256="b" * 64,
        expected_raw_inventory_sha256="c" * 64,
        frozen_feature_caches=(),
        observed_label_sets=(),
        runs_root=runs_root,
        supervisor_job_id="job-1",
        attempt_id="attempt-1",
        run_id=expected_run_directory.name,
        expected_run_directory=expected_run_directory,
        retry_of_run_id=None,
        resume_run_directory=None,
        lifecycle_readiness_run_directory=project_root / "readiness",
        artifact_scope=REAL_CONFIRMATORY_ARTIFACT_SCOPE,
        execution_mode="fresh",
        q_static_runner_binding_sha256="d" * 64,
        e_intent_core_sha256="e" * 64,
        expected_plan_sha256=canonical_sha256(controls.plan.as_dict()),
        expected_controls_binding_sha256=controls.binding_sha256,
        expected_bridge_binding_sha256="f" * 64,
        expected_gate_evidence_sha256=canonical_sha256(_confirmatory_gate(controls.plan).as_dict()),
        expected_cli_input_binding_sha256="1" * 64,
        scientific_request_projection_sha256="2" * 64,
        canonical_e_projection=projection,
        draft_checkpoint_contract=draft,
    )


def _published_t0_lifecycle_binding(
    project_root: Path,
    *,
    artifact_root_sha256: str,
    technical_authorization_sha256: str,
) -> dict[str, Any]:
    namespace = project_root / "artifacts" / "original_confirmatory_technical_authorities"
    technical_unsigned = {
        "schema_version": 1,
        "policy": "original_confirmatory_technical_authority_lifecycle_binding_v1",
        "authority_directory": str(namespace / "synthetic-authority"),
        "chain_depth": 3,
        "artifact_root_sha256": artifact_root_sha256,
        "sha256_manifest_sha256": "a" * 64,
        "execution_source_manifest_sha256": "b" * 64,
        "execution_source_root_sha256": "c" * 64,
        "parent_authority_directory": str(project_root / "freeze"),
        "parent_artifact_root_sha256": "d" * 64,
        "parent_sha256_manifest_sha256": "e" * 64,
        "technical_authorization_sha256": technical_authorization_sha256,
        "independent_review_receipt_sha256": "f" * 64,
        "immutable_marker_sha256": "1" * 64,
        "publication_attempt_sha256": "2" * 64,
        "publication_success_sha256": "3" * 64,
        "primary_outcomes_inspected": True,
        "confirmatory_outcomes_inspected": False,
        "confirmatory_outcome_values_read": False,
        "scientific_definition_changed": False,
        "automatic_retry_allowed": False,
    }
    technical = {
        **technical_unsigned,
        "binding_sha256": canonical_sha256(technical_unsigned),
    }
    published_unsigned = {
        "schema_version": 1,
        "policy": ("published_original_confirmatory_technical_authority_lifecycle_binding_v1"),
        "namespace_directory": str(namespace),
        "namespace_claim_sha256": "4" * 64,
        "review_attempt_claim_sha256": "5" * 64,
        "technical_authority": technical,
        "automatic_retry_allowed": False,
        "adoption_allowed": False,
        "cleanup_allowed": False,
    }
    return {
        **published_unsigned,
        "binding_sha256": canonical_sha256(published_unsigned),
    }


def _qualification() -> dict[str, Any]:
    return {
        "predecessor_class": "sealed_failed_demoted",
        "policy": "sealed_failed_demoted_confirmatory_predecessor_v1",
        "terminal_status": "failed",
        "sealed_integrity_valid": True,
        "integrity_registry_record_present": True,
        "artifact_root_sha256": "1" * 64,
        "artifact_manifest_sha256": "2" * 64,
        "immutable_marker_sha256": "3" * 64,
        "status_sha256": "4" * 64,
        "completion_evidence_sha256": "5" * 64,
        "process_receipt": {
            "expected_process_id": None,
            "expected_create_time_unix_us": None,
            "expected_identity_state": "not_supplied",
            "inspector_process_id": 1,
            "inspector_create_time_unix_us": 1,
            "exact_inspector_match_excluded": True,
            "exact_authorized_supervisor_match_excluded": True,
            "matching_predecessor_process_ids": [],
            "inspected_process_count": 1,
            "receipt_sha256": "6" * 64,
        },
        "active_lock_paths_before": [],
        "active_lock_paths_after": [],
        "stage_attestation_count": 0,
        "disposition_record_count": 0,
        "root_inventory_before_sha256": "7" * 64,
        "root_inventory_after_sha256": "7" * 64,
        "qualification_sha256": "8" * 64,
    }


def _successor_precopy_projection(
    request: OriginalConfirmatoryCapsuleExecutionRequest,
    *,
    predecessor_run_directory: Path,
    source_identities: dict[
        tuple[str, int],
        OriginalConfirmatoryCanonicalECheckpointIdentity,
    ],
    root_identity: OriginalConfirmatoryCanonicalEDirectoryIdentity | None = None,
) -> OriginalConfirmatorySuccessorPrecopyCheckpointProjection:
    directives: list[OriginalConfirmatorySuccessorPrecopyFitDirective] = []
    for value in request.draft_checkpoint_contract.directives:
        source = source_identities.get((value.cell_id, value.fold_id))
        action = "resume_incomplete_fit" if source is not None else "fresh_fit"
        directives.append(
            OriginalConfirmatorySuccessorPrecopyFitDirective(
                fit_id=f"{value.cell_id}::fold_{value.fold_id:02d}",
                execution_mode="successor_resume",
                cell_id=value.cell_id,
                fold_id=value.fold_id,
                action=action,
                source_predecessor_checkpoint=source,
                destination_checkpoint_relative_path=(
                    f"cells/{value.cell_id}/checkpoints/fold_{value.fold_id:02d}.pt"
                ),
                versioned_checkpoint_output_directory_relative_path=(
                    f"cells/{value.cell_id}/checkpoint_versions/fold_{value.fold_id:02d}"
                ),
                checkpoint_execution_manifest_relative_path=(
                    f"cells/{value.cell_id}/checkpoint_execution/fold_{value.fold_id:02d}.json"
                ),
                completed_epochs_before_fit=1 if source is not None else 0,
                stopped_early_before_fit=False if source is not None else None,
                next_epoch_index=1 if source is not None else 0,
                maximum_epochs=value.maximum_epochs,
                expected_configuration=json.loads(value.expected_configuration_json),
                expected_configuration_sha256=value.expected_configuration_sha256,
                expected_model_metadata=json.loads(value.expected_model_metadata_json),
                expected_model_metadata_sha256=value.expected_model_metadata_sha256,
                expected_data_and_split=json.loads(value.expected_data_and_split_json),
                expected_data_and_split_sha256=value.expected_data_and_split_sha256,
            )
        )
    values = tuple(directives)
    qualification = _qualification()
    source_records = [
        {
            "fit_id": value.fit_id,
            "source_predecessor_checkpoint": value.source_predecessor_checkpoint.as_dict(),
        }
        for value in values
        if value.source_predecessor_checkpoint is not None
    ]
    destination_paths = [value.destination_checkpoint_relative_path for value in values]
    root = root_identity or OriginalConfirmatoryCanonicalEDirectoryIdentity(
        path=predecessor_run_directory,
        device=1,
        inode=10,
        size_bytes=0,
        mode=stat.S_IFDIR | 0o555,
        link_count=1,
        modified_time_ns=1,
        changed_time_ns=1,
    )
    provisional = OriginalConfirmatorySuccessorPrecopyCheckpointProjection(
        schema_version=1,
        policy=ORIGINAL_CONFIRMATORY_SUCCESSOR_PRECOPY_POLICY,
        execution_mode="successor_resume",
        contract_profile="original_confirmatory_exact_180",
        retry_of_run_id=predecessor_run_directory.name,
        successor_run_id=request.run_id,
        predecessor_run_directory=predecessor_run_directory,
        predecessor_class="sealed_failed_demoted",
        predecessor_process_id=None,
        predecessor_process_create_time_unix_us=None,
        orphan_manual_diagnosis=False,
        resume_authority_binding_sha256="9" * 64,
        bound_resume_request_sha256="a" * 64,
        predecessor_qualification=qualification,
        predecessor_qualification_sha256=canonical_sha256(qualification),
        predecessor_root_identity=root,
        predecessor_snapshot_sha256="b" * 64,
        predecessor_checkpoint_tree_sha256="c" * 64,
        copy_policy=ORIGINAL_CONFIRMATORY_SUCCESSOR_COPY_POLICY,
        predecessor_checkpoint_read_performed=True,
        predecessor_checkpoint_copy_performed=False,
        outcome_artifacts_read=False,
        automatic_retry_allowed=False,
        predecessor_autodiscovery_allowed=False,
        checkpoint_fallback_allowed=False,
        max_attempt_count=1,
        base_template_directives_sha256=(request.draft_checkpoint_contract.directives_sha256),
        source_checkpoint_count=len(source_records),
        source_checkpoint_total_bytes=sum(
            value.source_predecessor_checkpoint.size_bytes
            for value in values
            if value.source_predecessor_checkpoint is not None
        ),
        source_checkpoint_inventory_sha256=canonical_sha256(source_records),
        destination_relative_paths_sha256=canonical_sha256(destination_paths),
        directives=values,
        directives_sha256=canonical_sha256([value.as_dict() for value in values]),
        projection_sha256="0" * 64,
    )
    projection = replace(
        provisional,
        projection_sha256=canonical_sha256(provisional.payload_without_self_hash()),
    )
    projection.validate()
    return projection


def _authority_pair(
    request: OriginalConfirmatoryCapsuleExecutionRequest,
    *,
    projection: (
        OriginalConfirmatoryCanonicalECheckpointProjection
        | OriginalConfirmatorySuccessorPrecopyCheckpointProjection
    ),
) -> tuple[dict[str, Any], dict[str, Any]]:
    absolute_primary_gate = replace(
        request.gate_evidence.primary_gate,
        freeze_directory=request.freeze_directory,
        base_freeze_directory=request.freeze_directory,
    )
    gate = replace(
        request.gate_evidence,
        primary_gate=absolute_primary_gate,
        primary_run_directory=request.primary_run_directory,
    )
    gate_raw = gate.as_dict()
    draft_raw = request.draft_checkpoint_contract.as_dict()
    cli_unsigned = {
        "schema_version": 1,
        "policy": "original_confirmatory_cli_input_binding_v1",
        "crop_cache_path": str(request.crop_cache_path),
        "expected_crop_cache_sha256": request.expected_crop_cache_sha256,
        "expected_crop_metadata_sha256": request.expected_crop_metadata_sha256,
        "expected_raw_inventory_sha256": request.expected_raw_inventory_sha256,
        "frozen_feature_caches": [],
        "frozen_feature_caches_sha256": canonical_sha256([]),
        "observed_label_sets": [],
        "observed_label_sets_sha256": canonical_sha256([]),
        "draft_checkpoint_contract": draft_raw,
        "draft_checkpoint_contract_sha256": canonical_sha256(draft_raw),
        "bridge_binding_sha256": request.expected_bridge_binding_sha256,
        "scientific_outcomes_read": False,
        "automatic_retry_allowed": False,
    }
    cli = {**cli_unsigned, "binding_sha256": canonical_sha256(cli_unsigned)}
    published_t0 = _published_t0_lifecycle_binding(
        request.project_root,
        artifact_root_sha256=request.technical_authority_artifact_root_sha256,
        technical_authorization_sha256=request.technical_authorization_sha256,
    )
    assert published_t0["technical_authority"]["authority_directory"] == str(
        request.technical_authority_directory
    )
    q_unsigned = {
        "schema_version": 3,
        "policy": "original_confirmatory_static_runner_binding_v3",
        "project_root": str(request.project_root),
        "primary_run_directory": str(request.primary_run_directory),
        "freeze_directory": str(request.freeze_directory),
        "technical_authority_directory": str(request.technical_authority_directory),
        "technical_authority_artifact_root_sha256": (
            request.technical_authority_artifact_root_sha256
        ),
        "technical_authorization_sha256": request.technical_authorization_sha256,
        "published_technical_authority_lifecycle_binding": published_t0,
        "lifecycle_readiness_run_directory": str(request.lifecycle_readiness_run_directory),
        "dataset_path": str(request.dataset_path),
        "manifest_path": str(request.manifest_path),
        "duplicate_audit_path": str(request.duplicate_audit_path),
        "pathology_encoder_audit_path": str(request.pathology_encoder_audit_path),
        "frozen_primary_config_path": str(request.frozen_primary_config_path),
        "frozen_confirmatory_config_path": str(request.frozen_confirmatory_config_path),
        "runs_root": str(request.runs_root),
        "expected_confirmatory_gate": gate_raw,
        "expected_confirmatory_gate_sha256": canonical_sha256(gate_raw),
        "expected_cli_input_binding": cli,
        "expected_cli_input_binding_sha256": canonical_sha256(cli),
        "artifact_scope": REAL_CONFIRMATORY_ARTIFACT_SCOPE,
        "semantic_outcome_read_scope": ("integrity/control_only_no_scientific_outcomes"),
    }
    q = {**q_unsigned, "binding_sha256": canonical_sha256(q_unsigned)}
    projection_raw = projection.as_dict()
    retry_of = (
        projection.retry_of_run_id
        if isinstance(
            projection,
            OriginalConfirmatorySuccessorPrecopyCheckpointProjection,
        )
        else None
    )
    mode = projection.execution_mode
    e_unsigned = {
        "schema_version": 1,
        "policy": "original_confirmatory_capsule_request_projection_v1",
        "q_static_runner_binding_sha256": q["binding_sha256"],
        "job_id": request.supervisor_job_id,
        "attempt_id": request.attempt_id,
        "run_id": request.run_id,
        "execution_mode": mode,
        "retry_of_run_id": retry_of,
        "runs_root": str(request.runs_root),
        "expected_run_directory": str(request.expected_run_directory),
        "plan_sha256": request.expected_plan_sha256,
        "controls_binding_sha256": request.expected_controls_binding_sha256,
        "bridge_binding_sha256": request.expected_bridge_binding_sha256,
        "gate_evidence_sha256": canonical_sha256(gate_raw),
        "cli_input_binding_sha256": canonical_sha256(cli),
        "checkpoint_authority_projection": projection_raw,
        "checkpoint_authority_projection_sha256": canonical_sha256(projection_raw),
        "checkpoint_contract_profile": "original_confirmatory_exact_180",
        "checkpoint_directive_count": 180,
        "artifact_scope": REAL_CONFIRMATORY_ARTIFACT_SCOPE,
        "scientific_outcomes_read": False,
        "selection_or_tuning_performed": False,
        "publication_performed": False,
        "automatic_retry_allowed": False,
    }
    return q, {**e_unsigned, "projection_sha256": canonical_sha256(e_unsigned)}


def test_fresh_builder_is_exact_180_outcome_blind_and_filesystem_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = _original_contract()

    def forbidden_filesystem_read(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("fresh contract builder touched the filesystem")

    monkeypatch.setattr(os.path, "lexists", forbidden_filesystem_read)
    monkeypatch.setattr(Path, "lstat", forbidden_filesystem_read)
    monkeypatch.setattr(Path, "read_text", forbidden_filesystem_read)

    execution = build_original_confirmatory_fresh_checkpoint_execution_contract(frozen)

    execution.validate(expected_directive_count=180)
    assert execution.execution_mode == "fresh"
    assert execution.retry_of_run_id is None
    assert not execution.predecessor_checkpoint_read_performed
    assert not execution.predecessor_checkpoint_copy_performed
    assert not execution.outcome_artifacts_read
    assert not execution.automatic_retry_allowed
    assert len(execution.directives) == 180
    assert all(value.action == "fresh_fit" for value in execution.directives)
    assert all(value.checkpoint_sha256 is None for value in execution.directives)
    assert {(value.cell_id, value.fold_id) for value in execution.directives} == {
        (cell_id, fold_id) for cell_id in frozen.cnn_cell_ids for fold_id in range(5)
    }


def test_authority_projection_is_stable_exact_and_outcome_blind() -> None:
    execution = build_original_confirmatory_fresh_checkpoint_execution_contract(
        _original_contract()
    )

    records = fit_directive_authority_records(execution)

    assert ORIGINAL_CONFIRMATORY_FIT_DIRECTIVE_POLICY.endswith("180_predeclared_fit_directives_v1")
    assert len(records) == 180
    assert records[0]["fit_id"] == "cnn_cell_000::fold_00"
    assert records[0]["cell_id"] == "cnn_cell_000"
    assert records[0]["fold_id"] == 0
    assert records[0]["action"] == "fresh_fit"
    assert records[0]["source_predecessor_checkpoint"] is None
    assert records[0]["destination_imported_checkpoint"] is None
    assert records[0]["versioned_checkpoint_output_directory_relative_path"] == (
        "cells/cnn_cell_000/checkpoint_versions/fold_00"
    )
    assert records[0]["checkpoint_execution_manifest_relative_path"] == (
        "cells/cnn_cell_000/checkpoint_execution/fold_00.json"
    )
    assert records[0]["completed_epochs_before_fit"] == 0
    assert records[0]["stopped_early_before_fit"] is None
    assert records[0]["next_epoch_index"] == 0
    assert records[0]["expected_configuration"]["epochs"] == 3
    assert records[0]["expected_configuration_sha256"]
    assert records[0]["expected_model_metadata_sha256"]
    assert records[0]["expected_data_and_split_sha256"]
    assert records[-1]["fit_id"] == "cnn_cell_035::fold_04"
    assert {record["action"] for record in records} == {"fresh_fit"}
    projection = build_original_confirmatory_canonical_e_checkpoint_projection(execution)
    assert isinstance(
        projection,
        OriginalConfirmatoryCanonicalECheckpointProjection,
    )
    assert projection.directives_sha256 == canonical_sha256(list(records))
    assert fit_directives_root_sha256(execution) == projection.directives_sha256
    assert (
        require_original_confirmatory_canonical_e_projection(
            projection,
            execution,
        )
        is execution
    )


def test_canonical_e_projection_is_typed_lossless_and_tamper_closed() -> None:
    execution = build_original_confirmatory_fresh_checkpoint_execution_contract(
        _original_contract()
    )
    projection = build_original_confirmatory_canonical_e_checkpoint_projection(execution)
    first = replace(
        projection.directives[0],
        action="restore_terminal_checkpoint_without_fit",
    )
    changed_directives = (first, *projection.directives[1:])
    tampered = replace(
        projection,
        directives=changed_directives,
        directives_sha256=canonical_sha256([value.as_dict() for value in changed_directives]),
    )

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="canonical E",
    ):
        require_original_confirmatory_canonical_e_projection(
            tampered,
            execution,
        )


def test_capsule_entry_is_one_typed_immutable_request_without_injection_surface(
    tmp_path: Path,
) -> None:
    from histo_audit.experiment import original_confirmatory_runner_core as module

    request = _capsule_request(tmp_path / "capsule")
    assert set(inspect.signature(_run_original_confirmatory_capsule_request).parameters) == {
        "request"
    }
    assert set(OriginalConfirmatoryCapsuleExecutionRequest.__dataclass_fields__) == {
        "gate_evidence",
        "primary_run_directory",
        "project_root",
        "freeze_directory",
        "technical_authority_directory",
        "technical_authority_artifact_root_sha256",
        "technical_authorization_sha256",
        "technical_authority_namespace_directory",
        "technical_authority_namespace_claim_sha256",
        "published_technical_authority_lifecycle_binding_sha256",
        "dataset_path",
        "manifest_path",
        "duplicate_audit_path",
        "pathology_encoder_audit_path",
        "frozen_primary_config_path",
        "frozen_confirmatory_config_path",
        "crop_cache_path",
        "expected_crop_cache_sha256",
        "expected_crop_metadata_sha256",
        "expected_raw_inventory_sha256",
        "frozen_feature_caches",
        "observed_label_sets",
        "runs_root",
        "supervisor_job_id",
        "attempt_id",
        "run_id",
        "expected_run_directory",
        "retry_of_run_id",
        "resume_run_directory",
        "lifecycle_readiness_run_directory",
        "artifact_scope",
        "execution_mode",
        "q_static_runner_binding_sha256",
        "e_intent_core_sha256",
        "expected_plan_sha256",
        "expected_controls_binding_sha256",
        "expected_bridge_binding_sha256",
        "expected_gate_evidence_sha256",
        "expected_cli_input_binding_sha256",
        "scientific_request_projection_sha256",
        "canonical_e_projection",
        "draft_checkpoint_contract",
    }
    assert {
        "callback",
        "progress_callback",
        "image_oof_runner",
        "frozen_oof_runner",
        "cpu_test_only",
        "lease",
        "capability",
        "rotations",
        "plan",
        "controls",
        "frozen_blockers",
    }.isdisjoint(OriginalConfirmatoryCapsuleExecutionRequest.__dataclass_fields__)
    for removed_name in (
        "_build_original_confirmatory_sealed_execution_request",
        "_execute_original_confirmatory_checkpoint_execution_lease",
        "_bind_original_confirmatory_checkpoint_authority_projection",
        "OriginalConfirmatoryCheckpointAuthorityProjectionView",
    ):
        assert not hasattr(module, removed_name)
    with pytest.raises(FrozenInstanceError):
        request.expected_run_directory = tmp_path / "changed"  # type: ignore[misc]
    with pytest.raises(TypeError, match="unexpected keyword"):
        OriginalConfirmatoryCapsuleExecutionRequest(
            **{
                **{
                    name: getattr(request, name)
                    for name in OriginalConfirmatoryCapsuleExecutionRequest.__dataclass_fields__
                },
                "progress_callback": lambda _event: None,
            }
        )


def test_authority_mapping_adapter_is_exact_filesystem_free_and_tamper_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _capsule_request(tmp_path / "runs" / "run-1")
    absolute_primary_gate = replace(
        request.gate_evidence.primary_gate,
        freeze_directory=request.freeze_directory,
        base_freeze_directory=request.freeze_directory,
    )
    gate = replace(
        request.gate_evidence,
        primary_gate=absolute_primary_gate,
        primary_run_directory=request.primary_run_directory,
    )
    gate_raw = gate.as_dict()
    cache_rows: list[dict[str, Any]] = []
    observed_rows: list[dict[str, Any]] = []
    draft_raw = request.draft_checkpoint_contract.as_dict()
    cli_unsigned = {
        "schema_version": 1,
        "policy": "original_confirmatory_cli_input_binding_v1",
        "crop_cache_path": str(request.crop_cache_path),
        "expected_crop_cache_sha256": request.expected_crop_cache_sha256,
        "expected_crop_metadata_sha256": request.expected_crop_metadata_sha256,
        "expected_raw_inventory_sha256": request.expected_raw_inventory_sha256,
        "frozen_feature_caches": cache_rows,
        "frozen_feature_caches_sha256": canonical_sha256(cache_rows),
        "observed_label_sets": observed_rows,
        "observed_label_sets_sha256": canonical_sha256(observed_rows),
        "draft_checkpoint_contract": draft_raw,
        "draft_checkpoint_contract_sha256": canonical_sha256(draft_raw),
        "bridge_binding_sha256": request.expected_bridge_binding_sha256,
        "scientific_outcomes_read": False,
        "automatic_retry_allowed": False,
    }
    cli = {**cli_unsigned, "binding_sha256": canonical_sha256(cli_unsigned)}
    published_t0 = _published_t0_lifecycle_binding(
        request.project_root,
        artifact_root_sha256=request.technical_authority_artifact_root_sha256,
        technical_authorization_sha256=request.technical_authorization_sha256,
    )
    q_unsigned = {
        "schema_version": 3,
        "policy": "original_confirmatory_static_runner_binding_v3",
        "project_root": str(request.project_root),
        "primary_run_directory": str(request.primary_run_directory),
        "freeze_directory": str(request.freeze_directory),
        "technical_authority_directory": str(request.technical_authority_directory),
        "technical_authority_artifact_root_sha256": (
            request.technical_authority_artifact_root_sha256
        ),
        "technical_authorization_sha256": request.technical_authorization_sha256,
        "published_technical_authority_lifecycle_binding": published_t0,
        "lifecycle_readiness_run_directory": str(request.lifecycle_readiness_run_directory),
        "dataset_path": str(request.dataset_path),
        "manifest_path": str(request.manifest_path),
        "duplicate_audit_path": str(request.duplicate_audit_path),
        "pathology_encoder_audit_path": str(request.pathology_encoder_audit_path),
        "frozen_primary_config_path": str(request.frozen_primary_config_path),
        "frozen_confirmatory_config_path": str(request.frozen_confirmatory_config_path),
        "runs_root": str(request.runs_root),
        "expected_confirmatory_gate": gate_raw,
        "expected_confirmatory_gate_sha256": canonical_sha256(gate_raw),
        "expected_cli_input_binding": cli,
        "expected_cli_input_binding_sha256": canonical_sha256(cli),
        "artifact_scope": REAL_CONFIRMATORY_ARTIFACT_SCOPE,
        "semantic_outcome_read_scope": "integrity/control_only_no_scientific_outcomes",
    }
    q = {**q_unsigned, "binding_sha256": canonical_sha256(q_unsigned)}
    projection_raw = request.canonical_e_projection.as_dict()
    e_unsigned = {
        "schema_version": 1,
        "policy": "original_confirmatory_capsule_request_projection_v1",
        "q_static_runner_binding_sha256": q["binding_sha256"],
        "job_id": request.supervisor_job_id,
        "attempt_id": request.attempt_id,
        "run_id": request.run_id,
        "execution_mode": "fresh",
        "retry_of_run_id": None,
        "runs_root": str(request.runs_root),
        "expected_run_directory": str(request.expected_run_directory),
        "plan_sha256": request.expected_plan_sha256,
        "controls_binding_sha256": request.expected_controls_binding_sha256,
        "bridge_binding_sha256": request.expected_bridge_binding_sha256,
        "gate_evidence_sha256": canonical_sha256(gate_raw),
        "cli_input_binding_sha256": canonical_sha256(cli),
        "checkpoint_authority_projection": projection_raw,
        "checkpoint_authority_projection_sha256": canonical_sha256(projection_raw),
        "checkpoint_contract_profile": "original_confirmatory_exact_180",
        "checkpoint_directive_count": 180,
        "artifact_scope": REAL_CONFIRMATORY_ARTIFACT_SCOPE,
        "scientific_outcomes_read": False,
        "selection_or_tuning_performed": False,
        "publication_performed": False,
        "automatic_retry_allowed": False,
    }
    e = {**e_unsigned, "projection_sha256": canonical_sha256(e_unsigned)}

    def forbidden_filesystem_read(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("authority mapping adapter touched the filesystem")

    monkeypatch.setattr(os.path, "lexists", forbidden_filesystem_read)
    monkeypatch.setattr(Path, "lstat", forbidden_filesystem_read)
    decoded = build_original_confirmatory_capsule_request_from_authority(
        q_static_runner_binding=q,
        e_scientific_request_projection=e,
        e_intent_core_sha256=request.e_intent_core_sha256,
    )

    assert decoded.gate_evidence == gate
    assert decoded.canonical_e_projection == request.canonical_e_projection
    assert decoded.draft_checkpoint_contract == request.draft_checkpoint_contract
    assert decoded.q_static_runner_binding_sha256 == q["binding_sha256"]
    assert decoded.technical_authority_directory == request.technical_authority_directory
    assert (
        decoded.technical_authority_artifact_root_sha256
        == request.technical_authority_artifact_root_sha256
    )
    assert decoded.technical_authorization_sha256 == request.technical_authorization_sha256
    assert (
        decoded.technical_authority_namespace_directory
        == request.technical_authority_namespace_directory
    )
    assert (
        decoded.technical_authority_namespace_claim_sha256
        == request.technical_authority_namespace_claim_sha256
    )
    assert (
        decoded.published_technical_authority_lifecycle_binding_sha256
        == request.published_technical_authority_lifecycle_binding_sha256
    )
    bad_q_unsigned = {
        **q_unsigned,
        "technical_authorization_sha256": "not-a-sha256",
    }
    bad_q = {
        **bad_q_unsigned,
        "binding_sha256": canonical_sha256(bad_q_unsigned),
    }
    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="Q static runner binding",
    ):
        build_original_confirmatory_capsule_request_from_authority(
            q_static_runner_binding=bad_q,
            e_scientific_request_projection=e,
            e_intent_core_sha256=request.e_intent_core_sha256,
        )
    flat_mismatch_unsigned = {
        **q_unsigned,
        "technical_authority_artifact_root_sha256": "0" * 64,
    }
    flat_mismatch = {
        **flat_mismatch_unsigned,
        "binding_sha256": canonical_sha256(flat_mismatch_unsigned),
    }
    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="Q static runner binding violates its exact policy",
    ):
        build_original_confirmatory_capsule_request_from_authority(
            q_static_runner_binding=flat_mismatch,
            e_scientific_request_projection=e,
            e_intent_core_sha256=request.e_intent_core_sha256,
        )
    directory_mismatch_unsigned = {
        **q_unsigned,
        "technical_authority_directory": str(
            request.technical_authority_namespace_directory / "substituted-authority"
        ),
    }
    directory_mismatch = {
        **directory_mismatch_unsigned,
        "binding_sha256": canonical_sha256(directory_mismatch_unsigned),
    }
    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="Q static runner binding violates its exact policy",
    ):
        build_original_confirmatory_capsule_request_from_authority(
            q_static_runner_binding=directory_mismatch,
            e_scientific_request_projection=e,
            e_intent_core_sha256=request.e_intent_core_sha256,
        )
    authorization_mismatch_unsigned = {
        **q_unsigned,
        "technical_authorization_sha256": "0" * 64,
    }
    authorization_mismatch = {
        **authorization_mismatch_unsigned,
        "binding_sha256": canonical_sha256(authorization_mismatch_unsigned),
    }
    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="Q static runner binding violates its exact policy",
    ):
        build_original_confirmatory_capsule_request_from_authority(
            q_static_runner_binding=authorization_mismatch,
            e_scientific_request_projection=e,
            e_intent_core_sha256=request.e_intent_core_sha256,
        )
    freeze_parent_mismatch_unsigned = {
        **q_unsigned,
        "freeze_directory": str(request.project_root / "substituted-freeze"),
    }
    freeze_parent_mismatch = {
        **freeze_parent_mismatch_unsigned,
        "binding_sha256": canonical_sha256(freeze_parent_mismatch_unsigned),
    }
    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="Q static runner binding violates its exact policy",
    ):
        build_original_confirmatory_capsule_request_from_authority(
            q_static_runner_binding=freeze_parent_mismatch,
            e_scientific_request_projection=e,
            e_intent_core_sha256=request.e_intent_core_sha256,
        )
    permissive_published_unsigned = {
        key: value for key, value in published_t0.items() if key != "binding_sha256"
    }
    permissive_published_unsigned["adoption_allowed"] = True
    permissive_published = {
        **permissive_published_unsigned,
        "binding_sha256": canonical_sha256(permissive_published_unsigned),
    }
    permissive_q_unsigned = {
        **q_unsigned,
        "published_technical_authority_lifecycle_binding": permissive_published,
    }
    permissive_q = {
        **permissive_q_unsigned,
        "binding_sha256": canonical_sha256(permissive_q_unsigned),
    }
    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="published technical authority lifecycle binding violates its exact one-use policy",
    ):
        build_original_confirmatory_capsule_request_from_authority(
            q_static_runner_binding=permissive_q,
            e_scientific_request_projection=e,
            e_intent_core_sha256=request.e_intent_core_sha256,
        )
    missing_review_attempt_unsigned = {
        key: value
        for key, value in published_t0.items()
        if key not in {"binding_sha256", "review_attempt_claim_sha256"}
    }
    missing_review_attempt = {
        **missing_review_attempt_unsigned,
        "binding_sha256": canonical_sha256(missing_review_attempt_unsigned),
    }
    missing_review_q_unsigned = {
        **q_unsigned,
        "published_technical_authority_lifecycle_binding": missing_review_attempt,
    }
    missing_review_q = {
        **missing_review_q_unsigned,
        "binding_sha256": canonical_sha256(missing_review_q_unsigned),
    }
    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="published technical authority lifecycle binding has an unexpected field set",
    ):
        build_original_confirmatory_capsule_request_from_authority(
            q_static_runner_binding=missing_review_q,
            e_scientific_request_projection=e,
            e_intent_core_sha256=request.e_intent_core_sha256,
        )
    legacy_q_unsigned = {
        **q_unsigned,
        "schema_version": 2,
        "policy": "original_confirmatory_static_runner_binding_v2",
    }
    legacy_q = {
        **legacy_q_unsigned,
        "binding_sha256": canonical_sha256(legacy_q_unsigned),
    }
    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="Q static runner binding violates its exact policy",
    ):
        build_original_confirmatory_capsule_request_from_authority(
            q_static_runner_binding=legacy_q,
            e_scientific_request_projection=e,
            e_intent_core_sha256=request.e_intent_core_sha256,
        )
    with pytest.raises(ConfirmatoryCheckpointContractError, match="E scientific"):
        build_original_confirmatory_capsule_request_from_authority(
            q_static_runner_binding=q,
            e_scientific_request_projection={
                **e,
                "bridge_binding_sha256": "0" * 64,
            },
            e_intent_core_sha256=request.e_intent_core_sha256,
        )


def test_authority_adapter_closed_union_is_selected_only_by_same_q_e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _capsule_request(tmp_path / "runs" / "successor-run")
    predecessor = (request.runs_root / "predecessor-run").resolve()
    first = request.draft_checkpoint_contract.directives[0]
    relative = f"cells/{first.cell_id}/checkpoints/fold_{first.fold_id:02d}.pt"
    source = OriginalConfirmatoryCanonicalECheckpointIdentity(
        path=predecessor / relative,
        physical_identity=ConfirmatoryCheckpointPhysicalIdentity(
            device=1,
            inode=20,
            size_bytes=123,
            mode=stat.S_IFREG | 0o400,
            link_count=1,
            modified_time_ns=2,
            changed_time_ns=2,
        ),
        size_bytes=123,
        sha256="d" * 64,
    )
    successor_projection = _successor_precopy_projection(
        request,
        predecessor_run_directory=predecessor,
        source_identities={(first.cell_id, first.fold_id): source},
    )
    q, fresh_e = _authority_pair(
        request,
        projection=request.canonical_e_projection,
    )
    q_before = json.loads(json.dumps(q))
    same_q, successor_e = _authority_pair(
        request,
        projection=successor_projection,
    )
    assert same_q == q

    def forbidden_filesystem(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("closed-union authority adapter touched the filesystem")

    monkeypatch.setattr(os.path, "lexists", forbidden_filesystem)
    monkeypatch.setattr(Path, "lstat", forbidden_filesystem)
    fresh = build_original_confirmatory_capsule_request_from_authority(
        q_static_runner_binding=q,
        e_scientific_request_projection=fresh_e,
        e_intent_core_sha256=request.e_intent_core_sha256,
    )
    successor = build_original_confirmatory_capsule_request_from_authority(
        q_static_runner_binding=q,
        e_scientific_request_projection=successor_e,
        e_intent_core_sha256=request.e_intent_core_sha256,
    )

    assert q == q_before
    assert fresh.execution_mode == "fresh"
    assert fresh.retry_of_run_id is None
    assert successor.execution_mode == "successor_resume"
    assert successor.retry_of_run_id == predecessor.name
    assert successor.resume_run_directory == predecessor
    assert successor.draft_checkpoint_contract == fresh.draft_checkpoint_contract
    assert successor.canonical_e_projection == successor_projection


def test_successor_e_cannot_change_q_scientific_baseline(tmp_path: Path) -> None:
    request = _capsule_request(tmp_path / "runs" / "successor-run")
    predecessor = (request.runs_root / "predecessor-run").resolve()
    first = request.draft_checkpoint_contract.directives[0]
    relative = f"cells/{first.cell_id}/checkpoints/fold_{first.fold_id:02d}.pt"
    source = OriginalConfirmatoryCanonicalECheckpointIdentity(
        path=predecessor / relative,
        physical_identity=ConfirmatoryCheckpointPhysicalIdentity(
            device=1,
            inode=21,
            size_bytes=123,
            mode=stat.S_IFREG | 0o400,
            link_count=1,
            modified_time_ns=2,
            changed_time_ns=2,
        ),
        size_bytes=123,
        sha256="d" * 64,
    )
    projection = _successor_precopy_projection(
        request,
        predecessor_run_directory=predecessor,
        source_identities={(first.cell_id, first.fold_id): source},
    )
    changed_configuration = dict(projection.directives[0].expected_configuration)
    changed_configuration["seed"] = 999_999
    changed_first = replace(
        projection.directives[0],
        expected_configuration=changed_configuration,
        expected_configuration_sha256=canonical_sha256(changed_configuration),
    )
    changed_values = (changed_first, *projection.directives[1:])
    changed_provisional = replace(
        projection,
        directives=changed_values,
        directives_sha256=canonical_sha256([value.as_dict() for value in changed_values]),
        projection_sha256="0" * 64,
    )
    changed = replace(
        changed_provisional,
        projection_sha256=canonical_sha256(changed_provisional.payload_without_self_hash()),
    )
    q, e = _authority_pair(request, projection=changed)

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match=r"Q draft contract|frozen fresh template|directives",
    ):
        build_original_confirmatory_capsule_request_from_authority(
            q_static_runner_binding=q,
            e_scientific_request_projection=e,
            e_intent_core_sha256=request.e_intent_core_sha256,
        )


def test_successor_materializes_only_after_run_creation_with_o_excl_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from histo_audit.experiment import original_confirmatory_runner_core as module

    request = _capsule_request(tmp_path / "runs" / "successor-run")
    predecessor = (request.runs_root / "predecessor-run").resolve()
    destination = request.expected_run_directory
    first = request.draft_checkpoint_contract.directives[0]
    relative = f"cells/{first.cell_id}/checkpoints/fold_{first.fold_id:02d}.pt"
    source_path = predecessor / relative
    source_path.parent.mkdir(parents=True)
    payload = b"synthetic exact checkpoint bytes"
    source_path.write_bytes(payload)
    source_path.chmod(stat.S_IREAD)
    destination.mkdir(parents=True)
    source_stat = source_path.lstat()
    root_stat = predecessor.lstat()
    source = OriginalConfirmatoryCanonicalECheckpointIdentity(
        path=source_path,
        physical_identity=ConfirmatoryCheckpointPhysicalIdentity.from_stat(source_stat),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    root_identity = OriginalConfirmatoryCanonicalEDirectoryIdentity(
        path=predecessor,
        device=int(root_stat.st_dev),
        inode=int(root_stat.st_ino),
        size_bytes=int(root_stat.st_size),
        mode=int(root_stat.st_mode),
        link_count=int(root_stat.st_nlink),
        modified_time_ns=int(root_stat.st_mtime_ns),
        changed_time_ns=int(root_stat.st_ctime_ns),
    )
    authority = _successor_precopy_projection(
        request,
        predecessor_run_directory=predecessor,
        source_identities={(first.cell_id, first.fold_id): source},
        root_identity=root_identity,
    )
    monkeypatch.setattr(
        module,
        "validate_confirmatory_checkpoint_artifact",
        lambda *_args, **_kwargs: None,
    )

    result = materialize_original_confirmatory_successor_checkpoint_execution(
        authority,
        destination_run_directory=destination,
        fresh_template_contract=request.draft_checkpoint_contract,
    )

    copied = destination / relative
    assert copied.read_bytes() == payload
    assert copied.stat().st_ino != source_path.stat().st_ino
    assert result.execution_contract.execution_mode == "successor_resume"
    assert result.execution_contract.retry_of_run_id == predecessor.name
    assert result.copy_receipt.copied_checkpoint_count == 1
    assert result.copy_receipt.automatic_retry_allowed is False
    assert result.copy_receipt.checkpoint_fallback_used is False
    escaped_identity = replace(
        result.copy_receipt.copied_checkpoints[0],
        path=(tmp_path / "escaped-checkpoint.pt").resolve(),
    )
    escaped_receipt = replace(
        result.copy_receipt,
        copied_checkpoints=(escaped_identity,),
    )
    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="escapes its exact run",
    ):
        escaped_receipt.validate()
    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match=r"failed closed|unreadable",
    ):
        materialize_original_confirmatory_successor_checkpoint_execution(
            authority,
            destination_run_directory=destination,
            fresh_template_contract=request.draft_checkpoint_contract,
        )
    assert copied.read_bytes() == payload


def test_successor_partial_copy_is_retained_and_cannot_be_rearmed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from histo_audit.experiment import original_confirmatory_runner_core as module

    request = _capsule_request(tmp_path / "runs" / "successor-run")
    predecessor = (request.runs_root / "predecessor-run").resolve()
    destination = request.expected_run_directory
    selected = request.draft_checkpoint_contract.directives[:2]
    source_identities: dict[
        tuple[str, int],
        OriginalConfirmatoryCanonicalECheckpointIdentity,
    ] = {}
    source_paths: list[Path] = []
    payloads = (b"first immutable checkpoint", b"second immutable checkpoint")
    for directive, payload in zip(selected, payloads, strict=True):
        relative = f"cells/{directive.cell_id}/checkpoints/fold_{directive.fold_id:02d}.pt"
        source_path = predecessor / relative
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(payload)
        source_path.chmod(stat.S_IREAD)
        observed = source_path.lstat()
        source_identities[(directive.cell_id, directive.fold_id)] = (
            OriginalConfirmatoryCanonicalECheckpointIdentity(
                path=source_path,
                physical_identity=ConfirmatoryCheckpointPhysicalIdentity.from_stat(observed),
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
        source_paths.append(source_path)
    destination.mkdir(parents=True)
    root_stat = predecessor.lstat()
    root_identity = OriginalConfirmatoryCanonicalEDirectoryIdentity(
        path=predecessor,
        device=int(root_stat.st_dev),
        inode=int(root_stat.st_ino),
        size_bytes=int(root_stat.st_size),
        mode=int(root_stat.st_mode),
        link_count=int(root_stat.st_nlink),
        modified_time_ns=int(root_stat.st_mtime_ns),
        changed_time_ns=int(root_stat.st_ctime_ns),
    )
    authority = _successor_precopy_projection(
        request,
        predecessor_run_directory=predecessor,
        source_identities=source_identities,
        root_identity=root_identity,
    )

    def reject_second(path: Path, **_kwargs: Any) -> None:
        if path == source_paths[1]:
            raise ConfirmatoryCheckpointContractError("synthetic second-source validation failure")

    monkeypatch.setattr(
        module,
        "validate_confirmatory_checkpoint_artifact",
        reject_second,
    )

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="synthetic second-source validation failure",
    ):
        materialize_original_confirmatory_successor_checkpoint_execution(
            authority,
            destination_run_directory=destination,
            fresh_template_contract=request.draft_checkpoint_contract,
        )

    first_relative = authority.directives[0].destination_checkpoint_relative_path
    second_relative = authority.directives[1].destination_checkpoint_relative_path
    first_destination = destination / first_relative
    assert first_destination.read_bytes() == payloads[0]
    assert not (destination / second_relative).exists()
    with pytest.raises(ConfirmatoryCheckpointContractError):
        materialize_original_confirmatory_successor_checkpoint_execution(
            authority,
            destination_run_directory=destination,
            fresh_template_contract=request.draft_checkpoint_contract,
        )
    assert first_destination.read_bytes() == payloads[0]


def test_capsule_directly_binds_qe_and_hardcodes_canonical_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from histo_audit.experiment import confirmatory_runner as lifecycle_module

    request = _capsule_request(tmp_path / "capsule")
    observed: list[Any] = []

    def stop_before_science(
        supplied: OriginalConfirmatoryCapsuleExecutionRequest,
    ) -> dict[str, Any]:
        observed.append(supplied)
        assert supplied is request
        raise RuntimeError("synthetic stop before science")

    monkeypatch.setattr(
        lifecycle_module,
        "_execute_original_confirmatory_capsule_lifecycle",
        stop_before_science,
    )
    with pytest.raises(RuntimeError, match="stop before science"):
        _run_original_confirmatory_capsule_request(request)
    assert len(observed) == 1
    assert not request.expected_run_directory.exists()
    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="already consumed",
    ):
        _run_original_confirmatory_capsule_request(request)


def test_guarded_matrix_point_binds_exact_e_and_has_no_callback_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from histo_audit.experiment import original_confirmatory_runner_core as module

    high_level = _capsule_request(tmp_path / "capsule")
    controls = confirmatory_execution_controls_from_frozen_config(complete_confirmatory_config())
    prepared = module._OriginalConfirmatoryPreparedMatrixRequest(
        rotations=tuple(_synthetic_rotation(fold, controls) for fold in controls.official_folds),
        plan=controls.plan,
        controls=controls,
        output_directory=high_level.expected_run_directory,
        frozen_blockers=tuple(sorted(_blockers(controls).items())),
        artifact_scope=REAL_CONFIRMATORY_ARTIFACT_SCOPE,
        gate_evidence=high_level.gate_evidence,
        canonical_e_projection=high_level.canonical_e_projection,
        scientific_authority_projection_sha256=canonical_sha256(
            high_level.canonical_e_projection.as_dict()
        ),
        draft_checkpoint_contract=high_level.draft_checkpoint_contract,
    )
    observed: list[Any] = []

    def stop_before_science(
        rotations: Any,
        plan: Any,
        supplied_controls: Any,
        **kwargs: Any,
    ) -> Any:
        observed.append((rotations, plan, supplied_controls, kwargs))
        contract = kwargs["checkpoint_execution_contract"]
        contract.validate(expected_directive_count=180)
        assert contract.authority_projection_sha256 == canonical_sha256(
            high_level.canonical_e_projection.as_dict()
        )
        assert (
            contract.authority_fit_directives_root_sha256
            == high_level.canonical_e_projection.directives_sha256
        )
        assert contract.authority_checkpoint_allowlist_sha256 == canonical_sha256(
            sorted(
                f"cells/{value.cell_id}/checkpoints/fold_{value.fold_id:02d}.pt"
                for value in high_level.draft_checkpoint_contract.directives
            )
        )
        assert kwargs["frozen_oof_runner"] is module.run_confirmatory_frozen_feature_oof
        assert kwargs["image_oof_runner"] is module.run_confirmatory_image_oof
        assert kwargs["cpu_test_only"] is False
        assert kwargs["progress_callback"] is None
        raise RuntimeError("synthetic stop at fixed matrix point")

    monkeypatch.setattr(module, "execute_confirmatory_matrix", stop_before_science)
    with pytest.raises(RuntimeError, match="fixed matrix point"):
        module._execute_original_confirmatory_prepared_matrix(prepared)
    assert len(observed) == 1
    assert not high_level.expected_run_directory.exists()


def test_capsule_rejects_run_directory_mismatch_before_lifecycle(
    tmp_path: Path,
) -> None:
    output = (tmp_path / "fresh-output").resolve()
    request = _capsule_request(output)

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="run identity",
    ):
        _run_original_confirmatory_capsule_request(
            replace(request, expected_run_directory=output.with_name("other"))
        )
    assert not output.exists()


def test_capsule_rejects_tampered_e_bound_draft_and_noncanonical_path(
    tmp_path: Path,
) -> None:
    request = _capsule_request(tmp_path / "capsule")
    first = replace(
        request.canonical_e_projection.directives[0],
        action="restore_terminal_checkpoint_without_fit",
    )
    changed_directives = (first, *request.canonical_e_projection.directives[1:])
    tampered_projection = replace(
        request.canonical_e_projection,
        directives=changed_directives,
        directives_sha256=canonical_sha256([value.as_dict() for value in changed_directives]),
    )
    with pytest.raises(ConfirmatoryCheckpointContractError, match="canonical E"):
        _run_original_confirmatory_capsule_request(
            replace(request, canonical_e_projection=tampered_projection)
        )
    assert not request.expected_run_directory.exists()

    bound = _register_checkpoint_execution_contract(
        replace(
            request.draft_checkpoint_contract,
            authority_projection_sha256="1" * 64,
            authority_fit_directives_root_sha256="2" * 64,
            authority_checkpoint_allowlist_sha256="3" * 64,
        ),
        expected_directive_count=180,
    )
    with pytest.raises(ConfirmatoryCheckpointContractError, match="unbound exact-180"):
        _run_original_confirmatory_capsule_request(
            replace(request, draft_checkpoint_contract=bound)
        )

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="lexically canonical",
    ):
        _run_original_confirmatory_capsule_request(
            replace(
                request,
                crop_cache_path=request.project_root / "cache" / ".." / "crop-cache",
            )
        )
    assert not request.expected_run_directory.exists()


def test_successor_canonical_e_carries_exact_capsule_lineage(tmp_path: Path) -> None:
    draft = _successor_draft(tmp_path)
    projection = build_original_confirmatory_canonical_e_checkpoint_projection(draft)

    projection.validate()
    assert projection.execution_mode == "successor_resume"
    assert projection.predecessor_snapshot_sha256 == "d" * 64
    assert projection.predecessor_copy_receipt_sha256 == "e" * 64
    assert require_original_confirmatory_canonical_e_projection(projection, draft) is draft


def test_original_grouped_oof_rejects_unbound_draft_before_inputs() -> None:
    draft = build_original_confirmatory_fresh_checkpoint_execution_contract(_original_contract())

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="unbound checkpoint-authority draft",
    ):
        grouped_oof_confirmatory_cnn(
            None,  # type: ignore[arg-type]
            (),
            (),
            (),
            sample_ids=(),
            audit_target_masks=None,
            reference_validation_rgb=None,  # type: ignore[arg-type]
            reference_validation_labels=(),
            reference_validation_sample_ids=(),
            reference_validation_group_ids=(),
            reference_validation_target_masks=None,
            final_reference_group_ids=(),
            base_config=ConfirmatoryCNNConfig(
                input_variant="context_rgb",
                epochs=3,
            ),
            cell_id="cnn_cell_000",
            checkpoint_directory="unused",
            checkpoint_execution_contract=draft,
            cpu_test_only=False,
        )


def test_real_matrix_rejects_unbound_draft_before_artifacts(tmp_path: Path) -> None:
    config = complete_confirmatory_config()
    controls = confirmatory_execution_controls_from_frozen_config(config)
    all_ids = tuple(sorted(cell.cell_id for cell in controls.plan.cells))
    cnn_ids = tuple(
        sorted(
            cell.cell_id
            for cell in controls.plan.cells
            if controls.scenarios_by_id[cell.scenario_id].family == "cnn"
        )
    )
    draft = build_original_confirmatory_fresh_checkpoint_execution_contract(
        _original_contract(
            all_cell_ids=all_ids,
            cnn_cell_ids=cnn_ids,
            maximum_epochs=controls.max_epochs,
        )
    )
    output = tmp_path / "unbound-matrix"

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="unbound checkpoint-authority draft",
    ):
        execute_confirmatory_matrix(
            tuple(_synthetic_rotation(fold, controls) for fold in controls.official_folds),
            controls.plan,
            controls,
            output_directory=output,
            frozen_oof_runner=_synthetic_frozen_runner,
            frozen_blockers=_blockers(controls),
            artifact_scope=REAL_CONFIRMATORY_ARTIFACT_SCOPE,
            cpu_test_only=False,
            checkpoint_execution_contract=draft,
        )

    assert not (output / "matrix_plan.json").exists()


def test_direct_caller_rehash_cannot_change_registered_execution() -> None:
    execution = build_original_confirmatory_fresh_checkpoint_execution_contract(
        _original_contract()
    )
    reversed_directives = tuple(reversed(execution.directives))
    caller_rehashed = replace(
        execution,
        directives=reversed_directives,
        directives_sha256=canonical_sha256([value.as_dict() for value in reversed_directives]),
    )

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="not derived by the sealed builder",
    ):
        caller_rehashed.validate(expected_directive_count=180)


def test_data_plane_has_no_predecessor_or_outcome_path_read_surface() -> None:
    fresh_names = set(
        inspect.signature(
            build_original_confirmatory_fresh_checkpoint_execution_contract
        ).parameters
    )
    consume_names = set(
        inspect.signature(
            build_original_confirmatory_successor_checkpoint_execution_contract
        ).parameters
    )
    prepare_names = set(
        inspect.signature(prepare_original_confirmatory_successor_checkpoint_execution).parameters
    )
    matrix_names = set(inspect.signature(execute_confirmatory_matrix).parameters)

    assert fresh_names == {"contract"}
    assert consume_names == {"prepared"}
    assert prepare_names == {"request", "contract"}
    assert "predecessor_run_directory" not in matrix_names
    assert "checkpoint_root" not in matrix_names
    assert "resume_checkpoints" not in matrix_names


def test_checkpoint_contract_error_escapes_per_cell_failure_demotion(
    tmp_path: Path,
) -> None:
    config = complete_confirmatory_config()
    controls = confirmatory_execution_controls_from_frozen_config(config)
    rotations = tuple(_synthetic_rotation(fold, controls) for fold in controls.official_folds)

    def structural_failure(_request: ConfirmatoryCellRequest) -> Any:
        raise ConfirmatoryCheckpointContractError("structural checkpoint breach")

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="structural checkpoint breach",
    ):
        execute_confirmatory_matrix(
            rotations,
            controls.plan,
            controls,
            output_directory=tmp_path / "structural-failure",
            frozen_oof_runner=_synthetic_frozen_runner,
            image_oof_runner=structural_failure,
            frozen_blockers=_blockers(controls),
        )
