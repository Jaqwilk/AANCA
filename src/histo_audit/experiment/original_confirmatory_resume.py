"""Bounded checkpoint successor for the frozen 108-cell confirmatory study.

The module is deliberately narrower than the confirmatory runner.  It has no
training, OOF, metric, ranking, outcome, finalisation, retry, or run-discovery
path.  It:

* proves that one caller-supplied plan is the exact frozen 108-cell plan;
* derives the complete 180 CNN-fold checkpoint allowlist before reading a
  predecessor;
* inspects only canonical checkpoint paths from one explicit predecessor;
* validates every present checkpoint with the production Torch validator;
* distinguishes incomplete fits from terminal checkpoints;
* physically copies validated files into one new explicit successor; and
* returns machine-readable evidence for the caller to persist in that new run.

Missing checkpoints are explicit fresh-fit decisions.  A terminal checkpoint is
restored for inference/evidence reconstruction but is never resumed for training.
Only an incomplete checkpoint may continue fitting.  Source annotations and
scientific outcome artifacts are outside this module's read surface.
"""

from __future__ import annotations

import csv
import ctypes
import hashlib
import json
import os
import platform
import re
import stat
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import psutil  # type: ignore[import-untyped]

from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.experiment.confirmatory_core import (
    ConfirmatoryExecutionControls,
    confirmatory_cnn_config_for_cell,
)
from histo_audit.experiment.resource_bounded_resume import (
    ReadOnlyPredecessorSnapshot,
    ResourceBoundedResumeCopyReceipt,
    ResourceBoundedResumeError,
    ResumeCheckpointExpectation,
    ResumePathIdentity,
    StrictCheckpointValidator,
    build_resource_bounded_resume_evidence,
    copy_validated_resume_checkpoints,
    inspect_read_only_resume_predecessor,
)
from histo_audit.models.cnn import (
    OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER,
    validate_confirmatory_checkpoint_artifact,
)
from histo_audit.representations.imagenet import official_resnet18_weight_cache_path
from histo_audit.utils.run_tracking import (
    ARTIFACT_MANIFEST_FILENAME,
    IMMUTABLE_MARKER,
    RUN_DISPOSITION_REGISTRY_FILENAME,
    RUN_STAGE_ATTESTATION_REGISTRY_FILENAME,
    capture_source_tree,
    read_run_dispositions,
    read_run_stage_attestations,
    sha256_file,
    verify_run_integrity,
)

ORIGINAL_CONFIRMATORY_RESUME_POLICY = "explicit_original_confirmatory_checkpoint_successor_v1"
ORIGINAL_CONFIRMATORY_COPY_POLICY = "physical_no_overwrite_no_link_checkpoint_copy_v1"
ORIGINAL_CONFIRMATORY_READ_SCOPE = (
    "canonical_checkpoint_state_only_no_oof_metrics_rankings_or_outcomes"
)
ORIGINAL_CONFIRMATORY_CONFIG_SEMANTIC_SHA256 = (
    "ff2ce8d5043813b08db23efe797abe444ba6b6bde292810a094d78757f74460b"
)
ORIGINAL_CONFIRMATORY_PLAN_SEMANTIC_SHA256 = (
    "c1993d4403982814a7259c524bbd21784537b7634e49a4f7150a9ca4de3c2c87"
)
ORIGINAL_CONFIRMATORY_CONTROLS_BINDING_SHA256 = (
    "7d529695a718ed07e09bbf338023bcdf579ce08f3b08933fd0b8eecddba1cb3f"
)
ORIGINAL_CONFIRMATORY_WEIGHT_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)
ORIGINAL_CONFIRMATORY_PLANNED_CELL_COUNT = 108
ORIGINAL_CONFIRMATORY_REQUIRED_CELL_COUNT = 90
ORIGINAL_CONFIRMATORY_OPTIONAL_CELL_COUNT = 18
ORIGINAL_CONFIRMATORY_CNN_CELL_COUNT = 36
ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT = 180
ORIGINAL_CONFIRMATORY_OOF_FOLD_COUNT = 5
ORIGINAL_CONFIRMATORY_RESUME_EVIDENCE_SCHEMA_VERSION = 1
ORIGINAL_CONFIRMATORY_RESUME_EVIDENCE_HASH_FIELD = "evidence_without_self_hash_sha256"
ORIGINAL_CONFIRMATORY_AUTHORITY_POLICY = (
    "original_confirmatory_checkpoint_successor_exact_authority_v1"
)
ORIGINAL_CONFIRMATORY_AUTHORIZATION_FILENAME = "original_confirmatory_resume_authorization.json"
ORIGINAL_CONFIRMATORY_Q_GATE_ADAPTER_POLICY = "opaque_original_confirmatory_q_dual_gate_adapter_v1"
ORIGINAL_CONFIRMATORY_ORPHAN_POLICY = "manual_diagnostic_unsealed_interrupted_orphan_v1"
ORIGINAL_CONFIRMATORY_FAILED_POLICY = "sealed_failed_demoted_confirmatory_predecessor_v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_NONCE = re.compile(r"^[0-9a-f]{32}$")
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,239}$")
_SAVED_SESSION_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_REPARSE_POINT = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
_DATA_AND_SPLIT_KEYS = frozenset(
    {
        "training_data_sha256",
        "reference_validation_data_sha256",
        "training_split_sha256",
        "reference_validation_split_sha256",
    }
)
_EXPECTED_OFFICIAL_FOLDS = frozenset({1, 2, 3})
_EXPECTED_MODEL_SEEDS = frozenset({303, 304, 305})
_EXPECTED_CORRUPTION_CELLS = frozenset({"clean_reference_cell", "confusion_targeted_ten_percent"})
_EXPECTED_SCENARIOS = frozenset(
    {
        "cnn_context_rgb",
        "cnn_context_target_mask",
        "imagenet_frozen_logistic",
        "imagenet_frozen_target_highlighted_logistic",
        "imagenet_frozen_context_morphometrics_logistic",
        "pathology_frozen_logistic",
    }
)
_EXPECTED_CNN_SCENARIOS = frozenset({"cnn_context_rgb", "cnn_context_target_mask"})
_EXPERIMENT_NAME = "pannuke_confirmatory_study"
_REAL_ARTIFACT_SCOPE = "real_pannuke_confirmatory_study"
_ORPHAN_FORBIDDEN_TERMINAL_ARTIFACTS = frozenset(
    {
        ARTIFACT_MANIFEST_FILENAME,
        IMMUTABLE_MARKER,
        "completion_evidence.json",
        "core_completion_evidence.json",
        "metrics.json",
        "runtime.json",
        "traceback.txt",
        "confirmatory_stage_eligibility.json",
        "confirmatory_stage_attestation.json",
    }
)
_ROOT_LOCK_NAMES = (
    ".integrity_registry.jsonl.lock",
    ".registry.csv.lock",
    ".run_dispositions.jsonl.lock",
    ".run_stage_attestations.jsonl.lock",
)
_SANITIZED_ENVIRONMENT_NAMES = (
    "AANCA_SAVED_CODEX_SESSION_ID",
    "AANCA_SUPERVISOR_RELEASE_ID",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "LOCALAPPDATA",
    "MKL_NUM_THREADS",
    "NVIDIA_TF32_OVERRIDE",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
)
_SUPERVISOR_ATTEMPT_NONCE_NAME = "AANCA_SUPERVISOR_ATTEMPT_NONCE"
_EXPECTED_SUPERVISOR_ENVIRONMENT_NAMES = frozenset(
    {"AANCA_SUPERVISOR_RELEASE_ID", _SUPERVISOR_ATTEMPT_NONCE_NAME}
)
_SUPERVISOR_MANIFEST_POLICY = "aanca_event_driven_supervisor_manifest_v1"
_SUPERVISOR_HANDOFF_POLICY = "aanca_event_driven_supervisor_handoff_v1"
_SUPERVISOR_RELEASE_ID = "original-confirmatory-resume-v1"
_CONTINUATION_PROMPT = (
    "Read STATUS.md and PLAN.md in full again. Verify the completed supervised "
    "process and its exact seal/integrity evidence. Continue only the next legal "
    "outcome-blind step. Never retry a scientific operation automatically."
)


class OriginalConfirmatoryResumeError(ResourceBoundedResumeError):
    """Fail-closed original-confirmatory successor contract violation."""


class CheckpointFitStateReader(Protocol):
    """Read only the structural fit state after strict checkpoint validation."""

    def __call__(
        self,
        path: str | Path,
        *,
        expected_configuration: Mapping[str, Any],
    ) -> tuple[int, bool]: ...


PredecessorClass = Literal[
    "sealed_failed_demoted",
    "unsealed_interrupted_orphan",
]


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryRuntimePins:
    """Exact runtime/supervisor pins published by the separate authority adapter."""

    project_root: Path
    execution_source_root_sha256: str
    execution_source_manifest_sha256: str
    project_plan_path: Path
    project_plan_sha256: str
    project_plan_root_sha256: str
    command_argv: tuple[str, ...]
    command_argv_sha256: str
    process_orig_argv: tuple[str, ...]
    process_orig_argv_sha256: str
    raw_process_command_line: str
    raw_process_command_line_sha256: str
    working_directory: Path
    interpreter_path: Path
    interpreter_sha256: str
    runtime_identity_sha256: str
    sanitized_environment: tuple[tuple[str, str | None], ...]
    environment_sha256: str
    supervisor_root: Path
    supervisor_release_path: Path
    supervisor_release_sha256: str
    supervisor_manifest_path: Path
    supervisor_manifest_sha256: str
    supervisor_handoff_path: Path
    supervisor_handoff_sha256: str
    supervisor_job_id: str
    supervisor_launch_intent_path: Path
    supervisor_launch_intent_sha256: str
    supervisor_process_id: int | None
    supervisor_process_create_time_unix_us: int | None
    saved_codex_session_id: str
    saved_codex_session_binding_sha256: str


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryRuntimeReceipt:
    """Fresh verification of the exact command, source, plan, and handoff."""

    project_root: Path
    execution_source_root_sha256: str
    execution_source_manifest_sha256: str
    project_plan_path: Path
    project_plan_sha256: str
    project_plan_root_sha256: str
    command_argv: tuple[str, ...]
    command_argv_sha256: str
    process_orig_argv: tuple[str, ...]
    process_orig_argv_sha256: str
    raw_process_command_line: str
    raw_process_command_line_sha256: str
    working_directory: Path
    interpreter_path: Path
    interpreter_sha256: str
    runtime_identity_sha256: str
    sanitized_environment: tuple[tuple[str, str | None], ...]
    environment_sha256: str
    supervisor_root: Path
    supervisor_release_path: Path
    supervisor_release_sha256: str
    supervisor_manifest_path: Path
    supervisor_manifest_sha256: str
    supervisor_handoff_path: Path
    supervisor_handoff_sha256: str
    supervisor_job_id: str
    supervisor_launch_intent_path: Path
    supervisor_launch_intent_sha256: str
    supervisor_attempt_nonce: str
    supervisor_process_id: int | None
    supervisor_process_create_time_unix_us: int | None
    saved_codex_session_id: str
    saved_codex_session_binding_sha256: str
    receipt_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "execution_source_root_sha256": (self.execution_source_root_sha256),
            "execution_source_manifest_sha256": (self.execution_source_manifest_sha256),
            "project_plan_path": str(self.project_plan_path),
            "project_plan_sha256": self.project_plan_sha256,
            "project_plan_root_sha256": self.project_plan_root_sha256,
            "command_argv": list(self.command_argv),
            "command_argv_sha256": self.command_argv_sha256,
            "process_orig_argv": list(self.process_orig_argv),
            "process_orig_argv_sha256": self.process_orig_argv_sha256,
            "raw_process_command_line": self.raw_process_command_line,
            "raw_process_command_line_sha256": (self.raw_process_command_line_sha256),
            "working_directory": str(self.working_directory),
            "interpreter_path": str(self.interpreter_path),
            "interpreter_sha256": self.interpreter_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "sanitized_environment": [
                {"name": name, "value": value} for name, value in self.sanitized_environment
            ],
            "environment_sha256": self.environment_sha256,
            "supervisor_root": str(self.supervisor_root),
            "supervisor_release_path": str(self.supervisor_release_path),
            "supervisor_release_sha256": self.supervisor_release_sha256,
            "supervisor_manifest_path": str(self.supervisor_manifest_path),
            "supervisor_manifest_sha256": self.supervisor_manifest_sha256,
            "supervisor_handoff_path": str(self.supervisor_handoff_path),
            "supervisor_handoff_sha256": self.supervisor_handoff_sha256,
            "supervisor_job_id": self.supervisor_job_id,
            "supervisor_launch_intent_path": str(self.supervisor_launch_intent_path),
            "supervisor_launch_intent_sha256": (self.supervisor_launch_intent_sha256),
            "supervisor_attempt_nonce": self.supervisor_attempt_nonce,
            "supervisor_process_id": self.supervisor_process_id,
            "supervisor_process_create_time_unix_us": (self.supervisor_process_create_time_unix_us),
            "saved_codex_session_id": self.saved_codex_session_id,
            "saved_codex_session_binding_sha256": (self.saved_codex_session_binding_sha256),
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryQGateAdapterPins:
    """Opaque pins emitted by the independently verified Q dual gate."""

    q_directory: Path
    q_tree_root_sha256: str
    q_authorization_path: Path
    q_authorization_sha256: str
    q_dual_gate_binding_sha256: str
    historical_p_authority_directory: Path
    historical_p_authority_root_sha256: str
    historical_primary_run_directory: Path
    historical_primary_run_id: str
    historical_primary_artifact_root_sha256: str
    confirmatory_config_semantic_sha256: str
    confirmatory_plan_semantic_sha256: str
    confirmatory_controls_binding_sha256: str


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryQGateAdapterReceipt:
    """Identity/hash-only receipt for the separately validated Q dual gate."""

    q_directory: Path
    q_directory_identity: ResumePathIdentity
    q_tree_root_sha256: str
    q_authorization_path: Path
    q_authorization_sha256: str
    q_dual_gate_binding_sha256: str
    historical_p_authority_directory: Path
    historical_p_authority_identity: ResumePathIdentity
    historical_p_authority_root_sha256: str
    historical_primary_run_directory: Path
    historical_primary_run_identity: ResumePathIdentity
    historical_primary_run_id: str
    historical_primary_artifact_root_sha256: str
    confirmatory_config_semantic_sha256: str
    confirmatory_plan_semantic_sha256: str
    confirmatory_controls_binding_sha256: str
    adapter_receipt_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": ORIGINAL_CONFIRMATORY_Q_GATE_ADAPTER_POLICY,
            "q_directory": str(self.q_directory),
            "q_directory_identity": asdict(self.q_directory_identity),
            "q_tree_root_sha256": self.q_tree_root_sha256,
            "q_authorization_path": str(self.q_authorization_path),
            "q_authorization_sha256": self.q_authorization_sha256,
            "q_dual_gate_binding_sha256": self.q_dual_gate_binding_sha256,
            "historical_p_authority_directory": str(self.historical_p_authority_directory),
            "historical_p_authority_identity": asdict(self.historical_p_authority_identity),
            "historical_p_authority_root_sha256": (self.historical_p_authority_root_sha256),
            "historical_primary_run_directory": str(self.historical_primary_run_directory),
            "historical_primary_run_identity": asdict(self.historical_primary_run_identity),
            "historical_primary_run_id": self.historical_primary_run_id,
            "historical_primary_artifact_root_sha256": (
                self.historical_primary_artifact_root_sha256
            ),
            "confirmatory_config_semantic_sha256": (self.confirmatory_config_semantic_sha256),
            "confirmatory_plan_semantic_sha256": (self.confirmatory_plan_semantic_sha256),
            "confirmatory_controls_binding_sha256": (self.confirmatory_controls_binding_sha256),
            "adapter_receipt_sha256": self.adapter_receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryResumeAuthorityPins:
    """Caller pins for independently published amendment and exact authority."""

    technical_amendment_directory: Path
    technical_amendment_tree_root_sha256: str
    execution_authority_directory: Path
    execution_authority_tree_root_sha256: str
    authorization_file_sha256: str
    runtime_pins: OriginalConfirmatoryRuntimePins
    q_gate_adapter_pins: OriginalConfirmatoryQGateAdapterPins


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryResumeAuthorityReceipt:
    """Read-only verification receipt for the separately published authority."""

    technical_amendment_directory: Path
    technical_amendment_identity: ResumePathIdentity
    technical_amendment_tree_root_sha256: str
    execution_authority_directory: Path
    execution_authority_identity: ResumePathIdentity
    execution_authority_tree_root_sha256: str
    authorization_file_sha256: str
    authorization_binding_sha256: str
    runtime: OriginalConfirmatoryRuntimeReceipt
    q_gate_adapter: OriginalConfirmatoryQGateAdapterReceipt

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": ORIGINAL_CONFIRMATORY_AUTHORITY_POLICY,
            "technical_amendment_directory": str(self.technical_amendment_directory),
            "technical_amendment_identity": asdict(self.technical_amendment_identity),
            "technical_amendment_tree_root_sha256": (self.technical_amendment_tree_root_sha256),
            "execution_authority_directory": str(self.execution_authority_directory),
            "execution_authority_identity": asdict(self.execution_authority_identity),
            "execution_authority_tree_root_sha256": (self.execution_authority_tree_root_sha256),
            "authorization_file_sha256": self.authorization_file_sha256,
            "authorization_binding_sha256": self.authorization_binding_sha256,
            "runtime": self.runtime.as_dict(),
            "q_gate_adapter": self.q_gate_adapter.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryResumeRequest:
    """One explicit predecessor and one never-existing successor destination."""

    runs_root: Path
    predecessor_run_directory: Path
    successor_run_directory: Path
    retry_of_run_id: str
    successor_run_id: str
    predecessor_class: PredecessorClass
    authority_pins: OriginalConfirmatoryResumeAuthorityPins
    predecessor_process_id: int | None = None
    predecessor_process_create_time_unix_us: int | None = None
    orphan_manual_diagnosis: bool = False
    supervisor_automatic_transition_allowed: bool = False
    max_attempt_count: int = 1
    automatic_retry_allowed: bool = False
    predecessor_autodiscovery_allowed: bool = False


@dataclass(frozen=True, slots=True)
class BoundOriginalConfirmatoryResumeRequest:
    """Canonical request frozen before RunTracker creation."""

    runs_root: Path
    predecessor_run_directory: Path
    successor_run_directory: Path
    retry_of_run_id: str
    successor_run_id: str
    predecessor_class: PredecessorClass
    authority: OriginalConfirmatoryResumeAuthorityReceipt
    predecessor_process_id: int | None
    predecessor_process_create_time_unix_us: int | None
    orphan_manual_diagnosis: bool
    predecessor_identity: ResumePathIdentity
    request_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "runs_root": str(self.runs_root),
            "predecessor_run_directory": str(self.predecessor_run_directory),
            "successor_run_directory": str(self.successor_run_directory),
            "retry_of_run_id": self.retry_of_run_id,
            "successor_run_id": self.successor_run_id,
            "predecessor_class": self.predecessor_class,
            "authority": self.authority.as_dict(),
            "predecessor_process_id": self.predecessor_process_id,
            "predecessor_process_create_time_unix_us": (
                self.predecessor_process_create_time_unix_us
            ),
            "orphan_manual_diagnosis": self.orphan_manual_diagnosis,
            "supervisor_automatic_transition_allowed": False,
            "predecessor_identity": asdict(self.predecessor_identity),
            "max_attempt_count": 1,
            "automatic_retry_allowed": False,
            "predecessor_autodiscovery_allowed": False,
            "request_sha256": self.request_sha256,
        }


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryResumeContract:
    """Exact frozen science and complete CNN-fold allowlist."""

    plan_semantic_sha256: str
    config_semantic_sha256: str
    controls_binding_sha256: str
    all_cell_ids: tuple[str, ...]
    cnn_cell_ids: tuple[str, ...]
    checkpoint_expectations: tuple[ResumeCheckpointExpectation, ...]
    checkpoint_allowlist_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": ORIGINAL_CONFIRMATORY_RESUME_POLICY,
            "plan_semantic_sha256": self.plan_semantic_sha256,
            "config_semantic_sha256": self.config_semantic_sha256,
            "controls_binding_sha256": self.controls_binding_sha256,
            "planned_cell_count": len(self.all_cell_ids),
            "required_cell_count": ORIGINAL_CONFIRMATORY_REQUIRED_CELL_COUNT,
            "optional_cell_count": ORIGINAL_CONFIRMATORY_OPTIONAL_CELL_COUNT,
            "cnn_cell_count": len(self.cnn_cell_ids),
            "checkpoint_count": len(self.checkpoint_expectations),
            "oof_fold_count": ORIGINAL_CONFIRMATORY_OOF_FOLD_COUNT,
            "all_cell_ids_sha256": canonical_sha256(list(self.all_cell_ids)),
            "cnn_cell_ids_sha256": canonical_sha256(list(self.cnn_cell_ids)),
            "checkpoint_allowlist_sha256": self.checkpoint_allowlist_sha256,
        }


FitAction = Literal[
    "fresh_fit",
    "resume_incomplete_fit",
    "restore_terminal_checkpoint_without_fit",
]


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryCheckpointState:
    """Outcome-blind structural action for one exact CNN fold."""

    relative_path: str
    cell_id: str
    fold_id: int
    action: FitAction
    completed_epochs: int | None
    maximum_epochs: int
    stopped_early: bool | None
    source_size_bytes: int | None
    source_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryProcessReceipt:
    """Exact dead/reused predecessor-process observation."""

    expected_process_id: int | None
    expected_create_time_unix_us: int | None
    expected_identity_state: Literal[
        "not_supplied",
        "gone",
        "pid_reused",
    ]
    inspector_process_id: int
    inspector_create_time_unix_us: int
    exact_inspector_match_excluded: bool
    exact_authorized_supervisor_match_excluded: bool
    matching_predecessor_process_ids: tuple[int, ...]
    inspected_process_count: int
    receipt_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryPredecessorQualification:
    """Fail-closed lineage qualification finalized after stable inspection."""

    predecessor_class: PredecessorClass
    policy: str
    terminal_status: str
    sealed_integrity_valid: bool
    integrity_registry_record_present: bool
    artifact_root_sha256: str | None
    artifact_manifest_sha256: str | None
    immutable_marker_sha256: str | None
    status_sha256: str
    completion_evidence_sha256: str | None
    process_receipt: OriginalConfirmatoryProcessReceipt
    active_lock_paths_before: tuple[str, ...]
    active_lock_paths_after: tuple[str, ...]
    stage_attestation_count: int
    disposition_record_count: int
    root_inventory_before_sha256: str
    root_inventory_after_sha256: str
    qualification_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "predecessor_class": self.predecessor_class,
            "policy": self.policy,
            "terminal_status": self.terminal_status,
            "sealed_integrity_valid": self.sealed_integrity_valid,
            "integrity_registry_record_present": (self.integrity_registry_record_present),
            "artifact_root_sha256": self.artifact_root_sha256,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "immutable_marker_sha256": self.immutable_marker_sha256,
            "status_sha256": self.status_sha256,
            "completion_evidence_sha256": self.completion_evidence_sha256,
            "process_receipt": self.process_receipt.as_dict(),
            "active_lock_paths_before": list(self.active_lock_paths_before),
            "active_lock_paths_after": list(self.active_lock_paths_after),
            "stage_attestation_count": self.stage_attestation_count,
            "disposition_record_count": self.disposition_record_count,
            "root_inventory_before_sha256": (self.root_inventory_before_sha256),
            "root_inventory_after_sha256": self.root_inventory_after_sha256,
            "qualification_sha256": self.qualification_sha256,
        }


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryPredecessorSnapshot:
    """Validated source snapshot plus exact no-training/resume decisions."""

    request: BoundOriginalConfirmatoryResumeRequest
    contract: OriginalConfirmatoryResumeContract
    qualification: OriginalConfirmatoryPredecessorQualification
    base_snapshot: ReadOnlyPredecessorSnapshot
    checkpoint_states: tuple[OriginalConfirmatoryCheckpointState, ...]
    checkpoint_tree_before_sha256: str
    checkpoint_tree_after_sha256: str
    snapshot_sha256: str

    @property
    def imported_checkpoint_count(self) -> int:
        return sum(state.source_sha256 is not None for state in self.checkpoint_states)

    @property
    def incomplete_fit_count(self) -> int:
        return sum(state.action == "resume_incomplete_fit" for state in self.checkpoint_states)

    @property
    def terminal_restore_count(self) -> int:
        return sum(
            state.action == "restore_terminal_checkpoint_without_fit"
            for state in self.checkpoint_states
        )

    @property
    def fresh_fit_count(self) -> int:
        return sum(state.action == "fresh_fit" for state in self.checkpoint_states)


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryResumeCopyReceipt:
    """Physical-copy receipt bound to the exact frozen-science snapshot."""

    snapshot_sha256: str
    base_receipt: ResourceBoundedResumeCopyReceipt
    destination_checkpoint_tree_sha256: str


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryFitDirective:
    """One exact fold action consumed by the successor integration adapter."""

    relative_path: str
    cell_id: str
    fold_id: int
    action: FitAction
    checkpoint_sha256: str | None
    checkpoint_must_exist: bool
    pass_resume_true_to_fit: bool
    training_required: bool
    next_epoch_index: int
    maximum_epochs: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OriginalConfirmatorySuccessorExecutionContract:
    """Exact matrix integration settings after the physical import receipt."""

    snapshot_sha256: str
    destination_checkpoint_tree_sha256: str
    resume_checkpoints: Literal[True]
    directives: tuple[OriginalConfirmatoryFitDirective, ...]
    directives_sha256: str

    def confirmatory_matrix_kwargs(self) -> dict[str, bool]:
        """Return the only legal checkpoint flag for the existing matrix executor."""

        return {"resume_checkpoints": True}


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _is_reparse(path: Path, value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    return path.is_symlink() or stat.S_ISLNK(value.st_mode) or bool(attributes & _REPARSE_POINT)


def _plain_directory_identity(path: Path, *, role: str) -> ResumePathIdentity:
    path = _lexical_absolute(path)
    if not path.anchor:
        raise OriginalConfirmatoryResumeError(f"{role} must be absolute")
    current = Path(path.parts[0])
    final: os.stat_result | None = None
    for index, component in enumerate(path.parts):
        if index:
            current /= component
        try:
            observed = current.lstat()
        except OSError as exc:
            raise OriginalConfirmatoryResumeError(
                f"{role} has a missing or inaccessible component: {current}"
            ) from exc
        if not stat.S_ISDIR(observed.st_mode) or _is_reparse(current, observed):
            raise OriginalConfirmatoryResumeError(
                f"{role} contains a link, reparse point, or non-directory: {current}"
            )
        final = observed
    if final is None:
        raise RuntimeError("absolute directory unexpectedly has no components")
    return ResumePathIdentity.from_stat(final)


def _canonical_request_payload(
    *,
    runs_root: Path,
    predecessor: Path,
    successor: Path,
    retry_of_run_id: str,
    successor_run_id: str,
    predecessor_class: PredecessorClass,
    authority: OriginalConfirmatoryResumeAuthorityReceipt,
    predecessor_process_id: int | None,
    predecessor_process_create_time_unix_us: int | None,
    orphan_manual_diagnosis: bool,
    predecessor_identity: ResumePathIdentity,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy": ORIGINAL_CONFIRMATORY_RESUME_POLICY,
        "runs_root": str(runs_root),
        "predecessor_run_directory": str(predecessor),
        "successor_run_directory": str(successor),
        "retry_of_run_id": retry_of_run_id,
        "successor_run_id": successor_run_id,
        "predecessor_class": predecessor_class,
        "authority": authority.as_dict(),
        "predecessor_process_id": predecessor_process_id,
        "predecessor_process_create_time_unix_us": (predecessor_process_create_time_unix_us),
        "orphan_manual_diagnosis": orphan_manual_diagnosis,
        "supervisor_automatic_transition_allowed": False,
        "predecessor_identity": asdict(predecessor_identity),
        "max_attempt_count": 1,
        "automatic_retry_allowed": False,
        "predecessor_autodiscovery_allowed": False,
    }


def bind_original_confirmatory_resume_request(
    request: OriginalConfirmatoryResumeRequest,
) -> BoundOriginalConfirmatoryResumeRequest:
    """Freeze one explicit predecessor/successor lineage before any run write."""

    if not isinstance(request, OriginalConfirmatoryResumeRequest):
        raise TypeError("resume request must use OriginalConfirmatoryResumeRequest")
    runs_root = _lexical_absolute(request.runs_root)
    predecessor = _lexical_absolute(request.predecessor_run_directory)
    successor = _lexical_absolute(request.successor_run_directory)
    if (
        _SAFE_RUN_ID.fullmatch(request.retry_of_run_id) is None
        or _SAFE_RUN_ID.fullmatch(request.successor_run_id) is None
        or request.retry_of_run_id != predecessor.name
        or request.successor_run_id != successor.name
        or request.retry_of_run_id.casefold() == request.successor_run_id.casefold()
        or os.path.normcase(str(predecessor)) == os.path.normcase(str(successor))
        or predecessor.parent != runs_root
        or successor.parent != runs_root
    ):
        raise OriginalConfirmatoryResumeError(
            "resume request must name two distinct direct runs-root children"
        )
    if (
        type(request.max_attempt_count) is not int
        or request.max_attempt_count != 1
        or request.automatic_retry_allowed is not False
        or request.predecessor_autodiscovery_allowed is not False
    ):
        raise OriginalConfirmatoryResumeError(
            "original confirmatory successor permits one explicit attempt and no discovery/retry"
        )
    if (
        request.predecessor_class not in {"sealed_failed_demoted", "unsealed_interrupted_orphan"}
        or request.supervisor_automatic_transition_allowed is not False
    ):
        raise OriginalConfirmatoryResumeError(
            "predecessor class must be explicit and supervisor transition is forbidden"
        )
    if request.predecessor_class == "sealed_failed_demoted":
        if (
            request.predecessor_process_id is not None
            or request.predecessor_process_create_time_unix_us is not None
            or request.orphan_manual_diagnosis is not False
        ):
            raise OriginalConfirmatoryResumeError(
                "sealed predecessor must not carry orphan process diagnostics"
            )
    elif (
        type(request.predecessor_process_id) is not int
        or request.predecessor_process_id <= 0
        or type(request.predecessor_process_create_time_unix_us) is not int
        or request.predecessor_process_create_time_unix_us <= 0
        or request.orphan_manual_diagnosis is not True
    ):
        raise OriginalConfirmatoryResumeError(
            "orphan predecessor requires one exact dead process identity and manual diagnosis"
        )
    _plain_directory_identity(runs_root, role="runs root")
    predecessor_identity = _plain_directory_identity(predecessor, role="explicit predecessor")
    if successor.exists() or successor.is_symlink():
        raise OriginalConfirmatoryResumeError(
            "successor run directory must not exist before RunTracker creation"
        )
    authority = _verify_original_confirmatory_resume_authority(
        request.authority_pins,
        predecessor_class=request.predecessor_class,
        retry_of_run_id=request.retry_of_run_id,
        successor_run_id=request.successor_run_id,
    )
    payload = _canonical_request_payload(
        runs_root=runs_root,
        predecessor=predecessor,
        successor=successor,
        retry_of_run_id=request.retry_of_run_id,
        successor_run_id=request.successor_run_id,
        predecessor_class=request.predecessor_class,
        authority=authority,
        predecessor_process_id=request.predecessor_process_id,
        predecessor_process_create_time_unix_us=(request.predecessor_process_create_time_unix_us),
        orphan_manual_diagnosis=request.orphan_manual_diagnosis,
        predecessor_identity=predecessor_identity,
    )
    return BoundOriginalConfirmatoryResumeRequest(
        runs_root=runs_root,
        predecessor_run_directory=predecessor,
        successor_run_directory=successor,
        retry_of_run_id=request.retry_of_run_id,
        successor_run_id=request.successor_run_id,
        predecessor_class=request.predecessor_class,
        authority=authority,
        predecessor_process_id=request.predecessor_process_id,
        predecessor_process_create_time_unix_us=(request.predecessor_process_create_time_unix_us),
        orphan_manual_diagnosis=request.orphan_manual_diagnosis,
        predecessor_identity=predecessor_identity,
        request_sha256=canonical_sha256(payload),
    )


def _require_exact_original_controls(controls: ConfirmatoryExecutionControls) -> None:
    if not isinstance(controls, ConfirmatoryExecutionControls):
        raise TypeError("controls must be ConfirmatoryExecutionControls")
    controls.validate_for_plan(controls.plan)
    plan = controls.plan
    cells = tuple(plan.cells)
    if (
        plan.schema_version != 2
        or plan.config_sha256 != ORIGINAL_CONFIRMATORY_CONFIG_SEMANTIC_SHA256
        or canonical_sha256(plan.as_dict()) != ORIGINAL_CONFIRMATORY_PLAN_SEMANTIC_SHA256
        or controls.binding_sha256 != ORIGINAL_CONFIRMATORY_CONTROLS_BINDING_SHA256
        or controls.config_semantic_sha256 != ORIGINAL_CONFIRMATORY_CONFIG_SEMANTIC_SHA256
        or len(cells) != ORIGINAL_CONFIRMATORY_PLANNED_CELL_COUNT
        or sum(cell.required for cell in cells) != ORIGINAL_CONFIRMATORY_REQUIRED_CELL_COUNT
        or sum(not cell.required for cell in cells) != ORIGINAL_CONFIRMATORY_OPTIONAL_CELL_COUNT
        or controls.n_splits != ORIGINAL_CONFIRMATORY_OOF_FOLD_COUNT
        or set(controls.official_folds) != _EXPECTED_OFFICIAL_FOLDS
        or {cell.outer_fold for cell in cells} != _EXPECTED_OFFICIAL_FOLDS
        or {cell.model_seed for cell in cells} != _EXPECTED_MODEL_SEEDS
        or {cell.corruption_cell_id for cell in cells} != _EXPECTED_CORRUPTION_CELLS
        or {cell.scenario_id for cell in cells} != _EXPECTED_SCENARIOS
    ):
        raise OriginalConfirmatoryResumeError(
            "controls are not the exact frozen 108-cell original confirmatory plan"
        )
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise OriginalConfirmatoryResumeError(
            "original confirmatory plan contains duplicate cell identities"
        )
    cnn_cells = tuple(
        cell for cell in cells if controls.scenarios_by_id[cell.scenario_id].family == "cnn"
    )
    if (
        len(cnn_cells) != ORIGINAL_CONFIRMATORY_CNN_CELL_COUNT
        or any(not cell.required for cell in cnn_cells)
        or {cell.scenario_id for cell in cnn_cells} != _EXPECTED_CNN_SCENARIOS
    ):
        raise OriginalConfirmatoryResumeError(
            "original confirmatory CNN cell surface differs from the frozen plan"
        )


@lru_cache(maxsize=1)
def _official_weight_binding() -> tuple[Path, str]:
    weight_path = official_resnet18_weight_cache_path(OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER).resolve()
    if not weight_path.is_file():
        raise OriginalConfirmatoryResumeError(
            "official ResNet-18 weight cache is absent; download is forbidden"
        )
    weight_sha256 = sha256_file(weight_path)
    if weight_sha256 != ORIGINAL_CONFIRMATORY_WEIGHT_SHA256:
        raise OriginalConfirmatoryResumeError(
            "official ResNet-18 weight cache differs from the frozen authority"
        )
    return weight_path, weight_sha256


def _model_metadata(configuration: Mapping[str, Any]) -> dict[str, Any]:
    weight_identifier = configuration.get("weight_identifier")
    if weight_identifier != OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER:
        raise OriginalConfirmatoryResumeError(
            "original confirmatory checkpoints require the official ImageNet weight"
        )
    weight_path, weight_sha256 = _official_weight_binding()
    input_variant = configuration.get("input_variant")
    if input_variant not in {
        "context_rgb",
        "context_rgb_plus_binary_target_mask",
    }:
        raise OriginalConfirmatoryResumeError(
            "original confirmatory CNN input variant is unsupported"
        )
    uses_target_mask = input_variant == "context_rgb_plus_binary_target_mask"
    return {
        "architecture": "torchvision.resnet18",
        "class_order": [0, 1, 2, 3, 4],
        "input_channels": 4 if uses_target_mask else 3,
        "weight_identifier": weight_identifier,
        "weight_path": str(weight_path),
        "weight_sha256": weight_sha256,
        "implicit_weight_download": False,
        "preprocessing": {
            "rgb_resize": "bilinear_antialias",
            "rgb_range_before_normalisation": [0.0, 1.0],
            "rgb_mean": [0.485, 0.456, 0.406],
            "rgb_standard_deviation": [0.229, 0.224, 0.225],
            "target_mask_resize": ("nearest_binary_unnormalised" if uses_target_mask else None),
        },
        "fourth_channel_initialisation": (
            configuration.get("fourth_channel_initialisation") if uses_target_mask else None
        ),
    }


def _require_fingerprint_mapping(
    values: Mapping[str, Mapping[str, Mapping[str, str]]],
    *,
    cnn_cell_ids: frozenset[str],
) -> None:
    if not isinstance(values, Mapping) or set(values) != cnn_cell_ids:
        raise OriginalConfirmatoryResumeError(
            "CNN preflight fingerprints must contain exactly every frozen CNN cell"
        )
    expected_folds = {str(value) for value in range(ORIGINAL_CONFIRMATORY_OOF_FOLD_COUNT)}
    for cell_id, folds in values.items():
        if not isinstance(folds, Mapping) or set(folds) != expected_folds:
            raise OriginalConfirmatoryResumeError(
                f"CNN preflight fingerprints are incomplete for {cell_id}"
            )
        for fold_id, fingerprints in folds.items():
            if (
                not isinstance(fingerprints, Mapping)
                or set(fingerprints) != _DATA_AND_SPLIT_KEYS
                or any(
                    not isinstance(value, str) or _SHA256.fullmatch(value) is None
                    for value in fingerprints.values()
                )
            ):
                raise OriginalConfirmatoryResumeError(
                    f"CNN data/split fingerprints are invalid for {cell_id}/fold {fold_id}"
                )


def build_original_confirmatory_resume_contract(
    *,
    controls: ConfirmatoryExecutionControls,
    cnn_preflight_fingerprints: Mapping[
        str,
        Mapping[str, Mapping[str, str]],
    ],
) -> OriginalConfirmatoryResumeContract:
    """Build the exact 180-checkpoint allowlist from frozen outcome-blind inputs."""

    _require_exact_original_controls(controls)
    all_cells = tuple(controls.plan.cells)
    cnn_cells = tuple(
        cell for cell in all_cells if controls.scenarios_by_id[cell.scenario_id].family == "cnn"
    )
    cnn_cell_ids = frozenset(cell.cell_id for cell in cnn_cells)
    _require_fingerprint_mapping(
        cnn_preflight_fingerprints,
        cnn_cell_ids=cnn_cell_ids,
    )
    expectations: list[ResumeCheckpointExpectation] = []
    for cell in cnn_cells:
        scenario = controls.scenarios_by_id[cell.scenario_id]
        for fold_id in range(ORIGINAL_CONFIRMATORY_OOF_FOLD_COUNT):
            configuration = asdict(
                confirmatory_cnn_config_for_cell(
                    scenario,
                    controls,
                    model_seed=cell.model_seed + fold_id,
                    cpu_test_only=False,
                )
            )
            metadata = _model_metadata(configuration)
            expectations.append(
                ResumeCheckpointExpectation(
                    relative_path=(f"cells/{cell.cell_id}/checkpoints/fold_{fold_id:02d}.pt"),
                    cell_id=cell.cell_id,
                    fold_id=fold_id,
                    expected_configuration=configuration,
                    expected_model_metadata=metadata,
                    expected_data_and_split_sha256=dict(
                        cnn_preflight_fingerprints[cell.cell_id][str(fold_id)]
                    ),
                )
            )
    expectations_tuple = tuple(sorted(expectations, key=lambda value: value.relative_path))
    if (
        len(expectations_tuple) != ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT
        or len({value.relative_path for value in expectations_tuple})
        != ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT
        or len({value.relative_path.casefold() for value in expectations_tuple})
        != ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT
    ):
        raise OriginalConfirmatoryResumeError(
            "original confirmatory checkpoint allowlist is not exact and unique"
        )
    allowlist_payload = [
        {
            "relative_path": value.relative_path,
            "cell_id": value.cell_id,
            "fold_id": value.fold_id,
            "expected_configuration": dict(value.expected_configuration),
            "expected_model_metadata": dict(value.expected_model_metadata),
            "expected_data_and_split_sha256": dict(value.expected_data_and_split_sha256),
        }
        for value in expectations_tuple
    ]
    return OriginalConfirmatoryResumeContract(
        plan_semantic_sha256=ORIGINAL_CONFIRMATORY_PLAN_SEMANTIC_SHA256,
        config_semantic_sha256=ORIGINAL_CONFIRMATORY_CONFIG_SEMANTIC_SHA256,
        controls_binding_sha256=ORIGINAL_CONFIRMATORY_CONTROLS_BINDING_SHA256,
        all_cell_ids=tuple(sorted(cell.cell_id for cell in all_cells)),
        cnn_cell_ids=tuple(sorted(cnn_cell_ids)),
        checkpoint_expectations=expectations_tuple,
        checkpoint_allowlist_sha256=canonical_sha256(allowlist_payload),
    )


if os.name == "nt":

    class _Win32FindStreamData(ctypes.Structure):
        _fields_ = [
            ("stream_size", ctypes.c_longlong),
            ("stream_name", ctypes.c_wchar * (260 + 36)),
        ]


def _named_streams(path: Path) -> tuple[str, ...]:
    """Return non-default NTFS stream names for one regular file."""

    if os.name != "nt":
        return ()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    find_first = kernel32.FindFirstStreamW
    find_first.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_int,
        ctypes.POINTER(_Win32FindStreamData),
        ctypes.c_uint32,
    ]
    find_first.restype = ctypes.c_void_p
    find_next = kernel32.FindNextStreamW
    find_next.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Win32FindStreamData),
    ]
    find_next.restype = ctypes.c_int
    find_close = kernel32.FindClose
    find_close.argtypes = [ctypes.c_void_p]
    find_close.restype = ctypes.c_int
    data = _Win32FindStreamData()
    handle = find_first(str(path), 0, ctypes.byref(data), 0)
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        error = ctypes.get_last_error()
        if error in {1, 38}:
            return ()
        raise OriginalConfirmatoryResumeError(
            f"cannot enumerate checkpoint streams: winerror={error}"
        )
    streams: list[str] = []
    try:
        while True:
            name = str(data.stream_name)
            if name != "::$DATA":
                streams.append(name)
            if not find_next(handle, ctypes.byref(data)):
                error = ctypes.get_last_error()
                if error == 38:
                    break
                raise OriginalConfirmatoryResumeError(
                    f"checkpoint stream enumeration failed: winerror={error}"
                )
    finally:
        find_close(handle)
    return tuple(sorted(streams))


def _hash_private_file(path: Path, initial: os.stat_result, *, role: str) -> str:
    if (
        not stat.S_ISREG(initial.st_mode)
        or _is_reparse(path, initial)
        or int(initial.st_nlink) != 1
    ):
        raise OriginalConfirmatoryResumeError(f"{role} is not one private regular file: {path}")
    if _named_streams(path):
        raise OriginalConfirmatoryResumeError(f"{role} has forbidden named streams")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        initial_identity = ResumePathIdentity.from_stat(initial)
        opened_identity = ResumePathIdentity.from_stat(opened)
        if (
            (
                opened_identity.device,
                opened_identity.inode,
                opened_identity.size_bytes,
                stat.S_IFMT(opened_identity.mode),
                opened_identity.link_count,
            )
            != (
                initial_identity.device,
                initial_identity.inode,
                initial_identity.size_bytes,
                stat.S_IFMT(initial_identity.mode),
                initial_identity.link_count,
            )
            or not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
        ):
            raise OriginalConfirmatoryResumeError(f"{role} changed while opening")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
        opened_after = os.fstat(descriptor)
        lexical_after = path.lstat()
        opened_after_identity = ResumePathIdentity.from_stat(opened_after)
        lexical_after_identity = ResumePathIdentity.from_stat(lexical_after)
        if (
            opened_after_identity != opened_identity
            or lexical_after_identity != initial_identity
            or (
                lexical_after_identity.device,
                lexical_after_identity.inode,
                lexical_after_identity.size_bytes,
                stat.S_IFMT(lexical_after_identity.mode),
                lexical_after_identity.link_count,
            )
            != (
                opened_after_identity.device,
                opened_after_identity.inode,
                opened_after_identity.size_bytes,
                stat.S_IFMT(opened_after_identity.mode),
                opened_after_identity.link_count,
            )
            or _is_reparse(path, lexical_after)
        ):
            raise OriginalConfirmatoryResumeError(f"{role} changed while hashing")
        return digest.hexdigest()
    except OSError as exc:
        raise OriginalConfirmatoryResumeError(f"{role} is unreadable: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _strict_tree_inventory(
    root: Path,
    *,
    role: str,
    hash_file_contents: bool,
) -> tuple[dict[str, Any], ...]:
    """Capture a no-link tree without interpreting any scientific artifact."""

    root_identity = _plain_directory_identity(root, role=role)
    del root_identity
    records: list[dict[str, Any]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda value: value.name)
        except OSError as exc:
            raise OriginalConfirmatoryResumeError(
                f"{role} cannot be enumerated: {directory}"
            ) from exc
        for child in children:
            try:
                observed = child.lstat()
            except OSError as exc:
                raise OriginalConfirmatoryResumeError(
                    f"{role} entry is inaccessible: {child}"
                ) from exc
            relative = child.relative_to(root).as_posix()
            if _is_reparse(child, observed):
                raise OriginalConfirmatoryResumeError(
                    f"{role} contains a link or reparse point: {relative}"
                )
            identity = ResumePathIdentity.from_stat(observed)
            if stat.S_ISDIR(observed.st_mode):
                records.append(
                    {
                        "relative_path": relative,
                        "entry_kind": "directory",
                        "identity": asdict(identity),
                    }
                )
                pending.append(child)
                continue
            if not stat.S_ISREG(observed.st_mode):
                raise OriginalConfirmatoryResumeError(
                    f"{role} contains a non-regular entry: {relative}"
                )
            if int(observed.st_nlink) != 1:
                raise OriginalConfirmatoryResumeError(
                    f"{role} contains a hardlinked file: {relative}"
                )
            streams = _named_streams(child)
            if streams:
                raise OriginalConfirmatoryResumeError(f"{role} contains named streams: {relative}")
            record: dict[str, Any] = {
                "relative_path": relative,
                "entry_kind": "file",
                "identity": asdict(identity),
            }
            if hash_file_contents:
                record["sha256"] = _hash_private_file(
                    child,
                    observed,
                    role=f"{role} file",
                )
            records.append(record)
    records.sort(key=lambda value: cast(str, value["relative_path"]).casefold())
    if len({cast(str, value["relative_path"]).casefold() for value in records}) != len(records):
        raise OriginalConfirmatoryResumeError(f"{role} contains case-aliased duplicate paths")
    return tuple(records)


def _authority_tree_root(
    directory: Path,
    *,
    role: str,
) -> tuple[ResumePathIdentity, str]:
    identity_before = _plain_directory_identity(directory, role=role)
    inventory_before = _strict_tree_inventory(
        directory,
        role=role,
        hash_file_contents=True,
    )
    identity_after = _plain_directory_identity(directory, role=role)
    if identity_after != identity_before:
        raise OriginalConfirmatoryResumeError(f"{role} identity changed while hashing")
    content_inventory = [
        (
            {
                "relative_path": record["relative_path"],
                "entry_kind": "file",
                "size_bytes": record["identity"]["size_bytes"],
                "sha256": record["sha256"],
            }
            if record["entry_kind"] == "file"
            else {
                "relative_path": record["relative_path"],
                "entry_kind": "directory",
            }
        )
        for record in inventory_before
    ]
    return identity_before, canonical_sha256(content_inventory)


def _runtime_identity_payload() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "sys_version": sys.version,
        "os_name": os.name,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "python_compiler": platform.python_compiler(),
    }


def _process_orig_argv() -> tuple[str, ...]:
    values = getattr(sys, "orig_argv", None)
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) for value in values)
    ):
        raise OriginalConfirmatoryResumeError("exact process orig_argv is unavailable")
    return tuple(values)


def _raw_process_command_line() -> str:
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        getter = kernel32.GetCommandLineW
        getter.argtypes = []
        getter.restype = ctypes.c_wchar_p
        value = getter()
        if not isinstance(value, str) or not value:
            raise OriginalConfirmatoryResumeError("Windows GetCommandLineW receipt is unavailable")
        return value
    return "\0".join(_process_orig_argv())


def _sanitized_environment() -> tuple[tuple[str, str | None], ...]:
    return tuple((name, os.environ.get(name)) for name in _SANITIZED_ENVIRONMENT_NAMES)


def _saved_session_binding(saved_session_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "saved_codex_session_id": saved_session_id,
        "resume_command_prefix": [
            "codex",
            "exec",
            "resume",
            saved_session_id,
        ],
        "use_last_session": False,
        "wake_count": 1,
    }


def _supervisor_canonical_bytes(value: Any) -> bytes:
    """Match the installed supervisor's canonical JSON envelope encoding."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _read_supervisor_launch_intent(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Read one exact, private, canonical supervisor launch-intent envelope."""

    observed_sha256 = _hash_runtime_file(path, role="supervisor launch intent")
    if observed_sha256 != expected_sha256:
        raise OriginalConfirmatoryResumeError(
            "supervisor launch intent differs from its exact authority pin"
        )
    try:
        raw_bytes = path.read_bytes()
        envelope = json.loads(raw_bytes.decode("utf-8"))
        canonical_bytes = _supervisor_canonical_bytes(envelope)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise OriginalConfirmatoryResumeError(
            "supervisor launch intent envelope is unreadable or non-canonical"
        ) from exc
    if (
        not isinstance(envelope, Mapping)
        or set(envelope) != {"schema_version", "payload", "payload_sha256"}
        or envelope.get("schema_version") != 1
        or not isinstance(envelope.get("payload"), Mapping)
        or raw_bytes != canonical_bytes
    ):
        raise OriginalConfirmatoryResumeError(
            "supervisor launch intent envelope has invalid closed structure"
        )
    payload = dict(cast(Mapping[str, Any], envelope["payload"]))
    payload_sha256 = hashlib.sha256(_supervisor_canonical_bytes(payload)).hexdigest()
    if envelope.get("payload_sha256") != payload_sha256:
        raise OriginalConfirmatoryResumeError(
            "supervisor launch intent envelope payload hash differs"
        )
    if _hash_runtime_file(path, role="supervisor launch intent readback") != (observed_sha256):
        raise OriginalConfirmatoryResumeError(
            "supervisor launch intent changed during verification"
        )
    return payload, observed_sha256


def _hash_runtime_file(path: Path, *, role: str) -> str:
    absolute = _lexical_absolute(path)
    try:
        observed = absolute.lstat()
    except OSError as exc:
        raise OriginalConfirmatoryResumeError(f"{role} is missing or inaccessible") from exc
    return _hash_private_file(absolute, observed, role=role)


def _verify_original_confirmatory_runtime(
    pins: OriginalConfirmatoryRuntimePins,
) -> OriginalConfirmatoryRuntimeReceipt:
    if not isinstance(pins, OriginalConfirmatoryRuntimePins):
        raise TypeError("runtime pins must use OriginalConfirmatoryRuntimePins")
    hash_values = (
        pins.execution_source_root_sha256,
        pins.execution_source_manifest_sha256,
        pins.project_plan_sha256,
        pins.project_plan_root_sha256,
        pins.command_argv_sha256,
        pins.process_orig_argv_sha256,
        pins.raw_process_command_line_sha256,
        pins.interpreter_sha256,
        pins.runtime_identity_sha256,
        pins.environment_sha256,
        pins.supervisor_release_sha256,
        pins.supervisor_manifest_sha256,
        pins.supervisor_handoff_sha256,
        pins.supervisor_launch_intent_sha256,
        pins.saved_codex_session_binding_sha256,
    )
    if any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in hash_values):
        raise OriginalConfirmatoryResumeError(
            "runtime/plan/supervisor pins must be exact lowercase SHA-256"
        )
    if (
        not isinstance(pins.command_argv, tuple)
        or not pins.command_argv
        or any(not isinstance(value, str) for value in pins.command_argv)
        or not isinstance(pins.process_orig_argv, tuple)
        or not pins.process_orig_argv
        or any(not isinstance(value, str) for value in pins.process_orig_argv)
        or not isinstance(pins.raw_process_command_line, str)
        or not pins.raw_process_command_line
        or tuple(name for name, _value in pins.sanitized_environment)
        != _SANITIZED_ENVIRONMENT_NAMES
        or any(
            not isinstance(value, str) and value is not None
            for _name, value in pins.sanitized_environment
        )
        or (
            (pins.supervisor_process_id is None)
            != (pins.supervisor_process_create_time_unix_us is None)
        )
        or (
            pins.supervisor_process_id is not None
            and (
                type(pins.supervisor_process_id) is not int
                or pins.supervisor_process_id <= 0
                or type(pins.supervisor_process_create_time_unix_us) is not int
                or pins.supervisor_process_create_time_unix_us <= 0
            )
        )
        or _SAVED_SESSION_ID.fullmatch(pins.saved_codex_session_id) is None
        or _SAFE_RUN_ID.fullmatch(pins.supervisor_job_id) is None
    ):
        raise OriginalConfirmatoryResumeError(
            "runtime authority lacks exact argv or saved Codex session ID"
        )
    project_root = _lexical_absolute(pins.project_root)
    _plain_directory_identity(project_root, role="execution project root")
    source_tree = capture_source_tree(project_root)
    source_root = source_tree.get("root_sha256")
    source_manifest_sha256 = canonical_sha256(source_tree)
    if (
        source_root != pins.execution_source_root_sha256
        or source_manifest_sha256 != pins.execution_source_manifest_sha256
    ):
        raise OriginalConfirmatoryResumeError(
            "live execution-source root/manifest differs from authority"
        )
    plan_path = _lexical_absolute(pins.project_plan_path)
    if (
        os.path.normcase(str(plan_path)) != os.path.normcase(str(project_root / "PLAN.md"))
        or not plan_path.is_file()
    ):
        raise OriginalConfirmatoryResumeError("authority must bind the exact project PLAN.md")
    plan_sha256 = _hash_runtime_file(plan_path, role="project PLAN.md")
    plan_root_sha256 = canonical_sha256(
        [
            {
                "relative_path": "PLAN.md",
                "size_bytes": int(plan_path.stat().st_size),
                "sha256": plan_sha256,
            }
        ]
    )
    if plan_sha256 != pins.project_plan_sha256 or plan_root_sha256 != pins.project_plan_root_sha256:
        raise OriginalConfirmatoryResumeError("project PLAN.md file/root differs from authority")
    current_argv = tuple(sys.argv)
    command_sha256 = canonical_sha256(list(current_argv))
    current_orig_argv = _process_orig_argv()
    orig_argv_sha256 = canonical_sha256(list(current_orig_argv))
    raw_command_line = _raw_process_command_line()
    raw_command_line_sha256 = hashlib.sha256(raw_command_line.encode("utf-16-le")).hexdigest()
    if (
        current_argv != pins.command_argv
        or command_sha256 != pins.command_argv_sha256
        or current_orig_argv != pins.process_orig_argv
        or orig_argv_sha256 != pins.process_orig_argv_sha256
        or raw_command_line != pins.raw_process_command_line
        or raw_command_line_sha256 != pins.raw_process_command_line_sha256
    ):
        raise OriginalConfirmatoryResumeError(
            "live argv/orig_argv/raw process command line differs from exact authority"
        )
    working_directory = _lexical_absolute(Path.cwd())
    if os.path.normcase(str(working_directory)) != os.path.normcase(
        str(_lexical_absolute(pins.working_directory))
    ):
        raise OriginalConfirmatoryResumeError("live working directory differs from exact authority")
    interpreter_path = _lexical_absolute(sys.executable)
    if os.path.normcase(str(interpreter_path)) != os.path.normcase(
        str(_lexical_absolute(pins.interpreter_path))
    ):
        raise OriginalConfirmatoryResumeError("live interpreter path differs from exact authority")
    interpreter_sha256 = _hash_runtime_file(
        interpreter_path,
        role="Python interpreter",
    )
    runtime_identity_sha256 = canonical_sha256(_runtime_identity_payload())
    sanitized_environment = _sanitized_environment()
    environment_sha256 = canonical_sha256(
        [{"name": name, "value": value} for name, value in sanitized_environment]
    )
    if (
        interpreter_sha256 != pins.interpreter_sha256
        or runtime_identity_sha256 != pins.runtime_identity_sha256
        or sanitized_environment != pins.sanitized_environment
        or environment_sha256 != pins.environment_sha256
        or dict(sanitized_environment).get("AANCA_SAVED_CODEX_SESSION_ID")
        != pins.saved_codex_session_id
        or dict(sanitized_environment).get("AANCA_SUPERVISOR_RELEASE_ID") != _SUPERVISOR_RELEASE_ID
    ):
        raise OriginalConfirmatoryResumeError(
            "live interpreter/runtime/environment differs from authority"
        )
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not isinstance(local_app_data, str) or not local_app_data:
        raise OriginalConfirmatoryResumeError(
            "LOCALAPPDATA is absent from the exact supervisor runtime"
        )
    expected_supervisor_root = _lexical_absolute(Path(local_app_data) / "AANCA-supervisor")
    supervisor_root = _lexical_absolute(pins.supervisor_root)
    if os.path.normcase(str(supervisor_root)) != os.path.normcase(str(expected_supervisor_root)):
        raise OriginalConfirmatoryResumeError(
            "supervisor root is not exact %LOCALAPPDATA%\\AANCA-supervisor"
        )
    _plain_directory_identity(supervisor_root, role="supervisor root")
    supervisor_release = _lexical_absolute(pins.supervisor_release_path)
    supervisor_manifest = _lexical_absolute(pins.supervisor_manifest_path)
    supervisor_handoff = _lexical_absolute(pins.supervisor_handoff_path)
    supervisor_launch_intent = _lexical_absolute(pins.supervisor_launch_intent_path)
    expected_launch_intent = (
        supervisor_root / "jobs" / pins.supervisor_job_id / "launch_intent.json"
    )
    if os.path.normcase(str(supervisor_launch_intent)) != os.path.normcase(
        str(expected_launch_intent)
    ):
        raise OriginalConfirmatoryResumeError(
            "supervisor launch intent is not the exact jobs/<job_id> receipt"
        )
    if any(
        not path.is_relative_to(supervisor_root)
        for path in (
            supervisor_release,
            supervisor_manifest,
            supervisor_handoff,
            supervisor_launch_intent,
        )
    ):
        raise OriginalConfirmatoryResumeError(
            "supervisor evidence paths escape the exact LOCALAPPDATA root"
        )
    supervisor_hashes = (
        _hash_runtime_file(
            supervisor_release,
            role="supervisor release",
        ),
        _hash_runtime_file(
            supervisor_manifest,
            role="supervisor manifest",
        ),
        _hash_runtime_file(
            supervisor_handoff,
            role="supervisor handoff",
        ),
    )
    if supervisor_hashes != (
        pins.supervisor_release_sha256,
        pins.supervisor_manifest_sha256,
        pins.supervisor_handoff_sha256,
    ):
        raise OriginalConfirmatoryResumeError(
            "supervisor release/manifest/handoff differs from authority"
        )
    manifest = _read_json_object(
        supervisor_manifest,
        role="supervisor semantic manifest",
    )
    handoff = _read_json_object(
        supervisor_handoff,
        role="supervisor semantic handoff",
    )
    expected_manifest = {
        "schema_version": 1,
        "policy": _SUPERVISOR_MANIFEST_POLICY,
        "supervisor_root": str(supervisor_root),
        "release_id": _SUPERVISOR_RELEASE_ID,
        "release_path": str(supervisor_release),
        "release_sha256": supervisor_hashes[0],
        "handoff_path": str(supervisor_handoff),
        "saved_codex_session_id": pins.saved_codex_session_id,
        "codex_resume_argv_prefix": [
            "codex",
            "exec",
            "resume",
            pins.saved_codex_session_id,
        ],
        "use_last_session": False,
        "automatic_scientific_retry_allowed": False,
        "supervisor_process_id": pins.supervisor_process_id,
        "supervisor_process_create_time_unix_us": (pins.supervisor_process_create_time_unix_us),
    }
    expected_handoff = {
        "schema_version": 1,
        "policy": _SUPERVISOR_HANDOFF_POLICY,
        "supervisor_root": str(supervisor_root),
        "release_id": _SUPERVISOR_RELEASE_ID,
        "release_path": str(supervisor_release),
        "release_sha256": supervisor_hashes[0],
        "manifest_path": str(supervisor_manifest),
        "manifest_sha256": supervisor_hashes[1],
        "saved_codex_session_id": pins.saved_codex_session_id,
        "codex_resume_argv": [
            "codex",
            "exec",
            "resume",
            pins.saved_codex_session_id,
            _CONTINUATION_PROMPT,
        ],
        "continuation_prompt_sha256": hashlib.sha256(
            _CONTINUATION_PROMPT.encode("utf-8")
        ).hexdigest(),
        "use_last_session": False,
        "wake_count": 1,
        "automatic_scientific_retry_allowed": False,
        "automatic_orphan_transition_allowed": False,
        "supervisor_process_id": pins.supervisor_process_id,
        "supervisor_process_create_time_unix_us": (pins.supervisor_process_create_time_unix_us),
    }
    if manifest != expected_manifest or handoff != expected_handoff:
        raise OriginalConfirmatoryResumeError(
            "supervisor manifest/handoff semantic crosslinks are incoherent"
        )
    dynamic_supervisor_environment_names = {
        name for name in os.environ if name.startswith("AANCA_SUPERVISOR_")
    }
    if dynamic_supervisor_environment_names != _EXPECTED_SUPERVISOR_ENVIRONMENT_NAMES:
        raise OriginalConfirmatoryResumeError(
            "supervisor runtime has unknown or missing dynamic environment keys"
        )
    attempt_nonce = os.environ.get(_SUPERVISOR_ATTEMPT_NONCE_NAME)
    if not isinstance(attempt_nonce, str) or _ATTEMPT_NONCE.fullmatch(attempt_nonce) is None:
        raise OriginalConfirmatoryResumeError("supervisor attempt nonce is absent or malformed")
    launch_intent, launch_intent_sha256 = _read_supervisor_launch_intent(
        supervisor_launch_intent,
        expected_sha256=pins.supervisor_launch_intent_sha256,
    )
    launch_command = launch_intent.get("command")
    command_without_hash = {
        "program_path": str(interpreter_path),
        "program_sha256": interpreter_sha256,
        "argv": list(current_orig_argv),
        "cwd": str(working_directory),
    }
    expected_launch_command = {
        **command_without_hash,
        "command_sha256": hashlib.sha256(
            _supervisor_canonical_bytes(command_without_hash)
        ).hexdigest(),
    }
    if (
        launch_intent.get("schema_version") != 1
        or launch_intent.get("job_id") != pins.supervisor_job_id
        or launch_intent.get("attempt_nonce") != attempt_nonce
        or launch_intent.get("attempt_count") != 1
        or launch_intent.get("max_attempt_count") != 1
        or launch_intent.get("automatic_retry_allowed") is not False
        or launch_command != expected_launch_command
        or launch_intent.get("command_sha256") != expected_launch_command["command_sha256"]
    ):
        raise OriginalConfirmatoryResumeError(
            "supervisor launch intent does not bind this exact one-use process"
        )
    saved_session_binding_sha256 = canonical_sha256(
        _saved_session_binding(pins.saved_codex_session_id)
    )
    if saved_session_binding_sha256 != pins.saved_codex_session_binding_sha256:
        raise OriginalConfirmatoryResumeError("saved Codex session binding differs from authority")
    payload: dict[str, Any] = {
        "project_root": str(project_root),
        "execution_source_root_sha256": source_root,
        "execution_source_manifest_sha256": source_manifest_sha256,
        "project_plan_path": str(plan_path),
        "project_plan_sha256": plan_sha256,
        "project_plan_root_sha256": plan_root_sha256,
        "command_argv": list(current_argv),
        "command_argv_sha256": command_sha256,
        "process_orig_argv": list(current_orig_argv),
        "process_orig_argv_sha256": orig_argv_sha256,
        "raw_process_command_line": raw_command_line,
        "raw_process_command_line_sha256": raw_command_line_sha256,
        "working_directory": str(working_directory),
        "interpreter_path": str(interpreter_path),
        "interpreter_sha256": interpreter_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "sanitized_environment": [
            {"name": name, "value": value} for name, value in sanitized_environment
        ],
        "environment_sha256": environment_sha256,
        "supervisor_root": str(supervisor_root),
        "supervisor_release_path": str(supervisor_release),
        "supervisor_release_sha256": supervisor_hashes[0],
        "supervisor_manifest_path": str(supervisor_manifest),
        "supervisor_manifest_sha256": supervisor_hashes[1],
        "supervisor_handoff_path": str(supervisor_handoff),
        "supervisor_handoff_sha256": supervisor_hashes[2],
        "supervisor_job_id": pins.supervisor_job_id,
        "supervisor_launch_intent_path": str(supervisor_launch_intent),
        "supervisor_launch_intent_sha256": launch_intent_sha256,
        "supervisor_attempt_nonce": attempt_nonce,
        "supervisor_process_id": pins.supervisor_process_id,
        "supervisor_process_create_time_unix_us": (pins.supervisor_process_create_time_unix_us),
        "saved_codex_session_id": pins.saved_codex_session_id,
        "saved_codex_session_binding_sha256": (saved_session_binding_sha256),
    }
    return OriginalConfirmatoryRuntimeReceipt(
        project_root=project_root,
        execution_source_root_sha256=cast(str, source_root),
        execution_source_manifest_sha256=source_manifest_sha256,
        project_plan_path=plan_path,
        project_plan_sha256=plan_sha256,
        project_plan_root_sha256=plan_root_sha256,
        command_argv=current_argv,
        command_argv_sha256=command_sha256,
        process_orig_argv=current_orig_argv,
        process_orig_argv_sha256=orig_argv_sha256,
        raw_process_command_line=raw_command_line,
        raw_process_command_line_sha256=raw_command_line_sha256,
        working_directory=working_directory,
        interpreter_path=interpreter_path,
        interpreter_sha256=interpreter_sha256,
        runtime_identity_sha256=runtime_identity_sha256,
        sanitized_environment=sanitized_environment,
        environment_sha256=environment_sha256,
        supervisor_root=supervisor_root,
        supervisor_release_path=supervisor_release,
        supervisor_release_sha256=supervisor_hashes[0],
        supervisor_manifest_path=supervisor_manifest,
        supervisor_manifest_sha256=supervisor_hashes[1],
        supervisor_handoff_path=supervisor_handoff,
        supervisor_handoff_sha256=supervisor_hashes[2],
        supervisor_job_id=pins.supervisor_job_id,
        supervisor_launch_intent_path=supervisor_launch_intent,
        supervisor_launch_intent_sha256=launch_intent_sha256,
        supervisor_attempt_nonce=attempt_nonce,
        supervisor_process_id=pins.supervisor_process_id,
        supervisor_process_create_time_unix_us=(pins.supervisor_process_create_time_unix_us),
        saved_codex_session_id=pins.saved_codex_session_id,
        saved_codex_session_binding_sha256=saved_session_binding_sha256,
        receipt_sha256=canonical_sha256(payload),
    )


def _runtime_pins_from_receipt(
    receipt: OriginalConfirmatoryRuntimeReceipt,
) -> OriginalConfirmatoryRuntimePins:
    return OriginalConfirmatoryRuntimePins(
        project_root=receipt.project_root,
        execution_source_root_sha256=receipt.execution_source_root_sha256,
        execution_source_manifest_sha256=(receipt.execution_source_manifest_sha256),
        project_plan_path=receipt.project_plan_path,
        project_plan_sha256=receipt.project_plan_sha256,
        project_plan_root_sha256=receipt.project_plan_root_sha256,
        command_argv=receipt.command_argv,
        command_argv_sha256=receipt.command_argv_sha256,
        process_orig_argv=receipt.process_orig_argv,
        process_orig_argv_sha256=receipt.process_orig_argv_sha256,
        raw_process_command_line=receipt.raw_process_command_line,
        raw_process_command_line_sha256=(receipt.raw_process_command_line_sha256),
        working_directory=receipt.working_directory,
        interpreter_path=receipt.interpreter_path,
        interpreter_sha256=receipt.interpreter_sha256,
        runtime_identity_sha256=receipt.runtime_identity_sha256,
        sanitized_environment=receipt.sanitized_environment,
        environment_sha256=receipt.environment_sha256,
        supervisor_root=receipt.supervisor_root,
        supervisor_release_path=receipt.supervisor_release_path,
        supervisor_release_sha256=receipt.supervisor_release_sha256,
        supervisor_manifest_path=receipt.supervisor_manifest_path,
        supervisor_manifest_sha256=receipt.supervisor_manifest_sha256,
        supervisor_handoff_path=receipt.supervisor_handoff_path,
        supervisor_handoff_sha256=receipt.supervisor_handoff_sha256,
        supervisor_job_id=receipt.supervisor_job_id,
        supervisor_launch_intent_path=receipt.supervisor_launch_intent_path,
        supervisor_launch_intent_sha256=receipt.supervisor_launch_intent_sha256,
        supervisor_process_id=receipt.supervisor_process_id,
        supervisor_process_create_time_unix_us=(receipt.supervisor_process_create_time_unix_us),
        saved_codex_session_id=receipt.saved_codex_session_id,
        saved_codex_session_binding_sha256=(receipt.saved_codex_session_binding_sha256),
    )


def _q_gate_adapter_pins_from_receipt(
    receipt: OriginalConfirmatoryQGateAdapterReceipt,
) -> OriginalConfirmatoryQGateAdapterPins:
    return OriginalConfirmatoryQGateAdapterPins(
        q_directory=receipt.q_directory,
        q_tree_root_sha256=receipt.q_tree_root_sha256,
        q_authorization_path=receipt.q_authorization_path,
        q_authorization_sha256=receipt.q_authorization_sha256,
        q_dual_gate_binding_sha256=receipt.q_dual_gate_binding_sha256,
        historical_p_authority_directory=(receipt.historical_p_authority_directory),
        historical_p_authority_root_sha256=(receipt.historical_p_authority_root_sha256),
        historical_primary_run_directory=(receipt.historical_primary_run_directory),
        historical_primary_run_id=receipt.historical_primary_run_id,
        historical_primary_artifact_root_sha256=(receipt.historical_primary_artifact_root_sha256),
        confirmatory_config_semantic_sha256=(receipt.confirmatory_config_semantic_sha256),
        confirmatory_plan_semantic_sha256=(receipt.confirmatory_plan_semantic_sha256),
        confirmatory_controls_binding_sha256=(receipt.confirmatory_controls_binding_sha256),
    )


def _verify_original_confirmatory_q_gate_adapter(
    pins: OriginalConfirmatoryQGateAdapterPins,
) -> OriginalConfirmatoryQGateAdapterReceipt:
    """Revalidate only opaque Q/P identities and hashes, never Q schema semantics."""

    if not isinstance(pins, OriginalConfirmatoryQGateAdapterPins):
        raise TypeError("Q gate adapter pins must use OriginalConfirmatoryQGateAdapterPins")
    hash_values = (
        pins.q_tree_root_sha256,
        pins.q_authorization_sha256,
        pins.q_dual_gate_binding_sha256,
        pins.historical_p_authority_root_sha256,
        pins.historical_primary_artifact_root_sha256,
        pins.confirmatory_config_semantic_sha256,
        pins.confirmatory_plan_semantic_sha256,
        pins.confirmatory_controls_binding_sha256,
    )
    if any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in hash_values):
        raise OriginalConfirmatoryResumeError("Q gate adapter pins must be exact lowercase SHA-256")
    if (
        pins.confirmatory_config_semantic_sha256 != ORIGINAL_CONFIRMATORY_CONFIG_SEMANTIC_SHA256
        or pins.confirmatory_plan_semantic_sha256 != ORIGINAL_CONFIRMATORY_PLAN_SEMANTIC_SHA256
        or pins.confirmatory_controls_binding_sha256
        != ORIGINAL_CONFIRMATORY_CONTROLS_BINDING_SHA256
    ):
        raise OriginalConfirmatoryResumeError(
            "Q gate adapter changes frozen confirmatory science bindings"
        )
    q_directory = _lexical_absolute(pins.q_directory)
    q_authorization = _lexical_absolute(pins.q_authorization_path)
    historical_p_authority = _lexical_absolute(pins.historical_p_authority_directory)
    historical_run = _lexical_absolute(pins.historical_primary_run_directory)
    if (
        _SAFE_RUN_ID.fullmatch(pins.historical_primary_run_id) is None
        or historical_run.name != pins.historical_primary_run_id
        or not q_authorization.is_relative_to(q_directory)
        or len(
            {
                os.path.normcase(str(path))
                for path in (
                    q_directory,
                    historical_p_authority,
                    historical_run,
                )
            }
        )
        != 3
    ):
        raise OriginalConfirmatoryResumeError(
            "Q gate adapter directories or historical run binding are malformed"
        )
    q_identity, q_root = _authority_tree_root(
        q_directory,
        role="independently verified Q directory",
    )
    historical_p_identity, historical_p_root = _authority_tree_root(
        historical_p_authority,
        role="historical P authority",
    )
    historical_run_identity = _plain_directory_identity(
        historical_run,
        role="historical primary run",
    )
    q_authorization_sha256 = _hash_runtime_file(
        q_authorization,
        role="Q authorization",
    )
    if (
        q_root != pins.q_tree_root_sha256
        or historical_p_root != pins.historical_p_authority_root_sha256
        or q_authorization_sha256 != pins.q_authorization_sha256
    ):
        raise OriginalConfirmatoryResumeError(
            "Q/P authority identity or content differs from its adapter pin"
        )
    q_identity_after, q_root_after = _authority_tree_root(
        q_directory,
        role="independently verified Q directory readback",
    )
    historical_p_identity_after, historical_p_root_after = _authority_tree_root(
        historical_p_authority,
        role="historical P authority readback",
    )
    historical_run_identity_after = _plain_directory_identity(
        historical_run,
        role="historical primary run readback",
    )
    if (
        q_identity_after != q_identity
        or q_root_after != q_root
        or historical_p_identity_after != historical_p_identity
        or historical_p_root_after != historical_p_root
        or historical_run_identity_after != historical_run_identity
        or _hash_runtime_file(
            q_authorization,
            role="Q authorization readback",
        )
        != q_authorization_sha256
    ):
        raise OriginalConfirmatoryResumeError("Q gate adapter inputs changed during verification")
    payload: dict[str, Any] = {
        "policy": ORIGINAL_CONFIRMATORY_Q_GATE_ADAPTER_POLICY,
        "q_directory": str(q_directory),
        "q_directory_identity": asdict(q_identity),
        "q_tree_root_sha256": q_root,
        "q_authorization_path": str(q_authorization),
        "q_authorization_sha256": q_authorization_sha256,
        "q_dual_gate_binding_sha256": pins.q_dual_gate_binding_sha256,
        "historical_p_authority_directory": str(historical_p_authority),
        "historical_p_authority_identity": asdict(historical_p_identity),
        "historical_p_authority_root_sha256": historical_p_root,
        "historical_primary_run_directory": str(historical_run),
        "historical_primary_run_identity": asdict(historical_run_identity),
        "historical_primary_run_id": pins.historical_primary_run_id,
        "historical_primary_artifact_root_sha256": (pins.historical_primary_artifact_root_sha256),
        "confirmatory_config_semantic_sha256": (pins.confirmatory_config_semantic_sha256),
        "confirmatory_plan_semantic_sha256": (pins.confirmatory_plan_semantic_sha256),
        "confirmatory_controls_binding_sha256": (pins.confirmatory_controls_binding_sha256),
    }
    return OriginalConfirmatoryQGateAdapterReceipt(
        q_directory=q_directory,
        q_directory_identity=q_identity,
        q_tree_root_sha256=q_root,
        q_authorization_path=q_authorization,
        q_authorization_sha256=q_authorization_sha256,
        q_dual_gate_binding_sha256=pins.q_dual_gate_binding_sha256,
        historical_p_authority_directory=historical_p_authority,
        historical_p_authority_identity=historical_p_identity,
        historical_p_authority_root_sha256=historical_p_root,
        historical_primary_run_directory=historical_run,
        historical_primary_run_identity=historical_run_identity,
        historical_primary_run_id=pins.historical_primary_run_id,
        historical_primary_artifact_root_sha256=(pins.historical_primary_artifact_root_sha256),
        confirmatory_config_semantic_sha256=(pins.confirmatory_config_semantic_sha256),
        confirmatory_plan_semantic_sha256=(pins.confirmatory_plan_semantic_sha256),
        confirmatory_controls_binding_sha256=(pins.confirmatory_controls_binding_sha256),
        adapter_receipt_sha256=canonical_sha256(payload),
    )


def _authorization_payload(
    *,
    predecessor_class: PredecessorClass,
    retry_of_run_id: str,
    successor_run_id: str,
    technical_amendment_directory: Path,
    technical_amendment_tree_root_sha256: str,
    runtime: OriginalConfirmatoryRuntimeReceipt,
    q_gate_adapter: OriginalConfirmatoryQGateAdapterReceipt,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy": ORIGINAL_CONFIRMATORY_AUTHORITY_POLICY,
        "predecessor_class": predecessor_class,
        "retry_of_run_id": retry_of_run_id,
        "successor_run_id": successor_run_id,
        "frozen_science": {
            "plan_semantic_sha256": ORIGINAL_CONFIRMATORY_PLAN_SEMANTIC_SHA256,
            "config_semantic_sha256": ORIGINAL_CONFIRMATORY_CONFIG_SEMANTIC_SHA256,
            "controls_binding_sha256": ORIGINAL_CONFIRMATORY_CONTROLS_BINDING_SHA256,
            "planned_cell_count": ORIGINAL_CONFIRMATORY_PLANNED_CELL_COUNT,
            "required_cell_count": ORIGINAL_CONFIRMATORY_REQUIRED_CELL_COUNT,
            "optional_cell_count": ORIGINAL_CONFIRMATORY_OPTIONAL_CELL_COUNT,
            "cnn_cell_count": ORIGINAL_CONFIRMATORY_CNN_CELL_COUNT,
            "checkpoint_count": ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT,
            "oof_fold_count": ORIGINAL_CONFIRMATORY_OOF_FOLD_COUNT,
        },
        "technical_amendment": {
            "directory": str(technical_amendment_directory),
            "tree_root_sha256": technical_amendment_tree_root_sha256,
        },
        "exact_execution_runtime": runtime.as_dict(),
        "q_gate_adapter": q_gate_adapter.as_dict(),
        "operation": {
            "max_attempt_count": 1,
            "automatic_retry_allowed": False,
            "predecessor_autodiscovery_allowed": False,
            "physical_copy_only": True,
            "resume_checkpoints": True,
            "resume_training_only_for_incomplete_fits": True,
            "terminal_checkpoint_refit": False,
            "supervisor_automatic_orphan_transition_allowed": False,
            "oof_artifacts_read": False,
            "metrics_artifacts_read": False,
            "ranking_artifacts_read": False,
            "outcome_artifacts_read": False,
        },
    }


def _verify_original_confirmatory_resume_authority(
    pins: OriginalConfirmatoryResumeAuthorityPins,
    *,
    predecessor_class: PredecessorClass,
    retry_of_run_id: str,
    successor_run_id: str,
) -> OriginalConfirmatoryResumeAuthorityReceipt:
    if not isinstance(pins, OriginalConfirmatoryResumeAuthorityPins):
        raise TypeError("authority pins must use OriginalConfirmatoryResumeAuthorityPins")
    for value in (
        pins.technical_amendment_tree_root_sha256,
        pins.execution_authority_tree_root_sha256,
        pins.authorization_file_sha256,
    ):
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise OriginalConfirmatoryResumeError(
                "amendment and authority hashes must be exact lowercase SHA-256"
            )
    amendment = _lexical_absolute(pins.technical_amendment_directory)
    authority = _lexical_absolute(pins.execution_authority_directory)
    if os.path.normcase(str(amendment)) == os.path.normcase(str(authority)):
        raise OriginalConfirmatoryResumeError(
            "technical amendment and execution authority must be distinct directories"
        )
    runtime = _verify_original_confirmatory_runtime(pins.runtime_pins)
    amendment_identity, amendment_root = _authority_tree_root(
        amendment,
        role="technical amendment",
    )
    authority_identity, authority_root = _authority_tree_root(
        authority,
        role="execution authority",
    )
    if (
        amendment_root != pins.technical_amendment_tree_root_sha256
        or authority_root != pins.execution_authority_tree_root_sha256
    ):
        raise OriginalConfirmatoryResumeError(
            "technical amendment or execution authority tree root differs from its pin"
        )
    q_gate_adapter = _verify_original_confirmatory_q_gate_adapter(pins.q_gate_adapter_pins)
    authorization_path = authority / ORIGINAL_CONFIRMATORY_AUTHORIZATION_FILENAME
    if not authorization_path.is_file():
        raise OriginalConfirmatoryResumeError(
            "execution authority lacks its exact resume authorization file"
        )
    authorization_file_sha256 = sha256_file(authorization_path)
    if authorization_file_sha256 != pins.authorization_file_sha256:
        raise OriginalConfirmatoryResumeError(
            "resume authorization file differs from its exact pin"
        )
    try:
        raw = json.loads(authorization_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OriginalConfirmatoryResumeError(
            "resume authorization is unreadable or invalid JSON"
        ) from exc
    expected = _authorization_payload(
        predecessor_class=predecessor_class,
        retry_of_run_id=retry_of_run_id,
        successor_run_id=successor_run_id,
        technical_amendment_directory=amendment,
        technical_amendment_tree_root_sha256=amendment_root,
        runtime=runtime,
        q_gate_adapter=q_gate_adapter,
    )
    if not isinstance(raw, Mapping) or dict(raw) != expected:
        raise OriginalConfirmatoryResumeError(
            "resume authorization does not exactly bind the requested successor"
        )
    amendment_identity_after, amendment_root_after = _authority_tree_root(
        amendment,
        role="technical amendment",
    )
    authority_identity_after, authority_root_after = _authority_tree_root(
        authority,
        role="execution authority",
    )
    if (
        amendment_identity_after != amendment_identity
        or authority_identity_after != authority_identity
        or amendment_root_after != amendment_root
        or authority_root_after != authority_root
    ):
        raise OriginalConfirmatoryResumeError(
            "technical amendment or execution authority changed during verification"
        )
    return OriginalConfirmatoryResumeAuthorityReceipt(
        technical_amendment_directory=amendment,
        technical_amendment_identity=amendment_identity,
        technical_amendment_tree_root_sha256=amendment_root,
        execution_authority_directory=authority,
        execution_authority_identity=authority_identity,
        execution_authority_tree_root_sha256=authority_root,
        authorization_file_sha256=authorization_file_sha256,
        authorization_binding_sha256=canonical_sha256(expected),
        runtime=runtime,
        q_gate_adapter=q_gate_adapter,
    )


def _plain_child_directory(path: Path, *, role: str) -> None:
    try:
        observed = path.lstat()
    except OSError as exc:
        raise OriginalConfirmatoryResumeError(f"{role} is inaccessible") from exc
    if not stat.S_ISDIR(observed.st_mode) or _is_reparse(path, observed):
        raise OriginalConfirmatoryResumeError(f"{role} is not a plain physical directory")


def _scan_checkpoint_tree(
    run_directory: Path,
    contract: OriginalConfirmatoryResumeContract,
) -> tuple[dict[str, Any], ...]:
    """Scan names/identities only inside canonical checkpoint directories."""

    expected_paths = {value.relative_path: value for value in contract.checkpoint_expectations}
    all_cell_ids = set(contract.all_cell_ids)
    cells_root = run_directory / "cells"
    if not cells_root.exists():
        return ()
    _plain_child_directory(cells_root, role="checkpoint cells root")
    records: list[dict[str, Any]] = []
    try:
        cell_entries = sorted(cells_root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise OriginalConfirmatoryResumeError(
            "cannot enumerate checkpoint cell directories"
        ) from exc
    for cell_directory in cell_entries:
        _plain_child_directory(cell_directory, role="checkpoint cell directory")
        if cell_directory.name not in all_cell_ids:
            raise OriginalConfirmatoryResumeError(
                f"unexpected cell directory in predecessor: {cell_directory.name}"
            )
        checkpoint_directory = cell_directory / "checkpoints"
        if not checkpoint_directory.exists():
            continue
        _plain_child_directory(checkpoint_directory, role="canonical checkpoint directory")
        try:
            checkpoint_entries = sorted(checkpoint_directory.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise OriginalConfirmatoryResumeError(
                "cannot enumerate canonical checkpoint directory"
            ) from exc
        for checkpoint in checkpoint_entries:
            try:
                observed = checkpoint.lstat()
            except OSError as exc:
                raise OriginalConfirmatoryResumeError(
                    "checkpoint tree entry is inaccessible"
                ) from exc
            relative = checkpoint.relative_to(run_directory).as_posix()
            if (
                relative not in expected_paths
                or not stat.S_ISREG(observed.st_mode)
                or _is_reparse(checkpoint, observed)
            ):
                raise OriginalConfirmatoryResumeError(
                    f"extra or noncanonical checkpoint tree entry: {relative}"
                )
            streams = _named_streams(checkpoint)
            if streams:
                raise OriginalConfirmatoryResumeError(
                    f"checkpoint has forbidden named streams: {relative}"
                )
            records.append(
                {
                    "relative_path": relative,
                    "device": int(observed.st_dev),
                    "inode": int(observed.st_ino),
                    "size_bytes": int(observed.st_size),
                    "link_count": int(observed.st_nlink),
                    "modified_time_ns": int(observed.st_mtime_ns),
                    "changed_time_ns": int(observed.st_ctime_ns),
                }
            )
    records.sort(key=lambda value: cast(str, value["relative_path"]))
    return tuple(records)


@dataclass(frozen=True, slots=True)
class _PredecessorQualificationProbe:
    root_inventory: tuple[dict[str, Any], ...]
    root_inventory_sha256: str
    status_sha256: str
    completion_evidence_sha256: str | None
    process_receipt: OriginalConfirmatoryProcessReceipt
    active_lock_paths: tuple[str, ...]
    stage_attestation_count: int
    disposition_record_count: int
    artifact_root_sha256: str | None
    artifact_manifest_sha256: str | None
    immutable_marker_sha256: str | None


def _read_json_object(path: Path, *, role: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OriginalConfirmatoryResumeError(f"{role} is unreadable or invalid JSON") from exc
    if not isinstance(raw, Mapping):
        raise OriginalConfirmatoryResumeError(f"{role} must be one JSON object")
    return dict(raw)


def _active_lock_paths(runs_root: Path) -> tuple[str, ...]:
    candidates: set[Path] = {
        *(runs_root / name for name in _ROOT_LOCK_NAMES),
    }
    try:
        candidates.update(runs_root.rglob("*.lock"))
    except OSError as exc:
        raise OriginalConfirmatoryResumeError("cannot enumerate runs-root locks") from exc
    present: list[str] = []
    for path in candidates:
        if not os.path.lexists(path):
            continue
        try:
            observed = path.lstat()
        except OSError as exc:
            raise OriginalConfirmatoryResumeError(
                f"active lock path is inaccessible: {path}"
            ) from exc
        if not stat.S_ISREG(observed.st_mode) or _is_reparse(path, observed):
            raise OriginalConfirmatoryResumeError(f"lock path is not one plain file: {path}")
        present.append(str(path.resolve()))
    return tuple(sorted(present, key=str.casefold))


def _process_create_time_unix_us(process: psutil.Process) -> int:
    try:
        return round(float(process.create_time()) * 1_000_000)
    except psutil.NoSuchProcess:
        raise
    except (psutil.AccessDenied, OSError, ValueError) as exc:
        raise OriginalConfirmatoryResumeError(
            "predecessor process creation identity is unavailable"
        ) from exc


def _inspect_predecessor_processes(
    request: BoundOriginalConfirmatoryResumeRequest,
) -> OriginalConfirmatoryProcessReceipt:
    expected_pid = request.predecessor_process_id
    expected_created = request.predecessor_process_create_time_unix_us
    identity_state: Literal["not_supplied", "gone", "pid_reused"] = "not_supplied"
    if expected_pid is not None:
        try:
            expected_process = psutil.Process(expected_pid)
            current_created = _process_create_time_unix_us(expected_process)
        except psutil.NoSuchProcess:
            current_created = None
        if current_created is None:
            identity_state = "gone"
        elif current_created == expected_created:
            raise OriginalConfirmatoryResumeError(
                "the exact predecessor PID/creation identity is still active"
            )
        else:
            identity_state = "pid_reused"

    inspector_pid = os.getpid()
    try:
        inspector_created = _process_create_time_unix_us(psutil.Process(inspector_pid))
    except psutil.NoSuchProcess as exc:
        raise OriginalConfirmatoryResumeError(
            "current inspector process identity disappeared"
        ) from exc
    predecessor_path_token = os.path.normcase(str(request.predecessor_run_directory)).casefold()
    run_id_token = request.retry_of_run_id.casefold()
    matching: set[int] = set()
    inspected = 0
    self_match_excluded = False
    supervisor_match_excluded = False
    for process in psutil.process_iter(["pid", "create_time", "cmdline"]):
        try:
            info = process.info
            pid = info.get("pid")
            created = info.get("create_time")
            command = info.get("cmdline")
            if type(pid) is not int or not isinstance(command, (list, tuple)):
                continue
            inspected += 1
            arguments = [str(value) for value in command if str(value)]
            normalized_joined = os.path.normcase("\0".join(arguments)).casefold()
            if predecessor_path_token in normalized_joined or any(
                argument.casefold() == run_id_token for argument in arguments
            ):
                observed_created = (
                    round(float(created) * 1_000_000)
                    if isinstance(created, (int, float)) and not isinstance(created, bool)
                    else None
                )
                if pid == inspector_pid and observed_created == inspector_created:
                    self_match_excluded = True
                    continue
                if (
                    pid == request.authority.runtime.supervisor_process_id
                    and observed_created
                    == request.authority.runtime.supervisor_process_create_time_unix_us
                ):
                    supervisor_match_excluded = True
                    continue
                matching.add(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    if matching:
        raise OriginalConfirmatoryResumeError(
            f"a live process still binds the explicit predecessor: {sorted(matching)}"
        )
    payload: dict[str, Any] = {
        "expected_process_id": expected_pid,
        "expected_create_time_unix_us": expected_created,
        "expected_identity_state": identity_state,
        "inspector_process_id": inspector_pid,
        "inspector_create_time_unix_us": inspector_created,
        "exact_inspector_match_excluded": self_match_excluded,
        "exact_authorized_supervisor_match_excluded": (supervisor_match_excluded),
        "matching_predecessor_process_ids": [],
        "inspected_process_count": inspected,
    }
    return OriginalConfirmatoryProcessReceipt(
        expected_process_id=expected_pid,
        expected_create_time_unix_us=expected_created,
        expected_identity_state=identity_state,
        inspector_process_id=inspector_pid,
        inspector_create_time_unix_us=inspector_created,
        exact_inspector_match_excluded=self_match_excluded,
        exact_authorized_supervisor_match_excluded=(supervisor_match_excluded),
        matching_predecessor_process_ids=(),
        inspected_process_count=inspected,
        receipt_sha256=canonical_sha256(payload),
    )


def _lineage_record_counts(
    request: BoundOriginalConfirmatoryResumeRequest,
) -> tuple[int, int]:
    try:
        stage_records = read_run_stage_attestations(
            request.runs_root / RUN_STAGE_ATTESTATION_REGISTRY_FILENAME
        )
        disposition_records = read_run_dispositions(
            request.runs_root / RUN_DISPOSITION_REGISTRY_FILENAME
        )
    except (OSError, TypeError, ValueError) as exc:
        raise OriginalConfirmatoryResumeError(
            "run stage/disposition ledgers are absent, corrupt, or unlocked"
        ) from exc
    stage_count = sum(record.get("run_id") == request.retry_of_run_id for record in stage_records)
    disposition_count = sum(
        record.get("run_id") == request.retry_of_run_id for record in disposition_records
    )
    if stage_count or disposition_count:
        raise OriginalConfirmatoryResumeError(
            "predecessor already has a stage attestation or disposition record"
        )
    return stage_count, disposition_count


def _require_failed_registry_row(
    request: BoundOriginalConfirmatoryResumeRequest,
) -> None:
    registry = request.runs_root / "registry.csv"
    try:
        with registry.open("r", encoding="utf-8", newline="") as handle:
            rows = tuple(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise OriginalConfirmatoryResumeError(
            "sealed failed predecessor registry is unreadable"
        ) from exc
    matches = [
        row
        for row in rows
        if row.get("run_id") == request.retry_of_run_id
        and row.get("status") == "failed"
        and row.get("experiment_name") == _EXPERIMENT_NAME
        and isinstance(row.get("run_path"), str)
        and os.path.normcase(str(_lexical_absolute(cast(str, row["run_path"]))))
        == os.path.normcase(str(request.predecessor_run_directory))
    ]
    if len(matches) != 1:
        raise OriginalConfirmatoryResumeError(
            "sealed failed predecessor lacks one exact failed registry row"
        )


def _probe_predecessor_qualification(
    request: BoundOriginalConfirmatoryResumeRequest,
) -> _PredecessorQualificationProbe:
    locks = _active_lock_paths(request.runs_root)
    if locks:
        raise OriginalConfirmatoryResumeError(
            f"active runs-root locks make predecessor state ambiguous: {list(locks)}"
        )
    process_receipt = _inspect_predecessor_processes(request)
    inventory = _strict_tree_inventory(
        request.predecessor_run_directory,
        role="explicit predecessor run",
        hash_file_contents=False,
    )
    inventory_sha256 = canonical_sha256(list(inventory))
    status_path = request.predecessor_run_directory / "status.json"
    status = _read_json_object(status_path, role="predecessor status")
    if (
        status.get("run_id") != request.retry_of_run_id
        or status.get("experiment_name") != _EXPERIMENT_NAME
    ):
        raise OriginalConfirmatoryResumeError(
            "predecessor status does not bind the exact confirmatory run"
        )
    stage_count, disposition_count = _lineage_record_counts(request)
    artifact_root_sha256: str | None = None
    artifact_manifest_sha256: str | None = None
    immutable_marker_sha256: str | None = None
    completion_sha256: str | None = None

    if request.predecessor_class == "sealed_failed_demoted":
        integrity = verify_run_integrity(request.predecessor_run_directory)
        if (
            not integrity.valid
            or not integrity.registry_record_present
            or integrity.run_id != request.retry_of_run_id
            or not isinstance(integrity.expected_root_sha256, str)
            or _SHA256.fullmatch(integrity.expected_root_sha256) is None
            or integrity.actual_root_sha256 != integrity.expected_root_sha256
        ):
            raise OriginalConfirmatoryResumeError(
                f"failed predecessor is not an exact integrity-valid sealed run: {integrity.errors}"
            )
        manifest_path = request.predecessor_run_directory / ARTIFACT_MANIFEST_FILENAME
        marker_path = request.predecessor_run_directory / IMMUTABLE_MARKER
        completion_path = request.predecessor_run_directory / "completion_evidence.json"
        manifest = _read_json_object(
            manifest_path,
            role="failed predecessor artifact manifest",
        )
        marker = _read_json_object(
            marker_path,
            role="failed predecessor immutable marker",
        )
        completion = _read_json_object(
            completion_path,
            role="failed predecessor demoted completion",
        )
        marker_path_value = marker.get("run_path")
        traceback_name = status.get("traceback")
        if (
            status.get("status") != "failed"
            or manifest.get("status") != "failed"
            or manifest.get("run_id") != request.retry_of_run_id
            or marker.get("status") != "failed"
            or marker.get("run_id") != request.retry_of_run_id
            or manifest.get("artifact_root_sha256") != integrity.expected_root_sha256
            or marker.get("artifact_root_sha256") != integrity.expected_root_sha256
            or marker.get("artifact_manifest_sha256") != sha256_file(manifest_path)
            or not isinstance(marker_path_value, str)
            or os.path.normcase(str(_lexical_absolute(marker_path_value)))
            != os.path.normcase(str(request.predecessor_run_directory))
            or completion.get("completion_stage") is not None
            or completion.get("study_outcome_eligible") is not False
            or completion.get("valid_completion_claim") is not False
            or completion.get("artifact_scope") != _REAL_ARTIFACT_SCOPE
            or not str(completion.get("runner_failure", "")).strip()
            or traceback_name != "traceback.txt"
            or not (request.predecessor_run_directory / "traceback.txt").is_file()
        ):
            raise OriginalConfirmatoryResumeError(
                "sealed predecessor is not an exact failed demoted confirmatory run"
            )
        _require_failed_registry_row(request)
        artifact_root_sha256 = integrity.expected_root_sha256
        artifact_manifest_sha256 = sha256_file(manifest_path)
        immutable_marker_sha256 = sha256_file(marker_path)
        completion_sha256 = sha256_file(completion_path)
    else:
        if status.get("status") != "running":
            raise OriginalConfirmatoryResumeError(
                "unsealed orphan must retain exact nonterminal status='running'"
            )
        forbidden = sorted(
            name
            for name in _ORPHAN_FORBIDDEN_TERMINAL_ARTIFACTS
            if os.path.lexists(request.predecessor_run_directory / name)
        )
        if forbidden:
            raise OriginalConfirmatoryResumeError(
                f"unsealed orphan contains terminal/success/completion artifacts: {forbidden}"
            )
        if (
            request.orphan_manual_diagnosis is not True
            or process_receipt.expected_identity_state not in {"gone", "pid_reused"}
        ):
            raise OriginalConfirmatoryResumeError(
                "unsealed orphan lacks a manual dead-process diagnosis"
            )

    return _PredecessorQualificationProbe(
        root_inventory=inventory,
        root_inventory_sha256=inventory_sha256,
        status_sha256=sha256_file(status_path),
        completion_evidence_sha256=completion_sha256,
        process_receipt=process_receipt,
        active_lock_paths=locks,
        stage_attestation_count=stage_count,
        disposition_record_count=disposition_count,
        artifact_root_sha256=artifact_root_sha256,
        artifact_manifest_sha256=artifact_manifest_sha256,
        immutable_marker_sha256=immutable_marker_sha256,
    )


def _finalize_predecessor_qualification(
    request: BoundOriginalConfirmatoryResumeRequest,
    before: _PredecessorQualificationProbe,
) -> OriginalConfirmatoryPredecessorQualification:
    after = _probe_predecessor_qualification(request)
    if (
        after.root_inventory != before.root_inventory
        or after.root_inventory_sha256 != before.root_inventory_sha256
        or after.status_sha256 != before.status_sha256
        or after.completion_evidence_sha256 != before.completion_evidence_sha256
        or after.artifact_root_sha256 != before.artifact_root_sha256
        or after.artifact_manifest_sha256 != before.artifact_manifest_sha256
        or after.immutable_marker_sha256 != before.immutable_marker_sha256
        or after.stage_attestation_count != before.stage_attestation_count
        or after.disposition_record_count != before.disposition_record_count
    ):
        raise OriginalConfirmatoryResumeError(
            "predecessor root, seal, or ledger state changed during inspection"
        )
    policy = (
        ORIGINAL_CONFIRMATORY_FAILED_POLICY
        if request.predecessor_class == "sealed_failed_demoted"
        else ORIGINAL_CONFIRMATORY_ORPHAN_POLICY
    )
    terminal_status = (
        "failed"
        if request.predecessor_class == "sealed_failed_demoted"
        else "unsealed_interrupted_orphan"
    )
    payload = {
        "schema_version": 1,
        "predecessor_class": request.predecessor_class,
        "policy": policy,
        "terminal_status": terminal_status,
        "sealed_integrity_valid": (request.predecessor_class == "sealed_failed_demoted"),
        "integrity_registry_record_present": (request.predecessor_class == "sealed_failed_demoted"),
        "artifact_root_sha256": before.artifact_root_sha256,
        "artifact_manifest_sha256": before.artifact_manifest_sha256,
        "immutable_marker_sha256": before.immutable_marker_sha256,
        "status_sha256": before.status_sha256,
        "completion_evidence_sha256": before.completion_evidence_sha256,
        "process_receipt": after.process_receipt.as_dict(),
        "active_lock_paths_before": list(before.active_lock_paths),
        "active_lock_paths_after": list(after.active_lock_paths),
        "stage_attestation_count": before.stage_attestation_count,
        "disposition_record_count": before.disposition_record_count,
        "root_inventory_before_sha256": before.root_inventory_sha256,
        "root_inventory_after_sha256": after.root_inventory_sha256,
        "technical_amendment_tree_root_sha256": (
            request.authority.technical_amendment_tree_root_sha256
        ),
        "execution_authority_tree_root_sha256": (
            request.authority.execution_authority_tree_root_sha256
        ),
        "authorization_binding_sha256": (request.authority.authorization_binding_sha256),
        "supervisor_automatic_orphan_transition_allowed": False,
        "oof_artifacts_parsed": False,
        "metrics_artifacts_parsed": False,
        "ranking_artifacts_parsed": False,
        "outcome_artifacts_parsed": False,
    }
    qualification_sha256 = canonical_sha256(payload)
    return OriginalConfirmatoryPredecessorQualification(
        predecessor_class=request.predecessor_class,
        policy=policy,
        terminal_status=terminal_status,
        sealed_integrity_valid=(request.predecessor_class == "sealed_failed_demoted"),
        integrity_registry_record_present=(request.predecessor_class == "sealed_failed_demoted"),
        artifact_root_sha256=before.artifact_root_sha256,
        artifact_manifest_sha256=before.artifact_manifest_sha256,
        immutable_marker_sha256=before.immutable_marker_sha256,
        status_sha256=before.status_sha256,
        completion_evidence_sha256=before.completion_evidence_sha256,
        process_receipt=after.process_receipt,
        active_lock_paths_before=before.active_lock_paths,
        active_lock_paths_after=after.active_lock_paths,
        stage_attestation_count=before.stage_attestation_count,
        disposition_record_count=before.disposition_record_count,
        root_inventory_before_sha256=before.root_inventory_sha256,
        root_inventory_after_sha256=after.root_inventory_sha256,
        qualification_sha256=qualification_sha256,
    )


def _read_production_fit_state(
    path: str | Path,
    *,
    expected_configuration: Mapping[str, Any],
) -> tuple[int, bool]:
    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise OriginalConfirmatoryResumeError("validated checkpoint payload is no longer a mapping")
    completed_epochs = payload.get("completed_epochs")
    early_stopping = payload.get("early_stopping_state")
    if (
        type(completed_epochs) is not int
        or not isinstance(early_stopping, Mapping)
        or type(early_stopping.get("stopped_early")) is not bool
        or type(expected_configuration.get("epochs")) is not int
    ):
        raise OriginalConfirmatoryResumeError("validated checkpoint lacks exact fit-state fields")
    return int(completed_epochs), bool(early_stopping["stopped_early"])


def _validate_bound_request_unchanged(
    request: BoundOriginalConfirmatoryResumeRequest,
    *,
    require_successor_absent: bool,
) -> None:
    if not isinstance(request, BoundOriginalConfirmatoryResumeRequest):
        raise TypeError("resume operation requires a bound request")
    current_identity = _plain_directory_identity(
        request.predecessor_run_directory,
        role="explicit predecessor",
    )
    if current_identity != request.predecessor_identity:
        raise OriginalConfirmatoryResumeError(
            "explicit predecessor identity changed after request binding"
        )
    authority = _verify_original_confirmatory_resume_authority(
        OriginalConfirmatoryResumeAuthorityPins(
            technical_amendment_directory=(request.authority.technical_amendment_directory),
            technical_amendment_tree_root_sha256=(
                request.authority.technical_amendment_tree_root_sha256
            ),
            execution_authority_directory=(request.authority.execution_authority_directory),
            execution_authority_tree_root_sha256=(
                request.authority.execution_authority_tree_root_sha256
            ),
            authorization_file_sha256=(request.authority.authorization_file_sha256),
            runtime_pins=_runtime_pins_from_receipt(request.authority.runtime),
            q_gate_adapter_pins=_q_gate_adapter_pins_from_receipt(request.authority.q_gate_adapter),
        ),
        predecessor_class=request.predecessor_class,
        retry_of_run_id=request.retry_of_run_id,
        successor_run_id=request.successor_run_id,
    )
    if authority != request.authority:
        raise OriginalConfirmatoryResumeError(
            "exact amendment or execution authority identity changed"
        )
    recomputed = canonical_sha256(
        _canonical_request_payload(
            runs_root=request.runs_root,
            predecessor=request.predecessor_run_directory,
            successor=request.successor_run_directory,
            retry_of_run_id=request.retry_of_run_id,
            successor_run_id=request.successor_run_id,
            predecessor_class=request.predecessor_class,
            authority=request.authority,
            predecessor_process_id=request.predecessor_process_id,
            predecessor_process_create_time_unix_us=(
                request.predecessor_process_create_time_unix_us
            ),
            orphan_manual_diagnosis=request.orphan_manual_diagnosis,
            predecessor_identity=request.predecessor_identity,
        )
    )
    if recomputed != request.request_sha256:
        raise OriginalConfirmatoryResumeError("bound resume request hash differs")
    if require_successor_absent and (
        request.successor_run_directory.exists() or request.successor_run_directory.is_symlink()
    ):
        raise OriginalConfirmatoryResumeError(
            "successor appeared before predecessor inspection completed"
        )


def _inspect_original_confirmatory_predecessor(
    request: BoundOriginalConfirmatoryResumeRequest,
    contract: OriginalConfirmatoryResumeContract,
    *,
    validator: StrictCheckpointValidator,
    fit_state_reader: CheckpointFitStateReader,
) -> OriginalConfirmatoryPredecessorSnapshot:
    if (
        not isinstance(contract, OriginalConfirmatoryResumeContract)
        or len(contract.checkpoint_expectations) != ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT
        or len(contract.all_cell_ids) != ORIGINAL_CONFIRMATORY_PLANNED_CELL_COUNT
    ):
        raise OriginalConfirmatoryResumeError(
            "resume inspection requires the complete original-confirmatory contract"
        )
    _validate_bound_request_unchanged(request, require_successor_absent=True)
    qualification_before = _probe_predecessor_qualification(request)
    tree_before = _scan_checkpoint_tree(
        request.predecessor_run_directory,
        contract,
    )
    tree_before_sha256 = canonical_sha256(list(tree_before))
    fit_states: dict[str, tuple[int, bool]] = {}

    def strict_validator(
        path: str | Path,
        *,
        expected_configuration: Mapping[str, Any],
        expected_model_metadata: Mapping[str, Any],
        expected_data_and_split_sha256: Mapping[str, str],
    ) -> None:
        validator(
            path,
            expected_configuration=expected_configuration,
            expected_model_metadata=expected_model_metadata,
            expected_data_and_split_sha256=expected_data_and_split_sha256,
        )
        completed_epochs, stopped_early = fit_state_reader(
            path,
            expected_configuration=expected_configuration,
        )
        maximum_epochs = expected_configuration.get("epochs")
        if (
            type(maximum_epochs) is not int
            or type(completed_epochs) is not int
            or type(stopped_early) is not bool
            or not 1 <= completed_epochs <= int(maximum_epochs)
        ):
            raise OriginalConfirmatoryResumeError(
                "checkpoint fit state is outside the frozen epoch contract"
            )
        fit_states[Path(path).relative_to(request.predecessor_run_directory).as_posix()] = (
            completed_epochs,
            stopped_early,
        )

    try:
        base_snapshot = inspect_read_only_resume_predecessor(
            request.predecessor_run_directory,
            retry_of_run_id=request.retry_of_run_id,
            checkpoint_allowlist=contract.checkpoint_expectations,
            validator=strict_validator,
        )
    except ResourceBoundedResumeError as exc:
        raise OriginalConfirmatoryResumeError(str(exc)) from exc
    tree_after = _scan_checkpoint_tree(
        request.predecessor_run_directory,
        contract,
    )
    tree_after_sha256 = canonical_sha256(list(tree_after))
    if tree_after != tree_before:
        raise OriginalConfirmatoryResumeError(
            "checkpoint tree changed during predecessor inspection"
        )
    present_paths = {
        record.relative_path for record in base_snapshot.records if record.decision == "resume"
    }
    if present_paths != set(fit_states):
        raise OriginalConfirmatoryResumeError(
            "strict fit-state validation did not cover every present checkpoint"
        )
    if not present_paths:
        raise OriginalConfirmatoryResumeError(
            "successor resume requires at least one validated predecessor checkpoint"
        )
    expectation_by_path = {value.relative_path: value for value in contract.checkpoint_expectations}
    states: list[OriginalConfirmatoryCheckpointState] = []
    for record in base_snapshot.records:
        expectation = expectation_by_path[record.relative_path]
        maximum_epochs = cast(int, expectation.expected_configuration["epochs"])
        if record.decision == "missing_fresh":
            states.append(
                OriginalConfirmatoryCheckpointState(
                    relative_path=record.relative_path,
                    cell_id=record.cell_id,
                    fold_id=record.fold_id,
                    action="fresh_fit",
                    completed_epochs=None,
                    maximum_epochs=maximum_epochs,
                    stopped_early=None,
                    source_size_bytes=None,
                    source_sha256=None,
                )
            )
            continue
        completed_epochs, stopped_early = fit_states[record.relative_path]
        terminal = stopped_early or completed_epochs == maximum_epochs
        states.append(
            OriginalConfirmatoryCheckpointState(
                relative_path=record.relative_path,
                cell_id=record.cell_id,
                fold_id=record.fold_id,
                action=(
                    "restore_terminal_checkpoint_without_fit"
                    if terminal
                    else "resume_incomplete_fit"
                ),
                completed_epochs=completed_epochs,
                maximum_epochs=maximum_epochs,
                stopped_early=stopped_early,
                source_size_bytes=record.size_bytes,
                source_sha256=record.sha256,
            )
        )
    state_payload = [state.as_dict() for state in states]
    snapshot_payload = {
        "schema_version": 1,
        "policy": ORIGINAL_CONFIRMATORY_RESUME_POLICY,
        "request_sha256": request.request_sha256,
        "contract": contract.as_dict(),
        "predecessor_qualification_pending_final_readback": True,
        "base_snapshot_sha256": base_snapshot.snapshot_sha256,
        "checkpoint_tree_before_sha256": tree_before_sha256,
        "checkpoint_tree_after_sha256": tree_after_sha256,
        "checkpoint_states": state_payload,
        "checkpoint_states_sha256": canonical_sha256(state_payload),
        "automatic_retry_allowed": False,
        "predecessor_autodiscovery_used": False,
        "oof_artifacts_read": False,
        "metrics_artifacts_read": False,
        "ranking_artifacts_read": False,
        "outcome_artifacts_read": False,
    }
    _validate_bound_request_unchanged(request, require_successor_absent=True)
    qualification = _finalize_predecessor_qualification(
        request,
        qualification_before,
    )
    snapshot_payload["predecessor_qualification_pending_final_readback"] = False
    snapshot_payload["predecessor_qualification"] = qualification.as_dict()
    return OriginalConfirmatoryPredecessorSnapshot(
        request=request,
        contract=contract,
        qualification=qualification,
        base_snapshot=base_snapshot,
        checkpoint_states=tuple(states),
        checkpoint_tree_before_sha256=tree_before_sha256,
        checkpoint_tree_after_sha256=tree_after_sha256,
        snapshot_sha256=canonical_sha256(snapshot_payload),
    )


def inspect_original_confirmatory_predecessor(
    request: BoundOriginalConfirmatoryResumeRequest,
    contract: OriginalConfirmatoryResumeContract,
) -> OriginalConfirmatoryPredecessorSnapshot:
    """Production predecessor inspection with the strict Torch checkpoint validator."""

    return _inspect_original_confirmatory_predecessor(
        request,
        contract,
        validator=validate_confirmatory_checkpoint_artifact,
        fit_state_reader=_read_production_fit_state,
    )


def _require_qualification_unchanged(
    snapshot: OriginalConfirmatoryPredecessorSnapshot,
) -> None:
    observed = _probe_predecessor_qualification(snapshot.request)
    expected = snapshot.qualification
    if (
        observed.active_lock_paths
        or observed.root_inventory_sha256 != expected.root_inventory_after_sha256
        or observed.status_sha256 != expected.status_sha256
        or observed.completion_evidence_sha256 != expected.completion_evidence_sha256
        or observed.artifact_root_sha256 != expected.artifact_root_sha256
        or observed.artifact_manifest_sha256 != expected.artifact_manifest_sha256
        or observed.immutable_marker_sha256 != expected.immutable_marker_sha256
        or observed.stage_attestation_count != expected.stage_attestation_count
        or observed.disposition_record_count != expected.disposition_record_count
    ):
        raise OriginalConfirmatoryResumeError(
            "qualified predecessor changed before/during checkpoint import"
        )


def _copy_original_confirmatory_checkpoints(
    snapshot: OriginalConfirmatoryPredecessorSnapshot,
    *,
    validator: StrictCheckpointValidator,
) -> OriginalConfirmatoryResumeCopyReceipt:
    if not isinstance(snapshot, OriginalConfirmatoryPredecessorSnapshot):
        raise TypeError("copy requires OriginalConfirmatoryPredecessorSnapshot")
    request = snapshot.request
    _validate_bound_request_unchanged(request, require_successor_absent=False)
    _require_qualification_unchanged(snapshot)
    _plain_directory_identity(
        request.successor_run_directory,
        role="new successor run directory",
    )
    before = _scan_checkpoint_tree(request.successor_run_directory, snapshot.contract)
    if before:
        raise OriginalConfirmatoryResumeError(
            "successor already contains checkpoint files before physical import"
        )
    base_receipt = copy_validated_resume_checkpoints(
        snapshot.base_snapshot,
        request.successor_run_directory,
        validator=validator,
    )
    after = _scan_checkpoint_tree(request.successor_run_directory, snapshot.contract)
    expected_imported = {
        state.relative_path
        for state in snapshot.checkpoint_states
        if state.source_sha256 is not None
    }
    if {cast(str, row["relative_path"]) for row in after} != expected_imported:
        raise OriginalConfirmatoryResumeError(
            "successor checkpoint tree differs from the exact imported allowlist"
        )
    _require_qualification_unchanged(snapshot)
    _validate_bound_request_unchanged(request, require_successor_absent=False)
    return OriginalConfirmatoryResumeCopyReceipt(
        snapshot_sha256=snapshot.snapshot_sha256,
        base_receipt=base_receipt,
        destination_checkpoint_tree_sha256=canonical_sha256(list(after)),
    )


def copy_original_confirmatory_checkpoints(
    snapshot: OriginalConfirmatoryPredecessorSnapshot,
) -> OriginalConfirmatoryResumeCopyReceipt:
    """Physically import only validated allowlisted checkpoints without retry."""

    return _copy_original_confirmatory_checkpoints(
        snapshot,
        validator=validate_confirmatory_checkpoint_artifact,
    )


def build_original_confirmatory_successor_execution_contract(
    snapshot: OriginalConfirmatoryPredecessorSnapshot,
    receipt: OriginalConfirmatoryResumeCopyReceipt,
) -> OriginalConfirmatorySuccessorExecutionContract:
    """Bind the existing matrix executor to global resume and exact fold actions."""

    if (
        not isinstance(snapshot, OriginalConfirmatoryPredecessorSnapshot)
        or not isinstance(receipt, OriginalConfirmatoryResumeCopyReceipt)
        or receipt.snapshot_sha256 != snapshot.snapshot_sha256
    ):
        raise OriginalConfirmatoryResumeError(
            "successor execution contract lacks an exact snapshot/copy lineage"
        )
    destination = snapshot.request.successor_run_directory
    tree = _scan_checkpoint_tree(destination, snapshot.contract)
    tree_sha256 = canonical_sha256(list(tree))
    if tree_sha256 != receipt.destination_checkpoint_tree_sha256:
        raise OriginalConfirmatoryResumeError(
            "successor checkpoint tree changed after physical import"
        )
    existing_paths = {cast(str, record["relative_path"]) for record in tree}
    directives: list[OriginalConfirmatoryFitDirective] = []
    for state in snapshot.checkpoint_states:
        exists = state.relative_path in existing_paths
        expected_exists = state.action != "fresh_fit"
        if exists != expected_exists:
            raise OriginalConfirmatoryResumeError(
                "existing checkpoint entered a fresh branch or a predeclared "
                f"checkpoint disappeared: {state.relative_path}"
            )
        if state.action == "fresh_fit":
            if state.completed_epochs is not None or state.stopped_early is not None:
                raise OriginalConfirmatoryResumeError(
                    "fresh fit carries predecessor training state"
                )
            directives.append(
                OriginalConfirmatoryFitDirective(
                    relative_path=state.relative_path,
                    cell_id=state.cell_id,
                    fold_id=state.fold_id,
                    action=state.action,
                    checkpoint_sha256=None,
                    checkpoint_must_exist=False,
                    pass_resume_true_to_fit=False,
                    training_required=True,
                    next_epoch_index=0,
                    maximum_epochs=state.maximum_epochs,
                )
            )
            continue
        completed_epochs = state.completed_epochs
        if (
            type(completed_epochs) is not int
            or not 1 <= completed_epochs <= state.maximum_epochs
            or not isinstance(state.source_sha256, str)
            or _SHA256.fullmatch(state.source_sha256) is None
        ):
            raise OriginalConfirmatoryResumeError(
                "imported checkpoint has invalid completed epoch state"
            )
        terminal = state.stopped_early is True or completed_epochs == state.maximum_epochs
        if terminal != (state.action == "restore_terminal_checkpoint_without_fit"):
            raise OriginalConfirmatoryResumeError(
                "terminal/incomplete checkpoint action is inconsistent"
            )
        directives.append(
            OriginalConfirmatoryFitDirective(
                relative_path=state.relative_path,
                cell_id=state.cell_id,
                fold_id=state.fold_id,
                action=state.action,
                checkpoint_sha256=state.source_sha256,
                checkpoint_must_exist=True,
                pass_resume_true_to_fit=True,
                training_required=not terminal,
                next_epoch_index=completed_epochs,
                maximum_epochs=state.maximum_epochs,
            )
        )
    if len(directives) != ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT or {
        value.relative_path for value in directives
    } != {value.relative_path for value in snapshot.contract.checkpoint_expectations}:
        raise OriginalConfirmatoryResumeError(
            "successor execution directives do not cover the exact 180 folds"
        )
    directive_payload = [value.as_dict() for value in directives]
    return OriginalConfirmatorySuccessorExecutionContract(
        snapshot_sha256=snapshot.snapshot_sha256,
        destination_checkpoint_tree_sha256=tree_sha256,
        resume_checkpoints=True,
        directives=tuple(directives),
        directives_sha256=canonical_sha256(directive_payload),
    )


def _require_original_confirmatory_fit_directive(
    execution: OriginalConfirmatorySuccessorExecutionContract,
    snapshot: OriginalConfirmatoryPredecessorSnapshot,
    *,
    cell_id: str,
    fold_id: int,
    validator: StrictCheckpointValidator,
    fit_state_reader: CheckpointFitStateReader,
) -> OriginalConfirmatoryFitDirective:
    """Revalidate one fold immediately before fit; never infer from existence."""

    if (
        not isinstance(execution, OriginalConfirmatorySuccessorExecutionContract)
        or not isinstance(snapshot, OriginalConfirmatoryPredecessorSnapshot)
        or execution.snapshot_sha256 != snapshot.snapshot_sha256
        or execution.resume_checkpoints is not True
        or canonical_sha256([directive.as_dict() for directive in execution.directives])
        != execution.directives_sha256
    ):
        raise OriginalConfirmatoryResumeError(
            "fit-time execution contract is absent, altered, or not global-resume"
        )
    matches = [
        directive
        for directive in execution.directives
        if directive.cell_id == cell_id and directive.fold_id == fold_id
    ]
    if len(matches) != 1:
        raise OriginalConfirmatoryResumeError(
            "fit request is outside the closed cell/fold directive map"
        )
    directive = matches[0]
    expectation_matches = [
        expectation
        for expectation in snapshot.contract.checkpoint_expectations
        if expectation.relative_path == directive.relative_path
        and expectation.cell_id == cell_id
        and expectation.fold_id == fold_id
    ]
    if len(expectation_matches) != 1:
        raise OriginalConfirmatoryResumeError(
            "fit directive does not bind one exact frozen expectation"
        )
    expectation = expectation_matches[0]
    checkpoint = snapshot.request.successor_run_directory / directive.relative_path
    if directive.action == "fresh_fit":
        if (
            directive.checkpoint_must_exist
            or directive.pass_resume_true_to_fit
            or directive.checkpoint_sha256 is not None
            or os.path.lexists(checkpoint)
        ):
            raise OriginalConfirmatoryResumeError(
                "fresh fit is legal only for its predeclared absent checkpoint"
            )
        return directive
    if (
        not directive.checkpoint_must_exist
        or not directive.pass_resume_true_to_fit
        or not isinstance(directive.checkpoint_sha256, str)
        or _SHA256.fullmatch(directive.checkpoint_sha256) is None
    ):
        raise OriginalConfirmatoryResumeError(
            "resume directive lacks its exact copied checkpoint binding"
        )
    try:
        observed = checkpoint.lstat()
    except OSError as exc:
        raise OriginalConfirmatoryResumeError(
            "authorized copied checkpoint disappeared before fit; fresh fallback is forbidden"
        ) from exc
    observed_sha256 = _hash_private_file(
        checkpoint,
        observed,
        role="fit-time copied checkpoint",
    )
    if observed_sha256 != directive.checkpoint_sha256:
        raise OriginalConfirmatoryResumeError(
            "fit-time copied checkpoint content differs from its import receipt"
        )
    try:
        validator(
            checkpoint,
            expected_configuration=expectation.expected_configuration,
            expected_model_metadata=expectation.expected_model_metadata,
            expected_data_and_split_sha256=(expectation.expected_data_and_split_sha256),
        )
        completed_epochs, stopped_early = fit_state_reader(
            checkpoint,
            expected_configuration=expectation.expected_configuration,
        )
    except Exception as exc:
        raise OriginalConfirmatoryResumeError(
            "fit-time copied checkpoint failed strict structural revalidation"
        ) from exc
    terminal = stopped_early or completed_epochs == directive.maximum_epochs
    if (
        completed_epochs != directive.next_epoch_index
        or terminal != (directive.action == "restore_terminal_checkpoint_without_fit")
        or directive.training_required == terminal
    ):
        raise OriginalConfirmatoryResumeError(
            "fit-time checkpoint state differs from its closed resume action"
        )
    return directive


def require_original_confirmatory_fit_directive(
    execution: OriginalConfirmatorySuccessorExecutionContract,
    snapshot: OriginalConfirmatoryPredecessorSnapshot,
    *,
    cell_id: str,
    fold_id: int,
) -> OriginalConfirmatoryFitDirective:
    """Production fail-fast guard to call immediately before each CNN fold fit.

    ``OriginalConfirmatoryResumeError`` is a run-level structural violation and
    must propagate out of matrix execution; a per-cell failure handler must not
    convert it into an optional/required cell result.
    """

    return _require_original_confirmatory_fit_directive(
        execution,
        snapshot,
        cell_id=cell_id,
        fold_id=fold_id,
        validator=validate_confirmatory_checkpoint_artifact,
        fit_state_reader=_read_production_fit_state,
    )


def _build_original_confirmatory_resume_evidence(
    snapshot: OriginalConfirmatoryPredecessorSnapshot,
    receipt: OriginalConfirmatoryResumeCopyReceipt,
    *,
    validator: StrictCheckpointValidator,
) -> dict[str, Any]:
    if (
        not isinstance(snapshot, OriginalConfirmatoryPredecessorSnapshot)
        or not isinstance(receipt, OriginalConfirmatoryResumeCopyReceipt)
        or receipt.snapshot_sha256 != snapshot.snapshot_sha256
    ):
        raise OriginalConfirmatoryResumeError(
            "resume evidence snapshot/copy receipt lineage differs"
        )
    base_evidence = build_resource_bounded_resume_evidence(
        snapshot.base_snapshot,
        receipt.base_receipt,
        validator=validator,
    )
    execution_contract = build_original_confirmatory_successor_execution_contract(
        snapshot,
        receipt,
    )
    payload: dict[str, Any] = {
        "schema_version": ORIGINAL_CONFIRMATORY_RESUME_EVIDENCE_SCHEMA_VERSION,
        "policy": ORIGINAL_CONFIRMATORY_RESUME_POLICY,
        "copy_policy": ORIGINAL_CONFIRMATORY_COPY_POLICY,
        "read_scope": ORIGINAL_CONFIRMATORY_READ_SCOPE,
        "request": snapshot.request.as_dict(),
        "contract": snapshot.contract.as_dict(),
        "predecessor_qualification": snapshot.qualification.as_dict(),
        "predecessor_snapshot_sha256": snapshot.snapshot_sha256,
        "base_resume_evidence": base_evidence,
        "base_resume_evidence_sha256": base_evidence["evidence_without_self_hash_sha256"],
        "destination_checkpoint_tree_sha256": (receipt.destination_checkpoint_tree_sha256),
        "successor_execution_contract": {
            "snapshot_sha256": execution_contract.snapshot_sha256,
            "destination_checkpoint_tree_sha256": (
                execution_contract.destination_checkpoint_tree_sha256
            ),
            "resume_checkpoints": execution_contract.resume_checkpoints,
            "directives_sha256": execution_contract.directives_sha256,
        },
        "checkpoint_states": [state.as_dict() for state in snapshot.checkpoint_states],
        "imported_checkpoint_count": snapshot.imported_checkpoint_count,
        "incomplete_fit_count": snapshot.incomplete_fit_count,
        "terminal_restore_without_fit_count": snapshot.terminal_restore_count,
        "fresh_fit_count": snapshot.fresh_fit_count,
        "resume_training_allowed_only_for_incomplete_fits": True,
        "terminal_checkpoints_train_again": False,
        "predecessor_autodiscovery_used": False,
        "automatic_retry_allowed": False,
        "max_attempt_count": 1,
        "resume_checkpoints": True,
        "hardlinks_used": False,
        "source_overwrite_used": False,
        "oof_artifacts_read": False,
        "metrics_artifacts_read": False,
        "ranking_artifacts_read": False,
        "outcome_artifacts_read": False,
    }
    payload[ORIGINAL_CONFIRMATORY_RESUME_EVIDENCE_HASH_FIELD] = canonical_sha256(payload)
    return payload


def build_original_confirmatory_resume_evidence(
    snapshot: OriginalConfirmatoryPredecessorSnapshot,
    receipt: OriginalConfirmatoryResumeCopyReceipt,
) -> dict[str, Any]:
    """Return exact evidence for persistence by the new successor RunTracker."""

    return _build_original_confirmatory_resume_evidence(
        snapshot,
        receipt,
        validator=validate_confirmatory_checkpoint_artifact,
    )


__all__ = [
    "ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT",
    "ORIGINAL_CONFIRMATORY_CONFIG_SEMANTIC_SHA256",
    "ORIGINAL_CONFIRMATORY_CONTROLS_BINDING_SHA256",
    "ORIGINAL_CONFIRMATORY_COPY_POLICY",
    "ORIGINAL_CONFIRMATORY_PLAN_SEMANTIC_SHA256",
    "ORIGINAL_CONFIRMATORY_READ_SCOPE",
    "ORIGINAL_CONFIRMATORY_RESUME_EVIDENCE_HASH_FIELD",
    "ORIGINAL_CONFIRMATORY_RESUME_POLICY",
    "BoundOriginalConfirmatoryResumeRequest",
    "OriginalConfirmatoryCheckpointState",
    "OriginalConfirmatoryPredecessorSnapshot",
    "OriginalConfirmatoryResumeContract",
    "OriginalConfirmatoryResumeCopyReceipt",
    "OriginalConfirmatoryResumeError",
    "OriginalConfirmatoryResumeRequest",
    "bind_original_confirmatory_resume_request",
    "build_original_confirmatory_resume_contract",
    "build_original_confirmatory_resume_evidence",
    "copy_original_confirmatory_checkpoints",
    "inspect_original_confirmatory_predecessor",
]
