"""Fixed terminal handlers for the original-confirmatory execution capsule.

The two public handlers in this module are the only terminal targets selected by
``original_confirmatory_capsule_entry``:

``verify-preterminal``
    Revalidates Q, E, the downstream supervisor spec, launch receipts, and the
    declared terminal scientific artifacts.  It publishes one CREATE_NEW
    preterminal pin (P), keeps the creating handle open, emits one canonical READY
    line, and closes the handle only after validating the supervisor ACK.

``verify-terminal``
    Revalidates the same authority and then participates in the single
    event-driven P -> T -> L -> C -> R custody chain.  It CREATE_NEW claims C before
    reading P/T/L, duplicates that same handle into the exact live supervisor,
    waits for CUSTODY_GRANT, reads the supervisor-retained inputs through duplicated
    handles, serializes C through the original creating handle, waits for the
    supervisor-retained R and FINAL_ACK, and emits one final canonical summary.

There is no generic callable, callback, plugin, fourth mode, automatic retry,
scientific training path, or outcome-value interface here.  Failures are terminal
non-zero returns.  A partial CREATE_NEW claim is deliberately never removed.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import stat
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

from . import original_confirmatory_capsule_authority as authority

TERMINAL_HANDLER_STOP_EXIT_CODE: Final = 70
TERMINAL_HANDLER_PROTOCOL_EXIT_CODE: Final = 74

PRETERMINAL_PIN_POLICY: Final = "original_confirmatory_capsule_preterminal_pin_v1"
PRETERMINAL_HANDSHAKE_POLICY: Final = "original_confirmatory_preterminal_pin_overlap_handshake_v1"
PRETERMINAL_READY_POLICY: Final = authority.PRETERMINAL_PIN_STDOUT_SUMMARY_POLICY
PRETERMINAL_READY_MESSAGE_TYPE: Final = authority.PRETERMINAL_READY_MESSAGE_TYPE
PRETERMINAL_ACK_POLICY: Final = authority.PRETERMINAL_OVERLAP_ACK_POLICY
PRETERMINAL_ACK_MESSAGE_TYPE: Final = authority.PRETERMINAL_ACK_MESSAGE_TYPE
PRETERMINAL_SUPERVISOR_OBSERVER_IDENTITY_POLICY: Final = (
    "original_confirmatory_preterminal_pin_supervisor_observer_identity_v1"
)

POSTWAKE_INPUT_LEASE_RECEIPT_POLICY: Final = authority.POSTWAKE_INPUT_LEASE_RECEIPT_POLICY
COMPOSED_CLAIM_READY_POLICY: Final = "original_confirmatory_composed_terminal_claim_ready_v1"
COMPOSED_CLAIM_PHYSICAL_IDENTITY_POLICY: Final = (
    "original_confirmatory_composed_terminal_claim_physical_identity_v1"
)
COMPOSED_CUSTODY_GRANT_POLICY: Final = "original_confirmatory_composed_terminal_custody_grant_v1"
COMPOSED_READY_POLICY: Final = "original_confirmatory_composed_terminal_ready_v1"
COMPOSED_FINAL_ACK_POLICY: Final = "original_confirmatory_composed_terminal_final_ack_v1"
COMPOSED_TERMINAL_RECEIPT_POLICY: Final = authority.COMPOSED_TERMINAL_RECEIPT_POLICY
POSTWAKE_COMPOSED_READBACK_RECEIPT_POLICY: Final = (
    authority.POSTWAKE_COMPOSED_READBACK_RECEIPT_POLICY
)
COMPOSED_TERMINAL_STDOUT_SUMMARY_POLICY: Final = authority.COMPOSED_TERMINAL_STDOUT_SUMMARY_POLICY
TERMINAL_CUSTODY_AUTHORITY_PROJECTION_POLICY: Final = (
    "original_confirmatory_terminal_custody_authority_projection_v1"
)
OUTCOME_BLIND_EXPECTED_ARTIFACT_PROJECTION_POLICY: Final = (
    "original_confirmatory_outcome_blind_expected_artifact_projection_v1"
)

CLAIM_READY_MESSAGE_TYPE: Final = "CLAIM_READY"
CUSTODY_GRANT_MESSAGE_TYPE: Final = "CUSTODY_GRANT"
COMPOSED_READY_MESSAGE_TYPE: Final = "COMPOSED_READY"
FINAL_ACK_MESSAGE_TYPE: Final = "FINAL_ACK"

SEMANTIC_OUTCOME_READ_SCOPE: Final = authority.SEMANTIC_OUTCOME_READ_SCOPE
GENERIC_READ: Final = 0x80000000
GENERIC_WRITE: Final = 0x40000000
FILE_GENERIC_READ_ACCESS_MASK: Final = 0x00120089
FILE_SHARE_READ: Final = 0x00000001
FILE_SHARE_WRITE: Final = 0x00000002
PROCESS_DUP_HANDLE: Final = 0x0040
PROCESS_QUERY_LIMITED_INFORMATION: Final = 0x1000
PROCESS_VM_READ: Final = 0x0010
SYNCHRONIZE: Final = 0x00100000
DUPLICATE_SAME_ACCESS: Final = 0x00000002
CREATE_NEW: Final = 1
OPEN_EXISTING: Final = 3
FILE_ATTRIBUTE_READONLY: Final = 0x00000001
FILE_ATTRIBUTE_NORMAL: Final = 0x00000080
FILE_FLAG_OPEN_REPARSE_POINT: Final = 0x00200000
FILE_FLAG_OVERLAPPED: Final = 0x40000000
INVALID_HANDLE_VALUE: Final = ctypes.c_void_p(-1).value
ERROR_MORE_DATA: Final = 234
ERROR_OPERATION_ABORTED: Final = 995
ERROR_IO_PENDING: Final = 997
ERROR_NOT_FOUND: Final = 1168
WAIT_OBJECT_0: Final = 0
WAIT_TIMEOUT: Final = 258

_SHA256 = authority._SHA256
_MAX_CONTROL_BYTES: Final = authority.MAX_CONTROL_FILE_BYTES
_MAX_PIPE_MESSAGE_BYTES: Final = 1024 * 1024

_EXPECTED_ARTIFACT_ROLE_ORDER: Final = (
    "terminal_seal",
    "integrity_receipt",
    "completion_evidence",
    "integrity_registry",
    "stage_attestation_registry",
    "stage_attestation_anchor",
    "disposition_anchor",
)
_EXPECTED_ARTIFACT_RULE_FIELDS: Final = {
    "role",
    "path",
    "expected_sha256",
    "must_be_absent_before",
    "json_equals",
}
_EXPECTED_ARTIFACT_TEMPLATE: Final = (
    {
        "role": "terminal_seal",
        "path_anchor": "expected_run_directory",
        "relative_path": ".immutable.json",
        "must_be_absent_before": True,
        "json_equals": {"run_id": "$RUN_ID", "status": "completed"},
    },
    {
        "role": "integrity_receipt",
        "path_anchor": "expected_run_directory",
        "relative_path": "artifact_manifest.json",
        "must_be_absent_before": True,
        "json_equals": {"run_id": "$RUN_ID", "status": "completed"},
    },
    {
        "role": "completion_evidence",
        "path_anchor": "expected_run_directory",
        "relative_path": "completion_evidence.json",
        "must_be_absent_before": True,
        "json_equals": {
            "run_id": "$RUN_ID",
            "completion_stage": "CONFIRMATORY_COMPLETE",
            "study_outcome_eligible": True,
        },
    },
    {
        "role": "integrity_registry",
        "path_anchor": "runs_root",
        "relative_path": "integrity_registry.jsonl",
        "must_be_absent_before": False,
        "json_equals": {},
    },
    {
        "role": "stage_attestation_registry",
        "path_anchor": "runs_root",
        "relative_path": "run_stage_attestations.jsonl",
        "must_be_absent_before": False,
        "json_equals": {},
    },
    {
        "role": "stage_attestation_anchor",
        "path_anchor": "runs_root",
        "relative_path": "run_stage_attestations.anchor.json",
        "must_be_absent_before": False,
        "json_equals": {},
    },
    {
        "role": "disposition_anchor",
        "path_anchor": "runs_root",
        "relative_path": "run_dispositions.anchor.json",
        "must_be_absent_before": False,
        "json_equals": {},
    },
)
_SUPERVISOR_ATTEMPT_POLICY: Final = "exactly_one_launch_attempt_no_automatic_retry_v1"
_SUPERVISOR_PROCESS_KIND: Final = "confirmatory"
_SUPERVISOR_SUCCESS_REASON: Final = "exit_zero_science_seal_preterminal_pin_and_integrity_verified"
_CANONICAL_SUPERVISOR_SPEC_FIELDS: Final = {
    "schema_version",
    "policy",
    "job_id",
    "process_kind",
    "external_control_plane_release_root_sha256",
    "external_control_plane_publication_id",
    "external_control_plane_release_qualification_attestation_path",
    "external_control_plane_release_qualification_attestation_file_sha256",
    "external_control_plane_release_qualification_attestation_root_sha256",
    "supervisor_code_root",
    "supervisor_state_root",
    "project_root",
    "program_path",
    "program_sha256",
    "argv",
    "expected_artifacts",
    "required_success_roles",
    "integrity_verifier",
    "authorization",
    "max_log_bytes",
    "main_timeout_ms",
    "verifier_timeout_ms",
    "codex_wake_timeout_seconds",
    "codex",
    "external_codex_handoff",
    "expected_environment",
    "process_environment_binding",
    "preterminal_pin_contract",
    "preterminal_overlap_handshake_contract",
    "postwake_custody_seed",
    "postwake_custody_handshake_contract",
    "postwake_input_lease_contract",
    "terminal_composition_contract",
    "e_consumption_contract",
    "q_e_custody_contract",
    "q_e_custody_handoff",
    "q_e_custody_receipt",
    "python_lease_identity",
    "python_lease_identity_root_sha256",
    "python_ancestor_lease",
    "python_ancestor_lease_root_sha256",
    "python_runtime_resolution_policy",
    "runtime_python_path",
    "runtime_python_sha256",
    "runtime_python_lease_identity",
    "runtime_python_lease_identity_root_sha256",
    "runtime_python_ancestor_lease",
    "runtime_python_ancestor_lease_root_sha256",
    "supervisor_launcher_sha256",
    "capsule_lease_identity",
    "capsule_lease_identity_root_sha256",
    "capsule_ancestor_lease",
    "capsule_ancestor_lease_root_sha256",
    "command",
    "handoff_session",
    "automatic_retry_allowed",
    "max_attempt_count",
}
_RUN_SPEC_PAYLOAD_FIELDS: Final = {
    "schema_version",
    "policy",
    "source_path",
    "source_file_sha256",
    "canonical_spec",
    "canonical_spec_sha256",
    "supervisor",
    "frozen_at_utc",
    "control_staging",
}
_CONTROL_STAGING_OUTER_BINDING_FIELDS: Final = {
    "schema_version",
    "policy",
    "job_id",
    "supervisor_root",
    "control_staging_root",
    "control_staging_dir",
    "control_staging_projection",
    "control_staging_projection_sha256",
    "expected_complete_leaf_names",
    "publication_order",
    "file_count",
    "files",
    "control_staging_ancestor_lease",
    "control_staging_ancestor_lease_root_sha256",
    "staging_attempt_root_sha256",
    "staging_ready_root_sha256",
    "source_path",
    "source_size_bytes",
    "source_file_sha256",
    "source_canonical_bytes_sha256",
    "source_bytes_equal_canonical_spec_serialization",
    "e_intent_path",
    "e_intent_file_sha256",
    "launch_authorization_path",
    "launch_authorization_file_sha256",
    "supervisor_process_identity",
    "retained_from_before_final_job_creation_through_terminal",
    "final_job_creation_owner",
    "pre_ack_final_job_publication_scope",
    "pre_ack_metadata_only_publication_allowed",
    "pre_ack_scientific_process_launch_allowed",
    "q_e_ack_required_before_scientific_process_launch",
    "automatic_retry_allowed",
    "adoption_allowed",
    "cleanup_allowed",
    "binding_root_sha256",
}
_CONTROL_STAGING_OUTER_BINDING_POLICY: Final = "aanca_supervisor_control_staging_binding_v1"
_EXTERNAL_CODEX_HANDOFF_FIELDS: Final = {
    "policy",
    "staged_e_intent_path",
    "staged_e_intent_file_sha256",
    "staged_e_intent_core_root_sha256",
    "attempt_creation_authority_payload_sha256",
    "attempt_authority_output_path",
    "terminal_handoff_receipt_output_path",
    "internal_codex_wake_allowed",
    "legacy_handoff_session_allowed",
    "single_wake_owner",
}
_LAUNCH_INTENT_FIELDS: Final = {
    "schema_version",
    "policy",
    "attempt_policy",
    "job_id",
    "spec_sha256",
    "command",
    "command_sha256",
    "attempt_nonce",
    "attempt_count",
    "max_attempt_count",
    "automatic_retry_allowed",
    "job_assignment_mode",
    "handle_list_restricted",
    "job_handle_inherited",
    "supervisor_process_identity",
    "main_timeout_ms",
    "windows_boot_time_utc",
    "prearm_process_absence",
    "prelaunch_artifacts",
    "created_at_utc",
    "supervisor_launcher_sha256",
    "expected_environment_envelope_sha256",
    "launch_environment_root_sha256",
    "process_environment_binding_sha256",
    "exact_supervisor_environment_sha256",
    "exact_environment_sha256",
    "exact_integrity_verifier_environment_sha256",
    "observed_supervisor_environment_sha256",
    "supervisor_environment_exact_match",
    "capsule_lease_identity",
    "capsule_lease_identity_root_sha256",
    "capsule_ancestor_lease",
    "capsule_ancestor_lease_root_sha256",
    "preterminal_overlap_handshake_contract_sha256",
    "python_lease_identity",
    "python_lease_identity_root_sha256",
    "python_ancestor_lease",
    "python_ancestor_lease_root_sha256",
    "python_runtime_resolution_policy",
    "runtime_python_lease_identity",
    "runtime_python_lease_identity_root_sha256",
    "runtime_python_ancestor_lease",
    "runtime_python_ancestor_lease_root_sha256",
    "e_consumption_contract_sha256",
}
_PROCESS_STARTED_FIELDS: Final = {
    "schema_version",
    "policy",
    "job_id",
    "spec_sha256",
    "launch_intent_sha256",
    "attempt_nonce",
    "process_identity",
    "job_assignment_mode",
    "atomic_job_assignment",
    "handle_list_restricted",
    "job_handle_inherited",
    "supervisor_process_identity",
    "main_timeout_ms",
    "stdout_partial_path",
    "stderr_partial_path",
    "windows_boot_time_utc",
    "started_at_utc",
    "attempt_count",
    "automatic_retry_allowed",
    "expected_environment_envelope_sha256",
    "launch_environment_root_sha256",
    "process_environment_binding_sha256",
    "exact_supervisor_environment_sha256",
    "observed_supervisor_environment_sha256",
    "exact_environment_sha256",
    "observed_child_environment_sha256",
    "exact_integrity_verifier_environment_sha256",
    "child_environment_exact_match",
    "child_environment_observation_method",
    "capsule_lease_identity_root_sha256",
    "capsule_ancestor_lease_root_sha256",
    "preterminal_overlap_handshake_contract_sha256",
    "python_lease_identity_root_sha256",
    "python_ancestor_lease_root_sha256",
    "interpreter_leaf_handle_active",
    "interpreter_ancestor_handles_active",
    "python_runtime_resolution_policy",
    "runtime_python_lease_identity_root_sha256",
    "runtime_python_ancestor_lease_root_sha256",
    "runtime_interpreter_leaf_handle_active",
    "runtime_interpreter_ancestor_handles_active",
    "e_consumption_contract_sha256",
}
_PREARM_PROCESS_ABSENCE_FIELDS: Final = {
    "schema_version",
    "policy",
    "observed_at_utc",
    "inventory_process_count",
    "target_program_path",
    "target_program_sha256",
    "target_command_sha256",
    "target_argv_sha256",
    "exact_command_matches",
    "protected_marker_matches",
    "absence_verified",
}

_PRETERMINAL_PIN_FIELDS: Final = {
    "schema_version",
    "policy",
    "status",
    "job_id",
    "attempt_id",
    "run_id",
    "execution_mode",
    "retry_of_run_id",
    "attempt_nonce",
    "q_authority_root_sha256",
    "e_intent_path",
    "e_intent_file_sha256",
    "e_intent_core_sha256",
    "e_consumption_claim_path",
    "e_consumption_claim_file_sha256",
    "e_consumption_claim_root_sha256",
    "capsule_contract_sha256",
    "capsule_sha256",
    "capsule_internal_manifest_sha256",
    "capsule_mode",
    "preterminal_pin_contract_sha256",
    "preterminal_overlap_handshake_contract_sha256",
    "run_spec_file_sha256",
    "canonical_spec_sha256",
    "launch_intent_file_sha256",
    "process_started_file_sha256",
    "scientific_command_sha256",
    "preterminal_command_sha256",
    "terminal_command_sha256",
    "observed_integrity_verifier_environment_sha256",
    "required_success_roles",
    "expected_artifact_evidence",
    "expected_artifact_evidence_root_sha256",
    "preterminal_scientific_core_sha256",
    "semantic_outcome_read_scope",
    "outcome_values_read",
    "outcome_values_emitted",
    "outcome_values_used_for_selection_or_tuning",
    "training_or_model_selection_allowed",
    "scientific_publication_allowed",
    "automatic_retry_allowed",
    "created_at_utc",
    "evidence_root_sha256",
}

_PRETERMINAL_READY_FIELDS: Final = {
    "schema_version",
    "policy",
    "message_type",
    "handshake_policy",
    "job_id",
    "attempt_id",
    "run_id",
    "execution_mode",
    "retry_of_run_id",
    "attempt_nonce",
    "capsule_contract_sha256",
    "capsule_sha256",
    "capsule_internal_manifest_sha256",
    "capsule_mode",
    "preterminal_pin_contract_sha256",
    "observed_integrity_verifier_environment_sha256",
    "preterminal_pin_receipt",
    "preterminal_pin_evidence_root_sha256",
    "preterminal_scientific_core_sha256",
    "semantic_outcome_read_scope",
    "outcome_values_read",
    "outcome_values_emitted",
    "outcome_values_used_for_selection_or_tuning",
    "training_or_model_selection_allowed",
    "scientific_publication_allowed",
    "automatic_retry_allowed",
    "pin_handle_open",
    "pin_handle_share_access",
    "awaiting_supervisor_ack",
}

_PRETERMINAL_ACK_FIELDS: Final = {
    "schema_version",
    "policy",
    "message_type",
    "job_id",
    "attempt_id",
    "run_id",
    "execution_mode",
    "retry_of_run_id",
    "attempt_nonce",
    "preterminal_pin_contract_sha256",
    "preterminal_overlap_handshake_contract_sha256",
    "ready_line_sha256",
    "child_reported_pin_identity",
    "child_reported_pin_identity_root_sha256",
    "supervisor_opened_pin_identity",
    "supervisor_opened_pin_identity_root_sha256",
    "stable_physical_identity_projection",
    "stable_physical_identity_root_sha256",
    "identity_exact_match",
    "pin_handle_overlap_verified",
    "automatic_retry_allowed",
    "acknowledged_at_utc",
}

_PROCESS_IDENTITY_FIELDS: Final = {
    "pid",
    "creation_time_100ns",
    "creation_time_utc",
    "program_path",
    "program_sha256",
    "command_sha256",
}

_RETAINED_BINDING_ROLES: Final = (
    "preterminal-pin",
    "preterminal-stdout",
    "preterminal-stderr",
    "supervisor-terminal",
)

_POSTWAKE_LEASE_FIELDS: Final = {
    "schema_version",
    "policy",
    "status",
    "job_id",
    "attempt_id",
    "run_id",
    "execution_mode",
    "retry_of_run_id",
    "attempt_nonce",
    "q_authority_root_sha256",
    "e_intent_file_sha256",
    "e_intent_core_sha256",
    "supervisor_spec_sha256",
    "postwake_custody_seed_sha256",
    "postwake_input_lease_contract_sha256",
    "preterminal_pin_contract_sha256",
    "preterminal_overlap_handshake_contract_sha256",
    "terminal_composition_contract_sha256",
    "supervisor_process_identity",
    "windows_boot_time_utc",
    "retained_input_bindings",
    "retained_input_bindings_root_sha256",
    "self_preserialization_handle_binding",
    "automatic_retry_allowed",
    "created_at_utc",
    "receipt_root_sha256",
}

_SUPERVISOR_TERMINAL_FIELDS: Final = {
    "schema_version",
    "policy",
    "attempt_policy",
    "job_id",
    "process_kind",
    "spec_sha256",
    "terminal_kind",
    "reason",
    "attempt_count",
    "automatic_retry_allowed",
    "exit_code",
    "launch_intent_receipt_sha256",
    "process_started_receipt_sha256",
    "stdout",
    "stderr",
    "expected_artifacts",
    "integrity_verifier",
    "descendants_after_root_exit",
    "recovery_evidence",
    "ended_at_utc",
    "preterminal_pin_contract_sha256",
    "preterminal_overlap_handshake_contract_sha256",
    "environment_binding",
    "capsule_lease_identity",
    "capsule_lease_identity_root_sha256",
    "capsule_ancestor_lease",
    "capsule_ancestor_lease_root_sha256",
    "python_lease_identity",
    "python_lease_identity_root_sha256",
    "python_ancestor_lease",
    "python_ancestor_lease_root_sha256",
    "python_runtime_resolution_policy",
    "runtime_python_lease_identity",
    "runtime_python_lease_identity_root_sha256",
    "runtime_python_ancestor_lease",
    "runtime_python_ancestor_lease_root_sha256",
    "e_consumption_contract_sha256",
    "e_consumption_custody_receipt_file_sha256",
    "e_consumption_custody_receipt_root_sha256",
    "e_consumption_ready_sha256",
    "e_consumption_ack_sha256",
}
_PRETERMINAL_ARTIFACT_EVIDENCE_FIELDS: Final = {
    "role",
    "path",
    "size_bytes",
    "sha256",
    "expected_sha256",
    "json_control_paths_checked",
    "valid",
}
_SUPERVISOR_TERMINAL_ARTIFACT_FIELDS: Final = {
    "role",
    "path",
    "valid",
    "errors",
    "size_bytes",
    "sha256",
}
_SUPERVISOR_LOG_RECORD_FIELDS: Final = {
    "path",
    "exists",
    "size_bytes",
    "sha256",
    "limit_bytes",
    "limit_exceeded",
    "capture_complete",
    "stream_size_bytes",
    "stream_sha256",
    "stored_sha256",
    "discarded_bytes",
}
_SUPERVISOR_TERMINAL_ENVIRONMENT_FIELDS: Final = {
    "expected_environment_envelope_sha256",
    "launch_environment_root_sha256",
    "process_environment_binding_sha256",
    "exact_supervisor_environment_sha256",
    "observed_supervisor_environment_sha256",
    "exact_environment_sha256",
    "observed_child_environment_sha256",
    "exact_integrity_verifier_environment_sha256",
    "observed_integrity_verifier_environment_sha256",
    "all_exact_matches",
}
_SUPERVISOR_VERIFIER_RECORD_FIELDS: Final = {
    "command",
    "process_identity",
    "started_at_utc",
    "ended_at_utc",
    "timeout_ms",
    "job_assignment_mode",
    "atomic_job_assignment",
    "handle_list_restricted",
    "job_handle_inherited",
    "exit_code",
    "descendants_after_root_exit",
    "stdout",
    "stderr",
    "error_type",
    "error_sha256",
    "cleanup_error_type",
    "cleanup_error_sha256",
    "tree_empty_verified",
    "valid",
    "capsule_contract_sha256",
    "capsule_sha256",
    "capsule_internal_manifest_sha256",
    "capsule_mode",
    "expected_environment_envelope_sha256",
    "process_environment_binding_sha256",
    "exact_integrity_verifier_environment_sha256",
    "observed_integrity_verifier_environment_sha256",
    "integrity_verifier_environment_exact_match",
    "integrity_verifier_environment_observation_method",
    "preterminal_pin_contract_sha256",
    "preterminal_overlap_handshake_contract_sha256",
    "preterminal_overlap_handshake_receipt",
    "capsule_lease_identity_root_sha256",
    "capsule_ancestor_lease_root_sha256",
    "python_lease_identity_root_sha256",
    "python_ancestor_lease_root_sha256",
    "interpreter_leaf_handle_active",
    "interpreter_ancestor_handles_active",
    "python_runtime_resolution_policy",
    "runtime_python_lease_identity_root_sha256",
    "runtime_python_ancestor_lease_root_sha256",
    "runtime_interpreter_leaf_handle_active",
    "runtime_interpreter_ancestor_handles_active",
}
_PRETERMINAL_HANDSHAKE_RECEIPT_FIELDS: Final = {
    "schema_version",
    "policy",
    "status",
    "job_id",
    "attempt_id",
    "run_id",
    "execution_mode",
    "retry_of_run_id",
    "attempt_nonce",
    "verifier_process_identity",
    "preterminal_pin_contract_sha256",
    "preterminal_overlap_handshake_contract_sha256",
    "ready_summary",
    "ready_line_sha256",
    "ack",
    "ack_line_sha256",
    "child_reported_pin_identity",
    "supervisor_opened_pin_identity",
    "preterminal_pin_identity_root_sha256",
    "stdout_log_identity",
    "stderr_log_identity",
    "stdout_eof_after_ack",
    "stdout_additional_bytes",
    "stderr_empty",
    "exit_code",
    "pin_handle_overlap_verified",
    "supervisor_pin_handle_active",
    "restricted_inherited_handle_list_verified",
    "automatic_retry_allowed",
    "created_at_utc",
    "evidence_root_sha256",
}
_PRETERMINAL_HANDSHAKE_RECEIPT_POLICY: Final = (
    "original_confirmatory_preterminal_pin_overlap_handshake_receipt_v1"
)

_CLAIM_PHYSICAL_IDENTITY_FIELDS: Final = {
    "schema_version",
    "policy",
    "role",
    "path",
    "volume_serial_number",
    "file_id_128",
    "device",
    "inode",
    "size_bytes",
    "mode",
    "file_attributes",
    "regular_file",
    "read_only",
    "link_count",
    "modified_time_ns",
    "changed_time_ns",
    "sha256",
    "named_alternate_data_streams",
    "opened_without_reparse_follow",
    "share_access",
}

_STABLE_PHYSICAL_IDENTITY_FIELD_ORDER: Final = (
    "schema_version",
    "role",
    "path",
    "volume_serial_number",
    "file_id_128",
    "device",
    "inode",
    "size_bytes",
    "mode",
    "file_attributes",
    "regular_file",
    "read_only",
    "link_count",
    "modified_time_ns",
    "changed_time_ns",
    "sha256",
    "named_alternate_data_streams",
    "opened_without_reparse_follow",
)
_STABLE_PHYSICAL_IDENTITY_FIELDS: Final = set(_STABLE_PHYSICAL_IDENTITY_FIELD_ORDER)

_CLAIM_READY_FIELDS: Final = {
    "schema_version",
    "policy",
    "message_type",
    "job_id",
    "attempt_id",
    "run_id",
    "execution_mode",
    "retry_of_run_id",
    "attempt_nonce",
    "q_authority_root_sha256",
    "e_intent_file_sha256",
    "e_intent_core_sha256",
    "e_consumption_claim_path",
    "e_consumption_claim_file_sha256",
    "e_consumption_claim_root_sha256",
    "supervisor_spec_sha256",
    "postwake_custody_seed_sha256",
    "postwake_custody_handshake_contract_sha256",
    "terminal_composition_contract_sha256",
    "terminal_command_sha256",
    "observed_integrity_verifier_environment_sha256",
    "client_process_identity",
    "immediate_venv_redirector_pid",
    "immediate_venv_redirector_process_identity",
    "terminal_client_launcher_process_identity",
    "supervisor_process_identity",
    "terminal_client_launch_intent_path",
    "terminal_client_launch_intent_policy",
    "terminal_client_launch_intent_read",
    "composed_terminal_path",
    "claim_physical_identity",
    "claim_physical_identity_root_sha256",
    "target_supervisor_handle_value",
    "target_access_mask",
    "duplicate_options",
    "close_source",
    "claim_before_terminal_input_read",
    "terminal_inputs_read",
    "automatic_retry_allowed",
    "claim_ready_sha256",
}

_CUSTODY_GRANT_FIELDS: Final = {
    "schema_version",
    "policy",
    "message_type",
    "job_id",
    "attempt_id",
    "run_id",
    "execution_mode",
    "retry_of_run_id",
    "attempt_nonce",
    "postwake_custody_seed_sha256",
    "postwake_custody_handshake_contract_sha256",
    "terminal_composition_contract_sha256",
    "claim_ready_sha256",
    "client_process_identity",
    "immediate_venv_redirector_pid",
    "immediate_venv_redirector_process_identity",
    "terminal_client_launcher_process_identity",
    "supervisor_process_identity",
    "terminal_client_launch_intent_path",
    "terminal_client_launch_intent_policy",
    "terminal_client_launch_intent_file_sha256",
    "terminal_client_launch_intent_root_sha256",
    "terminal_client_launch_intent_physical_identity",
    "terminal_client_launch_intent_physical_identity_root_sha256",
    "terminal_client_launch_intent_verified",
    "terminal_client_launch_intent_launcher_identity_verified",
    "terminal_client_launch_intent_create_new_before_child_verified",
    "terminal_client_launch_intent_supervisor_handle_slot",
    "terminal_client_launch_intent_supervisor_granted_access_mask",
    "terminal_client_launch_intent_child_duplicate_target_access_mask",
    "terminal_client_launch_intent_child_expected_granted_access_mask",
    "terminal_client_launch_intent_child_duplicate_options",
    "terminal_client_launch_intent_child_duplicate_close_source",
    "terminal_client_launch_intent_supervisor_custody_active",
    "terminal_client_launch_intent_child_open_after_grant_required",
    "target_supervisor_handle_value",
    "target_granted_access_mask",
    "claim_physical_identity",
    "claim_physical_identity_root_sha256",
    "postwake_input_lease_receipt_path",
    "postwake_input_lease_handle_slot",
    "postwake_input_lease_physical_identity",
    "postwake_input_lease_physical_identity_root_sha256",
    "same_supervisor_job_verified",
    "exact_wake_tree_descendant_verified",
    "client_process_identity_verified",
    "immediate_venv_redirector_process_identity_verified",
    "terminal_client_launcher_process_identity_verified",
    "launcher_redirector_child_grandparent_chain_verified",
    "launcher_redirector_child_same_supervisor_job_verified",
    "client_command_line_peb_readback_verified",
    "client_cwd_peb_readback_verified",
    "client_environment_peb_readback_verified",
    "supervisor_custody_active",
    "automatic_retry_allowed",
    "granted_at_utc",
    "custody_grant_sha256",
}

_COMPOSED_READY_FIELDS: Final = {
    "schema_version",
    "policy",
    "message_type",
    "job_id",
    "attempt_id",
    "run_id",
    "execution_mode",
    "retry_of_run_id",
    "attempt_nonce",
    "postwake_custody_seed_sha256",
    "postwake_custody_handshake_contract_sha256",
    "terminal_composition_contract_sha256",
    "claim_ready_sha256",
    "custody_grant_sha256",
    "target_supervisor_handle_value",
    "composed_terminal_path",
    "composed_terminal_file_sha256",
    "composed_terminal_receipt_root_sha256",
    "composed_terminal_physical_identity",
    "composed_terminal_physical_identity_root_sha256",
    "source_inputs_root_sha256",
    "terminal_client_launch_intent_path",
    "terminal_client_launch_intent_policy",
    "terminal_client_launch_intent_file_sha256",
    "terminal_client_launch_intent_root_sha256",
    "terminal_client_launch_intent_physical_identity",
    "terminal_client_launch_intent_physical_identity_root_sha256",
    "terminal_client_launch_intent_supervisor_handle_slot",
    "terminal_client_launch_intent_child_handle_slot",
    "terminal_client_launch_intent_child_granted_access_mask",
    "terminal_client_launch_intent_same_duplicated_supervisor_handle_used",
    "terminal_client_launch_intent_physical_identity_exact_match",
    "terminal_client_launch_intent_child_custody_active",
    "terminal_client_launch_intent_supervisor_custody_active",
    "same_held_create_new_handle_used",
    "supervisor_custody_active",
    "automatic_retry_allowed",
    "composed_ready_sha256",
}

_FINAL_ACK_FIELDS: Final = {
    "schema_version",
    "policy",
    "message_type",
    "job_id",
    "attempt_id",
    "run_id",
    "execution_mode",
    "retry_of_run_id",
    "attempt_nonce",
    "postwake_custody_seed_sha256",
    "postwake_custody_handshake_contract_sha256",
    "terminal_composition_contract_sha256",
    "claim_ready_sha256",
    "custody_grant_sha256",
    "composed_ready_sha256",
    "target_supervisor_handle_value",
    "composed_terminal_file_sha256",
    "composed_terminal_receipt_root_sha256",
    "supervisor_rehashed_composed_identity",
    "supervisor_rehashed_composed_identity_root_sha256",
    "identity_exact_match",
    "postwake_composed_readback_receipt_path",
    "postwake_composed_readback_receipt_file_sha256",
    "postwake_composed_readback_receipt_root_sha256",
    "terminal_client_launch_intent_path",
    "terminal_client_launch_intent_policy",
    "terminal_client_launch_intent_file_sha256",
    "terminal_client_launch_intent_root_sha256",
    "terminal_client_launch_intent_physical_identity",
    "terminal_client_launch_intent_physical_identity_root_sha256",
    "terminal_client_launch_intent_supervisor_handle_slot",
    "terminal_client_launch_intent_supervisor_custody_retained_through_ack",
    "launcher_redirector_child_process_handles_retained_through_ack",
    "immediate_venv_redirector_process_identity_reverified",
    "terminal_client_launcher_process_identity_reverified",
    "launcher_redirector_child_grandparent_chain_reverified",
    "launcher_redirector_child_same_supervisor_job_reverified",
    "immediate_venv_redirector_live_at_final_ack",
    "terminal_client_launcher_live_at_final_ack",
    "supervisor_custody_retained_through_ack",
    "automatic_retry_allowed",
    "acknowledged_at_utc",
    "final_ack_sha256",
}

_POSTWAKE_COMPOSED_READBACK_FIELDS: Final = {
    "schema_version",
    "policy",
    "status",
    "job_id",
    "attempt_id",
    "run_id",
    "execution_mode",
    "retry_of_run_id",
    "attempt_nonce",
    "postwake_custody_seed_sha256",
    "postwake_custody_handshake_contract_sha256",
    "terminal_composition_contract_sha256",
    "claim_ready_sha256",
    "custody_grant_sha256",
    "composed_ready_sha256",
    "target_supervisor_handle_value",
    "composed_terminal_path",
    "composed_terminal_file_sha256",
    "composed_terminal_receipt_root_sha256",
    "supervisor_rehashed_composed_identity",
    "supervisor_rehashed_composed_identity_root_sha256",
    "terminal_client_launch_intent_path",
    "terminal_client_launch_intent_policy",
    "terminal_client_launch_intent_file_sha256",
    "terminal_client_launch_intent_root_sha256",
    "terminal_client_launch_intent_physical_identity",
    "terminal_client_launch_intent_physical_identity_root_sha256",
    "terminal_client_launch_intent_supervisor_handle_slot",
    "terminal_client_launch_intent_supervisor_custody_retained_through_readback",
    "identity_exact_match",
    "same_duplicated_supervisor_handle_used",
    "readback_created_with_create_new",
    "readback_same_held_handle_used",
    "supervisor_custody_retained_through_readback",
    "automatic_retry_allowed",
    "created_at_utc",
    "receipt_root_sha256",
}

_COMPOSED_RECEIPT_FIELDS: Final = {
    "schema_version",
    "policy",
    "status",
    "job_id",
    "attempt_id",
    "run_id",
    "execution_mode",
    "retry_of_run_id",
    "attempt_nonce",
    "q_authority_root_sha256",
    "e_intent_path",
    "e_intent_file_sha256",
    "e_intent_core_sha256",
    "e_consumption_claim_path",
    "e_consumption_claim_file_sha256",
    "e_consumption_claim_root_sha256",
    "supervisor_spec_sha256",
    "canonical_spec_sha256",
    "capsule_contract_sha256",
    "capsule_sha256",
    "capsule_internal_manifest_sha256",
    "terminal_command_sha256",
    "preterminal_pin_contract_sha256",
    "preterminal_overlap_handshake_contract_sha256",
    "postwake_input_lease_contract_sha256",
    "postwake_custody_seed_sha256",
    "postwake_custody_handshake_contract_sha256",
    "terminal_composition_contract_sha256",
    "expected_integrity_verifier_environment_sha256",
    "observed_integrity_verifier_environment_sha256",
    "preterminal_pin_file_sha256",
    "preterminal_pin_evidence_root_sha256",
    "preterminal_scientific_core_sha256",
    "supervisor_terminal_file_sha256",
    "supervisor_terminal_payload_sha256",
    "supervisor_terminal_kind",
    "supervisor_terminal_reason",
    "supervisor_terminal_exit_code",
    "postwake_input_lease_file_sha256",
    "postwake_input_lease_receipt_root_sha256",
    "source_input_bindings",
    "source_input_bindings_root_sha256",
    "source_inputs_root_sha256",
    "terminal_client_launch_intent_path",
    "terminal_client_launch_intent_policy",
    "terminal_client_launch_intent_file_sha256",
    "terminal_client_launch_intent_root_sha256",
    "terminal_client_launch_intent_physical_identity",
    "terminal_client_launch_intent_physical_identity_root_sha256",
    "terminal_client_launch_intent_supervisor_handle_slot",
    "terminal_client_launch_intent_child_handle_slot",
    "terminal_client_launch_intent_child_granted_access_mask",
    "terminal_client_launch_intent_same_duplicated_supervisor_handle_used",
    "terminal_client_launch_intent_physical_identity_exact_match",
    "terminal_client_launch_intent_child_custody_active",
    "terminal_client_launch_intent_supervisor_custody_active",
    "claim_ready_sha256",
    "custody_grant_sha256",
    "same_held_create_new_handle_used",
    "supervisor_custody_active",
    "semantic_outcome_read_scope",
    "outcome_values_read",
    "outcome_values_emitted",
    "outcome_values_used_for_selection_or_tuning",
    "training_or_model_selection_allowed",
    "scientific_publication_allowed",
    "automatic_retry_allowed",
    "created_at_utc",
    "receipt_root_sha256",
}


class OriginalConfirmatoryTerminalError(ValueError):
    """A fixed terminal handler failed closed."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _is_utc(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    offset = parsed.utcoffset()
    return offset is not None and offset.total_seconds() == 0


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return authority.canonical_json_line_bytes(value)


def _canonical_sha256(value: Any) -> str:
    return authority.canonical_json_sha256(value)


def _outcome_blind_expected_artifact_projection() -> dict[str, Any]:
    unsigned = {
        "schema_version": 1,
        "policy": OUTCOME_BLIND_EXPECTED_ARTIFACT_PROJECTION_POLICY,
        "ordered_role_templates": [
            {
                **template,
                "json_equals": dict(cast(Mapping[str, Any], template["json_equals"])),
            }
            for template in _EXPECTED_ARTIFACT_TEMPLATE
        ],
        "rule_field_names": sorted(_EXPECTED_ARTIFACT_RULE_FIELDS),
        "allowed_flat_json_selectors": [
            "completion_stage",
            "run_id",
            "status",
            "study_outcome_eligible",
        ],
        "strict_expected_type_equality_required": True,
        "dotted_paths_allowed": False,
        "numeric_or_list_indirection_allowed": False,
        "empty_json_equals_requires_zero_json_decode": True,
        "scientific_metric_or_ranking_selectors_allowed": False,
        "pre_arm_validation_required": True,
    }
    return {
        **unsigned,
        "projection_root_sha256": _canonical_sha256(unsigned),
    }


def _terminal_duplex_authority_projection() -> dict[str, Any]:
    """Machine-readable terminal protocol schema for canonical authority binding."""

    return authority.build_original_confirmatory_terminal_custody_authority_template_projection()


def _canonical_terminal_duplex_authority_projection(
    value: Any,
) -> dict[str, Any]:
    try:
        return authority.canonical_original_confirmatory_terminal_custody_authority_template_projection(
            _require_mapping(
                value,
                role="terminal custody authority template projection",
            )
        )
    except authority.OriginalConfirmatoryCapsuleAuthorityError as exc:
        raise OriginalConfirmatoryTerminalError(
            "terminal custody authority template violates its exact policy"
        ) from exc


def _require_sha256(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise OriginalConfirmatoryTerminalError(f"{role} is not one SHA-256")
    return value


def _require_mapping(value: Any, *, role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise OriginalConfirmatoryTerminalError(f"{role} is not an object")
    return dict(value)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OriginalConfirmatoryTerminalError("JSON has a duplicate key")
        result[key] = value
    return result


def _decode_canonical_line(payload: bytes, *, role: str, maximum_bytes: int) -> dict[str, Any]:
    if (
        not payload
        or len(payload) > maximum_bytes
        or not payload.endswith(b"\n")
        or b"\r" in payload
        or b"\n" in payload[:-1]
    ):
        raise OriginalConfirmatoryTerminalError(f"{role} is not one bounded canonical JSON line")
    try:
        value = json.loads(
            payload[:-1].decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                OriginalConfirmatoryTerminalError(f"{role} contains non-finite token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OriginalConfirmatoryTerminalError(f"{role} is not strict UTF-8 JSON") from exc
    raw = _require_mapping(value, role=role)
    if _canonical_bytes(raw) != payload:
        raise OriginalConfirmatoryTerminalError(f"{role} bytes are not canonical")
    return raw


def _read_descriptor(descriptor: int, *, maximum_bytes: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    parts: list[bytes] = []
    remaining = maximum_bytes + 1
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        parts.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(parts)
    os.lseek(descriptor, 0, os.SEEK_SET)
    if len(payload) > maximum_bytes:
        raise OriginalConfirmatoryTerminalError("held file exceeds its fixed bound")
    return payload


@dataclass(slots=True)
class _HeldFile:
    path: Path
    descriptor: int
    payload: bytes
    role: str

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    @property
    def file_sha256(self) -> str:
        return _sha256_bytes(self.payload)

    def revalidate(self) -> None:
        if self.descriptor < 0:
            raise OriginalConfirmatoryTerminalError(f"{self.role} handle is closed")
        if (
            _read_descriptor(self.descriptor, maximum_bytes=max(len(self.payload), 1))
            != self.payload
        ):
            raise OriginalConfirmatoryTerminalError(f"{self.role} bytes changed")
        final = authority._windows_final_path_from_fd(self.descriptor)
        if final is not None and os.path.normcase(os.path.normpath(final)) != os.path.normcase(
            os.path.normpath(str(self.path))
        ):
            raise OriginalConfirmatoryTerminalError(f"{self.role} handle resolves to another path")


@dataclass(slots=True)
class _ConsumedEClaim:
    """Bootstrap-transferred read handle for the already-consumed one-use E."""

    held: _HeldFile
    launch_intent: _HeldFile
    process_started: _HeldFile
    claim: dict[str, Any]
    contract: Mapping[str, Any]
    e_intent: dict[str, Any]
    q_authority: dict[str, Any]

    def revalidate(self) -> None:
        self.held.revalidate()
        self.launch_intent.revalidate()
        self.process_started.revalidate()
        current = authority.canonical_original_confirmatory_e_consumption_claim(
            authority.decode_canonical_json_line(
                self.held.payload,
                role="retained consumed-E claim",
            ),
            contract=self.contract,
            e_intent=self.e_intent,
            q_authority=self.q_authority,
        )
        if (
            current != self.claim
            or current["process_started_sha256"] != self.process_started.file_sha256
        ):
            raise OriginalConfirmatoryTerminalError("consumed-E claim changed")

    def close(self) -> None:
        errors: list[BaseException] = []
        for item in (self.process_started, self.launch_intent, self.held):
            try:
                item.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise OriginalConfirmatoryTerminalError("consumed-E claim close failed") from errors[0]


@dataclass(frozen=True, slots=True)
class _BootstrapEClaimTransfer:
    descriptor: int
    path: Path
    sha256: str
    size_bytes: int


@dataclass(slots=True)
class _ValidatedLaunchEvidence:
    """Held, outcome-blind launch receipts validated before one-use E is armed."""

    launch_intent: _HeldFile
    process_started: _HeldFile

    def revalidate(self) -> None:
        self.launch_intent.revalidate()
        self.process_started.revalidate()

    def close(self) -> None:
        errors: list[BaseException] = []
        for item in (self.process_started, self.launch_intent):
            try:
                item.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise OriginalConfirmatoryTerminalError(
                "validated launch evidence close failed"
            ) from errors[0]


def _open_live_writer_overlap_read_descriptor(path: Path) -> int:
    """Open one read-only observer without denying an already-open writer."""

    if os.name != "nt":
        return authority._open_read_descriptor(path)
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    handle = create_file(
        str(path),
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        None,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if handle == INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(
            cast(int, handle),
            os.O_RDONLY | getattr(os, "O_BINARY", 0),
        )
    except BaseException:
        kernel32.CloseHandle(handle)
        raise


def _open_held_file(
    path: Path,
    *,
    role: str,
    maximum_bytes: int,
    allow_empty: bool = False,
    allow_live_writer: bool = False,
) -> _HeldFile:
    path = Path(authority._absolute_path(str(path), role=role))
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or path.is_symlink()
        or int(getattr(before, "st_file_attributes", 0)) & 0x400
        or int(before.st_nlink) != 1
        or int(before.st_size) > maximum_bytes
        or (not allow_empty and int(before.st_size) <= 0)
    ):
        raise OriginalConfirmatoryTerminalError(
            f"{role} is not one bounded single-link regular file"
        )
    descriptor = (
        _open_live_writer_overlap_read_descriptor(path)
        if allow_live_writer
        else authority._open_read_descriptor(path)
    )
    try:
        opened = os.fstat(descriptor)
        payload = _read_descriptor(descriptor, maximum_bytes=maximum_bytes)
        after = os.fstat(descriptor)
        after_path = path.lstat()
        comparable = ("st_dev", "st_ino", "st_size", "st_nlink", "st_mtime_ns")
        if any(
            int(getattr(item, field)) != int(getattr(before, field))
            for item in (opened, after, after_path)
            for field in comparable
        ):
            raise OriginalConfirmatoryTerminalError(f"{role} changed during open")
        if not allow_empty and not payload:
            raise OriginalConfirmatoryTerminalError(f"{role} is empty")
        if authority._windows_named_data_streams(path):
            raise OriginalConfirmatoryTerminalError(f"{role} has a named ADS")
        held = _HeldFile(path=path, descriptor=descriptor, payload=payload, role=role)
        descriptor = -1
        held.revalidate()
        return held
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _create_new_readonly_descriptor(path: Path) -> int:
    path = Path(authority._absolute_path(str(path), role="CREATE_NEW output"))
    if not path.parent.is_dir():
        raise OriginalConfirmatoryTerminalError("CREATE_NEW output parent must already exist")
    if os.name != "nt":
        return os.open(
            path,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o444,
        )
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    handle = create_file(
        str(path),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ,
        None,
        CREATE_NEW,
        FILE_ATTRIBUTE_READONLY | FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if handle == INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(
            cast(int, handle),
            os.O_RDWR | getattr(os, "O_BINARY", 0),
        )
    except BaseException:
        kernel32.CloseHandle(handle)
        raise


def _write_same_handle(descriptor: int, payload: bytes, *, maximum_bytes: int) -> None:
    if not payload or len(payload) > maximum_bytes:
        raise OriginalConfirmatoryTerminalError("output exceeds its exact bound")
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OriginalConfirmatoryTerminalError("same-handle write made no progress")
        view = view[written:]
    os.fsync(descriptor)
    if _read_descriptor(descriptor, maximum_bytes=maximum_bytes) != payload:
        raise OriginalConfirmatoryTerminalError("same-handle readback differs")


def _physical_identity(
    descriptor: int,
    *,
    path: Path,
    role: str,
    maximum_bytes: int = _MAX_CONTROL_BYTES,
    allow_empty: bool = False,
) -> dict[str, Any]:
    observed = os.fstat(descriptor)
    if maximum_bytes <= 0 or int(observed.st_size) < 0 or int(observed.st_size) > maximum_bytes:
        raise OriginalConfirmatoryTerminalError("held output exceeds its independent role bound")
    payload = _read_descriptor(
        descriptor,
        maximum_bytes=maximum_bytes,
    )
    volume, file_id = authority._native_file_identity_from_fd(descriptor)
    attributes = int(getattr(observed, "st_file_attributes", 0))
    read_only = (
        bool(attributes & FILE_ATTRIBUTE_READONLY)
        if os.name == "nt"
        else not bool(observed.st_mode & 0o222)
    )
    final = authority._windows_final_path_from_fd(descriptor)
    if final is not None and os.path.normcase(os.path.normpath(final)) != os.path.normcase(
        os.path.normpath(str(path))
    ):
        raise OriginalConfirmatoryTerminalError("held output resolves to another path")
    if (
        not stat.S_ISREG(observed.st_mode)
        or int(observed.st_nlink) != 1
        or (not allow_empty and not payload)
        or not read_only
        or authority._windows_named_data_streams(path)
    ):
        raise OriginalConfirmatoryTerminalError(
            "held output violates regular/read-only/single-link/no-ADS policy"
        )
    return {
        "schema_version": 1,
        "policy": authority.NO_FOLLOW_PHYSICAL_FILE_IDENTITY_POLICY,
        "role": role,
        "path": str(path),
        "volume_serial_number": volume,
        "file_id_128": file_id,
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "size_bytes": len(payload),
        "mode": int(observed.st_mode) & 0o7777,
        "file_attributes": attributes,
        "regular_file": True,
        "read_only": True,
        "link_count": 1,
        "modified_time_ns": int(observed.st_mtime_ns),
        "changed_time_ns": int(observed.st_ctime_ns),
        "sha256": _sha256_bytes(payload),
        "named_alternate_data_streams": [],
        "opened_without_reparse_follow": True,
        "share_access": ["FILE_SHARE_READ"],
    }


def _parse_tail(capsule_mode: str, tail: tuple[str, ...]) -> dict[str, str | None]:
    canonical = authority.canonical_original_confirmatory_capsule_mode_tail(
        capsule_mode=capsule_mode,
        tail_argv=tail,
    )
    if canonical != tail:
        raise OriginalConfirmatoryTerminalError("mode tail is not canonical")
    result: dict[str, str | None] = {}
    index = 0
    while index < len(tail):
        result[tail[index]] = tail[index + 1]
        index += 2
    if result.get("--execution-mode") == "fresh":
        result[authority.CAPSULE_SUCCESSOR_LINEAGE_FLAG] = None
    return result


def _environment_sha256() -> str:
    result: dict[str, str] = {}
    seen: set[str] = set()
    for key, value in os.environ.items():
        upper = key.upper()
        folded = upper.casefold()
        if not upper or "=" in upper or "\x00" in upper or "\x00" in value or folded in seen:
            raise OriginalConfirmatoryTerminalError(
                "observed process environment is not one closed map"
            )
        seen.add(folded)
        result[upper] = value
    return _canonical_sha256(dict(sorted(result.items())))


def _decode_supervisor_envelope(
    held: _HeldFile,
    *,
    role: str,
) -> tuple[dict[str, Any], str]:
    envelope = _decode_canonical_line(
        held.payload,
        role=role,
        maximum_bytes=max(_MAX_CONTROL_BYTES, len(held.payload)),
    )
    if set(envelope) != {"schema_version", "payload", "payload_sha256"}:
        raise OriginalConfirmatoryTerminalError(f"{role} envelope fields differ")
    payload = _require_mapping(envelope["payload"], role=f"{role} payload")
    if (
        type(envelope["schema_version"]) is not int
        or envelope["schema_version"] != 3
        or envelope["payload_sha256"] != _canonical_sha256(payload)
    ):
        raise OriginalConfirmatoryTerminalError(f"{role} envelope hash differs")
    return payload, held.file_sha256


def _canonical_command_dict(
    command: authority.OriginalConfirmatoryCapsuleCommand,
) -> dict[str, Any]:
    return command.as_dict()


@dataclass(slots=True)
class _VerifiedContext:
    mode: str
    tail: tuple[str, ...]
    values: dict[str, str | None]
    q_file: _HeldFile
    e_file: _HeldFile
    run_spec_file: _HeldFile
    q: dict[str, Any]
    e: dict[str, Any]
    run_spec_payload: dict[str, Any]
    spec: dict[str, Any]
    expected_artifact_rules: tuple[dict[str, Any], ...]
    spec_sha256: str
    scientific_command: authority.OriginalConfirmatoryCapsuleCommand
    preterminal_command: authority.OriginalConfirmatoryCapsuleCommand
    terminal_command: authority.OriginalConfirmatoryCapsuleCommand
    preterminal_contract: authority.OriginalConfirmatoryPreterminalPinContract
    overlap_contract: authority.OriginalConfirmatoryPreterminalOverlapHandshakeContract
    input_lease_contract: authority.OriginalConfirmatoryPostwakeInputLeaseContract
    custody_seed: authority.OriginalConfirmatoryPostwakeCustodySeed
    custody_contract: authority.OriginalConfirmatoryPostwakeCustodyHandshakeContract
    terminal_contract: authority.OriginalConfirmatoryTerminalCompositionContract
    supervisor_process_command_projection: dict[str, Any]
    terminal_launcher_release: dict[str, Any]
    terminal_launcher_projection: dict[str, Any]
    terminal_launcher_command: dict[str, Any] | None
    terminal_runtime_child_process_identity: dict[str, Any] | None
    immediate_venv_redirector_process_identity: dict[str, Any] | None
    immediate_venv_redirector_process_handle: int
    terminal_client_launcher_process_identity: dict[str, Any] | None
    terminal_client_launcher_process_handle: int
    launcher_source_file: _HeldFile | None
    observed_environment_sha256: str

    def close(self) -> None:
        errors: list[BaseException] = []
        files = [self.run_spec_file, self.e_file, self.q_file]
        if self.launcher_source_file is not None:
            files.insert(0, self.launcher_source_file)
        for item in files:
            try:
                item.close()
            except BaseException as exc:
                errors.append(exc)
        if self.immediate_venv_redirector_process_handle:
            try:
                _close_native_handle(self.immediate_venv_redirector_process_handle)
                self.immediate_venv_redirector_process_handle = 0
            except BaseException as exc:
                errors.append(exc)
        if self.terminal_client_launcher_process_handle:
            try:
                _close_native_handle(self.terminal_client_launcher_process_handle)
                self.terminal_client_launcher_process_handle = 0
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise OriginalConfirmatoryTerminalError(
                "verified terminal authority close failed"
            ) from errors[0]

    @property
    def job_id(self) -> str:
        return cast(str, self.e["job"]["job_id"])

    @property
    def job_dir(self) -> Path:
        return Path(cast(str, self.e["job"]["supervisor_job_dir"]))

    @property
    def attempt_id(self) -> str:
        return cast(str, self.e["job"]["attempt_id"])

    @property
    def run_id(self) -> str:
        return cast(str, self.e["job"]["run_id"])

    @property
    def attempt_nonce(self) -> str:
        return cast(str, self.e["job"]["launch_nonce"])

    @property
    def execution_mode(self) -> str:
        return cast(str, self.e["lineage"]["execution_mode"])

    @property
    def retry_of_run_id(self) -> str | None:
        return cast(str | None, self.e["lineage"]["retry_of_run_id"])


def _validate_control_staging_run_spec_binding(
    value: Any,
    *,
    run_spec_payload: Mapping[str, Any],
    spec: Mapping[str, Any],
    q: Mapping[str, Any],
    e: Mapping[str, Any],
    e_file_sha256: str,
) -> dict[str, Any]:
    binding = _require_mapping(value, role="control-staging run-spec binding")
    if set(binding) != _CONTROL_STAGING_OUTER_BINDING_FIELDS:
        raise OriginalConfirmatoryTerminalError("control-staging run-spec binding fields differ")
    projection = _require_mapping(
        q["control_staging_projection"],
        role="Q control-staging projection",
    )
    files = binding["files"]
    expected_names = list(authority.CONTROL_STAGING_EXACT_FILE_ALLOWLIST)
    expected_roles = [
        "staging-attempt",
        "e-intent",
        "launch-authorization",
        "supervisor-launch-spec",
        "staging-ready",
    ]
    if not isinstance(files, list) or len(files) != len(expected_names):
        raise OriginalConfirmatoryTerminalError("control-staging retained file inventory differs")
    canonical_files: list[dict[str, Any]] = []
    for index, value_record in enumerate(files):
        record = _require_mapping(value_record, role="control-staging retained file")
        if set(record) != {
            "role",
            "name",
            "path",
            "size_bytes",
            "file_sha256",
            "physical_identity",
            "physical_identity_root_sha256",
        }:
            raise OriginalConfirmatoryTerminalError("control-staging retained file fields differ")
        identity = _require_mapping(
            record["physical_identity"],
            role="control-staging retained file identity",
        )
        if (
            record["name"] != expected_names[index]
            or record["role"] != expected_roles[index]
            or record["path"]
            != str(Path(cast(str, projection["control_staging_dir"])) / expected_names[index])
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] <= 0
            or _require_sha256(record["file_sha256"], role="control-staging retained file")
            != record["file_sha256"]
            or _require_sha256(
                record["physical_identity_root_sha256"],
                role="control-staging retained identity",
            )
            != _canonical_sha256(identity)
        ):
            raise OriginalConfirmatoryTerminalError("control-staging retained file binding differs")
        canonical_files.append(record)
    source_record = canonical_files[3]
    e_record = canonical_files[1]
    authorization_record = canonical_files[2]
    authorization = _require_mapping(
        spec["authorization"], role="canonical supervisor authorization"
    )
    unsigned = {key: item for key, item in binding.items() if key != "binding_root_sha256"}
    supervisor_root = Path(cast(str, q["supervisor_release"]["supervisor_state_root"]))
    if (
        type(binding["schema_version"]) is not int
        or binding["schema_version"] != 1
        or binding["policy"] != _CONTROL_STAGING_OUTER_BINDING_POLICY
        or binding["job_id"] != spec["job_id"]
        or binding["supervisor_root"] != str(supervisor_root)
        or binding["control_staging_root"]
        != str(supervisor_root / authority.CONTROL_STAGING_DIRECTORY_NAME)
        or binding["control_staging_dir"] != projection["control_staging_dir"]
        or binding["control_staging_projection"] != projection
        or binding["control_staging_projection_sha256"] != q["control_staging_projection_sha256"]
        or binding["expected_complete_leaf_names"] != expected_names
        or binding["publication_order"] != expected_names
        or binding["file_count"] != len(expected_names)
        or binding["source_path"] != projection["supervisor_launch_spec_path"]
        or binding["source_path"] != source_record["path"]
        or binding["source_size_bytes"] != source_record["size_bytes"]
        or binding["source_file_sha256"] != source_record["file_sha256"]
        or binding["source_canonical_bytes_sha256"] != source_record["file_sha256"]
        or binding["source_bytes_equal_canonical_spec_serialization"] is not True
        or binding["e_intent_path"] != projection["e_intent_path"]
        or binding["e_intent_path"] != e_record["path"]
        or binding["e_intent_file_sha256"] != e_file_sha256
        or binding["e_intent_file_sha256"] != e_record["file_sha256"]
        or binding["launch_authorization_path"] != projection["launch_authorization_path"]
        or binding["launch_authorization_path"] != authorization_record["path"]
        or binding["launch_authorization_file_sha256"] != authorization_record["file_sha256"]
        or set(authorization) != {"path", "sha256"}
        or authorization["path"] != authorization_record["path"]
        or authorization["sha256"] != authorization_record["file_sha256"]
        or run_spec_payload["source_path"] != binding["source_path"]
        or run_spec_payload["source_file_sha256"] != binding["source_file_sha256"]
        or binding["retained_from_before_final_job_creation_through_terminal"] is not True
        or binding["final_job_creation_owner"] != "suspended_supervisor_after_resume_v1"
        or binding["pre_ack_final_job_publication_scope"]
        != ["jobs/<job_id>", "run_spec.json", "q_e_custody_receipt.json"]
        or binding["pre_ack_metadata_only_publication_allowed"] is not True
        or binding["pre_ack_scientific_process_launch_allowed"] is not False
        or binding["q_e_ack_required_before_scientific_process_launch"] is not True
        or binding["automatic_retry_allowed"] is not False
        or binding["adoption_allowed"] is not False
        or binding["cleanup_allowed"] is not False
        or binding["binding_root_sha256"] != _canonical_sha256(unsigned)
        or e["job"]["supervisor_job_dir"] != projection["final_job_dir"]
    ):
        raise OriginalConfirmatoryTerminalError(
            "control-staging run-spec binding differs from sealed Q/E"
        )
    return binding


def _take_bootstrap_consumed_e_transfer(
    job_dir: Path,
    *,
    bootstrap_module: Any | None = None,
) -> _BootstrapEClaimTransfer:
    """Arm and transfer the bootstrap claim once; the caller owns the descriptor."""

    module = bootstrap_module if bootstrap_module is not None else sys.modules.get("__main__")
    arm = getattr(
        module,
        "_arm_original_confirmatory_e_claim_after_full_prevalidation",
        None,
    )
    take = getattr(module, "_take_original_confirmatory_e_claim_read_handle", None)
    if not callable(arm) or not callable(take):
        raise OriginalConfirmatoryTerminalError(
            "capsule bootstrap verifier claim API is unavailable"
        )
    arm()
    transferred = take()
    if (
        not isinstance(transferred, tuple)
        or len(transferred) != 4
        or type(transferred[0]) is not int
        or not isinstance(transferred[1], str)
        or not isinstance(transferred[2], str)
        or type(transferred[3]) is not int
    ):
        if (
            isinstance(transferred, tuple)
            and transferred
            and type(transferred[0]) is int
            and transferred[0] >= 0
        ):
            os.close(transferred[0])
        raise OriginalConfirmatoryTerminalError(
            "capsule bootstrap returned an invalid verifier claim"
        )
    descriptor, path_text, expected_sha256, expected_size = transferred
    claim_path = job_dir / authority.E_CONSUMPTION_TOMBSTONE_FILENAME
    if (
        descriptor < 0
        or Path(path_text) != claim_path
        or expected_size <= 0
        or _require_sha256(expected_sha256, role="bootstrap consumed-E claim") != expected_sha256
    ):
        if descriptor >= 0:
            os.close(descriptor)
        raise OriginalConfirmatoryTerminalError("bootstrap verifier claim identity differs from E")
    return _BootstrapEClaimTransfer(
        descriptor=descriptor,
        path=claim_path,
        sha256=expected_sha256,
        size_bytes=expected_size,
    )


def _arm_and_take_consumed_e_claim(
    context: _VerifiedContext,
    *,
    launch_evidence: _ValidatedLaunchEvidence,
    bootstrap_module: Any | None = None,
) -> _ConsumedEClaim:
    """Take and validate consumed E after the complete outcome-blind context."""

    transfer = _take_bootstrap_consumed_e_transfer(
        context.job_dir,
        bootstrap_module=bootstrap_module,
    )
    descriptor = transfer.descriptor
    try:
        launch_evidence.revalidate()
        payload = _read_descriptor(descriptor, maximum_bytes=_MAX_CONTROL_BYTES)
        if len(payload) != transfer.size_bytes or _sha256_bytes(payload) != transfer.sha256:
            raise OriginalConfirmatoryTerminalError("bootstrap verifier claim bytes differ")
        claim = authority.canonical_original_confirmatory_e_consumption_claim(
            authority.decode_canonical_json_line(
                payload,
                role="consumed-E claim",
            ),
            contract=context.e["e_consumption_contract"],
            e_intent=context.e,
            q_authority=context.q,
        )
        if (
            claim["supervisor_spec_sha256"] != context.spec_sha256
            or claim["process_started_sha256"] != launch_evidence.process_started.file_sha256
            or claim["supervisor_job_id"] != context.job_id
            or claim["attempt_id"] != context.attempt_id
            or claim["run_id"] != context.run_id
            or claim["launch_nonce"] != context.attempt_nonce
            or claim["execution_mode"] != context.execution_mode
            or claim["retry_of_run_id"] != context.retry_of_run_id
        ):
            raise OriginalConfirmatoryTerminalError(
                "consumed-E claim differs from exact downstream attempt"
            )
        held = _HeldFile(
            path=transfer.path,
            descriptor=descriptor,
            payload=payload,
            role="consumed-E claim",
        )
        descriptor = -1
        try:
            consumed = _ConsumedEClaim(
                held=held,
                launch_intent=launch_evidence.launch_intent,
                process_started=launch_evidence.process_started,
                claim=claim,
                contract=_require_mapping(
                    context.e["e_consumption_contract"],
                    role="E consumption contract",
                ),
                e_intent=context.e,
                q_authority=context.q,
            )
            consumed.revalidate()
            return consumed
        except BaseException:
            held.close()
            raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _require_v3_external_runtime_shape(spec: Mapping[str, Any]) -> None:
    external = _require_mapping(
        spec.get("external_codex_handoff"),
        role="SPEC52 external Codex handoff",
    )
    if (
        set(spec) != _CANONICAL_SUPERVISOR_SPEC_FIELDS
        or type(spec.get("schema_version")) is not int
        or spec.get("schema_version") != 3
        or spec.get("policy") != authority.SUPERVISOR_V3_POLICY
        or spec.get("codex") is not None
        or spec.get("handoff_session") is not None
        or type(spec.get("max_attempt_count")) is not int
        or spec.get("max_attempt_count") != 1
        or spec.get("automatic_retry_allowed") is not False
        or set(external) != _EXTERNAL_CODEX_HANDOFF_FIELDS
        or external.get("policy") != authority.EXTERNAL_CODEX_HANDOFF_POLICY
        or external.get("internal_codex_wake_allowed") is not False
        or external.get("legacy_handoff_session_allowed") is not False
        or external.get("single_wake_owner") != authority.EXTERNAL_CODEX_HANDOFF_SINGLE_WAKE_OWNER
    ):
        raise OriginalConfirmatoryTerminalError(
            "SPEC57 violates external single-wake runtime ownership"
        )


def _validate_authority_built_spec52(
    spec: Mapping[str, Any],
    *,
    q: Mapping[str, Any],
    e: Mapping[str, Any],
    e_file_sha256: str,
) -> None:
    custody = _require_mapping(
        spec["postwake_custody_handshake_contract"],
        role="SPEC52 postwake custody contract",
    )
    pipe_owner_sid = custody.get("pipe_owner_sid")
    if not isinstance(pipe_owner_sid, str):
        raise OriginalConfirmatoryTerminalError("SPEC52 pipe-owner SID is absent")
    try:
        expected_source = authority.build_original_confirmatory_external_supervisor_spec_payload_v3(
            q_authority=q,
            e_intent=e,
            e_file_sha256=e_file_sha256,
            q_e_custody_spec_fields={
                key: spec[key]
                for key in (
                    "q_e_custody_contract",
                    "q_e_custody_handoff",
                    "q_e_custody_receipt",
                )
            },
            external_codex_handoff=_require_mapping(
                spec["external_codex_handoff"],
                role="SPEC52 external Codex handoff",
            ),
            pipe_owner_sid=pipe_owner_sid,
        )
    except authority.OriginalConfirmatoryCapsuleAuthorityError as exc:
        raise OriginalConfirmatoryTerminalError(
            "SPEC52 differs from admitted Q20/E23 authority"
        ) from exc
    observed_source = {key: spec[key] for key in expected_source}
    verifier = _require_mapping(
        observed_source["integrity_verifier"],
        role="SPEC52 source integrity verifier",
    )
    observed_source["integrity_verifier"] = {
        key: verifier[key] for key in ("program_path", "program_sha256", "argv", "cwd")
    }
    if observed_source != expected_source:
        raise OriginalConfirmatoryTerminalError(
            "sealed canonical SPEC57 differs from authority-built SPEC52"
        )


def _load_verified_context(
    capsule_mode: str,
    canonical_tail: tuple[str, ...],
) -> _VerifiedContext:
    values = _parse_tail(capsule_mode, canonical_tail)
    e_path = Path(cast(str, values["--e-intent"]))
    e_file = _open_held_file(
        e_path,
        role="sealed E",
        maximum_bytes=_MAX_CONTROL_BYTES,
    )
    q_file: _HeldFile | None = None
    run_spec_file: _HeldFile | None = None
    launcher_source_file: _HeldFile | None = None
    terminal_runtime_child_process_identity: dict[str, Any] | None = None
    immediate_venv_redirector_process_identity: dict[str, Any] | None = None
    immediate_venv_redirector_process_handle = 0
    terminal_client_launcher_process_identity: dict[str, Any] | None = None
    terminal_client_launcher_process_handle = 0
    terminal_launcher_command: dict[str, Any] | None = None
    try:
        if e_file.file_sha256 != values["--e-intent-sha256"]:
            raise OriginalConfirmatoryTerminalError("sealed E file hash differs from tail")
        e_raw = authority.decode_canonical_json_line(e_file.payload, role="sealed E")
        direct_q = _require_mapping(e_raw.get("q_authority"), role="E direct Q")
        q_path_value = direct_q.get("path")
        if not isinstance(q_path_value, str):
            raise OriginalConfirmatoryTerminalError("E does not bind one Q path")
        q_file = _open_held_file(
            Path(q_path_value),
            role="sealed Q",
            maximum_bytes=_MAX_CONTROL_BYTES,
        )
        if q_file.file_sha256 != direct_q.get("file_sha256"):
            raise OriginalConfirmatoryTerminalError("sealed Q file hash differs from E")
        q = authority.canonical_original_confirmatory_q_replacement_v2(
            authority.decode_canonical_json_line(q_file.payload, role="sealed Q")
        )
        if q["q_authority_root_sha256"] != values["--q-authority-root-sha256"]:
            raise OriginalConfirmatoryTerminalError("Q authority root differs from tail")
        e = authority.canonical_original_confirmatory_e_intent(
            e_raw,
            q_authority=q,
            expected_q_file_sha256=q_file.file_sha256,
        )
        if e["intent_core_sha256"] != values["--e-intent-core-sha256"]:
            raise OriginalConfirmatoryTerminalError("E core hash differs from tail")
        commands = {
            mode: authority.derive_original_confirmatory_capsule_command_from_e(
                e_intent=e,
                e_file_sha256=e_file.file_sha256,
                q_authority=q,
                capsule_mode=mode,
            )
            for mode in authority.CAPSULE_ALLOWED_MODES
        }
        expected_command = commands[capsule_mode]
        expected_tail = tuple(expected_command.argv[5:])
        if expected_tail != canonical_tail:
            raise OriginalConfirmatoryTerminalError(
                "current command differs from sealed E derivation"
            )
        if Path.cwd() != Path(q["project_root"]):
            raise OriginalConfirmatoryTerminalError(
                "terminal verifier cwd differs from Q project root"
            )
        observed_environment_sha256 = _environment_sha256()
        environment_binding = authority.canonical_original_confirmatory_process_environment_binding(
            e["process_environment_binding"],
            expected_environment=e["expected_launch_environment"],
        )
        if (
            observed_environment_sha256
            != environment_binding.exact_integrity_verifier_environment_sha256
        ):
            raise OriginalConfirmatoryTerminalError("terminal verifier environment differs from E")
        spec_path = Path(cast(str, e["job"]["supervisor_spec_path"]))
        supervisor_release = _require_mapping(
            q["supervisor_release"],
            role="Q supervisor release",
        )
        if e["supervisor_release"] != supervisor_release:
            raise OriginalConfirmatoryTerminalError("E supervisor release differs from Q")
        terminal_launcher_release = _require_mapping(
            supervisor_release["terminal_client_launcher_release"],
            role="Q terminal-client launcher release",
        )
        terminal_custody_projection = _require_mapping(
            e["job"]["terminal_custody_authority_projection"],
            role="E terminal custody authority projection",
        )
        terminal_launcher_projection = _require_mapping(
            terminal_custody_projection["terminal_client_launcher_projection"],
            role="E terminal-client launcher projection",
        )
        control_staging_projection = _require_mapping(
            q["control_staging_projection"],
            role="Q control staging projection",
        )
        supervisor_launch_spec_path = Path(
            cast(str, control_staging_projection["supervisor_launch_spec_path"])
        )
        staged_e_intent_path = Path(cast(str, control_staging_projection["e_intent_path"]))
        supervisor_process_command_projection = (
            authority.build_original_confirmatory_supervisor_process_command_projection(
                supervisor_release["supervisor_process_command_derivation_contract"],
                capsule=q["execution_capsule"],
                supervisor_launch_spec_path=supervisor_launch_spec_path,
                staged_e_intent_path=staged_e_intent_path,
            )
        )
        if (
            capsule_mode == authority.CAPSULE_PRETERMINAL_MODE
            and Path(cast(str, values["--run-spec"])) != spec_path
        ):
            raise OriginalConfirmatoryTerminalError("preterminal run-spec path differs from E")
        run_spec_file = _open_held_file(
            spec_path,
            role="sealed supervisor run spec",
            maximum_bytes=_MAX_CONTROL_BYTES,
        )
        run_spec_payload, spec_file_sha = _decode_supervisor_envelope(
            run_spec_file,
            role="sealed supervisor run spec",
        )
        if set(run_spec_payload) != _RUN_SPEC_PAYLOAD_FIELDS:
            raise OriginalConfirmatoryTerminalError("supervisor run-spec payload fields differ")
        expected_supervisor_source = {
            "path": supervisor_release["supervisor_source_path"],
            "sha256": supervisor_release["supervisor_source_sha256"],
        }
        if (
            type(run_spec_payload["schema_version"]) is not int
            or run_spec_payload["schema_version"] != 3
            or run_spec_payload["policy"] != authority.SUPERVISOR_V3_POLICY
            or run_spec_payload["supervisor"] != expected_supervisor_source
            or not isinstance(run_spec_payload["source_path"], str)
            or not Path(run_spec_payload["source_path"]).is_absolute()
            or _require_sha256(
                run_spec_payload["source_file_sha256"],
                role="supervisor source-spec file",
            )
            != run_spec_payload["source_file_sha256"]
            or not _is_utc(run_spec_payload["frozen_at_utc"])
        ):
            raise OriginalConfirmatoryTerminalError(
                "supervisor run-spec envelope differs from Q release"
            )
        spec = _require_mapping(
            run_spec_payload["canonical_spec"],
            role="canonical supervisor spec",
        )
        if set(spec) != _CANONICAL_SUPERVISOR_SPEC_FIELDS:
            raise OriginalConfirmatoryTerminalError("canonical supervisor spec fields differ")
        _require_v3_external_runtime_shape(spec)
        canonical_spec_sha = _canonical_sha256(spec)
        if run_spec_payload["canonical_spec_sha256"] != canonical_spec_sha:
            raise OriginalConfirmatoryTerminalError("canonical supervisor spec hash differs")
        _validate_control_staging_run_spec_binding(
            run_spec_payload["control_staging"],
            run_spec_payload=run_spec_payload,
            spec=spec,
            q=q,
            e=e,
            e_file_sha256=e_file.file_sha256,
        )
        _validate_authority_built_spec52(
            spec,
            q=q,
            e=e,
            e_file_sha256=e_file.file_sha256,
        )
        if (
            type(spec.get("schema_version")) is not int
            or spec.get("schema_version") != 3
            or spec.get("policy") != authority.SUPERVISOR_V3_POLICY
            or spec.get("job_id") != e["job"]["job_id"]
            or spec.get("process_kind") != _SUPERVISOR_PROCESS_KIND
            or spec.get("project_root") != q["project_root"]
            or spec.get("program_path")
            != str(commands[authority.CAPSULE_SCIENTIFIC_MODE].program_path)
            or spec.get("program_sha256")
            != commands[authority.CAPSULE_SCIENTIFIC_MODE].program_sha256
            or spec.get("argv") != list(commands[authority.CAPSULE_SCIENTIFIC_MODE].argv)
            or spec.get("command")
            != _canonical_command_dict(commands[authority.CAPSULE_SCIENTIFIC_MODE])
            or spec.get("max_attempt_count") != 1
            or spec.get("automatic_retry_allowed") is not False
            or spec.get("handoff_session") is not None
            or spec.get("codex") is not None
            or spec.get("expected_environment") != e["expected_launch_environment"]
            or spec.get("process_environment_binding") != e["process_environment_binding"]
            or spec.get("e_consumption_contract") != e["e_consumption_contract"]
        ):
            raise OriginalConfirmatoryTerminalError("supervisor spec main command differs from Q/E")
        integrity_verifier = _require_mapping(
            spec.get("integrity_verifier"),
            role="supervisor preterminal command",
        )
        if integrity_verifier != _canonical_command_dict(
            commands[authority.CAPSULE_PRETERMINAL_MODE]
        ):
            raise OriginalConfirmatoryTerminalError(
                "supervisor preterminal command differs from Q/E"
            )
        expected_artifact_rules = _canonical_outcome_blind_expected_artifact_rules(
            spec=spec,
            e_intent=e,
        )
        preterminal_contract = authority.canonical_original_confirmatory_preterminal_pin_contract(
            spec["preterminal_pin_contract"],
            capsule=q["execution_capsule"],
            verifier_command=commands[authority.CAPSULE_PRETERMINAL_MODE],
            verifier_command_tail_argv=tuple(commands[authority.CAPSULE_PRETERMINAL_MODE].argv[5:]),
        )
        overlap_contract = (
            authority.canonical_original_confirmatory_preterminal_overlap_handshake_contract(
                spec["preterminal_overlap_handshake_contract"]
            )
        )
        input_lease_contract = (
            authority.canonical_original_confirmatory_postwake_input_lease_contract(
                spec["postwake_input_lease_contract"]
            )
        )
        custody_seed = authority.canonical_original_confirmatory_postwake_custody_seed(
            spec["postwake_custody_seed"]
        )
        custody_contract = (
            authority.canonical_original_confirmatory_postwake_custody_handshake_contract(
                spec["postwake_custody_handshake_contract"],
                custody_seed=custody_seed,
            )
        )
        terminal_contract = authority.canonical_original_confirmatory_terminal_composition_contract(
            spec["terminal_composition_contract"],
            capsule=q["execution_capsule"],
            verifier_command=commands[authority.CAPSULE_TERMINAL_MODE],
            verifier_command_tail_argv=tuple(commands[authority.CAPSULE_TERMINAL_MODE].argv[5:]),
            preterminal_pin_contract_sha256=preterminal_contract.contract_sha256,
            preterminal_overlap_handshake_contract=overlap_contract,
            postwake_input_lease_contract=input_lease_contract,
            postwake_custody_seed=custody_seed,
            postwake_custody_handshake_contract=custody_contract,
            expected_run_directory=e["scientific_request_projection"]["expected_run_directory"],
            expected_terminal_custody_authority_projection=(terminal_custody_projection),
            terminal_client_launcher_release=terminal_launcher_release,
            expected_environment=e["expected_launch_environment"],
            process_environment_binding=e["process_environment_binding"],
        )
        if (
            terminal_contract.verifier_command
            != commands[authority.CAPSULE_TERMINAL_MODE].as_dict()
            or terminal_contract.verifier_command_sha256
            != commands[authority.CAPSULE_TERMINAL_MODE].command_sha256
            or custody_contract.expected_composed_command_sha256
            != commands[authority.CAPSULE_TERMINAL_MODE].command_sha256
            or custody_contract.terminal_client_arrival_timeout_ms
            != authority.TERMINAL_CLIENT_ARRIVAL_TIMEOUT_MS
            or custody_contract.custody_exchange_timeout_ms != authority.CUSTODY_EXCHANGE_TIMEOUT_MS
            or terminal_contract.terminal_custody_authority_projection
            != terminal_custody_projection
            or terminal_launcher_projection["projection_root_sha256"]
            != terminal_custody_projection["terminal_client_launcher_projection_root_sha256"]
            or custody_seed.payload["q_authority_root_sha256"] != q["q_authority_root_sha256"]
            or custody_seed.payload["e_intent_file_sha256"] != e_file.file_sha256
            or custody_seed.payload["e_intent_core_sha256"] != e["intent_core_sha256"]
            or custody_seed.payload["supervisor_spec_path"] != str(spec_path)
            or custody_seed.payload["supervisor_job_id"] != e["job"]["job_id"]
            or custody_seed.payload["launch_nonce"] != e["job"]["launch_nonce"]
            or custody_seed.payload["attempt_id"] != e["job"]["attempt_id"]
            or custody_seed.payload["run_id"] != e["job"]["run_id"]
            or custody_seed.payload["execution_mode"] != e["lineage"]["execution_mode"]
            or custody_seed.payload["retry_of_run_id"] != e["lineage"]["retry_of_run_id"]
            or spec_file_sha != run_spec_file.file_sha256
        ):
            raise OriginalConfirmatoryTerminalError(
                "downstream spec terminal chain differs from Q/E"
            )
        if capsule_mode == authority.CAPSULE_TERMINAL_MODE:
            (
                terminal_runtime_child_process_identity,
                immediate_venv_redirector_process_identity,
                immediate_venv_redirector_process_handle,
                terminal_client_launcher_process_identity,
                terminal_client_launcher_process_handle,
                terminal_launcher_command,
                launcher_source_file,
            ) = _establish_terminal_process_ancestry(
                terminal_command=commands[authority.CAPSULE_TERMINAL_MODE],
                launcher_release=terminal_launcher_release,
                launcher_projection=terminal_launcher_projection,
                capsule=q["execution_capsule"],
                supervisor_spec_sha256=run_spec_file.file_sha256,
                e_intent_file_sha256=e_file.file_sha256,
            )
        return _VerifiedContext(
            mode=capsule_mode,
            tail=canonical_tail,
            values=values,
            q_file=q_file,
            e_file=e_file,
            run_spec_file=run_spec_file,
            q=q,
            e=e,
            run_spec_payload=run_spec_payload,
            spec=spec,
            expected_artifact_rules=expected_artifact_rules,
            spec_sha256=run_spec_file.file_sha256,
            scientific_command=commands[authority.CAPSULE_SCIENTIFIC_MODE],
            preterminal_command=commands[authority.CAPSULE_PRETERMINAL_MODE],
            terminal_command=commands[authority.CAPSULE_TERMINAL_MODE],
            preterminal_contract=preterminal_contract,
            overlap_contract=overlap_contract,
            input_lease_contract=input_lease_contract,
            custody_seed=custody_seed,
            custody_contract=custody_contract,
            terminal_contract=terminal_contract,
            supervisor_process_command_projection=(supervisor_process_command_projection),
            terminal_launcher_release=terminal_launcher_release,
            terminal_launcher_projection=terminal_launcher_projection,
            terminal_launcher_command=terminal_launcher_command,
            terminal_runtime_child_process_identity=(terminal_runtime_child_process_identity),
            immediate_venv_redirector_process_identity=(immediate_venv_redirector_process_identity),
            immediate_venv_redirector_process_handle=(immediate_venv_redirector_process_handle),
            terminal_client_launcher_process_identity=(terminal_client_launcher_process_identity),
            terminal_client_launcher_process_handle=(terminal_client_launcher_process_handle),
            launcher_source_file=launcher_source_file,
            observed_environment_sha256=observed_environment_sha256,
        )
    except BaseException:
        if terminal_client_launcher_process_handle:
            _close_native_handle(terminal_client_launcher_process_handle)
        if immediate_venv_redirector_process_handle:
            _close_native_handle(immediate_venv_redirector_process_handle)
        if launcher_source_file is not None:
            launcher_source_file.close()
        if run_spec_file is not None:
            run_spec_file.close()
        if q_file is not None:
            q_file.close()
        e_file.close()
        raise


def _load_context_then_take_consumed_e(
    capsule_mode: str,
    canonical_tail: tuple[str, ...],
    *,
    bootstrap_module: Any | None = None,
) -> tuple[_VerifiedContext, _ConsumedEClaim]:
    """Complete no-data prevalidation before arming the irreversible E claim."""

    context = _load_verified_context(capsule_mode, canonical_tail)
    launch_evidence: _ValidatedLaunchEvidence | None = None
    try:
        _revalidate_terminal_process_ancestry(context)
        launch_evidence = _open_validated_launch_evidence(context)
        _revalidate_terminal_process_ancestry(context)
        consumed_e = _arm_and_take_consumed_e_claim(
            context,
            launch_evidence=launch_evidence,
            bootstrap_module=bootstrap_module,
        )
    except BaseException:
        if launch_evidence is not None:
            launch_evidence.close()
        context.close()
        raise
    return context, consumed_e


def _strict_json_value_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            _strict_json_value_equal(actual[key], item) for key, item in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_json_value_equal(observed, item)
            for observed, item in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _build_outcome_blind_expected_artifact_rules(
    *,
    e_intent: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    scientific = _require_mapping(
        e_intent["scientific_request_projection"],
        role="E scientific request projection for artifact inspection",
    )
    run_id = scientific.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise OriginalConfirmatoryTerminalError("artifact inspection run id is invalid")
    runs_root = Path(
        authority._absolute_path(
            scientific.get("runs_root"),
            role="artifact inspection runs root",
        )
    )
    run_directory = Path(
        authority._absolute_path(
            scientific.get("expected_run_directory"),
            role="artifact inspection run directory",
        )
    )
    if run_directory != runs_root / run_id:
        raise OriginalConfirmatoryTerminalError("artifact inspection run directory differs from E")
    anchors = {
        "expected_run_directory": run_directory,
        "runs_root": runs_root,
    }
    records: list[dict[str, Any]] = []
    for template in _EXPECTED_ARTIFACT_TEMPLATE:
        anchor_name = cast(str, template["path_anchor"])
        checks = {
            key: run_id if expected == "$RUN_ID" else expected
            for key, expected in cast(
                Mapping[str, Any],
                template["json_equals"],
            ).items()
        }
        records.append(
            {
                "role": template["role"],
                "path": str(anchors[anchor_name] / cast(str, template["relative_path"])),
                "expected_sha256": None,
                "must_be_absent_before": template["must_be_absent_before"],
                "json_equals": checks,
            }
        )
    return tuple(records)


def _canonical_outcome_blind_expected_artifact_rules(
    *,
    spec: Mapping[str, Any],
    e_intent: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    expected = _build_outcome_blind_expected_artifact_rules(
        e_intent=e_intent,
    )
    custody_projection = _require_mapping(
        _require_mapping(
            e_intent["job"],
            role="E job for artifact inspection",
        )["terminal_custody_authority_projection"],
        role="E terminal custody authority for artifact inspection",
    )
    artifact_instance = _require_mapping(
        custody_projection["outcome_blind_expected_artifact_instance"],
        role="E outcome-blind artifact instance",
    )
    supplied = spec.get("expected_artifacts")
    required_roles = spec.get("required_success_roles")
    if (
        not isinstance(supplied, list)
        or len(supplied) != len(expected)
        or required_roles != list(_EXPECTED_ARTIFACT_ROLE_ORDER)
        or artifact_instance["expected_artifacts"] != list(expected)
        or artifact_instance["required_success_roles"] != list(_EXPECTED_ARTIFACT_ROLE_ORDER)
        or artifact_instance["expected_artifacts_root_sha256"] != _canonical_sha256(list(expected))
    ):
        raise OriginalConfirmatoryTerminalError(
            "supervisor expected-artifact roles/order differ from outcome-blind allowlist"
        )
    for index, (raw_value, expected_rule) in enumerate(zip(supplied, expected, strict=True)):
        raw = _require_mapping(
            raw_value,
            role=f"expected artifact rule {index}",
        )
        if set(raw) != _EXPECTED_ARTIFACT_RULE_FIELDS or not _strict_json_value_equal(
            raw, expected_rule
        ):
            raise OriginalConfirmatoryTerminalError(
                "supervisor expected-artifact rule differs from outcome-blind allowlist"
            )
    return expected


def _inspect_expected_artifact(
    rule: Any,
    *,
    expected_rule: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _require_mapping(rule, role="expected artifact rule")
    if set(raw) != _EXPECTED_ARTIFACT_RULE_FIELDS or not _strict_json_value_equal(
        raw, expected_rule
    ):
        raise OriginalConfirmatoryTerminalError(
            "expected artifact rule differs from outcome-blind allowlist"
        )
    role = raw["role"]
    if not isinstance(role, str) or not role:
        raise OriginalConfirmatoryTerminalError("expected artifact role is invalid")
    path = Path(authority._absolute_path(raw["path"], role=f"artifact {role}"))
    held = _open_held_file(
        path,
        role=f"artifact {role}",
        maximum_bytes=_MAX_CONTROL_BYTES,
    )
    try:
        expected_sha = raw["expected_sha256"]
        if expected_sha is not None and held.file_sha256 != _require_sha256(
            expected_sha,
            role=f"artifact {role} expected hash",
        ):
            raise OriginalConfirmatoryTerminalError(f"artifact {role} hash differs")
        checks = _require_mapping(
            raw["json_equals"],
            role=f"artifact {role} JSON checks",
        )
        checked_paths: list[str] = []
        if checks:
            decoded = _require_mapping(
                json.loads(
                    held.payload.decode("utf-8"),
                    object_pairs_hook=_strict_object,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        OriginalConfirmatoryTerminalError(
                            f"artifact {role} contains non-finite {token}"
                        )
                    ),
                ),
                role=f"artifact {role} JSON control object",
            )
            for selector, expected in sorted(checks.items()):
                if (
                    "." in selector
                    or selector not in decoded
                    or not _strict_json_value_equal(
                        decoded[selector],
                        expected,
                    )
                ):
                    raise OriginalConfirmatoryTerminalError(
                        f"artifact {role} control value differs"
                    )
                checked_paths.append(selector)
        held.revalidate()
        return {
            "role": role,
            "path": str(path),
            "size_bytes": len(held.payload),
            "sha256": held.file_sha256,
            "expected_sha256": expected_sha,
            "json_control_paths_checked": checked_paths,
            "valid": True,
        }
    finally:
        held.close()


def _read_required_receipt(
    path: Path,
    *,
    role: str,
) -> tuple[_HeldFile, dict[str, Any], str]:
    held = _open_held_file(path, role=role, maximum_bytes=_MAX_CONTROL_BYTES)
    try:
        payload, file_sha = _decode_supervisor_envelope(held, role=role)
        return held, payload, file_sha
    except BaseException:
        held.close()
        raise


def _validate_prearm_process_absence(
    value: Any,
    *,
    context: _VerifiedContext,
) -> dict[str, Any]:
    raw = _require_mapping(value, role="pre-arm process absence")
    if (
        set(raw) != _PREARM_PROCESS_ABSENCE_FIELDS
        or type(raw["schema_version"]) is not int
        or raw["schema_version"] != 3
        or raw["policy"] != "exact_argv_and_protected_process_absence_v1"
        or not _is_utc(raw["observed_at_utc"])
        or type(raw["inventory_process_count"]) is not int
        or raw["inventory_process_count"] < 0
        or raw["target_program_path"] != context.spec["program_path"]
        or raw["target_program_sha256"] != context.spec["program_sha256"]
        or raw["target_command_sha256"] != context.scientific_command.command_sha256
        or raw["target_argv_sha256"] != _canonical_sha256(context.spec["argv"])
        or raw["exact_command_matches"] != []
        or raw["protected_marker_matches"] != []
        or raw["absence_verified"] is not True
    ):
        raise OriginalConfirmatoryTerminalError("pre-arm process-absence evidence is invalid")
    return raw


def _validate_launch_intent_payload(
    value: Any,
    *,
    context: _VerifiedContext,
) -> dict[str, Any]:
    launch = _require_mapping(value, role="launch intent payload")
    if set(launch) != _LAUNCH_INTENT_FIELDS:
        raise OriginalConfirmatoryTerminalError("launch intent payload fields differ")
    binding = _require_mapping(
        context.spec["process_environment_binding"],
        role="supervisor process-environment binding",
    )
    expected_environment = _require_mapping(
        context.spec["expected_environment"],
        role="supervisor expected environment",
    )
    supervisor_identity = _validate_process_identity(
        launch["supervisor_process_identity"],
        role="launch supervisor process identity",
    )
    supervisor_command = context.supervisor_process_command_projection
    prearm = _validate_prearm_process_absence(
        launch["prearm_process_absence"],
        context=context,
    )
    prelaunch = launch["prelaunch_artifacts"]
    if not isinstance(prelaunch, list) or len(prelaunch) != len(context.expected_artifact_rules):
        raise OriginalConfirmatoryTerminalError("launch pre-artifact inventory length differs")
    for record_value, rule in zip(
        prelaunch,
        context.expected_artifact_rules,
        strict=True,
    ):
        record = _require_mapping(
            record_value,
            role="launch pre-artifact inventory record",
        )
        if (
            set(record) != {"role", "path", "exists"}
            or record["role"] != rule["role"]
            or record["path"] != rule["path"]
            or type(record["exists"]) is not bool
            or (rule["must_be_absent_before"] and record["exists"] is not False)
        ):
            raise OriginalConfirmatoryTerminalError("launch pre-artifact inventory differs")
    if (
        type(launch["schema_version"]) is not int
        or launch["schema_version"] != 3
        or launch["policy"] != authority.SUPERVISOR_V3_POLICY
        or launch["attempt_policy"] != _SUPERVISOR_ATTEMPT_POLICY
        or launch["job_id"] != context.job_id
        or launch["spec_sha256"] != context.spec_sha256
        or launch["command"] != context.spec["command"]
        or launch["command_sha256"] != context.scientific_command.command_sha256
        or launch["attempt_nonce"] != context.attempt_nonce
        or launch["attempt_count"] != 1
        or type(launch["attempt_count"]) is not int
        or launch["max_attempt_count"] != 1
        or type(launch["max_attempt_count"]) is not int
        or launch["automatic_retry_allowed"] is not False
        or launch["job_assignment_mode"] != "PROC_THREAD_ATTRIBUTE_JOB_LIST"
        or launch["handle_list_restricted"] is not True
        or launch["job_handle_inherited"] is not False
        or launch["supervisor_process_identity"] != supervisor_identity
        or not _same_resolved_path(
            supervisor_identity["program_path"],
            supervisor_command["expected_live_image_path"],
        )
        or supervisor_identity["program_sha256"] != supervisor_command["expected_live_image_sha256"]
        or supervisor_identity["command_sha256"] != supervisor_command["command_sha256"]
        or launch["main_timeout_ms"] != context.spec["main_timeout_ms"]
        or not _is_utc(launch["windows_boot_time_utc"])
        or not _is_utc(launch["created_at_utc"])
        or prearm["observed_at_utc"] > launch["created_at_utc"]
        or launch["supervisor_launcher_sha256"] != context.spec["supervisor_launcher_sha256"]
        or launch["expected_environment_envelope_sha256"] != expected_environment["envelope_sha256"]
        or launch["launch_environment_root_sha256"]
        != expected_environment["launch_environment_root_sha256"]
        or launch["process_environment_binding_sha256"] != binding["binding_sha256"]
        or launch["exact_supervisor_environment_sha256"]
        != binding["exact_supervisor_environment_sha256"]
        or launch["observed_supervisor_environment_sha256"]
        != binding["exact_supervisor_environment_sha256"]
        or launch["exact_environment_sha256"] != binding["exact_environment_sha256"]
        or launch["exact_integrity_verifier_environment_sha256"]
        != binding["exact_integrity_verifier_environment_sha256"]
        or launch["supervisor_environment_exact_match"] is not True
        or launch["capsule_lease_identity"] != context.spec["capsule_lease_identity"]
        or launch["capsule_lease_identity_root_sha256"]
        != context.spec["capsule_lease_identity_root_sha256"]
        or launch["capsule_ancestor_lease"] != context.spec["capsule_ancestor_lease"]
        or launch["capsule_ancestor_lease_root_sha256"]
        != context.spec["capsule_ancestor_lease_root_sha256"]
        or launch["preterminal_overlap_handshake_contract_sha256"]
        != context.overlap_contract.contract_sha256
        or launch["python_lease_identity"] != context.spec["python_lease_identity"]
        or launch["python_lease_identity_root_sha256"]
        != context.spec["python_lease_identity_root_sha256"]
        or launch["python_ancestor_lease"] != context.spec["python_ancestor_lease"]
        or launch["python_ancestor_lease_root_sha256"]
        != context.spec["python_ancestor_lease_root_sha256"]
        or launch["python_runtime_resolution_policy"]
        != context.spec["python_runtime_resolution_policy"]
        or launch["runtime_python_lease_identity"] != context.spec["runtime_python_lease_identity"]
        or launch["runtime_python_lease_identity_root_sha256"]
        != context.spec["runtime_python_lease_identity_root_sha256"]
        or launch["runtime_python_ancestor_lease"] != context.spec["runtime_python_ancestor_lease"]
        or launch["runtime_python_ancestor_lease_root_sha256"]
        != context.spec["runtime_python_ancestor_lease_root_sha256"]
        or launch["e_consumption_contract_sha256"]
        != context.e["e_consumption_contract"]["contract_sha256"]
    ):
        raise OriginalConfirmatoryTerminalError("launch intent differs from Q/E/spec")
    return launch


def _validate_process_started_payload(
    value: Any,
    *,
    context: _VerifiedContext,
    launch: Mapping[str, Any],
    launch_file_sha256: str,
) -> dict[str, Any]:
    started = _require_mapping(value, role="process-started payload")
    if set(started) != _PROCESS_STARTED_FIELDS:
        raise OriginalConfirmatoryTerminalError("process-started payload fields differ")
    binding = _require_mapping(
        context.spec["process_environment_binding"],
        role="supervisor process-environment binding",
    )
    expected_environment = _require_mapping(
        context.spec["expected_environment"],
        role="supervisor expected environment",
    )
    process_identity = _validate_process_identity(
        started["process_identity"],
        role="scientific child process identity",
    )
    expected_command_identity = {
        "program_path": context.spec["program_path"],
        "program_sha256": context.spec["program_sha256"],
        "command_sha256": context.scientific_command.command_sha256,
    }
    if (
        type(started["schema_version"]) is not int
        or started["schema_version"] != 3
        or started["policy"] != authority.SUPERVISOR_V3_POLICY
        or started["job_id"] != context.job_id
        or started["spec_sha256"] != context.spec_sha256
        or started["launch_intent_sha256"] != launch_file_sha256
        or started["attempt_nonce"] != context.attempt_nonce
        or any(
            process_identity[key] != expected for key, expected in expected_command_identity.items()
        )
        or started["process_identity"] != process_identity
        or started["job_assignment_mode"] != "PROC_THREAD_ATTRIBUTE_JOB_LIST"
        or started["atomic_job_assignment"] is not True
        or started["handle_list_restricted"] is not True
        or started["job_handle_inherited"] is not False
        or started["supervisor_process_identity"] != launch["supervisor_process_identity"]
        or started["main_timeout_ms"] != context.spec["main_timeout_ms"]
        or started["stdout_partial_path"] != str(context.job_dir / "stdout.partial")
        or started["stderr_partial_path"] != str(context.job_dir / "stderr.partial")
        or started["windows_boot_time_utc"] != launch["windows_boot_time_utc"]
        or not _is_utc(started["started_at_utc"])
        or started["started_at_utc"] < launch["created_at_utc"]
        or started["attempt_count"] != 1
        or type(started["attempt_count"]) is not int
        or started["automatic_retry_allowed"] is not False
        or started["expected_environment_envelope_sha256"]
        != expected_environment["envelope_sha256"]
        or started["launch_environment_root_sha256"]
        != expected_environment["launch_environment_root_sha256"]
        or started["process_environment_binding_sha256"] != binding["binding_sha256"]
        or started["exact_supervisor_environment_sha256"]
        != binding["exact_supervisor_environment_sha256"]
        or started["observed_supervisor_environment_sha256"]
        != binding["exact_supervisor_environment_sha256"]
        or started["exact_environment_sha256"] != binding["exact_environment_sha256"]
        or started["observed_child_environment_sha256"] != binding["exact_environment_sha256"]
        or started["exact_integrity_verifier_environment_sha256"]
        != binding["exact_integrity_verifier_environment_sha256"]
        or started["child_environment_exact_match"] is not True
        or started["child_environment_observation_method"] != "windows_peb_process_parameters_v1"
        or started["capsule_lease_identity_root_sha256"]
        != context.spec["capsule_lease_identity_root_sha256"]
        or started["capsule_ancestor_lease_root_sha256"]
        != context.spec["capsule_ancestor_lease_root_sha256"]
        or started["preterminal_overlap_handshake_contract_sha256"]
        != context.overlap_contract.contract_sha256
        or started["python_lease_identity_root_sha256"]
        != context.spec["python_lease_identity_root_sha256"]
        or started["python_ancestor_lease_root_sha256"]
        != context.spec["python_ancestor_lease_root_sha256"]
        or started["interpreter_leaf_handle_active"]
        is not (context.spec["python_lease_identity"] is not None)
        or started["interpreter_ancestor_handles_active"]
        is not (context.spec["python_ancestor_lease"] is not None)
        or started["python_runtime_resolution_policy"]
        != context.spec["python_runtime_resolution_policy"]
        or started["runtime_python_lease_identity_root_sha256"]
        != context.spec["runtime_python_lease_identity_root_sha256"]
        or started["runtime_python_ancestor_lease_root_sha256"]
        != context.spec["runtime_python_ancestor_lease_root_sha256"]
        or started["runtime_interpreter_leaf_handle_active"]
        is not (context.spec["runtime_python_lease_identity"] is not None)
        or started["runtime_interpreter_ancestor_handles_active"]
        is not (context.spec["runtime_python_ancestor_lease"] is not None)
        or started["e_consumption_contract_sha256"]
        != context.e["e_consumption_contract"]["contract_sha256"]
    ):
        raise OriginalConfirmatoryTerminalError("process-started payload differs from Q/E/spec")
    return started


def _open_validated_launch_evidence(
    context: _VerifiedContext,
) -> _ValidatedLaunchEvidence:
    launch_value = context.values.get("--launch-intent")
    started_value = context.values.get("--process-started")
    launch_path = (
        Path(launch_value) if launch_value is not None else context.job_dir / "launch_intent.json"
    )
    started_path = (
        Path(started_value)
        if started_value is not None
        else context.job_dir / "process_started.json"
    )
    launch, launch_payload, launch_sha = _read_required_receipt(
        launch_path,
        role="launch intent",
    )
    started: _HeldFile | None = None
    try:
        started, started_payload, _started_sha = _read_required_receipt(
            started_path,
            role="process started",
        )
        launch_payload = _validate_launch_intent_payload(
            launch_payload,
            context=context,
        )
        _validate_process_started_payload(
            started_payload,
            context=context,
            launch=launch_payload,
            launch_file_sha256=launch_sha,
        )
        evidence = _ValidatedLaunchEvidence(
            launch_intent=launch,
            process_started=started,
        )
        evidence.revalidate()
        started = None
        return evidence
    except BaseException:
        if started is not None:
            started.close()
        launch.close()
        raise


def _build_preterminal_pin(
    context: _VerifiedContext,
    *,
    consumed_e: _ConsumedEClaim,
    launch_intent_sha256: str,
    process_started_sha256: str,
) -> dict[str, Any]:
    consumed_e.revalidate()
    raw_artifacts = context.spec.get("expected_artifacts")
    raw_required_roles = context.spec.get("required_success_roles")
    if not isinstance(raw_artifacts, list) or not isinstance(
        raw_required_roles,
        list,
    ):
        raise OriginalConfirmatoryTerminalError("supervisor spec artifact declarations are invalid")
    if len(raw_artifacts) != len(context.expected_artifact_rules):
        raise OriginalConfirmatoryTerminalError(
            "supervisor artifact declarations changed after prevalidation"
        )
    evidence = [
        _inspect_expected_artifact(
            item,
            expected_rule=context.expected_artifact_rules[index],
        )
        for index, item in enumerate(raw_artifacts)
    ]
    roles = [item["role"] for item in evidence]
    required_roles = tuple(raw_required_roles)
    if (
        required_roles != _EXPECTED_ARTIFACT_ROLE_ORDER
        or tuple(roles) != _EXPECTED_ARTIFACT_ROLE_ORDER
    ):
        raise OriginalConfirmatoryTerminalError(
            "required success roles are not exact artifact roles"
        )
    artifact_root = _canonical_sha256(evidence)
    scientific_core = {
        "run_spec_file_sha256": context.spec_sha256,
        "canonical_spec_sha256": context.run_spec_payload["canonical_spec_sha256"],
        "launch_intent_file_sha256": launch_intent_sha256,
        "process_started_file_sha256": process_started_sha256,
        "e_consumption_claim_file_sha256": consumed_e.held.file_sha256,
        "e_consumption_claim_root_sha256": consumed_e.claim["claim_root_sha256"],
        "scientific_command_sha256": context.scientific_command.command_sha256,
        "required_success_roles": list(required_roles),
        "expected_artifact_evidence_root_sha256": artifact_root,
    }
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "policy": PRETERMINAL_PIN_POLICY,
        "status": "passed",
        "job_id": context.job_id,
        "attempt_id": context.attempt_id,
        "run_id": context.run_id,
        "execution_mode": context.execution_mode,
        "retry_of_run_id": context.retry_of_run_id,
        "attempt_nonce": context.attempt_nonce,
        "q_authority_root_sha256": context.q["q_authority_root_sha256"],
        "e_intent_path": str(context.e_file.path),
        "e_intent_file_sha256": context.e_file.file_sha256,
        "e_intent_core_sha256": context.e["intent_core_sha256"],
        "e_consumption_claim_path": str(consumed_e.held.path),
        "e_consumption_claim_file_sha256": consumed_e.held.file_sha256,
        "e_consumption_claim_root_sha256": consumed_e.claim["claim_root_sha256"],
        "capsule_contract_sha256": context.q["execution_capsule"]["contract_sha256"],
        "capsule_sha256": context.q["execution_capsule"]["sha256"],
        "capsule_internal_manifest_sha256": context.q["execution_capsule"][
            "internal_manifest_sha256"
        ],
        "capsule_mode": authority.CAPSULE_PRETERMINAL_MODE,
        "preterminal_pin_contract_sha256": context.preterminal_contract.contract_sha256,
        "preterminal_overlap_handshake_contract_sha256": (context.overlap_contract.contract_sha256),
        "run_spec_file_sha256": context.spec_sha256,
        "canonical_spec_sha256": context.run_spec_payload["canonical_spec_sha256"],
        "launch_intent_file_sha256": launch_intent_sha256,
        "process_started_file_sha256": process_started_sha256,
        "scientific_command_sha256": context.scientific_command.command_sha256,
        "preterminal_command_sha256": context.preterminal_command.command_sha256,
        "terminal_command_sha256": context.terminal_command.command_sha256,
        "observed_integrity_verifier_environment_sha256": (context.observed_environment_sha256),
        "required_success_roles": list(required_roles),
        "expected_artifact_evidence": evidence,
        "expected_artifact_evidence_root_sha256": artifact_root,
        "preterminal_scientific_core_sha256": _canonical_sha256(scientific_core),
        "semantic_outcome_read_scope": SEMANTIC_OUTCOME_READ_SCOPE,
        "outcome_values_read": False,
        "outcome_values_emitted": False,
        "outcome_values_used_for_selection_or_tuning": False,
        "training_or_model_selection_allowed": False,
        "scientific_publication_allowed": False,
        "automatic_retry_allowed": False,
        "created_at_utc": _utc_now(),
    }
    pin = {**unsigned, "evidence_root_sha256": _canonical_sha256(unsigned)}
    if set(pin) != _PRETERMINAL_PIN_FIELDS:
        raise OriginalConfirmatoryTerminalError("internal preterminal pin fields differ")
    consumed_e.revalidate()
    return pin


def _build_preterminal_ready(
    context: _VerifiedContext,
    *,
    pin: Mapping[str, Any],
    pin_identity: Mapping[str, Any],
) -> dict[str, Any]:
    ready = {
        "schema_version": 1,
        "policy": PRETERMINAL_READY_POLICY,
        "message_type": PRETERMINAL_READY_MESSAGE_TYPE,
        "handshake_policy": PRETERMINAL_HANDSHAKE_POLICY,
        "job_id": context.job_id,
        "attempt_id": context.attempt_id,
        "run_id": context.run_id,
        "execution_mode": context.execution_mode,
        "retry_of_run_id": context.retry_of_run_id,
        "attempt_nonce": context.attempt_nonce,
        "capsule_contract_sha256": context.q["execution_capsule"]["contract_sha256"],
        "capsule_sha256": context.q["execution_capsule"]["sha256"],
        "capsule_internal_manifest_sha256": context.q["execution_capsule"][
            "internal_manifest_sha256"
        ],
        "capsule_mode": authority.CAPSULE_PRETERMINAL_MODE,
        "preterminal_pin_contract_sha256": context.preterminal_contract.contract_sha256,
        "observed_integrity_verifier_environment_sha256": (context.observed_environment_sha256),
        "preterminal_pin_receipt": dict(pin_identity),
        "preterminal_pin_evidence_root_sha256": pin["evidence_root_sha256"],
        "preterminal_scientific_core_sha256": pin["preterminal_scientific_core_sha256"],
        "semantic_outcome_read_scope": SEMANTIC_OUTCOME_READ_SCOPE,
        "outcome_values_read": False,
        "outcome_values_emitted": False,
        "outcome_values_used_for_selection_or_tuning": False,
        "training_or_model_selection_allowed": False,
        "scientific_publication_allowed": False,
        "automatic_retry_allowed": False,
        "pin_handle_open": True,
        "pin_handle_share_access": ["FILE_SHARE_READ"],
        "awaiting_supervisor_ack": True,
    }
    if set(ready) != _PRETERMINAL_READY_FIELDS:
        raise OriginalConfirmatoryTerminalError("internal preterminal READY fields differ")
    return ready


def _read_stream_line(stream: Any, *, maximum_bytes: int) -> bytes:
    line = stream.buffer.readline(maximum_bytes + 1)
    if not isinstance(line, bytes) or len(line) > maximum_bytes:
        raise OriginalConfirmatoryTerminalError("handshake line exceeds its bound")
    return line


def _stable_physical_identity_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = _require_mapping(value, role="physical identity stable projection input")
    if set(raw) != _CLAIM_PHYSICAL_IDENTITY_FIELDS:
        raise OriginalConfirmatoryTerminalError("physical identity fields differ")
    projection = {field: raw[field] for field in _STABLE_PHYSICAL_IDENTITY_FIELD_ORDER}
    if set(projection) != _STABLE_PHYSICAL_IDENTITY_FIELDS:
        raise OriginalConfirmatoryTerminalError("physical identity stable projection fields differ")
    return projection


def _canonical_preterminal_supervisor_observer_identity(
    value: Any,
) -> dict[str, Any]:
    raw = _require_mapping(value, role="supervisor P observer identity")
    if (
        set(raw) != _CLAIM_PHYSICAL_IDENTITY_FIELDS
        or raw.get("policy") != PRETERMINAL_SUPERVISOR_OBSERVER_IDENTITY_POLICY
        or raw.get("role") != "preterminal-pin"
        or raw.get("share_access") != ["FILE_SHARE_READ", "FILE_SHARE_WRITE"]
    ):
        raise OriginalConfirmatoryTerminalError("supervisor P observer identity fields differ")
    authority.canonical_original_confirmatory_physical_file_identity(
        {
            **raw,
            "policy": authority.NO_FOLLOW_PHYSICAL_FILE_IDENTITY_POLICY,
            "share_access": ["FILE_SHARE_READ"],
        },
        allowed_roles=("preterminal-pin",),
    )
    return raw


def _validate_preterminal_ack(
    value: Mapping[str, Any],
    *,
    context: _VerifiedContext,
    ready: Mapping[str, Any],
    ready_line: bytes,
    pin_identity: Mapping[str, Any],
) -> dict[str, Any]:
    ack = dict(value)
    if set(ack) != _PRETERMINAL_ACK_FIELDS:
        raise OriginalConfirmatoryTerminalError("preterminal ACK fields differ")
    ready_sha = _sha256_bytes(ready_line)
    canonical_child = authority.canonical_original_confirmatory_physical_file_identity(
        ack["child_reported_pin_identity"],
        allowed_roles=("preterminal-pin",),
    ).as_dict()
    canonical_expected = authority.canonical_original_confirmatory_physical_file_identity(
        pin_identity,
        allowed_roles=("preterminal-pin",),
    ).as_dict()
    supervisor_observer = _canonical_preterminal_supervisor_observer_identity(
        ack["supervisor_opened_pin_identity"]
    )
    stable_projection = _stable_physical_identity_projection(canonical_expected)
    if (
        ack["schema_version"] != 1
        or ack["policy"] != PRETERMINAL_ACK_POLICY
        or ack["message_type"] != PRETERMINAL_ACK_MESSAGE_TYPE
        or any(
            ack[key] != ready[key]
            for key in (
                "job_id",
                "attempt_id",
                "run_id",
                "execution_mode",
                "retry_of_run_id",
                "attempt_nonce",
            )
        )
        or ack["preterminal_pin_contract_sha256"] != context.preterminal_contract.contract_sha256
        or ack["preterminal_overlap_handshake_contract_sha256"]
        != context.overlap_contract.contract_sha256
        or ack["ready_line_sha256"] != ready_sha
        or canonical_child != canonical_expected
        or ack["child_reported_pin_identity"] != canonical_child
        or ack["child_reported_pin_identity_root_sha256"] != _canonical_sha256(canonical_child)
        or ack["supervisor_opened_pin_identity"] != supervisor_observer
        or ack["supervisor_opened_pin_identity_root_sha256"]
        != _canonical_sha256(supervisor_observer)
        or _stable_physical_identity_projection(supervisor_observer) != stable_projection
        or ack["stable_physical_identity_projection"] != stable_projection
        or ack["stable_physical_identity_root_sha256"] != _canonical_sha256(stable_projection)
        or ack["identity_exact_match"] is not True
        or ack["pin_handle_overlap_verified"] is not True
        or ack["automatic_retry_allowed"] is not False
        or not _is_utc(ack["acknowledged_at_utc"])
    ):
        raise OriginalConfirmatoryTerminalError("preterminal ACK violates its exact policy")
    return ack


def _run_preterminal_handshake(
    context: _VerifiedContext,
    *,
    pin: Mapping[str, Any],
    stdin: Any,
    stdout: Any,
) -> None:
    path = context.preterminal_contract.preterminal_pin_receipt_path
    descriptor = _create_new_readonly_descriptor(path)
    try:
        payload = _canonical_bytes(pin)
        _write_same_handle(
            descriptor,
            payload,
            maximum_bytes=context.preterminal_contract.preterminal_pin_receipt_max_bytes,
        )
        pin_identity = _physical_identity(
            descriptor,
            path=path,
            role="preterminal-pin",
            maximum_bytes=context.preterminal_contract.preterminal_pin_receipt_max_bytes,
        )
        ready = _build_preterminal_ready(
            context,
            pin=pin,
            pin_identity=pin_identity,
        )
        ready_line = _canonical_bytes(ready)
        if len(ready_line) > context.overlap_contract.ready_line_max_bytes:
            raise OriginalConfirmatoryTerminalError("preterminal READY exceeds its bound")
        stdout.buffer.write(ready_line)
        stdout.buffer.flush()
        ack_line = _read_stream_line(
            stdin,
            maximum_bytes=context.overlap_contract.ack_line_max_bytes,
        )
        ack = _decode_canonical_line(
            ack_line,
            role="preterminal ACK",
            maximum_bytes=context.overlap_contract.ack_line_max_bytes,
        )
        _validate_preterminal_ack(
            ack,
            context=context,
            ready=ready,
            ready_line=ready_line,
            pin_identity=pin_identity,
        )
        if (
            _read_descriptor(
                descriptor,
                maximum_bytes=context.preterminal_contract.preterminal_pin_receipt_max_bytes,
            )
            != payload
            or _physical_identity(
                descriptor,
                path=path,
                role="preterminal-pin",
                maximum_bytes=context.preterminal_contract.preterminal_pin_receipt_max_bytes,
            )
            != pin_identity
        ):
            raise OriginalConfirmatoryTerminalError("preterminal pin changed before ACK completion")
    finally:
        os.close(descriptor)


def _verify_original_confirmatory_preterminal_from_canonical_tail(
    canonical_tail: tuple[str, ...],
) -> int:
    """Fixed ``verify-preterminal`` handler selected by the capsule dispatcher."""

    context: _VerifiedContext | None = None
    consumed_e: _ConsumedEClaim | None = None
    try:
        context, consumed_e = _load_context_then_take_consumed_e(
            authority.CAPSULE_PRETERMINAL_MODE,
            canonical_tail,
        )
        pin = _build_preterminal_pin(
            context,
            consumed_e=consumed_e,
            launch_intent_sha256=consumed_e.launch_intent.file_sha256,
            process_started_sha256=consumed_e.process_started.file_sha256,
        )
        _run_preterminal_handshake(
            context,
            pin=pin,
            stdin=sys.stdin,
            stdout=sys.stdout,
        )
        consumed_e.revalidate()
        context.q_file.revalidate()
        context.e_file.revalidate()
        context.run_spec_file.revalidate()
        return 0
    except BaseException:
        return TERMINAL_HANDLER_STOP_EXIT_CODE
    finally:
        if consumed_e is not None:
            try:
                consumed_e.close()
            except BaseException:
                return TERMINAL_HANDLER_STOP_EXIT_CODE
        if context is not None:
            try:
                context.close()
            except BaseException:
                return TERMINAL_HANDLER_STOP_EXIT_CODE


class _WindowsOverlapped(ctypes.Structure):
    _fields_ = [
        ("internal", ctypes.c_size_t),
        ("internal_high", ctypes.c_size_t),
        ("offset", ctypes.c_uint32),
        ("offset_high", ctypes.c_uint32),
        ("event", ctypes.c_void_p),
    ]


class _DuplexPipeClient:
    """One overlapped message-mode client with one post-CLAIM deadline."""

    def __init__(
        self,
        pipe_name: str,
        *,
        outbound_maximum_message_bytes: int,
        inbound_maximum_message_bytes: int,
        custody_exchange_timeout_ms: int,
    ) -> None:
        if os.name != "nt":
            raise OriginalConfirmatoryTerminalError("postwake custody pipe is Windows-only")
        if (
            outbound_maximum_message_bytes <= 0
            or inbound_maximum_message_bytes <= 0
            or outbound_maximum_message_bytes > _MAX_PIPE_MESSAGE_BYTES
            or inbound_maximum_message_bytes > _MAX_PIPE_MESSAGE_BYTES
            or custody_exchange_timeout_ms <= 0
        ):
            raise OriginalConfirmatoryTerminalError("custody pipe bounds are invalid")
        self.outbound_maximum_message_bytes = outbound_maximum_message_bytes
        self.inbound_maximum_message_bytes = inbound_maximum_message_bytes
        self.exchange_timeout_ms = custody_exchange_timeout_ms
        self.deadline: float | None = None
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        create_file.restype = ctypes.c_void_p
        handle = create_file(
            pipe_name,
            GENERIC_READ | GENERIC_WRITE,
            0,
            None,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
            None,
        )
        if handle == INVALID_HANDLE_VALUE:
            raise ctypes.WinError(ctypes.get_last_error())
        self.handle = cast(int, handle)
        mode = ctypes.c_uint32(0x00000002)
        if not kernel32.SetNamedPipeHandleState(
            ctypes.c_void_p(self.handle),
            ctypes.byref(mode),
            None,
            None,
        ):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(ctypes.c_void_p(self.handle))
            self.handle = 0
            raise ctypes.WinError(error)

    def _arm_exchange_deadline(self) -> None:
        if self.deadline is not None:
            raise OriginalConfirmatoryTerminalError(
                "postwake custody exchange deadline was armed more than once"
            )
        self.deadline = time.monotonic() + self.exchange_timeout_ms / 1000

    def _remaining_timeout_ms(self) -> int:
        if self.deadline is None:
            raise OriginalConfirmatoryTerminalError(
                "postwake custody exchange deadline is not armed"
            )
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise OriginalConfirmatoryTerminalError("postwake custody shared deadline expired")
        milliseconds = int(remaining * 1000)
        if milliseconds / 1000 < remaining:
            milliseconds += 1
        return max(milliseconds, 1)

    def require_deadline_active(self) -> None:
        """Fail closed if the local shared-handshake deadline has elapsed."""

        self._remaining_timeout_ms()

    def _cancel_and_drain(
        self,
        kernel32: Any,
        overlapped: _WindowsOverlapped,
    ) -> None:
        cancel_io = kernel32.CancelIoEx
        cancel_io.argtypes = [ctypes.c_void_p, ctypes.POINTER(_WindowsOverlapped)]
        cancel_io.restype = ctypes.c_int
        if not cancel_io(
            ctypes.c_void_p(self.handle),
            ctypes.byref(overlapped),
        ):
            error = ctypes.get_last_error()
            if error != ERROR_NOT_FOUND:
                raise ctypes.WinError(error)
        transferred = ctypes.c_uint32()
        get_result = kernel32.GetOverlappedResult
        get_result.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_WindowsOverlapped),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_int,
        ]
        get_result.restype = ctypes.c_int
        if not get_result(
            ctypes.c_void_p(self.handle),
            ctypes.byref(overlapped),
            ctypes.byref(transferred),
            True,
        ):
            error = ctypes.get_last_error()
            if error not in {ERROR_OPERATION_ABORTED, ERROR_NOT_FOUND}:
                raise ctypes.WinError(error)

    def _overlapped_transfer(
        self,
        *,
        buffer: Any,
        size_bytes: int,
        write: bool,
    ) -> int:
        self._remaining_timeout_ms()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_event = kernel32.CreateEventW
        create_event.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_wchar_p,
        ]
        create_event.restype = ctypes.c_void_p
        event = create_event(None, True, False, None)
        if not event:
            raise ctypes.WinError(ctypes.get_last_error())
        overlapped = _WindowsOverlapped()
        overlapped.event = event
        transferred = ctypes.c_uint32()
        try:
            operation = kernel32.WriteFile if write else kernel32.ReadFile
            operation.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(_WindowsOverlapped),
            ]
            operation.restype = ctypes.c_int
            completed = operation(
                ctypes.c_void_p(self.handle),
                buffer,
                size_bytes,
                ctypes.byref(transferred),
                ctypes.byref(overlapped),
            )
            if completed:
                return int(transferred.value)
            error = ctypes.get_last_error()
            if not write and error == ERROR_MORE_DATA:
                raise OriginalConfirmatoryTerminalError(
                    "custody pipe inbound message exceeds bound"
                )
            if error != ERROR_IO_PENDING:
                raise ctypes.WinError(error)
            try:
                remaining_timeout_ms = self._remaining_timeout_ms()
            except BaseException:
                self._cancel_and_drain(kernel32, overlapped)
                raise
            wait = kernel32.WaitForSingleObject(
                ctypes.c_void_p(event),
                remaining_timeout_ms,
            )
            if wait == WAIT_TIMEOUT:
                self._cancel_and_drain(kernel32, overlapped)
                raise OriginalConfirmatoryTerminalError("postwake custody shared deadline expired")
            if wait != WAIT_OBJECT_0:
                self._cancel_and_drain(kernel32, overlapped)
                raise ctypes.WinError(ctypes.get_last_error())
            get_result = kernel32.GetOverlappedResult
            get_result.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_WindowsOverlapped),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_int,
            ]
            get_result.restype = ctypes.c_int
            if not get_result(
                ctypes.c_void_p(self.handle),
                ctypes.byref(overlapped),
                ctypes.byref(transferred),
                False,
            ):
                error = ctypes.get_last_error()
                if not write and error == ERROR_MORE_DATA:
                    raise OriginalConfirmatoryTerminalError(
                        "custody pipe inbound message exceeds bound"
                    )
                raise ctypes.WinError(error)
            return int(transferred.value)
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(event))

    def close(self) -> None:
        if self.handle:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            if not kernel32.CloseHandle(ctypes.c_void_p(self.handle)):
                raise ctypes.WinError(ctypes.get_last_error())
            self.handle = 0

    def server_pid(self) -> int:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        pid = ctypes.c_uint32()
        if not kernel32.GetNamedPipeServerProcessId(
            ctypes.c_void_p(self.handle),
            ctypes.byref(pid),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if pid.value <= 0:
            raise OriginalConfirmatoryTerminalError("custody pipe server PID is invalid")
        return int(pid.value)

    def send(self, value: Mapping[str, Any]) -> bytes:
        payload = _canonical_bytes(value)
        if len(payload) > self.outbound_maximum_message_bytes:
            raise OriginalConfirmatoryTerminalError("custody pipe outbound message exceeds bound")
        if self.deadline is None:
            # The first outbound message is CLAIM_READY.  Resume-to-arrival is
            # bounded independently by the supervisor; only CLAIM-to-FINAL_ACK
            # consumes this client's shared custody-exchange deadline.
            self._arm_exchange_deadline()
        buffer = ctypes.create_string_buffer(payload)
        written = self._overlapped_transfer(
            buffer=buffer,
            size_bytes=len(payload),
            write=True,
        )
        if written != len(payload):
            raise OriginalConfirmatoryTerminalError("custody pipe short write")
        return payload

    def receive(self) -> tuple[dict[str, Any], bytes]:
        buffer = ctypes.create_string_buffer(self.inbound_maximum_message_bytes + 1)
        read = self._overlapped_transfer(
            buffer=buffer,
            size_bytes=len(buffer),
            write=False,
        )
        payload = bytes(buffer.raw[:read])
        value = _decode_canonical_line(
            payload,
            role="custody pipe inbound message",
            maximum_bytes=self.inbound_maximum_message_bytes,
        )
        return value, payload


def _filetime_iso(value: int) -> str:
    unix_100ns = value - 116444736000000000
    seconds, remainder = divmod(unix_100ns, 10_000_000)
    instant = datetime.fromtimestamp(seconds + remainder / 10_000_000, tz=UTC)
    return instant.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _open_process(pid: int, *, access: int) -> int:
    if os.name != "nt":
        raise OriginalConfirmatoryTerminalError("process custody is Windows-only")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    open_process.restype = ctypes.c_void_p
    handle = open_process(access, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    return cast(int, handle)


def _close_native_handle(handle: int) -> None:
    if not handle:
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel32.CloseHandle(ctypes.c_void_p(handle)):
        raise ctypes.WinError(ctypes.get_last_error())


def _process_identity(
    process_handle: int,
    *,
    pid: int,
    command_sha256: str,
) -> dict[str, Any]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", ctypes.c_uint32),
            ("dwHighDateTime", ctypes.c_uint32),
        ]

    creation = FILETIME()
    exit_time = FILETIME()
    kernel_time = FILETIME()
    user_time = FILETIME()
    if not kernel32.GetProcessTimes(
        ctypes.c_void_p(process_handle),
        ctypes.byref(creation),
        ctypes.byref(exit_time),
        ctypes.byref(kernel_time),
        ctypes.byref(user_time),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    creation_value = (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)
    capacity = ctypes.c_uint32(32768)
    image_buffer = ctypes.create_unicode_buffer(capacity.value)
    if not kernel32.QueryFullProcessImageNameW(
        ctypes.c_void_p(process_handle),
        0,
        image_buffer,
        ctypes.byref(capacity),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    image = Path(image_buffer.value)
    image_held = _open_held_file(
        image,
        role="process image",
        maximum_bytes=max(_MAX_CONTROL_BYTES, image.stat().st_size),
    )
    try:
        identity = {
            "pid": pid,
            "creation_time_100ns": creation_value,
            "creation_time_utc": _filetime_iso(creation_value),
            "program_path": str(image),
            "program_sha256": image_held.file_sha256,
            "command_sha256": _require_sha256(
                command_sha256,
                role="process command",
            ),
        }
        if set(identity) != _PROCESS_IDENTITY_FIELDS:
            raise OriginalConfirmatoryTerminalError("internal process identity fields differ")
        return identity
    finally:
        image_held.close()


def _read_process_memory(
    process_handle: int,
    address: int,
    size: int,
) -> bytes:
    if os.name != "nt" or process_handle <= 0 or address <= 0 or size <= 0 or size > 64 * 1024:
        raise OriginalConfirmatoryTerminalError("remote process memory read bounds are invalid")
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    read_process_memory = kernel32.ReadProcessMemory
    read_process_memory.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    read_process_memory.restype = wintypes.BOOL
    buffer = ctypes.create_string_buffer(size)
    received = ctypes.c_size_t()
    if not read_process_memory(
        wintypes.HANDLE(process_handle),
        ctypes.c_void_p(address),
        buffer,
        size,
        ctypes.byref(received),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if received.value != size:
        raise OriginalConfirmatoryTerminalError(
            "remote process memory read returned a short result"
        )
    return bytes(buffer.raw)


def _read_remote_pointer(process_handle: int, address: int) -> int:
    payload = _read_process_memory(
        process_handle,
        address,
        ctypes.sizeof(ctypes.c_void_p),
    )
    return int.from_bytes(payload, byteorder="little", signed=False)


def _query_process_basic_information(process_handle: int) -> tuple[int, int]:
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        raise OriginalConfirmatoryTerminalError(
            "remote PEB command observation requires a 64-bit process"
        )
    from ctypes import wintypes

    class _ProcessBasicInformation(ctypes.Structure):
        _fields_ = [
            ("reserved1", ctypes.c_void_p),
            ("peb_base_address", ctypes.c_void_p),
            ("reserved2", ctypes.c_void_p * 2),
            ("unique_process_id", ctypes.c_size_t),
            ("inherited_from_unique_process_id", ctypes.c_size_t),
        ]

    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    query = ntdll.NtQueryInformationProcess
    query.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.ULONG,
        ctypes.POINTER(wintypes.ULONG),
    ]
    query.restype = ctypes.c_long
    information = _ProcessBasicInformation()
    returned = wintypes.ULONG()
    status = int(
        query(
            wintypes.HANDLE(process_handle),
            0,
            ctypes.byref(information),
            ctypes.sizeof(information),
            ctypes.byref(returned),
        )
    )
    if status != 0 or not information.peb_base_address:
        raise OriginalConfirmatoryTerminalError(
            "NtQueryInformationProcess failed for remote PEB command observation"
        )
    if returned.value != ctypes.sizeof(information):
        raise OriginalConfirmatoryTerminalError(
            "NtQueryInformationProcess returned an unexpected PEB identity size"
        )
    peb_base_address = cast(int, information.peb_base_address)
    parent_pid = int(information.inherited_from_unique_process_id)
    if peb_base_address <= 0 or parent_pid <= 0:
        raise OriginalConfirmatoryTerminalError("remote process basic information is incomplete")
    return peb_base_address, parent_pid


def _remote_process_parameters_address(process_handle: int) -> int:
    peb_base_address, _parent_pid = _query_process_basic_information(process_handle)
    parameters = _read_remote_pointer(
        process_handle,
        peb_base_address + 0x20,
    )
    if parameters <= 0:
        raise OriginalConfirmatoryTerminalError("remote PEB has no process parameters")
    return parameters


def _running_process_parent_pid(process_handle: int) -> int:
    """Return the kernel-recorded creator PID for one exact live process."""

    _peb_base_address, parent_pid = _query_process_basic_information(process_handle)
    return parent_pid


def _read_remote_unicode_string(
    process_handle: int,
    address: int,
) -> str:
    header = _read_process_memory(process_handle, address, 16)
    length = int.from_bytes(header[0:2], byteorder="little")
    maximum = int.from_bytes(header[2:4], byteorder="little")
    pointer = int.from_bytes(header[8:16], byteorder="little")
    if (
        length % 2
        or maximum % 2
        or length > maximum
        or length > 64 * 1024
        or (length and pointer <= 0)
    ):
        raise OriginalConfirmatoryTerminalError("remote process UNICODE_STRING is invalid")
    if not length:
        return ""
    try:
        return _read_process_memory(
            process_handle,
            pointer,
            length,
        ).decode("utf-16-le")
    except UnicodeDecodeError as exc:
        raise OriginalConfirmatoryTerminalError(
            "remote process UNICODE_STRING is not UTF-16LE"
        ) from exc


def _windows_command_line_argv(command_line: str) -> list[str]:
    if os.name != "nt" or not command_line or "\x00" in command_line:
        raise OriginalConfirmatoryTerminalError("remote process command line is invalid")
    from ctypes import wintypes

    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    parse = shell32.CommandLineToArgvW
    parse.argtypes = [
        wintypes.LPCWSTR,
        ctypes.POINTER(ctypes.c_int),
    ]
    parse.restype = ctypes.POINTER(wintypes.LPWSTR)
    count = ctypes.c_int()
    parsed = parse(command_line, ctypes.byref(count))
    if not parsed:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if count.value <= 0:
            raise OriginalConfirmatoryTerminalError(
                "remote process command line parsed to no arguments"
            )
        return [str(parsed[index]) for index in range(count.value)]
    finally:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        local_free = kernel32.LocalFree
        local_free.argtypes = [ctypes.c_void_p]
        local_free.restype = ctypes.c_void_p
        if local_free(ctypes.cast(parsed, ctypes.c_void_p)):
            raise OriginalConfirmatoryTerminalError(
                "CommandLineToArgvW result could not be released"
            )


def _running_process_command_view(process_handle: int) -> dict[str, Any]:
    """Read argv/cwd from the exact live process PEB without trusting a receipt."""

    parameters = _remote_process_parameters_address(process_handle)
    cwd = _read_remote_unicode_string(
        process_handle,
        parameters + 0x38,
    )
    command_line = _read_remote_unicode_string(
        process_handle,
        parameters + 0x70,
    )
    if not cwd or not command_line:
        raise OriginalConfirmatoryTerminalError("remote process command view is incomplete")
    resolved_cwd = Path(cwd).resolve(strict=True)
    argv = _windows_command_line_argv(command_line)
    return {
        "argv": argv,
        "cwd": str(resolved_cwd),
        "observation_method": "windows_peb_process_parameters_v1",
    }


def _require_live_process_handle(process_handle: int, *, role: str) -> None:
    if os.name != "nt" or process_handle <= 0:
        raise OriginalConfirmatoryTerminalError(f"{role} process handle is invalid")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    wait = kernel32.WaitForSingleObject(
        ctypes.c_void_p(process_handle),
        0,
    )
    if wait != WAIT_TIMEOUT:
        if wait == WAIT_OBJECT_0:
            raise OriginalConfirmatoryTerminalError(f"{role} process already exited")
        raise ctypes.WinError(ctypes.get_last_error())


def _validate_live_process_command_view(
    value: Any,
    *,
    expected_argv: list[str],
    expected_cwd: str | Path,
    role: str,
) -> dict[str, Any]:
    raw = _require_mapping(value, role=f"{role} live command view")
    canonical_cwd = str(
        Path(
            authority._absolute_path(
                str(expected_cwd),
                role=f"{role} expected cwd",
            )
        ).resolve(strict=True)
    )
    if (
        set(raw) != {"argv", "cwd", "observation_method"}
        or not isinstance(raw["argv"], list)
        or not all(isinstance(item, str) and item and "\x00" not in item for item in raw["argv"])
        or raw["argv"] != expected_argv
        or raw["cwd"] != canonical_cwd
        or raw["observation_method"] != "windows_peb_process_parameters_v1"
    ):
        raise OriginalConfirmatoryTerminalError(
            f"{role} live PEB command/cwd differs from sealed authority"
        )
    return raw


def _single_argv_flag_value(
    argv: list[str],
    *,
    flag: str,
    role: str,
) -> str:
    indexes = [index for index, item in enumerate(argv) if item == flag]
    if (
        len(indexes) != 1
        or indexes[0] + 1 >= len(argv)
        or not isinstance(argv[indexes[0] + 1], str)
    ):
        raise OriginalConfirmatoryTerminalError(f"{role} does not contain one exact {flag} value")
    return argv[indexes[0] + 1]


def _same_resolved_path(left: Any, right: Any) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    try:
        return Path(left).resolve(strict=True) == Path(right).resolve(strict=True)
    except (OSError, RuntimeError):
        return False


def _establish_terminal_process_ancestry(
    *,
    terminal_command: authority.OriginalConfirmatoryCapsuleCommand,
    launcher_release: Mapping[str, Any],
    launcher_projection: Mapping[str, Any],
    capsule: Mapping[str, Any],
    supervisor_spec_sha256: str,
    e_intent_file_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    int,
    dict[str, Any],
    int,
    dict[str, Any],
    _HeldFile,
]:
    """Validate and retain the live launcher -> venv redirector -> C chain."""

    current_handle = 0
    redirector_handle = 0
    launcher_handle = 0
    launcher_source_file: _HeldFile | None = None
    try:
        current_handle = _open_process(
            os.getpid(),
            access=PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ | SYNCHRONIZE,
        )
        _require_live_process_handle(current_handle, role="terminal runtime child")
        redirector_pid = _running_process_parent_pid(current_handle)
        redirector_handle = _open_process(
            redirector_pid,
            access=PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ | SYNCHRONIZE,
        )
        _require_live_process_handle(
            redirector_handle,
            role="terminal immediate venv redirector",
        )
        launcher_pid = _running_process_parent_pid(redirector_handle)
        launcher_handle = _open_process(
            launcher_pid,
            access=PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ | SYNCHRONIZE,
        )
        _require_live_process_handle(
            launcher_handle,
            role="terminal-client launcher",
        )
        if (
            _running_process_parent_pid(current_handle) != redirector_pid
            or _running_process_parent_pid(redirector_handle) != launcher_pid
        ):
            raise OriginalConfirmatoryTerminalError(
                "live launcher/redirector/runtime-child ancestry changed"
            )

        launcher_view = _running_process_command_view(launcher_handle)
        terminal_receipt_sha256 = _require_sha256(
            _single_argv_flag_value(
                cast(list[str], launcher_view["argv"]),
                flag="--terminal-receipt-sha256",
                role="terminal-client launcher argv",
            ),
            role="terminal-client launcher terminal receipt",
        )
        launcher_command = authority.build_original_confirmatory_terminal_client_launcher_command(
            launcher_projection=launcher_projection,
            launcher_release=launcher_release,
            capsule=capsule,
            supervisor_spec_sha256=supervisor_spec_sha256,
            e_intent_file_sha256=e_intent_file_sha256,
            terminal_receipt_sha256=terminal_receipt_sha256,
            verify_terminal_command=terminal_command,
        )
        _validate_live_process_command_view(
            launcher_view,
            expected_argv=cast(list[str], launcher_command["process_argv"]),
            expected_cwd=cast(str, launcher_command["cwd"]),
            role="terminal-client launcher",
        )
        _validate_live_process_command_view(
            _running_process_command_view(redirector_handle),
            expected_argv=list(terminal_command.argv),
            expected_cwd=terminal_command.cwd,
            role="terminal immediate venv redirector",
        )
        runtime_argv = [
            cast(str, launcher_projection["verify_terminal_runtime_child_program_path"]),
            *terminal_command.argv[1:],
        ]
        _validate_live_process_command_view(
            _running_process_command_view(current_handle),
            expected_argv=list(runtime_argv),
            expected_cwd=terminal_command.cwd,
            role="terminal runtime child",
        )

        runtime_identity = _process_identity(
            current_handle,
            pid=os.getpid(),
            command_sha256=terminal_command.command_sha256,
        )
        redirector_identity = _process_identity(
            redirector_handle,
            pid=redirector_pid,
            command_sha256=terminal_command.command_sha256,
        )
        launcher_identity = _process_identity(
            launcher_handle,
            pid=launcher_pid,
            command_sha256=cast(str, launcher_command["command_sha256"]),
        )
        if (
            launcher_release["verify_terminal_child_launch_topology"]
            != "launcher_base_direct_to_venv_redirector_to_runtime_child_v1"
            or launcher_projection["verify_terminal_child_launch_topology"]
            != launcher_release["verify_terminal_child_launch_topology"]
            or not _same_resolved_path(
                runtime_identity["program_path"],
                launcher_projection["verify_terminal_runtime_child_program_path"],
            )
            or runtime_identity["program_sha256"]
            != launcher_projection["verify_terminal_runtime_child_program_sha256"]
            or not _same_resolved_path(
                redirector_identity["program_path"],
                launcher_projection["verify_terminal_immediate_redirector_program_path"],
            )
            or redirector_identity["program_sha256"]
            != launcher_projection["verify_terminal_immediate_redirector_program_sha256"]
            or not _same_resolved_path(
                launcher_identity["program_path"],
                launcher_command["program_path"],
            )
            or launcher_identity["program_sha256"] != launcher_command["program_sha256"]
        ):
            raise OriginalConfirmatoryTerminalError(
                "live launcher/redirector/runtime-child identities differ from authority"
            )

        launcher_source_path = Path(cast(str, launcher_release["source_path"]))
        launcher_source_file = _open_held_file(
            launcher_source_path,
            role="terminal-client launcher source",
            maximum_bytes=cast(int, launcher_release["source_size_bytes"]),
        )
        source_identity = _physical_identity(
            launcher_source_file.descriptor,
            path=launcher_source_path,
            role="terminal-client-launcher",
            maximum_bytes=cast(int, launcher_release["source_size_bytes"]),
        )
        if (
            launcher_source_file.file_sha256 != launcher_release["source_sha256"]
            or source_identity != launcher_release["source_physical_identity"]
            or _canonical_sha256(source_identity)
            != launcher_release["source_physical_identity_root_sha256"]
        ):
            raise OriginalConfirmatoryTerminalError(
                "live terminal-client launcher source differs from Q"
            )

        _close_native_handle(current_handle)
        current_handle = 0
        retained_source = launcher_source_file
        launcher_source_file = None
        retained_redirector_handle = redirector_handle
        redirector_handle = 0
        retained_launcher_handle = launcher_handle
        launcher_handle = 0
        return (
            runtime_identity,
            redirector_identity,
            retained_redirector_handle,
            launcher_identity,
            retained_launcher_handle,
            launcher_command,
            retained_source,
        )
    except authority.OriginalConfirmatoryCapsuleAuthorityError as exc:
        raise OriginalConfirmatoryTerminalError(
            "terminal-client launcher command differs from sealed authority"
        ) from exc
    finally:
        cleanup_errors: list[BaseException] = []
        if current_handle:
            try:
                _close_native_handle(current_handle)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if launcher_source_file is not None:
            try:
                launcher_source_file.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if launcher_handle:
            try:
                _close_native_handle(launcher_handle)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if redirector_handle:
            try:
                _close_native_handle(redirector_handle)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors and sys.exc_info()[0] is None:
            raise OriginalConfirmatoryTerminalError(
                "terminal launcher ancestry cleanup failed"
            ) from cleanup_errors[0]


def _revalidate_terminal_process_ancestry(context: _VerifiedContext) -> None:
    if context.mode != authority.CAPSULE_TERMINAL_MODE:
        return
    if (
        context.terminal_launcher_command is None
        or context.terminal_runtime_child_process_identity is None
        or context.immediate_venv_redirector_process_identity is None
        or context.terminal_client_launcher_process_identity is None
        or context.immediate_venv_redirector_process_handle <= 0
        or context.terminal_client_launcher_process_handle <= 0
        or context.launcher_source_file is None
    ):
        raise OriginalConfirmatoryTerminalError("terminal launcher ancestry custody is incomplete")
    current_handle = _open_process(
        os.getpid(),
        access=PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ | SYNCHRONIZE,
    )
    try:
        _require_live_process_handle(current_handle, role="terminal runtime child")
        _require_live_process_handle(
            context.immediate_venv_redirector_process_handle,
            role="terminal immediate venv redirector",
        )
        _require_live_process_handle(
            context.terminal_client_launcher_process_handle,
            role="terminal-client launcher",
        )
        if (
            _running_process_parent_pid(current_handle)
            != context.immediate_venv_redirector_process_identity["pid"]
            or _running_process_parent_pid(context.immediate_venv_redirector_process_handle)
            != context.terminal_client_launcher_process_identity["pid"]
            or _process_identity(
                current_handle,
                pid=os.getpid(),
                command_sha256=context.terminal_command.command_sha256,
            )
            != context.terminal_runtime_child_process_identity
            or _process_identity(
                context.immediate_venv_redirector_process_handle,
                pid=cast(
                    int,
                    context.immediate_venv_redirector_process_identity["pid"],
                ),
                command_sha256=context.terminal_command.command_sha256,
            )
            != context.immediate_venv_redirector_process_identity
            or _process_identity(
                context.terminal_client_launcher_process_handle,
                pid=cast(
                    int,
                    context.terminal_client_launcher_process_identity["pid"],
                ),
                command_sha256=cast(
                    str,
                    context.terminal_launcher_command["command_sha256"],
                ),
            )
            != context.terminal_client_launcher_process_identity
        ):
            raise OriginalConfirmatoryTerminalError("terminal launcher ancestry identity changed")
        _validate_live_process_command_view(
            _running_process_command_view(current_handle),
            expected_argv=[
                cast(
                    str,
                    context.terminal_launcher_projection[
                        "verify_terminal_runtime_child_program_path"
                    ],
                ),
                *context.terminal_command.argv[1:],
            ],
            expected_cwd=context.terminal_command.cwd,
            role="terminal runtime child",
        )
        _validate_live_process_command_view(
            _running_process_command_view(context.immediate_venv_redirector_process_handle),
            expected_argv=list(context.terminal_command.argv),
            expected_cwd=context.terminal_command.cwd,
            role="terminal immediate venv redirector",
        )
        _validate_live_process_command_view(
            _running_process_command_view(context.terminal_client_launcher_process_handle),
            expected_argv=cast(
                list[str],
                context.terminal_launcher_command["process_argv"],
            ),
            expected_cwd=cast(str, context.terminal_launcher_command["cwd"]),
            role="terminal-client launcher",
        )
        context.launcher_source_file.revalidate()
        source_identity = _physical_identity(
            context.launcher_source_file.descriptor,
            path=context.launcher_source_file.path,
            role="terminal-client-launcher",
            maximum_bytes=cast(
                int,
                context.terminal_launcher_release["source_size_bytes"],
            ),
        )
        if source_identity != context.terminal_launcher_release["source_physical_identity"]:
            raise OriginalConfirmatoryTerminalError(
                "terminal-client launcher source identity changed"
            )
    finally:
        _close_native_handle(current_handle)


def _granted_access_mask_from_native_handle(handle: int) -> int:
    if os.name != "nt" or handle <= 0:
        raise OriginalConfirmatoryTerminalError("native HANDLE access query is invalid")

    class _PublicObjectBasicInformation(ctypes.Structure):
        _fields_ = [
            ("attributes", ctypes.c_uint32),
            ("granted_access", ctypes.c_uint32),
            ("handle_count", ctypes.c_uint32),
            ("pointer_count", ctypes.c_uint32),
            ("reserved", ctypes.c_uint32 * 10),
        ]

    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    query_object = ntdll.NtQueryObject
    query_object.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    query_object.restype = ctypes.c_long
    information = _PublicObjectBasicInformation()
    return_length = ctypes.c_uint32()
    status = query_object(
        ctypes.c_void_p(handle),
        0,
        ctypes.byref(information),
        ctypes.sizeof(information),
        ctypes.byref(return_length),
    )
    if status < 0:
        rtl_error = ntdll.RtlNtStatusToDosError
        rtl_error.argtypes = [ctypes.c_long]
        rtl_error.restype = ctypes.c_uint32
        raise ctypes.WinError(rtl_error(status))
    if return_length.value > ctypes.sizeof(information):
        raise OriginalConfirmatoryTerminalError(
            "native HANDLE access query returned an unexpected structure"
        )
    return int(information.granted_access)


def _duplicate_to_process(
    descriptor: int,
    *,
    target_process_handle: int,
    access_mask: int,
) -> int:
    import msvcrt

    if access_mask != GENERIC_READ:
        raise OriginalConfirmatoryTerminalError(
            "composed-terminal duplication requested non-read access"
        )
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = ctypes.c_void_p
    current = get_current_process()
    target_handle = ctypes.c_void_p()
    duplicate_handle = kernel32.DuplicateHandle
    duplicate_handle.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint32,
    ]
    duplicate_handle.restype = ctypes.c_int
    if not duplicate_handle(
        ctypes.c_void_p(current),
        ctypes.c_void_p(msvcrt.get_osfhandle(descriptor)),
        ctypes.c_void_p(target_process_handle),
        ctypes.byref(target_handle),
        access_mask,
        False,
        0,
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if not target_handle.value:
        raise OriginalConfirmatoryTerminalError("DuplicateHandle returned no supervisor slot")
    return target_handle.value


def _duplicate_from_process(
    source_process_handle: int,
    *,
    source_handle_slot: int,
    access_mask: int = GENERIC_READ,
) -> int:
    import msvcrt

    if access_mask != GENERIC_READ:
        raise OriginalConfirmatoryTerminalError(
            "supervisor source duplication requested non-read access"
        )
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = ctypes.c_void_p
    current = get_current_process()
    local_handle = ctypes.c_void_p()
    duplicate_handle = kernel32.DuplicateHandle
    duplicate_handle.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint32,
    ]
    duplicate_handle.restype = ctypes.c_int
    if not duplicate_handle(
        ctypes.c_void_p(source_process_handle),
        ctypes.c_void_p(source_handle_slot),
        ctypes.c_void_p(current),
        ctypes.byref(local_handle),
        access_mask,
        False,
        0,
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if (
        not local_handle.value
        or _granted_access_mask_from_native_handle(local_handle.value)
        != FILE_GENERIC_READ_ACCESS_MASK
    ):
        if local_handle.value:
            kernel32.CloseHandle(local_handle)
        raise OriginalConfirmatoryTerminalError(
            "duplicated supervisor source has a non-read granted access mask"
        )
    try:
        return msvcrt.open_osfhandle(
            local_handle.value,
            os.O_RDONLY | getattr(os, "O_BINARY", 0),
        )
    except BaseException:
        kernel32.CloseHandle(local_handle)
        raise


def _supervisor_identity_from_launch(
    context: _VerifiedContext,
    *,
    consumed_e: _ConsumedEClaim,
) -> tuple[dict[str, Any], str]:
    """Use the already-held, pre-arm validated launch receipt for live custody."""

    consumed_e.revalidate()
    payload, _file_sha = _decode_supervisor_envelope(
        consumed_e.launch_intent,
        role="retained launch intent for terminal custody",
    )
    payload = _validate_launch_intent_payload(
        payload,
        context=context,
    )
    identity = _validate_process_identity(
        payload["supervisor_process_identity"],
        role="launch supervisor identity",
    )
    consumed_e.revalidate()
    return identity, cast(str, payload["windows_boot_time_utc"])


def _validate_process_identity(
    value: Any,
    *,
    role: str,
) -> dict[str, Any]:
    raw = _require_mapping(value, role=role)
    if (
        set(raw) != _PROCESS_IDENTITY_FIELDS
        or type(raw["pid"]) is not int
        or raw["pid"] <= 0
        or type(raw["creation_time_100ns"]) is not int
        or raw["creation_time_100ns"] <= 0
        or not _is_utc(raw["creation_time_utc"])
        or raw["creation_time_utc"] != _filetime_iso(raw["creation_time_100ns"])
        or not isinstance(raw["program_path"], str)
        or not Path(raw["program_path"]).is_absolute()
    ):
        raise OriginalConfirmatoryTerminalError(f"{role} is invalid")
    _require_sha256(raw["program_sha256"], role=f"{role} program")
    _require_sha256(raw["command_sha256"], role=f"{role} command")
    return raw


def _claim_identity(descriptor: int, *, path: Path) -> dict[str, Any]:
    observed = _physical_identity(
        descriptor,
        path=path,
        role="composed-terminal",
        allow_empty=True,
    )
    observed["policy"] = COMPOSED_CLAIM_PHYSICAL_IDENTITY_POLICY
    return _canonical_claim_identity(observed, path=path)


def _canonical_claim_identity(value: Any, *, path: Path) -> dict[str, Any]:
    raw = _require_mapping(value, role="composed-terminal CREATE_NEW claim identity")
    if (
        set(raw) != _CLAIM_PHYSICAL_IDENTITY_FIELDS
        or raw["schema_version"] != 1
        or raw["policy"] != COMPOSED_CLAIM_PHYSICAL_IDENTITY_POLICY
        or raw["role"] != "composed-terminal"
        or raw["path"] != str(path)
        or type(raw["volume_serial_number"]) is not int
        or raw["volume_serial_number"] < 0
        or not isinstance(raw["file_id_128"], str)
        or len(raw["file_id_128"]) != 32
        or any(character not in "0123456789abcdef" for character in raw["file_id_128"])
        or type(raw["device"]) is not int
        or type(raw["inode"]) is not int
        or raw["size_bytes"] != 0
        or type(raw["mode"]) is not int
        or type(raw["file_attributes"]) is not int
        or raw["regular_file"] is not True
        or raw["read_only"] is not True
        or raw["link_count"] != 1
        or type(raw["modified_time_ns"]) is not int
        or type(raw["changed_time_ns"]) is not int
        or raw["sha256"] != _sha256_bytes(b"")
        or raw["named_alternate_data_streams"] != []
        or raw["opened_without_reparse_follow"] is not True
        or raw["share_access"] != ["FILE_SHARE_READ"]
    ):
        raise OriginalConfirmatoryTerminalError(
            "composed-terminal CREATE_NEW claim identity is invalid"
        )
    return raw


def _build_claim_ready(
    context: _VerifiedContext,
    *,
    consumed_e: _ConsumedEClaim,
    client_identity: Mapping[str, Any],
    immediate_venv_redirector_identity: Mapping[str, Any],
    terminal_client_launcher_identity: Mapping[str, Any],
    supervisor_identity: Mapping[str, Any],
    claim_identity: Mapping[str, Any],
    target_handle: int,
) -> dict[str, Any]:
    consumed_e.revalidate()
    canonical_claim_identity = _canonical_claim_identity(
        claim_identity,
        path=context.terminal_contract.composed_terminal_receipt_path,
    )
    unsigned = {
        "schema_version": 1,
        "policy": COMPOSED_CLAIM_READY_POLICY,
        "message_type": CLAIM_READY_MESSAGE_TYPE,
        "job_id": context.job_id,
        "attempt_id": context.attempt_id,
        "run_id": context.run_id,
        "execution_mode": context.execution_mode,
        "retry_of_run_id": context.retry_of_run_id,
        "attempt_nonce": context.attempt_nonce,
        "q_authority_root_sha256": context.q["q_authority_root_sha256"],
        "e_intent_file_sha256": context.e_file.file_sha256,
        "e_intent_core_sha256": context.e["intent_core_sha256"],
        "e_consumption_claim_path": str(consumed_e.held.path),
        "e_consumption_claim_file_sha256": consumed_e.held.file_sha256,
        "e_consumption_claim_root_sha256": consumed_e.claim["claim_root_sha256"],
        "supervisor_spec_sha256": context.spec_sha256,
        "postwake_custody_seed_sha256": context.custody_seed.seed_sha256,
        "postwake_custody_handshake_contract_sha256": (context.custody_contract.contract_sha256),
        "terminal_composition_contract_sha256": context.terminal_contract.contract_sha256,
        "terminal_command_sha256": context.terminal_command.command_sha256,
        "observed_integrity_verifier_environment_sha256": (context.observed_environment_sha256),
        "client_process_identity": dict(client_identity),
        "immediate_venv_redirector_pid": immediate_venv_redirector_identity["pid"],
        "immediate_venv_redirector_process_identity": dict(immediate_venv_redirector_identity),
        "terminal_client_launcher_process_identity": dict(terminal_client_launcher_identity),
        "supervisor_process_identity": dict(supervisor_identity),
        "terminal_client_launch_intent_path": context.terminal_launcher_projection[
            "launch_intent_path"
        ],
        "terminal_client_launch_intent_policy": (authority.TERMINAL_CLIENT_LAUNCH_INTENT_POLICY),
        "terminal_client_launch_intent_read": False,
        "composed_terminal_path": str(context.terminal_contract.composed_terminal_receipt_path),
        "claim_physical_identity": canonical_claim_identity,
        "claim_physical_identity_root_sha256": _canonical_sha256(canonical_claim_identity),
        "target_supervisor_handle_value": target_handle,
        "target_access_mask": GENERIC_READ,
        "duplicate_options": 0,
        "close_source": False,
        "claim_before_terminal_input_read": True,
        "terminal_inputs_read": False,
        "automatic_retry_allowed": False,
    }
    ready = {**unsigned, "claim_ready_sha256": _canonical_sha256(unsigned)}
    ready = _validate_claim_ready(
        ready,
        context=context,
        consumed_e=consumed_e,
        expected_client_identity=client_identity,
        expected_immediate_venv_redirector_identity=(immediate_venv_redirector_identity),
        expected_terminal_client_launcher_identity=(terminal_client_launcher_identity),
        expected_supervisor_identity=supervisor_identity,
        expected_target_handle=target_handle,
    )
    consumed_e.revalidate()
    return ready


def _validate_claim_ready(
    value: Any,
    *,
    context: _VerifiedContext,
    consumed_e: _ConsumedEClaim,
    expected_client_identity: Mapping[str, Any],
    expected_immediate_venv_redirector_identity: Mapping[str, Any],
    expected_terminal_client_launcher_identity: Mapping[str, Any],
    expected_supervisor_identity: Mapping[str, Any],
    expected_target_handle: int,
) -> dict[str, Any]:
    consumed_e.revalidate()
    ready = _require_mapping(value, role="CLAIM_READY")
    if set(ready) != _CLAIM_READY_FIELDS:
        raise OriginalConfirmatoryTerminalError("CLAIM_READY fields differ")
    client_identity = _validate_process_identity(
        ready["client_process_identity"],
        role="CLAIM_READY client process identity",
    )
    immediate_venv_redirector_identity = _validate_process_identity(
        ready["immediate_venv_redirector_process_identity"],
        role="CLAIM_READY immediate venv redirector process identity",
    )
    terminal_client_launcher_identity = _validate_process_identity(
        ready["terminal_client_launcher_process_identity"],
        role="CLAIM_READY terminal-client launcher process identity",
    )
    supervisor_identity = _validate_process_identity(
        ready["supervisor_process_identity"],
        role="CLAIM_READY supervisor process identity",
    )
    claim_identity = _canonical_claim_identity(
        ready["claim_physical_identity"],
        path=context.terminal_contract.composed_terminal_receipt_path,
    )
    unsigned = {key: item for key, item in ready.items() if key != "claim_ready_sha256"}
    if (
        ready["schema_version"] != 1
        or ready["policy"] != COMPOSED_CLAIM_READY_POLICY
        or ready["message_type"] != CLAIM_READY_MESSAGE_TYPE
        or ready["job_id"] != context.job_id
        or ready["attempt_id"] != context.attempt_id
        or ready["run_id"] != context.run_id
        or ready["execution_mode"] != context.execution_mode
        or ready["retry_of_run_id"] != context.retry_of_run_id
        or ready["attempt_nonce"] != context.attempt_nonce
        or ready["q_authority_root_sha256"] != context.q["q_authority_root_sha256"]
        or ready["e_intent_file_sha256"] != context.e_file.file_sha256
        or ready["e_intent_core_sha256"] != context.e["intent_core_sha256"]
        or ready["e_consumption_claim_path"] != str(consumed_e.held.path)
        or ready["e_consumption_claim_file_sha256"] != consumed_e.held.file_sha256
        or ready["e_consumption_claim_root_sha256"] != consumed_e.claim["claim_root_sha256"]
        or ready["supervisor_spec_sha256"] != context.spec_sha256
        or ready["postwake_custody_seed_sha256"] != context.custody_seed.seed_sha256
        or ready["postwake_custody_handshake_contract_sha256"]
        != context.custody_contract.contract_sha256
        or ready["terminal_composition_contract_sha256"]
        != context.terminal_contract.contract_sha256
        or ready["terminal_command_sha256"] != context.terminal_command.command_sha256
        or ready["observed_integrity_verifier_environment_sha256"]
        != context.observed_environment_sha256
        or client_identity != dict(expected_client_identity)
        or ready["client_process_identity"] != client_identity
        or type(ready["immediate_venv_redirector_pid"]) is not int
        or ready["immediate_venv_redirector_pid"] != immediate_venv_redirector_identity["pid"]
        or immediate_venv_redirector_identity != dict(expected_immediate_venv_redirector_identity)
        or ready["immediate_venv_redirector_process_identity"] != immediate_venv_redirector_identity
        or terminal_client_launcher_identity != dict(expected_terminal_client_launcher_identity)
        or ready["terminal_client_launcher_process_identity"] != terminal_client_launcher_identity
        or supervisor_identity != dict(expected_supervisor_identity)
        or ready["supervisor_process_identity"] != supervisor_identity
        or ready["terminal_client_launch_intent_path"]
        != context.terminal_launcher_projection["launch_intent_path"]
        or ready["terminal_client_launch_intent_policy"]
        != authority.TERMINAL_CLIENT_LAUNCH_INTENT_POLICY
        or ready["terminal_client_launch_intent_read"] is not False
        or ready["composed_terminal_path"]
        != str(context.terminal_contract.composed_terminal_receipt_path)
        or ready["claim_physical_identity"] != claim_identity
        or ready["claim_physical_identity_root_sha256"] != _canonical_sha256(claim_identity)
        or type(ready["target_supervisor_handle_value"]) is not int
        or ready["target_supervisor_handle_value"] != expected_target_handle
        or expected_target_handle <= 0
        or ready["target_access_mask"] != GENERIC_READ
        or type(ready["target_access_mask"]) is not int
        or ready["duplicate_options"] != 0
        or type(ready["duplicate_options"]) is not int
        or ready["close_source"] is not False
        or ready["claim_before_terminal_input_read"] is not True
        or ready["terminal_inputs_read"] is not False
        or ready["automatic_retry_allowed"] is not False
        or ready["claim_ready_sha256"] != _canonical_sha256(unsigned)
    ):
        raise OriginalConfirmatoryTerminalError("CLAIM_READY violates its exact policy")
    consumed_e.revalidate()
    return ready


def _build_custody_grant(
    context: _VerifiedContext,
    *,
    claim_ready: Mapping[str, Any],
    launch_intent_file_sha256: str,
    launch_intent_root_sha256: str,
    launch_intent_handle_slot: int,
    launch_intent_identity: Mapping[str, Any],
    lease_handle_slot: int,
    lease_identity: Mapping[str, Any],
    granted_at_utc: str,
) -> dict[str, Any]:
    canonical_launch_intent_identity = (
        authority.canonical_original_confirmatory_physical_file_identity(
            launch_intent_identity,
            allowed_roles=("terminal-client-launch-intent",),
        ).as_dict()
    )
    canonical_lease_identity = authority.canonical_original_confirmatory_physical_file_identity(
        lease_identity,
        allowed_roles=("postwake-lease-receipt",),
    ).as_dict()
    unsigned = {
        "schema_version": 1,
        "policy": COMPOSED_CUSTODY_GRANT_POLICY,
        "message_type": CUSTODY_GRANT_MESSAGE_TYPE,
        "job_id": context.job_id,
        "attempt_id": context.attempt_id,
        "run_id": context.run_id,
        "execution_mode": context.execution_mode,
        "retry_of_run_id": context.retry_of_run_id,
        "attempt_nonce": context.attempt_nonce,
        "postwake_custody_seed_sha256": context.custody_seed.seed_sha256,
        "postwake_custody_handshake_contract_sha256": (context.custody_contract.contract_sha256),
        "terminal_composition_contract_sha256": (context.terminal_contract.contract_sha256),
        "claim_ready_sha256": claim_ready["claim_ready_sha256"],
        "client_process_identity": claim_ready["client_process_identity"],
        "immediate_venv_redirector_pid": claim_ready["immediate_venv_redirector_pid"],
        "immediate_venv_redirector_process_identity": claim_ready[
            "immediate_venv_redirector_process_identity"
        ],
        "terminal_client_launcher_process_identity": claim_ready[
            "terminal_client_launcher_process_identity"
        ],
        "supervisor_process_identity": claim_ready["supervisor_process_identity"],
        "terminal_client_launch_intent_path": claim_ready["terminal_client_launch_intent_path"],
        "terminal_client_launch_intent_policy": claim_ready["terminal_client_launch_intent_policy"],
        "terminal_client_launch_intent_file_sha256": launch_intent_file_sha256,
        "terminal_client_launch_intent_root_sha256": launch_intent_root_sha256,
        "terminal_client_launch_intent_physical_identity": (canonical_launch_intent_identity),
        "terminal_client_launch_intent_physical_identity_root_sha256": (
            _canonical_sha256(canonical_launch_intent_identity)
        ),
        "terminal_client_launch_intent_verified": True,
        "terminal_client_launch_intent_launcher_identity_verified": True,
        "terminal_client_launch_intent_create_new_before_child_verified": True,
        "terminal_client_launch_intent_supervisor_handle_slot": (launch_intent_handle_slot),
        "terminal_client_launch_intent_supervisor_granted_access_mask": (
            FILE_GENERIC_READ_ACCESS_MASK
        ),
        "terminal_client_launch_intent_child_duplicate_target_access_mask": (GENERIC_READ),
        "terminal_client_launch_intent_child_expected_granted_access_mask": (
            FILE_GENERIC_READ_ACCESS_MASK
        ),
        "terminal_client_launch_intent_child_duplicate_options": 0,
        "terminal_client_launch_intent_child_duplicate_close_source": False,
        "terminal_client_launch_intent_supervisor_custody_active": True,
        "terminal_client_launch_intent_child_open_after_grant_required": True,
        "target_supervisor_handle_value": claim_ready["target_supervisor_handle_value"],
        "target_granted_access_mask": FILE_GENERIC_READ_ACCESS_MASK,
        "claim_physical_identity": claim_ready["claim_physical_identity"],
        "claim_physical_identity_root_sha256": claim_ready["claim_physical_identity_root_sha256"],
        "postwake_input_lease_receipt_path": str(context.input_lease_contract.lease_receipt_path),
        "postwake_input_lease_handle_slot": lease_handle_slot,
        "postwake_input_lease_physical_identity": canonical_lease_identity,
        "postwake_input_lease_physical_identity_root_sha256": (
            _canonical_sha256(canonical_lease_identity)
        ),
        "same_supervisor_job_verified": True,
        "exact_wake_tree_descendant_verified": True,
        "client_process_identity_verified": True,
        "immediate_venv_redirector_process_identity_verified": True,
        "terminal_client_launcher_process_identity_verified": True,
        "launcher_redirector_child_grandparent_chain_verified": True,
        "launcher_redirector_child_same_supervisor_job_verified": True,
        "client_command_line_peb_readback_verified": True,
        "client_cwd_peb_readback_verified": True,
        "client_environment_peb_readback_verified": True,
        "supervisor_custody_active": True,
        "automatic_retry_allowed": False,
        "granted_at_utc": granted_at_utc,
    }
    grant = {
        **unsigned,
        "custody_grant_sha256": _canonical_sha256(unsigned),
    }
    return _validate_custody_grant(
        grant,
        context=context,
        claim_ready=claim_ready,
    )


def _validate_custody_grant(
    value: Any,
    *,
    context: _VerifiedContext,
    claim_ready: Mapping[str, Any],
) -> dict[str, Any]:
    grant = _require_mapping(value, role="CUSTODY_GRANT")
    if set(grant) != _CUSTODY_GRANT_FIELDS:
        raise OriginalConfirmatoryTerminalError("CUSTODY_GRANT fields differ")
    unsigned = {key: item for key, item in grant.items() if key != "custody_grant_sha256"}
    launch_intent_identity = authority.canonical_original_confirmatory_physical_file_identity(
        grant["terminal_client_launch_intent_physical_identity"],
        allowed_roles=("terminal-client-launch-intent",),
    ).as_dict()
    lease_identity = authority.canonical_original_confirmatory_physical_file_identity(
        grant["postwake_input_lease_physical_identity"],
        allowed_roles=("postwake-lease-receipt",),
    ).as_dict()
    if (
        grant["schema_version"] != 1
        or grant["policy"] != COMPOSED_CUSTODY_GRANT_POLICY
        or grant["message_type"] != CUSTODY_GRANT_MESSAGE_TYPE
        or any(
            grant[key] != claim_ready[key]
            for key in (
                "job_id",
                "attempt_id",
                "run_id",
                "execution_mode",
                "retry_of_run_id",
                "attempt_nonce",
                "postwake_custody_seed_sha256",
                "postwake_custody_handshake_contract_sha256",
                "terminal_composition_contract_sha256",
                "claim_ready_sha256",
                "client_process_identity",
                "immediate_venv_redirector_pid",
                "immediate_venv_redirector_process_identity",
                "terminal_client_launcher_process_identity",
                "supervisor_process_identity",
                "terminal_client_launch_intent_path",
                "terminal_client_launch_intent_policy",
                "target_supervisor_handle_value",
                "claim_physical_identity",
                "claim_physical_identity_root_sha256",
            )
        )
        or grant["terminal_client_launch_intent_path"]
        != context.terminal_launcher_projection["launch_intent_path"]
        or grant["terminal_client_launch_intent_policy"]
        != authority.TERMINAL_CLIENT_LAUNCH_INTENT_POLICY
        or _require_sha256(
            grant["terminal_client_launch_intent_file_sha256"],
            role="terminal-client launch intent file",
        )
        != launch_intent_identity["sha256"]
        or _require_sha256(
            grant["terminal_client_launch_intent_root_sha256"],
            role="terminal-client launch intent root",
        )
        != grant["terminal_client_launch_intent_root_sha256"]
        or grant["terminal_client_launch_intent_physical_identity"] != launch_intent_identity
        or launch_intent_identity["path"] != grant["terminal_client_launch_intent_path"]
        or grant["terminal_client_launch_intent_physical_identity_root_sha256"]
        != _canonical_sha256(launch_intent_identity)
        or type(grant["terminal_client_launch_intent_supervisor_handle_slot"]) is not int
        or grant["terminal_client_launch_intent_supervisor_handle_slot"] <= 0
        or grant["terminal_client_launch_intent_supervisor_granted_access_mask"]
        != FILE_GENERIC_READ_ACCESS_MASK
        or type(grant["terminal_client_launch_intent_supervisor_granted_access_mask"]) is not int
        or grant["terminal_client_launch_intent_child_duplicate_target_access_mask"] != GENERIC_READ
        or type(grant["terminal_client_launch_intent_child_duplicate_target_access_mask"])
        is not int
        or grant["terminal_client_launch_intent_child_expected_granted_access_mask"]
        != FILE_GENERIC_READ_ACCESS_MASK
        or type(grant["terminal_client_launch_intent_child_expected_granted_access_mask"])
        is not int
        or grant["terminal_client_launch_intent_child_duplicate_options"] != 0
        or type(grant["terminal_client_launch_intent_child_duplicate_options"]) is not int
        or grant["terminal_client_launch_intent_child_duplicate_close_source"] is not False
        or grant["postwake_input_lease_receipt_path"]
        != str(context.input_lease_contract.lease_receipt_path)
        or grant["target_granted_access_mask"] != FILE_GENERIC_READ_ACCESS_MASK
        or type(grant["target_granted_access_mask"]) is not int
        or type(grant["postwake_input_lease_handle_slot"]) is not int
        or grant["postwake_input_lease_handle_slot"] <= 0
        or grant["postwake_input_lease_handle_slot"]
        == grant["terminal_client_launch_intent_supervisor_handle_slot"]
        or grant["postwake_input_lease_physical_identity"] != lease_identity
        or grant["postwake_input_lease_physical_identity_root_sha256"]
        != _canonical_sha256(lease_identity)
        or any(
            grant[key] is not True
            for key in (
                "same_supervisor_job_verified",
                "exact_wake_tree_descendant_verified",
                "client_process_identity_verified",
                "immediate_venv_redirector_process_identity_verified",
                "terminal_client_launcher_process_identity_verified",
                "launcher_redirector_child_grandparent_chain_verified",
                "launcher_redirector_child_same_supervisor_job_verified",
                "terminal_client_launch_intent_verified",
                "terminal_client_launch_intent_launcher_identity_verified",
                "terminal_client_launch_intent_create_new_before_child_verified",
                "terminal_client_launch_intent_supervisor_custody_active",
                "terminal_client_launch_intent_child_open_after_grant_required",
                "client_command_line_peb_readback_verified",
                "client_cwd_peb_readback_verified",
                "client_environment_peb_readback_verified",
                "supervisor_custody_active",
            )
        )
        or grant["automatic_retry_allowed"] is not False
        or not _is_utc(grant["granted_at_utc"])
        or grant["custody_grant_sha256"] != _canonical_sha256(unsigned)
    ):
        raise OriginalConfirmatoryTerminalError("CUSTODY_GRANT violates its exact policy")
    return grant


def _open_validated_terminal_client_launch_intent(
    context: _VerifiedContext,
    *,
    custody_grant: Mapping[str, Any],
    supervisor_process_handle: int,
) -> tuple[_HeldFile, dict[str, Any], int, int]:
    """Duplicate and validate the launcher intent only after CUSTODY_GRANT."""

    if (
        context.terminal_launcher_command is None
        or context.terminal_client_launcher_process_identity is None
    ):
        raise OriginalConfirmatoryTerminalError("terminal-client launcher authority is unavailable")
    binding = {
        "handle_slot": custody_grant["terminal_client_launch_intent_supervisor_handle_slot"],
        "physical_identity": custody_grant["terminal_client_launch_intent_physical_identity"],
    }
    held = _held_from_supervisor_slot(
        supervisor_process_handle,
        binding=binding,
        maximum_bytes=_MAX_CONTROL_BYTES,
    )
    try:
        intent = authority.canonical_original_confirmatory_terminal_client_launch_intent(
            authority.decode_canonical_json_line(
                held.payload,
                role="terminal-client launch intent",
            ),
            launcher_command=context.terminal_launcher_command,
            launcher_projection=context.terminal_launcher_projection,
            launcher_release=context.terminal_launcher_release,
            capsule=context.q["execution_capsule"],
            verify_terminal_command=context.terminal_command,
        )
        import msvcrt

        child_handle_slot = msvcrt.get_osfhandle(held.descriptor)
        child_granted_access = _granted_access_mask_from_native_handle(child_handle_slot)
        _revalidate_terminal_client_launch_intent(
            context,
            held=held,
            intent=intent,
            custody_grant=custody_grant,
            child_handle_slot=child_handle_slot,
            child_granted_access=child_granted_access,
        )
        return held, intent, child_handle_slot, child_granted_access
    except authority.OriginalConfirmatoryCapsuleAuthorityError as exc:
        held.close()
        raise OriginalConfirmatoryTerminalError(
            "terminal-client launch intent violates its exact sealed policy"
        ) from exc
    except BaseException:
        held.close()
        raise


def _revalidate_terminal_client_launch_intent(
    context: _VerifiedContext,
    *,
    held: _HeldFile,
    intent: Mapping[str, Any],
    custody_grant: Mapping[str, Any],
    child_handle_slot: int,
    child_granted_access: int,
) -> None:
    """Prove that the same retained post-GRANT launch-intent handle is exact."""

    import msvcrt

    held.revalidate()
    observed_identity = _physical_identity(
        held.descriptor,
        path=held.path,
        role="terminal-client-launch-intent",
        maximum_bytes=_MAX_CONTROL_BYTES,
    )
    current_child_handle_slot = msvcrt.get_osfhandle(held.descriptor)
    current_child_granted_access = _granted_access_mask_from_native_handle(
        current_child_handle_slot
    )
    if (
        context.terminal_client_launcher_process_identity is None
        or held.path
        != Path(
            cast(
                str,
                context.terminal_launcher_projection["launch_intent_path"],
            )
        )
        or held.file_sha256 != custody_grant["terminal_client_launch_intent_file_sha256"]
        or intent["policy"] != custody_grant["terminal_client_launch_intent_policy"]
        or intent["intent_root_sha256"]
        != custody_grant["terminal_client_launch_intent_root_sha256"]
        or intent["launcher_process_identity"] != context.terminal_client_launcher_process_identity
        or intent["launcher_process_identity"]
        != custody_grant["terminal_client_launcher_process_identity"]
        or observed_identity != custody_grant["terminal_client_launch_intent_physical_identity"]
        or _canonical_sha256(observed_identity)
        != custody_grant["terminal_client_launch_intent_physical_identity_root_sha256"]
        or child_handle_slot <= 0
        or current_child_handle_slot != child_handle_slot
        or child_granted_access != FILE_GENERIC_READ_ACCESS_MASK
        or current_child_granted_access != child_granted_access
        or child_granted_access
        != custody_grant["terminal_client_launch_intent_child_expected_granted_access_mask"]
    ):
        raise OriginalConfirmatoryTerminalError(
            "post-GRANT terminal-client launch intent differs from custody"
        )


def _validate_retained_binding(
    value: Any,
    *,
    role: str,
) -> dict[str, Any]:
    return authority.canonical_original_confirmatory_retained_handle_binding(
        value,
        allowed_roles=(role,),
    ).as_dict()


def _validate_postwake_lease(
    value: Any,
    *,
    context: _VerifiedContext,
    custody_grant: Mapping[str, Any],
    expected_supervisor_identity: Mapping[str, Any],
    expected_boot_time_utc: str,
) -> dict[str, Any]:
    raw = _require_mapping(value, role="postwake input lease receipt")
    if set(raw) != _POSTWAKE_LEASE_FIELDS:
        raise OriginalConfirmatoryTerminalError("postwake input lease receipt fields differ")
    bindings_raw = raw["retained_input_bindings"]
    if not isinstance(bindings_raw, list) or len(bindings_raw) != len(_RETAINED_BINDING_ROLES):
        raise OriginalConfirmatoryTerminalError("postwake input lease bindings differ")
    bindings = [
        _validate_retained_binding(item, role=role)
        for item, role in zip(
            bindings_raw,
            _RETAINED_BINDING_ROLES,
            strict=True,
        )
    ]
    expected_paths = {
        "preterminal-pin": context.input_lease_contract.preterminal_pin_path,
        "preterminal-stdout": context.input_lease_contract.verifier_stdout_path,
        "preterminal-stderr": context.input_lease_contract.verifier_stderr_path,
        "supervisor-terminal": context.input_lease_contract.terminal_receipt_path,
    }
    source_slots = [cast(int, binding["handle_slot"]) for binding in bindings]
    expected_owner_fields = {
        "owner_pid": expected_supervisor_identity["pid"],
        "owner_creation_time_100ns": expected_supervisor_identity["creation_time_100ns"],
        "owner_windows_boot_time_utc": expected_boot_time_utc,
    }
    for binding in bindings:
        role = cast(str, binding["role"])
        identity = cast(dict[str, Any], binding["physical_identity"])
        if (
            Path(cast(str, identity["path"])) != expected_paths[role]
            or any(binding[key] != expected for key, expected in expected_owner_fields.items())
            or binding["share_mode"] != (3 if role == "preterminal-pin" else 1)
        ):
            raise OriginalConfirmatoryTerminalError(
                f"postwake input lease {role} owner/path/share differs"
            )
    self_binding = authority.canonical_original_confirmatory_preserialization_handle_binding(
        raw["self_preserialization_handle_binding"],
        expected_role="postwake-lease-receipt",
    ).as_dict()
    unsigned = {key: item for key, item in raw.items() if key != "receipt_root_sha256"}
    if (
        raw["schema_version"] != 1
        or raw["policy"] != POSTWAKE_INPUT_LEASE_RECEIPT_POLICY
        or raw["status"] != "passed"
        or raw["job_id"] != context.job_id
        or raw["attempt_id"] != context.attempt_id
        or raw["run_id"] != context.run_id
        or raw["execution_mode"] != context.execution_mode
        or raw["retry_of_run_id"] != context.retry_of_run_id
        or raw["attempt_nonce"] != context.attempt_nonce
        or raw["q_authority_root_sha256"] != context.q["q_authority_root_sha256"]
        or raw["e_intent_file_sha256"] != context.e_file.file_sha256
        or raw["e_intent_core_sha256"] != context.e["intent_core_sha256"]
        or raw["supervisor_spec_sha256"] != context.spec_sha256
        or raw["postwake_custody_seed_sha256"] != context.custody_seed.seed_sha256
        or raw["postwake_input_lease_contract_sha256"]
        != context.input_lease_contract.contract_sha256
        or raw["preterminal_pin_contract_sha256"] != context.preterminal_contract.contract_sha256
        or raw["preterminal_overlap_handshake_contract_sha256"]
        != context.overlap_contract.contract_sha256
        or raw["terminal_composition_contract_sha256"] != context.terminal_contract.contract_sha256
        or raw["supervisor_process_identity"] != dict(expected_supervisor_identity)
        or raw["windows_boot_time_utc"] != expected_boot_time_utc
        or raw["retained_input_bindings"] != bindings
        or raw["retained_input_bindings_root_sha256"] != _canonical_sha256(bindings)
        or len(set(source_slots)) != len(source_slots)
        or cast(int, self_binding["handle_slot"]) in set(source_slots)
        or raw["self_preserialization_handle_binding"] != self_binding
        or self_binding["path"] != str(context.input_lease_contract.lease_receipt_path)
        or any(self_binding[key] != expected for key, expected in expected_owner_fields.items())
        or self_binding["handle_slot"] != custody_grant["postwake_input_lease_handle_slot"]
        or self_binding["volume_serial_number"]
        != custody_grant["postwake_input_lease_physical_identity"]["volume_serial_number"]
        or self_binding["file_id_128"]
        != custody_grant["postwake_input_lease_physical_identity"]["file_id_128"]
        or raw["automatic_retry_allowed"] is not False
        or not _is_utc(raw["created_at_utc"])
        or raw["receipt_root_sha256"] != _canonical_sha256(unsigned)
    ):
        raise OriginalConfirmatoryTerminalError(
            "postwake input lease receipt violates its exact policy"
        )
    return {**raw, "retained_input_bindings": bindings}


def _held_from_supervisor_slot(
    supervisor_process_handle: int,
    *,
    binding: Mapping[str, Any],
    maximum_bytes: int,
) -> _HeldFile:
    identity = _require_mapping(
        binding["physical_identity"],
        role="retained source identity",
    )
    descriptor = _duplicate_from_process(
        supervisor_process_handle,
        source_handle_slot=cast(int, binding["handle_slot"]),
    )
    path = Path(cast(str, identity["path"]))
    try:
        if (
            maximum_bytes <= 0
            or type(identity["size_bytes"]) is not int
            or identity["size_bytes"] > maximum_bytes
        ):
            raise OriginalConfirmatoryTerminalError(
                "retained supervisor source exceeds its independent role bound"
            )
        payload = _read_descriptor(
            descriptor,
            maximum_bytes=maximum_bytes,
        )
        observed = _physical_identity(
            descriptor,
            path=path,
            role=cast(str, identity["role"]),
            maximum_bytes=maximum_bytes,
            allow_empty=cast(str, identity["role"]) == "preterminal-stderr",
        )
        if observed != identity:
            raise OriginalConfirmatoryTerminalError("duplicated retained source differs from L")
        return _HeldFile(
            path=path,
            descriptor=descriptor,
            payload=payload,
            role=cast(str, identity["role"]),
        )
    except BaseException:
        os.close(descriptor)
        raise


def _validate_preterminal_pin_payload(
    value: Any,
    *,
    context: _VerifiedContext,
    consumed_e: _ConsumedEClaim,
) -> dict[str, Any]:
    consumed_e.revalidate()
    raw = _require_mapping(value, role="preterminal pin")
    if set(raw) != _PRETERMINAL_PIN_FIELDS:
        raise OriginalConfirmatoryTerminalError("preterminal pin fields differ")
    evidence_value = raw["expected_artifact_evidence"]
    if not isinstance(evidence_value, list) or len(evidence_value) != len(
        context.expected_artifact_rules
    ):
        raise OriginalConfirmatoryTerminalError("preterminal pin artifact evidence length differs")
    evidence: list[dict[str, Any]] = []
    for index, (record_value, rule) in enumerate(
        zip(evidence_value, context.expected_artifact_rules, strict=True)
    ):
        record = _require_mapping(
            record_value,
            role=f"preterminal artifact evidence {index}",
        )
        checked_paths = record.get("json_control_paths_checked")
        if (
            set(record) != _PRETERMINAL_ARTIFACT_EVIDENCE_FIELDS
            or record["role"] != rule["role"]
            or record["path"] != rule["path"]
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] <= 0
            or _require_sha256(
                record["sha256"],
                role=f"preterminal artifact {rule['role']} hash",
            )
            != record["sha256"]
            or not _strict_json_value_equal(
                record["expected_sha256"],
                rule["expected_sha256"],
            )
            or not isinstance(checked_paths, list)
            or not all(isinstance(item, str) for item in checked_paths)
            or checked_paths != sorted(rule["json_equals"])
            or record["valid"] is not True
        ):
            raise OriginalConfirmatoryTerminalError(
                "preterminal pin artifact evidence differs from exact policy"
            )
        evidence.append(record)
    required_roles = raw["required_success_roles"]
    if (
        not isinstance(required_roles, list)
        or required_roles != list(_EXPECTED_ARTIFACT_ROLE_ORDER)
        or [item["role"] for item in evidence] != list(_EXPECTED_ARTIFACT_ROLE_ORDER)
    ):
        raise OriginalConfirmatoryTerminalError("preterminal pin required artifact roles differ")
    evidence_root = _canonical_sha256(evidence)
    unsigned = {key: item for key, item in raw.items() if key != "evidence_root_sha256"}
    scientific_core = {
        "run_spec_file_sha256": raw["run_spec_file_sha256"],
        "canonical_spec_sha256": raw["canonical_spec_sha256"],
        "launch_intent_file_sha256": raw["launch_intent_file_sha256"],
        "process_started_file_sha256": raw["process_started_file_sha256"],
        "e_consumption_claim_file_sha256": raw["e_consumption_claim_file_sha256"],
        "e_consumption_claim_root_sha256": raw["e_consumption_claim_root_sha256"],
        "scientific_command_sha256": raw["scientific_command_sha256"],
        "required_success_roles": required_roles,
        "expected_artifact_evidence_root_sha256": evidence_root,
    }
    if (
        raw["schema_version"] != 1
        or raw["policy"] != PRETERMINAL_PIN_POLICY
        or raw["status"] != "passed"
        or raw["job_id"] != context.job_id
        or raw["attempt_id"] != context.attempt_id
        or raw["run_id"] != context.run_id
        or raw["execution_mode"] != context.execution_mode
        or raw["retry_of_run_id"] != context.retry_of_run_id
        or raw["attempt_nonce"] != context.attempt_nonce
        or raw["q_authority_root_sha256"] != context.q["q_authority_root_sha256"]
        or raw["e_intent_path"] != str(context.e_file.path)
        or raw["e_intent_file_sha256"] != context.e_file.file_sha256
        or raw["e_intent_core_sha256"] != context.e["intent_core_sha256"]
        or raw["e_consumption_claim_path"] != str(consumed_e.held.path)
        or raw["e_consumption_claim_file_sha256"] != consumed_e.held.file_sha256
        or raw["e_consumption_claim_root_sha256"] != consumed_e.claim["claim_root_sha256"]
        or raw["capsule_contract_sha256"] != context.q["execution_capsule"]["contract_sha256"]
        or raw["capsule_sha256"] != context.q["execution_capsule"]["sha256"]
        or raw["capsule_internal_manifest_sha256"]
        != context.q["execution_capsule"]["internal_manifest_sha256"]
        or raw["capsule_mode"] != authority.CAPSULE_PRETERMINAL_MODE
        or raw["preterminal_pin_contract_sha256"] != context.preterminal_contract.contract_sha256
        or raw["preterminal_overlap_handshake_contract_sha256"]
        != context.overlap_contract.contract_sha256
        or raw["run_spec_file_sha256"] != context.spec_sha256
        or raw["canonical_spec_sha256"] != context.run_spec_payload["canonical_spec_sha256"]
        or raw["scientific_command_sha256"] != context.scientific_command.command_sha256
        or raw["preterminal_command_sha256"] != context.preterminal_command.command_sha256
        or raw["terminal_command_sha256"] != context.terminal_command.command_sha256
        or raw["observed_integrity_verifier_environment_sha256"]
        != context.observed_environment_sha256
        or raw["launch_intent_file_sha256"] != consumed_e.launch_intent.file_sha256
        or raw["process_started_file_sha256"] != consumed_e.process_started.file_sha256
        or raw["expected_artifact_evidence"] != evidence
        or raw["expected_artifact_evidence_root_sha256"] != evidence_root
        or raw["preterminal_scientific_core_sha256"] != _canonical_sha256(scientific_core)
        or raw["semantic_outcome_read_scope"] != SEMANTIC_OUTCOME_READ_SCOPE
        or any(
            raw[key] is not False
            for key in (
                "outcome_values_read",
                "outcome_values_emitted",
                "outcome_values_used_for_selection_or_tuning",
                "training_or_model_selection_allowed",
                "scientific_publication_allowed",
                "automatic_retry_allowed",
            )
        )
        or not _is_utc(raw["created_at_utc"])
        or raw["evidence_root_sha256"] != _canonical_sha256(unsigned)
    ):
        raise OriginalConfirmatoryTerminalError("preterminal pin violates its exact policy")
    consumed_e.revalidate()
    return raw


def _validate_success_log_record(
    value: Any,
    *,
    role: str,
    expected_path: Path,
    maximum_bytes: int,
    held: _HeldFile | None = None,
    verify_path_bytes: bool = False,
) -> dict[str, Any]:
    raw = _require_mapping(value, role=f"{role} log record")
    if set(raw) != _SUPERVISOR_LOG_RECORD_FIELDS:
        raise OriginalConfirmatoryTerminalError(f"{role} log fields differ")
    size = raw["size_bytes"]
    stream_size = raw["stream_size_bytes"]
    discarded = raw["discarded_bytes"]
    sha256 = _require_sha256(raw["sha256"], role=f"{role} log hash")
    stored_sha256 = _require_sha256(
        raw["stored_sha256"],
        role=f"{role} stored log hash",
    )
    stream_sha256 = _require_sha256(
        raw["stream_sha256"],
        role=f"{role} stream log hash",
    )
    if (
        raw["path"] != str(expected_path)
        or raw["exists"] is not True
        or type(size) is not int
        or size < 0
        or size > maximum_bytes
        or raw["limit_bytes"] != maximum_bytes
        or type(raw["limit_bytes"]) is not int
        or raw["limit_exceeded"] is not False
        or raw["capture_complete"] is not True
        or type(stream_size) is not int
        or type(discarded) is not int
        or stream_size > maximum_bytes
        or discarded != 0
        or stream_size != size
        or sha256 != stored_sha256
        or stream_sha256 != sha256
    ):
        raise OriginalConfirmatoryTerminalError(f"{role} log is not one successful bounded capture")
    if held is not None:
        held.revalidate()
        if held.path != expected_path or len(held.payload) != size or held.file_sha256 != sha256:
            raise OriginalConfirmatoryTerminalError(f"{role} log differs from its retained handle")
    elif verify_path_bytes:
        # This is byte-level integrity verification only.  No log text is
        # decoded, emitted, selected, or used as a scientific outcome.
        observed = _open_held_file(
            expected_path,
            role=f"{role} sealed log",
            maximum_bytes=maximum_bytes,
            allow_empty=True,
        )
        try:
            if len(observed.payload) != size or observed.file_sha256 != sha256:
                raise OriginalConfirmatoryTerminalError(
                    f"{role} log differs from its sealed byte identity"
                )
            observed.revalidate()
        finally:
            observed.close()
    return raw


def _validate_preterminal_ready_summary(
    value: Any,
    *,
    context: _VerifiedContext,
    pin: Mapping[str, Any],
    pin_identity: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _require_mapping(value, role="preterminal READY summary")
    if set(raw) != _PRETERMINAL_READY_FIELDS:
        raise OriginalConfirmatoryTerminalError("preterminal READY fields differ")
    canonical_pin_identity = authority.canonical_original_confirmatory_physical_file_identity(
        raw["preterminal_pin_receipt"],
        allowed_roles=("preterminal-pin",),
    ).as_dict()
    if (
        raw["schema_version"] != 1
        or raw["policy"] != PRETERMINAL_READY_POLICY
        or raw["message_type"] != PRETERMINAL_READY_MESSAGE_TYPE
        or raw["handshake_policy"] != PRETERMINAL_HANDSHAKE_POLICY
        or raw["job_id"] != context.job_id
        or raw["attempt_id"] != context.attempt_id
        or raw["run_id"] != context.run_id
        or raw["execution_mode"] != context.execution_mode
        or raw["retry_of_run_id"] != context.retry_of_run_id
        or raw["attempt_nonce"] != context.attempt_nonce
        or raw["capsule_contract_sha256"] != context.q["execution_capsule"]["contract_sha256"]
        or raw["capsule_sha256"] != context.q["execution_capsule"]["sha256"]
        or raw["capsule_internal_manifest_sha256"]
        != context.q["execution_capsule"]["internal_manifest_sha256"]
        or raw["capsule_mode"] != authority.CAPSULE_PRETERMINAL_MODE
        or raw["preterminal_pin_contract_sha256"] != context.preterminal_contract.contract_sha256
        or raw["observed_integrity_verifier_environment_sha256"]
        != context.observed_environment_sha256
        or raw["preterminal_pin_receipt"] != canonical_pin_identity
        or canonical_pin_identity != dict(pin_identity)
        or raw["preterminal_pin_evidence_root_sha256"] != pin["evidence_root_sha256"]
        or raw["preterminal_scientific_core_sha256"] != pin["preterminal_scientific_core_sha256"]
        or raw["semantic_outcome_read_scope"] != SEMANTIC_OUTCOME_READ_SCOPE
        or any(
            raw[key] is not False
            for key in (
                "outcome_values_read",
                "outcome_values_emitted",
                "outcome_values_used_for_selection_or_tuning",
                "training_or_model_selection_allowed",
                "scientific_publication_allowed",
                "automatic_retry_allowed",
            )
        )
        or raw["pin_handle_open"] is not True
        or raw["pin_handle_share_access"] != ["FILE_SHARE_READ"]
        or raw["awaiting_supervisor_ack"] is not True
    ):
        raise OriginalConfirmatoryTerminalError("preterminal READY violates its exact policy")
    return raw


def _validate_preterminal_handshake_receipt(
    value: Any,
    *,
    context: _VerifiedContext,
    pin: Mapping[str, Any],
    verifier_process_identity: Mapping[str, Any],
    source_files: Mapping[str, _HeldFile],
    bindings: list[dict[str, Any]],
) -> dict[str, Any]:
    raw = _require_mapping(value, role="preterminal overlap handshake receipt")
    if set(raw) != _PRETERMINAL_HANDSHAKE_RECEIPT_FIELDS:
        raise OriginalConfirmatoryTerminalError(
            "preterminal overlap handshake receipt fields differ"
        )
    binding_by_role = {cast(str, binding["role"]): binding for binding in bindings}
    if set(binding_by_role) != set(_RETAINED_BINDING_ROLES):
        raise OriginalConfirmatoryTerminalError("preterminal overlap source roles differ")
    pin_identity = cast(
        dict[str, Any],
        binding_by_role["preterminal-pin"]["physical_identity"],
    )
    stdout_identity = cast(
        dict[str, Any],
        binding_by_role["preterminal-stdout"]["physical_identity"],
    )
    stderr_identity = cast(
        dict[str, Any],
        binding_by_role["preterminal-stderr"]["physical_identity"],
    )
    ready = _validate_preterminal_ready_summary(
        raw["ready_summary"],
        context=context,
        pin=pin,
        pin_identity=pin_identity,
    )
    ready_line = _canonical_bytes(ready)
    ack = _validate_preterminal_ack(
        _require_mapping(raw["ack"], role="preterminal receipt ACK"),
        context=context,
        ready=ready,
        ready_line=ready_line,
        pin_identity=pin_identity,
    )
    stable_pin = _stable_physical_identity_projection(pin_identity)
    unsigned = {key: item for key, item in raw.items() if key != "evidence_root_sha256"}
    for held in source_files.values():
        held.revalidate()
    if (
        raw["schema_version"] != 1
        or raw["policy"] != _PRETERMINAL_HANDSHAKE_RECEIPT_POLICY
        or raw["status"] != "passed"
        or raw["job_id"] != context.job_id
        or raw["attempt_id"] != context.attempt_id
        or raw["run_id"] != context.run_id
        or raw["execution_mode"] != context.execution_mode
        or raw["retry_of_run_id"] != context.retry_of_run_id
        or raw["attempt_nonce"] != context.attempt_nonce
        or raw["verifier_process_identity"] != dict(verifier_process_identity)
        or raw["preterminal_pin_contract_sha256"] != context.preterminal_contract.contract_sha256
        or raw["preterminal_overlap_handshake_contract_sha256"]
        != context.overlap_contract.contract_sha256
        or raw["ready_summary"] != ready
        or raw["ready_line_sha256"] != _sha256_bytes(ready_line)
        or raw["ack"] != ack
        or raw["ack_line_sha256"] != _sha256_bytes(_canonical_bytes(ack))
        or raw["child_reported_pin_identity"] != pin_identity
        or raw["supervisor_opened_pin_identity"] != ack["supervisor_opened_pin_identity"]
        or raw["preterminal_pin_identity_root_sha256"] != _canonical_sha256(stable_pin)
        or raw["stdout_log_identity"] != stdout_identity
        or raw["stderr_log_identity"] != stderr_identity
        or source_files["preterminal-pin"].file_sha256 != pin_identity["sha256"]
        or source_files["preterminal-stdout"].payload != ready_line
        or source_files["preterminal-stdout"].file_sha256 != stdout_identity["sha256"]
        or source_files["preterminal-stderr"].payload != b""
        or source_files["preterminal-stderr"].file_sha256 != stderr_identity["sha256"]
        or raw["stdout_eof_after_ack"] is not True
        or raw["stdout_additional_bytes"] != 0
        or type(raw["stdout_additional_bytes"]) is not int
        or raw["stderr_empty"] is not True
        or raw["exit_code"] != 0
        or type(raw["exit_code"]) is not int
        or raw["pin_handle_overlap_verified"] is not True
        or raw["supervisor_pin_handle_active"] is not True
        or raw["restricted_inherited_handle_list_verified"] is not True
        or raw["automatic_retry_allowed"] is not False
        or not _is_utc(raw["created_at_utc"])
        or raw["evidence_root_sha256"] != _canonical_sha256(unsigned)
    ):
        raise OriginalConfirmatoryTerminalError(
            "preterminal overlap handshake receipt violates its exact policy"
        )
    return raw


def _validate_terminal_artifact_records(
    value: Any,
    *,
    pin: Mapping[str, Any],
) -> list[dict[str, Any]]:
    # The terminal capsule must not broaden its outcome-read scope by parsing
    # scientific artifacts again.  Instead it requires the supervisor's
    # independently repeated byte identity inspection to match P record-for-record.
    raw_records = value
    pin_records = pin["expected_artifact_evidence"]
    if (
        not isinstance(raw_records, list)
        or not isinstance(pin_records, list)
        or len(raw_records) != len(pin_records)
    ):
        raise OriginalConfirmatoryTerminalError("terminal artifact evidence length differs from P")
    records: list[dict[str, Any]] = []
    for index, (raw_value, pin_value) in enumerate(zip(raw_records, pin_records, strict=True)):
        raw = _require_mapping(
            raw_value,
            role=f"terminal artifact evidence {index}",
        )
        pin_record = _require_mapping(
            pin_value,
            role=f"P artifact evidence {index}",
        )
        if (
            set(raw) != _SUPERVISOR_TERMINAL_ARTIFACT_FIELDS
            or raw["role"] != pin_record["role"]
            or raw["path"] != pin_record["path"]
            or raw["valid"] is not True
            or raw["errors"] != []
            or type(raw["size_bytes"]) is not int
            or raw["size_bytes"] <= 0
            or raw["size_bytes"] != pin_record["size_bytes"]
            or _require_sha256(
                raw["sha256"],
                role=f"terminal artifact {raw['role']} hash",
            )
            != pin_record["sha256"]
        ):
            raise OriginalConfirmatoryTerminalError(
                "terminal artifact evidence differs from exact P evidence"
            )
        records.append(raw)
    return records


def _validate_terminal_e_consumption_binding(
    terminal: Mapping[str, Any],
    *,
    context: _VerifiedContext,
    consumed_e: _ConsumedEClaim,
) -> None:
    consumed_e.revalidate()
    contract = authority.canonical_original_confirmatory_e_consumption_contract(
        context.e["e_consumption_contract"],
        supervisor_job_directory=context.job_dir,
    )
    receipt_held = _open_held_file(
        contract.custody_receipt_path,
        role="E consumption custody receipt",
        maximum_bytes=contract.ack_line_max_bytes,
    )
    try:
        receipt_value = authority.decode_canonical_json_line(
            receipt_held.payload,
            role="E consumption custody receipt",
        )
        contract_payload = contract.as_dict()
        ready_unsigned = {
            "schema_version": 1,
            "policy": authority.E_CONSUMPTION_READY_POLICY,
            "message_type": contract_payload["ready_message_type"],
            "contract_sha256": contract.contract_sha256,
            "claim_path": str(contract.claim_path),
            "claim_file_sha256": consumed_e.held.file_sha256,
            "claim_root_sha256": consumed_e.claim["claim_root_sha256"],
            "claim_physical_identity": receipt_value["claim_physical_identity"],
            "target_supervisor_handle_value": receipt_value["target_supervisor_handle_value"],
            "duplicate_target_access_mask": contract_payload["duplicate_target_access_mask"],
            "duplicate_options": contract_payload["duplicate_options"],
            "close_source": False,
            "child_process_identity": consumed_e.claim["child_process_identity"],
            "supervisor_process_identity": consumed_e.claim["supervisor_process_identity"],
            "supervisor_spec_sha256": context.spec_sha256,
            "process_started_sha256": consumed_e.process_started.file_sha256,
        }
        ready = authority.canonical_original_confirmatory_e_consumption_ready(
            {
                **ready_unsigned,
                "ready_sha256": receipt_value["ready_sha256"],
            },
            contract=contract,
            claim=consumed_e.claim,
            claim_physical_identity=receipt_value["claim_physical_identity"],
        )
        receipt = authority.canonical_original_confirmatory_e_consumption_custody_receipt(
            receipt_value,
            contract=contract,
            ready=ready,
        )
        ack_unsigned = {
            "schema_version": 1,
            "policy": authority.E_CONSUMPTION_ACK_POLICY,
            "message_type": contract_payload["ack_message_type"],
            "contract_sha256": contract.contract_sha256,
            "ready_sha256": ready["ready_sha256"],
            "claim_file_sha256": consumed_e.held.file_sha256,
            "claim_root_sha256": consumed_e.claim["claim_root_sha256"],
            "custody_receipt_path": str(contract.custody_receipt_path),
            "custody_receipt_sha256": receipt_held.file_sha256,
            "supervisor_process_identity": consumed_e.claim["supervisor_process_identity"],
            "target_supervisor_handle_value": ready["target_supervisor_handle_value"],
        }
        ack = authority.canonical_original_confirmatory_e_consumption_ack(
            {
                **ack_unsigned,
                "ack_sha256": terminal["e_consumption_ack_sha256"],
            },
            contract=contract,
            ready=ready,
            custody_receipt=receipt,
            custody_receipt_file_sha256=receipt_held.file_sha256,
        )
        if (
            terminal["e_consumption_contract_sha256"] != contract.contract_sha256
            or terminal["e_consumption_custody_receipt_file_sha256"] != receipt_held.file_sha256
            or terminal["e_consumption_custody_receipt_root_sha256"] != receipt["receipt_sha256"]
            or terminal["e_consumption_ready_sha256"] != ready["ready_sha256"]
            or terminal["e_consumption_ack_sha256"] != ack["ack_sha256"]
        ):
            raise OriginalConfirmatoryTerminalError(
                "terminal E-consumption evidence differs from exact custody readback"
            )
        receipt_held.revalidate()
        consumed_e.revalidate()
    finally:
        receipt_held.close()


def _validate_supervisor_verifier_record(
    value: Any,
    *,
    context: _VerifiedContext,
    pin: Mapping[str, Any],
    source_files: Mapping[str, _HeldFile],
    bindings: list[dict[str, Any]],
) -> dict[str, Any]:
    raw = _require_mapping(value, role="supervisor integrity verifier record")
    if set(raw) != _SUPERVISOR_VERIFIER_RECORD_FIELDS:
        raise OriginalConfirmatoryTerminalError("supervisor integrity verifier fields differ")
    identity = _validate_process_identity(
        raw["process_identity"],
        role="supervisor integrity verifier process identity",
    )
    stdout = _validate_success_log_record(
        raw["stdout"],
        role="integrity verifier stdout",
        expected_path=context.terminal_contract.verifier_stdout_path,
        maximum_bytes=context.terminal_contract.verifier_stdout_max_bytes,
        held=source_files["preterminal-stdout"],
    )
    stderr = _validate_success_log_record(
        raw["stderr"],
        role="integrity verifier stderr",
        expected_path=context.terminal_contract.verifier_stderr_path,
        maximum_bytes=context.terminal_contract.verifier_stderr_max_bytes,
        held=source_files["preterminal-stderr"],
    )
    handshake = _validate_preterminal_handshake_receipt(
        raw["preterminal_overlap_handshake_receipt"],
        context=context,
        pin=pin,
        verifier_process_identity=identity,
        source_files=source_files,
        bindings=bindings,
    )
    expected_identity = {
        "program_path": str(context.preterminal_command.program_path),
        "program_sha256": context.preterminal_command.program_sha256,
        "command_sha256": context.preterminal_command.command_sha256,
    }
    if (
        raw["command"] != _canonical_command_dict(context.preterminal_command)
        or any(identity[key] != expected for key, expected in expected_identity.items())
        or raw["process_identity"] != identity
        or not _is_utc(raw["started_at_utc"])
        or not _is_utc(raw["ended_at_utc"])
        or raw["started_at_utc"] > raw["ended_at_utc"]
        or raw["timeout_ms"] != context.spec["verifier_timeout_ms"]
        or type(raw["timeout_ms"]) is not int
        or raw["job_assignment_mode"] != "PROC_THREAD_ATTRIBUTE_JOB_LIST"
        or raw["atomic_job_assignment"] is not True
        or raw["handle_list_restricted"] is not True
        or raw["job_handle_inherited"] is not False
        or raw["exit_code"] != 0
        or type(raw["exit_code"]) is not int
        or raw["descendants_after_root_exit"] != 0
        or type(raw["descendants_after_root_exit"]) is not int
        or raw["stdout"] != stdout
        or raw["stderr"] != stderr
        or raw["error_type"] is not None
        or raw["error_sha256"] is not None
        or raw["cleanup_error_type"] is not None
        or raw["cleanup_error_sha256"] is not None
        or raw["tree_empty_verified"] is not True
        or raw["valid"] is not True
        or raw["capsule_contract_sha256"] != context.q["execution_capsule"]["contract_sha256"]
        or raw["capsule_sha256"] != context.q["execution_capsule"]["sha256"]
        or raw["capsule_internal_manifest_sha256"]
        != context.q["execution_capsule"]["internal_manifest_sha256"]
        or raw["capsule_mode"] != authority.CAPSULE_PRETERMINAL_MODE
        or raw["expected_environment_envelope_sha256"]
        != context.spec["expected_environment"]["envelope_sha256"]
        or raw["process_environment_binding_sha256"]
        != context.spec["process_environment_binding"]["binding_sha256"]
        or raw["exact_integrity_verifier_environment_sha256"]
        != context.spec["process_environment_binding"][
            "exact_integrity_verifier_environment_sha256"
        ]
        or raw["observed_integrity_verifier_environment_sha256"]
        != context.observed_environment_sha256
        or raw["integrity_verifier_environment_exact_match"] is not True
        or raw["integrity_verifier_environment_observation_method"]
        != "windows_peb_process_parameters_v1"
        or raw["preterminal_pin_contract_sha256"] != context.preterminal_contract.contract_sha256
        or raw["preterminal_overlap_handshake_contract_sha256"]
        != context.overlap_contract.contract_sha256
        or raw["preterminal_overlap_handshake_receipt"] != handshake
        or raw["capsule_lease_identity_root_sha256"]
        != context.spec["capsule_lease_identity_root_sha256"]
        or raw["capsule_ancestor_lease_root_sha256"]
        != context.spec["capsule_ancestor_lease_root_sha256"]
        or raw["python_lease_identity_root_sha256"]
        != context.spec["python_lease_identity_root_sha256"]
        or raw["python_ancestor_lease_root_sha256"]
        != context.spec["python_ancestor_lease_root_sha256"]
        or raw["interpreter_leaf_handle_active"]
        is not (context.spec["python_lease_identity"] is not None)
        or raw["interpreter_ancestor_handles_active"]
        is not (context.spec["python_ancestor_lease"] is not None)
        or raw["python_runtime_resolution_policy"]
        != context.spec["python_runtime_resolution_policy"]
        or raw["runtime_python_lease_identity_root_sha256"]
        != context.spec["runtime_python_lease_identity_root_sha256"]
        or raw["runtime_python_ancestor_lease_root_sha256"]
        != context.spec["runtime_python_ancestor_lease_root_sha256"]
        or raw["runtime_interpreter_leaf_handle_active"]
        is not (context.spec["runtime_python_lease_identity"] is not None)
        or raw["runtime_interpreter_ancestor_handles_active"]
        is not (context.spec["runtime_python_ancestor_lease"] is not None)
    ):
        raise OriginalConfirmatoryTerminalError(
            "supervisor integrity verifier is not exact successful preterminal evidence"
        )
    return raw


def _validate_supervisor_terminal(
    value: Any,
    *,
    context: _VerifiedContext,
    consumed_e: _ConsumedEClaim,
    pin: Mapping[str, Any],
    source_files: Mapping[str, _HeldFile],
    bindings: list[dict[str, Any]],
) -> dict[str, Any]:
    consumed_e.revalidate()
    raw = _require_mapping(value, role="supervisor terminal receipt")
    if set(raw) != _SUPERVISOR_TERMINAL_FIELDS:
        raise OriginalConfirmatoryTerminalError("supervisor terminal receipt fields differ")
    launch, _launch_file_sha = _decode_supervisor_envelope(
        consumed_e.launch_intent,
        role="retained terminal launch intent",
    )
    launch = _validate_launch_intent_payload(launch, context=context)
    started, _started_file_sha = _decode_supervisor_envelope(
        consumed_e.process_started,
        role="retained terminal process-started",
    )
    started = _validate_process_started_payload(
        started,
        context=context,
        launch=launch,
        launch_file_sha256=consumed_e.launch_intent.file_sha256,
    )
    stdout = _validate_success_log_record(
        raw["stdout"],
        role="scientific stdout",
        expected_path=context.job_dir / "stdout.log",
        maximum_bytes=context.spec["max_log_bytes"],
        verify_path_bytes=True,
    )
    stderr = _validate_success_log_record(
        raw["stderr"],
        role="scientific stderr",
        expected_path=context.job_dir / "stderr.log",
        maximum_bytes=context.spec["max_log_bytes"],
        verify_path_bytes=True,
    )
    artifacts = _validate_terminal_artifact_records(
        raw["expected_artifacts"],
        pin=pin,
    )
    verifier = _validate_supervisor_verifier_record(
        raw["integrity_verifier"],
        context=context,
        pin=pin,
        source_files=source_files,
        bindings=bindings,
    )
    environment = _require_mapping(
        raw["environment_binding"],
        role="supervisor terminal environment binding",
    )
    binding = _require_mapping(
        context.spec["process_environment_binding"],
        role="supervisor process-environment binding",
    )
    expected_environment = _require_mapping(
        context.spec["expected_environment"],
        role="supervisor expected environment",
    )
    expected_terminal_environment = {
        "expected_environment_envelope_sha256": expected_environment["envelope_sha256"],
        "launch_environment_root_sha256": expected_environment["launch_environment_root_sha256"],
        "process_environment_binding_sha256": binding["binding_sha256"],
        "exact_supervisor_environment_sha256": binding["exact_supervisor_environment_sha256"],
        "observed_supervisor_environment_sha256": launch["observed_supervisor_environment_sha256"],
        "exact_environment_sha256": binding["exact_environment_sha256"],
        "observed_child_environment_sha256": started["observed_child_environment_sha256"],
        "exact_integrity_verifier_environment_sha256": binding[
            "exact_integrity_verifier_environment_sha256"
        ],
        "observed_integrity_verifier_environment_sha256": verifier[
            "observed_integrity_verifier_environment_sha256"
        ],
        "all_exact_matches": True,
    }
    if set(environment) != _SUPERVISOR_TERMINAL_ENVIRONMENT_FIELDS or not _strict_json_value_equal(
        environment,
        expected_terminal_environment,
    ):
        raise OriginalConfirmatoryTerminalError("supervisor terminal environment binding differs")
    _validate_terminal_e_consumption_binding(
        raw,
        context=context,
        consumed_e=consumed_e,
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 3
        or raw["policy"] != authority.SUPERVISOR_V3_POLICY
        or raw["attempt_policy"] != _SUPERVISOR_ATTEMPT_POLICY
        or raw["job_id"] != context.job_id
        or raw["process_kind"] != _SUPERVISOR_PROCESS_KIND
        or raw["spec_sha256"] != context.spec_sha256
        or raw["terminal_kind"] != authority.SUPERVISOR_V2_SUCCESS_TERMINAL_KIND
        or raw["reason"] != _SUPERVISOR_SUCCESS_REASON
        or raw["attempt_count"] != 1
        or type(raw["attempt_count"]) is not int
        or raw["automatic_retry_allowed"] is not False
        or raw["exit_code"] != 0
        or type(raw["exit_code"]) is not int
        or raw["launch_intent_receipt_sha256"] != consumed_e.launch_intent.file_sha256
        or raw["process_started_receipt_sha256"] != consumed_e.process_started.file_sha256
        or raw["stdout"] != stdout
        or raw["stderr"] != stderr
        or raw["expected_artifacts"] != artifacts
        or raw["integrity_verifier"] != verifier
        or raw["descendants_after_root_exit"] != 0
        or type(raw["descendants_after_root_exit"]) is not int
        or raw["recovery_evidence"] is not None
        or not _is_utc(raw["ended_at_utc"])
        or raw["ended_at_utc"] < started["started_at_utc"]
        or raw["ended_at_utc"] < verifier["ended_at_utc"]
        or raw["preterminal_pin_contract_sha256"] != context.preterminal_contract.contract_sha256
        or raw["preterminal_overlap_handshake_contract_sha256"]
        != context.overlap_contract.contract_sha256
        or raw["environment_binding"] != environment
        or raw["capsule_lease_identity"] != context.spec["capsule_lease_identity"]
        or raw["capsule_lease_identity_root_sha256"]
        != context.spec["capsule_lease_identity_root_sha256"]
        or raw["capsule_ancestor_lease"] != context.spec["capsule_ancestor_lease"]
        or raw["capsule_ancestor_lease_root_sha256"]
        != context.spec["capsule_ancestor_lease_root_sha256"]
        or raw["python_lease_identity"] != context.spec["python_lease_identity"]
        or raw["python_lease_identity_root_sha256"]
        != context.spec["python_lease_identity_root_sha256"]
        or raw["python_ancestor_lease"] != context.spec["python_ancestor_lease"]
        or raw["python_ancestor_lease_root_sha256"]
        != context.spec["python_ancestor_lease_root_sha256"]
        or raw["python_runtime_resolution_policy"]
        != context.spec["python_runtime_resolution_policy"]
        or raw["runtime_python_lease_identity"] != context.spec["runtime_python_lease_identity"]
        or raw["runtime_python_lease_identity_root_sha256"]
        != context.spec["runtime_python_lease_identity_root_sha256"]
        or raw["runtime_python_ancestor_lease"] != context.spec["runtime_python_ancestor_lease"]
        or raw["runtime_python_ancestor_lease_root_sha256"]
        != context.spec["runtime_python_ancestor_lease_root_sha256"]
        or raw["e_consumption_contract_sha256"]
        != context.e["e_consumption_contract"]["contract_sha256"]
    ):
        raise OriginalConfirmatoryTerminalError(
            "supervisor terminal receipt is not exact success T"
        )
    consumed_e.revalidate()
    return raw


def _build_composed_receipt(
    context: _VerifiedContext,
    *,
    consumed_e: _ConsumedEClaim,
    claim_ready: Mapping[str, Any],
    custody_grant: Mapping[str, Any],
    launch_intent_file: _HeldFile,
    launch_intent: Mapping[str, Any],
    launch_intent_child_handle_slot: int,
    launch_intent_child_granted_access: int,
    lease_file_sha256: str,
    lease: Mapping[str, Any],
    source_files: Mapping[str, _HeldFile],
) -> dict[str, Any]:
    launch_intent_identity = custody_grant["terminal_client_launch_intent_physical_identity"]
    _revalidate_terminal_client_launch_intent(
        context,
        held=launch_intent_file,
        intent=launch_intent,
        custody_grant=custody_grant,
        child_handle_slot=launch_intent_child_handle_slot,
        child_granted_access=launch_intent_child_granted_access,
    )
    pin_held = source_files["preterminal-pin"]
    terminal_held = source_files["supervisor-terminal"]
    pin = _validate_preterminal_pin_payload(
        authority.decode_canonical_json_line(
            pin_held.payload,
            role="retained preterminal pin",
        ),
        context=context,
        consumed_e=consumed_e,
    )
    terminal_envelope = _decode_canonical_line(
        terminal_held.payload,
        role="retained supervisor terminal",
        maximum_bytes=_MAX_CONTROL_BYTES,
    )
    if (
        context.terminal_launcher_command is None
        or terminal_held.file_sha256 != context.terminal_launcher_command["terminal_receipt_sha256"]
        or set(terminal_envelope) != {"schema_version", "payload", "payload_sha256"}
    ):
        raise OriginalConfirmatoryTerminalError("supervisor terminal envelope fields differ")
    bindings = cast(list[dict[str, Any]], lease["retained_input_bindings"])
    terminal = _validate_supervisor_terminal(
        terminal_envelope["payload"],
        context=context,
        consumed_e=consumed_e,
        pin=pin,
        source_files=source_files,
        bindings=bindings,
    )
    if terminal_envelope["payload_sha256"] != _canonical_sha256(terminal):
        raise OriginalConfirmatoryTerminalError("supervisor terminal payload hash differs")
    source_roots = {
        role: {
            "file_sha256": source_files[role].file_sha256,
            "physical_identity_root_sha256": _canonical_sha256(
                bindings[index]["physical_identity"]
            ),
        }
        for index, role in enumerate(_RETAINED_BINDING_ROLES)
    }
    source_inputs_root = _canonical_sha256(
        {
            "postwake_input_lease_file_sha256": lease_file_sha256,
            "postwake_input_lease_receipt_root_sha256": lease["receipt_root_sha256"],
            "terminal_client_launch_intent": {
                "file_sha256": launch_intent_file.file_sha256,
                "intent_root_sha256": launch_intent["intent_root_sha256"],
                "physical_identity_root_sha256": custody_grant[
                    "terminal_client_launch_intent_physical_identity_root_sha256"
                ],
            },
            "source_roots": source_roots,
        }
    )
    unsigned = {
        "schema_version": 1,
        "policy": COMPOSED_TERMINAL_RECEIPT_POLICY,
        "status": "passed",
        "job_id": context.job_id,
        "attempt_id": context.attempt_id,
        "run_id": context.run_id,
        "execution_mode": context.execution_mode,
        "retry_of_run_id": context.retry_of_run_id,
        "attempt_nonce": context.attempt_nonce,
        "q_authority_root_sha256": context.q["q_authority_root_sha256"],
        "e_intent_path": str(context.e_file.path),
        "e_intent_file_sha256": context.e_file.file_sha256,
        "e_intent_core_sha256": context.e["intent_core_sha256"],
        "e_consumption_claim_path": str(consumed_e.held.path),
        "e_consumption_claim_file_sha256": consumed_e.held.file_sha256,
        "e_consumption_claim_root_sha256": consumed_e.claim["claim_root_sha256"],
        "supervisor_spec_sha256": context.spec_sha256,
        "canonical_spec_sha256": context.run_spec_payload["canonical_spec_sha256"],
        "capsule_contract_sha256": context.q["execution_capsule"]["contract_sha256"],
        "capsule_sha256": context.q["execution_capsule"]["sha256"],
        "capsule_internal_manifest_sha256": context.q["execution_capsule"][
            "internal_manifest_sha256"
        ],
        "terminal_command_sha256": context.terminal_command.command_sha256,
        "preterminal_pin_contract_sha256": context.preterminal_contract.contract_sha256,
        "preterminal_overlap_handshake_contract_sha256": (context.overlap_contract.contract_sha256),
        "postwake_input_lease_contract_sha256": (context.input_lease_contract.contract_sha256),
        "postwake_custody_seed_sha256": context.custody_seed.seed_sha256,
        "postwake_custody_handshake_contract_sha256": (context.custody_contract.contract_sha256),
        "terminal_composition_contract_sha256": context.terminal_contract.contract_sha256,
        "expected_integrity_verifier_environment_sha256": (
            context.terminal_contract.exact_integrity_verifier_environment_sha256
        ),
        "observed_integrity_verifier_environment_sha256": (context.observed_environment_sha256),
        "preterminal_pin_file_sha256": pin_held.file_sha256,
        "preterminal_pin_evidence_root_sha256": pin["evidence_root_sha256"],
        "preterminal_scientific_core_sha256": pin["preterminal_scientific_core_sha256"],
        "supervisor_terminal_file_sha256": terminal_held.file_sha256,
        "supervisor_terminal_payload_sha256": terminal_envelope["payload_sha256"],
        "supervisor_terminal_kind": terminal["terminal_kind"],
        "supervisor_terminal_reason": terminal["reason"],
        "supervisor_terminal_exit_code": terminal["exit_code"],
        "postwake_input_lease_file_sha256": lease_file_sha256,
        "postwake_input_lease_receipt_root_sha256": lease["receipt_root_sha256"],
        "source_input_bindings": bindings,
        "source_input_bindings_root_sha256": lease["retained_input_bindings_root_sha256"],
        "source_inputs_root_sha256": source_inputs_root,
        "terminal_client_launch_intent_path": custody_grant["terminal_client_launch_intent_path"],
        "terminal_client_launch_intent_policy": custody_grant[
            "terminal_client_launch_intent_policy"
        ],
        "terminal_client_launch_intent_file_sha256": launch_intent_file.file_sha256,
        "terminal_client_launch_intent_root_sha256": launch_intent["intent_root_sha256"],
        "terminal_client_launch_intent_physical_identity": dict(launch_intent_identity),
        "terminal_client_launch_intent_physical_identity_root_sha256": (
            custody_grant["terminal_client_launch_intent_physical_identity_root_sha256"]
        ),
        "terminal_client_launch_intent_supervisor_handle_slot": custody_grant[
            "terminal_client_launch_intent_supervisor_handle_slot"
        ],
        "terminal_client_launch_intent_child_handle_slot": (launch_intent_child_handle_slot),
        "terminal_client_launch_intent_child_granted_access_mask": (
            launch_intent_child_granted_access
        ),
        "terminal_client_launch_intent_same_duplicated_supervisor_handle_used": True,
        "terminal_client_launch_intent_physical_identity_exact_match": True,
        "terminal_client_launch_intent_child_custody_active": True,
        "terminal_client_launch_intent_supervisor_custody_active": True,
        "claim_ready_sha256": claim_ready["claim_ready_sha256"],
        "custody_grant_sha256": custody_grant["custody_grant_sha256"],
        "same_held_create_new_handle_used": True,
        "supervisor_custody_active": True,
        "semantic_outcome_read_scope": SEMANTIC_OUTCOME_READ_SCOPE,
        "outcome_values_read": False,
        "outcome_values_emitted": False,
        "outcome_values_used_for_selection_or_tuning": False,
        "training_or_model_selection_allowed": False,
        "scientific_publication_allowed": False,
        "automatic_retry_allowed": False,
        "created_at_utc": _utc_now(),
    }
    receipt = {**unsigned, "receipt_root_sha256": _canonical_sha256(unsigned)}
    if set(receipt) != _COMPOSED_RECEIPT_FIELDS:
        raise OriginalConfirmatoryTerminalError("internal composed receipt fields differ")
    consumed_e.revalidate()
    return receipt


def _build_composed_ready(
    context: _VerifiedContext,
    *,
    claim_ready: Mapping[str, Any],
    custody_grant: Mapping[str, Any],
    receipt: Mapping[str, Any],
    composed_identity: Mapping[str, Any],
    target_handle: int,
) -> dict[str, Any]:
    unsigned = {
        "schema_version": 1,
        "policy": COMPOSED_READY_POLICY,
        "message_type": COMPOSED_READY_MESSAGE_TYPE,
        "job_id": context.job_id,
        "attempt_id": context.attempt_id,
        "run_id": context.run_id,
        "execution_mode": context.execution_mode,
        "retry_of_run_id": context.retry_of_run_id,
        "attempt_nonce": context.attempt_nonce,
        "postwake_custody_seed_sha256": context.custody_seed.seed_sha256,
        "postwake_custody_handshake_contract_sha256": (context.custody_contract.contract_sha256),
        "terminal_composition_contract_sha256": context.terminal_contract.contract_sha256,
        "claim_ready_sha256": claim_ready["claim_ready_sha256"],
        "custody_grant_sha256": custody_grant["custody_grant_sha256"],
        "target_supervisor_handle_value": target_handle,
        "composed_terminal_path": str(context.terminal_contract.composed_terminal_receipt_path),
        "composed_terminal_file_sha256": composed_identity["sha256"],
        "composed_terminal_receipt_root_sha256": receipt["receipt_root_sha256"],
        "composed_terminal_physical_identity": dict(composed_identity),
        "composed_terminal_physical_identity_root_sha256": _canonical_sha256(composed_identity),
        "source_inputs_root_sha256": receipt["source_inputs_root_sha256"],
        "terminal_client_launch_intent_path": receipt["terminal_client_launch_intent_path"],
        "terminal_client_launch_intent_policy": receipt["terminal_client_launch_intent_policy"],
        "terminal_client_launch_intent_file_sha256": receipt[
            "terminal_client_launch_intent_file_sha256"
        ],
        "terminal_client_launch_intent_root_sha256": receipt[
            "terminal_client_launch_intent_root_sha256"
        ],
        "terminal_client_launch_intent_physical_identity": receipt[
            "terminal_client_launch_intent_physical_identity"
        ],
        "terminal_client_launch_intent_physical_identity_root_sha256": receipt[
            "terminal_client_launch_intent_physical_identity_root_sha256"
        ],
        "terminal_client_launch_intent_supervisor_handle_slot": receipt[
            "terminal_client_launch_intent_supervisor_handle_slot"
        ],
        "terminal_client_launch_intent_child_handle_slot": receipt[
            "terminal_client_launch_intent_child_handle_slot"
        ],
        "terminal_client_launch_intent_child_granted_access_mask": receipt[
            "terminal_client_launch_intent_child_granted_access_mask"
        ],
        "terminal_client_launch_intent_same_duplicated_supervisor_handle_used": True,
        "terminal_client_launch_intent_physical_identity_exact_match": True,
        "terminal_client_launch_intent_child_custody_active": True,
        "terminal_client_launch_intent_supervisor_custody_active": True,
        "same_held_create_new_handle_used": True,
        "supervisor_custody_active": True,
        "automatic_retry_allowed": False,
    }
    ready = {**unsigned, "composed_ready_sha256": _canonical_sha256(unsigned)}
    return _validate_composed_ready(
        ready,
        context=context,
        claim_ready=claim_ready,
        custody_grant=custody_grant,
        receipt=receipt,
        expected_composed_identity=composed_identity,
    )


def _validate_composed_ready(
    value: Any,
    *,
    context: _VerifiedContext,
    claim_ready: Mapping[str, Any],
    custody_grant: Mapping[str, Any],
    receipt: Mapping[str, Any],
    expected_composed_identity: Mapping[str, Any],
) -> dict[str, Any]:
    ready = _require_mapping(value, role="COMPOSED_READY")
    if set(ready) != _COMPOSED_READY_FIELDS:
        raise OriginalConfirmatoryTerminalError("COMPOSED_READY fields differ")
    composed_identity = authority.canonical_original_confirmatory_physical_file_identity(
        ready["composed_terminal_physical_identity"],
        allowed_roles=("composed-terminal",),
    ).as_dict()
    unsigned = {key: item for key, item in ready.items() if key != "composed_ready_sha256"}
    if (
        ready["schema_version"] != 1
        or ready["policy"] != COMPOSED_READY_POLICY
        or ready["message_type"] != COMPOSED_READY_MESSAGE_TYPE
        or any(
            ready[key] != claim_ready[key]
            for key in (
                "job_id",
                "attempt_id",
                "run_id",
                "execution_mode",
                "retry_of_run_id",
                "attempt_nonce",
                "postwake_custody_seed_sha256",
                "postwake_custody_handshake_contract_sha256",
                "terminal_composition_contract_sha256",
                "claim_ready_sha256",
                "target_supervisor_handle_value",
            )
        )
        or ready["custody_grant_sha256"] != custody_grant["custody_grant_sha256"]
        or ready["composed_terminal_path"]
        != str(context.terminal_contract.composed_terminal_receipt_path)
        or ready["composed_terminal_file_sha256"] != composed_identity["sha256"]
        or ready["composed_terminal_file_sha256"] != expected_composed_identity["sha256"]
        or ready["composed_terminal_receipt_root_sha256"] != receipt["receipt_root_sha256"]
        or ready["composed_terminal_physical_identity"] != composed_identity
        or composed_identity != dict(expected_composed_identity)
        or ready["composed_terminal_physical_identity_root_sha256"]
        != _canonical_sha256(composed_identity)
        or ready["source_inputs_root_sha256"] != receipt["source_inputs_root_sha256"]
        or any(
            ready[key] != receipt[key]
            for key in (
                "terminal_client_launch_intent_path",
                "terminal_client_launch_intent_policy",
                "terminal_client_launch_intent_file_sha256",
                "terminal_client_launch_intent_root_sha256",
                "terminal_client_launch_intent_physical_identity",
                "terminal_client_launch_intent_physical_identity_root_sha256",
                "terminal_client_launch_intent_supervisor_handle_slot",
                "terminal_client_launch_intent_child_handle_slot",
                "terminal_client_launch_intent_child_granted_access_mask",
            )
        )
        or any(
            ready[key] != custody_grant[key]
            for key in (
                "terminal_client_launch_intent_path",
                "terminal_client_launch_intent_policy",
                "terminal_client_launch_intent_file_sha256",
                "terminal_client_launch_intent_root_sha256",
                "terminal_client_launch_intent_physical_identity",
                "terminal_client_launch_intent_physical_identity_root_sha256",
                "terminal_client_launch_intent_supervisor_handle_slot",
            )
        )
        or type(ready["terminal_client_launch_intent_child_handle_slot"]) is not int
        or ready["terminal_client_launch_intent_child_handle_slot"] <= 0
        or ready["terminal_client_launch_intent_child_granted_access_mask"]
        != FILE_GENERIC_READ_ACCESS_MASK
        or type(ready["terminal_client_launch_intent_child_granted_access_mask"]) is not int
        or any(
            ready[key] is not True
            for key in (
                "terminal_client_launch_intent_same_duplicated_supervisor_handle_used",
                "terminal_client_launch_intent_physical_identity_exact_match",
                "terminal_client_launch_intent_child_custody_active",
                "terminal_client_launch_intent_supervisor_custody_active",
            )
        )
        or ready["same_held_create_new_handle_used"] is not True
        or ready["supervisor_custody_active"] is not True
        or ready["automatic_retry_allowed"] is not False
        or ready["composed_ready_sha256"] != _canonical_sha256(unsigned)
    ):
        raise OriginalConfirmatoryTerminalError("COMPOSED_READY violates its exact policy")
    return ready


def _build_readback_receipt(
    context: _VerifiedContext,
    *,
    composed_ready: Mapping[str, Any],
    supervisor_rehashed_composed_identity: Mapping[str, Any],
    created_at_utc: str,
) -> dict[str, Any]:
    observed_identity = authority.canonical_original_confirmatory_physical_file_identity(
        supervisor_rehashed_composed_identity,
        allowed_roles=("composed-terminal",),
    ).as_dict()
    unsigned = {
        "schema_version": 1,
        "policy": POSTWAKE_COMPOSED_READBACK_RECEIPT_POLICY,
        "status": "passed",
        "job_id": context.job_id,
        "attempt_id": context.attempt_id,
        "run_id": context.run_id,
        "execution_mode": context.execution_mode,
        "retry_of_run_id": context.retry_of_run_id,
        "attempt_nonce": context.attempt_nonce,
        "postwake_custody_seed_sha256": context.custody_seed.seed_sha256,
        "postwake_custody_handshake_contract_sha256": (context.custody_contract.contract_sha256),
        "terminal_composition_contract_sha256": (context.terminal_contract.contract_sha256),
        "claim_ready_sha256": composed_ready["claim_ready_sha256"],
        "custody_grant_sha256": composed_ready["custody_grant_sha256"],
        "composed_ready_sha256": composed_ready["composed_ready_sha256"],
        "target_supervisor_handle_value": composed_ready["target_supervisor_handle_value"],
        "composed_terminal_path": composed_ready["composed_terminal_path"],
        "composed_terminal_file_sha256": composed_ready["composed_terminal_file_sha256"],
        "composed_terminal_receipt_root_sha256": composed_ready[
            "composed_terminal_receipt_root_sha256"
        ],
        "supervisor_rehashed_composed_identity": observed_identity,
        "supervisor_rehashed_composed_identity_root_sha256": (_canonical_sha256(observed_identity)),
        "terminal_client_launch_intent_path": composed_ready["terminal_client_launch_intent_path"],
        "terminal_client_launch_intent_policy": composed_ready[
            "terminal_client_launch_intent_policy"
        ],
        "terminal_client_launch_intent_file_sha256": composed_ready[
            "terminal_client_launch_intent_file_sha256"
        ],
        "terminal_client_launch_intent_root_sha256": composed_ready[
            "terminal_client_launch_intent_root_sha256"
        ],
        "terminal_client_launch_intent_physical_identity": composed_ready[
            "terminal_client_launch_intent_physical_identity"
        ],
        "terminal_client_launch_intent_physical_identity_root_sha256": composed_ready[
            "terminal_client_launch_intent_physical_identity_root_sha256"
        ],
        "terminal_client_launch_intent_supervisor_handle_slot": composed_ready[
            "terminal_client_launch_intent_supervisor_handle_slot"
        ],
        "terminal_client_launch_intent_supervisor_custody_retained_through_readback": True,
        "identity_exact_match": True,
        "same_duplicated_supervisor_handle_used": True,
        "readback_created_with_create_new": True,
        "readback_same_held_handle_used": True,
        "supervisor_custody_retained_through_readback": True,
        "automatic_retry_allowed": False,
        "created_at_utc": created_at_utc,
    }
    receipt = {
        **unsigned,
        "receipt_root_sha256": _canonical_sha256(unsigned),
    }
    return _validate_readback_payload(
        receipt,
        context=context,
        composed_ready=composed_ready,
    )


def _validate_readback_payload(
    value: Any,
    *,
    context: _VerifiedContext,
    composed_ready: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _require_mapping(value, role="postwake composed readback receipt")
    if set(raw) != _POSTWAKE_COMPOSED_READBACK_FIELDS:
        raise OriginalConfirmatoryTerminalError("R fields differ")
    unsigned = {key: item for key, item in raw.items() if key != "receipt_root_sha256"}
    observed_identity = authority.canonical_original_confirmatory_physical_file_identity(
        raw["supervisor_rehashed_composed_identity"],
        allowed_roles=("composed-terminal",),
    ).as_dict()
    if (
        raw["schema_version"] != 1
        or raw["policy"] != POSTWAKE_COMPOSED_READBACK_RECEIPT_POLICY
        or raw["status"] != "passed"
        or raw["job_id"] != context.job_id
        or raw["attempt_id"] != context.attempt_id
        or raw["run_id"] != context.run_id
        or raw["execution_mode"] != context.execution_mode
        or raw["retry_of_run_id"] != context.retry_of_run_id
        or raw["attempt_nonce"] != context.attempt_nonce
        or raw["postwake_custody_seed_sha256"] != context.custody_seed.seed_sha256
        or raw["postwake_custody_handshake_contract_sha256"]
        != context.custody_contract.contract_sha256
        or raw["terminal_composition_contract_sha256"] != context.terminal_contract.contract_sha256
        or any(
            raw[key] != composed_ready[key]
            for key in (
                "claim_ready_sha256",
                "custody_grant_sha256",
                "composed_ready_sha256",
                "target_supervisor_handle_value",
                "composed_terminal_path",
                "composed_terminal_file_sha256",
                "composed_terminal_receipt_root_sha256",
                "terminal_client_launch_intent_path",
                "terminal_client_launch_intent_policy",
                "terminal_client_launch_intent_file_sha256",
                "terminal_client_launch_intent_root_sha256",
                "terminal_client_launch_intent_physical_identity",
                "terminal_client_launch_intent_physical_identity_root_sha256",
                "terminal_client_launch_intent_supervisor_handle_slot",
            )
        )
        or observed_identity != composed_ready["composed_terminal_physical_identity"]
        or raw["supervisor_rehashed_composed_identity"] != observed_identity
        or raw["supervisor_rehashed_composed_identity_root_sha256"]
        != _canonical_sha256(observed_identity)
        or raw["terminal_client_launch_intent_supervisor_custody_retained_through_readback"]
        is not True
        or raw["identity_exact_match"] is not True
        or raw["same_duplicated_supervisor_handle_used"] is not True
        or raw["readback_created_with_create_new"] is not True
        or raw["readback_same_held_handle_used"] is not True
        or raw["supervisor_custody_retained_through_readback"] is not True
        or raw["automatic_retry_allowed"] is not False
        or not _is_utc(raw["created_at_utc"])
        or raw["receipt_root_sha256"] != _canonical_sha256(unsigned)
    ):
        raise OriginalConfirmatoryTerminalError("R violates its exact retained readback policy")
    return raw


def _build_final_ack(
    context: _VerifiedContext,
    *,
    claim_ready: Mapping[str, Any],
    custody_grant: Mapping[str, Any],
    composed_ready: Mapping[str, Any],
    readback_receipt: Mapping[str, Any],
    readback_file_sha256: str,
    acknowledged_at_utc: str,
) -> dict[str, Any]:
    observed_identity = readback_receipt["supervisor_rehashed_composed_identity"]
    unsigned = {
        "schema_version": 1,
        "policy": COMPOSED_FINAL_ACK_POLICY,
        "message_type": FINAL_ACK_MESSAGE_TYPE,
        "job_id": context.job_id,
        "attempt_id": context.attempt_id,
        "run_id": context.run_id,
        "execution_mode": context.execution_mode,
        "retry_of_run_id": context.retry_of_run_id,
        "attempt_nonce": context.attempt_nonce,
        "postwake_custody_seed_sha256": context.custody_seed.seed_sha256,
        "postwake_custody_handshake_contract_sha256": (context.custody_contract.contract_sha256),
        "terminal_composition_contract_sha256": (context.terminal_contract.contract_sha256),
        "claim_ready_sha256": claim_ready["claim_ready_sha256"],
        "custody_grant_sha256": custody_grant["custody_grant_sha256"],
        "composed_ready_sha256": composed_ready["composed_ready_sha256"],
        "target_supervisor_handle_value": composed_ready["target_supervisor_handle_value"],
        "composed_terminal_file_sha256": composed_ready["composed_terminal_file_sha256"],
        "composed_terminal_receipt_root_sha256": composed_ready[
            "composed_terminal_receipt_root_sha256"
        ],
        "supervisor_rehashed_composed_identity": observed_identity,
        "supervisor_rehashed_composed_identity_root_sha256": (
            readback_receipt["supervisor_rehashed_composed_identity_root_sha256"]
        ),
        "identity_exact_match": True,
        "postwake_composed_readback_receipt_path": str(
            context.custody_contract.readback_receipt_path
        ),
        "postwake_composed_readback_receipt_file_sha256": (
            _require_sha256(readback_file_sha256, role="R file")
        ),
        "postwake_composed_readback_receipt_root_sha256": (readback_receipt["receipt_root_sha256"]),
        "terminal_client_launch_intent_path": composed_ready["terminal_client_launch_intent_path"],
        "terminal_client_launch_intent_policy": composed_ready[
            "terminal_client_launch_intent_policy"
        ],
        "terminal_client_launch_intent_file_sha256": composed_ready[
            "terminal_client_launch_intent_file_sha256"
        ],
        "terminal_client_launch_intent_root_sha256": composed_ready[
            "terminal_client_launch_intent_root_sha256"
        ],
        "terminal_client_launch_intent_physical_identity": composed_ready[
            "terminal_client_launch_intent_physical_identity"
        ],
        "terminal_client_launch_intent_physical_identity_root_sha256": composed_ready[
            "terminal_client_launch_intent_physical_identity_root_sha256"
        ],
        "terminal_client_launch_intent_supervisor_handle_slot": composed_ready[
            "terminal_client_launch_intent_supervisor_handle_slot"
        ],
        "terminal_client_launch_intent_supervisor_custody_retained_through_ack": True,
        "launcher_redirector_child_process_handles_retained_through_ack": True,
        "immediate_venv_redirector_process_identity_reverified": True,
        "terminal_client_launcher_process_identity_reverified": True,
        "launcher_redirector_child_grandparent_chain_reverified": True,
        "launcher_redirector_child_same_supervisor_job_reverified": True,
        "immediate_venv_redirector_live_at_final_ack": True,
        "terminal_client_launcher_live_at_final_ack": True,
        "supervisor_custody_retained_through_ack": True,
        "automatic_retry_allowed": False,
        "acknowledged_at_utc": acknowledged_at_utc,
    }
    ack = {
        **unsigned,
        "final_ack_sha256": _canonical_sha256(unsigned),
    }
    return _validate_final_ack(
        ack,
        context=context,
        claim_ready=claim_ready,
        custody_grant=custody_grant,
        composed_ready=composed_ready,
        composed_identity=cast(
            Mapping[str, Any],
            composed_ready["composed_terminal_physical_identity"],
        ),
    )


def _validate_final_ack(
    value: Any,
    *,
    context: _VerifiedContext,
    claim_ready: Mapping[str, Any],
    custody_grant: Mapping[str, Any],
    composed_ready: Mapping[str, Any],
    composed_identity: Mapping[str, Any],
) -> dict[str, Any]:
    ack = _require_mapping(value, role="FINAL_ACK")
    if set(ack) != _FINAL_ACK_FIELDS:
        raise OriginalConfirmatoryTerminalError("FINAL_ACK fields differ")
    unsigned = {key: item for key, item in ack.items() if key != "final_ack_sha256"}
    observed_identity = authority.canonical_original_confirmatory_physical_file_identity(
        ack["supervisor_rehashed_composed_identity"],
        allowed_roles=("composed-terminal",),
    ).as_dict()
    if (
        ack["schema_version"] != 1
        or ack["policy"] != COMPOSED_FINAL_ACK_POLICY
        or ack["message_type"] != FINAL_ACK_MESSAGE_TYPE
        or any(
            ack[key] != composed_ready[key]
            for key in (
                "job_id",
                "attempt_id",
                "run_id",
                "execution_mode",
                "retry_of_run_id",
                "attempt_nonce",
                "postwake_custody_seed_sha256",
                "postwake_custody_handshake_contract_sha256",
                "terminal_composition_contract_sha256",
                "claim_ready_sha256",
                "custody_grant_sha256",
                "composed_ready_sha256",
                "target_supervisor_handle_value",
                "composed_terminal_file_sha256",
                "composed_terminal_receipt_root_sha256",
                "terminal_client_launch_intent_path",
                "terminal_client_launch_intent_policy",
                "terminal_client_launch_intent_file_sha256",
                "terminal_client_launch_intent_root_sha256",
                "terminal_client_launch_intent_physical_identity",
                "terminal_client_launch_intent_physical_identity_root_sha256",
                "terminal_client_launch_intent_supervisor_handle_slot",
            )
        )
        or observed_identity != dict(composed_identity)
        or ack["supervisor_rehashed_composed_identity_root_sha256"]
        != _canonical_sha256(observed_identity)
        or ack["identity_exact_match"] is not True
        or ack["postwake_composed_readback_receipt_path"]
        != str(context.custody_contract.readback_receipt_path)
        or _require_sha256(
            ack["postwake_composed_readback_receipt_file_sha256"],
            role="R file",
        )
        != ack["postwake_composed_readback_receipt_file_sha256"]
        or _require_sha256(
            ack["postwake_composed_readback_receipt_root_sha256"],
            role="R root",
        )
        != ack["postwake_composed_readback_receipt_root_sha256"]
        or any(
            ack[key] is not True
            for key in (
                "terminal_client_launch_intent_supervisor_custody_retained_through_ack",
                "launcher_redirector_child_process_handles_retained_through_ack",
                "immediate_venv_redirector_process_identity_reverified",
                "terminal_client_launcher_process_identity_reverified",
                "launcher_redirector_child_grandparent_chain_reverified",
                "launcher_redirector_child_same_supervisor_job_reverified",
                "immediate_venv_redirector_live_at_final_ack",
                "terminal_client_launcher_live_at_final_ack",
            )
        )
        or ack["supervisor_custody_retained_through_ack"] is not True
        or ack["automatic_retry_allowed"] is not False
        or not _is_utc(ack["acknowledged_at_utc"])
        or ack["final_ack_sha256"] != _canonical_sha256(unsigned)
    ):
        raise OriginalConfirmatoryTerminalError("FINAL_ACK violates its exact policy")
    return ack


def _validate_readback_receipt(
    context: _VerifiedContext,
    *,
    final_ack: Mapping[str, Any],
    composed_ready: Mapping[str, Any],
) -> None:
    held = _open_held_file(
        context.custody_contract.readback_receipt_path,
        role="postwake composed readback receipt",
        maximum_bytes=context.terminal_contract.postwake_composed_readback_receipt_max_bytes,
        allow_live_writer=True,
    )
    try:
        if held.file_sha256 != final_ack["postwake_composed_readback_receipt_file_sha256"]:
            raise OriginalConfirmatoryTerminalError("R file hash differs from FINAL_ACK")
        raw = authority.decode_canonical_json_line(
            held.payload,
            role="postwake composed readback receipt",
        )
        raw = _validate_readback_payload(
            raw,
            context=context,
            composed_ready=composed_ready,
        )
        if (
            raw["supervisor_rehashed_composed_identity"]
            != final_ack["supervisor_rehashed_composed_identity"]
            or raw["receipt_root_sha256"]
            != final_ack["postwake_composed_readback_receipt_root_sha256"]
        ):
            raise OriginalConfirmatoryTerminalError("R violates its exact retained readback policy")
        held.revalidate()
    finally:
        held.close()


def _terminal_summary(
    context: _VerifiedContext,
    *,
    composed_ready: Mapping[str, Any],
    final_ack: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy": COMPOSED_TERMINAL_STDOUT_SUMMARY_POLICY,
        "status": "passed",
        "message_type": FINAL_ACK_MESSAGE_TYPE,
        "job_id": context.job_id,
        "attempt_id": context.attempt_id,
        "run_id": context.run_id,
        "execution_mode": context.execution_mode,
        "retry_of_run_id": context.retry_of_run_id,
        "attempt_nonce": context.attempt_nonce,
        "postwake_custody_seed_sha256": context.custody_seed.seed_sha256,
        "terminal_composition_contract_sha256": context.terminal_contract.contract_sha256,
        "composed_terminal_file_sha256": composed_ready["composed_terminal_file_sha256"],
        "composed_terminal_receipt_root_sha256": composed_ready[
            "composed_terminal_receipt_root_sha256"
        ],
        "postwake_composed_readback_receipt_file_sha256": final_ack[
            "postwake_composed_readback_receipt_file_sha256"
        ],
        "postwake_composed_readback_receipt_root_sha256": final_ack[
            "postwake_composed_readback_receipt_root_sha256"
        ],
        "final_ack_sha256": final_ack["final_ack_sha256"],
        "p_t_l_c_r_complete": True,
        "automatic_retry_allowed": False,
    }


def _run_terminal_duplex(
    context: _VerifiedContext,
    *,
    consumed_e: _ConsumedEClaim,
    stdout: Any,
    pipe_factory: Any = _DuplexPipeClient,
) -> None:
    path = context.terminal_contract.composed_terminal_receipt_path
    descriptor = _create_new_readonly_descriptor(path)
    pipe: _DuplexPipeClient | None = None
    supervisor_process_handle = 0
    source_files: dict[str, _HeldFile] = {}
    launch_intent_file: _HeldFile | None = None
    launch_intent: dict[str, Any] | None = None
    launch_intent_child_handle_slot = 0
    launch_intent_child_granted_access = 0
    lease_descriptor = -1
    try:
        claim_identity = _claim_identity(descriptor, path=path)
        _revalidate_terminal_process_ancestry(context)
        expected_supervisor_identity, boot_time = _supervisor_identity_from_launch(
            context,
            consumed_e=consumed_e,
        )
        pipe = pipe_factory(
            context.custody_contract.pipe_name,
            outbound_maximum_message_bytes=context.custody_contract.ready_max_bytes,
            inbound_maximum_message_bytes=context.custody_contract.ack_max_bytes,
            custody_exchange_timeout_ms=(context.custody_contract.custody_exchange_timeout_ms),
        )
        server_pid = pipe.server_pid()
        if server_pid != expected_supervisor_identity["pid"]:
            raise OriginalConfirmatoryTerminalError(
                "custody pipe PID differs from launch supervisor"
            )
        supervisor_process_handle = _open_process(
            server_pid,
            access=(
                PROCESS_DUP_HANDLE
                | PROCESS_QUERY_LIMITED_INFORMATION
                | PROCESS_VM_READ
                | SYNCHRONIZE
            ),
        )
        observed_supervisor_identity = _process_identity(
            supervisor_process_handle,
            pid=server_pid,
            command_sha256=cast(
                str,
                context.supervisor_process_command_projection["command_sha256"],
            ),
        )
        _validate_live_process_command_view(
            _running_process_command_view(supervisor_process_handle),
            expected_argv=cast(
                list[str],
                context.supervisor_process_command_projection["expected_live_peb_argv"],
            ),
            expected_cwd=cast(
                str,
                context.supervisor_process_command_projection["cwd"],
            ),
            role="custody supervisor",
        )
        if (
            observed_supervisor_identity != expected_supervisor_identity
            or not _same_resolved_path(
                observed_supervisor_identity["program_path"],
                context.supervisor_process_command_projection["expected_live_image_path"],
            )
            or observed_supervisor_identity["program_sha256"]
            != context.supervisor_process_command_projection["expected_live_image_sha256"]
        ):
            raise OriginalConfirmatoryTerminalError(
                "live custody supervisor identity differs from launch evidence"
            )
        if context.terminal_runtime_child_process_identity is None:
            raise OriginalConfirmatoryTerminalError(
                "terminal runtime child identity is unavailable"
            )
        if (
            context.immediate_venv_redirector_process_identity is None
            or context.terminal_client_launcher_process_identity is None
        ):
            raise OriginalConfirmatoryTerminalError(
                "terminal launcher ancestry identities are unavailable"
            )
        client_identity = dict(context.terminal_runtime_child_process_identity)
        target_handle = _duplicate_to_process(
            descriptor,
            target_process_handle=supervisor_process_handle,
            access_mask=GENERIC_READ,
        )
        claim_ready = _build_claim_ready(
            context,
            consumed_e=consumed_e,
            client_identity=client_identity,
            immediate_venv_redirector_identity=(context.immediate_venv_redirector_process_identity),
            terminal_client_launcher_identity=(context.terminal_client_launcher_process_identity),
            supervisor_identity=observed_supervisor_identity,
            claim_identity=claim_identity,
            target_handle=target_handle,
        )
        _revalidate_terminal_process_ancestry(context)
        consumed_e.revalidate()
        pipe.send(claim_ready)
        grant_raw, _grant_line = pipe.receive()
        custody_grant = _validate_custody_grant(
            grant_raw,
            context=context,
            claim_ready=claim_ready,
        )
        _revalidate_terminal_process_ancestry(context)
        (
            launch_intent_file,
            launch_intent,
            launch_intent_child_handle_slot,
            launch_intent_child_granted_access,
        ) = _open_validated_terminal_client_launch_intent(
            context,
            custody_grant=custody_grant,
            supervisor_process_handle=supervisor_process_handle,
        )
        _revalidate_terminal_process_ancestry(context)
        _revalidate_terminal_client_launch_intent(
            context,
            held=launch_intent_file,
            intent=launch_intent,
            custody_grant=custody_grant,
            child_handle_slot=launch_intent_child_handle_slot,
            child_granted_access=launch_intent_child_granted_access,
        )
        lease_descriptor = _duplicate_from_process(
            supervisor_process_handle,
            source_handle_slot=cast(
                int,
                custody_grant["postwake_input_lease_handle_slot"],
            ),
        )
        lease_path = context.input_lease_contract.lease_receipt_path
        lease_payload = _read_descriptor(
            lease_descriptor,
            maximum_bytes=context.terminal_contract.postwake_input_lease_receipt_max_bytes,
        )
        lease_identity = _physical_identity(
            lease_descriptor,
            path=lease_path,
            role="postwake-lease-receipt",
            maximum_bytes=context.terminal_contract.postwake_input_lease_receipt_max_bytes,
        )
        if lease_identity != custody_grant["postwake_input_lease_physical_identity"]:
            raise OriginalConfirmatoryTerminalError("duplicated L differs from CUSTODY_GRANT")

        def require_lease_unchanged(stage: str) -> None:
            if (
                _read_descriptor(
                    lease_descriptor,
                    maximum_bytes=(
                        context.terminal_contract.postwake_input_lease_receipt_max_bytes
                    ),
                )
                != lease_payload
                or _physical_identity(
                    lease_descriptor,
                    path=lease_path,
                    role="postwake-lease-receipt",
                    maximum_bytes=(
                        context.terminal_contract.postwake_input_lease_receipt_max_bytes
                    ),
                )
                != lease_identity
            ):
                raise OriginalConfirmatoryTerminalError(
                    f"postwake input lease changed before {stage}"
                )

        lease = _validate_postwake_lease(
            authority.decode_canonical_json_line(
                lease_payload,
                role="postwake input lease receipt",
            ),
            context=context,
            custody_grant=custody_grant,
            expected_supervisor_identity=observed_supervisor_identity,
            expected_boot_time_utc=boot_time,
        )
        retained_bounds = {
            "preterminal-pin": (context.terminal_contract.preterminal_pin_receipt_max_bytes),
            "preterminal-stdout": (context.terminal_contract.verifier_stdout_max_bytes),
            "preterminal-stderr": (context.terminal_contract.verifier_stderr_max_bytes),
            "supervisor-terminal": (
                context.terminal_contract.supervisor_terminal_receipt_max_bytes
            ),
        }
        for binding in cast(
            list[dict[str, Any]],
            lease["retained_input_bindings"],
        ):
            role = cast(str, binding["role"])
            source_files[role] = _held_from_supervisor_slot(
                supervisor_process_handle,
                binding=binding,
                maximum_bytes=retained_bounds[role],
            )
        receipt = _build_composed_receipt(
            context,
            consumed_e=consumed_e,
            claim_ready=claim_ready,
            custody_grant=custody_grant,
            launch_intent_file=launch_intent_file,
            launch_intent=launch_intent,
            launch_intent_child_handle_slot=launch_intent_child_handle_slot,
            launch_intent_child_granted_access=(launch_intent_child_granted_access),
            lease_file_sha256=_sha256_bytes(lease_payload),
            lease=lease,
            source_files=source_files,
        )
        _revalidate_terminal_client_launch_intent(
            context,
            held=launch_intent_file,
            intent=launch_intent,
            custody_grant=custody_grant,
            child_handle_slot=launch_intent_child_handle_slot,
            child_granted_access=launch_intent_child_granted_access,
        )
        require_lease_unchanged("composed-terminal serialization")
        receipt_bytes = _canonical_bytes(receipt)
        _write_same_handle(
            descriptor,
            receipt_bytes,
            maximum_bytes=context.terminal_contract.composed_terminal_receipt_max_bytes,
        )
        composed_identity = _physical_identity(
            descriptor,
            path=path,
            role="composed-terminal",
            maximum_bytes=context.terminal_contract.composed_terminal_receipt_max_bytes,
        )
        for source in source_files.values():
            source.revalidate()
        _revalidate_terminal_client_launch_intent(
            context,
            held=launch_intent_file,
            intent=launch_intent,
            custody_grant=custody_grant,
            child_handle_slot=launch_intent_child_handle_slot,
            child_granted_access=launch_intent_child_granted_access,
        )
        require_lease_unchanged("COMPOSED_READY")
        context.q_file.revalidate()
        context.e_file.revalidate()
        context.run_spec_file.revalidate()
        _revalidate_terminal_process_ancestry(context)
        consumed_e.revalidate()
        composed_ready = _build_composed_ready(
            context,
            claim_ready=claim_ready,
            custody_grant=custody_grant,
            receipt=receipt,
            composed_identity=composed_identity,
            target_handle=target_handle,
        )
        pipe.send(composed_ready)
        final_ack_raw, _final_ack_line = pipe.receive()
        _revalidate_terminal_client_launch_intent(
            context,
            held=launch_intent_file,
            intent=launch_intent,
            custody_grant=custody_grant,
            child_handle_slot=launch_intent_child_handle_slot,
            child_granted_access=launch_intent_child_granted_access,
        )
        _revalidate_terminal_process_ancestry(context)
        final_ack = _validate_final_ack(
            final_ack_raw,
            context=context,
            claim_ready=claim_ready,
            custody_grant=custody_grant,
            composed_ready=composed_ready,
            composed_identity=composed_identity,
        )
        _validate_readback_receipt(
            context,
            final_ack=final_ack,
            composed_ready=composed_ready,
        )
        if (
            _read_descriptor(
                descriptor,
                maximum_bytes=context.terminal_contract.composed_terminal_receipt_max_bytes,
            )
            != receipt_bytes
            or _physical_identity(
                descriptor,
                path=path,
                role="composed-terminal",
                maximum_bytes=(context.terminal_contract.composed_terminal_receipt_max_bytes),
            )
            != composed_identity
        ):
            raise OriginalConfirmatoryTerminalError("C changed before FINAL_ACK completion")
        for source in source_files.values():
            source.revalidate()
        _revalidate_terminal_client_launch_intent(
            context,
            held=launch_intent_file,
            intent=launch_intent,
            custody_grant=custody_grant,
            child_handle_slot=launch_intent_child_handle_slot,
            child_granted_access=launch_intent_child_granted_access,
        )
        require_lease_unchanged("FINAL_ACK completion")
        _revalidate_terminal_process_ancestry(context)
        consumed_e.revalidate()
        pipe.require_deadline_active()
        stdout.buffer.write(
            _canonical_bytes(
                _terminal_summary(
                    context,
                    composed_ready=composed_ready,
                    final_ack=final_ack,
                )
            )
        )
        stdout.buffer.flush()
    finally:
        cleanup_errors: list[BaseException] = []
        for held in reversed(tuple(source_files.values())):
            try:
                held.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if launch_intent_file is not None:
            try:
                launch_intent_file.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if lease_descriptor >= 0:
            try:
                os.close(lease_descriptor)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if supervisor_process_handle:
            try:
                _close_native_handle(supervisor_process_handle)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if pipe is not None:
            try:
                pipe.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        try:
            os.close(descriptor)
        except BaseException as exc:
            cleanup_errors.append(exc)
        if cleanup_errors:
            raise OriginalConfirmatoryTerminalError(
                "terminal custody cleanup failed"
            ) from cleanup_errors[0]


def _verify_original_confirmatory_terminal_from_canonical_tail(
    canonical_tail: tuple[str, ...],
) -> int:
    """Fixed ``verify-terminal`` handler selected by the capsule dispatcher."""

    context: _VerifiedContext | None = None
    consumed_e: _ConsumedEClaim | None = None
    try:
        context, consumed_e = _load_context_then_take_consumed_e(
            authority.CAPSULE_TERMINAL_MODE,
            canonical_tail,
        )
        _run_terminal_duplex(
            context,
            consumed_e=consumed_e,
            stdout=sys.stdout,
        )
        return 0
    except BaseException:
        return TERMINAL_HANDLER_PROTOCOL_EXIT_CODE
    finally:
        if consumed_e is not None:
            try:
                consumed_e.close()
            except BaseException:
                return TERMINAL_HANDLER_STOP_EXIT_CODE
        if context is not None:
            try:
                context.close()
            except BaseException:
                return TERMINAL_HANDLER_STOP_EXIT_CODE


__all__ = [
    "_verify_original_confirmatory_preterminal_from_canonical_tail",
    "_verify_original_confirmatory_terminal_from_canonical_tail",
]
