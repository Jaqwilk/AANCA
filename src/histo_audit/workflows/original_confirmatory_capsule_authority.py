"""Closed Q/E contracts for the original-confirmatory sealed execution capsule.

This module deliberately has no dependency on the retired in-process
capability/lease/terminal-pin chain.  It authenticates one deterministic PYZ,
one exact two-map launch environment, one concrete capsule command, and one
preterminal-pin verifier contract.  Scientific execution remains a separate
fresh process.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import ntpath
import os
import re
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any, Final, cast

SUPERVISOR_ATTEMPT_NONCE_KEY: Final = "AANCA_SUPERVISOR_ATTEMPT_NONCE"
EXPECTED_LAUNCH_ENVIRONMENT_ENVELOPE_POLICY: Final = (
    "expected_original_confirmatory_launch_environment_envelope_v1"
)
SCIENCE_SUPERVISOR_ENVIRONMENT_NAMES: Final = (
    "LOCALAPPDATA",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
)
PROCESS_ENVIRONMENT_BINDING_POLICY: Final = "aanca_three_process_environment_binding_v1"
EXECUTION_CAPSULE_POLICY: Final = "original_confirmatory_sealed_execution_capsule_v1"
PRETERMINAL_PIN_CONTRACT_POLICY: Final = (
    "original_confirmatory_capsule_preterminal_pin_contract_v1"
)
PRETERMINAL_PIN_PUBLICATION_POLICY: Final = "create_new_o_excl_single_link_read_only_v1"
PRETERMINAL_PIN_STDOUT_SUMMARY_POLICY: Final = (
    "original_confirmatory_preterminal_pin_summary_v1"
)
TERMINAL_COMPOSITION_CONTRACT_POLICY: Final = (
    "original_confirmatory_capsule_terminal_composition_contract_v1"
)
TERMINAL_CUSTODY_AUTHORITY_PROJECTION_POLICY: Final = (
    "original_confirmatory_terminal_custody_authority_projection_v1"
)
TERMINAL_CUSTODY_AUTHORITY_TEMPLATE_POLICY: Final = (
    "original_confirmatory_terminal_custody_authority_template_v1"
)
OUTCOME_BLIND_EXPECTED_ARTIFACT_PROJECTION_POLICY: Final = (
    "original_confirmatory_outcome_blind_expected_artifact_projection_v1"
)
OUTCOME_BLIND_EXPECTED_ARTIFACT_INSTANCE_POLICY: Final = (
    "original_confirmatory_outcome_blind_expected_artifact_instance_v1"
)
COMPOSED_CLAIM_READY_POLICY: Final = (
    "original_confirmatory_composed_terminal_claim_ready_v1"
)
COMPOSED_CLAIM_PHYSICAL_IDENTITY_POLICY: Final = (
    "original_confirmatory_composed_terminal_claim_physical_identity_v1"
)
COMPOSED_CUSTODY_GRANT_POLICY: Final = (
    "original_confirmatory_composed_terminal_custody_grant_v1"
)
COMPOSED_READY_POLICY: Final = "original_confirmatory_composed_terminal_ready_v1"
COMPOSED_FINAL_ACK_POLICY: Final = (
    "original_confirmatory_composed_terminal_final_ack_v1"
)
CLAIM_READY_MESSAGE_TYPE: Final = "CLAIM_READY"
CUSTODY_GRANT_MESSAGE_TYPE: Final = "CUSTODY_GRANT"
COMPOSED_READY_MESSAGE_TYPE: Final = "COMPOSED_READY"
FINAL_ACK_MESSAGE_TYPE: Final = "FINAL_ACK"
GENERIC_READ_ACCESS_REQUEST: Final = 0x80000000
GENERIC_WRITE_ACCESS_REQUEST: Final = 0x40000000
FILE_GENERIC_READ_ACCESS_MASK: Final = 0x00120089
FILE_SHARE_READ: Final = 0x00000001
FILE_SHARE_WRITE: Final = 0x00000002
TERMINAL_CUSTODY_OUTBOUND_MAX_BYTES: Final = 64 * 1024
TERMINAL_CUSTODY_INBOUND_MAX_BYTES: Final = 64 * 1024
TERMINAL_CLIENT_ARRIVAL_TIMEOUT_MS: Final = 1_800_000
CUSTODY_EXCHANGE_TIMEOUT_MS: Final = 60_000
TERMINAL_CUSTODY_OVERALL_TIMEOUT_MAX_MS: Final = 6 * 60 * 60 * 1_000
TERMINAL_CLIENT_LAUNCHER_POLICY: Final = (
    "original_confirmatory_terminal_client_launcher_v1"
)
TERMINAL_CLIENT_LAUNCHER_ANCESTOR_LEASE_POLICY: Final = (
    "original_confirmatory_terminal_client_launcher_ancestor_lease_v1"
)
TERMINAL_CLIENT_LAUNCHER_E_PROJECTION_POLICY: Final = (
    "original_confirmatory_terminal_client_launcher_e_projection_v1"
)
TERMINAL_CLIENT_LAUNCHER_COMMAND_POLICY: Final = (
    "original_confirmatory_terminal_client_launcher_command_v1"
)
TERMINAL_CLIENT_LAUNCH_INTENT_POLICY: Final = (
    "original_confirmatory_terminal_client_launch_intent_create_new_v1"
)
TERMINAL_CLIENT_LAUNCH_INTENT_FILENAME: Final = "terminal_client_launch_intent.json"
TERMINAL_CLIENT_LAUNCHER_SOURCE_FILENAME: Final = "terminal_client_launcher_v1.py"
TERMINAL_CLIENT_PYTHON_ISOLATED_FLAGS: Final = ("-I", "-S", "-B")
TERMINAL_CLIENT_CHILD_STDIO_POLICY: Final = (
    "fresh_restricted_three_handle_anonymous_pipe_stdio_v1"
)
CODEX_TERMINAL_WAKE_PROMPT_RENDER_POLICY: Final = (
    "original_confirmatory_codex_terminal_wake_prompt_render_v1"
)
CODEX_TERMINAL_WAKE_PROMPT_PLACEHOLDER_ORDER: Final = (
    "job_id",
    "supervisor_job_directory",
    "supervisor_spec_path",
    "supervisor_spec_sha256",
    "terminal_receipt_sha256",
    "terminal_client_launcher_argv_json",
    "terminal_client_launcher_command_sha256",
    "verify_terminal_command_sha256",
)
CODEX_TERMINAL_WAKE_PROMPT_TEMPLATE: Final = (
    "Read STATUS.md and PLAN.md completely as policy context. Do not select, discover, or "
    "reconstruct a job or command from either file.\n"
    "This wake is bound to job_id={job_id}, "
    "supervisor_job_directory={supervisor_job_directory}, "
    "supervisor_spec_path={supervisor_spec_path}, "
    "supervisor_spec_sha256={supervisor_spec_sha256}, "
    "terminal_receipt_sha256={terminal_receipt_sha256}.\n"
    "Your first executable action after reading those two files must launch exactly one "
    "sealed terminal-client launcher using this canonical process argv JSON, unchanged "
    "and in order:\n"
    "{terminal_client_launcher_argv_json}\n"
    "The canonical terminal-client launcher command SHA-256 is "
    "{terminal_client_launcher_command_sha256}.\n"
    "That launcher is authorized to launch exactly one new verify-terminal child using "
    "the already-sealed command whose canonical SHA-256 is "
    "{verify_terminal_command_sha256}. Do not launch verify-terminal directly.\n"
    "Do not use globbing, --last, autodiscovery, fallback, retry, an alternate job, an "
    "alternate launcher, or an alternate command. If the exact launcher, child launch, "
    "custody handshake, terminal verification, seal, or integrity check fails or is "
    "ambiguous, perform diagnosis only and do not retry.\n"
    "After the exact launcher returns success, verify the composed receipt and the bound "
    "seal and integrity evidence without reading scientific outcome values for tuning. "
    "Continue only the legal next short step. Stop this Codex session after all short "
    "steps finish or after one separately authorized next long process is launched and "
    "its event-driven supervisor is armed. Never retry an experiment, publication, or "
    "one-use authority.\n"
)
SUPERVISOR_PROCESS_COMMAND_DERIVATION_POLICY: Final = (
    "original_confirmatory_supervisor_process_command_derivation_v2"
)
SUPERVISOR_PROCESS_COMMAND_POLICY: Final = (
    "original_confirmatory_supervisor_process_command_v2"
)
SUPERVISOR_PROCESS_COMMAND_PROJECTION_POLICY: Final = (
    "original_confirmatory_supervisor_process_command_projection_v1"
)
SUPERVISOR_PROCESS_COMMAND_HASH_POLICY: Final = (
    "canonical_compact_sorted_json_sha256_no_lf_v1"
)
SUPERVISOR_PYTHON_ISOLATED_FLAGS: Final = ("-I", "-S", "-B")
SUPERVISOR_STAGED_LAUNCH_SPEC_FLAG: Final = "--staged-launch-spec"
SUPERVISOR_STAGED_E_INTENT_FLAG: Final = "--staged-e-intent"
DIRECT_BASE_RUNTIME_LIVE_PARITY_POLICY: Final = (
    "createprocess_bound_base_runtime_direct_image_and_argv_exact_v1"
)
COMPOSED_TERMINAL_RECEIPT_POLICY: Final = (
    "original_confirmatory_capsule_composed_terminal_receipt_v1"
)
COMPOSED_TERMINAL_STDOUT_SUMMARY_POLICY: Final = (
    "original_confirmatory_composed_terminal_summary_v1"
)
TERMINAL_INPUT_INTEGRITY_POLICY: Final = (
    "no_follow_single_link_read_only_no_named_ads_v1"
)
SUPERVISOR_V3_POLICY: Final = (
    "aanca_event_driven_unattended_supervisor_external_handoff_v3"
)
EXTERNAL_CODEX_HANDOFF_POLICY: Final = (
    "aanca_external_current_session_two_branch_handoff_v1"
)
INTERNAL_CODEX_WAKE_DISPOSITION: Final = "PROHIBITED_EXTERNAL_SINGLE_WAKE_OWNER"
EXTERNAL_CODEX_HANDOFF_SINGLE_WAKE_OWNER: Final = (
    "operational_current_session_successor"
)
ORIGINAL_CONFIRMATORY_PRODUCTION_MAX_LOG_BYTES: Final = 1024**3
ORIGINAL_CONFIRMATORY_PRODUCTION_MAIN_TIMEOUT_MS: Final = 30 * 24 * 60 * 60 * 1000
ORIGINAL_CONFIRMATORY_PRODUCTION_VERIFIER_TIMEOUT_MS: Final = 60 * 60 * 1000
ORIGINAL_CONFIRMATORY_PRODUCTION_CODEX_WAKE_TIMEOUT_SECONDS: Final = 6 * 60 * 60
SUPERVISOR_V2_SUCCESS_TERMINAL_KIND: Final = "SUCCESS"
SUPERVISOR_V2_SUCCESS_REASON: Final = (
    "exit_zero_science_seal_preterminal_pin_and_integrity_verified"
)
SUPERVISOR_TERMINAL_HASH_POLICY: Final = "canonical_json_line_sha256_v1"
TERMINAL_VERIFIER_ENVIRONMENT_OBSERVATION_METHOD: Final = "fresh_process_os_environ_v1"
NO_FOLLOW_PHYSICAL_FILE_IDENTITY_POLICY: Final = (
    "aanca_no_follow_physical_file_identity_v1"
)
RETAINED_NATIVE_HANDLE_BINDING_POLICY: Final = "aanca_retained_native_handle_binding_v1"
PRESERIALIZATION_NATIVE_HANDLE_BINDING_POLICY: Final = (
    "aanca_preserialization_native_handle_binding_v1"
)
SUPERVISOR_JOB_ANCESTOR_LEASE_CONTRACT_POLICY: Final = (
    "aanca_supervisor_job_ancestor_lease_contract_v1"
)
POSTWAKE_INPUT_LEASE_CONTRACT_POLICY: Final = "aanca_postwake_input_lease_contract_v1"
POSTWAKE_INPUT_IDENTITY_POLICY: Final = (
    "native_no_follow_readonly_single_link_no_ads_share_read_v1"
)
POSTWAKE_LOG_CREATION_POLICY: Final = (
    "create_new_retained_parent_rw_handle_bounded_writer_duplicate_v1"
)
POSTWAKE_TERMINAL_CREATION_POLICY: Final = (
    "create_new_retained_parent_rw_handle_canonical_fsync_readonly_v1"
)
POSTWAKE_RETAINED_THROUGH_POLICY: Final = (
    "through_blocking_exact_codex_wake_and_postwake_revalidation_v1"
)
POSTWAKE_SUPERVISOR_LOSS_POLICY: Final = (
    "stop_no_scientific_retry_if_continuity_lost_v1"
)
POSTWAKE_STDERR_DERIVATION_POLICY: Final = (
    "sibling_of_exact_verifier_stdout_fixed_filename_v1"
)
POSTWAKE_PROCESS_DUP_HANDLE_POLICY: Final = (
    "windows_process_dup_handle_reduced_generic_read_v1"
)
POSTWAKE_SOURCE_OWNER_OPEN_POLICY: Final = (
    "process_dup_handle_query_limited_and_synchronize_v1"
)
POSTWAKE_SOURCE_OWNER_OPEN_ACCESS_MASK: Final = 0x00101040
PRETERMINAL_OVERLAP_HANDSHAKE_CONTRACT_POLICY: Final = (
    "original_confirmatory_preterminal_pin_overlap_handshake_contract_v1"
)
PRETERMINAL_OVERLAP_HANDSHAKE_POLICY: Final = (
    "original_confirmatory_preterminal_pin_overlap_handshake_v1"
)
PRETERMINAL_OVERLAP_ACK_POLICY: Final = (
    "original_confirmatory_preterminal_pin_overlap_ack_v1"
)
PRETERMINAL_OVERLAP_HANDSHAKE_RECEIPT_POLICY: Final = (
    "original_confirmatory_preterminal_pin_overlap_handshake_receipt_v1"
)
PRETERMINAL_READY_MESSAGE_TYPE: Final = "PRETERMINAL_PIN_READY"
PRETERMINAL_ACK_MESSAGE_TYPE: Final = "PRETERMINAL_PIN_ACK"
PRETERMINAL_HANDSHAKE_PIPE_TRANSPORT: Final = "bounded_anonymous_pipe_blocking_v1"
PRETERMINAL_HANDSHAKE_STDERR_TRANSPORT: Final = "continuous_retained_bounded_log_v1"
COMPOSED_TERMINAL_ONE_USE_CLAIM_POLICY: Final = (
    "original_confirmatory_composed_terminal_create_new_one_use_claim_v1"
)
COMPOSED_TERMINAL_ONE_USE_CLAIM_DISPOSITION: Final = (
    "created_new_before_terminal_input_read_v1"
)
COMPOSED_TERMINAL_SUPERVISOR_CUSTODY_POLICY: Final = (
    "duplicate_reduced_read_into_live_supervisor_before_serialization_v1"
)
COMPOSED_TERMINAL_CUSTODY_BINDING_POLICY: Final = (
    "original_confirmatory_composed_terminal_supervisor_custody_v1"
)
POSTWAKE_INPUT_LEASE_RECEIPT_POLICY: Final = "aanca_postwake_input_lease_receipt_v2"
POSTWAKE_CUSTODY_HANDSHAKE_CONTRACT_POLICY: Final = (
    "original_confirmatory_postwake_custody_handshake_contract_v1"
)
POSTWAKE_CUSTODY_SEED_POLICY: Final = "original_confirmatory_postwake_custody_seed_v1"
POSTWAKE_CUSTODY_PIPE_DERIVATION_POLICY: Final = (
    "postwake_custody_seed_sha256_direct_suffix_v1"
)
POSTWAKE_CUSTODY_READY_POLICY: Final = "original_confirmatory_postwake_custody_ready_v1"
POSTWAKE_CUSTODY_ACK_POLICY: Final = "original_confirmatory_postwake_custody_ack_v1"
POSTWAKE_COMPOSED_READBACK_RECEIPT_POLICY: Final = (
    "aanca_postwake_composed_readback_receipt_v1"
)
POSTWAKE_COMPOSED_READBACK_CREATE_POLICY: Final = (
    "create_new_retained_rw_share_read_readonly_seal_no_recovery_v1"
)
POSTWAKE_CUSTODY_COMPLETION_SUMMARY_POLICY: Final = (
    "original_confirmatory_postwake_custody_completion_summary_v1"
)
POSTWAKE_PIPE_CREATION_POLICY: Final = (
    "precreated_before_wake_first_instance_single_client_overlapped_v1"
)
POSTWAKE_PIPE_SECURITY_POLICY: Final = (
    "protected_dacl_current_owner_and_local_system_only_v1"
)
POSTWAKE_SHARED_DEADLINE_POLICY: Final = (
    "one_absolute_monotonic_deadline_for_pipe_ready_ack_and_wake_exit_v1"
)
POSTWAKE_EVENT_WAIT_POLICY: Final = (
    "wait_for_multiple_objects_overlapped_pipe_event_and_exact_wake_process_v1"
)
Q_REPLACEMENT_V2_POLICY: Final = "original_confirmatory_q_replacement_v2"
Q_REPLACEMENT_V2_FILENAME: Final = "original_confirmatory_q_replacement_v2.json"
Q_REPLACEMENT_V2_DISPOSITION: Final = (
    "one_create_new_q_publication_for_exact_bound_inputs_v1"
)
Q_ATTEMPT_IDENTITY_DERIVATION_POLICY: Final = (
    "original_confirmatory_q_attempt_identity_derivation_v1"
)
Q_ATTEMPT_LAUNCH_NONCE_DERIVATION_POLICY: Final = (
    "original_confirmatory_q_attempt_launch_nonce_derivation_v1"
)
CONTROL_STAGING_PROJECTION_POLICY: Final = "original_confirmatory_control_staging_v2"
CONTROL_STAGING_DIRECTORY_NAME: Final = "control_staging"
SUPERVISOR_JOBS_DIRECTORY_NAME: Final = "jobs"
CONTROL_STAGING_EXACT_FILE_ALLOWLIST: Final = (
    "staging_attempt.json",
    "e_intent.json",
    "launch_authorization.json",
    "supervisor_launch_spec.json",
    "staging_ready.json",
)
SCIENTIFIC_AUTHORITY_PROJECTION_POLICY: Final = (
    "original_confirmatory_scientific_authority_projection_v1"
)
STATIC_RUNNER_BINDING_POLICY: Final = "original_confirmatory_static_runner_binding_v3"
PUBLISHED_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_POLICY: Final = (
    "published_original_confirmatory_technical_authority_lifecycle_binding_v1"
)
TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_POLICY: Final = (
    "original_confirmatory_technical_authority_lifecycle_binding_v1"
)
SCIENTIFIC_REQUEST_PROJECTION_POLICY: Final = (
    "original_confirmatory_capsule_request_projection_v1"
)
REAL_CONFIRMATORY_ARTIFACT_SCOPE: Final = "real_pannuke_confirmatory_study"
SCIENTIFIC_CONTROL_READ_SCOPE: Final = "integrity/control_only_no_scientific_outcomes"
COMMAND_DERIVATION_CONTRACT_POLICY: Final = (
    "original_confirmatory_capsule_command_derivation_contract_v1"
)
COMMAND_PROJECTION_POLICY: Final = "original_confirmatory_capsule_command_projection_v1"
COMMAND_E_FILE_INSERTION_POLICY: Final = (
    "append_value_to_terminal_e_file_sha256_flag_then_continue_v1"
)
COMMAND_E_CORE_INSERTION_POLICY: Final = (
    "append_value_to_terminal_e_core_sha256_flag_then_continue_v1"
)
SUPERVISOR_RELEASE_BINDING_POLICY: Final = (
    "original_confirmatory_supervisor_release_binding_v1"
)
EXTERNAL_CONTROL_PLANE_RELEASE_DIRECTORY_NAME: Final = "AANCA-control-plane-release-v2"
EXTERNAL_CONTROL_PLANE_VERIFICATIONS_DIRECTORY_NAME: Final = "verifications"
EXTERNAL_CONTROL_PLANE_QUALIFICATION_ATTESTATION_FILENAME: Final = (
    "release_qualification_attestation.json"
)
Q_E_CUSTODY_HANDOFF_POLICY: Final = (
    "original_confirmatory_q_e_supervisor_custody_handoff_v1"
)
Q_E_CUSTODY_TRANSPORT: Final = "bounded_anonymous_pipe_blocking_v1"
Q_E_CUSTODY_READY_MESSAGE_TYPE: Final = "Q_E_CUSTODY_READY"
Q_E_CUSTODY_ACK_MESSAGE_TYPE: Final = "Q_E_CUSTODY_ACK"
Q_E_CUSTODY_CONTRACT_POLICY: Final = (
    "original_confirmatory_q_e_supervisor_custody_contract_v1"
)
Q_E_CUSTODY_RECEIPT_POLICY: Final = (
    "original_confirmatory_q_e_supervisor_custody_receipt_v1"
)
Q_E_CUSTODY_ACK_POLICY: Final = "original_confirmatory_q_e_supervisor_custody_ack_v1"
Q_E_CUSTODY_RECEIPT_FILENAME: Final = "q_e_custody_receipt.json"
Q_E_CUSTODY_SUPERVISOR_RETENTION_POLICY: Final = (
    "through_science_preterminal_terminal_postwake_and_terminal_seal_v1"
)
Q_E_CUSTODY_LINE_MAX_BYTES: Final = 64 * 1024
E_INTENT_POLICY: Final = "original_confirmatory_e_intent_v1"
E_INTENT_DISPOSITION: Final = "one_create_new_e_intent_for_exact_supervisor_job_v1"
E_INTENT_FILENAME: Final = "e_intent.json"
E_CONSUMPTION_TOMBSTONE_FILENAME: Final = "e_intent_consumed.json"
E_JOB_BINDING_POLICY: Final = "original_confirmatory_supervisor_job_binding_v1"
E_LINEAGE_POLICY: Final = "original_confirmatory_execution_lineage_v1"
E_CONSUMPTION_CONTRACT_POLICY: Final = (
    "original_confirmatory_e_intent_consumed_supervisor_custody_v1"
)
E_CONSUMPTION_CLAIM_POLICY: Final = "original_confirmatory_e_intent_consumed_claim_v1"
E_CONSUMPTION_CUSTODY_RECEIPT_POLICY: Final = (
    "original_confirmatory_e_intent_consumed_custody_receipt_v1"
)

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
_EXPECTED_ARTIFACT_TEMPLATE_RULE_FIELDS: Final = {
    "role",
    "path_anchor",
    "relative_path",
    "expected_sha256",
    "must_be_absent_before",
    "json_equals",
}
_EXPECTED_ARTIFACT_TEMPLATE: Final[tuple[dict[str, Any], ...]] = (
    {
        "role": "terminal_seal",
        "path_anchor": "expected_run_directory",
        "relative_path": ".immutable.json",
        "expected_sha256": None,
        "must_be_absent_before": True,
        "json_equals": {"run_id": "$RUN_ID", "status": "completed"},
    },
    {
        "role": "integrity_receipt",
        "path_anchor": "expected_run_directory",
        "relative_path": "artifact_manifest.json",
        "expected_sha256": None,
        "must_be_absent_before": True,
        "json_equals": {"run_id": "$RUN_ID", "status": "completed"},
    },
    {
        "role": "completion_evidence",
        "path_anchor": "expected_run_directory",
        "relative_path": "completion_evidence.json",
        "expected_sha256": None,
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
        "expected_sha256": None,
        "must_be_absent_before": False,
        "json_equals": {},
    },
    {
        "role": "stage_attestation_registry",
        "path_anchor": "runs_root",
        "relative_path": "run_stage_attestations.jsonl",
        "expected_sha256": None,
        "must_be_absent_before": False,
        "json_equals": {},
    },
    {
        "role": "stage_attestation_anchor",
        "path_anchor": "runs_root",
        "relative_path": "run_stage_attestations.anchor.json",
        "expected_sha256": None,
        "must_be_absent_before": False,
        "json_equals": {},
    },
    {
        "role": "disposition_anchor",
        "path_anchor": "runs_root",
        "relative_path": "run_dispositions.anchor.json",
        "expected_sha256": None,
        "must_be_absent_before": False,
        "json_equals": {},
    },
)
E_CONSUMPTION_READY_POLICY: Final = "original_confirmatory_e_intent_consumed_ready_v1"
E_CONSUMPTION_ACK_POLICY: Final = "original_confirmatory_e_intent_consumed_ack_v1"
E_CONSUMPTION_READY_MESSAGE_TYPE: Final = "E_INTENT_CONSUMED_READY"
E_CONSUMPTION_ACK_MESSAGE_TYPE: Final = "E_INTENT_CONSUMED_ACK"
E_CONSUMPTION_TRANSPORT: Final = "bounded_anonymous_pipe_blocking_v1"
E_CONSUMPTION_SUPERVISOR_RETENTION_POLICY: Final = (
    "through_main_wait_preterminal_terminal_postwake_and_terminal_seal_v1"
)
E_CONSUMPTION_DUPLICATE_TARGET_ACCESS_MASK: Final = 0x80000000
E_CONSUMPTION_MAPPED_FILE_GENERIC_READ_ACCESS_MASK: Final = 0x00120089
E_CONSUMPTION_DUPLICATE_OPTIONS: Final = 0
E_CONSUMPTION_LINE_MAX_BYTES: Final = 16 * 1024
E_CONSUMPTION_CUSTODY_RECEIPT_FILENAME: Final = "e_intent_consumed_custody_receipt.json"
E_CONSUMPTION_CLAIM_DISPOSITION: Final = (
    "duplicated_reduced_read_into_exact_supervisor_before_same_handle_serialization_v1"
)
CANONICAL_CORE_HASH_POLICY: Final = "canonical_json_without_self_field_sha256_v1"
COMMAND_FINAL_CARRIER_POLICY: Final = "supervisor_job_spec_create_new_read_only_v1"
CONTROL_PUBLICATION_ANCESTOR_LEASE_POLICY: Final = (
    "original_confirmatory_control_publication_ancestor_lease_v1"
)
SEMANTIC_OUTCOME_READ_SCOPE: Final = (
    "integrity_and_completion_evidence_only_no_scientific_outcome_values_v1"
)
CAPSULE_PYTHON_ISOLATED_FLAGS: Final = ("-I", "-B")
CAPSULE_ALLOWED_MODES: Final = (
    "run-confirmatory",
    "verify-preterminal",
    "verify-terminal",
)
CAPSULE_SCIENTIFIC_MODE: Final = "run-confirmatory"
CAPSULE_PRETERMINAL_MODE: Final = "verify-preterminal"
CAPSULE_TERMINAL_MODE: Final = "verify-terminal"
CAPSULE_EXECUTION_MODES: Final = ("fresh", "successor_resume")
CAPSULE_COMMON_TAIL_FLAGS: Final = (
    "--e-intent",
    "--e-intent-sha256",
    "--e-intent-core-sha256",
    "--q-authority-root-sha256",
    "--launch-nonce",
    "--supervisor-job-id",
    "--supervisor-job-dir",
    "--attempt-id",
    "--run-id",
    "--execution-mode",
)
CAPSULE_SUCCESSOR_LINEAGE_FLAG: Final = "--retry-of-run-id"
CAPSULE_PRETERMINAL_SUFFIX_FLAGS: Final = (
    "--run-spec",
    "--launch-intent",
    "--process-started",
    "--preterminal-pin",
)
CAPSULE_TERMINAL_SUFFIX_FLAGS: Final = (
    "--supervisor-terminal",
    "--verifier-stdout",
    "--preterminal-pin",
    "--composed-terminal",
)
MAX_CONTROL_FILE_BYTES: Final = 16 * 1024 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}")
_NONCE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")
_Q_ATTEMPT_ID = re.compile(r"ocq-[0-9a-f]{32}")
_WINDOWS_SID = re.compile(r"S-1-(?:[0-9]+-)+[0-9]+")
_CODEX_HANDOFF_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
_CODEX_HANDOFF_UUID_V4 = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
_CODEX_HANDOFF_FILE_ID = re.compile(r"[0-9a-f]{32}")
_CODEX_HANDOFF_MARKER = re.compile(r"AANCA_CURRENT_SESSION_IDLE_[0-9a-f]{64}")
_CODEX_HANDOFF_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z"
)

CODEX_HANDOFF_BASE_SYNTHETIC_SCHEMA: Final = (
    "aanca.codex-handoff-base-authority.synthetic.v1"
)
CODEX_HANDOFF_BASE_OPERATIONAL_SCHEMA: Final = (
    "aanca.codex-handoff-base-authority.operational.v1"
)
CODEX_HANDOFF_ATTEMPT_CREATION_SCHEMA: Final = (
    "aanca.codex-handoff-attempt-creation-authority.operational.v1"
)
CODEX_HANDOFF_ATTEMPT_SCHEMA: Final = (
    "aanca.codex-handoff-attempt-authority.operational.v1"
)
CODEX_HANDOFF_BASE_SYNTHETIC_SCOPE: Final = (
    "aanca_unattended_codex_handoff_base_synthetic_v1"
)
CODEX_HANDOFF_BASE_OPERATIONAL_SCOPE: Final = (
    "aanca_unattended_codex_handoff_base_operational_v1"
)
CODEX_HANDOFF_ATTEMPT_CREATION_SCOPE: Final = (
    "aanca_unattended_codex_handoff_attempt_creation_operational_v1"
)
CODEX_HANDOFF_ATTEMPT_SCOPE: Final = (
    "aanca_unattended_codex_handoff_attempt_operational_v1"
)
CODEX_HANDOFF_ARM_ALGORITHM: Final = (
    "retained_identity_ReadDirectoryChangesW_before_snapshot_prefix_sha256_v1"
)
CODEX_HANDOFF_BRANCH_SELECTION_POLICY: Final = (
    "postterminal_exactly_one_branch_from_terminal_disposition_v1"
)
CODEX_HANDOFF_EXTERNAL_POLICY: Final = (
    "aanca_external_current_session_two_branch_handoff_v1"
)
CODEX_HANDOFF_PROMPT_TEMPLATE: Final = (
    "AANCA unattended handoff {marker}. Re-read STATUS.md and PLAN.md in "
    "C:\\Users\\NATAN\\Documents\\AANCA. Independently verify the completed "
    "process and the same-turn idle boundary {turn_id} in {session_jsonl_path}. "
    "Continue only the legal next step. Perform all short steps, or launch exactly "
    "one authorized long process and re-arm the tested supervisor, then end the "
    "session. Never retry a scientific operation automatically."
)
CODEX_HANDOFF_PROMPT_TEMPLATE_SHA256: Final = (
    "259b48776b530632a1c1235cb32a86cd02bb3a45d3f21874fded584c6bc0aaca"
)
CODEX_HANDOFF_MAX_SESSION_FILE_BYTES: Final = 1024 * 1024 * 1024
CODEX_HANDOFF_SUPERVISED_PROCESS_MAX_RUNTIME_MS: Final = 2_592_000_000
CODEX_HANDOFF_BOUNDARY_MAX_AGE_MS: Final = 3_600_000
CODEX_HANDOFF_ONE_SHOT_MAX_RUNTIME_SECONDS: Final = 21_600
CODEX_HANDOFF_WATCHER_BUFFER_BYTES: Final = 65_536
CODEX_HANDOFF_MAX_RECORD_BYTES: Final = 4_194_304
CODEX_HANDOFF_MAX_APPEND_BYTES: Final = 16_777_216


class OriginalConfirmatoryCapsuleAuthorityError(ValueError):
    """A closed capsule Q/E contract failed validation."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode strict canonical JSON without a line terminator."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "value is not strict canonical JSON"
        ) from exc


def canonical_json_line_bytes(value: Any) -> bytes:
    """Encode one strict canonical JSON line."""

    return canonical_json_bytes(value) + b"\n"


def canonical_json_sha256(value: Any) -> str:
    """SHA-256 of strict canonical JSON without a line terminator."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_json_line_sha256(value: Any) -> str:
    """SHA-256 of one strict canonical JSON line."""

    return hashlib.sha256(canonical_json_line_bytes(value)).hexdigest()


def _postwake_custody_seed_json_bytes(value: Any) -> bytes:
    """Encode the frozen custody seed with ASCII JSON escaping and no LF."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "postwake custody seed is not strict canonical JSON"
        ) from exc


def _postwake_custody_seed_sha256(value: Any) -> str:
    return hashlib.sha256(_postwake_custody_seed_json_bytes(value)).hexdigest()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "strict JSON contains a duplicate object key"
            )
        result[key] = value
    return result


def decode_canonical_json_line(payload: bytes, *, role: str) -> dict[str, Any]:
    """Decode exactly one canonical UTF-8 JSON object line."""

    if (
        not payload
        or len(payload) > MAX_CONTROL_FILE_BYTES
        or not payload.endswith(b"\n")
        or b"\r" in payload
        or b"\n" in payload[:-1]
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} is not one bounded canonical JSON line"
        )
    try:
        value = json.loads(
            payload[:-1].decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                OriginalConfirmatoryCapsuleAuthorityError(
                    f"{role} contains non-finite JSON token {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} is not strict UTF-8 JSON"
        ) from exc
    raw = _mapping(value, role=role)
    if canonical_json_line_bytes(raw) != payload:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} is not in canonical byte form"
        )
    return raw


def _mapping(value: Any, *, role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise OriginalConfirmatoryCapsuleAuthorityError(f"{role} must be an object")
    return dict(value)


def _contains_mapping_key(value: Any, forbidden_key: str) -> bool:
    if isinstance(value, Mapping):
        return forbidden_key in value or any(
            _contains_mapping_key(item, forbidden_key) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_mapping_key(item, forbidden_key) for item in value)
    return False


def _sha256(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be a lowercase SHA-256"
        )
    return value


def _positive_int(value: Any, *, role: str) -> int:
    if type(value) is not int or value <= 0:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be an exact positive integer"
        )
    return value


def _nonnegative_int(value: Any, *, role: str) -> int:
    if type(value) is not int or value < 0:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be an exact nonnegative integer"
        )
    return value


def _absolute_path(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be a canonical absolute path"
        )
    path = Path(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be a canonical absolute path"
        )
    if os.name == "nt" and value.startswith(("\\\\?\\", "\\\\.\\")):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} uses a forbidden Windows device namespace"
        )
    return value


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return true when either canonical absolute path contains the other."""

    left_key = os.path.normcase(str(left))
    right_key = os.path.normcase(str(right))
    return (
        left_key == right_key
        or left_key.startswith(f"{right_key}{os.sep}")
        or right_key.startswith(f"{left_key}{os.sep}")
    )


def _identifier(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be a canonical identifier"
        )
    return value


def _utc_timestamp(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z") or "\x00" in value:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be an explicit UTC timestamp"
        )
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as exc:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be an explicit UTC timestamp"
        ) from exc
    offset = parsed.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be an explicit UTC timestamp"
        )
    return value


def _codex_handoff_mapping(
    value: Any,
    fields: set[str] | frozenset[str],
    *,
    role: str,
) -> dict[str, Any]:
    raw = _mapping(value, role=role)
    if set(raw) != fields:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} has an unexpected field set"
        )
    return raw


def _codex_handoff_exact_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            _codex_handoff_exact_equal(actual[key], expected[key]) for key in actual
        )
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _codex_handoff_exact_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _codex_handoff_byte_count(value: Any, *, role: str) -> int:
    if type(value) is not int or not 0 <= value <= 9_007_199_254_740_991:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be an exact bounded JSON integer"
        )
    return value


def _codex_handoff_uuid(value: Any, *, role: str, version_four: bool = False) -> str:
    pattern = _CODEX_HANDOFF_UUID_V4 if version_four else _CODEX_HANDOFF_UUID
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be one lowercase canonical UUID"
        )
    return value


def _codex_handoff_ascii(value: Any, *, role: str, nonempty: bool) -> str:
    if (
        not isinstance(value, str)
        or (nonempty and not value)
        or "\x00" in value
        or "\r" in value
        or "\n" in value
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be one exact ASCII value"
        )
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be one exact ASCII value"
        ) from exc
    return value


def _codex_handoff_windows_path(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) < 4
        or value[0] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        or value[1:3] != ":\\"
        or value.endswith("\\")
        or "/" in value
        or "\x00" in value
        or value.startswith(("\\\\?\\", "\\\\.\\"))
        or ":" in value[2:]
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be one canonical DOS path"
        )
    parts = value[3:].split("\\")
    if any(
        not part or part in {".", ".."} or part.endswith((" ", ".")) for part in parts
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be one canonical DOS path"
        )
    if os.name == "nt" and (
        os.path.normpath(value) != value or os.path.abspath(value) != value
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be one canonical DOS path"
        )
    return value


def _canonical_codex_handoff_session_file_identity(
    value: Any,
    *,
    role: str,
) -> dict[str, Any]:
    raw = _codex_handoff_mapping(
        value,
        {
            "volume_serial_number",
            "file_id_128",
            "creation_time_100ns",
            "file_attributes",
            "link_count",
            "directory",
            "reparse_point",
        },
        role=role,
    )
    if (
        type(raw["volume_serial_number"]) is not int
        or not 0 <= raw["volume_serial_number"] <= 18_446_744_073_709_551_615
        or not isinstance(raw["file_id_128"], str)
        or _CODEX_HANDOFF_FILE_ID.fullmatch(raw["file_id_128"]) is None
        or type(raw["creation_time_100ns"]) is not int
        or raw["creation_time_100ns"] <= 0
        or type(raw["file_attributes"]) is not int
        or raw["file_attributes"] <= 0
        or type(raw["link_count"]) is not int
        or raw["link_count"] != 1
        or raw["directory"] is not False
        or raw["reparse_point"] is not False
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} violates its exact non-reparse single-link identity"
        )
    return raw


def _codex_handoff_resume_command_policy(
    *, operational: bool = False
) -> dict[str, Any]:
    return {
        "argv_prefix": ["exec", "resume"],
        "argv_length": 5,
        "program_path_policy": "exact_codex_cli_path",
        "session_argument_policy": "exact_session_origin_session_id",
        "prompt_argument_policy": (
            "exact_postterminal_selected_branch_rendered_prompt_utf8"
            if operational
            else "exact_attempt_rendered_prompt_text_utf8"
        ),
        "forbidden_tokens": ["--last"],
        "shell_allowed": False,
        "max_attempts": 1,
        "automatic_retry_allowed": False,
        "wake_intent_required_before_spawn": True,
    }


def _codex_handoff_prompt_policy() -> dict[str, Any]:
    return {
        "template_id": "aanca_codex_resume_continuation_v1",
        "template_text": CODEX_HANDOFF_PROMPT_TEMPLATE,
        "template_utf8_sha256": CODEX_HANDOFF_PROMPT_TEMPLATE_SHA256,
        "substitution_exact_keys": ["marker", "session_jsonl_path", "turn_id"],
        "text_normalization": "none_exact_UTF-8",
        "required_project_root": "C:\\Users\\NATAN\\Documents\\AANCA",
        "required_read_files": ["STATUS.md", "PLAN.md"],
        "short_step_exit_policy": (
            "finish_short_steps_or_arm_exactly_one_authorized_long_process_then_end"
        ),
        "scientific_retry_policy": "never_automatic",
    }


def _codex_handoff_completion_policy() -> dict[str, Any]:
    return {
        "event_wait_primitive": (
            "ReadDirectoryChangesW_overlapped_then_WaitForSingleObject_INFINITE"
        ),
        "watcher_registration_order": (
            "registered_before_pre_arm_offset_and_prefix_snapshot"
        ),
        "retained_session_file_open": {
            "desired_access": ["GENERIC_READ", "FILE_READ_ATTRIBUTES"],
            "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
            "delete_share_allowed": False,
            "open_reparse_point": True,
        },
        "notify_filter": [
            "FILE_NOTIFY_CHANGE_FILE_NAME",
            "FILE_NOTIFY_CHANGE_SIZE",
            "FILE_NOTIFY_CHANGE_LAST_WRITE",
        ],
        "watch_subtree": False,
        "required_sequence": ["assistant_final_answer", "same_turn_task_complete"],
        "assistant_final_answer_match": {
            "outer_type": "response_item",
            "payload_type": "message",
            "role": "assistant",
            "phase": "final_answer",
            "turn_id_json_pointer": (
                "/payload/internal_chat_message_metadata_passthrough/turn_id"
            ),
            "content_policy": "exactly_one_output_text",
            "marker_occurrences_exact": 1,
        },
        "task_complete_match": {
            "outer_type": "event_msg",
            "payload_type": "task_complete",
            "turn_id_json_pointer": "/payload/turn_id",
            "last_agent_message_json_pointer": "/payload/last_agent_message",
            "marker_occurrences_exact": 1,
            "must_follow_assistant_final_answer": True,
            "must_be_terminal_record": True,
            "idle_boundary": True,
        },
        "marker_policy": {
            "format_regex": "^AANCA_CURRENT_SESSION_IDLE_[0-9a-f]{64}$",
            "nonce_encoding": "lowercase_hex",
            "nonce_bytes": 32,
            "pre_arm_occurrences_exact": 0,
            "allowed_post_arm_records": [
                "assistant_final_answer",
                "same_turn_task_complete",
            ],
            "total_post_arm_occurrences_exact": 2,
        },
        "append_integrity_policy": {
            "retained_identity_required": True,
            "prefix_sha256_required": True,
            "append_only_required": True,
            "full_readback_required": True,
            "truncation_disposition": "durable_STOP",
            "identity_drift_disposition": "durable_STOP",
            "rotation_disposition": "durable_STOP",
            "buffer_overflow_disposition": "durable_STOP",
            "partial_record_disposition": "durable_STOP",
            "duplicate_boundary_disposition": "durable_STOP",
        },
        "ambiguity_disposition": "durable_STOP_no_wake_no_retry",
    }


def _codex_handoff_limits() -> dict[str, Any]:
    return {
        "max_session_file_bytes": CODEX_HANDOFF_MAX_SESSION_FILE_BYTES,
        "supervised_process_max_runtime_ms": (
            CODEX_HANDOFF_SUPERVISED_PROCESS_MAX_RUNTIME_MS
        ),
        "verification_or_terminal_boundary_max_age_ms": (
            CODEX_HANDOFF_BOUNDARY_MAX_AGE_MS
        ),
        "codex_one_shot_max_runtime_seconds": (
            CODEX_HANDOFF_ONE_SHOT_MAX_RUNTIME_SECONDS
        ),
        "watcher_buffer_bytes": CODEX_HANDOFF_WATCHER_BUFFER_BYTES,
        "max_record_bytes": CODEX_HANDOFF_MAX_RECORD_BYTES,
        "max_append_bytes": CODEX_HANDOFF_MAX_APPEND_BYTES,
    }


def _codex_handoff_success_template_policy() -> dict[str, Any]:
    return {
        "template_id": "aanca_external_qualified_success_terminal_client_v1",
        "allowed_terminal_dispositions": ["QUALIFIED_SUCCESS"],
        "substitution_exact_keys": [
            "marker",
            "turn_id",
            "session_jsonl_path",
            "base_authority_payload_sha256",
            "attempt_authority_payload_sha256",
            "job_id",
            "supervisor_job_directory",
            "supervisor_spec_path",
            "supervisor_spec_sha256",
            "terminal_receipt_sha256",
            "terminal_client_launcher_argv_json",
            "terminal_client_launcher_command_sha256",
            "verify_terminal_command_sha256",
        ],
        "template_text": (
            "AANCA external current-session handoff marker={marker}, "
            "turn_id={turn_id}, session_jsonl_path={session_jsonl_path}.\n"
            "Read STATUS.md and PLAN.md completely as policy context. Do not "
            "select, discover, or reconstruct a job or command from either file.\n"
            "This is the sole external wake, bound to "
            "base_authority_payload_sha256={base_authority_payload_sha256} and "
            "attempt_authority_payload_sha256={attempt_authority_payload_sha256}.\n"
            "This wake is bound to job_id={job_id}, "
            "supervisor_job_directory={supervisor_job_directory}, "
            "supervisor_spec_path={supervisor_spec_path}, "
            "supervisor_spec_sha256={supervisor_spec_sha256}, "
            "terminal_receipt_sha256={terminal_receipt_sha256}.\n"
            "Your first executable action after reading those two files must launch "
            "exactly one sealed terminal-client launcher using this canonical "
            "process argv JSON, unchanged and in order:\n"
            "{terminal_client_launcher_argv_json}\n"
            "The canonical terminal-client launcher command SHA-256 is "
            "{terminal_client_launcher_command_sha256}.\n"
            "That launcher is authorized to launch exactly one new verify-terminal "
            "child using the already-sealed command whose canonical SHA-256 is "
            "{verify_terminal_command_sha256}. Do not launch verify-terminal "
            "directly.\n"
            "Do not use globbing, --last, autodiscovery, fallback, retry, an "
            "alternate job, an alternate launcher, or an alternate command. If the "
            "exact launcher, child launch, custody handshake, terminal verification, "
            "seal, or integrity check fails or is ambiguous, perform diagnosis only "
            "and do not retry.\n"
            "After the exact launcher returns success, verify the composed receipt "
            "and bound seal/integrity evidence without reading scientific outcome "
            "values for tuning. Continue only the legal next short step. Stop after "
            "short steps finish or one separately authorized next long process is "
            "launched and its event-driven supervisor is armed. Never retry an "
            "experiment, publication, or one-use authority."
        ),
    }


def _codex_handoff_diagnosis_template_policy() -> dict[str, Any]:
    return {
        "template_id": "aanca_external_failure_diagnosis_only_v1",
        "allowed_terminal_dispositions": ["FAILED", "AMBIGUOUS", "LOST"],
        "substitution_exact_keys": [
            "marker",
            "turn_id",
            "session_jsonl_path",
            "base_authority_payload_sha256",
            "attempt_authority_payload_sha256",
            "job_id",
            "supervisor_job_directory",
            "supervisor_spec_path",
            "supervisor_spec_sha256",
            "terminal_disposition",
            "terminal_evidence_sha256",
        ],
        "template_text": (
            "AANCA external diagnosis-only handoff marker={marker}, "
            "turn_id={turn_id}, session_jsonl_path={session_jsonl_path}.\n"
            "Read STATUS.md and PLAN.md completely. This is the sole external wake, "
            "bound to base_authority_payload_sha256={base_authority_payload_sha256}, "
            "attempt_authority_payload_sha256={attempt_authority_payload_sha256}, "
            "job_id={job_id}, supervisor_job_directory={supervisor_job_directory}, "
            "supervisor_spec_path={supervisor_spec_path}, "
            "supervisor_spec_sha256={supervisor_spec_sha256}.\n"
            "The terminal disposition is {terminal_disposition}; "
            "terminal_evidence_sha256={terminal_evidence_sha256}. Verify the "
            "terminal STOP/loss/ambiguity evidence and perform diagnosis only. Do "
            "not launch the terminal-client launcher, verify-terminal, process C, a "
            "retry, another experiment, publication, or one-use authority. Do not "
            "use --last, autodiscovery, fallback, or an alternate job. Update "
            "evidence only when legal, then end the Codex session after the short "
            "diagnostic steps."
        ),
    }


def _codex_handoff_branch_template_policy() -> dict[str, Any]:
    success = _codex_handoff_success_template_policy()
    diagnosis = _codex_handoff_diagnosis_template_policy()
    return {
        "branch_selection_policy": CODEX_HANDOFF_BRANCH_SELECTION_POLICY,
        "success_template_policy": success,
        "success_template_policy_root_sha256": canonical_json_sha256(success),
        "diagnosis_template_policy": diagnosis,
        "diagnosis_template_policy_root_sha256": canonical_json_sha256(diagnosis),
    }


def _codex_handoff_external_supervisor_policy() -> dict[str, Any]:
    return {
        "policy": CODEX_HANDOFF_EXTERNAL_POLICY,
        "spec52_schema_version": 3,
        "spec52_policy": "aanca_event_driven_unattended_supervisor_external_handoff_v3",
        "codex_value": None,
        "internal_codex_wake_allowed": False,
        "legacy_handoff_session_allowed": False,
        "single_wake_owner": "operational_current_session_successor",
        "runtime_role_replaced": "option_a_supervisor",
        "custody_transport_policy": (
            "existing_six_pipe_JOB_CUSTODY_RELEASE_ACCEPTED_COMMIT_COMMITTED_unchanged"
        ),
        "terminal_handoff_receipt_schema": (
            "aanca.supervisor-external-codex-terminal-handoff.v1"
        ),
        "branch_selection_owner": "operational_current_session_successor_postterminal",
    }


def _codex_handoff_arm_algorithm_contract() -> dict[str, Any]:
    return {
        "policy": CODEX_HANDOFF_ARM_ALGORITHM,
        "retained_session_handle_policy": (
            "open exact session JSONL as an ordinary non-directory non-reparse file "
            "with read access and FILE_SHARE_READ|FILE_SHARE_WRITE but without "
            "FILE_SHARE_DELETE; retain the same handle until the event boundary is "
            "validated or STOP is durable"
        ),
        "session_identity_policy": (
            "before watcher registration and again after the pre-arm snapshot, "
            "require exact volume serial number, 128-bit file ID, creation time, "
            "attributes, link count, directory=false and reparse_point=false to equal "
            "the base authority pin"
        ),
        "watcher_primitive": (
            "ReadDirectoryChangesW_overlapped_then_WaitForSingleObject_INFINITE"
        ),
        "watcher_registration_order": (
            "open retained session handle; open non-reparse parent directory handle; "
            "issue one overlapped ReadDirectoryChangesW request; only after successful "
            "registration read offset, prefix bytes, record count and prefix SHA-256 "
            "through the retained session handle"
        ),
        "watcher_notify_filter": [
            "FILE_NOTIFY_CHANGE_FILE_NAME",
            "FILE_NOTIFY_CHANGE_SIZE",
            "FILE_NOTIFY_CHANGE_LAST_WRITE",
        ],
        "watch_subtree": False,
        "watcher_buffer_bytes": CODEX_HANDOFF_WATCHER_BUFFER_BYTES,
        "pre_arm_snapshot_exact_keys": [
            "offset_bytes",
            "prefix_sha256",
            "record_count",
            "ends_with_lf",
        ],
        "prefix_hash_domain": (
            "raw bytes from offset 0 inclusive to pre-arm offset exclusive, read "
            "through the retained session file handle, no delimiter added"
        ),
        "marker_policy": (
            "marker is AANCA_CURRENT_SESSION_IDLE_ followed by exact marker_nonce_hex; "
            "nonce is 64 lowercase hexadecimal characters representing 32 "
            "independently generated random bytes; marker must be absent from every "
            "complete pre-arm JSONL record"
        ),
        "append_integrity_policy": (
            "after each directory event, revalidate retained physical identity; "
            "reject rename, replacement, truncation, prefix change, partial UTF-8 or "
            "partial JSONL, duplicate marker, event-buffer overflow, malformed record, "
            "record over limit, append over limit, and any byte beyond the admitted "
            "terminal boundary"
        ),
        "event_sequence_policy": (
            "admit exactly one assistant final_answer record for exact turn_id "
            "containing exact marker, followed by exactly one same-turn task_complete "
            "terminal record; no other terminal or marker-bearing record is admitted"
        ),
        "failure_disposition": (
            "write durable CREATE_NEW STOP evidence and never launch science, Codex, "
            "retry, publication, or another one-use authority"
        ),
        "timeout_policy": (
            "attempt expires exactly 3600000 ms after armed_at_utc; expiry before the "
            "required current-session idle boundary is STOP"
        ),
    }


_CODEX_HANDOFF_OPERATIONAL_SOURCE_FIELDS = {
    "schema",
    "source_path",
    "source_size_bytes",
    "source_sha256",
    "source_inventory_path",
    "source_inventory_file_sha256",
    "source_inventory_payload_sha256",
    "source_inventory_root_sha256",
    "independent_audit_receipt_path",
    "independent_audit_receipt_sha256",
    "authority_spec_path",
    "authority_spec_file_sha256",
    "authority_spec_payload_sha256",
    "synthetic_inventory_path",
    "synthetic_inventory_size_bytes",
    "synthetic_inventory_file_sha256",
    "synthetic_inventory_root_sha256",
    "synthetic_gate_source_path",
    "synthetic_gate_source_size_bytes",
    "synthetic_gate_source_sha256",
}


def _canonical_codex_handoff_operational_source(value: Any) -> dict[str, Any]:
    raw = _codex_handoff_mapping(
        value,
        _CODEX_HANDOFF_OPERATIONAL_SOURCE_FIELDS,
        role="Codex handoff operational source",
    )
    _codex_handoff_ascii(raw["schema"], role="operational source schema", nonempty=True)
    for field in (
        "source_path",
        "source_inventory_path",
        "independent_audit_receipt_path",
        "authority_spec_path",
        "synthetic_inventory_path",
        "synthetic_gate_source_path",
    ):
        _codex_handoff_windows_path(raw[field], role=f"operational source {field}")
    for field in (
        "source_size_bytes",
        "synthetic_inventory_size_bytes",
        "synthetic_gate_source_size_bytes",
    ):
        if (
            _codex_handoff_byte_count(raw[field], role=f"operational source {field}")
            <= 0
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                f"operational source {field} must be positive"
            )
    for field in (
        "source_sha256",
        "source_inventory_file_sha256",
        "source_inventory_payload_sha256",
        "source_inventory_root_sha256",
        "independent_audit_receipt_sha256",
        "authority_spec_file_sha256",
        "authority_spec_payload_sha256",
        "synthetic_inventory_file_sha256",
        "synthetic_inventory_root_sha256",
        "synthetic_gate_source_sha256",
    ):
        _sha256(raw[field], role=f"operational source {field}")
    return raw


def canonical_original_confirmatory_codex_handoff_base_authority(
    value: Mapping[str, Any],
    *,
    require_operational: bool,
) -> dict[str, Any]:
    """Canonicalize one profile-specific base; never relabel between profiles."""

    envelope = _codex_handoff_mapping(
        value,
        {"schema", "payload", "payload_sha256"},
        role="Codex handoff base envelope",
    )
    profiles = {
        CODEX_HANDOFF_BASE_SYNTHETIC_SCHEMA: (
            CODEX_HANDOFF_BASE_SYNTHETIC_SCOPE,
            {
                "production_arm_enabled": False,
                "real_resume_enabled": False,
                "synthetic_only": True,
            },
        ),
        CODEX_HANDOFF_BASE_OPERATIONAL_SCHEMA: (
            CODEX_HANDOFF_BASE_OPERATIONAL_SCOPE,
            {
                "production_arm_enabled": True,
                "real_resume_enabled": True,
                "synthetic_only": False,
            },
        ),
    }
    selected = profiles.get(envelope["schema"])
    if selected is None or (
        require_operational
        and envelope["schema"] != CODEX_HANDOFF_BASE_OPERATIONAL_SCHEMA
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Codex handoff base profile is not authorized for this operation"
        )
    scope, capability = selected
    operational = envelope["schema"] == CODEX_HANDOFF_BASE_OPERATIONAL_SCHEMA
    payload_fields = {
        "authority_scope",
        "session_origin",
        "codex_cli",
        "resume_command_policy",
        "limits",
        "capability_policy",
    }
    payload_fields.update(
        {
            "branch_template_policy",
            "idle_completion_policy",
            "external_supervisor_handoff_policy",
            "operational_source",
        }
        if operational
        else {"continuation_prompt_policy", "completion_policy"}
    )
    payload = _codex_handoff_mapping(
        envelope["payload"],
        payload_fields,
        role="Codex handoff base payload",
    )
    session = _codex_handoff_mapping(
        payload["session_origin"],
        {
            "session_id",
            "session_jsonl_path",
            "expected_cwd",
            "first_record",
            "session_file_identity",
        },
        role="Codex handoff session origin",
    )
    session_id = _codex_handoff_uuid(
        session["session_id"], role="Codex handoff session id"
    )
    session_path = _codex_handoff_windows_path(
        session["session_jsonl_path"], role="Codex handoff session JSONL"
    )
    expected_cwd = _codex_handoff_windows_path(
        session["expected_cwd"], role="Codex handoff expected cwd"
    )
    first = _codex_handoff_mapping(
        session["first_record"],
        {
            "record_type",
            "payload_id",
            "payload_session_id",
            "payload_cli_version",
            "raw_record_bytes_excluding_delimiter",
            "raw_record_sha256_excluding_delimiter",
            "delimiter_hex",
        },
        role="Codex handoff first record",
    )
    raw_record_bytes = _codex_handoff_byte_count(
        first["raw_record_bytes_excluding_delimiter"],
        role="Codex handoff first-record byte count",
    )
    identity = _canonical_codex_handoff_session_file_identity(
        session["session_file_identity"],
        role="Codex handoff session file identity",
    )
    codex = _codex_handoff_mapping(
        payload["codex_cli"],
        {"path", "size_bytes", "sha256", "version_stdout"},
        role="Codex handoff CLI",
    )
    codex_path = _codex_handoff_windows_path(
        codex["path"], role="Codex handoff CLI path"
    )
    codex_size = _codex_handoff_byte_count(
        codex["size_bytes"], role="Codex handoff CLI size"
    )
    _codex_handoff_ascii(
        first["payload_cli_version"],
        role="first-record CLI version",
        nonempty=True,
    )
    _codex_handoff_ascii(
        codex["version_stdout"],
        role="Codex CLI version",
        nonempty=False,
    )
    expected_policies = {
        "resume_command_policy": _codex_handoff_resume_command_policy(
            operational=operational
        ),
        "limits": _codex_handoff_limits(),
        "capability_policy": capability,
    }
    if operational:
        expected_policies.update(
            {
                "branch_template_policy": _codex_handoff_branch_template_policy(),
                "idle_completion_policy": _codex_handoff_completion_policy(),
                "external_supervisor_handoff_policy": (
                    _codex_handoff_external_supervisor_policy()
                ),
            }
        )
        operational_source = _canonical_codex_handoff_operational_source(
            payload["operational_source"]
        )
    else:
        expected_policies.update(
            {
                "continuation_prompt_policy": _codex_handoff_prompt_policy(),
                "completion_policy": _codex_handoff_completion_policy(),
            }
        )
        operational_source = None
    if (
        payload["authority_scope"] != scope
        or first["record_type"] != "session_meta"
        or _codex_handoff_uuid(first["payload_id"], role="first-record payload id")
        != session_id
        or _codex_handoff_uuid(
            first["payload_session_id"], role="first-record payload session id"
        )
        != session_id
        or raw_record_bytes > CODEX_HANDOFF_MAX_RECORD_BYTES
        or _sha256(
            first["raw_record_sha256_excluding_delimiter"],
            role="first-record raw SHA-256",
        )
        != first["raw_record_sha256_excluding_delimiter"]
        or first["delimiter_hex"] != "0a"
        or expected_cwd != "C:\\Users\\NATAN\\Documents\\AANCA"
        or not session_path.lower().endswith(".jsonl")
        or not _codex_handoff_exact_equal(identity, session["session_file_identity"])
        or not codex_path.lower().endswith("\\codex.exe")
        or codex_size <= 0
        or _sha256(codex["sha256"], role="Codex handoff CLI SHA-256") != codex["sha256"]
        or any(
            not _codex_handoff_exact_equal(payload[field], expected)
            for field, expected in expected_policies.items()
        )
        or envelope["payload_sha256"] != canonical_json_sha256(payload)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Codex handoff base violates its exact closed profile"
        )
    return {
        "schema": envelope["schema"],
        "payload": {
            **payload,
            "session_origin": {
                **session,
                "first_record": first,
                "session_file_identity": identity,
            },
            "codex_cli": codex,
            **(
                {"operational_source": operational_source}
                if operational_source is not None
                else {}
            ),
        },
        "payload_sha256": envelope["payload_sha256"],
    }


def _codex_handoff_timestamp_value(value: Any, *, role: str) -> datetime:
    if not isinstance(value, str) or _CODEX_HANDOFF_TIMESTAMP.fullmatch(value) is None:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be exact six-digit UTC RFC3339"
        )
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be exact six-digit UTC RFC3339"
        ) from exc
    return parsed.replace(tzinfo=UTC)


def canonical_original_confirmatory_codex_handoff_attempt_creation_authority(
    value: Mapping[str, Any],
    *,
    base_authority: Mapping[str, Any],
    expected_attempt_authority_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Canonicalize the prelaunch one-use authority without arming or rendering."""

    base = canonical_original_confirmatory_codex_handoff_base_authority(
        base_authority,
        require_operational=True,
    )
    envelope = _codex_handoff_mapping(
        value,
        {"schema", "payload", "payload_sha256"},
        role="Codex handoff attempt-creation envelope",
    )
    if envelope["schema"] != CODEX_HANDOFF_ATTEMPT_CREATION_SCHEMA:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Codex handoff attempt-creation schema differs"
        )
    payload = _codex_handoff_mapping(
        envelope["payload"],
        {
            "authority_scope",
            "base_authority_payload_sha256",
            "session_id",
            "turn_id",
            "marker_nonce_hex",
            "marker",
            "success_template_policy_root_sha256",
            "diagnosis_template_policy_root_sha256",
            "authority_spec_payload_sha256",
            "arm_algorithm_contract_root_sha256",
            "attempt_authority_output_path",
            "attempt_authority_schema",
            "arm_algorithm",
            "required_absent_before",
            "create_new_required",
            "one_use_policy",
        },
        role="Codex handoff attempt-creation payload",
    )
    base_payload = cast(dict[str, Any], base["payload"])
    base_session = cast(dict[str, Any], base_payload["session_origin"])
    templates = cast(dict[str, Any], base_payload["branch_template_policy"])
    source = cast(dict[str, Any], base_payload["operational_source"])
    session_id = _codex_handoff_uuid(
        payload["session_id"], role="attempt-creation session id"
    )
    _codex_handoff_uuid(payload["turn_id"], role="attempt-creation turn id")
    marker_nonce = _sha256(
        payload["marker_nonce_hex"], role="attempt-creation marker nonce"
    )
    marker = payload["marker"]
    output_path = _codex_handoff_windows_path(
        payload["attempt_authority_output_path"],
        role="attempt-creation output path",
    )
    expected_path = (
        None
        if expected_attempt_authority_output_path is None
        else _codex_handoff_windows_path(
            str(expected_attempt_authority_output_path),
            role="expected attempt-creation output path",
        )
    )
    one_use = _codex_handoff_mapping(
        payload["one_use_policy"],
        {
            "attempt_number",
            "maximum_attempts",
            "automatic_retry_allowed",
            "max_age_after_arm_ms",
            "branch_selection_time",
            "rendered_prompt_at_creation_allowed",
        },
        role="attempt-creation one-use policy",
    )
    expected_one_use = {
        "attempt_number": 1,
        "maximum_attempts": 1,
        "automatic_retry_allowed": False,
        "max_age_after_arm_ms": CODEX_HANDOFF_BOUNDARY_MAX_AGE_MS,
        "branch_selection_time": "postterminal",
        "rendered_prompt_at_creation_allowed": False,
    }
    if (
        payload["authority_scope"] != CODEX_HANDOFF_ATTEMPT_CREATION_SCOPE
        or payload["base_authority_payload_sha256"] != base["payload_sha256"]
        or session_id != base_session["session_id"]
        or not isinstance(marker, str)
        or _CODEX_HANDOFF_MARKER.fullmatch(marker) is None
        or marker != f"AANCA_CURRENT_SESSION_IDLE_{marker_nonce}"
        or payload["success_template_policy_root_sha256"]
        != templates["success_template_policy_root_sha256"]
        or payload["diagnosis_template_policy_root_sha256"]
        != templates["diagnosis_template_policy_root_sha256"]
        or payload["authority_spec_payload_sha256"]
        != source["authority_spec_payload_sha256"]
        or payload["arm_algorithm_contract_root_sha256"]
        != canonical_json_sha256(_codex_handoff_arm_algorithm_contract())
        or not output_path.endswith("\\codex_handoff_attempt_authority.json")
        or (expected_path is not None and output_path != expected_path)
        or payload["attempt_authority_schema"] != CODEX_HANDOFF_ATTEMPT_SCHEMA
        or payload["arm_algorithm"] != CODEX_HANDOFF_ARM_ALGORITHM
        or payload["required_absent_before"] is not True
        or payload["create_new_required"] is not True
        or not _codex_handoff_exact_equal(one_use, expected_one_use)
        or envelope["payload_sha256"] != canonical_json_sha256(payload)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Codex handoff attempt-creation violates its exact closed crosslinks"
        )
    return {
        "schema": CODEX_HANDOFF_ATTEMPT_CREATION_SCHEMA,
        "payload": {**payload, "one_use_policy": one_use},
        "payload_sha256": envelope["payload_sha256"],
    }


def canonical_original_confirmatory_codex_handoff_attempt_authority(
    value: Mapping[str, Any],
    *,
    base_authority: Mapping[str, Any],
    creation_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalize the concrete post-arm attempt; prompt/argv remain prohibited."""

    base = canonical_original_confirmatory_codex_handoff_base_authority(
        base_authority,
        require_operational=True,
    )
    creation = canonical_original_confirmatory_codex_handoff_attempt_creation_authority(
        creation_authority,
        base_authority=base,
    )
    envelope = _codex_handoff_mapping(
        value,
        {"schema", "payload", "payload_sha256"},
        role="Codex handoff attempt envelope",
    )
    if envelope["schema"] != CODEX_HANDOFF_ATTEMPT_SCHEMA:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Codex handoff attempt schema differs"
        )
    payload = _codex_handoff_mapping(
        envelope["payload"],
        {
            "authority_scope",
            "attempt_creation_authority_payload_sha256",
            "base_authority_payload_sha256",
            "attempt_id",
            "session_id",
            "turn_id",
            "marker",
            "marker_nonce_hex",
            "armed_at_utc",
            "expires_at_utc",
            "session_file_identity",
            "pre_arm",
            "watch_registration",
            "success_template_policy_root_sha256",
            "diagnosis_template_policy_root_sha256",
            "branch_selection_policy",
            "one_use_policy",
        },
        role="Codex handoff attempt payload",
    )
    _codex_handoff_uuid(
        payload["attempt_id"], role="Codex handoff attempt id", version_four=True
    )
    session_id = _codex_handoff_uuid(
        payload["session_id"], role="Codex handoff attempt session id"
    )
    turn_id = _codex_handoff_uuid(
        payload["turn_id"], role="Codex handoff attempt turn id"
    )
    marker_nonce = _sha256(
        payload["marker_nonce_hex"], role="Codex handoff marker nonce"
    )
    marker = payload["marker"]
    armed = _codex_handoff_timestamp_value(
        payload["armed_at_utc"], role="Codex handoff armed time"
    )
    expires = _codex_handoff_timestamp_value(
        payload["expires_at_utc"], role="Codex handoff expiry"
    )
    attempt_identity = _canonical_codex_handoff_session_file_identity(
        payload["session_file_identity"],
        role="Codex handoff attempt session identity",
    )
    pre_arm = _codex_handoff_mapping(
        payload["pre_arm"],
        {"offset_bytes", "prefix_sha256", "record_count", "ends_with_lf"},
        role="Codex handoff pre-arm snapshot",
    )
    offset = _codex_handoff_byte_count(
        pre_arm["offset_bytes"], role="Codex handoff pre-arm offset"
    )
    watch = _codex_handoff_mapping(
        payload["watch_registration"],
        {
            "registered_at_utc",
            "primitive",
            "parent_path",
            "target_filename",
            "watch_subtree",
            "notify_filter",
            "buffer_bytes",
            "armed_before_snapshot",
        },
        role="Codex handoff watch registration",
    )
    registered = _codex_handoff_timestamp_value(
        watch["registered_at_utc"], role="Codex handoff watch registration time"
    )
    parent_path = _codex_handoff_windows_path(
        watch["parent_path"], role="Codex handoff watch parent"
    )
    target_filename = watch["target_filename"]
    one_use = _codex_handoff_mapping(
        payload["one_use_policy"],
        {
            "attempt_number",
            "maximum_attempts",
            "automatic_retry_allowed",
            "arm_receipt_create_new_required",
            "attempt_authority_create_new_required",
            "wake_intent_create_new_required",
            "wake_intent_required_before_spawn",
            "branch_selection_time",
            "rendered_prompt_at_arm_allowed",
        },
        role="Codex handoff one-use policy",
    )
    base_payload = cast(dict[str, Any], base["payload"])
    base_session = cast(dict[str, Any], base_payload["session_origin"])
    base_limits = cast(dict[str, Any], base_payload["limits"])
    creation_payload = cast(dict[str, Any], creation["payload"])
    session_parent, separator, session_leaf = cast(
        str, base_session["session_jsonl_path"]
    ).rpartition("\\")
    expected_watch = {
        "registered_at_utc": watch["registered_at_utc"],
        "primitive": "ReadDirectoryChangesW_overlapped_then_WaitForSingleObject_INFINITE",
        "parent_path": session_parent,
        "target_filename": session_leaf,
        "watch_subtree": False,
        "notify_filter": [
            "FILE_NOTIFY_CHANGE_FILE_NAME",
            "FILE_NOTIFY_CHANGE_SIZE",
            "FILE_NOTIFY_CHANGE_LAST_WRITE",
        ],
        "buffer_bytes": CODEX_HANDOFF_WATCHER_BUFFER_BYTES,
        "armed_before_snapshot": True,
    }
    expected_one_use = {
        "attempt_number": 1,
        "maximum_attempts": 1,
        "automatic_retry_allowed": False,
        "arm_receipt_create_new_required": True,
        "attempt_authority_create_new_required": True,
        "wake_intent_create_new_required": True,
        "wake_intent_required_before_spawn": True,
        "branch_selection_time": (
            "postterminal_after_terminal_handoff_receipt_validation"
        ),
        "rendered_prompt_at_arm_allowed": False,
    }
    if (
        payload["authority_scope"] != CODEX_HANDOFF_ATTEMPT_SCOPE
        or payload["attempt_creation_authority_payload_sha256"]
        != creation["payload_sha256"]
        or payload["base_authority_payload_sha256"] != base["payload_sha256"]
        or session_id != base_session["session_id"]
        or session_id != creation_payload["session_id"]
        or turn_id != creation_payload["turn_id"]
        or marker_nonce != creation_payload["marker_nonce_hex"]
        or marker != creation_payload["marker"]
        or not isinstance(marker, str)
        or _CODEX_HANDOFF_MARKER.fullmatch(marker) is None
        or marker != f"AANCA_CURRENT_SESSION_IDLE_{marker_nonce}"
        or expires != armed + timedelta(milliseconds=CODEX_HANDOFF_BOUNDARY_MAX_AGE_MS)
        or registered > armed
        or not _codex_handoff_exact_equal(
            attempt_identity, base_session["session_file_identity"]
        )
        or offset > base_limits["max_session_file_bytes"]
        or _sha256(pre_arm["prefix_sha256"], role="pre-arm prefix SHA-256")
        != pre_arm["prefix_sha256"]
        or type(pre_arm["record_count"]) is not int
        or pre_arm["record_count"] <= 0
        or pre_arm["ends_with_lf"] is not True
        or not separator
        or not isinstance(target_filename, str)
        or not target_filename
        or any(character in target_filename for character in "\\/:\x00")
        or target_filename in {".", ".."}
        or parent_path != session_parent
        or target_filename != session_leaf
        or not _codex_handoff_exact_equal(watch, expected_watch)
        or payload["success_template_policy_root_sha256"]
        != creation_payload["success_template_policy_root_sha256"]
        or payload["diagnosis_template_policy_root_sha256"]
        != creation_payload["diagnosis_template_policy_root_sha256"]
        or payload["branch_selection_policy"] != CODEX_HANDOFF_BRANCH_SELECTION_POLICY
        or not _codex_handoff_exact_equal(one_use, expected_one_use)
        or envelope["payload_sha256"] != canonical_json_sha256(payload)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Codex handoff attempt violates its exact one-use crosslinks"
        )
    return {
        "schema": CODEX_HANDOFF_ATTEMPT_SCHEMA,
        "payload": {
            **payload,
            "session_file_identity": attempt_identity,
            "pre_arm": pre_arm,
            "watch_registration": watch,
            "one_use_policy": one_use,
        },
        "payload_sha256": envelope["payload_sha256"],
    }


def _canonical_environment(value: Any, *, role: str) -> dict[str, str]:
    raw = _mapping(value, role=role)
    result: dict[str, str] = {}
    casefolded: set[str] = set()
    for key in sorted(raw):
        item = raw[key]
        if (
            not key
            or key != key.upper()
            or "\x00" in key
            or "=" in key
            or key.casefold() in casefolded
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                f"{role} contains a non-uppercase, colliding, or invalid name"
            )
        if not isinstance(item, str) or not item or "\x00" in item:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                f"{role} contains an invalid value for {key!r}"
            )
        casefolded.add(key.casefold())
        result[key] = item
    return result


def _canonical_windows_environment_path(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be one canonical absolute Windows path"
        )
    normalized = ntpath.normpath(value)
    drive, tail = ntpath.splitdrive(normalized)
    if (
        normalized != value
        or not drive
        or not tail.startswith("\\")
        or normalized.endswith("\\")
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be one canonical absolute Windows path"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class ExpectedLaunchEnvironmentEnvelopeV1:
    """Full explicit supervisor/child environment; ambient extras are forbidden."""

    attempt_nonce_key: str
    attempt_nonce: str
    supervisor_environment: Mapping[str, str]
    supervisor_environment_sha256: str
    child_environment: Mapping[str, str]
    exact_environment_sha256: str
    launch_environment_root_sha256: str
    envelope_sha256: str

    def payload_without_self_hash(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": EXPECTED_LAUNCH_ENVIRONMENT_ENVELOPE_POLICY,
            "attempt_nonce_key": self.attempt_nonce_key,
            "attempt_nonce": self.attempt_nonce,
            "supervisor_environment": dict(self.supervisor_environment),
            "supervisor_environment_sha256": self.supervisor_environment_sha256,
            "child_environment": dict(self.child_environment),
            "exact_environment_sha256": self.exact_environment_sha256,
            "launch_environment_root_sha256": self.launch_environment_root_sha256,
            "environment_names_uppercase": True,
            "case_collisions_rejected": True,
            "nul_and_equals_in_names_rejected": True,
            "nul_in_values_rejected": True,
            "unspecified_inherited_variables_allowed": False,
            "extra_variables_allowed": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.payload_without_self_hash(),
            "envelope_sha256": self.envelope_sha256,
        }


_EXPECTED_ENVIRONMENT_FIELDS = {
    "schema_version",
    "policy",
    "attempt_nonce_key",
    "attempt_nonce",
    "supervisor_environment",
    "supervisor_environment_sha256",
    "child_environment",
    "exact_environment_sha256",
    "launch_environment_root_sha256",
    "environment_names_uppercase",
    "case_collisions_rejected",
    "nul_and_equals_in_names_rejected",
    "nul_in_values_rejected",
    "unspecified_inherited_variables_allowed",
    "extra_variables_allowed",
    "envelope_sha256",
}


def canonical_expected_launch_environment_envelope_v1(
    value: ExpectedLaunchEnvironmentEnvelopeV1 | Mapping[str, Any],
) -> ExpectedLaunchEnvironmentEnvelopeV1:
    raw = (
        value.as_dict()
        if type(value) is ExpectedLaunchEnvironmentEnvelopeV1
        else _mapping(value, role="expected launch environment envelope")
    )
    if set(raw) != _EXPECTED_ENVIRONMENT_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "expected launch environment envelope has an unexpected field set"
        )
    supervisor = _canonical_environment(
        raw["supervisor_environment"],
        role="expected supervisor environment",
    )
    child = _canonical_environment(
        raw["child_environment"],
        role="expected child environment",
    )
    nonce = raw["attempt_nonce"]
    nonce_key = raw["attempt_nonce_key"]
    expected_supervisor_names = set(SCIENCE_SUPERVISOR_ENVIRONMENT_NAMES)
    expected_child = {**supervisor, SUPERVISOR_ATTEMPT_NONCE_KEY: nonce}
    local_app_data = supervisor.get("LOCALAPPDATA")
    expected_temp = (
        ntpath.join(local_app_data, "Temp") if isinstance(local_app_data, str) else None
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != EXPECTED_LAUNCH_ENVIRONMENT_ENVELOPE_POLICY
        or nonce_key != SUPERVISOR_ATTEMPT_NONCE_KEY
        or not isinstance(nonce, str)
        or _NONCE.fullmatch(nonce) is None
        or set(supervisor) != expected_supervisor_names
        or set(child) != {SUPERVISOR_ATTEMPT_NONCE_KEY, *expected_supervisor_names}
        or child != dict(sorted(expected_child.items()))
        or supervisor.get("TEMP") != expected_temp
        or supervisor.get("TMP") != expected_temp
        or raw["supervisor_environment_sha256"] != canonical_json_sha256(supervisor)
        or raw["exact_environment_sha256"] != canonical_json_sha256(child)
        or raw["environment_names_uppercase"] is not True
        or raw["case_collisions_rejected"] is not True
        or raw["nul_and_equals_in_names_rejected"] is not True
        or raw["nul_in_values_rejected"] is not True
        or raw["unspecified_inherited_variables_allowed"] is not False
        or raw["extra_variables_allowed"] is not False
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "expected launch environment violates its closed exact5/exact6 policy"
        )
    for name in SCIENCE_SUPERVISOR_ENVIRONMENT_NAMES:
        _canonical_windows_environment_path(
            supervisor[name],
            role=f"science supervisor environment {name}",
        )
    root = canonical_json_sha256(
        {
            "supervisor_environment_sha256": canonical_json_sha256(supervisor),
            "child_environment_sha256": canonical_json_sha256(child),
            "attempt_nonce_key": SUPERVISOR_ATTEMPT_NONCE_KEY,
            "attempt_nonce": nonce,
        }
    )
    unsigned = {key: item for key, item in raw.items() if key != "envelope_sha256"}
    unsigned.update(
        {
            "supervisor_environment": supervisor,
            "supervisor_environment_sha256": canonical_json_sha256(supervisor),
            "child_environment": child,
            "exact_environment_sha256": canonical_json_sha256(child),
            "launch_environment_root_sha256": root,
        }
    )
    envelope_sha256 = canonical_json_sha256(unsigned)
    if (
        raw["launch_environment_root_sha256"] != root
        or raw["envelope_sha256"] != envelope_sha256
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "expected launch environment envelope hash/root differs"
        )
    return ExpectedLaunchEnvironmentEnvelopeV1(
        attempt_nonce_key=SUPERVISOR_ATTEMPT_NONCE_KEY,
        attempt_nonce=nonce,
        supervisor_environment=supervisor,
        supervisor_environment_sha256=canonical_json_sha256(supervisor),
        child_environment=child,
        exact_environment_sha256=canonical_json_sha256(child),
        launch_environment_root_sha256=root,
        envelope_sha256=envelope_sha256,
    )


def build_expected_launch_environment_envelope_v1(
    *,
    attempt_nonce: str,
    supervisor_environment: Mapping[str, str],
    child_environment: Mapping[str, str],
) -> ExpectedLaunchEnvironmentEnvelopeV1:
    supervisor = _canonical_environment(
        supervisor_environment,
        role="expected supervisor environment",
    )
    child = _canonical_environment(
        child_environment,
        role="expected child environment",
    )
    if child.get(SUPERVISOR_ATTEMPT_NONCE_KEY) != attempt_nonce:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "child environment lacks the exact sole attempt nonce"
        )
    supervisor_sha256 = canonical_json_sha256(supervisor)
    child_sha256 = canonical_json_sha256(child)
    root = canonical_json_sha256(
        {
            "supervisor_environment_sha256": supervisor_sha256,
            "child_environment_sha256": child_sha256,
            "attempt_nonce_key": SUPERVISOR_ATTEMPT_NONCE_KEY,
            "attempt_nonce": attempt_nonce,
        }
    )
    provisional = ExpectedLaunchEnvironmentEnvelopeV1(
        attempt_nonce_key=SUPERVISOR_ATTEMPT_NONCE_KEY,
        attempt_nonce=attempt_nonce,
        supervisor_environment=supervisor,
        supervisor_environment_sha256=supervisor_sha256,
        child_environment=child,
        exact_environment_sha256=child_sha256,
        launch_environment_root_sha256=root,
        envelope_sha256="0" * 64,
    )
    unsigned = provisional.payload_without_self_hash()
    return canonical_expected_launch_environment_envelope_v1(
        {**unsigned, "envelope_sha256": canonical_json_sha256(unsigned)}
    )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryProcessEnvironmentBinding:
    """Exact supervisor/main/verifier environment hashes for one nonce."""

    expected_environment_envelope_sha256: str
    launch_environment_root_sha256: str
    attempt_nonce_key: str
    attempt_nonce_sha256: str
    exact_supervisor_environment_sha256: str
    exact_environment_sha256: str
    exact_integrity_verifier_environment_sha256: str
    binding_sha256: str

    def payload_without_self_hash(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": PROCESS_ENVIRONMENT_BINDING_POLICY,
            "expected_environment_envelope_sha256": (
                self.expected_environment_envelope_sha256
            ),
            "launch_environment_root_sha256": self.launch_environment_root_sha256,
            "attempt_nonce_key": self.attempt_nonce_key,
            "attempt_nonce_sha256": self.attempt_nonce_sha256,
            "exact_supervisor_environment_sha256": (
                self.exact_supervisor_environment_sha256
            ),
            "exact_environment_sha256": self.exact_environment_sha256,
            "exact_integrity_verifier_environment_sha256": (
                self.exact_integrity_verifier_environment_sha256
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.payload_without_self_hash(),
            "binding_sha256": self.binding_sha256,
        }


_PROCESS_ENVIRONMENT_BINDING_FIELDS = {
    "schema_version",
    "policy",
    "expected_environment_envelope_sha256",
    "launch_environment_root_sha256",
    "attempt_nonce_key",
    "attempt_nonce_sha256",
    "exact_supervisor_environment_sha256",
    "exact_environment_sha256",
    "exact_integrity_verifier_environment_sha256",
    "binding_sha256",
}


def canonical_original_confirmatory_process_environment_binding(
    value: OriginalConfirmatoryProcessEnvironmentBinding | Mapping[str, Any],
    *,
    expected_environment: ExpectedLaunchEnvironmentEnvelopeV1 | Mapping[str, Any],
) -> OriginalConfirmatoryProcessEnvironmentBinding:
    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatoryProcessEnvironmentBinding
        else _mapping(value, role="process environment binding")
    )
    if set(raw) != _PROCESS_ENVIRONMENT_BINDING_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "process environment binding has an unexpected field set"
        )
    envelope = canonical_expected_launch_environment_envelope_v1(expected_environment)
    unsigned = {key: item for key, item in raw.items() if key != "binding_sha256"}
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != PROCESS_ENVIRONMENT_BINDING_POLICY
        or raw["expected_environment_envelope_sha256"] != envelope.envelope_sha256
        or raw["launch_environment_root_sha256"]
        != envelope.launch_environment_root_sha256
        or raw["attempt_nonce_key"] != SUPERVISOR_ATTEMPT_NONCE_KEY
        or raw["attempt_nonce_sha256"]
        != hashlib.sha256(envelope.attempt_nonce.encode("ascii")).hexdigest()
        or raw["exact_supervisor_environment_sha256"]
        != envelope.supervisor_environment_sha256
        or raw["exact_environment_sha256"] != envelope.exact_environment_sha256
        or raw["exact_integrity_verifier_environment_sha256"]
        != envelope.exact_environment_sha256
        or raw["binding_sha256"] != canonical_json_sha256(unsigned)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "process environment binding differs from the exact two-map envelope"
        )
    return OriginalConfirmatoryProcessEnvironmentBinding(
        expected_environment_envelope_sha256=raw[
            "expected_environment_envelope_sha256"
        ],
        launch_environment_root_sha256=raw["launch_environment_root_sha256"],
        attempt_nonce_key=SUPERVISOR_ATTEMPT_NONCE_KEY,
        attempt_nonce_sha256=raw["attempt_nonce_sha256"],
        exact_supervisor_environment_sha256=raw["exact_supervisor_environment_sha256"],
        exact_environment_sha256=raw["exact_environment_sha256"],
        exact_integrity_verifier_environment_sha256=raw[
            "exact_integrity_verifier_environment_sha256"
        ],
        binding_sha256=raw["binding_sha256"],
    )


def build_original_confirmatory_process_environment_binding(
    expected_environment: ExpectedLaunchEnvironmentEnvelopeV1 | Mapping[str, Any],
) -> OriginalConfirmatoryProcessEnvironmentBinding:
    envelope = canonical_expected_launch_environment_envelope_v1(expected_environment)
    provisional = OriginalConfirmatoryProcessEnvironmentBinding(
        expected_environment_envelope_sha256=envelope.envelope_sha256,
        launch_environment_root_sha256=envelope.launch_environment_root_sha256,
        attempt_nonce_key=SUPERVISOR_ATTEMPT_NONCE_KEY,
        attempt_nonce_sha256=hashlib.sha256(
            envelope.attempt_nonce.encode("ascii")
        ).hexdigest(),
        exact_supervisor_environment_sha256=envelope.supervisor_environment_sha256,
        exact_environment_sha256=envelope.exact_environment_sha256,
        exact_integrity_verifier_environment_sha256=envelope.exact_environment_sha256,
        binding_sha256="0" * 64,
    )
    unsigned = provisional.payload_without_self_hash()
    return canonical_original_confirmatory_process_environment_binding(
        {**unsigned, "binding_sha256": canonical_json_sha256(unsigned)},
        expected_environment=envelope,
    )


CAPSULE_RETAINED_FILE_LEASE_POLICY: Final = (
    "original_confirmatory_capsule_retained_file_lease_v1"
)
CAPSULE_ANCESTOR_LEASE_POLICY: Final = "original_confirmatory_capsule_ancestor_lease_v1"
INTERPRETER_RETAINED_FILE_LEASE_POLICY: Final = (
    "original_confirmatory_interpreter_retained_file_lease_v1"
)
INTERPRETER_ANCESTOR_LEASE_POLICY: Final = (
    "original_confirmatory_interpreter_ancestor_lease_v1"
)
RUNTIME_INTERPRETER_RETAINED_FILE_LEASE_POLICY: Final = (
    "original_confirmatory_runtime_interpreter_retained_file_lease_v1"
)
RUNTIME_INTERPRETER_ANCESTOR_LEASE_POLICY: Final = (
    "original_confirmatory_runtime_interpreter_ancestor_lease_v1"
)
PYTHON_RUNTIME_RESOLUTION_POLICY: Final = (
    "windows_venv_redirector_native_base_executable_v1"
)
EXECUTABLE_LEAF_LEASE_DISPOSITION: Final = (
    "opened_before_first_phase_createprocess_retained_through_all_phase_waitforexit_v1"
)
EXECUTABLE_ANCESTOR_LEASE_DISPOSITION: Final = "directory_handles_opened_before_first_phase_createprocess_retained_through_all_phase_waitforexit_v1"
EXECUTABLE_LEAF_ACCESS_MASK: Final = 0x80000000
EXECUTABLE_ANCESTOR_ACCESS_MASK: Final = 0x80000080


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryCapsuleLeaseIdentity:
    """Q-bound leaf-file identity held by supervisor-v2 across every phase."""

    path: Path
    volume_serial_number: int
    file_id_128: str
    size_bytes: int
    sha256: str
    file_attributes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": CAPSULE_RETAINED_FILE_LEASE_POLICY,
            "path": str(self.path),
            "volume_serial_number": self.volume_serial_number,
            "file_id_128": self.file_id_128,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "file_attributes": self.file_attributes,
            "read_only": True,
            "link_count": 1,
            "named_alternate_data_streams": [],
            "opened_without_reparse_follow": True,
            "access_mask": EXECUTABLE_LEAF_ACCESS_MASK,
            "share_access": ["FILE_SHARE_READ"],
            "write_access": False,
            "delete_access": False,
            "retained_through_each_exact_phase_launch": True,
            "owner_process_identity_required": True,
            "handle_slot_required": True,
            "acquisition_disposition": EXECUTABLE_LEAF_LEASE_DISPOSITION,
        }


_CAPSULE_LEASE_IDENTITY_FIELDS = {
    "schema_version",
    "policy",
    "path",
    "volume_serial_number",
    "file_id_128",
    "size_bytes",
    "sha256",
    "file_attributes",
    "read_only",
    "link_count",
    "named_alternate_data_streams",
    "opened_without_reparse_follow",
    "access_mask",
    "share_access",
    "write_access",
    "delete_access",
    "retained_through_each_exact_phase_launch",
    "owner_process_identity_required",
    "handle_slot_required",
    "acquisition_disposition",
}


def canonical_original_confirmatory_capsule_lease_identity(
    value: OriginalConfirmatoryCapsuleLeaseIdentity | Mapping[str, Any],
) -> OriginalConfirmatoryCapsuleLeaseIdentity:
    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatoryCapsuleLeaseIdentity
        else _mapping(value, role="capsule retained-file lease identity")
    )
    if set(raw) != _CAPSULE_LEASE_IDENTITY_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "capsule retained-file lease identity has an unexpected field set"
        )
    path = Path(_absolute_path(raw["path"], role="capsule lease path"))
    volume = raw["volume_serial_number"]
    attributes = raw["file_attributes"]
    file_id = raw["file_id_128"]
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != CAPSULE_RETAINED_FILE_LEASE_POLICY
        or type(volume) is not int
        or volume < 0
        or not isinstance(file_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", file_id) is None
        or type(attributes) is not int
        or attributes < 0
        or attributes & 0x400
        or not attributes & 0x1
        or raw["read_only"] is not True
        or raw["link_count"] != 1
        or type(raw["link_count"]) is not int
        or not _strict_json_value_equal(raw["named_alternate_data_streams"], [])
        or raw["opened_without_reparse_follow"] is not True
        or raw["access_mask"] != EXECUTABLE_LEAF_ACCESS_MASK
        or type(raw["access_mask"]) is not int
        or not _strict_json_value_equal(raw["share_access"], ["FILE_SHARE_READ"])
        or raw["write_access"] is not False
        or raw["delete_access"] is not False
        or raw["retained_through_each_exact_phase_launch"] is not True
        or raw["owner_process_identity_required"] is not True
        or raw["handle_slot_required"] is not True
        or raw["acquisition_disposition"] != EXECUTABLE_LEAF_LEASE_DISPOSITION
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "capsule retained-file lease identity violates its exact policy"
        )
    return OriginalConfirmatoryCapsuleLeaseIdentity(
        path=path,
        volume_serial_number=volume,
        file_id_128=file_id,
        size_bytes=_positive_int(raw["size_bytes"], role="capsule lease size"),
        sha256=_sha256(raw["sha256"], role="capsule lease content"),
        file_attributes=attributes,
    )


def build_original_confirmatory_capsule_lease_identity(
    *,
    path: str | Path,
    volume_serial_number: int,
    file_id_128: str,
    size_bytes: int,
    sha256: str,
    file_attributes: int,
) -> OriginalConfirmatoryCapsuleLeaseIdentity:
    return canonical_original_confirmatory_capsule_lease_identity(
        {
            "schema_version": 1,
            "policy": CAPSULE_RETAINED_FILE_LEASE_POLICY,
            "path": str(path),
            "volume_serial_number": volume_serial_number,
            "file_id_128": file_id_128,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "file_attributes": file_attributes,
            "read_only": True,
            "link_count": 1,
            "named_alternate_data_streams": [],
            "opened_without_reparse_follow": True,
            "access_mask": EXECUTABLE_LEAF_ACCESS_MASK,
            "share_access": ["FILE_SHARE_READ"],
            "write_access": False,
            "delete_access": False,
            "retained_through_each_exact_phase_launch": True,
            "owner_process_identity_required": True,
            "handle_slot_required": True,
            "acquisition_disposition": EXECUTABLE_LEAF_LEASE_DISPOSITION,
        }
    )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryCapsuleAncestorLease:
    """Q-bound root-to-leaf directory identities retained by supervisor-v2."""

    anchor_path: Path
    records: tuple[Mapping[str, Any], ...]
    record_count: int
    records_root_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": CAPSULE_ANCESTOR_LEASE_POLICY,
            "anchor_path": str(self.anchor_path),
            "records": [dict(item) for item in self.records],
            "record_count": self.record_count,
            "records_root_sha256": self.records_root_sha256,
            "directory_access_mask": EXECUTABLE_ANCESTOR_ACCESS_MASK,
            "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
            "delete_share": False,
            "write_access": False,
            "retained_through_each_exact_phase_launch": True,
            "owner_process_identity_required": True,
            "handle_slot_per_record_required": True,
            "acquisition_disposition": EXECUTABLE_ANCESTOR_LEASE_DISPOSITION,
        }


_CAPSULE_ANCESTOR_LEASE_FIELDS = {
    "schema_version",
    "policy",
    "anchor_path",
    "records",
    "record_count",
    "records_root_sha256",
    "directory_access_mask",
    "share_access",
    "delete_share",
    "write_access",
    "retained_through_each_exact_phase_launch",
    "owner_process_identity_required",
    "handle_slot_per_record_required",
    "acquisition_disposition",
}
_CAPSULE_ANCESTOR_RECORD_FIELDS = {
    "path",
    "volume_serial_number",
    "file_id_128",
    "file_attributes",
    "reparse_point",
}


def canonical_original_confirmatory_capsule_ancestor_lease(
    value: OriginalConfirmatoryCapsuleAncestorLease | Mapping[str, Any],
) -> OriginalConfirmatoryCapsuleAncestorLease:
    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatoryCapsuleAncestorLease
        else _mapping(value, role="capsule ancestor lease")
    )
    if set(raw) != _CAPSULE_ANCESTOR_LEASE_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "capsule ancestor lease has an unexpected field set"
        )
    anchor = Path(_absolute_path(raw["anchor_path"], role="capsule ancestor anchor"))
    raw_records = raw["records"]
    if not isinstance(raw_records, list) or len(raw_records) != 4:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "capsule ancestor lease must contain the exact four-directory chain"
        )
    records: list[dict[str, Any]] = []
    for order, item in enumerate(raw_records, start=1):
        record = _mapping(item, role=f"capsule ancestor lease record {order}")
        if set(record) != _CAPSULE_ANCESTOR_RECORD_FIELDS:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "capsule ancestor lease record has an unexpected field set"
            )
        path = _absolute_path(
            record["path"],
            role=f"capsule ancestor lease record {order} path",
        )
        volume = record["volume_serial_number"]
        file_id = record["file_id_128"]
        attributes = record["file_attributes"]
        if (
            type(volume) is not int
            or volume < 0
            or not isinstance(file_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", file_id) is None
            or type(attributes) is not int
            or attributes < 0
            or attributes & 0x400
            or record["reparse_point"] is not False
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "capsule ancestor lease record violates its physical policy"
            )
        records.append(
            {
                "path": path,
                "volume_serial_number": volume,
                "file_id_128": file_id,
                "file_attributes": attributes,
                "reparse_point": False,
            }
        )
    paths = [Path(cast(str, item["path"])) for item in records]
    expected_paths = [
        anchor,
        anchor / "artifacts",
        anchor / "artifacts" / "execution_capsules",
        anchor / "artifacts" / "execution_capsules" / paths[-1].name,
    ]
    if (
        paths != expected_paths
        or _SHA256.fullmatch(paths[-1].name) is None
        or type(raw["record_count"]) is not int
        or raw["record_count"] != len(records)
        or raw["records_root_sha256"] != canonical_json_sha256(records)
        or raw["directory_access_mask"] != EXECUTABLE_ANCESTOR_ACCESS_MASK
        or type(raw["directory_access_mask"]) is not int
        or not _strict_json_value_equal(
            raw["share_access"],
            ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        )
        or raw["delete_share"] is not False
        or raw["write_access"] is not False
        or raw["retained_through_each_exact_phase_launch"] is not True
        or raw["owner_process_identity_required"] is not True
        or raw["handle_slot_per_record_required"] is not True
        or raw["acquisition_disposition"] != EXECUTABLE_ANCESTOR_LEASE_DISPOSITION
        or raw["schema_version"] != 1
        or type(raw["schema_version"]) is not int
        or raw["policy"] != CAPSULE_ANCESTOR_LEASE_POLICY
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "capsule ancestor lease violates its exact root-to-leaf policy"
        )
    return OriginalConfirmatoryCapsuleAncestorLease(
        anchor_path=anchor,
        records=tuple(records),
        record_count=len(records),
        records_root_sha256=raw["records_root_sha256"],
    )


def build_original_confirmatory_capsule_ancestor_lease(
    *,
    anchor_path: str | Path,
    records: Sequence[Mapping[str, Any]],
) -> OriginalConfirmatoryCapsuleAncestorLease:
    canonical_records = [dict(item) for item in records]
    return canonical_original_confirmatory_capsule_ancestor_lease(
        {
            "schema_version": 1,
            "policy": CAPSULE_ANCESTOR_LEASE_POLICY,
            "anchor_path": str(anchor_path),
            "records": canonical_records,
            "record_count": len(canonical_records),
            "records_root_sha256": canonical_json_sha256(canonical_records),
            "directory_access_mask": EXECUTABLE_ANCESTOR_ACCESS_MASK,
            "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
            "delete_share": False,
            "write_access": False,
            "retained_through_each_exact_phase_launch": True,
            "owner_process_identity_required": True,
            "handle_slot_per_record_required": True,
            "acquisition_disposition": EXECUTABLE_ANCESTOR_LEASE_DISPOSITION,
        }
    )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryInterpreterLeaseIdentity:
    """Q-bound interpreter leaf held read-only through every phase launch."""

    path: Path
    volume_serial_number: int
    file_id_128: str
    size_bytes: int
    sha256: str
    file_attributes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": INTERPRETER_RETAINED_FILE_LEASE_POLICY,
            "path": str(self.path),
            "volume_serial_number": self.volume_serial_number,
            "file_id_128": self.file_id_128,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "file_attributes": self.file_attributes,
            "regular_file": True,
            "link_count": 1,
            "named_alternate_data_streams": [],
            "opened_without_reparse_follow": True,
            "access_mask": EXECUTABLE_LEAF_ACCESS_MASK,
            "share_access": ["FILE_SHARE_READ"],
            "write_access": False,
            "delete_access": False,
            "retained_through_each_exact_phase_launch": True,
            "owner_process_identity_required": True,
            "handle_slot_required": True,
            "acquisition_disposition": EXECUTABLE_LEAF_LEASE_DISPOSITION,
        }


_INTERPRETER_LEASE_IDENTITY_FIELDS = {
    "schema_version",
    "policy",
    "path",
    "volume_serial_number",
    "file_id_128",
    "size_bytes",
    "sha256",
    "file_attributes",
    "regular_file",
    "link_count",
    "named_alternate_data_streams",
    "opened_without_reparse_follow",
    "access_mask",
    "share_access",
    "write_access",
    "delete_access",
    "retained_through_each_exact_phase_launch",
    "owner_process_identity_required",
    "handle_slot_required",
    "acquisition_disposition",
}


def canonical_original_confirmatory_interpreter_lease_identity(
    value: OriginalConfirmatoryInterpreterLeaseIdentity | Mapping[str, Any],
) -> OriginalConfirmatoryInterpreterLeaseIdentity:
    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatoryInterpreterLeaseIdentity
        else _mapping(value, role="interpreter retained-file lease identity")
    )
    if set(raw) != _INTERPRETER_LEASE_IDENTITY_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "interpreter retained-file lease identity has an unexpected field set"
        )
    file_id = raw["file_id_128"]
    attributes = raw["file_attributes"]
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != INTERPRETER_RETAINED_FILE_LEASE_POLICY
        or not isinstance(file_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", file_id) is None
        or type(attributes) is not int
        or attributes < 0
        or attributes & 0x400
        or raw["regular_file"] is not True
        or raw["link_count"] != 1
        or type(raw["link_count"]) is not int
        or not _strict_json_value_equal(raw["named_alternate_data_streams"], [])
        or raw["opened_without_reparse_follow"] is not True
        or raw["access_mask"] != EXECUTABLE_LEAF_ACCESS_MASK
        or type(raw["access_mask"]) is not int
        or not _strict_json_value_equal(raw["share_access"], ["FILE_SHARE_READ"])
        or raw["write_access"] is not False
        or raw["delete_access"] is not False
        or raw["retained_through_each_exact_phase_launch"] is not True
        or raw["owner_process_identity_required"] is not True
        or raw["handle_slot_required"] is not True
        or raw["acquisition_disposition"] != EXECUTABLE_LEAF_LEASE_DISPOSITION
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "interpreter retained-file lease identity violates its exact policy"
        )
    return OriginalConfirmatoryInterpreterLeaseIdentity(
        path=Path(_absolute_path(raw["path"], role="interpreter lease path")),
        volume_serial_number=_nonnegative_int(
            raw["volume_serial_number"],
            role="interpreter lease volume serial",
        ),
        file_id_128=file_id,
        size_bytes=_positive_int(raw["size_bytes"], role="interpreter lease size"),
        sha256=_sha256(raw["sha256"], role="interpreter lease content"),
        file_attributes=attributes,
    )


def build_original_confirmatory_interpreter_lease_identity(
    *,
    path: str | Path,
    volume_serial_number: int,
    file_id_128: str,
    size_bytes: int,
    sha256: str,
    file_attributes: int,
) -> OriginalConfirmatoryInterpreterLeaseIdentity:
    return canonical_original_confirmatory_interpreter_lease_identity(
        {
            "schema_version": 1,
            "policy": INTERPRETER_RETAINED_FILE_LEASE_POLICY,
            "path": str(path),
            "volume_serial_number": volume_serial_number,
            "file_id_128": file_id_128,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "file_attributes": file_attributes,
            "regular_file": True,
            "link_count": 1,
            "named_alternate_data_streams": [],
            "opened_without_reparse_follow": True,
            "access_mask": EXECUTABLE_LEAF_ACCESS_MASK,
            "share_access": ["FILE_SHARE_READ"],
            "write_access": False,
            "delete_access": False,
            "retained_through_each_exact_phase_launch": True,
            "owner_process_identity_required": True,
            "handle_slot_required": True,
            "acquisition_disposition": EXECUTABLE_LEAF_LEASE_DISPOSITION,
        }
    )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryInterpreterAncestorLease:
    """Contiguous no-reparse interpreter parent chain retained for launch."""

    anchor_path: Path
    records: tuple[Mapping[str, Any], ...]
    record_count: int
    records_root_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": INTERPRETER_ANCESTOR_LEASE_POLICY,
            "anchor_path": str(self.anchor_path),
            "records": [dict(item) for item in self.records],
            "record_count": self.record_count,
            "records_root_sha256": self.records_root_sha256,
            "retained_through_each_exact_phase_launch": True,
            "directory_access_mask": EXECUTABLE_ANCESTOR_ACCESS_MASK,
            "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
            "delete_share": False,
            "write_access": False,
            "owner_process_identity_required": True,
            "handle_slot_per_record_required": True,
            "acquisition_disposition": EXECUTABLE_ANCESTOR_LEASE_DISPOSITION,
        }


_INTERPRETER_ANCESTOR_LEASE_FIELDS = {
    "schema_version",
    "policy",
    "anchor_path",
    "records",
    "record_count",
    "records_root_sha256",
    "retained_through_each_exact_phase_launch",
    "directory_access_mask",
    "share_access",
    "delete_share",
    "write_access",
    "owner_process_identity_required",
    "handle_slot_per_record_required",
    "acquisition_disposition",
}


def canonical_original_confirmatory_interpreter_ancestor_lease(
    value: OriginalConfirmatoryInterpreterAncestorLease | Mapping[str, Any],
) -> OriginalConfirmatoryInterpreterAncestorLease:
    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatoryInterpreterAncestorLease
        else _mapping(value, role="interpreter ancestor lease")
    )
    if set(raw) != _INTERPRETER_ANCESTOR_LEASE_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "interpreter ancestor lease has an unexpected field set"
        )
    anchor = Path(
        _absolute_path(raw["anchor_path"], role="interpreter ancestor anchor")
    )
    raw_records = raw["records"]
    if not isinstance(raw_records, list) or not raw_records or len(raw_records) > 64:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "interpreter ancestor lease has an invalid chain length"
        )
    records: list[dict[str, Any]] = []
    for order, item in enumerate(raw_records, start=1):
        record = _mapping(item, role=f"interpreter ancestor record {order}")
        if set(record) != _CAPSULE_ANCESTOR_RECORD_FIELDS:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "interpreter ancestor record has an unexpected field set"
            )
        path = Path(
            _absolute_path(
                record["path"],
                role=f"interpreter ancestor record {order} path",
            )
        )
        file_id = record["file_id_128"]
        attributes = record["file_attributes"]
        if (
            not isinstance(file_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", file_id) is None
            or type(attributes) is not int
            or attributes < 0
            or attributes & 0x400
            or record["reparse_point"] is not False
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "interpreter ancestor record violates its physical policy"
            )
        records.append(
            {
                "path": str(path),
                "volume_serial_number": _nonnegative_int(
                    record["volume_serial_number"],
                    role=f"interpreter ancestor record {order} volume",
                ),
                "file_id_128": file_id,
                "file_attributes": attributes,
                "reparse_point": False,
            }
        )
    paths = [Path(cast(str, item["path"])) for item in records]
    if (
        paths[0] != anchor
        or any(child.parent != parent for parent, child in pairwise(paths))
        or raw["record_count"] != len(records)
        or type(raw["record_count"]) is not int
        or raw["records_root_sha256"] != canonical_json_sha256(records)
        or raw["retained_through_each_exact_phase_launch"] is not True
        or raw["directory_access_mask"] != EXECUTABLE_ANCESTOR_ACCESS_MASK
        or type(raw["directory_access_mask"]) is not int
        or not _strict_json_value_equal(
            raw["share_access"],
            ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        )
        or raw["delete_share"] is not False
        or raw["write_access"] is not False
        or raw["owner_process_identity_required"] is not True
        or raw["handle_slot_per_record_required"] is not True
        or raw["acquisition_disposition"] != EXECUTABLE_ANCESTOR_LEASE_DISPOSITION
        or raw["schema_version"] != 1
        or type(raw["schema_version"]) is not int
        or raw["policy"] != INTERPRETER_ANCESTOR_LEASE_POLICY
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "interpreter ancestor lease violates its exact contiguous policy"
        )
    return OriginalConfirmatoryInterpreterAncestorLease(
        anchor_path=anchor,
        records=tuple(records),
        record_count=len(records),
        records_root_sha256=raw["records_root_sha256"],
    )


def build_original_confirmatory_interpreter_ancestor_lease(
    *,
    anchor_path: str | Path,
    records: Sequence[Mapping[str, Any]],
) -> OriginalConfirmatoryInterpreterAncestorLease:
    canonical_records = [dict(item) for item in records]
    return canonical_original_confirmatory_interpreter_ancestor_lease(
        {
            "schema_version": 1,
            "policy": INTERPRETER_ANCESTOR_LEASE_POLICY,
            "anchor_path": str(anchor_path),
            "records": canonical_records,
            "record_count": len(canonical_records),
            "records_root_sha256": canonical_json_sha256(canonical_records),
            "retained_through_each_exact_phase_launch": True,
            "directory_access_mask": EXECUTABLE_ANCESTOR_ACCESS_MASK,
            "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
            "delete_share": False,
            "write_access": False,
            "owner_process_identity_required": True,
            "handle_slot_per_record_required": True,
            "acquisition_disposition": EXECUTABLE_ANCESTOR_LEASE_DISPOSITION,
        }
    )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryRuntimeInterpreterLeaseIdentity(
    OriginalConfirmatoryInterpreterLeaseIdentity
):
    """Native CPython image retained alongside the logical venv redirector."""

    def as_dict(self) -> dict[str, Any]:
        return {
            **OriginalConfirmatoryInterpreterLeaseIdentity.as_dict(self),
            "policy": RUNTIME_INTERPRETER_RETAINED_FILE_LEASE_POLICY,
        }


def canonical_original_confirmatory_runtime_interpreter_lease_identity(
    value: OriginalConfirmatoryRuntimeInterpreterLeaseIdentity | Mapping[str, Any],
) -> OriginalConfirmatoryRuntimeInterpreterLeaseIdentity:
    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatoryRuntimeInterpreterLeaseIdentity
        else _mapping(
            value,
            role="runtime interpreter retained-file lease identity",
        )
    )
    if (
        set(raw) != _INTERPRETER_LEASE_IDENTITY_FIELDS
        or raw["policy"] != RUNTIME_INTERPRETER_RETAINED_FILE_LEASE_POLICY
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "runtime interpreter retained-file lease identity has an unexpected field set or policy"
        )
    logical = canonical_original_confirmatory_interpreter_lease_identity(
        {
            **raw,
            "policy": INTERPRETER_RETAINED_FILE_LEASE_POLICY,
        }
    )
    return OriginalConfirmatoryRuntimeInterpreterLeaseIdentity(
        path=logical.path,
        volume_serial_number=logical.volume_serial_number,
        file_id_128=logical.file_id_128,
        size_bytes=logical.size_bytes,
        sha256=logical.sha256,
        file_attributes=logical.file_attributes,
    )


def build_original_confirmatory_runtime_interpreter_lease_identity(
    *,
    path: str | Path,
    volume_serial_number: int,
    file_id_128: str,
    size_bytes: int,
    sha256: str,
    file_attributes: int,
) -> OriginalConfirmatoryRuntimeInterpreterLeaseIdentity:
    return canonical_original_confirmatory_runtime_interpreter_lease_identity(
        {
            "schema_version": 1,
            "policy": RUNTIME_INTERPRETER_RETAINED_FILE_LEASE_POLICY,
            "path": str(path),
            "volume_serial_number": volume_serial_number,
            "file_id_128": file_id_128,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "file_attributes": file_attributes,
            "regular_file": True,
            "link_count": 1,
            "named_alternate_data_streams": [],
            "opened_without_reparse_follow": True,
            "access_mask": EXECUTABLE_LEAF_ACCESS_MASK,
            "share_access": ["FILE_SHARE_READ"],
            "write_access": False,
            "delete_access": False,
            "retained_through_each_exact_phase_launch": True,
            "owner_process_identity_required": True,
            "handle_slot_required": True,
            "acquisition_disposition": EXECUTABLE_LEAF_LEASE_DISPOSITION,
        }
    )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryRuntimeInterpreterAncestorLease(
    OriginalConfirmatoryInterpreterAncestorLease
):
    """Contiguous native-runtime chain anchored at the user-profile root."""

    def as_dict(self) -> dict[str, Any]:
        return {
            **OriginalConfirmatoryInterpreterAncestorLease.as_dict(self),
            "policy": RUNTIME_INTERPRETER_ANCESTOR_LEASE_POLICY,
        }


def canonical_original_confirmatory_runtime_interpreter_ancestor_lease(
    value: OriginalConfirmatoryRuntimeInterpreterAncestorLease | Mapping[str, Any],
) -> OriginalConfirmatoryRuntimeInterpreterAncestorLease:
    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatoryRuntimeInterpreterAncestorLease
        else _mapping(value, role="runtime interpreter ancestor lease")
    )
    if (
        set(raw) != _INTERPRETER_ANCESTOR_LEASE_FIELDS
        or raw["policy"] != RUNTIME_INTERPRETER_ANCESTOR_LEASE_POLICY
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "runtime interpreter ancestor lease has an unexpected field set or policy"
        )
    logical = canonical_original_confirmatory_interpreter_ancestor_lease(
        {
            **raw,
            "policy": INTERPRETER_ANCESTOR_LEASE_POLICY,
        }
    )
    return OriginalConfirmatoryRuntimeInterpreterAncestorLease(
        anchor_path=logical.anchor_path,
        records=logical.records,
        record_count=logical.record_count,
        records_root_sha256=logical.records_root_sha256,
    )


def build_original_confirmatory_runtime_interpreter_ancestor_lease(
    *,
    anchor_path: str | Path,
    records: Sequence[Mapping[str, Any]],
) -> OriginalConfirmatoryRuntimeInterpreterAncestorLease:
    canonical_records = [dict(item) for item in records]
    return canonical_original_confirmatory_runtime_interpreter_ancestor_lease(
        {
            "schema_version": 1,
            "policy": RUNTIME_INTERPRETER_ANCESTOR_LEASE_POLICY,
            "anchor_path": str(anchor_path),
            "records": canonical_records,
            "record_count": len(canonical_records),
            "records_root_sha256": canonical_json_sha256(canonical_records),
            "retained_through_each_exact_phase_launch": True,
            "directory_access_mask": EXECUTABLE_ANCESTOR_ACCESS_MASK,
            "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
            "delete_share": False,
            "write_access": False,
            "owner_process_identity_required": True,
            "handle_slot_per_record_required": True,
            "acquisition_disposition": EXECUTABLE_ANCESTOR_LEASE_DISPOSITION,
        }
    )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryPhysicalFileIdentity:
    """Expanded no-follow file identity shared by READY/ACK/lease receipts."""

    role: str
    path: Path
    volume_serial_number: int
    file_id_128: str
    device: int
    inode: int
    size_bytes: int
    mode: int
    file_attributes: int
    modified_time_ns: int
    changed_time_ns: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": NO_FOLLOW_PHYSICAL_FILE_IDENTITY_POLICY,
            "role": self.role,
            "path": str(self.path),
            "volume_serial_number": self.volume_serial_number,
            "file_id_128": self.file_id_128,
            "device": self.device,
            "inode": self.inode,
            "size_bytes": self.size_bytes,
            "mode": self.mode,
            "file_attributes": self.file_attributes,
            "regular_file": True,
            "read_only": True,
            "link_count": 1,
            "modified_time_ns": self.modified_time_ns,
            "changed_time_ns": self.changed_time_ns,
            "sha256": self.sha256,
            "named_alternate_data_streams": [],
            "opened_without_reparse_follow": True,
            "share_access": ["FILE_SHARE_READ"],
        }


_PHYSICAL_FILE_IDENTITY_FIELDS = {
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
_PHYSICAL_FILE_IDENTITY_ROLES = {
    "terminal-client-launcher",
    "terminal-client-launch-intent",
    "preterminal-pin",
    "preterminal-stdout",
    "preterminal-stderr",
    "supervisor-terminal",
    "postwake-lease-receipt",
    "postwake-composed-readback",
    "composed-terminal",
}
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def canonical_original_confirmatory_physical_file_identity(
    value: OriginalConfirmatoryPhysicalFileIdentity | Mapping[str, Any],
    *,
    allowed_roles: Sequence[str] | None = None,
) -> OriginalConfirmatoryPhysicalFileIdentity:
    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatoryPhysicalFileIdentity
        else _mapping(value, role="no-follow physical file identity")
    )
    if set(raw) != _PHYSICAL_FILE_IDENTITY_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "no-follow physical file identity has an unexpected field set"
        )
    role = raw["role"]
    accepted_roles = (
        set(allowed_roles)
        if allowed_roles is not None
        else _PHYSICAL_FILE_IDENTITY_ROLES
    )
    file_id = raw["file_id_128"]
    mode = raw["mode"]
    attributes = raw["file_attributes"]
    size = _nonnegative_int(
        raw["size_bytes"],
        role="physical file identity size",
    )
    sha256 = _sha256(raw["sha256"], role="physical file identity content")
    if (
        not isinstance(role, str)
        or role not in accepted_roles
        or not isinstance(file_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", file_id) is None
        or type(mode) is not int
        or mode < 0
        or mode > 0o7777
        or mode & 0o222
        or type(attributes) is not int
        or attributes < 0
        or attributes & 0x400
        or not attributes & 0x1
        or raw["regular_file"] is not True
        or raw["read_only"] is not True
        or raw["link_count"] != 1
        or type(raw["link_count"]) is not int
        or not _strict_json_value_equal(raw["named_alternate_data_streams"], [])
        or raw["opened_without_reparse_follow"] is not True
        or not _strict_json_value_equal(raw["share_access"], ["FILE_SHARE_READ"])
        or (role == "preterminal-stderr" and (size != 0 or sha256 != _EMPTY_SHA256))
        or (role != "preterminal-stderr" and size == 0)
        or type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != NO_FOLLOW_PHYSICAL_FILE_IDENTITY_POLICY
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "no-follow physical file identity violates its exact policy"
        )
    return OriginalConfirmatoryPhysicalFileIdentity(
        role=role,
        path=Path(_absolute_path(raw["path"], role="physical file identity path")),
        volume_serial_number=_nonnegative_int(
            raw["volume_serial_number"],
            role="physical file volume serial",
        ),
        file_id_128=file_id,
        device=_nonnegative_int(
            raw["device"],
            role="physical file device",
        ),
        inode=_nonnegative_int(raw["inode"], role="physical file inode"),
        size_bytes=size,
        mode=mode,
        file_attributes=attributes,
        modified_time_ns=_nonnegative_int(
            raw["modified_time_ns"],
            role="physical file modified time",
        ),
        changed_time_ns=_nonnegative_int(
            raw["changed_time_ns"],
            role="physical file changed time",
        ),
        sha256=sha256,
    )


def build_original_confirmatory_physical_file_identity(
    *,
    role: str,
    path: str | Path,
    volume_serial_number: int,
    file_id_128: str,
    device: int,
    inode: int,
    size_bytes: int,
    mode: int,
    file_attributes: int,
    modified_time_ns: int,
    changed_time_ns: int,
    sha256: str,
) -> OriginalConfirmatoryPhysicalFileIdentity:
    return canonical_original_confirmatory_physical_file_identity(
        {
            "schema_version": 1,
            "policy": NO_FOLLOW_PHYSICAL_FILE_IDENTITY_POLICY,
            "role": role,
            "path": str(path),
            "volume_serial_number": volume_serial_number,
            "file_id_128": file_id_128,
            "device": device,
            "inode": inode,
            "size_bytes": size_bytes,
            "mode": mode,
            "file_attributes": file_attributes,
            "regular_file": True,
            "read_only": True,
            "link_count": 1,
            "modified_time_ns": modified_time_ns,
            "changed_time_ns": changed_time_ns,
            "sha256": sha256,
            "named_alternate_data_streams": [],
            "opened_without_reparse_follow": True,
            "share_access": ["FILE_SHARE_READ"],
        }
    )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryRetainedHandleBinding:
    """A final file identity tied to one still-live supervisor HANDLE slot."""

    role: str
    physical_identity: OriginalConfirmatoryPhysicalFileIdentity
    owner_pid: int
    owner_creation_time_100ns: int
    owner_windows_boot_time_utc: str
    handle_slot: int
    access_mask: int
    share_mode: int
    acquisition_disposition: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": RETAINED_NATIVE_HANDLE_BINDING_POLICY,
            "role": self.role,
            "physical_identity": self.physical_identity.as_dict(),
            "owner_pid": self.owner_pid,
            "owner_creation_time_100ns": self.owner_creation_time_100ns,
            "owner_windows_boot_time_utc": self.owner_windows_boot_time_utc,
            "handle_slot": self.handle_slot,
            "access_mask": self.access_mask,
            "share_mode": self.share_mode,
            "retained_handle_active": True,
            "acquisition_disposition": self.acquisition_disposition,
        }


_RETAINED_HANDLE_BINDING_FIELDS = {
    "schema_version",
    "policy",
    "role",
    "physical_identity",
    "owner_pid",
    "owner_creation_time_100ns",
    "owner_windows_boot_time_utc",
    "handle_slot",
    "access_mask",
    "share_mode",
    "retained_handle_active",
    "acquisition_disposition",
}
_RETAINED_HANDLE_BINDING_ROLES = {
    "preterminal-pin",
    "preterminal-stdout",
    "preterminal-stderr",
    "supervisor-terminal",
    "postwake-lease-receipt",
    "postwake-composed-readback",
    "composed-terminal",
}
_RETAINED_HANDLE_ACCESS_MASK = 0x80000000
_RETAINED_HANDLE_SHARE_MODE_BY_ROLE = {
    role: (0x1 | 0x2 if role == "preterminal-pin" else 0x1)
    for role in _RETAINED_HANDLE_BINDING_ROLES
}
_RETAINED_HANDLE_DISPOSITION_BY_ROLE = {
    "preterminal-pin": (
        "child_and_supervisor_retained_handles_overlapped_before_ack_v1"
    ),
    "preterminal-stdout": (
        "continuous_parent_rw_handle_and_read_slot_from_before_verifier_launch_v1"
    ),
    "preterminal-stderr": (
        "continuous_parent_rw_handle_and_read_slot_from_before_verifier_launch_v1"
    ),
    "supervisor-terminal": (
        "continuous_parent_rw_handle_and_read_slot_from_before_terminal_serialization_v1"
    ),
    "postwake-lease-receipt": (
        "continuous_parent_rw_handle_and_read_slot_from_before_postwake_lease_serialization_v1"
    ),
    "postwake-composed-readback": (
        "continuous_parent_rw_handle_and_read_slot_from_before_postwake_composed_readback_serialization_v1"
    ),
    "composed-terminal": (
        "duplicated_reduced_read_into_live_supervisor_before_composed_terminal_serialization_v1"
    ),
}


def canonical_original_confirmatory_retained_handle_binding(
    value: OriginalConfirmatoryRetainedHandleBinding | Mapping[str, Any],
    *,
    allowed_roles: Sequence[str] | None = None,
) -> OriginalConfirmatoryRetainedHandleBinding:
    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatoryRetainedHandleBinding
        else _mapping(value, role="retained native handle binding")
    )
    if set(raw) != _RETAINED_HANDLE_BINDING_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "retained native handle binding has an unexpected field set"
        )
    accepted_roles = (
        set(allowed_roles)
        if allowed_roles is not None
        else _RETAINED_HANDLE_BINDING_ROLES
    )
    role = raw["role"]
    identity = canonical_original_confirmatory_physical_file_identity(
        raw["physical_identity"],
        allowed_roles=(role,) if isinstance(role, str) else (),
    )
    disposition = raw["acquisition_disposition"]
    expected_share_mode = _RETAINED_HANDLE_SHARE_MODE_BY_ROLE.get(role)
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != RETAINED_NATIVE_HANDLE_BINDING_POLICY
        or not isinstance(role, str)
        or role not in accepted_roles
        or identity.role != role
        or raw["access_mask"] != _RETAINED_HANDLE_ACCESS_MASK
        or type(raw["access_mask"]) is not int
        or type(raw["share_mode"]) is not int
        or raw["share_mode"] != expected_share_mode
        or raw["retained_handle_active"] is not True
        or disposition != _RETAINED_HANDLE_DISPOSITION_BY_ROLE.get(role)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "retained native handle binding violates its exact policy"
        )
    return OriginalConfirmatoryRetainedHandleBinding(
        role=role,
        physical_identity=identity,
        owner_pid=_positive_int(
            raw["owner_pid"],
            role="retained handle owner PID",
        ),
        owner_creation_time_100ns=_positive_int(
            raw["owner_creation_time_100ns"],
            role="retained handle owner creation time",
        ),
        owner_windows_boot_time_utc=_utc_timestamp(
            raw["owner_windows_boot_time_utc"],
            role="retained handle owner boot time",
        ),
        handle_slot=_positive_int(
            raw["handle_slot"],
            role="retained HANDLE slot",
        ),
        access_mask=_RETAINED_HANDLE_ACCESS_MASK,
        share_mode=expected_share_mode,
        acquisition_disposition=disposition,
    )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryPreserializationHandleBinding:
    """Pre-serialization identity for T or L, intentionally excluding bytes."""

    role: str
    path: Path
    volume_serial_number: int
    file_id_128: str
    owner_pid: int
    owner_creation_time_100ns: int
    owner_windows_boot_time_utc: str
    handle_slot: int
    access_mask: int
    share_mode: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": PRESERIALIZATION_NATIVE_HANDLE_BINDING_POLICY,
            "role": self.role,
            "path": str(self.path),
            "volume_serial_number": self.volume_serial_number,
            "file_id_128": self.file_id_128,
            "owner_pid": self.owner_pid,
            "owner_creation_time_100ns": self.owner_creation_time_100ns,
            "owner_windows_boot_time_utc": self.owner_windows_boot_time_utc,
            "handle_slot": self.handle_slot,
            "access_mask": self.access_mask,
            "share_mode": self.share_mode,
        }


_PRESERIALIZATION_HANDLE_BINDING_FIELDS = {
    "schema_version",
    "policy",
    "role",
    "path",
    "volume_serial_number",
    "file_id_128",
    "owner_pid",
    "owner_creation_time_100ns",
    "owner_windows_boot_time_utc",
    "handle_slot",
    "access_mask",
    "share_mode",
}
_PRESERIALIZATION_HANDLE_BINDING_ROLES = {
    "supervisor-terminal",
    "postwake-lease-receipt",
    "postwake-composed-readback",
}


def canonical_original_confirmatory_preserialization_handle_binding(
    value: OriginalConfirmatoryPreserializationHandleBinding | Mapping[str, Any],
    *,
    expected_role: str,
) -> OriginalConfirmatoryPreserializationHandleBinding:
    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatoryPreserializationHandleBinding
        else _mapping(value, role="preserialization native handle binding")
    )
    if set(raw) != _PRESERIALIZATION_HANDLE_BINDING_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "preserialization native handle binding has an unexpected field set"
        )
    file_id = raw["file_id_128"]
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != PRESERIALIZATION_NATIVE_HANDLE_BINDING_POLICY
        or expected_role not in _PRESERIALIZATION_HANDLE_BINDING_ROLES
        or raw["role"] != expected_role
        or not isinstance(file_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", file_id) is None
        or raw["access_mask"] != 0xC0000000
        or type(raw["access_mask"]) is not int
        or raw["share_mode"] != 1
        or type(raw["share_mode"]) is not int
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "preserialization native handle binding violates its exact policy"
        )
    return OriginalConfirmatoryPreserializationHandleBinding(
        role=expected_role,
        path=Path(
            _absolute_path(
                raw["path"],
                role="preserialization handle path",
            )
        ),
        volume_serial_number=_nonnegative_int(
            raw["volume_serial_number"],
            role="preserialization handle volume serial",
        ),
        file_id_128=file_id,
        owner_pid=_positive_int(
            raw["owner_pid"],
            role="preserialization handle owner PID",
        ),
        owner_creation_time_100ns=_positive_int(
            raw["owner_creation_time_100ns"],
            role="preserialization handle owner creation time",
        ),
        owner_windows_boot_time_utc=_utc_timestamp(
            raw["owner_windows_boot_time_utc"],
            role="preserialization handle boot time",
        ),
        handle_slot=_positive_int(
            raw["handle_slot"],
            role="preserialization HANDLE slot",
        ),
        access_mask=0xC0000000,
        share_mode=1,
    )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryExecutionCapsule:
    """Q's exact immutable sealed execution-capsule contract."""

    path: Path
    size_bytes: int
    sha256: str
    internal_manifest_sha256: str
    capsule_policy_sha256: str
    entry_contract_sha256: str
    plan_sha256: str
    runtime_release_root_sha256: str
    terminal_release_root_sha256: str
    python_path: Path
    python_sha256: str
    python_lease_identity: OriginalConfirmatoryInterpreterLeaseIdentity
    python_lease_identity_root_sha256: str
    python_ancestor_lease: OriginalConfirmatoryInterpreterAncestorLease
    python_ancestor_lease_root_sha256: str
    python_runtime_resolution_policy: str
    runtime_python_path: Path
    runtime_python_sha256: str
    runtime_python_lease_identity: OriginalConfirmatoryRuntimeInterpreterLeaseIdentity
    runtime_python_lease_identity_root_sha256: str
    runtime_python_ancestor_lease: OriginalConfirmatoryRuntimeInterpreterAncestorLease
    runtime_python_ancestor_lease_root_sha256: str
    capsule_lease_identity: OriginalConfirmatoryCapsuleLeaseIdentity
    capsule_lease_identity_root_sha256: str
    capsule_ancestor_lease: OriginalConfirmatoryCapsuleAncestorLease
    capsule_ancestor_lease_root_sha256: str
    contract_sha256: str

    def payload_without_self_hash(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": EXECUTION_CAPSULE_POLICY,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "internal_manifest_sha256": self.internal_manifest_sha256,
            "capsule_policy_sha256": self.capsule_policy_sha256,
            "entry_contract_sha256": self.entry_contract_sha256,
            "plan_sha256": self.plan_sha256,
            "runtime_release_root_sha256": self.runtime_release_root_sha256,
            "terminal_release_root_sha256": self.terminal_release_root_sha256,
            "python_path": str(self.python_path),
            "python_sha256": self.python_sha256,
            "python_lease_identity": self.python_lease_identity.as_dict(),
            "python_lease_identity_root_sha256": (
                self.python_lease_identity_root_sha256
            ),
            "python_ancestor_lease": self.python_ancestor_lease.as_dict(),
            "python_ancestor_lease_root_sha256": (
                self.python_ancestor_lease_root_sha256
            ),
            "python_runtime_resolution_policy": (self.python_runtime_resolution_policy),
            "runtime_python_path": str(self.runtime_python_path),
            "runtime_python_sha256": self.runtime_python_sha256,
            "runtime_python_lease_identity": (
                self.runtime_python_lease_identity.as_dict()
            ),
            "runtime_python_lease_identity_root_sha256": (
                self.runtime_python_lease_identity_root_sha256
            ),
            "runtime_python_ancestor_lease": (
                self.runtime_python_ancestor_lease.as_dict()
            ),
            "runtime_python_ancestor_lease_root_sha256": (
                self.runtime_python_ancestor_lease_root_sha256
            ),
            "python_isolated_flags": list(CAPSULE_PYTHON_ISOLATED_FLAGS),
            "allowed_modes": list(CAPSULE_ALLOWED_MODES),
            "capsule_lease_identity": self.capsule_lease_identity.as_dict(),
            "capsule_lease_identity_root_sha256": (
                self.capsule_lease_identity_root_sha256
            ),
            "capsule_ancestor_lease": self.capsule_ancestor_lease.as_dict(),
            "capsule_ancestor_lease_root_sha256": (
                self.capsule_ancestor_lease_root_sha256
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.payload_without_self_hash(),
            "contract_sha256": self.contract_sha256,
        }


_EXECUTION_CAPSULE_FIELDS = {
    "schema_version",
    "policy",
    "path",
    "size_bytes",
    "sha256",
    "internal_manifest_sha256",
    "capsule_policy_sha256",
    "entry_contract_sha256",
    "plan_sha256",
    "runtime_release_root_sha256",
    "terminal_release_root_sha256",
    "python_path",
    "python_sha256",
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
    "python_isolated_flags",
    "allowed_modes",
    "capsule_lease_identity",
    "capsule_lease_identity_root_sha256",
    "capsule_ancestor_lease",
    "capsule_ancestor_lease_root_sha256",
    "contract_sha256",
}


def canonical_original_confirmatory_execution_capsule(
    value: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
) -> OriginalConfirmatoryExecutionCapsule:
    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatoryExecutionCapsule
        else _mapping(value, role="Q execution capsule")
    )
    if set(raw) != _EXECUTION_CAPSULE_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q execution capsule has an unexpected field set"
        )
    capsule_path = Path(_absolute_path(raw["path"], role="Q execution capsule path"))
    capsule_size = _positive_int(
        raw["size_bytes"],
        role="Q execution capsule size",
    )
    capsule_sha256 = _sha256(raw["sha256"], role="Q execution capsule")
    leaf_lease = canonical_original_confirmatory_capsule_lease_identity(
        raw["capsule_lease_identity"]
    )
    leaf_lease_root = canonical_json_sha256(leaf_lease.as_dict())
    ancestor_lease = canonical_original_confirmatory_capsule_ancestor_lease(
        raw["capsule_ancestor_lease"]
    )
    ancestor_lease_root = canonical_json_sha256(ancestor_lease.as_dict())
    python_lease = canonical_original_confirmatory_interpreter_lease_identity(
        raw["python_lease_identity"]
    )
    python_lease_root = canonical_json_sha256(python_lease.as_dict())
    python_ancestor = canonical_original_confirmatory_interpreter_ancestor_lease(
        raw["python_ancestor_lease"]
    )
    python_ancestor_root = canonical_json_sha256(python_ancestor.as_dict())
    python_path = Path(
        _absolute_path(raw["python_path"], role="Q execution capsule Python")
    )
    runtime_python_lease = (
        canonical_original_confirmatory_runtime_interpreter_lease_identity(
            raw["runtime_python_lease_identity"]
        )
    )
    runtime_python_lease_root = canonical_json_sha256(runtime_python_lease.as_dict())
    runtime_python_ancestor = (
        canonical_original_confirmatory_runtime_interpreter_ancestor_lease(
            raw["runtime_python_ancestor_lease"]
        )
    )
    runtime_python_ancestor_root = canonical_json_sha256(
        runtime_python_ancestor.as_dict()
    )
    runtime_python_path = Path(
        _absolute_path(
            raw["runtime_python_path"],
            role="Q execution capsule runtime Python",
        )
    )
    # The retained runtime-ancestor lease is Q's sealed authority for the
    # user-profile anchor.  Canonicalization must not consult the verifier's
    # ambient USERPROFILE/HOME: doing so would make identical Q bytes validate
    # differently on another host or under an adversarial environment.
    user_profile_root = runtime_python_ancestor.anchor_path
    try:
        runtime_relative_parent = runtime_python_path.parent.relative_to(
            user_profile_root
        )
    except ValueError as exc:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q runtime Python is outside the exact user-profile root"
        ) from exc
    runtime_expected_paths = [user_profile_root]
    for part in runtime_relative_parent.parts:
        runtime_expected_paths.append(runtime_expected_paths[-1] / part)
    expected_capsule_path = (
        ancestor_lease.anchor_path
        / "artifacts"
        / "execution_capsules"
        / capsule_sha256
        / "original_confirmatory.pyz"
    )
    unsigned = {key: item for key, item in raw.items() if key != "contract_sha256"}
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != EXECUTION_CAPSULE_POLICY
        or not _strict_json_value_equal(
            raw["python_isolated_flags"],
            list(CAPSULE_PYTHON_ISOLATED_FLAGS),
        )
        or not _strict_json_value_equal(
            raw["allowed_modes"],
            list(CAPSULE_ALLOWED_MODES),
        )
        or leaf_lease.path != capsule_path
        or leaf_lease.size_bytes != capsule_size
        or leaf_lease.sha256 != capsule_sha256
        or raw["capsule_lease_identity_root_sha256"] != leaf_lease_root
        or raw["capsule_ancestor_lease_root_sha256"] != ancestor_lease_root
        or Path(cast(str, ancestor_lease.records[-1]["path"])) != capsule_path.parent
        or capsule_path != expected_capsule_path
        or python_lease.path != python_path
        or python_lease.sha256 != raw["python_sha256"]
        or raw["python_lease_identity_root_sha256"] != python_lease_root
        or raw["python_ancestor_lease_root_sha256"] != python_ancestor_root
        or python_ancestor.anchor_path != ancestor_lease.anchor_path
        or [Path(cast(str, item["path"])) for item in python_ancestor.records]
        != [
            ancestor_lease.anchor_path,
            ancestor_lease.anchor_path / ".venv",
            ancestor_lease.anchor_path / ".venv" / "Scripts",
        ]
        or python_path
        != ancestor_lease.anchor_path / ".venv" / "Scripts" / "python.exe"
        or raw["python_runtime_resolution_policy"] != PYTHON_RUNTIME_RESOLUTION_POLICY
        or runtime_python_path == python_path
        or runtime_python_path.name.lower() != "python.exe"
        or runtime_python_lease.path != runtime_python_path
        or runtime_python_lease.sha256 != raw["runtime_python_sha256"]
        or raw["runtime_python_lease_identity_root_sha256"] != runtime_python_lease_root
        or raw["runtime_python_ancestor_lease_root_sha256"]
        != runtime_python_ancestor_root
        or runtime_python_ancestor.anchor_path != user_profile_root
        or [Path(cast(str, item["path"])) for item in runtime_python_ancestor.records]
        != runtime_expected_paths
        or raw["contract_sha256"] != canonical_json_sha256(unsigned)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q execution capsule violates its exact closed policy"
        )
    return OriginalConfirmatoryExecutionCapsule(
        path=capsule_path,
        size_bytes=capsule_size,
        sha256=capsule_sha256,
        internal_manifest_sha256=_sha256(
            raw["internal_manifest_sha256"],
            role="Q execution capsule internal manifest",
        ),
        capsule_policy_sha256=_sha256(
            raw["capsule_policy_sha256"],
            role="Q execution capsule policy",
        ),
        entry_contract_sha256=_sha256(
            raw["entry_contract_sha256"],
            role="Q execution capsule entry contract",
        ),
        plan_sha256=_sha256(raw["plan_sha256"], role="Q execution capsule PLAN"),
        runtime_release_root_sha256=_sha256(
            raw["runtime_release_root_sha256"],
            role="Q execution capsule runtime release",
        ),
        terminal_release_root_sha256=_sha256(
            raw["terminal_release_root_sha256"],
            role="Q execution capsule terminal release",
        ),
        python_path=python_path,
        python_sha256=_sha256(raw["python_sha256"], role="Q execution capsule Python"),
        python_lease_identity=python_lease,
        python_lease_identity_root_sha256=python_lease_root,
        python_ancestor_lease=python_ancestor,
        python_ancestor_lease_root_sha256=python_ancestor_root,
        python_runtime_resolution_policy=PYTHON_RUNTIME_RESOLUTION_POLICY,
        runtime_python_path=runtime_python_path,
        runtime_python_sha256=_sha256(
            raw["runtime_python_sha256"],
            role="Q execution capsule runtime Python",
        ),
        runtime_python_lease_identity=runtime_python_lease,
        runtime_python_lease_identity_root_sha256=runtime_python_lease_root,
        runtime_python_ancestor_lease=runtime_python_ancestor,
        runtime_python_ancestor_lease_root_sha256=runtime_python_ancestor_root,
        capsule_lease_identity=leaf_lease,
        capsule_lease_identity_root_sha256=leaf_lease_root,
        capsule_ancestor_lease=ancestor_lease,
        capsule_ancestor_lease_root_sha256=ancestor_lease_root,
        contract_sha256=raw["contract_sha256"],
    )


def build_original_confirmatory_execution_capsule(
    *,
    path: str | Path,
    size_bytes: int,
    sha256: str,
    internal_manifest_sha256: str,
    capsule_policy_sha256: str,
    entry_contract_sha256: str,
    plan_sha256: str,
    runtime_release_root_sha256: str,
    terminal_release_root_sha256: str,
    python_path: str | Path,
    python_sha256: str,
    python_lease_identity: OriginalConfirmatoryInterpreterLeaseIdentity
    | Mapping[str, Any],
    python_ancestor_lease: OriginalConfirmatoryInterpreterAncestorLease
    | Mapping[str, Any],
    runtime_python_path: str | Path,
    runtime_python_sha256: str,
    runtime_python_lease_identity: OriginalConfirmatoryRuntimeInterpreterLeaseIdentity
    | Mapping[str, Any],
    runtime_python_ancestor_lease: OriginalConfirmatoryRuntimeInterpreterAncestorLease
    | Mapping[str, Any],
    capsule_lease_identity: OriginalConfirmatoryCapsuleLeaseIdentity
    | Mapping[str, Any],
    capsule_ancestor_lease: OriginalConfirmatoryCapsuleAncestorLease
    | Mapping[str, Any],
) -> OriginalConfirmatoryExecutionCapsule:
    canonical_leaf_lease = canonical_original_confirmatory_capsule_lease_identity(
        capsule_lease_identity
    )
    canonical_ancestor_lease = canonical_original_confirmatory_capsule_ancestor_lease(
        capsule_ancestor_lease
    )
    canonical_python_lease = canonical_original_confirmatory_interpreter_lease_identity(
        python_lease_identity
    )
    canonical_python_ancestor = (
        canonical_original_confirmatory_interpreter_ancestor_lease(
            python_ancestor_lease
        )
    )
    canonical_runtime_python_lease = (
        canonical_original_confirmatory_runtime_interpreter_lease_identity(
            runtime_python_lease_identity
        )
    )
    canonical_runtime_python_ancestor = (
        canonical_original_confirmatory_runtime_interpreter_ancestor_lease(
            runtime_python_ancestor_lease
        )
    )
    provisional = OriginalConfirmatoryExecutionCapsule(
        path=Path(_absolute_path(str(path), role="Q execution capsule path")),
        size_bytes=_positive_int(size_bytes, role="Q execution capsule size"),
        sha256=_sha256(sha256, role="Q execution capsule"),
        internal_manifest_sha256=_sha256(
            internal_manifest_sha256,
            role="Q execution capsule internal manifest",
        ),
        capsule_policy_sha256=_sha256(
            capsule_policy_sha256,
            role="Q execution capsule policy",
        ),
        entry_contract_sha256=_sha256(
            entry_contract_sha256,
            role="Q execution capsule entry contract",
        ),
        plan_sha256=_sha256(plan_sha256, role="Q execution capsule PLAN"),
        runtime_release_root_sha256=_sha256(
            runtime_release_root_sha256,
            role="Q execution capsule runtime release",
        ),
        terminal_release_root_sha256=_sha256(
            terminal_release_root_sha256,
            role="Q execution capsule terminal release",
        ),
        python_path=Path(
            _absolute_path(str(python_path), role="Q execution capsule Python")
        ),
        python_sha256=_sha256(python_sha256, role="Q execution capsule Python"),
        python_lease_identity=canonical_python_lease,
        python_lease_identity_root_sha256=canonical_json_sha256(
            canonical_python_lease.as_dict()
        ),
        python_ancestor_lease=canonical_python_ancestor,
        python_ancestor_lease_root_sha256=canonical_json_sha256(
            canonical_python_ancestor.as_dict()
        ),
        python_runtime_resolution_policy=PYTHON_RUNTIME_RESOLUTION_POLICY,
        runtime_python_path=Path(
            _absolute_path(
                str(runtime_python_path),
                role="Q execution capsule runtime Python",
            )
        ),
        runtime_python_sha256=_sha256(
            runtime_python_sha256,
            role="Q execution capsule runtime Python",
        ),
        runtime_python_lease_identity=canonical_runtime_python_lease,
        runtime_python_lease_identity_root_sha256=canonical_json_sha256(
            canonical_runtime_python_lease.as_dict()
        ),
        runtime_python_ancestor_lease=canonical_runtime_python_ancestor,
        runtime_python_ancestor_lease_root_sha256=canonical_json_sha256(
            canonical_runtime_python_ancestor.as_dict()
        ),
        capsule_lease_identity=canonical_leaf_lease,
        capsule_lease_identity_root_sha256=canonical_json_sha256(
            canonical_leaf_lease.as_dict()
        ),
        capsule_ancestor_lease=canonical_ancestor_lease,
        capsule_ancestor_lease_root_sha256=canonical_json_sha256(
            canonical_ancestor_lease.as_dict()
        ),
        contract_sha256="0" * 64,
    )
    unsigned = provisional.payload_without_self_hash()
    return canonical_original_confirmatory_execution_capsule(
        {**unsigned, "contract_sha256": canonical_json_sha256(unsigned)}
    )


def original_confirmatory_capsule_mode_tail(
    *,
    capsule_mode: str,
    e_intent_path: str | Path,
    e_intent_sha256: str,
    e_intent_core_sha256: str,
    q_authority_root_sha256: str,
    launch_nonce: str,
    supervisor_job_id: str,
    supervisor_job_directory: str | Path,
    attempt_id: str,
    run_id: str,
    execution_mode: str,
    retry_of_run_id: str | None,
    run_spec_path: str | Path | None = None,
    launch_intent_path: str | Path | None = None,
    process_started_path: str | Path | None = None,
    preterminal_pin_path: str | Path | None = None,
    supervisor_terminal_path: str | Path | None = None,
    verifier_stdout_path: str | Path | None = None,
    composed_terminal_path: str | Path | None = None,
) -> tuple[str, ...]:
    """Build the exact common/lineage/mode suffix; no trailing flags are possible."""

    if capsule_mode not in CAPSULE_ALLOWED_MODES:
        raise OriginalConfirmatoryCapsuleAuthorityError("capsule mode is not allowed")
    if execution_mode not in CAPSULE_EXECUTION_MODES:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "capsule execution mode is not fresh/successor_resume"
        )
    if not isinstance(launch_nonce, str) or _NONCE.fullmatch(launch_nonce) is None:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "capsule launch nonce must be exactly 64 lowercase hexadecimal digits"
        )
    common_values = (
        _absolute_path(str(e_intent_path), role="capsule E intent path"),
        _sha256(e_intent_sha256, role="capsule E intent"),
        _sha256(e_intent_core_sha256, role="capsule E intent core"),
        _sha256(q_authority_root_sha256, role="capsule Q authority root"),
        launch_nonce,
        _identifier(supervisor_job_id, role="capsule supervisor job id"),
        _absolute_path(
            str(supervisor_job_directory),
            role="capsule supervisor job directory",
        ),
        _identifier(attempt_id, role="capsule attempt id"),
        _identifier(run_id, role="capsule run id"),
        execution_mode,
    )
    tail: list[str] = []
    for flag, item in zip(CAPSULE_COMMON_TAIL_FLAGS, common_values, strict=True):
        tail.extend((flag, item))
    if execution_mode == "fresh":
        if retry_of_run_id is not None:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "fresh capsule execution forbids retry lineage"
            )
    else:
        tail.extend(
            (
                CAPSULE_SUCCESSOR_LINEAGE_FLAG,
                _identifier(retry_of_run_id, role="capsule retry-of run id"),
            )
        )

    preterminal_values = (
        run_spec_path,
        launch_intent_path,
        process_started_path,
        preterminal_pin_path,
    )
    terminal_values = (
        supervisor_terminal_path,
        verifier_stdout_path,
        preterminal_pin_path,
        composed_terminal_path,
    )
    if capsule_mode == CAPSULE_SCIENTIFIC_MODE:
        if any(item is not None for item in (*preterminal_values, *terminal_values)):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "run-confirmatory forbids verifier suffix paths"
            )
    elif capsule_mode == CAPSULE_PRETERMINAL_MODE:
        if any(item is None for item in preterminal_values) or any(
            item is not None
            for item in (
                supervisor_terminal_path,
                verifier_stdout_path,
                composed_terminal_path,
            )
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "verify-preterminal requires exactly its four suffix paths"
            )
        for flag, path_item in zip(
            CAPSULE_PRETERMINAL_SUFFIX_FLAGS,
            preterminal_values,
            strict=True,
        ):
            tail.extend(
                (
                    flag,
                    _absolute_path(
                        str(path_item),
                        role=f"capsule {flag.removeprefix('--')} path",
                    ),
                )
            )
    else:
        if any(item is None for item in terminal_values) or any(
            item is not None
            for item in (run_spec_path, launch_intent_path, process_started_path)
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "verify-terminal requires exactly its four suffix paths"
            )
        for flag, path_item in zip(
            CAPSULE_TERMINAL_SUFFIX_FLAGS,
            terminal_values,
            strict=True,
        ):
            tail.extend(
                (
                    flag,
                    _absolute_path(
                        str(path_item),
                        role=f"capsule {flag.removeprefix('--')} path",
                    ),
                )
            )
    return tuple(tail)


def canonical_original_confirmatory_capsule_mode_tail(
    *,
    capsule_mode: str,
    tail_argv: Sequence[str],
) -> tuple[str, ...]:
    """Parse and reconstruct one exact mode tail, rejecting reordering/extras."""

    raw = list(tail_argv)
    common_length = len(CAPSULE_COMMON_TAIL_FLAGS) * 2
    if (
        capsule_mode not in CAPSULE_ALLOWED_MODES
        or len(raw) < common_length
        or not all(
            isinstance(item, str) and item and "\x00" not in item for item in raw
        )
        or not _strict_json_value_equal(
            raw[:common_length:2],
            list(CAPSULE_COMMON_TAIL_FLAGS),
        )
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "capsule tail lacks its exact common prefix"
        )
    common = raw[1:common_length:2]
    execution_mode = common[-1]
    cursor = common_length
    retry_of_run_id: str | None = None
    if execution_mode == "successor_resume":
        if len(raw) < cursor + 2 or raw[cursor] != CAPSULE_SUCCESSOR_LINEAGE_FLAG:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "successor capsule tail lacks exact lineage"
            )
        retry_of_run_id = raw[cursor + 1]
        cursor += 2
    elif execution_mode != "fresh":
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "capsule tail execution mode is invalid"
        )

    if capsule_mode == CAPSULE_SCIENTIFIC_MODE:
        suffix_flags: tuple[str, ...] = ()
    elif capsule_mode == CAPSULE_PRETERMINAL_MODE:
        suffix_flags = CAPSULE_PRETERMINAL_SUFFIX_FLAGS
    else:
        suffix_flags = CAPSULE_TERMINAL_SUFFIX_FLAGS
    suffix_length = len(suffix_flags) * 2
    if len(raw) != cursor + suffix_length or not _strict_json_value_equal(
        raw[cursor : cursor + suffix_length : 2],
        list(suffix_flags),
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "capsule tail differs from its exact mode suffix"
        )
    suffix = raw[cursor + 1 : cursor + suffix_length : 2]
    suffix_values = dict(zip(suffix_flags, suffix, strict=True))
    return original_confirmatory_capsule_mode_tail(
        capsule_mode=capsule_mode,
        e_intent_path=common[0],
        e_intent_sha256=common[1],
        e_intent_core_sha256=common[2],
        q_authority_root_sha256=common[3],
        launch_nonce=common[4],
        supervisor_job_id=common[5],
        supervisor_job_directory=common[6],
        attempt_id=common[7],
        run_id=common[8],
        execution_mode=execution_mode,
        retry_of_run_id=retry_of_run_id,
        run_spec_path=suffix_values.get("--run-spec"),
        launch_intent_path=suffix_values.get("--launch-intent"),
        process_started_path=suffix_values.get("--process-started"),
        preterminal_pin_path=suffix_values.get("--preterminal-pin"),
        supervisor_terminal_path=suffix_values.get("--supervisor-terminal"),
        verifier_stdout_path=suffix_values.get("--verifier-stdout"),
        composed_terminal_path=suffix_values.get("--composed-terminal"),
    )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryCapsuleCommand:
    """One concrete E capsule command."""

    program_path: Path
    program_sha256: str
    argv: tuple[str, ...]
    cwd: Path
    command_sha256: str

    def payload_without_self_hash(self) -> dict[str, Any]:
        return {
            "program_path": str(self.program_path),
            "program_sha256": self.program_sha256,
            "argv": list(self.argv),
            "cwd": str(self.cwd),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.payload_without_self_hash(),
            "command_sha256": self.command_sha256,
        }


_CAPSULE_COMMAND_FIELDS = {
    "program_path",
    "program_sha256",
    "argv",
    "cwd",
    "command_sha256",
}


def canonical_original_confirmatory_capsule_command(
    value: OriginalConfirmatoryCapsuleCommand | Mapping[str, Any],
    *,
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    expected_mode: str,
    expected_tail_argv: Sequence[str],
) -> OriginalConfirmatoryCapsuleCommand:
    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatoryCapsuleCommand
        else _mapping(value, role="E capsule command")
    )
    if set(raw) != _CAPSULE_COMMAND_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E capsule command has an unexpected field set"
        )
    canonical_capsule = canonical_original_confirmatory_execution_capsule(capsule)
    argv = raw["argv"]
    if (
        expected_mode not in CAPSULE_ALLOWED_MODES
        or not isinstance(argv, list)
        or not all(
            isinstance(item, str) and item and "\x00" not in item for item in argv
        )
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E capsule command mode/argv is invalid"
        )
    canonical_tail = canonical_original_confirmatory_capsule_mode_tail(
        capsule_mode=expected_mode,
        tail_argv=expected_tail_argv,
    )
    expected_prefix = [
        str(canonical_capsule.python_path),
        *CAPSULE_PYTHON_ISOLATED_FLAGS,
        str(canonical_capsule.path),
        expected_mode,
    ]
    expected_argv = [*expected_prefix, *canonical_tail]
    unsigned = {key: item for key, item in raw.items() if key != "command_sha256"}
    if (
        not _strict_json_value_equal(argv, expected_argv)
        or raw["program_path"] != str(canonical_capsule.python_path)
        or raw["program_sha256"] != canonical_capsule.python_sha256
        or raw["command_sha256"] != canonical_json_sha256(unsigned)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E capsule command differs from Q/interpreter/mode"
        )
    return OriginalConfirmatoryCapsuleCommand(
        program_path=canonical_capsule.python_path,
        program_sha256=canonical_capsule.python_sha256,
        argv=tuple(argv),
        cwd=Path(_absolute_path(raw["cwd"], role="E capsule command cwd")),
        command_sha256=raw["command_sha256"],
    )


def build_original_confirmatory_capsule_command(
    *,
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    mode: str,
    tail_argv: Sequence[str],
    cwd: str | Path,
) -> OriginalConfirmatoryCapsuleCommand:
    canonical_capsule = canonical_original_confirmatory_execution_capsule(capsule)
    canonical_tail = canonical_original_confirmatory_capsule_mode_tail(
        capsule_mode=mode,
        tail_argv=tail_argv,
    )
    argv = [
        str(canonical_capsule.python_path),
        *CAPSULE_PYTHON_ISOLATED_FLAGS,
        str(canonical_capsule.path),
        mode,
        *canonical_tail,
    ]
    unsigned = {
        "program_path": str(canonical_capsule.python_path),
        "program_sha256": canonical_capsule.python_sha256,
        "argv": argv,
        "cwd": _absolute_path(str(cwd), role="E capsule command cwd"),
    }
    return canonical_original_confirmatory_capsule_command(
        {**unsigned, "command_sha256": canonical_json_sha256(unsigned)},
        capsule=canonical_capsule,
        expected_mode=mode,
        expected_tail_argv=canonical_tail,
    )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryPreterminalPinContract:
    """E's exact verify-preterminal command/output contract."""

    capsule_contract_sha256: str
    capsule_path: Path
    capsule_sha256: str
    capsule_internal_manifest_sha256: str
    capsule_mode: str
    verifier_command: dict[str, Any]
    verifier_command_sha256: str
    preterminal_pin_receipt_path: Path
    preterminal_pin_receipt_max_bytes: int
    semantic_outcome_read_scope: str
    contract_sha256: str

    def payload_without_self_hash(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": PRETERMINAL_PIN_CONTRACT_POLICY,
            "capsule_contract_sha256": self.capsule_contract_sha256,
            "capsule_path": str(self.capsule_path),
            "capsule_sha256": self.capsule_sha256,
            "capsule_internal_manifest_sha256": (self.capsule_internal_manifest_sha256),
            "capsule_mode": self.capsule_mode,
            "verifier_command": self.verifier_command,
            "verifier_command_sha256": self.verifier_command_sha256,
            "preterminal_pin_receipt_path": str(self.preterminal_pin_receipt_path),
            "preterminal_pin_receipt_max_bytes": self.preterminal_pin_receipt_max_bytes,
            "preterminal_pin_receipt_must_be_absent_before": True,
            "publication_policy": PRETERMINAL_PIN_PUBLICATION_POLICY,
            "stdout_summary_policy": PRETERMINAL_PIN_STDOUT_SUMMARY_POLICY,
            "stdout_single_canonical_json_line_required": True,
            "stderr_empty_required": True,
            "semantic_outcome_read_scope": self.semantic_outcome_read_scope,
            "outcome_values_read": False,
            "outcome_values_emitted": False,
            "outcome_values_used_for_selection_or_tuning": False,
            "training_or_model_selection_allowed": False,
            "scientific_publication_allowed": False,
            "automatic_retry_allowed": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.payload_without_self_hash(),
            "contract_sha256": self.contract_sha256,
        }


_PRETERMINAL_PIN_CONTRACT_FIELDS = {
    "schema_version",
    "policy",
    "capsule_contract_sha256",
    "capsule_path",
    "capsule_sha256",
    "capsule_internal_manifest_sha256",
    "capsule_mode",
    "verifier_command",
    "verifier_command_sha256",
    "preterminal_pin_receipt_path",
    "preterminal_pin_receipt_max_bytes",
    "preterminal_pin_receipt_must_be_absent_before",
    "publication_policy",
    "stdout_summary_policy",
    "stdout_single_canonical_json_line_required",
    "stderr_empty_required",
    "semantic_outcome_read_scope",
    "outcome_values_read",
    "outcome_values_emitted",
    "outcome_values_used_for_selection_or_tuning",
    "training_or_model_selection_allowed",
    "scientific_publication_allowed",
    "automatic_retry_allowed",
    "contract_sha256",
}


def canonical_original_confirmatory_preterminal_pin_contract(
    value: OriginalConfirmatoryPreterminalPinContract | Mapping[str, Any],
    *,
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    verifier_command: OriginalConfirmatoryCapsuleCommand | Mapping[str, Any],
    verifier_command_tail_argv: Sequence[str],
) -> OriginalConfirmatoryPreterminalPinContract:
    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatoryPreterminalPinContract
        else _mapping(value, role="E preterminal pin contract")
    )
    if set(raw) != _PRETERMINAL_PIN_CONTRACT_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E preterminal pin contract has an unexpected field set"
        )
    canonical_capsule = canonical_original_confirmatory_execution_capsule(capsule)
    canonical_command = canonical_original_confirmatory_capsule_command(
        verifier_command,
        capsule=canonical_capsule,
        expected_mode=CAPSULE_PRETERMINAL_MODE,
        expected_tail_argv=verifier_command_tail_argv,
    )
    canonical_verifier_tail = canonical_original_confirmatory_capsule_mode_tail(
        capsule_mode=CAPSULE_PRETERMINAL_MODE,
        tail_argv=verifier_command_tail_argv,
    )
    pin_flag_index = canonical_verifier_tail.index("--preterminal-pin")
    expected_pin_path = canonical_verifier_tail[pin_flag_index + 1]
    scope = raw["semantic_outcome_read_scope"]
    unsigned = {key: item for key, item in raw.items() if key != "contract_sha256"}
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != PRETERMINAL_PIN_CONTRACT_POLICY
        or raw["capsule_contract_sha256"] != canonical_capsule.contract_sha256
        or raw["capsule_path"] != str(canonical_capsule.path)
        or raw["capsule_sha256"] != canonical_capsule.sha256
        or raw["capsule_internal_manifest_sha256"]
        != canonical_capsule.internal_manifest_sha256
        or raw["capsule_mode"] != CAPSULE_PRETERMINAL_MODE
        or not _strict_json_value_equal(
            raw["verifier_command"],
            canonical_command.as_dict(),
        )
        or raw["verifier_command_sha256"] != canonical_command.command_sha256
        or raw["preterminal_pin_receipt_path"] != expected_pin_path
        or raw["preterminal_pin_receipt_must_be_absent_before"] is not True
        or raw["publication_policy"] != PRETERMINAL_PIN_PUBLICATION_POLICY
        or raw["stdout_summary_policy"] != PRETERMINAL_PIN_STDOUT_SUMMARY_POLICY
        or raw["stdout_single_canonical_json_line_required"] is not True
        or raw["stderr_empty_required"] is not True
        or scope != SEMANTIC_OUTCOME_READ_SCOPE
        or raw["outcome_values_read"] is not False
        or raw["outcome_values_emitted"] is not False
        or raw["outcome_values_used_for_selection_or_tuning"] is not False
        or raw["training_or_model_selection_allowed"] is not False
        or raw["scientific_publication_allowed"] is not False
        or raw["automatic_retry_allowed"] is not False
        or raw["contract_sha256"] != canonical_json_sha256(unsigned)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E preterminal pin contract violates its exact closed policy"
        )
    return OriginalConfirmatoryPreterminalPinContract(
        capsule_contract_sha256=canonical_capsule.contract_sha256,
        capsule_path=canonical_capsule.path,
        capsule_sha256=canonical_capsule.sha256,
        capsule_internal_manifest_sha256=canonical_capsule.internal_manifest_sha256,
        capsule_mode=CAPSULE_PRETERMINAL_MODE,
        verifier_command=canonical_command.as_dict(),
        verifier_command_sha256=canonical_command.command_sha256,
        preterminal_pin_receipt_path=Path(
            _absolute_path(
                raw["preterminal_pin_receipt_path"],
                role="E preterminal pin receipt path",
            )
        ),
        preterminal_pin_receipt_max_bytes=_positive_int(
            raw["preterminal_pin_receipt_max_bytes"],
            role="E preterminal pin receipt bound",
        ),
        semantic_outcome_read_scope=scope,
        contract_sha256=raw["contract_sha256"],
    )


def build_original_confirmatory_preterminal_pin_contract(
    *,
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    verifier_command: OriginalConfirmatoryCapsuleCommand | Mapping[str, Any],
    verifier_command_tail_argv: Sequence[str],
    preterminal_pin_receipt_path: str | Path,
    preterminal_pin_receipt_max_bytes: int,
) -> OriginalConfirmatoryPreterminalPinContract:
    canonical_capsule = canonical_original_confirmatory_execution_capsule(capsule)
    canonical_command = canonical_original_confirmatory_capsule_command(
        verifier_command,
        capsule=canonical_capsule,
        expected_mode=CAPSULE_PRETERMINAL_MODE,
        expected_tail_argv=verifier_command_tail_argv,
    )
    provisional = OriginalConfirmatoryPreterminalPinContract(
        capsule_contract_sha256=canonical_capsule.contract_sha256,
        capsule_path=canonical_capsule.path,
        capsule_sha256=canonical_capsule.sha256,
        capsule_internal_manifest_sha256=canonical_capsule.internal_manifest_sha256,
        capsule_mode=CAPSULE_PRETERMINAL_MODE,
        verifier_command=canonical_command.as_dict(),
        verifier_command_sha256=canonical_command.command_sha256,
        preterminal_pin_receipt_path=Path(
            _absolute_path(
                str(preterminal_pin_receipt_path),
                role="E preterminal pin receipt path",
            )
        ),
        preterminal_pin_receipt_max_bytes=_positive_int(
            preterminal_pin_receipt_max_bytes,
            role="E preterminal pin receipt bound",
        ),
        semantic_outcome_read_scope=SEMANTIC_OUTCOME_READ_SCOPE,
        contract_sha256="0" * 64,
    )
    unsigned = provisional.payload_without_self_hash()
    return canonical_original_confirmatory_preterminal_pin_contract(
        {**unsigned, "contract_sha256": canonical_json_sha256(unsigned)},
        capsule=canonical_capsule,
        verifier_command=canonical_command,
        verifier_command_tail_argv=verifier_command_tail_argv,
    )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryPreterminalOverlapHandshakeContract:
    """Closed READY/ACK overlap contract for pin handle continuity."""

    handshake_receipt_path: Path
    ready_line_max_bytes: int
    ack_line_max_bytes: int
    handshake_timeout_ms: int
    contract_sha256: str

    def payload_without_self_hash(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": PRETERMINAL_OVERLAP_HANDSHAKE_CONTRACT_POLICY,
            "handshake_receipt_path": str(self.handshake_receipt_path),
            "ready_summary_policy": PRETERMINAL_PIN_STDOUT_SUMMARY_POLICY,
            "ready_message_type": PRETERMINAL_READY_MESSAGE_TYPE,
            "ack_policy": PRETERMINAL_OVERLAP_ACK_POLICY,
            "ack_message_type": PRETERMINAL_ACK_MESSAGE_TYPE,
            "stdout_ready_transport": PRETERMINAL_HANDSHAKE_PIPE_TRANSPORT,
            "stdin_ack_transport": PRETERMINAL_HANDSHAKE_PIPE_TRANSPORT,
            "stderr_transport": PRETERMINAL_HANDSHAKE_STDERR_TRANSPORT,
            "ready_line_max_bytes": self.ready_line_max_bytes,
            "ack_line_max_bytes": self.ack_line_max_bytes,
            "handshake_timeout_ms": self.handshake_timeout_ms,
            "child_pin_share_access": ["FILE_SHARE_READ"],
            "supervisor_pin_share_access": [
                "FILE_SHARE_READ",
                "FILE_SHARE_WRITE",
            ],
            "pin_handle_overlap_required": True,
            "restricted_inherited_handle_list_required": True,
            "stdout_exactly_one_canonical_line_required": True,
            "stdout_eof_after_ack_required": True,
            "stderr_empty_required": True,
            "exit_zero_required": True,
            "automatic_retry_allowed": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.payload_without_self_hash(),
            "contract_sha256": self.contract_sha256,
        }


_PRETERMINAL_OVERLAP_HANDSHAKE_CONTRACT_FIELDS = {
    "schema_version",
    "policy",
    "handshake_receipt_path",
    "ready_summary_policy",
    "ready_message_type",
    "ack_policy",
    "ack_message_type",
    "stdout_ready_transport",
    "stdin_ack_transport",
    "stderr_transport",
    "ready_line_max_bytes",
    "ack_line_max_bytes",
    "handshake_timeout_ms",
    "child_pin_share_access",
    "supervisor_pin_share_access",
    "pin_handle_overlap_required",
    "restricted_inherited_handle_list_required",
    "stdout_exactly_one_canonical_line_required",
    "stdout_eof_after_ack_required",
    "stderr_empty_required",
    "exit_zero_required",
    "automatic_retry_allowed",
    "contract_sha256",
}


def canonical_original_confirmatory_preterminal_overlap_handshake_contract(
    value: OriginalConfirmatoryPreterminalOverlapHandshakeContract | Mapping[str, Any],
) -> OriginalConfirmatoryPreterminalOverlapHandshakeContract:
    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatoryPreterminalOverlapHandshakeContract
        else _mapping(value, role="preterminal overlap handshake contract")
    )
    if set(raw) != _PRETERMINAL_OVERLAP_HANDSHAKE_CONTRACT_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "preterminal overlap handshake contract has an unexpected field set"
        )
    unsigned = {key: item for key, item in raw.items() if key != "contract_sha256"}
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != PRETERMINAL_OVERLAP_HANDSHAKE_CONTRACT_POLICY
        or raw["ready_summary_policy"] != PRETERMINAL_PIN_STDOUT_SUMMARY_POLICY
        or raw["ready_message_type"] != PRETERMINAL_READY_MESSAGE_TYPE
        or raw["ack_policy"] != PRETERMINAL_OVERLAP_ACK_POLICY
        or raw["ack_message_type"] != PRETERMINAL_ACK_MESSAGE_TYPE
        or raw["stdout_ready_transport"] != PRETERMINAL_HANDSHAKE_PIPE_TRANSPORT
        or raw["stdin_ack_transport"] != PRETERMINAL_HANDSHAKE_PIPE_TRANSPORT
        or raw["stderr_transport"] != PRETERMINAL_HANDSHAKE_STDERR_TRANSPORT
        or not _strict_json_value_equal(
            raw["child_pin_share_access"],
            ["FILE_SHARE_READ"],
        )
        or not _strict_json_value_equal(
            raw["supervisor_pin_share_access"],
            ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        )
        or raw["pin_handle_overlap_required"] is not True
        or raw["restricted_inherited_handle_list_required"] is not True
        or raw["stdout_exactly_one_canonical_line_required"] is not True
        or raw["stdout_eof_after_ack_required"] is not True
        or raw["stderr_empty_required"] is not True
        or raw["exit_zero_required"] is not True
        or raw["automatic_retry_allowed"] is not False
        or raw["contract_sha256"] != canonical_json_sha256(unsigned)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "preterminal overlap handshake contract violates its exact policy"
        )
    return OriginalConfirmatoryPreterminalOverlapHandshakeContract(
        handshake_receipt_path=Path(
            _absolute_path(
                raw["handshake_receipt_path"],
                role="preterminal overlap handshake receipt path",
            )
        ),
        ready_line_max_bytes=_positive_int(
            raw["ready_line_max_bytes"],
            role="preterminal READY line bound",
        ),
        ack_line_max_bytes=_positive_int(
            raw["ack_line_max_bytes"],
            role="preterminal ACK line bound",
        ),
        handshake_timeout_ms=_positive_int(
            raw["handshake_timeout_ms"],
            role="preterminal overlap handshake timeout",
        ),
        contract_sha256=raw["contract_sha256"],
    )


def build_original_confirmatory_preterminal_overlap_handshake_contract(
    *,
    handshake_receipt_path: str | Path,
    ready_line_max_bytes: int,
    ack_line_max_bytes: int,
    handshake_timeout_ms: int,
) -> OriginalConfirmatoryPreterminalOverlapHandshakeContract:
    provisional = OriginalConfirmatoryPreterminalOverlapHandshakeContract(
        handshake_receipt_path=Path(str(handshake_receipt_path)),
        ready_line_max_bytes=ready_line_max_bytes,
        ack_line_max_bytes=ack_line_max_bytes,
        handshake_timeout_ms=handshake_timeout_ms,
        contract_sha256="0" * 64,
    )
    unsigned = provisional.payload_without_self_hash()
    return canonical_original_confirmatory_preterminal_overlap_handshake_contract(
        {**unsigned, "contract_sha256": canonical_json_sha256(unsigned)}
    )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatorySupervisorJobAncestorLeaseContract:
    """Exact three-directory no-delete chain for one supervisor job."""

    anchor_path: Path
    paths: tuple[Path, ...]
    contract_sha256: str

    def payload_without_self_hash(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": SUPERVISOR_JOB_ANCESTOR_LEASE_CONTRACT_POLICY,
            "anchor_path": str(self.anchor_path),
            "paths": [str(path) for path in self.paths],
            "record_count": 3,
            "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
            "delete_sharing_allowed": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.payload_without_self_hash(),
            "contract_sha256": self.contract_sha256,
        }


_SUPERVISOR_JOB_ANCESTOR_LEASE_CONTRACT_FIELDS = {
    "schema_version",
    "policy",
    "anchor_path",
    "paths",
    "record_count",
    "share_access",
    "delete_sharing_allowed",
    "contract_sha256",
}


def canonical_original_confirmatory_supervisor_job_ancestor_lease_contract(
    value: OriginalConfirmatorySupervisorJobAncestorLeaseContract | Mapping[str, Any],
) -> OriginalConfirmatorySupervisorJobAncestorLeaseContract:
    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatorySupervisorJobAncestorLeaseContract
        else _mapping(value, role="supervisor job ancestor lease contract")
    )
    if set(raw) != _SUPERVISOR_JOB_ANCESTOR_LEASE_CONTRACT_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "supervisor job ancestor lease contract has an unexpected field set"
        )
    anchor = Path(
        _absolute_path(
            raw["anchor_path"],
            role="supervisor job ancestor anchor",
        )
    )
    raw_paths = raw["paths"]
    if not isinstance(raw_paths, list) or len(raw_paths) != 3:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "supervisor job ancestor lease contract must have exactly three paths"
        )
    paths = tuple(
        Path(
            _absolute_path(
                item,
                role=f"supervisor job ancestor path {index}",
            )
        )
        for index, item in enumerate(raw_paths, start=1)
    )
    unsigned = {key: item for key, item in raw.items() if key != "contract_sha256"}
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != SUPERVISOR_JOB_ANCESTOR_LEASE_CONTRACT_POLICY
        or paths[0] != anchor
        or paths[1] != anchor / "jobs"
        or paths[2].parent != paths[1]
        or raw["record_count"] != 3
        or type(raw["record_count"]) is not int
        or not _strict_json_value_equal(
            raw["share_access"],
            ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        )
        or raw["delete_sharing_allowed"] is not False
        or raw["contract_sha256"] != canonical_json_sha256(unsigned)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "supervisor job ancestor lease contract violates its exact policy"
        )
    return OriginalConfirmatorySupervisorJobAncestorLeaseContract(
        anchor_path=anchor,
        paths=paths,
        contract_sha256=raw["contract_sha256"],
    )


def build_original_confirmatory_supervisor_job_ancestor_lease_contract(
    *,
    supervisor_root: str | Path,
    job_id: str,
) -> OriginalConfirmatorySupervisorJobAncestorLeaseContract:
    root = Path(
        _absolute_path(
            str(supervisor_root),
            role="supervisor job ancestor anchor",
        )
    )
    canonical_job_id = _identifier(job_id, role="supervisor job ancestor job id")
    provisional = OriginalConfirmatorySupervisorJobAncestorLeaseContract(
        anchor_path=root,
        paths=(root, root / "jobs", root / "jobs" / canonical_job_id),
        contract_sha256="0" * 64,
    )
    unsigned = provisional.payload_without_self_hash()
    return canonical_original_confirmatory_supervisor_job_ancestor_lease_contract(
        {**unsigned, "contract_sha256": canonical_json_sha256(unsigned)}
    )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryPostwakeInputLeaseContract:
    """Q/E-bound paths and native-handle continuity policy for P/T/L/C."""

    supervisor_job_dir: Path
    preterminal_pin_path: Path
    verifier_stdout_path: Path
    verifier_stderr_path: Path
    verifier_log_max_bytes: int
    terminal_receipt_path: Path
    lease_receipt_path: Path
    supervisor_job_ancestor_lease_contract: (
        OriginalConfirmatorySupervisorJobAncestorLeaseContract
    )
    contract_sha256: str

    def payload_without_self_hash(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": POSTWAKE_INPUT_LEASE_CONTRACT_POLICY,
            "supervisor_job_dir": str(self.supervisor_job_dir),
            "preterminal_pin_path": str(self.preterminal_pin_path),
            "verifier_stdout_path": str(self.verifier_stdout_path),
            "verifier_stderr_path": str(self.verifier_stderr_path),
            "verifier_stderr_derivation_policy": (POSTWAKE_STDERR_DERIVATION_POLICY),
            "verifier_stderr_filename": "verifier.stderr.log",
            "verifier_log_max_bytes": self.verifier_log_max_bytes,
            "terminal_receipt_path": str(self.terminal_receipt_path),
            "lease_receipt_path": str(self.lease_receipt_path),
            "supervisor_job_ancestor_lease_contract": (
                self.supervisor_job_ancestor_lease_contract.as_dict()
            ),
            "identity_policy": POSTWAKE_INPUT_IDENTITY_POLICY,
            "stdout_stderr_creation_policy": POSTWAKE_LOG_CREATION_POLICY,
            "terminal_creation_policy": POSTWAKE_TERMINAL_CREATION_POLICY,
            "retained_through_policy": POSTWAKE_RETAINED_THROUGH_POLICY,
            "supervisor_loss_policy": POSTWAKE_SUPERVISOR_LOSS_POLICY,
            "process_duplicate_handle_required": True,
            "process_duplicate_handle_policy": POSTWAKE_PROCESS_DUP_HANDLE_POLICY,
            "source_owner_identity_required": True,
            "source_handle_slots_bound": True,
            "source_owner_open_access_mask": (POSTWAKE_SOURCE_OWNER_OPEN_ACCESS_MASK),
            "source_owner_open_policy": POSTWAKE_SOURCE_OWNER_OPEN_POLICY,
            "source_owner_nonsignaled_before_after_required": True,
            "duplicated_handles_retained_through_composed_publication_required": (True),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.payload_without_self_hash(),
            "contract_sha256": self.contract_sha256,
        }


_POSTWAKE_INPUT_LEASE_CONTRACT_FIELDS = {
    "schema_version",
    "policy",
    "supervisor_job_dir",
    "preterminal_pin_path",
    "verifier_stdout_path",
    "verifier_stderr_path",
    "verifier_stderr_derivation_policy",
    "verifier_stderr_filename",
    "verifier_log_max_bytes",
    "terminal_receipt_path",
    "lease_receipt_path",
    "supervisor_job_ancestor_lease_contract",
    "identity_policy",
    "stdout_stderr_creation_policy",
    "terminal_creation_policy",
    "retained_through_policy",
    "supervisor_loss_policy",
    "process_duplicate_handle_required",
    "process_duplicate_handle_policy",
    "source_owner_identity_required",
    "source_handle_slots_bound",
    "source_owner_open_access_mask",
    "source_owner_open_policy",
    "source_owner_nonsignaled_before_after_required",
    "duplicated_handles_retained_through_composed_publication_required",
    "contract_sha256",
}


def canonical_original_confirmatory_postwake_input_lease_contract(
    value: OriginalConfirmatoryPostwakeInputLeaseContract | Mapping[str, Any],
) -> OriginalConfirmatoryPostwakeInputLeaseContract:
    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatoryPostwakeInputLeaseContract
        else _mapping(value, role="postwake input lease contract")
    )
    if set(raw) != _POSTWAKE_INPUT_LEASE_CONTRACT_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "postwake input lease contract has an unexpected field set"
        )
    job_dir = Path(
        _absolute_path(raw["supervisor_job_dir"], role="supervisor job directory")
    )
    pin_path = Path(
        _absolute_path(raw["preterminal_pin_path"], role="preterminal pin path")
    )
    stdout_path = Path(
        _absolute_path(raw["verifier_stdout_path"], role="verifier stdout path")
    )
    stderr_path = Path(
        _absolute_path(raw["verifier_stderr_path"], role="verifier stderr path")
    )
    terminal_path = Path(
        _absolute_path(raw["terminal_receipt_path"], role="terminal receipt path")
    )
    lease_path = Path(
        _absolute_path(raw["lease_receipt_path"], role="lease receipt path")
    )
    ancestor = canonical_original_confirmatory_supervisor_job_ancestor_lease_contract(
        raw["supervisor_job_ancestor_lease_contract"]
    )
    input_paths = (pin_path, stdout_path, stderr_path, terminal_path, lease_path)
    unsigned = {key: item for key, item in raw.items() if key != "contract_sha256"}
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != POSTWAKE_INPUT_LEASE_CONTRACT_POLICY
        or ancestor.paths[-1] != job_dir
        or any(path.parent != job_dir for path in input_paths)
        or len(set(input_paths)) != len(input_paths)
        or stderr_path != stdout_path.parent / "verifier.stderr.log"
        or raw["verifier_stderr_derivation_policy"] != POSTWAKE_STDERR_DERIVATION_POLICY
        or raw["verifier_stderr_filename"] != "verifier.stderr.log"
        or lease_path.name != "postwake_input_lease_receipt.json"
        or raw["identity_policy"] != POSTWAKE_INPUT_IDENTITY_POLICY
        or raw["stdout_stderr_creation_policy"] != POSTWAKE_LOG_CREATION_POLICY
        or raw["terminal_creation_policy"] != POSTWAKE_TERMINAL_CREATION_POLICY
        or raw["retained_through_policy"] != POSTWAKE_RETAINED_THROUGH_POLICY
        or raw["supervisor_loss_policy"] != POSTWAKE_SUPERVISOR_LOSS_POLICY
        or raw["process_duplicate_handle_required"] is not True
        or raw["process_duplicate_handle_policy"] != POSTWAKE_PROCESS_DUP_HANDLE_POLICY
        or raw["source_owner_identity_required"] is not True
        or raw["source_handle_slots_bound"] is not True
        or raw["source_owner_open_access_mask"]
        != POSTWAKE_SOURCE_OWNER_OPEN_ACCESS_MASK
        or type(raw["source_owner_open_access_mask"]) is not int
        or raw["source_owner_open_policy"] != POSTWAKE_SOURCE_OWNER_OPEN_POLICY
        or raw["source_owner_nonsignaled_before_after_required"] is not True
        or raw["duplicated_handles_retained_through_composed_publication_required"]
        is not True
        or raw["contract_sha256"] != canonical_json_sha256(unsigned)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "postwake input lease contract violates its exact closed policy"
        )
    return OriginalConfirmatoryPostwakeInputLeaseContract(
        supervisor_job_dir=job_dir,
        preterminal_pin_path=pin_path,
        verifier_stdout_path=stdout_path,
        verifier_stderr_path=stderr_path,
        verifier_log_max_bytes=_positive_int(
            raw["verifier_log_max_bytes"],
            role="verifier log bound",
        ),
        terminal_receipt_path=terminal_path,
        lease_receipt_path=lease_path,
        supervisor_job_ancestor_lease_contract=ancestor,
        contract_sha256=raw["contract_sha256"],
    )


def build_original_confirmatory_postwake_input_lease_contract(
    *,
    supervisor_job_dir: str | Path,
    preterminal_pin_path: str | Path,
    verifier_stdout_path: str | Path,
    verifier_log_max_bytes: int,
    terminal_receipt_path: str | Path,
    lease_receipt_path: str | Path,
    supervisor_job_ancestor_lease_contract: (
        OriginalConfirmatorySupervisorJobAncestorLeaseContract | Mapping[str, Any]
    ),
) -> OriginalConfirmatoryPostwakeInputLeaseContract:
    job_dir = Path(str(supervisor_job_dir))
    stderr_path = Path(str(verifier_stdout_path)).parent / "verifier.stderr.log"
    ancestor = canonical_original_confirmatory_supervisor_job_ancestor_lease_contract(
        supervisor_job_ancestor_lease_contract
    )
    provisional = OriginalConfirmatoryPostwakeInputLeaseContract(
        supervisor_job_dir=job_dir,
        preterminal_pin_path=Path(str(preterminal_pin_path)),
        verifier_stdout_path=Path(str(verifier_stdout_path)),
        verifier_stderr_path=stderr_path,
        verifier_log_max_bytes=verifier_log_max_bytes,
        terminal_receipt_path=Path(str(terminal_receipt_path)),
        lease_receipt_path=Path(str(lease_receipt_path)),
        supervisor_job_ancestor_lease_contract=ancestor,
        contract_sha256="0" * 64,
    )
    unsigned = provisional.payload_without_self_hash()
    return canonical_original_confirmatory_postwake_input_lease_contract(
        {**unsigned, "contract_sha256": canonical_json_sha256(unsigned)}
    )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryPostwakeCustodySeed:
    """Acyclic named-pipe seed containing only already-final upstream facts."""

    payload: Mapping[str, Any]
    seed_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            **dict(self.payload),
            "seed_sha256": self.seed_sha256,
        }


_POSTWAKE_CUSTODY_SEED_FIELDS = {
    "schema_version",
    "policy",
    "q_authority_root_sha256",
    "e_intent_path",
    "e_intent_file_sha256",
    "e_intent_core_sha256",
    "supervisor_job_id",
    "supervisor_job_dir",
    "supervisor_spec_path",
    "launch_nonce",
    "attempt_id",
    "run_id",
    "execution_mode",
    "retry_of_run_id",
    "execution_capsule_contract_sha256",
    "capsule_sha256",
    "supervisor_release_root_sha256",
    "terminal_release_root_sha256",
    "supervisor_terminal_receipt_path",
    "preterminal_pin_receipt_path",
    "postwake_input_lease_receipt_path",
    "composed_terminal_receipt_path",
    "postwake_composed_readback_receipt_path",
    "seed_sha256",
}


def canonical_original_confirmatory_postwake_custody_seed(
    value: OriginalConfirmatoryPostwakeCustodySeed | Mapping[str, Any],
) -> OriginalConfirmatoryPostwakeCustodySeed:
    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatoryPostwakeCustodySeed
        else _mapping(value, role="postwake custody seed")
    )
    if set(raw) != _POSTWAKE_CUSTODY_SEED_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "postwake custody seed has an unexpected field set"
        )
    job_id = _identifier(
        raw["supervisor_job_id"],
        role="postwake custody seed job",
    )
    job_dir = Path(
        _absolute_path(
            raw["supervisor_job_dir"],
            role="postwake custody seed job directory",
        )
    )
    e_path = Path(
        _absolute_path(
            raw["e_intent_path"],
            role="postwake custody seed E path",
        )
    )
    spec_path = Path(
        _absolute_path(
            raw["supervisor_spec_path"],
            role="postwake custody seed supervisor spec path",
        )
    )
    terminal_path = Path(
        _absolute_path(
            raw["supervisor_terminal_receipt_path"],
            role="postwake custody seed terminal path",
        )
    )
    pin_path = Path(
        _absolute_path(
            raw["preterminal_pin_receipt_path"],
            role="postwake custody seed preterminal pin path",
        )
    )
    input_lease_path = Path(
        _absolute_path(
            raw["postwake_input_lease_receipt_path"],
            role="postwake custody seed input lease path",
        )
    )
    composed_path = Path(
        _absolute_path(
            raw["composed_terminal_receipt_path"],
            role="postwake custody seed composed path",
        )
    )
    readback_path = Path(
        _absolute_path(
            raw["postwake_composed_readback_receipt_path"],
            role="postwake custody seed readback path",
        )
    )
    execution_mode = raw["execution_mode"]
    retry_of = raw["retry_of_run_id"]
    unsigned = {key: item for key, item in raw.items() if key != "seed_sha256"}
    hash_fields = (
        "q_authority_root_sha256",
        "e_intent_file_sha256",
        "e_intent_core_sha256",
        "execution_capsule_contract_sha256",
        "capsule_sha256",
        "supervisor_release_root_sha256",
        "terminal_release_root_sha256",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != POSTWAKE_CUSTODY_SEED_POLICY
        or any(
            _sha256(raw[field], role=f"postwake custody seed {field}") != raw[field]
            for field in hash_fields
        )
        or job_dir.name != job_id
        or e_path
        != job_dir.parent.parent
        / CONTROL_STAGING_DIRECTORY_NAME
        / job_id
        / E_INTENT_FILENAME
        or spec_path != job_dir / "run_spec.json"
        or terminal_path != job_dir / "terminal_receipt.json"
        or pin_path != job_dir / "preterminal_pin.json"
        or input_lease_path != job_dir / "postwake_input_lease_receipt.json"
        or composed_path != job_dir / "composed_terminal.json"
        or readback_path != job_dir / "postwake_composed_readback_receipt.json"
        or not isinstance(raw["launch_nonce"], str)
        or _NONCE.fullmatch(raw["launch_nonce"]) is None
        or _identifier(raw["attempt_id"], role="postwake custody seed attempt")
        != raw["attempt_id"]
        or _identifier(raw["run_id"], role="postwake custody seed run") != raw["run_id"]
        or execution_mode not in CAPSULE_EXECUTION_MODES
        or (execution_mode == "fresh" and retry_of is not None)
        or (
            execution_mode == "successor_resume"
            and (
                not isinstance(retry_of, str)
                or _IDENTIFIER.fullmatch(retry_of) is None
                or retry_of == raw["run_id"]
            )
        )
        or raw["seed_sha256"] != _postwake_custody_seed_sha256(unsigned)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "postwake custody seed violates its acyclic exact policy"
        )
    return OriginalConfirmatoryPostwakeCustodySeed(
        payload=unsigned,
        seed_sha256=raw["seed_sha256"],
    )


def build_original_confirmatory_postwake_custody_seed(
    *,
    q_authority_root_sha256: str,
    e_intent_path: str | Path,
    e_intent_file_sha256: str,
    e_intent_core_sha256: str,
    supervisor_job_id: str,
    supervisor_job_dir: str | Path,
    supervisor_spec_path: str | Path,
    launch_nonce: str,
    attempt_id: str,
    run_id: str,
    execution_mode: str,
    retry_of_run_id: str | None,
    execution_capsule_contract_sha256: str,
    capsule_sha256: str,
    supervisor_release_root_sha256: str,
    terminal_release_root_sha256: str,
    supervisor_terminal_receipt_path: str | Path,
    preterminal_pin_receipt_path: str | Path,
    postwake_input_lease_receipt_path: str | Path,
    composed_terminal_receipt_path: str | Path,
    postwake_composed_readback_receipt_path: str | Path,
) -> OriginalConfirmatoryPostwakeCustodySeed:
    unsigned = {
        "schema_version": 1,
        "policy": POSTWAKE_CUSTODY_SEED_POLICY,
        "q_authority_root_sha256": q_authority_root_sha256,
        "e_intent_path": str(e_intent_path),
        "e_intent_file_sha256": e_intent_file_sha256,
        "e_intent_core_sha256": e_intent_core_sha256,
        "supervisor_job_id": supervisor_job_id,
        "supervisor_job_dir": str(supervisor_job_dir),
        "supervisor_spec_path": str(supervisor_spec_path),
        "launch_nonce": launch_nonce,
        "attempt_id": attempt_id,
        "run_id": run_id,
        "execution_mode": execution_mode,
        "retry_of_run_id": retry_of_run_id,
        "execution_capsule_contract_sha256": (execution_capsule_contract_sha256),
        "capsule_sha256": capsule_sha256,
        "supervisor_release_root_sha256": supervisor_release_root_sha256,
        "terminal_release_root_sha256": terminal_release_root_sha256,
        "supervisor_terminal_receipt_path": str(supervisor_terminal_receipt_path),
        "preterminal_pin_receipt_path": str(preterminal_pin_receipt_path),
        "postwake_input_lease_receipt_path": str(postwake_input_lease_receipt_path),
        "composed_terminal_receipt_path": str(composed_terminal_receipt_path),
        "postwake_composed_readback_receipt_path": str(
            postwake_composed_readback_receipt_path
        ),
    }
    return canonical_original_confirmatory_postwake_custody_seed(
        {
            **unsigned,
            "seed_sha256": _postwake_custody_seed_sha256(unsigned),
        }
    )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryPostwakeCustodyHandshakeContract:
    """Exact live supervisor/C custody pipe and readback receipt contract."""

    supervisor_job_id: str
    postwake_custody_seed_sha256: str
    pipe_name: str
    pipe_owner_sid: str
    readback_receipt_path: Path
    expected_composed_command_sha256: str
    expected_composed_cwd: Path
    expected_composed_environment_sha256: str
    ready_max_bytes: int
    ack_max_bytes: int
    terminal_client_arrival_timeout_ms: int
    custody_exchange_timeout_ms: int
    contract_sha256: str

    def payload_without_self_hash(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": POSTWAKE_CUSTODY_HANDSHAKE_CONTRACT_POLICY,
            "supervisor_job_id": self.supervisor_job_id,
            "postwake_custody_seed_sha256": (self.postwake_custody_seed_sha256),
            "pipe_name": self.pipe_name,
            "pipe_name_derivation_policy": (POSTWAKE_CUSTODY_PIPE_DERIVATION_POLICY),
            "pipe_creation_policy": POSTWAKE_PIPE_CREATION_POLICY,
            "pipe_access_mode": "PIPE_ACCESS_DUPLEX",
            "pipe_open_mode_flags": [
                "FILE_FLAG_FIRST_PIPE_INSTANCE",
                "FILE_FLAG_OVERLAPPED",
            ],
            "pipe_mode_flags": [
                "PIPE_TYPE_MESSAGE",
                "PIPE_READMODE_MESSAGE",
                "PIPE_WAIT",
                "PIPE_REJECT_REMOTE_CLIENTS",
            ],
            "pipe_max_instances": 1,
            "pipe_inbound_buffer_bytes": self.ready_max_bytes,
            "pipe_outbound_buffer_bytes": self.ack_max_bytes,
            "pipe_security_policy": POSTWAKE_PIPE_SECURITY_POLICY,
            "pipe_owner_sid": self.pipe_owner_sid,
            "pipe_allowed_sids": sorted([self.pipe_owner_sid, "S-1-5-18"]),
            "pipe_dacl_protected": True,
            "pipe_handle_inheritable": False,
            "pipe_precreated_before_wake": True,
            "pipe_reject_remote_clients": True,
            "pipe_first_instance_required": True,
            "pipe_single_client_required": True,
            "shared_deadline_policy": POSTWAKE_SHARED_DEADLINE_POLICY,
            "event_wait_policy": POSTWAKE_EVENT_WAIT_POLICY,
            "ready_policy": POSTWAKE_CUSTODY_READY_POLICY,
            "ack_policy": POSTWAKE_CUSTODY_ACK_POLICY,
            "readback_receipt_path": str(self.readback_receipt_path),
            "readback_receipt_policy": POSTWAKE_COMPOSED_READBACK_RECEIPT_POLICY,
            "expected_composed_command_sha256": (self.expected_composed_command_sha256),
            "expected_composed_cwd": str(self.expected_composed_cwd),
            "expected_composed_environment_sha256": (
                self.expected_composed_environment_sha256
            ),
            "ready_max_bytes": self.ready_max_bytes,
            "ack_max_bytes": self.ack_max_bytes,
            "terminal_client_arrival_timeout_ms": (
                self.terminal_client_arrival_timeout_ms
            ),
            "terminal_client_arrival_deadline_scope": (
                "successful_codex_resume_to_accepted_claim_ready_v1"
            ),
            "custody_exchange_timeout_ms": self.custody_exchange_timeout_ms,
            "custody_exchange_deadline_scope": (
                "accepted_claim_ready_through_final_ack_v1"
            ),
            "overall_timeout_max_ms": TERMINAL_CUSTODY_OVERALL_TIMEOUT_MAX_MS,
            "arrival_and_exchange_waits_event_driven": True,
            "local_single_client_required": True,
            "exact_wake_tree_descendant_required": True,
            "native_supervisor_job_membership_required": True,
            "exact_wake_process_job_membership_required": True,
            "custody_client_process_job_membership_required": True,
            "same_supervisor_job_required": True,
            "client_process_identity_required": True,
            "client_command_line_peb_readback_required": True,
            "client_cwd_peb_readback_required": True,
            "client_environment_peb_readback_required": True,
            "process_duplicate_handle_required": True,
            "readback_create_new_retained_policy": (
                POSTWAKE_COMPOSED_READBACK_CREATE_POLICY
            ),
            "automatic_retry_allowed": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.payload_without_self_hash(),
            "contract_sha256": self.contract_sha256,
        }


_POSTWAKE_CUSTODY_HANDSHAKE_CONTRACT_FIELDS = {
    "schema_version",
    "policy",
    "supervisor_job_id",
    "postwake_custody_seed_sha256",
    "pipe_name",
    "pipe_name_derivation_policy",
    "pipe_creation_policy",
    "pipe_access_mode",
    "pipe_open_mode_flags",
    "pipe_mode_flags",
    "pipe_max_instances",
    "pipe_inbound_buffer_bytes",
    "pipe_outbound_buffer_bytes",
    "pipe_security_policy",
    "pipe_owner_sid",
    "pipe_allowed_sids",
    "pipe_dacl_protected",
    "pipe_handle_inheritable",
    "pipe_precreated_before_wake",
    "pipe_reject_remote_clients",
    "pipe_first_instance_required",
    "pipe_single_client_required",
    "shared_deadline_policy",
    "event_wait_policy",
    "ready_policy",
    "ack_policy",
    "readback_receipt_path",
    "readback_receipt_policy",
    "expected_composed_command_sha256",
    "expected_composed_cwd",
    "expected_composed_environment_sha256",
    "ready_max_bytes",
    "ack_max_bytes",
    "terminal_client_arrival_timeout_ms",
    "terminal_client_arrival_deadline_scope",
    "custody_exchange_timeout_ms",
    "custody_exchange_deadline_scope",
    "overall_timeout_max_ms",
    "arrival_and_exchange_waits_event_driven",
    "local_single_client_required",
    "exact_wake_tree_descendant_required",
    "native_supervisor_job_membership_required",
    "exact_wake_process_job_membership_required",
    "custody_client_process_job_membership_required",
    "same_supervisor_job_required",
    "client_process_identity_required",
    "client_command_line_peb_readback_required",
    "client_cwd_peb_readback_required",
    "client_environment_peb_readback_required",
    "process_duplicate_handle_required",
    "readback_create_new_retained_policy",
    "automatic_retry_allowed",
    "contract_sha256",
}
_POSTWAKE_PIPE_NAME = re.compile(r"\\\\\.\\pipe\\AANCA-composed-custody-[0-9a-f]{64}")


def canonical_original_confirmatory_postwake_custody_handshake_contract(
    value: OriginalConfirmatoryPostwakeCustodyHandshakeContract | Mapping[str, Any],
    *,
    custody_seed: OriginalConfirmatoryPostwakeCustodySeed | Mapping[str, Any],
) -> OriginalConfirmatoryPostwakeCustodyHandshakeContract:
    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatoryPostwakeCustodyHandshakeContract
        else _mapping(value, role="postwake custody handshake contract")
    )
    if set(raw) != _POSTWAKE_CUSTODY_HANDSHAKE_CONTRACT_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "postwake custody handshake contract has an unexpected field set"
        )
    canonical_seed = canonical_original_confirmatory_postwake_custody_seed(custody_seed)
    canonical_job_id = cast(
        str,
        canonical_seed.payload["supervisor_job_id"],
    )
    expected_pipe = "\\\\.\\pipe\\AANCA-composed-custody-" + canonical_seed.seed_sha256
    owner_sid = raw["pipe_owner_sid"]
    unsigned = {key: item for key, item in raw.items() if key != "contract_sha256"}
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != POSTWAKE_CUSTODY_HANDSHAKE_CONTRACT_POLICY
        or raw["supervisor_job_id"] != canonical_job_id
        or raw["postwake_custody_seed_sha256"] != canonical_seed.seed_sha256
        or raw["pipe_name"] != expected_pipe
        or not isinstance(raw["pipe_name"], str)
        or _POSTWAKE_PIPE_NAME.fullmatch(raw["pipe_name"]) is None
        or raw["pipe_name_derivation_policy"] != POSTWAKE_CUSTODY_PIPE_DERIVATION_POLICY
        or raw["pipe_creation_policy"] != POSTWAKE_PIPE_CREATION_POLICY
        or raw["pipe_access_mode"] != "PIPE_ACCESS_DUPLEX"
        or not _strict_json_value_equal(
            raw["pipe_open_mode_flags"],
            ["FILE_FLAG_FIRST_PIPE_INSTANCE", "FILE_FLAG_OVERLAPPED"],
        )
        or not _strict_json_value_equal(
            raw["pipe_mode_flags"],
            [
                "PIPE_TYPE_MESSAGE",
                "PIPE_READMODE_MESSAGE",
                "PIPE_WAIT",
                "PIPE_REJECT_REMOTE_CLIENTS",
            ],
        )
        or raw["pipe_max_instances"] != 1
        or type(raw["pipe_max_instances"]) is not int
        or type(raw["pipe_inbound_buffer_bytes"]) is not int
        or raw["pipe_inbound_buffer_bytes"] != raw["ready_max_bytes"]
        or type(raw["pipe_outbound_buffer_bytes"]) is not int
        or raw["pipe_outbound_buffer_bytes"] != raw["ack_max_bytes"]
        or raw["pipe_security_policy"] != POSTWAKE_PIPE_SECURITY_POLICY
        or not isinstance(owner_sid, str)
        or _WINDOWS_SID.fullmatch(owner_sid) is None
        or not _strict_json_value_equal(
            raw["pipe_allowed_sids"],
            sorted([owner_sid, "S-1-5-18"]),
        )
        or raw["pipe_dacl_protected"] is not True
        or raw["pipe_handle_inheritable"] is not False
        or raw["pipe_precreated_before_wake"] is not True
        or raw["pipe_reject_remote_clients"] is not True
        or raw["pipe_first_instance_required"] is not True
        or raw["pipe_single_client_required"] is not True
        or raw["shared_deadline_policy"] != POSTWAKE_SHARED_DEADLINE_POLICY
        or raw["event_wait_policy"] != POSTWAKE_EVENT_WAIT_POLICY
        or raw["ready_policy"] != POSTWAKE_CUSTODY_READY_POLICY
        or raw["ack_policy"] != POSTWAKE_CUSTODY_ACK_POLICY
        or raw["readback_receipt_policy"] != POSTWAKE_COMPOSED_READBACK_RECEIPT_POLICY
        or raw["terminal_client_arrival_deadline_scope"]
        != "successful_codex_resume_to_accepted_claim_ready_v1"
        or raw["custody_exchange_deadline_scope"]
        != "accepted_claim_ready_through_final_ack_v1"
        or raw["overall_timeout_max_ms"] != TERMINAL_CUSTODY_OVERALL_TIMEOUT_MAX_MS
        or type(raw["overall_timeout_max_ms"]) is not int
        or raw["arrival_and_exchange_waits_event_driven"] is not True
        or type(raw["terminal_client_arrival_timeout_ms"]) is not int
        or raw["terminal_client_arrival_timeout_ms"] <= 0
        or raw["terminal_client_arrival_timeout_ms"]
        > TERMINAL_CUSTODY_OVERALL_TIMEOUT_MAX_MS
        or type(raw["custody_exchange_timeout_ms"]) is not int
        or raw["custody_exchange_timeout_ms"] <= 0
        or raw["custody_exchange_timeout_ms"] > TERMINAL_CUSTODY_OVERALL_TIMEOUT_MAX_MS
        or raw["local_single_client_required"] is not True
        or raw["exact_wake_tree_descendant_required"] is not True
        or raw["native_supervisor_job_membership_required"] is not True
        or raw["exact_wake_process_job_membership_required"] is not True
        or raw["custody_client_process_job_membership_required"] is not True
        or raw["same_supervisor_job_required"] is not True
        or raw["client_process_identity_required"] is not True
        or raw["client_command_line_peb_readback_required"] is not True
        or raw["client_cwd_peb_readback_required"] is not True
        or raw["client_environment_peb_readback_required"] is not True
        or raw["process_duplicate_handle_required"] is not True
        or raw["readback_create_new_retained_policy"]
        != POSTWAKE_COMPOSED_READBACK_CREATE_POLICY
        or raw["automatic_retry_allowed"] is not False
        or raw["contract_sha256"] != canonical_json_sha256(unsigned)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "postwake custody handshake contract violates its exact policy"
        )
    return OriginalConfirmatoryPostwakeCustodyHandshakeContract(
        supervisor_job_id=canonical_job_id,
        postwake_custody_seed_sha256=canonical_seed.seed_sha256,
        pipe_name=expected_pipe,
        pipe_owner_sid=owner_sid,
        readback_receipt_path=Path(
            _absolute_path(
                raw["readback_receipt_path"],
                role="postwake composed readback receipt path",
            )
        ),
        expected_composed_command_sha256=_sha256(
            raw["expected_composed_command_sha256"],
            role="postwake composed command",
        ),
        expected_composed_cwd=Path(
            _absolute_path(
                raw["expected_composed_cwd"],
                role="postwake composed cwd",
            )
        ),
        expected_composed_environment_sha256=_sha256(
            raw["expected_composed_environment_sha256"],
            role="postwake composed environment",
        ),
        ready_max_bytes=_positive_int(
            raw["ready_max_bytes"],
            role="postwake custody READY bound",
        ),
        ack_max_bytes=_positive_int(
            raw["ack_max_bytes"],
            role="postwake custody ACK bound",
        ),
        terminal_client_arrival_timeout_ms=_positive_int(
            raw["terminal_client_arrival_timeout_ms"],
            role="postwake terminal-client arrival timeout",
        ),
        custody_exchange_timeout_ms=_positive_int(
            raw["custody_exchange_timeout_ms"],
            role="postwake custody exchange timeout",
        ),
        contract_sha256=raw["contract_sha256"],
    )


def build_original_confirmatory_postwake_custody_handshake_contract(
    *,
    custody_seed: OriginalConfirmatoryPostwakeCustodySeed | Mapping[str, Any],
    pipe_owner_sid: str,
    readback_receipt_path: str | Path,
    expected_composed_command_sha256: str,
    expected_composed_cwd: str | Path,
    expected_composed_environment_sha256: str,
    ready_max_bytes: int,
    ack_max_bytes: int,
    terminal_client_arrival_timeout_ms: int,
    custody_exchange_timeout_ms: int,
) -> OriginalConfirmatoryPostwakeCustodyHandshakeContract:
    canonical_seed = canonical_original_confirmatory_postwake_custody_seed(custody_seed)
    pipe_name = "\\\\.\\pipe\\AANCA-composed-custody-" + canonical_seed.seed_sha256
    provisional = OriginalConfirmatoryPostwakeCustodyHandshakeContract(
        supervisor_job_id=cast(
            str,
            canonical_seed.payload["supervisor_job_id"],
        ),
        postwake_custody_seed_sha256=canonical_seed.seed_sha256,
        pipe_name=pipe_name,
        pipe_owner_sid=pipe_owner_sid,
        readback_receipt_path=Path(str(readback_receipt_path)),
        expected_composed_command_sha256=expected_composed_command_sha256,
        expected_composed_cwd=Path(str(expected_composed_cwd)),
        expected_composed_environment_sha256=(expected_composed_environment_sha256),
        ready_max_bytes=ready_max_bytes,
        ack_max_bytes=ack_max_bytes,
        terminal_client_arrival_timeout_ms=terminal_client_arrival_timeout_ms,
        custody_exchange_timeout_ms=custody_exchange_timeout_ms,
        contract_sha256="0" * 64,
    )
    unsigned = provisional.payload_without_self_hash()
    return canonical_original_confirmatory_postwake_custody_handshake_contract(
        {**unsigned, "contract_sha256": canonical_json_sha256(unsigned)},
        custody_seed=canonical_seed,
    )


_OUTCOME_BLIND_EXPECTED_ARTIFACT_TEMPLATE_PROJECTION_FIELDS = {
    "schema_version",
    "policy",
    "ordered_role_templates",
    "template_rule_field_names",
    "instance_rule_field_names",
    "required_success_roles",
    "allowed_flat_json_selectors",
    "allowed_selector_json_types",
    "strict_expected_type_equality_required",
    "dotted_paths_allowed",
    "numeric_or_list_indirection_allowed",
    "empty_json_equals_requires_zero_json_decode",
    "exact_anchor_plus_relative_path_required",
    "expected_sha256_policy",
    "non_allowlisted_selectors_allowed",
    "forbidden_non_allowlisted_selector_tokens",
    "scientific_metric_ranking_prediction_or_outcome_selectors_allowed",
    "eligibility_control_exception",
    "pre_arm_validation_required",
    "outcome_values_read",
    "outcome_values_emitted",
    "outcome_values_used_for_selection_or_tuning",
    "projection_root_sha256",
}

_OUTCOME_BLIND_EXPECTED_ARTIFACT_INSTANCE_FIELDS = {
    "schema_version",
    "policy",
    "template_projection_root_sha256",
    "run_id",
    "expected_run_directory",
    "runs_root",
    "required_success_roles",
    "expected_artifacts",
    "expected_artifacts_root_sha256",
    "exact_run_directory_binding_required",
    "exact_path_order_and_containment_required",
    "strict_rule_type_equality_required",
    "non_allowlisted_rules_allowed",
    "pre_arm_validation_required",
    "outcome_values_read",
    "outcome_values_emitted",
    "outcome_values_used_for_selection_or_tuning",
    "projection_root_sha256",
}

_TERMINAL_PROCESS_IDENTITY_FIELDS = {
    "pid",
    "creation_time_100ns",
    "creation_time_utc",
    "program_path",
    "program_sha256",
    "command_sha256",
}

_TERMINAL_CLAIM_PHYSICAL_IDENTITY_FIELDS = {
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

_TERMINAL_CLAIM_STABLE_PHYSICAL_IDENTITY_FIELDS = {
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
}

_TERMINAL_CLAIM_READY_FIELDS = {
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

_TERMINAL_CUSTODY_GRANT_FIELDS = {
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

_TERMINAL_COMPOSED_READY_FIELDS = {
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

_TERMINAL_FINAL_ACK_FIELDS = {
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

_TERMINAL_POSTWAKE_COMPOSED_READBACK_FIELDS = {
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


def _strict_json_value_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            _strict_json_value_equal(actual[key], item)
            for key, item in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_json_value_equal(observed, item)
            for observed, item in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def build_original_confirmatory_outcome_blind_expected_artifact_projection() -> dict[
    str, Any
]:
    """Build Q's exact static, outcome-blind artifact-inspection template."""

    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "policy": OUTCOME_BLIND_EXPECTED_ARTIFACT_PROJECTION_POLICY,
        "ordered_role_templates": [
            {
                **template,
                "json_equals": dict(cast(Mapping[str, Any], template["json_equals"])),
            }
            for template in _EXPECTED_ARTIFACT_TEMPLATE
        ],
        "template_rule_field_names": sorted(_EXPECTED_ARTIFACT_TEMPLATE_RULE_FIELDS),
        "instance_rule_field_names": sorted(_EXPECTED_ARTIFACT_RULE_FIELDS),
        "required_success_roles": list(_EXPECTED_ARTIFACT_ROLE_ORDER),
        "allowed_flat_json_selectors": [
            "completion_stage",
            "run_id",
            "status",
            "study_outcome_eligible",
        ],
        "allowed_selector_json_types": {
            "completion_stage": "string",
            "run_id": "string",
            "status": "string",
            "study_outcome_eligible": "boolean",
        },
        "strict_expected_type_equality_required": True,
        "dotted_paths_allowed": False,
        "numeric_or_list_indirection_allowed": False,
        "empty_json_equals_requires_zero_json_decode": True,
        "exact_anchor_plus_relative_path_required": True,
        "expected_sha256_policy": "null_only_v1",
        "non_allowlisted_selectors_allowed": False,
        "forbidden_non_allowlisted_selector_tokens": [
            "metric",
            "metrics",
            "outcome",
            "p_value",
            "prediction",
            "predictions",
            "rank",
            "ranking",
            "rankings",
            "restoration",
        ],
        "scientific_metric_ranking_prediction_or_outcome_selectors_allowed": False,
        "eligibility_control_exception": {
            "role": "completion_evidence",
            "selector": "study_outcome_eligible",
            "classification": "frozen_eligibility_control_not_scientific_outcome_value",
            "expected_json_type": "boolean",
            "expected_value": True,
        },
        "pre_arm_validation_required": True,
        "outcome_values_read": False,
        "outcome_values_emitted": False,
        "outcome_values_used_for_selection_or_tuning": False,
    }
    return {
        **unsigned,
        "projection_root_sha256": canonical_json_sha256(unsigned),
    }


def canonical_original_confirmatory_outcome_blind_expected_artifact_projection(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, role="outcome-blind expected-artifact template")
    expected = build_original_confirmatory_outcome_blind_expected_artifact_projection()
    if (
        set(raw) != _OUTCOME_BLIND_EXPECTED_ARTIFACT_TEMPLATE_PROJECTION_FIELDS
        or not _strict_json_value_equal(raw, expected)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "outcome-blind expected-artifact template violates its exact policy"
        )
    return expected


def build_original_confirmatory_outcome_blind_expected_artifact_instance(
    *,
    run_id: str,
    expected_run_directory: str | Path,
) -> dict[str, Any]:
    """Instantiate the static artifact template for one exact confirmatory run."""

    canonical_run_id = _identifier(run_id, role="artifact inspection run id")
    run_directory = Path(
        _absolute_path(
            str(expected_run_directory),
            role="artifact inspection expected run directory",
        )
    )
    runs_root = run_directory.parent
    if run_directory != runs_root / canonical_run_id:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "artifact inspection run directory differs from its exact run id"
        )
    template_projection = (
        build_original_confirmatory_outcome_blind_expected_artifact_projection()
    )
    anchors = {
        "expected_run_directory": run_directory,
        "runs_root": runs_root,
    }
    rules: list[dict[str, Any]] = []
    for template_value in _EXPECTED_ARTIFACT_TEMPLATE:
        template = dict(template_value)
        anchor_name = cast(str, template["path_anchor"])
        relative_path = Path(cast(str, template["relative_path"]))
        if (
            anchor_name not in anchors
            or relative_path.is_absolute()
            or len(relative_path.parts) != 1
            or relative_path.parts[0] in {"", ".", ".."}
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "authority-owned artifact template has an unsafe path"
            )
        checks = {
            key: canonical_run_id if expected == "$RUN_ID" else expected
            for key, expected in cast(
                Mapping[str, Any],
                template["json_equals"],
            ).items()
        }
        rules.append(
            {
                "role": template["role"],
                "path": str(anchors[anchor_name] / relative_path),
                "expected_sha256": template["expected_sha256"],
                "must_be_absent_before": template["must_be_absent_before"],
                "json_equals": checks,
            }
        )
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "policy": OUTCOME_BLIND_EXPECTED_ARTIFACT_INSTANCE_POLICY,
        "template_projection_root_sha256": template_projection[
            "projection_root_sha256"
        ],
        "run_id": canonical_run_id,
        "expected_run_directory": str(run_directory),
        "runs_root": str(runs_root),
        "required_success_roles": list(_EXPECTED_ARTIFACT_ROLE_ORDER),
        "expected_artifacts": rules,
        "expected_artifacts_root_sha256": canonical_json_sha256(rules),
        "exact_run_directory_binding_required": True,
        "exact_path_order_and_containment_required": True,
        "strict_rule_type_equality_required": True,
        "non_allowlisted_rules_allowed": False,
        "pre_arm_validation_required": True,
        "outcome_values_read": False,
        "outcome_values_emitted": False,
        "outcome_values_used_for_selection_or_tuning": False,
    }
    return {
        **unsigned,
        "projection_root_sha256": canonical_json_sha256(unsigned),
    }


def canonical_original_confirmatory_outcome_blind_expected_artifact_instance(
    value: Mapping[str, Any],
    *,
    run_id: str,
    expected_run_directory: str | Path,
) -> dict[str, Any]:
    raw = _mapping(value, role="outcome-blind expected-artifact instance")
    expected = build_original_confirmatory_outcome_blind_expected_artifact_instance(
        run_id=run_id,
        expected_run_directory=expected_run_directory,
    )
    if (
        set(raw) != _OUTCOME_BLIND_EXPECTED_ARTIFACT_INSTANCE_FIELDS
        or not _strict_json_value_equal(raw, expected)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "outcome-blind expected-artifact instance violates its exact policy"
        )
    return expected


_CODEX_TERMINAL_WAKE_PROMPT_TEMPLATE_PROJECTION_FIELDS = {
    "schema_version",
    "policy",
    "template_utf8_lf",
    "template_sha256",
    "placeholder_order",
    "placeholder_occurrence_policy",
    "render_substitution_policy",
    "terminal_client_launcher_argv_json_policy",
    "remaining_braces_allowed",
    "render_output_terminal_lf_required",
    "rendered_prompt_sha256_required",
    "job_or_command_discovery_allowed",
    "fallback_allowed",
    "automatic_retry_allowed",
    "outcome_values_read",
    "projection_root_sha256",
}


def build_original_confirmatory_codex_terminal_wake_prompt_template_projection() -> (
    dict[str, Any]
):
    """Build Q's static deterministic Codex terminal-wake prompt contract."""

    template = CODEX_TERMINAL_WAKE_PROMPT_TEMPLATE
    if (
        not template.endswith("\n")
        or "\r" in template
        or any(ord(character) > 127 for character in template)
        or any(
            template.count("{" + placeholder + "}") != 1
            for placeholder in CODEX_TERMINAL_WAKE_PROMPT_PLACEHOLDER_ORDER
        )
        or not _strict_json_value_equal(
            re.findall(r"\{([a-z0-9_]+)\}", template),
            list(CODEX_TERMINAL_WAKE_PROMPT_PLACEHOLDER_ORDER),
        )
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Codex terminal wake prompt template violates its exact placeholder policy"
        )
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "policy": CODEX_TERMINAL_WAKE_PROMPT_RENDER_POLICY,
        "template_utf8_lf": template,
        "template_sha256": hashlib.sha256(template.encode("ascii")).hexdigest(),
        "placeholder_order": list(CODEX_TERMINAL_WAKE_PROMPT_PLACEHOLDER_ORDER),
        "placeholder_occurrence_policy": "each_exactly_once_in_declared_order_v1",
        "render_substitution_policy": "literal_ordered_str_replace_v1",
        "terminal_client_launcher_argv_json_policy": (
            "utf8_single_line_compact_json_list_ensure_ascii_true_v1"
        ),
        "remaining_braces_allowed": False,
        "render_output_terminal_lf_required": True,
        "rendered_prompt_sha256_required": True,
        "job_or_command_discovery_allowed": False,
        "fallback_allowed": False,
        "automatic_retry_allowed": False,
        "outcome_values_read": False,
    }
    return {
        **unsigned,
        "projection_root_sha256": canonical_json_sha256(unsigned),
    }


def canonical_original_confirmatory_codex_terminal_wake_prompt_template_projection(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, role="Codex terminal wake prompt template")
    expected = (
        build_original_confirmatory_codex_terminal_wake_prompt_template_projection()
    )
    if (
        set(raw) != _CODEX_TERMINAL_WAKE_PROMPT_TEMPLATE_PROJECTION_FIELDS
        or not _strict_json_value_equal(raw, expected)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Codex terminal wake prompt template violates its exact policy"
        )
    return expected


def render_original_confirmatory_codex_terminal_wake_prompt(
    *,
    job_id: str,
    supervisor_job_directory: str | Path,
    supervisor_spec_path: str | Path,
    supervisor_spec_sha256: str,
    terminal_receipt_sha256: str,
    terminal_client_launcher_argv: Sequence[str],
    terminal_client_launcher_command_sha256: str,
    verify_terminal_command_sha256: str,
) -> tuple[str, str]:
    """Render one exact per-job prompt with literal ordered replacement."""

    canonical_job_id = _identifier(job_id, role="Codex terminal wake job id")
    job_directory = _absolute_path(
        str(supervisor_job_directory),
        role="Codex terminal wake job directory",
    )
    spec_path = _absolute_path(
        str(supervisor_spec_path),
        role="Codex terminal wake spec path",
    )
    if Path(spec_path) != Path(job_directory) / "run_spec.json":
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Codex terminal wake spec path differs from its exact job directory"
        )
    argv = list(terminal_client_launcher_argv)
    if not argv or any(
        type(item) is not str
        or not item
        or any(ord(character) < 32 or ord(character) == 127 for character in item)
        or "{" in item
        or "}" in item
        for item in argv
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Codex terminal wake launcher argv is not an exact safe string list"
        )
    replacements = {
        "job_id": canonical_job_id,
        "supervisor_job_directory": job_directory,
        "supervisor_spec_path": spec_path,
        "supervisor_spec_sha256": _sha256(
            supervisor_spec_sha256,
            role="Codex terminal wake supervisor spec",
        ),
        "terminal_receipt_sha256": _sha256(
            terminal_receipt_sha256,
            role="Codex terminal wake terminal receipt",
        ),
        "terminal_client_launcher_argv_json": json.dumps(
            argv,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        "terminal_client_launcher_command_sha256": _sha256(
            terminal_client_launcher_command_sha256,
            role="Codex terminal wake launcher command",
        ),
        "verify_terminal_command_sha256": _sha256(
            verify_terminal_command_sha256,
            role="Codex terminal wake child command",
        ),
    }
    rendered = CODEX_TERMINAL_WAKE_PROMPT_TEMPLATE
    for placeholder in CODEX_TERMINAL_WAKE_PROMPT_PLACEHOLDER_ORDER:
        replacement = replacements[placeholder]
        if (
            "{" in replacement
            or "}" in replacement
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in replacement
            )
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "Codex terminal wake replacement contains a forbidden brace/control"
            )
        rendered = rendered.replace("{" + placeholder + "}", replacement)
    if (
        "{" in rendered
        or "}" in rendered
        or "\r" in rendered
        or not rendered.endswith("\n")
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Codex terminal wake rendered prompt violates its exact policy"
        )
    return rendered, hashlib.sha256(rendered.encode("utf-8")).hexdigest()


_TERMINAL_CLIENT_LAUNCHER_E_PROJECTION_FIELDS = {
    "schema_version",
    "policy",
    "launcher_release_root_sha256",
    "program_path",
    "program_sha256",
    "runtime_python_path",
    "runtime_python_sha256",
    "source_path",
    "source_size_bytes",
    "source_sha256",
    "python_isolated_flags",
    "job_id",
    "supervisor_job_directory",
    "supervisor_spec_path",
    "e_intent_path",
    "terminal_receipt_path",
    "launch_intent_path",
    "python_sys_argv_template",
    "process_argv_template",
    "downstream_placeholder_order",
    "verify_terminal_command_projection_sha256",
    "verify_terminal_command_projection_binding",
    "verify_terminal_environment_sha256",
    "verify_terminal_environment_binding",
    "verify_terminal_cwd",
    "verify_terminal_cwd_root_sha256",
    "verify_terminal_launch_root_sha256",
    "verify_terminal_child_launch_topology",
    "verify_terminal_immediate_redirector_program_path",
    "verify_terminal_immediate_redirector_program_sha256",
    "verify_terminal_runtime_child_program_path",
    "verify_terminal_runtime_child_program_sha256",
    "supervisor_retains_launcher_redirector_child_process_handles_through_final_ack_required",
    "immediate_venv_redirector_live_through_runtime_child_exit_required",
    "terminal_client_launcher_live_through_redirector_waitforexit_required",
    "process_liveness_reverified_at_final_ack_required",
    "launch_intent_write_through_fsync_readonly_required",
    "launch_intent_physical_identity_policy",
    "launch_intent_physical_identity_role",
    "launch_intent_physical_identity_field_names",
    "launch_intent_creator_desired_access_mask",
    "launch_intent_creator_share_access",
    "launch_intent_delete_access_allowed",
    "launch_intent_delete_share_allowed",
    "launch_intent_cleanup_allowed",
    "launch_intent_same_handle_write_fsync_readback_required",
    "launch_intent_set_readonly_before_child_create_required",
    "launch_intent_creator_handle_retained_through_terminal_child_waitforexit_required",
    "terminal_child_launch_intent_no_follow_readonly_single_link_no_ads_required",
    "terminal_child_launch_intent_handle_retained_through_final_ack_required",
    "supervisor_launch_intent_handle_retained_through_final_ack_required",
    "claim_binds_launch_intent_deterministic_path_and_policy_only",
    "claim_launch_intent_read_must_be_false",
    "grant_required_before_terminal_child_launch_intent_read",
    "grant_independently_verifies_launch_intent_and_launcher_identity",
    "launch_intent_supervisor_granted_access_mask",
    "launch_intent_child_duplicate_target_access_mask",
    "launch_intent_child_expected_granted_access_mask",
    "launch_intent_child_duplicate_options",
    "launch_intent_child_duplicate_close_source",
    "child_stdio_policy",
    "child_stdin_source",
    "child_stdout_transport",
    "child_stderr_transport",
    "child_inherited_handle_list_policy",
    "child_inherited_handle_count",
    "createprocess_inherit_handles",
    "startupinfoex_use_std_handles_required",
    "proc_thread_attribute_handle_list_required",
    "non_stdio_inherited_handles_allowed",
    "preterminal_or_terminal_input_file_handles_inherited_allowed",
    "child_stdout_max_bytes",
    "child_stderr_max_bytes",
    "child_stdout_summary_policy",
    "child_stdout_single_canonical_json_line_required",
    "child_stderr_empty_required",
    "stdio_pipe_drains_event_driven_concurrent_required",
    "launcher_forwards_validated_child_stdout_once_required",
    "launcher_cwd",
    "command_preimage_field_names",
    "command_sha256_policy",
    "wake_intent_hash_in_launcher_argv_allowed",
    "preterminal_pin_terminal_or_lease_input_read_allowed",
    "launch_intent_create_new_required",
    "same_job_no_breakaway_required",
    "automatic_retry_allowed",
    "fallback_allowed",
    "projection_root_sha256",
}
_TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDER_ORDER = (
    "supervisor_spec_sha256",
    "e_intent_file_sha256",
    "terminal_receipt_sha256",
    "verify_terminal_command_sha256",
)
_TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDERS = {
    "supervisor_spec_sha256": "$SUPERVISOR_SPEC_SHA256",
    "e_intent_file_sha256": "$E_INTENT_FILE_SHA256",
    "terminal_receipt_sha256": "$TERMINAL_RECEIPT_SHA256",
    "verify_terminal_command_sha256": "$VERIFY_TERMINAL_COMMAND_SHA256",
}
_TERMINAL_CLIENT_LAUNCHER_COMMAND_PREIMAGE_FIELDS = {
    "schema_version",
    "policy",
    "launcher_projection_root_sha256",
    "program_path",
    "program_sha256",
    "python_sys_argv",
    "process_argv",
    "cwd",
    "supervisor_spec_sha256",
    "e_intent_file_sha256",
    "terminal_receipt_sha256",
    "verify_terminal_command_sha256",
    "verify_terminal_launch_root_sha256",
}
_TERMINAL_CLIENT_LAUNCHER_COMMAND_FIELDS = {
    *_TERMINAL_CLIENT_LAUNCHER_COMMAND_PREIMAGE_FIELDS,
    "command_sha256",
}
_TERMINAL_CLIENT_LAUNCH_INTENT_FIELDS = {
    "schema_version",
    "policy",
    "status",
    "job_id",
    "supervisor_job_directory",
    "supervisor_spec_path",
    "supervisor_spec_sha256",
    "e_intent_path",
    "e_intent_file_sha256",
    "terminal_receipt_path",
    "terminal_receipt_sha256",
    "launcher_command_sha256",
    "verify_terminal_command_projection_sha256",
    "verify_terminal_command_sha256",
    "verify_terminal_environment_sha256",
    "verify_terminal_cwd",
    "launcher_process_identity",
    "launch_attempt_count",
    "create_disposition",
    "child_process_created_before_intent",
    "existing_or_partial_intent_is_stop",
    "automatic_retry_allowed",
    "created_at_utc",
    "intent_root_sha256",
}
_TERMINAL_CLIENT_LAUNCH_INTENT_STATUS = "reserved_before_verify_terminal_createprocess"


def _terminal_client_launcher_argv_template(
    *,
    source_path: str,
    job_id: str,
    job_directory: str,
    supervisor_spec_path: str,
    e_intent_path: str,
    terminal_receipt_path: str,
    launch_intent_path: str,
    verify_terminal_command_projection_sha256: str,
    verify_terminal_environment_sha256: str,
    verify_terminal_cwd: str,
) -> list[str]:
    return [
        source_path,
        "--job-id",
        job_id,
        "--supervisor-job-directory",
        job_directory,
        "--supervisor-spec",
        supervisor_spec_path,
        "--supervisor-spec-sha256",
        _TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDERS["supervisor_spec_sha256"],
        "--e-intent",
        e_intent_path,
        "--e-intent-sha256",
        _TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDERS["e_intent_file_sha256"],
        "--terminal-receipt",
        terminal_receipt_path,
        "--terminal-receipt-sha256",
        _TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDERS["terminal_receipt_sha256"],
        "--terminal-client-launch-intent",
        launch_intent_path,
        "--verify-terminal-command-projection-sha256",
        verify_terminal_command_projection_sha256,
        "--verify-terminal-command-sha256",
        _TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDERS["verify_terminal_command_sha256"],
        "--verify-terminal-environment-sha256",
        verify_terminal_environment_sha256,
        "--verify-terminal-cwd",
        verify_terminal_cwd,
    ]


def build_original_confirmatory_terminal_client_launcher_projection(
    *,
    launcher_release: Mapping[str, Any],
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    job_id: str,
    supervisor_job_directory: str | Path,
    verify_terminal_command_projection_sha256: str,
    verify_terminal_environment_sha256: str,
    verify_terminal_cwd: str | Path,
) -> dict[str, Any]:
    """Build E's acyclic launcher derivation; it contains no future hashes."""

    canonical_capsule = canonical_original_confirmatory_execution_capsule(capsule)
    release = canonical_original_confirmatory_terminal_client_launcher_release(
        launcher_release,
        capsule=canonical_capsule,
    )
    canonical_job_id = _identifier(job_id, role="terminal-client launcher job id")
    job_directory = Path(
        _absolute_path(
            str(supervisor_job_directory),
            role="terminal-client launcher job directory",
        )
    )
    if job_directory.name != canonical_job_id:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "terminal-client launcher job directory differs from its exact job id"
        )
    spec_path = job_directory / "run_spec.json"
    e_path = (
        job_directory.parent.parent
        / CONTROL_STAGING_DIRECTORY_NAME
        / canonical_job_id
        / E_INTENT_FILENAME
    )
    terminal_path = job_directory / "terminal_receipt.json"
    launch_intent_path = job_directory / TERMINAL_CLIENT_LAUNCH_INTENT_FILENAME
    child_projection_sha256 = _sha256(
        verify_terminal_command_projection_sha256,
        role="terminal-client child command projection",
    )
    child_environment_sha256 = _sha256(
        verify_terminal_environment_sha256,
        role="terminal-client child environment",
    )
    child_cwd = _absolute_path(
        str(verify_terminal_cwd),
        role="terminal-client child cwd",
    )
    child_cwd_root = canonical_json_sha256({"cwd": child_cwd})
    child_launch_root = canonical_json_sha256(
        {
            "verify_terminal_command_projection_sha256": child_projection_sha256,
            "verify_terminal_environment_sha256": child_environment_sha256,
            "verify_terminal_cwd": child_cwd,
            "verify_terminal_cwd_root_sha256": child_cwd_root,
        }
    )
    python_sys_argv_template = _terminal_client_launcher_argv_template(
        source_path=release["source_path"],
        job_id=canonical_job_id,
        job_directory=str(job_directory),
        supervisor_spec_path=str(spec_path),
        e_intent_path=str(e_path),
        terminal_receipt_path=str(terminal_path),
        launch_intent_path=str(launch_intent_path),
        verify_terminal_command_projection_sha256=child_projection_sha256,
        verify_terminal_environment_sha256=child_environment_sha256,
        verify_terminal_cwd=child_cwd,
    )
    process_argv_template = [
        release["program_path"],
        *release["python_isolated_flags"],
        *python_sys_argv_template,
    ]
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "policy": TERMINAL_CLIENT_LAUNCHER_E_PROJECTION_POLICY,
        "launcher_release_root_sha256": release["release_root_sha256"],
        "program_path": release["program_path"],
        "program_sha256": release["program_sha256"],
        "runtime_python_path": release["runtime_python_path"],
        "runtime_python_sha256": release["runtime_python_sha256"],
        "source_path": release["source_path"],
        "source_size_bytes": release["source_size_bytes"],
        "source_sha256": release["source_sha256"],
        "python_isolated_flags": list(TERMINAL_CLIENT_PYTHON_ISOLATED_FLAGS),
        "job_id": canonical_job_id,
        "supervisor_job_directory": str(job_directory),
        "supervisor_spec_path": str(spec_path),
        "e_intent_path": str(e_path),
        "terminal_receipt_path": str(terminal_path),
        "launch_intent_path": str(launch_intent_path),
        "python_sys_argv_template": python_sys_argv_template,
        "process_argv_template": process_argv_template,
        "downstream_placeholder_order": list(
            _TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDER_ORDER
        ),
        "verify_terminal_command_projection_sha256": child_projection_sha256,
        "verify_terminal_command_projection_binding": (
            "E.command_projections.verify-terminal.projection_sha256"
        ),
        "verify_terminal_environment_sha256": child_environment_sha256,
        "verify_terminal_environment_binding": (
            "E.process_environment_binding.exact_integrity_verifier_environment_sha256"
        ),
        "verify_terminal_cwd": child_cwd,
        "verify_terminal_cwd_root_sha256": child_cwd_root,
        "verify_terminal_launch_root_sha256": child_launch_root,
        "verify_terminal_child_launch_topology": release[
            "verify_terminal_child_launch_topology"
        ],
        "verify_terminal_immediate_redirector_program_path": release[
            "verify_terminal_immediate_redirector_program_path"
        ],
        "verify_terminal_immediate_redirector_program_sha256": release[
            "verify_terminal_immediate_redirector_program_sha256"
        ],
        "verify_terminal_runtime_child_program_path": release[
            "verify_terminal_runtime_child_program_path"
        ],
        "verify_terminal_runtime_child_program_sha256": release[
            "verify_terminal_runtime_child_program_sha256"
        ],
        "supervisor_retains_launcher_redirector_child_process_handles_through_final_ack_required": (
            release[
                "supervisor_retains_launcher_redirector_child_process_handles_through_final_ack_required"
            ]
        ),
        "immediate_venv_redirector_live_through_runtime_child_exit_required": release[
            "immediate_venv_redirector_live_through_runtime_child_exit_required"
        ],
        "terminal_client_launcher_live_through_redirector_waitforexit_required": release[
            "terminal_client_launcher_live_through_redirector_waitforexit_required"
        ],
        "process_liveness_reverified_at_final_ack_required": release[
            "process_liveness_reverified_at_final_ack_required"
        ],
        "launch_intent_write_through_fsync_readonly_required": release[
            "launch_intent_write_through_fsync_readonly_required"
        ],
        "launch_intent_physical_identity_policy": release[
            "launch_intent_physical_identity_policy"
        ],
        "launch_intent_physical_identity_role": release[
            "launch_intent_physical_identity_role"
        ],
        "launch_intent_physical_identity_field_names": release[
            "launch_intent_physical_identity_field_names"
        ],
        "launch_intent_creator_desired_access_mask": release[
            "launch_intent_creator_desired_access_mask"
        ],
        "launch_intent_creator_share_access": release[
            "launch_intent_creator_share_access"
        ],
        "launch_intent_delete_access_allowed": release[
            "launch_intent_delete_access_allowed"
        ],
        "launch_intent_delete_share_allowed": release[
            "launch_intent_delete_share_allowed"
        ],
        "launch_intent_cleanup_allowed": release["launch_intent_cleanup_allowed"],
        "launch_intent_same_handle_write_fsync_readback_required": release[
            "launch_intent_same_handle_write_fsync_readback_required"
        ],
        "launch_intent_set_readonly_before_child_create_required": release[
            "launch_intent_set_readonly_before_child_create_required"
        ],
        "launch_intent_creator_handle_retained_through_terminal_child_waitforexit_required": (
            release[
                "launch_intent_creator_handle_retained_through_terminal_child_waitforexit_required"
            ]
        ),
        "terminal_child_launch_intent_no_follow_readonly_single_link_no_ads_required": (
            release[
                "terminal_child_launch_intent_no_follow_readonly_single_link_no_ads_required"
            ]
        ),
        "terminal_child_launch_intent_handle_retained_through_final_ack_required": release[
            "terminal_child_launch_intent_handle_retained_through_final_ack_required"
        ],
        "supervisor_launch_intent_handle_retained_through_final_ack_required": release[
            "supervisor_launch_intent_handle_retained_through_final_ack_required"
        ],
        "claim_binds_launch_intent_deterministic_path_and_policy_only": release[
            "claim_binds_launch_intent_deterministic_path_and_policy_only"
        ],
        "claim_launch_intent_read_must_be_false": release[
            "claim_launch_intent_read_must_be_false"
        ],
        "grant_required_before_terminal_child_launch_intent_read": release[
            "grant_required_before_terminal_child_launch_intent_read"
        ],
        "grant_independently_verifies_launch_intent_and_launcher_identity": release[
            "grant_independently_verifies_launch_intent_and_launcher_identity"
        ],
        "launch_intent_supervisor_granted_access_mask": release[
            "launch_intent_supervisor_granted_access_mask"
        ],
        "launch_intent_child_duplicate_target_access_mask": release[
            "launch_intent_child_duplicate_target_access_mask"
        ],
        "launch_intent_child_expected_granted_access_mask": release[
            "launch_intent_child_expected_granted_access_mask"
        ],
        "launch_intent_child_duplicate_options": release[
            "launch_intent_child_duplicate_options"
        ],
        "launch_intent_child_duplicate_close_source": release[
            "launch_intent_child_duplicate_close_source"
        ],
        "child_stdio_policy": release["child_stdio_policy"],
        "child_stdin_source": release["child_stdin_source"],
        "child_stdout_transport": release["child_stdout_transport"],
        "child_stderr_transport": release["child_stderr_transport"],
        "child_inherited_handle_list_policy": release[
            "child_inherited_handle_list_policy"
        ],
        "child_inherited_handle_count": release["child_inherited_handle_count"],
        "createprocess_inherit_handles": release["createprocess_inherit_handles"],
        "startupinfoex_use_std_handles_required": release[
            "startupinfoex_use_std_handles_required"
        ],
        "proc_thread_attribute_handle_list_required": release[
            "proc_thread_attribute_handle_list_required"
        ],
        "non_stdio_inherited_handles_allowed": release[
            "non_stdio_inherited_handles_allowed"
        ],
        "preterminal_or_terminal_input_file_handles_inherited_allowed": release[
            "preterminal_or_terminal_input_file_handles_inherited_allowed"
        ],
        "child_stdout_max_bytes": release["child_stdout_max_bytes"],
        "child_stderr_max_bytes": release["child_stderr_max_bytes"],
        "child_stdout_summary_policy": release["child_stdout_summary_policy"],
        "child_stdout_single_canonical_json_line_required": release[
            "child_stdout_single_canonical_json_line_required"
        ],
        "child_stderr_empty_required": release["child_stderr_empty_required"],
        "stdio_pipe_drains_event_driven_concurrent_required": release[
            "stdio_pipe_drains_event_driven_concurrent_required"
        ],
        "launcher_forwards_validated_child_stdout_once_required": release[
            "launcher_forwards_validated_child_stdout_once_required"
        ],
        "launcher_cwd": child_cwd,
        "command_preimage_field_names": sorted(
            _TERMINAL_CLIENT_LAUNCHER_COMMAND_PREIMAGE_FIELDS
        ),
        "command_sha256_policy": SUPERVISOR_PROCESS_COMMAND_HASH_POLICY,
        "wake_intent_hash_in_launcher_argv_allowed": False,
        "preterminal_pin_terminal_or_lease_input_read_allowed": False,
        "launch_intent_create_new_required": True,
        "same_job_no_breakaway_required": True,
        "automatic_retry_allowed": False,
        "fallback_allowed": False,
    }
    return {
        **unsigned,
        "projection_root_sha256": canonical_json_sha256(unsigned),
    }


def canonical_original_confirmatory_terminal_client_launcher_projection(
    value: Mapping[str, Any],
    *,
    launcher_release: Mapping[str, Any],
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, role="terminal-client launcher E projection")
    expected = build_original_confirmatory_terminal_client_launcher_projection(
        launcher_release=launcher_release,
        capsule=capsule,
        job_id=cast(str, raw.get("job_id")),
        supervisor_job_directory=cast(str, raw.get("supervisor_job_directory")),
        verify_terminal_command_projection_sha256=cast(
            str,
            raw.get("verify_terminal_command_projection_sha256"),
        ),
        verify_terminal_environment_sha256=cast(
            str,
            raw.get("verify_terminal_environment_sha256"),
        ),
        verify_terminal_cwd=cast(str, raw.get("verify_terminal_cwd")),
    )
    if set(
        raw
    ) != _TERMINAL_CLIENT_LAUNCHER_E_PROJECTION_FIELDS or not _strict_json_value_equal(
        raw, expected
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "terminal-client launcher E projection violates its exact acyclic policy"
        )
    return expected


def _derive_original_confirmatory_terminal_command_projection_from_concrete(
    *,
    command: OriginalConfirmatoryCapsuleCommand,
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    canonical_tail: Sequence[str],
) -> OriginalConfirmatoryCapsuleCommandProjection:
    """Recover the exact acyclic E projection represented by one final C."""

    tail = list(canonical_tail)
    if len(tail) % 2:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "concrete verify-terminal tail is not exact flag/value pairs"
        )
    flags = tail[::2]
    if len(set(flags)) != len(flags):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "concrete verify-terminal tail repeats a flag"
        )
    values = dict(zip(flags, tail[1::2], strict=True))
    required = {
        *CAPSULE_COMMON_TAIL_FLAGS,
        *CAPSULE_TERMINAL_SUFFIX_FLAGS,
    }
    if not required <= values.keys():
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "concrete verify-terminal tail lacks a projection field"
        )
    return build_original_confirmatory_capsule_command_projection(
        capsule=capsule,
        capsule_mode=CAPSULE_TERMINAL_MODE,
        e_intent_path=values["--e-intent"],
        q_authority_root_sha256=values["--q-authority-root-sha256"],
        launch_nonce=values["--launch-nonce"],
        supervisor_job_id=values["--supervisor-job-id"],
        supervisor_job_directory=values["--supervisor-job-dir"],
        attempt_id=values["--attempt-id"],
        run_id=values["--run-id"],
        execution_mode=values["--execution-mode"],
        retry_of_run_id=values.get(CAPSULE_SUCCESSOR_LINEAGE_FLAG),
        cwd=command.cwd,
        supervisor_terminal_path=values["--supervisor-terminal"],
        verifier_stdout_path=values["--verifier-stdout"],
        preterminal_pin_path=values["--preterminal-pin"],
        composed_terminal_path=values["--composed-terminal"],
    )


def build_original_confirmatory_terminal_client_launcher_command(
    *,
    launcher_projection: Mapping[str, Any],
    launcher_release: Mapping[str, Any],
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    supervisor_spec_sha256: str,
    e_intent_file_sha256: str,
    terminal_receipt_sha256: str,
    verify_terminal_command: OriginalConfirmatoryCapsuleCommand | Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize the sole launcher command after E, spec, and T are sealed."""

    canonical_capsule = canonical_original_confirmatory_execution_capsule(capsule)
    release = canonical_original_confirmatory_terminal_client_launcher_release(
        launcher_release,
        capsule=canonical_capsule,
    )
    projection = canonical_original_confirmatory_terminal_client_launcher_projection(
        launcher_projection,
        launcher_release=release,
        capsule=canonical_capsule,
    )
    command_raw = (
        verify_terminal_command.as_dict()
        if type(verify_terminal_command) is OriginalConfirmatoryCapsuleCommand
        else _mapping(verify_terminal_command, role="terminal-client child command")
    )
    child_argv = command_raw.get("argv")
    child_prefix_length = 1 + len(CAPSULE_PYTHON_ISOLATED_FLAGS) + 2
    if not isinstance(child_argv, list) or len(child_argv) < child_prefix_length:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "terminal-client child command argv is incomplete"
        )
    child = canonical_original_confirmatory_capsule_command(
        command_raw,
        capsule=canonical_capsule,
        expected_mode=CAPSULE_TERMINAL_MODE,
        expected_tail_argv=child_argv[child_prefix_length:],
    )
    child_tail = list(child.argv[child_prefix_length:])
    tail_values = dict(zip(child_tail[::2], child_tail[1::2], strict=True))
    child_projection = (
        _derive_original_confirmatory_terminal_command_projection_from_concrete(
            command=child,
            capsule=canonical_capsule,
            canonical_tail=child_tail,
        )
    )
    if (
        child.cwd != Path(projection["verify_terminal_cwd"])
        or child_projection.projection_sha256
        != projection["verify_terminal_command_projection_sha256"]
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "terminal-client child command differs from E's exact non-self projection"
        )
    replacements = {
        _TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDERS["supervisor_spec_sha256"]: _sha256(
            supervisor_spec_sha256,
            role="terminal-client downstream supervisor spec",
        ),
        _TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDERS["e_intent_file_sha256"]: _sha256(
            e_intent_file_sha256,
            role="terminal-client downstream E",
        ),
        _TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDERS["terminal_receipt_sha256"]: _sha256(
            terminal_receipt_sha256,
            role="terminal-client downstream terminal receipt",
        ),
        _TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDERS[
            "verify_terminal_command_sha256"
        ]: child.command_sha256,
    }
    if (
        tail_values["--e-intent-sha256"]
        != replacements[
            _TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDERS["e_intent_file_sha256"]
        ]
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "terminal-client child command E hash differs from the launcher binding"
        )

    def finalize(template: Sequence[str]) -> list[str]:
        final: list[str] = []
        observed_placeholders: list[str] = []
        reverse = {
            placeholder: name
            for name, placeholder in _TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDERS.items()
        }
        for item in template:
            if item in reverse:
                observed_placeholders.append(reverse[item])
                final.append(replacements[item])
            else:
                if "$" in item:
                    raise OriginalConfirmatoryCapsuleAuthorityError(
                        "terminal-client launcher template contains an unknown placeholder"
                    )
                final.append(item)
        if not _strict_json_value_equal(
            observed_placeholders,
            list(_TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDER_ORDER),
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "terminal-client launcher placeholders differ from their exact order"
            )
        return final

    python_sys_argv = finalize(projection["python_sys_argv_template"])
    process_argv = [
        release["program_path"],
        *release["python_isolated_flags"],
        *python_sys_argv,
    ]
    if not _strict_json_value_equal(
        process_argv,
        finalize(projection["process_argv_template"]),
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "terminal-client launcher process/sys argv derivations disagree"
        )
    preimage: dict[str, Any] = {
        "schema_version": 1,
        "policy": TERMINAL_CLIENT_LAUNCHER_COMMAND_POLICY,
        "launcher_projection_root_sha256": projection["projection_root_sha256"],
        "program_path": release["program_path"],
        "program_sha256": release["program_sha256"],
        "python_sys_argv": python_sys_argv,
        "process_argv": process_argv,
        "cwd": projection["launcher_cwd"],
        "supervisor_spec_sha256": replacements[
            _TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDERS["supervisor_spec_sha256"]
        ],
        "e_intent_file_sha256": replacements[
            _TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDERS["e_intent_file_sha256"]
        ],
        "terminal_receipt_sha256": replacements[
            _TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDERS["terminal_receipt_sha256"]
        ],
        "verify_terminal_command_sha256": child.command_sha256,
        "verify_terminal_launch_root_sha256": projection[
            "verify_terminal_launch_root_sha256"
        ],
    }
    return {
        **preimage,
        "command_sha256": canonical_json_sha256(preimage),
    }


def canonical_original_confirmatory_terminal_client_launcher_command(
    value: Mapping[str, Any],
    *,
    launcher_projection: Mapping[str, Any],
    launcher_release: Mapping[str, Any],
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    verify_terminal_command: OriginalConfirmatoryCapsuleCommand | Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, role="terminal-client launcher command")
    expected = build_original_confirmatory_terminal_client_launcher_command(
        launcher_projection=launcher_projection,
        launcher_release=launcher_release,
        capsule=capsule,
        supervisor_spec_sha256=cast(str, raw.get("supervisor_spec_sha256")),
        e_intent_file_sha256=cast(str, raw.get("e_intent_file_sha256")),
        terminal_receipt_sha256=cast(str, raw.get("terminal_receipt_sha256")),
        verify_terminal_command=verify_terminal_command,
    )
    if set(
        raw
    ) != _TERMINAL_CLIENT_LAUNCHER_COMMAND_FIELDS or not _strict_json_value_equal(
        raw, expected
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "terminal-client launcher command violates its exact final policy"
        )
    return expected


def build_original_confirmatory_terminal_client_launch_intent(
    *,
    launcher_command: Mapping[str, Any],
    launcher_projection: Mapping[str, Any],
    launcher_release: Mapping[str, Any],
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    verify_terminal_command: OriginalConfirmatoryCapsuleCommand | Mapping[str, Any],
    launcher_process_identity: Mapping[str, Any],
    created_at_utc: str,
) -> dict[str, Any]:
    """Build the sole CREATE_NEW reservation before creating the terminal child."""

    canonical_capsule = canonical_original_confirmatory_execution_capsule(capsule)
    release = canonical_original_confirmatory_terminal_client_launcher_release(
        launcher_release,
        capsule=canonical_capsule,
    )
    projection = canonical_original_confirmatory_terminal_client_launcher_projection(
        launcher_projection,
        launcher_release=release,
        capsule=canonical_capsule,
    )
    command = canonical_original_confirmatory_terminal_client_launcher_command(
        launcher_command,
        launcher_projection=projection,
        launcher_release=release,
        capsule=canonical_capsule,
        verify_terminal_command=verify_terminal_command,
    )
    process_identity = _canonical_e_process_identity(
        launcher_process_identity,
        role="terminal-client launch-intent launcher process identity",
    )
    if (
        process_identity["program_path"] != command["program_path"]
        or process_identity["program_sha256"] != command["program_sha256"]
        or process_identity["command_sha256"] != command["command_sha256"]
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "terminal-client launch-intent process identity differs from its launcher command"
        )
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "policy": TERMINAL_CLIENT_LAUNCH_INTENT_POLICY,
        "status": _TERMINAL_CLIENT_LAUNCH_INTENT_STATUS,
        "job_id": projection["job_id"],
        "supervisor_job_directory": projection["supervisor_job_directory"],
        "supervisor_spec_path": projection["supervisor_spec_path"],
        "supervisor_spec_sha256": command["supervisor_spec_sha256"],
        "e_intent_path": projection["e_intent_path"],
        "e_intent_file_sha256": command["e_intent_file_sha256"],
        "terminal_receipt_path": projection["terminal_receipt_path"],
        "terminal_receipt_sha256": command["terminal_receipt_sha256"],
        "launcher_command_sha256": command["command_sha256"],
        "verify_terminal_command_projection_sha256": projection[
            "verify_terminal_command_projection_sha256"
        ],
        "verify_terminal_command_sha256": command["verify_terminal_command_sha256"],
        "verify_terminal_environment_sha256": projection[
            "verify_terminal_environment_sha256"
        ],
        "verify_terminal_cwd": projection["verify_terminal_cwd"],
        "launcher_process_identity": process_identity,
        "launch_attempt_count": 1,
        "create_disposition": "CREATE_NEW",
        "child_process_created_before_intent": False,
        "existing_or_partial_intent_is_stop": True,
        "automatic_retry_allowed": False,
        "created_at_utc": _utc_timestamp(
            created_at_utc,
            role="terminal-client launch-intent creation time",
        ),
    }
    return {
        **unsigned,
        "intent_root_sha256": canonical_json_sha256(unsigned),
    }


def canonical_original_confirmatory_terminal_client_launch_intent(
    value: Mapping[str, Any],
    *,
    launcher_command: Mapping[str, Any],
    launcher_projection: Mapping[str, Any],
    launcher_release: Mapping[str, Any],
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    verify_terminal_command: OriginalConfirmatoryCapsuleCommand | Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, role="terminal-client launch intent")
    expected = build_original_confirmatory_terminal_client_launch_intent(
        launcher_command=launcher_command,
        launcher_projection=launcher_projection,
        launcher_release=launcher_release,
        capsule=capsule,
        verify_terminal_command=verify_terminal_command,
        launcher_process_identity=_mapping(
            raw.get("launcher_process_identity"),
            role="terminal-client launch-intent launcher process identity",
        ),
        created_at_utc=cast(str, raw.get("created_at_utc")),
    )
    if set(
        raw
    ) != _TERMINAL_CLIENT_LAUNCH_INTENT_FIELDS or not _strict_json_value_equal(
        raw, expected
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "terminal-client launch intent violates its exact CREATE_NEW policy"
        )
    return expected


_TERMINAL_CUSTODY_AUTHORITY_TEMPLATE_FIELDS = {
    "schema_version",
    "policy",
    "message_sequence",
    "message_contracts",
    "readback_contract",
    "process_identity_contract",
    "claim_identity_contract",
    "duplicate_handle_contract",
    "transport_contract",
    "consumed_e_binding_fields",
    "outcome_blind_expected_artifact_projection",
    "terminal_client_launcher_contract",
    "codex_terminal_wake_prompt_contract",
    "execution_control_contract",
    "protocol_invariants",
    "template_root_sha256",
}

_TERMINAL_CUSTODY_AUTHORITY_PROJECTION_FIELDS = {
    "schema_version",
    "policy",
    "terminal_custody_authority_template_root_sha256",
    "outcome_blind_expected_artifact_instance",
    "terminal_client_launcher_projection",
    "terminal_client_launcher_projection_root_sha256",
    "projection_root_sha256",
}


def _terminal_custody_authority_template_unsigned() -> dict[str, Any]:
    artifact_template = (
        build_original_confirmatory_outcome_blind_expected_artifact_projection()
    )
    wake_prompt = (
        build_original_confirmatory_codex_terminal_wake_prompt_template_projection()
    )
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "policy": TERMINAL_CUSTODY_AUTHORITY_TEMPLATE_POLICY,
        "message_sequence": [
            CLAIM_READY_MESSAGE_TYPE,
            CUSTODY_GRANT_MESSAGE_TYPE,
            COMPOSED_READY_MESSAGE_TYPE,
            FINAL_ACK_MESSAGE_TYPE,
        ],
        "message_contracts": {
            CLAIM_READY_MESSAGE_TYPE: {
                "producer": "capsule-terminal",
                "consumer": "supervisor",
                "policy": COMPOSED_CLAIM_READY_POLICY,
                "message_type": CLAIM_READY_MESSAGE_TYPE,
                "field_names": sorted(_TERMINAL_CLAIM_READY_FIELDS),
                "integrity_field": "claim_ready_sha256",
            },
            CUSTODY_GRANT_MESSAGE_TYPE: {
                "producer": "supervisor",
                "consumer": "capsule-terminal",
                "policy": COMPOSED_CUSTODY_GRANT_POLICY,
                "message_type": CUSTODY_GRANT_MESSAGE_TYPE,
                "field_names": sorted(_TERMINAL_CUSTODY_GRANT_FIELDS),
                "integrity_field": "custody_grant_sha256",
            },
            COMPOSED_READY_MESSAGE_TYPE: {
                "producer": "capsule-terminal",
                "consumer": "supervisor",
                "policy": COMPOSED_READY_POLICY,
                "message_type": COMPOSED_READY_MESSAGE_TYPE,
                "field_names": sorted(_TERMINAL_COMPOSED_READY_FIELDS),
                "integrity_field": "composed_ready_sha256",
            },
            FINAL_ACK_MESSAGE_TYPE: {
                "producer": "supervisor",
                "consumer": "capsule-terminal",
                "policy": COMPOSED_FINAL_ACK_POLICY,
                "message_type": FINAL_ACK_MESSAGE_TYPE,
                "field_names": sorted(_TERMINAL_FINAL_ACK_FIELDS),
                "integrity_field": "final_ack_sha256",
            },
        },
        "readback_contract": {
            "producer": "supervisor",
            "consumer": "capsule-terminal",
            "policy": POSTWAKE_COMPOSED_READBACK_RECEIPT_POLICY,
            "field_names": sorted(_TERMINAL_POSTWAKE_COMPOSED_READBACK_FIELDS),
            "integrity_field": "receipt_root_sha256",
            "create_disposition": "CREATE_NEW",
            "same_held_handle_required": True,
            "delete_or_replace_allowed": False,
        },
        "process_identity_contract": {
            "field_names": sorted(_TERMINAL_PROCESS_IDENTITY_FIELDS),
            "pid_reuse_protection_required": True,
            "creation_time_100ns_required": True,
            "creation_time_utc_required": True,
            "program_path_and_sha256_required": True,
            "command_sha256_required": True,
            "claim_immediate_venv_redirector_pid_required": True,
            "claim_immediate_venv_redirector_process_identity_required": True,
            "claim_terminal_client_launcher_process_identity_required": True,
            "claim_terminal_client_launch_intent_deterministic_path_and_policy_required": True,
            "claim_terminal_client_launch_intent_unread_required": True,
            "claim_terminal_client_launch_intent_hash_or_identity_allowed": False,
            "grant_independent_terminal_client_launch_intent_verification_required": True,
            "grant_terminal_client_launch_intent_launcher_identity_verification_required": True,
            "grant_terminal_client_launch_intent_create_new_before_child_verification_required": True,
            "grant_terminal_client_launch_intent_supervisor_handle_retention_required": True,
            "grant_terminal_client_launch_intent_supervisor_access_mask": (
                FILE_GENERIC_READ_ACCESS_MASK
            ),
            "grant_terminal_client_launch_intent_child_duplicate_target_access_mask": (
                GENERIC_READ_ACCESS_REQUEST
            ),
            "grant_terminal_client_launch_intent_child_expected_granted_access_mask": (
                FILE_GENERIC_READ_ACCESS_MASK
            ),
            "grant_terminal_client_launch_intent_child_duplicate_options": 0,
            "grant_terminal_client_launch_intent_child_duplicate_close_source": False,
            "grant_requires_terminal_client_launch_intent_child_open_only_after_grant": True,
            "composed_ready_terminal_client_launch_intent_binding_and_dual_custody_required": True,
            "composed_ready_terminal_client_launch_intent_postgrant_child_handle_slot_required": True,
            "composed_ready_terminal_client_launch_intent_physical_identity_exact_match_required": True,
            "readback_terminal_client_launch_intent_binding_and_supervisor_custody_required": True,
            "final_ack_terminal_client_launch_intent_identity_mirror_required": True,
            "final_ack_terminal_client_launch_intent_supervisor_custody_required": True,
            "live_launcher_redirector_child_grandparent_chain_required": True,
            "launcher_redirector_child_same_supervisor_job_required": True,
            "launcher_redirector_child_process_handles_retained_through_final_ack_required": True,
            "immediate_venv_redirector_process_identity_reverified_at_final_ack_required": True,
            "terminal_client_launcher_process_identity_reverified_at_final_ack_required": True,
            "launcher_redirector_child_grandparent_chain_reverified_at_final_ack_required": True,
            "launcher_redirector_child_same_supervisor_job_reverified_at_final_ack_required": True,
            "immediate_venv_redirector_live_at_final_ack_required": True,
            "terminal_client_launcher_live_at_final_ack_required": True,
            "child_peb_command_line_cwd_and_environment_readback_required": True,
        },
        "claim_identity_contract": {
            "policy": COMPOSED_CLAIM_PHYSICAL_IDENTITY_POLICY,
            "role": "composed-terminal",
            "field_names": sorted(_TERMINAL_CLAIM_PHYSICAL_IDENTITY_FIELDS),
            "stable_field_names": sorted(
                _TERMINAL_CLAIM_STABLE_PHYSICAL_IDENTITY_FIELDS
            ),
            "required_size_bytes": 0,
            "required_sha256": hashlib.sha256(b"").hexdigest(),
            "creator_desired_access_mask": (
                GENERIC_READ_ACCESS_REQUEST | GENERIC_WRITE_ACCESS_REQUEST
            ),
            "creator_share_mode": FILE_SHARE_READ,
            "creator_share_access": ["FILE_SHARE_READ"],
            "delete_access_allowed": False,
            "delete_share_allowed": False,
            "delete_on_close_allowed": False,
            "cleanup_allowed": False,
            "create_disposition": "CREATE_NEW",
            "claim_before_terminal_input_read_required": True,
            "same_held_handle_through_final_ack_required": True,
        },
        "duplicate_handle_contract": {
            "requested_target_access_mask": GENERIC_READ_ACCESS_REQUEST,
            "expected_mapped_granted_access_mask": FILE_GENERIC_READ_ACCESS_MASK,
            "duplicate_options": 0,
            "close_source": False,
            "supervisor_read_only_access_required": True,
            "source_handle_retained_through_final_ack_required": True,
        },
        "transport_contract": {
            "policy": POSTWAKE_CUSTODY_HANDSHAKE_CONTRACT_POLICY,
            "pipe_name_derivation_policy": POSTWAKE_CUSTODY_PIPE_DERIVATION_POLICY,
            "pipe_name_template": (
                "\\\\.\\pipe\\AANCA-composed-custody-{postwake_custody_seed_sha256}"
            ),
            "pipe_name_seed_binding": "postwake_custody_seed.seed_sha256",
            "concrete_pipe_name_or_seed_in_e_allowed": False,
            "outbound_message_types": [
                CLAIM_READY_MESSAGE_TYPE,
                COMPOSED_READY_MESSAGE_TYPE,
            ],
            "inbound_message_types": [
                CUSTODY_GRANT_MESSAGE_TYPE,
                FINAL_ACK_MESSAGE_TYPE,
            ],
            "outbound_max_bytes": TERMINAL_CUSTODY_OUTBOUND_MAX_BYTES,
            "inbound_max_bytes": TERMINAL_CUSTODY_INBOUND_MAX_BYTES,
            "shared_deadline_policy": POSTWAKE_SHARED_DEADLINE_POLICY,
            "terminal_client_arrival_timeout_ms": TERMINAL_CLIENT_ARRIVAL_TIMEOUT_MS,
            "terminal_client_arrival_deadline_scope": (
                "successful_codex_resume_to_accepted_claim_ready_v1"
            ),
            "custody_exchange_timeout_ms": CUSTODY_EXCHANGE_TIMEOUT_MS,
            "custody_exchange_deadline_scope": (
                "accepted_claim_ready_through_final_ack_v1"
            ),
            "overall_timeout_max_ms": TERMINAL_CUSTODY_OVERALL_TIMEOUT_MAX_MS,
            "arrival_and_exchange_waits_event_driven_required": True,
            "one_shared_exchange_deadline_required": True,
            "direction_specific_message_bounds_required": True,
            "local_single_client_required": True,
            "automatic_retry_allowed": False,
        },
        "consumed_e_binding_fields": [
            "e_consumption_claim_path",
            "e_consumption_claim_file_sha256",
            "e_consumption_claim_root_sha256",
        ],
        "outcome_blind_expected_artifact_projection": artifact_template,
        "terminal_client_launcher_contract": {
            "release_policy": TERMINAL_CLIENT_LAUNCHER_POLICY,
            "release_field_names": sorted(_TERMINAL_CLIENT_LAUNCHER_RELEASE_FIELDS),
            "e_projection_policy": TERMINAL_CLIENT_LAUNCHER_E_PROJECTION_POLICY,
            "e_projection_field_names": sorted(
                _TERMINAL_CLIENT_LAUNCHER_E_PROJECTION_FIELDS
            ),
            "final_command_policy": TERMINAL_CLIENT_LAUNCHER_COMMAND_POLICY,
            "final_command_field_names": sorted(
                _TERMINAL_CLIENT_LAUNCHER_COMMAND_FIELDS
            ),
            "launch_intent_policy": TERMINAL_CLIENT_LAUNCH_INTENT_POLICY,
            "launch_intent_filename": TERMINAL_CLIENT_LAUNCH_INTENT_FILENAME,
            "launch_intent_field_names": sorted(_TERMINAL_CLIENT_LAUNCH_INTENT_FIELDS),
            "launch_intent_status": _TERMINAL_CLIENT_LAUNCH_INTENT_STATUS,
            "launch_intent_create_disposition": "CREATE_NEW",
            "launch_intent_created_before_child_process_required": True,
            "existing_or_partial_launch_intent_is_stop": True,
            "launch_intent_physical_identity_policy": (
                NO_FOLLOW_PHYSICAL_FILE_IDENTITY_POLICY
            ),
            "launch_intent_physical_identity_role": "terminal-client-launch-intent",
            "launch_intent_physical_identity_field_names": sorted(
                _PHYSICAL_FILE_IDENTITY_FIELDS
            ),
            "launch_intent_write_through_fsync_readonly_required": True,
            "launch_intent_creator_desired_access_mask": (
                GENERIC_READ_ACCESS_REQUEST | GENERIC_WRITE_ACCESS_REQUEST
            ),
            "launch_intent_creator_share_access": ["FILE_SHARE_READ"],
            "launch_intent_delete_access_allowed": False,
            "launch_intent_delete_share_allowed": False,
            "launch_intent_cleanup_allowed": False,
            "launch_intent_same_handle_write_fsync_readback_required": True,
            "launch_intent_set_readonly_before_child_create_required": True,
            "launch_intent_creator_handle_retained_through_terminal_child_waitforexit_required": True,
            "terminal_child_launch_intent_no_follow_readonly_single_link_no_ads_required": True,
            "terminal_child_launch_intent_handle_retained_through_final_ack_required": True,
            "supervisor_launch_intent_handle_retained_through_final_ack_required": True,
            "claim_binds_launch_intent_deterministic_path_and_policy_only": True,
            "claim_launch_intent_read_must_be_false": True,
            "grant_required_before_terminal_child_launch_intent_read": True,
            "grant_independently_verifies_launch_intent_and_launcher_identity": True,
            "launch_intent_supervisor_granted_access_mask": FILE_GENERIC_READ_ACCESS_MASK,
            "launch_intent_child_duplicate_target_access_mask": (
                GENERIC_READ_ACCESS_REQUEST
            ),
            "launch_intent_child_expected_granted_access_mask": (
                FILE_GENERIC_READ_ACCESS_MASK
            ),
            "launch_intent_child_duplicate_options": 0,
            "launch_intent_child_duplicate_close_source": False,
            "child_stdio_policy": TERMINAL_CLIENT_CHILD_STDIO_POLICY,
            "child_stdin_source": "fresh_readonly_NUL_handle_v1",
            "child_stdout_transport": "fresh_anonymous_pipe_v1",
            "child_stderr_transport": "fresh_anonymous_pipe_v1",
            "child_inherited_handle_list_policy": (
                "exact_stdin_stdout_stderr_only_proc_thread_attribute_handle_list_v1"
            ),
            "child_inherited_handle_count": 3,
            "createprocess_inherit_handles": True,
            "startupinfoex_use_std_handles_required": True,
            "proc_thread_attribute_handle_list_required": True,
            "non_stdio_inherited_handles_allowed": False,
            "preterminal_or_terminal_input_file_handles_inherited_allowed": False,
            "child_stdout_max_bytes": TERMINAL_CUSTODY_OUTBOUND_MAX_BYTES,
            "child_stderr_max_bytes": TERMINAL_CUSTODY_OUTBOUND_MAX_BYTES,
            "child_stdout_summary_policy": COMPOSED_TERMINAL_STDOUT_SUMMARY_POLICY,
            "child_stdout_single_canonical_json_line_required": True,
            "child_stderr_empty_required": True,
            "stdio_pipe_drains_event_driven_concurrent_required": True,
            "launcher_forwards_validated_child_stdout_once_required": True,
            "supervisor_retains_launcher_redirector_child_process_handles_through_final_ack_required": True,
            "immediate_venv_redirector_live_through_runtime_child_exit_required": True,
            "terminal_client_launcher_live_through_redirector_waitforexit_required": True,
            "process_liveness_reverified_at_final_ack_required": True,
            "python_isolated_flags": list(TERMINAL_CLIENT_PYTHON_ISOLATED_FLAGS),
            "final_argument_order": list(
                _TERMINAL_CLIENT_LAUNCHER_FINAL_ARGUMENT_ORDER
            ),
            "downstream_placeholder_order": list(
                _TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDER_ORDER
            ),
            "q_contains_static_release_only": True,
            "e_contains_closed_derivation_only": True,
            "final_command_after_e_spec_and_terminal_receipt_only": True,
            "wake_intent_hash_in_command_allowed": False,
            "direct_codex_to_verify_terminal_allowed": False,
            "automatic_retry_allowed": False,
        },
        "codex_terminal_wake_prompt_contract": {
            "render_policy": wake_prompt["policy"],
            "template_sha256": wake_prompt["template_sha256"],
            "template_projection_root_sha256": wake_prompt["projection_root_sha256"],
            "placeholder_order": wake_prompt["placeholder_order"],
            "final_render_input_fields": [
                "job_id",
                "supervisor_job_directory",
                "supervisor_spec_path",
                "supervisor_spec_sha256",
                "terminal_receipt_sha256",
                "terminal_client_launcher_argv_json",
                "terminal_client_launcher_command_sha256",
                "verify_terminal_command_sha256",
            ],
            "render_before_final_spec_and_terminal_receipt_allowed": False,
            "concrete_rendered_prompt_in_q_or_e_allowed": False,
            "job_or_command_discovery_allowed": False,
            "automatic_retry_allowed": False,
        },
        "execution_control_contract": {
            "allowed_capsule_modes": list(CAPSULE_ALLOWED_MODES),
            "fourth_execution_mode_allowed": False,
            "automatic_retry_allowed": False,
            "training_or_model_selection_allowed": False,
            "selection_or_tuning_allowed": False,
            "scientific_publication_allowed": False,
            "semantic_outcome_read_scope": SEMANTIC_OUTCOME_READ_SCOPE,
            "outcome_values_read": False,
            "outcome_values_emitted": False,
            "outcome_values_used_for_selection_or_tuning": False,
        },
        "protocol_invariants": [
            "claim-created-before-terminal-input-read",
            "event-driven-arrival-deadline-before-claim",
            "one-shared-exchange-deadline-after-claim",
            "direction-specific-message-bounds",
            "exact-process-identity-and-pid-reuse-protection",
            "live-launcher-redirector-child-grandparent-chain-and-same-job",
            "sealed-launcher-create-new-intent-at-most-once",
            "exact-read-only-duplicate-access",
            "no-delete-no-replace-no-cleanup",
            "P-to-T-to-L-to-C-to-R",
            "zero-automatic-retry",
            "zero-selection-or-tuning",
            "no-fourth-execution-mode",
        ],
    }
    return unsigned


def build_original_confirmatory_terminal_custody_authority_template_projection() -> (
    dict[str, Any]
):
    """Build Q's full static terminal custody protocol template."""

    unsigned = _terminal_custody_authority_template_unsigned()
    return {
        **unsigned,
        "template_root_sha256": canonical_json_sha256(unsigned),
    }


def canonical_original_confirmatory_terminal_custody_authority_template_projection(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, role="terminal custody authority template")
    expected = (
        build_original_confirmatory_terminal_custody_authority_template_projection()
    )
    if set(
        raw
    ) != _TERMINAL_CUSTODY_AUTHORITY_TEMPLATE_FIELDS or not _strict_json_value_equal(
        raw, expected
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "terminal custody authority template violates its exact policy"
        )
    return expected


def build_original_confirmatory_terminal_custody_authority_projection(
    *,
    run_id: str,
    expected_run_directory: str | Path,
    launcher_release: Mapping[str, Any],
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    supervisor_job_id: str,
    supervisor_job_directory: str | Path,
    verify_terminal_command_projection_sha256: str,
    verify_terminal_environment_sha256: str,
    verify_terminal_cwd: str | Path,
) -> dict[str, Any]:
    """Build E's compact acyclic per-run terminal authority instance."""

    template = (
        build_original_confirmatory_terminal_custody_authority_template_projection()
    )
    artifact_instance = (
        build_original_confirmatory_outcome_blind_expected_artifact_instance(
            run_id=run_id,
            expected_run_directory=expected_run_directory,
        )
    )
    launcher_projection = (
        build_original_confirmatory_terminal_client_launcher_projection(
            launcher_release=launcher_release,
            capsule=capsule,
            job_id=supervisor_job_id,
            supervisor_job_directory=supervisor_job_directory,
            verify_terminal_command_projection_sha256=(
                verify_terminal_command_projection_sha256
            ),
            verify_terminal_environment_sha256=verify_terminal_environment_sha256,
            verify_terminal_cwd=verify_terminal_cwd,
        )
    )
    unsigned = {
        "schema_version": 1,
        "policy": TERMINAL_CUSTODY_AUTHORITY_PROJECTION_POLICY,
        "terminal_custody_authority_template_root_sha256": template[
            "template_root_sha256"
        ],
        "outcome_blind_expected_artifact_instance": artifact_instance,
        "terminal_client_launcher_projection": launcher_projection,
        "terminal_client_launcher_projection_root_sha256": launcher_projection[
            "projection_root_sha256"
        ],
    }
    return {
        **unsigned,
        "projection_root_sha256": canonical_json_sha256(unsigned),
    }


def canonical_original_confirmatory_terminal_custody_authority_projection_self_contained(
    value: Mapping[str, Any],
    *,
    run_id: str,
    expected_run_directory: str | Path,
    launcher_release: Mapping[str, Any],
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    supervisor_job_id: str,
    supervisor_job_directory: str | Path,
    verify_terminal_command_projection_sha256: str,
    verify_terminal_environment_sha256: str,
    verify_terminal_cwd: str | Path,
) -> dict[str, Any]:
    """Validate a sealed E projection before its downstream spec is imported."""

    raw = _mapping(value, role="terminal custody authority projection")
    expected = build_original_confirmatory_terminal_custody_authority_projection(
        run_id=run_id,
        expected_run_directory=expected_run_directory,
        launcher_release=launcher_release,
        capsule=capsule,
        supervisor_job_id=supervisor_job_id,
        supervisor_job_directory=supervisor_job_directory,
        verify_terminal_command_projection_sha256=(
            verify_terminal_command_projection_sha256
        ),
        verify_terminal_environment_sha256=verify_terminal_environment_sha256,
        verify_terminal_cwd=verify_terminal_cwd,
    )
    if set(
        raw
    ) != _TERMINAL_CUSTODY_AUTHORITY_PROJECTION_FIELDS or not _strict_json_value_equal(
        raw, expected
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "terminal custody authority projection violates its exact sealed-E policy"
        )
    return expected


def canonical_original_confirmatory_terminal_custody_authority_projection(
    value: Mapping[str, Any],
    *,
    run_id: str,
    expected_run_directory: str | Path,
    custody_contract: OriginalConfirmatoryPostwakeCustodyHandshakeContract,
    launcher_release: Mapping[str, Any],
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    supervisor_job_id: str,
    supervisor_job_directory: str | Path,
    verify_terminal_command_projection_sha256: str,
    verify_terminal_environment_sha256: str,
    verify_terminal_cwd: str | Path,
) -> dict[str, Any]:
    raw = _mapping(value, role="terminal custody authority projection")
    expected = build_original_confirmatory_terminal_custody_authority_projection(
        run_id=run_id,
        expected_run_directory=expected_run_directory,
        launcher_release=launcher_release,
        capsule=capsule,
        supervisor_job_id=supervisor_job_id,
        supervisor_job_directory=supervisor_job_directory,
        verify_terminal_command_projection_sha256=(
            verify_terminal_command_projection_sha256
        ),
        verify_terminal_environment_sha256=verify_terminal_environment_sha256,
        verify_terminal_cwd=verify_terminal_cwd,
    )
    if (
        set(raw) != _TERMINAL_CUSTODY_AUTHORITY_PROJECTION_FIELDS
        or not _strict_json_value_equal(raw, expected)
        or custody_contract.ready_max_bytes != TERMINAL_CUSTODY_OUTBOUND_MAX_BYTES
        or custody_contract.ack_max_bytes != TERMINAL_CUSTODY_INBOUND_MAX_BYTES
        or custody_contract.terminal_client_arrival_timeout_ms
        != TERMINAL_CLIENT_ARRIVAL_TIMEOUT_MS
        or custody_contract.custody_exchange_timeout_ms != CUSTODY_EXCHANGE_TIMEOUT_MS
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "terminal custody authority projection violates its exact policy"
        )
    return expected


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryTerminalCompositionContract:
    """E's exact postwake verify-terminal input/output contract."""

    capsule_contract_sha256: str
    capsule_path: Path
    capsule_sha256: str
    capsule_internal_manifest_sha256: str
    capsule_mode: str
    verifier_command: dict[str, Any]
    verifier_command_sha256: str
    preterminal_pin_contract_sha256: str
    preterminal_overlap_handshake_contract_sha256: str
    postwake_input_lease_contract_sha256: str
    postwake_custody_seed_sha256: str
    postwake_custody_handshake_contract_sha256: str
    terminal_custody_authority_projection: dict[str, Any]
    expected_environment_envelope_sha256: str
    process_environment_binding_sha256: str
    exact_integrity_verifier_environment_sha256: str
    capsule_lease_identity_root_sha256: str
    capsule_ancestor_lease_root_sha256: str
    supervisor_terminal_receipt_path: Path
    supervisor_terminal_receipt_max_bytes: int
    verifier_stdout_path: Path
    verifier_stdout_max_bytes: int
    verifier_stderr_path: Path
    verifier_stderr_max_bytes: int
    preterminal_pin_receipt_path: Path
    preterminal_pin_receipt_max_bytes: int
    postwake_input_lease_receipt_path: Path
    postwake_input_lease_receipt_max_bytes: int
    postwake_composed_readback_receipt_path: Path
    postwake_composed_readback_receipt_max_bytes: int
    composed_terminal_receipt_path: Path
    composed_terminal_receipt_max_bytes: int
    semantic_outcome_read_scope: str
    contract_sha256: str

    def payload_without_self_hash(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": TERMINAL_COMPOSITION_CONTRACT_POLICY,
            "capsule_contract_sha256": self.capsule_contract_sha256,
            "capsule_path": str(self.capsule_path),
            "capsule_sha256": self.capsule_sha256,
            "capsule_internal_manifest_sha256": (self.capsule_internal_manifest_sha256),
            "capsule_mode": self.capsule_mode,
            "verifier_command": self.verifier_command,
            "verifier_command_sha256": self.verifier_command_sha256,
            "preterminal_pin_contract_sha256": (self.preterminal_pin_contract_sha256),
            "preterminal_overlap_handshake_contract_sha256": (
                self.preterminal_overlap_handshake_contract_sha256
            ),
            "postwake_input_lease_contract_sha256": (
                self.postwake_input_lease_contract_sha256
            ),
            "postwake_custody_seed_sha256": (self.postwake_custody_seed_sha256),
            "postwake_custody_handshake_contract_sha256": (
                self.postwake_custody_handshake_contract_sha256
            ),
            "terminal_custody_authority_projection": (
                self.terminal_custody_authority_projection
            ),
            "expected_environment_envelope_sha256": (
                self.expected_environment_envelope_sha256
            ),
            "process_environment_binding_sha256": (
                self.process_environment_binding_sha256
            ),
            "exact_integrity_verifier_environment_sha256": (
                self.exact_integrity_verifier_environment_sha256
            ),
            "capsule_lease_identity_root_sha256": (
                self.capsule_lease_identity_root_sha256
            ),
            "capsule_ancestor_lease_root_sha256": (
                self.capsule_ancestor_lease_root_sha256
            ),
            "supervisor_terminal_receipt_path": str(
                self.supervisor_terminal_receipt_path
            ),
            "supervisor_terminal_receipt_max_bytes": (
                self.supervisor_terminal_receipt_max_bytes
            ),
            "supervisor_terminal_envelope_schema_version": 2,
            "supervisor_terminal_payload_schema_version": 2,
            "supervisor_terminal_payload_policy": SUPERVISOR_V3_POLICY,
            "supervisor_terminal_success_terminal_kind": (
                SUPERVISOR_V2_SUCCESS_TERMINAL_KIND
            ),
            "supervisor_terminal_success_reason": SUPERVISOR_V2_SUCCESS_REASON,
            "supervisor_terminal_success_exit_code": 0,
            "supervisor_terminal_hash_policy": SUPERVISOR_TERMINAL_HASH_POLICY,
            "verifier_stdout_path": str(self.verifier_stdout_path),
            "verifier_stdout_max_bytes": self.verifier_stdout_max_bytes,
            "verifier_stderr_path": str(self.verifier_stderr_path),
            "verifier_stderr_max_bytes": self.verifier_stderr_max_bytes,
            "preterminal_pin_receipt_path": str(self.preterminal_pin_receipt_path),
            "preterminal_pin_receipt_max_bytes": (
                self.preterminal_pin_receipt_max_bytes
            ),
            "postwake_input_lease_receipt_path": str(
                self.postwake_input_lease_receipt_path
            ),
            "postwake_input_lease_receipt_max_bytes": (
                self.postwake_input_lease_receipt_max_bytes
            ),
            "postwake_composed_readback_receipt_path": str(
                self.postwake_composed_readback_receipt_path
            ),
            "postwake_composed_readback_receipt_max_bytes": (
                self.postwake_composed_readback_receipt_max_bytes
            ),
            "composed_terminal_receipt_path": str(self.composed_terminal_receipt_path),
            "composed_terminal_receipt_max_bytes": (
                self.composed_terminal_receipt_max_bytes
            ),
            "composed_output_claim_policy": (COMPOSED_TERMINAL_ONE_USE_CLAIM_POLICY),
            "composed_output_claim_disposition": (
                COMPOSED_TERMINAL_ONE_USE_CLAIM_DISPOSITION
            ),
            "composed_output_claim_before_input_read_required": True,
            "composed_output_readonly_at_create_required": True,
            "composed_output_same_handle_write_required": True,
            "composed_output_cleanup_allowed": False,
            "composed_output_retry_allowed": False,
            "supervisor_custody_required": True,
            "supervisor_custody_policy": (COMPOSED_TERMINAL_SUPERVISOR_CUSTODY_POLICY),
            "stdout_summary_policy": COMPOSED_TERMINAL_STDOUT_SUMMARY_POLICY,
            "stdout_single_canonical_json_line_required": True,
            "stderr_empty_required": True,
            "semantic_outcome_read_scope": self.semantic_outcome_read_scope,
            "outcome_values_read": False,
            "outcome_values_emitted": False,
            "outcome_values_used_for_selection_or_tuning": False,
            "training_or_model_selection_allowed": False,
            "scientific_publication_allowed": False,
            "automatic_retry_allowed": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.payload_without_self_hash(),
            "contract_sha256": self.contract_sha256,
        }


_TERMINAL_COMPOSITION_CONTRACT_FIELDS = {
    "schema_version",
    "policy",
    "capsule_contract_sha256",
    "capsule_path",
    "capsule_sha256",
    "capsule_internal_manifest_sha256",
    "capsule_mode",
    "verifier_command",
    "verifier_command_sha256",
    "preterminal_pin_contract_sha256",
    "preterminal_overlap_handshake_contract_sha256",
    "postwake_input_lease_contract_sha256",
    "postwake_custody_seed_sha256",
    "postwake_custody_handshake_contract_sha256",
    "terminal_custody_authority_projection",
    "expected_environment_envelope_sha256",
    "process_environment_binding_sha256",
    "exact_integrity_verifier_environment_sha256",
    "capsule_lease_identity_root_sha256",
    "capsule_ancestor_lease_root_sha256",
    "supervisor_terminal_receipt_path",
    "supervisor_terminal_receipt_max_bytes",
    "supervisor_terminal_envelope_schema_version",
    "supervisor_terminal_payload_schema_version",
    "supervisor_terminal_payload_policy",
    "supervisor_terminal_success_terminal_kind",
    "supervisor_terminal_success_reason",
    "supervisor_terminal_success_exit_code",
    "supervisor_terminal_hash_policy",
    "verifier_stdout_path",
    "verifier_stdout_max_bytes",
    "verifier_stderr_path",
    "verifier_stderr_max_bytes",
    "preterminal_pin_receipt_path",
    "preterminal_pin_receipt_max_bytes",
    "postwake_input_lease_receipt_path",
    "postwake_input_lease_receipt_max_bytes",
    "postwake_composed_readback_receipt_path",
    "postwake_composed_readback_receipt_max_bytes",
    "composed_terminal_receipt_path",
    "composed_terminal_receipt_max_bytes",
    "composed_output_claim_policy",
    "composed_output_claim_disposition",
    "composed_output_claim_before_input_read_required",
    "composed_output_readonly_at_create_required",
    "composed_output_same_handle_write_required",
    "composed_output_cleanup_allowed",
    "composed_output_retry_allowed",
    "supervisor_custody_required",
    "supervisor_custody_policy",
    "stdout_summary_policy",
    "stdout_single_canonical_json_line_required",
    "stderr_empty_required",
    "semantic_outcome_read_scope",
    "outcome_values_read",
    "outcome_values_emitted",
    "outcome_values_used_for_selection_or_tuning",
    "training_or_model_selection_allowed",
    "scientific_publication_allowed",
    "automatic_retry_allowed",
    "contract_sha256",
}


def canonical_original_confirmatory_terminal_composition_contract(
    value: OriginalConfirmatoryTerminalCompositionContract | Mapping[str, Any],
    *,
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    verifier_command: OriginalConfirmatoryCapsuleCommand | Mapping[str, Any],
    verifier_command_tail_argv: Sequence[str],
    preterminal_pin_contract_sha256: str,
    preterminal_overlap_handshake_contract: (
        OriginalConfirmatoryPreterminalOverlapHandshakeContract | Mapping[str, Any]
    ),
    postwake_input_lease_contract: OriginalConfirmatoryPostwakeInputLeaseContract
    | Mapping[str, Any],
    postwake_custody_seed: OriginalConfirmatoryPostwakeCustodySeed | Mapping[str, Any],
    postwake_custody_handshake_contract: (
        OriginalConfirmatoryPostwakeCustodyHandshakeContract | Mapping[str, Any]
    ),
    expected_run_directory: str | Path,
    expected_terminal_custody_authority_projection: Mapping[str, Any],
    terminal_client_launcher_release: Mapping[str, Any],
    expected_environment: ExpectedLaunchEnvironmentEnvelopeV1 | Mapping[str, Any],
    process_environment_binding: OriginalConfirmatoryProcessEnvironmentBinding
    | Mapping[str, Any],
) -> OriginalConfirmatoryTerminalCompositionContract:
    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatoryTerminalCompositionContract
        else _mapping(value, role="E terminal composition contract")
    )
    if set(raw) != _TERMINAL_COMPOSITION_CONTRACT_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E terminal composition contract has an unexpected field set"
        )
    canonical_capsule = canonical_original_confirmatory_execution_capsule(capsule)
    canonical_tail = canonical_original_confirmatory_capsule_mode_tail(
        capsule_mode=CAPSULE_TERMINAL_MODE,
        tail_argv=verifier_command_tail_argv,
    )
    canonical_command = canonical_original_confirmatory_capsule_command(
        verifier_command,
        capsule=canonical_capsule,
        expected_mode=CAPSULE_TERMINAL_MODE,
        expected_tail_argv=canonical_tail,
    )
    envelope = canonical_expected_launch_environment_envelope_v1(expected_environment)
    environment_binding = canonical_original_confirmatory_process_environment_binding(
        process_environment_binding,
        expected_environment=envelope,
    )
    handshake_contract = (
        canonical_original_confirmatory_preterminal_overlap_handshake_contract(
            preterminal_overlap_handshake_contract
        )
    )
    input_lease_contract = (
        canonical_original_confirmatory_postwake_input_lease_contract(
            postwake_input_lease_contract
        )
    )
    custody_seed = canonical_original_confirmatory_postwake_custody_seed(
        postwake_custody_seed
    )
    job_id = canonical_tail[canonical_tail.index("--supervisor-job-id") + 1]
    job_dir = Path(canonical_tail[canonical_tail.index("--supervisor-job-dir") + 1])
    custody_contract = (
        canonical_original_confirmatory_postwake_custody_handshake_contract(
            postwake_custody_handshake_contract,
            custody_seed=custody_seed,
        )
    )
    expected_projection_input = _mapping(
        expected_terminal_custody_authority_projection,
        role="expected terminal custody authority projection",
    )
    expected_launcher_projection_input = _mapping(
        expected_projection_input.get("terminal_client_launcher_projection"),
        role="expected terminal-client launcher projection",
    )
    concrete_terminal_command_projection = (
        _derive_original_confirmatory_terminal_command_projection_from_concrete(
            command=canonical_command,
            capsule=canonical_capsule,
            canonical_tail=canonical_tail,
        )
    )
    terminal_custody_projection = (
        canonical_original_confirmatory_terminal_custody_authority_projection(
            _mapping(
                raw["terminal_custody_authority_projection"],
                role="terminal composition custody authority projection",
            ),
            run_id=cast(str, custody_seed.payload["run_id"]),
            expected_run_directory=expected_run_directory,
            custody_contract=custody_contract,
            launcher_release=terminal_client_launcher_release,
            capsule=canonical_capsule,
            supervisor_job_id=job_id,
            supervisor_job_directory=job_dir,
            verify_terminal_command_projection_sha256=cast(
                str,
                expected_launcher_projection_input.get(
                    "verify_terminal_command_projection_sha256"
                ),
            ),
            verify_terminal_environment_sha256=(
                environment_binding.exact_integrity_verifier_environment_sha256
            ),
            verify_terminal_cwd=canonical_command.cwd,
        )
    )
    expected_terminal_custody_projection = (
        canonical_original_confirmatory_terminal_custody_authority_projection(
            expected_projection_input,
            run_id=cast(str, custody_seed.payload["run_id"]),
            expected_run_directory=expected_run_directory,
            custody_contract=custody_contract,
            launcher_release=terminal_client_launcher_release,
            capsule=canonical_capsule,
            supervisor_job_id=job_id,
            supervisor_job_directory=job_dir,
            verify_terminal_command_projection_sha256=cast(
                str,
                expected_launcher_projection_input.get(
                    "verify_terminal_command_projection_sha256"
                ),
            ),
            verify_terminal_environment_sha256=(
                environment_binding.exact_integrity_verifier_environment_sha256
            ),
            verify_terminal_cwd=canonical_command.cwd,
        )
    )
    path_for_flag = {
        flag: canonical_tail[canonical_tail.index(flag) + 1]
        for flag in CAPSULE_TERMINAL_SUFFIX_FLAGS
    }
    scope = raw["semantic_outcome_read_scope"]
    input_paths = (
        raw["supervisor_terminal_receipt_path"],
        raw["verifier_stdout_path"],
        raw["verifier_stderr_path"],
        raw["preterminal_pin_receipt_path"],
        raw["postwake_input_lease_receipt_path"],
        raw["postwake_composed_readback_receipt_path"],
        raw["composed_terminal_receipt_path"],
    )
    unsigned = {key: item for key, item in raw.items() if key != "contract_sha256"}
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != TERMINAL_COMPOSITION_CONTRACT_POLICY
        or raw["capsule_contract_sha256"] != canonical_capsule.contract_sha256
        or raw["capsule_path"] != str(canonical_capsule.path)
        or raw["capsule_sha256"] != canonical_capsule.sha256
        or raw["capsule_internal_manifest_sha256"]
        != canonical_capsule.internal_manifest_sha256
        or raw["capsule_mode"] != CAPSULE_TERMINAL_MODE
        or not _strict_json_value_equal(
            raw["verifier_command"],
            canonical_command.as_dict(),
        )
        or raw["verifier_command_sha256"] != canonical_command.command_sha256
        or raw["preterminal_pin_contract_sha256"]
        != _sha256(
            preterminal_pin_contract_sha256,
            role="terminal composition preterminal contract",
        )
        or raw["preterminal_overlap_handshake_contract_sha256"]
        != handshake_contract.contract_sha256
        or raw["postwake_input_lease_contract_sha256"]
        != input_lease_contract.contract_sha256
        or raw["postwake_custody_seed_sha256"] != custody_seed.seed_sha256
        or custody_seed.payload["supervisor_job_id"] != job_id
        or custody_seed.payload["supervisor_job_dir"] != str(job_dir)
        or custody_seed.payload["execution_capsule_contract_sha256"]
        != canonical_capsule.contract_sha256
        or custody_seed.payload["capsule_sha256"] != canonical_capsule.sha256
        or custody_seed.payload["terminal_release_root_sha256"]
        != canonical_capsule.terminal_release_root_sha256
        or input_lease_contract.supervisor_job_dir != job_dir
        or input_lease_contract.supervisor_job_ancestor_lease_contract.paths[-1]
        != job_dir
        or job_dir.name != job_id
        or raw["postwake_custody_handshake_contract_sha256"]
        != custody_contract.contract_sha256
        or not _strict_json_value_equal(
            raw["terminal_custody_authority_projection"],
            terminal_custody_projection,
        )
        or not _strict_json_value_equal(
            terminal_custody_projection,
            expected_terminal_custody_projection,
        )
        or concrete_terminal_command_projection.projection_sha256
        != expected_launcher_projection_input[
            "verify_terminal_command_projection_sha256"
        ]
        or raw["expected_environment_envelope_sha256"] != envelope.envelope_sha256
        or raw["process_environment_binding_sha256"]
        != environment_binding.binding_sha256
        or raw["exact_integrity_verifier_environment_sha256"]
        != environment_binding.exact_integrity_verifier_environment_sha256
        or raw["capsule_lease_identity_root_sha256"]
        != canonical_capsule.capsule_lease_identity_root_sha256
        or raw["capsule_ancestor_lease_root_sha256"]
        != canonical_capsule.capsule_ancestor_lease_root_sha256
        or raw["supervisor_terminal_receipt_path"]
        != path_for_flag["--supervisor-terminal"]
        or custody_seed.payload["supervisor_terminal_receipt_path"]
        != raw["supervisor_terminal_receipt_path"]
        or raw["verifier_stdout_path"] != path_for_flag["--verifier-stdout"]
        or raw["verifier_stderr_path"] != str(input_lease_contract.verifier_stderr_path)
        or raw["verifier_stdout_max_bytes"]
        != input_lease_contract.verifier_log_max_bytes
        or raw["verifier_stderr_max_bytes"]
        != input_lease_contract.verifier_log_max_bytes
        or raw["preterminal_pin_receipt_path"] != path_for_flag["--preterminal-pin"]
        or custody_seed.payload["preterminal_pin_receipt_path"]
        != raw["preterminal_pin_receipt_path"]
        or handshake_contract.handshake_receipt_path.parent != job_dir
        or handshake_contract.handshake_receipt_path.name
        != "preterminal_overlap_handshake.json"
        or raw["postwake_input_lease_receipt_path"]
        != str(input_lease_contract.lease_receipt_path)
        or custody_seed.payload["postwake_input_lease_receipt_path"]
        != raw["postwake_input_lease_receipt_path"]
        or raw["postwake_composed_readback_receipt_path"]
        != str(custody_contract.readback_receipt_path)
        or custody_seed.payload["postwake_composed_readback_receipt_path"]
        != raw["postwake_composed_readback_receipt_path"]
        or custody_contract.readback_receipt_path.parent != job_dir
        or custody_contract.readback_receipt_path.name
        != "postwake_composed_readback_receipt.json"
        or custody_contract.expected_composed_command_sha256
        != canonical_command.command_sha256
        or custody_contract.expected_composed_cwd != canonical_command.cwd
        or custody_contract.expected_composed_environment_sha256
        != environment_binding.exact_integrity_verifier_environment_sha256
        or raw["composed_terminal_receipt_path"] != path_for_flag["--composed-terminal"]
        or custody_seed.payload["composed_terminal_receipt_path"]
        != raw["composed_terminal_receipt_path"]
        or Path(cast(str, raw["composed_terminal_receipt_path"])).parent != job_dir
        or Path(cast(str, raw["composed_terminal_receipt_path"])).name
        != "composed_terminal.json"
        or raw["supervisor_terminal_receipt_path"]
        != str(input_lease_contract.terminal_receipt_path)
        or raw["verifier_stdout_path"] != str(input_lease_contract.verifier_stdout_path)
        or raw["preterminal_pin_receipt_path"]
        != str(input_lease_contract.preterminal_pin_path)
        or len(set(input_paths)) != len(input_paths)
        or raw["supervisor_terminal_envelope_schema_version"] != 2
        or type(raw["supervisor_terminal_envelope_schema_version"]) is not int
        or raw["supervisor_terminal_payload_schema_version"] != 2
        or type(raw["supervisor_terminal_payload_schema_version"]) is not int
        or raw["supervisor_terminal_payload_policy"] != SUPERVISOR_V3_POLICY
        or raw["supervisor_terminal_success_terminal_kind"]
        != SUPERVISOR_V2_SUCCESS_TERMINAL_KIND
        or raw["supervisor_terminal_success_reason"] != SUPERVISOR_V2_SUCCESS_REASON
        or raw["supervisor_terminal_success_exit_code"] != 0
        or type(raw["supervisor_terminal_success_exit_code"]) is not int
        or raw["supervisor_terminal_hash_policy"] != SUPERVISOR_TERMINAL_HASH_POLICY
        or raw["composed_output_claim_policy"] != COMPOSED_TERMINAL_ONE_USE_CLAIM_POLICY
        or raw["composed_output_claim_disposition"]
        != COMPOSED_TERMINAL_ONE_USE_CLAIM_DISPOSITION
        or raw["composed_output_claim_before_input_read_required"] is not True
        or raw["composed_output_readonly_at_create_required"] is not True
        or raw["composed_output_same_handle_write_required"] is not True
        or raw["composed_output_cleanup_allowed"] is not False
        or raw["composed_output_retry_allowed"] is not False
        or raw["supervisor_custody_required"] is not True
        or raw["supervisor_custody_policy"]
        != COMPOSED_TERMINAL_SUPERVISOR_CUSTODY_POLICY
        or raw["stdout_summary_policy"] != COMPOSED_TERMINAL_STDOUT_SUMMARY_POLICY
        or raw["stdout_single_canonical_json_line_required"] is not True
        or raw["stderr_empty_required"] is not True
        or scope != SEMANTIC_OUTCOME_READ_SCOPE
        or raw["outcome_values_read"] is not False
        or raw["outcome_values_emitted"] is not False
        or raw["outcome_values_used_for_selection_or_tuning"] is not False
        or raw["training_or_model_selection_allowed"] is not False
        or raw["scientific_publication_allowed"] is not False
        or raw["automatic_retry_allowed"] is not False
        or raw["contract_sha256"] != canonical_json_sha256(unsigned)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E terminal composition contract violates its exact closed policy"
        )
    return OriginalConfirmatoryTerminalCompositionContract(
        capsule_contract_sha256=canonical_capsule.contract_sha256,
        capsule_path=canonical_capsule.path,
        capsule_sha256=canonical_capsule.sha256,
        capsule_internal_manifest_sha256=(canonical_capsule.internal_manifest_sha256),
        capsule_mode=CAPSULE_TERMINAL_MODE,
        verifier_command=canonical_command.as_dict(),
        verifier_command_sha256=canonical_command.command_sha256,
        preterminal_pin_contract_sha256=raw["preterminal_pin_contract_sha256"],
        preterminal_overlap_handshake_contract_sha256=(
            handshake_contract.contract_sha256
        ),
        postwake_input_lease_contract_sha256=(input_lease_contract.contract_sha256),
        postwake_custody_seed_sha256=custody_seed.seed_sha256,
        postwake_custody_handshake_contract_sha256=(custody_contract.contract_sha256),
        terminal_custody_authority_projection=terminal_custody_projection,
        expected_environment_envelope_sha256=envelope.envelope_sha256,
        process_environment_binding_sha256=environment_binding.binding_sha256,
        exact_integrity_verifier_environment_sha256=(
            environment_binding.exact_integrity_verifier_environment_sha256
        ),
        capsule_lease_identity_root_sha256=(
            canonical_capsule.capsule_lease_identity_root_sha256
        ),
        capsule_ancestor_lease_root_sha256=(
            canonical_capsule.capsule_ancestor_lease_root_sha256
        ),
        supervisor_terminal_receipt_path=Path(
            _absolute_path(
                raw["supervisor_terminal_receipt_path"],
                role="supervisor terminal receipt path",
            )
        ),
        supervisor_terminal_receipt_max_bytes=_positive_int(
            raw["supervisor_terminal_receipt_max_bytes"],
            role="supervisor terminal receipt bound",
        ),
        verifier_stdout_path=Path(
            _absolute_path(
                raw["verifier_stdout_path"],
                role="preterminal verifier stdout path",
            )
        ),
        verifier_stdout_max_bytes=_positive_int(
            raw["verifier_stdout_max_bytes"],
            role="preterminal verifier stdout bound",
        ),
        verifier_stderr_path=Path(
            _absolute_path(
                raw["verifier_stderr_path"],
                role="preterminal verifier stderr path",
            )
        ),
        verifier_stderr_max_bytes=_positive_int(
            raw["verifier_stderr_max_bytes"],
            role="preterminal verifier stderr bound",
        ),
        preterminal_pin_receipt_path=Path(
            _absolute_path(
                raw["preterminal_pin_receipt_path"],
                role="preterminal pin receipt path",
            )
        ),
        preterminal_pin_receipt_max_bytes=_positive_int(
            raw["preterminal_pin_receipt_max_bytes"],
            role="preterminal pin receipt bound",
        ),
        postwake_input_lease_receipt_path=Path(
            _absolute_path(
                raw["postwake_input_lease_receipt_path"],
                role="postwake input lease receipt path",
            )
        ),
        postwake_input_lease_receipt_max_bytes=_positive_int(
            raw["postwake_input_lease_receipt_max_bytes"],
            role="postwake input lease receipt bound",
        ),
        postwake_composed_readback_receipt_path=Path(
            _absolute_path(
                raw["postwake_composed_readback_receipt_path"],
                role="postwake composed readback receipt path",
            )
        ),
        postwake_composed_readback_receipt_max_bytes=_positive_int(
            raw["postwake_composed_readback_receipt_max_bytes"],
            role="postwake composed readback receipt bound",
        ),
        composed_terminal_receipt_path=Path(
            _absolute_path(
                raw["composed_terminal_receipt_path"],
                role="composed terminal receipt path",
            )
        ),
        composed_terminal_receipt_max_bytes=_positive_int(
            raw["composed_terminal_receipt_max_bytes"],
            role="composed terminal receipt bound",
        ),
        semantic_outcome_read_scope=scope,
        contract_sha256=raw["contract_sha256"],
    )


def build_original_confirmatory_terminal_composition_contract(
    *,
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    verifier_command: OriginalConfirmatoryCapsuleCommand | Mapping[str, Any],
    verifier_command_tail_argv: Sequence[str],
    preterminal_pin_contract_sha256: str,
    preterminal_overlap_handshake_contract: (
        OriginalConfirmatoryPreterminalOverlapHandshakeContract | Mapping[str, Any]
    ),
    postwake_input_lease_contract: OriginalConfirmatoryPostwakeInputLeaseContract
    | Mapping[str, Any],
    postwake_custody_seed: OriginalConfirmatoryPostwakeCustodySeed | Mapping[str, Any],
    postwake_custody_handshake_contract: (
        OriginalConfirmatoryPostwakeCustodyHandshakeContract | Mapping[str, Any]
    ),
    expected_run_directory: str | Path,
    expected_terminal_custody_authority_projection: Mapping[str, Any],
    terminal_client_launcher_release: Mapping[str, Any],
    expected_environment: ExpectedLaunchEnvironmentEnvelopeV1 | Mapping[str, Any],
    process_environment_binding: OriginalConfirmatoryProcessEnvironmentBinding
    | Mapping[str, Any],
    supervisor_terminal_receipt_path: str | Path,
    supervisor_terminal_receipt_max_bytes: int,
    verifier_stdout_path: str | Path,
    verifier_stdout_max_bytes: int,
    preterminal_pin_receipt_path: str | Path,
    preterminal_pin_receipt_max_bytes: int,
    postwake_input_lease_receipt_max_bytes: int,
    postwake_composed_readback_receipt_max_bytes: int,
    composed_terminal_receipt_path: str | Path,
    composed_terminal_receipt_max_bytes: int,
) -> OriginalConfirmatoryTerminalCompositionContract:
    canonical_capsule = canonical_original_confirmatory_execution_capsule(capsule)
    envelope = canonical_expected_launch_environment_envelope_v1(expected_environment)
    environment_binding = canonical_original_confirmatory_process_environment_binding(
        process_environment_binding,
        expected_environment=envelope,
    )
    canonical_command = canonical_original_confirmatory_capsule_command(
        verifier_command,
        capsule=canonical_capsule,
        expected_mode=CAPSULE_TERMINAL_MODE,
        expected_tail_argv=verifier_command_tail_argv,
    )
    handshake_contract = (
        canonical_original_confirmatory_preterminal_overlap_handshake_contract(
            preterminal_overlap_handshake_contract
        )
    )
    input_lease_contract = (
        canonical_original_confirmatory_postwake_input_lease_contract(
            postwake_input_lease_contract
        )
    )
    custody_seed = canonical_original_confirmatory_postwake_custody_seed(
        postwake_custody_seed
    )
    custody_contract = (
        canonical_original_confirmatory_postwake_custody_handshake_contract(
            postwake_custody_handshake_contract,
            custody_seed=custody_seed,
        )
    )
    expected_projection_input = _mapping(
        expected_terminal_custody_authority_projection,
        role="expected terminal custody authority projection",
    )
    expected_launcher_projection_input = _mapping(
        expected_projection_input.get("terminal_client_launcher_projection"),
        role="expected terminal-client launcher projection",
    )
    terminal_custody_projection = (
        canonical_original_confirmatory_terminal_custody_authority_projection(
            expected_projection_input,
            run_id=cast(str, custody_seed.payload["run_id"]),
            expected_run_directory=expected_run_directory,
            custody_contract=custody_contract,
            launcher_release=terminal_client_launcher_release,
            capsule=canonical_capsule,
            supervisor_job_id=cast(str, custody_seed.payload["supervisor_job_id"]),
            supervisor_job_directory=cast(
                str,
                custody_seed.payload["supervisor_job_dir"],
            ),
            verify_terminal_command_projection_sha256=cast(
                str,
                expected_launcher_projection_input.get(
                    "verify_terminal_command_projection_sha256"
                ),
            ),
            verify_terminal_environment_sha256=(
                environment_binding.exact_integrity_verifier_environment_sha256
            ),
            verify_terminal_cwd=canonical_command.cwd,
        )
    )
    provisional = OriginalConfirmatoryTerminalCompositionContract(
        capsule_contract_sha256=canonical_capsule.contract_sha256,
        capsule_path=canonical_capsule.path,
        capsule_sha256=canonical_capsule.sha256,
        capsule_internal_manifest_sha256=(canonical_capsule.internal_manifest_sha256),
        capsule_mode=CAPSULE_TERMINAL_MODE,
        verifier_command=canonical_command.as_dict(),
        verifier_command_sha256=canonical_command.command_sha256,
        preterminal_pin_contract_sha256=_sha256(
            preterminal_pin_contract_sha256,
            role="terminal composition preterminal contract",
        ),
        preterminal_overlap_handshake_contract_sha256=(
            handshake_contract.contract_sha256
        ),
        postwake_input_lease_contract_sha256=(input_lease_contract.contract_sha256),
        postwake_custody_seed_sha256=custody_seed.seed_sha256,
        postwake_custody_handshake_contract_sha256=(custody_contract.contract_sha256),
        terminal_custody_authority_projection=terminal_custody_projection,
        expected_environment_envelope_sha256=envelope.envelope_sha256,
        process_environment_binding_sha256=environment_binding.binding_sha256,
        exact_integrity_verifier_environment_sha256=(
            environment_binding.exact_integrity_verifier_environment_sha256
        ),
        capsule_lease_identity_root_sha256=(
            canonical_capsule.capsule_lease_identity_root_sha256
        ),
        capsule_ancestor_lease_root_sha256=(
            canonical_capsule.capsule_ancestor_lease_root_sha256
        ),
        supervisor_terminal_receipt_path=Path(
            _absolute_path(
                str(supervisor_terminal_receipt_path),
                role="supervisor terminal receipt path",
            )
        ),
        supervisor_terminal_receipt_max_bytes=_positive_int(
            supervisor_terminal_receipt_max_bytes,
            role="supervisor terminal receipt bound",
        ),
        verifier_stdout_path=Path(
            _absolute_path(
                str(verifier_stdout_path),
                role="preterminal verifier stdout path",
            )
        ),
        verifier_stdout_max_bytes=_positive_int(
            verifier_stdout_max_bytes,
            role="preterminal verifier stdout bound",
        ),
        verifier_stderr_path=input_lease_contract.verifier_stderr_path,
        verifier_stderr_max_bytes=input_lease_contract.verifier_log_max_bytes,
        preterminal_pin_receipt_path=Path(
            _absolute_path(
                str(preterminal_pin_receipt_path),
                role="preterminal pin receipt path",
            )
        ),
        preterminal_pin_receipt_max_bytes=_positive_int(
            preterminal_pin_receipt_max_bytes,
            role="preterminal pin receipt bound",
        ),
        postwake_input_lease_receipt_path=input_lease_contract.lease_receipt_path,
        postwake_input_lease_receipt_max_bytes=_positive_int(
            postwake_input_lease_receipt_max_bytes,
            role="postwake input lease receipt bound",
        ),
        postwake_composed_readback_receipt_path=(
            custody_contract.readback_receipt_path
        ),
        postwake_composed_readback_receipt_max_bytes=_positive_int(
            postwake_composed_readback_receipt_max_bytes,
            role="postwake composed readback receipt bound",
        ),
        composed_terminal_receipt_path=Path(
            _absolute_path(
                str(composed_terminal_receipt_path),
                role="composed terminal receipt path",
            )
        ),
        composed_terminal_receipt_max_bytes=_positive_int(
            composed_terminal_receipt_max_bytes,
            role="composed terminal receipt bound",
        ),
        semantic_outcome_read_scope=SEMANTIC_OUTCOME_READ_SCOPE,
        contract_sha256="0" * 64,
    )
    unsigned = provisional.payload_without_self_hash()
    return canonical_original_confirmatory_terminal_composition_contract(
        {**unsigned, "contract_sha256": canonical_json_sha256(unsigned)},
        capsule=canonical_capsule,
        verifier_command=canonical_command,
        verifier_command_tail_argv=verifier_command_tail_argv,
        preterminal_pin_contract_sha256=preterminal_pin_contract_sha256,
        preterminal_overlap_handshake_contract=handshake_contract,
        postwake_input_lease_contract=input_lease_contract,
        postwake_custody_seed=custody_seed,
        postwake_custody_handshake_contract=custody_contract,
        expected_run_directory=expected_run_directory,
        expected_terminal_custody_authority_projection=terminal_custody_projection,
        terminal_client_launcher_release=terminal_client_launcher_release,
        expected_environment=envelope,
        process_environment_binding=environment_binding,
    )


_SCIENTIFIC_AUTHORITY_PROJECTION_FIELDS = {
    "schema_version",
    "policy",
    "historical_primary_authority_artifact_root_sha256",
    "historical_primary_evidence_sha256",
    "technical_authorization_sha256",
    "technical_execution_source_root_sha256",
    "technical_execution_source_manifest_sha256",
    "source_delta_sha256",
    "confirmatory_storage_policy_sha256",
    "independent_review_receipt_sha256",
    "static_runner_binding",
    "static_runner_binding_sha256",
    "scientific_authority_root_sha256",
}


_STATIC_RUNNER_BINDING_FIELDS = {
    "schema_version",
    "policy",
    "project_root",
    "primary_run_directory",
    "freeze_directory",
    "technical_authority_directory",
    "technical_authority_artifact_root_sha256",
    "technical_authorization_sha256",
    "published_technical_authority_lifecycle_binding",
    "lifecycle_readiness_run_directory",
    "dataset_path",
    "manifest_path",
    "duplicate_audit_path",
    "pathology_encoder_audit_path",
    "frozen_primary_config_path",
    "frozen_confirmatory_config_path",
    "runs_root",
    "expected_confirmatory_gate",
    "expected_confirmatory_gate_sha256",
    "expected_cli_input_binding",
    "expected_cli_input_binding_sha256",
    "artifact_scope",
    "semantic_outcome_read_scope",
    "binding_sha256",
}
_PUBLISHED_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_FIELDS = {
    "schema_version",
    "policy",
    "namespace_directory",
    "namespace_claim_sha256",
    "review_attempt_claim_sha256",
    "technical_authority",
    "automatic_retry_allowed",
    "adoption_allowed",
    "cleanup_allowed",
    "binding_sha256",
}
_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_FIELDS = {
    "schema_version",
    "policy",
    "authority_directory",
    "chain_depth",
    "artifact_root_sha256",
    "sha256_manifest_sha256",
    "execution_source_manifest_sha256",
    "execution_source_root_sha256",
    "parent_authority_directory",
    "parent_artifact_root_sha256",
    "parent_sha256_manifest_sha256",
    "technical_authorization_sha256",
    "independent_review_receipt_sha256",
    "immutable_marker_sha256",
    "publication_attempt_sha256",
    "publication_success_sha256",
    "primary_outcomes_inspected",
    "confirmatory_outcomes_inspected",
    "confirmatory_outcome_values_read",
    "scientific_definition_changed",
    "automatic_retry_allowed",
    "binding_sha256",
}
_CLI_INPUT_BINDING_FIELDS = {
    "schema_version",
    "policy",
    "crop_cache_path",
    "expected_crop_cache_sha256",
    "expected_crop_metadata_sha256",
    "expected_raw_inventory_sha256",
    "frozen_feature_caches",
    "frozen_feature_caches_sha256",
    "observed_label_sets",
    "observed_label_sets_sha256",
    "draft_checkpoint_contract",
    "draft_checkpoint_contract_sha256",
    "bridge_binding_sha256",
    "scientific_outcomes_read",
    "automatic_retry_allowed",
    "binding_sha256",
}
_CLI_INPUT_BINDING_POLICY = "original_confirmatory_cli_input_binding_v1"


def canonical_original_confirmatory_cli_input_binding(
    value: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    raw = _mapping(value, role="Q CLI input binding")
    if set(raw) != _CLI_INPUT_BINDING_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q CLI input binding has an unexpected field set"
        )
    root = Path(_absolute_path(str(project_root), role="Q CLI input project root"))
    crop_path = Path(_absolute_path(raw["crop_cache_path"], role="Q CLI crop cache"))
    caches = raw["frozen_feature_caches"]
    labels = raw["observed_label_sets"]
    draft = _mapping(
        raw["draft_checkpoint_contract"],
        role="Q CLI draft checkpoint contract",
    )
    unsigned = {key: item for key, item in raw.items() if key != "binding_sha256"}
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _CLI_INPUT_BINDING_POLICY
        or root not in crop_path.parents
        or any(
            _SHA256.fullmatch(raw[field]) is None
            for field in (
                "expected_crop_cache_sha256",
                "expected_crop_metadata_sha256",
                "expected_raw_inventory_sha256",
                "frozen_feature_caches_sha256",
                "observed_label_sets_sha256",
                "draft_checkpoint_contract_sha256",
                "bridge_binding_sha256",
            )
            if isinstance(raw[field], str)
        )
        or not all(
            isinstance(raw[field], str)
            for field in (
                "expected_crop_cache_sha256",
                "expected_crop_metadata_sha256",
                "expected_raw_inventory_sha256",
                "frozen_feature_caches_sha256",
                "observed_label_sets_sha256",
                "draft_checkpoint_contract_sha256",
                "bridge_binding_sha256",
            )
        )
        or not isinstance(caches, list)
        or not all(isinstance(item, dict) and item for item in caches)
        or not isinstance(labels, list)
        or not all(isinstance(item, dict) and item for item in labels)
        or raw["frozen_feature_caches_sha256"] != canonical_json_sha256(caches)
        or raw["observed_label_sets_sha256"] != canonical_json_sha256(labels)
        or not draft
        or raw["draft_checkpoint_contract_sha256"] != canonical_json_sha256(draft)
        or raw["scientific_outcomes_read"] is not False
        or raw["automatic_retry_allowed"] is not False
        or _contains_mapping_key(raw, "supervisor_spec_sha256")
        or raw["binding_sha256"] != canonical_json_sha256(unsigned)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q CLI input binding violates its exact outcome-blind policy"
        )
    return {
        **raw,
        "crop_cache_path": str(crop_path),
        "draft_checkpoint_contract": draft,
    }


def build_original_confirmatory_cli_input_binding(
    *,
    project_root: str | Path,
    crop_cache_path: str | Path,
    expected_crop_cache_sha256: str,
    expected_crop_metadata_sha256: str,
    expected_raw_inventory_sha256: str,
    frozen_feature_caches: Sequence[Mapping[str, Any]],
    observed_label_sets: Sequence[Mapping[str, Any]],
    draft_checkpoint_contract: Mapping[str, Any],
    bridge_binding_sha256: str,
) -> dict[str, Any]:
    caches = [dict(item) for item in frozen_feature_caches]
    labels = [dict(item) for item in observed_label_sets]
    draft = dict(draft_checkpoint_contract)
    unsigned = {
        "schema_version": 1,
        "policy": _CLI_INPUT_BINDING_POLICY,
        "crop_cache_path": str(crop_cache_path),
        "expected_crop_cache_sha256": expected_crop_cache_sha256,
        "expected_crop_metadata_sha256": expected_crop_metadata_sha256,
        "expected_raw_inventory_sha256": expected_raw_inventory_sha256,
        "frozen_feature_caches": caches,
        "frozen_feature_caches_sha256": canonical_json_sha256(caches),
        "observed_label_sets": labels,
        "observed_label_sets_sha256": canonical_json_sha256(labels),
        "draft_checkpoint_contract": draft,
        "draft_checkpoint_contract_sha256": canonical_json_sha256(draft),
        "bridge_binding_sha256": bridge_binding_sha256,
        "scientific_outcomes_read": False,
        "automatic_retry_allowed": False,
    }
    return canonical_original_confirmatory_cli_input_binding(
        {**unsigned, "binding_sha256": canonical_json_sha256(unsigned)},
        project_root=project_root,
    )


def canonical_published_original_confirmatory_technical_authority_lifecycle_binding(
    value: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Validate the exact publisher-composite lifecycle receipt without discovery."""

    raw = _mapping(value, role="published technical authority lifecycle binding")
    if set(raw) != _PUBLISHED_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "published technical authority lifecycle binding has an unexpected field set"
        )
    technical = _mapping(
        raw["technical_authority"],
        role="published technical authority nested lifecycle binding",
    )
    if set(technical) != _TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "published technical authority nested lifecycle binding has an unexpected field set"
        )
    root = Path(
        _absolute_path(
            str(project_root), role="published technical authority project root"
        )
    )
    namespace = Path(
        _absolute_path(
            raw["namespace_directory"],
            role="published technical authority namespace directory",
        )
    )
    authority_directory = Path(
        _absolute_path(
            technical["authority_directory"],
            role="published technical authority directory",
        )
    )
    parent_authority_directory = Path(
        _absolute_path(
            technical["parent_authority_directory"],
            role="published technical authority parent directory",
        )
    )
    technical_hash_fields = (
        "artifact_root_sha256",
        "sha256_manifest_sha256",
        "execution_source_manifest_sha256",
        "execution_source_root_sha256",
        "parent_artifact_root_sha256",
        "parent_sha256_manifest_sha256",
        "technical_authorization_sha256",
        "independent_review_receipt_sha256",
        "immutable_marker_sha256",
        "publication_attempt_sha256",
        "publication_success_sha256",
    )
    for field in technical_hash_fields:
        _sha256(
            technical[field],
            role=f"published technical authority {field}",
        )
    _sha256(
        raw["namespace_claim_sha256"],
        role="published technical authority namespace claim",
    )
    _sha256(
        raw["review_attempt_claim_sha256"],
        role="published technical authority review-attempt claim",
    )
    technical_unsigned = {
        key: item for key, item in technical.items() if key != "binding_sha256"
    }
    published_unsigned = {
        key: item for key, item in raw.items() if key != "binding_sha256"
    }
    if (
        type(technical["schema_version"]) is not int
        or technical["schema_version"] != 1
        or technical["policy"] != TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_POLICY
        or type(technical["chain_depth"]) is not int
        or technical["chain_depth"] < 1
        or technical["primary_outcomes_inspected"] is not True
        or technical["confirmatory_outcomes_inspected"] is not False
        or technical["confirmatory_outcome_values_read"] is not False
        or technical["scientific_definition_changed"] is not False
        or technical["automatic_retry_allowed"] is not False
        or technical["binding_sha256"] != canonical_json_sha256(technical_unsigned)
        or type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != PUBLISHED_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_POLICY
        or raw["automatic_retry_allowed"] is not False
        or raw["adoption_allowed"] is not False
        or raw["cleanup_allowed"] is not False
        or raw["binding_sha256"] != canonical_json_sha256(published_unsigned)
        or namespace
        != root / "artifacts" / "original_confirmatory_technical_authorities"
        or authority_directory.parent != namespace
        or root not in parent_authority_directory.parents
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "published technical authority lifecycle binding violates its exact one-use policy"
        )
    return {
        **raw,
        "namespace_directory": str(namespace),
        "technical_authority": {
            **technical,
            "authority_directory": str(authority_directory),
            "parent_authority_directory": str(parent_authority_directory),
        },
    }


def canonical_original_confirmatory_static_runner_binding(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, role="Q static runner binding")
    if set(raw) != _STATIC_RUNNER_BINDING_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q static runner binding has an unexpected field set"
        )
    project_root = Path(
        _absolute_path(raw["project_root"], role="Q static runner project root")
    )
    path_fields = (
        "primary_run_directory",
        "freeze_directory",
        "technical_authority_directory",
        "lifecycle_readiness_run_directory",
        "dataset_path",
        "manifest_path",
        "duplicate_audit_path",
        "pathology_encoder_audit_path",
        "frozen_primary_config_path",
        "frozen_confirmatory_config_path",
        "runs_root",
    )
    paths = {
        field: _absolute_path(
            raw[field],
            role=f"Q static runner {field}",
        )
        for field in path_fields
    }
    gate = _mapping(
        raw["expected_confirmatory_gate"],
        role="Q static runner confirmatory gate",
    )
    cli_binding = canonical_original_confirmatory_cli_input_binding(
        raw["expected_cli_input_binding"],
        project_root=project_root,
    )
    published_binding = (
        canonical_published_original_confirmatory_technical_authority_lifecycle_binding(
            raw["published_technical_authority_lifecycle_binding"],
            project_root=project_root,
        )
    )
    published_technical = published_binding["technical_authority"]
    _sha256(
        raw["technical_authority_artifact_root_sha256"],
        role="Q technical authority artifact root",
    )
    _sha256(
        raw["technical_authorization_sha256"],
        role="Q technical authorization",
    )
    if not gate or not cli_binding:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q static runner binding lacks its typed control projections"
        )
    unsigned = {key: item for key, item in raw.items() if key != "binding_sha256"}
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 3
        or raw["policy"] != STATIC_RUNNER_BINDING_POLICY
        or paths["technical_authority_directory"]
        != published_technical["authority_directory"]
        or paths["freeze_directory"]
        != published_technical["parent_authority_directory"]
        or raw["technical_authority_artifact_root_sha256"]
        != published_technical["artifact_root_sha256"]
        or raw["technical_authorization_sha256"]
        != published_technical["technical_authorization_sha256"]
        or raw["expected_confirmatory_gate_sha256"] != canonical_json_sha256(gate)
        or raw["expected_cli_input_binding_sha256"]
        != canonical_json_sha256(cli_binding)
        or raw["artifact_scope"] != REAL_CONFIRMATORY_ARTIFACT_SCOPE
        or raw["semantic_outcome_read_scope"] != SCIENTIFIC_CONTROL_READ_SCOPE
        or Path(paths["runs_root"]).parent != project_root / "artifacts"
        or not all(
            path == project_root or project_root in path.parents
            for path in (Path(item) for item in paths.values())
        )
        or _contains_mapping_key(raw, "supervisor_spec_sha256")
        or raw["binding_sha256"] != canonical_json_sha256(unsigned)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q static runner binding violates its exact outcome-blind policy"
        )
    return {
        **raw,
        **paths,
        "published_technical_authority_lifecycle_binding": published_binding,
        "expected_confirmatory_gate": gate,
        "expected_cli_input_binding": cli_binding,
    }


def build_original_confirmatory_static_runner_binding(
    *,
    project_root: str | Path,
    primary_run_directory: str | Path,
    freeze_directory: str | Path,
    technical_authority_directory: str | Path,
    technical_authority_artifact_root_sha256: str,
    technical_authorization_sha256: str,
    published_technical_authority_lifecycle_binding: Mapping[str, Any],
    lifecycle_readiness_run_directory: str | Path,
    dataset_path: str | Path,
    manifest_path: str | Path,
    duplicate_audit_path: str | Path,
    pathology_encoder_audit_path: str | Path,
    frozen_primary_config_path: str | Path,
    frozen_confirmatory_config_path: str | Path,
    runs_root: str | Path,
    expected_confirmatory_gate: Mapping[str, Any],
    expected_cli_input_binding: Mapping[str, Any],
) -> dict[str, Any]:
    gate = _mapping(
        expected_confirmatory_gate,
        role="Q static runner confirmatory gate",
    )
    cli_binding = _mapping(
        expected_cli_input_binding,
        role="Q static runner CLI input binding",
    )
    published_binding = (
        canonical_published_original_confirmatory_technical_authority_lifecycle_binding(
            published_technical_authority_lifecycle_binding,
            project_root=project_root,
        )
    )
    unsigned = {
        "schema_version": 3,
        "policy": STATIC_RUNNER_BINDING_POLICY,
        "project_root": str(project_root),
        "primary_run_directory": str(primary_run_directory),
        "freeze_directory": str(freeze_directory),
        "technical_authority_directory": str(technical_authority_directory),
        "technical_authority_artifact_root_sha256": (
            technical_authority_artifact_root_sha256
        ),
        "technical_authorization_sha256": technical_authorization_sha256,
        "published_technical_authority_lifecycle_binding": published_binding,
        "lifecycle_readiness_run_directory": str(lifecycle_readiness_run_directory),
        "dataset_path": str(dataset_path),
        "manifest_path": str(manifest_path),
        "duplicate_audit_path": str(duplicate_audit_path),
        "pathology_encoder_audit_path": str(pathology_encoder_audit_path),
        "frozen_primary_config_path": str(frozen_primary_config_path),
        "frozen_confirmatory_config_path": str(frozen_confirmatory_config_path),
        "runs_root": str(runs_root),
        "expected_confirmatory_gate": gate,
        "expected_confirmatory_gate_sha256": canonical_json_sha256(gate),
        "expected_cli_input_binding": cli_binding,
        "expected_cli_input_binding_sha256": canonical_json_sha256(cli_binding),
        "artifact_scope": REAL_CONFIRMATORY_ARTIFACT_SCOPE,
        "semantic_outcome_read_scope": SCIENTIFIC_CONTROL_READ_SCOPE,
    }
    return canonical_original_confirmatory_static_runner_binding(
        {**unsigned, "binding_sha256": canonical_json_sha256(unsigned)}
    )


def canonical_original_confirmatory_scientific_authority_projection(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, role="Q scientific authority projection")
    if set(raw) != _SCIENTIFIC_AUTHORITY_PROJECTION_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q scientific authority projection has an unexpected field set"
        )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != SCIENTIFIC_AUTHORITY_PROJECTION_POLICY
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q scientific authority projection violates its exact policy"
        )
    static_binding = canonical_original_confirmatory_static_runner_binding(
        raw["static_runner_binding"]
    )
    published_technical = static_binding[
        "published_technical_authority_lifecycle_binding"
    ]["technical_authority"]
    unsigned = {
        "schema_version": 1,
        "policy": SCIENTIFIC_AUTHORITY_PROJECTION_POLICY,
        **{
            field: _sha256(raw[field], role=f"Q scientific authority {field}")
            for field in sorted(
                _SCIENTIFIC_AUTHORITY_PROJECTION_FIELDS
                - {
                    "schema_version",
                    "policy",
                    "static_runner_binding",
                    "static_runner_binding_sha256",
                    "scientific_authority_root_sha256",
                }
            )
        },
        "static_runner_binding": static_binding,
        "static_runner_binding_sha256": static_binding["binding_sha256"],
    }
    if (
        raw["static_runner_binding_sha256"] != static_binding["binding_sha256"]
        or raw["technical_authorization_sha256"]
        != static_binding["technical_authorization_sha256"]
        or raw["historical_primary_authority_artifact_root_sha256"]
        != published_technical["parent_artifact_root_sha256"]
        or raw["technical_execution_source_root_sha256"]
        != published_technical["execution_source_root_sha256"]
        or raw["technical_execution_source_manifest_sha256"]
        != published_technical["execution_source_manifest_sha256"]
        or raw["independent_review_receipt_sha256"]
        != published_technical["independent_review_receipt_sha256"]
        or raw["scientific_authority_root_sha256"] != canonical_json_sha256(unsigned)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q scientific authority nested roots differ"
        )
    return {
        **unsigned,
        "scientific_authority_root_sha256": raw["scientific_authority_root_sha256"],
    }


def build_original_confirmatory_scientific_authority_projection(
    *,
    static_runner_binding: Mapping[str, Any],
    **hashes: str,
) -> dict[str, Any]:
    expected = _SCIENTIFIC_AUTHORITY_PROJECTION_FIELDS - {
        "schema_version",
        "policy",
        "static_runner_binding",
        "static_runner_binding_sha256",
        "scientific_authority_root_sha256",
    }
    if set(hashes) != expected:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q scientific authority builder received an unexpected field set"
        )
    binding = canonical_original_confirmatory_static_runner_binding(
        static_runner_binding
    )
    unsigned = {
        "schema_version": 1,
        "policy": SCIENTIFIC_AUTHORITY_PROJECTION_POLICY,
        **hashes,
        "static_runner_binding": binding,
        "static_runner_binding_sha256": binding["binding_sha256"],
    }
    return canonical_original_confirmatory_scientific_authority_projection(
        {
            **unsigned,
            "scientific_authority_root_sha256": canonical_json_sha256(unsigned),
        }
    )


_COMMAND_DERIVATION_CONTRACT_FIELDS = {
    "schema_version",
    "policy",
    "projection_policy",
    "canonical_file_hash_policy",
    "canonical_core_hash_policy",
    "e_file_sha256_flag",
    "e_core_sha256_flag",
    "e_file_sha256_insertion_policy",
    "e_core_sha256_insertion_policy",
    "python_isolated_flags",
    "allowed_modes",
    "common_tail_flags",
    "successor_lineage_flag",
    "preterminal_suffix_flags",
    "terminal_suffix_flags",
    "exact_argv_rederivation_required",
    "final_command_carrier",
    "post_wait_rederivation_required",
    "extra_argv_allowed",
    "extra_environment_allowed",
    "contract_sha256",
}


def _command_derivation_contract_unsigned() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy": COMMAND_DERIVATION_CONTRACT_POLICY,
        "projection_policy": COMMAND_PROJECTION_POLICY,
        "canonical_file_hash_policy": SUPERVISOR_TERMINAL_HASH_POLICY,
        "canonical_core_hash_policy": CANONICAL_CORE_HASH_POLICY,
        "e_file_sha256_flag": "--e-intent-sha256",
        "e_core_sha256_flag": "--e-intent-core-sha256",
        "e_file_sha256_insertion_policy": COMMAND_E_FILE_INSERTION_POLICY,
        "e_core_sha256_insertion_policy": COMMAND_E_CORE_INSERTION_POLICY,
        "python_isolated_flags": list(CAPSULE_PYTHON_ISOLATED_FLAGS),
        "allowed_modes": list(CAPSULE_ALLOWED_MODES),
        "common_tail_flags": list(CAPSULE_COMMON_TAIL_FLAGS),
        "successor_lineage_flag": CAPSULE_SUCCESSOR_LINEAGE_FLAG,
        "preterminal_suffix_flags": list(CAPSULE_PRETERMINAL_SUFFIX_FLAGS),
        "terminal_suffix_flags": list(CAPSULE_TERMINAL_SUFFIX_FLAGS),
        "exact_argv_rederivation_required": True,
        "final_command_carrier": COMMAND_FINAL_CARRIER_POLICY,
        "post_wait_rederivation_required": True,
        "extra_argv_allowed": False,
        "extra_environment_allowed": False,
    }


def build_original_confirmatory_command_derivation_contract() -> dict[str, Any]:
    unsigned = _command_derivation_contract_unsigned()
    return {**unsigned, "contract_sha256": canonical_json_sha256(unsigned)}


def canonical_original_confirmatory_command_derivation_contract(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, role="Q command derivation contract")
    expected = build_original_confirmatory_command_derivation_contract()
    if set(raw) != _COMMAND_DERIVATION_CONTRACT_FIELDS or not _strict_json_value_equal(
        raw, expected
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q command derivation contract violates its exact policy"
        )
    return expected


_TERMINAL_CLIENT_LAUNCHER_ANCESTOR_LEASE_FIELDS = {
    "schema_version",
    "policy",
    "supervisor_root",
    "records",
    "record_count",
    "records_root_sha256",
    "directory_access_mask",
    "share_access",
    "delete_share",
    "write_access",
    "owner_process_identity_required",
    "handle_slot_per_record_required",
    "retained_from_q_verification_through_terminal_child_waitforexit",
    "acquisition_disposition",
}
_TERMINAL_CLIENT_LAUNCHER_ANCESTOR_DISPOSITION = (
    "supervisor_root_handle_opened_before_q_verification_retained_through_"
    "terminal_child_waitforexit_v1"
)


def canonical_original_confirmatory_terminal_client_launcher_ancestor_lease(
    value: Mapping[str, Any],
    *,
    supervisor_root: str | Path,
) -> dict[str, Any]:
    """Validate the retained no-delete directory identity for the launcher."""

    raw = _mapping(value, role="terminal-client launcher ancestor lease")
    root = Path(
        _absolute_path(
            str(supervisor_root),
            role="terminal-client launcher supervisor root",
        )
    )
    items = raw.get("records")
    if not isinstance(items, list) or len(items) != 1:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "terminal-client launcher ancestor lease must contain exactly its supervisor root"
        )
    record = _mapping(items[0], role="terminal-client launcher ancestor record")
    file_id = record.get("file_id_128")
    attributes = record.get("file_attributes")
    if (
        set(raw) != _TERMINAL_CLIENT_LAUNCHER_ANCESTOR_LEASE_FIELDS
        or set(record) != _CAPSULE_ANCESTOR_RECORD_FIELDS
        or Path(
            _absolute_path(
                cast(str, record.get("path")),
                role="terminal-client launcher ancestor path",
            )
        )
        != root
        or type(record.get("volume_serial_number")) is not int
        or cast(int, record["volume_serial_number"]) < 0
        or not isinstance(file_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", file_id) is None
        or type(attributes) is not int
        or attributes < 0
        or attributes & 0x400
        or record.get("reparse_point") is not False
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "terminal-client launcher ancestor lease has an invalid physical identity"
        )
    canonical_record = {
        "path": str(root),
        "volume_serial_number": record["volume_serial_number"],
        "file_id_128": file_id,
        "file_attributes": attributes,
        "reparse_point": False,
    }
    expected = {
        "schema_version": 1,
        "policy": TERMINAL_CLIENT_LAUNCHER_ANCESTOR_LEASE_POLICY,
        "supervisor_root": str(root),
        "records": [canonical_record],
        "record_count": 1,
        "records_root_sha256": canonical_json_sha256([canonical_record]),
        "directory_access_mask": EXECUTABLE_ANCESTOR_ACCESS_MASK,
        "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        "delete_share": False,
        "write_access": False,
        "owner_process_identity_required": True,
        "handle_slot_per_record_required": True,
        "retained_from_q_verification_through_terminal_child_waitforexit": True,
        "acquisition_disposition": _TERMINAL_CLIENT_LAUNCHER_ANCESTOR_DISPOSITION,
    }
    if not _strict_json_value_equal(raw, expected):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "terminal-client launcher ancestor lease violates its exact policy"
        )
    return expected


def build_original_confirmatory_terminal_client_launcher_ancestor_lease(
    *,
    supervisor_root: str | Path,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root = Path(
        _absolute_path(
            str(supervisor_root),
            role="terminal-client launcher ancestor build root",
        )
    )
    canonical_records = [dict(item) for item in records]
    raw = {
        "schema_version": 1,
        "policy": TERMINAL_CLIENT_LAUNCHER_ANCESTOR_LEASE_POLICY,
        "supervisor_root": str(root),
        "records": canonical_records,
        "record_count": len(canonical_records),
        "records_root_sha256": canonical_json_sha256(canonical_records),
        "directory_access_mask": EXECUTABLE_ANCESTOR_ACCESS_MASK,
        "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        "delete_share": False,
        "write_access": False,
        "owner_process_identity_required": True,
        "handle_slot_per_record_required": True,
        "retained_from_q_verification_through_terminal_child_waitforexit": True,
        "acquisition_disposition": _TERMINAL_CLIENT_LAUNCHER_ANCESTOR_DISPOSITION,
    }
    return canonical_original_confirmatory_terminal_client_launcher_ancestor_lease(
        raw,
        supervisor_root=root,
    )


_TERMINAL_CLIENT_LAUNCHER_RELEASE_FIELDS = {
    "schema_version",
    "policy",
    "supervisor_root",
    "source_path",
    "source_size_bytes",
    "source_sha256",
    "source_physical_identity",
    "source_physical_identity_root_sha256",
    "source_ancestor_lease",
    "source_ancestor_lease_root_sha256",
    "source_leaf_access_mask",
    "source_leaf_share_access",
    "source_delete_access",
    "source_handle_retained_through_terminal_child_waitforexit",
    "program_path",
    "program_sha256",
    "createprocess_application_path",
    "createprocess_application_sha256",
    "program_lease_identity_root_sha256",
    "program_ancestor_lease_root_sha256",
    "logical_venv_python_path",
    "logical_venv_python_sha256",
    "logical_venv_python_lease_identity_root_sha256",
    "logical_venv_python_ancestor_lease_root_sha256",
    "runtime_python_path",
    "runtime_python_sha256",
    "expected_live_image_path",
    "expected_live_image_sha256",
    "direct_base_runtime_live_parity_policy",
    "runtime_python_lease_identity_root_sha256",
    "runtime_python_ancestor_lease_root_sha256",
    "python_isolated_flags",
    "python_sys_argv_prefix",
    "process_argv_prefix",
    "cwd_binding",
    "final_argument_order",
    "downstream_hash_insertions",
    "command_preimage_policy",
    "command_sha256_policy",
    "launch_intent_filename",
    "launch_intent_path_binding",
    "launch_intent_publication_policy",
    "launch_intent_schema_version",
    "launch_intent_field_names",
    "launch_intent_status",
    "launch_intent_create_disposition",
    "launch_intent_physical_identity_policy",
    "launch_intent_physical_identity_role",
    "launch_intent_physical_identity_field_names",
    "launch_intent_created_before_child_process_required",
    "existing_or_partial_launch_intent_is_stop",
    "launch_intent_write_through_fsync_readonly_required",
    "launch_intent_creator_desired_access_mask",
    "launch_intent_creator_share_access",
    "launch_intent_delete_access_allowed",
    "launch_intent_delete_share_allowed",
    "launch_intent_cleanup_allowed",
    "launch_intent_same_handle_write_fsync_readback_required",
    "launch_intent_set_readonly_before_child_create_required",
    "launch_intent_creator_handle_retained_through_terminal_child_waitforexit_required",
    "terminal_child_launch_intent_no_follow_readonly_single_link_no_ads_required",
    "terminal_child_launch_intent_handle_retained_through_final_ack_required",
    "supervisor_launch_intent_handle_retained_through_final_ack_required",
    "claim_binds_launch_intent_deterministic_path_and_policy_only",
    "claim_launch_intent_read_must_be_false",
    "grant_required_before_terminal_child_launch_intent_read",
    "grant_independently_verifies_launch_intent_and_launcher_identity",
    "launch_intent_supervisor_granted_access_mask",
    "launch_intent_child_duplicate_target_access_mask",
    "launch_intent_child_expected_granted_access_mask",
    "launch_intent_child_duplicate_options",
    "launch_intent_child_duplicate_close_source",
    "child_stdio_policy",
    "child_stdin_source",
    "child_stdout_transport",
    "child_stderr_transport",
    "child_inherited_handle_list_policy",
    "child_inherited_handle_count",
    "createprocess_inherit_handles",
    "startupinfoex_use_std_handles_required",
    "proc_thread_attribute_handle_list_required",
    "non_stdio_inherited_handles_allowed",
    "preterminal_or_terminal_input_file_handles_inherited_allowed",
    "child_stdout_max_bytes",
    "child_stderr_max_bytes",
    "child_stdout_summary_policy",
    "child_stdout_single_canonical_json_line_required",
    "child_stderr_empty_required",
    "stdio_pipe_drains_event_driven_concurrent_required",
    "launcher_forwards_validated_child_stdout_once_required",
    "sealed_input_allowlist",
    "project_import_allowed",
    "inherited_environment_for_child_allowed",
    "createprocessw_exact_child_required",
    "child_environment_encoding",
    "child_environment_source",
    "verify_terminal_child_launch_topology",
    "verify_terminal_immediate_redirector_program_path",
    "verify_terminal_immediate_redirector_program_sha256",
    "verify_terminal_runtime_child_program_path",
    "verify_terminal_runtime_child_program_sha256",
    "launcher_is_runtime_child_grandparent_required",
    "same_job_no_breakaway_required",
    "launcher_waits_for_child_exit_required",
    "launcher_parent_live_through_child_exit_required",
    "supervisor_retains_launcher_redirector_child_process_handles_through_final_ack_required",
    "immediate_venv_redirector_live_through_runtime_child_exit_required",
    "terminal_client_launcher_live_through_redirector_waitforexit_required",
    "process_liveness_reverified_at_final_ack_required",
    "automatic_retry_allowed",
    "fallback_allowed",
    "release_root_sha256",
}

_TERMINAL_CLIENT_LAUNCHER_FINAL_ARGUMENT_ORDER = (
    "--job-id",
    "--supervisor-job-directory",
    "--supervisor-spec",
    "--supervisor-spec-sha256",
    "--e-intent",
    "--e-intent-sha256",
    "--terminal-receipt",
    "--terminal-receipt-sha256",
    "--terminal-client-launch-intent",
    "--verify-terminal-command-projection-sha256",
    "--verify-terminal-command-sha256",
    "--verify-terminal-environment-sha256",
    "--verify-terminal-cwd",
)


def build_original_confirmatory_terminal_client_launcher_release(
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    *,
    supervisor_root: str | Path,
    source_physical_identity: OriginalConfirmatoryPhysicalFileIdentity
    | Mapping[str, Any],
    source_ancestor_lease: Mapping[str, Any],
) -> dict[str, Any]:
    """Build Q's retained, isolated, stdlib-only terminal-client release."""

    canonical_capsule = canonical_original_confirmatory_execution_capsule(capsule)
    root = Path(
        _absolute_path(
            str(supervisor_root),
            role="terminal-client launcher release root",
        )
    )
    source_path = root / TERMINAL_CLIENT_LAUNCHER_SOURCE_FILENAME
    source_identity = canonical_original_confirmatory_physical_file_identity(
        source_physical_identity,
        allowed_roles=("terminal-client-launcher",),
    )
    ancestor = canonical_original_confirmatory_terminal_client_launcher_ancestor_lease(
        source_ancestor_lease,
        supervisor_root=root,
    )
    if source_identity.path != source_path:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "terminal-client launcher physical identity differs from its exact source path"
        )
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "policy": TERMINAL_CLIENT_LAUNCHER_POLICY,
        "supervisor_root": str(root),
        "source_path": str(source_path),
        "source_size_bytes": source_identity.size_bytes,
        "source_sha256": source_identity.sha256,
        "source_physical_identity": source_identity.as_dict(),
        "source_physical_identity_root_sha256": canonical_json_sha256(
            source_identity.as_dict()
        ),
        "source_ancestor_lease": ancestor,
        "source_ancestor_lease_root_sha256": canonical_json_sha256(ancestor),
        "source_leaf_access_mask": EXECUTABLE_LEAF_ACCESS_MASK,
        "source_leaf_share_access": ["FILE_SHARE_READ"],
        "source_delete_access": False,
        "source_handle_retained_through_terminal_child_waitforexit": True,
        "program_path": str(canonical_capsule.runtime_python_path),
        "program_sha256": canonical_capsule.runtime_python_sha256,
        "createprocess_application_path": str(canonical_capsule.runtime_python_path),
        "createprocess_application_sha256": canonical_capsule.runtime_python_sha256,
        "program_lease_identity_root_sha256": (
            canonical_capsule.runtime_python_lease_identity_root_sha256
        ),
        "program_ancestor_lease_root_sha256": (
            canonical_capsule.runtime_python_ancestor_lease_root_sha256
        ),
        "logical_venv_python_path": str(canonical_capsule.python_path),
        "logical_venv_python_sha256": canonical_capsule.python_sha256,
        "logical_venv_python_lease_identity_root_sha256": (
            canonical_capsule.python_lease_identity_root_sha256
        ),
        "logical_venv_python_ancestor_lease_root_sha256": (
            canonical_capsule.python_ancestor_lease_root_sha256
        ),
        "runtime_python_path": str(canonical_capsule.runtime_python_path),
        "runtime_python_sha256": canonical_capsule.runtime_python_sha256,
        "expected_live_image_path": str(canonical_capsule.runtime_python_path),
        "expected_live_image_sha256": canonical_capsule.runtime_python_sha256,
        "direct_base_runtime_live_parity_policy": (
            DIRECT_BASE_RUNTIME_LIVE_PARITY_POLICY
        ),
        "runtime_python_lease_identity_root_sha256": (
            canonical_capsule.runtime_python_lease_identity_root_sha256
        ),
        "runtime_python_ancestor_lease_root_sha256": (
            canonical_capsule.runtime_python_ancestor_lease_root_sha256
        ),
        "python_isolated_flags": list(TERMINAL_CLIENT_PYTHON_ISOLATED_FLAGS),
        "python_sys_argv_prefix": [str(source_path)],
        "process_argv_prefix": [
            str(canonical_capsule.runtime_python_path),
            *TERMINAL_CLIENT_PYTHON_ISOLATED_FLAGS,
            str(source_path),
        ],
        "cwd_binding": "E.project_root",
        "final_argument_order": list(_TERMINAL_CLIENT_LAUNCHER_FINAL_ARGUMENT_ORDER),
        "downstream_hash_insertions": [
            "supervisor_spec_sha256",
            "e_intent_file_sha256",
            "terminal_receipt_sha256",
            "verify_terminal_command_sha256",
        ],
        "command_preimage_policy": TERMINAL_CLIENT_LAUNCHER_COMMAND_POLICY,
        "command_sha256_policy": SUPERVISOR_PROCESS_COMMAND_HASH_POLICY,
        "launch_intent_filename": TERMINAL_CLIENT_LAUNCH_INTENT_FILENAME,
        "launch_intent_path_binding": (
            "E.job.supervisor_job_dir/terminal_client_launch_intent.json"
        ),
        "launch_intent_publication_policy": TERMINAL_CLIENT_LAUNCH_INTENT_POLICY,
        "launch_intent_schema_version": 1,
        "launch_intent_field_names": sorted(_TERMINAL_CLIENT_LAUNCH_INTENT_FIELDS),
        "launch_intent_status": _TERMINAL_CLIENT_LAUNCH_INTENT_STATUS,
        "launch_intent_create_disposition": "CREATE_NEW",
        "launch_intent_physical_identity_policy": NO_FOLLOW_PHYSICAL_FILE_IDENTITY_POLICY,
        "launch_intent_physical_identity_role": "terminal-client-launch-intent",
        "launch_intent_physical_identity_field_names": sorted(
            _PHYSICAL_FILE_IDENTITY_FIELDS
        ),
        "launch_intent_created_before_child_process_required": True,
        "existing_or_partial_launch_intent_is_stop": True,
        "launch_intent_write_through_fsync_readonly_required": True,
        "launch_intent_creator_desired_access_mask": (
            GENERIC_READ_ACCESS_REQUEST | GENERIC_WRITE_ACCESS_REQUEST
        ),
        "launch_intent_creator_share_access": ["FILE_SHARE_READ"],
        "launch_intent_delete_access_allowed": False,
        "launch_intent_delete_share_allowed": False,
        "launch_intent_cleanup_allowed": False,
        "launch_intent_same_handle_write_fsync_readback_required": True,
        "launch_intent_set_readonly_before_child_create_required": True,
        "launch_intent_creator_handle_retained_through_terminal_child_waitforexit_required": True,
        "terminal_child_launch_intent_no_follow_readonly_single_link_no_ads_required": True,
        "terminal_child_launch_intent_handle_retained_through_final_ack_required": True,
        "supervisor_launch_intent_handle_retained_through_final_ack_required": True,
        "claim_binds_launch_intent_deterministic_path_and_policy_only": True,
        "claim_launch_intent_read_must_be_false": True,
        "grant_required_before_terminal_child_launch_intent_read": True,
        "grant_independently_verifies_launch_intent_and_launcher_identity": True,
        "launch_intent_supervisor_granted_access_mask": FILE_GENERIC_READ_ACCESS_MASK,
        "launch_intent_child_duplicate_target_access_mask": (
            GENERIC_READ_ACCESS_REQUEST
        ),
        "launch_intent_child_expected_granted_access_mask": (
            FILE_GENERIC_READ_ACCESS_MASK
        ),
        "launch_intent_child_duplicate_options": 0,
        "launch_intent_child_duplicate_close_source": False,
        "child_stdio_policy": TERMINAL_CLIENT_CHILD_STDIO_POLICY,
        "child_stdin_source": "fresh_readonly_NUL_handle_v1",
        "child_stdout_transport": "fresh_anonymous_pipe_v1",
        "child_stderr_transport": "fresh_anonymous_pipe_v1",
        "child_inherited_handle_list_policy": (
            "exact_stdin_stdout_stderr_only_proc_thread_attribute_handle_list_v1"
        ),
        "child_inherited_handle_count": 3,
        "createprocess_inherit_handles": True,
        "startupinfoex_use_std_handles_required": True,
        "proc_thread_attribute_handle_list_required": True,
        "non_stdio_inherited_handles_allowed": False,
        "preterminal_or_terminal_input_file_handles_inherited_allowed": False,
        "child_stdout_max_bytes": TERMINAL_CUSTODY_OUTBOUND_MAX_BYTES,
        "child_stderr_max_bytes": TERMINAL_CUSTODY_OUTBOUND_MAX_BYTES,
        "child_stdout_summary_policy": COMPOSED_TERMINAL_STDOUT_SUMMARY_POLICY,
        "child_stdout_single_canonical_json_line_required": True,
        "child_stderr_empty_required": True,
        "stdio_pipe_drains_event_driven_concurrent_required": True,
        "launcher_forwards_validated_child_stdout_once_required": True,
        "sealed_input_allowlist": [
            "supervisor_spec",
            "E",
        ],
        "project_import_allowed": False,
        "inherited_environment_for_child_allowed": False,
        "createprocessw_exact_child_required": True,
        "child_environment_encoding": "sorted_utf16le_double_nul_block_v1",
        "child_environment_source": "sealed_E.expected_launch_environment.child_environment",
        "verify_terminal_child_launch_topology": (
            "launcher_base_direct_to_venv_redirector_to_runtime_child_v1"
        ),
        "verify_terminal_immediate_redirector_program_path": str(
            canonical_capsule.python_path
        ),
        "verify_terminal_immediate_redirector_program_sha256": (
            canonical_capsule.python_sha256
        ),
        "verify_terminal_runtime_child_program_path": str(
            canonical_capsule.runtime_python_path
        ),
        "verify_terminal_runtime_child_program_sha256": (
            canonical_capsule.runtime_python_sha256
        ),
        "launcher_is_runtime_child_grandparent_required": True,
        "same_job_no_breakaway_required": True,
        "launcher_waits_for_child_exit_required": True,
        "launcher_parent_live_through_child_exit_required": True,
        "supervisor_retains_launcher_redirector_child_process_handles_through_final_ack_required": True,
        "immediate_venv_redirector_live_through_runtime_child_exit_required": True,
        "terminal_client_launcher_live_through_redirector_waitforexit_required": True,
        "process_liveness_reverified_at_final_ack_required": True,
        "automatic_retry_allowed": False,
        "fallback_allowed": False,
    }
    return {
        **unsigned,
        "release_root_sha256": canonical_json_sha256(unsigned),
    }


def canonical_original_confirmatory_terminal_client_launcher_release(
    value: Mapping[str, Any],
    *,
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, role="terminal-client launcher release")
    identity = _mapping(
        raw.get("source_physical_identity"),
        role="terminal-client launcher release identity",
    )
    expected = build_original_confirmatory_terminal_client_launcher_release(
        capsule,
        supervisor_root=cast(str, raw.get("supervisor_root")),
        source_physical_identity=identity,
        source_ancestor_lease=_mapping(
            raw.get("source_ancestor_lease"),
            role="terminal-client launcher release ancestor lease",
        ),
    )
    if set(
        raw
    ) != _TERMINAL_CLIENT_LAUNCHER_RELEASE_FIELDS or not _strict_json_value_equal(
        raw, expected
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "terminal-client launcher release violates its exact policy"
        )
    return expected


_SUPERVISOR_PROCESS_COMMAND_DERIVATION_CONTRACT_FIELDS = {
    "schema_version",
    "policy",
    "supervisor_code_root",
    "supervisor_state_root",
    "supervisor_source_path",
    "supervisor_source_sha256",
    "supervisor_launcher_path",
    "supervisor_launcher_sha256",
    "program_path",
    "program_sha256",
    "createprocess_application_path",
    "createprocess_application_sha256",
    "logical_venv_python_path",
    "logical_venv_python_sha256",
    "runtime_python_path",
    "runtime_python_sha256",
    "expected_live_image_path",
    "expected_live_image_sha256",
    "direct_base_runtime_live_parity_policy",
    "python_interpreter_flags",
    "python_sys_argv_prefix",
    "supervisor_launch_spec_path_binding",
    "staged_e_intent_path_binding",
    "os_launch_vector_policy",
    "cwd",
    "command_preimage_policy",
    "command_preimage_field_names",
    "command_sha256_policy",
    "peb_command_line_exact_direct_base_runtime_match_required",
    "in_process_sys_argv_exact_match_required",
    "isolated_flag_required",
    "no_site_flag_required",
    "dont_write_bytecode_flag_required",
    "logical_venv_identity_separately_bound_required",
    "supervisor_launcher_role",
    "supervisor_launcher_used_for_authorized_process_launch",
    "extra_argv_allowed",
    "extra_cwd_allowed",
    "contract_sha256",
}

_SUPERVISOR_PROCESS_COMMAND_PROJECTION_FIELDS = {
    "schema_version",
    "policy",
    "derivation_contract_sha256",
    "program_path",
    "program_sha256",
    "createprocess_application_path",
    "createprocess_application_sha256",
    "logical_venv_python_path",
    "logical_venv_python_sha256",
    "runtime_python_path",
    "runtime_python_sha256",
    "python_interpreter_flags",
    "python_sys_argv",
    "os_launch_vector",
    "expected_live_image_path",
    "expected_live_image_sha256",
    "expected_live_peb_argv",
    "direct_base_runtime_live_parity_policy",
    "cwd",
    "supervisor_source_path",
    "supervisor_source_sha256",
    "supervisor_launcher_path",
    "supervisor_launcher_sha256",
    "command_preimage",
    "command_sha256",
    "projection_root_sha256",
}

_SUPERVISOR_PROCESS_COMMAND_PREIMAGE_FIELDS = {
    "schema_version",
    "policy",
    "program_path",
    "program_sha256",
    "python_interpreter_flags",
    "python_sys_argv",
    "os_launch_vector",
    "cwd",
    "supervisor_source_path",
    "supervisor_source_sha256",
}


def build_original_confirmatory_supervisor_process_command_derivation_contract(
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    *,
    supervisor_code_root: str | Path,
    supervisor_state_root: str | Path,
    supervisor_source_path: str | Path,
    supervisor_source_sha256: str,
    supervisor_launcher_path: str | Path,
    supervisor_launcher_sha256: str,
) -> dict[str, Any]:
    """Build Q's acyclic exact supervisor process-command derivation."""

    canonical_capsule = canonical_original_confirmatory_execution_capsule(capsule)
    code_root = Path(
        _absolute_path(
            str(supervisor_code_root),
            role="supervisor process command code root",
        )
    )
    state_root = Path(
        _absolute_path(
            str(supervisor_state_root),
            role="supervisor process command state root",
        )
    )
    source_path = Path(
        _absolute_path(
            str(supervisor_source_path),
            role="supervisor process command source",
        )
    )
    launcher_path = Path(
        _absolute_path(
            str(supervisor_launcher_path),
            role="supervisor process command wrapper launcher",
        )
    )
    if (
        source_path != code_root / "aanca_supervisor.py"
        or launcher_path != code_root / "launch_hidden.ps1"
        or _paths_overlap(code_root, state_root)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "supervisor process source/launcher/state paths violate their split-root policy"
        )
    unsigned: dict[str, Any] = {
        "schema_version": 2,
        "policy": SUPERVISOR_PROCESS_COMMAND_DERIVATION_POLICY,
        "supervisor_code_root": str(code_root),
        "supervisor_state_root": str(state_root),
        "supervisor_source_path": str(source_path),
        "supervisor_source_sha256": _sha256(
            supervisor_source_sha256,
            role="supervisor process command source",
        ),
        "supervisor_launcher_path": str(launcher_path),
        "supervisor_launcher_sha256": _sha256(
            supervisor_launcher_sha256,
            role="supervisor process command wrapper launcher",
        ),
        "program_path": str(canonical_capsule.runtime_python_path),
        "program_sha256": canonical_capsule.runtime_python_sha256,
        "createprocess_application_path": str(canonical_capsule.runtime_python_path),
        "createprocess_application_sha256": canonical_capsule.runtime_python_sha256,
        "logical_venv_python_path": str(canonical_capsule.python_path),
        "logical_venv_python_sha256": canonical_capsule.python_sha256,
        "runtime_python_path": str(canonical_capsule.runtime_python_path),
        "runtime_python_sha256": canonical_capsule.runtime_python_sha256,
        "expected_live_image_path": str(canonical_capsule.runtime_python_path),
        "expected_live_image_sha256": canonical_capsule.runtime_python_sha256,
        "direct_base_runtime_live_parity_policy": (
            DIRECT_BASE_RUNTIME_LIVE_PARITY_POLICY
        ),
        "python_interpreter_flags": list(SUPERVISOR_PYTHON_ISOLATED_FLAGS),
        "python_sys_argv_prefix": [
            str(source_path),
            "--root",
            str(state_root),
            "run",
        ],
        "supervisor_launch_spec_path_binding": (
            "Q.control_staging_projection.supervisor_launch_spec_path"
        ),
        "staged_e_intent_path_binding": "Q.control_staging_projection.e_intent_path",
        "os_launch_vector_policy": (
            "program_path_then_python_flags_then_exact_option_a_staged_argv_v2"
        ),
        "cwd": str(canonical_capsule.capsule_ancestor_lease.anchor_path),
        "command_preimage_policy": SUPERVISOR_PROCESS_COMMAND_POLICY,
        "command_preimage_field_names": sorted(
            _SUPERVISOR_PROCESS_COMMAND_PREIMAGE_FIELDS
        ),
        "command_sha256_policy": SUPERVISOR_PROCESS_COMMAND_HASH_POLICY,
        "peb_command_line_exact_direct_base_runtime_match_required": True,
        "in_process_sys_argv_exact_match_required": True,
        "isolated_flag_required": 1,
        "no_site_flag_required": 1,
        "dont_write_bytecode_flag_required": 1,
        "logical_venv_identity_separately_bound_required": True,
        "supervisor_launcher_role": "nonexecuted_install_or_manual_recovery_helper",
        "supervisor_launcher_used_for_authorized_process_launch": False,
        "extra_argv_allowed": False,
        "extra_cwd_allowed": False,
    }
    return {
        **unsigned,
        "contract_sha256": canonical_json_sha256(unsigned),
    }


def canonical_original_confirmatory_supervisor_process_command_derivation_contract(
    value: Mapping[str, Any],
    *,
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, role="supervisor process command derivation")
    expected = (
        build_original_confirmatory_supervisor_process_command_derivation_contract(
            capsule,
            supervisor_code_root=cast(str, raw.get("supervisor_code_root")),
            supervisor_state_root=cast(str, raw.get("supervisor_state_root")),
            supervisor_source_path=cast(str, raw.get("supervisor_source_path")),
            supervisor_source_sha256=cast(str, raw.get("supervisor_source_sha256")),
            supervisor_launcher_path=cast(str, raw.get("supervisor_launcher_path")),
            supervisor_launcher_sha256=cast(
                str,
                raw.get("supervisor_launcher_sha256"),
            ),
        )
    )
    if (
        set(raw) != _SUPERVISOR_PROCESS_COMMAND_DERIVATION_CONTRACT_FIELDS
        or not _strict_json_value_equal(raw, expected)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "supervisor process command derivation violates its exact policy"
        )
    return expected


def build_original_confirmatory_supervisor_process_command_projection(
    derivation_contract: Mapping[str, Any],
    *,
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    supervisor_launch_spec_path: str | Path,
    staged_e_intent_path: str | Path,
) -> dict[str, Any]:
    """Materialize one exact downstream supervisor process command."""

    derivation = (
        canonical_original_confirmatory_supervisor_process_command_derivation_contract(
            derivation_contract,
            capsule=capsule,
        )
    )
    spec_path = Path(
        _absolute_path(
            str(supervisor_launch_spec_path),
            role="supervisor process staged launch spec",
        )
    )
    e_intent_path = Path(
        _absolute_path(
            str(staged_e_intent_path),
            role="supervisor process staged E intent",
        )
    )
    state_root = Path(derivation["supervisor_state_root"])
    stage_dir = spec_path.parent
    if (
        spec_path.name != "supervisor_launch_spec.json"
        or e_intent_path != stage_dir / E_INTENT_FILENAME
        or stage_dir.parent != state_root / CONTROL_STAGING_DIRECTORY_NAME
        or not stage_dir.name
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "supervisor process staged inputs violate the exact Option-A state layout"
        )
    python_sys_argv = [
        *derivation["python_sys_argv_prefix"],
        SUPERVISOR_STAGED_LAUNCH_SPEC_FLAG,
        str(spec_path),
        SUPERVISOR_STAGED_E_INTENT_FLAG,
        str(e_intent_path),
    ]
    os_launch_vector = [
        derivation["program_path"],
        *derivation["python_interpreter_flags"],
        *python_sys_argv,
    ]
    expected_live_peb_argv = [
        derivation["runtime_python_path"],
        *os_launch_vector[1:],
    ]
    preimage: dict[str, Any] = {
        "schema_version": 1,
        "policy": SUPERVISOR_PROCESS_COMMAND_POLICY,
        "program_path": derivation["program_path"],
        "program_sha256": derivation["program_sha256"],
        "python_interpreter_flags": derivation["python_interpreter_flags"],
        "python_sys_argv": python_sys_argv,
        "os_launch_vector": os_launch_vector,
        "cwd": derivation["cwd"],
        "supervisor_source_path": derivation["supervisor_source_path"],
        "supervisor_source_sha256": derivation["supervisor_source_sha256"],
    }
    command_sha256 = canonical_json_sha256(preimage)
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "policy": SUPERVISOR_PROCESS_COMMAND_PROJECTION_POLICY,
        "derivation_contract_sha256": derivation["contract_sha256"],
        "program_path": derivation["program_path"],
        "program_sha256": derivation["program_sha256"],
        "createprocess_application_path": derivation["createprocess_application_path"],
        "createprocess_application_sha256": derivation[
            "createprocess_application_sha256"
        ],
        "logical_venv_python_path": derivation["logical_venv_python_path"],
        "logical_venv_python_sha256": derivation["logical_venv_python_sha256"],
        "runtime_python_path": derivation["runtime_python_path"],
        "runtime_python_sha256": derivation["runtime_python_sha256"],
        "python_interpreter_flags": derivation["python_interpreter_flags"],
        "python_sys_argv": python_sys_argv,
        "os_launch_vector": os_launch_vector,
        "expected_live_image_path": derivation["expected_live_image_path"],
        "expected_live_image_sha256": derivation["expected_live_image_sha256"],
        "expected_live_peb_argv": expected_live_peb_argv,
        "direct_base_runtime_live_parity_policy": derivation[
            "direct_base_runtime_live_parity_policy"
        ],
        "cwd": derivation["cwd"],
        "supervisor_source_path": derivation["supervisor_source_path"],
        "supervisor_source_sha256": derivation["supervisor_source_sha256"],
        "supervisor_launcher_path": derivation["supervisor_launcher_path"],
        "supervisor_launcher_sha256": derivation["supervisor_launcher_sha256"],
        "command_preimage": preimage,
        "command_sha256": command_sha256,
    }
    return {
        **unsigned,
        "projection_root_sha256": canonical_json_sha256(unsigned),
    }


def canonical_original_confirmatory_supervisor_process_command_projection(
    value: Mapping[str, Any],
    *,
    derivation_contract: Mapping[str, Any],
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    supervisor_launch_spec_path: str | Path,
    staged_e_intent_path: str | Path,
) -> dict[str, Any]:
    raw = _mapping(value, role="supervisor process command projection")
    expected = build_original_confirmatory_supervisor_process_command_projection(
        derivation_contract,
        capsule=capsule,
        supervisor_launch_spec_path=supervisor_launch_spec_path,
        staged_e_intent_path=staged_e_intent_path,
    )
    if (
        set(raw) != _SUPERVISOR_PROCESS_COMMAND_PROJECTION_FIELDS
        or set(
            _mapping(raw.get("command_preimage"), role="supervisor command preimage")
        )
        != _SUPERVISOR_PROCESS_COMMAND_PREIMAGE_FIELDS
        or not _strict_json_value_equal(raw, expected)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "supervisor process command projection violates its exact policy"
        )
    return expected


_SUPERVISOR_RELEASE_FIELDS = {
    "schema_version",
    "policy",
    "supervisor_policy",
    "supervisor_spec_schema_version",
    "external_control_plane_release_root_sha256",
    "external_control_plane_publication_id",
    "external_control_plane_release_qualification_attestation_path",
    "external_control_plane_release_qualification_attestation_file_sha256",
    "external_control_plane_release_qualification_attestation_root_sha256",
    "supervisor_code_root",
    "supervisor_state_root",
    "supervisor_source_path",
    "supervisor_source_sha256",
    "supervisor_launcher_path",
    "supervisor_launcher_sha256",
    "supervisor_program_path",
    "supervisor_program_sha256",
    "supervisor_runtime_python_path",
    "supervisor_runtime_python_sha256",
    "supervisor_process_command_derivation_contract",
    "supervisor_process_command_derivation_contract_sha256",
    "terminal_client_launcher_release",
    "terminal_client_launcher_release_root_sha256",
    "supervisor_release_root_sha256",
    "postwake_custody_seed_policy",
    "postwake_custody_pipe_derivation_policy",
    "q_e_custody_contract_policy",
    "q_e_custody_handoff_policy",
    "q_e_custody_transport",
    "q_e_custody_ready_message_type",
    "q_e_custody_ack_policy",
    "q_e_custody_ack_message_type",
    "q_e_custody_receipt_policy",
    "q_e_custody_receipt_filename",
    "q_e_custody_ready_max_bytes",
    "q_e_custody_ack_max_bytes",
    "q_e_independent_verifier_receipt_required",
    "q_e_no_science_before_custody_ack",
    "terminal_custody_authority_template_root_sha256",
    "external_codex_handoff_policy",
    "external_codex_handoff_authority_spec_file_sha256",
    "external_codex_handoff_authority_spec_canonical_root_sha256",
    "internal_codex_wake_disposition",
    "plan_sha256",
    "runtime_release_root_sha256",
    "terminal_release_root_sha256",
    "exact_job_object_membership_required",
    "contract_sha256",
}


def build_original_confirmatory_supervisor_release_binding(
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    *,
    external_control_plane_release_root_sha256: str,
    external_control_plane_publication_id: str,
    external_control_plane_release_qualification_attestation_path: str | Path,
    external_control_plane_release_qualification_attestation_file_sha256: str,
    external_control_plane_release_qualification_attestation_root_sha256: str,
    supervisor_code_root: str | Path,
    supervisor_state_root: str | Path,
    supervisor_source_path: str | Path,
    supervisor_source_sha256: str,
    supervisor_launcher_path: str | Path,
    supervisor_launcher_sha256: str,
    external_codex_handoff_policy: str,
    external_codex_handoff_authority_spec_file_sha256: str,
    external_codex_handoff_authority_spec_canonical_root_sha256: str,
    internal_codex_wake_disposition: str,
    terminal_client_launcher_source_physical_identity: (
        OriginalConfirmatoryPhysicalFileIdentity | Mapping[str, Any]
    ),
    terminal_client_launcher_source_ancestor_lease: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_capsule = canonical_original_confirmatory_execution_capsule(capsule)
    external_release_root_sha256 = _sha256(
        external_control_plane_release_root_sha256,
        role="external control-plane release root",
    )
    publication_id = external_control_plane_publication_id
    if (
        type(publication_id) is not str
        or re.fullmatch(r"cpr-[0-9a-f]{32}", publication_id) is None
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "external control-plane publication id violates its exact policy"
        )
    qualification_attestation_path = Path(
        _absolute_path(
            str(external_control_plane_release_qualification_attestation_path),
            role="external control-plane release qualification attestation",
        )
    )
    if (
        qualification_attestation_path.name
        != EXTERNAL_CONTROL_PLANE_QUALIFICATION_ATTESTATION_FILENAME
        or qualification_attestation_path.parent.name != publication_id
        or qualification_attestation_path.parent.parent.name
        != EXTERNAL_CONTROL_PLANE_VERIFICATIONS_DIRECTORY_NAME
        or qualification_attestation_path.parent.parent.parent.name
        != EXTERNAL_CONTROL_PLANE_RELEASE_DIRECTORY_NAME
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "external control-plane release qualification attestation path violates its exact policy"
        )
    qualification_attestation_file_sha256 = _sha256(
        external_control_plane_release_qualification_attestation_file_sha256,
        role="external control-plane release qualification attestation file",
    )
    qualification_attestation_root_sha256 = _sha256(
        external_control_plane_release_qualification_attestation_root_sha256,
        role="external control-plane release qualification attestation root",
    )
    code_root = Path(
        _absolute_path(
            str(supervisor_code_root),
            role="supervisor release code root",
        )
    )
    state_root = Path(
        _absolute_path(
            str(supervisor_state_root),
            role="supervisor release state root",
        )
    )
    if (
        code_root.name != "supervisor"
        or code_root.parent.name != external_release_root_sha256
        or _paths_overlap(code_root, state_root)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "supervisor release code/state roots violate the content-addressed split policy"
        )
    source_sha256 = _sha256(
        supervisor_source_sha256,
        role="supervisor-v2 source release",
    )
    launcher_sha256 = _sha256(
        supervisor_launcher_sha256,
        role="supervisor-v2 launcher release",
    )
    terminal_custody_template_root = cast(
        str,
        build_original_confirmatory_terminal_custody_authority_template_projection()[
            "template_root_sha256"
        ],
    )
    authority_spec_file_sha256 = _sha256(
        external_codex_handoff_authority_spec_file_sha256,
        role="external Codex handoff authority-spec file",
    )
    authority_spec_canonical_root_sha256 = _sha256(
        external_codex_handoff_authority_spec_canonical_root_sha256,
        role="external Codex handoff authority-spec canonical root",
    )
    if (
        external_codex_handoff_policy != EXTERNAL_CODEX_HANDOFF_POLICY
        or internal_codex_wake_disposition != INTERNAL_CODEX_WAKE_DISPOSITION
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "external Codex handoff release policy differs"
        )
    process_command_derivation = (
        build_original_confirmatory_supervisor_process_command_derivation_contract(
            canonical_capsule,
            supervisor_code_root=code_root,
            supervisor_state_root=state_root,
            supervisor_source_path=supervisor_source_path,
            supervisor_source_sha256=source_sha256,
            supervisor_launcher_path=supervisor_launcher_path,
            supervisor_launcher_sha256=launcher_sha256,
        )
    )
    terminal_client_launcher_release = (
        build_original_confirmatory_terminal_client_launcher_release(
            canonical_capsule,
            supervisor_root=code_root,
            source_physical_identity=terminal_client_launcher_source_physical_identity,
            source_ancestor_lease=terminal_client_launcher_source_ancestor_lease,
        )
    )
    release_root = canonical_json_sha256(
        {
            "policy": SUPERVISOR_V3_POLICY,
            "external_control_plane_release_root_sha256": external_release_root_sha256,
            "external_control_plane_publication_id": publication_id,
            "external_control_plane_release_qualification_attestation_path": str(
                qualification_attestation_path
            ),
            "external_control_plane_release_qualification_attestation_file_sha256": (
                qualification_attestation_file_sha256
            ),
            "external_control_plane_release_qualification_attestation_root_sha256": (
                qualification_attestation_root_sha256
            ),
            "supervisor_code_root": str(code_root),
            "supervisor_state_root": str(state_root),
            "supervisor_source_path": process_command_derivation[
                "supervisor_source_path"
            ],
            "supervisor_source_sha256": source_sha256,
            "supervisor_launcher_path": process_command_derivation[
                "supervisor_launcher_path"
            ],
            "supervisor_launcher_sha256": launcher_sha256,
            "supervisor_program_path": process_command_derivation["program_path"],
            "supervisor_program_sha256": process_command_derivation["program_sha256"],
            "supervisor_runtime_python_path": process_command_derivation[
                "runtime_python_path"
            ],
            "supervisor_runtime_python_sha256": process_command_derivation[
                "runtime_python_sha256"
            ],
            "supervisor_process_command_derivation_contract_sha256": (
                process_command_derivation["contract_sha256"]
            ),
            "terminal_client_launcher_release_root_sha256": (
                terminal_client_launcher_release["release_root_sha256"]
            ),
            "postwake_custody_seed_policy": POSTWAKE_CUSTODY_SEED_POLICY,
            "postwake_custody_pipe_derivation_policy": (
                POSTWAKE_CUSTODY_PIPE_DERIVATION_POLICY
            ),
            "q_e_custody_contract_policy": Q_E_CUSTODY_CONTRACT_POLICY,
            "q_e_custody_handoff_policy": Q_E_CUSTODY_HANDOFF_POLICY,
            "q_e_custody_transport": Q_E_CUSTODY_TRANSPORT,
            "q_e_custody_ack_policy": Q_E_CUSTODY_ACK_POLICY,
            "q_e_custody_receipt_policy": Q_E_CUSTODY_RECEIPT_POLICY,
            "q_e_custody_receipt_filename": Q_E_CUSTODY_RECEIPT_FILENAME,
            "terminal_custody_authority_template_root_sha256": (
                terminal_custody_template_root
            ),
            "external_codex_handoff_policy": external_codex_handoff_policy,
            "external_codex_handoff_authority_spec_file_sha256": (
                authority_spec_file_sha256
            ),
            "external_codex_handoff_authority_spec_canonical_root_sha256": (
                authority_spec_canonical_root_sha256
            ),
            "internal_codex_wake_disposition": internal_codex_wake_disposition,
        }
    )
    unsigned = {
        "schema_version": 1,
        "policy": SUPERVISOR_RELEASE_BINDING_POLICY,
        "supervisor_policy": SUPERVISOR_V3_POLICY,
        "supervisor_spec_schema_version": 3,
        "external_control_plane_release_root_sha256": external_release_root_sha256,
        "external_control_plane_publication_id": publication_id,
        "external_control_plane_release_qualification_attestation_path": str(
            qualification_attestation_path
        ),
        "external_control_plane_release_qualification_attestation_file_sha256": (
            qualification_attestation_file_sha256
        ),
        "external_control_plane_release_qualification_attestation_root_sha256": (
            qualification_attestation_root_sha256
        ),
        "supervisor_code_root": str(code_root),
        "supervisor_state_root": str(state_root),
        "supervisor_source_path": process_command_derivation["supervisor_source_path"],
        "supervisor_source_sha256": source_sha256,
        "supervisor_launcher_path": process_command_derivation[
            "supervisor_launcher_path"
        ],
        "supervisor_launcher_sha256": launcher_sha256,
        "supervisor_program_path": process_command_derivation["program_path"],
        "supervisor_program_sha256": process_command_derivation["program_sha256"],
        "supervisor_runtime_python_path": process_command_derivation[
            "runtime_python_path"
        ],
        "supervisor_runtime_python_sha256": process_command_derivation[
            "runtime_python_sha256"
        ],
        "supervisor_process_command_derivation_contract": process_command_derivation,
        "supervisor_process_command_derivation_contract_sha256": (
            process_command_derivation["contract_sha256"]
        ),
        "terminal_client_launcher_release": terminal_client_launcher_release,
        "terminal_client_launcher_release_root_sha256": (
            terminal_client_launcher_release["release_root_sha256"]
        ),
        "supervisor_release_root_sha256": release_root,
        "postwake_custody_seed_policy": POSTWAKE_CUSTODY_SEED_POLICY,
        "postwake_custody_pipe_derivation_policy": (
            POSTWAKE_CUSTODY_PIPE_DERIVATION_POLICY
        ),
        "q_e_custody_contract_policy": Q_E_CUSTODY_CONTRACT_POLICY,
        "q_e_custody_handoff_policy": Q_E_CUSTODY_HANDOFF_POLICY,
        "q_e_custody_transport": Q_E_CUSTODY_TRANSPORT,
        "q_e_custody_ready_message_type": Q_E_CUSTODY_READY_MESSAGE_TYPE,
        "q_e_custody_ack_policy": Q_E_CUSTODY_ACK_POLICY,
        "q_e_custody_ack_message_type": Q_E_CUSTODY_ACK_MESSAGE_TYPE,
        "q_e_custody_receipt_policy": Q_E_CUSTODY_RECEIPT_POLICY,
        "q_e_custody_receipt_filename": Q_E_CUSTODY_RECEIPT_FILENAME,
        "q_e_custody_ready_max_bytes": Q_E_CUSTODY_LINE_MAX_BYTES,
        "q_e_custody_ack_max_bytes": Q_E_CUSTODY_LINE_MAX_BYTES,
        "q_e_independent_verifier_receipt_required": True,
        "q_e_no_science_before_custody_ack": True,
        "terminal_custody_authority_template_root_sha256": (
            terminal_custody_template_root
        ),
        "external_codex_handoff_policy": external_codex_handoff_policy,
        "external_codex_handoff_authority_spec_file_sha256": (
            authority_spec_file_sha256
        ),
        "external_codex_handoff_authority_spec_canonical_root_sha256": (
            authority_spec_canonical_root_sha256
        ),
        "internal_codex_wake_disposition": internal_codex_wake_disposition,
        "plan_sha256": canonical_capsule.plan_sha256,
        "runtime_release_root_sha256": (canonical_capsule.runtime_release_root_sha256),
        "terminal_release_root_sha256": (
            canonical_capsule.terminal_release_root_sha256
        ),
        "exact_job_object_membership_required": True,
    }
    return {**unsigned, "contract_sha256": canonical_json_sha256(unsigned)}


def canonical_original_confirmatory_supervisor_release_binding(
    value: Mapping[str, Any],
    *,
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, role="Q/E supervisor release binding")
    if set(raw) != _SUPERVISOR_RELEASE_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E supervisor release binding has an unexpected field set"
        )
    expected = build_original_confirmatory_supervisor_release_binding(
        capsule,
        external_control_plane_release_root_sha256=_sha256(
            raw["external_control_plane_release_root_sha256"],
            role="external control-plane release root",
        ),
        external_control_plane_publication_id=cast(
            str,
            raw["external_control_plane_publication_id"],
        ),
        external_control_plane_release_qualification_attestation_path=cast(
            str,
            raw["external_control_plane_release_qualification_attestation_path"],
        ),
        external_control_plane_release_qualification_attestation_file_sha256=_sha256(
            raw["external_control_plane_release_qualification_attestation_file_sha256"],
            role="external control-plane release qualification attestation file",
        ),
        external_control_plane_release_qualification_attestation_root_sha256=_sha256(
            raw["external_control_plane_release_qualification_attestation_root_sha256"],
            role="external control-plane release qualification attestation root",
        ),
        supervisor_code_root=cast(str, raw["supervisor_code_root"]),
        supervisor_state_root=cast(str, raw["supervisor_state_root"]),
        supervisor_source_path=cast(str, raw["supervisor_source_path"]),
        supervisor_source_sha256=_sha256(
            raw["supervisor_source_sha256"],
            role="supervisor-v2 source release",
        ),
        supervisor_launcher_path=cast(str, raw["supervisor_launcher_path"]),
        supervisor_launcher_sha256=_sha256(
            raw["supervisor_launcher_sha256"],
            role="supervisor-v2 launcher release",
        ),
        external_codex_handoff_policy=cast(
            str,
            raw["external_codex_handoff_policy"],
        ),
        external_codex_handoff_authority_spec_file_sha256=_sha256(
            raw["external_codex_handoff_authority_spec_file_sha256"],
            role="external Codex handoff authority-spec file",
        ),
        external_codex_handoff_authority_spec_canonical_root_sha256=_sha256(
            raw["external_codex_handoff_authority_spec_canonical_root_sha256"],
            role="external Codex handoff authority-spec canonical root",
        ),
        internal_codex_wake_disposition=cast(
            str,
            raw["internal_codex_wake_disposition"],
        ),
        terminal_client_launcher_source_physical_identity=_mapping(
            _mapping(
                raw["terminal_client_launcher_release"],
                role="Q/E terminal-client launcher release",
            )["source_physical_identity"],
            role="Q/E terminal-client launcher identity",
        ),
        terminal_client_launcher_source_ancestor_lease=_mapping(
            _mapping(
                raw["terminal_client_launcher_release"],
                role="Q/E terminal-client launcher release",
            )["source_ancestor_lease"],
            role="Q/E terminal-client launcher ancestor lease",
        ),
    )
    if not _strict_json_value_equal(raw, expected):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E supervisor release binding violates its exact policy"
        )
    return expected


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryEConsumptionContract:
    """Closed, acyclic custody contract for the bootstrap-created E claim."""

    claim_path: Path
    custody_receipt_path: Path
    ready_line_max_bytes: int
    ack_line_max_bytes: int
    contract_sha256: str

    def payload_without_self_hash(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": E_CONSUMPTION_CONTRACT_POLICY,
            "claim_policy": E_CONSUMPTION_CLAIM_POLICY,
            "claim_path": str(self.claim_path),
            "custody_receipt_path": str(self.custody_receipt_path),
            "transport": E_CONSUMPTION_TRANSPORT,
            "ready_message_type": E_CONSUMPTION_READY_MESSAGE_TYPE,
            "ack_message_type": E_CONSUMPTION_ACK_MESSAGE_TYPE,
            "ready_line_max_bytes": self.ready_line_max_bytes,
            "ack_line_max_bytes": self.ack_line_max_bytes,
            "duplicate_target_access_mask": (
                E_CONSUMPTION_DUPLICATE_TARGET_ACCESS_MASK
            ),
            "duplicate_options": E_CONSUMPTION_DUPLICATE_OPTIONS,
            "close_source": False,
            "source_handle_retained_through_ack": True,
            "supervisor_handle_retention_policy": (
                E_CONSUMPTION_SUPERVISOR_RETENTION_POLICY
            ),
            "exact_job_object_membership_required": True,
            "exact_supervisor_process_identity_required": True,
            "exact_downstream_spec_rederivation_required": True,
            "scientific_inputs_before_ack_allowed": False,
            "automatic_retry_allowed": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.payload_without_self_hash(),
            "contract_sha256": self.contract_sha256,
        }


_E_CONSUMPTION_CONTRACT_FIELDS = {
    "schema_version",
    "policy",
    "claim_policy",
    "claim_path",
    "custody_receipt_path",
    "transport",
    "ready_message_type",
    "ack_message_type",
    "ready_line_max_bytes",
    "ack_line_max_bytes",
    "duplicate_target_access_mask",
    "duplicate_options",
    "close_source",
    "source_handle_retained_through_ack",
    "supervisor_handle_retention_policy",
    "exact_job_object_membership_required",
    "exact_supervisor_process_identity_required",
    "exact_downstream_spec_rederivation_required",
    "scientific_inputs_before_ack_allowed",
    "automatic_retry_allowed",
    "contract_sha256",
}


def canonical_original_confirmatory_e_consumption_contract(
    value: OriginalConfirmatoryEConsumptionContract | Mapping[str, Any],
    *,
    supervisor_job_directory: str | Path,
) -> OriginalConfirmatoryEConsumptionContract:
    """Validate the exact bootstrap-to-supervisor consumption continuity policy."""

    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatoryEConsumptionContract
        else _mapping(value, role="E consumption contract")
    )
    if set(raw) != _E_CONSUMPTION_CONTRACT_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E consumption contract has an unexpected field set"
        )
    job_dir = Path(
        _absolute_path(
            str(supervisor_job_directory),
            role="E consumption supervisor job directory",
        )
    )
    claim_path = Path(
        _absolute_path(raw["claim_path"], role="E consumption claim path")
    )
    receipt_path = Path(
        _absolute_path(
            raw["custody_receipt_path"],
            role="E consumption custody receipt path",
        )
    )
    ready_bound = _positive_int(
        raw["ready_line_max_bytes"],
        role="E consumption READY line bound",
    )
    ack_bound = _positive_int(
        raw["ack_line_max_bytes"],
        role="E consumption ACK line bound",
    )
    unsigned = {key: item for key, item in raw.items() if key != "contract_sha256"}
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != E_CONSUMPTION_CONTRACT_POLICY
        or raw["claim_policy"] != E_CONSUMPTION_CLAIM_POLICY
        or claim_path != job_dir / E_CONSUMPTION_TOMBSTONE_FILENAME
        or receipt_path != job_dir / E_CONSUMPTION_CUSTODY_RECEIPT_FILENAME
        or raw["transport"] != E_CONSUMPTION_TRANSPORT
        or raw["ready_message_type"] != E_CONSUMPTION_READY_MESSAGE_TYPE
        or raw["ack_message_type"] != E_CONSUMPTION_ACK_MESSAGE_TYPE
        or ready_bound != E_CONSUMPTION_LINE_MAX_BYTES
        or ack_bound != E_CONSUMPTION_LINE_MAX_BYTES
        or raw["duplicate_target_access_mask"]
        != E_CONSUMPTION_DUPLICATE_TARGET_ACCESS_MASK
        or type(raw["duplicate_target_access_mask"]) is not int
        or raw["duplicate_options"] != E_CONSUMPTION_DUPLICATE_OPTIONS
        or type(raw["duplicate_options"]) is not int
        or raw["close_source"] is not False
        or raw["source_handle_retained_through_ack"] is not True
        or raw["supervisor_handle_retention_policy"]
        != E_CONSUMPTION_SUPERVISOR_RETENTION_POLICY
        or raw["exact_job_object_membership_required"] is not True
        or raw["exact_supervisor_process_identity_required"] is not True
        or raw["exact_downstream_spec_rederivation_required"] is not True
        or raw["scientific_inputs_before_ack_allowed"] is not False
        or raw["automatic_retry_allowed"] is not False
        or raw["contract_sha256"] != canonical_json_sha256(unsigned)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E consumption contract violates its exact fail-closed policy"
        )
    return OriginalConfirmatoryEConsumptionContract(
        claim_path=claim_path,
        custody_receipt_path=receipt_path,
        ready_line_max_bytes=ready_bound,
        ack_line_max_bytes=ack_bound,
        contract_sha256=raw["contract_sha256"],
    )


def build_original_confirmatory_e_consumption_contract(
    *,
    supervisor_job_directory: str | Path,
) -> OriginalConfirmatoryEConsumptionContract:
    job_dir = Path(
        _absolute_path(
            str(supervisor_job_directory),
            role="E consumption supervisor job directory",
        )
    )
    provisional = OriginalConfirmatoryEConsumptionContract(
        claim_path=job_dir / E_CONSUMPTION_TOMBSTONE_FILENAME,
        custody_receipt_path=(job_dir / E_CONSUMPTION_CUSTODY_RECEIPT_FILENAME),
        ready_line_max_bytes=E_CONSUMPTION_LINE_MAX_BYTES,
        ack_line_max_bytes=E_CONSUMPTION_LINE_MAX_BYTES,
        contract_sha256="0" * 64,
    )
    unsigned = provisional.payload_without_self_hash()
    return canonical_original_confirmatory_e_consumption_contract(
        {
            **unsigned,
            "contract_sha256": canonical_json_sha256(unsigned),
        },
        supervisor_job_directory=job_dir,
    )


_E_PROCESS_IDENTITY_FIELDS = {
    "pid",
    "creation_time_100ns",
    "creation_time_utc",
    "program_path",
    "program_sha256",
    "command_sha256",
}
_E_CONSUMPTION_CLAIM_FIELDS = {
    "schema_version",
    "policy",
    "disposition",
    "contract_sha256",
    "q_authority_root_sha256",
    "e_intent_path",
    "e_intent_file_sha256",
    "e_intent_core_sha256",
    "supervisor_job_id",
    "attempt_id",
    "run_id",
    "launch_nonce",
    "execution_mode",
    "retry_of_run_id",
    "supervisor_spec_path",
    "supervisor_spec_sha256",
    "process_started_path",
    "process_started_sha256",
    "child_process_identity",
    "supervisor_process_identity",
    "logical_python_path",
    "logical_python_sha256",
    "runtime_python_path",
    "runtime_python_sha256",
    "scientific_inputs_read",
    "automatic_retry_allowed",
    "claim_root_sha256",
}
_E_CONSUMPTION_PHYSICAL_IDENTITY_POLICY = (
    "original_confirmatory_e_intent_consumed_physical_identity_v1"
)
_E_CONSUMPTION_PHYSICAL_IDENTITY_FIELDS = {
    "schema_version",
    "policy",
    "role",
    "path",
    "volume_serial_number",
    "file_id_128",
    "size_bytes",
    "sha256",
    "file_attributes",
    "regular_file",
    "read_only",
    "link_count",
    "named_alternate_data_streams",
    "opened_without_reparse_follow",
    "share_access",
    "write_handle_retained",
    "delete_access",
}
_E_CONSUMPTION_READY_FIELDS = {
    "schema_version",
    "policy",
    "message_type",
    "contract_sha256",
    "claim_path",
    "claim_file_sha256",
    "claim_root_sha256",
    "claim_physical_identity",
    "target_supervisor_handle_value",
    "duplicate_target_access_mask",
    "duplicate_options",
    "close_source",
    "child_process_identity",
    "supervisor_process_identity",
    "supervisor_spec_sha256",
    "process_started_sha256",
    "ready_sha256",
}
_E_CONSUMPTION_CUSTODY_RECEIPT_FIELDS = {
    "schema_version",
    "policy",
    "contract_sha256",
    "ready_sha256",
    "claim_path",
    "claim_file_sha256",
    "claim_root_sha256",
    "claim_physical_identity",
    "target_supervisor_handle_value",
    "target_granted_access_mask",
    "close_source",
    "child_process_identity",
    "supervisor_process_identity",
    "supervisor_spec_sha256",
    "process_started_sha256",
    "exact_job_object_membership_verified",
    "exact_supervisor_process_identity_verified",
    "exact_downstream_spec_rederivation_verified",
    "supervisor_handle_retention_policy",
    "automatic_retry_allowed",
    "created_at_utc",
    "receipt_sha256",
}
_E_CONSUMPTION_ACK_FIELDS = {
    "schema_version",
    "policy",
    "message_type",
    "contract_sha256",
    "ready_sha256",
    "claim_file_sha256",
    "claim_root_sha256",
    "custody_receipt_path",
    "custody_receipt_sha256",
    "supervisor_process_identity",
    "target_supervisor_handle_value",
    "ack_sha256",
}


def _canonical_e_process_identity(value: Any, *, role: str) -> dict[str, Any]:
    raw = _mapping(value, role=role)
    if set(raw) != _E_PROCESS_IDENTITY_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} has an unexpected field set"
        )
    creation_time_100ns = _positive_int(
        raw["creation_time_100ns"],
        role=f"{role} creation time",
    )
    creation_time_utc = _utc_timestamp(
        raw["creation_time_utc"],
        role=f"{role} creation time",
    )
    if creation_time_utc != _windows_filetime_100ns_to_utc(creation_time_100ns):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} UTC creation time differs from its FILETIME"
        )
    return {
        "pid": _positive_int(raw["pid"], role=f"{role} PID"),
        "creation_time_100ns": creation_time_100ns,
        "creation_time_utc": creation_time_utc,
        "program_path": _absolute_path(
            raw["program_path"],
            role=f"{role} program",
        ),
        "program_sha256": _sha256(
            raw["program_sha256"],
            role=f"{role} program",
        ),
        "command_sha256": _sha256(
            raw["command_sha256"],
            role=f"{role} command",
        ),
    }


def canonical_original_confirmatory_e_consumption_claim(
    value: Mapping[str, Any],
    *,
    contract: OriginalConfirmatoryEConsumptionContract | Mapping[str, Any],
    e_intent: Mapping[str, Any],
    q_authority: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, role="E consumption claim")
    if set(raw) != _E_CONSUMPTION_CLAIM_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E consumption claim has an unexpected field set"
        )
    q = canonical_original_confirmatory_q_replacement_v2(q_authority)
    e = canonical_original_confirmatory_e_intent(
        e_intent,
        q_authority=q,
    )
    canonical_contract = canonical_original_confirmatory_e_consumption_contract(
        contract,
        supervisor_job_directory=e["job"]["supervisor_job_dir"],
    )
    child = _canonical_e_process_identity(
        raw["child_process_identity"],
        role="E claim child process identity",
    )
    supervisor = _canonical_e_process_identity(
        raw["supervisor_process_identity"],
        role="E claim supervisor process identity",
    )
    job_dir = Path(e["job"]["supervisor_job_dir"])
    unsigned = {key: item for key, item in raw.items() if key != "claim_root_sha256"}
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != E_CONSUMPTION_CLAIM_POLICY
        or raw["disposition"] != E_CONSUMPTION_CLAIM_DISPOSITION
        or raw["contract_sha256"] != canonical_contract.contract_sha256
        or raw["q_authority_root_sha256"] != q["q_authority_root_sha256"]
        or raw["e_intent_path"] != q["control_staging_projection"]["e_intent_path"]
        or raw["e_intent_file_sha256"] != canonical_json_line_sha256(e)
        or raw["e_intent_core_sha256"] != e["intent_core_sha256"]
        or raw["supervisor_job_id"] != e["job"]["job_id"]
        or raw["attempt_id"] != e["job"]["attempt_id"]
        or raw["run_id"] != e["job"]["run_id"]
        or raw["launch_nonce"] != e["job"]["launch_nonce"]
        or raw["execution_mode"] != e["lineage"]["execution_mode"]
        or raw["retry_of_run_id"] != e["lineage"]["retry_of_run_id"]
        or raw["supervisor_spec_path"] != str(job_dir / "run_spec.json")
        or _SHA256.fullmatch(raw["supervisor_spec_sha256"]) is None
        or raw["process_started_path"] != str(job_dir / "process_started.json")
        or _SHA256.fullmatch(raw["process_started_sha256"]) is None
        or child["program_path"]
        != e["command_projections"][CAPSULE_SCIENTIFIC_MODE]["program_path"]
        or child["program_sha256"]
        != e["command_projections"][CAPSULE_SCIENTIFIC_MODE]["program_sha256"]
        or raw["logical_python_path"] != q["execution_capsule"]["python_path"]
        or raw["logical_python_sha256"] != q["execution_capsule"]["python_sha256"]
        or raw["runtime_python_path"] != q["execution_capsule"]["runtime_python_path"]
        or raw["runtime_python_sha256"]
        != q["execution_capsule"]["runtime_python_sha256"]
        or raw["scientific_inputs_read"] is not False
        or raw["automatic_retry_allowed"] is not False
        or raw["claim_root_sha256"] != canonical_json_sha256(unsigned)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E consumption claim violates its exact pre-science policy"
        )
    return {
        **raw,
        "child_process_identity": child,
        "supervisor_process_identity": supervisor,
    }


def canonical_original_confirmatory_e_consumption_physical_identity(
    value: Mapping[str, Any],
    *,
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, role="E consumption claim physical identity")
    if set(raw) != _E_CONSUMPTION_PHYSICAL_IDENTITY_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E claim physical identity has an unexpected field set"
        )
    claim_path = Path(claim["supervisor_spec_path"]).parent / (
        E_CONSUMPTION_TOMBSTONE_FILENAME
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _E_CONSUMPTION_PHYSICAL_IDENTITY_POLICY
        or raw["role"] != "e-intent-consumed-claim"
        or raw["path"] != str(claim_path)
        or type(raw["volume_serial_number"]) is not int
        or raw["volume_serial_number"] < 0
        or not isinstance(raw["file_id_128"], str)
        or re.fullmatch(r"[0-9a-f]{32}", raw["file_id_128"]) is None
        or type(raw["size_bytes"]) is not int
        or raw["size_bytes"] <= 0
        or _SHA256.fullmatch(raw["sha256"]) is None
        or raw["sha256"] != canonical_json_line_sha256(claim)
        or type(raw["file_attributes"]) is not int
        or raw["regular_file"] is not True
        or raw["read_only"] is not True
        or raw["link_count"] != 1
        or type(raw["link_count"]) is not int
        or not _strict_json_value_equal(raw["named_alternate_data_streams"], [])
        or raw["opened_without_reparse_follow"] is not True
        or not _strict_json_value_equal(raw["share_access"], ["FILE_SHARE_READ"])
        or raw["write_handle_retained"] is not True
        or raw["delete_access"] is not False
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E claim physical identity violates its exact immutable policy"
        )
    return dict(raw)


def canonical_original_confirmatory_e_consumption_ready(
    value: Mapping[str, Any],
    *,
    contract: OriginalConfirmatoryEConsumptionContract | Mapping[str, Any],
    claim: Mapping[str, Any],
    claim_physical_identity: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, role="E consumption READY")
    if set(raw) != _E_CONSUMPTION_READY_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E consumption READY has an unexpected field set"
        )
    job_dir = Path(claim["supervisor_spec_path"]).parent
    canonical_contract = canonical_original_confirmatory_e_consumption_contract(
        contract,
        supervisor_job_directory=job_dir,
    )
    identity = canonical_original_confirmatory_e_consumption_physical_identity(
        claim_physical_identity,
        claim=claim,
    )
    child = _canonical_e_process_identity(
        raw["child_process_identity"],
        role="E READY child process identity",
    )
    supervisor = _canonical_e_process_identity(
        raw["supervisor_process_identity"],
        role="E READY supervisor process identity",
    )
    unsigned = {key: item for key, item in raw.items() if key != "ready_sha256"}
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != E_CONSUMPTION_READY_POLICY
        or raw["message_type"] != E_CONSUMPTION_READY_MESSAGE_TYPE
        or raw["contract_sha256"] != canonical_contract.contract_sha256
        or raw["claim_path"] != str(canonical_contract.claim_path)
        or raw["claim_file_sha256"] != identity["sha256"]
        or raw["claim_root_sha256"] != claim["claim_root_sha256"]
        or not _strict_json_value_equal(raw["claim_physical_identity"], identity)
        or type(raw["target_supervisor_handle_value"]) is not int
        or raw["target_supervisor_handle_value"] <= 0
        or raw["duplicate_target_access_mask"]
        != E_CONSUMPTION_DUPLICATE_TARGET_ACCESS_MASK
        or type(raw["duplicate_target_access_mask"]) is not int
        or raw["duplicate_options"] != E_CONSUMPTION_DUPLICATE_OPTIONS
        or type(raw["duplicate_options"]) is not int
        or raw["close_source"] is not False
        or not _strict_json_value_equal(child, claim["child_process_identity"])
        or not _strict_json_value_equal(
            supervisor,
            claim["supervisor_process_identity"],
        )
        or raw["supervisor_spec_sha256"] != claim["supervisor_spec_sha256"]
        or raw["process_started_sha256"] != claim["process_started_sha256"]
        or raw["ready_sha256"] != canonical_json_sha256(unsigned)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E consumption READY violates its exact custody policy"
        )
    return {
        **raw,
        "child_process_identity": child,
        "supervisor_process_identity": supervisor,
        "claim_physical_identity": identity,
    }


def canonical_original_confirmatory_e_consumption_custody_receipt(
    value: Mapping[str, Any],
    *,
    contract: OriginalConfirmatoryEConsumptionContract | Mapping[str, Any],
    ready: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, role="E consumption custody receipt")
    if set(raw) != _E_CONSUMPTION_CUSTODY_RECEIPT_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E custody receipt has an unexpected field set"
        )
    job_dir = Path(raw["claim_path"]).parent
    canonical_contract = canonical_original_confirmatory_e_consumption_contract(
        contract,
        supervisor_job_directory=job_dir,
    )
    child = _canonical_e_process_identity(
        raw["child_process_identity"],
        role="E receipt child process identity",
    )
    supervisor = _canonical_e_process_identity(
        raw["supervisor_process_identity"],
        role="E receipt supervisor process identity",
    )
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != E_CONSUMPTION_CUSTODY_RECEIPT_POLICY
        or raw["contract_sha256"] != canonical_contract.contract_sha256
        or raw["ready_sha256"] != ready["ready_sha256"]
        or raw["claim_path"] != ready["claim_path"]
        or raw["claim_file_sha256"] != ready["claim_file_sha256"]
        or raw["claim_root_sha256"] != ready["claim_root_sha256"]
        or not _strict_json_value_equal(
            raw["claim_physical_identity"],
            ready["claim_physical_identity"],
        )
        or raw["target_supervisor_handle_value"]
        != ready["target_supervisor_handle_value"]
        or raw["target_granted_access_mask"]
        != E_CONSUMPTION_MAPPED_FILE_GENERIC_READ_ACCESS_MASK
        or type(raw["target_granted_access_mask"]) is not int
        or raw["close_source"] is not False
        or not _strict_json_value_equal(child, ready["child_process_identity"])
        or not _strict_json_value_equal(
            supervisor,
            ready["supervisor_process_identity"],
        )
        or raw["supervisor_spec_sha256"] != ready["supervisor_spec_sha256"]
        or raw["process_started_sha256"] != ready["process_started_sha256"]
        or raw["exact_job_object_membership_verified"] is not True
        or raw["exact_supervisor_process_identity_verified"] is not True
        or raw["exact_downstream_spec_rederivation_verified"] is not True
        or raw["supervisor_handle_retention_policy"]
        != E_CONSUMPTION_SUPERVISOR_RETENTION_POLICY
        or raw["automatic_retry_allowed"] is not False
        or _utc_timestamp(
            raw["created_at_utc"],
            role="E custody receipt timestamp",
        )
        != raw["created_at_utc"]
        or raw["receipt_sha256"] != canonical_json_sha256(unsigned)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E custody receipt violates its exact retained-handle policy"
        )
    return {
        **raw,
        "child_process_identity": child,
        "supervisor_process_identity": supervisor,
    }


def canonical_original_confirmatory_e_consumption_ack(
    value: Mapping[str, Any],
    *,
    contract: OriginalConfirmatoryEConsumptionContract | Mapping[str, Any],
    ready: Mapping[str, Any],
    custody_receipt: Mapping[str, Any],
    custody_receipt_file_sha256: str,
) -> dict[str, Any]:
    raw = _mapping(value, role="E consumption ACK")
    if set(raw) != _E_CONSUMPTION_ACK_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E consumption ACK has an unexpected field set"
        )
    job_dir = Path(ready["claim_path"]).parent
    canonical_contract = canonical_original_confirmatory_e_consumption_contract(
        contract,
        supervisor_job_directory=job_dir,
    )
    receipt = canonical_original_confirmatory_e_consumption_custody_receipt(
        custody_receipt,
        contract=canonical_contract,
        ready=ready,
    )
    supervisor = _canonical_e_process_identity(
        raw["supervisor_process_identity"],
        role="E ACK supervisor process identity",
    )
    unsigned = {key: item for key, item in raw.items() if key != "ack_sha256"}
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != E_CONSUMPTION_ACK_POLICY
        or raw["message_type"] != E_CONSUMPTION_ACK_MESSAGE_TYPE
        or raw["contract_sha256"] != canonical_contract.contract_sha256
        or raw["ready_sha256"] != ready["ready_sha256"]
        or raw["claim_file_sha256"] != ready["claim_file_sha256"]
        or raw["claim_root_sha256"] != ready["claim_root_sha256"]
        or raw["custody_receipt_path"] != str(canonical_contract.custody_receipt_path)
        or raw["custody_receipt_sha256"]
        != _sha256(
            custody_receipt_file_sha256,
            role="E custody receipt file",
        )
        or raw["custody_receipt_sha256"] != canonical_json_line_sha256(receipt)
        or not _strict_json_value_equal(
            supervisor,
            ready["supervisor_process_identity"],
        )
        or raw["target_supervisor_handle_value"]
        != ready["target_supervisor_handle_value"]
        or raw["ack_sha256"] != canonical_json_sha256(unsigned)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E consumption ACK violates its exact one-shot policy"
        )
    return {**raw, "supervisor_process_identity": supervisor}


_CONTROL_PUBLICATION_ANCESTOR_LEASE_FIELDS = {
    "schema_version",
    "policy",
    "project_root",
    "records",
    "record_count",
    "records_root_sha256",
    "directory_access_mask",
    "share_access",
    "delete_share",
    "write_access",
    "owner_process_identity_required",
    "handle_slot_per_record_required",
    "continuous_overlap_through_independent_verification_required",
    "continuous_overlap_into_supervisor_required",
    "acquisition_disposition",
}
_CONTROL_PUBLICATION_ANCESTOR_DISPOSITION = (
    "opened_before_q_create_new_retained_through_verifier_and_supervisor_overlap_v1"
)


def canonical_original_confirmatory_control_publication_ancestor_lease(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, role="Q publication ancestor lease")
    if set(raw) != _CONTROL_PUBLICATION_ANCESTOR_LEASE_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q publication ancestor lease has an unexpected field set"
        )
    root = Path(
        _absolute_path(
            raw["project_root"],
            role="Q publication ancestor project root",
        )
    )
    items = raw["records"]
    expected_paths = (
        root,
        root / "artifacts",
        root / "artifacts" / "resource_control",
    )
    records: list[dict[str, Any]] = []
    if not isinstance(items, list) or len(items) != len(expected_paths):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q publication ancestor lease must contain its exact three directories"
        )
    for index, (item, expected_path) in enumerate(
        zip(items, expected_paths, strict=True),
        start=1,
    ):
        record = _mapping(item, role=f"Q publication ancestor record {index}")
        if set(record) != _CAPSULE_ANCESTOR_RECORD_FIELDS:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "Q publication ancestor record has an unexpected field set"
            )
        file_id = record["file_id_128"]
        attributes = record["file_attributes"]
        if (
            Path(
                _absolute_path(
                    record["path"],
                    role=f"Q publication ancestor record {index} path",
                )
            )
            != expected_path
            or not isinstance(file_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", file_id) is None
            or type(attributes) is not int
            or attributes < 0
            or attributes & 0x400
            or record["reparse_point"] is not False
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "Q publication ancestor record violates its exact identity"
            )
        records.append(
            {
                "path": str(expected_path),
                "volume_serial_number": _nonnegative_int(
                    record["volume_serial_number"],
                    role=f"Q publication ancestor record {index} volume",
                ),
                "file_id_128": file_id,
                "file_attributes": attributes,
                "reparse_point": False,
            }
        )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != CONTROL_PUBLICATION_ANCESTOR_LEASE_POLICY
        or raw["record_count"] != len(records)
        or type(raw["record_count"]) is not int
        or raw["records_root_sha256"] != canonical_json_sha256(records)
        or raw["directory_access_mask"] != EXECUTABLE_ANCESTOR_ACCESS_MASK
        or type(raw["directory_access_mask"]) is not int
        or not _strict_json_value_equal(
            raw["share_access"],
            ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        )
        or raw["delete_share"] is not False
        or raw["write_access"] is not False
        or raw["owner_process_identity_required"] is not True
        or raw["handle_slot_per_record_required"] is not True
        or raw["continuous_overlap_through_independent_verification_required"]
        is not True
        or raw["continuous_overlap_into_supervisor_required"] is not True
        or raw["acquisition_disposition"] != _CONTROL_PUBLICATION_ANCESTOR_DISPOSITION
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q publication ancestor lease violates its exact custody policy"
        )
    return {
        "schema_version": 1,
        "policy": CONTROL_PUBLICATION_ANCESTOR_LEASE_POLICY,
        "project_root": str(root),
        "records": records,
        "record_count": len(records),
        "records_root_sha256": canonical_json_sha256(records),
        "directory_access_mask": EXECUTABLE_ANCESTOR_ACCESS_MASK,
        "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        "delete_share": False,
        "write_access": False,
        "owner_process_identity_required": True,
        "handle_slot_per_record_required": True,
        "continuous_overlap_through_independent_verification_required": True,
        "continuous_overlap_into_supervisor_required": True,
        "acquisition_disposition": _CONTROL_PUBLICATION_ANCESTOR_DISPOSITION,
    }


def build_original_confirmatory_control_publication_ancestor_lease(
    *,
    project_root: str | Path,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    canonical_records = [dict(item) for item in records]
    return canonical_original_confirmatory_control_publication_ancestor_lease(
        {
            "schema_version": 1,
            "policy": CONTROL_PUBLICATION_ANCESTOR_LEASE_POLICY,
            "project_root": str(project_root),
            "records": canonical_records,
            "record_count": len(canonical_records),
            "records_root_sha256": canonical_json_sha256(canonical_records),
            "directory_access_mask": EXECUTABLE_ANCESTOR_ACCESS_MASK,
            "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
            "delete_share": False,
            "write_access": False,
            "owner_process_identity_required": True,
            "handle_slot_per_record_required": True,
            "continuous_overlap_through_independent_verification_required": True,
            "continuous_overlap_into_supervisor_required": True,
            "acquisition_disposition": (_CONTROL_PUBLICATION_ANCESTOR_DISPOSITION),
        }
    )


def observe_original_confirmatory_control_publication_ancestor_lease(
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(
        _absolute_path(
            str(project_root),
            role="Q publication observed project root",
        )
    )
    paths = (root, root / "artifacts", root / "artifacts" / "resource_control")
    if not all(path.is_dir() for path in paths):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q publication ancestors must already exist before observation"
        )
    records: list[dict[str, Any]] = []
    if os.name == "nt":
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
        for path in paths:
            handle = create_file(
                str(path),
                EXECUTABLE_ANCESTOR_ACCESS_MASK,
                0x1 | 0x2,
                None,
                3,
                0x00200000 | 0x02000000,
                None,
            )
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                volume, file_id, attributes = _windows_directory_handle_facts(
                    cast(int, handle)
                )
            finally:
                kernel32.CloseHandle(handle)
            records.append(
                {
                    "path": str(path),
                    "volume_serial_number": volume,
                    "file_id_128": file_id,
                    "file_attributes": attributes,
                    "reparse_point": False,
                }
            )
    else:
        for path in paths:
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                observed = os.fstat(descriptor)
                volume, file_id = _native_file_identity_from_fd(descriptor)
            finally:
                os.close(descriptor)
            records.append(
                {
                    "path": str(path),
                    "volume_serial_number": volume,
                    "file_id_128": file_id,
                    "file_attributes": int(getattr(observed, "st_file_attributes", 0)),
                    "reparse_point": False,
                }
            )
    return build_original_confirmatory_control_publication_ancestor_lease(
        project_root=root,
        records=records,
    )


_Q_BASE_AUTHORITY_FIELDS = (
    "schema_version",
    "policy",
    "authority_disposition",
    "q_path",
    "project_root",
    "scientific_authority",
    "execution_capsule",
    "publication_ancestor_lease",
    "publication_ancestor_lease_root_sha256",
    "command_derivation_contract",
    "supervisor_release",
    "codex_handoff_base_authority",
)
_Q_ATTEMPT_IDENTITY_FIELDS = {
    "schema_version",
    "policy",
    "attempt_id",
    "q_base_authority_root_sha256",
    "execution_mode",
    "retry_of_run_id",
    "attempt_identity_root_sha256",
    "job_id",
    "run_id",
    "launch_nonce",
}
_CONTROL_STAGING_PROJECTION_FIELDS = {
    "schema_version",
    "policy",
    "supervisor_state_root",
    "control_staging_dir",
    "final_job_dir",
    "staging_attempt_path",
    "e_intent_path",
    "launch_authorization_path",
    "supervisor_launch_spec_path",
    "staging_ready_path",
    "exact_file_allowlist",
}
_Q_REPLACEMENT_V2_FIELDS = {
    "schema_version",
    "policy",
    "authority_disposition",
    "q_path",
    "project_root",
    "scientific_authority",
    "execution_capsule",
    "publication_ancestor_lease",
    "publication_ancestor_lease_root_sha256",
    "command_derivation_contract",
    "supervisor_release",
    "codex_handoff_base_authority",
    "codex_handoff_attempt_creation_authority_payload_sha256",
    "q_base_authority_root_sha256",
    "attempt_identity_projection",
    "attempt_identity_root_sha256",
    "control_staging_projection",
    "control_staging_projection_sha256",
    "expected_launch_environment",
    "q_authority_root_sha256",
}


def build_original_confirmatory_q_attempt_identity_projection(
    *,
    attempt_id: str,
    q_base_authority_root_sha256: str,
) -> dict[str, Any]:
    base_root = _sha256(
        q_base_authority_root_sha256,
        role="Q base authority root",
    )
    if not isinstance(attempt_id, str) or _Q_ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q attempt id violates the exact one-use planner syntax"
        )
    root_preimage = {
        "schema_version": 1,
        "policy": Q_ATTEMPT_IDENTITY_DERIVATION_POLICY,
        "attempt_id": attempt_id,
        "q_base_authority_root_sha256": base_root,
        "execution_mode": "fresh",
        "retry_of_run_id": None,
    }
    identity_root = canonical_json_sha256(root_preimage)
    nonce_preimage = {
        "schema_version": 1,
        "policy": Q_ATTEMPT_LAUNCH_NONCE_DERIVATION_POLICY,
        "attempt_identity_root_sha256": identity_root,
    }
    return {
        **root_preimage,
        "attempt_identity_root_sha256": identity_root,
        "job_id": f"oc-{identity_root}",
        "run_id": f"original-confirmatory-{identity_root}",
        "launch_nonce": canonical_json_sha256(nonce_preimage),
    }


def canonical_original_confirmatory_q_attempt_identity_projection(
    value: Mapping[str, Any],
    *,
    q_base_authority_root_sha256: str,
) -> dict[str, Any]:
    raw = _mapping(value, role="Q attempt identity projection")
    if (
        set(raw) != _Q_ATTEMPT_IDENTITY_FIELDS
        or type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or not isinstance(raw["policy"], str)
        or raw["policy"] != Q_ATTEMPT_IDENTITY_DERIVATION_POLICY
        or not isinstance(raw["attempt_id"], str)
        or _Q_ATTEMPT_ID.fullmatch(raw["attempt_id"]) is None
        or not isinstance(raw["q_base_authority_root_sha256"], str)
        or not isinstance(raw["execution_mode"], str)
        or raw["execution_mode"] != "fresh"
        or raw["retry_of_run_id"] is not None
        or not all(
            isinstance(raw[field], str)
            for field in (
                "attempt_identity_root_sha256",
                "job_id",
                "run_id",
                "launch_nonce",
            )
        )
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q attempt identity projection violates its exact typed schema"
        )
    expected = build_original_confirmatory_q_attempt_identity_projection(
        attempt_id=raw["attempt_id"],
        q_base_authority_root_sha256=q_base_authority_root_sha256,
    )
    if not _strict_json_value_equal(raw, expected):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q attempt identity projection differs from its exact derivation"
        )
    return expected


def build_original_confirmatory_control_staging_projection(
    *,
    supervisor_state_root: str | Path,
    job_id: str,
) -> dict[str, Any]:
    state_root = Path(
        _absolute_path(
            str(supervisor_state_root),
            role="control-staging supervisor state root",
        )
    )
    canonical_job_id = _identifier(job_id, role="control-staging job id")
    control_staging_dir = state_root / CONTROL_STAGING_DIRECTORY_NAME / canonical_job_id
    final_job_dir = state_root / SUPERVISOR_JOBS_DIRECTORY_NAME / canonical_job_id
    return {
        "schema_version": 2,
        "policy": CONTROL_STAGING_PROJECTION_POLICY,
        "supervisor_state_root": str(state_root),
        "control_staging_dir": str(control_staging_dir),
        "final_job_dir": str(final_job_dir),
        "staging_attempt_path": str(control_staging_dir / "staging_attempt.json"),
        "e_intent_path": str(control_staging_dir / E_INTENT_FILENAME),
        "launch_authorization_path": str(
            control_staging_dir / "launch_authorization.json"
        ),
        "supervisor_launch_spec_path": str(
            control_staging_dir / "supervisor_launch_spec.json"
        ),
        "staging_ready_path": str(control_staging_dir / "staging_ready.json"),
        "exact_file_allowlist": list(CONTROL_STAGING_EXACT_FILE_ALLOWLIST),
    }


def canonical_original_confirmatory_control_staging_projection(
    value: Mapping[str, Any],
    *,
    job_id: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    raw = _mapping(value, role="Q control-staging projection")
    if (
        set(raw) != _CONTROL_STAGING_PROJECTION_FIELDS
        or type(raw["schema_version"]) is not int
        or raw["schema_version"] != 2
        or not isinstance(raw["policy"], str)
        or raw["policy"] != CONTROL_STAGING_PROJECTION_POLICY
        or not all(
            isinstance(raw[field], str)
            for field in _CONTROL_STAGING_PROJECTION_FIELDS
            - {"schema_version", "exact_file_allowlist"}
        )
        or type(raw["exact_file_allowlist"]) is not list
        or not all(type(item) is str for item in raw["exact_file_allowlist"])
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q control-staging projection violates its exact v2 typed schema"
        )
    expected = build_original_confirmatory_control_staging_projection(
        supervisor_state_root=_absolute_path(
            raw["supervisor_state_root"],
            role="control-staging supervisor state root",
        ),
        job_id=job_id,
    )
    if not _strict_json_value_equal(raw, expected) or (
        expected_sha256 is not None
        and _sha256(
            expected_sha256,
            role="control-staging projection",
        )
        != canonical_json_sha256(expected)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q control-staging projection differs from its exact v2 derivation"
        )
    return expected


def original_confirmatory_q_replacement_v2_path(
    project_root: str | Path,
) -> Path:
    root = Path(
        _absolute_path(
            str(project_root),
            role="Q replacement-v2 project root",
        )
    )
    return root / "artifacts" / "resource_control" / Q_REPLACEMENT_V2_FILENAME


def canonical_original_confirmatory_q_replacement_v2(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, role="Q replacement-v2")
    if (
        set(raw) != _Q_REPLACEMENT_V2_FIELDS
        or type(raw["schema_version"]) is not int
        or raw["schema_version"] != 2
        or not isinstance(raw["policy"], str)
        or raw["policy"] != Q_REPLACEMENT_V2_POLICY
        or not isinstance(raw["authority_disposition"], str)
        or raw["authority_disposition"] != Q_REPLACEMENT_V2_DISPOSITION
        or not all(
            isinstance(raw[field], str)
            for field in (
                "q_path",
                "project_root",
                "publication_ancestor_lease_root_sha256",
                "q_base_authority_root_sha256",
                "attempt_identity_root_sha256",
                "control_staging_projection_sha256",
                "q_authority_root_sha256",
                "codex_handoff_attempt_creation_authority_payload_sha256",
            )
        )
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q replacement-v2 violates its exact 19-field typed schema"
        )
    project_root = Path(
        _absolute_path(raw["project_root"], role="Q replacement-v2 project root")
    )
    q_path = Path(_absolute_path(raw["q_path"], role="Q replacement-v2 path"))
    capsule = canonical_original_confirmatory_execution_capsule(
        raw["execution_capsule"]
    )
    scientific = canonical_original_confirmatory_scientific_authority_projection(
        raw["scientific_authority"]
    )
    derivation = canonical_original_confirmatory_command_derivation_contract(
        raw["command_derivation_contract"]
    )
    release = canonical_original_confirmatory_supervisor_release_binding(
        raw["supervisor_release"],
        capsule=capsule,
    )
    codex_handoff_base = canonical_original_confirmatory_codex_handoff_base_authority(
        raw["codex_handoff_base_authority"],
        require_operational=True,
    )
    publication_ancestor = (
        canonical_original_confirmatory_control_publication_ancestor_lease(
            raw["publication_ancestor_lease"]
        )
    )
    base_projection = {
        "schema_version": 2,
        "policy": Q_REPLACEMENT_V2_POLICY,
        "authority_disposition": Q_REPLACEMENT_V2_DISPOSITION,
        "q_path": str(q_path),
        "project_root": str(project_root),
        "scientific_authority": scientific,
        "execution_capsule": capsule.as_dict(),
        "publication_ancestor_lease": publication_ancestor,
        "publication_ancestor_lease_root_sha256": canonical_json_sha256(
            publication_ancestor
        ),
        "command_derivation_contract": derivation,
        "supervisor_release": release,
        "codex_handoff_base_authority": codex_handoff_base,
    }
    raw_base_projection = {field: raw[field] for field in _Q_BASE_AUTHORITY_FIELDS}
    base_root = canonical_json_sha256(base_projection)
    attempt_identity = canonical_original_confirmatory_q_attempt_identity_projection(
        raw["attempt_identity_projection"],
        q_base_authority_root_sha256=base_root,
    )
    expected_launch_environment = canonical_expected_launch_environment_envelope_v1(
        cast(Mapping[str, Any], raw["expected_launch_environment"])
    )
    sealed_user_profile_root = Path(
        _absolute_path(
            expected_launch_environment.supervisor_environment["USERPROFILE"],
            role="Q sealed supervisor user-profile root",
        )
    )
    if expected_launch_environment.attempt_nonce != attempt_identity["launch_nonce"]:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q launch environment differs from its exact attempt nonce"
        )
    control_staging = canonical_original_confirmatory_control_staging_projection(
        raw["control_staging_projection"],
        job_id=attempt_identity["job_id"],
        expected_sha256=raw["control_staging_projection_sha256"],
    )
    unsigned = {
        **base_projection,
        "codex_handoff_attempt_creation_authority_payload_sha256": _sha256(
            raw["codex_handoff_attempt_creation_authority_payload_sha256"],
            role="Q Codex handoff attempt-creation payload SHA-256",
        ),
        "q_base_authority_root_sha256": base_root,
        "attempt_identity_projection": attempt_identity,
        "attempt_identity_root_sha256": attempt_identity[
            "attempt_identity_root_sha256"
        ],
        "control_staging_projection": control_staging,
        "control_staging_projection_sha256": canonical_json_sha256(control_staging),
        "expected_launch_environment": expected_launch_environment.as_dict(),
    }
    expected = {
        **unsigned,
        "q_authority_root_sha256": canonical_json_sha256(unsigned),
    }
    if (
        q_path != original_confirmatory_q_replacement_v2_path(project_root)
        or capsule.capsule_ancestor_lease.anchor_path != project_root
        or capsule.runtime_python_ancestor_lease.anchor_path != sealed_user_profile_root
        or publication_ancestor["project_root"] != str(project_root)
        or release["supervisor_state_root"] != control_staging["supervisor_state_root"]
        or not _strict_json_value_equal(raw_base_projection, base_projection)
        or raw["q_base_authority_root_sha256"] != base_root
        or raw["attempt_identity_root_sha256"]
        != attempt_identity["attempt_identity_root_sha256"]
        or not _strict_json_value_equal(raw, expected)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q replacement-v2 violates its exact closed policy"
        )
    return expected


def build_original_confirmatory_q_replacement_v2(
    *,
    project_root: str | Path,
    attempt_id: str,
    scientific_authority: Mapping[str, Any],
    execution_capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    publication_ancestor_lease: Mapping[str, Any],
    external_control_plane_release_root_sha256: str,
    external_control_plane_publication_id: str,
    external_control_plane_release_qualification_attestation_path: str | Path,
    external_control_plane_release_qualification_attestation_file_sha256: str,
    external_control_plane_release_qualification_attestation_root_sha256: str,
    supervisor_code_root: str | Path,
    supervisor_state_root: str | Path,
    supervisor_source_sha256: str,
    supervisor_launcher_sha256: str,
    external_codex_handoff_policy: str,
    external_codex_handoff_authority_spec_file_sha256: str,
    external_codex_handoff_authority_spec_canonical_root_sha256: str,
    internal_codex_wake_disposition: str,
    terminal_client_launcher_source_physical_identity: (
        OriginalConfirmatoryPhysicalFileIdentity | Mapping[str, Any]
    ),
    terminal_client_launcher_source_ancestor_lease: Mapping[str, Any],
    codex_handoff_base_authority: Mapping[str, Any],
    codex_handoff_attempt_creation_authority: Mapping[str, Any],
    expected_launch_environment: (
        ExpectedLaunchEnvironmentEnvelopeV1 | Mapping[str, Any]
    ),
) -> dict[str, Any]:
    capsule = canonical_original_confirmatory_execution_capsule(execution_capsule)
    root = Path(
        _absolute_path(
            str(project_root),
            role="Q replacement-v2 project root",
        )
    )
    publication_ancestor = (
        canonical_original_confirmatory_control_publication_ancestor_lease(
            publication_ancestor_lease
        )
    )
    codex_handoff_base = canonical_original_confirmatory_codex_handoff_base_authority(
        codex_handoff_base_authority,
        require_operational=True,
    )
    base_projection = {
        "schema_version": 2,
        "policy": Q_REPLACEMENT_V2_POLICY,
        "authority_disposition": Q_REPLACEMENT_V2_DISPOSITION,
        "q_path": str(original_confirmatory_q_replacement_v2_path(root)),
        "project_root": str(root),
        "scientific_authority": (
            canonical_original_confirmatory_scientific_authority_projection(
                scientific_authority
            )
        ),
        "execution_capsule": capsule.as_dict(),
        "publication_ancestor_lease": publication_ancestor,
        "publication_ancestor_lease_root_sha256": canonical_json_sha256(
            publication_ancestor
        ),
        "command_derivation_contract": (
            build_original_confirmatory_command_derivation_contract()
        ),
        "supervisor_release": (
            build_original_confirmatory_supervisor_release_binding(
                capsule,
                external_control_plane_release_root_sha256=(
                    external_control_plane_release_root_sha256
                ),
                external_control_plane_publication_id=(
                    external_control_plane_publication_id
                ),
                external_control_plane_release_qualification_attestation_path=(
                    external_control_plane_release_qualification_attestation_path
                ),
                external_control_plane_release_qualification_attestation_file_sha256=(
                    external_control_plane_release_qualification_attestation_file_sha256
                ),
                external_control_plane_release_qualification_attestation_root_sha256=(
                    external_control_plane_release_qualification_attestation_root_sha256
                ),
                supervisor_code_root=supervisor_code_root,
                supervisor_state_root=supervisor_state_root,
                supervisor_source_path=(
                    Path(str(supervisor_code_root)) / "aanca_supervisor.py"
                ),
                supervisor_source_sha256=supervisor_source_sha256,
                supervisor_launcher_path=(
                    Path(str(supervisor_code_root)) / "launch_hidden.ps1"
                ),
                supervisor_launcher_sha256=supervisor_launcher_sha256,
                external_codex_handoff_policy=external_codex_handoff_policy,
                external_codex_handoff_authority_spec_file_sha256=(
                    external_codex_handoff_authority_spec_file_sha256
                ),
                external_codex_handoff_authority_spec_canonical_root_sha256=(
                    external_codex_handoff_authority_spec_canonical_root_sha256
                ),
                internal_codex_wake_disposition=internal_codex_wake_disposition,
                terminal_client_launcher_source_physical_identity=(
                    terminal_client_launcher_source_physical_identity
                ),
                terminal_client_launcher_source_ancestor_lease=(
                    terminal_client_launcher_source_ancestor_lease
                ),
            )
        ),
        "codex_handoff_base_authority": codex_handoff_base,
    }
    base_root = canonical_json_sha256(base_projection)
    attempt_identity = build_original_confirmatory_q_attempt_identity_projection(
        attempt_id=attempt_id,
        q_base_authority_root_sha256=base_root,
    )
    environment = canonical_expected_launch_environment_envelope_v1(
        expected_launch_environment
    )
    if environment.attempt_nonce != attempt_identity["launch_nonce"]:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q launch environment differs from its exact attempt nonce"
        )
    control_staging = build_original_confirmatory_control_staging_projection(
        supervisor_state_root=supervisor_state_root,
        job_id=attempt_identity["job_id"],
    )
    creation = canonical_original_confirmatory_codex_handoff_attempt_creation_authority(
        codex_handoff_attempt_creation_authority,
        base_authority=codex_handoff_base,
        expected_attempt_authority_output_path=(
            Path(control_staging["final_job_dir"])
            / "codex_handoff_attempt_authority.json"
        ),
    )
    unsigned = {
        **base_projection,
        "codex_handoff_attempt_creation_authority_payload_sha256": creation[
            "payload_sha256"
        ],
        "q_base_authority_root_sha256": base_root,
        "attempt_identity_projection": attempt_identity,
        "attempt_identity_root_sha256": attempt_identity[
            "attempt_identity_root_sha256"
        ],
        "control_staging_projection": control_staging,
        "control_staging_projection_sha256": canonical_json_sha256(control_staging),
        "expected_launch_environment": environment.as_dict(),
    }
    return canonical_original_confirmatory_q_replacement_v2(
        {
            **unsigned,
            "q_authority_root_sha256": canonical_json_sha256(unsigned),
        }
    )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryCapsuleCommandProjection:
    """Acyclic E projection whose two E hashes are inserted only after sealing."""

    capsule_mode: str
    program_path: Path
    program_sha256: str
    capsule_path: Path
    capsule_sha256: str
    cwd: Path
    argv_prefix: tuple[str, ...]
    tail_argv_before_e_file_sha256: tuple[str, ...]
    tail_argv_between_e_hashes: tuple[str, ...]
    tail_argv_after_e_core_sha256: tuple[str, ...]
    projection_sha256: str

    def payload_without_self_hash(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": COMMAND_PROJECTION_POLICY,
            "capsule_mode": self.capsule_mode,
            "program_path": str(self.program_path),
            "program_sha256": self.program_sha256,
            "python_isolated_flags": list(CAPSULE_PYTHON_ISOLATED_FLAGS),
            "capsule_path": str(self.capsule_path),
            "capsule_sha256": self.capsule_sha256,
            "cwd": str(self.cwd),
            "argv_prefix": list(self.argv_prefix),
            "tail_argv_before_e_file_sha256": list(self.tail_argv_before_e_file_sha256),
            "tail_argv_between_e_hashes": list(self.tail_argv_between_e_hashes),
            "tail_argv_after_e_core_sha256": list(self.tail_argv_after_e_core_sha256),
            "e_file_sha256_insertion_policy": COMMAND_E_FILE_INSERTION_POLICY,
            "e_core_sha256_insertion_policy": COMMAND_E_CORE_INSERTION_POLICY,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.payload_without_self_hash(),
            "projection_sha256": self.projection_sha256,
        }


_COMMAND_PROJECTION_FIELDS = {
    "schema_version",
    "policy",
    "capsule_mode",
    "program_path",
    "program_sha256",
    "python_isolated_flags",
    "capsule_path",
    "capsule_sha256",
    "cwd",
    "argv_prefix",
    "tail_argv_before_e_file_sha256",
    "tail_argv_between_e_hashes",
    "tail_argv_after_e_core_sha256",
    "e_file_sha256_insertion_policy",
    "e_core_sha256_insertion_policy",
    "projection_sha256",
}


def _exact_string_sequence(value: Any, *, role: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        type(item) is not str or not item or "\x00" in item for item in value
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} must be an exact nonempty string list"
        )
    return tuple(value)


def canonical_original_confirmatory_capsule_command_projection(
    value: OriginalConfirmatoryCapsuleCommandProjection | Mapping[str, Any],
    *,
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    expected_mode: str,
) -> OriginalConfirmatoryCapsuleCommandProjection:
    raw = (
        value.as_dict()
        if type(value) is OriginalConfirmatoryCapsuleCommandProjection
        else _mapping(value, role="E command projection")
    )
    if set(raw) != _COMMAND_PROJECTION_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E command projection has an unexpected field set"
        )
    canonical_capsule = canonical_original_confirmatory_execution_capsule(capsule)
    prefix = _exact_string_sequence(
        raw["argv_prefix"],
        role="E projection argv prefix",
    )
    before = _exact_string_sequence(
        raw["tail_argv_before_e_file_sha256"],
        role="E projection prefix before file hash",
    )
    between = _exact_string_sequence(
        raw["tail_argv_between_e_hashes"],
        role="E projection bridge between hashes",
    )
    after = _exact_string_sequence(
        raw["tail_argv_after_e_core_sha256"],
        role="E projection suffix after core hash",
    )
    expected_prefix = (
        str(canonical_capsule.python_path),
        *CAPSULE_PYTHON_ISOLATED_FLAGS,
        str(canonical_capsule.path),
        expected_mode,
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != COMMAND_PROJECTION_POLICY
        or expected_mode not in CAPSULE_ALLOWED_MODES
        or raw["capsule_mode"] != expected_mode
        or raw["program_path"] != str(canonical_capsule.python_path)
        or raw["program_sha256"] != canonical_capsule.python_sha256
        or not _strict_json_value_equal(
            raw["python_isolated_flags"],
            list(CAPSULE_PYTHON_ISOLATED_FLAGS),
        )
        or raw["capsule_path"] != str(canonical_capsule.path)
        or raw["capsule_sha256"] != canonical_capsule.sha256
        or prefix != expected_prefix
        or len(before) != 3
        or before[0] != "--e-intent"
        or before[2] != "--e-intent-sha256"
        or between != ("--e-intent-core-sha256",)
        or raw["e_file_sha256_insertion_policy"] != COMMAND_E_FILE_INSERTION_POLICY
        or raw["e_core_sha256_insertion_policy"] != COMMAND_E_CORE_INSERTION_POLICY
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E command projection violates its exact identity/slice policy"
        )
    placeholder_file = "0" * 64
    placeholder_core = "1" * 64
    reconstructed_tail = (
        *before,
        placeholder_file,
        *between,
        placeholder_core,
        *after,
    )
    canonical_tail = canonical_original_confirmatory_capsule_mode_tail(
        capsule_mode=expected_mode,
        tail_argv=reconstructed_tail,
    )
    if reconstructed_tail != canonical_tail:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E command projection slices do not reconstruct one canonical tail"
        )
    cwd = Path(_absolute_path(raw["cwd"], role="E command projection cwd"))
    unsigned = {key: item for key, item in raw.items() if key != "projection_sha256"}
    if raw["projection_sha256"] != canonical_json_sha256(unsigned):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E command projection self hash differs"
        )
    return OriginalConfirmatoryCapsuleCommandProjection(
        capsule_mode=expected_mode,
        program_path=canonical_capsule.python_path,
        program_sha256=canonical_capsule.python_sha256,
        capsule_path=canonical_capsule.path,
        capsule_sha256=canonical_capsule.sha256,
        cwd=cwd,
        argv_prefix=prefix,
        tail_argv_before_e_file_sha256=before,
        tail_argv_between_e_hashes=between,
        tail_argv_after_e_core_sha256=after,
        projection_sha256=raw["projection_sha256"],
    )


def build_original_confirmatory_capsule_command_projection(
    *,
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    capsule_mode: str,
    e_intent_path: str | Path,
    q_authority_root_sha256: str,
    launch_nonce: str,
    supervisor_job_id: str,
    supervisor_job_directory: str | Path,
    attempt_id: str,
    run_id: str,
    execution_mode: str,
    retry_of_run_id: str | None,
    cwd: str | Path,
    run_spec_path: str | Path | None = None,
    launch_intent_path: str | Path | None = None,
    process_started_path: str | Path | None = None,
    preterminal_pin_path: str | Path | None = None,
    supervisor_terminal_path: str | Path | None = None,
    verifier_stdout_path: str | Path | None = None,
    composed_terminal_path: str | Path | None = None,
) -> OriginalConfirmatoryCapsuleCommandProjection:
    canonical_capsule = canonical_original_confirmatory_execution_capsule(capsule)
    tail = original_confirmatory_capsule_mode_tail(
        capsule_mode=capsule_mode,
        e_intent_path=e_intent_path,
        e_intent_sha256="0" * 64,
        e_intent_core_sha256="1" * 64,
        q_authority_root_sha256=q_authority_root_sha256,
        launch_nonce=launch_nonce,
        supervisor_job_id=supervisor_job_id,
        supervisor_job_directory=supervisor_job_directory,
        attempt_id=attempt_id,
        run_id=run_id,
        execution_mode=execution_mode,
        retry_of_run_id=retry_of_run_id,
        run_spec_path=run_spec_path,
        launch_intent_path=launch_intent_path,
        process_started_path=process_started_path,
        preterminal_pin_path=preterminal_pin_path,
        supervisor_terminal_path=supervisor_terminal_path,
        verifier_stdout_path=verifier_stdout_path,
        composed_terminal_path=composed_terminal_path,
    )
    if (
        tail[:3]
        != (
            "--e-intent",
            _absolute_path(str(e_intent_path), role="E intent path"),
            "--e-intent-sha256",
        )
        or tail[3] != "0" * 64
        or tail[4:6] != ("--e-intent-core-sha256", "1" * 64)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "internal E command projection split is not canonical"
        )
    prefix = (
        str(canonical_capsule.python_path),
        *CAPSULE_PYTHON_ISOLATED_FLAGS,
        str(canonical_capsule.path),
        capsule_mode,
    )
    provisional = OriginalConfirmatoryCapsuleCommandProjection(
        capsule_mode=capsule_mode,
        program_path=canonical_capsule.python_path,
        program_sha256=canonical_capsule.python_sha256,
        capsule_path=canonical_capsule.path,
        capsule_sha256=canonical_capsule.sha256,
        cwd=Path(_absolute_path(str(cwd), role="E command projection cwd")),
        argv_prefix=prefix,
        tail_argv_before_e_file_sha256=tail[:3],
        tail_argv_between_e_hashes=tail[4:5],
        tail_argv_after_e_core_sha256=tail[6:],
        projection_sha256="0" * 64,
    )
    unsigned = provisional.payload_without_self_hash()
    return canonical_original_confirmatory_capsule_command_projection(
        {
            **unsigned,
            "projection_sha256": canonical_json_sha256(unsigned),
        },
        capsule=canonical_capsule,
        expected_mode=capsule_mode,
    )


_E_Q_AUTHORITY_FIELDS = {"path", "file_sha256", "root_sha256"}
_E_JOB_FIELDS = {
    "schema_version",
    "policy",
    "job_id",
    "supervisor_job_dir",
    "attempt_id",
    "run_id",
    "launch_nonce",
    "supervisor_spec_path",
    "supervisor_spec_schema_version",
    "supervisor_spec_policy",
    "supervisor_release_root_sha256",
    "terminal_custody_authority_projection",
    "terminal_custody_authority_projection_root_sha256",
}
_E_LINEAGE_FIELDS = {
    "schema_version",
    "policy",
    "execution_mode",
    "retry_of_run_id",
}
_SCIENTIFIC_REQUEST_PROJECTION_FIELDS = {
    "schema_version",
    "policy",
    "q_static_runner_binding_sha256",
    "job_id",
    "attempt_id",
    "run_id",
    "execution_mode",
    "retry_of_run_id",
    "runs_root",
    "expected_run_directory",
    "plan_sha256",
    "controls_binding_sha256",
    "bridge_binding_sha256",
    "gate_evidence_sha256",
    "cli_input_binding_sha256",
    "checkpoint_authority_projection",
    "checkpoint_authority_projection_sha256",
    "checkpoint_contract_profile",
    "checkpoint_directive_count",
    "artifact_scope",
    "scientific_outcomes_read",
    "selection_or_tuning_performed",
    "publication_performed",
    "automatic_retry_allowed",
    "projection_sha256",
}
_E_INTENT_FIELDS = {
    "schema_version",
    "policy",
    "authority_disposition",
    "q_authority",
    "project_root",
    "execution_capsule_contract_sha256",
    "command_derivation_contract_sha256",
    "supervisor_release",
    "codex_handoff_attempt_creation_authority",
    "q_codex_handoff_base_authority_payload_sha256",
    "q_codex_handoff_attempt_creation_authority_payload_sha256",
    "job",
    "lineage",
    "expected_launch_environment",
    "process_environment_binding",
    "e_consumption_contract",
    "scientific_request_projection",
    "command_projections",
    "attempt_count",
    "max_attempt_count",
    "automatic_retry_allowed",
    "scientific_outcomes_read",
    "intent_core_sha256",
}


def canonical_original_confirmatory_scientific_request_projection(
    value: Mapping[str, Any],
    *,
    scientific_authority: Mapping[str, Any],
    job: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, role="E scientific request projection")
    if set(raw) != _SCIENTIFIC_REQUEST_PROJECTION_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E scientific request projection has an unexpected field set"
        )
    scientific = canonical_original_confirmatory_scientific_authority_projection(
        scientific_authority
    )
    static = scientific["static_runner_binding"]
    checkpoint = _mapping(
        raw["checkpoint_authority_projection"],
        role="E checkpoint authority projection",
    )
    if not checkpoint:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E checkpoint authority projection is empty"
        )
    runs_root = Path(
        _absolute_path(
            raw["runs_root"],
            role="E scientific request runs root",
        )
    )
    run_directory = Path(
        _absolute_path(
            raw["expected_run_directory"],
            role="E expected scientific run directory",
        )
    )
    unsigned = {key: item for key, item in raw.items() if key != "projection_sha256"}
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != SCIENTIFIC_REQUEST_PROJECTION_POLICY
        or raw["q_static_runner_binding_sha256"]
        != scientific["static_runner_binding_sha256"]
        or raw["job_id"] != job["job_id"]
        or raw["attempt_id"] != job["attempt_id"]
        or raw["run_id"] != job["run_id"]
        or raw["execution_mode"] != lineage["execution_mode"]
        or raw["retry_of_run_id"] != lineage["retry_of_run_id"]
        or runs_root != Path(static["runs_root"])
        or run_directory != runs_root / raw["run_id"]
        or any(
            _SHA256.fullmatch(raw[field]) is None
            for field in (
                "plan_sha256",
                "controls_binding_sha256",
                "bridge_binding_sha256",
                "gate_evidence_sha256",
                "cli_input_binding_sha256",
                "checkpoint_authority_projection_sha256",
            )
            if isinstance(raw[field], str)
        )
        or not all(
            isinstance(raw[field], str)
            for field in (
                "plan_sha256",
                "controls_binding_sha256",
                "bridge_binding_sha256",
                "gate_evidence_sha256",
                "cli_input_binding_sha256",
                "checkpoint_authority_projection_sha256",
            )
        )
        or raw["gate_evidence_sha256"] != static["expected_confirmatory_gate_sha256"]
        or raw["cli_input_binding_sha256"]
        != static["expected_cli_input_binding_sha256"]
        or raw["bridge_binding_sha256"]
        != static["expected_cli_input_binding"]["bridge_binding_sha256"]
        or raw["checkpoint_authority_projection_sha256"]
        != canonical_json_sha256(checkpoint)
        or raw["checkpoint_contract_profile"] != "original_confirmatory_exact_180"
        or type(raw["checkpoint_directive_count"]) is not int
        or raw["checkpoint_directive_count"] != 180
        or raw["artifact_scope"] != REAL_CONFIRMATORY_ARTIFACT_SCOPE
        or raw["scientific_outcomes_read"] is not False
        or raw["selection_or_tuning_performed"] is not False
        or raw["publication_performed"] is not False
        or raw["automatic_retry_allowed"] is not False
        or _contains_mapping_key(raw, "supervisor_spec_sha256")
        or raw["projection_sha256"] != canonical_json_sha256(unsigned)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E scientific request projection violates its exact typed policy"
        )
    return {
        **raw,
        "runs_root": str(runs_root),
        "expected_run_directory": str(run_directory),
        "checkpoint_authority_projection": checkpoint,
    }


def build_original_confirmatory_scientific_request_projection(
    *,
    scientific_authority: Mapping[str, Any],
    job_id: str,
    attempt_id: str,
    run_id: str,
    execution_mode: str,
    retry_of_run_id: str | None,
    plan_sha256: str,
    controls_binding_sha256: str,
    bridge_binding_sha256: str,
    checkpoint_authority_projection: Mapping[str, Any],
) -> dict[str, Any]:
    scientific = canonical_original_confirmatory_scientific_authority_projection(
        scientific_authority
    )
    static = scientific["static_runner_binding"]
    job = {
        "job_id": _identifier(job_id, role="E scientific request job"),
        "attempt_id": _identifier(
            attempt_id,
            role="E scientific request attempt",
        ),
        "run_id": _identifier(run_id, role="E scientific request run"),
    }
    lineage = {
        "execution_mode": execution_mode,
        "retry_of_run_id": retry_of_run_id,
    }
    checkpoint = _mapping(
        checkpoint_authority_projection,
        role="E checkpoint authority projection",
    )
    unsigned = {
        "schema_version": 1,
        "policy": SCIENTIFIC_REQUEST_PROJECTION_POLICY,
        "q_static_runner_binding_sha256": scientific["static_runner_binding_sha256"],
        **job,
        **lineage,
        "runs_root": static["runs_root"],
        "expected_run_directory": str(Path(static["runs_root"]) / job["run_id"]),
        "plan_sha256": _sha256(
            plan_sha256,
            role="E scientific request plan",
        ),
        "controls_binding_sha256": _sha256(
            controls_binding_sha256,
            role="E scientific request controls",
        ),
        "bridge_binding_sha256": _sha256(
            bridge_binding_sha256,
            role="E scientific request bridge",
        ),
        "gate_evidence_sha256": static["expected_confirmatory_gate_sha256"],
        "cli_input_binding_sha256": static["expected_cli_input_binding_sha256"],
        "checkpoint_authority_projection": checkpoint,
        "checkpoint_authority_projection_sha256": canonical_json_sha256(checkpoint),
        "checkpoint_contract_profile": "original_confirmatory_exact_180",
        "checkpoint_directive_count": 180,
        "artifact_scope": REAL_CONFIRMATORY_ARTIFACT_SCOPE,
        "scientific_outcomes_read": False,
        "selection_or_tuning_performed": False,
        "publication_performed": False,
        "automatic_retry_allowed": False,
    }
    return canonical_original_confirmatory_scientific_request_projection(
        {
            **unsigned,
            "projection_sha256": canonical_json_sha256(unsigned),
        },
        scientific_authority=scientific,
        job=job,
        lineage=lineage,
    )


def _canonical_original_confirmatory_e_job(
    value: Any,
    *,
    project_root: Path,
    supervisor_release_root_sha256: str,
    launcher_release: Mapping[str, Any],
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
    verify_terminal_command_projection_sha256: str,
    verify_terminal_environment_sha256: str,
    verify_terminal_cwd: str | Path,
) -> dict[str, Any]:
    raw = _mapping(value, role="E supervisor job binding")
    if set(raw) != _E_JOB_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E supervisor job binding has an unexpected field set"
        )
    job_id = _identifier(raw["job_id"], role="E supervisor job id")
    job_dir = Path(
        _absolute_path(raw["supervisor_job_dir"], role="E supervisor job directory")
    )
    spec_path = Path(
        _absolute_path(raw["supervisor_spec_path"], role="E supervisor spec path")
    )
    projection_value = _mapping(
        raw["terminal_custody_authority_projection"],
        role="E job terminal custody authority projection",
    )
    instance_value = _mapping(
        projection_value.get("outcome_blind_expected_artifact_instance"),
        role="E job artifact instance",
    )
    projection = canonical_original_confirmatory_terminal_custody_authority_projection_self_contained(
        projection_value,
        run_id=cast(str, raw["run_id"]),
        expected_run_directory=cast(str, instance_value.get("expected_run_directory")),
        launcher_release=launcher_release,
        capsule=capsule,
        supervisor_job_id=cast(str, raw["job_id"]),
        supervisor_job_directory=cast(str, raw["supervisor_job_dir"]),
        verify_terminal_command_projection_sha256=(
            verify_terminal_command_projection_sha256
        ),
        verify_terminal_environment_sha256=verify_terminal_environment_sha256,
        verify_terminal_cwd=verify_terminal_cwd,
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != E_JOB_BINDING_POLICY
        or job_dir.name != job_id
        or spec_path != job_dir / "run_spec.json"
        or job_dir == project_root
        or project_root in job_dir.parents
        or _identifier(raw["attempt_id"], role="E attempt id") != raw["attempt_id"]
        or _identifier(raw["run_id"], role="E run id") != raw["run_id"]
        or not isinstance(raw["launch_nonce"], str)
        or _NONCE.fullmatch(raw["launch_nonce"]) is None
        or type(raw["supervisor_spec_schema_version"]) is not int
        or raw["supervisor_spec_schema_version"] != 3
        or raw["supervisor_spec_policy"] != SUPERVISOR_V3_POLICY
        or raw["supervisor_release_root_sha256"] != supervisor_release_root_sha256
        or not _strict_json_value_equal(
            raw["terminal_custody_authority_projection"],
            projection,
        )
        or raw["terminal_custody_authority_projection_root_sha256"]
        != projection["projection_root_sha256"]
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E supervisor job binding violates its exact acyclic policy"
        )
    return dict(raw)


def _canonical_original_confirmatory_e_lineage(value: Any) -> dict[str, Any]:
    raw = _mapping(value, role="E execution lineage")
    if set(raw) != _E_LINEAGE_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E execution lineage has an unexpected field set"
        )
    mode = raw["execution_mode"]
    retry = raw["retry_of_run_id"]
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != E_LINEAGE_POLICY
        or mode not in CAPSULE_EXECUTION_MODES
        or (mode == "fresh" and retry is not None)
        or (
            mode == "successor_resume"
            and _identifier(retry, role="E retry-of run id") != retry
        )
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E execution lineage violates its exact policy"
        )
    return dict(raw)


def canonical_original_confirmatory_e_intent(
    value: Mapping[str, Any],
    *,
    q_authority: Mapping[str, Any],
    expected_q_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Canonicalize one E without importing a downstream supervisor spec hash."""

    raw = _mapping(value, role="E intent")
    if set(raw) != _E_INTENT_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E intent has an unexpected field set"
        )
    q = canonical_original_confirmatory_q_replacement_v2(q_authority)
    project_root = Path(q["project_root"])
    direct_q = _mapping(raw["q_authority"], role="E direct Q authority")
    if set(direct_q) != _E_Q_AUTHORITY_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E direct Q authority has an unexpected field set"
        )
    q_file_sha256 = _sha256(
        direct_q["file_sha256"],
        role="E direct Q file hash",
    )
    canonical_q_file_sha256 = canonical_json_line_sha256(q)
    if q_file_sha256 != canonical_q_file_sha256:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E direct Q file hash differs from canonical Q bytes"
        )
    if expected_q_file_sha256 is not None and q_file_sha256 != _sha256(
        expected_q_file_sha256,
        role="E expected direct Q file hash",
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E direct Q file hash differs from verified Q bytes"
        )
    capsule = canonical_original_confirmatory_execution_capsule(q["execution_capsule"])
    command_derivation = canonical_original_confirmatory_command_derivation_contract(
        q["command_derivation_contract"]
    )
    supervisor_release = canonical_original_confirmatory_supervisor_release_binding(
        raw["supervisor_release"],
        capsule=capsule,
    )
    if not _strict_json_value_equal(supervisor_release, q["supervisor_release"]):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E supervisor release differs from Q"
        )
    codex_handoff_base = canonical_original_confirmatory_codex_handoff_base_authority(
        q["codex_handoff_base_authority"],
        require_operational=True,
    )
    codex_handoff_creation = (
        canonical_original_confirmatory_codex_handoff_attempt_creation_authority(
            raw["codex_handoff_attempt_creation_authority"],
            base_authority=codex_handoff_base,
        )
    )
    if (
        _sha256(
            raw["q_codex_handoff_base_authority_payload_sha256"],
            role="E Q Codex handoff base payload SHA-256",
        )
        != codex_handoff_base["payload_sha256"]
        or _sha256(
            raw["q_codex_handoff_attempt_creation_authority_payload_sha256"],
            role="E Q Codex handoff attempt-creation payload SHA-256",
        )
        != q["codex_handoff_attempt_creation_authority_payload_sha256"]
        or codex_handoff_creation["payload_sha256"]
        != q["codex_handoff_attempt_creation_authority_payload_sha256"]
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E Codex handoff crosslinks differ from sealed Q"
        )
    job_value = _mapping(raw["job"], role="E supervisor job binding")
    if set(job_value) != _E_JOB_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E supervisor job binding has an unexpected field set"
        )
    job = dict(job_value)
    lineage = _canonical_original_confirmatory_e_lineage(raw["lineage"])
    attempt_identity = _mapping(
        q["attempt_identity_projection"],
        role="Q attempt identity projection",
    )
    control_staging = _mapping(
        q["control_staging_projection"],
        role="Q control-staging projection",
    )
    if (
        job["job_id"] != attempt_identity["job_id"]
        or job["attempt_id"] != attempt_identity["attempt_id"]
        or job["run_id"] != attempt_identity["run_id"]
        or job["launch_nonce"] != attempt_identity["launch_nonce"]
        or lineage["execution_mode"] != attempt_identity["execution_mode"]
        or lineage["retry_of_run_id"] != attempt_identity["retry_of_run_id"]
        or job["supervisor_job_dir"] != control_staging["final_job_dir"]
        or job["supervisor_spec_path"]
        != str(Path(control_staging["final_job_dir"]) / "run_spec.json")
        or codex_handoff_creation["payload"]["attempt_authority_output_path"]
        != str(
            Path(control_staging["final_job_dir"])
            / "codex_handoff_attempt_authority.json"
        )
        or supervisor_release["supervisor_state_root"]
        != control_staging["supervisor_state_root"]
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E identity/state paths differ from Q's exact attempt and staging projections"
        )
    expected_environment = canonical_expected_launch_environment_envelope_v1(
        raw["expected_launch_environment"]
    )
    q_expected_environment = canonical_expected_launch_environment_envelope_v1(
        q["expected_launch_environment"]
    )
    environment_binding = canonical_original_confirmatory_process_environment_binding(
        raw["process_environment_binding"],
        expected_environment=expected_environment,
    )
    if (
        not _strict_json_value_equal(
            expected_environment.as_dict(), q_expected_environment.as_dict()
        )
        or not _strict_json_value_equal(
            environment_binding.as_dict(),
            build_original_confirmatory_process_environment_binding(
                q_expected_environment
            ).as_dict(),
        )
        or expected_environment.attempt_nonce != job["launch_nonce"]
        or environment_binding.attempt_nonce_sha256
        != hashlib.sha256(job["launch_nonce"].encode("ascii")).hexdigest()
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E environment binding differs from its exact launch nonce"
        )
    consumption = canonical_original_confirmatory_e_consumption_contract(
        raw["e_consumption_contract"],
        supervisor_job_directory=job["supervisor_job_dir"],
    )
    scientific_request = canonical_original_confirmatory_scientific_request_projection(
        raw["scientific_request_projection"],
        scientific_authority=q["scientific_authority"],
        job=job,
        lineage=lineage,
    )
    projections = raw["command_projections"]
    if not isinstance(projections, dict) or set(projections) != set(
        CAPSULE_ALLOWED_MODES
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E command projections do not cover the exact mode union"
        )
    canonical_projections = {
        mode: canonical_original_confirmatory_capsule_command_projection(
            projections[mode],
            capsule=capsule,
            expected_mode=mode,
        ).as_dict()
        for mode in CAPSULE_ALLOWED_MODES
    }
    e_path = Path(control_staging["e_intent_path"])
    common_projection = {
        "capsule": capsule,
        "e_intent_path": e_path,
        "q_authority_root_sha256": q["q_authority_root_sha256"],
        "launch_nonce": job["launch_nonce"],
        "supervisor_job_id": job["job_id"],
        "supervisor_job_directory": job["supervisor_job_dir"],
        "attempt_id": job["attempt_id"],
        "run_id": job["run_id"],
        "execution_mode": lineage["execution_mode"],
        "retry_of_run_id": lineage["retry_of_run_id"],
        "cwd": project_root,
    }
    job_dir = Path(job["supervisor_job_dir"])
    expected_projections = {
        CAPSULE_SCIENTIFIC_MODE: (
            build_original_confirmatory_capsule_command_projection(
                capsule_mode=CAPSULE_SCIENTIFIC_MODE,
                **common_projection,
            ).as_dict()
        ),
        CAPSULE_PRETERMINAL_MODE: (
            build_original_confirmatory_capsule_command_projection(
                capsule_mode=CAPSULE_PRETERMINAL_MODE,
                run_spec_path=job_dir / "run_spec.json",
                launch_intent_path=job_dir / "launch_intent.json",
                process_started_path=job_dir / "process_started.json",
                preterminal_pin_path=job_dir / "preterminal_pin.json",
                **common_projection,
            ).as_dict()
        ),
        CAPSULE_TERMINAL_MODE: (
            build_original_confirmatory_capsule_command_projection(
                capsule_mode=CAPSULE_TERMINAL_MODE,
                supervisor_terminal_path=job_dir / "terminal_receipt.json",
                verifier_stdout_path=job_dir / "verifier.stdout.log",
                preterminal_pin_path=job_dir / "preterminal_pin.json",
                composed_terminal_path=job_dir / "composed_terminal.json",
                **common_projection,
            ).as_dict()
        ),
    }
    if not _strict_json_value_equal(canonical_projections, expected_projections):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E command projections differ from the exact one-file launch DAG"
        )
    job = _canonical_original_confirmatory_e_job(
        job,
        project_root=project_root,
        supervisor_release_root_sha256=supervisor_release[
            "supervisor_release_root_sha256"
        ],
        launcher_release=supervisor_release["terminal_client_launcher_release"],
        capsule=capsule,
        verify_terminal_command_projection_sha256=canonical_projections[
            CAPSULE_TERMINAL_MODE
        ]["projection_sha256"],
        verify_terminal_environment_sha256=(
            environment_binding.exact_integrity_verifier_environment_sha256
        ),
        verify_terminal_cwd=project_root,
    )
    terminal_custody_projection = _mapping(
        job["terminal_custody_authority_projection"],
        role="E terminal custody projection",
    )
    artifact_instance = _mapping(
        terminal_custody_projection["outcome_blind_expected_artifact_instance"],
        role="E terminal custody artifact instance",
    )
    launcher_projection = _mapping(
        terminal_custody_projection["terminal_client_launcher_projection"],
        role="E terminal-client launcher projection",
    )
    if (
        artifact_instance["run_id"] != scientific_request["run_id"]
        or artifact_instance["expected_run_directory"]
        != scientific_request["expected_run_directory"]
        or artifact_instance["runs_root"] != scientific_request["runs_root"]
        or terminal_custody_projection[
            "terminal_custody_authority_template_root_sha256"
        ]
        != supervisor_release["terminal_custody_authority_template_root_sha256"]
        or launcher_projection["launcher_release_root_sha256"]
        != supervisor_release["terminal_client_launcher_release_root_sha256"]
        or launcher_projection["verify_terminal_command_projection_sha256"]
        != canonical_projections[CAPSULE_TERMINAL_MODE]["projection_sha256"]
        or launcher_projection["verify_terminal_environment_sha256"]
        != environment_binding.exact_integrity_verifier_environment_sha256
        or launcher_projection["verify_terminal_cwd"] != str(project_root)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E terminal custody/launcher projection differs from Q/scientific launch"
        )
    unsigned = {key: item for key, item in raw.items() if key != "intent_core_sha256"}
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != E_INTENT_POLICY
        or raw["authority_disposition"] != E_INTENT_DISPOSITION
        or direct_q["path"] != q["q_path"]
        or direct_q["root_sha256"] != q["q_authority_root_sha256"]
        or raw["project_root"] != str(project_root)
        or raw["execution_capsule_contract_sha256"] != capsule.contract_sha256
        or raw["command_derivation_contract_sha256"]
        != command_derivation["contract_sha256"]
        or type(raw["attempt_count"]) is not int
        or raw["attempt_count"] != 1
        or type(raw["max_attempt_count"]) is not int
        or raw["max_attempt_count"] != 1
        or raw["automatic_retry_allowed"] is not False
        or raw["scientific_outcomes_read"] is not False
        or raw["intent_core_sha256"] != canonical_json_sha256(unsigned)
        or _contains_mapping_key(raw, "supervisor_spec_sha256")
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E intent violates its exact acyclic no-retry policy"
        )
    return {
        **raw,
        "q_authority": dict(direct_q),
        "supervisor_release": supervisor_release,
        "codex_handoff_attempt_creation_authority": codex_handoff_creation,
        "job": job,
        "lineage": lineage,
        "expected_launch_environment": expected_environment.as_dict(),
        "process_environment_binding": environment_binding.as_dict(),
        "e_consumption_contract": consumption.as_dict(),
        "scientific_request_projection": scientific_request,
        "command_projections": canonical_projections,
    }


def build_original_confirmatory_e_intent(
    *,
    q_authority: Mapping[str, Any],
    q_file_sha256: str,
    supervisor_job_id: str,
    supervisor_job_directory: str | Path,
    attempt_id: str,
    run_id: str,
    launch_nonce: str,
    execution_mode: str,
    retry_of_run_id: str | None,
    scientific_request_projection: Mapping[str, Any],
    codex_handoff_attempt_creation_authority: Mapping[str, Any],
    terminal_custody_authority_projection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the sole E file; all three final commands derive from its slices."""

    q = canonical_original_confirmatory_q_replacement_v2(q_authority)
    codex_handoff_base = canonical_original_confirmatory_codex_handoff_base_authority(
        q["codex_handoff_base_authority"],
        require_operational=True,
    )
    capsule = canonical_original_confirmatory_execution_capsule(q["execution_capsule"])
    root = Path(q["project_root"])
    attempt_identity = _mapping(
        q["attempt_identity_projection"],
        role="Q attempt identity projection",
    )
    control_staging = _mapping(
        q["control_staging_projection"],
        role="Q control-staging projection",
    )
    job_dir = Path(
        _absolute_path(
            str(supervisor_job_directory),
            role="E supervisor job directory",
        )
    )
    job_id = _identifier(supervisor_job_id, role="E supervisor job id")
    codex_handoff_creation = (
        canonical_original_confirmatory_codex_handoff_attempt_creation_authority(
            codex_handoff_attempt_creation_authority,
            base_authority=codex_handoff_base,
            expected_attempt_authority_output_path=(
                job_dir / "codex_handoff_attempt_authority.json"
            ),
        )
    )
    if (
        codex_handoff_creation["payload_sha256"]
        != q["codex_handoff_attempt_creation_authority_payload_sha256"]
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E attempt-creation authority differs from sealed Q"
        )
    if (
        job_dir.name != job_id
        or job_id != attempt_identity["job_id"]
        or job_dir != Path(control_staging["final_job_dir"])
        or attempt_id != attempt_identity["attempt_id"]
        or run_id != attempt_identity["run_id"]
        or launch_nonce != attempt_identity["launch_nonce"]
        or execution_mode != attempt_identity["execution_mode"]
        or retry_of_run_id != attempt_identity["retry_of_run_id"]
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E launch identity differs from Q's exact attempt/staging projection"
        )
    envelope = canonical_expected_launch_environment_envelope_v1(
        q["expected_launch_environment"]
    )
    if envelope.attempt_nonce != launch_nonce:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E environment nonce differs from launch nonce"
        )
    lineage = {
        "schema_version": 1,
        "policy": E_LINEAGE_POLICY,
        "execution_mode": execution_mode,
        "retry_of_run_id": retry_of_run_id,
    }
    scientific_request_raw = _mapping(
        scientific_request_projection,
        role="E scientific request projection input",
    )
    e_path = Path(control_staging["e_intent_path"])
    common_projection = {
        "capsule": capsule,
        "e_intent_path": e_path,
        "q_authority_root_sha256": q["q_authority_root_sha256"],
        "launch_nonce": launch_nonce,
        "supervisor_job_id": job_id,
        "supervisor_job_directory": job_dir,
        "attempt_id": attempt_id,
        "run_id": run_id,
        "execution_mode": execution_mode,
        "retry_of_run_id": retry_of_run_id,
        "cwd": root,
    }
    projections = {
        CAPSULE_SCIENTIFIC_MODE: (
            build_original_confirmatory_capsule_command_projection(
                capsule_mode=CAPSULE_SCIENTIFIC_MODE,
                **common_projection,
            ).as_dict()
        ),
        CAPSULE_PRETERMINAL_MODE: (
            build_original_confirmatory_capsule_command_projection(
                capsule_mode=CAPSULE_PRETERMINAL_MODE,
                run_spec_path=job_dir / "run_spec.json",
                launch_intent_path=job_dir / "launch_intent.json",
                process_started_path=job_dir / "process_started.json",
                preterminal_pin_path=job_dir / "preterminal_pin.json",
                **common_projection,
            ).as_dict()
        ),
        CAPSULE_TERMINAL_MODE: (
            build_original_confirmatory_capsule_command_projection(
                capsule_mode=CAPSULE_TERMINAL_MODE,
                supervisor_terminal_path=job_dir / "terminal_receipt.json",
                verifier_stdout_path=job_dir / "verifier.stdout.log",
                preterminal_pin_path=job_dir / "preterminal_pin.json",
                composed_terminal_path=job_dir / "composed_terminal.json",
                **common_projection,
            ).as_dict()
        ),
    }
    environment_binding = build_original_confirmatory_process_environment_binding(
        envelope
    )
    built_terminal_custody_projection = (
        build_original_confirmatory_terminal_custody_authority_projection(
            run_id=run_id,
            expected_run_directory=cast(
                str,
                scientific_request_raw.get("expected_run_directory"),
            ),
            launcher_release=q["supervisor_release"][
                "terminal_client_launcher_release"
            ],
            capsule=capsule,
            supervisor_job_id=job_id,
            supervisor_job_directory=job_dir,
            verify_terminal_command_projection_sha256=projections[
                CAPSULE_TERMINAL_MODE
            ]["projection_sha256"],
            verify_terminal_environment_sha256=(
                environment_binding.exact_integrity_verifier_environment_sha256
            ),
            verify_terminal_cwd=root,
        )
    )
    if (
        terminal_custody_authority_projection is not None
        and not _strict_json_value_equal(
            _mapping(
                terminal_custody_authority_projection,
                role="provided E terminal custody authority projection",
            ),
            built_terminal_custody_projection,
        )
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "provided E terminal custody projection differs from its exact derivation"
        )
    terminal_custody_projection = built_terminal_custody_projection
    job = {
        "schema_version": 1,
        "policy": E_JOB_BINDING_POLICY,
        "job_id": job_id,
        "supervisor_job_dir": str(job_dir),
        "attempt_id": attempt_id,
        "run_id": run_id,
        "launch_nonce": launch_nonce,
        "supervisor_spec_path": str(job_dir / "run_spec.json"),
        "supervisor_spec_schema_version": 3,
        "supervisor_spec_policy": SUPERVISOR_V3_POLICY,
        "supervisor_release_root_sha256": q["supervisor_release"][
            "supervisor_release_root_sha256"
        ],
        "terminal_custody_authority_projection": terminal_custody_projection,
        "terminal_custody_authority_projection_root_sha256": (
            terminal_custody_projection["projection_root_sha256"]
        ),
    }
    unsigned = {
        "schema_version": 1,
        "policy": E_INTENT_POLICY,
        "authority_disposition": E_INTENT_DISPOSITION,
        "q_authority": {
            "path": q["q_path"],
            "file_sha256": _sha256(q_file_sha256, role="E Q file hash"),
            "root_sha256": q["q_authority_root_sha256"],
        },
        "project_root": str(root),
        "execution_capsule_contract_sha256": capsule.contract_sha256,
        "command_derivation_contract_sha256": q["command_derivation_contract"][
            "contract_sha256"
        ],
        "supervisor_release": q["supervisor_release"],
        "codex_handoff_attempt_creation_authority": codex_handoff_creation,
        "q_codex_handoff_base_authority_payload_sha256": codex_handoff_base[
            "payload_sha256"
        ],
        "q_codex_handoff_attempt_creation_authority_payload_sha256": (
            codex_handoff_creation["payload_sha256"]
        ),
        "job": job,
        "lineage": lineage,
        "expected_launch_environment": envelope.as_dict(),
        "process_environment_binding": environment_binding.as_dict(),
        "e_consumption_contract": (
            build_original_confirmatory_e_consumption_contract(
                supervisor_job_directory=job_dir
            ).as_dict()
        ),
        "scientific_request_projection": (
            canonical_original_confirmatory_scientific_request_projection(
                scientific_request_projection,
                scientific_authority=q["scientific_authority"],
                job=job,
                lineage=lineage,
            )
        ),
        "command_projections": projections,
        "attempt_count": 1,
        "max_attempt_count": 1,
        "automatic_retry_allowed": False,
        "scientific_outcomes_read": False,
    }
    return canonical_original_confirmatory_e_intent(
        {
            **unsigned,
            "intent_core_sha256": canonical_json_sha256(unsigned),
        },
        q_authority=q,
        expected_q_file_sha256=q_file_sha256,
    )


def derive_original_confirmatory_capsule_command_from_e(
    *,
    e_intent: Mapping[str, Any],
    e_file_sha256: str,
    q_authority: Mapping[str, Any],
    capsule_mode: str,
) -> OriginalConfirmatoryCapsuleCommand:
    """Insert the two final E hashes into one sealed projection, exactly once."""

    canonical_e = canonical_original_confirmatory_e_intent(
        e_intent,
        q_authority=q_authority,
    )
    file_sha256 = _sha256(e_file_sha256, role="sealed E file")
    actual_file_sha256 = canonical_json_line_sha256(canonical_e)
    if file_sha256 != actual_file_sha256:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "sealed E file SHA-256 differs from canonical E bytes"
        )
    if capsule_mode not in CAPSULE_ALLOWED_MODES:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "sealed E command mode is outside the exact union"
        )
    q = canonical_original_confirmatory_q_replacement_v2(q_authority)
    projection = canonical_original_confirmatory_capsule_command_projection(
        canonical_e["command_projections"][capsule_mode],
        capsule=q["execution_capsule"],
        expected_mode=capsule_mode,
    )
    tail = (
        *projection.tail_argv_before_e_file_sha256,
        file_sha256,
        *projection.tail_argv_between_e_hashes,
        canonical_e["intent_core_sha256"],
        *projection.tail_argv_after_e_core_sha256,
    )
    return build_original_confirmatory_capsule_command(
        capsule=q["execution_capsule"],
        mode=capsule_mode,
        tail_argv=tail,
        cwd=projection.cwd,
    )


_CONTROL_PUBLICATION_IDENTITY_POLICY = (
    "original_confirmatory_control_publication_physical_identity_v1"
)
_CONTROL_PUBLICATION_IDENTITY_FIELDS = {
    "schema_version",
    "policy",
    "path",
    "volume_serial_number",
    "file_id_128",
    "size_bytes",
    "sha256",
    "file_attributes",
    "regular_file",
    "read_only",
    "link_count",
    "named_alternate_data_streams",
    "opened_without_reparse_follow",
    "share_access",
    "write_handle_retained",
    "delete_access",
}


def _windows_directory_handle_facts(handle: int) -> tuple[int, str, int]:
    class FILE_ID_128(ctypes.Structure):
        _fields_ = [("identifier", ctypes.c_ubyte * 16)]

    class FILE_ID_INFO(ctypes.Structure):
        _fields_ = [
            ("volume_serial_number", ctypes.c_ulonglong),
            ("file_id", FILE_ID_128),
        ]

    class FILE_ATTRIBUTE_TAG_INFO(ctypes.Structure):
        _fields_ = [
            ("file_attributes", ctypes.c_uint32),
            ("reparse_tag", ctypes.c_uint32),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    getter = kernel32.GetFileInformationByHandleEx
    getter.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    getter.restype = ctypes.c_int
    file_id = FILE_ID_INFO()
    tag = FILE_ATTRIBUTE_TAG_INFO()
    for info_class, target in ((18, file_id), (9, tag)):
        if not getter(
            ctypes.c_void_p(handle),
            info_class,
            ctypes.byref(target),
            ctypes.sizeof(target),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    return (
        int(file_id.volume_serial_number),
        bytes(file_id.file_id.identifier).hex(),
        int(tag.file_attributes),
    )


def _open_control_publication_ancestor_handles(
    lease: Mapping[str, Any],
) -> tuple[tuple[int, ...], bool]:
    canonical = canonical_original_confirmatory_control_publication_ancestor_lease(
        lease
    )
    handles: list[int] = []
    windows_native = os.name == "nt"
    try:
        for record in canonical["records"]:
            path = Path(cast(str, record["path"]))
            if windows_native:
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
                    EXECUTABLE_ANCESTOR_ACCESS_MASK,
                    0x1 | 0x2,
                    None,
                    3,
                    0x00200000 | 0x02000000,
                    None,
                )
                if handle == ctypes.c_void_p(-1).value:
                    raise ctypes.WinError(ctypes.get_last_error())
                numeric = cast(int, handle)
                volume, file_id, attributes = _windows_directory_handle_facts(numeric)
                if (
                    volume != record["volume_serial_number"]
                    or file_id != record["file_id_128"]
                    or attributes != record["file_attributes"]
                    or attributes & 0x400
                    or not attributes & 0x10
                ):
                    kernel32.CloseHandle(ctypes.c_void_p(numeric))
                    raise OriginalConfirmatoryCapsuleAuthorityError(
                        "Q publication ancestor differs from its Q-bound identity"
                    )
                handles.append(numeric)
            else:
                descriptor = os.open(
                    path,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
                observed = os.fstat(descriptor)
                volume, file_id = _native_file_identity_from_fd(descriptor)
                if (
                    not stat.S_ISDIR(observed.st_mode)
                    or volume != record["volume_serial_number"]
                    or file_id != record["file_id_128"]
                ):
                    os.close(descriptor)
                    raise OriginalConfirmatoryCapsuleAuthorityError(
                        "Q publication ancestor differs from its Q-bound identity"
                    )
                handles.append(descriptor)
    except BaseException:
        _close_control_publication_ancestor_handles(
            tuple(handles),
            windows_native=windows_native,
        )
        raise
    return tuple(handles), windows_native


def _close_control_publication_ancestor_handles(
    handles: Sequence[int],
    *,
    windows_native: bool,
) -> None:
    if windows_native:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        for handle in reversed(handles):
            kernel32.CloseHandle(ctypes.c_void_p(handle))
    else:
        for descriptor in reversed(handles):
            os.close(descriptor)


def _create_new_control_publication_descriptor(path: Path) -> int:
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
        0xC0000000,
        0x1,
        None,
        1,
        0x1 | 0x00200000 | 0x80000000,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(
            cast(int, handle),
            os.O_RDWR | getattr(os, "O_BINARY", 0),
        )
    except BaseException:
        kernel32.CloseHandle(handle)
        raise


def _open_control_publication_transition_guard(path: Path) -> int:
    if os.name != "nt":
        return _open_read_descriptor(path)
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
        0x80000000,
        0x1 | 0x2,
        None,
        3,
        0x00000080 | 0x00200000,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(
            cast(int, handle),
            os.O_RDONLY | getattr(os, "O_BINARY", 0),
        )
    except BaseException:
        kernel32.CloseHandle(handle)
        raise


def _control_publication_identity_from_descriptor(
    descriptor: int,
    *,
    path: Path,
    payload: bytes,
    write_handle_retained: bool,
) -> dict[str, Any]:
    before = os.fstat(descriptor)
    volume, file_id = _native_file_identity_from_fd(descriptor)
    final_path = _windows_final_path_from_fd(descriptor)
    if final_path is not None and os.path.normcase(os.path.normpath(final_path)) != (
        os.path.normcase(os.path.normpath(str(path)))
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q publication handle resolves to an unexpected final path"
        )
    attributes = int(getattr(before, "st_file_attributes", 0))
    read_only = (
        bool(attributes & 0x1) if os.name == "nt" else not bool(before.st_mode & 0o222)
    )
    streams = list(_windows_named_data_streams(path))
    identity = {
        "schema_version": 1,
        "policy": _CONTROL_PUBLICATION_IDENTITY_POLICY,
        "path": str(path),
        "volume_serial_number": volume,
        "file_id_128": file_id,
        "size_bytes": int(before.st_size),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "file_attributes": attributes,
        "regular_file": stat.S_ISREG(before.st_mode),
        "read_only": read_only,
        "link_count": int(before.st_nlink),
        "named_alternate_data_streams": streams,
        "opened_without_reparse_follow": True,
        "share_access": ["FILE_SHARE_READ"],
        "write_handle_retained": write_handle_retained,
        "delete_access": False,
    }
    if (
        set(identity) != _CONTROL_PUBLICATION_IDENTITY_FIELDS
        or identity["regular_file"] is not True
        or identity["read_only"] is not True
        or identity["link_count"] != 1
        or streams
        or identity["size_bytes"] != len(payload)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q publication physical identity violates its immutable policy"
        )
    return identity


@dataclass(slots=True)
class OriginalConfirmatoryQPublicationCustody:
    """Live Q leaf and ancestor handles overlapped into verifier/supervisor."""

    path: Path
    descriptor: int
    ancestor_handles: tuple[int, ...]
    windows_native_ancestor_handles: bool
    publication_identity: dict[str, Any]
    role: str
    closed: bool = False

    def require_active(self) -> None:
        if self.closed or self.descriptor < 0:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "Q publication custody is no longer active"
            )
        os.fstat(self.descriptor)
        if len(self.ancestor_handles) != 3:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "Q publication ancestor custody is incomplete"
            )
        payload = _read_all_from_descriptor(
            self.descriptor,
            maximum_bytes=MAX_CONTROL_FILE_BYTES,
        )
        canonical = canonical_original_confirmatory_q_replacement_v2(
            decode_canonical_json_line(
                payload,
                role="retained Q publication custody",
            )
        )
        identity = _control_publication_identity_from_descriptor(
            self.descriptor,
            path=self.path,
            payload=payload,
            write_handle_retained=self.role == "author",
        )
        if not _strict_json_value_equal(identity, self.publication_identity):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "retained Q leaf identity/content changed"
            )
        records = canonical["publication_ancestor_lease"]["records"]
        for handle, record in zip(
            self.ancestor_handles,
            records,
            strict=True,
        ):
            if self.windows_native_ancestor_handles:
                volume, file_id, attributes = _windows_directory_handle_facts(handle)
            else:
                observed = os.fstat(handle)
                volume, file_id = _native_file_identity_from_fd(handle)
                attributes = int(getattr(observed, "st_file_attributes", 0))
            if (
                volume != record["volume_serial_number"]
                or file_id != record["file_id_128"]
                or attributes != record["file_attributes"]
            ):
                raise OriginalConfirmatoryCapsuleAuthorityError(
                    "retained Q ancestor identity changed"
                )

    def close(self) -> None:
        if self.closed:
            return
        try:
            if self.descriptor >= 0:
                os.close(self.descriptor)
        finally:
            self.descriptor = -1
            _close_control_publication_ancestor_handles(
                self.ancestor_handles,
                windows_native=self.windows_native_ancestor_handles,
            )
            self.ancestor_handles = ()
            self.closed = True


def _open_e_job_directory_handle(
    path: Path,
) -> tuple[int, bool, tuple[int, str, int]]:
    """Open one no-delete-share job-directory anchor for E publication."""

    observed_path = path.lstat()
    if (
        not stat.S_ISDIR(observed_path.st_mode)
        or path.is_symlink()
        or int(getattr(observed_path, "st_file_attributes", 0)) & 0x400
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E supervisor job directory is not one plain directory"
        )
    if os.name != "nt":
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            volume, file_id = _native_file_identity_from_fd(descriptor)
            if not stat.S_ISDIR(opened.st_mode):
                raise OriginalConfirmatoryCapsuleAuthorityError(
                    "E supervisor job anchor is not a directory"
                )
            return (
                descriptor,
                False,
                (
                    volume,
                    file_id,
                    int(getattr(opened, "st_file_attributes", 0)),
                ),
            )
        except BaseException:
            os.close(descriptor)
            raise
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
        EXECUTABLE_ANCESTOR_ACCESS_MASK,
        0x1 | 0x2,
        None,
        3,
        0x00200000 | 0x02000000,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    numeric = cast(int, handle)
    try:
        facts = _windows_directory_handle_facts(numeric)
        observation_descriptor = _descriptor_from_windows_handle_for_observation(
            numeric
        )
        try:
            final_path = _windows_final_path_from_fd(observation_descriptor)
        finally:
            os.close(observation_descriptor)
        if (
            facts[2] & 0x400
            or not facts[2] & 0x10
            or (
                final_path is not None
                and os.path.normcase(os.path.normpath(final_path))
                != os.path.normcase(os.path.normpath(str(path)))
            )
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "E supervisor job anchor resolves to a different directory"
            )
        return numeric, True, facts
    except BaseException:
        kernel32.CloseHandle(ctypes.c_void_p(numeric))
        raise


def _descriptor_from_windows_handle_for_observation(handle: int) -> int:
    """Duplicate a native handle into a temporary CRT descriptor."""

    if os.name != "nt":
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "native Windows handle observation used off Windows"
        )
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = ctypes.c_void_p
    current = get_current_process()
    duplicate = ctypes.c_void_p()
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
        current,
        ctypes.c_void_p(handle),
        current,
        ctypes.byref(duplicate),
        0,
        0,
        0x2,
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(
            cast(int, duplicate.value),
            os.O_RDONLY | getattr(os, "O_BINARY", 0),
        )
    except BaseException:
        kernel32.CloseHandle(duplicate)
        raise


def _close_e_job_directory_handle(handle: int, *, windows_native: bool) -> None:
    if windows_native:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel32.CloseHandle(ctypes.c_void_p(handle)):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        os.close(handle)


def _open_e_job_ancestor_handles(
    job_directory: Path,
) -> tuple[
    tuple[int, ...],
    bool,
    tuple[tuple[int, str, int], ...],
    tuple[Path, ...],
]:
    """Retain supervisor-root -> jobs -> exact-job with no delete sharing."""

    if job_directory.parent.name != "jobs":
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E job directory is outside the exact supervisor-root/jobs chain"
        )
    paths = (
        job_directory.parent.parent,
        job_directory.parent,
        job_directory,
    )
    handles: list[int] = []
    facts: list[tuple[int, str, int]] = []
    windows_native: bool | None = None
    try:
        for path in paths:
            handle, is_native, observed = _open_e_job_directory_handle(path)
            if windows_native is None:
                windows_native = is_native
            elif windows_native != is_native:
                raise OriginalConfirmatoryCapsuleAuthorityError(
                    "E job ancestor handles use inconsistent handle kinds"
                )
            handles.append(handle)
            facts.append(observed)
    except BaseException:
        for handle in reversed(handles):
            _close_e_job_directory_handle(
                handle,
                windows_native=bool(windows_native),
            )
        raise
    return tuple(handles), bool(windows_native), tuple(facts), paths


def _require_staged_e_and_final_job_layout(
    *,
    staged_e_intent_path: str | Path,
    supervisor_job_directory: str | Path,
) -> tuple[Path, Path, tuple[Path, Path, Path], tuple[Path, Path]]:
    """Bind the staged E namespace separately from the future final job.

    The wire-level Q/E custody schemas deliberately stay at version 1 for
    compatibility with the already-qualified supervisor.  This helper closes
    the path ambiguity that the original producer introduced by deriving the
    receipt/job domain from ``E.parent``.  E is always staged below
    ``control_staging/<job_id>`` while receipts always belong to the distinct
    ``jobs/<job_id>`` directory, which must not exist before resume.
    """

    staged_e = Path(
        _absolute_path(
            str(staged_e_intent_path),
            role="staged E publication path",
        )
    )
    final_job = Path(
        _absolute_path(
            str(supervisor_job_directory),
            role="final supervisor job directory",
        )
    )
    stage_dir = staged_e.parent
    control_staging_root = stage_dir.parent
    jobs_root = final_job.parent
    supervisor_state_root = control_staging_root.parent
    if (
        staged_e.name != E_INTENT_FILENAME
        or control_staging_root.name != "control_staging"
        or jobs_root.name != "jobs"
        or jobs_root.parent != supervisor_state_root
        or stage_dir.name != final_job.name
        or stage_dir == final_job
        or stage_dir in final_job.parents
        or final_job in stage_dir.parents
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "staged E and final supervisor job namespaces differ from the fixed layout"
        )
    return (
        staged_e,
        final_job,
        (supervisor_state_root, control_staging_root, stage_dir),
        (supervisor_state_root, jobs_root),
    )


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _open_exact_directory_chain(
    paths: Sequence[Path],
    *,
    role: str,
) -> tuple[
    tuple[int, ...],
    bool,
    tuple[tuple[int, str, int], ...],
    tuple[Path, ...],
]:
    handles: list[int] = []
    facts: list[tuple[int, str, int]] = []
    windows_native: bool | None = None
    exact_paths = tuple(paths)
    try:
        for index, path in enumerate(exact_paths, start=1):
            handle, is_native, observed = _open_e_job_directory_handle(path)
            if windows_native is None:
                windows_native = is_native
            elif windows_native != is_native:
                raise OriginalConfirmatoryCapsuleAuthorityError(
                    f"{role} ancestor handle kinds differ"
                )
            handles.append(handle)
            facts.append(observed)
            if _path_lexists(path) is not True:
                raise OriginalConfirmatoryCapsuleAuthorityError(
                    f"{role} ancestor {index} disappeared"
                )
    except BaseException:
        for handle in reversed(handles):
            _close_e_job_directory_handle(
                handle,
                windows_native=bool(windows_native),
            )
        raise
    return tuple(handles), bool(windows_native), tuple(facts), exact_paths


def _revalidate_exact_directory_chain(
    handles: Sequence[int],
    *,
    windows_native: bool,
    expected_facts: Sequence[tuple[int, str, int]],
    expected_count: int,
    role: str,
) -> None:
    if len(handles) != expected_count or len(expected_facts) != expected_count:
        raise OriginalConfirmatoryCapsuleAuthorityError(f"{role} custody is incomplete")
    for handle, facts in zip(handles, expected_facts, strict=True):
        _revalidate_e_job_directory_handle(
            handle,
            windows_native=windows_native,
            expected_facts=facts,
        )


def _revalidate_e_job_directory_handle(
    handle: int,
    *,
    windows_native: bool,
    expected_facts: tuple[int, str, int],
) -> None:
    if windows_native:
        observed = _windows_directory_handle_facts(handle)
    else:
        value = os.fstat(handle)
        volume, file_id = _native_file_identity_from_fd(handle)
        observed = (
            volume,
            file_id,
            int(getattr(value, "st_file_attributes", 0)),
        )
    if observed != expected_facts:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E supervisor job directory anchor changed"
        )


def _revalidate_e_job_ancestor_handles(
    handles: Sequence[int],
    *,
    windows_native: bool,
    expected_facts: Sequence[tuple[int, str, int]],
) -> None:
    if len(handles) != 3 or len(expected_facts) != 3:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E supervisor job ancestor custody is incomplete"
        )
    for handle, facts in zip(handles, expected_facts, strict=True):
        _revalidate_e_job_directory_handle(
            handle,
            windows_native=windows_native,
            expected_facts=facts,
        )


@dataclass(slots=True)
class OriginalConfirmatoryEPublicationCustody:
    """Continuous E leaf/job-directory custody tied to retained verified Q."""

    path: Path
    descriptor: int
    job_ancestor_handles: tuple[int, ...]
    windows_native_job_ancestor_handles: bool
    job_ancestor_facts: tuple[tuple[int, str, int], ...]
    job_ancestor_paths: tuple[Path, ...]
    publication_identity: dict[str, Any]
    q_custody: OriginalConfirmatoryQPublicationCustody
    role: str
    closed: bool = False

    def require_active(self) -> None:
        if self.closed or self.descriptor < 0 or len(self.job_ancestor_handles) != 3:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "E publication custody is no longer active"
            )
        os.fstat(self.descriptor)
        _revalidate_e_job_ancestor_handles(
            self.job_ancestor_handles,
            windows_native=self.windows_native_job_ancestor_handles,
            expected_facts=self.job_ancestor_facts,
        )
        if self.job_ancestor_paths[-1] != self.path.parent:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "E job ancestor custody paths differ from E path"
            )
        self.q_custody.require_active()
        if self.q_custody.role != "independent-verifier":
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "E custody lacks retained independently verified Q"
            )
        q_payload = _read_all_from_descriptor(
            self.q_custody.descriptor,
            maximum_bytes=MAX_CONTROL_FILE_BYTES,
        )
        q = canonical_original_confirmatory_q_replacement_v2(
            decode_canonical_json_line(
                q_payload,
                role="E custody retained Q",
            )
        )
        payload = _read_all_from_descriptor(
            self.descriptor,
            maximum_bytes=MAX_CONTROL_FILE_BYTES,
        )
        canonical = canonical_original_confirmatory_e_intent(
            decode_canonical_json_line(
                payload,
                role="retained E publication custody",
            ),
            q_authority=q,
            expected_q_file_sha256=hashlib.sha256(q_payload).hexdigest(),
        )
        if (
            Path(canonical["job"]["supervisor_job_dir"]) != self.path.parent
            or self.path.name != E_INTENT_FILENAME
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "retained E payload/path binding changed"
            )
        identity = _control_publication_identity_from_descriptor(
            self.descriptor,
            path=self.path,
            payload=payload,
            write_handle_retained=self.role == "author",
        )
        if not _strict_json_value_equal(identity, self.publication_identity):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "retained E leaf identity/content changed"
            )

    def close(self) -> None:
        if self.closed:
            return
        errors: list[BaseException] = []
        if self.descriptor >= 0:
            try:
                os.close(self.descriptor)
            except BaseException as exc:
                errors.append(exc)
            self.descriptor = -1
        for handle in reversed(self.job_ancestor_handles):
            try:
                _close_e_job_directory_handle(
                    handle,
                    windows_native=self.windows_native_job_ancestor_handles,
                )
            except BaseException as exc:
                errors.append(exc)
        self.job_ancestor_handles = ()
        self.closed = True
        if errors:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "E publication custody close failed"
            ) from errors[0]


@dataclass(slots=True)
class OriginalConfirmatoryStagedEPublicationCustody:
    """Continuous staged-E custody plus a distinct future-job parent lease.

    ``path`` and ``staging_ancestor_*`` bind the immutable E leaf below
    ``control_staging/<job_id>``.  ``final_job_parent_*`` separately bind the
    supervisor-state and ``jobs`` directories while the exact final job is
    still absent.  No final path is inferred from ``path.parent``.
    """

    path: Path
    final_supervisor_job_directory: Path
    descriptor: int
    staging_ancestor_handles: tuple[int, ...]
    windows_native_staging_ancestor_handles: bool
    staging_ancestor_facts: tuple[tuple[int, str, int], ...]
    staging_ancestor_paths: tuple[Path, ...]
    final_job_parent_handles: tuple[int, ...]
    windows_native_final_job_parent_handles: bool
    final_job_parent_facts: tuple[tuple[int, str, int], ...]
    final_job_parent_paths: tuple[Path, ...]
    publication_identity: dict[str, Any]
    q_custody: OriginalConfirmatoryQPublicationCustody
    role: str
    closed: bool = False

    def require_active(self, *, final_job_must_be_absent: bool) -> None:
        if (
            self.closed
            or self.descriptor < 0
            or len(self.staging_ancestor_handles) != 3
            or len(self.final_job_parent_handles) != 2
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "staged E publication custody is no longer active"
            )
        os.fstat(self.descriptor)
        _revalidate_exact_directory_chain(
            self.staging_ancestor_handles,
            windows_native=self.windows_native_staging_ancestor_handles,
            expected_facts=self.staging_ancestor_facts,
            expected_count=3,
            role="staged E ancestor",
        )
        _revalidate_exact_directory_chain(
            self.final_job_parent_handles,
            windows_native=self.windows_native_final_job_parent_handles,
            expected_facts=self.final_job_parent_facts,
            expected_count=2,
            role="final supervisor job parent",
        )
        staged_e, final_job, staging_paths, final_parent_paths = (
            _require_staged_e_and_final_job_layout(
                staged_e_intent_path=self.path,
                supervisor_job_directory=self.final_supervisor_job_directory,
            )
        )
        if (
            staged_e != self.path
            or final_job != self.final_supervisor_job_directory
            or staging_paths != self.staging_ancestor_paths
            or final_parent_paths != self.final_job_parent_paths
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "staged E custody path domains changed"
            )
        if final_job_must_be_absent and _path_lexists(final_job):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "final supervisor job appeared before resume"
            )
        self.q_custody.require_active()
        if self.q_custody.role != "independent-verifier":
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "staged E custody lacks retained independently verified Q"
            )
        q_payload = _read_all_from_descriptor(
            self.q_custody.descriptor,
            maximum_bytes=MAX_CONTROL_FILE_BYTES,
        )
        q = canonical_original_confirmatory_q_replacement_v2(
            decode_canonical_json_line(q_payload, role="staged E custody retained Q")
        )
        payload = _read_all_from_descriptor(
            self.descriptor,
            maximum_bytes=MAX_CONTROL_FILE_BYTES,
        )
        canonical = canonical_original_confirmatory_e_intent(
            decode_canonical_json_line(
                payload, role="retained staged E publication custody"
            ),
            q_authority=q,
            expected_q_file_sha256=hashlib.sha256(q_payload).hexdigest(),
        )
        control_staging = _mapping(
            q["control_staging_projection"],
            role="staged E custody Q control-staging projection",
        )
        if (
            Path(canonical["job"]["supervisor_job_dir"]) != final_job
            or Path(control_staging["e_intent_path"]) != staged_e
            or Path(control_staging["final_job_dir"]) != final_job
            or canonical["job"]["job_id"] != final_job.name
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "retained staged E payload/path binding changed"
            )
        identity = _control_publication_identity_from_descriptor(
            self.descriptor,
            path=self.path,
            payload=payload,
            write_handle_retained=self.role == "author",
        )
        if not _strict_json_value_equal(identity, self.publication_identity):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "retained staged E leaf identity/content changed"
            )

    def close(self) -> None:
        if self.closed:
            return
        errors: list[BaseException] = []
        if self.descriptor >= 0:
            try:
                os.close(self.descriptor)
            except BaseException as exc:
                errors.append(exc)
            self.descriptor = -1
        for handles, windows_native in (
            (
                self.staging_ancestor_handles,
                self.windows_native_staging_ancestor_handles,
            ),
            (
                self.final_job_parent_handles,
                self.windows_native_final_job_parent_handles,
            ),
        ):
            for handle in reversed(handles):
                try:
                    _close_e_job_directory_handle(
                        handle,
                        windows_native=windows_native,
                    )
                except BaseException as exc:
                    errors.append(exc)
        self.staging_ancestor_handles = ()
        self.final_job_parent_handles = ()
        self.closed = True
        if errors:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "staged E publication custody close failed"
            ) from errors[0]


_Q_E_CUSTODY_CONTRACT_FIELDS = {
    "schema_version",
    "policy",
    "transport",
    "ready_message_type",
    "ack_policy",
    "ack_message_type",
    "receipt_policy",
    "receipt_path",
    "ready_max_bytes",
    "ack_max_bytes",
    "leaf_target_access_mask",
    "ancestor_target_access_mask",
    "duplicate_options",
    "close_source",
    "source_custody_retained_until_supervisor_ack",
    "supervisor_retention_policy",
    "independent_verifier_receipt_required",
    "scientific_inputs_before_ack_allowed",
    "automatic_retry_allowed",
    "contract_sha256",
}
_Q_E_CUSTODY_READY_FIELDS = {
    "schema_version",
    "policy",
    "message_type",
    "transport",
    "contract_sha256",
    "supervisor_job_id",
    "supervisor_process_identity",
    "controller_process_identity",
    "windows_boot_time_utc",
    "q_authority_root_sha256",
    "q_file_sha256",
    "e_file_sha256",
    "q_leaf_physical_identity",
    "q_ancestor_lease",
    "e_leaf_physical_identity",
    "e_ancestor_lease",
    "q_leaf_handle",
    "q_ancestor_handles",
    "e_leaf_handle",
    "e_ancestor_handles",
    "leaf_target_access_mask",
    "ancestor_target_access_mask",
    "duplicate_options",
    "close_source",
    "source_custody_retained_until_supervisor_ack",
    "supervisor_retention_policy",
    "independent_verifier_receipt_sha256",
    "scientific_inputs_before_ack_allowed",
    "automatic_retry_allowed",
    "handoff_root_sha256",
}
_Q_E_CUSTODY_RECEIPT_FIELDS = {
    "schema_version",
    "policy",
    "status",
    "contract_sha256",
    "handoff_root_sha256",
    "supervisor_job_id",
    "supervisor_process_identity",
    "controller_process_identity",
    "windows_boot_time_utc",
    "q_authority_root_sha256",
    "q_file_sha256",
    "e_file_sha256",
    "q_leaf_retained_binding",
    "q_ancestor_retained_bindings",
    "e_leaf_retained_binding",
    "e_ancestor_retained_bindings",
    "exact_controller_process_identity_verified",
    "exact_supervisor_process_identity_verified",
    "exact_target_access_verified",
    "exact_physical_identities_verified",
    "independent_verifier_receipt_verified",
    "source_custody_overlap_verified",
    "supervisor_retention_policy",
    "scientific_inputs_read",
    "automatic_retry_allowed",
    "receipt_root_sha256",
}
_Q_E_CUSTODY_ACK_FIELDS = {
    "schema_version",
    "policy",
    "message_type",
    "contract_sha256",
    "handoff_root_sha256",
    "receipt_path",
    "receipt_file_sha256",
    "receipt_root_sha256",
    "supervisor_job_id",
    "supervisor_process_identity",
    "controller_process_identity",
    "all_target_handles_retained",
    "scientific_inputs_read",
    "automatic_retry_allowed",
    "ack_sha256",
}
_Q_E_CUSTODY_SPEC_RECEIPT_BINDING_FIELDS = {
    "policy",
    "path",
    "file_sha256",
    "receipt_root_sha256",
    "handoff_root_sha256",
}
_Q_E_CUSTODY_SPEC_FIELDS = {
    "q_e_custody_contract",
    "q_e_custody_handoff",
    "q_e_custody_receipt",
}
_Q_E_CUSTODY_E_ANCESTOR_LEASE_POLICY = (
    "original_confirmatory_e_job_publication_ancestor_lease_v1"
)
_Q_E_CUSTODY_E_ANCESTOR_LEASE_FIELDS = {
    "schema_version",
    "policy",
    "supervisor_root",
    "records",
    "record_count",
    "records_root_sha256",
    "directory_access_mask",
    "share_access",
    "delete_share",
    "write_access",
    "owner_process_identity_required",
    "handle_slot_per_record_required",
    "continuous_overlap_through_independent_verification_required",
    "continuous_overlap_into_supervisor_required",
    "acquisition_disposition",
}
_Q_E_CUSTODY_E_ANCESTOR_DISPOSITION = (
    "opened_before_e_create_new_retained_through_verifier_and_supervisor_overlap_v1"
)
_Q_E_CUSTODY_LEAF_RETAINED_BINDING_POLICY = (
    "original_confirmatory_q_e_leaf_retained_binding_v1"
)
_Q_E_CUSTODY_LEAF_RETAINED_BINDING_FIELDS = {
    "schema_version",
    "policy",
    "role",
    "path",
    "target_handle_value",
    "target_granted_access_mask",
    "physical_identity",
    "retained",
    "retention_policy",
    "binding_sha256",
}
_Q_E_CUSTODY_ANCESTOR_RETAINED_BINDING_POLICY = (
    "original_confirmatory_q_e_ancestor_retained_binding_v1"
)
_Q_E_CUSTODY_ANCESTOR_RETAINED_BINDING_FIELDS = {
    "schema_version",
    "policy",
    "role",
    "index",
    "path",
    "target_handle_value",
    "target_granted_access_mask",
    "volume_serial_number",
    "file_id_128",
    "file_attributes",
    "reparse_point",
    "retained",
    "retention_policy",
    "binding_sha256",
}
_Q_E_CUSTODY_RECEIPT_STATUS = "retained_verified_before_scientific_inputs_v1"


def build_original_confirmatory_q_e_custody_contract(
    *,
    supervisor_job_directory: str | Path,
) -> dict[str, Any]:
    job_dir = Path(
        _absolute_path(
            str(supervisor_job_directory),
            role="Q/E custody supervisor job directory",
        )
    )
    unsigned = {
        "schema_version": 1,
        "policy": Q_E_CUSTODY_CONTRACT_POLICY,
        "transport": Q_E_CUSTODY_TRANSPORT,
        "ready_message_type": Q_E_CUSTODY_READY_MESSAGE_TYPE,
        "ack_policy": Q_E_CUSTODY_ACK_POLICY,
        "ack_message_type": Q_E_CUSTODY_ACK_MESSAGE_TYPE,
        "receipt_policy": Q_E_CUSTODY_RECEIPT_POLICY,
        "receipt_path": str(job_dir / Q_E_CUSTODY_RECEIPT_FILENAME),
        "ready_max_bytes": Q_E_CUSTODY_LINE_MAX_BYTES,
        "ack_max_bytes": Q_E_CUSTODY_LINE_MAX_BYTES,
        "leaf_target_access_mask": E_CONSUMPTION_DUPLICATE_TARGET_ACCESS_MASK,
        "ancestor_target_access_mask": EXECUTABLE_ANCESTOR_ACCESS_MASK,
        "duplicate_options": 0,
        "close_source": False,
        "source_custody_retained_until_supervisor_ack": True,
        "supervisor_retention_policy": Q_E_CUSTODY_SUPERVISOR_RETENTION_POLICY,
        "independent_verifier_receipt_required": True,
        "scientific_inputs_before_ack_allowed": False,
        "automatic_retry_allowed": False,
    }
    return {**unsigned, "contract_sha256": canonical_json_sha256(unsigned)}


def canonical_original_confirmatory_q_e_custody_contract(
    value: Mapping[str, Any],
    *,
    supervisor_job_directory: str | Path,
) -> dict[str, Any]:
    raw = _mapping(value, role="Q/E custody contract")
    expected = build_original_confirmatory_q_e_custody_contract(
        supervisor_job_directory=supervisor_job_directory,
    )
    if set(raw) != _Q_E_CUSTODY_CONTRACT_FIELDS or not _strict_json_value_equal(
        raw,
        expected,
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E custody contract violates its exact one-shot policy"
        )
    return expected


def _canonical_q_e_control_physical_identity(
    value: Any,
    *,
    role: str,
    expected_path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    raw = _mapping(value, role=role)
    if set(raw) != _CONTROL_PUBLICATION_IDENTITY_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} has an unexpected field set"
        )
    path = Path(_absolute_path(raw["path"], role=f"{role} path"))
    file_id = raw["file_id_128"]
    attributes = raw["file_attributes"]
    streams = raw["named_alternate_data_streams"]
    if (
        path != expected_path
        or type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _CONTROL_PUBLICATION_IDENTITY_POLICY
        or _nonnegative_int(
            raw["volume_serial_number"],
            role=f"{role} volume",
        )
        != raw["volume_serial_number"]
        or not isinstance(file_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", file_id) is None
        or _positive_int(raw["size_bytes"], role=f"{role} size") != raw["size_bytes"]
        or raw["sha256"] != _sha256(expected_sha256, role=f"{role} SHA-256")
        or type(attributes) is not int
        or attributes < 0
        or raw["regular_file"] is not True
        or raw["read_only"] is not True
        or raw["link_count"] != 1
        or type(raw["link_count"]) is not int
        or not _strict_json_value_equal(streams, [])
        or raw["opened_without_reparse_follow"] is not True
        or not _strict_json_value_equal(raw["share_access"], ["FILE_SHARE_READ"])
        or raw["write_handle_retained"] is not False
        or raw["delete_access"] is not False
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} violates its exact immutable identity"
        )
    return dict(raw)


def _canonical_q_e_e_ancestor_lease(
    value: Any,
    *,
    supervisor_job_directory: Path,
    staged_e_intent_path: Path | None = None,
) -> dict[str, Any]:
    raw = _mapping(value, role="Q/E custody E ancestor lease")
    if set(raw) != _Q_E_CUSTODY_E_ANCESTOR_LEASE_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E custody E ancestor lease has an unexpected field set"
        )
    if staged_e_intent_path is None:
        if supervisor_job_directory.parent.name != "jobs":
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "Q/E custody E job is outside supervisor-root/jobs"
            )
        supervisor_root = supervisor_job_directory.parent.parent
        expected_paths = (
            supervisor_root,
            supervisor_job_directory.parent,
            supervisor_job_directory,
        )
    else:
        staged_e, final_job, expected_paths, _final_parent_paths = (
            _require_staged_e_and_final_job_layout(
                staged_e_intent_path=staged_e_intent_path,
                supervisor_job_directory=supervisor_job_directory,
            )
        )
        if staged_e != staged_e_intent_path or final_job != supervisor_job_directory:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "Q/E custody E ancestor lease paths are not canonical"
            )
        supervisor_root = expected_paths[0]
    items = raw["records"]
    if not isinstance(items, list) or len(items) != 3:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E custody E ancestor lease must contain exactly three records"
        )
    records: list[dict[str, Any]] = []
    for index, (item, expected_path) in enumerate(
        zip(items, expected_paths, strict=True),
        start=1,
    ):
        record = _mapping(item, role=f"Q/E custody E ancestor record {index}")
        file_id = record.get("file_id_128")
        attributes = record.get("file_attributes")
        if (
            set(record) != _CAPSULE_ANCESTOR_RECORD_FIELDS
            or Path(
                _absolute_path(
                    record["path"],
                    role=f"Q/E custody E ancestor record {index} path",
                )
            )
            != expected_path
            or _nonnegative_int(
                record["volume_serial_number"],
                role=f"Q/E custody E ancestor record {index} volume",
            )
            != record["volume_serial_number"]
            or not isinstance(file_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", file_id) is None
            or type(attributes) is not int
            or attributes < 0
            or attributes & 0x400
            or record["reparse_point"] is not False
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                f"Q/E custody E ancestor record {index} is invalid"
            )
        records.append(dict(record))
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _Q_E_CUSTODY_E_ANCESTOR_LEASE_POLICY
        or raw["supervisor_root"] != str(supervisor_root)
        or raw["record_count"] != 3
        or type(raw["record_count"]) is not int
        or raw["records_root_sha256"] != canonical_json_sha256(records)
        or raw["directory_access_mask"] != EXECUTABLE_ANCESTOR_ACCESS_MASK
        or type(raw["directory_access_mask"]) is not int
        or not _strict_json_value_equal(
            raw["share_access"],
            ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        )
        or raw["delete_share"] is not False
        or raw["write_access"] is not False
        or raw["owner_process_identity_required"] is not True
        or raw["handle_slot_per_record_required"] is not True
        or raw["continuous_overlap_through_independent_verification_required"]
        is not True
        or raw["continuous_overlap_into_supervisor_required"] is not True
        or raw["acquisition_disposition"] != _Q_E_CUSTODY_E_ANCESTOR_DISPOSITION
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E custody E ancestor lease violates its exact policy"
        )
    return {**raw, "records": records}


def _build_q_e_e_ancestor_lease(
    custody: (
        OriginalConfirmatoryEPublicationCustody
        | OriginalConfirmatoryStagedEPublicationCustody
    ),
) -> dict[str, Any]:
    if isinstance(custody, OriginalConfirmatoryStagedEPublicationCustody):
        ancestor_paths = custody.staging_ancestor_paths
        ancestor_facts = custody.staging_ancestor_facts
        supervisor_job_directory = custody.final_supervisor_job_directory
        staged_e_intent_path: Path | None = custody.path
    else:
        ancestor_paths = custody.job_ancestor_paths
        ancestor_facts = custody.job_ancestor_facts
        supervisor_job_directory = custody.path.parent
        staged_e_intent_path = None
    records = [
        {
            "path": str(path),
            "volume_serial_number": facts[0],
            "file_id_128": facts[1],
            "file_attributes": facts[2],
            "reparse_point": False,
        }
        for path, facts in zip(
            ancestor_paths,
            ancestor_facts,
            strict=True,
        )
    ]
    unsigned = {
        "schema_version": 1,
        "policy": _Q_E_CUSTODY_E_ANCESTOR_LEASE_POLICY,
        "supervisor_root": str(ancestor_paths[0]),
        "records": records,
        "record_count": 3,
        "records_root_sha256": canonical_json_sha256(records),
        "directory_access_mask": EXECUTABLE_ANCESTOR_ACCESS_MASK,
        "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        "delete_share": False,
        "write_access": False,
        "owner_process_identity_required": True,
        "handle_slot_per_record_required": True,
        "continuous_overlap_through_independent_verification_required": True,
        "continuous_overlap_into_supervisor_required": True,
        "acquisition_disposition": _Q_E_CUSTODY_E_ANCESTOR_DISPOSITION,
    }
    return _canonical_q_e_e_ancestor_lease(
        unsigned,
        supervisor_job_directory=supervisor_job_directory,
        staged_e_intent_path=staged_e_intent_path,
    )


def _resolve_q_e_ready_layout(
    *,
    e_path: Path,
    supervisor_job_directory: Path | None,
    staged_e_intent_path: Path | None,
) -> tuple[Path, Path | None]:
    """Resolve the explicit or uniquely derivable staged-E/final-job split."""

    if supervisor_job_directory is None and staged_e_intent_path is None:
        stage_dir = e_path.parent
        control_staging_root = stage_dir.parent
        supervisor_state_root = control_staging_root.parent
        inferred_final_job = supervisor_state_root / "jobs" / stage_dir.name
        staged_e, final_job, _staging_paths, _final_parent_paths = (
            _require_staged_e_and_final_job_layout(
                staged_e_intent_path=e_path,
                supervisor_job_directory=inferred_final_job,
            )
        )
        return final_job, staged_e
    if supervisor_job_directory is None or staged_e_intent_path is None:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "staged Q/E custody requires both explicit E and final-job paths"
        )
    staged_e, final_job, _staging_paths, _final_parent_paths = (
        _require_staged_e_and_final_job_layout(
            staged_e_intent_path=staged_e_intent_path,
            supervisor_job_directory=supervisor_job_directory,
        )
    )
    if e_path != staged_e:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E custody E identity differs from the explicit staged E path"
        )
    return final_job, staged_e


def canonical_original_confirmatory_q_e_custody_ready(
    value: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    supervisor_job_directory: Path | None = None,
    staged_e_intent_path: Path | None = None,
) -> dict[str, Any]:
    raw = _mapping(value, role="Q/E custody READY")
    if set(raw) != _Q_E_CUSTODY_READY_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E custody READY has an unexpected field set"
        )
    e_identity_raw = _mapping(
        raw["e_leaf_physical_identity"],
        role="Q/E custody E leaf identity",
    )
    e_path = Path(
        _absolute_path(
            e_identity_raw.get("path"),
            role="Q/E custody E leaf path",
        )
    )
    job_dir, selected_staged_e = _resolve_q_e_ready_layout(
        e_path=e_path,
        supervisor_job_directory=supervisor_job_directory,
        staged_e_intent_path=staged_e_intent_path,
    )
    canonical_contract = canonical_original_confirmatory_q_e_custody_contract(
        contract,
        supervisor_job_directory=job_dir,
    )
    supervisor = _canonical_e_process_identity(
        raw["supervisor_process_identity"],
        role="Q/E custody READY supervisor identity",
    )
    controller = _canonical_e_process_identity(
        raw["controller_process_identity"],
        role="Q/E custody READY controller identity",
    )
    q_sha256 = _sha256(raw["q_file_sha256"], role="Q/E custody Q file")
    e_sha256 = _sha256(raw["e_file_sha256"], role="Q/E custody E file")
    q_identity_raw = _mapping(
        raw["q_leaf_physical_identity"],
        role="Q/E custody Q leaf identity",
    )
    q_path = Path(
        _absolute_path(
            q_identity_raw.get("path"),
            role="Q/E custody Q leaf path",
        )
    )
    q_identity = _canonical_q_e_control_physical_identity(
        q_identity_raw,
        role="Q/E custody Q leaf identity",
        expected_path=q_path,
        expected_sha256=q_sha256,
    )
    e_identity = _canonical_q_e_control_physical_identity(
        e_identity_raw,
        role="Q/E custody E leaf identity",
        expected_path=e_path,
        expected_sha256=e_sha256,
    )
    q_lease = canonical_original_confirmatory_control_publication_ancestor_lease(
        _mapping(
            raw["q_ancestor_lease"],
            role="Q/E custody Q ancestor lease",
        )
    )
    e_lease = _canonical_q_e_e_ancestor_lease(
        raw["e_ancestor_lease"],
        supervisor_job_directory=job_dir,
        staged_e_intent_path=selected_staged_e,
    )
    q_handles = raw["q_ancestor_handles"]
    e_handles = raw["e_ancestor_handles"]
    if (
        not isinstance(q_handles, list)
        or len(q_handles) != 3
        or not isinstance(e_handles, list)
        or len(e_handles) != 3
        or any(type(item) is not int or item <= 0 for item in (*q_handles, *e_handles))
        or type(raw["q_leaf_handle"]) is not int
        or raw["q_leaf_handle"] <= 0
        or type(raw["e_leaf_handle"]) is not int
        or raw["e_leaf_handle"] <= 0
        or len(
            {
                raw["q_leaf_handle"],
                *q_handles,
                raw["e_leaf_handle"],
                *e_handles,
            }
        )
        != 8
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E custody READY handle slots are incomplete or not unique"
        )
    unsigned = {key: item for key, item in raw.items() if key != "handoff_root_sha256"}
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != Q_E_CUSTODY_HANDOFF_POLICY
        or raw["message_type"] != Q_E_CUSTODY_READY_MESSAGE_TYPE
        or raw["transport"] != Q_E_CUSTODY_TRANSPORT
        or raw["contract_sha256"] != canonical_contract["contract_sha256"]
        or raw["supervisor_job_id"] != job_dir.name
        or _identifier(
            raw["supervisor_job_id"],
            role="Q/E custody supervisor job ID",
        )
        != raw["supervisor_job_id"]
        or _utc_timestamp(
            raw["windows_boot_time_utc"],
            role="Q/E custody Windows boot time",
        )
        != raw["windows_boot_time_utc"]
        or _sha256(
            raw["q_authority_root_sha256"],
            role="Q/E custody Q authority root",
        )
        != raw["q_authority_root_sha256"]
        or q_path
        != Path(q_lease["project_root"])
        / "artifacts"
        / "resource_control"
        / Q_REPLACEMENT_V2_FILENAME
        or e_path.name != E_INTENT_FILENAME
        or len(q_lease["records"]) != len(q_handles)
        or len(e_lease["records"]) != len(e_handles)
        or raw["leaf_target_access_mask"]
        != canonical_contract["leaf_target_access_mask"]
        or type(raw["leaf_target_access_mask"]) is not int
        or raw["ancestor_target_access_mask"]
        != canonical_contract["ancestor_target_access_mask"]
        or type(raw["ancestor_target_access_mask"]) is not int
        or raw["duplicate_options"] != canonical_contract["duplicate_options"]
        or type(raw["duplicate_options"]) is not int
        or raw["close_source"] is not canonical_contract["close_source"]
        or raw["source_custody_retained_until_supervisor_ack"]
        is not canonical_contract["source_custody_retained_until_supervisor_ack"]
        or raw["supervisor_retention_policy"]
        != canonical_contract["supervisor_retention_policy"]
        or _sha256(
            raw["independent_verifier_receipt_sha256"],
            role="Q/E independent verifier receipt",
        )
        != raw["independent_verifier_receipt_sha256"]
        or raw["scientific_inputs_before_ack_allowed"]
        is not canonical_contract["scientific_inputs_before_ack_allowed"]
        or raw["automatic_retry_allowed"]
        is not canonical_contract["automatic_retry_allowed"]
        or raw["handoff_root_sha256"] != canonical_json_sha256(unsigned)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E custody READY violates its exact fail-closed policy"
        )
    return {
        **raw,
        "supervisor_process_identity": supervisor,
        "controller_process_identity": controller,
        "q_leaf_physical_identity": q_identity,
        "q_ancestor_lease": q_lease,
        "e_leaf_physical_identity": e_identity,
        "e_ancestor_lease": e_lease,
        "q_ancestor_handles": list(q_handles),
        "e_ancestor_handles": list(e_handles),
    }


def build_original_confirmatory_q_e_custody_ready(
    *,
    contract: Mapping[str, Any],
    supervisor_job_id: str,
    supervisor_process_identity: Mapping[str, Any],
    controller_process_identity: Mapping[str, Any],
    windows_boot_time_utc: str,
    q_authority_root_sha256: str,
    q_file_sha256: str,
    e_file_sha256: str,
    q_leaf_physical_identity: Mapping[str, Any],
    q_ancestor_lease: Mapping[str, Any],
    e_leaf_physical_identity: Mapping[str, Any],
    e_ancestor_lease: Mapping[str, Any],
    q_leaf_handle: int,
    q_ancestor_handles: Sequence[int],
    e_leaf_handle: int,
    e_ancestor_handles: Sequence[int],
    independent_verifier_receipt_sha256: str,
    supervisor_job_directory: Path | None = None,
    staged_e_intent_path: Path | None = None,
) -> dict[str, Any]:
    e_path = Path(
        _absolute_path(
            _mapping(
                e_leaf_physical_identity,
                role="Q/E READY E identity",
            )["path"],
            role="Q/E READY E path",
        )
    )
    job_dir, selected_staged_e = _resolve_q_e_ready_layout(
        e_path=e_path,
        supervisor_job_directory=supervisor_job_directory,
        staged_e_intent_path=staged_e_intent_path,
    )
    canonical_contract = canonical_original_confirmatory_q_e_custody_contract(
        contract,
        supervisor_job_directory=job_dir,
    )
    unsigned = {
        "schema_version": 1,
        "policy": Q_E_CUSTODY_HANDOFF_POLICY,
        "message_type": Q_E_CUSTODY_READY_MESSAGE_TYPE,
        "transport": Q_E_CUSTODY_TRANSPORT,
        "contract_sha256": canonical_contract["contract_sha256"],
        "supervisor_job_id": supervisor_job_id,
        "supervisor_process_identity": dict(supervisor_process_identity),
        "controller_process_identity": dict(controller_process_identity),
        "windows_boot_time_utc": windows_boot_time_utc,
        "q_authority_root_sha256": q_authority_root_sha256,
        "q_file_sha256": q_file_sha256,
        "e_file_sha256": e_file_sha256,
        "q_leaf_physical_identity": dict(q_leaf_physical_identity),
        "q_ancestor_lease": dict(q_ancestor_lease),
        "e_leaf_physical_identity": dict(e_leaf_physical_identity),
        "e_ancestor_lease": dict(e_ancestor_lease),
        "q_leaf_handle": q_leaf_handle,
        "q_ancestor_handles": list(q_ancestor_handles),
        "e_leaf_handle": e_leaf_handle,
        "e_ancestor_handles": list(e_ancestor_handles),
        "leaf_target_access_mask": canonical_contract["leaf_target_access_mask"],
        "ancestor_target_access_mask": canonical_contract[
            "ancestor_target_access_mask"
        ],
        "duplicate_options": canonical_contract["duplicate_options"],
        "close_source": canonical_contract["close_source"],
        "source_custody_retained_until_supervisor_ack": canonical_contract[
            "source_custody_retained_until_supervisor_ack"
        ],
        "supervisor_retention_policy": canonical_contract[
            "supervisor_retention_policy"
        ],
        "independent_verifier_receipt_sha256": independent_verifier_receipt_sha256,
        "scientific_inputs_before_ack_allowed": canonical_contract[
            "scientific_inputs_before_ack_allowed"
        ],
        "automatic_retry_allowed": canonical_contract["automatic_retry_allowed"],
    }
    return canonical_original_confirmatory_q_e_custody_ready(
        {
            **unsigned,
            "handoff_root_sha256": canonical_json_sha256(unsigned),
        },
        contract=canonical_contract,
        supervisor_job_directory=job_dir if selected_staged_e is not None else None,
        staged_e_intent_path=selected_staged_e,
    )


def _build_q_e_leaf_retained_binding(
    *,
    role: str,
    path: str,
    target_handle_value: int,
    physical_identity: Mapping[str, Any],
    retention_policy: str,
) -> dict[str, Any]:
    unsigned = {
        "schema_version": 1,
        "policy": _Q_E_CUSTODY_LEAF_RETAINED_BINDING_POLICY,
        "role": role,
        "path": path,
        "target_handle_value": target_handle_value,
        "target_granted_access_mask": E_CONSUMPTION_MAPPED_FILE_GENERIC_READ_ACCESS_MASK,
        "physical_identity": dict(physical_identity),
        "retained": True,
        "retention_policy": retention_policy,
    }
    return {**unsigned, "binding_sha256": canonical_json_sha256(unsigned)}


def _build_q_e_ancestor_retained_bindings(
    *,
    role: str,
    records: Sequence[Mapping[str, Any]],
    target_handles: Sequence[int],
    retention_policy: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, (record, target_handle) in enumerate(
        zip(records, target_handles, strict=True),
        start=1,
    ):
        unsigned = {
            "schema_version": 1,
            "policy": _Q_E_CUSTODY_ANCESTOR_RETAINED_BINDING_POLICY,
            "role": role,
            "index": index,
            "path": record["path"],
            "target_handle_value": target_handle,
            "target_granted_access_mask": E_CONSUMPTION_MAPPED_FILE_GENERIC_READ_ACCESS_MASK,
            "volume_serial_number": record["volume_serial_number"],
            "file_id_128": record["file_id_128"],
            "file_attributes": record["file_attributes"],
            "reparse_point": False,
            "retained": True,
            "retention_policy": retention_policy,
        }
        result.append(
            {
                **unsigned,
                "binding_sha256": canonical_json_sha256(unsigned),
            }
        )
    return result


def build_original_confirmatory_q_e_custody_receipt(
    *,
    contract: Mapping[str, Any],
    ready: Mapping[str, Any],
    supervisor_job_directory: Path | None = None,
    staged_e_intent_path: Path | None = None,
) -> dict[str, Any]:
    ready_raw = _mapping(ready, role="Q/E custody READY for receipt")
    e_identity = _mapping(
        ready_raw.get("e_leaf_physical_identity"),
        role="Q/E custody E identity for receipt",
    )
    e_path = Path(
        _absolute_path(
            e_identity.get("path"),
            role="Q/E custody E path for receipt",
        )
    )
    job_dir, selected_staged_e = _resolve_q_e_ready_layout(
        e_path=e_path,
        supervisor_job_directory=supervisor_job_directory,
        staged_e_intent_path=staged_e_intent_path,
    )
    canonical_contract = canonical_original_confirmatory_q_e_custody_contract(
        contract,
        supervisor_job_directory=job_dir,
    )
    canonical_ready = canonical_original_confirmatory_q_e_custody_ready(
        ready_raw,
        contract=canonical_contract,
        supervisor_job_directory=job_dir if selected_staged_e is not None else None,
        staged_e_intent_path=selected_staged_e,
    )
    retention = canonical_contract["supervisor_retention_policy"]
    q_leaf = _build_q_e_leaf_retained_binding(
        role="q-leaf",
        path=canonical_ready["q_leaf_physical_identity"]["path"],
        target_handle_value=canonical_ready["q_leaf_handle"],
        physical_identity=canonical_ready["q_leaf_physical_identity"],
        retention_policy=retention,
    )
    e_leaf = _build_q_e_leaf_retained_binding(
        role="e-leaf",
        path=canonical_ready["e_leaf_physical_identity"]["path"],
        target_handle_value=canonical_ready["e_leaf_handle"],
        physical_identity=canonical_ready["e_leaf_physical_identity"],
        retention_policy=retention,
    )
    q_ancestors = _build_q_e_ancestor_retained_bindings(
        role="q-ancestor",
        records=canonical_ready["q_ancestor_lease"]["records"],
        target_handles=canonical_ready["q_ancestor_handles"],
        retention_policy=retention,
    )
    e_ancestors = _build_q_e_ancestor_retained_bindings(
        role="e-ancestor",
        records=canonical_ready["e_ancestor_lease"]["records"],
        target_handles=canonical_ready["e_ancestor_handles"],
        retention_policy=retention,
    )
    unsigned = {
        "schema_version": 1,
        "policy": Q_E_CUSTODY_RECEIPT_POLICY,
        "status": _Q_E_CUSTODY_RECEIPT_STATUS,
        "contract_sha256": canonical_contract["contract_sha256"],
        "handoff_root_sha256": canonical_ready["handoff_root_sha256"],
        "supervisor_job_id": canonical_ready["supervisor_job_id"],
        "supervisor_process_identity": canonical_ready["supervisor_process_identity"],
        "controller_process_identity": canonical_ready["controller_process_identity"],
        "windows_boot_time_utc": canonical_ready["windows_boot_time_utc"],
        "q_authority_root_sha256": canonical_ready["q_authority_root_sha256"],
        "q_file_sha256": canonical_ready["q_file_sha256"],
        "e_file_sha256": canonical_ready["e_file_sha256"],
        "q_leaf_retained_binding": q_leaf,
        "q_ancestor_retained_bindings": q_ancestors,
        "e_leaf_retained_binding": e_leaf,
        "e_ancestor_retained_bindings": e_ancestors,
        "exact_controller_process_identity_verified": True,
        "exact_supervisor_process_identity_verified": True,
        "exact_target_access_verified": True,
        "exact_physical_identities_verified": True,
        "independent_verifier_receipt_verified": True,
        "source_custody_overlap_verified": True,
        "supervisor_retention_policy": retention,
        "scientific_inputs_read": False,
        "automatic_retry_allowed": False,
    }
    return {**unsigned, "receipt_root_sha256": canonical_json_sha256(unsigned)}


def canonical_original_confirmatory_q_e_custody_receipt(
    value: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    ready: Mapping[str, Any],
    supervisor_job_directory: Path | None = None,
    staged_e_intent_path: Path | None = None,
) -> dict[str, Any]:
    raw = _mapping(value, role="Q/E custody receipt")
    expected = build_original_confirmatory_q_e_custody_receipt(
        contract=contract,
        ready=ready,
        supervisor_job_directory=supervisor_job_directory,
        staged_e_intent_path=staged_e_intent_path,
    )
    if set(raw) != _Q_E_CUSTODY_RECEIPT_FIELDS or not _strict_json_value_equal(
        raw,
        expected,
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E custody receipt violates its deterministic retained-handle policy"
        )
    if (
        set(raw["q_leaf_retained_binding"]) != _Q_E_CUSTODY_LEAF_RETAINED_BINDING_FIELDS
        or set(raw["e_leaf_retained_binding"])
        != _Q_E_CUSTODY_LEAF_RETAINED_BINDING_FIELDS
        or any(
            set(item) != _Q_E_CUSTODY_ANCESTOR_RETAINED_BINDING_FIELDS
            for item in (
                *raw["q_ancestor_retained_bindings"],
                *raw["e_ancestor_retained_bindings"],
            )
        )
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E custody receipt contains an open nested binding"
        )
    return expected


def build_original_confirmatory_q_e_custody_ack(
    *,
    contract: Mapping[str, Any],
    ready: Mapping[str, Any],
    receipt: Mapping[str, Any],
    supervisor_job_directory: Path | None = None,
    staged_e_intent_path: Path | None = None,
) -> dict[str, Any]:
    ready_raw = _mapping(ready, role="Q/E custody READY for ACK")
    e_identity = _mapping(
        ready_raw.get("e_leaf_physical_identity"),
        role="Q/E custody E identity for ACK",
    )
    e_path = Path(
        _absolute_path(
            e_identity.get("path"),
            role="Q/E custody E path for ACK",
        )
    )
    job_dir, selected_staged_e = _resolve_q_e_ready_layout(
        e_path=e_path,
        supervisor_job_directory=supervisor_job_directory,
        staged_e_intent_path=staged_e_intent_path,
    )
    canonical_contract = canonical_original_confirmatory_q_e_custody_contract(
        contract,
        supervisor_job_directory=job_dir,
    )
    canonical_ready = canonical_original_confirmatory_q_e_custody_ready(
        ready_raw,
        contract=canonical_contract,
        supervisor_job_directory=job_dir if selected_staged_e is not None else None,
        staged_e_intent_path=selected_staged_e,
    )
    canonical_receipt = canonical_original_confirmatory_q_e_custody_receipt(
        receipt,
        contract=canonical_contract,
        ready=canonical_ready,
        supervisor_job_directory=job_dir if selected_staged_e is not None else None,
        staged_e_intent_path=selected_staged_e,
    )
    unsigned = {
        "schema_version": 1,
        "policy": Q_E_CUSTODY_ACK_POLICY,
        "message_type": Q_E_CUSTODY_ACK_MESSAGE_TYPE,
        "contract_sha256": canonical_contract["contract_sha256"],
        "handoff_root_sha256": canonical_ready["handoff_root_sha256"],
        "receipt_path": canonical_contract["receipt_path"],
        "receipt_file_sha256": canonical_json_line_sha256(canonical_receipt),
        "receipt_root_sha256": canonical_receipt["receipt_root_sha256"],
        "supervisor_job_id": canonical_ready["supervisor_job_id"],
        "supervisor_process_identity": canonical_ready["supervisor_process_identity"],
        "controller_process_identity": canonical_ready["controller_process_identity"],
        "all_target_handles_retained": True,
        "scientific_inputs_read": False,
        "automatic_retry_allowed": False,
    }
    return {**unsigned, "ack_sha256": canonical_json_sha256(unsigned)}


def canonical_original_confirmatory_q_e_custody_ack(
    value: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    ready: Mapping[str, Any],
    receipt: Mapping[str, Any],
    supervisor_job_directory: Path | None = None,
    staged_e_intent_path: Path | None = None,
) -> dict[str, Any]:
    raw = _mapping(value, role="Q/E custody ACK")
    expected = build_original_confirmatory_q_e_custody_ack(
        contract=contract,
        ready=ready,
        receipt=receipt,
        supervisor_job_directory=supervisor_job_directory,
        staged_e_intent_path=staged_e_intent_path,
    )
    if set(raw) != _Q_E_CUSTODY_ACK_FIELDS or not _strict_json_value_equal(
        raw,
        expected,
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E custody ACK violates its exact one-shot policy"
        )
    return expected


def build_original_confirmatory_q_e_custody_spec_fields(
    *,
    contract: Mapping[str, Any],
    ready: Mapping[str, Any],
    receipt: Mapping[str, Any],
    supervisor_job_directory: Path | None = None,
    staged_e_intent_path: Path | None = None,
) -> dict[str, Any]:
    ready_raw = _mapping(ready, role="Q/E custody READY for downstream spec")
    e_identity = _mapping(
        ready_raw.get("e_leaf_physical_identity"),
        role="Q/E custody E identity for downstream spec",
    )
    e_path = Path(
        _absolute_path(
            e_identity.get("path"),
            role="Q/E custody E path for downstream spec",
        )
    )
    job_dir, selected_staged_e = _resolve_q_e_ready_layout(
        e_path=e_path,
        supervisor_job_directory=supervisor_job_directory,
        staged_e_intent_path=staged_e_intent_path,
    )
    canonical_contract = canonical_original_confirmatory_q_e_custody_contract(
        contract,
        supervisor_job_directory=job_dir,
    )
    canonical_ready = canonical_original_confirmatory_q_e_custody_ready(
        ready_raw,
        contract=canonical_contract,
        supervisor_job_directory=job_dir if selected_staged_e is not None else None,
        staged_e_intent_path=selected_staged_e,
    )
    canonical_receipt = canonical_original_confirmatory_q_e_custody_receipt(
        receipt,
        contract=canonical_contract,
        ready=canonical_ready,
        supervisor_job_directory=job_dir if selected_staged_e is not None else None,
        staged_e_intent_path=selected_staged_e,
    )
    receipt_binding = {
        "policy": Q_E_CUSTODY_RECEIPT_POLICY,
        "path": canonical_contract["receipt_path"],
        "file_sha256": canonical_json_line_sha256(canonical_receipt),
        "receipt_root_sha256": canonical_receipt["receipt_root_sha256"],
        "handoff_root_sha256": canonical_ready["handoff_root_sha256"],
    }
    return {
        "q_e_custody_contract": canonical_contract,
        "q_e_custody_handoff": canonical_ready,
        "q_e_custody_receipt": receipt_binding,
    }


def canonical_original_confirmatory_q_e_custody_spec_fields(
    value: Mapping[str, Any],
    *,
    supervisor_job_directory: Path | None = None,
    staged_e_intent_path: Path | None = None,
) -> dict[str, Any]:
    raw = _mapping(value, role="Q/E downstream supervisor spec fields")
    if set(raw) != _Q_E_CUSTODY_SPEC_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E downstream supervisor spec has an unexpected field set"
        )
    receipt_binding = _mapping(
        raw["q_e_custody_receipt"],
        role="Q/E downstream spec receipt binding",
    )
    if set(receipt_binding) != _Q_E_CUSTODY_SPEC_RECEIPT_BINDING_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E downstream spec receipt binding has an unexpected field set"
        )
    expected = build_original_confirmatory_q_e_custody_spec_fields(
        contract=_mapping(
            raw["q_e_custody_contract"],
            role="Q/E downstream spec contract",
        ),
        ready=_mapping(
            raw["q_e_custody_handoff"],
            role="Q/E downstream spec handoff",
        ),
        receipt={
            **build_original_confirmatory_q_e_custody_receipt(
                contract=_mapping(
                    raw["q_e_custody_contract"],
                    role="Q/E downstream spec receipt contract",
                ),
                ready=_mapping(
                    raw["q_e_custody_handoff"],
                    role="Q/E downstream spec receipt handoff",
                ),
                supervisor_job_directory=supervisor_job_directory,
                staged_e_intent_path=staged_e_intent_path,
            )
        },
        supervisor_job_directory=supervisor_job_directory,
        staged_e_intent_path=staged_e_intent_path,
    )
    if not _strict_json_value_equal(raw, expected):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E downstream supervisor spec fields violate their exact binding"
        )
    return expected


_EXTERNAL_CODEX_HANDOFF_FIELDS = {
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

_EXTERNAL_SUPERVISOR_SPEC_INPUT_FIELDS = {
    "schema_version",
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
}


def _canonical_original_confirmatory_external_codex_handoff(
    value: Mapping[str, Any],
    *,
    q: Mapping[str, Any],
    e: Mapping[str, Any],
    e_file_sha256: str,
) -> dict[str, Any]:
    raw = _mapping(value, role="external Codex handoff")
    if set(raw) != _EXTERNAL_CODEX_HANDOFF_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "external Codex handoff has an unexpected field set"
        )
    staged_e = Path(q["control_staging_projection"]["e_intent_path"])
    job_dir = Path(e["job"]["supervisor_job_dir"])
    attempt_creation = _mapping(
        e["codex_handoff_attempt_creation_authority"],
        role="E Codex handoff attempt-creation authority",
    )
    attempt_payload = _mapping(
        attempt_creation["payload"],
        role="E Codex handoff attempt-creation payload",
    )
    expected = {
        "policy": EXTERNAL_CODEX_HANDOFF_POLICY,
        "staged_e_intent_path": str(staged_e),
        "staged_e_intent_file_sha256": _sha256(
            e_file_sha256,
            role="external Codex staged E file",
        ),
        "staged_e_intent_core_root_sha256": e["intent_core_sha256"],
        "attempt_creation_authority_payload_sha256": attempt_creation["payload_sha256"],
        "attempt_authority_output_path": attempt_payload[
            "attempt_authority_output_path"
        ],
        "terminal_handoff_receipt_output_path": str(
            job_dir / "external_codex_terminal_handoff.json"
        ),
        "internal_codex_wake_allowed": False,
        "legacy_handoff_session_allowed": False,
        "single_wake_owner": EXTERNAL_CODEX_HANDOFF_SINGLE_WAKE_OWNER,
    }
    if not _strict_json_value_equal(raw, expected):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "external Codex handoff differs from its exact Q/E derivation"
        )
    return expected


def _original_confirmatory_command_tail(
    command: OriginalConfirmatoryCapsuleCommand,
    *,
    capsule: OriginalConfirmatoryExecutionCapsule,
    mode: str,
) -> tuple[str, ...]:
    prefix = (
        str(capsule.python_path),
        *CAPSULE_PYTHON_ISOLATED_FLAGS,
        str(capsule.path),
        mode,
    )
    if tuple(command.argv[: len(prefix)]) != prefix:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{mode} concrete command differs from its immutable capsule prefix"
        )
    return tuple(command.argv[len(prefix) :])


def build_original_confirmatory_external_supervisor_spec_payload_v3(
    *,
    q_authority: Mapping[str, Any],
    e_intent: Mapping[str, Any],
    e_file_sha256: str,
    q_e_custody_spec_fields: Mapping[str, Any],
    external_codex_handoff: Mapping[str, Any],
    pipe_owner_sid: str,
) -> dict[str, Any]:
    """Build the authorization-free exact SPEC52 payload from sealed Q20/E23.

    This is deliberately pure: it performs no filesystem write, environment
    discovery, process launch, authorization, or scientific-outcome read.
    """

    q = canonical_original_confirmatory_q_replacement_v2(q_authority)
    e = canonical_original_confirmatory_e_intent(
        e_intent,
        q_authority=q,
    )
    canonical_e_file_sha256 = _sha256(
        e_file_sha256,
        role="external supervisor sealed E file",
    )
    if canonical_e_file_sha256 != canonical_json_line_sha256(e):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "external supervisor E file SHA-256 differs from canonical E bytes"
        )
    capsule = canonical_original_confirmatory_execution_capsule(q["execution_capsule"])
    capsule_payload = capsule.as_dict()
    release = canonical_original_confirmatory_supervisor_release_binding(
        q["supervisor_release"],
        capsule=capsule,
    )
    job = _mapping(e["job"], role="external supervisor E job")
    job_id = cast(str, job["job_id"])
    job_dir = Path(cast(str, job["supervisor_job_dir"]))
    staged_e = Path(q["control_staging_projection"]["e_intent_path"])
    if (
        job_dir != Path(q["control_staging_projection"]["final_job_dir"])
        or staged_e
        != job_dir.parent.parent
        / CONTROL_STAGING_DIRECTORY_NAME
        / job_id
        / E_INTENT_FILENAME
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "external supervisor staged-E/final-job layout differs from Q20"
        )
    custody_fields = canonical_original_confirmatory_q_e_custody_spec_fields(
        q_e_custody_spec_fields,
        supervisor_job_directory=job_dir,
        staged_e_intent_path=staged_e,
    )
    external_handoff = _canonical_original_confirmatory_external_codex_handoff(
        external_codex_handoff,
        q=q,
        e=e,
        e_file_sha256=canonical_e_file_sha256,
    )
    commands = {
        mode: derive_original_confirmatory_capsule_command_from_e(
            e_intent=e,
            e_file_sha256=canonical_e_file_sha256,
            q_authority=q,
            capsule_mode=mode,
        )
        for mode in CAPSULE_ALLOWED_MODES
    }
    scientific_command = commands[CAPSULE_SCIENTIFIC_MODE]
    preterminal_command = commands[CAPSULE_PRETERMINAL_MODE]
    terminal_command = commands[CAPSULE_TERMINAL_MODE]
    preterminal_tail = _original_confirmatory_command_tail(
        preterminal_command,
        capsule=capsule,
        mode=CAPSULE_PRETERMINAL_MODE,
    )
    terminal_tail = _original_confirmatory_command_tail(
        terminal_command,
        capsule=capsule,
        mode=CAPSULE_TERMINAL_MODE,
    )
    preterminal_pin_path = job_dir / "preterminal_pin.json"
    verifier_stdout_path = job_dir / "verifier.stdout.log"
    supervisor_terminal_path = job_dir / "terminal_receipt.json"
    input_lease_receipt_path = job_dir / "postwake_input_lease_receipt.json"
    composed_terminal_path = job_dir / "composed_terminal.json"
    composed_readback_path = job_dir / "postwake_composed_readback_receipt.json"
    preterminal_pin = build_original_confirmatory_preterminal_pin_contract(
        capsule=capsule,
        verifier_command=preterminal_command,
        verifier_command_tail_argv=preterminal_tail,
        preterminal_pin_receipt_path=preterminal_pin_path,
        preterminal_pin_receipt_max_bytes=1024 * 1024,
    )
    overlap = build_original_confirmatory_preterminal_overlap_handshake_contract(
        handshake_receipt_path=job_dir / "preterminal_overlap_handshake.json",
        ready_line_max_bytes=64 * 1024,
        ack_line_max_bytes=64 * 1024,
        handshake_timeout_ms=30_000,
    )
    job_ancestor = build_original_confirmatory_supervisor_job_ancestor_lease_contract(
        supervisor_root=job_dir.parent.parent,
        job_id=job_id,
    )
    input_lease = build_original_confirmatory_postwake_input_lease_contract(
        supervisor_job_dir=job_dir,
        preterminal_pin_path=preterminal_pin_path,
        verifier_stdout_path=verifier_stdout_path,
        verifier_log_max_bytes=64 * 1024,
        terminal_receipt_path=supervisor_terminal_path,
        lease_receipt_path=input_lease_receipt_path,
        supervisor_job_ancestor_lease_contract=job_ancestor,
    )
    seed = build_original_confirmatory_postwake_custody_seed(
        q_authority_root_sha256=q["q_authority_root_sha256"],
        e_intent_path=staged_e,
        e_intent_file_sha256=canonical_e_file_sha256,
        e_intent_core_sha256=e["intent_core_sha256"],
        supervisor_job_id=job_id,
        supervisor_job_dir=job_dir,
        supervisor_spec_path=job["supervisor_spec_path"],
        launch_nonce=job["launch_nonce"],
        attempt_id=job["attempt_id"],
        run_id=job["run_id"],
        execution_mode=e["lineage"]["execution_mode"],
        retry_of_run_id=e["lineage"]["retry_of_run_id"],
        execution_capsule_contract_sha256=capsule.contract_sha256,
        capsule_sha256=capsule.sha256,
        supervisor_release_root_sha256=release["supervisor_release_root_sha256"],
        terminal_release_root_sha256=capsule.terminal_release_root_sha256,
        supervisor_terminal_receipt_path=supervisor_terminal_path,
        preterminal_pin_receipt_path=preterminal_pin_path,
        postwake_input_lease_receipt_path=input_lease_receipt_path,
        composed_terminal_receipt_path=composed_terminal_path,
        postwake_composed_readback_receipt_path=composed_readback_path,
    )
    process_environment_binding = (
        canonical_original_confirmatory_process_environment_binding(
            e["process_environment_binding"],
            expected_environment=e["expected_launch_environment"],
        )
    )
    postwake_custody = build_original_confirmatory_postwake_custody_handshake_contract(
        custody_seed=seed,
        pipe_owner_sid=pipe_owner_sid,
        readback_receipt_path=composed_readback_path,
        expected_composed_command_sha256=terminal_command.command_sha256,
        expected_composed_cwd=terminal_command.cwd,
        expected_composed_environment_sha256=(
            process_environment_binding.exact_integrity_verifier_environment_sha256
        ),
        ready_max_bytes=64 * 1024,
        ack_max_bytes=64 * 1024,
        terminal_client_arrival_timeout_ms=30 * 60 * 1000,
        custody_exchange_timeout_ms=60 * 1000,
    )
    terminal_custody_projection = _mapping(
        job["terminal_custody_authority_projection"],
        role="external supervisor terminal custody projection",
    )
    artifact_instance = _mapping(
        terminal_custody_projection["outcome_blind_expected_artifact_instance"],
        role="external supervisor outcome-blind artifact instance",
    )
    terminal_composition = build_original_confirmatory_terminal_composition_contract(
        capsule=capsule,
        verifier_command=terminal_command,
        verifier_command_tail_argv=terminal_tail,
        preterminal_pin_contract_sha256=preterminal_pin.contract_sha256,
        preterminal_overlap_handshake_contract=overlap,
        postwake_input_lease_contract=input_lease,
        postwake_custody_seed=seed,
        postwake_custody_handshake_contract=postwake_custody,
        expected_run_directory=artifact_instance["expected_run_directory"],
        expected_terminal_custody_authority_projection=(terminal_custody_projection),
        terminal_client_launcher_release=release["terminal_client_launcher_release"],
        expected_environment=e["expected_launch_environment"],
        process_environment_binding=process_environment_binding,
        supervisor_terminal_receipt_path=supervisor_terminal_path,
        supervisor_terminal_receipt_max_bytes=1024 * 1024,
        verifier_stdout_path=verifier_stdout_path,
        verifier_stdout_max_bytes=64 * 1024,
        preterminal_pin_receipt_path=preterminal_pin_path,
        preterminal_pin_receipt_max_bytes=1024 * 1024,
        postwake_input_lease_receipt_max_bytes=1024 * 1024,
        postwake_composed_readback_receipt_max_bytes=1024 * 1024,
        composed_terminal_receipt_path=composed_terminal_path,
        composed_terminal_receipt_max_bytes=1024 * 1024,
    )
    scientific = scientific_command.as_dict()
    verifier = preterminal_command.as_dict()
    raw_spec = {
        "schema_version": 3,
        "job_id": job_id,
        "process_kind": "confirmatory",
        "external_control_plane_release_root_sha256": release[
            "external_control_plane_release_root_sha256"
        ],
        "external_control_plane_publication_id": release[
            "external_control_plane_publication_id"
        ],
        "external_control_plane_release_qualification_attestation_path": release[
            "external_control_plane_release_qualification_attestation_path"
        ],
        "external_control_plane_release_qualification_attestation_file_sha256": release[
            "external_control_plane_release_qualification_attestation_file_sha256"
        ],
        "external_control_plane_release_qualification_attestation_root_sha256": release[
            "external_control_plane_release_qualification_attestation_root_sha256"
        ],
        "supervisor_code_root": release["supervisor_code_root"],
        "supervisor_state_root": release["supervisor_state_root"],
        "project_root": e["project_root"],
        "program_path": scientific["program_path"],
        "program_sha256": scientific["program_sha256"],
        "argv": list(scientific["argv"]),
        "expected_artifacts": list(artifact_instance["expected_artifacts"]),
        "required_success_roles": list(artifact_instance["required_success_roles"]),
        "integrity_verifier": {
            field: verifier[field]
            for field in ("program_path", "program_sha256", "argv", "cwd")
        },
        "max_log_bytes": ORIGINAL_CONFIRMATORY_PRODUCTION_MAX_LOG_BYTES,
        "main_timeout_ms": ORIGINAL_CONFIRMATORY_PRODUCTION_MAIN_TIMEOUT_MS,
        "verifier_timeout_ms": (ORIGINAL_CONFIRMATORY_PRODUCTION_VERIFIER_TIMEOUT_MS),
        "codex_wake_timeout_seconds": (
            ORIGINAL_CONFIRMATORY_PRODUCTION_CODEX_WAKE_TIMEOUT_SECONDS
        ),
        "codex": None,
        "external_codex_handoff": external_handoff,
        "expected_environment": e["expected_launch_environment"],
        "process_environment_binding": process_environment_binding.as_dict(),
        "preterminal_pin_contract": preterminal_pin.as_dict(),
        "preterminal_overlap_handshake_contract": overlap.as_dict(),
        "postwake_custody_seed": seed.as_dict(),
        "postwake_custody_handshake_contract": postwake_custody.as_dict(),
        "postwake_input_lease_contract": input_lease.as_dict(),
        "terminal_composition_contract": terminal_composition.as_dict(),
        "e_consumption_contract": e["e_consumption_contract"],
        **custody_fields,
        "python_lease_identity": capsule_payload["python_lease_identity"],
        "python_lease_identity_root_sha256": capsule.python_lease_identity_root_sha256,
        "python_ancestor_lease": capsule_payload["python_ancestor_lease"],
        "python_ancestor_lease_root_sha256": capsule.python_ancestor_lease_root_sha256,
        "python_runtime_resolution_policy": capsule.python_runtime_resolution_policy,
        "runtime_python_path": str(capsule.runtime_python_path),
        "runtime_python_sha256": capsule.runtime_python_sha256,
        "runtime_python_lease_identity": capsule_payload[
            "runtime_python_lease_identity"
        ],
        "runtime_python_lease_identity_root_sha256": (
            capsule.runtime_python_lease_identity_root_sha256
        ),
        "runtime_python_ancestor_lease": capsule_payload[
            "runtime_python_ancestor_lease"
        ],
        "runtime_python_ancestor_lease_root_sha256": (
            capsule.runtime_python_ancestor_lease_root_sha256
        ),
        "supervisor_launcher_sha256": release["supervisor_launcher_sha256"],
        "capsule_lease_identity": capsule_payload["capsule_lease_identity"],
        "capsule_lease_identity_root_sha256": capsule.capsule_lease_identity_root_sha256,
        "capsule_ancestor_lease": capsule_payload["capsule_ancestor_lease"],
        "capsule_ancestor_lease_root_sha256": capsule.capsule_ancestor_lease_root_sha256,
    }
    if set(raw_spec) != _EXTERNAL_SUPERVISOR_SPEC_INPUT_FIELDS:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "authority-built external supervisor SPEC52 field inventory differs"
        )
    return raw_spec


def _require_exact_q_e_process_handle_identity(
    process_handle: int,
    expected_identity: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    if os.name != "nt" or process_handle <= 0:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} requires a live Windows process handle"
        )
    expected = _canonical_e_process_identity(expected_identity, role=role)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_process_id = kernel32.GetProcessId
    get_process_id.argtypes = [ctypes.c_void_p]
    get_process_id.restype = ctypes.c_uint32
    observed_pid = int(get_process_id(ctypes.c_void_p(process_handle)))
    if observed_pid == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    image = _windows_process_image_path(process_handle)
    identity, _payload = _read_stable_file(
        image,
        maximum_bytes=max(MAX_CONTROL_FILE_BYTES, int(image.stat().st_size)),
    )
    creation_time_100ns = _windows_process_creation_time_100ns(process_handle)
    if (
        observed_pid != expected["pid"]
        or creation_time_100ns != expected["creation_time_100ns"]
        or _windows_filetime_100ns_to_utc(creation_time_100ns)
        != expected["creation_time_utc"]
        or os.path.normcase(str(image)) != os.path.normcase(expected["program_path"])
        or identity.sha256 != expected["program_sha256"]
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} was reused or differs from its exact process identity"
        )
    return expected


def _require_current_q_e_controller_identity(
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if os.name != "nt":
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E controller identity requires Windows"
        )
    expected = _canonical_e_process_identity(
        expected_identity,
        role="Q/E current controller identity",
    )
    if expected["pid"] != os.getpid():
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E controller PID differs from the current process"
        )
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = ctypes.c_void_p
    return _require_exact_q_e_process_handle_identity(
        cast(int, get_current_process()),
        expected,
        role="Q/E current controller identity",
    )


def _require_q_e_independent_review_receipt(
    q_custody: OriginalConfirmatoryQPublicationCustody,
    *,
    expected_receipt_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    q_payload = _read_all_from_descriptor(
        q_custody.descriptor,
        maximum_bytes=MAX_CONTROL_FILE_BYTES,
    )
    q_authority = canonical_original_confirmatory_q_replacement_v2(
        decode_canonical_json_line(
            q_payload,
            role="Q/E custody retained Q",
        )
    )
    if (
        expected_receipt_sha256
        != q_authority["scientific_authority"]["independent_review_receipt_sha256"]
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E custody independent verifier receipt differs from Q"
        )
    return q_authority, q_payload


def duplicate_original_confirmatory_q_e_custody_to_supervisor(
    custody: (
        OriginalConfirmatoryEPublicationCustody
        | OriginalConfirmatoryStagedEPublicationCustody
    ),
    *,
    contract: Mapping[str, Any],
    supervisor_process_handle: int,
    supervisor_process_identity: Mapping[str, Any],
    controller_process_identity: Mapping[str, Any],
    windows_boot_time_utc: str,
    supervisor_job_id: str,
    independent_verifier_receipt_sha256: str,
    supervisor_job_directory: Path | None = None,
    staged_e_intent_path: Path | None = None,
) -> dict[str, Any]:
    """Duplicate verified Q/E leaf+ancestor custody without closing any source."""

    if os.name != "nt":
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E supervisor custody transfer requires Windows DuplicateHandle"
        )
    if isinstance(custody, OriginalConfirmatoryStagedEPublicationCustody):
        if supervisor_job_directory is None or staged_e_intent_path is None:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "staged Q/E duplication requires explicit E and final-job paths"
            )
        staged_e, final_job, _staging_paths, _final_parent_paths = (
            _require_staged_e_and_final_job_layout(
                staged_e_intent_path=staged_e_intent_path,
                supervisor_job_directory=supervisor_job_directory,
            )
        )
        if (
            custody.path != staged_e
            or custody.final_supervisor_job_directory != final_job
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "staged Q/E duplication paths differ from retained custody"
            )
        custody.require_active(final_job_must_be_absent=True)
        job_dir = final_job
        selected_staged_e: Path | None = staged_e
        e_ancestor_handles = custody.staging_ancestor_handles
    else:
        if supervisor_job_directory is not None or staged_e_intent_path is not None:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "legacy Q/E duplication cannot select a staged layout"
            )
        custody.require_active()
        job_dir = custody.path.parent
        selected_staged_e = None
        e_ancestor_handles = custody.job_ancestor_handles
    supervisor = _require_exact_q_e_process_handle_identity(
        supervisor_process_handle,
        supervisor_process_identity,
        role="Q/E custody target supervisor identity",
    )
    controller = _require_current_q_e_controller_identity(
        controller_process_identity,
    )
    canonical_contract = canonical_original_confirmatory_q_e_custody_contract(
        contract,
        supervisor_job_directory=job_dir,
    )
    if custody.role != "independent-verifier":
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E custody transfer requires independently verified E"
        )
    import msvcrt

    q = custody.q_custody
    if q.role != "independent-verifier":
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E custody transfer requires independently verified Q"
        )
    q_authority, q_payload = _require_q_e_independent_review_receipt(
        q,
        expected_receipt_sha256=independent_verifier_receipt_sha256,
    )
    e_payload = _read_all_from_descriptor(
        custody.descriptor,
        maximum_bytes=MAX_CONTROL_FILE_BYTES,
    )
    e_intent = canonical_original_confirmatory_e_intent(
        decode_canonical_json_line(
            e_payload,
            role="Q/E custody handoff E",
        ),
        q_authority=q_authority,
        expected_q_file_sha256=hashlib.sha256(q_payload).hexdigest(),
    )
    if (
        Path(e_intent["job"]["supervisor_job_dir"]) != job_dir
        or e_intent["job"]["job_id"] != supervisor_job_id
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E custody job differs from verified E"
        )
    source_groups = {
        "q_leaf_handle": (
            msvcrt.get_osfhandle(q.descriptor),
            E_CONSUMPTION_DUPLICATE_TARGET_ACCESS_MASK,
        ),
        "q_ancestor_handles": tuple(
            (handle, EXECUTABLE_ANCESTOR_ACCESS_MASK) for handle in q.ancestor_handles
        ),
        "e_leaf_handle": (
            msvcrt.get_osfhandle(custody.descriptor),
            E_CONSUMPTION_DUPLICATE_TARGET_ACCESS_MASK,
        ),
        "e_ancestor_handles": tuple(
            (handle, EXECUTABLE_ANCESTOR_ACCESS_MASK) for handle in e_ancestor_handles
        ),
    }
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = ctypes.c_void_p
    source_process = get_current_process()
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
    remote_values: list[int] = []

    def close_remote_values() -> None:
        for remote in reversed(remote_values):
            local = ctypes.c_void_p()
            if (
                duplicate_handle(
                    ctypes.c_void_p(supervisor_process_handle),
                    ctypes.c_void_p(remote),
                    source_process,
                    ctypes.byref(local),
                    0,
                    0,
                    0x1 | 0x2,
                )
                and local.value
            ):
                kernel32.CloseHandle(local)

    def duplicate_one(source_handle: int, access_mask: int) -> int:
        target = ctypes.c_void_p()
        if not duplicate_handle(
            source_process,
            ctypes.c_void_p(source_handle),
            ctypes.c_void_p(supervisor_process_handle),
            ctypes.byref(target),
            access_mask,
            0,
            0,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        value = cast(int, target.value)
        remote_values.append(value)
        return value

    try:
        q_leaf = duplicate_one(*cast(tuple[int, int], source_groups["q_leaf_handle"]))
        q_ancestors = [
            duplicate_one(handle, access)
            for handle, access in cast(
                tuple[tuple[int, int], ...],
                source_groups["q_ancestor_handles"],
            )
        ]
        e_leaf = duplicate_one(*cast(tuple[int, int], source_groups["e_leaf_handle"]))
        e_ancestors = [
            duplicate_one(handle, access)
            for handle, access in cast(
                tuple[tuple[int, int], ...],
                source_groups["e_ancestor_handles"],
            )
        ]
        _require_exact_q_e_process_handle_identity(
            supervisor_process_handle,
            supervisor,
            role="post-DuplicateHandle Q/E target supervisor identity",
        )
        if isinstance(custody, OriginalConfirmatoryStagedEPublicationCustody):
            custody.require_active(final_job_must_be_absent=True)
        else:
            custody.require_active()
        return build_original_confirmatory_q_e_custody_ready(
            contract=canonical_contract,
            supervisor_job_id=supervisor_job_id,
            supervisor_process_identity=supervisor,
            controller_process_identity=controller,
            windows_boot_time_utc=windows_boot_time_utc,
            q_authority_root_sha256=q_authority["q_authority_root_sha256"],
            q_file_sha256=q.publication_identity["sha256"],
            e_file_sha256=custody.publication_identity["sha256"],
            q_leaf_physical_identity=q.publication_identity,
            q_ancestor_lease=q_authority["publication_ancestor_lease"],
            e_leaf_physical_identity=custody.publication_identity,
            e_ancestor_lease=_build_q_e_e_ancestor_lease(custody),
            q_leaf_handle=q_leaf,
            q_ancestor_handles=q_ancestors,
            e_leaf_handle=e_leaf,
            e_ancestor_handles=e_ancestors,
            independent_verifier_receipt_sha256=(independent_verifier_receipt_sha256),
            supervisor_job_directory=job_dir,
            staged_e_intent_path=selected_staged_e,
        )
    except BaseException:
        # Close only target duplicates; source custody remains live and retry is forbidden.
        close_remote_values()
        raise


def duplicate_original_confirmatory_staged_q_e_custody_to_supervisor(
    custody: OriginalConfirmatoryStagedEPublicationCustody,
    *,
    contract: Mapping[str, Any],
    supervisor_process_handle: int,
    supervisor_process_identity: Mapping[str, Any],
    controller_process_identity: Mapping[str, Any],
    windows_boot_time_utc: str,
    supervisor_job_id: str,
    independent_verifier_receipt_sha256: str,
    supervisor_job_directory: Path,
    staged_e_intent_path: Path,
) -> dict[str, Any]:
    """Duplicate exactly eight Q/E roles from explicit staged/final domains."""

    return duplicate_original_confirmatory_q_e_custody_to_supervisor(
        custody,
        contract=contract,
        supervisor_process_handle=supervisor_process_handle,
        supervisor_process_identity=supervisor_process_identity,
        controller_process_identity=controller_process_identity,
        windows_boot_time_utc=windows_boot_time_utc,
        supervisor_job_id=supervisor_job_id,
        independent_verifier_receipt_sha256=independent_verifier_receipt_sha256,
        supervisor_job_directory=supervisor_job_directory,
        staged_e_intent_path=staged_e_intent_path,
    )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryQESuspendedSupervisorTarget:
    """One suspended supervisor plus its exact bounded custody transport seams.

    ``finalize_success_once`` is deliberately an opaque downstream seam here.
    A production Option-A factory must not return from it until the outer Job
    custody sequence has completed ACCEPTED, controller round-trip validation,
    source-handle close, COMMIT, and sealed COMMITTED-receipt readback.  This
    upstream authority does not duplicate those downstream Job schemas.
    """

    process_handle: int
    process_identity: Mapping[str, Any]
    windows_boot_time_utc: str
    suspended: bool
    transport: str
    exact_job_object_membership_verified: bool
    bounded_anonymous_pipes_created_before_process: bool
    automatic_retry_allowed: bool
    finalize_downstream_spec_create_new: Callable[[Mapping[str, Any]], None]
    resume_supervisor_once: Callable[[], None]
    send_ready_line_once: Callable[[bytes], None]
    receive_ack_line_once: Callable[[int], bytes]
    read_receipt_once: Callable[[Path, int], bytes]
    finalize_success_once: Callable[[], None]
    abort_supervisor_on_failure: Callable[[], None]


def _decode_q_e_bounded_canonical_line(
    payload: bytes,
    *,
    maximum_bytes: int,
    role: str,
) -> dict[str, Any]:
    if (
        not isinstance(payload, bytes)
        or not payload
        or len(payload) > maximum_bytes
        or not payload.endswith(b"\n")
        or b"\n" in payload[:-1]
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} is not one bounded canonical line"
        )
    value = decode_canonical_json_line(payload, role=role)
    if canonical_json_line_bytes(value) != payload:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            f"{role} differs from canonical bytes"
        )
    return value


def orchestrate_original_confirmatory_q_e_custody_once(
    custody: (
        OriginalConfirmatoryEPublicationCustody
        | OriginalConfirmatoryStagedEPublicationCustody
    ),
    *,
    contract: Mapping[str, Any],
    controller_process_identity: Mapping[str, Any],
    supervisor_job_id: str,
    independent_verifier_receipt_sha256: str,
    start_supervisor_suspended_once: Callable[
        [], OriginalConfirmatoryQESuspendedSupervisorTarget
    ],
    supervisor_job_directory: Path | None = None,
    staged_e_intent_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Perform the one legal suspended-target Q/E handoff, with no retry."""

    if isinstance(custody, OriginalConfirmatoryStagedEPublicationCustody):
        if supervisor_job_directory is None or staged_e_intent_path is None:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "staged Q/E handoff requires explicit E and final-job paths"
            )
        staged_e, final_job, _staging_paths, _final_parent_paths = (
            _require_staged_e_and_final_job_layout(
                staged_e_intent_path=staged_e_intent_path,
                supervisor_job_directory=supervisor_job_directory,
            )
        )
        if (
            custody.path != staged_e
            or custody.final_supervisor_job_directory != final_job
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "staged Q/E handoff paths differ from retained custody"
            )
        custody.require_active(final_job_must_be_absent=True)
        job_dir = final_job
        selected_staged_e: Path | None = staged_e
    else:
        if supervisor_job_directory is not None or staged_e_intent_path is not None:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "non-staged custody cannot override its uniquely derived layout"
            )
        custody.require_active()
        job_dir, selected_staged_e = _resolve_q_e_ready_layout(
            e_path=custody.path,
            supervisor_job_directory=None,
            staged_e_intent_path=None,
        )
    if custody.role != "independent-verifier":
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E one-shot handoff requires independently verified E custody"
        )
    q_custody = custody.q_custody
    if q_custody.role != "independent-verifier":
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q/E one-shot handoff requires independently verified Q custody"
        )
    _q_authority, _q_payload = _require_q_e_independent_review_receipt(
        q_custody,
        expected_receipt_sha256=independent_verifier_receipt_sha256,
    )
    canonical_contract = canonical_original_confirmatory_q_e_custody_contract(
        contract,
        supervisor_job_directory=job_dir,
    )
    controller = _require_current_q_e_controller_identity(controller_process_identity)
    target: OriginalConfirmatoryQESuspendedSupervisorTarget | None = None
    ack_validated = False
    try:
        target = start_supervisor_suspended_once()
        if (
            not isinstance(target, OriginalConfirmatoryQESuspendedSupervisorTarget)
            or target.suspended is not True
            or target.transport != Q_E_CUSTODY_TRANSPORT
            or target.exact_job_object_membership_verified is not True
            or target.bounded_anonymous_pipes_created_before_process is not True
            or target.automatic_retry_allowed is not False
            or not callable(target.finalize_success_once)
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "Q/E target was not created once in the exact suspended pipe mode"
            )
        supervisor = _require_exact_q_e_process_handle_identity(
            target.process_handle,
            target.process_identity,
            role="pre-DuplicateHandle suspended Q/E supervisor identity",
        )
        ready = duplicate_original_confirmatory_q_e_custody_to_supervisor(
            custody,
            contract=canonical_contract,
            supervisor_process_handle=target.process_handle,
            supervisor_process_identity=supervisor,
            controller_process_identity=controller,
            windows_boot_time_utc=target.windows_boot_time_utc,
            supervisor_job_id=supervisor_job_id,
            independent_verifier_receipt_sha256=(independent_verifier_receipt_sha256),
            supervisor_job_directory=job_dir if selected_staged_e is not None else None,
            staged_e_intent_path=selected_staged_e,
        )
        receipt_expected = build_original_confirmatory_q_e_custody_receipt(
            contract=canonical_contract,
            ready=ready,
            supervisor_job_directory=job_dir if selected_staged_e is not None else None,
            staged_e_intent_path=selected_staged_e,
        )
        downstream_spec_fields = build_original_confirmatory_q_e_custody_spec_fields(
            contract=canonical_contract,
            ready=ready,
            receipt=receipt_expected,
            supervisor_job_directory=job_dir if selected_staged_e is not None else None,
            staged_e_intent_path=selected_staged_e,
        )
        target.finalize_downstream_spec_create_new(downstream_spec_fields)
        if (
            not _strict_json_value_equal(
                canonical_original_confirmatory_q_e_custody_spec_fields(
                    downstream_spec_fields,
                    supervisor_job_directory=(
                        job_dir if selected_staged_e is not None else None
                    ),
                    staged_e_intent_path=selected_staged_e,
                ),
                downstream_spec_fields,
            )
            or not _strict_json_value_equal(
                canonical_original_confirmatory_q_e_custody_ready(
                    ready,
                    contract=canonical_contract,
                    supervisor_job_directory=(
                        job_dir if selected_staged_e is not None else None
                    ),
                    staged_e_intent_path=selected_staged_e,
                ),
                ready,
            )
            or not _strict_json_value_equal(
                build_original_confirmatory_q_e_custody_receipt(
                    contract=canonical_contract,
                    ready=ready,
                    supervisor_job_directory=(
                        job_dir if selected_staged_e is not None else None
                    ),
                    staged_e_intent_path=selected_staged_e,
                ),
                receipt_expected,
            )
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "Q/E handoff changed while downstream spec was finalized"
            )
        _require_exact_q_e_process_handle_identity(
            target.process_handle,
            supervisor,
            role="pre-resume suspended Q/E supervisor identity",
        )
        if isinstance(custody, OriginalConfirmatoryStagedEPublicationCustody):
            custody.require_active(final_job_must_be_absent=True)
        else:
            custody.require_active()
        target.resume_supervisor_once()
        ready_line = canonical_json_line_bytes(ready)
        if len(ready_line) > canonical_contract["ready_max_bytes"]:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "Q/E custody READY exceeds its exact pipe bound"
            )
        target.send_ready_line_once(ready_line)
        ack_line = target.receive_ack_line_once(canonical_contract["ack_max_bytes"])
        ack_value = _decode_q_e_bounded_canonical_line(
            ack_line,
            maximum_bytes=canonical_contract["ack_max_bytes"],
            role="Q/E custody ACK",
        )
        receipt_line = target.read_receipt_once(
            Path(canonical_contract["receipt_path"]),
            canonical_contract["ack_max_bytes"],
        )
        receipt_value = _decode_q_e_bounded_canonical_line(
            receipt_line,
            maximum_bytes=canonical_contract["ack_max_bytes"],
            role="Q/E custody receipt",
        )
        receipt = canonical_original_confirmatory_q_e_custody_receipt(
            receipt_value,
            contract=canonical_contract,
            ready=ready,
            supervisor_job_directory=job_dir if selected_staged_e is not None else None,
            staged_e_intent_path=selected_staged_e,
        )
        if not _strict_json_value_equal(
            receipt,
            receipt_expected,
        ) or hashlib.sha256(receipt_line).hexdigest() != canonical_json_line_sha256(
            receipt
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "Q/E deterministic receipt differs from its prebound bytes"
            )
        ack = canonical_original_confirmatory_q_e_custody_ack(
            ack_value,
            contract=canonical_contract,
            ready=ready,
            receipt=receipt,
            supervisor_job_directory=job_dir if selected_staged_e is not None else None,
            staged_e_intent_path=selected_staged_e,
        )
        _require_exact_q_e_process_handle_identity(
            target.process_handle,
            supervisor,
            role="post-ACK Q/E supervisor identity",
        )
        if isinstance(custody, OriginalConfirmatoryStagedEPublicationCustody):
            custody.require_active(final_job_must_be_absent=False)
        else:
            custody.require_active()
        ack_validated = True
        try:
            custody.close()
            q_custody.close()
            # The downstream Option-A implementation owns the later
            # ACCEPTED -> COMMIT -> COMMITTED protocol.  Returning from this
            # callback is the only success signal available to this layer.
            target.finalize_success_once()
        except BaseException as exc:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "Q/E custody is in permanent ambiguous STOP after validated ACK; "
                "ACCEPTED-to-COMMITTED success finalization failed or remained "
                "ambiguous and automatic retry is forbidden"
            ) from exc
        return ready, receipt, ack
    except BaseException:
        if target is not None and not ack_validated:
            target.abort_supervisor_on_failure()
        raise


def orchestrate_original_confirmatory_staged_q_e_custody_once(
    custody: OriginalConfirmatoryStagedEPublicationCustody,
    *,
    contract: Mapping[str, Any],
    controller_process_identity: Mapping[str, Any],
    supervisor_job_id: str,
    independent_verifier_receipt_sha256: str,
    start_supervisor_suspended_once: Callable[
        [], OriginalConfirmatoryQESuspendedSupervisorTarget
    ],
    supervisor_job_directory: Path,
    staged_e_intent_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Perform the explicit staged-E/final-job one-shot custody handoff."""

    return orchestrate_original_confirmatory_q_e_custody_once(
        custody,
        contract=contract,
        controller_process_identity=controller_process_identity,
        supervisor_job_id=supervisor_job_id,
        independent_verifier_receipt_sha256=independent_verifier_receipt_sha256,
        start_supervisor_suspended_once=start_supervisor_suspended_once,
        supervisor_job_directory=supervisor_job_directory,
        staged_e_intent_path=staged_e_intent_path,
    )


def _read_all_from_descriptor(
    descriptor: int,
    *,
    maximum_bytes: int,
) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = maximum_bytes + 1
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    if len(payload) > maximum_bytes:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "retained control file exceeds its exact bound"
        )
    os.lseek(descriptor, 0, os.SEEK_SET)
    return payload


def publish_original_confirmatory_q_replacement_v2_once(
    value: Mapping[str, Any],
) -> tuple[Path, str, dict[str, Any], OriginalConfirmatoryQPublicationCustody]:
    canonical = canonical_original_confirmatory_q_replacement_v2(value)
    path = Path(canonical["q_path"])
    payload = canonical_json_line_bytes(canonical)
    if not payload or len(payload) > MAX_CONTROL_FILE_BYTES:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q payload exceeds its pre-publication bound"
        )
    if not path.parent.is_dir():
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q publication parent must already exist"
        )
    ancestor_handles, windows_native = _open_control_publication_ancestor_handles(
        canonical["publication_ancestor_lease"]
    )
    descriptor = -1
    try:
        descriptor = _create_new_control_publication_descriptor(path)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Q CREATE_NEW write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        readback = b""
        while len(readback) <= MAX_CONTROL_FILE_BYTES:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, MAX_CONTROL_FILE_BYTES + 1 - len(readback)),
            )
            if not chunk:
                break
            readback += chunk
        if readback != payload:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "Q same-handle publication readback differs from written bytes"
            )
        identity = _control_publication_identity_from_descriptor(
            descriptor,
            path=path,
            payload=payload,
            write_handle_retained=True,
        )
        custody = OriginalConfirmatoryQPublicationCustody(
            path=path,
            descriptor=descriptor,
            ancestor_handles=ancestor_handles,
            windows_native_ancestor_handles=windows_native,
            publication_identity=identity,
            role="author",
        )
        descriptor = -1
        ancestor_handles = ()
        return path, identity["sha256"], identity, custody
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if ancestor_handles:
            _close_control_publication_ancestor_handles(
                ancestor_handles,
                windows_native=windows_native,
            )


def require_original_confirmatory_q_replacement_v2(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_publication_identity: Mapping[str, Any],
    author_custody: OriginalConfirmatoryQPublicationCustody,
) -> tuple[dict[str, Any], str, OriginalConfirmatoryQPublicationCustody]:
    canonical_path = Path(_absolute_path(str(path), role="Q replacement-v2 read path"))
    author_custody.require_active()
    if author_custody.role != "author" or author_custody.path != canonical_path:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q verifier lacks exact overlapping author custody"
        )
    expected_identity = _mapping(
        expected_publication_identity,
        role="Q expected publication identity",
    )
    if set(
        expected_identity
    ) != _CONTROL_PUBLICATION_IDENTITY_FIELDS or not _strict_json_value_equal(
        expected_identity,
        author_custody.publication_identity,
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "Q verifier expected identity differs from author custody"
        )
    q_from_author = canonical_original_confirmatory_q_replacement_v2(
        decode_canonical_json_line(
            _read_all_from_descriptor(
                author_custody.descriptor,
                maximum_bytes=MAX_CONTROL_FILE_BYTES,
            ),
            role="Q author retained file",
        )
    )
    ancestor_handles, windows_native = _open_control_publication_ancestor_handles(
        q_from_author["publication_ancestor_lease"]
    )
    descriptor = -1
    transition_guard = -1
    try:
        transition_guard = _open_control_publication_transition_guard(canonical_path)
        guard_payload = _read_all_from_descriptor(
            transition_guard,
            maximum_bytes=MAX_CONTROL_FILE_BYTES,
        )
        guard_identity = _control_publication_identity_from_descriptor(
            transition_guard,
            path=canonical_path,
            payload=guard_payload,
            write_handle_retained=False,
        )
        if not _strict_json_value_equal(
            guard_identity,
            {
                **expected_identity,
                "write_handle_retained": False,
            },
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "Q transition guard opened a different physical identity"
            )
        author_custody.close()
        descriptor = _open_read_descriptor(canonical_path)
        os.close(transition_guard)
        transition_guard = -1
        payload = _read_all_from_descriptor(
            descriptor,
            maximum_bytes=MAX_CONTROL_FILE_BYTES,
        )
        file_sha256 = hashlib.sha256(payload).hexdigest()
        if file_sha256 != _sha256(
            expected_file_sha256,
            role="Q replacement-v2 expected file",
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "Q replacement-v2 file SHA-256 differs from author evidence"
            )
        identity = _control_publication_identity_from_descriptor(
            descriptor,
            path=canonical_path,
            payload=payload,
            write_handle_retained=False,
        )
        comparable_expected = {
            **expected_identity,
            "write_handle_retained": False,
        }
        if not _strict_json_value_equal(identity, comparable_expected):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "Q verifier opened a different physical file identity"
            )
        canonical = canonical_original_confirmatory_q_replacement_v2(
            decode_canonical_json_line(payload, role="Q replacement-v2 file")
        )
        if (
            not _strict_json_value_equal(
                canonical,
                q_from_author,
            )
            or Path(canonical["q_path"]) != canonical_path
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "Q verifier canonical payload differs from author custody"
            )
        verifier_custody = OriginalConfirmatoryQPublicationCustody(
            path=canonical_path,
            descriptor=descriptor,
            ancestor_handles=ancestor_handles,
            windows_native_ancestor_handles=windows_native,
            publication_identity=identity,
            role="independent-verifier",
        )
        descriptor = -1
        ancestor_handles = ()
        return canonical, file_sha256, verifier_custody
    finally:
        if transition_guard >= 0:
            os.close(transition_guard)
        if descriptor >= 0:
            os.close(descriptor)
        if ancestor_handles:
            _close_control_publication_ancestor_handles(
                ancestor_handles,
                windows_native=windows_native,
            )


def publish_original_confirmatory_e_intent_once(
    value: Mapping[str, Any],
    *,
    q_authority: Mapping[str, Any],
    expected_q_file_sha256: str,
    q_verifier_custody: OriginalConfirmatoryQPublicationCustody,
) -> tuple[
    Path,
    str,
    dict[str, Any],
    OriginalConfirmatoryEPublicationCustody,
]:
    """CREATE_NEW E while retaining its leaf, job directory, and verified Q."""

    canonical = canonical_original_confirmatory_e_intent(
        value,
        q_authority=q_authority,
        expected_q_file_sha256=expected_q_file_sha256,
    )
    payload = canonical_json_line_bytes(canonical)
    if not payload or len(payload) > MAX_CONTROL_FILE_BYTES:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E payload exceeds its pre-publication bound"
        )
    q_verifier_custody.require_active()
    if (
        q_verifier_custody.role != "independent-verifier"
        or q_verifier_custody.path != Path(canonical["q_authority"]["path"])
        or q_verifier_custody.publication_identity["sha256"]
        != canonical["q_authority"]["file_sha256"]
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E publisher lacks exact retained independently verified Q bytes"
        )
    path = Path(canonical["job"]["supervisor_job_dir"]) / E_INTENT_FILENAME
    if path.parent != Path(canonical["job"]["supervisor_job_dir"]):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E publication path escaped the bound supervisor job directory"
        )
    job_handles: tuple[int, ...] = ()
    windows_native = False
    job_facts: tuple[tuple[int, str, int], ...] = ()
    job_paths: tuple[Path, ...] = ()
    descriptor = -1
    try:
        job_handles, windows_native, job_facts, job_paths = (
            _open_e_job_ancestor_handles(path.parent)
        )
        descriptor = _create_new_control_publication_descriptor(path)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("E CREATE_NEW write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        readback = _read_all_from_descriptor(
            descriptor,
            maximum_bytes=MAX_CONTROL_FILE_BYTES,
        )
        if readback != payload:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "E same-handle publication readback differs from written bytes"
            )
        _revalidate_e_job_ancestor_handles(
            job_handles,
            windows_native=windows_native,
            expected_facts=job_facts,
        )
        identity = _control_publication_identity_from_descriptor(
            descriptor,
            path=path,
            payload=payload,
            write_handle_retained=True,
        )
        custody = OriginalConfirmatoryEPublicationCustody(
            path=path,
            descriptor=descriptor,
            job_ancestor_handles=job_handles,
            windows_native_job_ancestor_handles=windows_native,
            job_ancestor_facts=job_facts,
            job_ancestor_paths=job_paths,
            publication_identity=identity,
            q_custody=q_verifier_custody,
            role="author",
        )
        descriptor = -1
        job_handles = ()
        return path, identity["sha256"], identity, custody
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        for handle in reversed(job_handles):
            _close_e_job_directory_handle(
                handle,
                windows_native=windows_native,
            )


def require_original_confirmatory_e_intent(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_publication_identity: Mapping[str, Any],
    q_authority: Mapping[str, Any],
    expected_q_file_sha256: str,
    author_custody: OriginalConfirmatoryEPublicationCustody,
) -> tuple[dict[str, Any], str, OriginalConfirmatoryEPublicationCustody]:
    """Independently read E with uninterrupted guard and Q/job/leaf custody."""

    canonical_path = Path(
        _absolute_path(str(path), role="E intent independent read path")
    )
    author_custody.require_active()
    if author_custody.role != "author" or author_custody.path != canonical_path:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E verifier lacks exact overlapping author custody"
        )
    expected_identity = _mapping(
        expected_publication_identity,
        role="E expected publication identity",
    )
    if set(
        expected_identity
    ) != _CONTROL_PUBLICATION_IDENTITY_FIELDS or not _strict_json_value_equal(
        expected_identity,
        author_custody.publication_identity,
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E verifier expected identity differs from author custody"
        )
    e_from_author = canonical_original_confirmatory_e_intent(
        decode_canonical_json_line(
            _read_all_from_descriptor(
                author_custody.descriptor,
                maximum_bytes=MAX_CONTROL_FILE_BYTES,
            ),
            role="E author retained file",
        ),
        q_authority=q_authority,
        expected_q_file_sha256=expected_q_file_sha256,
    )
    job_handles: tuple[int, ...] = ()
    windows_native = False
    job_facts: tuple[tuple[int, str, int], ...] = ()
    job_paths: tuple[Path, ...] = ()
    descriptor = -1
    transition_guard = -1
    try:
        job_handles, windows_native, job_facts, job_paths = (
            _open_e_job_ancestor_handles(canonical_path.parent)
        )
        transition_guard = _open_control_publication_transition_guard(canonical_path)
        guard_payload = _read_all_from_descriptor(
            transition_guard,
            maximum_bytes=MAX_CONTROL_FILE_BYTES,
        )
        guard_identity = _control_publication_identity_from_descriptor(
            transition_guard,
            path=canonical_path,
            payload=guard_payload,
            write_handle_retained=False,
        )
        if not _strict_json_value_equal(
            guard_identity,
            {
                **expected_identity,
                "write_handle_retained": False,
            },
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "E transition guard opened a different physical identity"
            )
        author_custody.close()
        descriptor = _open_read_descriptor(canonical_path)
        os.close(transition_guard)
        transition_guard = -1
        payload = _read_all_from_descriptor(
            descriptor,
            maximum_bytes=MAX_CONTROL_FILE_BYTES,
        )
        file_sha256 = hashlib.sha256(payload).hexdigest()
        if file_sha256 != _sha256(
            expected_file_sha256,
            role="E expected file SHA-256",
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "E file hash differs from author evidence"
            )
        identity = _control_publication_identity_from_descriptor(
            descriptor,
            path=canonical_path,
            payload=payload,
            write_handle_retained=False,
        )
        if not _strict_json_value_equal(
            identity,
            {
                **expected_identity,
                "write_handle_retained": False,
            },
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "E verifier opened a different physical file identity"
            )
        canonical = canonical_original_confirmatory_e_intent(
            decode_canonical_json_line(payload, role="E intent file"),
            q_authority=q_authority,
            expected_q_file_sha256=expected_q_file_sha256,
        )
        if not _strict_json_value_equal(canonical, e_from_author):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "E independent payload differs from author-held E"
            )
        verifier_custody = OriginalConfirmatoryEPublicationCustody(
            path=canonical_path,
            descriptor=descriptor,
            job_ancestor_handles=job_handles,
            windows_native_job_ancestor_handles=windows_native,
            job_ancestor_facts=job_facts,
            job_ancestor_paths=job_paths,
            publication_identity=identity,
            q_custody=author_custody.q_custody,
            role="independent-verifier",
        )
        descriptor = -1
        job_handles = ()
        return canonical, file_sha256, verifier_custody
    finally:
        if transition_guard >= 0:
            os.close(transition_guard)
        if descriptor >= 0:
            os.close(descriptor)
        for handle in reversed(job_handles):
            _close_e_job_directory_handle(
                handle,
                windows_native=windows_native,
            )


def _require_staged_e_matches_q_and_payload(
    *,
    canonical_e: Mapping[str, Any],
    canonical_q: Mapping[str, Any],
    staged_e_intent_path: str | Path,
    supervisor_job_directory: str | Path,
) -> tuple[Path, Path, tuple[Path, Path, Path], tuple[Path, Path]]:
    staged_e, final_job, staging_paths, final_parent_paths = (
        _require_staged_e_and_final_job_layout(
            staged_e_intent_path=staged_e_intent_path,
            supervisor_job_directory=supervisor_job_directory,
        )
    )
    control_staging = _mapping(
        canonical_q["control_staging_projection"],
        role="staged E Q control-staging projection",
    )
    job = _mapping(canonical_e["job"], role="staged E final supervisor job")
    if (
        Path(control_staging["e_intent_path"]) != staged_e
        or Path(control_staging["final_job_dir"]) != final_job
        or Path(job["supervisor_job_dir"]) != final_job
        or job["job_id"] != final_job.name
        or staged_e.parent.name != final_job.name
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "staged E publication and final job differ from Q/E authority"
        )
    return staged_e, final_job, staging_paths, final_parent_paths


def publish_original_confirmatory_staged_e_intent_once(
    value: Mapping[str, Any],
    *,
    publication_path: str | Path,
    supervisor_job_directory: str | Path,
    q_authority: Mapping[str, Any],
    expected_q_file_sha256: str,
    q_verifier_custody: OriginalConfirmatoryQPublicationCustody,
) -> tuple[
    Path,
    str,
    dict[str, Any],
    OriginalConfirmatoryStagedEPublicationCustody,
]:
    """CREATE_NEW E in control staging while final ``jobs/<id>`` stays absent."""

    canonical_q = canonical_original_confirmatory_q_replacement_v2(q_authority)
    canonical_e = canonical_original_confirmatory_e_intent(
        value,
        q_authority=canonical_q,
        expected_q_file_sha256=expected_q_file_sha256,
    )
    payload = canonical_json_line_bytes(canonical_e)
    if not payload or len(payload) > MAX_CONTROL_FILE_BYTES:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "staged E payload exceeds its pre-publication bound"
        )
    q_verifier_custody.require_active()
    if (
        q_verifier_custody.role != "independent-verifier"
        or q_verifier_custody.path != Path(canonical_e["q_authority"]["path"])
        or q_verifier_custody.publication_identity["sha256"]
        != canonical_e["q_authority"]["file_sha256"]
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "staged E publisher lacks exact retained independently verified Q bytes"
        )
    staged_e, final_job, staging_paths, final_parent_paths = (
        _require_staged_e_matches_q_and_payload(
            canonical_e=canonical_e,
            canonical_q=canonical_q,
            staged_e_intent_path=publication_path,
            supervisor_job_directory=supervisor_job_directory,
        )
    )
    if _path_lexists(final_job):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "final supervisor job must be absent before staged E publication"
        )
    staging_handles: tuple[int, ...] = ()
    staging_windows_native = False
    staging_facts: tuple[tuple[int, str, int], ...] = ()
    final_handles: tuple[int, ...] = ()
    final_windows_native = False
    final_facts: tuple[tuple[int, str, int], ...] = ()
    descriptor = -1
    try:
        (
            staging_handles,
            staging_windows_native,
            staging_facts,
            observed_staging_paths,
        ) = _open_exact_directory_chain(staging_paths, role="staged E")
        (
            final_handles,
            final_windows_native,
            final_facts,
            observed_final_paths,
        ) = _open_exact_directory_chain(
            final_parent_paths, role="final supervisor job parent"
        )
        if _path_lexists(final_job):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "final supervisor job appeared before staged E CREATE_NEW"
            )
        descriptor = _create_new_control_publication_descriptor(staged_e)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("staged E CREATE_NEW write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        if (
            _read_all_from_descriptor(descriptor, maximum_bytes=MAX_CONTROL_FILE_BYTES)
            != payload
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "staged E same-handle readback differs from written bytes"
            )
        _revalidate_exact_directory_chain(
            staging_handles,
            windows_native=staging_windows_native,
            expected_facts=staging_facts,
            expected_count=3,
            role="staged E ancestor",
        )
        _revalidate_exact_directory_chain(
            final_handles,
            windows_native=final_windows_native,
            expected_facts=final_facts,
            expected_count=2,
            role="final supervisor job parent",
        )
        if _path_lexists(final_job):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "final supervisor job appeared during staged E publication"
            )
        identity = _control_publication_identity_from_descriptor(
            descriptor,
            path=staged_e,
            payload=payload,
            write_handle_retained=True,
        )
        custody = OriginalConfirmatoryStagedEPublicationCustody(
            path=staged_e,
            final_supervisor_job_directory=final_job,
            descriptor=descriptor,
            staging_ancestor_handles=staging_handles,
            windows_native_staging_ancestor_handles=staging_windows_native,
            staging_ancestor_facts=staging_facts,
            staging_ancestor_paths=observed_staging_paths,
            final_job_parent_handles=final_handles,
            windows_native_final_job_parent_handles=final_windows_native,
            final_job_parent_facts=final_facts,
            final_job_parent_paths=observed_final_paths,
            publication_identity=identity,
            q_custody=q_verifier_custody,
            role="author",
        )
        custody.require_active(final_job_must_be_absent=True)
        descriptor = -1
        staging_handles = ()
        final_handles = ()
        return staged_e, identity["sha256"], identity, custody
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        for handles, windows_native in (
            (staging_handles, staging_windows_native),
            (final_handles, final_windows_native),
        ):
            for handle in reversed(handles):
                _close_e_job_directory_handle(handle, windows_native=windows_native)


def require_original_confirmatory_staged_e_intent(
    path: str | Path,
    *,
    supervisor_job_directory: str | Path,
    expected_file_sha256: str,
    expected_publication_identity: Mapping[str, Any],
    q_authority: Mapping[str, Any],
    expected_q_file_sha256: str,
    author_custody: OriginalConfirmatoryStagedEPublicationCustody,
) -> tuple[
    dict[str, Any],
    str,
    OriginalConfirmatoryStagedEPublicationCustody,
]:
    """Independently reopen staged E with uninterrupted stage/final-parent custody."""

    canonical_q = canonical_original_confirmatory_q_replacement_v2(q_authority)
    canonical_path = Path(
        _absolute_path(str(path), role="staged E independent read path")
    )
    final_job = Path(
        _absolute_path(
            str(supervisor_job_directory),
            role="staged E independent final job directory",
        )
    )
    author_custody.require_active(final_job_must_be_absent=True)
    if (
        type(author_custody) is not OriginalConfirmatoryStagedEPublicationCustody
        or author_custody.role != "author"
        or author_custody.path != canonical_path
        or author_custody.final_supervisor_job_directory != final_job
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "staged E verifier lacks exact overlapping author custody"
        )
    expected_identity = _mapping(
        expected_publication_identity,
        role="staged E expected publication identity",
    )
    if not _strict_json_value_equal(
        expected_identity, author_custody.publication_identity
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "staged E verifier expected identity differs from author custody"
        )
    e_from_author = canonical_original_confirmatory_e_intent(
        decode_canonical_json_line(
            _read_all_from_descriptor(
                author_custody.descriptor,
                maximum_bytes=MAX_CONTROL_FILE_BYTES,
            ),
            role="staged E author retained file",
        ),
        q_authority=canonical_q,
        expected_q_file_sha256=expected_q_file_sha256,
    )
    staged_e, final_job, staging_paths, final_parent_paths = (
        _require_staged_e_matches_q_and_payload(
            canonical_e=e_from_author,
            canonical_q=canonical_q,
            staged_e_intent_path=canonical_path,
            supervisor_job_directory=final_job,
        )
    )
    staging_handles: tuple[int, ...] = ()
    staging_windows_native = False
    staging_facts: tuple[tuple[int, str, int], ...] = ()
    final_handles: tuple[int, ...] = ()
    final_windows_native = False
    final_facts: tuple[tuple[int, str, int], ...] = ()
    descriptor = -1
    transition_guard = -1
    try:
        (
            staging_handles,
            staging_windows_native,
            staging_facts,
            observed_staging_paths,
        ) = _open_exact_directory_chain(staging_paths, role="staged E verifier")
        (
            final_handles,
            final_windows_native,
            final_facts,
            observed_final_paths,
        ) = _open_exact_directory_chain(
            final_parent_paths,
            role="staged E verifier final job parent",
        )
        if _path_lexists(final_job):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "final supervisor job appeared before staged E verification"
            )
        transition_guard = _open_control_publication_transition_guard(staged_e)
        guard_payload = _read_all_from_descriptor(
            transition_guard,
            maximum_bytes=MAX_CONTROL_FILE_BYTES,
        )
        guard_identity = _control_publication_identity_from_descriptor(
            transition_guard,
            path=staged_e,
            payload=guard_payload,
            write_handle_retained=False,
        )
        if not _strict_json_value_equal(
            guard_identity,
            {**expected_identity, "write_handle_retained": False},
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "staged E transition guard opened a different physical identity"
            )
        author_custody.close()
        descriptor = _open_read_descriptor(staged_e)
        os.close(transition_guard)
        transition_guard = -1
        payload = _read_all_from_descriptor(
            descriptor, maximum_bytes=MAX_CONTROL_FILE_BYTES
        )
        file_sha256 = hashlib.sha256(payload).hexdigest()
        if file_sha256 != _sha256(
            expected_file_sha256, role="staged E expected SHA-256"
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "staged E file hash differs from author evidence"
            )
        identity = _control_publication_identity_from_descriptor(
            descriptor,
            path=staged_e,
            payload=payload,
            write_handle_retained=False,
        )
        if not _strict_json_value_equal(
            identity,
            {**expected_identity, "write_handle_retained": False},
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "staged E verifier opened a different physical identity"
            )
        canonical = canonical_original_confirmatory_e_intent(
            decode_canonical_json_line(payload, role="staged E intent file"),
            q_authority=canonical_q,
            expected_q_file_sha256=expected_q_file_sha256,
        )
        if not _strict_json_value_equal(canonical, e_from_author):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "staged E independent payload differs from author-held E"
            )
        custody = OriginalConfirmatoryStagedEPublicationCustody(
            path=staged_e,
            final_supervisor_job_directory=final_job,
            descriptor=descriptor,
            staging_ancestor_handles=staging_handles,
            windows_native_staging_ancestor_handles=staging_windows_native,
            staging_ancestor_facts=staging_facts,
            staging_ancestor_paths=observed_staging_paths,
            final_job_parent_handles=final_handles,
            windows_native_final_job_parent_handles=final_windows_native,
            final_job_parent_facts=final_facts,
            final_job_parent_paths=observed_final_paths,
            publication_identity=identity,
            q_custody=author_custody.q_custody,
            role="independent-verifier",
        )
        custody.require_active(final_job_must_be_absent=True)
        descriptor = -1
        staging_handles = ()
        final_handles = ()
        return canonical, file_sha256, custody
    finally:
        if transition_guard >= 0:
            os.close(transition_guard)
        if descriptor >= 0:
            os.close(descriptor)
        for handles, windows_native in (
            (staging_handles, staging_windows_native),
            (final_handles, final_windows_native),
        ):
            for handle in reversed(handles):
                _close_e_job_directory_handle(handle, windows_native=windows_native)


@dataclass(frozen=True, slots=True)
class CapsulePhysicalIdentity:
    """Stable no-follow physical readback of a Q-bound executable file."""

    path: Path
    volume_serial_number: int
    file_id_128: str
    device: int
    inode: int
    size_bytes: int
    mode: int
    link_count: int
    file_attributes: int
    modified_time_ns: int
    changed_time_ns: int
    sha256: str


def _open_read_descriptor(path: Path) -> int:
    if os.name != "nt":
        return os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
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
        0x80000000,
        0x00000001,
        None,
        3,
        0x00000080 | 0x00200000,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(
            cast(int, handle),
            os.O_RDONLY | getattr(os, "O_BINARY", 0),
        )
    except Exception:
        kernel32.CloseHandle(handle)
        raise


def _native_file_identity_from_fd(descriptor: int) -> tuple[int, str]:
    observed = os.fstat(descriptor)
    if os.name != "nt":
        mask = (1 << 64) - 1
        volume = int(observed.st_dev) & mask
        return (
            volume,
            f"{volume:016x}{int(observed.st_ino) & mask:016x}",
        )
    import msvcrt

    class FILE_ID_128(ctypes.Structure):
        _fields_ = [("identifier", ctypes.c_ubyte * 16)]

    class FILE_ID_INFO(ctypes.Structure):
        _fields_ = [
            ("volume_serial_number", ctypes.c_ulonglong),
            ("file_id", FILE_ID_128),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_information = kernel32.GetFileInformationByHandleEx
    get_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    get_information.restype = ctypes.c_int
    information = FILE_ID_INFO()
    if not get_information(
        ctypes.c_void_p(msvcrt.get_osfhandle(descriptor)),
        18,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return (
        int(information.volume_serial_number),
        bytes(information.file_id.identifier).hex(),
    )


def _windows_named_data_streams(path: Path) -> tuple[str, ...]:
    if os.name != "nt":
        return ()

    class WIN32_FIND_STREAM_DATA(ctypes.Structure):
        _fields_ = [
            ("stream_size", ctypes.c_longlong),
            ("stream_name", ctypes.c_wchar * 296),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    find_first = kernel32.FindFirstStreamW
    find_first.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_int,
        ctypes.POINTER(WIN32_FIND_STREAM_DATA),
        ctypes.c_uint32,
    ]
    find_first.restype = ctypes.c_void_p
    find_next = kernel32.FindNextStreamW
    find_next.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(WIN32_FIND_STREAM_DATA),
    ]
    find_next.restype = ctypes.c_int
    find_close = kernel32.FindClose
    find_close.argtypes = [ctypes.c_void_p]
    find_close.restype = ctypes.c_int
    data = WIN32_FIND_STREAM_DATA()
    handle = find_first(str(path), 0, ctypes.byref(data), 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    streams: list[str] = []
    try:
        while True:
            name = data.stream_name
            if name != "::$DATA":
                streams.append(name)
            if not find_next(handle, ctypes.byref(data)):
                error = ctypes.get_last_error()
                if error == 38:
                    break
                raise ctypes.WinError(error)
    finally:
        find_close(handle)
    return tuple(sorted(streams))


def _windows_final_path_from_fd(descriptor: int) -> str | None:
    if os.name != "nt":
        return None
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_final_path = kernel32.GetFinalPathNameByHandleW
    get_final_path.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    get_final_path.restype = ctypes.c_uint32
    handle = ctypes.c_void_p(msvcrt.get_osfhandle(descriptor))
    required = get_final_path(handle, None, 0, 0)
    if required == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_unicode_buffer(required + 1)
    written = get_final_path(handle, buffer, len(buffer), 0)
    if written == 0 or written >= len(buffer):
        raise ctypes.WinError(ctypes.get_last_error())
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        return "\\\\" + value[8:]
    if value.startswith("\\\\?\\"):
        return value[4:]
    return value


def _read_stable_file(
    path: Path, *, maximum_bytes: int
) -> tuple[CapsulePhysicalIdentity, bytes]:
    path = Path(_absolute_path(str(path), role="stable executable file"))
    before = path.lstat()
    reparse = int(getattr(before, "st_file_attributes", 0)) & 0x400
    if (
        not stat.S_ISREG(before.st_mode)
        or path.is_symlink()
        or reparse
        or int(before.st_nlink) != 1
        or int(before.st_size) <= 0
        or int(before.st_size) > maximum_bytes
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "stable executable file must be one bounded single-link real file"
        )
    descriptor = -1
    try:
        descriptor = _open_read_descriptor(path)
        opened = os.fstat(descriptor)
        final_path = _windows_final_path_from_fd(descriptor)
        volume_serial_number, file_id = _native_file_identity_from_fd(descriptor)
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after_descriptor = os.fstat(descriptor)
        after_path = path.lstat()
        after_volume_serial_number, after_file_id = _native_file_identity_from_fd(
            descriptor
        )
    except OSError as exc:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "stable executable file failed its no-follow readback"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    comparable = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_nlink",
        "st_mtime_ns",
        "st_file_attributes",
    )
    observations = (opened, after_descriptor, after_path)
    if (
        any(
            int(getattr(item, field, 0)) != int(getattr(before, field, 0))
            for item in observations
            for field in comparable
        )
        or any(
            stat.S_IFMT(item.st_mode) != stat.S_IFMT(before.st_mode)
            for item in observations
        )
        or int(before.st_ctime_ns) != int(after_path.st_ctime_ns)
        or int(opened.st_ctime_ns) != int(after_descriptor.st_ctime_ns)
        or volume_serial_number != after_volume_serial_number
        or file_id != after_file_id
        or (
            final_path is not None
            and os.path.normcase(os.path.normpath(final_path))
            != os.path.normcase(os.path.normpath(str(path)))
        )
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "stable executable file changed during no-follow readback"
        )
    payload = b"".join(chunks)
    if len(payload) != int(before.st_size) or len(payload) > maximum_bytes:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "stable executable file exceeded its exact bound"
        )
    return (
        CapsulePhysicalIdentity(
            path=path,
            volume_serial_number=volume_serial_number,
            file_id_128=file_id,
            device=int(before.st_dev),
            inode=int(before.st_ino),
            size_bytes=int(before.st_size),
            mode=int(before.st_mode),
            link_count=int(before.st_nlink),
            file_attributes=int(getattr(before, "st_file_attributes", 0)),
            modified_time_ns=int(before.st_mtime_ns),
            changed_time_ns=int(before.st_ctime_ns),
            sha256=hashlib.sha256(payload).hexdigest(),
        ),
        payload,
    )


def _require_live_ancestor_records(
    records: Sequence[Mapping[str, Any]],
    *,
    role: str,
) -> None:
    handles: list[int] = []
    windows_native: bool | None = None
    try:
        for index, record in enumerate(records, start=1):
            path = Path(
                _absolute_path(
                    cast(str, record["path"]),
                    role=f"{role} ancestor {index}",
                )
            )
            handle, is_native, facts = _open_e_job_directory_handle(path)
            handles.append(handle)
            if windows_native is None:
                windows_native = is_native
            elif windows_native != is_native:
                raise OriginalConfirmatoryCapsuleAuthorityError(
                    f"{role} ancestor handles use inconsistent kinds"
                )
            if facts != (
                record["volume_serial_number"],
                record["file_id_128"],
                record["file_attributes"],
            ):
                raise OriginalConfirmatoryCapsuleAuthorityError(
                    f"{role} ancestor {index} differs from Q"
                )
    finally:
        for handle in reversed(handles):
            _close_e_job_directory_handle(
                handle,
                windows_native=bool(windows_native),
            )


def _read_canonical_control_object(
    path: Path,
    *,
    role: str,
) -> tuple[dict[str, Any], str, bytes]:
    _identity, payload = _read_stable_file(
        path,
        maximum_bytes=MAX_CONTROL_FILE_BYTES,
    )
    value = decode_canonical_json_line(payload, role=role)
    return value, hashlib.sha256(payload).hexdigest(), payload


def _windows_process_creation_time_100ns(handle: int) -> int:
    class FILETIME(ctypes.Structure):
        _fields_ = [
            ("low", ctypes.c_uint32),
            ("high", ctypes.c_uint32),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    getter = kernel32.GetProcessTimes
    getter.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(FILETIME),
        ctypes.POINTER(FILETIME),
        ctypes.POINTER(FILETIME),
        ctypes.POINTER(FILETIME),
    ]
    getter.restype = ctypes.c_int
    created = FILETIME()
    exited = FILETIME()
    kernel = FILETIME()
    user = FILETIME()
    if not getter(
        ctypes.c_void_p(handle),
        ctypes.byref(created),
        ctypes.byref(exited),
        ctypes.byref(kernel),
        ctypes.byref(user),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return (int(created.high) << 32) | int(created.low)


def _windows_filetime_100ns_to_utc(value: int) -> str:
    unix_100ns = value - 116444736000000000
    seconds, remainder = divmod(unix_100ns, 10_000_000)
    instant = datetime.fromtimestamp(seconds + remainder / 10_000_000, tz=UTC)
    return instant.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _windows_process_image_path(handle: int) -> Path:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel32.QueryFullProcessImageNameW
    query.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    query.restype = ctypes.c_int
    size = ctypes.c_uint32(32768)
    buffer = ctypes.create_unicode_buffer(size.value)
    if not query(
        ctypes.c_void_p(handle),
        0,
        buffer,
        ctypes.byref(size),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return Path(
        _absolute_path(
            buffer.value,
            role="Windows process image path",
        )
    )


def _current_process_identity_for_command(
    command: OriginalConfirmatoryCapsuleCommand,
) -> dict[str, Any]:
    if os.name != "nt":
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "original confirmatory capsule execution requires Windows"
        )
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = ctypes.c_void_p
    handle = cast(int, get_current_process())
    image_path = _windows_process_image_path(handle)
    if (
        os.path.normcase(str(image_path)) != os.path.normcase(str(command.program_path))
        or Path(sys.executable) != command.program_path
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "live process image/sys.executable differs from logical Q interpreter"
        )
    creation_time_100ns = _windows_process_creation_time_100ns(handle)
    return {
        "pid": os.getpid(),
        "creation_time_100ns": creation_time_100ns,
        "creation_time_utc": _windows_filetime_100ns_to_utc(creation_time_100ns),
        "program_path": str(command.program_path),
        "program_sha256": command.program_sha256,
        "command_sha256": command.command_sha256,
    }


def _open_exact_supervisor_process(
    identity: Mapping[str, Any],
) -> int:
    expected = _canonical_e_process_identity(
        identity,
        role="live supervisor process identity",
    )
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    opener = kernel32.OpenProcess
    opener.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    opener.restype = ctypes.c_void_p
    handle = opener(
        POSTWAKE_SOURCE_OWNER_OPEN_ACCESS_MASK,
        0,
        expected["pid"],
    )
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    numeric = cast(int, handle)
    try:
        image = _windows_process_image_path(numeric)
        image_identity, _payload = _read_stable_file(
            image,
            maximum_bytes=max(
                MAX_CONTROL_FILE_BYTES,
                int(image.stat().st_size),
            ),
        )
        if (
            _windows_process_creation_time_100ns(numeric)
            != expected["creation_time_100ns"]
            or os.path.normcase(str(image))
            != os.path.normcase(expected["program_path"])
            or image_identity.sha256 != expected["program_sha256"]
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "live supervisor PID was reused or its image differs"
            )
        return numeric
    except BaseException:
        kernel32.CloseHandle(ctypes.c_void_p(numeric))
        raise


def _write_same_held_claim(
    descriptor: int,
    *,
    path: Path,
    claim: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    payload = canonical_json_line_bytes(claim)
    if not payload or len(payload) > MAX_CONTROL_FILE_BYTES:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E consumption claim exceeds its exact bound"
        )
    if os.fstat(descriptor).st_size != 0:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "E consumption claim handle is not its initial empty identity"
        )
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("same-held E consumption claim write made no progress")
        view = view[written:]
    os.fsync(descriptor)
    readback = _read_all_from_descriptor(
        descriptor,
        maximum_bytes=MAX_CONTROL_FILE_BYTES,
    )
    if readback != payload:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "same-held E consumption claim readback differs"
        )
    base = _control_publication_identity_from_descriptor(
        descriptor,
        path=path,
        payload=payload,
        write_handle_retained=True,
    )
    identity = {
        **base,
        "policy": _E_CONSUMPTION_PHYSICAL_IDENTITY_POLICY,
        "role": "e-intent-consumed-claim",
    }
    return payload, identity


def _duplicate_claim_into_supervisor(
    descriptor: int,
    *,
    supervisor_process_handle: int,
) -> int:
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = ctypes.c_void_p
    duplicate = kernel32.DuplicateHandle
    duplicate.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint32,
    ]
    duplicate.restype = ctypes.c_int
    target = ctypes.c_void_p()
    if not duplicate(
        get_current_process(),
        ctypes.c_void_p(msvcrt.get_osfhandle(descriptor)),
        ctypes.c_void_p(supervisor_process_handle),
        ctypes.byref(target),
        E_CONSUMPTION_DUPLICATE_TARGET_ACCESS_MASK,
        0,
        E_CONSUMPTION_DUPLICATE_OPTIONS,
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return cast(int, target.value)


def _emit_e_consumption_ready_and_require_ack(
    *,
    descriptor: int,
    claim_path: Path,
    q: Mapping[str, Any],
    e: Mapping[str, Any],
    e_file_sha256: str,
    supervisor_spec_sha256: str,
    process_started_sha256: str,
    child_process_identity: Mapping[str, Any],
    supervisor_process_identity: Mapping[str, Any],
) -> None:
    contract = canonical_original_confirmatory_e_consumption_contract(
        e["e_consumption_contract"],
        supervisor_job_directory=e["job"]["supervisor_job_dir"],
    )
    supervisor_handle = _open_exact_supervisor_process(supervisor_process_identity)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    try:
        target_handle = _duplicate_claim_into_supervisor(
            descriptor,
            supervisor_process_handle=supervisor_handle,
        )
        claim_unsigned = {
            "schema_version": 1,
            "policy": E_CONSUMPTION_CLAIM_POLICY,
            "disposition": E_CONSUMPTION_CLAIM_DISPOSITION,
            "contract_sha256": contract.contract_sha256,
            "q_authority_root_sha256": q["q_authority_root_sha256"],
            "e_intent_path": q["control_staging_projection"]["e_intent_path"],
            "e_intent_file_sha256": e_file_sha256,
            "e_intent_core_sha256": e["intent_core_sha256"],
            "supervisor_job_id": e["job"]["job_id"],
            "attempt_id": e["job"]["attempt_id"],
            "run_id": e["job"]["run_id"],
            "launch_nonce": e["job"]["launch_nonce"],
            "execution_mode": e["lineage"]["execution_mode"],
            "retry_of_run_id": e["lineage"]["retry_of_run_id"],
            "supervisor_spec_path": e["job"]["supervisor_spec_path"],
            "supervisor_spec_sha256": supervisor_spec_sha256,
            "process_started_path": str(
                Path(e["job"]["supervisor_job_dir"]) / "process_started.json"
            ),
            "process_started_sha256": process_started_sha256,
            "child_process_identity": dict(child_process_identity),
            "supervisor_process_identity": dict(supervisor_process_identity),
            "logical_python_path": q["execution_capsule"]["python_path"],
            "logical_python_sha256": q["execution_capsule"]["python_sha256"],
            "runtime_python_path": q["execution_capsule"]["runtime_python_path"],
            "runtime_python_sha256": q["execution_capsule"]["runtime_python_sha256"],
            "scientific_inputs_read": False,
            "automatic_retry_allowed": False,
        }
        claim = canonical_original_confirmatory_e_consumption_claim(
            {
                **claim_unsigned,
                "claim_root_sha256": canonical_json_sha256(claim_unsigned),
            },
            contract=contract,
            e_intent=e,
            q_authority=q,
        )
        claim_payload, physical = _write_same_held_claim(
            descriptor,
            path=claim_path,
            claim=claim,
        )
        ready_unsigned = {
            "schema_version": 1,
            "policy": E_CONSUMPTION_READY_POLICY,
            "message_type": E_CONSUMPTION_READY_MESSAGE_TYPE,
            "contract_sha256": contract.contract_sha256,
            "claim_path": str(claim_path),
            "claim_file_sha256": hashlib.sha256(claim_payload).hexdigest(),
            "claim_root_sha256": claim["claim_root_sha256"],
            "claim_physical_identity": physical,
            "target_supervisor_handle_value": target_handle,
            "duplicate_target_access_mask": (
                E_CONSUMPTION_DUPLICATE_TARGET_ACCESS_MASK
            ),
            "duplicate_options": E_CONSUMPTION_DUPLICATE_OPTIONS,
            "close_source": False,
            "child_process_identity": dict(child_process_identity),
            "supervisor_process_identity": dict(supervisor_process_identity),
            "supervisor_spec_sha256": supervisor_spec_sha256,
            "process_started_sha256": process_started_sha256,
        }
        ready = canonical_original_confirmatory_e_consumption_ready(
            {
                **ready_unsigned,
                "ready_sha256": canonical_json_sha256(ready_unsigned),
            },
            contract=contract,
            claim=claim,
            claim_physical_identity=physical,
        )
        ready_line = canonical_json_line_bytes(ready)
        if len(ready_line) > contract.ready_line_max_bytes:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "E consumption READY exceeds its exact bound"
            )
        sys.stdout.buffer.write(ready_line)
        sys.stdout.buffer.flush()
        ack_line = sys.stdin.buffer.readline(contract.ack_line_max_bytes + 1)
        if (
            not ack_line
            or len(ack_line) > contract.ack_line_max_bytes
            or not ack_line.endswith(b"\n")
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "E consumption ACK is absent or oversized"
            )
        ack_raw = decode_canonical_json_line(
            ack_line,
            role="E consumption ACK",
        )
        receipt_path = Path(
            _absolute_path(
                ack_raw.get("custody_receipt_path"),
                role="E custody receipt path from ACK",
            )
        )
        receipt, receipt_file_sha256, _receipt_payload = _read_canonical_control_object(
            receipt_path,
            role="E consumption custody receipt",
        )
        canonical_original_confirmatory_e_consumption_ack(
            ack_raw,
            contract=contract,
            ready=ready,
            custody_receipt=receipt,
            custody_receipt_file_sha256=receipt_file_sha256,
        )
        final_payload = _read_all_from_descriptor(
            descriptor,
            maximum_bytes=MAX_CONTROL_FILE_BYTES,
        )
        if final_payload != claim_payload:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "E consumption claim changed before ACK completion"
            )
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(supervisor_handle))


def _prevalidate_original_confirmatory_scientific_tail(
    canonical_tail: tuple[str, ...],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str,
    str,
    str,
    dict[str, Any],
    dict[str, Any],
]:
    command_tail = canonical_original_confirmatory_capsule_mode_tail(
        capsule_mode=CAPSULE_SCIENTIFIC_MODE,
        tail_argv=canonical_tail,
    )
    values = dict(zip(command_tail[::2], command_tail[1::2], strict=True))
    e_path = Path(values["--e-intent"])
    e_raw, e_file_sha256, _e_payload = _read_canonical_control_object(
        e_path,
        role="live E intent",
    )
    q_path = Path(e_raw["q_authority"]["path"])
    q_raw, q_file_sha256, _q_payload = _read_canonical_control_object(
        q_path,
        role="live Q replacement-v2",
    )
    q = canonical_original_confirmatory_q_replacement_v2(q_raw)
    e = canonical_original_confirmatory_e_intent(
        e_raw,
        q_authority=q,
        expected_q_file_sha256=q_file_sha256,
    )
    if (
        e_file_sha256 != values["--e-intent-sha256"]
        or e["intent_core_sha256"] != values["--e-intent-core-sha256"]
        or q["q_authority_root_sha256"] != values["--q-authority-root-sha256"]
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "live Q/E hashes differ from exact scientific argv"
        )
    command = derive_original_confirmatory_capsule_command_from_e(
        e_intent=e,
        e_file_sha256=e_file_sha256,
        q_authority=q,
        capsule_mode=CAPSULE_SCIENTIFIC_MODE,
    )
    if tuple(command.argv[3:]) != tuple(sys.argv) or Path(os.getcwd()) != command.cwd:
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "live scientific process argv/cwd differs from E projection"
        )
    require_live_original_confirmatory_execution_capsule(q["execution_capsule"])
    runtime_path = Path(q["execution_capsule"]["runtime_python_path"])
    if (
        Path(getattr(sys, "_base_executable", "")) != runtime_path
        or not sys.orig_argv
        or Path(sys.orig_argv[0]) != runtime_path
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "live base runtime attestations differ from Q"
        )
    spec_path = Path(e["job"]["supervisor_spec_path"])
    spec, spec_sha256, _spec_payload = _read_canonical_control_object(
        spec_path,
        role="downstream supervisor spec",
    )
    if (
        type(spec.get("schema_version")) is not int
        or spec.get("schema_version") != 3
        or spec.get("job_id") != e["job"]["job_id"]
        or spec.get("project_root") != e["project_root"]
        or spec.get("program_path") != command.as_dict()["program_path"]
        or spec.get("program_sha256") != command.program_sha256
        or not _strict_json_value_equal(spec.get("argv"), list(command.argv))
        or not _strict_json_value_equal(
            spec.get("expected_environment"),
            e["expected_launch_environment"],
        )
        or not _strict_json_value_equal(
            spec.get("process_environment_binding"),
            e["process_environment_binding"],
        )
        or spec.get("supervisor_launcher_sha256")
        != q["supervisor_release"]["supervisor_launcher_sha256"]
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "downstream supervisor spec does not rederive exact Q/E launch"
        )
    process_started_path = spec_path.parent / "process_started.json"
    process_started, process_started_sha256, _started_payload = (
        _read_canonical_control_object(
            process_started_path,
            role="supervisor process-started receipt",
        )
    )
    child_identity = _current_process_identity_for_command(command)
    supervisor_identity = _canonical_e_process_identity(
        process_started.get("supervisor_process_identity"),
        role="process-started supervisor identity",
    )
    if (
        process_started.get("job_id") != e["job"]["job_id"]
        or process_started.get("spec_sha256") != spec_sha256
        or process_started.get("attempt_nonce") != e["job"]["launch_nonce"]
        or not _strict_json_value_equal(
            _canonical_e_process_identity(
                process_started.get("process_identity"),
                role="process-started child identity",
            ),
            child_identity,
        )
        or process_started.get("automatic_retry_allowed") is not False
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "process-started receipt differs from the live child/spec"
        )
    observed_environment = _canonical_environment(
        dict(os.environ),
        role="live scientific environment",
    )
    if not _strict_json_value_equal(
        observed_environment,
        e["expected_launch_environment"]["child_environment"],
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "live scientific environment differs from E"
        )
    return (
        q,
        e,
        e_file_sha256,
        spec_sha256,
        process_started_sha256,
        child_identity,
        supervisor_identity,
    )


def _dispatch_original_confirmatory_run_from_canonical_tail(
    canonical_tail: tuple[str, ...],
) -> int:
    """Full prevalidation -> permanent claim custody ACK -> sole typed lifecycle."""

    descriptor = -1
    try:
        (
            q,
            e,
            e_file_sha256,
            spec_sha256,
            process_started_sha256,
            child_identity,
            supervisor_identity,
        ) = _prevalidate_original_confirmatory_scientific_tail(canonical_tail)
        bootstrap = sys.modules.get("__main__")
        arm = getattr(
            bootstrap,
            "_arm_original_confirmatory_e_claim_after_full_prevalidation",
            None,
        )
        take = getattr(
            bootstrap,
            "_take_original_confirmatory_e_claim_handle",
            None,
        )
        if not callable(arm) or not callable(take):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "sealed bootstrap E claim seams are unavailable"
            )
        arm()
        descriptor, claim_path_raw = take()
        claim_path = Path(
            _absolute_path(
                claim_path_raw,
                role="bootstrap E consumption claim path",
            )
        )
        _emit_e_consumption_ready_and_require_ack(
            descriptor=descriptor,
            claim_path=claim_path,
            q=q,
            e=e,
            e_file_sha256=e_file_sha256,
            supervisor_spec_sha256=spec_sha256,
            process_started_sha256=process_started_sha256,
            child_process_identity=child_identity,
            supervisor_process_identity=supervisor_identity,
        )
        os.close(descriptor)
        descriptor = -1
        from histo_audit.experiment.original_confirmatory_runner_core import (
            _run_original_confirmatory_capsule_request,
            build_original_confirmatory_capsule_request_from_authority,
            prepare_original_confirmatory_capsule_request,
        )

        request = build_original_confirmatory_capsule_request_from_authority(
            q_static_runner_binding=q["scientific_authority"]["static_runner_binding"],
            e_scientific_request_projection=e["scientific_request_projection"],
            e_intent_core_sha256=e["intent_core_sha256"],
        )
        prepared = prepare_original_confirmatory_capsule_request(request)
        if prepared is not request:
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "runner preparation did not retain the exact typed request"
            )
        result = _run_original_confirmatory_capsule_request(request)
        required = {
            "status": "completed",
            "completion_stage": "CONFIRMATORY_COMPLETE",
            "study_outcome_eligible": True,
            "artifact_scope": REAL_CONFIRMATORY_ARTIFACT_SCOPE,
            "run_id": e["job"]["run_id"],
            "run_directory": e["scientific_request_projection"][
                "expected_run_directory"
            ],
            "registry_record_present": True,
            "retry_of_run_id": None,
        }
        if (
            not isinstance(result, dict)
            or any(result.get(key) != value for key, value in required.items())
            or any(
                not isinstance(result.get(field), str)
                or _SHA256.fullmatch(cast(str, result[field])) is None
                for field in (
                    "artifact_root_sha256",
                    "post_seal_attestation_record_sha256",
                    "post_seal_verification_sha256",
                    "confirmatory_storage_policy_sha256",
                    "completion_evidence_sha256",
                )
            )
        ):
            raise OriginalConfirmatoryCapsuleAuthorityError(
                "full confirmatory lifecycle returned a nonqualifying result"
            )
        return 0
    except BaseException:
        return 70
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def require_live_original_confirmatory_execution_capsule(
    capsule: OriginalConfirmatoryExecutionCapsule | Mapping[str, Any],
) -> tuple[
    CapsulePhysicalIdentity,
    CapsulePhysicalIdentity,
    CapsulePhysicalIdentity,
]:
    """Re-read the exact capsule and Python executable through no-follow handles."""

    canonical = canonical_original_confirmatory_execution_capsule(capsule)
    capsule_identity, _capsule_bytes = _read_stable_file(
        canonical.path,
        maximum_bytes=max(MAX_CONTROL_FILE_BYTES, canonical.size_bytes),
    )
    python_stat = canonical.python_path.lstat()
    python_identity, _python_bytes = _read_stable_file(
        canonical.python_path,
        maximum_bytes=max(MAX_CONTROL_FILE_BYTES, int(python_stat.st_size)),
    )
    runtime_python_stat = canonical.runtime_python_path.lstat()
    runtime_python_identity, _runtime_python_bytes = _read_stable_file(
        canonical.runtime_python_path,
        maximum_bytes=max(
            MAX_CONTROL_FILE_BYTES,
            int(runtime_python_stat.st_size),
        ),
    )
    _require_live_ancestor_records(
        canonical.capsule_ancestor_lease.records,
        role="capsule",
    )
    _require_live_ancestor_records(
        canonical.python_ancestor_lease.records,
        role="logical interpreter",
    )
    _require_live_ancestor_records(
        canonical.runtime_python_ancestor_lease.records,
        role="runtime interpreter",
    )
    if (
        capsule_identity.size_bytes != canonical.size_bytes
        or capsule_identity.sha256 != canonical.sha256
        or capsule_identity.volume_serial_number
        != canonical.capsule_lease_identity.volume_serial_number
        or capsule_identity.file_id_128 != canonical.capsule_lease_identity.file_id_128
        or capsule_identity.file_attributes
        != canonical.capsule_lease_identity.file_attributes
        or not capsule_identity.file_attributes & 0x1
        or _windows_named_data_streams(canonical.path)
        or python_identity.sha256 != canonical.python_sha256
        or python_identity.size_bytes != canonical.python_lease_identity.size_bytes
        or python_identity.volume_serial_number
        != canonical.python_lease_identity.volume_serial_number
        or python_identity.file_id_128 != canonical.python_lease_identity.file_id_128
        or python_identity.file_attributes
        != canonical.python_lease_identity.file_attributes
        or _windows_named_data_streams(canonical.python_path)
        or runtime_python_identity.sha256 != canonical.runtime_python_sha256
        or runtime_python_identity.size_bytes
        != canonical.runtime_python_lease_identity.size_bytes
        or runtime_python_identity.volume_serial_number
        != canonical.runtime_python_lease_identity.volume_serial_number
        or runtime_python_identity.file_id_128
        != canonical.runtime_python_lease_identity.file_id_128
        or runtime_python_identity.file_attributes
        != canonical.runtime_python_lease_identity.file_attributes
        or _windows_named_data_streams(canonical.runtime_python_path)
    ):
        raise OriginalConfirmatoryCapsuleAuthorityError(
            "live capsule/logical/runtime interpreter differs from Q"
        )
    return capsule_identity, python_identity, runtime_python_identity


__all__ = [
    "CAPSULE_ALLOWED_MODES",
    "CAPSULE_ANCESTOR_LEASE_POLICY",
    "CAPSULE_PRETERMINAL_MODE",
    "CAPSULE_PYTHON_ISOLATED_FLAGS",
    "CAPSULE_RETAINED_FILE_LEASE_POLICY",
    "CAPSULE_SCIENTIFIC_MODE",
    "CAPSULE_TERMINAL_MODE",
    "CODEX_HANDOFF_ATTEMPT_CREATION_SCHEMA",
    "CODEX_HANDOFF_ATTEMPT_SCHEMA",
    "CODEX_HANDOFF_BASE_OPERATIONAL_SCHEMA",
    "CODEX_HANDOFF_BASE_SYNTHETIC_SCHEMA",
    "CODEX_HANDOFF_MAX_SESSION_FILE_BYTES",
    "CODEX_TERMINAL_WAKE_PROMPT_PLACEHOLDER_ORDER",
    "CODEX_TERMINAL_WAKE_PROMPT_RENDER_POLICY",
    "CONTROL_STAGING_EXACT_FILE_ALLOWLIST",
    "CONTROL_STAGING_PROJECTION_POLICY",
    "CUSTODY_EXCHANGE_TIMEOUT_MS",
    "DIRECT_BASE_RUNTIME_LIVE_PARITY_POLICY",
    "EXECUTION_CAPSULE_POLICY",
    "EXPECTED_LAUNCH_ENVIRONMENT_ENVELOPE_POLICY",
    "EXTERNAL_CODEX_HANDOFF_POLICY",
    "EXTERNAL_CODEX_HANDOFF_SINGLE_WAKE_OWNER",
    "E_CONSUMPTION_ACK_MESSAGE_TYPE",
    "E_CONSUMPTION_CLAIM_POLICY",
    "E_CONSUMPTION_CONTRACT_POLICY",
    "E_CONSUMPTION_READY_MESSAGE_TYPE",
    "E_INTENT_POLICY",
    "INTERNAL_CODEX_WAKE_DISPOSITION",
    "OUTCOME_BLIND_EXPECTED_ARTIFACT_INSTANCE_POLICY",
    "OUTCOME_BLIND_EXPECTED_ARTIFACT_PROJECTION_POLICY",
    "PRETERMINAL_PIN_CONTRACT_POLICY",
    "PROCESS_ENVIRONMENT_BINDING_POLICY",
    "PUBLISHED_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_POLICY",
    "Q_ATTEMPT_IDENTITY_DERIVATION_POLICY",
    "Q_ATTEMPT_LAUNCH_NONCE_DERIVATION_POLICY",
    "Q_E_CUSTODY_ACK_MESSAGE_TYPE",
    "Q_E_CUSTODY_ACK_POLICY",
    "Q_E_CUSTODY_CONTRACT_POLICY",
    "Q_E_CUSTODY_HANDOFF_POLICY",
    "Q_E_CUSTODY_READY_MESSAGE_TYPE",
    "Q_E_CUSTODY_RECEIPT_FILENAME",
    "Q_E_CUSTODY_TRANSPORT",
    "SUPERVISOR_ATTEMPT_NONCE_KEY",
    "SUPERVISOR_PROCESS_COMMAND_DERIVATION_POLICY",
    "SUPERVISOR_PROCESS_COMMAND_POLICY",
    "SUPERVISOR_PROCESS_COMMAND_PROJECTION_POLICY",
    "SUPERVISOR_PYTHON_ISOLATED_FLAGS",
    "SUPERVISOR_STAGED_E_INTENT_FLAG",
    "SUPERVISOR_STAGED_LAUNCH_SPEC_FLAG",
    "TERMINAL_CLIENT_ARRIVAL_TIMEOUT_MS",
    "TERMINAL_CLIENT_CHILD_STDIO_POLICY",
    "TERMINAL_CLIENT_LAUNCHER_E_PROJECTION_POLICY",
    "TERMINAL_CLIENT_LAUNCHER_POLICY",
    "TERMINAL_CLIENT_LAUNCHER_SOURCE_FILENAME",
    "TERMINAL_CLIENT_LAUNCH_INTENT_FILENAME",
    "TERMINAL_CLIENT_LAUNCH_INTENT_POLICY",
    "TERMINAL_CUSTODY_AUTHORITY_PROJECTION_POLICY",
    "TERMINAL_CUSTODY_AUTHORITY_TEMPLATE_POLICY",
    "CapsulePhysicalIdentity",
    "ExpectedLaunchEnvironmentEnvelopeV1",
    "OriginalConfirmatoryCapsuleAncestorLease",
    "OriginalConfirmatoryCapsuleAuthorityError",
    "OriginalConfirmatoryCapsuleCommand",
    "OriginalConfirmatoryCapsuleCommandProjection",
    "OriginalConfirmatoryCapsuleLeaseIdentity",
    "OriginalConfirmatoryEConsumptionContract",
    "OriginalConfirmatoryEPublicationCustody",
    "OriginalConfirmatoryExecutionCapsule",
    "OriginalConfirmatoryPreterminalPinContract",
    "OriginalConfirmatoryProcessEnvironmentBinding",
    "OriginalConfirmatoryQESuspendedSupervisorTarget",
    "OriginalConfirmatoryQPublicationCustody",
    "OriginalConfirmatoryStagedEPublicationCustody",
    "OriginalConfirmatoryTerminalCompositionContract",
    "build_expected_launch_environment_envelope_v1",
    "build_original_confirmatory_capsule_ancestor_lease",
    "build_original_confirmatory_capsule_command",
    "build_original_confirmatory_capsule_command_projection",
    "build_original_confirmatory_capsule_lease_identity",
    "build_original_confirmatory_cli_input_binding",
    "build_original_confirmatory_codex_terminal_wake_prompt_template_projection",
    "build_original_confirmatory_control_staging_projection",
    "build_original_confirmatory_e_consumption_contract",
    "build_original_confirmatory_e_intent",
    "build_original_confirmatory_execution_capsule",
    "build_original_confirmatory_external_supervisor_spec_payload_v3",
    "build_original_confirmatory_outcome_blind_expected_artifact_instance",
    "build_original_confirmatory_outcome_blind_expected_artifact_projection",
    "build_original_confirmatory_preterminal_pin_contract",
    "build_original_confirmatory_process_environment_binding",
    "build_original_confirmatory_q_attempt_identity_projection",
    "build_original_confirmatory_q_e_custody_ack",
    "build_original_confirmatory_q_e_custody_contract",
    "build_original_confirmatory_q_e_custody_ready",
    "build_original_confirmatory_q_e_custody_receipt",
    "build_original_confirmatory_q_e_custody_spec_fields",
    "build_original_confirmatory_q_replacement_v2",
    "build_original_confirmatory_scientific_authority_projection",
    "build_original_confirmatory_scientific_request_projection",
    "build_original_confirmatory_static_runner_binding",
    "build_original_confirmatory_supervisor_process_command_derivation_contract",
    "build_original_confirmatory_supervisor_process_command_projection",
    "build_original_confirmatory_supervisor_release_binding",
    "build_original_confirmatory_terminal_client_launch_intent",
    "build_original_confirmatory_terminal_client_launcher_ancestor_lease",
    "build_original_confirmatory_terminal_client_launcher_command",
    "build_original_confirmatory_terminal_client_launcher_projection",
    "build_original_confirmatory_terminal_client_launcher_release",
    "build_original_confirmatory_terminal_composition_contract",
    "build_original_confirmatory_terminal_custody_authority_projection",
    "build_original_confirmatory_terminal_custody_authority_template_projection",
    "canonical_expected_launch_environment_envelope_v1",
    "canonical_json_bytes",
    "canonical_json_line_bytes",
    "canonical_json_line_sha256",
    "canonical_json_sha256",
    "canonical_original_confirmatory_capsule_ancestor_lease",
    "canonical_original_confirmatory_capsule_command",
    "canonical_original_confirmatory_capsule_command_projection",
    "canonical_original_confirmatory_capsule_lease_identity",
    "canonical_original_confirmatory_capsule_mode_tail",
    "canonical_original_confirmatory_cli_input_binding",
    "canonical_original_confirmatory_codex_handoff_attempt_authority",
    "canonical_original_confirmatory_codex_handoff_attempt_creation_authority",
    "canonical_original_confirmatory_codex_handoff_base_authority",
    "canonical_original_confirmatory_codex_terminal_wake_prompt_template_projection",
    "canonical_original_confirmatory_control_staging_projection",
    "canonical_original_confirmatory_e_consumption_ack",
    "canonical_original_confirmatory_e_consumption_claim",
    "canonical_original_confirmatory_e_consumption_contract",
    "canonical_original_confirmatory_e_consumption_custody_receipt",
    "canonical_original_confirmatory_e_consumption_physical_identity",
    "canonical_original_confirmatory_e_consumption_ready",
    "canonical_original_confirmatory_e_intent",
    "canonical_original_confirmatory_execution_capsule",
    "canonical_original_confirmatory_outcome_blind_expected_artifact_instance",
    "canonical_original_confirmatory_outcome_blind_expected_artifact_projection",
    "canonical_original_confirmatory_preterminal_pin_contract",
    "canonical_original_confirmatory_process_environment_binding",
    "canonical_original_confirmatory_q_attempt_identity_projection",
    "canonical_original_confirmatory_q_e_custody_ack",
    "canonical_original_confirmatory_q_e_custody_contract",
    "canonical_original_confirmatory_q_e_custody_ready",
    "canonical_original_confirmatory_q_e_custody_receipt",
    "canonical_original_confirmatory_q_e_custody_spec_fields",
    "canonical_original_confirmatory_q_replacement_v2",
    "canonical_original_confirmatory_scientific_authority_projection",
    "canonical_original_confirmatory_scientific_request_projection",
    "canonical_original_confirmatory_static_runner_binding",
    "canonical_original_confirmatory_supervisor_process_command_derivation_contract",
    "canonical_original_confirmatory_supervisor_process_command_projection",
    "canonical_original_confirmatory_supervisor_release_binding",
    "canonical_original_confirmatory_terminal_client_launch_intent",
    "canonical_original_confirmatory_terminal_client_launcher_ancestor_lease",
    "canonical_original_confirmatory_terminal_client_launcher_command",
    "canonical_original_confirmatory_terminal_client_launcher_projection",
    "canonical_original_confirmatory_terminal_client_launcher_release",
    "canonical_original_confirmatory_terminal_composition_contract",
    "canonical_original_confirmatory_terminal_custody_authority_projection",
    "canonical_original_confirmatory_terminal_custody_authority_projection_self_contained",
    "canonical_original_confirmatory_terminal_custody_authority_template_projection",
    "canonical_published_original_confirmatory_technical_authority_lifecycle_binding",
    "derive_original_confirmatory_capsule_command_from_e",
    "duplicate_original_confirmatory_q_e_custody_to_supervisor",
    "duplicate_original_confirmatory_staged_q_e_custody_to_supervisor",
    "observe_original_confirmatory_control_publication_ancestor_lease",
    "orchestrate_original_confirmatory_q_e_custody_once",
    "orchestrate_original_confirmatory_staged_q_e_custody_once",
    "original_confirmatory_capsule_mode_tail",
    "original_confirmatory_q_replacement_v2_path",
    "publish_original_confirmatory_e_intent_once",
    "publish_original_confirmatory_q_replacement_v2_once",
    "publish_original_confirmatory_staged_e_intent_once",
    "render_original_confirmatory_codex_terminal_wake_prompt",
    "require_live_original_confirmatory_execution_capsule",
    "require_original_confirmatory_e_intent",
    "require_original_confirmatory_q_replacement_v2",
    "require_original_confirmatory_staged_e_intent",
]
