"""Minimal sealed-capsule bootstrap.

This module verifies the deterministic archive and a minimal held Q/E launch
anchor before importing any ``histo_audit`` module. The sealed authority called
by the dispatcher then performs the full environment, supervisor-custody,
scientific-input, and mode-specific validation before any scientific action.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import json
import os
import re
import stat
import struct
import sys
import zipfile
import zipimport
import zlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Final, NamedTuple, NoReturn, cast

MANIFEST_NAME: Final = "AANCA_CAPSULE_MANIFEST.json"
MANIFEST_POLICY: Final = "original_confirmatory_execution_capsule_manifest_v1"
CAPSULE_FILENAME: Final = "original_confirmatory.pyz"
CAPSULE_POLICY_MEMBER: Final = "aanca_capsule/capsule_policy.json"
ENTRY_CONTRACT_MEMBER: Final = "aanca_capsule/entry_contract.json"
DISPATCHER_MEMBER: Final = "histo_audit/workflows/original_confirmatory_capsule_entry.py"
AUTHORITY_MEMBER: Final = "histo_audit/workflows/original_confirmatory_capsule_authority.py"
ADMITTED_AUTHORITY_SHA256: Final = (
    "b0b68f745e6e9e6ae7e8e83278fb2863959d7601dd24b50c51741d20b10be1ff"
)
DISPATCHER_PATH: Final = (
    "histo_audit.workflows.original_confirmatory_capsule_entry:"
    "_dispatch_original_confirmatory_capsule"
)
ALLOWED_MODES: Final = (
    "run-confirmatory",
    "verify-preterminal",
    "verify-terminal",
)
SCIENTIFIC_MODE: Final = "run-confirmatory"
PRETERMINAL_MODE: Final = "verify-preterminal"
TERMINAL_MODE: Final = "verify-terminal"
_COMMON_TAIL_FLAGS: Final = (
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
_SUCCESSOR_LINEAGE_FLAG: Final = "--retry-of-run-id"
_PRETERMINAL_SUFFIX_FLAGS: Final = (
    "--run-spec",
    "--launch-intent",
    "--process-started",
    "--preterminal-pin",
)
_TERMINAL_SUFFIX_FLAGS: Final = (
    "--supervisor-terminal",
    "--verifier-stdout",
    "--preterminal-pin",
    "--composed-terminal",
)
FIXED_ZIP_DATETIME: Final = (1980, 1, 1, 0, 0, 0)
FIXED_EXTERNAL_ATTR: Final = (stat.S_IFREG | 0o444) << 16
_MEMBER_PATH = re.compile(r"^[a-z0-9_][a-z0-9_.-]*(/[a-z0-9_][a-z0-9_.-]*)*$")
_ROLE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FILE_ID_128 = re.compile(r"^[0-9a-f]{32}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
_Q_ATTEMPT_ID = re.compile(r"^ocq-[0-9a-f]{32}$")
_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
_CENTRAL_HEADER = struct.Struct("<IHHHHHHIIIHHHHHII")
_END_OF_CENTRAL_DIRECTORY = struct.Struct("<IHHHHIIH")
_MAX_CONTROL_BYTES: Final = 16 * 1024 * 1024
_MANIFEST_FIELDS: Final = {
    "schema_version",
    "policy",
    "archive_policy",
    "entries",
    "entry_count",
    "payload_size_bytes",
    "records_root_sha256",
}
_ENTRY_FIELDS: Final = {"relative_path", "role", "size_bytes", "sha256"}
_Q_FIELDS: Final = {
    "schema_version",
    "policy",
    "authority_disposition",
    "q_path",
    "project_root",
    "scientific_authority",
    "publication_ancestor_lease",
    "publication_ancestor_lease_root_sha256",
    "execution_capsule",
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
_Q_BASE_AUTHORITY_FIELDS: Final = (
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
_Q_ATTEMPT_IDENTITY_FIELDS: Final = {
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
_CONTROL_STAGING_PROJECTION_FIELDS: Final = {
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
_E_FIELDS: Final = {
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
    "e_consumption_contract",
    "scientific_request_projection",
    "lineage",
    "expected_launch_environment",
    "process_environment_binding",
    "command_projections",
    "attempt_count",
    "max_attempt_count",
    "automatic_retry_allowed",
    "scientific_outcomes_read",
    "intent_core_sha256",
}
_EXPECTED_ENVIRONMENT_FIELDS: Final = {
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
_PROCESS_ENVIRONMENT_BINDING_FIELDS: Final = {
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
_SCIENTIFIC_AUTHORITY_FIELDS: Final = {
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
_STATIC_RUNNER_BINDING_FIELDS: Final = {
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
_PUBLISHED_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_FIELDS: Final = {
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
_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_FIELDS: Final = {
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
_EXECUTION_CAPSULE_FIELDS: Final = {
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
    "python_lease_identity",
    "python_lease_identity_root_sha256",
    "python_ancestor_lease",
    "python_ancestor_lease_root_sha256",
    "contract_sha256",
}
_COMMAND_DERIVATION_FIELDS: Final = {
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
_SUPERVISOR_RELEASE_FIELDS: Final = {
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
_PROTECTED_EXPECTED_ARTIFACT_RULE_FIELDS: Final = {
    "role",
    "path",
    "expected_sha256",
    "must_be_absent_before",
    "json_equals",
}
_PROTECTED_EXPECTED_ARTIFACT_ROLES: Final = (
    "terminal_seal",
    "integrity_receipt",
    "completion_evidence",
    "integrity_registry",
    "stage_attestation_registry",
    "stage_attestation_anchor",
    "disposition_anchor",
)
_OUTCOME_BLIND_EXPECTED_ARTIFACT_PROJECTION_POLICY: Final = (
    "original_confirmatory_outcome_blind_expected_artifact_projection_v1"
)
_OUTCOME_BLIND_EXPECTED_ARTIFACT_INSTANCE_POLICY: Final = (
    "original_confirmatory_outcome_blind_expected_artifact_instance_v1"
)
_TERMINAL_CUSTODY_AUTHORITY_TEMPLATE_POLICY: Final = (
    "original_confirmatory_terminal_custody_authority_template_v1"
)
_TERMINAL_CUSTODY_AUTHORITY_TEMPLATE_ROOT_SHA256: Final = (
    "43b23fc71c17a52630de6a3b4f4e876e9805ffc5c9805c6eb67ffcf026a39b21"
)
_TERMINAL_CUSTODY_AUTHORITY_PROJECTION_POLICY: Final = (
    "original_confirmatory_terminal_custody_authority_projection_v1"
)
_TERMINAL_COMPOSITION_CONTRACT_POLICY: Final = (
    "original_confirmatory_capsule_terminal_composition_contract_v1"
)
_POSTWAKE_CUSTODY_HANDSHAKE_CONTRACT_POLICY: Final = (
    "original_confirmatory_postwake_custody_handshake_contract_v1"
)
_SEMANTIC_OUTCOME_READ_SCOPE: Final = (
    "integrity_and_completion_evidence_only_no_scientific_outcome_values_v1"
)
_TERMINAL_CUSTODY_AUTHORITY_PROJECTION_FIELDS: Final = {
    "schema_version",
    "policy",
    "terminal_custody_authority_template_root_sha256",
    "outcome_blind_expected_artifact_instance",
    "terminal_client_launcher_projection",
    "terminal_client_launcher_projection_root_sha256",
    "projection_root_sha256",
}
_NO_FOLLOW_PHYSICAL_FILE_IDENTITY_POLICY: Final = "aanca_no_follow_physical_file_identity_v1"
_TERMINAL_CLIENT_LAUNCHER_POLICY: Final = "original_confirmatory_terminal_client_launcher_v1"
_TERMINAL_CLIENT_LAUNCHER_ANCESTOR_LEASE_POLICY: Final = (
    "original_confirmatory_terminal_client_launcher_ancestor_lease_v1"
)
_TERMINAL_CLIENT_LAUNCHER_E_PROJECTION_POLICY: Final = (
    "original_confirmatory_terminal_client_launcher_e_projection_v1"
)
_TERMINAL_CLIENT_LAUNCHER_COMMAND_POLICY: Final = (
    "original_confirmatory_terminal_client_launcher_command_v1"
)
_TERMINAL_CLIENT_LAUNCH_INTENT_POLICY: Final = (
    "original_confirmatory_terminal_client_launch_intent_create_new_v1"
)
_TERMINAL_CLIENT_LAUNCHER_SOURCE_FILENAME: Final = "terminal_client_launcher_v1.py"
_TERMINAL_CLIENT_LAUNCH_INTENT_FILENAME: Final = "terminal_client_launch_intent.json"
_TERMINAL_CLIENT_PYTHON_ISOLATED_FLAGS: Final = ("-I", "-S", "-B")
_DIRECT_BASE_RUNTIME_LIVE_PARITY_POLICY: Final = (
    "createprocess_bound_base_runtime_direct_image_and_argv_exact_v1"
)
_TERMINAL_CLIENT_LAUNCH_INTENT_FIELDS: Final = {
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
_TERMINAL_CLIENT_LAUNCH_INTENT_STATUS: Final = "reserved_before_verify_terminal_createprocess"
_TERMINAL_CLIENT_CHILD_STDIO_POLICY: Final = "fresh_restricted_three_handle_anonymous_pipe_stdio_v1"
_TERMINAL_CUSTODY_OUTBOUND_MAX_BYTES: Final = 64 * 1024
_COMPOSED_TERMINAL_STDOUT_SUMMARY_POLICY: Final = (
    "original_confirmatory_composed_terminal_summary_v1"
)
_TERMINAL_CLIENT_ARRIVAL_TIMEOUT_MS: Final = 1_800_000
_CUSTODY_EXCHANGE_TIMEOUT_MS: Final = 60_000
_TERMINAL_CUSTODY_OVERALL_TIMEOUT_MAX_MS: Final = 6 * 60 * 60 * 1_000
_TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDER_ORDER: Final = (
    "supervisor_spec_sha256",
    "e_intent_file_sha256",
    "terminal_receipt_sha256",
    "verify_terminal_command_sha256",
)
_TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDERS: Final = {
    "supervisor_spec_sha256": "$SUPERVISOR_SPEC_SHA256",
    "e_intent_file_sha256": "$E_INTENT_FILE_SHA256",
    "terminal_receipt_sha256": "$TERMINAL_RECEIPT_SHA256",
    "verify_terminal_command_sha256": "$VERIFY_TERMINAL_COMMAND_SHA256",
}
_TERMINAL_CLIENT_LAUNCHER_FINAL_ARGUMENT_ORDER: Final = (
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
_PHYSICAL_FILE_IDENTITY_FIELDS: Final = {
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
_TERMINAL_CLIENT_LAUNCHER_ANCESTOR_LEASE_FIELDS: Final = {
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
_TERMINAL_CLIENT_LAUNCHER_ANCESTOR_DISPOSITION: Final = (
    "supervisor_root_handle_opened_before_q_verification_retained_through_"
    "terminal_child_waitforexit_v1"
)
_TERMINAL_CLIENT_LAUNCHER_RELEASE_FIELDS: Final = {
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
    "launch_intent_physical_identity_policy",
    "launch_intent_physical_identity_role",
    "launch_intent_physical_identity_field_names",
    "launch_intent_schema_version",
    "launch_intent_field_names",
    "launch_intent_status",
    "launch_intent_create_disposition",
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
_TERMINAL_CLIENT_LAUNCHER_E_PROJECTION_FIELDS: Final = {
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
    "launch_intent_physical_identity_policy",
    "launch_intent_physical_identity_role",
    "launch_intent_physical_identity_field_names",
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
_TERMINAL_CLIENT_LAUNCHER_STATIC_CONTROL_VALUES: Final[dict[str, Any]] = {
    "launch_intent_schema_version": 1,
    "launch_intent_field_names": sorted(_TERMINAL_CLIENT_LAUNCH_INTENT_FIELDS),
    "launch_intent_status": _TERMINAL_CLIENT_LAUNCH_INTENT_STATUS,
    "launch_intent_create_disposition": "CREATE_NEW",
    "launch_intent_created_before_child_process_required": True,
    "existing_or_partial_launch_intent_is_stop": True,
    "launch_intent_write_through_fsync_readonly_required": True,
    "launch_intent_physical_identity_policy": _NO_FOLLOW_PHYSICAL_FILE_IDENTITY_POLICY,
    "launch_intent_physical_identity_role": "terminal-client-launch-intent",
    "launch_intent_physical_identity_field_names": sorted(_PHYSICAL_FILE_IDENTITY_FIELDS),
    "launch_intent_creator_desired_access_mask": 0xC0000000,
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
    "launch_intent_supervisor_granted_access_mask": 0x00120089,
    "launch_intent_child_duplicate_target_access_mask": 0x80000000,
    "launch_intent_child_expected_granted_access_mask": 0x00120089,
    "launch_intent_child_duplicate_options": 0,
    "launch_intent_child_duplicate_close_source": False,
    "child_stdio_policy": _TERMINAL_CLIENT_CHILD_STDIO_POLICY,
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
    "child_stdout_max_bytes": _TERMINAL_CUSTODY_OUTBOUND_MAX_BYTES,
    "child_stderr_max_bytes": _TERMINAL_CUSTODY_OUTBOUND_MAX_BYTES,
    "child_stdout_summary_policy": _COMPOSED_TERMINAL_STDOUT_SUMMARY_POLICY,
    "child_stdout_single_canonical_json_line_required": True,
    "child_stderr_empty_required": True,
    "stdio_pipe_drains_event_driven_concurrent_required": True,
    "launcher_forwards_validated_child_stdout_once_required": True,
    "supervisor_retains_launcher_redirector_child_process_handles_through_final_ack_required": True,
    "immediate_venv_redirector_live_through_runtime_child_exit_required": True,
    "terminal_client_launcher_live_through_redirector_waitforexit_required": True,
    "process_liveness_reverified_at_final_ack_required": True,
}
_TERMINAL_CLIENT_LAUNCHER_E_STATIC_COPY_FIELDS: Final = tuple(
    field
    for field in _TERMINAL_CLIENT_LAUNCHER_STATIC_CONTROL_VALUES
    if field in _TERMINAL_CLIENT_LAUNCHER_E_PROJECTION_FIELDS
)
_TERMINAL_CLIENT_LAUNCHER_COMMAND_PREIMAGE_FIELDS: Final = {
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
_SUPERVISOR_PROCESS_COMMAND_DERIVATION_POLICY: Final = (
    "original_confirmatory_supervisor_process_command_derivation_v2"
)
_SUPERVISOR_PROCESS_COMMAND_POLICY: Final = "original_confirmatory_supervisor_process_command_v2"
_SUPERVISOR_PROCESS_COMMAND_HASH_POLICY: Final = "canonical_compact_sorted_json_sha256_no_lf_v1"
_SUPERVISOR_PYTHON_ISOLATED_FLAGS: Final = ("-I", "-S", "-B")
_SUPERVISOR_STAGED_LAUNCH_SPEC_FLAG: Final = "--staged-launch-spec"
_SUPERVISOR_STAGED_E_INTENT_FLAG: Final = "--staged-e-intent"
_SUPERVISOR_PROCESS_COMMAND_PREIMAGE_FIELDS: Final = {
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
_SUPERVISOR_PROCESS_COMMAND_DERIVATION_FIELDS: Final = {
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
_PROTECTED_EXPECTED_ARTIFACT_TEMPLATE_RULE_FIELDS: Final = {
    "role",
    "path_anchor",
    "relative_path",
    "expected_sha256",
    "must_be_absent_before",
    "json_equals",
}
_PROTECTED_EXPECTED_ARTIFACT_TEMPLATE: Final[tuple[dict[str, Any], ...]] = (
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
_E_JOB_FIELDS: Final = {
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
_SUPERVISOR_RUN_SPEC_ENVELOPE_FIELDS: Final = {
    "schema_version",
    "payload",
    "payload_sha256",
}
_SUPERVISOR_RUN_SPEC_PAYLOAD_FIELDS: Final = {
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
_SUPERVISOR_RUN_SPEC_IDENTITY_FIELDS: Final = {
    "path",
    "sha256",
}
_SUPERVISOR_CANONICAL_SPEC_FIELDS: Final = {
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
    "policy",
}
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
_CONTROL_STAGING_OUTER_FILE_FIELDS: Final = {
    "role",
    "name",
    "path",
    "size_bytes",
    "file_sha256",
    "physical_identity",
    "physical_identity_root_sha256",
}
_Q_E_CUSTODY_CONTRACT_FIELDS: Final = {
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
_Q_E_CUSTODY_READY_FIELDS: Final = {
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
_Q_E_CUSTODY_SPEC_RECEIPT_BINDING_FIELDS: Final = {
    "policy",
    "path",
    "file_sha256",
    "receipt_root_sha256",
    "handoff_root_sha256",
}
_Q_E_CUSTODY_SPEC_FIELDS: Final = {
    "q_e_custody_contract",
    "q_e_custody_handoff",
    "q_e_custody_receipt",
}
_Q_E_CUSTODY_E_ANCESTOR_LEASE_FIELDS: Final = {
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
_CONTROL_STAGING_ANCESTOR_LEASE_FIELDS: Final = {
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
    "continuous_overlap_through_supervisor_stage_ack_required",
    "acquisition_disposition",
}
_Q_E_CUSTODY_LEAF_RETAINED_BINDING_FIELDS: Final = {
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
_Q_E_CUSTODY_ANCESTOR_RETAINED_BINDING_FIELDS: Final = {
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
_E_PROCESS_IDENTITY_FIELDS: Final = {
    "pid",
    "creation_time_100ns",
    "creation_time_utc",
    "program_path",
    "program_sha256",
    "command_sha256",
}
_CONTROL_PUBLICATION_IDENTITY_POLICY: Final = (
    "original_confirmatory_control_publication_physical_identity_v1"
)
_CONTROL_PUBLICATION_IDENTITY_FIELDS: Final = {
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
_CAPSULE_COMMAND_FIELDS: Final = {
    "program_path",
    "program_sha256",
    "argv",
    "cwd",
    "command_sha256",
}
_TERMINAL_COMPOSITION_CONTRACT_FIELDS: Final = {
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
_E_CONSUMPTION_FIELDS: Final = {
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
_SCIENTIFIC_REQUEST_PROJECTION_FIELDS: Final = {
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
_E_LINEAGE_FIELDS: Final = {
    "schema_version",
    "policy",
    "execution_mode",
    "retry_of_run_id",
}
_Q_AUTHORITY_FIELDS: Final = {
    "path",
    "file_sha256",
    "root_sha256",
}
_COMMAND_PROJECTION_FIELDS: Final = {
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
_CAPSULE_LEASE_FIELDS: Final = {
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
_ANCESTOR_LEASE_FIELDS: Final = {
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
_ANCESTOR_RECORD_FIELDS: Final = {
    "path",
    "volume_serial_number",
    "file_id_128",
    "file_attributes",
    "reparse_point",
}
_PYTHON_LEASE_FIELDS: Final = {
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
_PYTHON_ANCESTOR_LEASE_FIELDS: Final = {
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
_PUBLICATION_ANCESTOR_LEASE_FIELDS: Final = {
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
_EMITTED_ROLES: Final = frozenset(
    {
        "capsule_authority",
        "capsule_bootstrap",
        "capsule_contract",
        "capsule_dispatcher",
        "capsule_policy",
        "capsule_terminal",
        "package_initializer",
        "project_source",
        "scientific_completion",
        "scientific_entry",
    }
)
_CAPSULE_POLICY: Final = {
    "allowed_modes": list(ALLOWED_MODES),
    "automatic_retry_allowed": False,
    "generic_histo_audit_cli_allowed": False,
    "policy": "original_confirmatory_sealed_execution_capsule_v1",
    "schema_version": 1,
    "source_policy": "every_regular_python_file_under_final_histo_audit_tree_v1",
}
_ENTRY_CONTRACT: Final = {
    "allowed_modes": list(ALLOWED_MODES),
    "contract_status": "ready",
    "dispatcher": DISPATCHER_PATH,
    "policy": "original_confirmatory_execution_capsule_entry_contract_v1",
    "schema_version": 1,
}
_REQUIRED_MEMBER_ROLES: Final = {
    "__main__.py": "capsule_bootstrap",
    CAPSULE_POLICY_MEMBER: "capsule_policy",
    ENTRY_CONTRACT_MEMBER: "capsule_contract",
    "histo_audit/experiment/confirmatory_completion.py": "scientific_completion",
    "histo_audit/experiment/original_confirmatory_runner_core.py": "scientific_entry",
    "histo_audit/workflows/original_confirmatory_capsule_authority.py": ("capsule_authority"),
    DISPATCHER_MEMBER: "capsule_dispatcher",
    "histo_audit/workflows/original_confirmatory_capsule_terminal.py": ("capsule_terminal"),
}
_NON_PROJECT_PAYLOAD_MEMBERS: Final = frozenset(
    {
        "__main__.py",
        CAPSULE_POLICY_MEMBER,
        ENTRY_CONTRACT_MEMBER,
    }
)
_WINDOWS_RESERVED: Final = {
    "aux",
    "clock$",
    "com1",
    "com2",
    "com3",
    "com4",
    "com5",
    "com6",
    "com7",
    "com8",
    "com9",
    "con",
    "lpt1",
    "lpt2",
    "lpt3",
    "lpt4",
    "lpt5",
    "lpt6",
    "lpt7",
    "lpt8",
    "lpt9",
    "nul",
    "prn",
}
_ARCHIVE_POLICY: Final = {
    "format": "zip",
    "compression": "stored",
    "payload_entry_order": "ordinal_then_manifest_last",
    "fixed_dos_datetime": "1980-01-01T00:00:00",
    "create_system": 3,
    "unix_mode": "100444",
    "zip64": False,
    "archive_comment_empty": True,
    "directory_entries": False,
    "manifest_self_entry": False,
}
_Q_POLICY: Final = "original_confirmatory_q_replacement_v2"
_Q_DISPOSITION: Final = "one_create_new_q_publication_for_exact_bound_inputs_v1"
_Q_ATTEMPT_IDENTITY_DERIVATION_POLICY: Final = (
    "original_confirmatory_q_attempt_identity_derivation_v1"
)
_Q_ATTEMPT_LAUNCH_NONCE_DERIVATION_POLICY: Final = (
    "original_confirmatory_q_attempt_launch_nonce_derivation_v1"
)
_CONTROL_STAGING_PROJECTION_POLICY: Final = "original_confirmatory_control_staging_v2"
_CONTROL_STAGING_DIRECTORY_NAME: Final = "control_staging"
_SUPERVISOR_JOBS_DIRECTORY_NAME: Final = "jobs"
_CONTROL_STAGING_EXACT_FILE_ALLOWLIST: Final = (
    "staging_attempt.json",
    "e_intent.json",
    "launch_authorization.json",
    "supervisor_launch_spec.json",
    "staging_ready.json",
)
_CONTROL_STAGING_OUTER_FILE_ROLES: Final = (
    "staging-attempt",
    "e-intent",
    "launch-authorization",
    "supervisor-launch-spec",
    "staging-ready",
)
_E_POLICY: Final = "original_confirmatory_e_intent_v1"
_E_DISPOSITION: Final = "one_create_new_e_intent_for_exact_supervisor_job_v1"
_SCIENTIFIC_AUTHORITY_POLICY: Final = "original_confirmatory_scientific_authority_projection_v1"
_STATIC_RUNNER_BINDING_POLICY: Final = "original_confirmatory_static_runner_binding_v3"
_PUBLISHED_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_POLICY: Final = (
    "published_original_confirmatory_technical_authority_lifecycle_binding_v1"
)
_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_POLICY: Final = (
    "original_confirmatory_technical_authority_lifecycle_binding_v1"
)
_SCIENTIFIC_REQUEST_PROJECTION_POLICY: Final = "original_confirmatory_capsule_request_projection_v1"
_REAL_CONFIRMATORY_ARTIFACT_SCOPE: Final = "real_pannuke_confirmatory_study"
_SCIENTIFIC_CONTROL_READ_SCOPE: Final = "integrity/control_only_no_scientific_outcomes"
_CAPSULE_CONTRACT_POLICY: Final = "original_confirmatory_sealed_execution_capsule_v1"
_COMMAND_DERIVATION_POLICY: Final = "original_confirmatory_capsule_command_derivation_contract_v1"
_COMMAND_PROJECTION_POLICY: Final = "original_confirmatory_capsule_command_projection_v1"
_SUPERVISOR_RELEASE_POLICY: Final = "original_confirmatory_supervisor_release_binding_v1"
_SUPERVISOR_POLICY: Final = "aanca_event_driven_unattended_supervisor_external_handoff_v3"
_EXTERNAL_CODEX_HANDOFF_POLICY: Final = "aanca_external_current_session_two_branch_handoff_v1"
_EXTERNAL_CODEX_HANDOFF_SINGLE_WAKE_OWNER: Final = "operational_current_session_successor"
_INTERNAL_CODEX_WAKE_DISPOSITION: Final = "PROHIBITED_EXTERNAL_SINGLE_WAKE_OWNER"
_CODEX_HANDOFF_BASE_OPERATIONAL_SCHEMA: Final = "aanca.codex-handoff-base-authority.operational.v1"
_CODEX_HANDOFF_BASE_OPERATIONAL_SCOPE: Final = "aanca_unattended_codex_handoff_base_operational_v1"
_CODEX_HANDOFF_ATTEMPT_CREATION_SCHEMA: Final = (
    "aanca.codex-handoff-attempt-creation-authority.operational.v1"
)
_CODEX_HANDOFF_ATTEMPT_CREATION_SCOPE: Final = (
    "aanca_unattended_codex_handoff_attempt_creation_operational_v1"
)
_CODEX_HANDOFF_ATTEMPT_SCHEMA: Final = "aanca.codex-handoff-attempt-authority.operational.v1"
_CODEX_HANDOFF_ARM_ALGORITHM: Final = (
    "retained_identity_ReadDirectoryChangesW_before_snapshot_prefix_sha256_v1"
)
_CODEX_HANDOFF_BOUNDARY_MAX_AGE_MS: Final = 3_600_000
_CODEX_HANDOFF_FIXED_POLICY_ROOTS: Final = {
    "resume_command_policy": ("90422b1dc85f9f1d6e3de92777761e573af449699fab3d370a6de7117073ae90"),
    "limits": "a76ac3ca5b23b5c20b190302d648f2d776f92cbb22b81dc3372abf1e7d56e659",
    "branch_template_policy": ("60c45745e16d27389ef22cf00a8d260f34b05e30084d02275751e3baba82b5bd"),
    "idle_completion_policy": ("eba7ec4bd1602166388e8b1724323d372a82fcc870329b3cd5ab4753e2e16804"),
    "external_supervisor_handoff_policy": (
        "0ffeee124c70de53ea37c15687f2c29966d70fff4427dc587b259214bcf8a97f"
    ),
}
_CODEX_HANDOFF_ARM_ALGORITHM_CONTRACT_ROOT_SHA256: Final = (
    "056c90a59d0b5921d4aef625786419ab5397818904dcfb1314dd9a403c3562ad"
)
_CONTROL_STAGING_OUTER_BINDING_POLICY: Final = "aanca_supervisor_control_staging_binding_v1"
_CONTROL_STAGING_ANCESTOR_LEASE_POLICY: Final = (
    "original_confirmatory_control_staging_ancestor_lease_v1"
)
_CONTROL_STAGING_ANCESTOR_ACQUISITION_DISPOSITION: Final = (
    "opened_after_create_new_stage_dir_before_first_leaf_retained_through_stage_ack_v1"
)
_EXPECTED_LAUNCH_ENVIRONMENT_POLICY: Final = (
    "expected_original_confirmatory_launch_environment_envelope_v1"
)
_PROCESS_ENVIRONMENT_BINDING_POLICY: Final = "aanca_three_process_environment_binding_v1"
_SUPERVISOR_ATTEMPT_NONCE_KEY: Final = "AANCA_SUPERVISOR_ATTEMPT_NONCE"
_SCIENCE_SUPERVISOR_ENVIRONMENT_NAMES: Final = {
    "LOCALAPPDATA",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
}
_EXTERNAL_CONTROL_PLANE_RELEASE_DIRECTORY_NAME: Final = "AANCA-control-plane-release-v2"
_EXTERNAL_CONTROL_PLANE_VERIFICATIONS_DIRECTORY_NAME: Final = "verifications"
_EXTERNAL_CONTROL_PLANE_QUALIFICATION_ATTESTATION_FILENAME: Final = (
    "release_qualification_attestation.json"
)
_POSTWAKE_CUSTODY_SEED_POLICY: Final = "original_confirmatory_postwake_custody_seed_v1"
_POSTWAKE_CUSTODY_PIPE_DERIVATION_POLICY: Final = "postwake_custody_seed_sha256_direct_suffix_v1"
_Q_E_CUSTODY_CONTRACT_POLICY: Final = "original_confirmatory_q_e_supervisor_custody_contract_v1"
_Q_E_CUSTODY_HANDOFF_POLICY: Final = "original_confirmatory_q_e_supervisor_custody_handoff_v1"
_Q_E_CUSTODY_TRANSPORT: Final = "bounded_anonymous_pipe_blocking_v1"
_Q_E_CUSTODY_READY_MESSAGE_TYPE: Final = "Q_E_CUSTODY_READY"
_Q_E_CUSTODY_ACK_POLICY: Final = "original_confirmatory_q_e_supervisor_custody_ack_v1"
_Q_E_CUSTODY_ACK_MESSAGE_TYPE: Final = "Q_E_CUSTODY_ACK"
_Q_E_CUSTODY_RECEIPT_POLICY: Final = "original_confirmatory_q_e_supervisor_custody_receipt_v1"
_Q_E_CUSTODY_RECEIPT_FILENAME: Final = "q_e_custody_receipt.json"
_Q_E_CUSTODY_SUPERVISOR_RETENTION_POLICY: Final = (
    "through_science_preterminal_terminal_postwake_and_terminal_seal_v1"
)
_Q_E_CUSTODY_LINE_MAX_BYTES: Final = 64 * 1024
_Q_E_CUSTODY_LEAF_TARGET_ACCESS_MASK: Final = 0x80000000
_Q_E_CUSTODY_ANCESTOR_TARGET_ACCESS_MASK: Final = 0x80000080
_Q_E_CUSTODY_MAPPED_FILE_GRANTED_ACCESS_MASK: Final = 0x00120089
_Q_E_CUSTODY_DUPLICATE_OPTIONS: Final = 0
_Q_E_CUSTODY_E_ANCESTOR_LEASE_POLICY: Final = (
    "original_confirmatory_e_job_publication_ancestor_lease_v1"
)
_Q_E_CUSTODY_E_ANCESTOR_DISPOSITION: Final = (
    "opened_before_e_create_new_retained_through_verifier_and_supervisor_overlap_v1"
)
_Q_E_CUSTODY_LEAF_RETAINED_BINDING_POLICY: Final = (
    "original_confirmatory_q_e_leaf_retained_binding_v1"
)
_Q_E_CUSTODY_ANCESTOR_RETAINED_BINDING_POLICY: Final = (
    "original_confirmatory_q_e_ancestor_retained_binding_v1"
)
_Q_E_CUSTODY_RECEIPT_STATUS: Final = "retained_verified_before_scientific_inputs_v1"
_Q_REPLACEMENT_V2_FILENAME: Final = "original_confirmatory_q_replacement_v2.json"
_E_INTENT_FILENAME: Final = "e_intent.json"
_E_JOB_POLICY: Final = "original_confirmatory_supervisor_job_binding_v1"
_E_CONSUMPTION_POLICY: Final = "original_confirmatory_e_intent_consumed_supervisor_custody_v1"
_E_CONSUMPTION_CLAIM_POLICY: Final = "original_confirmatory_e_intent_consumed_claim_v1"
_E_CONSUMPTION_TRANSPORT: Final = "bounded_anonymous_pipe_blocking_v1"
_E_CONSUMPTION_READY_MESSAGE_TYPE: Final = "E_INTENT_CONSUMED_READY"
_E_CONSUMPTION_ACK_MESSAGE_TYPE: Final = "E_INTENT_CONSUMED_ACK"
_E_CONSUMPTION_SUPERVISOR_RETENTION_POLICY: Final = (
    "through_main_wait_preterminal_terminal_postwake_and_terminal_seal_v1"
)
_E_CONSUMPTION_LINE_MAX_BYTES: Final = 16 * 1024
_E_CONSUMPTION_DUPLICATE_TARGET_ACCESS_MASK: Final = 0x80000000
_E_CONSUMPTION_DUPLICATE_OPTIONS: Final = 0
_E_LINEAGE_POLICY: Final = "original_confirmatory_execution_lineage_v1"
_FILE_HASH_POLICY: Final = "canonical_json_line_sha256_v1"
_CORE_HASH_POLICY: Final = "canonical_json_without_self_field_sha256_v1"
_E_FILE_INSERTION_POLICY: Final = "append_value_to_terminal_e_file_sha256_flag_then_continue_v1"
_E_CORE_INSERTION_POLICY: Final = "append_value_to_terminal_e_core_sha256_flag_then_continue_v1"
_FINAL_COMMAND_CARRIER: Final = "supervisor_job_spec_create_new_read_only_v1"
_CAPSULE_LEASE_POLICY: Final = "original_confirmatory_capsule_retained_file_lease_v1"
_CAPSULE_ANCESTOR_LEASE_POLICY: Final = "original_confirmatory_capsule_ancestor_lease_v1"
_PYTHON_LEASE_POLICY: Final = "original_confirmatory_interpreter_retained_file_lease_v1"
_PYTHON_ANCESTOR_LEASE_POLICY: Final = "original_confirmatory_interpreter_ancestor_lease_v1"
_PYTHON_RUNTIME_RESOLUTION_POLICY: Final = "windows_venv_redirector_native_base_executable_v1"
_RUNTIME_PYTHON_LEASE_POLICY: Final = (
    "original_confirmatory_runtime_interpreter_retained_file_lease_v1"
)
_RUNTIME_PYTHON_ANCESTOR_LEASE_POLICY: Final = (
    "original_confirmatory_runtime_interpreter_ancestor_lease_v1"
)
_PUBLICATION_ANCESTOR_LEASE_POLICY: Final = (
    "original_confirmatory_control_publication_ancestor_lease_v1"
)
_LEAF_ACQUISITION_DISPOSITION: Final = (
    "opened_before_first_phase_createprocess_retained_through_all_phase_waitforexit_v1"
)
_ANCESTOR_ACQUISITION_DISPOSITION: Final = "directory_handles_opened_before_first_phase_createprocess_retained_through_all_phase_waitforexit_v1"
_PUBLICATION_ANCESTOR_ACQUISITION_DISPOSITION: Final = (
    "opened_before_q_create_new_retained_through_verifier_and_supervisor_overlap_v1"
)


class CapsuleBootstrapError(RuntimeError):
    pass


class _ArchiveEvidence(NamedTuple):
    entries: tuple[dict[str, Any], ...]
    internal_manifest_sha256: str
    capsule_policy_sha256: str
    entry_contract_sha256: str


class _HeldAuthorityFile(NamedTuple):
    descriptor: int
    path: Path
    payload: bytes
    sha256: str
    size_bytes: int
    identity: tuple[int, ...]


class _PreimportAuthorityAnchor(NamedTuple):
    q_file: _HeldAuthorityFile
    e_file: _HeldAuthorityFile
    supervisor_spec_file: _HeldAuthorityFile
    terminal_client_launcher_file: _HeldAuthorityFile
    python_file: _HeldAuthorityFile
    runtime_python_file: _HeldAuthorityFile
    staging_attempt_file: _HeldAuthorityFile
    launch_authorization_file: _HeldAuthorityFile
    supervisor_launch_spec_source_file: _HeldAuthorityFile
    staging_ready_file: _HeldAuthorityFile


def _stop(message: str) -> NoReturn:
    raise CapsuleBootstrapError(message)


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            _stop(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _canonical_json_line(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _authority_canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _authority_canonical_json_line(value: Any) -> bytes:
    return _authority_canonical_json(value) + b"\n"


def _authority_json_sha256(value: Any) -> str:
    return hashlib.sha256(_authority_canonical_json(value)).hexdigest()


def _ascii_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _parse_authority_json_line(payload: bytes, *, label: str) -> dict[str, Any]:
    if (
        not payload.endswith(b"\n")
        or payload.endswith(b"\n\n")
        or payload.startswith(b"\xef\xbb\xbf")
    ):
        _stop(f"{label} is not canonical UTF-8 JSON plus one LF")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda token: _stop(f"non-finite JSON token: {token}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CapsuleBootstrapError(f"{label} JSON is invalid") from exc
    if not isinstance(value, dict) or _authority_canonical_json_line(value) != payload:
        _stop(f"{label} bytes are not authority-canonical")
    return value


def _parse_canonical_json_line(payload: bytes, *, label: str) -> dict[str, Any]:
    if (
        not payload.endswith(b"\n")
        or payload.endswith(b"\n\n")
        or payload.startswith(b"\xef\xbb\xbf")
    ):
        _stop(f"{label} is not canonical UTF-8 JSON plus one LF")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda token: _stop(f"non-finite JSON token: {token}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CapsuleBootstrapError(f"{label} JSON is invalid") from exc
    if not isinstance(value, dict) or _canonical_json_line(value) != payload:
        _stop(f"{label} bytes are not canonical")
    return value


def _parse_canonical_manifest(payload: bytes) -> dict[str, Any]:
    return _parse_canonical_json_line(payload, label="capsule manifest")


def _safe_member_path(value: Any) -> str:
    if (
        type(value) is not str
        or value.casefold() == MANIFEST_NAME.casefold()
        or _MEMBER_PATH.fullmatch(value) is None
        or str(PurePosixPath(value)) != value
        or "\\" in value
        or ":" in value
        or "\x00" in value
    ):
        _stop("capsule manifest contains an unsafe member path")
    for segment in value.split("/"):
        stem = segment.split(".", 1)[0].casefold()
        if (
            segment in {".", ".."}
            or segment.endswith((".", " "))
            or stem in _WINDOWS_RESERVED
            or any(ord(character) < 32 or ord(character) > 126 for character in segment)
        ):
            _stop("capsule manifest contains an unsafe path segment")
    return value


def _records_root(entries: Sequence[Mapping[str, Any]]) -> str:
    preimage = b"".join(
        (
            f"{entry['relative_path']}\0{entry['role']}\0{entry['size_bytes']}\0{entry['sha256']}\n"
        ).encode("ascii")
        for entry in entries
    )
    return hashlib.sha256(preimage).hexdigest()


def _verify_manifest_structure(value: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if (
        set(value) != _MANIFEST_FIELDS
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["policy"] != MANIFEST_POLICY
        or _canonical_json_line(value["archive_policy"]) != _canonical_json_line(_ARCHIVE_POLICY)
        or not isinstance(value["entries"], list)
        or type(value["entry_count"]) is not int
        or type(value["payload_size_bytes"]) is not int
        or type(value["records_root_sha256"]) is not str
    ):
        _stop("capsule manifest violates its closed schema")
    entries: list[dict[str, Any]] = []
    exact: set[str] = set()
    folded: set[str] = set()
    for raw in value["entries"]:
        if not isinstance(raw, dict) or set(raw) != _ENTRY_FIELDS:
            _stop("capsule manifest entry violates its closed schema")
        relative_path = _safe_member_path(raw["relative_path"])
        role = raw["role"]
        size_bytes = raw["size_bytes"]
        sha256 = raw["sha256"]
        if (
            type(role) is not str
            or _ROLE.fullmatch(role) is None
            or role not in _EMITTED_ROLES
            or type(size_bytes) is not int
            or size_bytes < 0
            or type(sha256) is not str
            or _SHA256.fullmatch(sha256) is None
            or relative_path in exact
            or relative_path.casefold() in folded
        ):
            _stop("capsule manifest entry values are invalid or aliased")
        exact.add(relative_path)
        folded.add(relative_path.casefold())
        entries.append(dict(raw))
    paths = [entry["relative_path"] for entry in entries]
    if (
        not _strict_json_value_equal(paths, sorted(paths))
        or value["entry_count"] != len(entries)
        or value["payload_size_bytes"] != sum(entry["size_bytes"] for entry in entries)
        or value["records_root_sha256"] != _records_root(entries)
    ):
        _stop("capsule manifest aggregate bindings are invalid")
    return tuple(entries)


def _project_module_inventory(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, bool]]:
    modules: dict[str, tuple[str, bool]] = {}
    for entry in entries:
        relative_path = entry["relative_path"]
        if relative_path in _NON_PROJECT_PAYLOAD_MEMBERS:
            continue
        if not relative_path.startswith("histo_audit/") or not relative_path.endswith(".py"):
            _stop(f"capsule contains a non-policy payload member: {relative_path}")
        parts = relative_path.removesuffix(".py").split("/")
        is_package = parts[-1] == "__init__"
        module_parts = parts[:-1] if is_package else parts
        if not module_parts or module_parts[0] != "histo_audit":
            _stop(f"capsule project module path is invalid: {relative_path}")
        module_name = ".".join(module_parts)
        if module_name in modules:
            _stop(f"capsule module inventory is ambiguous: {module_name}")
        modules[module_name] = (relative_path, is_package)
    root = modules.get("histo_audit")
    if root != ("histo_audit/__init__.py", True):
        _stop("capsule lacks one regular histo_audit root package")
    for module_name in modules:
        segments = module_name.split(".")
        for length in range(1, len(segments)):
            parent_name = ".".join(segments[:length])
            parent = modules.get(parent_name)
            if parent is None or parent[1] is not True:
                _stop(f"capsule module lacks a regular sealed parent package: {module_name}")
    return modules


def _verify_control_members(payloads: Mapping[str, bytes]) -> None:
    policy_payload = payloads.get(CAPSULE_POLICY_MEMBER)
    contract_payload = payloads.get(ENTRY_CONTRACT_MEMBER)
    authority_payload = payloads.get(AUTHORITY_MEMBER)
    if policy_payload is None or contract_payload is None or authority_payload is None:
        _stop("capsule control member is absent")
    _parse_canonical_json_line(policy_payload, label="capsule policy")
    _parse_canonical_json_line(contract_payload, label="capsule entry contract")
    if policy_payload != _canonical_json_line(_CAPSULE_POLICY):
        _stop("capsule policy violates its exact closed contract")
    if contract_payload != _canonical_json_line(_ENTRY_CONTRACT):
        _stop("capsule entry contract is not exact and ready")
    if hashlib.sha256(authority_payload).hexdigest() != ADMITTED_AUTHORITY_SHA256:
        _stop("capsule authority differs from the independently admitted authority")


def _unpack_exact(
    parser: struct.Struct,
    raw: bytes,
    offset: int,
    *,
    label: str,
) -> tuple[Any, ...]:
    if offset < 0 or offset + parser.size > len(raw):
        _stop(f"capsule {label} is truncated")
    return parser.unpack_from(raw, offset)


def _verify_raw_zip_layout(
    raw: bytes,
    names: Sequence[str],
    payloads: Sequence[bytes],
) -> None:
    if len(names) != len(payloads):
        _stop("capsule raw ZIP verifier received inconsistent records")
    local_offsets: list[int] = []
    offset = 0
    for name, payload in zip(names, payloads, strict=True):
        local_offsets.append(offset)
        encoded_name = name.encode("ascii")
        crc32 = zlib.crc32(payload) & 0xFFFFFFFF
        fields = _unpack_exact(_LOCAL_HEADER, raw, offset, label="local header")
        name_length = fields[9]
        extra_length = fields[10]
        if fields != (
            0x04034B50,
            20,
            0,
            zipfile.ZIP_STORED,
            0,
            33,
            crc32,
            len(payload),
            len(payload),
            len(encoded_name),
            0,
        ):
            _stop(f"capsule local ZIP header is non-canonical: {name}")
        offset += _LOCAL_HEADER.size
        if raw[offset : offset + name_length] != encoded_name:
            _stop(f"capsule local ZIP filename differs from policy: {name}")
        offset += name_length
        if extra_length != 0:
            _stop(f"capsule local ZIP extra field is not empty: {name}")
        if raw[offset : offset + len(payload)] != payload:
            _stop(f"capsule local ZIP payload differs from manifest: {name}")
        offset += len(payload)

    central_offset = offset
    for index, (name, payload) in enumerate(zip(names, payloads, strict=True)):
        encoded_name = name.encode("ascii")
        crc32 = zlib.crc32(payload) & 0xFFFFFFFF
        fields = _unpack_exact(_CENTRAL_HEADER, raw, offset, label="central header")
        name_length = fields[10]
        extra_length = fields[11]
        comment_length = fields[12]
        if fields != (
            0x02014B50,
            (3 << 8) | 20,
            20,
            0,
            zipfile.ZIP_STORED,
            0,
            33,
            crc32,
            len(payload),
            len(payload),
            len(encoded_name),
            0,
            0,
            0,
            0,
            FIXED_EXTERNAL_ATTR,
            local_offsets[index],
        ):
            _stop(f"capsule central ZIP header is non-canonical: {name}")
        offset += _CENTRAL_HEADER.size
        if raw[offset : offset + name_length] != encoded_name:
            _stop(f"capsule central ZIP filename differs from policy: {name}")
        offset += name_length
        if extra_length != 0 or comment_length != 0:
            _stop(f"capsule central ZIP extension is not empty: {name}")

    central_size = offset - central_offset
    fields = _unpack_exact(
        _END_OF_CENTRAL_DIRECTORY,
        raw,
        offset,
        label="end-of-central-directory record",
    )
    if fields != (
        0x06054B50,
        0,
        0,
        len(names),
        len(names),
        central_size,
        central_offset,
        0,
    ):
        _stop("capsule end-of-central-directory record is non-canonical")
    offset += _END_OF_CENTRAL_DIRECTORY.size
    if offset != len(raw):
        _stop("capsule has bytes outside its exact canonical ZIP layout")


def _verify_archive(descriptor: int) -> _ArchiveEvidence:
    duplicate = os.dup(descriptor)
    try:
        with os.fdopen(duplicate, mode="rb", closefd=True) as stream:
            duplicate = -1
            stream.seek(0)
            raw_archive = stream.read()
            stream.seek(0)
            with zipfile.ZipFile(stream, mode="r", allowZip64=False) as archive:
                if archive.comment != b"":
                    _stop("capsule archive comment is not empty")
                infos = archive.infolist()
                if not infos or infos[-1].filename != MANIFEST_NAME:
                    _stop("capsule manifest is not the final ZIP entry")
                if sum(info.filename == MANIFEST_NAME for info in infos) != 1:
                    _stop("capsule manifest entry is absent or duplicated")
                manifest_payload = archive.read(infos[-1])
                manifest = _parse_canonical_manifest(manifest_payload)
                entries = _verify_manifest_structure(manifest)
                expected_names = [entry["relative_path"] for entry in entries] + [MANIFEST_NAME]
                if [info.filename for info in infos] != expected_names:
                    _stop("capsule ZIP member set/order differs from the manifest")
                manifest_by_path = {entry["relative_path"]: entry for entry in entries}
                for path, role in _REQUIRED_MEMBER_ROLES.items():
                    record = manifest_by_path.get(path)
                    if record is None or record["role"] != role:
                        _stop(f"required capsule member/role is absent: {path}")
                control_payloads: dict[str, bytes] = {}
                verified_payloads: list[bytes] = []
                for index, info in enumerate(infos):
                    expected_size = (
                        len(manifest_payload)
                        if info.filename == MANIFEST_NAME
                        else entries[index]["size_bytes"]
                    )
                    expected_sha = (
                        hashlib.sha256(manifest_payload).hexdigest()
                        if info.filename == MANIFEST_NAME
                        else entries[index]["sha256"]
                    )
                    payload = archive.read(info)
                    if (
                        info.date_time != FIXED_ZIP_DATETIME
                        or info.compress_type != zipfile.ZIP_STORED
                        or info.create_system != 3
                        or info.external_attr != FIXED_EXTERNAL_ATTR
                        or info.extra != b""
                        or info.comment != b""
                        or info.file_size != expected_size
                        or info.compress_size != expected_size
                        or len(payload) != expected_size
                        or hashlib.sha256(payload).hexdigest() != expected_sha
                    ):
                        _stop(f"capsule ZIP entry failed exact verification: {info.filename}")
                    if info.filename in {
                        CAPSULE_POLICY_MEMBER,
                        ENTRY_CONTRACT_MEMBER,
                        AUTHORITY_MEMBER,
                    }:
                        control_payloads[info.filename] = payload
                    verified_payloads.append(payload)
                if archive.testzip() is not None:
                    _stop("capsule ZIP CRC validation failed")
                _verify_control_members(control_payloads)
                _verify_raw_zip_layout(
                    raw_archive,
                    expected_names,
                    verified_payloads,
                )
    finally:
        if duplicate >= 0:
            os.close(duplicate)
    return _ArchiveEvidence(
        entries=entries,
        internal_manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        capsule_policy_sha256=hashlib.sha256(control_payloads[CAPSULE_POLICY_MEMBER]).hexdigest(),
        entry_contract_sha256=hashlib.sha256(control_payloads[ENTRY_CONTRACT_MEMBER]).hexdigest(),
    )


def _file_attributes(value: os.stat_result) -> int:
    return int(getattr(value, "st_file_attributes", 0))


def _is_reparse(value: os.stat_result) -> bool:
    return bool(_file_attributes(value) & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)))


def _stable_file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(stat.S_IFMT(value.st_mode)),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_nlink),
        _file_attributes(value),
    )


def _open_capsule_no_follow(path: Path) -> int:
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_CLOEXEC", 0))
    if os.name != "nt":
        return os.open(path, flags | int(getattr(os, "O_NOFOLLOW", 0)))

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
        0x00200000 | 0x08000000,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        raise OSError(ctypes.get_last_error(), f"CreateFileW failed for {path}")
    try:
        return msvcrt.open_osfhandle(int(handle), flags)
    except BaseException:
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int
        close_handle(ctypes.c_void_p(handle))
        raise


def _hash_held_file(descriptor: int) -> tuple[str, int, tuple[int, ...]]:
    first = os.fstat(descriptor)
    if not stat.S_ISREG(first.st_mode) or _is_reparse(first) or int(first.st_nlink) != 1:
        _stop("capsule handle is not a plain single-link regular file")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size_bytes = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        size_bytes += len(chunk)
        digest.update(chunk)
    second = os.fstat(descriptor)
    if _stable_file_identity(first) != _stable_file_identity(second) or size_bytes != int(
        second.st_size
    ):
        _stop("capsule handle changed during whole-file hashing")
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest(), size_bytes, _stable_file_identity(second)


def _native_file_identity_from_fd(descriptor: int) -> tuple[int, str]:
    observed = os.fstat(descriptor)
    if os.name != "nt":
        mask = (1 << 64) - 1
        volume = int(observed.st_dev) & mask
        return volume, f"{volume:016x}{int(observed.st_ino) & mask:016x}"

    import msvcrt

    class _FileId128(ctypes.Structure):
        _fields_ = [("identifier", ctypes.c_ubyte * 16)]

    class _FileIdInfo(ctypes.Structure):
        _fields_ = [
            ("volume_serial_number", ctypes.c_ulonglong),
            ("file_id", _FileId128),
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
    information = _FileIdInfo()
    if not get_information(
        ctypes.c_void_p(msvcrt.get_osfhandle(descriptor)),
        18,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        raise OSError(
            ctypes.get_last_error(),
            "GetFileInformationByHandleEx(FileIdInfo) failed",
        )
    return (
        int(information.volume_serial_number),
        bytes(information.file_id.identifier).hex(),
    )


def _native_path_identity_open_masks(*, directory: bool) -> tuple[int, int, int]:
    desired_access = 0x80000000
    share_mode = 0x00000001 | 0x00000002
    flags = 0x00200000 | (0x02000000 if directory else 0)
    return desired_access, share_mode, flags


def _native_path_identity(path: Path, *, directory: bool) -> tuple[int, str]:
    if os.name != "nt":
        value = os.lstat(path)
        mask = (1 << 64) - 1
        volume = int(value.st_dev) & mask
        return volume, f"{volume:016x}{int(value.st_ino) & mask:016x}"

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
    desired_access, share_mode, flags = _native_path_identity_open_masks(directory=directory)
    handle = create_file(
        str(path),
        desired_access,
        share_mode,
        None,
        3,
        flags,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        raise OSError(
            ctypes.get_last_error(),
            f"CreateFileW identity open failed for {path}",
        )
    try:

        class _FileId128(ctypes.Structure):
            _fields_ = [("identifier", ctypes.c_ubyte * 16)]

        class _FileIdInfo(ctypes.Structure):
            _fields_ = [
                ("volume_serial_number", ctypes.c_ulonglong),
                ("file_id", _FileId128),
            ]

        get_information = kernel32.GetFileInformationByHandleEx
        get_information.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        get_information.restype = ctypes.c_int
        information = _FileIdInfo()
        if not get_information(
            ctypes.c_void_p(handle),
            18,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            raise OSError(
                ctypes.get_last_error(),
                "GetFileInformationByHandleEx(FileIdInfo) failed",
            )
        return (
            int(information.volume_serial_number),
            bytes(information.file_id.identifier).hex(),
        )
    finally:
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int
        close_handle(ctypes.c_void_p(handle))


def _windows_named_data_streams(path: Path) -> tuple[str, ...]:
    if os.name != "nt":
        return ()

    class _StreamData(ctypes.Structure):
        _fields_ = [
            ("stream_size", ctypes.c_longlong),
            ("stream_name", ctypes.c_wchar * 296),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    find_first = kernel32.FindFirstStreamW
    find_first.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_int,
        ctypes.POINTER(_StreamData),
        ctypes.c_uint32,
    ]
    find_first.restype = ctypes.c_void_p
    find_next = kernel32.FindNextStreamW
    find_next.argtypes = [ctypes.c_void_p, ctypes.POINTER(_StreamData)]
    find_next.restype = ctypes.c_int
    find_close = kernel32.FindClose
    find_close.argtypes = [ctypes.c_void_p]
    find_close.restype = ctypes.c_int
    data = _StreamData()
    handle = find_first(str(path), 0, ctypes.byref(data), 0)
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        raise OSError(
            ctypes.get_last_error(),
            f"FindFirstStreamW failed for {path}",
        )
    streams: list[str] = []
    try:
        while True:
            name = str(data.stream_name)
            if name and name.casefold() != "::$data":
                streams.append(name)
            if not find_next(ctypes.c_void_p(handle), ctypes.byref(data)):
                error = ctypes.get_last_error()
                if error == 38:
                    break
                raise OSError(error, f"FindNextStreamW failed for {path}")
    finally:
        find_close(ctypes.c_void_p(handle))
    return tuple(streams)


def _open_held_plain_file(
    path: Path,
    *,
    label: str,
    require_read_only: bool,
    maximum_bytes: int,
) -> _HeldAuthorityFile:
    _exact_absolute_path(str(path), label=label)
    _require_plain_parent_chain(path, label=label)
    descriptor = -1
    try:
        path_value = os.lstat(path)
        if (
            not stat.S_ISREG(path_value.st_mode)
            or stat.S_ISLNK(path_value.st_mode)
            or _is_reparse(path_value)
            or int(path_value.st_nlink) != 1
            or int(path_value.st_size) <= 0
            or int(path_value.st_size) > maximum_bytes
            or (require_read_only and not _readonly_file(path_value))
        ):
            _stop(f"{label} is not one bounded retained plain file")
        try:
            streams = _windows_named_data_streams(path)
        except OSError as exc:
            raise CapsuleBootstrapError(f"{label} alternate-stream enumeration failed") from exc
        if streams:
            _stop(f"{label} has a named alternate data stream")
        descriptor = _open_capsule_no_follow(path)
        digest, size_bytes, identity = _hash_held_file(descriptor)
        if identity != _stable_file_identity(path_value) or size_bytes != int(path_value.st_size):
            _stop(f"{label} path differs from its held identity")
        chunks: list[bytes] = []
        remaining = size_bytes
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        os.lseek(descriptor, 0, os.SEEK_SET)
        payload = b"".join(chunks)
        if (
            len(payload) != size_bytes
            or hashlib.sha256(payload).hexdigest() != digest
            or _stable_file_identity(os.fstat(descriptor)) != identity
            or _stable_file_identity(os.lstat(path)) != identity
        ):
            _stop(f"{label} changed during retained read")
        return _HeldAuthorityFile(
            descriptor=descriptor,
            path=path,
            payload=payload,
            sha256=digest,
            size_bytes=size_bytes,
            identity=identity,
        )
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _open_held_authority_file(path: Path, *, label: str) -> _HeldAuthorityFile:
    return _open_held_plain_file(
        path,
        label=label,
        require_read_only=True,
        maximum_bytes=_MAX_CONTROL_BYTES,
    )


def _require_held_file_unchanged(
    held: _HeldAuthorityFile,
    *,
    label: str,
) -> None:
    digest, size_bytes, identity = _hash_held_file(held.descriptor)
    try:
        path_identity = _stable_file_identity(os.lstat(held.path))
    except OSError as exc:
        raise CapsuleBootstrapError(f"{label} path became unavailable") from exc
    if (
        digest != held.sha256
        or size_bytes != held.size_bytes
        or identity != held.identity
        or path_identity != held.identity
    ):
        _stop(f"{label} changed while retained")


def _close_preimport_anchor(anchor: _PreimportAuthorityAnchor | None) -> None:
    if anchor is None:
        return
    for held in anchor:
        os.close(held.descriptor)


def _require_content_addressed_held_capsule(
    archive_path: Path,
    descriptor: int,
) -> tuple[str, int, tuple[int, ...]]:
    if str(archive_path) != os.path.abspath(os.path.normpath(str(archive_path))):
        _stop("capsule path is not exact absolute lexical canonical form")
    if archive_path.name != CAPSULE_FILENAME:
        _stop("capsule filename is not exact")
    _require_plain_parent_chain(archive_path, label="execution capsule")
    path_value = os.lstat(archive_path)
    parent_value = os.lstat(archive_path.parent)
    if (
        not stat.S_ISREG(path_value.st_mode)
        or stat.S_ISLNK(path_value.st_mode)
        or _is_reparse(path_value)
        or int(path_value.st_nlink) != 1
        or not stat.S_ISDIR(parent_value.st_mode)
        or stat.S_ISLNK(parent_value.st_mode)
        or _is_reparse(parent_value)
    ):
        _stop("capsule path or content-address parent is not plain")
    digest, size_bytes, identity = _hash_held_file(descriptor)
    if (
        archive_path.parent.name != digest
        or _SHA256.fullmatch(archive_path.parent.name) is None
        or _stable_file_identity(path_value) != identity
    ):
        _stop("capsule path does not bind its exact whole-file SHA-256")
    return digest, size_bytes, identity


def _project_root_from_capsule_path(
    archive_path: Path,
    *,
    capsule_sha256: str,
) -> Path:
    try:
        project_root = archive_path.parents[3]
    except IndexError:
        _stop("capsule path is too shallow for the project layout")
    expected = project_root / "artifacts" / "execution_capsules" / capsule_sha256 / CAPSULE_FILENAME
    if archive_path != expected:
        _stop("capsule is outside the exact project execution-capsules layout")
    return project_root


def _exact_object(
    value: Any,
    *,
    fields: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        _stop(f"{label} violates its exact field set")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _stop(f"{label} is not one lowercase SHA-256")
    return value


def _require_self_hash(
    value: Mapping[str, Any],
    *,
    self_field: str,
    label: str,
) -> None:
    observed = _require_sha256(value.get(self_field), label=f"{label} self hash")
    unsigned = {key: item for key, item in value.items() if key != self_field}
    if observed != _authority_json_sha256(unsigned):
        _stop(f"{label} self hash is invalid")


def _contains_mapping_key(value: Any, key: str) -> bool:
    if isinstance(value, Mapping):
        return key in value or any(_contains_mapping_key(item, key) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_mapping_key(item, key) for item in value)
    return False


def _require_published_technical_authority_lifecycle_binding(
    raw_value: Any,
    *,
    project_root: Path,
) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_PUBLISHED_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_FIELDS,
        label="published technical authority lifecycle binding",
    )
    _require_self_hash(
        raw,
        self_field="binding_sha256",
        label="published technical authority lifecycle binding",
    )
    technical = _exact_object(
        raw["technical_authority"],
        fields=_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_FIELDS,
        label="published technical authority nested lifecycle binding",
    )
    _require_self_hash(
        technical,
        self_field="binding_sha256",
        label="published technical authority nested lifecycle binding",
    )
    namespace = _exact_absolute_path(
        raw["namespace_directory"],
        label="published technical authority namespace directory",
    )
    authority_directory = _exact_absolute_path(
        technical["authority_directory"],
        label="published technical authority directory",
    )
    parent_authority_directory = _exact_absolute_path(
        technical["parent_authority_directory"],
        label="published technical authority parent directory",
    )
    _require_sha256(
        raw["namespace_claim_sha256"],
        label="published technical authority namespace claim",
    )
    _require_sha256(
        raw["review_attempt_claim_sha256"],
        label="published technical authority review-attempt claim",
    )
    for field in (
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
    ):
        _require_sha256(
            technical[field],
            label=f"published technical authority {field}",
        )
    if (
        type(technical["schema_version"]) is not int
        or technical["schema_version"] != 1
        or technical["policy"] != _TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_POLICY
        or type(technical["chain_depth"]) is not int
        or technical["chain_depth"] < 1
        or technical["primary_outcomes_inspected"] is not True
        or technical["confirmatory_outcomes_inspected"] is not False
        or technical["confirmatory_outcome_values_read"] is not False
        or technical["scientific_definition_changed"] is not False
        or technical["automatic_retry_allowed"] is not False
        or type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _PUBLISHED_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_POLICY
        or raw["automatic_retry_allowed"] is not False
        or raw["adoption_allowed"] is not False
        or raw["cleanup_allowed"] is not False
        or namespace != project_root / "artifacts" / "original_confirmatory_technical_authorities"
        or authority_directory.parent != namespace
        or project_root not in parent_authority_directory.parents
    ):
        _stop("published technical authority lifecycle binding violates its exact one-use policy")
    return raw


def _require_static_runner_binding(
    raw_value: Any,
    *,
    project_root: Path,
) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_STATIC_RUNNER_BINDING_FIELDS,
        label="Q static runner binding",
    )
    _require_self_hash(
        raw,
        self_field="binding_sha256",
        label="Q static runner binding",
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
        field: _exact_absolute_path(
            raw[field],
            label=f"Q static runner {field}",
        )
        for field in path_fields
    }
    gate = raw["expected_confirmatory_gate"]
    cli_binding = raw["expected_cli_input_binding"]
    published_binding = _require_published_technical_authority_lifecycle_binding(
        raw["published_technical_authority_lifecycle_binding"],
        project_root=project_root,
    )
    published_technical = published_binding["technical_authority"]
    _require_sha256(
        raw["technical_authority_artifact_root_sha256"],
        label="Q technical authority artifact root",
    )
    _require_sha256(
        raw["technical_authorization_sha256"],
        label="Q technical authorization",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 3
        or raw["policy"] != _STATIC_RUNNER_BINDING_POLICY
        or raw["project_root"] != str(project_root)
        or paths["technical_authority_directory"]
        != Path(published_technical["authority_directory"])
        or paths["freeze_directory"] != Path(published_technical["parent_authority_directory"])
        or raw["technical_authority_artifact_root_sha256"]
        != published_technical["artifact_root_sha256"]
        or raw["technical_authorization_sha256"]
        != published_technical["technical_authorization_sha256"]
        or not isinstance(gate, dict)
        or not gate
        or not isinstance(cli_binding, dict)
        or not cli_binding
        or raw["expected_confirmatory_gate_sha256"] != _authority_json_sha256(gate)
        or raw["expected_cli_input_binding_sha256"] != _authority_json_sha256(cli_binding)
        or raw["artifact_scope"] != _REAL_CONFIRMATORY_ARTIFACT_SCOPE
        or raw["semantic_outcome_read_scope"] != _SCIENTIFIC_CONTROL_READ_SCOPE
        or paths["runs_root"].parent != project_root / "artifacts"
        or not all(path == project_root or project_root in path.parents for path in paths.values())
        or _contains_mapping_key(raw, "supervisor_spec_sha256")
    ):
        _stop("Q static runner binding violates its exact outcome-blind policy")
    return raw


def _require_scientific_authority(
    raw_value: Any,
    *,
    project_root: Path,
) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_SCIENTIFIC_AUTHORITY_FIELDS,
        label="Q scientific authority projection",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _SCIENTIFIC_AUTHORITY_POLICY
    ):
        _stop("Q scientific authority projection violates its exact policy")
    hash_fields = _SCIENTIFIC_AUTHORITY_FIELDS - {
        "schema_version",
        "policy",
        "static_runner_binding",
        "static_runner_binding_sha256",
        "scientific_authority_root_sha256",
    }
    for field in hash_fields:
        _require_sha256(raw[field], label=f"Q scientific authority {field}")
    static = _require_static_runner_binding(
        raw["static_runner_binding"],
        project_root=project_root,
    )
    published_technical = static["published_technical_authority_lifecycle_binding"][
        "technical_authority"
    ]
    if (
        raw["static_runner_binding_sha256"] != static["binding_sha256"]
        or raw["technical_authorization_sha256"] != static["technical_authorization_sha256"]
        or raw["historical_primary_authority_artifact_root_sha256"]
        != published_technical["parent_artifact_root_sha256"]
        or raw["technical_execution_source_root_sha256"]
        != published_technical["execution_source_root_sha256"]
        or raw["technical_execution_source_manifest_sha256"]
        != published_technical["execution_source_manifest_sha256"]
        or raw["independent_review_receipt_sha256"]
        != published_technical["independent_review_receipt_sha256"]
        or raw["scientific_authority_root_sha256"]
        != _authority_json_sha256(
            {key: item for key, item in raw.items() if key != "scientific_authority_root_sha256"}
        )
    ):
        _stop("Q scientific authority nested roots differ")
    return raw


def _require_scientific_request_projection(
    raw_value: Any,
    *,
    scientific_authority: Mapping[str, Any],
    e_job: Mapping[str, Any],
    e_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_SCIENTIFIC_REQUEST_PROJECTION_FIELDS,
        label="E scientific request projection",
    )
    _require_self_hash(
        raw,
        self_field="projection_sha256",
        label="E scientific request projection",
    )
    static = _exact_object(
        scientific_authority["static_runner_binding"],
        fields=_STATIC_RUNNER_BINDING_FIELDS,
        label="Q static runner binding",
    )
    checkpoint = raw["checkpoint_authority_projection"]
    runs_root = _exact_absolute_path(
        raw["runs_root"],
        label="E scientific request runs root",
    )
    run_directory = _exact_absolute_path(
        raw["expected_run_directory"],
        label="E expected scientific run directory",
    )
    for field in (
        "plan_sha256",
        "controls_binding_sha256",
        "bridge_binding_sha256",
        "gate_evidence_sha256",
        "cli_input_binding_sha256",
        "checkpoint_authority_projection_sha256",
    ):
        _require_sha256(raw[field], label=f"E scientific request {field}")
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _SCIENTIFIC_REQUEST_PROJECTION_POLICY
        or raw["q_static_runner_binding_sha256"]
        != scientific_authority["static_runner_binding_sha256"]
        or raw["job_id"] != e_job["job_id"]
        or raw["attempt_id"] != e_job["attempt_id"]
        or raw["run_id"] != e_job["run_id"]
        or raw["execution_mode"] != e_lineage["execution_mode"]
        or raw["retry_of_run_id"] != e_lineage["retry_of_run_id"]
        or runs_root != Path(static["runs_root"])
        or run_directory != runs_root / raw["run_id"]
        or raw["gate_evidence_sha256"] != static["expected_confirmatory_gate_sha256"]
        or raw["cli_input_binding_sha256"] != static["expected_cli_input_binding_sha256"]
        or not isinstance(checkpoint, dict)
        or not checkpoint
        or raw["checkpoint_authority_projection_sha256"] != _authority_json_sha256(checkpoint)
        or raw["checkpoint_contract_profile"] != "original_confirmatory_exact_180"
        or type(raw["checkpoint_directive_count"]) is not int
        or raw["checkpoint_directive_count"] != 180
        or raw["artifact_scope"] != _REAL_CONFIRMATORY_ARTIFACT_SCOPE
        or raw["scientific_outcomes_read"] is not False
        or raw["selection_or_tuning_performed"] is not False
        or raw["publication_performed"] is not False
        or raw["automatic_retry_allowed"] is not False
        or _contains_mapping_key(raw, "supervisor_spec_sha256")
    ):
        _stop("E scientific request projection violates its exact typed policy")
    return raw


def _require_exact_string_list(
    value: Any,
    expected: Sequence[str],
    *,
    label: str,
) -> None:
    if (
        not isinstance(value, list)
        or not _strict_json_value_equal(value, list(expected))
        or not all(type(item) is str for item in value)
    ):
        _stop(f"{label} differs from its exact sequence")


def _require_live_ancestor_records(
    raw: Mapping[str, Any],
    *,
    expected_paths: Sequence[Path],
    policy: str,
    interpreter: bool,
) -> None:
    fields = _PYTHON_ANCESTOR_LEASE_FIELDS if interpreter else _ANCESTOR_LEASE_FIELDS
    label = "interpreter ancestor lease" if interpreter else "capsule ancestor lease"
    _exact_object(raw, fields=fields, label=label)
    records = raw["records"]
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != policy
        or type(raw["anchor_path"]) is not str
        or not isinstance(records, list)
        or len(records) != len(expected_paths)
        or not records
        or (interpreter and len(records) > 64)
        or type(raw["record_count"]) is not int
        or raw["record_count"] != len(records)
        or raw["records_root_sha256"] != _authority_json_sha256(records)
        or raw["directory_access_mask"] != 0x80000080
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
        or raw["acquisition_disposition"] != _ANCESTOR_ACQUISITION_DISPOSITION
    ):
        _stop(f"{label} violates its closed aggregate contract")
    anchor = _exact_absolute_path(raw["anchor_path"], label=f"{label} anchor")
    if anchor != expected_paths[0]:
        _stop(f"{label} anchor differs from the expected chain")
    for order, (item, expected_path) in enumerate(
        zip(records, expected_paths, strict=True),
        start=1,
    ):
        record = _exact_object(
            item,
            fields=_ANCESTOR_RECORD_FIELDS,
            label=f"{label} record {order}",
        )
        path = _exact_absolute_path(
            record["path"],
            label=f"{label} record {order} path",
        )
        try:
            value = os.lstat(path)
            volume, file_id = _native_path_identity(path, directory=True)
        except OSError as exc:
            raise CapsuleBootstrapError(
                f"{label} record {order} could not be read without reparse follow"
            ) from exc
        if (
            path != expected_path
            or not stat.S_ISDIR(value.st_mode)
            or stat.S_ISLNK(value.st_mode)
            or _is_reparse(value)
            or type(record["volume_serial_number"]) is not int
            or record["volume_serial_number"] != volume
            or type(record["file_id_128"]) is not str
            or _FILE_ID_128.fullmatch(record["file_id_128"]) is None
            or record["file_id_128"] != file_id
            or type(record["file_attributes"]) is not int
            or record["file_attributes"] != _file_attributes(value)
            or record["reparse_point"] is not False
        ):
            _stop(f"{label} record {order} differs from the live directory")


def _require_plain_parent_chain(path: Path, *, label: str) -> None:
    parents = tuple(reversed(path.parents))
    if not parents:
        _stop(f"{label} has no absolute parent chain")
    previous: Path | None = None
    for order, parent in enumerate(parents, start=1):
        try:
            value = os.lstat(parent)
        except OSError as exc:
            raise CapsuleBootstrapError(f"{label} parent {order} is unavailable") from exc
        if (
            not stat.S_ISDIR(value.st_mode)
            or stat.S_ISLNK(value.st_mode)
            or _is_reparse(value)
            or (previous is not None and parent.parent != previous)
        ):
            _stop(f"{label} parent chain contains a reparse or discontinuity")
        previous = parent


def _require_publication_ancestor_lease(
    raw_value: Any,
    *,
    project_root: Path,
) -> None:
    label = "Q publication ancestor lease"
    raw = _exact_object(
        raw_value,
        fields=_PUBLICATION_ANCESTOR_LEASE_FIELDS,
        label=label,
    )
    records = raw["records"]
    expected_paths = [
        project_root,
        project_root / "artifacts",
        project_root / "artifacts" / "resource_control",
    ]
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _PUBLICATION_ANCESTOR_LEASE_POLICY
        or raw["project_root"] != str(project_root)
        or not isinstance(records, list)
        or len(records) != 3
        or raw["record_count"] != 3
        or type(raw["record_count"]) is not int
        or raw["records_root_sha256"] != _authority_json_sha256(records)
        or raw["directory_access_mask"] != 0x80000080
        or type(raw["directory_access_mask"]) is not int
        or not _strict_json_value_equal(
            raw["share_access"],
            ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        )
        or raw["delete_share"] is not False
        or raw["write_access"] is not False
        or raw["owner_process_identity_required"] is not True
        or raw["handle_slot_per_record_required"] is not True
        or raw["continuous_overlap_through_independent_verification_required"] is not True
        or raw["continuous_overlap_into_supervisor_required"] is not True
        or raw["acquisition_disposition"] != _PUBLICATION_ANCESTOR_ACQUISITION_DISPOSITION
    ):
        _stop(f"{label} violates its exact aggregate contract")
    for order, (item, expected_path) in enumerate(
        zip(records, expected_paths, strict=True),
        start=1,
    ):
        record = _exact_object(
            item,
            fields=_ANCESTOR_RECORD_FIELDS,
            label=f"{label} record {order}",
        )
        path = _exact_absolute_path(
            record["path"],
            label=f"{label} record {order} path",
        )
        try:
            value = os.lstat(path)
            volume, file_id = _native_path_identity(path, directory=True)
        except OSError as exc:
            raise CapsuleBootstrapError(f"{label} record {order} live readback failed") from exc
        if (
            path != expected_path
            or not stat.S_ISDIR(value.st_mode)
            or stat.S_ISLNK(value.st_mode)
            or _is_reparse(value)
            or record["volume_serial_number"] != volume
            or type(record["volume_serial_number"]) is not int
            or record["file_id_128"] != file_id
            or type(record["file_id_128"]) is not str
            or _FILE_ID_128.fullmatch(record["file_id_128"]) is None
            or record["file_attributes"] != _file_attributes(value)
            or type(record["file_attributes"]) is not int
            or record["reparse_point"] is not False
        ):
            _stop(f"{label} record {order} differs from the live directory")


def _require_capsule_leaf_lease(
    raw: Mapping[str, Any],
    *,
    archive_path: Path,
    descriptor: int,
    capsule_sha256: str,
    capsule_size_bytes: int,
) -> None:
    label = "capsule retained-file lease"
    _exact_object(raw, fields=_CAPSULE_LEASE_FIELDS, label=label)
    try:
        value = os.fstat(descriptor)
        volume, file_id = _native_file_identity_from_fd(descriptor)
        streams = _windows_named_data_streams(archive_path)
    except OSError as exc:
        raise CapsuleBootstrapError(f"{label} native readback failed") from exc
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _CAPSULE_LEASE_POLICY
        or raw["path"] != str(archive_path)
        or type(raw["volume_serial_number"]) is not int
        or raw["volume_serial_number"] != volume
        or type(raw["file_id_128"]) is not str
        or _FILE_ID_128.fullmatch(raw["file_id_128"]) is None
        or raw["file_id_128"] != file_id
        or raw["size_bytes"] != capsule_size_bytes
        or type(raw["size_bytes"]) is not int
        or raw["sha256"] != capsule_sha256
        or raw["file_attributes"] != _file_attributes(value)
        or type(raw["file_attributes"]) is not int
        or raw["read_only"] is not True
        or not _readonly_file(value)
        or raw["link_count"] != 1
        or type(raw["link_count"]) is not int
        or not _strict_json_value_equal(raw["named_alternate_data_streams"], [])
        or streams
        or raw["opened_without_reparse_follow"] is not True
        or raw["access_mask"] != 0x80000000
        or type(raw["access_mask"]) is not int
        or not _strict_json_value_equal(raw["share_access"], ["FILE_SHARE_READ"])
        or raw["write_access"] is not False
        or raw["delete_access"] is not False
        or raw["retained_through_each_exact_phase_launch"] is not True
        or raw["owner_process_identity_required"] is not True
        or raw["handle_slot_required"] is not True
        or raw["acquisition_disposition"] != _LEAF_ACQUISITION_DISPOSITION
    ):
        _stop(f"{label} differs from the retained capsule")


def _require_python_leaf_lease(
    raw: Mapping[str, Any],
    *,
    python_file: _HeldAuthorityFile,
    policy: str,
    label: str,
) -> None:
    _exact_object(raw, fields=_PYTHON_LEASE_FIELDS, label=label)
    try:
        value = os.fstat(python_file.descriptor)
        volume, file_id = _native_file_identity_from_fd(python_file.descriptor)
        streams = _windows_named_data_streams(python_file.path)
    except OSError as exc:
        raise CapsuleBootstrapError(f"{label} native readback failed") from exc
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != policy
        or raw["path"] != str(python_file.path)
        or type(raw["volume_serial_number"]) is not int
        or raw["volume_serial_number"] != volume
        or type(raw["file_id_128"]) is not str
        or _FILE_ID_128.fullmatch(raw["file_id_128"]) is None
        or raw["file_id_128"] != file_id
        or raw["size_bytes"] != python_file.size_bytes
        or type(raw["size_bytes"]) is not int
        or raw["sha256"] != python_file.sha256
        or raw["file_attributes"] != _file_attributes(value)
        or type(raw["file_attributes"]) is not int
        or raw["regular_file"] is not True
        or raw["link_count"] != 1
        or type(raw["link_count"]) is not int
        or not _strict_json_value_equal(raw["named_alternate_data_streams"], [])
        or streams
        or raw["opened_without_reparse_follow"] is not True
        or raw["access_mask"] != 0x80000000
        or type(raw["access_mask"]) is not int
        or not _strict_json_value_equal(raw["share_access"], ["FILE_SHARE_READ"])
        or raw["write_access"] is not False
        or raw["delete_access"] is not False
        or raw["retained_through_each_exact_phase_launch"] is not True
        or raw["owner_process_identity_required"] is not True
        or raw["handle_slot_required"] is not True
        or raw["acquisition_disposition"] != _LEAF_ACQUISITION_DISPOSITION
    ):
        _stop(f"{label} differs from the retained interpreter")


def _require_command_derivation_contract(raw_value: Any) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_COMMAND_DERIVATION_FIELDS,
        label="command derivation contract",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _COMMAND_DERIVATION_POLICY
        or raw["projection_policy"] != _COMMAND_PROJECTION_POLICY
        or raw["canonical_file_hash_policy"] != _FILE_HASH_POLICY
        or raw["canonical_core_hash_policy"] != _CORE_HASH_POLICY
        or raw["e_file_sha256_flag"] != "--e-intent-sha256"
        or raw["e_core_sha256_flag"] != "--e-intent-core-sha256"
        or raw["e_file_sha256_insertion_policy"] != _E_FILE_INSERTION_POLICY
        or raw["e_core_sha256_insertion_policy"] != _E_CORE_INSERTION_POLICY
        or raw["successor_lineage_flag"] != _SUCCESSOR_LINEAGE_FLAG
        or raw["exact_argv_rederivation_required"] is not True
        or raw["final_command_carrier"] != _FINAL_COMMAND_CARRIER
        or raw["post_wait_rederivation_required"] is not True
        or raw["extra_argv_allowed"] is not False
        or raw["extra_environment_allowed"] is not False
    ):
        _stop("command derivation contract violates its exact policy")
    _require_exact_string_list(
        raw["python_isolated_flags"],
        ("-I", "-B"),
        label="command derivation Python flags",
    )
    _require_exact_string_list(
        raw["allowed_modes"],
        ALLOWED_MODES,
        label="command derivation modes",
    )
    _require_exact_string_list(
        raw["common_tail_flags"],
        _COMMON_TAIL_FLAGS,
        label="command derivation common flags",
    )
    _require_exact_string_list(
        raw["preterminal_suffix_flags"],
        _PRETERMINAL_SUFFIX_FLAGS,
        label="command derivation preterminal flags",
    )
    _require_exact_string_list(
        raw["terminal_suffix_flags"],
        _TERMINAL_SUFFIX_FLAGS,
        label="command derivation terminal flags",
    )
    _require_self_hash(
        raw,
        self_field="contract_sha256",
        label="command derivation contract",
    )
    return raw


def _require_supervisor_process_command_derivation(
    raw_value: Any,
    *,
    execution_capsule: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_SUPERVISOR_PROCESS_COMMAND_DERIVATION_FIELDS,
        label="supervisor process command derivation",
    )
    int_fields = {
        "schema_version",
        "isolated_flag_required",
        "no_site_flag_required",
        "dont_write_bytecode_flag_required",
    }
    bool_fields = {
        "peb_command_line_exact_direct_base_runtime_match_required",
        "in_process_sys_argv_exact_match_required",
        "logical_venv_identity_separately_bound_required",
        "supervisor_launcher_used_for_authorized_process_launch",
        "extra_argv_allowed",
        "extra_cwd_allowed",
    }
    list_fields = {
        "python_interpreter_flags",
        "python_sys_argv_prefix",
        "command_preimage_field_names",
    }
    string_fields = _SUPERVISOR_PROCESS_COMMAND_DERIVATION_FIELDS - (
        int_fields | bool_fields | list_fields
    )
    if (
        any(type(raw[field]) is not int for field in int_fields)
        or any(type(raw[field]) is not bool for field in bool_fields)
        or any(
            type(raw[field]) is not list or any(type(item) is not str for item in raw[field])
            for field in list_fields
        )
        or any(type(raw[field]) is not str for field in string_fields)
    ):
        _stop("supervisor process command derivation violates its exact JSON types")
    supervisor_code_root = _exact_absolute_path(
        raw["supervisor_code_root"],
        label="supervisor process command code root",
    )
    supervisor_state_root = _exact_absolute_path(
        raw["supervisor_state_root"],
        label="supervisor process command state root",
    )
    source_path = _exact_absolute_path(
        raw["supervisor_source_path"],
        label="supervisor process source path",
    )
    launcher_path = _exact_absolute_path(
        raw["supervisor_launcher_path"],
        label="supervisor process launcher path",
    )
    capsule_ancestor = _exact_object(
        execution_capsule["capsule_ancestor_lease"],
        fields=_ANCESTOR_LEASE_FIELDS,
        label="execution capsule ancestor lease",
    )
    expected_cwd = _exact_absolute_path(
        capsule_ancestor["anchor_path"],
        label="supervisor process cwd",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 2
        or raw["policy"] != _SUPERVISOR_PROCESS_COMMAND_DERIVATION_POLICY
        or source_path != supervisor_code_root / "aanca_supervisor.py"
        or launcher_path != supervisor_code_root / "launch_hidden.ps1"
        or _paths_overlap(supervisor_code_root, supervisor_state_root)
        or raw["supervisor_source_sha256"]
        != _require_sha256(
            raw["supervisor_source_sha256"],
            label="supervisor process source SHA-256",
        )
        or raw["supervisor_launcher_sha256"]
        != _require_sha256(
            raw["supervisor_launcher_sha256"],
            label="supervisor process launcher SHA-256",
        )
        or raw["program_path"] != execution_capsule["runtime_python_path"]
        or raw["program_sha256"] != execution_capsule["runtime_python_sha256"]
        or raw["createprocess_application_path"] != execution_capsule["runtime_python_path"]
        or raw["createprocess_application_sha256"] != execution_capsule["runtime_python_sha256"]
        or raw["logical_venv_python_path"] != execution_capsule["python_path"]
        or raw["logical_venv_python_sha256"] != execution_capsule["python_sha256"]
        or raw["runtime_python_path"] != execution_capsule["runtime_python_path"]
        or raw["runtime_python_sha256"] != execution_capsule["runtime_python_sha256"]
        or raw["expected_live_image_path"] != execution_capsule["runtime_python_path"]
        or raw["expected_live_image_sha256"] != execution_capsule["runtime_python_sha256"]
        or raw["direct_base_runtime_live_parity_policy"] != _DIRECT_BASE_RUNTIME_LIVE_PARITY_POLICY
        or not _strict_json_value_equal(
            raw["python_interpreter_flags"],
            list(_SUPERVISOR_PYTHON_ISOLATED_FLAGS),
        )
        or not _strict_json_value_equal(
            raw["python_sys_argv_prefix"],
            [
                str(source_path),
                "--root",
                str(supervisor_state_root),
                "run",
            ],
        )
        or raw["supervisor_launch_spec_path_binding"]
        != "Q.control_staging_projection.supervisor_launch_spec_path"
        or raw["staged_e_intent_path_binding"] != "Q.control_staging_projection.e_intent_path"
        or raw["os_launch_vector_policy"]
        != "program_path_then_python_flags_then_exact_option_a_staged_argv_v2"
        or raw["cwd"] != str(expected_cwd)
        or raw["command_preimage_policy"] != _SUPERVISOR_PROCESS_COMMAND_POLICY
        or not _strict_json_value_equal(
            raw["command_preimage_field_names"],
            sorted(_SUPERVISOR_PROCESS_COMMAND_PREIMAGE_FIELDS),
        )
        or raw["command_sha256_policy"] != _SUPERVISOR_PROCESS_COMMAND_HASH_POLICY
        or type(raw["isolated_flag_required"]) is not int
        or raw["isolated_flag_required"] != 1
        or type(raw["no_site_flag_required"]) is not int
        or raw["no_site_flag_required"] != 1
        or type(raw["dont_write_bytecode_flag_required"]) is not int
        or raw["dont_write_bytecode_flag_required"] != 1
        or raw["peb_command_line_exact_direct_base_runtime_match_required"] is not True
        or raw["in_process_sys_argv_exact_match_required"] is not True
        or raw["logical_venv_identity_separately_bound_required"] is not True
        or raw["supervisor_launcher_role"] != "nonexecuted_install_or_manual_recovery_helper"
        or raw["supervisor_launcher_used_for_authorized_process_launch"] is not False
        or raw["extra_argv_allowed"] is not False
        or raw["extra_cwd_allowed"] is not False
    ):
        _stop("supervisor process command derivation violates its exact policy")
    _require_self_hash(
        raw,
        self_field="contract_sha256",
        label="supervisor process command derivation",
    )
    return raw


def _require_terminal_client_launcher_ancestor_lease(
    raw_value: Any,
    *,
    supervisor_root: Path,
) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_TERMINAL_CLIENT_LAUNCHER_ANCESTOR_LEASE_FIELDS,
        label="terminal-client launcher ancestor lease",
    )
    records = raw["records"]
    if not isinstance(records, list) or len(records) != 1:
        _stop("terminal-client launcher ancestor lease has invalid records")
    record = _exact_object(
        records[0],
        fields=_ANCESTOR_RECORD_FIELDS,
        label="terminal-client launcher ancestor record",
    )
    record_path = _exact_absolute_path(
        record["path"],
        label="terminal-client launcher ancestor path",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _TERMINAL_CLIENT_LAUNCHER_ANCESTOR_LEASE_POLICY
        or raw["supervisor_root"] != str(supervisor_root)
        or record_path != supervisor_root
        or type(record["volume_serial_number"]) is not int
        or record["volume_serial_number"] < 0
        or type(record["file_id_128"]) is not str
        or _FILE_ID_128.fullmatch(record["file_id_128"]) is None
        or type(record["file_attributes"]) is not int
        or record["file_attributes"] < 0
        or record["file_attributes"] & 0x400
        or record["reparse_point"] is not False
        or type(raw["record_count"]) is not int
        or raw["record_count"] != 1
        or raw["records_root_sha256"] != _authority_json_sha256([record])
        or type(raw["directory_access_mask"]) is not int
        or raw["directory_access_mask"] != 0x80000080
        or not _strict_json_value_equal(
            raw["share_access"],
            ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        )
        or raw["delete_share"] is not False
        or raw["write_access"] is not False
        or raw["owner_process_identity_required"] is not True
        or raw["handle_slot_per_record_required"] is not True
        or raw["retained_from_q_verification_through_terminal_child_waitforexit"] is not True
        or raw["acquisition_disposition"] != _TERMINAL_CLIENT_LAUNCHER_ANCESTOR_DISPOSITION
    ):
        _stop("terminal-client launcher ancestor lease violates its exact policy")
    return raw


def _require_terminal_client_launcher_release(
    raw_value: Any,
    *,
    execution_capsule: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_TERMINAL_CLIENT_LAUNCHER_RELEASE_FIELDS,
        label="terminal-client launcher release",
    )
    supervisor_root = _exact_absolute_path(
        raw["supervisor_root"],
        label="terminal-client launcher supervisor root",
    )
    source_path = _exact_absolute_path(
        raw["source_path"],
        label="terminal-client launcher source path",
    )
    identity = _exact_object(
        raw["source_physical_identity"],
        fields=_PHYSICAL_FILE_IDENTITY_FIELDS,
        label="terminal-client launcher physical identity",
    )
    identity_path = _exact_absolute_path(
        identity["path"],
        label="terminal-client launcher physical path",
    )
    ancestor = _require_terminal_client_launcher_ancestor_lease(
        raw["source_ancestor_lease"],
        supervisor_root=supervisor_root,
    )
    nonnegative_int_fields = (
        "volume_serial_number",
        "device",
        "inode",
        "size_bytes",
        "mode",
        "file_attributes",
        "modified_time_ns",
        "changed_time_ns",
    )
    if (
        type(identity["schema_version"]) is not int
        or identity["schema_version"] != 1
        or identity["policy"] != _NO_FOLLOW_PHYSICAL_FILE_IDENTITY_POLICY
        or identity["role"] != "terminal-client-launcher"
        or identity_path != source_path
        or any(
            type(identity[field]) is not int or identity[field] < 0
            for field in nonnegative_int_fields
        )
        or type(identity["file_id_128"]) is not str
        or _FILE_ID_128.fullmatch(identity["file_id_128"]) is None
        or identity["mode"] > 0o7777
        or identity["mode"] & 0o222
        or identity["file_attributes"] & 0x400
        or not identity["file_attributes"] & 0x1
        or identity["regular_file"] is not True
        or identity["read_only"] is not True
        or type(identity["link_count"]) is not int
        or identity["link_count"] != 1
        or not _strict_json_value_equal(identity["named_alternate_data_streams"], [])
        or identity["opened_without_reparse_follow"] is not True
        or not _strict_json_value_equal(identity["share_access"], ["FILE_SHARE_READ"])
        or source_path != supervisor_root / _TERMINAL_CLIENT_LAUNCHER_SOURCE_FILENAME
        or type(raw["source_size_bytes"]) is not int
        or raw["source_size_bytes"] <= 0
        or raw["source_size_bytes"] != identity["size_bytes"]
        or raw["source_sha256"]
        != _require_sha256(
            identity["sha256"],
            label="terminal-client launcher source SHA-256",
        )
        or raw["source_physical_identity_root_sha256"] != _authority_json_sha256(identity)
        or raw["source_ancestor_lease_root_sha256"] != _authority_json_sha256(ancestor)
        or type(raw["source_leaf_access_mask"]) is not int
        or raw["source_leaf_access_mask"] != 0x80000000
        or not _strict_json_value_equal(
            raw["source_leaf_share_access"],
            ["FILE_SHARE_READ"],
        )
        or raw["source_delete_access"] is not False
        or raw["source_handle_retained_through_terminal_child_waitforexit"] is not True
        or raw["program_path"] != execution_capsule["runtime_python_path"]
        or raw["program_sha256"] != execution_capsule["runtime_python_sha256"]
        or raw["createprocess_application_path"] != execution_capsule["runtime_python_path"]
        or raw["createprocess_application_sha256"] != execution_capsule["runtime_python_sha256"]
        or raw["program_lease_identity_root_sha256"]
        != execution_capsule["runtime_python_lease_identity_root_sha256"]
        or raw["program_ancestor_lease_root_sha256"]
        != execution_capsule["runtime_python_ancestor_lease_root_sha256"]
        or raw["logical_venv_python_path"] != execution_capsule["python_path"]
        or raw["logical_venv_python_sha256"] != execution_capsule["python_sha256"]
        or raw["logical_venv_python_lease_identity_root_sha256"]
        != execution_capsule["python_lease_identity_root_sha256"]
        or raw["logical_venv_python_ancestor_lease_root_sha256"]
        != execution_capsule["python_ancestor_lease_root_sha256"]
        or raw["runtime_python_path"] != execution_capsule["runtime_python_path"]
        or raw["runtime_python_sha256"] != execution_capsule["runtime_python_sha256"]
        or raw["expected_live_image_path"] != execution_capsule["runtime_python_path"]
        or raw["expected_live_image_sha256"] != execution_capsule["runtime_python_sha256"]
        or raw["direct_base_runtime_live_parity_policy"] != _DIRECT_BASE_RUNTIME_LIVE_PARITY_POLICY
        or raw["runtime_python_lease_identity_root_sha256"]
        != execution_capsule["runtime_python_lease_identity_root_sha256"]
        or raw["runtime_python_ancestor_lease_root_sha256"]
        != execution_capsule["runtime_python_ancestor_lease_root_sha256"]
        or not _strict_json_value_equal(
            raw["python_isolated_flags"],
            list(_TERMINAL_CLIENT_PYTHON_ISOLATED_FLAGS),
        )
        or not _strict_json_value_equal(
            raw["python_sys_argv_prefix"],
            [str(source_path)],
        )
        or not _strict_json_value_equal(
            raw["process_argv_prefix"],
            [
                execution_capsule["runtime_python_path"],
                *_TERMINAL_CLIENT_PYTHON_ISOLATED_FLAGS,
                str(source_path),
            ],
        )
        or raw["cwd_binding"] != "E.project_root"
        or not _strict_json_value_equal(
            raw["final_argument_order"],
            list(_TERMINAL_CLIENT_LAUNCHER_FINAL_ARGUMENT_ORDER),
        )
        or not _strict_json_value_equal(
            raw["downstream_hash_insertions"],
            list(_TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDER_ORDER),
        )
        or raw["command_preimage_policy"] != _TERMINAL_CLIENT_LAUNCHER_COMMAND_POLICY
        or raw["command_sha256_policy"] != _SUPERVISOR_PROCESS_COMMAND_HASH_POLICY
        or raw["launch_intent_filename"] != _TERMINAL_CLIENT_LAUNCH_INTENT_FILENAME
        or raw["launch_intent_path_binding"]
        != "E.job.supervisor_job_dir/terminal_client_launch_intent.json"
        or raw["launch_intent_publication_policy"] != _TERMINAL_CLIENT_LAUNCH_INTENT_POLICY
        or any(
            not _strict_json_value_equal(raw[field], expected)
            for field, expected in (_TERMINAL_CLIENT_LAUNCHER_STATIC_CONTROL_VALUES.items())
        )
        or not _strict_json_value_equal(
            raw["sealed_input_allowlist"],
            ["supervisor_spec", "E"],
        )
        or raw["project_import_allowed"] is not False
        or raw["inherited_environment_for_child_allowed"] is not False
        or raw["createprocessw_exact_child_required"] is not True
        or raw["child_environment_encoding"] != "sorted_utf16le_double_nul_block_v1"
        or raw["child_environment_source"]
        != "sealed_E.expected_launch_environment.child_environment"
        or raw["verify_terminal_child_launch_topology"]
        != "launcher_base_direct_to_venv_redirector_to_runtime_child_v1"
        or raw["verify_terminal_immediate_redirector_program_path"]
        != execution_capsule["python_path"]
        or raw["verify_terminal_immediate_redirector_program_sha256"]
        != execution_capsule["python_sha256"]
        or raw["verify_terminal_runtime_child_program_path"]
        != execution_capsule["runtime_python_path"]
        or raw["verify_terminal_runtime_child_program_sha256"]
        != execution_capsule["runtime_python_sha256"]
        or raw["launcher_is_runtime_child_grandparent_required"] is not True
        or raw["same_job_no_breakaway_required"] is not True
        or raw["launcher_waits_for_child_exit_required"] is not True
        or raw["launcher_parent_live_through_child_exit_required"] is not True
        or raw["automatic_retry_allowed"] is not False
        or raw["fallback_allowed"] is not False
    ):
        _stop("terminal-client launcher release violates its exact policy")
    _require_self_hash(
        raw,
        self_field="release_root_sha256",
        label="terminal-client launcher release",
    )
    return raw


def _require_held_terminal_client_launcher(
    release: Mapping[str, Any],
) -> _HeldAuthorityFile:
    source_path = _exact_absolute_path(
        release["source_path"],
        label="terminal-client launcher source path",
    )
    held = _open_held_plain_file(
        source_path,
        label="terminal-client launcher source",
        require_read_only=True,
        maximum_bytes=_MAX_CONTROL_BYTES,
    )
    try:
        identity = _exact_object(
            release["source_physical_identity"],
            fields=_PHYSICAL_FILE_IDENTITY_FIELDS,
            label="terminal-client launcher physical identity",
        )
        observed = os.fstat(held.descriptor)
        volume, file_id = _native_file_identity_from_fd(held.descriptor)
        streams = _windows_named_data_streams(held.path)
        if (
            held.size_bytes != release["source_size_bytes"]
            or held.sha256 != release["source_sha256"]
            or identity["path"] != str(held.path)
            or identity["volume_serial_number"] != volume
            or identity["file_id_128"] != file_id
            or identity["device"] != int(observed.st_dev)
            or identity["inode"] != int(observed.st_ino)
            or identity["size_bytes"] != held.size_bytes
            or identity["mode"] != stat.S_IMODE(observed.st_mode)
            or identity["file_attributes"] != _file_attributes(observed)
            or identity["modified_time_ns"] != int(observed.st_mtime_ns)
            or identity["changed_time_ns"] != int(observed.st_ctime_ns)
            or identity["sha256"] != held.sha256
            or not _readonly_file(observed)
            or streams
        ):
            _stop("terminal-client launcher differs from its retained Q identity")
        ancestor = _exact_object(
            release["source_ancestor_lease"],
            fields=_TERMINAL_CLIENT_LAUNCHER_ANCESTOR_LEASE_FIELDS,
            label="terminal-client launcher ancestor lease",
        )
        record = _exact_object(
            cast(list[Any], ancestor["records"])[0],
            fields=_ANCESTOR_RECORD_FIELDS,
            label="terminal-client launcher ancestor record",
        )
        root = Path(cast(str, release["supervisor_root"]))
        root_stat = os.lstat(root)
        root_volume, root_file_id = _native_path_identity(root, directory=True)
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or stat.S_ISLNK(root_stat.st_mode)
            or _is_reparse(root_stat)
            or record["path"] != str(root)
            or record["volume_serial_number"] != root_volume
            or record["file_id_128"] != root_file_id
            or record["file_attributes"] != _file_attributes(root_stat)
            or record["reparse_point"] is not False
        ):
            _stop("terminal-client launcher ancestor differs from its live identity")
        return held
    except BaseException:
        os.close(held.descriptor)
        raise


def _require_supervisor_release(
    raw_value: Any,
    *,
    execution_capsule: Mapping[str, Any],
    plan_sha256: str,
    runtime_release_root_sha256: str,
    terminal_release_root_sha256: str,
) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_SUPERVISOR_RELEASE_FIELDS,
        label="supervisor release binding",
    )
    int_fields = {
        "schema_version",
        "supervisor_spec_schema_version",
        "q_e_custody_ready_max_bytes",
        "q_e_custody_ack_max_bytes",
    }
    bool_fields = {
        "q_e_independent_verifier_receipt_required",
        "q_e_no_science_before_custody_ack",
        "exact_job_object_membership_required",
    }
    object_fields = {
        "supervisor_process_command_derivation_contract",
        "terminal_client_launcher_release",
    }
    string_fields = _SUPERVISOR_RELEASE_FIELDS - (int_fields | bool_fields | object_fields)
    if (
        any(type(raw[field]) is not int for field in int_fields)
        or any(type(raw[field]) is not bool for field in bool_fields)
        or any(type(raw[field]) is not dict for field in object_fields)
        or any(type(raw[field]) is not str for field in string_fields)
    ):
        _stop("supervisor release binding violates its exact JSON types")
    source_sha256 = _require_sha256(
        raw["supervisor_source_sha256"],
        label="supervisor source SHA-256",
    )
    launcher_sha256 = _require_sha256(
        raw["supervisor_launcher_sha256"],
        label="supervisor launcher SHA-256",
    )
    process_derivation = _require_supervisor_process_command_derivation(
        raw["supervisor_process_command_derivation_contract"],
        execution_capsule=execution_capsule,
    )
    terminal_client_release = _require_terminal_client_launcher_release(
        raw["terminal_client_launcher_release"],
        execution_capsule=execution_capsule,
    )
    external_codex_handoff_authority_spec_file_sha256 = _require_sha256(
        raw["external_codex_handoff_authority_spec_file_sha256"],
        label="external Codex handoff authority-spec file",
    )
    external_codex_handoff_authority_spec_canonical_root_sha256 = _require_sha256(
        raw["external_codex_handoff_authority_spec_canonical_root_sha256"],
        label="external Codex handoff authority-spec canonical root",
    )
    release_root = _require_sha256(
        raw["supervisor_release_root_sha256"],
        label="supervisor release root",
    )
    external_release_root = _require_sha256(
        raw["external_control_plane_release_root_sha256"],
        label="external control-plane release root",
    )
    publication_id = raw["external_control_plane_publication_id"]
    qualification_attestation_path = _exact_absolute_path(
        raw["external_control_plane_release_qualification_attestation_path"],
        label="external control-plane release qualification attestation",
    )
    qualification_attestation_file_sha256 = _require_sha256(
        raw["external_control_plane_release_qualification_attestation_file_sha256"],
        label="external control-plane release qualification attestation file",
    )
    qualification_attestation_root_sha256 = _require_sha256(
        raw["external_control_plane_release_qualification_attestation_root_sha256"],
        label="external control-plane release qualification attestation root",
    )
    supervisor_code_root = _exact_absolute_path(
        raw["supervisor_code_root"],
        label="supervisor release code root",
    )
    supervisor_state_root = _exact_absolute_path(
        raw["supervisor_state_root"],
        label="supervisor release state root",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _SUPERVISOR_RELEASE_POLICY
        or raw["supervisor_policy"] != _SUPERVISOR_POLICY
        or type(raw["supervisor_spec_schema_version"]) is not int
        or raw["supervisor_spec_schema_version"] != 3
        or type(publication_id) is not str
        or re.fullmatch(r"cpr-[0-9a-f]{32}", publication_id) is None
        or qualification_attestation_path.name
        != _EXTERNAL_CONTROL_PLANE_QUALIFICATION_ATTESTATION_FILENAME
        or qualification_attestation_path.parent.name != publication_id
        or qualification_attestation_path.parent.parent.name
        != _EXTERNAL_CONTROL_PLANE_VERIFICATIONS_DIRECTORY_NAME
        or qualification_attestation_path.parent.parent.parent.name
        != _EXTERNAL_CONTROL_PLANE_RELEASE_DIRECTORY_NAME
        or supervisor_code_root.name != "supervisor"
        or supervisor_code_root.parent.name != external_release_root
        or _paths_overlap(supervisor_code_root, supervisor_state_root)
        or raw["supervisor_code_root"] != process_derivation["supervisor_code_root"]
        or raw["supervisor_state_root"] != process_derivation["supervisor_state_root"]
        or terminal_client_release["supervisor_root"] != str(supervisor_code_root)
        or raw["supervisor_source_path"] != process_derivation["supervisor_source_path"]
        or source_sha256 != process_derivation["supervisor_source_sha256"]
        or raw["supervisor_launcher_path"] != process_derivation["supervisor_launcher_path"]
        or launcher_sha256 != process_derivation["supervisor_launcher_sha256"]
        or raw["supervisor_program_path"] != process_derivation["program_path"]
        or raw["supervisor_program_sha256"] != process_derivation["program_sha256"]
        or raw["supervisor_runtime_python_path"] != process_derivation["runtime_python_path"]
        or raw["supervisor_runtime_python_sha256"] != process_derivation["runtime_python_sha256"]
        or raw["supervisor_process_command_derivation_contract_sha256"]
        != process_derivation["contract_sha256"]
        or raw["terminal_client_launcher_release_root_sha256"]
        != terminal_client_release["release_root_sha256"]
        or raw["plan_sha256"] != plan_sha256
        or raw["runtime_release_root_sha256"] != runtime_release_root_sha256
        or raw["terminal_release_root_sha256"] != terminal_release_root_sha256
        or release_root
        != _authority_json_sha256(
            {
                "policy": _SUPERVISOR_POLICY,
                "external_control_plane_release_root_sha256": external_release_root,
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
                "supervisor_code_root": str(supervisor_code_root),
                "supervisor_state_root": str(supervisor_state_root),
                "supervisor_source_path": process_derivation["supervisor_source_path"],
                "supervisor_source_sha256": source_sha256,
                "supervisor_launcher_path": process_derivation["supervisor_launcher_path"],
                "supervisor_launcher_sha256": launcher_sha256,
                "supervisor_program_path": process_derivation["program_path"],
                "supervisor_program_sha256": process_derivation["program_sha256"],
                "supervisor_runtime_python_path": process_derivation["runtime_python_path"],
                "supervisor_runtime_python_sha256": process_derivation["runtime_python_sha256"],
                "supervisor_process_command_derivation_contract_sha256": (
                    process_derivation["contract_sha256"]
                ),
                "terminal_client_launcher_release_root_sha256": (
                    terminal_client_release["release_root_sha256"]
                ),
                "postwake_custody_seed_policy": _POSTWAKE_CUSTODY_SEED_POLICY,
                "postwake_custody_pipe_derivation_policy": (
                    _POSTWAKE_CUSTODY_PIPE_DERIVATION_POLICY
                ),
                "q_e_custody_contract_policy": _Q_E_CUSTODY_CONTRACT_POLICY,
                "q_e_custody_handoff_policy": _Q_E_CUSTODY_HANDOFF_POLICY,
                "q_e_custody_transport": _Q_E_CUSTODY_TRANSPORT,
                "q_e_custody_ack_policy": _Q_E_CUSTODY_ACK_POLICY,
                "q_e_custody_receipt_policy": _Q_E_CUSTODY_RECEIPT_POLICY,
                "q_e_custody_receipt_filename": _Q_E_CUSTODY_RECEIPT_FILENAME,
                "terminal_custody_authority_template_root_sha256": (
                    _TERMINAL_CUSTODY_AUTHORITY_TEMPLATE_ROOT_SHA256
                ),
                "external_codex_handoff_policy": _EXTERNAL_CODEX_HANDOFF_POLICY,
                "external_codex_handoff_authority_spec_file_sha256": (
                    external_codex_handoff_authority_spec_file_sha256
                ),
                "external_codex_handoff_authority_spec_canonical_root_sha256": (
                    external_codex_handoff_authority_spec_canonical_root_sha256
                ),
                "internal_codex_wake_disposition": _INTERNAL_CODEX_WAKE_DISPOSITION,
            }
        )
        or raw["postwake_custody_seed_policy"] != _POSTWAKE_CUSTODY_SEED_POLICY
        or raw["postwake_custody_pipe_derivation_policy"]
        != _POSTWAKE_CUSTODY_PIPE_DERIVATION_POLICY
        or raw["q_e_custody_contract_policy"] != _Q_E_CUSTODY_CONTRACT_POLICY
        or raw["q_e_custody_handoff_policy"] != _Q_E_CUSTODY_HANDOFF_POLICY
        or raw["q_e_custody_transport"] != _Q_E_CUSTODY_TRANSPORT
        or raw["q_e_custody_ready_message_type"] != _Q_E_CUSTODY_READY_MESSAGE_TYPE
        or raw["q_e_custody_ack_policy"] != _Q_E_CUSTODY_ACK_POLICY
        or raw["q_e_custody_ack_message_type"] != _Q_E_CUSTODY_ACK_MESSAGE_TYPE
        or raw["q_e_custody_receipt_policy"] != _Q_E_CUSTODY_RECEIPT_POLICY
        or raw["q_e_custody_receipt_filename"] != _Q_E_CUSTODY_RECEIPT_FILENAME
        or type(raw["q_e_custody_ready_max_bytes"]) is not int
        or raw["q_e_custody_ready_max_bytes"] != _Q_E_CUSTODY_LINE_MAX_BYTES
        or type(raw["q_e_custody_ack_max_bytes"]) is not int
        or raw["q_e_custody_ack_max_bytes"] != _Q_E_CUSTODY_LINE_MAX_BYTES
        or raw["q_e_independent_verifier_receipt_required"] is not True
        or raw["q_e_no_science_before_custody_ack"] is not True
        or raw["terminal_custody_authority_template_root_sha256"]
        != _TERMINAL_CUSTODY_AUTHORITY_TEMPLATE_ROOT_SHA256
        or raw["external_codex_handoff_policy"] != _EXTERNAL_CODEX_HANDOFF_POLICY
        or raw["external_codex_handoff_authority_spec_file_sha256"]
        != external_codex_handoff_authority_spec_file_sha256
        or raw["external_codex_handoff_authority_spec_canonical_root_sha256"]
        != external_codex_handoff_authority_spec_canonical_root_sha256
        or raw["internal_codex_wake_disposition"] != _INTERNAL_CODEX_WAKE_DISPOSITION
        or raw["exact_job_object_membership_required"] is not True
    ):
        _stop("supervisor release binding violates its exact policy")
    _require_self_hash(
        raw,
        self_field="contract_sha256",
        label="supervisor release binding",
    )
    return raw


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


def _require_codex_handoff_base_authority(
    raw_value: Any,
    *,
    project_root: Path,
) -> dict[str, Any]:
    envelope = _exact_object(
        raw_value,
        fields={"schema", "payload", "payload_sha256"},
        label="Codex handoff base authority",
    )
    payload = _exact_object(
        envelope["payload"],
        fields={
            "authority_scope",
            "session_origin",
            "codex_cli",
            "resume_command_policy",
            "limits",
            "capability_policy",
            "branch_template_policy",
            "idle_completion_policy",
            "external_supervisor_handoff_policy",
            "operational_source",
        },
        label="Codex handoff base payload",
    )
    session = _exact_object(
        payload["session_origin"],
        fields={
            "session_id",
            "session_jsonl_path",
            "expected_cwd",
            "first_record",
            "session_file_identity",
        },
        label="Codex handoff session origin",
    )
    first = _exact_object(
        session["first_record"],
        fields={
            "record_type",
            "payload_id",
            "payload_session_id",
            "payload_cli_version",
            "raw_record_bytes_excluding_delimiter",
            "raw_record_sha256_excluding_delimiter",
            "delimiter_hex",
        },
        label="Codex handoff first record",
    )
    identity = _exact_object(
        session["session_file_identity"],
        fields={
            "volume_serial_number",
            "file_id_128",
            "creation_time_100ns",
            "file_attributes",
            "link_count",
            "directory",
            "reparse_point",
        },
        label="Codex handoff session file identity",
    )
    codex = _exact_object(
        payload["codex_cli"],
        fields={"path", "size_bytes", "sha256", "version_stdout"},
        label="Codex handoff CLI",
    )
    operational_source = _exact_object(
        payload["operational_source"],
        fields={
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
            "synthetic_inventory_file_sha256",
            "synthetic_inventory_root_sha256",
            "synthetic_inventory_size_bytes",
            "synthetic_gate_source_path",
            "synthetic_gate_source_sha256",
            "synthetic_gate_source_size_bytes",
        },
        label="Codex handoff operational source",
    )
    session_id = session["session_id"]
    uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    policy_roots_valid = all(
        _authority_json_sha256(payload[field]) == expected
        for field, expected in _CODEX_HANDOFF_FIXED_POLICY_ROOTS.items()
    )
    operational_path_fields = {
        "source_path",
        "source_inventory_path",
        "independent_audit_receipt_path",
        "authority_spec_path",
        "synthetic_inventory_path",
        "synthetic_gate_source_path",
    }
    operational_size_fields = {
        "source_size_bytes",
        "synthetic_inventory_size_bytes",
        "synthetic_gate_source_size_bytes",
    }
    operational_hash_fields = set(operational_source) - (
        operational_path_fields | operational_size_fields | {"schema"}
    )
    try:
        session_path = _exact_absolute_path(
            session["session_jsonl_path"],
            label="Codex handoff session JSONL",
        )
        codex_path = _exact_absolute_path(codex["path"], label="Codex handoff CLI")
        for field in operational_path_fields:
            _exact_absolute_path(
                operational_source[field],
                label=f"Codex handoff operational source {field}",
            )
        for field in operational_hash_fields:
            _require_sha256(
                operational_source[field],
                label=f"Codex handoff operational source {field}",
            )
    except (TypeError, ValueError):
        _stop("Codex handoff base contains an invalid absolute path or hash")
    if (
        envelope["schema"] != _CODEX_HANDOFF_BASE_OPERATIONAL_SCHEMA
        or payload["authority_scope"] != _CODEX_HANDOFF_BASE_OPERATIONAL_SCOPE
        or envelope["payload_sha256"] != _authority_json_sha256(payload)
        or not policy_roots_valid
        or not _strict_json_value_equal(
            payload["capability_policy"],
            {
                "production_arm_enabled": True,
                "real_resume_enabled": True,
                "synthetic_only": False,
            },
        )
        or re.fullmatch(uuid_pattern, session_id) is None
        or first["record_type"] != "session_meta"
        or first["payload_id"] != session_id
        or first["payload_session_id"] != session_id
        or type(first["payload_cli_version"]) is not str
        or type(first["raw_record_bytes_excluding_delimiter"]) is not int
        or not 0 <= first["raw_record_bytes_excluding_delimiter"] <= 4_194_304
        or _require_sha256(
            first["raw_record_sha256_excluding_delimiter"],
            label="Codex handoff first-record hash",
        )
        != first["raw_record_sha256_excluding_delimiter"]
        or first["delimiter_hex"] != "0a"
        or session_path.suffix.lower() != ".jsonl"
        or session["expected_cwd"] != str(project_root)
        or type(identity["volume_serial_number"]) is not int
        or not 0 <= identity["volume_serial_number"] <= 18_446_744_073_709_551_615
        or type(identity["file_id_128"]) is not str
        or _FILE_ID_128.fullmatch(identity["file_id_128"]) is None
        or type(identity["creation_time_100ns"]) is not int
        or identity["creation_time_100ns"] <= 0
        or type(identity["file_attributes"]) is not int
        or identity["file_attributes"] <= 0
        or identity["link_count"] != 1
        or type(identity["link_count"]) is not int
        or identity["directory"] is not False
        or identity["reparse_point"] is not False
        or codex_path.name.lower() != "codex.exe"
        or type(codex["size_bytes"]) is not int
        or codex["size_bytes"] <= 0
        or _require_sha256(codex["sha256"], label="Codex handoff CLI hash") != codex["sha256"]
        or type(codex["version_stdout"]) is not str
        or type(operational_source["schema"]) is not str
        or not operational_source["schema"]
        or any(
            type(operational_source[field]) is not int or operational_source[field] <= 0
            for field in operational_size_fields
        )
    ):
        _stop("Codex handoff base violates its exact operational profile")
    return envelope


def _require_codex_handoff_attempt_creation_authority(
    raw_value: Any,
    *,
    base_authority: Mapping[str, Any],
    expected_output_path: Path,
) -> dict[str, Any]:
    envelope = _exact_object(
        raw_value,
        fields={"schema", "payload", "payload_sha256"},
        label="Codex handoff attempt-creation authority",
    )
    payload = _exact_object(
        envelope["payload"],
        fields={
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
        label="Codex handoff attempt-creation payload",
    )
    one_use = _exact_object(
        payload["one_use_policy"],
        fields={
            "attempt_number",
            "maximum_attempts",
            "automatic_retry_allowed",
            "max_age_after_arm_ms",
            "branch_selection_time",
            "rendered_prompt_at_creation_allowed",
        },
        label="Codex handoff attempt-creation one-use policy",
    )
    base_payload = cast(Mapping[str, Any], base_authority["payload"])
    base_session = cast(Mapping[str, Any], base_payload["session_origin"])
    templates = cast(Mapping[str, Any], base_payload["branch_template_policy"])
    source = cast(Mapping[str, Any], base_payload["operational_source"])
    uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    marker_nonce = _require_sha256(
        payload["marker_nonce_hex"],
        label="Codex handoff marker nonce",
    )
    if (
        envelope["schema"] != _CODEX_HANDOFF_ATTEMPT_CREATION_SCHEMA
        or payload["authority_scope"] != _CODEX_HANDOFF_ATTEMPT_CREATION_SCOPE
        or payload["base_authority_payload_sha256"] != base_authority["payload_sha256"]
        or payload["session_id"] != base_session["session_id"]
        or re.fullmatch(uuid_pattern, cast(str, payload["session_id"])) is None
        or re.fullmatch(uuid_pattern, cast(str, payload["turn_id"])) is None
        or payload["marker"] != f"AANCA_CURRENT_SESSION_IDLE_{marker_nonce}"
        or payload["success_template_policy_root_sha256"]
        != templates["success_template_policy_root_sha256"]
        or payload["diagnosis_template_policy_root_sha256"]
        != templates["diagnosis_template_policy_root_sha256"]
        or payload["authority_spec_payload_sha256"] != source["authority_spec_payload_sha256"]
        or payload["arm_algorithm_contract_root_sha256"]
        != _CODEX_HANDOFF_ARM_ALGORITHM_CONTRACT_ROOT_SHA256
        or payload["attempt_authority_output_path"] != str(expected_output_path)
        or payload["attempt_authority_schema"] != _CODEX_HANDOFF_ATTEMPT_SCHEMA
        or payload["arm_algorithm"] != _CODEX_HANDOFF_ARM_ALGORITHM
        or payload["required_absent_before"] is not True
        or payload["create_new_required"] is not True
        or not _strict_json_value_equal(
            one_use,
            {
                "attempt_number": 1,
                "maximum_attempts": 1,
                "automatic_retry_allowed": False,
                "max_age_after_arm_ms": _CODEX_HANDOFF_BOUNDARY_MAX_AGE_MS,
                "branch_selection_time": "postterminal",
                "rendered_prompt_at_creation_allowed": False,
            },
        )
        or envelope["payload_sha256"] != _authority_json_sha256(payload)
    ):
        _stop("Codex handoff attempt-creation authority violates its exact crosslinks")
    return envelope


def _require_external_codex_handoff(
    raw_value: Any,
    *,
    control_staging: Mapping[str, Any],
    e_file_sha256: str,
    e_core_sha256: str,
    attempt_creation: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_EXTERNAL_CODEX_HANDOFF_FIELDS,
        label="external Codex handoff",
    )
    job_dir = Path(cast(str, control_staging["final_job_dir"]))
    creation_payload = cast(Mapping[str, Any], attempt_creation["payload"])
    expected = {
        "policy": _EXTERNAL_CODEX_HANDOFF_POLICY,
        "staged_e_intent_path": control_staging["e_intent_path"],
        "staged_e_intent_file_sha256": e_file_sha256,
        "staged_e_intent_core_root_sha256": e_core_sha256,
        "attempt_creation_authority_payload_sha256": attempt_creation["payload_sha256"],
        "attempt_authority_output_path": creation_payload["attempt_authority_output_path"],
        "terminal_handoff_receipt_output_path": str(
            job_dir / "external_codex_terminal_handoff.json"
        ),
        "internal_codex_wake_allowed": False,
        "legacy_handoff_session_allowed": False,
        "single_wake_owner": _EXTERNAL_CODEX_HANDOFF_SINGLE_WAKE_OWNER,
    }
    if not _strict_json_value_equal(raw, expected):
        _stop("external Codex handoff differs from the exact Q/E derivation")
    return raw


def _require_expected_launch_environment(
    raw_value: Any,
    *,
    attempt_nonce: str,
) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_EXPECTED_ENVIRONMENT_FIELDS,
        label="expected launch environment",
    )
    supervisor = _exact_object(
        raw["supervisor_environment"],
        fields=_SCIENCE_SUPERVISOR_ENVIRONMENT_NAMES,
        label="expected supervisor environment",
    )
    child = _exact_object(
        raw["child_environment"],
        fields={*_SCIENCE_SUPERVISOR_ENVIRONMENT_NAMES, _SUPERVISOR_ATTEMPT_NONCE_KEY},
        label="expected child environment",
    )
    for name, value in {**supervisor, **child}.items():
        if (
            type(name) is not str
            or name != name.upper()
            or not name
            or "\x00" in name
            or "=" in name
            or type(value) is not str
            or "\x00" in value
        ):
            _stop("expected launch environment contains an invalid name or value")
    for name in _SCIENCE_SUPERVISOR_ENVIRONMENT_NAMES:
        _exact_absolute_path(
            supervisor[name],
            label=f"expected supervisor environment {name}",
        )
    expected_child = {**supervisor, _SUPERVISOR_ATTEMPT_NONCE_KEY: attempt_nonce}
    supervisor_sha256 = _authority_json_sha256(supervisor)
    child_sha256 = _authority_json_sha256(child)
    launch_root = _authority_json_sha256(
        {
            "supervisor_environment_sha256": supervisor_sha256,
            "child_environment_sha256": child_sha256,
            "attempt_nonce_key": _SUPERVISOR_ATTEMPT_NONCE_KEY,
            "attempt_nonce": attempt_nonce,
        }
    )
    unsigned = {key: item for key, item in raw.items() if key != "envelope_sha256"}
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _EXPECTED_LAUNCH_ENVIRONMENT_POLICY
        or raw["attempt_nonce_key"] != _SUPERVISOR_ATTEMPT_NONCE_KEY
        or raw["attempt_nonce"] != attempt_nonce
        or not _strict_json_value_equal(child, expected_child)
        or supervisor["TEMP"] != str(Path(supervisor["LOCALAPPDATA"]) / "Temp")
        or supervisor["TMP"] != str(Path(supervisor["LOCALAPPDATA"]) / "Temp")
        or raw["supervisor_environment_sha256"] != supervisor_sha256
        or raw["exact_environment_sha256"] != child_sha256
        or raw["launch_environment_root_sha256"] != launch_root
        or raw["environment_names_uppercase"] is not True
        or raw["case_collisions_rejected"] is not True
        or raw["nul_and_equals_in_names_rejected"] is not True
        or raw["nul_in_values_rejected"] is not True
        or raw["unspecified_inherited_variables_allowed"] is not False
        or raw["extra_variables_allowed"] is not False
        or raw["envelope_sha256"] != _authority_json_sha256(unsigned)
    ):
        _stop("expected launch environment violates its exact16 policy")
    return raw


def _require_process_environment_binding(
    raw_value: Any,
    *,
    expected_environment: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_PROCESS_ENVIRONMENT_BINDING_FIELDS,
        label="process environment binding",
    )
    unsigned = {key: item for key, item in raw.items() if key != "binding_sha256"}
    nonce = cast(str, expected_environment["attempt_nonce"])
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _PROCESS_ENVIRONMENT_BINDING_POLICY
        or raw["expected_environment_envelope_sha256"] != expected_environment["envelope_sha256"]
        or raw["launch_environment_root_sha256"]
        != expected_environment["launch_environment_root_sha256"]
        or raw["attempt_nonce_key"] != _SUPERVISOR_ATTEMPT_NONCE_KEY
        or raw["attempt_nonce_sha256"] != hashlib.sha256(nonce.encode("ascii")).hexdigest()
        or raw["exact_supervisor_environment_sha256"]
        != expected_environment["supervisor_environment_sha256"]
        or raw["exact_environment_sha256"] != expected_environment["exact_environment_sha256"]
        or raw["exact_integrity_verifier_environment_sha256"]
        != expected_environment["exact_environment_sha256"]
        or raw["binding_sha256"] != _authority_json_sha256(unsigned)
    ):
        _stop("process environment binding differs from exact Q environment")
    return raw


def _require_explicit_utc_timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str or not value.endswith("Z") or "\x00" in value:
        _stop(f"{label} is not one explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as exc:
        raise CapsuleBootstrapError(f"{label} is not one explicit UTC timestamp") from exc
    offset = parsed.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        _stop(f"{label} is not one explicit UTC timestamp")
    return value


def _windows_filetime_100ns_to_utc(value: int, *, label: str) -> str:
    unix_100ns = value - 116444736000000000
    seconds, remainder = divmod(unix_100ns, 10_000_000)
    try:
        instant = datetime.fromtimestamp(
            seconds + remainder / 10_000_000,
            tz=UTC,
        )
    except (OSError, OverflowError, ValueError) as exc:
        raise CapsuleBootstrapError(f"{label} FILETIME is outside the UTC range") from exc
    return instant.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _build_q_e_custody_contract(*, supervisor_job_directory: Path) -> dict[str, Any]:
    job_directory = _exact_absolute_path(
        str(supervisor_job_directory),
        label="Q/E custody supervisor job directory",
    )
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "policy": _Q_E_CUSTODY_CONTRACT_POLICY,
        "transport": _Q_E_CUSTODY_TRANSPORT,
        "ready_message_type": _Q_E_CUSTODY_READY_MESSAGE_TYPE,
        "ack_policy": _Q_E_CUSTODY_ACK_POLICY,
        "ack_message_type": _Q_E_CUSTODY_ACK_MESSAGE_TYPE,
        "receipt_policy": _Q_E_CUSTODY_RECEIPT_POLICY,
        "receipt_path": str(job_directory / _Q_E_CUSTODY_RECEIPT_FILENAME),
        "ready_max_bytes": _Q_E_CUSTODY_LINE_MAX_BYTES,
        "ack_max_bytes": _Q_E_CUSTODY_LINE_MAX_BYTES,
        "leaf_target_access_mask": _Q_E_CUSTODY_LEAF_TARGET_ACCESS_MASK,
        "ancestor_target_access_mask": _Q_E_CUSTODY_ANCESTOR_TARGET_ACCESS_MASK,
        "duplicate_options": _Q_E_CUSTODY_DUPLICATE_OPTIONS,
        "close_source": False,
        "source_custody_retained_until_supervisor_ack": True,
        "supervisor_retention_policy": _Q_E_CUSTODY_SUPERVISOR_RETENTION_POLICY,
        "independent_verifier_receipt_required": True,
        "scientific_inputs_before_ack_allowed": False,
        "automatic_retry_allowed": False,
    }
    return {
        **unsigned,
        "contract_sha256": _authority_json_sha256(unsigned),
    }


def _require_q_e_custody_contract(
    raw_value: Any,
    *,
    supervisor_job_directory: Path,
) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_Q_E_CUSTODY_CONTRACT_FIELDS,
        label="Q/E custody contract",
    )
    expected = _build_q_e_custody_contract(
        supervisor_job_directory=supervisor_job_directory,
    )
    if not _strict_json_value_equal(raw, expected):
        _stop("Q/E custody contract violates its exact one-shot policy")
    return expected


def _require_q_e_process_identity(
    raw_value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_E_PROCESS_IDENTITY_FIELDS,
        label=label,
    )
    if (
        type(raw["pid"]) is not int
        or raw["pid"] <= 0
        or type(raw["creation_time_100ns"]) is not int
        or raw["creation_time_100ns"] <= 0
        or _require_explicit_utc_timestamp(
            raw["creation_time_utc"],
            label=f"{label} creation time",
        )
        != _windows_filetime_100ns_to_utc(
            raw["creation_time_100ns"],
            label=f"{label} creation time",
        )
        or str(
            _exact_absolute_path(
                raw["program_path"],
                label=f"{label} program path",
            )
        )
        != raw["program_path"]
        or _require_sha256(
            raw["program_sha256"],
            label=f"{label} program SHA-256",
        )
        != raw["program_sha256"]
        or _require_sha256(
            raw["command_sha256"],
            label=f"{label} command SHA-256",
        )
        != raw["command_sha256"]
    ):
        _stop(f"{label} violates its exact process identity")
    return dict(raw)


def _require_q_e_control_physical_identity(
    raw_value: Any,
    *,
    label: str,
    expected_path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_CONTROL_PUBLICATION_IDENTITY_FIELDS,
        label=label,
    )
    path = _exact_absolute_path(raw["path"], label=f"{label} path")
    integer_fields = (
        "volume_serial_number",
        "size_bytes",
        "file_attributes",
        "link_count",
    )
    if (
        path != expected_path
        or type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _CONTROL_PUBLICATION_IDENTITY_POLICY
        or any(type(raw[field]) is not int or raw[field] < 0 for field in integer_fields)
        or raw["size_bytes"] <= 0
        or type(raw["file_id_128"]) is not str
        or _FILE_ID_128.fullmatch(raw["file_id_128"]) is None
        or _require_sha256(raw["sha256"], label=f"{label} SHA-256")
        != _require_sha256(expected_sha256, label=f"{label} expected SHA-256")
        or raw["file_attributes"] & 0x400
        or raw["regular_file"] is not True
        or raw["read_only"] is not True
        or raw["link_count"] != 1
        or not _strict_json_value_equal(raw["named_alternate_data_streams"], [])
        or raw["opened_without_reparse_follow"] is not True
        or not _strict_json_value_equal(raw["share_access"], ["FILE_SHARE_READ"])
        or raw["write_handle_retained"] is not False
        or raw["delete_access"] is not False
    ):
        _stop(f"{label} violates its exact immutable identity")
    try:
        value = os.lstat(path)
        volume, file_id = _native_path_identity(path, directory=False)
        streams = _windows_named_data_streams(path)
    except OSError as exc:
        raise CapsuleBootstrapError(f"{label} live readback failed") from exc
    if (
        not stat.S_ISREG(value.st_mode)
        or stat.S_ISLNK(value.st_mode)
        or _is_reparse(value)
        or raw["volume_serial_number"] != volume
        or raw["file_id_128"] != file_id
        or raw["size_bytes"] != int(value.st_size)
        or raw["file_attributes"] != _file_attributes(value)
        or not _readonly_file(value)
        or int(value.st_nlink) != 1
        or streams
    ):
        _stop(f"{label} differs from its live retained file")
    return dict(raw)


def _require_q_e_e_ancestor_lease(
    raw_value: Any,
    *,
    supervisor_job_directory: Path,
    e_intent_path: Path,
) -> dict[str, Any]:
    label = "Q/E custody E ancestor lease"
    raw = _exact_object(
        raw_value,
        fields=_Q_E_CUSTODY_E_ANCESTOR_LEASE_FIELDS,
        label=label,
    )
    if supervisor_job_directory.parent.name != _SUPERVISOR_JOBS_DIRECTORY_NAME:
        _stop("Q/E custody E job is outside supervisor-root/jobs")
    supervisor_root = supervisor_job_directory.parent.parent
    expected_staging_root = supervisor_root / _CONTROL_STAGING_DIRECTORY_NAME
    expected_staging_directory = expected_staging_root / supervisor_job_directory.name
    expected_e_intent_path = expected_staging_directory / _E_INTENT_FILENAME
    if e_intent_path != expected_e_intent_path:
        _stop("Q/E custody E leaf is outside its exact control-staging directory")
    expected_paths = (
        supervisor_root,
        expected_staging_root,
        expected_staging_directory,
    )
    items = raw["records"]
    if type(items) is not list or len(items) != 3:
        _stop("Q/E custody E ancestor lease must contain exactly three records")
    records: list[dict[str, Any]] = []
    for index, (item, expected_path) in enumerate(
        zip(items, expected_paths, strict=True),
        start=1,
    ):
        record = _exact_object(
            item,
            fields=_ANCESTOR_RECORD_FIELDS,
            label=f"{label} record {index}",
        )
        path = _exact_absolute_path(
            record["path"],
            label=f"{label} record {index} path",
        )
        if (
            path != expected_path
            or type(record["volume_serial_number"]) is not int
            or record["volume_serial_number"] < 0
            or type(record["file_id_128"]) is not str
            or _FILE_ID_128.fullmatch(record["file_id_128"]) is None
            or type(record["file_attributes"]) is not int
            or record["file_attributes"] < 0
            or record["file_attributes"] & 0x400
            or record["reparse_point"] is not False
        ):
            _stop(f"{label} record {index} is invalid")
        try:
            value = os.lstat(path)
            volume, file_id = _native_path_identity(path, directory=True)
        except OSError as exc:
            raise CapsuleBootstrapError(f"{label} record {index} live readback failed") from exc
        if (
            not stat.S_ISDIR(value.st_mode)
            or stat.S_ISLNK(value.st_mode)
            or _is_reparse(value)
            or record["volume_serial_number"] != volume
            or record["file_id_128"] != file_id
            or record["file_attributes"] != _file_attributes(value)
        ):
            _stop(f"{label} record {index} differs from the live directory")
        records.append(dict(record))
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _Q_E_CUSTODY_E_ANCESTOR_LEASE_POLICY
        or raw["supervisor_root"] != str(supervisor_root)
        or type(raw["record_count"]) is not int
        or raw["record_count"] != 3
        or raw["records_root_sha256"] != _authority_json_sha256(records)
        or type(raw["directory_access_mask"]) is not int
        or raw["directory_access_mask"] != _Q_E_CUSTODY_ANCESTOR_TARGET_ACCESS_MASK
        or not _strict_json_value_equal(
            raw["share_access"],
            ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        )
        or raw["delete_share"] is not False
        or raw["write_access"] is not False
        or raw["owner_process_identity_required"] is not True
        or raw["handle_slot_per_record_required"] is not True
        or raw["continuous_overlap_through_independent_verification_required"] is not True
        or raw["continuous_overlap_into_supervisor_required"] is not True
        or raw["acquisition_disposition"] != _Q_E_CUSTODY_E_ANCESTOR_DISPOSITION
    ):
        _stop("Q/E custody E ancestor lease violates its exact policy")
    return {**raw, "records": records}


def _require_control_staging_ancestor_lease(
    raw_value: Any,
    *,
    supervisor_root: Path,
    staging_directory: Path,
) -> dict[str, Any]:
    label = "control-staging ancestor lease"
    raw = _exact_object(
        raw_value,
        fields=_CONTROL_STAGING_ANCESTOR_LEASE_FIELDS,
        label=label,
    )
    expected_staging_root = supervisor_root / _CONTROL_STAGING_DIRECTORY_NAME
    if staging_directory.parent != expected_staging_root:
        _stop("control-staging ancestor lease directory is outside its exact root")
    expected_paths = (
        supervisor_root,
        expected_staging_root,
        staging_directory,
    )
    items = raw["records"]
    if type(items) is not list or len(items) != 3:
        _stop("control-staging ancestor lease must contain exactly three records")
    records: list[dict[str, Any]] = []
    for index, (item, expected_path) in enumerate(
        zip(items, expected_paths, strict=True),
        start=1,
    ):
        record = _exact_object(
            item,
            fields=_ANCESTOR_RECORD_FIELDS,
            label=f"{label} record {index}",
        )
        path = _exact_absolute_path(
            record["path"],
            label=f"{label} record {index} path",
        )
        if (
            path != expected_path
            or type(record["volume_serial_number"]) is not int
            or record["volume_serial_number"] < 0
            or type(record["file_id_128"]) is not str
            or _FILE_ID_128.fullmatch(record["file_id_128"]) is None
            or type(record["file_attributes"]) is not int
            or record["file_attributes"] < 0
            or record["file_attributes"] & 0x400
            or record["reparse_point"] is not False
        ):
            _stop(f"{label} record {index} is invalid")
        try:
            value = os.lstat(path)
            volume, file_id = _native_path_identity(path, directory=True)
        except OSError as exc:
            raise CapsuleBootstrapError(f"{label} record {index} live readback failed") from exc
        if (
            not stat.S_ISDIR(value.st_mode)
            or stat.S_ISLNK(value.st_mode)
            or _is_reparse(value)
            or record["volume_serial_number"] != volume
            or record["file_id_128"] != file_id
            or record["file_attributes"] != _file_attributes(value)
        ):
            _stop(f"{label} record {index} differs from the live directory")
        records.append(dict(record))
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _CONTROL_STAGING_ANCESTOR_LEASE_POLICY
        or raw["supervisor_root"] != str(supervisor_root)
        or type(raw["record_count"]) is not int
        or raw["record_count"] != 3
        or raw["records_root_sha256"] != _authority_json_sha256(records)
        or type(raw["directory_access_mask"]) is not int
        or raw["directory_access_mask"] != _Q_E_CUSTODY_ANCESTOR_TARGET_ACCESS_MASK
        or not _strict_json_value_equal(
            raw["share_access"],
            ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        )
        or raw["delete_share"] is not False
        or raw["write_access"] is not False
        or raw["owner_process_identity_required"] is not True
        or raw["handle_slot_per_record_required"] is not True
        or raw["continuous_overlap_through_supervisor_stage_ack_required"] is not True
        or raw["acquisition_disposition"] != _CONTROL_STAGING_ANCESTOR_ACQUISITION_DISPOSITION
    ):
        _stop("control-staging ancestor lease violates its exact policy")
    return {**raw, "records": records}


def _require_q_e_custody_ready(
    raw_value: Any,
    *,
    contract: Mapping[str, Any],
    project_root: Path,
    supervisor_job_directory: Path,
    supervisor_job_id: str,
    q_authority_root_sha256: str,
    q_file_sha256: str,
    e_intent_path: Path,
    e_file_sha256: str,
    independent_verifier_receipt_sha256: str,
) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_Q_E_CUSTODY_READY_FIELDS,
        label="Q/E custody READY",
    )
    canonical_contract = _require_q_e_custody_contract(
        contract,
        supervisor_job_directory=supervisor_job_directory,
    )
    supervisor = _require_q_e_process_identity(
        raw["supervisor_process_identity"],
        label="Q/E custody READY supervisor identity",
    )
    controller = _require_q_e_process_identity(
        raw["controller_process_identity"],
        label="Q/E custody READY controller identity",
    )
    expected_q_path = project_root / "artifacts" / "resource_control" / _Q_REPLACEMENT_V2_FILENAME
    q_identity = _require_q_e_control_physical_identity(
        raw["q_leaf_physical_identity"],
        label="Q/E custody Q leaf identity",
        expected_path=expected_q_path,
        expected_sha256=q_file_sha256,
    )
    e_identity = _require_q_e_control_physical_identity(
        raw["e_leaf_physical_identity"],
        label="Q/E custody E leaf identity",
        expected_path=e_intent_path,
        expected_sha256=e_file_sha256,
    )
    q_lease = _exact_object(
        raw["q_ancestor_lease"],
        fields=_PUBLICATION_ANCESTOR_LEASE_FIELDS,
        label="Q/E custody Q ancestor lease",
    )
    _require_publication_ancestor_lease(q_lease, project_root=project_root)
    e_lease = _require_q_e_e_ancestor_lease(
        raw["e_ancestor_lease"],
        supervisor_job_directory=supervisor_job_directory,
        e_intent_path=e_intent_path,
    )
    q_handles = raw["q_ancestor_handles"]
    e_handles = raw["e_ancestor_handles"]
    if (
        type(q_handles) is not list
        or len(q_handles) != 3
        or type(e_handles) is not list
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
        _stop("Q/E custody READY handle slots are incomplete or not unique")
    unsigned = {key: item for key, item in raw.items() if key != "handoff_root_sha256"}
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _Q_E_CUSTODY_HANDOFF_POLICY
        or raw["message_type"] != _Q_E_CUSTODY_READY_MESSAGE_TYPE
        or raw["transport"] != _Q_E_CUSTODY_TRANSPORT
        or raw["contract_sha256"] != canonical_contract["contract_sha256"]
        or type(raw["supervisor_job_id"]) is not str
        or _IDENTIFIER.fullmatch(raw["supervisor_job_id"]) is None
        or raw["supervisor_job_id"] != supervisor_job_id
        or supervisor_job_directory.name != supervisor_job_id
        or _require_explicit_utc_timestamp(
            raw["windows_boot_time_utc"],
            label="Q/E custody Windows boot time",
        )
        != raw["windows_boot_time_utc"]
        or _require_sha256(
            raw["q_authority_root_sha256"],
            label="Q/E custody Q authority root",
        )
        != _require_sha256(
            q_authority_root_sha256,
            label="trusted Q authority root",
        )
        or _require_sha256(
            raw["q_file_sha256"],
            label="Q/E custody Q file SHA-256",
        )
        != _require_sha256(q_file_sha256, label="trusted Q file SHA-256")
        or _require_sha256(
            raw["e_file_sha256"],
            label="Q/E custody E file SHA-256",
        )
        != _require_sha256(e_file_sha256, label="trusted E file SHA-256")
        or e_intent_path
        != (
            supervisor_job_directory.parent.parent
            / _CONTROL_STAGING_DIRECTORY_NAME
            / supervisor_job_id
            / _E_INTENT_FILENAME
        )
        or len(q_lease["records"]) != len(q_handles)
        or len(e_lease["records"]) != len(e_handles)
        or type(raw["leaf_target_access_mask"]) is not int
        or raw["leaf_target_access_mask"] != canonical_contract["leaf_target_access_mask"]
        or type(raw["ancestor_target_access_mask"]) is not int
        or raw["ancestor_target_access_mask"] != canonical_contract["ancestor_target_access_mask"]
        or type(raw["duplicate_options"]) is not int
        or raw["duplicate_options"] != canonical_contract["duplicate_options"]
        or raw["close_source"] is not canonical_contract["close_source"]
        or raw["source_custody_retained_until_supervisor_ack"]
        is not canonical_contract["source_custody_retained_until_supervisor_ack"]
        or raw["supervisor_retention_policy"] != canonical_contract["supervisor_retention_policy"]
        or _require_sha256(
            raw["independent_verifier_receipt_sha256"],
            label="Q/E independent verifier receipt SHA-256",
        )
        != _require_sha256(
            independent_verifier_receipt_sha256,
            label="trusted independent verifier receipt SHA-256",
        )
        or raw["scientific_inputs_before_ack_allowed"]
        is not canonical_contract["scientific_inputs_before_ack_allowed"]
        or raw["automatic_retry_allowed"] is not canonical_contract["automatic_retry_allowed"]
        or raw["handoff_root_sha256"] != _authority_json_sha256(unsigned)
    ):
        _stop("Q/E custody READY violates its exact fail-closed policy")
    return {
        **raw,
        "supervisor_process_identity": supervisor,
        "controller_process_identity": controller,
        "q_leaf_physical_identity": q_identity,
        "q_ancestor_lease": dict(q_lease),
        "e_leaf_physical_identity": e_identity,
        "e_ancestor_lease": e_lease,
        "q_ancestor_handles": list(q_handles),
        "e_ancestor_handles": list(e_handles),
    }


def _build_q_e_leaf_retained_binding(
    *,
    role: str,
    physical_identity: Mapping[str, Any],
    target_handle_value: int,
    retention_policy: str,
) -> dict[str, Any]:
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "policy": _Q_E_CUSTODY_LEAF_RETAINED_BINDING_POLICY,
        "role": role,
        "path": physical_identity["path"],
        "target_handle_value": target_handle_value,
        "target_granted_access_mask": _Q_E_CUSTODY_MAPPED_FILE_GRANTED_ACCESS_MASK,
        "physical_identity": dict(physical_identity),
        "retained": True,
        "retention_policy": retention_policy,
    }
    result = {
        **unsigned,
        "binding_sha256": _authority_json_sha256(unsigned),
    }
    _exact_object(
        result,
        fields=_Q_E_CUSTODY_LEAF_RETAINED_BINDING_FIELDS,
        label=f"Q/E custody {role} retained binding",
    )
    return result


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
        unsigned: dict[str, Any] = {
            "schema_version": 1,
            "policy": _Q_E_CUSTODY_ANCESTOR_RETAINED_BINDING_POLICY,
            "role": role,
            "index": index,
            "path": record["path"],
            "target_handle_value": target_handle,
            "target_granted_access_mask": _Q_E_CUSTODY_MAPPED_FILE_GRANTED_ACCESS_MASK,
            "volume_serial_number": record["volume_serial_number"],
            "file_id_128": record["file_id_128"],
            "file_attributes": record["file_attributes"],
            "reparse_point": False,
            "retained": True,
            "retention_policy": retention_policy,
        }
        binding = {
            **unsigned,
            "binding_sha256": _authority_json_sha256(unsigned),
        }
        _exact_object(
            binding,
            fields=_Q_E_CUSTODY_ANCESTOR_RETAINED_BINDING_FIELDS,
            label=f"Q/E custody {role} retained binding {index}",
        )
        result.append(binding)
    return result


def _build_q_e_custody_receipt(
    *,
    contract: Mapping[str, Any],
    ready: Mapping[str, Any],
) -> dict[str, Any]:
    retention = contract["supervisor_retention_policy"]
    q_leaf = _build_q_e_leaf_retained_binding(
        role="q-leaf",
        physical_identity=cast(Mapping[str, Any], ready["q_leaf_physical_identity"]),
        target_handle_value=cast(int, ready["q_leaf_handle"]),
        retention_policy=cast(str, retention),
    )
    e_leaf = _build_q_e_leaf_retained_binding(
        role="e-leaf",
        physical_identity=cast(Mapping[str, Any], ready["e_leaf_physical_identity"]),
        target_handle_value=cast(int, ready["e_leaf_handle"]),
        retention_policy=cast(str, retention),
    )
    q_ancestors = _build_q_e_ancestor_retained_bindings(
        role="q-ancestor",
        records=cast(
            Sequence[Mapping[str, Any]],
            cast(Mapping[str, Any], ready["q_ancestor_lease"])["records"],
        ),
        target_handles=cast(Sequence[int], ready["q_ancestor_handles"]),
        retention_policy=cast(str, retention),
    )
    e_ancestors = _build_q_e_ancestor_retained_bindings(
        role="e-ancestor",
        records=cast(
            Sequence[Mapping[str, Any]],
            cast(Mapping[str, Any], ready["e_ancestor_lease"])["records"],
        ),
        target_handles=cast(Sequence[int], ready["e_ancestor_handles"]),
        retention_policy=cast(str, retention),
    )
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "policy": _Q_E_CUSTODY_RECEIPT_POLICY,
        "status": _Q_E_CUSTODY_RECEIPT_STATUS,
        "contract_sha256": contract["contract_sha256"],
        "handoff_root_sha256": ready["handoff_root_sha256"],
        "supervisor_job_id": ready["supervisor_job_id"],
        "supervisor_process_identity": ready["supervisor_process_identity"],
        "controller_process_identity": ready["controller_process_identity"],
        "windows_boot_time_utc": ready["windows_boot_time_utc"],
        "q_authority_root_sha256": ready["q_authority_root_sha256"],
        "q_file_sha256": ready["q_file_sha256"],
        "e_file_sha256": ready["e_file_sha256"],
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
    return {
        **unsigned,
        "receipt_root_sha256": _authority_json_sha256(unsigned),
    }


def _require_q_e_custody_spec_fields(
    canonical_spec: Mapping[str, Any],
    *,
    project_root: Path,
    supervisor_job_directory: Path,
    supervisor_job_id: str,
    q_authority_root_sha256: str,
    q_file_sha256: str,
    e_intent_path: Path,
    e_file_sha256: str,
    independent_verifier_receipt_sha256: str,
) -> dict[str, Any]:
    raw = {field: canonical_spec[field] for field in _Q_E_CUSTODY_SPEC_FIELDS}
    contract = _require_q_e_custody_contract(
        raw["q_e_custody_contract"],
        supervisor_job_directory=supervisor_job_directory,
    )
    ready = _require_q_e_custody_ready(
        raw["q_e_custody_handoff"],
        contract=contract,
        project_root=project_root,
        supervisor_job_directory=supervisor_job_directory,
        supervisor_job_id=supervisor_job_id,
        q_authority_root_sha256=q_authority_root_sha256,
        q_file_sha256=q_file_sha256,
        e_intent_path=e_intent_path,
        e_file_sha256=e_file_sha256,
        independent_verifier_receipt_sha256=(independent_verifier_receipt_sha256),
    )
    receipt = _build_q_e_custody_receipt(
        contract=contract,
        ready=ready,
    )
    expected_binding = {
        "policy": _Q_E_CUSTODY_RECEIPT_POLICY,
        "path": contract["receipt_path"],
        "file_sha256": hashlib.sha256(_authority_canonical_json_line(receipt)).hexdigest(),
        "receipt_root_sha256": receipt["receipt_root_sha256"],
        "handoff_root_sha256": ready["handoff_root_sha256"],
    }
    receipt_binding = _exact_object(
        raw["q_e_custody_receipt"],
        fields=_Q_E_CUSTODY_SPEC_RECEIPT_BINDING_FIELDS,
        label="Q/E downstream spec receipt binding",
    )
    if not _strict_json_value_equal(receipt_binding, expected_binding):
        _stop("Q/E downstream supervisor spec fields violate their exact binding")
    return {
        "q_e_custody_contract": contract,
        "q_e_custody_handoff": ready,
        "q_e_custody_receipt": expected_binding,
    }


def _require_q_e_custody_spec_structure(
    canonical_spec: Mapping[str, Any],
) -> None:
    project_root = _exact_absolute_path(
        canonical_spec["project_root"],
        label="supervisor spec Q/E project root",
    )
    job_id = canonical_spec["job_id"]
    if type(job_id) is not str or _IDENTIFIER.fullmatch(job_id) is None:
        _stop("supervisor spec Q/E job ID is invalid")
    contract = _exact_object(
        canonical_spec["q_e_custody_contract"],
        fields=_Q_E_CUSTODY_CONTRACT_FIELDS,
        label="supervisor spec Q/E custody contract",
    )
    receipt_path = _exact_absolute_path(
        contract["receipt_path"],
        label="supervisor spec Q/E receipt path",
    )
    job_directory = receipt_path.parent
    ready = _exact_object(
        canonical_spec["q_e_custody_handoff"],
        fields=_Q_E_CUSTODY_READY_FIELDS,
        label="supervisor spec Q/E custody handoff",
    )
    e_identity = _exact_object(
        ready["e_leaf_physical_identity"],
        fields=_CONTROL_PUBLICATION_IDENTITY_FIELDS,
        label="supervisor spec structural E leaf identity",
    )
    e_intent_path = _exact_absolute_path(
        e_identity["path"],
        label="supervisor spec structural E path",
    )
    _require_q_e_custody_spec_fields(
        canonical_spec,
        project_root=project_root,
        supervisor_job_directory=job_directory,
        supervisor_job_id=job_id,
        q_authority_root_sha256=_require_sha256(
            ready["q_authority_root_sha256"],
            label="supervisor spec structural Q root",
        ),
        q_file_sha256=_require_sha256(
            ready["q_file_sha256"],
            label="supervisor spec structural Q file SHA-256",
        ),
        e_intent_path=e_intent_path,
        e_file_sha256=_require_sha256(
            ready["e_file_sha256"],
            label="supervisor spec structural E file SHA-256",
        ),
        independent_verifier_receipt_sha256=_require_sha256(
            ready["independent_verifier_receipt_sha256"],
            label="supervisor spec structural independent verifier receipt",
        ),
    )


def _require_q_attempt_identity_projection(
    raw_value: Any,
    *,
    q_base_authority_root_sha256: str,
) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_Q_ATTEMPT_IDENTITY_FIELDS,
        label="Q attempt identity projection",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or type(raw["policy"]) is not str
        or raw["policy"] != _Q_ATTEMPT_IDENTITY_DERIVATION_POLICY
        or type(raw["attempt_id"]) is not str
        or _Q_ATTEMPT_ID.fullmatch(raw["attempt_id"]) is None
        or type(raw["q_base_authority_root_sha256"]) is not str
        or raw["q_base_authority_root_sha256"] != q_base_authority_root_sha256
        or type(raw["execution_mode"]) is not str
        or raw["execution_mode"] != "fresh"
        or raw["retry_of_run_id"] is not None
        or any(
            type(raw[field]) is not str
            for field in (
                "attempt_identity_root_sha256",
                "job_id",
                "run_id",
                "launch_nonce",
            )
        )
    ):
        _stop("Q attempt identity projection violates its exact typed schema")
    root_preimage = {
        "schema_version": 1,
        "policy": _Q_ATTEMPT_IDENTITY_DERIVATION_POLICY,
        "attempt_id": raw["attempt_id"],
        "q_base_authority_root_sha256": q_base_authority_root_sha256,
        "execution_mode": "fresh",
        "retry_of_run_id": None,
    }
    identity_root = _authority_json_sha256(root_preimage)
    nonce_preimage = {
        "schema_version": 1,
        "policy": _Q_ATTEMPT_LAUNCH_NONCE_DERIVATION_POLICY,
        "attempt_identity_root_sha256": identity_root,
    }
    expected = {
        **root_preimage,
        "attempt_identity_root_sha256": identity_root,
        "job_id": f"oc-{identity_root}",
        "run_id": f"original-confirmatory-{identity_root}",
        "launch_nonce": _authority_json_sha256(nonce_preimage),
    }
    if not _strict_json_value_equal(raw, expected):
        _stop("Q attempt identity projection differs from its exact derivation")
    return expected


def _require_control_staging_projection(
    raw_value: Any,
    *,
    job_id: str,
    supervisor_state_root: Path,
    expected_sha256: Any,
) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_CONTROL_STAGING_PROJECTION_FIELDS,
        label="Q control-staging projection",
    )
    path_fields = _CONTROL_STAGING_PROJECTION_FIELDS - {
        "schema_version",
        "policy",
        "exact_file_allowlist",
    }
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 2
        or type(raw["policy"]) is not str
        or raw["policy"] != _CONTROL_STAGING_PROJECTION_POLICY
        or any(type(raw[field]) is not str for field in path_fields)
        or type(raw["exact_file_allowlist"]) is not list
        or any(type(item) is not str for item in raw["exact_file_allowlist"])
    ):
        _stop("Q control-staging projection violates its exact v2 typed schema")
    observed_state_root = _exact_absolute_path(
        raw["supervisor_state_root"],
        label="Q control-staging state root",
    )
    control_staging_dir = supervisor_state_root / _CONTROL_STAGING_DIRECTORY_NAME / job_id
    final_job_dir = supervisor_state_root / _SUPERVISOR_JOBS_DIRECTORY_NAME / job_id
    expected = {
        "schema_version": 2,
        "policy": _CONTROL_STAGING_PROJECTION_POLICY,
        "supervisor_state_root": str(supervisor_state_root),
        "control_staging_dir": str(control_staging_dir),
        "final_job_dir": str(final_job_dir),
        "staging_attempt_path": str(control_staging_dir / "staging_attempt.json"),
        "e_intent_path": str(control_staging_dir / "e_intent.json"),
        "launch_authorization_path": str(control_staging_dir / "launch_authorization.json"),
        "supervisor_launch_spec_path": str(control_staging_dir / "supervisor_launch_spec.json"),
        "staging_ready_path": str(control_staging_dir / "staging_ready.json"),
        "exact_file_allowlist": list(_CONTROL_STAGING_EXACT_FILE_ALLOWLIST),
    }
    projection_sha256 = _require_sha256(
        expected_sha256,
        label="Q control-staging projection",
    )
    if (
        observed_state_root != supervisor_state_root
        or not _strict_json_value_equal(raw, expected)
        or projection_sha256 != _authority_json_sha256(expected)
    ):
        _stop("Q control-staging projection differs from its exact v2 derivation")
    return expected


def _protected_expected_artifact_template_projection() -> dict[str, Any]:
    unsigned = {
        "schema_version": 1,
        "policy": _OUTCOME_BLIND_EXPECTED_ARTIFACT_PROJECTION_POLICY,
        "ordered_role_templates": [
            {
                **template,
                "json_equals": dict(cast(Mapping[str, Any], template["json_equals"])),
            }
            for template in _PROTECTED_EXPECTED_ARTIFACT_TEMPLATE
        ],
        "template_rule_field_names": sorted(_PROTECTED_EXPECTED_ARTIFACT_TEMPLATE_RULE_FIELDS),
        "instance_rule_field_names": sorted(_PROTECTED_EXPECTED_ARTIFACT_RULE_FIELDS),
        "required_success_roles": list(_PROTECTED_EXPECTED_ARTIFACT_ROLES),
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
        "projection_root_sha256": _authority_json_sha256(unsigned),
    }


def _require_protected_expected_artifact_template_projection(
    raw_value: Any,
) -> dict[str, Any]:
    expected = _protected_expected_artifact_template_projection()
    if (
        type(raw_value) is not dict
        or set(raw_value) != set(expected)
        or not _strict_json_value_equal(raw_value, expected)
    ):
        _stop("protected expected-artifact template violates its exact control-only policy")
    return expected


def _protected_expected_artifact_instance(
    *,
    run_id: str,
    expected_run_directory: Path,
) -> dict[str, Any]:
    if (
        type(run_id) is not str
        or _IDENTIFIER.fullmatch(run_id) is None
        or expected_run_directory != expected_run_directory.parent / run_id
    ):
        _stop("protected expected-artifact instance has an invalid run binding")
    template_projection = _protected_expected_artifact_template_projection()
    anchors = {
        "expected_run_directory": expected_run_directory,
        "runs_root": expected_run_directory.parent,
    }
    rules: list[dict[str, Any]] = []
    for template_value in _PROTECTED_EXPECTED_ARTIFACT_TEMPLATE:
        template = dict(template_value)
        anchor_name = cast(str, template["path_anchor"])
        relative_path = Path(cast(str, template["relative_path"]))
        if (
            anchor_name not in anchors
            or relative_path.is_absolute()
            or len(relative_path.parts) != 1
            or relative_path.parts[0] in {"", ".", ".."}
        ):
            _stop("authority-owned protected artifact template has an unsafe path")
        checks = {
            key: run_id if expected == "$RUN_ID" else expected
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
    unsigned = {
        "schema_version": 1,
        "policy": _OUTCOME_BLIND_EXPECTED_ARTIFACT_INSTANCE_POLICY,
        "template_projection_root_sha256": template_projection["projection_root_sha256"],
        "run_id": run_id,
        "expected_run_directory": str(expected_run_directory),
        "runs_root": str(expected_run_directory.parent),
        "required_success_roles": list(_PROTECTED_EXPECTED_ARTIFACT_ROLES),
        "expected_artifacts": rules,
        "expected_artifacts_root_sha256": _authority_json_sha256(rules),
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
        "projection_root_sha256": _authority_json_sha256(unsigned),
    }


def _require_protected_expected_artifact_instance(
    raw_value: Any,
    *,
    run_id: str,
    expected_run_directory: Path,
) -> dict[str, Any]:
    expected = _protected_expected_artifact_instance(
        run_id=run_id,
        expected_run_directory=expected_run_directory,
    )
    if (
        type(raw_value) is not dict
        or set(raw_value) != set(expected)
        or not _strict_json_value_equal(raw_value, expected)
    ):
        _stop("protected expected-artifact instance violates its exact control-only policy")
    return expected


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


def _terminal_client_launcher_projection(
    *,
    launcher_release: Mapping[str, Any],
    job_id: str,
    supervisor_job_directory: Path,
    verify_terminal_command_projection_sha256: str,
    verify_terminal_environment_sha256: str,
    verify_terminal_cwd: Path,
) -> dict[str, Any]:
    spec_path = supervisor_job_directory / "run_spec.json"
    e_intent_path = supervisor_job_directory / "e_intent.json"
    terminal_receipt_path = supervisor_job_directory / "terminal_receipt.json"
    launch_intent_path = supervisor_job_directory / _TERMINAL_CLIENT_LAUNCH_INTENT_FILENAME
    child_projection_sha256 = _require_sha256(
        verify_terminal_command_projection_sha256,
        label="terminal-client child command projection",
    )
    child_environment_sha256 = _require_sha256(
        verify_terminal_environment_sha256,
        label="terminal-client child environment",
    )
    child_cwd = str(verify_terminal_cwd)
    child_cwd_root = _authority_json_sha256({"cwd": child_cwd})
    child_launch_root = _authority_json_sha256(
        {
            "verify_terminal_command_projection_sha256": (child_projection_sha256),
            "verify_terminal_environment_sha256": child_environment_sha256,
            "verify_terminal_cwd": child_cwd,
            "verify_terminal_cwd_root_sha256": child_cwd_root,
        }
    )
    python_sys_argv_template = _terminal_client_launcher_argv_template(
        source_path=cast(str, launcher_release["source_path"]),
        job_id=job_id,
        job_directory=str(supervisor_job_directory),
        supervisor_spec_path=str(spec_path),
        e_intent_path=str(e_intent_path),
        terminal_receipt_path=str(terminal_receipt_path),
        launch_intent_path=str(launch_intent_path),
        verify_terminal_command_projection_sha256=child_projection_sha256,
        verify_terminal_environment_sha256=child_environment_sha256,
        verify_terminal_cwd=child_cwd,
    )
    process_argv_template = [
        launcher_release["program_path"],
        *cast(list[str], launcher_release["python_isolated_flags"]),
        *python_sys_argv_template,
    ]
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "policy": _TERMINAL_CLIENT_LAUNCHER_E_PROJECTION_POLICY,
        "launcher_release_root_sha256": launcher_release["release_root_sha256"],
        "program_path": launcher_release["program_path"],
        "program_sha256": launcher_release["program_sha256"],
        "runtime_python_path": launcher_release["runtime_python_path"],
        "runtime_python_sha256": launcher_release["runtime_python_sha256"],
        "source_path": launcher_release["source_path"],
        "source_size_bytes": launcher_release["source_size_bytes"],
        "source_sha256": launcher_release["source_sha256"],
        "python_isolated_flags": list(_TERMINAL_CLIENT_PYTHON_ISOLATED_FLAGS),
        "job_id": job_id,
        "supervisor_job_directory": str(supervisor_job_directory),
        "supervisor_spec_path": str(spec_path),
        "e_intent_path": str(e_intent_path),
        "terminal_receipt_path": str(terminal_receipt_path),
        "launch_intent_path": str(launch_intent_path),
        "python_sys_argv_template": python_sys_argv_template,
        "process_argv_template": process_argv_template,
        "downstream_placeholder_order": list(_TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDER_ORDER),
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
        "verify_terminal_child_launch_topology": launcher_release[
            "verify_terminal_child_launch_topology"
        ],
        "verify_terminal_immediate_redirector_program_path": launcher_release[
            "verify_terminal_immediate_redirector_program_path"
        ],
        "verify_terminal_immediate_redirector_program_sha256": launcher_release[
            "verify_terminal_immediate_redirector_program_sha256"
        ],
        "verify_terminal_runtime_child_program_path": launcher_release[
            "verify_terminal_runtime_child_program_path"
        ],
        "verify_terminal_runtime_child_program_sha256": launcher_release[
            "verify_terminal_runtime_child_program_sha256"
        ],
        "launcher_cwd": child_cwd,
        "command_preimage_field_names": sorted(_TERMINAL_CLIENT_LAUNCHER_COMMAND_PREIMAGE_FIELDS),
        "command_sha256_policy": _SUPERVISOR_PROCESS_COMMAND_HASH_POLICY,
        "wake_intent_hash_in_launcher_argv_allowed": False,
        "preterminal_pin_terminal_or_lease_input_read_allowed": False,
        "launch_intent_create_new_required": True,
        "same_job_no_breakaway_required": True,
        "automatic_retry_allowed": False,
        "fallback_allowed": False,
    }
    unsigned.update(
        {field: launcher_release[field] for field in _TERMINAL_CLIENT_LAUNCHER_E_STATIC_COPY_FIELDS}
    )
    return {
        **unsigned,
        "projection_root_sha256": _authority_json_sha256(unsigned),
    }


def _require_terminal_client_launcher_projection(
    raw_value: Any,
    *,
    launcher_release: Mapping[str, Any],
    job_id: str,
    supervisor_job_directory: Path,
    verify_terminal_command_projection_sha256: str,
    verify_terminal_environment_sha256: str,
    verify_terminal_cwd: Path,
) -> dict[str, Any]:
    expected = _terminal_client_launcher_projection(
        launcher_release=launcher_release,
        job_id=job_id,
        supervisor_job_directory=supervisor_job_directory,
        verify_terminal_command_projection_sha256=(verify_terminal_command_projection_sha256),
        verify_terminal_environment_sha256=verify_terminal_environment_sha256,
        verify_terminal_cwd=verify_terminal_cwd,
    )
    if (
        type(raw_value) is not dict
        or set(raw_value) != _TERMINAL_CLIENT_LAUNCHER_E_PROJECTION_FIELDS
        or not _strict_json_value_equal(raw_value, expected)
    ):
        _stop("terminal-client launcher E projection violates its exact acyclic policy")
    return expected


def _require_terminal_custody_artifact_authority_projection(
    raw_value: Any,
    *,
    project_root: Path,
    run_id: str,
    expected_run_directory: Path,
    supervisor_release: Mapping[str, Any],
    supervisor_job_id: str,
    supervisor_job_directory: Path,
    verify_terminal_command_projection_sha256: str,
    verify_terminal_environment_sha256: str,
) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_TERMINAL_CUSTODY_AUTHORITY_PROJECTION_FIELDS,
        label="E terminal custody authority projection",
    )
    _require_self_hash(
        raw,
        self_field="projection_root_sha256",
        label="E terminal custody authority projection",
    )
    instance = _require_protected_expected_artifact_instance(
        raw["outcome_blind_expected_artifact_instance"],
        run_id=run_id,
        expected_run_directory=expected_run_directory,
    )
    launcher_release = _exact_object(
        supervisor_release["terminal_client_launcher_release"],
        fields=_TERMINAL_CLIENT_LAUNCHER_RELEASE_FIELDS,
        label="Q terminal-client launcher release",
    )
    launcher_projection = _require_terminal_client_launcher_projection(
        raw["terminal_client_launcher_projection"],
        launcher_release=launcher_release,
        job_id=supervisor_job_id,
        supervisor_job_directory=supervisor_job_directory,
        verify_terminal_command_projection_sha256=(verify_terminal_command_projection_sha256),
        verify_terminal_environment_sha256=verify_terminal_environment_sha256,
        verify_terminal_cwd=project_root,
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _TERMINAL_CUSTODY_AUTHORITY_PROJECTION_POLICY
        or expected_run_directory != project_root / "artifacts" / "runs" / run_id
        or raw["terminal_custody_authority_template_root_sha256"]
        != supervisor_release["terminal_custody_authority_template_root_sha256"]
        or raw["terminal_custody_authority_template_root_sha256"]
        != _TERMINAL_CUSTODY_AUTHORITY_TEMPLATE_ROOT_SHA256
        or instance["template_projection_root_sha256"]
        != _protected_expected_artifact_template_projection()["projection_root_sha256"]
        or raw["terminal_client_launcher_projection_root_sha256"]
        != launcher_projection["projection_root_sha256"]
    ):
        _stop("E terminal custody authority projection violates its pre-import control-only policy")
    return instance


def _require_protected_expected_artifacts(
    raw_value: Any,
    *,
    project_root: Path,
    run_directory: Path,
    run_id: str,
) -> list[dict[str, Any]]:
    if type(raw_value) is not list or run_directory != project_root / "artifacts" / "runs" / run_id:
        _stop("protected expected-artifact projection violates its exact run binding")
    instance = _protected_expected_artifact_instance(
        run_id=run_id,
        expected_run_directory=run_directory,
    )
    expected = cast(list[dict[str, Any]], instance["expected_artifacts"])
    if len(raw_value) != len(expected):
        _stop("protected expected-artifact roles/order differ")
    records: list[dict[str, Any]] = []
    for index, (raw_rule, expected_rule) in enumerate(zip(raw_value, expected, strict=True)):
        raw = _exact_object(
            raw_rule,
            fields=_PROTECTED_EXPECTED_ARTIFACT_RULE_FIELDS,
            label=f"protected expected artifact {index}",
        )
        if not _strict_json_value_equal(raw, expected_rule):
            _stop(f"protected expected artifact {index} differs from its exact control-only rule")
        records.append(dict(raw))
    if not _strict_json_value_equal(
        [record["role"] for record in records],
        list(_PROTECTED_EXPECTED_ARTIFACT_ROLES),
    ):
        _stop("protected expected-artifact roles/order differ")
    return records


def _require_control_staging_physical_identity(
    raw_value: Any,
    *,
    held: _HeldAuthorityFile,
    expected_role: str,
) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_PHYSICAL_FILE_IDENTITY_FIELDS,
        label="control-staging physical identity",
    )
    observed = os.lstat(held.path)
    volume, file_id = _native_path_identity(held.path, directory=False)
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _NO_FOLLOW_PHYSICAL_FILE_IDENTITY_POLICY
        or raw["role"] != expected_role
        or raw["path"] != str(held.path)
        or type(raw["volume_serial_number"]) is not int
        or raw["volume_serial_number"] != volume
        or raw["file_id_128"] != file_id
        or raw["device"] != int(observed.st_dev)
        or raw["inode"] != int(observed.st_ino)
        or type(raw["size_bytes"]) is not int
        or raw["size_bytes"] != held.size_bytes
        or raw["mode"] != stat.S_IMODE(observed.st_mode)
        or type(raw["file_attributes"]) is not int
        or raw["file_attributes"] != _file_attributes(observed)
        or raw["regular_file"] is not True
        or raw["read_only"] is not True
        or type(raw["link_count"]) is not int
        or raw["link_count"] != 1
        or raw["modified_time_ns"] != int(observed.st_mtime_ns)
        or raw["changed_time_ns"] != int(observed.st_ctime_ns)
        or raw["sha256"] != held.sha256
        or not _strict_json_value_equal(raw["named_alternate_data_streams"], [])
        or raw["opened_without_reparse_follow"] is not True
        or not _strict_json_value_equal(raw["share_access"], ["FILE_SHARE_READ"])
    ):
        _stop(f"control-staging {expected_role} physical identity differs from retained file")
    return raw


def _require_control_staging_outer_binding(
    raw_value: Any,
    *,
    payload: Mapping[str, Any],
    canonical_spec: Mapping[str, Any],
    supervisor_release: Mapping[str, Any],
    control_staging_projection: Mapping[str, Any],
    e_intent_path: Path,
    e_file_sha256: str,
    e_held_file: _HeldAuthorityFile,
) -> tuple[dict[str, Any], tuple[_HeldAuthorityFile, ...]]:
    raw = _exact_object(
        raw_value,
        fields=_CONTROL_STAGING_OUTER_BINDING_FIELDS,
        label="supervisor control-staging outer binding",
    )
    files_value = raw["files"]
    if type(files_value) is not list or len(files_value) != len(
        _CONTROL_STAGING_EXACT_FILE_ALLOWLIST
    ):
        _stop("supervisor control-staging outer file inventory differs")
    files: list[dict[str, Any]] = []
    staging_dir = Path(cast(str, control_staging_projection["control_staging_dir"]))
    supervisor_root = Path(cast(str, supervisor_release["supervisor_state_root"]))
    for index, item in enumerate(files_value):
        record = _exact_object(
            item,
            fields=_CONTROL_STAGING_OUTER_FILE_FIELDS,
            label="supervisor control-staging outer file",
        )
        identity = record["physical_identity"]
        if (
            type(identity) is not dict
            or record["name"] != _CONTROL_STAGING_EXACT_FILE_ALLOWLIST[index]
            or record["role"] != _CONTROL_STAGING_OUTER_FILE_ROLES[index]
            or record["path"] != str(staging_dir / cast(str, record["name"]))
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] <= 0
            or _require_sha256(
                record["file_sha256"],
                label="supervisor control-staging outer file",
            )
            != record["file_sha256"]
            or _require_sha256(
                record["physical_identity_root_sha256"],
                label="supervisor control-staging physical identity",
            )
            != _authority_json_sha256(identity)
        ):
            _stop("supervisor control-staging outer file binding differs")
        files.append(record)
    unsigned = {key: item for key, item in raw.items() if key != "binding_root_sha256"}
    expected_leaf_names = list(_CONTROL_STAGING_EXACT_FILE_ALLOWLIST)
    expected_pre_ack_scope = [
        "jobs/<job_id>",
        "run_spec.json",
        "q_e_custody_receipt.json",
    ]
    source_record = files[3]
    e_record = files[1]
    authorization_record = files[2]
    ancestor_lease = _require_control_staging_ancestor_lease(
        raw["control_staging_ancestor_lease"],
        supervisor_root=supervisor_root,
        staging_directory=staging_dir,
    )
    authorization = _exact_object(
        canonical_spec["authorization"],
        fields={"path", "sha256"},
        label="supervisor canonical authorization",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _CONTROL_STAGING_OUTER_BINDING_POLICY
        or raw["job_id"] != canonical_spec["job_id"]
        or raw["supervisor_root"] != str(supervisor_root)
        or raw["control_staging_root"] != str(supervisor_root / _CONTROL_STAGING_DIRECTORY_NAME)
        or raw["control_staging_dir"] != str(staging_dir)
        or not _strict_json_value_equal(
            raw["control_staging_projection"], control_staging_projection
        )
        or raw["control_staging_projection_sha256"]
        != _authority_json_sha256(control_staging_projection)
        or raw["expected_complete_leaf_names"] != expected_leaf_names
        or raw["publication_order"] != expected_leaf_names
        or raw["file_count"] != 5
        or type(raw["file_count"]) is not int
        or _require_sha256(
            raw["control_staging_ancestor_lease_root_sha256"],
            label="control-staging ancestor lease",
        )
        != _authority_json_sha256(ancestor_lease)
        or _require_sha256(
            raw["staging_attempt_root_sha256"],
            label="control-staging attempt root",
        )
        != raw["staging_attempt_root_sha256"]
        or _require_sha256(
            raw["staging_ready_root_sha256"],
            label="control-staging ready root",
        )
        != raw["staging_ready_root_sha256"]
        or raw["source_path"] != control_staging_projection["supervisor_launch_spec_path"]
        or raw["source_path"] != source_record["path"]
        or type(raw["source_size_bytes"]) is not int
        or raw["source_size_bytes"] != source_record["size_bytes"]
        or raw["source_file_sha256"] != source_record["file_sha256"]
        or raw["source_canonical_bytes_sha256"] != source_record["file_sha256"]
        or raw["source_bytes_equal_canonical_spec_serialization"] is not True
        or raw["e_intent_path"] != str(e_intent_path)
        or raw["e_intent_path"] != e_record["path"]
        or raw["e_intent_file_sha256"] != e_file_sha256
        or raw["e_intent_file_sha256"] != e_record["file_sha256"]
        or raw["launch_authorization_path"]
        != control_staging_projection["launch_authorization_path"]
        or raw["launch_authorization_path"] != authorization_record["path"]
        or raw["launch_authorization_file_sha256"] != authorization_record["file_sha256"]
        or authorization["path"] != authorization_record["path"]
        or authorization["sha256"] != authorization_record["file_sha256"]
        or type(raw["supervisor_process_identity"]) is not dict
        or raw["retained_from_before_final_job_creation_through_terminal"] is not True
        or raw["final_job_creation_owner"] != "suspended_supervisor_after_resume_v1"
        or raw["pre_ack_final_job_publication_scope"] != expected_pre_ack_scope
        or raw["pre_ack_metadata_only_publication_allowed"] is not True
        or raw["pre_ack_scientific_process_launch_allowed"] is not False
        or raw["q_e_ack_required_before_scientific_process_launch"] is not True
        or raw["automatic_retry_allowed"] is not False
        or raw["adoption_allowed"] is not False
        or raw["cleanup_allowed"] is not False
        or raw["binding_root_sha256"] != _authority_json_sha256(unsigned)
        or payload["source_path"] != raw["source_path"]
        or payload["source_file_sha256"] != raw["source_file_sha256"]
    ):
        _stop("supervisor control-staging outer binding violates exact custody")
    retained: list[_HeldAuthorityFile] = []
    retained_by_name: dict[str, _HeldAuthorityFile] = {}
    try:
        for index, record in enumerate(files):
            if index == 1:
                continue
            held = _open_held_plain_file(
                Path(cast(str, record["path"])),
                label=f"control-staging retained {record['role']}",
                require_read_only=True,
                maximum_bytes=_MAX_CONTROL_BYTES,
            )
            retained.append(held)
            retained_by_name[cast(str, record["name"])] = held
            if held.sha256 != record["file_sha256"] or held.size_bytes != record["size_bytes"]:
                _stop("control-staging retained file differs from outer binding")
            _require_control_staging_physical_identity(
                record["physical_identity"],
                held=held,
                expected_role=cast(str, record["role"]),
            )
        if (
            e_record["file_sha256"] != e_file_sha256
            or e_held_file.path != e_intent_path
            or e_held_file.sha256 != e_file_sha256
            or e_held_file.size_bytes != e_record["size_bytes"]
        ):
            _stop("control-staging E differs from the retained preimport E")
        _require_control_staging_physical_identity(
            e_record["physical_identity"],
            held=e_held_file,
            expected_role=cast(str, e_record["role"]),
        )
        attempt = _parse_authority_json_line(
            retained_by_name["staging_attempt.json"].payload,
            label="control-staging attempt marker",
        )
        ready = _parse_authority_json_line(
            retained_by_name["staging_ready.json"].payload,
            label="control-staging ready marker",
        )
        for document, root_field, expected_root, label in (
            (
                attempt,
                "attempt_marker_root_sha256",
                raw["staging_attempt_root_sha256"],
                "control-staging attempt marker",
            ),
            (
                ready,
                "ready_marker_root_sha256",
                raw["staging_ready_root_sha256"],
                "control-staging ready marker",
            ),
        ):
            if type(document) is not dict or root_field not in document:
                _stop(f"{label} lacks its exact self-root")
            unsigned_document = {key: item for key, item in document.items() if key != root_field}
            if _require_sha256(document[root_field], label=label) != expected_root or document[
                root_field
            ] != _authority_json_sha256(unsigned_document):
                _stop(f"{label} self-root differs from retained contents")
        if not _strict_json_value_equal(
            ready.get("supervisor_process_identity"),
            raw["supervisor_process_identity"],
        ):
            _stop("control-staging supervisor process identity differs from READY")
    except BaseException:
        for held in retained:
            os.close(held.descriptor)
        raise
    return raw, tuple(retained)


def _parse_held_supervisor_run_spec(
    held: _HeldAuthorityFile,
    *,
    expected_path: Path,
    supervisor_release: Mapping[str, Any],
    control_staging_projection: Mapping[str, Any],
    e_intent_path: Path,
    e_file_sha256: str,
    e_held_file: _HeldAuthorityFile,
) -> tuple[dict[str, Any], tuple[_HeldAuthorityFile, ...]]:
    if held.path != expected_path:
        _stop("held supervisor run spec differs from E's exact path")
    envelope = _parse_authority_json_line(
        held.payload,
        label="supervisor run spec envelope",
    )
    _exact_object(
        envelope,
        fields=_SUPERVISOR_RUN_SPEC_ENVELOPE_FIELDS,
        label="supervisor run spec envelope",
    )
    payload = _exact_object(
        envelope["payload"],
        fields=_SUPERVISOR_RUN_SPEC_PAYLOAD_FIELDS,
        label="supervisor run spec payload",
    )
    supervisor = _exact_object(
        payload["supervisor"],
        fields=_SUPERVISOR_RUN_SPEC_IDENTITY_FIELDS,
        label="supervisor run spec source identity",
    )
    canonical_spec = payload["canonical_spec"]
    source_path = _exact_absolute_path(
        payload["source_path"],
        label="supervisor source spec path",
    )
    supervisor_path = _exact_absolute_path(
        supervisor["path"],
        label="supervisor source path",
    )
    payload_sha256 = hashlib.sha256(_authority_canonical_json_line(payload)).hexdigest()
    canonical_spec_sha256 = (
        hashlib.sha256(_authority_canonical_json_line(canonical_spec)).hexdigest()
        if type(canonical_spec) is dict
        else None
    )
    if (
        type(envelope["schema_version"]) is not int
        or envelope["schema_version"] != 3
        or envelope["payload_sha256"] != payload_sha256
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 3
        or payload["policy"] != _SUPERVISOR_POLICY
        or source_path == expected_path
        or _require_sha256(
            payload["source_file_sha256"],
            label="supervisor source spec SHA-256",
        )
        != payload["source_file_sha256"]
        or type(canonical_spec) is not dict
        or payload["canonical_spec_sha256"] != canonical_spec_sha256
        or _require_sha256(
            supervisor["sha256"],
            label="supervisor run spec source SHA-256",
        )
        != supervisor_release["supervisor_source_sha256"]
        or supervisor_path != Path(cast(str, supervisor_release["supervisor_source_path"]))
        or type(payload["frozen_at_utc"]) is not str
        or not payload["frozen_at_utc"].endswith("Z")
        or "\x00" in payload["frozen_at_utc"]
    ):
        _stop("supervisor run spec envelope violates its exact frozen binding")
    canonical = _exact_object(
        canonical_spec,
        fields=_SUPERVISOR_CANONICAL_SPEC_FIELDS,
        label="supervisor canonical spec",
    )
    _require_q_e_custody_spec_structure(canonical)
    _binding, retained_staging_files = _require_control_staging_outer_binding(
        payload["control_staging"],
        payload=payload,
        canonical_spec=canonical,
        supervisor_release=supervisor_release,
        control_staging_projection=control_staging_projection,
        e_intent_path=e_intent_path,
        e_file_sha256=e_file_sha256,
        e_held_file=e_held_file,
    )
    return canonical, retained_staging_files


def _require_supervisor_spec_protected_artifact_projection(
    canonical_spec: Mapping[str, Any],
    *,
    project_root: Path,
    run_id: str,
    expected_run_directory: Path,
    e_terminal_custody_projection: Mapping[str, Any],
) -> None:
    _require_protected_expected_artifacts(
        canonical_spec.get("expected_artifacts"),
        project_root=project_root,
        run_directory=expected_run_directory,
        run_id=run_id,
    )
    instance = _protected_expected_artifact_instance(
        run_id=run_id,
        expected_run_directory=expected_run_directory,
    )
    if not _strict_json_value_equal(
        canonical_spec.get("required_success_roles"),
        instance["required_success_roles"],
    ):
        _stop("supervisor spec required success roles differ from sealed E")
    terminal_contract = canonical_spec.get("terminal_composition_contract")
    if type(terminal_contract) is not dict:
        _stop("supervisor spec lacks its terminal composition contract")
    spec_terminal_projection = terminal_contract.get("terminal_custody_authority_projection")
    if not _strict_json_value_equal(
        spec_terminal_projection,
        e_terminal_custody_projection,
    ):
        _stop("supervisor spec terminal custody projection differs from sealed E")
    non_artifact_terminal = {
        key: item
        for key, item in terminal_contract.items()
        if key != "terminal_custody_authority_projection"
    }
    non_artifact_spec = {
        key: (non_artifact_terminal if key == "terminal_composition_contract" else item)
        for key, item in canonical_spec.items()
        if key != "expected_artifacts"
    }
    if _contains_mapping_key(non_artifact_spec, "json_equals"):
        _stop("supervisor spec contains a non-authority expected-artifact selector")


def _expected_concrete_capsule_command(
    argv: Sequence[str],
    *,
    execution_capsule: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    vector = list(argv)
    unsigned: dict[str, Any] = {
        "program_path": execution_capsule["python_path"],
        "program_sha256": execution_capsule["python_sha256"],
        "argv": vector,
        "cwd": str(project_root),
    }
    return {
        **unsigned,
        "command_sha256": _authority_json_sha256(unsigned),
    }


def _derive_terminal_projection_from_spec_command(
    raw_command: Mapping[str, Any],
    *,
    execution_capsule: Mapping[str, Any],
) -> dict[str, Any]:
    command = _exact_object(
        raw_command,
        fields=_CAPSULE_COMMAND_FIELDS,
        label="supervisor spec verify-terminal command",
    )
    argv = command["argv"]
    if (
        not isinstance(argv, list)
        or len(argv) < 12
        or any(type(item) is not str or not item or "\x00" in item for item in argv)
        or not _strict_json_value_equal(
            argv[:5],
            [
                execution_capsule["python_path"],
                "-I",
                "-B",
                execution_capsule["path"],
                TERMINAL_MODE,
            ],
        )
        or argv[5] != "--e-intent"
        or argv[7] != "--e-intent-sha256"
        or argv[9] != "--e-intent-core-sha256"
    ):
        _stop("supervisor spec verify-terminal command has an invalid exact tail")
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "policy": _COMMAND_PROJECTION_POLICY,
        "capsule_mode": TERMINAL_MODE,
        "program_path": command["program_path"],
        "program_sha256": command["program_sha256"],
        "python_isolated_flags": ["-I", "-B"],
        "capsule_path": execution_capsule["path"],
        "capsule_sha256": execution_capsule["sha256"],
        "cwd": command["cwd"],
        "argv_prefix": argv[:5],
        "tail_argv_before_e_file_sha256": argv[5:8],
        "tail_argv_between_e_hashes": argv[9:10],
        "tail_argv_after_e_core_sha256": argv[11:],
        "e_file_sha256_insertion_policy": _E_FILE_INSERTION_POLICY,
        "e_core_sha256_insertion_policy": _E_CORE_INSERTION_POLICY,
    }
    return {
        **unsigned,
        "projection_sha256": _authority_json_sha256(unsigned),
    }


def _require_supervisor_spec_exact_q_e_parity(
    canonical_spec: Mapping[str, Any],
    *,
    project_root: Path,
    e_job: Mapping[str, Any],
    e_lineage: Mapping[str, Any],
    e_intent_path: Path,
    e_file_sha256: str,
    e_core_sha256: str,
    q_root_sha256: str,
    q_file_sha256: str,
    independent_verifier_receipt_sha256: str,
    execution_capsule: Mapping[str, Any],
    supervisor_release: Mapping[str, Any],
    e_consumption_contract: Mapping[str, Any],
    expected_launch_environment: Mapping[str, Any],
    process_environment_binding: Mapping[str, Any],
    e_terminal_custody_projection: Mapping[str, Any],
    e_command_projections: Mapping[str, Any],
    final_by_mode: Mapping[str, tuple[str, ...]],
    control_staging: Mapping[str, Any],
    attempt_creation: Mapping[str, Any],
) -> None:
    _require_q_e_custody_spec_fields(
        canonical_spec,
        project_root=project_root,
        supervisor_job_directory=Path(cast(str, e_job["supervisor_job_dir"])),
        supervisor_job_id=cast(str, e_job["job_id"]),
        q_authority_root_sha256=q_root_sha256,
        q_file_sha256=q_file_sha256,
        e_intent_path=e_intent_path,
        e_file_sha256=e_file_sha256,
        independent_verifier_receipt_sha256=(independent_verifier_receipt_sha256),
    )
    expected_commands = {
        mode: _expected_concrete_capsule_command(
            final_by_mode[mode],
            execution_capsule=execution_capsule,
            project_root=project_root,
        )
        for mode in ALLOWED_MODES
    }
    main_command = expected_commands[SCIENTIFIC_MODE]
    preterminal_command = expected_commands[PRETERMINAL_MODE]
    terminal_command = expected_commands[TERMINAL_MODE]
    preterminal_spec_command = {
        key: item for key, item in preterminal_command.items() if key != "command_sha256"
    }
    _require_external_codex_handoff(
        canonical_spec["external_codex_handoff"],
        control_staging=control_staging,
        e_file_sha256=e_file_sha256,
        e_core_sha256=e_core_sha256,
        attempt_creation=attempt_creation,
    )
    if (
        type(canonical_spec["schema_version"]) is not int
        or canonical_spec["schema_version"] != 3
        or canonical_spec["policy"] != _SUPERVISOR_POLICY
        or canonical_spec["job_id"] != e_job["job_id"]
        or canonical_spec["process_kind"] != "confirmatory"
        or canonical_spec["external_control_plane_release_root_sha256"]
        != supervisor_release["external_control_plane_release_root_sha256"]
        or canonical_spec["external_control_plane_publication_id"]
        != supervisor_release["external_control_plane_publication_id"]
        or canonical_spec["external_control_plane_release_qualification_attestation_path"]
        != supervisor_release["external_control_plane_release_qualification_attestation_path"]
        or canonical_spec["external_control_plane_release_qualification_attestation_file_sha256"]
        != supervisor_release[
            "external_control_plane_release_qualification_attestation_file_sha256"
        ]
        or canonical_spec["external_control_plane_release_qualification_attestation_root_sha256"]
        != supervisor_release[
            "external_control_plane_release_qualification_attestation_root_sha256"
        ]
        or canonical_spec["supervisor_code_root"] != supervisor_release["supervisor_code_root"]
        or canonical_spec["supervisor_state_root"] != supervisor_release["supervisor_state_root"]
        or canonical_spec["project_root"] != str(project_root)
        or canonical_spec["program_path"] != execution_capsule["python_path"]
        or canonical_spec["program_sha256"] != execution_capsule["python_sha256"]
        or not _strict_json_value_equal(
            canonical_spec["argv"],
            main_command["argv"],
        )
        or not _strict_json_value_equal(canonical_spec["command"], main_command)
        or not _strict_json_value_equal(
            canonical_spec["integrity_verifier"],
            preterminal_spec_command,
        )
        or canonical_spec["codex"] is not None
        or canonical_spec["handoff_session"] is not None
        or not _strict_json_value_equal(
            canonical_spec["expected_environment"],
            expected_launch_environment,
        )
        or not _strict_json_value_equal(
            canonical_spec["process_environment_binding"],
            process_environment_binding,
        )
        or not _strict_json_value_equal(
            canonical_spec["e_consumption_contract"],
            e_consumption_contract,
        )
        or canonical_spec["supervisor_launcher_sha256"]
        != supervisor_release["supervisor_launcher_sha256"]
        or not _strict_json_value_equal(
            canonical_spec["capsule_lease_identity"],
            execution_capsule["capsule_lease_identity"],
        )
        or canonical_spec["capsule_lease_identity_root_sha256"]
        != execution_capsule["capsule_lease_identity_root_sha256"]
        or not _strict_json_value_equal(
            canonical_spec["capsule_ancestor_lease"],
            execution_capsule["capsule_ancestor_lease"],
        )
        or canonical_spec["capsule_ancestor_lease_root_sha256"]
        != execution_capsule["capsule_ancestor_lease_root_sha256"]
        or not _strict_json_value_equal(
            canonical_spec["python_lease_identity"],
            execution_capsule["python_lease_identity"],
        )
        or canonical_spec["python_lease_identity_root_sha256"]
        != execution_capsule["python_lease_identity_root_sha256"]
        or not _strict_json_value_equal(
            canonical_spec["python_ancestor_lease"],
            execution_capsule["python_ancestor_lease"],
        )
        or canonical_spec["python_ancestor_lease_root_sha256"]
        != execution_capsule["python_ancestor_lease_root_sha256"]
        or canonical_spec["python_runtime_resolution_policy"]
        != execution_capsule["python_runtime_resolution_policy"]
        or canonical_spec["runtime_python_path"] != execution_capsule["runtime_python_path"]
        or canonical_spec["runtime_python_sha256"] != execution_capsule["runtime_python_sha256"]
        or not _strict_json_value_equal(
            canonical_spec["runtime_python_lease_identity"],
            execution_capsule["runtime_python_lease_identity"],
        )
        or canonical_spec["runtime_python_lease_identity_root_sha256"]
        != execution_capsule["runtime_python_lease_identity_root_sha256"]
        or not _strict_json_value_equal(
            canonical_spec["runtime_python_ancestor_lease"],
            execution_capsule["runtime_python_ancestor_lease"],
        )
        or canonical_spec["runtime_python_ancestor_lease_root_sha256"]
        != execution_capsule["runtime_python_ancestor_lease_root_sha256"]
        or type(canonical_spec["max_attempt_count"]) is not int
        or canonical_spec["max_attempt_count"] != 1
        or canonical_spec["automatic_retry_allowed"] is not False
    ):
        _stop("supervisor spec differs from sealed Q/E command or runtime authority")

    terminal_contract = _exact_object(
        canonical_spec["terminal_composition_contract"],
        fields=_TERMINAL_COMPOSITION_CONTRACT_FIELDS,
        label="supervisor terminal composition contract",
    )
    _require_self_hash(
        terminal_contract,
        self_field="contract_sha256",
        label="supervisor terminal composition contract",
    )
    derived_terminal_projection = _derive_terminal_projection_from_spec_command(
        cast(Mapping[str, Any], terminal_contract["verifier_command"]),
        execution_capsule=execution_capsule,
    )
    sealed_terminal_projection = _exact_object(
        e_command_projections[TERMINAL_MODE],
        fields=_COMMAND_PROJECTION_FIELDS,
        label="sealed E verify-terminal projection",
    )
    launcher_projection = _exact_object(
        e_terminal_custody_projection["terminal_client_launcher_projection"],
        fields=_TERMINAL_CLIENT_LAUNCHER_E_PROJECTION_FIELDS,
        label="sealed E terminal-client launcher projection",
    )
    if (
        not _strict_json_value_equal(
            terminal_contract["verifier_command"],
            terminal_command,
        )
        or terminal_contract["verifier_command_sha256"] != terminal_command["command_sha256"]
        or not _strict_json_value_equal(
            derived_terminal_projection,
            sealed_terminal_projection,
        )
        or derived_terminal_projection["projection_sha256"]
        != launcher_projection["verify_terminal_command_projection_sha256"]
        or not _strict_json_value_equal(
            terminal_contract["terminal_custody_authority_projection"],
            e_terminal_custody_projection,
        )
        or terminal_contract["capsule_contract_sha256"] != execution_capsule["contract_sha256"]
        or terminal_contract["capsule_path"] != execution_capsule["path"]
        or terminal_contract["capsule_sha256"] != execution_capsule["sha256"]
        or terminal_contract["capsule_internal_manifest_sha256"]
        != execution_capsule["internal_manifest_sha256"]
        or terminal_contract["capsule_mode"] != TERMINAL_MODE
        or terminal_contract["expected_environment_envelope_sha256"]
        != expected_launch_environment.get("envelope_sha256")
        or terminal_contract["process_environment_binding_sha256"]
        != process_environment_binding.get("binding_sha256")
        or terminal_contract["exact_integrity_verifier_environment_sha256"]
        != process_environment_binding.get("exact_integrity_verifier_environment_sha256")
        or terminal_contract["capsule_lease_identity_root_sha256"]
        != execution_capsule["capsule_lease_identity_root_sha256"]
        or terminal_contract["capsule_ancestor_lease_root_sha256"]
        != execution_capsule["capsule_ancestor_lease_root_sha256"]
        or terminal_contract["semantic_outcome_read_scope"] != _SEMANTIC_OUTCOME_READ_SCOPE
        or terminal_contract["outcome_values_read"] is not False
        or terminal_contract["outcome_values_emitted"] is not False
        or terminal_contract["outcome_values_used_for_selection_or_tuning"] is not False
        or terminal_contract["training_or_model_selection_allowed"] is not False
        or terminal_contract["scientific_publication_allowed"] is not False
        or terminal_contract["automatic_retry_allowed"] is not False
    ):
        _stop("supervisor spec verify-terminal command/projection differs from sealed E")

    job_directory = Path(cast(str, e_job["supervisor_job_dir"]))
    seed_unsigned: dict[str, Any] = {
        "schema_version": 1,
        "policy": _POSTWAKE_CUSTODY_SEED_POLICY,
        "q_authority_root_sha256": q_root_sha256,
        "e_intent_path": str(e_intent_path),
        "e_intent_file_sha256": e_file_sha256,
        "e_intent_core_sha256": e_core_sha256,
        "supervisor_job_id": e_job["job_id"],
        "supervisor_job_dir": str(job_directory),
        "supervisor_spec_path": e_job["supervisor_spec_path"],
        "launch_nonce": e_job["launch_nonce"],
        "attempt_id": e_job["attempt_id"],
        "run_id": e_job["run_id"],
        "execution_mode": e_lineage["execution_mode"],
        "retry_of_run_id": e_lineage["retry_of_run_id"],
        "execution_capsule_contract_sha256": execution_capsule["contract_sha256"],
        "capsule_sha256": execution_capsule["sha256"],
        "supervisor_release_root_sha256": supervisor_release["supervisor_release_root_sha256"],
        "terminal_release_root_sha256": execution_capsule["terminal_release_root_sha256"],
        "supervisor_terminal_receipt_path": str(job_directory / "terminal_receipt.json"),
        "preterminal_pin_receipt_path": str(job_directory / "preterminal_pin.json"),
        "postwake_input_lease_receipt_path": str(
            job_directory / "postwake_input_lease_receipt.json"
        ),
        "composed_terminal_receipt_path": str(job_directory / "composed_terminal.json"),
        "postwake_composed_readback_receipt_path": str(
            job_directory / "postwake_composed_readback_receipt.json"
        ),
    }
    expected_seed = {
        **seed_unsigned,
        "seed_sha256": _ascii_json_sha256(seed_unsigned),
    }
    if not _strict_json_value_equal(
        canonical_spec["postwake_custody_seed"],
        expected_seed,
    ):
        _stop("supervisor spec postwake custody seed differs from final sealed Q/E")

    handshake = canonical_spec["postwake_custody_handshake_contract"]
    if not isinstance(handshake, dict):
        _stop("supervisor spec lacks its postwake custody handshake contract")
    _require_self_hash(
        handshake,
        self_field="contract_sha256",
        label="supervisor postwake custody handshake contract",
    )
    seed_sha256 = expected_seed["seed_sha256"]
    terminal_environment_sha256 = _require_sha256(
        process_environment_binding.get("exact_integrity_verifier_environment_sha256"),
        label="supervisor spec terminal environment SHA-256",
    )
    if (
        handshake.get("schema_version") != 1
        or type(handshake.get("schema_version")) is not int
        or handshake.get("policy") != _POSTWAKE_CUSTODY_HANDSHAKE_CONTRACT_POLICY
        or handshake.get("supervisor_job_id") != e_job["job_id"]
        or handshake.get("postwake_custody_seed_sha256") != seed_sha256
        or handshake.get("pipe_name") != "\\\\.\\pipe\\AANCA-composed-custody-" + seed_sha256
        or handshake.get("expected_composed_command_sha256") != terminal_command["command_sha256"]
        or handshake.get("expected_composed_cwd") != str(project_root)
        or handshake.get("expected_composed_environment_sha256") != terminal_environment_sha256
        or handshake.get("readback_receipt_path")
        != str(job_directory / "postwake_composed_readback_receipt.json")
        or handshake.get("ready_max_bytes") != 64 * 1024
        or type(handshake.get("ready_max_bytes")) is not int
        or handshake.get("ack_max_bytes") != 64 * 1024
        or type(handshake.get("ack_max_bytes")) is not int
        or handshake.get("terminal_client_arrival_timeout_ms")
        != _TERMINAL_CLIENT_ARRIVAL_TIMEOUT_MS
        or type(handshake.get("terminal_client_arrival_timeout_ms")) is not int
        or handshake.get("custody_exchange_timeout_ms") != _CUSTODY_EXCHANGE_TIMEOUT_MS
        or type(handshake.get("custody_exchange_timeout_ms")) is not int
        or handshake.get("overall_timeout_max_ms") != _TERMINAL_CUSTODY_OVERALL_TIMEOUT_MAX_MS
        or type(handshake.get("overall_timeout_max_ms")) is not int
        or handshake.get("arrival_and_exchange_waits_event_driven") is not True
        or handshake.get("automatic_retry_allowed") is not False
        or terminal_contract["postwake_custody_seed_sha256"] != seed_sha256
        or terminal_contract["postwake_custody_handshake_contract_sha256"]
        != handshake.get("contract_sha256")
    ):
        _stop("supervisor spec postwake custody handshake differs from sealed Q/E")


def _require_e_consumption_contract(
    raw_value: Any,
    *,
    supervisor_job_directory: Path,
) -> dict[str, Any]:
    raw = _exact_object(
        raw_value,
        fields=_E_CONSUMPTION_FIELDS,
        label="E consumption contract",
    )
    _require_self_hash(
        raw,
        self_field="contract_sha256",
        label="E consumption contract",
    )
    claim_path = _exact_absolute_path(
        raw["claim_path"],
        label="E consumption claim path",
    )
    custody_receipt_path = _exact_absolute_path(
        raw["custody_receipt_path"],
        label="E consumption custody receipt path",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _E_CONSUMPTION_POLICY
        or raw["claim_policy"] != _E_CONSUMPTION_CLAIM_POLICY
        or claim_path != supervisor_job_directory / "e_intent_consumed.json"
        or custody_receipt_path
        != supervisor_job_directory / "e_intent_consumed_custody_receipt.json"
        or raw["transport"] != _E_CONSUMPTION_TRANSPORT
        or raw["ready_message_type"] != _E_CONSUMPTION_READY_MESSAGE_TYPE
        or raw["ack_message_type"] != _E_CONSUMPTION_ACK_MESSAGE_TYPE
        or type(raw["ready_line_max_bytes"]) is not int
        or raw["ready_line_max_bytes"] != _E_CONSUMPTION_LINE_MAX_BYTES
        or type(raw["ack_line_max_bytes"]) is not int
        or raw["ack_line_max_bytes"] != _E_CONSUMPTION_LINE_MAX_BYTES
        or type(raw["duplicate_target_access_mask"]) is not int
        or raw["duplicate_target_access_mask"] != _E_CONSUMPTION_DUPLICATE_TARGET_ACCESS_MASK
        or type(raw["duplicate_options"]) is not int
        or raw["duplicate_options"] != _E_CONSUMPTION_DUPLICATE_OPTIONS
        or raw["close_source"] is not False
        or raw["source_handle_retained_through_ack"] is not True
        or raw["supervisor_handle_retention_policy"] != _E_CONSUMPTION_SUPERVISOR_RETENTION_POLICY
        or raw["exact_job_object_membership_required"] is not True
        or raw["exact_supervisor_process_identity_required"] is not True
        or raw["exact_downstream_spec_rederivation_required"] is not True
        or raw["scientific_inputs_before_ack_allowed"] is not False
        or raw["automatic_retry_allowed"] is not False
    ):
        _stop("E consumption contract violates its exact fail-closed policy")
    return raw


def _require_held_capsule_unchanged(
    archive_path: Path,
    descriptor: int,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    expected_identity: tuple[int, ...],
) -> None:
    digest, size_bytes, identity = _require_content_addressed_held_capsule(
        archive_path,
        descriptor,
    )
    if (
        digest != expected_sha256
        or size_bytes != expected_size_bytes
        or identity != expected_identity
    ):
        _stop("held capsule changed after initial verification")


class _BootstrapTail(NamedTuple):
    mode: str
    e_intent_path: Path
    e_intent_sha256: str
    e_intent_core_sha256: str
    q_authority_root_sha256: str
    launch_nonce: str
    supervisor_job_id: str
    supervisor_job_directory: Path
    attempt_id: str
    run_id: str
    execution_mode: str
    retry_of_run_id: str | None


class _EarlyEClaimState:
    __slots__ = (
        "descriptor",
        "mode",
        "path",
        "sha256",
        "size_bytes",
        "taken",
    )

    def __init__(self) -> None:
        self.descriptor = -1
        self.mode: str | None = None
        self.path: Path | None = None
        self.sha256: str | None = None
        self.size_bytes: int | None = None
        self.taken = False


_EARLY_E_CLAIM = _EarlyEClaimState()
_VALIDATED_E_CLAIM_TAIL: _BootstrapTail | None = None


def _exact_absolute_path(value: str, *, label: str) -> Path:
    if not value or "\x00" in value or value != os.path.abspath(os.path.normpath(value)):
        _stop(f"{label} is not exact absolute lexical canonical form")
    path = Path(value)
    if not path.is_absolute():
        _stop(f"{label} is not absolute")
    return path


def _paths_overlap(left: Path, right: Path) -> bool:
    left_key = os.path.normcase(str(left))
    right_key = os.path.normcase(str(right))
    return (
        left_key == right_key
        or left_key.startswith(f"{right_key}{os.sep}")
        or right_key.startswith(f"{left_key}{os.sep}")
    )


def _parse_exact_bootstrap_tail(argv: Sequence[str]) -> _BootstrapTail:
    raw = list(argv)
    common_length = len(_COMMON_TAIL_FLAGS) * 2
    if (
        not raw
        or raw[0] not in ALLOWED_MODES
        or len(raw) < 1 + common_length
        or not all(type(item) is str and item and "\x00" not in item for item in raw)
    ):
        _stop("capsule argv lacks one exact mode and common tail")
    mode = raw[0]
    tail = raw[1:]
    if not _strict_json_value_equal(
        tail[:common_length:2],
        list(_COMMON_TAIL_FLAGS),
    ):
        _stop("capsule argv common flags differ from the closed order")
    common = tail[1:common_length:2]
    e_intent_path = _exact_absolute_path(common[0], label="E intent path")
    e_intent_sha256 = common[1]
    e_intent_core_sha256 = common[2]
    q_authority_root_sha256 = common[3]
    launch_nonce = common[4]
    supervisor_job_id = common[5]
    supervisor_job_directory = _exact_absolute_path(
        common[6],
        label="supervisor job directory",
    )
    attempt_id = common[7]
    run_id = common[8]
    execution_mode = common[9]
    if (
        _SHA256.fullmatch(e_intent_sha256) is None
        or _SHA256.fullmatch(e_intent_core_sha256) is None
        or _SHA256.fullmatch(q_authority_root_sha256) is None
        or _SHA256.fullmatch(launch_nonce) is None
        or _IDENTIFIER.fullmatch(supervisor_job_id) is None
        or _IDENTIFIER.fullmatch(attempt_id) is None
        or _IDENTIFIER.fullmatch(run_id) is None
        or supervisor_job_directory.name != supervisor_job_id
        or supervisor_job_directory.parent.name != _SUPERVISOR_JOBS_DIRECTORY_NAME
        or e_intent_path
        != (
            supervisor_job_directory.parent.parent
            / _CONTROL_STAGING_DIRECTORY_NAME
            / supervisor_job_id
            / _E_INTENT_FILENAME
        )
    ):
        _stop("capsule argv common values violate the closed contract")
    cursor = common_length
    retry_of_run_id: str | None = None
    if execution_mode == "successor_resume":
        if (
            len(tail) < cursor + 2
            or tail[cursor] != _SUCCESSOR_LINEAGE_FLAG
            or _IDENTIFIER.fullmatch(tail[cursor + 1]) is None
        ):
            _stop("successor capsule argv lacks exact retry lineage")
        retry_of_run_id = tail[cursor + 1]
        cursor += 2
    elif execution_mode != "fresh":
        _stop("capsule execution mode is outside the closed union")
    if mode == SCIENTIFIC_MODE:
        suffix_flags: tuple[str, ...] = ()
    elif mode == PRETERMINAL_MODE:
        suffix_flags = _PRETERMINAL_SUFFIX_FLAGS
    else:
        suffix_flags = _TERMINAL_SUFFIX_FLAGS
    suffix_length = len(suffix_flags) * 2
    if len(tail) != cursor + suffix_length or not _strict_json_value_equal(
        tail[cursor : cursor + suffix_length : 2],
        list(suffix_flags),
    ):
        _stop("capsule argv suffix differs from its exact mode")
    for flag, value in zip(
        suffix_flags,
        tail[cursor + 1 : cursor + suffix_length : 2],
        strict=True,
    ):
        _exact_absolute_path(
            value,
            label=f"{flag.removeprefix('--')} path",
        )
    return _BootstrapTail(
        mode=mode,
        e_intent_path=e_intent_path,
        e_intent_sha256=e_intent_sha256,
        e_intent_core_sha256=e_intent_core_sha256,
        q_authority_root_sha256=q_authority_root_sha256,
        launch_nonce=launch_nonce,
        supervisor_job_id=supervisor_job_id,
        supervisor_job_directory=supervisor_job_directory,
        attempt_id=attempt_id,
        run_id=run_id,
        execution_mode=execution_mode,
        retry_of_run_id=retry_of_run_id,
    )


def _require_execution_capsule_contract(
    raw_value: Any,
    *,
    archive_path: Path,
    archive_descriptor: int,
    archive_evidence: _ArchiveEvidence,
    capsule_sha256: str,
    capsule_size_bytes: int,
    project_root: Path,
) -> tuple[_HeldAuthorityFile, _HeldAuthorityFile, dict[str, Any]]:
    raw = _exact_object(
        raw_value,
        fields=_EXECUTION_CAPSULE_FIELDS,
        label="execution capsule contract",
    )
    _require_self_hash(
        raw,
        self_field="contract_sha256",
        label="execution capsule contract",
    )
    plan_sha256 = _require_sha256(raw["plan_sha256"], label="capsule PLAN hash")
    runtime_root = _require_sha256(
        raw["runtime_release_root_sha256"],
        label="capsule runtime release root",
    )
    terminal_root = _require_sha256(
        raw["terminal_release_root_sha256"],
        label="capsule terminal release root",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _CAPSULE_CONTRACT_POLICY
        or raw["path"] != str(archive_path)
        or raw["size_bytes"] != capsule_size_bytes
        or type(raw["size_bytes"]) is not int
        or raw["sha256"] != capsule_sha256
        or raw["internal_manifest_sha256"] != archive_evidence.internal_manifest_sha256
        or raw["capsule_policy_sha256"] != archive_evidence.capsule_policy_sha256
        or raw["entry_contract_sha256"] != archive_evidence.entry_contract_sha256
    ):
        _stop("execution capsule contract does not select this exact archive")
    _require_exact_string_list(
        raw["python_isolated_flags"],
        ("-I", "-B"),
        label="execution capsule Python flags",
    )
    _require_exact_string_list(
        raw["allowed_modes"],
        ALLOWED_MODES,
        label="execution capsule modes",
    )
    leaf = _exact_object(
        raw["capsule_lease_identity"],
        fields=_CAPSULE_LEASE_FIELDS,
        label="capsule retained-file lease",
    )
    if raw["capsule_lease_identity_root_sha256"] != _authority_json_sha256(leaf):
        _stop("capsule retained-file lease root is invalid")
    _require_capsule_leaf_lease(
        leaf,
        archive_path=archive_path,
        descriptor=archive_descriptor,
        capsule_sha256=capsule_sha256,
        capsule_size_bytes=capsule_size_bytes,
    )
    capsule_ancestors = _exact_object(
        raw["capsule_ancestor_lease"],
        fields=_ANCESTOR_LEASE_FIELDS,
        label="capsule ancestor lease",
    )
    if raw["capsule_ancestor_lease_root_sha256"] != _authority_json_sha256(capsule_ancestors):
        _stop("capsule ancestor lease root is invalid")
    _require_live_ancestor_records(
        capsule_ancestors,
        expected_paths=(
            project_root,
            project_root / "artifacts",
            project_root / "artifacts" / "execution_capsules",
            archive_path.parent,
        ),
        policy=_CAPSULE_ANCESTOR_LEASE_POLICY,
        interpreter=False,
    )
    if type(raw["python_path"]) is not str or type(raw["python_sha256"]) is not str:
        _stop("execution capsule interpreter binding is not textual")
    python_path = _exact_absolute_path(
        raw["python_path"],
        label="execution capsule interpreter path",
    )
    expected_python = project_root / ".venv" / "Scripts" / "python.exe"
    process_argv = _native_process_argv()
    base_executable = str(getattr(sys, "_base_executable", sys.executable))
    logical_image_path = _exact_absolute_path(
        sys.executable,
        label="logical CPython executable path",
    )
    native_image_path = _native_process_image_path()
    if (
        python_path != expected_python
        or logical_image_path != python_path
        or not process_argv
        or process_argv[0] != base_executable
    ):
        _stop("execution capsule interpreter differs from this exact process")
    python_file = _open_held_plain_file(
        python_path,
        label="Q-bound interpreter",
        require_read_only=False,
        maximum_bytes=64 * 1024 * 1024,
    )
    runtime_python_file: _HeldAuthorityFile | None = None
    try:
        if python_file.sha256 != raw["python_sha256"]:
            _stop("execution capsule interpreter SHA-256 differs from Q")
        python_leaf = _exact_object(
            raw["python_lease_identity"],
            fields=_PYTHON_LEASE_FIELDS,
            label="interpreter retained-file lease",
        )
        if raw["python_lease_identity_root_sha256"] != _authority_json_sha256(python_leaf):
            _stop("interpreter retained-file lease root is invalid")
        _require_python_leaf_lease(
            python_leaf,
            python_file=python_file,
            policy=_PYTHON_LEASE_POLICY,
            label="interpreter retained-file lease",
        )
        python_ancestors = _exact_object(
            raw["python_ancestor_lease"],
            fields=_PYTHON_ANCESTOR_LEASE_FIELDS,
            label="interpreter ancestor lease",
        )
        if raw["python_ancestor_lease_root_sha256"] != _authority_json_sha256(python_ancestors):
            _stop("interpreter ancestor lease root is invalid")
        expected_paths = [
            project_root,
            project_root / ".venv",
            project_root / ".venv" / "Scripts",
        ]
        _require_live_ancestor_records(
            python_ancestors,
            expected_paths=expected_paths,
            policy=_PYTHON_ANCESTOR_LEASE_POLICY,
            interpreter=True,
        )
        if (
            raw["python_runtime_resolution_policy"] != _PYTHON_RUNTIME_RESOLUTION_POLICY
            or type(raw["runtime_python_path"]) is not str
            or type(raw["runtime_python_sha256"]) is not str
        ):
            _stop("runtime interpreter resolution policy is invalid")
        runtime_python_path = _exact_absolute_path(
            raw["runtime_python_path"],
            label="runtime interpreter path",
        )
        base_python_path = _exact_absolute_path(
            str(getattr(sys, "_base_executable", "")),
            label="CPython base executable path",
        )
        native_argv_path = _exact_absolute_path(
            process_argv[0],
            label="native argv[0] interpreter path",
        )
        user_profile = _exact_absolute_path(
            str(Path.home()),
            label="runtime interpreter user-profile anchor",
        )
        expected_runtime_ancestors = [
            user_profile,
            user_profile / "AppData",
            user_profile / "AppData" / "Local",
            user_profile / "AppData" / "Local" / "Programs",
            user_profile / "AppData" / "Local" / "Programs" / "Python",
            user_profile / "AppData" / "Local" / "Programs" / "Python" / "Python312",
        ]
        if (
            runtime_python_path != native_image_path
            or runtime_python_path != base_python_path
            or runtime_python_path != native_argv_path
            or runtime_python_path == python_path
            or runtime_python_path != expected_runtime_ancestors[-1] / "python.exe"
        ):
            _stop("runtime interpreter differs across Q and native process evidence")
        runtime_python_file = _open_held_plain_file(
            runtime_python_path,
            label="Q-bound runtime interpreter",
            require_read_only=False,
            maximum_bytes=64 * 1024 * 1024,
        )
        if runtime_python_file.sha256 != raw["runtime_python_sha256"]:
            _stop("runtime interpreter SHA-256 differs from Q")
        runtime_leaf = _exact_object(
            raw["runtime_python_lease_identity"],
            fields=_PYTHON_LEASE_FIELDS,
            label="runtime interpreter retained-file lease",
        )
        if raw["runtime_python_lease_identity_root_sha256"] != _authority_json_sha256(runtime_leaf):
            _stop("runtime interpreter retained-file lease root is invalid")
        _require_python_leaf_lease(
            runtime_leaf,
            python_file=runtime_python_file,
            policy=_RUNTIME_PYTHON_LEASE_POLICY,
            label="runtime interpreter retained-file lease",
        )
        runtime_ancestors = _exact_object(
            raw["runtime_python_ancestor_lease"],
            fields=_PYTHON_ANCESTOR_LEASE_FIELDS,
            label="runtime interpreter ancestor lease",
        )
        if raw["runtime_python_ancestor_lease_root_sha256"] != _authority_json_sha256(
            runtime_ancestors
        ):
            _stop("runtime interpreter ancestor lease root is invalid")
        _require_live_ancestor_records(
            runtime_ancestors,
            expected_paths=expected_runtime_ancestors,
            policy=_RUNTIME_PYTHON_ANCESTOR_LEASE_POLICY,
            interpreter=True,
        )
    except BaseException:
        if runtime_python_file is not None:
            os.close(runtime_python_file.descriptor)
        os.close(python_file.descriptor)
        raise
    if runtime_python_file is None:
        _stop("runtime interpreter retained handle was not established")
    return (
        python_file,
        runtime_python_file,
        {
            "plan_sha256": plan_sha256,
            "runtime_release_root_sha256": runtime_root,
            "terminal_release_root_sha256": terminal_root,
        },
    )


def _require_projection(
    raw_value: Any,
    *,
    mode: str,
    e_file_sha256: str,
    e_core_sha256: str,
    q_root_sha256: str,
    e_intent_path: Path,
    capsule_path: Path,
    capsule_sha256: str,
    python_path: Path,
    python_sha256: str,
    e_job: Mapping[str, Any],
    e_lineage: Mapping[str, Any],
) -> tuple[str, ...]:
    raw = _exact_object(
        raw_value,
        fields=_COMMAND_PROJECTION_FIELDS,
        label=f"{mode} command projection",
    )
    _require_self_hash(
        raw,
        self_field="projection_sha256",
        label=f"{mode} command projection",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["policy"] != _COMMAND_PROJECTION_POLICY
        or raw["capsule_mode"] != mode
        or raw["program_path"] != str(python_path)
        or raw["program_sha256"] != python_sha256
        or raw["capsule_path"] != str(capsule_path)
        or raw["capsule_sha256"] != capsule_sha256
        or raw["e_file_sha256_insertion_policy"] != _E_FILE_INSERTION_POLICY
        or raw["e_core_sha256_insertion_policy"] != _E_CORE_INSERTION_POLICY
    ):
        _stop(f"{mode} command projection violates its exact identity")
    _require_exact_string_list(
        raw["python_isolated_flags"],
        ("-I", "-B"),
        label=f"{mode} projection Python flags",
    )
    expected_prefix = [
        str(python_path),
        "-I",
        "-B",
        str(capsule_path),
        mode,
    ]
    _require_exact_string_list(
        raw["argv_prefix"],
        expected_prefix,
        label=f"{mode} projection argv prefix",
    )
    before = raw["tail_argv_before_e_file_sha256"]
    between = raw["tail_argv_between_e_hashes"]
    after = raw["tail_argv_after_e_core_sha256"]
    if (
        not _strict_json_value_equal(
            before,
            ["--e-intent", str(e_intent_path), "--e-intent-sha256"],
        )
        or not _strict_json_value_equal(
            between,
            ["--e-intent-core-sha256"],
        )
        or not isinstance(after, list)
        or not all(type(item) is str and item and "\x00" not in item for item in after)
    ):
        _stop(f"{mode} command projection has invalid E-hash insertion slices")
    final = tuple(expected_prefix + before + [e_file_sha256] + between + [e_core_sha256] + after)
    parsed = _parse_exact_bootstrap_tail(final[4:])
    if (
        parsed.mode != mode
        or parsed.e_intent_path != e_intent_path
        or parsed.e_intent_sha256 != e_file_sha256
        or parsed.e_intent_core_sha256 != e_core_sha256
        or parsed.q_authority_root_sha256 != q_root_sha256
        or parsed.launch_nonce != e_job["launch_nonce"]
        or parsed.supervisor_job_id != e_job["job_id"]
        or str(parsed.supervisor_job_directory) != e_job["supervisor_job_dir"]
        or parsed.attempt_id != e_job["attempt_id"]
        or parsed.run_id != e_job["run_id"]
        or parsed.execution_mode != e_lineage["execution_mode"]
        or parsed.retry_of_run_id != e_lineage["retry_of_run_id"]
    ):
        _stop(f"{mode} command projection does not rederive E's exact launch")
    cwd = _exact_absolute_path(raw["cwd"], label=f"{mode} projection cwd")
    if cwd != Path(os.getcwd()):
        _stop(f"{mode} command projection differs from the live cwd")
    return final


def _require_preimport_q_e_anchor(
    *,
    tail: _BootstrapTail,
    archive_path: Path,
    archive_descriptor: int,
    archive_evidence: _ArchiveEvidence,
    capsule_sha256: str,
    capsule_size_bytes: int,
) -> _PreimportAuthorityAnchor:
    project_root = _project_root_from_capsule_path(
        archive_path,
        capsule_sha256=capsule_sha256,
    )
    expected_q_path = (
        project_root
        / "artifacts"
        / "resource_control"
        / "original_confirmatory_q_replacement_v2.json"
    )
    e_file = _open_held_authority_file(tail.e_intent_path, label="E intent")
    q_file: _HeldAuthorityFile | None = None
    supervisor_spec_file: _HeldAuthorityFile | None = None
    terminal_client_launcher_file: _HeldAuthorityFile | None = None
    python_file: _HeldAuthorityFile | None = None
    runtime_python_file: _HeldAuthorityFile | None = None
    retained_staging_files: tuple[_HeldAuthorityFile, ...] = ()
    try:
        if e_file.sha256 != tail.e_intent_sha256:
            _stop("E intent file SHA-256 differs from exact argv")
        e = _parse_authority_json_line(e_file.payload, label="E intent")
        _exact_object(e, fields=_E_FIELDS, label="E intent")
        if (
            type(e["schema_version"]) is not int
            or e["schema_version"] != 1
            or e["policy"] != _E_POLICY
            or e["authority_disposition"] != _E_DISPOSITION
            or e["project_root"] != str(project_root)
            or e["attempt_count"] != 1
            or type(e["attempt_count"]) is not int
            or e["max_attempt_count"] != 1
            or type(e["max_attempt_count"]) is not int
            or e["automatic_retry_allowed"] is not False
            or e["scientific_outcomes_read"] is not False
        ):
            _stop("E intent violates its exact top-level policy")
        e_core = _require_sha256(
            e["intent_core_sha256"],
            label="E intent core hash",
        )
        if e_core != tail.e_intent_core_sha256 or e_core != _authority_json_sha256(
            {key: item for key, item in e.items() if key != "intent_core_sha256"}
        ):
            _stop("E intent core hash is invalid")
        q_authority = _exact_object(
            e["q_authority"],
            fields=_Q_AUTHORITY_FIELDS,
            label="E direct Q authority",
        )
        if q_authority["path"] != str(expected_q_path):
            _stop("E direct Q authority does not select the canonical Q path")
        q_file = _open_held_authority_file(expected_q_path, label="Q replacement-v2")
        if q_file.sha256 != q_authority["file_sha256"]:
            _stop("E direct Q file SHA-256 differs from retained Q")
        q = _parse_authority_json_line(q_file.payload, label="Q replacement-v2")
        _exact_object(q, fields=_Q_FIELDS, label="Q replacement-v2")
        q_root = _require_sha256(
            q["q_authority_root_sha256"],
            label="Q authority root",
        )
        if (
            type(q["schema_version"]) is not int
            or q["schema_version"] != 2
            or type(q["policy"]) is not str
            or q["policy"] != _Q_POLICY
            or type(q["authority_disposition"]) is not str
            or q["authority_disposition"] != _Q_DISPOSITION
            or type(q["q_path"]) is not str
            or q["q_path"] != str(expected_q_path)
            or type(q["project_root"]) is not str
            or q["project_root"] != str(project_root)
        ):
            _stop("Q replacement-v2 violates its exact top-level policy")
        scientific_authority = _require_scientific_authority(
            q["scientific_authority"],
            project_root=project_root,
        )
        publication_lease = _exact_object(
            q["publication_ancestor_lease"],
            fields=_PUBLICATION_ANCESTOR_LEASE_FIELDS,
            label="Q publication ancestor lease",
        )
        if q["publication_ancestor_lease_root_sha256"] != _authority_json_sha256(publication_lease):
            _stop("Q publication ancestor lease root is invalid")
        _require_publication_ancestor_lease(
            publication_lease,
            project_root=project_root,
        )
        command = _require_command_derivation_contract(q["command_derivation_contract"])
        (
            python_file,
            runtime_python_file,
            release_roots,
        ) = _require_execution_capsule_contract(
            q["execution_capsule"],
            archive_path=archive_path,
            archive_descriptor=archive_descriptor,
            archive_evidence=archive_evidence,
            capsule_sha256=capsule_sha256,
            capsule_size_bytes=capsule_size_bytes,
            project_root=project_root,
        )
        execution_capsule = _exact_object(
            q["execution_capsule"],
            fields=_EXECUTION_CAPSULE_FIELDS,
            label="execution capsule contract",
        )
        supervisor_release = _require_supervisor_release(
            q["supervisor_release"],
            execution_capsule=execution_capsule,
            plan_sha256=release_roots["plan_sha256"],
            runtime_release_root_sha256=release_roots["runtime_release_root_sha256"],
            terminal_release_root_sha256=release_roots["terminal_release_root_sha256"],
        )
        codex_handoff_base = _require_codex_handoff_base_authority(
            q["codex_handoff_base_authority"],
            project_root=project_root,
        )
        q_base_projection = {field: q[field] for field in _Q_BASE_AUTHORITY_FIELDS}
        q_base_root = _require_sha256(
            q["q_base_authority_root_sha256"],
            label="Q base authority root",
        )
        if q_base_root != _authority_json_sha256(q_base_projection):
            _stop("Q base authority root differs from its exact 12-field projection")
        attempt_identity = _require_q_attempt_identity_projection(
            q["attempt_identity_projection"],
            q_base_authority_root_sha256=q_base_root,
        )
        if (
            _require_sha256(
                q["attempt_identity_root_sha256"],
                label="Q attempt identity root",
            )
            != attempt_identity["attempt_identity_root_sha256"]
        ):
            _stop("Q attempt identity root differs from its closed projection")
        supervisor_state_root = _exact_absolute_path(
            supervisor_release["supervisor_state_root"],
            label="Q supervisor state root",
        )
        control_staging = _require_control_staging_projection(
            q["control_staging_projection"],
            job_id=attempt_identity["job_id"],
            supervisor_state_root=supervisor_state_root,
            expected_sha256=q["control_staging_projection_sha256"],
        )
        q_expected_launch_environment = _require_expected_launch_environment(
            q["expected_launch_environment"],
            attempt_nonce=cast(str, attempt_identity["launch_nonce"]),
        )
        attempt_creation = _require_codex_handoff_attempt_creation_authority(
            e["codex_handoff_attempt_creation_authority"],
            base_authority=codex_handoff_base,
            expected_output_path=(
                Path(cast(str, control_staging["final_job_dir"]))
                / "codex_handoff_attempt_authority.json"
            ),
        )
        if (
            _require_sha256(
                q["codex_handoff_attempt_creation_authority_payload_sha256"],
                label="Q Codex handoff attempt-creation payload",
            )
            != attempt_creation["payload_sha256"]
            or _require_sha256(
                e["q_codex_handoff_base_authority_payload_sha256"],
                label="E Q Codex handoff base payload",
            )
            != codex_handoff_base["payload_sha256"]
            or _require_sha256(
                e["q_codex_handoff_attempt_creation_authority_payload_sha256"],
                label="E Q Codex handoff attempt-creation payload",
            )
            != attempt_creation["payload_sha256"]
        ):
            _stop("Q/E Codex handoff authority crosslinks differ")
        if (
            q_root != tail.q_authority_root_sha256
            or q_root != q_authority["root_sha256"]
            or q_root
            != _authority_json_sha256(
                {key: item for key, item in q.items() if key != "q_authority_root_sha256"}
            )
        ):
            _stop("Q authority root differs across Q, E, and exact argv")
        terminal_client_launcher_file = _require_held_terminal_client_launcher(
            cast(
                Mapping[str, Any],
                supervisor_release["terminal_client_launcher_release"],
            )
        )
        if (
            e["execution_capsule_contract_sha256"] != execution_capsule["contract_sha256"]
            or e["command_derivation_contract_sha256"] != command["contract_sha256"]
            or not _strict_json_value_equal(
                e["supervisor_release"],
                supervisor_release,
            )
        ):
            _stop("E does not bind Q's exact capsule/command/supervisor release")
        e_job = _exact_object(
            e["job"],
            fields=_E_JOB_FIELDS,
            label="E supervisor job binding",
        )
        if (
            type(e_job["schema_version"]) is not int
            or e_job["schema_version"] != 1
            or e_job["policy"] != _E_JOB_POLICY
            or e_job["job_id"] != tail.supervisor_job_id
            or e_job["supervisor_job_dir"] != str(tail.supervisor_job_directory)
            or e_job["attempt_id"] != tail.attempt_id
            or e_job["run_id"] != tail.run_id
            or e_job["launch_nonce"] != tail.launch_nonce
            or e_job["supervisor_spec_path"] != str(tail.supervisor_job_directory / "run_spec.json")
            or type(e_job["supervisor_spec_schema_version"]) is not int
            or e_job["supervisor_spec_schema_version"] != 3
            or e_job["supervisor_spec_policy"] != _SUPERVISOR_POLICY
            or e_job["supervisor_release_root_sha256"]
            != supervisor_release["supervisor_release_root_sha256"]
            or type(e_job["terminal_custody_authority_projection"]) is not dict
            or e_job["job_id"] != attempt_identity["job_id"]
            or e_job["attempt_id"] != attempt_identity["attempt_id"]
            or e_job["run_id"] != attempt_identity["run_id"]
            or e_job["launch_nonce"] != attempt_identity["launch_nonce"]
            or e_job["supervisor_job_dir"] != control_staging["final_job_dir"]
        ):
            _stop("E supervisor job binding differs from exact argv")
        supervisor_spec_path = _exact_absolute_path(
            e_job["supervisor_spec_path"],
            label="E supervisor spec path",
        )
        supervisor_spec_file = _open_held_plain_file(
            supervisor_spec_path,
            label="supervisor run spec",
            require_read_only=False,
            maximum_bytes=_MAX_CONTROL_BYTES,
        )
        canonical_supervisor_spec, retained_staging_files = _parse_held_supervisor_run_spec(
            supervisor_spec_file,
            expected_path=supervisor_spec_path,
            supervisor_release=supervisor_release,
            control_staging_projection=control_staging,
            e_intent_path=e_file.path,
            e_file_sha256=e_file.sha256,
            e_held_file=e_file,
        )
        _require_e_consumption_contract(
            e["e_consumption_contract"],
            supervisor_job_directory=tail.supervisor_job_directory,
        )
        lineage = _exact_object(
            e["lineage"],
            fields=_E_LINEAGE_FIELDS,
            label="E execution lineage",
        )
        if (
            type(lineage["schema_version"]) is not int
            or lineage["schema_version"] != 1
            or lineage["policy"] != _E_LINEAGE_POLICY
            or lineage["execution_mode"] != tail.execution_mode
            or lineage["retry_of_run_id"] != tail.retry_of_run_id
        ):
            _stop("E execution lineage differs from exact argv")
        scientific_request = _require_scientific_request_projection(
            e["scientific_request_projection"],
            scientific_authority=scientific_authority,
            e_job=e_job,
            e_lineage=lineage,
        )
        expected_run_directory = _exact_absolute_path(
            scientific_request["expected_run_directory"],
            label="E terminal custody expected run directory",
        )
        e_expected_launch_environment = _require_expected_launch_environment(
            e["expected_launch_environment"],
            attempt_nonce=cast(str, e_job["launch_nonce"]),
        )
        e_process_environment_binding = _require_process_environment_binding(
            e["process_environment_binding"],
            expected_environment=e_expected_launch_environment,
        )
        if not _strict_json_value_equal(
            e_expected_launch_environment,
            q_expected_launch_environment,
        ):
            _stop("E launch environment differs from Q and its exact nonce")
        projections = e["command_projections"]
        if not isinstance(projections, dict) or set(projections) != set(ALLOWED_MODES):
            _stop("E command projections do not cover the exact three modes")
        terminal_projection = _exact_object(
            projections[TERMINAL_MODE],
            fields=_COMMAND_PROJECTION_FIELDS,
            label="verify-terminal command projection",
        )
        terminal_projection_sha256 = _require_sha256(
            terminal_projection["projection_sha256"],
            label="verify-terminal command projection SHA-256",
        )
        terminal_environment_sha256 = _require_sha256(
            e["process_environment_binding"].get("exact_integrity_verifier_environment_sha256"),
            label="verify-terminal environment SHA-256",
        )
        _require_terminal_custody_artifact_authority_projection(
            e_job["terminal_custody_authority_projection"],
            project_root=project_root,
            run_id=e_job["run_id"],
            expected_run_directory=expected_run_directory,
            supervisor_release=supervisor_release,
            supervisor_job_id=e_job["job_id"],
            supervisor_job_directory=tail.supervisor_job_directory,
            verify_terminal_command_projection_sha256=(terminal_projection_sha256),
            verify_terminal_environment_sha256=terminal_environment_sha256,
        )
        if (
            e_job["terminal_custody_authority_projection_root_sha256"]
            != e_job["terminal_custody_authority_projection"]["projection_root_sha256"]
        ):
            _stop("E terminal custody authority projection root differs")
        _require_supervisor_spec_protected_artifact_projection(
            canonical_supervisor_spec,
            project_root=project_root,
            run_id=e_job["run_id"],
            expected_run_directory=expected_run_directory,
            e_terminal_custody_projection=e_job["terminal_custody_authority_projection"],
        )
        final_by_mode = {
            mode: _require_projection(
                projections[mode],
                mode=mode,
                e_file_sha256=e_file.sha256,
                e_core_sha256=e_core,
                q_root_sha256=q_root,
                e_intent_path=e_file.path,
                capsule_path=archive_path,
                capsule_sha256=capsule_sha256,
                python_path=python_file.path,
                python_sha256=python_file.sha256,
                e_job=e_job,
                e_lineage=lineage,
            )
            for mode in ALLOWED_MODES
        }
        _require_supervisor_spec_exact_q_e_parity(
            canonical_supervisor_spec,
            project_root=project_root,
            e_job=e_job,
            e_lineage=lineage,
            e_intent_path=e_file.path,
            e_file_sha256=e_file.sha256,
            e_core_sha256=e_core,
            q_root_sha256=q_root,
            q_file_sha256=q_file.sha256,
            independent_verifier_receipt_sha256=cast(
                str,
                scientific_authority["independent_review_receipt_sha256"],
            ),
            execution_capsule=execution_capsule,
            supervisor_release=supervisor_release,
            e_consumption_contract=cast(
                Mapping[str, Any],
                e["e_consumption_contract"],
            ),
            expected_launch_environment=cast(
                Mapping[str, Any],
                e_expected_launch_environment,
            ),
            process_environment_binding=cast(
                Mapping[str, Any],
                e_process_environment_binding,
            ),
            e_terminal_custody_projection=cast(
                Mapping[str, Any],
                e_job["terminal_custody_authority_projection"],
            ),
            e_command_projections=cast(
                Mapping[str, Any],
                projections,
            ),
            final_by_mode=final_by_mode,
            control_staging=control_staging,
            attempt_creation=attempt_creation,
        )
        if (
            final_by_mode[tail.mode][1:] != _native_process_argv()[1:]
            or final_by_mode[tail.mode][0] != sys.executable
            or final_by_mode[tail.mode][4:] != tuple(sys.argv[1:])
        ):
            _stop("live argv is not the exact E-selected command projection")
        _require_held_file_unchanged(q_file, label="Q replacement-v2")
        _require_held_file_unchanged(e_file, label="E intent")
        _require_held_file_unchanged(
            supervisor_spec_file,
            label="supervisor run spec",
        )
        _require_held_file_unchanged(
            terminal_client_launcher_file,
            label="terminal-client launcher source",
        )
        _require_held_file_unchanged(python_file, label="Q-bound interpreter")
        _require_held_file_unchanged(
            runtime_python_file,
            label="Q-bound runtime interpreter",
        )
        for held, label in zip(
            retained_staging_files,
            (
                "control-staging attempt",
                "control-staging launch authorization",
                "control-staging supervisor source spec",
                "control-staging ready marker",
            ),
            strict=True,
        ):
            _require_held_file_unchanged(held, label=label)
        return _PreimportAuthorityAnchor(
            q_file=q_file,
            e_file=e_file,
            supervisor_spec_file=supervisor_spec_file,
            terminal_client_launcher_file=terminal_client_launcher_file,
            python_file=python_file,
            runtime_python_file=runtime_python_file,
            staging_attempt_file=retained_staging_files[0],
            launch_authorization_file=retained_staging_files[1],
            supervisor_launch_spec_source_file=retained_staging_files[2],
            staging_ready_file=retained_staging_files[3],
        )
    except BaseException:
        for held in retained_staging_files:
            os.close(held.descriptor)
        if runtime_python_file is not None:
            os.close(runtime_python_file.descriptor)
        if python_file is not None:
            os.close(python_file.descriptor)
        if supervisor_spec_file is not None:
            os.close(supervisor_spec_file.descriptor)
        if terminal_client_launcher_file is not None:
            os.close(terminal_client_launcher_file.descriptor)
        if q_file is not None:
            os.close(q_file.descriptor)
        os.close(e_file.descriptor)
        raise


def _readonly_file(value: os.stat_result) -> bool:
    if os.name == "nt":
        return bool(_file_attributes(value) & 0x00000001)
    return stat.S_IMODE(value.st_mode) & 0o222 == 0


def _require_plain_job_directory(path: Path) -> None:
    try:
        value = os.lstat(path)
    except OSError as exc:
        raise CapsuleBootstrapError("supervisor job directory is unavailable") from exc
    if not stat.S_ISDIR(value.st_mode) or stat.S_ISLNK(value.st_mode) or _is_reparse(value):
        _stop("supervisor job directory is not a plain directory")


def _create_early_e_claim(path: Path) -> int:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | int(getattr(os, "O_BINARY", 0))
        | int(getattr(os, "O_CLOEXEC", 0))
    )
    if os.name != "nt":
        return os.open(
            path,
            flags | int(getattr(os, "O_NOFOLLOW", 0)),
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
        0x00000001,
        None,
        1,
        0x00000001 | 0x00200000,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        raise OSError(
            ctypes.get_last_error(),
            f"CreateFileW CREATE_NEW failed for {path}",
        )
    try:
        return msvcrt.open_osfhandle(int(handle), flags & ~(os.O_CREAT | os.O_EXCL))
    except BaseException:
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int
        close_handle(ctypes.c_void_p(handle))
        raise


def _require_claim_path_matches_descriptor(path: Path, descriptor: int) -> os.stat_result:
    path_value = os.lstat(path)
    descriptor_value = os.fstat(descriptor)
    if (
        not stat.S_ISREG(path_value.st_mode)
        or stat.S_ISLNK(path_value.st_mode)
        or _is_reparse(path_value)
        or int(path_value.st_nlink) != 1
        or not _readonly_file(path_value)
        or _stable_file_identity(path_value) != _stable_file_identity(descriptor_value)
    ):
        _stop("early E claim path/handle identity is invalid")
    return descriptor_value


def _arm_original_confirmatory_e_claim_after_full_prevalidation() -> None:
    tail = _VALIDATED_E_CLAIM_TAIL
    if tail is None:
        _stop("E claim cannot be armed before the pre-import authority anchor")
    if _EARLY_E_CLAIM.descriptor >= 0 or _EARLY_E_CLAIM.path is not None or _EARLY_E_CLAIM.taken:
        _stop("validated E claim state was already armed")
    _require_plain_job_directory(tail.supervisor_job_directory)
    claim_path = tail.supervisor_job_directory / "e_intent_consumed.json"
    if tail.mode == SCIENTIFIC_MODE:
        try:
            descriptor = _create_early_e_claim(claim_path)
        except OSError as exc:
            raise CapsuleBootstrapError(
                "run-confirmatory could not CREATE_NEW its permanent E claim"
            ) from exc
        try:
            os.fsync(descriptor)
            value = _require_claim_path_matches_descriptor(claim_path, descriptor)
            if int(value.st_size) != 0:
                _stop("new early E claim is not empty")
        except BaseException:
            os.close(descriptor)
            raise
        sha256: str | None = None
        size_bytes: int | None = None
    else:
        try:
            descriptor = _open_capsule_no_follow(claim_path)
            value = _require_claim_path_matches_descriptor(claim_path, descriptor)
            sha256, size_bytes, identity = _hash_held_file(descriptor)
            if int(value.st_size) <= 0 or identity != _stable_file_identity(value):
                _stop("verifier E claim is not one final nonempty identity")
        except BaseException:
            if "descriptor" in locals():
                os.close(descriptor)
            raise
    _EARLY_E_CLAIM.descriptor = descriptor
    _EARLY_E_CLAIM.mode = tail.mode
    _EARLY_E_CLAIM.path = claim_path
    _EARLY_E_CLAIM.sha256 = sha256
    _EARLY_E_CLAIM.size_bytes = size_bytes


def _take_original_confirmatory_e_claim_handle() -> tuple[int, str]:
    state = _EARLY_E_CLAIM
    if state.taken or state.mode != SCIENTIFIC_MODE or state.descriptor < 0 or state.path is None:
        _stop("run-confirmatory early E claim is absent, wrong-mode, or already taken")
    _require_claim_path_matches_descriptor(state.path, state.descriptor)
    descriptor = state.descriptor
    state.descriptor = -1
    state.taken = True
    return descriptor, str(state.path)


def _take_original_confirmatory_e_claim_read_handle() -> tuple[int, str, str, int]:
    state = _EARLY_E_CLAIM
    if (
        state.taken
        or state.mode not in {PRETERMINAL_MODE, TERMINAL_MODE}
        or state.descriptor < 0
        or state.path is None
        or state.sha256 is None
        or state.size_bytes is None
    ):
        _stop("verifier early E claim is absent, wrong-mode, or already taken")
    digest, size_bytes, identity = _hash_held_file(state.descriptor)
    path_value = _require_claim_path_matches_descriptor(state.path, state.descriptor)
    if (
        digest != state.sha256
        or size_bytes != state.size_bytes
        or identity != _stable_file_identity(path_value)
    ):
        _stop("verifier early E claim changed before transfer")
    descriptor = state.descriptor
    state.descriptor = -1
    state.taken = True
    return descriptor, str(state.path), digest, size_bytes


def _close_untaken_early_e_claim() -> None:
    state = _EARLY_E_CLAIM
    if state.descriptor >= 0:
        os.close(state.descriptor)
        state.descriptor = -1


def _native_process_argv() -> tuple[str, ...]:
    if os.name != "nt":
        return tuple(sys.orig_argv)
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_command_line = kernel32.GetCommandLineW
    get_command_line.argtypes = []
    get_command_line.restype = ctypes.c_wchar_p
    command_line_to_argv = shell32.CommandLineToArgvW
    command_line_to_argv.argtypes = [
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    command_line_to_argv.restype = ctypes.POINTER(ctypes.c_wchar_p)
    count = ctypes.c_int()
    raw = command_line_to_argv(get_command_line(), ctypes.byref(count))
    if not raw or count.value <= 0:
        raise CapsuleBootstrapError("native Windows command-line parsing failed")
    try:
        return tuple(raw[index] for index in range(count.value))
    finally:
        local_free = kernel32.LocalFree
        local_free.argtypes = [ctypes.c_void_p]
        local_free.restype = ctypes.c_void_p
        local_free(ctypes.cast(raw, ctypes.c_void_p))


def _native_process_image_path() -> Path:
    if os.name != "nt":
        return Path(str(getattr(sys, "_base_executable", sys.executable)))
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_module_filename = kernel32.GetModuleFileNameW
    get_module_filename.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
    ]
    get_module_filename.restype = ctypes.c_uint32
    for size in (512, 1024, 2048, 4096, 8192, 16384, 32768):
        buffer = ctypes.create_unicode_buffer(size)
        length = int(get_module_filename(None, buffer, size))
        if 0 < length < size - 1:
            return _exact_absolute_path(
                buffer.value[:length],
                label="internal CPython module image path",
            )
        if length == 0:
            raise OSError(
                ctypes.get_last_error(),
                "GetModuleFileNameW(NULL) failed",
            )
    _stop("internal CPython module image path exceeds the closed Windows bound")


def _require_exact_process_shape(archive_path: Path) -> tuple[str, ...]:
    if (
        sys.flags.isolated != 1
        or sys.flags.dont_write_bytecode != 1
        or sys.flags.optimize != 0
        or not archive_path.is_absolute()
        or len(sys.argv) < 2
        or sys.argv[1] not in ALLOWED_MODES
    ):
        _stop("capsule requires exact isolated -I -B optimize=0 launch and one valid mode")
    original = _native_process_argv()
    expected_prefix = (
        str(getattr(sys, "_base_executable", sys.executable)),
        "-I",
        "-B",
        str(archive_path),
    )
    if len(original) < 5 or original[:4] != expected_prefix or original[4:] != tuple(sys.argv[1:]):
        _stop("capsule sys.orig_argv differs from the exact launch shape")
    return tuple(sys.argv[1:])


class _SealedProjectFinder(importlib.abc.MetaPathFinder):
    def __init__(
        self,
        *,
        archive_path: Path,
        modules: Mapping[str, tuple[str, bool]],
    ) -> None:
        self._archive_path = archive_path
        self._modules = dict(modules)
        self._issued_specs: dict[str, importlib.machinery.ModuleSpec] = {}

    @property
    def issued_specs(self) -> Mapping[str, importlib.machinery.ModuleSpec]:
        return dict(self._issued_specs)

    def _expected_package_path(self, module_name: str) -> str:
        segments = module_name.split(".")
        return str(self._archive_path.joinpath(*segments))

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del target
        if fullname != "histo_audit" and not fullname.startswith("histo_audit."):
            return None
        record = self._modules.get(fullname)
        if record is None:
            raise ModuleNotFoundError(
                f"project module is absent from the sealed capsule manifest: {fullname}"
            )
        if fullname in self._issued_specs:
            raise ImportError(f"project module spec was requested repeatedly: {fullname}")
        relative_path, is_package = record
        parent_name = fullname.rpartition(".")[0]
        if parent_name:
            expected_parent_path = self._expected_package_path(parent_name)
            if path is None or list(path) != [expected_parent_path]:
                raise ImportError(
                    f"project package path differs from the sealed capsule: {fullname}"
                )
            importer_path = expected_parent_path
        else:
            if path is not None:
                raise ImportError("root project package received a parent search path")
            importer_path = str(self._archive_path)
        importer = zipimport.zipimporter(importer_path)
        spec = importer.find_spec(fullname)
        expected_origin = str(self._archive_path.joinpath(*PurePosixPath(relative_path).parts))
        expected_locations = [self._expected_package_path(fullname)] if is_package else None
        if (
            spec is None
            or spec.name != fullname
            or not isinstance(spec.loader, zipimport.zipimporter)
            or os.path.normcase(os.path.abspath(spec.origin or ""))
            != os.path.normcase(os.path.abspath(expected_origin))
            or (
                list(spec.submodule_search_locations)
                if spec.submodule_search_locations is not None
                else None
            )
            != expected_locations
        ):
            raise ImportError(f"zip importer returned a non-manifest project spec: {fullname}")
        self._issued_specs[fullname] = spec
        return spec


def _sanitize_import_state(
    archive_path: Path,
    *,
    modules: Mapping[str, tuple[str, bool]],
) -> _SealedProjectFinder:
    if any(name == "histo_audit" or name.startswith("histo_audit.") for name in sys.modules):
        _stop("project module was imported before capsule verification")
    if "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
        _stop("custom site module executed before capsule verification")
    archive_text = os.path.normcase(os.path.abspath(str(archive_path)))
    cleaned: list[str] = []
    for raw in sys.path:
        if raw == "":
            continue
        normalized = os.path.normcase(os.path.abspath(raw))
        if normalized == os.path.normcase(os.path.abspath(os.getcwd())):
            continue
        if normalized not in {os.path.normcase(os.path.abspath(value)) for value in cleaned}:
            cleaned.append(raw)
    if not cleaned or os.path.normcase(os.path.abspath(cleaned[0])) != archive_text:
        _stop("capsule path is not the first isolated import path")
    sys.path[:] = cleaned
    source_loader = (
        importlib.machinery.SourceFileLoader,
        importlib.machinery.SOURCE_SUFFIXES,
    )
    bytecode_loader = (
        importlib.machinery.SourcelessFileLoader,
        importlib.machinery.BYTECODE_SUFFIXES,
    )
    extension_loader = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES,
    )
    sys.path_hooks[:] = [
        zipimport.zipimporter,
        importlib.machinery.FileFinder.path_hook(
            source_loader,
            bytecode_loader,
            extension_loader,
        ),
    ]
    sys.path_importer_cache.clear()
    sealed_project_finder = _SealedProjectFinder(
        archive_path=archive_path,
        modules=modules,
    )
    sys.meta_path[:] = [
        importlib.machinery.BuiltinImporter,
        importlib.machinery.FrozenImporter,
        sealed_project_finder,
        importlib.machinery.PathFinder,
    ]
    importlib.invalidate_caches()
    return sealed_project_finder


def _origin_is_capsule(origin: str, archive_path: Path) -> bool:
    archive_prefix = os.path.normcase(os.path.abspath(str(archive_path)))
    normalized = os.path.normcase(os.path.abspath(origin))
    return normalized.startswith(archive_prefix + os.sep)


def _require_project_origins(
    archive_path: Path,
    *,
    modules: Mapping[str, tuple[str, bool]],
    finder: _SealedProjectFinder,
) -> None:
    archive_text = os.path.normcase(os.path.abspath(str(archive_path)))
    seen: set[str] = set()
    issued_specs = finder.issued_specs
    for name, raw_module in tuple(sys.modules.items()):
        if name != "histo_audit" and not name.startswith("histo_audit."):
            continue
        record = modules.get(name)
        issued_spec = issued_specs.get(name)
        if record is None or issued_spec is None:
            _stop(f"project module was not issued by the sealed finder: {name}")
        relative_path, is_package = record
        expected_origin = str(archive_path.joinpath(*PurePosixPath(relative_path).parts))
        expected_locations = [str(archive_path.joinpath(*name.split(".")))] if is_package else None
        module = raw_module
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
        loader = getattr(spec, "loader", None)
        loader_archive = getattr(loader, "archive", None)
        module_file = getattr(module, "__file__", None)
        module_loader = getattr(module, "__loader__", None)
        locations = getattr(spec, "submodule_search_locations", None)
        module_path = getattr(module, "__path__", None)
        normalized_locations = list(locations) if locations is not None else None
        normalized_module_path = list(module_path) if module_path is not None else None
        if (
            spec is not issued_spec
            or not isinstance(origin, str)
            or not _origin_is_capsule(origin, archive_path)
            or os.path.normcase(os.path.abspath(origin))
            != os.path.normcase(os.path.abspath(expected_origin))
            or not isinstance(loader, zipimport.zipimporter)
            or not isinstance(loader_archive, str)
            or os.path.normcase(os.path.abspath(loader_archive)) != archive_text
            or module_loader is not loader
            or not isinstance(module_file, str)
            or os.path.normcase(os.path.abspath(module_file))
            != os.path.normcase(os.path.abspath(origin))
            or normalized_locations != expected_locations
            or normalized_module_path != expected_locations
        ):
            _stop(f"project module origin is outside the capsule: {name}")
        seen.add(name)
    if not seen:
        _stop("capsule dispatcher imported no project module")
    if seen != set(issued_specs):
        _stop("sealed project finder issued an unaccounted module spec")


def main() -> int:
    global _VALIDATED_E_CLAIM_TAIL

    archive_path = Path(sys.argv[0])
    argv = _require_exact_process_shape(archive_path)
    tail = _parse_exact_bootstrap_tail(argv)
    descriptor = -1
    authority_anchor: _PreimportAuthorityAnchor | None = None
    try:
        descriptor = _open_capsule_no_follow(archive_path)
        capsule_sha256, capsule_size_bytes, capsule_identity = (
            _require_content_addressed_held_capsule(archive_path, descriptor)
        )
        archive_evidence = _verify_archive(descriptor)
        modules = _project_module_inventory(archive_evidence.entries)
        _require_held_capsule_unchanged(
            archive_path,
            descriptor,
            expected_sha256=capsule_sha256,
            expected_size_bytes=capsule_size_bytes,
            expected_identity=capsule_identity,
        )
        authority_anchor = _require_preimport_q_e_anchor(
            tail=tail,
            archive_path=archive_path,
            archive_descriptor=descriptor,
            archive_evidence=archive_evidence,
            capsule_sha256=capsule_sha256,
            capsule_size_bytes=capsule_size_bytes,
        )
        _VALIDATED_E_CLAIM_TAIL = tail
        finder = _sanitize_import_state(
            archive_path,
            modules=modules,
        )
        try:
            entry = importlib.import_module(
                "histo_audit.workflows.original_confirmatory_capsule_entry"
            )
        finally:
            _require_project_origins(
                archive_path,
                modules=modules,
                finder=finder,
            )
        dispatcher = getattr(entry, "_dispatch_original_confirmatory_capsule", None)
        if not callable(dispatcher):
            _stop("sealed capsule dispatcher is absent")
        _require_held_capsule_unchanged(
            archive_path,
            descriptor,
            expected_sha256=capsule_sha256,
            expected_size_bytes=capsule_size_bytes,
            expected_identity=capsule_identity,
        )
        try:
            result = dispatcher(argv)
        finally:
            try:
                _require_project_origins(
                    archive_path,
                    modules=modules,
                    finder=finder,
                )
            finally:
                try:
                    _require_held_capsule_unchanged(
                        archive_path,
                        descriptor,
                        expected_sha256=capsule_sha256,
                        expected_size_bytes=capsule_size_bytes,
                        expected_identity=capsule_identity,
                    )
                finally:
                    for held, label in zip(
                        authority_anchor,
                        (
                            "Q replacement-v2",
                            "E intent",
                            "supervisor run spec",
                            "terminal-client launcher source",
                            "Q-bound interpreter",
                            "Q-bound runtime interpreter",
                            "control-staging attempt",
                            "control-staging launch authorization",
                            "control-staging supervisor source spec",
                            "control-staging ready marker",
                        ),
                        strict=True,
                    ):
                        _require_held_file_unchanged(held, label=label)
        if type(result) is not int:
            _stop("sealed capsule dispatcher returned a non-integer exit code")
        if not _EARLY_E_CLAIM.taken:
            _stop("sealed capsule dispatcher did not take the one-use E claim handle")
        return result
    finally:
        _VALIDATED_E_CLAIM_TAIL = None
        _close_preimport_anchor(authority_anchor)
        if descriptor >= 0:
            os.close(descriptor)
        _close_untaken_early_e_claim()


if __name__ == "__main__":
    raise SystemExit(main())
