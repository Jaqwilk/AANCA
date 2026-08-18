from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from carrier_import_guard import PACKAGE_IMPORT_ROOT, import_exact

authority = import_exact(
    "histo_audit.workflows.original_confirmatory_capsule_authority",
    PACKAGE_IMPORT_ROOT
    / "histo_audit"
    / "workflows"
    / "original_confirmatory_capsule_authority.py",
)


def _base(*, operational: bool = False) -> dict[str, Any]:
    schema = (
        authority.CODEX_HANDOFF_BASE_OPERATIONAL_SCHEMA
        if operational
        else authority.CODEX_HANDOFF_BASE_SYNTHETIC_SCHEMA
    )
    scope = (
        authority.CODEX_HANDOFF_BASE_OPERATIONAL_SCOPE
        if operational
        else authority.CODEX_HANDOFF_BASE_SYNTHETIC_SCOPE
    )
    capability = (
        {
            "production_arm_enabled": True,
            "real_resume_enabled": True,
            "synthetic_only": False,
        }
        if operational
        else {
            "production_arm_enabled": False,
            "real_resume_enabled": False,
            "synthetic_only": True,
        }
    )
    payload = {
        "authority_scope": scope,
        "session_origin": {
            "session_id": "12345678-1234-1234-9234-123456789abc",
            "session_jsonl_path": "C:\\Users\\NATAN\\.codex\\sessions\\one.jsonl",
            "expected_cwd": "C:\\Users\\NATAN\\Documents\\AANCA",
            "first_record": {
                "record_type": "session_meta",
                "payload_id": "12345678-1234-1234-9234-123456789abc",
                "payload_session_id": "12345678-1234-1234-9234-123456789abc",
                "payload_cli_version": "0.145.0-alpha.18",
                "raw_record_bytes_excluding_delimiter": 41079,
                "raw_record_sha256_excluding_delimiter": "1" * 64,
                "delimiter_hex": "0a",
            },
            "session_file_identity": {
                "volume_serial_number": 123,
                "file_id_128": "2" * 32,
                "creation_time_100ns": 456,
                "file_attributes": 32,
                "link_count": 1,
                "directory": False,
                "reparse_point": False,
            },
        },
        "codex_cli": {
            "path": "C:\\Tools\\codex.exe",
            "size_bytes": 123456,
            "sha256": "3" * 64,
            "version_stdout": "codex-cli 0.145.0",
        },
        "resume_command_policy": authority._codex_handoff_resume_command_policy(
            operational=operational
        ),
        "limits": authority._codex_handoff_limits(),
        "capability_policy": capability,
    }
    if operational:
        payload.update(
            {
                "branch_template_policy": authority._codex_handoff_branch_template_policy(),
                "idle_completion_policy": authority._codex_handoff_completion_policy(),
                "external_supervisor_handoff_policy": (
                    authority._codex_handoff_external_supervisor_policy()
                ),
                "operational_source": {
                    "schema": "aanca.operational-handoff-source.v1",
                    "source_path": "C:\\AANCA\\operational_handoff.py",
                    "source_size_bytes": 1000,
                    "source_sha256": "a" * 64,
                    "source_inventory_path": "C:\\AANCA\\source_inventory.json",
                    "source_inventory_file_sha256": "b" * 64,
                    "source_inventory_payload_sha256": "c" * 64,
                    "source_inventory_root_sha256": "d" * 64,
                    "independent_audit_receipt_path": "C:\\AANCA\\audit.json",
                    "independent_audit_receipt_sha256": "e" * 64,
                    "authority_spec_path": "C:\\AANCA\\authority_spec.json",
                    "authority_spec_file_sha256": "f" * 64,
                    "authority_spec_payload_sha256": "1" * 64,
                    "synthetic_inventory_path": "C:\\AANCA\\synthetic_inventory.json",
                    "synthetic_inventory_size_bytes": 2000,
                    "synthetic_inventory_file_sha256": "2" * 64,
                    "synthetic_inventory_root_sha256": "3" * 64,
                    "synthetic_gate_source_path": "C:\\AANCA\\synthetic_gate.py",
                    "synthetic_gate_source_size_bytes": 3000,
                    "synthetic_gate_source_sha256": "4" * 64,
                },
            }
        )
    else:
        payload.update(
            {
                "continuation_prompt_policy": authority._codex_handoff_prompt_policy(),
                "completion_policy": authority._codex_handoff_completion_policy(),
            }
        )
    return {
        "schema": schema,
        "payload": payload,
        "payload_sha256": authority.canonical_json_sha256(payload),
    }


def _creation(
    base: dict[str, Any],
    output_path: str = "C:\\AANCA\\state\\jobs\\job-1\\codex_handoff_attempt_authority.json",
) -> dict[str, Any]:
    marker_nonce = "4" * 64
    payload = {
        "authority_scope": authority.CODEX_HANDOFF_ATTEMPT_CREATION_SCOPE,
        "base_authority_payload_sha256": base["payload_sha256"],
        "session_id": base["payload"]["session_origin"]["session_id"],
        "turn_id": "87654321-4321-1234-9234-cba987654321",
        "marker_nonce_hex": marker_nonce,
        "marker": f"AANCA_CURRENT_SESSION_IDLE_{marker_nonce}",
        "success_template_policy_root_sha256": base["payload"]["branch_template_policy"][
            "success_template_policy_root_sha256"
        ],
        "diagnosis_template_policy_root_sha256": base["payload"]["branch_template_policy"][
            "diagnosis_template_policy_root_sha256"
        ],
        "authority_spec_payload_sha256": base["payload"]["operational_source"][
            "authority_spec_payload_sha256"
        ],
        "arm_algorithm_contract_root_sha256": authority.canonical_json_sha256(
            authority._codex_handoff_arm_algorithm_contract()
        ),
        "attempt_authority_output_path": output_path,
        "attempt_authority_schema": authority.CODEX_HANDOFF_ATTEMPT_SCHEMA,
        "arm_algorithm": authority.CODEX_HANDOFF_ARM_ALGORITHM,
        "required_absent_before": True,
        "create_new_required": True,
        "one_use_policy": {
            "attempt_number": 1,
            "maximum_attempts": 1,
            "automatic_retry_allowed": False,
            "max_age_after_arm_ms": 3_600_000,
            "branch_selection_time": "postterminal",
            "rendered_prompt_at_creation_allowed": False,
        },
    }
    return {
        "schema": authority.CODEX_HANDOFF_ATTEMPT_CREATION_SCHEMA,
        "payload": payload,
        "payload_sha256": authority.canonical_json_sha256(payload),
    }


def _attempt(base: dict[str, Any], creation: dict[str, Any]) -> dict[str, Any]:
    armed = datetime(2026, 7, 31, 12, 0, 1, tzinfo=UTC)
    registered = armed - timedelta(seconds=1)
    marker_nonce = creation["payload"]["marker_nonce_hex"]
    marker = creation["payload"]["marker"]
    turn_id = creation["payload"]["turn_id"]
    session = base["payload"]["session_origin"]
    payload = {
        "authority_scope": authority.CODEX_HANDOFF_ATTEMPT_SCOPE,
        "attempt_creation_authority_payload_sha256": creation["payload_sha256"],
        "base_authority_payload_sha256": base["payload_sha256"],
        "attempt_id": "12345678-1234-4234-9234-123456789abc",
        "session_id": session["session_id"],
        "turn_id": turn_id,
        "marker": marker,
        "marker_nonce_hex": marker_nonce,
        "armed_at_utc": armed.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "expires_at_utc": (armed + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "session_file_identity": copy.deepcopy(session["session_file_identity"]),
        "pre_arm": {
            "offset_bytes": 200_466_075,
            "prefix_sha256": "5" * 64,
            "record_count": 100,
            "ends_with_lf": True,
        },
        "watch_registration": {
            "registered_at_utc": registered.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "primitive": ("ReadDirectoryChangesW_overlapped_then_WaitForSingleObject_INFINITE"),
            "parent_path": "C:\\Users\\NATAN\\.codex\\sessions",
            "target_filename": "one.jsonl",
            "watch_subtree": False,
            "notify_filter": [
                "FILE_NOTIFY_CHANGE_FILE_NAME",
                "FILE_NOTIFY_CHANGE_SIZE",
                "FILE_NOTIFY_CHANGE_LAST_WRITE",
            ],
            "buffer_bytes": 65536,
            "armed_before_snapshot": True,
        },
        "success_template_policy_root_sha256": creation["payload"][
            "success_template_policy_root_sha256"
        ],
        "diagnosis_template_policy_root_sha256": creation["payload"][
            "diagnosis_template_policy_root_sha256"
        ],
        "branch_selection_policy": authority.CODEX_HANDOFF_BRANCH_SELECTION_POLICY,
        "one_use_policy": {
            "attempt_number": 1,
            "maximum_attempts": 1,
            "automatic_retry_allowed": False,
            "arm_receipt_create_new_required": True,
            "attempt_authority_create_new_required": True,
            "wake_intent_create_new_required": True,
            "wake_intent_required_before_spawn": True,
            "branch_selection_time": ("postterminal_after_terminal_handoff_receipt_validation"),
            "rendered_prompt_at_arm_allowed": False,
        },
    }
    return {
        "schema": authority.CODEX_HANDOFF_ATTEMPT_SCHEMA,
        "payload": payload,
        "payload_sha256": authority.canonical_json_sha256(payload),
    }


def _reseal(value: dict[str, Any]) -> None:
    value["payload_sha256"] = authority.canonical_json_sha256(value["payload"])


def test_profiles_are_distinct_and_synthetic_is_not_operational() -> None:
    synthetic = _base()
    operational = _base(operational=True)
    assert (
        authority.canonical_original_confirmatory_codex_handoff_base_authority(
            synthetic, require_operational=False
        )
        == synthetic
    )
    assert (
        authority.canonical_original_confirmatory_codex_handoff_base_authority(
            operational, require_operational=True
        )
        == operational
    )
    with pytest.raises(
        authority.OriginalConfirmatoryCapsuleAuthorityError,
        match="not authorized",
    ):
        authority.canonical_original_confirmatory_codex_handoff_base_authority(
            synthetic, require_operational=True
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "missing",
        "unknown",
        "hash",
        "path",
        "session",
        "identity",
        "limit",
        "profile_relabel",
    ],
)
def test_base_rejects_closed_schema_and_crosslink_tamper(tamper: str) -> None:
    value = _base()
    if tamper == "missing":
        del value["payload"]["limits"]
    elif tamper == "unknown":
        value["payload"]["unknown"] = False
    elif tamper == "hash":
        value["payload_sha256"] = "f" * 64
        with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
            authority.canonical_original_confirmatory_codex_handoff_base_authority(
                value, require_operational=False
            )
        return
    elif tamper == "path":
        value["payload"]["session_origin"]["session_jsonl_path"] = (
            "c:\\Users\\NATAN\\.codex\\sessions\\one.jsonl"
        )
    elif tamper == "session":
        value["payload"]["session_origin"]["first_record"]["payload_session_id"] = (
            "aaaaaaaa-aaaa-aaaa-8aaa-aaaaaaaaaaaa"
        )
    elif tamper == "identity":
        value["payload"]["session_origin"]["session_file_identity"]["link_count"] = 2
    elif tamper == "limit":
        value["payload"]["limits"]["max_session_file_bytes"] -= 1
    else:
        value["schema"] = authority.CODEX_HANDOFF_BASE_OPERATIONAL_SCHEMA
        value["payload"]["authority_scope"] = authority.CODEX_HANDOFF_BASE_OPERATIONAL_SCOPE
    _reseal(value)
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_codex_handoff_base_authority(
            value, require_operational=False
        )


def test_creation_is_exact_and_rejects_resealed_tamper() -> None:
    base = _base(operational=True)
    expected = _creation(base)
    assert (
        authority.canonical_original_confirmatory_codex_handoff_attempt_creation_authority(
            expected, base_authority=base
        )
        == expected
    )
    changes = (
        ("base_authority_payload_sha256", "6" * 64),
        ("session_id", "aaaaaaaa-aaaa-aaaa-8aaa-aaaaaaaaaaaa"),
        ("marker_nonce_hex", "7" * 64),
        ("authority_spec_payload_sha256", "8" * 64),
        ("arm_algorithm_contract_root_sha256", "9" * 64),
    )
    for field, replacement in changes:
        changed = copy.deepcopy(expected)
        changed["payload"][field] = replacement
        _reseal(changed)
        with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
            authority.canonical_original_confirmatory_codex_handoff_attempt_creation_authority(
                changed, base_authority=base
            )
    for path, replacement in (
        (("one_use_policy", "maximum_attempts"), 2),
        (("one_use_policy", "rendered_prompt_at_creation_allowed"), True),
    ):
        changed = copy.deepcopy(expected)
        changed["payload"][path[0]][path[1]] = replacement
        _reseal(changed)
        with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
            authority.canonical_original_confirmatory_codex_handoff_attempt_creation_authority(
                changed, base_authority=base
            )


def test_creation_rejects_missing_unknown_hash_and_rendered_fields() -> None:
    base = _base(operational=True)
    expected = _creation(base)
    variants = []
    missing = copy.deepcopy(expected)
    del missing["payload"]["one_use_policy"]
    _reseal(missing)
    variants.append(missing)
    unknown = copy.deepcopy(expected)
    unknown["payload"]["unknown"] = None
    _reseal(unknown)
    variants.append(unknown)
    bad_hash = copy.deepcopy(expected)
    bad_hash["payload_sha256"] = "a" * 64
    variants.append(bad_hash)
    rendered = copy.deepcopy(expected)
    rendered["payload"]["rendered_prompt"] = "prohibited"
    _reseal(rendered)
    variants.append(rendered)
    for changed in variants:
        with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
            authority.canonical_original_confirmatory_codex_handoff_attempt_creation_authority(
                changed, base_authority=base
            )


def test_concrete_attempt_is_exact_and_prohibits_prelaunch_rendering() -> None:
    base = _base(operational=True)
    creation = _creation(base)
    expected = _attempt(base, creation)
    assert (
        authority.canonical_original_confirmatory_codex_handoff_attempt_authority(
            expected,
            base_authority=base,
            creation_authority=creation,
        )
        == expected
    )
    for path, replacement in (
        (("session_file_identity", "file_id_128"), "8" * 32),
        (("watch_registration", "parent_path"), "C:\\Wrong"),
        (("pre_arm", "offset_bytes"), 1_073_741_825),
    ):
        changed = copy.deepcopy(expected)
        changed["payload"][path[0]][path[1]] = replacement
        _reseal(changed)
        with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
            authority.canonical_original_confirmatory_codex_handoff_attempt_authority(
                changed,
                base_authority=base,
                creation_authority=creation,
            )
    rendered = copy.deepcopy(expected)
    rendered["payload"]["rendered_argv"] = ["prohibited"]
    _reseal(rendered)
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_codex_handoff_attempt_authority(
            rendered,
            base_authority=base,
            creation_authority=creation,
        )
