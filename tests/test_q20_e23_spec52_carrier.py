from __future__ import annotations

import copy
import hashlib
import importlib.machinery
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from carrier_import_guard import (
    CARRIER_ROOT,
    PACKAGE_IMPORT_ROOT,
    import_exact,
    require_ordinary_carrier_file,
    resolve_carrier_root,
    resolve_package_import_root,
)

bootstrap = import_exact("capsule_bootstrap", CARRIER_ROOT / "capsule_bootstrap.py")
authority = import_exact(
    "histo_audit.workflows.original_confirmatory_capsule_authority",
    PACKAGE_IMPORT_ROOT
    / "histo_audit"
    / "workflows"
    / "original_confirmatory_capsule_authority.py",
)
terminal = import_exact(
    "histo_audit.workflows.original_confirmatory_capsule_terminal",
    PACKAGE_IMPORT_ROOT / "histo_audit" / "workflows" / "original_confirmatory_capsule_terminal.py",
)


def test_import_guard_rejects_preloaded_wrong_origin_without_reloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "carrier_preloaded_wrong_origin_probe"
    wrong_origin = CARRIER_ROOT / "tests" / "carrier_import_guard.py"
    module = ModuleType(module_name)
    module.__file__ = str(wrong_origin)
    module.__spec__ = importlib.machinery.ModuleSpec(
        module_name,
        loader=None,
        origin=str(wrong_origin),
    )
    monkeypatch.setitem(sys.modules, module_name, module)

    with pytest.raises(RuntimeError, match="carrier test imported"):
        import_exact(module_name, CARRIER_ROOT / "capsule_bootstrap.py")
    assert sys.modules[module_name] is module


def test_import_guard_rejects_ambiguous_package_layout(tmp_path: Path) -> None:
    for package_root in (tmp_path, tmp_path / "src"):
        authority_path = (
            package_root
            / "histo_audit"
            / "workflows"
            / "original_confirmatory_capsule_authority.py"
        )
        authority_path.parent.mkdir(parents=True, exist_ok=True)
        authority_path.write_bytes(b"# synthetic ambiguity probe\n")

    with pytest.raises(RuntimeError, match="exactly one"):
        resolve_package_import_root(tmp_path)


def test_import_guard_rejects_missing_target() -> None:
    with pytest.raises(RuntimeError, match="missing or outside"):
        import_exact(
            "carrier_missing_target_probe",
            CARRIER_ROOT / "tests" / "definitely_missing_carrier_target.py",
        )


def test_import_guard_accepts_mapped_repository_layout(tmp_path: Path) -> None:
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tmp_path / "capsule_bootstrap.py").write_bytes(b"# mapped bootstrap\n")
    authority_path = (
        tmp_path
        / "src"
        / "histo_audit"
        / "workflows"
        / "original_confirmatory_capsule_authority.py"
    )
    authority_path.parent.mkdir(parents=True)
    authority_path.write_bytes(b"# mapped authority\n")

    assert resolve_carrier_root(tests_root) == tmp_path.resolve()
    assert resolve_package_import_root(tmp_path) == (tmp_path / "src")


def test_staged_reference_dependencies_are_exact_and_layout_explicit() -> None:
    if PACKAGE_IMPORT_ROOT == CARRIER_ROOT:
        entry_path = CARRIER_ROOT / "project_capsule_entry_reference.py"
        contract_path = CARRIER_ROOT / "project_entry_contract_reference.json"
    else:
        entry_path = (
            PACKAGE_IMPORT_ROOT
            / "histo_audit"
            / "workflows"
            / "original_confirmatory_capsule_entry.py"
        )
        contract_path = CARRIER_ROOT / "entry_contract.json"
    expected = {
        CARRIER_ROOT / "capsule_builder.py": (
            46_472,
            "dd436f2255eb735ffab2db26cf016149512651ae0fc17a8fa7108e0acffbf85e",
        ),
        entry_path: (
            5_733,
            "7c50b6cac8fe6772514a58a2afd6a9d8d09c4e5038b5e0002ac3d3482c23fd68",
        ),
        contract_path: (
            305,
            "50c2796e0a3e1e06ec3fea3964c9ed1795f9552f85dbd394618529eba61bb844",
        ),
    }
    for path, (expected_size, expected_sha256) in expected.items():
        resolved = require_ordinary_carrier_file(path, carrier_root=CARRIER_ROOT)
        payload = resolved.read_bytes()
        assert len(payload) == expected_size
        assert hashlib.sha256(payload).hexdigest() == expected_sha256


def _resolve_source_root(here: Path) -> Path:
    candidates = [
        candidate
        for candidate in (here, here.parent)
        if (candidate / "capsule_bootstrap.py").is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "carrier test requires exactly one external-root or repo-tests source layout"
        )
    return candidates[0]


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = _resolve_source_root(HERE)
PROJECT_ROOT = SOURCE_ROOT


def _base_authority() -> dict[str, Any]:
    session_id = "12345678-1234-4234-9234-123456789abc"
    payload: dict[str, Any] = {
        "authority_scope": authority.CODEX_HANDOFF_BASE_OPERATIONAL_SCOPE,
        "session_origin": {
            "session_id": session_id,
            "session_jsonl_path": r"C:\Users\NATAN\.codex\sessions\session.jsonl",
            "expected_cwd": str(PROJECT_ROOT),
            "first_record": {
                "record_type": "session_meta",
                "payload_id": session_id,
                "payload_session_id": session_id,
                "payload_cli_version": "codex-cli 1.0.0",
                "raw_record_bytes_excluding_delimiter": 100,
                "raw_record_sha256_excluding_delimiter": "1" * 64,
                "delimiter_hex": "0a",
            },
            "session_file_identity": {
                "volume_serial_number": 1,
                "file_id_128": "2" * 32,
                "creation_time_100ns": 1,
                "file_attributes": 32,
                "link_count": 1,
                "directory": False,
                "reparse_point": False,
            },
        },
        "codex_cli": {
            "path": r"C:\Tools\codex.exe",
            "size_bytes": 123,
            "sha256": "3" * 64,
            "version_stdout": "codex-cli 1.0.0",
        },
        "resume_command_policy": authority._codex_handoff_resume_command_policy(operational=True),
        "limits": authority._codex_handoff_limits(),
        "capability_policy": {
            "production_arm_enabled": True,
            "real_resume_enabled": True,
            "synthetic_only": False,
        },
        "branch_template_policy": authority._codex_handoff_branch_template_policy(),
        "idle_completion_policy": authority._codex_handoff_completion_policy(),
        "external_supervisor_handoff_policy": (
            authority._codex_handoff_external_supervisor_policy()
        ),
        "operational_source": {
            "schema": "aanca.operational-handoff-source.v1",
            "source_path": r"C:\AANCA\operational_handoff.py",
            "source_size_bytes": 1000,
            "source_sha256": "a" * 64,
            "source_inventory_path": r"C:\AANCA\source_inventory.json",
            "source_inventory_file_sha256": "b" * 64,
            "source_inventory_payload_sha256": "c" * 64,
            "source_inventory_root_sha256": "d" * 64,
            "independent_audit_receipt_path": r"C:\AANCA\audit.json",
            "independent_audit_receipt_sha256": "e" * 64,
            "authority_spec_path": r"C:\AANCA\authority_spec.json",
            "authority_spec_file_sha256": "f" * 64,
            "authority_spec_payload_sha256": "1" * 64,
            "synthetic_inventory_path": r"C:\AANCA\synthetic_inventory.json",
            "synthetic_inventory_file_sha256": "2" * 64,
            "synthetic_inventory_root_sha256": "3" * 64,
            "synthetic_inventory_size_bytes": 2000,
            "synthetic_gate_source_path": r"C:\AANCA\synthetic_gate.py",
            "synthetic_gate_source_sha256": "4" * 64,
            "synthetic_gate_source_size_bytes": 3000,
        },
    }
    return {
        "schema": authority.CODEX_HANDOFF_BASE_OPERATIONAL_SCHEMA,
        "payload": payload,
        "payload_sha256": authority.canonical_json_sha256(payload),
    }


def _creation(base: dict[str, Any], output_path: Path) -> dict[str, Any]:
    nonce = "4" * 64
    payload = {
        "authority_scope": authority.CODEX_HANDOFF_ATTEMPT_CREATION_SCOPE,
        "base_authority_payload_sha256": base["payload_sha256"],
        "session_id": base["payload"]["session_origin"]["session_id"],
        "turn_id": "87654321-4321-1234-9234-cba987654321",
        "marker_nonce_hex": nonce,
        "marker": f"AANCA_CURRENT_SESSION_IDLE_{nonce}",
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
        "attempt_authority_output_path": str(output_path),
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


def _environment(nonce: str) -> tuple[dict[str, Any], dict[str, Any]]:
    supervisor = {
        "LOCALAPPDATA": r"C:\Users\NATAN\AppData\Local",
        "SYSTEMROOT": r"C:\Windows",
        "TEMP": r"C:\Users\NATAN\AppData\Local\Temp",
        "TMP": r"C:\Users\NATAN\AppData\Local\Temp",
        "USERPROFILE": r"C:\Users\NATAN",
    }
    child = {**supervisor, authority.SUPERVISOR_ATTEMPT_NONCE_KEY: nonce}
    envelope = authority.build_expected_launch_environment_envelope_v1(
        attempt_nonce=nonce,
        supervisor_environment=supervisor,
        child_environment=child,
    ).as_dict()
    binding = authority.build_original_confirmatory_process_environment_binding(envelope).as_dict()
    return envelope, binding


def _canonical_line(value: Any) -> bytes:
    payload = authority.canonical_json_bytes(value)
    if type(payload) is not bytes:
        raise TypeError("canonical JSON helper did not return bytes")
    return payload + b"\n"


def _runtime_shape() -> dict[str, Any]:
    spec: dict[str, Any] = {key: None for key in terminal._CANONICAL_SUPERVISOR_SPEC_FIELDS}
    spec.update(
        {
            "schema_version": 3,
            "policy": authority.SUPERVISOR_V3_POLICY,
            "codex": None,
            "handoff_session": None,
            "max_attempt_count": 1,
            "automatic_retry_allowed": False,
            "external_codex_handoff": {
                "policy": authority.EXTERNAL_CODEX_HANDOFF_POLICY,
                "staged_e_intent_path": r"C:\AANCA\state\control_staging\job\e_intent.json",
                "staged_e_intent_file_sha256": "1" * 64,
                "staged_e_intent_core_root_sha256": "2" * 64,
                "attempt_creation_authority_payload_sha256": "3" * 64,
                "attempt_authority_output_path": r"C:\AANCA\state\jobs\job\codex_handoff_attempt_authority.json",
                "terminal_handoff_receipt_output_path": r"C:\AANCA\state\jobs\job\external_codex_terminal_handoff.json",
                "internal_codex_wake_allowed": False,
                "legacy_handoff_session_allowed": False,
                "single_wake_owner": authority.EXTERNAL_CODEX_HANDOFF_SINGLE_WAKE_OWNER,
            },
        }
    )
    return spec


def _control_staging_fixture() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    state_root = Path(r"C:\AANCA\state")
    job_id = "oc-" + "a" * 64
    projection = authority.build_original_confirmatory_control_staging_projection(
        supervisor_state_root=state_root,
        job_id=job_id,
    )
    names = list(authority.CONTROL_STAGING_EXACT_FILE_ALLOWLIST)
    roles = [
        "staging-attempt",
        "e-intent",
        "launch-authorization",
        "supervisor-launch-spec",
        "staging-ready",
    ]
    files = []
    for index, (name, role) in enumerate(zip(names, roles, strict=True), start=1):
        identity = {"path": str(Path(projection["control_staging_dir"]) / name)}
        files.append(
            {
                "role": role,
                "name": name,
                "path": identity["path"],
                "size_bytes": index,
                "file_sha256": f"{index}" * 64,
                "physical_identity": identity,
                "physical_identity_root_sha256": terminal._canonical_sha256(identity),
            }
        )
    authorization = {"path": files[2]["path"], "sha256": files[2]["file_sha256"]}
    spec = {"job_id": job_id, "authorization": authorization}
    payload = {
        "source_path": files[3]["path"],
        "source_file_sha256": files[3]["file_sha256"],
    }
    unsigned = {
        "schema_version": 1,
        "policy": terminal._CONTROL_STAGING_OUTER_BINDING_POLICY,
        "job_id": job_id,
        "supervisor_root": str(state_root),
        "control_staging_root": str(state_root / authority.CONTROL_STAGING_DIRECTORY_NAME),
        "control_staging_dir": projection["control_staging_dir"],
        "control_staging_projection": projection,
        "control_staging_projection_sha256": authority.canonical_json_sha256(projection),
        "expected_complete_leaf_names": names,
        "publication_order": names,
        "file_count": 5,
        "files": files,
        "control_staging_ancestor_lease": {},
        "control_staging_ancestor_lease_root_sha256": "6" * 64,
        "staging_attempt_root_sha256": "7" * 64,
        "staging_ready_root_sha256": "8" * 64,
        "source_path": files[3]["path"],
        "source_size_bytes": files[3]["size_bytes"],
        "source_file_sha256": files[3]["file_sha256"],
        "source_canonical_bytes_sha256": files[3]["file_sha256"],
        "source_bytes_equal_canonical_spec_serialization": True,
        "e_intent_path": files[1]["path"],
        "e_intent_file_sha256": files[1]["file_sha256"],
        "launch_authorization_path": files[2]["path"],
        "launch_authorization_file_sha256": files[2]["file_sha256"],
        "supervisor_process_identity": {},
        "retained_from_before_final_job_creation_through_terminal": True,
        "final_job_creation_owner": "suspended_supervisor_after_resume_v1",
        "pre_ack_final_job_publication_scope": [
            "jobs/<job_id>",
            "run_spec.json",
            "q_e_custody_receipt.json",
        ],
        "pre_ack_metadata_only_publication_allowed": True,
        "pre_ack_scientific_process_launch_allowed": False,
        "q_e_ack_required_before_scientific_process_launch": True,
        "automatic_retry_allowed": False,
        "adoption_allowed": False,
        "cleanup_allowed": False,
    }
    binding = {**unsigned, "binding_root_sha256": terminal._canonical_sha256(unsigned)}
    q = {
        "control_staging_projection": projection,
        "control_staging_projection_sha256": authority.canonical_json_sha256(projection),
        "supervisor_release": {"supervisor_state_root": str(state_root)},
    }
    e = {"job": {"supervisor_job_dir": projection["final_job_dir"]}}
    return binding, payload, spec, {"q": q, "e": e}


def test_admitted_authority_is_exact_and_bootstrap_rejects_byte_tamper() -> None:
    authority_path = (
        PACKAGE_IMPORT_ROOT / "histo_audit/workflows/original_confirmatory_capsule_authority.py"
    )
    authority_bytes = authority_path.read_bytes()
    assert hashlib.sha256(authority_bytes).hexdigest() == bootstrap.ADMITTED_AUTHORITY_SHA256
    payloads = {
        bootstrap.CAPSULE_POLICY_MEMBER: bootstrap._canonical_json_line(bootstrap._CAPSULE_POLICY),
        bootstrap.ENTRY_CONTRACT_MEMBER: bootstrap._canonical_json_line(bootstrap._ENTRY_CONTRACT),
        bootstrap.AUTHORITY_MEMBER: authority_bytes,
    }
    bootstrap._verify_control_members(payloads)
    payloads[bootstrap.AUTHORITY_MEMBER] = authority_bytes + b"\n"
    with pytest.raises(bootstrap.CapsuleBootstrapError):
        bootstrap._verify_control_members(payloads)


def test_source_root_resolution_supports_external_and_repo_test_layouts(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (external / "capsule_bootstrap.py").touch()
    assert _resolve_source_root(external) == external

    repository = tmp_path / "repository"
    repository_tests = repository / "tests"
    repository_tests.mkdir(parents=True)
    (repository / "capsule_bootstrap.py").touch()
    assert _resolve_source_root(repository_tests) == repository

    (repository_tests / "capsule_bootstrap.py").touch()
    with pytest.raises(RuntimeError):
        _resolve_source_root(repository_tests)
    with pytest.raises(RuntimeError):
        _resolve_source_root(tmp_path / "missing")


def test_exact_contract_inventories_match_admitted_authority() -> None:
    assert bootstrap._Q_FIELDS == authority._Q_REPLACEMENT_V2_FIELDS
    assert set(bootstrap._Q_BASE_AUTHORITY_FIELDS) == set(authority._Q_BASE_AUTHORITY_FIELDS)
    assert bootstrap._E_FIELDS == authority._E_INTENT_FIELDS
    assert len(bootstrap._Q_FIELDS) == 20
    assert len(bootstrap._E_FIELDS) == 23
    assert len(bootstrap._SUPERVISOR_RUN_SPEC_PAYLOAD_FIELDS) == 9
    assert len(bootstrap._SUPERVISOR_CANONICAL_SPEC_FIELDS) == 57
    assert bootstrap._SUPERVISOR_CANONICAL_SPEC_FIELDS == (
        terminal._CANONICAL_SUPERVISOR_SPEC_FIELDS
    )


def test_q17_and_e20_are_fail_closed() -> None:
    q17 = {key: None for key in bootstrap._Q_FIELDS}
    for key in (
        "codex_handoff_base_authority",
        "codex_handoff_attempt_creation_authority_payload_sha256",
        "expected_launch_environment",
    ):
        q17.pop(key)
    e20 = {key: None for key in bootstrap._E_FIELDS}
    for key in (
        "codex_handoff_attempt_creation_authority",
        "q_codex_handoff_base_authority_payload_sha256",
        "q_codex_handoff_attempt_creation_authority_payload_sha256",
    ):
        e20.pop(key)
    with pytest.raises(bootstrap.CapsuleBootstrapError):
        bootstrap._exact_object(q17, fields=bootstrap._Q_FIELDS, label="Q17")
    with pytest.raises(bootstrap.CapsuleBootstrapError):
        bootstrap._exact_object(e20, fields=bootstrap._E_FIELDS, label="E20")


def test_schema3_envelope_passes_and_schema2_fails() -> None:
    payload = {"closed": True}
    for schema, succeeds in ((3, True), (2, False), (True, False), (3.0, False)):
        envelope = {
            "schema_version": schema,
            "payload": payload,
            "payload_sha256": terminal._canonical_sha256(payload),
        }
        held = SimpleNamespace(payload=_canonical_line(envelope), file_sha256="a" * 64)
        if succeeds:
            assert terminal._decode_supervisor_envelope(held, role="test")[0] == payload
        else:
            with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
                terminal._decode_supervisor_envelope(held, role="test")


def test_stale_release_and_legacy_wake_fields_are_rejected() -> None:
    stale = {key: None for key in bootstrap._SUPERVISOR_RELEASE_FIELDS}
    for key in (
        "external_codex_handoff_policy",
        "external_codex_handoff_authority_spec_file_sha256",
        "external_codex_handoff_authority_spec_canonical_root_sha256",
        "internal_codex_wake_disposition",
    ):
        stale.pop(key)
    stale.update(
        {
            "codex_terminal_wake_prompt_render_policy": None,
            "codex_terminal_wake_prompt_template_sha256": None,
            "codex_terminal_wake_prompt_template_root_sha256": None,
            "codex_terminal_wake_prompt_template_projection": None,
        }
    )
    with pytest.raises(bootstrap.CapsuleBootstrapError):
        bootstrap._exact_object(
            stale,
            fields=bootstrap._SUPERVISOR_RELEASE_FIELDS,
            label="stale release",
        )
    for path in (
        SOURCE_ROOT / "capsule_bootstrap.py",
        PACKAGE_IMPORT_ROOT / "histo_audit/workflows/original_confirmatory_capsule_terminal.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "SUPERVISOR_V2_POLICY" not in source
        assert "aanca_event_driven_unattended_supervisor_v2" not in source
        assert "codex_terminal_wake_prompt" not in source


@pytest.mark.skipif(os.name != "nt", reason="Codex carrier seals Windows session paths")
def test_operational_base_creation_and_exact_environment_crosslinks() -> None:
    base = _base_authority()
    canonical_base = bootstrap._require_codex_handoff_base_authority(
        base, project_root=PROJECT_ROOT
    )
    output = Path(r"C:\AANCA\state\jobs\job-1\codex_handoff_attempt_authority.json")
    creation = _creation(base, output)
    canonical_creation = bootstrap._require_codex_handoff_attempt_creation_authority(
        creation,
        base_authority=canonical_base,
        expected_output_path=output,
    )
    assert canonical_creation == creation
    bad_creation = copy.deepcopy(creation)
    bad_creation["payload"]["attempt_authority_output_path"] = (
        r"C:\AANCA\state\jobs\other\codex_handoff_attempt_authority.json"
    )
    bad_creation["payload_sha256"] = authority.canonical_json_sha256(bad_creation["payload"])
    with pytest.raises(bootstrap.CapsuleBootstrapError):
        bootstrap._require_codex_handoff_attempt_creation_authority(
            bad_creation,
            base_authority=canonical_base,
            expected_output_path=output,
        )

    nonce = "a" * 64
    environment, binding = _environment(nonce)
    canonical_environment = bootstrap._require_expected_launch_environment(
        environment, attempt_nonce=nonce
    )
    assert (
        bootstrap._require_process_environment_binding(
            binding,
            expected_environment=canonical_environment,
        )
        == binding
    )
    bad_environment = copy.deepcopy(environment)
    bad_environment["child_environment"]["EXTRA"] = "forbidden"
    with pytest.raises(bootstrap.CapsuleBootstrapError):
        bootstrap._require_expected_launch_environment(bad_environment, attempt_nonce=nonce)


def test_external_handoff_rejects_internal_wake_and_crosslink_tamper() -> None:
    base = _base_authority()
    job_dir = Path(r"C:\AANCA\state\jobs\job-1")
    creation = _creation(base, job_dir / "codex_handoff_attempt_authority.json")
    control = {
        "final_job_dir": str(job_dir),
        "e_intent_path": r"C:\AANCA\state\control_staging\job-1\e_intent.json",
    }
    handoff = {
        "policy": authority.EXTERNAL_CODEX_HANDOFF_POLICY,
        "staged_e_intent_path": control["e_intent_path"],
        "staged_e_intent_file_sha256": "1" * 64,
        "staged_e_intent_core_root_sha256": "2" * 64,
        "attempt_creation_authority_payload_sha256": creation["payload_sha256"],
        "attempt_authority_output_path": creation["payload"]["attempt_authority_output_path"],
        "terminal_handoff_receipt_output_path": str(
            job_dir / "external_codex_terminal_handoff.json"
        ),
        "internal_codex_wake_allowed": False,
        "legacy_handoff_session_allowed": False,
        "single_wake_owner": authority.EXTERNAL_CODEX_HANDOFF_SINGLE_WAKE_OWNER,
    }
    bootstrap._require_external_codex_handoff(
        handoff,
        control_staging=control,
        e_file_sha256="1" * 64,
        e_core_sha256="2" * 64,
        attempt_creation=creation,
    )
    for field, value in (
        ("internal_codex_wake_allowed", True),
        ("staged_e_intent_file_sha256", "f" * 64),
    ):
        bad = {**handoff, field: value}
        with pytest.raises(bootstrap.CapsuleBootstrapError):
            bootstrap._require_external_codex_handoff(
                bad,
                control_staging=control,
                e_file_sha256="1" * 64,
                e_core_sha256="2" * 64,
                attempt_creation=creation,
            )


def test_runtime_shape_rejects_v2_non_null_codex_and_legacy_session() -> None:
    valid = _runtime_shape()
    terminal._require_v3_external_runtime_shape(valid)
    mutations: list[dict[str, Any]] = [
        {"schema_version": 2},
        {"schema_version": True},
        {"schema_version": 3.0},
        {"policy": "aanca_event_driven_unattended_supervisor_v2"},
        {"codex": {}},
        {"handoff_session": "legacy"},
        {"max_attempt_count": True},
    ]
    for mutation in mutations:
        bad = {**valid, **mutation}
        with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
            terminal._require_v3_external_runtime_shape(bad)


@pytest.mark.skipif(os.name != "nt", reason="control-staging carrier seals Windows paths")
def test_control_staging_binding_tamper_is_fail_closed() -> None:
    binding, payload, spec, values = _control_staging_fixture()
    terminal._validate_control_staging_run_spec_binding(
        binding,
        run_spec_payload=payload,
        spec=spec,
        q=values["q"],
        e=values["e"],
        e_file_sha256=binding["e_intent_file_sha256"],
    )
    bad = copy.deepcopy(binding)
    bad["source_file_sha256"] = "f" * 64
    with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
        terminal._validate_control_staging_run_spec_binding(
            bad,
            run_spec_payload=payload,
            spec=spec,
            q=values["q"],
            e=values["e"],
            e_file_sha256=binding["e_intent_file_sha256"],
        )


def test_prearm_process_absence_accepts_only_exact_int_schema3() -> None:
    spec = {
        "program_path": r"C:\Python\python.exe",
        "program_sha256": "1" * 64,
        "argv": ["python.exe", "-I", "-B"],
    }
    context = SimpleNamespace(
        spec=spec,
        scientific_command=SimpleNamespace(command_sha256="2" * 64),
    )
    raw = {
        "schema_version": 3,
        "policy": "exact_argv_and_protected_process_absence_v1",
        "observed_at_utc": "2026-08-04T12:00:00.000000Z",
        "inventory_process_count": 0,
        "target_program_path": spec["program_path"],
        "target_program_sha256": spec["program_sha256"],
        "target_command_sha256": "2" * 64,
        "target_argv_sha256": terminal._canonical_sha256(spec["argv"]),
        "exact_command_matches": [],
        "protected_marker_matches": [],
        "absence_verified": True,
    }
    assert terminal._validate_prearm_process_absence(raw, context=context) == raw
    for schema in (2, True, 3.0):
        with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
            terminal._validate_prearm_process_absence(
                {**raw, "schema_version": schema}, context=context
            )


def test_q_e_custody_uses_staged_e_and_rejects_legacy_final_job_e(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    job_id = "oc-" + "a" * 64
    job_directory = state_root / "jobs" / job_id
    staging_root = state_root / bootstrap._CONTROL_STAGING_DIRECTORY_NAME
    staging_directory = staging_root / job_id
    job_directory.mkdir(parents=True)
    staging_directory.mkdir(parents=True)
    staged_e = staging_directory / bootstrap._E_INTENT_FILENAME

    def record(path: Path) -> dict[str, Any]:
        value = os.lstat(path)
        volume, file_id = bootstrap._native_path_identity(path, directory=True)
        return {
            "path": str(path),
            "volume_serial_number": volume,
            "file_id_128": file_id,
            "file_attributes": bootstrap._file_attributes(value),
            "reparse_point": False,
        }

    records = [record(state_root), record(staging_root), record(staging_directory)]
    lease = {
        "schema_version": 1,
        "policy": bootstrap._Q_E_CUSTODY_E_ANCESTOR_LEASE_POLICY,
        "supervisor_root": str(state_root),
        "records": records,
        "record_count": 3,
        "records_root_sha256": authority.canonical_json_sha256(records),
        "directory_access_mask": bootstrap._Q_E_CUSTODY_ANCESTOR_TARGET_ACCESS_MASK,
        "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        "delete_share": False,
        "write_access": False,
        "owner_process_identity_required": True,
        "handle_slot_per_record_required": True,
        "continuous_overlap_through_independent_verification_required": True,
        "continuous_overlap_into_supervisor_required": True,
        "acquisition_disposition": bootstrap._Q_E_CUSTODY_E_ANCESTOR_DISPOSITION,
    }
    assert (
        bootstrap._require_q_e_e_ancestor_lease(
            lease,
            supervisor_job_directory=job_directory,
            e_intent_path=staged_e,
        )["records"]
        == records
    )
    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="outside its exact control-staging directory",
    ):
        bootstrap._require_q_e_e_ancestor_lease(
            lease,
            supervisor_job_directory=job_directory,
            e_intent_path=job_directory / bootstrap._E_INTENT_FILENAME,
        )
