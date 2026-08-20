from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest
from carrier_import_guard import CARRIER_ROOT, PACKAGE_IMPORT_ROOT, import_exact

bootstrap = import_exact("capsule_bootstrap", CARRIER_ROOT / "capsule_bootstrap.py")
capsule_authority = import_exact(
    "histo_audit.workflows.original_confirmatory_capsule_authority",
    PACKAGE_IMPORT_ROOT
    / "histo_audit"
    / "workflows"
    / "original_confirmatory_capsule_authority.py",
)
_capsule_builder = import_exact("capsule_builder", CARRIER_ROOT / "capsule_builder.py")
MANIFEST_NAME = _capsule_builder.MANIFEST_NAME
CapsuleBuildError = _capsule_builder.CapsuleBuildError
CapsuleBuildResult = _capsule_builder.CapsuleBuildResult
CapsuleByteBuild = _capsule_builder.CapsuleByteBuild
build_capsule_bytes = _capsule_builder.build_capsule_bytes
build_project_capsule_bytes = _capsule_builder.build_project_capsule_bytes
discover_project_payload = _capsule_builder.discover_project_payload
source_inventory = _capsule_builder.source_inventory


def _authority_source_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    candidates = [
        path
        for path in (
            root / "histo_audit" / "workflows" / "original_confirmatory_capsule_authority.py",
            root
            / "src"
            / "histo_audit"
            / "workflows"
            / "original_confirmatory_capsule_authority.py",
        )
        if path.is_file()
    ]
    if len(candidates) != 1:
        raise AssertionError("expected exactly one external or repository authority source")
    return candidates[0]


def _synthetic_codex_handoff_base(project_root: Path) -> dict[str, object]:
    session_id = "12345678-1234-4234-9234-123456789abc"
    operational_root = project_root / "synthetic-codex-handoff"
    payload: dict[str, object] = {
        "authority_scope": capsule_authority.CODEX_HANDOFF_BASE_OPERATIONAL_SCOPE,
        "session_origin": {
            "session_id": session_id,
            "session_jsonl_path": str(operational_root / "session.jsonl"),
            "expected_cwd": str(project_root),
            "first_record": {
                "record_type": "session_meta",
                "payload_id": session_id,
                "payload_session_id": session_id,
                "payload_cli_version": "codex-cli synthetic",
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
            "path": str(operational_root / "codex.exe"),
            "size_bytes": 123,
            "sha256": "3" * 64,
            "version_stdout": "codex-cli synthetic",
        },
        "resume_command_policy": capsule_authority._codex_handoff_resume_command_policy(
            operational=True
        ),
        "limits": capsule_authority._codex_handoff_limits(),
        "capability_policy": {
            "production_arm_enabled": True,
            "real_resume_enabled": True,
            "synthetic_only": False,
        },
        "branch_template_policy": capsule_authority._codex_handoff_branch_template_policy(),
        "idle_completion_policy": capsule_authority._codex_handoff_completion_policy(),
        "external_supervisor_handoff_policy": (
            capsule_authority._codex_handoff_external_supervisor_policy()
        ),
        "operational_source": {
            "schema": "aanca.operational-handoff-source.synthetic.v1",
            "source_path": str(operational_root / "operational_handoff.py"),
            "source_size_bytes": 1000,
            "source_sha256": "a" * 64,
            "source_inventory_path": str(operational_root / "source_inventory.json"),
            "source_inventory_file_sha256": "b" * 64,
            "source_inventory_payload_sha256": "c" * 64,
            "source_inventory_root_sha256": "d" * 64,
            "independent_audit_receipt_path": str(operational_root / "audit.json"),
            "independent_audit_receipt_sha256": "e" * 64,
            "authority_spec_path": str(operational_root / "authority_spec.json"),
            "authority_spec_file_sha256": "f" * 64,
            "authority_spec_payload_sha256": "1" * 64,
            "synthetic_inventory_path": str(operational_root / "synthetic_inventory.json"),
            "synthetic_inventory_file_sha256": "2" * 64,
            "synthetic_inventory_root_sha256": "3" * 64,
            "synthetic_inventory_size_bytes": 2000,
            "synthetic_gate_source_path": str(operational_root / "synthetic_gate.py"),
            "synthetic_gate_source_sha256": "4" * 64,
            "synthetic_gate_source_size_bytes": 3000,
        },
    }
    return {
        "schema": capsule_authority.CODEX_HANDOFF_BASE_OPERATIONAL_SCHEMA,
        "payload": payload,
        "payload_sha256": _authority_root(payload),
    }


def _synthetic_codex_handoff_creation(
    base: dict[str, object],
    *,
    output_path: Path,
) -> dict[str, object]:
    base_payload = base["payload"]
    assert isinstance(base_payload, dict)
    session = base_payload["session_origin"]
    branch = base_payload["branch_template_policy"]
    source = base_payload["operational_source"]
    assert isinstance(session, dict)
    assert isinstance(branch, dict)
    assert isinstance(source, dict)
    nonce = "4" * 64
    payload: dict[str, object] = {
        "authority_scope": capsule_authority.CODEX_HANDOFF_ATTEMPT_CREATION_SCOPE,
        "base_authority_payload_sha256": base["payload_sha256"],
        "session_id": session["session_id"],
        "turn_id": "87654321-4321-1234-9234-cba987654321",
        "marker_nonce_hex": nonce,
        "marker": f"AANCA_CURRENT_SESSION_IDLE_{nonce}",
        "success_template_policy_root_sha256": branch["success_template_policy_root_sha256"],
        "diagnosis_template_policy_root_sha256": branch["diagnosis_template_policy_root_sha256"],
        "authority_spec_payload_sha256": source["authority_spec_payload_sha256"],
        "arm_algorithm_contract_root_sha256": _authority_root(
            capsule_authority._codex_handoff_arm_algorithm_contract()
        ),
        "attempt_authority_output_path": str(output_path),
        "attempt_authority_schema": capsule_authority.CODEX_HANDOFF_ATTEMPT_SCHEMA,
        "arm_algorithm": capsule_authority.CODEX_HANDOFF_ARM_ALGORITHM,
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
        "schema": capsule_authority.CODEX_HANDOFF_ATTEMPT_CREATION_SCHEMA,
        "payload": payload,
        "payload_sha256": _authority_root(payload),
    }


def _synthetic_launch_environment(
    attempt_nonce: str,
) -> tuple[dict[str, object], dict[str, object]]:
    # This authority deliberately seals a Windows launch contract even when its
    # pure validation tests run on a POSIX CI host. Do not derive these values
    # from the host's home or temporary directory syntax.
    user_profile = str(Path.home()) if os.name == "nt" else r"C:\Users\NATAN"
    local_app_data = user_profile + r"\AppData\Local"
    supervisor_environment = {
        "LOCALAPPDATA": local_app_data,
        "SYSTEMROOT": r"C:\Windows",
        "TEMP": local_app_data + r"\Temp",
        "TMP": local_app_data + r"\Temp",
        "USERPROFILE": user_profile,
    }
    child_environment = {
        **supervisor_environment,
        capsule_authority.SUPERVISOR_ATTEMPT_NONCE_KEY: attempt_nonce,
    }
    environment = capsule_authority.build_expected_launch_environment_envelope_v1(
        attempt_nonce=attempt_nonce,
        supervisor_environment=supervisor_environment,
        child_environment=child_environment,
    ).as_dict()
    binding = capsule_authority.build_original_confirmatory_process_environment_binding(
        environment
    ).as_dict()
    return environment, binding


def _materialize_private_test_build(
    build: CapsuleByteBuild,
    output_path: str | Path,
) -> CapsuleBuildResult:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(build.archive_bytes)
    os.chmod(destination, stat.S_IREAD)
    return CapsuleBuildResult(
        output_path=destination,
        size_bytes=build.size_bytes,
        sha256=build.sha256,
        internal_manifest_sha256=build.internal_manifest_sha256,
        records_root_sha256=build.records_root_sha256,
        entry_count=build.entry_count,
        payload_size_bytes=build.payload_size_bytes,
    )


def build_capsule(
    *,
    members,
    expected_inventory,
    output_path,
) -> CapsuleBuildResult:
    return _materialize_private_test_build(
        build_capsule_bytes(
            members=members,
            expected_inventory=expected_inventory,
        ),
        output_path,
    )


def build_project_capsule(
    *,
    package_root,
    bootstrap_path,
    policy_path,
    entry_contract_path,
    expected_inventory,
    output_path,
) -> CapsuleBuildResult:
    return _materialize_private_test_build(
        build_project_capsule_bytes(
            package_root=package_root,
            bootstrap_path=bootstrap_path,
            policy_path=policy_path,
            entry_contract_path=entry_contract_path,
            expected_inventory=expected_inventory,
        ),
        output_path,
    )


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _canonical_json_line(value: object) -> bytes:
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


def _authority_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _authority_json_line(value: object) -> bytes:
    return _authority_json(value) + b"\n"


def _authority_root(value: object) -> str:
    return hashlib.sha256(_authority_json(value)).hexdigest()


def _supervisor_file_root(value: object) -> str:
    return hashlib.sha256(_authority_json_line(value)).hexdigest()


def _self_hashed(value: dict[str, object], field: str) -> dict[str, object]:
    return {**value, field: _authority_root(value)}


def _synthetic_tree(root: Path) -> dict[str, Path]:
    package = root / "src" / "histo_audit"
    bootstrap_source = Path(__file__).resolve().parents[1] / "capsule_bootstrap.py"
    files = {
        "init": _write(package / "__init__.py", b'__version__ = "synthetic"\n'),
        "experiment_init": _write(package / "experiment" / "__init__.py", b""),
        "completion": _write(
            package / "experiment" / "confirmatory_completion.py",
            b"VALUE = 'completion'\n",
        ),
        "runner": _write(
            package / "experiment" / "original_confirmatory_runner_core.py",
            b"VALUE = 'runner'\n",
        ),
        "authority": _write(
            package / "workflows" / "original_confirmatory_capsule_authority.py",
            _authority_source_path().read_bytes(),
        ),
        "workflows_init": _write(package / "workflows" / "__init__.py", b""),
        "entry": _write(
            package / "workflows" / "original_confirmatory_capsule_entry.py",
            (
                b"import __main__\n"
                b"import json\n"
                b"import os\n"
                b"from histo_audit.models import cnn\n"
                b"def _dispatch_original_confirmatory_capsule(argv):\n"
                b"    __main__._arm_original_confirmatory_e_claim_after_full_prevalidation()\n"
                b"    if argv[0] == 'run-confirmatory':\n"
                b"        descriptor, _path = "
                b"__main__._take_original_confirmatory_e_claim_handle()\n"
                b"        os.write(descriptor, b'{\"synthetic\":true}\\n')\n"
                b"        os.fsync(descriptor)\n"
                b"    else:\n"
                b"        descriptor, _path, _sha256, _size = "
                b"__main__._take_original_confirmatory_e_claim_read_handle()\n"
                b"    os.close(descriptor)\n"
                b"    print(json.dumps({'argv': list(argv), "
                b"'cnn_origin': cnn.__spec__.origin, 'entry_origin': __spec__.origin}, "
                b"sort_keys=True, separators=(',', ':')))\n"
                b"    return 0\n"
            ),
        ),
        "terminal": _write(
            package / "workflows" / "original_confirmatory_capsule_terminal.py",
            b"VALUE = 'terminal'\n",
        ),
        "models_init": _write(package / "models" / "__init__.py", b""),
        "dependency": _write(package / "models" / "cnn.py", b"VALUE = 'cnn'\n"),
        "bootstrap": _write(root / "capsule_bootstrap.py", bootstrap_source.read_bytes()),
        "policy": _write(
            root / "capsule_policy.json",
            _canonical_json_line(
                {
                    "allowed_modes": [
                        "run-confirmatory",
                        "verify-preterminal",
                        "verify-terminal",
                    ],
                    "automatic_retry_allowed": False,
                    "generic_histo_audit_cli_allowed": False,
                    "policy": "original_confirmatory_sealed_execution_capsule_v1",
                    "schema_version": 1,
                    "source_policy": "every_regular_python_file_under_final_histo_audit_tree_v1",
                }
            ),
        ),
        "contract": _write(
            root / "entry_contract.json",
            _canonical_json_line(
                {
                    "allowed_modes": [
                        "run-confirmatory",
                        "verify-preterminal",
                        "verify-terminal",
                    ],
                    "contract_status": "ready",
                    "dispatcher": (
                        "histo_audit.workflows.original_confirmatory_capsule_entry:"
                        "_dispatch_original_confirmatory_capsule"
                    ),
                    "policy": "original_confirmatory_execution_capsule_entry_contract_v1",
                    "schema_version": 1,
                }
            ),
        ),
    }
    files["package"] = package
    return files


def _discover(files: dict[str, Path]):
    return discover_project_payload(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
    )


@pytest.mark.parametrize(
    ("directory", "expected_flags"),
    [
        (False, 0x00200000),
        (True, 0x02200000),
    ],
)
def test_native_path_identity_open_masks_deny_delete_share(
    directory: bool,
    expected_flags: int,
) -> None:
    desired_access, share_mode, flags = bootstrap._native_path_identity_open_masks(
        directory=directory
    )

    assert desired_access == 0x80000000
    assert share_mode == 0x00000001 | 0x00000002
    assert share_mode & 0x00000004 == 0
    assert flags == expected_flags


def _stage_content_addressed_capsule(
    result,
    root: Path,
) -> Path:
    destination = _stage_raw_content_addressed_capsule(
        result.output_path.read_bytes(),
        root,
    )
    assert destination.parent.name == result.sha256
    return destination


def _stage_raw_content_addressed_capsule(payload: bytes, root: Path) -> Path:
    sha256 = hashlib.sha256(payload).hexdigest()
    destination = root.parent / "artifacts" / root.name / sha256 / "original_confirmatory.pyz"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)
    os.chmod(destination, stat.S_IREAD)
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == sha256
    assert os.stat(destination).st_nlink == 1
    return destination


def _lease_record(path: Path) -> dict[str, object]:
    value = os.lstat(path)
    volume, file_id = bootstrap._native_path_identity(path, directory=True)
    return {
        "path": str(path),
        "volume_serial_number": volume,
        "file_id_128": file_id,
        "file_attributes": bootstrap._file_attributes(value),
        "reparse_point": False,
    }


def _ancestor_lease(
    paths: list[Path],
    *,
    interpreter: bool,
    policy: str | None = None,
) -> dict[str, object]:
    records = [_lease_record(path) for path in paths]
    value: dict[str, object] = {
        "schema_version": 1,
        "policy": policy
        or (
            "original_confirmatory_interpreter_ancestor_lease_v1"
            if interpreter
            else "original_confirmatory_capsule_ancestor_lease_v1"
        ),
        "anchor_path": str(paths[0]),
        "records": records,
        "record_count": len(records),
        "records_root_sha256": _authority_root(records),
        "directory_access_mask": 0x80000080,
        "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        "delete_share": False,
        "write_access": False,
        "retained_through_each_exact_phase_launch": True,
        "owner_process_identity_required": True,
        "handle_slot_per_record_required": True,
        "acquisition_disposition": (
            "directory_handles_opened_before_first_phase_createprocess_"
            "retained_through_all_phase_waitforexit_v1"
        ),
    }
    return value


def _leaf_lease(
    path: Path,
    *,
    interpreter: bool,
    policy: str | None = None,
) -> dict[str, object]:
    descriptor = bootstrap._open_capsule_no_follow(path)
    try:
        digest, size_bytes, _identity = bootstrap._hash_held_file(descriptor)
        volume, file_id = bootstrap._native_file_identity_from_fd(descriptor)
    finally:
        os.close(descriptor)
    value = os.lstat(path)
    lease: dict[str, object] = {
        "schema_version": 1,
        "policy": policy
        or (
            "original_confirmatory_interpreter_retained_file_lease_v1"
            if interpreter
            else "original_confirmatory_capsule_retained_file_lease_v1"
        ),
        "path": str(path),
        "volume_serial_number": volume,
        "file_id_128": file_id,
        "size_bytes": size_bytes,
        "sha256": digest,
        "file_attributes": bootstrap._file_attributes(value),
        "link_count": 1,
        "named_alternate_data_streams": [],
        "opened_without_reparse_follow": True,
        "access_mask": 0x80000000,
        "share_access": ["FILE_SHARE_READ"],
        "write_access": False,
        "delete_access": False,
        "retained_through_each_exact_phase_launch": True,
        "owner_process_identity_required": True,
        "handle_slot_required": True,
        "acquisition_disposition": (
            "opened_before_first_phase_createprocess_retained_through_all_phase_waitforexit_v1"
        ),
    }
    lease["regular_file" if interpreter else "read_only"] = True
    return lease


def _terminal_client_physical_identity(path: Path) -> dict[str, object]:
    descriptor = bootstrap._open_capsule_no_follow(path)
    try:
        digest, size_bytes, _identity = bootstrap._hash_held_file(descriptor)
        volume, file_id = bootstrap._native_file_identity_from_fd(descriptor)
        value = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    return {
        "schema_version": 1,
        "policy": "aanca_no_follow_physical_file_identity_v1",
        "role": "terminal-client-launcher",
        "path": str(path),
        "volume_serial_number": volume,
        "file_id_128": file_id,
        "device": value.st_dev,
        "inode": value.st_ino,
        "size_bytes": size_bytes,
        "mode": stat.S_IMODE(value.st_mode),
        "file_attributes": bootstrap._file_attributes(value),
        "regular_file": stat.S_ISREG(value.st_mode),
        "read_only": stat.S_IMODE(value.st_mode) & 0o222 == 0,
        "link_count": value.st_nlink,
        "modified_time_ns": value.st_mtime_ns,
        "changed_time_ns": value.st_ctime_ns,
        "sha256": digest,
        "named_alternate_data_streams": [],
        "opened_without_reparse_follow": True,
        "share_access": ["FILE_SHARE_READ"],
    }


def _q_e_control_physical_identity(path: Path) -> dict[str, object]:
    descriptor = bootstrap._open_capsule_no_follow(path)
    try:
        digest, size_bytes, _identity = bootstrap._hash_held_file(descriptor)
        volume, file_id = bootstrap._native_file_identity_from_fd(descriptor)
        value = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    return {
        "schema_version": 1,
        "policy": "original_confirmatory_control_publication_physical_identity_v1",
        "path": str(path),
        "volume_serial_number": volume,
        "file_id_128": file_id,
        "size_bytes": size_bytes,
        "sha256": digest,
        "file_attributes": bootstrap._file_attributes(value),
        "regular_file": True,
        "read_only": True,
        "link_count": 1,
        "named_alternate_data_streams": [],
        "opened_without_reparse_follow": True,
        "share_access": ["FILE_SHARE_READ"],
        "write_handle_retained": False,
        "delete_access": False,
    }


def _control_staging_physical_identity(
    path: Path,
    *,
    role: str,
) -> dict[str, object]:
    descriptor = bootstrap._open_capsule_no_follow(path)
    try:
        digest, size_bytes, _identity = bootstrap._hash_held_file(descriptor)
        volume, file_id = bootstrap._native_file_identity_from_fd(descriptor)
    finally:
        os.close(descriptor)
    value = os.lstat(path)
    return {
        "schema_version": 1,
        "policy": bootstrap._NO_FOLLOW_PHYSICAL_FILE_IDENTITY_POLICY,
        "role": role,
        "path": str(path),
        "volume_serial_number": volume,
        "file_id_128": file_id,
        "device": value.st_dev,
        "inode": value.st_ino,
        "size_bytes": size_bytes,
        "mode": stat.S_IMODE(value.st_mode),
        "file_attributes": bootstrap._file_attributes(value),
        "regular_file": True,
        "read_only": True,
        "link_count": 1,
        "modified_time_ns": value.st_mtime_ns,
        "changed_time_ns": value.st_ctime_ns,
        "sha256": digest,
        "named_alternate_data_streams": [],
        "opened_without_reparse_follow": True,
        "share_access": ["FILE_SHARE_READ"],
    }


def _control_staging_ancestor_lease(
    state_root: Path,
    staging_directory: Path,
) -> dict[str, object]:
    records = [
        _lease_record(state_root),
        _lease_record(staging_directory.parent),
        _lease_record(staging_directory),
    ]
    return {
        "schema_version": 1,
        "policy": "original_confirmatory_control_staging_ancestor_lease_v1",
        "supervisor_root": str(state_root),
        "records": records,
        "record_count": 3,
        "records_root_sha256": _authority_root(records),
        "directory_access_mask": 0x80000080,
        "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        "delete_share": False,
        "write_access": False,
        "owner_process_identity_required": True,
        "handle_slot_per_record_required": True,
        "continuous_overlap_through_supervisor_stage_ack_required": True,
        "acquisition_disposition": (
            "opened_after_create_new_stage_dir_before_first_leaf_retained_through_stage_ack_v1"
        ),
    }


def _q_e_e_ancestor_lease(
    job_directory: Path,
    e_intent_path: Path,
) -> dict[str, object]:
    supervisor_root = job_directory.parent.parent
    records = [
        _lease_record(supervisor_root),
        _lease_record(e_intent_path.parent.parent),
        _lease_record(e_intent_path.parent),
    ]
    return {
        "schema_version": 1,
        "policy": "original_confirmatory_e_job_publication_ancestor_lease_v1",
        "supervisor_root": str(supervisor_root),
        "records": records,
        "record_count": 3,
        "records_root_sha256": _authority_root(records),
        "directory_access_mask": 0x80000080,
        "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        "delete_share": False,
        "write_access": False,
        "owner_process_identity_required": True,
        "handle_slot_per_record_required": True,
        "continuous_overlap_through_independent_verification_required": True,
        "continuous_overlap_into_supervisor_required": True,
        "acquisition_disposition": (
            "opened_before_e_create_new_retained_through_verifier_and_supervisor_overlap_v1"
        ),
    }


def _q_e_process_identity(
    *, pid: int, program_path: Path, command_seed: bytes
) -> dict[str, object]:
    return {
        "pid": pid,
        "creation_time_100ns": 116444736000000000,
        "creation_time_utc": "1970-01-01T00:00:00.000000Z",
        "program_path": str(program_path),
        "program_sha256": hashlib.sha256(program_path.read_bytes()).hexdigest(),
        "command_sha256": hashlib.sha256(command_seed).hexdigest(),
    }


def _terminal_client_ancestor_lease(root: Path) -> dict[str, object]:
    record = _lease_record(root)
    return {
        "schema_version": 1,
        "policy": "original_confirmatory_terminal_client_launcher_ancestor_lease_v1",
        "supervisor_root": str(root),
        "records": [record],
        "record_count": 1,
        "records_root_sha256": _authority_root([record]),
        "directory_access_mask": 0x80000080,
        "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        "delete_share": False,
        "write_access": False,
        "owner_process_identity_required": True,
        "handle_slot_per_record_required": True,
        "retained_from_q_verification_through_terminal_child_waitforexit": True,
        "acquisition_disposition": (
            "supervisor_root_handle_opened_before_q_verification_retained_through_"
            "terminal_child_waitforexit_v1"
        ),
    }


def _command_derivation_contract() -> dict[str, object]:
    return _self_hashed(
        {
            "schema_version": 1,
            "policy": "original_confirmatory_capsule_command_derivation_contract_v1",
            "projection_policy": ("original_confirmatory_capsule_command_projection_v1"),
            "canonical_file_hash_policy": "canonical_json_line_sha256_v1",
            "canonical_core_hash_policy": ("canonical_json_without_self_field_sha256_v1"),
            "e_file_sha256_flag": "--e-intent-sha256",
            "e_core_sha256_flag": "--e-intent-core-sha256",
            "e_file_sha256_insertion_policy": (
                "append_value_to_terminal_e_file_sha256_flag_then_continue_v1"
            ),
            "e_core_sha256_insertion_policy": (
                "append_value_to_terminal_e_core_sha256_flag_then_continue_v1"
            ),
            "python_isolated_flags": ["-I", "-B"],
            "allowed_modes": [
                "run-confirmatory",
                "verify-preterminal",
                "verify-terminal",
            ],
            "common_tail_flags": [
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
            ],
            "successor_lineage_flag": "--retry-of-run-id",
            "preterminal_suffix_flags": [
                "--run-spec",
                "--launch-intent",
                "--process-started",
                "--preterminal-pin",
            ],
            "terminal_suffix_flags": [
                "--supervisor-terminal",
                "--verifier-stdout",
                "--preterminal-pin",
                "--composed-terminal",
            ],
            "exact_argv_rederivation_required": True,
            "final_command_carrier": ("supervisor_job_spec_create_new_read_only_v1"),
            "post_wait_rederivation_required": True,
            "extra_argv_allowed": False,
            "extra_environment_allowed": False,
        },
        "contract_sha256",
    )


def _synthetic_release_capsule_binding() -> dict[str, object]:
    anchor = Path.cwd()
    dummy_record = {
        "path": str(anchor),
        "volume_serial_number": 0,
        "file_id_128": "0" * 32,
        "file_attributes": 16,
        "reparse_point": False,
    }
    capsule_ancestor = {
        "schema_version": 1,
        "policy": "original_confirmatory_capsule_ancestor_lease_v1",
        "anchor_path": str(anchor),
        "records": [dummy_record],
        "record_count": 1,
        "records_root_sha256": _authority_root([dummy_record]),
        "directory_access_mask": 0x80000080,
        "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        "delete_share": False,
        "write_access": False,
        "retained_through_each_exact_phase_launch": True,
        "owner_process_identity_required": True,
        "handle_slot_per_record_required": True,
        "acquisition_disposition": (
            "directory_handles_opened_before_first_phase_createprocess_"
            "retained_through_all_phase_waitforexit_v1"
        ),
    }
    return {
        "path": str(anchor / "synthetic.pyz"),
        "sha256": hashlib.sha256(b"synthetic-capsule").hexdigest(),
        "internal_manifest_sha256": hashlib.sha256(b"synthetic-manifest").hexdigest(),
        "python_path": str(anchor / ".venv" / "Scripts" / "python.exe"),
        "python_sha256": hashlib.sha256(b"synthetic-python").hexdigest(),
        "runtime_python_path": str(anchor / "runtime" / "python.exe"),
        "runtime_python_sha256": hashlib.sha256(b"synthetic-runtime-python").hexdigest(),
        "python_lease_identity_root_sha256": hashlib.sha256(b"synthetic-python-lease").hexdigest(),
        "python_ancestor_lease_root_sha256": hashlib.sha256(
            b"synthetic-python-ancestor"
        ).hexdigest(),
        "runtime_python_lease_identity_root_sha256": hashlib.sha256(
            b"synthetic-runtime-python-lease"
        ).hexdigest(),
        "runtime_python_ancestor_lease_root_sha256": hashlib.sha256(
            b"synthetic-runtime-python-ancestor"
        ).hexdigest(),
        "capsule_ancestor_lease": capsule_ancestor,
        "capsule_ancestor_lease_root_sha256": _authority_root(capsule_ancestor),
        "capsule_lease_identity_root_sha256": hashlib.sha256(
            b"synthetic-capsule-lease"
        ).hexdigest(),
        "contract_sha256": hashlib.sha256(b"synthetic-capsule-contract").hexdigest(),
        "terminal_release_root_sha256": hashlib.sha256(b"synthetic-terminal").hexdigest(),
    }


def _supervisor_release(
    execution_capsule: dict[str, object] | None = None,
    *,
    supervisor_state_root: Path | None = None,
    terminal_client_identity: dict[str, object] | None = None,
    terminal_client_ancestor: dict[str, object] | None = None,
) -> dict[str, object]:
    capsule = execution_capsule or _synthetic_release_capsule_binding()
    state_root = supervisor_state_root or (Path.cwd() / "synthetic-supervisor-state")
    external_release_root_sha256 = hashlib.sha256(
        b"synthetic-external-control-plane-release"
    ).hexdigest()
    external_publication_id = f"cpr-{'1' * 32}"
    external_attestation_path = (
        state_root.parent
        / bootstrap._EXTERNAL_CONTROL_PLANE_RELEASE_DIRECTORY_NAME
        / bootstrap._EXTERNAL_CONTROL_PLANE_VERIFICATIONS_DIRECTORY_NAME
        / external_publication_id
        / bootstrap._EXTERNAL_CONTROL_PLANE_QUALIFICATION_ATTESTATION_FILENAME
    )
    external_attestation_file_sha256 = hashlib.sha256(
        b"synthetic-release-qualification-attestation-file"
    ).hexdigest()
    external_attestation_root_sha256 = hashlib.sha256(
        b"synthetic-release-qualification-attestation-root"
    ).hexdigest()
    code_root = (
        state_root.parent
        / "synthetic-control-plane"
        / "releases"
        / external_release_root_sha256
        / "supervisor"
    )
    supervisor_source_sha256 = hashlib.sha256(b"synthetic-supervisor").hexdigest()
    supervisor_launcher_sha256 = hashlib.sha256(b"synthetic-launcher").hexdigest()
    supervisor_source_path = code_root / "aanca_supervisor.py"
    supervisor_launcher_path = code_root / "launch_hidden.ps1"
    terminal_client_path = code_root / "terminal_client_launcher_v1.py"
    source_sha256 = hashlib.sha256(b"synthetic-terminal-client").hexdigest()
    source_identity = terminal_client_identity or {
        "schema_version": 1,
        "policy": "aanca_no_follow_physical_file_identity_v1",
        "role": "terminal-client-launcher",
        "path": str(terminal_client_path),
        "volume_serial_number": 0,
        "file_id_128": "0" * 32,
        "device": 0,
        "inode": 0,
        "size_bytes": len(b"synthetic-terminal-client"),
        "mode": 0o444,
        "file_attributes": 1,
        "regular_file": True,
        "read_only": True,
        "link_count": 1,
        "modified_time_ns": 0,
        "changed_time_ns": 0,
        "sha256": source_sha256,
        "named_alternate_data_streams": [],
        "opened_without_reparse_follow": True,
        "share_access": ["FILE_SHARE_READ"],
    }
    root_record = {
        "path": str(code_root),
        "volume_serial_number": 0,
        "file_id_128": "0" * 32,
        "file_attributes": 16,
        "reparse_point": False,
    }
    source_ancestor = terminal_client_ancestor or {
        "schema_version": 1,
        "policy": "original_confirmatory_terminal_client_launcher_ancestor_lease_v1",
        "supervisor_root": str(code_root),
        "records": [root_record],
        "record_count": 1,
        "records_root_sha256": _authority_root([root_record]),
        "directory_access_mask": 0x80000080,
        "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        "delete_share": False,
        "write_access": False,
        "owner_process_identity_required": True,
        "handle_slot_per_record_required": True,
        "retained_from_q_verification_through_terminal_child_waitforexit": True,
        "acquisition_disposition": (
            "supervisor_root_handle_opened_before_q_verification_retained_through_"
            "terminal_child_waitforexit_v1"
        ),
    }
    terminal_client_release = _self_hashed(
        {
            "schema_version": 1,
            "policy": "original_confirmatory_terminal_client_launcher_v1",
            "supervisor_root": str(code_root),
            "source_path": str(terminal_client_path),
            "source_size_bytes": source_identity["size_bytes"],
            "source_sha256": source_identity["sha256"],
            "source_physical_identity": source_identity,
            "source_physical_identity_root_sha256": _authority_root(source_identity),
            "source_ancestor_lease": source_ancestor,
            "source_ancestor_lease_root_sha256": _authority_root(source_ancestor),
            "source_leaf_access_mask": 0x80000000,
            "source_leaf_share_access": ["FILE_SHARE_READ"],
            "source_delete_access": False,
            "source_handle_retained_through_terminal_child_waitforexit": True,
            "program_path": capsule["runtime_python_path"],
            "program_sha256": capsule["runtime_python_sha256"],
            "createprocess_application_path": capsule["runtime_python_path"],
            "createprocess_application_sha256": capsule["runtime_python_sha256"],
            "program_lease_identity_root_sha256": capsule[
                "runtime_python_lease_identity_root_sha256"
            ],
            "program_ancestor_lease_root_sha256": capsule[
                "runtime_python_ancestor_lease_root_sha256"
            ],
            "logical_venv_python_path": capsule["python_path"],
            "logical_venv_python_sha256": capsule["python_sha256"],
            "logical_venv_python_lease_identity_root_sha256": capsule[
                "python_lease_identity_root_sha256"
            ],
            "logical_venv_python_ancestor_lease_root_sha256": capsule[
                "python_ancestor_lease_root_sha256"
            ],
            "runtime_python_path": capsule["runtime_python_path"],
            "runtime_python_sha256": capsule["runtime_python_sha256"],
            "expected_live_image_path": capsule["runtime_python_path"],
            "expected_live_image_sha256": capsule["runtime_python_sha256"],
            "direct_base_runtime_live_parity_policy": (
                bootstrap._DIRECT_BASE_RUNTIME_LIVE_PARITY_POLICY
            ),
            "runtime_python_lease_identity_root_sha256": capsule[
                "runtime_python_lease_identity_root_sha256"
            ],
            "runtime_python_ancestor_lease_root_sha256": capsule[
                "runtime_python_ancestor_lease_root_sha256"
            ],
            "python_isolated_flags": ["-I", "-S", "-B"],
            "python_sys_argv_prefix": [str(terminal_client_path)],
            "process_argv_prefix": [
                capsule["runtime_python_path"],
                "-I",
                "-S",
                "-B",
                str(terminal_client_path),
            ],
            "cwd_binding": "E.project_root",
            "final_argument_order": list(bootstrap._TERMINAL_CLIENT_LAUNCHER_FINAL_ARGUMENT_ORDER),
            "downstream_hash_insertions": list(
                bootstrap._TERMINAL_CLIENT_DOWNSTREAM_PLACEHOLDER_ORDER
            ),
            "command_preimage_policy": (
                "original_confirmatory_terminal_client_launcher_command_v1"
            ),
            "command_sha256_policy": ("canonical_compact_sorted_json_sha256_no_lf_v1"),
            "launch_intent_filename": "terminal_client_launch_intent.json",
            "launch_intent_path_binding": (
                "E.job.supervisor_job_dir/terminal_client_launch_intent.json"
            ),
            "launch_intent_publication_policy": (
                "original_confirmatory_terminal_client_launch_intent_create_new_v1"
            ),
            **bootstrap._TERMINAL_CLIENT_LAUNCHER_STATIC_CONTROL_VALUES,
            "sealed_input_allowlist": ["supervisor_spec", "E"],
            "project_import_allowed": False,
            "inherited_environment_for_child_allowed": False,
            "createprocessw_exact_child_required": True,
            "child_environment_encoding": "sorted_utf16le_double_nul_block_v1",
            "child_environment_source": ("sealed_E.expected_launch_environment.child_environment"),
            "verify_terminal_child_launch_topology": (
                "launcher_base_direct_to_venv_redirector_to_runtime_child_v1"
            ),
            "verify_terminal_immediate_redirector_program_path": capsule["python_path"],
            "verify_terminal_immediate_redirector_program_sha256": capsule["python_sha256"],
            "verify_terminal_runtime_child_program_path": capsule["runtime_python_path"],
            "verify_terminal_runtime_child_program_sha256": capsule["runtime_python_sha256"],
            "launcher_is_runtime_child_grandparent_required": True,
            "same_job_no_breakaway_required": True,
            "launcher_waits_for_child_exit_required": True,
            "launcher_parent_live_through_child_exit_required": True,
            "automatic_retry_allowed": False,
            "fallback_allowed": False,
        },
        "release_root_sha256",
    )
    capsule_ancestor = capsule["capsule_ancestor_lease"]
    assert isinstance(capsule_ancestor, dict)
    process_derivation = _self_hashed(
        {
            "schema_version": 2,
            "policy": ("original_confirmatory_supervisor_process_command_derivation_v2"),
            "supervisor_code_root": str(code_root),
            "supervisor_state_root": str(state_root),
            "supervisor_source_path": str(supervisor_source_path),
            "supervisor_source_sha256": supervisor_source_sha256,
            "supervisor_launcher_path": str(supervisor_launcher_path),
            "supervisor_launcher_sha256": supervisor_launcher_sha256,
            "program_path": capsule["runtime_python_path"],
            "program_sha256": capsule["runtime_python_sha256"],
            "createprocess_application_path": capsule["runtime_python_path"],
            "createprocess_application_sha256": capsule["runtime_python_sha256"],
            "logical_venv_python_path": capsule["python_path"],
            "logical_venv_python_sha256": capsule["python_sha256"],
            "runtime_python_path": capsule["runtime_python_path"],
            "runtime_python_sha256": capsule["runtime_python_sha256"],
            "expected_live_image_path": capsule["runtime_python_path"],
            "expected_live_image_sha256": capsule["runtime_python_sha256"],
            "direct_base_runtime_live_parity_policy": (
                bootstrap._DIRECT_BASE_RUNTIME_LIVE_PARITY_POLICY
            ),
            "python_interpreter_flags": ["-I", "-S", "-B"],
            "python_sys_argv_prefix": [
                str(supervisor_source_path),
                "--root",
                str(state_root),
                "run",
            ],
            "supervisor_launch_spec_path_binding": (
                "Q.control_staging_projection.supervisor_launch_spec_path"
            ),
            "staged_e_intent_path_binding": ("Q.control_staging_projection.e_intent_path"),
            "os_launch_vector_policy": (
                "program_path_then_python_flags_then_exact_option_a_staged_argv_v2"
            ),
            "cwd": capsule_ancestor["anchor_path"],
            "command_preimage_policy": ("original_confirmatory_supervisor_process_command_v2"),
            "command_preimage_field_names": sorted(
                bootstrap._SUPERVISOR_PROCESS_COMMAND_PREIMAGE_FIELDS
            ),
            "command_sha256_policy": ("canonical_compact_sorted_json_sha256_no_lf_v1"),
            "peb_command_line_exact_direct_base_runtime_match_required": True,
            "in_process_sys_argv_exact_match_required": True,
            "isolated_flag_required": 1,
            "no_site_flag_required": 1,
            "dont_write_bytecode_flag_required": 1,
            "logical_venv_identity_separately_bound_required": True,
            "supervisor_launcher_role": ("nonexecuted_install_or_manual_recovery_helper"),
            "supervisor_launcher_used_for_authorized_process_launch": False,
            "extra_argv_allowed": False,
            "extra_cwd_allowed": False,
        },
        "contract_sha256",
    )
    postwake_custody_seed_policy = "original_confirmatory_postwake_custody_seed_v1"
    postwake_custody_pipe_derivation_policy = "postwake_custody_seed_sha256_direct_suffix_v1"
    q_e_custody_contract_policy = "original_confirmatory_q_e_supervisor_custody_contract_v1"
    q_e_custody_handoff_policy = "original_confirmatory_q_e_supervisor_custody_handoff_v1"
    q_e_custody_transport = "bounded_anonymous_pipe_blocking_v1"
    q_e_custody_ack_policy = "original_confirmatory_q_e_supervisor_custody_ack_v1"
    q_e_custody_receipt_policy = "original_confirmatory_q_e_supervisor_custody_receipt_v1"
    q_e_custody_receipt_filename = "q_e_custody_receipt.json"
    terminal_custody_template_root = bootstrap._TERMINAL_CUSTODY_AUTHORITY_TEMPLATE_ROOT_SHA256
    external_handoff_authority_spec_file_sha256 = hashlib.sha256(
        b"synthetic-external-handoff-authority-spec-file"
    ).hexdigest()
    external_handoff_authority_spec_canonical_root_sha256 = hashlib.sha256(
        b"synthetic-external-handoff-authority-spec-canonical-root"
    ).hexdigest()
    supervisor_release_root_sha256 = _authority_root(
        {
            "policy": bootstrap._SUPERVISOR_POLICY,
            "external_control_plane_release_root_sha256": (external_release_root_sha256),
            "external_control_plane_publication_id": external_publication_id,
            "external_control_plane_release_qualification_attestation_path": str(
                external_attestation_path
            ),
            "external_control_plane_release_qualification_attestation_file_sha256": (
                external_attestation_file_sha256
            ),
            "external_control_plane_release_qualification_attestation_root_sha256": (
                external_attestation_root_sha256
            ),
            "supervisor_code_root": str(code_root),
            "supervisor_state_root": str(state_root),
            "supervisor_source_path": str(supervisor_source_path),
            "supervisor_source_sha256": supervisor_source_sha256,
            "supervisor_launcher_path": str(supervisor_launcher_path),
            "supervisor_launcher_sha256": supervisor_launcher_sha256,
            "supervisor_program_path": capsule["runtime_python_path"],
            "supervisor_program_sha256": capsule["runtime_python_sha256"],
            "supervisor_runtime_python_path": capsule["runtime_python_path"],
            "supervisor_runtime_python_sha256": capsule["runtime_python_sha256"],
            "supervisor_process_command_derivation_contract_sha256": (
                process_derivation["contract_sha256"]
            ),
            "terminal_client_launcher_release_root_sha256": (
                terminal_client_release["release_root_sha256"]
            ),
            "postwake_custody_seed_policy": postwake_custody_seed_policy,
            "postwake_custody_pipe_derivation_policy": (postwake_custody_pipe_derivation_policy),
            "q_e_custody_contract_policy": q_e_custody_contract_policy,
            "q_e_custody_handoff_policy": q_e_custody_handoff_policy,
            "q_e_custody_transport": q_e_custody_transport,
            "q_e_custody_ack_policy": q_e_custody_ack_policy,
            "q_e_custody_receipt_policy": q_e_custody_receipt_policy,
            "q_e_custody_receipt_filename": q_e_custody_receipt_filename,
            "terminal_custody_authority_template_root_sha256": (terminal_custody_template_root),
            "external_codex_handoff_policy": bootstrap._EXTERNAL_CODEX_HANDOFF_POLICY,
            "external_codex_handoff_authority_spec_file_sha256": (
                external_handoff_authority_spec_file_sha256
            ),
            "external_codex_handoff_authority_spec_canonical_root_sha256": (
                external_handoff_authority_spec_canonical_root_sha256
            ),
            "internal_codex_wake_disposition": bootstrap._INTERNAL_CODEX_WAKE_DISPOSITION,
        }
    )
    return _self_hashed(
        {
            "schema_version": 1,
            "policy": "original_confirmatory_supervisor_release_binding_v1",
            "supervisor_policy": bootstrap._SUPERVISOR_POLICY,
            "supervisor_spec_schema_version": 3,
            "external_control_plane_release_root_sha256": (external_release_root_sha256),
            "external_control_plane_publication_id": external_publication_id,
            "external_control_plane_release_qualification_attestation_path": str(
                external_attestation_path
            ),
            "external_control_plane_release_qualification_attestation_file_sha256": (
                external_attestation_file_sha256
            ),
            "external_control_plane_release_qualification_attestation_root_sha256": (
                external_attestation_root_sha256
            ),
            "supervisor_code_root": str(code_root),
            "supervisor_state_root": str(state_root),
            "supervisor_source_path": str(supervisor_source_path),
            "supervisor_source_sha256": supervisor_source_sha256,
            "supervisor_launcher_path": str(supervisor_launcher_path),
            "supervisor_launcher_sha256": supervisor_launcher_sha256,
            "supervisor_program_path": capsule["runtime_python_path"],
            "supervisor_program_sha256": capsule["runtime_python_sha256"],
            "supervisor_runtime_python_path": capsule["runtime_python_path"],
            "supervisor_runtime_python_sha256": capsule["runtime_python_sha256"],
            "supervisor_process_command_derivation_contract": process_derivation,
            "supervisor_process_command_derivation_contract_sha256": (
                process_derivation["contract_sha256"]
            ),
            "terminal_client_launcher_release": terminal_client_release,
            "terminal_client_launcher_release_root_sha256": (
                terminal_client_release["release_root_sha256"]
            ),
            "plan_sha256": hashlib.sha256(b"synthetic-plan").hexdigest(),
            "runtime_release_root_sha256": hashlib.sha256(b"synthetic-runtime").hexdigest(),
            "terminal_release_root_sha256": hashlib.sha256(b"synthetic-terminal").hexdigest(),
            "supervisor_release_root_sha256": supervisor_release_root_sha256,
            "postwake_custody_seed_policy": postwake_custody_seed_policy,
            "postwake_custody_pipe_derivation_policy": (postwake_custody_pipe_derivation_policy),
            "q_e_custody_contract_policy": q_e_custody_contract_policy,
            "q_e_custody_handoff_policy": q_e_custody_handoff_policy,
            "q_e_custody_transport": q_e_custody_transport,
            "q_e_custody_ready_message_type": "Q_E_CUSTODY_READY",
            "q_e_custody_ack_policy": q_e_custody_ack_policy,
            "q_e_custody_ack_message_type": "Q_E_CUSTODY_ACK",
            "q_e_custody_receipt_policy": q_e_custody_receipt_policy,
            "q_e_custody_receipt_filename": q_e_custody_receipt_filename,
            "q_e_custody_ready_max_bytes": 64 * 1024,
            "q_e_custody_ack_max_bytes": 64 * 1024,
            "q_e_independent_verifier_receipt_required": True,
            "q_e_no_science_before_custody_ack": True,
            "terminal_custody_authority_template_root_sha256": (terminal_custody_template_root),
            "external_codex_handoff_policy": bootstrap._EXTERNAL_CODEX_HANDOFF_POLICY,
            "external_codex_handoff_authority_spec_file_sha256": (
                external_handoff_authority_spec_file_sha256
            ),
            "external_codex_handoff_authority_spec_canonical_root_sha256": (
                external_handoff_authority_spec_canonical_root_sha256
            ),
            "internal_codex_wake_disposition": bootstrap._INTERNAL_CODEX_WAKE_DISPOSITION,
            "exact_job_object_membership_required": True,
        },
        "contract_sha256",
    )


def _rehash_supervisor_release(release: dict[str, object]) -> None:
    release["supervisor_release_root_sha256"] = _authority_root(
        {
            "policy": release["supervisor_policy"],
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
            "supervisor_source_path": release["supervisor_source_path"],
            "supervisor_source_sha256": release["supervisor_source_sha256"],
            "supervisor_launcher_path": release["supervisor_launcher_path"],
            "supervisor_launcher_sha256": release["supervisor_launcher_sha256"],
            "supervisor_program_path": release["supervisor_program_path"],
            "supervisor_program_sha256": release["supervisor_program_sha256"],
            "supervisor_runtime_python_path": release["supervisor_runtime_python_path"],
            "supervisor_runtime_python_sha256": release["supervisor_runtime_python_sha256"],
            "supervisor_process_command_derivation_contract_sha256": release[
                "supervisor_process_command_derivation_contract_sha256"
            ],
            "terminal_client_launcher_release_root_sha256": release[
                "terminal_client_launcher_release_root_sha256"
            ],
            "postwake_custody_seed_policy": release["postwake_custody_seed_policy"],
            "postwake_custody_pipe_derivation_policy": release[
                "postwake_custody_pipe_derivation_policy"
            ],
            "q_e_custody_contract_policy": release["q_e_custody_contract_policy"],
            "q_e_custody_handoff_policy": release["q_e_custody_handoff_policy"],
            "q_e_custody_transport": release["q_e_custody_transport"],
            "q_e_custody_ack_policy": release["q_e_custody_ack_policy"],
            "q_e_custody_receipt_policy": release["q_e_custody_receipt_policy"],
            "q_e_custody_receipt_filename": release["q_e_custody_receipt_filename"],
            "terminal_custody_authority_template_root_sha256": release[
                "terminal_custody_authority_template_root_sha256"
            ],
            "external_codex_handoff_policy": release["external_codex_handoff_policy"],
            "external_codex_handoff_authority_spec_file_sha256": release[
                "external_codex_handoff_authority_spec_file_sha256"
            ],
            "external_codex_handoff_authority_spec_canonical_root_sha256": release[
                "external_codex_handoff_authority_spec_canonical_root_sha256"
            ],
            "internal_codex_wake_disposition": release["internal_codex_wake_disposition"],
        }
    )
    unsigned = {key: item for key, item in release.items() if key != "contract_sha256"}
    release["contract_sha256"] = _authority_root(unsigned)


@pytest.mark.parametrize(
    "missing_field",
    [
        "postwake_custody_seed_policy",
        "external_control_plane_publication_id",
        "external_control_plane_release_qualification_attestation_path",
        "external_control_plane_release_qualification_attestation_file_sha256",
        "external_control_plane_release_qualification_attestation_root_sha256",
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
    ],
)
def test_supervisor_release_requires_every_static_custody_field(
    missing_field: str,
) -> None:
    execution_capsule = _synthetic_release_capsule_binding()
    release = _supervisor_release(execution_capsule)
    del release[missing_field]
    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="supervisor release binding violates its exact field set",
    ):
        bootstrap._require_supervisor_release(
            release,
            execution_capsule=execution_capsule,
            plan_sha256=str(release["plan_sha256"]),
            runtime_release_root_sha256=str(release["runtime_release_root_sha256"]),
            terminal_release_root_sha256=str(release["terminal_release_root_sha256"]),
        )


def test_self_consistent_unicode_seed_policy_substitution_is_rejected() -> None:
    execution_capsule = _synthetic_release_capsule_binding()
    release = _supervisor_release(execution_capsule)
    release["postwake_custody_seed_policy"] = (
        "original_confirmatory_postwake_custody_seed_v1_\u017c"
    )
    _rehash_supervisor_release(release)
    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="supervisor release binding violates its exact policy",
    ):
        bootstrap._require_supervisor_release(
            release,
            execution_capsule=execution_capsule,
            plan_sha256=str(release["plan_sha256"]),
            runtime_release_root_sha256=str(release["runtime_release_root_sha256"]),
            terminal_release_root_sha256=str(release["terminal_release_root_sha256"]),
        )


@pytest.mark.parametrize(
    ("field", "substitute"),
    [
        (
            "q_e_custody_contract_policy",
            "original_confirmatory_q_e_supervisor_custody_contract_v2",
        ),
        (
            "q_e_custody_handoff_policy",
            "original_confirmatory_q_e_supervisor_custody_handoff_v2",
        ),
        ("q_e_custody_transport", "unbounded_pipe_v1"),
        ("q_e_custody_ready_message_type", "Q_E_CUSTODY_READY_V2"),
        (
            "q_e_custody_ack_policy",
            "original_confirmatory_q_e_supervisor_custody_ack_v2",
        ),
        ("q_e_custody_ack_message_type", "Q_E_CUSTODY_ACK_V2"),
        (
            "q_e_custody_receipt_policy",
            "original_confirmatory_q_e_supervisor_custody_receipt_v2",
        ),
        ("q_e_custody_receipt_filename", "q_e_custody_receipt-v2.json"),
        ("q_e_custody_ready_max_bytes", 64 * 1024 + 1),
        ("q_e_custody_ack_max_bytes", 64 * 1024 + 1),
        ("q_e_independent_verifier_receipt_required", False),
        ("q_e_no_science_before_custody_ack", False),
        ("terminal_custody_authority_template_root_sha256", "0" * 64),
        ("external_codex_handoff_policy", "internal_codex_wake_v1"),
        ("internal_codex_wake_disposition", "ALLOWED"),
    ],
)
def test_self_consistent_q_e_custody_release_substitution_is_rejected(
    field: str,
    substitute: object,
) -> None:
    execution_capsule = _synthetic_release_capsule_binding()
    release = _supervisor_release(execution_capsule)
    release[field] = substitute
    _rehash_supervisor_release(release)

    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="supervisor release binding violates its exact policy",
    ):
        bootstrap._require_supervisor_release(
            release,
            execution_capsule=execution_capsule,
            plan_sha256=str(release["plan_sha256"]),
            runtime_release_root_sha256=str(release["runtime_release_root_sha256"]),
            terminal_release_root_sha256=str(release["terminal_release_root_sha256"]),
        )


def test_self_consistent_internal_codex_wake_substitution_is_rejected() -> None:
    execution_capsule = _synthetic_release_capsule_binding()
    release = _supervisor_release(execution_capsule)
    release["internal_codex_wake_disposition"] = "ALLOWED"
    _rehash_supervisor_release(release)

    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="supervisor release binding violates its exact policy",
    ):
        bootstrap._require_supervisor_release(
            release,
            execution_capsule=execution_capsule,
            plan_sha256=str(release["plan_sha256"]),
            runtime_release_root_sha256=str(release["runtime_release_root_sha256"]),
            terminal_release_root_sha256=str(release["terminal_release_root_sha256"]),
        )


def _literal_authority_assignment(node: ast.expr) -> object:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left = _literal_authority_assignment(node.left)
        right = _literal_authority_assignment(node.right)
        if type(left) is int and type(right) is int:
            return left * right
    return ast.literal_eval(node)


def test_supervisor_release_ast_schema_matches_current_authority_snapshot() -> None:
    authority_path = _authority_source_path()
    assert authority_path.is_file()
    tree = ast.parse(authority_path.read_text(encoding="utf-8"))
    assignments: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if node.value is None:
                continue
            try:
                value = _literal_authority_assignment(node.value)
            except (TypeError, ValueError):
                continue
            assignments[target.id] = value

    assert assignments["_SUPERVISOR_RELEASE_FIELDS"] == bootstrap._SUPERVISOR_RELEASE_FIELDS
    assert len(bootstrap._SUPERVISOR_RELEASE_FIELDS) == 48
    expected_constants = {
        "SUPERVISOR_RELEASE_BINDING_POLICY": bootstrap._SUPERVISOR_RELEASE_POLICY,
        "SUPERVISOR_V3_POLICY": bootstrap._SUPERVISOR_POLICY,
        "EXTERNAL_CONTROL_PLANE_RELEASE_DIRECTORY_NAME": (
            bootstrap._EXTERNAL_CONTROL_PLANE_RELEASE_DIRECTORY_NAME
        ),
        "EXTERNAL_CONTROL_PLANE_VERIFICATIONS_DIRECTORY_NAME": (
            bootstrap._EXTERNAL_CONTROL_PLANE_VERIFICATIONS_DIRECTORY_NAME
        ),
        "EXTERNAL_CONTROL_PLANE_QUALIFICATION_ATTESTATION_FILENAME": (
            bootstrap._EXTERNAL_CONTROL_PLANE_QUALIFICATION_ATTESTATION_FILENAME
        ),
        "POSTWAKE_CUSTODY_SEED_POLICY": bootstrap._POSTWAKE_CUSTODY_SEED_POLICY,
        "POSTWAKE_CUSTODY_PIPE_DERIVATION_POLICY": (
            bootstrap._POSTWAKE_CUSTODY_PIPE_DERIVATION_POLICY
        ),
        "Q_E_CUSTODY_CONTRACT_POLICY": bootstrap._Q_E_CUSTODY_CONTRACT_POLICY,
        "Q_E_CUSTODY_HANDOFF_POLICY": bootstrap._Q_E_CUSTODY_HANDOFF_POLICY,
        "Q_E_CUSTODY_TRANSPORT": bootstrap._Q_E_CUSTODY_TRANSPORT,
        "Q_E_CUSTODY_READY_MESSAGE_TYPE": bootstrap._Q_E_CUSTODY_READY_MESSAGE_TYPE,
        "Q_E_CUSTODY_ACK_POLICY": bootstrap._Q_E_CUSTODY_ACK_POLICY,
        "Q_E_CUSTODY_ACK_MESSAGE_TYPE": bootstrap._Q_E_CUSTODY_ACK_MESSAGE_TYPE,
        "Q_E_CUSTODY_RECEIPT_POLICY": bootstrap._Q_E_CUSTODY_RECEIPT_POLICY,
        "Q_E_CUSTODY_RECEIPT_FILENAME": bootstrap._Q_E_CUSTODY_RECEIPT_FILENAME,
        "Q_E_CUSTODY_LINE_MAX_BYTES": bootstrap._Q_E_CUSTODY_LINE_MAX_BYTES,
        "EXTERNAL_CODEX_HANDOFF_POLICY": bootstrap._EXTERNAL_CODEX_HANDOFF_POLICY,
        "EXTERNAL_CODEX_HANDOFF_SINGLE_WAKE_OWNER": (
            bootstrap._EXTERNAL_CODEX_HANDOFF_SINGLE_WAKE_OWNER
        ),
    }
    assert {name: assignments[name] for name in expected_constants} == expected_constants

    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_original_confirmatory_supervisor_release_binding"
    )
    release_root_assignment = next(
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "release_root" for target in node.targets
        )
    )
    assert isinstance(release_root_assignment.value, ast.Call)
    root_mapping = release_root_assignment.value.args[0]
    assert isinstance(root_mapping, ast.Dict)
    assert {_literal_authority_assignment(key) for key in root_mapping.keys if key is not None} == {
        "policy",
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
        "supervisor_process_command_derivation_contract_sha256",
        "terminal_client_launcher_release_root_sha256",
        "postwake_custody_seed_policy",
        "postwake_custody_pipe_derivation_policy",
        "q_e_custody_contract_policy",
        "q_e_custody_handoff_policy",
        "q_e_custody_transport",
        "q_e_custody_ack_policy",
        "q_e_custody_receipt_policy",
        "q_e_custody_receipt_filename",
        "terminal_custody_authority_template_root_sha256",
        "external_codex_handoff_policy",
        "external_codex_handoff_authority_spec_file_sha256",
        "external_codex_handoff_authority_spec_canonical_root_sha256",
        "internal_codex_wake_disposition",
    }


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        (
            "external_control_plane_publication_id",
            f"cpr-{'2' * 32}",
            "exact policy",
        ),
        (
            "external_control_plane_release_qualification_attestation_path",
            str(
                Path.cwd()
                / bootstrap._EXTERNAL_CONTROL_PLANE_RELEASE_DIRECTORY_NAME
                / bootstrap._EXTERNAL_CONTROL_PLANE_VERIFICATIONS_DIRECTORY_NAME
                / f"cpr-{'1' * 32}"
                / "other.json"
            ),
            "exact policy",
        ),
        (
            "external_control_plane_release_qualification_attestation_file_sha256",
            "g" * 64,
            "attestation file is not one lowercase SHA-256",
        ),
        (
            "external_control_plane_release_qualification_attestation_root_sha256",
            True,
            "exact JSON types",
        ),
    ],
)
def test_bootstrap_release48_rejects_invalid_attestation_transport_after_reseal(
    field: str,
    replacement: object,
    error: str,
) -> None:
    execution_capsule = _synthetic_release_capsule_binding()
    release = _supervisor_release(execution_capsule)
    release[field] = replacement
    _rehash_supervisor_release(release)

    with pytest.raises(bootstrap.CapsuleBootstrapError, match=error):
        bootstrap._require_supervisor_release(
            release,
            execution_capsule=execution_capsule,
            plan_sha256=str(release["plan_sha256"]),
            runtime_release_root_sha256=str(release["runtime_release_root_sha256"]),
            terminal_release_root_sha256=str(release["terminal_release_root_sha256"]),
        )


def test_bootstrap_recursive_wrong_type_aliases_fail_after_coherent_reseal(
    tmp_path: Path,
) -> None:
    execution_capsule = _synthetic_release_capsule_binding()
    release = _supervisor_release(execution_capsule)

    release["q_e_independent_verifier_receipt_required"] = 1
    _rehash_supervisor_release(release)
    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="supervisor release binding violates its exact JSON types",
    ):
        bootstrap._require_supervisor_release(
            release,
            execution_capsule=execution_capsule,
            plan_sha256=str(release["plan_sha256"]),
            runtime_release_root_sha256=str(release["runtime_release_root_sha256"]),
            terminal_release_root_sha256=str(release["terminal_release_root_sha256"]),
        )

    release = _supervisor_release(execution_capsule)
    derivation = release["supervisor_process_command_derivation_contract"]
    assert isinstance(derivation, dict)
    derivation["isolated_flag_required"] = True
    derivation_unsigned = {
        key: item for key, item in derivation.items() if key != "contract_sha256"
    }
    derivation["contract_sha256"] = _authority_root(derivation_unsigned)
    release["supervisor_process_command_derivation_contract_sha256"] = derivation["contract_sha256"]
    _rehash_supervisor_release(release)
    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="supervisor process command derivation violates its exact JSON types",
    ):
        bootstrap._require_supervisor_release(
            release,
            execution_capsule=execution_capsule,
            plan_sha256=str(release["plan_sha256"]),
            runtime_release_root_sha256=str(release["runtime_release_root_sha256"]),
            terminal_release_root_sha256=str(release["terminal_release_root_sha256"]),
        )

    state_root = tmp_path / "state"
    job_id = "oc-" + ("b" * 64)
    control_staging_dir = state_root / "control_staging" / job_id
    control_staging: dict[str, object] = {
        "schema_version": 2.0,
        "policy": "original_confirmatory_control_staging_v2",
        "supervisor_state_root": str(state_root),
        "control_staging_dir": str(control_staging_dir),
        "final_job_dir": str(state_root / "jobs" / job_id),
        "staging_attempt_path": str(control_staging_dir / "staging_attempt.json"),
        "e_intent_path": str(control_staging_dir / "e_intent.json"),
        "launch_authorization_path": str(control_staging_dir / "launch_authorization.json"),
        "supervisor_launch_spec_path": str(control_staging_dir / "supervisor_launch_spec.json"),
        "staging_ready_path": str(control_staging_dir / "staging_ready.json"),
        "exact_file_allowlist": [
            "staging_attempt.json",
            "e_intent.json",
            "launch_authorization.json",
            "supervisor_launch_spec.json",
            "staging_ready.json",
        ],
    }
    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="control-staging projection violates its exact v2 typed schema",
    ):
        bootstrap._require_control_staging_projection(
            control_staging,
            job_id=job_id,
            supervisor_state_root=state_root,
            expected_sha256=_authority_root(control_staging),
        )


def test_terminal_custody_artifact_ast_schema_matches_current_authority_snapshot() -> None:
    authority_path = _authority_source_path()
    tree = ast.parse(authority_path.read_text(encoding="utf-8"))
    assignments: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if node.value is None:
                continue
            try:
                assignments[target.id] = _literal_authority_assignment(node.value)
            except (TypeError, ValueError):
                continue

    assert assignments["_EXPECTED_ARTIFACT_ROLE_ORDER"] == (
        bootstrap._PROTECTED_EXPECTED_ARTIFACT_ROLES
    )
    assert assignments["_EXPECTED_ARTIFACT_RULE_FIELDS"] == (
        bootstrap._PROTECTED_EXPECTED_ARTIFACT_RULE_FIELDS
    )
    assert assignments["_EXPECTED_ARTIFACT_TEMPLATE_RULE_FIELDS"] == (
        bootstrap._PROTECTED_EXPECTED_ARTIFACT_TEMPLATE_RULE_FIELDS
    )
    assert assignments["_EXPECTED_ARTIFACT_TEMPLATE"] == (
        bootstrap._PROTECTED_EXPECTED_ARTIFACT_TEMPLATE
    )
    assert assignments["_TERMINAL_CUSTODY_AUTHORITY_PROJECTION_FIELDS"] == (
        bootstrap._TERMINAL_CUSTODY_AUTHORITY_PROJECTION_FIELDS
    )
    assert assignments["_E_JOB_FIELDS"] == bootstrap._E_JOB_FIELDS
    assert assignments["_TERMINAL_CLIENT_LAUNCHER_RELEASE_FIELDS"] == (
        bootstrap._TERMINAL_CLIENT_LAUNCHER_RELEASE_FIELDS
    )
    assert assignments["_TERMINAL_CLIENT_LAUNCHER_E_PROJECTION_FIELDS"] == (
        bootstrap._TERMINAL_CLIENT_LAUNCHER_E_PROJECTION_FIELDS
    )
    assert assignments["_TERMINAL_CLIENT_LAUNCH_INTENT_FIELDS"] == (
        bootstrap._TERMINAL_CLIENT_LAUNCH_INTENT_FIELDS
    )
    assert assignments["OUTCOME_BLIND_EXPECTED_ARTIFACT_PROJECTION_POLICY"] == (
        bootstrap._OUTCOME_BLIND_EXPECTED_ARTIFACT_PROJECTION_POLICY
    )
    assert assignments["OUTCOME_BLIND_EXPECTED_ARTIFACT_INSTANCE_POLICY"] == (
        bootstrap._OUTCOME_BLIND_EXPECTED_ARTIFACT_INSTANCE_POLICY
    )
    assert assignments["TERMINAL_CUSTODY_AUTHORITY_PROJECTION_POLICY"] == (
        bootstrap._TERMINAL_CUSTODY_AUTHORITY_PROJECTION_POLICY
    )
    assert assignments["TERMINAL_CUSTODY_AUTHORITY_TEMPLATE_POLICY"] == (
        bootstrap._TERMINAL_CUSTODY_AUTHORITY_TEMPLATE_POLICY
    )


def test_bootstrap_q_e_canonicalizers_are_reached_from_both_preimport_spec_paths() -> None:
    """Structural parsing and trusted parity independently validate Q/E custody."""

    tree = ast.parse(Path(bootstrap.__file__).read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def called_names(function_name: str) -> set[str]:
        function = functions[function_name]
        return {
            call.func.id
            for call in ast.walk(function)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        }

    assert "_require_q_e_custody_spec_structure" in called_names("_parse_held_supervisor_run_spec")
    assert "_require_q_e_custody_spec_fields" in called_names(
        "_require_supervisor_spec_exact_q_e_parity"
    )
    assert "_parse_held_supervisor_run_spec" in called_names("_require_preimport_q_e_anchor")
    assert "_require_supervisor_spec_exact_q_e_parity" in called_names(
        "_require_preimport_q_e_anchor"
    )


def test_future_terminal_client_launch_intent_has_no_pregrant_read_call_path() -> None:
    bootstrap_path = Path(bootstrap.__file__).resolve()
    tree = ast.parse(bootstrap_path.read_text(encoding="utf-8"))
    launch_intent_functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(item, ast.Name)
            and item.id
            in {
                "_TERMINAL_CLIENT_LAUNCH_INTENT_FILENAME",
                "launch_intent_path",
            }
            for item in ast.walk(node)
        )
    }
    assert set(launch_intent_functions) == {
        "_require_terminal_client_launcher_release",
        "_terminal_client_launcher_argv_template",
        "_terminal_client_launcher_projection",
    }

    forbidden_read_or_open_calls = {
        "_open_capsule_no_follow",
        "_open_held_authority_file",
        "_open_held_plain_file",
        "exists",
        "lstat",
        "open",
        "read_bytes",
        "read_text",
        "stat",
    }
    observed_forbidden_calls: list[tuple[str, str]] = []
    for function_name, function in launch_intent_functions.items():
        for call in (item for item in ast.walk(function) if isinstance(item, ast.Call)):
            called_name = (
                call.func.id
                if isinstance(call.func, ast.Name)
                else call.func.attr
                if isinstance(call.func, ast.Attribute)
                else ""
            )
            if called_name in forbidden_read_or_open_calls:
                observed_forbidden_calls.append((function_name, called_name))
    assert observed_forbidden_calls == []


def test_terminal_custody_static_template_root_matches_current_authority_snapshot() -> None:
    authority_path = _authority_source_path()
    module_name = "_capsule_builder_terminal_template_authority_snapshot"
    module_spec = importlib.util.spec_from_file_location(module_name, authority_path)
    assert module_spec is not None and module_spec.loader is not None
    authority = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = authority
    try:
        module_spec.loader.exec_module(authority)
        template = (
            authority.build_original_confirmatory_terminal_custody_authority_template_projection()
        )
    finally:
        sys.modules.pop(module_name, None)

    assert template["template_root_sha256"] == (
        bootstrap._TERMINAL_CUSTODY_AUTHORITY_TEMPLATE_ROOT_SHA256
    )
    launcher_contract = template["terminal_client_launcher_contract"]
    assert launcher_contract["release_field_names"] == sorted(
        bootstrap._TERMINAL_CLIENT_LAUNCHER_RELEASE_FIELDS
    )
    assert launcher_contract["e_projection_field_names"] == sorted(
        bootstrap._TERMINAL_CLIENT_LAUNCHER_E_PROJECTION_FIELDS
    )
    for (
        field,
        expected,
    ) in bootstrap._TERMINAL_CLIENT_LAUNCHER_STATIC_CONTROL_VALUES.items():
        if field in launcher_contract:
            assert launcher_contract[field] == expected


def _synthetic_protected_expected_artifacts(
    project_root: Path,
    *,
    run_id: str = "synthetic-run",
) -> list[dict[str, object]]:
    run_directory = project_root / "artifacts" / "runs" / run_id
    return [
        {
            "role": "terminal_seal",
            "path": str(run_directory / ".immutable.json"),
            "expected_sha256": None,
            "must_be_absent_before": True,
            "json_equals": {"run_id": run_id, "status": "completed"},
        },
        {
            "role": "integrity_receipt",
            "path": str(run_directory / "artifact_manifest.json"),
            "expected_sha256": None,
            "must_be_absent_before": True,
            "json_equals": {"run_id": run_id, "status": "completed"},
        },
        {
            "role": "completion_evidence",
            "path": str(run_directory / "completion_evidence.json"),
            "expected_sha256": None,
            "must_be_absent_before": True,
            "json_equals": {
                "run_id": run_id,
                "completion_stage": "CONFIRMATORY_COMPLETE",
                "study_outcome_eligible": True,
            },
        },
        {
            "role": "integrity_registry",
            "path": str(run_directory.parent / "integrity_registry.jsonl"),
            "expected_sha256": None,
            "must_be_absent_before": False,
            "json_equals": {},
        },
        {
            "role": "stage_attestation_registry",
            "path": str(run_directory.parent / "run_stage_attestations.jsonl"),
            "expected_sha256": None,
            "must_be_absent_before": False,
            "json_equals": {},
        },
        {
            "role": "stage_attestation_anchor",
            "path": str(run_directory.parent / "run_stage_attestations.anchor.json"),
            "expected_sha256": None,
            "must_be_absent_before": False,
            "json_equals": {},
        },
        {
            "role": "disposition_anchor",
            "path": str(run_directory.parent / "run_dispositions.anchor.json"),
            "expected_sha256": None,
            "must_be_absent_before": False,
            "json_equals": {},
        },
    ]


def test_protected_expected_artifact_template_has_frozen_terminal_root() -> None:
    projection = bootstrap._protected_expected_artifact_template_projection()

    assert (
        projection["projection_root_sha256"]
        == "dcc09116e40384ecbd398c953357498cc88ac9d65dd4f100e96d198f88e19011"
    )
    assert (
        bootstrap._require_protected_expected_artifact_template_projection(projection) == projection
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("dotted_paths_allowed", True),
        ("numeric_or_list_indirection_allowed", True),
        (
            "scientific_metric_ranking_prediction_or_outcome_selectors_allowed",
            True,
        ),
        ("strict_expected_type_equality_required", False),
        (
            "allowed_flat_json_selectors",
            [
                "completion_stage",
                "metrics",
                "run_id",
                "status",
                "study_outcome_eligible",
            ],
        ),
    ],
)
def test_self_consistent_weakened_expected_artifact_template_is_rejected(
    field: str,
    replacement: object,
) -> None:
    projection = bootstrap._protected_expected_artifact_template_projection()
    projection[field] = replacement
    unsigned = {key: item for key, item in projection.items() if key != "projection_root_sha256"}
    projection["projection_root_sha256"] = _authority_root(unsigned)

    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="template violates its exact control-only policy",
    ):
        bootstrap._require_protected_expected_artifact_template_projection(projection)


def test_protected_expected_artifact_instance_is_exact_and_self_hashed(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "artifacts" / "runs" / "synthetic-run"
    instance = bootstrap._protected_expected_artifact_instance(
        run_id="synthetic-run",
        expected_run_directory=run_directory,
    )

    assert (
        instance["template_projection_root_sha256"]
        == (bootstrap._protected_expected_artifact_template_projection()["projection_root_sha256"])
    )
    assert instance["expected_artifacts"] == _synthetic_protected_expected_artifacts(tmp_path)
    assert instance["expected_artifacts_root_sha256"] == _authority_root(
        instance["expected_artifacts"]
    )
    assert (
        bootstrap._require_protected_expected_artifact_instance(
            instance,
            run_id="synthetic-run",
            expected_run_directory=run_directory,
        )
        == instance
    )


def test_self_consistent_weakened_expected_artifact_instance_is_rejected(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "artifacts" / "runs" / "synthetic-run"
    instance = bootstrap._protected_expected_artifact_instance(
        run_id="synthetic-run",
        expected_run_directory=run_directory,
    )
    instance["outcome_values_read"] = True
    unsigned = {key: item for key, item in instance.items() if key != "projection_root_sha256"}
    instance["projection_root_sha256"] = _authority_root(unsigned)

    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="instance violates its exact control-only policy",
    ):
        bootstrap._require_protected_expected_artifact_instance(
            instance,
            run_id="synthetic-run",
            expected_run_directory=run_directory,
        )


def _synthetic_terminal_custody_artifact_projection(
    project_root: Path,
    *,
    supervisor_release: dict[str, object] | None = None,
    supervisor_job_directory: Path | None = None,
    run_id: str = "synthetic-run",
    verify_terminal_command_projection_sha256: str = "a" * 64,
    verify_terminal_environment_sha256: str = "b" * 64,
) -> dict[str, object]:
    release = supervisor_release or _supervisor_release()
    job_directory = supervisor_job_directory or (
        project_root / "synthetic-supervisor" / "jobs" / "synthetic-job"
    )
    launcher_release = release["terminal_client_launcher_release"]
    assert isinstance(launcher_release, dict)
    launcher_projection = bootstrap._terminal_client_launcher_projection(
        launcher_release=launcher_release,
        job_id=job_directory.name,
        supervisor_job_directory=job_directory,
        verify_terminal_command_projection_sha256=(verify_terminal_command_projection_sha256),
        verify_terminal_environment_sha256=verify_terminal_environment_sha256,
        verify_terminal_cwd=project_root,
    )
    run_directory = project_root / "artifacts" / "runs" / run_id
    instance = bootstrap._protected_expected_artifact_instance(
        run_id=run_id,
        expected_run_directory=run_directory,
    )
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "policy": "original_confirmatory_terminal_custody_authority_projection_v1",
        "terminal_custody_authority_template_root_sha256": (
            bootstrap._TERMINAL_CUSTODY_AUTHORITY_TEMPLATE_ROOT_SHA256
        ),
        "outcome_blind_expected_artifact_instance": instance,
        "terminal_client_launcher_projection": launcher_projection,
        "terminal_client_launcher_projection_root_sha256": launcher_projection[
            "projection_root_sha256"
        ],
    }
    return {
        **unsigned,
        "projection_root_sha256": _authority_root(unsigned),
    }


def _rehash_terminal_custody_projection(projection: dict[str, object]) -> None:
    unsigned = {key: item for key, item in projection.items() if key != "projection_root_sha256"}
    projection["projection_root_sha256"] = _authority_root(unsigned)


def test_terminal_custody_artifact_projection_is_q_e_bound_preimport(
    tmp_path: Path,
) -> None:
    release = _supervisor_release()
    job_directory = tmp_path / "synthetic-supervisor" / "jobs" / "synthetic-job"
    projection = _synthetic_terminal_custody_artifact_projection(
        tmp_path,
        supervisor_release=release,
        supervisor_job_directory=job_directory,
    )

    instance = bootstrap._require_terminal_custody_artifact_authority_projection(
        projection,
        project_root=tmp_path,
        run_id="synthetic-run",
        expected_run_directory=tmp_path / "artifacts" / "runs" / "synthetic-run",
        supervisor_release=release,
        supervisor_job_id="synthetic-job",
        supervisor_job_directory=job_directory,
        verify_terminal_command_projection_sha256="a" * 64,
        verify_terminal_environment_sha256="b" * 64,
    )

    assert instance == projection["outcome_blind_expected_artifact_instance"]


def test_self_consistent_artifact_selector_in_e_projection_is_rejected(
    tmp_path: Path,
) -> None:
    projection = _synthetic_terminal_custody_artifact_projection(tmp_path)
    instance = projection["outcome_blind_expected_artifact_instance"]
    assert isinstance(instance, dict)
    artifacts = instance["expected_artifacts"]
    assert isinstance(artifacts, list)
    completion = artifacts[2]
    assert isinstance(completion, dict)
    checks = completion["json_equals"]
    assert isinstance(checks, dict)
    checks["metrics"] = {"auprc": 0.99}
    instance["expected_artifacts_root_sha256"] = _authority_root(artifacts)
    instance_unsigned = {
        key: item for key, item in instance.items() if key != "projection_root_sha256"
    }
    instance["projection_root_sha256"] = _authority_root(instance_unsigned)
    _rehash_terminal_custody_projection(projection)

    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="instance violates its exact control-only policy",
    ):
        bootstrap._require_terminal_custody_artifact_authority_projection(
            projection,
            project_root=tmp_path,
            run_id="synthetic-run",
            expected_run_directory=tmp_path / "artifacts" / "runs" / "synthetic-run",
            supervisor_release=_supervisor_release(),
            supervisor_job_id="synthetic-job",
            supervisor_job_directory=(tmp_path / "synthetic-supervisor" / "jobs" / "synthetic-job"),
            verify_terminal_command_projection_sha256="a" * 64,
            verify_terminal_environment_sha256="b" * 64,
        )


def test_self_consistent_terminal_template_substitution_is_rejected(
    tmp_path: Path,
) -> None:
    projection = _synthetic_terminal_custody_artifact_projection(tmp_path)
    projection["terminal_custody_authority_template_root_sha256"] = "0" * 64
    _rehash_terminal_custody_projection(projection)

    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="pre-import control-only policy",
    ):
        bootstrap._require_terminal_custody_artifact_authority_projection(
            projection,
            project_root=tmp_path,
            run_id="synthetic-run",
            expected_run_directory=tmp_path / "artifacts" / "runs" / "synthetic-run",
            supervisor_release=_supervisor_release(),
            supervisor_job_id="synthetic-job",
            supervisor_job_directory=(tmp_path / "synthetic-supervisor" / "jobs" / "synthetic-job"),
            verify_terminal_command_projection_sha256="a" * 64,
            verify_terminal_environment_sha256="b" * 64,
        )


def test_exact_protected_expected_artifacts_are_control_only(tmp_path: Path) -> None:
    run_directory = tmp_path / "artifacts" / "runs" / "synthetic-run"
    artifacts = _synthetic_protected_expected_artifacts(tmp_path)

    assert (
        bootstrap._require_protected_expected_artifacts(
            artifacts,
            project_root=tmp_path,
            run_directory=run_directory,
            run_id="synthetic-run",
        )
        == artifacts
    )


@pytest.mark.parametrize(
    ("role", "field", "value"),
    [
        ("terminal_seal", "metrics", {"auprc": 0.99}),
        ("integrity_receipt", "ranking", ["nucleus-1"]),
        ("completion_evidence", "outcome", "favorable"),
        ("completion_evidence", "predictions", ["malignant"]),
        ("completion_evidence", "p_value", 0.01),
        ("completion_evidence", "restoration", {"delta": 0.1}),
        ("integrity_registry", "status", "valid"),
    ],
)
def test_arbitrary_expected_artifact_json_equals_is_rejected(
    tmp_path: Path,
    role: str,
    field: str,
    value: object,
) -> None:
    artifacts = _synthetic_protected_expected_artifacts(tmp_path)
    rule = next(record for record in artifacts if record["role"] == role)
    checks = rule["json_equals"]
    assert isinstance(checks, dict)
    checks[field] = value

    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="differs from its exact control-only rule",
    ):
        bootstrap._require_protected_expected_artifacts(
            artifacts,
            project_root=tmp_path,
            run_directory=tmp_path / "artifacts" / "runs" / "synthetic-run",
            run_id="synthetic-run",
        )


def test_protected_expected_artifact_strict_type_equality_rejects_bool_as_int(
    tmp_path: Path,
) -> None:
    artifacts = _synthetic_protected_expected_artifacts(tmp_path)
    completion = next(record for record in artifacts if record["role"] == "completion_evidence")
    checks = completion["json_equals"]
    assert isinstance(checks, dict)
    checks["study_outcome_eligible"] = 1

    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="differs from its exact control-only rule",
    ):
        bootstrap._require_protected_expected_artifacts(
            artifacts,
            project_root=tmp_path,
            run_directory=tmp_path / "artifacts" / "runs" / "synthetic-run",
            run_id="synthetic-run",
        )


@pytest.mark.parametrize("mutation", ["path", "order", "sha256", "absence"])
def test_protected_expected_artifact_cross_binding_is_exact(
    tmp_path: Path,
    mutation: str,
) -> None:
    artifacts = _synthetic_protected_expected_artifacts(tmp_path)
    if mutation == "path":
        artifacts[0]["path"] = str(
            tmp_path / "artifacts" / "runs" / "other-run" / ".immutable.json"
        )
    elif mutation == "order":
        artifacts[0], artifacts[1] = artifacts[1], artifacts[0]
    elif mutation == "sha256":
        artifacts[0]["expected_sha256"] = "0" * 64
    else:
        artifacts[0]["must_be_absent_before"] = False

    with pytest.raises(bootstrap.CapsuleBootstrapError):
        bootstrap._require_protected_expected_artifacts(
            artifacts,
            project_root=tmp_path,
            run_directory=tmp_path / "artifacts" / "runs" / "synthetic-run",
            run_id="synthetic-run",
        )


def _synthetic_e_consumption_contract(
    job_directory: Path,
) -> dict[str, object]:
    return _self_hashed(
        {
            "schema_version": 1,
            "policy": ("original_confirmatory_e_intent_consumed_supervisor_custody_v1"),
            "claim_policy": "original_confirmatory_e_intent_consumed_claim_v1",
            "claim_path": str(job_directory / "e_intent_consumed.json"),
            "custody_receipt_path": str(job_directory / "e_intent_consumed_custody_receipt.json"),
            "transport": "bounded_anonymous_pipe_blocking_v1",
            "ready_message_type": "E_INTENT_CONSUMED_READY",
            "ack_message_type": "E_INTENT_CONSUMED_ACK",
            "ready_line_max_bytes": 16 * 1024,
            "ack_line_max_bytes": 16 * 1024,
            "duplicate_target_access_mask": 0x80000000,
            "duplicate_options": 0,
            "close_source": False,
            "source_handle_retained_through_ack": True,
            "supervisor_handle_retention_policy": (
                "through_main_wait_preterminal_terminal_postwake_and_terminal_seal_v1"
            ),
            "exact_job_object_membership_required": True,
            "exact_supervisor_process_identity_required": True,
            "exact_downstream_spec_rederivation_required": True,
            "scientific_inputs_before_ack_allowed": False,
            "automatic_retry_allowed": False,
        },
        "contract_sha256",
    )


@pytest.mark.parametrize(
    ("field", "substitute"),
    [
        ("scientific_inputs_before_ack_allowed", True),
        ("source_handle_retained_through_ack", False),
        ("duplicate_target_access_mask", 0xC0000000),
        ("ready_line_max_bytes", 16 * 1024 + 1),
    ],
)
def test_self_consistent_permissive_e_consumption_contract_is_rejected(
    tmp_path: Path,
    field: str,
    substitute: object,
) -> None:
    contract = _synthetic_e_consumption_contract(tmp_path)
    contract[field] = substitute
    unsigned = {key: item for key, item in contract.items() if key != "contract_sha256"}
    contract["contract_sha256"] = _authority_root(unsigned)
    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="E consumption contract violates its exact fail-closed policy",
    ):
        bootstrap._require_e_consumption_contract(
            contract,
            supervisor_job_directory=tmp_path,
        )


def _synthetic_scientific_authority(
    project_root: Path,
) -> dict[str, object]:
    expected_confirmatory_gate = {
        "schema_version": 1,
        "policy": "synthetic_confirmatory_gate_v1",
    }
    expected_cli_input_binding = {
        "schema_version": 1,
        "policy": "synthetic_cli_input_binding_v1",
    }
    runs_root = project_root / "artifacts" / "runs"
    namespace = project_root / "artifacts" / "original_confirmatory_technical_authorities"
    technical_authority_artifact_root_sha256 = hashlib.sha256(
        b"synthetic:technical-authority-artifact-root"
    ).hexdigest()
    technical_authorization_sha256 = hashlib.sha256(
        b"synthetic:technical_authorization_sha256"
    ).hexdigest()
    technical_authority = _self_hashed(
        {
            "schema_version": 1,
            "policy": "original_confirmatory_technical_authority_lifecycle_binding_v1",
            "authority_directory": str(namespace / "synthetic-authority"),
            "chain_depth": 3,
            "artifact_root_sha256": technical_authority_artifact_root_sha256,
            "sha256_manifest_sha256": hashlib.sha256(b"synthetic:t0:manifest").hexdigest(),
            "execution_source_manifest_sha256": hashlib.sha256(
                b"synthetic:t0:source-manifest"
            ).hexdigest(),
            "execution_source_root_sha256": hashlib.sha256(b"synthetic:t0:source-root").hexdigest(),
            "parent_authority_directory": str(project_root / "artifacts" / "freezes" / "protocol"),
            "parent_artifact_root_sha256": hashlib.sha256(b"synthetic:t0:parent-root").hexdigest(),
            "parent_sha256_manifest_sha256": hashlib.sha256(
                b"synthetic:t0:parent-manifest"
            ).hexdigest(),
            "technical_authorization_sha256": technical_authorization_sha256,
            "independent_review_receipt_sha256": hashlib.sha256(b"synthetic:t0:review").hexdigest(),
            "immutable_marker_sha256": hashlib.sha256(b"synthetic:t0:immutable").hexdigest(),
            "publication_attempt_sha256": hashlib.sha256(b"synthetic:t0:attempt").hexdigest(),
            "publication_success_sha256": hashlib.sha256(b"synthetic:t0:success").hexdigest(),
            "primary_outcomes_inspected": True,
            "confirmatory_outcomes_inspected": False,
            "confirmatory_outcome_values_read": False,
            "scientific_definition_changed": False,
            "automatic_retry_allowed": False,
        },
        "binding_sha256",
    )
    published_technical_authority_lifecycle_binding = _self_hashed(
        {
            "schema_version": 1,
            "policy": ("published_original_confirmatory_technical_authority_lifecycle_binding_v1"),
            "namespace_directory": str(namespace),
            "namespace_claim_sha256": hashlib.sha256(b"synthetic:t0:namespace-claim").hexdigest(),
            "review_attempt_claim_sha256": hashlib.sha256(
                b"synthetic:t0:review-attempt-claim"
            ).hexdigest(),
            "technical_authority": technical_authority,
            "automatic_retry_allowed": False,
            "adoption_allowed": False,
            "cleanup_allowed": False,
        },
        "binding_sha256",
    )
    static_runner_binding = _self_hashed(
        {
            "schema_version": 3,
            "policy": "original_confirmatory_static_runner_binding_v3",
            "project_root": str(project_root),
            "primary_run_directory": str(runs_root / "primary"),
            "freeze_directory": str(project_root / "artifacts" / "freezes" / "protocol"),
            "technical_authority_directory": technical_authority["authority_directory"],
            "technical_authority_artifact_root_sha256": (technical_authority_artifact_root_sha256),
            "technical_authorization_sha256": technical_authorization_sha256,
            "published_technical_authority_lifecycle_binding": (
                published_technical_authority_lifecycle_binding
            ),
            "lifecycle_readiness_run_directory": str(runs_root / "lifecycle"),
            "dataset_path": str(project_root / "data" / "manifests" / "pannuke_instances.parquet"),
            "manifest_path": str(project_root / "data" / "manifests" / "pannuke_manifest.json"),
            "duplicate_audit_path": str(
                project_root / "artifacts" / "data_validation" / "duplicate_audit.npz"
            ),
            "pathology_encoder_audit_path": str(
                project_root / "artifacts" / "data_validation" / "pathology_encoder_audit.json"
            ),
            "frozen_primary_config_path": str(
                project_root / "configs" / "pannuke_primary_frozen.yaml"
            ),
            "frozen_confirmatory_config_path": str(
                project_root / "configs" / "pannuke_confirmatory_frozen.yaml"
            ),
            "runs_root": str(runs_root),
            "expected_confirmatory_gate": expected_confirmatory_gate,
            "expected_confirmatory_gate_sha256": _authority_root(expected_confirmatory_gate),
            "expected_cli_input_binding": expected_cli_input_binding,
            "expected_cli_input_binding_sha256": _authority_root(expected_cli_input_binding),
            "artifact_scope": "real_pannuke_confirmatory_study",
            "semantic_outcome_read_scope": ("integrity/control_only_no_scientific_outcomes"),
        },
        "binding_sha256",
    )
    scientific_hash_fields = (
        "historical_primary_authority_artifact_root_sha256",
        "historical_primary_evidence_sha256",
        "technical_authorization_sha256",
        "technical_execution_source_root_sha256",
        "technical_execution_source_manifest_sha256",
        "source_delta_sha256",
        "confirmatory_storage_policy_sha256",
        "independent_review_receipt_sha256",
    )
    without_root: dict[str, object] = {
        "schema_version": 1,
        "policy": "original_confirmatory_scientific_authority_projection_v1",
        **{
            field: hashlib.sha256(f"synthetic:{field}".encode()).hexdigest()
            for field in scientific_hash_fields
        },
        "static_runner_binding": static_runner_binding,
        "static_runner_binding_sha256": static_runner_binding["binding_sha256"],
    }
    without_root["technical_execution_source_root_sha256"] = technical_authority[
        "execution_source_root_sha256"
    ]
    without_root["technical_execution_source_manifest_sha256"] = technical_authority[
        "execution_source_manifest_sha256"
    ]
    without_root["independent_review_receipt_sha256"] = technical_authority[
        "independent_review_receipt_sha256"
    ]
    without_root["historical_primary_authority_artifact_root_sha256"] = technical_authority[
        "parent_artifact_root_sha256"
    ]
    return {
        **without_root,
        "scientific_authority_root_sha256": _authority_root(without_root),
    }


def _synthetic_scientific_request_projection(
    *,
    project_root: Path,
    scientific_authority: dict[str, object],
    job: dict[str, object],
    lineage: dict[str, object],
    plan_sha256: str,
) -> dict[str, object]:
    static_runner_binding = scientific_authority["static_runner_binding"]
    assert isinstance(static_runner_binding, dict)
    runs_root = project_root / "artifacts" / "runs"
    checkpoint_authority_projection = {
        "schema_version": 1,
        "policy": "synthetic_checkpoint_authority_v1",
    }
    return _self_hashed(
        {
            "schema_version": 1,
            "policy": "original_confirmatory_capsule_request_projection_v1",
            "q_static_runner_binding_sha256": static_runner_binding["binding_sha256"],
            "job_id": job["job_id"],
            "attempt_id": job["attempt_id"],
            "run_id": job["run_id"],
            "execution_mode": lineage["execution_mode"],
            "retry_of_run_id": lineage["retry_of_run_id"],
            "runs_root": str(runs_root),
            "expected_run_directory": str(runs_root / str(job["run_id"])),
            "plan_sha256": plan_sha256,
            "controls_binding_sha256": hashlib.sha256(b"synthetic:controls-binding").hexdigest(),
            "bridge_binding_sha256": hashlib.sha256(b"synthetic:bridge-binding").hexdigest(),
            "gate_evidence_sha256": static_runner_binding["expected_confirmatory_gate_sha256"],
            "cli_input_binding_sha256": static_runner_binding["expected_cli_input_binding_sha256"],
            "checkpoint_authority_projection": checkpoint_authority_projection,
            "checkpoint_authority_projection_sha256": _authority_root(
                checkpoint_authority_projection
            ),
            "checkpoint_contract_profile": "original_confirmatory_exact_180",
            "checkpoint_directive_count": 180,
            "artifact_scope": "real_pannuke_confirmatory_study",
            "scientific_outcomes_read": False,
            "selection_or_tuning_performed": False,
            "publication_performed": False,
            "automatic_retry_allowed": False,
        },
        "projection_sha256",
    )


def test_self_consistent_permissive_static_runner_binding_is_rejected(
    tmp_path: Path,
) -> None:
    scientific_authority = _synthetic_scientific_authority(tmp_path)
    raw_static_runner_binding = scientific_authority["static_runner_binding"]
    assert isinstance(raw_static_runner_binding, dict)
    static_runner_binding = dict(raw_static_runner_binding)
    static_runner_binding["semantic_outcome_read_scope"] = "all_outcomes"
    static_unsigned = {
        key: item for key, item in static_runner_binding.items() if key != "binding_sha256"
    }
    static_runner_binding["binding_sha256"] = _authority_root(static_unsigned)
    scientific_authority["static_runner_binding"] = static_runner_binding
    scientific_authority["static_runner_binding_sha256"] = static_runner_binding["binding_sha256"]
    authority_unsigned = {
        key: item
        for key, item in scientific_authority.items()
        if key != "scientific_authority_root_sha256"
    }
    scientific_authority["scientific_authority_root_sha256"] = _authority_root(authority_unsigned)

    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="Q static runner binding violates its exact outcome-blind policy",
    ):
        bootstrap._require_scientific_authority(
            scientific_authority,
            project_root=tmp_path,
        )


@pytest.mark.parametrize(
    "field",
    [
        "technical_authority_directory",
        "technical_authority_artifact_root_sha256",
        "technical_authorization_sha256",
        "published_technical_authority_lifecycle_binding",
    ],
)
def test_bootstrap_rejects_missing_t0_static_binding_field(
    tmp_path: Path,
    field: str,
) -> None:
    scientific = _synthetic_scientific_authority(tmp_path)
    raw_static = scientific["static_runner_binding"]
    assert isinstance(raw_static, dict)
    static = dict(raw_static)
    static.pop(field)
    static = _self_hashed(
        {key: value for key, value in static.items() if key != "binding_sha256"},
        "binding_sha256",
    )
    scientific["static_runner_binding"] = static
    scientific["static_runner_binding_sha256"] = static["binding_sha256"]
    scientific = _self_hashed(
        {
            key: value
            for key, value in scientific.items()
            if key != "scientific_authority_root_sha256"
        },
        "scientific_authority_root_sha256",
    )

    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="Q static runner binding violates its exact field set",
    ):
        bootstrap._require_scientific_authority(
            scientific,
            project_root=tmp_path,
        )


def test_bootstrap_rejects_rehashed_static_t0_authorization_mismatch(
    tmp_path: Path,
) -> None:
    scientific = _synthetic_scientific_authority(tmp_path)
    raw_static = scientific["static_runner_binding"]
    assert isinstance(raw_static, dict)
    static = dict(raw_static)
    static["technical_authorization_sha256"] = "f" * 64
    static = _self_hashed(
        {key: value for key, value in static.items() if key != "binding_sha256"},
        "binding_sha256",
    )
    scientific["static_runner_binding"] = static
    scientific["static_runner_binding_sha256"] = static["binding_sha256"]
    scientific = _self_hashed(
        {
            key: value
            for key, value in scientific.items()
            if key != "scientific_authority_root_sha256"
        },
        "scientific_authority_root_sha256",
    )

    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="Q static runner binding violates its exact outcome-blind policy",
    ):
        bootstrap._require_scientific_authority(
            scientific,
            project_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("flat_field", "nested_field"),
    [
        (
            "historical_primary_authority_artifact_root_sha256",
            "parent_artifact_root_sha256",
        ),
        ("technical_execution_source_root_sha256", "execution_source_root_sha256"),
        (
            "technical_execution_source_manifest_sha256",
            "execution_source_manifest_sha256",
        ),
        ("independent_review_receipt_sha256", "independent_review_receipt_sha256"),
    ],
)
def test_bootstrap_rejects_rehashed_flat_published_t0_mismatch(
    tmp_path: Path,
    flat_field: str,
    nested_field: str,
) -> None:
    scientific = _synthetic_scientific_authority(tmp_path)
    raw_static = scientific["static_runner_binding"]
    assert isinstance(raw_static, dict)
    raw_published = raw_static["published_technical_authority_lifecycle_binding"]
    assert isinstance(raw_published, dict)
    raw_technical = raw_published["technical_authority"]
    assert isinstance(raw_technical, dict)
    assert scientific[flat_field] == raw_technical[nested_field]
    scientific[flat_field] = "0" * 64
    scientific = _self_hashed(
        {
            key: value
            for key, value in scientific.items()
            if key != "scientific_authority_root_sha256"
        },
        "scientific_authority_root_sha256",
    )

    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="Q scientific authority nested roots differ",
    ):
        bootstrap._require_scientific_authority(
            scientific,
            project_root=tmp_path,
        )


def test_bootstrap_rejects_rehashed_freeze_t0_parent_mismatch(
    tmp_path: Path,
) -> None:
    scientific = _synthetic_scientific_authority(tmp_path)
    raw_static = scientific["static_runner_binding"]
    assert isinstance(raw_static, dict)
    static = dict(raw_static)
    static["freeze_directory"] = str(tmp_path / "artifacts" / "freezes" / "substituted")
    static = _self_hashed(
        {key: value for key, value in static.items() if key != "binding_sha256"},
        "binding_sha256",
    )
    scientific["static_runner_binding"] = static
    scientific["static_runner_binding_sha256"] = static["binding_sha256"]
    scientific = _self_hashed(
        {
            key: value
            for key, value in scientific.items()
            if key != "scientific_authority_root_sha256"
        },
        "scientific_authority_root_sha256",
    )

    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="Q static runner binding violates its exact outcome-blind policy",
    ):
        bootstrap._require_scientific_authority(
            scientific,
            project_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("flat_field", "replacement"),
    [
        ("technical_authority_directory", "other-technical-authority"),
        ("technical_authority_artifact_root_sha256", "0" * 64),
    ],
)
def test_bootstrap_rejects_other_rehashed_flat_t0_mismatches(
    tmp_path: Path,
    flat_field: str,
    replacement: str,
) -> None:
    scientific = _synthetic_scientific_authority(tmp_path)
    raw_static = scientific["static_runner_binding"]
    assert isinstance(raw_static, dict)
    static = dict(raw_static)
    static[flat_field] = (
        str(tmp_path / "artifacts" / "original_confirmatory_technical_authorities" / replacement)
        if flat_field == "technical_authority_directory"
        else replacement
    )
    static = _self_hashed(
        {key: value for key, value in static.items() if key != "binding_sha256"},
        "binding_sha256",
    )
    scientific["static_runner_binding"] = static
    scientific["static_runner_binding_sha256"] = static["binding_sha256"]
    scientific = _self_hashed(
        {
            key: value
            for key, value in scientific.items()
            if key != "scientific_authority_root_sha256"
        },
        "scientific_authority_root_sha256",
    )

    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="Q static runner binding violates its exact outcome-blind policy",
    ):
        bootstrap._require_scientific_authority(
            scientific,
            project_root=tmp_path,
        )


def test_bootstrap_rejects_rehashed_permissive_published_t0_binding(
    tmp_path: Path,
) -> None:
    scientific = _synthetic_scientific_authority(tmp_path)
    raw_static = scientific["static_runner_binding"]
    assert isinstance(raw_static, dict)
    static = dict(raw_static)
    raw_published = static["published_technical_authority_lifecycle_binding"]
    assert isinstance(raw_published, dict)
    published = dict(raw_published)
    published["cleanup_allowed"] = True
    published = _self_hashed(
        {key: value for key, value in published.items() if key != "binding_sha256"},
        "binding_sha256",
    )
    static["published_technical_authority_lifecycle_binding"] = published
    static = _self_hashed(
        {key: value for key, value in static.items() if key != "binding_sha256"},
        "binding_sha256",
    )
    scientific["static_runner_binding"] = static
    scientific["static_runner_binding_sha256"] = static["binding_sha256"]
    scientific = _self_hashed(
        {
            key: value
            for key, value in scientific.items()
            if key != "scientific_authority_root_sha256"
        },
        "scientific_authority_root_sha256",
    )

    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="published technical authority lifecycle binding violates its exact one-use policy",
    ):
        bootstrap._require_scientific_authority(
            scientific,
            project_root=tmp_path,
        )


def test_bootstrap_rejects_published_t0_without_review_attempt_claim(
    tmp_path: Path,
) -> None:
    scientific = _synthetic_scientific_authority(tmp_path)
    raw_static = scientific["static_runner_binding"]
    assert isinstance(raw_static, dict)
    static = dict(raw_static)
    raw_published = static["published_technical_authority_lifecycle_binding"]
    assert isinstance(raw_published, dict)
    published = dict(raw_published)
    published.pop("review_attempt_claim_sha256")
    published = _self_hashed(
        {key: value for key, value in published.items() if key != "binding_sha256"},
        "binding_sha256",
    )
    static["published_technical_authority_lifecycle_binding"] = published
    static = _self_hashed(
        {key: value for key, value in static.items() if key != "binding_sha256"},
        "binding_sha256",
    )
    scientific["static_runner_binding"] = static
    scientific["static_runner_binding_sha256"] = static["binding_sha256"]
    scientific = _self_hashed(
        {
            key: value
            for key, value in scientific.items()
            if key != "scientific_authority_root_sha256"
        },
        "scientific_authority_root_sha256",
    )

    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="published technical authority lifecycle binding violates its exact field set",
    ):
        bootstrap._require_scientific_authority(
            scientific,
            project_root=tmp_path,
        )


def test_bootstrap_rejects_self_hashed_static_v2_without_fallback(
    tmp_path: Path,
) -> None:
    scientific = _synthetic_scientific_authority(tmp_path)
    raw_static = scientific["static_runner_binding"]
    assert isinstance(raw_static, dict)
    static = dict(raw_static)
    static["schema_version"] = 2
    static["policy"] = "original_confirmatory_static_runner_binding_v2"
    static = _self_hashed(
        {key: value for key, value in static.items() if key != "binding_sha256"},
        "binding_sha256",
    )
    scientific["static_runner_binding"] = static
    scientific["static_runner_binding_sha256"] = static["binding_sha256"]
    scientific = _self_hashed(
        {
            key: value
            for key, value in scientific.items()
            if key != "scientific_authority_root_sha256"
        },
        "scientific_authority_root_sha256",
    )

    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="Q static runner binding violates its exact outcome-blind policy",
    ):
        bootstrap._require_scientific_authority(
            scientific,
            project_root=tmp_path,
        )


def test_self_consistent_permissive_scientific_request_is_rejected(
    tmp_path: Path,
) -> None:
    scientific_authority = _synthetic_scientific_authority(tmp_path)
    job: dict[str, object] = {
        "job_id": "synthetic-job",
        "attempt_id": "synthetic-attempt",
        "run_id": "synthetic-run",
    }
    lineage: dict[str, object] = {
        "execution_mode": "fresh",
        "retry_of_run_id": None,
    }
    request = _synthetic_scientific_request_projection(
        project_root=tmp_path,
        scientific_authority=scientific_authority,
        job=job,
        lineage=lineage,
        plan_sha256=hashlib.sha256(b"synthetic:plan").hexdigest(),
    )
    request["publication_performed"] = True
    request_unsigned = {key: item for key, item in request.items() if key != "projection_sha256"}
    request["projection_sha256"] = _authority_root(request_unsigned)

    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="E scientific request projection violates its exact typed policy",
    ):
        bootstrap._require_scientific_request_projection(
            request,
            scientific_authority=scientific_authority,
            e_job=job,
            e_lineage=lineage,
        )


def _projection_suffix(mode: str, job_directory: Path) -> list[str]:
    if mode == "run-confirmatory":
        return []
    if mode == "verify-preterminal":
        return [
            "--run-spec",
            str(job_directory / "run_spec.json"),
            "--launch-intent",
            str(job_directory / "launch_intent.json"),
            "--process-started",
            str(job_directory / "process_started.json"),
            "--preterminal-pin",
            str(job_directory / "preterminal_pin.json"),
        ]
    return [
        "--supervisor-terminal",
        str(job_directory / "terminal_receipt.json"),
        "--verifier-stdout",
        str(job_directory / "verifier.stdout.log"),
        "--preterminal-pin",
        str(job_directory / "preterminal_pin.json"),
        "--composed-terminal",
        str(job_directory / "composed_terminal.json"),
    ]


def _synthetic_project_python(project_root: Path) -> Path:
    destination = project_root / ".venv" / "Scripts" / "python.exe"
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(sys.executable), destination)
    source_configuration = Path(sys.executable).parent.parent / "pyvenv.cfg"
    shutil.copy2(source_configuration, project_root / ".venv" / "pyvenv.cfg")
    return destination


def _prepare_exact_q_e(
    capsule: Path,
    *,
    cwd: Path,
    mode: str,
    runtime_python_override: Path | None = None,
) -> list[str]:
    project_root = capsule.parents[3]
    assert capsule == (
        project_root
        / "artifacts"
        / "execution_capsules"
        / capsule.parent.name
        / "original_confirmatory.pyz"
    )
    supervisor_state_root = project_root / "synthetic-supervisor-state"
    job_id = "synthetic-job-pending-q-derivation"
    job_directory = supervisor_state_root / "jobs" / job_id
    e_intent = job_directory / "e_intent.json"
    q_path = (
        project_root
        / "artifacts"
        / "resource_control"
        / "original_confirmatory_q_replacement_v2.json"
    )
    q_path.parent.mkdir(parents=True, exist_ok=True)
    capsule_payload = capsule.read_bytes()
    capsule_sha256 = hashlib.sha256(capsule_payload).hexdigest()
    with zipfile.ZipFile(capsule) as archive:
        internal_manifest_sha256 = hashlib.sha256(archive.read(MANIFEST_NAME)).hexdigest()
        capsule_policy_sha256 = hashlib.sha256(
            archive.read("aanca_capsule/capsule_policy.json")
        ).hexdigest()
        entry_contract_sha256 = hashlib.sha256(
            archive.read("aanca_capsule/entry_contract.json")
        ).hexdigest()
    python_path = _synthetic_project_python(project_root)
    python_payload = python_path.read_bytes()
    python_sha256 = hashlib.sha256(python_payload).hexdigest()
    runtime_python_path = runtime_python_override or Path(
        str(getattr(sys, "_base_executable", sys.executable))
    )
    runtime_python_payload = runtime_python_path.read_bytes()
    runtime_python_sha256 = hashlib.sha256(runtime_python_payload).hexdigest()
    capsule_leaf = _leaf_lease(capsule, interpreter=False)
    capsule_ancestors = _ancestor_lease(
        [
            project_root,
            project_root / "artifacts",
            project_root / "artifacts" / "execution_capsules",
            capsule.parent,
        ],
        interpreter=False,
    )
    python_leaf = _leaf_lease(python_path, interpreter=True)
    python_ancestors = _ancestor_lease(
        [
            project_root,
            project_root / ".venv",
            project_root / ".venv" / "Scripts",
        ],
        interpreter=True,
    )
    user_profile = Path.home().resolve()
    if runtime_python_override is None:
        runtime_parent = runtime_python_path.parent.resolve()
        try:
            relative_parent = runtime_parent.relative_to(user_profile)
            runtime_anchor = user_profile
        except ValueError:
            runtime_anchor = Path(runtime_parent.anchor)
            relative_parent = runtime_parent.relative_to(runtime_anchor)
        runtime_python_ancestor_paths = [runtime_anchor]
        for part in relative_parent.parts:
            runtime_python_ancestor_paths.append(runtime_python_ancestor_paths[-1] / part)
        assert runtime_parent == runtime_python_ancestor_paths[-1]
    else:
        runtime_python_ancestor_paths = [runtime_python_path.parent]
    runtime_python_leaf = _leaf_lease(
        runtime_python_path,
        interpreter=True,
        policy="original_confirmatory_runtime_interpreter_retained_file_lease_v1",
    )
    runtime_python_ancestors = _ancestor_lease(
        runtime_python_ancestor_paths,
        interpreter=True,
        policy="original_confirmatory_runtime_interpreter_ancestor_lease_v1",
    )
    command = _command_derivation_contract()
    plan_sha256 = hashlib.sha256(b"synthetic-plan").hexdigest()
    runtime_release_root_sha256 = hashlib.sha256(b"synthetic-runtime").hexdigest()
    terminal_release_root_sha256 = hashlib.sha256(b"synthetic-terminal").hexdigest()
    publication_records = [
        _lease_record(project_root),
        _lease_record(project_root / "artifacts"),
        _lease_record(project_root / "artifacts" / "resource_control"),
    ]
    publication_lease = {
        "schema_version": 1,
        "policy": "original_confirmatory_control_publication_ancestor_lease_v1",
        "project_root": str(project_root),
        "records": publication_records,
        "record_count": len(publication_records),
        "records_root_sha256": _authority_root(publication_records),
        "directory_access_mask": 0x80000080,
        "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        "delete_share": False,
        "write_access": False,
        "owner_process_identity_required": True,
        "handle_slot_per_record_required": True,
        "continuous_overlap_through_independent_verification_required": True,
        "continuous_overlap_into_supervisor_required": True,
        "acquisition_disposition": (
            "opened_before_q_create_new_retained_through_verifier_and_supervisor_overlap_v1"
        ),
    }
    execution_capsule = _self_hashed(
        {
            "schema_version": 1,
            "policy": "original_confirmatory_sealed_execution_capsule_v1",
            "path": str(capsule),
            "size_bytes": len(capsule_payload),
            "sha256": capsule_sha256,
            "internal_manifest_sha256": internal_manifest_sha256,
            "capsule_policy_sha256": capsule_policy_sha256,
            "entry_contract_sha256": entry_contract_sha256,
            "plan_sha256": plan_sha256,
            "runtime_release_root_sha256": runtime_release_root_sha256,
            "terminal_release_root_sha256": terminal_release_root_sha256,
            "python_path": str(python_path),
            "python_sha256": python_sha256,
            "python_runtime_resolution_policy": (
                "windows_venv_redirector_native_base_executable_v1"
            ),
            "runtime_python_path": str(runtime_python_path),
            "runtime_python_sha256": runtime_python_sha256,
            "runtime_python_lease_identity": runtime_python_leaf,
            "runtime_python_lease_identity_root_sha256": _authority_root(runtime_python_leaf),
            "runtime_python_ancestor_lease": runtime_python_ancestors,
            "runtime_python_ancestor_lease_root_sha256": _authority_root(runtime_python_ancestors),
            "python_isolated_flags": ["-I", "-B"],
            "allowed_modes": [
                "run-confirmatory",
                "verify-preterminal",
                "verify-terminal",
            ],
            "capsule_lease_identity": capsule_leaf,
            "capsule_lease_identity_root_sha256": _authority_root(capsule_leaf),
            "capsule_ancestor_lease": capsule_ancestors,
            "capsule_ancestor_lease_root_sha256": _authority_root(capsule_ancestors),
            "python_lease_identity": python_leaf,
            "python_lease_identity_root_sha256": _authority_root(python_leaf),
            "python_ancestor_lease": python_ancestors,
            "python_ancestor_lease_root_sha256": _authority_root(python_ancestors),
        },
        "contract_sha256",
    )
    external_release_root_sha256 = hashlib.sha256(
        b"synthetic-external-control-plane-release"
    ).hexdigest()
    supervisor_code_root = (
        supervisor_state_root.parent
        / "synthetic-control-plane"
        / "releases"
        / external_release_root_sha256
        / "supervisor"
    )
    _write(supervisor_code_root / "aanca_supervisor.py", b"synthetic-supervisor")
    _write(supervisor_code_root / "launch_hidden.ps1", b"synthetic-launcher")
    terminal_client_path = _write(
        supervisor_code_root / "terminal_client_launcher_v1.py",
        b"synthetic-terminal-client",
    )
    os.chmod(terminal_client_path, stat.S_IREAD)
    release = _supervisor_release(
        execution_capsule,
        supervisor_state_root=supervisor_state_root,
        terminal_client_identity=_terminal_client_physical_identity(terminal_client_path),
        terminal_client_ancestor=_terminal_client_ancestor_lease(supervisor_code_root),
    )
    scientific_authority = _synthetic_scientific_authority(project_root)
    codex_handoff_base = _synthetic_codex_handoff_base(project_root)
    q_base: dict[str, object] = {
        "schema_version": 2,
        "policy": "original_confirmatory_q_replacement_v2",
        "authority_disposition": ("one_create_new_q_publication_for_exact_bound_inputs_v1"),
        "q_path": str(q_path),
        "project_root": str(project_root),
        "scientific_authority": scientific_authority,
        "publication_ancestor_lease": publication_lease,
        "publication_ancestor_lease_root_sha256": _authority_root(publication_lease),
        "execution_capsule": execution_capsule,
        "command_derivation_contract": command,
        "supervisor_release": release,
        "codex_handoff_base_authority": codex_handoff_base,
    }
    q_base_root = _authority_root(q_base)
    attempt_id = "ocq-" + ("a" * 32)
    attempt_root_preimage: dict[str, object] = {
        "schema_version": 1,
        "policy": "original_confirmatory_q_attempt_identity_derivation_v1",
        "attempt_id": attempt_id,
        "q_base_authority_root_sha256": q_base_root,
        "execution_mode": "fresh",
        "retry_of_run_id": None,
    }
    attempt_identity_root = _authority_root(attempt_root_preimage)
    launch_nonce = _authority_root(
        {
            "schema_version": 1,
            "policy": "original_confirmatory_q_attempt_launch_nonce_derivation_v1",
            "attempt_identity_root_sha256": attempt_identity_root,
        }
    )
    job_id = f"oc-{attempt_identity_root}"
    run_id = f"original-confirmatory-{attempt_identity_root}"
    attempt_identity_projection = {
        **attempt_root_preimage,
        "attempt_identity_root_sha256": attempt_identity_root,
        "job_id": job_id,
        "run_id": run_id,
        "launch_nonce": launch_nonce,
    }
    control_staging_dir = supervisor_state_root / "control_staging" / job_id
    job_directory = supervisor_state_root / "jobs" / job_id
    job_directory.mkdir(parents=True, exist_ok=True)
    control_staging_dir.mkdir(parents=True, exist_ok=True)
    e_intent = control_staging_dir / "e_intent.json"
    control_staging_projection: dict[str, object] = {
        "schema_version": 2,
        "policy": "original_confirmatory_control_staging_v2",
        "supervisor_state_root": str(supervisor_state_root),
        "control_staging_dir": str(control_staging_dir),
        "final_job_dir": str(job_directory),
        "staging_attempt_path": str(control_staging_dir / "staging_attempt.json"),
        "e_intent_path": str(control_staging_dir / "e_intent.json"),
        "launch_authorization_path": str(control_staging_dir / "launch_authorization.json"),
        "supervisor_launch_spec_path": str(control_staging_dir / "supervisor_launch_spec.json"),
        "staging_ready_path": str(control_staging_dir / "staging_ready.json"),
        "exact_file_allowlist": [
            "staging_attempt.json",
            "e_intent.json",
            "launch_authorization.json",
            "supervisor_launch_spec.json",
            "staging_ready.json",
        ],
    }
    expected_launch_environment, process_environment_binding = _synthetic_launch_environment(
        launch_nonce
    )
    codex_handoff_creation = _synthetic_codex_handoff_creation(
        codex_handoff_base,
        output_path=job_directory / "codex_handoff_attempt_authority.json",
    )
    q_without_root: dict[str, object] = {
        **q_base,
        "codex_handoff_attempt_creation_authority_payload_sha256": (
            codex_handoff_creation["payload_sha256"]
        ),
        "q_base_authority_root_sha256": q_base_root,
        "attempt_identity_projection": attempt_identity_projection,
        "attempt_identity_root_sha256": attempt_identity_root,
        "control_staging_projection": control_staging_projection,
        "control_staging_projection_sha256": _authority_root(control_staging_projection),
        "expected_launch_environment": expected_launch_environment,
    }
    q = {
        **q_without_root,
        "q_authority_root_sha256": _authority_root(q_without_root),
    }
    q_payload = _authority_json_line(q)
    q_path.write_bytes(q_payload)
    os.chmod(q_path, stat.S_IREAD)
    q_root = q["q_authority_root_sha256"]
    common_after = [
        "--q-authority-root-sha256",
        q_root,
        "--launch-nonce",
        launch_nonce,
        "--supervisor-job-id",
        job_id,
        "--supervisor-job-dir",
        str(job_directory),
        "--attempt-id",
        attempt_id,
        "--run-id",
        run_id,
        "--execution-mode",
        "fresh",
    ]
    projections: dict[str, object] = {}
    for projection_mode in (
        "run-confirmatory",
        "verify-preterminal",
        "verify-terminal",
    ):
        projection = {
            "schema_version": 1,
            "policy": "original_confirmatory_capsule_command_projection_v1",
            "capsule_mode": projection_mode,
            "program_path": str(python_path),
            "program_sha256": python_sha256,
            "python_isolated_flags": ["-I", "-B"],
            "capsule_path": str(capsule),
            "capsule_sha256": capsule_sha256,
            "cwd": str(project_root),
            "argv_prefix": [
                str(python_path),
                "-I",
                "-B",
                str(capsule),
                projection_mode,
            ],
            "tail_argv_before_e_file_sha256": [
                "--e-intent",
                str(e_intent),
                "--e-intent-sha256",
            ],
            "tail_argv_between_e_hashes": ["--e-intent-core-sha256"],
            "tail_argv_after_e_core_sha256": (
                common_after + _projection_suffix(projection_mode, job_directory)
            ),
            "e_file_sha256_insertion_policy": (
                "append_value_to_terminal_e_file_sha256_flag_then_continue_v1"
            ),
            "e_core_sha256_insertion_policy": (
                "append_value_to_terminal_e_core_sha256_flag_then_continue_v1"
            ),
        }
        projections[projection_mode] = _self_hashed(
            projection,
            "projection_sha256",
        )
    terminal_projection = projections["verify-terminal"]
    assert isinstance(terminal_projection, dict)
    terminal_environment_sha256 = process_environment_binding[
        "exact_integrity_verifier_environment_sha256"
    ]
    assert isinstance(terminal_environment_sha256, str)
    terminal_custody_projection = _synthetic_terminal_custody_artifact_projection(
        project_root,
        supervisor_release=release,
        supervisor_job_directory=job_directory,
        run_id=run_id,
        verify_terminal_command_projection_sha256=str(terminal_projection["projection_sha256"]),
        verify_terminal_environment_sha256=terminal_environment_sha256,
    )
    job = {
        "schema_version": 1,
        "policy": "original_confirmatory_supervisor_job_binding_v1",
        "job_id": job_id,
        "supervisor_job_dir": str(job_directory),
        "attempt_id": attempt_id,
        "run_id": run_id,
        "launch_nonce": launch_nonce,
        "supervisor_spec_path": str(job_directory / "run_spec.json"),
        "supervisor_spec_schema_version": 3,
        "supervisor_spec_policy": bootstrap._SUPERVISOR_POLICY,
        "supervisor_release_root_sha256": release["supervisor_release_root_sha256"],
        "terminal_custody_authority_projection": terminal_custody_projection,
        "terminal_custody_authority_projection_root_sha256": (
            terminal_custody_projection["projection_root_sha256"]
        ),
    }
    e_consumption_contract = _synthetic_e_consumption_contract(job_directory)
    lineage = {
        "schema_version": 1,
        "policy": "original_confirmatory_execution_lineage_v1",
        "execution_mode": "fresh",
        "retry_of_run_id": None,
    }
    scientific_request_projection = _synthetic_scientific_request_projection(
        project_root=project_root,
        scientific_authority=scientific_authority,
        job=job,
        lineage=lineage,
        plan_sha256=str(release["plan_sha256"]),
    )
    e_without_core: dict[str, object] = {
        "schema_version": 1,
        "policy": "original_confirmatory_e_intent_v1",
        "authority_disposition": ("one_create_new_e_intent_for_exact_supervisor_job_v1"),
        "q_authority": {
            "path": str(q_path),
            "file_sha256": hashlib.sha256(q_payload).hexdigest(),
            "root_sha256": q_root,
        },
        "codex_handoff_attempt_creation_authority": codex_handoff_creation,
        "q_codex_handoff_base_authority_payload_sha256": codex_handoff_base["payload_sha256"],
        "q_codex_handoff_attempt_creation_authority_payload_sha256": (
            codex_handoff_creation["payload_sha256"]
        ),
        "project_root": str(project_root),
        "execution_capsule_contract_sha256": execution_capsule["contract_sha256"],
        "command_derivation_contract_sha256": command["contract_sha256"],
        "supervisor_release": release,
        "job": job,
        "e_consumption_contract": e_consumption_contract,
        "scientific_request_projection": scientific_request_projection,
        "lineage": lineage,
        "expected_launch_environment": expected_launch_environment,
        "process_environment_binding": process_environment_binding,
        "command_projections": projections,
        "attempt_count": 1,
        "max_attempt_count": 1,
        "automatic_retry_allowed": False,
        "scientific_outcomes_read": False,
    }
    e_core_sha256 = _authority_root(e_without_core)
    e = {**e_without_core, "intent_core_sha256": e_core_sha256}
    e_payload = _authority_json_line(e)
    e_intent.write_bytes(e_payload)
    os.chmod(e_intent, stat.S_IREAD)
    e_file_sha256 = hashlib.sha256(e_payload).hexdigest()
    q_e_custody_contract = bootstrap._build_q_e_custody_contract(
        supervisor_job_directory=job_directory,
    )
    q_e_ready_unsigned: dict[str, object] = {
        "schema_version": 1,
        "policy": "original_confirmatory_q_e_supervisor_custody_handoff_v1",
        "message_type": "Q_E_CUSTODY_READY",
        "transport": "bounded_anonymous_pipe_blocking_v1",
        "contract_sha256": q_e_custody_contract["contract_sha256"],
        "supervisor_job_id": job_id,
        "supervisor_process_identity": _q_e_process_identity(
            pid=41001,
            program_path=python_path,
            command_seed=b"synthetic-q-e-supervisor",
        ),
        "controller_process_identity": _q_e_process_identity(
            pid=41002,
            program_path=python_path,
            command_seed=b"synthetic-q-e-controller",
        ),
        "windows_boot_time_utc": "2026-07-30T00:00:00.000000Z",
        "q_authority_root_sha256": q_root,
        "q_file_sha256": hashlib.sha256(q_payload).hexdigest(),
        "e_file_sha256": e_file_sha256,
        "q_leaf_physical_identity": _q_e_control_physical_identity(q_path),
        "q_ancestor_lease": publication_lease,
        "e_leaf_physical_identity": _q_e_control_physical_identity(e_intent),
        "e_ancestor_lease": _q_e_e_ancestor_lease(job_directory, e_intent),
        "q_leaf_handle": 51001,
        "q_ancestor_handles": [51002, 51003, 51004],
        "e_leaf_handle": 51005,
        "e_ancestor_handles": [51006, 51007, 51008],
        "leaf_target_access_mask": 0x80000000,
        "ancestor_target_access_mask": 0x80000080,
        "duplicate_options": 0,
        "close_source": False,
        "source_custody_retained_until_supervisor_ack": True,
        "supervisor_retention_policy": (
            "through_science_preterminal_terminal_postwake_and_terminal_seal_v1"
        ),
        "independent_verifier_receipt_sha256": scientific_authority[
            "independent_review_receipt_sha256"
        ],
        "scientific_inputs_before_ack_allowed": False,
        "automatic_retry_allowed": False,
    }
    q_e_custody_handoff = {
        **q_e_ready_unsigned,
        "handoff_root_sha256": _authority_root(q_e_ready_unsigned),
    }
    q_e_full_receipt = bootstrap._build_q_e_custody_receipt(
        contract=q_e_custody_contract,
        ready=q_e_custody_handoff,
    )
    q_e_custody_receipt = {
        "policy": "original_confirmatory_q_e_supervisor_custody_receipt_v1",
        "path": q_e_custody_contract["receipt_path"],
        "file_sha256": hashlib.sha256(_authority_json_line(q_e_full_receipt)).hexdigest(),
        "receipt_root_sha256": q_e_full_receipt["receipt_root_sha256"],
        "handoff_root_sha256": q_e_custody_handoff["handoff_root_sha256"],
    }
    concrete_commands: dict[str, dict[str, object]] = {}
    for projection_mode, raw_projection in projections.items():
        assert isinstance(raw_projection, dict)
        final_argv = [
            *raw_projection["argv_prefix"],
            *raw_projection["tail_argv_before_e_file_sha256"],
            e_file_sha256,
            *raw_projection["tail_argv_between_e_hashes"],
            e_core_sha256,
            *raw_projection["tail_argv_after_e_core_sha256"],
        ]
        assert all(isinstance(item, str) for item in final_argv)
        concrete_commands[projection_mode] = bootstrap._expected_concrete_capsule_command(
            final_argv,
            execution_capsule=execution_capsule,
            project_root=project_root,
        )
    terminal_command = concrete_commands["verify-terminal"]
    seed_unsigned: dict[str, object] = {
        "schema_version": 1,
        "policy": "original_confirmatory_postwake_custody_seed_v1",
        "q_authority_root_sha256": q_root,
        "e_intent_path": str(e_intent),
        "e_intent_file_sha256": e_file_sha256,
        "e_intent_core_sha256": e_core_sha256,
        "supervisor_job_id": job_id,
        "supervisor_job_dir": str(job_directory),
        "supervisor_spec_path": str(job_directory / "run_spec.json"),
        "launch_nonce": launch_nonce,
        "attempt_id": attempt_id,
        "run_id": run_id,
        "execution_mode": "fresh",
        "retry_of_run_id": None,
        "execution_capsule_contract_sha256": execution_capsule["contract_sha256"],
        "capsule_sha256": execution_capsule["sha256"],
        "supervisor_release_root_sha256": release["supervisor_release_root_sha256"],
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
    postwake_custody_seed = {
        **seed_unsigned,
        "seed_sha256": _authority_root(seed_unsigned),
    }
    seed_sha256 = postwake_custody_seed["seed_sha256"]
    assert isinstance(seed_sha256, str)
    terminal_command_sha256 = terminal_command["command_sha256"]
    assert isinstance(terminal_command_sha256, str)
    handshake_unsigned: dict[str, object] = {
        "schema_version": 1,
        "policy": "original_confirmatory_postwake_custody_handshake_contract_v1",
        "supervisor_job_id": job_id,
        "postwake_custody_seed_sha256": seed_sha256,
        "pipe_name": "\\\\.\\pipe\\AANCA-composed-custody-" + seed_sha256,
        "expected_composed_command_sha256": terminal_command_sha256,
        "expected_composed_cwd": str(project_root),
        "expected_composed_environment_sha256": terminal_environment_sha256,
        "readback_receipt_path": str(job_directory / "postwake_composed_readback_receipt.json"),
        "ready_max_bytes": 64 * 1024,
        "ack_max_bytes": 64 * 1024,
        "terminal_client_arrival_timeout_ms": 1_800_000,
        "custody_exchange_timeout_ms": 60_000,
        "overall_timeout_max_ms": 6 * 60 * 60 * 1_000,
        "arrival_and_exchange_waits_event_driven": True,
        "automatic_retry_allowed": False,
    }
    postwake_custody_handshake_contract = _self_hashed(
        handshake_unsigned,
        "contract_sha256",
    )
    terminal_contract_unsigned: dict[str, object] = {
        field: None
        for field in bootstrap._TERMINAL_COMPOSITION_CONTRACT_FIELDS
        if field != "contract_sha256"
    }
    terminal_contract_unsigned.update(
        {
            "schema_version": 1,
            "policy": "original_confirmatory_capsule_terminal_composition_contract_v1",
            "capsule_contract_sha256": execution_capsule["contract_sha256"],
            "capsule_path": execution_capsule["path"],
            "capsule_sha256": execution_capsule["sha256"],
            "capsule_internal_manifest_sha256": execution_capsule["internal_manifest_sha256"],
            "capsule_mode": "verify-terminal",
            "verifier_command": terminal_command,
            "verifier_command_sha256": terminal_command_sha256,
            "terminal_custody_authority_projection": terminal_custody_projection,
            "expected_environment_envelope_sha256": expected_launch_environment["envelope_sha256"],
            "process_environment_binding_sha256": process_environment_binding["binding_sha256"],
            "exact_integrity_verifier_environment_sha256": (terminal_environment_sha256),
            "capsule_lease_identity_root_sha256": execution_capsule[
                "capsule_lease_identity_root_sha256"
            ],
            "capsule_ancestor_lease_root_sha256": execution_capsule[
                "capsule_ancestor_lease_root_sha256"
            ],
            "postwake_custody_seed_sha256": seed_sha256,
            "postwake_custody_handshake_contract_sha256": (
                postwake_custody_handshake_contract["contract_sha256"]
            ),
            "semantic_outcome_read_scope": (
                "integrity_and_completion_evidence_only_no_scientific_outcome_values_v1"
            ),
            "outcome_values_read": False,
            "outcome_values_emitted": False,
            "outcome_values_used_for_selection_or_tuning": False,
            "training_or_model_selection_allowed": False,
            "scientific_publication_allowed": False,
            "automatic_retry_allowed": False,
            "supervisor_terminal_receipt_path": str(job_directory / "terminal_receipt.json"),
            "preterminal_pin_receipt_path": str(job_directory / "preterminal_pin.json"),
            "postwake_input_lease_receipt_path": str(
                job_directory / "postwake_input_lease_receipt.json"
            ),
            "composed_terminal_receipt_path": str(job_directory / "composed_terminal.json"),
            "postwake_composed_readback_receipt_path": str(
                job_directory / "postwake_composed_readback_receipt.json"
            ),
            "verifier_stdout_path": str(job_directory / "verifier.stdout.log"),
            "verifier_stderr_path": str(job_directory / "verifier.stderr.log"),
        }
    )
    terminal_composition_contract = _self_hashed(
        terminal_contract_unsigned,
        "contract_sha256",
    )
    artifact_instance = terminal_custody_projection["outcome_blind_expected_artifact_instance"]
    assert isinstance(artifact_instance, dict)
    staging_attempt_unsigned = {
        "schema_version": 1,
        "policy": "original_confirmatory_control_staging_attempt_v1",
        "job_id": job_id,
    }
    staging_attempt = {
        **staging_attempt_unsigned,
        "attempt_marker_root_sha256": _authority_root(staging_attempt_unsigned),
    }
    staging_attempt_path = control_staging_dir / "staging_attempt.json"
    staging_attempt_path.write_bytes(_authority_json_line(staging_attempt))
    os.chmod(staging_attempt_path, stat.S_IREAD)
    launch_authorization = {
        "schema_version": 1,
        "policy": "aanca_supervisor_one_attempt_authorization_v2",
        "job_id": job_id,
        "automatic_retry_allowed": False,
    }
    launch_authorization_path = control_staging_dir / "launch_authorization.json"
    launch_authorization_payload = _authority_json_line(launch_authorization)
    launch_authorization_path.write_bytes(launch_authorization_payload)
    os.chmod(launch_authorization_path, stat.S_IREAD)
    launch_authorization_file_sha256 = hashlib.sha256(launch_authorization_payload).hexdigest()
    external_codex_handoff = {
        "policy": bootstrap._EXTERNAL_CODEX_HANDOFF_POLICY,
        "staged_e_intent_path": str(e_intent),
        "staged_e_intent_file_sha256": e_file_sha256,
        "staged_e_intent_core_root_sha256": e_core_sha256,
        "attempt_creation_authority_payload_sha256": codex_handoff_creation["payload_sha256"],
        "attempt_authority_output_path": str(
            job_directory / "codex_handoff_attempt_authority.json"
        ),
        "terminal_handoff_receipt_output_path": str(
            job_directory / "external_codex_terminal_handoff.json"
        ),
        "internal_codex_wake_allowed": False,
        "legacy_handoff_session_allowed": False,
        "single_wake_owner": bootstrap._EXTERNAL_CODEX_HANDOFF_SINGLE_WAKE_OWNER,
    }
    canonical_spec = {
        "schema_version": 3,
        "policy": bootstrap._SUPERVISOR_POLICY,
        "job_id": job_id,
        "process_kind": "confirmatory",
        "external_control_plane_release_root_sha256": release[
            "external_control_plane_release_root_sha256"
        ],
        "external_control_plane_publication_id": release["external_control_plane_publication_id"],
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
        "project_root": str(project_root),
        "program_path": execution_capsule["python_path"],
        "program_sha256": execution_capsule["python_sha256"],
        "argv": concrete_commands["run-confirmatory"]["argv"],
        "command": concrete_commands["run-confirmatory"],
        "integrity_verifier": {
            key: item
            for key, item in concrete_commands["verify-preterminal"].items()
            if key != "command_sha256"
        },
        "expected_environment": expected_launch_environment,
        "process_environment_binding": process_environment_binding,
        "e_consumption_contract": e_consumption_contract,
        "supervisor_launcher_sha256": release["supervisor_launcher_sha256"],
        "capsule_lease_identity": execution_capsule["capsule_lease_identity"],
        "capsule_lease_identity_root_sha256": execution_capsule[
            "capsule_lease_identity_root_sha256"
        ],
        "capsule_ancestor_lease": execution_capsule["capsule_ancestor_lease"],
        "capsule_ancestor_lease_root_sha256": execution_capsule[
            "capsule_ancestor_lease_root_sha256"
        ],
        "python_lease_identity": execution_capsule["python_lease_identity"],
        "python_lease_identity_root_sha256": execution_capsule["python_lease_identity_root_sha256"],
        "python_ancestor_lease": execution_capsule["python_ancestor_lease"],
        "python_ancestor_lease_root_sha256": execution_capsule["python_ancestor_lease_root_sha256"],
        "python_runtime_resolution_policy": execution_capsule["python_runtime_resolution_policy"],
        "runtime_python_path": execution_capsule["runtime_python_path"],
        "runtime_python_sha256": execution_capsule["runtime_python_sha256"],
        "runtime_python_lease_identity": execution_capsule["runtime_python_lease_identity"],
        "runtime_python_lease_identity_root_sha256": execution_capsule[
            "runtime_python_lease_identity_root_sha256"
        ],
        "runtime_python_ancestor_lease": execution_capsule["runtime_python_ancestor_lease"],
        "runtime_python_ancestor_lease_root_sha256": execution_capsule[
            "runtime_python_ancestor_lease_root_sha256"
        ],
        "expected_artifacts": artifact_instance["expected_artifacts"],
        "required_success_roles": artifact_instance["required_success_roles"],
        "terminal_composition_contract": terminal_composition_contract,
        "postwake_custody_seed": postwake_custody_seed,
        "postwake_custody_handshake_contract": (postwake_custody_handshake_contract),
        "max_attempt_count": 1,
        "automatic_retry_allowed": False,
        "authorization": {
            "path": str(launch_authorization_path),
            "sha256": launch_authorization_file_sha256,
        },
        "codex": None,
        "external_codex_handoff": external_codex_handoff,
        "codex_wake_timeout_seconds": 1,
        "handoff_session": None,
        "main_timeout_ms": 1,
        "max_log_bytes": 1,
        "postwake_input_lease_contract": {},
        "preterminal_overlap_handshake_contract": {},
        "preterminal_pin_contract": {},
        "q_e_custody_contract": q_e_custody_contract,
        "q_e_custody_handoff": q_e_custody_handoff,
        "q_e_custody_receipt": q_e_custody_receipt,
        "verifier_timeout_ms": 1,
    }
    assert set(canonical_spec) == bootstrap._SUPERVISOR_CANONICAL_SPEC_FIELDS
    source_spec_path = control_staging_dir / "supervisor_launch_spec.json"
    source_spec_payload = _authority_json_line(canonical_spec)
    source_spec_path.write_bytes(source_spec_payload)
    os.chmod(source_spec_path, stat.S_IREAD)
    staging_ready_unsigned = {
        "schema_version": 1,
        "policy": "original_confirmatory_control_staging_ready_v1",
        "job_id": job_id,
        "supervisor_process_identity": q_e_ready_unsigned["supervisor_process_identity"],
    }
    staging_ready = {
        **staging_ready_unsigned,
        "ready_marker_root_sha256": _authority_root(staging_ready_unsigned),
    }
    staging_ready_path = control_staging_dir / "staging_ready.json"
    staging_ready_path.write_bytes(_authority_json_line(staging_ready))
    os.chmod(staging_ready_path, stat.S_IREAD)
    staging_paths = [
        staging_attempt_path,
        e_intent,
        launch_authorization_path,
        source_spec_path,
        staging_ready_path,
    ]
    staging_roles = list(bootstrap._CONTROL_STAGING_OUTER_FILE_ROLES)
    staging_files = []
    for path, role in zip(staging_paths, staging_roles, strict=True):
        identity = _control_staging_physical_identity(path, role=role)
        staging_files.append(
            {
                "role": role,
                "name": path.name,
                "path": str(path),
                "size_bytes": identity["size_bytes"],
                "file_sha256": identity["sha256"],
                "physical_identity": identity,
                "physical_identity_root_sha256": _authority_root(identity),
            }
        )
    control_staging_ancestor_lease = _control_staging_ancestor_lease(
        supervisor_state_root,
        control_staging_dir,
    )
    control_staging_unsigned = {
        "schema_version": 1,
        "policy": bootstrap._CONTROL_STAGING_OUTER_BINDING_POLICY,
        "job_id": job_id,
        "supervisor_root": str(supervisor_state_root),
        "control_staging_root": str(control_staging_dir.parent),
        "control_staging_dir": str(control_staging_dir),
        "control_staging_projection": control_staging_projection,
        "control_staging_projection_sha256": _authority_root(control_staging_projection),
        "expected_complete_leaf_names": list(bootstrap._CONTROL_STAGING_EXACT_FILE_ALLOWLIST),
        "publication_order": list(bootstrap._CONTROL_STAGING_EXACT_FILE_ALLOWLIST),
        "file_count": 5,
        "files": staging_files,
        "control_staging_ancestor_lease": control_staging_ancestor_lease,
        "control_staging_ancestor_lease_root_sha256": _authority_root(
            control_staging_ancestor_lease
        ),
        "staging_attempt_root_sha256": staging_attempt["attempt_marker_root_sha256"],
        "staging_ready_root_sha256": staging_ready["ready_marker_root_sha256"],
        "source_path": str(source_spec_path),
        "source_size_bytes": len(source_spec_payload),
        "source_file_sha256": hashlib.sha256(source_spec_payload).hexdigest(),
        "source_canonical_bytes_sha256": hashlib.sha256(source_spec_payload).hexdigest(),
        "source_bytes_equal_canonical_spec_serialization": True,
        "e_intent_path": str(e_intent),
        "e_intent_file_sha256": e_file_sha256,
        "launch_authorization_path": str(launch_authorization_path),
        "launch_authorization_file_sha256": launch_authorization_file_sha256,
        "supervisor_process_identity": q_e_ready_unsigned["supervisor_process_identity"],
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
    control_staging_outer = {
        **control_staging_unsigned,
        "binding_root_sha256": _authority_root(control_staging_unsigned),
    }
    supervisor_identity = {
        "path": release["supervisor_source_path"],
        "sha256": release["supervisor_source_sha256"],
    }
    run_spec_payload = {
        "schema_version": 3,
        "policy": bootstrap._SUPERVISOR_POLICY,
        "source_path": str(source_spec_path),
        "source_file_sha256": hashlib.sha256(source_spec_payload).hexdigest(),
        "canonical_spec": canonical_spec,
        "canonical_spec_sha256": _supervisor_file_root(canonical_spec),
        "supervisor": supervisor_identity,
        "frozen_at_utc": "2026-07-30T00:00:00.000000Z",
        "control_staging": control_staging_outer,
    }
    run_spec_envelope = {
        "schema_version": 3,
        "payload": run_spec_payload,
        "payload_sha256": _supervisor_file_root(run_spec_payload),
    }
    (job_directory / "run_spec.json").write_bytes(_authority_json_line(run_spec_envelope))
    claim = job_directory / "e_intent_consumed.json"
    if mode != "run-confirmatory":
        claim.write_bytes(b'{"synthetic_claim":true}\n')
        os.chmod(claim, stat.S_IREAD)
    selected_projection = projections[mode]
    assert isinstance(selected_projection, dict)
    before_e_hash = selected_projection["tail_argv_before_e_file_sha256"]
    between_e_hashes = selected_projection["tail_argv_between_e_hashes"]
    after_e_hash = selected_projection["tail_argv_after_e_core_sha256"]
    assert isinstance(before_e_hash, list)
    assert isinstance(between_e_hashes, list)
    assert isinstance(after_e_hash, list)
    return [
        mode,
        *before_e_hash,
        hashlib.sha256(e_payload).hexdigest(),
        *between_e_hashes,
        e_core_sha256,
        *after_e_hash,
    ]


def _run_capsule(capsule: Path, *, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    try:
        common_tail = _prepare_exact_q_e(
            capsule,
            cwd=cwd,
            mode="verify-terminal",
        )
        program = str(capsule.parents[3] / ".venv" / "Scripts" / "python.exe")
    except (AssertionError, IndexError):
        job_directory = cwd / "synthetic-supervisor" / "jobs" / "synthetic-job"
        job_directory.mkdir(parents=True, exist_ok=True)
        e_intent = (
            job_directory.parent.parent
            / bootstrap._CONTROL_STAGING_DIRECTORY_NAME
            / job_directory.name
            / bootstrap._E_INTENT_FILENAME
        )
        e_intent.parent.mkdir(parents=True, exist_ok=True)
        e_intent.write_bytes(b'{"synthetic":true}\n')
        os.chmod(e_intent, stat.S_IREAD)
        claim = job_directory / "e_intent_consumed.json"
        claim.write_bytes(b'{"synthetic_claim":true}\n')
        os.chmod(claim, stat.S_IREAD)
        common_tail = [
            "verify-terminal",
            "--e-intent",
            str(e_intent),
            "--e-intent-sha256",
            "1" * 64,
            "--e-intent-core-sha256",
            "2" * 64,
            "--q-authority-root-sha256",
            "3" * 64,
            "--launch-nonce",
            "4" * 64,
            "--supervisor-job-id",
            "synthetic-job",
            "--supervisor-job-dir",
            str(job_directory),
            "--attempt-id",
            "synthetic-attempt",
            "--run-id",
            "synthetic-run",
            "--execution-mode",
            "fresh",
            *_projection_suffix("verify-terminal", job_directory),
        ]
        program = (
            sys.executable
            if os.name == "nt"
            else str(getattr(sys, "_base_executable", sys.executable))
        )
    return _invoke_capsule(
        capsule,
        cwd=cwd,
        program=program,
        tail=common_tail,
    )


def _synthetic_prepared_job_directory(project_root: Path) -> Path:
    jobs_root = project_root / "synthetic-supervisor-state" / "jobs"
    job_directories = [
        path for path in jobs_root.iterdir() if path.is_dir() and path.name.startswith("oc-")
    ]
    assert len(job_directories) == 1
    return job_directories[0]


def _synthetic_staged_e_intent_path(job_directory: Path) -> Path:
    return (
        job_directory.parent.parent
        / bootstrap._CONTROL_STAGING_DIRECTORY_NAME
        / job_directory.name
        / bootstrap._E_INTENT_FILENAME
    )


def _invoke_capsule(
    capsule: Path,
    *,
    cwd: Path,
    program: str,
    tail: list[str],
) -> subprocess.CompletedProcess[bytes]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(cwd)
    launch_cwd = (
        capsule.parents[3]
        if (
            capsule.name == "original_confirmatory.pyz"
            and capsule.parents[1].name == "execution_capsules"
            and capsule.parents[2].name == "artifacts"
        )
        else cwd
    )
    return subprocess.run(
        [
            program,
            "-I",
            "-B",
            str(capsule),
            *tail,
        ],
        cwd=launch_cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=30,
    )


def _rewrite_synthetic_run_spec(
    run_spec_path: Path,
    *,
    mutate_spec: Callable[[dict[str, object]], None],
) -> None:
    envelope = json.loads(run_spec_path.read_text(encoding="utf-8"))
    assert isinstance(envelope, dict)
    payload = envelope["payload"]
    assert isinstance(payload, dict)
    canonical_spec = payload["canonical_spec"]
    assert isinstance(canonical_spec, dict)
    mutate_spec(canonical_spec)
    source_path = Path(str(payload["source_path"]))
    os.chmod(source_path, stat.S_IREAD | stat.S_IWRITE)
    source_payload = _authority_json_line(canonical_spec)
    source_path.write_bytes(source_payload)
    os.chmod(source_path, stat.S_IREAD)
    source_sha256 = hashlib.sha256(source_payload).hexdigest()
    control_staging = payload["control_staging"]
    assert isinstance(control_staging, dict)
    files = control_staging["files"]
    assert isinstance(files, list)
    source_record = files[3]
    assert isinstance(source_record, dict)
    source_identity = _control_staging_physical_identity(
        source_path,
        role="supervisor-launch-spec",
    )
    source_record.update(
        {
            "size_bytes": len(source_payload),
            "file_sha256": source_sha256,
            "physical_identity": source_identity,
            "physical_identity_root_sha256": _authority_root(source_identity),
        }
    )
    control_staging["source_size_bytes"] = len(source_payload)
    control_staging["source_file_sha256"] = source_sha256
    control_staging["source_canonical_bytes_sha256"] = source_sha256
    unsigned_control_staging = {
        key: item for key, item in control_staging.items() if key != "binding_root_sha256"
    }
    control_staging["binding_root_sha256"] = _authority_root(unsigned_control_staging)
    payload["source_file_sha256"] = source_sha256
    payload["canonical_spec_sha256"] = _supervisor_file_root(canonical_spec)
    envelope["payload_sha256"] = _supervisor_file_root(payload)
    run_spec_path.write_bytes(_authority_json_line(envelope))


def _reseal_q_e_spec_chain(canonical_spec: dict[str, object]) -> None:
    contract = canonical_spec["q_e_custody_contract"]
    handoff = canonical_spec["q_e_custody_handoff"]
    receipt_binding = canonical_spec["q_e_custody_receipt"]
    assert isinstance(contract, dict)
    assert isinstance(handoff, dict)
    assert isinstance(receipt_binding, dict)

    contract_unsigned = {key: item for key, item in contract.items() if key != "contract_sha256"}
    contract["contract_sha256"] = _authority_root(contract_unsigned)
    handoff["contract_sha256"] = contract["contract_sha256"]
    handoff_unsigned = {key: item for key, item in handoff.items() if key != "handoff_root_sha256"}
    handoff["handoff_root_sha256"] = _authority_root(handoff_unsigned)
    full_receipt = bootstrap._build_q_e_custody_receipt(
        contract=contract,
        ready=handoff,
    )
    receipt_binding.clear()
    receipt_binding.update(
        {
            "policy": "original_confirmatory_q_e_supervisor_custody_receipt_v1",
            "path": contract["receipt_path"],
            "file_sha256": hashlib.sha256(_authority_json_line(full_receipt)).hexdigest(),
            "receipt_root_sha256": full_receipt["receipt_root_sha256"],
            "handoff_root_sha256": handoff["handoff_root_sha256"],
        }
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "contract_schema_bool",
        "handoff_q_leaf_schema_float",
        "handoff_e_ancestor_schema_float",
        "receipt_binding_extra_field",
    ],
)
def test_coherently_resealed_q_e_spec_aliases_stop_before_claim_and_import(
    tmp_path: Path,
    mutation: str,
) -> None:
    files = _synthetic_tree(tmp_path / "source")
    marker = tmp_path / "PROJECT_IMPORTED"
    files["entry"].write_bytes(
        (
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
        ).encode()
        + files["entry"].read_bytes()
    )
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    working = tmp_path / "working"
    working.mkdir()
    tail = _prepare_exact_q_e(
        capsule,
        cwd=working,
        mode="run-confirmatory",
    )
    job_directory = _synthetic_prepared_job_directory(tmp_path)

    def mutate_spec(canonical_spec: dict[str, object]) -> None:
        contract = canonical_spec["q_e_custody_contract"]
        handoff = canonical_spec["q_e_custody_handoff"]
        receipt_binding = canonical_spec["q_e_custody_receipt"]
        assert isinstance(contract, dict)
        assert isinstance(handoff, dict)
        assert isinstance(receipt_binding, dict)
        if mutation == "contract_schema_bool":
            assert contract["schema_version"] == 1
            contract["schema_version"] = True
        elif mutation == "handoff_q_leaf_schema_float":
            q_leaf = handoff["q_leaf_physical_identity"]
            assert isinstance(q_leaf, dict)
            assert q_leaf["schema_version"] == 1
            q_leaf["schema_version"] = 1.0
        elif mutation == "handoff_e_ancestor_schema_float":
            e_ancestor = handoff["e_ancestor_lease"]
            assert isinstance(e_ancestor, dict)
            assert e_ancestor["schema_version"] == 1
            e_ancestor["schema_version"] = 1.0
        elif mutation != "receipt_binding_extra_field":
            raise AssertionError(f"unknown mutation: {mutation}")
        _reseal_q_e_spec_chain(canonical_spec)
        if mutation == "receipt_binding_extra_field":
            receipt_binding["unexpected_alias_carrier"] = False

    _rewrite_synthetic_run_spec(
        job_directory / "run_spec.json",
        mutate_spec=mutate_spec,
    )
    completed = _invoke_capsule(
        capsule,
        cwd=working,
        program=str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        tail=tail,
    )

    assert completed.returncode != 0
    assert completed.stdout == b""
    assert b"Q/E" in completed.stderr
    assert not marker.exists()
    assert not (job_directory / "e_intent_consumed.json").exists()


@pytest.mark.parametrize(
    ("selector", "value"),
    [
        ("metrics", {"auprc": 0.99}),
        ("rankings", ["nucleus-1"]),
        ("predictions", ["class-2"]),
        ("outcome", "favorable"),
    ],
)
def test_self_consistent_run_spec_outcome_selector_stops_before_claim_and_import(
    tmp_path: Path,
    selector: str,
    value: object,
) -> None:
    files = _synthetic_tree(tmp_path / "source")
    marker = tmp_path / "PROJECT_IMPORTED"
    files["entry"].write_bytes(
        (
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
        ).encode()
        + files["entry"].read_bytes()
    )
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    working = tmp_path / "working"
    working.mkdir()
    tail = _prepare_exact_q_e(
        capsule,
        cwd=working,
        mode="run-confirmatory",
    )
    job_directory = _synthetic_prepared_job_directory(tmp_path)

    def mutate_spec(canonical_spec: dict[str, object]) -> None:
        artifacts = canonical_spec["expected_artifacts"]
        assert isinstance(artifacts, list)
        completion = artifacts[2]
        assert isinstance(completion, dict)
        checks = completion["json_equals"]
        assert isinstance(checks, dict)
        checks[selector] = value

    _rewrite_synthetic_run_spec(
        job_directory / "run_spec.json",
        mutate_spec=mutate_spec,
    )
    completed = _invoke_capsule(
        capsule,
        cwd=working,
        program=str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        tail=tail,
    )

    assert completed.returncode != 0
    assert completed.stdout == b""
    assert b"differs from its exact control-only rule" in completed.stderr
    assert not marker.exists()
    assert not (job_directory / "e_intent_consumed.json").exists()


def test_self_consistent_run_spec_terminal_projection_stops_before_claim_and_import(
    tmp_path: Path,
) -> None:
    files = _synthetic_tree(tmp_path / "source")
    marker = tmp_path / "PROJECT_IMPORTED"
    files["entry"].write_bytes(
        (
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
        ).encode()
        + files["entry"].read_bytes()
    )
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    working = tmp_path / "working"
    working.mkdir()
    tail = _prepare_exact_q_e(
        capsule,
        cwd=working,
        mode="run-confirmatory",
    )
    job_directory = _synthetic_prepared_job_directory(tmp_path)

    def mutate_spec(canonical_spec: dict[str, object]) -> None:
        terminal = canonical_spec["terminal_composition_contract"]
        assert isinstance(terminal, dict)
        projection = terminal["terminal_custody_authority_projection"]
        assert isinstance(projection, dict)
        projection["terminal_custody_authority_template_root_sha256"] = "0" * 64
        _rehash_terminal_custody_projection(projection)

    _rewrite_synthetic_run_spec(
        job_directory / "run_spec.json",
        mutate_spec=mutate_spec,
    )
    completed = _invoke_capsule(
        capsule,
        cwd=working,
        program=str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        tail=tail,
    )

    assert completed.returncode != 0
    assert completed.stdout == b""
    assert b"terminal custody projection differs from sealed E" in completed.stderr
    assert not marker.exists()
    assert not (job_directory / "e_intent_consumed.json").exists()


def test_self_consistent_concrete_terminal_command_stops_before_claim_and_import(
    tmp_path: Path,
) -> None:
    files = _synthetic_tree(tmp_path / "source")
    marker = tmp_path / "PROJECT_IMPORTED"
    files["entry"].write_bytes(
        (
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
        ).encode()
        + files["entry"].read_bytes()
    )
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    working = tmp_path / "working"
    working.mkdir()
    tail = _prepare_exact_q_e(
        capsule,
        cwd=working,
        mode="run-confirmatory",
    )
    job_directory = _synthetic_prepared_job_directory(tmp_path)

    def mutate_spec(canonical_spec: dict[str, object]) -> None:
        terminal = canonical_spec["terminal_composition_contract"]
        assert isinstance(terminal, dict)
        command = terminal["verifier_command"]
        assert isinstance(command, dict)
        argv = command["argv"]
        assert isinstance(argv, list)
        run_id_index = argv.index("--run-id") + 1
        argv[run_id_index] = "different-run"
        command_unsigned = {key: item for key, item in command.items() if key != "command_sha256"}
        command["command_sha256"] = _authority_root(command_unsigned)
        terminal["verifier_command_sha256"] = command["command_sha256"]
        terminal_unsigned = {
            key: item for key, item in terminal.items() if key != "contract_sha256"
        }
        terminal["contract_sha256"] = _authority_root(terminal_unsigned)

    _rewrite_synthetic_run_spec(
        job_directory / "run_spec.json",
        mutate_spec=mutate_spec,
    )
    completed = _invoke_capsule(
        capsule,
        cwd=working,
        program=str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        tail=tail,
    )

    assert completed.returncode != 0
    assert completed.stdout == b""
    assert b"verify-terminal command/projection differs from sealed E" in completed.stderr
    assert not marker.exists()
    assert not (job_directory / "e_intent_consumed.json").exists()


def test_coherently_resealed_nested_spec_int_bool_alias_stops_before_claim_and_import(
    tmp_path: Path,
) -> None:
    files = _synthetic_tree(tmp_path / "source")
    marker = tmp_path / "PROJECT_IMPORTED"
    files["entry"].write_bytes(
        (
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
        ).encode()
        + files["entry"].read_bytes()
    )
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    working = tmp_path / "working"
    working.mkdir()
    tail = _prepare_exact_q_e(
        capsule,
        cwd=working,
        mode="run-confirmatory",
    )
    job_directory = _synthetic_prepared_job_directory(tmp_path)

    def mutate_spec(canonical_spec: dict[str, object]) -> None:
        capsule_lease = canonical_spec["capsule_lease_identity"]
        assert isinstance(capsule_lease, dict)
        assert capsule_lease["link_count"] == 1
        capsule_lease["link_count"] = True

    _rewrite_synthetic_run_spec(
        job_directory / "run_spec.json",
        mutate_spec=mutate_spec,
    )
    completed = _invoke_capsule(
        capsule,
        cwd=working,
        program=str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        tail=tail,
    )

    assert completed.returncode != 0
    assert completed.stdout == b""
    assert b"supervisor spec differs from sealed Q/E command or runtime authority" in (
        completed.stderr
    )
    assert not marker.exists()
    assert not (job_directory / "e_intent_consumed.json").exists()


def test_self_consistent_nested_launcher_projection_stops_before_claim_and_import(
    tmp_path: Path,
) -> None:
    files = _synthetic_tree(tmp_path / "source")
    marker = tmp_path / "PROJECT_IMPORTED"
    files["entry"].write_bytes(
        (
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
        ).encode()
        + files["entry"].read_bytes()
    )
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    working = tmp_path / "working"
    working.mkdir()
    tail = _prepare_exact_q_e(
        capsule,
        cwd=working,
        mode="run-confirmatory",
    )
    job_directory = _synthetic_prepared_job_directory(tmp_path)

    def mutate_spec(canonical_spec: dict[str, object]) -> None:
        terminal = canonical_spec["terminal_composition_contract"]
        assert isinstance(terminal, dict)
        custody = terminal["terminal_custody_authority_projection"]
        assert isinstance(custody, dict)
        launcher = custody["terminal_client_launcher_projection"]
        assert isinstance(launcher, dict)
        launcher["fallback_allowed"] = True
        launcher_unsigned = {
            key: item for key, item in launcher.items() if key != "projection_root_sha256"
        }
        launcher["projection_root_sha256"] = _authority_root(launcher_unsigned)
        custody["terminal_client_launcher_projection_root_sha256"] = launcher[
            "projection_root_sha256"
        ]
        _rehash_terminal_custody_projection(custody)
        terminal_unsigned = {
            key: item for key, item in terminal.items() if key != "contract_sha256"
        }
        terminal["contract_sha256"] = _authority_root(terminal_unsigned)

    _rewrite_synthetic_run_spec(
        job_directory / "run_spec.json",
        mutate_spec=mutate_spec,
    )
    completed = _invoke_capsule(
        capsule,
        cwd=working,
        program=str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        tail=tail,
    )

    assert completed.returncode != 0
    assert completed.stdout == b""
    assert b"terminal custody projection differs from sealed E" in completed.stderr
    assert not marker.exists()
    assert not (job_directory / "e_intent_consumed.json").exists()


def test_self_consistent_hidden_run_spec_selector_stops_before_claim_and_import(
    tmp_path: Path,
) -> None:
    files = _synthetic_tree(tmp_path / "source")
    marker = tmp_path / "PROJECT_IMPORTED"
    files["entry"].write_bytes(
        (
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
        ).encode()
        + files["entry"].read_bytes()
    )
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    working = tmp_path / "working"
    working.mkdir()
    tail = _prepare_exact_q_e(
        capsule,
        cwd=working,
        mode="run-confirmatory",
    )
    job_directory = _synthetic_prepared_job_directory(tmp_path)

    def mutate_spec(canonical_spec: dict[str, object]) -> None:
        canonical_spec["untrusted_selector"] = {"json_equals": {"metrics": {"auprc": 0.99}}}

    _rewrite_synthetic_run_spec(
        job_directory / "run_spec.json",
        mutate_spec=mutate_spec,
    )
    completed = _invoke_capsule(
        capsule,
        cwd=working,
        program=str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        tail=tail,
    )

    assert completed.returncode != 0
    assert completed.stdout == b""
    assert b"supervisor canonical spec violates its exact field set" in completed.stderr
    assert not marker.exists()
    assert not (job_directory / "e_intent_consumed.json").exists()


def test_two_independent_builds_are_byte_identical(tmp_path: Path) -> None:
    first_root = tmp_path / "source-a"
    second_root = tmp_path / "source-b"
    first = _synthetic_tree(first_root)
    shutil.copytree(first_root, second_root)
    second = {
        key: (
            second_root / value.relative_to(first_root)
            if key != "package"
            else second_root / "src" / "histo_audit"
        )
        for key, value in first.items()
    }
    first_members = _discover(first)
    expected = source_inventory(first_members)
    second_members = _discover(second)
    assert source_inventory(second_members) == expected

    output = tmp_path / "output"
    output.mkdir()
    first_result = build_project_capsule(
        package_root=first["package"],
        bootstrap_path=first["bootstrap"],
        policy_path=first["policy"],
        entry_contract_path=first["contract"],
        expected_inventory=expected,
        output_path=output / "first.pyz",
    )
    second_result = build_project_capsule(
        package_root=second["package"],
        bootstrap_path=second["bootstrap"],
        policy_path=second["policy"],
        entry_contract_path=second["contract"],
        expected_inventory=expected,
        output_path=output / "second.pyz",
    )

    first_bytes = first_result.output_path.read_bytes()
    second_bytes = second_result.output_path.read_bytes()
    assert first_bytes == second_bytes
    assert first_result.sha256 == second_result.sha256 == hashlib.sha256(first_bytes).hexdigest()
    assert first_result.internal_manifest_sha256 == second_result.internal_manifest_sha256
    assert first_result.records_root_sha256 == expected.records_root_sha256
    assert first_result.entry_count == len(expected.entries)
    assert stat.S_IMODE(os.stat(first_result.output_path).st_mode) & stat.S_IWUSR == 0

    with zipfile.ZipFile(first_result.output_path) as archive:
        names = archive.namelist()
        assert names[:-1] == sorted(names[:-1])
        assert names[-1] == MANIFEST_NAME
        manifest_bytes = archive.read(MANIFEST_NAME)
        assert manifest_bytes.endswith(b"\n")
        assert not manifest_bytes.endswith(b"\n\n")
        manifest = json.loads(manifest_bytes)
        assert set(manifest) == {
            "archive_policy",
            "entries",
            "entry_count",
            "payload_size_bytes",
            "policy",
            "records_root_sha256",
            "schema_version",
        }
        assert MANIFEST_NAME not in {entry["relative_path"] for entry in manifest["entries"]}
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.create_system == 3
            assert info.external_attr == (stat.S_IFREG | 0o444) << 16
            assert info.extra == b""
            assert info.comment == b""


def test_every_frozen_member_is_required_before_output(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baseline"
    baseline = _synthetic_tree(baseline_root)
    expected = source_inventory(_discover(baseline))
    member_to_source = {
        "histo_audit/__init__.py": Path(baseline["init"]),
        "histo_audit/experiment/confirmatory_completion.py": Path(baseline["completion"]),
        "histo_audit/experiment/__init__.py": Path(baseline["experiment_init"]),
        "histo_audit/experiment/original_confirmatory_runner_core.py": Path(baseline["runner"]),
        "histo_audit/models/cnn.py": Path(baseline["dependency"]),
        "histo_audit/models/__init__.py": Path(baseline["models_init"]),
        "histo_audit/workflows/original_confirmatory_capsule_authority.py": Path(
            baseline["authority"]
        ),
        "histo_audit/workflows/__init__.py": Path(baseline["workflows_init"]),
        "histo_audit/workflows/original_confirmatory_capsule_entry.py": Path(baseline["entry"]),
        "histo_audit/workflows/original_confirmatory_capsule_terminal.py": Path(
            baseline["terminal"]
        ),
        "__main__.py": Path(baseline["bootstrap"]),
        "aanca_capsule/capsule_policy.json": Path(baseline["policy"]),
        "aanca_capsule/entry_contract.json": Path(baseline["contract"]),
    }
    assert set(member_to_source) == {entry["relative_path"] for entry in expected.entries}

    for index, relative_path in enumerate(sorted(member_to_source)):
        case_root = tmp_path / f"missing-{index}"
        shutil.copytree(baseline_root, case_root)
        missing = case_root / member_to_source[relative_path].relative_to(baseline_root)
        missing.unlink()
        files = {
            key: (
                case_root / value.relative_to(baseline_root)
                if key != "package"
                else case_root / "src" / "histo_audit"
            )
            for key, value in baseline.items()
        }
        output = tmp_path / f"must-not-exist-{index}.pyz"
        with pytest.raises((CapsuleBuildError, FileNotFoundError)):
            build_project_capsule(
                package_root=files["package"],
                bootstrap_path=files["bootstrap"],
                policy_path=files["policy"],
                entry_contract_path=files["contract"],
                expected_inventory=expected,
                output_path=output,
            )
        assert not output.exists()


@pytest.mark.parametrize("mutation", ["added_py", "changed_payload"])
def test_frozen_source_inventory_rejects_added_or_changed_python(
    tmp_path: Path,
    mutation: str,
) -> None:
    baseline_root = tmp_path / "inventory-baseline"
    baseline = _synthetic_tree(baseline_root)
    expected = source_inventory(_discover(baseline))
    case_root = tmp_path / f"inventory-{mutation}"
    shutil.copytree(baseline_root, case_root)
    files = {
        key: (
            case_root / value.relative_to(baseline_root)
            if key != "package"
            else case_root / "src" / "histo_audit"
        )
        for key, value in baseline.items()
    }
    if mutation == "added_py":
        _write(files["package"] / "unexpected.py", b"VALUE = 'unexpected'\n")
    else:
        files["dependency"].write_bytes(b"VALUE = 'changed-after-freeze'\n")

    output = tmp_path / f"inventory-{mutation}-must-not-exist.pyz"
    with pytest.raises(
        CapsuleBuildError,
        match="current source inventory differs from the frozen inventory",
    ):
        build_project_capsule(
            package_root=files["package"],
            bootstrap_path=files["bootstrap"],
            policy_path=files["policy"],
            entry_contract_path=files["contract"],
            expected_inventory=expected,
            output_path=output,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "unsafe",
    [
        "../escape.py",
        "/absolute.py",
        "c:/drive.py",
        "histo_audit\\module.py",
        "histo_audit//module.py",
        "histo_audit/./module.py",
        "histo_audit/module.py.",
        "histo_audit/con.py",
        "HISTO_AUDIT/module.py",
        MANIFEST_NAME,
    ],
)
def test_unsafe_or_reserved_member_path_is_rejected(tmp_path: Path, unsafe: str) -> None:
    files = _synthetic_tree(tmp_path / "source")
    members = list(_discover(files))
    original = members[0]
    members[0] = type(original)(
        source_path=original.source_path,
        relative_path=unsafe,
        role=original.role,
        size_bytes=original.size_bytes,
        sha256=original.sha256,
        payload=original.payload,
        identity=original.identity,
    )
    with pytest.raises(CapsuleBuildError):
        source_inventory(members)


def test_isolated_capsule_imports_project_modules_only_from_pyz(tmp_path: Path) -> None:
    files = _synthetic_tree(tmp_path / "source")
    members = _discover(files)
    expected = source_inventory(members)
    output = tmp_path / "capsule.pyz"
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=output,
    )
    trap = tmp_path / "trap"
    _write(
        trap / "histo_audit" / "__init__.py",
        b"raise RuntimeError('mutable project trap imported')\n",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    completed = _run_capsule(capsule, cwd=trap)
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    assert completed.stderr == b""
    payload = json.loads(completed.stdout)
    assert payload["argv"][0] == "verify-terminal"
    assert payload["argv"][1:3] == [
        "--e-intent",
        str(_synthetic_staged_e_intent_path(_synthetic_prepared_job_directory(trap.parent))),
    ]
    assert ".pyz" in payload["entry_origin"]
    assert ".pyz" in payload["cnn_origin"]
    assert str(trap) not in payload["entry_origin"]
    assert str(trap) not in payload["cnn_origin"]


def test_internally_valid_unselected_capsule_cannot_import_or_dispatch(
    tmp_path: Path,
) -> None:
    selected_files = _synthetic_tree(tmp_path / "selected-source")
    selected_expected = source_inventory(_discover(selected_files))
    selected_result = build_project_capsule(
        package_root=selected_files["package"],
        bootstrap_path=selected_files["bootstrap"],
        policy_path=selected_files["policy"],
        entry_contract_path=selected_files["contract"],
        expected_inventory=selected_expected,
        output_path=tmp_path / "selected.pyz",
    )
    selected = _stage_content_addressed_capsule(
        selected_result,
        tmp_path / "execution_capsules",
    )
    marker = tmp_path / "UNSELECTED_PROJECT_IMPORTED"
    unselected_files = _synthetic_tree(tmp_path / "unselected-source")
    unselected_files["entry"].write_bytes(
        (
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
            "def _dispatch_original_confirmatory_capsule(argv):\n"
            "    return 73\n"
        ).encode()
    )
    unselected_expected = source_inventory(_discover(unselected_files))
    unselected_result = build_project_capsule(
        package_root=unselected_files["package"],
        bootstrap_path=unselected_files["bootstrap"],
        policy_path=unselected_files["policy"],
        entry_contract_path=unselected_files["contract"],
        expected_inventory=unselected_expected,
        output_path=tmp_path / "unselected.pyz",
    )
    unselected = _stage_content_addressed_capsule(
        unselected_result,
        tmp_path / "execution_capsules",
    )
    working = tmp_path / "working"
    working.mkdir()
    tail = _prepare_exact_q_e(
        selected,
        cwd=working,
        mode="verify-terminal",
    )
    completed = _invoke_capsule(
        unselected,
        cwd=working,
        program=str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        tail=tail,
    )
    assert completed.returncode != 0
    assert completed.stdout == b""
    assert not marker.exists()
    assert b"execution capsule contract does not select this exact archive" in completed.stderr


def test_project_venv_identity_swap_stops_before_project_import(
    tmp_path: Path,
) -> None:
    files = _synthetic_tree(tmp_path / "source")
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    working = tmp_path / "working"
    working.mkdir()
    tail = _prepare_exact_q_e(
        capsule,
        cwd=working,
        mode="verify-terminal",
    )
    original_venv = tmp_path / ".venv"
    displaced_venv = tmp_path / ".venv-displaced"
    original_venv.rename(displaced_venv)
    shutil.copytree(displaced_venv, original_venv)
    completed = _invoke_capsule(
        capsule,
        cwd=working,
        program=str(original_venv / "Scripts" / "python.exe"),
        tail=tail,
    )
    assert completed.returncode != 0
    assert completed.stdout == b""
    assert (
        b"interpreter retained-file lease differs from the retained interpreter" in completed.stderr
        or b"interpreter ancestor lease record" in completed.stderr
    )


def test_self_consistent_substitute_runtime_python_stops_before_project_import(
    tmp_path: Path,
) -> None:
    files = _synthetic_tree(tmp_path / "source")
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    working = tmp_path / "working"
    working.mkdir()
    substitute = tmp_path / "substitute-runtime" / "python.exe"
    substitute.parent.mkdir()
    shutil.copy2(Path(str(getattr(sys, "_base_executable", sys.executable))), substitute)
    tail = _prepare_exact_q_e(
        capsule,
        cwd=working,
        mode="verify-terminal",
        runtime_python_override=substitute,
    )
    completed = _invoke_capsule(
        capsule,
        cwd=working,
        program=str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        tail=tail,
    )
    assert completed.returncode != 0
    assert completed.stdout == b""
    assert b"runtime interpreter differs across Q and native process evidence" in completed.stderr


def test_invalid_e_anchor_does_not_create_consumption_claim(
    tmp_path: Path,
) -> None:
    files = _synthetic_tree(tmp_path / "source")
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    working = tmp_path / "working"
    working.mkdir()
    tail = _prepare_exact_q_e(
        capsule,
        cwd=working,
        mode="run-confirmatory",
    )
    e_hash_index = tail.index("--e-intent-sha256") + 1
    tail[e_hash_index] = "0" * 64
    program = str(tmp_path / ".venv" / "Scripts" / "python.exe")
    first = _invoke_capsule(
        capsule,
        cwd=working,
        program=program,
        tail=tail,
    )
    claim = _synthetic_prepared_job_directory(tmp_path) / "e_intent_consumed.json"
    assert first.returncode != 0
    assert first.stdout == b""
    assert b"E intent file SHA-256 differs from exact argv" in first.stderr
    assert not claim.exists()
    second = _invoke_capsule(
        capsule,
        cwd=working,
        program=program,
        tail=tail,
    )
    assert second.returncode != 0
    assert second.stdout == b""
    assert b"E intent file SHA-256 differs from exact argv" in second.stderr
    assert not claim.exists()


def test_valid_prevalidated_run_claims_exactly_once(tmp_path: Path) -> None:
    files = _synthetic_tree(tmp_path / "source")
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    working = tmp_path / "working"
    working.mkdir()
    tail = _prepare_exact_q_e(
        capsule,
        cwd=working,
        mode="run-confirmatory",
    )
    program = str(tmp_path / ".venv" / "Scripts" / "python.exe")
    job_directory = _synthetic_prepared_job_directory(tmp_path)
    future_terminal_client_intent = job_directory / "terminal_client_launch_intent.json"
    assert not future_terminal_client_intent.exists()
    first = _invoke_capsule(
        capsule,
        cwd=working,
        program=program,
        tail=tail,
    )
    claim = job_directory / "e_intent_consumed.json"
    assert first.returncode == 0, first.stderr.decode(errors="replace")
    assert not future_terminal_client_intent.exists()
    assert claim.stat().st_size > 0
    assert bootstrap._readonly_file(os.lstat(claim))
    before = os.lstat(claim)
    second = _invoke_capsule(
        capsule,
        cwd=working,
        program=program,
        tail=tail,
    )
    after = os.lstat(claim)
    assert second.returncode != 0
    assert second.stdout == b""
    assert b"could not CREATE_NEW its permanent E claim" in second.stderr
    assert not future_terminal_client_intent.exists()
    assert bootstrap._stable_file_identity(before) == bootstrap._stable_file_identity(after)


def test_invalid_capsule_does_not_create_consumption_claim(tmp_path: Path) -> None:
    files = _synthetic_tree(tmp_path / "source")
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    working = tmp_path / "working"
    working.mkdir()
    tail = _prepare_exact_q_e(
        capsule,
        cwd=working,
        mode="run-confirmatory",
    )
    os.chmod(capsule, stat.S_IWRITE | stat.S_IREAD)
    capsule.write_bytes(capsule.read_bytes() + b"x")
    os.chmod(capsule, stat.S_IREAD)
    completed = _invoke_capsule(
        capsule,
        cwd=working,
        program=str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        tail=tail,
    )
    claim = _synthetic_prepared_job_directory(tmp_path) / "e_intent_consumed.json"
    assert completed.returncode != 0
    assert not claim.exists()


def test_invalid_q_does_not_create_consumption_claim(tmp_path: Path) -> None:
    files = _synthetic_tree(tmp_path / "source")
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    working = tmp_path / "working"
    working.mkdir()
    tail = _prepare_exact_q_e(
        capsule,
        cwd=working,
        mode="run-confirmatory",
    )
    q_path = (
        tmp_path / "artifacts" / "resource_control" / "original_confirmatory_q_replacement_v2.json"
    )
    os.chmod(q_path, stat.S_IWRITE | stat.S_IREAD)
    q_path.write_bytes(b"{}\n")
    os.chmod(q_path, stat.S_IREAD)
    completed = _invoke_capsule(
        capsule,
        cwd=working,
        program=str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        tail=tail,
    )
    claim = _synthetic_prepared_job_directory(tmp_path) / "e_intent_consumed.json"
    assert completed.returncode != 0
    assert not claim.exists()


def test_invalid_command_does_not_create_consumption_claim(tmp_path: Path) -> None:
    files = _synthetic_tree(tmp_path / "source")
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    working = tmp_path / "working"
    working.mkdir()
    tail = _prepare_exact_q_e(
        capsule,
        cwd=working,
        mode="run-confirmatory",
    )
    attempt_index = tail.index("--attempt-id") + 1
    tail[attempt_index] = "different-attempt"
    completed = _invoke_capsule(
        capsule,
        cwd=working,
        program=str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        tail=tail,
    )
    claim = _synthetic_prepared_job_directory(tmp_path) / "e_intent_consumed.json"
    assert completed.returncode != 0
    assert not claim.exists()


def test_run_dispatcher_must_take_e_claim_before_return(tmp_path: Path) -> None:
    files = _synthetic_tree(tmp_path / "source")
    files["entry"].write_bytes(
        b"import __main__\n"
        b"def _dispatch_original_confirmatory_capsule(argv):\n"
        b"    __main__._arm_original_confirmatory_e_claim_after_full_prevalidation()\n"
        b"    del argv\n"
        b"    return 0\n"
    )
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    working = tmp_path / "working"
    working.mkdir()
    tail = _prepare_exact_q_e(
        capsule,
        cwd=working,
        mode="run-confirmatory",
    )
    completed = _invoke_capsule(
        capsule,
        cwd=working,
        program=str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        tail=tail,
    )
    claim = _synthetic_prepared_job_directory(tmp_path) / "e_intent_consumed.json"
    assert completed.returncode != 0
    assert completed.stdout == b""
    assert b"dispatcher did not take the one-use E claim handle" in completed.stderr
    assert claim.stat().st_size == 0
    assert bootstrap._readonly_file(os.lstat(claim))


def test_verifier_missing_consumed_e_claim_stops_before_capsule_read(
    tmp_path: Path,
) -> None:
    files = _synthetic_tree(tmp_path / "source")
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    working = tmp_path / "working"
    working.mkdir()
    tail = _prepare_exact_q_e(
        capsule,
        cwd=working,
        mode="verify-terminal",
    )
    claim = _synthetic_prepared_job_directory(tmp_path) / "e_intent_consumed.json"
    os.chmod(claim, stat.S_IWRITE)
    claim.unlink()
    completed = _invoke_capsule(
        capsule,
        cwd=working,
        program=str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        tail=tail,
    )
    assert completed.returncode != 0
    assert completed.stdout == b""
    assert not claim.exists()


def test_consumed_e_claim_read_handle_is_one_use(tmp_path: Path) -> None:
    files = _synthetic_tree(tmp_path / "source")
    files["entry"].write_bytes(
        b"import __main__\n"
        b"import os\n"
        b"def _dispatch_original_confirmatory_capsule(argv):\n"
        b"    __main__._arm_original_confirmatory_e_claim_after_full_prevalidation()\n"
        b"    descriptor, _path, _sha256, _size = "
        b"__main__._take_original_confirmatory_e_claim_read_handle()\n"
        b"    os.close(descriptor)\n"
        b"    try:\n"
        b"        __main__._take_original_confirmatory_e_claim_read_handle()\n"
        b"    except Exception:\n"
        b"        return 0\n"
        b"    return 79\n"
    )
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    working = tmp_path / "working"
    working.mkdir()
    tail = _prepare_exact_q_e(
        capsule,
        cwd=working,
        mode="verify-terminal",
    )
    completed = _invoke_capsule(
        capsule,
        cwd=working,
        program=str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        tail=tail,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8",
        errors="replace",
    )
    assert completed.stdout == b""
    assert completed.stderr == b""


def test_wrong_content_address_path_stops_before_dispatch(tmp_path: Path) -> None:
    files = _synthetic_tree(tmp_path / "source")
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "not-content-addressed.pyz",
    )
    completed = _run_capsule(result.output_path, cwd=tmp_path)
    assert completed.returncode != 0
    assert completed.stdout == b""
    assert b"capsule filename is not exact" in completed.stderr


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("contract_status", "incomplete"),
        ("dispatcher", "histo_audit.mutable:dispatcher"),
        ("allowed_modes", ["run-confirmatory", "verify-terminal"]),
    ],
)
def test_invalid_entry_contract_stops_before_dispatch(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    files = _synthetic_tree(tmp_path / "source")
    import_marker = tmp_path / f"invalid-entry-{field}-imported"
    files["entry"].write_bytes(
        (
            "from pathlib import Path\n"
            f"Path({str(import_marker)!r}).write_text('imported', encoding='ascii')\n"
        ).encode("ascii")
        + files["entry"].read_bytes()
    )
    contract = json.loads(files["contract"].read_bytes())
    contract[field] = replacement
    files["contract"].write_bytes(_canonical_json_line(contract))
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "nonready-source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    completed = _run_capsule(capsule, cwd=tmp_path)
    assert completed.returncode != 0
    assert completed.stdout == b""
    assert b"capsule entry contract is not exact and ready" in completed.stderr
    assert not import_marker.exists()


def test_historical_incomplete_entry_contract_stops_before_import_or_claim(
    tmp_path: Path,
) -> None:
    historical_incomplete_contract = (
        b'{"allowed_modes":["run-confirmatory","verify-preterminal","verify-terminal"],'
        b'"contract_status":"incomplete_fail_closed_pending_terminal_composed_receipt_v1",'
        b'"policy":"original_confirmatory_execution_capsule_entry_contract_v1",'
        b'"schema_version":1}\n'
    )
    assert len(historical_incomplete_contract) == 246
    assert (
        hashlib.sha256(historical_incomplete_contract).hexdigest()
        == "8003dd488e708712972561d21f46220f2821535b0de33e28b8d7583946f4ba64"
    )
    files = _synthetic_tree(tmp_path / "source")
    import_marker = tmp_path / "historical-incomplete-entry-imported"
    files["entry"].write_bytes(
        (
            "from pathlib import Path\n"
            f"Path({str(import_marker)!r}).write_text('imported', encoding='ascii')\n"
        ).encode("ascii")
        + files["entry"].read_bytes()
    )
    files["contract"].write_bytes(historical_incomplete_contract)
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "checked-in-incomplete-source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    working = tmp_path / "working"
    working.mkdir()
    tail = _prepare_exact_q_e(
        capsule,
        cwd=working,
        mode="run-confirmatory",
    )
    claim = _synthetic_prepared_job_directory(tmp_path) / "e_intent_consumed.json"
    assert not claim.exists()
    completed = _invoke_capsule(
        capsule,
        cwd=working,
        program=str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        tail=tail,
    )

    assert completed.returncode != 0
    assert completed.stdout == b""
    assert b"capsule entry contract is not exact and ready" in completed.stderr
    assert not import_marker.exists()
    assert not claim.exists()


def test_missing_capsule_dependency_cannot_fall_back_to_mutable_tree(
    tmp_path: Path,
) -> None:
    files = _synthetic_tree(tmp_path / "source")
    trap = tmp_path / "mutable-tree"
    _write(trap / "histo_audit" / "__init__.py", b"")
    _write(trap / "histo_audit" / "models" / "__init__.py", b"")
    _write(
        trap / "histo_audit" / "models" / "cnn.py",
        b"raise RuntimeError('MUTABLE_FALLBACK_IMPORTED')\n",
    )
    files["entry"].write_bytes(
        (
            "import histo_audit.models\n"
            f"histo_audit.models.__path__.append({str(trap / 'histo_audit' / 'models')!r})\n"
            "from histo_audit.models import cnn\n"
            "def _dispatch_original_confirmatory_capsule(argv):\n"
            "    return int(cnn.VALUE == 'cnn') - 1\n"
        ).encode()
    )
    members = tuple(
        member for member in _discover(files) if member.relative_path != "histo_audit/models/cnn.py"
    )
    expected = source_inventory(members)
    result = build_capsule(
        members=members,
        expected_inventory=expected,
        output_path=tmp_path / "missing-dependency-source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    completed = _run_capsule(capsule, cwd=trap)
    assert completed.returncode != 0
    assert completed.stdout == b""
    assert b"project module origin is outside the capsule: histo_audit.models" in completed.stderr
    assert b"MUTABLE_FALLBACK_IMPORTED" not in completed.stderr


@pytest.mark.skipif(os.name != "nt", reason="Windows retained-handle semantics")
def test_dispatch_cannot_open_held_capsule_for_write(tmp_path: Path) -> None:
    files = _synthetic_tree(tmp_path / "source")
    files["entry"].write_bytes(
        b"import __main__\n"
        b"import os\n"
        b"import sys\n"
        b"def _dispatch_original_confirmatory_capsule(argv):\n"
        b"    __main__._arm_original_confirmatory_e_claim_after_full_prevalidation()\n"
        b"    descriptor, _path, _sha256, _size = "
        b"__main__._take_original_confirmatory_e_claim_read_handle()\n"
        b"    os.close(descriptor)\n"
        b"    try:\n"
        b"        open(sys.argv[0], 'r+b').close()\n"
        b"    except PermissionError:\n"
        b"        return 0\n"
        b"    return 91\n"
    )
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "held-source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    completed = _run_capsule(capsule, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    assert completed.stdout == b""
    assert completed.stderr == b""


@pytest.mark.skipif(os.name != "nt", reason="Windows retained-handle semantics")
def test_dispatch_cannot_open_held_run_spec_for_write(tmp_path: Path) -> None:
    files = _synthetic_tree(tmp_path / "source")
    files["entry"].write_bytes(
        b"import __main__\n"
        b"import json\n"
        b"import os\n"
        b"from pathlib import Path\n"
        b"def _dispatch_original_confirmatory_capsule(argv):\n"
        b"    __main__._arm_original_confirmatory_e_claim_after_full_prevalidation()\n"
        b"    descriptor, _path, _sha256, _size = "
        b"__main__._take_original_confirmatory_e_claim_read_handle()\n"
        b"    os.close(descriptor)\n"
        b"    job_dir = Path(argv[argv.index('--supervisor-job-dir') + 1])\n"
        b"    try:\n"
        b"        open(job_dir / 'run_spec.json', 'r+b').close()\n"
        b"    except PermissionError:\n"
        b"        return 0\n"
        b"    return 91\n"
    )
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "held-run-spec-source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    completed = _run_capsule(capsule, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8",
        errors="replace",
    )
    assert completed.stdout == b""
    assert completed.stderr == b""


@pytest.mark.skipif(os.name != "nt", reason="Windows retained-handle semantics")
def test_dispatch_cannot_replace_held_terminal_client_launcher(
    tmp_path: Path,
) -> None:
    files = _synthetic_tree(tmp_path / "source")
    files["entry"].write_bytes(
        b"import __main__\n"
        b"import json\n"
        b"import os\n"
        b"from pathlib import Path\n"
        b"def _dispatch_original_confirmatory_capsule(argv):\n"
        b"    __main__._arm_original_confirmatory_e_claim_after_full_prevalidation()\n"
        b"    descriptor, _path, _sha256, _size = "
        b"__main__._take_original_confirmatory_e_claim_read_handle()\n"
        b"    os.close(descriptor)\n"
        b"    job_dir = Path(argv[argv.index('--supervisor-job-dir') + 1])\n"
        b"    q_path = job_dir.parents[2] / 'artifacts' / 'resource_control' / "
        b"'original_confirmatory_q_replacement_v2.json'\n"
        b"    q_release = json.loads(q_path.read_text(encoding='utf-8'))\n"
        b"    launcher = Path(q_release['supervisor_release']"
        b"['terminal_client_launcher_release']['source_path'])\n"
        b"    replacement = job_dir / 'terminal_client_launcher.replacement.py'\n"
        b"    replacement.write_bytes(b'replacement')\n"
        b"    try:\n"
        b"        os.replace(replacement, launcher)\n"
        b"    except PermissionError:\n"
        b"        replacement.unlink()\n"
        b"        return 0\n"
        b"    return 91\n"
    )
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "held-terminal-client-source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    completed = _run_capsule(capsule, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8",
        errors="replace",
    )
    assert completed.stdout == b""
    assert completed.stderr == b""


@pytest.mark.skipif(os.name != "nt", reason="Windows retained-handle semantics")
@pytest.mark.parametrize(
    ("target_name", "target_expression"),
    [
        (
            "q",
            "job_dir.parents[2] / 'artifacts' / 'resource_control' / "
            "'original_confirmatory_q_replacement_v2.json'",
        ),
        (
            "e",
            "job_dir.parent.parent / 'control_staging' / job_dir.name / 'e_intent.json'",
        ),
        ("spec", "job_dir / 'run_spec.json'"),
        (
            "launcher",
            "Path(q_release['supervisor_release']"
            "['terminal_client_launcher_release']['source_path'])",
        ),
    ],
)
def test_dispatch_cannot_write_or_replace_any_preimport_authority_anchor(
    tmp_path: Path,
    target_name: str,
    target_expression: str,
) -> None:
    files = _synthetic_tree(tmp_path / "source")
    files["entry"].write_bytes(
        (
            "import __main__\n"
            "import json\n"
            "import os\n"
            "from pathlib import Path\n"
            "def _dispatch_original_confirmatory_capsule(argv):\n"
            "    __main__._arm_original_confirmatory_e_claim_after_full_prevalidation()\n"
            "    descriptor, _path, _sha256, _size = "
            "__main__._take_original_confirmatory_e_claim_read_handle()\n"
            "    os.close(descriptor)\n"
            "    job_dir = Path(argv[argv.index('--supervisor-job-dir') + 1])\n"
            "    q_path = job_dir.parents[2] / 'artifacts' / 'resource_control' / "
            "'original_confirmatory_q_replacement_v2.json'\n"
            "    q_release = json.loads(q_path.read_text(encoding='utf-8'))\n"
            f"    target = {target_expression}\n"
            "    try:\n"
            "        open(target, 'r+b').close()\n"
            "    except PermissionError:\n"
            "        pass\n"
            "    else:\n"
            "        return 91\n"
            "    replacement = target.with_name(target.name + '.replacement')\n"
            "    replacement.write_bytes(b'replacement')\n"
            "    try:\n"
            "        os.replace(replacement, target)\n"
            "    except PermissionError:\n"
            "        replacement.unlink()\n"
            "        return 0\n"
            "    return 92\n"
        ).encode("ascii")
    )
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / f"held-{target_name}-source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    working = tmp_path / "working"
    working.mkdir()
    tail = _prepare_exact_q_e(
        capsule,
        cwd=working,
        mode="verify-terminal",
    )
    job_directory = _synthetic_prepared_job_directory(tmp_path)
    q_path = (
        tmp_path / "artifacts" / "resource_control" / "original_confirmatory_q_replacement_v2.json"
    )
    q_release = json.loads(q_path.read_text(encoding="utf-8"))
    targets = {
        "q": q_path,
        "e": _synthetic_staged_e_intent_path(job_directory),
        "spec": job_directory / "run_spec.json",
        "launcher": Path(
            q_release["supervisor_release"]["terminal_client_launcher_release"]["source_path"]
        ),
    }
    target = targets[target_name]
    before = target.read_bytes()
    completed = _invoke_capsule(
        capsule,
        cwd=working,
        program=str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        tail=tail,
    )

    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8",
        errors="replace",
    )
    assert completed.stdout == b""
    assert completed.stderr == b""
    assert target.read_bytes() == before
    assert not target.with_name(target.name + ".replacement").exists()


@pytest.mark.parametrize("outcome", ["success", "raise", "non_int"])
def test_post_dispatch_finally_rechecks_every_authority_anchor(
    tmp_path: Path,
    outcome: str,
) -> None:
    marker = tmp_path / f"post-dispatch-{outcome}-labels.txt"
    outcome_statement = {
        "success": "    return 0\n",
        "raise": "    raise RuntimeError('synthetic dispatcher failure')\n",
        "non_int": "    return 'not-an-integer'\n",
    }[outcome]
    files = _synthetic_tree(tmp_path / "source")
    files["entry"].write_bytes(
        (
            "import __main__\n"
            "import os\n"
            "from pathlib import Path\n"
            f"_MARKER = Path({str(marker)!r})\n"
            "_ORIGINAL_RECHECK = __main__._require_held_file_unchanged\n"
            "def _record_recheck(held, *, label):\n"
            "    _ORIGINAL_RECHECK(held, label=label)\n"
            "    with _MARKER.open('a', encoding='ascii', newline='\\n') as stream:\n"
            "        stream.write(label + '\\n')\n"
            "__main__._require_held_file_unchanged = _record_recheck\n"
            "def _dispatch_original_confirmatory_capsule(argv):\n"
            "    __main__._arm_original_confirmatory_e_claim_after_full_prevalidation()\n"
            "    descriptor, _path, _sha256, _size = "
            "__main__._take_original_confirmatory_e_claim_read_handle()\n"
            "    os.close(descriptor)\n" + outcome_statement
        ).encode("ascii")
    )
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / f"post-dispatch-{outcome}-source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    completed = _run_capsule(capsule, cwd=tmp_path)

    assert marker.read_text(encoding="ascii").splitlines() == [
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
    ]
    if outcome == "success":
        assert completed.returncode == 0
        assert completed.stderr == b""
    elif outcome == "raise":
        assert completed.returncode != 0
        assert b"synthetic dispatcher failure" in completed.stderr
    else:
        assert completed.returncode != 0
        assert b"sealed capsule dispatcher returned a non-integer exit code" in completed.stderr
    assert completed.stdout == b""


def test_bootstrap_rejects_local_header_only_tamper(tmp_path: Path) -> None:
    files = _synthetic_tree(tmp_path / "source")
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "untampered-source.pyz",
    )
    payload = bytearray(result.output_path.read_bytes())
    assert payload[:4] == b"PK\x03\x04"
    payload[10] = 1
    capsule = _stage_raw_content_addressed_capsule(
        bytes(payload),
        tmp_path / "execution_capsules",
    )
    completed = _run_capsule(capsule, cwd=tmp_path)
    assert completed.returncode != 0
    assert completed.stdout == b""
    assert b"capsule local ZIP header is non-canonical" in completed.stderr


@pytest.mark.parametrize("module_name", ["histo_audit", "histo_audit.foreign"])
def test_import_sanitizer_stops_when_project_module_was_preimported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    for loaded_name in tuple(sys.modules):
        if loaded_name == "histo_audit" or loaded_name.startswith("histo_audit."):
            monkeypatch.delitem(sys.modules, loaded_name)
    monkeypatch.setitem(sys.modules, module_name, object())
    path_before = list(sys.path)
    meta_path_before = list(sys.meta_path)
    path_hooks_before = list(sys.path_hooks)
    importer_cache_before = dict(sys.path_importer_cache)

    with pytest.raises(
        bootstrap.CapsuleBootstrapError,
        match="project module was imported before capsule verification",
    ):
        bootstrap._sanitize_import_state(
            tmp_path / "capsule.pyz",
            modules={"histo_audit": ("histo_audit/__init__.py", True)},
        )

    assert sys.path == path_before
    assert sys.meta_path == meta_path_before
    assert sys.path_hooks == path_hooks_before
    assert sys.path_importer_cache == importer_cache_before


def test_import_sanitizer_removes_external_finders_hooks_and_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "capsule.pyz"
    working = tmp_path / "working"
    working.mkdir()
    external = object()

    def external_path_hook(path: str) -> object:
        raise AssertionError(f"external path hook executed: {path}")

    for module_name in tuple(sys.modules):
        if module_name == "histo_audit" or module_name.startswith("histo_audit."):
            monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.chdir(working)
    monkeypatch.setattr(sys, "path", [str(archive), str(tmp_path / "site")])
    monkeypatch.setattr(sys, "meta_path", [external])
    monkeypatch.setattr(sys, "path_hooks", [external_path_hook])
    monkeypatch.setattr(sys, "path_importer_cache", {"external": external})
    finder = bootstrap._sanitize_import_state(
        archive,
        modules={"histo_audit": ("histo_audit/__init__.py", True)},
    )
    assert finder in sys.meta_path
    assert external not in sys.meta_path
    assert external_path_hook not in sys.path_hooks
    assert sys.path_importer_cache == {}


@pytest.mark.parametrize("behavior", ["raise", "non_integer"])
def test_dispatch_finally_rechecks_project_package_path(
    tmp_path: Path,
    behavior: str,
) -> None:
    files = _synthetic_tree(tmp_path / "source")
    outcome = (
        "    raise RuntimeError('dispatcher failed')\n"
        if behavior == "raise"
        else "    return 'not-an-integer'\n"
    )
    files["entry"].write_bytes(
        (
            "import __main__\n"
            "import os\n"
            "import histo_audit.models\n"
            "def _dispatch_original_confirmatory_capsule(argv):\n"
            "    __main__._arm_original_confirmatory_e_claim_after_full_prevalidation()\n"
            "    descriptor, _path, _sha256, _size = "
            "__main__._take_original_confirmatory_e_claim_read_handle()\n"
            "    os.close(descriptor)\n"
            "    histo_audit.models.__path__.append('C:\\\\mutable-fallback')\n"
            f"{outcome}"
        ).encode()
    )
    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / f"{behavior}-source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    completed = _run_capsule(capsule, cwd=tmp_path)
    assert completed.returncode != 0
    assert completed.stdout == b""
    assert b"project module origin is outside the capsule: histo_audit.models" in completed.stderr


def _install_control_only_production_entry_path(files: dict[str, Path]) -> None:
    project_root = Path(__file__).resolve().parents[1]
    entry_candidates = [
        path
        for path in (
            project_root
            / "src"
            / "histo_audit"
            / "workflows"
            / "original_confirmatory_capsule_entry.py",
            project_root / "project_capsule_entry_reference.py",
        )
        if path.is_file()
    ]
    assert len(entry_candidates) == 1
    files["entry"].write_bytes(entry_candidates[0].read_bytes())
    files["terminal"].write_bytes(
        b"import __main__\n"
        b"import json\n"
        b"import os\n"
        b"def _verify(mode, tail):\n"
        b"    __main__._arm_original_confirmatory_e_claim_after_full_prevalidation()\n"
        b"    descriptor, _path, claim_sha256, claim_size = "
        b"__main__._take_original_confirmatory_e_claim_read_handle()\n"
        b"    claim = os.read(descriptor, claim_size + 1)\n"
        b"    os.close(descriptor)\n"
        b"    print(json.dumps({'argv': [mode, *tail], 'handler': mode, "
        b"'claim_sha256': claim_sha256, 'claim_size': claim_size, "
        b"'claim_text': claim.decode('utf-8'), 'terminal_origin': __spec__.origin}, "
        b"sort_keys=True, separators=(',', ':')))\n"
        b"    return 0\n"
        b"def _verify_original_confirmatory_preterminal_from_canonical_tail(tail):\n"
        b"    return _verify('verify-preterminal', tail)\n"
        b"def _verify_original_confirmatory_terminal_from_canonical_tail(tail):\n"
        b"    return _verify('verify-terminal', tail)\n"
    )


def _tail_from_existing_e(job_directory: Path, mode: str) -> list[str]:
    e_payload = _synthetic_staged_e_intent_path(job_directory).read_bytes()
    e_intent = json.loads(e_payload)
    projection = e_intent["command_projections"][mode]
    return [
        mode,
        *projection["tail_argv_before_e_file_sha256"],
        hashlib.sha256(e_payload).hexdigest(),
        *projection["tail_argv_between_e_hashes"],
        e_intent["intent_core_sha256"],
        *projection["tail_argv_after_e_core_sha256"],
    ]


def _argv_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_ready_contract_shared_e_reaches_both_terminal_modes_without_science(
    tmp_path: Path,
) -> None:
    ready_entry_contract = (
        b'{"allowed_modes":["run-confirmatory","verify-preterminal","verify-terminal"],'
        b'"contract_status":"ready","dispatcher":"histo_audit.workflows.'
        b'original_confirmatory_capsule_entry:_dispatch_original_confirmatory_capsule",'
        b'"policy":"original_confirmatory_execution_capsule_entry_contract_v1",'
        b'"schema_version":1}\n'
    )
    ready_entry_contract_sha256 = "50c2796e0a3e1e06ec3fea3964c9ed1795f9552f85dbd394618529eba61bb844"
    carrier_root = Path(__file__).resolve().parents[1]
    contract_candidates = [
        path
        for path in (
            carrier_root / "entry_contract.json",
            carrier_root / "project_entry_contract_reference.json",
        )
        if path.is_file()
    ]
    assert len(contract_candidates) == 1
    checked_in_contract = contract_candidates[0]
    checked_in_contract_bytes = checked_in_contract.read_bytes()
    assert checked_in_contract_bytes == ready_entry_contract
    files = _synthetic_tree(tmp_path / "source")
    _install_control_only_production_entry_path(files)
    files["contract"].write_bytes(checked_in_contract_bytes)
    assert files["contract"].read_bytes() == ready_entry_contract
    assert len(ready_entry_contract) == 305
    assert hashlib.sha256(ready_entry_contract).hexdigest() == ready_entry_contract_sha256

    expected = source_inventory(_discover(files))
    result = build_project_capsule(
        package_root=files["package"],
        bootstrap_path=files["bootstrap"],
        policy_path=files["policy"],
        entry_contract_path=files["contract"],
        expected_inventory=expected,
        output_path=tmp_path / "source.pyz",
    )
    capsule = _stage_content_addressed_capsule(
        result,
        tmp_path / "execution_capsules",
    )
    with zipfile.ZipFile(capsule) as archive:
        archived_contract = archive.read("aanca_capsule/entry_contract.json")
    assert archived_contract == ready_entry_contract
    assert hashlib.sha256(archived_contract).hexdigest() == ready_entry_contract_sha256

    working = tmp_path / "working"
    working.mkdir()
    preterminal_tail = _prepare_exact_q_e(
        capsule,
        cwd=working,
        mode="verify-preterminal",
    )
    job_directory = _synthetic_prepared_job_directory(tmp_path)
    e_path = _synthetic_staged_e_intent_path(job_directory)
    claim_path = job_directory / "e_intent_consumed.json"
    e_payload_before = e_path.read_bytes()
    e_identity_before = bootstrap._stable_file_identity(os.lstat(e_path))
    assert claim_path.exists()
    assert not (tmp_path / "artifacts" / "runs").exists()

    program = str(tmp_path / ".venv" / "Scripts" / "python.exe")
    assert claim_path.read_bytes() == b'{"synthetic_claim":true}\n'
    assert bootstrap._readonly_file(os.lstat(claim_path))
    claim_payload_before = claim_path.read_bytes()
    claim_identity_before = bootstrap._stable_file_identity(os.lstat(claim_path))
    assert not (tmp_path / "artifacts" / "runs").exists()

    observed_modes: list[str] = []
    for mode in ("verify-preterminal", "verify-terminal"):
        tail = (
            preterminal_tail
            if mode == "verify-preterminal"
            else _tail_from_existing_e(job_directory, mode)
        )
        assert _argv_value(tail, "--e-intent") == str(e_path)
        assert (
            _argv_value(tail, "--e-intent-sha256") == hashlib.sha256(e_payload_before).hexdigest()
        )
        verify = _invoke_capsule(
            capsule,
            cwd=working,
            program=program,
            tail=tail,
        )
        assert verify.returncode == 0, verify.stderr.decode("utf-8", errors="replace")
        assert verify.stderr == b""
        verify_payload = json.loads(verify.stdout)
        observed_modes.append(verify_payload["argv"][0])
        assert verify_payload["argv"] == tail
        assert verify_payload["handler"] == mode
        assert verify_payload["claim_text"] == claim_payload_before.decode("utf-8")
        assert verify_payload["claim_sha256"] == hashlib.sha256(claim_payload_before).hexdigest()
        assert verify_payload["claim_size"] == len(claim_payload_before)
        assert ".pyz" in verify_payload["terminal_origin"]

    assert observed_modes == ["verify-preterminal", "verify-terminal"]
    assert e_path.read_bytes() == e_payload_before
    assert bootstrap._stable_file_identity(os.lstat(e_path)) == e_identity_before
    assert claim_path.read_bytes() == claim_payload_before
    assert bootstrap._stable_file_identity(os.lstat(claim_path)) == claim_identity_before
    assert not (tmp_path / "artifacts" / "runs").exists()
