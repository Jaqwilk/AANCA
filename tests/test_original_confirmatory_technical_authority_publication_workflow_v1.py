"""Synthetic transaction tests against the live T0 schema/builder/verifier."""

from __future__ import annotations

import dis
import gc
import inspect
import json
import os
import stat
import subprocess
import sys
import threading
import uuid
import weakref
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from histo_audit.workflows import (
    original_confirmatory_technical_authority_publication_v1 as publication,
)
from histo_audit.workflows import (
    original_confirmatory_technical_authority_v1 as schema,
)

PROJECT_ROOT = Path(os.environ.get("AANCA_PROJECT_ROOT", Path.cwd())).resolve()


def _opcode_offset(
    function: Any,
    *,
    opname: str,
    argval: Any,
    occurrence: int = 0,
) -> int:
    matches = [
        instruction.offset
        for instruction in dis.get_instructions(function)
        if instruction.opname == opname and instruction.argval == argval
    ]
    assert len(matches) > occurrence, (function.__name__, opname, argval, matches)
    return matches[occurrence]


def _next_opcode_offset_after_global_call(
    function: Any,
    *,
    global_name: str,
    next_opname: str,
    occurrence: int = 0,
) -> int:
    instructions = list(dis.get_instructions(function))
    call_indexes: list[int] = []
    for index, instruction in enumerate(instructions):
        if instruction.opname != "LOAD_GLOBAL" or instruction.argval != global_name:
            continue
        call_indexes.append(
            next(
                candidate
                for candidate in range(index + 1, len(instructions))
                if instructions[candidate].opname == "CALL"
            )
        )
    assert len(call_indexes) > occurrence, (
        function.__name__,
        global_name,
        call_indexes,
    )
    call_index = call_indexes[occurrence]
    return next(
        instruction.offset
        for instruction in instructions[call_index + 1 :]
        if instruction.opname == next_opname
    )


def _run_single_instruction_crash_cut(
    function: Any,
    *,
    offset: int,
    invoke: Any,
) -> None:
    tool_id = 5
    sys.monitoring.use_tool_id(tool_id, "aanca-t0-single-crash-cut")

    def callback(code: Any, instruction_offset: int) -> None:
        if code is function.__code__ and instruction_offset == offset:
            sys.monitoring.set_local_events(tool_id, function.__code__, 0)
            raise KeyboardInterrupt(f"synthetic single crash cut at {function.__name__}:{offset}")

    sys.monitoring.register_callback(
        tool_id,
        sys.monitoring.events.INSTRUCTION,
        callback,
    )
    sys.monitoring.set_local_events(
        tool_id,
        function.__code__,
        sys.monitoring.events.INSTRUCTION,
    )
    try:
        invoke()
    finally:
        sys.monitoring.set_local_events(tool_id, function.__code__, 0)
        sys.monitoring.register_callback(
            tool_id,
            sys.monitoring.events.INSTRUCTION,
            None,
        )
        sys.monitoring.free_tool_id(tool_id)


def _atomic_test_child(
    code: str,
) -> tuple[list[str], dict[str, str]]:
    executable = Path(publication.psutil.Process(os.getpid()).exe()).resolve()
    environment = os.environ.copy()
    if sys.prefix != sys.base_prefix:
        environment["__PYVENV_LAUNCHER__"] = sys.executable
    return [str(executable), "-B", "-c", code], environment


def _launch_retained_test_child(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> publication._RetainedReviewerChildV1:
    owner = publication._RetainedReviewerChildOwnerV1()
    publication._launch_retained_reviewer_child_v1(
        argv,
        cwd=cwd,
        env=env,
        child_owner=owner,
    )
    assert owner.child is not None
    return owner.child


def _assert_process_instance_is_terminal(
    process_id: int,
    process_created_at: float,
) -> None:
    try:
        current = publication.psutil.Process(process_id)
        assert current.create_time() != process_created_at
    except publication.psutil.NoSuchProcess:
        pass


def _create_named_windows_test_event(name: str) -> int:
    kernel32 = publication.ctypes.WinDLL("kernel32", use_last_error=True)
    create_event = kernel32.CreateEventW
    create_event.argtypes = [
        publication.ctypes.c_void_p,
        publication.ctypes.c_int,
        publication.ctypes.c_int,
        publication.ctypes.c_wchar_p,
    ]
    create_event.restype = publication.ctypes.c_void_p
    publication.ctypes.set_last_error(0)
    handle = int(create_event(None, 1, 0, name) or 0)
    if not handle:
        raise publication.ctypes.WinError(publication.ctypes.get_last_error())
    return handle


def _create_inheritable_unnamed_windows_test_event() -> int:
    kernel32 = publication.ctypes.WinDLL("kernel32", use_last_error=True)
    create_event = kernel32.CreateEventW
    create_event.argtypes = [
        publication.ctypes.POINTER(publication._SecurityAttributes),
        publication.ctypes.c_int,
        publication.ctypes.c_int,
        publication.ctypes.c_wchar_p,
    ]
    create_event.restype = publication.ctypes.c_void_p
    security = publication._SecurityAttributes(
        nLength=publication.ctypes.sizeof(publication._SecurityAttributes),
        lpSecurityDescriptor=None,
        bInheritHandle=1,
    )
    publication.ctypes.set_last_error(0)
    handle = int(create_event(publication.ctypes.byref(security), 1, 0, None) or 0)
    if not handle:
        raise publication.ctypes.WinError(publication.ctypes.get_last_error())
    return handle


def _wait_windows_test_handle(handle: int, timeout_milliseconds: int) -> int:
    kernel32 = publication.ctypes.WinDLL("kernel32", use_last_error=True)
    wait = kernel32.WaitForSingleObject
    wait.argtypes = [publication.ctypes.c_void_p, publication.ctypes.c_ulong]
    wait.restype = publication.ctypes.c_ulong
    return int(wait(publication.ctypes.c_void_p(handle), timeout_milliseconds))


def _record_valid_windows_handle_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[int], Any]:
    original_close = publication._close_windows_handle_v1
    closed_handles: list[int] = []
    invalid_handles = {0, int(publication.ctypes.c_void_p(-1).value)}

    def recording_close(handle: int, *, role: str) -> None:
        if handle not in invalid_handles:
            closed_handles.append(handle)
        original_close(handle, role=role)

    monkeypatch.setattr(publication, "_close_windows_handle_v1", recording_close)
    return closed_handles, original_close


def _open_windows_process_wait_handle(process_id: int) -> int:
    kernel32 = publication.ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [
        publication.ctypes.c_ulong,
        publication.ctypes.c_int,
        publication.ctypes.c_ulong,
    ]
    open_process.restype = publication.ctypes.c_void_p
    publication.ctypes.set_last_error(0)
    # SYNCHRONIZE | PROCESS_TERMINATE: terminal wait is the normal path and
    # PROCESS_TERMINATE makes the test's failure-only cleanup effective.
    handle = int(open_process(0x00100000 | 0x0001, 0, process_id) or 0)
    if not handle:
        raise publication.ctypes.WinError(publication.ctypes.get_last_error())
    return handle


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.write_bytes(schema.canonical_json_line_bytes(value))
    return path


def _install_frozen_snapshots(project_root: Path) -> None:
    (project_root / "PRE_REGISTRATION.md").write_bytes(
        (PROJECT_ROOT / "PRE_REGISTRATION.md").read_bytes()
    )
    configs = project_root / "configs"
    configs.mkdir(exist_ok=True)
    for name in ("primary_frozen.yaml", "confirmatory_frozen.yaml"):
        (configs / name).write_bytes((PROJECT_ROOT / "configs" / name).read_bytes())


def _install_request_chain(
    project_root: Path,
    inputs: dict[str, Any],
) -> dict[str, Path]:
    request_directory = project_root / "artifacts" / publication.AUTHORITY_REQUEST_DIRECTORY_NAME
    request_directory.mkdir(parents=True, exist_ok=True)
    intent_path = request_directory / publication.INTENT_REQUEST_FILENAME
    review_path = request_directory / publication.REVIEW_REQUEST_FILENAME
    attempt_path = request_directory / publication.REVIEW_ATTEMPT_FILENAME
    controller = publication.capture_current_process_identity_v1(
        Path(publication.__file__).resolve()
    )
    reviewer_spec = publication.importlib.util.find_spec(publication.REVIEWER_MODULE_NAME)
    assert reviewer_spec is not None and reviewer_spec.origin is not None
    if not intent_path.exists():
        _write_json(intent_path, inputs["intent"])
        intent_path.chmod(stat.S_IREAD)
    attempt = publication._build_original_confirmatory_technical_review_attempt_claim_at_v1(
        intent=inputs["intent"],
        project_root=project_root,
        controller_process=controller,
        reviewer_implementation_path=Path(reviewer_spec.origin).resolve(),
        attempt_created_at_utc=_timestamp_after_process_creation(controller),
    )
    if not attempt_path.exists():
        _write_json(attempt_path, attempt)
        attempt_path.chmod(stat.S_IREAD)
    if not review_path.exists():
        _write_json(review_path, inputs["review"])
        review_path.chmod(stat.S_IREAD)
    return {
        "intent": intent_path,
        "attempt": attempt_path,
        "review": review_path,
    }


def _timestamp_after_process_creation(
    process: dict[str, Any],
    *,
    offset_microseconds: int = 1,
) -> str:
    created = datetime.fromisoformat(str(process["process_created_at_utc"]).replace("Z", "+00:00"))
    return (
        (created + timedelta(microseconds=offset_microseconds))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _process(root: Path, *, process_id: int, created_at_utc: str) -> dict[str, Any]:
    return {
        "process_id": process_id,
        "process_created_at_utc": created_at_utc,
        "executable_path": str((root / f"python-{process_id}.exe").resolve()),
        "executable_size_bytes": 1,
        "executable_sha256": f"{process_id % 10}" * 64,
        "implementation_path": str((root / f"implementation-{process_id}.py").resolve()),
        "implementation_sha256": f"{(process_id + 1) % 10}" * 64,
    }


def _inputs(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path.resolve()
    test_process = publication.capture_current_process_identity_v1(
        Path(publication.__file__).resolve()
    )
    review_started_at_utc = _timestamp_after_process_creation(
        test_process,
        offset_microseconds=10,
    )
    review_completed_at_utc = _timestamp_after_process_creation(
        test_process,
        offset_microseconds=20,
    )
    publication_timestamp_utc = _timestamp_after_process_creation(
        test_process,
        offset_microseconds=30,
    )
    source_inventory = {
        "schema_version": 3,
        "policy": "synthetic_runtracker_capture_source_tree",
        "root_sha256": "1" * 64,
        "artifacts": [
            {
                "path": "src/histo_audit/synthetic.py",
                "size_bytes": 3,
                "sha256": "2" * 64,
            }
        ],
    }
    source_bytes = schema.canonical_json_line_bytes(source_inventory)
    capsule_sha256 = "3" * 64
    parent = {
        "schema_version": 1,
        "authority_kind": "preregistration_amendment",
        "authority_directory": str(
            (
                root / "artifacts" / "preregistration_amendments" / "20260727T133947.089370Z"
            ).resolve()
        ),
        "chain_depth": schema.PARENT_CHAIN_DEPTH,
        "artifact_root_sha256": schema.PARENT_ARTIFACT_ROOT_SHA256,
        "sha256_manifest_sha256": schema.PARENT_MANIFEST_SHA256,
        "execution_source_root_sha256": schema.PARENT_SOURCE_ROOT_SHA256,
        "execution_source_manifest_sha256": schema.PARENT_SOURCE_MANIFEST_SHA256,
    }
    frozen_science = {
        "schema_version": 1,
        "preregistration_path": str((PROJECT_ROOT / "PRE_REGISTRATION.md").resolve()),
        "preregistration_sha256": schema.PREREGISTRATION_SHA256,
        "primary_config_path": str((PROJECT_ROOT / "configs" / "primary_frozen.yaml").resolve()),
        "primary_config_sha256": schema.PRIMARY_CONFIG_SHA256,
        "primary_config_semantic_sha256": schema.PRIMARY_CONFIG_SEMANTIC_SHA256,
        "confirmatory_config_path": str(
            (PROJECT_ROOT / "configs" / "confirmatory_frozen.yaml").resolve()
        ),
        "confirmatory_config_sha256": schema.CONFIRMATORY_CONFIG_SHA256,
        "confirmatory_config_semantic_sha256": (schema.CONFIRMATORY_CONFIG_SEMANTIC_SHA256),
        "scientific_definition_changed": False,
    }
    historical_primary = {
        "schema_version": 1,
        "run_directory": str(
            (root / "artifacts" / "runs" / schema.HISTORICAL_PRIMARY_RUN_ID).resolve()
        ),
        "run_id": schema.HISTORICAL_PRIMARY_RUN_ID,
        "terminal_status": "completed",
        "completion_stage": "PRIMARY_STUDY_COMPLETE",
        "artifact_root_sha256": schema.HISTORICAL_PRIMARY_ARTIFACT_ROOT_SHA256,
        "artifact_manifest_sha256": schema.HISTORICAL_PRIMARY_MANIFEST_SHA256,
        "required_cell_count": 185,
        "completed_required_cell_count": 185,
        "skipped_optional_cell_count": 37,
        "failed_required_cell_count": 0,
        "retrained_cell_count": 0,
        "verification_scope": ("integrity_and_control_metadata_only_no_scientific_outcome_values"),
        "outcome_values_read": False,
    }
    execution_source = {
        "schema_version": 1,
        "policy": "runtracker_capture_source_tree_exact_object_v1",
        "manifest_path": str((root / "source_inventory.json").resolve()),
        "manifest_sha256": publication._sha256_bytes(source_bytes),
        "root_sha256": source_inventory["root_sha256"],
        "record_count": len(source_inventory["artifacts"]),
    }
    execution_capsule = {
        "schema_version": 1,
        "policy": "content_addressed_original_confirmatory_execution_capsule_v1",
        "path": str((root / "capsules" / capsule_sha256 / "original_confirmatory.pyz").resolve()),
        "size_bytes": 10,
        "sha256": capsule_sha256,
        "internal_manifest_sha256": "4" * 64,
        "source_records_root_sha256": "5" * 64,
        "publication_receipt_path": str((root / "capsule_publication.json").resolve()),
        "publication_receipt_sha256": "6" * 64,
        "independent_readback_path": str((root / "capsule_readback.json").resolve()),
        "independent_readback_sha256": "7" * 64,
        "content_addressed_create_new_verified": True,
        "scientific_execution_performed": False,
    }
    capacity_v2 = {
        "schema_version": schema.CAPACITY_SCHEMA_VERSION,
        "policy": schema.CAPACITY_POLICY_NAME,
        "policy_sha256": schema.CAPACITY_POLICY_SHA256,
        "receipt_path": str((root / "capacity_v2.json").resolve()),
        "receipt_sha256": "8" * 64,
        "required_free_bytes": schema.CAPACITY_REQUIRED_FREE_BYTES,
        "observed_free_bytes": schema.CAPACITY_REQUIRED_FREE_BYTES + 1,
        "passed": True,
        "capsule_sha256": capsule_sha256,
        "execution_source_root_sha256": source_inventory["root_sha256"],
        "outcome_values_read": False,
        "scientific_execution_performed": False,
    }
    outcome_scope = {
        "schema_version": 1,
        "primary_outcomes_inspected": True,
        "primary_outcomes_inspected_at_utc": schema.PRIMARY_OUTCOME_INSPECTION_AT_UTC,
        "primary_analysis_disposition": "amended_or_exploratory",
        "confirmatory_outcomes_inspected": False,
        "confirmatory_outcome_values_read": False,
        "confirmatory_registration_status": "original_frozen_confirmatory_unchanged",
        "selection_performed": False,
        "tuning_performed": False,
        "scientific_execution_performed": False,
        "automatic_retry_allowed": False,
    }
    intent = schema.build_original_confirmatory_technical_authority_intent_v1(
        created_at_utc="2026-07-31T10:00:00.000000Z",
        builder_process=_process(
            root,
            process_id=101,
            created_at_utc="2026-07-31T09:59:59.000000Z",
        ),
        parent=parent,
        frozen_science=frozen_science,
        historical_primary=historical_primary,
        execution_source=execution_source,
        execution_capsule=execution_capsule,
        capacity_v2=capacity_v2,
        outcome_scope=outcome_scope,
    )
    review = schema.build_original_confirmatory_technical_authority_review_v1(
        intent=intent,
        review_started_at_utc=review_started_at_utc,
        review_completed_at_utc=review_completed_at_utc,
        reviewer_process=_process(
            root,
            process_id=202,
            created_at_utc="2026-07-31T10:00:30.000000Z",
        ),
    )
    return {
        "intent": intent,
        "review": review,
        "source_inventory": source_inventory,
        "parent": parent,
        "frozen_science": frozen_science,
        "historical_primary": historical_primary,
        "execution_source": execution_source,
        "execution_capsule": execution_capsule,
        "capacity_v2": capacity_v2,
        "outcome_scope": outcome_scope,
        "publication_timestamp_utc": publication_timestamp_utc,
    }


def _bundle(
    tmp_path: Path,
    *,
    destination: Path | None = None,
) -> schema.OriginalConfirmatoryTechnicalAuthorityBundle:
    inputs = _inputs(tmp_path)
    _install_request_chain(tmp_path, inputs)
    namespace = (tmp_path / "artifacts" / publication.AUTHORITY_NAMESPACE_DIRECTORY_NAME).resolve()
    namespace.mkdir(parents=True, exist_ok=True)
    target = (destination or (namespace / "authority")).resolve()
    return schema.build_original_confirmatory_technical_authority_bundle_v1(
        authority_directory=target,
        intent=inputs["intent"],
        independent_review=inputs["review"],
        publication_timestamp_utc=inputs["publication_timestamp_utc"],
        preregistration_bytes=(PROJECT_ROOT / "PRE_REGISTRATION.md").read_bytes(),
        primary_config_bytes=(PROJECT_ROOT / "configs" / "primary_frozen.yaml").read_bytes(),
        confirmatory_config_bytes=(
            PROJECT_ROOT / "configs" / "confirmatory_frozen.yaml"
        ).read_bytes(),
        source_inventory=inputs["source_inventory"],
    )


def test_current_process_identity_is_captured_not_caller_declared() -> None:
    implementation = Path(publication.__file__).resolve()
    identity = publication.capture_current_process_identity_v1(implementation)
    executable = Path(identity["executable_path"])

    assert identity["process_id"] == os.getpid()
    assert Path(identity["implementation_path"]) == implementation
    assert identity["implementation_sha256"] == publication._sha256_bytes(
        implementation.read_bytes()
    )
    assert identity["executable_size_bytes"] == executable.stat().st_size
    assert identity["executable_sha256"] == publication._sha256_bytes(executable.read_bytes())
    assert identity["process_created_at_utc"].endswith("Z")


def test_atomic_launcher_has_no_popen_postassignment_or_breakaway_fallback() -> None:
    source = Path(publication.__file__).read_text(encoding="utf-8")
    atomic_create_source = inspect.getsource(publication._create_atomic_job_bound_process_v1)
    attribute_source = inspect.getsource(publication._create_reviewer_attribute_list_v1)
    launcher_source = inspect.getsource(publication._launch_retained_reviewer_child_v1)
    launch_capture_source = inspect.getsource(publication._launch_capture_and_wait_for_reviewer_v1)
    pipe_capture_source = "\n".join(
        (
            inspect.getsource(publication._ReviewerPipeCaptureV1),
            inspect.getsource(publication._read_reviewer_pipe_chunk_v1),
            inspect.getsource(publication._reviewer_pipe_capture_from_handle_v1),
        )
    )

    assert "subprocess.Popen" not in source
    assert "AssignProcessToJobObject" not in source
    assert "CREATE_BREAKAWAY_FROM_JOB" not in source
    assert "open_osfhandle" not in pipe_capture_source
    assert "os.fdopen" not in pipe_capture_source
    assert "_CREATE_SUSPENDED" in atomic_create_source
    assert "_EXTENDED_STARTUPINFO_PRESENT" in atomic_create_source
    assert "process_information: _ProcessInformation" in atomic_create_source
    assert "attribute_owner: _ReviewerAttributeListOwnerV1" in atomic_create_source
    assert "return process_information" not in atomic_create_source
    assert "_PROC_THREAD_ATTRIBUTE_HANDLE_LIST" in attribute_source
    assert "_PROC_THREAD_ATTRIBUTE_JOB_LIST" in attribute_source
    assert "owner: _ReviewerRawLaunchHandleOwnerV1" in inspect.getsource(
        publication._create_reviewer_kill_job_v1
    )
    assert "owner: _ReviewerRawLaunchHandleOwnerV1" in inspect.getsource(
        publication._create_reviewer_null_input_v1
    )
    assert "owner: _ReviewerRawLaunchHandleOwnerV1" in inspect.getsource(
        publication._create_reviewer_pipe_v1
    )
    assert "_create_reviewer_pipe_v1(owner=raw_handles" in launcher_source
    assert "= _create_reviewer_kill_job_v1(" not in launcher_source
    assert "= _create_reviewer_null_input_v1(" not in launcher_source
    assert "= _create_reviewer_pipe_v1(" not in launcher_source
    assert "child_owner: _RetainedReviewerChildOwnerV1" in launcher_source
    assert "child_owner = _RetainedReviewerChildOwnerV1()" in launch_capture_source
    assert "child = _launch_retained_reviewer_child_v1" not in launch_capture_source


@pytest.mark.skipif(os.name != "nt", reason="Windows HANDLE ownership contract")
@pytest.mark.parametrize(
    ("creator_name", "store_opname", "store_argval"),
    (
        ("_create_reviewer_kill_job_v1", "STORE_FAST", "raw_handle"),
        ("_create_reviewer_kill_job_v1", "STORE_ATTR", "job"),
        ("_create_reviewer_null_input_v1", "STORE_FAST", "raw_handle"),
        ("_create_reviewer_null_input_v1", "STORE_ATTR", "stdin"),
    ),
)
def test_handle_return_api_raii_closes_native_call_to_owner_store_crash_cuts(
    creator_name: str,
    store_opname: str,
    store_argval: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One async cut cannot orphan a HANDLE returned on the evaluation stack."""

    owner = publication._ReviewerRawLaunchHandleOwnerV1()
    creator = getattr(publication, creator_name)
    offset = _opcode_offset(
        creator,
        opname=store_opname,
        argval=store_argval,
    )
    original_close = publication._close_windows_handle_v1
    closed_handles: list[int] = []

    def recording_close(handle: int, *, role: str) -> None:
        closed_handles.append(handle)
        original_close(handle, role=role)

    monkeypatch.setattr(publication, "_close_windows_handle_v1", recording_close)

    with pytest.raises(KeyboardInterrupt, match="synthetic single crash cut"):
        _run_single_instruction_crash_cut(
            creator,
            offset=offset,
            invoke=lambda: creator(owner=owner),
        )
    gc.collect()

    assert not owner.job.is_valid()
    assert not owner.stdin.is_valid()
    assert len(closed_handles) == 1
    assert closed_handles[0] not in {0, int(publication.ctypes.c_void_p(-1).value)}


@pytest.mark.skipif(os.name != "nt", reason="Windows HANDLE ownership contract")
def test_owned_windows_handle_alias_double_close_and_cleanup_failure_are_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = publication._ReviewerRawLaunchHandleOwnerV1()
    publication._create_reviewer_kill_job_v1(owner=owner)
    alias = owner.job
    handle = alias.value_int()
    original_close = publication._close_windows_handle_v1
    closed_handles: list[int] = []

    def recording_close(raw: int, *, role: str) -> None:
        closed_handles.append(raw)
        original_close(raw, role=role)

    monkeypatch.setattr(publication, "_close_windows_handle_v1", recording_close)
    assert owner.job.close_noexcept() is True
    assert alias.close_noexcept() is True
    gc.collect()

    assert closed_handles == [handle]
    assert not owner.job.is_valid()

    synthetic = publication._OwnedWinHandleV1(123)
    failed_calls = 0

    def fail_close(_raw: int, *, role: str) -> None:
        nonlocal failed_calls
        failed_calls += 1
        raise OSError(f"synthetic cleanup failure for {role}")

    monkeypatch.setattr(publication, "_close_windows_handle_v1", fail_close)
    assert synthetic.close_noexcept() is False
    assert synthetic.close_noexcept() is True
    del synthetic
    gc.collect()
    assert failed_calls == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows HANDLE ownership contract")
def test_create_pipe_partial_configuration_failure_closes_both_owned_handles_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = publication._ReviewerRawLaunchHandleOwnerV1()
    original_close = publication._close_windows_handle_v1
    closed_handles: list[int] = []

    def fail_inheritance(_handle: int, *, inheritable: bool) -> None:
        assert inheritable is False
        raise KeyboardInterrupt("synthetic post-CreatePipe configuration cut")

    def recording_close(handle: int, *, role: str) -> None:
        closed_handles.append(handle)
        original_close(handle, role=role)

    monkeypatch.setattr(
        publication,
        "_set_windows_handle_inheritance_v1",
        fail_inheritance,
    )
    monkeypatch.setattr(publication, "_close_windows_handle_v1", recording_close)

    with pytest.raises(KeyboardInterrupt, match="post-CreatePipe configuration cut"):
        publication._create_reviewer_pipe_v1(owner=owner, stream="stdout")
    gc.collect()

    assert len(closed_handles) == 2
    assert len(set(closed_handles)) == 2
    assert not owner.stdout_read.is_valid()
    assert not owner.stdout_write.is_valid()


@pytest.mark.skipif(os.name != "nt", reason="Windows HANDLE ownership contract")
def test_create_pipe_native_return_crash_cut_is_closed_by_preallocated_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CreatePipe writes both owned wrappers before Python regains control."""

    argv, environment = _atomic_test_child("pass")
    function = publication._create_reviewer_pipe_v1
    instructions = list(dis.get_instructions(function))
    create_call_index = next(
        index
        for index, instruction in enumerate(instructions)
        if instruction.opname == "LOAD_FAST" and instruction.argval == "create_pipe"
    )
    target = next(
        instruction.offset
        for instruction in instructions[create_call_index + 1 :]
        if instruction.opname == "POP_JUMP_IF_TRUE"
    )
    gc.collect()
    sentinel = _create_named_windows_test_event(
        f"Local\\aanca-create-pipe-crash-cut-{uuid.uuid4()}"
    )
    closed_handles, original_close = _record_valid_windows_handle_closes(monkeypatch)
    try:
        with pytest.raises(KeyboardInterrupt, match="synthetic single crash cut"):
            _run_single_instruction_crash_cut(
                function,
                offset=target,
                invoke=lambda: _launch_retained_test_child(
                    argv,
                    cwd=tmp_path.resolve(),
                    env=environment,
                ),
            )
        gc.collect()

        assert len(closed_handles) == 4
        assert len(set(closed_handles)) == 4
        assert sentinel not in closed_handles
        assert _wait_windows_test_handle(sentinel, 0) == 258
    finally:
        original_close(sentinel, role="test crash-cut sentinel")


@pytest.mark.skipif(os.name != "nt", reason="Windows HANDLE ownership contract")
@pytest.mark.parametrize(
    ("creator_name", "occurrence", "expected_closed_handles"),
    (
        ("_create_reviewer_kill_job_v1", 0, 1),
        ("_create_reviewer_null_input_v1", 0, 2),
        ("_create_reviewer_pipe_v1", 0, 4),
        ("_create_reviewer_pipe_v1", 1, 6),
    ),
)
def test_launcher_post_creator_return_crash_cuts_close_preowned_handles(
    creator_name: str,
    occurrence: int,
    expected_closed_handles: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function = publication._launch_retained_reviewer_child_v1
    target = _next_opcode_offset_after_global_call(
        function,
        global_name=creator_name,
        next_opname="POP_TOP",
        occurrence=occurrence,
    )
    argv, environment = _atomic_test_child("pass")
    gc.collect()
    sentinel = _create_named_windows_test_event(f"Local\\aanca-launcher-crash-cut-{uuid.uuid4()}")
    closed_handles, original_close = _record_valid_windows_handle_closes(monkeypatch)
    try:
        with pytest.raises(KeyboardInterrupt, match="synthetic single crash cut"):
            _run_single_instruction_crash_cut(
                function,
                offset=target,
                invoke=lambda: _launch_retained_test_child(
                    argv,
                    cwd=tmp_path.resolve(),
                    env=environment,
                ),
            )
        gc.collect()

        assert len(closed_handles) == expected_closed_handles
        assert len(set(closed_handles)) == expected_closed_handles
        assert sentinel not in closed_handles
        assert _wait_windows_test_handle(sentinel, 0) == 258
    finally:
        original_close(sentinel, role="test crash-cut sentinel")


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_atomic_job_launcher_waits_and_drains_large_stdout_and_stderr(
    tmp_path: Path,
) -> None:
    size = 2 * 1024 * 1024
    argv, environment = _atomic_test_child(
        "import sys;"
        f"sys.stdout.buffer.write(b'O'*{size});"
        "sys.stdout.buffer.flush();"
        f"sys.stderr.buffer.write(b'E'*{size});"
        "sys.stderr.buffer.flush()"
    )
    child = _launch_retained_test_child(
        argv,
        cwd=tmp_path.resolve(),
        env=environment,
    )
    try:
        assert child.resumed is False
        child.resume_exactly_once()
        stdout, stderr = child.communicate()
        assert child.returncode == 0
        assert stdout == b"O" * size
        assert stderr == b"E" * size
        child.close_after_wait()
        assert child.custody_closed is True
    finally:
        child.close_job_then_wait_noexcept()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_pipe_capture_construction_failure_closes_each_read_handle_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, environment = _atomic_test_child("pass")
    original_pipe = publication._create_reviewer_pipe_v1
    original_capture = publication._reviewer_pipe_capture_from_handle_v1
    original_close = publication._close_windows_handle_v1
    read_handles: list[int] = []
    closed_handles: list[int] = []
    capture_calls = 0

    def recording_pipe(
        *,
        owner: publication._ReviewerRawLaunchHandleOwnerV1,
        stream: str,
    ) -> None:
        original_pipe(owner=owner, stream=stream)
        read_handles.append(
            (owner.stdout_read if stream == "stdout" else owner.stderr_read).value_int()
        )

    def fail_second_capture(
        handle: publication._OwnedWinHandleV1,
    ) -> publication._ReviewerPipeCaptureV1:
        nonlocal capture_calls
        capture_calls += 1
        if capture_calls == 2:
            raise RuntimeError("synthetic native capture construction failure")
        return original_capture(handle)

    def recording_close(handle: int, *, role: str) -> None:
        closed_handles.append(handle)
        original_close(handle, role=role)

    monkeypatch.setattr(publication, "_create_reviewer_pipe_v1", recording_pipe)
    monkeypatch.setattr(
        publication,
        "_reviewer_pipe_capture_from_handle_v1",
        fail_second_capture,
    )
    monkeypatch.setattr(publication, "_close_windows_handle_v1", recording_close)

    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="synthetic native capture construction failure",
    ):
        _launch_retained_test_child(
            argv,
            cwd=tmp_path.resolve(),
            env=environment,
        )

    assert len(read_handles) == 2
    assert all(closed_handles.count(handle) == 1 for handle in read_handles)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_pipe_drain_start_failure_closes_each_owned_read_handle_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, environment = _atomic_test_child("pass")
    original_pipe = publication._create_reviewer_pipe_v1
    original_close = publication._close_windows_handle_v1
    read_handles: list[int] = []
    closed_handles: list[int] = []

    def recording_pipe(
        *,
        owner: publication._ReviewerRawLaunchHandleOwnerV1,
        stream: str,
    ) -> None:
        original_pipe(owner=owner, stream=stream)
        read_handles.append(
            (owner.stdout_read if stream == "stdout" else owner.stderr_read).value_int()
        )

    def fail_start(_thread: threading.Thread) -> None:
        raise RuntimeError("synthetic native Thread.start failure")

    def recording_close(handle: int, *, role: str) -> None:
        closed_handles.append(handle)
        original_close(handle, role=role)

    monkeypatch.setattr(publication, "_create_reviewer_pipe_v1", recording_pipe)
    monkeypatch.setattr(publication.threading.Thread, "start", fail_start)
    monkeypatch.setattr(publication, "_close_windows_handle_v1", recording_close)

    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match=r"synthetic native Thread\.start failure",
    ):
        _launch_retained_test_child(
            argv,
            cwd=tmp_path.resolve(),
            env=environment,
        )

    assert len(read_handles) == 2
    assert all(closed_handles.count(handle) == 1 for handle in read_handles)


def test_launch_to_capture_async_exception_closes_child_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SyntheticChild:
        close_calls = 0

        def close_job_then_wait_noexcept(self) -> None:
            self.close_calls += 1

    child = SyntheticChild()

    def fake_launch(*_args: Any, **kwargs: Any) -> None:
        kwargs["child_owner"].child = child

    def interrupt_before_capture(*_args: Any, **_kwargs: Any) -> Any:
        raise KeyboardInterrupt("synthetic launch-to-capture interruption")

    monkeypatch.setattr(
        publication,
        "_launch_retained_reviewer_child_v1",
        fake_launch,
    )
    monkeypatch.setattr(
        publication,
        "_capture_and_wait_for_reviewer_child_v1",
        interrupt_before_capture,
    )

    with pytest.raises(KeyboardInterrupt, match="launch-to-capture interruption"):
        publication._launch_capture_and_wait_for_reviewer_v1(
            ["C:\\exact\\python.exe", "-B", "-c", "pass"],
            cwd=tmp_path.resolve(),
            env={},
            reviewer_implementation=(tmp_path / "reviewer.py").resolve(),
        )

    assert child.close_calls == 1


def test_launch_return_async_exception_uses_preallocated_child_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SyntheticChild:
        close_calls = 0

        def close_job_then_wait_noexcept(self) -> None:
            self.close_calls += 1

    child = SyntheticChild()
    capture_calls = 0

    def fill_owner_then_interrupt(*_args: Any, **kwargs: Any) -> None:
        kwargs["child_owner"].child = child
        raise KeyboardInterrupt("synthetic child-owner return interruption")

    def forbidden_capture(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal capture_calls
        capture_calls += 1
        raise AssertionError("capture cannot run after launch interruption")

    monkeypatch.setattr(
        publication,
        "_launch_retained_reviewer_child_v1",
        fill_owner_then_interrupt,
    )
    monkeypatch.setattr(
        publication,
        "_capture_and_wait_for_reviewer_child_v1",
        forbidden_capture,
    )

    with pytest.raises(KeyboardInterrupt, match="child-owner return interruption"):
        publication._launch_capture_and_wait_for_reviewer_v1(
            ["C:\\exact\\python.exe", "-B", "-c", "pass"],
            cwd=tmp_path.resolve(),
            env={},
            reviewer_implementation=(tmp_path / "reviewer.py").resolve(),
        )

    assert child.close_calls == 1
    assert capture_calls == 0


def test_atomic_job_launcher_fails_explicitly_off_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(publication.os, "name", "posix")

    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="requires Windows",
    ):
        _launch_retained_test_child(
            ["C:\\exact\\python.exe", "-c", "pass"],
            cwd=tmp_path,
            env={},
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_atomic_job_membership_readback_failure_kills_suspended_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = (tmp_path / "membership-failure-escaped.txt").resolve()
    argv, environment = _atomic_test_child(
        f"from pathlib import Path;Path({str(sentinel)!r}).write_text('escaped', encoding='utf-8')"
    )
    original_create = publication._create_atomic_job_bound_process_v1
    created: dict[str, int | float] = {}

    def recording_create(*args: Any, **kwargs: Any) -> None:
        process_information = kwargs["process_information"]
        original_create(*args, **kwargs)
        process_id = int(process_information.dwProcessId)
        created["process_id"] = process_id
        created["process_created_at"] = publication.psutil.Process(process_id).create_time()

    def fail_membership(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("synthetic atomic membership readback failure")

    monkeypatch.setattr(
        publication,
        "_create_atomic_job_bound_process_v1",
        recording_create,
    )
    monkeypatch.setattr(
        publication,
        "_require_atomic_reviewer_process_identity_v1",
        fail_membership,
    )

    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="synthetic atomic membership readback failure",
    ):
        _launch_retained_test_child(
            argv,
            cwd=tmp_path.resolve(),
            env=environment,
        )

    assert created.keys() == {"process_id", "process_created_at"}
    assert not sentinel.exists()
    _assert_process_instance_is_terminal(
        int(created["process_id"]),
        float(created["process_created_at"]),
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows STARTUPINFOEX contract")
def test_attribute_list_post_initialize_failure_deletes_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = publication._ReviewerAttributeListOwnerV1()
    original_delete = publication._delete_reviewer_attribute_list_noexcept_v1
    deleted: list[int] = []

    def fail_before_arrays(*_args: Any, **_kwargs: Any) -> Any:
        raise KeyboardInterrupt("synthetic post-initialize interruption")

    def recording_delete(attribute_list: Any) -> None:
        deleted.append(int(attribute_list.value or 0))
        original_delete(attribute_list)

    monkeypatch.setattr(
        publication,
        "_build_reviewer_attribute_value_arrays_v1",
        fail_before_arrays,
    )
    monkeypatch.setattr(
        publication,
        "_delete_reviewer_attribute_list_noexcept_v1",
        recording_delete,
    )

    with pytest.raises(KeyboardInterrupt, match="post-initialize interruption"):
        publication._create_reviewer_attribute_list_v1(
            job_handle=1,
            inherited_handles=(2, 3, 4),
            owner=owner,
        )

    assert len(deleted) == 1
    assert deleted[0] != 0
    assert owner.delete_armed is False
    assert not owner.attribute_list
    assert owner.buffer is None


def test_attribute_initialization_token_retains_buffer_through_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = publication.ctypes.create_string_buffer(256)
    buffer_ref = weakref.ref(buffer)
    context = publication._ReviewerAttributeInitializationContextV1(
        buffer=buffer,
        attribute_list=publication.ctypes.cast(buffer, publication.ctypes.c_void_p),
    )
    result_type = publication._bound_reviewer_attribute_initialization_result_type_v1(context)
    token = result_type(1)
    owner = publication._ReviewerAttributeListOwnerV1()
    owner.adopt(token)
    delete_calls = 0

    def recording_delete(_attribute_list: Any) -> None:
        nonlocal delete_calls
        delete_calls += 1
        assert buffer_ref() is not None

    monkeypatch.setattr(
        publication,
        "_delete_reviewer_attribute_list_noexcept_v1",
        recording_delete,
    )
    del buffer
    del context

    owner.close_noexcept()
    gc.collect()

    assert delete_calls == 1
    assert buffer_ref() is None
    assert token._context.buffer is None
    assert not token._context.attribute_list


@pytest.mark.skipif(os.name != "nt", reason="Windows STARTUPINFOEX contract")
@pytest.mark.parametrize("initialize_success", (False, True))
def test_attribute_initialization_native_return_token_crash_cut_deletes_only_success(
    initialize_success: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    size = publication.ctypes.c_size_t()
    publication._initialize_reviewer_attribute_list_call_v1(None, size)
    assert size.value > 1
    allocated = int(size.value if initialize_success else size.value - 1)
    buffer = publication.ctypes.create_string_buffer(allocated)
    attribute_list = publication.ctypes.cast(buffer, publication.ctypes.c_void_p)
    call_size = publication.ctypes.c_size_t(allocated)
    function = publication._initialize_reviewer_attribute_list_owned_call_v1
    target = _opcode_offset(
        function,
        opname="RETURN_VALUE",
        argval=None,
    )
    original_delete = publication._delete_reviewer_attribute_list_noexcept_v1
    deleted: list[int] = []

    def recording_delete(value: Any) -> None:
        deleted.append(int(value.value or 0))
        original_delete(value)

    monkeypatch.setattr(
        publication,
        "_delete_reviewer_attribute_list_noexcept_v1",
        recording_delete,
    )

    with pytest.raises(KeyboardInterrupt, match="synthetic single crash cut"):
        _run_single_instruction_crash_cut(
            function,
            offset=target,
            invoke=lambda: function(
                attribute_list=attribute_list,
                size=call_size,
                buffer=buffer,
            ),
        )
    gc.collect()

    assert len(deleted) == int(initialize_success)


@pytest.mark.skipif(os.name != "nt", reason="Windows STARTUPINFOEX contract")
def test_failed_attribute_initialization_return_crash_cut_never_deletes_invalid_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = publication._ReviewerAttributeListOwnerV1()
    delete_calls = 0

    def sizing(_attribute_list: Any, size: Any) -> bool:
        size.value = 256
        publication.ctypes.set_last_error(publication._ERROR_INSUFFICIENT_BUFFER)
        return False

    def failed_owned(
        *,
        attribute_list: Any,
        size: Any,
        buffer: Any,
    ) -> Any:
        context = publication._ReviewerAttributeInitializationContextV1(
            buffer=buffer,
            attribute_list=attribute_list,
        )
        result_type = publication._bound_reviewer_attribute_initialization_result_type_v1(context)
        return result_type(0)

    def forbidden_delete(_attribute_list: Any) -> None:
        nonlocal delete_calls
        delete_calls += 1

    monkeypatch.setattr(
        publication,
        "_initialize_reviewer_attribute_list_call_v1",
        sizing,
    )
    monkeypatch.setattr(
        publication,
        "_initialize_reviewer_attribute_list_owned_call_v1",
        failed_owned,
    )
    monkeypatch.setattr(
        publication,
        "_delete_reviewer_attribute_list_noexcept_v1",
        forbidden_delete,
    )
    function = publication._create_reviewer_attribute_list_v1
    target = _opcode_offset(
        function,
        opname="STORE_FAST",
        argval="initialization",
    )

    with pytest.raises(KeyboardInterrupt, match="synthetic single crash cut"):
        _run_single_instruction_crash_cut(
            function,
            offset=target,
            invoke=lambda: function(
                job_handle=1,
                inherited_handles=(2, 3, 4),
                owner=owner,
            ),
        )
    gc.collect()

    assert delete_calls == 0
    assert owner.delete_armed is False
    assert not owner.attribute_list
    assert owner.buffer is None


@pytest.mark.skipif(os.name != "nt", reason="Windows STARTUPINFOEX contract")
@pytest.mark.parametrize(
    ("cut_function_name", "cut_opname", "cut_argval"),
    (
        (
            "_create_reviewer_attribute_list_v1",
            "STORE_FAST",
            "initialization",
        ),
        ("_ReviewerAttributeListOwnerV1.adopt", "STORE_ATTR", "initialization"),
    ),
)
def test_successful_attribute_initialization_transfer_crash_cut_deletes_exactly_once(
    cut_function_name: str,
    cut_opname: str,
    cut_argval: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = publication._ReviewerAttributeListOwnerV1()
    original_delete = publication._delete_reviewer_attribute_list_noexcept_v1
    deleted: list[int] = []
    if cut_function_name.endswith(".adopt"):
        cut_function = publication._ReviewerAttributeListOwnerV1.adopt
    else:
        cut_function = publication._create_reviewer_attribute_list_v1
    target = _opcode_offset(
        cut_function,
        opname=cut_opname,
        argval=cut_argval,
    )

    def recording_delete(attribute_list: Any) -> None:
        deleted.append(int(attribute_list.value or 0))
        original_delete(attribute_list)

    monkeypatch.setattr(
        publication,
        "_delete_reviewer_attribute_list_noexcept_v1",
        recording_delete,
    )

    with pytest.raises(KeyboardInterrupt, match="synthetic single crash cut"):
        _run_single_instruction_crash_cut(
            cut_function,
            offset=target,
            invoke=lambda: publication._create_reviewer_attribute_list_v1(
                job_handle=1,
                inherited_handles=(2, 3, 4),
                owner=owner,
            ),
        )
    gc.collect()

    assert len(deleted) == 1
    assert deleted[0] != 0
    assert owner.delete_armed is False
    assert not owner.attribute_list
    assert owner.buffer is None


@pytest.mark.skipif(os.name != "nt", reason="Windows STARTUPINFOEX contract")
@pytest.mark.parametrize(
    ("failed_call", "message"),
    (
        (1, "sizing failed"),
        (2, "initialization failed"),
    ),
)
def test_failed_attribute_list_initialize_never_deletes_invalid_state(
    failed_call: int,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = publication._ReviewerAttributeListOwnerV1()
    initialize_calls = 0
    delete_calls = 0

    def fail_selected_initialize(
        _attribute_list: Any,
        size: Any,
    ) -> bool:
        nonlocal initialize_calls
        initialize_calls += 1
        if initialize_calls == 1 and failed_call == 2:
            size.value = 256
            publication.ctypes.set_last_error(publication._ERROR_INSUFFICIENT_BUFFER)
        return False

    def fail_owned_initialize(
        *,
        attribute_list: Any,
        size: Any,
        buffer: Any,
    ) -> Any:
        nonlocal initialize_calls
        initialize_calls += 1
        context = publication._ReviewerAttributeInitializationContextV1(
            buffer=buffer,
            attribute_list=attribute_list,
        )
        result_type = publication._bound_reviewer_attribute_initialization_result_type_v1(context)
        return result_type(0)

    def forbidden_delete(_attribute_list: Any) -> None:
        nonlocal delete_calls
        delete_calls += 1

    monkeypatch.setattr(
        publication,
        "_initialize_reviewer_attribute_list_call_v1",
        fail_selected_initialize,
    )
    monkeypatch.setattr(
        publication,
        "_initialize_reviewer_attribute_list_owned_call_v1",
        fail_owned_initialize,
    )
    monkeypatch.setattr(
        publication,
        "_delete_reviewer_attribute_list_noexcept_v1",
        forbidden_delete,
    )

    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match=message,
    ):
        publication._create_reviewer_attribute_list_v1(
            job_handle=1,
            inherited_handles=(2, 3, 4),
            owner=owner,
        )

    assert initialize_calls == failed_call
    assert delete_calls == 0
    assert owner.delete_armed is False
    assert not owner.attribute_list
    assert owner.buffer is None


@pytest.mark.skipif(os.name != "nt", reason="Windows STARTUPINFOEX contract")
@pytest.mark.parametrize(
    ("sizing_result", "sizing_error", "sizing_size"),
    (
        (True, publication._ERROR_INSUFFICIENT_BUFFER, 256),
        (False, 5, 256),
        (False, publication._ERROR_INSUFFICIENT_BUFFER, 0),
    ),
)
def test_attribute_list_sizing_requires_exact_false_insufficient_buffer_contract(
    sizing_result: bool,
    sizing_error: int,
    sizing_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = publication._ReviewerAttributeListOwnerV1()
    owned_initialize_calls = 0

    def invalid_sizing(_attribute_list: Any, size: Any) -> bool:
        size.value = sizing_size
        publication.ctypes.set_last_error(sizing_error)
        return sizing_result

    def forbidden_owned_initialize(**_kwargs: Any) -> Any:
        nonlocal owned_initialize_calls
        owned_initialize_calls += 1
        raise AssertionError("invalid sizing cannot reach second Initialize")

    monkeypatch.setattr(
        publication,
        "_initialize_reviewer_attribute_list_call_v1",
        invalid_sizing,
    )
    monkeypatch.setattr(
        publication,
        "_initialize_reviewer_attribute_list_owned_call_v1",
        forbidden_owned_initialize,
    )

    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="sizing failed exact contract",
    ):
        publication._create_reviewer_attribute_list_v1(
            job_handle=1,
            inherited_handles=(2, 3, 4),
            owner=owner,
        )

    assert owned_initialize_calls == 0
    assert owner.delete_armed is False
    assert not owner.attribute_list
    assert owner.buffer is None


@pytest.mark.skipif(os.name != "nt", reason="Windows STARTUPINFOEX contract")
def test_attribute_list_post_return_environment_failure_deletes_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, environment = _atomic_test_child("pass")
    raw_handles = publication._ReviewerRawLaunchHandleOwnerV1()
    publication._create_reviewer_kill_job_v1(owner=raw_handles)
    publication._create_reviewer_null_input_v1(owner=raw_handles)
    publication._create_reviewer_pipe_v1(owner=raw_handles, stream="stdout")
    publication._create_reviewer_pipe_v1(owner=raw_handles, stream="stderr")
    process_information = publication._ProcessInformation()
    owner = publication._ReviewerAttributeListOwnerV1()
    original_delete = publication._delete_reviewer_attribute_list_noexcept_v1
    deleted: list[int] = []

    def fail_environment(_environment: Any) -> Any:
        raise KeyboardInterrupt("synthetic post-attribute return interruption")

    def recording_delete(attribute_list: Any) -> None:
        deleted.append(int(attribute_list.value or 0))
        original_delete(attribute_list)

    monkeypatch.setattr(
        publication,
        "_canonical_windows_environment_buffer_v1",
        fail_environment,
    )
    monkeypatch.setattr(
        publication,
        "_delete_reviewer_attribute_list_noexcept_v1",
        recording_delete,
    )
    try:
        with pytest.raises(
            KeyboardInterrupt,
            match="post-attribute return interruption",
        ):
            publication._create_atomic_job_bound_process_v1(
                argv,
                cwd=tmp_path.resolve(),
                environment=environment,
                job_handle=raw_handles.job.value_int(),
                stdin_handle=raw_handles.stdin.value_int(),
                stdout_handle=raw_handles.stdout_write.value_int(),
                stderr_handle=raw_handles.stderr_write.value_int(),
                process_information=process_information,
                attribute_owner=owner,
            )
    finally:
        raw_handles.close_noexcept()

    assert len(deleted) == 1
    assert deleted[0] != 0
    assert owner.delete_armed is False
    assert not owner.attribute_list
    assert owner.buffer is None
    assert not process_information.hProcess
    assert not process_information.hThread


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_process_information_preowner_closes_handles_on_create_return_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = (tmp_path / "create-return-interrupt-escaped.txt").resolve()
    argv, environment = _atomic_test_child(
        f"from pathlib import Path;Path({str(sentinel)!r}).write_text('escaped', encoding='utf-8')"
    )
    original_create = publication._create_atomic_job_bound_process_v1
    original_close = publication._close_windows_handle_v1
    created: dict[str, Any] = {}
    closed_handles: list[int] = []

    def create_then_interrupt(*args: Any, **kwargs: Any) -> None:
        process_information = kwargs["process_information"]
        original_create(*args, **kwargs)
        process_id = int(process_information.dwProcessId)
        created.update(
            {
                "owner": process_information,
                "process_handle": int(process_information.hProcess),
                "thread_handle": int(process_information.hThread),
                "process_id": process_id,
                "process_created_at": (publication.psutil.Process(process_id).create_time()),
            }
        )
        raise KeyboardInterrupt("synthetic CreateProcessW return interruption")

    def recording_close(handle: int, *, role: str) -> None:
        closed_handles.append(handle)
        original_close(handle, role=role)

    monkeypatch.setattr(
        publication,
        "_create_atomic_job_bound_process_v1",
        create_then_interrupt,
    )
    monkeypatch.setattr(publication, "_close_windows_handle_v1", recording_close)

    with pytest.raises(KeyboardInterrupt, match="CreateProcessW return interruption"):
        _launch_retained_test_child(
            argv,
            cwd=tmp_path.resolve(),
            env=environment,
        )

    process_handle = int(created["process_handle"])
    thread_handle = int(created["thread_handle"])
    assert process_handle != thread_handle
    assert closed_handles.count(process_handle) == 1
    assert closed_handles.count(thread_handle) == 1
    assert created["owner"].hProcess is None
    assert created["owner"].hThread is None
    assert not sentinel.exists()
    _assert_process_instance_is_terminal(
        int(created["process_id"]),
        float(created["process_created_at"]),
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_atomic_job_list_installation_failure_never_calls_create_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = (tmp_path / "job-list-failure-escaped.txt").resolve()
    argv, environment = _atomic_test_child(
        f"from pathlib import Path;Path({str(sentinel)!r}).write_text('escaped', encoding='utf-8')"
    )

    def fail_attribute_list(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("synthetic JOB_LIST installation failure")

    monkeypatch.setattr(
        publication,
        "_create_reviewer_attribute_list_v1",
        fail_attribute_list,
    )

    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="synthetic JOB_LIST installation failure",
    ):
        _launch_retained_test_child(
            argv,
            cwd=tmp_path.resolve(),
            env=environment,
        )

    assert not sentinel.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_atomic_job_resume_failure_kills_suspended_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = (tmp_path / "resume-failure-escaped.txt").resolve()
    argv, environment = _atomic_test_child(
        f"from pathlib import Path;Path({str(sentinel)!r}).write_text('escaped', encoding='utf-8')"
    )
    child = _launch_retained_test_child(
        argv,
        cwd=tmp_path.resolve(),
        env=environment,
    )
    process_created_at = publication.psutil.Process(child.pid).create_time()

    def fail_resume(_thread_handle: int) -> None:
        raise RuntimeError("synthetic ResumeThread failure")

    monkeypatch.setattr(
        publication,
        "_resume_reviewer_initial_thread_v1",
        fail_resume,
    )

    with pytest.raises(RuntimeError, match="synthetic ResumeThread failure"):
        publication._capture_and_wait_for_reviewer_child_v1(
            child,
            Path(publication.__file__).resolve(),
        )

    assert child.custody_closed is True
    assert child.waited is True
    assert not sentinel.exists()
    _assert_process_instance_is_terminal(child.pid, process_created_at)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_atomic_handle_list_does_not_inherit_unlisted_parent_handle(
    tmp_path: Path,
) -> None:
    sentinel_handle = _create_inheritable_unnamed_windows_test_event()
    child_code = (
        "import ctypes,os;"
        "kernel32=ctypes.WinDLL('kernel32',use_last_error=True);"
        "set_event=kernel32.SetEvent;"
        "set_event.argtypes=[ctypes.c_void_p];"
        "set_event.restype=ctypes.c_int;"
        "result=set_event(ctypes.c_void_p(int(os.environ['AANCA_TEST_SENTINEL_HANDLE'])));"
        "print('SET_RESULT:'+str(int(result)))"
    )
    argv, environment = _atomic_test_child(child_code)
    environment["AANCA_TEST_SENTINEL_HANDLE"] = str(sentinel_handle)
    child: publication._RetainedReviewerChildV1 | None = None
    try:
        child = _launch_retained_test_child(
            argv,
            cwd=tmp_path.resolve(),
            env=environment,
        )
        child.resume_exactly_once()
        stdout, stderr = child.communicate()
        assert child.returncode == 0
        assert stdout.startswith(b"SET_RESULT:")
        assert stderr == b""
        child.close_after_wait()
        assert _wait_windows_test_handle(sentinel_handle, 0) == 258
    finally:
        if child is not None:
            child.close_job_then_wait_noexcept()
        publication._close_windows_handle_noexcept_v1(sentinel_handle)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_reviewer_cannot_break_away_or_spawn_a_second_process(
    tmp_path: Path,
) -> None:
    sentinel = (tmp_path / "breakaway-escaped.txt").resolve()
    grandchild_code = (
        f"from pathlib import Path;Path({str(sentinel)!r}).write_text('escaped', encoding='utf-8')"
    )
    base_executable = str(Path(publication.psutil.Process(os.getpid()).exe()).resolve())
    child_code = (
        "import os,subprocess,sys;"
        "grandchild=[os.environ['AANCA_TEST_BASE_EXE'],'-B','-c',"
        f"{grandchild_code!r}];"
        "\ntry:\n"
        " p=subprocess.Popen(grandchild,creationflags=0x01000000);"
        " p.wait();print('ESCAPED:'+str(p.returncode))\n"
        "except OSError as exc:\n"
        " print('BLOCKED:'+str(getattr(exc,'winerror',None)))\n"
    )
    argv, environment = _atomic_test_child(child_code)
    environment["AANCA_TEST_BASE_EXE"] = base_executable
    child = _launch_retained_test_child(
        argv,
        cwd=tmp_path.resolve(),
        env=environment,
    )
    try:
        child.resume_exactly_once()
        stdout, stderr = child.communicate()
        assert child.returncode == 0
        assert stdout.decode("utf-8").strip().startswith("BLOCKED:")
        assert stderr == b""
        assert not sentinel.exists()
        child.close_after_wait()
    finally:
        child.close_job_then_wait_noexcept()


def test_reviewer_identity_capture_failure_still_waits_without_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SyntheticChild:
        pid = 42_424
        communicate_calls = 0
        resume_calls = 0
        kill_wait_calls = 0

        def communicate(self) -> tuple[bytes, bytes]:
            self.communicate_calls += 1
            return b"", b""

        def resume_exactly_once(self) -> None:
            self.resume_calls += 1

        def close_after_wait(self) -> None:
            raise AssertionError("identity failure cannot reach normal custody close")

        def close_job_then_wait_noexcept(self) -> None:
            self.kill_wait_calls += 1

    child = SyntheticChild()

    def fail_identity(
        _process_id: int,
        _implementation_path: Path,
    ) -> dict[str, Any]:
        raise RuntimeError("synthetic identity capture failure")

    monkeypatch.setattr(publication, "_capture_process_identity_v1", fail_identity)

    with pytest.raises(RuntimeError, match="synthetic identity capture failure"):
        publication._capture_and_wait_for_reviewer_child_v1(
            child,  # type: ignore[arg-type]
            (tmp_path / "reviewer.py").resolve(),
        )

    assert child.resume_calls == 0
    assert child.communicate_calls == 0
    assert child.kill_wait_calls == 1


def test_control_leaf_is_create_new_read_only_and_never_adopted(tmp_path: Path) -> None:
    destination = (tmp_path / "receipt.json").resolve()
    payload = b'{"schema_version":1}\n'

    digest = publication.publish_canonical_control_leaf_create_new_v1(destination, payload)

    assert digest == publication._sha256_bytes(payload)
    assert destination.read_bytes() == payload
    assert publication._is_read_only(destination.stat(follow_symlinks=False))
    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="overwrite/adoption/retry are forbidden",
    ):
        publication.publish_canonical_control_leaf_create_new_v1(
            destination,
            b'{"replacement":true}\n',
        )
    assert destination.read_bytes() == payload


def test_live_intent_preverification_passes_actual_reviewer_process_to_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path)
    _write_json(Path(inputs["execution_source"]["manifest_path"]), inputs["source_inventory"])
    calls: list[dict[str, Any]] = []

    def fake_live_verify(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(schema, "_verify_live_bindings", fake_live_verify)

    verified = publication.verify_original_confirmatory_technical_intent_live_bindings_v1(
        intent=inputs["intent"],
        source_inventory=inputs["source_inventory"],
        project_root=tmp_path,
        reviewer_process=inputs["review"]["reviewer_process"],
    )

    assert verified == inputs["intent"]
    assert len(calls) == 1
    assert calls[0]["intent"] == inputs["intent"]
    assert calls[0]["review"] == {"reviewer_process": inputs["review"]["reviewer_process"]}
    assert calls[0]["source_inventory"] == inputs["source_inventory"]
    assert calls[0]["project_root"] == tmp_path.resolve()


def test_build_intent_cli_uses_actual_process_and_create_new_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path)
    request_directory = (
        tmp_path / "artifacts" / publication.AUTHORITY_REQUEST_DIRECTORY_NAME
    ).resolve()
    request_directory.mkdir(parents=True)
    source_inventory_path = Path(inputs["execution_source"]["manifest_path"])
    _write_json(source_inventory_path, inputs["source_inventory"])
    option_paths: dict[str, Path] = {}
    for name in (
        "parent",
        "frozen_science",
        "historical_primary",
        "execution_source",
        "execution_capsule",
        "capacity_v2",
        "outcome_scope",
    ):
        option_paths[name] = _write_json(tmp_path / f"{name}.json", inputs[name])
    output_path = request_directory / publication.INTENT_REQUEST_FILENAME
    live_calls = 0

    def forbidden_live(**_kwargs: Any) -> dict[str, Any]:
        nonlocal live_calls
        live_calls += 1
        raise AssertionError("intent builder must not run the full live verifier")

    monkeypatch.setattr(
        publication,
        "verify_original_confirmatory_technical_intent_live_bindings_v1",
        forbidden_live,
    )
    argv = [
        "build-intent",
        "--parent-json",
        str(option_paths["parent"]),
        "--frozen-science-json",
        str(option_paths["frozen_science"]),
        "--historical-primary-json",
        str(option_paths["historical_primary"]),
        "--execution-source-json",
        str(option_paths["execution_source"]),
        "--source-inventory-json",
        str(source_inventory_path),
        "--execution-capsule-json",
        str(option_paths["execution_capsule"]),
        "--capacity-v2-json",
        str(option_paths["capacity_v2"]),
        "--outcome-scope-json",
        str(option_paths["outcome_scope"]),
        "--project-root",
        str(tmp_path),
    ]

    result = CliRunner().invoke(
        publication.original_confirmatory_technical_authority_app,
        argv,
    )

    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)
    intent = schema.canonical_original_confirmatory_technical_authority_intent_v1(
        json.loads(output_path.read_text(encoding="utf-8"))
    )
    assert receipt["builder_process"]["process_id"] == os.getpid()
    assert (
        Path(intent["builder_process"]["implementation_path"])
        == Path(publication.__file__).resolve()
    )
    assert intent["downstream_bindings_included"] is False
    assert intent["automatic_retry_allowed"] is False
    assert publication._is_read_only(output_path.stat(follow_symlinks=False))
    assert live_calls == 0

    failed = CliRunner().invoke(
        publication.original_confirmatory_technical_authority_app,
        argv,
    )
    assert failed.exit_code == 1
    assert "pre-build request inventory is not exact" in failed.output
    assert live_calls == 0


def test_review_attempt_public_builder_has_no_caller_timestamp_parameter() -> None:
    parameters = inspect.signature(
        publication.build_original_confirmatory_technical_review_attempt_claim_v1
    ).parameters

    assert "attempt_created_at_utc" not in parameters


def test_two_concurrent_build_intents_create_exactly_one_fixed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path)
    (tmp_path / "artifacts").mkdir()
    source_inventory_path = _write_json(
        Path(inputs["execution_source"]["manifest_path"]),
        inputs["source_inventory"],
    )
    option_paths = {
        name: _write_json(tmp_path / f"{name}.json", inputs[name])
        for name in (
            "parent",
            "frozen_science",
            "historical_primary",
            "execution_source",
            "execution_capsule",
            "capacity_v2",
            "outcome_scope",
        )
    }
    real_publish = publication.publish_canonical_control_leaf_create_new_v1
    publication_barrier = threading.Barrier(2)

    def race_create_new(destination: Path, payload: bytes) -> str:
        publication_barrier.wait(timeout=10)
        return real_publish(destination, payload)

    monkeypatch.setattr(
        publication,
        "publish_canonical_control_leaf_create_new_v1",
        race_create_new,
    )
    successes: list[None] = []
    failures: list[BaseException] = []

    def invoke() -> None:
        try:
            publication.build_original_confirmatory_technical_authority_intent_v1_command(
                parent_json=option_paths["parent"],
                frozen_science_json=option_paths["frozen_science"],
                historical_primary_json=option_paths["historical_primary"],
                execution_source_json=option_paths["execution_source"],
                source_inventory_json=source_inventory_path,
                execution_capsule_json=option_paths["execution_capsule"],
                capacity_v2_json=option_paths["capacity_v2"],
                outcome_scope_json=option_paths["outcome_scope"],
                project_root=tmp_path,
            )
        except BaseException as exc:
            failures.append(exc)
        else:
            successes.append(None)

    threads = [threading.Thread(target=invoke) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    request_directory = tmp_path / "artifacts" / publication.AUTHORITY_REQUEST_DIRECTORY_NAME
    assert all(not thread.is_alive() for thread in threads)
    assert len(successes) == 1
    assert len(failures) == 1
    assert {path.name for path in request_directory.iterdir()} == {
        publication.INTENT_REQUEST_FILENAME
    }
    assert publication._is_read_only(
        (request_directory / publication.INTENT_REQUEST_FILENAME).stat(follow_symlinks=False)
    )


@pytest.mark.parametrize(
    "expected_names",
    [
        frozenset(),
        frozenset({publication.INTENT_REQUEST_FILENAME}),
        frozenset(
            {
                publication.INTENT_REQUEST_FILENAME,
                publication.REVIEW_ATTEMPT_FILENAME,
            }
        ),
        frozenset(
            {
                publication.INTENT_REQUEST_FILENAME,
                publication.REVIEW_ATTEMPT_FILENAME,
                publication.REVIEW_REQUEST_FILENAME,
            }
        ),
    ],
)
def test_every_request_phase_rejects_an_extra_file(
    tmp_path: Path,
    expected_names: frozenset[str],
) -> None:
    request_directory = tmp_path / "artifacts" / publication.AUTHORITY_REQUEST_DIRECTORY_NAME
    request_directory.mkdir(parents=True)
    for name in expected_names | {"unexpected.json"}:
        path = request_directory / name
        path.write_bytes(b"{}\n")
        path.chmod(stat.S_IREAD)
    anchor = publication._RetainedDirectoryAnchor.open(request_directory)
    try:
        with pytest.raises(
            publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
            match="request inventory is not exact",
        ):
            publication._require_authority_request_inventory(
                anchor,
                expected_names=expected_names,
                phase="synthetic-phase",
            )
    finally:
        anchor.close_noexcept()


@pytest.mark.skipif(os.name != "nt", reason="fresh-child review mutex is Windows-only")
def test_review_intent_cli_binds_receipt_to_the_spawned_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        publication,
        "_UTC_CLOCK_FOR_TESTS_ONLY",
        lambda: datetime(2026, 7, 31, 10, 0, 45, tzinfo=UTC),
    )
    inputs = _inputs(tmp_path)
    request_directory = (
        tmp_path / "artifacts" / publication.AUTHORITY_REQUEST_DIRECTORY_NAME
    ).resolve()
    request_directory.mkdir(parents=True)
    intent_path = _write_json(
        request_directory / publication.INTENT_REQUEST_FILENAME,
        inputs["intent"],
    )
    intent_path.chmod(stat.S_IREAD)
    _write_json(Path(inputs["execution_source"]["manifest_path"]), inputs["source_inventory"])
    output_path = request_directory / publication.REVIEW_REQUEST_FILENAME
    reviewer_spec = publication.importlib.util.find_spec(publication.REVIEWER_MODULE_NAME)
    assert reviewer_spec is not None and reviewer_spec.origin is not None
    reviewer_implementation = Path(reviewer_spec.origin).resolve()
    reviewer_implementation_sha256 = publication._sha256_bytes(reviewer_implementation.read_bytes())
    executable = Path(publication.psutil.Process(os.getpid()).exe()).resolve()
    executable_bytes = executable.read_bytes()
    child_pid = 909_090
    reviewer_identity = {
        "process_id": child_pid,
        "process_created_at_utc": "2026-07-31T10:00:30.000000Z",
        "executable_path": str(executable),
        "executable_size_bytes": len(executable_bytes),
        "executable_sha256": publication._sha256_bytes(executable_bytes),
        "implementation_path": str(reviewer_implementation),
        "implementation_sha256": reviewer_implementation_sha256,
    }
    controller_identity = {
        **reviewer_identity,
        "process_id": os.getpid(),
        "process_created_at_utc": "2026-07-31T09:59:00.000000Z",
        "implementation_path": str(Path(publication.__file__).resolve()),
        "implementation_sha256": publication._sha256_bytes(Path(publication.__file__).read_bytes()),
    }
    spawned: list[list[str]] = []
    blocked_intent_rename = False

    class FakeJobChild:
        pid = child_pid
        returncode: int | None = None
        resumed = False

        def resume_exactly_once(self) -> None:
            assert self.resumed is False
            self.resumed = True

        def communicate(self) -> tuple[bytes, bytes]:
            assert self.resumed is True
            nonlocal blocked_intent_rename
            try:
                os.rename(
                    intent_path,
                    intent_path.with_name("renamed.intent.json"),
                )
            except OSError:
                blocked_intent_rename = True
            review = schema.build_original_confirmatory_technical_authority_review_v1(
                intent=inputs["intent"],
                review_started_at_utc="2026-07-31T10:01:00.000000Z",
                review_completed_at_utc="2026-07-31T10:02:00.000000Z",
                reviewer_process=reviewer_identity,
            )
            publication.publish_canonical_control_leaf_create_new_v1(
                output_path,
                schema.canonical_json_line_bytes(review),
            )
            self.returncode = 0
            return b'{"decision":"passed"}\n', b""

        def close_after_wait(self) -> None:
            assert self.returncode == 0

        def close_job_then_wait_noexcept(self) -> None:
            raise AssertionError("successful reviewer must not take failure cleanup")

    def fake_launch(
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        child_owner: publication._RetainedReviewerChildOwnerV1,
    ) -> None:
        assert cwd == tmp_path.resolve()
        if sys.prefix != sys.base_prefix:
            assert env["__PYVENV_LAUNCHER__"] == sys.executable
        spawned.append(argv)
        child_owner.child = FakeJobChild()  # type: ignore[assignment]

    def fake_process_identity(process_id: int, implementation_path: Path) -> dict[str, Any]:
        if process_id == child_pid:
            assert Path(implementation_path) == reviewer_implementation
            return reviewer_identity
        assert process_id == os.getpid()
        return controller_identity

    monkeypatch.setattr(publication, "_capture_process_identity_v1", fake_process_identity)
    monkeypatch.setattr(publication, "_launch_retained_reviewer_child_v1", fake_launch)

    result = CliRunner().invoke(
        publication.original_confirmatory_technical_authority_app,
        [
            "review-intent",
            "--project-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)
    assert receipt["reviewer_process"] == reviewer_identity
    assert receipt["fresh_child_stdout_sha256"] == publication._sha256_bytes(
        b'{"decision":"passed"}\n'
    )
    assert spawned == [
        [
            str(executable),
            "-B",
            "-m",
            publication.REVIEWER_MODULE_NAME,
            "--intent-json",
            str(intent_path),
            "--output",
            str(output_path),
            "--project-root",
            str(tmp_path.resolve()),
        ]
    ]
    assert publication._is_read_only(output_path.stat(follow_symlinks=False))
    assert blocked_intent_rename is True

    failed = CliRunner().invoke(
        publication.original_confirmatory_technical_authority_app,
        [
            "review-intent",
            "--project-root",
            str(tmp_path),
        ],
    )
    assert failed.exit_code == 1
    assert "pre-review request inventory is not exact" in failed.output
    assert len(spawned) == 1


@pytest.mark.skipif(os.name != "nt", reason="fresh-child review mutex is Windows-only")
def test_two_concurrent_review_controllers_spawn_exactly_one_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        publication,
        "_UTC_CLOCK_FOR_TESTS_ONLY",
        lambda: datetime(2026, 7, 31, 10, 0, 45, tzinfo=UTC),
    )
    inputs = _inputs(tmp_path)
    request_directory = (
        tmp_path / "artifacts" / publication.AUTHORITY_REQUEST_DIRECTORY_NAME
    ).resolve()
    request_directory.mkdir(parents=True)
    intent_path = _write_json(
        request_directory / publication.INTENT_REQUEST_FILENAME,
        inputs["intent"],
    )
    intent_path.chmod(stat.S_IREAD)
    _write_json(
        Path(inputs["execution_source"]["manifest_path"]),
        inputs["source_inventory"],
    )
    output_path = request_directory / publication.REVIEW_REQUEST_FILENAME
    reviewer_spec = publication.importlib.util.find_spec(publication.REVIEWER_MODULE_NAME)
    assert reviewer_spec is not None and reviewer_spec.origin is not None
    reviewer_implementation = Path(reviewer_spec.origin).resolve()
    executable = Path(publication.psutil.Process(os.getpid()).exe()).resolve()
    executable_bytes = executable.read_bytes()
    reviewer_identity = {
        "process_id": 808_080,
        "process_created_at_utc": "2026-07-31T10:00:30.000000Z",
        "executable_path": str(executable),
        "executable_size_bytes": len(executable_bytes),
        "executable_sha256": publication._sha256_bytes(executable_bytes),
        "implementation_path": str(reviewer_implementation),
        "implementation_sha256": publication._sha256_bytes(reviewer_implementation.read_bytes()),
    }
    controller_identity = {
        **reviewer_identity,
        "process_id": os.getpid(),
        "process_created_at_utc": "2026-07-31T09:59:00.000000Z",
        "implementation_path": str(Path(publication.__file__).resolve()),
        "implementation_sha256": publication._sha256_bytes(Path(publication.__file__).read_bytes()),
    }
    child_entered = threading.Event()
    release_child = threading.Event()
    spawned = 0

    class SlowFakeJobChild:
        pid = reviewer_identity["process_id"]
        returncode: int | None = None
        resumed = False

        def resume_exactly_once(self) -> None:
            assert self.resumed is False
            self.resumed = True

        def communicate(self) -> tuple[bytes, bytes]:
            assert self.resumed is True
            assert release_child.wait(timeout=10)
            review = schema.build_original_confirmatory_technical_authority_review_v1(
                intent=inputs["intent"],
                review_started_at_utc="2026-07-31T10:01:00.000000Z",
                review_completed_at_utc="2026-07-31T10:02:00.000000Z",
                reviewer_process=reviewer_identity,
            )
            publication.publish_canonical_control_leaf_create_new_v1(
                output_path,
                schema.canonical_json_line_bytes(review),
            )
            self.returncode = 0
            return b"{}\n", b""

        def close_after_wait(self) -> None:
            assert self.returncode == 0

        def close_job_then_wait_noexcept(self) -> None:
            raise AssertionError("successful reviewer must not take failure cleanup")

    def fake_launch(
        _argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        child_owner: publication._RetainedReviewerChildOwnerV1,
    ) -> None:
        nonlocal spawned
        del cwd, env
        spawned += 1
        child_entered.set()
        child_owner.child = SlowFakeJobChild()  # type: ignore[assignment]

    def fake_process_identity(
        process_id: int,
        _implementation_path: Path,
    ) -> dict[str, Any]:
        return (
            reviewer_identity
            if process_id == reviewer_identity["process_id"]
            else controller_identity
        )

    monkeypatch.setattr(
        publication,
        "_capture_process_identity_v1",
        fake_process_identity,
    )
    monkeypatch.setattr(publication, "_launch_retained_reviewer_child_v1", fake_launch)
    successes: list[None] = []
    failures: list[BaseException] = []

    def invoke() -> None:
        try:
            publication.review_original_confirmatory_technical_authority_intent_v1_command(
                project_root=tmp_path,
            )
        except BaseException as exc:
            failures.append(exc)
        else:
            successes.append(None)

    first = threading.Thread(target=invoke)
    first.start()
    assert child_entered.wait(timeout=10)
    second = threading.Thread(target=invoke)
    second.start()
    second.join(timeout=10)
    release_child.set()
    first.join(timeout=20)

    assert not first.is_alive()
    assert not second.is_alive()
    assert spawned == 1
    assert len(successes) == 1
    assert len(failures) == 1
    assert {path.name for path in request_directory.iterdir()} == {
        publication.INTENT_REQUEST_FILENAME,
        publication.REVIEW_ATTEMPT_FILENAME,
        publication.REVIEW_REQUEST_FILENAME,
    }


@pytest.mark.skipif(os.name != "nt", reason="fresh-child review mutex is Windows-only")
def test_abandoned_review_attempt_after_parent_death_blocks_second_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path)
    request_directory = tmp_path / "artifacts" / publication.AUTHORITY_REQUEST_DIRECTORY_NAME
    request_directory.mkdir(parents=True)
    intent_path = _write_json(
        request_directory / publication.INTENT_REQUEST_FILENAME,
        inputs["intent"],
    )
    intent_path.chmod(stat.S_IREAD)
    _write_json(
        Path(inputs["execution_source"]["manifest_path"]),
        inputs["source_inventory"],
    )
    reviewer_spec = publication.importlib.util.find_spec(publication.REVIEWER_MODULE_NAME)
    assert reviewer_spec is not None and reviewer_spec.origin is not None
    controller_process = publication.capture_current_process_identity_v1(
        Path(publication.__file__).resolve()
    )
    attempt = publication._build_original_confirmatory_technical_review_attempt_claim_at_v1(
        intent=inputs["intent"],
        project_root=tmp_path,
        controller_process=controller_process,
        reviewer_implementation_path=Path(reviewer_spec.origin).resolve(),
        attempt_created_at_utc=_timestamp_after_process_creation(controller_process),
    )
    attempt_path = _write_json(
        request_directory / publication.REVIEW_ATTEMPT_FILENAME,
        attempt,
    )
    attempt_path.chmod(stat.S_IREAD)
    attempt_before = attempt_path.read_bytes()
    spawn_calls = 0

    def forbidden_launch(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal spawn_calls
        spawn_calls += 1
        raise AssertionError("abandoned permanent attempt must block before CreateProcessW")

    monkeypatch.setattr(
        publication,
        "_launch_retained_reviewer_child_v1",
        forbidden_launch,
    )

    result = CliRunner().invoke(
        publication.original_confirmatory_technical_authority_app,
        [
            "review-intent",
            "--project-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "pre-review request inventory is not exact" in result.output
    assert spawn_calls == 0
    assert attempt_path.read_bytes() == attempt_before
    assert not (request_directory / publication.REVIEW_REQUEST_FILENAME).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_real_hard_parent_death_kills_only_child_and_permanent_attempt_blocks_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path)
    request_directory = (
        tmp_path / "artifacts" / publication.AUTHORITY_REQUEST_DIRECTORY_NAME
    ).resolve()
    request_directory.mkdir(parents=True)
    intent_path = _write_json(
        request_directory / publication.INTENT_REQUEST_FILENAME,
        inputs["intent"],
    )
    intent_path.chmod(stat.S_IREAD)
    _write_json(
        Path(inputs["execution_source"]["manifest_path"]),
        inputs["source_inventory"],
    )

    module_directory = (tmp_path / "synthetic-reviewer-module").resolve()
    module_directory.mkdir()
    module_name = f"_aanca_slow_atomic_reviewer_{uuid.uuid4().hex}"
    reviewer_path = module_directory / f"{module_name}.py"
    reviewer_pid_path = (tmp_path / "synthetic-reviewer.pid").resolve()
    reviewer_start_log = (tmp_path / "synthetic-reviewer-starts.log").resolve()
    event_name = f"Local\\AANCA-T0-ready-{uuid.uuid4()}"
    reviewer_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import argparse",
                "import ctypes",
                "import os",
                "import time",
                "from pathlib import Path",
                "",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--intent-json', required=True)",
                "parser.add_argument('--output', required=True)",
                "parser.add_argument('--project-root', required=True)",
                "args = parser.parse_args()",
                "pid_path = Path(os.environ['AANCA_TEST_REVIEWER_PID'])",
                "pid_path.write_text(str(os.getpid()) + '\\n', encoding='ascii')",
                "log_fd = os.open(",
                "    os.environ['AANCA_TEST_REVIEWER_START_LOG'],",
                "    os.O_WRONLY | os.O_CREAT | os.O_APPEND,",
                "    0o600,",
                ")",
                "try:",
                "    os.write(log_fd, (str(os.getpid()) + '\\n').encode('ascii'))",
                "    os.fsync(log_fd)",
                "finally:",
                "    os.close(log_fd)",
                "kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)",
                "open_event = kernel32.OpenEventW",
                "open_event.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_wchar_p]",
                "open_event.restype = ctypes.c_void_p",
                "set_event = kernel32.SetEvent",
                "set_event.argtypes = [ctypes.c_void_p]",
                "set_event.restype = ctypes.c_int",
                "close_handle = kernel32.CloseHandle",
                "close_handle.argtypes = [ctypes.c_void_p]",
                "close_handle.restype = ctypes.c_int",
                "event_handle = int(",
                "    open_event(0x0002, 0, os.environ['AANCA_TEST_READY_EVENT']) or 0",
                ")",
                "if not event_handle:",
                "    raise ctypes.WinError(ctypes.get_last_error())",
                "try:",
                "    if not set_event(ctypes.c_void_p(event_handle)):",
                "        raise ctypes.WinError(ctypes.get_last_error())",
                "finally:",
                "    close_handle(ctypes.c_void_p(event_handle))",
                "time.sleep(60)",
                "output_fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)",
                "try:",
                "    os.write(output_fd, b'{}\\n')",
                "    os.fsync(output_fd)",
                "finally:",
                "    os.close(output_fd)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    driver_path = (tmp_path / "synthetic-review-controller.py").resolve()
    driver_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import sys",
                "from histo_audit.workflows import (",
                "    original_confirmatory_technical_authority_publication_v1 as publication,",
                ")",
                "publication.REVIEWER_MODULE_NAME = sys.argv[2]",
                "publication.review_original_confirmatory_technical_authority_intent_v1_command(",
                "    project_root=Path(sys.argv[1]),",
                ")",
                "",
            ]
        ),
        encoding="utf-8",
    )

    executable = str(Path(publication.psutil.Process(os.getpid()).exe()).resolve())
    environment = os.environ.copy()
    if sys.prefix != sys.base_prefix:
        environment["__PYVENV_LAUNCHER__"] = sys.executable
    source_root = str(Path(publication.__file__).resolve().parents[2])
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (
            str(module_directory),
            source_root,
            inherited_pythonpath,
        )
        if part
    )
    environment["AANCA_TEST_REVIEWER_PID"] = str(reviewer_pid_path)
    environment["AANCA_TEST_REVIEWER_START_LOG"] = str(reviewer_start_log)
    environment["AANCA_TEST_READY_EVENT"] = event_name
    driver_argv = [
        executable,
        "-B",
        str(driver_path),
        str(tmp_path.resolve()),
        module_name,
    ]

    ready_handle = _create_named_windows_test_event(event_name)
    child_wait_handle = 0
    driver = subprocess.Popen(
        driver_argv,
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        ready_result = _wait_windows_test_handle(ready_handle, 20_000)
        if ready_result != 0:
            try:
                stdout, stderr = driver.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                driver.kill()
                stdout, stderr = driver.communicate(timeout=10)
            pytest.fail(
                "synthetic reviewer did not signal readiness: "
                f"wait={ready_result}; stdout={stdout!r}; stderr={stderr!r}"
            )
        reviewer_pid = int(reviewer_pid_path.read_text(encoding="ascii").strip())
        process_created_at = publication.psutil.Process(reviewer_pid).create_time()
        child_wait_handle = _open_windows_process_wait_handle(reviewer_pid)

        driver.kill()
        driver_stdout, driver_stderr = driver.communicate(timeout=20)
        assert driver.returncode != 0
        assert driver_stdout == b""
        assert driver_stderr == b""
        assert _wait_windows_test_handle(child_wait_handle, 20_000) == 0
        _assert_process_instance_is_terminal(reviewer_pid, process_created_at)
    finally:
        if driver.poll() is None:
            driver.kill()
            driver.communicate(timeout=20)
        if child_wait_handle:
            if _wait_windows_test_handle(child_wait_handle, 0) != 0:
                publication._terminate_reviewer_process_noexcept_v1(child_wait_handle)
                _wait_windows_test_handle(child_wait_handle, 20_000)
            publication._close_windows_handle_noexcept_v1(child_wait_handle)
        publication._close_windows_handle_noexcept_v1(ready_handle)

    attempt_path = request_directory / publication.REVIEW_ATTEMPT_FILENAME
    review_path = request_directory / publication.REVIEW_REQUEST_FILENAME
    assert {path.name for path in request_directory.iterdir()} == {
        publication.INTENT_REQUEST_FILENAME,
        publication.REVIEW_ATTEMPT_FILENAME,
    }
    assert publication._is_read_only(attempt_path.stat(follow_symlinks=False))
    attempt_before_retry = attempt_path.read_bytes()
    assert not review_path.exists()
    assert reviewer_start_log.read_text(encoding="ascii").splitlines() == [str(reviewer_pid)]

    monkeypatch.syspath_prepend(str(module_directory))
    monkeypatch.setattr(publication, "REVIEWER_MODULE_NAME", module_name)
    verified_attempt = publication.verify_original_confirmatory_technical_review_attempt_claim_v1(
        attempt_path,
        intent=inputs["intent"],
        project_root=tmp_path,
    )
    assert verified_attempt["attempt_count"] == 1
    assert verified_attempt["max_attempt_count"] == 1
    assert verified_attempt["automatic_retry_allowed"] is False

    second = subprocess.run(
        driver_argv,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert second.returncode != 0
    assert b"pre-review request inventory is not exact" in second.stderr
    assert attempt_path.read_bytes() == attempt_before_retry
    assert not review_path.exists()
    assert reviewer_start_log.read_text(encoding="ascii").splitlines() == [str(reviewer_pid)]


def test_success_exact_inventory_and_live_schema_terminal_verifier(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    result = publication.publish_original_confirmatory_technical_authority_v1_once(bundle)

    destination = bundle.authority_directory
    assert {path.name for path in destination.iterdir()} == schema.QUALIFYING_FILENAMES
    assert not (destination / schema.STOP_FILENAME).exists()
    assert {path.name for path in destination.parent.iterdir()} == {
        publication.NAMESPACE_CLAIM_FILENAME,
        destination.name,
    }
    assert result.artifact_root_sha256 == bundle.artifact_root_sha256
    assert result.sha256_manifest_sha256 == bundle.sha256_manifest_sha256
    assert result.technical_authorization_sha256 == bundle.technical_authorization_sha256
    assert result.independent_verification_required is True
    assert all(
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_nlink == 1
        and not path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        for path in destination.iterdir()
    )

    verified = schema.verify_original_confirmatory_technical_authority_v1(
        destination,
        verify_live=False,
    )
    assert verified.artifact_root_sha256 == bundle.artifact_root_sha256
    assert verified.technical_authorization_sha256 == bundle.technical_authorization_sha256
    assert verified.publication_success_sha256 == result.publication_success_sha256
    assert (
        publication.verify_original_confirmatory_technical_authority_namespace_claim_v1(destination)
        == result.namespace_claim_sha256
    )


def test_forged_typed_bundle_is_rejected_before_directory_claim(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    forged = replace(
        bundle,
        publication_success_bytes=b'{"forged":true}\n',
    )

    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="differs byte-for-byte",
    ):
        publication.publish_original_confirmatory_technical_authority_v1_once(forged)

    assert not bundle.authority_directory.exists()


def test_non_typed_bundle_is_rejected_before_directory_claim(tmp_path: Path) -> None:
    destination = tmp_path / "authority"
    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="exact typed bundle",
    ):
        publication.publish_original_confirmatory_technical_authority_v1_once(  # type: ignore[arg-type]
            {"authority_directory": destination}
        )
    assert not destination.exists()


def test_destination_must_derive_from_exact_parent_p_project_root(
    tmp_path: Path,
) -> None:
    outside_namespace = (
        tmp_path / "other-project" / "artifacts" / publication.AUTHORITY_NAMESPACE_DIRECTORY_NAME
    )
    outside_namespace.mkdir(parents=True)
    bundle = _bundle(
        tmp_path,
        destination=outside_namespace / "authority",
    )

    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="direct child",
    ):
        publication.publish_original_confirmatory_technical_authority_v1_once(bundle)

    assert not (outside_namespace / publication.NAMESPACE_CLAIM_FILENAME).exists()
    assert not bundle.authority_directory.exists()


def test_existing_directory_is_never_adopted_or_modified(tmp_path: Path) -> None:
    destination = (
        tmp_path / "artifacts" / publication.AUTHORITY_NAMESPACE_DIRECTORY_NAME / "authority"
    )
    destination.parent.mkdir(parents=True)
    destination.mkdir()
    foreign = destination / "foreign.txt"
    foreign.write_bytes(b"retain")
    bundle = _bundle(tmp_path, destination=destination)

    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="adoption/retry forbidden",
    ):
        publication.publish_original_confirmatory_technical_authority_v1_once(bundle)

    assert foreign.read_bytes() == b"retain"
    assert {path.name for path in destination.iterdir()} == {"foreign.txt"}


def test_artifact_write_failure_creates_permanent_stop_without_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    real_create = publication._create_new_file_no_cleanup

    def fail_one(path: Path, payload: bytes, **kwargs: Any) -> Any:
        if path.name == schema.CAPSULE_BINDING_FILENAME:
            raise OSError("synthetic artifact crash")
        return real_create(path, payload, **kwargs)

    monkeypatch.setattr(publication, "_create_new_file_no_cleanup", fail_one)
    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="durable_child_stop_sha256=",
    ):
        publication.publish_original_confirmatory_technical_authority_v1_once(bundle)

    destination = bundle.authority_directory
    assert (destination / schema.ATTEMPT_FILENAME).is_file()
    assert (destination / schema.STOP_FILENAME).is_file()
    assert (destination.parent / publication.NAMESPACE_STOP_FILENAME).is_file()
    assert not (destination / schema.SUCCESS_FILENAME).exists()
    before = {path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()}
    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="adoption/retry forbidden",
    ):
        publication.publish_original_confirmatory_technical_authority_v1_once(bundle)
    assert before == {
        path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()
    }


def test_extra_preterminal_path_forces_stop(tmp_path: Path, monkeypatch: Any) -> None:
    bundle = _bundle(tmp_path)
    destination = bundle.authority_directory
    original_iterdir = Path.iterdir
    injected = False

    def inject(path: Path) -> Any:
        nonlocal injected
        if path == destination and not injected:
            injected = True
            (path / "foreign.txt").write_bytes(b"foreign")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", inject)
    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="preterminal T0 inventory is not exact",
    ):
        publication.publish_original_confirmatory_technical_authority_v1_once(bundle)

    assert (destination / "foreign.txt").read_bytes() == b"foreign"
    assert (destination / schema.STOP_FILENAME).is_file()
    assert not (destination / schema.SUCCESS_FILENAME).exists()


def test_retained_leaf_handles_deny_write_until_terminal_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    destination = bundle.authority_directory
    original_iterdir = Path.iterdir
    blocked = False

    def attempt_write(path: Path) -> Any:
        nonlocal blocked
        if path == destination and not blocked:
            try:
                descriptor = os.open(
                    destination / schema.INTENT_FILENAME,
                    os.O_WRONLY,
                )
            except OSError:
                blocked = True
            else:
                os.close(descriptor)
                raise AssertionError("retained leaf unexpectedly allowed write access")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", attempt_write)
    publication.publish_original_confirmatory_technical_authority_v1_once(bundle)

    assert blocked is True
    assert (destination / schema.SUCCESS_FILENAME).is_file()


def test_real_directory_flushes_cover_claim_attempt_preterminal_and_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    real_flush = publication._RetainedDirectoryAnchor.flush_and_assert
    flushed: list[Path] = []

    def record_flush(
        anchor: publication._RetainedDirectoryAnchor,
    ) -> None:
        flushed.append(anchor.path)
        real_flush(anchor)

    monkeypatch.setattr(
        publication._RetainedDirectoryAnchor,
        "flush_and_assert",
        record_flush,
    )
    publication.publish_original_confirmatory_technical_authority_v1_once(bundle)

    assert flushed.count(bundle.authority_directory.parent) >= 2
    assert flushed.count(bundle.authority_directory) >= 3


def test_success_write_is_last_fallible_publication(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    bundle = _bundle(tmp_path)
    destination = bundle.authority_directory
    original_iterdir = Path.iterdir
    created: list[str] = []
    real_create = publication._create_new_file_no_cleanup

    def record_create(path: Path, payload: bytes, **kwargs: Any) -> Any:
        created.append(path.name)
        return real_create(path, payload, **kwargs)

    def reject_post_success_iteration(path: Path) -> Any:
        if (path / schema.SUCCESS_FILENAME).exists():
            raise AssertionError("post-success directory enumeration is forbidden")
        return original_iterdir(path)

    monkeypatch.setattr(publication, "_create_new_file_no_cleanup", record_create)
    monkeypatch.setattr(Path, "iterdir", reject_post_success_iteration)
    result = publication.publish_original_confirmatory_technical_authority_v1_once(bundle)

    assert result.terminal_disposition == "success"
    assert created[-1] == schema.SUCCESS_FILENAME
    assert (destination / schema.SUCCESS_FILENAME).is_file()
    assert not (destination / schema.STOP_FILENAME).exists()


def test_success_write_failure_creates_stop(tmp_path: Path, monkeypatch: Any) -> None:
    bundle = _bundle(tmp_path)
    real_create = publication._create_new_file_no_cleanup

    def fail_success(path: Path, payload: bytes, **kwargs: Any) -> Any:
        if path.name == schema.SUCCESS_FILENAME:
            raise OSError("synthetic terminal success crash")
        return real_create(path, payload, **kwargs)

    monkeypatch.setattr(publication, "_create_new_file_no_cleanup", fail_success)
    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="durable_child_stop_sha256=",
    ):
        publication.publish_original_confirmatory_technical_authority_v1_once(bundle)

    assert not (bundle.authority_directory / schema.SUCCESS_FILENAME).exists()
    assert (bundle.authority_directory / schema.STOP_FILENAME).is_file()


def test_success_leaf_readback_failure_creates_explicit_stop(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    bundle = _bundle(tmp_path)
    real_assert = publication._RetainedPublishedFile.assert_current

    def fail_success_readback(
        retained: publication._RetainedPublishedFile,
    ) -> None:
        if retained.path.name == schema.SUCCESS_FILENAME:
            raise OSError("synthetic post-close success readback crash")
        real_assert(retained)

    monkeypatch.setattr(
        publication._RetainedPublishedFile,
        "assert_current",
        fail_success_readback,
    )
    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="durable_child_stop_sha256=",
    ):
        publication.publish_original_confirmatory_technical_authority_v1_once(bundle)

    assert (bundle.authority_directory / schema.SUCCESS_FILENAME).is_file()
    assert (bundle.authority_directory / schema.STOP_FILENAME).is_file()
    with pytest.raises(schema.OriginalConfirmatoryTechnicalAuthorityError):
        schema.verify_original_confirmatory_technical_authority_v1(
            bundle.authority_directory,
            verify_live=False,
        )


def test_failure_after_namespace_claim_handle_acquisition_writes_external_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    real_write = publication._write_all
    calls = 0

    def fail_first_write(descriptor: int, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic crash after claim CREATE_NEW")
        real_write(descriptor, payload)

    monkeypatch.setattr(publication, "_write_all", fail_first_write)
    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="durable_namespace_stop_sha256=",
    ):
        publication.publish_original_confirmatory_technical_authority_v1_once(bundle)

    namespace = bundle.authority_directory.parent
    assert (namespace / publication.NAMESPACE_CLAIM_FILENAME).is_file()
    assert (namespace / publication.NAMESPACE_STOP_FILENAME).is_file()
    assert not bundle.authority_directory.exists()


def test_namespace_flush_failure_after_claim_writes_external_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    namespace = bundle.authority_directory.parent
    real_flush = publication._RetainedDirectoryAnchor.flush_and_assert
    failed = False

    def fail_first_namespace_flush(
        anchor: publication._RetainedDirectoryAnchor,
    ) -> None:
        nonlocal failed
        if anchor.path == namespace and not failed:
            failed = True
            raise OSError("synthetic namespace FlushFileBuffers failure")
        real_flush(anchor)

    monkeypatch.setattr(
        publication._RetainedDirectoryAnchor,
        "flush_and_assert",
        fail_first_namespace_flush,
    )
    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="durable_namespace_stop_sha256=",
    ):
        publication.publish_original_confirmatory_technical_authority_v1_once(bundle)

    assert (namespace / publication.NAMESPACE_CLAIM_FILENAME).is_file()
    assert (namespace / publication.NAMESPACE_STOP_FILENAME).is_file()
    assert not bundle.authority_directory.exists()


def test_stop_write_failure_is_reported_without_cleanup(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    bundle = _bundle(tmp_path)
    real_create = publication._create_new_file_no_cleanup

    def fail_artifact_and_stop(path: Path, payload: bytes, **kwargs: Any) -> Any:
        if path.name in {
            schema.CAPSULE_BINDING_FILENAME,
            schema.STOP_FILENAME,
            publication.NAMESPACE_STOP_FILENAME,
        }:
            raise OSError(f"synthetic {path.name} crash")
        return real_create(path, payload, **kwargs)

    monkeypatch.setattr(
        publication,
        "_create_new_file_no_cleanup",
        fail_artifact_and_stop,
    )
    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="child_STOP_failed=OSError",
    ):
        publication.publish_original_confirmatory_technical_authority_v1_once(bundle)

    assert bundle.authority_directory.is_dir()
    assert (bundle.authority_directory / schema.ATTEMPT_FILENAME).is_file()
    assert not (bundle.authority_directory / schema.SUCCESS_FILENAME).exists()
    assert not (bundle.authority_directory / schema.STOP_FILENAME).exists()


def test_two_callers_cannot_merge_or_publish_twice(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    barrier = threading.Barrier(2)
    successes: list[Any] = []
    failures: list[BaseException] = []

    def caller() -> None:
        barrier.wait()
        try:
            successes.append(
                publication.publish_original_confirmatory_technical_authority_v1_once(bundle)
            )
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=caller), threading.Thread(target=caller)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(
        failures[0],
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
    )
    assert (bundle.authority_directory / schema.SUCCESS_FILENAME).is_file()
    assert not (bundle.authority_directory / schema.STOP_FILENAME).exists()


def test_two_different_destinations_share_one_global_singleton_claim(
    tmp_path: Path,
) -> None:
    namespace = tmp_path / "artifacts" / publication.AUTHORITY_NAMESPACE_DIRECTORY_NAME
    first = _bundle(tmp_path, destination=namespace / "authority-a")
    second = _bundle(tmp_path, destination=namespace / "authority-b")
    barrier = threading.Barrier(2)
    successes: list[Any] = []
    failures: list[BaseException] = []

    def caller(
        bundle: schema.OriginalConfirmatoryTechnicalAuthorityBundle,
    ) -> None:
        barrier.wait()
        try:
            successes.append(
                publication.publish_original_confirmatory_technical_authority_v1_once(bundle)
            )
        except BaseException as exc:
            failures.append(exc)

    threads = [
        threading.Thread(target=caller, args=(first,)),
        threading.Thread(target=caller, args=(second,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert len(successes) == 1
    assert len(failures) == 1
    winning_directory = successes[0].authority_directory
    assert {path.name for path in namespace.iterdir()} == {
        publication.NAMESPACE_CLAIM_FILENAME,
        winning_directory.name,
    }
    assert (
        sum(
            path.is_dir()
            for path in (
                first.authority_directory,
                second.authority_directory,
            )
        )
        == 1
    )


@pytest.mark.skipif(os.name != "nt", reason="retained rename-denying handles are Windows-only")
def test_publish_cli_does_not_run_terminal_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path)
    _write_json(
        Path(inputs["execution_source"]["manifest_path"]),
        inputs["source_inventory"],
    )
    request_paths = _install_request_chain(tmp_path, inputs)
    intent_path = request_paths["intent"]
    _install_frozen_snapshots(tmp_path)

    def forbidden_verify(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("publish process must not run terminal verifier")

    monkeypatch.setattr(
        schema,
        "verify_original_confirmatory_technical_authority_v1",
        forbidden_verify,
    )
    publication_timestamp = datetime.fromisoformat(
        inputs["publication_timestamp_utc"].replace("Z", "+00:00")
    )
    monkeypatch.setattr(
        publication,
        "_UTC_CLOCK_FOR_TESTS_ONLY",
        lambda: publication_timestamp,
    )
    real_publish = publication.publish_original_confirmatory_technical_authority_v1_once
    blocked_intent_rename = False

    def publish_while_inputs_are_retained(
        bundle: schema.OriginalConfirmatoryTechnicalAuthorityBundle,
    ) -> publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Result:
        nonlocal blocked_intent_rename
        try:
            os.rename(
                intent_path,
                intent_path.with_name("renamed.intent.json"),
            )
        except OSError:
            blocked_intent_rename = True
        return real_publish(bundle)

    monkeypatch.setattr(
        publication,
        "publish_original_confirmatory_technical_authority_v1_once",
        publish_while_inputs_are_retained,
    )
    namespace = tmp_path / "artifacts" / publication.AUTHORITY_NAMESPACE_DIRECTORY_NAME
    namespace.mkdir(parents=True)
    destination = namespace / "cli-authority"
    result = CliRunner().invoke(
        publication.original_confirmatory_technical_authority_app,
        [
            "publish",
            "--destination",
            str(destination),
            "--project-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    output = json.loads(result.output)
    assert output["terminal_disposition"] == "success"
    assert output["independent_verification_performed"] is False
    assert output["publication_timestamp_utc"] > inputs["review"]["review_completed_at_utc"]
    assert output["review_attempt_claim_sha256"] == publication._sha256_bytes(
        request_paths["attempt"].read_bytes()
    )
    assert (destination / schema.SUCCESS_FILENAME).is_file()
    assert blocked_intent_rename is True


def test_publish_cli_rejects_extra_request_namespace_file(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    _write_json(
        Path(inputs["execution_source"]["manifest_path"]),
        inputs["source_inventory"],
    )
    request_paths = _install_request_chain(tmp_path, inputs)
    request_directory = request_paths["intent"].parent
    extra = request_directory / "unexpected.json"
    extra.write_bytes(b"{}\n")
    extra.chmod(stat.S_IREAD)
    _install_frozen_snapshots(tmp_path)
    namespace = tmp_path / "artifacts" / publication.AUTHORITY_NAMESPACE_DIRECTORY_NAME
    namespace.mkdir(parents=True)

    result = CliRunner().invoke(
        publication.original_confirmatory_technical_authority_app,
        [
            "publish",
            "--destination",
            str(namespace / "must-not-exist"),
            "--project-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "pre-publication request inventory is not exact" in result.output
    assert tuple(namespace.iterdir()) == ()


def test_verify_cli_is_read_only_and_requests_live_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    publication.publish_original_confirmatory_technical_authority_v1_once(bundle)
    typed = schema.verify_original_confirmatory_technical_authority_v1(
        bundle.authority_directory,
        verify_live=False,
    )
    calls: list[tuple[Path, Path | None, bool]] = []

    def fake_verify(
        directory: str | Path,
        *,
        project_root: str | Path | None = None,
        verify_live: bool = True,
    ) -> schema.VerifiedOriginalConfirmatoryTechnicalAuthority:
        calls.append(
            (
                Path(directory),
                None if project_root is None else Path(project_root),
                verify_live,
            )
        )
        return typed

    monkeypatch.setattr(
        schema,
        "verify_original_confirmatory_technical_authority_v1",
        fake_verify,
    )
    before = {path.name: path.read_bytes() for path in bundle.authority_directory.iterdir()}
    result = CliRunner().invoke(
        publication.original_confirmatory_technical_authority_app,
        [
            "verify",
            "--authority-directory",
            str(bundle.authority_directory),
            "--project-root",
            str(PROJECT_ROOT),
        ],
    )
    after = {path.name: path.read_bytes() for path in bundle.authority_directory.iterdir()}

    assert result.exit_code == 0, result.output
    assert calls == [(bundle.authority_directory, PROJECT_ROOT, True)]
    assert before == after
    output = json.loads(result.output)
    assert output["decision"] == "passed"
    assert output["read_only"] is True
    assert output["review_attempt_claim_sha256"] == publication._sha256_bytes(
        (
            tmp_path
            / "artifacts"
            / publication.AUTHORITY_REQUEST_DIRECTORY_NAME
            / publication.REVIEW_ATTEMPT_FILENAME
        ).read_bytes()
    )


def test_cli_input_reader_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"{}\n")
    linked = tmp_path / "linked.json"
    try:
        linked.symlink_to(target)
    except OSError:
        pytest.skip("creating a test symlink is not permitted on this host")

    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="canonical regular non-link",
    ):
        publication._stable_regular_file_bytes(linked, role="synthetic input")


def test_namespace_claim_tamper_fails_combined_terminal_verifier(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    publication.publish_original_confirmatory_technical_authority_v1_once(bundle)
    combined = publication.verify_published_original_confirmatory_technical_authority_v1(
        bundle.authority_directory,
        verify_live=False,
    )
    assert combined.lifecycle_binding()["namespace_claim_sha256"] == combined.namespace_claim_sha256
    assert (
        combined.lifecycle_binding()["review_attempt_claim_sha256"]
        == combined.review_attempt_claim_sha256
    )

    claim = bundle.authority_directory.parent / publication.NAMESPACE_CLAIM_FILENAME
    claim.chmod(stat.S_IREAD | stat.S_IWRITE)
    claim.write_bytes(b'{"tampered":true}\n')
    with pytest.raises(publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error):
        publication.verify_published_original_confirmatory_technical_authority_v1(
            bundle.authority_directory,
            verify_live=False,
        )


def test_review_attempt_tamper_fails_combined_terminal_verifier(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    publication.publish_original_confirmatory_technical_authority_v1_once(bundle)
    attempt = (
        tmp_path
        / "artifacts"
        / publication.AUTHORITY_REQUEST_DIRECTORY_NAME
        / publication.REVIEW_ATTEMPT_FILENAME
    )
    tampered = json.loads(attempt.read_bytes())
    tampered["attempt_number"] = 2
    attempt.chmod(stat.S_IREAD | stat.S_IWRITE)
    attempt.write_bytes(schema.canonical_json_line_bytes(tampered))

    with pytest.raises(publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error):
        publication.verify_published_original_confirmatory_technical_authority_v1(
            bundle.authority_directory,
            verify_live=False,
        )


def test_combined_terminal_verifier_rejects_writable_authority_leaf(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    publication.publish_original_confirmatory_technical_authority_v1_once(bundle)
    leaf = bundle.authority_directory / schema.SUCCESS_FILENAME
    leaf.chmod(stat.S_IREAD | stat.S_IWRITE)

    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="retained CREATE_NEW handle differs",
    ):
        publication.verify_published_original_confirmatory_technical_authority_v1(
            bundle.authority_directory,
            verify_live=False,
        )


@pytest.mark.skipif(os.name != "nt", reason="retained rename-denying handles are Windows-only")
def test_combined_verifier_retains_directory_and_leaf_handles_across_live_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    publication.publish_original_confirmatory_technical_authority_v1_once(bundle)
    typed = schema.verify_original_confirmatory_technical_authority_v1(
        bundle.authority_directory,
        verify_live=False,
    )
    blocked_write = False
    blocked_rename = False

    def exercise_retained_handles(
        _directory: str | Path,
        *,
        project_root: str | Path | None = None,
        verify_live: bool = True,
    ) -> schema.VerifiedOriginalConfirmatoryTechnicalAuthority:
        nonlocal blocked_rename, blocked_write
        del project_root, verify_live
        try:
            descriptor = os.open(
                bundle.authority_directory / schema.SUCCESS_FILENAME,
                os.O_WRONLY,
            )
        except OSError:
            blocked_write = True
        else:
            os.close(descriptor)
        try:
            os.rename(
                bundle.authority_directory,
                bundle.authority_directory.with_name("renamed-authority"),
            )
        except OSError:
            blocked_rename = True
        return typed

    monkeypatch.setattr(
        schema,
        "verify_original_confirmatory_technical_authority_v1",
        exercise_retained_handles,
    )
    publication.verify_published_original_confirmatory_technical_authority_v1(
        bundle.authority_directory,
        verify_live=False,
    )

    assert blocked_write is True
    assert blocked_rename is True


def test_input_swap_between_lstat_and_open_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "input.json"
    replacement = tmp_path / "replacement.json"
    target.write_bytes(b'{"value":"first"}\n')
    replacement.write_bytes(b'{"value":"second"}\n')
    real_open = publication.os.open
    swapped = False

    def swap_then_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        nonlocal swapped
        if Path(path) == target and not swapped:
            swapped = True
            os.replace(replacement, target)
        return real_open(path, flags, mode)

    monkeypatch.setattr(publication.os, "open", swap_then_open)
    with pytest.raises(
        publication.OriginalConfirmatoryTechnicalAuthorityPublicationV1Error,
        match="changed during same-handle readback",
    ):
        publication._stable_regular_file_bytes(target, role="synthetic input")
