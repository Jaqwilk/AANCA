"""Security regressions for the replacement-v2 fresh-verifier boundary."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NoReturn

import psutil
import pytest

from histo_audit.workflows import (
    resource_authority_d_replacement_v2_controller as controller,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _windows_typer_json_bytes(payload: dict[str, Any]) -> bytes:
    text = json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2) + "\n"
    return text.replace("\n", "\r\n").encode("utf-8")


def _captured_stream(payload: bytes, *, limit: int) -> dict[str, Any]:
    record = controller._empty_stream_diagnostic(limit)
    record.update(
        {
            "capture_started": True,
            "captured_size_bytes": len(payload),
            "captured_sha256": _sha256(payload),
            "eof_observed": True,
            "reader_joined": True,
            "pipe_closed": True,
        }
    )
    return record


def _completed_cleanup(*, returncode_observed: bool) -> dict[str, Any]:
    cleanup = controller._empty_cleanup_diagnostic()
    cleanup["returncode_observed"] = returncode_observed
    if os.name == "nt":
        cleanup["containment"].update(
            {
                "created": True,
                "kill_on_close_configured": True,
                "child_created_suspended": True,
                "process_handle_proven": True,
                "assigned": True,
                "assignment_membership_proven": True,
                "thread_enumeration_succeeded": True,
                "owned_thread_count": 1,
                "thread_opened": True,
                "thread_resumed": True,
                "terminate_attempted": True,
                "terminate_succeeded": True,
                "job_empty_proven": True,
                "close_attempted": True,
                "close_succeeded": True,
                "complete": True,
            }
        )
    return cleanup


def _diagnostic(
    status: str,
    *,
    request: controller.VerifyRequestV2 | None = None,
    stdout_payload: bytes | None = None,
) -> dict[str, Any]:
    requested = str(Path(sys.executable).resolve(strict=True))
    override = controller._fresh_verifier_spawn_executable(requested)
    effective = requested if override is None else override
    if request is None:
        root = Path.cwd().resolve()
        request = controller.VerifyRequestV2(
            project_root=root,
            successor_directory=root / "synthetic-authority-d",
            parent_directory=root / "synthetic-authority-c",
            artifact_root_sha256="a" * 64,
            manifest_sha256="b" * 64,
            authorization_sha256="c" * 64,
            intent_sha256="d" * 64,
            nonce="e" * 64,
            python_executable=requested,
        ).checked()
    request_record = controller._fresh_request_record(request)
    controller_pid = os.getpid()
    argv = [] if status == "not_invoked" else list(request.argv(controller_pid))
    payload: dict[str, Any] = {
        "schema_version": controller.FRESH_DIAGNOSTIC_SCHEMA_VERSION,
        "policy": controller.FRESH_DIAGNOSTIC_POLICY,
        "status": status,
        "failure_phase": "not_started",
        "requested_python_executable": requested,
        "effective_spawn_executable": effective,
        "executable_override_used": override is not None,
        "request": request_record,
        "request_sha256": controller._compact_sha256(request_record),
        "argv_sha256": controller._compact_sha256(argv),
        "controller_process_id": controller_pid,
        "verifier_process_id": None,
        "returncode": None,
        "timeout_milliseconds": 1_000,
        "timed_out": False,
        "stdout": controller._empty_stream_diagnostic(controller._MAX_STDOUT_BYTES),
        "stderr": controller._empty_stream_diagnostic(controller._MAX_STDERR_BYTES),
        "cleanup": controller._empty_cleanup_diagnostic(),
        "payload_sha256": None,
        "payload_validation_completed": False,
        "stdout_content_included": False,
        "stderr_content_included": False,
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }
    if status == "spawn_failed":
        payload["failure_phase"] = "spawn"
    elif status in {"failed", "passed"}:
        stdout = b'{"status":"verified"}' if stdout_payload is None else stdout_payload
        payload.update(
            {
                "failure_phase": ("payload_validation" if status == "failed" else "completed"),
                "verifier_process_id": os.getpid() + 100_000,
                "returncode": 0,
                "stdout": _captured_stream(
                    stdout,
                    limit=controller._MAX_STDOUT_BYTES,
                ),
                "stderr": _captured_stream(
                    b"",
                    limit=controller._MAX_STDERR_BYTES,
                ),
                "cleanup": _completed_cleanup(returncode_observed=True),
                "payload_sha256": _sha256(stdout),
                "payload_validation_completed": status == "passed",
            }
        )
    return payload


@pytest.mark.parametrize(
    "status",
    ("not_invoked", "spawn_failed", "failed", "passed"),
)
def test_fresh_diagnostic_accepts_each_exact_status_contract(status: str) -> None:
    payload = _diagnostic(status)

    assert controller._canonical_fresh_diagnostic(payload) == payload


@pytest.mark.parametrize(
    ("status", "path", "replacement"),
    (
        ("not_invoked", ("failure_phase",), "completed"),
        ("not_invoked", ("verifier_process_id",), 7),
        ("not_invoked", ("timed_out",), True),
        ("not_invoked", ("payload_validation_completed",), True),
        ("not_invoked", ("payload_sha256",), "a" * 64),
        ("not_invoked", ("stdout", "capture_started"), True),
        ("spawn_failed", ("failure_phase",), "not_started"),
        ("spawn_failed", ("verifier_process_id",), 7),
        ("spawn_failed", ("timed_out",), True),
        ("spawn_failed", ("payload_sha256",), "a" * 64),
        ("spawn_failed", ("stdout", "capture_started"), True),
        ("failed", ("failure_phase",), "completed"),
        ("failed", ("failure_phase",), "spawn"),
        ("failed", ("verifier_process_id",), None),
        ("failed", ("payload_validation_completed",), True),
        ("failed", ("timed_out",), True),
        ("failed", ("payload_sha256",), None),
        ("failed", ("payload_sha256",), "f" * 64),
        ("passed", ("failure_phase",), "payload_validation"),
        ("passed", ("verifier_process_id",), None),
        ("passed", ("verifier_process_id",), os.getpid()),
        ("passed", ("returncode",), 1),
        ("passed", ("timed_out",), True),
        ("passed", ("payload_validation_completed",), False),
        ("passed", ("payload_sha256",), None),
        ("passed", ("payload_sha256",), "f" * 64),
        ("passed", ("stdout", "capture_started"), False),
        ("passed", ("stderr", "captured_size_bytes"), 1),
    ),
)
def test_fresh_diagnostic_rejects_impossible_status_combinations(
    status: str,
    path: tuple[str, ...],
    replacement: object,
) -> None:
    payload = copy.deepcopy(_diagnostic(status))
    target: dict[str, Any] = payload
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = replacement

    with pytest.raises(controller.ControlError):
        controller._canonical_fresh_diagnostic(payload)


class _AlreadyReapedProcess:
    returncode = 0


def test_cleanup_without_process_tree_proof_is_fail_closed() -> None:
    diagnostic = controller._cleanup_verifier_process(
        _AlreadyReapedProcess(),
        (),
        force=False,
        tree=None,
    )

    assert diagnostic["descendant_quiescence_proven"] is False
    assert diagnostic["complete"] is False
    assert "process_tree_probe_failed" in diagnostic["error_codes"]


def _is_alive(process: psutil.Process | None) -> bool:
    if process is None:
        return False
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _kill_owned_processes(processes: list[psutil.Process]) -> None:
    unique = {process.pid: process for process in processes}
    for process in tuple(unique.values()):
        try:
            for descendant in process.children(recursive=True):
                unique.setdefault(descendant.pid, descendant)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    for process in reversed(tuple(unique.values())):
        try:
            if process.is_running():
                process.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    psutil.wait_procs(list(unique.values()), timeout=5.0)


@pytest.mark.skipif(os.name != "nt", reason="Windows process-tree regression")
def test_windows_timed_out_verifier_reaps_sleeping_grandchild(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "sleeping-grandchild.pid"
    child_code = """
import subprocess
import sys
import time
from pathlib import Path

grandchild = subprocess.Popen(
    [sys.executable, "-I", "-B", "-c", "import time; time.sleep(120)"],
    executable=sys._base_executable,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    close_fds=True,
)
Path(sys.argv[1]).write_text(str(grandchild.pid), encoding="ascii")
time.sleep(120)
"""
    requested = str(Path(sys.executable).resolve(strict=True))
    spawn_executable = controller._fresh_verifier_spawn_executable(requested)
    popen_arguments: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if spawn_executable is not None:
        popen_arguments["executable"] = spawn_executable
    verifier = subprocess.Popen(
        [requested, "-I", "-B", "-c", child_code, str(pid_path)],
        **popen_arguments,
    )
    owned: list[psutil.Process] = [psutil.Process(verifier.pid)]
    grandchild: psutil.Process | None = None
    try:
        deadline = time.monotonic() + 15.0
        while not pid_path.is_file() and time.monotonic() < deadline:
            if verifier.poll() is not None:
                pytest.fail("synthetic verifier exited before publishing its child PID")
            time.sleep(0.05)
        assert pid_path.is_file()
        grandchild_pid = int(pid_path.read_text(encoding="ascii"))
        grandchild = psutil.Process(grandchild_pid)
        owned.append(grandchild)

        tree = controller._ProcessTreeTracker(verifier.pid)
        while grandchild_pid not in tree.known and time.monotonic() < deadline:
            tree.refresh()
            time.sleep(0.05)
        assert grandchild_pid in tree.known

        diagnostic = controller._cleanup_verifier_process(
            verifier,
            (),
            force=True,
            tree=tree,
        )

        assert diagnostic["descendant_quiescence_proven"] is True
        assert diagnostic["descendants_reaped"] is True
        assert diagnostic["complete"] is True
        assert diagnostic["error_codes"] == []
        assert not _is_alive(owned[0])
        assert not _is_alive(grandchild)
    finally:
        _kill_owned_processes(owned)
        if verifier.poll() is None:
            verifier.kill()
            verifier.wait(timeout=5)


@pytest.mark.skipif(os.name != "nt", reason="Windows process-tree regression")
def test_windows_runner_monitors_and_reaps_late_descendant_during_wait(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "late-grandchild.pid"
    child_code = """
import subprocess
import sys
import time
from pathlib import Path

time.sleep(0.25)
grandchild = subprocess.Popen(
    [sys.executable, "-I", "-B", "-c", "import time; time.sleep(120)"],
    executable=sys._base_executable,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    close_fds=True,
)
Path(sys.argv[1]).write_text(str(grandchild.pid), encoding="ascii")
time.sleep(120)
"""
    requested = str(Path(sys.executable).resolve(strict=True))
    spawn_executable = controller._fresh_verifier_spawn_executable(requested)
    owned: list[psutil.Process] = []
    verifier: subprocess.Popen[bytes] | None = None

    def popen_factory(_argv: list[str], **_kwargs: Any) -> subprocess.Popen[bytes]:
        nonlocal verifier
        popen_arguments: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "close_fds": True,
            "creationflags": _kwargs.get("creationflags", 0),
        }
        if spawn_executable is not None:
            popen_arguments["executable"] = spawn_executable
        verifier = subprocess.Popen(
            [requested, "-I", "-B", "-c", child_code, str(pid_path)],
            **popen_arguments,
        )
        owned.append(psutil.Process(verifier.pid))
        return verifier

    try:
        with pytest.raises(controller.FreshVerifierError) as raised:
            controller.run_fresh_verifier_v2(
                _verify_request(tmp_path),
                timeout_seconds=10.0,
                popen_factory=popen_factory,
            )
        diagnostic = raised.value.diagnostic
        assert diagnostic is not None
        assert diagnostic["status"] == "failed"
        assert diagnostic["failure_phase"] == "wait"
        assert diagnostic["timed_out"] is False
        assert diagnostic["cleanup"]["descendant_quiescence_proven"] is True
        assert diagnostic["cleanup"]["descendants_reaped"] is True
        assert diagnostic["cleanup"]["complete"] is True

        assert pid_path.is_file()
        grandchild_pid = int(pid_path.read_text(encoding="ascii"))
        try:
            grandchild = psutil.Process(grandchild_pid)
        except psutil.NoSuchProcess:
            grandchild = None
        if grandchild is not None:
            owned.append(grandchild)
        assert all(not _is_alive(process) for process in owned)
    finally:
        _kill_owned_processes(owned)
        if verifier is not None and verifier.poll() is None:
            verifier.kill()
            verifier.wait(timeout=5)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object regression")
def test_windows_runner_invokes_actual_typer_verifier_child_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(controller.FreshVerifierError) as raised:
        controller.run_fresh_verifier_v2(
            _verify_request(tmp_path),
            timeout_seconds=30.0,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic is not None
    assert diagnostic["status"] == "failed"
    assert diagnostic["failure_phase"] == "returncode"
    assert diagnostic["verifier_process_id"] != os.getpid()
    assert diagnostic["returncode"] != 0
    assert diagnostic["stdout"]["captured_size_bytes"] == 0
    assert diagnostic["stderr"]["captured_size_bytes"] > 0
    assert diagnostic["cleanup"]["complete"] is True
    containment = diagnostic["cleanup"]["containment"]
    assert containment["required"] is True
    assert containment["created"] is True
    assert containment["kill_on_close_configured"] is True
    assert containment["child_created_suspended"] is True
    assert containment["assignment_membership_proven"] is True
    assert containment["thread_resumed"] is True
    assert containment["terminate_succeeded"] is True
    assert containment["job_empty_proven"] is True
    assert containment["close_succeeded"] is True
    assert containment["complete"] is True
    assert tuple(tmp_path.iterdir()) == ()


def _verify_request(tmp_path: Path) -> controller.VerifyRequestV2:
    return controller.VerifyRequestV2(
        project_root=tmp_path,
        successor_directory=tmp_path / "authority-d",
        parent_directory=tmp_path / "authority-c",
        artifact_root_sha256="a" * 64,
        manifest_sha256="b" * 64,
        authorization_sha256="c" * 64,
        intent_sha256="d" * 64,
        nonce="e" * 64,
        python_executable=sys.executable,
    )


def _fresh_payload(
    request: controller.VerifyRequestV2,
    *,
    controller_pid: int,
    verifier_pid: int,
) -> dict[str, Any]:
    return {
        "status": "verified",
        "verification_schema_version": 2,
        "verification_kind": "resource_bounded_technical_successor_fresh_process",
        "process_boundary": {
            "controller_process_id": controller_pid,
            "verifier_process_id": verifier_pid,
            "verifier_parent_process_id": controller_pid,
            "distinct_processes": True,
            "direct_child_process": True,
            "verification_nonce": request.nonce,
        },
        "successor_authority": {
            "directory": str(request.successor_directory),
            "schema_version": 5,
            "purpose": controller._TECHNICAL_SUCCESSOR_PURPOSE,
            "chain_depth": 4,
            "artifact_root_sha256": request.artifact_root_sha256,
            "sha256_manifest_sha256": request.manifest_sha256,
            "authorization_sha256": request.authorization_sha256,
            "intent_sha256": request.intent_sha256,
        },
        "superseded_authority": {
            "directory": str(request.parent_directory),
            "schema_version": 4,
            "historically_verified": True,
            "effective_execution_leaf": False,
        },
        "bundle": {
            "flat_file_count": 8,
            "manifest_artifact_count": 6,
            "flat_file_inventory_sha256": "f" * 64,
            "flat_file_hashes_verified": True,
        },
        "confirmatory_storage_policy_sha256": "1" * 64,
        "successor_candidate_count": 1,
        "checks": {field: True for field in controller._FRESH_CHECK_FIELDS},
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }


def test_fresh_payload_rejects_post_outcome_purpose_alias(tmp_path: Path) -> None:
    request = _verify_request(tmp_path)
    controller_pid = os.getpid()
    verifier_pid = controller_pid + 100_000
    payload = _fresh_payload(
        request,
        controller_pid=controller_pid,
        verifier_pid=verifier_pid,
    )
    assert (
        controller._verified_fresh_payload(
            payload,
            request=request,
            controller_pid=controller_pid,
            child_pid=verifier_pid,
        )
        == payload
    )
    payload["successor_authority"]["purpose"] = (
        "post_outcome_resource_bounded_confirmatory_technical_successor"
    )

    with pytest.raises(controller.FreshVerifierError):
        controller._verified_fresh_payload(
            payload,
            request=request,
            controller_pid=controller_pid,
            child_pid=verifier_pid,
        )


def _terminal_attempt(tmp_path: Path) -> tuple[dict[str, Any], str]:
    claimed_at = datetime.now(UTC) - timedelta(minutes=1)
    amendment_root = tmp_path / "artifacts" / "preregistration_amendments"
    return (
        {
            "claimed_at_utc": controller._timestamp(claimed_at),
            "attempt_id": "2" * 64,
            "intended_authority_directory": str((amendment_root / "authority-d").resolve()),
            "parent_authority_directory": str((amendment_root / "authority-c").resolve()),
            "technical_authorization_sha256": "3" * 64,
            "intent_sha256": "4" * 64,
            "verification_nonce": "9" * 64,
            "run_state": {"sha256": "5" * 64},
        },
        "6" * 64,
    )


def _success_marker(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    attempt, attempt_sha256 = _terminal_attempt(tmp_path)
    request = controller.VerifyRequestV2(
        project_root=tmp_path.resolve(),
        successor_directory=Path(attempt["intended_authority_directory"]),
        parent_directory=Path(attempt["parent_authority_directory"]),
        artifact_root_sha256="7" * 64,
        manifest_sha256="8" * 64,
        authorization_sha256=attempt["technical_authorization_sha256"],
        intent_sha256=attempt["intent_sha256"],
        nonce=attempt["verification_nonce"],
        python_executable=sys.executable,
    ).checked()
    controller_pid = os.getpid()
    verifier_pid = controller_pid + 100_000
    validated_payload = _fresh_payload(
        request,
        controller_pid=controller_pid,
        verifier_pid=verifier_pid,
    )
    diagnostic = _diagnostic(
        "passed",
        request=request,
        stdout_payload=_windows_typer_json_bytes(validated_payload),
    )
    payload = {
        "schema_version": 2,
        "policy": controller.SUCCESS_V2_POLICY,
        "status": "committed",
        "committed_at_utc": controller._timestamp(datetime.now(UTC)),
        "automatic_retry_allowed": False,
        "attempt_id": attempt["attempt_id"],
        "attempt_marker_sha256": attempt_sha256,
        "authority_directory": attempt["intended_authority_directory"],
        "parent_authority_directory": attempt["parent_authority_directory"],
        "artifact_root_sha256": "7" * 64,
        "sha256_manifest_sha256": "8" * 64,
        "authorization_sha256": attempt["technical_authorization_sha256"],
        "intent_sha256": attempt["intent_sha256"],
        "verification_nonce": attempt["verification_nonce"],
        "fresh_verifier_validated_payload": validated_payload,
        "fresh_verifier_validated_payload_sha256": hashlib.sha256(
            controller._canonical_bytes(validated_payload)
        ).hexdigest(),
        "fresh_verifier_payload_sha256": diagnostic["payload_sha256"],
        "fresh_verifier_diagnostic": diagnostic,
        "fresh_verifier_diagnostic_sha256": controller._compact_sha256(diagnostic),
        "controller_process_id": diagnostic["controller_process_id"],
        "verifier_process_id": diagnostic["verifier_process_id"],
        "verifier_parent_process_id": diagnostic["controller_process_id"],
        "chain_depth": 4,
        "run_state_sha256": attempt["run_state"]["sha256"],
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": True,
    }
    return payload, attempt, attempt_sha256


def _failure_marker(
    tmp_path: Path,
    *,
    failure_phase: str,
    diagnostic_status: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    attempt, attempt_sha256 = _terminal_attempt(tmp_path)
    request = controller.VerifyRequestV2(
        project_root=tmp_path.resolve(),
        successor_directory=Path(attempt["intended_authority_directory"]),
        parent_directory=Path(attempt["parent_authority_directory"]),
        artifact_root_sha256=("0" * 64 if diagnostic_status == "not_invoked" else "c" * 64),
        manifest_sha256=("0" * 64 if diagnostic_status == "not_invoked" else "d" * 64),
        authorization_sha256=attempt["technical_authorization_sha256"],
        intent_sha256=attempt["intent_sha256"],
        nonce=attempt["verification_nonce"],
        python_executable=sys.executable,
    ).checked()
    validated_payload: dict[str, Any] | None = None
    stdout_payload: bytes | None = None
    if diagnostic_status == "passed":
        controller_pid = os.getpid()
        verifier_pid = controller_pid + 100_000
        validated_payload = _fresh_payload(
            request,
            controller_pid=controller_pid,
            verifier_pid=verifier_pid,
        )
        stdout_payload = _windows_typer_json_bytes(validated_payload)
    diagnostic = _diagnostic(
        diagnostic_status,
        request=request,
        stdout_payload=stdout_payload,
    )
    failed_at = datetime.now(UTC) - timedelta(seconds=1)
    payload = {
        "schema_version": 2,
        "policy": controller.FAILURE_V2_POLICY,
        "status": "rolled_back_failure_no_retry",
        "failed_at_utc": controller._timestamp(failed_at),
        "failure_phase": failure_phase,
        "automatic_retry_allowed": False,
        "attempt_id": attempt["attempt_id"],
        "attempt_marker_sha256": attempt_sha256,
        "intended_authority_directory": attempt["intended_authority_directory"],
        "parent_authority_directory": attempt["parent_authority_directory"],
        "error_type_sha256": "a" * 64,
        "error_sha256": "b" * 64,
        "fresh_verifier_validated_payload": validated_payload,
        "fresh_verifier_validated_payload_sha256": (
            None
            if validated_payload is None
            else hashlib.sha256(controller._canonical_bytes(validated_payload)).hexdigest()
        ),
        "fresh_verifier_diagnostic": diagnostic,
        "fresh_verifier_diagnostic_sha256": controller._compact_sha256(diagnostic),
        "rollback_checked_at_utc": controller._timestamp(datetime.now(UTC)),
        "rollback_scan_count": 2,
        "candidate_directories_after_rollback": [],
        "authority_absent_after_rollback": True,
        "run_state_sha256": attempt["run_state"]["sha256"],
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }
    return payload, attempt, attempt_sha256


def test_success_marker_binds_exact_passed_diagnostic_and_payload_hash(
    tmp_path: Path,
) -> None:
    payload, attempt, attempt_sha256 = _success_marker(tmp_path)
    assert (
        controller._canonical_success_v2(
            payload,
            attempt=attempt,
            attempt_sha256=attempt_sha256,
        )
        == payload
    )

    failed = _diagnostic("failed")
    payload["fresh_verifier_diagnostic"] = failed
    payload["fresh_verifier_diagnostic_sha256"] = controller._compact_sha256(failed)
    payload["fresh_verifier_payload_sha256"] = failed["payload_sha256"]
    payload["controller_process_id"] = failed["controller_process_id"]
    payload["verifier_process_id"] = failed["verifier_process_id"]
    payload["verifier_parent_process_id"] = failed["controller_process_id"]

    with pytest.raises(controller.ControlError):
        controller._canonical_success_v2(
            payload,
            attempt=attempt,
            attempt_sha256=attempt_sha256,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("authority_directory", "other-authority-d"),
        ("parent_authority_directory", "other-authority-c"),
        ("artifact_root_sha256", "0" * 64),
        ("sha256_manifest_sha256", "1" * 64),
        ("authorization_sha256", "2" * 64),
        ("intent_sha256", "3" * 64),
        ("verification_nonce", "4" * 64),
        ("chain_depth", 3),
    ),
)
def test_success_marker_rejects_request_pin_tampering(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    payload, attempt, attempt_sha256 = _success_marker(tmp_path)
    if field.endswith("_directory"):
        replacement = str((tmp_path / str(replacement)).resolve())
    payload[field] = replacement

    with pytest.raises(controller.ControlError):
        controller._canonical_success_v2(
            payload,
            attempt=attempt,
            attempt_sha256=attempt_sha256,
        )


def test_success_marker_rejects_valid_diagnostic_from_other_request(
    tmp_path: Path,
) -> None:
    payload, attempt, attempt_sha256 = _success_marker(tmp_path)
    other = controller.VerifyRequestV2(
        project_root=tmp_path.resolve(),
        successor_directory=tmp_path / "other-authority-d",
        parent_directory=tmp_path / "other-authority-c",
        artifact_root_sha256="0" * 64,
        manifest_sha256="1" * 64,
        authorization_sha256="2" * 64,
        intent_sha256="3" * 64,
        nonce="4" * 64,
        python_executable=sys.executable,
    ).checked()
    transplanted = _diagnostic("passed", request=other)
    payload["fresh_verifier_diagnostic"] = transplanted
    payload["fresh_verifier_diagnostic_sha256"] = controller._compact_sha256(transplanted)
    payload["fresh_verifier_payload_sha256"] = transplanted["payload_sha256"]
    payload["controller_process_id"] = transplanted["controller_process_id"]
    payload["verifier_process_id"] = transplanted["verifier_process_id"]
    payload["verifier_parent_process_id"] = transplanted["controller_process_id"]

    with pytest.raises(controller.ControlError):
        controller._canonical_success_v2(
            payload,
            attempt=attempt,
            attempt_sha256=attempt_sha256,
        )


def test_semantic_payload_binding_is_independent_of_raw_json_framing(
    tmp_path: Path,
) -> None:
    payload, attempt, attempt_sha256 = _success_marker(tmp_path / "zażółć")
    first = controller._canonical_success_v2(
        payload,
        attempt=attempt,
        attempt_sha256=attempt_sha256,
    )
    validated_payload = payload["fresh_verifier_validated_payload"]
    request_record = payload["fresh_verifier_diagnostic"]["request"]
    request = controller.VerifyRequestV2(
        project_root=Path(request_record["project_root"]),
        successor_directory=Path(request_record["successor_directory"]),
        parent_directory=Path(request_record["parent_directory"]),
        artifact_root_sha256=request_record["artifact_root_sha256"],
        manifest_sha256=request_record["manifest_sha256"],
        authorization_sha256=request_record["authorization_sha256"],
        intent_sha256=request_record["intent_sha256"],
        nonce=request_record["nonce"],
        chain_depth=request_record["chain_depth"],
        python_executable=request_record["python_executable"],
    ).checked()
    alternative_raw = (
        json.dumps(
            validated_payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    alternative_diagnostic = _diagnostic(
        "passed",
        request=request,
        stdout_payload=alternative_raw,
    )
    alternative = copy.deepcopy(payload)
    alternative["fresh_verifier_diagnostic"] = alternative_diagnostic
    alternative["fresh_verifier_diagnostic_sha256"] = controller._compact_sha256(
        alternative_diagnostic
    )
    alternative["fresh_verifier_payload_sha256"] = alternative_diagnostic["payload_sha256"]
    second = controller._canonical_success_v2(
        alternative,
        attempt=attempt,
        attempt_sha256=attempt_sha256,
    )

    assert b"\r\n" in _windows_typer_json_bytes(validated_payload)
    assert b"\\u" in alternative_raw
    assert first["fresh_verifier_payload_sha256"] != second["fresh_verifier_payload_sha256"]
    assert (
        first["fresh_verifier_validated_payload_sha256"]
        == second["fresh_verifier_validated_payload_sha256"]
    )
    assert first["fresh_verifier_validated_payload"] == second["fresh_verifier_validated_payload"]


@pytest.mark.parametrize(
    ("failure_phase", "diagnostic_status"),
    (
        ("authority_creation_before_fresh_verifier", "not_invoked"),
        ("fresh_verifier", "spawn_failed"),
        ("fresh_verifier", "failed"),
        ("authority_creation_after_fresh_verifier", "passed"),
    ),
)
def test_failure_marker_accepts_only_exact_phase_diagnostic_status(
    tmp_path: Path,
    failure_phase: str,
    diagnostic_status: str,
) -> None:
    payload, attempt, attempt_sha256 = _failure_marker(
        tmp_path,
        failure_phase=failure_phase,
        diagnostic_status=diagnostic_status,
    )

    assert (
        controller._canonical_failure_v2(
            payload,
            attempt=attempt,
            attempt_sha256=attempt_sha256,
        )
        == payload
    )


@pytest.mark.parametrize(
    ("failure_phase", "wrong_diagnostic_status"),
    (
        ("authority_creation_before_fresh_verifier", "failed"),
        ("fresh_verifier", "not_invoked"),
        ("fresh_verifier", "passed"),
        ("authority_creation_after_fresh_verifier", "failed"),
    ),
)
def test_failure_marker_rejects_valid_but_phase_inconsistent_diagnostic(
    tmp_path: Path,
    failure_phase: str,
    wrong_diagnostic_status: str,
) -> None:
    payload, attempt, attempt_sha256 = _failure_marker(
        tmp_path,
        failure_phase=failure_phase,
        diagnostic_status=wrong_diagnostic_status,
    )

    with pytest.raises(controller.ControlError):
        controller._canonical_failure_v2(
            payload,
            attempt=attempt,
            attempt_sha256=attempt_sha256,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("project_root", "other-project"),
        ("successor_directory", "other-authority-d"),
        ("parent_directory", "other-authority-c"),
        ("authorization_sha256", "0" * 64),
        ("intent_sha256", "1" * 64),
    ),
)
def test_failure_marker_rejects_diagnostic_transplant_from_other_request(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    payload, attempt, attempt_sha256 = _failure_marker(
        tmp_path,
        failure_phase="fresh_verifier",
        diagnostic_status="failed",
    )
    request_record = dict(payload["fresh_verifier_diagnostic"]["request"])
    if field in {"project_root", "successor_directory", "parent_directory"}:
        replacement = str((tmp_path / replacement).resolve())
    request_record[field] = replacement
    other = controller.VerifyRequestV2(
        project_root=Path(request_record["project_root"]),
        successor_directory=Path(request_record["successor_directory"]),
        parent_directory=Path(request_record["parent_directory"]),
        artifact_root_sha256=request_record["artifact_root_sha256"],
        manifest_sha256=request_record["manifest_sha256"],
        authorization_sha256=request_record["authorization_sha256"],
        intent_sha256=request_record["intent_sha256"],
        nonce=request_record["nonce"],
        chain_depth=request_record["chain_depth"],
        python_executable=request_record["python_executable"],
    ).checked()
    transplanted = _diagnostic("failed", request=other)
    payload["fresh_verifier_diagnostic"] = transplanted
    payload["fresh_verifier_diagnostic_sha256"] = controller._compact_sha256(transplanted)

    with pytest.raises(controller.ControlError):
        controller._canonical_failure_v2(
            payload,
            attempt=attempt,
            attempt_sha256=attempt_sha256,
        )


def test_failure_marker_rejects_canonical_incomplete_cleanup(
    tmp_path: Path,
) -> None:
    payload, attempt, attempt_sha256 = _failure_marker(
        tmp_path,
        failure_phase="fresh_verifier",
        diagnostic_status="failed",
    )
    diagnostic = copy.deepcopy(payload["fresh_verifier_diagnostic"])
    diagnostic["failure_phase"] = "cleanup"
    diagnostic["payload_sha256"] = None
    diagnostic["cleanup"]["descendant_quiescence_proven"] = False
    diagnostic["cleanup"]["complete"] = False
    diagnostic["cleanup"]["error_codes"] = ["process_tree_probe_failed"]
    assert controller._canonical_fresh_diagnostic(diagnostic) == diagnostic
    payload["fresh_verifier_diagnostic"] = diagnostic
    payload["fresh_verifier_diagnostic_sha256"] = controller._compact_sha256(diagnostic)

    with pytest.raises(controller.ControlError):
        controller._canonical_failure_v2(
            payload,
            attempt=attempt,
            attempt_sha256=attempt_sha256,
        )


class _IntLikeHandle:
    def __init__(self, value: int) -> None:
        self.value = value

    def __int__(self) -> int:
        return self.value


class _SyntheticSuspendedProcess:
    pid = 4242
    _handle = _IntLikeHandle(4343)


class _SyntheticCanaryProcess(_SyntheticSuspendedProcess):
    returncode: int | None = None

    def wait(self, *, timeout: float) -> int:
        assert timeout > 0
        self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class _FakeWindowsJobApi:
    job_handle = 101
    thread_handle = 202

    def __init__(
        self,
        *,
        failure: str | None = None,
        thread_ids: tuple[int, ...] = (303,),
        resume_count: int = 1,
        active_pids: tuple[int, ...] | None = None,
    ) -> None:
        self.failure = failure
        self.thread_ids = thread_ids
        self.resume_count = resume_count
        self.active_pids = active_pids
        self.assigned = False
        self.terminated = False
        self.events: list[tuple[str, int | None]] = []

    def _record(self, role: str, value: int | None = None) -> None:
        self.events.append((role, value))
        if self.failure == role:
            raise OSError(f"synthetic {role} failure")

    def create_job(self) -> int:
        self._record("create")
        return self.job_handle

    def set_kill_on_close(self, job_handle: int) -> None:
        self._record("set_info", job_handle)

    def assign_process(self, job_handle: int, process_handle: int) -> None:
        assert job_handle == self.job_handle
        self._record("assign", process_handle)
        self.assigned = True

    def enumerate_threads(self, owner_pid: int) -> tuple[int, ...]:
        self._record("enumerate", owner_pid)
        return self.thread_ids

    def open_thread(self, thread_id: int) -> int:
        self._record("open_thread", thread_id)
        return self.thread_handle

    def resume_thread(self, thread_handle: int) -> int:
        self._record("resume_thread", thread_handle)
        return self.resume_count

    def terminate_job(self, job_handle: int) -> None:
        self._record("terminate_job", job_handle)
        self.terminated = True

    def active_process_ids(self, job_handle: int) -> tuple[int, ...]:
        self._record("active_process_ids", job_handle)
        if self.assigned and not self.terminated:
            if self.active_pids is not None:
                return self.active_pids
            return (_SyntheticSuspendedProcess.pid,)
        return ()

    def close_handle(self, handle: int) -> None:
        role = "close_thread" if handle == self.thread_handle else "close_job"
        self._record(role, handle)


def test_windows_job_capability_gate_configures_kill_on_close_and_closes() -> None:
    api = _FakeWindowsJobApi()

    diagnostic = controller._preflight_windows_job_containment(
        python_executable=str(Path(sys.executable).resolve(strict=True)),
        containment_factory=lambda: controller._create_windows_job_containment(api=api),
        popen_factory=lambda *_args, **_kwargs: _SyntheticCanaryProcess(),
    )

    assert diagnostic["required"] is (os.name == "nt")
    assert diagnostic["complete"] is True
    if os.name == "nt":
        assert api.events == [
            ("create", None),
            ("set_info", api.job_handle),
            ("assign", int(_SyntheticSuspendedProcess._handle)),
            ("active_process_ids", api.job_handle),
            ("enumerate", _SyntheticSuspendedProcess.pid),
            ("open_thread", 303),
            ("resume_thread", api.thread_handle),
            ("close_thread", api.thread_handle),
            ("terminate_job", api.job_handle),
            ("active_process_ids", api.job_handle),
            ("close_job", api.job_handle),
        ]


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object regression")
def test_spawn_failure_retains_pre_spawn_job_lifecycle_evidence(
    tmp_path: Path,
) -> None:
    api = _FakeWindowsJobApi()

    def fail_spawn(*_args: object, **kwargs: object) -> NoReturn:
        assert kwargs["creationflags"] == getattr(
            subprocess,
            "CREATE_SUSPENDED",
            0x00000004,
        )
        raise OSError("synthetic Popen failure")

    with pytest.raises(controller.FreshVerifierError) as raised:
        controller.run_fresh_verifier_v2(
            _verify_request(tmp_path),
            popen_factory=fail_spawn,
            containment_factory=lambda: controller._create_windows_job_containment(api=api),
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic is not None
    assert diagnostic["status"] == "spawn_failed"
    assert diagnostic["failure_phase"] == "spawn"
    assert diagnostic["verifier_process_id"] is None
    assert diagnostic["returncode"] is None
    assert diagnostic["cleanup"]["complete"] is True
    containment = diagnostic["cleanup"]["containment"]
    assert containment["created"] is True
    assert containment["kill_on_close_configured"] is True
    assert containment["child_created_suspended"] is False
    assert containment["assigned"] is False
    assert containment["close_succeeded"] is True
    assert containment["complete"] is True
    assert api.events == [
        ("create", None),
        ("set_info", api.job_handle),
        ("close_job", api.job_handle),
    ]
    assert tuple(tmp_path.iterdir()) == ()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object regression")
def test_pre_spawn_job_setup_failure_is_canonical_but_not_complete(
    tmp_path: Path,
) -> None:
    api = _FakeWindowsJobApi(failure="set_info")

    with pytest.raises(controller.FreshVerifierError) as raised:
        controller.run_fresh_verifier_v2(
            _verify_request(tmp_path),
            popen_factory=lambda *_args, **_kwargs: pytest.fail(
                "Popen must not run after failed pre-spawn containment"
            ),
            containment_factory=lambda: controller._create_windows_job_containment(api=api),
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic is not None
    assert diagnostic["status"] == "spawn_failed"
    assert diagnostic["verifier_process_id"] is None
    assert diagnostic["cleanup"]["complete"] is False
    assert diagnostic["cleanup"]["containment"]["error_codes"] == ["job_limit_failed"]
    assert tuple(tmp_path.iterdir()) == ()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object regression")
@pytest.mark.parametrize(
    ("failure", "expected_code"),
    (
        ("create", "job_create_failed"),
        ("set_info", "job_limit_failed"),
        ("close_job", "job_close_failed"),
    ),
)
def test_windows_job_capability_gate_fails_closed_without_artifacts(
    tmp_path: Path,
    failure: str,
    expected_code: str,
) -> None:
    api = _FakeWindowsJobApi(failure=failure)
    created: list[controller._WindowsJobContainment] = []

    def factory() -> controller._WindowsJobContainment:
        containment = controller._create_windows_job_containment(api=api)
        created.append(containment)
        return containment

    with pytest.raises(controller.FreshVerifierError):
        controller._preflight_windows_job_containment(
            python_executable=str(Path(sys.executable).resolve(strict=True)),
            containment_factory=factory,
            popen_factory=lambda *_args, **_kwargs: _SyntheticCanaryProcess(),
        )

    assert created
    assert expected_code in created[0].diagnostic()["error_codes"]
    assert tuple(tmp_path.iterdir()) == ()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object regression")
@pytest.mark.parametrize(
    ("failure", "thread_ids", "resume_count", "expected_code"),
    (
        ("assign", (303,), 1, "job_assign_failed"),
        ("active_process_ids", (303,), 1, "job_query_failed"),
        ("enumerate", (303,), 1, "thread_snapshot_failed"),
        (None, (), 1, "initial_thread_count_invalid"),
        (None, (303, 304), 1, "initial_thread_count_invalid"),
        ("open_thread", (303,), 1, "thread_open_failed"),
        ("resume_thread", (303,), 1, "thread_resume_failed"),
        (None, (303,), 0, "thread_resume_failed"),
        (None, (303,), 2, "thread_resume_failed"),
        ("close_thread", (303,), 1, "thread_handle_close_failed"),
    ),
)
def test_windows_job_assign_resume_failures_are_incomplete(
    failure: str | None,
    thread_ids: tuple[int, ...],
    resume_count: int,
    expected_code: str,
) -> None:
    api = _FakeWindowsJobApi(
        failure=failure,
        thread_ids=thread_ids,
        resume_count=resume_count,
    )
    containment = controller._create_windows_job_containment(api=api)
    containment.mark_child_created(suspended=True)

    with pytest.raises(controller.FreshVerifierError):
        containment.assign_and_resume(_SyntheticSuspendedProcess())

    assert expected_code in containment.diagnostic()["error_codes"]
    assert containment.diagnostic()["complete"] is False
    if containment.assigned:
        containment.terminate()
    containment.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object regression")
@pytest.mark.parametrize("invalid_handle", (None, True, 0, -1))
def test_windows_job_invalid_process_handle_is_incomplete(
    invalid_handle: object,
) -> None:
    api = _FakeWindowsJobApi()
    containment = controller._create_windows_job_containment(api=api)
    containment.mark_child_created(suspended=True)
    process = _SyntheticSuspendedProcess()
    process._handle = invalid_handle

    with pytest.raises(controller.FreshVerifierError):
        containment.assign_and_resume(process)

    assert containment.diagnostic()["error_codes"] == ["process_handle_unavailable"]
    assert containment.diagnostic()["complete"] is False
    containment.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object regression")
def test_windows_job_rejects_non_exact_assignment_membership() -> None:
    api = _FakeWindowsJobApi(
        active_pids=(_SyntheticSuspendedProcess.pid, 4344),
    )
    containment = controller._create_windows_job_containment(api=api)
    containment.mark_child_created(suspended=True)

    with pytest.raises(controller.FreshVerifierError):
        containment.assign_and_resume(_SyntheticSuspendedProcess())

    assert containment.diagnostic()["error_codes"] == ["job_query_failed"]
    assert containment.diagnostic()["assignment_membership_proven"] is False
    containment.terminate()
    containment.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object regression")
def test_windows_job_busy_query_times_out_fail_closed() -> None:
    api = _FakeWindowsJobApi()
    containment = controller._create_windows_job_containment(api=api)
    containment.mark_child_created(suspended=True)
    containment.assign_and_resume(_SyntheticSuspendedProcess())

    with pytest.raises(controller.FreshVerifierError):
        containment.prove_empty(timeout_seconds=0.0)

    assert "job_not_empty" in containment.diagnostic()["error_codes"]
    assert containment.diagnostic()["job_empty_proven"] is False
    assert containment.diagnostic()["complete"] is False
    containment.terminate()
    containment.prove_empty(timeout_seconds=0.0)
    containment.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object regression")
@pytest.mark.parametrize(
    ("failure", "expected_code"),
    (
        ("terminate_job", "job_terminate_failed"),
        ("close_job", "job_close_failed"),
    ),
)
def test_windows_job_cleanup_failures_are_incomplete(
    failure: str,
    expected_code: str,
) -> None:
    api = _FakeWindowsJobApi(failure=failure)
    containment = controller._create_windows_job_containment(api=api)
    containment.mark_child_created(suspended=True)
    containment.assign_and_resume(_SyntheticSuspendedProcess())

    with pytest.raises(controller.FreshVerifierError):
        if failure == "terminate_job":
            containment.terminate()
        else:
            containment.close()

    assert expected_code in containment.diagnostic()["error_codes"]
    assert containment.diagnostic()["complete"] is False
    if failure == "terminate_job":
        containment.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object regression")
def test_windows_job_close_kills_fast_exit_childs_sleeping_grandchild(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "job-grandchild.pid"
    child_code = """
import subprocess
import sys
from pathlib import Path

grandchild = subprocess.Popen(
    [sys.executable, "-I", "-B", "-c", "import time; time.sleep(120)"],
    executable=sys._base_executable,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    close_fds=True,
)
Path(sys.argv[1]).write_text(str(grandchild.pid), encoding="ascii")
"""
    requested = str(Path(sys.executable).resolve(strict=True))
    spawn_executable = controller._fresh_verifier_spawn_executable(requested)
    popen_arguments: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
        "creationflags": 0x00000004,
    }
    if spawn_executable is not None:
        popen_arguments["executable"] = spawn_executable
    containment = controller._create_windows_job_containment()
    process: subprocess.Popen[bytes] | None = None
    owned: list[psutil.Process] = []
    try:
        assert containment.ready_before_spawn
        process = subprocess.Popen(
            [requested, "-I", "-B", "-c", child_code, str(pid_path)],
            **popen_arguments,
        )
        owned.append(psutil.Process(process.pid))
        containment.mark_child_created(suspended=True)
        containment.assign_and_resume(process)
        process.wait(timeout=15)
        deadline = time.monotonic() + 10.0
        while not pid_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_path.is_file()
        grandchild = psutil.Process(int(pid_path.read_text(encoding="ascii")))
        owned.append(grandchild)
        assert _is_alive(grandchild)

        containment.close()

        while _is_alive(grandchild) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _is_alive(grandchild)
        containment_diagnostic = containment.diagnostic()
        assert containment_diagnostic["close_succeeded"] is True
        assert containment_diagnostic["job_empty_proven"] is False
        assert containment_diagnostic["complete"] is False
    finally:
        if containment.handle is not None:
            if containment.assigned:
                with suppress(controller.FreshVerifierError):
                    containment.terminate()
            with suppress(controller.FreshVerifierError):
                containment.close()
        _kill_owned_processes(owned)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)


class _OwnedTestLock:
    def __enter__(self) -> _OwnedTestLock:
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc: object,
        _traceback: object,
    ) -> None:
        return None

    def assert_owned(self) -> None:
        return None


class _CreatorFailsBeforeCallback:
    post_publication_check: object | None = None

    def create_authority(
        self,
        *,
        authorization: dict[str, Any],
        post_publication_check: object,
    ) -> object:
        assert authorization == {"kind": "synthetic-technical-authorization"}
        self.post_publication_check = post_publication_check
        raise RuntimeError("synthetic creator failure before callback")

    def authority_pins(self, _published: object) -> controller.AuthorityPins:
        raise AssertionError("creator-before-callback must not inspect pins")


@pytest.mark.parametrize(
    "canary_spawn_fails",
    (
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                os.name != "nt",
                reason="the disposable Job Object canary is a Windows-only gate",
            ),
        ),
    ),
    ids=("creator-fails-after-canary", "canary-fails-before-a2"),
)
def test_containment_canary_precedes_a2_and_creator_failure_has_scoped_f2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canary_spawn_fails: bool,
) -> None:
    project = tmp_path.resolve()
    namespace = controller.Namespace.for_project(project)
    namespace.control_root.mkdir(parents=True)
    amendment_root = project / "artifacts" / "preregistration_amendments"
    parent = amendment_root / controller._AUTHORITY_C_COMPONENT
    destination = amendment_root / "synthetic-authority-d"
    parent.mkdir(parents=True)
    authorization_sha256 = "3" * 64
    intent_sha256 = "4" * 64
    authorization_receipt_sha256 = "5" * 64
    attempt_id = "6" * 64
    authorization = {
        "authorized_attempt_id": attempt_id,
        "publication": {
            "intended_authority_directory": str(destination.resolve()),
        },
        "preflight": {
            "contract": {
                "technical_successor": {
                    "authorization": {"kind": "synthetic-technical-authorization"},
                    "authorization_sha256": authorization_sha256,
                    "intent_sha256": intent_sha256,
                },
                "terminal_qualification": {"kind": "synthetic-terminal"},
            }
        },
    }
    verification_nonce = controller._verification_nonce_v2(
        authorization,
        authorization_receipt_sha256,
    )
    attempt = {
        "claimed_at_utc": controller._timestamp(datetime.now(UTC) - timedelta(minutes=1)),
        "attempt_id": attempt_id,
        "intended_authority_directory": str(destination.resolve()),
        "parent_authority_directory": str(parent.resolve()),
        "technical_authorization_sha256": authorization_sha256,
        "intent_sha256": intent_sha256,
        "verification_nonce": verification_nonce,
        "run_state": {"sha256": "7" * 64},
    }
    attempt_bytes = controller._canonical_bytes(attempt)
    attempt_sha256 = _sha256(attempt_bytes)

    def presence(_namespace: controller.Namespace) -> dict[str, bool]:
        return {
            "qualification": True,
            "inputs": True,
            "authorization": True,
            "attempt": namespace.attempt_v2.exists(),
            "success": namespace.success_v2.exists(),
            "failure": namespace.failure_v2.exists(),
        }

    monkeypatch.setattr(
        controller,
        "_protocol_lock",
        lambda *_args, **_kwargs: _OwnedTestLock(),
    )
    monkeypatch.setattr(
        controller,
        "ExclusiveBundlePublicationLock",
        lambda *_args, **_kwargs: _OwnedTestLock(),
    )
    monkeypatch.setattr(
        controller,
        "_require_legacy_lock_state_under_protocol_lock",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        controller,
        "_legacy_scoped_lock_paths",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        controller,
        "_read_publication_authorization_v2",
        lambda *_args, **_kwargs: (
            authorization,
            authorization_receipt_sha256,
        ),
    )
    monkeypatch.setattr(controller, "_reserved_family_presence", presence)
    monkeypatch.setattr(
        controller,
        "_stable_amendment_inventory",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        controller,
        "_build_attempt_v2",
        lambda **_kwargs: attempt,
    )
    monkeypatch.setattr(
        controller,
        "_read_attempt_v2",
        lambda *_args, **_kwargs: (attempt, attempt_sha256),
    )
    monkeypatch.setattr(
        controller,
        "_verify_live_governed_baseline_v2",
        lambda *_args, **_kwargs: None,
    )
    api = _CreatorFailsBeforeCallback()

    def fail_canary_spawn(*_args: object, **_kwargs: object) -> NoReturn:
        raise OSError("synthetic pre-A2 canary spawn failure")

    verifier = controller.TransactionVerifierV2(
        api=api,
        project_root=project,
        parent=parent,
        destination=destination,
        authorization_sha256=authorization_sha256,
        intent_sha256=intent_sha256,
        verification_nonce=verification_nonce,
        canary_popen_factory=(fail_canary_spawn if canary_spawn_fails else subprocess.Popen),
    )
    started = datetime.now(UTC) - timedelta(seconds=10)
    clock_values = iter(
        (
            started,
            started + timedelta(seconds=1),
            started + timedelta(seconds=2),
        )
    )

    def execute() -> controller.PublicationResultV2:
        return controller._execute_publication_v2_once(
            namespace=namespace,
            parent_authority_directory=parent,
            authorization=authorization,
            authorization_receipt_sha256=authorization_receipt_sha256,
            api=api,
            verifier=verifier,
            clock=lambda: next(clock_values),
            candidate_discoverer=lambda _parent: (),
        )

    if canary_spawn_fails:
        with pytest.raises(
            controller.FreshVerifierError,
            match="full disposable capability canary failed",
        ):
            execute()
        assert not namespace.attempt_v2.exists()
        assert not namespace.failure_v2.exists()
        assert not namespace.success_v2.exists()
        assert not destination.exists()
        assert api.post_publication_check is None
        assert verifier.containment_preflight_completed is False
        assert verifier.invoked is False
        assert verifier.result is None
        return

    result = execute()

    assert result.state is controller.State.ROLLED_BACK_FAILURE
    assert result.marker_path == namespace.failure_v2
    assert result.authority_directory is None
    assert namespace.attempt_v2.read_bytes() == attempt_bytes
    assert namespace.failure_v2.is_file()
    assert not namespace.success_v2.exists()
    assert not destination.exists()
    assert api.post_publication_check is not None
    assert verifier.invoked is False
    assert verifier.result is None
    assert verifier.published_result is None
    failure, failure_sha256 = controller._read_failure_v2(
        namespace,
        attempt=attempt,
        attempt_sha256=attempt_sha256,
    )
    assert failure_sha256 == result.marker_sha256
    assert failure["failure_phase"] == "authority_creation_before_fresh_verifier"
    assert failure["fresh_verifier_diagnostic"] == verifier.diagnostic
    assert failure["fresh_verifier_diagnostic"]["status"] == "not_invoked"
    assert failure["fresh_verifier_diagnostic"]["cleanup"]["complete"] is True
    assert failure["automatic_retry_allowed"] is False
