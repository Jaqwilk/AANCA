from __future__ import annotations

import copy
import ctypes
import hashlib
import inspect
import os
import queue
import stat
import sys
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

if os.name == "nt":
    import msvcrt
else:
    msvcrt = None

from carrier_import_guard import PACKAGE_IMPORT_ROOT, import_exact

terminal = import_exact(
    "histo_audit.workflows.original_confirmatory_capsule_terminal",
    PACKAGE_IMPORT_ROOT / "histo_audit" / "workflows" / "original_confirmatory_capsule_terminal.py",
)
authority = import_exact(
    "histo_audit.workflows.original_confirmatory_capsule_authority",
    PACKAGE_IMPORT_ROOT
    / "histo_audit"
    / "workflows"
    / "original_confirmatory_capsule_authority.py",
)

_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64
_HASH_D = "d" * 64


class _CountingBootstrap:
    def __init__(self, claim_path: Path) -> None:
        self.claim_path = claim_path
        self.arm_count = 0
        self.take_count = 0
        self._descriptor = -1

    def _arm_original_confirmatory_e_claim_after_full_prevalidation(self) -> None:
        if self.arm_count or self.take_count:
            raise AssertionError("bootstrap claim was armed more than once")
        self.arm_count += 1
        self._descriptor = authority._open_read_descriptor(self.claim_path)

    def _take_original_confirmatory_e_claim_read_handle(
        self,
    ) -> tuple[int, str, str, int]:
        if self.arm_count != 1 or self.take_count:
            raise AssertionError("bootstrap claim was taken out of order")
        self.take_count += 1
        descriptor = self._descriptor
        self._descriptor = -1
        payload = self.claim_path.read_bytes()
        return (
            descriptor,
            str(self.claim_path),
            hashlib.sha256(payload).hexdigest(),
            len(payload),
        )


class _NeverCalledBootstrap:
    def __init__(self) -> None:
        self.arm_count = 0
        self.take_count = 0

    def _arm_original_confirmatory_e_claim_after_full_prevalidation(self) -> None:
        self.arm_count += 1
        raise AssertionError("invalid context must not arm consumed E")

    def _take_original_confirmatory_e_claim_read_handle(
        self,
    ) -> tuple[int, str, str, int]:
        self.take_count += 1
        raise AssertionError("invalid context must not take consumed E")


class _Stream:
    def __init__(self, buffer: Any) -> None:
        self.buffer = buffer


class _QueueWriter:
    def __init__(self, destination: queue.Queue[bytes]) -> None:
        self.destination = destination
        self.write_count = 0

    def write(self, payload: bytes) -> int:
        self.write_count += 1
        self.destination.put(payload)
        return len(payload)

    def flush(self) -> None:
        return None


class _QueueReader:
    def __init__(self, source: queue.Queue[bytes]) -> None:
        self.source = source

    def readline(self, maximum_bytes: int) -> bytes:
        payload = self.source.get(timeout=10)
        return payload[:maximum_bytes]


class _FixedReader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.read_count = 0

    def readline(self, maximum_bytes: int) -> bytes:
        self.read_count += 1
        return self.payload[:maximum_bytes]


class _CaptureWriter:
    def __init__(self) -> None:
        self.payloads: list[bytes] = []

    def write(self, payload: bytes) -> int:
        self.payloads.append(payload)
        return len(payload)

    def flush(self) -> None:
        return None


def _readonly(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.chmod(stat.S_IREAD)


def _make_preterminal_context(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        job_id="job-1",
        attempt_id="attempt-1",
        run_id="run-1",
        execution_mode="fresh",
        retry_of_run_id=None,
        attempt_nonce=_HASH_D,
        q={
            "execution_capsule": {
                "contract_sha256": _HASH_A,
                "sha256": _HASH_B,
                "internal_manifest_sha256": _HASH_C,
            }
        },
        observed_environment_sha256=_HASH_A,
        preterminal_contract=SimpleNamespace(
            preterminal_pin_receipt_path=tmp_path / "preterminal_pin.json",
            preterminal_pin_receipt_max_bytes=1024 * 1024,
            contract_sha256=_HASH_B,
        ),
        overlap_contract=SimpleNamespace(
            ready_line_max_bytes=1024 * 1024,
            ack_line_max_bytes=1024 * 1024,
            contract_sha256=_HASH_C,
        ),
    )


def _make_terminal_context(tmp_path: Path) -> SimpleNamespace:
    composed = tmp_path / "composed_terminal.json"
    readback = tmp_path / "postwake_readback.json"
    return SimpleNamespace(
        job_id="job-1",
        attempt_id="attempt-1",
        run_id="run-1",
        execution_mode="fresh",
        retry_of_run_id=None,
        attempt_nonce=_HASH_D,
        q={"q_authority_root_sha256": _HASH_A},
        e={"intent_core_sha256": _HASH_B},
        e_file=SimpleNamespace(file_sha256=_HASH_C),
        spec_sha256=_HASH_D,
        observed_environment_sha256=_HASH_A,
        terminal_command=SimpleNamespace(command_sha256=_HASH_B),
        terminal_launcher_projection={
            "launch_intent_path": str((tmp_path / "terminal_client_launch_intent.json").resolve()),
        },
        custody_seed=SimpleNamespace(seed_sha256=_HASH_C),
        input_lease_contract=SimpleNamespace(
            lease_receipt_path=tmp_path / "postwake_lease.json",
        ),
        custody_contract=SimpleNamespace(
            contract_sha256=_HASH_D,
            readback_receipt_path=readback,
        ),
        terminal_contract=SimpleNamespace(
            contract_sha256=_HASH_A,
            composed_terminal_receipt_path=composed,
            composed_terminal_receipt_max_bytes=1024 * 1024,
            postwake_composed_readback_receipt_max_bytes=1024 * 1024,
        ),
    )


def _make_consumed_e_stub(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        held=SimpleNamespace(
            path=tmp_path / authority.E_CONSUMPTION_TOMBSTONE_FILENAME,
            file_sha256=_HASH_A,
        ),
        launch_intent=SimpleNamespace(file_sha256=_HASH_B),
        process_started=SimpleNamespace(file_sha256=_HASH_C),
        claim={"claim_root_sha256": _HASH_D},
        revalidate=lambda: None,
    )


def _make_outcome_blind_artifact_inputs(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_id = "run-1"
    runs_root = (tmp_path / "runs").resolve()
    e_intent = {
        "scientific_request_projection": {
            "run_id": run_id,
            "runs_root": str(runs_root),
            "expected_run_directory": str(runs_root / run_id),
        }
    }
    rules = list(
        terminal._build_outcome_blind_expected_artifact_rules(
            e_intent=e_intent,
        )
    )
    e_intent["job"] = {
        "terminal_custody_authority_projection": {
            "outcome_blind_expected_artifact_instance": {
                "expected_artifacts": rules,
                "required_success_roles": list(terminal._EXPECTED_ARTIFACT_ROLE_ORDER),
                "expected_artifacts_root_sha256": terminal._canonical_sha256(rules),
            }
        }
    }
    spec = {
        "expected_artifacts": rules,
        "required_success_roles": list(terminal._EXPECTED_ARTIFACT_ROLE_ORDER),
    }
    return e_intent, spec


def _make_pin_validation_bundle(
    tmp_path: Path,
) -> tuple[SimpleNamespace, SimpleNamespace, dict[str, Any]]:
    e_intent, artifact_spec = _make_outcome_blind_artifact_inputs(tmp_path)
    rules = tuple(copy.deepcopy(artifact_spec["expected_artifacts"]))
    consumed = _make_consumed_e_stub(tmp_path)
    context = SimpleNamespace(
        job_id="job-1",
        attempt_id="attempt-1",
        run_id="run-1",
        execution_mode="fresh",
        retry_of_run_id=None,
        attempt_nonce=_HASH_D,
        q={
            "q_authority_root_sha256": _HASH_A,
            "execution_capsule": {
                "contract_sha256": _HASH_A,
                "sha256": _HASH_B,
                "internal_manifest_sha256": _HASH_C,
            },
        },
        e={
            "intent_core_sha256": _HASH_B,
            "scientific_request_projection": e_intent["scientific_request_projection"],
        },
        e_file=SimpleNamespace(
            path=(tmp_path / "e_intent.json").resolve(),
            file_sha256=_HASH_C,
        ),
        spec_sha256=_HASH_D,
        run_spec_payload={"canonical_spec_sha256": _HASH_A},
        scientific_command=SimpleNamespace(command_sha256=_HASH_B),
        preterminal_command=SimpleNamespace(command_sha256=_HASH_C),
        terminal_command=SimpleNamespace(command_sha256=_HASH_D),
        preterminal_contract=SimpleNamespace(contract_sha256=_HASH_B),
        overlap_contract=SimpleNamespace(contract_sha256=_HASH_C),
        observed_environment_sha256=_HASH_A,
        expected_artifact_rules=rules,
    )
    evidence = [
        {
            "role": rule["role"],
            "path": rule["path"],
            "size_bytes": index + 1,
            "sha256": hashlib.sha256(f"artifact-{index}".encode()).hexdigest(),
            "expected_sha256": rule["expected_sha256"],
            "json_control_paths_checked": sorted(rule["json_equals"]),
            "valid": True,
        }
        for index, rule in enumerate(rules)
    ]
    artifact_root = terminal._canonical_sha256(evidence)
    scientific_core = {
        "run_spec_file_sha256": context.spec_sha256,
        "canonical_spec_sha256": context.run_spec_payload["canonical_spec_sha256"],
        "launch_intent_file_sha256": consumed.launch_intent.file_sha256,
        "process_started_file_sha256": consumed.process_started.file_sha256,
        "e_consumption_claim_file_sha256": consumed.held.file_sha256,
        "e_consumption_claim_root_sha256": consumed.claim["claim_root_sha256"],
        "scientific_command_sha256": context.scientific_command.command_sha256,
        "required_success_roles": list(terminal._EXPECTED_ARTIFACT_ROLE_ORDER),
        "expected_artifact_evidence_root_sha256": artifact_root,
    }
    unsigned = {
        "schema_version": 1,
        "policy": terminal.PRETERMINAL_PIN_POLICY,
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
        "e_consumption_claim_path": str(consumed.held.path),
        "e_consumption_claim_file_sha256": consumed.held.file_sha256,
        "e_consumption_claim_root_sha256": consumed.claim["claim_root_sha256"],
        "capsule_contract_sha256": context.q["execution_capsule"]["contract_sha256"],
        "capsule_sha256": context.q["execution_capsule"]["sha256"],
        "capsule_internal_manifest_sha256": context.q["execution_capsule"][
            "internal_manifest_sha256"
        ],
        "capsule_mode": authority.CAPSULE_PRETERMINAL_MODE,
        "preterminal_pin_contract_sha256": (context.preterminal_contract.contract_sha256),
        "preterminal_overlap_handshake_contract_sha256": (context.overlap_contract.contract_sha256),
        "run_spec_file_sha256": context.spec_sha256,
        "canonical_spec_sha256": context.run_spec_payload["canonical_spec_sha256"],
        "launch_intent_file_sha256": consumed.launch_intent.file_sha256,
        "process_started_file_sha256": consumed.process_started.file_sha256,
        "scientific_command_sha256": context.scientific_command.command_sha256,
        "preterminal_command_sha256": context.preterminal_command.command_sha256,
        "terminal_command_sha256": context.terminal_command.command_sha256,
        "observed_integrity_verifier_environment_sha256": (context.observed_environment_sha256),
        "required_success_roles": list(terminal._EXPECTED_ARTIFACT_ROLE_ORDER),
        "expected_artifact_evidence": evidence,
        "expected_artifact_evidence_root_sha256": artifact_root,
        "preterminal_scientific_core_sha256": terminal._canonical_sha256(scientific_core),
        "semantic_outcome_read_scope": terminal.SEMANTIC_OUTCOME_READ_SCOPE,
        "outcome_values_read": False,
        "outcome_values_emitted": False,
        "outcome_values_used_for_selection_or_tuning": False,
        "training_or_model_selection_allowed": False,
        "scientific_publication_allowed": False,
        "automatic_retry_allowed": False,
        "created_at_utc": "2026-07-30T00:00:00.000000Z",
    }
    pin = {
        **unsigned,
        "evidence_root_sha256": terminal._canonical_sha256(unsigned),
    }
    assert set(pin) == terminal._PRETERMINAL_PIN_FIELDS
    return context, consumed, pin


def _resign_pin(pin: dict[str, Any]) -> None:
    evidence = pin["expected_artifact_evidence"]
    evidence_root = terminal._canonical_sha256(evidence)
    pin["expected_artifact_evidence_root_sha256"] = evidence_root
    scientific_core = {
        "run_spec_file_sha256": pin["run_spec_file_sha256"],
        "canonical_spec_sha256": pin["canonical_spec_sha256"],
        "launch_intent_file_sha256": pin["launch_intent_file_sha256"],
        "process_started_file_sha256": pin["process_started_file_sha256"],
        "e_consumption_claim_file_sha256": pin["e_consumption_claim_file_sha256"],
        "e_consumption_claim_root_sha256": pin["e_consumption_claim_root_sha256"],
        "scientific_command_sha256": pin["scientific_command_sha256"],
        "required_success_roles": pin["required_success_roles"],
        "expected_artifact_evidence_root_sha256": evidence_root,
    }
    pin["preterminal_scientific_core_sha256"] = terminal._canonical_sha256(scientific_core)
    unsigned = {key: value for key, value in pin.items() if key != "evidence_root_sha256"}
    pin["evidence_root_sha256"] = terminal._canonical_sha256(unsigned)


def _valid_log_record(path: Path, *, maximum_bytes: int) -> dict[str, Any]:
    payload_sha = hashlib.sha256(b"synthetic-log").hexdigest()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": len(b"synthetic-log"),
        "sha256": payload_sha,
        "limit_bytes": maximum_bytes,
        "limit_exceeded": False,
        "capture_complete": True,
        "stream_size_bytes": len(b"synthetic-log"),
        "stream_sha256": payload_sha,
        "stored_sha256": payload_sha,
        "discarded_bytes": 0,
    }


def _make_terminal_validation_bundle(
    tmp_path: Path,
) -> tuple[
    SimpleNamespace,
    SimpleNamespace,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    max_log_bytes = 1024
    binding = {
        "binding_sha256": _HASH_A,
        "exact_supervisor_environment_sha256": _HASH_B,
        "exact_environment_sha256": _HASH_C,
        "exact_integrity_verifier_environment_sha256": _HASH_D,
    }
    spec = {
        "max_log_bytes": max_log_bytes,
        "verifier_timeout_ms": 60_000,
        "expected_environment": {
            "envelope_sha256": _HASH_A,
            "launch_environment_root_sha256": _HASH_B,
        },
        "process_environment_binding": binding,
        "capsule_lease_identity": {"synthetic": "capsule-leaf"},
        "capsule_lease_identity_root_sha256": _HASH_A,
        "capsule_ancestor_lease": {"synthetic": "capsule-ancestor"},
        "capsule_ancestor_lease_root_sha256": _HASH_B,
        "python_lease_identity": {"synthetic": "python-leaf"},
        "python_lease_identity_root_sha256": _HASH_C,
        "python_ancestor_lease": {"synthetic": "python-ancestor"},
        "python_ancestor_lease_root_sha256": _HASH_D,
        "python_runtime_resolution_policy": "synthetic-runtime-resolution-v1",
        "runtime_python_lease_identity": {"synthetic": "runtime-python-leaf"},
        "runtime_python_lease_identity_root_sha256": _HASH_A,
        "runtime_python_ancestor_lease": {"synthetic": "runtime-python-ancestor"},
        "runtime_python_ancestor_lease_root_sha256": _HASH_B,
    }
    context = SimpleNamespace(
        job_dir=(tmp_path / "job").resolve(),
        job_id="job-1",
        attempt_id="attempt-1",
        run_id="run-1",
        execution_mode="fresh",
        retry_of_run_id=None,
        attempt_nonce=_HASH_D,
        spec_sha256=_HASH_C,
        spec=spec,
        q={
            "execution_capsule": {
                "contract_sha256": _HASH_A,
                "sha256": _HASH_B,
                "internal_manifest_sha256": _HASH_C,
            }
        },
        e={
            "e_consumption_contract": {
                "contract_sha256": _HASH_D,
            }
        },
        preterminal_contract=SimpleNamespace(contract_sha256=_HASH_A),
        overlap_contract=SimpleNamespace(contract_sha256=_HASH_B),
        preterminal_command=SimpleNamespace(
            program_path=Path(sys.executable).resolve(),
            program_sha256=_HASH_A,
            command_sha256=_HASH_B,
        ),
    )
    context.job_dir.mkdir(parents=True, exist_ok=True)
    (context.job_dir / "stdout.log").write_bytes(b"synthetic-log")
    (context.job_dir / "stderr.log").write_bytes(b"synthetic-log")
    consumed = SimpleNamespace(
        launch_intent=SimpleNamespace(file_sha256=_HASH_A),
        process_started=SimpleNamespace(file_sha256=_HASH_B),
        revalidate=lambda: None,
    )
    launch = {
        "observed_supervisor_environment_sha256": _HASH_B,
        "created_at_utc": "2026-07-30T00:00:00.000000Z",
    }
    started = {
        "observed_child_environment_sha256": _HASH_C,
        "started_at_utc": "2026-07-30T00:00:01.000000Z",
    }
    verifier = {
        "observed_integrity_verifier_environment_sha256": _HASH_D,
        "ended_at_utc": "2026-07-30T00:00:02.000000Z",
    }
    pin_records = [
        {
            "role": role,
            "path": str(context.job_dir / f"{index}.json"),
            "size_bytes": index + 1,
            "sha256": hashlib.sha256(f"terminal-{index}".encode()).hexdigest(),
            "expected_sha256": None,
            "json_control_paths_checked": [],
            "valid": True,
        }
        for index, role in enumerate(terminal._EXPECTED_ARTIFACT_ROLE_ORDER)
    ]
    pin = {"expected_artifact_evidence": pin_records}
    artifacts = [
        {
            "role": item["role"],
            "path": item["path"],
            "valid": True,
            "errors": [],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in pin_records
    ]
    environment = {
        "expected_environment_envelope_sha256": _HASH_A,
        "launch_environment_root_sha256": _HASH_B,
        "process_environment_binding_sha256": _HASH_A,
        "exact_supervisor_environment_sha256": _HASH_B,
        "observed_supervisor_environment_sha256": _HASH_B,
        "exact_environment_sha256": _HASH_C,
        "observed_child_environment_sha256": _HASH_C,
        "exact_integrity_verifier_environment_sha256": _HASH_D,
        "observed_integrity_verifier_environment_sha256": _HASH_D,
        "all_exact_matches": True,
    }
    terminal_receipt = {
        "schema_version": 3,
        "policy": authority.SUPERVISOR_V3_POLICY,
        "attempt_policy": terminal._SUPERVISOR_ATTEMPT_POLICY,
        "job_id": context.job_id,
        "process_kind": terminal._SUPERVISOR_PROCESS_KIND,
        "spec_sha256": context.spec_sha256,
        "terminal_kind": authority.SUPERVISOR_V2_SUCCESS_TERMINAL_KIND,
        "reason": terminal._SUPERVISOR_SUCCESS_REASON,
        "attempt_count": 1,
        "automatic_retry_allowed": False,
        "exit_code": 0,
        "launch_intent_receipt_sha256": consumed.launch_intent.file_sha256,
        "process_started_receipt_sha256": consumed.process_started.file_sha256,
        "stdout": _valid_log_record(
            context.job_dir / "stdout.log",
            maximum_bytes=max_log_bytes,
        ),
        "stderr": _valid_log_record(
            context.job_dir / "stderr.log",
            maximum_bytes=max_log_bytes,
        ),
        "expected_artifacts": artifacts,
        "integrity_verifier": verifier,
        "descendants_after_root_exit": 0,
        "recovery_evidence": None,
        "ended_at_utc": "2026-07-30T00:00:03.000000Z",
        "preterminal_pin_contract_sha256": (context.preterminal_contract.contract_sha256),
        "preterminal_overlap_handshake_contract_sha256": (context.overlap_contract.contract_sha256),
        "environment_binding": environment,
        "capsule_lease_identity": spec["capsule_lease_identity"],
        "capsule_lease_identity_root_sha256": spec["capsule_lease_identity_root_sha256"],
        "capsule_ancestor_lease": spec["capsule_ancestor_lease"],
        "capsule_ancestor_lease_root_sha256": spec["capsule_ancestor_lease_root_sha256"],
        "python_lease_identity": spec["python_lease_identity"],
        "python_lease_identity_root_sha256": spec["python_lease_identity_root_sha256"],
        "python_ancestor_lease": spec["python_ancestor_lease"],
        "python_ancestor_lease_root_sha256": spec["python_ancestor_lease_root_sha256"],
        "python_runtime_resolution_policy": spec["python_runtime_resolution_policy"],
        "runtime_python_lease_identity": spec["runtime_python_lease_identity"],
        "runtime_python_lease_identity_root_sha256": spec[
            "runtime_python_lease_identity_root_sha256"
        ],
        "runtime_python_ancestor_lease": spec["runtime_python_ancestor_lease"],
        "runtime_python_ancestor_lease_root_sha256": spec[
            "runtime_python_ancestor_lease_root_sha256"
        ],
        "e_consumption_contract_sha256": _HASH_D,
        "e_consumption_custody_receipt_file_sha256": _HASH_A,
        "e_consumption_custody_receipt_root_sha256": _HASH_B,
        "e_consumption_ready_sha256": _HASH_C,
        "e_consumption_ack_sha256": _HASH_D,
    }
    assert set(terminal_receipt) == terminal._SUPERVISOR_TERMINAL_FIELDS
    return context, consumed, pin, launch, started, terminal_receipt


def _process_identity(pid: int) -> dict[str, Any]:
    return {
        "pid": pid,
        "creation_time_100ns": 134298432000000000,
        "creation_time_utc": "2026-07-30T00:00:00.000000Z",
        "program_path": str(Path(sys.executable).resolve()),
        "program_sha256": _HASH_A,
        "command_sha256": _HASH_B,
    }


def _create_test_message_pipe(pipe_name: str) -> tuple[Any, int]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_named_pipe = kernel32.CreateNamedPipeW
    create_named_pipe.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_named_pipe.restype = ctypes.c_void_p
    server_handle = create_named_pipe(
        pipe_name,
        0x00000003,
        0x00000004 | 0x00000002,
        1,
        64 * 1024,
        64 * 1024,
        0,
        None,
    )
    if server_handle == terminal.INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())
    return kernel32, int(server_handle)


def test_invalid_context_never_arms_or_takes_consumed_e() -> None:
    bootstrap = _NeverCalledBootstrap()
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        terminal._load_context_then_take_consumed_e(
            authority.CAPSULE_PRETERMINAL_MODE,
            (),
            bootstrap_module=bootstrap,
        )
    assert bootstrap.arm_count == 0
    assert bootstrap.take_count == 0


def test_launch_evidence_failure_never_arms_consumed_e(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(close_count=0)

    def close() -> None:
        context.close_count += 1

    context.close = close
    monkeypatch.setattr(
        terminal,
        "_load_verified_context",
        lambda _mode, _tail: context,
    )
    monkeypatch.setattr(
        terminal,
        "_revalidate_terminal_process_ancestry",
        lambda _context: None,
    )

    def fail_prevalidation(_context: Any) -> Any:
        raise terminal.OriginalConfirmatoryTerminalError("synthetic launch evidence failure")

    monkeypatch.setattr(
        terminal,
        "_open_validated_launch_evidence",
        fail_prevalidation,
    )

    def forbidden_arm(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("consumed E armed before launch evidence passed")

    monkeypatch.setattr(
        terminal,
        "_arm_and_take_consumed_e_claim",
        forbidden_arm,
    )
    with pytest.raises(
        terminal.OriginalConfirmatoryTerminalError,
        match="synthetic launch evidence failure",
    ):
        terminal._load_context_then_take_consumed_e(
            authority.CAPSULE_TERMINAL_MODE,
            (),
        )
    assert context.close_count == 1


def test_valid_launch_evidence_precedes_one_consumed_e_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    context = SimpleNamespace(close=lambda: order.append("context-close"))
    evidence = SimpleNamespace(close=lambda: order.append("evidence-close"))
    consumed = SimpleNamespace()
    bootstrap = object()
    monkeypatch.setattr(
        terminal,
        "_load_verified_context",
        lambda _mode, _tail: context,
    )
    monkeypatch.setattr(
        terminal,
        "_revalidate_terminal_process_ancestry",
        lambda _context: order.append("ancestry-revalidation"),
    )

    def prevalidate(_context: Any) -> Any:
        order.append("full-prevalidation")
        return evidence

    def arm(
        observed_context: Any,
        *,
        launch_evidence: Any,
        bootstrap_module: Any,
    ) -> Any:
        assert observed_context is context
        assert launch_evidence is evidence
        assert bootstrap_module is bootstrap
        assert order == [
            "ancestry-revalidation",
            "full-prevalidation",
            "ancestry-revalidation",
        ]
        order.append("arm-take-once")
        return consumed

    monkeypatch.setattr(
        terminal,
        "_open_validated_launch_evidence",
        prevalidate,
    )
    monkeypatch.setattr(
        terminal,
        "_arm_and_take_consumed_e_claim",
        arm,
    )
    observed_context, observed_consumed = terminal._load_context_then_take_consumed_e(
        authority.CAPSULE_TERMINAL_MODE,
        (),
        bootstrap_module=bootstrap,
    )
    assert observed_context is context
    assert observed_consumed is consumed
    assert order == [
        "ancestry-revalidation",
        "full-prevalidation",
        "ancestry-revalidation",
        "arm-take-once",
    ]


def test_terminal_launch_evidence_paths_are_derived_from_job_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_paths: list[Path] = []

    class Receipt:
        def __init__(self, path: Path, sha256: str) -> None:
            self.path = path
            self.file_sha256 = sha256
            self.close_count = 0
            self.revalidate_count = 0

        def close(self) -> None:
            self.close_count += 1

        def revalidate(self) -> None:
            self.revalidate_count += 1

    launch_sha = _HASH_A

    def read_receipt(
        path: Path,
        *,
        role: str,
    ) -> tuple[Receipt, dict[str, Any], str]:
        observed_paths.append(path)
        if role == "launch intent":
            payload = {
                "job_id": "job-1",
                "spec_sha256": _HASH_B,
                "attempt_nonce": _HASH_C,
                "command_sha256": _HASH_D,
                "automatic_retry_allowed": False,
            }
            return Receipt(path, launch_sha), payload, launch_sha
        payload = {
            "job_id": "job-1",
            "spec_sha256": _HASH_B,
            "attempt_nonce": _HASH_C,
            "launch_intent_sha256": launch_sha,
            "automatic_retry_allowed": False,
            "child_environment_exact_match": True,
            "observed_child_environment_sha256": _HASH_A,
        }
        return Receipt(path, _HASH_B), payload, _HASH_B

    monkeypatch.setattr(terminal, "_read_required_receipt", read_receipt)
    monkeypatch.setattr(
        terminal,
        "_validate_launch_intent_payload",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        terminal,
        "_validate_process_started_payload",
        lambda value, **_kwargs: value,
    )
    context = SimpleNamespace(
        values={},
        job_dir=tmp_path / "job",
        job_id="job-1",
        spec_sha256=_HASH_B,
        attempt_nonce=_HASH_C,
        scientific_command=SimpleNamespace(command_sha256=_HASH_D),
        e={
            "process_environment_binding": {
                "exact_environment_sha256": _HASH_A,
            }
        },
    )
    evidence = terminal._open_validated_launch_evidence(context)
    try:
        assert observed_paths == [
            context.job_dir / "launch_intent.json",
            context.job_dir / "process_started.json",
        ]
        assert evidence.launch_intent.revalidate_count == 1
        assert evidence.process_started.revalidate_count == 1
    finally:
        evidence.close()


def test_bootstrap_consumed_e_transfer_arms_and_takes_exactly_once(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "jobs" / "job-1"
    job_dir.mkdir(parents=True)
    claim = job_dir / authority.E_CONSUMPTION_TOMBSTONE_FILENAME
    _readonly(claim, b'{"synthetic":"control-only"}\n')
    bootstrap = _CountingBootstrap(claim)

    transfer = terminal._take_bootstrap_consumed_e_transfer(
        job_dir,
        bootstrap_module=bootstrap,
    )
    try:
        assert bootstrap.arm_count == 1
        assert bootstrap.take_count == 1
        assert transfer.path == claim
        assert transfer.size_bytes == claim.stat().st_size
        assert (
            terminal._read_descriptor(
                transfer.descriptor,
                maximum_bytes=1024,
            )
            == claim.read_bytes()
        )
    finally:
        os.close(transfer.descriptor)
    with pytest.raises(AssertionError, match="armed more than once"):
        terminal._take_bootstrap_consumed_e_transfer(
            job_dir,
            bootstrap_module=bootstrap,
        )
    try:
        assert bootstrap.arm_count == 1
        assert bootstrap.take_count == 1
    finally:
        claim.chmod(stat.S_IWRITE)


@pytest.mark.skipif(os.name != "nt", reason="Windows named-pipe bound validation")
def test_duplex_pipe_rejects_unbounded_message_allocation() -> None:
    with pytest.raises(
        terminal.OriginalConfirmatoryTerminalError,
        match="bounds are invalid",
    ):
        terminal._DuplexPipeClient(
            rf"\\.\pipe\aanca-not-created-{uuid.uuid4().hex}",
            outbound_maximum_message_bytes=terminal._MAX_PIPE_MESSAGE_BYTES + 1,
            inbound_maximum_message_bytes=1024,
            custody_exchange_timeout_ms=1000,
        )


def test_terminal_authority_projection_is_exact_and_self_hashed() -> None:
    projection = terminal._terminal_duplex_authority_projection()
    unsigned = {key: value for key, value in projection.items() if key != "template_root_sha256"}
    assert projection["template_root_sha256"] == terminal._canonical_sha256(unsigned)
    assert projection["message_sequence"] == [
        terminal.CLAIM_READY_MESSAGE_TYPE,
        terminal.CUSTODY_GRANT_MESSAGE_TYPE,
        terminal.COMPOSED_READY_MESSAGE_TYPE,
        terminal.FINAL_ACK_MESSAGE_TYPE,
    ]
    for message_type, fields in (
        (terminal.CLAIM_READY_MESSAGE_TYPE, terminal._CLAIM_READY_FIELDS),
        (terminal.CUSTODY_GRANT_MESSAGE_TYPE, terminal._CUSTODY_GRANT_FIELDS),
        (terminal.COMPOSED_READY_MESSAGE_TYPE, terminal._COMPOSED_READY_FIELDS),
        (terminal.FINAL_ACK_MESSAGE_TYPE, terminal._FINAL_ACK_FIELDS),
    ):
        assert set(projection["message_contracts"][message_type]["field_names"]) == fields
    assert (
        set(projection["readback_contract"]["field_names"])
        == terminal._POSTWAKE_COMPOSED_READBACK_FIELDS
    )
    transport = projection["transport_contract"]
    assert (
        transport["terminal_client_arrival_timeout_ms"]
        == authority.TERMINAL_CLIENT_ARRIVAL_TIMEOUT_MS
    )
    assert transport["custody_exchange_timeout_ms"] == authority.CUSTODY_EXCHANGE_TIMEOUT_MS
    assert (
        transport["terminal_client_arrival_timeout_ms"] > transport["custody_exchange_timeout_ms"]
    )
    assert terminal._canonical_terminal_duplex_authority_projection(projection) == projection
    mutated = dict(projection)
    mutated["message_sequence"] = list(reversed(projection["message_sequence"]))
    with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
        terminal._canonical_terminal_duplex_authority_projection(mutated)


def test_terminal_duplex_reads_launch_intent_only_after_grant() -> None:
    source = inspect.getsource(terminal._run_terminal_duplex)
    claim_send = source.index("pipe.send(claim_ready)")
    grant_receive = source.index("grant_raw, _grant_line = pipe.receive()")
    grant_validate = source.index("custody_grant = _validate_custody_grant")
    launch_intent_open = source.index("_open_validated_terminal_client_launch_intent")
    lease_open = source.index("lease_descriptor = _duplicate_from_process")
    assert claim_send < grant_receive < grant_validate < launch_intent_open < lease_open


def test_verified_context_projects_exact_staged_supervisor_inputs_from_q() -> None:
    source = inspect.getsource(terminal._load_verified_context)

    control_staging_lookup = 'q["control_staging_projection"]'
    launch_spec_lookup = 'control_staging_projection["supervisor_launch_spec_path"]'
    staged_e_lookup = 'control_staging_projection["e_intent_path"]'
    projection_call = "authority.build_original_confirmatory_supervisor_process_command_projection"
    assert control_staging_lookup in source
    assert launch_spec_lookup in source
    assert staged_e_lookup in source
    assert "supervisor_launch_spec_path=supervisor_launch_spec_path" in source
    assert "staged_e_intent_path=staged_e_intent_path" in source
    assert "supervisor_spec_path=spec_path" not in source
    assert source.index(control_staging_lookup) < source.index(projection_call)


@pytest.mark.parametrize(
    "forbidden_selector",
    [
        "metrics.auroc",
        "ranking.0.score",
        "outcome.value",
        "predictions",
        "p_value",
        "restoration.statistics",
    ],
)
def test_outcome_blind_artifact_rule_mutations_stop_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forbidden_selector: str,
) -> None:
    e_intent, spec = _make_outcome_blind_artifact_inputs(tmp_path)
    mutated = {
        **spec["expected_artifacts"][0],
        "json_equals": {
            **spec["expected_artifacts"][0]["json_equals"],
            forbidden_selector: True,
        },
    }
    spec["expected_artifacts"][0] = mutated
    read_count = 0

    def forbidden_read(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal read_count
        read_count += 1
        raise AssertionError("mutated selector reached artifact read")

    monkeypatch.setattr(terminal, "_open_held_file", forbidden_read)
    with pytest.raises(
        terminal.OriginalConfirmatoryTerminalError,
        match="outcome-blind allowlist",
    ):
        terminal._canonical_outcome_blind_expected_artifact_rules(
            spec=spec,
            e_intent=e_intent,
        )
    assert read_count == 0


def test_outcome_blind_artifact_bool_does_not_accept_integer(
    tmp_path: Path,
) -> None:
    e_intent, spec = _make_outcome_blind_artifact_inputs(tmp_path)
    completion = dict(spec["expected_artifacts"][2])
    completion["json_equals"] = {
        **completion["json_equals"],
        "study_outcome_eligible": 1,
    }
    spec["expected_artifacts"][2] = completion
    with pytest.raises(
        terminal.OriginalConfirmatoryTerminalError,
        match="outcome-blind allowlist",
    ):
        terminal._canonical_outcome_blind_expected_artifact_rules(
            spec=spec,
            e_intent=e_intent,
        )


def test_preterminal_pin_reconstructs_all_bindings_and_evidence(
    tmp_path: Path,
) -> None:
    context, consumed, pin = _make_pin_validation_bundle(tmp_path)
    assert (
        terminal._validate_preterminal_pin_payload(
            pin,
            context=context,
            consumed_e=consumed,
        )
        == pin
    )
    mutations: list[tuple[str, Any]] = [
        ("capsule_contract_sha256", _HASH_D),
        ("e_intent_path", str((tmp_path / "other-e.json").resolve())),
        ("capsule_mode", authority.CAPSULE_TERMINAL_MODE),
    ]
    for field, replacement in mutations:
        mutated = copy.deepcopy(pin)
        mutated[field] = replacement
        _resign_pin(mutated)
        with pytest.raises(
            terminal.OriginalConfirmatoryTerminalError,
            match="exact policy",
        ):
            terminal._validate_preterminal_pin_payload(
                mutated,
                context=context,
                consumed_e=consumed,
            )

    mutated_evidence = copy.deepcopy(pin)
    mutated_evidence["expected_artifact_evidence"][0]["path"] = str(
        (tmp_path / "forbidden-artifact.json").resolve()
    )
    _resign_pin(mutated_evidence)
    with pytest.raises(
        terminal.OriginalConfirmatoryTerminalError,
        match="artifact evidence differs",
    ):
        terminal._validate_preterminal_pin_payload(
            mutated_evidence,
            context=context,
            consumed_e=consumed,
        )

    empty_evidence = copy.deepcopy(pin)
    empty_evidence["expected_artifact_evidence"][0]["size_bytes"] = 0
    _resign_pin(empty_evidence)
    with pytest.raises(
        terminal.OriginalConfirmatoryTerminalError,
        match="artifact evidence differs",
    ):
        terminal._validate_preterminal_pin_payload(
            empty_evidence,
            context=context,
            consumed_e=consumed,
        )

    mutated_roles = copy.deepcopy(pin)
    mutated_roles["required_success_roles"] = list(reversed(terminal._EXPECTED_ARTIFACT_ROLE_ORDER))
    _resign_pin(mutated_roles)
    with pytest.raises(
        terminal.OriginalConfirmatoryTerminalError,
        match="required artifact roles differ",
    ):
        terminal._validate_preterminal_pin_payload(
            mutated_roles,
            context=context,
            consumed_e=consumed,
        )


def test_preterminal_pin_preserves_exact_successor_lineage(
    tmp_path: Path,
) -> None:
    context, consumed, pin = _make_pin_validation_bundle(tmp_path)
    context.execution_mode = "successor_resume"
    context.retry_of_run_id = "failed-run-185-cells"
    pin["execution_mode"] = context.execution_mode
    pin["retry_of_run_id"] = context.retry_of_run_id
    _resign_pin(pin)
    assert (
        terminal._validate_preterminal_pin_payload(
            pin,
            context=context,
            consumed_e=consumed,
        )
        == pin
    )

    mismatched = copy.deepcopy(pin)
    mismatched["retry_of_run_id"] = "another-run"
    _resign_pin(mismatched)
    with pytest.raises(
        terminal.OriginalConfirmatoryTerminalError,
        match="exact policy",
    ):
        terminal._validate_preterminal_pin_payload(
            mismatched,
            context=context,
            consumed_e=consumed,
        )


def test_supervisor_terminal_rejects_semantic_mutations_across_all_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, consumed, pin, launch, started, valid = _make_terminal_validation_bundle(tmp_path)

    def decode(held: Any, *, role: str) -> tuple[dict[str, Any], str]:
        del role
        if held is consumed.launch_intent:
            return launch, held.file_sha256
        assert held is consumed.process_started
        return started, held.file_sha256

    monkeypatch.setattr(terminal, "_decode_supervisor_envelope", decode)
    monkeypatch.setattr(
        terminal,
        "_validate_launch_intent_payload",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        terminal,
        "_validate_process_started_payload",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        terminal,
        "_validate_supervisor_verifier_record",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        terminal,
        "_validate_terminal_e_consumption_binding",
        lambda _value, **_kwargs: None,
    )
    assert (
        terminal._validate_supervisor_terminal(
            valid,
            context=context,
            consumed_e=consumed,
            pin=pin,
            source_files={},
            bindings=[],
        )
        == valid
    )

    def set_environment_mismatch(value: dict[str, Any]) -> None:
        value["environment_binding"]["all_exact_matches"] = False

    def set_log_mismatch(value: dict[str, Any]) -> None:
        value["stdout"]["path"] = str(context.job_dir / "other.log")

    def set_silent_log_discard(value: dict[str, Any]) -> None:
        value["stdout"]["discarded_bytes"] = 1
        value["stdout"]["stream_size_bytes"] = value["stdout"]["size_bytes"] + 1
        value["stdout"]["stream_sha256"] = _HASH_A

    def set_artifact_mismatch(value: dict[str, Any]) -> None:
        value["expected_artifacts"][0]["sha256"] = _HASH_A

    def set_empty_artifact(value: dict[str, Any]) -> None:
        value["expected_artifacts"][0]["size_bytes"] = 0

    mutators = [
        lambda value: value.__setitem__("schema_version", 2),
        lambda value: value.__setitem__("schema_version", True),
        lambda value: value.__setitem__("schema_version", 3.0),
        lambda value: value.__setitem__("attempt_count", True),
        lambda value: value.__setitem__("attempt_count", 1.0),
        lambda value: value.__setitem__("exit_code", False),
        lambda value: value.__setitem__("exit_code", 0.0),
        lambda value: value.__setitem__("descendants_after_root_exit", False),
        lambda value: value.__setitem__("descendants_after_root_exit", 0.0),
        lambda value: value.__setitem__("attempt_policy", "automatic-retry"),
        lambda value: value.__setitem__("process_kind", "primary"),
        lambda value: value.__setitem__(
            "launch_intent_receipt_sha256",
            _HASH_D,
        ),
        lambda value: value.__setitem__("descendants_after_root_exit", 1),
        lambda value: value.__setitem__("recovery_evidence", {}),
        lambda value: value.__setitem__(
            "ended_at_utc",
            "2026-07-29T00:00:00.000000Z",
        ),
        lambda value: value.__setitem__(
            "capsule_lease_identity_root_sha256",
            _HASH_D,
        ),
        lambda value: value.__setitem__(
            "python_runtime_resolution_policy",
            "unbound-runtime-policy",
        ),
        lambda value: value.__setitem__(
            "e_consumption_contract_sha256",
            _HASH_A,
        ),
        set_environment_mismatch,
        set_log_mismatch,
        set_silent_log_discard,
        set_artifact_mismatch,
        set_empty_artifact,
    ]
    for mutate in mutators:
        mutated = copy.deepcopy(valid)
        mutate(mutated)
        with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
            terminal._validate_supervisor_terminal(
                mutated,
                context=context,
                consumed_e=consumed,
                pin=pin,
                source_files={},
                bindings=[],
            )

    unknown = copy.deepcopy(valid)
    unknown["unexpected_terminal_field"] = True
    with pytest.raises(
        terminal.OriginalConfirmatoryTerminalError,
        match="fields differ",
    ):
        terminal._validate_supervisor_terminal(
            unknown,
            context=context,
            consumed_e=consumed,
            pin=pin,
            source_files={},
            bindings=[],
        )


def test_integrity_verifier_record_rejects_nested_semantic_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _consumed, _pin, _launch, _started, _terminal_receipt = (
        _make_terminal_validation_bundle(tmp_path)
    )
    command = {
        "program_path": str(Path(sys.executable).resolve()),
        "program_sha256": _HASH_A,
        "argv": [str(Path(sys.executable).resolve()), "-m", "synthetic"],
        "cwd": str(tmp_path.resolve()),
        "command_sha256": _HASH_B,
    }
    context.preterminal_command = SimpleNamespace(
        program_path=Path(command["program_path"]),
        program_sha256=command["program_sha256"],
        command_sha256=command["command_sha256"],
        as_dict=lambda: command,
    )
    context.observed_environment_sha256 = _HASH_D
    context.terminal_contract = SimpleNamespace(
        verifier_stdout_path=context.job_dir / "verifier.stdout.log",
        verifier_stdout_max_bytes=context.spec["max_log_bytes"],
        verifier_stderr_path=context.job_dir / "verifier.stderr.log",
        verifier_stderr_max_bytes=context.spec["max_log_bytes"],
    )
    payload = b"synthetic-log"
    payload_sha = hashlib.sha256(payload).hexdigest()
    source_files = {
        "preterminal-stdout": SimpleNamespace(
            path=context.terminal_contract.verifier_stdout_path,
            payload=payload,
            file_sha256=payload_sha,
            revalidate=lambda: None,
        ),
        "preterminal-stderr": SimpleNamespace(
            path=context.terminal_contract.verifier_stderr_path,
            payload=payload,
            file_sha256=payload_sha,
            revalidate=lambda: None,
        ),
    }
    handshake = {"synthetic": "already-deep-validated"}
    monkeypatch.setattr(
        terminal,
        "_validate_preterminal_handshake_receipt",
        lambda value, **_kwargs: value,
    )
    record = {
        "command": command,
        "process_identity": {
            **_process_identity(1003),
            "program_path": command["program_path"],
            "program_sha256": command["program_sha256"],
            "command_sha256": command["command_sha256"],
        },
        "started_at_utc": "2026-07-30T00:00:00.000000Z",
        "ended_at_utc": "2026-07-30T00:00:01.000000Z",
        "timeout_ms": context.spec["verifier_timeout_ms"],
        "job_assignment_mode": "PROC_THREAD_ATTRIBUTE_JOB_LIST",
        "atomic_job_assignment": True,
        "handle_list_restricted": True,
        "job_handle_inherited": False,
        "exit_code": 0,
        "descendants_after_root_exit": 0,
        "stdout": _valid_log_record(
            context.terminal_contract.verifier_stdout_path,
            maximum_bytes=context.spec["max_log_bytes"],
        ),
        "stderr": _valid_log_record(
            context.terminal_contract.verifier_stderr_path,
            maximum_bytes=context.spec["max_log_bytes"],
        ),
        "error_type": None,
        "error_sha256": None,
        "cleanup_error_type": None,
        "cleanup_error_sha256": None,
        "tree_empty_verified": True,
        "valid": True,
        "capsule_contract_sha256": context.q["execution_capsule"]["contract_sha256"],
        "capsule_sha256": context.q["execution_capsule"]["sha256"],
        "capsule_internal_manifest_sha256": context.q["execution_capsule"][
            "internal_manifest_sha256"
        ],
        "capsule_mode": authority.CAPSULE_PRETERMINAL_MODE,
        "expected_environment_envelope_sha256": context.spec["expected_environment"][
            "envelope_sha256"
        ],
        "process_environment_binding_sha256": context.spec["process_environment_binding"][
            "binding_sha256"
        ],
        "exact_integrity_verifier_environment_sha256": _HASH_D,
        "observed_integrity_verifier_environment_sha256": _HASH_D,
        "integrity_verifier_environment_exact_match": True,
        "integrity_verifier_environment_observation_method": ("windows_peb_process_parameters_v1"),
        "preterminal_pin_contract_sha256": (context.preterminal_contract.contract_sha256),
        "preterminal_overlap_handshake_contract_sha256": (context.overlap_contract.contract_sha256),
        "preterminal_overlap_handshake_receipt": handshake,
        "capsule_lease_identity_root_sha256": context.spec["capsule_lease_identity_root_sha256"],
        "capsule_ancestor_lease_root_sha256": context.spec["capsule_ancestor_lease_root_sha256"],
        "python_lease_identity_root_sha256": context.spec["python_lease_identity_root_sha256"],
        "python_ancestor_lease_root_sha256": context.spec["python_ancestor_lease_root_sha256"],
        "interpreter_leaf_handle_active": True,
        "interpreter_ancestor_handles_active": True,
        "python_runtime_resolution_policy": context.spec["python_runtime_resolution_policy"],
        "runtime_python_lease_identity_root_sha256": context.spec[
            "runtime_python_lease_identity_root_sha256"
        ],
        "runtime_python_ancestor_lease_root_sha256": context.spec[
            "runtime_python_ancestor_lease_root_sha256"
        ],
        "runtime_interpreter_leaf_handle_active": True,
        "runtime_interpreter_ancestor_handles_active": True,
    }
    assert set(record) == terminal._SUPERVISOR_VERIFIER_RECORD_FIELDS
    assert (
        terminal._validate_supervisor_verifier_record(
            record,
            context=context,
            pin={},
            source_files=source_files,
            bindings=[],
        )
        == record
    )
    mutators = [
        lambda value: value.__setitem__("valid", False),
        lambda value: value.__setitem__("exit_code", 1),
        lambda value: value.__setitem__(
            "observed_integrity_verifier_environment_sha256",
            _HASH_A,
        ),
        lambda value: value.__setitem__("capsule_sha256", _HASH_D),
        lambda value: value.__setitem__(
            "runtime_interpreter_leaf_handle_active",
            False,
        ),
        lambda value: value["stdout"].__setitem__(
            "capture_complete",
            False,
        ),
    ]
    for mutate in mutators:
        mutated = copy.deepcopy(record)
        mutate(mutated)
        with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
            terminal._validate_supervisor_verifier_record(
                mutated,
                context=context,
                pin={},
                source_files=source_files,
                bindings=[],
            )


def test_exact_launch_and_process_started_schemas_and_semantics(
    tmp_path: Path,
) -> None:
    context, _consumed, _pin, _launch, _started, _terminal_receipt = (
        _make_terminal_validation_bundle(tmp_path)
    )
    context.scientific_command = SimpleNamespace(command_sha256=_HASH_B)
    context.spec.update(
        {
            "program_path": str(Path(sys.executable).resolve()),
            "program_sha256": _HASH_A,
            "argv": [str(Path(sys.executable).resolve()), "-m", "synthetic"],
            "command": {
                "program_path": str(Path(sys.executable).resolve()),
                "program_sha256": _HASH_A,
                "argv": [str(Path(sys.executable).resolve()), "-m", "synthetic"],
                "cwd": str(tmp_path.resolve()),
                "command_sha256": _HASH_B,
            },
            "main_timeout_ms": 86_400_000,
            "supervisor_launcher_sha256": _HASH_D,
        }
    )
    context.expected_artifact_rules = tuple(
        {
            "role": role,
            "path": str(context.job_dir / f"expected-{index}.json"),
            "must_be_absent_before": index < 3,
        }
        for index, role in enumerate(terminal._EXPECTED_ARTIFACT_ROLE_ORDER)
    )
    supervisor_identity = {
        **_process_identity(1001),
        "command_sha256": _HASH_D,
    }
    context.supervisor_process_command_projection = {
        "expected_live_image_path": supervisor_identity["program_path"],
        "expected_live_image_sha256": supervisor_identity["program_sha256"],
        "command_sha256": supervisor_identity["command_sha256"],
    }
    prearm = {
        "schema_version": 3,
        "policy": "exact_argv_and_protected_process_absence_v1",
        "observed_at_utc": "2026-07-30T00:00:00.000000Z",
        "inventory_process_count": 1,
        "target_program_path": context.spec["program_path"],
        "target_program_sha256": context.spec["program_sha256"],
        "target_command_sha256": context.scientific_command.command_sha256,
        "target_argv_sha256": terminal._canonical_sha256(context.spec["argv"]),
        "exact_command_matches": [],
        "protected_marker_matches": [],
        "absence_verified": True,
    }
    binding = context.spec["process_environment_binding"]
    expected_environment = context.spec["expected_environment"]
    launch = {
        "schema_version": 3,
        "policy": authority.SUPERVISOR_V3_POLICY,
        "attempt_policy": terminal._SUPERVISOR_ATTEMPT_POLICY,
        "job_id": context.job_id,
        "spec_sha256": context.spec_sha256,
        "command": context.spec["command"],
        "command_sha256": context.scientific_command.command_sha256,
        "attempt_nonce": context.attempt_nonce,
        "attempt_count": 1,
        "max_attempt_count": 1,
        "automatic_retry_allowed": False,
        "job_assignment_mode": "PROC_THREAD_ATTRIBUTE_JOB_LIST",
        "handle_list_restricted": True,
        "job_handle_inherited": False,
        "supervisor_process_identity": supervisor_identity,
        "main_timeout_ms": context.spec["main_timeout_ms"],
        "windows_boot_time_utc": "2026-07-29T00:00:00.000000Z",
        "prearm_process_absence": prearm,
        "prelaunch_artifacts": [
            {
                "role": rule["role"],
                "path": rule["path"],
                "exists": False,
            }
            for rule in context.expected_artifact_rules
        ],
        "created_at_utc": "2026-07-30T00:00:01.000000Z",
        "supervisor_launcher_sha256": context.spec["supervisor_launcher_sha256"],
        "expected_environment_envelope_sha256": expected_environment["envelope_sha256"],
        "launch_environment_root_sha256": expected_environment["launch_environment_root_sha256"],
        "process_environment_binding_sha256": binding["binding_sha256"],
        "exact_supervisor_environment_sha256": binding["exact_supervisor_environment_sha256"],
        "exact_environment_sha256": binding["exact_environment_sha256"],
        "exact_integrity_verifier_environment_sha256": binding[
            "exact_integrity_verifier_environment_sha256"
        ],
        "observed_supervisor_environment_sha256": binding["exact_supervisor_environment_sha256"],
        "supervisor_environment_exact_match": True,
        "capsule_lease_identity": context.spec["capsule_lease_identity"],
        "capsule_lease_identity_root_sha256": context.spec["capsule_lease_identity_root_sha256"],
        "capsule_ancestor_lease": context.spec["capsule_ancestor_lease"],
        "capsule_ancestor_lease_root_sha256": context.spec["capsule_ancestor_lease_root_sha256"],
        "preterminal_overlap_handshake_contract_sha256": (context.overlap_contract.contract_sha256),
        "python_lease_identity": context.spec["python_lease_identity"],
        "python_lease_identity_root_sha256": context.spec["python_lease_identity_root_sha256"],
        "python_ancestor_lease": context.spec["python_ancestor_lease"],
        "python_ancestor_lease_root_sha256": context.spec["python_ancestor_lease_root_sha256"],
        "python_runtime_resolution_policy": context.spec["python_runtime_resolution_policy"],
        "runtime_python_lease_identity": context.spec["runtime_python_lease_identity"],
        "runtime_python_lease_identity_root_sha256": context.spec[
            "runtime_python_lease_identity_root_sha256"
        ],
        "runtime_python_ancestor_lease": context.spec["runtime_python_ancestor_lease"],
        "runtime_python_ancestor_lease_root_sha256": context.spec[
            "runtime_python_ancestor_lease_root_sha256"
        ],
        "e_consumption_contract_sha256": context.e["e_consumption_contract"]["contract_sha256"],
    }
    assert set(launch) == terminal._LAUNCH_INTENT_FIELDS
    assert (
        terminal._validate_launch_intent_payload(
            launch,
            context=context,
        )
        == launch
    )
    launch_file_sha256 = _HASH_A
    started = {
        "schema_version": 3,
        "policy": authority.SUPERVISOR_V3_POLICY,
        "job_id": context.job_id,
        "spec_sha256": context.spec_sha256,
        "launch_intent_sha256": launch_file_sha256,
        "attempt_nonce": context.attempt_nonce,
        "process_identity": {
            **_process_identity(1002),
            "program_path": context.spec["program_path"],
            "program_sha256": context.spec["program_sha256"],
            "command_sha256": context.scientific_command.command_sha256,
        },
        "job_assignment_mode": "PROC_THREAD_ATTRIBUTE_JOB_LIST",
        "atomic_job_assignment": True,
        "handle_list_restricted": True,
        "job_handle_inherited": False,
        "supervisor_process_identity": supervisor_identity,
        "main_timeout_ms": context.spec["main_timeout_ms"],
        "stdout_partial_path": str(context.job_dir / "stdout.partial"),
        "stderr_partial_path": str(context.job_dir / "stderr.partial"),
        "windows_boot_time_utc": launch["windows_boot_time_utc"],
        "started_at_utc": "2026-07-30T00:00:02.000000Z",
        "attempt_count": 1,
        "automatic_retry_allowed": False,
        "expected_environment_envelope_sha256": expected_environment["envelope_sha256"],
        "launch_environment_root_sha256": expected_environment["launch_environment_root_sha256"],
        "process_environment_binding_sha256": binding["binding_sha256"],
        "exact_supervisor_environment_sha256": binding["exact_supervisor_environment_sha256"],
        "observed_supervisor_environment_sha256": binding["exact_supervisor_environment_sha256"],
        "exact_environment_sha256": binding["exact_environment_sha256"],
        "observed_child_environment_sha256": binding["exact_environment_sha256"],
        "exact_integrity_verifier_environment_sha256": binding[
            "exact_integrity_verifier_environment_sha256"
        ],
        "child_environment_exact_match": True,
        "child_environment_observation_method": ("windows_peb_process_parameters_v1"),
        "capsule_lease_identity_root_sha256": context.spec["capsule_lease_identity_root_sha256"],
        "capsule_ancestor_lease_root_sha256": context.spec["capsule_ancestor_lease_root_sha256"],
        "preterminal_overlap_handshake_contract_sha256": (context.overlap_contract.contract_sha256),
        "python_lease_identity_root_sha256": context.spec["python_lease_identity_root_sha256"],
        "python_ancestor_lease_root_sha256": context.spec["python_ancestor_lease_root_sha256"],
        "interpreter_leaf_handle_active": True,
        "interpreter_ancestor_handles_active": True,
        "python_runtime_resolution_policy": context.spec["python_runtime_resolution_policy"],
        "runtime_python_lease_identity_root_sha256": context.spec[
            "runtime_python_lease_identity_root_sha256"
        ],
        "runtime_python_ancestor_lease_root_sha256": context.spec[
            "runtime_python_ancestor_lease_root_sha256"
        ],
        "runtime_interpreter_leaf_handle_active": True,
        "runtime_interpreter_ancestor_handles_active": True,
        "e_consumption_contract_sha256": context.e["e_consumption_contract"]["contract_sha256"],
    }
    assert set(started) == terminal._PROCESS_STARTED_FIELDS
    assert (
        terminal._validate_process_started_payload(
            started,
            context=context,
            launch=launch,
            launch_file_sha256=launch_file_sha256,
        )
        == started
    )

    for record, validator, kwargs in (
        (
            launch,
            terminal._validate_launch_intent_payload,
            {"context": context},
        ),
        (
            started,
            terminal._validate_process_started_payload,
            {
                "context": context,
                "launch": launch,
                "launch_file_sha256": launch_file_sha256,
            },
        ),
    ):
        unknown = copy.deepcopy(record)
        unknown["unexpected"] = True
        with pytest.raises(
            terminal.OriginalConfirmatoryTerminalError,
            match="fields differ",
        ):
            validator(unknown, **kwargs)

    mutated_launch = copy.deepcopy(launch)
    mutated_launch["supervisor_environment_exact_match"] = False
    with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
        terminal._validate_launch_intent_payload(
            mutated_launch,
            context=context,
        )
    mutated_started = copy.deepcopy(started)
    mutated_started["runtime_interpreter_leaf_handle_active"] = False
    with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
        terminal._validate_process_started_payload(
            mutated_started,
            context=context,
            launch=launch,
            launch_file_sha256=launch_file_sha256,
        )

    launch_integer_mutations = [
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", 3.0),
        ("attempt_count", True),
        ("attempt_count", 1.0),
        ("max_attempt_count", True),
        ("max_attempt_count", 1.0),
    ]
    for field, invalid_value in launch_integer_mutations:
        mutated = copy.deepcopy(launch)
        mutated[field] = invalid_value
        with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
            terminal._validate_launch_intent_payload(mutated, context=context)

    process_integer_mutations = [
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", 3.0),
        ("attempt_count", True),
        ("attempt_count", 1.0),
    ]
    for field, invalid_value in process_integer_mutations:
        mutated = copy.deepcopy(started)
        mutated[field] = invalid_value
        with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
            terminal._validate_process_started_payload(
                mutated,
                context=context,
                launch=launch,
                launch_file_sha256=launch_file_sha256,
            )


@pytest.mark.skipif(os.name != "nt", reason="Windows immutable artifact evidence")
def test_expected_artifact_evidence_reads_only_exact_control_selectors(
    tmp_path: Path,
) -> None:
    e_intent, spec = _make_outcome_blind_artifact_inputs(tmp_path)
    expected = terminal._canonical_outcome_blind_expected_artifact_rules(
        spec=spec,
        e_intent=e_intent,
    )
    rule = expected[0]
    artifact_path = Path(rule["path"])
    artifact_path.parent.mkdir(parents=True)
    _readonly(
        artifact_path,
        terminal._canonical_bytes(
            {
                "run_id": "run-1",
                "status": "completed",
                "unselected_scientific_payload": {
                    "metrics": [0.1, 0.2],
                    "ranking": [2, 1],
                },
            }
        ),
    )
    try:
        evidence = terminal._inspect_expected_artifact(
            rule,
            expected_rule=rule,
        )
        assert evidence["valid"] is True
        assert evidence["json_control_paths_checked"] == [
            "run_id",
            "status",
        ]
        assert evidence["sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    finally:
        artifact_path.chmod(stat.S_IWRITE)


@pytest.mark.skipif(os.name != "nt", reason="Windows CREATE_NEW identity policy")
def test_composed_claim_is_create_new_readonly_and_never_overwritten(
    tmp_path: Path,
) -> None:
    path = tmp_path / "composed_terminal.json"
    descriptor = terminal._create_new_readonly_descriptor(path)
    try:
        identity = terminal._claim_identity(descriptor, path=path)
        assert set(identity) == terminal._CLAIM_PHYSICAL_IDENTITY_FIELDS
        assert identity["policy"] == terminal.COMPOSED_CLAIM_PHYSICAL_IDENTITY_POLICY
        assert identity["size_bytes"] == 0
        assert identity["sha256"] == _EMPTY_SHA256
        with pytest.raises(OSError):
            terminal._create_new_readonly_descriptor(path)
        terminal._write_same_handle(
            descriptor,
            b'{"synthetic":"composed"}\n',
            maximum_bytes=1024,
        )
        final_identity = terminal._physical_identity(
            descriptor,
            path=path,
            role="composed-terminal",
        )
        assert final_identity["file_id_128"] == identity["file_id_128"]
        assert final_identity["volume_serial_number"] == identity["volume_serial_number"]
    finally:
        os.close(descriptor)
        path.chmod(stat.S_IWRITE)


@pytest.mark.skipif(os.name != "nt", reason="Windows retained-handle overlap")
def test_preterminal_ready_ack_holds_same_create_new_handle(
    tmp_path: Path,
) -> None:
    context = _make_preterminal_context(tmp_path)
    pin = {
        "evidence_root_sha256": _HASH_A,
        "preterminal_scientific_core_sha256": _HASH_B,
        "synthetic_control_only": True,
    }
    ready_queue: queue.Queue[bytes] = queue.Queue()
    ack_queue: queue.Queue[bytes] = queue.Queue()
    errors: queue.Queue[BaseException] = queue.Queue()
    ack_completed = threading.Event()
    writer = _QueueWriter(ready_queue)

    def supervisor() -> None:
        held: Any = None
        try:
            ready_line = ready_queue.get(timeout=10)
            ready = terminal._decode_canonical_line(
                ready_line,
                role="synthetic preterminal READY",
                maximum_bytes=1024 * 1024,
            )
            pin_path = context.preterminal_contract.preterminal_pin_receipt_path
            supervisor_descriptor = terminal._open_live_writer_overlap_read_descriptor(pin_path)
            held = terminal._HeldFile(
                path=pin_path,
                descriptor=supervisor_descriptor,
                payload=terminal._read_descriptor(
                    supervisor_descriptor,
                    maximum_bytes=1024 * 1024,
                ),
                role="synthetic supervisor-held P",
            )
            held.revalidate()
            supervisor_identity = terminal._physical_identity(
                held.descriptor,
                path=pin_path,
                role="preterminal-pin",
            )
            pin_identity = ready["preterminal_pin_receipt"]
            supervisor_identity = {
                **supervisor_identity,
                "policy": terminal.PRETERMINAL_SUPERVISOR_OBSERVER_IDENTITY_POLICY,
                "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
            }
            stable_projection = terminal._stable_physical_identity_projection(pin_identity)
            assert (
                terminal._stable_physical_identity_projection(supervisor_identity)
                == stable_projection
            )
            pin_root = terminal._canonical_sha256(pin_identity)
            ack = {
                "schema_version": 1,
                "policy": terminal.PRETERMINAL_ACK_POLICY,
                "message_type": terminal.PRETERMINAL_ACK_MESSAGE_TYPE,
                "job_id": context.job_id,
                "attempt_id": context.attempt_id,
                "run_id": context.run_id,
                "execution_mode": context.execution_mode,
                "retry_of_run_id": context.retry_of_run_id,
                "attempt_nonce": context.attempt_nonce,
                "preterminal_pin_contract_sha256": (context.preterminal_contract.contract_sha256),
                "preterminal_overlap_handshake_contract_sha256": (
                    context.overlap_contract.contract_sha256
                ),
                "ready_line_sha256": hashlib.sha256(ready_line).hexdigest(),
                "child_reported_pin_identity": pin_identity,
                "child_reported_pin_identity_root_sha256": pin_root,
                "supervisor_opened_pin_identity": supervisor_identity,
                "supervisor_opened_pin_identity_root_sha256": (
                    terminal._canonical_sha256(supervisor_identity)
                ),
                "stable_physical_identity_projection": stable_projection,
                "stable_physical_identity_root_sha256": (
                    terminal._canonical_sha256(stable_projection)
                ),
                "identity_exact_match": True,
                "pin_handle_overlap_verified": True,
                "automatic_retry_allowed": False,
                "acknowledged_at_utc": "2026-07-30T00:00:02.000000Z",
            }
            ack_queue.put(terminal._canonical_bytes(ack))
            if not ack_completed.wait(timeout=10):
                raise AssertionError("child did not complete ACK while P remained held")
        except BaseException as exc:
            errors.put(exc)
        finally:
            if held is not None:
                held.close()

    thread = threading.Thread(target=supervisor, daemon=True)
    thread.start()
    try:
        terminal._run_preterminal_handshake(
            context,
            pin=pin,
            stdin=_Stream(_QueueReader(ack_queue)),
            stdout=_Stream(writer),
        )
    finally:
        ack_completed.set()
        thread.join(timeout=10)
        context.preterminal_contract.preterminal_pin_receipt_path.chmod(stat.S_IWRITE)
    assert not thread.is_alive()
    assert errors.empty(), list(errors.queue)
    assert writer.write_count == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows retained-handle overlap")
def test_invalid_preterminal_ack_keeps_nonretriable_create_new_artifact(
    tmp_path: Path,
) -> None:
    context = _make_preterminal_context(tmp_path)
    pin = {
        "evidence_root_sha256": _HASH_A,
        "preterminal_scientific_core_sha256": _HASH_B,
        "synthetic_control_only": True,
    }
    invalid_ack = terminal._canonical_bytes(
        {
            "schema_version": 1,
            "policy": "invalid",
            "automatic_retry_allowed": False,
        }
    )
    reader = _FixedReader(invalid_ack)
    writer = _CaptureWriter()
    pin_path = context.preterminal_contract.preterminal_pin_receipt_path
    try:
        with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
            terminal._run_preterminal_handshake(
                context,
                pin=pin,
                stdin=_Stream(reader),
                stdout=_Stream(writer),
            )
        assert reader.read_count == 1
        assert len(writer.payloads) == 1
        assert pin_path.read_bytes() == terminal._canonical_bytes(pin)
        with pytest.raises(OSError):
            terminal._create_new_readonly_descriptor(pin_path)
    finally:
        if pin_path.exists():
            pin_path.chmod(stat.S_IWRITE)


@pytest.mark.skipif(os.name != "nt", reason="Windows terminal file identities")
def test_closed_claim_grant_composed_readback_and_final_ack_schemas(
    tmp_path: Path,
) -> None:
    context = _make_terminal_context(tmp_path)
    composed_descriptor = terminal._create_new_readonly_descriptor(
        context.terminal_contract.composed_terminal_receipt_path
    )
    lease_path = tmp_path / "postwake_lease.json"
    lease_descriptor = terminal._create_new_readonly_descriptor(lease_path)
    launch_intent_path = Path(context.terminal_launcher_projection["launch_intent_path"])
    launch_intent_descriptor = terminal._create_new_readonly_descriptor(launch_intent_path)
    readback_descriptor = -1
    try:
        terminal._write_same_handle(
            lease_descriptor,
            b'{"synthetic":"lease"}\n',
            maximum_bytes=1024,
        )
        lease_identity = terminal._physical_identity(
            lease_descriptor,
            path=lease_path,
            role="postwake-lease-receipt",
        )
        terminal._write_same_handle(
            launch_intent_descriptor,
            b'{"synthetic":"terminal-client-launch-intent"}\n',
            maximum_bytes=1024,
        )
        launch_intent_identity = terminal._physical_identity(
            launch_intent_descriptor,
            path=launch_intent_path,
            role="terminal-client-launch-intent",
        )
        claim_identity = terminal._claim_identity(
            composed_descriptor,
            path=context.terminal_contract.composed_terminal_receipt_path,
        )
        client = _process_identity(os.getpid())
        redirector = _process_identity(os.getpid() + 1)
        launcher = {
            **_process_identity(os.getpid() + 2),
            "command_sha256": _HASH_D,
        }
        supervisor = {**client, "command_sha256": _HASH_C}
        claim_ready = terminal._build_claim_ready(
            context,
            consumed_e=_make_consumed_e_stub(tmp_path),
            client_identity=client,
            immediate_venv_redirector_identity=redirector,
            terminal_client_launcher_identity=launcher,
            supervisor_identity=supervisor,
            claim_identity=claim_identity,
            target_handle=4321,
        )
        assert set(claim_ready) == terminal._CLAIM_READY_FIELDS
        invalid_claim = dict(claim_ready)
        invalid_claim["e_consumption_claim_root_sha256"] = _HASH_A
        invalid_claim_unsigned = {
            key: value for key, value in invalid_claim.items() if key != "claim_ready_sha256"
        }
        invalid_claim["claim_ready_sha256"] = terminal._canonical_sha256(invalid_claim_unsigned)
        with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
            terminal._validate_claim_ready(
                invalid_claim,
                context=context,
                consumed_e=_make_consumed_e_stub(tmp_path),
                expected_client_identity=client,
                expected_immediate_venv_redirector_identity=redirector,
                expected_terminal_client_launcher_identity=launcher,
                expected_supervisor_identity=supervisor,
                expected_target_handle=4321,
            )
        for field, value in (
            ("immediate_venv_redirector_pid", redirector["pid"] + 100),
            (
                "immediate_venv_redirector_process_identity",
                {
                    **redirector,
                    "creation_time_100ns": (redirector["creation_time_100ns"] + 10_000_000),
                    "creation_time_utc": terminal._filetime_iso(
                        redirector["creation_time_100ns"] + 10_000_000
                    ),
                },
            ),
            (
                "terminal_client_launcher_process_identity",
                {**launcher, "command_sha256": _HASH_A},
            ),
            ("terminal_client_launch_intent_read", True),
        ):
            mutated_claim = {**claim_ready, field: value}
            mutated_unsigned = {
                key: item for key, item in mutated_claim.items() if key != "claim_ready_sha256"
            }
            mutated_claim["claim_ready_sha256"] = terminal._canonical_sha256(mutated_unsigned)
            with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
                terminal._validate_claim_ready(
                    mutated_claim,
                    context=context,
                    consumed_e=_make_consumed_e_stub(tmp_path),
                    expected_client_identity=client,
                    expected_immediate_venv_redirector_identity=redirector,
                    expected_terminal_client_launcher_identity=launcher,
                    expected_supervisor_identity=supervisor,
                    expected_target_handle=4321,
                )
        grant = terminal._build_custody_grant(
            context,
            claim_ready=claim_ready,
            launch_intent_file_sha256=launch_intent_identity["sha256"],
            launch_intent_root_sha256=_HASH_C,
            launch_intent_handle_slot=5678,
            launch_intent_identity=launch_intent_identity,
            lease_handle_slot=1234,
            lease_identity=lease_identity,
            granted_at_utc="2026-07-30T00:00:01.000000Z",
        )
        assert set(grant) == terminal._CUSTODY_GRANT_FIELDS
        assert (
            terminal._validate_custody_grant(
                grant,
                context=context,
                claim_ready=claim_ready,
            )
            == grant
        )
        for field in (
            "immediate_venv_redirector_process_identity_verified",
            "terminal_client_launcher_process_identity_verified",
            "launcher_redirector_child_grandparent_chain_verified",
            "launcher_redirector_child_same_supervisor_job_verified",
            "terminal_client_launch_intent_verified",
            "terminal_client_launch_intent_launcher_identity_verified",
            "terminal_client_launch_intent_create_new_before_child_verified",
            "terminal_client_launch_intent_supervisor_custody_active",
            "terminal_client_launch_intent_child_open_after_grant_required",
        ):
            invalid_grant = {**grant, field: False}
            invalid_grant_unsigned = {
                key: item for key, item in invalid_grant.items() if key != "custody_grant_sha256"
            }
            invalid_grant["custody_grant_sha256"] = terminal._canonical_sha256(
                invalid_grant_unsigned
            )
            with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
                terminal._validate_custody_grant(
                    invalid_grant,
                    context=context,
                    claim_ready=claim_ready,
                )
        for field, value in (
            ("terminal_client_launch_intent_file_sha256", _HASH_A),
            (
                "terminal_client_launch_intent_physical_identity_root_sha256",
                _HASH_A,
            ),
            (
                "terminal_client_launch_intent_supervisor_handle_slot",
                grant["postwake_input_lease_handle_slot"],
            ),
            (
                "terminal_client_launch_intent_supervisor_granted_access_mask",
                terminal.GENERIC_READ,
            ),
            (
                "terminal_client_launch_intent_child_duplicate_target_access_mask",
                terminal.FILE_GENERIC_READ_ACCESS_MASK,
            ),
            (
                "terminal_client_launch_intent_child_expected_granted_access_mask",
                terminal.GENERIC_READ,
            ),
        ):
            invalid_grant = {**grant, field: value}
            invalid_grant_unsigned = {
                key: item for key, item in invalid_grant.items() if key != "custody_grant_sha256"
            }
            invalid_grant["custody_grant_sha256"] = terminal._canonical_sha256(
                invalid_grant_unsigned
            )
            with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
                terminal._validate_custody_grant(
                    invalid_grant,
                    context=context,
                    claim_ready=claim_ready,
                )

        composed_receipt = {
            "receipt_root_sha256": _HASH_A,
            "source_inputs_root_sha256": _HASH_B,
            "terminal_client_launch_intent_path": grant["terminal_client_launch_intent_path"],
            "terminal_client_launch_intent_policy": grant["terminal_client_launch_intent_policy"],
            "terminal_client_launch_intent_file_sha256": grant[
                "terminal_client_launch_intent_file_sha256"
            ],
            "terminal_client_launch_intent_root_sha256": grant[
                "terminal_client_launch_intent_root_sha256"
            ],
            "terminal_client_launch_intent_physical_identity": grant[
                "terminal_client_launch_intent_physical_identity"
            ],
            "terminal_client_launch_intent_physical_identity_root_sha256": grant[
                "terminal_client_launch_intent_physical_identity_root_sha256"
            ],
            "terminal_client_launch_intent_supervisor_handle_slot": grant[
                "terminal_client_launch_intent_supervisor_handle_slot"
            ],
            "terminal_client_launch_intent_child_handle_slot": 8765,
            "terminal_client_launch_intent_child_granted_access_mask": (
                terminal.FILE_GENERIC_READ_ACCESS_MASK
            ),
        }
        terminal._write_same_handle(
            composed_descriptor,
            terminal._canonical_bytes(composed_receipt),
            maximum_bytes=1024 * 1024,
        )
        composed_identity = terminal._physical_identity(
            composed_descriptor,
            path=context.terminal_contract.composed_terminal_receipt_path,
            role="composed-terminal",
        )
        composed_ready = terminal._build_composed_ready(
            context,
            claim_ready=claim_ready,
            custody_grant=grant,
            receipt=composed_receipt,
            composed_identity=composed_identity,
            target_handle=4321,
        )
        assert set(composed_ready) == terminal._COMPOSED_READY_FIELDS
        for field, value in (
            ("terminal_client_launch_intent_file_sha256", _HASH_A),
            ("terminal_client_launch_intent_child_handle_slot", 0),
            (
                "terminal_client_launch_intent_child_granted_access_mask",
                terminal.GENERIC_READ,
            ),
            (
                "terminal_client_launch_intent_same_duplicated_supervisor_handle_used",
                False,
            ),
            (
                "terminal_client_launch_intent_physical_identity_exact_match",
                False,
            ),
            ("terminal_client_launch_intent_child_custody_active", False),
            ("terminal_client_launch_intent_supervisor_custody_active", False),
        ):
            invalid_ready = {**composed_ready, field: value}
            invalid_ready_unsigned = {
                key: item for key, item in invalid_ready.items() if key != "composed_ready_sha256"
            }
            invalid_ready["composed_ready_sha256"] = terminal._canonical_sha256(
                invalid_ready_unsigned
            )
            with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
                terminal._validate_composed_ready(
                    invalid_ready,
                    context=context,
                    claim_ready=claim_ready,
                    custody_grant=grant,
                    receipt=composed_receipt,
                    expected_composed_identity=composed_identity,
                )

        readback = terminal._build_readback_receipt(
            context,
            composed_ready=composed_ready,
            supervisor_rehashed_composed_identity=composed_identity,
            created_at_utc="2026-07-30T00:00:03.000000Z",
        )
        assert set(readback) == terminal._POSTWAKE_COMPOSED_READBACK_FIELDS
        invalid_readback = {
            **readback,
            "terminal_client_launch_intent_supervisor_custody_retained_through_readback": (False),
        }
        invalid_readback_unsigned = {
            key: item for key, item in invalid_readback.items() if key != "receipt_root_sha256"
        }
        invalid_readback["receipt_root_sha256"] = terminal._canonical_sha256(
            invalid_readback_unsigned
        )
        with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
            terminal._validate_readback_payload(
                invalid_readback,
                context=context,
                composed_ready=composed_ready,
            )
        readback_descriptor = terminal._create_new_readonly_descriptor(
            context.custody_contract.readback_receipt_path
        )
        terminal._write_same_handle(
            readback_descriptor,
            terminal._canonical_bytes(readback),
            maximum_bytes=1024 * 1024,
        )
        readback_file_sha256 = hashlib.sha256(terminal._canonical_bytes(readback)).hexdigest()
        final_ack = terminal._build_final_ack(
            context,
            claim_ready=claim_ready,
            custody_grant=grant,
            composed_ready=composed_ready,
            readback_receipt=readback,
            readback_file_sha256=readback_file_sha256,
            acknowledged_at_utc="2026-07-30T00:00:04.000000Z",
        )
        assert set(final_ack) == terminal._FINAL_ACK_FIELDS
        assert (
            terminal._validate_final_ack(
                final_ack,
                context=context,
                claim_ready=claim_ready,
                custody_grant=grant,
                composed_ready=composed_ready,
                composed_identity=composed_identity,
            )
            == final_ack
        )
        for field in (
            "launcher_redirector_child_process_handles_retained_through_ack",
            "immediate_venv_redirector_process_identity_reverified",
            "terminal_client_launcher_process_identity_reverified",
            "launcher_redirector_child_grandparent_chain_reverified",
            "launcher_redirector_child_same_supervisor_job_reverified",
            "immediate_venv_redirector_live_at_final_ack",
            "terminal_client_launcher_live_at_final_ack",
            "terminal_client_launch_intent_supervisor_custody_retained_through_ack",
        ):
            stale_process_ack = {**final_ack, field: False}
            stale_process_ack_unsigned = {
                key: item for key, item in stale_process_ack.items() if key != "final_ack_sha256"
            }
            stale_process_ack["final_ack_sha256"] = terminal._canonical_sha256(
                stale_process_ack_unsigned
            )
            with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
                terminal._validate_final_ack(
                    stale_process_ack,
                    context=context,
                    claim_ready=claim_ready,
                    custody_grant=grant,
                    composed_ready=composed_ready,
                    composed_identity=composed_identity,
                )
        terminal._validate_readback_receipt(
            context,
            final_ack=final_ack,
            composed_ready=composed_ready,
        )

        permissive = dict(final_ack)
        permissive["automatic_retry_allowed"] = True
        permissive_unsigned = {
            key: item for key, item in permissive.items() if key != "final_ack_sha256"
        }
        permissive["final_ack_sha256"] = terminal._canonical_sha256(permissive_unsigned)
        with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
            terminal._validate_final_ack(
                permissive,
                context=context,
                claim_ready=claim_ready,
                custody_grant=grant,
                composed_ready=composed_ready,
                composed_identity=composed_identity,
            )
    finally:
        if readback_descriptor >= 0:
            os.close(readback_descriptor)
        os.close(launch_intent_descriptor)
        os.close(lease_descriptor)
        os.close(composed_descriptor)
        for path in (
            context.custody_contract.readback_receipt_path,
            launch_intent_path,
            lease_path,
            context.terminal_contract.composed_terminal_receipt_path,
        ):
            if path.exists():
                path.chmod(stat.S_IWRITE)


@pytest.mark.skipif(os.name != "nt", reason="Windows message-mode named pipe")
def test_real_named_pipe_is_duplex_and_bound_to_server_pid() -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_named_pipe = kernel32.CreateNamedPipeW
    create_named_pipe.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_named_pipe.restype = ctypes.c_void_p
    pipe_name = rf"\\.\pipe\aanca-terminal-test-{uuid.uuid4().hex}"
    server_handle = create_named_pipe(
        pipe_name,
        0x00000003,
        0x00000004 | 0x00000002,
        1,
        64 * 1024,
        64 * 1024,
        0,
        None,
    )
    assert server_handle != terminal.INVALID_HANDLE_VALUE
    server = int(server_handle)
    observed: queue.Queue[dict[str, Any]] = queue.Queue()
    errors: queue.Queue[BaseException] = queue.Queue()

    def serve() -> None:
        try:
            connected = kernel32.ConnectNamedPipe(ctypes.c_void_p(server), None)
            if not connected and ctypes.get_last_error() != 535:
                raise ctypes.WinError(ctypes.get_last_error())
            buffer = ctypes.create_string_buffer(64 * 1024)
            read = ctypes.c_uint32()
            if not kernel32.ReadFile(
                ctypes.c_void_p(server),
                buffer,
                len(buffer),
                ctypes.byref(read),
                None,
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            request = terminal._decode_canonical_line(
                bytes(buffer.raw[: read.value]),
                role="synthetic duplex request",
                maximum_bytes=64 * 1024,
            )
            observed.put(request)
            response = terminal._canonical_bytes(
                {"message_type": "SYNTHETIC_ACK", "automatic_retry_allowed": False}
            )
            written = ctypes.c_uint32()
            output = ctypes.create_string_buffer(response)
            if not kernel32.WriteFile(
                ctypes.c_void_p(server),
                output,
                len(response),
                ctypes.byref(written),
                None,
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if written.value != len(response):
                raise AssertionError("named-pipe server made a short write")
            kernel32.FlushFileBuffers(ctypes.c_void_p(server))
        except BaseException as exc:
            errors.put(exc)
        finally:
            kernel32.DisconnectNamedPipe(ctypes.c_void_p(server))
            kernel32.CloseHandle(ctypes.c_void_p(server))

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    client = terminal._DuplexPipeClient(
        pipe_name,
        outbound_maximum_message_bytes=64 * 1024,
        inbound_maximum_message_bytes=64 * 1024,
        custody_exchange_timeout_ms=10_000,
    )
    try:
        assert client.server_pid() == os.getpid()
        request = {"message_type": "SYNTHETIC_READY", "automatic_retry_allowed": False}
        client.send(request)
        response, _line = client.receive()
        assert response == {
            "message_type": "SYNTHETIC_ACK",
            "automatic_retry_allowed": False,
        }
        assert observed.get(timeout=10) == request
    finally:
        client.close()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert errors.empty(), list(errors.queue)


@pytest.mark.skipif(os.name != "nt", reason="Windows overlapped named-pipe deadline")
def test_named_pipe_receive_fails_closed_at_shared_deadline() -> None:
    pipe_name = rf"\\.\pipe\aanca-terminal-timeout-{uuid.uuid4().hex}"
    kernel32, server = _create_test_message_pipe(pipe_name)
    connected = threading.Event()
    release = threading.Event()
    errors: queue.Queue[BaseException] = queue.Queue()

    def serve_without_ack() -> None:
        try:
            result = kernel32.ConnectNamedPipe(ctypes.c_void_p(server), None)
            if not result and ctypes.get_last_error() != 535:
                raise ctypes.WinError(ctypes.get_last_error())
            connected.set()
            if not release.wait(timeout=5):
                raise AssertionError("timeout test did not release pipe server")
        except BaseException as exc:
            errors.put(exc)
        finally:
            kernel32.DisconnectNamedPipe(ctypes.c_void_p(server))
            kernel32.CloseHandle(ctypes.c_void_p(server))

    thread = threading.Thread(target=serve_without_ack, daemon=True)
    thread.start()
    client = terminal._DuplexPipeClient(
        pipe_name,
        outbound_maximum_message_bytes=1024,
        inbound_maximum_message_bytes=2048,
        custody_exchange_timeout_ms=100,
    )
    try:
        assert connected.wait(timeout=5)
        assert client.deadline is None
        assert client.server_pid() == os.getpid()
        assert client.deadline is None
        client._arm_exchange_deadline()
        with pytest.raises(
            terminal.OriginalConfirmatoryTerminalError,
            match="shared deadline expired",
        ):
            client.receive()
    finally:
        release.set()
        client.close()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors.empty(), list(errors.queue)


@pytest.mark.skipif(os.name != "nt", reason="Windows native process identity")
def test_native_process_handle_is_not_truncated_and_identity_is_live() -> None:
    handle = terminal._open_process(
        os.getpid(),
        access=terminal.PROCESS_QUERY_LIMITED_INFORMATION | terminal.SYNCHRONIZE,
    )
    try:
        identity = terminal._process_identity(
            handle,
            pid=os.getpid(),
            command_sha256=_HASH_A,
        )
        assert set(identity) == terminal._PROCESS_IDENTITY_FIELDS
        assert identity["pid"] == os.getpid()
        assert identity["command_sha256"] == _HASH_A
        assert Path(identity["program_path"]).samefile(
            Path(getattr(sys, "_base_executable", sys.executable))
        )
        assert (
            terminal._validate_process_identity(
                identity,
                role="synthetic live process",
            )
            == identity
        )
        inconsistent_time = {
            **identity,
            "creation_time_utc": terminal._filetime_iso(
                identity["creation_time_100ns"] + 10_000_000
            ),
        }
        with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
            terminal._validate_process_identity(
                inconsistent_time,
                role="synthetic live process",
            )
    finally:
        terminal._close_native_handle(handle)


@pytest.mark.skipif(os.name != "nt", reason="Windows remote PEB process view")
def test_running_process_command_view_reads_exact_live_argv_and_cwd() -> None:
    handle = terminal._open_process(
        os.getpid(),
        access=(
            terminal.PROCESS_QUERY_LIMITED_INFORMATION
            | terminal.PROCESS_VM_READ
            | terminal.SYNCHRONIZE
        ),
    )
    try:
        view = terminal._running_process_command_view(handle)
        assert set(view) == {"argv", "cwd", "observation_method"}
        assert isinstance(view["argv"], list)
        assert view["argv"]
        assert Path(view["argv"][0]).samefile(
            Path(getattr(sys, "_base_executable", sys.executable))
        )
        assert Path(view["cwd"]).samefile(Path.cwd())
        assert view["observation_method"] == "windows_peb_process_parameters_v1"
        assert terminal._running_process_parent_pid(handle) == os.getppid()
    finally:
        terminal._close_native_handle(handle)


def test_live_process_command_view_and_launcher_flag_are_closed_contracts(
    tmp_path: Path,
) -> None:
    argv = [
        str(Path(getattr(sys, "_base_executable", sys.executable)).resolve()),
        "-I",
        "-S",
        "-B",
        str((tmp_path / "launcher.py").resolve()),
        "--terminal-receipt-sha256",
        _HASH_A,
    ]
    view = {
        "argv": argv,
        "cwd": str(tmp_path.resolve()),
        "observation_method": "windows_peb_process_parameters_v1",
    }
    assert (
        terminal._validate_live_process_command_view(
            view,
            expected_argv=argv,
            expected_cwd=tmp_path,
            role="synthetic launcher",
        )
        == view
    )
    assert (
        terminal._single_argv_flag_value(
            argv,
            flag="--terminal-receipt-sha256",
            role="synthetic launcher",
        )
        == _HASH_A
    )
    for mutated in (
        {**view, "argv": [str(Path(sys.executable).resolve()), *argv[1:]]},
        {**view, "argv": [*argv, "--unexpected"]},
        {**view, "cwd": str(tmp_path.parent.resolve())},
        {**view, "observation_method": "receipt_claim_v1"},
    ):
        with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
            terminal._validate_live_process_command_view(
                mutated,
                expected_argv=argv,
                expected_cwd=tmp_path,
                role="synthetic launcher",
            )
    for invalid_argv in (
        argv[:-2],
        [*argv, "--terminal-receipt-sha256", _HASH_B],
        [*argv[:-1]],
    ):
        with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
            terminal._single_argv_flag_value(
                invalid_argv,
                flag="--terminal-receipt-sha256",
                role="synthetic launcher",
            )


def test_terminal_ancestry_revalidation_rejects_chain_and_pid_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_path = str(Path(getattr(sys, "_base_executable", sys.executable)).resolve())
    redirector_path = str(Path(sys.executable).resolve())
    child_argv = (redirector_path, "-I", "-B", "capsule.py", "verify-terminal")
    launcher_argv = [
        runtime_path,
        "-I",
        "-S",
        "-B",
        str((tmp_path / "launcher.py").resolve()),
    ]
    runtime_identity = {
        **_process_identity(os.getpid()),
        "program_path": runtime_path,
        "command_sha256": _HASH_A,
    }
    redirector_identity = {
        **_process_identity(200),
        "program_path": redirector_path,
        "command_sha256": _HASH_A,
    }
    launcher_identity = {
        **_process_identity(300),
        "program_path": runtime_path,
        "command_sha256": _HASH_B,
    }
    source_identity = {"synthetic": "source-identity"}
    source = SimpleNamespace(
        descriptor=99,
        path=(tmp_path / "launcher.py").resolve(),
        revalidate_count=0,
    )

    def revalidate() -> None:
        source.revalidate_count += 1

    source.revalidate = revalidate
    context = SimpleNamespace(
        mode=authority.CAPSULE_TERMINAL_MODE,
        terminal_launcher_command={
            "command_sha256": _HASH_B,
            "process_argv": launcher_argv,
            "cwd": str(tmp_path.resolve()),
        },
        terminal_runtime_child_process_identity=runtime_identity,
        immediate_venv_redirector_process_identity=redirector_identity,
        terminal_client_launcher_process_identity=launcher_identity,
        immediate_venv_redirector_process_handle=20,
        terminal_client_launcher_process_handle=30,
        launcher_source_file=source,
        terminal_command=SimpleNamespace(
            command_sha256=_HASH_A,
            argv=child_argv,
            cwd=tmp_path.resolve(),
        ),
        terminal_launcher_projection={
            "verify_terminal_runtime_child_program_path": runtime_path,
        },
        terminal_launcher_release={
            "source_size_bytes": 123,
            "source_physical_identity": source_identity,
        },
    )
    parent_by_handle = {10: 200, 20: 300}
    identity_by_handle = {
        10: runtime_identity,
        20: redirector_identity,
        30: launcher_identity,
    }
    view_by_handle = {
        10: {
            "argv": [runtime_path, *child_argv[1:]],
            "cwd": str(tmp_path.resolve()),
            "observation_method": "windows_peb_process_parameters_v1",
        },
        20: {
            "argv": list(child_argv),
            "cwd": str(tmp_path.resolve()),
            "observation_method": "windows_peb_process_parameters_v1",
        },
        30: {
            "argv": launcher_argv,
            "cwd": str(tmp_path.resolve()),
            "observation_method": "windows_peb_process_parameters_v1",
        },
    }
    monkeypatch.setattr(terminal, "_open_process", lambda _pid, access: 10)
    monkeypatch.setattr(
        terminal,
        "_require_live_process_handle",
        lambda _handle, role: None,
    )
    monkeypatch.setattr(
        terminal,
        "_running_process_parent_pid",
        lambda handle: parent_by_handle[handle],
    )
    monkeypatch.setattr(
        terminal,
        "_process_identity",
        lambda handle, *, pid, command_sha256: identity_by_handle[handle],
    )
    monkeypatch.setattr(
        terminal,
        "_running_process_command_view",
        lambda handle: view_by_handle[handle],
    )
    monkeypatch.setattr(
        terminal,
        "_physical_identity",
        lambda *_args, **_kwargs: source_identity,
    )
    monkeypatch.setattr(terminal, "_close_native_handle", lambda _handle: None)

    terminal._revalidate_terminal_process_ancestry(context)
    assert source.revalidate_count == 1

    parent_by_handle[20] = 301
    with pytest.raises(
        terminal.OriginalConfirmatoryTerminalError,
        match="identity changed",
    ):
        terminal._revalidate_terminal_process_ancestry(context)
    parent_by_handle[20] = 300

    identity_by_handle[30] = {
        **launcher_identity,
        "creation_time_100ns": launcher_identity["creation_time_100ns"] + 10_000_000,
        "creation_time_utc": terminal._filetime_iso(
            launcher_identity["creation_time_100ns"] + 10_000_000
        ),
    }
    with pytest.raises(
        terminal.OriginalConfirmatoryTerminalError,
        match="identity changed",
    ):
        terminal._revalidate_terminal_process_ancestry(context)


def test_terminal_ancestry_establishment_binds_final_launcher_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_path = str(Path(getattr(sys, "_base_executable", sys.executable)).resolve())
    redirector_path = str(Path(sys.executable).resolve())
    launcher_source_path = (tmp_path / "launcher.py").resolve()
    child_argv = (redirector_path, "-I", "-B", "capsule.py", "verify-terminal")
    launcher_argv = [
        runtime_path,
        "-I",
        "-S",
        "-B",
        str(launcher_source_path),
        "--terminal-receipt-sha256",
        _HASH_D,
    ]
    terminal_command = SimpleNamespace(
        argv=child_argv,
        cwd=tmp_path.resolve(),
        command_sha256=_HASH_A,
    )
    launcher_command = {
        "program_path": runtime_path,
        "program_sha256": _HASH_A,
        "process_argv": launcher_argv,
        "cwd": str(tmp_path.resolve()),
        "terminal_receipt_sha256": _HASH_D,
        "command_sha256": _HASH_B,
    }
    runtime_identity = {
        **_process_identity(os.getpid()),
        "program_path": runtime_path,
        "program_sha256": _HASH_A,
        "command_sha256": _HASH_A,
    }
    redirector_identity = {
        **_process_identity(200),
        "program_path": redirector_path,
        "program_sha256": _HASH_A,
        "command_sha256": _HASH_A,
    }
    launcher_identity = {
        **_process_identity(300),
        "program_path": runtime_path,
        "program_sha256": _HASH_A,
        "command_sha256": _HASH_B,
    }
    source_identity = {"synthetic": "launcher-source"}
    source = SimpleNamespace(
        descriptor=99,
        path=launcher_source_path,
        file_sha256=_HASH_C,
        close_count=0,
    )
    source.close = lambda: setattr(source, "close_count", source.close_count + 1)
    release = {
        "verify_terminal_child_launch_topology": (
            "launcher_base_direct_to_venv_redirector_to_runtime_child_v1"
        ),
        "source_path": str(launcher_source_path),
        "source_size_bytes": 123,
        "source_sha256": _HASH_C,
        "source_physical_identity": source_identity,
        "source_physical_identity_root_sha256": terminal._canonical_sha256(source_identity),
    }
    projection = {
        "verify_terminal_child_launch_topology": release["verify_terminal_child_launch_topology"],
        "verify_terminal_runtime_child_program_path": runtime_path,
        "verify_terminal_runtime_child_program_sha256": _HASH_A,
        "verify_terminal_immediate_redirector_program_path": redirector_path,
        "verify_terminal_immediate_redirector_program_sha256": _HASH_A,
    }
    opened = {os.getpid(): 10, 200: 20, 300: 30}
    parents = {10: 200, 20: 300}
    identities = {
        10: runtime_identity,
        20: redirector_identity,
        30: launcher_identity,
    }
    views = {
        10: {
            "argv": [runtime_path, *child_argv[1:]],
            "cwd": str(tmp_path.resolve()),
            "observation_method": "windows_peb_process_parameters_v1",
        },
        20: {
            "argv": list(child_argv),
            "cwd": str(tmp_path.resolve()),
            "observation_method": "windows_peb_process_parameters_v1",
        },
        30: {
            "argv": launcher_argv,
            "cwd": str(tmp_path.resolve()),
            "observation_method": "windows_peb_process_parameters_v1",
        },
    }
    closed: list[int] = []
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        terminal,
        "_open_process",
        lambda pid, *, access: opened[pid],
    )
    monkeypatch.setattr(
        terminal,
        "_require_live_process_handle",
        lambda _handle, role: None,
    )
    monkeypatch.setattr(
        terminal,
        "_running_process_parent_pid",
        lambda handle: parents[handle],
    )
    monkeypatch.setattr(
        terminal,
        "_running_process_command_view",
        lambda handle: views[handle],
    )
    monkeypatch.setattr(
        terminal,
        "_process_identity",
        lambda handle, *, pid, command_sha256: identities[handle],
    )
    monkeypatch.setattr(
        terminal,
        "_open_held_file",
        lambda *_args, **_kwargs: source,
    )
    monkeypatch.setattr(
        terminal,
        "_physical_identity",
        lambda *_args, **_kwargs: source_identity,
    )
    monkeypatch.setattr(
        terminal,
        "_close_native_handle",
        lambda handle: closed.append(handle),
    )

    def build_launcher_command(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return launcher_command

    monkeypatch.setattr(
        authority,
        "build_original_confirmatory_terminal_client_launcher_command",
        build_launcher_command,
    )
    (
        observed_runtime,
        observed_redirector,
        redirector_handle,
        observed_launcher,
        launcher_handle,
        observed_command,
        observed_source,
    ) = terminal._establish_terminal_process_ancestry(
        terminal_command=terminal_command,
        launcher_release=release,
        launcher_projection=projection,
        capsule={},
        supervisor_spec_sha256=_HASH_B,
        e_intent_file_sha256=_HASH_C,
    )
    assert observed_runtime == runtime_identity
    assert observed_redirector == redirector_identity
    assert observed_launcher == launcher_identity
    assert observed_command == launcher_command
    assert observed_source is source
    assert redirector_handle == 20
    assert launcher_handle == 30
    assert captured["terminal_receipt_sha256"] == _HASH_D
    assert captured["supervisor_spec_sha256"] == _HASH_B
    assert captured["e_intent_file_sha256"] == _HASH_C
    assert closed == [10]
    assert source.close_count == 0


@pytest.mark.skipif(os.name != "nt", reason="Windows DuplicateHandle access mask")
def test_duplicate_from_process_is_mapped_file_generic_read_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate-source.json"
    source_descriptor = terminal._create_new_readonly_descriptor(path)
    process_handle = terminal._open_process(
        os.getpid(),
        access=terminal.PROCESS_DUP_HANDLE,
    )
    duplicate_descriptor = -1
    try:
        terminal._write_same_handle(
            source_descriptor,
            b'{"synthetic":"duplicate-source"}\n',
            maximum_bytes=1024,
        )
        duplicate_descriptor = terminal._duplicate_from_process(
            process_handle,
            source_handle_slot=msvcrt.get_osfhandle(source_descriptor),
        )
        granted = terminal._granted_access_mask_from_native_handle(
            msvcrt.get_osfhandle(duplicate_descriptor)
        )
        assert granted == terminal.FILE_GENERIC_READ_ACCESS_MASK
        assert terminal._read_descriptor(
            duplicate_descriptor,
            maximum_bytes=1024,
        ) == terminal._read_descriptor(
            source_descriptor,
            maximum_bytes=1024,
        )
    finally:
        if duplicate_descriptor >= 0:
            os.close(duplicate_descriptor)
        terminal._close_native_handle(process_handle)
        os.close(source_descriptor)
        path.chmod(stat.S_IWRITE)


@pytest.mark.skipif(os.name != "nt", reason="Windows retained launch-intent handle")
def test_post_grant_launch_intent_duplicate_is_exact_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (tmp_path / authority.TERMINAL_CLIENT_LAUNCH_INTENT_FILENAME).resolve()
    source_descriptor = terminal._create_new_readonly_descriptor(path)
    supervisor_process_handle = terminal._open_process(
        os.getpid(),
        access=terminal.PROCESS_DUP_HANDLE,
    )
    launcher_identity = {
        **_process_identity(os.getpid()),
        "command_sha256": _HASH_D,
    }
    returned_launcher_identity = {"value": launcher_identity}
    try:
        payload = terminal._canonical_bytes({"synthetic": "launch-intent"})
        terminal._write_same_handle(
            source_descriptor,
            payload,
            maximum_bytes=1024,
        )
        source_identity = terminal._physical_identity(
            source_descriptor,
            path=path,
            role="terminal-client-launch-intent",
        )
        context = SimpleNamespace(
            terminal_launcher_command={},
            terminal_launcher_projection={"launch_intent_path": str(path)},
            terminal_launcher_release={},
            terminal_client_launcher_process_identity=launcher_identity,
            q={"execution_capsule": {}},
            terminal_command=SimpleNamespace(),
        )
        grant = {
            "terminal_client_launch_intent_supervisor_handle_slot": (
                msvcrt.get_osfhandle(source_descriptor)
            ),
            "terminal_client_launch_intent_physical_identity": source_identity,
            "terminal_client_launch_intent_file_sha256": source_identity["sha256"],
            "terminal_client_launch_intent_root_sha256": _HASH_C,
            "terminal_client_launch_intent_child_expected_granted_access_mask": (
                terminal.FILE_GENERIC_READ_ACCESS_MASK
            ),
            "terminal_client_launch_intent_policy": (
                authority.TERMINAL_CLIENT_LAUNCH_INTENT_POLICY
            ),
            "terminal_client_launch_intent_physical_identity_root_sha256": (
                terminal._canonical_sha256(source_identity)
            ),
            "terminal_client_launcher_process_identity": launcher_identity,
        }

        def canonical_launch_intent(
            value: dict[str, Any],
            **_kwargs: Any,
        ) -> dict[str, Any]:
            assert value == {"synthetic": "launch-intent"}
            return {
                "policy": authority.TERMINAL_CLIENT_LAUNCH_INTENT_POLICY,
                "intent_root_sha256": _HASH_C,
                "launcher_process_identity": returned_launcher_identity["value"],
            }

        monkeypatch.setattr(
            authority,
            "canonical_original_confirmatory_terminal_client_launch_intent",
            canonical_launch_intent,
        )
        held, intent, child_slot, child_access = (
            terminal._open_validated_terminal_client_launch_intent(
                context,
                custody_grant=grant,
                supervisor_process_handle=supervisor_process_handle,
            )
        )
        try:
            assert held.payload == payload
            assert intent["intent_root_sha256"] == _HASH_C
            assert child_slot > 0
            assert child_access == terminal.FILE_GENERIC_READ_ACCESS_MASK
            held.revalidate()
        finally:
            held.close()

        for mutated_grant in (
            {**grant, "terminal_client_launch_intent_file_sha256": _HASH_A},
            {**grant, "terminal_client_launch_intent_root_sha256": _HASH_A},
            {
                **grant,
                "terminal_client_launch_intent_child_expected_granted_access_mask": (
                    terminal.GENERIC_READ
                ),
            },
            {
                **grant,
                "terminal_client_launch_intent_physical_identity": {
                    **source_identity,
                    "sha256": _HASH_A,
                },
            },
        ):
            with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
                terminal._open_validated_terminal_client_launch_intent(
                    context,
                    custody_grant=mutated_grant,
                    supervisor_process_handle=supervisor_process_handle,
                )

        returned_launcher_identity["value"] = {
            **launcher_identity,
            "command_sha256": _HASH_A,
        }
        with pytest.raises(terminal.OriginalConfirmatoryTerminalError):
            terminal._open_validated_terminal_client_launch_intent(
                context,
                custody_grant=grant,
                supervisor_process_handle=supervisor_process_handle,
            )
    finally:
        terminal._close_native_handle(supervisor_process_handle)
        os.close(source_descriptor)
        if path.exists():
            path.chmod(stat.S_IWRITE)


@pytest.mark.skipif(os.name != "nt", reason="Windows retained-source role bound")
def test_retained_source_size_cannot_expand_its_own_read_bound(
    tmp_path: Path,
) -> None:
    path = tmp_path / "retained-terminal.json"
    payload = b'{"synthetic":"retained-terminal"}\n'
    source_descriptor = terminal._create_new_readonly_descriptor(path)
    process_handle = terminal._open_process(
        os.getpid(),
        access=terminal.PROCESS_DUP_HANDLE,
    )
    held: Any = None
    try:
        terminal._write_same_handle(
            source_descriptor,
            payload,
            maximum_bytes=1024,
        )
        identity = terminal._physical_identity(
            source_descriptor,
            path=path,
            role="supervisor-terminal",
            maximum_bytes=1024,
        )
        binding = {
            "handle_slot": msvcrt.get_osfhandle(source_descriptor),
            "physical_identity": identity,
        }
        with pytest.raises(
            terminal.OriginalConfirmatoryTerminalError,
            match="independent role bound",
        ):
            terminal._held_from_supervisor_slot(
                process_handle,
                binding=binding,
                maximum_bytes=len(payload) - 1,
            )
        held = terminal._held_from_supervisor_slot(
            process_handle,
            binding=binding,
            maximum_bytes=len(payload),
        )
        assert held.payload == payload
    finally:
        if held is not None:
            held.close()
        terminal._close_native_handle(process_handle)
        os.close(source_descriptor)
        path.chmod(stat.S_IWRITE)
