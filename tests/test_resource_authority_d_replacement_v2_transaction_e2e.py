"""Hermetic transaction-level regressions for replacement-v2 publication."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from histo_audit.workflows import (
    resource_authority_d_replacement_v2_controller as controller,
)


def _windows_typer_json_bytes(payload: Mapping[str, Any]) -> bytes:
    text = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return text.replace("\n", "\r\n").encode("utf-8")


def _captured_stream(payload: bytes, *, limit: int) -> dict[str, Any]:
    record = controller._empty_stream_diagnostic(limit)
    record.update(
        {
            "capture_started": True,
            "captured_size_bytes": len(payload),
            "captured_sha256": hashlib.sha256(payload).hexdigest(),
            "eof_observed": True,
            "reader_joined": True,
            "pipe_closed": True,
        }
    )
    return record


def _passed_diagnostic(
    request: controller.VerifyRequestV2,
    *,
    stdout_payload: bytes,
) -> dict[str, Any]:
    checked = request.checked()
    requested = str(Path(sys.executable).resolve(strict=True))
    override = controller._fresh_verifier_spawn_executable(requested)
    effective = requested if override is None else override
    controller_pid = os.getpid()
    verifier_pid = controller_pid + 100_000
    request_record = controller._fresh_request_record(checked)
    cleanup = controller._empty_cleanup_diagnostic()
    cleanup["returncode_observed"] = True
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
    diagnostic = {
        "schema_version": controller.FRESH_DIAGNOSTIC_SCHEMA_VERSION,
        "policy": controller.FRESH_DIAGNOSTIC_POLICY,
        "status": "passed",
        "failure_phase": "completed",
        "requested_python_executable": requested,
        "effective_spawn_executable": effective,
        "executable_override_used": override is not None,
        "request": request_record,
        "request_sha256": controller._compact_sha256(request_record),
        "argv_sha256": controller._compact_sha256(list(checked.argv(controller_pid))),
        "controller_process_id": controller_pid,
        "verifier_process_id": verifier_pid,
        "returncode": 0,
        "timeout_milliseconds": 1_000,
        "timed_out": False,
        "stdout": _captured_stream(
            stdout_payload,
            limit=controller._MAX_STDOUT_BYTES,
        ),
        "stderr": _captured_stream(
            b"",
            limit=controller._MAX_STDERR_BYTES,
        ),
        "cleanup": cleanup,
        "payload_sha256": hashlib.sha256(stdout_payload).hexdigest(),
        "payload_validation_completed": True,
        "stdout_content_included": False,
        "stderr_content_included": False,
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }
    assert controller._canonical_fresh_diagnostic(diagnostic) == diagnostic
    return diagnostic


def _fresh_payload(
    request: controller.VerifyRequestV2,
    *,
    controller_pid: int,
    verifier_pid: int,
) -> dict[str, Any]:
    return {
        "status": "verified",
        "verification_schema_version": 2,
        "verification_kind": ("resource_bounded_technical_successor_fresh_process"),
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


class _OwnedLock:
    def __enter__(self) -> _OwnedLock:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def assert_owned(self) -> None:
        return None


@dataclass
class _MutableState:
    live_drift: bool = False
    activate_drift_in_runner: bool = False
    transplant_fresh_diagnostic: bool = False
    fail_preclaim_authorization: bool = False


class _SyntheticSchemaV3API:
    def __init__(
        self,
        *,
        parent: Path,
        destination: Path,
        pins: controller.AuthorityPins,
        technical_authorization: Mapping[str, Any],
        state: _MutableState,
    ) -> None:
        self.parent = parent
        self.destination = destination
        self.pins = pins
        self.technical_authorization = dict(technical_authorization)
        self.state = state
        self.mode = "success"
        self.restore_drift_after_callback_failure = False
        self.published = object()
        self.callback_values: list[object] = []
        self.authority_pin_values: list[object] = []
        self.committed_verifications: list[
            tuple[Path, controller.AuthorityPins, Mapping[str, Any]]
        ] = []

    def canonicalize_authorization(
        self,
        authorization: Mapping[str, Any],
        *,
        replacement_publication_failure_lineage: Mapping[str, Any],
    ) -> dict[str, Any]:
        del replacement_publication_failure_lineage
        return dict(authorization)

    def _remove_destination(self) -> None:
        if self.destination.is_dir():
            self.destination.rmdir()

    def create_authority(
        self,
        *,
        authorization: Mapping[str, Any],
        post_publication_check: Callable[[Any], None],
    ) -> object:
        assert dict(authorization) == self.technical_authorization
        if self.mode == "fail_before_callback":
            raise RuntimeError("synthetic creator failure before callback")

        self.destination.mkdir()
        self.callback_values.append(self.published)
        try:
            post_publication_check(self.published)
        except BaseException:
            self._remove_destination()
            if self.restore_drift_after_callback_failure:
                self.state.live_drift = False
            raise

        if self.mode == "fail_after_callback":
            self._remove_destination()
            raise RuntimeError("synthetic creator rollback after callback")
        return self.published

    def authority_pins(self, published: object) -> controller.AuthorityPins:
        self.authority_pin_values.append(published)
        if published is not self.published:
            raise controller.ControlError("synthetic callback identity changed")
        return self.pins

    def verify_committed(
        self,
        authority: Path,
        *,
        expected: controller.AuthorityPins,
        replacement_publication_failure_lineage: Mapping[str, Any],
    ) -> None:
        assert authority.resolve() == self.destination
        assert self.destination.is_dir()
        assert expected == self.pins
        self.committed_verifications.append(
            (
                authority.resolve(),
                expected,
                dict(replacement_publication_failure_lineage),
            )
        )


def _fresh_result(
    request: controller.VerifyRequestV2,
    *,
    transplant_diagnostic: bool,
) -> controller.VerifyResultV2:
    checked = request.checked()
    controller_pid = os.getpid()
    verifier_pid = controller_pid + 100_000
    payload = _fresh_payload(
        checked,
        controller_pid=controller_pid,
        verifier_pid=verifier_pid,
    )
    diagnostic_request = checked
    if transplant_diagnostic:
        diagnostic_request = controller.VerifyRequestV2(
            project_root=checked.project_root,
            successor_directory=checked.successor_directory.parent / "foreign-authority-d",
            parent_directory=checked.parent_directory,
            artifact_root_sha256=checked.artifact_root_sha256,
            manifest_sha256=checked.manifest_sha256,
            authorization_sha256=checked.authorization_sha256,
            intent_sha256=checked.intent_sha256,
            nonce=checked.nonce,
            python_executable=checked.python_executable,
        ).checked()
    diagnostic = _passed_diagnostic(
        diagnostic_request,
        stdout_payload=_windows_typer_json_bytes(payload),
    )
    return controller.VerifyResultV2(
        request=checked,
        argv=checked.argv(controller_pid),
        process_id=verifier_pid,
        payload=payload,
        payload_sha256=diagnostic["payload_sha256"],
        diagnostic=diagnostic,
    )


@pytest.fixture
def transaction_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    project = tmp_path.resolve()
    namespace = controller.Namespace.for_project(project)
    namespace.control_root.mkdir(parents=True)
    amendment_root = project / "artifacts" / "preregistration_amendments"
    parent = amendment_root / controller._AUTHORITY_C_COMPONENT
    destination = amendment_root / "20990101T000000.000000Z"
    parent.mkdir(parents=True)

    authorization_sha256 = "3" * 64
    intent_sha256 = "4" * 64
    authorization_receipt_sha256 = "5" * 64
    attempt_id = "6" * 64
    technical_authorization = {"kind": "synthetic-technical-authorization-v3"}
    terminal_lineage = {"kind": "synthetic-terminal-lineage"}
    authorization = {
        "authorized_attempt_id": attempt_id,
        "publication": {
            "intended_authority_directory": str(destination.resolve()),
        },
        "preflight": {
            "contract": {
                "technical_successor": {
                    "authorization": technical_authorization,
                    "authorization_sha256": authorization_sha256,
                    "intent_sha256": intent_sha256,
                },
                "terminal_qualification": terminal_lineage,
            }
        },
    }
    verification_nonce = controller._verification_nonce_v2(
        authorization,
        authorization_receipt_sha256,
    )
    attempt = {
        "claimed_at_utc": controller._timestamp(datetime.now(UTC) - timedelta(minutes=2)),
        "attempt_id": attempt_id,
        "intended_authority_directory": str(destination.resolve()),
        "parent_authority_directory": str(parent.resolve()),
        "technical_authorization_sha256": authorization_sha256,
        "intent_sha256": intent_sha256,
        "verification_nonce": verification_nonce,
        "run_state": {"sha256": "7" * 64},
    }
    attempt_bytes = controller._canonical_bytes(attempt)
    attempt_sha256 = hashlib.sha256(attempt_bytes).hexdigest()
    pins = controller.AuthorityPins(
        directory=destination.resolve(),
        parent_directory=parent.resolve(),
        artifact_root_sha256="8" * 64,
        sha256_manifest_sha256="9" * 64,
        authorization_sha256=authorization_sha256,
        intent_sha256=intent_sha256,
        chain_depth=4,
    )
    state = _MutableState()
    api = _SyntheticSchemaV3API(
        parent=parent.resolve(),
        destination=destination.resolve(),
        pins=pins,
        technical_authorization=technical_authorization,
        state=state,
    )
    verifier = controller.TransactionVerifierV2(
        api=api,
        project_root=project,
        parent=parent,
        destination=destination,
        authorization_sha256=authorization_sha256,
        intent_sha256=intent_sha256,
        verification_nonce=verification_nonce,
    )
    containment_preflight_calls: list[bool] = []

    def hermetic_containment_preflight() -> None:
        assert verifier.containment_preflight_completed is False
        containment_preflight_calls.append(True)
        verifier.containment_preflight_completed = True

    monkeypatch.setattr(
        verifier,
        "preflight_containment",
        hermetic_containment_preflight,
    )

    def presence(_namespace: controller.Namespace) -> dict[str, bool]:
        return {
            "qualification": True,
            "inputs": True,
            "authorization": True,
            "attempt": namespace.attempt_v2.is_file(),
            "success": namespace.success_v2.is_file(),
            "failure": namespace.failure_v2.is_file(),
        }

    def candidates(_parent: str | Path) -> tuple[Path, ...]:
        return (destination.resolve(),) if destination.is_dir() else ()

    def read_authorization(
        *_args: object,
        verify_live: bool,
        **_kwargs: object,
    ) -> tuple[dict[str, Any], str]:
        if verify_live and state.fail_preclaim_authorization:
            raise controller.ControlError("synthetic pre-A2 authorization drift")
        return authorization, authorization_receipt_sha256

    def read_attempt(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[dict[str, Any], str]:
        encoded = namespace.attempt_v2.read_bytes()
        if encoded != attempt_bytes:
            raise controller.ControlError("synthetic A2 bytes changed")
        return attempt, hashlib.sha256(encoded).hexdigest()

    baseline_calls: list[dict[str, Any]] = []

    def governed_baseline(
        _namespace: controller.Namespace,
        **kwargs: Any,
    ) -> None:
        expected_presence = dict(kwargs["expected_presence"])
        expected_candidates = tuple(Path(path).resolve() for path in kwargs["expected_candidates"])
        if expected_presence != presence(namespace):
            raise controller.ControlError("synthetic governed presence changed")
        if expected_candidates != candidates(parent):
            raise controller.ControlError("synthetic governed candidate set changed")
        baseline_calls.append(dict(kwargs))
        if state.live_drift:
            raise controller.ControlError("synthetic live referent drift")

    fresh_requests: list[controller.VerifyRequestV2] = []

    def fresh_runner(
        request: controller.VerifyRequestV2,
        **_kwargs: object,
    ) -> controller.VerifyResultV2:
        fresh_requests.append(request.checked())
        if state.activate_drift_in_runner:
            state.live_drift = True
        return _fresh_result(
            request,
            transplant_diagnostic=state.transplant_fresh_diagnostic,
        )

    monkeypatch.setattr(
        controller,
        "_protocol_lock",
        lambda *_args, **_kwargs: _OwnedLock(),
    )
    monkeypatch.setattr(
        controller,
        "ExclusiveBundlePublicationLock",
        lambda *_args, **_kwargs: _OwnedLock(),
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
    monkeypatch.setattr(controller, "_reserved_family_presence", presence)
    monkeypatch.setattr(
        controller,
        "_stable_amendment_inventory",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(controller, "_read_publication_authorization_v2", read_authorization)
    monkeypatch.setattr(
        controller,
        "_build_attempt_v2",
        lambda **_kwargs: attempt,
    )
    monkeypatch.setattr(controller, "_read_attempt_v2", read_attempt)
    monkeypatch.setattr(
        controller,
        "_verify_live_governed_baseline_v2",
        governed_baseline,
    )
    monkeypatch.setattr(
        controller,
        "_read_terminal_qualification",
        lambda *_args, **_kwargs: ({"status": "qualified"}, "a" * 64),
    )
    monkeypatch.setattr(
        controller,
        "_read_input_v3",
        lambda *_args, **_kwargs: ({}, {}, "b" * 64),
    )
    monkeypatch.setattr(controller, "run_fresh_verifier_v2", fresh_runner)

    clock_count = 0
    clock_start = datetime.now(UTC) - timedelta(minutes=1)

    def clock() -> datetime:
        nonlocal clock_count
        value = clock_start + timedelta(seconds=clock_count)
        clock_count += 1
        return value

    def execute() -> controller.PublicationResultV2:
        return controller._execute_publication_v2_once(
            namespace=namespace,
            parent_authority_directory=parent,
            authorization=authorization,
            authorization_receipt_sha256=authorization_receipt_sha256,
            api=api,
            verifier=verifier,
            clock=clock,
            candidate_discoverer=candidates,
        )

    committed_classification_values: list[tuple[Path, Mapping[str, Any]]] = []

    def committed_readback(
        candidate: Path,
        success: Mapping[str, Any],
    ) -> None:
        expected = controller.AuthorityPins(
            directory=candidate.resolve(),
            parent_directory=Path(success["parent_authority_directory"]).resolve(),
            artifact_root_sha256=success["artifact_root_sha256"],
            sha256_manifest_sha256=success["sha256_manifest_sha256"],
            authorization_sha256=success["authorization_sha256"],
            intent_sha256=success["intent_sha256"],
            chain_depth=success["chain_depth"],
        )
        api.verify_committed(
            candidate,
            expected=expected,
            replacement_publication_failure_lineage=terminal_lineage,
        )
        committed_classification_values.append((candidate.resolve(), dict(success)))

    def classify() -> controller.Classification:
        return controller.classify(
            namespace,
            parent_authority_directory=parent,
            candidate_discoverer=candidates,
            committed_candidate_verifier=committed_readback,
        )

    return SimpleNamespace(
        project=project,
        namespace=namespace,
        parent=parent.resolve(),
        destination=destination.resolve(),
        authorization=authorization,
        authorization_receipt_sha256=authorization_receipt_sha256,
        terminal_lineage=terminal_lineage,
        attempt=attempt,
        attempt_bytes=attempt_bytes,
        attempt_sha256=attempt_sha256,
        pins=pins,
        state=state,
        api=api,
        verifier=verifier,
        baseline_calls=baseline_calls,
        fresh_requests=fresh_requests,
        execute=execute,
        classify=classify,
        committed_classification_values=committed_classification_values,
        containment_preflight_calls=containment_preflight_calls,
    )


def _assert_no_a_s_f_or_d(tree: SimpleNamespace) -> None:
    assert not tree.namespace.attempt_v2.exists()
    assert not tree.namespace.success_v2.exists()
    assert not tree.namespace.failure_v2.exists()
    assert not tree.destination.exists()


def test_pre_a2_live_authorization_failure_leaves_no_transaction_artifacts(
    transaction_tree: SimpleNamespace,
) -> None:
    transaction_tree.state.fail_preclaim_authorization = True

    with pytest.raises(
        controller.ControlError,
        match="synthetic pre-A2 authorization drift",
    ):
        transaction_tree.execute()

    _assert_no_a_s_f_or_d(transaction_tree)
    assert transaction_tree.verifier.invoked is False
    assert transaction_tree.fresh_requests == []


def test_pre_a2_transplanted_not_invoked_diagnostic_is_rejected_without_writes(
    transaction_tree: SimpleNamespace,
) -> None:
    other = controller.TransactionVerifierV2(
        api=transaction_tree.api,
        project_root=transaction_tree.project,
        parent=transaction_tree.parent,
        destination=transaction_tree.destination.parent / "foreign-authority-d",
        authorization_sha256=transaction_tree.pins.authorization_sha256,
        intent_sha256=transaction_tree.pins.intent_sha256,
        verification_nonce=transaction_tree.attempt["verification_nonce"],
    )
    transaction_tree.verifier.diagnostic = other.diagnostic

    with pytest.raises(
        controller.ControlError,
        match="stale or differently scoped",
    ):
        transaction_tree.execute()

    _assert_no_a_s_f_or_d(transaction_tree)
    assert transaction_tree.fresh_requests == []


def test_a2_to_d2_and_s2_happy_path_preserves_callback_identity_and_classifies(
    transaction_tree: SimpleNamespace,
) -> None:
    result = transaction_tree.execute()

    assert result.state is controller.State.COMMITTED
    assert result.authority_directory == transaction_tree.destination
    assert transaction_tree.namespace.attempt_v2.read_bytes() == (transaction_tree.attempt_bytes)
    assert transaction_tree.destination.is_dir()
    assert transaction_tree.namespace.success_v2.is_file()
    assert not transaction_tree.namespace.failure_v2.exists()
    assert transaction_tree.api.callback_values == [transaction_tree.api.published]
    assert transaction_tree.verifier.published_result is transaction_tree.api.published
    assert all(
        value is transaction_tree.api.published
        for value in transaction_tree.api.authority_pin_values
    )
    assert transaction_tree.fresh_requests == [transaction_tree.verifier.result.request]
    success, success_sha256 = controller._read_success_v2(
        transaction_tree.namespace,
        attempt=transaction_tree.attempt,
        attempt_sha256=transaction_tree.attempt_sha256,
    )
    assert success_sha256 == result.marker_sha256
    assert success["status"] == "committed"

    classification = transaction_tree.classify()

    assert classification.state is controller.State.COMMITTED
    assert classification.success_v2_sha256 == result.marker_sha256
    assert classification.candidates == (transaction_tree.destination,)
    assert len(transaction_tree.committed_classification_values) == 2
    assert (
        transaction_tree.committed_classification_values[0]
        == transaction_tree.committed_classification_values[1]
    )


@pytest.mark.parametrize(
    ("mode", "expected_phase", "expected_diagnostic_status", "callback_count"),
    (
        (
            "fail_before_callback",
            "authority_creation_before_fresh_verifier",
            "not_invoked",
            0,
        ),
        (
            "fail_after_callback",
            "authority_creation_after_fresh_verifier",
            "passed",
            1,
        ),
    ),
)
def test_creator_rollback_before_or_after_fresh_callback_writes_exact_f2(
    transaction_tree: SimpleNamespace,
    mode: str,
    expected_phase: str,
    expected_diagnostic_status: str,
    callback_count: int,
) -> None:
    transaction_tree.api.mode = mode

    result = transaction_tree.execute()

    assert result.state is controller.State.ROLLED_BACK_FAILURE
    assert transaction_tree.namespace.attempt_v2.is_file()
    assert transaction_tree.namespace.failure_v2.is_file()
    assert not transaction_tree.namespace.success_v2.exists()
    assert not transaction_tree.destination.exists()
    assert len(transaction_tree.api.callback_values) == callback_count
    failure, failure_sha256 = controller._read_failure_v2(
        transaction_tree.namespace,
        attempt=transaction_tree.attempt,
        attempt_sha256=transaction_tree.attempt_sha256,
    )
    assert failure_sha256 == result.marker_sha256
    assert failure["failure_phase"] == expected_phase
    assert failure["fresh_verifier_diagnostic"]["status"] == (expected_diagnostic_status)
    assert failure["authority_absent_after_rollback"] is True

    classification = transaction_tree.classify()

    assert classification.state is controller.State.ROLLED_BACK_FAILURE
    assert classification.failure_v2_sha256 == result.marker_sha256
    assert classification.candidates == ()


@pytest.mark.parametrize(
    ("restore_drift", "expected_f2"),
    ((True, True), (False, False)),
)
def test_live_drift_inside_callback_allows_f2_only_after_restored_rollback_baseline(
    transaction_tree: SimpleNamespace,
    restore_drift: bool,
    expected_f2: bool,
) -> None:
    transaction_tree.state.activate_drift_in_runner = True
    transaction_tree.api.restore_drift_after_callback_failure = restore_drift

    if expected_f2:
        result = transaction_tree.execute()
        assert result.state is controller.State.ROLLED_BACK_FAILURE
    else:
        with pytest.raises(
            controller.ControlError,
            match="synthetic live referent drift",
        ):
            transaction_tree.execute()

    assert transaction_tree.namespace.attempt_v2.is_file()
    assert transaction_tree.namespace.failure_v2.is_file() is expected_f2
    assert not transaction_tree.namespace.success_v2.exists()
    assert not transaction_tree.destination.exists()
    assert transaction_tree.verifier.invoked is True
    assert transaction_tree.verifier.result is not None
    assert transaction_tree.verifier.diagnostic["status"] == "passed"
    if expected_f2:
        failure, _failure_sha256 = controller._read_failure_v2(
            transaction_tree.namespace,
            attempt=transaction_tree.attempt,
            attempt_sha256=transaction_tree.attempt_sha256,
        )
        assert failure["failure_phase"] == ("authority_creation_after_fresh_verifier")


def test_transplanted_passed_diagnostic_is_rejected_before_s2(
    transaction_tree: SimpleNamespace,
) -> None:
    transaction_tree.state.transplant_fresh_diagnostic = True

    with pytest.raises(controller.ControlError):
        transaction_tree.execute()

    assert transaction_tree.namespace.attempt_v2.is_file()
    assert transaction_tree.destination.is_dir()
    assert not transaction_tree.namespace.success_v2.exists()
    assert not transaction_tree.namespace.failure_v2.exists()
    assert transaction_tree.verifier.invoked is True
    assert transaction_tree.verifier.result is not None
    assert transaction_tree.verifier.result.request.successor_directory == (
        transaction_tree.destination
    )
    assert transaction_tree.verifier.diagnostic["request"]["successor_directory"] != str(
        transaction_tree.destination
    )
