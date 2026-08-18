"""Hermetic public-contract E2E tests for Authority-D replacement-v2."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from histo_audit.workflows import (
    resource_authority_d_replacement_v2_controller as controller,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = controller._canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return _sha256(encoded)


def _tree_snapshot(root: Path) -> tuple[tuple[str, str, int, str], ...]:
    records: list[tuple[str, str, int, str]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            records.append((relative, "directory", 0, ""))
        else:
            payload = path.read_bytes()
            records.append((relative, "file", len(payload), _sha256(payload)))
    return tuple(records)


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


def _passed_fresh_result(
    request: controller.VerifyRequestV2,
    *,
    timeout_seconds: float,
) -> controller.VerifyResultV2:
    request = request.checked()
    controller_pid = os.getpid()
    verifier_pid = controller_pid + 100_000
    payload = _fresh_payload(
        request,
        controller_pid=controller_pid,
        verifier_pid=verifier_pid,
    )
    stdout = (
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2)
        .replace("\n", "\r\n")
        .encode("utf-8")
        + b"\r\n"
    )
    requested = str(Path(sys.executable).resolve(strict=True))
    override = controller._fresh_verifier_spawn_executable(requested)
    effective = requested if override is None else override
    argv = request.argv(controller_pid)
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
        "request": controller._fresh_request_record(request),
        "request_sha256": controller._compact_sha256(controller._fresh_request_record(request)),
        "argv_sha256": controller._compact_sha256(list(argv)),
        "controller_process_id": controller_pid,
        "verifier_process_id": verifier_pid,
        "returncode": 0,
        "timeout_milliseconds": int(timeout_seconds * 1_000),
        "timed_out": False,
        "stdout": _captured_stream(
            stdout,
            limit=controller._MAX_STDOUT_BYTES,
        ),
        "stderr": _captured_stream(
            b"",
            limit=controller._MAX_STDERR_BYTES,
        ),
        "cleanup": cleanup,
        "payload_sha256": _sha256(stdout),
        "payload_validation_completed": True,
        "stdout_content_included": False,
        "stderr_content_included": False,
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }
    assert controller._canonical_fresh_diagnostic(diagnostic) == diagnostic
    return controller.VerifyResultV2(
        request=request,
        argv=argv,
        process_id=verifier_pid,
        payload=payload,
        payload_sha256=_sha256(stdout),
        diagnostic=diagnostic,
    )


class _SchemaV3Adapter:
    def __init__(
        self,
        *,
        destination: Path,
        parent: Path,
        fail_before_callback: bool,
        authorization_sha256: str,
        intent_sha256: str,
        terminal_lineage: Mapping[str, Any],
    ) -> None:
        self.destination = destination
        self.parent = parent
        self.fail_before_callback = fail_before_callback
        self.authorization_sha256 = authorization_sha256
        self.intent_sha256 = intent_sha256
        self.terminal_lineage = dict(terminal_lineage)
        self.published = object()
        self.creator_called = False
        self.callback_received = False
        self.callback_called = False
        self.committed_verifications = 0
        self.pins = controller.AuthorityPins(
            directory=destination,
            parent_directory=parent,
            artifact_root_sha256="7" * 64,
            sha256_manifest_sha256="8" * 64,
            authorization_sha256=authorization_sha256,
            intent_sha256=intent_sha256,
            chain_depth=4,
        )

    def canonicalize_authorization(
        self,
        authorization: Mapping[str, Any],
        *,
        replacement_publication_failure_lineage: Mapping[str, Any],
    ) -> dict[str, Any]:
        assert dict(replacement_publication_failure_lineage) == self.terminal_lineage
        return dict(authorization)

    def create_authority(
        self,
        *,
        authorization: Mapping[str, Any],
        post_publication_check: Callable[[Any], None],
    ) -> object:
        assert authorization == {"kind": "synthetic-technical-authorization"}
        self.creator_called = True
        self.callback_received = True
        if self.fail_before_callback:
            raise RuntimeError("synthetic creator failure before callback")
        self.destination.mkdir()
        post_publication_check(self.published)
        self.callback_called = True
        return self.published

    def authority_pins(self, published: object) -> controller.AuthorityPins:
        assert published is self.published
        return self.pins

    def verify_committed(
        self,
        authority: Path,
        *,
        expected: controller.AuthorityPins,
        replacement_publication_failure_lineage: Mapping[str, Any],
    ) -> None:
        assert authority == self.destination
        assert expected == self.pins
        assert dict(replacement_publication_failure_lineage) == self.terminal_lineage
        self.committed_verifications += 1


@dataclass(slots=True)
class _PublicHarness:
    project: Path
    namespace: controller.Namespace
    parent: Path
    destination: Path
    authorization: dict[str, Any]
    authorization_sha256: str
    attempt: dict[str, Any]
    attempt_sha256: str
    api: _SchemaV3Adapter
    phase_calls: list[str]


def _install_public_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_before_callback: bool,
) -> _PublicHarness:
    project = tmp_path.resolve()
    namespace = controller.Namespace.for_project(project)
    namespace.control_root.mkdir(parents=True)
    amendment_root = project / "artifacts" / "preregistration_amendments"
    for component in controller._AMENDMENT_BASELINE:
        (amendment_root / component).mkdir(parents=True)
    parent = amendment_root / controller._AUTHORITY_C_COMPONENT
    destination = amendment_root / "20990101T000000.000000Z"

    authorization_sha256 = "3" * 64
    intent_sha256 = "4" * 64
    authorization_bytes = b'{"synthetic":"authorization-v2"}\n'
    authorization_receipt_sha256 = _sha256(authorization_bytes)
    attempt_id = "6" * 64
    amendment_time = datetime.now(UTC) - timedelta(minutes=2)
    terminal_lineage = {"kind": "synthetic-terminal-qualification"}
    technical_authorization = {"kind": "synthetic-technical-authorization"}
    contract = {
        "terminal_qualification": terminal_lineage,
        "frozen_input_bundle": {"kind": "synthetic-input-v3"},
        "config": {"path": str(project / "synthetic-confirmatory.yaml")},
        "technical_successor": {
            "authorization": technical_authorization,
            "authorization_sha256": authorization_sha256,
            "intent_sha256": intent_sha256,
            "storage_policy": {"kind": "synthetic-storage-policy"},
        },
    }
    authorization = {
        "authorized_attempt_id": attempt_id,
        "publication": {
            "parent_authority_directory": str(parent),
            "intended_authority_directory": str(destination),
            "amendment_timestamp_utc": controller._timestamp(amendment_time),
        },
        "preflight": {
            "contract": contract,
            "preflight_fingerprint_sha256": "9" * 64,
        },
    }
    verification_nonce = controller._verification_nonce_v2(
        authorization,
        authorization_receipt_sha256,
    )
    attempt = {
        "claimed_at_utc": controller._timestamp(datetime.now(UTC) - timedelta(minutes=1)),
        "attempt_id": attempt_id,
        "intended_authority_directory": str(destination),
        "parent_authority_directory": str(parent),
        "technical_authorization_sha256": authorization_sha256,
        "intent_sha256": intent_sha256,
        "verification_nonce": verification_nonce,
        "run_state": {"sha256": "a" * 64},
    }
    attempt_bytes = controller._canonical_bytes(attempt)
    attempt_sha256 = _sha256(attempt_bytes)
    qualification = {"phase": "qualification"}
    qualification_sha256 = _sha256(controller._canonical_bytes(qualification))
    frozen_payloads = {"frozen_source_receipt": {"phase": "input-v3"}}
    frozen_records = {
        "frozen_source_receipt": {
            "path": str(namespace.input_v3 / "frozen.json"),
            "size_bytes": 1,
            "sha256": "b" * 64,
        }
    }
    frozen_root = "c" * 64
    phase_calls: list[str] = []

    def qualify_adapter(**kwargs: Any) -> tuple[dict[str, Any], str]:
        assert kwargs["namespace"] == namespace
        assert kwargs["parent_authority_directory"] == parent
        assert kwargs["pins"] is controller.DEFAULT_HISTORICAL_PINS
        assert callable(kwargs["clock"])
        assert callable(kwargs["process_probe"])
        phase_calls.append("qualify")
        _write_json(namespace.terminal_qualification, qualification)
        return qualification, qualification_sha256

    def freeze_adapter(
        **kwargs: Any,
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        str,
    ]:
        assert kwargs["namespace"] == namespace
        assert kwargs["parent_authority_directory"] == parent
        assert isinstance(
            kwargs["reconstructor"],
            controller.ProductionInputV3Reconstructor,
        )
        assert kwargs["pins"] is controller.DEFAULT_HISTORICAL_PINS
        assert namespace.terminal_qualification.is_file()
        phase_calls.append("freeze")
        _write_json(
            namespace.input_v3 / "frozen.json",
            frozen_payloads["frozen_source_receipt"],
        )
        return frozen_payloads, frozen_records, frozen_root

    def authorize_adapter(**kwargs: Any) -> tuple[dict[str, Any], str]:
        assert kwargs["namespace"] == namespace
        assert kwargs["parent_authority_directory"] == parent
        assert callable(kwargs["clock"])
        assert namespace.input_v3.is_dir()
        phase_calls.append("authorize")
        namespace.authorization_v2.write_bytes(authorization_bytes)
        return authorization, authorization_receipt_sha256

    def read_attempt(
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[dict[str, Any], str]:
        assert namespace.attempt_v2.read_bytes() == attempt_bytes
        return attempt, attempt_sha256

    api = _SchemaV3Adapter(
        destination=destination,
        parent=parent,
        fail_before_callback=fail_before_callback,
        authorization_sha256=authorization_sha256,
        intent_sha256=intent_sha256,
        terminal_lineage=terminal_lineage,
    )

    def api_factory(**kwargs: Any) -> _SchemaV3Adapter:
        assert kwargs["project_root"] == project
        assert kwargs["parent"] == parent
        assert kwargs["destination"] == destination
        assert kwargs["terminal_lineage"] == terminal_lineage
        assert kwargs["expected_authorization_sha256"] == authorization_sha256
        assert kwargs["expected_intent_sha256"] == intent_sha256
        return api

    monkeypatch.setattr(
        controller,
        "_qualify_historical_terminal_once",
        qualify_adapter,
    )
    monkeypatch.setattr(controller, "_freeze_input_v3_once", freeze_adapter)
    monkeypatch.setattr(
        controller,
        "_authorize_publication_v2_once",
        authorize_adapter,
    )
    monkeypatch.setattr(
        controller,
        "_read_terminal_qualification",
        lambda *_args, **_kwargs: (
            qualification,
            qualification_sha256,
        ),
    )
    monkeypatch.setattr(
        controller,
        "_read_input_v3",
        lambda *_args, **_kwargs: (
            frozen_payloads,
            frozen_records,
            frozen_root,
        ),
    )
    monkeypatch.setattr(
        controller,
        "_read_publication_authorization_v2",
        lambda *_args, **_kwargs: (
            authorization,
            authorization_receipt_sha256,
        ),
    )
    monkeypatch.setattr(
        controller,
        "_build_live_preflight_v2",
        lambda **_kwargs: {
            "contract": contract,
            "preflight_fingerprint_sha256": "9" * 64,
            "context": {"kind": "synthetic-context"},
        },
    )
    monkeypatch.setattr(
        controller,
        "_build_attempt_v2",
        lambda **_kwargs: attempt,
    )
    monkeypatch.setattr(controller, "_read_attempt_v2", read_attempt)
    monkeypatch.setattr(
        controller,
        "_verify_live_governed_baseline_v2",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        controller,
        "_stable_amendment_inventory",
        lambda *_args, **_kwargs: (),
    )
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
        "_legacy_scoped_lock_paths",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        controller,
        "_require_legacy_lock_state_under_protocol_lock",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(controller, "ProductionSchemaV3API", api_factory)
    if not fail_before_callback:
        monkeypatch.setattr(
            controller,
            "run_fresh_verifier_v2",
            lambda request, *, timeout_seconds, **_kwargs: _passed_fresh_result(
                request,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _PublicHarness(
        project=project,
        namespace=namespace,
        parent=parent,
        destination=destination,
        authorization=authorization,
        authorization_sha256=authorization_receipt_sha256,
        attempt=attempt,
        attempt_sha256=attempt_sha256,
        api=api,
        phase_calls=phase_calls,
    )


def _classify_without_write(
    harness: _PublicHarness,
    expected: controller.State,
) -> controller.Classification:
    before = _tree_snapshot(harness.project)
    classification = controller.classify(
        harness.namespace,
        parent_authority_directory=harness.parent,
        committed_candidate_verifier=lambda _candidate, _success: None,
    )
    after = _tree_snapshot(harness.project)
    assert after == before
    assert classification.state is expected
    return classification


def _advance_public_phases_to_ready(
    harness: _PublicHarness,
) -> None:
    assert (
        _classify_without_write(
            harness,
            controller.State.QUALIFICATION_REQUIRED,
        ).as_dict()["publication_performed"]
        is False
    )

    qualification, qualification_sha256 = controller.qualify_historical_terminal_once(
        namespace=harness.namespace,
        parent_authority_directory=harness.parent,
    )
    assert qualification == {"phase": "qualification"}
    assert qualification_sha256 == _sha256(controller._canonical_bytes(qualification))
    _classify_without_write(
        harness,
        controller.State.INPUT_FREEZE_REQUIRED,
    )

    frozen_payloads, frozen_records, frozen_root = controller.freeze_input_v3_once(
        namespace=harness.namespace,
        parent_authority_directory=harness.parent,
    )
    assert frozen_payloads["frozen_source_receipt"]["phase"] == "input-v3"
    assert set(frozen_records) == {"frozen_source_receipt"}
    assert frozen_root == "c" * 64
    _classify_without_write(
        harness,
        controller.State.AUTHORIZATION_REQUIRED,
    )

    authorization, authorization_sha256 = controller.authorize_publication_v2_once(
        namespace=harness.namespace,
        parent_authority_directory=harness.parent,
    )
    assert authorization == harness.authorization
    assert authorization_sha256 == harness.authorization_sha256
    ready = _classify_without_write(harness, controller.State.READY)
    assert ready.as_dict()["publication_performed"] is False
    assert harness.phase_calls == ["qualify", "freeze", "authorize"]


def test_public_creator_failure_before_callback_seals_f2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_public_harness(
        tmp_path,
        monkeypatch,
        fail_before_callback=True,
    )
    _advance_public_phases_to_ready(harness)

    result = controller.publish_replacement_authority_once(
        namespace=harness.namespace,
        parent_authority_directory=harness.parent,
    )

    assert result.state is controller.State.ROLLED_BACK_FAILURE
    assert result.marker_path == harness.namespace.failure_v2
    assert result.authority_directory is None
    assert harness.namespace.attempt_v2.is_file()
    assert harness.namespace.failure_v2.is_file()
    assert not harness.namespace.success_v2.exists()
    assert not harness.destination.exists()
    assert harness.api.creator_called is True
    assert harness.api.callback_received is True
    assert harness.api.callback_called is False
    assert harness.api.committed_verifications == 0
    failure = json.loads(harness.namespace.failure_v2.read_text("utf-8"))
    assert failure["status"] == "rolled_back_failure_no_retry"
    assert failure["failure_phase"] == "authority_creation_before_fresh_verifier"
    assert failure["fresh_verifier_diagnostic"]["status"] == "not_invoked"
    assert failure["publication_performed"] is False
    sealed = _classify_without_write(
        harness,
        controller.State.ROLLED_BACK_FAILURE,
    )
    assert sealed.failure_v2_sha256 == result.marker_sha256
    assert sealed.as_dict()["publication_performed"] is False


def test_public_callback_pass_seals_exact_s2_and_d(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_public_harness(
        tmp_path,
        monkeypatch,
        fail_before_callback=False,
    )
    _advance_public_phases_to_ready(harness)

    result = controller.publish_replacement_authority_once(
        namespace=harness.namespace,
        parent_authority_directory=harness.parent,
    )

    assert result.state is controller.State.COMMITTED
    assert result.marker_path == harness.namespace.success_v2
    assert result.authority_directory == harness.destination
    assert harness.namespace.attempt_v2.is_file()
    assert harness.namespace.success_v2.is_file()
    assert not harness.namespace.failure_v2.exists()
    assert harness.destination.is_dir()
    assert harness.api.creator_called is True
    assert harness.api.callback_received is True
    assert harness.api.callback_called is True
    assert harness.api.committed_verifications == 2
    success = json.loads(harness.namespace.success_v2.read_text("utf-8"))
    assert success["status"] == "committed"
    assert success["authority_directory"] == str(harness.destination)
    assert success["fresh_verifier_diagnostic"]["status"] == "passed"
    assert success["publication_performed"] is True
    committed = _classify_without_write(
        harness,
        controller.State.COMMITTED,
    )
    assert committed.candidates == (harness.destination,)
    assert committed.success_v2_sha256 == result.marker_sha256
    assert committed.as_dict()["publication_performed"] is True
