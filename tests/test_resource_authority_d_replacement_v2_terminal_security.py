"""Security regression tests for the replacement-v1 terminal qualification."""

from __future__ import annotations

import copy
import hashlib
import inspect
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from histo_audit.pannuke.publication import ExclusiveBundlePublicationLock
from histo_audit.workflows import (
    resource_authority_d_replacement_v2_controller as controller,
)

_CONSUMED_AT = datetime(2026, 7, 28, 18, 19, 20, 303224, tzinfo=UTC)
_QUALIFIED_AT = datetime(2026, 7, 28, 19, 0, 30, tzinfo=UTC)
_PROCESS_AT = _QUALIFIED_AT - timedelta(seconds=30)


def _record(path: Path, size_bytes: int, sha256: str) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def _timestamp(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _set_path(payload: dict[str, Any], keys: tuple[str | int, ...], value: Any) -> None:
    current: Any = payload
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


def _synthetic_terminal_receipt(
    tmp_path: Path,
) -> tuple[
    controller.Namespace,
    Path,
    dict[str, Any],
    controller.HistoricalPins,
]:
    """Build one exact receipt without depending on ignored local evidence."""

    project = (tmp_path / "project").resolve()
    control_root = project / "artifacts" / "resource_control"
    control_root.mkdir(parents=True)
    namespace = controller.Namespace.for_project(project)
    parent = (
        project / "artifacts" / "preregistration_amendments" / controller._AUTHORITY_C_COMPONENT
    )
    amendment_root = parent.parent
    historical_d1 = amendment_root / controller._HISTORICAL_D1_COMPONENT
    run_root = project / "artifacts" / "runs"
    v2_root = control_root / controller._HISTORICAL_INPUT_V2_DIRECTORY_NAME
    pins = controller.DEFAULT_HISTORICAL_PINS

    authority_files = {
        role: _record(parent / filename, size_bytes, sha256)
        for role, (filename, size_bytes, sha256) in controller._AUTHORITY_C_FILE_PINS.items()
    }
    authority = {
        "directory": str(parent),
        "schema_version": 4,
        "chain_depth": 3,
        "artifact_root_sha256": controller._AUTHORITY_C_ARTIFACT_ROOT_SHA256,
        "sha256_manifest_sha256": controller._AUTHORITY_C_MANIFEST_SHA256,
        "flat_file_count": 8,
        "files": authority_files,
        "integrity_verified": True,
    }
    authorization_record = _record(
        control_root / controller._HISTORICAL_AUTH_V1_FILENAME,
        pins.authorization_v1.size_bytes,
        pins.authorization_v1.sha256,
    )
    attempt_record = _record(
        control_root / controller._HISTORICAL_ATTEMPT_V1_FILENAME,
        pins.attempt_v1.size_bytes,
        pins.attempt_v1.sha256,
    )
    failure_record = _record(
        control_root / controller._HISTORICAL_FAILURE_V1_FILENAME,
        pins.failure_v1.size_bytes,
        pins.failure_v1.sha256,
    )
    frozen_files = {
        role: _record(
            v2_root / filename,
            pins.input_v2[role].size_bytes,
            pins.input_v2[role].sha256,
        )
        for role, filename in controller._HISTORICAL_INPUT_V2_FILENAMES.items()
    }
    pins = replace(
        pins,
        input_v2_records_sha256=controller._compact_sha256(frozen_files),
    )
    protected = {
        role: _record(project / relative_path, size_bytes, sha256)
        for role, (
            relative_path,
            size_bytes,
            sha256,
        ) in controller._PROTECTED_BINDINGS.items()
    }
    run_files = {
        filename: _record(run_root / filename, size_bytes, sha256)
        for filename, (size_bytes, sha256) in controller._RUN_STATE_PINS.items()
    }
    run_state = {
        "root": str(run_root),
        "files": run_files,
        "sha256": pins.run_state_sha256,
    }
    qualifying_controller = controller._controller_identity()
    old_controller = (
        project
        / "src"
        / "histo_audit"
        / "workflows"
        / "resource_authority_d_replacement_controller.py"
    )
    terminal = {
        "classification": "rolled_back_failure",
        "classification_reason": "exact A+F exist and S/D/candidates are absent",
        "publication_authorization_receipt": authorization_record,
        "attempt_marker": attempt_record,
        "failure_marker": failure_record,
        "success_marker_absence": {
            "path": str(control_root / controller._HISTORICAL_SUCCESS_V1_FILENAME),
            "absent": True,
        },
        "intended_authority_absence": {
            "path": str(historical_d1),
            "absent": True,
        },
        "candidate_count": 0,
        "candidate_paths": [],
    }
    auxiliary = {
        "retired_v1_invalidation_receipt": (
            controller._HISTORICAL_INVALIDATION_FILENAME,
            pins.invalidation,
        ),
        "prior_publication_failure_receipt": (
            controller._HISTORICAL_PRIOR_FAILURE_FILENAME,
            pins.prior_failure,
        ),
        "failed_preflight_receipt": (
            controller._HISTORICAL_FAILED_PREFLIGHT_FILENAME,
            pins.failed_preflight,
        ),
    }
    reads: list[dict[str, Any]] = [
        {"role": "publication_authorization_receipt", **authorization_record},
        {"role": "publication_attempt_marker", **attempt_record},
        {"role": "publication_failure_marker", **failure_record},
    ]
    reads.extend(
        {
            "role": role,
            **_record(control_root / filename, pin.size_bytes, pin.sha256),
        }
        for role, (filename, pin) in auxiliary.items()
    )
    reads.extend(
        {"role": f"v2_{role}", **frozen_files[role]}
        for role in (
            "cnn_correction_receipt",
            "frozen_source_receipt",
            "source_allowlist",
            "workspace_plan",
        )
    )
    reads.extend(
        {"role": f"protected_{role}", **protected[role]}
        for role in (
            "specification",
            "pre_registration",
            "primary_config",
            "confirmatory_config",
        )
    )
    reads.extend(
        {"role": f"authority_c_{role}", **authority_files[role]}
        for role in (
            "amendment_evidence",
            "amendment_report",
            "confirmatory_config",
            "immutable_marker",
            "preregistration",
            "primary_config",
            "sha256_manifest",
            "source_tree_manifest",
        )
    )
    reads.extend(
        {"role": f"run_state_{filename}", **run_files[filename]}
        for filename in controller._RUN_STATE_FILENAMES
    )
    assert len(reads) == 28

    receipt = {
        "schema_version": 1,
        "policy": controller.TERMINAL_QUALIFICATION_POLICY,
        "status": controller.TERMINAL_QUALIFICATION_STATUS,
        "qualified_at_utc": _timestamp(_QUALIFIED_AT),
        "project_root": str(project),
        "authority_c": authority,
        "terminal_namespace": terminal,
        "terminal_links": {
            "attempt_id": pins.attempt_id,
            "max_attempt_count": 1,
            "automatic_retry_allowed": False,
            "authorization_receipt_sha256": pins.authorization_v1.sha256,
            "attempt_marker_sha256": pins.attempt_v1.sha256,
            "technical_successor_authorization_sha256": (pins.technical_authorization_sha256),
            "intent_sha256": pins.intent_sha256,
            "preflight_fingerprint_sha256": pins.preflight_fingerprint_sha256,
            "amendment_timestamp_utc": "2026-07-28T18:19:20.303224Z",
            "parent_authority_directory": str(parent),
            "run_state_sha256": pins.run_state_sha256,
        },
        "frozen_v2_inputs": {
            "directory": str(v2_root),
            "files": frozen_files,
            "execution_source_root_sha256": controller._HISTORICAL_V2_SOURCE_ROOT_SHA256,
            "execution_source_manifest_sha256": (controller._HISTORICAL_V2_SOURCE_MANIFEST_SHA256),
            "execution_source_delta_sha256": controller._HISTORICAL_V2_SOURCE_DELTA_SHA256,
            "records_sha256": pins.input_v2_records_sha256,
        },
        "controller_identities": {
            "consumed_attempt_controller": {
                "path": str(old_controller),
                "size_bytes": pins.controller.size_bytes,
                "sha256": pins.controller.sha256,
                "live_file_match": False,
                "attested_by": [
                    "retired_v1_invalidation_receipt",
                    "attempt_marker",
                    "publication_authorization_receipt",
                    "v2_frozen_source_receipt",
                    "v2_source_allowlist",
                ],
            },
            "diagnosed_fixed_legacy_controller": {
                "path": str(old_controller),
                "size_bytes": controller._DIAGNOSED_FIXED_LEGACY_CONTROLLER_SIZE_BYTES,
                "sha256": controller._DIAGNOSED_FIXED_LEGACY_CONTROLLER_SHA256,
                "distinct_from_consumed_attempt_controller": True,
                "authorized_to_retry_v1": False,
                "diagnostic_scope": ("future_process_boundary_regression_only_no_v1_retry"),
            },
            "qualifying_live_controller": {
                **qualifying_controller,
                "distinct_from_consumed_attempt_controller": True,
                "authorized_to_retry_v1": False,
            },
        },
        "failure_cause": {
            "error_type": controller._HISTORICAL_FAILURE_ERROR_TYPE,
            "error_type_sha256": pins.failure_error_type_sha256,
            "error_text": controller._HISTORICAL_FAILURE_ERROR_TEXT,
            "error_sha256": pins.failure_error_sha256,
            "reason_code": "windows_venv_launcher_breaks_direct_child_ppid_contract",
            "scientific_or_evidence_corruption": False,
        },
        "run_state": run_state,
        "protected_bindings": protected,
        "process_quiescence": {
            "query_method": controller._PROCESS_QUERY_METHOD,
            "observer_pid": 12345,
            "observed_at_utc": _timestamp(_PROCESS_AT),
            "matches": [],
            "historical_pid_inference_performed": False,
        },
        "lock_quiescence": {
            "scan_method": controller._LOCK_SCAN_METHOD,
            "first_scan_paths": [],
            "second_scan_paths": [],
            "reads_between_scans": reads,
        },
        "disposition": {
            "v1_attempt_consumed": True,
            "v1_retry_allowed": False,
            "v1_artifacts_may_be_modified_moved_or_deleted": False,
            "successor_requires_new_namespace": True,
            "successor_may_reuse_v2_inputs": False,
            "qualification_authorizes_publication": False,
            "outcome_values_read": False,
        },
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }
    return namespace, parent, receipt, pins


@pytest.mark.parametrize(
    ("keys", "replacement"),
    [
        (("schema_version",), True),
        (("schema_version",), 1.0),
        (("authority_c", "chain_depth"), 3.0),
        (("authority_c", "flat_file_count"), 8.0),
        (("authority_c", "files", "amendment_evidence", "size_bytes"), 16677.0),
        (("terminal_namespace", "candidate_count"), False),
        (("terminal_namespace", "candidate_count"), 0.0),
        (
            (
                "terminal_namespace",
                "publication_authorization_receipt",
                "size_bytes",
            ),
            9396.0,
        ),
        (("terminal_links", "max_attempt_count"), True),
        (("terminal_links", "max_attempt_count"), 1.0),
        (
            (
                "controller_identities",
                "consumed_attempt_controller",
                "size_bytes",
            ),
            216288.0,
        ),
        (
            (
                "controller_identities",
                "diagnosed_fixed_legacy_controller",
                "size_bytes",
            ),
            218766.0,
        ),
        (("run_state", "files", "registry.csv", "size_bytes"), 8123.0),
        (("protected_bindings", "specification", "size_bytes"), 11275.0),
        (("failure_cause", "scientific_or_evidence_corruption"), 0),
        (("disposition", "v1_retry_allowed"), 0),
    ],
)
def test_terminal_canonicalizer_rejects_numeric_and_boolean_aliases(
    tmp_path: Path,
    keys: tuple[str | int, ...],
    replacement: Any,
) -> None:
    namespace, _parent, receipt, pins = _synthetic_terminal_receipt(tmp_path)
    tampered = copy.deepcopy(receipt)
    _set_path(tampered, keys, replacement)

    with pytest.raises(controller.ControlError):
        controller._canonical_terminal_receipt(
            tampered,
            namespace=namespace,
            pins=pins,
        )


def test_terminal_canonicalizer_requires_exact_ordered_reads_and_run_state(
    tmp_path: Path,
) -> None:
    namespace, _parent, receipt, pins = _synthetic_terminal_receipt(tmp_path)
    canonical = controller._canonical_terminal_receipt(
        copy.deepcopy(receipt),
        namespace=namespace,
        pins=pins,
    )
    assert canonical == receipt
    assert len(canonical["lock_quiescence"]["reads_between_scans"]) == 28
    assert list(canonical["run_state"]["files"]) == list(controller._RUN_STATE_FILENAMES)

    tampered_payloads: list[dict[str, Any]] = []

    reversed_reads = copy.deepcopy(receipt)
    reversed_reads["lock_quiescence"]["reads_between_scans"].reverse()
    tampered_payloads.append(reversed_reads)

    duplicate_reads = copy.deepcopy(receipt)
    first = duplicate_reads["lock_quiescence"]["reads_between_scans"][0]
    duplicate_reads["lock_quiescence"]["reads_between_scans"] = [
        copy.deepcopy(first) for _ in range(28)
    ]
    tampered_payloads.append(duplicate_reads)

    wrong_read_record = copy.deepcopy(receipt)
    wrong_read_record["lock_quiescence"]["reads_between_scans"][0]["size_bytes"] = 9396.0
    tampered_payloads.append(wrong_read_record)

    wrong_run_record = copy.deepcopy(receipt)
    wrong_run_record["run_state"]["files"]["registry.csv"] = copy.deepcopy(
        wrong_run_record["run_state"]["files"]["integrity_registry.jsonl"]
    )
    tampered_payloads.append(wrong_run_record)

    missing_run_record = copy.deepcopy(receipt)
    missing_run_record["run_state"]["files"].pop("registry.csv")
    tampered_payloads.append(missing_run_record)

    wrong_run_root = copy.deepcopy(receipt)
    wrong_run_root["run_state"]["sha256"] = "0" * 64
    tampered_payloads.append(wrong_run_root)

    for tampered in tampered_payloads:
        with pytest.raises(controller.ControlError):
            controller._canonical_terminal_receipt(
                tampered,
                namespace=namespace,
                pins=pins,
            )


def test_terminal_canonicalizer_rejects_future_negative_and_stale_process_time(
    tmp_path: Path,
) -> None:
    namespace, _parent, receipt, pins = _synthetic_terminal_receipt(tmp_path)

    future = copy.deepcopy(receipt)
    future_qualified = datetime.now(UTC) + timedelta(days=1)
    future["qualified_at_utc"] = _timestamp(future_qualified)
    future["process_quiescence"]["observed_at_utc"] = _timestamp(
        future_qualified - timedelta(seconds=1)
    )

    stale = copy.deepcopy(receipt)
    stale["process_quiescence"]["observed_at_utc"] = _timestamp(
        _QUALIFIED_AT - timedelta(seconds=61)
    )

    negative = copy.deepcopy(receipt)
    negative["process_quiescence"]["observed_at_utc"] = _timestamp(
        _QUALIFIED_AT + timedelta(microseconds=1)
    )

    for tampered in (future, stale, negative):
        with pytest.raises(controller.ControlError):
            controller._canonical_terminal_receipt(
                tampered,
                namespace=namespace,
                pins=pins,
            )

    boundary = copy.deepcopy(receipt)
    boundary["process_quiescence"]["observed_at_utc"] = _timestamp(
        _QUALIFIED_AT - timedelta(seconds=60)
    )
    assert (
        controller._canonical_terminal_receipt(
            boundary,
            namespace=namespace,
            pins=pins,
        )
        == boundary
    )


def test_public_terminal_qualifier_has_no_forgeable_seams() -> None:
    signature = inspect.signature(controller.qualify_historical_terminal_once)
    assert tuple(signature.parameters) == (
        "namespace",
        "parent_authority_directory",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert {
        "pins",
        "clock",
        "qualified_at",
        "process_probe",
    }.isdisjoint(signature.parameters)


def test_private_qualifier_uses_second_probe_under_both_locks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = (tmp_path / "project").resolve()
    parent = (
        project / "artifacts" / "preregistration_amendments" / controller._AUTHORITY_C_COMPONENT
    )
    namespace = controller.Namespace.for_project(project)
    lock_state = {"protocol": False, "parent": False}
    probe_states: list[tuple[bool, bool]] = []
    parent_lock_arguments: list[tuple[tuple[Path, ...], str]] = []
    owned_lock_calls: list[tuple[Any, ...]] = []
    built_processes: list[dict[str, Any]] = []
    candidate: dict[str, Any] = {}
    encoded_publication = b""
    rollback_calls: list[list[Any]] = []
    force_final_match = False

    class FakeLock:
        def __init__(self, name: str) -> None:
            self.name = name
            self.lock_paths = (tmp_path / f"{name}.lock",)

        def __enter__(self) -> FakeLock:
            assert lock_state[self.name] is False
            lock_state[self.name] = True
            return self

        def __exit__(self, *_args: object) -> None:
            lock_state[self.name] = False

        def assert_owned(self) -> None:
            assert lock_state[self.name] is True

    class FakePublished:
        def __init__(self, encoded: bytes) -> None:
            self.sha256 = hashlib.sha256(encoded).hexdigest()

        @staticmethod
        def still_owned() -> bool:
            return True

    protocol_lock = FakeLock("protocol")
    parent_lock = FakeLock("parent")

    monkeypatch.setattr(
        controller,
        "_require_parent",
        lambda *_args, **_kwargs: (project, parent),
    )
    monkeypatch.setattr(controller, "_stable_amendment_inventory", lambda *_a, **_k: ())
    monkeypatch.setattr(
        controller,
        "_require_no_preliminary_successor_assets",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(controller, "discover_candidates", lambda *_a, **_k: ())
    monkeypatch.setattr(controller, "_legacy_scoped_lock_paths", lambda *_a, **_k: ())
    monkeypatch.setattr(controller, "_scan_present_paths", lambda *_a, **_k: [])
    monkeypatch.setattr(
        controller,
        "_protocol_lock",
        lambda *_a, **_k: protocol_lock,
    )

    def parent_lock_factory(paths: Any, *, role: str) -> FakeLock:
        materialized = tuple(Path(path) for path in paths)
        parent_lock_arguments.append((materialized, role))
        return parent_lock

    monkeypatch.setattr(controller, "ExclusiveBundlePublicationLock", parent_lock_factory)

    def require_owned(*, legacy_paths: Any, owned_locks: Any) -> None:
        assert tuple(legacy_paths) == ()
        materialized = tuple(owned_locks)
        assert materialized == (protocol_lock, parent_lock)
        for lock in materialized:
            lock.assert_owned()
        owned_lock_calls.append(materialized)

    monkeypatch.setattr(
        controller,
        "_require_legacy_lock_state_under_protocol_lock",
        require_owned,
    )
    monkeypatch.setattr(
        controller,
        "_historical_terminal_payloads",
        lambda *_a, **_k: (
            project,
            parent,
            {},
            {},
            {},
            {"authorization_record": {}},
            {"attempt_record": {}},
            {"failure_record": {}},
        ),
    )
    monkeypatch.setattr(
        controller,
        "_historical_support_records",
        lambda **_kwargs: ({}, {}, {}, []),
    )

    process_times = (
        _CONSUMED_AT + timedelta(seconds=1),
        _CONSUMED_AT + timedelta(seconds=2),
        _CONSUMED_AT + timedelta(seconds=3),
    )

    def process_probe(observer_pid: int) -> dict[str, Any]:
        assert observer_pid == os.getpid()
        probe_states.append((lock_state["protocol"], lock_state["parent"]))
        observed = process_times[len(probe_states) - 1]
        matches = [{"process_id": 99999}] if force_final_match and len(probe_states) == 3 else []
        return {
            "query_method": controller._PROCESS_QUERY_METHOD,
            "observer_pid": observer_pid,
            "observed_at_utc": _timestamp(observed),
            "matches": matches,
            "historical_pid_inference_performed": False,
        }

    qualified_at = process_times[1] + timedelta(seconds=1)
    clock_values = iter((qualified_at, qualified_at))

    def build_receipt(
        _namespace: controller.Namespace,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert lock_state == {"protocol": True, "parent": True}
        process = dict(kwargs["process_quiescence"])
        built_processes.append(process)
        candidate.clear()
        candidate.update(
            {
                "qualified_at_utc": _timestamp(kwargs["qualified_at"]),
                "process_quiescence": process,
            }
        )
        return dict(candidate)

    monkeypatch.setattr(controller, "_build_terminal_receipt", build_receipt)

    def publish(encoded: bytes, destination: Path) -> FakePublished:
        nonlocal encoded_publication
        assert lock_state == {"protocol": True, "parent": True}
        assert destination == namespace.terminal_qualification
        encoded_publication = encoded
        return FakePublished(encoded)

    monkeypatch.setattr(controller, "publish_bytes_no_overwrite", publish)

    def readback(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], str]:
        assert lock_state == {"protocol": True, "parent": True}
        return dict(candidate), hashlib.sha256(encoded_publication).hexdigest()

    monkeypatch.setattr(controller, "_read_terminal_qualification", readback)
    monkeypatch.setattr(
        controller,
        "rollback_owned_publications",
        lambda publications: rollback_calls.append(list(publications)),
    )
    monkeypatch.setattr(
        controller,
        "_reserved_family_presence",
        lambda *_a, **_k: {
            "qualification": True,
            "inputs": False,
            "authorization": False,
            "attempt": False,
            "success": False,
            "failure": False,
        },
    )

    receipt, digest = controller._qualify_historical_terminal_once(
        namespace=namespace,
        parent_authority_directory=parent,
        pins=controller.DEFAULT_HISTORICAL_PINS,
        clock=lambda: next(clock_values),
        process_probe=process_probe,
    )

    assert probe_states == [
        (False, False),
        (True, True),
        (True, True),
    ]
    assert built_processes == [
        {
            "query_method": controller._PROCESS_QUERY_METHOD,
            "observer_pid": os.getpid(),
            "observed_at_utc": _timestamp(process_times[1]),
            "matches": [],
            "historical_pid_inference_performed": False,
        }
    ]
    assert receipt["process_quiescence"] == built_processes[0]
    assert digest == hashlib.sha256(encoded_publication).hexdigest()
    assert parent_lock_arguments == [
        (
            (parent,),
            "resource Authority-D replacement-v2 Authority-C parent guard",
        )
    ]
    assert owned_lock_calls
    assert lock_state == {"protocol": False, "parent": False}

    probe_states.clear()
    owned_lock_calls.clear()
    built_processes.clear()
    candidate.clear()
    encoded_publication = b""
    rollback_calls.clear()
    force_final_match = True
    clock_values = iter((qualified_at, qualified_at))

    with pytest.raises(controller.ControlError, match="process quiescence"):
        controller._qualify_historical_terminal_once(
            namespace=namespace,
            parent_authority_directory=parent,
            pins=controller.DEFAULT_HISTORICAL_PINS,
            clock=lambda: next(clock_values),
            process_probe=process_probe,
        )

    assert probe_states == [
        (False, False),
        (True, True),
        (True, True),
    ]
    assert len(rollback_calls) == 1
    assert len(rollback_calls[0]) == 1
    assert not namespace.terminal_qualification.exists()
    assert lock_state == {"protocol": False, "parent": False}


def test_creator_parent_guard_and_future_protocol_lock_topologies(tmp_path: Path) -> None:
    project = (tmp_path / "project").resolve()
    namespace = controller.Namespace.for_project(project)
    amendment_root = project / "artifacts" / "preregistration_amendments"
    for component in controller._AMENDMENT_BASELINE:
        (amendment_root / component).mkdir(parents=True)
    parent = amendment_root / controller._AUTHORITY_C_COMPONENT
    historical_d1 = amendment_root / controller._HISTORICAL_D1_COMPONENT

    legacy_locks = controller._legacy_topology_locks(namespace, parent=parent)
    assert tuple(len(lock.lock_paths) for lock in legacy_locks) == (7, 6, 6, 3)
    assert len({path for lock in legacy_locks for path in lock.lock_paths}) == 16
    creator = next(lock for lock in legacy_locks if "amendment-creator" in lock.role)
    assert set(creator.lock_paths).issubset(
        set(controller._legacy_scoped_lock_paths(namespace, parent=parent))
    )

    expected_future_targets = (
        namespace.terminal_qualification,
        namespace.input_v3,
        namespace.authorization_v2,
        namespace.attempt_v2,
        namespace.success_v2,
        namespace.failure_v2,
        namespace.control_root / controller._HISTORICAL_INPUT_V2_DIRECTORY_NAME,
        namespace.control_root / controller._HISTORICAL_AUTH_V1_FILENAME,
        namespace.control_root / controller._HISTORICAL_ATTEMPT_V1_FILENAME,
        namespace.control_root / controller._HISTORICAL_SUCCESS_V1_FILENAME,
        namespace.control_root / controller._HISTORICAL_FAILURE_V1_FILENAME,
    )
    assert controller._protocol_lock_paths(namespace, parent=parent) == expected_future_targets
    future_lock = controller._protocol_lock(
        namespace,
        parent=parent,
        role="test future protocol topology",
    )
    assert len(future_lock.logical_paths) == 11
    assert len(future_lock.lock_paths) == 12
    assert set(future_lock.lock_paths).isdisjoint(creator.lock_paths)

    parent_guard = ExclusiveBundlePublicationLock(
        (parent,),
        role="test terminal-qualification Authority-C parent guard",
    )
    assert len(parent_guard.logical_paths) == 1
    assert len(parent_guard.lock_paths) == 2
    shared_creator_paths = set(parent_guard.lock_paths) & set(creator.lock_paths)
    assert len(shared_creator_paths) == 1
    assert next(iter(shared_creator_paths)).name.startswith("target-")

    with (
        controller._protocol_lock(
            namespace,
            parent=parent,
            role="test unguarded candidate injection",
        ),
        ExclusiveBundlePublicationLock(
            (parent, historical_d1),
            role="test concurrent amendment creator",
        ),
    ):
        historical_d1.mkdir()
        assert controller.discover_candidates(parent) == (historical_d1.resolve(),)
    historical_d1.rmdir()

    with (
        controller._protocol_lock(
            namespace,
            parent=parent,
            role="test guarded candidate injection",
        ),
        ExclusiveBundlePublicationLock(
            (parent,),
            role="test terminal-qualification parent guard",
        ),
    ):
        with (
            pytest.raises(FileExistsError),
            ExclusiveBundlePublicationLock(
                (parent, historical_d1),
                role="test blocked concurrent amendment creator",
            ),
        ):
            pytest.fail("creator unexpectedly acquired the shared target-C lock")
        assert not historical_d1.exists()


def test_consumed_controller_requires_five_semantic_attestations(tmp_path: Path) -> None:
    namespace, _parent, receipt, synthetic_pins = _synthetic_terminal_receipt(tmp_path)
    attestations = receipt["controller_identities"]["consumed_attempt_controller"]["attested_by"]
    assert attestations == [
        "retired_v1_invalidation_receipt",
        "attempt_marker",
        "publication_authorization_receipt",
        "v2_frozen_source_receipt",
        "v2_source_allowlist",
    ]

    reordered = copy.deepcopy(receipt)
    reordered["controller_identities"]["consumed_attempt_controller"]["attested_by"].reverse()
    with pytest.raises(controller.ControlError):
        controller._canonical_terminal_receipt(
            reordered,
            namespace=namespace,
            pins=synthetic_pins,
        )

    helper_parameters = inspect.signature(
        controller._require_historical_controller_attestations
    ).parameters
    assert "invalidation" in helper_parameters

    project_root = Path(controller.__file__).resolve().parents[3]
    old_controller = (
        project_root
        / "src"
        / "histo_audit"
        / "workflows"
        / "resource_authority_d_replacement_controller.py"
    )
    pins = controller.DEFAULT_HISTORICAL_PINS
    historical_record = {
        "path": str(old_controller),
        "size_bytes": pins.controller.size_bytes,
        "sha256": pins.controller.sha256,
    }
    authorization = {
        "preflight": {
            "contract": {
                "controller": copy.deepcopy(historical_record),
            }
        }
    }
    attempt = {
        "controller_path": str(old_controller),
        "controller_size_bytes": pins.controller.size_bytes,
        "controller_sha256": pins.controller.sha256,
    }
    invalidation = {"corrected_controller": copy.deepcopy(historical_record)}
    input_payloads = {
        "frozen_source_receipt": {
            "controller_path": str(old_controller),
            "controller_size_bytes": pins.controller.size_bytes,
            "controller_sha256": pins.controller.sha256,
        },
        "source_allowlist": {
            "records": [
                {
                    "change_kind": "added",
                    "path": (
                        "src/histo_audit/workflows/resource_authority_d_replacement_controller.py"
                    ),
                    "size_bytes": pins.controller.size_bytes,
                    "sha256": pins.controller.sha256,
                }
            ]
        },
    }
    arguments = {
        "authorization": authorization,
        "attempt": attempt,
        "invalidation": invalidation,
        "input_payloads": input_payloads,
        "expected_controller_path": old_controller,
        "pins": pins,
    }
    assert controller._require_historical_controller_attestations(**arguments) == str(
        old_controller
    )

    mutations = (
        (("invalidation", "corrected_controller", "sha256"),),
        (("authorization", "preflight", "contract", "controller", "sha256"),),
        (("attempt", "controller_sha256"),),
        (("input_payloads", "frozen_source_receipt", "controller_sha256"),),
        (("input_payloads", "source_allowlist", "records", 0, "sha256"),),
    )
    for (keys,) in mutations:
        tampered = copy.deepcopy(arguments)
        _set_path(tampered, keys, "0" * 64)
        with pytest.raises(controller.ControlError):
            controller._require_historical_controller_attestations(**tampered)
