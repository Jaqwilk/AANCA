from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import histo_audit.experiment.resource_bounded_runner as resource_bounded_runner
import histo_audit.workflows as workflows
import histo_audit.workflows.preregistration_amendment as amendment
from histo_audit.workflows.preregistration_amendment import (
    PreregistrationAmendmentResult,
)

CONTROLLER_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "histo_audit"
    / "workflows"
    / "resource_authority_d_replacement_controller.py"
)


def _load_controller() -> Any:
    name = "prepare_resource_authority_d_replacement_once_for_test"
    spec = importlib.util.spec_from_file_location(name, CONTROLLER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def controller() -> Any:
    return _load_controller()


@pytest.fixture
def control_tree(
    tmp_path: Path,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    root = tmp_path / "project"
    control_root = root / "artifacts" / "resource_control"
    amendment_root = root / "artifacts" / "preregistration_amendments"
    parent = amendment_root / "authority_c"
    amendment_timestamp = "2026-07-28T12:00:00.000000Z"
    destination = amendment_root / "20260728T120000.000000Z"
    control_root.mkdir(parents=True)
    parent.mkdir(parents=True)
    run_root = root / "artifacts" / "runs"
    run_root.mkdir()
    for index, filename in enumerate(controller._RUN_STATE_FILENAMES):
        (run_root / filename).write_bytes(f"run-state-{index}\n".encode())
    failed_preflight = control_root / "failed_resource_preflight_test.json"
    prior_failure = control_root / controller._PRIOR_FAILURE_RECEIPT_FILENAME
    failed_preflight.write_bytes(b'{"status":"failed"}\n')
    prior_failure.write_bytes(b'{"status":"verified"}\n')
    monkeypatch.setattr(
        controller,
        "_FAILED_PREFLIGHT_FILENAME",
        failed_preflight.name,
    )
    monkeypatch.setattr(
        controller,
        "_FAILED_PREFLIGHT_SIZE_BYTES",
        failed_preflight.stat().st_size,
    )
    monkeypatch.setattr(
        controller,
        "_FAILED_PREFLIGHT_SHA256",
        hashlib.sha256(failed_preflight.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        controller,
        "_PRIOR_FAILURE_RECEIPT_SIZE_BYTES",
        prior_failure.stat().st_size,
    )
    monkeypatch.setattr(
        controller,
        "_PRIOR_FAILURE_RECEIPT_SHA256",
        hashlib.sha256(prior_failure.read_bytes()).hexdigest(),
    )
    retired_root = control_root / controller._RETIRED_INPUT_DIRECTORY_NAME
    retired_root.mkdir()
    for role, filename in controller._REPLACEMENT_INPUT_FILENAMES.items():
        (retired_root / filename).write_bytes(f"retired-{role}\n".encode())
    invalidation = control_root / controller._RETIRED_INPUT_INVALIDATION_FILENAME
    invalidation_payload = {"status": "synthetic-invalidated-v1"}
    invalidation_bytes = controller._canonical_bytes(invalidation_payload)
    invalidation.write_bytes(invalidation_bytes)
    invalidation_sha256 = hashlib.sha256(invalidation_bytes).hexdigest()

    def read_synthetic_invalidation(
        _namespace: Any,
        **_kwargs: Any,
    ) -> tuple[dict[str, Any], str]:
        encoded = invalidation.read_bytes()
        if encoded != invalidation_bytes:
            raise controller.ControlError("synthetic retired invalidation changed")
        return invalidation_payload, hashlib.sha256(encoded).hexdigest()

    monkeypatch.setattr(
        controller,
        "_read_retired_input_invalidation",
        read_synthetic_invalidation,
    )
    frozen_root = control_root / controller._REPLACEMENT_INPUT_DIRECTORY_NAME
    frozen_root.mkdir()
    frozen_paths = {
        role: frozen_root / filename
        for role, filename in controller._REPLACEMENT_INPUT_FILENAMES.items()
    }
    frozen_paths["source_allowlist"].write_text(
        json.dumps(
            {
                "records": [
                    {
                        "path": "src/histo_audit/example.py",
                        "change_kind": "modified",
                    }
                ]
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    frozen_paths["workspace_plan"].write_text(
        json.dumps({"plan": "test"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frozen_paths["cnn_correction_receipt"].write_text(
        json.dumps({"correction": {"correction": "test"}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frozen_paths["frozen_source_receipt"].write_text(
        json.dumps(
            {
                f"{role}_sha256": hashlib.sha256(frozen_paths[role].read_bytes()).hexdigest()
                for role in (
                    "source_allowlist",
                    "workspace_plan",
                    "cnn_correction_receipt",
                )
            }
            | {
                "execution_source_root_sha256": "6" * 64,
                "execution_source_manifest_sha256": "7" * 64,
                "execution_source_delta_sha256": "8" * 64,
                "retired_input_invalidation_receipt_path": str(invalidation),
                "retired_input_invalidation_receipt_sha256": invalidation_sha256,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    frozen_inputs = controller.FrozenInputBindings(
        failed_preflight_receipt=failed_preflight,
        prior_failure_receipt=prior_failure,
        retired_input_invalidation=invalidation,
        frozen_source_receipt=frozen_paths["frozen_source_receipt"],
        source_allowlist=frozen_paths["source_allowlist"],
        workspace_plan=frozen_paths["workspace_plan"],
        cnn_correction_receipt=frozen_paths["cnn_correction_receipt"],
    )
    namespace = controller.Namespace(control_root)
    config_path = root / "configs" / "confirmatory_resource_bounded_amended.yaml"
    config_path.parent.mkdir()
    config_path.write_bytes(b"schema_version: 1\n")
    manifest_path = root / "data" / "manifests" / "pannuke" / "pannuke_nucleus_manifest.parquet"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(b"synthetic-pannuke-manifest\n")

    def file_record(path: Path) -> dict[str, Any]:
        encoded = path.read_bytes()
        return {
            "path": str(path),
            "size_bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }

    run_files = {
        name: hashlib.sha256((run_root / name).read_bytes()).hexdigest()
        for name in controller._RUN_STATE_FILENAMES
    }
    stable_run = 1024
    safety_margin = 10 * 1024**3
    tracker_minimum = stable_run + safety_margin
    maximum_workspace = 2048
    before_workspace = tracker_minimum + maximum_workspace
    capacity_contract = {
        "resource_capacity_policy_sha256": "a" * 64,
        "workspace_plan_sha256": file_record(frozen_paths["workspace_plan"])["sha256"],
        "workspace_plan_without_self_hash_sha256": "b" * 64,
        "projected_stable_run_bytes": stable_run,
        "fixed_safety_margin_bytes": safety_margin,
        "minimum_free_bytes_before_tracker": tracker_minimum,
        "maximum_workspace_bytes": maximum_workspace,
        "minimum_free_bytes_before_workspace_build": before_workspace,
        "planned_workspace_bytes": 1024,
        "required_free_bytes_before": before_workspace - 1,
        "required_free_bytes": before_workspace,
    }
    contract = {
        "project_root": str(root),
        "parent_authority_directory": str(parent),
        "controller": file_record(CONTROLLER_PATH),
        "failed_preflight_receipt": file_record(failed_preflight),
        "prior_failure_receipt": file_record(prior_failure),
        "retired_input_invalidation_receipt": file_record(invalidation),
        "frozen_input_bundle": {
            "directory": str(frozen_root),
            **{
                role: file_record(frozen_paths[role])
                for role in controller._REPLACEMENT_INPUT_FILENAMES
            },
        },
        "source": {
            "root_sha256": "5" * 64,
            "manifest_sha256": "6" * 64,
            "delta_sha256": "7" * 64,
            "allowlisted_change_count": 1,
        },
        "config": {
            "path": str(config_path),
            "file_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "semantic_sha256": "8" * 64,
        },
        "manifest": file_record(manifest_path),
        "run_state": {
            "root": str(run_root),
            "files": run_files,
            "sha256": controller._compact_sha256(run_files),
        },
        "technical_successor": {
            "authorization_sha256": "3" * 64,
            "intent_sha256": "4" * 64,
        },
        "replacement_state": {
            "state": "ready",
            "candidate_count": 0,
            "attempt_marker_absent": True,
            "success_marker_absent": True,
            "failure_marker_absent": True,
            "intended_authority_absent": True,
        },
        "capacity_contract": capacity_contract,
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }
    publication_authorization = {
        "schema_version": 1,
        "policy": controller._PUBLICATION_AUTHORIZATION_POLICY,
        "status": "authorized_for_one_attempt",
        "authorized_at_utc": amendment_timestamp,
        "automatic_retry_allowed": False,
        "max_attempt_count": 1,
        "authorized_attempt_id": "1" * 64,
        "publication": {
            "amendment_timestamp_utc": amendment_timestamp,
            "intended_authority_directory": str(destination),
            "parent_authority_directory": str(parent),
            "amendment_schema_version": 5,
            "amendment_purpose": controller._TECHNICAL_SUCCESSOR_PURPOSE,
            "chain_depth": 4,
        },
        "preflight": {
            "schema_version": 1,
            "policy": controller._LIVE_PREFLIGHT_POLICY,
            "status": "passed",
            "contract": contract,
            "preflight_fingerprint_sha256": controller._compact_sha256(contract),
            "capacity_observation": {
                "observed_at_utc": amendment_timestamp,
                "filesystem_path": str(run_root),
                "observed_free_bytes": before_workspace,
                "required_free_bytes": before_workspace,
                "passed": True,
            },
            "compute_observation": {
                "evidence": {
                    "schema_version": 1,
                    "phase": "guarded_before_data_loading",
                    "minimum_available_ram_bytes": 1,
                    "policy_sha256": capacity_contract["resource_capacity_policy_sha256"],
                    "observation": {"synthetic": True},
                    "observation_sha256": controller._compact_sha256({"synthetic": True}),
                    "checked_at_utc": amendment_timestamp,
                    "passed": True,
                    "outcome_values_read": False,
                    "prohibited_for_selection_tuning": True,
                    "adaptive_execution_changes_allowed": False,
                },
                "evidence_sha256": "",
            },
        },
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }
    publication_authorization["preflight"]["compute_observation"]["evidence_sha256"] = (
        controller._compact_sha256(
            publication_authorization["preflight"]["compute_observation"]["evidence"]
        )
    )
    publication_authorization_path = namespace.publication_authorization
    publication_authorization_path.write_bytes(
        controller._canonical_bytes(publication_authorization)
    )
    attempt = controller.attempt_record(
        attempt_id="1" * 64,
        destination=destination,
        parent=parent,
        controller_path=CONTROLLER_PATH,
        project_root=root,
        frozen_inputs=frozen_inputs,
        publication_authorization_receipt=publication_authorization_path,
        authorization_sha256="3" * 64,
        intent_sha256="4" * 64,
    )
    return SimpleNamespace(
        root=root,
        control_root=control_root,
        parent=parent,
        destination=destination,
        namespace=namespace,
        attempt=attempt,
        frozen_inputs=frozen_inputs,
        run_root=run_root,
        publication_authorization=publication_authorization_path,
        publication_authorization_payload=publication_authorization,
        retired_root=retired_root,
        invalidation=invalidation,
        invalidation_bytes=invalidation_bytes,
    )


def _discover(tree: SimpleNamespace) -> Callable[[str | Path], list[Path]]:
    def discover(_: str | Path) -> list[Path]:
        return [tree.destination] if tree.destination.is_dir() else []

    return discover


def _classify(controller: Any, tree: SimpleNamespace) -> Any:
    return controller.classify(
        tree.namespace,
        parent_authority_directory=tree.parent,
        candidate_discoverer=_discover(tree),
        committed_candidate_verifier=lambda _candidate, _success: None,
    )


def _install_bundle_lock_probe(
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    fail_assertions_from: int | None = None,
    fail_exit: bool = False,
) -> list[tuple[Path, ...]]:
    """Observe exclusion/rollback ordering without weakening the real lock."""

    real_lock = controller.ExclusiveBundlePublicationLock
    real_rollback = controller.rollback_owned_publications
    captured_paths: list[tuple[Path, ...]] = []
    active_probes: list[Any] = []

    class ProbeLock:
        def __init__(self, paths: Any, *, role: str) -> None:
            raw_paths = tuple(paths)
            captured_paths.append(tuple(Path(path).resolve() for path in raw_paths))
            self._inner = real_lock(raw_paths, role=role)
            self._held = False
            self._assertion_count = 0

        def __enter__(self) -> Any:
            self._inner.__enter__()
            self._held = True
            active_probes.append(self)
            events.append("lock_enter")
            return self

        def assert_owned(self) -> None:
            self._assertion_count += 1
            events.append("assert_owned")
            if fail_assertions_from is not None and self._assertion_count >= fail_assertions_from:
                raise RuntimeError("injected publication lock ownership loss")
            assert self._held
            self._inner.assert_owned()

        def __exit__(self, *args: Any) -> Any:
            events.append("lock_exit")
            try:
                result = self._inner.__exit__(*args)
            finally:
                self._held = False
                assert active_probes.pop() is self
            if fail_exit:
                raise RuntimeError("injected publication lock cleanup failure")
            return result

    def observed_rollback(publications: Any) -> None:
        events.append("rollback")
        assert active_probes and active_probes[-1]._held
        real_rollback(publications)

    monkeypatch.setattr(controller, "ExclusiveBundlePublicationLock", ProbeLock)
    monkeypatch.setattr(controller, "rollback_owned_publications", observed_rollback)
    return captured_paths


def _verify_request(controller: Any, tree: SimpleNamespace) -> Any:
    return controller.VerifyRequest(
        project_root=tree.root,
        successor_directory=tree.destination,
        parent_directory=tree.parent,
        artifact_root_sha256="6" * 64,
        manifest_sha256="7" * 64,
        authorization_sha256=tree.attempt["authorization_sha256"],
        intent_sha256=tree.attempt["intent_sha256"],
        nonce="8" * 64,
    ).checked()


def _verify_payload(
    controller: Any,
    request: Any,
    *,
    controller_pid: int,
    child_pid: int,
) -> dict[str, Any]:
    return {
        "status": "verified",
        "verification_schema_version": 2,
        "verification_kind": "resource_bounded_technical_successor_fresh_process",
        "process_boundary": {
            "controller_process_id": controller_pid,
            "verifier_process_id": child_pid,
            "verifier_parent_process_id": controller_pid,
            "distinct_processes": True,
            "direct_child_process": True,
            "verification_nonce": request.nonce,
        },
        "successor_authority": {
            "directory": str(request.successor_directory),
            "schema_version": 5,
            "purpose": "resource_bounded_confirmatory_technical_successor",
            "chain_depth": request.chain_depth,
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
            "flat_file_inventory_sha256": "9" * 64,
            "flat_file_hashes_verified": True,
        },
        "confirmatory_storage_policy_sha256": "a" * 64,
        "successor_candidate_count": 1,
        "checks": {field: True for field in controller._CHECK_FIELDS},
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }


class _FakePipe:
    def __init__(self, payload: bytes) -> None:
        self._stream = io.BytesIO(payload)
        self.read_requests: list[int] = []
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        self.read_requests.append(size)
        return self._stream.read(size)

    def close(self) -> None:
        self.closed = True
        self._stream.close()


class _RaisingPipe(_FakePipe):
    def read(self, size: int = -1) -> bytes:
        self.read_requests.append(size)
        raise OSError("injected bounded pipe read failure")


class _FakeProcess:
    def __init__(
        self,
        *,
        pid: int,
        stdout: bytes,
        stderr: bytes = b"",
        returncode: int = 0,
        timeout: bool = False,
    ) -> None:
        self.pid = pid
        self.stdout = _FakePipe(stdout)
        self.stderr = _FakePipe(stderr)
        self.returncode: int | None = None
        self.desired_returncode = returncode
        self.timeout = timeout
        self.terminated = False
        self.killed = False
        self.wait_timeouts: list[float | None] = []

    def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
        del timeout
        raise AssertionError("bounded verifier must not call communicate()")

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self.timeout and not self.killed:
            raise subprocess.TimeoutExpired("verifier", timeout)
        self.returncode = self.desired_returncode
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class _Factory:
    def __init__(self, process: _FakeProcess) -> None:
        self.process = process
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> _FakeProcess:
        self.calls.append((argv, kwargs))
        return self.process


def _success_verifier_result(controller: Any, tree: SimpleNamespace) -> Any:
    request = _verify_request(controller, tree)
    controller_pid = os.getpid()
    process_id = controller_pid + 1000
    payload = _verify_payload(
        controller,
        request,
        controller_pid=controller_pid,
        child_pid=process_id,
    )
    return controller.VerifyResult(
        request=request,
        argv=request.argv(controller_pid),
        process_id=process_id,
        payload=payload,
        payload_sha256="b" * 64,
    )


def _published_result(
    tree: SimpleNamespace,
    request: Any,
    *,
    destination: Path | None = None,
) -> PreregistrationAmendmentResult:
    authority = destination or tree.destination
    return PreregistrationAmendmentResult(
        amendment_directory=authority,
        parent_authority_directory=tree.parent,
        amendment_timestamp_utc="2026-07-28T00:00:00+00:00",
        chain_depth=request.chain_depth,
        amendment_evidence_path=authority / "amendment_evidence.json",
        amended_preregistration_path=authority / "PRE_REGISTRATION_FROZEN.md",
        amended_primary_config_path=authority / "primary_frozen.yaml",
        amended_confirmatory_config_path=authority / "confirmatory_frozen.yaml",
        source_tree_manifest_path=authority / "source_tree_manifest.json",
        sha256_manifest_path=authority / "sha256_manifest.json",
        immutable_marker_path=authority / ".immutable.json",
        artifact_root_sha256=request.artifact_root_sha256,
        sha256_manifest_sha256=request.manifest_sha256,
    )


def test_import_is_side_effect_free_and_namespace_is_new(
    tmp_path: Path,
    controller: Any,
) -> None:
    before = tuple(tmp_path.iterdir())
    _load_controller()

    assert tuple(tmp_path.iterdir()) == before
    assert "replacement_v1" in controller.ATTEMPT_FILENAME
    assert controller.ATTEMPT_FILENAME != "resource_authority_d_publication_attempt.json"
    assert controller._RETIRED_INPUT_DIRECTORY_NAME == "authority_d_replacement_inputs_v1"
    assert (
        controller._RETIRED_INPUT_INVALIDATION_FILENAME
        == "authority_d_replacement_inputs_v1.invalidation.json"
    )
    assert controller._REPLACEMENT_INPUT_DIRECTORY_NAME == "authority_d_replacement_inputs_v2"
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    assert "import prepare_resource_authority_d_once" not in source
    assert "authority_d_inputs_20260727Tfinal_source_v2" not in source


def test_cli_exposes_separate_freeze_preflight_authorize_publish_modes(
    controller: Any,
) -> None:
    parser = controller._parser()
    parent = str(Path.cwd())

    assert parser.parse_args(["--classify", "--parent-authority-dir", parent]).classify is True
    assert (
        parser.parse_args(["--invalidate-v1", "--parent-authority-dir", parent]).invalidate_v1
        is True
    )
    assert (
        parser.parse_args(["--preflight-only", "--parent-authority-dir", parent]).preflight_only
        is True
    )
    assert (
        parser.parse_args(["--publish-once", "--parent-authority-dir", parent]).publish_once is True
    )
    assert parser.parse_args(
        ["--freeze-inputs", "somewhere", "--parent-authority-dir", parent]
    ).freeze_inputs == Path("somewhere")
    assert (
        parser.parse_args(
            ["--authorize-publication", "--parent-authority-dir", parent]
        ).authorize_publication
        is True
    )
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_historical_failed_timestamp_acceptance_is_exact_and_narrow(
    controller: Any,
) -> None:
    parsed = controller._canonical_historical_failed_timestamp(
        "2026-07-27T17:30:54.689Z",
        "historical failed observation",
    )

    assert parsed.microsecond == 689_000
    assert parsed.tzinfo is UTC
    with pytest.raises(controller.ControlError, match="microseconds"):
        controller._canonical_timestamp(
            "2026-07-27T17:30:54.689Z",
            "new evidence",
        )
    assert (
        controller._canonical_timestamp(
            "2026-07-27T17:30:54.689000Z",
            "new evidence",
        ).microsecond
        == 689_000
    )


@pytest.mark.parametrize(
    "value",
    [
        "2026-07-27T17:30:54.689000Z",
        "2026-07-27T17:30:54Z",
        "2026-07-27T17:30:54.68Z",
        "2026-07-27 17:30:54.689Z",
        "2026-07-27T19:30:54.689+02:00",
        "2026-07-27T17:30:54.689001Z",
        None,
    ],
)
def test_historical_failed_timestamp_rejects_every_alternate_spelling(
    controller: Any,
    value: object,
) -> None:
    with pytest.raises(controller.ControlError, match="exact historical timestamp"):
        controller._canonical_historical_failed_timestamp(
            value,
            "historical failed observation",
        )


def test_exact_pinned_historical_json_accepts_crlf_without_rewriting(
    tmp_path: Path,
    controller: Any,
) -> None:
    path = tmp_path / "historical.log"
    encoded = b'{\r\n  "status": "stopped_without_write"\r\n}\r\n'
    path.write_bytes(encoded)

    payload, record = controller._live_pinned_json_record(
        path,
        "historical controller log",
        expected_size_bytes=len(encoded),
        expected_sha256=hashlib.sha256(encoded).hexdigest(),
    )

    assert payload == {"status": "stopped_without_write"}
    assert record["size_bytes"] == len(encoded)
    assert path.read_bytes() == encoded
    with pytest.raises(controller.ControlError, match="exact pin"):
        controller._live_pinned_json_record(
            path,
            "historical controller log",
            expected_size_bytes=len(encoded),
            expected_sha256="0" * 64,
        )


def test_truth_table_ready(control_tree: SimpleNamespace, controller: Any) -> None:
    result = _classify(controller, control_tree)

    assert result.state is controller.State.READY


def test_classifier_requires_exact_retired_v1_invalidation_pair(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    control_tree.invalidation.unlink()

    missing = _classify(controller, control_tree)

    assert missing.state is controller.State.STOP_AMBIGUOUS
    assert "must both exist" in missing.reason

    control_tree.invalidation.write_bytes(control_tree.invalidation_bytes + b" ")
    tampered = _classify(controller, control_tree)
    assert tampered.state is controller.State.STOP_AMBIGUOUS
    assert "synthetic retired invalidation changed" in tampered.reason


def test_retired_v1_can_never_be_selected_as_active_input(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    with pytest.raises(controller.ControlError, match="canonical singleton"):
        controller._replacement_bindings(
            {
                "control_root": control_tree.control_root,
                "failed_preflight": control_tree.frozen_inputs.failed_preflight_receipt,
                "prior_failure": control_tree.frozen_inputs.prior_failure_receipt,
                "retired_input_invalidation": control_tree.invalidation,
            },
            control_tree.retired_root,
        )


def test_ready_state_accepts_one_exact_authorization_receipt(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    receipt, digest = controller._read_publication_authorization(control_tree.namespace)

    assert receipt == control_tree.publication_authorization_payload
    assert digest == hashlib.sha256(control_tree.publication_authorization.read_bytes()).hexdigest()
    assert _classify(controller, control_tree).state is controller.State.READY
    assert control_tree.attempt["publication_authorization_receipt_sha256"] == digest
    assert (
        control_tree.attempt["preflight_fingerprint_sha256"]
        == receipt["preflight"]["preflight_fingerprint_sha256"]
    )
    assert control_tree.attempt["max_attempt_count"] == 1


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("max_attempt_count", True, "fixed policy"),
        ("max_attempt_count", 2, "fixed policy"),
        ("authorized_at_utc", "2026-07-28T12:00:00Z", "microseconds"),
    ],
)
def test_publication_authorization_rejects_type_policy_and_timestamp_tamper(
    control_tree: SimpleNamespace,
    controller: Any,
    field: str,
    value: Any,
    match: str,
) -> None:
    tampered = copy.deepcopy(control_tree.publication_authorization_payload)
    tampered[field] = value

    with pytest.raises(controller.ControlError, match=match):
        controller._canonical_publication_authorization(
            tampered,
            namespace=control_tree.namespace,
        )


def test_publication_authorization_rejects_capacity_one_byte_below(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    tampered = copy.deepcopy(control_tree.publication_authorization_payload)
    observation = tampered["preflight"]["capacity_observation"]
    observation["observed_free_bytes"] = observation["required_free_bytes"] - 1

    with pytest.raises(controller.ControlError, match="capacity observation"):
        controller._canonical_publication_authorization(
            tampered,
            namespace=control_tree.namespace,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", "failed_resource_preflight_other.json"),
        ("size_bytes", 1),
        ("sha256", "f" * 64),
    ],
)
def test_publication_authorization_requires_exact_historical_failed_receipt_pin(
    control_tree: SimpleNamespace,
    controller: Any,
    field: str,
    value: Any,
) -> None:
    tampered = copy.deepcopy(control_tree.publication_authorization_payload)
    record = tampered["preflight"]["contract"]["failed_preflight_receipt"]
    record[field] = str(control_tree.control_root / value) if field == "path" else value

    with pytest.raises(controller.ControlError, match="receipt paths"):
        controller._canonical_publication_authorization(
            tampered,
            namespace=control_tree.namespace,
        )


def test_classifier_rejects_noncanonical_or_changed_authorization_receipt(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    payload = control_tree.publication_authorization_payload
    control_tree.publication_authorization.write_bytes(json.dumps(payload, sort_keys=True).encode())

    assert _classify(controller, control_tree).state is controller.State.STOP_AMBIGUOUS


def test_classifier_rejects_extra_frozen_bundle_entry_bound_by_authorization(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    extra = control_tree.frozen_inputs.workspace_plan.parent / "unexpected.json"
    extra.write_text("{}\n", encoding="utf-8")

    assert _classify(controller, control_tree).state is controller.State.STOP_AMBIGUOUS


def test_truth_table_exact_attempt_failure_without_d(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    attempt_sha = controller.write_marker(
        control_tree.namespace.attempt,
        control_tree.attempt,
    )
    failure = controller._failure_record(
        control_tree.attempt,
        attempt_sha,
        RuntimeError("expected rollback"),
    )
    controller.write_marker(control_tree.namespace.failure, failure)

    result = _classify(controller, control_tree)

    assert result.state is controller.State.ROLLED_BACK_FAILURE


def test_truth_table_exact_attempt_success_and_single_d(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    attempt_sha = controller.write_marker(
        control_tree.namespace.attempt,
        control_tree.attempt,
    )
    control_tree.destination.mkdir()
    success = controller._success_record(
        control_tree.attempt,
        attempt_sha,
        _success_verifier_result(controller, control_tree),
    )
    controller.write_marker(control_tree.namespace.success, success)

    result = _classify(controller, control_tree)

    assert result.state is controller.State.COMMITTED


@pytest.mark.parametrize(
    ("write_attempt", "write_success", "write_failure", "create_d"),
    [
        (True, False, False, False),
        (False, True, False, True),
        (False, False, True, False),
        (False, False, True, True),
        (False, True, False, False),
        (False, True, True, False),
        (False, True, True, True),
        (True, True, False, False),
        (True, False, False, True),
        (True, False, True, True),
        (True, True, True, False),
        (True, True, True, True),
        (False, False, False, True),
    ],
)
def test_every_other_marker_d_combination_is_ambiguous(
    control_tree: SimpleNamespace,
    controller: Any,
    write_attempt: bool,
    write_success: bool,
    write_failure: bool,
    create_d: bool,
) -> None:
    attempt_sha = "c" * 64
    if write_attempt:
        attempt_sha = controller.write_marker(
            control_tree.namespace.attempt,
            control_tree.attempt,
        )
    if create_d:
        control_tree.destination.mkdir()
    if write_success:
        controller.write_marker(
            control_tree.namespace.success,
            controller._success_record(
                control_tree.attempt,
                attempt_sha,
                _success_verifier_result(controller, control_tree),
            ),
        )
    if write_failure:
        controller.write_marker(
            control_tree.namespace.failure,
            controller._failure_record(
                control_tree.attempt,
                attempt_sha,
                RuntimeError("failure"),
            ),
        )

    result = _classify(controller, control_tree)

    assert result.state is controller.State.STOP_AMBIGUOUS


def test_classifier_rejects_failed_discovery_and_namespace_extra(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    incomplete = controller.classify(
        control_tree.namespace,
        parent_authority_directory=control_tree.parent,
        candidate_discoverer=lambda _: (_ for _ in ()).throw(
            controller.ControlError("incomplete inventory")
        ),
    )
    extra = control_tree.control_root / f"{controller.MARKER_PREFIX}unexpected.json"
    extra.write_text("{}\n", encoding="utf-8")
    unexpected = _classify(controller, control_tree)

    assert incomplete.state is controller.State.STOP_AMBIGUOUS
    assert unexpected.state is controller.State.STOP_AMBIGUOUS


def test_default_discovery_cannot_omit_existing_successor_shape(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    control_tree.destination.mkdir()
    (control_tree.destination / "amendment_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "amendment_purpose": ("resource_bounded_confirmatory_technical_successor"),
            }
        ),
        encoding="utf-8",
    )
    (control_tree.destination / ".immutable.json").write_text(
        '{"status":"amended"}\n',
        encoding="utf-8",
    )

    result = controller.classify(
        control_tree.namespace,
        parent_authority_directory=control_tree.parent,
    )

    assert result.state is controller.State.STOP_AMBIGUOUS
    assert result.candidates == (control_tree.destination,)


def test_classifier_rejects_case_variant_namespace_entry(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    variant = control_tree.control_root / f"{controller.MARKER_PREFIX.upper()}unexpected.json"
    variant.write_text("{}\n", encoding="utf-8")

    result = _classify(controller, control_tree)

    assert result.state is controller.State.STOP_AMBIGUOUS


def test_fresh_verifier_uses_exact_shell_free_direct_child_command(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    request = _verify_request(controller, control_tree)
    child_pid = os.getpid() + 1000
    payload = _verify_payload(
        controller,
        request,
        controller_pid=os.getpid(),
        child_pid=child_pid,
    )
    process = _FakeProcess(
        pid=child_pid,
        stdout=json.dumps(payload, sort_keys=True).encode(),
    )
    factory = _Factory(process)

    result = controller.run_fresh_verifier(request, popen_factory=factory)

    argv, kwargs = factory.calls[0]
    assert argv[:7] == [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "histo_audit",
        "preregistration",
        "verify-resource-technical-successor",
    ]
    assert kwargs["shell"] is False
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.PIPE
    assert kwargs["text"] is False
    expected_spawn_executable = controller._fresh_verifier_spawn_executable(
        request.python_executable
    )
    if expected_spawn_executable is None:
        assert "executable" not in kwargs
    else:
        assert kwargs["executable"] == expected_spawn_executable
    assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in kwargs["env"]
    assert result.process_id == child_pid
    assert result.payload["process_boundary"]["direct_child_process"] is True


@pytest.mark.skipif(os.name != "nt", reason="Windows venv launcher regression")
def test_windows_spawn_executable_is_real_direct_child_and_preserves_venv(
    controller: Any,
) -> None:
    spawn_executable = controller._fresh_verifier_spawn_executable(sys.executable)
    assert spawn_executable is not None
    probe = (
        "import histo_audit, json, os, sys; "
        "print(json.dumps({"
        "'pid': os.getpid(), "
        "'ppid': os.getppid(), "
        "'executable': sys.executable, "
        "'prefix': sys.prefix, "
        "'base_prefix': sys.base_prefix, "
        "'histo_audit_imported': histo_audit.__name__ == 'histo_audit'"
        "}, sort_keys=True))"
    )
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", probe],
        executable=spawn_executable,
        cwd=str(Path.cwd()),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        text=False,
        close_fds=True,
        env=os.environ.copy(),
    )
    stdout, stderr = process.communicate(timeout=30)
    payload = json.loads(stdout)

    assert process.returncode == 0
    assert stderr == b""
    assert payload["pid"] == process.pid
    assert payload["ppid"] == os.getpid()
    assert payload["histo_audit_imported"] is True
    assert os.path.normcase(str(Path(payload["executable"]).resolve())) == os.path.normcase(
        str(Path(sys.executable).resolve())
    )
    assert os.path.normcase(str(Path(payload["prefix"]).resolve())) == os.path.normcase(
        str(Path(sys.prefix).resolve())
    )
    assert os.path.normcase(str(Path(payload["base_prefix"]).resolve())) == os.path.normcase(
        str(Path(sys.base_prefix).resolve())
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows base executable validation")
@pytest.mark.parametrize(
    "untrusted_value",
    [None, 7, "", "relative-python.exe"],
)
def test_windows_spawn_executable_rejects_invalid_scalar_before_popen(
    control_tree: SimpleNamespace,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
    untrusted_value: object,
) -> None:
    request = _verify_request(controller, control_tree)
    factory = _Factory(
        _FakeProcess(
            pid=os.getpid() + 1000,
            stdout=b"{}",
        )
    )
    monkeypatch.setattr(controller.sys, "_base_executable", untrusted_value)

    with pytest.raises(controller.FreshVerifierError):
        controller.run_fresh_verifier(request, popen_factory=factory)

    assert factory.calls == []


@pytest.mark.skipif(os.name != "nt", reason="Windows base executable validation")
def test_windows_spawn_executable_fails_closed_when_base_is_untrusted(
    control_tree: SimpleNamespace,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _verify_request(controller, control_tree)
    factory = _Factory(
        _FakeProcess(
            pid=os.getpid() + 1000,
            stdout=b"{}",
        )
    )
    monkeypatch.setattr(
        controller.sys,
        "_base_executable",
        str(control_tree.root / "missing-python.exe"),
    )

    with pytest.raises(
        controller.FreshVerifierError,
        match="cannot be trusted",
    ):
        controller.run_fresh_verifier(request, popen_factory=factory)

    assert factory.calls == []


@pytest.mark.skipif(os.name != "nt", reason="Windows base executable validation")
@pytest.mark.parametrize("candidate_kind", ["directory", "wrong_parent"])
def test_windows_spawn_executable_rejects_noncanonical_existing_path(
    control_tree: SimpleNamespace,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
    candidate_kind: str,
) -> None:
    request = _verify_request(controller, control_tree)
    candidate = control_tree.root
    if candidate_kind == "wrong_parent":
        candidate = control_tree.root / Path(sys.executable).name
        candidate.write_bytes(b"not-an-interpreter")
    factory = _Factory(
        _FakeProcess(
            pid=os.getpid() + 1000,
            stdout=b"{}",
        )
    )
    monkeypatch.setattr(controller.sys, "_base_executable", str(candidate))

    with pytest.raises(controller.FreshVerifierError):
        controller.run_fresh_verifier(request, popen_factory=factory)

    assert factory.calls == []


@pytest.mark.parametrize(
    "mutation",
    ["stderr", "returncode", "child_pid", "extra_field", "failed_check"],
)
def test_fresh_verifier_rejects_process_or_schema_mismatch(
    control_tree: SimpleNamespace,
    controller: Any,
    mutation: str,
) -> None:
    request = _verify_request(controller, control_tree)
    child_pid = os.getpid() + 1000
    payload = _verify_payload(
        controller,
        request,
        controller_pid=os.getpid(),
        child_pid=child_pid,
    )
    stderr = b""
    returncode = 0
    process_pid = child_pid
    if mutation == "stderr":
        stderr = b"unexpected"
    elif mutation == "returncode":
        returncode = 1
    elif mutation == "child_pid":
        process_pid += 1
    elif mutation == "extra_field":
        payload["unexpected"] = True
    else:
        payload["checks"]["generic_chain_integrity"] = False
    factory = _Factory(
        _FakeProcess(
            pid=process_pid,
            stdout=json.dumps(payload).encode(),
            stderr=stderr,
            returncode=returncode,
        )
    )

    with pytest.raises((controller.FreshVerifierError, controller.ControlError)):
        controller.run_fresh_verifier(request, popen_factory=factory)


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("process_boundary", "controller_process_id", float(os.getpid())),
        ("process_boundary", "distinct_processes", 1),
        ("successor_authority", "schema_version", 5.0),
        ("successor_authority", "chain_depth", 4.0),
        ("superseded_authority", "schema_version", 4.0),
        ("superseded_authority", "historically_verified", 1),
    ],
)
def test_fresh_verifier_rejects_json_type_aliases(
    control_tree: SimpleNamespace,
    controller: Any,
    section: str,
    field: str,
    replacement: Any,
) -> None:
    request = _verify_request(controller, control_tree)
    child_pid = os.getpid() + 1000
    payload = _verify_payload(
        controller,
        request,
        controller_pid=os.getpid(),
        child_pid=child_pid,
    )
    payload[section][field] = replacement
    process = _FakeProcess(
        pid=child_pid,
        stdout=json.dumps(payload).encode(),
    )

    with pytest.raises(controller.FreshVerifierError):
        controller.run_fresh_verifier(
            request,
            popen_factory=_Factory(process),
        )


def test_fresh_verifier_timeout_kills_child(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    request = _verify_request(controller, control_tree)
    process = _FakeProcess(pid=os.getpid() + 1000, stdout=b"{}", timeout=True)

    with pytest.raises(controller.FreshVerifierError, match="timed out"):
        controller.run_fresh_verifier(
            request,
            timeout_seconds=0.01,
            popen_factory=_Factory(process),
        )

    assert process.terminated is True
    assert process.killed is True
    assert process.returncode == 0
    assert process.stdout.closed is True
    assert process.stderr.closed is True
    assert process.wait_timeouts
    assert all(timeout is not None for timeout in process.wait_timeouts)


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
def test_fresh_verifier_hard_caps_each_pipe_and_reaps_child(
    control_tree: SimpleNamespace,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
    stream_name: str,
) -> None:
    request = _verify_request(controller, control_tree)
    monkeypatch.setattr(controller, "_MAX_BYTES", 32)
    monkeypatch.setattr(controller, "_MAX_STDERR_BYTES", 32)
    process = _FakeProcess(
        pid=os.getpid() + 1000,
        stdout=b"x" * 33 if stream_name == "stdout" else b"",
        stderr=b"x" * 33 if stream_name == "stderr" else b"",
        timeout=True,
    )
    before = tuple(control_tree.control_root.iterdir())

    with pytest.raises(controller.FreshVerifierError, match="bounded byte limit"):
        controller.run_fresh_verifier(
            request,
            timeout_seconds=1.0,
            popen_factory=_Factory(process),
        )

    selected = getattr(process, stream_name)
    assert selected.read_requests
    assert max(selected.read_requests) <= 33
    assert process.terminated is True
    assert process.killed is True
    assert process.returncode == 0
    assert process.stdout.closed is True
    assert process.stderr.closed is True
    assert tuple(control_tree.control_root.iterdir()) == before
    assert not control_tree.destination.exists()


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
def test_fresh_verifier_pipe_error_still_reaps_and_closes(
    control_tree: SimpleNamespace,
    controller: Any,
    stream_name: str,
) -> None:
    request = _verify_request(controller, control_tree)
    process = _FakeProcess(
        pid=os.getpid() + 1000,
        stdout=b"",
        stderr=b"",
        timeout=True,
    )
    setattr(process, stream_name, _RaisingPipe(b""))

    with pytest.raises(controller.FreshVerifierError, match="pipe read failed"):
        controller.run_fresh_verifier(
            request,
            timeout_seconds=1.0,
            popen_factory=_Factory(process),
        )

    assert process.terminated is True
    assert process.killed is True
    assert process.returncode == 0
    assert process.stdout.closed is True
    assert process.stderr.closed is True


def test_fresh_verifier_invalid_pid_is_cleaned_up(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    request = _verify_request(controller, control_tree)
    process = _FakeProcess(pid=0, stdout=b"", timeout=True)

    with pytest.raises(controller.FreshVerifierError, match="positive child PID"):
        controller.run_fresh_verifier(
            request,
            timeout_seconds=1.0,
            popen_factory=_Factory(process),
        )

    assert process.terminated is True
    assert process.killed is True
    assert process.returncode == 0


def _dynamic_factory(controller: Any, request: Any, child_pid: int) -> Callable[..., Any]:
    def factory(argv: list[str], **_: Any) -> _FakeProcess:
        controller_pid = int(argv[argv.index("--expected-controller-pid") + 1])
        payload = _verify_payload(
            controller,
            request,
            controller_pid=controller_pid,
            child_pid=child_pid,
        )
        return _FakeProcess(pid=child_pid, stdout=json.dumps(payload).encode())

    return factory


def _transaction_verifier(controller: Any, tree: SimpleNamespace, request: Any) -> Any:
    return controller.TransactionVerifier(
        project_root=tree.root,
        parent=tree.parent,
        destination=tree.destination,
        authorization_sha256=request.authorization_sha256,
        intent_sha256=request.intent_sha256,
        nonce_factory=lambda: request.nonce,
        popen_factory=_dynamic_factory(controller, request, os.getpid() + 1000),
    )


def test_execute_once_rejects_mismatched_or_preused_verifier_before_attempt(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    request = _verify_request(controller, control_tree)
    mismatched = controller.TransactionVerifier(
        project_root=control_tree.root,
        parent=control_tree.parent,
        destination=control_tree.destination,
        authorization_sha256="f" * 64,
        intent_sha256=request.intent_sha256,
    )

    with pytest.raises(controller.AmbiguousStateError, match="bindings"):
        controller.execute_once(
            namespace=control_tree.namespace,
            attempt=control_tree.attempt,
            transaction=lambda _: pytest.fail("transaction must not run"),
            verifier=mismatched,
            candidate_discoverer=_discover(control_tree),
        )
    assert not control_tree.namespace.attempt.exists()

    preused = controller.TransactionVerifier(
        project_root=control_tree.root,
        parent=control_tree.parent,
        destination=control_tree.destination,
        authorization_sha256=request.authorization_sha256,
        intent_sha256=request.intent_sha256,
    )
    preused.invoked = True
    with pytest.raises(controller.AmbiguousStateError, match="fresh"):
        controller.execute_once(
            namespace=control_tree.namespace,
            attempt=control_tree.attempt,
            transaction=lambda _: pytest.fail("transaction must not run"),
            verifier=preused,
            candidate_discoverer=_discover(control_tree),
        )
    assert not control_tree.namespace.attempt.exists()


def test_execute_once_rejects_cross_project_namespace_before_attempt(
    tmp_path: Path,
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    other_root = tmp_path / "other-project"
    other_control = other_root / "artifacts" / "resource_control"
    other_control.mkdir(parents=True)
    other_namespace = controller.Namespace(other_control)
    request = _verify_request(controller, control_tree)
    verifier = _transaction_verifier(controller, control_tree, request)

    with pytest.raises(controller.ControlError, match="bound project"):
        controller.execute_once(
            namespace=other_namespace,
            attempt=control_tree.attempt,
            transaction=lambda _: pytest.fail("transaction must not run"),
            verifier=verifier,
            candidate_discoverer=_discover(control_tree),
        )

    assert tuple(other_control.iterdir()) == ()
    assert not control_tree.namespace.attempt.exists()


@pytest.mark.parametrize(
    "tamper_kind",
    ["frozen_input", "retired_input_invalidation", "run_state"],
)
def test_execute_once_rechecks_all_live_attempt_inputs_before_claim(
    control_tree: SimpleNamespace,
    controller: Any,
    tamper_kind: str,
) -> None:
    request = _verify_request(controller, control_tree)
    verifier = _transaction_verifier(controller, control_tree, request)
    if tamper_kind == "frozen_input":
        with control_tree.frozen_inputs.workspace_plan.open("ab") as stream:
            stream.write(b"tampered")
    elif tamper_kind == "retired_input_invalidation":
        with control_tree.invalidation.open("ab") as stream:
            stream.write(b"tampered")
    else:
        with (control_tree.run_root / controller._RUN_STATE_FILENAMES[0]).open("ab") as stream:
            stream.write(b"tampered")

    with pytest.raises(
        controller.ControlError,
        match=r"input|run-state|frozen-source|initial state",
    ):
        controller.execute_once(
            namespace=control_tree.namespace,
            attempt=control_tree.attempt,
            transaction=lambda _: pytest.fail("transaction must not run"),
            verifier=verifier,
            candidate_discoverer=_discover(control_tree),
        )

    assert not control_tree.namespace.attempt.exists()
    assert not control_tree.namespace.success.exists()
    assert not control_tree.namespace.failure.exists()


@pytest.mark.parametrize("return_value", [None, "not-none"])
def test_execute_once_runs_preclaim_check_before_attempt(
    control_tree: SimpleNamespace,
    controller: Any,
    return_value: Any,
) -> None:
    request = _verify_request(controller, control_tree)
    verifier = _transaction_verifier(controller, control_tree, request)
    calls = 0
    transaction_calls = 0

    def preclaim() -> Any:
        nonlocal calls
        calls += 1
        if return_value is None:
            raise controller.ControlError("injected resource-gate failure")
        return return_value

    def transaction(_: Any) -> Any:
        nonlocal transaction_calls
        transaction_calls += 1
        raise AssertionError("transaction must not run")

    match = "resource-gate failure" if return_value is None else "exactly None"
    with pytest.raises(controller.ControlError, match=match):
        controller.execute_once(
            namespace=control_tree.namespace,
            attempt=control_tree.attempt,
            transaction=transaction,
            verifier=verifier,
            candidate_discoverer=_discover(control_tree),
            committed_candidate_verifier=lambda _candidate, _success: None,
            preclaim_check=preclaim,
        )

    assert calls == 1
    assert transaction_calls == 0
    assert not control_tree.namespace.attempt.exists()


def test_execute_once_commits_success_last_from_in_memory_verification(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    request = _verify_request(controller, control_tree)
    verifier = controller.TransactionVerifier(
        project_root=control_tree.root,
        parent=control_tree.parent,
        destination=control_tree.destination,
        authorization_sha256=request.authorization_sha256,
        intent_sha256=request.intent_sha256,
        nonce_factory=lambda: request.nonce,
        popen_factory=_dynamic_factory(controller, request, os.getpid() + 1000),
    )
    published = _published_result(control_tree, request)

    def transaction(callback: Callable[[Any], None]) -> Any:
        control_tree.destination.mkdir()
        callback(published)
        return published

    result = controller.execute_once(
        namespace=control_tree.namespace,
        attempt=control_tree.attempt,
        transaction=transaction,
        verifier=verifier,
        candidate_discoverer=lambda _: (
            [control_tree.destination] if control_tree.destination.is_dir() else []
        ),
        committed_candidate_verifier=lambda _candidate, _success: None,
    )

    assert result.state is controller.State.COMMITTED
    assert control_tree.namespace.success.is_file()
    assert not control_tree.namespace.failure.exists()
    classified = _classify(controller, control_tree)
    assert classified.state is controller.State.COMMITTED
    with (control_tree.run_root / controller._RUN_STATE_FILENAMES[0]).open("ab") as stream:
        stream.write(b"later-confirmatory-run-state")
    assert _classify(controller, control_tree).state is controller.State.COMMITTED
    marker_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (control_tree.namespace.attempt, control_tree.namespace.success)
    }
    second_verifier = controller.TransactionVerifier(
        project_root=control_tree.root,
        parent=control_tree.parent,
        destination=control_tree.destination,
        authorization_sha256=request.authorization_sha256,
        intent_sha256=request.intent_sha256,
    )
    with pytest.raises(controller.AmbiguousStateError, match="not READY"):
        controller.execute_once(
            namespace=control_tree.namespace,
            attempt=control_tree.attempt,
            transaction=lambda _: pytest.fail("terminal state must not retry"),
            verifier=second_verifier,
            candidate_discoverer=_discover(control_tree),
            committed_candidate_verifier=lambda _candidate, _success: None,
        )
    assert marker_hashes == {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in marker_hashes
    }


@pytest.mark.parametrize("return_mode", ["none", "mismatched", "deleted"])
def test_execute_once_refuses_untrusted_post_callback_result(
    control_tree: SimpleNamespace,
    controller: Any,
    return_mode: str,
) -> None:
    request = _verify_request(controller, control_tree)
    verifier = _transaction_verifier(controller, control_tree, request)
    published = _published_result(control_tree, request)

    def transaction(callback: Callable[[Any], None]) -> Any:
        control_tree.destination.mkdir()
        callback(published)
        if return_mode == "none":
            return None
        if return_mode == "mismatched":
            return _published_result(control_tree, request)
        control_tree.destination.rmdir()
        return published

    with pytest.raises(controller.AmbiguousStateError):
        controller.execute_once(
            namespace=control_tree.namespace,
            attempt=control_tree.attempt,
            transaction=transaction,
            verifier=verifier,
            candidate_discoverer=_discover(control_tree),
            committed_candidate_verifier=lambda _candidate, _success: None,
        )

    assert control_tree.namespace.attempt.is_file()
    assert not control_tree.namespace.success.exists()
    assert not control_tree.namespace.failure.exists()
    if return_mode == "deleted":
        assert not control_tree.destination.exists()


def test_execute_once_records_failure_after_post_callback_creator_rollback(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    request = _verify_request(controller, control_tree)
    verifier = _transaction_verifier(controller, control_tree, request)
    published = _published_result(control_tree, request)

    def transaction(callback: Callable[[Any], None]) -> Any:
        control_tree.destination.mkdir()
        try:
            callback(published)
            raise RuntimeError("injected creator finalization failure")
        finally:
            control_tree.destination.rmdir()

    result = controller.execute_once(
        namespace=control_tree.namespace,
        attempt=control_tree.attempt,
        transaction=transaction,
        verifier=verifier,
        candidate_discoverer=_discover(control_tree),
    )

    assert result.state is controller.State.ROLLED_BACK_FAILURE
    assert control_tree.namespace.failure.is_file()
    assert not control_tree.namespace.success.exists()
    assert not control_tree.destination.exists()


@pytest.mark.parametrize(
    "tamper_kind",
    ["frozen_input", "retired_input_invalidation"],
)
def test_execute_once_stops_if_bound_input_changes_after_callback(
    control_tree: SimpleNamespace,
    controller: Any,
    tamper_kind: str,
) -> None:
    request = _verify_request(controller, control_tree)
    verifier = _transaction_verifier(controller, control_tree, request)
    published = _published_result(control_tree, request)

    def transaction(callback: Callable[[Any], None]) -> Any:
        control_tree.destination.mkdir()
        callback(published)
        tamper_path = (
            control_tree.frozen_inputs.workspace_plan
            if tamper_kind == "frozen_input"
            else control_tree.invalidation
        )
        with tamper_path.open("ab") as stream:
            stream.write(b"post-callback-tamper")
        return published

    with pytest.raises(controller.ControlError, match=r"input|frozen-source"):
        controller.execute_once(
            namespace=control_tree.namespace,
            attempt=control_tree.attempt,
            transaction=transaction,
            verifier=verifier,
            candidate_discoverer=_discover(control_tree),
            committed_candidate_verifier=lambda _candidate, _success: None,
        )

    assert control_tree.namespace.attempt.is_file()
    assert not control_tree.namespace.success.exists()
    assert not control_tree.namespace.failure.exists()


def test_execute_once_protocol_lock_includes_active_v2_root(
    control_tree: SimpleNamespace,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _verify_request(controller, control_tree)
    mismatched = controller.TransactionVerifier(
        project_root=control_tree.root,
        parent=control_tree.parent,
        destination=control_tree.destination,
        authorization_sha256="f" * 64,
        intent_sha256=request.intent_sha256,
    )
    events: list[str] = []
    captured_paths = _install_bundle_lock_probe(controller, monkeypatch, events)

    with pytest.raises(controller.AmbiguousStateError, match="bindings"):
        controller.execute_once(
            namespace=control_tree.namespace,
            attempt=control_tree.attempt,
            transaction=lambda _: pytest.fail("transaction must not run"),
            verifier=mismatched,
            candidate_discoverer=_discover(control_tree),
        )

    assert len(captured_paths) == 1
    assert set(captured_paths[0]) == {
        control_tree.frozen_inputs.frozen_source_receipt.parent.resolve(),
        control_tree.namespace.publication_authorization.resolve(),
        control_tree.namespace.attempt.resolve(),
        control_tree.namespace.success.resolve(),
        control_tree.namespace.failure.resolve(),
    }
    assert events == ["lock_enter", "lock_exit"]
    assert not control_tree.namespace.attempt.exists()


def test_protocol_lock_cleanup_precedes_transaction_and_terminal_marker(
    control_tree: SimpleNamespace,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _verify_request(controller, control_tree)
    verifier = _transaction_verifier(controller, control_tree, request)
    original_exit = controller.ExclusiveBundlePublicationLock.__exit__
    transaction_calls = 0

    def broken_exit(self: Any, *args: Any) -> Any:
        original_exit(self, *args)
        raise RuntimeError("injected protocol lock cleanup failure")

    def transaction(_: Callable[[Any], None]) -> Any:
        nonlocal transaction_calls
        transaction_calls += 1
        pytest.fail("transaction must not start after lock cleanup failure")

    monkeypatch.setattr(
        controller.ExclusiveBundlePublicationLock,
        "__exit__",
        broken_exit,
    )

    with pytest.raises(RuntimeError, match="lock cleanup failure"):
        controller.execute_once(
            namespace=control_tree.namespace,
            attempt=control_tree.attempt,
            transaction=transaction,
            verifier=verifier,
            candidate_discoverer=_discover(control_tree),
        )

    assert transaction_calls == 0
    assert control_tree.namespace.attempt.is_file()
    assert not control_tree.namespace.success.exists()
    assert not control_tree.namespace.failure.exists()
    assert not control_tree.destination.exists()


def test_recursive_execute_is_terminal_failure_without_second_transaction(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    request = _verify_request(controller, control_tree)
    outer_verifier = _transaction_verifier(controller, control_tree, request)
    inner_verifier = _transaction_verifier(controller, control_tree, request)
    inner_transaction_calls = 0

    def inner_transaction(_: Callable[[Any], None]) -> Any:
        nonlocal inner_transaction_calls
        inner_transaction_calls += 1
        pytest.fail("recursive transaction must not run")

    def outer_transaction(_: Callable[[Any], None]) -> Any:
        controller.execute_once(
            namespace=control_tree.namespace,
            attempt=control_tree.attempt,
            transaction=inner_transaction,
            verifier=inner_verifier,
            candidate_discoverer=_discover(control_tree),
        )

    result = controller.execute_once(
        namespace=control_tree.namespace,
        attempt=control_tree.attempt,
        transaction=outer_transaction,
        verifier=outer_verifier,
        candidate_discoverer=_discover(control_tree),
    )

    assert result.state is controller.State.ROLLED_BACK_FAILURE
    assert inner_transaction_calls == 0
    assert control_tree.namespace.attempt.is_file()
    assert control_tree.namespace.failure.is_file()
    assert not control_tree.namespace.success.exists()
    assert not control_tree.destination.exists()


def test_execute_once_records_failure_only_after_exact_rollback_absence(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    request = _verify_request(controller, control_tree)
    verifier = controller.TransactionVerifier(
        project_root=control_tree.root,
        parent=control_tree.parent,
        destination=control_tree.destination,
        authorization_sha256=request.authorization_sha256,
        intent_sha256=request.intent_sha256,
    )

    result = controller.execute_once(
        namespace=control_tree.namespace,
        attempt=control_tree.attempt,
        transaction=lambda _: (_ for _ in ()).throw(RuntimeError("creator failed")),
        verifier=verifier,
        candidate_discoverer=lambda _: [],
    )

    assert result.state is controller.State.ROLLED_BACK_FAILURE
    assert control_tree.namespace.failure.is_file()
    assert not control_tree.namespace.success.exists()
    marker_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (control_tree.namespace.attempt, control_tree.namespace.failure)
    }
    second_verifier = controller.TransactionVerifier(
        project_root=control_tree.root,
        parent=control_tree.parent,
        destination=control_tree.destination,
        authorization_sha256=request.authorization_sha256,
        intent_sha256=request.intent_sha256,
    )
    with pytest.raises(controller.AmbiguousStateError, match="not READY"):
        controller.execute_once(
            namespace=control_tree.namespace,
            attempt=control_tree.attempt,
            transaction=lambda _: pytest.fail("terminal state must not retry"),
            verifier=second_verifier,
            candidate_discoverer=_discover(control_tree),
        )
    assert marker_hashes == {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in marker_hashes
    }


@pytest.mark.parametrize("failure_kind", ["fresh_verifier", "wrong_destination"])
def test_transaction_callback_failure_rolls_back_before_failure_marker(
    control_tree: SimpleNamespace,
    controller: Any,
    failure_kind: str,
) -> None:
    request = _verify_request(controller, control_tree)
    if failure_kind == "fresh_verifier":
        child_pid = os.getpid() + 1000
        payload = _verify_payload(
            controller,
            request,
            controller_pid=os.getpid(),
            child_pid=child_pid,
        )
        popen_factory: Callable[..., Any] = _Factory(
            _FakeProcess(
                pid=child_pid,
                stdout=json.dumps(payload).encode(),
                stderr=b"unexpected",
            )
        )
        published_destination = control_tree.destination
    else:
        popen_factory = _dynamic_factory(
            controller,
            request,
            os.getpid() + 1000,
        )
        published_destination = control_tree.destination.with_name("wrong_d")
    verifier = controller.TransactionVerifier(
        project_root=control_tree.root,
        parent=control_tree.parent,
        destination=control_tree.destination,
        authorization_sha256=request.authorization_sha256,
        intent_sha256=request.intent_sha256,
        nonce_factory=lambda: request.nonce,
        popen_factory=popen_factory,
    )
    published = _published_result(
        control_tree,
        request,
        destination=published_destination,
    )

    def transaction(callback: Callable[[Any], None]) -> Any:
        published_destination.mkdir()
        try:
            callback(published)
        finally:
            published_destination.rmdir()

    result = controller.execute_once(
        namespace=control_tree.namespace,
        attempt=control_tree.attempt,
        transaction=transaction,
        verifier=verifier,
        candidate_discoverer=_discover(control_tree),
    )

    assert result.state is controller.State.ROLLED_BACK_FAILURE
    assert control_tree.namespace.attempt.is_file()
    assert control_tree.namespace.failure.is_file()
    assert not control_tree.namespace.success.exists()
    assert not published_destination.exists()


def test_execute_once_stops_ambiguous_if_failed_creator_left_d(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    request = _verify_request(controller, control_tree)
    verifier = controller.TransactionVerifier(
        project_root=control_tree.root,
        parent=control_tree.parent,
        destination=control_tree.destination,
        authorization_sha256=request.authorization_sha256,
        intent_sha256=request.intent_sha256,
    )

    def failed_transaction(_: Callable[[Any], None]) -> Any:
        control_tree.destination.mkdir()
        raise RuntimeError("creator failed after D appeared")

    with pytest.raises(controller.AmbiguousStateError, match="D is present"):
        controller.execute_once(
            namespace=control_tree.namespace,
            attempt=control_tree.attempt,
            transaction=failed_transaction,
            verifier=verifier,
            candidate_discoverer=lambda _: (
                [control_tree.destination] if control_tree.destination.is_dir() else []
            ),
        )

    assert control_tree.namespace.attempt.is_file()
    assert not control_tree.namespace.failure.exists()
    assert not control_tree.namespace.success.exists()


def test_callback_is_one_use(control_tree: SimpleNamespace, controller: Any) -> None:
    request = _verify_request(controller, control_tree)
    control_tree.destination.mkdir()
    verifier = controller.TransactionVerifier(
        project_root=control_tree.root,
        parent=control_tree.parent,
        destination=control_tree.destination,
        authorization_sha256=request.authorization_sha256,
        intent_sha256=request.intent_sha256,
        nonce_factory=lambda: request.nonce,
        popen_factory=_dynamic_factory(controller, request, os.getpid() + 1000),
    )
    published = _published_result(control_tree, request)

    assert verifier(published) is None
    with pytest.raises(controller.FreshVerifierError, match="twice"):
        verifier(published)


@pytest.mark.parametrize("fault_marker", ["attempt", "failure", "success"])
def test_partial_marker_write_is_terminal_ambiguous_without_opposite_marker(
    control_tree: SimpleNamespace,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
    fault_marker: str,
) -> None:
    request = _verify_request(controller, control_tree)
    verifier = controller.TransactionVerifier(
        project_root=control_tree.root,
        parent=control_tree.parent,
        destination=control_tree.destination,
        authorization_sha256=request.authorization_sha256,
        intent_sha256=request.intent_sha256,
        nonce_factory=lambda: request.nonce,
        popen_factory=_dynamic_factory(
            controller,
            request,
            os.getpid() + 1000,
        ),
    )
    published = _published_result(control_tree, request)
    real_publish = controller.publish_bytes_no_overwrite
    fault_path = getattr(control_tree.namespace, fault_marker)

    def publish(payload: bytes, path: Path) -> Any:
        if path != fault_path:
            return real_publish(payload, path)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, payload[: max(1, len(payload) // 2)])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        raise OSError(f"injected partial {fault_marker} marker")

    monkeypatch.setattr(controller, "publish_bytes_no_overwrite", publish)
    transaction_calls = 0

    def transaction(callback: Callable[[Any], None]) -> Any:
        nonlocal transaction_calls
        transaction_calls += 1
        if fault_marker == "failure":
            raise RuntimeError("creator failed")
        control_tree.destination.mkdir()
        callback(published)
        return published

    with pytest.raises(OSError, match="injected partial"):
        controller.execute_once(
            namespace=control_tree.namespace,
            attempt=control_tree.attempt,
            transaction=transaction,
            verifier=verifier,
            candidate_discoverer=_discover(control_tree),
            committed_candidate_verifier=lambda _candidate, _success: None,
        )

    assert fault_path.exists()
    assert not (control_tree.namespace.failure.exists() and control_tree.namespace.success.exists())
    assert _classify(controller, control_tree).state is controller.State.STOP_AMBIGUOUS
    if fault_marker == "attempt":
        assert transaction_calls == 0
    elif fault_marker == "failure":
        assert transaction_calls == 1
        assert not control_tree.namespace.success.exists()
    else:
        assert transaction_calls == 1
        assert not control_tree.namespace.failure.exists()


def test_publish_cli_requires_frozen_inputs_and_writes_nothing(
    control_tree: SimpleNamespace,
    controller: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    before = tuple(control_tree.control_root.iterdir())

    exit_code = controller.main(
        [
            "--publish-once",
            "--project-root",
            str(control_tree.root),
            "--parent-authority-dir",
            str(control_tree.parent),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["status"] == "stopped_without_write"
    assert tuple(control_tree.control_root.iterdir()) == before


def test_invalidate_v1_cli_dispatches_without_frozen_input_argument(
    control_tree: SimpleNamespace,
    controller: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    expected = {
        "schema_version": 1,
        "status": "retired_v1_preserved_invalid_nonpublishable",
        "publication_performed": False,
    }

    def publish(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(
        controller,
        "publish_retired_input_invalidation_once",
        publish,
    )

    exit_code = controller.main(
        [
            "--invalidate-v1",
            "--project-root",
            str(control_tree.root),
            "--parent-authority-dir",
            str(control_tree.parent),
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert len(calls) == 1
    assert calls[0]["namespace"] == control_tree.namespace
    assert calls[0]["parent_authority_directory"] == control_tree.parent


def test_publish_cli_runtime_error_after_attempt_is_fail_closed(
    control_tree: SimpleNamespace,
    controller: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_after_attempt(**kwargs: Any) -> Any:
        controller.write_marker(kwargs["namespace"].attempt, control_tree.attempt)
        raise RuntimeError("injected failure after A")

    monkeypatch.setattr(
        controller,
        "publish_replacement_authority_once",
        fail_after_attempt,
    )

    exit_code = controller.main(
        [
            "--publish-once",
            "--project-root",
            str(control_tree.root),
            "--parent-authority-dir",
            str(control_tree.parent),
            "--frozen-input-dir",
            str(control_tree.frozen_inputs.frozen_source_receipt.parent),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert output["status"] == "stopped_after_control_write"
    assert output["replacement_state"] == "stop_ambiguous"
    assert output["publication_performed"] is None
    assert control_tree.namespace.attempt.is_file()


def test_publish_cli_runtime_error_with_attempt_and_d_is_ambiguous(
    control_tree: SimpleNamespace,
    controller: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_after_attempt_and_d(**kwargs: Any) -> Any:
        controller.write_marker(kwargs["namespace"].attempt, control_tree.attempt)
        control_tree.destination.mkdir()
        raise RuntimeError("injected failure after A+D")

    monkeypatch.setattr(
        controller,
        "publish_replacement_authority_once",
        fail_after_attempt_and_d,
    )

    exit_code = controller.main(
        [
            "--publish-once",
            "--project-root",
            str(control_tree.root),
            "--parent-authority-dir",
            str(control_tree.parent),
            "--frozen-input-dir",
            str(control_tree.frozen_inputs.frozen_source_receipt.parent),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert output["status"] == "stopped_after_control_write"
    assert output["replacement_state"] == "stop_ambiguous"
    assert output["publication_performed"] is None
    assert control_tree.namespace.attempt.is_file()
    assert control_tree.destination.is_dir()


def test_freeze_cli_reports_retained_control_write_after_runtime_error(
    control_tree: SimpleNamespace,
    controller: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_tree.frozen_inputs.prior_failure_receipt.unlink()
    control_tree.publication_authorization.unlink()

    def fail_after_prior_receipt(**kwargs: Any) -> Any:
        kwargs["namespace"].control_root.joinpath(
            controller._PRIOR_FAILURE_RECEIPT_FILENAME
        ).write_bytes(b'{"status":"retained"}\n')
        raise RuntimeError("injected failure after retained receipt")

    monkeypatch.setattr(
        controller,
        "freeze_replacement_inputs_once",
        fail_after_prior_receipt,
    )

    exit_code = controller.main(
        [
            "--freeze-inputs",
            str(control_tree.control_root / controller._REPLACEMENT_INPUT_DIRECTORY_NAME),
            "--project-root",
            str(control_tree.root),
            "--parent-authority-dir",
            str(control_tree.parent),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert output["status"] == "stopped_after_control_write"
    assert output["replacement_state"] == "ready"
    assert output["publication_performed"] is False


def test_publish_cli_exception_after_committed_state_reports_publication(
    control_tree: SimpleNamespace,
    controller: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_publish(**_kwargs: Any) -> Any:
        raise RuntimeError("injected exception after commit")

    monkeypatch.setattr(
        controller,
        "publish_replacement_authority_once",
        fail_publish,
    )
    monkeypatch.setattr(
        controller,
        "classify",
        lambda *_args, **_kwargs: controller.Classification(
            controller.State.COMMITTED,
            "synthetic exact commit",
            (control_tree.destination,),
        ),
    )

    exit_code = controller.main(
        [
            "--publish-once",
            "--project-root",
            str(control_tree.root),
            "--parent-authority-dir",
            str(control_tree.parent),
            "--frozen-input-dir",
            str(control_tree.frozen_inputs.frozen_source_receipt.parent),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert output["status"] == "stopped_after_attempt"
    assert output["replacement_state"] == "committed"
    assert output["publication_performed"] is True


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("ready", False),
        ("rolled_back_failure", False),
        ("committed", True),
        ("stop_ambiguous", None),
    ],
)
def test_classification_json_reports_historical_publication_state(
    controller: Any,
    state: str,
    expected: bool | None,
) -> None:
    classification = controller.Classification(
        controller.State(state),
        "synthetic state",
        (),
    )

    assert classification.as_dict()["publication_performed"] is expected


def test_success_schema_tamper_is_stop_ambiguous(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    attempt_sha = controller.write_marker(
        control_tree.namespace.attempt,
        control_tree.attempt,
    )
    control_tree.destination.mkdir()
    success = controller._success_record(
        control_tree.attempt,
        attempt_sha,
        _success_verifier_result(controller, control_tree),
    )
    tampered = copy.deepcopy(success)
    tampered["schema_version"] = True
    controller.write_marker(control_tree.namespace.success, tampered)

    result = _classify(controller, control_tree)

    assert result.state is controller.State.STOP_AMBIGUOUS


def test_success_schema_rejects_non_string_verification_nonce(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    success = controller._success_record(
        control_tree.attempt,
        "a" * 64,
        _success_verifier_result(controller, control_tree),
    )
    success["verification_nonce"] = 123

    with pytest.raises(controller.ControlError, match="verification nonce"):
        controller._record(success, "success")


def test_committed_readback_rejects_chain_depth_mismatch_before_authorization(
    control_tree: SimpleNamespace,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    success = controller._success_record(
        control_tree.attempt,
        "a" * 64,
        _success_verifier_result(controller, control_tree),
    )
    verification = SimpleNamespace(
        valid=True,
        parent_authority_directory=control_tree.parent,
        artifact_root_sha256=success["artifact_root_sha256"],
        sha256_manifest_sha256=success["sha256_manifest_sha256"],
        chain_depth=success["chain_depth"] + 1,
    )
    monkeypatch.setattr(
        amendment,
        "verify_preregistration_amendment",
        lambda _: verification,
    )

    with pytest.raises(controller.ControlError, match="generic success pins"):
        controller._verify_committed_candidate(
            control_tree.destination,
            success,
            control_tree.attempt,
        )


def test_authorization_is_bound_to_every_frozen_attempt_input(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    authorization = {
        "failed_preflight": {
            "receipt_path": control_tree.attempt["failed_preflight_receipt_path"],
            "receipt_sha256": control_tree.attempt["failed_preflight_receipt_sha256"],
        },
        "prior_publication_failure": {
            "receipt_path": control_tree.attempt["prior_failure_receipt_path"],
            "receipt_sha256": control_tree.attempt["prior_failure_receipt_sha256"],
        },
        "execution_source_delta": {
            "allowlisted_change_kinds": {
                "src/histo_audit/example.py": "modified",
            },
            "resource_root_sha256": "6" * 64,
            "resource_manifest_sha256": "7" * 64,
            "delta_sha256": "8" * 64,
        },
        "resource_input_workspace_plan": {"plan": "test"},
        "cnn_provenance_correction": {"correction": "test"},
    }

    controller._require_authorization_input_bindings(
        authorization,
        control_tree.attempt,
    )

    tampered = copy.deepcopy(authorization)
    tampered["resource_input_workspace_plan"]["plan"] = "different"
    with pytest.raises(controller.ControlError, match="workspace plan"):
        controller._require_authorization_input_bindings(
            tampered,
            control_tree.attempt,
        )


def _synthetic_live_freeze_context(
    tmp_path: Path,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    root = tmp_path / "live-project"
    control_root = root / "artifacts" / "resource_control"
    control_root.mkdir(parents=True)
    failed = control_root / "failed_resource_preflight_synthetic.json"
    prior = control_root / controller._PRIOR_FAILURE_RECEIPT_FILENAME
    failed.write_bytes(b'{"status":"failed"}\n')
    prior.write_bytes(b'{"status":"verified"}\n')
    monkeypatch.setattr(controller, "_FAILED_PREFLIGHT_FILENAME", failed.name)
    monkeypatch.setattr(
        controller,
        "_FAILED_PREFLIGHT_SIZE_BYTES",
        failed.stat().st_size,
    )
    monkeypatch.setattr(
        controller,
        "_FAILED_PREFLIGHT_SHA256",
        hashlib.sha256(failed.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        controller,
        "_PRIOR_FAILURE_RECEIPT_SIZE_BYTES",
        prior.stat().st_size,
    )
    monkeypatch.setattr(
        controller,
        "_PRIOR_FAILURE_RECEIPT_SHA256",
        hashlib.sha256(prior.read_bytes()).hexdigest(),
    )
    namespace = controller.Namespace(control_root)
    retired_root = control_root / controller._RETIRED_INPUT_DIRECTORY_NAME
    retired_root.mkdir()
    for role, filename in controller._REPLACEMENT_INPUT_FILENAMES.items():
        (retired_root / filename).write_bytes(f"retired-{role}\n".encode())
    invalidation = control_root / controller._RETIRED_INPUT_INVALIDATION_FILENAME
    invalidation_payload = {"status": "synthetic-invalidated-v1"}
    invalidation.write_bytes(controller._canonical_bytes(invalidation_payload))
    invalidation_record = controller._live_file_record(
        invalidation,
        "retired input invalidation receipt",
    )

    source_allowlist = {
        "records": [
            {
                "path": "src/histo_audit/example.py",
                "change_kind": "modified",
            }
        ]
    }
    workspace_plan = {"schema_version": 1, "planned_workspace_bytes": 1024}
    correction = {"schema_version": 1, "correction": "synthetic"}
    frozen_receipt = {
        "source_allowlist_sha256": hashlib.sha256(
            controller._canonical_bytes(source_allowlist)
        ).hexdigest(),
        "workspace_plan_sha256": hashlib.sha256(
            controller._canonical_bytes(workspace_plan)
        ).hexdigest(),
        "cnn_correction_receipt_sha256": hashlib.sha256(
            controller._canonical_bytes(correction)
        ).hexdigest(),
        "execution_source_root_sha256": "1" * 64,
        "execution_source_manifest_sha256": "2" * 64,
        "execution_source_delta_sha256": "3" * 64,
        "retired_input_invalidation_receipt_path": str(invalidation),
        "retired_input_invalidation_receipt_sha256": invalidation_record["sha256"],
    }
    run_state = {"root": str(root / "artifacts" / "runs"), "sha256": "4" * 64}
    parent = root / "artifacts" / "preregistration_amendments" / "authority-c"
    parent.mkdir(parents=True)
    context = {
        "namespace": namespace,
        "paths": {
            "project_root": root,
            "control_root": control_root,
            "failed_preflight": failed,
            "prior_failure": prior,
            "retired_input": retired_root,
            "retired_input_invalidation": invalidation,
            "parent": parent,
        },
        "payloads": {
            "source_allowlist": source_allowlist,
            "workspace_plan": workspace_plan,
            "cnn_correction_receipt": correction,
            "frozen_source_receipt": frozen_receipt,
        },
        "run_state": run_state,
        "retired_input_invalidation": {
            "receipt": invalidation_payload,
            "record": invalidation_record,
            "sha256": invalidation_record["sha256"],
        },
        "prior": {
            "receipt_sha256": hashlib.sha256(prior.read_bytes()).hexdigest(),
        },
        "authorization_sha256": "5" * 64,
        "authority": {"status": "synthetic-authority"},
        "controller_record": controller._live_file_record(
            CONTROLLER_PATH,
            "replacement controller",
        ),
        "source": {
            "current_source": {
                "root_sha256": frozen_receipt["execution_source_root_sha256"],
            },
            "delta_sha256": frozen_receipt["execution_source_delta_sha256"],
        },
    }
    resources = {
        "capacity_dict": {"passed": True, "free_bytes": 10_000},
        "compute": SimpleNamespace(evidence_sha256="6" * 64),
    }
    return SimpleNamespace(
        root=root,
        control_root=control_root,
        namespace=namespace,
        failed=failed,
        prior=prior,
        retired_root=retired_root,
        invalidation=invalidation,
        invalidation_payload=invalidation_payload,
        context=context,
        resources=resources,
        destination=control_root / controller._REPLACEMENT_INPUT_DIRECTORY_NAME,
    )


def _synthetic_invalidation_publisher_context(
    tmp_path: Path,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_readback: bool = False,
) -> SimpleNamespace:
    root = tmp_path / "invalidation-project"
    control_root = root / "artifacts" / "resource_control"
    amendment_root = root / "artifacts" / "preregistration_amendments"
    parent = amendment_root / controller._AUTHORITY_C_COMPONENT
    control_root.mkdir(parents=True)
    parent.mkdir(parents=True)
    retired_root = control_root / controller._RETIRED_INPUT_DIRECTORY_NAME
    retired_root.mkdir()
    prior_path = control_root / controller._PRIOR_FAILURE_RECEIPT_FILENAME
    prior_path.write_bytes(b"preserved-prior\n")
    run_root = root / "artifacts" / "runs"
    run_root.mkdir()
    namespace = controller.Namespace(control_root)
    destination = control_root / controller._RETIRED_INPUT_INVALIDATION_FILENAME
    files = {
        role: {
            "path": str(retired_root / filename),
            "size_bytes": controller._RETIRED_INPUT_FILE_PINS[role]["size_bytes"],
            "sha256": controller._RETIRED_INPUT_FILE_PINS[role]["sha256"],
        }
        for role, filename in controller._REPLACEMENT_INPUT_FILENAMES.items()
    }
    files = {role: files[role] for role in sorted(files)}
    retired = {
        "directory": str(retired_root),
        "files": files,
        "records_sha256": controller._compact_sha256(files),
        "controller_path": str(CONTROLLER_PATH),
        "controller_size_bytes": controller._RETIRED_CONTROLLER_SIZE_BYTES,
        "controller_sha256": controller._RETIRED_CONTROLLER_SHA256,
        "execution_source_root_sha256": controller._RETIRED_SOURCE_ROOT_SHA256,
        "execution_source_manifest_sha256": controller._RETIRED_SOURCE_MANIFEST_SHA256,
        "execution_source_delta_sha256": controller._RETIRED_SOURCE_DELTA_SHA256,
        "authorization_sha256": controller._RETIRED_AUTHORIZATION_SHA256,
    }
    prior_record = {
        "path": str(prior_path),
        "size_bytes": controller._PRIOR_FAILURE_RECEIPT_SIZE_BYTES,
        "sha256": controller._PRIOR_FAILURE_RECEIPT_SHA256,
    }
    logs = {
        role: {
            "path": str(control_root / pin["filename"]),
            "size_bytes": pin["size_bytes"],
            "sha256": pin["sha256"],
        }
        for role, pin in controller._RETIRED_LOG_PINS.items()
    }
    failed = {
        "receipt": {
            "path": str(control_root / controller._FAILED_PREFLIGHT_FILENAME),
            "size_bytes": controller._FAILED_PREFLIGHT_SIZE_BYTES,
            "sha256": controller._FAILED_PREFLIGHT_SHA256,
        },
        "stored_observed_at_utc": controller._HISTORICAL_FAILED_AT_UTC,
        "normalized_observed_at_utc": "2026-07-27T17:30:54.689000Z",
        "logs": {role: logs[role] for role in sorted(logs)},
        "error_type": "ControlError",
        "error_message": controller._RETIRED_FAILURE_ERROR_MESSAGE,
        "error_sha256": controller._RETIRED_FAILURE_ERROR_SHA256,
        "publication_authorization_created": False,
        "attempt_marker_created": False,
        "authority_d_created": False,
        "scientific_run_started": False,
    }
    real_live_file_record = controller._live_file_record
    controller_record = real_live_file_record(
        CONTROLLER_PATH,
        "corrected replacement controller",
    )
    run_state = {
        "root": str(run_root),
        "files": {},
        "sha256": controller._RETIRED_RUN_STATE_SHA256,
    }

    def live_file_record(path: Path, role: str, **kwargs: Any) -> dict[str, Any]:
        candidate = Path(path).resolve()
        if candidate == prior_path.resolve():
            return dict(prior_record)
        if candidate == CONTROLLER_PATH.resolve():
            return dict(controller_record)
        return real_live_file_record(path, role, **kwargs)

    def readback(
        _namespace: Any,
        **_kwargs: Any,
    ) -> tuple[dict[str, Any], str]:
        if fail_readback:
            raise controller.ControlError("injected post-write invalidation readback failure")
        encoded = destination.read_bytes()
        raw = controller._strict_json_object(encoded, "synthetic invalidation")
        canonical = controller._canonical_retired_input_invalidation(
            raw,
            namespace=namespace,
        )
        assert encoded == controller._canonical_bytes(canonical)
        return canonical, hashlib.sha256(encoded).hexdigest()

    monkeypatch.setattr(
        workflows,
        "verify_resource_bounded_prior_publication_failure_receipt",
        lambda **_kwargs: {"status": "verified"},
    )
    monkeypatch.setattr(controller, "discover_candidates", lambda _parent: [])
    monkeypatch.setattr(controller, "_retired_bundle_snapshot", lambda _namespace: retired)
    monkeypatch.setattr(
        controller,
        "_retired_failure_evidence_snapshot",
        lambda _namespace: failed,
    )
    monkeypatch.setattr(controller, "_live_file_record", live_file_record)
    monkeypatch.setattr(controller, "_live_run_state", lambda _paths: run_state)
    monkeypatch.setattr(controller, "_read_retired_input_invalidation", readback)
    return SimpleNamespace(
        root=root,
        control_root=control_root,
        namespace=namespace,
        parent=parent,
        retired_root=retired_root,
        prior_path=prior_path,
        destination=destination,
        retired=retired,
        failed=failed,
        controller_record=controller_record,
        run_state=run_state,
    )


def test_retired_v1_invalidation_publisher_is_o_excl_and_exactly_read_back(
    tmp_path: Path,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _synthetic_invalidation_publisher_context(
        tmp_path,
        controller,
        monkeypatch,
    )

    result = controller.publish_retired_input_invalidation_once(
        namespace=tree.namespace,
        parent_authority_directory=tree.parent,
        invalidated_at=datetime.now(UTC),
    )

    assert result["status"] == "retired_v1_preserved_invalid_nonpublishable"
    assert result["receipt_sha256"] == hashlib.sha256(tree.destination.read_bytes()).hexdigest()
    payload = controller._strict_json_object(
        tree.destination.read_bytes(),
        "retired invalidation",
    )
    assert payload["retired_bundle"] == tree.retired
    assert payload["failed_preflight_evidence"] == tree.failed
    assert payload["corrected_controller"] == tree.controller_record
    assert payload["run_state_sha256"] == tree.run_state["sha256"]
    tampered = copy.deepcopy(payload)
    tampered["run_state_sha256"] = "0" * 64
    with pytest.raises(controller.ControlError, match="frozen v1 lineage"):
        controller._canonical_retired_input_invalidation(
            tampered,
            namespace=tree.namespace,
        )
    assert tuple(tree.retired_root.iterdir()) == ()
    assert tree.prior_path.read_bytes() == b"preserved-prior\n"
    with pytest.raises(FileExistsError, match="already exists"):
        controller.publish_retired_input_invalidation_once(
            namespace=tree.namespace,
            parent_authority_directory=tree.parent,
            invalidated_at=datetime.now(UTC),
        )


def test_retired_v1_invalidation_postwrite_failure_rolls_back_only_owned_receipt(
    tmp_path: Path,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _synthetic_invalidation_publisher_context(
        tmp_path,
        controller,
        monkeypatch,
        fail_readback=True,
    )
    events: list[str] = []
    _install_bundle_lock_probe(controller, monkeypatch, events)

    with pytest.raises(controller.ControlError, match="post-write invalidation readback"):
        controller.publish_retired_input_invalidation_once(
            namespace=tree.namespace,
            parent_authority_directory=tree.parent,
            invalidated_at=datetime.now(UTC),
        )

    assert not tree.destination.exists()
    assert tree.retired_root.is_dir()
    assert tree.prior_path.read_bytes() == b"preserved-prior\n"
    rollback_index = events.index("rollback")
    assert events[rollback_index - 1 : rollback_index + 3] == [
        "assert_owned",
        "rollback",
        "assert_owned",
        "lock_exit",
    ]


def test_retired_v1_invalidation_preserves_receipt_when_lock_ownership_is_lost(
    tmp_path: Path,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _synthetic_invalidation_publisher_context(
        tmp_path,
        controller,
        monkeypatch,
    )
    events: list[str] = []
    _install_bundle_lock_probe(
        controller,
        monkeypatch,
        events,
        fail_assertions_from=3,
    )

    with pytest.raises(controller.AmbiguousStateError, match="exact disposition is ambiguous"):
        controller.publish_retired_input_invalidation_once(
            namespace=tree.namespace,
            parent_authority_directory=tree.parent,
            invalidated_at=datetime.now(UTC),
        )

    assert tree.destination.is_file()
    assert "rollback" not in events
    assert events[-1] == "lock_exit"


def test_retired_v1_invalidation_rejects_existing_v2_before_reconstruction(
    tmp_path: Path,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _synthetic_invalidation_publisher_context(
        tmp_path,
        controller,
        monkeypatch,
    )
    (tree.control_root / controller._REPLACEMENT_INPUT_DIRECTORY_NAME).mkdir()
    reconstruction_calls = 0

    def forbidden_reconstruction(_namespace: Any) -> Any:
        nonlocal reconstruction_calls
        reconstruction_calls += 1
        raise AssertionError("forbidden state must stop before reconstruction")

    monkeypatch.setattr(
        controller,
        "_retired_bundle_snapshot",
        forbidden_reconstruction,
    )

    with pytest.raises(controller.ControlError, match="requires no v2"):
        controller.publish_retired_input_invalidation_once(
            namespace=tree.namespace,
            parent_authority_directory=tree.parent,
        )

    assert reconstruction_calls == 0
    assert not tree.destination.exists()


def test_retired_v1_invalidation_rejects_future_timestamp_without_write(
    tmp_path: Path,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _synthetic_invalidation_publisher_context(
        tmp_path,
        controller,
        monkeypatch,
    )

    with pytest.raises(controller.ControlError, match="must not be in the future"):
        controller.publish_retired_input_invalidation_once(
            namespace=tree.namespace,
            parent_authority_directory=tree.parent,
            invalidated_at=datetime(2099, 1, 1, tzinfo=UTC),
        )

    assert not tree.destination.exists()


def _install_synthetic_freeze_adapters(
    controller: Any,
    tree: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    *,
    live_run_state: dict[str, Any] | None = None,
) -> None:
    monkeypatch.setattr(
        controller,
        "_derive_live_foundation",
        lambda _namespace, _parent: tree.context,
    )
    monkeypatch.setattr(
        controller,
        "_live_paths",
        lambda _namespace, _parent: tree.context["paths"],
    )
    monkeypatch.setattr(
        controller,
        "_read_retired_input_invalidation",
        lambda _namespace, **_kwargs: (
            tree.context["retired_input_invalidation"]["receipt"],
            tree.context["retired_input_invalidation"]["sha256"],
        ),
    )
    monkeypatch.setattr(controller, "_finalize_live_context", lambda foundation: foundation)
    monkeypatch.setattr(
        controller,
        "_require_live_capacity_and_compute",
        lambda _context: tree.resources,
    )
    monkeypatch.setattr(
        controller,
        "_derive_live_source",
        lambda _paths: tree.context["source"],
    )
    monkeypatch.setattr(
        controller,
        "_require_live_authority_and_config",
        lambda _paths: tree.context["authority"],
    )
    monkeypatch.setattr(
        controller,
        "_live_run_state",
        lambda _paths: live_run_state if live_run_state is not None else tree.context["run_state"],
    )
    monkeypatch.setattr(
        workflows,
        "verify_resource_bounded_prior_publication_failure_receipt",
        lambda **_kwargs: {"status": "verified"},
    )


def test_live_freeze_publishes_and_verifies_exact_singleton_bundle(
    tmp_path: Path,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _synthetic_live_freeze_context(tmp_path, controller, monkeypatch)
    _install_synthetic_freeze_adapters(controller, tree, monkeypatch)
    retired_before = {path.name: path.read_bytes() for path in tree.retired_root.iterdir()}
    invalidation_before = tree.invalidation.read_bytes()

    result = controller.freeze_replacement_inputs_once(
        namespace=tree.namespace,
        parent_authority_directory=tree.root / "authority-c",
        output_directory=tree.destination,
    )

    assert result["status"] == "replacement_inputs_frozen_and_verified"
    assert result["output_directory"] == str(tree.destination)
    assert result["publication_performed"] is False
    assert tuple(sorted(path.name for path in tree.destination.iterdir())) == tuple(
        sorted(controller._REPLACEMENT_INPUT_FILENAMES.values())
    )
    for role, filename in controller._REPLACEMENT_INPUT_FILENAMES.items():
        path = tree.destination / filename
        assert path.read_bytes() == controller._canonical_bytes(tree.context["payloads"][role])
        assert result["files"][role]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    frozen_source = json.loads(
        (
            tree.destination / controller._REPLACEMENT_INPUT_FILENAMES["frozen_source_receipt"]
        ).read_text(encoding="utf-8")
    )
    assert frozen_source["retired_input_invalidation_receipt_path"] == str(tree.invalidation)
    assert (
        frozen_source["retired_input_invalidation_receipt_sha256"]
        == hashlib.sha256(invalidation_before).hexdigest()
    )
    assert {path.name: path.read_bytes() for path in tree.retired_root.iterdir()} == retired_before
    assert tree.invalidation.read_bytes() == invalidation_before


def test_live_freeze_rejects_noncanonical_singleton_name_without_writing(
    tmp_path: Path,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _synthetic_live_freeze_context(tmp_path, controller, monkeypatch)
    alternate = tree.control_root / f"{controller._REPLACEMENT_INPUT_PREFIX}alternate"
    derivation_calls = 0

    def forbidden_derivation(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal derivation_calls
        derivation_calls += 1
        raise AssertionError("noncanonical destination must fail before derivation")

    monkeypatch.setattr(controller, "_derive_live_foundation", forbidden_derivation)

    with pytest.raises(controller.ControlError, match="canonical singleton"):
        controller.freeze_replacement_inputs_once(
            namespace=tree.namespace,
            parent_authority_directory=tree.root / "authority-c",
            output_directory=alternate,
        )

    assert not alternate.exists()
    assert not tree.destination.exists()
    assert derivation_calls == 0


def test_live_freeze_uses_one_foundation_and_two_resource_gates(
    tmp_path: Path,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _synthetic_live_freeze_context(tmp_path, controller, monkeypatch)
    _install_synthetic_freeze_adapters(controller, tree, monkeypatch)
    events: list[str] = []

    def foundation(_namespace: Any, _parent: Any) -> dict[str, Any]:
        events.append("foundation")
        return tree.context

    def resource_gate(_context: Any) -> dict[str, Any]:
        events.append("resource_gate")
        return tree.resources

    real_create_directory = controller.create_directory_no_overwrite

    def create_directory(path: Path) -> Any:
        events.append("first_write")
        return real_create_directory(path)

    monkeypatch.setattr(controller, "_derive_live_foundation", foundation)
    monkeypatch.setattr(controller, "_require_live_capacity_and_compute", resource_gate)
    monkeypatch.setattr(controller, "create_directory_no_overwrite", create_directory)

    controller.freeze_replacement_inputs_once(
        namespace=tree.namespace,
        parent_authority_directory=tree.root / "authority-c",
        output_directory=tree.destination,
    )

    assert events.count("foundation") == 1
    assert events.count("resource_gate") == 2
    assert events.index("first_write") == len(events) - 1
    assert events[-2:] == ["resource_gate", "first_write"]


def test_live_freeze_second_resource_gate_failure_leaves_bundle_absent(
    tmp_path: Path,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _synthetic_live_freeze_context(tmp_path, controller, monkeypatch)
    _install_synthetic_freeze_adapters(controller, tree, monkeypatch)
    gate_calls = 0

    def resource_gate(_context: Any) -> dict[str, Any]:
        nonlocal gate_calls
        gate_calls += 1
        if gate_calls == 2:
            raise controller.ControlError("injected second resource gate failure")
        return tree.resources

    monkeypatch.setattr(controller, "_require_live_capacity_and_compute", resource_gate)

    with pytest.raises(controller.ControlError, match="second resource gate"):
        controller.freeze_replacement_inputs_once(
            namespace=tree.namespace,
            parent_authority_directory=tree.root / "authority-c",
            output_directory=tree.destination,
        )

    assert gate_calls == 2
    assert not tree.destination.exists()
    assert tree.prior.is_file()


def test_live_freeze_rolls_back_owned_bundle_when_run_state_drifts(
    tmp_path: Path,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _synthetic_live_freeze_context(tmp_path, controller, monkeypatch)
    _install_synthetic_freeze_adapters(controller, tree, monkeypatch)
    events: list[str] = []
    _install_bundle_lock_probe(controller, monkeypatch, events)
    states = iter(
        (
            tree.context["run_state"],
            tree.context["run_state"],
            {"root": str(tree.root / "artifacts" / "runs"), "sha256": "9" * 64},
        )
    )
    monkeypatch.setattr(controller, "_live_run_state", lambda _paths: next(states))

    with pytest.raises(controller.ControlError, match="run-state changed"):
        controller.freeze_replacement_inputs_once(
            namespace=tree.namespace,
            parent_authority_directory=tree.root / "authority-c",
            output_directory=tree.destination,
        )

    assert not tree.destination.exists()
    assert tree.prior.is_file()
    rollback_index = events.index("rollback")
    assert events[rollback_index - 1 : rollback_index + 3] == [
        "assert_owned",
        "rollback",
        "assert_owned",
        "lock_exit",
    ]


def test_live_freeze_preserves_bundle_when_lock_ownership_is_lost(
    tmp_path: Path,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _synthetic_live_freeze_context(tmp_path, controller, monkeypatch)
    _install_synthetic_freeze_adapters(controller, tree, monkeypatch)
    events: list[str] = []
    _install_bundle_lock_probe(
        controller,
        monkeypatch,
        events,
        fail_assertions_from=13,
    )

    with pytest.raises(controller.AmbiguousStateError, match="exact disposition is ambiguous"):
        controller.freeze_replacement_inputs_once(
            namespace=tree.namespace,
            parent_authority_directory=tree.root / "authority-c",
            output_directory=tree.destination,
        )

    assert tree.destination.is_dir()
    assert tuple(sorted(path.name for path in tree.destination.iterdir())) == tuple(
        sorted(controller._REPLACEMENT_INPUT_FILENAMES.values())
    )
    assert "rollback" not in events
    assert events[-1] == "lock_exit"


@pytest.mark.parametrize(
    "drift_role",
    ["source", "authority", "controller", "prior", "invalidation"],
)
def test_live_freeze_rolls_back_bundle_on_post_write_live_input_drift(
    tmp_path: Path,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
    drift_role: str,
) -> None:
    tree = _synthetic_live_freeze_context(tmp_path, controller, monkeypatch)
    _install_synthetic_freeze_adapters(controller, tree, monkeypatch)

    if drift_role == "source":
        calls = 0

        def source(_paths: Any) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            if calls == 3:
                return tree.context["source"] | {"delta_sha256": "f" * 64}
            return tree.context["source"]

        monkeypatch.setattr(controller, "_derive_live_source", source)
    elif drift_role == "authority":
        calls = 0

        def authority(_paths: Any) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            if calls == 3:
                return {"status": "drifted-authority"}
            return tree.context["authority"]

        monkeypatch.setattr(controller, "_require_live_authority_and_config", authority)
    elif drift_role == "invalidation":
        invalidation_calls = 0

        def invalidation(
            _namespace: Any,
            **_kwargs: Any,
        ) -> tuple[dict[str, Any], str]:
            nonlocal invalidation_calls
            invalidation_calls += 1
            sha256 = tree.context["retired_input_invalidation"]["sha256"]
            if invalidation_calls == 4:
                sha256 = "c" * 64
            return tree.context["retired_input_invalidation"]["receipt"], sha256

        monkeypatch.setattr(
            controller,
            "_read_retired_input_invalidation",
            invalidation,
        )
    else:
        real_record = controller._live_file_record
        controller_calls = 0
        prior_calls = 0

        def file_record(path: Path, role: str) -> dict[str, Any]:
            nonlocal controller_calls, prior_calls
            if role == "replacement controller":
                controller_calls += 1
                record = dict(tree.context["controller_record"])
                if drift_role == "controller" and controller_calls == 3:
                    record["sha256"] = "e" * 64
                return record
            if role == "prior-publication failure receipt":
                prior_calls += 1
                record = real_record(path, role)
                if drift_role == "prior" and prior_calls == 3:
                    record["sha256"] = "d" * 64
                return record
            return real_record(path, role)

        monkeypatch.setattr(controller, "_live_file_record", file_record)

    with pytest.raises(controller.ControlError, match="changed during"):
        controller.freeze_replacement_inputs_once(
            namespace=tree.namespace,
            parent_authority_directory=tree.root / "authority-c",
            output_directory=tree.destination,
        )

    assert not tree.destination.exists()
    assert tree.prior.is_file()


def test_prefixed_replacement_bundle_alias_is_rejected_everywhere(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    alias = control_tree.control_root / f"{controller._REPLACEMENT_INPUT_PREFIX}v3"
    alias_bindings = controller.FrozenInputBindings(
        failed_preflight_receipt=control_tree.frozen_inputs.failed_preflight_receipt,
        prior_failure_receipt=control_tree.frozen_inputs.prior_failure_receipt,
        retired_input_invalidation=control_tree.frozen_inputs.retired_input_invalidation,
        frozen_source_receipt=(
            alias / controller._REPLACEMENT_INPUT_FILENAMES["frozen_source_receipt"]
        ),
        source_allowlist=alias / controller._REPLACEMENT_INPUT_FILENAMES["source_allowlist"],
        workspace_plan=alias / controller._REPLACEMENT_INPUT_FILENAMES["workspace_plan"],
        cnn_correction_receipt=(
            alias / controller._REPLACEMENT_INPUT_FILENAMES["cnn_correction_receipt"]
        ),
    )

    with pytest.raises(controller.ControlError, match="canonical singleton"):
        controller._capture_attempt_inputs(
            control_tree.namespace,
            alias_bindings,
            capture_run_state=False,
        )
    with pytest.raises(controller.ControlError, match="canonical singleton"):
        controller._replacement_bindings(
            {"control_root": control_tree.control_root},
            alias,
        )

    authorization = copy.deepcopy(control_tree.publication_authorization_payload)
    authorization["preflight"]["contract"]["frozen_input_bundle"]["directory"] = str(alias)
    with pytest.raises(controller.ControlError, match="canonical singleton"):
        controller._canonical_publication_authorization(
            authorization,
            namespace=control_tree.namespace,
        )

    alias.mkdir()
    state = _classify(controller, control_tree)
    assert state.state is controller.State.STOP_AMBIGUOUS
    assert "unexpected replacement namespace entries" in state.reason


def test_case_variant_replacement_singleton_is_rejected_by_api_and_classifier(
    control_tree: SimpleNamespace,
    controller: Any,
) -> None:
    case_variant = control_tree.control_root / "Authority_D_Replacement_Inputs_V2"
    with pytest.raises(controller.ControlError, match="canonical singleton"):
        controller._replacement_bindings(
            {"control_root": control_tree.control_root},
            case_variant,
        )

    canonical = control_tree.frozen_inputs.workspace_plan.parent
    intermediate = control_tree.control_root / "replacement-input-case-rename"
    canonical.rename(intermediate)
    intermediate.rename(case_variant)

    state = _classify(controller, control_tree)
    assert state.state is controller.State.STOP_AMBIGUOUS
    assert "unexpected replacement namespace entries" in state.reason


def _synthetic_live_preflight(control_tree: SimpleNamespace) -> dict[str, Any]:
    payload = control_tree.publication_authorization_payload
    timestamp_text = payload["publication"]["amendment_timestamp_utc"]
    timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00")).astimezone(UTC)
    return {
        "context": {
            "paths": {
                "project_root": control_tree.root,
                "parent": control_tree.parent,
                "amendment_root": control_tree.parent.parent,
                "config": Path(payload["preflight"]["contract"]["config"]["path"]),
            },
            "authorization": {"policy": "synthetic-resource-authority"},
            "authorization_sha256": payload["preflight"]["contract"]["technical_successor"][
                "authorization_sha256"
            ],
        },
        "bindings": control_tree.frozen_inputs,
        "storage_policy": {"policy": "synthetic-storage"},
        "timestamp": timestamp,
        "timestamp_text": timestamp_text,
        "destination": control_tree.destination,
        "intent_sha256": payload["preflight"]["contract"]["technical_successor"]["intent_sha256"],
        "contract": copy.deepcopy(payload["preflight"]["contract"]),
        "preflight_fingerprint_sha256": payload["preflight"]["preflight_fingerprint_sha256"],
        "capacity_observation": copy.deepcopy(payload["preflight"]["capacity_observation"]),
        "compute_observation": copy.deepcopy(payload["preflight"]["compute_observation"]),
    }


def test_live_authorize_publication_success_uses_no_overwrite_receipt(
    control_tree: SimpleNamespace,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_tree.publication_authorization.unlink()
    preflight = _synthetic_live_preflight(control_tree)
    preflight_calls: list[dict[str, Any]] = []

    def build_preflight(**kwargs: Any) -> dict[str, Any]:
        preflight_calls.append(kwargs)
        return preflight

    monkeypatch.setattr(controller, "_build_live_preflight", build_preflight)
    monkeypatch.setattr(controller.secrets, "token_hex", lambda _size: "a" * 64)

    result = controller.authorize_replacement_publication_once(
        namespace=control_tree.namespace,
        parent_authority_directory=control_tree.parent,
        input_directory=control_tree.frozen_inputs.workspace_plan.parent,
    )

    assert result["status"] == "publication_authorized_for_one_attempt"
    assert result["authorized_attempt_id"] == "a" * 64
    assert result["max_attempt_count"] == 1
    assert result["publication_performed"] is False
    encoded = control_tree.publication_authorization.read_bytes()
    assert encoded == controller._canonical_bytes(
        controller._read_publication_authorization(control_tree.namespace)[0]
    )
    assert len(preflight_calls) == 2
    assert "amendment_timestamp" not in preflight_calls[0]
    assert preflight_calls[1]["amendment_timestamp"] == preflight["timestamp"]


def test_live_authorize_publication_refuses_existing_receipt_before_preflight(
    control_tree: SimpleNamespace,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = control_tree.publication_authorization.read_bytes()

    def forbidden_preflight(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("preflight must not run after the one-attempt receipt exists")

    monkeypatch.setattr(controller, "_build_live_preflight", forbidden_preflight)

    with pytest.raises(FileExistsError, match="already exists"):
        controller.authorize_replacement_publication_once(
            namespace=control_tree.namespace,
            parent_authority_directory=control_tree.parent,
            input_directory=control_tree.frozen_inputs.workspace_plan.parent,
        )

    assert control_tree.publication_authorization.read_bytes() == before


def test_live_authorize_publication_rolls_back_receipt_on_repeated_preflight_drift(
    control_tree: SimpleNamespace,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_tree.publication_authorization.unlink()
    stable = _synthetic_live_preflight(control_tree)
    drifted = copy.deepcopy(stable)
    drifted["intent_sha256"] = "f" * 64
    builds = iter((stable, drifted))
    monkeypatch.setattr(controller, "_build_live_preflight", lambda **_kwargs: next(builds))
    monkeypatch.setattr(controller.secrets, "token_hex", lambda _size: "b" * 64)
    events: list[str] = []
    captured_paths = _install_bundle_lock_probe(controller, monkeypatch, events)

    with pytest.raises(controller.ControlError, match="changed during publication authorization"):
        controller.authorize_replacement_publication_once(
            namespace=control_tree.namespace,
            parent_authority_directory=control_tree.parent,
            input_directory=control_tree.frozen_inputs.workspace_plan.parent,
        )

    assert not control_tree.publication_authorization.exists()
    assert len(captured_paths) == 1
    assert set(captured_paths[0]) == {
        control_tree.frozen_inputs.workspace_plan.parent.resolve(),
        control_tree.namespace.publication_authorization.resolve(),
        control_tree.namespace.attempt.resolve(),
        control_tree.namespace.success.resolve(),
        control_tree.namespace.failure.resolve(),
    }
    rollback_index = events.index("rollback")
    assert events[rollback_index - 1 : rollback_index + 3] == [
        "assert_owned",
        "rollback",
        "assert_owned",
        "lock_exit",
    ]


def test_live_authorize_preserves_receipt_when_lock_ownership_is_lost(
    control_tree: SimpleNamespace,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_tree.publication_authorization.unlink()
    preflight = _synthetic_live_preflight(control_tree)
    monkeypatch.setattr(
        controller,
        "_build_live_preflight",
        lambda **_kwargs: preflight,
    )
    monkeypatch.setattr(controller.secrets, "token_hex", lambda _size: "c" * 64)
    events: list[str] = []
    _install_bundle_lock_probe(
        controller,
        monkeypatch,
        events,
        fail_assertions_from=3,
    )

    with pytest.raises(controller.AmbiguousStateError, match="exact disposition is ambiguous"):
        controller.authorize_replacement_publication_once(
            namespace=control_tree.namespace,
            parent_authority_directory=control_tree.parent,
            input_directory=control_tree.frozen_inputs.workspace_plan.parent,
        )

    assert control_tree.publication_authorization.is_file()
    assert "rollback" not in events
    assert events[-1] == "lock_exit"


def test_live_authorize_preserves_exact_receipt_on_lock_exit_failure(
    control_tree: SimpleNamespace,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_tree.publication_authorization.unlink()
    preflight = _synthetic_live_preflight(control_tree)
    monkeypatch.setattr(
        controller,
        "_build_live_preflight",
        lambda **_kwargs: preflight,
    )
    monkeypatch.setattr(controller.secrets, "token_hex", lambda _size: "d" * 64)
    events: list[str] = []
    _install_bundle_lock_probe(
        controller,
        monkeypatch,
        events,
        fail_exit=True,
    )

    with pytest.raises(controller.AmbiguousStateError, match="exact disposition is ambiguous"):
        controller.authorize_replacement_publication_once(
            namespace=control_tree.namespace,
            parent_authority_directory=control_tree.parent,
            input_directory=control_tree.frozen_inputs.workspace_plan.parent,
        )

    encoded = control_tree.publication_authorization.read_bytes()
    receipt, digest = controller._read_publication_authorization(control_tree.namespace)
    assert encoded == controller._canonical_bytes(receipt)
    assert hashlib.sha256(encoded).hexdigest() == digest
    assert "rollback" not in events
    assert events[-1] == "lock_exit"


def test_live_publish_wrapper_calls_creator_once_with_guarded_callback_and_no_training(
    control_tree: SimpleNamespace,
    controller: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _synthetic_live_preflight(control_tree)
    creator_calls: list[dict[str, Any]] = []
    callback_values: list[Any] = []
    preflight_calls: list[dict[str, Any]] = []
    published = SimpleNamespace(token="published")

    def build_preflight(**kwargs: Any) -> dict[str, Any]:
        preflight_calls.append(kwargs)
        return preflight

    monkeypatch.setattr(controller, "_build_live_preflight", build_preflight)

    def forbidden_training(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("publication wrapper must not start scientific training")

    monkeypatch.setattr(
        resource_bounded_runner,
        "execute_resource_bounded_sensitivity",
        forbidden_training,
    )

    def fake_creator(**kwargs: Any) -> Any:
        creator_calls.append(kwargs)
        kwargs["post_publication_check"](published)
        return "created-authority-d"

    monkeypatch.setattr(amendment, "create_preregistration_amendment", fake_creator)

    def fake_execute_once(**kwargs: Any) -> Any:
        kwargs["preclaim_check"]()

        def callback(value: Any) -> None:
            callback_values.append(value)

        return kwargs["transaction"](callback)

    monkeypatch.setattr(controller, "execute_once", fake_execute_once)

    result = controller.publish_replacement_authority_once(
        namespace=control_tree.namespace,
        parent_authority_directory=control_tree.parent,
        input_directory=control_tree.frozen_inputs.workspace_plan.parent,
    )

    assert result == "created-authority-d"
    assert callback_values == [published]
    assert len(preflight_calls) == 2
    assert preflight_calls[0]["amendment_timestamp"] == preflight["timestamp"]
    assert preflight_calls[1]["amendment_timestamp"] == preflight["timestamp"]
    assert len(creator_calls) == 1
    creator_kwargs = creator_calls[0]
    assert creator_kwargs["outcomes_inspected"] is True
    assert creator_kwargs["timestamp"] == preflight["timestamp"]
    assert creator_kwargs["parent_authority_directory"] == control_tree.parent
    assert creator_kwargs["amendment_root"] == control_tree.parent.parent
    assert (
        creator_kwargs["resource_bounded_technical_successor_authorization"]
        == preflight["context"]["authorization"]
    )
    assert creator_kwargs["confirmatory_storage_policy"] == preflight["storage_policy"]
