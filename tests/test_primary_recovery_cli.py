"""Focused CLI contracts for bounded interrupted-primary recovery."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner

import histo_audit.config as config_module
import histo_audit.experiment as experiment
import histo_audit.experiment.primary_recovery_runner as recovery_runner
import histo_audit.workflows as workflows
from histo_audit.cli import app
from histo_audit.workflows import (
    PreregistrationAmendmentResult,
    PreregistrationAmendmentVerification,
)


def _amend_arguments(project: Path) -> list[str]:
    return [
        "preregistration",
        "amend",
        "--project-root",
        str(project),
        "--parent-authority",
        "authorities/base",
        "--amended-preregistration",
        "PRE_REGISTRATION_NEXT.md",
        "--amended-primary-config",
        "configs/primary_next.yaml",
        "--amended-confirmatory-config",
        "configs/confirmatory_next.yaml",
        "--reason",
        "Recover the interrupted primary without changing the scientific method.",
        "--affected-hypothesis",
        "H1",
        "--affected-analysis",
        "primary_ranking",
        "--amendment-timestamp-utc",
        "2026-07-27T12:00:00Z",
        "--amendment-root",
        "artifacts/amendments",
    ]


def _fake_result(project: Path) -> PreregistrationAmendmentResult:
    destination = (project / "artifacts/amendments/20260727T120000.000000Z").resolve()
    return PreregistrationAmendmentResult(
        amendment_directory=destination,
        parent_authority_directory=(project / "authorities/base").resolve(),
        amendment_timestamp_utc="2026-07-27T12:00:00.000000Z",
        chain_depth=2,
        amendment_evidence_path=destination / "amendment_evidence.json",
        amended_preregistration_path=destination / "PRE_REGISTRATION_FROZEN.md",
        amended_primary_config_path=destination / "primary_frozen.yaml",
        amended_confirmatory_config_path=destination / "confirmatory_frozen.yaml",
        source_tree_manifest_path=destination / "source_tree_manifest.json",
        sha256_manifest_path=destination / "sha256_manifest.json",
        immutable_marker_path=destination / ".immutable.json",
        artifact_root_sha256="a" * 64,
        sha256_manifest_sha256="b" * 64,
    )


def _valid_verification(
    result: PreregistrationAmendmentResult,
) -> PreregistrationAmendmentVerification:
    return PreregistrationAmendmentVerification(
        valid=True,
        amendment_directory=result.amendment_directory,
        chain_depth=result.chain_depth,
        artifact_root_sha256=result.artifact_root_sha256,
        sha256_manifest_sha256=result.sha256_manifest_sha256,
        parent_authority_directory=result.parent_authority_directory,
        errors=(),
    )


def test_amend_cli_maps_primary_recovery_authorization_json(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    authorization = {
        "schema_version": 1,
        "policy": "interrupted_unsealed_primary_recovery_v1",
        "source_run_id": "source-run",
    }
    authorization_path = project / "recovery_authorization.json"
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    expected = _fake_result(project)
    captured: dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> PreregistrationAmendmentResult:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(workflows, "create_preregistration_amendment", fake_create)
    monkeypatch.setattr(
        workflows,
        "verify_preregistration_amendment",
        lambda _path: _valid_verification(expected),
    )

    result = CliRunner().invoke(
        app,
        [
            *_amend_arguments(project),
            "--outcomes-inspected",
            "--outcomes-inspected-at-utc",
            "2026-07-27T10:57:07Z",
            "--primary-recovery-authorization-json",
            authorization_path.name,
            "--confirmatory-single-copy-checkpoint-storage",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["primary_recovery_authorization"] == authorization
    assert captured["outcomes_inspected"] is True
    assert captured["outcomes_inspected_at"].isoformat() == "2026-07-27T10:57:07+00:00"
    assert captured["finalization_successor_authorization"] is None
    assert captured["confirmatory_storage_policy"] is not None


def test_amend_cli_recovery_is_mutually_exclusive_with_finalization(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    called = False

    def fake_create(**_kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(workflows, "create_preregistration_amendment", fake_create)
    result = CliRunner().invoke(
        app,
        [
            *_amend_arguments(tmp_path),
            "--outcomes-inspected",
            "--outcomes-inspected-at-utc",
            "2026-07-27T10:57:07Z",
            "--primary-recovery-authorization-json",
            "recovery.json",
            "--finalization-predecessor-run-dir",
            "artifacts/runs/failed-primary",
        ],
    )

    assert result.exit_code == 1
    assert "mutually exclusive" in result.output
    assert not called


def test_amend_cli_recovery_requires_post_outcome_declaration(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    called = False

    def fake_create(**_kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(workflows, "create_preregistration_amendment", fake_create)
    result = CliRunner().invoke(
        app,
        [
            *_amend_arguments(tmp_path),
            "--outcomes-not-inspected",
            "--primary-recovery-authorization-json",
            "recovery.json",
        ],
    )

    assert result.exit_code == 1
    assert "requires --outcomes-inspected" in result.output
    assert not called


def test_amend_cli_recovery_json_must_be_an_object(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "recovery.json").write_text("[]", encoding="utf-8")
    called = False

    def fake_create(**_kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(workflows, "create_preregistration_amendment", fake_create)
    result = CliRunner().invoke(
        app,
        [
            *_amend_arguments(project),
            "--outcomes-inspected",
            "--outcomes-inspected-at-utc",
            "2026-07-27T10:57:07Z",
            "--primary-recovery-authorization-json",
            "recovery.json",
        ],
    )

    assert result.exit_code == 1
    assert "must contain one JSON object" in result.output
    assert not called


def test_primary_orphan_recovery_cli_maps_one_authorized_executor_call(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    authority = (project / "authorities/recovery").resolve()
    gate = SimpleNamespace(
        freeze_artifact_root_sha256="a" * 64,
        freeze_manifest_sha256="b" * 64,
    )
    plan = object()
    controls = object()
    typed_authorization = SimpleNamespace(source_run_id="interrupted-primary")
    captured: dict[str, Any] = {}

    class FakeRecoveryAuthorization:
        @classmethod
        def from_mapping(cls, value: Any, **kwargs: Any) -> Any:
            captured["authorization_mapping"] = value
            captured["authorization_outer_binding"] = kwargs
            return typed_authorization

    def fake_gate(**kwargs: Any) -> Any:
        captured["gate"] = kwargs
        return gate

    def fake_executor(**kwargs: Any) -> dict[str, Any]:
        captured["executor"] = kwargs
        return {
            "status": "completed",
            "completion_stage": "PRIMARY_STUDY_COMPLETE",
            "run_id": "recovery-run",
        }

    monkeypatch.setattr(workflows, "validate_primary_execution_gate", fake_gate)
    monkeypatch.setattr(
        workflows,
        "require_primary_recovery_authorization",
        lambda path: {"authority": str(path)},
    )
    monkeypatch.setattr(experiment, "RecoveryAuthorization", FakeRecoveryAuthorization)
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda path: {"config_path": str(path)},
    )
    monkeypatch.setattr(
        experiment,
        "build_primary_matrix_plan",
        lambda payload: plan,
    )
    monkeypatch.setattr(
        experiment,
        "primary_execution_controls_from_frozen_config",
        lambda payload: controls,
    )
    monkeypatch.setattr(
        recovery_runner,
        "execute_primary_orphan_recovery",
        fake_executor,
    )

    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "primary-orphan-recovery",
            "--project-root",
            str(project),
            "--authority-dir",
            "authorities/recovery",
            "--runs-root",
            "artifacts/runs",
            "--run-id",
            "recovery-run",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["workflow"] == "primary_orphan_recovery"
    assert payload["status"] == "executor_returned"
    assert payload["automatic_retry_allowed"] is False
    assert payload["training_invoked"] is False
    assert captured["authorization_mapping"] == {"authority": str(authority)}
    assert captured["authorization_outer_binding"] == {
        "authority_directory": authority,
        "authority_artifact_root_sha256": "a" * 64,
        "authority_manifest_sha256": "b" * 64,
    }
    assert captured["gate"]["freeze_directory"] == authority
    assert captured["gate"]["experiment_name"] == (workflows.PRIMARY_RECOVERY_EXPERIMENT_NAME)
    assert captured["gate"]["frozen_primary_config_path"] == (authority / "primary_frozen.yaml")
    assert captured["gate"]["frozen_confirmatory_config_path"] == (
        authority / "confirmatory_frozen.yaml"
    )
    assert captured["executor"] == {
        "gate_evidence": gate,
        "plan": plan,
        "controls": controls,
        "authorization": typed_authorization,
        "project_root": project.resolve(),
        "runs_root": (project / "artifacts/runs").resolve(),
        "run_id": "recovery-run",
    }


def test_primary_orphan_recovery_cli_gate_failure_never_loads_executor(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    called = False

    def fail_gate(**_kwargs: Any) -> Any:
        raise ValueError("recovery authority rejected")

    def fake_executor(**_kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(workflows, "validate_primary_execution_gate", fail_gate)
    monkeypatch.setattr(
        recovery_runner,
        "execute_primary_orphan_recovery",
        fake_executor,
    )

    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "primary-orphan-recovery",
            "--project-root",
            str(tmp_path),
            "--authority-dir",
            "authorities/recovery",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["status"] == "gated"
    assert payload["completion_stage"] is None
    assert payload["automatic_retry_allowed"] is False
    assert "recovery authority rejected" in payload["error"]
    assert not called


def test_primary_orphan_recovery_cli_preflight_forbids_tracker_copy_and_execute(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    gate = SimpleNamespace(
        freeze_artifact_root_sha256="a" * 64,
        freeze_manifest_sha256="b" * 64,
    )
    plan = object()
    controls = object()
    typed_authorization = SimpleNamespace(
        source_run_id="interrupted-primary",
        canonical_sha256="c" * 64,
    )

    class FakeRecoveryAuthorization:
        @classmethod
        def from_mapping(cls, _value: Any, **_kwargs: Any) -> Any:
            return typed_authorization

    snapshot = SimpleNamespace(
        snapshot_root_sha256="d" * 64,
        completed_required_cell_count=185,
        skipped_optional_cell_count=37,
        artifacts=(object(), object()),
        total_bytes=1234,
    )
    inspection = SimpleNamespace(snapshot=snapshot)
    disk = SimpleNamespace(
        as_dict=lambda: {
            "copy_bytes": 1234,
            "observed_free_bytes": 100_000,
        }
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("preflight touched a mutating recovery boundary")

    monkeypatch.setattr(
        workflows,
        "validate_primary_execution_gate",
        lambda **_kwargs: gate,
    )
    monkeypatch.setattr(
        workflows,
        "require_primary_recovery_authorization",
        lambda _path: {"authority": "recovery"},
    )
    monkeypatch.setattr(experiment, "RecoveryAuthorization", FakeRecoveryAuthorization)
    monkeypatch.setattr(config_module, "load_config", lambda _path: {"primary": "frozen"})
    monkeypatch.setattr(experiment, "build_primary_matrix_plan", lambda _payload: plan)
    monkeypatch.setattr(
        experiment,
        "primary_execution_controls_from_frozen_config",
        lambda _payload: controls,
    )
    monkeypatch.setattr(
        recovery_runner,
        "_prepare_recovery",
        lambda **_kwargs: (inspection, disk, object()),
    )
    monkeypatch.setattr(recovery_runner.RunTracker, "start", forbidden)
    monkeypatch.setattr(
        recovery_runner,
        "copy_authorized_orphan_artifacts",
        forbidden,
    )
    monkeypatch.setattr(
        recovery_runner,
        "execute_primary_orphan_recovery",
        forbidden,
    )

    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "primary-orphan-recovery",
            "--project-root",
            str(project),
            "--authority-dir",
            "authorities/recovery",
            "--runs-root",
            "artifacts/runs",
            "--preflight-only",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "preflight_passed"
    assert payload["preflight_only"] is True
    assert payload["completion_stage"] is None
    assert payload["study_outcome_eligible"] is False
    assert payload["run_tracker_created"] is False
    assert payload["copy_invoked"] is False
    assert payload["result"]["run_tracker_created"] is False
    assert payload["result"]["copy_invoked"] is False
