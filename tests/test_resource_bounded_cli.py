"""CLI isolation tests for the non-claiming resource-bounded sensitivity."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import typer
from typer.testing import CliRunner

import histo_audit.cli as cli_module
from histo_audit.cli import app
from histo_audit.experiment.resource_bounded_runner import (
    execute_resource_bounded_sensitivity,
    preflight_resource_bounded_sensitivity,
)


def _safe_result(**extra: Any) -> dict[str, Any]:
    return {
        "status": "completed",
        "completion_stage": None,
        "study_outcome_eligible": False,
        "analysis_disposition": "amended_or_exploratory",
        **extra,
    }


def test_resource_bounded_help_exposes_exact_inputs_without_retry_option() -> None:
    result = CliRunner().invoke(
        app,
        ["experiment", "resource-bounded-sensitivity", "--help"],
        terminal_width=240,
    )

    assert result.exit_code == 0, result.output
    root = typer.main.get_command(app)
    experiment = root.commands["experiment"]
    command = experiment.commands["resource-bounded-sensitivity"]
    registered_options = {
        option
        for parameter in command.params
        if hasattr(parameter, "opts")
        for option in parameter.opts
    }
    expected_options = {
        "--project-root",
        "--primary-run-dir",
        "--resource-authority-dir",
        "--lifecycle-readiness-run-dir",
        "--dataset",
        "--manifest",
        "--duplicate-audit",
        "--pathology-encoder-audit",
        "--runs-root",
        "--run-id",
        "--checkpoint-predecessor-run-dir",
        "--preflight-only",
    }
    assert expected_options <= registered_options
    assert "--retry-of-run-id" not in registered_options


def test_resource_bounded_fresh_cli_forwards_only_explicit_inputs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    events: list[str] = []
    captured: dict[str, Any] = {}

    def _load(module_name: str, function_name: str) -> Any:
        assert module_name == "histo_audit.experiment.resource_bounded_runner"
        assert function_name == "execute_resource_bounded_sensitivity"
        events.append("executor_load")

        def _execute(**kwargs: Any) -> dict[str, Any]:
            events.append("execute")
            captured.update(kwargs)
            return _safe_result(
                run_id="new-resource-run",
                run_directory=str(tmp_path / "runs" / "new-resource-run"),
            )

        return _execute

    monkeypatch.setattr(cli_module, "_load_optional_study_executor", _load)
    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "resource-bounded-sensitivity",
            "--project-root",
            str(tmp_path),
            "--primary-run-dir",
            "primary",
            "--resource-authority-dir",
            "authority-c",
            "--lifecycle-readiness-run-dir",
            "readiness",
            "--dataset",
            "dataset",
            "--manifest",
            "manifest.parquet",
            "--duplicate-audit",
            "duplicates.json",
            "--pathology-encoder-audit",
            "pathology.json",
            "--runs-root",
            "runs",
            "--run-id",
            "new-resource-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert events == ["executor_load", "execute"]
    inspect.signature(execute_resource_bounded_sensitivity).bind(**captured)
    assert captured["run_mode"] == "fresh"
    assert captured["retry_of_run_id"] is None
    assert captured["checkpoint_predecessor_run_directory"] is None
    assert captured["run_id"] == "new-resource-run"
    assert captured["project_root"] == tmp_path.resolve()
    assert captured["primary_run_directory"] == (tmp_path / "primary").resolve()
    assert captured["resource_authority_directory"] == (tmp_path / "authority-c").resolve()
    assert captured["lifecycle_readiness_run_directory"] == (tmp_path / "readiness").resolve()
    assert captured["runs_root"] == (tmp_path / "runs").resolve()
    payload = json.loads(result.output)
    assert payload["completion_stage"] is None
    assert payload["study_outcome_eligible"] is False
    assert payload["analysis_disposition"] == "amended_or_exploratory"
    assert payload["automatic_retry_allowed"] is False
    assert payload["evidence"]["run_id"] == "new-resource-run"
    assert "CONFIRMATORY_COMPLETE" not in result.output


def test_resource_bounded_preflight_uses_dedicated_noncreating_api(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    events: list[str] = []
    captured: dict[str, Any] = {}

    def _load(module_name: str, function_name: str) -> Any:
        assert module_name == "histo_audit.experiment.resource_bounded_runner"
        assert function_name == "preflight_resource_bounded_sensitivity"
        events.append("preflight_load")

        def _preflight(**kwargs: Any) -> dict[str, Any]:
            events.append("preflight")
            captured.update(kwargs)
            return _safe_result(
                capacity_checks=2,
                compute_checks=[{"phase": "before_data", "passed": True}],
                resource_compute_evidence_sha256="d" * 64,
                checkpoint_allowlist_count=30,
                reusable_checkpoint_count=0,
                missing_checkpoint_count=30,
                full_pc_gate_validation_count=1,
            )

        return _preflight

    monkeypatch.setattr(cli_module, "_load_optional_study_executor", _load)
    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "resource-bounded-sensitivity",
            "--project-root",
            str(tmp_path),
            "--preflight-only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert events == ["preflight_load", "preflight"]
    inspect.signature(preflight_resource_bounded_sensitivity).bind(**captured)
    assert captured["run_mode"] == "fresh"
    payload = json.loads(result.output)
    assert payload["status"] == "preflight_passed"
    assert payload["preflight_only"] is True
    assert payload["completion_stage"] is None
    assert payload["study_outcome_eligible"] is False
    assert payload["analysis_disposition"] == "amended_or_exploratory"
    assert payload["evidence"]["checkpoint_allowlist_count"] == 30
    assert payload["evidence"]["resource_compute_evidence_sha256"] == "d" * 64
    assert payload["evidence"]["compute_checks"][0]["passed"] is True
    assert payload["evidence"]["full_pc_gate_validation_count"] == 1
    assert not (tmp_path / "artifacts" / "runs").exists()


def test_resource_bounded_runner_gate_error_is_not_retried_or_reclassified(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    load_count = 0
    execute_count = 0

    def _load(*_: Any) -> Any:
        nonlocal load_count
        load_count += 1

        def _reject_gate(**kwargs: Any) -> None:
            nonlocal execute_count
            execute_count += 1
            inspect.signature(preflight_resource_bounded_sensitivity).bind(**kwargs)
            raise ValueError("authority C mismatch")

        return _reject_gate

    monkeypatch.setattr(cli_module, "_load_optional_study_executor", _load)
    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "resource-bounded-sensitivity",
            "--project-root",
            str(tmp_path),
            "--preflight-only",
        ],
    )

    assert result.exit_code == 1, result.output
    assert load_count == 1
    assert execute_count == 1
    payload = json.loads(result.output)
    assert payload["status"] == "preflight_failed"
    assert payload["completion_stage"] is None
    assert payload["study_outcome_eligible"] is False
    assert payload["analysis_disposition"] == "amended_or_exploratory"
    assert not (tmp_path / "artifacts" / "runs").exists()


def test_resource_bounded_successor_derives_exact_retry_basename(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}
    predecessor = tmp_path / "runs" / "failed-resource-run"

    def _load(*_: Any) -> Any:
        def _execute(**kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return _safe_result()

        return _execute

    monkeypatch.setattr(cli_module, "_load_optional_study_executor", _load)
    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "resource-bounded-sensitivity",
            "--project-root",
            str(tmp_path),
            "--checkpoint-predecessor-run-dir",
            str(predecessor),
            "--run-id",
            "successor-resource-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["run_mode"] == "successor_resume"
    assert captured["checkpoint_predecessor_run_directory"] == predecessor.resolve()
    assert captured["retry_of_run_id"] == "failed-resource-run"
    payload = json.loads(result.output)
    assert payload["retry_of_run_id"] == "failed-resource-run"
    assert payload["automatic_retry_allowed"] is False


def test_resource_bounded_successor_rejects_reused_run_id_before_any_boundary(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    boundary_called = False

    def _unexpected(**_: Any) -> None:
        nonlocal boundary_called
        boundary_called = True
        raise AssertionError("invalid successor identity must fail before a boundary")

    monkeypatch.setattr(cli_module, "_load_optional_study_executor", _unexpected)
    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "resource-bounded-sensitivity",
            "--project-root",
            str(tmp_path),
            "--checkpoint-predecessor-run-dir",
            "runs/predecessor",
            "--run-id",
            "predecessor",
        ],
    )

    assert result.exit_code == 2, result.output
    assert not boundary_called
    payload = json.loads(result.output)
    assert payload["status"] == "invalid_run_identity"
    assert payload["retry_of_run_id"] == "predecessor"
    assert payload["automatic_retry_allowed"] is False


def test_resource_bounded_executor_failure_is_not_retried(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    load_count = 0
    execute_count = 0

    def _load(*_: Any) -> Any:
        nonlocal load_count
        load_count += 1

        def _execute(**_: Any) -> None:
            nonlocal execute_count
            execute_count += 1
            raise RuntimeError(
                "deliberate single-attempt failure: "
                "CONFIRMATORY_COMPLETE study_outcome_eligible=true"
            )

        return _execute

    monkeypatch.setattr(cli_module, "_load_optional_study_executor", _load)
    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "resource-bounded-sensitivity",
            "--project-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1, result.output
    assert load_count == 1
    assert execute_count == 1
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert payload["automatic_retry_allowed"] is False
    assert payload["completion_stage"] is None
    assert payload["study_outcome_eligible"] is False
    assert "CONFIRMATORY_COMPLETE" not in result.output
    assert "study_outcome_eligible=true" not in result.output
