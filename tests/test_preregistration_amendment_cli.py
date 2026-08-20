"""Focused CLI contracts for immutable preregistration amendments."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

import histo_audit.workflows as workflows
from histo_audit.cli import app
from histo_audit.workflows import (
    PreregistrationAmendmentResult,
    PreregistrationAmendmentVerification,
)
from tests.cli_contracts import cli_options, resolve_cli_command


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
        "Clarify the frozen primary estimand.",
        "--affected-hypothesis",
        "H1",
        "--affected-hypothesis",
        "H2",
        "--affected-analysis",
        "primary_ranking",
        "--affected-analysis",
        "sensitivity_macro_f1",
        "--amendment-timestamp-utc",
        "2026-07-18T22:00:00Z",
        "--amendment-root",
        "artifacts/amendments",
    ]


def _fake_result(project: Path) -> PreregistrationAmendmentResult:
    destination = (project / "artifacts/amendments/20260718T220000.000000Z").resolve()
    return PreregistrationAmendmentResult(
        amendment_directory=destination,
        parent_authority_directory=(project / "authorities/base").resolve(),
        amendment_timestamp_utc="2026-07-18T22:00:00.000000Z",
        chain_depth=1,
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


def _valid_verification(result: PreregistrationAmendmentResult) -> Any:
    return PreregistrationAmendmentVerification(
        valid=True,
        amendment_directory=result.amendment_directory,
        chain_depth=result.chain_depth,
        artifact_root_sha256=result.artifact_root_sha256,
        sha256_manifest_sha256=result.sha256_manifest_sha256,
        parent_authority_directory=result.parent_authority_directory,
        errors=(),
    )


def test_amendment_cli_help_exposes_every_explicit_authority_input() -> None:
    options = cli_options(app, ("preregistration", "amend"))
    assert {
        "--parent-authority-dir",
        "--amended-preregistration",
        "--amended-primary-config",
        "--amended-confirmatory-config",
        "--reason",
        "--affected-hypothesis",
        "--affected-analysis",
        "--amendment-timestamp-utc",
        "--amendment-root",
        "--outcomes-inspected",
        "--outcomes-not-inspected",
        "--outcomes-inspected-at-utc",
    }.issubset(options)


def test_amend_cli_maps_explicit_pre_outcome_inputs_and_verifies_publication(
    tmp_path: Path, monkeypatch: Any
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    expected = _fake_result(project)
    captured: dict[str, Any] = {}
    verified: list[Path] = []

    def fake_create(**kwargs: Any) -> PreregistrationAmendmentResult:
        captured.update(kwargs)
        return expected

    def fake_verify(path: Path, **_kwargs: Any) -> Any:
        verified.append(path)
        return _valid_verification(expected)

    monkeypatch.setattr(workflows, "create_preregistration_amendment", fake_create)
    monkeypatch.setattr(workflows, "verify_preregistration_amendment", fake_verify)

    result = CliRunner().invoke(
        app,
        [*_amend_arguments(project), "--outcomes-not-inspected"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["authority_status"] == "amended"
    assert payload["integrity_verified"] is True
    assert captured == {
        "project_root": project.resolve(),
        "parent_authority_directory": (project / "authorities/base").resolve(),
        "amendment_root": (project / "artifacts/amendments").resolve(),
        "preregistration_path": (project / "PRE_REGISTRATION_NEXT.md").resolve(),
        "primary_config_path": (project / "configs/primary_next.yaml").resolve(),
        "confirmatory_config_path": (project / "configs/confirmatory_next.yaml").resolve(),
        "reason": "Clarify the frozen primary estimand.",
        "affected_hypotheses": ["H1", "H2"],
        "affected_analyses": ["primary_ranking", "sensitivity_macro_f1"],
        "outcomes_inspected": False,
        "outcomes_inspected_at": None,
        "timestamp": datetime(2026, 7, 18, 22, 0, tzinfo=UTC),
    }
    assert verified == [expected.amendment_directory]


def test_amend_cli_requires_exactly_one_outcome_declaration(
    tmp_path: Path, monkeypatch: Any
) -> None:
    called = False

    def fake_create(**_kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(workflows, "create_preregistration_amendment", fake_create)
    runner = CliRunner()

    omitted = runner.invoke(app, _amend_arguments(tmp_path))
    contradictory = runner.invoke(
        app,
        [
            *_amend_arguments(tmp_path),
            "--outcomes-inspected",
            "--outcomes-not-inspected",
        ],
    )

    assert omitted.exit_code == 1
    assert contradictory.exit_code == 1
    assert "declare exactly one" in omitted.output
    assert "declare exactly one" in contradictory.output
    assert not called


def test_amend_cli_maps_post_outcome_timestamp(tmp_path: Path, monkeypatch: Any) -> None:
    expected = _fake_result(tmp_path)
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
            *_amend_arguments(tmp_path),
            "--outcomes-inspected",
            "--outcomes-inspected-at-utc",
            "2026-07-18T21:30:00Z",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["outcomes_inspected"] is True
    assert captured["outcomes_inspected_at"] == datetime(2026, 7, 18, 21, 30, tzinfo=UTC)


def test_amend_cli_rejects_non_utc_timestamp_before_publication(
    tmp_path: Path, monkeypatch: Any
) -> None:
    called = False

    def fake_create(**_kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(workflows, "create_preregistration_amendment", fake_create)
    arguments = _amend_arguments(tmp_path)
    arguments[arguments.index("2026-07-18T22:00:00Z")] = "2026-07-18T22:00:00"

    result = CliRunner().invoke(app, [*arguments, "--outcomes-not-inspected"])

    assert result.exit_code == 1
    assert "ending in Z" in result.output
    assert not called


def test_amend_cli_surfaces_no_overwrite_failure(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        workflows,
        "create_preregistration_amendment",
        lambda **_kwargs: (_ for _ in ()).throw(FileExistsError("authority already exists")),
    )

    result = CliRunner().invoke(
        app,
        [*_amend_arguments(tmp_path), "--outcomes-not-inspected"],
    )

    assert result.exit_code == 1
    assert "FileExistsError: authority already exists" in result.output


def test_verify_amendment_cli_reports_recursive_identity(tmp_path: Path, monkeypatch: Any) -> None:
    expected = _fake_result(tmp_path)
    captured: dict[str, Any] = {}

    def fake_verify(path: Path, *, max_chain_depth: int) -> Any:
        captured.update(path=path, max_chain_depth=max_chain_depth)
        return _valid_verification(expected)

    monkeypatch.setattr(workflows, "verify_preregistration_amendment", fake_verify)
    result = CliRunner().invoke(
        app,
        [
            "preregistration",
            "verify-amendment",
            "--project-root",
            str(tmp_path),
            "--amendment-dir",
            "artifacts/amendments/authority",
            "--max-chain-depth",
            "12",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["authority_status"] == "verified_amendment"
    assert payload["chain_depth"] == 1
    assert payload["integrity_verified"] is True
    assert captured == {
        "path": (tmp_path / "artifacts/amendments/authority").resolve(),
        "max_chain_depth": 12,
    }


def test_verify_amendment_cli_fails_closed_on_invalid_chain(
    tmp_path: Path, monkeypatch: Any
) -> None:
    directory = (tmp_path / "broken").resolve()
    monkeypatch.setattr(
        workflows,
        "verify_preregistration_amendment",
        lambda _path, **_kwargs: PreregistrationAmendmentVerification(
            valid=False,
            amendment_directory=directory,
            chain_depth=None,
            artifact_root_sha256=None,
            sha256_manifest_sha256=None,
            parent_authority_directory=None,
            errors=("cycle detected",),
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "preregistration",
            "verify-amendment",
            "--project-root",
            str(tmp_path),
            "--amendment-dir",
            "broken",
        ],
    )

    assert result.exit_code == 1
    assert "cycle detected" in result.output


def test_verify_resource_technical_successor_cli_help_exposes_required_pins() -> None:
    path = ("preregistration", "verify-resource-technical-successor")
    command = resolve_cli_command(app, path)
    options = cli_options(app, path)
    assert "read-only fresh-process boundary" in (command.help or "")
    assert {"--successor-dir", "--verification-nonce", "--project-root"}.issubset(options)
    descriptions = "\n".join(option.help or "" for option in set(options.values()))
    for description in (
        "Exact superseded",
        "Externally pinned",
        "Pre-mutation canonical",
        "PID of the publication",
        "One-use 64-hex",
    ):
        assert description in descriptions
    assert sum(option.required for option in set(options.values())) == 8


def test_verify_resource_technical_successor_cli_maps_exact_external_pins(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    successor = (project / "artifacts/amendments/successor-d").resolve()
    parent = (project / "artifacts/amendments/authority-c").resolve()
    expected = workflows.ResourceBoundedTechnicalSuccessorVerification(
        successor_directory=successor,
        parent_authority_directory=parent,
        chain_depth=4,
        artifact_root_sha256="a" * 64,
        sha256_manifest_sha256="b" * 64,
        authorization_sha256="c" * 64,
        intent_sha256="d" * 64,
        flat_file_inventory_sha256="e" * 64,
        confirmatory_storage_policy_sha256="f" * 64,
        flat_file_count=8,
        manifest_artifact_count=6,
        controller_process_id=4242,
        verifier_process_id=4243,
        verifier_parent_process_id=4242,
        verification_nonce="1" * 64,
    )
    captured: dict[str, Any] = {}

    def fake_verify(path: Path, **kwargs: Any) -> Any:
        captured.update(successor_directory=path, **kwargs)
        return expected

    monkeypatch.setattr(
        workflows,
        "verify_resource_bounded_technical_successor",
        fake_verify,
    )
    result = CliRunner().invoke(
        app,
        [
            "preregistration",
            "verify-resource-technical-successor",
            "--project-root",
            str(project),
            "--successor-dir",
            "artifacts/amendments/successor-d",
            "--expected-parent-authority-dir",
            "artifacts/amendments/authority-c",
            "--expected-artifact-root-sha256",
            "a" * 64,
            "--expected-sha256-manifest-sha256",
            "b" * 64,
            "--expected-authorization-sha256",
            "c" * 64,
            "--expected-intent-sha256",
            "d" * 64,
            "--expected-controller-pid",
            "4242",
            "--verification-nonce",
            "1" * 64,
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == expected.as_dict()
    assert captured == {
        "successor_directory": successor,
        "expected_parent_authority_directory": parent,
        "expected_artifact_root_sha256": "a" * 64,
        "expected_sha256_manifest_sha256": "b" * 64,
        "expected_authorization_sha256": "c" * 64,
        "expected_intent_sha256": "d" * 64,
        "expected_controller_process_id": 4242,
        "verification_nonce": "1" * 64,
    }


def test_verify_resource_technical_successor_cli_fails_closed_on_verifier_error(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        workflows,
        "verify_resource_bounded_technical_successor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("external publication intent mismatch")
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "preregistration",
            "verify-resource-technical-successor",
            "--project-root",
            str(tmp_path),
            "--successor-dir",
            "successor-d",
            "--expected-parent-authority-dir",
            "authority-c",
            "--expected-artifact-root-sha256",
            "a" * 64,
            "--expected-sha256-manifest-sha256",
            "b" * 64,
            "--expected-authorization-sha256",
            "c" * 64,
            "--expected-intent-sha256",
            "d" * 64,
            "--expected-controller-pid",
            "4242",
            "--verification-nonce",
            "1" * 64,
        ],
    )

    assert result.exit_code == 1
    assert (
        "resource technical successor verification failed: "
        "ValueError: external publication intent mismatch"
    ) in result.output
    assert '"status": "verified"' not in result.output
