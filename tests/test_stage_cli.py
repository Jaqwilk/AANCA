"""Focused fail-closed tests for post-pilot workflow CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cli_contracts import cli_options
from typer.testing import CliRunner

import histo_audit.cli as cli_module
import histo_audit.external_validation as external_validation
import histo_audit.workflows as workflows
from histo_audit.cli import app
from histo_audit.external_validation import ReviewPackageValidationResult


def test_stage_command_help_exposes_required_evidence_options() -> None:
    freeze_options = cli_options(app, ("preregistration", "freeze"))
    audit_options = cli_options(app, ("audit", "original"))
    external_options = cli_options(app, ("external", "build-review-package"))
    representation_options = cli_options(app, ("representations", "extract"))

    assert {
        "--pilot-run-dir",
        "--dataset",
        "--manifest",
        "--raw-checksum-manifest",
        "--duplicate-audit",
        "--pathology-encoder-audit",
        "--pilot-dev-manifest",
        "--pilot-gate-certificate",
        "--preregistration",
        "--primary-config",
        "--confirmatory-config",
        "--freeze-root",
    }.issubset(freeze_options)
    assert {
        "--manifest",
        "--feature-cache",
        "--final-reference-groups",
        "--output-dir",
        "--class-order",
        "--n-splits",
    }.issubset(audit_options)
    assert {
        "--manifest",
        "--ranking",
        "--output-dir",
        "--private-key",
        "--top",
        "--random",
        "--seed",
    }.issubset(external_options)
    assert {
        "--independence-output",
        "--primary-config",
        "--include-context-embeddings",
    }.issubset(representation_options)


def test_study_command_help_exposes_all_immutable_gate_inputs() -> None:
    primary_options = cli_options(app, ("experiment", "primary"))
    required_options = {
        "--project-root",
        "--freeze-dir",
        "--dataset",
        "--manifest",
        "--duplicate-audit",
        "--pathology-encoder-audit",
        "--primary-config",
        "--confirmatory-config",
    }
    assert required_options.issubset(primary_options)


def test_primary_cli_validates_gate_before_loading_executor_or_creating_run(
    tmp_path: Path, monkeypatch: Any
) -> None:
    captured: dict[str, Any] = {}
    executor_loader_called = False

    def _reject_gate(**kwargs: Any) -> None:
        captured.update(kwargs)
        raise FileNotFoundError("deliberately missing frozen evidence")

    def _unexpected_executor_loader(*args: Any, **kwargs: Any) -> None:
        nonlocal executor_loader_called
        executor_loader_called = True
        raise AssertionError("executor must not load before the gate passes")

    monkeypatch.setattr(workflows, "validate_primary_execution_gate", _reject_gate)
    monkeypatch.setattr(cli_module, "_load_optional_study_executor", _unexpected_executor_loader)

    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "primary",
            "--project-root",
            str(tmp_path),
            "--freeze-dir",
            "freeze",
            "--dataset",
            "dataset",
            "--manifest",
            "manifest.parquet",
            "--duplicate-audit",
            "duplicates.json",
            "--pathology-encoder-audit",
            "pathology.json",
            "--primary-config",
            "primary.yaml",
            "--confirmatory-config",
            "confirmatory.yaml",
        ],
    )

    assert result.exit_code == 2, result.output
    assert "GATED [PRIMARY_STUDY_LOCKED]" in result.output
    assert "deliberately missing frozen evidence" in result.output
    assert "Next: python -m histo_audit preregistration freeze --help" in result.output
    assert "PRIMARY_STUDY_COMPLETE" not in result.output
    assert not executor_loader_called
    assert captured == {
        "project_root": tmp_path.resolve(),
        "freeze_directory": (tmp_path / "freeze").resolve(),
        "dataset_path": (tmp_path / "dataset").resolve(),
        "manifest_path": (tmp_path / "manifest.parquet").resolve(),
        "duplicate_audit_path": (tmp_path / "duplicates.json").resolve(),
        "pathology_encoder_audit_path": (tmp_path / "pathology.json").resolve(),
        "frozen_primary_config_path": (tmp_path / "primary.yaml").resolve(),
        "frozen_confirmatory_config_path": (tmp_path / "confirmatory.yaml").resolve(),
    }
    assert not (tmp_path / "artifacts" / "runs").exists()


def test_passed_study_gate_reports_executor_unavailable_without_stage_claim(
    tmp_path: Path, monkeypatch: Any
) -> None:
    events: list[str] = []

    def _pass_gate(**kwargs: Any) -> object:
        events.append("gate")
        return object()

    def _missing_executor(module_name: str, function_name: str) -> None:
        assert module_name == "histo_audit.experiment.primary_runner"
        assert function_name == "execute_primary_study"
        events.append("executor_lookup")
        return None

    monkeypatch.setattr(workflows, "validate_primary_execution_gate", _pass_gate)
    monkeypatch.setattr(cli_module, "_load_optional_study_executor", _missing_executor)

    result = CliRunner().invoke(
        app,
        ["experiment", "primary", "--project-root", str(tmp_path)],
    )

    assert result.exit_code == 2, result.output
    assert events == ["gate", "executor_lookup"]
    assert "GATED [EXECUTOR_UNAVAILABLE]" in result.output
    assert "no run directory was created" in result.output
    assert "PRIMARY_STUDY_COMPLETE" not in result.output
    assert not (tmp_path / "artifacts" / "runs").exists()


def test_preregistration_freeze_cli_refuses_missing_pilot_evidence(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "preregistration",
            "freeze",
            "--project-root",
            str(tmp_path),
            "--pilot-run-dir",
            "missing-pilot",
            "--dataset",
            "missing-dataset",
            "--manifest",
            "missing-manifest.parquet",
            "--raw-checksum-manifest",
            "missing-checksums.csv",
            "--duplicate-audit",
            "missing-duplicates.json",
            "--pathology-encoder-audit",
            "missing-pathology-audit.json",
            "--pilot-development-manifest",
            "missing-pilot-development.parquet",
            "--pilot-gate-certificate",
            "missing-pilot-gate.json",
        ],
    )

    assert result.exit_code == 1
    assert "ERROR: preregistration freeze failed" in result.output
    assert "PRE_REGISTRATION_FROZEN" not in result.output
    assert not (tmp_path / "configs" / "primary_frozen.yaml").exists()
    assert not (tmp_path / "configs" / "confirmatory_frozen.yaml").exists()


def test_original_audit_cli_rejects_incomplete_feature_cache(tmp_path: Path) -> None:
    feature_cache = tmp_path / "features.npz"
    np.savez(feature_cache, features=np.ones((2, 2), dtype=np.float64))
    final_groups = tmp_path / "final-groups.txt"
    final_groups.write_text("final-patch-001\n", encoding="utf-8")
    output = tmp_path / "original-audit"

    result = CliRunner().invoke(
        app,
        [
            "audit",
            "original",
            "--project-root",
            str(tmp_path),
            "--manifest",
            "manifest.csv",
            "--feature-cache",
            feature_cache.name,
            "--final-reference-groups",
            final_groups.name,
            "--output-dir",
            output.name,
        ],
    )

    assert result.exit_code == 1
    assert (
        "feature cache must contain sample_ids and exactly one of features, embeddings, or values"
        in result.output
    )
    assert not output.exists()


def test_review_package_cli_fails_when_structural_validation_fails(
    tmp_path: Path, monkeypatch: Any
) -> None:
    asset = tmp_path / "target.png"
    asset.write_bytes(b"test display asset")
    manifest = tmp_path / "manifest.csv"
    ranking = tmp_path / "ranking.csv"
    sample_ids = [f"sample-{index}" for index in range(4)]
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "observed_label": [0, 1, 0, 1],
            "full_patch_path": [str(asset)] * 4,
            "target_crop_path": [str(asset)] * 4,
            "target_contour_path": [str(asset)] * 4,
        }
    ).to_csv(manifest, index=False)
    pd.DataFrame({"sample_id": sample_ids, "risk_score": [0.9, 0.7, 0.4, 0.1]}).to_csv(
        ranking, index=False
    )
    validation_called = False

    def _invalid_validation(
        package_directory: str | Path,
        *,
        private_unblinding_key_path: str | Path | None = None,
    ) -> ReviewPackageValidationResult:
        nonlocal validation_called
        validation_called = True
        assert private_unblinding_key_path is not None
        return ReviewPackageValidationResult(
            valid=False,
            package_directory=Path(package_directory),
            item_count=2,
            asset_count=6,
            private_linkage_validated=False,
            errors=("forced structural validation failure",),
            warnings=(),
        )

    monkeypatch.setattr(
        external_validation,
        "validate_blinded_review_package",
        _invalid_validation,
    )
    package = tmp_path / "review-package"
    private_key = tmp_path / "private" / "unblinding.csv"
    result = CliRunner().invoke(
        app,
        [
            "external",
            "build-review-package",
            "--project-root",
            str(tmp_path),
            "--manifest",
            manifest.name,
            "--ranking",
            ranking.name,
            "--output-dir",
            package.name,
            "--private-key",
            str(private_key.relative_to(tmp_path)),
            "--top",
            "1",
            "--random",
            "1",
            "--seed",
            "17",
        ],
    )

    assert validation_called
    assert result.exit_code == 1
    assert "generated review package failed validation" in result.output
    assert "EXTERNAL_VALIDATION_READY" not in result.output
