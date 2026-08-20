"""Cross-platform CLI boundaries for the public synthetic quick-start."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner

from histo_audit.cli import app
from histo_audit.reporting import smoke_runner


def test_smoke_cli_uses_an_isolated_portable_registry_by_default(
    tmp_path: Path, monkeypatch: Any
) -> None:
    captured: dict[str, Any] = {}

    def fake_execute_synthetic_smoke(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        run = tmp_path / "artifacts" / "smoke_runs" / "synthetic-run"
        return SimpleNamespace(
            success=True,
            status="completed",
            run_id="synthetic-run",
            run_directory=run,
            metrics_path=run / "metrics.json",
            report=SimpleNamespace(
                markdown_path=run / "report.md",
                html_path=run / "report.html",
                figure_paths=(),
            ),
        )

    monkeypatch.setattr(
        smoke_runner,
        "execute_synthetic_smoke",
        fake_execute_synthetic_smoke,
    )
    result = CliRunner().invoke(
        app,
        ["experiment", "smoke", "--project-root", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert captured["project_root"] == tmp_path.resolve()
    assert captured["runs_root"] == (tmp_path / "artifacts" / "smoke_runs").resolve()


def test_smoke_cli_accepts_a_caller_owned_clean_registry(tmp_path: Path, monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_execute_synthetic_smoke(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        run = tmp_path / "portable-runs" / "synthetic-run"
        return SimpleNamespace(
            success=True,
            status="completed",
            run_id="synthetic-run",
            run_directory=run,
            metrics_path=run / "metrics.json",
            report=SimpleNamespace(
                markdown_path=run / "report.md",
                html_path=run / "report.html",
                figure_paths=(),
            ),
        )

    monkeypatch.setattr(
        smoke_runner,
        "execute_synthetic_smoke",
        fake_execute_synthetic_smoke,
    )
    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "smoke",
            "--project-root",
            str(tmp_path),
            "--runs-root",
            "portable-runs",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["runs_root"] == (tmp_path / "portable-runs").resolve()
