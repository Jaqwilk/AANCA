"""Main ``python -m histo_audit`` registration for T0 publication controls."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner

from histo_audit.cli import app
from histo_audit.workflows import (
    original_confirmatory_technical_authority_publication_v1 as publication,
)


def test_main_cli_exposes_original_confirmatory_authority_group() -> None:
    result = CliRunner().invoke(
        app,
        ["original-confirmatory-technical-authority", "--help"],
    )

    assert result.exit_code == 0, result.output
    assert "build-intent" in result.output
    assert "review-intent" in result.output
    assert "publish" in result.output
    assert "verify" in result.output


def test_main_cli_verify_routes_to_combined_read_only_verifier(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    authority = tmp_path / "authority"
    authority.mkdir()
    calls: list[tuple[Path, Path | None, bool]] = []
    combined = SimpleNamespace(
        namespace_claim_sha256="a" * 64,
        review_attempt_claim_sha256="d" * 64,
        as_dict=lambda: {"authority": {"artifact_root_sha256": "b" * 64}},
        lifecycle_binding=lambda: {
            "policy": ("published_original_confirmatory_technical_authority_lifecycle_binding_v1"),
            "namespace_claim_sha256": "a" * 64,
            "review_attempt_claim_sha256": "d" * 64,
            "binding_sha256": "c" * 64,
        },
    )

    def fake_verify(
        directory: str | Path,
        *,
        project_root: str | Path | None = None,
        verify_live: bool = True,
    ) -> Any:
        calls.append(
            (
                Path(directory),
                None if project_root is None else Path(project_root),
                verify_live,
            )
        )
        return combined

    monkeypatch.setattr(
        publication,
        "verify_published_original_confirmatory_technical_authority_v1",
        fake_verify,
    )
    before = tuple(authority.iterdir())
    result = CliRunner().invoke(
        app,
        [
            "original-confirmatory-technical-authority",
            "verify",
            "--authority-directory",
            str(authority),
            "--project-root",
            str(tmp_path),
        ],
    )
    after = tuple(authority.iterdir())

    assert result.exit_code == 0, result.output
    assert calls == [(authority, tmp_path, True)]
    assert before == after == ()
    output = json.loads(result.output)
    assert output["decision"] == "passed"
    assert output["read_only"] is True
    assert output["namespace_claim_sha256"] == "a" * 64
    assert output["review_attempt_claim_sha256"] == "d" * 64
