"""Fresh-process regression coverage for the preregistration CLI import boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_preregistration_freeze_cli_imports_in_a_fresh_process(tmp_path: Path) -> None:
    """The real callback must reach fail-closed input validation, not a circular import."""

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "histo_audit",
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
            "--pilot-dev-manifest",
            "missing-pilot-development.parquet",
            "--pilot-gate-certificate",
            "missing-pilot-gate.json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "ERROR: preregistration freeze failed" in output
    assert "partially initialized module" not in output
    assert "circular import" not in output.casefold()
