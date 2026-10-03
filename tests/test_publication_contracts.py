"""Publication checks must detect content drift and preserve living status dates."""

from __future__ import annotations

import hashlib
import json
import runpy
from datetime import date
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _remote_case(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any], dict[str, bytes]]:
    module = runpy.run_path(str(ROOT / "scripts/verify_deployed_presentation.py"))
    files = {path: path.encode("utf-8") for path in sorted(module["OUTPUT_FILES"])}
    records = [
        {"path": path, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for path, data in files.items()
    ]
    manifest = {
        "schema_version": 7,
        "files": records,
        "manifest_root_sha256": module["_canonical_sha256"](records),
    }
    files["manifest.json"] = json.dumps(manifest).encode("utf-8")

    def fetch(url: str, **kwargs: Any) -> bytes:
        return files[url.removeprefix("https://aancastudy.org/")]

    function = module["verify_remote_package"]
    monkeypatch.setitem(function.__globals__, "_fetch", fetch)
    return function, manifest, files


def test_complete_remote_package_matches_the_local_release(monkeypatch: pytest.MonkeyPatch) -> None:
    verify, manifest, _ = _remote_case(monkeypatch)
    result = verify("https://aancastudy.org/", expected_manifest=manifest)
    assert result["status"] == "valid"
    assert result["remote_package_integrity_valid"] is True
    assert result["matches_local_release"] is True
    assert result["file_count"] == 13


def test_remote_asset_transformation_cannot_pass_the_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verify, manifest, files = _remote_case(monkeypatch)
    files["assets/hero/nuclei/nucleus-compact.png"] += b"transformed"
    result = verify("https://aancastudy.org/", expected_manifest=manifest)
    assert result["status"] == "failed"
    assert result["remote_package_integrity_valid"] is False
    assert sum(not check["valid"] for check in result["files"]) == 1


def test_internally_valid_old_release_cannot_pass_as_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verify, manifest, _ = _remote_case(monkeypatch)
    result = verify("https://aancastudy.org/", expected_manifest={**manifest, "schema_version": 99})
    assert result["status"] == "failed"
    assert result["remote_package_integrity_valid"] is True
    assert result["matches_local_release"] is False


def test_remote_manifest_cannot_request_an_unlisted_path(monkeypatch: pytest.MonkeyPatch) -> None:
    verify, manifest, files = _remote_case(monkeypatch)
    manifest["files"][0]["path"] = "../secrets"
    files["manifest.json"] = json.dumps(manifest).encode("utf-8")
    with pytest.raises(ValueError, match="unexpected file"):
        verify("https://aancastudy.org/")


@pytest.mark.parametrize(
    "value, expected",
    [("1 September 2026", date(2026, 9, 1)), ("2 October 2026", date(2026, 10, 2))],
)
def test_status_date_can_advance_without_changing_the_release(value: str, expected: date) -> None:
    verifier = runpy.run_path(str(ROOT / "scripts/verify_professor_release.py"))[
        "_verify_status_document"
    ]
    text = f"# AANCA status\n\nUpdated: {value}\n\nFinal professor-readiness audit\n"
    assert verifier(text) == expected


@pytest.mark.parametrize(
    "header", ["", "Updated: 32 October 2026", "Updated: 2 October 2026\nUpdated: 1 September 2026"]
)
def test_status_requires_one_real_date(header: str) -> None:
    verifier = runpy.run_path(str(ROOT / "scripts/verify_professor_release.py"))[
        "_verify_status_document"
    ]
    with pytest.raises(ValueError, match="date"):
        verifier(f"# AANCA status\n{header}\nFinal professor-readiness audit\n")
