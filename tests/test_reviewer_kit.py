"""Portable kit integrity, reproducibility and actual isolated execution gates."""

from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
REVIEW = runpy.run_path(str(ROOT / "scripts/review_project.py"))
BUILD = runpy.run_path(str(ROOT / "scripts/build_reviewer_kit.py"))


def _small_kit(tmp_path):
    content = b"original authority"
    (tmp_path / "authority.txt").write_bytes(content)
    manifest = {
        "schema_version": 1,
        "source_revision": "a" * 40,
        "source_tree_dirty": False,
        "files": [
            {
                "path": "authority.txt",
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        ],
    }
    manifest["manifest_root_sha256"] = REVIEW["canonical_digest"](manifest)
    (tmp_path / "reviewer-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_kit_build_is_deterministic_and_runs_outside_checkout(tmp_path):
    first = BUILD["build_kit"](ROOT, tmp_path / "first")
    second = BUILD["build_kit"](ROOT, tmp_path / "second")
    assert first["sha256"] == second["sha256"]
    with zipfile.ZipFile(first["archive"]) as archive:
        archive.extractall(tmp_path / "extracted")
    kit = tmp_path / "extracted/aanca-reviewer-kit-v1"
    result = REVIEW["verify_kit"](kit)
    assert result["file_count"] == first["file_count"]
    completed = subprocess.run(
        [sys.executable, "-I", str(kit / "scripts/review_project.py"), "--json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "passed"
    assert receipt["models_trained"] is False
    assert receipt["numeric_recalculation_requested"] is False
    assert {item["name"] for item in receipt["checks"]} == {"offline_kit", "article_and_sources"}
    guide = (kit / "index.html").read_text(encoding="utf-8")
    assert "__EVIDENCE_ROWS__" not in guide
    assert "not a group-bootstrap confidence interval" in guide
    assert "not a clinical validation" in guide


@pytest.mark.parametrize(
    "defect",
    ["altered", "missing", "extra", "traversal", "absolute", "duplicate", "size_boolean", "root"],
)
def test_kit_rejects_missing_changed_and_unsafe_materials(tmp_path, defect):
    manifest = _small_kit(tmp_path)
    record = manifest["files"][0]
    if defect == "altered":
        (tmp_path / "authority.txt").write_text("changed", encoding="utf-8")
    elif defect == "missing":
        (tmp_path / "authority.txt").unlink()
    elif defect == "extra":
        (tmp_path / "unexpected.txt").write_text("extra", encoding="utf-8")
    elif defect == "traversal":
        record["path"] = "../authority.txt"
    elif defect == "absolute":
        record["path"] = str(tmp_path / "authority.txt")
    elif defect == "duplicate":
        manifest["files"].append(record.copy())
    elif defect == "size_boolean":
        record["size_bytes"] = True
    elif defect == "root":
        manifest["source_revision"] = "b" * 40
    if defect != "root":
        manifest["manifest_root_sha256"] = REVIEW["canonical_digest"](
            {key: value for key, value in manifest.items() if key != "manifest_root_sha256"}
        )
    (tmp_path / "reviewer-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        REVIEW["verify_kit"](tmp_path)


def test_bad_kit_is_rejected_before_any_embedded_script_runs(tmp_path):
    _small_kit(tmp_path)
    (tmp_path / "authority.txt").write_text("tampered", encoding="utf-8")
    result = REVIEW["review_project"](tmp_path, numeric=True, online="https://example.invalid")
    assert result["status"] == "failed"
    assert [check["name"] for check in result["checks"]] == ["offline_kit"]
