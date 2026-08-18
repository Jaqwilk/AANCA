"""Contract tests for the execution/governance identity boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from histo_audit.utils.run_tracking import (
    SOURCE_GOVERNANCE_FILENAMES,
    capture_governance_tree,
    capture_source_tree,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.mark.parametrize(
    ("relative_path", "initial", "changed"),
    [
        ("src/histo_audit/module.py", "VALUE = 1\n", "VALUE = 2\n"),
        ("configs/primary.yaml", "seed: 1\n", "seed: 2\n"),
        ("pyproject.toml", "[project]\nname='audit'\n", "[project]\nname='audit-v2'\n"),
        ("uv.lock", "version = 1\n", "version = 2\n"),
    ],
)
def test_execution_identity_changes_for_every_execution_scope(
    tmp_path: Path,
    relative_path: str,
    initial: str,
    changed: str,
) -> None:
    target = tmp_path / relative_path
    _write(target, initial)
    before = capture_source_tree(tmp_path)

    target.write_text(changed, encoding="utf-8")
    after = capture_source_tree(tmp_path)

    assert before["root_sha256"] != after["root_sha256"]


def test_execution_and_governance_manifests_have_exact_disjoint_scopes(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "histo_audit" / "module.py", "VALUE = 1\n")
    _write(tmp_path / "configs" / "primary.yaml", "seed: 1\n")
    _write(tmp_path / "pyproject.toml", "[project]\nname='audit'\n")
    _write(tmp_path / "uv.lock", "version = 1\n")
    for filename in SOURCE_GOVERNANCE_FILENAMES:
        _write(tmp_path / filename, f"governance: {filename}\n")

    execution = capture_source_tree(tmp_path)
    governance = capture_governance_tree(tmp_path)
    execution_paths = {record["path"] for record in execution["artifacts"]}
    governance_paths = {record["path"] for record in governance["artifacts"]}

    assert execution["schema_version"] == 3
    assert execution["scope_kind"] == "execution_source"
    assert execution["scope"] == ["src/**", "configs/**", "pyproject.toml", "uv.lock"]
    assert governance["schema_version"] == 1
    assert governance["scope_kind"] == "governance_snapshot"
    assert governance["scope"] == list(SOURCE_GOVERNANCE_FILENAMES)
    assert {"STATUS.md", ".gitignore"} <= governance_paths
    assert execution_paths.isdisjoint(governance_paths)


def test_governance_addition_changes_only_governance_root(tmp_path: Path) -> None:
    before_execution = capture_source_tree(tmp_path)
    before_governance = capture_governance_tree(tmp_path)

    _write(tmp_path / "STATUS.md", "# Status\n\nPILOT_COMPLETE\n")
    _write(tmp_path / ".gitignore", "data/raw/\n")

    after_execution = capture_source_tree(tmp_path)
    after_governance = capture_governance_tree(tmp_path)
    assert after_execution["root_sha256"] == before_execution["root_sha256"]
    assert after_governance["root_sha256"] != before_governance["root_sha256"]


def test_generated_canonical_freeze_configs_do_not_change_execution_identity(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "configs" / "primary.yaml"
    _write(primary, "seed: 1\n")
    _write(tmp_path / "configs" / "confirmatory.yaml", "seed: 2\n")
    before = capture_source_tree(tmp_path)

    _write(tmp_path / "configs" / "primary_frozen.yaml", "seed: 1\n")
    _write(tmp_path / "configs" / "confirmatory_frozen.yaml", "seed: 2\n")
    published = capture_source_tree(tmp_path)

    assert published["root_sha256"] == before["root_sha256"]
    assert published["excluded_paths"] == [
        "configs/confirmatory_frozen.yaml",
        "configs/primary_frozen.yaml",
    ]
    assert not {
        "configs/primary_frozen.yaml",
        "configs/confirmatory_frozen.yaml",
    }.intersection(record["path"] for record in published["artifacts"])

    primary.write_text("seed: 3\n", encoding="utf-8")
    after_live_change = capture_source_tree(tmp_path)
    assert after_live_change["root_sha256"] != published["root_sha256"]


def test_only_exact_canonical_freeze_config_paths_are_excluded(tmp_path: Path) -> None:
    _write(tmp_path / "configs" / "nested" / "primary_frozen.yaml", "seed: 1\n")
    _write(tmp_path / "configs" / "other_frozen.yaml", "seed: 2\n")

    execution_paths = {record["path"] for record in capture_source_tree(tmp_path)["artifacts"]}

    assert execution_paths == {
        "configs/nested/primary_frozen.yaml",
        "configs/other_frozen.yaml",
    }
