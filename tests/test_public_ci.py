from pathlib import Path

import yaml


def _jobs() -> dict:
    path = Path(__file__).resolve().parents[1] / ".github/workflows/scientific-software.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]


def test_scientific_workflow_materialises_git_lfs_objects_before_validation() -> None:
    jobs = _jobs()
    full = jobs["cross-platform"]
    checkout = next(
        step for step in full["steps"] if step.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"]["lfs"] is True
    assert set(full["strategy"]["matrix"]["os"]) == {"ubuntu-latest", "windows-latest"}
    assert full["strategy"]["matrix"]["shard"] == [0, 1]
    reviewer_checkout = next(
        step
        for step in jobs["reviewer"]["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )
    assert reviewer_checkout["with"]["lfs"] is False


def test_scientific_workflow_runs_the_documented_type_gate() -> None:
    steps = _jobs()["cross-platform"]["steps"]
    install = next(
        index for index, step in enumerate(steps) if step.get("run") == "uv sync --dev --frozen"
    )
    type_check = next(
        index for index, step in enumerate(steps) if step.get("run") == "uv run mypy src"
    )
    tests = next(
        index for index, step in enumerate(steps) if "uv run pytest" in step.get("run", "")
    )
    assert install < type_check < tests
    assert steps[type_check]["if"] == "runner.os == 'Windows' && matrix.shard == 0"


def test_complete_coverage_requires_every_platform_and_all_other_gates() -> None:
    job = _jobs()["complete-coverage"]
    assert set(job["needs"]) == {"reviewer", "cross-platform"}
    assert job["if"] == "always()"
    coverage = next(
        step["run"] for step in job["steps"] if "verify_test_shards.py" in step.get("run", "")
    )
    assert "--platform Linux --platform Windows --shard-count 2" in coverage
    assert "--revision ${{ github.sha }}" in coverage
    results = job["steps"][-1]
    assert results["env"] == {
        "REVIEWER_RESULT": "${{ needs.reviewer.result }}",
        "FULL_RESULT": "${{ needs.cross-platform.result }}",
    }
    assert "== 'success'" in results["run"]


def test_closed_demo_sprite_assets_are_explicitly_publishable() -> None:
    ignore_rules = set(
        (Path(__file__).resolve().parents[1] / ".gitignore")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    assert {
        "!artifacts/mvp_demo/assets/",
        "!artifacts/mvp_demo/assets/hero/",
        "!artifacts/mvp_demo/assets/hero/nuclei/",
        "!artifacts/mvp_demo/assets/hero/nuclei/*.png",
    } <= ignore_rules
