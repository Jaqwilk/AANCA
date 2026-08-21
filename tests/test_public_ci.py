from pathlib import Path


def test_scientific_workflow_materialises_git_lfs_objects_before_validation() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "scientific-software.yml"
    ).read_text(encoding="utf-8")
    checkout_start = workflow.index("uses: actions/checkout@")
    checkout_end = workflow.index("- name: Set up Python", checkout_start)

    assert "lfs: true" in workflow[checkout_start:checkout_end]
