"""Focused tests for immutable run provenance and sourced reporting."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from histo_audit.config import (
    config_sha256,
    load_config,
    load_config_with_file_sha256,
    load_pinned_config,
)
from histo_audit.data.targets import extract_target_crop, highlight_target, mask_bbox
from histo_audit.experiment.smoke import run_synthetic_smoke
from histo_audit.reporting import build_synthetic_report, smoke_runner
from histo_audit.reporting.reconciliation import (
    ArtifactReconciliationError,
    reconcile_synthetic_smoke_artifacts,
)
from histo_audit.utils.run_tracking import (
    RunTracker,
    append_registry_row,
    atomic_write_npz,
    capture_governance_tree,
    capture_source_tree,
    is_run_immutable,
    sha256_path,
    verify_run_integrity,
    windows_compatible_relative_path_sort_key,
)


def test_yaml_config_hash_is_semantic_and_order_independent(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("seed:\n  split: 2\nname: smoke\n", encoding="utf-8")
    second.write_text("name: smoke\nseed: {split: 2}\n", encoding="utf-8")

    assert config_sha256(load_config(first)) == config_sha256(load_config(second))


def test_config_loader_returns_exact_file_identity_and_enforces_pin(tmp_path: Path) -> None:
    path = tmp_path / "frozen.yaml"
    raw = b"schema_version: 1\nfreeze_date: 2026-08-21\nvalue: frozen\n"
    path.write_bytes(raw)

    config, digest = load_config_with_file_sha256(path)

    assert config == {"schema_version": 1, "freeze_date": "2026-08-21", "value": "frozen"}
    assert digest == hashlib.sha256(raw).hexdigest()
    assert load_pinned_config(path, digest) == (config, digest)
    with pytest.raises(RuntimeError, match="changed after freeze"):
        load_pinned_config(path, "0" * 64)


@pytest.mark.parametrize("compressed", [False, True])
def test_atomic_npz_supports_both_archive_modes(tmp_path: Path, compressed: bool) -> None:
    destination = tmp_path / "arrays.npz"
    arrays = {
        "ids": np.asarray(["a", "b"], dtype=np.str_),
        "values": np.asarray([1.5, 2.5], dtype=np.float64),
    }

    assert atomic_write_npz(destination, arrays, compressed=compressed) == destination
    with np.load(destination, allow_pickle=False) as saved:
        assert set(saved.files) == set(arrays)
        for name, expected in arrays.items():
            np.testing.assert_array_equal(saved[name], expected)
    assert list(tmp_path.glob(".arrays.npz.*.tmp")) == []


def test_completed_run_is_unique_registered_and_immutable(tmp_path: Path) -> None:
    config = {"experiment_name": "smoke", "seed": {"split": 2, "model": 3, "corruption": 4}}
    runs = tmp_path / "runs"
    tracker = RunTracker.start(
        experiment_name="smoke",
        config=config,
        project_root=tmp_path,
        runs_root=runs,
    )
    tracker.write_metrics({"average_precision": 0.75})
    tracker.complete()

    assert is_run_immutable(tracker.run_directory)
    for name in (
        "resolved_config.yaml",
        "environment.json",
        "git_state.json",
        "checksums.json",
        "artifact_manifest.json",
        "events.jsonl",
        "run.log",
        "source_tree_manifest.json",
        "run_provenance.json",
        "metrics.json",
        "runtime.json",
        "status.json",
        ".immutable.json",
    ):
        assert (tracker.run_directory / name).is_file()
    status = json.loads((tracker.run_directory / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    environment = json.loads(
        (tracker.run_directory / "environment.json").read_text(encoding="utf-8")
    )
    assert environment["packages"]
    assert list(environment["packages"]) == sorted(environment["packages"])
    assert {"version", "cuda_build", "cuda_available", "cudnn_version"} <= environment[
        "pytorch"
    ].keys()
    assert environment["cuda"]["functional_test_attempted"] is False
    assert {"pytorch_devices", "nvidia_smi"} <= environment["gpu"].keys()
    assert environment["ram"]
    assert environment["disk"]["total_bytes"] > 0
    provenance = json.loads(
        (tracker.run_directory / "run_provenance.json").read_text(encoding="utf-8")
    )
    assert len(provenance["source_tree"]["root_sha256"]) == 64
    events = [
        json.loads(line)
        for line in (tracker.run_directory / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["event"] for event in events] == [
        "run_started",
        "run_finalization_started",
    ]
    with (runs / "registry.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["run_id"] == tracker.run_id
    assert rows[0]["status"] == "completed"
    with pytest.raises(PermissionError, match="immutable"):
        tracker.write_text("late.txt", "must not be written")
    with pytest.raises(RuntimeError, match="already been finalized"):
        tracker.complete()

    other = RunTracker.start(
        experiment_name="smoke",
        config=config,
        project_root=tmp_path,
        runs_root=runs,
    )
    assert other.run_id != tracker.run_id
    other.complete()


def test_source_tree_hash_changes_on_generating_source_edit_but_ignores_artifacts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "package" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "smoke.yaml").write_text("seed: 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='example'\n", encoding="utf-8")
    first = capture_source_tree(tmp_path)
    assert first["schema_version"] == 3
    assert first["scope_kind"] == "execution_source"
    assert first["scope"] == ["src/**", "configs/**", "pyproject.toml", "uv.lock"]

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    artifact = artifacts / "result.json"
    artifact.write_text("{}", encoding="utf-8")
    assert capture_source_tree(tmp_path)["root_sha256"] == first["root_sha256"]
    artifact.write_text('{"result": "changed"}', encoding="utf-8")
    assert capture_source_tree(tmp_path)["root_sha256"] == first["root_sha256"]

    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = capture_source_tree(tmp_path)
    assert second["root_sha256"] != first["root_sha256"]


@pytest.mark.parametrize(
    "governance_filename",
    [
        "AGENTS.md",
        "SPEC.md",
        "PLAN.md",
        "STATUS.md",
        "PRE_REGISTRATION.md",
        "DATASET_SETUP.md",
        "DECISIONS.md",
        "ETHICS_AND_LIMITATIONS.md",
        ".gitignore",
    ],
)
def test_governance_edit_changes_only_governance_identity(
    tmp_path: Path,
    governance_filename: str,
) -> None:
    governance = tmp_path / governance_filename
    governance.write_text("frozen scientific rule\n", encoding="utf-8")

    first_source = capture_source_tree(tmp_path)
    first_governance = capture_governance_tree(tmp_path)
    assert first_governance["schema_version"] == 1
    assert first_governance["scope_kind"] == "governance_snapshot"
    assert governance_filename in first_governance["governance_files"]
    assert governance_filename in {artifact["path"] for artifact in first_governance["artifacts"]}
    assert governance_filename not in {artifact["path"] for artifact in first_source["artifacts"]}

    governance.write_text("amended scientific rule\n", encoding="utf-8")
    second_source = capture_source_tree(tmp_path)
    second_governance = capture_governance_tree(tmp_path)
    assert second_source["root_sha256"] == first_source["root_sha256"]
    assert second_governance["root_sha256"] != first_governance["root_sha256"]


def test_integrity_verification_detects_edit_addition_and_deletion(tmp_path: Path) -> None:
    tracker = RunTracker.start(
        experiment_name="integrity-test",
        config={"seed": {"split": 7}},
        project_root=tmp_path,
        runs_root=tmp_path / "runs",
    )
    tracker.write_metrics({"average_precision": 0.75})
    tracker.complete()

    assert verify_run_integrity(tracker.run_directory).valid
    metrics_path = tracker.run_directory / "metrics.json"
    original_metrics = metrics_path.read_bytes()

    metrics_path.write_text('{"average_precision": 0.76}\n', encoding="utf-8")
    edited = verify_run_integrity(tracker.run_directory)
    assert not edited.valid
    assert edited.changed_paths == ("metrics.json",)

    metrics_path.write_bytes(original_metrics)
    assert verify_run_integrity(tracker.run_directory).valid
    added_path = tracker.run_directory / "unsealed-extra.txt"
    added_path.write_text("unexpected", encoding="utf-8")
    added = verify_run_integrity(tracker.run_directory)
    assert not added.valid
    assert added.added_paths == ("unsealed-extra.txt",)

    added_path.unlink()
    assert verify_run_integrity(tracker.run_directory).valid
    metrics_path.unlink()
    deleted = verify_run_integrity(tracker.run_directory)
    assert not deleted.valid
    assert deleted.missing_paths == ("metrics.json",)

    metrics_path.write_bytes(original_metrics)
    assert verify_run_integrity(tracker.run_directory).valid


def test_failed_run_saves_traceback_and_registry_status(tmp_path: Path) -> None:
    tracker = RunTracker.start(
        experiment_name="failure-test",
        config={"seed": {}},
        project_root=tmp_path,
        runs_root=tmp_path / "runs",
    )
    try:
        raise ValueError("deliberate test failure")
    except ValueError as error:
        tracker.fail(error)

    traceback_text = (tracker.run_directory / "traceback.txt").read_text(encoding="utf-8")
    status = json.loads((tracker.run_directory / "status.json").read_text(encoding="utf-8"))
    assert "ValueError: deliberate test failure" in traceback_text
    assert status["status"] == "failed"
    assert status["traceback"] == "traceback.txt"
    assert is_run_immutable(tracker.run_directory)
    events = [
        json.loads(line)
        for line in (tracker.run_directory / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["event"] for event in events] == [
        "run_started",
        "run_failed",
        "run_finalization_started",
    ]


def test_registry_append_preserves_existing_rows(tmp_path: Path) -> None:
    registry = tmp_path / "registry.csv"
    append_registry_row(registry, {"run_id": "one", "status": "completed"})
    before = registry.read_bytes()
    append_registry_row(registry, {"run_id": "two", "status": "failed"})

    after = registry.read_bytes()
    assert after.startswith(before)
    with registry.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["run_id"] for row in rows] == ["one", "two"]


def test_tree_checksum_includes_names_and_contents(tmp_path: Path) -> None:
    folder = tmp_path / "tree"
    folder.mkdir()
    (folder / "a.txt").write_text("alpha", encoding="utf-8")
    before = sha256_path(folder)
    (folder / "b.txt").write_text("beta", encoding="utf-8")

    assert sha256_path(folder) != before


def test_tree_checksum_path_order_is_platform_independent_and_windows_compatible() -> None:
    relative_paths = (
        "alpha/A.txt",
        "Alpha/z.txt",
        "a-.txt",
        "a/file.txt",
        "Fold 1/README.md",
        "Fold 1/images/fold1/images.npy",
        "fold_1.zip",
    )
    expected = (
        "a/file.txt",
        "a-.txt",
        "alpha/A.txt",
        "Alpha/z.txt",
        "Fold 1/images/fold1/images.npy",
        "Fold 1/README.md",
        "fold_1.zip",
    )
    posix_paths = tuple(PurePosixPath(value).as_posix() for value in relative_paths)
    windows_paths = tuple(PureWindowsPath(value).as_posix() for value in relative_paths)

    assert tuple(sorted(posix_paths, key=windows_compatible_relative_path_sort_key)) == expected
    assert tuple(sorted(windows_paths, key=windows_compatible_relative_path_sort_key)) == expected


def test_report_values_are_sourced_and_placeholders_are_rejected(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "run_id": "source-run",
                "ranking": {"average_precision": 0.8125},
                "review_budget": {"fraction": 0.05, "recall": 0.6},
            }
        ),
        encoding="utf-8",
    )

    report = build_synthetic_report(metrics)

    markdown = report.markdown_path.read_text(encoding="utf-8")
    html = report.html_path.read_text(encoding="utf-8")
    assert "0.8125" in markdown
    assert "0.6" in markdown
    assert report.metrics_sha256 in markdown
    assert "<!doctype html>" in html
    assert "0.8125" in html

    metrics.write_text('{"average_precision": "PLACEHOLDER"}', encoding="utf-8")
    with pytest.raises(ValueError, match="placeholder"):
        build_synthetic_report(metrics, markdown_name="bad.md", html_name="bad.html")
    assert not (tmp_path / "bad.md").exists()


def test_report_allows_documented_optional_null_metadata(tmp_path: Path) -> None:
    successful = tmp_path / "successful.json"
    successful.write_text(
        json.dumps(
            {
                "average_precision": 0.8,
                "cleanlab": {"available": True, "issue_count": 2, "error": None},
            }
        ),
        encoding="utf-8",
    )
    unavailable = tmp_path / "unavailable.json"
    unavailable.write_text(
        json.dumps(
            {
                "average_precision": 0.8,
                "cleanlab": {
                    "available": False,
                    "issue_count": None,
                    "error": "optional package unavailable",
                },
            }
        ),
        encoding="utf-8",
    )

    assert build_synthetic_report(successful, output_directory=tmp_path / "success")
    assert build_synthetic_report(unavailable, output_directory=tmp_path / "unavailable")

    undocumented = tmp_path / "undocumented.json"
    undocumented.write_text('{"average_precision": null}', encoding="utf-8")
    with pytest.raises(ValueError, match="undocumented missing metric"):
        build_synthetic_report(undocumented)


def test_report_refuses_in_place_write_to_sealed_run_but_allows_external_output(
    tmp_path: Path,
) -> None:
    tracker = RunTracker.start(
        experiment_name="sealed-report",
        config={"seed": {}},
        project_root=tmp_path,
        runs_root=tmp_path / "runs",
    )
    metrics_path = tracker.write_metrics({"run_id": tracker.run_id, "average_precision": 0.8})
    tracker.complete()

    with pytest.raises(PermissionError, match="sealed immutable run"):
        build_synthetic_report(metrics_path, generate_figures=False)

    external = build_synthetic_report(
        metrics_path,
        output_directory=tmp_path / "external-report",
        generate_figures=False,
    )
    assert external.markdown_path.is_file()
    assert external.html_path.is_file()
    assert verify_run_integrity(tracker.run_directory).valid


def test_report_omits_repeat_runs_and_seed_lists(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    repeat_runs = [
        {f"metric_{metric}": repeat + metric / 100 for metric in range(20)} for repeat in range(100)
    ]
    metrics.write_text(
        json.dumps(
            {
                "artifact_scope": "synthetic_software_validation",
                "sample_counts": {"total": 300, "source_groups": 60},
                "random_review": {
                    "mean_recall": 0.5,
                    "seeds": [987_654_321 + index for index in range(100)],
                },
                "downstream_restoration": {
                    "random_review_restoration": {
                        "mean_macro_f1": 0.7,
                        "runs": repeat_runs,
                    },
                    "per_class": {"neoplastic": {"f1": 0.61}},
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_synthetic_report(metrics)
    markdown = report.markdown_path.read_text(encoding="utf-8")
    static_html = report.html_path.read_text(encoding="utf-8")

    assert "runs[" not in markdown
    assert "seeds[" not in markdown
    assert "987654321" not in markdown
    assert "mean_macro_f1" in markdown
    assert len(markdown.encode("utf-8")) < 50_000
    assert len(static_html.encode("utf-8")) < 100_000
    dataset_section = markdown.split("## Dataset provenance", maxsplit=1)[1].split(
        "## Terminology", maxsplit=1
    )[0]
    assert "per_class" not in dataset_section


def _make_consistent_fake_core(tmp_path: Path) -> SimpleNamespace:
    core_directory = tmp_path / "artifacts" / "synthetic_core" / "core-run"
    core_directory.mkdir(parents=True)
    predictions = core_directory / "oof_predictions.npz"
    representation = core_directory / "target_representation_example.npz"
    ranking = core_directory / "ranking.csv"
    corruption = core_directory / "corruption_manifest.json"
    report_inputs = core_directory / "report_inputs.json"
    oof_provenance = core_directory / "oof_provenance.json"
    metrics_path = core_directory / "metrics.json"
    configuration_payload = {"mechanism": "symmetric_random_corruption"}
    configuration_hash = hashlib.sha256(
        json.dumps(configuration_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    probabilities = np.asarray([[0.2, 0.8], [0.1, 0.9], [0.7, 0.3], [0.8, 0.2]])
    np.savez_compressed(
        predictions,
        sample_ids=np.asarray(["s0", "s1", "s2", "s3"]),
        group_ids=np.asarray(["g0", "g1", "g0", "g1"]),
        tissue_type=np.asarray(["t0", "t0", "t1", "t1"]),
        probabilities=probabilities,
        pre_corruption_label=np.asarray([0, 1, 0, 1]),
        observed_label=np.asarray([1, 1, 0, 0]),
        is_injected_corruption=np.asarray([True, False, False, True]),
        predicted_class=np.asarray([1, 1, 0, 0]),
        fold_id=np.asarray([0, 1, 0, 1]),
        class_order=np.asarray([0, 1]),
    )
    full_patch = np.full((32, 32, 3), [205, 155, 185], dtype=np.uint8)
    full_mask = np.zeros((32, 32), dtype=bool)
    full_mask[10:19, 11:20] = True
    full_patch[full_mask] = np.asarray([95, 55, 125], dtype=np.uint8)
    target_crop = extract_target_crop(full_patch, full_mask, output_size=24, padding=3)
    np.savez_compressed(
        representation,
        sample_id=np.asarray("s0", dtype=np.str_),
        target_instance_id=np.asarray(1, dtype=np.int64),
        full_patch=full_patch,
        full_target_mask=full_mask,
        source_bbox=np.asarray(mask_bbox(full_mask), dtype=np.int64),
        crop_source_box=np.asarray(target_crop.source_box, dtype=np.int64),
        target_crop=target_crop.image,
        crop_target_mask=target_crop.target_mask,
        highlighted_full_patch=highlight_target(full_patch, full_mask),
        highlighted_crop=highlight_target(target_crop.image, target_crop.target_mask),
    )
    ranking.write_text(
        "rank,sample_id,group_id,tissue_type,pre_corruption_label,observed_label,"
        "is_injected_corruption,predicted_class,self_confidence\n"
        "1,s0,g0,t0,0,1,True,1,0.9\n"
        "2,s3,g1,t1,1,0,True,0,0.8\n"
        "3,s2,g0,t1,0,0,False,0,0.2\n"
        "4,s1,g1,t0,1,1,False,1,0.1\n",
        encoding="utf-8",
    )
    corruption.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "sample_id": "s0",
                        "group_id": "g0",
                        "pre_corruption_label": 0,
                        "observed_label": 1,
                        "is_injected_corruption": True,
                        "configuration_hash": configuration_hash,
                    },
                    {
                        "sample_id": "s1",
                        "group_id": "g1",
                        "pre_corruption_label": 1,
                        "observed_label": 1,
                        "is_injected_corruption": False,
                        "configuration_hash": configuration_hash,
                    },
                    {
                        "sample_id": "s2",
                        "group_id": "g0",
                        "pre_corruption_label": 0,
                        "observed_label": 0,
                        "is_injected_corruption": False,
                        "configuration_hash": configuration_hash,
                    },
                    {
                        "sample_id": "s3",
                        "group_id": "g1",
                        "pre_corruption_label": 1,
                        "observed_label": 0,
                        "is_injected_corruption": True,
                        "configuration_hash": configuration_hash,
                    },
                ],
                "configuration_hash": configuration_hash,
                "configuration_payload": configuration_payload,
            }
        ),
        encoding="utf-8",
    )
    report_inputs.write_text(
        json.dumps(
            {
                "software_validation_only": True,
                "class_names": ["class_zero", "class_one"],
                "split": {
                    "audit_groups": ["g0", "g1"],
                    "reference_validation_groups": ["g2"],
                    "final_test_groups": ["g3"],
                },
            }
        ),
        encoding="utf-8",
    )
    oof_provenance.write_text(
        json.dumps(
            {
                "class_order": [0, 1],
                "folds": [
                    {
                        "fold_id": 0,
                        "training_groups": ["g1"],
                        "held_out_groups": ["g0"],
                        "held_out_sample_ids": ["s0", "s2"],
                        "group_overlap": [],
                    },
                    {
                        "fold_id": 1,
                        "training_groups": ["g0"],
                        "held_out_groups": ["g1"],
                        "held_out_sample_ids": ["s1", "s3"],
                        "group_overlap": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    metrics = {
        "artifact_scope": "synthetic_software_validation",
        "run_id": "core-run",
        "resolved_core_config": {
            "dataset_seed": 1,
            "n_groups": 4,
            "instances_per_group": 5,
            "patch_size": 40,
        },
        "sample_counts": {
            "total": 6,
            "audit_pool": 4,
            "reference_validation": 1,
            "final_reference_test": 1,
            "source_groups": 4,
        },
        "corruption": {"exact_count": 2, "configuration_hash": configuration_hash},
        "oof": {
            "folds": 2,
            "complete_once_coverage": True,
            "group_overlap_count": 0,
            "maximum_probability_sum_error": float(
                np.max(np.abs(probabilities.sum(axis=1) - 1.0), initial=0.0)
            ),
        },
        "ranking": {
            "self_confidence": {
                "auroc": 1.0,
                "subgroups": {
                    "pre_corruption_class": [
                        {
                            "subgroup": "class_zero",
                            "total_examples": 2,
                            "injected_corruptions": 1,
                            "average_precision": {
                                "status": "not_applicable",
                                "value": None,
                                "reason": "requires more support",
                            },
                            "status": "insufficient_support",
                            "reason": "requires more support",
                        },
                        {
                            "subgroup": "class_one",
                            "total_examples": 2,
                            "injected_corruptions": 1,
                            "average_precision": {
                                "status": "not_applicable",
                                "value": None,
                                "reason": "requires more support",
                            },
                            "status": "insufficient_support",
                            "reason": "requires more support",
                        },
                    ],
                    "tissue_type": [
                        {
                            "subgroup": "t0",
                            "total_examples": 2,
                            "injected_corruptions": 1,
                            "average_precision": {
                                "status": "not_applicable",
                                "value": None,
                                "reason": "requires more support",
                            },
                            "status": "insufficient_support",
                            "reason": "requires more support",
                        },
                        {
                            "subgroup": "t1",
                            "total_examples": 2,
                            "injected_corruptions": 1,
                            "average_precision": {
                                "status": "not_applicable",
                                "value": None,
                                "reason": "requires more support",
                            },
                            "status": "insufficient_support",
                            "reason": "requires more support",
                        },
                    ],
                },
                "review_budgets": {
                    "0.25": {
                        "average_precision": 1.0,
                        "budget_fraction": 0.25,
                        "reviewed_count": 1,
                        "injected_reviewed": 1,
                        "total_examples": 4,
                        "injected_total": 2,
                        "false_alert_count": 0,
                        "precision": 1.0,
                        "recall": 0.5,
                        "expected_random_recall": 0.25,
                        "lift_over_random": 2.0,
                    },
                    "0.5": {
                        "average_precision": 1.0,
                        "budget_fraction": 0.5,
                        "reviewed_count": 2,
                        "injected_reviewed": 2,
                        "total_examples": 4,
                        "injected_total": 2,
                        "false_alert_count": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "expected_random_recall": 0.5,
                        "lift_over_random": 2.0,
                    },
                },
            }
        },
        "downstream_restoration": {
            "uncorrupted_reference_baseline": {"metrics": {"macro_f1": 0.9}},
            "corrupted_observed_baseline": {"metrics": {"macro_f1": 0.7}},
            "random_review_restoration": {
                "macro_f1_mean": 0.75,
                "macro_f1_interval_95": [0.72, 0.78],
            },
            "audit_guided_restoration": {"metrics": {"macro_f1": 0.82}},
        },
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    return SimpleNamespace(
        success=True,
        status="PIPELINE_COMPLETE",
        run_id="core-run",
        run_dir=core_directory,
        metrics=metrics,
        metrics_path=metrics_path,
        predictions_path=predictions,
        rankings_path=ranking,
        corruption_manifest_path=corruption,
        oof_provenance_path=oof_provenance,
        report_inputs_path=report_inputs,
        representation_example_path=representation,
    )


def _make_current_core(tmp_path: Path) -> Any:
    return run_synthetic_smoke(
        project_root=tmp_path,
        config={
            "n_groups": 12,
            "patch_size": 40,
            "oof_splits": 3,
            "random_review_repeats": 3,
            "downstream_random_repeats": 2,
            "bootstrap_iterations": 5,
        },
    )


def _rewrite_npz(path: Path, **updates: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays.update(updates)
    np.savez_compressed(path, **arrays)


def test_tracked_smoke_preserves_core_file_formats(tmp_path: Path, monkeypatch: Any) -> None:
    core_result = _make_current_core(tmp_path)

    def fake_runner(**_: Any) -> SimpleNamespace:
        return core_result

    monkeypatch.setattr(smoke_runner, "_load_core_runner", lambda: fake_runner)
    execution = smoke_runner.execute_synthetic_smoke(
        project_root=tmp_path,
        config={"experiment_name": "synthetic_smoke", "seed": {"split": 2}},
    )

    with np.load(execution.run_directory / "oof_predictions.npz") as loaded:
        assert loaded["probabilities"].shape[1] == 5
    assert (
        (execution.run_directory / "ranking.csv")
        .read_text(encoding="utf-8")
        .startswith("rank,sample_id,group_id")
    )
    corruption_manifest = json.loads(
        (execution.run_directory / "corruption_manifest.json").read_text(encoding="utf-8")
    )
    assert len(corruption_manifest["rows"]) == core_result.metrics["sample_counts"]["audit_pool"]
    assert (
        json.loads((execution.run_directory / "report_inputs.json").read_text(encoding="utf-8"))[
            "artifact_scope"
        ]
        == "synthetic_software_validation"
    )
    provenance = json.loads(
        (execution.run_directory / "oof_provenance.json").read_text(encoding="utf-8")
    )
    assert len(provenance["folds"]) == 3
    reconciliation = json.loads(
        (execution.run_directory / "artifact_reconciliation.json").read_text(encoding="utf-8")
    )
    assert reconciliation["status"] == "passed"
    assert (
        reconciliation["neighbour_evidence_sample_count"]
        == (core_result.metrics["sample_counts"]["audit_pool"])
    )
    assert (
        reconciliation["dataset_evidence_sample_count"]
        == (core_result.metrics["sample_counts"]["total"])
    )
    assert reconciliation["synthetic_duplicate_audit_status"] == "passed"
    assert reconciliation["synthetic_duplicate_audit_real_data_gate_eligible"] is False
    assert reconciliation["bootstrap_status"] == "reported"
    assert reconciliation["bootstrap_valid_iterations"] > 0
    assert reconciliation["bootstrap_comparison_count"] >= 5
    tracked_report_inputs = json.loads(
        (execution.run_directory / "report_inputs.json").read_text(encoding="utf-8")
    )
    for key in (
        "predictions_path",
        "rankings_path",
        "corruption_manifest_path",
        "oof_provenance_path",
        "representation_example_path",
        "neighbour_evidence_path",
        "restoration_evidence_path",
        "bootstrap_evidence_path",
        "dataset_evidence_path",
        "source_manifest_path",
        "source_manifest_csv_path",
        "duplicate_audit_path",
        "duplicate_candidates_csv_path",
        "duplicate_candidates_figure_path",
        "report_inputs_path",
    ):
        tracked_path = Path(tracked_report_inputs[key])
        expected_parent = (
            execution.run_directory.resolve() / "figures"
            if key == "duplicate_candidates_figure_path"
            else execution.run_directory.resolve()
        )
        assert tracked_path.parent == expected_parent
        assert tracked_path.is_file()
    figure_paths = sorted((execution.run_directory / "figures").glob("*.png"))
    assert len(figure_paths) == 20
    assert (execution.run_directory / "figures" / "target_representation_example.png").is_file()
    for required_figure in (
        "precision_recall_curves.png",
        "paired_bootstrap_interval.png",
        "paired_bootstrap_distribution.png",
        "paired_method_differences.png",
        "per_class_results_support.png",
        "per_tissue_results_support.png",
        "top_suspicious_controlled_examples.png",
        "fold_safe_neighbour_explanation_grid.png",
        "false_high_and_low_risk_examples.png",
        "target_example_audit_evidence.png",
        "duplicate_candidates.png",
    ):
        assert (execution.run_directory / "figures" / required_figure).is_file()
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in figure_paths)
    markdown = execution.report.markdown_path.read_text(encoding="utf-8")
    static_html = execution.report.html_path.read_text(encoding="utf-8")
    assert "![" in markdown and "](figures/" in markdown
    assert "figures/tissue_distribution.png" in markdown
    assert "figures/precision_recall_curves.png" in markdown
    assert "figures/paired_method_differences.png" in markdown
    assert "figures/paired_bootstrap_distribution.png" in markdown
    assert "figures/false_high_and_low_risk_examples.png" in markdown
    assert "figures/duplicate_candidates.png" in markdown
    assert "cannot satisfy a real-data duplicate gate" in markdown
    assert '<img src="figures/' in static_html
    assert 'src="figures/tissue_distribution.png"' in static_html
    assert 'src="figures/precision_recall_curves.png"' in static_html
    figure_sources = json.loads(
        (execution.run_directory / "figures" / "figure_sources.json").read_text(encoding="utf-8")
    )
    assert set(figure_sources["evidence_inputs"]) == {
        "corruption_manifest_path",
        "dataset_evidence_path",
        "neighbour_evidence_path",
        "rankings_path",
    }
    assert all(value["sha256"] for value in figure_sources["evidence_inputs"].values())
    assert figure_sources["bootstrap_evidence_input"]["sha256"]
    assert set(figure_sources["duplicate_audit_inputs"]) == {
        "duplicate_audit_path",
        "duplicate_candidates_csv_path",
        "duplicate_candidates_figure_path",
    }
    assert all(value["sha256"] for value in figure_sources["duplicate_audit_inputs"].values())
    assert len(figure_sources["class_names_sha256"]) == 64
    sourced_by_key = {figure["key"]: figure for figure in figure_sources["figures"]}
    assert sourced_by_key["top_suspicious_controlled_examples"]["provenance"]["selected_sample_ids"]
    assert (
        sourced_by_key["fold_safe_neighbour_explanation_grid"]["provenance"]["fold_safe_rule"]
        == "query/self group absent from saved neighbour groups"
    )
    checksums = json.loads((execution.run_directory / "checksums.json").read_text(encoding="utf-8"))
    assert checksums["dataset"]["sha256"]
    assert checksums["dataset"]["is_raw_real_dataset_checksum"] is False
    assert Path(checksums["dataset"]["path"]).name == "synthetic_dataset_evidence.npz"
    assert checksums["manifest"]["sha256"]
    assert Path(checksums["manifest"]["path"]).name == "synthetic_source_manifest.json"
    assert checksums["corruption_manifest"]["sha256"]
    assert set(checksums["machine_readable_evidence"]) == {
        "neighbour_evidence.npz",
        "restoration_evidence.npz",
        "bootstrap_evidence.npz",
    }
    assert checksums["duplicate_audit_status"] == (
        "synthetic_patch_audit_complete_not_real_data_gate_eligible"
    )
    assert checksums["duplicate_audit"]["json"]["sha256"]
    assert checksums["duplicate_audit"]["candidate_csv"]["sha256"]
    assert checksums["duplicate_audit"]["candidate_figure"]["sha256"]
    assert checksums["duplicate_audit"]["real_data_duplicate_gate_eligible"] is False
    duplicate_audit = json.loads(
        (execution.run_directory / "duplicate_audit.json").read_text(encoding="utf-8")
    )
    assert (
        duplicate_audit["dataset_evidence"]["nucleus_sample_count"]
        == (core_result.metrics["sample_counts"]["total"])
    )
    assert (
        duplicate_audit["dataset_evidence"]["unique_patch_count"]
        == (core_result.metrics["sample_counts"]["source_groups"])
    )
    assert duplicate_audit["real_data_duplicate_gate_eligible"] is False
    with (tmp_path / "artifacts" / "runs" / "registry.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        registry_row = list(csv.DictReader(handle))[-1]
    assert registry_row["dataset_sha256"] == checksums["dataset"]["sha256"]
    assert registry_row["manifest_sha256"] == checksums["manifest"]["sha256"]
    assert is_run_immutable(execution.run_directory)
    assert verify_run_integrity(execution.run_directory).valid
    event_names = [
        json.loads(line)["event"]
        for line in (execution.run_directory / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert event_names == [
        "run_started",
        "artifact_reconciliation_passed",
        "run_finalization_started",
    ]


def test_tracked_smoke_reconciliation_failure_is_sealed_as_failed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    core_result = _make_current_core(tmp_path)
    ranking_path = Path(core_result.rankings_path)
    with ranking_path.open(encoding="utf-8", newline="") as stream:
        ranking_rows = list(csv.reader(stream))
    self_confidence_column = ranking_rows[0].index("self_confidence")
    ranking_rows[1][self_confidence_column] = "nan"
    with ranking_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(ranking_rows)
    monkeypatch.setattr(smoke_runner, "_load_core_runner", lambda: lambda **_: core_result)

    with pytest.raises(ArtifactReconciliationError, match="non-finite ranking score"):
        smoke_runner.execute_synthetic_smoke(
            project_root=tmp_path,
            config={"experiment_name": "synthetic_smoke", "seed": {"split": 2}},
        )

    registry_path = tmp_path / "artifacts" / "runs" / "registry.csv"
    with registry_path.open(encoding="utf-8", newline="") as handle:
        failed_row = list(csv.DictReader(handle))[-1]
    failed_run = Path(failed_row["run_path"])
    assert failed_row["status"] == "failed"
    assert "ArtifactReconciliationError" in (failed_run / "traceback.txt").read_text(
        encoding="utf-8"
    )
    assert verify_run_integrity(failed_run).valid


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("configuration_hash", "disagrees with metrics.corruption.configuration_hash"),
        ("oof_fold_count", "OOF fold count disagrees"),
        ("source_group_count", "source-group count disagrees"),
    ],
)
def test_reconciliation_rejects_rendered_evidence_mismatches(
    tmp_path: Path,
    mismatch: str,
    message: str,
) -> None:
    core_result = _make_current_core(tmp_path / mismatch)
    if mismatch == "configuration_hash":
        manifest_path = Path(core_result.corruption_manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["configuration_hash"] = "b" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif mismatch == "oof_fold_count":
        core_result.metrics["oof"]["folds"] += 1
    else:
        core_result.metrics["sample_counts"]["source_groups"] += 1

    with pytest.raises(ArtifactReconciliationError, match=message):
        reconcile_synthetic_smoke_artifacts(core_result.run_dir, core_result.metrics)


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("missing_neighbours", "references missing neighbour_evidence.npz"),
        ("same_group_neighbour", "includes query/self group"),
        ("restoration_budget", "budget count is inconsistent"),
        ("restored_label", "guided restored labels/mask are not reconstructible"),
        ("restoration_probability", "probabilities are invalid"),
        ("corrupted_final_dataset", "dataset corruption labels/flags are inconsistent"),
    ],
)
def test_reconciliation_rejects_machine_evidence_mismatches(
    tmp_path: Path,
    mismatch: str,
    message: str,
) -> None:
    core_result = _make_current_core(tmp_path / mismatch)
    run_dir = Path(core_result.run_dir)
    neighbour_path = run_dir / "neighbour_evidence.npz"
    restoration_path = run_dir / "restoration_evidence.npz"
    dataset_path = run_dir / "synthetic_dataset_evidence.npz"
    if mismatch == "missing_neighbours":
        neighbour_path.unlink()
    elif mismatch == "same_group_neighbour":
        with np.load(neighbour_path, allow_pickle=False) as payload:
            groups = np.asarray(payload["neighbour_groups"]).copy()
            query_groups = np.asarray(payload["group_ids"])
        groups[0, 0] = query_groups[0]
        _rewrite_npz(neighbour_path, neighbour_groups=groups)
    elif mismatch == "restoration_budget":
        with np.load(restoration_path, allow_pickle=False) as payload:
            budget_count = int(payload["review_budget_count"].item())
        _rewrite_npz(
            restoration_path,
            review_budget_count=np.asarray(budget_count + 1, dtype=np.int64),
        )
    elif mismatch == "restored_label":
        with np.load(restoration_path, allow_pickle=False) as payload:
            restored = np.asarray(payload["audit_guided_restored_label"]).copy()
            class_order = np.asarray(payload["class_order"])
        restored[0] = class_order[(int(restored[0]) + 1) % len(class_order)]
        _rewrite_npz(restoration_path, audit_guided_restored_label=restored)
    elif mismatch == "restoration_probability":
        key = "audit_guided_restoration_final_test_probabilities"
        with np.load(restoration_path, allow_pickle=False) as payload:
            probabilities = np.asarray(payload[key]).copy()
        probabilities[0] = 0.0
        _rewrite_npz(restoration_path, **{key: probabilities})
    else:
        with np.load(dataset_path, allow_pickle=False) as payload:
            injected = np.asarray(payload["is_injected_corruption"]).copy()
            partition = np.asarray(payload["split_partition"])
        injected[np.flatnonzero(partition == "final_reference_test")[0]] = True
        _rewrite_npz(dataset_path, is_injected_corruption=injected)

    with pytest.raises(ArtifactReconciliationError, match=message):
        reconcile_synthetic_smoke_artifacts(run_dir, core_result.metrics)


def test_zero_corruption_tracked_smoke_uses_valid_alternative_figures(tmp_path: Path) -> None:
    execution = smoke_runner.execute_synthetic_smoke(
        project_root=tmp_path,
        config={
            "experiment_name": "zero_corruption_smoke",
            "n_groups": 12,
            "patch_size": 40,
            "oof_splits": 3,
            "corruption_rate": 0,
            "random_review_repeats": 3,
            "downstream_random_repeats": 2,
            "bootstrap_iterations": 5,
        },
    )

    figure_names = {path.name for path in execution.report.figure_paths}
    assert {
        "class_distribution.png",
        "tissue_distribution.png",
        "oof_fold_distribution.png",
        "corruption_transition_matrix.png",
        "target_representation_example.png",
        "score_distribution_by_method.png",
        "false_alerts_vs_review_budget.png",
        "downstream_macro_f1.png",
        "per_class_results_support.png",
        "per_tissue_results_support.png",
        "top_suspicious_controlled_examples.png",
        "fold_safe_neighbour_explanation_grid.png",
        "target_example_audit_evidence.png",
        "false_high_risk_examples.png",
        "duplicate_candidates.png",
    } == figure_names
    assert "average_precision_by_method.png" not in figure_names
    assert "recall_vs_review_budget.png" not in figure_names
    assert "lift_vs_review_budget.png" not in figure_names
    assert "precision_recall_curves.png" not in figure_names
    assert "paired_bootstrap_interval.png" not in figure_names
    assert "paired_bootstrap_distribution.png" not in figure_names
    assert "paired_method_differences.png" not in figure_names
    assert "false_high_and_low_risk_examples.png" not in figure_names
    reconciliation = json.loads(
        (execution.run_directory / "artifact_reconciliation.json").read_text(encoding="utf-8")
    )
    assert reconciliation["bootstrap_status"] == "not_applicable"
    assert reconciliation["bootstrap_valid_iterations"] == 0
    assert verify_run_integrity(execution.run_directory).valid


@pytest.mark.parametrize("mismatch", ["difference", "partial_group_draw"])
def test_reconciliation_rejects_bootstrap_evidence_mismatches(
    tmp_path: Path,
    mismatch: str,
) -> None:
    core_result = _make_current_core(tmp_path / mismatch)
    run_dir = Path(core_result.run_dir)
    bootstrap_path = run_dir / "bootstrap_evidence.npz"
    with np.load(bootstrap_path, allow_pickle=False) as payload:
        if mismatch == "difference":
            differences = np.asarray(payload["differences"]).copy()
            differences[0, 0] += 0.125
            updates = {"differences": differences}
            expected = "numeric arrays are invalid or inconsistent"
        else:
            draw_indices = np.asarray(payload["draw_indices"]).copy()
            draw_indices[0] = draw_indices[1]
            updates = {"draw_indices": draw_indices}
            expected = "does not resample whole source group"
    _rewrite_npz(bootstrap_path, **updates)

    with pytest.raises(ArtifactReconciliationError, match=expected):
        reconcile_synthetic_smoke_artifacts(run_dir, core_result.metrics)
