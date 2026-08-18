"""Focused tests for environment evidence and the public Typer surface."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from typer.testing import CliRunner

import histo_audit.doctor as doctor_module
from histo_audit.cli import app
from histo_audit.utils.run_tracking import RunTracker, verify_run_integrity


def _patch_expensive_probes(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        doctor_module,
        "_torch_evidence",
        lambda: (
            {"installed": True, "version": "test", "cuda_build": None, "cuda_available": False},
            {
                "available": False,
                "device_count": 0,
                "cudnn_version": None,
                "functional_test": {
                    "attempted": False,
                    "success": False,
                    "finite_gradient": None,
                    "error": None,
                },
            },
            [],
        ),
    )
    monkeypatch.setattr(
        doctor_module,
        "_nvidia_smi",
        lambda: {"available": False, "devices": [], "error": "not present in test"},
    )
    monkeypatch.setattr(doctor_module, "_package_versions", lambda: {"histo-audit": "test"})


def test_collect_doctor_report_has_required_evidence(tmp_path: Path, monkeypatch: Any) -> None:
    _patch_expensive_probes(monkeypatch)
    (tmp_path / "data" / "raw" / "pannuke").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "pannuke" / ".gitkeep").write_text("keep", encoding="utf-8")

    report = doctor_module.collect_doctor_report(tmp_path)

    required = {
        "os",
        "python",
        "packages",
        "pytorch",
        "cuda",
        "gpu",
        "vram",
        "ram",
        "disk",
        "dataset",
        "write_access",
    }
    assert required <= report.keys()
    assert report["write_access"]["writable"] is True
    assert report["dataset"]["status"] == "not_found"
    assert not list(tmp_path.glob(".histo-audit-write-probe-*"))


def test_doctor_cli_prints_and_atomically_saves_json(tmp_path: Path, monkeypatch: Any) -> None:
    _patch_expensive_probes(monkeypatch)
    output = tmp_path / "evidence" / "doctor.json"

    result = CliRunner().invoke(
        app,
        ["doctor", "--project-root", str(tmp_path), "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["project_root"] == str(tmp_path.resolve())
    assert '"write_access"' in result.output
    assert "Saved doctor evidence:" in result.output
    assert not list(output.parent.glob("*.tmp"))


def test_required_command_tree_is_visible() -> None:
    runner = CliRunner()
    root_help = runner.invoke(app, ["--help"])
    experiment_help = runner.invoke(app, ["experiment", "--help"])
    data_help = runner.invoke(app, ["data", "--help"])

    assert root_help.exit_code == 0
    for command in (
        "doctor",
        "data",
        "representations",
        "experiment",
        "preregistration",
        "audit",
        "external",
        "report",
        "demo",
    ):
        assert command in root_help.output
    for command in ("smoke", "pilot", "primary", "confirmatory"):
        assert command in experiment_help.output
    for command in (
        "generate-synthetic",
        "verify-pannuke-acquisition",
        "validate-pannuke",
        "audit-duplicates",
        "build-manifest",
    ):
        assert command in data_help.output


def test_real_data_command_fails_with_clear_gate(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["data", "validate-pannuke", "--project-root", str(tmp_path)],
    )

    assert result.exit_code == 2
    assert "GATED [REAL_DATA_UNAVAILABLE]" in result.output
    assert "DATASET_SETUP.md" in result.output


def test_pannuke_commands_wire_to_read_only_gate_apis(tmp_path: Path, monkeypatch: Any) -> None:
    import histo_audit.cli as cli_module
    import histo_audit.pannuke as pannuke

    source = tmp_path / "pannuke"
    source.mkdir()
    (source / "images.npy").touch()
    output = tmp_path / "outputs"
    monkeypatch.setattr(
        pannuke,
        "locate_pannuke_root",
        lambda explicit_path=None, project_root=None: source,
    )
    validation = SimpleNamespace(
        result=SimpleNamespace(
            root=source,
            folds=(object(), object(), object()),
            fold_validation=(
                SimpleNamespace(
                    n_patches=2,
                    validation_scope="full_semantic_scan",
                    full_scan_patch_count=2,
                ),
                SimpleNamespace(
                    n_patches=3,
                    validation_scope="full_semantic_scan",
                    full_scan_patch_count=3,
                ),
            ),
            inventory=(object(), object()),
            release_complete=True,
            global_mask_qc=SimpleNamespace(
                patch_count=5,
                cross_class_overlap_pixel_count=1,
                cross_class_overlap_patch_count=1,
                void_pixel_count=2,
                void_patch_count=1,
                positive_and_background_pixel_count=0,
                positive_and_background_patch_count=0,
                anomaly_union_patch_count=2,
                normal_patch_count=3,
                affected_instance_count=2,
                overlap_touching_instance_count=2,
            ),
            qc_policy=SimpleNamespace(
                source_masks_modified=False,
                no_class_arbitration=True,
                supplied_background_is_exact_complement_required=False,
                release_annotation_anomalies_are_fatal=False,
                structural_invalidity_is_fatal=True,
                analysis_instance_exclusion_reason="touches_cross_class_overlap",
                applies_identically_to_primary_and_confirmatory=True,
            ),
        ),
        json_path=output / "validation.json",
        markdown_path=output / "validation.md",
        overlay_path=output / "overlay.png",
        raw_inventory_csv_path=output / "raw_files_sha256.csv",
    )
    qc = SimpleNamespace(
        bundle_dir=output / "pannuke_mask_qc",
        json_path=output / "pannuke_mask_qc" / "qc.json",
        patch_csv_path=output / "pannuke_mask_qc" / "patches.csv",
        instance_csv_path=output / "pannuke_mask_qc" / "instances.csv",
        markdown_path=output / "pannuke_mask_qc" / "qc.md",
        overlay_path=output / "pannuke_mask_qc" / "overlay.png",
        overlay_selection_path=output / "pannuke_mask_qc" / "selection.json",
        artifact_manifest_path=output / "pannuke_mask_qc" / "artifacts.json",
        selection_sha256="b" * 64,
        overlay_sha256="c" * 64,
        patch_row_count=5,
        instance_row_count=2,
    )
    manifest = SimpleNamespace(
        row_count=9,
        patch_count=5,
        sha256="a" * 64,
        parquet_path=output / "manifest.parquet",
        summary_csv_path=output / "summary.csv",
    )
    duplicates = SimpleNamespace(
        exact_pair_count=1,
        perceptual_pair_count=2,
        sampled_patch_count=5,
        json_path=output / "duplicates.json",
        csv_path=output / "duplicates.csv",
    )
    monkeypatch.setattr(pannuke, "validate_pannuke", lambda *args, **kwargs: validation)
    monkeypatch.setattr(pannuke, "verify_raw_inventory_unchanged", lambda *args, **kwargs: ())
    monkeypatch.setattr(pannuke, "write_mask_qc_report_bundle", lambda *args, **kwargs: qc)
    monkeypatch.setattr(pannuke, "validate_mask_qc_report_bundle", lambda *args, **kwargs: qc)
    monkeypatch.setattr(
        cli_module,
        "_publish_immutable_validation_artifacts",
        lambda *args, **kwargs: "published",
    )
    monkeypatch.setattr(pannuke, "build_nucleus_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(pannuke, "audit_pannuke_duplicates", lambda *args, **kwargs: duplicates)
    runner = CliRunner()

    validate_result = runner.invoke(
        app,
        [
            "data",
            "validate-pannuke",
            "--project-root",
            str(tmp_path),
            "--root",
            str(source),
            "--output-dir",
            str(output),
        ],
    )
    manifest_result = runner.invoke(
        app,
        [
            "data",
            "build-manifest",
            "--project-root",
            str(tmp_path),
            "--root",
            str(source),
            "--output-dir",
            str(output),
        ],
    )
    duplicates_result = runner.invoke(
        app,
        [
            "data",
            "audit-duplicates",
            "--project-root",
            str(tmp_path),
            "--root",
            str(source),
            "--output-dir",
            str(output),
        ],
    )

    assert validate_result.exit_code == 0, validate_result.output
    assert '"patch_count": 5' in validate_result.output
    assert '"cross_class_overlap_pixel_count": 1' in validate_result.output
    assert '"void_pixel_count": 2' in validate_result.output
    assert '"analysis_exclusion_reason": "touches_cross_class_overlap"' in validate_result.output
    assert '"applies_identically_to_primary_and_confirmatory": true' in validate_result.output
    assert '"no_class_arbitration": true' in validate_result.output
    assert '"qc_artifact_manifest"' in validate_result.output
    assert manifest_result.exit_code == 0, manifest_result.output
    assert '"row_count": 9' in manifest_result.output
    assert duplicates_result.exit_code == 0, duplicates_result.output
    assert '"automatic_deletion": false' in duplicates_result.output


def test_report_cli_rejects_placeholder_metric(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"average_precision": "TODO"}', encoding="utf-8")

    result = CliRunner().invoke(app, ["report", "build", "--metrics", str(metrics)])

    assert result.exit_code == 1
    assert "placeholder metric value rejected" in result.output
    assert not (tmp_path / "report.md").exists()


def test_report_cli_verifies_sealed_run_and_rebuilds_figures_externally(tmp_path: Path) -> None:
    tracker = RunTracker.start(
        experiment_name="report-source",
        config={"seed": {}},
        project_root=tmp_path,
        runs_root=tmp_path / "runs",
    )
    tracker.write_metrics({"run_id": tracker.run_id, "average_precision": 0.8})
    np.savez_compressed(
        tracker.run_directory / "oof_predictions.npz",
        pre_corruption_label=np.asarray([0, 1]),
        observed_label=np.asarray([1, 1]),
        is_injected_corruption=np.asarray([True, False]),
        fold_id=np.asarray([0, 1]),
        class_order=np.asarray([0, 1]),
        tissue_type=np.asarray(["t0", "t1"]),
    )
    tracker.complete()
    runner = CliRunner()

    in_place = runner.invoke(app, ["report", "build", "--run-dir", str(tracker.run_directory)])
    assert in_place.exit_code == 1
    assert "sealed immutable run" in in_place.output

    destination = tmp_path / "external-report"
    rebuilt = runner.invoke(
        app,
        [
            "report",
            "build",
            "--run-dir",
            str(tracker.run_directory),
            "--output-dir",
            str(destination),
        ],
    )
    assert rebuilt.exit_code == 0, rebuilt.output
    payload = json.loads(rebuilt.output)
    assert payload["source_run_integrity"]["valid"] is True
    assert len(payload["figures"]) == 4
    assert (destination / "figures" / "class_distribution.png").is_file()
    assert (destination / "figures" / "tissue_distribution.png").is_file()
    assert verify_run_integrity(tracker.run_directory).valid
