"""CLI regression tests for the PanNuke M5/M6 data and privacy gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from histo_audit.cli import app


def _patch_local_release(tmp_path: Path, monkeypatch: Any) -> Path:
    import histo_audit.pannuke as pannuke

    source = tmp_path / "data" / "raw" / "pannuke"
    source.mkdir(parents=True)
    (source / "images.npy").touch()
    monkeypatch.setattr(
        pannuke,
        "locate_pannuke_root",
        lambda explicit_path=None, project_root=None: source,
    )
    return source


def _acquisition_manifest() -> dict[str, Any]:
    archives = [
        {
            "fold": fold,
            "size_bytes": 100 + fold,
            "sha256": str(fold) * 64,
            "zip_crc_status": "passed",
            "path_safety_status": "passed",
        }
        for fold in (1, 2, 3)
    ]
    return {
        "archives": archives,
        "extracted_npy_inventory": [{} for _ in range(9)],
        "extracted_document_inventory": [{} for _ in range(9)],
        "raw_release_read_only_verification": {
            "status": "passed",
            "pre_post_path_size_mtime_match": True,
        },
    }


def _write_tiny_three_fold_qc_release(root: Path) -> tuple[Path, list[bytes]]:
    original_masks: list[bytes] = []
    for fold_id in (1, 2, 3):
        directory = root / f"Fold {fold_id}" / "release_arrays"
        directory.mkdir(parents=True)
        images = np.full((2, 8, 8, 3), 20 * fold_id, dtype=np.uint8)
        masks = np.zeros((2, 8, 8, 6), dtype=np.uint16)
        masks[..., 5] = 1
        masks[0, 1:4, 1:4, 0] = 11
        masks[0, 1:4, 1:4, 5] = 0
        masks[0, 2, 2, 1] = 21  # cross-class overlap; no winner is selected
        masks[1, 4:7, 4:7, 2] = 31
        masks[1, 4:7, 4:7, 5] = 0
        masks[1, 0, 0, 5] = 0  # supplied-background void
        tissues = np.asarray(["Breast", "Colon"], dtype="<U16")
        np.save(directory / "images.npy", images, allow_pickle=False)
        np.save(directory / "masks.npy", masks, allow_pickle=False)
        np.save(directory / "types.npy", tissues, allow_pickle=False)
        original_masks.append((directory / "masks.npy").read_bytes())
    return root, original_masks


def _tree_snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(value for value in root.rglob("*") if value.is_file())
    }


def _validation_cli_arguments(
    project_root: Path,
    source: Path,
    output: Path,
    *,
    max_overlay_patches: int,
) -> list[str]:
    return [
        "data",
        "validate-pannuke",
        "--project-root",
        str(project_root),
        "--root",
        str(source),
        "--output-dir",
        str(output),
        "--max-samples-per-fold",
        "4",
        "--max-overlay-patches",
        str(max_overlay_patches),
    ]


def test_pilot_cli_uses_prebuilt_privacy_gate_without_full_release_revalidation(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import histo_audit.experiment as experiment
    import histo_audit.external_validation as external_validation
    import histo_audit.pannuke as pannuke

    source = _patch_local_release(tmp_path, monkeypatch)
    manifest = tmp_path / "data" / "manifests" / "pannuke" / "manifest.parquet"
    development_manifest = manifest.with_name("pilot_development.parquet")
    gate_certificate = development_manifest.with_suffix(".parquet.metadata.json")
    duplicate_json = tmp_path / "artifacts" / "duplicate_audit" / "duplicates.json"
    for path, payload in (
        (manifest, b"canonical manifest fixture"),
        (development_manifest, b"folds 1 and 2 only"),
        (gate_certificate, b'{"policy":"pre_pilot_privacy_gate_v1"}'),
        (duplicate_json, b'{"status":"complete"}'),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    duplicate_sha256 = hashlib.sha256(duplicate_json.read_bytes()).hexdigest()
    calls: dict[str, Any] = {}

    def forbidden_full_release_validation(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the pilot CLI must not re-read full-release M5 evidence")

    monkeypatch.setattr(
        pannuke,
        "validate_pannuke",
        forbidden_full_release_validation,
    )
    monkeypatch.setattr(
        external_validation,
        "validate_real_dataset_evidence",
        forbidden_full_release_validation,
    )

    def run_pilot(*args: Any, **kwargs: Any) -> Any:
        calls["pilot"] = (args, kwargs)
        return SimpleNamespace(
            run_id="privacy-safe-pilot",
            run_directory=tmp_path / "artifacts" / "runs" / "privacy-safe-pilot",
            metrics_path=tmp_path / "artifacts" / "runs" / "metrics.json",
            report_path=tmp_path / "artifacts" / "runs" / "report.md",
            selected_ids_path=tmp_path / "artifacts" / "runs" / "selected_ids.json",
            audit_sample_count=100,
            final_reference_sample_count=200,
            exact_corruption_count=10,
        )

    monkeypatch.setattr(experiment, "run_pannuke_pilot", run_pilot)

    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "pilot",
            "--project-root",
            str(tmp_path),
            "--data-root",
            str(source),
            "--manifest",
            str(manifest),
            "--development-manifest",
            str(development_manifest),
            "--gate-certificate",
            str(gate_certificate),
            "--duplicate-audit-json",
            str(duplicate_json),
        ],
    )

    assert result.exit_code == 0, result.output
    output = json.loads(result.output)
    assert output["status"] == "completed"
    assert output["completion_stage_candidate"] == "PILOT_COMPLETE"
    assert output["independent_post_seal_inspection_required"] is True
    pilot_args, pilot_kwargs = calls["pilot"]
    assert pilot_args == (gate_certificate.resolve(), manifest.resolve())
    assert pilot_kwargs["development_manifest_source"] == development_manifest.resolve()
    assert pilot_kwargs["expected_data_root"] == source
    assert pilot_kwargs["duplicate_audit_status"] == (f"complete_sha256:{duplicate_sha256}")


def test_build_pilot_development_view_cli_publishes_gate_certificate(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import histo_audit.experiment as experiment

    validation_json = tmp_path / "reports" / "pannuke_validation.json"
    manifest = tmp_path / "data" / "manifests" / "pannuke" / "manifest.parquet"
    duplicate_json = tmp_path / "artifacts" / "duplicate_audit" / "duplicates.json"
    output_path = manifest.with_name("pilot_development.parquet")
    metadata_path = output_path.with_suffix(".parquet.metadata.json")
    calls: dict[str, Any] = {}

    def build(*args: Any, **kwargs: Any) -> Any:
        calls["build"] = (args, kwargs)
        return SimpleNamespace(
            parquet_path=output_path,
            metadata_path=metadata_path,
            canonical_manifest_sha256="a" * 64,
            development_manifest_sha256="b" * 64,
            development_instance_count=123_090,
        )

    monkeypatch.setattr(
        experiment,
        "build_pannuke_pilot_development_manifest_view",
        build,
    )
    result = CliRunner().invoke(
        app,
        [
            "data",
            "build-pilot-development-view",
            "--project-root",
            str(tmp_path),
            "--validation-json",
            str(validation_json),
            "--manifest",
            str(manifest),
            "--duplicate-audit-json",
            str(duplicate_json),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {
        "status": "completed",
        "policy": "pre_pilot_privacy_gate_v1",
        "development_manifest": str(output_path.resolve()),
        "gate_certificate": str(metadata_path.resolve()),
        "canonical_manifest_sha256": "a" * 64,
        "development_manifest_sha256": "b" * 64,
        "development_instance_count": 123_090,
    }
    assert calls["build"] == (
        (
            validation_json.resolve(),
            manifest.resolve(),
            duplicate_json.resolve(),
            output_path.resolve(),
        ),
        {},
    )


def test_acquisition_cli_writes_bound_bundle_and_reports_local_evidence(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import histo_audit.pannuke as pannuke

    source = _patch_local_release(tmp_path, monkeypatch)
    manifest = _acquisition_manifest()
    calls: dict[str, Any] = {}

    def build(project_root: Path, raw_root: Path, *, verification_timestamp_utc: str) -> Any:
        calls["build"] = (project_root, raw_root, verification_timestamp_utc)
        return manifest

    def write_bundle(
        manifest_path: Path,
        report_path: Path,
        payload: Any,
        **kwargs: Any,
    ) -> tuple[Path, Path]:
        calls["write"] = (manifest_path, report_path, payload, kwargs)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        report_path.write_text(
            json.dumps({"status": "passed", "scientific_stage_advanced": False}),
            encoding="utf-8",
        )
        return manifest_path, report_path

    monkeypatch.setattr(pannuke, "build_pannuke_acquisition_manifest", build)
    monkeypatch.setattr(pannuke, "write_acquisition_artifact_bundle", write_bundle)
    result = CliRunner().invoke(
        app,
        [
            "data",
            "verify-pannuke-acquisition",
            "--project-root",
            str(tmp_path),
            "--root",
            str(source),
            "--verification-timestamp-utc",
            "2026-07-18T00:00:00Z",
        ],
    )

    assert result.exit_code == 0, result.output
    output = json.loads(result.output)
    assert output["status"] == "passed"
    assert output["archive_count"] == 3
    assert output["extracted_npy_count"] == 9
    assert output["download_performed"] is False
    assert output["extraction_performed"] is False
    assert output["scientific_stage_advanced"] is False
    assert calls["build"] == (tmp_path.resolve(), source, "2026-07-18T00:00:00Z")
    assert calls["write"][3]["project_root"] == tmp_path.resolve()


def test_acquisition_cli_fails_closed_when_bound_bundle_write_fails(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import histo_audit.pannuke as pannuke

    source = _patch_local_release(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pannuke,
        "build_pannuke_acquisition_manifest",
        lambda *args, **kwargs: _acquisition_manifest(),
    )

    def reject(*args: Any, **kwargs: Any) -> Any:
        raise pannuke.PanNukeAcquisitionError("bound bundle is incomplete or tampered")

    monkeypatch.setattr(pannuke, "write_acquisition_artifact_bundle", reject)
    result = CliRunner().invoke(
        app,
        [
            "data",
            "verify-pannuke-acquisition",
            "--project-root",
            str(tmp_path),
            "--root",
            str(source),
            "--verification-timestamp-utc",
            "2026-07-18T00:00:00Z",
        ],
    )

    assert result.exit_code == 1
    assert "bound bundle is incomplete or tampered" in result.output
    assert '"status": "passed"' not in result.output


def test_acquisition_cli_refuses_outputs_inside_immutable_raw_release(
    tmp_path: Path, monkeypatch: Any
) -> None:
    source = _patch_local_release(tmp_path, monkeypatch)
    result = CliRunner().invoke(
        app,
        [
            "data",
            "verify-pannuke-acquisition",
            "--project-root",
            str(tmp_path),
            "--root",
            str(source),
            "--verification-timestamp-utc",
            "2026-07-18T00:00:00Z",
            "--manifest-output",
            str(source / "forbidden.json"),
        ],
    )

    assert result.exit_code == 1
    assert "inside the immutable raw release" in result.output
    assert not (source / "forbidden.json").exists()


def test_validation_cli_fails_closed_when_qc_bundle_is_incomplete_or_tampered(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import histo_audit.pannuke as pannuke

    source = _patch_local_release(tmp_path, monkeypatch)
    validation = SimpleNamespace(result=SimpleNamespace(root=source))
    calls: dict[str, dict[str, Any]] = {}

    def validate(*args: Any, **kwargs: Any) -> Any:
        calls["validation"] = kwargs
        return validation

    monkeypatch.setattr(pannuke, "validate_pannuke", validate)

    def reject(*args: Any, **kwargs: Any) -> Any:
        calls["qc"] = kwargs
        raise pannuke.MaskQCReportError("mask-QC bundle is partial or tampered")

    monkeypatch.setattr(pannuke, "write_mask_qc_report_bundle", reject)
    result = CliRunner().invoke(
        app,
        [
            "data",
            "validate-pannuke",
            "--project-root",
            str(tmp_path),
            "--root",
            str(source),
            "--output-dir",
            str(tmp_path / "reports"),
        ],
    )

    assert result.exit_code == 1
    assert "mask-QC bundle is partial or tampered" in result.output
    assert '"status": "valid"' not in result.output
    assert calls["validation"]["max_samples_per_fold"] == 100_000
    assert calls["validation"]["max_overlay_patches"] == 24
    assert calls["qc"]["max_overlay_patches"] == 24


def test_validation_cli_end_to_end_writes_full_qc_without_mutating_raw_masks(
    tmp_path: Path,
) -> None:
    source, original_masks = _write_tiny_three_fold_qc_release(tmp_path / "pannuke")
    output = tmp_path / "reports"
    result = CliRunner().invoke(
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
            "--max-samples-per-fold",
            "4",
            "--max-overlay-patches",
            "6",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["validation_scope"] == "full_semantic_scan"
    assert payload["patch_count"] == 6
    assert payload["cross_class_overlap_pixel_count"] == 3
    assert payload["void_pixel_count"] == 3
    assert payload["analysis_excluded_instance_count"] == 6
    assert payload["primary_excluded_instance_count"] == 6
    assert payload["confirmatory_excluded_instance_count"] == 6
    assert payload["analysis_exclusion_reason"] == "touches_cross_class_overlap"
    assert payload["applies_identically_to_primary_and_confirmatory"] is True
    assert payload["no_class_arbitration"] is True
    assert payload["source_masks_modified"] is False
    bundle = Path(payload["qc_bundle"])
    assert {path.name for path in bundle.iterdir()} == {
        "artifact_manifest.json",
        "pannuke_mask_qc.json",
        "pannuke_mask_qc.md",
        "pannuke_mask_qc_instances.csv",
        "pannuke_mask_qc_overlay_selection.json",
        "pannuke_mask_qc_overlays.png",
        "pannuke_mask_qc_patches.csv",
    }
    for fold_id, expected in zip((1, 2, 3), original_masks, strict=True):
        assert (
            source / f"Fold {fold_id}" / "release_arrays" / "masks.npy"
        ).read_bytes() == expected


def test_validation_cli_no_flag_defaults_are_idempotent_and_conflict_preserves_every_byte(
    tmp_path: Path,
) -> None:
    source, _ = _write_tiny_three_fold_qc_release(tmp_path / "pannuke")
    output = tmp_path / "reports"
    runner = CliRunner()
    default_arguments = [
        "data",
        "validate-pannuke",
        "--project-root",
        str(tmp_path),
        "--root",
        str(source),
        "--output-dir",
        str(output),
    ]
    initial = runner.invoke(app, default_arguments)
    assert initial.exit_code == 0, initial.output
    assert json.loads(initial.output)["publication"] == "published"
    assert (
        json.loads((output / "pannuke_validation.json").read_text(encoding="utf-8"))[
            "anomaly_overlay_selection"
        ]["requested_max_patches"]
        == 24
    )
    expected = _tree_snapshot(output)
    assert len(expected) == 11

    identical = runner.invoke(app, default_arguments)
    assert identical.exit_code == 0, identical.output
    assert json.loads(identical.output)["publication"] == "idempotent"
    assert _tree_snapshot(output) == expected

    conflicting = runner.invoke(
        app,
        [*default_arguments, "--max-overlay-patches", "3"],
    )
    assert conflicting.exit_code == 1
    assert "immutable PanNuke validation artifacts differ" in conflicting.output
    assert _tree_snapshot(output) == expected


def test_validation_cli_rolls_back_an_injected_partial_publication(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import histo_audit.cli as cli_module

    source, _ = _write_tiny_three_fold_qc_release(tmp_path / "pannuke")
    output = tmp_path / "reports"
    raw_before = _tree_snapshot(source)
    real_promote = cli_module._promote_staged_artifact
    calls = 0

    def fail_before_success_marker(source_path: Path, destination_path: Path) -> list[Any]:
        nonlocal calls
        calls += 1
        if calls == 5:
            raise OSError("injected publication interruption")
        return real_promote(source_path, destination_path)

    monkeypatch.setattr(cli_module, "_promote_staged_artifact", fail_before_success_marker)
    failed = CliRunner().invoke(
        app,
        _validation_cli_arguments(tmp_path, source, output, max_overlay_patches=4),
    )

    assert failed.exit_code == 1
    assert "injected publication interruption" in failed.output
    assert _tree_snapshot(output) == {}
    assert _tree_snapshot(source) == raw_before
    assert not (tmp_path / "artifacts" / ".pannuke_validation_publish.lock").exists()


def test_validation_cli_rejects_an_active_publication_lock(tmp_path: Path) -> None:
    import histo_audit.cli as cli_module
    from histo_audit.pannuke.publication import ExclusiveBundlePublicationLock

    source, _ = _write_tiny_three_fold_qc_release(tmp_path / "pannuke")
    output = tmp_path / "reports"
    targets = cli_module._validation_command_destinations(tmp_path.resolve(), output)

    with ExclusiveBundlePublicationLock(targets, role="test CLI validation"):
        blocked = CliRunner().invoke(
            app,
            _validation_cli_arguments(tmp_path, source, output, max_overlay_patches=4),
        )

    assert blocked.exit_code == 1
    assert isinstance(blocked.exception, FileExistsError)
    assert "another PanNuke CLI validation publication is active" in str(blocked.exception)
    assert not output.exists()


def test_validation_cli_shares_output_lock_with_direct_qc_library_writer(
    tmp_path: Path,
) -> None:
    from histo_audit.pannuke.publication import ExclusiveBundlePublicationLock

    source, _ = _write_tiny_three_fold_qc_release(tmp_path / "pannuke")
    output = tmp_path / "reports"
    qc_bundle = output / "pannuke_qc"

    with ExclusiveBundlePublicationLock((output, qc_bundle), role="direct PanNuke mask-QC bundle"):
        blocked = CliRunner().invoke(
            app,
            _validation_cli_arguments(tmp_path, source, output, max_overlay_patches=4),
        )

    assert blocked.exit_code == 1
    assert isinstance(blocked.exception, FileExistsError)
    assert "another PanNuke CLI validation publication is active" in str(blocked.exception)
    assert not output.exists()


def test_validation_cli_rejects_final_outputs_inside_its_staging_directory(
    tmp_path: Path,
) -> None:
    source, _ = _write_tiny_three_fold_qc_release(tmp_path / "pannuke")
    output = tmp_path / "artifacts" / ".pannuke_validation_staging"

    failed = CliRunner().invoke(
        app,
        _validation_cli_arguments(tmp_path, source, output, max_overlay_patches=4),
    )

    assert failed.exit_code == 1
    assert "validation staging directory" in failed.output
    assert "overlap" in failed.output
    assert not output.exists()


def test_validation_cli_rollback_preserves_a_foreign_replacement(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import histo_audit.cli as cli_module

    source, _ = _write_tiny_three_fold_qc_release(tmp_path / "pannuke")
    output = tmp_path / "reports"
    foreign_payload = b"foreign concurrent CLI validation owner\n"
    real_promote = cli_module._promote_staged_artifact
    calls = 0

    def replace_first_then_fail(source_path: Path, destination_path: Path) -> list[Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second-promotion failure")
        publications = real_promote(source_path, destination_path)
        destination_path.unlink()
        destination_path.write_bytes(foreign_payload)
        return publications

    monkeypatch.setattr(cli_module, "_promote_staged_artifact", replace_first_then_fail)
    failed = CliRunner().invoke(
        app,
        _validation_cli_arguments(tmp_path, source, output, max_overlay_patches=4),
    )

    assert failed.exit_code == 1
    assert calls == 2
    assert "ownership-safe rollback was incomplete" in failed.output
    assert (output / "pannuke_validation.md").read_bytes() == foreign_payload
    assert not (output / "pannuke_validation.json").exists()
    assert not (output / "pannuke_overlay_grid.png").exists()
    assert not (output / "raw_files_sha256.csv").exists()
    assert not (output / "pannuke_qc").exists()


def test_validation_cli_final_consistency_check_preserves_foreign_replacement(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import histo_audit.cli as cli_module

    source, _ = _write_tiny_three_fold_qc_release(tmp_path / "pannuke")
    output = tmp_path / "reports"
    foreign_payload = b"foreign CLI validation after success-marker publication\n"
    real_promote = cli_module._promote_staged_artifact
    calls = 0

    def replace_after_success_marker(source_path: Path, destination_path: Path) -> list[Any]:
        nonlocal calls
        calls += 1
        publications = real_promote(source_path, destination_path)
        if calls == 5:
            markdown = output / "pannuke_validation.md"
            markdown.unlink()
            markdown.write_bytes(foreign_payload)
        return publications

    monkeypatch.setattr(cli_module, "_promote_staged_artifact", replace_after_success_marker)
    failed = CliRunner().invoke(
        app,
        _validation_cli_arguments(tmp_path, source, output, max_overlay_patches=4),
    )

    assert failed.exit_code == 1
    assert calls == 5
    assert "ownership-safe rollback was incomplete" in failed.output
    assert (output / "pannuke_validation.md").read_bytes() == foreign_payload
    assert not (output / "pannuke_validation.json").exists()
    assert not (output / "pannuke_overlay_grid.png").exists()
    assert not (output / "raw_files_sha256.csv").exists()
    assert not (output / "pannuke_qc").exists()


def test_validation_cli_rejects_a_partial_existing_set_without_modification(
    tmp_path: Path,
) -> None:
    source, _ = _write_tiny_three_fold_qc_release(tmp_path / "pannuke")
    output = tmp_path / "reports"
    output.mkdir()
    partial = output / "pannuke_validation.md"
    partial.write_text("do not modify\n", encoding="utf-8")
    expected = _tree_snapshot(output)

    failed = CliRunner().invoke(
        app,
        _validation_cli_arguments(tmp_path, source, output, max_overlay_patches=4),
    )

    assert failed.exit_code == 1
    assert "artifact set is partial" in failed.output
    assert _tree_snapshot(output) == expected


def test_validation_cli_treats_a_broken_final_symlink_as_occupied(tmp_path: Path) -> None:
    source, _ = _write_tiny_three_fold_qc_release(tmp_path / "pannuke")
    output = tmp_path / "reports"
    output.mkdir()
    missing_target = tmp_path / "foreign-owner-missing.md"
    broken_final = output / "pannuke_validation.md"
    try:
        broken_final.symlink_to(missing_target)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable in this Windows environment: {error}")

    failed = CliRunner().invoke(
        app,
        _validation_cli_arguments(tmp_path, source, output, max_overlay_patches=4),
    )

    assert failed.exit_code == 1
    assert "artifact set is partial" in failed.output
    assert broken_final.is_symlink()
    assert not broken_final.exists()
    assert not missing_target.exists()
    assert not (output / "pannuke_validation.json").exists()


def test_validation_cli_final_inventory_check_blocks_reporter_time_raw_mutation(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import histo_audit.pannuke as pannuke

    source, _ = _write_tiny_three_fold_qc_release(tmp_path / "pannuke")
    output = tmp_path / "reports"
    initial = CliRunner().invoke(
        app,
        _validation_cli_arguments(tmp_path, source, output, max_overlay_patches=4),
    )
    assert initial.exit_code == 0, initial.output
    expected = _tree_snapshot(output)
    real_writer = pannuke.write_mask_qc_report_bundle

    def mutate_after_report(*args: Any, **kwargs: Any) -> Any:
        artifacts = real_writer(*args, **kwargs)
        mask_path = source / "Fold 1" / "release_arrays" / "masks.npy"
        masks = np.load(mask_path, mmap_mode="r+", allow_pickle=False)
        masks[1, 7, 7, 5] = 0
        masks.flush()
        return artifacts

    monkeypatch.setattr(pannuke, "write_mask_qc_report_bundle", mutate_after_report)
    failed = CliRunner().invoke(
        app,
        _validation_cli_arguments(tmp_path, source, output, max_overlay_patches=4),
    )

    assert failed.exit_code == 1
    assert "raw inventory changed after semantic validation" in failed.output
    assert _tree_snapshot(output) == expected


def test_validation_cli_publish_time_raw_mutation_rolls_back_complete_bundle(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import histo_audit.cli as cli_module

    source, _ = _write_tiny_three_fold_qc_release(tmp_path / "pannuke")
    output = tmp_path / "reports"
    image_path = source / "Fold 1" / "release_arrays" / "images.npy"
    real_promote = cli_module._promote_staged_artifact
    mutation_injected = False

    def mutate_at_first_promotion(source_path: Path, destination_path: Path) -> list[Any]:
        nonlocal mutation_injected
        if not mutation_injected:
            mutation_injected = True
            images = np.load(image_path, mmap_mode="r+", allow_pickle=False)
            images[0, 0, 0, 0] = (int(images[0, 0, 0, 0]) + 1) % 256
            images.flush()
            del images
        return real_promote(source_path, destination_path)

    monkeypatch.setattr(cli_module, "_promote_staged_artifact", mutate_at_first_promotion)
    failed = CliRunner().invoke(
        app,
        _validation_cli_arguments(tmp_path, source, output, max_overlay_patches=4),
    )

    assert failed.exit_code == 1
    assert mutation_injected
    assert "raw inventory changed after semantic validation" in failed.output
    assert _tree_snapshot(output) == {}
    assert not (tmp_path / "artifacts" / ".pannuke_validation_publish.lock").exists()


def test_pannuke_data_commands_reject_raw_tree_destinations_before_any_write(
    tmp_path: Path,
) -> None:
    source, _ = _write_tiny_three_fold_qc_release(tmp_path / "pannuke")
    expected = _tree_snapshot(source)
    runner = CliRunner()
    destinations = (source, source / "derived", source / "nested" / ".." / "derived")
    for destination in destinations:
        validation = runner.invoke(
            app,
            _validation_cli_arguments(
                tmp_path,
                source,
                destination,
                max_overlay_patches=4,
            ),
        )
        assert validation.exit_code == 1
        assert "immutable raw release" in validation.output
        assert _tree_snapshot(source) == expected

    manifest = runner.invoke(
        app,
        [
            "data",
            "build-manifest",
            "--project-root",
            str(tmp_path),
            "--root",
            str(source),
            "--output-dir",
            str(source / "manifest"),
        ],
    )
    assert manifest.exit_code == 1
    assert "immutable raw release" in manifest.output
    assert _tree_snapshot(source) == expected

    duplicate_output = runner.invoke(
        app,
        [
            "data",
            "audit-duplicates",
            "--project-root",
            str(tmp_path),
            "--root",
            str(source),
            "--output-dir",
            str(source / "duplicates"),
        ],
    )
    assert duplicate_output.exit_code == 1
    assert "immutable raw release" in duplicate_output.output
    assert _tree_snapshot(source) == expected

    duplicate_ranking = runner.invoke(
        app,
        [
            "data",
            "audit-duplicates",
            "--project-root",
            str(tmp_path),
            "--root",
            str(source),
            "--output-dir",
            str(tmp_path / "duplicates"),
            "--rankings-csv",
            str(source / "rankings.csv"),
        ],
    )
    assert duplicate_ranking.exit_code == 1
    assert "immutable raw release" in duplicate_ranking.output
    assert _tree_snapshot(source) == expected
