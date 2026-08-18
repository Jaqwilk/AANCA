"""Focused manifest regression tests for anomaly-safe PanNuke mask handling."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import histo_audit.pannuke.manifest as manifest_module
from histo_audit.pannuke import (
    PanNukeSemanticsError,
    build_nucleus_manifest,
    sha256_file,
    validate_manifest_invariants,
    validate_pannuke,
)


def _write_qc_release(root: Path) -> tuple[Path, Path]:
    fold = root / "Fold 1" / "release_arrays"
    fold.mkdir(parents=True)
    height, width = 13, 11
    y_grid, x_grid = np.mgrid[:height, :width]
    images = np.zeros((3, height, width, 3), dtype=np.uint8)
    images[..., 0] = (x_grid * 13 + y_grid * 3)[None, ...] % 256
    images[..., 1] = (y_grid * 17 + 11)[None, ...] % 256
    images[0, ..., 2] = 37
    images[1, ..., 2] = 91
    images[2, ..., 2] = 143

    masks = np.zeros((3, height, width, 6), dtype=np.int32)
    masks[0, 2:5, 2:5, 0] = 11
    masks[0, 4:7, 4:7, 1] = 22
    masks[0, 8:10, 7:9, 2] = 33
    masks[1, 5:7, 5:7, 4] = 44
    masks[2, 3:6, 3:6, 3] = 55
    occupied = np.any(masks[..., :5] > 0, axis=-1)
    masks[..., 5] = (~occupied).astype(np.int32)
    # One supplied-background void. It remains unlabelled and is never repaired.
    masks[0, 0, 0, 5] = 0
    # One positive+supplied-background pixel is flagged, never class-arbitrated.
    masks[2, 4, 4, 5] = 1
    tissues = np.asarray(["Breast", "Colon", "Liver"], dtype="<U16")
    np.save(fold / "pixels.npy", images)
    mask_path = fold / "labels.npy"
    np.save(mask_path, masks)
    np.save(fold / "organs.npy", tissues)
    return root, mask_path


def _replace_column(table: pa.Table, name: str, values: list[object]) -> pa.Table:
    index = table.schema.get_field_index(name)
    field = table.schema.field(index)
    return table.set_column(index, field, pa.array(values, type=field.type))


def test_manifest_preserves_overlap_identities_void_and_non_touching_instances(
    tmp_path: Path,
) -> None:
    root, mask_path = _write_qc_release(tmp_path / "pannuke")
    raw_paths = sorted(root.rglob("*.npy"))
    hashes_before = {path: sha256_file(path) for path in raw_paths}
    mask_before = np.load(mask_path).copy()
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        explicit_background_channel=5,
        expected_fold_ids=(1,),
        max_overlay_patches=2,
    )

    first = build_nucleus_manifest(validation, tmp_path / "manifest_a", batch_rows=2)
    second = build_nucleus_manifest(validation, tmp_path / "manifest_b", batch_rows=2)
    table = pq.read_table(first.parquet_path)
    second_table = pq.read_table(second.parquet_path)
    validate_manifest_invariants(table, validation=validation.result)

    assert table.equals(second_table)
    assert first.sha256 == second.sha256
    assert np.array_equal(np.load(mask_path), mask_before)
    assert {path: sha256_file(path) for path in raw_paths} == hashes_before

    frame = table.to_pandas()
    assert len(frame) == 5
    patch_zero = frame[frame["source_patch_index"] == 0].copy()
    assert len(patch_zero) == 3
    assert patch_zero["patch_has_overlap"].all()
    assert patch_zero["patch_has_void"].all()
    assert not patch_zero["patch_has_positive_and_background"].any()
    assert (patch_zero["patch_overlap_pixel_count"] == 1).all()
    assert (patch_zero["patch_void_pixel_count"] == 1).all()
    assert np.allclose(patch_zero["patch_void_pixel_rate"], 1 / (13 * 11))
    assert (patch_zero["patch_positive_occupied_pixel_count"] == 21).all()
    assert (patch_zero["patch_supplied_background_pixel_count"] == 121).all()
    assert (patch_zero["patch_positive_background_conflict_pixel_count"] == 0).all()

    touching = patch_zero[patch_zero["cross_class_overlap_touching"]]
    assert set(touching["instance_id"].astype(int)) == {11, 22}
    assert (touching["overlap_pixel_count_for_instance"] == 1).all()
    assert not touching["primary_eligible"].any()
    assert not touching["confirmatory_eligible"].any()
    assert set(touching["qc_exclusion_reason"]) == {"touches_cross_class_overlap"}
    assert set(touching["nucleus_class_index"].astype(int)) == {0, 1}
    for _, row in touching.iterrows():
        assert "touches_cross_class_overlap" in row["quality_flags"]
        assert "patch_contains_cross_class_overlap" in row["quality_flags"]
        assert "patch_contains_unlabeled_void" in row["quality_flags"]
        assert list(row["overlap_class_channel_indices"]) == [0, 1]
        evidence = list(
            zip(
                row["overlap_instance_channel_indices"],
                row["overlap_instance_ids"],
                row["overlap_instance_pixel_counts"],
                strict=True,
            )
        )
        assert evidence == [(0, 11, 1), (1, 22, 1)]

    untouched = patch_zero[patch_zero["instance_id"] == 33].iloc[0]
    assert not bool(untouched["cross_class_overlap_touching"])
    assert bool(untouched["primary_eligible"])
    assert bool(untouched["confirmatory_eligible"])
    assert untouched["qc_exclusion_reason"] is None
    assert list(untouched["overlap_class_channel_indices"]) == []
    assert "patch_contains_cross_class_overlap" in untouched["quality_flags"]
    assert "touches_cross_class_overlap" not in untouched["quality_flags"]

    normal = frame[frame["source_patch_index"] == 1].iloc[0]
    assert not bool(normal["patch_has_overlap"])
    assert not bool(normal["patch_has_void"])
    assert not bool(normal["patch_has_positive_and_background"])
    assert bool(normal["primary_eligible"])
    assert bool(normal["confirmatory_eligible"])
    assert normal["qc_exclusion_reason"] is None
    assert not {
        "patch_contains_cross_class_overlap",
        "patch_contains_unlabeled_void",
        "patch_contains_positive_and_background",
        "touches_cross_class_overlap",
    }.intersection(normal["quality_flags"])

    positive_background = frame[frame["source_patch_index"] == 2].iloc[0]
    assert bool(positive_background["patch_has_positive_and_background"])
    assert positive_background["patch_positive_background_conflict_pixel_count"] == 1
    assert bool(positive_background["positive_background_touching"])
    assert positive_background["positive_background_pixel_count_for_instance"] == 1
    assert "patch_contains_positive_and_background" in positive_background["quality_flags"]
    assert "touches_positive_and_background" in positive_background["quality_flags"]
    assert bool(positive_background["primary_eligible"])
    assert bool(positive_background["confirmatory_eligible"])
    assert positive_background["qc_exclusion_reason"] is None

    # Void is patch-level unlabeled evidence, never a synthetic nucleus class row.
    assert set(frame["nucleus_class_name"]).isdisjoint({"background", "void", "unlabeled"})
    metadata = table.schema.metadata or {}
    assert metadata[b"cross_class_overlap_exclusion_reason"] == b"touches_cross_class_overlap"
    assert metadata[b"eligibility_policy"] == (
        b"one_identical_primary_and_confirmatory_instance_mask"
    )
    assert metadata[b"void_pixel_policy"].startswith(b"retain_as_unlabeled_void")


def test_manifest_invariants_reject_qc_tampering_and_changed_raw_source(tmp_path: Path) -> None:
    root, mask_path = _write_qc_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        explicit_background_channel=5,
        expected_fold_ids=(1,),
        max_overlay_patches=1,
    )
    manifest = build_nucleus_manifest(validation, tmp_path / "manifest", batch_rows=2)
    table = pq.read_table(manifest.parquet_path)

    eligible = table["confirmatory_eligible"].to_pylist()
    touching = table["cross_class_overlap_touching"].to_pylist()
    eligible[touching.index(True)] = True
    with pytest.raises(ValueError, match="eligibility masks must be identical"):
        validate_manifest_invariants(_replace_column(table, "confirmatory_eligible", eligible))

    primary_eligible = table["primary_eligible"].to_pylist()
    primary_eligible[touching.index(True)] = True
    with pytest.raises(ValueError, match="eligibility masks must be identical"):
        validate_manifest_invariants(_replace_column(table, "primary_eligible", primary_eligible))

    void_counts = table["patch_void_pixel_count"].to_pylist()
    void_counts[0] = 2
    with pytest.raises(
        ValueError,
        match=r"patch-level QC field|does not match its exact count",
    ):
        validate_manifest_invariants(_replace_column(table, "patch_void_pixel_count", void_counts))

    changed = np.load(mask_path)
    changed[0, 1, 1, 5] = 0
    np.save(mask_path, changed)
    with pytest.raises(PanNukeSemanticsError, match="raw file changed"):
        validate_manifest_invariants(table, validation=validation.result)


def test_manifest_fails_closed_on_incomplete_qc_or_changed_policy(tmp_path: Path) -> None:
    root, _ = _write_qc_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        explicit_background_channel=5,
        expected_fold_ids=(1,),
        max_overlay_patches=1,
    )
    result = validation.result
    fold = result.fold_validation[0]
    incomplete_qc = replace(fold.mask_qc, patches=fold.mask_qc.patches[:-1])
    incomplete = replace(result, fold_validation=(replace(fold, mask_qc=incomplete_qc),))
    with pytest.raises(PanNukeSemanticsError, match="does not contain every source patch"):
        build_nucleus_manifest(incomplete, tmp_path / "incomplete")

    changed_policy = replace(
        result,
        qc_policy=replace(
            result.qc_policy,
            analysis_instance_exclusion_reason="arbitrate_overlap",
        ),
    )
    with pytest.raises(PanNukeSemanticsError, match="fixed anomaly-safe manifest policy"):
        build_nucleus_manifest(changed_policy, tmp_path / "policy")


def test_manifest_transaction_leaves_no_outputs_when_raw_changes_during_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, mask_path = _write_qc_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        explicit_background_channel=5,
        expected_fold_ids=(1,),
    )
    original = manifest_module._write_summary_csv

    def write_then_mutate(*args: object, **kwargs: object) -> None:
        original(*args, **kwargs)
        masks = np.load(mask_path)
        masks[0, 1, 1, 5] = 0
        np.save(mask_path, masks)

    monkeypatch.setattr(manifest_module, "_write_summary_csv", write_then_mutate)
    output = tmp_path / "manifest"
    with pytest.raises(PanNukeSemanticsError, match="raw file changed"):
        build_nucleus_manifest(validation, output, batch_rows=2)
    assert not (output / "pannuke_nucleus_manifest.parquet").exists()
    assert not (output / "pannuke_manifest_summary.csv").exists()
    assert not list(output.glob(".pannuke-manifest-stage-*"))


@pytest.mark.parametrize("inventory_change", ("addition", "removal"))
def test_manifest_full_inventory_change_blocks_publication(
    tmp_path: Path,
    inventory_change: str,
) -> None:
    root, _ = _write_qc_release(tmp_path / "pannuke")
    sidecar = root / "inventory-only.txt"
    if inventory_change == "removal":
        sidecar.write_text("bound before validation\n", encoding="utf-8")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        explicit_background_channel=5,
        expected_fold_ids=(1,),
    )
    if inventory_change == "addition":
        sidecar.write_text("added after validation\n", encoding="utf-8")
    else:
        sidecar.unlink()

    output = tmp_path / f"manifest-{inventory_change}"
    with pytest.raises(PanNukeSemanticsError, match="raw inventory changed"):
        build_nucleus_manifest(validation, output, batch_rows=2)
    assert not (output / "pannuke_nucleus_manifest.parquet").exists()
    assert not (output / "pannuke_manifest_summary.csv").exists()


def test_manifest_publish_time_raw_mutation_rolls_back_complete_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, mask_path = _write_qc_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        explicit_background_channel=5,
        expected_fold_ids=(1,),
    )
    output = tmp_path / "manifest-publish-race"
    real_promote = manifest_module._promote_staged_manifest_file
    mutation_injected = False

    def mutate_at_first_promotion(source: Path, destination: Path) -> manifest_module.PublishedPath:
        nonlocal mutation_injected
        if not mutation_injected:
            mutation_injected = True
            masks = np.load(mask_path, mmap_mode="r+", allow_pickle=False)
            masks[0, 0, 0, 5] = 1
            masks.flush()
            del masks
        return real_promote(source, destination)

    monkeypatch.setattr(manifest_module, "_promote_staged_manifest_file", mutate_at_first_promotion)
    with pytest.raises(PanNukeSemanticsError, match="raw inventory changed"):
        build_nucleus_manifest(validation, output, batch_rows=2)

    assert mutation_injected
    assert not (output / "pannuke_nucleus_manifest.parquet").exists()
    assert not (output / "pannuke_manifest_summary.csv").exists()
    assert not list(output.glob(".pannuke-manifest-stage-*"))


def test_manifest_concurrent_final_is_never_overwritten_or_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _write_qc_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        explicit_background_channel=5,
        expected_fold_ids=(1,),
    )
    output = tmp_path / "manifest-concurrent-final"
    foreign = b"foreign concurrent summary\n"
    real_promote = manifest_module._promote_staged_manifest_file
    injected = False

    def insert_foreign_then_promote(
        staged: Path, destination: Path
    ) -> manifest_module.PublishedPath:
        nonlocal injected
        if not injected:
            injected = True
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(foreign)
        return real_promote(staged, destination)

    monkeypatch.setattr(
        manifest_module, "_promote_staged_manifest_file", insert_foreign_then_promote
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_nucleus_manifest(validation, output, batch_rows=2)

    assert injected
    assert (output / "pannuke_manifest_summary.csv").read_bytes() == foreign
    assert not (output / "pannuke_nucleus_manifest.parquet").exists()


def test_manifest_rollback_preserves_foreign_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _write_qc_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        explicit_background_channel=5,
        expected_fold_ids=(1,),
    )
    output = tmp_path / "manifest-foreign-replacement"
    foreign = b"foreign replacement\n"
    real_promote = manifest_module._promote_staged_manifest_file
    calls = 0

    def replace_first_then_fail_second(
        staged: Path, destination: Path
    ) -> manifest_module.PublishedPath:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second promotion failure")
        publication = real_promote(staged, destination)
        destination.unlink()
        destination.write_bytes(foreign)
        return publication

    monkeypatch.setattr(
        manifest_module, "_promote_staged_manifest_file", replace_first_then_fail_second
    )
    with pytest.raises(RuntimeError, match="ownership-safe rollback was incomplete"):
        build_nucleus_manifest(validation, output, batch_rows=2)

    assert (output / "pannuke_manifest_summary.csv").read_bytes() == foreign
    assert not (output / "pannuke_nucleus_manifest.parquet").exists()


def test_manifest_final_readback_preserves_foreign_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _write_qc_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        explicit_background_channel=5,
        expected_fold_ids=(1,),
    )
    output = tmp_path / "manifest-final-readback-replacement"
    summary = output / "pannuke_manifest_summary.csv"
    parquet = output / "pannuke_nucleus_manifest.parquet"
    foreign = b"foreign replacement after complete pair publication\n"
    real_promote = manifest_module._promote_staged_manifest_file
    calls = 0

    def replace_first_after_second_promotion(
        staged: Path, destination: Path
    ) -> manifest_module.PublishedPath:
        nonlocal calls
        calls += 1
        publication = real_promote(staged, destination)
        if calls == 2:
            summary.unlink()
            summary.write_bytes(foreign)
        return publication

    monkeypatch.setattr(
        manifest_module,
        "_promote_staged_manifest_file",
        replace_first_after_second_promotion,
    )
    with pytest.raises(RuntimeError, match="ownership-safe rollback was incomplete"):
        build_nucleus_manifest(validation, output, batch_rows=2)

    assert calls == 2
    assert summary.read_bytes() == foreign
    assert not parquet.exists()
    assert not list(output.glob(".pannuke-manifest-stage-*"))


def test_manifest_post_publication_failure_rolls_back_complete_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _write_qc_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        explicit_background_channel=5,
        expected_fold_ids=(1,),
    )
    output = tmp_path / "manifest-post-publication-failure"

    def fail_artifact_result(**kwargs: object) -> object:
        del kwargs
        raise OSError("injected post-publication artifact-result failure")

    monkeypatch.setattr(manifest_module, "ManifestArtifacts", fail_artifact_result)
    with pytest.raises(OSError, match="post-publication artifact-result failure"):
        build_nucleus_manifest(validation, output, batch_rows=2)

    assert not (output / "pannuke_nucleus_manifest.parquet").exists()
    assert not (output / "pannuke_manifest_summary.csv").exists()
    assert not list(output.glob(".pannuke-manifest-stage-*"))


def test_manifest_transaction_completion_preserves_foreign_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _write_qc_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        explicit_background_channel=5,
        expected_fold_ids=(1,),
    )
    output = tmp_path / "manifest-transaction-completion-replacement"
    summary = output / "pannuke_manifest_summary.csv"
    parquet = output / "pannuke_nucleus_manifest.parquet"
    foreign = b"foreign replacement during artifact-result construction\n"
    real_artifact_result = manifest_module.ManifestArtifacts

    def replace_during_artifact_result(**kwargs: object) -> object:
        summary.unlink()
        summary.write_bytes(foreign)
        return real_artifact_result(**kwargs)

    monkeypatch.setattr(
        manifest_module,
        "ManifestArtifacts",
        replace_during_artifact_result,
    )
    with pytest.raises(
        RuntimeError,
        match="transaction failed and ownership-safe rollback was incomplete",
    ):
        build_nucleus_manifest(validation, output, batch_rows=2)

    assert summary.read_bytes() == foreign
    assert not parquet.exists()
    assert not list(output.glob(".pannuke-manifest-stage-*"))


def test_manifest_transaction_rolls_back_csv_and_second_publish_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _write_qc_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        explicit_background_channel=5,
        expected_fold_ids=(1,),
    )
    csv_failure_output = tmp_path / "csv-failure"

    def fail_csv(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("simulated CSV staging failure")

    monkeypatch.setattr(manifest_module, "_write_summary_csv", fail_csv)
    with pytest.raises(OSError, match="simulated CSV staging failure"):
        build_nucleus_manifest(validation, csv_failure_output, batch_rows=2)
    assert not (csv_failure_output / "pannuke_nucleus_manifest.parquet").exists()
    assert not (csv_failure_output / "pannuke_manifest_summary.csv").exists()

    monkeypatch.undo()
    publish_failure_output = tmp_path / "publish-failure"
    final_parquet = publish_failure_output / "pannuke_nucleus_manifest.parquet"
    real_promote = manifest_module._promote_staged_manifest_file

    def fail_second_replace(source: Path, destination: Path) -> manifest_module.PublishedPath:
        if Path(destination) == final_parquet:
            raise OSError("simulated second publish failure")
        return real_promote(source, destination)

    monkeypatch.setattr(manifest_module, "_promote_staged_manifest_file", fail_second_replace)
    with pytest.raises(OSError, match="simulated second publish failure"):
        build_nucleus_manifest(validation, publish_failure_output, batch_rows=2)
    assert not (publish_failure_output / "pannuke_nucleus_manifest.parquet").exists()
    assert not (publish_failure_output / "pannuke_manifest_summary.csv").exists()
    assert not list(publish_failure_output.glob(".pannuke-manifest-stage-*"))


def test_manifest_no_overwrite_conflict_and_identical_rerun_is_mtime_stable(
    tmp_path: Path,
) -> None:
    root, _ = _write_qc_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        explicit_background_channel=5,
        expected_fold_ids=(1,),
    )
    conflict = tmp_path / "conflict"
    conflict.mkdir()
    conflict_parquet = conflict / "pannuke_nucleus_manifest.parquet"
    conflict_summary = conflict / "pannuke_manifest_summary.csv"
    conflict_parquet.write_bytes(b"pre-existing parquet")
    conflict_summary.write_bytes(b"pre-existing summary")
    with pytest.raises(FileExistsError, match="non-identical existing"):
        build_nucleus_manifest(validation, conflict, batch_rows=2)
    assert conflict_parquet.read_bytes() == b"pre-existing parquet"
    assert conflict_summary.read_bytes() == b"pre-existing summary"

    incomplete = tmp_path / "incomplete-existing"
    incomplete.mkdir()
    incomplete_parquet = incomplete / "pannuke_nucleus_manifest.parquet"
    incomplete_parquet.write_bytes(b"pre-existing partial")
    with pytest.raises(FileExistsError, match="incomplete existing"):
        build_nucleus_manifest(validation, incomplete, batch_rows=2)
    assert incomplete_parquet.read_bytes() == b"pre-existing partial"
    assert not (incomplete / "pannuke_manifest_summary.csv").exists()

    output = tmp_path / "idempotent"
    first = build_nucleus_manifest(validation, output, batch_rows=2)
    parquet = first.parquet_path
    summary = first.summary_csv_path
    parquet_bytes = parquet.read_bytes()
    summary_bytes = summary.read_bytes()
    historical_ns = min(parquet.stat().st_mtime_ns, summary.stat().st_mtime_ns) - 10_000_000_000
    os.utime(parquet, ns=(historical_ns, historical_ns))
    os.utime(summary, ns=(historical_ns, historical_ns))
    mtimes_before = (parquet.stat().st_mtime_ns, summary.stat().st_mtime_ns)

    second = build_nucleus_manifest(validation, output, batch_rows=2)
    assert second == first
    assert parquet.read_bytes() == parquet_bytes
    assert summary.read_bytes() == summary_bytes
    assert (parquet.stat().st_mtime_ns, summary.stat().st_mtime_ns) == mtimes_before


def test_manifest_bundle_lock_blocks_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _write_qc_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        explicit_background_channel=5,
        expected_fold_ids=(1,),
    )
    output = tmp_path / "manifest-active-bundle-lock"
    final_paths = (
        output / "pannuke_nucleus_manifest.parquet",
        output / "pannuke_manifest_summary.csv",
    )
    partially_overlapping_bundle = (final_paths[1], output / "unrelated-report.json")

    def fail_if_staging_starts(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("manifest construction started while bundle lock was held")

    monkeypatch.setattr(manifest_module, "_write_parquet_streaming", fail_if_staging_starts)
    with (
        manifest_module.ExclusiveBundlePublicationLock(
            partially_overlapping_bundle,
            role="test manifest",
        ),
        pytest.raises(FileExistsError, match="publication is active"),
    ):
        build_nucleus_manifest(validation, output, batch_rows=2)

    assert not final_paths[0].exists()
    assert not final_paths[1].exists()
    assert not list(output.glob(".pannuke-manifest-stage-*"))


def test_manifest_rejects_colliding_bundle_destinations(tmp_path: Path) -> None:
    root, _ = _write_qc_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        explicit_background_channel=5,
        expected_fold_ids=(1,),
    )
    destination = tmp_path / "one-final-path"

    with pytest.raises(PanNukeSemanticsError, match="distinct and non-colliding"):
        manifest_module._verify_manifest_publication_destinations(
            (destination, destination),
            validation=validation.result,
        )


def test_manifest_rejects_output_inside_immutable_raw_root(tmp_path: Path) -> None:
    root, _ = _write_qc_release(tmp_path / "pannuke")
    raw_paths = sorted(root.rglob("*.npy"))
    hashes_before = {path: sha256_file(path) for path in raw_paths}
    direct_output = root / "direct-derived-manifest"
    with pytest.raises(PanNukeSemanticsError, match="outside the immutable PanNuke raw root"):
        build_nucleus_manifest(
            root,
            direct_output,
            explicit_background_channel=5,
        )
    assert not direct_output.exists()
    assert {path: sha256_file(path) for path in raw_paths} == hashes_before

    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        explicit_background_channel=5,
        expected_fold_ids=(1,),
    )
    output = root / "derived-manifest"
    with pytest.raises(PanNukeSemanticsError, match="outside the immutable PanNuke raw root"):
        build_nucleus_manifest(validation, output)
    assert not output.exists()
    assert {path: sha256_file(path) for path in raw_paths} == hashes_before


def test_manifest_rejects_final_symlink_into_immutable_raw_root(tmp_path: Path) -> None:
    root, mask_path = _write_qc_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        explicit_background_channel=5,
        expected_fold_ids=(1,),
    )
    output = tmp_path / "manifest-final-symlink"
    output.mkdir()
    parquet = output / "pannuke_nucleus_manifest.parquet"
    raw_hash = sha256_file(mask_path)
    try:
        os.symlink(mask_path, parquet)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")

    with pytest.raises(PanNukeSemanticsError, match="outside the immutable PanNuke raw root"):
        build_nucleus_manifest(validation, output, batch_rows=2)

    assert parquet.is_symlink()
    assert sha256_file(mask_path) == raw_hash
    assert not (output / "pannuke_manifest_summary.csv").exists()
    assert not list(output.glob(".pannuke-manifest-stage-*"))


def test_manifest_resolves_directory_symlink_before_raw_containment_check(
    tmp_path: Path,
) -> None:
    root, _ = _write_qc_release(tmp_path / "pannuke")
    raw_paths = sorted(root.rglob("*.npy"))
    hashes_before = {path: sha256_file(path) for path in raw_paths}
    alias = tmp_path / "pannuke-alias"
    try:
        os.symlink(root, alias, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    output = alias / "derived-manifest"

    with pytest.raises(PanNukeSemanticsError, match="outside the immutable PanNuke raw root"):
        build_nucleus_manifest(
            root,
            output,
            explicit_background_channel=5,
        )

    assert not (root / "derived-manifest").exists()
    assert {path: sha256_file(path) for path in raw_paths} == hashes_before
