"""Tiny synthetic fixtures for the local, read-only PanNuke data gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from histo_audit.pannuke import (
    MANIFEST_REQUIRED_COLUMNS,
    OFFICIAL_METRICS_CLASS_MAPPING,
    SOURCE_PATCH_INDEPENDENCE_STATEMENT,
    PanNukeDiscoveryError,
    PanNukeNotFoundError,
    PanNukeSemanticsError,
    audit_pannuke_duplicates,
    build_nucleus_manifest,
    discover_pannuke_release,
    inventory_raw_files,
    locate_pannuke_root,
    open_npy_mmap,
    patch_sha256,
    sha256_file,
    validate_manifest_invariants,
    validate_pannuke,
    write_overlay_grid,
)
from histo_audit.pannuke.duplicates import (
    RankedDuplicateCandidate,
    _candidate_grid_label,
)
from histo_audit.representations.imagenet import (
    EmbeddingResult,
    PretrainedWeightsUnavailableError,
)


def _arrays(*, fold_id: int, duplicate_first: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
    height, width = 13, 11
    y_grid, x_grid = np.mgrid[:height, :width]
    images = np.empty((2, height, width, 3), dtype=np.uint8)
    for patch_index in range(2):
        images[patch_index, ..., 0] = (x_grid * 13 + fold_id * 17 + patch_index) % 256
        images[patch_index, ..., 1] = (y_grid * 11 + fold_id * 19 + patch_index * 3) % 256
        images[patch_index, ..., 2] = ((x_grid + y_grid) * 7 + fold_id * 23 + patch_index * 5) % 256
    if duplicate_first is not None:
        images[0] = duplicate_first

    masks = np.zeros((2, height, width, 6), dtype=np.int32)
    masks[0, 1:4, 1:4, 0] = 11
    masks[0, 7:11, 5:9, 1] = 27
    masks[1, 2:6, 6:10, 2] = 103
    masks[1, 8:12, 0:3, 4] = 205
    occupied = np.any(masks[..., :5] > 0, axis=-1)
    masks[..., 5] = (~occupied).astype(np.int32)
    tissues = np.asarray(["Breast", "Colon"], dtype="<U16")
    return images, masks, tissues


def _write_release(
    root: Path,
    *,
    folds: int = 2,
    channel_first_second_fold: bool = True,
) -> tuple[Path, np.ndarray]:
    first_image: np.ndarray | None = None
    for fold_id in range(1, folds + 1):
        fold_dir = root / f"Fold {fold_id}" / "release_arrays"
        fold_dir.mkdir(parents=True)
        images, masks, tissues = _arrays(
            fold_id=fold_id,
            duplicate_first=first_image if fold_id == 2 else None,
        )
        if first_image is None:
            first_image = images[0].copy()
        if channel_first_second_fold and fold_id == 2:
            images = np.moveaxis(images, -1, 1)
            masks = np.moveaxis(masks, -1, 1)
        np.save(fold_dir / "pixels.npy", images)
        np.save(fold_dir / "labels.npy", masks)
        np.save(fold_dir / "organs.npy", tissues)
    assert first_image is not None
    return root, first_image


def _snapshot_paths(paths: tuple[Path, ...]) -> dict[str, tuple[bytes, int]]:
    return {
        str(path.resolve()): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in paths
        if path.is_file()
    }


def test_root_resolution_precedence_and_clear_failure(tmp_path: Path) -> None:
    explicit = _write_release(tmp_path / "explicit", folds=1)[0]
    configured = _write_release(tmp_path / "configured", folds=1)[0]
    project = tmp_path / "project"
    default = _write_release(project / "data" / "raw" / "pannuke", folds=1)[0]

    assert locate_pannuke_root(explicit, project_root=project) == explicit.resolve()
    assert (
        locate_pannuke_root(project_root=project, environ={"PANNUKE_ROOT": str(configured)})
        == configured.resolve()
    )
    assert locate_pannuke_root(project_root=project, environ={}) == default.resolve()
    with pytest.raises(PanNukeNotFoundError, match="explicit PanNuke root"):
        locate_pannuke_root(tmp_path / "missing", project_root=project)


def test_discovery_uses_headers_contents_and_fold_markers(tmp_path: Path) -> None:
    root, _ = _write_release(tmp_path / "pannuke")
    discovery = discover_pannuke_release(root)

    assert discovery.fold_ids == (1, 2)
    assert len(discovery.folds) == 2
    assert discovery.folds[0].image_path.name == "pixels.npy"
    assert discovery.folds[0].mask_path.name == "labels.npy"
    assert discovery.folds[0].tissue_path.name == "organs.npy"
    assert discovery.folds[0].image_channel_axis == 3
    assert discovery.folds[1].image_channel_axis == 1
    assert isinstance(open_npy_mmap(discovery.folds[0].image_path), np.memmap)

    ambiguous = root / "Fold 1" / "release_arrays" / "pixels_copy.npy"
    np.save(ambiguous, open_npy_mmap(discovery.folds[0].image_path))
    with pytest.raises(PanNukeDiscoveryError, match="ambiguous image arrays"):
        discover_pannuke_release(root)


def test_archive_only_and_unmarked_arrays_fail_conservatively(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive_only"
    archive_root.mkdir()
    (archive_root / "PanNuke_Fold_1.zip").write_bytes(b"not extracted")
    with pytest.raises(PanNukeDiscoveryError, match="extract the verified release"):
        discover_pannuke_release(archive_root)

    unmarked = tmp_path / "unmarked"
    unmarked.mkdir()
    images, masks, tissues = _arrays(fold_id=1)
    np.save(unmarked / "images.npy", images)
    np.save(unmarked / "masks.npy", masks)
    np.save(unmarked / "types.npy", tissues)
    with pytest.raises(PanNukeDiscoveryError, match="explicit fold marker"):
        discover_pannuke_release(unmarked)


def test_inventory_hashes_every_raw_file_stably(tmp_path: Path) -> None:
    root, _ = _write_release(tmp_path / "pannuke", folds=1)
    records = inventory_raw_files(root)

    assert len(records) == 3
    assert [record.relative_path for record in records] == sorted(
        record.relative_path for record in records
    )
    for record in records:
        source = root / record.relative_path
        expected = hashlib.sha256(source.read_bytes()).hexdigest()
        assert record.sha256 == expected == sha256_file(source)
        assert len(record.sha256) == 64


def test_validation_reports_mapping_ranges_hashes_and_overlay(tmp_path: Path) -> None:
    root, _ = _write_release(tmp_path / "pannuke")
    artifacts = validate_pannuke(
        root,
        tmp_path / "validation",
        max_samples_per_fold=2,
        max_overlay_patches=3,
        expected_fold_ids=(1, 2),
    )

    assert artifacts.json_path.is_file()
    assert artifacts.markdown_path.is_file()
    assert artifacts.overlay_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifacts.raw_inventory_csv_path.is_file()
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "valid"
    assert payload["release_complete"] is True
    assert payload["expected_fold_ids"] == [1, 2]
    assert payload["class_mapping"]["class_names"] == list(
        OFFICIAL_METRICS_CLASS_MAPPING.class_names
    )
    assert "PanNuke-metrics" in payload["class_mapping"]["source"]
    assert len(payload["raw_file_inventory"]) == 6
    assert payload["independence_statement"] == SOURCE_PATCH_INDEPENDENCE_STATEMENT
    assert payload["grouping_unit"] == "source_patch"
    assert all(fold["background_channel_index"] == 5 for fold in payload["fold_validation"])
    assert all(fold["sampled_instance_ids_by_class"] for fold in payload["fold_validation"])
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    assert "Patient- and WSI-level independence could not be guaranteed" in markdown
    assert "does not modify source annotations" in markdown


def test_library_validation_and_overlay_reject_raw_destinations_before_writing(
    tmp_path: Path,
) -> None:
    root, _ = _write_release(tmp_path / "pannuke", folds=3)
    raw_files = tuple(path for path in root.rglob("*") if path.is_file())
    expected = _snapshot_paths(raw_files)
    mask_path = root / "Fold 1" / "release_arrays" / "labels.npy"

    with pytest.raises(PanNukeSemanticsError, match="overlaps the immutable"):
        validate_pannuke(root, root / "derived")
    with pytest.raises(PanNukeSemanticsError, match="overlaps the immutable"):
        validate_pannuke(
            root,
            tmp_path / "safe-output-a",
            overlay_path=mask_path,
        )
    with pytest.raises(PanNukeSemanticsError, match="overlaps the immutable"):
        validate_pannuke(
            root,
            tmp_path / "safe-output-b",
            raw_inventory_csv_path=mask_path,
        )
    assert _snapshot_paths(raw_files) == expected
    assert not (root / "derived").exists()
    assert not (tmp_path / "safe-output-a").exists()
    assert not (tmp_path / "safe-output-b").exists()

    validation = validate_pannuke(root, tmp_path / "safe-validation")
    with pytest.raises(PanNukeSemanticsError, match="overlaps the immutable"):
        write_overlay_grid(validation.result, mask_path)
    assert _snapshot_paths(raw_files) == expected


def test_library_validation_rejects_suffix_and_resolved_alias_before_output_creation(
    tmp_path: Path,
) -> None:
    root, _ = _write_release(tmp_path / "pannuke", folds=3)
    with pytest.raises(PanNukeSemanticsError, match=r"must use suffix \.png"):
        validate_pannuke(
            root,
            tmp_path / "output-a",
            overlay_path=tmp_path / "wrong.csv",
        )
    with pytest.raises(PanNukeSemanticsError, match=r"must use suffix \.csv"):
        validate_pannuke(
            root,
            tmp_path / "output-b",
            raw_inventory_csv_path=tmp_path / "wrong.png",
        )
    aliased_output = tmp_path / "aliased.png"
    with pytest.raises(PanNukeSemanticsError, match="alias after resolution"):
        validate_pannuke(root, aliased_output, overlay_path=aliased_output)
    assert not (tmp_path / "output-a").exists()
    assert not (tmp_path / "output-b").exists()
    assert not aliased_output.exists()


def test_library_validation_rejects_file_target_ancestry_collision_before_build(
    tmp_path: Path,
) -> None:
    root, _ = _write_release(tmp_path / "pannuke", folds=3)
    output = tmp_path / "validation"
    nested_overlay = output / "pannuke_validation.json" / "overlay.png"

    with pytest.raises(PanNukeSemanticsError, match="file artifact paths collide"):
        validate_pannuke(root, output, overlay_path=nested_overlay)

    assert not output.exists()


def test_library_validation_resolves_symlink_destination_into_raw(tmp_path: Path) -> None:
    root, _ = _write_release(tmp_path / "pannuke", folds=3)
    mask_path = root / "Fold 1" / "release_arrays" / "labels.npy"
    alias = tmp_path / "outside-overlay.png"
    try:
        alias.symlink_to(mask_path)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable in this Windows environment: {error}")
    expected = _snapshot_paths((mask_path,))

    with pytest.raises(PanNukeSemanticsError, match="overlaps the immutable"):
        validate_pannuke(root, tmp_path / "output", overlay_path=alias)
    assert _snapshot_paths((mask_path,)) == expected
    assert not (tmp_path / "output").exists()


def test_library_validation_final_inventory_failure_preserves_previous_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import histo_audit.pannuke.validation as validation_module

    root, _ = _write_release(tmp_path / "pannuke", folds=3)
    output = tmp_path / "validation"
    initial = validate_pannuke(root, output, max_overlay_patches=4)
    final_paths = (
        initial.json_path,
        initial.markdown_path,
        initial.overlay_path,
        initial.raw_inventory_csv_path,
    )
    expected = _snapshot_paths(final_paths)

    def fail_final_inventory(*args: object, **kwargs: object) -> object:
        raise PanNukeSemanticsError("injected final raw inventory mismatch")

    monkeypatch.setattr(validation_module, "verify_raw_inventory_unchanged", fail_final_inventory)
    with pytest.raises(PanNukeSemanticsError, match="injected final raw inventory mismatch"):
        validate_pannuke(root, output, max_overlay_patches=4)
    assert _snapshot_paths(final_paths) == expected


def test_library_validation_publish_time_raw_mutation_rolls_back_all_base_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import histo_audit.pannuke.validation as validation_module

    root, _ = _write_release(tmp_path / "pannuke", folds=3)
    image_path = root / "Fold 1" / "release_arrays" / "pixels.npy"
    output = tmp_path / "validation"
    real_promote = validation_module._promote_without_overwrite
    mutation_injected = False

    def mutate_at_first_promotion(
        source: Path, destination: Path
    ) -> validation_module.PublishedPath:
        nonlocal mutation_injected
        if not mutation_injected:
            mutation_injected = True
            images = np.load(image_path, mmap_mode="r+", allow_pickle=False)
            images[0, 0, 0, 0] = (int(images[0, 0, 0, 0]) + 1) % 256
            images.flush()
            del images
        return real_promote(source, destination)

    monkeypatch.setattr(validation_module, "_promote_without_overwrite", mutate_at_first_promotion)
    with pytest.raises(PanNukeSemanticsError, match="raw inventory changed"):
        validate_pannuke(root, output, max_overlay_patches=4)

    assert mutation_injected
    assert not (output / "pannuke_validation.json").exists()
    assert not (output / "pannuke_validation.md").exists()
    assert not (output / "pannuke_overlay_grid.png").exists()
    assert not (output / "raw_files_sha256.csv").exists()
    assert not (tmp_path / ".validation.pannuke-validation-staging").exists()


def test_library_validation_rejects_an_active_publication_lock(tmp_path: Path) -> None:
    from histo_audit.pannuke.publication import ExclusiveBundlePublicationLock

    root, _ = _write_release(tmp_path / "pannuke", folds=3)
    output = tmp_path / "validation"
    targets = (
        output,
        output / "pannuke_validation.json",
        output / "pannuke_validation.md",
        output / "pannuke_overlay_grid.png",
        output / "raw_files_sha256.csv",
    )

    with (
        ExclusiveBundlePublicationLock(targets, role="test validation"),
        pytest.raises(FileExistsError, match="another PanNuke base validation"),
    ):
        validate_pannuke(root, output, max_overlay_patches=4)

    assert not output.exists()


def test_library_validation_lock_rejects_partially_overlapping_output_sets(
    tmp_path: Path,
) -> None:
    from histo_audit.pannuke.publication import ExclusiveBundlePublicationLock

    root, _ = _write_release(tmp_path / "pannuke", folds=3)
    first_output = tmp_path / "validation-a"
    second_output = tmp_path / "validation-b"
    shared_overlay = tmp_path / "shared-overlay.png"
    first_targets = (
        first_output,
        first_output / "pannuke_validation.json",
        first_output / "pannuke_validation.md",
        shared_overlay,
        first_output / "raw_files_sha256.csv",
    )

    with (
        ExclusiveBundlePublicationLock(first_targets, role="first validation"),
        pytest.raises(FileExistsError, match="another PanNuke base validation"),
    ):
        validate_pannuke(root, second_output, overlay_path=shared_overlay)

    assert not first_output.exists()
    assert not second_output.exists()
    assert not shared_overlay.exists()


def test_library_validation_rollback_preserves_a_foreign_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import histo_audit.pannuke.validation as validation_module

    root, _ = _write_release(tmp_path / "pannuke", folds=3)
    output = tmp_path / "validation"
    foreign_payload = b"foreign concurrent validation owner\n"
    real_promote = validation_module._promote_without_overwrite
    calls = 0

    def replace_first_then_fail(source: Path, destination: Path) -> validation_module.PublishedPath:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second-promotion failure")
        publication = real_promote(source, destination)
        destination.unlink()
        destination.write_bytes(foreign_payload)
        return publication

    monkeypatch.setattr(validation_module, "_promote_without_overwrite", replace_first_then_fail)
    with pytest.raises(RuntimeError, match="ownership-safe rollback was incomplete"):
        validate_pannuke(root, output, max_overlay_patches=4)

    assert calls == 2
    assert (output / "pannuke_validation.md").read_bytes() == foreign_payload
    assert not (output / "pannuke_validation.json").exists()
    assert not (output / "pannuke_overlay_grid.png").exists()
    assert not (output / "raw_files_sha256.csv").exists()


def test_library_validation_final_consistency_check_preserves_foreign_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import histo_audit.pannuke.validation as validation_module

    root, _ = _write_release(tmp_path / "pannuke", folds=3)
    output = tmp_path / "validation"
    foreign_payload = b"foreign replacement after final raw rehash\n"
    real_verify = validation_module.verify_raw_inventory_unchanged
    calls = 0

    def replace_after_publish(result: object) -> object:
        nonlocal calls
        calls += 1
        verified = real_verify(result)  # type: ignore[arg-type]
        if calls == 3:
            destination = output / "pannuke_validation.md"
            destination.unlink()
            destination.write_bytes(foreign_payload)
        return verified

    monkeypatch.setattr(validation_module, "verify_raw_inventory_unchanged", replace_after_publish)
    with pytest.raises(RuntimeError, match="ownership-safe rollback was incomplete"):
        validate_pannuke(root, output, max_overlay_patches=4)

    assert calls == 3
    assert (output / "pannuke_validation.md").read_bytes() == foreign_payload
    assert not (output / "pannuke_validation.json").exists()
    assert not (output / "pannuke_overlay_grid.png").exists()
    assert not (output / "raw_files_sha256.csv").exists()


def test_library_validation_treats_a_broken_final_symlink_as_occupied(
    tmp_path: Path,
) -> None:
    root, _ = _write_release(tmp_path / "pannuke", folds=3)
    output = tmp_path / "validation"
    output.mkdir()
    missing_target = tmp_path / "foreign-owner-missing.md"
    broken_final = output / "pannuke_validation.md"
    try:
        broken_final.symlink_to(missing_target)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable in this Windows environment: {error}")

    with pytest.raises(FileExistsError, match="artifact set is partial"):
        validate_pannuke(root, output, max_overlay_patches=4)

    assert broken_final.is_symlink()
    assert not broken_final.exists()
    assert not missing_target.exists()
    assert not (output / "pannuke_validation.json").exists()


def test_library_validation_conflicting_rerun_preserves_all_base_bytes_and_mtimes(
    tmp_path: Path,
) -> None:
    root, _ = _write_release(tmp_path / "pannuke", folds=3)
    output = tmp_path / "validation"
    initial = validate_pannuke(root, output, max_overlay_patches=4)
    final_paths = (
        initial.json_path,
        initial.markdown_path,
        initial.overlay_path,
        initial.raw_inventory_csv_path,
    )
    expected = _snapshot_paths(final_paths)
    identical = validate_pannuke(root, output, max_overlay_patches=4)
    assert (
        _snapshot_paths(
            (
                identical.json_path,
                identical.markdown_path,
                identical.overlay_path,
                identical.raw_inventory_csv_path,
            )
        )
        == expected
    )

    with pytest.raises(FileExistsError, match="base-validation artifacts differ"):
        validate_pannuke(root, output, max_overlay_patches=3)
    assert _snapshot_paths(final_paths) == expected


def test_validation_rejects_alignment_and_ambiguous_semantics(tmp_path: Path) -> None:
    root, _ = _write_release(tmp_path / "shape_failure", folds=1)
    mask_path = root / "Fold 1" / "release_arrays" / "labels.npy"
    mask = np.load(mask_path)
    np.save(mask_path, mask[:, :-1, ...])
    with pytest.raises(PanNukeSemanticsError, match="alignment mismatch"):
        validate_pannuke(root, tmp_path / "shape_output", expected_fold_ids=(1,))

    ambiguous_root, _ = _write_release(tmp_path / "semantic_failure", folds=1)
    ambiguous_path = ambiguous_root / "Fold 1" / "release_arrays" / "labels.npy"
    ambiguous_mask = np.load(ambiguous_path)
    ambiguous_mask = np.concatenate([ambiguous_mask, ambiguous_mask[..., 5:6]], axis=-1)
    np.save(ambiguous_path, ambiguous_mask)
    with pytest.raises(PanNukeSemanticsError, match="semantics are ambiguous"):
        validate_pannuke(
            ambiguous_root,
            tmp_path / "semantic_output",
            expected_fold_ids=(1,),
        )

    valid_root, _ = _write_release(tmp_path / "mapping_failure", folds=1)
    with pytest.raises(PanNukeSemanticsError, match="cannot be inferred"):
        validate_pannuke(
            valid_root,
            tmp_path / "mapping_output",
            use_documented_default_mapping=False,
            expected_fold_ids=(1,),
        )


def test_validation_rejects_an_incomplete_official_release(tmp_path: Path) -> None:
    root, _ = _write_release(tmp_path / "incomplete_release", folds=1)
    with pytest.raises(PanNukeSemanticsError, match="subset cannot receive"):
        validate_pannuke(root, tmp_path / "output")


def test_validation_full_scan_records_unsampled_disconnected_raw_instance(tmp_path: Path) -> None:
    """Release connectivity QC must cover patches outside the bounded evidence sample."""

    root, _ = _write_release(tmp_path / "full_scan_failure", folds=1)
    mask_path = root / "Fold 1" / "release_arrays" / "labels.npy"
    mask = np.load(mask_path)
    # With max_samples_per_fold=1 the evidence sample is patch 0. Give one raw ID
    # in patch 1 two components while keeping all array/background structure valid.
    mask[1, 2:6, 6:10, 2] = 0
    mask[1, 2:4, 6:8, 2] = 103
    mask[1, 4:6, 8:10, 2] = 103
    occupied = np.any(mask[1, ..., :5] > 0, axis=-1)
    mask[1, ..., 5] = (~occupied).astype(np.int32)
    np.save(mask_path, mask)

    validation = validate_pannuke(
        root,
        tmp_path / "full_scan_output",
        max_samples_per_fold=1,
        expected_fold_ids=(1,),
    )
    facts = validation.result.fold_validation[0]
    assert facts.disconnected_instance_count_full_scan == 1
    assert facts.disconnected_patch_count_full_scan == 1
    assert facts.disconnected_instance_count_sampled == 0
    assert facts.malformed_instance_count_sampled == 0
    assert validation.result.qc_policy.disconnected_instance_ids_are_fatal is False

    manifest = build_nucleus_manifest(
        validation,
        tmp_path / "full_scan_manifest",
        batch_rows=2,
    )
    frame = pq.read_table(manifest.parquet_path).to_pandas()
    row = frame[
        (frame["official_fold"] == 1)
        & (frame["source_patch_index"] == 1)
        & (frame["nucleus_class_index"] == 2)
        & (frame["instance_id"] == 103)
    ].iloc[0]
    assert "disconnected_instance_id" in set(row["quality_flags"])
    assert bool(row["primary_eligible"]) is True
    assert bool(row["confirmatory_eligible"]) is True


def test_manifest_is_dynamic_group_safe_immutable_and_crop_free(tmp_path: Path) -> None:
    root, _ = _write_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        max_overlay_patches=2,
        expected_fold_ids=(1, 2),
    )
    manifest = build_nucleus_manifest(validation, tmp_path / "manifest", batch_rows=2)
    table = pq.read_table(manifest.parquet_path)

    validate_manifest_invariants(table)
    assert set(MANIFEST_REQUIRED_COLUMNS).issubset(table.column_names)
    frame = table.to_pandas()
    assert manifest.row_count == len(frame) == 8
    assert manifest.patch_count == 4
    assert manifest.sha256 == sha256_file(manifest.parquet_path)
    assert frame["sample_id"].is_unique
    assert (frame["group_id"] == frame["patch_id"]).all()
    assert (frame["grouping_unit"] == "source_patch").all()
    assert frame["patient_id"].isna().all() and frame["wsi_id"].isna().all()
    assert (frame["pre_corruption_label"] == frame["observed_label"]).all()
    assert not frame["is_injected_corruption"].any()
    assert not frame["crop_generated"].any() and frame["crop_path"].isna().all()
    assert frame["configuration_hash"].nunique() == 1
    assert len(frame["configuration_hash"].iloc[0]) == 64
    assert (frame["area"] > 0).all()
    assert (frame["bbox_x_max"] <= 11).all() and (frame["bbox_y_max"] <= 13).all()
    metadata = table.schema.metadata or {}
    assert metadata[b"grouping_unit"] == b"source_patch"
    assert metadata[b"tiny_crops_generated"] == b"false"
    assert metadata[b"manifest_configuration_hash"].decode() == frame["configuration_hash"].iloc[0]
    summary = manifest.summary_csv_path.read_text(encoding="utf-8")
    assert "source_patch_count" in summary
    assert "Patient- and WSI-level independence could not be guaranteed" in summary


def test_manifest_rejects_raw_data_changed_after_validation(tmp_path: Path) -> None:
    root, _ = _write_release(tmp_path / "pannuke", folds=1)
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        expected_fold_ids=(1,),
    )
    image_path = root / "Fold 1" / "release_arrays" / "pixels.npy"
    images = np.load(image_path)
    images[0, 0, 0, 0] = (int(images[0, 0, 0, 0]) + 1) % 256
    np.save(image_path, images)

    with pytest.raises(PanNukeSemanticsError, match="changed before manifest construction"):
        build_nucleus_manifest(validation, tmp_path / "manifest")


def test_duplicate_audit_hashes_all_and_only_recommends_review(tmp_path: Path) -> None:
    root, first_image = _write_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        max_overlay_patches=2,
        expected_fold_ids=(1, 2),
    )
    audit = audit_pannuke_duplicates(
        validation,
        tmp_path / "duplicates",
        max_perceptual_patches=4,
        max_hamming_distance=0,
        run_embedding_signal=False,
    )

    assert audit.exact_pair_count >= 1
    assert audit.sampled_patch_count == 4
    payload = json.loads(audit.json_path.read_text(encoding="utf-8"))
    assert payload["total_source_patches_exactly_hashed"] == 4
    assert payload["cross_fold_exact_pair_count"] >= 1
    assert payload["policy"]["automatic_deletion"] is False
    assert all(pair["recommended_action"] == "review_only" for pair in payload["pairs"])
    assert all(
        len(pair["exact_sha256"]) == 64
        for pair in payload["pairs"]
        if pair["method"] == "exact_sha256"
    )
    assert audit.csv_path.is_file()
    assert patch_sha256(first_image) == patch_sha256(first_image.copy())


def test_duplicate_audit_adds_independent_embedding_signal_and_canonical_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _write_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        max_overlay_patches=2,
        expected_fold_ids=(1, 2),
    )

    def fake_extract(
        images: np.ndarray,
        sample_ids: list[str],
        **_: object,
    ) -> EmbeddingResult:
        embeddings = np.zeros((len(sample_ids), 512), dtype=np.float32)
        for index in range(len(sample_ids)):
            embeddings[index, min(index, 511)] = 1.0
        # The known exact cross-fold pair receives identical independent vectors.
        first_id = "pannuke-fold-1-patch-000000"
        second_id = "pannuke-fold-2-patch-000000"
        embeddings[sample_ids.index(first_id)] = embeddings[sample_ids.index(second_id)]
        assert images.shape[0] == len(sample_ids)
        input_digest = hashlib.sha256()
        input_digest.update(str(images.shape).encode("ascii"))
        input_digest.update(images.dtype.str.encode("ascii"))
        input_digest.update(np.ascontiguousarray(images).tobytes())
        return EmbeddingResult(
            embeddings=embeddings,
            sample_ids=np.asarray(sample_ids, dtype=np.str_),
            metadata={
                "encoder_name": "torchvision.resnet18",
                "encoder_frozen": True,
                "classification_head": "removed (fc=Identity)",
                "weight_identifier": "ResNet18_Weights.IMAGENET1K_V1",
                "weight_sha256": "a" * 64,
                "preprocessing": {"api": "unit-test fixture"},
                "input_variant": "rgb",
                "input_sha256": input_digest.hexdigest(),
                "output_dimension": 512,
            },
        )

    monkeypatch.setattr("histo_audit.pannuke.duplicates.extract_resnet18_embeddings", fake_extract)
    rankings_path = tmp_path / "artifacts" / "rankings" / "cross_fold_duplicate_candidates.csv"
    audit = audit_pannuke_duplicates(
        validation,
        tmp_path / "audit",
        max_hamming_distance=0,
        min_embedding_cosine_similarity=0.999,
        rankings_csv_path=rankings_path,
    )

    assert audit.embedding_status == "passed"
    assert audit.required_two_signal_gate_complete is True
    assert audit.embedding_pair_count >= 1
    assert audit.csv_path == rankings_path
    assert audit.markdown_path.is_file()
    assert audit.visual_grid_path.is_file()
    assert audit.hash_provenance_csv_path.is_file()
    provenance_lines = audit.hash_provenance_csv_path.read_text(encoding="utf-8").splitlines()
    assert len(provenance_lines) == 5  # header plus every one of four patches
    payload = json.loads(audit.json_path.read_text(encoding="utf-8"))
    assert payload["required_two_signal_near_duplicate_gate_complete"] is True
    assert payload["coverage"]["patches_with_full_hash_provenance"] == 4
    assert payload["embedding_signal"]["status"] == "passed"
    assert any(
        pair["embedding_cosine_similarity"] is not None for pair in payload["ranked_candidates"]
    )
    assert all(pair["crosses_fold"] is True for pair in payload["ranked_candidates"])
    assert all(pair["recommended_action"] == "review_only" for pair in payload["pairs"])
    assert all(pair["automatic_deletion"] is False for pair in payload["pairs"])
    assert audit.embedding_cache_path is not None
    sidecar = json.loads(
        audit.embedding_cache_path.with_suffix(
            f"{audit.embedding_cache_path.suffix}.metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert sidecar["input_variant"] == "context_rgb"
    assert payload["embedding_signal"]["metadata"] == sidecar
    report = audit.markdown_path.read_text(encoding="utf-8")
    assert "No patch was deleted" in report
    assert "number of distinct evidence signals" in report


def test_duplicate_candidate_caption_contains_patch_ids_scores_and_review_status() -> None:
    candidate = RankedDuplicateCandidate(
        rank=3,
        candidate_id="candidate",
        sample_id_a="pannuke-fold-1-patch-000123",
        sample_id_b="pannuke-fold-3-patch-000456",
        fold_a=1,
        fold_b=3,
        patch_index_a=123,
        patch_index_b=456,
        exact_match=False,
        exact_sha256=None,
        perceptual_hamming_distance=2,
        embedding_cosine_similarity=None,
        evidence_methods="perceptual_average_hash",
        evidence_count=1,
    )

    caption = _candidate_grid_label(candidate)

    assert "fold/patch 1/123 vs 3/456" in caption
    assert "pHash=2" in caption
    assert "cosine=not_available" in caption
    assert "evidence=perceptual_average_hash" in caption
    assert "review_only" in caption


def test_duplicate_audit_records_offline_weight_blocker_without_fake_cosines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _write_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        max_overlay_patches=2,
        expected_fold_ids=(1, 2),
    )

    def blocked(*_: object, **__: object) -> EmbeddingResult:
        raise PretrainedWeightsUnavailableError("official test checkpoint is offline")

    monkeypatch.setattr("histo_audit.pannuke.duplicates.extract_resnet18_embeddings", blocked)
    audit = audit_pannuke_duplicates(
        validation,
        tmp_path / "audit",
        max_perceptual_patches=4,
        max_embedding_patches=4,
    )

    payload = json.loads(audit.json_path.read_text(encoding="utf-8"))
    assert audit.embedding_status == "blocked"
    assert audit.required_two_signal_gate_complete is False
    assert payload["embedding_signal"]["blocker"] == "official test checkpoint is offline"
    assert payload["counts"]["embedding_pair_count"] == 0
    assert all(pair["embedding_cosine_similarity"] is None for pair in payload["ranked_candidates"])
    report = audit.markdown_path.read_text(encoding="utf-8")
    assert "No cosine values were fabricated" in report
