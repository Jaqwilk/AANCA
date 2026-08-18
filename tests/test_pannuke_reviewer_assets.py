from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

import histo_audit.external_validation.pannuke_assets as assets_module
from histo_audit.external_validation import (
    build_blinded_review_package,
    validate_blinded_review_package,
)
from histo_audit.external_validation.pannuke_assets import build_pannuke_reviewer_assets
from histo_audit.pannuke import (
    PanNukeSemanticsError,
    PanNukeValidationResult,
    build_nucleus_manifest,
    validate_pannuke,
)
from histo_audit.pannuke.validation import verify_raw_inventory_unchanged
from histo_audit.representations import PanNukeCropConfig, extract_pannuke_crop_batch
from histo_audit.utils.run_tracking import atomic_write_bytes, sha256_file


def _tiny_release(root: Path) -> Path:
    fold = root / "Fold 1" / "release_arrays"
    fold.mkdir(parents=True)
    height = width = 24
    y, x = np.mgrid[:height, :width]
    images = np.zeros((3, height, width, 3), dtype=np.uint8)
    for patch in range(3):
        images[patch, ..., 0] = (20 + x * 5 + patch * 7) % 256
        images[patch, ..., 1] = (30 + y * 4 + patch * 11) % 256
        images[patch, ..., 2] = (40 + (x + y) * 3 + patch * 13) % 256
    masks = np.zeros((3, height, width, 6), dtype=np.int32)
    masks[0, 2:7, 3:9, 0] = 101
    masks[0, 13:20, 12:19, 1] = 202
    masks[1, 3:9, 13:20, 2] = 303
    masks[1, 14:20, 3:9, 3] = 404
    masks[2, 7:17, 8:16, 4] = 505
    masks[..., 5] = (~np.any(masks[..., :5] > 0, axis=-1)).astype(np.int32)
    np.save(fold / "pixels.npy", images)
    np.save(fold / "labels.npy", masks)
    np.save(fold / "organs.npy", np.asarray(["Breast", "Colon", "Lung"], dtype="<U16"))
    return root


def _validated_manifest(tmp_path: Path) -> tuple[PanNukeValidationResult, Path, tuple[str, ...]]:
    validation = validate_pannuke(
        _tiny_release(tmp_path / "pannuke"),
        tmp_path / "validation",
        expected_fold_ids=(1,),
        max_overlay_patches=1,
    )
    manifest = build_nucleus_manifest(validation, tmp_path / "manifest", batch_rows=2)
    frame = pq.read_table(manifest.parquet_path).to_pandas()
    sample_ids = tuple(frame["sample_id"].astype(str).tolist())
    return validation.result, manifest.parquet_path, sample_ids


def _validated_disconnected_manifest(
    tmp_path: Path,
) -> tuple[PanNukeValidationResult, Path, tuple[str, ...]]:
    root = _tiny_release(tmp_path / "pannuke-disconnected")
    mask_path = root / "Fold 1" / "release_arrays" / "labels.npy"
    masks = np.load(mask_path)
    masks[0, 10:12, 10:12, 0] = 101
    masks[..., 5] = (~np.any(masks[..., :5] > 0, axis=-1)).astype(np.int32)
    np.save(mask_path, masks)
    validation = validate_pannuke(
        root,
        tmp_path / "validation-disconnected",
        expected_fold_ids=(1,),
        max_overlay_patches=1,
    )
    manifest = build_nucleus_manifest(
        validation,
        tmp_path / "manifest-disconnected",
        batch_rows=2,
    )
    frame = pq.read_table(manifest.parquet_path).to_pandas()
    disconnected_ids = tuple(
        frame.loc[
            frame["quality_flags"].map(lambda flags: "disconnected_instance_id" in set(flags)),
            "sample_id",
        ].astype(str)
    )
    return validation.result, manifest.parquet_path, disconnected_ids


def _build(
    validation: PanNukeValidationResult,
    manifest: Path,
    sample_ids: tuple[str, ...],
    destination: Path,
) -> assets_module.PanNukeReviewerAssetsResult:
    return build_pannuke_reviewer_assets(
        validation,
        manifest,
        sample_ids,
        destination,
        expected_manifest_sha256=sha256_file(manifest),
        crop_config=PanNukeCropConfig(output_size=40, padding=4, context_brightness=0.5),
    )


def test_assets_retain_exact_raw_target_identity_and_contours(tmp_path: Path) -> None:
    validation, manifest, sample_ids = _validated_manifest(tmp_path)
    result = _build(validation, manifest, sample_ids, tmp_path / "reviewer-assets")

    asset_manifest = pd.read_csv(result.asset_manifest_csv, dtype={"sample_id": str})
    assert tuple(asset_manifest["sample_id"]) == sample_ids
    assert asset_manifest["target_identity_verified"].all()
    assert not asset_manifest["source_annotations_modified"].all()
    assert set(asset_manifest["official_fold"]) == {1}
    assert asset_manifest["instance_id"].nunique() == len(sample_ids)
    assert asset_manifest["instance_channel_index"].nunique() == 5
    crops = extract_pannuke_crop_batch(
        validation,
        manifest,
        sample_ids=sample_ids,
        config=PanNukeCropConfig(output_size=40, padding=4, context_brightness=0.5),
    )
    raw_images = np.load(validation.folds[0].image_path, allow_pickle=False)
    source_frame = pq.read_table(manifest).to_pandas().set_index("sample_id")
    magenta = np.asarray([255, 0, 255], dtype=np.uint8)
    for index, sample_id in enumerate(sample_ids):
        row = asset_manifest.iloc[index]
        full_patch = np.asarray(
            Image.open(result.output_directory / row["full_patch_path"]).convert("RGB")
        )
        raw_patch = raw_images[int(source_frame.loc[sample_id, "source_patch_index"])]
        contour = crops.source_contours_xy[index]
        contour_mask = np.zeros(raw_patch.shape[:2], dtype=bool)
        contour_mask[contour[:, 1], contour[:, 0]] = True
        assert np.all(full_patch[contour_mask] == magenta)
        assert np.array_equal(full_patch[~contour_mask], raw_patch[~contour_mask])

        target_crop = np.asarray(
            Image.open(result.output_directory / row["target_crop_path"]).convert("RGB")
        )
        target_contour = np.asarray(
            Image.open(result.output_directory / row["target_contour_path"]).convert("RGB")
        )
        assert np.array_equal(target_crop, crops.context_rgb[index])
        crop_contour = crops.target_contour_masks[index]
        assert np.all(target_contour[crop_contour] == magenta)
        assert np.array_equal(target_contour[~crop_contour], target_crop[~crop_contour])
        for role in ("full_patch", "target_crop", "target_contour"):
            path = result.output_directory / row[f"{role}_path"]
            assert sha256_file(path) == row[f"{role}_sha256"]

    metadata = json.loads(result.metadata_json.read_text(encoding="utf-8"))
    assert metadata["class_encoded_by_contour_colour"] is False
    assert metadata["contour_policy"]["colour_is_constant_across_classes"] is True
    assert metadata["target_identity_verified_for_every_sample"] is True
    assert metadata["source_annotations_modified"] is False


def test_assets_render_flagged_disconnected_identity_without_repair(tmp_path: Path) -> None:
    validation, manifest, sample_ids = _validated_disconnected_manifest(tmp_path)
    assert len(sample_ids) == 1
    result = _build(validation, manifest, sample_ids, tmp_path / "disconnected-assets")

    asset_manifest = pd.read_csv(
        result.asset_manifest_csv,
        dtype={"sample_id": str, "disconnected_raw_target_mask_sha256": str},
        keep_default_na=False,
    )
    row = asset_manifest.iloc[0]
    assert bool(row["disconnected_instance_id"])
    assert int(row["raw_target_component_count"]) == 2
    assert int(row["projected_union_component_count"]) >= 1
    assert 0 <= int(row["projection_fallback_component_count"]) <= 2
    assert len(row["disconnected_raw_target_mask_sha256"]) == 64
    metadata = json.loads(result.metadata_json.read_text(encoding="utf-8"))
    policy = metadata["target_mask_projection"]
    assert policy["raw_identity_action"] == (
        "retain_one_raw_identity_without_split_merge_repair_or_relabel"
    )
    assert policy["disconnected_instance_count"] == 1
    assert policy["disconnected_instances"][0]["area"] == 34
    assert policy["disconnected_instances"][0]["bbox"] == [3, 2, 12, 12]
    for role in ("full_patch", "target_crop", "target_contour"):
        assert (result.output_directory / row[f"{role}_path"]).is_file()
    assert verify_raw_inventory_unchanged(validation) == validation.inventory


def test_asset_manifest_feeds_blinded_package_end_to_end(tmp_path: Path) -> None:
    validation, manifest, sample_ids = _validated_manifest(tmp_path)
    assets = _build(validation, manifest, sample_ids, tmp_path / "assets")
    ranking = pd.DataFrame(
        {"sample_id": sample_ids, "risk_score": np.linspace(1.0, 0.0, len(sample_ids))}
    )

    package = build_blinded_review_package(
        assets.asset_manifest_csv,
        ranking,
        tmp_path / "blinded-package",
        top_count=2,
        random_count=2,
        seed=17,
    )
    validation_result = validate_blinded_review_package(
        package.package_directory,
        private_unblinding_key_path=package.private_unblinding_key_csv,
    )

    assert validation_result.valid, validation_result.errors
    assert validation_result.item_count == 4
    assert validation_result.asset_count == 12
    reviewer_items = package.review_items_csv.read_text(encoding="utf-8")
    assert all(sample_id not in reviewer_items for sample_id in sample_ids)


def test_rejects_wrong_manifest_hash_and_missing_sample(tmp_path: Path) -> None:
    validation, manifest, sample_ids = _validated_manifest(tmp_path)

    with pytest.raises(ValueError, match="manifest SHA-256 differs"):
        build_pannuke_reviewer_assets(
            validation,
            manifest,
            sample_ids,
            tmp_path / "wrong-hash",
            expected_manifest_sha256="0" * 64,
        )
    with pytest.raises(KeyError, match="lacks requested sample IDs"):
        build_pannuke_reviewer_assets(
            validation,
            manifest,
            (*sample_ids, "not-a-real-sample"),
            tmp_path / "missing-sample",
            expected_manifest_sha256=sha256_file(manifest),
        )
    assert not (tmp_path / "wrong-hash").exists()
    assert not (tmp_path / "missing-sample").exists()


def test_revalidates_instance_id_against_raw_class_channel(tmp_path: Path) -> None:
    validation, manifest, sample_ids = _validated_manifest(tmp_path)
    table = pq.read_table(manifest)
    frame = table.to_pandas()
    frame.loc[0, "instance_id"] = 999999
    altered = tmp_path / "wrong-instance.parquet"
    pq.write_table(pa.Table.from_pandas(frame, schema=table.schema, preserve_index=False), altered)

    with pytest.raises(
        (ValueError, PanNukeSemanticsError),
        match=r"canonical patch/sample identity is inconsistent|is absent from its raw class channel",
    ):
        build_pannuke_reviewer_assets(
            validation,
            altered,
            sample_ids,
            tmp_path / "wrong-instance-output",
            expected_manifest_sha256=sha256_file(altered),
        )
    assert not (tmp_path / "wrong-instance-output").exists()


def test_raw_hash_change_is_rejected_without_output(tmp_path: Path) -> None:
    validation, manifest, sample_ids = _validated_manifest(tmp_path)
    source = validation.folds[0].image_path
    with source.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(PanNukeSemanticsError, match="raw inventory changed"):
        _build(validation, manifest, sample_ids, tmp_path / "tampered-raw-output")
    assert not (tmp_path / "tampered-raw-output").exists()


def test_manifest_change_after_crop_extraction_is_rejected_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation, manifest, sample_ids = _validated_manifest(tmp_path)
    original = assets_module.extract_pannuke_crop_batch

    def mutate_manifest_after_crop(*args: object, **kwargs: object):
        result = original(*args, **kwargs)
        table = pq.read_table(manifest)
        frame = table.to_pandas()
        frame.loc[0, "tissue_type"] = "MutatedAfterCrop"
        pq.write_table(
            pa.Table.from_pandas(frame, schema=table.schema, preserve_index=False),
            manifest,
        )
        return result

    monkeypatch.setattr(
        assets_module,
        "extract_pannuke_crop_batch",
        mutate_manifest_after_crop,
    )
    destination = tmp_path / "manifest-race-output"
    with pytest.raises(PanNukeSemanticsError, match="manifest changed"):
        _build(validation, manifest, sample_ids, destination)
    assert not destination.exists()


def test_raw_change_after_crop_extraction_is_rejected_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation, manifest, sample_ids = _validated_manifest(tmp_path)
    original = assets_module.extract_pannuke_crop_batch

    def mutate_raw_after_crop(*args: object, **kwargs: object):
        result = original(*args, **kwargs)
        with validation.folds[0].image_path.open("ab") as handle:
            handle.write(b"tamper-after-crop")
        return result

    monkeypatch.setattr(
        assets_module,
        "extract_pannuke_crop_batch",
        mutate_raw_after_crop,
    )
    destination = tmp_path / "raw-race-output"
    with pytest.raises(PanNukeSemanticsError, match="raw inventory changed"):
        _build(validation, manifest, sample_ids, destination)
    assert not destination.exists()


def test_reviewer_post_publish_source_check_rolls_back_owned_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation, manifest, sample_ids = _validated_manifest(tmp_path)
    original = assets_module.verify_raw_inventory_unchanged
    checks = 0

    def mutate_on_post_publish(binding: PanNukeValidationResult):
        nonlocal checks
        checks += 1
        if checks == 3:
            with binding.folds[0].image_path.open("ab") as handle:
                handle.write(b"post-publish-tamper")
        return original(binding)

    monkeypatch.setattr(
        assets_module,
        "verify_raw_inventory_unchanged",
        mutate_on_post_publish,
    )
    destination = tmp_path / "review-post-publish"
    with pytest.raises(PanNukeSemanticsError, match="raw inventory changed"):
        _build(validation, manifest, sample_ids, destination)
    assert checks == 3
    assert not destination.exists()


def test_reviewer_output_traversal_is_rejected_without_raw_tree_change(tmp_path: Path) -> None:
    validation, manifest, sample_ids = _validated_manifest(tmp_path)
    destination = validation.root / ".." / validation.root.name / "reviewer-assets"

    with pytest.raises(PanNukeSemanticsError, match="outside the immutable PanNuke raw release"):
        _build(validation, manifest, sample_ids, destination)
    assert not destination.resolve().exists()
    assert verify_raw_inventory_unchanged(validation) == validation.inventory


def test_reviewer_output_symlink_into_raw_is_rejected(tmp_path: Path) -> None:
    validation, manifest, sample_ids = _validated_manifest(tmp_path)
    link = tmp_path / "review-raw-link"
    try:
        os.symlink(validation.root, link, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    destination = link / "reviewer-assets"

    with pytest.raises(PanNukeSemanticsError, match="outside the immutable PanNuke raw release"):
        _build(validation, manifest, sample_ids, destination)
    assert not (validation.root / "reviewer-assets").exists()
    assert verify_raw_inventory_unchanged(validation) == validation.inventory


def test_generation_is_byte_deterministic_and_never_overwrites(tmp_path: Path) -> None:
    validation, manifest, sample_ids = _validated_manifest(tmp_path)
    first = _build(validation, manifest, sample_ids, tmp_path / "assets-one")
    second = _build(validation, manifest, sample_ids, tmp_path / "assets-two")

    first_files = {
        path.relative_to(first.output_directory).as_posix(): path.read_bytes()
        for path in first.output_directory.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second.output_directory).as_posix(): path.read_bytes()
        for path in second.output_directory.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files
    with pytest.raises(FileExistsError, match="already exists"):
        _build(validation, manifest, sample_ids, first.output_directory)


def test_write_failure_rolls_back_staging_and_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation, manifest, sample_ids = _validated_manifest(tmp_path)
    destination = tmp_path / "assets-failed"
    calls = 0

    def fail_after_first(path: str | Path, content: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected PNG write failure")
        atomic_write_bytes(path, content)

    monkeypatch.setattr(assets_module, "atomic_write_bytes", fail_after_first)

    with pytest.raises(OSError, match="injected PNG write failure"):
        _build(validation, manifest, sample_ids, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".assets-failed.*"))
