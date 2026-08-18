"""Exact raw-instance crop and representation-cache tests on a tiny release."""

from __future__ import annotations

import copy
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import histo_audit.representations.pannuke as pannuke_representation_module
from histo_audit.pannuke import (
    PanNukeSemanticsError,
    PanNukeValidationResult,
    build_nucleus_manifest,
    validate_pannuke,
)
from histo_audit.pannuke.validation import verify_raw_inventory_unchanged
from histo_audit.representations import (
    EmbeddingResult,
    EngineeredFeatureSet,
    PanNukeCropConfig,
    PretrainedWeightsUnavailableError,
    ResNet18EmbeddingConfig,
    build_pannuke_representation_cache,
    confirmatory_cache_provenance_record,
    extract_pannuke_crop_batch,
    official_resnet18_weight_cache_path,
    ordered_sample_ids_sha256,
    save_context_morphometrics_cache,
    save_embedding_cache,
    save_engineered_feature_cache,
    save_pannuke_crop_cache,
    verify_frozen_cache_sidecar,
)
from histo_audit.representations.cache_provenance import array_artifact_sha256


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


def _validated_manifest(tmp_path: Path) -> tuple[PanNukeValidationResult, Path]:
    root = _tiny_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        expected_fold_ids=(1,),
        max_overlay_patches=1,
    )
    manifest = build_nucleus_manifest(validation, tmp_path / "manifest", batch_rows=2)
    return validation.result, manifest.parquet_path


def _validated_overlap_manifest(tmp_path: Path) -> tuple[PanNukeValidationResult, Path]:
    root = _tiny_release(tmp_path / "pannuke_overlap")
    mask_path = root / "Fold 1" / "release_arrays" / "labels.npy"
    masks = np.load(mask_path)
    masks[0, 3:5, 4:7, 2] = 606
    np.save(mask_path, masks)
    validation = validate_pannuke(
        root,
        tmp_path / "validation_overlap",
        expected_fold_ids=(1,),
        max_overlay_patches=1,
    )
    manifest = build_nucleus_manifest(validation, tmp_path / "manifest_overlap", batch_rows=2)
    return validation.result, manifest.parquet_path


def _validated_disconnected_manifest(tmp_path: Path) -> tuple[PanNukeValidationResult, Path]:
    root = _tiny_release(tmp_path / "pannuke_disconnected")
    mask_path = root / "Fold 1" / "release_arrays" / "labels.npy"
    masks = np.load(mask_path)
    masks[0, 10:12, 10:12, 0] = 101
    masks[..., 5] = (~np.any(masks[..., :5] > 0, axis=-1)).astype(np.int32)
    np.save(mask_path, masks)
    validation = validate_pannuke(
        root,
        tmp_path / "validation_disconnected",
        expected_fold_ids=(1,),
        max_overlay_patches=1,
    )
    manifest = build_nucleus_manifest(
        validation,
        tmp_path / "manifest_disconnected",
        batch_rows=2,
    )
    return validation.result, manifest.parquet_path


def _validated_overlap_disconnected_manifest(
    tmp_path: Path,
) -> tuple[PanNukeValidationResult, Path]:
    root = _tiny_release(tmp_path / "pannuke_overlap_disconnected")
    mask_path = root / "Fold 1" / "release_arrays" / "labels.npy"
    masks = np.load(mask_path)
    masks[0, 10:12, 10:12, 0] = 101
    masks[0, 3:5, 4:7, 2] = 606
    masks[..., 5] = (~np.any(masks[..., :5] > 0, axis=-1)).astype(np.int32)
    np.save(mask_path, masks)
    validation = validate_pannuke(
        root,
        tmp_path / "validation_overlap_disconnected",
        expected_fold_ids=(1,),
        max_overlay_patches=1,
    )
    manifest = build_nucleus_manifest(
        validation,
        tmp_path / "manifest_overlap_disconnected",
        batch_rows=2,
    )
    return validation.result, manifest.parquet_path


def test_component_covering_projection_repairs_lost_and_split_components() -> None:
    lost = np.zeros((91, 91), dtype=bool)
    lost[8, 23] = True
    lost[9:83, 24:68] = True
    lost_before = lost.copy()
    lost_projection = pannuke_representation_module._component_covering_projection(lost, 64)
    assert np.array_equal(lost, lost_before)
    assert lost_projection.raw_component_count == 2
    assert lost_projection.baseline_projected_component_counts == (0, 1)
    assert lost_projection.fallback_used == (True, False)
    assert lost_projection.projected_component_unique_pixel_counts[0] >= 1
    assert lost_projection.projected_union_component_count == 2

    split = np.zeros((74, 74), dtype=bool)
    split[7:16, 5:15] = True
    split[7:16, 30:40] = True
    split[11, 15:30] = True
    split_projection = pannuke_representation_module._component_covering_projection(split, 64)
    assert split_projection.raw_component_count == 1
    assert split_projection.baseline_projected_component_counts == (2,)
    assert split_projection.fallback_used == (True,)
    assert split_projection.projected_union_component_count == 1
    assert split_projection.projected_component_unique_pixel_counts[0] > 0

    repeated = pannuke_representation_module._component_covering_projection(split, 64)
    assert np.array_equal(repeated.mask, split_projection.mask)
    assert repeated.projected_component_pixel_counts == (
        split_projection.projected_component_pixel_counts
    )
    assert repeated.fallback_used == split_projection.fallback_used


def test_component_covering_projection_preserves_diagonal_component_ledger() -> None:
    source = np.zeros((32, 32), dtype=bool)
    source[8, 8] = True
    source[9, 9] = True
    source[10, 10] = True
    source[18:21, 18:21] = True
    projection = pannuke_representation_module._component_covering_projection(source, 64)

    assert projection.raw_component_count == 4
    assert projection.projected_union_component_count == 4
    assert projection.baseline_projected_component_counts == (1, 1, 1, 1)
    assert projection.fallback_used == (False, False, False, False)
    assert projection.collision_pixel_count == 0
    assert projection.projected_component_unique_pixel_counts == (
        projection.projected_component_pixel_counts
    )


def test_component_covering_crop_recovers_when_baseline_loses_every_component() -> None:
    from histo_audit.data.targets import extract_target_crop

    image = np.zeros((91, 91, 3), dtype=np.uint8)
    target = np.zeros((91, 91), dtype=bool)
    target[1, 1] = True
    target[89, 89] = True

    with pytest.raises(RuntimeError, match="target identity was lost"):
        extract_target_crop(image, target, output_size=64, padding=8)

    crop, projection, source_mask = pannuke_representation_module._component_covering_target_crop(
        image,
        target,
        output_size=64,
        padding=8,
    )
    assert source_mask.shape == (91, 91)
    assert projection.raw_component_count == 2
    assert projection.baseline_projected_component_counts == (0, 0)
    assert projection.fallback_used == (True, True)
    assert projection.projected_component_unique_pixel_counts == (1, 1)
    assert int(crop.target_mask.sum()) == 2


def test_crop_batch_recomputes_union_topology_and_projection_bounds(tmp_path: Path) -> None:
    validation, manifest_path = _validated_manifest(tmp_path)
    crops = extract_pannuke_crop_batch(validation, manifest_path)

    impossible_union = crops.projected_union_component_counts.copy()
    impossible_union[0] = int(crops.raw_component_counts[0]) + 1
    impossible_topology = crops.projection_topology_changed.copy()
    impossible_topology[0] = True
    with pytest.raises(ValueError, match="component-projection vectors are inconsistent"):
        replace(
            crops,
            projected_union_component_counts=impossible_union,
            projection_topology_changed=impossible_topology,
        ).validate()

    impossible_collision = crops.projection_collision_pixel_counts.copy()
    impossible_excess = crops.projection_collision_excess_counts.copy()
    impossible_collision[0] = 2
    impossible_excess[0] = 1
    with pytest.raises(ValueError, match="component-projection vectors are inconsistent"):
        replace(
            crops,
            projection_collision_pixel_counts=impossible_collision,
            projection_collision_excess_counts=impossible_excess,
        ).validate()


def test_manifest_rows_resolve_to_exact_raw_instance_and_contour(tmp_path: Path) -> None:
    validation, manifest_path = _validated_manifest(tmp_path)
    frame = pq.read_table(manifest_path).to_pandas()
    requested = tuple(frame["sample_id"].astype(str).iloc[[4, 0, 2]].tolist())
    crops = extract_pannuke_crop_batch(
        validation,
        manifest_path,
        sample_ids=requested,
        config=PanNukeCropConfig(output_size=40, padding=4, context_brightness=0.5),
    )

    crops.validate()
    assert tuple(crops.sample_ids) == requested
    assert crops.context_rgb.shape == (3, 40, 40, 3)
    assert crops.target_masks.dtype == bool
    assert crops.identity_verified.all()
    assert np.array_equal(
        crops.target_highlighted_rgb[crops.target_masks],
        crops.context_rgb[crops.target_masks],
    )
    assert np.all(
        crops.target_highlighted_rgb[~crops.target_masks] <= crops.context_rgb[~crops.target_masks]
    )
    for contour, target_box in zip(
        crops.source_contours_xy, crops.source_target_boxes, strict=True
    ):
        assert len(contour) > 0
        assert contour[:, 0].min() >= target_box[0]
        assert contour[:, 1].min() >= target_box[1]
        assert contour[:, 0].max() < target_box[2]
        assert contour[:, 1].max() < target_box[3]
    assert crops.metadata["source_annotations_modified"] is False
    assert len(crops.metadata["manifest_sha256"]) == 64

    cache, sidecar = save_pannuke_crop_cache(crops, tmp_path / "cache" / "crops.npz")
    with np.load(cache, allow_pickle=False) as payload:
        assert tuple(payload["sample_ids"].tolist()) == requested
        assert payload["source_contour_offsets"].shape == (len(requested) + 1,)
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["cache_npz_sha256"] == metadata["cache_file_sha256"]
    assert metadata["representation_id"] == "pannuke_component_covering_target_crops"
    assert metadata["sample_order_sha256"]
    assert metadata["manifest_sha256"] == crops.metadata["manifest_sha256"]
    assert metadata["raw_inventory_sha256"] == crops.metadata["raw_inventory_sha256"]
    assert metadata["weight_identifier"].startswith("unlearned:")
    verify_frozen_cache_sidecar(cache)


def test_crop_extraction_fails_if_manifest_instance_identity_is_wrong(tmp_path: Path) -> None:
    validation, manifest_path = _validated_manifest(tmp_path)
    table = pq.read_table(manifest_path)
    frame = table.to_pandas()
    frame.loc[0, "instance_id"] = 999999
    altered = tmp_path / "altered_manifest.parquet"
    pq.write_table(pa.Table.from_pandas(frame, schema=table.schema, preserve_index=False), altered)

    with pytest.raises(
        (ValueError, PanNukeSemanticsError),
        match=r"canonical patch/sample identity is inconsistent|is absent from its raw class channel",
    ):
        extract_pannuke_crop_batch(validation, altered)


def test_analysis_excludes_overlap_touching_instances_but_review_can_render_them(
    tmp_path: Path,
) -> None:
    validation, manifest_path = _validated_overlap_manifest(tmp_path)
    frame = pq.read_table(manifest_path).to_pandas()
    touching_ids = tuple(
        frame.loc[frame["cross_class_overlap_touching"], "sample_id"].astype(str).tolist()
    )
    assert touching_ids

    analysis = extract_pannuke_crop_batch(validation, manifest_path)
    provenance = analysis.metadata["analysis_eligibility"]
    assert not set(analysis.sample_ids.tolist()).intersection(touching_ids)
    assert provenance["manifest_excluded_instance_count"] == len(touching_ids)
    assert provenance["output_sample_count"] == len(analysis.sample_ids)
    assert analysis.primary_eligible.all()
    assert analysis.confirmatory_eligible.all()

    with pytest.raises(ValueError, match="excluded by touches_cross_class_overlap"):
        extract_pannuke_crop_batch(
            validation,
            manifest_path,
            sample_ids=(touching_ids[0],),
        )

    review = extract_pannuke_crop_batch(
        validation,
        manifest_path,
        sample_ids=(touching_ids[0],),
        eligibility_scope="review_only",
    )
    assert review.metadata["analysis_eligibility"]["selection_scope"] == "review_only"
    assert not bool(review.primary_eligible[0])
    with pytest.raises(ValueError, match="analysis-eligibility policy"):
        save_pannuke_crop_cache(review, tmp_path / "review_must_not_become_analysis.npz")


def test_disconnected_raw_identity_retains_union_geometry_and_cache_provenance(
    tmp_path: Path,
) -> None:
    validation, manifest_path = _validated_disconnected_manifest(tmp_path)
    frame = pq.read_table(manifest_path).to_pandas()
    row = frame.loc[frame["instance_id"] == 101].iloc[0]
    sample_id = str(row["sample_id"])
    assert "disconnected_instance_id" in set(row["quality_flags"])
    assert bool(row["primary_eligible"])
    assert bool(row["confirmatory_eligible"])

    raw_masks = np.load(validation.folds[0].mask_path, allow_pickle=False)
    raw_target = raw_masks[0, ..., 0] == 101
    raw_labels, component_count = pannuke_representation_module.ndimage.label(
        raw_target,
        structure=pannuke_representation_module._INSTANCE_CONNECTIVITY_4,
    )
    assert component_count == 2
    crops = extract_pannuke_crop_batch(
        validation,
        manifest_path,
        sample_ids=(sample_id,),
        config=PanNukeCropConfig(output_size=40, padding=2),
    )

    assert crops.source_target_boxes[0].tolist() == [3, 2, 12, 12]
    assert int(row["area"]) == int(raw_target.sum()) == 34
    assert bool(crops.primary_eligible[0])
    assert bool(crops.confirmatory_eligible[0])
    contour = crops.source_contours_xy[0]
    for component_id in range(1, component_count + 1):
        component_y, component_x = np.nonzero(raw_labels == component_id)
        component_coordinates = set(zip(component_x.tolist(), component_y.tolist(), strict=True))
        assert component_coordinates.intersection(map(tuple, contour.tolist()))
    policy = crops.metadata["target_mask_projection"]
    assert policy["disconnected_instance_count"] == 1
    record = policy["disconnected_instances"][0]
    assert record["sample_id"] == sample_id
    assert record["component_count"] == 2
    assert record["area"] == int(raw_target.sum())
    assert record["bbox"] == [3, 2, 12, 12]
    assert record["raw_target_mask_sha256"] == array_artifact_sha256(raw_target)
    assert record["primary_eligible"] is True
    assert record["confirmatory_eligible"] is True
    assert crops.raw_component_counts.tolist() == [2]
    assert crops.disconnected_instance_flags.tolist() == [True]
    assert crops.projected_component_offsets.tolist() == [0, 2]
    assert crops.projected_component_unique_pixel_counts.shape == (2,)

    cache, sidecar = save_pannuke_crop_cache(crops, tmp_path / "cache" / "disconnected.npz")
    verify_frozen_cache_sidecar(cache)
    with np.load(cache, allow_pickle=False) as payload:
        assert payload["raw_component_counts"].tolist() == [2]
        assert payload["disconnected_instance_flags"].tolist() == [True]
        assert payload["projected_component_offsets"].tolist() == [0, 2]
        assert np.all(payload["projected_component_pixel_counts"] > 0)
        assert np.all(payload["projected_component_unique_pixel_counts"] > 0)
    saved_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert saved_metadata["target_mask_projection"] == policy
    assert verify_raw_inventory_unchanged(validation) == validation.inventory


def test_mixed_crop_batch_binds_component_vectors_to_disconnected_ledger(
    tmp_path: Path,
) -> None:
    validation, manifest_path = _validated_disconnected_manifest(tmp_path)
    frame = pq.read_table(manifest_path).to_pandas()
    disconnected_id = str(frame.loc[frame["instance_id"] == 101, "sample_id"].iloc[0])
    connected_id = str(frame.loc[frame["instance_id"] == 202, "sample_id"].iloc[0])
    crops = extract_pannuke_crop_batch(
        validation,
        manifest_path,
        sample_ids=(connected_id, disconnected_id),
        config=PanNukeCropConfig(output_size=40, padding=2),
    )

    assert crops.raw_component_counts.tolist() == [1, 2]
    assert crops.disconnected_instance_flags.tolist() == [False, True]
    assert crops.projected_component_offsets.tolist() == [0, 1, 3]
    records = crops.metadata["target_mask_projection"]["disconnected_instances"]
    assert [record["sample_id"] for record in records] == [disconnected_id]
    assert records[0]["projected_component_pixel_counts"] == (
        crops.projected_component_pixel_counts[1:3].tolist()
    )

    tampered = copy.deepcopy(crops.metadata["target_mask_projection"])
    tampered["disconnected_instances"] = []
    tampered["disconnected_instance_count"] = 0
    tampered.pop("semantic_sha256")
    tampered["semantic_sha256"] = pannuke_representation_module.canonical_sha256(tampered)
    crops.metadata["target_mask_projection"] = tampered
    with pytest.raises(ValueError, match="target-mask projection policy is invalid"):
        crops.validate()


def test_overlap_excluded_disconnected_identity_remains_renderable_review_only(
    tmp_path: Path,
) -> None:
    validation, manifest_path = _validated_overlap_disconnected_manifest(tmp_path)
    frame = pq.read_table(manifest_path).to_pandas()
    row = frame.loc[frame["instance_id"] == 101].iloc[0]
    sample_id = str(row["sample_id"])
    assert "disconnected_instance_id" in set(row["quality_flags"])
    assert bool(row["cross_class_overlap_touching"])
    assert not bool(row["primary_eligible"])

    with pytest.raises(ValueError, match="excluded by touches_cross_class_overlap"):
        extract_pannuke_crop_batch(validation, manifest_path, sample_ids=(sample_id,))
    review = extract_pannuke_crop_batch(
        validation,
        manifest_path,
        sample_ids=(sample_id,),
        eligibility_scope="review_only",
    )
    review.validate()
    assert review.raw_component_counts.tolist() == [2]
    assert review.disconnected_instance_flags.tolist() == [True]
    assert review.primary_eligible.tolist() == [False]
    assert review.confirmatory_eligible.tolist() == [False]
    record = review.metadata["target_mask_projection"]["disconnected_instances"][0]
    assert record["primary_eligible"] is False
    assert record["confirmatory_eligible"] is False


def test_disconnected_component_flag_must_match_raw_target(tmp_path: Path) -> None:
    validation, manifest_path = _validated_disconnected_manifest(tmp_path)
    table = pq.read_table(manifest_path)
    frame = table.to_pandas()
    target = frame["instance_id"] == 101
    frame.loc[target, "quality_flags"] = frame.loc[target, "quality_flags"].map(
        lambda values: [value for value in values if value != "disconnected_instance_id"]
    )
    unflagged = tmp_path / "unflagged-disconnected.parquet"
    pq.write_table(
        pa.Table.from_pandas(frame, schema=table.schema, preserve_index=False),
        unflagged,
    )
    sample_id = str(frame.loc[target, "sample_id"].iloc[0])
    with pytest.raises(
        PanNukeSemanticsError,
        match="raw target is disconnected but manifest lacks disconnected_instance_id",
    ):
        extract_pannuke_crop_batch(validation, unflagged, sample_ids=(sample_id,))

    connected_validation, connected_manifest = _validated_manifest(tmp_path / "connected")
    connected_table = pq.read_table(connected_manifest)
    connected_frame = connected_table.to_pandas()
    connected_frame.at[0, "quality_flags"] = [
        *connected_frame.at[0, "quality_flags"],
        "disconnected_instance_id",
    ]
    falsely_flagged = tmp_path / "falsely-flagged-connected.parquet"
    pq.write_table(
        pa.Table.from_pandas(
            connected_frame,
            schema=connected_table.schema,
            preserve_index=False,
        ),
        falsely_flagged,
    )
    connected_sample_id = str(connected_frame.at[0, "sample_id"])
    with pytest.raises(
        PanNukeSemanticsError,
        match="manifest disconnected_instance_id differs from connected raw target",
    ):
        extract_pannuke_crop_batch(
            connected_validation,
            falsely_flagged,
            sample_ids=(connected_sample_id,),
        )


def _flip_first_npy_data_byte(path: Path) -> None:
    mapped = np.load(path, mmap_mode="r", allow_pickle=False)
    offset = int(mapped.offset)
    del mapped
    with path.open("r+b") as handle:
        handle.seek(offset)
        original = handle.read(1)
        handle.seek(offset)
        handle.write(bytes((original[0] ^ 1,)))


def test_raw_mutation_during_later_crop_fails_before_any_cache_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation, manifest_path = _validated_manifest(tmp_path)
    image_path = validation.folds[0].image_path
    original = pannuke_representation_module._component_covering_projection
    mutated = False

    def mutate_after_first_raw_crop(*args: object, **kwargs: object):
        nonlocal mutated
        result = original(*args, **kwargs)
        if not mutated:
            _flip_first_npy_data_byte(image_path)
            mutated = True
        return result

    monkeypatch.setattr(
        pannuke_representation_module,
        "_component_covering_projection",
        mutate_after_first_raw_crop,
    )
    output = tmp_path / "race-output"
    with pytest.raises(PanNukeSemanticsError, match="raw inventory changed"):
        build_pannuke_representation_cache(validation, manifest_path, output)
    assert mutated
    assert not list(output.glob("*.npz"))


def test_manifest_tissue_change_during_crop_fails_and_binds_pre_read_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation, manifest_path = _validated_manifest(tmp_path)
    original_table = pq.read_table(manifest_path)
    frame = original_table.to_pandas()
    frame.loc[1, "tissue_type"] = "MutatedTissue"
    replacement = tmp_path / "replacement-manifest.parquet"
    pq.write_table(
        pa.Table.from_pandas(frame, schema=original_table.schema, preserve_index=False),
        replacement,
    )
    pre_read_sha = pannuke_representation_module.sha256_file(manifest_path)
    original = pannuke_representation_module._component_covering_projection
    mutated = False

    def replace_manifest_after_first_crop(*args: object, **kwargs: object):
        nonlocal mutated
        result = original(*args, **kwargs)
        if not mutated:
            shutil.copyfile(replacement, manifest_path)
            mutated = True
        return result

    monkeypatch.setattr(
        pannuke_representation_module,
        "_component_covering_projection",
        replace_manifest_after_first_crop,
    )
    output = tmp_path / "manifest-race-output"
    with pytest.raises(PanNukeSemanticsError, match="manifest changed during crop extraction"):
        build_pannuke_representation_cache(validation, manifest_path, output)
    assert mutated
    assert pannuke_representation_module.sha256_file(manifest_path) != pre_read_sha
    assert not list(output.glob("*.npz"))


def test_raw_mutation_during_embedding_rolls_back_entire_representation_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation, manifest_path = _validated_manifest(tmp_path)
    image_path = validation.folds[0].image_path

    def mutate_then_return_embedding(
        images: np.ndarray,
        sample_ids: object,
        **_: object,
    ) -> EmbeddingResult:
        _flip_first_npy_data_byte(image_path)
        identifiers = np.asarray(sample_ids, dtype=np.str_)
        return EmbeddingResult(
            embeddings=np.zeros((len(images), 512), dtype=np.float32),
            sample_ids=identifiers,
            metadata={"weight_sha256": "d" * 64},
        )

    monkeypatch.setattr(
        pannuke_representation_module,
        "extract_resnet18_embeddings",
        mutate_then_return_embedding,
    )
    destination = tmp_path / "representation-transaction"
    with pytest.raises(PanNukeSemanticsError, match="raw inventory changed"):
        build_pannuke_representation_cache(validation, manifest_path, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".representation-transaction.*"))


def test_representation_post_publish_source_check_rolls_back_owned_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation, manifest_path = _validated_manifest(tmp_path)
    destination = tmp_path / "representation-post-publish"
    original_verify = pannuke_representation_module.verify_raw_inventory_unchanged
    mutated = False

    def mutate_after_directory_promotion(binding: PanNukeValidationResult):
        nonlocal mutated
        if destination.exists() and not mutated:
            _flip_first_npy_data_byte(binding.folds[0].image_path)
            mutated = True
        return original_verify(binding)

    def return_embedding(
        images: np.ndarray,
        sample_ids: object,
        **_: object,
    ) -> EmbeddingResult:
        return EmbeddingResult(
            embeddings=np.zeros((len(images), 512), dtype=np.float32),
            sample_ids=np.asarray(sample_ids, dtype=np.str_),
            metadata={"weight_sha256": "d" * 64},
        )

    monkeypatch.setattr(
        pannuke_representation_module,
        "verify_raw_inventory_unchanged",
        mutate_after_directory_promotion,
    )
    monkeypatch.setattr(
        pannuke_representation_module,
        "extract_resnet18_embeddings",
        return_embedding,
    )
    with pytest.raises(PanNukeSemanticsError, match="raw inventory changed"):
        build_pannuke_representation_cache(validation, manifest_path, destination)
    assert mutated
    assert not destination.exists()


def test_stale_crop_batch_cannot_publish_after_raw_mutation(tmp_path: Path) -> None:
    validation, manifest_path = _validated_manifest(tmp_path)
    crops = extract_pannuke_crop_batch(validation, manifest_path)
    _flip_first_npy_data_byte(validation.folds[0].image_path)
    destination = tmp_path / "stale-cache" / "crops.npz"

    with pytest.raises(PanNukeSemanticsError, match="raw inventory changed"):
        save_pannuke_crop_cache(crops, destination)
    assert not destination.exists()
    assert not destination.with_suffix(".npz.metadata.json").exists()


def test_stale_crop_batch_cannot_publish_after_manifest_mutation(tmp_path: Path) -> None:
    validation, manifest_path = _validated_manifest(tmp_path)
    crops = extract_pannuke_crop_batch(validation, manifest_path)
    original_table = pq.read_table(manifest_path)
    frame = original_table.to_pandas()
    frame.loc[0, "tissue_type"] = "MutatedAfterExtraction"
    pq.write_table(
        pa.Table.from_pandas(frame, schema=original_table.schema, preserve_index=False),
        manifest_path,
    )
    destination = tmp_path / "stale-manifest-cache" / "crops.npz"

    with pytest.raises(
        PanNukeSemanticsError,
        match="manifest changed before crop-cache publication",
    ):
        save_pannuke_crop_cache(crops, destination)

    assert not destination.exists()
    assert not destination.with_suffix(".npz.metadata.json").exists()


def test_crop_cache_post_publish_source_check_rolls_back_owned_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation, manifest_path = _validated_manifest(tmp_path)
    crops = extract_pannuke_crop_batch(validation, manifest_path)
    original = pannuke_representation_module.verify_raw_inventory_unchanged
    checks = 0

    def mutate_on_post_publish(binding: PanNukeValidationResult):
        nonlocal checks
        checks += 1
        if checks == 3:
            _flip_first_npy_data_byte(binding.folds[0].image_path)
        return original(binding)

    monkeypatch.setattr(
        pannuke_representation_module,
        "verify_raw_inventory_unchanged",
        mutate_on_post_publish,
    )
    destination = tmp_path / "post-publish-race" / "crops.npz"
    with pytest.raises(PanNukeSemanticsError, match="raw inventory changed"):
        save_pannuke_crop_cache(crops, destination)
    assert checks == 3
    assert not destination.exists()
    assert not destination.with_suffix(".npz.metadata.json").exists()


def test_representation_output_traversal_is_rejected_without_raw_tree_change(
    tmp_path: Path,
) -> None:
    validation, manifest_path = _validated_manifest(tmp_path)
    destination = validation.root / ".." / validation.root.name / "derived-representations"

    with pytest.raises(PanNukeSemanticsError, match="outside the immutable PanNuke raw release"):
        build_pannuke_representation_cache(validation, manifest_path, destination)
    assert not destination.resolve().exists()
    assert verify_raw_inventory_unchanged(validation) == validation.inventory


def test_standalone_crop_cache_destination_inside_raw_is_rejected(tmp_path: Path) -> None:
    validation, manifest_path = _validated_manifest(tmp_path)
    crops = extract_pannuke_crop_batch(validation, manifest_path)
    destination = validation.root / "forbidden-crops.npz"

    with pytest.raises(PanNukeSemanticsError, match="outside the immutable PanNuke raw release"):
        save_pannuke_crop_cache(crops, destination)
    assert not destination.exists()
    assert not destination.with_suffix(".npz.metadata.json").exists()
    assert verify_raw_inventory_unchanged(validation) == validation.inventory


def test_representation_output_symlink_into_raw_is_rejected(tmp_path: Path) -> None:
    validation, manifest_path = _validated_manifest(tmp_path)
    link = tmp_path / "raw-link"
    try:
        os.symlink(validation.root, link, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    destination = link / "derived-representations"

    with pytest.raises(PanNukeSemanticsError, match="outside the immutable PanNuke raw release"):
        build_pannuke_representation_cache(validation, manifest_path, destination)
    assert not (validation.root / "derived-representations").exists()
    assert verify_raw_inventory_unchanged(validation) == validation.inventory


def _context_morphometrics_lineage_fixture(
    tmp_path: Path,
) -> tuple[EmbeddingResult, EngineeredFeatureSet, dict[str, object], np.ndarray, str, str]:
    sample_ids = np.asarray(["sample-a", "sample-b"], dtype=np.str_)
    manifest_sha256 = "a" * 64
    raw_inventory_sha256 = "b" * 64
    common_crop_binding = {
        "crop_cache_file_sha256": "c" * 64,
        "crop_cache_sidecar_file_sha256": "d" * 64,
        "crop_cache_content_sha256": "e" * 64,
        "crop_manifest_sha256": manifest_sha256,
        "raw_inventory_sha256": raw_inventory_sha256,
        "sample_order_sha256": ordered_sample_ids_sha256(sample_ids),
        "target_mask_projection_semantic_sha256": "f" * 64,
    }
    context_crop_binding = {
        "schema_version": 1,
        "binding_type": "pannuke_component_covering_crop_cache_v1",
        **common_crop_binding,
        "input_variant": "context_rgb",
        "input_array_key": "context_rgb",
        "input_array_sha256": "1" * 64,
    }
    context_values = np.zeros((2, 512), dtype=np.float32)
    context_path = tmp_path / "context.npz"
    _, _, context_metadata = save_embedding_cache(
        context_path,
        context_values,
        sample_ids,
        {
            "input_variant": "rgb",
            "weight_sha256": "2" * 64,
            "manifest_sha256": manifest_sha256,
            "raw_inventory_sha256": raw_inventory_sha256,
            "source_crop_cache_binding": context_crop_binding,
        },
    )
    context = EmbeddingResult(
        embeddings=context_values,
        sample_ids=sample_ids,
        metadata=context_metadata,
        cache_path=context_path,
        metadata_path=context_path.with_suffix(".npz.metadata.json"),
    )
    engineered = EngineeredFeatureSet(
        values=np.asarray([[1.0, 10.0], [2.0, 20.0]], dtype=np.float64),
        names=("morphology.area", "mask.count"),
    )
    engineered_source_binding = {
        "schema_version": 1,
        "binding_type": "pannuke_component_covering_crop_cache_engineered_v1",
        **common_crop_binding,
        "input_variant": "context_rgb_plus_component_covering_target_masks",
        "input_array_sha256_by_name": {
            "context_rgb": "1" * 64,
            "target_masks": "3" * 64,
        },
    }
    engineered_path, _, _ = save_engineered_feature_cache(
        engineered,
        sample_ids,
        tmp_path / "engineered.npz",
        manifest_sha256=manifest_sha256,
        raw_inventory_sha256=raw_inventory_sha256,
        source_crop_cache_binding=engineered_source_binding,
    )
    binding = pannuke_representation_module._engineered_cache_binding(engineered_path)
    return (
        context,
        engineered,
        binding,
        sample_ids,
        manifest_sha256,
        raw_inventory_sha256,
    )


def test_context_morphometrics_rejects_cache_a_binding_with_features_b(
    tmp_path: Path,
) -> None:
    context, engineered_a, binding_a, sample_ids, manifest_sha, raw_sha = (
        _context_morphometrics_lineage_fixture(tmp_path)
    )
    engineered_b = EngineeredFeatureSet(
        values=engineered_a.values + np.asarray([[100.0, 0.0], [100.0, 0.0]]),
        names=engineered_a.names,
    )

    with pytest.raises(ValueError, match="binding differs from in-memory features"):
        save_context_morphometrics_cache(
            context,
            engineered_b,
            sample_ids,
            tmp_path / "mismatched-context-morphometrics.npz",
            manifest_sha256=manifest_sha,
            raw_inventory_sha256=raw_sha,
            engineered_cache_binding=binding_a,
        )
    assert not (tmp_path / "mismatched-context-morphometrics.npz").exists()


def test_context_morphometrics_rejects_omitted_stage_lineage_binding(
    tmp_path: Path,
) -> None:
    context, engineered, _, sample_ids, manifest_sha, raw_sha = (
        _context_morphometrics_lineage_fixture(tmp_path)
    )

    with pytest.raises(ValueError, match="require an exact engineered cache binding"):
        save_context_morphometrics_cache(
            context,
            engineered,
            sample_ids,
            tmp_path / "unbound-context-morphometrics.npz",
            manifest_sha256=manifest_sha,
            raw_inventory_sha256=raw_sha,
        )
    assert not (tmp_path / "unbound-context-morphometrics.npz").exists()


def test_unbound_context_morphometrics_is_fixture_only_and_cannot_enter_confirmatory(
    tmp_path: Path,
) -> None:
    _, engineered, _, sample_ids, manifest_sha, raw_sha = _context_morphometrics_lineage_fixture(
        tmp_path / "lineage"
    )
    context_values = np.zeros((len(sample_ids), 512), dtype=np.float32)
    context_path = tmp_path / "unbound-context.npz"
    _, _, context_metadata = save_embedding_cache(
        context_path,
        context_values,
        sample_ids,
        {
            "input_variant": "rgb",
            "weight_sha256": "2" * 64,
            "manifest_sha256": manifest_sha,
            "raw_inventory_sha256": raw_sha,
        },
    )
    context = EmbeddingResult(
        embeddings=context_values,
        sample_ids=sample_ids,
        metadata=context_metadata,
        cache_path=context_path,
        metadata_path=context_path.with_suffix(".npz.metadata.json"),
    )
    result = save_context_morphometrics_cache(
        context,
        engineered,
        sample_ids,
        tmp_path / "fixture-only-context-morphometrics.npz",
        manifest_sha256=manifest_sha,
        raw_inventory_sha256=raw_sha,
    )

    assert result.metadata["provenance_scope"] == "non_stage_fixture"
    assert result.metadata["lineage_binding_status"] == "absent_non_stage_fixture"
    with pytest.raises(ValueError, match="stage-eligible"):
        confirmatory_cache_provenance_record(
            result.metadata,
            record_id="fixture_only_context_morphometrics",
        )


def test_real_representation_orchestrator_fails_closed_without_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation, manifest_path = _validated_manifest(tmp_path)
    empty_hub = tmp_path / "empty_torch_hub"
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(empty_hub))
    with pytest.raises(PretrainedWeightsUnavailableError, match="allow_weight_download=True"):
        build_pannuke_representation_cache(
            validation,
            manifest_path,
            tmp_path / "representations",
            crop_config=PanNukeCropConfig(output_size=32, padding=3),
            resnet_config=ResNet18EmbeddingConfig(
                input_variant="target_highlighted_rgb",
                context_brightness=0.45,
                device="cpu",
                batch_size=2,
                allow_weight_download=False,
            ),
        )
    assert not empty_hub.exists()


def test_real_representation_orchestrator_uses_cached_official_weights(tmp_path: Path) -> None:
    weight_path = official_resnet18_weight_cache_path("IMAGENET1K_V1")
    if not weight_path.is_file():
        pytest.skip("official ResNet-18 weights are not cached; this test never downloads them")
    validation, manifest_path = _validated_manifest(tmp_path)
    sample_ids = tuple(pq.read_table(manifest_path).column("sample_id").to_pylist()[:2])
    artifacts = build_pannuke_representation_cache(
        validation,
        manifest_path,
        tmp_path / "representations",
        sample_ids=sample_ids,
        crop_config=PanNukeCropConfig(output_size=32, padding=3),
        resnet_config=ResNet18EmbeddingConfig(
            input_variant="target_highlighted_rgb",
            context_brightness=0.45,
            device="cpu",
            batch_size=2,
            allow_weight_download=False,
        ),
        include_context_embeddings=True,
    )

    assert artifacts.embeddings.embeddings.shape == (2, 512)
    assert tuple(artifacts.embeddings.sample_ids.tolist()) == sample_ids
    assert artifacts.embeddings.metadata["encoder_frozen"] is True
    assert artifacts.embeddings.metadata["weight_sha256"]
    assert artifacts.context_embeddings is not None
    assert artifacts.context_embeddings.embeddings.shape == (2, 512)
    assert tuple(artifacts.context_embeddings.sample_ids.tolist()) == sample_ids
    assert artifacts.context_embeddings.metadata["input_variant"] == "rgb"
    assert (
        artifacts.context_embeddings.metadata["weight_sha256"]
        == artifacts.embeddings.metadata["weight_sha256"]
    )
    assert artifacts.context_embeddings.cache_path is not None
    assert artifacts.context_embeddings.cache_path.name == (
        "pannuke_resnet18_context_rgb_embeddings.npz"
    )
    assert artifacts.context_morphometrics is not None
    assert artifacts.context_morphometrics.values.shape[0] == 2
    assert artifacts.context_morphometrics.values.shape[1] > 512
    assert artifacts.context_morphometrics.metadata["representation_id"] == (
        "imagenet_context_embeddings_plus_target_morphometrics"
    )
    verify_frozen_cache_sidecar(artifacts.context_morphometrics.cache_path)
    assert artifacts.crop_cache_path.is_file()
    assert artifacts.engineered_cache_path.is_file()
    crop_verification = verify_frozen_cache_sidecar(artifacts.crop_cache_path)
    engineered_verification = verify_frozen_cache_sidecar(artifacts.engineered_cache_path)
    engineered_binding = engineered_verification.metadata["source_crop_cache_binding"]
    assert engineered_binding["crop_cache_file_sha256"] == crop_verification.cache_file_sha256
    assert (
        engineered_binding["crop_cache_content_sha256"]
        == crop_verification.metadata["cache_content_sha256"]
    )
    assert engineered_binding["input_array_sha256_by_name"] == {
        key: crop_verification.metadata["cache_array_sha256_by_name"][key]
        for key in ("context_rgb", "target_masks")
    }
    assert artifacts.context_morphometrics is not None
    morphometrics_binding = artifacts.context_morphometrics.metadata[
        "component_engineered_cache_binding"
    ]
    assert morphometrics_binding["engineered_cache_file_sha256"] == (
        engineered_verification.cache_file_sha256
    )
    assert (
        morphometrics_binding["engineered_cache_content_sha256"]
        == (engineered_verification.metadata["cache_content_sha256"])
    )
    for result, variant, array_key in (
        (artifacts.embeddings, "target_highlighted_rgb", "target_highlighted_rgb"),
        (artifacts.context_embeddings, "context_rgb", "context_rgb"),
    ):
        assert result is not None
        binding = result.metadata["source_crop_cache_binding"]
        assert binding["crop_cache_file_sha256"] == crop_verification.cache_file_sha256
        assert binding["crop_cache_sidecar_file_sha256"] == crop_verification.sidecar_file_sha256
        assert (
            binding["crop_cache_content_sha256"]
            == crop_verification.metadata["cache_content_sha256"]
        )
        assert (
            binding["target_mask_projection_semantic_sha256"]
            == artifacts.crops.metadata["target_mask_projection"]["semantic_sha256"]
        )
        assert binding["input_variant"] == variant
        assert binding["input_array_key"] == array_key
        assert (
            binding["input_array_sha256"]
            == crop_verification.metadata["cache_array_sha256_by_name"][array_key]
        )
