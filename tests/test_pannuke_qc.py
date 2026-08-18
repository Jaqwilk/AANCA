"""Focused regression tests for anomaly-safe PanNuke mask QC."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import histo_audit.pannuke.qc as qc_module
import histo_audit.pannuke.validation as validation_module
from histo_audit.pannuke.exceptions import PanNukeSemanticsError
from histo_audit.pannuke.models import DiscoveredFold
from histo_audit.pannuke.qc import (
    analyse_patch_mask_qc,
    select_anomaly_overlay_patches,
    summarise_fold_mask_qc,
    summarise_global_mask_qc,
)
from histo_audit.pannuke.validation import (
    _instance_ids_and_disconnected_components,
    validate_pannuke,
    verify_raw_inventory_unchanged,
)

_CLASS_NAMES = ("neoplastic", "inflammatory", "connective", "dead", "epithelial")
_POSITIVE_CHANNELS = (0, 1, 2, 3, 4)


def _normal_mask(*, height: int = 13, width: int = 11) -> np.ndarray:
    mask = np.zeros((height, width, 6), dtype=np.int32)
    mask[2:5, 2:5, 0] = 11
    mask[7:10, 6:9, 2] = 33
    mask[..., 5] = (~np.any(mask[..., :5] > 0, axis=-1)).astype(np.int32)
    return mask


def _anomalous_mask() -> np.ndarray:
    mask = _normal_mask()
    mask[4:7, 4:7, 1] = 22
    mask[..., 5] = (~np.any(mask[..., :5] > 0, axis=-1)).astype(np.int32)
    # (4, 4) belongs to class channels 0 and 1.  Both identities are retained.
    # One void pixel and one separate positive+background conflict exercise the
    # independent states without changing any source value in the analyser.
    mask[0, 0, 5] = 0
    mask[6, 6, 5] = 1
    return mask


def _analyse(mask: np.ndarray, *, patch_index: int = 0):
    return analyse_patch_mask_qc(
        mask,
        fold_id=1,
        patch_index=patch_index,
        class_names=_CLASS_NAMES,
        positive_channel_indices=_POSITIVE_CHANNELS,
        background_channel_index=5,
    )


def _write_validation_release(root: Path, *, fold_count: int) -> Path:
    y_grid, x_grid = np.mgrid[:13, :11]
    for fold_id in range(1, fold_count + 1):
        fold = root / f"Fold {fold_id}" / "release_arrays"
        fold.mkdir(parents=True)
        image = np.empty((1, 13, 11, 3), dtype=np.uint8)
        image[0, ..., 0] = (x_grid * 13 + fold_id * 17) % 256
        image[0, ..., 1] = (y_grid * 11 + fold_id * 19) % 256
        image[0, ..., 2] = ((x_grid + y_grid) * 7 + fold_id * 23) % 256
        np.save(fold / "pixels.npy", image)
        np.save(fold / "labels.npy", _normal_mask()[None, ...])
        np.save(fold / "organs.npy", np.asarray(["Breast"], dtype="<U16"))
    return root


def test_normal_mask_has_no_anomaly_affected_instances_or_per_instance_full_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask = _normal_mask()
    before = mask.copy()
    unique_input_sizes: list[int] = []
    real_unique = np.unique

    def recording_unique(values: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        unique_input_sizes.append(int(np.asarray(values).size))
        return real_unique(values, *args, **kwargs)

    monkeypatch.setattr(qc_module.np, "unique", recording_unique)
    record = _analyse(mask)

    assert np.array_equal(mask, before)
    assert record.void_pixel_count == 0
    assert record.cross_class_overlap_pixel_count == 0
    assert record.positive_and_background_pixel_count == 0
    assert record.anomaly_union_pixel_count == 0
    assert record.affected_instances == ()
    assert record.affected_instance_count == 0
    # There is one call per class, but each receives only anomaly-pixel values.
    assert unique_input_sizes == [0, 0, 0, 0, 0]
    assert set(record.mask_sha256_by_kind) == {
        "positive_any",
        "supplied_background",
        "void_unlabelled",
        "cross_class_overlap",
        "positive_and_background",
        "anomaly_union",
    }
    assert all(len(value) == 64 for value in record.mask_sha256_by_kind.values())
    assert record.mask_sha256_by_kind == _analyse(mask).mask_sha256_by_kind


def test_overlap_void_and_positive_background_are_separate_without_class_arbitration() -> None:
    mask = _anomalous_mask()
    before = mask.copy()
    record = _analyse(mask)

    assert np.array_equal(mask, before)
    assert record.cross_class_overlap_pixel_count == 1
    assert record.void_pixel_count == 1
    assert record.positive_and_background_pixel_count == 1
    assert record.anomaly_union_pixel_count == 3
    assert record.has_cross_class_overlap and record.has_void
    assert record.has_positive_and_background
    assert record.affected_instance_count == 2
    by_target = {(item.channel_index, item.instance_id): item for item in record.affected_instances}
    class_zero = by_target[(0, 11)]
    class_one = by_target[(1, 22)]
    assert class_zero.overlap_pixel_count == 1
    assert class_zero.positive_background_pixel_count == 0
    assert class_zero.overlapping_class_indices == (1,)
    assert class_zero.overlapping_instance_ids == (22,)
    assert class_zero.overlapping_instance_ids_by_class == {"inflammatory": (22,)}
    assert class_one.overlap_pixel_count == 1
    assert class_one.positive_background_pixel_count == 1
    assert class_one.overlapping_class_indices == (0,)
    assert class_one.overlapping_instance_ids == (11,)
    assert class_one.overlapping_instance_ids_by_class == {"neoplastic": (11,)}
    # The target itself is never inserted into the "other raw identity" fields.
    assert 11 not in class_zero.overlapping_instance_ids
    assert 22 not in class_one.overlapping_instance_ids


def test_fold_global_summaries_and_overlay_selection_reconcile_every_patch() -> None:
    anomalous = _analyse(_anomalous_mask(), patch_index=0)
    normal = _analyse(_normal_mask(), patch_index=1)
    fold = summarise_fold_mask_qc(1, (normal, anomalous))
    global_qc = summarise_global_mask_qc((fold,))
    selection = select_anomaly_overlay_patches((fold,), max_patches=4)

    assert tuple(item.patch_index for item in fold.patches) == (0, 1)
    assert fold.patch_count == 2
    assert fold.cross_class_overlap_patch_indices == (0,)
    assert fold.void_patch_indices == (0,)
    assert fold.positive_and_background_patch_indices == (0,)
    assert fold.normal_patch_count == 1
    assert fold.overlap_touching_instance_count == 2
    assert global_qc.cross_class_overlap_pixel_count == 1
    assert global_qc.void_pixel_count == 1
    assert global_qc.positive_and_background_pixel_count == 1
    assert selection.selected_patch_keys == ("fold_1:patch_0", "fold_1:patch_1")
    assert selection.category_candidate_counts == {
        "cross_class_overlap": 1,
        "positive_and_background": 1,
        "void_unlabelled": 1,
        "normal": 1,
    }


def test_structurally_invalid_background_and_mask_values_remain_fatal() -> None:
    non_binary = _normal_mask()
    non_binary[0, 0, 5] = 2
    with pytest.raises(PanNukeSemanticsError, match="background channel is not binary"):
        _analyse(non_binary)

    negative = _normal_mask().astype(np.float32)
    negative[0, 0, 0] = -1
    with pytest.raises(PanNukeSemanticsError, match="instance IDs are negative"):
        _analyse(negative)

    non_integer = _normal_mask().astype(np.float32)
    non_integer[2, 2, 0] = 1.5
    with pytest.raises(PanNukeSemanticsError, match="not integer-like"):
        _analyse(non_integer)


def test_dense_bounding_box_component_check_preserves_connectedness_semantics() -> None:
    channel = np.zeros((13, 11), dtype=np.int64)
    channel[1:4, 1:4] = 7
    channel[7:10, 6:9] = 1_000_000_000
    ids, disconnected = _instance_ids_and_disconnected_components(channel)

    assert ids.tolist() == [7, 1_000_000_000]
    assert disconnected == {}

    channel[8, 6:9] = 0
    ids, disconnected = _instance_ids_and_disconnected_components(channel)
    assert ids.tolist() == [7, 1_000_000_000]
    assert disconnected == {1_000_000_000: 2}


def test_validation_accepts_release_anomalies_and_persists_shared_exclusion_policy(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pannuke"
    fold = root / "Fold 1" / "release_arrays"
    fold.mkdir(parents=True)
    y_grid, x_grid = np.mgrid[:13, :11]
    image = np.empty((1, 13, 11, 3), dtype=np.uint8)
    image[0, ..., 0] = (x_grid * 13 + 17) % 256
    image[0, ..., 1] = (y_grid * 11 + 19) % 256
    image[0, ..., 2] = ((x_grid + y_grid) * 7 + 23) % 256
    mask = _anomalous_mask()[None, ...]
    tissue = np.asarray(["Breast"], dtype="<U16")
    np.save(fold / "pixels.npy", image)
    np.save(fold / "labels.npy", mask)
    np.save(fold / "organs.npy", tissue)

    artifacts = validate_pannuke(
        root,
        tmp_path / "validation",
        expected_fold_ids=(1,),
        max_overlay_patches=1,
    )
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))

    assert payload["status"] == "valid"
    assert payload["global_mask_qc"]["cross_class_overlap_pixel_count"] == 1
    assert payload["global_mask_qc"]["void_pixel_count"] == 1
    assert payload["global_mask_qc"]["positive_and_background_pixel_count"] == 1
    assert payload["global_mask_qc"]["overlap_touching_instance_count"] == 2
    policy = payload["qc_policy"]
    assert policy["supplied_background_is_exact_complement_required"] is False
    assert policy["no_class_arbitration"] is True
    assert policy["source_masks_modified"] is False
    assert policy["release_annotation_anomalies_are_fatal"] is False
    assert policy["structural_invalidity_is_fatal"] is True
    assert policy["disconnected_instance_ids_are_fatal"] is False
    assert "retain raw identity" in policy["disconnected_instance_action"]
    assert policy["analysis_instance_exclusion_reason"] == "touches_cross_class_overlap"
    assert policy["applies_identically_to_primary_and_confirmatory"] is True
    assert payload["anomaly_overlay_selection"]["requested_max_patches"] == 1
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    assert "unlabeled (`void`)" in markdown
    assert "never arbitrate a class" in markdown
    assert "shared primary/confirmatory analysis exclusion reason" in markdown


def test_validation_fails_when_an_already_scanned_fold_changes_during_later_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_validation_release(tmp_path / "pannuke", fold_count=2)
    original_validate_fold = validation_module._validate_fold
    mutation_performed = False

    def mutate_then_validate(fold: DiscoveredFold, **kwargs: object):
        nonlocal mutation_performed
        if fold.fold_id == 2 and not mutation_performed:
            first_image_path = root / "Fold 1" / "release_arrays" / "pixels.npy"
            first_images = np.load(first_image_path)
            first_images[0, 0, 0, 0] = (int(first_images[0, 0, 0, 0]) + 1) % 256
            np.save(first_image_path, first_images)
            mutation_performed = True
        return original_validate_fold(fold, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(validation_module, "_validate_fold", mutate_then_validate)
    output = tmp_path / "validation"
    with pytest.raises(PanNukeSemanticsError, match="raw inventory changed during semantic"):
        validate_pannuke(
            root,
            output,
            expected_fold_ids=(1, 2),
            max_overlay_patches=1,
        )

    assert mutation_performed
    assert not (output / "pannuke_validation.json").exists()


def test_public_inventory_verifier_rehashes_every_file_and_detects_later_change(
    tmp_path: Path,
) -> None:
    root = _write_validation_release(tmp_path / "pannuke", fold_count=1)
    temporary_looking_raw = root / ".must_not_be_hidden.tmp"
    temporary_looking_raw.write_bytes(b"raw inventory evidence")
    artifacts = validate_pannuke(
        root,
        tmp_path / "validation",
        expected_fold_ids=(1,),
        max_overlay_patches=1,
    )

    assert ".must_not_be_hidden.tmp" in {item.relative_path for item in artifacts.result.inventory}
    assert verify_raw_inventory_unchanged(artifacts.result) == artifacts.result.inventory
    tissue_path = root / "Fold 1" / "release_arrays" / "organs.npy"
    np.save(tissue_path, np.asarray(["Colon"], dtype="<U16"))
    with pytest.raises(PanNukeSemanticsError, match="raw inventory changed after semantic"):
        verify_raw_inventory_unchanged(artifacts.result)


def test_raw_inventory_exclusion_cannot_hide_a_raw_subtree(tmp_path: Path) -> None:
    root = _write_validation_release(tmp_path / "pannuke", fold_count=1)
    discovery = validation_module.discover_pannuke_release(root)

    with pytest.raises(PanNukeSemanticsError, match="exclusions may not overlap"):
        validation_module.validate_discovered_release(
            discovery,
            expected_fold_ids=(1,),
            inventory_exclude_paths=(root / "Fold 1",),
        )
