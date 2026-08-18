from __future__ import annotations

import numpy as np

from histo_audit.data.manifest import validate_manifest
from histo_audit.data.splitting import make_outer_audit_split
from histo_audit.data.synthetic import CLASS_NAMES, generate_synthetic_dataset
from histo_audit.data.targets import (
    extract_target_colour_features,
    extract_target_crop,
    highlight_target,
    mask_bbox,
)


def test_five_class_mapping_and_determinism() -> None:
    first = generate_synthetic_dataset(n_groups=6, patch_size=40, seed=91)
    second = generate_synthetic_dataset(n_groups=6, patch_size=40, seed=91)
    assert CLASS_NAMES == (
        "neoplastic",
        "inflammatory",
        "connective_soft_tissue",
        "dead",
        "non_neoplastic_epithelial",
    )
    assert set(first.pre_corruption_labels) == set(range(5))
    np.testing.assert_array_equal(first.images, second.images)
    np.testing.assert_array_equal(first.target_masks, second.target_masks)
    np.testing.assert_allclose(first.audit_features, second.audit_features)
    assert first.records == second.records


def test_manifest_masks_boxes_and_target_identity(synthetic_dataset) -> None:
    dataset = synthetic_dataset
    validate_manifest(dataset.records, image_shape=dataset.images.shape[1:3])
    assert len(set(dataset.sample_ids.tolist())) == len(dataset.records)
    assert len({(row.patch_id, row.instance_id) for row in dataset.records}) == len(dataset.records)
    for row, mask in zip(dataset.records, dataset.target_masks, strict=True):
        assert mask.any()
        assert row.area == int(mask.sum()) > 0
        assert row.bbox == mask_bbox(mask)
        assert row.pre_corruption_label == row.observed_label
        assert not row.is_injected_corruption


def test_auditor_features_are_colour_only_and_disjoint_from_morphology(
    synthetic_dataset,
) -> None:
    expected = extract_target_colour_features(
        synthetic_dataset.images, synthetic_dataset.target_masks
    )
    np.testing.assert_array_equal(synthetic_dataset.audit_features, expected)
    assert synthetic_dataset.audit_features.shape[1] == 9
    assert synthetic_dataset.corruption_features.shape[1] == 5


def test_crop_is_deterministic_and_keeps_exact_target(synthetic_dataset) -> None:
    image = synthetic_dataset.images[0]
    mask = synthetic_dataset.target_masks[0]
    first = extract_target_crop(image, mask, output_size=32, padding=5)
    second = extract_target_crop(image, mask, output_size=32, padding=5)
    np.testing.assert_array_equal(first.image, second.image)
    np.testing.assert_array_equal(first.target_mask, second.target_mask)
    assert first.target_mask.any()
    assert first.source_box == second.source_box


def test_highlight_preserves_target_and_dims_context_without_label_input(
    synthetic_dataset,
) -> None:
    image = synthetic_dataset.images[3]
    mask = synthetic_dataset.target_masks[3]
    highlighted = highlight_target(image, mask, context_brightness=0.5)
    np.testing.assert_array_equal(highlighted[mask], image[mask])
    expected_context = np.rint(image[~mask].astype(float) * 0.5).astype(np.uint8)
    np.testing.assert_array_equal(highlighted[~mask], expected_context)
    # The interface deliberately receives no class argument.
    assert highlighted.shape == image.shape


def test_outer_split_separates_groups_and_protects_final_fold(synthetic_dataset) -> None:
    split = make_outer_audit_split(
        synthetic_dataset.official_folds,
        synthetic_dataset.group_ids,
        final_test_fold=2,
        reference_validation_fraction=0.1,
        seed=44,
    )
    assert set(split.audit_groups).isdisjoint(split.reference_validation_groups)
    assert set(split.audit_groups).isdisjoint(split.final_test_groups)
    assert set(split.reference_validation_groups).isdisjoint(split.final_test_groups)
    assert np.all(synthetic_dataset.official_folds[split.final_test_indices] == 2)
    assert np.all(
        synthetic_dataset.observed_labels[split.final_test_indices]
        == synthetic_dataset.pre_corruption_labels[split.final_test_indices]
    )
    split.validate(len(synthetic_dataset.records))
