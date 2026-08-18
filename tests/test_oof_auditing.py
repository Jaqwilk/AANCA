from __future__ import annotations

import numpy as np
import pytest

from histo_audit.auditing.neighbours import fold_safe_neighbour_disagreement
from histo_audit.auditing.scores import (
    cleanlab_scores,
    fixed_hybrid_score,
    score_annotations,
)
from histo_audit.corruption.controlled import apply_controlled_corruption
from histo_audit.cross_validation.oof import grouped_oof_logistic
from histo_audit.data.splitting import make_outer_audit_split


@pytest.fixture(scope="module")
def oof_case(synthetic_dataset):
    split = make_outer_audit_split(
        synthetic_dataset.official_folds,
        synthetic_dataset.group_ids,
        final_test_fold=2,
        seed=55,
    )
    indices = split.audit_indices
    corruption = apply_controlled_corruption(
        synthetic_dataset.pre_corruption_labels[indices],
        sample_ids=synthetic_dataset.sample_ids[indices],
        group_ids=synthetic_dataset.group_ids[indices],
        rate=10,
        seed=57,
    )
    oof = grouped_oof_logistic(
        synthetic_dataset.audit_features[indices],
        corruption.observed_labels,
        synthetic_dataset.group_ids[indices],
        final_reference_group_ids=split.final_test_groups,
        sample_ids=synthetic_dataset.sample_ids[indices],
        n_splits=4,
        class_order=(0, 1, 2, 3, 4),
        representation="target_colour_statistics",
    )
    return split, indices, corruption, oof


def test_oof_complete_once_probabilities_and_fixed_class_order(oof_case) -> None:
    _, indices, _, oof = oof_case
    assert oof.probabilities.shape == (len(indices), 5)
    np.testing.assert_array_equal(oof.coverage_count, np.ones(len(indices), dtype=int))
    assert np.isfinite(oof.probabilities).all()
    np.testing.assert_allclose(oof.probabilities.sum(axis=1), 1.0, atol=1e-8)
    assert oof.class_order == (0, 1, 2, 3, 4)
    assert len(set(oof.sample_ids)) == len(oof.sample_ids)


def test_oof_training_never_contains_held_out_source_group(oof_case) -> None:
    _, _, _, oof = oof_case
    for fold in oof.folds:
        assert set(fold.training_groups).isdisjoint(fold.held_out_groups)
        for sample_id in fold.held_out_sample_ids:
            sample_index = oof.sample_ids.index(sample_id)
            assert oof.group_ids[sample_index] in fold.held_out_groups
            assert oof.group_ids[sample_index] not in fold.training_groups


def test_oof_fails_closed_on_final_reference_contamination(synthetic_dataset) -> None:
    all_groups = tuple(sorted(set(synthetic_dataset.group_ids.tolist())))
    with pytest.raises(ValueError, match="present in audit pool"):
        grouped_oof_logistic(
            synthetic_dataset.audit_features,
            synthetic_dataset.observed_labels,
            synthetic_dataset.group_ids,
            final_reference_group_ids=all_groups,
            n_splits=3,
        )


def test_oof_rejects_empty_final_reference_evidence(synthetic_dataset) -> None:
    with pytest.raises(ValueError, match="evidence is mandatory"):
        grouped_oof_logistic(
            synthetic_dataset.audit_features,
            synthetic_dataset.observed_labels,
            synthetic_dataset.group_ids,
            final_reference_group_ids=(),
            n_splits=3,
        )


def test_handcrafted_risk_direction_and_no_nan() -> None:
    observed = np.array([0, 0, 1])
    probabilities = np.array(
        [
            [0.95, 0.03, 0.01, 0.005, 0.005],
            [0.05, 0.80, 0.05, 0.05, 0.05],
            [0.15, 0.60, 0.10, 0.10, 0.05],
        ]
    )
    for method in (
        "self_confidence",
        "negative_log_likelihood",
        "prediction_margin",
        "predictive_entropy",
    ):
        scores = score_annotations(observed, probabilities, method=method)
        assert np.isfinite(scores).all()
    confidence_risk = score_annotations(observed, probabilities, method="self_confidence")
    assert confidence_risk[1] > confidence_risk[0]
    nll = score_annotations(observed, probabilities, method="negative_log_likelihood")
    assert nll[1] > nll[0]
    margin = score_annotations(observed, probabilities, method="prediction_margin")
    assert margin[1] > margin[0]


def test_fold_safe_neighbours_exclude_self_and_same_group(synthetic_dataset, oof_case) -> None:
    _, indices, corruption, oof = oof_case
    result = fold_safe_neighbour_disagreement(
        synthetic_dataset.audit_features[indices],
        corruption.observed_labels,
        synthetic_dataset.group_ids[indices],
        oof.fold_id,
        oof.training_groups_by_fold,
        sample_ids=synthetic_dataset.sample_ids[indices],
        class_order=oof.class_order,
        k=5,
    )
    assert np.isfinite(result.risk_scores).all()
    for sample_id, group_id, neighbour_ids, neighbour_groups in zip(
        synthetic_dataset.sample_ids[indices],
        synthetic_dataset.group_ids[indices],
        result.neighbour_ids,
        result.neighbour_groups,
        strict=True,
    ):
        assert sample_id not in neighbour_ids
        assert group_id not in neighbour_groups


def test_cleanlab_optional_behavior_is_explicit(oof_case) -> None:
    _, _, corruption, oof = oof_case
    result = cleanlab_scores(corruption.observed_labels, oof.probabilities)
    if result.available:
        assert result.risk_scores is not None
        assert result.quality_scores is not None
        np.testing.assert_allclose(result.risk_scores, 1.0 - result.quality_scores)
        assert result.package_version
    else:
        assert result.risk_scores is None
        assert result.quality_scores is None
        assert result.error


def test_fixed_hybrid_is_reproducible_and_larger_remains_more_suspicious() -> None:
    components = {
        "self_confidence": np.array([0.1, 0.8, 0.2, 0.9]),
        "prediction_margin": np.array([-0.5, 0.4, -0.2, 0.7]),
        "neighbour_disagreement": np.array([0.0, 0.7, 0.1, 1.0]),
    }
    first = fixed_hybrid_score(components)
    second = fixed_hybrid_score(components)
    np.testing.assert_array_equal(first, second)
    assert first[3] > first[0]
    assert np.isfinite(first).all()
