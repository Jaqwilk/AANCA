from __future__ import annotations

import numpy as np
import pytest

from histo_audit.evaluation.review_interventions import derive_review_interventions
from histo_audit.evaluation.review_training import fit_review_intervention_model


def test_review_actions_preserve_votes_ambiguity_and_source_labels() -> None:
    observed = np.asarray([0, 0, 1, 1, 0, 1, 0], dtype=np.int64)
    original = observed.copy()
    votes = np.asarray(
        [
            [-1, 0, 0, 0, 1, -1, 1],
            [-1, 0, 1, 1, 1, -1, 0],
            [-1, 0, -1, 0, 1, -1, 1],
        ],
        dtype=np.int64,
    )
    result = derive_review_interventions(
        observed,
        votes,
        [False, True, True, True, True, True, True],
        class_order=(0, 1),
        ambiguous_mask=[False, False, False, True, False, False, False],
        insufficient_context_mask=[False, False, False, False, False, True, False],
        technical_exclusion_mask=[False, False, False, False, False, False, True],
        allow_hard_change=True,
    )

    assert np.array_equal(observed, original)
    assert np.array_equal(result.source_observed_labels, original)
    assert result.source_annotations_modified is False
    assert result.source_observed_labels.flags.writeable is False
    assert result.soft_targets.flags.writeable is False
    assert result.actions == (
        "keep",
        "keep",
        "soft_label",
        "soft_label",
        "hard_change",
        "exclude",
        "exclude",
    )
    assert result.derived_hard_labels.tolist() == [0, 0, 1, 1, 1, 1, 0]
    assert result.soft_targets[2].tolist() == pytest.approx([0.5, 0.5])
    assert result.soft_targets[3].tolist() == pytest.approx([2 / 3, 1 / 3])
    assert result.training_weights.tolist() == pytest.approx([1, 1, 0.5, 0.5, 1, 0, 0])
    assert np.allclose(result.soft_targets.sum(axis=1), 1.0)


def test_hard_change_is_opt_in_and_supports_noncontiguous_class_values() -> None:
    votes = np.asarray([[7], [7], [-3]], dtype=np.int64)
    soft_only = derive_review_interventions(
        [-3],
        votes,
        [True],
        class_order=(-3, 7),
        allow_hard_change=False,
    )
    hard = derive_review_interventions(
        [-3],
        votes,
        [True],
        class_order=(-3, 7),
        allow_hard_change=True,
    )

    assert soft_only.actions == ("soft_label",)
    assert soft_only.derived_hard_labels.tolist() == [-3]
    assert hard.actions == ("hard_change",)
    assert hard.derived_hard_labels.tolist() == [7]
    assert hard.majority_fractions.tolist() == pytest.approx([2 / 3])


def test_unreviewed_samples_cannot_contain_votes() -> None:
    with pytest.raises(ValueError, match="unreviewed samples"):
        derive_review_interventions(
            [0, 1],
            np.asarray([[0, -1]], dtype=np.int64),
            [False, False],
            class_order=(0, 1),
        )


def test_soft_target_training_consumes_derived_targets_and_weights() -> None:
    observed = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    votes = np.asarray(
        [
            [-1, -1, 0, 1, -1, -1],
            [-1, -1, 1, 1, -1, -1],
            [-1, -1, 0, 1, -1, -1],
        ],
        dtype=np.int64,
    )
    intervention = derive_review_interventions(
        observed,
        votes,
        [False, False, True, True, False, False],
        class_order=(0, 1),
    )
    features = np.asarray([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]])

    model = fit_review_intervention_model(features, intervention, class_order=(0, 1))
    probabilities = model.predict_proba(features)

    assert model.converged_ is True
    assert probabilities.shape == (6, 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
