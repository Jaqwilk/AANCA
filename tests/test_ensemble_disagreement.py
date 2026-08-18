from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from histo_audit.auditing import (
    PrimaryEnsembleRisk,
    ensemble_disagreement,
    predeclared_ensemble_risk,
)
from histo_audit.auditing.scores import ensemble_disagreement as legacy_import


@pytest.fixture
def ensemble_probabilities() -> NDArray[np.float64]:
    return np.asarray(
        [
            [[0.8, 0.1, 0.1], [0.6, 0.2, 0.2], [0.5, 0.5, 0.0]],
            [[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.4, 0.4, 0.2]],
            [[0.7, 0.2, 0.1], [0.2, 0.2, 0.6], [0.1, 0.8, 0.1]],
        ],
        dtype=np.float64,
    )


def test_all_disagreement_outputs_have_known_values(
    ensemble_probabilities: NDArray[np.float64],
) -> None:
    result = ensemble_disagreement(
        ensemble_probabilities,
        observed_labels=np.asarray([10, 30, 90]),
        class_order=(10, 30, 90),
    )

    np.testing.assert_allclose(
        result.averaged_probabilities,
        ensemble_probabilities.mean(axis=0),
    )
    np.testing.assert_allclose(
        result.entropy_of_mean,
        -np.sum(
            np.where(
                result.averaged_probabilities > 0,
                result.averaged_probabilities * np.log(result.averaged_probabilities),
                0.0,
            ),
            axis=1,
        ),
    )
    np.testing.assert_allclose(result.variation_ratio, [0.0, 2.0 / 3.0, 1.0 / 3.0])
    np.testing.assert_allclose(
        result.predicted_class_disagreement,
        [0.0, 1.0, 2.0 / 3.0],
    )
    np.testing.assert_allclose(
        result.observed_label_probability_variance,
        [np.var([0.8, 0.6, 0.7]), np.var([0.2, 0.6, 0.2]), np.var([0.0, 0.2, 0.1])],
    )
    np.testing.assert_array_equal(
        result.observed_probability_variance,
        result.observed_label_probability_variance,
    )
    assert result.class_order == (10, 30, 90)
    assert result.model_count == 3
    assert not result.averaged_probabilities.flags.writeable


def test_hard_prediction_ties_follow_explicit_column_order() -> None:
    probabilities = np.asarray(
        [
            [[0.5, 0.5, 0.0]],
            [[0.4, 0.4, 0.2]],
            [[0.1, 0.8, 0.1]],
        ]
    )
    result = ensemble_disagreement(
        probabilities,
        observed_labels=[-7],
        class_order=(-7, 42, 1000),
    )

    # The first two tied models vote for class -7 (column zero); the third votes 42.
    np.testing.assert_allclose(result.variation_ratio, [1.0 / 3.0])
    np.testing.assert_allclose(result.predicted_class_disagreement, [2.0 / 3.0])


def test_identical_models_have_zero_disagreement_but_retain_predictive_entropy() -> None:
    probability = np.asarray([[0.7, 0.2, 0.1], [0.0, 1.0, 0.0]])
    result = ensemble_disagreement(
        np.stack([probability, probability, probability]),
        observed_labels=[11, 29],
        class_order=(11, 17, 29),
    )

    np.testing.assert_array_equal(result.variation_ratio, np.zeros(2))
    np.testing.assert_array_equal(result.predicted_class_disagreement, np.zeros(2))
    np.testing.assert_array_equal(result.observed_label_probability_variance, np.zeros(2))
    np.testing.assert_array_equal(result.mean_pairwise_js_divergence, np.zeros(2))
    assert result.entropy_of_mean[0] > 0.0
    assert result.entropy_of_mean[1] == 0.0


def test_pairwise_js_is_symmetric_and_reaches_ln_two_for_disjoint_predictions() -> None:
    left = np.asarray([[1.0, 0.0], [0.75, 0.25]])
    right = np.asarray([[0.0, 1.0], [0.25, 0.75]])
    forward = ensemble_disagreement(
        [left, right],
        observed_labels=[3, 8],
        class_order=(3, 8),
    )
    reverse = ensemble_disagreement(
        [right, left],
        observed_labels=[3, 8],
        class_order=(3, 8),
    )

    np.testing.assert_array_equal(
        forward.mean_pairwise_js_divergence,
        reverse.mean_pairwise_js_divergence,
    )
    np.testing.assert_allclose(forward.mean_pairwise_js_divergence[0], np.log(2.0))
    assert np.all(forward.mean_pairwise_js_divergence >= 0.0)
    assert np.all(forward.mean_pairwise_js_divergence <= np.log(2.0))


def test_js_is_mean_over_every_unordered_model_pair() -> None:
    class_zero = np.asarray([[1.0, 0.0]])
    class_one = np.asarray([[0.0, 1.0]])
    result = ensemble_disagreement(
        [class_zero, class_zero, class_one],
        observed_labels=[0],
        class_order=(0, 1),
    )

    # Pair divergences are 0, ln(2), ln(2).
    np.testing.assert_allclose(result.mean_pairwise_js_divergence, [2.0 * np.log(2.0) / 3.0])


def test_model_order_and_sequence_or_tensor_representation_do_not_change_results(
    ensemble_probabilities: NDArray[np.float64],
) -> None:
    expected = ensemble_disagreement(
        ensemble_probabilities,
        observed_labels=[10, 30, 90],
        class_order=(10, 30, 90),
    )
    actual = ensemble_disagreement(
        [ensemble_probabilities[2], ensemble_probabilities[0], ensemble_probabilities[1]],
        observed_labels=[10, 30, 90],
        class_order=(10, 30, 90),
    )

    for name in (
        "averaged_probabilities",
        "entropy_of_mean",
        "variation_ratio",
        "observed_label_probability_variance",
        "predicted_class_disagreement",
        "mean_pairwise_js_divergence",
    ):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize(
    ("primary_risk", "field"),
    [
        ("predictive_entropy_of_mean", "entropy_of_mean"),
        ("mean_pairwise_js_divergence", "mean_pairwise_js_divergence"),
        ("variation_ratio", "variation_ratio"),
        (
            "observed_label_probability_variance",
            "observed_label_probability_variance",
        ),
        ("predicted_class_disagreement", "predicted_class_disagreement"),
    ],
)
def test_predeclared_primary_risk_is_a_fixed_outcome_free_mapping(
    ensemble_probabilities: NDArray[np.float64],
    primary_risk: PrimaryEnsembleRisk,
    field: str,
) -> None:
    result = ensemble_disagreement(
        ensemble_probabilities,
        observed_labels=[10, 30, 90],
        class_order=(10, 30, 90),
    )
    selected = predeclared_ensemble_risk(result, primary_risk=primary_risk)

    np.testing.assert_array_equal(selected, getattr(result, field))
    assert selected is not getattr(result, field)
    assert not selected.flags.writeable


def test_predeclared_primary_risk_rejects_outcome_selected_method(
    ensemble_probabilities: NDArray[np.float64],
) -> None:
    result = ensemble_disagreement(
        ensemble_probabilities,
        observed_labels=[10, 30, 90],
        class_order=(10, 30, 90),
    )
    unsupported = cast(PrimaryEnsembleRisk, "highest_validation_ap")

    with pytest.raises(ValueError, match="unsupported predeclared ensemble risk"):
        predeclared_ensemble_risk(result, primary_risk=unsupported)


def test_legacy_scores_import_points_to_strict_implementation(
    ensemble_probabilities: NDArray[np.float64],
) -> None:
    assert legacy_import is ensemble_disagreement
    with pytest.raises(TypeError):
        legacy_import(ensemble_probabilities)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (np.asarray([[0.5, 0.5], [0.4, 0.6]]), "shape"),
        (np.asarray([[[0.5, 0.5]]]), "at least two models"),
        (np.empty((2, 0, 2)), "at least one sample"),
        (np.ones((2, 1, 1)), "two classes"),
        (np.asarray([[[np.nan, np.nan]], [[0.5, 0.5]]]), "non-finite"),
        (np.asarray([[[-0.1, 1.1]], [[0.5, 0.5]]]), r"within \[0, 1\]"),
        (np.asarray([[[0.4, 0.4]], [[0.5, 0.5]]]), "sum to one"),
    ],
)
def test_probability_validation_fails_closed(
    probabilities: NDArray[np.float64],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ensemble_disagreement(
            probabilities,
            observed_labels=[0] * (probabilities.shape[1] if probabilities.ndim == 3 else 1),
            class_order=tuple(range(probabilities.shape[-1])),
        )


def test_mismatched_model_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="identically shaped"):
        ensemble_disagreement(
            [np.asarray([[0.5, 0.5]]), np.asarray([[0.5, 0.5], [0.4, 0.6]])],
            observed_labels=[0],
            class_order=(0, 1),
        )


@pytest.mark.parametrize(
    ("observed_labels", "class_order", "message"),
    [
        ([10, 30], (10, 30, 90), "shape"),
        ([10, 30, 77], (10, 30, 90), "absent"),
        ([10.0, 30.0, 90.0], (10, 30, 90), "integer"),
        ([10, 30, 90], (10, 30), "one entry per"),
        ([10, 30, 90], (10, 10, 90), "unique"),
        ([10, 30, 90], (10.0, 30.0, 90.0), "integer"),
    ],
)
def test_class_order_and_observed_label_validation(
    ensemble_probabilities: NDArray[np.float64],
    observed_labels: list[int] | list[float],
    class_order: tuple[int, ...] | tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ensemble_disagreement(
            ensemble_probabilities,
            observed_labels=observed_labels,  # type: ignore[arg-type]
            class_order=class_order,  # type: ignore[arg-type]
        )
