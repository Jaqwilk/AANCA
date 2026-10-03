"""Regression cases for the invalid inputs reproduced by the V1 review."""

from __future__ import annotations

import builtins
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from histo_audit.auditing.neighbours import fold_safe_neighbour_disagreement
from histo_audit.auditing.scores import score_annotations
from histo_audit.auditing.two_queue import (
    CROSS_FITTED_UTILITY_EVIDENCE,
    GROUP_SAFE_OOF_EVIDENCE,
    QueueConstraints,
    build_two_review_queues,
)
from histo_audit.cross_validation.oof import (
    MultinomialLogisticRegression,
    grouped_oof_logistic,
    grouped_oof_predict,
)
from histo_audit.evaluation.restoration import classification_metrics
from histo_audit.evaluation.retraining_guard import (
    INDEPENDENT_GROUP_VALIDATION,
    evaluate_multicriteria_retraining_guard,
    evaluate_retraining_guard,
)
from histo_audit.evaluation.review_training import SoftTargetMultinomialLogisticRegression
from histo_audit.validation import integer_vector
from histo_audit.workflows.original_audit import _oof_provenance, _validate_manifest


@pytest.mark.parametrize(
    "probabilities",
    [
        [[1.1, -0.1], [-0.1, 1.1]],
        [[np.nan, 0.5], [0.5, 0.5]],
        [[np.inf, 0.0], [0.5, 0.5]],
        [[0.5, 0.500001], [0.5, 0.5]],
        [[0.5, 0.4], [0.5, 0.5]],
    ],
)
def test_metrics_reject_invalid_distributions(probabilities: list[list[float]]) -> None:
    with pytest.raises(ValueError, match="probabilit"):
        classification_metrics([0, 1], np.asarray(probabilities), class_order=(0, 1))


def test_metrics_reject_duplicate_and_fractional_class_identifiers() -> None:
    for classes in ((0, 0), (0, 1.9)):
        with pytest.raises(ValueError, match="class_order"):
            classification_metrics([0, 0], np.eye(2), class_order=classes)


def test_valid_perfect_probabilities_retain_their_metrics() -> None:
    result = classification_metrics([0.0, 1.0], np.eye(2), class_order=(0, 1))
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.expected_calibration_error == 0.0


@pytest.mark.parametrize("epsilon", [-1.0, 0.0, 1.0, 2.0, np.inf, np.nan])
@pytest.mark.parametrize("method", ["entropy", "negative_log_likelihood"])
def test_invalid_epsilon_cannot_hide_annotation_risk(epsilon: float, method: str) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        score_annotations([0, 1], np.eye(2), method=method, epsilon=epsilon)


@pytest.mark.parametrize(
    "probabilities", [[[1.000001, 0.0], [0.0, 1.000001]], [[0.5, 0.500001]] * 2]
)
def test_soft_target_training_rejects_non_distributions(
    probabilities: list[list[float]],
) -> None:
    model = SoftTargetMultinomialLogisticRegression(class_order=(0, 1))
    with pytest.raises(ValueError, match="probabilit"):
        model.fit_soft_targets(np.asarray([[-1.0], [1.0]]), np.asarray(probabilities))
    assert model.coef_ is None


@pytest.mark.parametrize("promote_to_double", [False, True])
def test_real_float32_softmax_rounding_preserves_valid_probabilities(
    promote_to_double: bool,
) -> None:
    import torch

    logits = torch.tensor(
        [[0.76441616, 1.8399488, 0.8255816, -0.5653991, 0.2101873]], dtype=torch.float32
    )
    probabilities = torch.softmax(logits, dim=1).numpy()
    assert abs(probabilities.astype(np.float64).sum() - 1.0) > 1.0e-7
    if promote_to_double:
        probabilities = probabilities.astype(np.float64)
    classes = (0, 1, 2, 3, 4)
    metrics = classification_metrics([1], probabilities, class_order=classes)
    assert metrics.accuracy == 1.0
    risk = score_annotations([1], probabilities, method="self_confidence", class_order=classes)
    assert risk[0] == 1.0 - float(probabilities[0, 1])
    model = SoftTargetMultinomialLogisticRegression(class_order=classes)
    model.fit_soft_targets(np.asarray([[0.0], [1.0]]), np.repeat(probabilities, 2, axis=0))
    assert model.coef_ is not None


@pytest.mark.parametrize("labels", [[0.2, 1.9], [0, np.nan], [0, np.inf]])
def test_fractional_or_nonfinite_labels_are_not_truncated(labels: list[float]) -> None:
    with pytest.raises(ValueError, match="integer"):
        classification_metrics(labels, np.eye(2), class_order=(0, 1))
    with pytest.raises(ValueError, match="integer"):
        score_annotations(labels, np.eye(2), class_order=(0, 1), method="self_confidence")


@pytest.mark.parametrize("values", [np.asarray([2**63], dtype=np.uint64), [float(2**63)]])
def test_label_integer_overflow_is_rejected(values: object) -> None:
    with pytest.raises(ValueError, match="int64"):
        integer_vector(values, name="labels")


def _manifest(labels: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "group_id": ["g0", "g1"],
            "tissue_type": ["test", "test"],
            "pre_corruption_label": labels,
            "observed_label": labels,
            "is_injected_corruption": [False, False],
        }
    )


def test_manifest_preserves_integer_labels_and_rejects_fractional_labels() -> None:
    source = _manifest([0.2, 1.9])
    before = source.copy(deep=True)
    with pytest.raises(ValueError, match="integer"):
        _validate_manifest(source)
    pd.testing.assert_frame_equal(source, before)
    result = _validate_manifest(_manifest([0.0, 1.0]))
    assert result["observed_label"].tolist() == [0, 1]


def test_negative_global_adoption_threshold_is_rejected_by_both_guards() -> None:
    truth = np.tile([0, 1, 0, 1], 3)
    baseline = np.eye(2)[truth]
    worse = truth.copy()
    worse[::4] = 1
    common = {
        "class_order": (0, 1),
        "evidence_role": INDEPENDENT_GROUP_VALIDATION,
        "n_iterations": 20,
    }
    groups = np.repeat(["a", "b", "c"], 4)
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_retraining_guard(
            truth, baseline, np.eye(2)[worse], groups, minimum_effect=-1.0, **common
        )
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_multicriteria_retraining_guard(
            truth, baseline, np.eye(2)[worse], groups, minimum_macro_f1_effect=-1.0, **common
        )


def test_negative_gain_threshold_cannot_select_adverse_utility() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_two_review_queues(
            [0.9, 0.8],
            ["a", "b"],
            [0, 1],
            ["s0", "s1"],
            quality_constraints=QueueConstraints(requested_count=1),
            model_constraints=QueueConstraints(requested_count=1),
            annotation_evidence_role=GROUP_SAFE_OOF_EVIDENCE,
            expected_downstream_gain=[-0.1, -0.1],
            downstream_gain_lower_bound=[-0.2, -0.2],
            utility_evidence_role=CROSS_FITTED_UTILITY_EVIDENCE,
            minimum_downstream_gain=-0.3,
        )


def _oof_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.tile([0, 1, 0, 1], 3)
    features = np.column_stack((np.arange(12), labels))
    return features, labels, np.repeat(["a", "b", "c"], 4)


def test_nonconverged_oof_fails_before_publishing_probabilities() -> None:
    features, labels, groups = _oof_inputs()
    with pytest.raises(RuntimeError, match=r"OOF fold .*did not converge"):
        grouped_oof_logistic(
            features,
            labels,
            groups,
            final_reference_group_ids=("final",),
            class_order=(0, 1),
            n_splits=3,
            max_iter=1,
        )


def test_oof_provenance_persists_real_optimizer_diagnostics() -> None:
    features, labels, groups = _oof_inputs()
    result = grouped_oof_logistic(
        features,
        labels,
        groups,
        final_reference_group_ids=("final",),
        class_order=(0, 1),
        n_splits=3,
    )
    for fold in _oof_provenance(result)["folds"]:
        assert fold["fit_status"] == "converged"
        assert fold["fit_optimizer"] == "scipy_lbfgsb"
        assert fold["fit_iterations"] > 0
        assert fold["fit_message"]


def test_estimator_without_convergence_flag_is_recorded_as_unknown() -> None:
    class FixedEstimator:
        classes_ = np.asarray([0, 1])

        def fit(self, features: np.ndarray, labels: np.ndarray) -> FixedEstimator:
            return self

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            return np.tile([0.5, 0.5], (len(features), 1))

    result = grouped_oof_predict(
        *_oof_inputs(),
        estimator_factory=lambda context: FixedEstimator(),
        model_name="fixed",
        final_reference_group_ids=("final",),
        class_order=(0, 1),
        n_splits=3,
    )
    assert all(asdict(fold)["fit_status"] == "unknown" for fold in result.folds)


@pytest.mark.parametrize("soft_targets", [False, True])
def test_adam_fallback_does_not_invent_convergence(
    monkeypatch: pytest.MonkeyPatch, soft_targets: bool
) -> None:
    original_import = builtins.__import__

    def without_scipy(name: str, *args: object, **kwargs: object) -> object:
        if name == "scipy.optimize":
            raise ImportError("test forces the deterministic fallback")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scipy)
    model_type = (
        SoftTargetMultinomialLogisticRegression if soft_targets else MultinomialLogisticRegression
    )
    model = model_type(class_order=(0, 1), max_iter=1)
    features, labels, _ = _oof_inputs()
    if soft_targets:
        model.fit_soft_targets(features, np.eye(2)[labels])
    else:
        model.fit(features, labels)
    assert model.optimizer_ == "adam_fallback"
    assert model.converged_ is False


def test_neighbour_provenance_rejects_the_whole_holdout_before_index_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_fit(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid provenance reached nearest-neighbour fitting")

    monkeypatch.setattr("histo_audit.auditing.neighbours.NearestNeighbors.fit", unexpected_fit)
    with pytest.raises(ValueError, match="overlaps held-out"):
        fold_safe_neighbour_disagreement(
            np.asarray([[0, 0], [0.001, 0], [10, 1], [20, 1]]),
            [0, 1, 0, 1],
            ["a", "b", "c", "d"],
            [0, 0, 1, 1],
            {0: ("a", "b", "c", "d"), 1: ("a", "b", "c", "d")},
            sample_ids=("s0", "s1", "s2", "s3"),
            class_order=(0, 1),
            k=1,
        )


def test_neighbour_provenance_rejects_one_group_split_between_folds() -> None:
    with pytest.raises(ValueError, match="exactly one held-out fold"):
        fold_safe_neighbour_disagreement(
            np.zeros((4, 2)),
            [0, 1, 0, 1],
            ["a", "a", "b", "b"],
            [0, 1, 0, 1],
            {0: ("b",), 1: ("a",)},
            class_order=(0, 1),
        )
