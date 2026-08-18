from __future__ import annotations

import numpy as np
import pytest

from histo_audit.corruption.controlled import apply_controlled_corruption
from histo_audit.evaluation.restoration import (
    evaluate_downstream_restoration,
    restore_reviewed_labels,
)
from histo_audit.statistics.review import (
    average_precision,
    binary_auroc,
    budget_count,
    draw_group_bootstrap_indices,
    evaluate_review_budget,
    holm_adjust,
    paired_group_bootstrap,
    random_review_baseline,
    subgroup_average_precision,
    subgroup_is_reportable,
)


class _RecordingEstimator:
    def __init__(
        self,
        factory: _RecordingEstimatorFactory,
        class_order: tuple[int, ...],
        model_seed: int,
    ) -> None:
        self.factory = factory
        self.class_order = class_order
        self.model_seed = model_seed
        self.probabilities: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> _RecordingEstimator:
        self.factory.fit_inputs.append(
            (features.copy(), labels.copy(), bool(labels.flags.writeable))
        )
        counts = np.asarray([(labels == label).sum() for label in self.class_order], dtype=float)
        self.probabilities = (counts + 1.0) / float(counts.sum() + len(counts))
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        assert self.probabilities is not None
        return np.tile(self.probabilities, (len(features), 1))


class _RecordingEstimatorFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], int]] = []
        self.fit_inputs: list[tuple[np.ndarray, np.ndarray, bool]] = []

    def __call__(
        self,
        *,
        class_order: tuple[int, ...],
        model_seed: int,
    ) -> _RecordingEstimator:
        self.calls.append((class_order, model_seed))
        return _RecordingEstimator(self, class_order, model_seed)


class _MutatingEstimator(_RecordingEstimator):
    def fit(self, features: np.ndarray, labels: np.ndarray) -> _MutatingEstimator:
        labels[0] = labels[-1]
        return self


class _MutatingEstimatorFactory(_RecordingEstimatorFactory):
    def __call__(
        self,
        *,
        class_order: tuple[int, ...],
        model_seed: int,
    ) -> _MutatingEstimator:
        self.calls.append((class_order, model_seed))
        return _MutatingEstimator(self, class_order, model_seed)


def test_budget_calculations_and_lift() -> None:
    injected = np.array([True, False, True, False, False, False, False, False, False, False])
    scores = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    result = evaluate_review_budget(injected, scores, budget=20)
    assert budget_count(10, 0.01) == 1
    assert result.reviewed_count == 2
    assert result.injected_reviewed == 1
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.expected_random_recall == pytest.approx(0.2)
    assert result.lift_over_random == pytest.approx(2.5)


def test_zero_corruption_marks_ap_and_recall_not_applicable() -> None:
    injected = np.zeros(20, dtype=bool)
    scores = np.linspace(0.0, 1.0, 20)
    result = evaluate_review_budget(injected, scores, budget=5)
    assert result.average_precision is None
    assert average_precision(injected, scores) is None
    assert result.recall is None
    assert result.lift_over_random is None
    assert result.injected_reviewed == 0
    assert binary_auroc(injected, scores) is None


def test_tie_aware_auroc_subgroup_gate_and_holm_adjustment() -> None:
    injected = np.asarray([False, True, False, True, False, True])
    perfect = np.asarray([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    tied = np.ones(6)
    assert binary_auroc(injected, perfect) == pytest.approx(1.0)
    assert binary_auroc(injected, tied) == pytest.approx(0.5)

    subgroups = subgroup_average_precision(
        injected,
        perfect,
        ["a", "a", "a", "b", "b", "b"],
        min_samples=3,
        min_injected_corruptions=1,
    )
    assert [entry.status for entry in subgroups] == ["reported", "reported"]
    assert all(entry.average_precision is not None for entry in subgroups)
    gated = subgroup_average_precision(
        injected,
        perfect,
        ["a", "a", "a", "b", "b", "b"],
        min_samples=4,
        min_injected_corruptions=1,
    )
    assert all(entry.status == "insufficient_support" for entry in gated)
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_random_review_is_deterministic_and_uses_identical_budget() -> None:
    injected = np.array([True] * 10 + [False] * 90)
    first = random_review_baseline(injected, budget=5, repeats=100, seed=700)
    second = random_review_baseline(injected, budget=5, repeats=100, seed=700)
    assert first.reviewed_count == 5
    assert first.seeds == second.seeds
    np.testing.assert_array_equal(first.injected_reviewed, second.injected_reviewed)


def test_paired_group_bootstrap_reuses_identical_observation_draws() -> None:
    groups = np.repeat(["a", "b", "c", "d"], 5)
    injected = np.array([True, False, False, False, False] * 4)
    score_a = injected.astype(float) + np.linspace(0, 0.1, len(injected))
    score_b = np.linspace(0, 1, len(injected))
    draws = draw_group_bootstrap_indices(groups, n_iterations=50, seed=5)
    result = paired_group_bootstrap(
        injected,
        score_a,
        score_b,
        groups,
        n_iterations=50,
        seed=999,
        bootstrap_indices=draws,
    )
    assert result.requested_iterations == 50
    assert result.valid_iterations == 50
    np.testing.assert_allclose(result.differences, result.metric_a - result.metric_b)
    # Supplying the same draws makes the seed irrelevant and proves paired resampling.
    repeated = paired_group_bootstrap(
        injected,
        score_a,
        score_b,
        groups,
        seed=1,
        bootstrap_indices=draws,
    )
    np.testing.assert_array_equal(result.differences, repeated.differences)


def test_subgroup_threshold_gate() -> None:
    assert subgroup_is_reportable(100, 10)
    assert not subgroup_is_reportable(99, 10)
    assert not subgroup_is_reportable(100, 9)


def test_only_reviewed_injected_labels_are_restored(synthetic_dataset) -> None:
    corruption = apply_controlled_corruption(
        synthetic_dataset.pre_corruption_labels,
        sample_ids=synthetic_dataset.sample_ids,
        group_ids=synthetic_dataset.group_ids,
        rate=20,
        seed=10,
    )
    reviewed = np.array([corruption.selected_indices[0], 1], dtype=int)
    result = restore_reviewed_labels(
        corruption.pre_corruption_labels,
        corruption.observed_labels,
        corruption.is_injected_corruption,
        reviewed,
    )
    expected_restored = np.zeros(len(corruption.observed_labels), dtype=bool)
    expected_restored[reviewed] = corruption.is_injected_corruption[reviewed]
    np.testing.assert_array_equal(result.restored_mask, expected_restored)
    np.testing.assert_array_equal(
        result.restored_labels[~result.restored_mask],
        corruption.observed_labels[~result.restored_mask],
    )


def test_downstream_uses_identical_budgets_and_untouched_final_test(
    synthetic_dataset,
) -> None:
    development = np.flatnonzero(synthetic_dataset.official_folds != 2)
    final_test = np.flatnonzero(synthetic_dataset.official_folds == 2)
    corruption = apply_controlled_corruption(
        synthetic_dataset.pre_corruption_labels[development],
        sample_ids=synthetic_dataset.sample_ids[development],
        group_ids=synthetic_dataset.group_ids[development],
        rate=10,
        seed=77,
    )
    # Controlled perfect ranking is used only to test restoration semantics.
    risk = corruption.is_injected_corruption.astype(float)
    final_labels = synthetic_dataset.pre_corruption_labels[final_test].copy()
    evaluation = evaluate_downstream_restoration(
        synthetic_dataset.audit_features[development],
        corruption.pre_corruption_labels,
        corruption.observed_labels,
        corruption.is_injected_corruption,
        synthetic_dataset.audit_features[final_test],
        final_labels,
        risk,
        review_budget=5,
        development_group_ids=synthetic_dataset.group_ids[development],
        final_test_group_ids=synthetic_dataset.group_ids[final_test],
        final_test_is_injected_corruption=np.zeros(len(final_test), dtype=bool),
        random_repeats=3,
        max_iter=100,
    )
    assert evaluation.audit_guided_restoration.reviewed_count == evaluation.review_budget_count
    assert all(
        run.reviewed_count == evaluation.review_budget_count
        for run in evaluation.random_review_restoration
    )
    assert evaluation.audit_guided_restoration_evidence.reviewed_count == (
        evaluation.review_budget_count
    )
    assert len(evaluation.random_review_restoration_evidence) == 3
    for run in (
        evaluation.uncorrupted_reference_baseline,
        evaluation.corrupted_observed_baseline,
        evaluation.audit_guided_restoration,
        *evaluation.random_review_restoration,
    ):
        assert run.final_test_probabilities.shape == (len(final_test), 5)
        assert run.final_test_predicted_class.shape == (len(final_test),)
        np.testing.assert_allclose(run.final_test_probabilities.sum(axis=1), 1.0)
        np.testing.assert_array_equal(
            run.final_test_predicted_class,
            np.asarray([0, 1, 2, 3, 4])[np.argmax(run.final_test_probabilities, axis=1)],
        )
    for restoration, indices in zip(
        evaluation.random_review_restoration_evidence,
        evaluation.random_reviewed_indices,
        strict=True,
    ):
        expected_mask = np.zeros(len(development), dtype=bool)
        expected_mask[indices] = corruption.is_injected_corruption[indices]
        np.testing.assert_array_equal(restoration.restored_mask, expected_mask)
        np.testing.assert_array_equal(
            restoration.restored_labels[~expected_mask],
            corruption.observed_labels[~expected_mask],
        )
    np.testing.assert_array_equal(final_labels, synthetic_dataset.pre_corruption_labels[final_test])
    names = {
        evaluation.uncorrupted_reference_baseline.experiment_name,
        evaluation.corrupted_observed_baseline.experiment_name,
        evaluation.audit_guided_restoration.experiment_name,
        evaluation.random_review_restoration[0].experiment_name,
    }
    assert names == {
        "uncorrupted_reference_baseline",
        "corrupted_observed_baseline",
        "random_review_restoration",
        "audit_guided_restoration",
    }


def test_downstream_rejects_corrupted_final_reference(synthetic_dataset) -> None:
    development = np.flatnonzero(synthetic_dataset.official_folds != 2)
    final_test = np.flatnonzero(synthetic_dataset.official_folds == 2)
    labels = synthetic_dataset.pre_corruption_labels[development]
    with pytest.raises(ValueError, match="must remain uncorrupted"):
        evaluate_downstream_restoration(
            synthetic_dataset.audit_features[development],
            labels,
            labels,
            np.zeros(len(development), dtype=bool),
            synthetic_dataset.audit_features[final_test],
            synthetic_dataset.pre_corruption_labels[final_test],
            np.zeros(len(development)),
            development_group_ids=synthetic_dataset.group_ids[development],
            final_test_group_ids=synthetic_dataset.group_ids[final_test],
            final_test_is_injected_corruption=np.ones(len(final_test), dtype=bool),
            random_repeats=1,
        )


def test_downstream_factory_and_clean_reference_validation_are_identical_across_conditions(
    synthetic_dataset,
) -> None:
    audit_pool = np.flatnonzero(synthetic_dataset.official_folds == 0)
    reference_validation = np.flatnonzero(synthetic_dataset.official_folds == 1)
    final_test = np.flatnonzero(synthetic_dataset.official_folds == 2)
    corruption = apply_controlled_corruption(
        synthetic_dataset.pre_corruption_labels[audit_pool],
        sample_ids=synthetic_dataset.sample_ids[audit_pool],
        group_ids=synthetic_dataset.group_ids[audit_pool],
        rate=10,
        seed=79,
    )
    validation_labels = synthetic_dataset.pre_corruption_labels[reference_validation].copy()
    validation_before = validation_labels.copy()
    reference_before = corruption.pre_corruption_labels.copy()
    observed_before = corruption.observed_labels.copy()
    final_labels = synthetic_dataset.pre_corruption_labels[final_test].copy()
    final_before = final_labels.copy()
    factory = _RecordingEstimatorFactory()

    evaluation = evaluate_downstream_restoration(
        synthetic_dataset.audit_features[audit_pool],
        corruption.pre_corruption_labels,
        corruption.observed_labels,
        corruption.is_injected_corruption,
        synthetic_dataset.audit_features[final_test],
        final_labels,
        corruption.is_injected_corruption.astype(float),
        development_group_ids=synthetic_dataset.group_ids[audit_pool],
        final_test_group_ids=synthetic_dataset.group_ids[final_test],
        final_test_is_injected_corruption=np.zeros(len(final_test), dtype=bool),
        reference_validation_features=synthetic_dataset.audit_features[reference_validation],
        reference_validation_labels=validation_labels,
        reference_validation_group_ids=synthetic_dataset.group_ids[reference_validation],
        reference_validation_is_injected_corruption=np.zeros(len(reference_validation), dtype=bool),
        estimator_factory=factory,
        random_repeats=2,
        model_seed=991,
    )

    assert factory.calls == [((0, 1, 2, 3, 4), 991)] * 5
    assert len(factory.fit_inputs) == 5
    audit_count = len(audit_pool)
    expected_condition_labels = (
        corruption.pre_corruption_labels,
        corruption.observed_labels,
        evaluation.audit_guided_restoration_evidence.restored_labels,
        *(item.restored_labels for item in evaluation.random_review_restoration_evidence),
    )
    for (features, labels, labels_writeable), condition_labels in zip(
        factory.fit_inputs, expected_condition_labels, strict=True
    ):
        assert features.shape[0] == audit_count + len(reference_validation)
        np.testing.assert_array_equal(
            features[audit_count:], synthetic_dataset.audit_features[reference_validation]
        )
        np.testing.assert_array_equal(labels[:audit_count], condition_labels)
        np.testing.assert_array_equal(labels[audit_count:], validation_labels)
        assert not labels_writeable

    assert evaluation.reference_validation_sample_count == len(reference_validation)
    assert set(evaluation.reference_validation_groups) == set(
        synthetic_dataset.group_ids[reference_validation]
    )
    assert evaluation.review_budget_count == budget_count(len(audit_pool), 0.05)
    payload = evaluation.as_dict()
    assert {
        "uncorrupted_reference_baseline",
        "corrupted_observed_baseline",
        "random_review_restoration",
        "audit_guided_restoration",
        "review_budget_fraction",
        "review_budget_count",
        "partition_evidence",
    }.issubset(payload)
    assert payload["partition_evidence"]["reference_validation_in_training"] is True
    assert payload["model_evidence"]["model_seed"] == 991
    np.testing.assert_array_equal(corruption.pre_corruption_labels, reference_before)
    np.testing.assert_array_equal(corruption.observed_labels, observed_before)
    np.testing.assert_array_equal(validation_labels, validation_before)
    np.testing.assert_array_equal(final_labels, final_before)


@pytest.mark.parametrize(
    ("overlap_partition", "message"),
    [
        ("development", "development/reference-validation source-group leakage"),
        ("final", "reference-validation/final-test source-group leakage"),
    ],
)
def test_downstream_rejects_reference_validation_group_leakage(
    synthetic_dataset,
    overlap_partition: str,
    message: str,
) -> None:
    audit_pool = np.flatnonzero(synthetic_dataset.official_folds == 0)
    reference_validation = np.flatnonzero(synthetic_dataset.official_folds == 1)
    final_test = np.flatnonzero(synthetic_dataset.official_folds == 2)
    labels = synthetic_dataset.pre_corruption_labels[audit_pool]
    validation_groups = synthetic_dataset.group_ids[reference_validation].astype(object, copy=True)
    source_indices = audit_pool if overlap_partition == "development" else final_test
    validation_groups[0] = synthetic_dataset.group_ids[source_indices[0]]

    with pytest.raises(ValueError, match=message):
        evaluate_downstream_restoration(
            synthetic_dataset.audit_features[audit_pool],
            labels,
            labels,
            np.zeros(len(audit_pool), dtype=bool),
            synthetic_dataset.audit_features[final_test],
            synthetic_dataset.pre_corruption_labels[final_test],
            np.zeros(len(audit_pool)),
            development_group_ids=synthetic_dataset.group_ids[audit_pool],
            final_test_group_ids=synthetic_dataset.group_ids[final_test],
            final_test_is_injected_corruption=np.zeros(len(final_test), dtype=bool),
            reference_validation_features=synthetic_dataset.audit_features[reference_validation],
            reference_validation_labels=synthetic_dataset.pre_corruption_labels[
                reference_validation
            ],
            reference_validation_group_ids=validation_groups,
            reference_validation_is_injected_corruption=np.zeros(
                len(reference_validation), dtype=bool
            ),
            random_repeats=1,
        )


def test_downstream_reference_validation_is_fail_closed_and_labels_are_read_only(
    synthetic_dataset,
) -> None:
    audit_pool = np.flatnonzero(synthetic_dataset.official_folds == 0)
    reference_validation = np.flatnonzero(synthetic_dataset.official_folds == 1)
    final_test = np.flatnonzero(synthetic_dataset.official_folds == 2)
    labels = synthetic_dataset.pre_corruption_labels[audit_pool]
    common = {
        "development_group_ids": synthetic_dataset.group_ids[audit_pool],
        "final_test_group_ids": synthetic_dataset.group_ids[final_test],
        "final_test_is_injected_corruption": np.zeros(len(final_test), dtype=bool),
        "reference_validation_features": synthetic_dataset.audit_features[reference_validation],
        "reference_validation_labels": synthetic_dataset.pre_corruption_labels[
            reference_validation
        ],
        "reference_validation_group_ids": synthetic_dataset.group_ids[reference_validation],
        "random_repeats": 1,
    }
    positional = (
        synthetic_dataset.audit_features[audit_pool],
        labels,
        labels,
        np.zeros(len(audit_pool), dtype=bool),
        synthetic_dataset.audit_features[final_test],
        synthetic_dataset.pre_corruption_labels[final_test],
        np.zeros(len(audit_pool)),
    )

    with pytest.raises(ValueError, match="must be supplied together"):
        evaluate_downstream_restoration(*positional, **common)

    injected_validation = np.zeros(len(reference_validation), dtype=bool)
    injected_validation[0] = True
    with pytest.raises(ValueError, match="must remain clean and uncorrupted"):
        evaluate_downstream_restoration(
            *positional,
            **common,
            reference_validation_is_injected_corruption=injected_validation,
        )

    immutable_labels = synthetic_dataset.pre_corruption_labels[reference_validation].copy()
    immutable_before = immutable_labels.copy()
    with pytest.raises(ValueError, match="read-only"):
        evaluate_downstream_restoration(
            *positional,
            **{
                **common,
                "reference_validation_labels": immutable_labels,
                "reference_validation_is_injected_corruption": np.zeros(
                    len(reference_validation), dtype=bool
                ),
                "estimator_factory": _MutatingEstimatorFactory(),
            },
        )
    np.testing.assert_array_equal(immutable_labels, immutable_before)
