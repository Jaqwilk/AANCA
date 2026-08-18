from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from histo_audit.cross_validation import (
    OOFFoldEstimatorContext,
    grouped_oof_frozen_embedding_mlp,
    grouped_oof_predict,
    make_group_stratified_fold_plan,
)
from histo_audit.models import FrozenEmbeddingMLPConfig


class _FixedProbabilityEstimator:
    def __init__(
        self,
        classes: tuple[int, ...],
        row: tuple[float, ...],
    ) -> None:
        self.classes_ = np.asarray(classes, dtype=np.int64)
        self._row = np.asarray(row, dtype=np.float64)
        self.fit_labels: NDArray[np.int64] | None = None

    def fit(
        self,
        features: NDArray[np.generic],
        labels: NDArray[np.generic],
    ) -> _FixedProbabilityEstimator:
        assert features.shape[0] == labels.shape[0]
        self.fit_labels = np.asarray(labels, dtype=np.int64)
        return self

    def predict_proba(self, features: NDArray[np.generic]) -> NDArray[np.float64]:
        assert self.fit_labels is not None
        return np.tile(self._row, (len(features), 1))


def _balanced_group_case() -> tuple[
    NDArray[np.float64], NDArray[np.int64], tuple[str, ...], tuple[str, ...]
]:
    group_ids = tuple(f"group_{group:02d}" for group in range(9) for _ in range(3))
    labels = np.tile(np.arange(3, dtype=np.int64), 9)
    sample_ids = tuple(f"sample_{index:03d}" for index in range(len(labels)))
    features = np.column_stack(
        (
            np.arange(len(labels), dtype=np.float64),
            labels.astype(np.float64),
        )
    )
    return features, labels, group_ids, sample_ids


def test_generic_oof_has_once_only_coverage_and_maps_fixed_class_order() -> None:
    features, labels, groups, sample_ids = _balanced_group_case()
    contexts: list[OOFFoldEstimatorContext] = []

    def factory(context: OOFFoldEstimatorContext) -> _FixedProbabilityEstimator:
        contexts.append(context)
        # The engine must map these columns from (2, 0, 1) to fixed (0, 1, 2).
        return _FixedProbabilityEstimator((2, 0, 1), (0.6, 0.1, 0.3))

    result = grouped_oof_predict(
        features,
        labels,
        groups,
        estimator_factory=factory,
        model_name="fixed_test_estimator",
        final_reference_group_ids=("final_group_a", "final_group_b"),
        sample_ids=sample_ids,
        n_splits=3,
        class_order=(0, 1, 2),
        split_seed=17,
        model_seed=101,
        representation="test_features",
    )

    np.testing.assert_array_equal(result.coverage_count, np.ones(len(labels), dtype=np.int64))
    np.testing.assert_allclose(
        result.probabilities,
        np.tile(np.array([0.1, 0.3, 0.6]), (len(labels), 1)),
    )
    np.testing.assert_array_equal(result.predicted_class, np.full(len(labels), 2))
    assert result.class_order == (0, 1, 2)
    assert [context.fold_id for context in contexts] == [0, 1, 2]
    assert [context.model_seed for context in contexts] == [101, 102, 103]
    assert all(context.class_order == (0, 1, 2) for context in contexts)
    assert result.final_reference_groups == ("final_group_a", "final_group_b")
    assert result.fold_assignment_label_source == "observed_label"
    assert len(result.fold_assignment_labels_sha256) == 64
    assert result.splitter_class_name == "sklearn.model_selection.StratifiedGroupKFold"
    assert result.splitter_fallback_status == "not_used"
    assert result.splitter_fallback_reason is None
    np.testing.assert_array_equal(result.fold_assignment_labels, labels)
    for fold in result.folds:
        assert set(fold.training_groups).isdisjoint(fold.held_out_groups)
        assert set(fold.training_groups).isdisjoint(result.final_reference_groups)
        assert set(fold.held_out_groups).isdisjoint(result.final_reference_groups)


def test_group_fold_plan_exactly_matches_seeded_sklearn_stratified_splitter() -> None:
    features, labels, groups, _ = _balanced_group_case()
    plan = make_group_stratified_fold_plan(
        labels,
        groups,
        n_splits=3,
        class_order=(0, 1, 2),
        seed=71,
    )

    from sklearn.model_selection import StratifiedGroupKFold  # type: ignore[import-untyped]

    expected = tuple(
        StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=71,
        ).split(features, labels, groups)
    )
    assert plan.splitter_class_name == "sklearn.model_selection.StratifiedGroupKFold"
    assert plan.splitter_fallback_status == "not_used"
    assert plan.splitter_fallback_reason is None
    for actual, (expected_train, expected_holdout) in zip(plan.folds, expected, strict=True):
        np.testing.assert_array_equal(actual.train_indices, expected_train)
        np.testing.assert_array_equal(actual.holdout_indices, expected_holdout)


def test_group_fold_plan_records_deterministic_groupkfold_fallback() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    groups = ("group_0", "group_1", "group_2", "group_3")
    features = np.column_stack((np.arange(4, dtype=np.float64), labels))

    def factory(context: OOFFoldEstimatorContext) -> _FixedProbabilityEstimator:
        return _FixedProbabilityEstimator(context.class_order, (0.4, 0.6))

    first = grouped_oof_predict(
        features,
        labels,
        groups,
        estimator_factory=factory,
        model_name="fixed_test_estimator",
        final_reference_group_ids=("final",),
        n_splits=3,
        class_order=(0, 1),
        split_seed=41,
    )
    second = grouped_oof_predict(
        features,
        labels,
        groups,
        estimator_factory=factory,
        model_name="fixed_test_estimator",
        final_reference_group_ids=("final",),
        n_splits=3,
        class_order=(0, 1),
        split_seed=41,
    )

    assert first.splitter_class_name == "sklearn.model_selection.GroupKFold"
    assert first.splitter_fallback_status == "used"
    assert first.splitter_fallback_reason is not None
    assert first.splitter_fallback_reason == (
        "StratifiedGroupKFold infeasible: split raised ValueError"
    )
    assert first.splitter_fallback_reason == second.splitter_fallback_reason
    np.testing.assert_array_equal(first.fold_id, second.fold_id)
    assert [fold.held_out_groups for fold in first.folds] == [
        fold.held_out_groups for fold in second.folds
    ]

    from sklearn.model_selection import GroupKFold  # type: ignore[import-untyped]

    expected_fold_id = np.full(len(labels), -1, dtype=np.int64)
    for fold_id, (_, holdout) in enumerate(
        GroupKFold(n_splits=3, shuffle=True, random_state=41).split(
            features,
            labels,
            groups,
        )
    ):
        expected_fold_id[holdout] = fold_id
    np.testing.assert_array_equal(first.fold_id, expected_fold_id)


def test_group_fold_valueerror_fallback_reason_is_cross_version_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    groups = ("group_0", "group_1", "group_2", "group_3")
    raw_messages = iter(
        (
            "version A wording with implementation details",
            "completely different version B wording",
        )
    )

    from sklearn.model_selection import (  # type: ignore[import-untyped]
        StratifiedGroupKFold,
    )

    def fail_with_version_specific_message(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ValueError(next(raw_messages))

    monkeypatch.setattr(StratifiedGroupKFold, "split", fail_with_version_specific_message)
    reasons = tuple(
        make_group_stratified_fold_plan(
            labels,
            groups,
            n_splits=3,
            class_order=(0, 1),
            seed=41,
        ).splitter_fallback_reason
        for _ in range(2)
    )

    assert reasons == (
        "StratifiedGroupKFold infeasible: split raised ValueError",
        "StratifiedGroupKFold infeasible: split raised ValueError",
    )
    assert all("wording" not in str(reason) for reason in reasons)


def test_group_fold_plan_falls_back_when_stratified_training_loses_a_class() -> None:
    group_class_counts = {
        "g0": (3, 0, 0),
        "g1": (0, 1, 4),
        "g2": (2, 3, 1),
        "g3": (0, 0, 1),
    }
    labels = np.asarray(
        [
            class_id
            for counts in group_class_counts.values()
            for class_id, count in enumerate(counts)
            for _ in range(count)
        ],
        dtype=np.int64,
    )
    groups = tuple(
        group_id
        for group_id, counts in group_class_counts.items()
        for count in counts
        for _ in range(count)
    )
    plan = make_group_stratified_fold_plan(
        labels,
        groups,
        n_splits=2,
        class_order=(0, 1, 2),
        seed=23,
    )

    assert plan.splitter_class_name == "sklearn.model_selection.GroupKFold"
    assert plan.splitter_fallback_status == "used"
    assert plan.splitter_fallback_reason == (
        "StratifiedGroupKFold infeasible: training partition is missing fixed "
        "class_order values: fold 1 missing [0]"
    )
    assert all(set(labels[fold.train_indices].tolist()) == {0, 1, 2} for fold in plan.folds)

    from sklearn.model_selection import GroupKFold  # type: ignore[import-untyped]

    features = np.zeros((len(labels), 1), dtype=np.float64)
    expected = tuple(
        GroupKFold(n_splits=2, shuffle=True, random_state=23).split(
            features,
            labels,
            groups,
        )
    )
    for actual, (expected_train, expected_holdout) in zip(
        plan.folds,
        expected,
        strict=True,
    ):
        np.testing.assert_array_equal(actual.train_indices, expected_train)
        np.testing.assert_array_equal(actual.holdout_indices, expected_holdout)


def test_fixed_pre_corruption_assignment_is_shared_across_corruptions_but_fit_uses_observed() -> (
    None
):
    features, pre_corruption_labels, groups, sample_ids = _balanced_group_case()
    observed_first = pre_corruption_labels.copy()
    observed_second = pre_corruption_labels.copy()
    observed_first[::4] = (observed_first[::4] + 1) % 3
    observed_second[1::4] = (observed_second[1::4] + 2) % 3
    fit_records: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []

    class RecordingEstimator(_FixedProbabilityEstimator):
        def fit(
            self,
            fold_features: NDArray[np.generic],
            fold_labels: NDArray[np.generic],
        ) -> RecordingEstimator:
            indices = np.asarray(fold_features[:, 0], dtype=np.int64)
            labels = np.asarray(fold_labels, dtype=np.int64)
            fit_records.append((indices.copy(), labels.copy()))
            super().fit(fold_features, fold_labels)
            return self

    def factory(context: OOFFoldEstimatorContext) -> RecordingEstimator:
        return RecordingEstimator(context.class_order, (0.2, 0.3, 0.5))

    first = grouped_oof_predict(
        features,
        observed_first,
        groups,
        estimator_factory=factory,
        model_name="fixed_test_estimator",
        final_reference_group_ids=("final",),
        sample_ids=sample_ids,
        fold_assignment_labels=pre_corruption_labels,
        fold_assignment_label_source="pre_corruption_label",
        n_splits=3,
        class_order=(0, 1, 2),
        split_seed=31,
    )
    first_fit_records = tuple(fit_records)
    fit_records.clear()
    second = grouped_oof_predict(
        features,
        observed_second,
        groups,
        estimator_factory=factory,
        model_name="fixed_test_estimator",
        final_reference_group_ids=("final",),
        sample_ids=sample_ids,
        fold_assignment_labels=pre_corruption_labels,
        fold_assignment_label_source="pre_corruption_label",
        n_splits=3,
        class_order=(0, 1, 2),
        split_seed=31,
    )

    np.testing.assert_array_equal(first.fold_id, second.fold_id)
    assert first.fold_assignment_labels_sha256 == second.fold_assignment_labels_sha256
    assert first.fold_assignment_label_source == "pre_corruption_label"
    for indices, fitted_labels in first_fit_records:
        np.testing.assert_array_equal(fitted_labels, observed_first[indices])
    for indices, fitted_labels in fit_records:
        np.testing.assert_array_equal(fitted_labels, observed_second[indices])


@pytest.mark.parametrize(
    ("assignment_labels", "source", "message"),
    [
        (None, "pre_corruption_label", "requires explicit"),
        (np.arange(2, dtype=np.int64), "pre_corruption_label", "aligned"),
        (
            np.tile(np.arange(3, dtype=np.float64), 9),
            "pre_corruption_label",
            "integer vector",
        ),
        (
            np.full(27, 8, dtype=np.int64),
            "pre_corruption_label",
            "outside class_order",
        ),
        (None, "unregistered_source", "must be observed_label or pre_corruption_label"),
    ],
)
def test_generic_oof_rejects_invalid_fold_assignment_contract(
    assignment_labels: NDArray[np.generic] | None,
    source: str,
    message: str,
) -> None:
    features, labels, groups, _ = _balanced_group_case()

    def factory(context: OOFFoldEstimatorContext) -> _FixedProbabilityEstimator:
        return _FixedProbabilityEstimator(context.class_order, (0.2, 0.3, 0.5))

    with pytest.raises(ValueError, match=message):
        grouped_oof_predict(
            features,
            labels,
            groups,
            estimator_factory=factory,
            model_name="fixed_test_estimator",
            final_reference_group_ids=("final",),
            fold_assignment_labels=assignment_labels,
            fold_assignment_label_source=source,
            n_splits=3,
            class_order=(0, 1, 2),
        )


def test_oof_result_validation_detects_assignment_vector_tampering() -> None:
    features, labels, groups, _ = _balanced_group_case()

    def factory(context: OOFFoldEstimatorContext) -> _FixedProbabilityEstimator:
        return _FixedProbabilityEstimator(context.class_order, (0.2, 0.3, 0.5))

    result = grouped_oof_predict(
        features,
        labels,
        groups,
        estimator_factory=factory,
        model_name="fixed_test_estimator",
        final_reference_group_ids=("final",),
        n_splits=3,
        class_order=(0, 1, 2),
    )
    result.fold_assignment_labels[[0, 1]] = result.fold_assignment_labels[[1, 0]]
    with pytest.raises(ValueError, match="SHA-256"):
        result.validate()


@pytest.mark.parametrize("final_groups", [(), ("group_00",)])
def test_generic_oof_fails_closed_without_disjoint_final_group_evidence(
    final_groups: tuple[str, ...],
) -> None:
    features, labels, groups, _ = _balanced_group_case()

    def factory(context: OOFFoldEstimatorContext) -> _FixedProbabilityEstimator:
        return _FixedProbabilityEstimator(context.class_order, (0.2, 0.3, 0.5))

    expected = "evidence is mandatory" if not final_groups else "present in audit pool"
    with pytest.raises(ValueError, match=expected):
        grouped_oof_predict(
            features,
            labels,
            groups,
            estimator_factory=factory,
            model_name="fixed_test_estimator",
            final_reference_group_ids=final_groups,
            n_splits=3,
            class_order=(0, 1, 2),
        )


def test_generic_oof_rejects_training_fold_missing_a_fixed_class() -> None:
    labels = np.array([2, 2, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    groups = ("rare", "rare", "a", "a", "b", "b", "c", "c")
    features = np.column_stack((np.arange(len(labels)), labels)).astype(np.float64)

    def factory(context: OOFFoldEstimatorContext) -> _FixedProbabilityEstimator:
        return _FixedProbabilityEstimator(context.class_order, (0.2, 0.3, 0.5))

    with pytest.raises(ValueError, match="training partition is missing fixed class_order"):
        grouped_oof_predict(
            features,
            labels,
            groups,
            estimator_factory=factory,
            model_name="fixed_test_estimator",
            final_reference_group_ids=("final",),
            n_splits=2,
            class_order=(0, 1, 2),
        )


def test_generic_oof_rejects_estimator_missing_probability_class() -> None:
    features, labels, groups, _ = _balanced_group_case()

    def factory(context: OOFFoldEstimatorContext) -> _FixedProbabilityEstimator:
        del context
        return _FixedProbabilityEstimator((0, 1), (0.4, 0.6))

    with pytest.raises(ValueError, match=r"missing=\[2\]"):
        grouped_oof_predict(
            features,
            labels,
            groups,
            estimator_factory=factory,
            model_name="incomplete_test_estimator",
            final_reference_group_ids=("final",),
            n_splits=3,
            class_order=(0, 1, 2),
        )


def test_frozen_embedding_mlp_adapter_uses_generic_group_safe_contract() -> None:
    features, labels, groups, sample_ids = _balanced_group_case()
    result = grouped_oof_frozen_embedding_mlp(
        features,
        labels,
        groups,
        final_reference_group_ids=("final",),
        sample_ids=sample_ids,
        n_splits=2,
        class_order=(2, 0, 1),
        split_seed=19,
        model_seed=211,
        config=FrozenEmbeddingMLPConfig(
            hidden_dimensions=(4,),
            dropout=0.0,
            epochs=1,
            batch_size=8,
            device="cpu",
        ),
    )

    assert result.model_name == "frozen_embedding_mlp"
    assert result.class_order == (2, 0, 1)
    np.testing.assert_array_equal(result.coverage_count, np.ones(len(labels), dtype=np.int64))
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-6)
    for fold in result.folds:
        assert set(fold.training_groups).isdisjoint(fold.held_out_groups)
