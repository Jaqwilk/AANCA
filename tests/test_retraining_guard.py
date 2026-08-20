from __future__ import annotations

import numpy as np

from histo_audit.evaluation.retraining_guard import (
    INDEPENDENT_GROUP_VALIDATION,
    evaluate_multicriteria_retraining_guard,
    evaluate_retraining_guard,
)


def _probabilities(predictions: list[int]) -> np.ndarray:
    return np.eye(2, dtype=np.float64)[np.asarray(predictions, dtype=np.int64)]


def test_guard_accepts_only_clearly_better_independent_candidate() -> None:
    reference = np.asarray([0, 1] * 4, dtype=np.int64)
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    baseline = _probabilities([1, 0] * 4)
    candidate = _probabilities(reference.tolist())

    decision = evaluate_retraining_guard(
        reference,
        baseline,
        candidate,
        groups,
        class_order=(0, 1),
        evidence_role=INDEPENDENT_GROUP_VALIDATION,
        n_iterations=100,
        seed=7,
    )

    assert decision.apply_candidate is True
    assert decision.action == "apply_candidate"
    assert decision.interval_95 == (1.0, 1.0)


def test_guard_retains_uncorrected_for_adverse_or_non_independent_evidence() -> None:
    reference = np.asarray([0, 1] * 4, dtype=np.int64)
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    baseline = _probabilities(reference.tolist())
    adverse = _probabilities([1, 0] * 4)

    adverse_decision = evaluate_retraining_guard(
        reference,
        baseline,
        adverse,
        groups,
        class_order=(0, 1),
        evidence_role=INDEPENDENT_GROUP_VALIDATION,
        n_iterations=100,
        seed=7,
    )
    non_independent_decision = evaluate_retraining_guard(
        reference,
        adverse,
        baseline,
        groups,
        class_order=(0, 1),
        evidence_role="training_resubstitution",
        n_iterations=100,
        seed=7,
    )

    assert adverse_decision.apply_candidate is False
    assert adverse_decision.action == "retain_uncorrected"
    assert adverse_decision.interval_95 == (-1.0, -1.0)
    assert non_independent_decision.apply_candidate is False
    assert "not an independent" in non_independent_decision.reason


def test_multicriteria_guard_rejects_macro_gain_that_harms_an_important_class() -> None:
    reference = np.tile(np.asarray([0, 1, 2], dtype=np.int64), 4)
    groups = [group for group in ("a", "b", "c", "d") for _ in range(3)]
    baseline = np.eye(3)[np.tile(np.asarray([0, 0, 0]), 4)]
    candidate = np.eye(3)[np.tile(np.asarray([1, 1, 2]), 4)]

    decision = evaluate_multicriteria_retraining_guard(
        reference,
        baseline,
        candidate,
        groups,
        class_order=(0, 1, 2),
        important_classes=(0, 1, 2),
        evidence_role=INDEPENDENT_GROUP_VALIDATION,
        n_iterations=100,
        seed=9,
        minimum_per_class_recall_effect=-0.1,
    )

    assert decision.macro_f1.apply_candidate is True
    assert decision.candidate_minus_uncorrected_recall[0] == -1.0
    assert decision.per_class_recall_intervals_95[0] == (-1.0, -1.0)
    assert decision.important_classes_pass is False
    assert decision.apply_candidate is False
    assert decision.action == "retain_uncorrected"


def test_multicriteria_guard_accepts_only_when_macro_and_all_classes_pass() -> None:
    reference = np.tile(np.asarray([0, 1, 2], dtype=np.int64), 4)
    groups = [group for group in ("a", "b", "c", "d") for _ in range(3)]
    baseline = np.eye(3)[np.tile(np.asarray([1, 2, 0]), 4)]
    candidate = np.eye(3)[reference]

    decision = evaluate_multicriteria_retraining_guard(
        reference,
        baseline,
        candidate,
        groups,
        class_order=(0, 1, 2),
        evidence_role=INDEPENDENT_GROUP_VALIDATION,
        n_iterations=100,
        seed=9,
        minimum_per_class_recall_effect=0.0,
    )

    assert decision.apply_candidate is True
    assert decision.important_classes_pass is True
    assert all(
        interval == (1.0, 1.0) for interval in decision.per_class_recall_intervals_95.values()
    )
    assert decision.as_dict()["action"] == "apply_candidate"


def test_multicriteria_guard_rejects_sparse_class_bootstrap_coverage() -> None:
    reference = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 2], dtype=np.int64)
    groups = ["a", "a", "b", "b", "c", "c", "d", "d", "d"]
    baseline = np.eye(3)[np.asarray([1, 0, 1, 0, 1, 0, 1, 0, 0])]
    candidate = np.eye(3)[reference]

    decision = evaluate_multicriteria_retraining_guard(
        reference,
        baseline,
        candidate,
        groups,
        class_order=(0, 1, 2),
        important_classes=(2,),
        evidence_role=INDEPENDENT_GROUP_VALIDATION,
        n_iterations=500,
        seed=21,
        minimum_per_class_recall_effect=0.0,
        minimum_valid_iteration_fraction=0.95,
    )

    assert decision.macro_f1.apply_candidate is True
    assert decision.per_class_valid_iterations[2] < 475
    assert decision.important_classes_pass is False
    assert decision.apply_candidate is False
