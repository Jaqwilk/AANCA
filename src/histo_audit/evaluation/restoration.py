"""Controlled restoration semantics and fixed-model downstream evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from histo_audit.corruption.controlled import normalise_rate
from histo_audit.cross_validation.oof import MultinomialLogisticRegression
from histo_audit.statistics.review import budget_count, rank_indices


class DownstreamEstimator(Protocol):
    """Minimal probabilistic estimator contract used by restoration evaluation."""

    def fit(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> Any:
        """Fit one fresh estimator on one restoration condition."""

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.generic]:
        """Return probabilities whose columns follow the factory's class order."""


class DownstreamEstimatorFactory(Protocol):
    """Create fresh, identically configured estimators for all conditions."""

    def __call__(
        self,
        *,
        class_order: tuple[int, ...],
        model_seed: int,
    ) -> DownstreamEstimator:
        """Build an estimator using the supplied fixed class order and seed."""


@dataclass(frozen=True, slots=True)
class _LogisticEstimatorFactory:
    """Backward-compatible default downstream estimator factory."""

    l2: float
    max_iter: int

    def __call__(
        self,
        *,
        class_order: tuple[int, ...],
        model_seed: int,
    ) -> DownstreamEstimator:
        # The convex deterministic implementation has no stochastic operations,
        # but every factory receives the same seed so stochastic alternatives can
        # obey the exact same-condition contract.
        _ = model_seed
        return MultinomialLogisticRegression(
            class_order=class_order,
            l2=self.l2,
            max_iter=self.max_iter,
            class_weight_balanced=True,
        )


@dataclass(frozen=True, slots=True)
class RestorationResult:
    """Labels after simulated review; only found injected corruptions are restored."""

    restored_labels: NDArray[np.int64]
    reviewed_mask: NDArray[np.bool_]
    restored_mask: NDArray[np.bool_]
    reviewed_count: int
    restored_count: int


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Downstream multiclass metrics derived from one saved probability matrix."""

    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    per_class_precision: tuple[float, ...]
    per_class_recall: tuple[float, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    multiclass_brier_score: float
    expected_calibration_error: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DownstreamRun:
    """One fixed-classifier downstream experiment condition."""

    experiment_name: str
    metrics: ClassificationMetrics
    final_test_probabilities: NDArray[np.float64]
    final_test_predicted_class: NDArray[np.int64]
    reviewed_count: int
    restored_count: int
    review_seed: int | None


@dataclass(frozen=True, slots=True)
class DownstreamEvaluation:
    """All four required restoration conditions at identical review budgets."""

    uncorrupted_reference_baseline: DownstreamRun
    corrupted_observed_baseline: DownstreamRun
    random_review_restoration: tuple[DownstreamRun, ...]
    audit_guided_restoration: DownstreamRun
    review_budget_fraction: float
    review_budget_count: int
    audit_reviewed_indices: NDArray[np.int64]
    random_reviewed_indices: tuple[NDArray[np.int64], ...]
    audit_guided_restoration_evidence: RestorationResult
    random_review_restoration_evidence: tuple[RestorationResult, ...]
    random_macro_f1_mean: float
    random_macro_f1_interval_95: tuple[float, float]
    development_groups: tuple[str, ...]
    final_reference_groups: tuple[str, ...]
    final_test_uncorrupted_verified: bool
    reference_validation_groups: tuple[str, ...] = ()
    reference_validation_sample_count: int = 0
    model_seed: int = 313

    def as_dict(self) -> dict[str, Any]:
        """Convert metrics and counts to JSON-serializable primitives."""

        return {
            "uncorrupted_reference_baseline": {
                "metrics": self.uncorrupted_reference_baseline.metrics.as_dict(),
                "reviewed_count": 0,
                "restored_count": 0,
            },
            "corrupted_observed_baseline": {
                "metrics": self.corrupted_observed_baseline.metrics.as_dict(),
                "reviewed_count": 0,
                "restored_count": 0,
            },
            "random_review_restoration": {
                "runs": [
                    {
                        "metrics": run.metrics.as_dict(),
                        "reviewed_count": run.reviewed_count,
                        "restored_count": run.restored_count,
                        "review_seed": run.review_seed,
                    }
                    for run in self.random_review_restoration
                ],
                "macro_f1_mean": self.random_macro_f1_mean,
                "macro_f1_interval_95": self.random_macro_f1_interval_95,
            },
            "audit_guided_restoration": {
                "metrics": self.audit_guided_restoration.metrics.as_dict(),
                "reviewed_count": self.audit_guided_restoration.reviewed_count,
                "restored_count": self.audit_guided_restoration.restored_count,
            },
            "review_budget_fraction": self.review_budget_fraction,
            "review_budget_count": self.review_budget_count,
            "partition_evidence": {
                "development_groups": self.development_groups,
                "reference_validation_groups": self.reference_validation_groups,
                "reference_validation_sample_count": self.reference_validation_sample_count,
                "reference_validation_in_training": self.reference_validation_sample_count > 0,
                "final_reference_groups": self.final_reference_groups,
                "group_overlap_count": 0,
                "final_test_uncorrupted_verified": self.final_test_uncorrupted_verified,
            },
            "model_evidence": {
                "model_seed": self.model_seed,
                "same_factory_configuration_all_conditions": True,
            },
        }


def _review_mask(
    n_samples: int, reviewed: Sequence[int] | NDArray[np.generic]
) -> NDArray[np.bool_]:
    values = np.asarray(reviewed)
    if values.dtype == bool:
        if values.shape != (n_samples,):
            raise ValueError("reviewed boolean mask must align with labels")
        return values.astype(bool, copy=True)
    indices = np.asarray(reviewed, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= n_samples):
        raise ValueError("reviewed indices are invalid")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("reviewed indices must be unique")
    mask = np.zeros(n_samples, dtype=bool)
    mask[indices] = True
    return mask


def restore_reviewed_labels(
    pre_corruption_labels: Sequence[int] | NDArray[np.integer],
    observed_labels: Sequence[int] | NDArray[np.integer],
    is_injected_corruption: Sequence[bool] | NDArray[np.bool_],
    reviewed: Sequence[int] | NDArray[np.generic],
) -> RestorationResult:
    """Restore only reviewed, known injected corruptions to their reference label."""

    reference = np.asarray(pre_corruption_labels, dtype=np.int64)
    observed = np.asarray(observed_labels, dtype=np.int64)
    injected = np.asarray(is_injected_corruption, dtype=bool)
    if (
        reference.ndim != 1
        or observed.shape != reference.shape
        or injected.shape != reference.shape
    ):
        raise ValueError("reference, observed, and injected arrays must align")
    if not np.array_equal(reference != observed, injected):
        raise ValueError("injected mask must exactly identify controlled label changes")
    reviewed_mask = _review_mask(len(reference), reviewed)
    restored_mask = reviewed_mask & injected
    restored = observed.copy()
    restored[restored_mask] = reference[restored_mask]
    if not np.array_equal(restored[~restored_mask], observed[~restored_mask]):
        raise RuntimeError("unreviewed labels changed during restoration")
    return RestorationResult(
        restored_labels=restored,
        reviewed_mask=reviewed_mask,
        restored_mask=restored_mask,
        reviewed_count=int(reviewed_mask.sum()),
        restored_count=int(restored_mask.sum()),
    )


def macro_f1_from_confusion(confusion: NDArray[np.integer]) -> float:
    """Return unweighted multiclass F1 from a square confusion matrix."""

    matrix = np.asarray(confusion, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
        raise ValueError("confusion matrix must be non-empty and square")
    true_positive = np.diag(matrix)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0
    )
    recall = np.divide(true_positive, actual, out=np.zeros_like(true_positive), where=actual > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return float(f1.mean())


def per_class_recall_from_confusion(
    confusion: NDArray[np.integer],
) -> NDArray[np.float64]:
    """Return class recall, using NaN for classes absent from the reference."""

    matrix = np.asarray(confusion, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
        raise ValueError("confusion matrix must be non-empty and square")
    actual = matrix.sum(axis=1)
    return np.divide(
        np.diag(matrix),
        actual,
        out=np.full(len(matrix), np.nan, dtype=np.float64),
        where=actual > 0,
    )


def classification_metrics(
    reference_labels: Sequence[int] | NDArray[np.integer],
    probabilities: NDArray[np.generic],
    *,
    class_order: Sequence[int] = (0, 1, 2, 3, 4),
    calibration_bins: int = 10,
) -> ClassificationMetrics:
    """Calculate deterministic multiclass classification and calibration metrics."""

    reference = np.asarray(reference_labels, dtype=np.int64)
    matrix = np.asarray(probabilities, dtype=np.float64)
    classes = tuple(int(value) for value in class_order)
    if matrix.shape != (len(reference), len(classes)) or not len(reference):
        raise ValueError("probabilities must align with labels and class order")
    if not np.isfinite(matrix).all() or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-7):
        raise ValueError("probabilities must be finite and sum to one")
    if calibration_bins <= 0:
        raise ValueError("calibration_bins must be positive")
    lookup = {label: column for column, label in enumerate(classes)}
    if any(int(label) not in lookup for label in reference):
        raise ValueError("reference label absent from class_order")
    predictions = np.asarray(classes, dtype=np.int64)[np.argmax(matrix, axis=1)]
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for true_label, predicted_label in zip(reference, predictions, strict=True):
        confusion[lookup[int(true_label)], lookup[int(predicted_label)]] += 1
    true_positive = np.diag(confusion).astype(np.float64)
    predicted_count = confusion.sum(axis=0).astype(np.float64)
    actual_count = confusion.sum(axis=1).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive),
        where=predicted_count > 0,
    )
    recall = np.divide(
        true_positive,
        actual_count,
        out=np.zeros_like(true_positive),
        where=actual_count > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    supported = actual_count > 0
    one_hot = np.zeros_like(matrix)
    one_hot[np.arange(len(reference)), [lookup[int(label)] for label in reference]] = 1.0
    brier = float(np.mean(np.sum((matrix - one_hot) ** 2, axis=1)))
    confidence = matrix.max(axis=1)
    correct = predictions == reference
    edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    expected_calibration_error = 0.0
    for bin_index in range(calibration_bins):
        if bin_index == calibration_bins - 1:
            members = (confidence >= edges[bin_index]) & (confidence <= edges[bin_index + 1])
        else:
            members = (confidence >= edges[bin_index]) & (confidence < edges[bin_index + 1])
        if members.any():
            expected_calibration_error += float(members.mean()) * abs(
                float(correct[members].mean()) - float(confidence[members].mean())
            )
    return ClassificationMetrics(
        accuracy=float(np.mean(predictions == reference)),
        macro_f1=float(f1.mean()),
        balanced_accuracy=float(recall[supported].mean()) if supported.any() else 0.0,
        per_class_precision=tuple(float(value) for value in precision),
        per_class_recall=tuple(float(value) for value in recall),
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in confusion),
        multiclass_brier_score=brier,
        expected_calibration_error=expected_calibration_error,
    )


def _fit_evaluate(
    train_features: NDArray[np.float64],
    train_labels: NDArray[np.int64],
    test_features: NDArray[np.float64],
    test_labels: NDArray[np.int64],
    *,
    class_order: tuple[int, ...],
    estimator_factory: DownstreamEstimatorFactory,
    model_seed: int,
) -> tuple[ClassificationMetrics, NDArray[np.float64], NDArray[np.int64]]:
    classifier = estimator_factory(class_order=class_order, model_seed=model_seed)
    if not callable(getattr(classifier, "fit", None)) or not callable(
        getattr(classifier, "predict_proba", None)
    ):
        raise TypeError("estimator_factory must return an estimator with fit and predict_proba")
    fit_features = np.array(train_features, dtype=np.float64, copy=True)
    fit_labels = np.array(train_labels, dtype=np.int64, copy=True)
    fit_labels.setflags(write=False)
    classifier.fit(fit_features, fit_labels)
    probabilities = np.asarray(
        classifier.predict_proba(np.array(test_features, dtype=np.float64, copy=True)),
        dtype=np.float64,
    )
    reported_order = getattr(classifier, "class_order", None)
    if reported_order is None:
        reported_order = getattr(classifier, "classes_", None)
    if reported_order is not None and tuple(int(value) for value in reported_order) != class_order:
        raise ValueError("estimator probability columns do not follow the fixed class_order")
    metrics = classification_metrics(test_labels, probabilities, class_order=class_order)
    predicted_class = np.asarray(class_order, dtype=np.int64)[np.argmax(probabilities, axis=1)]
    return (
        metrics,
        probabilities,
        predicted_class,
    )


def evaluate_downstream_restoration(
    development_features: NDArray[np.generic],
    pre_corruption_labels: Sequence[int] | NDArray[np.integer],
    observed_labels: Sequence[int] | NDArray[np.integer],
    is_injected_corruption: Sequence[bool] | NDArray[np.bool_],
    final_test_features: NDArray[np.generic],
    final_test_reference_labels: Sequence[int] | NDArray[np.integer],
    audit_risk_scores: Sequence[float] | NDArray[np.floating],
    *,
    development_group_ids: Sequence[str],
    final_test_group_ids: Sequence[str],
    final_test_is_injected_corruption: Sequence[bool],
    review_budget: float | int = 0.05,
    sample_ids: Sequence[str] | None = None,
    class_order: Sequence[int] = (0, 1, 2, 3, 4),
    random_repeats: int = 20,
    random_seed: int = 307,
    model_seed: int = 313,
    l2: float = 1.0e-2,
    max_iter: int = 400,
    estimator_factory: DownstreamEstimatorFactory | None = None,
    reference_validation_features: NDArray[np.generic] | None = None,
    reference_validation_labels: Sequence[int] | NDArray[np.integer] | None = None,
    reference_validation_group_ids: Sequence[str] | None = None,
    reference_validation_is_injected_corruption: (Sequence[bool] | NDArray[np.bool_] | None) = None,
) -> DownstreamEvaluation:
    """Run the four required conditions with a fixed model and untouched final test.

    When supplied, the clean reference-validation partition is appended unchanged
    to the training data for every condition. It is never eligible for review and
    must carry explicit evidence that it contains no injected corruption.
    """

    train_matrix = np.asarray(development_features, dtype=np.float64)
    reference = np.asarray(pre_corruption_labels, dtype=np.int64)
    observed = np.asarray(observed_labels, dtype=np.int64)
    injected = np.asarray(is_injected_corruption, dtype=bool)
    test_matrix = np.asarray(final_test_features, dtype=np.float64)
    test_reference = np.asarray(final_test_reference_labels, dtype=np.int64)
    reference_before = reference.copy()
    observed_before = observed.copy()
    test_reference_before = test_reference.copy()
    risks = np.asarray(audit_risk_scores, dtype=np.float64)
    if train_matrix.ndim != 2 or train_matrix.shape[0] != len(reference):
        raise ValueError("development features and labels must align")
    if (
        observed.shape != reference.shape
        or injected.shape != reference.shape
        or risks.shape != reference.shape
    ):
        raise ValueError("development label, flag, and risk vectors must align")
    if test_matrix.ndim != 2 or test_matrix.shape[0] != len(test_reference):
        raise ValueError("final-test features and labels must align")
    if train_matrix.shape[1] != test_matrix.shape[1]:
        raise ValueError("development and final-test feature dimensions differ")
    if not np.isfinite(train_matrix).all() or not np.isfinite(test_matrix).all():
        raise ValueError("development and final-test features must be finite")
    if not np.array_equal(reference != observed, injected):
        raise ValueError("injected mask must exactly identify development label changes")
    final_injected = np.asarray(final_test_is_injected_corruption, dtype=bool)
    if final_injected.shape != (len(test_reference),):
        raise ValueError("final-test corruption evidence must align with final-test labels")
    if final_injected.any():
        raise ValueError("final reference test must remain uncorrupted")
    if len(development_group_ids) != len(reference) or len(final_test_group_ids) != len(
        test_reference
    ):
        raise ValueError("group IDs must align with their partitions")
    development_groups = {str(value) for value in development_group_ids}
    final_groups = {str(value) for value in final_test_group_ids}
    if not development_groups or not final_groups or "" in development_groups | final_groups:
        raise ValueError("non-empty development and final-test group evidence is mandatory")
    overlap = development_groups.intersection(final_groups)
    if overlap:
        raise ValueError(f"development/final-test source-group leakage: {sorted(overlap)}")
    validation_arguments = (
        reference_validation_features,
        reference_validation_labels,
        reference_validation_group_ids,
        reference_validation_is_injected_corruption,
    )
    supplied_validation_arguments = tuple(value is not None for value in validation_arguments)
    if any(supplied_validation_arguments) and not all(supplied_validation_arguments):
        raise ValueError(
            "reference-validation features, labels, groups, and corruption evidence "
            "must be supplied together"
        )
    if all(supplied_validation_arguments):
        assert reference_validation_features is not None
        assert reference_validation_labels is not None
        assert reference_validation_group_ids is not None
        assert reference_validation_is_injected_corruption is not None
        validation_matrix = np.asarray(reference_validation_features, dtype=np.float64)
        validation_labels = np.asarray(reference_validation_labels, dtype=np.int64)
        validation_injected = np.asarray(reference_validation_is_injected_corruption, dtype=bool)
        validation_group_values = tuple(str(value) for value in reference_validation_group_ids)
        if validation_matrix.ndim != 2 or not len(validation_matrix):
            raise ValueError("reference-validation features must be a non-empty matrix")
        if validation_matrix.shape[1] != train_matrix.shape[1]:
            raise ValueError("reference-validation feature dimension differs from development")
        if validation_labels.shape != (len(validation_matrix),) or validation_injected.shape != (
            len(validation_matrix),
        ):
            raise ValueError("reference-validation labels and corruption evidence must align")
        if len(validation_group_values) != len(validation_matrix):
            raise ValueError("reference-validation group IDs must align with the partition")
        if not np.isfinite(validation_matrix).all():
            raise ValueError("reference-validation features must be finite")
        if validation_injected.any():
            raise ValueError("reference validation must remain clean and uncorrupted")
        reference_validation_groups = set(validation_group_values)
        if not reference_validation_groups or "" in reference_validation_groups:
            raise ValueError("non-empty reference-validation group evidence is mandatory")
        development_validation_overlap = development_groups.intersection(
            reference_validation_groups
        )
        if development_validation_overlap:
            raise ValueError(
                "development/reference-validation source-group leakage: "
                f"{sorted(development_validation_overlap)}"
            )
        validation_final_overlap = reference_validation_groups.intersection(final_groups)
        if validation_final_overlap:
            raise ValueError(
                "reference-validation/final-test source-group leakage: "
                f"{sorted(validation_final_overlap)}"
            )
    else:
        validation_matrix = np.empty((0, train_matrix.shape[1]), dtype=np.float64)
        validation_labels = np.empty(0, dtype=np.int64)
        validation_injected = np.empty(0, dtype=bool)
        reference_validation_groups = set()
    validation_labels_before = validation_labels.copy()
    if random_repeats <= 0:
        raise ValueError("random_repeats must be positive")
    classes = tuple(int(value) for value in class_order)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("class_order must contain at least two unique values")
    if any(
        int(label) not in classes
        for labels in (reference, observed, validation_labels, test_reference)
        for label in labels
    ):
        raise ValueError("all downstream labels must be present in class_order")
    factory = (
        estimator_factory
        if estimator_factory is not None
        else _LogisticEstimatorFactory(l2=l2, max_iter=max_iter)
    )
    training_features = np.concatenate((train_matrix, validation_matrix), axis=0)

    def training_labels(condition_labels: NDArray[np.int64]) -> NDArray[np.int64]:
        return np.concatenate((condition_labels, validation_labels), axis=0)

    count = budget_count(len(reference), review_budget)
    identifiers = sample_ids if sample_ids is not None else None
    audit_indices = rank_indices(risks, tie_break_ids=identifiers)[:count]

    uncorrupted_metrics, uncorrupted_probabilities, uncorrupted_predicted_class = _fit_evaluate(
        training_features,
        training_labels(reference),
        test_matrix,
        test_reference,
        class_order=classes,
        estimator_factory=factory,
        model_seed=model_seed,
    )
    corrupted_metrics, corrupted_probabilities, corrupted_predicted_class = _fit_evaluate(
        training_features,
        training_labels(observed),
        test_matrix,
        test_reference,
        class_order=classes,
        estimator_factory=factory,
        model_seed=model_seed,
    )
    guided_restoration = restore_reviewed_labels(reference, observed, injected, audit_indices)
    guided_metrics, guided_probabilities, guided_predicted_class = _fit_evaluate(
        training_features,
        training_labels(guided_restoration.restored_labels),
        test_matrix,
        test_reference,
        class_order=classes,
        estimator_factory=factory,
        model_seed=model_seed,
    )
    random_runs: list[DownstreamRun] = []
    random_indices: list[NDArray[np.int64]] = []
    random_restorations: list[RestorationResult] = []
    for repeat in range(random_repeats):
        review_seed = random_seed + repeat
        rng = np.random.default_rng(review_seed)
        selected = np.sort(rng.choice(len(reference), size=count, replace=False)).astype(np.int64)
        random_indices.append(selected)
        random_restoration = restore_reviewed_labels(reference, observed, injected, selected)
        random_restorations.append(random_restoration)
        random_metrics, random_probabilities, random_predicted_class = _fit_evaluate(
            training_features,
            training_labels(random_restoration.restored_labels),
            test_matrix,
            test_reference,
            class_order=classes,
            estimator_factory=factory,
            model_seed=model_seed,
        )
        random_runs.append(
            DownstreamRun(
                experiment_name="random_review_restoration",
                metrics=random_metrics,
                final_test_probabilities=random_probabilities,
                final_test_predicted_class=random_predicted_class,
                reviewed_count=random_restoration.reviewed_count,
                restored_count=random_restoration.restored_count,
                review_seed=review_seed,
            )
        )
    random_f1 = np.asarray([run.metrics.macro_f1 for run in random_runs])
    if (
        not np.array_equal(reference, reference_before)
        or not np.array_equal(observed, observed_before)
        or not np.array_equal(validation_labels, validation_labels_before)
        or not np.array_equal(test_reference, test_reference_before)
    ):
        raise RuntimeError("an immutable downstream label partition changed during evaluation")
    if any(len(indices) != count for indices in random_indices) or len(audit_indices) != count:
        raise RuntimeError("audit and random review budgets differ")
    return DownstreamEvaluation(
        uncorrupted_reference_baseline=DownstreamRun(
            experiment_name="uncorrupted_reference_baseline",
            metrics=uncorrupted_metrics,
            final_test_probabilities=uncorrupted_probabilities,
            final_test_predicted_class=uncorrupted_predicted_class,
            reviewed_count=0,
            restored_count=0,
            review_seed=None,
        ),
        corrupted_observed_baseline=DownstreamRun(
            experiment_name="corrupted_observed_baseline",
            metrics=corrupted_metrics,
            final_test_probabilities=corrupted_probabilities,
            final_test_predicted_class=corrupted_predicted_class,
            reviewed_count=0,
            restored_count=0,
            review_seed=None,
        ),
        random_review_restoration=tuple(random_runs),
        audit_guided_restoration=DownstreamRun(
            experiment_name="audit_guided_restoration",
            metrics=guided_metrics,
            final_test_probabilities=guided_probabilities,
            final_test_predicted_class=guided_predicted_class,
            reviewed_count=guided_restoration.reviewed_count,
            restored_count=guided_restoration.restored_count,
            review_seed=None,
        ),
        review_budget_fraction=normalise_rate(review_budget),
        review_budget_count=count,
        audit_reviewed_indices=audit_indices,
        random_reviewed_indices=tuple(random_indices),
        audit_guided_restoration_evidence=guided_restoration,
        random_review_restoration_evidence=tuple(random_restorations),
        random_macro_f1_mean=float(random_f1.mean()),
        random_macro_f1_interval_95=(
            float(np.quantile(random_f1, 0.025)),
            float(np.quantile(random_f1, 0.975)),
        ),
        development_groups=tuple(sorted(development_groups)),
        final_reference_groups=tuple(sorted(final_groups)),
        final_test_uncorrupted_verified=True,
        reference_validation_groups=tuple(sorted(reference_validation_groups)),
        reference_validation_sample_count=len(validation_labels),
        model_seed=model_seed,
    )
