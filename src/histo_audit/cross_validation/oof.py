"""Leakage-safe grouped out-of-fold probabilistic predictions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from histo_audit.models.mlp import FrozenEmbeddingMLPConfig


class ProbabilisticEstimator(Protocol):
    """Minimum estimator interface consumed by the generic OOF engine.

    After ``fit``, implementations must expose either ``classes_`` or
    ``class_order``. The engine uses that metadata to map probability columns
    onto the preregistered full class order and rejects missing/extra classes.
    """

    def fit(self, features: Any, labels: Any) -> Any:
        """Fit one independent fold model."""

    def predict_proba(self, features: Any) -> Any:
        """Return holdout probabilities in the estimator's declared class order."""


@dataclass(frozen=True, slots=True)
class OOFFoldEstimatorContext:
    """Immutable inputs supplied to the estimator factory for one fold."""

    fold_id: int
    class_order: tuple[int, ...]
    model_seed: int


class OOFEstimatorFactory(Protocol):
    """Create a fresh probabilistic estimator for one OOF fold."""

    def __call__(self, context: OOFFoldEstimatorContext, /) -> ProbabilisticEstimator:
        """Build the estimator described by ``context``."""


@dataclass(frozen=True, slots=True)
class GroupFold:
    """One group-disjoint train/holdout split."""

    fold_id: int
    train_indices: NDArray[np.int64]
    holdout_indices: NDArray[np.int64]
    training_groups: tuple[str, ...]
    held_out_groups: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GroupFoldPlan:
    """Exact sklearn splitter choice and the resulting group-safe folds."""

    folds: tuple[GroupFold, ...]
    splitter_class_name: str
    splitter_fallback_status: str
    splitter_fallback_reason: str | None


@dataclass(frozen=True, slots=True)
class OOFFoldProvenance:
    """Persistable group and sample provenance for one OOF fold."""

    fold_id: int
    training_groups: tuple[str, ...]
    held_out_groups: tuple[str, ...]
    held_out_sample_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OOFResult:
    """Exactly one out-of-fold probability vector for every audit sample."""

    probabilities: NDArray[np.float64]
    predicted_class: NDArray[np.int64]
    fold_id: NDArray[np.int64]
    coverage_count: NDArray[np.int64]
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    final_reference_groups: tuple[str, ...]
    class_order: tuple[int, ...]
    folds: tuple[OOFFoldProvenance, ...]
    model_name: str
    representation: str
    model_seed: int
    split_seed: int
    fold_assignment_labels: NDArray[np.int64]
    fold_assignment_label_source: str
    fold_assignment_labels_sha256: str
    calibration_status: str = "uncalibrated"
    splitter_class_name: str = field(init=False)
    splitter_fallback_status: str = field(init=False)
    splitter_fallback_reason: str | None = field(init=False)

    def __post_init__(self) -> None:
        """Derive splitter provenance from the immutable split inputs."""

        plan = make_group_stratified_fold_plan(
            self.fold_assignment_labels,
            self.group_ids,
            n_splits=len(self.folds),
            class_order=self.class_order,
            seed=self.split_seed,
        )
        object.__setattr__(self, "splitter_class_name", plan.splitter_class_name)
        object.__setattr__(
            self,
            "splitter_fallback_status",
            plan.splitter_fallback_status,
        )
        object.__setattr__(
            self,
            "splitter_fallback_reason",
            plan.splitter_fallback_reason,
        )

    @property
    def training_groups_by_fold(self) -> dict[int, tuple[str, ...]]:
        return {fold.fold_id: fold.training_groups for fold in self.folds}

    def validate(self) -> None:
        """Raise if coverage, probabilities, class order, or group safety fails."""

        n = len(self.sample_ids)
        if len(set(self.sample_ids)) != n:
            raise ValueError("OOF sample IDs must be unique")
        if self.probabilities.shape != (n, len(self.class_order)):
            raise ValueError("OOF probability shape does not match samples/classes")
        if self.predicted_class.shape != (n,) or self.fold_id.shape != (n,):
            raise ValueError("OOF predictions and fold IDs must align")
        assignment_labels = np.asarray(self.fold_assignment_labels)
        if assignment_labels.shape != (n,) or not np.issubdtype(
            assignment_labels.dtype, np.integer
        ):
            raise ValueError("OOF fold-assignment labels must be an aligned integer vector")
        if self.fold_assignment_label_source not in {
            "observed_label",
            "pre_corruption_label",
        }:
            raise ValueError("invalid OOF fold-assignment label source")
        assignment_classes = set(int(value) for value in assignment_labels)
        unexpected_assignment_classes = sorted(assignment_classes.difference(self.class_order))
        if unexpected_assignment_classes:
            raise ValueError(
                f"OOF fold-assignment label outside class_order: {unexpected_assignment_classes}"
            )
        absent_assignment_classes = sorted(set(self.class_order).difference(assignment_classes))
        if absent_assignment_classes:
            raise ValueError(
                "OOF fold-assignment labels are missing fixed class_order values: "
                f"{absent_assignment_classes}"
            )
        expected_assignment_hash = _fold_assignment_labels_sha256(assignment_labels)
        if self.fold_assignment_labels_sha256 != expected_assignment_hash:
            raise ValueError("OOF fold-assignment label SHA-256 does not match its vector")
        if not np.array_equal(self.coverage_count, np.ones(n, dtype=np.int64)):
            raise ValueError("every audit sample must receive exactly one OOF prediction")
        if not np.isfinite(self.probabilities).all():
            raise ValueError("OOF probabilities contain non-finite values")
        if np.any(self.probabilities < -1e-12) or np.any(self.probabilities > 1 + 1e-12):
            raise ValueError("OOF probabilities lie outside [0, 1]")
        if not np.allclose(self.probabilities.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("OOF probabilities do not sum to one")
        expected_predictions = np.asarray(self.class_order, dtype=np.int64)[
            np.argmax(self.probabilities, axis=1)
        ]
        if not np.array_equal(self.predicted_class, expected_predictions):
            raise ValueError("predicted class does not follow fixed class order")
        seen_holdout_groups: set[str] = set()
        provenance_samples: set[str] = set()
        for fold in self.folds:
            overlap = set(fold.training_groups).intersection(fold.held_out_groups)
            if overlap:
                raise ValueError(f"source-group leakage in fold {fold.fold_id}: {overlap}")
            repeated = seen_holdout_groups.intersection(fold.held_out_groups)
            if repeated:
                raise ValueError(f"groups held out more than once: {repeated}")
            seen_holdout_groups.update(fold.held_out_groups)
            provenance_samples.update(fold.held_out_sample_ids)
        if seen_holdout_groups != set(self.group_ids):
            raise ValueError("OOF fold provenance does not cover every group")
        final_overlap = set(self.group_ids).intersection(self.final_reference_groups)
        if not self.final_reference_groups or final_overlap:
            raise ValueError("OOF result lacks disjoint final-reference group evidence")
        if provenance_samples != set(self.sample_ids):
            raise ValueError("OOF fold provenance does not cover every sample")
        expected_plan = make_group_stratified_fold_plan(
            assignment_labels,
            self.group_ids,
            n_splits=len(self.folds),
            class_order=self.class_order,
            seed=self.split_seed,
        )
        if (
            self.splitter_class_name != expected_plan.splitter_class_name
            or self.splitter_fallback_status != expected_plan.splitter_fallback_status
            or self.splitter_fallback_reason != expected_plan.splitter_fallback_reason
        ):
            raise ValueError("OOF splitter provenance does not match the saved split inputs")
        expected_folds = expected_plan.folds
        if len(self.folds) != len(expected_folds):
            raise ValueError("OOF fold provenance count differs from the recreated splitter")
        expected_fold_ids = np.full(n, -1, dtype=np.int64)
        for recorded_fold, expected_fold in zip(self.folds, expected_folds, strict=True):
            expected_fold_ids[expected_fold.holdout_indices] = expected_fold.fold_id
            expected_held_out_sample_ids = tuple(
                self.sample_ids[index] for index in expected_fold.holdout_indices
            )
            if (
                recorded_fold.fold_id != expected_fold.fold_id
                or recorded_fold.training_groups != expected_fold.training_groups
                or recorded_fold.held_out_groups != expected_fold.held_out_groups
                or recorded_fold.held_out_sample_ids != expected_held_out_sample_ids
            ):
                raise ValueError(
                    "OOF fold provenance does not exactly match the recreated splitter"
                )
        if not np.array_equal(self.fold_id, expected_fold_ids):
            raise ValueError(
                "OOF fold IDs do not match the saved fold-assignment labels and split seed"
            )


def _fold_assignment_labels_sha256(
    labels: Sequence[int] | NDArray[np.integer],
) -> str:
    """Hash an integer label vector through a platform-independent JSON encoding."""

    vector = np.asarray(labels)
    if vector.ndim != 1 or not np.issubdtype(vector.dtype, np.integer):
        raise ValueError("fold-assignment labels must be a one-dimensional integer vector")
    payload = json.dumps(
        [int(value) for value in vector],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_inputs(
    features: NDArray[np.generic],
    labels: Sequence[int] | NDArray[np.integer],
    group_ids: Sequence[str],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.str_]]:
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(group_ids, dtype=np.str_)
    if matrix.ndim != 2 or target.ndim != 1 or groups.ndim != 1:
        raise ValueError("features must be 2-D and labels/groups one-dimensional")
    if matrix.shape[0] != len(target) or len(groups) != len(target) or not len(target):
        raise ValueError("features, labels, and groups must be non-empty and aligned")
    if not np.isfinite(matrix).all():
        raise ValueError("features contain non-finite values")
    if any(not str(group) for group in groups):
        raise ValueError("group IDs must be non-empty")
    return matrix, target, groups


def make_group_stratified_fold_plan(
    labels: Sequence[int] | NDArray[np.integer],
    group_ids: Sequence[str],
    *,
    n_splits: int = 5,
    class_order: Sequence[int] | None = None,
    seed: int = 23,
) -> GroupFoldPlan:
    """Use sklearn stratified-group folds or a documented deterministic fallback."""

    target = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(group_ids, dtype=np.str_)
    if target.ndim != 1 or groups.shape != target.shape or not len(target):
        raise ValueError("labels and group IDs must be aligned vectors")
    unique_groups = np.unique(groups)
    if n_splits < 2 or n_splits > len(unique_groups):
        raise ValueError("n_splits must be between two and the number of unique groups")
    classes = (
        np.asarray(tuple(class_order), dtype=np.int64)
        if class_order is not None
        else np.unique(target)
    )
    if len(classes) < 2 or len(set(classes.tolist())) != len(classes):
        raise ValueError("class_order must contain unique classes")
    allowed_classes = {int(label) for label in classes}
    if any(int(label) not in allowed_classes for label in target):
        raise ValueError("label outside class_order")

    from sklearn.model_selection import (  # type: ignore[import-untyped]
        GroupKFold,
        StratifiedGroupKFold,
    )

    dummy_features = np.zeros((len(target), 1), dtype=np.float64)
    fixed_classes = tuple(int(value) for value in classes)

    def missing_training_classes(
        splits: Sequence[tuple[NDArray[np.integer], NDArray[np.integer]]],
    ) -> tuple[tuple[int, tuple[int, ...]], ...]:
        required = set(fixed_classes)
        missing_by_fold: list[tuple[int, tuple[int, ...]]] = []
        for fold_id, (raw_train, _) in enumerate(splits):
            training_classes = set(int(value) for value in target[np.asarray(raw_train)])
            missing = tuple(sorted(required.difference(training_classes)))
            if missing:
                missing_by_fold.append((fold_id, missing))
        return tuple(missing_by_fold)

    def missing_reason(
        missing_by_fold: Sequence[tuple[int, tuple[int, ...]]],
    ) -> str:
        details = "; ".join(
            f"fold {fold_id} missing {list(missing)}" for fold_id, missing in missing_by_fold
        )
        return "training partition is missing fixed class_order values: " + details

    splitter_class_name = "sklearn.model_selection.StratifiedGroupKFold"
    fallback_status = "not_used"
    fallback_reason: str | None = None
    raw_splits: tuple[tuple[NDArray[np.integer], NDArray[np.integer]], ...] | None = None
    try:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        raw_splits = tuple(splitter.split(dummy_features, target, groups))
    except ValueError:
        fallback_reason = "StratifiedGroupKFold infeasible: split raised ValueError"
    else:
        stratified_missing = missing_training_classes(raw_splits)
        if stratified_missing:
            fallback_reason = "StratifiedGroupKFold infeasible: " + missing_reason(
                stratified_missing
            )

    if fallback_reason is not None:
        splitter_class_name = "sklearn.model_selection.GroupKFold"
        fallback_status = "used"
        splitter = GroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        raw_splits = tuple(splitter.split(dummy_features, target, groups))
        fallback_missing = missing_training_classes(raw_splits)
        if fallback_missing:
            raise ValueError("GroupKFold fallback " + missing_reason(fallback_missing))

    if raw_splits is None:
        raise RuntimeError("group splitter did not produce a fold allocation")

    folds: list[GroupFold] = []
    for fold_id, (raw_train, raw_holdout) in enumerate(raw_splits):
        train = np.asarray(raw_train, dtype=np.int64)
        holdout = np.asarray(raw_holdout, dtype=np.int64)
        held_groups = tuple(sorted(str(value) for value in np.unique(groups[holdout])))
        training_groups = tuple(sorted(str(value) for value in np.unique(groups[train])))
        if not len(holdout) or not len(train):
            raise RuntimeError("group-fold allocation produced an empty partition")
        if set(held_groups).intersection(training_groups):
            raise RuntimeError("internal group-split leakage")
        folds.append(
            GroupFold(
                fold_id=fold_id,
                train_indices=train,
                holdout_indices=holdout,
                training_groups=training_groups,
                held_out_groups=held_groups,
            )
        )
    if len(folds) != n_splits:
        raise RuntimeError("sklearn group splitter returned an unexpected fold count")
    return GroupFoldPlan(
        folds=tuple(folds),
        splitter_class_name=splitter_class_name,
        splitter_fallback_status=fallback_status,
        splitter_fallback_reason=fallback_reason,
    )


def make_group_stratified_folds(
    labels: Sequence[int] | NDArray[np.integer],
    group_ids: Sequence[str],
    *,
    n_splits: int = 5,
    class_order: Sequence[int] | None = None,
    seed: int = 23,
) -> tuple[GroupFold, ...]:
    """Return the exact folds from the provenance-bearing sklearn split plan."""

    return make_group_stratified_fold_plan(
        labels,
        group_ids,
        n_splits=n_splits,
        class_order=class_order,
        seed=seed,
    ).folds


class MultinomialLogisticRegression:
    """Small deterministic L2-regularised multinomial logistic classifier.

    It uses SciPy L-BFGS when available and a deterministic Adam fallback. This
    keeps the synthetic CPU gate functional even before scikit-learn is installed.
    """

    def __init__(
        self,
        *,
        class_order: Sequence[int],
        l2: float = 1.0e-2,
        max_iter: int = 400,
        class_weight_balanced: bool = True,
    ) -> None:
        classes = tuple(int(value) for value in class_order)
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError("class_order must contain at least two unique values")
        if l2 < 0 or max_iter <= 0:
            raise ValueError("l2 must be non-negative and max_iter positive")
        self.class_order = classes
        self.classes_ = np.asarray(classes, dtype=np.int64)
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.class_weight_balanced = class_weight_balanced
        self.mean_: NDArray[np.float64] | None = None
        self.scale_: NDArray[np.float64] | None = None
        self.coef_: NDArray[np.float64] | None = None
        self.converged_: bool = False

    @staticmethod
    def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / exponent.sum(axis=1, keepdims=True)

    def fit(
        self, features: NDArray[np.generic], labels: Sequence[int] | NDArray[np.integer]
    ) -> MultinomialLogisticRegression:
        matrix = np.asarray(features, dtype=np.float64)
        target = np.asarray(labels, dtype=np.int64)
        if matrix.ndim != 2 or target.shape != (matrix.shape[0],) or not len(target):
            raise ValueError("features and labels must be non-empty and aligned")
        if not np.isfinite(matrix).all():
            raise ValueError("features contain non-finite values")
        class_to_column = {value: index for index, value in enumerate(self.class_order)}
        if any(int(value) not in class_to_column for value in target):
            raise ValueError("training label outside class_order")
        columns = np.asarray([class_to_column[int(value)] for value in target], dtype=np.int64)
        self.mean_ = matrix.mean(axis=0)
        self.scale_ = matrix.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        standardised = (matrix - self.mean_) / self.scale_
        design = np.column_stack([standardised, np.ones(len(standardised))])
        one_hot = np.eye(len(self.class_order), dtype=np.float64)[columns]
        if self.class_weight_balanced:
            counts = np.bincount(columns, minlength=len(self.class_order)).astype(np.float64)
            present = counts > 0
            weights_by_class = np.ones(len(self.class_order), dtype=np.float64)
            weights_by_class[present] = len(columns) / (
                max(int(present.sum()), 1) * counts[present]
            )
            sample_weight = weights_by_class[columns]
        else:
            sample_weight = np.ones(len(columns), dtype=np.float64)
        weight_total = float(sample_weight.sum())
        shape = (design.shape[1], len(self.class_order))

        def objective(flat: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            coefficients = flat.reshape(shape)
            probabilities = self._softmax(design @ coefficients)
            log_likelihood = np.log(np.clip(probabilities, 1e-15, 1.0))
            loss = -float(np.sum(sample_weight[:, None] * one_hot * log_likelihood)) / weight_total
            loss += 0.5 * self.l2 * float(np.sum(coefficients[:-1] ** 2))
            residual = (probabilities - one_hot) * sample_weight[:, None] / weight_total
            gradient = design.T @ residual
            gradient[:-1] += self.l2 * coefficients[:-1]
            return loss, gradient.ravel()

        initial = np.zeros(shape, dtype=np.float64).ravel()
        try:
            from scipy.optimize import minimize  # type: ignore[import-untyped]

            optimisation = minimize(
                objective,
                initial,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": self.max_iter, "ftol": 1e-11},
            )
            flat_result = np.asarray(optimisation.x, dtype=np.float64)
            self.converged_ = bool(optimisation.success)
        except ImportError:
            flat_result = initial
            first_moment = np.zeros_like(flat_result)
            second_moment = np.zeros_like(flat_result)
            for iteration in range(1, self.max_iter + 1):
                _, gradient = objective(flat_result)
                first_moment = 0.9 * first_moment + 0.1 * gradient
                second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
                corrected_first = first_moment / (1.0 - 0.9**iteration)
                corrected_second = second_moment / (1.0 - 0.999**iteration)
                flat_result -= 0.03 * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            self.converged_ = True
        self.coef_ = flat_result.reshape(shape)
        return self

    def predict_proba(self, features: NDArray[np.generic]) -> NDArray[np.float64]:
        if self.mean_ is None or self.scale_ is None or self.coef_ is None:
            raise RuntimeError("classifier must be fitted before prediction")
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.mean_):
            raise ValueError("prediction feature shape differs from fitted feature shape")
        design = np.column_stack([(matrix - self.mean_) / self.scale_, np.ones(len(matrix))])
        return self._softmax(design @ self.coef_)


def _estimator_class_order(estimator: ProbabilisticEstimator) -> tuple[int, ...]:
    raw_classes = getattr(estimator, "classes_", None)
    if raw_classes is None:
        raw_classes = getattr(estimator, "class_order", None)
    if raw_classes is None:
        raise ValueError(
            "fitted estimator must expose classes_ or class_order for probability mapping"
        )
    array = np.asarray(raw_classes)
    if array.ndim != 1 or not len(array):
        raise ValueError("fitted estimator class order must be a non-empty vector")
    try:
        classes = tuple(int(value) for value in array.tolist())
    except (TypeError, ValueError) as error:
        raise ValueError("fitted estimator class order must contain integer labels") from error
    if len(set(classes)) != len(classes):
        raise ValueError("fitted estimator class order contains duplicate labels")
    return classes


def _map_fold_probabilities(
    estimator: ProbabilisticEstimator,
    raw_probabilities: Any,
    *,
    class_order: tuple[int, ...],
    holdout_size: int,
) -> NDArray[np.float64]:
    estimator_classes = _estimator_class_order(estimator)
    missing = sorted(set(class_order).difference(estimator_classes))
    unexpected = sorted(set(estimator_classes).difference(class_order))
    if missing or unexpected:
        raise ValueError(
            "fitted estimator classes do not match the fixed full class_order; "
            f"missing={missing}, unexpected={unexpected}"
        )
    probabilities = np.asarray(raw_probabilities, dtype=np.float64)
    expected_shape = (holdout_size, len(estimator_classes))
    if probabilities.shape != expected_shape:
        raise ValueError(
            "estimator probability shape does not match holdout/classes: "
            f"expected {expected_shape}, received {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("estimator probabilities contain non-finite values")
    if np.any(probabilities < -1e-12) or np.any(probabilities > 1.0 + 1e-12):
        raise ValueError("estimator probabilities lie outside [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("estimator probability rows do not sum to one")
    class_to_column = {label: index for index, label in enumerate(estimator_classes)}
    return probabilities[:, [class_to_column[label] for label in class_order]]


def grouped_oof_predict(
    features: NDArray[np.generic],
    observed_labels: Sequence[int] | NDArray[np.integer],
    group_ids: Sequence[str],
    *,
    estimator_factory: OOFEstimatorFactory,
    model_name: str,
    final_reference_group_ids: Collection[str],
    sample_ids: Sequence[str] | None = None,
    fold_assignment_labels: Sequence[int] | NDArray[np.integer] | None = None,
    fold_assignment_label_source: str = "observed_label",
    n_splits: int = 5,
    class_order: Sequence[int] = (0, 1, 2, 3, 4),
    split_seed: int = 23,
    model_seed: int = 29,
    representation: str = "unspecified",
) -> OOFResult:
    """Generate group-safe OOF probabilities with an estimator factory.

    The final-reference group collection is mandatory and must be disjoint from
    every supplied sample. Every fold's training partition must contain the full
    fixed ``class_order``. Estimator probability columns may be returned in a
    different declared order, but missing or extra classes fail closed rather
    than being silently filled.
    """

    matrix, labels, groups = _validate_inputs(features, observed_labels, group_ids)
    identifiers = (
        tuple(str(value) for value in sample_ids)
        if sample_ids is not None
        else tuple(f"sample_{index:08d}" for index in range(len(labels)))
    )
    if len(identifiers) != len(labels) or len(set(identifiers)) != len(identifiers):
        raise ValueError("sample IDs must be aligned and unique")
    final_groups = {str(value) for value in final_reference_group_ids}
    if not final_groups or any(not value for value in final_groups):
        raise ValueError("non-empty final-reference group evidence is mandatory")
    overlap = set(str(value) for value in groups).intersection(final_groups)
    if overlap:
        raise ValueError(f"final-reference groups present in audit pool: {sorted(overlap)}")
    classes = tuple(int(value) for value in class_order)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("class_order must contain at least two unique values")
    observed_classes = set(int(value) for value in labels)
    outside = sorted(observed_classes.difference(classes))
    if outside:
        raise ValueError(f"observed label outside class_order: {outside}")
    absent = sorted(set(classes).difference(observed_classes))
    if absent:
        raise ValueError(f"audit pool is missing fixed class_order values: {absent}")
    if fold_assignment_label_source not in {
        "observed_label",
        "pre_corruption_label",
    }:
        raise ValueError(
            "fold_assignment_label_source must be observed_label or pre_corruption_label"
        )
    if fold_assignment_labels is None:
        if fold_assignment_label_source != "observed_label":
            raise ValueError(
                "pre_corruption_label fold assignment requires explicit fold_assignment_labels"
            )
        assignment_labels = labels.copy()
    else:
        raw_assignment_labels = np.asarray(fold_assignment_labels)
        if raw_assignment_labels.shape != labels.shape or not np.issubdtype(
            raw_assignment_labels.dtype, np.integer
        ):
            raise ValueError(
                "fold_assignment_labels must be an aligned one-dimensional integer vector"
            )
        assignment_labels = raw_assignment_labels.astype(np.int64, copy=True)
        if fold_assignment_label_source == "observed_label" and not np.array_equal(
            assignment_labels, labels
        ):
            raise ValueError(
                "observed_label fold assignments must equal the supplied observed_labels"
            )
    assignment_classes = set(int(value) for value in assignment_labels)
    outside_assignment = sorted(assignment_classes.difference(classes))
    if outside_assignment:
        raise ValueError(f"fold-assignment label outside class_order: {outside_assignment}")
    absent_assignment = sorted(set(classes).difference(assignment_classes))
    if absent_assignment:
        raise ValueError(
            f"fold-assignment labels are missing fixed class_order values: {absent_assignment}"
        )
    assignment_hash = _fold_assignment_labels_sha256(assignment_labels)
    if not model_name.strip():
        raise ValueError("model_name must be non-empty")

    folds = make_group_stratified_folds(
        assignment_labels,
        tuple(str(value) for value in groups),
        n_splits=n_splits,
        class_order=classes,
        seed=split_seed,
    )
    probabilities = np.full((len(labels), len(classes)), np.nan, dtype=np.float64)
    fold_assignment = np.full(len(labels), -1, dtype=np.int64)
    coverage = np.zeros(len(labels), dtype=np.int64)
    provenance: list[OOFFoldProvenance] = []
    required_classes = set(classes)
    for fold in folds:
        training_classes = set(int(value) for value in labels[fold.train_indices])
        missing_training_classes = sorted(required_classes.difference(training_classes))
        if missing_training_classes:
            raise ValueError(
                f"OOF fold {fold.fold_id} training partition is missing fixed "
                f"class_order values: {missing_training_classes}"
            )
        context = OOFFoldEstimatorContext(
            fold_id=fold.fold_id,
            class_order=classes,
            model_seed=model_seed + fold.fold_id,
        )
        estimator = estimator_factory(context)
        if estimator is None:
            raise TypeError(f"estimator factory returned None for OOF fold {fold.fold_id}")
        estimator.fit(matrix[fold.train_indices], labels[fold.train_indices])
        fold_probabilities = _map_fold_probabilities(
            estimator,
            estimator.predict_proba(matrix[fold.holdout_indices]),
            class_order=classes,
            holdout_size=len(fold.holdout_indices),
        )
        probabilities[fold.holdout_indices] = fold_probabilities
        fold_assignment[fold.holdout_indices] = fold.fold_id
        coverage[fold.holdout_indices] += 1
        provenance.append(
            OOFFoldProvenance(
                fold_id=fold.fold_id,
                training_groups=fold.training_groups,
                held_out_groups=fold.held_out_groups,
                held_out_sample_ids=tuple(identifiers[index] for index in fold.holdout_indices),
            )
        )
    predicted = np.asarray(classes, dtype=np.int64)[np.argmax(probabilities, axis=1)]
    result = OOFResult(
        probabilities=probabilities,
        predicted_class=predicted,
        fold_id=fold_assignment,
        coverage_count=coverage,
        sample_ids=identifiers,
        group_ids=tuple(str(value) for value in groups),
        final_reference_groups=tuple(sorted(final_groups)),
        class_order=classes,
        folds=tuple(provenance),
        model_name=model_name,
        representation=representation,
        model_seed=model_seed,
        split_seed=split_seed,
        fold_assignment_labels=assignment_labels,
        fold_assignment_label_source=fold_assignment_label_source,
        fold_assignment_labels_sha256=assignment_hash,
    )
    result.validate()
    return result


def grouped_oof_logistic(
    features: NDArray[np.generic],
    observed_labels: Sequence[int] | NDArray[np.integer],
    group_ids: Sequence[str],
    *,
    final_reference_group_ids: Collection[str],
    sample_ids: Sequence[str] | None = None,
    fold_assignment_labels: Sequence[int] | NDArray[np.integer] | None = None,
    fold_assignment_label_source: str = "observed_label",
    n_splits: int = 5,
    class_order: Sequence[int] = (0, 1, 2, 3, 4),
    split_seed: int = 23,
    model_seed: int = 29,
    representation: str = "unspecified",
    l2: float = 1.0e-2,
    max_iter: int = 400,
) -> OOFResult:
    """Generate one group-safe OOF probability vector per audit-pool sample.

    ``final_reference_group_ids`` is mandatory fail-closed evidence identifying
    the untouched outer partition; an empty collection is rejected.
    """

    classes = tuple(int(value) for value in class_order)

    def estimator_factory(context: OOFFoldEstimatorContext) -> ProbabilisticEstimator:
        del context  # Convex zero-initialised fitting does not consume the recorded fold seed.
        return MultinomialLogisticRegression(
            class_order=classes,
            l2=l2,
            max_iter=max_iter,
            class_weight_balanced=True,
        )

    return grouped_oof_predict(
        features,
        observed_labels,
        group_ids,
        estimator_factory=estimator_factory,
        model_name="multinomial_logistic_regression",
        final_reference_group_ids=final_reference_group_ids,
        sample_ids=sample_ids,
        fold_assignment_labels=fold_assignment_labels,
        fold_assignment_label_source=fold_assignment_label_source,
        n_splits=n_splits,
        class_order=classes,
        split_seed=split_seed,
        model_seed=model_seed,
        representation=representation,
    )


def grouped_oof_frozen_embedding_mlp(
    embeddings: NDArray[np.generic],
    observed_labels: Sequence[int] | NDArray[np.integer],
    group_ids: Sequence[str],
    *,
    final_reference_group_ids: Collection[str],
    sample_ids: Sequence[str] | None = None,
    fold_assignment_labels: Sequence[int] | NDArray[np.integer] | None = None,
    fold_assignment_label_source: str = "observed_label",
    n_splits: int = 5,
    class_order: Sequence[int] = (0, 1, 2, 3, 4),
    split_seed: int = 23,
    model_seed: int = 29,
    representation: str = "frozen_embeddings",
    config: FrozenEmbeddingMLPConfig | None = None,
) -> OOFResult:
    """Run the compact frozen-embedding MLP through the generic OOF engine."""

    from histo_audit.models.mlp import (
        FrozenEmbeddingMLPClassifier,
        FrozenEmbeddingMLPConfig,
    )

    base_config = config or FrozenEmbeddingMLPConfig()

    def estimator_factory(context: OOFFoldEstimatorContext) -> ProbabilisticEstimator:
        return FrozenEmbeddingMLPClassifier(replace(base_config, seed=context.model_seed))

    return grouped_oof_predict(
        embeddings,
        observed_labels,
        group_ids,
        estimator_factory=estimator_factory,
        model_name="frozen_embedding_mlp",
        final_reference_group_ids=final_reference_group_ids,
        sample_ids=sample_ids,
        fold_assignment_labels=fold_assignment_labels,
        fold_assignment_label_source=fold_assignment_label_source,
        n_splits=n_splits,
        class_order=class_order,
        split_seed=split_seed,
        model_seed=model_seed,
        representation=representation,
    )
