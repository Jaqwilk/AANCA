"""Reusable primary-matrix engine and a non-eligible synthetic integration fixture."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
from numpy.typing import NDArray

from histo_audit.auditing.neighbours import fold_safe_neighbour_disagreement
from histo_audit.auditing.scores import cleanlab_scores, fixed_hybrid_score, score_annotations
from histo_audit.config import canonical_config_bytes, config_sha256
from histo_audit.corruption.controlled import (
    CorruptionResult,
    FeatureIndependenceEvidence,
    FeatureSpaceEvidence,
    apply_controlled_corruption,
    array_artifact_sha256,
    canonical_sha256,
    semantic_sha256,
)
from histo_audit.cross_validation import (
    OOFFoldEstimatorContext,
    grouped_oof_logistic,
    grouped_oof_predict,
)
from histo_audit.data.splitting import make_outer_audit_split
from histo_audit.data.synthetic import generate_synthetic_dataset
from histo_audit.evaluation.restoration import (
    DownstreamEstimator,
    DownstreamEstimatorFactory,
    DownstreamEvaluation,
    evaluate_downstream_restoration,
)
from histo_audit.experiment.primary_completion import (
    REAL_PRIMARY_ARTIFACT_SCOPE,
    SYNTHETIC_PRIMARY_ARTIFACT_SCOPE,
    PrimaryMatrixReconciliation,
    build_primary_completion_evidence,
    reconcile_primary_cell_outcomes,
)
from histo_audit.experiment.study_contracts import (
    PRIMARY_AUDIT_METHODS,
    PrimaryCell,
    PrimaryMatrixPlan,
    PrimaryScenario,
    StudyContractError,
    build_primary_matrix_plan,
    validate_frozen_primary_config,
)
from histo_audit.models import FrozenEmbeddingMLPClassifier, FrozenEmbeddingMLPConfig
from histo_audit.statistics.review import (
    average_precision,
    binary_auroc,
    draw_group_bootstrap_indices,
    evaluate_review_budget,
    paired_group_bootstrap,
    random_review_baseline,
    rank_indices,
)
from histo_audit.utils.run_tracking import (
    RunTracker,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    verify_run_integrity,
)


@dataclass(frozen=True, slots=True)
class PrimaryMatrixInputs:
    """Partitioned arrays supplied to the representation-agnostic matrix engine."""

    audit_sample_ids: tuple[str, ...]
    audit_group_ids: tuple[str, ...]
    audit_pre_corruption_labels: NDArray[np.int64]
    audit_features: Mapping[str, NDArray[np.float64]]
    reference_validation_sample_ids: tuple[str, ...]
    reference_validation_group_ids: tuple[str, ...]
    reference_validation_labels: NDArray[np.int64]
    reference_validation_features: Mapping[str, NDArray[np.float64]]
    final_test_sample_ids: tuple[str, ...]
    final_test_group_ids: tuple[str, ...]
    final_test_labels: NDArray[np.int64]
    final_test_features: Mapping[str, NDArray[np.float64]]
    corruption_generator_features: NDArray[np.float64]
    corruption_generator_representation: str
    corruption_auditor_representation: str | None
    independence_evidence: FeatureIndependenceEvidence | None
    dataset_seed: int | None
    class_order: tuple[int, ...] = (0, 1, 2, 3, 4)
    corruption_parameters_by_scenario: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    corruption_timestamp_utc: str | None = None
    audit_grouping_values: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    corruption_auditor_family: str | None = None
    independence_matrix_artifact_sha256: str | None = None
    independence_evidence_by_representation: Mapping[str, FeatureIndependenceEvidence] = field(
        default_factory=dict
    )
    independence_matrix_artifact_sha256_by_representation: Mapping[str, str] = field(
        default_factory=dict
    )
    corruption_auditor_family_by_representation: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """Reject misalignment, leakage, missing classes, and invalid feature caches."""

        partitions = (
            (
                "audit",
                self.audit_sample_ids,
                self.audit_group_ids,
                self.audit_pre_corruption_labels,
                self.audit_features,
            ),
            (
                "reference validation",
                self.reference_validation_sample_ids,
                self.reference_validation_group_ids,
                self.reference_validation_labels,
                self.reference_validation_features,
            ),
            (
                "final test",
                self.final_test_sample_ids,
                self.final_test_group_ids,
                self.final_test_labels,
                self.final_test_features,
            ),
        )
        sample_sets: list[set[str]] = []
        group_sets: list[set[str]] = []
        representation_sets: list[set[str]] = []
        for name, sample_ids, group_ids, labels, features_by_representation in partitions:
            n = len(sample_ids)
            if not n or len(set(sample_ids)) != n or any(not value for value in sample_ids):
                raise ValueError(f"{name} sample IDs must be non-empty and unique")
            if len(group_ids) != n or any(not value for value in group_ids):
                raise ValueError(f"{name} group IDs must align and be non-empty")
            label_values = np.asarray(labels)
            if label_values.shape != (n,) or not np.issubdtype(label_values.dtype, np.integer):
                raise ValueError(f"{name} labels must be an aligned integer vector")
            if any(int(value) not in self.class_order for value in label_values):
                raise ValueError(f"{name} label is absent from class_order")
            representation_sets.append(set(features_by_representation))
            if not features_by_representation:
                raise ValueError(f"{name} feature mapping must be non-empty")
            for representation_id, raw_features in features_by_representation.items():
                features = np.asarray(raw_features)
                if features.ndim != 2 or features.shape[0] != n or not features.shape[1]:
                    raise ValueError(f"{name}/{representation_id} features are misaligned")
                if not np.isfinite(features).all():
                    raise ValueError(f"{name}/{representation_id} features are non-finite")
            sample_sets.append(set(sample_ids))
            group_sets.append(set(group_ids))
        if len(set(self.class_order)) != len(self.class_order) or len(self.class_order) < 2:
            raise ValueError("class_order must contain unique classes")
        for first in range(3):
            for second in range(first + 1, 3):
                if sample_sets[first].intersection(sample_sets[second]):
                    raise ValueError("sample leakage across primary partitions")
                if group_sets[first].intersection(group_sets[second]):
                    raise ValueError("source-group leakage across primary partitions")
        if len({frozenset(values) for values in representation_sets}) != 1:
            raise ValueError("every partition must expose the same representation IDs")
        for representation_id in representation_sets[0]:
            dimensions = {
                np.asarray(features[representation_id]).shape[1]
                for _, _, _, _, features in partitions
            }
            if len(dimensions) != 1:
                raise ValueError(f"feature dimension differs for {representation_id}")
        generator = np.asarray(self.corruption_generator_features)
        if generator.ndim != 2 or generator.shape[0] != len(self.audit_sample_ids):
            raise ValueError("corruption generator features must align with the audit pool")
        if not np.isfinite(generator).all():
            raise ValueError("corruption generator features contain non-finite values")
        for grouping_field, raw_values in self.audit_grouping_values.items():
            values = tuple(str(value) for value in raw_values)
            if not str(grouping_field).strip() or len(values) != len(self.audit_sample_ids):
                raise ValueError(
                    "audit grouping values must be named and align with the audit pool"
                )
            if any(not value for value in values):
                raise ValueError("audit grouping values must be non-empty")
        if self.corruption_auditor_family is not None and not self.corruption_auditor_family:
            raise ValueError("corruption_auditor_family must be non-empty when supplied")
        if self.independence_matrix_artifact_sha256 is not None and (
            len(self.independence_matrix_artifact_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.independence_matrix_artifact_sha256
            )
        ):
            raise ValueError("independence_matrix_artifact_sha256 must be a lowercase SHA-256")
        evidence_keys = set(self.independence_evidence_by_representation)
        artifact_keys = set(self.independence_matrix_artifact_sha256_by_representation)
        family_keys = set(self.corruption_auditor_family_by_representation)
        unknown_evidence = (evidence_keys | artifact_keys | family_keys).difference(
            self.audit_features
        )
        if unknown_evidence:
            raise ValueError(
                "independence metadata names unavailable audit representations: "
                f"{sorted(unknown_evidence)}"
            )
        for representation_id, evidence in self.independence_evidence_by_representation.items():
            evidence.validate()
            if evidence.auditor.representation_name != representation_id:
                raise ValueError(
                    "independence evidence auditor identity must equal its representation key"
                )
            if evidence.auditor.feature_artifact_hash != array_artifact_sha256(
                np.asarray(self.audit_features[representation_id])
            ):
                raise ValueError(f"auditor feature hash mismatch for {representation_id!r}")
            artifact_sha = self.independence_matrix_artifact_sha256_by_representation.get(
                representation_id
            )
            if artifact_sha is not None and (
                len(artifact_sha) != 64
                or any(character not in "0123456789abcdef" for character in artifact_sha)
            ):
                raise ValueError(
                    f"independence matrix artifact SHA is invalid for {representation_id!r}"
                )
            family = self.corruption_auditor_family_by_representation.get(representation_id)
            if family is not None and not family:
                raise ValueError(f"corruption auditor family is empty for {representation_id!r}")
        if self.independence_evidence is not None:
            self.independence_evidence.validate()


@dataclass(frozen=True, slots=True)
class PrimaryPairedComparison:
    """Legacy synthetic-v1 method comparison used only for software validation."""

    comparison_id: str
    method_a: str
    method_b: str

    def validate(self) -> None:
        """Reject ambiguous identifiers and self-comparisons."""

        if not self.comparison_id.strip():
            raise ValueError("paired comparison_id must be non-empty")
        if not self.method_a.strip() or not self.method_b.strip():
            raise ValueError("paired comparison methods must be non-empty")
        if self.method_a == self.method_b:
            raise ValueError("paired comparison methods must differ")


@dataclass(frozen=True, slots=True)
class PrimaryCalibrationControls:
    """Immutable calibration policy, including canonicalized method parameters."""

    enabled: bool
    method: str
    source: str
    reporting: str
    fit_labels_policy: str
    seed: int
    parameters: tuple[tuple[str, str], ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PrimaryCalibrationControls:
        """Freeze arbitrary JSON-compatible parameters without retaining mutable objects."""

        parameters = cast(Mapping[str, Any], value["parameters"])
        frozen_parameters = tuple(
            sorted(
                (
                    str(key),
                    json.dumps(
                        parameter,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                )
                for key, parameter in parameters.items()
            )
        )
        return cls(
            enabled=bool(value["enabled"]),
            method=str(value["method"]),
            source=str(value["source"]),
            reporting=str(value["reporting"]),
            fit_labels_policy=str(value["fit_labels_policy"]),
            seed=int(value["seed"]),
            parameters=frozen_parameters,
        )

    def as_dict(self) -> dict[str, Any]:
        """Restore the exact JSON-compatible configuration shape for evidence."""

        return {
            "enabled": self.enabled,
            "method": self.method,
            "source": self.source,
            "reporting": self.reporting,
            "fit_labels_policy": self.fit_labels_policy,
            "seed": self.seed,
            "parameters": {
                key: json.loads(serialized_value) for key, serialized_value in self.parameters
            },
        }


@dataclass(frozen=True, slots=True)
class PrimaryCellSelector:
    """Exact-one selector: cell ID XOR full scenario/representation/classifier tuple."""

    cell_id: str | None = None
    mechanism: str | None = None
    rate: float | None = None
    seed: int | None = None
    representation_id: str | None = None
    classifier_id: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PrimaryCellSelector:
        """Normalize one already schema-validated selector."""

        if set(value) == {"cell_id"}:
            selector = cls(cell_id=str(value["cell_id"]))
        elif set(value) == {
            "mechanism",
            "rate",
            "seed",
            "representation_id",
            "classifier_id",
        }:
            selector = cls(
                mechanism=str(value["mechanism"]),
                rate=float(value["rate"]),
                seed=int(value["seed"]),
                representation_id=str(value["representation_id"]),
                classifier_id=str(value["classifier_id"]),
            )
        else:
            raise StudyContractError(
                "primary comparison selector must contain exactly cell_id or the exact "
                "scenario/representation/classifier fields"
            )
        selector.validate()
        return selector

    def validate(self) -> None:
        """Reject partial, empty, or mixed selector variants."""

        by_cell = self.cell_id is not None
        tuple_values = (
            self.mechanism,
            self.rate,
            self.seed,
            self.representation_id,
            self.classifier_id,
        )
        by_tuple = any(value is not None for value in tuple_values)
        if by_cell == by_tuple:
            raise ValueError("primary cell selector requires cell_id XOR full matrix coordinates")
        if by_cell:
            if not str(self.cell_id).strip():
                raise ValueError("primary cell selector cell_id must be non-empty")
            return
        if (
            self.mechanism is None
            or self.rate is None
            or self.seed is None
            or self.representation_id is None
            or self.classifier_id is None
            or not self.mechanism.strip()
            or not self.representation_id.strip()
            or not self.classifier_id.strip()
            or not np.isfinite(self.rate)
            or self.rate < 0.0
            or self.rate > 1.0
            or self.seed < 0
        ):
            raise ValueError("primary cell selector requires valid full matrix coordinates")

    def as_dict(self) -> dict[str, Any]:
        """Return the exact frozen selector variant."""

        self.validate()
        if self.cell_id is not None:
            return {"cell_id": self.cell_id}
        if self.rate is None or self.seed is None:
            raise AssertionError("validated full selector lacks numeric coordinates")
        return {
            "mechanism": str(self.mechanism),
            "rate": float(self.rate),
            "seed": int(self.seed),
            "representation_id": str(self.representation_id),
            "classifier_id": str(self.classifier_id),
        }


@dataclass(frozen=True, slots=True)
class PrimaryWithinCellComparison:
    """One exact paired audit-method comparison within a selected matrix cell."""

    comparison_id: str
    selector: PrimaryCellSelector
    method_a: str
    method_b: str
    metric: str
    direction: str
    holm_family: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "selector": self.selector.as_dict(),
            "method_a": self.method_a,
            "method_b": self.method_b,
            "metric": self.metric,
            "direction": self.direction,
            "holm_family": self.holm_family,
        }


@dataclass(frozen=True, slots=True)
class PrimaryCrossCellComparison:
    """One exact paired comparison between two distinct frozen matrix cells."""

    comparison_id: str
    selector_a: PrimaryCellSelector
    selector_b: PrimaryCellSelector
    method_a: str
    method_b: str
    metric: str
    direction: str
    holm_family: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "selector_a": self.selector_a.as_dict(),
            "selector_b": self.selector_b.as_dict(),
            "method_a": self.method_a,
            "method_b": self.method_b,
            "metric": self.metric,
            "direction": self.direction,
            "holm_family": self.holm_family,
        }


@dataclass(frozen=True, slots=True)
class PrimaryMethodVsRandomComparison:
    """One exact configured audit-method versus random-review comparison."""

    comparison_id: str
    selector: PrimaryCellSelector
    method_a: str
    method_b: str
    metric: str
    review_budget: float
    direction: str
    holm_family: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "selector": self.selector.as_dict(),
            "method_a": self.method_a,
            "method_b": self.method_b,
            "metric": self.metric,
            "review_budget": self.review_budget,
            "direction": self.direction,
            "holm_family": self.holm_family,
        }


@dataclass(frozen=True, slots=True)
class PrimaryDownstreamComparison:
    """One exact restoration comparison, kept separate from ranking statistics."""

    comparison_id: str
    method_a: str
    method_b: str
    metric: str
    direction: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PrimaryExecutionControls:
    """Immutable execution controls derived from one valid schema-v2 config.

    The frozen configuration schema and the matrix-plan artifact schema are
    intentionally separate.  The current schema-v2 configuration expands to a
    version-1 matrix artifact.  Both semantic objects, plus every executable
    control below, are covered by ``binding_sha256``.
    """

    frozen_config_schema_version: int
    frozen_config_canonical_json: str | None
    config_semantic_sha256: str
    plan: PrimaryMatrixPlan
    plan_sha256: str
    binding_sha256: str
    dataset_source: str
    class_order: tuple[int, ...]
    statistical_group_unit: str
    development_official_folds: tuple[int, ...]
    final_test_fold: int
    reference_validation_fraction_groups: float
    fold_assignment_label_source: str
    corruption_rounding_policy: str
    n_splits: int
    split_seed: int
    oof_split_kind: str
    no_nucleus_level_fallback: bool
    logistic_l2: float
    logistic_max_iter: int
    logistic_class_weight: str
    logistic_class_weight_label_source: str
    logistic_fit_label_source: str
    logistic_model_seed: int
    mlp_config: FrozenEmbeddingMLPConfig
    mlp_fit_label_source: str
    calibration: PrimaryCalibrationControls
    audit_methods: tuple[str, ...]
    primary_ranking_method: str
    neighbour_k: int
    neighbour_metric: str
    neighbour_exclude_same_group: bool
    fixed_hybrid_components: tuple[str, ...]
    fixed_hybrid_weights: tuple[float, ...]
    cleanlab_failure_policy: str
    primary_metric: str
    primary_review_budget: float
    secondary_review_budgets: tuple[float, ...]
    random_review_repeats: int
    random_review_seed: int
    subgroup_min_samples: int
    subgroup_min_corruptions: int
    paired_method_comparisons: tuple[PrimaryPairedComparison, ...]
    bootstrap_iterations: int
    bootstrap_seed: int
    holm_families: tuple[str, ...]
    exploratory_multiple_comparison_correction: str
    within_cell_comparisons: tuple[PrimaryWithinCellComparison, ...]
    method_vs_random_comparisons: tuple[PrimaryMethodVsRandomComparison, ...]
    cross_cell_comparisons: tuple[PrimaryCrossCellComparison, ...]
    restoration_enabled_cells: tuple[str, ...]
    restoration_cell_ids: tuple[str, ...]
    restoration_ranking_method: str
    restoration_review_budget: float
    restoration_random_repeats: int
    restoration_random_seed: int
    restoration_include_reference_validation_in_training: bool
    restoration_required_experiments: tuple[str, ...]
    restoration_downstream_comparisons: tuple[PrimaryDownstreamComparison, ...]
    confusion_transition_matrix: tuple[tuple[float, ...], ...]
    group_conditional_grouping_field: str
    group_conditional_weights_by_value: tuple[tuple[str, float], ...]
    group_conditional_default_weight: float
    instance_generator_representation: str
    instance_auditor_representation_families: tuple[str, ...]
    instance_independence_matrix_path: str
    instance_independence_matrix_sha256: str

    def _binding_payload(self) -> dict[str, Any]:
        return {
            "frozen_config_schema_version": self.frozen_config_schema_version,
            "frozen_config_canonical_json": self.frozen_config_canonical_json,
            "config_semantic_sha256": self.config_semantic_sha256,
            "plan_config_sha256": self.plan.config_sha256,
            "plan_schema_version": self.plan.schema_version,
            "plan_sha256": self.plan_sha256,
            "dataset_source": self.dataset_source,
            "class_order": self.class_order,
            "statistical_group_unit": self.statistical_group_unit,
            "development_official_folds": self.development_official_folds,
            "final_test_fold": self.final_test_fold,
            "reference_validation_fraction_groups": (self.reference_validation_fraction_groups),
            "fold_assignment_label_source": self.fold_assignment_label_source,
            "corruption_rounding_policy": self.corruption_rounding_policy,
            "n_splits": self.n_splits,
            "split_seed": self.split_seed,
            "oof_split_kind": self.oof_split_kind,
            "no_nucleus_level_fallback": self.no_nucleus_level_fallback,
            "logistic_l2": self.logistic_l2,
            "logistic_max_iter": self.logistic_max_iter,
            "logistic_class_weight": self.logistic_class_weight,
            "logistic_class_weight_label_source": self.logistic_class_weight_label_source,
            "logistic_fit_label_source": self.logistic_fit_label_source,
            "logistic_model_seed": self.logistic_model_seed,
            "mlp_config": asdict(self.mlp_config),
            "mlp_fit_label_source": self.mlp_fit_label_source,
            "calibration": self.calibration.as_dict(),
            "audit_methods": self.audit_methods,
            "primary_ranking_method": self.primary_ranking_method,
            "neighbour_k": self.neighbour_k,
            "neighbour_metric": self.neighbour_metric,
            "neighbour_exclude_same_group": self.neighbour_exclude_same_group,
            "fixed_hybrid_components": self.fixed_hybrid_components,
            "fixed_hybrid_weights": self.fixed_hybrid_weights,
            "cleanlab_failure_policy": self.cleanlab_failure_policy,
            "primary_metric": self.primary_metric,
            "primary_review_budget": self.primary_review_budget,
            "secondary_review_budgets": self.secondary_review_budgets,
            "random_review_repeats": self.random_review_repeats,
            "random_review_seed": self.random_review_seed,
            "subgroup_min_samples": self.subgroup_min_samples,
            "subgroup_min_corruptions": self.subgroup_min_corruptions,
            "paired_method_comparisons": [
                asdict(comparison) for comparison in self.paired_method_comparisons
            ],
            "bootstrap_iterations": self.bootstrap_iterations,
            "bootstrap_seed": self.bootstrap_seed,
            "holm_families": self.holm_families,
            "exploratory_multiple_comparison_correction": (
                self.exploratory_multiple_comparison_correction
            ),
            "within_cell_comparisons": [
                comparison.as_dict() for comparison in self.within_cell_comparisons
            ],
            "method_vs_random_comparisons": [
                comparison.as_dict() for comparison in self.method_vs_random_comparisons
            ],
            "cross_cell_comparisons": [
                comparison.as_dict() for comparison in self.cross_cell_comparisons
            ],
            "restoration_enabled_cells": self.restoration_enabled_cells,
            "restoration_cell_ids": self.restoration_cell_ids,
            "restoration_ranking_method": self.restoration_ranking_method,
            "restoration_review_budget": self.restoration_review_budget,
            "restoration_random_repeats": self.restoration_random_repeats,
            "restoration_random_seed": self.restoration_random_seed,
            "restoration_include_reference_validation_in_training": (
                self.restoration_include_reference_validation_in_training
            ),
            "restoration_required_experiments": self.restoration_required_experiments,
            "restoration_downstream_comparisons": [
                comparison.as_dict() for comparison in self.restoration_downstream_comparisons
            ],
            "confusion_transition_matrix": self.confusion_transition_matrix,
            "group_conditional_grouping_field": self.group_conditional_grouping_field,
            "group_conditional_weights_by_value": self.group_conditional_weights_by_value,
            "group_conditional_default_weight": self.group_conditional_default_weight,
            "instance_generator_representation": self.instance_generator_representation,
            "instance_auditor_representation_families": (
                self.instance_auditor_representation_families
            ),
            "instance_independence_matrix_path": self.instance_independence_matrix_path,
            "instance_independence_matrix_sha256": self.instance_independence_matrix_sha256,
        }

    def validate_for_plan(self, plan: PrimaryMatrixPlan) -> None:
        """Verify the semantic-config, plan, and execution-control bindings."""

        if self.frozen_config_schema_version != 2:
            raise ValueError("frozen primary execution controls require a schema-v2 config")
        if self.plan != plan:
            raise ValueError("execution controls were derived for a different primary plan")
        if self.config_semantic_sha256 != plan.config_sha256:
            raise ValueError("frozen config semantic SHA does not match the primary plan")
        if self.plan_sha256 != canonical_sha256(plan.as_dict()):
            raise ValueError("primary matrix plan SHA is invalid")
        expected_binding = canonical_sha256(self._binding_payload())
        if self.binding_sha256 != expected_binding:
            raise ValueError("primary execution controls binding SHA is invalid")
        if self.frozen_config_canonical_json is None:
            raise ValueError("schema-v2 controls lack their canonical frozen configuration")
        try:
            frozen_config = json.loads(self.frozen_config_canonical_json)
        except json.JSONDecodeError as exc:
            raise ValueError("canonical frozen primary configuration is invalid JSON") from exc
        if not isinstance(frozen_config, dict):
            raise ValueError("canonical frozen primary configuration must be a JSON object")
        if config_sha256(frozen_config) != self.config_semantic_sha256:
            raise ValueError("canonical frozen primary configuration SHA is invalid")
        expected_controls = _primary_execution_controls_from_frozen_config(
            frozen_config,
            validate_controls=False,
        )
        if self != expected_controls:
            raise ValueError(
                "primary execution controls are not the exact derivation of the frozen config"
            )
        _validate_runtime_controls(self)

    def as_dict(self) -> dict[str, Any]:
        """Return machine-readable frozen-control evidence."""

        return {
            "schema_version": 1,
            "source": (
                "validated_frozen_primary_config_schema_v2"
                if self.frozen_config_schema_version == 2
                else "sealed_synthetic_primary_fixture_v1"
            ),
            "binding_sha256": self.binding_sha256,
            **self._binding_payload(),
        }


@dataclass(frozen=True, slots=True)
class PrimaryMatrixArtifacts:
    """Saved evidence produced by one complete matrix execution."""

    output_directory: Path
    matrix_plan_path: Path
    execution_controls_path: Path
    cell_index_path: Path
    reconciliation_path: Path
    completion_evidence_path: Path
    restoration_path: Path
    report_path: Path
    outcomes: tuple[Mapping[str, Any], ...]
    reconciliation: PrimaryMatrixReconciliation


@dataclass(frozen=True, slots=True)
class SyntheticPrimaryFixtureResult:
    """Sealed synthetic integration result that can never enable a research stage."""

    run_id: str
    run_directory: Path
    metrics_path: Path
    report_path: Path
    completion_evidence_path: Path
    reconciliation_path: Path
    matrix_cell_count: int


def _atomic_npz(path: Path, arrays: Mapping[str, NDArray[np.generic]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return path


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> Path:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, stream.getvalue())
    return path


def _jsonable_review(result: Any) -> dict[str, Any]:
    payload = asdict(result)
    payload.pop("reviewed_indices", None)
    return payload


def _artifact_manifest(directory: Path, names: Sequence[str]) -> Path:
    records = [
        {
            "path": name,
            "size_bytes": (directory / name).stat().st_size,
            "sha256": sha256_file(directory / name),
        }
        for name in names
    ]
    return atomic_write_json(
        directory / "artifact_manifest.json",
        {"schema_version": 1, "artifacts": records},
    )


def _flatten_draws(
    draws: Sequence[NDArray[np.int64]],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    offsets = np.zeros(len(draws) + 1, dtype=np.int64)
    for index, draw in enumerate(draws):
        offsets[index + 1] = offsets[index] + len(draw)
    flattened = (
        np.concatenate(tuple(np.asarray(draw, dtype=np.int64) for draw in draws))
        if draws
        else np.empty(0, dtype=np.int64)
    )
    return flattened, offsets


def _parse_frozen_comparison(
    value: str,
    *,
    configured_methods: Sequence[str],
) -> PrimaryPairedComparison:
    """Parse a frozen ``method_a_vs_method_b`` comparison without fuzzy matching."""

    methods = tuple(str(method) for method in configured_methods)
    aliases = {method: method for method in methods}
    if "fixed_hybrid" in methods:
        aliases["hybrid"] = "fixed_hybrid"
    matches = [
        (alias_a, alias_b, method_a, method_b)
        for alias_a, method_a in aliases.items()
        for alias_b, method_b in aliases.items()
        if value == f"{alias_a}_vs_{alias_b}"
    ]
    if len(matches) != 1:
        raise StudyContractError(
            "statistics.preregistered_comparisons entries must be exact "
            "'<configured_method>_vs_<configured_method>' values"
        )
    _, _, method_a, method_b = matches[0]
    comparison = PrimaryPairedComparison(value, method_a, method_b)
    try:
        comparison.validate()
    except ValueError as exc:
        raise StudyContractError(str(exc)) from exc
    return comparison


def _parse_within_cell_comparison(value: Mapping[str, Any]) -> PrimaryWithinCellComparison:
    return PrimaryWithinCellComparison(
        comparison_id=str(value["comparison_id"]),
        selector=PrimaryCellSelector.from_mapping(cast(Mapping[str, Any], value["selector"])),
        method_a=str(value["method_a"]),
        method_b=str(value["method_b"]),
        metric=str(value["metric"]),
        direction=str(value["direction"]),
        holm_family=str(value["holm_family"]),
    )


def _parse_method_vs_random_comparison(
    value: Mapping[str, Any],
) -> PrimaryMethodVsRandomComparison:
    return PrimaryMethodVsRandomComparison(
        comparison_id=str(value["comparison_id"]),
        selector=PrimaryCellSelector.from_mapping(cast(Mapping[str, Any], value["selector"])),
        method_a=str(value["method_a"]),
        method_b=str(value["method_b"]),
        metric=str(value["metric"]),
        review_budget=float(value["review_budget"]),
        direction=str(value["direction"]),
        holm_family=str(value["holm_family"]),
    )


def _parse_cross_cell_comparison(value: Mapping[str, Any]) -> PrimaryCrossCellComparison:
    return PrimaryCrossCellComparison(
        comparison_id=str(value["comparison_id"]),
        selector_a=PrimaryCellSelector.from_mapping(cast(Mapping[str, Any], value["selector_a"])),
        selector_b=PrimaryCellSelector.from_mapping(cast(Mapping[str, Any], value["selector_b"])),
        method_a=str(value["method_a"]),
        method_b=str(value["method_b"]),
        metric=str(value["metric"]),
        direction=str(value["direction"]),
        holm_family=str(value["holm_family"]),
    )


def _comparison_selector_cell(
    plan: PrimaryMatrixPlan,
    selector: PrimaryCellSelector,
) -> PrimaryCell:
    """Resolve a frozen exact selector to exactly one matrix cell."""

    selector.validate()
    if selector.cell_id is not None:
        matches = tuple(cell for cell in plan.cells if cell.cell_id == selector.cell_id)
    else:
        matches = tuple(
            cell
            for cell in plan.cells
            if cell.mechanism == selector.mechanism
            and cell.rate == selector.rate
            and cell.corruption_seed == selector.seed
            and cell.representation_id == selector.representation_id
            and cell.classifier_id == selector.classifier_id
        )
    if len(matches) != 1:
        raise ValueError("primary comparison selector must resolve exactly one frozen matrix cell")
    return matches[0]


def _restoration_selector_matches(
    plan: PrimaryMatrixPlan,
    selector: str,
) -> tuple[str, ...]:
    """Resolve one exact frozen selector without substring or fuzzy matching."""

    aliases = {
        "multinomial_logistic_regression": "logistic",
        "small_mlp": "mlp",
    }
    matches: list[str] = []
    for cell in plan.cells:
        exact_values = {
            cell.cell_id,
            f"{cell.representation_id}_{cell.classifier_id}",
            f"{cell.scenario_id}_{cell.representation_id}_{cell.classifier_id}",
        }
        classifier_alias = aliases.get(cell.classifier_id)
        if classifier_alias is not None:
            exact_values.update(
                {
                    f"{cell.representation_id}_{classifier_alias}",
                    f"{cell.scenario_id}_{cell.representation_id}_{classifier_alias}",
                }
            )
        if selector in exact_values:
            matches.append(cell.cell_id)
    if not matches:
        raise StudyContractError(
            f"restoration.enabled_cells selector {selector!r} matches no frozen matrix cell"
        )
    return tuple(matches)


def _resolve_restoration_cell_ids(
    plan: PrimaryMatrixPlan,
    selectors: Sequence[str],
) -> tuple[str, ...]:
    """Expand every frozen restoration selector to deterministic exact cell IDs."""

    resolved: list[str] = []
    owner: dict[str, str] = {}
    for raw_selector in selectors:
        selector = str(raw_selector)
        for cell_id in _restoration_selector_matches(plan, selector):
            previous = owner.get(cell_id)
            if previous is not None:
                raise StudyContractError(
                    "restoration selectors overlap for cell "
                    f"{cell_id}: {previous!r} and {selector!r}"
                )
            owner[cell_id] = selector
            resolved.append(cell_id)
    if not resolved:
        raise StudyContractError("at least one frozen restoration cell must resolve")
    plan_order = {cell.cell_id: index for index, cell in enumerate(plan.cells)}
    return tuple(sorted(resolved, key=plan_order.__getitem__))


def _primary_execution_controls_from_frozen_config(
    config: Mapping[str, Any],
    *,
    validate_controls: bool,
) -> PrimaryExecutionControls:
    """Bind every executable primary control to one validated schema-v2 config.

    No caller-supplied fallback is consulted.  The returned object contains the
    deterministically expanded plan, both classifiers' full hyperparameters, OOF
    controls, audit/ranking choices, every review budget, paired comparisons, and
    restoration controls.
    """

    resolved = validate_frozen_primary_config(config)
    plan = build_primary_matrix_plan(resolved)
    semantic_hash = config_sha256(resolved)
    if semantic_hash != plan.config_sha256:
        raise RuntimeError("primary config semantic SHA does not match the expanded plan")

    data = cast(Mapping[str, Any], resolved["data"])
    classifiers = cast(Mapping[str, Mapping[str, Any]], resolved["classifiers"])
    logistic = classifiers["multinomial_logistic_regression"]
    mlp = classifiers["small_mlp"]
    calibration = cast(Mapping[str, Any], resolved["calibration"])
    oof = cast(Mapping[str, Any], resolved["oof"])
    audit = cast(Mapping[str, Any], resolved["audit"])
    neighbour = cast(Mapping[str, Any], audit["nearest_neighbour"])
    hybrid = cast(Mapping[str, Any], audit["fixed_hybrid"])
    evaluation = cast(Mapping[str, Any], resolved["evaluation"])
    statistics = cast(Mapping[str, Any], resolved["statistics"])
    restoration = cast(Mapping[str, Any], resolved["restoration"])
    corruption = cast(Mapping[str, Any], resolved["corruption"])
    mechanisms = cast(Mapping[str, Mapping[str, Any]], corruption["mechanisms"])
    confusion = mechanisms["confusion_targeted_corruption"]
    group_conditional = mechanisms["group_conditional_corruption"]
    instance = mechanisms["instance_dependent_corruption"]
    methods = tuple(str(value) for value in cast(Sequence[Any], audit["methods"]))
    within_cell_comparisons = tuple(
        _parse_within_cell_comparison(cast(Mapping[str, Any], value))
        for value in cast(Sequence[Any], statistics["within_cell_comparisons"])
    )
    method_vs_random_comparisons = tuple(
        _parse_method_vs_random_comparison(cast(Mapping[str, Any], value))
        for value in cast(Sequence[Any], statistics["method_vs_random_comparisons"])
    )
    cross_cell_comparisons = tuple(
        _parse_cross_cell_comparison(cast(Mapping[str, Any], value))
        for value in cast(Sequence[Any], statistics["cross_cell_comparisons"])
    )
    amp_dtype = str(mlp["amp_dtype"])
    mlp_config = FrozenEmbeddingMLPConfig(
        hidden_dimensions=tuple(int(value) for value in mlp["hidden_dimensions"]),
        dropout=float(mlp["dropout"]),
        epochs=int(mlp["epochs"]),
        batch_size=int(mlp["batch_size"]),
        learning_rate=float(mlp["learning_rate"]),
        weight_decay=float(mlp["weight_decay"]),
        class_weight="balanced",
        seed=int(mlp["model_seed"]),
        device="cuda" if bool(mlp["amp"]) else "cpu",
        amp=bool(mlp["amp"]),
        amp_dtype=cast(Any, amp_dtype),
        gradient_accumulation_steps=int(mlp["gradient_accumulation_steps"]),
        minimum_batch_size=int(mlp["minimum_batch_size"]),
        early_stopping_patience=(
            None if mlp["early_stopping_patience"] is None else int(mlp["early_stopping_patience"])
        ),
        early_stopping_min_delta=float(mlp["early_stopping_min_delta"]),
    )
    mlp_config.validate()
    restoration_enabled_cells = tuple(
        str(value) for value in cast(Sequence[Any], restoration["enabled_cells"])
    )
    controls = PrimaryExecutionControls(
        frozen_config_schema_version=2,
        frozen_config_canonical_json=canonical_config_bytes(resolved).decode("utf-8"),
        config_semantic_sha256=semantic_hash,
        plan=plan,
        plan_sha256=canonical_sha256(plan.as_dict()),
        binding_sha256="",
        dataset_source=str(data["source"]),
        class_order=tuple(int(value) for value in cast(Sequence[Any], data["class_order"])),
        statistical_group_unit=str(data["group_unit"]),
        development_official_folds=tuple(
            int(value) for value in cast(Sequence[Any], data["development_official_folds"])
        ),
        final_test_fold=int(data["final_test_fold"]),
        reference_validation_fraction_groups=float(data["reference_validation_fraction_groups"]),
        fold_assignment_label_source=str(data["fold_assignment_labels"]),
        corruption_rounding_policy=str(corruption["rounding_policy"]),
        n_splits=int(oof["n_splits"]),
        split_seed=int(data["split_seed"]),
        oof_split_kind=str(oof["split_kind"]),
        no_nucleus_level_fallback=bool(oof["no_nucleus_level_fallback"]),
        logistic_l2=float(logistic["l2"]),
        logistic_max_iter=int(logistic["max_iter"]),
        logistic_class_weight=str(logistic["class_weight"]),
        logistic_class_weight_label_source=str(logistic["class_weight_label_source"]),
        logistic_fit_label_source=str(logistic["fit_label_source"]),
        logistic_model_seed=int(logistic["model_seed"]),
        mlp_config=mlp_config,
        mlp_fit_label_source=str(mlp["fit_label_source"]),
        calibration=PrimaryCalibrationControls.from_mapping(calibration),
        audit_methods=methods,
        primary_ranking_method=str(audit["primary_method"]),
        neighbour_k=int(neighbour["k"]),
        neighbour_metric=str(neighbour["metric"]),
        neighbour_exclude_same_group=bool(neighbour["exclude_same_group"]),
        fixed_hybrid_components=tuple(str(value) for value in hybrid["components"]),
        fixed_hybrid_weights=tuple(float(value) for value in hybrid["weights"]),
        cleanlab_failure_policy=str(audit["cleanlab_failure_policy"]),
        primary_metric=str(evaluation["primary_metric"]),
        primary_review_budget=float(evaluation["primary_review_budget"]),
        secondary_review_budgets=tuple(
            float(value) for value in evaluation["secondary_review_budgets"]
        ),
        random_review_repeats=int(evaluation["random_review_repeats"]),
        random_review_seed=int(evaluation["random_review_seed"]),
        subgroup_min_samples=int(evaluation["subgroup_min_samples"]),
        subgroup_min_corruptions=int(evaluation["subgroup_min_corruptions"]),
        paired_method_comparisons=(),
        bootstrap_iterations=int(statistics["paired_group_bootstrap_iterations"]),
        bootstrap_seed=int(statistics["bootstrap_seed"]),
        holm_families=tuple(str(value) for value in statistics["holm_families"]),
        exploratory_multiple_comparison_correction=str(
            statistics["exploratory_multiple_comparison_correction"]
        ),
        within_cell_comparisons=within_cell_comparisons,
        method_vs_random_comparisons=method_vs_random_comparisons,
        cross_cell_comparisons=cross_cell_comparisons,
        restoration_enabled_cells=restoration_enabled_cells,
        restoration_cell_ids=_resolve_restoration_cell_ids(plan, restoration_enabled_cells),
        restoration_ranking_method=str(restoration["ranking_method"]),
        restoration_review_budget=float(restoration["review_budget"]),
        restoration_random_repeats=int(restoration["random_repeats"]),
        restoration_random_seed=int(restoration["random_seed"]),
        restoration_include_reference_validation_in_training=bool(
            restoration["include_reference_validation_in_training"]
        ),
        restoration_required_experiments=tuple(
            str(value) for value in restoration["required_experiments"]
        ),
        restoration_downstream_comparisons=tuple(
            PrimaryDownstreamComparison(
                comparison_id=str(value["comparison_id"]),
                method_a=str(value["method_a"]),
                method_b=str(value["method_b"]),
                metric=str(value["metric"]),
                direction=str(value["direction"]),
            )
            for value in cast(Sequence[Mapping[str, Any]], restoration["downstream_comparisons"])
        ),
        confusion_transition_matrix=tuple(
            tuple(float(value) for value in row)
            for row in cast(Sequence[Sequence[Any]], confusion["transition_matrix"])
        ),
        group_conditional_grouping_field=str(group_conditional["grouping_field"]),
        group_conditional_weights_by_value=tuple(
            sorted(
                (str(key), float(value))
                for key, value in cast(
                    Mapping[str, Any], group_conditional["weights_by_value"]
                ).items()
            )
        ),
        group_conditional_default_weight=float(group_conditional["default_weight"]),
        instance_generator_representation=str(instance["generator_representation"]),
        instance_auditor_representation_families=tuple(
            str(value) for value in cast(Sequence[Any], instance["auditor_representation_families"])
        ),
        instance_independence_matrix_path=str(instance["independence_matrix_path"]),
        instance_independence_matrix_sha256=str(instance["independence_matrix_sha256"]),
    )
    controls = replace(
        controls,
        binding_sha256=canonical_sha256(controls._binding_payload()),
    )
    if validate_controls:
        controls.validate_for_plan(plan)
    return controls


def primary_execution_controls_from_frozen_config(
    config: Mapping[str, Any],
) -> PrimaryExecutionControls:
    """Return exact immutable controls re-verifiable against the embedded frozen config."""

    return _primary_execution_controls_from_frozen_config(config, validate_controls=True)


def _validate_execution_configuration(
    *,
    audit_methods: Sequence[str],
    primary_ranking_method: str,
    restoration_ranking_method: str,
    fixed_hybrid_components: Sequence[str],
    fixed_hybrid_weights: Sequence[float],
    primary_review_budget: float,
    secondary_review_budgets: Sequence[float],
    paired_method_comparisons: Sequence[PrimaryPairedComparison],
) -> tuple[float, ...]:
    """Validate direct/synthetic controls before writing any matrix artifacts."""

    methods = tuple(str(method) for method in audit_methods)
    if len(methods) != len(set(methods)) or not methods:
        raise ValueError("audit_methods must be non-empty and unique")
    unknown_methods = set(methods).difference(PRIMARY_AUDIT_METHODS)
    if unknown_methods:
        raise ValueError(f"unknown configured audit methods: {sorted(unknown_methods)}")
    if "fixed_hybrid" not in methods:
        raise ValueError("audit_methods must include fixed_hybrid")
    components = tuple(str(method) for method in fixed_hybrid_components)
    weights = tuple(float(weight) for weight in fixed_hybrid_weights)
    if len(components) < 2 or len(components) != len(weights):
        raise ValueError("fixed hybrid components and weights must align")
    if len(set(components)) != len(components) or "fixed_hybrid" in components:
        raise ValueError("fixed hybrid components must be unique non-hybrid methods")
    if not set(components).issubset(methods):
        raise ValueError("fixed hybrid components must name configured audit methods")
    if any(not np.isfinite(weight) or weight <= 0.0 for weight in weights):
        raise ValueError("fixed hybrid weights must be finite and positive")
    if not np.isclose(sum(weights), 1.0, atol=1e-9):
        raise ValueError("fixed hybrid weights must sum to one")
    for name, value in (
        ("primary_ranking_method", primary_ranking_method),
        ("restoration_ranking_method", restoration_ranking_method),
    ):
        if value not in methods:
            raise ValueError(f"{name} must name a configured audit method")
    budgets = (
        float(primary_review_budget),
        *(float(value) for value in secondary_review_budgets),
    )
    if len(budgets) != len(set(budgets)):
        raise ValueError("primary and secondary review budgets must be unique")
    if any(not np.isfinite(value) or value <= 0.0 or value > 1.0 for value in budgets):
        raise ValueError("review budgets must be finite fractions in (0, 1]")
    comparisons = tuple(paired_method_comparisons)
    comparison_ids: list[str] = []
    for comparison in comparisons:
        comparison.validate()
        comparison_ids.append(comparison.comparison_id)
        if comparison.method_a not in methods or comparison.method_b not in methods:
            raise ValueError(
                f"paired comparison {comparison.comparison_id!r} names an unconfigured method"
            )
    if len(comparison_ids) != len(set(comparison_ids)):
        raise ValueError("paired comparison IDs must be unique")
    return budgets


def _validate_frozen_primary_comparisons(
    controls: PrimaryExecutionControls,
    budgets: Sequence[float],
) -> None:
    """Reconcile every exact comparison definition without generating additional pairs."""

    methods = set(controls.audit_methods)
    allowed_metrics = {
        "average_precision",
        "auroc",
        "precision_at_budget",
        "recall_at_budget",
        "lift_at_budget",
    }
    if (
        not controls.holm_families
        or len(set(controls.holm_families)) != len(controls.holm_families)
        or any(not value.strip() for value in controls.holm_families)
    ):
        raise ValueError("primary Holm families must be non-empty and unique")
    if controls.exploratory_multiple_comparison_correction != "holm":
        raise ValueError("exploratory primary comparisons must use Holm correction")
    if (
        not controls.within_cell_comparisons
        or not controls.method_vs_random_comparisons
        or not controls.cross_cell_comparisons
    ):
        raise ValueError("all three exact primary comparison families must be non-empty")
    all_comparisons: tuple[
        PrimaryWithinCellComparison | PrimaryMethodVsRandomComparison | PrimaryCrossCellComparison,
        ...,
    ] = (
        *controls.within_cell_comparisons,
        *controls.method_vs_random_comparisons,
        *controls.cross_cell_comparisons,
    )
    if not all_comparisons:
        raise ValueError("real primary controls require exact frozen comparisons")
    comparison_ids = tuple(comparison.comparison_id for comparison in all_comparisons)
    if len(set(comparison_ids)) != len(comparison_ids) or any(
        not value.strip() for value in comparison_ids
    ):
        raise ValueError("primary comparison IDs must be globally non-empty and unique")
    referenced_families: set[str] = set()
    for comparison in all_comparisons:
        if comparison.metric not in allowed_metrics:
            raise ValueError(
                f"primary comparison {comparison.comparison_id!r} has an invalid metric"
            )
        if comparison.direction != "method_a_minus_method_b":
            raise ValueError(
                f"primary comparison {comparison.comparison_id!r} has an invalid direction"
            )
        if comparison.holm_family not in controls.holm_families:
            raise ValueError(
                f"primary comparison {comparison.comparison_id!r} has an unknown Holm family"
            )
        referenced_families.add(comparison.holm_family)
    if referenced_families != set(controls.holm_families):
        raise ValueError("every frozen primary Holm family must be referenced")
    for comparison in controls.within_cell_comparisons:
        _comparison_selector_cell(controls.plan, comparison.selector)
        if comparison.metric.endswith("_at_budget"):
            raise ValueError(
                f"within-cell comparison {comparison.comparison_id!r} lacks a review budget"
            )
        if (
            comparison.method_a not in methods
            or comparison.method_b not in methods
            or comparison.method_a == comparison.method_b
        ):
            raise ValueError(
                f"within-cell comparison {comparison.comparison_id!r} has invalid methods"
            )
    for comparison in controls.method_vs_random_comparisons:
        _comparison_selector_cell(controls.plan, comparison.selector)
        if comparison.method_a not in methods or comparison.method_b != "random_review":
            raise ValueError(
                f"method-vs-random comparison {comparison.comparison_id!r} has invalid methods"
            )
        if comparison.review_budget not in budgets:
            raise ValueError(
                f"method-vs-random comparison {comparison.comparison_id!r} has an unfrozen budget"
            )
    for comparison in controls.cross_cell_comparisons:
        cell_a = _comparison_selector_cell(controls.plan, comparison.selector_a)
        cell_b = _comparison_selector_cell(controls.plan, comparison.selector_b)
        if cell_a.cell_id == cell_b.cell_id:
            raise ValueError(
                f"cross-cell comparison {comparison.comparison_id!r} selects one cell twice"
            )
        if cell_a.scenario_id != cell_b.scenario_id:
            if comparison.method_a != comparison.method_b:
                raise ValueError(
                    f"controlled cross-scenario comparison {comparison.comparison_id!r} must "
                    "hold the audit method fixed"
                )
            if (
                cell_a.corruption_seed != cell_b.corruption_seed
                or cell_a.representation_id != cell_b.representation_id
                or cell_a.classifier_id != cell_b.classifier_id
            ):
                raise ValueError(
                    f"controlled cross-scenario comparison {comparison.comparison_id!r} must "
                    "hold seed, representation, and classifier fixed"
                )
            if cell_a.rate <= 0.0 or cell_b.rate <= 0.0:
                raise ValueError(
                    f"controlled cross-scenario comparison {comparison.comparison_id!r} cannot "
                    "use the clean-reference scenario"
                )
            changed_corruption_factors = sum(
                (
                    cell_a.mechanism != cell_b.mechanism,
                    cell_a.rate != cell_b.rate,
                )
            )
            if changed_corruption_factors != 1:
                raise ValueError(
                    f"controlled cross-scenario comparison {comparison.comparison_id!r} must "
                    "vary exactly one of mechanism or rate"
                )
        if comparison.metric.endswith("_at_budget"):
            raise ValueError(
                f"cross-cell comparison {comparison.comparison_id!r} lacks a review budget"
            )
        if comparison.method_a not in methods or comparison.method_b not in methods:
            raise ValueError(
                f"cross-cell comparison {comparison.comparison_id!r} has invalid methods"
            )


def _validate_runtime_controls(controls: PrimaryExecutionControls) -> tuple[float, ...]:
    """Recheck every executable invariant, including after dataclass tampering."""

    budgets = _validate_execution_configuration(
        audit_methods=controls.audit_methods,
        primary_ranking_method=controls.primary_ranking_method,
        restoration_ranking_method=controls.restoration_ranking_method,
        fixed_hybrid_components=controls.fixed_hybrid_components,
        fixed_hybrid_weights=controls.fixed_hybrid_weights,
        primary_review_budget=controls.primary_review_budget,
        secondary_review_budgets=controls.secondary_review_budgets,
        paired_method_comparisons=controls.paired_method_comparisons,
    )
    if controls.dataset_source.casefold() != "pannuke":
        raise ValueError("real primary controls require the PanNuke dataset source")
    if controls.class_order != (0, 1, 2, 3, 4):
        raise ValueError("primary execution class_order must be exactly (0, 1, 2, 3, 4)")
    if controls.statistical_group_unit in {
        "",
        "strongest_available",
        "strongest_verified_or_source_patch",
    }:
        raise ValueError("primary statistical group unit must be exact")
    if (
        len(controls.development_official_folds) != 2
        or len(set(controls.development_official_folds)) != 2
        or controls.final_test_fold in controls.development_official_folds
    ):
        raise ValueError("primary official-fold partition controls are invalid")
    if (
        not np.isfinite(controls.reference_validation_fraction_groups)
        or controls.reference_validation_fraction_groups <= 0.0
        or controls.reference_validation_fraction_groups >= 1.0
        or controls.fold_assignment_label_source != "pre_corruption_label"
        or controls.corruption_rounding_policy != "round_half_up"
    ):
        raise ValueError("primary split/corruption policy controls are invalid")
    if controls.n_splits < 2 or controls.split_seed < 0:
        raise ValueError("primary OOF controls are invalid")
    if controls.oof_split_kind != "stratified_group" or not controls.no_nucleus_level_fallback:
        raise ValueError("primary OOF must remain stratified-group with no nucleus fallback")
    if (
        not np.isfinite(controls.logistic_l2)
        or controls.logistic_l2 < 0.0
        or controls.logistic_max_iter <= 0
        or controls.logistic_class_weight != "balanced"
        or controls.logistic_class_weight_label_source != "observed_development_labels_only"
        or controls.logistic_fit_label_source != "observed_development_labels_only"
        or controls.mlp_fit_label_source != "observed_development_labels_only"
        or controls.logistic_model_seed < 0
    ):
        raise ValueError("primary logistic controls are invalid")
    controls.mlp_config.validate()
    calibration = controls.calibration
    if (
        calibration.source != "reference_validation_only"
        or calibration.reporting != "calibrated_and_uncalibrated"
        or calibration.fit_labels_policy != "observed_reference_validation_labels_only"
        or calibration.seed < 0
        or len(calibration.parameters) != len(dict(calibration.parameters))
    ):
        raise ValueError("primary calibration controls are invalid")
    if calibration.enabled:
        if calibration.method == "none":
            raise ValueError("enabled primary calibration cannot use method=none")
    elif calibration.method != "none" or calibration.parameters:
        raise ValueError("disabled primary calibration requires method=none and no parameters")
    if controls.primary_metric != "average_precision":
        raise ValueError("primary metric must remain average_precision")
    if (
        controls.neighbour_k <= 0
        or controls.neighbour_metric not in {"cosine", "euclidean"}
        or not controls.neighbour_exclude_same_group
    ):
        raise ValueError("nearest-neighbour controls are invalid")
    if controls.cleanlab_failure_policy != "missing_with_recorded_blocker":
        raise ValueError("Cleanlab failures must remain missing with a recorded blocker")
    if controls.random_review_repeats < 100 or controls.random_review_seed < 0:
        raise ValueError("primary random-review controls are invalid")
    if controls.subgroup_min_samples < 100 or controls.subgroup_min_corruptions < 10:
        raise ValueError("primary subgroup reliability thresholds are invalid")
    if controls.bootstrap_iterations < 2_000 or controls.bootstrap_seed < 0:
        raise ValueError("primary paired group bootstrap controls are invalid")
    if controls.frozen_config_schema_version == 2 and controls.paired_method_comparisons:
        raise ValueError("real primary controls cannot contain broadcast legacy comparisons")
    if controls.frozen_config_schema_version == 2:
        _validate_frozen_primary_comparisons(controls, budgets)
    if not controls.restoration_cell_ids or len(set(controls.restoration_cell_ids)) != len(
        controls.restoration_cell_ids
    ):
        raise ValueError("frozen restoration cell IDs must be non-empty and unique")
    planned_ids = {cell.cell_id for cell in controls.plan.cells}
    if not set(controls.restoration_cell_ids).issubset(planned_ids):
        raise ValueError("frozen restoration cell IDs are absent from the primary plan")
    if (
        controls.restoration_review_budget != controls.primary_review_budget
        or controls.restoration_random_repeats < 100
        or controls.restoration_random_seed < 0
        or not controls.restoration_include_reference_validation_in_training
    ):
        raise ValueError("primary restoration controls are invalid")
    if controls.restoration_required_experiments != (
        "uncorrupted_reference_baseline",
        "corrupted_observed_baseline",
        "random_review_restoration",
        "audit_guided_restoration",
    ):
        raise ValueError("primary restoration experiment names are invalid")
    if tuple(
        comparison.as_dict() for comparison in controls.restoration_downstream_comparisons
    ) != (
        {
            "comparison_id": "audit_guided_minus_random_macro_f1",
            "method_a": "audit_guided_restoration",
            "method_b": "random_review_restoration",
            "metric": "macro_f1",
            "direction": "method_a_minus_method_b",
        },
    ):
        raise ValueError("primary downstream restoration comparison is invalid")
    transition = np.asarray(controls.confusion_transition_matrix, dtype=np.float64)
    if transition.shape != (len(controls.class_order), len(controls.class_order)):
        raise ValueError("frozen confusion transition matrix has the wrong shape")
    if not np.isfinite(transition).all() or not np.allclose(transition.sum(axis=1), 1.0):
        raise ValueError("frozen confusion transition rows must be finite and sum to one")
    if not np.allclose(np.diag(transition), 0.0):
        raise ValueError("frozen confusion transition matrix must forbid self-replacement")
    if not controls.group_conditional_grouping_field:
        raise ValueError("group-conditional grouping field must be frozen")
    conditional_weights = dict(controls.group_conditional_weights_by_value)
    if not conditional_weights or len(conditional_weights) != len(
        controls.group_conditional_weights_by_value
    ):
        raise ValueError("group-conditional value weights must be non-empty and unique")
    all_weights = (*conditional_weights.values(), controls.group_conditional_default_weight)
    if any(not np.isfinite(value) or value < 0.0 for value in all_weights):
        raise ValueError("group-conditional weights must be finite and non-negative")
    if (
        not controls.instance_generator_representation
        or not controls.instance_auditor_representation_families
        or not controls.instance_independence_matrix_path
        or len(controls.instance_independence_matrix_sha256) != 64
    ):
        raise ValueError("instance-dependent independence controls are incomplete")
    return budgets


def _bootstrap_dict(result: Any, *, shared_draw_evidence: str) -> dict[str, Any]:
    return {
        "iterations": result.requested_iterations,
        "valid_iterations": result.valid_iterations,
        "mean_difference": result.mean_difference,
        "interval_95": result.interval_95,
        "probability_positive": result.probability_positive,
        "shared_draw_evidence": shared_draw_evidence,
    }


class _ReferenceValidationMLP:
    """Adapter that exposes the generic estimator interface without test access."""

    def __init__(
        self,
        config: FrozenEmbeddingMLPConfig,
        validation_features: NDArray[np.float64],
        validation_labels: NDArray[np.int64],
    ) -> None:
        self._classifier = FrozenEmbeddingMLPClassifier(config)
        self._validation_features = np.asarray(validation_features, dtype=np.float64)
        self._validation_labels = np.asarray(validation_labels, dtype=np.int64)
        self.classes_: NDArray[np.int64] | None = None

    def fit(
        self,
        features: NDArray[np.generic],
        labels: NDArray[np.generic],
    ) -> Self:
        kwargs: dict[str, Any] = {}
        if self._classifier.config.early_stopping_patience is not None:
            kwargs = {
                "validation_data": (self._validation_features, self._validation_labels),
                "validation_role": "reference_validation",
            }
        self._classifier.fit(features, labels, **kwargs)
        if self._classifier.classes_ is None:
            raise RuntimeError("frozen-embedding MLP did not expose fitted classes")
        self.classes_ = self._classifier.classes_.copy()
        return self

    def predict_proba(self, features: NDArray[np.generic]) -> NDArray[np.float64]:
        return self._classifier.predict_proba(features)


@dataclass(frozen=True, slots=True)
class _MLPDownstreamFactory:
    config: FrozenEmbeddingMLPConfig
    validation_features: NDArray[np.float64]
    validation_labels: NDArray[np.int64]

    def __call__(
        self,
        *,
        class_order: tuple[int, ...],
        model_seed: int,
    ) -> DownstreamEstimator:
        del class_order
        return _ReferenceValidationMLP(
            replace(self.config, seed=model_seed),
            self.validation_features,
            self.validation_labels,
        )


def _grouped_oof_mlp(
    features: NDArray[np.float64],
    observed_labels: NDArray[np.int64],
    inputs: PrimaryMatrixInputs,
    controls: PrimaryExecutionControls,
    *,
    representation: str,
) -> Any:
    """Run MLP OOF with optional early stopping on reference validation only."""

    validation_features = np.asarray(
        inputs.reference_validation_features[representation], dtype=np.float64
    )
    validation_labels = np.asarray(inputs.reference_validation_labels, dtype=np.int64)

    def estimator_factory(context: OOFFoldEstimatorContext) -> _ReferenceValidationMLP:
        return _ReferenceValidationMLP(
            replace(controls.mlp_config, seed=context.model_seed),
            validation_features,
            validation_labels,
        )

    return grouped_oof_predict(
        features,
        observed_labels,
        inputs.audit_group_ids,
        estimator_factory=estimator_factory,
        model_name="frozen_embedding_mlp",
        final_reference_group_ids=inputs.final_test_group_ids,
        sample_ids=inputs.audit_sample_ids,
        fold_assignment_labels=inputs.audit_pre_corruption_labels,
        fold_assignment_label_source=controls.fold_assignment_label_source,
        n_splits=controls.n_splits,
        class_order=inputs.class_order,
        split_seed=controls.split_seed,
        model_seed=controls.mlp_config.seed,
        representation=representation,
    )


def _budget_key(value: float) -> str:
    return format(float(value), ".12g")


def _frozen_group_weights(
    inputs: PrimaryMatrixInputs,
    controls: PrimaryExecutionControls,
) -> dict[str, float]:
    values = inputs.audit_grouping_values.get(controls.group_conditional_grouping_field)
    if values is None:
        raise ValueError(
            "audit inputs lack frozen group-conditional field "
            f"{controls.group_conditional_grouping_field!r}"
        )
    lookup = dict(controls.group_conditional_weights_by_value)
    group_values: dict[str, str] = {}
    for group_id, grouping_value in zip(inputs.audit_group_ids, values, strict=True):
        previous = group_values.setdefault(group_id, grouping_value)
        if previous != grouping_value:
            raise ValueError(f"source group {group_id!r} spans multiple group-conditional values")
    return {
        group_id: float(lookup.get(grouping_value, controls.group_conditional_default_weight))
        for group_id, grouping_value in group_values.items()
    }


def _validate_real_input_bindings(
    inputs: PrimaryMatrixInputs,
    controls: PrimaryExecutionControls,
) -> None:
    if tuple(inputs.class_order) != controls.class_order:
        raise ValueError("input class_order differs from the frozen primary controls")
    if inputs.corruption_generator_representation != controls.instance_generator_representation:
        raise ValueError("corruption generator representation differs from the frozen config")
    available_representations = set(inputs.audit_features)
    if not set(inputs.independence_evidence_by_representation).issubset(available_representations):
        raise ValueError("real primary independence evidence names an unavailable representation")
    if set(inputs.independence_matrix_artifact_sha256_by_representation) != (
        available_representations
    ):
        raise ValueError(
            "real primary inputs require one independence artifact SHA per representation"
        )
    if set(inputs.corruption_auditor_family_by_representation) != available_representations:
        raise ValueError("real primary inputs require one auditor family per representation")
    allowed_families = set(controls.instance_auditor_representation_families)
    for representation_id in sorted(available_representations):
        family = inputs.corruption_auditor_family_by_representation[representation_id]
        if family not in allowed_families:
            raise ValueError(
                f"auditor family for {representation_id!r} is absent from the frozen plan"
            )
        artifact_sha = inputs.independence_matrix_artifact_sha256_by_representation[
            representation_id
        ]
        if artifact_sha != controls.instance_independence_matrix_sha256:
            raise ValueError(f"independence matrix artifact SHA differs for {representation_id!r}")


@dataclass(frozen=True, slots=True)
class _CellIndependence:
    evidence: FeatureIndependenceEvidence | None
    auditor_representation: str | None
    matrix_artifact_sha256: str | None
    evidence_sha256: str | None


def _cell_independence(
    inputs: PrimaryMatrixInputs,
    controls: PrimaryExecutionControls,
    representation_id: str,
) -> _CellIndependence:
    if controls.frozen_config_schema_version == 1:
        evidence = inputs.independence_evidence
        return _CellIndependence(
            evidence=evidence,
            auditor_representation=inputs.corruption_auditor_representation,
            matrix_artifact_sha256=inputs.independence_matrix_artifact_sha256,
            evidence_sha256=(
                canonical_sha256(evidence.as_dict()) if evidence is not None else None
            ),
        )
    evidence = inputs.independence_evidence_by_representation.get(representation_id)
    if evidence is None:
        return _CellIndependence(
            None,
            representation_id,
            inputs.independence_matrix_artifact_sha256_by_representation.get(representation_id),
            None,
        )
    if evidence.auditor.representation_name != representation_id:
        raise ValueError(f"independence evidence identity mismatch for {representation_id!r}")
    if evidence.auditor.feature_artifact_hash != array_artifact_sha256(
        np.asarray(inputs.audit_features[representation_id])
    ):
        raise ValueError(f"independence auditor array mismatch for {representation_id!r}")
    return _CellIndependence(
        evidence=evidence,
        auditor_representation=representation_id,
        matrix_artifact_sha256=(
            inputs.independence_matrix_artifact_sha256_by_representation.get(representation_id)
        ),
        evidence_sha256=canonical_sha256(evidence.as_dict()),
    )


def _apply_cell_corruption(
    inputs: PrimaryMatrixInputs,
    controls: PrimaryExecutionControls,
    scenario: PrimaryScenario,
    *,
    upstream_manifest_hash: str,
    representation_id: str | None,
) -> tuple[CorruptionResult, _CellIndependence]:
    is_instance = scenario.mechanism == "instance_dependent_corruption"
    if is_instance and representation_id is None:
        raise ValueError("instance-dependent corruption requires an auditor representation")
    independence = (
        _cell_independence(inputs, controls, str(representation_id))
        if is_instance
        else _CellIndependence(None, None, None, None)
    )
    if controls.frozen_config_schema_version == 2:
        transition_matrix = (
            np.asarray(controls.confusion_transition_matrix, dtype=np.float64)
            if scenario.mechanism == "confusion_targeted_corruption"
            else None
        )
        group_weights = (
            _frozen_group_weights(inputs, controls)
            if scenario.mechanism == "group_conditional_corruption"
            else None
        )
    else:
        parameters = inputs.corruption_parameters_by_scenario.get(scenario.scenario_id, {})
        transition_matrix = (
            np.asarray(parameters["transition_matrix"], dtype=np.float64)
            if "transition_matrix" in parameters
            else None
        )
        raw_group_weights = parameters.get("group_weights")
        group_weights = (
            {str(key): float(value) for key, value in dict(raw_group_weights).items()}
            if raw_group_weights is not None
            else None
        )
    if scenario.mechanism == "confusion_targeted_corruption" and transition_matrix is None:
        raise ValueError(
            f"confusion scenario {scenario.scenario_id} lacks its frozen transition matrix"
        )
    if scenario.mechanism == "group_conditional_corruption" and group_weights is None:
        raise ValueError(
            f"group-conditional scenario {scenario.scenario_id} lacks frozen group weights"
        )
    corruption = apply_controlled_corruption(
        inputs.audit_pre_corruption_labels,
        sample_ids=inputs.audit_sample_ids,
        group_ids=inputs.audit_group_ids,
        rate=scenario.rate,
        mechanism=scenario.mechanism,
        seed=scenario.corruption_seed,
        n_classes=len(inputs.class_order),
        generator_features=inputs.corruption_generator_features if is_instance else None,
        generator_representation=inputs.corruption_generator_representation,
        auditor_representation=independence.auditor_representation,
        independence_evidence=independence.evidence,
        transition_matrix=transition_matrix,
        group_weights=group_weights,
        upstream_manifest_hash=upstream_manifest_hash,
        dataset_seed=inputs.dataset_seed,
        timestamp_utc=inputs.corruption_timestamp_utc,
    )
    return corruption, independence


def _shared_corruption_sha256(
    scenario: PrimaryScenario,
    corruption: CorruptionResult,
) -> str:
    """Hash the shared injected-label assignment, excluding auditor-specific evidence."""

    return canonical_sha256(
        {
            "schema_version": 1,
            "scenario": asdict(scenario),
            "pre_corruption_labels": corruption.pre_corruption_labels.tolist(),
            "observed_labels": corruption.observed_labels.tolist(),
            "is_injected_corruption": corruption.is_injected_corruption.tolist(),
            "selected_indices": corruption.selected_indices.tolist(),
            "replacement_class": corruption.replacement_class.tolist(),
        }
    )


def _save_restoration(
    output_directory: Path,
    downstream: DownstreamEvaluation,
    *,
    selected_cell: PrimaryCell,
    execution_controls_binding_sha256: str,
    shared_scenario_corruption_hash: str,
    ranking_method: str,
    audit_risk_scores: NDArray[np.float64],
    audit_sample_ids: Sequence[str],
    audit_group_ids: Sequence[str],
    audit_pre_corruption_labels: NDArray[np.int64],
    audit_observed_labels: NDArray[np.int64],
    audit_is_injected_corruption: NDArray[np.bool_],
    final_test_sample_ids: Sequence[str],
    final_test_group_ids: Sequence[str],
    final_test_labels: NDArray[np.int64],
    class_order: Sequence[int],
    review_budget: float,
    required_experiments: Sequence[str],
    downstream_comparisons: Sequence[PrimaryDownstreamComparison],
) -> tuple[Path, Path, Path]:
    random_macro_f1 = np.asarray(
        [run.metrics.macro_f1 for run in downstream.random_review_restoration],
        dtype=np.float64,
    )
    if random_macro_f1.shape != (len(downstream.random_review_restoration),) or not len(
        random_macro_f1
    ):
        raise ValueError("downstream comparison requires random-review restoration runs")
    guided_macro_f1 = float(downstream.audit_guided_restoration.metrics.macro_f1)
    random_review_seeds = tuple(run.review_seed for run in downstream.random_review_restoration)
    if any(seed is None for seed in random_review_seeds):
        raise ValueError("random-review restoration run lacks its frozen seed")
    comparison_records: list[dict[str, Any]] = []
    comparison_arrays: dict[str, NDArray[np.generic]] = {
        "downstream_comparison_ids": np.asarray(
            [comparison.comparison_id for comparison in downstream_comparisons],
            dtype=np.str_,
        ),
        "random_review_seeds": np.asarray(
            [int(seed) for seed in random_review_seeds if seed is not None],
            dtype=np.int64,
        ),
    }
    for comparison_index, comparison in enumerate(downstream_comparisons):
        if comparison.as_dict() != {
            "comparison_id": "audit_guided_minus_random_macro_f1",
            "method_a": "audit_guided_restoration",
            "method_b": "random_review_restoration",
            "metric": "macro_f1",
            "direction": "method_a_minus_method_b",
        }:
            raise ValueError("unsupported frozen downstream restoration comparison")
        metric_a = np.full(len(random_macro_f1), guided_macro_f1, dtype=np.float64)
        differences = metric_a - random_macro_f1
        prefix = f"downstream_comparison_{comparison_index:03d}"
        comparison_arrays[f"{prefix}_metric_a"] = metric_a
        comparison_arrays[f"{prefix}_metric_b"] = random_macro_f1
        comparison_arrays[f"{prefix}_differences"] = differences
        comparison_records.append(
            {
                **comparison.as_dict(),
                "status": "reported",
                "pairing": "same_final_reference_set_across_frozen_random_review_repetitions",
                "random_repetitions": len(random_macro_f1),
                "point_metric_a": guided_macro_f1,
                "point_metric_b": float(random_macro_f1.mean()),
                "point_difference": float(guided_macro_f1 - random_macro_f1.mean()),
                "mean_difference": float(differences.mean()),
                "interval_95": [
                    float(np.quantile(differences, 0.025)),
                    float(np.quantile(differences, 0.975)),
                ],
                "probability_positive": float(
                    np.mean(differences > 0.0) + 0.5 * np.mean(differences == 0.0)
                ),
            }
        )
    json_path = atomic_write_json(
        output_directory / "restoration.json",
        {
            "schema_version": 1,
            "cell": asdict(selected_cell),
            "selected_cell_id": selected_cell.cell_id,
            "execution_controls_binding_sha256": execution_controls_binding_sha256,
            "shared_scenario_corruption_hash": shared_scenario_corruption_hash,
            "ranking_method": ranking_method,
            "review_budget": review_budget,
            "required_experiments": list(required_experiments),
            "downstream_comparisons": comparison_records,
            "evaluation": downstream.as_dict(),
        },
    )
    arrays_path = _atomic_npz(
        output_directory / "restoration_evidence.npz",
        {
            "class_order": np.asarray(class_order, dtype=np.int64),
            "audit_sample_ids": np.asarray(audit_sample_ids, dtype=np.str_),
            "audit_group_ids": np.asarray(audit_group_ids, dtype=np.str_),
            "audit_pre_corruption_labels": np.asarray(audit_pre_corruption_labels, dtype=np.int64),
            "audit_observed_labels": np.asarray(audit_observed_labels, dtype=np.int64),
            "audit_is_injected_corruption": np.asarray(
                audit_is_injected_corruption, dtype=np.bool_
            ),
            "audit_risk_scores": np.asarray(audit_risk_scores, dtype=np.float64),
            "final_test_sample_ids": np.asarray(final_test_sample_ids, dtype=np.str_),
            "final_test_group_ids": np.asarray(final_test_group_ids, dtype=np.str_),
            "final_test_labels": np.asarray(final_test_labels, dtype=np.int64),
            "audit_reviewed_indices": downstream.audit_reviewed_indices,
            "guided_reviewed_mask": downstream.audit_guided_restoration_evidence.reviewed_mask,
            "guided_restored_mask": downstream.audit_guided_restoration_evidence.restored_mask,
            "guided_restored_labels": downstream.audit_guided_restoration_evidence.restored_labels,
            "random_reviewed_masks": np.stack(
                [value.reviewed_mask for value in downstream.random_review_restoration_evidence]
            ),
            "random_restored_masks": np.stack(
                [value.restored_mask for value in downstream.random_review_restoration_evidence]
            ),
            "random_restored_labels": np.stack(
                [value.restored_labels for value in downstream.random_review_restoration_evidence]
            ),
            "uncorrupted_final_probabilities": (
                downstream.uncorrupted_reference_baseline.final_test_probabilities
            ),
            "uncorrupted_final_predicted_class": (
                downstream.uncorrupted_reference_baseline.final_test_predicted_class
            ),
            "corrupted_final_probabilities": (
                downstream.corrupted_observed_baseline.final_test_probabilities
            ),
            "corrupted_final_predicted_class": (
                downstream.corrupted_observed_baseline.final_test_predicted_class
            ),
            "guided_final_probabilities": (
                downstream.audit_guided_restoration.final_test_probabilities
            ),
            "guided_final_predicted_class": (
                downstream.audit_guided_restoration.final_test_predicted_class
            ),
            "random_final_probabilities": np.stack(
                [run.final_test_probabilities for run in downstream.random_review_restoration]
            ),
            "random_final_predicted_class": np.stack(
                [run.final_test_predicted_class for run in downstream.random_review_restoration]
            ),
            "random_reviewed_indices": np.stack(downstream.random_reviewed_indices),
            **comparison_arrays,
        },
    )
    manifest_path = atomic_write_json(
        output_directory / "restoration_manifest.json",
        {
            "schema_version": 1,
            "cell_id": selected_cell.cell_id,
            "execution_controls_binding_sha256": execution_controls_binding_sha256,
            "artifacts": [
                {"path": json_path.name, "sha256": sha256_file(json_path)},
                {"path": arrays_path.name, "sha256": sha256_file(arrays_path)},
            ],
        },
    )
    return json_path, arrays_path, manifest_path


def execute_primary_matrix(
    inputs: PrimaryMatrixInputs,
    plan: PrimaryMatrixPlan,
    *,
    output_directory: str | Path,
    execution_controls: PrimaryExecutionControls,
) -> PrimaryMatrixArtifacts:
    """Execute a schema-v2 frozen matrix with no loose control overrides."""

    execution_controls.validate_for_plan(plan)
    return _execute_primary_matrix_core(
        inputs,
        plan,
        execution_controls,
        output_directory=output_directory,
        artifact_scope=REAL_PRIMARY_ARTIFACT_SCOPE,
        synthetic_layout=False,
    )


def _execute_primary_matrix_core(
    inputs: PrimaryMatrixInputs,
    plan: PrimaryMatrixPlan,
    controls: PrimaryExecutionControls,
    *,
    output_directory: str | Path,
    artifact_scope: str,
    synthetic_layout: bool,
) -> PrimaryMatrixArtifacts:
    """Execute the bound controls; schema-v1 is private to the sealed fixture.

    This core never enables a completion stage. Real primary eligibility remains
    the responsibility of the independently hash-bound workflow gate.
    """

    inputs.validate()
    if controls.plan != plan or controls.plan_sha256 != canonical_sha256(plan.as_dict()):
        raise ValueError("execution controls do not match the supplied primary plan")
    if controls.binding_sha256 != canonical_sha256(controls._binding_payload()):
        raise ValueError("primary execution controls binding SHA is invalid")
    budgets = _validate_execution_configuration(
        audit_methods=controls.audit_methods,
        primary_ranking_method=controls.primary_ranking_method,
        restoration_ranking_method=controls.restoration_ranking_method,
        fixed_hybrid_components=controls.fixed_hybrid_components,
        fixed_hybrid_weights=controls.fixed_hybrid_weights,
        primary_review_budget=controls.primary_review_budget,
        secondary_review_budgets=controls.secondary_review_budgets,
        paired_method_comparisons=controls.paired_method_comparisons,
    )
    if controls.frozen_config_schema_version == 2:
        _validate_runtime_controls(controls)
        if controls.calibration.enabled:
            raise ValueError(
                "frozen primary calibration is enabled but its exact method is not implemented; "
                "execution refuses to emit uncalibrated probabilities as calibrated evidence"
            )
        _validate_real_input_bindings(inputs, controls)
        if any(inputs.corruption_parameters_by_scenario.values()):
            raise ValueError("schema-v2 execution rejects loose corruption parameter overrides")
    elif not (
        controls.frozen_config_schema_version == 1
        and artifact_scope == SYNTHETIC_PRIMARY_ARTIFACT_SCOPE
        and synthetic_layout
    ):
        raise ValueError("schema-v1 controls are restricted to the sealed synthetic fixture")
    destination = Path(output_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for reserved in (
        "matrix_plan.json",
        "execution_controls.json",
        "cell_index.csv",
        "completion_evidence.json",
    ):
        if (destination / reserved).exists():
            raise FileExistsError(f"primary matrix output already exists: {destination / reserved}")
    matrix_plan_path = atomic_write_json(destination / "matrix_plan.json", plan.as_dict())
    execution_controls_path = atomic_write_json(
        destination / "execution_controls.json", controls.as_dict()
    )
    cells_root = destination / "cells"
    scenarios_root = destination / "corruption_scenarios"
    cells_root.mkdir()
    scenarios_root.mkdir()
    scenario_by_id = {scenario.scenario_id: scenario for scenario in plan.scenarios}
    if set(cell.scenario_id for cell in plan.cells) != set(scenario_by_id):
        raise ValueError("primary cells and corruption scenarios are inconsistent")
    upstream_manifest_hash = canonical_sha256(
        {
            "sample_ids": inputs.audit_sample_ids,
            "group_ids": inputs.audit_group_ids,
            "pre_corruption_labels": inputs.audit_pre_corruption_labels.tolist(),
        }
    )
    corruptions: dict[str, CorruptionResult] = {}
    shared_corruption_hashes: dict[str, str] = {}
    for scenario in plan.scenarios:
        if scenario.mechanism == "instance_dependent_corruption":
            atomic_write_json(
                scenarios_root / f"{scenario.scenario_id}.json",
                {
                    "scenario": asdict(scenario),
                    "configuration_scope": "per_auditor_representation",
                    "reason": (
                        "Injected rows are generator-determined and reproducible, but each "
                        "cell retains its exact generator/auditor independence evidence."
                    ),
                },
            )
            continue
        corruption, _ = _apply_cell_corruption(
            inputs,
            controls,
            scenario,
            upstream_manifest_hash=upstream_manifest_hash,
            representation_id=None,
        )
        corruptions[scenario.scenario_id] = corruption
        shared_corruption_hash = _shared_corruption_sha256(scenario, corruption)
        shared_corruption_hashes[scenario.scenario_id] = shared_corruption_hash
        atomic_write_json(
            scenarios_root / f"{scenario.scenario_id}.json",
            {
                "scenario": asdict(scenario),
                "configuration_hash": shared_corruption_hash,
                "shared_scenario_corruption_hash": shared_corruption_hash,
                "cell_corruption_provenance_sha256": corruption.configuration_hash,
                "configuration_payload": json.loads(corruption.configuration_payload_json),
                "rows": corruption.manifest_rows(inputs.audit_sample_ids, inputs.audit_group_ids),
                "shared_across_every_representation_and_classifier": True,
            },
        )

    outcomes: list[dict[str, Any]] = []
    restoration_targets = set(controls.restoration_cell_ids)
    selected_restorations: dict[str, tuple[PrimaryCell, CorruptionResult, NDArray[np.float64]]] = {}
    for cell in plan.cells:
        cell_directory = cells_root / cell.cell_id
        cell_directory.mkdir()
        if cell.representation_id not in inputs.audit_features:
            if cell.required:
                outcomes.append(
                    {
                        "cell_id": cell.cell_id,
                        "scenario_id": cell.scenario_id,
                        "mechanism": cell.mechanism,
                        "rate": cell.rate,
                        "corruption_seed": cell.corruption_seed,
                        "representation_id": cell.representation_id,
                        "classifier_id": cell.classifier_id,
                        "required": True,
                        "status": "failed",
                        "execution_controls_binding_sha256": controls.binding_sha256,
                        "error": f"required representation unavailable: {cell.representation_id}",
                    }
                )
            else:
                outcomes.append(
                    {
                        "cell_id": cell.cell_id,
                        "scenario_id": cell.scenario_id,
                        "mechanism": cell.mechanism,
                        "rate": cell.rate,
                        "corruption_seed": cell.corruption_seed,
                        "representation_id": cell.representation_id,
                        "classifier_id": cell.classifier_id,
                        "required": False,
                        "status": "skipped_with_frozen_blocker",
                        "execution_controls_binding_sha256": controls.binding_sha256,
                        "frozen_unavailability": True,
                        "blocker": f"optional representation unavailable: {cell.representation_id}",
                    }
                )
            continue
        scenario = scenario_by_id[cell.scenario_id]
        features = np.asarray(inputs.audit_features[cell.representation_id], dtype=np.float64)
        try:
            if scenario.mechanism == "instance_dependent_corruption":
                corruption, independence = _apply_cell_corruption(
                    inputs,
                    controls,
                    scenario,
                    upstream_manifest_hash=upstream_manifest_hash,
                    representation_id=cell.representation_id,
                )
            else:
                corruption = corruptions[cell.scenario_id]
                independence = _CellIndependence(None, None, None, None)
            shared_corruption_hash = _shared_corruption_sha256(scenario, corruption)
            previous_shared_hash = shared_corruption_hashes.setdefault(
                scenario.scenario_id, shared_corruption_hash
            )
            if previous_shared_hash != shared_corruption_hash:
                raise RuntimeError(
                    "auditor-specific evidence changed the shared injected-label assignment"
                )
            primary_confirmatory_eligible = not corruption.circularity_risk
            if cell.classifier_id == "multinomial_logistic_regression":
                oof = grouped_oof_logistic(
                    features,
                    corruption.observed_labels,
                    inputs.audit_group_ids,
                    final_reference_group_ids=inputs.final_test_group_ids,
                    sample_ids=inputs.audit_sample_ids,
                    fold_assignment_labels=inputs.audit_pre_corruption_labels,
                    fold_assignment_label_source=controls.fold_assignment_label_source,
                    n_splits=controls.n_splits,
                    class_order=inputs.class_order,
                    split_seed=controls.split_seed,
                    model_seed=controls.logistic_model_seed,
                    representation=cell.representation_id,
                    l2=controls.logistic_l2,
                    max_iter=controls.logistic_max_iter,
                )
            elif cell.classifier_id == "small_mlp":
                oof = _grouped_oof_mlp(
                    features,
                    corruption.observed_labels,
                    inputs,
                    controls,
                    representation=cell.representation_id,
                )
            else:
                raise ValueError(f"unsupported primary classifier: {cell.classifier_id}")
            base_score_methods = {
                "self_confidence",
                "negative_log_likelihood",
                "prediction_margin",
                "predictive_entropy",
            }
            risks = {
                method: score_annotations(
                    corruption.observed_labels,
                    oof.probabilities,
                    method=method,
                    class_order=inputs.class_order,
                )
                for method in controls.audit_methods
                if method in base_score_methods
            }
            cleanlab = cleanlab_scores(corruption.observed_labels, oof.probabilities)
            if (
                "cleanlab" in controls.audit_methods
                and cleanlab.available
                and cleanlab.risk_scores is not None
            ):
                risks["cleanlab"] = cleanlab.risk_scores
            neighbours = None
            if "nearest_neighbour_disagreement" in controls.audit_methods:
                neighbours = fold_safe_neighbour_disagreement(
                    features,
                    corruption.observed_labels,
                    inputs.audit_group_ids,
                    oof.fold_id,
                    oof.training_groups_by_fold,
                    sample_ids=inputs.audit_sample_ids,
                    class_order=inputs.class_order,
                    k=controls.neighbour_k,
                    metric=controls.neighbour_metric,
                )
                risks["nearest_neighbour_disagreement"] = neighbours.risk_scores
            missing_hybrid = set(controls.fixed_hybrid_components).difference(risks)
            if missing_hybrid:
                raise RuntimeError(
                    "fixed hybrid component unavailable with recorded blocker: "
                    f"{sorted(missing_hybrid)}; cleanlab_error={cleanlab.error!r}"
                )
            risks["fixed_hybrid"] = fixed_hybrid_score(
                risks,
                components=controls.fixed_hybrid_components,
                weights=controls.fixed_hybrid_weights,
            )
            required_risks = {
                controls.primary_ranking_method,
                *(item.method_a for item in controls.paired_method_comparisons),
                *(item.method_b for item in controls.paired_method_comparisons),
            }
            if cell.cell_id in restoration_targets:
                required_risks.add(controls.restoration_ranking_method)
            missing_required = required_risks.difference(risks)
            if missing_required:
                raise RuntimeError(
                    "frozen risk method unavailable with recorded blocker: "
                    f"{sorted(missing_required)}; cleanlab_error={cleanlab.error!r}"
                )
            review_by_method = {
                method: {
                    _budget_key(budget): evaluate_review_budget(
                        corruption.is_injected_corruption,
                        risk,
                        budget=budget,
                        tie_break_ids=inputs.audit_sample_ids,
                    )
                    for budget in budgets
                }
                for method, risk in risks.items()
            }
            random_review_by_budget = {
                _budget_key(budget): random_review_baseline(
                    corruption.is_injected_corruption,
                    budget=budget,
                    repeats=controls.random_review_repeats,
                    seed=controls.random_review_seed,
                )
                for budget in budgets
            }
            draws = (
                draw_group_bootstrap_indices(
                    inputs.audit_group_ids,
                    n_iterations=controls.bootstrap_iterations,
                    seed=controls.bootstrap_seed,
                )
                if controls.frozen_config_schema_version == 1
                else ()
            )
            paired_results = {
                comparison.comparison_id: paired_group_bootstrap(
                    corruption.is_injected_corruption,
                    risks[comparison.method_a],
                    risks[comparison.method_b],
                    inputs.audit_group_ids,
                    n_iterations=controls.bootstrap_iterations,
                    seed=controls.bootstrap_seed,
                    bootstrap_indices=draws,
                )
                for comparison in controls.paired_method_comparisons
            }
            flattened_draws, draw_offsets = _flatten_draws(draws)
            oof_path = _atomic_npz(
                cell_directory / "oof_predictions.npz",
                {
                    "sample_ids": np.asarray(inputs.audit_sample_ids, dtype=np.str_),
                    "group_ids": np.asarray(inputs.audit_group_ids, dtype=np.str_),
                    "pre_corruption_label": corruption.pre_corruption_labels,
                    "observed_label": corruption.observed_labels,
                    "is_injected_corruption": corruption.is_injected_corruption,
                    "probabilities": oof.probabilities,
                    "predicted_class": oof.predicted_class,
                    "fold_id": oof.fold_id,
                    "coverage_count": oof.coverage_count,
                    "fold_assignment_labels": oof.fold_assignment_labels,
                    "fold_assignment_label_source": np.asarray(
                        [oof.fold_assignment_label_source], dtype=np.str_
                    ),
                    "fold_assignment_labels_sha256": np.asarray(
                        [oof.fold_assignment_labels_sha256], dtype=np.str_
                    ),
                },
            )
            risks_path = _atomic_npz(
                cell_directory / "risk_scores.npz",
                {name: values for name, values in risks.items()},
            )
            bootstrap_arrays: dict[str, NDArray[np.generic]] = {
                "draw_indices": flattened_draws,
                "draw_offsets": draw_offsets,
                "comparison_ids": np.asarray(
                    [item.comparison_id for item in controls.paired_method_comparisons],
                    dtype=np.str_,
                ),
                "method_a": np.asarray(
                    [item.method_a for item in controls.paired_method_comparisons], dtype=np.str_
                ),
                "method_b": np.asarray(
                    [item.method_b for item in controls.paired_method_comparisons], dtype=np.str_
                ),
            }
            for comparison_index, comparison in enumerate(controls.paired_method_comparisons):
                paired = paired_results[comparison.comparison_id]
                prefix = f"comparison_{comparison_index:03d}"
                bootstrap_arrays[f"{prefix}_metric_a"] = paired.metric_a
                bootstrap_arrays[f"{prefix}_metric_b"] = paired.metric_b
                bootstrap_arrays[f"{prefix}_differences"] = paired.differences
                if (
                    comparison.method_a == "fixed_hybrid"
                    and comparison.method_b == "self_confidence"
                ):
                    bootstrap_arrays["metric_hybrid"] = paired.metric_a
                    bootstrap_arrays["metric_self_confidence"] = paired.metric_b
                    bootstrap_arrays["differences"] = paired.differences
            bootstrap_path = _atomic_npz(
                cell_directory / "bootstrap_evidence.npz", bootstrap_arrays
            )
            independence_path = atomic_write_json(
                cell_directory / "independence_evidence.json",
                {
                    "schema_version": 1,
                    "mechanism": scenario.mechanism,
                    "representation_id": cell.representation_id,
                    "status": corruption.independence_status,
                    "reason": corruption.independence_reason,
                    "circularity_risk": corruption.circularity_risk,
                    "primary_confirmatory_eligible": primary_confirmatory_eligible,
                    "matrix_artifact_sha256": independence.matrix_artifact_sha256,
                    "evidence_sha256": independence.evidence_sha256,
                    "evidence": (
                        independence.evidence.as_dict()
                        if independence.evidence is not None
                        else None
                    ),
                },
            )
            corruption_manifest_path = atomic_write_json(
                cell_directory / "corruption_manifest.json",
                {
                    "schema_version": 1,
                    "cell": asdict(cell),
                    "scenario": asdict(scenario),
                    "configuration_hash": shared_corruption_hash,
                    "shared_scenario_corruption_hash": shared_corruption_hash,
                    "cell_corruption_provenance_sha256": corruption.configuration_hash,
                    "rows": corruption.manifest_rows(
                        inputs.audit_sample_ids, inputs.audit_group_ids
                    ),
                    "independence_status": corruption.independence_status,
                    "circularity_risk": corruption.circularity_risk,
                },
            )
            if neighbours is None:
                raise RuntimeError("frozen nearest-neighbour evidence was not produced")
            neighbour_offsets = np.zeros(len(neighbours.neighbour_ids) + 1, dtype=np.int64)
            for neighbour_row_index, neighbour_row in enumerate(neighbours.neighbour_ids):
                neighbour_offsets[neighbour_row_index + 1] = neighbour_offsets[
                    neighbour_row_index
                ] + len(neighbour_row)
            neighbour_path = _atomic_npz(
                cell_directory / "neighbour_evidence.npz",
                {
                    "sample_ids": np.asarray(inputs.audit_sample_ids, dtype=np.str_),
                    "group_ids": np.asarray(inputs.audit_group_ids, dtype=np.str_),
                    "risk_scores": neighbours.risk_scores,
                    "alternative_class_support": neighbours.alternative_class_support,
                    "suggested_class": neighbours.suggested_class,
                    "neighbour_offsets": neighbour_offsets,
                    "neighbour_ids": np.asarray(
                        [value for row in neighbours.neighbour_ids for value in row],
                        dtype=np.str_,
                    ),
                    "neighbour_groups": np.asarray(
                        [value for row in neighbours.neighbour_groups for value in row],
                        dtype=np.str_,
                    ),
                    "neighbour_distances": np.asarray(
                        [value for row in neighbours.neighbour_distances for value in row],
                        dtype=np.float64,
                    ),
                    "k": np.asarray(neighbours.k, dtype=np.int64),
                    "metric": np.asarray([neighbours.metric], dtype=np.str_),
                },
            )
            cleanlab_path = _atomic_npz(
                cell_directory / "cleanlab_evidence.npz",
                {
                    "available": np.asarray(cleanlab.available, dtype=np.bool_),
                    "quality_scores": (
                        cleanlab.quality_scores
                        if cleanlab.quality_scores is not None
                        else np.empty(0, dtype=np.float64)
                    ),
                    "risk_scores": (
                        cleanlab.risk_scores
                        if cleanlab.risk_scores is not None
                        else np.empty(0, dtype=np.float64)
                    ),
                    "issue_mask": (
                        cleanlab.issue_mask
                        if cleanlab.issue_mask is not None
                        else np.empty(0, dtype=np.bool_)
                    ),
                    "suggested_class": (
                        cleanlab.suggested_class
                        if cleanlab.suggested_class is not None
                        else np.empty(0, dtype=np.int64)
                    ),
                },
            )
            cleanlab_json_path = atomic_write_json(
                cell_directory / "cleanlab_evidence.json",
                {
                    "schema_version": 1,
                    "available": cleanlab.available,
                    "package_version": cleanlab.package_version,
                    "api_path": cleanlab.api_path,
                    "error": cleanlab.error,
                    "blocker": None if cleanlab.available else cleanlab.error,
                    "failure_policy": controls.cleanlab_failure_policy,
                },
            )
            provenance_path = atomic_write_json(
                cell_directory / "oof_provenance.json",
                {
                    "class_order": oof.class_order,
                    "model_name": oof.model_name,
                    "representation": oof.representation,
                    "model_seed": oof.model_seed,
                    "split_seed": oof.split_seed,
                    "splitter_class_name": oof.splitter_class_name,
                    "splitter_fallback_status": oof.splitter_fallback_status,
                    "splitter_fallback_reason": oof.splitter_fallback_reason,
                    "fold_assignment_label_source": oof.fold_assignment_label_source,
                    "fold_assignment_labels_sha256": oof.fold_assignment_labels_sha256,
                    "coverage_exactly_once": bool(np.all(oof.coverage_count == 1)),
                    "final_reference_groups": oof.final_reference_groups,
                    "folds": [asdict(fold) for fold in oof.folds],
                    "execution_controls_binding_sha256": controls.binding_sha256,
                    "fit_label_source": (
                        controls.logistic_fit_label_source
                        if cell.classifier_id == "multinomial_logistic_regression"
                        else controls.mlp_fit_label_source
                    ),
                    "class_weight_label_source": "observed_development_labels_only",
                    "calibration": controls.calibration.as_dict(),
                    "calibration_applied": False,
                    "probability_variant": "uncalibrated",
                },
            )
            primary_budget_key = _budget_key(controls.primary_review_budget)
            ranking_metrics: dict[str, Any] = {}
            for method in controls.audit_methods:
                if method not in risks:
                    ranking_metrics[method] = {
                        "status": "missing",
                        "blocker": cleanlab.error if method == "cleanlab" else "unavailable",
                    }
                    continue
                risk = risks[method]
                ranking_metrics[method] = {
                    "status": "available",
                    "average_precision": average_precision(corruption.is_injected_corruption, risk),
                    "auroc": binary_auroc(corruption.is_injected_corruption, risk),
                    "review_budget": _jsonable_review(review_by_method[method][primary_budget_key]),
                    "review_budgets": {
                        key: _jsonable_review(value)
                        for key, value in review_by_method[method].items()
                    },
                }
            paired_metrics = {
                comparison.comparison_id: {
                    "method_a": comparison.method_a,
                    "method_b": comparison.method_b,
                    "claim_status": (
                        "primary_confirmatory_eligible"
                        if primary_confirmatory_eligible
                        else "excluded_circularity_risk"
                    ),
                    **_bootstrap_dict(
                        paired_results[comparison.comparison_id],
                        shared_draw_evidence=bootstrap_path.name,
                    ),
                }
                for comparison in controls.paired_method_comparisons
            }
            primary_random = random_review_by_budget[primary_budget_key]
            metrics: dict[str, Any] = {
                "cell": asdict(cell),
                "scenario": asdict(scenario),
                "execution_controls_binding_sha256": controls.binding_sha256,
                "corruption_configuration_hash": shared_corruption_hash,
                "shared_scenario_corruption_hash": shared_corruption_hash,
                "cell_corruption_provenance_sha256": corruption.configuration_hash,
                "independence_status": corruption.independence_status,
                "independence_evidence_sha256": independence.evidence_sha256,
                "independence_matrix_artifact_sha256": independence.matrix_artifact_sha256,
                "circularity_risk": corruption.circularity_risk,
                "primary_confirmatory_eligible": primary_confirmatory_eligible,
                "sample_count": len(inputs.audit_sample_ids),
                "group_count": len(set(inputs.audit_group_ids)),
                "exact_corruption_count": corruption.exact_count,
                "oof_coverage_exactly_once": bool(np.all(oof.coverage_count == 1)),
                "oof_group_overlap_count": 0,
                "final_reference_group_overlap_count": 0,
                "primary_ranking_method": controls.primary_ranking_method,
                "primary_metric": controls.primary_metric,
                "review_budget_order": list(budgets),
                "ranking": ranking_metrics,
                "cleanlab": {
                    "available": cleanlab.available,
                    "package_version": cleanlab.package_version,
                    "api_path": cleanlab.api_path,
                    "error": cleanlab.error,
                    "failure_policy": controls.cleanlab_failure_policy,
                },
                "random_review": {
                    "reviewed_count": primary_random.reviewed_count,
                    "repeats": len(primary_random.seeds),
                    "seeds": primary_random.seeds,
                    "mean_precision": primary_random.mean_precision,
                    "mean_recall": primary_random.mean_recall,
                    "recall_interval_95": primary_random.recall_interval_95,
                },
                "random_review_by_budget": {
                    key: {
                        "reviewed_count": result.reviewed_count,
                        "repeats": len(result.seeds),
                        "seeds": result.seeds,
                        "mean_precision": result.mean_precision,
                        "mean_recall": result.mean_recall,
                        "recall_interval_95": result.recall_interval_95,
                    }
                    for key, result in random_review_by_budget.items()
                },
                "paired_group_bootstrap": paired_metrics,
                "subgroup_reliability_thresholds": {
                    "minimum_samples": controls.subgroup_min_samples,
                    "minimum_injected_corruptions": controls.subgroup_min_corruptions,
                },
                "holm_families": list(controls.holm_families),
                "exploratory_multiple_comparison_correction": (
                    controls.exploratory_multiple_comparison_correction
                ),
                "comparison_execution_scope": (
                    "legacy_synthetic_cell"
                    if controls.frozen_config_schema_version == 1
                    else "deferred_exact_frozen_selectors"
                ),
            }
            legacy_comparison = next(
                (
                    comparison
                    for comparison in controls.paired_method_comparisons
                    if comparison.method_a == "fixed_hybrid"
                    and comparison.method_b == "self_confidence"
                ),
                None,
            )
            if legacy_comparison is not None:
                metrics["paired_group_bootstrap_hybrid_minus_self_confidence"] = paired_metrics[
                    legacy_comparison.comparison_id
                ]
            metrics_path = atomic_write_json(cell_directory / "metrics.json", metrics)
            ranking_order = rank_indices(
                risks[controls.primary_ranking_method], tie_break_ids=inputs.audit_sample_ids
            )
            ranking_rows = [
                {
                    "rank": rank,
                    "sample_id": inputs.audit_sample_ids[index],
                    "group_id": inputs.audit_group_ids[index],
                    "pre_corruption_label": int(corruption.pre_corruption_labels[index]),
                    "observed_label": int(corruption.observed_labels[index]),
                    "is_injected_corruption": bool(corruption.is_injected_corruption[index]),
                    **{name: float(values[index]) for name, values in risks.items()},
                }
                for rank, index in enumerate(ranking_order, start=1)
            ]
            ranking_path = _write_csv(
                cell_directory / "ranking.csv",
                tuple(ranking_rows[0]),
                ranking_rows,
            )
            manifest_path = _artifact_manifest(
                cell_directory,
                (
                    oof_path.name,
                    risks_path.name,
                    bootstrap_path.name,
                    independence_path.name,
                    corruption_manifest_path.name,
                    neighbour_path.name,
                    cleanlab_path.name,
                    cleanlab_json_path.name,
                    provenance_path.name,
                    metrics_path.name,
                    ranking_path.name,
                ),
            )
            outcome = {
                "cell_id": cell.cell_id,
                "scenario_id": cell.scenario_id,
                "mechanism": cell.mechanism,
                "rate": cell.rate,
                "corruption_seed": cell.corruption_seed,
                "representation_id": cell.representation_id,
                "classifier_id": cell.classifier_id,
                "required": cell.required,
                "status": "completed",
                "artifact_manifest_sha256": sha256_file(manifest_path),
                "metrics_sha256": sha256_file(metrics_path),
                "corruption_configuration_hash": shared_corruption_hash,
                "shared_scenario_corruption_hash": shared_corruption_hash,
                "cell_corruption_provenance_sha256": corruption.configuration_hash,
                "execution_controls_binding_sha256": controls.binding_sha256,
                "independence_status": corruption.independence_status,
                "independence_evidence_sha256": independence.evidence_sha256,
                "independence_matrix_artifact_sha256": independence.matrix_artifact_sha256,
                "circularity_risk": corruption.circularity_risk,
                "primary_confirmatory_eligible": primary_confirmatory_eligible,
            }
            outcomes.append(outcome)
            if cell.cell_id in restoration_targets and primary_confirmatory_eligible:
                selected_restorations[cell.cell_id] = (
                    cell,
                    corruption,
                    np.asarray(risks[controls.restoration_ranking_method], dtype=np.float64),
                )
        except Exception as exc:
            outcomes.append(
                {
                    "cell_id": cell.cell_id,
                    "scenario_id": cell.scenario_id,
                    "mechanism": cell.mechanism,
                    "rate": cell.rate,
                    "corruption_seed": cell.corruption_seed,
                    "representation_id": cell.representation_id,
                    "classifier_id": cell.classifier_id,
                    "required": cell.required,
                    "status": "failed",
                    "execution_controls_binding_sha256": controls.binding_sha256,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            atomic_write_json(
                cell_directory / "failure.json",
                {"error_type": type(exc).__name__, "error": str(exc)},
            )

    for scenario in plan.scenarios:
        if scenario.mechanism != "instance_dependent_corruption":
            continue
        shared_hash = shared_corruption_hashes.get(scenario.scenario_id)
        atomic_write_json(
            scenarios_root / f"{scenario.scenario_id}.json",
            {
                "scenario": asdict(scenario),
                "configuration_scope": "per_auditor_representation",
                "shared_scenario_corruption_hash": shared_hash,
                "shared_assignment_verified": shared_hash is not None,
                "reason": (
                    "Injected rows are generator-determined and shared; each cell retains "
                    "its exact generator/auditor independence evidence separately."
                ),
            },
        )
    missing_restorations = restoration_targets.difference(selected_restorations)
    if missing_restorations:
        raise RuntimeError(
            f"frozen restoration cells did not complete: {sorted(missing_restorations)}"
        )
    if synthetic_layout and len(selected_restorations) != 1:
        raise RuntimeError("synthetic fixture requires exactly one restoration cell")
    restoration_root = destination if synthetic_layout else destination / "restorations"
    if not synthetic_layout:
        restoration_root.mkdir()
    restoration_rows: list[dict[str, Any]] = []
    for restoration_cell_id in controls.restoration_cell_ids:
        selected_cell, selected_corruption, selected_risk = selected_restorations[
            restoration_cell_id
        ]
        selected_representation = selected_cell.representation_id
        estimator_factory: DownstreamEstimatorFactory | None = None
        if selected_cell.classifier_id == "multinomial_logistic_regression":
            downstream_model_seed = controls.logistic_model_seed
        elif selected_cell.classifier_id == "small_mlp":
            downstream_model_seed = controls.mlp_config.seed
            estimator_factory = _MLPDownstreamFactory(
                controls.mlp_config,
                np.asarray(
                    inputs.reference_validation_features[selected_representation],
                    dtype=np.float64,
                ),
                np.asarray(inputs.reference_validation_labels, dtype=np.int64),
            )
        else:
            raise ValueError(f"unsupported restoration classifier: {selected_cell.classifier_id}")
        downstream = evaluate_downstream_restoration(
            inputs.audit_features[selected_representation],
            selected_corruption.pre_corruption_labels,
            selected_corruption.observed_labels,
            selected_corruption.is_injected_corruption,
            inputs.final_test_features[selected_representation],
            inputs.final_test_labels,
            selected_risk,
            development_group_ids=inputs.audit_group_ids,
            final_test_group_ids=inputs.final_test_group_ids,
            final_test_is_injected_corruption=tuple(False for _ in inputs.final_test_sample_ids),
            review_budget=controls.restoration_review_budget,
            sample_ids=inputs.audit_sample_ids,
            class_order=inputs.class_order,
            random_repeats=controls.restoration_random_repeats,
            random_seed=controls.restoration_random_seed,
            model_seed=downstream_model_seed,
            l2=controls.logistic_l2,
            max_iter=controls.logistic_max_iter,
            estimator_factory=estimator_factory,
            reference_validation_features=inputs.reference_validation_features[
                selected_representation
            ],
            reference_validation_labels=inputs.reference_validation_labels,
            reference_validation_group_ids=inputs.reference_validation_group_ids,
            reference_validation_is_injected_corruption=np.zeros(
                len(inputs.reference_validation_sample_ids), dtype=bool
            ),
        )
        cell_restoration_directory = (
            restoration_root if synthetic_layout else restoration_root / selected_cell.cell_id
        )
        cell_restoration_directory.mkdir(parents=True, exist_ok=synthetic_layout)
        restoration_json, restoration_arrays, restoration_manifest = _save_restoration(
            cell_restoration_directory,
            downstream,
            selected_cell=selected_cell,
            execution_controls_binding_sha256=controls.binding_sha256,
            shared_scenario_corruption_hash=shared_corruption_hashes[selected_cell.scenario_id],
            ranking_method=controls.restoration_ranking_method,
            audit_risk_scores=selected_risk,
            audit_sample_ids=inputs.audit_sample_ids,
            audit_group_ids=inputs.audit_group_ids,
            audit_pre_corruption_labels=selected_corruption.pre_corruption_labels,
            audit_observed_labels=selected_corruption.observed_labels,
            audit_is_injected_corruption=selected_corruption.is_injected_corruption,
            final_test_sample_ids=inputs.final_test_sample_ids,
            final_test_group_ids=inputs.final_test_group_ids,
            final_test_labels=inputs.final_test_labels,
            class_order=inputs.class_order,
            review_budget=controls.restoration_review_budget,
            required_experiments=controls.restoration_required_experiments,
            downstream_comparisons=controls.restoration_downstream_comparisons,
        )
        restoration_rows.append(
            {
                "schema_version": 1,
                "cell": asdict(selected_cell),
                "ranking_method": controls.restoration_ranking_method,
                "json_path": str(restoration_json.relative_to(destination)).replace("\\", "/"),
                "json_sha256": sha256_file(restoration_json),
                "evidence_path": str(restoration_arrays.relative_to(destination)).replace(
                    "\\", "/"
                ),
                "evidence_sha256": sha256_file(restoration_arrays),
                "manifest_path": str(restoration_manifest.relative_to(destination)).replace(
                    "\\", "/"
                ),
                "manifest_sha256": sha256_file(restoration_manifest),
            }
        )
    restoration_path = (
        destination / "restoration.json"
        if synthetic_layout
        else atomic_write_json(
            destination / "restoration_index.json",
            {
                "schema_version": 1,
                "execution_controls_binding_sha256": controls.binding_sha256,
                "restoration_cell_ids": list(controls.restoration_cell_ids),
                "restoration_cell_count": len(restoration_rows),
                "downstream_comparisons": [
                    comparison.as_dict()
                    for comparison in controls.restoration_downstream_comparisons
                ],
                "cells": restoration_rows,
            },
        )
    )
    reconciliation = reconcile_primary_cell_outcomes(plan, outcomes)
    reconciliation_path = atomic_write_json(
        destination / "reconciliation.json", reconciliation.as_dict()
    )
    completion = build_primary_completion_evidence(
        plan=plan,
        reconciliation=reconciliation,
        artifact_scope=artifact_scope,
        study_outcome_eligible=False,
    )
    circularity_excluded_cell_ids = sorted(
        str(outcome["cell_id"]) for outcome in outcomes if outcome.get("circularity_risk") is True
    )
    completion.update(
        {
            "execution_controls_binding_sha256": controls.binding_sha256,
            "circularity_excluded_cell_count": len(circularity_excluded_cell_ids),
            "circularity_excluded_cell_ids": circularity_excluded_cell_ids,
            "primary_confirmatory_claims_require_exclusion_of_these_cells": bool(
                circularity_excluded_cell_ids
            ),
        }
    )
    completion_evidence_path = atomic_write_json(
        destination / "completion_evidence.json", completion
    )
    cell_index_rows = [dict(outcome) for outcome in outcomes]
    all_fields: list[str] = []
    for row in cell_index_rows:
        all_fields.extend(field for field in row if field not in all_fields)
    cell_index_path = _write_csv(destination / "cell_index.csv", all_fields, cell_index_rows)
    synthetic = artifact_scope == SYNTHETIC_PRIMARY_ARTIFACT_SCOPE
    title = (
        "# Synthetic primary integration fixture — not PanNuke evidence"
        if synthetic
        else "# Frozen primary matrix core — workflow gate not yet applied"
    )
    scope_note = (
        "This run exercises the reusable matrix, group-safe OOF, risk, paired-bootstrap, "
        "and four-condition restoration contracts. It cannot enable a research stage."
        if synthetic
        else "This real-scope core output remains ineligible until immutable workflow-gate "
        "evidence and sealed-run integrity are verified."
    )
    report_path = atomic_write_text(
        destination / "report.md",
        "\n".join(
            (
                title,
                "",
                scope_note,
                "",
                f"- Matrix cells: {len(plan.cells)}",
                f"- Required cells completed: {reconciliation.completed_required_cell_count}",
                f"- Matrix reconciliation: `{reconciliation.status}`",
                f"- Circularity-excluded cells: {len(circularity_excluded_cell_ids)}",
                f"- Frozen restoration cells: {len(restoration_rows)}",
                "- `study_outcome_eligible`: `false`",
                "- `completion_stage`: `null`",
                "- Final reference groups remained disjoint and uncorrupted.",
                "",
            )
        ),
    )
    return PrimaryMatrixArtifacts(
        output_directory=destination,
        matrix_plan_path=matrix_plan_path,
        execution_controls_path=execution_controls_path,
        cell_index_path=cell_index_path,
        reconciliation_path=reconciliation_path,
        completion_evidence_path=completion_evidence_path,
        restoration_path=restoration_path,
        report_path=report_path,
        outcomes=tuple(outcomes),
        reconciliation=reconciliation,
    )


def _synthetic_fixture_plan() -> PrimaryMatrixPlan:
    scenarios = (
        PrimaryScenario(
            "fixture_sym10",
            "symmetric_random_corruption",
            0.10,
            17,
        ),
        PrimaryScenario(
            "fixture_inst10",
            "instance_dependent_corruption",
            0.10,
            19,
        ),
    )
    cells: list[PrimaryCell] = []
    for scenario in scenarios:
        for representation in (
            "synthetic_colour_statistics",
            "synthetic_colour_projection",
        ):
            for classifier in ("multinomial_logistic_regression", "small_mlp"):
                cells.append(
                    PrimaryCell(
                        cell_id=f"fixture_cell_{len(cells):03d}",
                        scenario_id=scenario.scenario_id,
                        mechanism=scenario.mechanism,
                        rate=scenario.rate,
                        corruption_seed=scenario.corruption_seed,
                        representation_id=representation,
                        classifier_id=classifier,
                        required=True,
                    )
                )
    fixture_config = {
        "artifact_scope": SYNTHETIC_PRIMARY_ARTIFACT_SCOPE,
        "study_outcome_eligible": False,
        "scenarios": [asdict(value) for value in scenarios],
        "cells": [asdict(value) for value in cells],
    }
    return PrimaryMatrixPlan(1, config_sha256(fixture_config), scenarios, tuple(cells))


def _synthetic_fixture_controls(plan: PrimaryMatrixPlan) -> PrimaryExecutionControls:
    """Return sealed CPU controls that can never cross the real-study gate."""

    restoration_cell_id = plan.cells[0].cell_id
    controls = PrimaryExecutionControls(
        frozen_config_schema_version=1,
        frozen_config_canonical_json=None,
        config_semantic_sha256=plan.config_sha256,
        plan=plan,
        plan_sha256=canonical_sha256(plan.as_dict()),
        binding_sha256="",
        dataset_source="synthetic_fixture",
        class_order=(0, 1, 2, 3, 4),
        statistical_group_unit="synthetic_source_group",
        development_official_folds=(0, 1),
        final_test_fold=2,
        reference_validation_fraction_groups=0.10,
        fold_assignment_label_source="pre_corruption_label",
        corruption_rounding_policy="round_half_up",
        n_splits=2,
        split_seed=23,
        oof_split_kind="stratified_group",
        no_nucleus_level_fallback=True,
        logistic_l2=1.0e-2,
        logistic_max_iter=300,
        logistic_class_weight="balanced",
        logistic_class_weight_label_source="observed_development_labels_only",
        logistic_fit_label_source="observed_development_labels_only",
        logistic_model_seed=29,
        mlp_config=FrozenEmbeddingMLPConfig(
            hidden_dimensions=(16,),
            dropout=0.0,
            epochs=1,
            batch_size=32,
            seed=29,
            device="cpu",
        ),
        mlp_fit_label_source="observed_development_labels_only",
        calibration=PrimaryCalibrationControls(
            enabled=False,
            method="none",
            source="reference_validation_only",
            reporting="calibrated_and_uncalibrated",
            fit_labels_policy="observed_reference_validation_labels_only",
            seed=0,
            parameters=(),
        ),
        audit_methods=PRIMARY_AUDIT_METHODS,
        primary_ranking_method="self_confidence",
        neighbour_k=5,
        neighbour_metric="cosine",
        neighbour_exclude_same_group=True,
        fixed_hybrid_components=("self_confidence", "nearest_neighbour_disagreement"),
        fixed_hybrid_weights=(0.5, 0.5),
        cleanlab_failure_policy="missing_with_recorded_blocker",
        primary_metric="average_precision",
        primary_review_budget=0.05,
        secondary_review_budgets=(0.01, 0.10, 0.20),
        random_review_repeats=100,
        random_review_seed=101,
        subgroup_min_samples=100,
        subgroup_min_corruptions=10,
        paired_method_comparisons=(
            PrimaryPairedComparison("hybrid_vs_self_confidence", "fixed_hybrid", "self_confidence"),
        ),
        bootstrap_iterations=2_000,
        bootstrap_seed=211,
        holm_families=(),
        exploratory_multiple_comparison_correction="holm",
        within_cell_comparisons=(),
        method_vs_random_comparisons=(),
        cross_cell_comparisons=(),
        restoration_enabled_cells=(restoration_cell_id,),
        restoration_cell_ids=(restoration_cell_id,),
        restoration_ranking_method="self_confidence",
        restoration_review_budget=0.05,
        restoration_random_repeats=100,
        restoration_random_seed=101,
        restoration_include_reference_validation_in_training=True,
        restoration_required_experiments=(
            "uncorrupted_reference_baseline",
            "corrupted_observed_baseline",
            "random_review_restoration",
            "audit_guided_restoration",
        ),
        restoration_downstream_comparisons=(
            PrimaryDownstreamComparison(
                comparison_id="audit_guided_minus_random_macro_f1",
                method_a="audit_guided_restoration",
                method_b="random_review_restoration",
                metric="macro_f1",
                direction="method_a_minus_method_b",
            ),
        ),
        confusion_transition_matrix=tuple(
            tuple(0.0 if row == column else 0.25 for column in range(5)) for row in range(5)
        ),
        group_conditional_grouping_field="synthetic_group",
        group_conditional_weights_by_value=(("all", 1.0),),
        group_conditional_default_weight=1.0,
        instance_generator_representation="synthetic_target_morphology",
        instance_auditor_representation_families=("synthetic_target_colour",),
        instance_independence_matrix_path="embedded_synthetic_fixture_v1",
        instance_independence_matrix_sha256="0" * 64,
    )
    return replace(controls, binding_sha256=canonical_sha256(controls._binding_payload()))


def run_synthetic_primary_integration_fixture(
    *,
    project_root: str | Path,
    runs_root: str | Path | None = None,
) -> SyntheticPrimaryFixtureResult:
    """Run and seal a reduced primary matrix without claiming PanNuke evidence."""

    root = Path(project_root).resolve()
    plan = _synthetic_fixture_plan()
    tracker = RunTracker.start(
        experiment_name="primary_fixture",
        config={
            "schema_version": 1,
            "experiment_name": "primary_synthetic_integration_fixture",
            "artifact_scope": SYNTHETIC_PRIMARY_ARTIFACT_SCOPE,
            "study_outcome_eligible": False,
            "completion_stage_enabled": None,
            "matrix_config_sha256": plan.config_sha256,
        },
        project_root=root,
        runs_root=runs_root,
        environment={"fixture": True, "cuda_requirement": "not_applicable"},
    )
    with tracker:
        dataset = generate_synthetic_dataset(
            n_groups=18,
            instances_per_group=5,
            patch_size=48,
            seed=2027,
        )
        split = make_outer_audit_split(
            dataset.official_folds,
            tuple(str(value) for value in dataset.group_ids),
            final_test_fold=2,
            reference_validation_fraction=0.10,
            seed=41,
        )
        audit = split.audit_indices
        reference = split.reference_validation_indices
        final = split.final_test_indices
        projection = np.asarray(
            [
                [1.0 if row == column else 0.05 * (row + column + 1) for column in range(9)]
                for row in range(9)
            ],
            dtype=np.float64,
        )
        projected = dataset.audit_features @ projection
        fitted_data_hash = canonical_sha256(
            {
                "sample_ids": tuple(str(dataset.sample_ids[index]) for index in audit),
                "group_ids": tuple(str(dataset.group_ids[index]) for index in audit),
            }
        )
        generator_evidence = FeatureSpaceEvidence.from_array(
            dataset.corruption_features[audit],
            representation_name="synthetic_target_morphology",
            family="synthetic_mask_geometry",
            implementation_hash=semantic_sha256("synthetic_morphology:v1"),
            weights_hash=semantic_sha256("unlearned:no_weights"),
            preprocessing_hash=semantic_sha256("binary_target_mask:v1"),
            fitted_data_hash=fitted_data_hash,
        )
        auditor_evidence = FeatureSpaceEvidence.from_array(
            dataset.audit_features[audit],
            representation_name="synthetic_colour_family",
            family="synthetic_target_colour",
            implementation_hash=semantic_sha256("synthetic_colour:v1"),
            weights_hash=semantic_sha256("unlearned:no_weights"),
            preprocessing_hash=semantic_sha256("target_rgb_statistics:v1"),
            fitted_data_hash=fitted_data_hash,
        )
        independence = FeatureIndependenceEvidence.create(
            matrix_version="synthetic_primary_fixture_v1",
            matrix_decision="verified_independent",
            matrix_reason=(
                "Fixture corruption uses mask geometry while both auditors use colour-only "
                "statistics; the hash-bound arrays and implementations are distinct."
            ),
            generator=generator_evidence,
            auditor=auditor_evidence,
        )
        inputs = PrimaryMatrixInputs(
            audit_sample_ids=tuple(str(dataset.sample_ids[index]) for index in audit),
            audit_group_ids=tuple(str(dataset.group_ids[index]) for index in audit),
            audit_pre_corruption_labels=dataset.pre_corruption_labels[audit],
            audit_features={
                "synthetic_colour_statistics": dataset.audit_features[audit],
                "synthetic_colour_projection": projected[audit],
            },
            reference_validation_sample_ids=tuple(
                str(dataset.sample_ids[index]) for index in reference
            ),
            reference_validation_group_ids=tuple(
                str(dataset.group_ids[index]) for index in reference
            ),
            reference_validation_labels=dataset.pre_corruption_labels[reference],
            reference_validation_features={
                "synthetic_colour_statistics": dataset.audit_features[reference],
                "synthetic_colour_projection": projected[reference],
            },
            final_test_sample_ids=tuple(str(dataset.sample_ids[index]) for index in final),
            final_test_group_ids=tuple(str(dataset.group_ids[index]) for index in final),
            final_test_labels=dataset.pre_corruption_labels[final],
            final_test_features={
                "synthetic_colour_statistics": dataset.audit_features[final],
                "synthetic_colour_projection": projected[final],
            },
            corruption_generator_features=dataset.corruption_features[audit],
            corruption_generator_representation="synthetic_target_morphology",
            corruption_auditor_representation="synthetic_colour_family",
            independence_evidence=independence,
            dataset_seed=2027,
            corruption_timestamp_utc="1970-01-01T00:00:00+00:00",
        )
        restoration_cell_id = plan.cells[0].cell_id
        fixture_controls = _synthetic_fixture_controls(plan)
        artifacts = _execute_primary_matrix_core(
            inputs,
            plan,
            fixture_controls,
            output_directory=tracker.run_directory,
            artifact_scope=SYNTHETIC_PRIMARY_ARTIFACT_SCOPE,
            synthetic_layout=True,
        )
        if not artifacts.reconciliation.passed:
            raise RuntimeError(
                f"synthetic primary matrix reconciliation failed: {artifacts.reconciliation.errors}"
            )
        completion = json.loads(artifacts.completion_evidence_path.read_text(encoding="utf-8"))
        if completion.get("completion_stage") is not None:
            raise RuntimeError("synthetic primary fixture attempted to enable a completion stage")
        metrics_path = tracker.write_metrics(
            {
                "artifact_scope": SYNTHETIC_PRIMARY_ARTIFACT_SCOPE,
                "study_outcome_eligible": False,
                "completion_stage": None,
                "matrix_config_sha256": plan.config_sha256,
                "execution_controls_binding_sha256": fixture_controls.binding_sha256,
                "matrix_cell_count": len(plan.cells),
                "completed_required_cell_count": (
                    artifacts.reconciliation.completed_required_cell_count
                ),
                "reconciliation_status": artifacts.reconciliation.status,
                "restoration_cell_id": restoration_cell_id,
                "final_reference_fold": 2,
                "final_reference_uncorrupted": True,
                "partition_group_overlap_count": 0,
            }
        )
        tracker.write_provenance(
            artifact_scope=SYNTHETIC_PRIMARY_ARTIFACT_SCOPE,
            study_outcome_eligible=False,
            completion_stage_enabled=None,
            matrix_plan_sha256=sha256_file(artifacts.matrix_plan_path),
            execution_controls_sha256=sha256_file(artifacts.execution_controls_path),
            execution_controls_binding_sha256=fixture_controls.binding_sha256,
            reconciliation_sha256=sha256_file(artifacts.reconciliation_path),
            final_reference_uncorrupted=True,
        )
    integrity = verify_run_integrity(tracker.run_directory)
    if not integrity.valid or not integrity.registry_record_present:
        raise RuntimeError(f"sealed synthetic primary fixture failed integrity: {integrity.errors}")
    return SyntheticPrimaryFixtureResult(
        run_id=tracker.run_id,
        run_directory=tracker.run_directory,
        metrics_path=metrics_path,
        report_path=artifacts.report_path,
        completion_evidence_path=artifacts.completion_evidence_path,
        reconciliation_path=artifacts.reconciliation_path,
        matrix_cell_count=len(plan.cells),
    )


__all__ = [
    "PrimaryCalibrationControls",
    "PrimaryCellSelector",
    "PrimaryCrossCellComparison",
    "PrimaryDownstreamComparison",
    "PrimaryExecutionControls",
    "PrimaryMatrixArtifacts",
    "PrimaryMatrixInputs",
    "PrimaryMethodVsRandomComparison",
    "PrimaryPairedComparison",
    "PrimaryWithinCellComparison",
    "SyntheticPrimaryFixtureResult",
    "execute_primary_matrix",
    "primary_execution_controls_from_frozen_config",
    "run_synthetic_primary_integration_fixture",
]
