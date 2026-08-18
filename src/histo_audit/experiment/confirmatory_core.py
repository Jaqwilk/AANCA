"""Strict execution core for the frozen confirmatory matrix.

The core operates on already partitioned, corruption-bound inputs.  It does not
choose a scenario, seed, representation, ensemble member, or model from outcomes.
Final-reference samples, labels, images, and features are deliberately absent from
its OOF requests; only group IDs needed to prove disjointness are exposed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import stat
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from histo_audit.auditing.ensemble import (
    EnsembleDisagreementResult,
    ensemble_disagreement,
    predeclared_ensemble_risk,
)
from histo_audit.auditing.scores import (
    FixedHybridDropOneResult,
    fixed_hybrid_drop_one_ablations,
    score_annotations,
)
from histo_audit.config import config_sha256
from histo_audit.corruption.controlled import (
    array_artifact_sha256,
)
from histo_audit.corruption.controlled import (
    canonical_sha256 as _controlled_canonical_sha256,
)
from histo_audit.cross_validation.image_oof import (
    ConfirmatoryCheckpointContractError,
    ConfirmatoryCheckpointDirective,
    ConfirmatoryCheckpointExecutionContract,
    ConfirmatoryCheckpointPhysicalIdentity,
    ConfirmatoryImageOOFFoldEvidence,
    ConfirmatoryImageOOFResult,
    _checkpoint_file_identity_from_exact_manifest,
    _checkpoint_physical_identity_from_exact_manifest,
    _create_checkpoint_descriptor,
    _hold_private_checkpoint_snapshot,
    _is_reparse,
    _native_identity_from_descriptor,
    _native_identity_from_live_writer_path,
    _native_identity_is_plain_file,
    _PrivateCheckpointSnapshot,
    _require_exact_checkpoint_execution_manifest_payload,
    _same_file_except_read_only_transition,
    _same_native_file,
    _same_open_file_object,
    _set_posix_descriptor_mode,
    _stat_identity,
    _win32_set_read_only_on_descriptor,
    grouped_oof_confirmatory_cnn,
    require_original_confirmatory_checkpoint_authority_binding,
)
from histo_audit.cross_validation.image_oof import (
    _named_streams as _checkpoint_named_streams,
)
from histo_audit.cross_validation.oof import (
    OOFFoldProvenance,
    OOFResult,
    grouped_oof_logistic,
)
from histo_audit.experiment.confirmatory_completion import (
    CONFIRMATORY_ARTIFACT_MANIFEST_FILENAME,
    REAL_CONFIRMATORY_ARTIFACT_SCOPE,
    RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
    SYNTHETIC_CONFIRMATORY_ARTIFACT_SCOPE,
    ConfirmatoryMatrixReconciliation,
    build_confirmatory_completion_evidence,
    reconcile_confirmatory_cell_outcomes,
)
from histo_audit.experiment.confirmatory_memory_workspace import RowIndexedArray
from histo_audit.experiment.study_contracts import (
    CLASS_ORDER,
    RESOURCE_BOUNDED_CONFIRMATORY_CONFIG_SHA256,
    ConfirmatoryCell,
    ConfirmatoryMatrixPlan,
    StudyContractError,
    build_confirmatory_matrix_plan,
    validate_confirmatory_execution_config,
)
from histo_audit.models.cnn import (
    CPU_TEST_ONLY_WEIGHT_IDENTIFIER,
    OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER,
    ConfirmatoryCNNConfig,
)
from histo_audit.statistics.review import average_precision, binary_auroc, rank_indices
from histo_audit.utils.run_tracking import (
    _fsync_directory,
    _json_default,
)

_SHA256_LENGTH = 64
_SUPPORTED_SCENARIO_FAMILIES = {"cnn", "imagenet_frozen", "pathology_frozen"}
_SUPPORTED_HYBRID_COMPONENTS = {"self_confidence", "ensemble_disagreement"}
_SUPPORTED_COMPARISON_RISKS = {
    "self_confidence",
    "ensemble_disagreement",
    "fixed_hybrid",
    "hybrid_drop_self_confidence",
    "hybrid_drop_ensemble_disagreement",
}
ConfirmatoryProgressCallback = Callable[[Mapping[str, Any]], None]
type ConfirmatoryArray = NDArray[np.generic] | RowIndexedArray


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _controlled_array_sha256(array: ConfirmatoryArray) -> str:
    """Preserve the legacy controlled-array digest without full materialisation."""

    if not isinstance(array, RowIndexedArray):
        return array_artifact_sha256(array)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    for chunk in array.iter_chunks():
        digest.update(memoryview(np.ascontiguousarray(chunk)).cast("B"))
    return digest.hexdigest()


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StudyContractError(f"{location} must be a mapping")
    return value


def _require_sequence(value: Any, location: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StudyContractError(f"{location} must be a sequence")
    return value


def _required_value(mapping: Mapping[str, Any], field: str, location: str) -> Any:
    if field not in mapping:
        raise StudyContractError(
            f"{location}.{field} must be explicitly frozen; real execution has no default"
        )
    return mapping[field]


def _emit_confirmatory_progress(
    callback: ConfirmatoryProgressCallback | None,
    *,
    cell: ConfirmatoryCell,
    status: str,
) -> None:
    """Emit outcome-value-free operational telemetry with no timing duration."""

    if callback is None:
        return
    if status not in {"started", "model_completed", "failed", "skipped"}:
        raise ValueError(f"unsupported confirmatory progress status: {status!r}")
    callback(
        MappingProxyType(
            {
                "cell_id": cell.cell_id,
                "scenario": cell.scenario_id,
                "fold": cell.outer_fold,
                "corruption": cell.corruption_cell_id,
                "seed": cell.model_seed,
                "status": status,
                "timestamp": datetime.now(UTC).isoformat(),
                "telemetry_contract": "outcome_value_free_operational_telemetry",
                "prohibited_for_selection_tuning": True,
                "adaptive_execution_changes_allowed": False,
            }
        )
    )


@dataclass(frozen=True, slots=True)
class ConfirmatoryCorruptionSpec:
    """One exact corruption cell from the frozen configuration."""

    corruption_cell_id: str
    mechanism: str
    rate: float
    seed: int
    parameters: Mapping[str, Any]
    parameters_sha256: str


@dataclass(frozen=True, slots=True)
class ConfirmatoryScenarioSpec:
    """One exact model/representation scenario from the frozen configuration."""

    scenario_id: str
    family: str
    input_variant: str
    encoder: str
    classifier: str
    representation_id: str
    cache_provenance_id: str
    required: bool
    availability_audit_sha256: str | None


@dataclass(frozen=True, slots=True)
class ConfirmatoryEnsembleMember:
    """A preregistered scenario/seed member of every rotation-level ensemble."""

    scenario_id: str
    model_seed: int


@dataclass(frozen=True, slots=True)
class ConfirmatoryComparisonOperand:
    """One exact scenario/risk selector used by a frozen paired comparison."""

    scenario_id: str
    representation_id: str
    classifier_id: str
    risk_id: str
    model_seed: str
    outer_fold: int | str
    corruption_cell: str


@dataclass(frozen=True, slots=True)
class ConfirmatoryPairedComparison:
    """A preregistered A-minus-B comparison with no inferred selectors."""

    comparison_id: str
    metric: str
    operand_a: ConfirmatoryComparisonOperand
    operand_b: ConfirmatoryComparisonOperand
    direction: str
    holm_family: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "metric": self.metric,
            "operand_a": asdict(self.operand_a),
            "operand_b": asdict(self.operand_b),
            "direction": self.direction,
            "holm_family": self.holm_family,
        }


@dataclass(frozen=True, slots=True)
class ConfirmatoryExecutionControls:
    """Immutable executable controls derived from one validated confirmatory config."""

    config_semantic_sha256: str
    plan: ConfirmatoryMatrixPlan
    binding_sha256: str
    official_folds: tuple[int, ...]
    statistical_group_unit: str
    split_seed: int
    n_splits: int
    corruption_specs: tuple[ConfirmatoryCorruptionSpec, ...]
    scenario_specs: tuple[ConfirmatoryScenarioSpec, ...]
    cache_provenance_records: tuple[Mapping[str, Any], ...]
    model_seeds: tuple[int, ...]
    input_size: int
    gradient_accumulation_steps: int
    class_weight: str
    learning_rate: float
    weight_decay: float
    max_epochs: int
    initial_batch_size: int
    minimum_batch_size: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    amp_dtype: str
    ensemble_members: tuple[ConfirmatoryEnsembleMember, ...]
    ensemble_primary_risk: str
    ensemble_secondary_risks: tuple[str, ...]
    hybrid_components: tuple[str, ...]
    hybrid_weights: tuple[float, ...]
    hybrid_drop_one_ablations: tuple[str, ...]
    restoration_scenario_id: str
    restoration_model_seed: int
    restoration_representation_id: str
    restoration_ranking_method: str
    restoration_review_budget: float
    restoration_random_repeats: int
    restoration_random_seed: int
    restoration_conditions: tuple[str, ...]
    original_audit_selection: Mapping[str, Any]
    paired_group_bootstrap_iterations: int
    bootstrap_seed: int
    paired_comparisons: tuple[ConfirmatoryPairedComparison, ...]
    holm_families: tuple[str, ...]

    @property
    def scenarios_by_id(self) -> dict[str, ConfirmatoryScenarioSpec]:
        return {item.scenario_id: item for item in self.scenario_specs}

    @property
    def corruptions_by_id(self) -> dict[str, ConfirmatoryCorruptionSpec]:
        return {item.corruption_cell_id: item for item in self.corruption_specs}

    @property
    def cache_provenance_by_id(self) -> dict[str, Mapping[str, Any]]:
        return {str(item["id"]): item for item in self.cache_provenance_records}

    def _binding_payload(self) -> dict[str, Any]:
        payload = {
            "config_semantic_sha256": self.config_semantic_sha256,
            "plan": self.plan.as_dict(),
            "official_folds": self.official_folds,
            "statistical_group_unit": self.statistical_group_unit,
            "split_seed": self.split_seed,
            "n_splits": self.n_splits,
            "corruption_specs": [asdict(item) for item in self.corruption_specs],
            "scenario_specs": [asdict(item) for item in self.scenario_specs],
            "cache_provenance_records": [dict(item) for item in self.cache_provenance_records],
            "model_seeds": self.model_seeds,
            "training": {
                "input_size": self.input_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "class_weight": self.class_weight,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "max_epochs": self.max_epochs,
                "initial_batch_size": self.initial_batch_size,
                "minimum_batch_size": self.minimum_batch_size,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
                "amp_dtype": self.amp_dtype,
            },
            "ensemble_members": [asdict(item) for item in self.ensemble_members],
            "ensemble_primary_risk": self.ensemble_primary_risk,
            "ensemble_secondary_risks": self.ensemble_secondary_risks,
            "hybrid_components": self.hybrid_components,
            "hybrid_weights": self.hybrid_weights,
            "hybrid_drop_one_ablations": self.hybrid_drop_one_ablations,
            "restoration": {
                "scenario_id": self.restoration_scenario_id,
                "model_seed": self.restoration_model_seed,
                "representation_id": self.restoration_representation_id,
                "ranking_method": self.restoration_ranking_method,
                "review_budget": self.restoration_review_budget,
                "random_repeats": self.restoration_random_repeats,
                "random_seed": self.restoration_random_seed,
                "conditions": self.restoration_conditions,
            },
            "original_audit_selection": dict(self.original_audit_selection),
            "paired_group_bootstrap_iterations": self.paired_group_bootstrap_iterations,
            "bootstrap_seed": self.bootstrap_seed,
            "paired_comparisons": [item.as_dict() for item in self.paired_comparisons],
            "holm_families": self.holm_families,
        }
        if self.plan.schema_version == 3:
            payload.update(
                execution_profile="resource_bounded_confirmatory_v1",
                analysis_disposition="amended_or_exploratory",
                original_confirmatory_claim_allowed=False,
                completion_stage=None,
            )
        return payload

    def validate_for_plan(self, plan: ConfirmatoryMatrixPlan) -> None:
        if self.plan != plan:
            raise ValueError("confirmatory controls were derived for a different matrix plan")
        if self.config_semantic_sha256 != plan.config_sha256:
            raise ValueError("confirmatory config SHA does not match the matrix plan")
        if self.binding_sha256 != _canonical_sha256(self._binding_payload()):
            raise ValueError("confirmatory execution-control binding SHA is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source": (
                "validated_frozen_confirmatory_config_schema_v2"
                if self.plan.schema_version == 2
                else "validated_resource_bounded_confirmatory_v1_schema_v3"
            ),
            "binding_sha256": self.binding_sha256,
            **self._binding_payload(),
        }


def _comparison_operand(value: Any, location: str) -> ConfirmatoryComparisonOperand:
    operand = _require_mapping(value, location)
    outer_fold = _required_value(operand, "outer_fold", location)
    if not isinstance(outer_fold, (int, str)) or isinstance(outer_fold, bool):
        raise StudyContractError(f"{location}.outer_fold must be an integer or all_matched")
    return ConfirmatoryComparisonOperand(
        scenario_id=str(_required_value(operand, "scenario_id", location)),
        representation_id=str(_required_value(operand, "representation_id", location)),
        classifier_id=str(_required_value(operand, "classifier_id", location)),
        risk_id=str(_required_value(operand, "risk_id", location)),
        model_seed=str(_required_value(operand, "model_seed", location)),
        outer_fold=outer_fold,
        corruption_cell=str(_required_value(operand, "corruption_cell", location)),
    )


def _paired_comparison(value: Any, location: str) -> ConfirmatoryPairedComparison:
    comparison = _require_mapping(value, location)
    return ConfirmatoryPairedComparison(
        comparison_id=str(_required_value(comparison, "comparison_id", location)),
        metric=str(_required_value(comparison, "metric", location)),
        operand_a=_comparison_operand(
            _required_value(comparison, "operand_a", location), f"{location}.operand_a"
        ),
        operand_b=_comparison_operand(
            _required_value(comparison, "operand_b", location), f"{location}.operand_b"
        ),
        direction=str(_required_value(comparison, "direction", location)),
        holm_family=str(_required_value(comparison, "holm_family", location)),
    )


def confirmatory_execution_controls_from_frozen_config(
    config: Mapping[str, Any],
) -> ConfirmatoryExecutionControls:
    """Validate and bind every executable confirmatory choice without defaults."""

    resolved = validate_confirmatory_execution_config(config)
    plan = build_confirmatory_matrix_plan(resolved)
    semantic_sha = config_sha256(resolved)
    if semantic_sha != plan.config_sha256:
        raise RuntimeError("confirmatory plan/config semantic SHA mismatch")

    data = _require_mapping(resolved["data"], "data")
    training = _require_mapping(resolved["training"], "training")
    oof = _require_mapping(resolved["oof"], "oof")
    ensemble = _require_mapping(resolved["ensemble"], "ensemble")
    hybrid = _require_mapping(resolved["fixed_hybrid"], "fixed_hybrid")
    restoration = _require_mapping(resolved["restoration"], "restoration")
    original_audit_selection = _require_mapping(
        _required_value(resolved, "original_audit_selection", "config"),
        "original_audit_selection",
    )
    statistics = _require_mapping(resolved["statistics"], "statistics")

    corruption_specs: list[ConfirmatoryCorruptionSpec] = []
    corruption = _require_mapping(resolved["corruption"], "corruption")
    for index, raw in enumerate(_require_sequence(corruption["cells"], "corruption.cells")):
        cell = _require_mapping(raw, f"corruption.cells[{index}]")
        parameters = dict(
            _require_mapping(
                _required_value(cell, "parameters", f"corruption.cells[{index}]"),
                f"corruption.cells[{index}].parameters",
            )
        )
        corruption_specs.append(
            ConfirmatoryCorruptionSpec(
                corruption_cell_id=str(cell["id"]),
                mechanism=str(cell["mechanism"]),
                rate=float(cell["rate"]),
                seed=int(cell["seed"]),
                parameters=parameters,
                parameters_sha256=_canonical_sha256(parameters),
            )
        )

    scenario_specs: list[ConfirmatoryScenarioSpec] = []
    for index, raw in enumerate(_require_sequence(resolved["scenarios"], "scenarios")):
        scenario = _require_mapping(raw, f"scenarios[{index}]")
        family = str(scenario["family"])
        if family not in _SUPPORTED_SCENARIO_FAMILIES:
            raise StudyContractError(f"unsupported confirmatory scenario family: {family!r}")
        availability = scenario.get("availability_audit_sha256")
        scenario_specs.append(
            ConfirmatoryScenarioSpec(
                scenario_id=str(scenario["id"]),
                family=family,
                input_variant=str(scenario["input_variant"]),
                encoder=str(scenario["encoder"]),
                classifier=str(scenario["classifier"]),
                representation_id=str(
                    _required_value(scenario, "representation_id", f"scenarios[{index}]")
                ),
                cache_provenance_id=str(
                    _required_value(scenario, "cache_provenance_id", f"scenarios[{index}]")
                ),
                required=bool(scenario["required"]),
                availability_audit_sha256=(str(availability) if availability is not None else None),
            )
        )

    members = tuple(
        ConfirmatoryEnsembleMember(
            scenario_id=str(_require_mapping(raw, "ensemble.members[]")["scenario_id"]),
            model_seed=int(_require_mapping(raw, "ensemble.members[]")["model_seed"]),
        )
        for raw in _require_sequence(
            _required_value(ensemble, "members", "ensemble"), "ensemble.members"
        )
    )
    if len(members) < 2 or len(set(members)) != len(members):
        raise StudyContractError("ensemble.members must contain at least two unique members")

    components = tuple(str(value) for value in hybrid["components"])
    if set(components).difference(_SUPPORTED_HYBRID_COMPONENTS):
        raise StudyContractError(
            "confirmatory core supports only frozen self_confidence and "
            "ensemble_disagreement hybrid components"
        )
    controls = ConfirmatoryExecutionControls(
        config_semantic_sha256=semantic_sha,
        plan=plan,
        binding_sha256="",
        official_folds=tuple(int(value) for value in data["official_folds"]),
        statistical_group_unit=str(_required_value(data, "group_unit", "data")),
        split_seed=int(data["split_seed"]),
        n_splits=int(oof["n_splits"]),
        corruption_specs=tuple(corruption_specs),
        scenario_specs=tuple(scenario_specs),
        cache_provenance_records=tuple(
            dict(_require_mapping(value, "cache_provenance[]"))
            for value in _require_sequence(
                _required_value(resolved, "cache_provenance", "config"),
                "cache_provenance",
            )
        ),
        model_seeds=tuple(int(value) for value in resolved["model_seeds"]),
        input_size=int(_required_value(training, "input_size", "training")),
        gradient_accumulation_steps=int(
            _required_value(training, "gradient_accumulation_steps", "training")
        ),
        class_weight=str(_required_value(training, "class_weight", "training")),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        max_epochs=int(training["max_epochs"]),
        initial_batch_size=int(training["initial_batch_size"]),
        minimum_batch_size=int(training["minimum_batch_size"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
        early_stopping_min_delta=float(training["early_stopping_min_delta"]),
        amp_dtype=str(training["amp_dtype"]),
        ensemble_members=members,
        ensemble_primary_risk=str(ensemble["primary_risk"]),
        ensemble_secondary_risks=tuple(str(value) for value in ensemble["secondary_risks"]),
        hybrid_components=components,
        hybrid_weights=tuple(float(value) for value in hybrid["weights"]),
        hybrid_drop_one_ablations=tuple(str(value) for value in hybrid["drop_one_ablations"]),
        restoration_scenario_id=str(_required_value(restoration, "scenario_id", "restoration")),
        restoration_model_seed=int(_required_value(restoration, "model_seed", "restoration")),
        restoration_representation_id=str(
            _required_value(restoration, "representation_id", "restoration")
        ),
        restoration_ranking_method=str(restoration["ranking_method"]),
        restoration_review_budget=float(
            _required_value(restoration, "review_budget", "restoration")
        ),
        restoration_random_repeats=int(
            _required_value(restoration, "random_repeats", "restoration")
        ),
        restoration_random_seed=int(_required_value(restoration, "random_seed", "restoration")),
        restoration_conditions=tuple(
            str(value)
            for value in _require_sequence(
                _required_value(restoration, "conditions", "restoration"),
                "restoration.conditions",
            )
        ),
        original_audit_selection=dict(original_audit_selection),
        paired_group_bootstrap_iterations=int(
            _required_value(
                statistics,
                "paired_group_bootstrap_iterations",
                "statistics",
            )
        ),
        bootstrap_seed=int(_required_value(statistics, "bootstrap_seed", "statistics")),
        paired_comparisons=tuple(
            _paired_comparison(
                value,
                f"statistics.preregistered_paired_comparisons[{index}]",
            )
            for index, value in enumerate(
                _require_sequence(
                    _required_value(
                        statistics,
                        "preregistered_paired_comparisons",
                        "statistics",
                    ),
                    "statistics.preregistered_paired_comparisons",
                )
            )
        ),
        holm_families=tuple(
            str(value)
            for value in _require_sequence(
                _required_value(statistics, "holm_families", "statistics"),
                "statistics.holm_families",
            )
        ),
    )
    controls = dataclass_replace_binding(controls)
    controls.validate_for_plan(plan)
    _validate_controls(controls)
    return controls


def dataclass_replace_binding(
    controls: ConfirmatoryExecutionControls,
) -> ConfirmatoryExecutionControls:
    """Return controls with their self-binding digest (kept separate for typing clarity)."""

    from dataclasses import replace

    return replace(controls, binding_sha256=_canonical_sha256(controls._binding_payload()))


def _validate_controls(controls: ConfirmatoryExecutionControls) -> None:
    scenario_ids = set(controls.scenarios_by_id)
    seeds = set(controls.model_seeds)
    if not controls.statistical_group_unit.strip():
        raise StudyContractError("confirmatory statistical group unit must be explicit")
    for member in controls.ensemble_members:
        if member.scenario_id not in scenario_ids or member.model_seed not in seeds:
            raise StudyContractError("ensemble member is absent from the frozen matrix")
        if not controls.scenarios_by_id[member.scenario_id].required:
            raise StudyContractError("optional scenarios cannot be mandatory ensemble members")
    restoration = controls.scenarios_by_id.get(controls.restoration_scenario_id)
    if restoration is None or controls.restoration_model_seed not in seeds:
        raise StudyContractError("restoration scenario/model seed is absent from the matrix")
    if restoration.representation_id != controls.restoration_representation_id:
        raise StudyContractError("restoration representation differs from its scenario")
    if controls.restoration_ranking_method != "fixed_hybrid":
        raise StudyContractError("confirmatory restoration ranking must be fixed_hybrid")
    expected_restoration_conditions = {
        "uncorrupted_reference_baseline",
        "corrupted_observed_baseline",
        "random_review_restoration",
        "audit_guided_restoration",
    }
    if set(controls.restoration_conditions) != expected_restoration_conditions:
        raise StudyContractError("confirmatory restoration conditions differ from the frozen four")
    if controls.restoration_random_seed < 0:
        raise StudyContractError("confirmatory restoration random seed must be non-negative")
    if set(controls.hybrid_drop_one_ablations) != set(controls.hybrid_components):
        raise StudyContractError("hybrid drop-one list must exactly cover frozen components")
    if controls.class_weight != "balanced":
        raise StudyContractError("confirmatory class_weight must be explicitly balanced")
    cache_records = controls.cache_provenance_by_id
    if len(cache_records) != len(controls.cache_provenance_records):
        raise StudyContractError("confirmatory cache-provenance IDs must be unique")
    for scenario in controls.scenario_specs:
        record = cache_records.get(scenario.cache_provenance_id)
        if record is None or record.get("representation_id") != scenario.representation_id:
            raise StudyContractError("scenario cache provenance is absent or misbound")
    selection_cache_id = str(controls.original_audit_selection["cache_provenance_id"])
    selection_scenario = controls.scenarios_by_id[
        str(controls.original_audit_selection["scenario_id"])
    ]
    if selection_cache_id != selection_scenario.cache_provenance_id:
        raise StudyContractError("original-audit selection cache record differs from scenario")
    comparison_ids: set[str] = set()
    for comparison in controls.paired_comparisons:
        if comparison.comparison_id in comparison_ids:
            raise StudyContractError("confirmatory paired comparison IDs must be unique")
        comparison_ids.add(comparison.comparison_id)
        if (
            comparison.metric not in {"average_precision", "macro_f1"}
            or comparison.direction != "method_a_minus_method_b"
            or comparison.holm_family not in controls.holm_families
        ):
            raise StudyContractError("confirmatory paired comparison metadata is invalid")
        for operand in (comparison.operand_a, comparison.operand_b):
            operand_scenario = controls.scenarios_by_id.get(operand.scenario_id)
            if (
                operand_scenario is None
                or operand.representation_id != operand_scenario.representation_id
                or operand.classifier_id != operand_scenario.classifier
                or operand.risk_id not in _SUPPORTED_COMPARISON_RISKS
                or operand.model_seed != "matched"
                or (
                    operand.outer_fold != "all_matched"
                    and operand.outer_fold not in controls.official_folds
                )
                or (
                    operand.corruption_cell != "all_matched"
                    and operand.corruption_cell not in controls.corruptions_by_id
                )
            ):
                raise StudyContractError("confirmatory paired comparison operand is misbound")
        if (
            comparison.operand_a.outer_fold != comparison.operand_b.outer_fold
            or comparison.operand_a.corruption_cell != comparison.operand_b.corruption_cell
        ):
            raise StudyContractError("confirmatory paired comparison selectors are not matched")


@dataclass(frozen=True, slots=True)
class ConfirmatoryCorruptionInput:
    """Observed audit labels and immutable corruption identity for one cell."""

    corruption_cell_id: str
    mechanism: str
    rate: float
    seed: int
    parameters: Mapping[str, Any]
    pre_corruption_labels: NDArray[np.int64]
    observed_labels: NDArray[np.int64]
    is_injected_corruption: NDArray[np.bool_]

    def validate(self, spec: ConfirmatoryCorruptionSpec, n_samples: int) -> None:
        if (
            self.corruption_cell_id != spec.corruption_cell_id
            or self.mechanism != spec.mechanism
            or self.rate != spec.rate
            or self.seed != spec.seed
            or _canonical_sha256(dict(self.parameters)) != spec.parameters_sha256
        ):
            raise ValueError(f"corruption input {self.corruption_cell_id!r} differs from plan")
        pre = np.asarray(self.pre_corruption_labels)
        observed = np.asarray(self.observed_labels)
        injected = np.asarray(self.is_injected_corruption)
        if pre.shape != (n_samples,) or observed.shape != (n_samples,):
            raise ValueError("corruption labels must align with the audit pool")
        if not np.issubdtype(pre.dtype, np.integer) or not np.issubdtype(
            observed.dtype, np.integer
        ):
            raise ValueError("corruption labels must be integer vectors")
        if injected.shape != (n_samples,) or not np.issubdtype(injected.dtype, np.bool_):
            raise ValueError("is_injected_corruption must be an aligned boolean vector")
        changed = observed != pre
        if not np.array_equal(changed, injected):
            raise ValueError("is_injected_corruption must exactly identify changed labels")
        expected_count = int(np.floor(n_samples * spec.rate + 0.5))
        if int(injected.sum()) != expected_count:
            raise ValueError("corruption count differs from frozen round-half-up rate")
        if any(int(value) not in CLASS_ORDER for value in np.concatenate((pre, observed))):
            raise ValueError("corruption labels lie outside the fixed class order")


@dataclass(frozen=True, slots=True)
class FrozenFeatureProvenance:
    """Cryptographic identity of one frozen feature cache used by a rotation."""

    cache_provenance_id: str
    representation_id: str
    cache_file_sha256: str | None
    sidecar_semantic_sha256: str | None
    sample_order_sha256: str
    manifest_sha256: str
    encoder_identifier: str
    encoder_metadata_sha256: str
    weight_identifier: str
    weights_sha256: str
    preprocessing_identifier: str
    preprocessing_sha256: str
    input_variant: str
    audit_sample_order_sha256: str

    def validate(
        self,
        *,
        representation_id: str,
        audit_sample_ids: Sequence[str],
    ) -> None:
        if self.representation_id != representation_id:
            raise ValueError("frozen-feature provenance representation ID differs from cache")
        if not self.cache_provenance_id.strip():
            raise ValueError("frozen-feature cache provenance ID must be non-empty")
        cache_sha_valid = _valid_sha256_or_none(self.cache_file_sha256)
        sidecar_sha_valid = _valid_sha256_or_none(self.sidecar_semantic_sha256)
        if cache_sha_valid == sidecar_sha_valid:
            raise ValueError(
                "frozen-feature provenance requires exactly one cache-file or sidecar SHA"
            )
        for field in (
            "sample_order_sha256",
            "manifest_sha256",
            "encoder_metadata_sha256",
            "weights_sha256",
            "preprocessing_sha256",
            "audit_sample_order_sha256",
        ):
            value = str(getattr(self, field))
            if len(value) != _SHA256_LENGTH or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"frozen-feature provenance {field} is not a SHA-256")
        for field in (
            "encoder_identifier",
            "weight_identifier",
            "preprocessing_identifier",
            "input_variant",
        ):
            if not str(getattr(self, field)).strip():
                raise ValueError(f"frozen-feature provenance {field} must be non-empty")
        expected_audit_order = _canonical_sha256([str(value) for value in audit_sample_ids])
        if self.audit_sample_order_sha256 != expected_audit_order:
            raise ValueError("frozen-feature provenance audit sample order SHA is invalid")

    @property
    def semantic_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


def _valid_sha256_or_none(value: str | None) -> bool:
    return (
        value is not None
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class ConfirmatoryRotationInputs:
    """One official-fold rotation with no final pixels/features exposed to OOF."""

    outer_fold: int
    audit_sample_ids: tuple[str, ...]
    audit_group_ids: tuple[str, ...]
    audit_official_folds: NDArray[np.int64]
    audit_rgb: ConfirmatoryArray
    audit_target_masks: ConfirmatoryArray
    corruptions: Mapping[str, ConfirmatoryCorruptionInput]
    frozen_audit_features: Mapping[str, NDArray[np.float64] | RowIndexedArray]
    frozen_feature_provenance: Mapping[str, FrozenFeatureProvenance]
    reference_validation_sample_ids: tuple[str, ...]
    reference_validation_group_ids: tuple[str, ...]
    reference_validation_official_folds: NDArray[np.int64]
    reference_validation_labels: NDArray[np.int64]
    reference_validation_rgb: ConfirmatoryArray
    reference_validation_target_masks: ConfirmatoryArray
    final_sample_ids: tuple[str, ...]
    final_group_ids: tuple[str, ...]
    final_official_folds: NDArray[np.int64]
    final_pre_corruption_labels: NDArray[np.int64]
    final_observed_labels: NDArray[np.int64]
    final_is_injected_corruption: NDArray[np.bool_]

    def validate(self, controls: ConfirmatoryExecutionControls) -> None:
        if self.outer_fold not in controls.official_folds:
            raise ValueError("rotation outer_fold is absent from frozen official folds")
        partitions = (
            ("audit", self.audit_sample_ids, self.audit_group_ids),
            (
                "reference validation",
                self.reference_validation_sample_ids,
                self.reference_validation_group_ids,
            ),
            ("final", self.final_sample_ids, self.final_group_ids),
        )
        sample_sets: list[set[str]] = []
        group_sets: list[set[str]] = []
        for name, samples, groups in partitions:
            if not samples or len(samples) != len(groups) or len(set(samples)) != len(samples):
                raise ValueError(f"{name} sample/group identities are empty or misaligned")
            if any(not value for value in samples + groups):
                raise ValueError(f"{name} sample/group identities must be non-empty")
            sample_sets.append(set(samples))
            group_sets.append(set(groups))
        for left in range(3):
            for right in range(left + 1, 3):
                if sample_sets[left].intersection(sample_sets[right]):
                    raise ValueError("sample overlap across confirmatory partitions")
                if group_sets[left].intersection(group_sets[right]):
                    raise ValueError("source-group overlap across confirmatory partitions")

        n_audit = len(self.audit_sample_ids)
        n_validation = len(self.reference_validation_sample_ids)
        n_final = len(self.final_sample_ids)
        _validate_official_folds(
            self.audit_official_folds,
            n_audit,
            allowed=set(controls.official_folds).difference({self.outer_fold}),
            name="audit_official_folds",
        )
        _validate_official_folds(
            self.reference_validation_official_folds,
            n_validation,
            allowed=set(controls.official_folds).difference({self.outer_fold}),
            name="reference_validation_official_folds",
        )
        _validate_official_folds(
            self.final_official_folds,
            n_final,
            allowed={self.outer_fold},
            name="final_official_folds",
        )
        _validate_images(self.audit_rgb, self.audit_target_masks, n_audit, "audit")
        _validate_images(
            self.reference_validation_rgb,
            self.reference_validation_target_masks,
            n_validation,
            "reference validation",
        )
        _validate_labels(self.reference_validation_labels, n_validation, "reference labels")
        _validate_labels(self.final_pre_corruption_labels, n_final, "final pre-corruption labels")
        _validate_labels(self.final_observed_labels, n_final, "final observed labels")
        final_injected = np.asarray(self.final_is_injected_corruption)
        if final_injected.shape != (n_final,) or final_injected.any():
            raise ValueError("final reference must be wholly uncorrupted")
        if not np.array_equal(self.final_pre_corruption_labels, self.final_observed_labels):
            raise ValueError("final observed labels differ from untouched reference labels")

        expected_corruptions = set(controls.corruptions_by_id)
        if set(self.corruptions) != expected_corruptions:
            raise ValueError("rotation corruption inputs do not exactly match frozen cells")
        common_pre: NDArray[np.int64] | None = None
        for corruption_id, corruption in self.corruptions.items():
            corruption.validate(controls.corruptions_by_id[corruption_id], n_audit)
            pre = np.asarray(corruption.pre_corruption_labels, dtype=np.int64)
            if common_pre is None:
                common_pre = pre
            elif not np.array_equal(common_pre, pre):
                raise ValueError("corruption cells disagree on pre_corruption_label")
        if common_pre is None or set(common_pre.tolist()) != set(CLASS_ORDER):
            raise ValueError("audit pre_corruption_label must contain all five classes")
        for representation_id, raw in self.frozen_audit_features.items():
            if isinstance(raw, RowIndexedArray):
                valid = (
                    raw.ndim == 2
                    and raw.shape[0] == n_audit
                    and bool(raw.shape[1])
                    and all(np.isfinite(chunk).all() for chunk in raw.iter_chunks())
                )
            else:
                features = np.asarray(raw)
                valid = (
                    features.ndim == 2
                    and features.shape[0] == n_audit
                    and bool(features.shape[1])
                    and bool(np.isfinite(features).all())
                )
            if not valid:
                raise ValueError(f"frozen feature cache {representation_id!r} is invalid")
        if not set(self.frozen_audit_features).issubset(self.frozen_feature_provenance):
            raise ValueError("every frozen feature cache requires a provenance record")
        for representation_id, provenance in self.frozen_feature_provenance.items():
            provenance.validate(
                representation_id=representation_id,
                audit_sample_ids=self.audit_sample_ids,
            )
        scenarios_by_representation = {
            scenario.representation_id: scenario for scenario in controls.scenario_specs
        }
        frozen_scenarios = {
            scenario.representation_id: scenario
            for scenario in controls.scenario_specs
            if scenario.family != "cnn"
        }
        unknown_representations = set(self.frozen_audit_features).difference(frozen_scenarios)
        if unknown_representations:
            raise ValueError(
                "rotation contains unplanned frozen representations: "
                f"{sorted(unknown_representations)}"
            )
        missing_required = {
            scenario.representation_id
            for scenario in frozen_scenarios.values()
            if scenario.required
        }.difference(self.frozen_audit_features)
        if missing_required:
            raise ValueError(
                f"rotation lacks required frozen representations: {sorted(missing_required)}"
            )
        expected_available_provenance = {
            scenario.representation_id
            for scenario in controls.scenario_specs
            if controls.cache_provenance_by_id[scenario.cache_provenance_id]["status"]
            == "available"
        }
        if set(self.frozen_feature_provenance) != expected_available_provenance:
            raise ValueError(
                "rotation cache provenance does not exactly cover every available scenario"
            )
        for representation_id, provenance in self.frozen_feature_provenance.items():
            scenario = scenarios_by_representation[representation_id]
            if provenance.cache_provenance_id != scenario.cache_provenance_id:
                raise ValueError("frozen-feature provenance ID differs from its scenario")
            if provenance.encoder_identifier != scenario.encoder or (
                provenance.input_variant != scenario.input_variant
            ):
                raise ValueError(
                    "frozen-feature encoder/input provenance differs from its scenario"
                )
            frozen_record = controls.cache_provenance_by_id[scenario.cache_provenance_id]
            expected = {
                "cache_provenance_id": str(frozen_record["id"]),
                "representation_id": str(frozen_record["representation_id"]),
                "cache_file_sha256": frozen_record["cache_file_sha256"],
                "sidecar_semantic_sha256": frozen_record["sidecar_semantic_sha256"],
                "sample_order_sha256": str(frozen_record["sample_order_sha256"]),
                "manifest_sha256": str(frozen_record["manifest_sha256"]),
                "encoder_identifier": str(frozen_record["encoder_identifier"]),
                "encoder_metadata_sha256": str(frozen_record["encoder_metadata_sha256"]),
                "weight_identifier": str(frozen_record["weight_identifier"]),
                "weights_sha256": str(frozen_record["weights_sha256"]),
                "preprocessing_identifier": str(frozen_record["preprocessing_identifier"]),
                "preprocessing_sha256": str(frozen_record["preprocessing_sha256"]),
                "input_variant": str(frozen_record["input_variant"]),
            }
            actual = asdict(provenance)
            actual.pop("audit_sample_order_sha256")
            if actual != expected:
                raise ValueError("rotation cache provenance differs from frozen record")


def _validate_official_folds(
    values: NDArray[np.generic], n: int, *, allowed: set[int], name: str
) -> None:
    array = np.asarray(values)
    if array.shape != (n,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be an aligned integer vector")
    unexpected = set(int(value) for value in array).difference(allowed)
    if unexpected:
        raise ValueError(f"{name} contains forbidden outer-fold values: {sorted(unexpected)}")


def _validate_labels(values: NDArray[np.generic], n: int, name: str) -> None:
    array = np.asarray(values)
    if array.shape != (n,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be an aligned integer vector")
    if set(int(value) for value in array).difference(CLASS_ORDER):
        raise ValueError(f"{name} contains values outside fixed class order")


def _validate_images(
    rgb: ConfirmatoryArray,
    masks: ConfirmatoryArray,
    n: int,
    name: str,
) -> None:
    images = rgb if isinstance(rgb, RowIndexedArray) else np.asarray(rgb)
    mask_array = masks if isinstance(masks, RowIndexedArray) else np.asarray(masks)
    if images.ndim != 4 or images.shape[0] != n or images.shape[-1] != 3:
        raise ValueError(f"{name} RGB cache must have shape (n, height, width, 3)")
    if mask_array.shape != images.shape[:3]:
        raise ValueError(f"{name} target masks must be aligned and binary")
    chunks = mask_array.iter_chunks() if isinstance(mask_array, RowIndexedArray) else (mask_array,)
    for chunk in chunks:
        if not np.isin(chunk, (0, 1)).all():
            raise ValueError(f"{name} target masks must be aligned and binary")
        if not chunk.reshape(len(chunk), -1).any(axis=1).all():
            raise ValueError(f"every {name} target mask must be non-empty")


@dataclass(frozen=True, slots=True)
class ConfirmatoryFrozenBlocker:
    """A pre-execution optional-scenario blocker bound to the frozen audit hash."""

    scenario_id: str
    config_semantic_sha256: str
    availability_audit_sha256: str
    blocker: str


def _read_only_view(values: ConfirmatoryArray) -> ConfirmatoryArray:
    if isinstance(values, RowIndexedArray):
        return values
    view = np.asarray(values).view()
    view.setflags(write=False)
    return view


def _runner_corruption_input(
    source: ConfirmatoryCorruptionInput,
) -> ConfirmatoryCorruptionInput:
    return ConfirmatoryCorruptionInput(
        corruption_cell_id=source.corruption_cell_id,
        mechanism=source.mechanism,
        rate=source.rate,
        seed=source.seed,
        parameters=MappingProxyType(dict(source.parameters)),
        pre_corruption_labels=cast(
            NDArray[np.int64], _read_only_view(source.pre_corruption_labels)
        ),
        observed_labels=cast(NDArray[np.int64], _read_only_view(source.observed_labels)),
        is_injected_corruption=cast(
            NDArray[np.bool_], _read_only_view(source.is_injected_corruption)
        ),
    )


@dataclass(frozen=True, slots=True)
class ConfirmatoryRunnerInputs:
    """Model-visible rotation data with final-reference outcomes withheld."""

    audit_sample_ids: tuple[str, ...]
    audit_group_ids: tuple[str, ...]
    audit_rgb: ConfirmatoryArray
    audit_target_masks: ConfirmatoryArray
    frozen_audit_features: Mapping[str, NDArray[np.float64] | RowIndexedArray]
    frozen_feature_provenance: Mapping[str, FrozenFeatureProvenance]
    reference_validation_sample_ids: tuple[str, ...]
    reference_validation_group_ids: tuple[str, ...]
    reference_validation_labels: NDArray[np.int64]
    reference_validation_rgb: ConfirmatoryArray
    reference_validation_target_masks: ConfirmatoryArray
    final_reference_group_ids: tuple[str, ...]

    @classmethod
    def from_rotation(cls, rotation: ConfirmatoryRotationInputs) -> ConfirmatoryRunnerInputs:
        """Create a read-only view that cannot reveal final sample IDs or labels."""

        return cls(
            audit_sample_ids=rotation.audit_sample_ids,
            audit_group_ids=rotation.audit_group_ids,
            audit_rgb=_read_only_view(rotation.audit_rgb),
            audit_target_masks=_read_only_view(rotation.audit_target_masks),
            frozen_audit_features=MappingProxyType(
                {
                    key: cast(
                        NDArray[np.float64] | RowIndexedArray,
                        _read_only_view(value),
                    )
                    for key, value in rotation.frozen_audit_features.items()
                }
            ),
            frozen_feature_provenance=MappingProxyType(dict(rotation.frozen_feature_provenance)),
            reference_validation_sample_ids=rotation.reference_validation_sample_ids,
            reference_validation_group_ids=rotation.reference_validation_group_ids,
            reference_validation_labels=cast(
                NDArray[np.int64], _read_only_view(rotation.reference_validation_labels)
            ),
            reference_validation_rgb=_read_only_view(rotation.reference_validation_rgb),
            reference_validation_target_masks=_read_only_view(
                rotation.reference_validation_target_masks
            ),
            final_reference_group_ids=rotation.final_group_ids,
        )


@dataclass(frozen=True, slots=True)
class ConfirmatoryCellRequest:
    """Dependency-injection request for one exact confirmatory matrix cell."""

    cell: ConfirmatoryCell
    scenario: ConfirmatoryScenarioSpec
    corruption: ConfirmatoryCorruptionInput
    inputs: ConfirmatoryRunnerInputs
    controls: ConfirmatoryExecutionControls
    checkpoint_directory: Path
    checkpoint_execution_contract: ConfirmatoryCheckpointExecutionContract | None
    checkpoint_directives: tuple[ConfirmatoryCheckpointDirective, ...]
    cpu_test_only: bool


@dataclass(frozen=True, slots=True)
class FrozenFeatureOOFExecution:
    """Generic grouped-OOF output plus an immutable model/configuration binding."""

    oof_result: OOFResult
    execution_mode: str
    study_outcome_eligible: bool
    configuration_sha256: str
    evidence: Mapping[str, Any]


class ImageOOFRunner(Protocol):
    def __call__(self, request: ConfirmatoryCellRequest, /) -> ConfirmatoryImageOOFResult: ...


class FrozenFeatureOOFRunner(Protocol):
    def __call__(self, request: ConfirmatoryCellRequest, /) -> FrozenFeatureOOFExecution: ...


def _confirmatory_frozen_logistic_parameters(
    controls: ConfirmatoryExecutionControls,
) -> tuple[float, int]:
    """Return the exact frozen logistic parameters without runner-module lookup."""

    selection = controls.original_audit_selection
    classifier = selection.get("classifier")
    if not isinstance(classifier, Mapping) or classifier.get("id") != (
        "multinomial_logistic_regression"
    ):
        raise ValueError("frozen original-audit logistic classifier is unavailable")
    parameters = classifier.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("frozen logistic parameters are unavailable")
    if parameters.get("class_weight") != "balanced":
        raise ValueError("confirmatory frozen logistic requires balanced class weights")
    l2_raw = parameters.get("l2")
    max_iter_raw = parameters.get("max_iter")
    if not isinstance(l2_raw, (int, float)) or isinstance(l2_raw, bool):
        raise ValueError("frozen logistic l2 must be numeric")
    if not isinstance(max_iter_raw, int) or isinstance(max_iter_raw, bool):
        raise ValueError("frozen logistic max_iter must be an integer")
    l2 = float(l2_raw)
    max_iter = max_iter_raw
    if not np.isfinite(l2) or l2 <= 0.0 or max_iter < 1:
        raise ValueError("frozen logistic l2/max_iter are invalid")
    return l2, max_iter


def run_confirmatory_frozen_feature_oof(
    request: ConfirmatoryCellRequest,
) -> FrozenFeatureOOFExecution:
    """Run the canonical grouped logistic OOF path from this sealed core module."""

    if request.scenario.family not in {"imagenet_frozen", "pathology_frozen"}:
        raise ValueError("frozen-feature runner received a non-frozen scenario")
    if request.scenario.classifier != "multinomial_logistic_regression":
        raise ValueError("unsupported confirmatory frozen-feature classifier")
    representation = request.scenario.representation_id
    source_features = request.inputs.frozen_audit_features.get(representation)
    provenance = request.inputs.frozen_feature_provenance.get(representation)
    if source_features is None or provenance is None:
        raise ValueError("frozen-feature matrix/provenance is unavailable")
    if isinstance(source_features, RowIndexedArray):
        gathered = source_features.gather_rows(max_rows=len(source_features))
        if gathered.dtype != np.dtype(np.float64):
            raise ValueError("indexed confirmatory input exposes an unexpected logical dtype")
        features = cast(NDArray[np.float64], gathered)
    else:
        features = np.asarray(source_features, dtype=np.float64)
    l2, max_iter = _confirmatory_frozen_logistic_parameters(request.controls)
    oof = grouped_oof_logistic(
        features,
        request.corruption.observed_labels,
        request.inputs.audit_group_ids,
        final_reference_group_ids=request.inputs.final_reference_group_ids,
        sample_ids=request.inputs.audit_sample_ids,
        fold_assignment_labels=request.corruption.pre_corruption_labels,
        fold_assignment_label_source="pre_corruption_label",
        n_splits=request.controls.n_splits,
        class_order=CLASS_ORDER,
        split_seed=request.controls.split_seed,
        model_seed=request.cell.model_seed,
        representation=representation,
        l2=l2,
        max_iter=max_iter,
    )
    configuration = {
        "schema_version": 1,
        "classifier": "multinomial_logistic_regression",
        "representation_id": representation,
        "model_seed": request.cell.model_seed,
        "split_seed": request.controls.split_seed,
        "n_splits": request.controls.n_splits,
        "l2": l2,
        "max_iter": max_iter,
        "class_weight": "balanced",
        "class_order": list(CLASS_ORDER),
        "fold_assignment_label_source": "pre_corruption_label",
        "frozen_feature_provenance_sha256": provenance.semantic_sha256,
    }
    configuration_sha256 = _controlled_canonical_sha256(configuration)
    return FrozenFeatureOOFExecution(
        oof_result=oof,
        execution_mode=("cpu_test_only" if request.cpu_test_only else "real_study_cpu"),
        study_outcome_eligible=not request.cpu_test_only,
        configuration_sha256=configuration_sha256,
        evidence={
            **configuration,
            "configuration_sha256": configuration_sha256,
            "frozen_feature_provenance_sha256": provenance.semantic_sha256,
            "feature_array_sha256": _controlled_array_sha256(source_features),
            "observed_labels_sha256": array_artifact_sha256(request.corruption.observed_labels),
            "fold_assignment_labels_sha256": array_artifact_sha256(
                request.corruption.pre_corruption_labels
            ),
            "estimator_device": "cpu",
            "cuda_execution_gate_required": False,
        },
    )


def confirmatory_cnn_config_for_cell(
    scenario: ConfirmatoryScenarioSpec,
    controls: ConfirmatoryExecutionControls,
    *,
    model_seed: int,
    cpu_test_only: bool,
) -> ConfirmatoryCNNConfig:
    """Derive the exact CNN configuration used by one planned cell."""

    if scenario.family != "cnn":
        raise ValueError("CNN configuration requires a CNN scenario")
    input_variant = cast(Any, scenario.input_variant)
    weight_identifier = (
        CPU_TEST_ONLY_WEIGHT_IDENTIFIER if cpu_test_only else OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER
    )
    return ConfirmatoryCNNConfig(
        input_variant=input_variant,
        weight_identifier=weight_identifier,
        input_size=controls.input_size,
        epochs=controls.max_epochs,
        batch_size=controls.initial_batch_size,
        minimum_batch_size=controls.minimum_batch_size,
        gradient_accumulation_steps=controls.gradient_accumulation_steps,
        learning_rate=controls.learning_rate,
        weight_decay=controls.weight_decay,
        early_stopping_patience=controls.early_stopping_patience,
        early_stopping_min_delta=controls.early_stopping_min_delta,
        amp_dtype=cast(Any, controls.amp_dtype),
        class_weight_balanced=controls.class_weight == "balanced",
        seed=model_seed,
    )


def _confirmatory_cnn_config(
    request: ConfirmatoryCellRequest,
    *,
    seed: int,
) -> ConfirmatoryCNNConfig:
    return confirmatory_cnn_config_for_cell(
        request.scenario,
        request.controls,
        model_seed=seed,
        cpu_test_only=request.cpu_test_only,
    )


def run_confirmatory_image_oof(request: ConfirmatoryCellRequest) -> ConfirmatoryImageOOFResult:
    """Production adapter from one matrix request to the guarded image OOF API."""

    config = _confirmatory_cnn_config(request, seed=request.cell.model_seed)
    controls = request.controls
    uses_mask = request.scenario.input_variant == "context_rgb_plus_binary_target_mask"
    contract = request.checkpoint_execution_contract
    if contract is None:
        raise ConfirmatoryCheckpointContractError(
            "production image OOF requires an explicit checkpoint execution contract"
        )
    expected_directives = contract.directives_for_cell(
        request.cell.cell_id,
        expected_fold_count=controls.n_splits,
    )
    if request.checkpoint_directives != expected_directives:
        raise ConfirmatoryCheckpointContractError(
            "cell request checkpoint directives differ from the full contract"
        )
    return grouped_oof_confirmatory_cnn(
        request.inputs.audit_rgb,
        request.corruption.observed_labels,
        request.corruption.pre_corruption_labels,
        request.inputs.audit_group_ids,
        sample_ids=request.inputs.audit_sample_ids,
        audit_target_masks=request.inputs.audit_target_masks if uses_mask else None,
        reference_validation_rgb=request.inputs.reference_validation_rgb,
        reference_validation_labels=request.inputs.reference_validation_labels,
        reference_validation_sample_ids=request.inputs.reference_validation_sample_ids,
        reference_validation_group_ids=request.inputs.reference_validation_group_ids,
        reference_validation_target_masks=(
            request.inputs.reference_validation_target_masks if uses_mask else None
        ),
        final_reference_group_ids=request.inputs.final_reference_group_ids,
        base_config=config,
        cell_id=request.cell.cell_id,
        checkpoint_directory=request.checkpoint_directory,
        checkpoint_execution_contract=contract,
        cpu_test_only=request.cpu_test_only,
        n_splits=controls.n_splits,
        split_seed=controls.split_seed,
    )


@dataclass(frozen=True, slots=True)
class ConfirmatoryMatrixArtifacts:
    """Persisted matrix outputs and exact completion reconciliation."""

    output_directory: Path
    matrix_plan_path: Path
    execution_controls_path: Path
    frozen_feature_provenance_path: Path
    original_audit_selection_path: Path
    cell_index_path: Path
    ensemble_evidence_path: Path
    hybrid_ablations_path: Path
    fold_aggregate_path: Path
    reconciliation_path: Path
    completion_evidence_path: Path
    analysis_gaps_path: Path
    figure_manifest_path: Path
    report_path: Path
    artifact_manifest_path: Path
    outcomes: tuple[Mapping[str, Any], ...]
    reconciliation: ConfirmatoryMatrixReconciliation
    study_outcome_eligible: bool


@dataclass(frozen=True, slots=True)
class SyntheticConfirmatoryFixtureResult:
    """Structural fixture result that can never enable ``CONFIRMATORY_COMPLETE``."""

    artifacts: ConfirmatoryMatrixArtifacts
    completed_cell_count: int
    skipped_optional_cell_count: int


@dataclass(slots=True)
class _ExecutedCell:
    request: ConfirmatoryCellRequest
    oof: OOFResult
    execution_mode: str
    study_outcome_eligible: bool
    configuration_sha256: str
    execution_evidence: Mapping[str, Any]


def _validate_blockers(
    blockers: Mapping[str, ConfirmatoryFrozenBlocker], controls: ConfirmatoryExecutionControls
) -> None:
    for scenario_id, blocker in blockers.items():
        scenario = controls.scenarios_by_id.get(scenario_id)
        if scenario is None:
            raise ValueError(f"frozen blocker names unknown scenario {scenario_id!r}")
        if scenario.required:
            raise ValueError("required confirmatory scenarios cannot have availability blockers")
        if blocker.scenario_id != scenario_id or not blocker.blocker.strip():
            raise ValueError("frozen blocker identity/text is invalid")
        if blocker.config_semantic_sha256 != controls.config_semantic_sha256:
            raise ValueError("frozen blocker is bound to a different confirmatory config")
        if (
            scenario.availability_audit_sha256 is None
            or blocker.availability_audit_sha256 != scenario.availability_audit_sha256
        ):
            raise ValueError("frozen blocker availability hash differs from the scenario")
        cache_record = controls.cache_provenance_by_id[scenario.cache_provenance_id]
        if cache_record["status"] != "unavailable_with_frozen_blocker":
            raise ValueError("optional scenario with available cache provenance cannot be skipped")
        if cache_record.get("blocker_evidence_sha256") != blocker.availability_audit_sha256:
            raise ValueError("frozen blocker differs from cache-provenance blocker evidence")


def _aggregate_frozen_feature_provenance(
    rotations_by_fold: Mapping[int, ConfirmatoryRotationInputs],
    controls: ConfirmatoryExecutionControls,
) -> dict[str, Any]:
    representation_ids = sorted(
        {
            representation_id
            for rotation in rotations_by_fold.values()
            for representation_id in rotation.frozen_feature_provenance
        }
    )
    return {
        "schema_version": 1,
        "status": "completed",
        "confirmatory_config_semantic_sha256": controls.config_semantic_sha256,
        "matrix_plan_config_sha256": controls.plan.config_sha256,
        "representations": {
            representation_id: {
                "rotations": {
                    str(outer_fold): asdict(
                        rotations_by_fold[outer_fold].frozen_feature_provenance[representation_id]
                    )
                    for outer_fold in sorted(rotations_by_fold)
                    if representation_id in rotations_by_fold[outer_fold].frozen_feature_provenance
                }
            }
            for representation_id in representation_ids
        },
    }


def _original_audit_selection_artifact(
    rotations_by_fold: Mapping[int, ConfirmatoryRotationInputs],
    controls: ConfirmatoryExecutionControls,
) -> dict[str, Any]:
    selection = dict(controls.original_audit_selection)
    representation_id = str(selection["representation_id"])
    cache_provenance_id = str(selection["cache_provenance_id"])
    cache_contract = controls.cache_provenance_by_id[cache_provenance_id]
    actual_by_rotation: dict[str, Any] = {}
    for outer_fold in sorted(rotations_by_fold):
        provenance = rotations_by_fold[outer_fold].frozen_feature_provenance.get(representation_id)
        if provenance is None:
            raise ValueError(
                "original-audit selected representation lacks rotation cache provenance"
            )
        if provenance.cache_provenance_id != cache_provenance_id:
            raise ValueError("original-audit selected cache provenance ID differs from config")
        for field in (
            "representation_id",
            "cache_file_sha256",
            "sidecar_semantic_sha256",
            "sample_order_sha256",
            "manifest_sha256",
            "encoder_identifier",
            "encoder_metadata_sha256",
            "weight_identifier",
            "weights_sha256",
            "preprocessing_identifier",
            "preprocessing_sha256",
            "input_variant",
        ):
            if getattr(provenance, field) != cache_contract[field]:
                raise ValueError(
                    "original-audit selected cache provenance differs from frozen contract "
                    f"for {field}"
                )
        actual_by_rotation[str(outer_fold)] = {
            **asdict(provenance),
            "final_reference_group_ids_sha256": _canonical_sha256(
                sorted(rotations_by_fold[outer_fold].final_group_ids)
            ),
        }
    return {
        "schema_version": 1,
        "status": "completed",
        "confirmatory_config_semantic_sha256": controls.config_semantic_sha256,
        "matrix_plan_config_sha256": controls.plan.config_sha256,
        "execution_controls_binding_sha256": controls.binding_sha256,
        "selection_semantic_sha256": _canonical_sha256(selection),
        "selection": selection,
        "frozen_cache_provenance_record": dict(cache_contract),
        "sealed_feature_cache_provenance_by_rotation": actual_by_rotation,
    }


def _validate_oof(request: ConfirmatoryCellRequest, oof: OOFResult) -> None:
    oof.validate()
    inputs = request.inputs
    if oof.sample_ids != inputs.audit_sample_ids:
        raise ValueError("OOF sample order differs from rotation audit order")
    if oof.group_ids != inputs.audit_group_ids:
        raise ValueError("OOF group order differs from rotation audit order")
    if set(oof.final_reference_groups) != set(inputs.final_reference_group_ids):
        raise ValueError("OOF final-reference group evidence differs from rotation")
    if oof.class_order != CLASS_ORDER:
        raise ValueError("OOF class order differs from fixed five-class order")
    if oof.model_seed != request.cell.model_seed or oof.split_seed != request.controls.split_seed:
        raise ValueError("OOF model/split seed differs from frozen cell controls")
    if oof.fold_assignment_label_source != "pre_corruption_label" or not np.array_equal(
        oof.fold_assignment_labels, request.corruption.pre_corruption_labels
    ):
        raise ValueError("OOF folds are not fixed from pre_corruption_label")


def _hold_checkpoint_execution_history(
    stack: ExitStack,
    manifest_payload: Mapping[str, Any],
    *,
    run_directory: Path,
    role: str,
) -> tuple[dict[str, Any], ...]:
    """Hold every exact version checkpoint and sidecar named by schema v3."""

    raw_outputs = manifest_payload.get("versioned_outputs")
    if not isinstance(raw_outputs, list):
        raise ConfirmatoryCheckpointContractError(
            f"{role} lacks its exact versioned checkpoint list"
        )
    held_records: list[dict[str, Any]] = []
    for index, raw_output in enumerate(raw_outputs, start=1):
        if not isinstance(raw_output, Mapping):
            raise ConfirmatoryCheckpointContractError(
                f"{role} versioned checkpoint {index} is not a mapping"
            )
        checkpoint_identity = _checkpoint_file_identity_from_exact_manifest(
            raw_output.get("checkpoint"),
            role=f"{role} versioned checkpoint {index}",
        )
        checkpoint_relative = raw_output.get("checkpoint_relative_path")
        commit_relative = raw_output.get("commit_manifest_relative_path")
        if not isinstance(checkpoint_relative, str) or not isinstance(commit_relative, str):
            raise ConfirmatoryCheckpointContractError(
                f"{role} versioned checkpoint {index} paths are incomplete"
            )
        checkpoint_path = (run_directory / checkpoint_relative).resolve()
        commit_path = (run_directory / commit_relative).resolve()
        if checkpoint_path != checkpoint_identity.path:
            raise ConfirmatoryCheckpointContractError(
                f"{role} versioned checkpoint {index} absolute/relative paths differ"
            )
        commit_identity = _checkpoint_physical_identity_from_exact_manifest(
            raw_output.get("commit_manifest_physical_identity"),
            role=f"{role} commit sidecar {index}",
        )
        held_checkpoint = stack.enter_context(
            _hold_private_checkpoint_snapshot(
                checkpoint_path,
                role=f"{role} versioned checkpoint {index}",
            )
        )
        held_commit = stack.enter_context(
            _hold_private_checkpoint_snapshot(
                commit_path,
                role=f"{role} commit sidecar {index}",
            )
        )
        if (
            held_checkpoint.identity != checkpoint_identity.physical_identity
            or held_checkpoint.size_bytes != checkpoint_identity.size_bytes
            or held_checkpoint.sha256 != checkpoint_identity.sha256
            or held_commit.identity != commit_identity
            or held_commit.size_bytes != raw_output.get("commit_manifest_size_bytes")
            or held_commit.sha256 != raw_output.get("commit_manifest_sha256")
            or stat.S_IMODE(held_checkpoint.identity.mode)
            & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            or stat.S_IMODE(held_commit.identity.mode)
            & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise ConfirmatoryCheckpointContractError(
                f"{role} versioned checkpoint/sidecar {index} differs from its manifest"
            )
        held_records.append(
            {
                "publication_index": raw_output.get("publication_index"),
                "completed_epochs": raw_output.get("completed_epochs"),
                "checkpoint_relative_path": checkpoint_relative,
                "checkpoint_sha256": checkpoint_identity.sha256,
                "checkpoint_size_bytes": checkpoint_identity.size_bytes,
                "checkpoint_physical_identity": (checkpoint_identity.physical_identity.as_dict()),
                "commit_manifest_relative_path": commit_relative,
                "commit_manifest_sha256": held_commit.sha256,
                "commit_manifest_size_bytes": held_commit.size_bytes,
                "commit_manifest_physical_identity": held_commit.identity.as_dict(),
            }
        )
    return tuple(held_records)


def _hold_checkpoint_execution_canonical(
    stack: ExitStack,
    manifest_payload: Mapping[str, Any],
    *,
    run_directory: Path,
    role: str,
) -> dict[str, Any]:
    """Hold the exact read-only canonical working checkpoint named by schema v3."""

    identity = _checkpoint_file_identity_from_exact_manifest(
        manifest_payload.get("canonical_working_checkpoint"),
        role=f"{role} canonical working checkpoint",
    )
    try:
        relative = identity.path.relative_to(run_directory.resolve())
    except ValueError as exc:
        raise ConfirmatoryCheckpointContractError(
            f"{role} canonical working checkpoint is outside the run"
        ) from exc
    held = stack.enter_context(
        _hold_private_checkpoint_snapshot(
            identity.path,
            role=f"{role} canonical working checkpoint",
        )
    )
    if (
        held.identity != identity.physical_identity
        or held.size_bytes != identity.size_bytes
        or held.sha256 != identity.sha256
        or stat.S_IMODE(held.identity.mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise ConfirmatoryCheckpointContractError(
            f"{role} canonical working checkpoint differs or remains writable"
        )
    return {
        "path": relative.as_posix(),
        "sha256": held.sha256,
        "size_bytes": held.size_bytes,
        "file_id_128": held.identity.file_id_128,
        "physical_identity": held.identity.as_dict(),
        "read_only": True,
    }


def _normalise_image_execution(
    request: ConfirmatoryCellRequest, result: ConfirmatoryImageOOFResult
) -> _ExecutedCell:
    result.validate()
    _validate_oof(request, result.oof_result)
    provenance = request.inputs.frozen_feature_provenance[request.scenario.representation_id]
    hashes = tuple(row.configuration_sha256 for row in result.fold_evidence)
    fold_evidence_payloads = [asdict(row) for row in result.fold_evidence]
    if not request.cpu_test_only:
        if result.execution_mode != "real_study_cuda" or not result.study_outcome_eligible:
            raise ValueError("real CNN aggregate execution mode/eligibility is invalid")
        uses_mask = request.scenario.input_variant == "context_rgb_plus_binary_target_mask"
        expected_preprocessing = {
            "rgb_resize": "bilinear_antialias",
            "rgb_range_before_normalisation": [0.0, 1.0],
            "rgb_mean": [0.485, 0.456, 0.406],
            "rgb_standard_deviation": [0.229, 0.224, 0.225],
            "target_mask_resize": ("nearest_binary_unnormalised" if uses_mask else None),
        }
        fold_evidence_payloads = []
        for row in result.fold_evidence:
            expected_config = _confirmatory_cnn_config(
                request,
                seed=request.cell.model_seed + row.fold_id,
            )
            metadata = row.model_metadata
            checkpoint = Path(row.checkpoint_path).resolve()
            checkpoint_execution_manifest = Path(row.checkpoint_execution_manifest_path).resolve()
            checkpoint_directory = request.checkpoint_directory.resolve()
            if row.checkpoint_action == "restore_terminal_checkpoint_without_fit":
                expected_checkpoint_name = f"fold_{row.fold_id:02d}.pt"
                expected_checkpoint_parent = checkpoint_directory
            else:
                expected_checkpoint_name = f"epoch_{row.completed_epochs:04d}.pt"
                expected_checkpoint_parent = (
                    checkpoint_directory.parent / "checkpoint_versions" / f"fold_{row.fold_id:02d}"
                )
            expected_execution_manifest = (
                checkpoint_directory.parent
                / "checkpoint_execution"
                / f"fold_{row.fold_id:02d}.json"
            )
            with ExitStack() as checkpoint_sources:
                held_checkpoint = checkpoint_sources.enter_context(
                    _hold_private_checkpoint_snapshot(
                        checkpoint,
                        role="normalised image execution checkpoint",
                    )
                )
                held_manifest = checkpoint_sources.enter_context(
                    _hold_private_checkpoint_snapshot(
                        checkpoint_execution_manifest,
                        role="normalised image checkpoint execution manifest",
                    )
                )
                try:
                    manifest_payload = json.loads(held_manifest.payload.decode("ascii"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        "checkpoint execution manifest is not canonical ASCII JSON"
                    ) from exc
                expected_manifest_bytes = (
                    json.dumps(
                        manifest_payload,
                        allow_nan=False,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("ascii")
                    + b"\n"
                )
                directive_by_fold = {
                    directive.fold_id: directive for directive in request.checkpoint_directives
                }
                expected_directive = directive_by_fold.get(row.fold_id)
                if (
                    expected_directive is None
                    or len(directive_by_fold) != request.controls.n_splits
                ):
                    raise ConfirmatoryCheckpointContractError(
                        "image execution lacks its exact checkpoint directive"
                    )
                _require_exact_checkpoint_execution_manifest_payload(
                    manifest_payload,
                    expected_directive,
                    run_directory=request.checkpoint_directory.resolve().parents[2],
                )
                run_directory = request.checkpoint_directory.resolve().parents[2]
                canonical_working = _hold_checkpoint_execution_canonical(
                    checkpoint_sources,
                    manifest_payload,
                    run_directory=run_directory,
                    role="normalised image execution",
                )
                versioned_history = _hold_checkpoint_execution_history(
                    checkpoint_sources,
                    manifest_payload,
                    run_directory=run_directory,
                    role="normalised image execution",
                )
                final_checkpoint = (
                    manifest_payload.get("final_checkpoint")
                    if isinstance(manifest_payload, Mapping)
                    else None
                )
                versioned_outputs = (
                    manifest_payload.get("versioned_outputs")
                    if isinstance(manifest_payload, Mapping)
                    else None
                )
                if (
                    row.execution_mode != "real_study_cuda"
                    or not row.study_outcome_eligible
                    or row.model_seed != expected_config.seed
                    or row.configuration_sha256 != _canonical_sha256(asdict(expected_config))
                    or row.telemetry.get("execution_mode") != "real_study_cuda"
                    or row.telemetry.get("study_outcome_eligible") is not True
                    or metadata.get("weight_identifier") != provenance.weight_identifier
                    or metadata.get("weight_sha256") != provenance.weights_sha256
                    or metadata.get("architecture") != "torchvision.resnet18"
                    or metadata.get("class_order") != list(CLASS_ORDER)
                    or metadata.get("input_channels") != (4 if uses_mask else 3)
                    or metadata.get("preprocessing") != expected_preprocessing
                    or metadata.get("fourth_channel_initialisation")
                    != ("zeros" if uses_mask else None)
                    or checkpoint.parent != expected_checkpoint_parent
                    or checkpoint.name != expected_checkpoint_name
                    or held_checkpoint.size_bytes != row.checkpoint_size_bytes
                    or held_checkpoint.identity != row.checkpoint_physical_identity
                    or held_checkpoint.sha256 != row.checkpoint_sha256
                    or stat.S_IMODE(held_checkpoint.identity.mode)
                    & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                    or checkpoint_execution_manifest != expected_execution_manifest
                    or held_manifest.identity != row.checkpoint_execution_manifest_physical_identity
                    or held_manifest.sha256 != row.checkpoint_execution_manifest_sha256
                    or stat.S_IMODE(held_manifest.identity.mode)
                    & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                    or held_manifest.payload != expected_manifest_bytes
                    or not isinstance(manifest_payload, Mapping)
                    or manifest_payload.get("schema_version") != 3
                    or manifest_payload.get("policy")
                    != "aanca_fold_boundary_checkpoint_execution_v3"
                    or manifest_payload.get("action") != row.checkpoint_action
                    or manifest_payload.get("trained_epochs") != row.trained_epochs_this_invocation
                    or not isinstance(final_checkpoint, Mapping)
                    or final_checkpoint.get("path") != str(checkpoint)
                    or final_checkpoint.get("sha256") != row.checkpoint_sha256
                    or final_checkpoint.get("file_id_128")
                    != row.checkpoint_physical_identity.file_id_128
                    or not isinstance(versioned_outputs, list)
                    or (
                        row.checkpoint_action == "restore_terminal_checkpoint_without_fit"
                        and (
                            versioned_outputs != []
                            or manifest_payload.get(
                                "versioned_checkpoint_output_directory_relative_path"
                            )
                            is not None
                            or canonical_working["path"]
                            != checkpoint.relative_to(run_directory).as_posix()
                            or canonical_working["file_id_128"]
                            != row.checkpoint_physical_identity.file_id_128
                        )
                    )
                    or (
                        row.checkpoint_action in {"fresh_fit", "resume_incomplete_fit"}
                        and (
                            not versioned_outputs
                            or not isinstance(versioned_outputs[-1], Mapping)
                            or not isinstance(
                                versioned_outputs[-1].get("checkpoint"),
                                Mapping,
                            )
                            or versioned_outputs[-1].get("checkpoint", {}).get("sha256")
                            != row.checkpoint_sha256
                            or canonical_working["sha256"] != row.checkpoint_sha256
                            or canonical_working["size_bytes"] != row.checkpoint_size_bytes
                            or canonical_working["file_id_128"]
                            == row.checkpoint_physical_identity.file_id_128
                        )
                    )
                ):
                    raise ValueError(
                        "real CNN fold evidence differs from frozen mode, weights, "
                        "configuration, preprocessing, or checkpoint execution bytes"
                    )
                fold_evidence_payloads.append(
                    {
                        **asdict(row),
                        "checkpoint_canonical_working": dict(canonical_working),
                        "checkpoint_versioned_outputs": list(versioned_history),
                    }
                )
    return _ExecutedCell(
        request=request,
        oof=result.oof_result,
        execution_mode=result.execution_mode,
        study_outcome_eligible=result.study_outcome_eligible,
        configuration_sha256=_canonical_sha256(hashes),
        execution_evidence={
            "fold_evidence": fold_evidence_payloads,
            "execution_mode": result.execution_mode,
            "study_outcome_eligible": result.study_outcome_eligible,
            "scenario_cache_provenance_sha256": provenance.semantic_sha256,
        },
    )


def _normalise_frozen_execution(
    request: ConfirmatoryCellRequest, result: FrozenFeatureOOFExecution
) -> _ExecutedCell:
    if len(result.configuration_sha256) != _SHA256_LENGTH:
        raise ValueError("frozen-feature runner lacks a configuration SHA-256")
    if not result.execution_mode.strip():
        raise ValueError("frozen-feature runner lacks execution mode evidence")
    _validate_oof(request, result.oof_result)
    provenance = request.inputs.frozen_feature_provenance[request.scenario.representation_id]
    if result.evidence.get("frozen_feature_provenance_sha256") != (provenance.semantic_sha256):
        raise ValueError("frozen-feature runner evidence is not bound to its cache provenance")
    if not request.cpu_test_only:
        selection = request.controls.original_audit_selection
        classifier = cast(Mapping[str, Any], selection["classifier"])
        parameters = cast(Mapping[str, Any], classifier["parameters"])
        configuration = {
            "schema_version": 1,
            "classifier": "multinomial_logistic_regression",
            "representation_id": request.scenario.representation_id,
            "model_seed": request.cell.model_seed,
            "split_seed": request.controls.split_seed,
            "n_splits": request.controls.n_splits,
            "l2": float(parameters["l2"]),
            "max_iter": int(parameters["max_iter"]),
            "class_weight": "balanced",
            "class_order": list(CLASS_ORDER),
            "fold_assignment_label_source": "pre_corruption_label",
            "frozen_feature_provenance_sha256": provenance.semantic_sha256,
        }
        configuration_sha256 = _canonical_sha256(configuration)
        expected_evidence = {
            **configuration,
            "configuration_sha256": configuration_sha256,
            "feature_array_sha256": _controlled_array_sha256(
                request.inputs.frozen_audit_features[request.scenario.representation_id]
            ),
            "observed_labels_sha256": array_artifact_sha256(request.corruption.observed_labels),
            "fold_assignment_labels_sha256": array_artifact_sha256(
                request.corruption.pre_corruption_labels
            ),
            "estimator_device": "cpu",
            "cuda_execution_gate_required": False,
        }
        if (
            result.execution_mode != "real_study_cpu"
            or not result.study_outcome_eligible
            or result.configuration_sha256 != configuration_sha256
            or dict(result.evidence) != expected_evidence
        ):
            raise ValueError(
                "real frozen-feature execution evidence differs from its exact "
                "configuration, features, labels, or CPU mode"
            )
    return _ExecutedCell(
        request=request,
        oof=result.oof_result,
        execution_mode=result.execution_mode,
        study_outcome_eligible=result.study_outcome_eligible,
        configuration_sha256=result.configuration_sha256,
        execution_evidence=dict(result.evidence),
    )


@dataclass(frozen=True, slots=True)
class _PublishedScientificArtifact:
    path: Path
    sha256: str
    size_bytes: int
    identity: ConfirmatoryCheckpointPhysicalIdentity


_SCIENTIFIC_ARTIFACT_PUBLICATIONS: dict[str, _PublishedScientificArtifact] = {}
_SCIENTIFIC_ARTIFACT_PUBLICATIONS_LOCK = threading.RLock()


def _scientific_artifact_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _scientific_artifact_record(path: Path) -> _PublishedScientificArtifact:
    key = _scientific_artifact_key(path)
    with _SCIENTIFIC_ARTIFACT_PUBLICATIONS_LOCK:
        record = _SCIENTIFIC_ARTIFACT_PUBLICATIONS.get(key)
    if record is None or os.path.normcase(str(record.path)) != os.path.normcase(
        os.path.abspath(path)
    ):
        raise ConfirmatoryCheckpointContractError(
            f"scientific artifact was not published by this execution: {path}"
        )
    return record


def _require_scientific_snapshot(
    snapshot: _PrivateCheckpointSnapshot,
    record: _PublishedScientificArtifact,
    *,
    role: str,
) -> None:
    if (
        snapshot.sha256 != record.sha256
        or snapshot.size_bytes != record.size_bytes
        or snapshot.identity != record.identity
    ):
        raise ConfirmatoryCheckpointContractError(
            f"{role} differs from its exact create-if-absent publication"
        )


def _publish_scientific_file(
    path: Path,
    writer: Callable[[Any], None],
    *,
    role: str,
) -> Path:
    """Publish directly to one retained O_EXCL destination.

    The final pathname is the creation pathname.  It is never linked, replaced,
    adopted, removed, or reopened while publication authority is live.  A failure
    after the O_EXCL claim deliberately leaves the partial destination in place so
    that a caller cannot silently retry or adopt another file at the same path.
    """

    destination = Path(os.path.abspath(path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    key = _scientific_artifact_key(destination)
    with _SCIENTIFIC_ARTIFACT_PUBLICATIONS_LOCK:
        if key in _SCIENTIFIC_ARTIFACT_PUBLICATIONS:
            raise ConfirmatoryCheckpointContractError(
                f"{role} path was already published by this execution"
            )
        if os.path.lexists(destination):
            raise FileExistsError(
                f"{role} destination already exists; overwrite is forbidden: {destination}"
            )
        parent_stat = destination.parent.lstat()
        if (
            not stat.S_ISDIR(parent_stat.st_mode)
            or _is_reparse_stat(parent_stat)
            or _checkpoint_named_streams(destination.parent)
            or os.path.normcase(str(destination.parent.resolve()))
            != os.path.normcase(str(destination.parent))
        ):
            raise ConfirmatoryCheckpointContractError(
                f"{role} parent is not a plain stream-free directory"
            )
        descriptor: int | None = None
        registered_record: _PublishedScientificArtifact | None = None
        try:
            descriptor, native_created = _create_checkpoint_descriptor(destination)
            with os.fdopen(os.dup(descriptor), "wb") as handle:
                writer(handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.lseek(descriptor, 0, os.SEEK_SET)
            payload = bytearray()
            while chunk := os.read(descriptor, 8 * 1024 * 1024):
                payload.extend(chunk)
            held_before = _stat_identity(os.fstat(descriptor))
            lexical_before_stat = destination.lstat()
            lexical_before = _stat_identity(lexical_before_stat)
            native_before = _native_identity_from_descriptor(descriptor)
            native_path_before = _native_identity_from_live_writer_path(destination)
            if (
                not payload
                or not _same_open_file_object(held_before, lexical_before)
                or held_before.size_bytes != len(payload)
                or lexical_before.size_bytes != len(payload)
                or held_before.link_count != 1
                or lexical_before.link_count != 1
                or not stat.S_ISREG(held_before.mode)
                or _is_reparse(lexical_before_stat)
                or _checkpoint_named_streams(destination)
                or not _same_native_file(native_before, native_created)
                or not _same_native_file(native_path_before, native_created)
                or not _native_identity_is_plain_file(native_before)
            ):
                raise ConfirmatoryCheckpointContractError(
                    f"{role} O_EXCL destination is not exact and private"
                )
            if os.name == "nt":
                _win32_set_read_only_on_descriptor(descriptor)
            else:
                _set_posix_descriptor_mode(
                    descriptor,
                    stat.S_IMODE(held_before.mode) & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH),
                )
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            frozen_payload = bytearray()
            while chunk := os.read(descriptor, 8 * 1024 * 1024):
                frozen_payload.extend(chunk)
            held_after = _stat_identity(os.fstat(descriptor))
            lexical_after_stat = destination.lstat()
            lexical_after = _stat_identity(lexical_after_stat)
            native_after = _native_identity_from_descriptor(descriptor)
            native_path_after = _native_identity_from_live_writer_path(destination)
            if (
                frozen_payload != payload
                or not _same_file_except_read_only_transition(held_before, held_after)
                or not _same_file_except_read_only_transition(
                    lexical_before,
                    lexical_after,
                )
                or not _same_open_file_object(held_after, lexical_after)
                or held_after.link_count != 1
                or lexical_after.link_count != 1
                or not _same_native_file(
                    native_created,
                    native_after,
                    allow_read_only_transition=True,
                )
                or not _same_native_file(
                    native_created,
                    native_path_after,
                    allow_read_only_transition=True,
                )
                or _is_reparse(lexical_after_stat)
                or _checkpoint_named_streams(destination)
            ):
                raise ConfirmatoryCheckpointContractError(
                    f"{role} changed while being frozen on its creation handle"
                )
            registered_record = _PublishedScientificArtifact(
                path=destination,
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                identity=lexical_after,
            )
            _SCIENTIFIC_ARTIFACT_PUBLICATIONS[key] = registered_record
            _fsync_directory(destination.parent)
        except ConfirmatoryCheckpointContractError:
            if (
                registered_record is not None
                and _SCIENTIFIC_ARTIFACT_PUBLICATIONS.get(key) is registered_record
            ):
                del _SCIENTIFIC_ARTIFACT_PUBLICATIONS[key]
            raise
        except (OSError, RuntimeError) as exc:
            if (
                registered_record is not None
                and _SCIENTIFIC_ARTIFACT_PUBLICATIONS.get(key) is registered_record
            ):
                del _SCIENTIFIC_ARTIFACT_PUBLICATIONS[key]
            raise ConfirmatoryCheckpointContractError(
                f"{role} immutable no-overwrite publication failed"
            ) from exc
        except BaseException:
            if (
                registered_record is not None
                and _SCIENTIFIC_ARTIFACT_PUBLICATIONS.get(key) is registered_record
            ):
                del _SCIENTIFIC_ARTIFACT_PUBLICATIONS[key]
            raise
        finally:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
        return destination


def atomic_write_text(path: str | Path, content: str) -> Path:
    """Create one immutable UTF-8 artifact with platform-independent newlines."""

    payload = content.replace("\r\n", "\n").encode("utf-8")
    return _publish_scientific_file(
        Path(path),
        lambda handle: handle.write(payload),
        role="scientific text artifact",
    )


def atomic_write_json(path: str | Path, value: Any, *, indent: int = 2) -> Path:
    """Create one immutable strict-JSON artifact; replacement is forbidden."""

    content = json.dumps(
        value,
        allow_nan=False,
        default=_json_default,
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
    )
    return atomic_write_text(path, f"{content}\n")


def _atomic_npz(path: Path, arrays: Mapping[str, NDArray[np.generic]]) -> Path:
    return _publish_scientific_file(
        path,
        lambda handle: np.savez_compressed(
            handle,
            **cast(dict[str, Any], dict(arrays)),
        ),
        role="scientific NPZ artifact",
    )


def _publish_scientific_hash_manifest(
    path: Path,
    sources: Sequence[Path],
    *,
    relative_to: Path | None = None,
) -> Path:
    """Publish a manifest while every exact original source object remains held."""

    manifest_payload: dict[str, str] = {}
    with ExitStack() as held_sources:
        for source in sources:
            source = Path(os.path.abspath(source))
            record = _scientific_artifact_record(source)
            held = held_sources.enter_context(
                _hold_private_checkpoint_snapshot(
                    source,
                    role="scientific artifact-manifest source",
                )
            )
            _require_scientific_snapshot(
                held,
                record,
                role="scientific artifact-manifest source",
            )
            key = (
                source.relative_to(Path(os.path.abspath(relative_to))).as_posix()
                if relative_to is not None
                else source.name
            )
            if key in manifest_payload:
                raise ConfirmatoryCheckpointContractError(
                    "scientific artifact manifest contains a duplicate source path"
                )
            manifest_payload[key] = held.sha256
        manifest = atomic_write_json(path, manifest_payload)
        manifest_record = _scientific_artifact_record(manifest)
        held_manifest = held_sources.enter_context(
            _hold_private_checkpoint_snapshot(
                manifest,
                role="scientific artifact manifest",
            )
        )
        _require_scientific_snapshot(
            held_manifest,
            manifest_record,
            role="scientific artifact manifest",
        )
    return manifest


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    fields: list[str] = []
    for row in rows:
        fields.extend(field for field in row if field not in fields)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return atomic_write_text(path, buffer.getvalue())


def _ensemble_risks(result: EnsembleDisagreementResult) -> dict[str, NDArray[np.float64]]:
    return {
        "predictive_entropy_of_mean": result.entropy_of_mean,
        "mean_pairwise_js_divergence": result.mean_pairwise_js_divergence,
        "variation_ratio": result.variation_ratio,
        "observed_label_probability_variance": result.observed_label_probability_variance,
        "predicted_class_disagreement": result.predicted_class_disagreement,
    }


def _base_outcome(
    cell: ConfirmatoryCell,
    scenario: ConfirmatoryScenarioSpec,
    corruption: ConfirmatoryCorruptionInput,
) -> dict[str, Any]:
    return {
        "cell_id": cell.cell_id,
        "outer_fold": cell.outer_fold,
        "corruption_cell_id": cell.corruption_cell_id,
        "corruption_mechanism": corruption.mechanism,
        "corruption_rate": corruption.rate,
        "corruption_seed": corruption.seed,
        "scenario_id": cell.scenario_id,
        "scenario_family": scenario.family,
        "representation_id": scenario.representation_id,
        "cache_provenance_id": scenario.cache_provenance_id,
        "model_seed": cell.model_seed,
        "required": cell.required,
    }


def _is_reparse_stat(value: os.stat_result) -> bool:
    return bool(
        int(getattr(value, "st_file_attributes", 0))
        & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    )


def _checkpoint_physical_identity_from_evidence(
    value: Any,
    *,
    role: str,
) -> ConfirmatoryCheckpointPhysicalIdentity:
    if isinstance(value, ConfirmatoryCheckpointPhysicalIdentity):
        identity = value
    elif isinstance(value, Mapping):
        fields = {
            "device",
            "inode",
            "size_bytes",
            "mode",
            "link_count",
            "modified_time_ns",
            "changed_time_ns",
        }
        if set(value) != fields or any(type(value[field]) is not int for field in fields):
            raise ConfirmatoryCheckpointContractError(f"{role} physical identity is malformed")
        identity = ConfirmatoryCheckpointPhysicalIdentity(
            device=int(value["device"]),
            inode=int(value["inode"]),
            size_bytes=int(value["size_bytes"]),
            mode=int(value["mode"]),
            link_count=int(value["link_count"]),
            modified_time_ns=int(value["modified_time_ns"]),
            changed_time_ns=int(value["changed_time_ns"]),
        )
    else:
        raise ConfirmatoryCheckpointContractError(f"{role} physical identity is untyped")
    identity.validate()
    return identity


def _checkpoint_tree_records(cells_root: Path) -> tuple[str, ...]:
    """Return a no-follow relative inventory of a successor's checkpoint-only tree."""

    if not os.path.lexists(cells_root):
        return ()
    try:
        root_stat = cells_root.lstat()
    except OSError as exc:
        raise ConfirmatoryCheckpointContractError("checkpoint cells root is unreadable") from exc
    if not stat.S_ISDIR(root_stat.st_mode) or _is_reparse_stat(root_stat):
        raise ConfirmatoryCheckpointContractError(
            "checkpoint cells root is linked, reparse, or not a directory"
        )
    if _checkpoint_named_streams(cells_root):
        raise ConfirmatoryCheckpointContractError("checkpoint cells root contains a named stream")
    records: list[str] = []
    pending = [cells_root]
    while pending:
        directory = pending.pop()
        try:
            directory_stat = directory.lstat()
        except OSError as exc:
            raise ConfirmatoryCheckpointContractError(
                "checkpoint directory became unreadable"
            ) from exc
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or _is_reparse_stat(directory_stat)
            or _checkpoint_named_streams(directory)
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint tree contains a linked/reparse/streamed directory"
            )
        try:
            entries = sorted(os.scandir(directory), key=lambda value: value.name)
        except OSError as exc:
            raise ConfirmatoryCheckpointContractError(
                "checkpoint tree cannot be enumerated"
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(cells_root.parent).as_posix()
            try:
                observed = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ConfirmatoryCheckpointContractError(
                    f"checkpoint tree entry is unreadable: {relative}"
                ) from exc
            if _is_reparse_stat(observed) or entry.is_symlink():
                raise ConfirmatoryCheckpointContractError(
                    f"checkpoint tree contains a linked/reparse entry: {relative}"
                )
            if stat.S_ISDIR(observed.st_mode):
                records.append(f"directory:{relative}")
                pending.append(path)
            elif (
                stat.S_ISREG(observed.st_mode)
                and int(observed.st_nlink) == 1
                and not _checkpoint_named_streams(path)
            ):
                records.append(f"file:{relative}")
            else:
                raise ConfirmatoryCheckpointContractError(
                    f"checkpoint tree contains a non-private entry: {relative}"
                )
    return tuple(sorted(records))


def _expected_successor_tree_records(
    contract: ConfirmatoryCheckpointExecutionContract,
) -> tuple[str, ...]:
    files = {
        f"cells/{value.cell_id}/checkpoints/fold_{value.fold_id:02d}.pt"
        for value in contract.directives
        if value.action != "fresh_fit"
    }
    directories: set[str] = set()
    for relative in files:
        path = Path(relative)
        parent = path.parent
        while parent.as_posix() != ".":
            directories.add(parent.as_posix())
            if parent.as_posix() == "cells":
                break
            parent = parent.parent
    records = [f"directory:{value}" for value in directories if value != "cells"]
    records.extend(f"file:{value}" for value in files)
    return tuple(sorted(records))


def _require_matrix_checkpoint_contract(
    contract: ConfirmatoryCheckpointExecutionContract | None,
    *,
    plan: ConfirmatoryMatrixPlan,
    controls: ConfirmatoryExecutionControls,
    artifact_scope: str,
    cpu_test_only: bool,
    cells_root: Path,
) -> None:
    if contract is None:
        if artifact_scope == REAL_CONFIRMATORY_ARTIFACT_SCOPE or not cpu_test_only:
            raise ConfirmatoryCheckpointContractError(
                "real confirmatory matrix requires an explicit checkpoint contract"
            )
        if os.path.lexists(cells_root):
            raise ConfirmatoryCheckpointContractError(
                "synthetic fresh matrix checkpoint root already exists"
            )
        return
    expected_count = len(contract.directives)
    if artifact_scope == REAL_CONFIRMATORY_ARTIFACT_SCOPE:
        expected_count = 180
        if contract.contract_profile != "original_confirmatory_exact_180":
            raise ConfirmatoryCheckpointContractError(
                "original confirmatory matrix requires the exact-180 checkpoint profile"
            )
        require_original_confirmatory_checkpoint_authority_binding(contract)
    elif artifact_scope == RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE:
        expected_count = 30
        if contract.contract_profile != "resource_bounded_confirmatory_exact_30":
            raise ConfirmatoryCheckpointContractError(
                "resource-bounded matrix requires the exact-30 checkpoint profile"
            )
    elif contract.contract_profile != "cpu_test_only":
        raise ConfirmatoryCheckpointContractError(
            "synthetic matrix requires the test-only checkpoint profile"
        )
    contract.validate(expected_directive_count=expected_count)
    cnn_cell_ids = {
        cell.cell_id
        for cell in plan.cells
        if controls.scenarios_by_id[cell.scenario_id].family == "cnn"
    }
    directive_cell_ids = {value.cell_id for value in contract.directives}
    if directive_cell_ids != cnn_cell_ids:
        raise ConfirmatoryCheckpointContractError(
            "checkpoint contract CNN-cell universe differs from the matrix plan"
        )
    for cell_id in sorted(cnn_cell_ids):
        contract.directives_for_cell(
            cell_id,
            expected_fold_count=controls.n_splits,
        )
    if contract.execution_mode == "fresh":
        if os.path.lexists(cells_root):
            raise ConfirmatoryCheckpointContractError(
                "fresh matrix requires an initially absent cells tree"
            )
        return
    observed = _checkpoint_tree_records(cells_root)
    expected = _expected_successor_tree_records(contract)
    if observed != expected:
        raise ConfirmatoryCheckpointContractError(
            "successor checkpoint tree differs from the exact copied directive set"
        )


def _prepare_cell_checkpoint_directory(
    checkpoint_directory: Path,
    directives: tuple[ConfirmatoryCheckpointDirective, ...],
) -> None:
    has_imported = any(value.action != "fresh_fit" for value in directives)
    if os.path.lexists(checkpoint_directory):
        try:
            observed = checkpoint_directory.lstat()
        except OSError as exc:
            raise ConfirmatoryCheckpointContractError(
                "cell checkpoint directory is unreadable"
            ) from exc
        if not stat.S_ISDIR(observed.st_mode) or _is_reparse_stat(observed):
            raise ConfirmatoryCheckpointContractError(
                "cell checkpoint directory is linked/reparse/invalid"
            )
        if _checkpoint_named_streams(checkpoint_directory):
            raise ConfirmatoryCheckpointContractError(
                "cell checkpoint directory contains a named stream"
            )
        return
    if has_imported:
        raise ConfirmatoryCheckpointContractError(
            "authorized imported checkpoint directory disappeared before its cell"
        )
    cell_directory = checkpoint_directory.parent
    if os.path.lexists(cell_directory):
        observed = cell_directory.lstat()
        if not stat.S_ISDIR(observed.st_mode) or _is_reparse_stat(observed):
            raise ConfirmatoryCheckpointContractError(
                "cell output directory is linked/reparse/invalid"
            )
        if _checkpoint_named_streams(cell_directory):
            raise ConfirmatoryCheckpointContractError(
                "cell output directory contains a named stream"
            )
    else:
        cell_directory.mkdir(exist_ok=False)
    checkpoint_directory.mkdir(exist_ok=False)
    if _checkpoint_named_streams(cell_directory) or _checkpoint_named_streams(checkpoint_directory):
        raise ConfirmatoryCheckpointContractError(
            "new checkpoint directory unexpectedly contains a named stream"
        )


def execute_confirmatory_matrix(
    rotations: Sequence[ConfirmatoryRotationInputs],
    plan: ConfirmatoryMatrixPlan,
    controls: ConfirmatoryExecutionControls,
    *,
    output_directory: str | Path,
    frozen_oof_runner: FrozenFeatureOOFRunner,
    image_oof_runner: ImageOOFRunner = run_confirmatory_image_oof,
    frozen_blockers: Mapping[str, ConfirmatoryFrozenBlocker] | None = None,
    artifact_scope: str = SYNTHETIC_CONFIRMATORY_ARTIFACT_SCOPE,
    cpu_test_only: bool = True,
    checkpoint_execution_contract: ConfirmatoryCheckpointExecutionContract | None = None,
    gate_evidence: object | None = None,
    progress_callback: ConfirmatoryProgressCallback | None = None,
) -> ConfirmatoryMatrixArtifacts:
    """Execute every frozen cell and reconcile exact fold/seed/scenario coverage."""

    controls.validate_for_plan(plan)
    if plan != controls.plan:
        raise ValueError("executor plan differs from frozen confirmatory controls")
    supported_scopes = {
        SYNTHETIC_CONFIRMATORY_ARTIFACT_SCOPE,
        REAL_CONFIRMATORY_ARTIFACT_SCOPE,
        RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
    }
    if artifact_scope not in supported_scopes:
        raise ValueError(f"unsupported confirmatory artifact scope: {artifact_scope!r}")
    if artifact_scope == REAL_CONFIRMATORY_ARTIFACT_SCOPE and plan.schema_version != 2:
        raise ValueError("real confirmatory artifact scope requires the original schema-v2 plan")
    if artifact_scope == RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE and (
        plan.schema_version != 3
        or plan.config_sha256 != RESOURCE_BOUNDED_CONFIRMATORY_CONFIG_SHA256
    ):
        raise ValueError(
            "resource-bounded artifact scope requires the exact frozen schema-v3 profile"
        )
    if artifact_scope == RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE:
        from histo_audit.workflows.study_gates import (
            ResourceBoundedExecutionGateEvidence,
        )

        if not isinstance(gate_evidence, ResourceBoundedExecutionGateEvidence):
            raise ValueError("resource-bounded matrix execution requires typed P+C gate evidence")
    blockers = dict(frozen_blockers or {})
    _validate_blockers(blockers, controls)
    by_fold: dict[int, ConfirmatoryRotationInputs] = {}
    for rotation in rotations:
        if rotation.outer_fold in by_fold:
            raise ValueError(f"duplicate rotation for official fold {rotation.outer_fold}")
        rotation.validate(controls)
        by_fold[rotation.outer_fold] = rotation
    if set(by_fold) != set(controls.official_folds) or len(by_fold) != 3:
        raise ValueError("executor requires exactly all three frozen official-fold rotations")

    destination = Path(output_directory).resolve()
    cells_root = destination / "cells"
    _require_matrix_checkpoint_contract(
        checkpoint_execution_contract,
        plan=plan,
        controls=controls,
        artifact_scope=artifact_scope,
        cpu_test_only=cpu_test_only,
        cells_root=cells_root,
    )
    destination.mkdir(parents=True, exist_ok=True)
    if not os.path.lexists(cells_root):
        cells_root.mkdir(exist_ok=False)
    reserved = (
        "matrix_plan.json",
        "execution_controls.json",
        "frozen_feature_provenance.json",
        "original_audit_selection.json",
        "cell_index.csv",
        "ensemble_evidence.json",
        "fixed_hybrid_drop_one_ablations.json",
        "matrix_core_fold_aggregate.json",
        "matrix_core_completion_evidence.json",
        "matrix_core_analysis_gaps.json",
        "matrix_core_figure_manifest.json",
        "matrix_core_report.md",
        "matrix_core_artifact_manifest.json",
        "fold_aggregate.json",
        "reconciliation.json",
        "completion_evidence.json",
        "analysis_gaps.json",
        "figure_manifest.json",
        "report.md",
        CONFIRMATORY_ARTIFACT_MANIFEST_FILENAME,
    )
    if any(os.path.lexists(destination / name) for name in reserved):
        raise FileExistsError("confirmatory output directory contains reserved artifacts")
    matrix_plan_path = atomic_write_json(destination / "matrix_plan.json", plan.as_dict())
    execution_controls_path = atomic_write_json(
        destination / "execution_controls.json", controls.as_dict()
    )
    frozen_feature_provenance_path = atomic_write_json(
        destination / "frozen_feature_provenance.json",
        _aggregate_frozen_feature_provenance(by_fold, controls),
    )
    original_audit_selection_path = atomic_write_json(
        destination / "original_audit_selection.json",
        _original_audit_selection_artifact(by_fold, controls),
    )

    executed: dict[str, _ExecutedCell] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    for cell in plan.cells:
        _emit_confirmatory_progress(
            progress_callback,
            cell=cell,
            status="started",
        )
        scenario = controls.scenarios_by_id[cell.scenario_id]
        rotation = by_fold[cell.outer_fold]
        if scenario.scenario_id in blockers:
            blocker = blockers[scenario.scenario_id]
            outcomes[cell.cell_id] = _persist_skipped_cell(
                cells_root / cell.cell_id,
                cell=cell,
                scenario=scenario,
                corruption=rotation.corruptions[cell.corruption_cell_id],
                controls=controls,
                blocker=blocker,
            )
            _emit_confirmatory_progress(
                progress_callback,
                cell=cell,
                status="skipped",
            )
            continue
        if scenario.family != "cnn" and scenario.representation_id not in (
            rotation.frozen_audit_features
        ):
            outcomes[cell.cell_id] = {
                **_base_outcome(
                    cell,
                    scenario,
                    rotation.corruptions[cell.corruption_cell_id],
                ),
                "status": "failed",
                "error": f"frozen representation unavailable: {scenario.representation_id}",
            }
            _emit_confirmatory_progress(
                progress_callback,
                cell=cell,
                status="failed",
            )
            continue
        checkpoint_directives: tuple[ConfirmatoryCheckpointDirective, ...] = ()
        if scenario.family == "cnn" and checkpoint_execution_contract is not None:
            checkpoint_directives = checkpoint_execution_contract.directives_for_cell(
                cell.cell_id,
                expected_fold_count=controls.n_splits,
            )
        checkpoint_directory = cells_root / cell.cell_id / "checkpoints"
        request = ConfirmatoryCellRequest(
            cell=cell,
            scenario=scenario,
            corruption=_runner_corruption_input(rotation.corruptions[cell.corruption_cell_id]),
            inputs=ConfirmatoryRunnerInputs.from_rotation(rotation),
            controls=controls,
            checkpoint_directory=checkpoint_directory,
            checkpoint_execution_contract=checkpoint_execution_contract,
            checkpoint_directives=checkpoint_directives,
            cpu_test_only=cpu_test_only,
        )
        if scenario.family == "cnn":
            if checkpoint_execution_contract is None:
                request.checkpoint_directory.mkdir(parents=True, exist_ok=False)
            else:
                _prepare_cell_checkpoint_directory(
                    request.checkpoint_directory,
                    checkpoint_directives,
                )
        try:
            if scenario.family == "cnn":
                image_result = image_oof_runner(request)
                if not isinstance(image_result, ConfirmatoryImageOOFResult):
                    raise TypeError("CNN runner must return ConfirmatoryImageOOFResult")
                execution = _normalise_image_execution(request, image_result)
            else:
                frozen_result = frozen_oof_runner(request)
                if not isinstance(frozen_result, FrozenFeatureOOFExecution):
                    raise TypeError("frozen runner must return FrozenFeatureOOFExecution")
                execution = _normalise_frozen_execution(request, frozen_result)
            if cpu_test_only and execution.study_outcome_eligible:
                raise ValueError("CPU test-only execution cannot be study-outcome eligible")
            expected_mode = "real_study_cuda" if scenario.family == "cnn" else "real_study_cpu"
            if not cpu_test_only and (
                execution.execution_mode != expected_mode or not execution.study_outcome_eligible
            ):
                raise ValueError(
                    "real confirmatory output lacks scenario-appropriate execution evidence"
                )
            executed[cell.cell_id] = execution
        except ConfirmatoryCheckpointContractError:
            raise
        except Exception as exc:
            outcomes[cell.cell_id] = {
                **_base_outcome(cell, scenario, request.corruption),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            atomic_write_json(
                cells_root / cell.cell_id / "failure.json",
                {"error_type": type(exc).__name__, "error": str(exc)},
            )
            _emit_confirmatory_progress(
                progress_callback,
                cell=cell,
                status="failed",
            )
        else:
            _emit_confirmatory_progress(
                progress_callback,
                cell=cell,
                status="model_completed",
            )

    ensemble_member_keys = {
        (member.scenario_id, member.model_seed) for member in controls.ensemble_members
    }
    for outer_fold in controls.official_folds:
        for corruption in controls.corruption_specs:
            group_cells = [
                cell
                for cell in plan.cells
                if cell.outer_fold == outer_fold
                and cell.corruption_cell_id == corruption.corruption_cell_id
            ]
            member_cells = [
                cell
                for cell in group_cells
                if (cell.scenario_id, cell.model_seed) in ensemble_member_keys
            ]
            if len(member_cells) != len(ensemble_member_keys) or any(
                cell.cell_id not in executed for cell in member_cells
            ):
                message = "frozen ensemble members did not all complete"
                for cell in group_cells:
                    if cell.cell_id in executed and cell.cell_id not in outcomes:
                        outcomes[cell.cell_id] = {
                            **_base_outcome(
                                cell,
                                controls.scenarios_by_id[cell.scenario_id],
                                by_fold[cell.outer_fold].corruptions[cell.corruption_cell_id],
                            ),
                            "status": "failed",
                            "error": message,
                        }
                        _emit_confirmatory_progress(
                            progress_callback,
                            cell=cell,
                            status="failed",
                        )
                continue
            try:
                member_executions = [executed[cell.cell_id] for cell in member_cells]
                observed = (
                    by_fold[outer_fold].corruptions[corruption.corruption_cell_id].observed_labels
                )
                ensemble = ensemble_disagreement(
                    [item.oof.probabilities for item in member_executions],
                    observed_labels=observed,
                    class_order=CLASS_ORDER,
                )
                risks = _ensemble_risks(ensemble)
                ensemble_primary = predeclared_ensemble_risk(
                    ensemble,
                    primary_risk=cast(Any, controls.ensemble_primary_risk),
                )
                for cell in group_cells:
                    cell_execution = executed.get(cell.cell_id)
                    if cell_execution is None or cell.cell_id in outcomes:
                        continue
                    self_confidence = score_annotations(
                        observed,
                        cell_execution.oof.probabilities,
                        method="self_confidence",
                        class_order=CLASS_ORDER,
                    )
                    components = {
                        "self_confidence": self_confidence,
                        "ensemble_disagreement": ensemble_primary,
                    }
                    hybrid = fixed_hybrid_drop_one_ablations(
                        components,
                        components=controls.hybrid_components,
                        weights=controls.hybrid_weights,
                    )
                    outcome = _persist_completed_cell(
                        cells_root / cell.cell_id,
                        cell_execution,
                        ensemble,
                        risks,
                        self_confidence,
                        hybrid,
                    )
                    outcomes[cell.cell_id] = outcome
            except ConfirmatoryCheckpointContractError:
                raise
            except Exception as exc:
                for cell in group_cells:
                    if cell.cell_id in executed and cell.cell_id not in outcomes:
                        outcomes[cell.cell_id] = {
                            **_base_outcome(
                                cell,
                                controls.scenarios_by_id[cell.scenario_id],
                                by_fold[cell.outer_fold].corruptions[cell.corruption_cell_id],
                            ),
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        _emit_confirmatory_progress(
                            progress_callback,
                            cell=cell,
                            status="failed",
                        )

    ordered_outcomes = tuple(outcomes[cell.cell_id] for cell in plan.cells)
    reconciliation = reconcile_confirmatory_cell_outcomes(plan, ordered_outcomes)
    reconciliation_path = atomic_write_json(
        destination / "reconciliation.json", reconciliation.as_dict()
    )
    cell_by_id = {cell.cell_id: cell for cell in plan.cells}
    all_completed_eligible = all(
        execution.study_outcome_eligible
        and execution.execution_mode
        == (
            "real_study_cuda"
            if controls.scenarios_by_id[cell_by_id[cell_id].scenario_id].family == "cnn"
            else "real_study_cpu"
        )
        for cell_id, execution in executed.items()
    )
    # This matrix core deliberately does not fabricate the preregistered paired
    # statistics, four-condition restoration, figures, or final report.  A stage
    # runner must add and reconcile those artifacts before it may request
    # CONFIRMATORY_COMPLETE, even when every model cell ran on its eligible execution
    # path (CUDA for CNNs; CPU for frozen-feature logistic models).
    stage_analysis_complete = False
    technical_execution_payload = {
        "schema_version": 1,
        "policy": "real_per_cell_execution_telemetry_and_reconciliation_v1",
        "cpu_test_only": cpu_test_only,
        "reconciliation_sha256": _canonical_sha256(reconciliation.as_dict()),
        "cells": [
            {
                "cell_id": cell_id,
                "execution_mode": execution.execution_mode,
                "study_outcome_eligible": execution.study_outcome_eligible,
                "configuration_sha256": execution.configuration_sha256,
            }
            for cell_id, execution in sorted(executed.items())
        ],
    }
    model_matrix_execution_eligible = bool(
        artifact_scope
        in {
            REAL_CONFIRMATORY_ARTIFACT_SCOPE,
            RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
        }
        and not cpu_test_only
        and reconciliation.passed
        and len(executed) == reconciliation.completed_cell_count
        and all_completed_eligible
    )
    study_outcome_eligible = bool(
        artifact_scope == REAL_CONFIRMATORY_ARTIFACT_SCOPE
        and model_matrix_execution_eligible
        and stage_analysis_complete
    )
    completion = build_confirmatory_completion_evidence(
        plan=plan,
        reconciliation=reconciliation,
        artifact_scope=artifact_scope,
        study_outcome_eligible=study_outcome_eligible,
        gate_evidence=cast(Any, gate_evidence),
    )
    completion.update(
        model_matrix_execution_eligible=model_matrix_execution_eligible,
        matrix_execution_telemetry_sha256=_canonical_sha256(technical_execution_payload),
        matrix_execution_eligibility_source=(
            "real_per_cell_execution_telemetry_and_reconciliation"
        ),
    )
    completion_evidence_path = atomic_write_json(
        destination / "matrix_core_completion_evidence.json", completion
    )
    cell_index_path = _write_csv(destination / "cell_index.csv", ordered_outcomes)
    ensemble_evidence_path = atomic_write_json(
        destination / "ensemble_evidence.json",
        _root_ensemble_evidence(plan, controls, ordered_outcomes),
    )
    hybrid_ablations_path = atomic_write_json(
        destination / "fixed_hybrid_drop_one_ablations.json",
        _root_hybrid_evidence(destination, controls, ordered_outcomes),
    )
    fold_aggregate_path = atomic_write_json(
        destination / "matrix_core_fold_aggregate.json",
        _root_fold_aggregate(controls, ordered_outcomes),
    )
    analysis_gaps_path = atomic_write_json(
        destination / "matrix_core_analysis_gaps.json",
        {
            "schema_version": 1,
            "model_matrix_execution_eligible": model_matrix_execution_eligible,
            "matrix_execution_telemetry_sha256": _canonical_sha256(technical_execution_payload),
            "matrix_execution_eligibility_source": (
                "real_per_cell_execution_telemetry_and_reconciliation"
            ),
            "study_outcome_eligible": False,
            "missing_stage_analyses": [
                "paired_statistics.json",
                "paired_bootstrap_evidence.npz",
                "restoration_metrics.json",
                "verified figures and final scientific report",
            ],
            "reason": (
                "matrix core stops before paired inference, four-condition downstream "
                "restoration, and reporting; an outer sealed stage runner must supply and "
                "reconcile them"
            ),
        },
    )
    figure_manifest_path = atomic_write_json(
        destination / "matrix_core_figure_manifest.json",
        {
            "schema_version": 1,
            "status": "not_generated_by_matrix_core",
            "figures": [],
            "study_outcome_eligible": False,
            "blocker": "paired statistics and restoration analyses are not executed here",
        },
    )
    report_path = atomic_write_text(
        destination / "matrix_core_report.md",
        _structural_report(controls, reconciliation, model_matrix_execution_eligible),
    )
    root_paths = (
        matrix_plan_path,
        execution_controls_path,
        frozen_feature_provenance_path,
        original_audit_selection_path,
        cell_index_path,
        ensemble_evidence_path,
        hybrid_ablations_path,
        fold_aggregate_path,
        reconciliation_path,
        completion_evidence_path,
        analysis_gaps_path,
        figure_manifest_path,
        report_path,
    )
    cell_manifests = tuple(
        cells_root / str(row["cell_id"]) / "artifact_manifest.json"
        for row in ordered_outcomes
        if row["status"] in {"completed", "skipped_with_frozen_blocker"}
    )
    artifact_manifest_path = _publish_scientific_hash_manifest(
        destination / "matrix_core_artifact_manifest.json",
        (*root_paths, *cell_manifests),
        relative_to=destination,
    )
    return ConfirmatoryMatrixArtifacts(
        output_directory=destination,
        matrix_plan_path=matrix_plan_path,
        execution_controls_path=execution_controls_path,
        frozen_feature_provenance_path=frozen_feature_provenance_path,
        original_audit_selection_path=original_audit_selection_path,
        cell_index_path=cell_index_path,
        ensemble_evidence_path=ensemble_evidence_path,
        hybrid_ablations_path=hybrid_ablations_path,
        fold_aggregate_path=fold_aggregate_path,
        reconciliation_path=reconciliation_path,
        completion_evidence_path=completion_evidence_path,
        analysis_gaps_path=analysis_gaps_path,
        figure_manifest_path=figure_manifest_path,
        report_path=report_path,
        artifact_manifest_path=artifact_manifest_path,
        outcomes=ordered_outcomes,
        reconciliation=reconciliation,
        study_outcome_eligible=study_outcome_eligible,
    )


def _root_ensemble_evidence(
    plan: ConfirmatoryMatrixPlan,
    controls: ConfirmatoryExecutionControls,
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    status_by_cell = {str(row["cell_id"]): str(row["status"]) for row in outcomes}
    member_keys = {(member.scenario_id, member.model_seed) for member in controls.ensemble_members}
    groups: list[dict[str, Any]] = []
    for outer_fold in controls.official_folds:
        for corruption in controls.corruption_specs:
            member_cells = [
                cell
                for cell in plan.cells
                if cell.outer_fold == outer_fold
                and cell.corruption_cell_id == corruption.corruption_cell_id
                and (cell.scenario_id, cell.model_seed) in member_keys
            ]
            groups.append(
                {
                    "outer_fold": outer_fold,
                    "corruption_cell_id": corruption.corruption_cell_id,
                    "member_cell_ids": [cell.cell_id for cell in member_cells],
                    "all_members_completed": all(
                        status_by_cell.get(cell.cell_id) == "completed" for cell in member_cells
                    ),
                }
            )
    return {
        "schema_version": 1,
        "config_semantic_sha256": controls.config_semantic_sha256,
        "outer_folds": list(controls.official_folds),
        "corruption_cell_ids": [item.corruption_cell_id for item in controls.corruption_specs],
        "members": [asdict(member) for member in controls.ensemble_members],
        "primary_risk": controls.ensemble_primary_risk,
        "secondary_risks": list(controls.ensemble_secondary_risks),
        "groups": groups,
        "risk_arrays_are_saved_per_cell": "cells/<cell_id>/risk_scores.npz",
    }


def _root_hybrid_evidence(
    destination: Path,
    controls: ConfirmatoryExecutionControls,
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cell_evidence: list[dict[str, Any]] = []
    for outcome in outcomes:
        if outcome["status"] != "completed":
            continue
        relative = Path("cells") / str(outcome["cell_id"]) / "risk_scores.npz"
        path = destination / relative
        cell_evidence.append(
            {
                "cell_id": outcome["cell_id"],
                "risk_scores_path": relative.as_posix(),
                "risk_scores_sha256": _scientific_artifact_record(path).sha256,
            }
        )
    return {
        "schema_version": 1,
        "config_semantic_sha256": controls.config_semantic_sha256,
        "outer_folds": list(controls.official_folds),
        "components": list(controls.hybrid_components),
        "weights": list(controls.hybrid_weights),
        "drop_one_ablations": list(controls.hybrid_drop_one_ablations),
        "cell_evidence": cell_evidence,
    }


def _root_fold_aggregate(
    controls: ConfirmatoryExecutionControls,
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for outer_fold in controls.official_folds:
        rows = [row for row in outcomes if row.get("outer_fold") == outer_fold]
        folds.append(
            {
                "outer_fold": outer_fold,
                "planned_cell_count": len(rows),
                "completed_cell_count": sum(row["status"] == "completed" for row in rows),
                "skipped_optional_cell_count": sum(
                    row["status"] == "skipped_with_frozen_blocker" for row in rows
                ),
                "failed_cell_count": sum(row["status"] == "failed" for row in rows),
                "reported_separately": True,
            }
        )
    return {
        "schema_version": 1,
        "outer_folds": list(controls.official_folds),
        "aggregate_policy": "report_each_rotation_and_descriptive_fold_mean",
        "folds": folds,
        "outcome_metrics_aggregation_status": "deferred_to_stage_statistics_runner",
    }


def _structural_report(
    controls: ConfirmatoryExecutionControls,
    reconciliation: ConfirmatoryMatrixReconciliation,
    model_matrix_execution_eligible: bool,
) -> str:
    return "\n".join(
        (
            "# Confirmatory matrix structural execution",
            "",
            f"- Frozen config SHA-256: `{controls.config_semantic_sha256}`",
            f"- Planned cells: {reconciliation.planned_cell_count}",
            f"- Completed cells: {reconciliation.completed_cell_count}",
            f"- Skipped optional cells: {reconciliation.skipped_optional_cell_count}",
            f"- Exact reconciliation: `{reconciliation.status}`",
            "- Scenario-appropriate model execution eligibility: "
            f"`{str(model_matrix_execution_eligible).lower()}`",
            "- Study-outcome eligibility: `false`",
            "",
            "This is a structural matrix report, not a completed confirmatory-study report. ",
            "Preregistered paired statistics, four-condition downstream restoration, verified ",
            "figures, sealing, and integrity readback remain mandatory before ",
            "`CONFIRMATORY_COMPLETE` can be enabled.",
            "",
        )
    )


def _persist_skipped_cell(
    directory: Path,
    *,
    cell: ConfirmatoryCell,
    scenario: ConfirmatoryScenarioSpec,
    corruption: ConfirmatoryCorruptionInput,
    controls: ConfirmatoryExecutionControls,
    blocker: ConfirmatoryFrozenBlocker,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    identity_path = atomic_write_json(
        directory / "cell_identity.json",
        {
            "schema_version": 1,
            "cell_id": cell.cell_id,
            "outer_fold": cell.outer_fold,
            "corruption_cell_id": corruption.corruption_cell_id,
            "corruption_mechanism": corruption.mechanism,
            "corruption_rate": corruption.rate,
            "corruption_seed": corruption.seed,
            "scenario_id": scenario.scenario_id,
            "scenario_family": scenario.family,
            "representation_id": scenario.representation_id,
            "cache_provenance_id": scenario.cache_provenance_id,
            "model_seed": cell.model_seed,
            "required": cell.required,
            "config_semantic_sha256": controls.config_semantic_sha256,
        },
    )
    cache_record = controls.cache_provenance_by_id[scenario.cache_provenance_id]
    blocker_evidence_sha256 = str(
        cache_record.get("blocker_evidence_sha256", blocker.availability_audit_sha256)
    )
    blocker_path = atomic_write_json(
        directory / "blocker.json",
        {
            "schema_version": 1,
            "cell_id": cell.cell_id,
            "frozen_unavailability": True,
            "blocker": blocker.blocker,
            "blocker_evidence_sha256": blocker_evidence_sha256,
            "cache_provenance_id": scenario.cache_provenance_id,
            "config_semantic_sha256": controls.config_semantic_sha256,
        },
    )
    manifest_path = _publish_scientific_hash_manifest(
        directory / "artifact_manifest.json",
        (identity_path, blocker_path),
    )
    return {
        **_base_outcome(cell, scenario, corruption),
        "status": "skipped_with_frozen_blocker",
        "frozen_unavailability": True,
        "availability_audit_sha256": blocker.availability_audit_sha256,
        "blocker": blocker.blocker,
        "artifact_manifest_sha256": _scientific_artifact_record(manifest_path).sha256,
    }


def _persist_completed_cell(
    directory: Path,
    execution: _ExecutedCell,
    ensemble: EnsembleDisagreementResult,
    ensemble_risks: Mapping[str, NDArray[np.float64]],
    self_confidence: NDArray[np.float64],
    hybrid: FixedHybridDropOneResult,
) -> dict[str, Any]:
    request = execution.request
    cell = request.cell
    directory.mkdir(parents=True, exist_ok=True)
    cell_identity = {
        "schema_version": 1,
        "cell_id": cell.cell_id,
        "outer_fold": cell.outer_fold,
        "corruption_cell_id": request.corruption.corruption_cell_id,
        "corruption_mechanism": request.corruption.mechanism,
        "corruption_rate": request.corruption.rate,
        "corruption_seed": request.corruption.seed,
        "scenario_id": request.scenario.scenario_id,
        "scenario_family": request.scenario.family,
        "representation_id": request.scenario.representation_id,
        "cache_provenance_id": request.scenario.cache_provenance_id,
        "model_seed": cell.model_seed,
        "required": cell.required,
        "config_semantic_sha256": request.controls.config_semantic_sha256,
    }
    cell_identity_path = atomic_write_json(
        directory / "cell_identity.json",
        cell_identity,
    )
    oof_path = _atomic_npz(
        directory / "oof_evidence.npz",
        {
            "sample_ids": np.asarray(request.inputs.audit_sample_ids, dtype=np.str_),
            "group_ids": np.asarray(request.inputs.audit_group_ids, dtype=np.str_),
            "pre_corruption_label": request.corruption.pre_corruption_labels,
            "observed_label": request.corruption.observed_labels,
            "is_injected_corruption": request.corruption.is_injected_corruption,
            "probabilities": execution.oof.probabilities,
            "fold_id": execution.oof.fold_id,
            "fold_assignment_labels": execution.oof.fold_assignment_labels,
            "coverage_count": execution.oof.coverage_count,
        },
    )
    ensemble_primary = predeclared_ensemble_risk(
        ensemble,
        primary_risk=cast(Any, request.controls.ensemble_primary_risk),
    )
    risk_path = _atomic_npz(
        directory / "risk_scores.npz",
        {
            "sample_ids": np.asarray(request.inputs.audit_sample_ids, dtype=np.str_),
            "ensemble_mean_probabilities": ensemble.averaged_probabilities,
            "self_confidence": self_confidence,
            "ensemble_disagreement": ensemble_primary,
            "fixed_hybrid": hybrid.full_score,
            **{f"ensemble_{name}": values for name, values in ensemble_risks.items()},
            **{f"hybrid_drop_{name}": values for name, values in hybrid.drop_one_scores.items()},
        },
    )
    if request.scenario.family == "cnn":
        checkpoint_rows: list[dict[str, Any]] = []
        with ExitStack() as checkpoint_sources:
            for row in cast(
                Sequence[Mapping[str, Any]],
                execution.execution_evidence.get("fold_evidence", []),
            ):
                checkpoint_file = Path(str(row["checkpoint_path"])).resolve()
                checkpoint_identity = _checkpoint_physical_identity_from_evidence(
                    row.get("checkpoint_physical_identity"),
                    role="persisted checkpoint",
                )
                try:
                    relative_checkpoint = checkpoint_file.relative_to(directory.resolve())
                except ValueError as error:
                    raise ValueError(
                        "checkpoint is outside its immutable cell artifact directory"
                    ) from error
                checkpoint_execution_manifest_value = row.get("checkpoint_execution_manifest_path")
                checkpoint_execution_manifest_sha256 = row.get(
                    "checkpoint_execution_manifest_sha256"
                )
                if not isinstance(checkpoint_execution_manifest_value, str) or not isinstance(
                    checkpoint_execution_manifest_sha256, str
                ):
                    raise ValueError("checkpoint execution manifest evidence is incomplete")
                checkpoint_execution_manifest = Path(checkpoint_execution_manifest_value).resolve()
                checkpoint_execution_manifest_identity = (
                    _checkpoint_physical_identity_from_evidence(
                        row.get("checkpoint_execution_manifest_physical_identity"),
                        role="persisted checkpoint execution manifest",
                    )
                )
                try:
                    relative_execution_manifest = checkpoint_execution_manifest.relative_to(
                        directory.resolve()
                    )
                except ValueError as error:
                    raise ValueError(
                        "checkpoint execution manifest is outside its immutable "
                        "cell artifact directory"
                    ) from error
                held_checkpoint = checkpoint_sources.enter_context(
                    _hold_private_checkpoint_snapshot(
                        checkpoint_file,
                        role="persisted image-OOF checkpoint",
                    )
                )
                held_manifest = checkpoint_sources.enter_context(
                    _hold_private_checkpoint_snapshot(
                        checkpoint_execution_manifest,
                        role="persisted image-OOF checkpoint execution manifest",
                    )
                )
                if (
                    held_checkpoint.identity != checkpoint_identity
                    or held_checkpoint.sha256 != row["checkpoint_sha256"]
                    or held_manifest.identity != checkpoint_execution_manifest_identity
                    or held_manifest.sha256 != checkpoint_execution_manifest_sha256
                ):
                    raise ValueError(
                        "checkpoint/execution manifest differs from image-OOF fold evidence"
                    )
                canonical_working: dict[str, Any] | None = None
                versioned_history: tuple[dict[str, Any], ...] = ()
                expected_directive: ConfirmatoryCheckpointDirective | None = None
                if not request.cpu_test_only:
                    try:
                        execution_manifest_payload = json.loads(
                            held_manifest.payload.decode("ascii")
                        )
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ConfirmatoryCheckpointContractError(
                            "persisted checkpoint execution manifest is not canonical JSON"
                        ) from exc
                    directive_by_fold = {
                        directive.fold_id: directive for directive in request.checkpoint_directives
                    }
                    fold_id = int(row["fold_id"])
                    expected_directive = directive_by_fold.get(fold_id)
                    if expected_directive is None:
                        raise ConfirmatoryCheckpointContractError(
                            "persisted checkpoint lacks its exact execution directive"
                        )
                    run_directory = request.checkpoint_directory.resolve().parents[2]
                    _require_exact_checkpoint_execution_manifest_payload(
                        execution_manifest_payload,
                        expected_directive,
                        run_directory=run_directory,
                    )
                    canonical_working = _hold_checkpoint_execution_canonical(
                        checkpoint_sources,
                        execution_manifest_payload,
                        run_directory=run_directory,
                        role="persisted image-OOF execution",
                    )
                    versioned_history = _hold_checkpoint_execution_history(
                        checkpoint_sources,
                        execution_manifest_payload,
                        run_directory=run_directory,
                        role="persisted image-OOF execution",
                    )
                    if list(versioned_history) != row.get("checkpoint_versioned_outputs"):
                        raise ConfirmatoryCheckpointContractError(
                            "persisted checkpoint history differs from normalised fold evidence"
                        )
                    if canonical_working is None or canonical_working != row.get(
                        "checkpoint_canonical_working"
                    ):
                        raise ConfirmatoryCheckpointContractError(
                            "persisted canonical working checkpoint differs from "
                            "normalised fold evidence"
                        )
                checkpoint_row: dict[str, Any] = {
                    "fold_id": row["fold_id"],
                    "status": "complete",
                    "path": relative_checkpoint.as_posix(),
                    "sha256": row["checkpoint_sha256"],
                    "configuration_sha256": row["configuration_sha256"],
                }
                if not request.cpu_test_only:
                    if (
                        expected_directive is None or canonical_working is None
                    ):  # pragma: no cover - guarded above
                        raise ConfirmatoryCheckpointContractError(
                            "real checkpoint row lacks its exact directive/canonical evidence"
                        )
                    checkpoint_row.update(
                        {
                            "physical_identity": checkpoint_identity.as_dict(),
                            "execution_manifest_path": (relative_execution_manifest.as_posix()),
                            "execution_manifest_sha256": (checkpoint_execution_manifest_sha256),
                            "execution_manifest_physical_identity": (
                                checkpoint_execution_manifest_identity.as_dict()
                            ),
                            "directive": expected_directive.as_dict(),
                            "directive_sha256": expected_directive.directive_sha256,
                            "canonical_working_checkpoint": dict(canonical_working),
                            "versioned_outputs": list(versioned_history),
                        }
                    )
                checkpoint_rows.append(checkpoint_row)
            checkpoint_manifest: dict[str, Any] = {
                "schema_version": 1 if request.cpu_test_only else 3,
                "cell_id": cell.cell_id,
                "status": "complete",
                "checkpoints": checkpoint_rows,
            }
            checkpoint_manifest_path = atomic_write_json(
                directory / "checkpoint_manifest.json",
                checkpoint_manifest,
            )
            checkpoint_manifest_record = _scientific_artifact_record(checkpoint_manifest_path)
            held_checkpoint_manifest = checkpoint_sources.enter_context(
                _hold_private_checkpoint_snapshot(
                    checkpoint_manifest_path,
                    role="persisted checkpoint index",
                )
            )
            _require_scientific_snapshot(
                held_checkpoint_manifest,
                checkpoint_manifest_record,
                role="persisted checkpoint index",
            )
    else:
        checkpoint_manifest = {
            "schema_version": 1,
            "cell_id": cell.cell_id,
            "status": "not_applicable_frozen_feature",
            "checkpoints": [],
        }
        checkpoint_manifest_path = atomic_write_json(
            directory / "checkpoint_manifest.json",
            checkpoint_manifest,
        )
    telemetry_path = atomic_write_json(
        directory / "telemetry.json",
        {
            "schema_version": 1,
            "cell_id": cell.cell_id,
            "execution_mode": execution.execution_mode,
            "study_outcome_eligible": execution.study_outcome_eligible,
            "configuration_sha256": execution.configuration_sha256,
            "evidence": dict(execution.execution_evidence),
            "splitter_class_name": execution.oof.splitter_class_name,
            "splitter_fallback_status": execution.oof.splitter_fallback_status,
            "splitter_fallback_reason": execution.oof.splitter_fallback_reason,
            "folds": [asdict(fold) for fold in execution.oof.folds],
        },
    )
    metrics_path = atomic_write_json(
        directory / "metrics.json",
        {
            "cell_identity": cell_identity,
            "cell": asdict(cell),
            "scenario": asdict(request.scenario),
            "corruption": {
                "corruption_cell_id": request.corruption.corruption_cell_id,
                "mechanism": request.corruption.mechanism,
                "rate": request.corruption.rate,
                "seed": request.corruption.seed,
                "injected_count": int(request.corruption.is_injected_corruption.sum()),
            },
            "oof_coverage_exactly_once": bool(np.all(execution.oof.coverage_count == 1)),
            "oof_splitter_class_name": execution.oof.splitter_class_name,
            "oof_splitter_fallback_status": execution.oof.splitter_fallback_status,
            "oof_splitter_fallback_reason": execution.oof.splitter_fallback_reason,
            "ensemble_member_count": ensemble.model_count,
            "ensemble_primary_risk": request.controls.ensemble_primary_risk,
            "hybrid_components": list(hybrid.components),
            "hybrid_weights": list(hybrid.normalised_weights),
            "ranking": {
                "self_confidence": {
                    "average_precision": average_precision(
                        request.corruption.is_injected_corruption, self_confidence
                    ),
                    "auroc": binary_auroc(
                        request.corruption.is_injected_corruption, self_confidence
                    ),
                },
                "ensemble_disagreement": {
                    "average_precision": average_precision(
                        request.corruption.is_injected_corruption,
                        ensemble_primary,
                    )
                },
                "fixed_hybrid": {
                    "average_precision": average_precision(
                        request.corruption.is_injected_corruption, hybrid.full_score
                    )
                },
            },
        },
    )
    ranking_order = rank_indices(
        hybrid.full_score,
        tie_break_ids=request.inputs.audit_sample_ids,
    )
    ranking_path = _write_csv(
        directory / "ranking.csv",
        [
            {
                "rank": rank,
                "sample_id": request.inputs.audit_sample_ids[index],
                "risk_method": "fixed_hybrid",
                "risk_score": float(hybrid.full_score[index]),
            }
            for rank, index in enumerate(ranking_order, start=1)
        ],
    )
    manifest_sources = (
        cell_identity_path,
        oof_path,
        checkpoint_manifest_path,
        telemetry_path,
        risk_path,
        ranking_path,
        metrics_path,
    )
    manifest_path = _publish_scientific_hash_manifest(
        directory / "artifact_manifest.json",
        manifest_sources,
    )
    return {
        **_base_outcome(cell, request.scenario, request.corruption),
        "status": "completed",
        "artifact_manifest_sha256": _scientific_artifact_record(manifest_path).sha256,
        "metrics_sha256": _scientific_artifact_record(metrics_path).sha256,
        "execution_mode": execution.execution_mode,
        "study_outcome_eligible": execution.study_outcome_eligible,
    }


def _synthetic_oof(request: ConfirmatoryCellRequest) -> OOFResult:
    from histo_audit.cross_validation.oof import make_group_stratified_folds

    labels = request.corruption.pre_corruption_labels
    observed = request.corruption.observed_labels
    groups = request.inputs.audit_group_ids
    folds = make_group_stratified_folds(
        labels,
        groups,
        n_splits=request.controls.n_splits,
        class_order=CLASS_ORDER,
        seed=request.controls.split_seed,
    )
    rng = np.random.default_rng(request.cell.model_seed)
    probabilities = np.full((len(labels), len(CLASS_ORDER)), 0.075, dtype=np.float64)
    probabilities[np.arange(len(labels)), observed] = 0.70
    probabilities += rng.uniform(0.0, 0.005, size=probabilities.shape)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    fold_ids = np.full(len(labels), -1, dtype=np.int64)
    provenance: list[OOFFoldProvenance] = []
    for fold in folds:
        fold_ids[fold.holdout_indices] = fold.fold_id
        provenance.append(
            OOFFoldProvenance(
                fold_id=fold.fold_id,
                training_groups=fold.training_groups,
                held_out_groups=fold.held_out_groups,
                held_out_sample_ids=tuple(
                    request.inputs.audit_sample_ids[index] for index in fold.holdout_indices
                ),
            )
        )
    assignment_hash = _canonical_sha256([int(value) for value in labels])
    result = OOFResult(
        probabilities=probabilities,
        predicted_class=np.asarray(CLASS_ORDER)[np.argmax(probabilities, axis=1)],
        fold_id=fold_ids,
        coverage_count=np.ones(len(labels), dtype=np.int64),
        sample_ids=request.inputs.audit_sample_ids,
        group_ids=request.inputs.audit_group_ids,
        final_reference_groups=request.inputs.final_reference_group_ids,
        class_order=CLASS_ORDER,
        folds=tuple(provenance),
        model_name="synthetic_contract_adapter",
        representation=request.scenario.representation_id,
        model_seed=request.cell.model_seed,
        split_seed=request.controls.split_seed,
        fold_assignment_labels=labels.copy(),
        fold_assignment_label_source="pre_corruption_label",
        fold_assignment_labels_sha256=assignment_hash,
    )
    result.validate()
    return result


def _synthetic_frozen_runner(request: ConfirmatoryCellRequest) -> FrozenFeatureOOFExecution:
    provenance = request.inputs.frozen_feature_provenance[request.scenario.representation_id]
    return FrozenFeatureOOFExecution(
        oof_result=_synthetic_oof(request),
        execution_mode="cpu_test_only_non_evidence",
        study_outcome_eligible=False,
        configuration_sha256=_canonical_sha256(
            {"adapter": "synthetic_frozen", "scenario": request.scenario.scenario_id}
        ),
        evidence={
            "synthetic_contract_fixture": True,
            "frozen_feature_provenance_sha256": provenance.semantic_sha256,
        },
    )


def _synthetic_image_runner(request: ConfirmatoryCellRequest) -> ConfirmatoryImageOOFResult:
    oof = _synthetic_oof(request)
    evidence: list[ConfirmatoryImageOOFFoldEvidence] = []
    for fold in oof.folds:
        checkpoint = request.checkpoint_directory / f"fold_{fold.fold_id:02d}.pt"
        atomic_write_text(
            checkpoint, f"synthetic;cell={request.cell.cell_id};fold={fold.fold_id}\n"
        )
        checkpoint_execution_manifest = request.checkpoint_directory / (
            f"fold_{fold.fold_id:02d}.synthetic.execution.json"
        )
        atomic_write_json(
            checkpoint_execution_manifest,
            {
                "schema_version": 1,
                "policy": "synthetic_checkpoint_commit_non_evidence",
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _scientific_artifact_record(checkpoint).sha256,
            },
        )
        held_out = set(fold.held_out_sample_ids)
        training_samples = tuple(
            sample_id for sample_id in oof.sample_ids if sample_id not in held_out
        )
        evidence.append(
            ConfirmatoryImageOOFFoldEvidence(
                fold_id=fold.fold_id,
                model_seed=request.cell.model_seed + fold.fold_id,
                training_sample_ids=training_samples,
                held_out_sample_ids=fold.held_out_sample_ids,
                training_groups=fold.training_groups,
                held_out_groups=fold.held_out_groups,
                reference_validation_sample_ids=request.inputs.reference_validation_sample_ids,
                reference_validation_groups=tuple(
                    sorted(set(request.inputs.reference_validation_group_ids))
                ),
                checkpoint_path=str(checkpoint),
                checkpoint_sha256=_scientific_artifact_record(checkpoint).sha256,
                checkpoint_size_bytes=checkpoint.stat().st_size,
                checkpoint_physical_identity=(
                    ConfirmatoryCheckpointPhysicalIdentity.from_stat(checkpoint.lstat())
                ),
                checkpoint_execution_manifest_path=str(checkpoint_execution_manifest),
                checkpoint_execution_manifest_sha256=(
                    _scientific_artifact_record(checkpoint_execution_manifest).sha256
                ),
                checkpoint_execution_manifest_physical_identity=(
                    ConfirmatoryCheckpointPhysicalIdentity.from_stat(
                        checkpoint_execution_manifest.lstat()
                    )
                ),
                configuration_sha256=_canonical_sha256(
                    {"cell": request.cell.cell_id, "fold": fold.fold_id}
                ),
                resumed_from_checkpoint=False,
                checkpoint_execution_mode="fresh",
                checkpoint_action="fresh_fit",
                checkpoint_sha256_before_fit=None,
                completed_epochs_before_fit=0,
                trained_epochs_this_invocation=1,
                successful_optimiser_steps_before_fit=0,
                successful_optimiser_steps_after_fit=1,
                successful_optimiser_steps_this_invocation=1,
                execution_mode="cpu_test_only_non_evidence",
                study_outcome_eligible=False,
                completed_epochs=1,
                best_epoch=0,
                best_reference_validation_loss=None,
                telemetry={
                    "execution_mode": "cpu_test_only_non_evidence",
                    "study_outcome_eligible": False,
                },
                model_metadata={"synthetic_contract_fixture": True},
                data_and_split_sha256={
                    role: _canonical_sha256(
                        {
                            "synthetic_contract_fixture": True,
                            "cell_id": request.cell.cell_id,
                            "fold_id": fold.fold_id,
                            "role": role,
                        }
                    )
                    for role in (
                        "training_data_sha256",
                        "reference_validation_data_sha256",
                        "training_split_sha256",
                        "reference_validation_split_sha256",
                    )
                },
            )
        )
    result = ConfirmatoryImageOOFResult(
        oof_result=oof,
        fold_evidence=tuple(evidence),
        study_outcome_eligible=False,
        execution_mode="cpu_test_only_non_evidence",
    )
    result.validate()
    return result


def run_synthetic_confirmatory_contract_fixture(
    config: Mapping[str, Any], *, output_directory: str | Path
) -> SyntheticConfirmatoryFixtureResult:
    """Run the full frozen matrix on tiny inputs; evidence is permanently ineligible."""

    controls = confirmatory_execution_controls_from_frozen_config(config)
    if any(spec.mechanism != "symmetric_random_corruption" for spec in controls.corruption_specs):
        raise ValueError("synthetic confirmatory fixture supports only symmetric corruption")
    rotations = tuple(
        _synthetic_rotation(outer_fold, controls) for outer_fold in controls.official_folds
    )
    blockers = {
        scenario.scenario_id: ConfirmatoryFrozenBlocker(
            scenario_id=scenario.scenario_id,
            config_semantic_sha256=controls.config_semantic_sha256,
            availability_audit_sha256=cast(str, scenario.availability_audit_sha256),
            blocker="frozen pathology encoder unavailable in synthetic contract fixture",
        )
        for scenario in controls.scenario_specs
        if not scenario.required
    }
    artifacts = execute_confirmatory_matrix(
        rotations,
        controls.plan,
        controls,
        output_directory=output_directory,
        frozen_oof_runner=_synthetic_frozen_runner,
        image_oof_runner=_synthetic_image_runner,
        frozen_blockers=blockers,
        artifact_scope=SYNTHETIC_CONFIRMATORY_ARTIFACT_SCOPE,
        cpu_test_only=True,
    )
    return SyntheticConfirmatoryFixtureResult(
        artifacts=artifacts,
        completed_cell_count=artifacts.reconciliation.completed_cell_count,
        skipped_optional_cell_count=artifacts.reconciliation.skipped_optional_cell_count,
    )


def _synthetic_rotation(
    outer_fold: int, controls: ConfirmatoryExecutionControls
) -> ConfirmatoryRotationInputs:
    other_folds = tuple(value for value in controls.official_folds if value != outer_fold)
    n_audit = 25
    prefix = f"rotation_{outer_fold}"
    pre = np.tile(np.arange(5, dtype=np.int64), 5)
    groups = tuple(f"{prefix}_audit_group_{index // 5}" for index in range(n_audit))
    audit_fold_values = np.asarray(
        [other_folds[(index // 5) % 2] for index in range(n_audit)], dtype=np.int64
    )
    rng = np.random.default_rng(8_000 + outer_fold)
    rgb = rng.integers(0, 256, size=(n_audit, 12, 12, 3), dtype=np.uint8)
    masks = np.zeros((n_audit, 12, 12), dtype=bool)
    masks[:, 4:8, 4:8] = True
    corruptions: dict[str, ConfirmatoryCorruptionInput] = {}
    for spec in controls.corruption_specs:
        observed = pre.copy()
        count = int(np.floor(n_audit * spec.rate + 0.5))
        if count:
            observed[:count] = (observed[:count] + 1) % len(CLASS_ORDER)
        injected = observed != pre
        corruptions[spec.corruption_cell_id] = ConfirmatoryCorruptionInput(
            corruption_cell_id=spec.corruption_cell_id,
            mechanism=spec.mechanism,
            rate=spec.rate,
            seed=spec.seed,
            parameters=dict(spec.parameters),
            pre_corruption_labels=pre.copy(),
            observed_labels=observed,
            is_injected_corruption=injected,
        )
    feature_representations = {
        scenario.representation_id
        for scenario in controls.scenario_specs
        if scenario.family != "cnn" and scenario.required
    }
    features = {
        representation: rng.normal(size=(n_audit, 8)).astype(np.float64)
        for representation in feature_representations
    }
    scenario_by_representation = {
        scenario.representation_id: scenario for scenario in controls.scenario_specs
    }
    available_representations = {
        scenario.representation_id
        for scenario in controls.scenario_specs
        if controls.cache_provenance_by_id[scenario.cache_provenance_id]["status"] == "available"
    }
    audit_sample_ids = tuple(f"{prefix}_audit_{index:03d}" for index in range(n_audit))
    provenance = {
        representation: FrozenFeatureProvenance(
            cache_provenance_id=scenario_by_representation[representation].cache_provenance_id,
            representation_id=representation,
            cache_file_sha256=controls.cache_provenance_by_id[
                scenario_by_representation[representation].cache_provenance_id
            ]["cache_file_sha256"],
            sidecar_semantic_sha256=controls.cache_provenance_by_id[
                scenario_by_representation[representation].cache_provenance_id
            ]["sidecar_semantic_sha256"],
            sample_order_sha256=str(
                controls.cache_provenance_by_id[
                    scenario_by_representation[representation].cache_provenance_id
                ]["sample_order_sha256"]
            ),
            manifest_sha256=str(
                controls.cache_provenance_by_id[
                    scenario_by_representation[representation].cache_provenance_id
                ]["manifest_sha256"]
            ),
            encoder_identifier=str(
                controls.cache_provenance_by_id[
                    scenario_by_representation[representation].cache_provenance_id
                ]["encoder_identifier"]
            ),
            encoder_metadata_sha256=str(
                controls.cache_provenance_by_id[
                    scenario_by_representation[representation].cache_provenance_id
                ]["encoder_metadata_sha256"]
            ),
            weight_identifier=str(
                controls.cache_provenance_by_id[
                    scenario_by_representation[representation].cache_provenance_id
                ]["weight_identifier"]
            ),
            weights_sha256=str(
                controls.cache_provenance_by_id[
                    scenario_by_representation[representation].cache_provenance_id
                ]["weights_sha256"]
            ),
            preprocessing_identifier=str(
                controls.cache_provenance_by_id[
                    scenario_by_representation[representation].cache_provenance_id
                ]["preprocessing_identifier"]
            ),
            preprocessing_sha256=str(
                controls.cache_provenance_by_id[
                    scenario_by_representation[representation].cache_provenance_id
                ]["preprocessing_sha256"]
            ),
            input_variant=str(
                controls.cache_provenance_by_id[
                    scenario_by_representation[representation].cache_provenance_id
                ]["input_variant"]
            ),
            audit_sample_order_sha256=_canonical_sha256(list(audit_sample_ids)),
        )
        for representation in available_representations
    }
    validation_rgb = rng.integers(0, 256, size=(5, 12, 12, 3), dtype=np.uint8)
    validation_masks = np.zeros((5, 12, 12), dtype=bool)
    validation_masks[:, 3:9, 3:9] = True
    return ConfirmatoryRotationInputs(
        outer_fold=outer_fold,
        audit_sample_ids=audit_sample_ids,
        audit_group_ids=groups,
        audit_official_folds=audit_fold_values,
        audit_rgb=rgb,
        audit_target_masks=masks,
        corruptions=corruptions,
        frozen_audit_features=features,
        frozen_feature_provenance=provenance,
        reference_validation_sample_ids=tuple(f"{prefix}_validation_{index}" for index in range(5)),
        reference_validation_group_ids=tuple(f"{prefix}_validation_group" for _ in range(5)),
        reference_validation_official_folds=np.full(5, other_folds[0], dtype=np.int64),
        reference_validation_labels=np.arange(5, dtype=np.int64),
        reference_validation_rgb=validation_rgb,
        reference_validation_target_masks=validation_masks,
        final_sample_ids=tuple(f"{prefix}_final_{index}" for index in range(5)),
        final_group_ids=tuple(f"{prefix}_final_group" for _ in range(5)),
        final_official_folds=np.full(5, outer_fold, dtype=np.int64),
        final_pre_corruption_labels=np.arange(5, dtype=np.int64),
        final_observed_labels=np.arange(5, dtype=np.int64),
        final_is_injected_corruption=np.zeros(5, dtype=bool),
    )


__all__ = [
    "ConfirmatoryCellRequest",
    "ConfirmatoryComparisonOperand",
    "ConfirmatoryCorruptionInput",
    "ConfirmatoryCorruptionSpec",
    "ConfirmatoryEnsembleMember",
    "ConfirmatoryExecutionControls",
    "ConfirmatoryFrozenBlocker",
    "ConfirmatoryMatrixArtifacts",
    "ConfirmatoryPairedComparison",
    "ConfirmatoryProgressCallback",
    "ConfirmatoryRotationInputs",
    "ConfirmatoryRunnerInputs",
    "ConfirmatoryScenarioSpec",
    "FrozenFeatureOOFExecution",
    "FrozenFeatureOOFRunner",
    "FrozenFeatureProvenance",
    "ImageOOFRunner",
    "SyntheticConfirmatoryFixtureResult",
    "confirmatory_cnn_config_for_cell",
    "confirmatory_execution_controls_from_frozen_config",
    "execute_confirmatory_matrix",
    "run_confirmatory_frozen_feature_oof",
    "run_confirmatory_image_oof",
    "run_synthetic_confirmatory_contract_fixture",
]
