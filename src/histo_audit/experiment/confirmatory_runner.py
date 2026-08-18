"""Production, fail-closed orchestration for the frozen confirmatory study.

This module deliberately separates three boundaries:

* :func:`bridge_pannuke_confirmatory_inputs` converts checksum-validated PanNuke
  partitions into the stricter core rotation contract without exposing final
  reference outcomes to model runners;
* :func:`run_confirmatory_frozen_feature_oof` is the real grouped frozen-feature
  logistic OOF implementation; and
* :func:`execute_confirmatory_study` repeats the live execution gate, tracks the
  run, executes the matrix and production four-condition finalizer, independently
  reads every scientific artifact back before and after sealing, and only then
  returns ``CONFIRMATORY_COMPLETE``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from histo_audit.config import load_config
from histo_audit.corruption.controlled import (
    apply_controlled_corruption,
    array_artifact_sha256,
    canonical_sha256,
)
from histo_audit.cross_validation.oof import grouped_oof_logistic, make_group_stratified_folds
from histo_audit.evaluation.restoration import evaluate_downstream_restoration
from histo_audit.experiment.confirmatory_completion import (
    CONFIRMATORY_ARTIFACT_MANIFEST_FILENAME,
    REAL_CONFIRMATORY_ARTIFACT_SCOPE,
    RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
    ConfirmatoryFilesystemReadback,
    build_confirmatory_completion_evidence,
    confirmatory_report_contract_block,
    read_confirmatory_run_directory,
)
from histo_audit.experiment.confirmatory_core import (
    ConfirmatoryCellRequest,
    ConfirmatoryCorruptionInput,
    ConfirmatoryExecutionControls,
    ConfirmatoryFrozenBlocker,
    ConfirmatoryMatrixArtifacts,
    ConfirmatoryRotationInputs,
    FrozenFeatureOOFExecution,
    FrozenFeatureProvenance,
    _atomic_npz,
    atomic_write_json,
    atomic_write_text,
    confirmatory_execution_controls_from_frozen_config,
    execute_confirmatory_matrix,
)
from histo_audit.experiment.confirmatory_memory_workspace import RowIndexedArray
from histo_audit.experiment.confirmatory_statistics import (
    ConfirmatoryStatisticsArtifacts,
    ConfirmatoryStatisticsVerification,
    aggregate_confirmatory_statistics,
    verify_confirmatory_statistics_artifacts,
)
from histo_audit.experiment.m7_config_finalization import (
    derive_confirmatory_cnn_logical_provenance,
)
from histo_audit.experiment.original_confirmatory_preflight import (
    ORIGINAL_CONFIRMATORY_CAPACITY_POLICY_SHA256,
    ORIGINAL_CONFIRMATORY_FINAL_PREFLIGHT_FILENAME,
    ORIGINAL_CONFIRMATORY_INITIAL_PREFLIGHT_FILENAME,
    ORIGINAL_CONFIRMATORY_REQUIRED_FEATURE_CACHE_SCENARIOS,
    OriginalConfirmatoryCacheBinding,
    OriginalConfirmatoryPreflightError,
    OriginalConfirmatoryPreflightReceipt,
    recheck_original_confirmatory_capacity,
    require_original_confirmatory_preflight,
)
from histo_audit.experiment.original_confirmatory_resume import (
    build_original_confirmatory_resume_contract,
)
from histo_audit.experiment.original_confirmatory_runner_core import (
    ORIGINAL_CONFIRMATORY_CHECKPOINT_EXECUTION_CONTRACT_FILENAME,
    OriginalConfirmatoryCapsuleExecutionRequest,
    OriginalConfirmatoryScientificCheckpointAuthority,
    OriginalConfirmatorySuccessorPrecopyCheckpointProjection,
    _execute_original_confirmatory_prepared_matrix,
    _OriginalConfirmatoryPreparedMatrixRequest,
    build_original_confirmatory_canonical_e_checkpoint_projection,
    build_original_confirmatory_fresh_checkpoint_execution_contract,
    materialize_original_confirmatory_successor_checkpoint_execution,
    require_original_confirmatory_canonical_e_projection,
)
from histo_audit.experiment.pannuke_confirmatory_inputs import (
    ConfirmatoryFrozenFeatureAvailability,
    ConfirmatoryFrozenFeatureCacheSpec,
    ConfirmatoryObservedLabelSet,
    ConfirmatoryPartitionInputs,
    PanNukeConfirmatoryInputs,
    PanNukeConfirmatoryRotationInputs,
    load_pannuke_confirmatory_inputs,
)
from histo_audit.experiment.study_contracts import (
    CLASS_ORDER,
    RESOURCE_BOUNDED_CONFIRMATORY_PROFILE,
    ConfirmatoryMatrixPlan,
    build_confirmatory_matrix_plan,
)
from histo_audit.models.cnn import confirmatory_cnn_data_and_split_sha256
from histo_audit.representations.cache_provenance import (
    FrozenCacheVerification,
    confirmatory_cache_provenance_record,
    verify_frozen_cache_sidecar,
)
from histo_audit.utils.run_tracking import (
    IntegrityVerification,
    RunTracker,
    attest_run_stage_eligibility,
    guard_run_stage_eligibility,
    sha256_file,
    verify_run_integrity,
    withdraw_run_eligibility,
)
from histo_audit.workflows.preregistration_amendment import (
    require_confirmatory_storage_policy,
)
from histo_audit.workflows.study_gates import (
    ConfirmatoryExecutionGateEvidence,
    ResourceBoundedExecutionGateEvidence,
    validate_confirmatory_execution_gate,
)

_COMPLETION_STAGE = "CONFIRMATORY_COMPLETE"
_EXPERIMENT_NAME = "pannuke_confirmatory_study"
_HEX = frozenset("0123456789abcdef")
type RunnerArray = NDArray[Any] | RowIndexedArray
_PRIMARY_GATE_COMPLETION_HASHES = (
    "freeze_artifact_root_sha256",
    "freeze_manifest_sha256",
    "preregistration_sha256",
    "frozen_primary_config_sha256",
    "frozen_confirmatory_config_sha256",
    "primary_config_semantic_sha256",
    "confirmatory_config_semantic_sha256",
    "pilot_artifact_root_sha256",
    "dataset_sha256",
    "manifest_sha256",
    "duplicate_audit_sha256",
    "pathology_encoder_audit_sha256",
    "source_tree_root_sha256",
)
_CONFIRMATORY_GATE_COMPLETION_HASHES = (
    "primary_artifact_root_sha256",
    "primary_completion_evidence_sha256",
    "primary_reconciliation_sha256",
    "confirmatory_storage_policy_sha256",
)


class ConfirmatoryStudyRunnerError(RuntimeError):
    """The real confirmatory workflow failed without a valid stage claim."""


class ConfirmatoryStudyIntegrityError(ConfirmatoryStudyRunnerError):
    """A sealed confirmatory candidate failed independent integrity verification."""


def _require_legacy_confirmatory_execution_profile(
    config: Mapping[str, Any],
    plan: ConfirmatoryMatrixPlan,
) -> None:
    """Keep schema-v3 sensitivity plans out of the stage-eligible legacy runner."""

    if (
        config.get("schema_version") != 2
        or config.get("experiment_name") != "confirmatory_study"
        or config.get("execution_profile") == RESOURCE_BOUNDED_CONFIRMATORY_PROFILE
        or plan.schema_version != 2
    ):
        raise ConfirmatoryStudyRunnerError(
            "outcome-eligible confirmatory execution requires the original "
            "schema-v2 frozen execution profile"
        )


@dataclass(frozen=True, slots=True)
class ConfirmatoryBridgeResult:
    """Core rotations and immutable provenance created from PanNuke inputs."""

    rotations: tuple[ConfirmatoryRotationInputs, ...]
    frozen_blockers: Mapping[str, ConfirmatoryFrozenBlocker]
    sample_order_sha256: str
    partition_assignment_sha256: str
    corruption_assignment_sha256: str
    provenance_binding_sha256: str
    partition_content_sha256: str
    partition_bindings: Mapping[str, Mapping[str, Mapping[str, Any]]]
    final_reference_bindings: Mapping[str, Mapping[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "outer_folds": [rotation.outer_fold for rotation in self.rotations],
            "frozen_blockers": {
                key: asdict(value) for key, value in sorted(self.frozen_blockers.items())
            },
            "sample_order_sha256": self.sample_order_sha256,
            "partition_assignment_sha256": self.partition_assignment_sha256,
            "corruption_assignment_sha256": self.corruption_assignment_sha256,
            "provenance_binding_sha256": self.provenance_binding_sha256,
            "partition_content_sha256": self.partition_content_sha256,
            "partition_bindings": {
                fold: {role: dict(binding) for role, binding in sorted(partitions.items())}
                for fold, partitions in sorted(self.partition_bindings.items())
            },
            "final_reference_bindings": {
                key: dict(value) for key, value in sorted(self.final_reference_bindings.items())
            },
        }


class ConfirmatoryStageFinalizer(Protocol):
    """Complete restoration, figures, report, and the exact root manifest."""

    def __call__(
        self,
        *,
        run_directory: Path,
        matrix_artifacts: ConfirmatoryMatrixArtifacts,
        statistics_artifacts: ConfirmatoryStatisticsArtifacts,
        prepared_inputs: PanNukeConfirmatoryInputs,
        bridge: ConfirmatoryBridgeResult,
        controls: ConfirmatoryExecutionControls,
        gate_evidence: ConfirmatoryExecutionGateEvidence,
    ) -> None: ...


class ConfirmatoryRestorationVerifier(Protocol):
    """Replay and verify saved restoration evidence from frozen inputs."""

    def __call__(
        self,
        *,
        run_directory: Path,
        prepared_inputs: PanNukeConfirmatoryInputs,
        bridge: ConfirmatoryBridgeResult,
        controls: ConfirmatoryExecutionControls,
    ) -> None: ...


def _default_stage_finalizer(
    *,
    run_directory: Path,
    matrix_artifacts: ConfirmatoryMatrixArtifacts,
    statistics_artifacts: ConfirmatoryStatisticsArtifacts,
    prepared_inputs: PanNukeConfirmatoryInputs,
    bridge: ConfirmatoryBridgeResult,
    controls: ConfirmatoryExecutionControls,
    gate_evidence: ConfirmatoryExecutionGateEvidence,
) -> None:
    finalize_confirmatory_stage(
        run_directory=run_directory,
        matrix_artifacts=matrix_artifacts,
        statistics_artifacts=statistics_artifacts,
        prepared_inputs=prepared_inputs,
        bridge=bridge,
        controls=controls,
        gate_evidence=gate_evidence,
    )


def _default_restoration_verifier(
    *,
    run_directory: Path,
    prepared_inputs: PanNukeConfirmatoryInputs,
    bridge: ConfirmatoryBridgeResult,
    controls: ConfirmatoryExecutionControls,
) -> None:
    _validate_restoration_source_binding(
        run_directory,
        prepared_inputs=prepared_inputs,
        bridge=bridge,
        controls=controls,
    )


@dataclass(frozen=True, slots=True)
class ConfirmatoryRunnerDependencies:
    """Injectable boundaries for orchestration tests and production execution."""

    gate_validator: Callable[..., ConfirmatoryExecutionGateEvidence] = (
        validate_confirmatory_execution_gate
    )
    config_loader: Callable[[str | Path], dict[str, Any]] = load_config
    plan_builder: Callable[[Mapping[str, Any]], ConfirmatoryMatrixPlan] = (
        build_confirmatory_matrix_plan
    )
    controls_builder: Callable[[Mapping[str, Any]], ConfirmatoryExecutionControls] = (
        confirmatory_execution_controls_from_frozen_config
    )
    input_builder: Callable[..., PanNukeConfirmatoryInputs] = load_pannuke_confirmatory_inputs
    bridge_builder: Callable[..., ConfirmatoryBridgeResult] = lambda prepared, controls, **kwargs: (
        bridge_pannuke_confirmatory_inputs(prepared, controls, **kwargs)
    )
    matrix_executor: Callable[..., ConfirmatoryMatrixArtifacts] = execute_confirmatory_matrix
    statistics_aggregator: Callable[
        [str | Path, ConfirmatoryExecutionControls], ConfirmatoryStatisticsArtifacts
    ] = aggregate_confirmatory_statistics
    statistics_verifier: Callable[
        [str | Path, ConfirmatoryExecutionControls], ConfirmatoryStatisticsVerification
    ] = verify_confirmatory_statistics_artifacts
    stage_finalizer: ConfirmatoryStageFinalizer = _default_stage_finalizer
    restoration_verifier: ConfirmatoryRestorationVerifier = _default_restoration_verifier
    filesystem_reader: Callable[..., ConfirmatoryFilesystemReadback] = (
        read_confirmatory_run_directory
    )
    completion_builder: Callable[..., dict[str, Any]] = build_confirmatory_completion_evidence
    tracker_starter: Callable[..., RunTracker] = RunTracker.start
    integrity_verifier: Callable[[str | Path], IntegrityVerification] = verify_run_integrity
    eligibility_withdrawer: Callable[..., Mapping[str, Any]] = withdraw_run_eligibility


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value).issubset(_HEX)


def _require_sha(value: object, role: str) -> str:
    if not _valid_sha(value):
        raise ConfirmatoryStudyRunnerError(f"{role} must be a lowercase SHA-256")
    return str(value)


def _read_confirmatory_storage_policy_sha256(authority_directory: str | Path) -> str:
    policy = require_confirmatory_storage_policy(authority_directory)
    return _require_sha(canonical_sha256(policy), "confirmatory storage-policy SHA-256")


def _require_unchanged_confirmatory_storage_policy(
    authority_directory: str | Path,
    expected_sha256: str,
    *,
    phase: str,
) -> str:
    expected = _require_sha(expected_sha256, "expected confirmatory storage-policy SHA-256")
    observed = _read_confirmatory_storage_policy_sha256(authority_directory)
    if observed != expected:
        raise ConfirmatoryStudyIntegrityError(f"confirmatory storage policy changed during {phase}")
    return observed


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _readonly(values: NDArray[Any], dtype: Any) -> NDArray[Any]:
    output = np.asarray(values, dtype=dtype).copy()
    output.setflags(write=False)
    return output


def _readonly_or_indexed(values: RunnerArray, dtype: Any) -> RunnerArray:
    if isinstance(values, RowIndexedArray):
        if values.dtype != np.dtype(dtype):
            raise ConfirmatoryStudyRunnerError(
                "indexed confirmatory input exposes an unexpected logical dtype"
            )
        return values
    return _readonly(values, dtype)


def _select_rows(values: RunnerArray, indices: NDArray[np.int64]) -> RunnerArray:
    if isinstance(values, RowIndexedArray):
        return values.select_rows(indices)
    return np.asarray(values)[indices]


def _materialise_rows(values: RunnerArray, dtype: Any) -> NDArray[Any]:
    if isinstance(values, RowIndexedArray):
        gathered = values.gather_rows(max_rows=len(values))
        if gathered.dtype != np.dtype(dtype):
            raise ConfirmatoryStudyRunnerError(
                "indexed confirmatory input exposes an unexpected logical dtype"
            )
        return gathered
    return np.asarray(values, dtype=dtype)


def _controlled_array_sha256(values: RunnerArray) -> str:
    """Stream the legacy controlled-array digest for an indexed descriptor."""

    if not isinstance(values, RowIndexedArray):
        return array_artifact_sha256(values)
    header = json.dumps(
        {"dtype": values.dtype.str, "shape": list(values.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    for chunk in values.iter_chunks():
        digest.update(memoryview(np.ascontiguousarray(chunk)).cast("B"))
    return digest.hexdigest()


def _json_object(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfirmatoryStudyRunnerError(
            f"{role} is missing or invalid JSON: {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ConfirmatoryStudyRunnerError(f"{role} must be a JSON object")
    return cast(dict[str, Any], value)


def _partition_rows(rotation: PanNukeConfirmatoryRotationInputs) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for partition in (rotation.audit, rotation.reference_validation, rotation.final_reference):
        rows.extend(
            (int(index), sample_id)
            for index, sample_id in zip(partition.source_indices, partition.sample_ids, strict=True)
        )
    return rows


def _partition_content_binding(
    partition: ConfirmatoryPartitionInputs,
) -> dict[str, Any]:
    return {
        "role": partition.role,
        "sample_ids_sha256": canonical_sha256(list(partition.sample_ids)),
        "group_ids_sha256": canonical_sha256(list(partition.group_ids)),
        "source_indices_sha256": _controlled_array_sha256(partition.source_indices),
        "context_rgb_sha256": _controlled_array_sha256(partition.context_rgb),
        "target_masks_sha256": _controlled_array_sha256(partition.target_masks),
        "pre_corruption_labels_sha256": _controlled_array_sha256(partition.pre_corruption_labels),
        "observed_labels_sha256": _controlled_array_sha256(partition.observed_labels),
        "is_injected_corruption_sha256": _controlled_array_sha256(partition.is_injected_corruption),
        "corruption_types_sha256": canonical_sha256(list(partition.corruption_types)),
        "frozen_features_by_scenario": {
            feature.scenario_id: _controlled_array_sha256(feature.values)
            for feature in sorted(partition.frozen_features, key=lambda value: value.scenario_id)
        },
    }


def _prepared_partition_bindings(
    prepared: PanNukeConfirmatoryInputs,
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        str(rotation.outer_fold): {
            partition.role: _partition_content_binding(partition)
            for partition in (
                rotation.audit,
                rotation.reference_validation,
                rotation.final_reference,
            )
        }
        for rotation in prepared.rotations
    }


def _confirmatory_cnn_preflight_fingerprints(
    prepared: PanNukeConfirmatoryInputs,
    bridge: ConfirmatoryBridgeResult,
    controls: ConfirmatoryExecutionControls,
) -> dict[str, dict[str, dict[str, str]]]:
    """Freeze exact per-cell/fold CNN input hashes before any model executes."""

    source_by_fold = {value.outer_fold: value for value in prepared.rotations}
    core_by_fold = {value.outer_fold: value for value in bridge.rotations}
    fingerprints: dict[str, dict[str, dict[str, str]]] = {}
    fingerprints_by_inputs: dict[
        tuple[int, str, str],
        dict[str, dict[str, str]],
    ] = {}
    for cell in controls.plan.cells:
        scenario = controls.scenarios_by_id[cell.scenario_id]
        if scenario.family != "cnn":
            continue
        input_key = (
            cell.outer_fold,
            cell.corruption_cell_id,
            scenario.input_variant,
        )
        if input_key not in fingerprints_by_inputs:
            source = source_by_fold[cell.outer_fold]
            core = core_by_fold[cell.outer_fold]
            corruption = core.corruptions[cell.corruption_cell_id]
            folds = make_group_stratified_folds(
                source.audit.pre_corruption_labels,
                source.audit.group_ids,
                n_splits=controls.n_splits,
                class_order=CLASS_ORDER,
                seed=controls.split_seed,
            )
            uses_target_mask = scenario.input_variant == "context_rgb_plus_binary_target_mask"
            fingerprints_by_inputs[input_key] = {
                str(fold.fold_id): confirmatory_cnn_data_and_split_sha256(
                    _select_rows(source.audit.context_rgb, fold.train_indices),
                    corruption.observed_labels[fold.train_indices],
                    training_sample_ids=np.asarray(source.audit.sample_ids, dtype=np.str_)[
                        fold.train_indices
                    ],
                    training_group_ids=np.asarray(source.audit.group_ids, dtype=np.str_)[
                        fold.train_indices
                    ],
                    reference_validation_images=source.reference_validation.context_rgb,
                    reference_validation_labels=(source.reference_validation.pre_corruption_labels),
                    reference_validation_sample_ids=(source.reference_validation.sample_ids),
                    reference_validation_group_ids=(source.reference_validation.group_ids),
                    input_variant=cast(Any, scenario.input_variant),
                    training_target_masks=(
                        _select_rows(source.audit.target_masks, fold.train_indices)
                        if uses_target_mask
                        else None
                    ),
                    reference_validation_target_masks=(
                        source.reference_validation.target_masks if uses_target_mask else None
                    ),
                )
                for fold in folds
            }
        fingerprints[cell.cell_id] = {
            fold_id: dict(values) for fold_id, values in fingerprints_by_inputs[input_key].items()
        }
    return fingerprints


def _global_sample_order(
    rotations: Sequence[PanNukeConfirmatoryRotationInputs],
) -> tuple[str, ...]:
    if not rotations:
        raise ValueError("PanNuke confirmatory inputs contain no rotations")
    expected: tuple[str, ...] | None = None
    for rotation in rotations:
        rows = sorted(_partition_rows(rotation))
        indices = tuple(index for index, _ in rows)
        if indices != tuple(range(len(rows))):
            raise ValueError("PanNuke source indices are not an exact cache-order permutation")
        current = tuple(sample_id for _, sample_id in rows)
        if expected is None:
            expected = current
        elif current != expected:
            raise ValueError("PanNuke rotations disagree on the global sample order")
    if expected is None:
        raise RuntimeError("global PanNuke sample order unexpectedly unavailable")
    return expected


def _sample_fold_map(
    rotations: Sequence[PanNukeConfirmatoryRotationInputs],
    global_order: Sequence[str],
) -> dict[str, int]:
    sample_fold: dict[str, int] = {}
    for rotation in rotations:
        for sample_id in rotation.final_reference.sample_ids:
            if sample_id in sample_fold:
                raise ValueError("a sample is final reference in more than one rotation")
            sample_fold[sample_id] = rotation.outer_fold
    if set(sample_fold) != set(global_order):
        raise ValueError("final-reference rotations do not partition the full sample universe")
    return sample_fold


def _availability_by_scenario(
    prepared: PanNukeConfirmatoryInputs,
) -> dict[str, ConfirmatoryFrozenFeatureAvailability]:
    values = {item.scenario_id: item for item in prepared.frozen_feature_availability}
    if len(values) != len(prepared.frozen_feature_availability):
        raise ValueError("PanNuke feature availability has duplicate scenario IDs")
    return values


def _verify_runtime_provenance_records(
    prepared: PanNukeConfirmatoryInputs,
    controls: ConfirmatoryExecutionControls,
    availability: Mapping[str, ConfirmatoryFrozenFeatureAvailability],
    global_order: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    """Rebuild available provenance from cache bytes/recipes, never config claims."""

    try:
        crop = verify_frozen_cache_sidecar(
            prepared.crop_cache_path,
            expected_manifest_sha256=prepared.manifest_sha256,
            expected_representation_id="pannuke_component_covering_target_crops",
        )
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        raise ValueError(
            "confirmatory crop cache failed semantic provenance verification"
        ) from error
    if (
        crop.cache_file_sha256 != prepared.crop_cache_sha256
        or crop.sidecar_file_sha256 != prepared.crop_metadata_sha256
        or crop.sidecar_path != Path(prepared.crop_metadata_path).resolve()
        or crop.metadata.get("sample_order_sha256")
        != canonical_sha256([str(value) for value in global_order])
        or crop.metadata.get("raw_inventory_sha256") != prepared.raw_inventory_sha256
    ):
        raise ValueError("confirmatory crop provenance differs from loaded PanNuke inputs")

    output: dict[str, Mapping[str, Any]] = {}
    verified_frozen: dict[str, FrozenCacheVerification] = {}
    for scenario in controls.scenario_specs:
        record = controls.cache_provenance_by_id[scenario.cache_provenance_id]
        if str(record.get("status")) != "available" or scenario.family == "cnn":
            continue
        loaded = availability.get(scenario.scenario_id)
        if (
            loaded is None
            or not loaded.available
            or loaded.cache_path is None
            or loaded.metadata_path is None
            or loaded.cache_sha256 is None
            or loaded.metadata_sha256 is None
            or loaded.weight_sha256 is None
        ):
            raise ValueError(
                f"available frozen scenario lacks runtime provenance: {scenario.scenario_id}"
            )
        try:
            verification = verify_frozen_cache_sidecar(
                loaded.cache_path,
                expected_manifest_sha256=prepared.manifest_sha256,
                expected_representation_id=scenario.representation_id,
            )
        except (FileNotFoundError, OSError, ValueError, KeyError) as error:
            raise ValueError(
                f"frozen scenario failed semantic provenance verification: {scenario.scenario_id}"
            ) from error
        if (
            verification.cache_file_sha256 != loaded.cache_sha256
            or verification.sidecar_file_sha256 != loaded.metadata_sha256
            or verification.sidecar_path != Path(loaded.metadata_path).resolve()
            or verification.metadata.get("weights_sha256") != loaded.weight_sha256
        ):
            raise ValueError(
                f"frozen scenario provenance differs from loaded cache: {scenario.scenario_id}"
            )
        projected = confirmatory_cache_provenance_record(
            verification.metadata,
            record_id=scenario.cache_provenance_id,
            bind_sidecar_semantics=True,
        )
        if dict(record) != projected:
            raise ValueError(
                "frozen config encoder/preprocessing provenance does not recompute from "
                f"cache sidecar: {scenario.scenario_id}"
            )
        verified_frozen[scenario.scenario_id] = verification
        output[scenario.cache_provenance_id] = MappingProxyType(projected)

    imagenet_weights = {
        (
            str(value.metadata.get("weight_identifier")),
            str(value.metadata.get("weights_sha256")),
        )
        for scenario_id, value in verified_frozen.items()
        if controls.scenarios_by_id[scenario_id].family == "imagenet_frozen"
    }
    if len(imagenet_weights) != 1:
        raise ValueError("available ImageNet frozen caches do not prove one CNN weight identity")
    weight_identifier, weights_sha256 = next(iter(imagenet_weights))
    derived_cnn = derive_confirmatory_cnn_logical_provenance(
        crop,
        weight_identifier=weight_identifier,
        weights_sha256=weights_sha256,
        input_size=controls.input_size,
    )
    for scenario in controls.scenario_specs:
        if scenario.family != "cnn":
            continue
        record = controls.cache_provenance_by_id[scenario.cache_provenance_id]
        derived = derived_cnn.get(scenario.cache_provenance_id)
        if derived is None or dict(record) != derived:
            raise ValueError(
                "CNN logical encoder/preprocessing provenance does not recompute from "
                f"the verified crop view: {scenario.scenario_id}"
            )
        output[scenario.cache_provenance_id] = MappingProxyType(derived)
    return output


def _actual_cache_binding(
    prepared: PanNukeConfirmatoryInputs,
    scenario_family: str,
    scenario_id: str,
    availability: Mapping[str, ConfirmatoryFrozenFeatureAvailability],
) -> tuple[str, Path, str | None]:
    if scenario_family == "cnn":
        return (
            prepared.crop_cache_sha256,
            Path(prepared.crop_cache_path),
            None,
        )
    record = availability.get(scenario_id)
    if record is None or not record.available or record.cache_sha256 is None:
        raise ValueError(f"available frozen scenario lacks a loaded cache: {scenario_id}")
    if record.metadata_path is None or record.cache_path is None:
        raise ValueError(f"available frozen scenario lacks cache metadata: {scenario_id}")
    return record.cache_sha256, Path(record.cache_path), record.weight_sha256


def _provenance_for_rotation(
    prepared: PanNukeConfirmatoryInputs,
    rotation: PanNukeConfirmatoryRotationInputs,
    controls: ConfirmatoryExecutionControls,
    availability: Mapping[str, ConfirmatoryFrozenFeatureAvailability],
    global_order: Sequence[str],
    runtime_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, FrozenFeatureProvenance]:
    output: dict[str, FrozenFeatureProvenance] = {}
    global_order_sha = canonical_sha256([str(value) for value in global_order])
    audit_order_sha = canonical_sha256([str(value) for value in rotation.audit.sample_ids])
    for scenario in controls.scenario_specs:
        frozen_record = controls.cache_provenance_by_id[scenario.cache_provenance_id]
        status = str(frozen_record.get("status"))
        if status != "available":
            continue
        record = runtime_records.get(scenario.cache_provenance_id)
        if record is None or dict(record) != dict(frozen_record):
            raise ValueError(f"runtime provenance is absent or differs for {scenario.scenario_id}")
        if (
            record.get("representation_id") != scenario.representation_id
            or record.get("manifest_sha256") != prepared.manifest_sha256
            or record.get("sample_order_sha256") != global_order_sha
            or record.get("encoder_identifier") != scenario.encoder
            or record.get("input_variant") != scenario.input_variant
        ):
            raise ValueError(
                f"frozen cache provenance is misbound for scenario {scenario.scenario_id}"
            )
        cache_sha, _cache_path, actual_weight = _actual_cache_binding(
            prepared,
            scenario.family,
            scenario.scenario_id,
            availability,
        )
        cache_record_sha = record.get("cache_file_sha256")
        sidecar_semantic_sha = record.get("sidecar_semantic_sha256")
        if (cache_record_sha is None) == (sidecar_semantic_sha is None):
            raise ValueError("frozen provenance requires exactly one cache or sidecar SHA")
        if cache_record_sha is not None and cache_record_sha != cache_sha:
            raise ValueError(f"cache bytes differ from frozen provenance: {scenario.scenario_id}")
        if actual_weight is not None and record.get("weights_sha256") != actual_weight:
            raise ValueError(f"encoder weights differ for scenario {scenario.scenario_id}")
        provenance = FrozenFeatureProvenance(
            cache_provenance_id=scenario.cache_provenance_id,
            representation_id=scenario.representation_id,
            cache_file_sha256=(str(cache_record_sha) if cache_record_sha is not None else None),
            sidecar_semantic_sha256=(
                str(sidecar_semantic_sha) if sidecar_semantic_sha is not None else None
            ),
            sample_order_sha256=str(record["sample_order_sha256"]),
            manifest_sha256=str(record["manifest_sha256"]),
            encoder_identifier=str(record["encoder_identifier"]),
            encoder_metadata_sha256=str(record["encoder_metadata_sha256"]),
            weight_identifier=str(record["weight_identifier"]),
            weights_sha256=str(record["weights_sha256"]),
            preprocessing_identifier=str(record["preprocessing_identifier"]),
            preprocessing_sha256=str(record["preprocessing_sha256"]),
            input_variant=str(record["input_variant"]),
            audit_sample_order_sha256=audit_order_sha,
        )
        provenance.validate(
            representation_id=scenario.representation_id,
            audit_sample_ids=rotation.audit.sample_ids,
        )
        output[scenario.representation_id] = provenance
    return output


def _frozen_features_for_rotation(
    rotation: PanNukeConfirmatoryRotationInputs,
    controls: ConfirmatoryExecutionControls,
) -> dict[str, NDArray[np.float64] | RowIndexedArray]:
    scenario_features = {item.scenario_id: item.values for item in rotation.audit.frozen_features}
    output: dict[str, NDArray[np.float64] | RowIndexedArray] = {}
    for scenario in controls.scenario_specs:
        if scenario.family == "cnn" or scenario.scenario_id not in scenario_features:
            continue
        output[scenario.representation_id] = cast(
            NDArray[np.float64] | RowIndexedArray,
            _readonly_or_indexed(scenario_features[scenario.scenario_id], np.float64),
        )
    return output


def _corruptions_for_rotation(
    rotation: PanNukeConfirmatoryRotationInputs,
    controls: ConfirmatoryExecutionControls,
) -> dict[str, ConfirmatoryCorruptionInput]:
    nonzero = [spec for spec in controls.corruption_specs if spec.rate > 0.0]
    if len(nonzero) > 1:
        raise ValueError(
            "the PanNuke adapter carries one observed-label assignment per rotation; "
            "more than one nonzero frozen corruption cell requires an explicit multi-cell "
            "materializer"
        )
    output: dict[str, ConfirmatoryCorruptionInput] = {}
    audit = rotation.audit
    for spec in controls.corruption_specs:
        if spec.rate == 0.0:
            observed = audit.pre_corruption_labels
            injected = np.zeros(len(audit.sample_ids), dtype=bool)
        else:
            observed = audit.observed_labels
            injected = audit.is_injected_corruption
            if any(
                corruption_type not in {"none", spec.mechanism}
                for corruption_type in audit.corruption_types
            ):
                raise ValueError(
                    f"rotation {rotation.outer_fold} corruption metadata differs from "
                    f"{spec.corruption_cell_id}"
                )
            replay_kwargs: dict[str, Any] = {}
            if spec.mechanism == "confusion_targeted_corruption":
                replay_kwargs["transition_matrix"] = np.asarray(
                    spec.parameters["transition_matrix"], dtype=np.float64
                )
            elif spec.mechanism == "group_conditional_corruption":
                grouping_field = str(spec.parameters["grouping_field"])
                if grouping_field != controls.statistical_group_unit:
                    raise ValueError(
                        "group-conditional confirmatory corruption cannot be replayed: "
                        "the PanNuke bridge lacks the frozen grouping field"
                    )
                raw_weights = spec.parameters["weights_by_value"]
                if not isinstance(raw_weights, Mapping):
                    raise ValueError("group-conditional frozen weights are malformed")
                default_weight = float(spec.parameters["default_weight"])
                replay_kwargs["group_weights"] = {
                    group_id: float(raw_weights.get(group_id, default_weight))
                    for group_id in set(audit.group_ids)
                }
            elif spec.mechanism == "instance_dependent_corruption":
                raise ValueError(
                    "instance-dependent confirmatory assignment requires a dedicated "
                    "independence-bound multi-feature materializer before execution"
                )
            elif spec.mechanism != "symmetric_random_corruption":
                raise ValueError(f"unsupported confirmatory corruption: {spec.mechanism}")
            replay = apply_controlled_corruption(
                audit.pre_corruption_labels,
                sample_ids=audit.sample_ids,
                group_ids=audit.group_ids,
                rate=spec.rate,
                mechanism=spec.mechanism,
                seed=spec.seed,
                n_classes=len(CLASS_ORDER),
                **replay_kwargs,
            )
            if not np.array_equal(observed, replay.observed_labels) or not np.array_equal(
                injected, replay.is_injected_corruption
            ):
                raise ValueError(
                    f"rotation {rotation.outer_fold} observed labels do not replay from "
                    f"frozen mechanism/rate/seed for {spec.corruption_cell_id}"
                )
        item = ConfirmatoryCorruptionInput(
            corruption_cell_id=spec.corruption_cell_id,
            mechanism=spec.mechanism,
            rate=spec.rate,
            seed=spec.seed,
            parameters=MappingProxyType(dict(spec.parameters)),
            pre_corruption_labels=cast(
                NDArray[np.int64], _readonly(audit.pre_corruption_labels, np.int64)
            ),
            observed_labels=cast(NDArray[np.int64], _readonly(observed, np.int64)),
            is_injected_corruption=cast(NDArray[np.bool_], _readonly(injected, np.bool_)),
        )
        item.validate(spec, len(audit.sample_ids))
        output[spec.corruption_cell_id] = item
    return output


def _official_fold_vector(
    sample_ids: Sequence[str], sample_fold: Mapping[str, int]
) -> NDArray[np.int64]:
    return cast(
        NDArray[np.int64],
        _readonly(np.asarray([sample_fold[value] for value in sample_ids]), np.int64),
    )


def bridge_pannuke_confirmatory_inputs(
    prepared: PanNukeConfirmatoryInputs,
    controls: ConfirmatoryExecutionControls,
    *,
    pathology_encoder_audit_sha256: str,
) -> ConfirmatoryBridgeResult:
    """Convert strict PanNuke partitions into all core fold rotations.

    The current PanNuke adapter intentionally carries one nonzero observed-label
    assignment per official-fold rotation.  A frozen plan with multiple nonzero
    corruption cells is rejected rather than reusing one assignment silently.
    """

    controls.validate_for_plan(controls.plan)
    if prepared.config_sha256 != controls.config_semantic_sha256:
        raise ValueError("PanNuke inputs differ from frozen confirmatory controls")
    if prepared.execution_mode != "real_study" or not prepared.study_outcome_eligible:
        raise ValueError("production confirmatory bridge requires eligible real-study inputs")
    prepared.validate(official_folds=controls.official_folds, oof_splits=controls.n_splits)
    partition_bindings = _prepared_partition_bindings(prepared)
    partition_content_sha256 = canonical_sha256(partition_bindings)
    global_order = _global_sample_order(prepared.rotations)
    sample_fold = _sample_fold_map(prepared.rotations, global_order)
    availability = _availability_by_scenario(prepared)
    runtime_records = _verify_runtime_provenance_records(
        prepared,
        controls,
        availability,
        global_order,
    )

    blockers: dict[str, ConfirmatoryFrozenBlocker] = {}
    for scenario in controls.scenario_specs:
        cache_record = controls.cache_provenance_by_id[scenario.cache_provenance_id]
        status = str(cache_record.get("status"))
        if status == "available":
            continue
        if scenario.required or status != "unavailable_with_frozen_blocker":
            raise ValueError(f"invalid unavailable scenario state: {scenario.scenario_id}")
        audit_sha = _require_sha(scenario.availability_audit_sha256, "availability audit")
        if audit_sha != pathology_encoder_audit_sha256 or (
            cache_record.get("blocker_evidence_sha256") != audit_sha
        ):
            raise ValueError("optional pathology blocker differs from gated audit evidence")
        availability_record = availability.get(scenario.scenario_id)
        blocker_text = (
            availability_record.blocker
            if availability_record is not None and availability_record.blocker
            else "optional frozen-feature cache unavailable under the frozen audit"
        )
        blockers[scenario.scenario_id] = ConfirmatoryFrozenBlocker(
            scenario_id=scenario.scenario_id,
            config_semantic_sha256=controls.config_semantic_sha256,
            availability_audit_sha256=audit_sha,
            blocker=blocker_text,
        )

    rotations: list[ConfirmatoryRotationInputs] = []
    corruption_payload: list[dict[str, Any]] = []
    partition_payload: list[dict[str, Any]] = []
    provenance_payload: list[dict[str, Any]] = []
    final_reference_bindings: dict[str, dict[str, Any]] = {}
    for source in prepared.rotations:
        corruptions = _corruptions_for_rotation(source, controls)
        features = _frozen_features_for_rotation(source, controls)
        provenance = _provenance_for_rotation(
            prepared,
            source,
            controls,
            availability,
            global_order,
            runtime_records,
        )
        core = ConfirmatoryRotationInputs(
            outer_fold=source.outer_fold,
            audit_sample_ids=source.audit.sample_ids,
            audit_group_ids=source.audit.group_ids,
            audit_official_folds=_official_fold_vector(source.audit.sample_ids, sample_fold),
            audit_rgb=source.audit.context_rgb,
            audit_target_masks=source.audit.target_masks,
            corruptions=MappingProxyType(corruptions),
            frozen_audit_features=MappingProxyType(features),
            frozen_feature_provenance=MappingProxyType(provenance),
            reference_validation_sample_ids=source.reference_validation.sample_ids,
            reference_validation_group_ids=source.reference_validation.group_ids,
            reference_validation_official_folds=_official_fold_vector(
                source.reference_validation.sample_ids, sample_fold
            ),
            reference_validation_labels=source.reference_validation.pre_corruption_labels,
            reference_validation_rgb=source.reference_validation.context_rgb,
            reference_validation_target_masks=source.reference_validation.target_masks,
            final_sample_ids=source.final_reference.sample_ids,
            final_group_ids=source.final_reference.group_ids,
            final_official_folds=_official_fold_vector(
                source.final_reference.sample_ids, sample_fold
            ),
            final_pre_corruption_labels=source.final_reference.pre_corruption_labels,
            final_observed_labels=source.final_reference.observed_labels,
            final_is_injected_corruption=source.final_reference.is_injected_corruption,
        )
        core.validate(controls)
        rotations.append(core)
        partition_payload.append(
            {
                "outer_fold": core.outer_fold,
                "audit_sample_ids": list(core.audit_sample_ids),
                "reference_validation_sample_ids": list(core.reference_validation_sample_ids),
                "final_sample_ids": list(core.final_sample_ids),
            }
        )
        corruption_payload.append(
            {
                "outer_fold": core.outer_fold,
                "cells": {
                    key: {
                        "pre_corruption_labels_sha256": array_artifact_sha256(
                            value.pre_corruption_labels
                        ),
                        "observed_labels_sha256": array_artifact_sha256(value.observed_labels),
                        "is_injected_corruption_sha256": array_artifact_sha256(
                            value.is_injected_corruption
                        ),
                    }
                    for key, value in sorted(core.corruptions.items())
                },
            }
        )
        provenance_payload.append(
            {
                "outer_fold": core.outer_fold,
                "representations": {
                    key: value.semantic_sha256
                    for key, value in sorted(core.frozen_feature_provenance.items())
                },
            }
        )
        final_reference_bindings[str(core.outer_fold)] = {
            "sample_ids_sha256": canonical_sha256(list(core.final_sample_ids)),
            "group_ids_sha256": canonical_sha256(list(core.final_group_ids)),
            "pre_corruption_labels_sha256": array_artifact_sha256(core.final_pre_corruption_labels),
            "observed_labels_sha256": array_artifact_sha256(core.final_observed_labels),
            "is_injected_corruption_sha256": array_artifact_sha256(
                core.final_is_injected_corruption
            ),
            "source_indices_sha256": array_artifact_sha256(source.final_reference.source_indices),
            "context_rgb_sha256": _controlled_array_sha256(source.final_reference.context_rgb),
            "target_masks_sha256": _controlled_array_sha256(source.final_reference.target_masks),
            "frozen_features_by_scenario": {
                feature.scenario_id: _controlled_array_sha256(feature.values)
                for feature in source.final_reference.frozen_features
            },
        }
    if tuple(item.outer_fold for item in rotations) != controls.official_folds:
        raise ValueError("bridged rotations differ from frozen official-fold order")
    return ConfirmatoryBridgeResult(
        rotations=tuple(rotations),
        frozen_blockers=MappingProxyType(blockers),
        sample_order_sha256=canonical_sha256(list(global_order)),
        partition_assignment_sha256=canonical_sha256(partition_payload),
        corruption_assignment_sha256=canonical_sha256(corruption_payload),
        provenance_binding_sha256=canonical_sha256(provenance_payload),
        partition_content_sha256=partition_content_sha256,
        partition_bindings=MappingProxyType(
            {
                fold: MappingProxyType(
                    {role: MappingProxyType(dict(binding)) for role, binding in partitions.items()}
                )
                for fold, partitions in partition_bindings.items()
            }
        ),
        final_reference_bindings=MappingProxyType(
            {key: MappingProxyType(value) for key, value in final_reference_bindings.items()}
        ),
    )


def _logistic_parameters(controls: ConfirmatoryExecutionControls) -> tuple[float, int]:
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
    """Run the real grouped logistic OOF path on a frozen feature cache."""

    if request.scenario.family not in {"imagenet_frozen", "pathology_frozen"}:
        raise ValueError("frozen-feature runner received a non-frozen scenario")
    if request.scenario.classifier != "multinomial_logistic_regression":
        raise ValueError("unsupported confirmatory frozen-feature classifier")
    representation = request.scenario.representation_id
    source_features = request.inputs.frozen_audit_features.get(representation)
    provenance = request.inputs.frozen_feature_provenance.get(representation)
    if source_features is None or provenance is None:
        raise ValueError("frozen-feature matrix/provenance is unavailable")
    features = cast(
        NDArray[np.float64],
        _materialise_rows(source_features, np.float64),
    )
    l2, max_iter = _logistic_parameters(request.controls)
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
    return FrozenFeatureOOFExecution(
        oof_result=oof,
        execution_mode=("cpu_test_only" if request.cpu_test_only else "real_study_cpu"),
        study_outcome_eligible=not request.cpu_test_only,
        configuration_sha256=canonical_sha256(configuration),
        evidence={
            **configuration,
            "configuration_sha256": canonical_sha256(configuration),
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


def _verify_cache_files(
    crop_cache: Path,
    *,
    expected_crop_cache_sha256: str,
    expected_crop_metadata_sha256: str,
    frozen_feature_caches: Sequence[ConfirmatoryFrozenFeatureCacheSpec],
) -> None:
    metadata = crop_cache.with_suffix(f"{crop_cache.suffix}.metadata.json")
    if not crop_cache.is_file() or not metadata.is_file():
        raise FileNotFoundError("confirmatory crop cache and sidecar must exist")
    if sha256_file(crop_cache) != expected_crop_cache_sha256:
        raise ConfirmatoryStudyRunnerError("crop cache differs from explicit SHA-256")
    if sha256_file(metadata) != expected_crop_metadata_sha256:
        raise ConfirmatoryStudyRunnerError("crop metadata differs from explicit SHA-256")
    seen: set[str] = set()
    for spec in frozen_feature_caches:
        if spec.scenario_id in seen:
            raise ValueError("duplicate frozen-feature cache scenario")
        seen.add(spec.scenario_id)
        path = Path(spec.cache_path).resolve()
        sidecar = path.with_suffix(f"{path.suffix}.metadata.json")
        if not path.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"frozen-feature cache/sidecar missing: {path}")
        if sha256_file(path) != spec.expected_cache_sha256:
            raise ConfirmatoryStudyRunnerError(
                f"feature cache differs from SHA-256: {spec.scenario_id}"
            )
        if sha256_file(sidecar) != spec.expected_metadata_sha256:
            raise ConfirmatoryStudyRunnerError(
                f"feature metadata differs from SHA-256: {spec.scenario_id}"
            )


def _original_confirmatory_cache_bindings(
    crop_cache: Path,
    *,
    expected_crop_cache_sha256: str,
    expected_crop_metadata_sha256: str,
    frozen_feature_caches: Sequence[ConfirmatoryFrozenFeatureCacheSpec],
) -> tuple[OriginalConfirmatoryCacheBinding, ...]:
    """Project every original-confirmatory cache into the typed preflight surface."""

    scenario_ids = tuple(value.scenario_id for value in frozen_feature_caches)
    if (
        len(set(scenario_ids)) != len(scenario_ids)
        or tuple(sorted(scenario_ids)) != ORIGINAL_CONFIRMATORY_REQUIRED_FEATURE_CACHE_SCENARIOS
    ):
        raise ConfirmatoryStudyRunnerError(
            "original confirmatory preflight requires exactly the three frozen "
            "ImageNet context/highlight/context+morphometrics cache scenarios"
        )
    bindings = [
        OriginalConfirmatoryCacheBinding(
            role="crop_cache",
            path=crop_cache,
            expected_sha256=expected_crop_cache_sha256,
        ),
        OriginalConfirmatoryCacheBinding(
            role="crop_cache_metadata",
            path=crop_cache.with_suffix(f"{crop_cache.suffix}.metadata.json"),
            expected_sha256=expected_crop_metadata_sha256,
        ),
    ]
    for spec in sorted(frozen_feature_caches, key=lambda value: value.scenario_id):
        path = Path(spec.cache_path)
        bindings.extend(
            (
                OriginalConfirmatoryCacheBinding(
                    role=f"frozen_feature_cache:{spec.scenario_id}",
                    path=path,
                    expected_sha256=spec.expected_cache_sha256,
                ),
                OriginalConfirmatoryCacheBinding(
                    role=f"frozen_feature_metadata:{spec.scenario_id}",
                    path=path.with_suffix(f"{path.suffix}.metadata.json"),
                    expected_sha256=spec.expected_metadata_sha256,
                ),
            )
        )
    return tuple(bindings)


def _validate_gate_equality(
    supplied: ConfirmatoryExecutionGateEvidence,
    live: ConfirmatoryExecutionGateEvidence,
) -> None:
    if not isinstance(supplied, ConfirmatoryExecutionGateEvidence):
        raise TypeError("gate_evidence must be typed ConfirmatoryExecutionGateEvidence")
    if supplied != live:
        raise ConfirmatoryStudyRunnerError(
            "supplied confirmatory gate differs from mandatory live revalidation"
        )


def _guarded_final_gate_and_start(
    *,
    primary_run_directory: str | Path,
    expected_gate: ConfirmatoryExecutionGateEvidence,
    gate_validator: Callable[..., ConfirmatoryExecutionGateEvidence],
    gate_kwargs: Mapping[str, Any],
    cache_recheck: Callable[[], Mapping[str, Any] | None],
    tracker_starter: Callable[..., RunTracker],
    tracker_kwargs: Mapping[str, Any],
) -> tuple[ConfirmatoryExecutionGateEvidence, RunTracker]:
    """Totally order primary withdrawal against final-gate authority and run start."""

    if "primary_stage_eligibility_receipt" in gate_kwargs:
        raise ConfirmatoryStudyRunnerError(
            "guarded final-gate arguments must not supply their own eligibility receipt"
        )
    supplied_primary = gate_kwargs.get("primary_run_directory")
    if (
        supplied_primary is None
        or Path(supplied_primary).resolve() != Path(primary_run_directory).resolve()
    ):
        raise ConfirmatoryStudyRunnerError(
            "guarded final-gate arguments target a different primary run"
        )

    with guard_run_stage_eligibility(primary_run_directory) as guarded_receipt:
        if guarded_receipt is None:
            raise ConfirmatoryStudyRunnerError(
                "primary run lacks active stage authority before dependent run start"
            )
        final_gate = gate_validator(
            **dict(gate_kwargs),
            primary_stage_eligibility_receipt=guarded_receipt,
        )
        _validate_gate_equality(expected_gate, final_gate)
        environment_updates = cache_recheck()
        start_kwargs = dict(tracker_kwargs)
        if environment_updates is not None:
            if not isinstance(environment_updates, Mapping):
                raise ConfirmatoryStudyRunnerError(
                    "guarded final recheck returned non-mapping tracker evidence"
                )
            environment = start_kwargs.get("environment")
            if not isinstance(environment, Mapping):
                raise ConfirmatoryStudyRunnerError(
                    "guarded tracker start lacks a mapping environment"
                )
            overlap = set(environment).intersection(environment_updates)
            if overlap:
                raise ConfirmatoryStudyRunnerError(
                    f"guarded final recheck overwrites tracker environment fields: {sorted(overlap)}"
                )
            start_kwargs["environment"] = {
                **dict(environment),
                **dict(environment_updates),
            }
        tracker = tracker_starter(**start_kwargs)
    return final_gate, tracker


def _validate_core_artifacts(
    artifacts: ConfirmatoryMatrixArtifacts,
    *,
    run_directory: Path,
    plan: ConfirmatoryMatrixPlan,
    expected_artifact_scope: str = REAL_CONFIRMATORY_ARTIFACT_SCOPE,
    expected_resource_gate: ResourceBoundedExecutionGateEvidence | None = None,
) -> dict[str, Any]:
    if Path(artifacts.output_directory).resolve() != run_directory:
        raise ConfirmatoryStudyRunnerError("matrix executor redirected its output")
    required = (
        artifacts.matrix_plan_path,
        artifacts.execution_controls_path,
        artifacts.frozen_feature_provenance_path,
        artifacts.original_audit_selection_path,
        artifacts.cell_index_path,
        artifacts.ensemble_evidence_path,
        artifacts.hybrid_ablations_path,
        artifacts.fold_aggregate_path,
        artifacts.reconciliation_path,
        artifacts.completion_evidence_path,
        artifacts.analysis_gaps_path,
        artifacts.figure_manifest_path,
        artifacts.report_path,
        artifacts.artifact_manifest_path,
    )
    if any(not Path(path).is_file() for path in required):
        raise ConfirmatoryStudyRunnerError("matrix core omitted a required draft artifact")
    completion = _json_object(
        Path(artifacts.completion_evidence_path), "confirmatory core completion draft"
    )
    if (
        completion.get("completion_stage") is not None
        or completion.get("study_outcome_eligible") is not False
        or completion.get("artifact_scope") != expected_artifact_scope
        or completion.get("matrix_config_sha256") != plan.config_sha256
        or artifacts.study_outcome_eligible
    ):
        raise ConfirmatoryStudyRunnerError(
            "matrix core must remain an ineligible completion draft in the requested scope"
        )
    gaps = _json_object(
        Path(artifacts.analysis_gaps_path),
        "confirmatory core technical execution evidence",
    )
    if expected_artifact_scope == RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE:
        if not isinstance(
            expected_resource_gate,
            ResourceBoundedExecutionGateEvidence,
        ):
            raise ConfirmatoryStudyRunnerError(
                "resource matrix draft requires typed P+C gate readback"
            )
        disposition = {
            "outcomes_inspected": True,
            "analysis_disposition": "amended_or_exploratory",
            "original_confirmatory_claim_allowed": False,
            "study_outcome_eligible": False,
            "completion_stage": None,
            "m9_unlock_allowed": False,
        }
        authority = expected_resource_gate.execution_authority
        historical = expected_resource_gate.historical_primary
        if (
            completion.get("resource_gate_sha256")
            != canonical_sha256(expected_resource_gate.as_dict())
            or completion.get("historical_primary_dependency_sha256")
            != canonical_sha256(historical.as_dict())
            or completion.get("resource_execution_authority_sha256")
            != canonical_sha256(authority.as_dict())
            or completion.get("resource_capacity_policy_sha256")
            != canonical_sha256(dict(authority.resource_capacity_policy))
            or completion.get("resource_disposition_binding_sha256")
            != canonical_sha256(disposition)
            or completion.get("analysis_disposition") != "amended_or_exploratory"
            or completion.get("outcomes_inspected") is not True
            or completion.get("original_confirmatory_claim_allowed") is not False
            or completion.get("m9_unlock_allowed") is not False
            or completion.get("model_matrix_execution_eligible") is not True
            or gaps.get("model_matrix_execution_eligible") is not True
            or completion.get("matrix_execution_telemetry_sha256")
            != gaps.get("matrix_execution_telemetry_sha256")
            or not _valid_sha(completion.get("matrix_execution_telemetry_sha256"))
        ):
            raise ConfirmatoryStudyRunnerError(
                "resource matrix draft lacks exact P+C/disposition/technical bindings"
            )
    elif expected_resource_gate is not None:
        raise ConfirmatoryStudyRunnerError(
            "resource P+C gate cannot be supplied to a non-resource matrix scope"
        )
    if not artifacts.reconciliation.passed:
        raise ConfirmatoryStudyRunnerError(
            f"confirmatory model matrix reconciliation failed: {artifacts.reconciliation.errors}"
        )
    return completion


def _validate_final_readback(
    readback: ConfirmatoryFilesystemReadback,
    artifacts: ConfirmatoryMatrixArtifacts,
    run_directory: Path,
    *,
    expected_confirmatory_storage_policy_sha256: str,
) -> None:
    if (
        not readback.passed
        or Path(readback.run_directory).resolve() != run_directory
        or readback.reconciliation is None
        or readback.reconciliation.as_dict() != artifacts.reconciliation.as_dict()
        or readback.confirmatory_storage_policy_sha256
        != expected_confirmatory_storage_policy_sha256
    ):
        raise ConfirmatoryStudyRunnerError(
            f"confirmatory filesystem readback failed: {readback.errors}"
        )


def _partition_feature(
    partition: ConfirmatoryPartitionInputs,
    scenario_id: str,
) -> NDArray[np.float64]:
    matches = [
        value.values for value in partition.frozen_features if value.scenario_id == scenario_id
    ]
    if len(matches) != 1:
        raise ConfirmatoryStudyRunnerError(
            f"restoration feature cache is unavailable or duplicated: {scenario_id}/{partition.role}"
        )
    return cast(NDArray[np.float64], _materialise_rows(matches[0], np.float64))


def _restoration_source_cell_id(
    controls: ConfirmatoryExecutionControls,
    *,
    outer_fold: int,
    corruption_cell_id: str,
) -> str:
    matches = [
        cell.cell_id
        for cell in controls.plan.cells
        if cell.outer_fold == outer_fold
        and cell.corruption_cell_id == corruption_cell_id
        and cell.scenario_id == controls.restoration_scenario_id
        and cell.model_seed == controls.restoration_model_seed
    ]
    if len(matches) != 1:
        raise ConfirmatoryStudyRunnerError("frozen restoration source cell is absent or ambiguous")
    return matches[0]


def _restoration_evidence_arrays(
    prefix: str,
    *,
    core: ConfirmatoryRotationInputs,
    corruption: ConfirmatoryCorruptionInput,
    downstream: Any,
) -> dict[str, NDArray[Any]]:
    """Materialise the exact frozen four-condition evidence schema."""

    return {
        f"{prefix}__audit_sample_ids": np.asarray(core.audit_sample_ids, dtype=np.str_),
        f"{prefix}__audit_group_ids": np.asarray(core.audit_group_ids, dtype=np.str_),
        f"{prefix}__pre_corruption_label": np.asarray(
            corruption.pre_corruption_labels, dtype=np.int64
        ),
        f"{prefix}__observed_label": np.asarray(corruption.observed_labels, dtype=np.int64),
        f"{prefix}__is_injected_corruption": np.asarray(
            corruption.is_injected_corruption, dtype=bool
        ),
        f"{prefix}__guided_reviewed_mask": (
            downstream.audit_guided_restoration_evidence.reviewed_mask
        ),
        f"{prefix}__guided_restored_mask": (
            downstream.audit_guided_restoration_evidence.restored_mask
        ),
        f"{prefix}__guided_restored_label": (
            downstream.audit_guided_restoration_evidence.restored_labels
        ),
        f"{prefix}__random_reviewed_mask": np.stack(
            [value.reviewed_mask for value in downstream.random_review_restoration_evidence]
        ),
        f"{prefix}__random_restored_mask": np.stack(
            [value.restored_mask for value in downstream.random_review_restoration_evidence]
        ),
        f"{prefix}__random_restored_label": np.stack(
            [value.restored_labels for value in downstream.random_review_restoration_evidence]
        ),
        f"{prefix}__final_sample_ids": np.asarray(core.final_sample_ids, dtype=np.str_),
        f"{prefix}__final_group_ids": np.asarray(core.final_group_ids, dtype=np.str_),
        f"{prefix}__final_pre_corruption_label": np.asarray(
            core.final_pre_corruption_labels, dtype=np.int64
        ),
        f"{prefix}__final_observed_label": np.asarray(core.final_observed_labels, dtype=np.int64),
        f"{prefix}__final_is_injected_corruption": np.asarray(
            core.final_is_injected_corruption, dtype=bool
        ),
        f"{prefix}__probabilities__uncorrupted_reference_baseline": (
            downstream.uncorrupted_reference_baseline.final_test_probabilities
        ),
        f"{prefix}__probabilities__corrupted_observed_baseline": (
            downstream.corrupted_observed_baseline.final_test_probabilities
        ),
        f"{prefix}__probabilities__random_review_restoration": np.stack(
            [value.final_test_probabilities for value in downstream.random_review_restoration]
        ),
        f"{prefix}__probabilities__audit_guided_restoration": (
            downstream.audit_guided_restoration.final_test_probabilities
        ),
    }


def _restoration_figure_svg(rotations: Sequence[Mapping[str, Any]]) -> str:
    rows: list[tuple[str, float, float]] = []
    for rotation in rotations:
        conditions = cast(Mapping[str, Any], rotation["conditions"])
        guided_condition = cast(Mapping[str, Any], conditions["audit_guided_restoration"])
        random_condition = cast(Mapping[str, Any], conditions["random_review_restoration"])
        guided_metrics = cast(Mapping[str, Any], guided_condition["metrics"])
        rows.append(
            (
                f"F{rotation['outer_fold']} {rotation['corruption_cell_id']}",
                float(guided_metrics["macro_f1"]),
                float(random_condition["macro_f1_mean"]),
            )
        )
    width = 900
    row_height = 54
    height = 90 + row_height * len(rows)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="32" font-family="sans-serif" font-size="20" '
        'font-weight="bold">Confirmatory restoration macro F1</text>',
        '<text x="520" y="58" font-family="sans-serif" font-size="13" fill="#1f77b4">'
        "audit-guided</text>",
        '<text x="650" y="58" font-family="sans-serif" font-size="13" fill="#999999">'
        "random mean</text>",
    ]
    for index, (label, guided, random_mean) in enumerate(rows):
        y = 78 + index * row_height
        safe_label = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        elements.extend(
            (
                f'<text x="24" y="{y + 18}" font-family="monospace" font-size="12">'
                f"{safe_label}</text>",
                f'<rect x="330" y="{y}" width="{max(0.0, guided) * 450:.3f}" '
                'height="16" fill="#1f77b4"/>',
                f'<rect x="330" y="{y + 20}" width="{max(0.0, random_mean) * 450:.3f}" '
                'height="16" fill="#999999"/>',
                f'<text x="790" y="{y + 14}" font-family="monospace" font-size="11">'
                f"{guided:.4f}</text>",
                f'<text x="790" y="{y + 34}" font-family="monospace" font-size="11">'
                f"{random_mean:.4f}</text>",
            )
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def _confirmatory_report(
    controls: ConfirmatoryExecutionControls,
    matrix_artifacts: ConfirmatoryMatrixArtifacts,
    restoration_metrics: Mapping[str, Any],
    statistics_payload: Mapping[str, Any],
    *,
    analysis_disposition: str | None = None,
) -> str:
    lines = [
        (
            "# Frozen real PanNuke confirmatory study"
            if analysis_disposition is None
            else "# Resource-bounded PanNuke confirmatory sensitivity"
        ),
        "",
        "This controlled, non-diagnostic research benchmark ranks each potentially "
        "inconsistent annotation. High-ranked items are recommended for expert review; "
        "source annotations are never modified automatically.",
        "",
        f"- Frozen config SHA-256: `{controls.config_semantic_sha256}`",
        f"- Planned matrix cells: {len(controls.plan.cells)}",
        f"- Completed required cells: {matrix_artifacts.reconciliation.completed_required_cell_count}",
        f"- Skipped optional cells: {matrix_artifacts.reconciliation.skipped_optional_cell_count}",
        f"- Official-fold rotations: {', '.join(str(value) for value in controls.official_folds)}",
        f"- Paired bootstrap iterations: {controls.paired_group_bootstrap_iterations}",
        "",
    ]
    if analysis_disposition is not None:
        lines.extend(
            (
                f"- Analysis disposition: `{analysis_disposition}`",
                "- Original confirmatory claim allowed: `false`",
                "- Study-outcome eligible: `false`",
                "- Completion stage: `null`",
                "",
                "This post-outcome resource-bounded sensitivity is not the original "
                "confirmatory analysis and cannot unlock external-validation milestone M9.",
                "",
            )
        )
    lines.extend(
        (
            "## Four-condition downstream restoration",
            "",
            "| Fold | Corruption cell | Uncorrupted macro F1 | Corrupted macro F1 | "
            "Random mean macro F1 (95% interval) | Audit-guided macro F1 | "
            "Restored by guided review |",
            "|---:|---|---:|---:|---:|---:|---:|",
        )
    )
    raw_rotations = restoration_metrics.get("rotations")
    rotations = raw_rotations if isinstance(raw_rotations, list) else []
    for raw in rotations:
        if not isinstance(raw, Mapping):
            continue
        conditions = cast(Mapping[str, Any], raw["conditions"])
        clean = cast(Mapping[str, Any], conditions["uncorrupted_reference_baseline"])
        corrupted = cast(Mapping[str, Any], conditions["corrupted_observed_baseline"])
        random = cast(Mapping[str, Any], conditions["random_review_restoration"])
        guided = cast(Mapping[str, Any], conditions["audit_guided_restoration"])
        lines.append(
            "| "
            f"{raw['outer_fold']} | `{raw['corruption_cell_id']}` | "
            f"{float(cast(Mapping[str, Any], clean['metrics'])['macro_f1']):.6f} | "
            f"{float(cast(Mapping[str, Any], corrupted['metrics'])['macro_f1']):.6f} | "
            f"{float(random['macro_f1_mean']):.6f} "
            f"[{float(cast(Sequence[Any], random['macro_f1_interval_95'])[0]):.6f}, "
            f"{float(cast(Sequence[Any], random['macro_f1_interval_95'])[1]):.6f}] | "
            f"{float(cast(Mapping[str, Any], guided['metrics'])['macro_f1']):.6f} | "
            f"{int(guided['restored_count'])} |"
        )
    comparisons = statistics_payload.get("comparisons")
    comparison_rows = comparisons if isinstance(comparisons, list) else []
    lines.extend(("", "## Preregistered paired comparisons", ""))
    for raw in comparison_rows:
        if not isinstance(raw, Mapping):
            continue
        comparison_id = raw.get("comparison_id")
        status = raw.get("status")
        direction = raw.get("direction")
        if status == "completed":
            lines.append(
                f"- `{comparison_id}`: status `completed`, direction `{direction}`, "
                "observed A-minus-B delta "
                f"`{float(raw['observed_delta']):.6f}`, 95% paired group-bootstrap "
                f"interval `[{float(raw['ci_low']):.6f}, "
                f"{float(raw['ci_high']):.6f}]`, Holm-adjusted p "
                f"`{float(raw['holm_adjusted_p']):.6g}`, probability delta > 0 "
                f"`{float(raw['probability_positive']):.6f}`."
            )
        else:
            reason = raw.get("blocker", "no injected events under the frozen selector")
            lines.append(
                f"- `{comparison_id}`: status `{status}`, direction `{direction}`; "
                f"probability delta > 0 not estimated ({reason})."
            )
    lines.extend(
        (
            "",
            "The final reference folds were untouched, uncorrupted, and unavailable for "
            "model selection or tuning. Results describe controlled injected corruption "
            "and do not establish naturally occurring annotation error or clinical truth.",
            "",
            "## Machine-readable scientific result contract",
            "",
            confirmatory_report_contract_block(restoration_metrics, statistics_payload),
            "",
        )
    )
    return "\n".join(lines)


def _restoration_replay_certificate_payload(
    run_directory: Path,
    *,
    bridge: ConfirmatoryBridgeResult,
    controls: ConfirmatoryExecutionControls,
    evidence_path: Path,
    evidence_arrays: Mapping[str, NDArray[Any]],
) -> dict[str, Any]:
    active_corruptions = [value for value in controls.corruption_specs if value.rate > 0.0]
    if not active_corruptions:
        active_corruptions = list(controls.corruption_specs)
    risk_sources: dict[str, dict[str, str]] = {}
    for outer_fold in controls.official_folds:
        for corruption in active_corruptions:
            cell_id = _restoration_source_cell_id(
                controls,
                outer_fold=outer_fold,
                corruption_cell_id=corruption.corruption_cell_id,
            )
            relative = f"cells/{cell_id}/risk_scores.npz"
            risk_sources[f"fold_{outer_fold}__{corruption.corruption_cell_id}"] = {
                "cell_id": cell_id,
                "relative_path": relative,
                "sha256": sha256_file(run_directory / relative),
            }
    l2, max_iter = _logistic_parameters(controls)
    return {
        "schema_version": 1,
        "status": "passed",
        "policy": "deterministic_checksum_bound_restoration_replay_v1",
        "confirmatory_config_semantic_sha256": controls.config_semantic_sha256,
        "execution_controls_binding_sha256": controls.binding_sha256,
        "bridge_partition_content_sha256": bridge.partition_content_sha256,
        "bridge_corruption_assignment_sha256": bridge.corruption_assignment_sha256,
        "bridge_provenance_binding_sha256": bridge.provenance_binding_sha256,
        "scenario_id": controls.restoration_scenario_id,
        "representation_id": controls.restoration_representation_id,
        "model_seed": controls.restoration_model_seed,
        "ranking_method": controls.restoration_ranking_method,
        "review_budget": controls.restoration_review_budget,
        "random_repeats": controls.restoration_random_repeats,
        "random_seed": controls.restoration_random_seed,
        "l2": l2,
        "max_iter": max_iter,
        "evidence_relative_path": evidence_path.relative_to(run_directory).as_posix(),
        "evidence_sha256": sha256_file(evidence_path),
        "evidence_arrays": {
            key: array_artifact_sha256(value) for key, value in sorted(evidence_arrays.items())
        },
        "risk_sources": risk_sources,
    }


def _finalize_confirmatory_analysis(
    *,
    run_directory: Path,
    matrix_artifacts: ConfirmatoryMatrixArtifacts,
    statistics_artifacts: ConfirmatoryStatisticsArtifacts,
    prepared_inputs: PanNukeConfirmatoryInputs,
    bridge: ConfirmatoryBridgeResult,
    controls: ConfirmatoryExecutionControls,
    config_semantic_authority_sha256: str,
    artifact_scope: str | None,
    analysis_disposition: str | None,
    stage_claim_allowed: bool,
) -> None:
    """Execute shared restoration/reporting under one explicit claim policy."""

    run = Path(run_directory).resolve()
    if (
        Path(matrix_artifacts.output_directory).resolve() != run
        or Path(statistics_artifacts.output_directory).resolve() != run
    ):
        raise ConfirmatoryStudyRunnerError("finalizer inputs target different run directories")
    if (
        prepared_inputs.config_sha256 != controls.config_semantic_sha256
        or config_semantic_authority_sha256 != controls.config_semantic_sha256
    ):
        raise ConfirmatoryStudyRunnerError("finalizer differs from frozen config/gate")
    if stage_claim_allowed:
        if artifact_scope not in {None, REAL_CONFIRMATORY_ARTIFACT_SCOPE} or (
            analysis_disposition is not None
        ):
            raise ConfirmatoryStudyRunnerError(
                "stage-claiming finalization requires the original real confirmatory scope"
            )
    elif (
        artifact_scope != RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE
        or analysis_disposition != "amended_or_exploratory"
    ):
        raise ConfirmatoryStudyRunnerError(
            "non-stage finalization requires the exact amended resource-bounded policy"
        )
    draft_completion = _json_object(
        Path(matrix_artifacts.completion_evidence_path),
        "confirmatory matrix completion draft",
    )
    if (
        draft_completion.get("completion_stage") is not None
        or draft_completion.get("study_outcome_eligible") is not False
        or (artifact_scope is not None and draft_completion.get("artifact_scope") != artifact_scope)
    ):
        raise ConfirmatoryStudyRunnerError(
            "finalizer received a matrix draft outside its explicit claim policy"
        )
    scenario = controls.scenarios_by_id[controls.restoration_scenario_id]
    if scenario.family not in {"imagenet_frozen", "pathology_frozen"}:
        raise ConfirmatoryStudyRunnerError(
            "production four-condition finalizer requires a frozen-feature restoration "
            "scenario; CNN restoration is not permitted by this executable contract"
        )
    l2, max_iter = _logistic_parameters(controls)
    source_by_fold = {value.outer_fold: value for value in prepared_inputs.rotations}
    core_by_fold = {value.outer_fold: value for value in bridge.rotations}
    active_corruptions = [value for value in controls.corruption_specs if value.rate > 0.0]
    if not active_corruptions:
        active_corruptions = list(controls.corruption_specs)
    evidence_arrays: dict[str, NDArray[Any]] = {}
    rotation_metrics: list[dict[str, Any]] = []
    for outer_fold in controls.official_folds:
        source = source_by_fold[outer_fold]
        core = core_by_fold[outer_fold]
        audit_features = _partition_feature(source.audit, scenario.scenario_id)
        validation_features = _partition_feature(source.reference_validation, scenario.scenario_id)
        final_features = _partition_feature(source.final_reference, scenario.scenario_id)
        for corruption_spec in active_corruptions:
            corruption = core.corruptions[corruption_spec.corruption_cell_id]
            source_cell_id = _restoration_source_cell_id(
                controls,
                outer_fold=outer_fold,
                corruption_cell_id=corruption_spec.corruption_cell_id,
            )
            risk_path = run / "cells" / source_cell_id / "risk_scores.npz"
            try:
                with np.load(risk_path, allow_pickle=False) as risks:
                    risk_scores = np.asarray(
                        risks[controls.restoration_ranking_method], dtype=np.float64
                    )
            except (OSError, KeyError, ValueError) as error:
                raise ConfirmatoryStudyRunnerError(
                    f"restoration source risk is unavailable: {source_cell_id}"
                ) from error
            downstream = evaluate_downstream_restoration(
                audit_features,
                corruption.pre_corruption_labels,
                corruption.observed_labels,
                corruption.is_injected_corruption,
                final_features,
                core.final_pre_corruption_labels,
                risk_scores,
                development_group_ids=core.audit_group_ids,
                final_test_group_ids=core.final_group_ids,
                final_test_is_injected_corruption=tuple(
                    bool(value) for value in core.final_is_injected_corruption
                ),
                review_budget=controls.restoration_review_budget,
                sample_ids=core.audit_sample_ids,
                class_order=CLASS_ORDER,
                random_repeats=controls.restoration_random_repeats,
                random_seed=controls.restoration_random_seed,
                model_seed=controls.restoration_model_seed,
                l2=l2,
                max_iter=max_iter,
                reference_validation_features=validation_features,
                reference_validation_labels=source.reference_validation.pre_corruption_labels,
                reference_validation_group_ids=source.reference_validation.group_ids,
                reference_validation_is_injected_corruption=(
                    source.reference_validation.is_injected_corruption
                ),
            )
            prefix = f"fold_{outer_fold}__{corruption_spec.corruption_cell_id}"
            evidence_arrays.update(
                _restoration_evidence_arrays(
                    prefix,
                    core=core,
                    corruption=corruption,
                    downstream=downstream,
                )
            )
            downstream_payload = downstream.as_dict()
            rotation_metrics.append(
                {
                    "outer_fold": outer_fold,
                    "corruption_cell_id": corruption_spec.corruption_cell_id,
                    "audit_sample_count": len(core.audit_sample_ids),
                    "final_sample_count": len(core.final_sample_ids),
                    "review_budget_count": downstream.review_budget_count,
                    "random_review_seeds": [
                        controls.restoration_random_seed + repeat
                        for repeat in range(controls.restoration_random_repeats)
                    ],
                    "final_reference_group_ids_sha256": canonical_sha256(
                        sorted(set(core.final_group_ids))
                    ),
                    "conditions": {
                        condition: downstream_payload[condition]
                        for condition in controls.restoration_conditions
                    },
                }
            )
        del audit_features, validation_features, final_features
    evidence_path = _atomic_npz(run / "restoration_evidence.npz", evidence_arrays)
    atomic_write_json(
        run / "restoration_input_bindings.json",
        {
            "schema_version": 1,
            "policy": "immutable_pre_replay_partition_bindings_v1",
            "confirmatory_config_semantic_sha256": controls.config_semantic_sha256,
            "bridge_partition_content_sha256": bridge.partition_content_sha256,
            "bridge_corruption_assignment_sha256": bridge.corruption_assignment_sha256,
            "bridge_provenance_binding_sha256": bridge.provenance_binding_sha256,
            "partition_bindings": {
                fold: {role: dict(binding) for role, binding in partitions.items()}
                for fold, partitions in bridge.partition_bindings.items()
            },
        },
    )
    atomic_write_json(
        run / "restoration_replay_certificate.json",
        _restoration_replay_certificate_payload(
            run,
            bridge=bridge,
            controls=controls,
            evidence_path=evidence_path,
            evidence_arrays=evidence_arrays,
        ),
    )
    restoration_metrics = {
        "schema_version": 1,
        "status": "completed",
        "config_semantic_sha256": controls.config_semantic_sha256,
        "outer_folds": list(controls.official_folds),
        "scenario_id": controls.restoration_scenario_id,
        "model_seed": controls.restoration_model_seed,
        "representation_id": controls.restoration_representation_id,
        "ranking_method": controls.restoration_ranking_method,
        "review_budget": controls.restoration_review_budget,
        "random_repeats": controls.restoration_random_repeats,
        "random_seed": controls.restoration_random_seed,
        "conditions": list(controls.restoration_conditions),
        "evidence_path": evidence_path.name,
        "evidence_sha256": sha256_file(evidence_path),
        "rotations": rotation_metrics,
    }
    atomic_write_json(run / "restoration_metrics.json", restoration_metrics)

    fold_aggregate = _json_object(
        Path(matrix_artifacts.fold_aggregate_path),
        "matrix-core fold aggregate",
    )
    fold_aggregate["outcome_metrics_aggregation_status"] = "completed_by_stage_statistics_runner"
    atomic_write_json(run / "fold_aggregate.json", fold_aggregate)
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    figure_path = atomic_write_text(
        figures / "confirmatory_restoration_macro_f1.svg",
        _restoration_figure_svg(rotation_metrics),
    )
    relative_figure = figure_path.relative_to(run).as_posix()
    atomic_write_json(
        run / "figure_manifest.json",
        {relative_figure: sha256_file(figure_path)},
    )
    statistics_payload = _json_object(
        Path(statistics_artifacts.statistics_path), "paired statistics"
    )
    atomic_write_text(
        run / "report.md",
        _confirmatory_report(
            controls,
            matrix_artifacts,
            restoration_metrics,
            statistics_payload,
            analysis_disposition=analysis_disposition,
        ),
    )
    draft_analysis_gaps = _json_object(
        Path(matrix_artifacts.analysis_gaps_path),
        "matrix-core analysis gaps",
    )
    matrix_execution_eligible = draft_analysis_gaps.get("model_matrix_execution_eligible")
    matrix_execution_telemetry_sha256 = draft_analysis_gaps.get("matrix_execution_telemetry_sha256")
    if (
        type(matrix_execution_eligible) is not bool
        or not _valid_sha(matrix_execution_telemetry_sha256)
        or draft_analysis_gaps.get("matrix_execution_eligibility_source")
        != "real_per_cell_execution_telemetry_and_reconciliation"
    ):
        raise ConfirmatoryStudyRunnerError(
            "finalizer cannot authenticate matrix-core technical execution eligibility"
        )
    analysis_gaps: dict[str, Any] = (
        {
            "schema_version": 1,
            "model_matrix_execution_eligible": matrix_execution_eligible,
            "matrix_execution_telemetry_sha256": (matrix_execution_telemetry_sha256),
            "matrix_execution_eligibility_source": (
                "real_per_cell_execution_telemetry_and_reconciliation"
            ),
            "study_outcome_eligible_pending_seal": (matrix_execution_eligible),
            "missing_stage_analyses": [],
            "reason": "all frozen stage analyses completed; immutable seal remains",
        }
        if stage_claim_allowed
        else {
            "schema_version": 1,
            "model_matrix_execution_eligible": matrix_execution_eligible,
            "matrix_execution_telemetry_sha256": (matrix_execution_telemetry_sha256),
            "matrix_execution_eligibility_source": (
                "real_per_cell_execution_telemetry_and_reconciliation"
            ),
            "study_outcome_eligible_pending_seal": False,
            "missing_stage_analyses": [],
            "analysis_disposition": "amended_or_exploratory",
            "original_confirmatory_claim_allowed": False,
            "completion_stage": None,
            "reason": (
                "all amended resource-bounded analyses completed; this sensitivity "
                "cannot claim CONFIRMATORY_COMPLETE or unlock M9"
            ),
        }
    )
    atomic_write_json(run / "analysis_gaps.json", analysis_gaps)
    root_names = (
        "confirmatory_input_bindings.json",
        "matrix_plan.json",
        "execution_controls.json",
        "frozen_feature_provenance.json",
        "cell_index.csv",
        "reconciliation.json",
        "ensemble_evidence.json",
        "fixed_hybrid_drop_one_ablations.json",
        "paired_statistics.json",
        "paired_bootstrap_evidence.npz",
        "restoration_metrics.json",
        "restoration_evidence.npz",
        "restoration_input_bindings.json",
        "restoration_replay_certificate.json",
        "fold_aggregate.json",
        "original_audit_selection.json",
        "report.md",
        "figure_manifest.json",
    )
    cell_manifests = tuple(
        f"cells/{cell.cell_id}/artifact_manifest.json" for cell in controls.plan.cells
    )
    manifest_paths = (*root_names, *cell_manifests)
    missing = [value for value in manifest_paths if not (run / value).is_file()]
    if missing:
        raise ConfirmatoryStudyRunnerError(
            f"finalizer cannot build exact scientific manifest; missing={missing}"
        )
    atomic_write_json(
        run / CONFIRMATORY_ARTIFACT_MANIFEST_FILENAME,
        {value: sha256_file(run / value) for value in manifest_paths},
    )


def finalize_confirmatory_stage(
    *,
    run_directory: Path,
    matrix_artifacts: ConfirmatoryMatrixArtifacts,
    statistics_artifacts: ConfirmatoryStatisticsArtifacts,
    prepared_inputs: PanNukeConfirmatoryInputs,
    bridge: ConfirmatoryBridgeResult,
    controls: ConfirmatoryExecutionControls,
    gate_evidence: ConfirmatoryExecutionGateEvidence,
) -> None:
    """Execute and persist the original stage-eligible confirmatory analyses."""

    _finalize_confirmatory_analysis(
        run_directory=run_directory,
        matrix_artifacts=matrix_artifacts,
        statistics_artifacts=statistics_artifacts,
        prepared_inputs=prepared_inputs,
        bridge=bridge,
        controls=controls,
        config_semantic_authority_sha256=(
            gate_evidence.primary_gate.confirmatory_config_semantic_sha256
        ),
        artifact_scope=None,
        analysis_disposition=None,
        stage_claim_allowed=True,
    )


def finalize_resource_bounded_analysis(
    *,
    run_directory: Path,
    matrix_artifacts: ConfirmatoryMatrixArtifacts,
    statistics_artifacts: ConfirmatoryStatisticsArtifacts,
    prepared_inputs: PanNukeConfirmatoryInputs,
    bridge: ConfirmatoryBridgeResult,
    controls: ConfirmatoryExecutionControls,
    config_semantic_authority_sha256: str,
) -> None:
    """Persist the amended resource-bounded analyses without a stage claim."""

    _finalize_confirmatory_analysis(
        run_directory=run_directory,
        matrix_artifacts=matrix_artifacts,
        statistics_artifacts=statistics_artifacts,
        prepared_inputs=prepared_inputs,
        bridge=bridge,
        controls=controls,
        config_semantic_authority_sha256=config_semantic_authority_sha256,
        artifact_scope=RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
        analysis_disposition="amended_or_exploratory",
        stage_claim_allowed=False,
    )


def _replay_restoration_arrays(
    run_directory: Path,
    *,
    prepared_inputs: PanNukeConfirmatoryInputs,
    bridge: ConfirmatoryBridgeResult,
    controls: ConfirmatoryExecutionControls,
) -> dict[str, NDArray[Any]]:
    """Re-run every frozen restoration condition from checksum-bound inputs."""

    current_bindings = _prepared_partition_bindings(prepared_inputs)
    frozen_bindings = {
        fold: {role: dict(binding) for role, binding in partitions.items()}
        for fold, partitions in bridge.partition_bindings.items()
    }
    if (
        canonical_sha256(current_bindings) != bridge.partition_content_sha256
        or current_bindings != frozen_bindings
    ):
        raise ConfirmatoryStudyRunnerError(
            "restoration inputs changed after checksum-bound bridge preflight"
        )
    scenario = controls.scenarios_by_id[controls.restoration_scenario_id]
    if scenario.family not in {"imagenet_frozen", "pathology_frozen"}:
        raise ConfirmatoryStudyRunnerError(
            "restoration replay requires the frozen-feature scenario from the config"
        )
    source_by_fold = {value.outer_fold: value for value in prepared_inputs.rotations}
    core_by_fold = {value.outer_fold: value for value in bridge.rotations}
    expected_folds = set(controls.official_folds)
    if set(source_by_fold) != expected_folds or set(core_by_fold) != expected_folds:
        raise ConfirmatoryStudyRunnerError(
            "restoration replay inputs differ from the frozen outer-fold rotation"
        )
    l2, max_iter = _logistic_parameters(controls)
    active_corruptions = [value for value in controls.corruption_specs if value.rate > 0.0]
    if not active_corruptions:
        active_corruptions = list(controls.corruption_specs)
    expected: dict[str, NDArray[Any]] = {}
    for outer_fold in controls.official_folds:
        source = source_by_fold[outer_fold]
        core = core_by_fold[outer_fold]
        audit_features = _partition_feature(source.audit, scenario.scenario_id)
        validation_features = _partition_feature(source.reference_validation, scenario.scenario_id)
        final_features = _partition_feature(source.final_reference, scenario.scenario_id)
        for corruption_spec in active_corruptions:
            corruption = core.corruptions[corruption_spec.corruption_cell_id]
            source_cell_id = _restoration_source_cell_id(
                controls,
                outer_fold=outer_fold,
                corruption_cell_id=corruption_spec.corruption_cell_id,
            )
            risk_path = run_directory / "cells" / source_cell_id / "risk_scores.npz"
            try:
                with np.load(risk_path, allow_pickle=False) as risks:
                    risk_sample_ids = np.asarray(risks["sample_ids"])
                    risk_scores = np.asarray(
                        risks[controls.restoration_ranking_method], dtype=np.float64
                    )
            except (OSError, KeyError, ValueError) as error:
                raise ConfirmatoryStudyRunnerError(
                    f"restoration replay source risk is unavailable: {source_cell_id}"
                ) from error
            if risk_sample_ids.dtype.kind not in {"U", "S"} or not np.array_equal(
                risk_sample_ids, np.asarray(core.audit_sample_ids, dtype=np.str_)
            ):
                raise ConfirmatoryStudyRunnerError(
                    f"restoration replay source risk order differs: {source_cell_id}"
                )
            downstream = evaluate_downstream_restoration(
                audit_features,
                corruption.pre_corruption_labels,
                corruption.observed_labels,
                corruption.is_injected_corruption,
                final_features,
                core.final_pre_corruption_labels,
                risk_scores,
                development_group_ids=core.audit_group_ids,
                final_test_group_ids=core.final_group_ids,
                final_test_is_injected_corruption=tuple(
                    bool(value) for value in core.final_is_injected_corruption
                ),
                review_budget=controls.restoration_review_budget,
                sample_ids=core.audit_sample_ids,
                class_order=CLASS_ORDER,
                random_repeats=controls.restoration_random_repeats,
                random_seed=controls.restoration_random_seed,
                model_seed=controls.restoration_model_seed,
                l2=l2,
                max_iter=max_iter,
                reference_validation_features=validation_features,
                reference_validation_labels=source.reference_validation.pre_corruption_labels,
                reference_validation_group_ids=source.reference_validation.group_ids,
                reference_validation_is_injected_corruption=(
                    source.reference_validation.is_injected_corruption
                ),
            )
            prefix = f"fold_{outer_fold}__{corruption_spec.corruption_cell_id}"
            expected.update(
                _restoration_evidence_arrays(
                    prefix,
                    core=core,
                    corruption=corruption,
                    downstream=downstream,
                )
            )
        del audit_features, validation_features, final_features
    return expected


def _validate_restoration_source_binding(
    run_directory: Path,
    *,
    prepared_inputs: PanNukeConfirmatoryInputs,
    bridge: ConfirmatoryBridgeResult,
    controls: ConfirmatoryExecutionControls,
) -> None:
    """Replay and bind every restoration array to loaded PanNuke/cache bytes."""

    evidence_path = run_directory / "restoration_evidence.npz"
    if not evidence_path.is_file():
        raise ConfirmatoryStudyRunnerError("stage finalizer omitted restoration_evidence.npz")
    expected_arrays = _replay_restoration_arrays(
        run_directory,
        prepared_inputs=prepared_inputs,
        bridge=bridge,
        controls=controls,
    )
    saved_certificate = _json_object(
        run_directory / "restoration_replay_certificate.json",
        "restoration replay certificate",
    )
    expected_certificate = _restoration_replay_certificate_payload(
        run_directory,
        bridge=bridge,
        controls=controls,
        evidence_path=evidence_path,
        evidence_arrays=expected_arrays,
    )
    if saved_certificate != expected_certificate:
        raise ConfirmatoryStudyRunnerError(
            "restoration replay certificate differs from deterministic replay and checksum binding"
        )
    try:
        with np.load(evidence_path, allow_pickle=False) as payload:
            if set(payload.files) != set(expected_arrays):
                missing = sorted(set(expected_arrays).difference(payload.files))
                extra = sorted(set(payload.files).difference(expected_arrays))
                raise ConfirmatoryStudyRunnerError(
                    "restoration evidence schema differs from exact replay: "
                    f"missing={missing}, extra={extra}"
                )
            for key, expected in expected_arrays.items():
                actual = np.asarray(payload[key])
                if actual.dtype.hasobject or not np.array_equal(actual, expected):
                    raise ConfirmatoryStudyRunnerError(
                        "restoration evidence differs from deterministic replay of "
                        f"checksum-bound features, labels, risks, and seeds: {key}"
                    )
            for rotation in bridge.rotations:
                binding = bridge.final_reference_bindings[str(rotation.outer_fold)]
                if (
                    canonical_sha256(list(rotation.final_sample_ids))
                    != binding["sample_ids_sha256"]
                    or canonical_sha256(list(rotation.final_group_ids))
                    != binding["group_ids_sha256"]
                    or array_artifact_sha256(rotation.final_pre_corruption_labels)
                    != binding["pre_corruption_labels_sha256"]
                    or array_artifact_sha256(rotation.final_observed_labels)
                    != binding["observed_labels_sha256"]
                    or array_artifact_sha256(rotation.final_is_injected_corruption)
                    != binding["is_injected_corruption_sha256"]
                ):
                    raise ConfirmatoryStudyRunnerError(
                        "in-memory final-reference binding changed after preflight"
                    )
    except (OSError, ValueError) as error:
        if isinstance(error, ConfirmatoryStudyRunnerError):
            raise
        raise ConfirmatoryStudyRunnerError(
            "restoration evidence cannot be read without pickle"
        ) from error


def _validate_completion_candidate(
    candidate: Mapping[str, Any],
    *,
    plan: ConfirmatoryMatrixPlan,
    readback: ConfirmatoryFilesystemReadback,
    gate: ConfirmatoryExecutionGateEvidence,
) -> dict[str, Any]:
    if (
        candidate.get("schema_version") != 1
        or candidate.get("completion_stage") != _COMPLETION_STAGE
        or candidate.get("study_outcome_eligible") is not True
        or candidate.get("artifact_scope") != REAL_CONFIRMATORY_ARTIFACT_SCOPE
        or candidate.get("matrix_config_sha256") != plan.config_sha256
        or candidate.get("planned_cell_count") != len(plan.cells)
        or candidate.get("required_cell_count") != plan.required_cell_count
        or candidate.get("completed_required_cell_count") != plan.required_cell_count
        or candidate.get("failed_required_cell_count") != 0
        or candidate.get("reconciliation_status") != "passed"
        or candidate.get("fold_rotation_complete") is not True
        or candidate.get("filesystem_readback_status") != "passed"
        or candidate.get("filesystem_matrix_plan_sha256") != readback.matrix_plan_sha256
        or candidate.get("filesystem_cell_index_sha256") != readback.cell_index_sha256
        or candidate.get("filesystem_root_artifact_manifest_sha256")
        != readback.root_artifact_manifest_sha256
        or candidate.get("filesystem_confirmatory_storage_policy_sha256")
        != readback.confirmatory_storage_policy_sha256
        or candidate.get("primary_run_id") != gate.primary_run_id
    ):
        raise ConfirmatoryStudyRunnerError(
            "completion builder did not produce an exact eligible confirmatory claim"
        )
    for field in _PRIMARY_GATE_COMPLETION_HASHES:
        if candidate.get(field) != getattr(gate.primary_gate, field):
            raise ConfirmatoryStudyRunnerError(
                f"completion builder did not preserve primary gate binding {field}"
            )
    for field in _CONFIRMATORY_GATE_COMPLETION_HASHES:
        if candidate.get(field) != getattr(gate, field):
            raise ConfirmatoryStudyRunnerError(
                f"completion builder did not preserve confirmatory gate binding {field}"
            )
    if (
        candidate.get("primary_required_cell_count")
        != gate.primary_gate.primary_required_cell_count
        or candidate.get("completed_primary_required_cell_count")
        != gate.completed_required_cell_count
    ):
        raise ConfirmatoryStudyRunnerError(
            "completion builder did not preserve completed primary matrix counts"
        )
    return dict(candidate)


def _demote_claim(
    tracker: RunTracker,
    error: BaseException,
    *,
    core_completion: Mapping[str, Any] | None,
) -> None:
    if tracker.finalized:
        return
    fallback = dict(core_completion or {})
    fallback.update(
        completion_stage=None,
        study_outcome_eligible=False,
        artifact_scope=REAL_CONFIRMATORY_ARTIFACT_SCOPE,
        valid_completion_claim=False,
        runner_failure=f"{type(error).__name__}: {error}",
    )
    tracker.write_json("completion_evidence.json", fallback)


def _finish_failed_run(
    tracker: RunTracker,
    error: BaseException,
    *,
    core_completion: Mapping[str, Any] | None,
) -> None:
    if tracker.finalized:
        return
    try:
        _demote_claim(tracker, error, core_completion=core_completion)
    except BaseException as demotion_error:
        error.add_note(
            "completion claim demotion failed; run deliberately remains unsealed: "
            f"{type(demotion_error).__name__}: {demotion_error}"
        )
        return
    try:
        tracker.fail(error)
    except BaseException as seal_error:
        error.add_note(
            "failed-run sealing failed after claim demotion: "
            f"{type(seal_error).__name__}: {seal_error}"
        )


def _require_exact_published_t0_lifecycle_pins(
    request: OriginalConfirmatoryCapsuleExecutionRequest,
    readiness: Any,
) -> None:
    """Cross-check the one live publisher/lifecycle result against STATIC v3."""

    published_pins = readiness.published_t0_pins
    published_binding = readiness.verified_published_technical_authority.lifecycle_binding()
    published_binding_unsigned = {
        key: item for key, item in published_binding.items() if key != "binding_sha256"
    }
    if (
        published_pins.namespace_directory != request.technical_authority_namespace_directory
        or published_pins.namespace_claim_sha256
        != request.technical_authority_namespace_claim_sha256
        or published_pins.technical_authority_directory != request.technical_authority_directory
        or published_pins.technical_authority_artifact_root_sha256
        != request.technical_authority_artifact_root_sha256
        or published_pins.technical_authorization_sha256 != request.technical_authorization_sha256
        or published_pins.published_technical_authority_lifecycle_binding_sha256
        != request.published_technical_authority_lifecycle_binding_sha256
        or published_binding.get("binding_sha256")
        != request.published_technical_authority_lifecycle_binding_sha256
        or canonical_sha256(published_binding_unsigned)
        != request.published_technical_authority_lifecycle_binding_sha256
    ):
        raise ConfirmatoryStudyRunnerError(
            "live published technical-authority lifecycle pins differ from "
            "the exact STATIC-v3 capsule request"
        )


def _execute_confirmatory_study_lifecycle(
    *,
    gate_evidence: ConfirmatoryExecutionGateEvidence,
    primary_run_directory: str | Path,
    project_root: str | Path,
    freeze_directory: str | Path,
    dataset_path: str | Path,
    manifest_path: str | Path,
    duplicate_audit_path: str | Path,
    pathology_encoder_audit_path: str | Path,
    frozen_primary_config_path: str | Path,
    frozen_confirmatory_config_path: str | Path,
    crop_cache_path: str | Path,
    expected_crop_cache_sha256: str,
    expected_crop_metadata_sha256: str,
    expected_raw_inventory_sha256: str,
    frozen_feature_caches: Sequence[ConfirmatoryFrozenFeatureCacheSpec],
    observed_label_sets: Sequence[ConfirmatoryObservedLabelSet] = (),
    runs_root: str | Path | None = None,
    run_id: str | None = None,
    retry_of_run_id: str | None = None,
    resume_run_directory: str | Path | None = None,
    lifecycle_readiness_run_directory: str | Path | None = None,
    dependencies: ConfirmatoryRunnerDependencies | None = None,
    capsule_request: OriginalConfirmatoryCapsuleExecutionRequest | None = None,
) -> dict[str, Any]:
    """Execute and seal the complete real confirmatory stage.

    Raw keyword callers remain fresh-only.  The sealed capsule request is a
    closed E-selected fresh/successor union.  A successor is copied only after
    E custody and exact RunTracker creation, with no fallback or retry.
    """

    if dependencies is not None:
        raise ConfirmatoryStudyRunnerError(
            "outcome-eligible confirmatory execution forbids injected dependencies; "
            "structural test doubles must use the permanently non-evidence matrix fixture"
        )
    if capsule_request is not None:
        if type(capsule_request) is not OriginalConfirmatoryCapsuleExecutionRequest:
            raise TypeError("capsule lifecycle requires its exact high-level request type")
        capsule_request.validate()
        if (
            canonical_sha256(capsule_request.gate_evidence.as_dict())
            != capsule_request.expected_gate_evidence_sha256
        ):
            raise ConfirmatoryStudyRunnerError(
                "capsule gate differs from its exact E scientific projection"
            )
        exact_identity_bindings = (
            (gate_evidence, capsule_request.gate_evidence, "gate evidence"),
            (frozen_feature_caches, capsule_request.frozen_feature_caches, "feature caches"),
            (observed_label_sets, capsule_request.observed_label_sets, "observed labels"),
        )
        if any(supplied is not expected for supplied, expected, _ in exact_identity_bindings):
            changed = next(
                role
                for supplied, expected, role in exact_identity_bindings
                if supplied is not expected
            )
            raise ConfirmatoryStudyRunnerError(
                f"capsule lifecycle {changed} escaped its exact closed request"
            )
        exact_value_bindings = {
            "primary_run_directory": (primary_run_directory, capsule_request.primary_run_directory),
            "project_root": (project_root, capsule_request.project_root),
            "freeze_directory": (freeze_directory, capsule_request.freeze_directory),
            "dataset_path": (dataset_path, capsule_request.dataset_path),
            "manifest_path": (manifest_path, capsule_request.manifest_path),
            "duplicate_audit_path": (duplicate_audit_path, capsule_request.duplicate_audit_path),
            "pathology_encoder_audit_path": (
                pathology_encoder_audit_path,
                capsule_request.pathology_encoder_audit_path,
            ),
            "frozen_primary_config_path": (
                frozen_primary_config_path,
                capsule_request.frozen_primary_config_path,
            ),
            "frozen_confirmatory_config_path": (
                frozen_confirmatory_config_path,
                capsule_request.frozen_confirmatory_config_path,
            ),
            "crop_cache_path": (crop_cache_path, capsule_request.crop_cache_path),
            "expected_crop_cache_sha256": (
                expected_crop_cache_sha256,
                capsule_request.expected_crop_cache_sha256,
            ),
            "expected_crop_metadata_sha256": (
                expected_crop_metadata_sha256,
                capsule_request.expected_crop_metadata_sha256,
            ),
            "expected_raw_inventory_sha256": (
                expected_raw_inventory_sha256,
                capsule_request.expected_raw_inventory_sha256,
            ),
            "runs_root": (runs_root, capsule_request.runs_root),
            "run_id": (run_id, capsule_request.run_id),
            "retry_of_run_id": (retry_of_run_id, capsule_request.retry_of_run_id),
            "resume_run_directory": (
                resume_run_directory,
                capsule_request.resume_run_directory,
            ),
            "lifecycle_readiness_run_directory": (
                lifecycle_readiness_run_directory,
                capsule_request.lifecycle_readiness_run_directory,
            ),
        }
        changed_bindings = [
            role
            for role, (supplied, expected) in exact_value_bindings.items()
            if supplied != expected
        ]
        if changed_bindings:
            raise ConfirmatoryStudyRunnerError(
                "capsule lifecycle escaped its exact closed request: " + ", ".join(changed_bindings)
            )
    root = Path(project_root).resolve()
    freeze = _resolve(root, freeze_directory)
    if lifecycle_readiness_run_directory is None:
        raise ConfirmatoryStudyRunnerError(
            "confirmatory lifecycle readiness is required before gate validation, data access, "
            "or RunTracker creation"
        )
    lifecycle_readiness = _resolve(root, lifecycle_readiness_run_directory)
    deps = ConfirmatoryRunnerDependencies()
    primary_run = _resolve(root, primary_run_directory)
    dataset = _resolve(root, dataset_path)
    manifest = _resolve(root, manifest_path)
    duplicate_audit = _resolve(root, duplicate_audit_path)
    pathology_audit = _resolve(root, pathology_encoder_audit_path)
    primary_config_path = _resolve(root, frozen_primary_config_path)
    confirmatory_config_path = _resolve(root, frozen_confirmatory_config_path)
    crop_cache = _resolve(root, crop_cache_path)
    expected_crop_cache_sha256 = _require_sha(expected_crop_cache_sha256, "crop cache SHA-256")
    expected_crop_metadata_sha256 = _require_sha(
        expected_crop_metadata_sha256, "crop metadata SHA-256"
    )
    expected_raw_inventory_sha256 = _require_sha(
        expected_raw_inventory_sha256, "raw inventory SHA-256"
    )
    if capsule_request is None and resume_run_directory is not None:
        raise NotImplementedError(
            "confirmatory successor resume is unavailable through the fresh entrypoint"
        )
    if capsule_request is None and retry_of_run_id is not None:
        raise ConfirmatoryStudyRunnerError(
            "the fresh original-confirmatory runner requires retry_of_run_id=null; "
            "an explicit successor entrypoint must bind and verify predecessor lineage"
        )
    config = deps.config_loader(confirmatory_config_path)
    plan = deps.plan_builder(config)
    _require_legacy_confirmatory_execution_profile(config, plan)
    controls = deps.controls_builder(config)
    controls.validate_for_plan(plan)
    if capsule_request is not None and (
        canonical_sha256(plan.as_dict()) != capsule_request.expected_plan_sha256
        or controls.binding_sha256 != capsule_request.expected_controls_binding_sha256
    ):
        raise ConfirmatoryStudyRunnerError(
            "live plan/controls differ from the exact E scientific projection"
        )
    resolved_specs = tuple(
        ConfirmatoryFrozenFeatureCacheSpec(
            scenario_id=value.scenario_id,
            cache_path=_resolve(root, value.cache_path),
            expected_cache_sha256=value.expected_cache_sha256,
            expected_metadata_sha256=value.expected_metadata_sha256,
            expected_weight_sha256=value.expected_weight_sha256,
        )
        for value in frozen_feature_caches
    )
    _verify_cache_files(
        crop_cache,
        expected_crop_cache_sha256=expected_crop_cache_sha256,
        expected_crop_metadata_sha256=expected_crop_metadata_sha256,
        frozen_feature_caches=resolved_specs,
    )
    preflight_cache_bindings = _original_confirmatory_cache_bindings(
        crop_cache,
        expected_crop_cache_sha256=expected_crop_cache_sha256,
        expected_crop_metadata_sha256=expected_crop_metadata_sha256,
        frozen_feature_caches=resolved_specs,
    )
    run_root = _resolve(root, runs_root) if runs_root is not None else root / "artifacts" / "runs"
    try:
        initial_preflight = require_original_confirmatory_preflight(
            controls,
            target=run_root,
            cache_bindings=preflight_cache_bindings,
        )
    except (OriginalConfirmatoryPreflightError, OSError, ValueError, TypeError) as exc:
        raise ConfirmatoryStudyRunnerError(
            "original confirmatory preflight failed before lifecycle/authority "
            f"consumption, data loading, or RunTracker creation: {type(exc).__name__}: {exc}"
        ) from exc

    from histo_audit.workflows.lifecycle_qualification import (
        LifecycleQualificationError,
        OriginalConfirmatoryPublishedT0LifecycleReadinessVerification,
        require_current_lifecycle_readiness,
        require_current_original_confirmatory_lifecycle_readiness,
    )

    try:
        if capsule_request is None:
            require_current_lifecycle_readiness(
                project_root=root,
                authority_directory=freeze,
                readiness_run_directory=lifecycle_readiness,
            )
        else:
            original_readiness = require_current_original_confirmatory_lifecycle_readiness(
                project_root=root,
                historical_authority_directory=freeze,
                technical_authority_directory=(capsule_request.technical_authority_directory),
                readiness_run_directory=lifecycle_readiness,
            )
            if (
                type(original_readiness)
                is not OriginalConfirmatoryPublishedT0LifecycleReadinessVerification
            ):
                raise LifecycleQualificationError(
                    "strict original-confirmatory lifecycle returned a substituted result type"
                )
            _require_exact_published_t0_lifecycle_pins(
                capsule_request,
                original_readiness,
            )
    except (LifecycleQualificationError, OSError, ValueError, TypeError) as exc:
        raise ConfirmatoryStudyRunnerError(
            "confirmatory lifecycle readiness failed after resource preflight but before "
            "gate validation, data access, or RunTracker creation: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    live_gate = deps.gate_validator(
        primary_run_directory=primary_run,
        project_root=root,
        freeze_directory=freeze,
        dataset_path=dataset,
        manifest_path=manifest,
        duplicate_audit_path=duplicate_audit,
        pathology_encoder_audit_path=pathology_audit,
        frozen_primary_config_path=primary_config_path,
        frozen_confirmatory_config_path=confirmatory_config_path,
    )
    _validate_gate_equality(gate_evidence, live_gate)
    confirmatory_storage_policy_sha256 = _require_sha(
        live_gate.confirmatory_storage_policy_sha256,
        "gated confirmatory storage-policy SHA-256",
    )
    _require_unchanged_confirmatory_storage_policy(
        live_gate.primary_gate.freeze_directory,
        confirmatory_storage_policy_sha256,
        phase="preflight gate revalidation",
    )
    if (
        plan.config_sha256 != live_gate.primary_gate.confirmatory_config_semantic_sha256
        or len(plan.cells) != live_gate.primary_gate.confirmatory_matrix_cell_count
    ):
        raise ConfirmatoryStudyRunnerError("confirmatory plan differs from gated freeze")
    if any(value.configuration_sha256 != plan.config_sha256 for value in observed_label_sets):
        raise ConfirmatoryStudyRunnerError(
            "observed-label materialization differs from the frozen confirmatory config"
        )
    prepared = deps.input_builder(
        crop_cache,
        confirmatory_config=config,
        expected_config_sha256=plan.config_sha256,
        expected_crop_cache_sha256=expected_crop_cache_sha256,
        expected_crop_metadata_sha256=expected_crop_metadata_sha256,
        expected_manifest_sha256=live_gate.primary_gate.manifest_sha256,
        expected_raw_inventory_sha256=expected_raw_inventory_sha256,
        frozen_feature_caches=resolved_specs,
        observed_label_sets=tuple(observed_label_sets),
    )
    bridge = deps.bridge_builder(
        prepared,
        controls,
        pathology_encoder_audit_sha256=(live_gate.primary_gate.pathology_encoder_audit_sha256),
    )
    if (
        capsule_request is not None
        and canonical_sha256(bridge.as_dict()) != capsule_request.expected_bridge_binding_sha256
    ):
        raise ConfirmatoryStudyRunnerError(
            "live outcome-blind bridge differs from the exact E scientific projection"
        )
    cnn_preflight_fingerprints = _confirmatory_cnn_preflight_fingerprints(
        prepared,
        bridge,
        controls,
    )
    fresh_checkpoint_template = build_original_confirmatory_fresh_checkpoint_execution_contract(
        build_original_confirmatory_resume_contract(
            controls=controls,
            cnn_preflight_fingerprints=cnn_preflight_fingerprints,
        )
    )
    fresh_checkpoint_projection = build_original_confirmatory_canonical_e_checkpoint_projection(
        fresh_checkpoint_template
    )
    scientific_checkpoint_authority: OriginalConfirmatoryScientificCheckpointAuthority = (
        fresh_checkpoint_projection
    )
    if capsule_request is not None:
        if fresh_checkpoint_template != capsule_request.draft_checkpoint_contract:
            raise ConfirmatoryStudyRunnerError(
                "live original-confirmatory inputs differ from the exact Q template"
            )
        scientific_checkpoint_authority = capsule_request.canonical_e_projection
        if capsule_request.execution_mode == "fresh":
            if scientific_checkpoint_authority != fresh_checkpoint_projection:
                raise ConfirmatoryStudyRunnerError(
                    "live original-confirmatory inputs differ from fresh E"
                )
            require_original_confirmatory_canonical_e_projection(
                fresh_checkpoint_projection,
                fresh_checkpoint_template,
            )
        elif (
            type(scientific_checkpoint_authority)
            is not OriginalConfirmatorySuccessorPrecopyCheckpointProjection
        ):
            raise ConfirmatoryStudyRunnerError(
                "successor capsule lacks its exact tagged pre-copy E authority"
            )
    checkpoint_execution_contract = fresh_checkpoint_template
    checkpoint_execution_projection = fresh_checkpoint_projection
    scientific_checkpoint_authority_sha256 = canonical_sha256(
        scientific_checkpoint_authority.as_dict()
    )
    scientific_checkpoint_contract_sha256 = (
        scientific_checkpoint_authority.projection_sha256
        if isinstance(
            scientific_checkpoint_authority,
            OriginalConfirmatorySuccessorPrecopyCheckpointProjection,
        )
        else scientific_checkpoint_authority.checkpoint_contract_sha256
    )
    final_preflight_holder: dict[str, OriginalConfirmatoryPreflightReceipt] = {}

    def final_preflight_recheck() -> Mapping[str, Any]:
        _verify_cache_files(
            crop_cache,
            expected_crop_cache_sha256=expected_crop_cache_sha256,
            expected_crop_metadata_sha256=expected_crop_metadata_sha256,
            frozen_feature_caches=resolved_specs,
        )
        try:
            final_receipt = recheck_original_confirmatory_capacity(
                initial_preflight,
                controls,
                target=run_root,
                cache_bindings=preflight_cache_bindings,
            )
            final_preflight_holder["receipt"] = final_receipt
        except (OriginalConfirmatoryPreflightError, OSError, ValueError, TypeError) as exc:
            raise ConfirmatoryStudyRunnerError(
                "original confirmatory guarded final cache/compute/capacity "
                f"recheck failed closed: {type(exc).__name__}: {exc}"
            ) from exc
        return {
            "original_confirmatory_final_preflight_receipt": (final_receipt.as_dict()),
            "original_confirmatory_final_preflight_receipt_sha256": (
                final_receipt.receipt_without_self_hash_sha256
            ),
        }

    # The preliminary public gate above is evidence, never execution authority.
    # The production helper holds the primary mutation lock across the complete
    # final gate, cache recheck, and successful dependent RunTracker creation.
    final_gate, tracker = _guarded_final_gate_and_start(
        primary_run_directory=primary_run,
        expected_gate=live_gate,
        gate_validator=deps.gate_validator,
        gate_kwargs={
            "primary_run_directory": primary_run,
            "project_root": root,
            "freeze_directory": freeze,
            "dataset_path": dataset,
            "manifest_path": manifest,
            "duplicate_audit_path": duplicate_audit,
            "pathology_encoder_audit_path": pathology_audit,
            "frozen_primary_config_path": primary_config_path,
            "frozen_confirmatory_config_path": confirmatory_config_path,
        },
        cache_recheck=final_preflight_recheck,
        tracker_starter=deps.tracker_starter,
        tracker_kwargs={
            "experiment_name": _EXPERIMENT_NAME,
            "config": config,
            "project_root": root,
            "runs_root": run_root,
            "run_id": run_id,
            "environment": {
                "artifact_scope": REAL_CONFIRMATORY_ARTIFACT_SCOPE,
                "confirmatory_gate": live_gate.as_dict(),
                "original_confirmatory_capacity_policy_sha256": (
                    ORIGINAL_CONFIRMATORY_CAPACITY_POLICY_SHA256
                ),
                "original_confirmatory_initial_preflight_sha256": (
                    initial_preflight.receipt_without_self_hash_sha256
                ),
                "original_confirmatory_checkpoint_execution_mode": (
                    scientific_checkpoint_authority.execution_mode
                ),
                "original_confirmatory_checkpoint_contract_profile": (
                    scientific_checkpoint_authority.contract_profile
                ),
                "original_confirmatory_checkpoint_contract_sha256": (
                    scientific_checkpoint_contract_sha256
                ),
                "original_confirmatory_checkpoint_directives_sha256": (
                    scientific_checkpoint_authority.directives_sha256
                ),
                "original_confirmatory_scientific_authority_projection_sha256": (
                    scientific_checkpoint_authority_sha256
                ),
                "completion_valid_only_after_post_seal_integrity": True,
            },
            "dataset_path": dataset,
            "manifest_path": manifest,
            "duplicate_audit_status": (
                f"complete_sha256:{live_gate.primary_gate.duplicate_audit_sha256}"
            ),
        },
    )
    core_completion: dict[str, Any] | None = None
    candidate: dict[str, Any] | None = None
    successor_copy_receipt: dict[str, Any] | None = None
    try:
        if (
            capsule_request is not None
            and tracker.run_directory != capsule_request.expected_run_directory
        ):
            raise ConfirmatoryStudyRunnerError(
                "RunTracker directory differs from the exact E attempt projection"
            )
        if capsule_request is not None and capsule_request.execution_mode == "successor_resume":
            authority = capsule_request.canonical_e_projection
            if type(authority) is not OriginalConfirmatorySuccessorPrecopyCheckpointProjection:
                raise ConfirmatoryStudyRunnerError(
                    "successor materialization lacks its exact pre-copy E authority"
                )
            materialized = materialize_original_confirmatory_successor_checkpoint_execution(
                authority,
                destination_run_directory=tracker.run_directory,
                fresh_template_contract=fresh_checkpoint_template,
            )
            checkpoint_execution_contract = materialized.execution_contract
            checkpoint_execution_projection = materialized.checkpoint_projection
            successor_copy_receipt = materialized.copy_receipt.as_dict()
        final_preflight = final_preflight_holder.get("receipt")
        if not isinstance(final_preflight, OriginalConfirmatoryPreflightReceipt):
            raise ConfirmatoryStudyRunnerError(
                "guarded final preflight did not produce its typed receipt"
            )
        if (
            tracker.source_tree.get("root_sha256")
            != final_gate.primary_gate.source_tree_root_sha256
        ):
            raise ConfirmatoryStudyRunnerError(
                "source tree changed between final gate and RunTracker capture"
            )
        tracker.write_json("confirmatory_execution_gate.json", live_gate.as_dict())
        initial_preflight_path = tracker.write_json(
            ORIGINAL_CONFIRMATORY_INITIAL_PREFLIGHT_FILENAME,
            initial_preflight.as_dict(),
        )
        final_preflight_path = tracker.write_json(
            ORIGINAL_CONFIRMATORY_FINAL_PREFLIGHT_FILENAME,
            final_preflight.as_dict(),
        )
        scientific_authority_path = tracker.write_json(
            "original_confirmatory_scientific_checkpoint_authority.json",
            scientific_checkpoint_authority.as_dict(),
        )
        successor_copy_receipt_path = (
            tracker.write_json(
                "original_confirmatory_successor_copy_receipt.json",
                successor_copy_receipt,
            )
            if successor_copy_receipt is not None
            else None
        )
        checkpoint_contract_path = tracker.write_json(
            ORIGINAL_CONFIRMATORY_CHECKPOINT_EXECUTION_CONTRACT_FILENAME,
            checkpoint_execution_projection.as_dict(),
        )
        bindings_path = tracker.write_json(
            "confirmatory_input_bindings.json",
            {
                "schema_version": 1,
                "crop_cache_path": str(crop_cache),
                "crop_cache_sha256": expected_crop_cache_sha256,
                "crop_metadata_sha256": expected_crop_metadata_sha256,
                "raw_inventory_sha256": expected_raw_inventory_sha256,
                "manifest_sha256": prepared.manifest_sha256,
                "config_semantic_sha256": prepared.config_sha256,
                "execution_controls_binding_sha256": controls.binding_sha256,
                "confirmatory_storage_policy_sha256": (confirmatory_storage_policy_sha256),
                "bridge": bridge.as_dict(),
                "cnn_fold_data_and_split_sha256": cnn_preflight_fingerprints,
                "original_confirmatory_capacity_policy_sha256": (
                    ORIGINAL_CONFIRMATORY_CAPACITY_POLICY_SHA256
                ),
                "original_confirmatory_initial_preflight_sha256": (
                    sha256_file(initial_preflight_path)
                ),
                "original_confirmatory_initial_preflight_receipt_sha256": (
                    initial_preflight.receipt_without_self_hash_sha256
                ),
                "original_confirmatory_final_preflight_sha256": (sha256_file(final_preflight_path)),
                "original_confirmatory_final_preflight_receipt_sha256": (
                    final_preflight.receipt_without_self_hash_sha256
                ),
                "original_confirmatory_checkpoint_execution_contract_sha256": (
                    sha256_file(checkpoint_contract_path)
                ),
                "original_confirmatory_scientific_checkpoint_authority_sha256": (
                    sha256_file(scientific_authority_path)
                ),
                "original_confirmatory_scientific_authority_projection_sha256": (
                    scientific_checkpoint_authority_sha256
                ),
                "original_confirmatory_successor_copy_receipt_sha256": (
                    sha256_file(successor_copy_receipt_path)
                    if successor_copy_receipt_path is not None
                    else None
                ),
                "original_confirmatory_checkpoint_execution_mode": (
                    checkpoint_execution_projection.execution_mode
                ),
                "original_confirmatory_checkpoint_contract_profile": (
                    checkpoint_execution_projection.contract_profile
                ),
                "original_confirmatory_checkpoint_contract_sha256": (
                    checkpoint_execution_projection.checkpoint_contract_sha256
                ),
                "original_confirmatory_checkpoint_directives_sha256": (
                    checkpoint_execution_projection.directives_sha256
                ),
                "feature_caches": [
                    {
                        **asdict(spec),
                        "cache_path": str(spec.cache_path),
                    }
                    for spec in resolved_specs
                ],
                "retry_of_run_id": retry_of_run_id,
                "resume_policy": (
                    "fresh_exact_180_no_predecessor_no_retry_no_fallback"
                    if retry_of_run_id is None
                    else (
                        "successor_exact_180_post_e_o_excl_copy_no_discovery_no_retry_no_fallback"
                    )
                ),
            },
        )
        tracker.write_provenance(
            artifact_scope=REAL_CONFIRMATORY_ARTIFACT_SCOPE,
            confirmatory_gate=live_gate.as_dict(),
            confirmatory_input_bindings_sha256=sha256_file(bindings_path),
            confirmatory_storage_policy_sha256=confirmatory_storage_policy_sha256,
            original_confirmatory_capacity_policy_sha256=(
                ORIGINAL_CONFIRMATORY_CAPACITY_POLICY_SHA256
            ),
            original_confirmatory_initial_preflight_sha256=(sha256_file(initial_preflight_path)),
            original_confirmatory_initial_preflight_receipt_sha256=(
                initial_preflight.receipt_without_self_hash_sha256
            ),
            original_confirmatory_final_preflight_sha256=sha256_file(final_preflight_path),
            original_confirmatory_final_preflight_receipt_sha256=(
                final_preflight.receipt_without_self_hash_sha256
            ),
            original_confirmatory_checkpoint_execution_contract_sha256=(
                sha256_file(checkpoint_contract_path)
            ),
            original_confirmatory_scientific_checkpoint_authority_sha256=(
                sha256_file(scientific_authority_path)
            ),
            original_confirmatory_scientific_authority_projection_sha256=(
                scientific_checkpoint_authority_sha256
            ),
            original_confirmatory_successor_copy_receipt_sha256=(
                sha256_file(successor_copy_receipt_path)
                if successor_copy_receipt_path is not None
                else None
            ),
            original_confirmatory_checkpoint_execution_mode=(
                checkpoint_execution_projection.execution_mode
            ),
            original_confirmatory_checkpoint_contract_profile=(
                checkpoint_execution_projection.contract_profile
            ),
            original_confirmatory_checkpoint_contract_sha256=(
                checkpoint_execution_projection.checkpoint_contract_sha256
            ),
            original_confirmatory_checkpoint_directives_sha256=(
                checkpoint_execution_projection.directives_sha256
            ),
            matrix_plan_sha256=canonical_sha256(plan.as_dict()),
            execution_controls_binding_sha256=controls.binding_sha256,
            retry_of_run_id=retry_of_run_id,
            completion_stage_valid_only_after_post_seal_integrity=True,
        )
        artifacts = _execute_original_confirmatory_prepared_matrix(
            _OriginalConfirmatoryPreparedMatrixRequest(
                rotations=bridge.rotations,
                plan=plan,
                controls=controls,
                output_directory=tracker.run_directory,
                frozen_blockers=tuple(sorted(bridge.frozen_blockers.items())),
                artifact_scope=REAL_CONFIRMATORY_ARTIFACT_SCOPE,
                gate_evidence=live_gate,
                canonical_e_projection=checkpoint_execution_projection,
                scientific_authority_projection_sha256=(scientific_checkpoint_authority_sha256),
                draft_checkpoint_contract=checkpoint_execution_contract,
            )
        )
        core_completion = _validate_core_artifacts(
            artifacts,
            run_directory=tracker.run_directory,
            plan=plan,
        )
        statistics = deps.statistics_aggregator(tracker.run_directory, controls)
        statistics_verification = deps.statistics_verifier(tracker.run_directory, controls)
        if (
            statistics_verification.status != "passed"
            or Path(statistics.output_directory).resolve() != tracker.run_directory
            or Path(statistics_verification.output_directory).resolve() != tracker.run_directory
            or statistics.statistics_sha256 != statistics_verification.statistics_sha256
            or statistics.bootstrap_evidence_sha256
            != statistics_verification.bootstrap_evidence_sha256
        ):
            raise ConfirmatoryStudyRunnerError(
                "confirmatory paired statistics failed strict verification"
            )
        deps.stage_finalizer(
            run_directory=tracker.run_directory,
            matrix_artifacts=artifacts,
            statistics_artifacts=statistics,
            prepared_inputs=prepared,
            bridge=bridge,
            controls=controls,
            gate_evidence=live_gate,
        )
        deps.restoration_verifier(
            run_directory=tracker.run_directory,
            prepared_inputs=prepared,
            bridge=bridge,
            controls=controls,
        )
        readback = deps.filesystem_reader(
            plan,
            tracker.run_directory,
            frozen_confirmatory_config_path=confirmatory_config_path,
            expected_frozen_config_sha256=(
                live_gate.primary_gate.frozen_confirmatory_config_sha256
            ),
            expected_confirmatory_storage_policy_sha256=(confirmatory_storage_policy_sha256),
            require_final_policy_bindings=False,
        )
        _validate_final_readback(
            readback,
            artifacts,
            tracker.run_directory,
            expected_confirmatory_storage_policy_sha256=(confirmatory_storage_policy_sha256),
        )
        raw_candidate = deps.completion_builder(
            plan=plan,
            reconciliation=artifacts.reconciliation,
            artifact_scope=REAL_CONFIRMATORY_ARTIFACT_SCOPE,
            study_outcome_eligible=True,
            gate_evidence=live_gate,
            run_directory=tracker.run_directory,
        )
        candidate = _validate_completion_candidate(
            raw_candidate,
            plan=plan,
            readback=readback,
            gate=live_gate,
        )
        candidate.update(
            run_id=tracker.run_id,
            retry_of_run_id=retry_of_run_id,
            statistics_sha256=statistics_verification.statistics_sha256,
            bootstrap_evidence_sha256=(statistics_verification.bootstrap_evidence_sha256),
            post_seal_integrity_verification_required=True,
            post_seal_attestation_required=True,
        )
        tracker.write_json("core_completion_evidence.json", core_completion)
        completion_path = tracker.write_json("completion_evidence.json", candidate)
        report = (tracker.run_directory / "report.md").read_text(encoding="utf-8")
        if (
            "potentially inconsistent annotation" not in report
            or "recommended for expert review" not in report
        ):
            raise ConfirmatoryStudyRunnerError(
                "final report lacks mandatory non-diagnostic review terminology"
            )
        tracker.write_metrics(
            {
                "schema_version": 1,
                "artifact_scope": REAL_CONFIRMATORY_ARTIFACT_SCOPE,
                "study_outcome_eligible": True,
                "completion_stage": _COMPLETION_STAGE,
                "run_id": tracker.run_id,
                "matrix_config_sha256": plan.config_sha256,
                "planned_cell_count": len(plan.cells),
                "required_cell_count": plan.required_cell_count,
                "completed_required_cell_count": plan.required_cell_count,
                "filesystem_readback_status": readback.status,
                "filesystem_checked_artifact_count": readback.checked_artifact_count,
                "completion_evidence_sha256": sha256_file(completion_path),
                "statistics_sha256": statistics_verification.statistics_sha256,
                "bootstrap_evidence_sha256": (statistics_verification.bootstrap_evidence_sha256),
                "confirmatory_storage_policy_sha256": (confirmatory_storage_policy_sha256),
                "valid_completion_claim": "pending_post_seal_verification",
            }
        )
        tracker.log_event(
            "confirmatory_completion_candidate_written",
            completion_stage=_COMPLETION_STAGE,
            completion_evidence_sha256=sha256_file(completion_path),
        )
        # If start won the initial race, a later primary withdrawal must still
        # default-deny this candidate before it can become an immutable success.
        with guard_run_stage_eligibility(primary_run) as preseal_receipt:
            preseal_gate = deps.gate_validator(
                primary_run_directory=primary_run,
                project_root=root,
                freeze_directory=freeze,
                dataset_path=dataset,
                manifest_path=manifest,
                duplicate_audit_path=duplicate_audit,
                pathology_encoder_audit_path=pathology_audit,
                frozen_primary_config_path=primary_config_path,
                frozen_confirmatory_config_path=confirmatory_config_path,
                primary_stage_eligibility_receipt=preseal_receipt,
            )
            _validate_gate_equality(final_gate, preseal_gate)
            tracker.complete()
    except BaseException as error:
        _finish_failed_run(tracker, error, core_completion=core_completion)
        raise

    integrity = deps.integrity_verifier(tracker.run_directory)
    if (
        not integrity.valid
        or not integrity.registry_record_present
        or integrity.run_id != tracker.run_id
        or not _valid_sha(integrity.expected_root_sha256)
    ):
        raise ConfirmatoryStudyIntegrityError(
            "sealed confirmatory candidate failed post-seal integrity verification; "
            f"no valid {_COMPLETION_STAGE} claim is returned: {integrity.errors}"
        )
    try:
        sealed = _json_object(
            tracker.run_directory / "completion_evidence.json",
            "sealed confirmatory completion evidence",
        )
        if candidate is None or sealed != candidate:
            raise ConfirmatoryStudyIntegrityError(
                "sealed confirmatory completion differs from verified candidate"
            )
        post_seal_readback = deps.filesystem_reader(
            plan,
            tracker.run_directory,
            frozen_confirmatory_config_path=confirmatory_config_path,
            expected_frozen_config_sha256=(
                live_gate.primary_gate.frozen_confirmatory_config_sha256
            ),
            expected_confirmatory_storage_policy_sha256=(confirmatory_storage_policy_sha256),
            require_final_policy_bindings=True,
        )
        _validate_final_readback(
            post_seal_readback,
            artifacts,
            tracker.run_directory,
            expected_confirmatory_storage_policy_sha256=(confirmatory_storage_policy_sha256),
        )
        if (
            post_seal_readback.matrix_plan_sha256 != readback.matrix_plan_sha256
            or post_seal_readback.cell_index_sha256 != readback.cell_index_sha256
            or post_seal_readback.root_artifact_manifest_sha256
            != readback.root_artifact_manifest_sha256
            or post_seal_readback.reconciliation != readback.reconciliation
        ):
            raise ConfirmatoryStudyIntegrityError(
                "post-seal scientific readback differs from the exact pre-seal readback"
            )
        _require_unchanged_confirmatory_storage_policy(
            live_gate.primary_gate.freeze_directory,
            confirmatory_storage_policy_sha256,
            phase="post-seal authority revalidation",
        )
        final_integrity = deps.integrity_verifier(tracker.run_directory)
        if (
            not final_integrity.valid
            or not final_integrity.registry_record_present
            or final_integrity.run_id != tracker.run_id
            or final_integrity.expected_root_sha256 != integrity.expected_root_sha256
            or final_integrity.actual_root_sha256 != integrity.actual_root_sha256
            or not _valid_sha(final_integrity.expected_root_sha256)
        ):
            raise ConfirmatoryStudyIntegrityError(
                "sealed confirmatory candidate failed the final integrity verification "
                "after scientific readback"
            )
    except BaseException as error:
        failure = (
            error
            if isinstance(error, ConfirmatoryStudyIntegrityError)
            else ConfirmatoryStudyIntegrityError(
                "sealed confirmatory candidate failed post-seal scientific readback; "
                f"no valid {_COMPLETION_STAGE} claim is returned"
            )
        )
        try:
            deps.eligibility_withdrawer(
                tracker.run_directory,
                reason_code="confirmatory_postseal_verification_failed",
                reason=(
                    "The sealed confirmatory candidate failed semantic readback or the "
                    "immediately following integrity verification; scientific stage "
                    "eligibility is permanently withdrawn."
                ),
            )
        except BaseException as withdrawal_error:
            failure.add_note(
                "automatic eligibility withdrawal could not be committed (the sealed "
                "run may already be integrity-invalid): "
                f"{type(withdrawal_error).__name__}: {withdrawal_error}"
            )
        if failure is error:
            raise
        raise failure from error
    try:
        # The upstream primary remains guarded through the final live gate,
        # positive confirmatory attestation, exact readback, and return payload
        # construction.  A withdrawal that wins first makes this sealed run
        # permanently ineligible in the failure handler below.
        with guard_run_stage_eligibility(primary_run) as postseal_receipt:
            if postseal_receipt is None:
                raise ConfirmatoryStudyIntegrityError(
                    "primary run lacks active stage authority before final attestation"
                )
            postseal_gate = deps.gate_validator(
                primary_run_directory=primary_run,
                project_root=root,
                freeze_directory=freeze,
                dataset_path=dataset,
                manifest_path=manifest,
                duplicate_audit_path=duplicate_audit,
                pathology_encoder_audit_path=pathology_audit,
                frozen_primary_config_path=primary_config_path,
                frozen_confirmatory_config_path=confirmatory_config_path,
                primary_stage_eligibility_receipt=postseal_receipt,
            )
            _validate_gate_equality(final_gate, postseal_gate)
            guarded_final_integrity = deps.integrity_verifier(tracker.run_directory)
            if (
                not guarded_final_integrity.valid
                or not guarded_final_integrity.registry_record_present
                or guarded_final_integrity.run_id != tracker.run_id
                or guarded_final_integrity.expected_root_sha256
                != final_integrity.expected_root_sha256
                or guarded_final_integrity.actual_root_sha256 != final_integrity.actual_root_sha256
            ):
                raise ConfirmatoryStudyIntegrityError(
                    "confirmatory integrity changed before final positive attestation"
                )
            post_seal_verification = {
                "schema_version": 1,
                "policy": "confirmatory_postseal_attestation_v1",
                "run_id": tracker.run_id,
                "completion_stage": _COMPLETION_STAGE,
                "first_integrity_root_sha256": integrity.expected_root_sha256,
                "final_integrity_root_sha256": guarded_final_integrity.expected_root_sha256,
                "matrix_plan_sha256": post_seal_readback.matrix_plan_sha256,
                "cell_index_sha256": post_seal_readback.cell_index_sha256,
                "scientific_artifact_manifest_sha256": (
                    post_seal_readback.root_artifact_manifest_sha256
                ),
                "confirmatory_storage_policy_sha256": confirmatory_storage_policy_sha256,
                "reconciliation_sha256": canonical_sha256(
                    post_seal_readback.reconciliation.as_dict()
                    if post_seal_readback.reconciliation is not None
                    else None
                ),
                "semantic_readback_status": post_seal_readback.status,
                "semantic_checked_artifact_count": post_seal_readback.checked_artifact_count,
            }
            verification_sha256 = canonical_sha256(post_seal_verification)
            attestation = attest_run_stage_eligibility(
                tracker.run_directory,
                completion_stage=_COMPLETION_STAGE,
                verification=post_seal_verification,
                primary_eligibility_receipt=postseal_receipt,
            )
            if (
                attestation.get("run_id") != tracker.run_id
                or attestation.get("artifact_root_sha256")
                != guarded_final_integrity.expected_root_sha256
                or attestation.get("verification_sha256") != verification_sha256
            ):
                raise ConfirmatoryStudyIntegrityError(
                    "positive post-seal attestation failed exact readback"
                )
            return {
                "status": "completed",
                "completion_stage": _COMPLETION_STAGE,
                "study_outcome_eligible": True,
                "artifact_scope": REAL_CONFIRMATORY_ARTIFACT_SCOPE,
                "run_id": tracker.run_id,
                "run_directory": str(tracker.run_directory),
                "artifact_root_sha256": guarded_final_integrity.expected_root_sha256,
                "registry_record_present": guarded_final_integrity.registry_record_present,
                "post_seal_filesystem_readback_status": post_seal_readback.status,
                "post_seal_attestation_record_sha256": attestation["record_sha256"],
                "post_seal_verification_sha256": verification_sha256,
                "confirmatory_storage_policy_sha256": confirmatory_storage_policy_sha256,
                "completion_evidence_path": str(tracker.run_directory / "completion_evidence.json"),
                "completion_evidence_sha256": sha256_file(
                    tracker.run_directory / "completion_evidence.json"
                ),
                "reconciliation_path": str(tracker.run_directory / "reconciliation.json"),
                "metrics_path": str(tracker.run_directory / "metrics.json"),
                "report_path": str(tracker.run_directory / "report.md"),
                "planned_cell_count": len(plan.cells),
                "completed_required_cell_count": plan.required_cell_count,
                "retry_of_run_id": retry_of_run_id,
            }
    except BaseException as error:
        failure = (
            error
            if isinstance(error, ConfirmatoryStudyIntegrityError)
            else ConfirmatoryStudyIntegrityError(
                "primary eligibility or confirmatory integrity changed before final positive "
                f"attestation; no valid {_COMPLETION_STAGE} claim is returned"
            )
        )
        try:
            deps.eligibility_withdrawer(
                tracker.run_directory,
                reason_code="confirmatory_final_upstream_recheck_failed",
                reason=(
                    "The final guarded primary-stage recheck or confirmatory attestation "
                    "failed; scientific stage eligibility is permanently withdrawn."
                ),
            )
        except BaseException as withdrawal_error:
            failure.add_note(
                "automatic final eligibility withdrawal could not be committed: "
                f"{type(withdrawal_error).__name__}: {withdrawal_error}"
            )
        if failure is error:
            raise
        raise failure from error


def _execute_original_confirmatory_capsule_lifecycle(
    request: OriginalConfirmatoryCapsuleExecutionRequest,
) -> dict[str, Any]:
    """Internal target of the sole sealed high-level capsule entry."""

    if type(request) is not OriginalConfirmatoryCapsuleExecutionRequest:
        raise TypeError("capsule lifecycle requires its exact high-level request type")
    request.validate()
    return _execute_confirmatory_study_lifecycle(
        gate_evidence=request.gate_evidence,
        primary_run_directory=request.primary_run_directory,
        project_root=request.project_root,
        freeze_directory=request.freeze_directory,
        dataset_path=request.dataset_path,
        manifest_path=request.manifest_path,
        duplicate_audit_path=request.duplicate_audit_path,
        pathology_encoder_audit_path=request.pathology_encoder_audit_path,
        frozen_primary_config_path=request.frozen_primary_config_path,
        frozen_confirmatory_config_path=request.frozen_confirmatory_config_path,
        crop_cache_path=request.crop_cache_path,
        expected_crop_cache_sha256=request.expected_crop_cache_sha256,
        expected_crop_metadata_sha256=request.expected_crop_metadata_sha256,
        expected_raw_inventory_sha256=request.expected_raw_inventory_sha256,
        frozen_feature_caches=request.frozen_feature_caches,
        observed_label_sets=request.observed_label_sets,
        runs_root=request.runs_root,
        run_id=request.run_id,
        retry_of_run_id=request.retry_of_run_id,
        resume_run_directory=request.resume_run_directory,
        lifecycle_readiness_run_directory=request.lifecycle_readiness_run_directory,
        dependencies=None,
        capsule_request=request,
    )


def execute_confirmatory_study(
    *,
    gate_evidence: ConfirmatoryExecutionGateEvidence,
    primary_run_directory: str | Path,
    project_root: str | Path,
    freeze_directory: str | Path,
    dataset_path: str | Path,
    manifest_path: str | Path,
    duplicate_audit_path: str | Path,
    pathology_encoder_audit_path: str | Path,
    frozen_primary_config_path: str | Path,
    frozen_confirmatory_config_path: str | Path,
    crop_cache_path: str | Path,
    expected_crop_cache_sha256: str,
    expected_crop_metadata_sha256: str,
    expected_raw_inventory_sha256: str,
    frozen_feature_caches: Sequence[ConfirmatoryFrozenFeatureCacheSpec],
    observed_label_sets: Sequence[ConfirmatoryObservedLabelSet] = (),
    runs_root: str | Path | None = None,
    run_id: str | None = None,
    retry_of_run_id: str | None = None,
    resume_run_directory: str | Path | None = None,
    lifecycle_readiness_run_directory: str | Path | None = None,
    dependencies: ConfirmatoryRunnerDependencies | None = None,
) -> dict[str, Any]:
    """Compatibility entry for tests and non-capsule preflight validation.

    The sealed production handler does not call this raw-keyword surface; it can
    enter only through ``OriginalConfirmatoryCapsuleExecutionRequest``.
    """

    return _execute_confirmatory_study_lifecycle(
        gate_evidence=gate_evidence,
        primary_run_directory=primary_run_directory,
        project_root=project_root,
        freeze_directory=freeze_directory,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        duplicate_audit_path=duplicate_audit_path,
        pathology_encoder_audit_path=pathology_encoder_audit_path,
        frozen_primary_config_path=frozen_primary_config_path,
        frozen_confirmatory_config_path=frozen_confirmatory_config_path,
        crop_cache_path=crop_cache_path,
        expected_crop_cache_sha256=expected_crop_cache_sha256,
        expected_crop_metadata_sha256=expected_crop_metadata_sha256,
        expected_raw_inventory_sha256=expected_raw_inventory_sha256,
        frozen_feature_caches=frozen_feature_caches,
        observed_label_sets=observed_label_sets,
        runs_root=runs_root,
        run_id=run_id,
        retry_of_run_id=retry_of_run_id,
        resume_run_directory=resume_run_directory,
        lifecycle_readiness_run_directory=lifecycle_readiness_run_directory,
        dependencies=dependencies,
        capsule_request=None,
    )


__all__ = [
    "ConfirmatoryBridgeResult",
    "ConfirmatoryRestorationVerifier",
    "ConfirmatoryRunnerDependencies",
    "ConfirmatoryStageFinalizer",
    "ConfirmatoryStudyIntegrityError",
    "ConfirmatoryStudyRunnerError",
    "bridge_pannuke_confirmatory_inputs",
    "execute_confirmatory_study",
    "finalize_confirmatory_stage",
    "finalize_resource_bounded_analysis",
    "run_confirmatory_frozen_feature_oof",
]
