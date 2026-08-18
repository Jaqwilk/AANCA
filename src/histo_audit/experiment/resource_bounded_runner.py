"""Fail-closed runner for the amended resource-bounded sensitivity analysis.

This execution path is intentionally separate from the original confirmatory
runner.  It can reuse only explicitly allowlisted CNN checkpoints from one
explicit predecessor, and every successful or failed run remains permanently
``amended_or_exploratory`` with no completion stage and no M9 authority.
"""

from __future__ import annotations

import json
import shutil
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from histo_audit.config import config_sha256, load_config
from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.experiment.confirmatory_cli_inputs import (
    ConfirmatoryCLIInputs,
    resolve_resource_bounded_cli_inputs,
)
from histo_audit.experiment.confirmatory_completion import (
    RESOURCE_BOUNDED_CAPACITY_POLICY,
    RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
    ConfirmatoryFilesystemReadback,
    build_confirmatory_completion_evidence,
    read_confirmatory_run_directory,
)
from histo_audit.experiment.confirmatory_core import (
    ConfirmatoryExecutionControls,
    ConfirmatoryMatrixArtifacts,
    confirmatory_cnn_config_for_cell,
    confirmatory_execution_controls_from_frozen_config,
    execute_confirmatory_matrix,
)
from histo_audit.experiment.confirmatory_memory_workspace import (
    ConfirmatoryMemoryWorkspace,
    ConfirmatoryWorkspaceArraySpec,
    ConfirmatoryWorkspaceIndexSpec,
    build_confirmatory_memory_workspace,
    close_and_cleanup_confirmatory_memory_workspace,
    verify_confirmatory_memory_workspace,
)
from histo_audit.experiment.confirmatory_runner import (
    ConfirmatoryBridgeResult,
    ConfirmatoryStudyRunnerError,
    _confirmatory_cnn_preflight_fingerprints,
    _resolve,
    _validate_core_artifacts,
    _validate_final_readback,
    _validate_restoration_source_binding,
    _verify_cache_files,
    bridge_pannuke_confirmatory_inputs,
    finalize_resource_bounded_analysis,
    run_confirmatory_frozen_feature_oof,
)
from histo_audit.experiment.confirmatory_statistics import (
    ConfirmatoryStatisticsArtifacts,
    ConfirmatoryStatisticsVerification,
    aggregate_confirmatory_statistics,
    verify_confirmatory_statistics_artifacts,
)
from histo_audit.experiment.pannuke_confirmatory_inputs import (
    PanNukeConfirmatoryInputs,
    derive_pannuke_confirmatory_workspace_array_specs,
    derive_pannuke_confirmatory_workspace_index_specs,
    load_pannuke_confirmatory_inputs,
)
from histo_audit.experiment.resource_bounded_checkpoint_execution import (
    ResourceBoundedCheckpointExecutionPreparation,
    prepare_fresh_resource_bounded_checkpoint_execution,
    prepare_successor_resource_bounded_checkpoint_execution,
)
from histo_audit.experiment.resource_bounded_resume import (
    RESOURCE_BOUNDED_RESUME_EVIDENCE_HASH_FIELD,
    ReadOnlyPredecessorSnapshot,
    ResourceBoundedResumeCopyReceipt,
    ResumeCheckpointExpectation,
    copy_validated_resume_checkpoints,
    inspect_read_only_resume_predecessor,
)
from histo_audit.experiment.study_contracts import (
    RESOURCE_BOUNDED_CONFIRMATORY_CONFIG_SHA256,
    RESOURCE_BOUNDED_CONFIRMATORY_PROFILE,
    ConfirmatoryMatrixPlan,
    build_confirmatory_matrix_plan,
    validate_resource_bounded_confirmatory_config,
)
from histo_audit.models.cnn import (
    OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER,
    validate_confirmatory_checkpoint_artifact,
)
from histo_audit.representations.imagenet import official_resnet18_weight_cache_path
from histo_audit.utils.run_tracking import (
    RUN_DISPOSITION_REGISTRY_FILENAME,
    RUN_STAGE_ATTESTATION_REGISTRY_FILENAME,
    IntegrityVerification,
    RunTracker,
    guard_run_stage_eligibility,
    read_run_dispositions,
    read_run_stage_attestations,
    sha256_file,
    verify_run_integrity,
)
from histo_audit.workflows.preregistration_amendment import (
    validate_resource_bounded_capacity_v3,
)
from histo_audit.workflows.study_gates import (
    ResourceBoundedExecutionGateEvidence,
)

RESOURCE_BOUNDED_ANALYSIS_DISPOSITION = "amended_or_exploratory"
RESOURCE_BOUNDED_EXPERIMENT_NAME = "pannuke_resource_bounded_confirmatory"
_RESOURCE_COMPUTE_PHASES = {
    "guarded_before_data_loading": "minimum_available_ram_bytes_before_data",
    "guarded_immediately_before_tracker": ("minimum_available_ram_bytes_before_tracker"),
}
_RESOURCE_COMPUTE_OBSERVATION_FIELDS = frozenset(
    {
        "total_host_ram_bytes",
        "available_host_ram_bytes",
        "cuda_available",
        "cuda_device_count",
        "selected_cuda_device_index",
        "selected_cuda_device_name",
        "total_vram_bytes",
        "free_vram_bytes",
        "cudnn_available",
        "amp_available",
        "amp_dtype",
        "weight_identifier",
        "weight_path",
        "weight_present",
        "weight_sha256",
        "smoke_attempted",
        "smoke_completed",
        "smoke_input_shape",
        "smoke_forward_finite",
        "smoke_backward_finite",
        "smoke_peak_allocated_bytes",
        "smoke_error",
    }
)
_RESOURCE_PROGRESS_FIELDS = frozenset(
    {
        "cell_id",
        "scenario",
        "fold",
        "corruption",
        "seed",
        "status",
        "timestamp",
        "telemetry_contract",
        "prohibited_for_selection_tuning",
        "adaptive_execution_changes_allowed",
    }
)
_RESOURCE_PROGRESS_STATUSES = frozenset({"started", "model_completed", "failed", "skipped"})
_HEX = frozenset("0123456789abcdef")


class ResourceBoundedStudyRunnerError(ConfirmatoryStudyRunnerError):
    """The amended sensitivity failed without creating any scientific stage claim."""


class ResourceBoundedStudyIntegrityError(ResourceBoundedStudyRunnerError):
    """A sealed sensitivity run failed exact non-claiming readback."""


class _DiskUsage(Protocol):
    free: int


def _default_disk_usage(path: Path) -> _DiskUsage:
    return cast(_DiskUsage, shutil.disk_usage(path))


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _default_compute_probe() -> Mapping[str, Any]:
    import psutil  # type: ignore[import-untyped]
    import torch
    from torchvision.models import resnet18  # type: ignore[import-untyped]

    policy = RESOURCE_BOUNDED_CAPACITY_POLICY
    memory = psutil.virtual_memory()
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    selected_device = cast(int, policy["cuda_device_index"])
    free_vram: int | None = None
    total_vram: int | None = None
    device_name: str | None = None
    cudnn_available = bool(
        cuda_available and torch.backends.cudnn.is_available() and torch.backends.cudnn.enabled
    )
    amp_available = bool(cuda_available and hasattr(torch, "autocast"))
    if cuda_available and device_count > selected_device:
        free_value, total_value = torch.cuda.mem_get_info(selected_device)
        free_vram = int(free_value)
        total_vram = int(total_value)
        device_name = str(torch.cuda.get_device_name(selected_device))
    weight_path = official_resnet18_weight_cache_path(OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER).resolve()
    smoke_attempted = False
    smoke_completed = False
    smoke_forward_finite = False
    smoke_backward_finite = False
    smoke_peak_allocated_bytes: int | None = None
    smoke_error: str | None = None
    if (
        cuda_available
        and device_count > selected_device
        and cudnn_available
        and amp_available
        and weight_path.is_file()
    ):
        smoke_attempted = True
        model: Any = None
        state_dict: Any = None
        inputs: Any = None
        outputs: Any = None
        loss: Any = None
        gradients: Any = None
        try:
            device = torch.device("cuda", selected_device)
            model = resnet18(weights=None)
            state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            model.train()
            model.to(device)
            torch.cuda.reset_peak_memory_stats(device)
            inputs = torch.ones(
                tuple(cast(list[int], policy["cuda_smoke_input_shape"])),
                device=device,
                dtype=torch.float32,
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):
                outputs = model(inputs)
                loss = outputs.float().square().mean()
            smoke_forward_finite = bool(
                torch.isfinite(outputs).all().item() and torch.isfinite(loss).item()
            )
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            smoke_backward_finite = bool(
                gradients and all(torch.isfinite(gradient).all().item() for gradient in gradients)
            )
            torch.cuda.synchronize(device)
            smoke_peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
            smoke_completed = True
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            smoke_error = f"{type(exc).__name__}: {exc}"
        finally:
            gradients = None
            loss = None
            outputs = None
            inputs = None
            model = None
            state_dict = None
            torch.cuda.empty_cache()
    return {
        "total_host_ram_bytes": int(memory.total),
        "available_host_ram_bytes": int(memory.available),
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "selected_cuda_device_index": selected_device,
        "selected_cuda_device_name": device_name,
        "free_vram_bytes": free_vram,
        "total_vram_bytes": total_vram,
        "cudnn_available": cudnn_available,
        "amp_available": amp_available,
        "amp_dtype": "float16",
        "weight_identifier": OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER,
        "weight_path": str(weight_path),
        "weight_present": weight_path.is_file(),
        "weight_sha256": sha256_file(weight_path) if weight_path.is_file() else None,
        "smoke_attempted": smoke_attempted,
        "smoke_completed": smoke_completed,
        "smoke_input_shape": list(cast(Sequence[int], policy["cuda_smoke_input_shape"])),
        "smoke_forward_finite": smoke_forward_finite,
        "smoke_backward_finite": smoke_backward_finite,
        "smoke_peak_allocated_bytes": smoke_peak_allocated_bytes,
        "smoke_error": smoke_error,
    }


@dataclass(frozen=True, slots=True)
class ResourceCapacityEvidence:
    """One read-only disk-capacity observation against authority C's exact policy."""

    phase: str
    probe_path: Path
    free_bytes: int
    minimum_free_bytes: int
    policy_sha256: str
    checked_at_utc: str
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "probe_path": str(self.probe_path),
            "free_bytes": self.free_bytes,
            "minimum_free_bytes": self.minimum_free_bytes,
            "policy_sha256": self.policy_sha256,
            "checked_at_utc": self.checked_at_utc,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class ResourceComputeEvidence:
    """One outcome-blind compute observation against authority C's exact policy."""

    phase: str
    minimum_available_ram_bytes: int
    policy_sha256: str
    observation: Mapping[str, Any]
    observation_sha256: str
    checked_at_utc: str
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "phase": self.phase,
            "minimum_available_ram_bytes": self.minimum_available_ram_bytes,
            "policy_sha256": self.policy_sha256,
            "observation": dict(self.observation),
            "observation_sha256": self.observation_sha256,
            "checked_at_utc": self.checked_at_utc,
            "passed": self.passed,
            "outcome_values_read": False,
            "prohibited_for_selection_tuning": True,
            "adaptive_execution_changes_allowed": False,
        }

    @property
    def evidence_sha256(self) -> str:
        return canonical_sha256(self.as_dict())


def _default_resource_gate_validator(**kwargs: Any) -> ResourceBoundedExecutionGateEvidence:
    # Kept late-bound so the gate and runner can be reviewed independently.
    from histo_audit.workflows.study_gates import (
        validate_resource_bounded_execution_gate,
    )

    return validate_resource_bounded_execution_gate(**kwargs)


def _default_lifecycle_validator(**kwargs: Any) -> Any:
    from histo_audit.workflows.lifecycle_qualification import (
        require_current_lifecycle_readiness,
    )

    return require_current_lifecycle_readiness(**kwargs)


@dataclass(frozen=True, slots=True)
class ResourceBoundedRunnerDependencies:
    """Injectable non-claiming boundaries used by structural runner tests."""

    gate_validator: Callable[..., ResourceBoundedExecutionGateEvidence] = (
        _default_resource_gate_validator
    )
    lifecycle_validator: Callable[..., Any] = _default_lifecycle_validator
    config_loader: Callable[[str | Path], dict[str, Any]] = load_config
    plan_builder: Callable[[Mapping[str, Any]], ConfirmatoryMatrixPlan] = (
        build_confirmatory_matrix_plan
    )
    controls_builder: Callable[[Mapping[str, Any]], ConfirmatoryExecutionControls] = (
        confirmatory_execution_controls_from_frozen_config
    )
    input_builder: Callable[..., PanNukeConfirmatoryInputs] = load_pannuke_confirmatory_inputs
    input_resolver: Callable[..., ConfirmatoryCLIInputs] = resolve_resource_bounded_cli_inputs
    workspace_array_spec_builder: Callable[..., tuple[ConfirmatoryWorkspaceArraySpec, ...]] = (
        derive_pannuke_confirmatory_workspace_array_specs
    )
    workspace_index_spec_builder: Callable[..., tuple[ConfirmatoryWorkspaceIndexSpec, ...]] = (
        derive_pannuke_confirmatory_workspace_index_specs
    )
    workspace_builder: Callable[..., ConfirmatoryMemoryWorkspace] = (
        build_confirmatory_memory_workspace
    )
    workspace_verifier: Callable[..., ConfirmatoryMemoryWorkspace] = (
        verify_confirmatory_memory_workspace
    )
    workspace_cleaner: Callable[..., dict[str, Any]] = (
        close_and_cleanup_confirmatory_memory_workspace
    )
    bridge_builder: Callable[..., ConfirmatoryBridgeResult] = lambda prepared, controls, **kwargs: (
        bridge_pannuke_confirmatory_inputs(prepared, controls, **kwargs)
    )
    matrix_executor: Callable[..., ConfirmatoryMatrixArtifacts] = execute_confirmatory_matrix
    statistics_aggregator: Callable[
        [str | Path, ConfirmatoryExecutionControls], ConfirmatoryStatisticsArtifacts
    ] = aggregate_confirmatory_statistics
    statistics_verifier: Callable[
        [str | Path, ConfirmatoryExecutionControls],
        ConfirmatoryStatisticsVerification,
    ] = verify_confirmatory_statistics_artifacts
    filesystem_reader: Callable[..., ConfirmatoryFilesystemReadback] = (
        read_confirmatory_run_directory
    )
    completion_builder: Callable[..., dict[str, Any]] = build_confirmatory_completion_evidence
    tracker_starter: Callable[..., RunTracker] = RunTracker.start
    integrity_verifier: Callable[[str | Path], IntegrityVerification] = verify_run_integrity
    predecessor_inspector: Callable[..., ReadOnlyPredecessorSnapshot] = (
        inspect_read_only_resume_predecessor
    )
    checkpoint_copier: Callable[..., ResourceBoundedResumeCopyReceipt] = (
        copy_validated_resume_checkpoints
    )
    fresh_checkpoint_execution_preparer: Callable[
        ...,
        ResourceBoundedCheckpointExecutionPreparation,
    ] = prepare_fresh_resource_bounded_checkpoint_execution
    successor_checkpoint_execution_preparer: Callable[
        ...,
        ResourceBoundedCheckpointExecutionPreparation,
    ] = prepare_successor_resource_bounded_checkpoint_execution
    checkpoint_validator: Callable[..., None] = validate_confirmatory_checkpoint_artifact
    disk_usage: Callable[[Path], _DiskUsage] = _default_disk_usage
    clock: Callable[[], str] = _utc_now_iso
    compute_probe: Callable[[], Mapping[str, Any]] = _default_compute_probe


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value).issubset(_HEX)


def _safe_run_id(value: object, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or Path(value).name != value
        or value in {".", ".."}
    ):
        raise ResourceBoundedStudyRunnerError(f"{role} must be one safe non-empty run ID")
    return value


def _validate_run_mode(
    *,
    run_mode: str,
    checkpoint_predecessor_run_directory: str | Path | None,
    retry_of_run_id: str | None,
    run_id: str | None,
) -> Literal["fresh", "successor_resume"]:
    if run_mode not in {"fresh", "successor_resume"}:
        raise ResourceBoundedStudyRunnerError(
            "run_mode must be explicitly 'fresh' or 'successor_resume'"
        )
    if run_mode == "fresh":
        if checkpoint_predecessor_run_directory is not None or retry_of_run_id is not None:
            raise ResourceBoundedStudyRunnerError(
                "fresh execution forbids predecessor and retry bindings"
            )
        return "fresh"
    if checkpoint_predecessor_run_directory is None or retry_of_run_id is None:
        raise ResourceBoundedStudyRunnerError(
            "successor_resume requires one explicit predecessor and retry_of_run_id"
        )
    retry = _safe_run_id(retry_of_run_id, "retry_of_run_id")
    predecessor = Path(checkpoint_predecessor_run_directory).resolve()
    if predecessor.name != retry:
        raise ResourceBoundedStudyRunnerError(
            "retry_of_run_id must exactly equal the explicit predecessor directory name"
        )
    if run_id is not None and run_id == retry:
        raise ResourceBoundedStudyRunnerError(
            "a successor must use a new run ID distinct from its predecessor"
        )
    return "successor_resume"


def _require_reusable_successor_checkpoint(
    snapshot: ReadOnlyPredecessorSnapshot,
) -> None:
    if not snapshot.resume_records:
        raise ResourceBoundedStudyRunnerError(
            "successor_resume requires at least one validated reusable checkpoint; "
            "zero reuse is not semantically a resume"
        )


def _exact_capacity_policy(
    value: Mapping[str, Any],
    *,
    resource_input_workspace_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    observed = dict(value)
    if resource_input_workspace_plan is None:
        if observed == RESOURCE_BOUNDED_CAPACITY_POLICY:
            return observed
        raise ResourceBoundedStudyRunnerError(
            "resource capacity policy must match the exact schema-v2 frozen policy"
        )
    if resource_input_workspace_plan is not None:
        try:
            canonical_capacity, _ = validate_resource_bounded_capacity_v3(
                observed,
                resource_input_workspace_plan,
            )
        except (TypeError, ValueError) as error:
            raise ResourceBoundedStudyRunnerError(
                "effective authority D capacity/workspace contract is invalid"
            ) from error
        return canonical_capacity
    raise AssertionError("unreachable capacity-policy branch")


def _exact_capacity_v3_authority(
    gate: ResourceBoundedExecutionGateEvidence,
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = gate.execution_authority
    raw_plan = authority.resource_input_workspace_plan
    if not isinstance(raw_plan, Mapping):
        raise ResourceBoundedStudyRunnerError(
            "resource-bounded execution requires effective authority D with a typed "
            "input-workspace plan"
        )
    try:
        capacity, plan = validate_resource_bounded_capacity_v3(
            authority.resource_capacity_policy,
            raw_plan,
        )
    except (TypeError, ValueError) as error:
        raise ResourceBoundedStudyRunnerError(
            "effective authority D capacity/workspace contract is invalid"
        ) from error
    if authority.resource_input_workspace_plan_sha256 != plan["plan_without_self_hash_sha256"]:
        raise ResourceBoundedStudyRunnerError(
            "effective authority D workspace-plan SHA-256 is invalid"
        )
    return capacity, plan


def _nearest_existing_capacity_probe(target: Path) -> Path:
    probe = target.resolve()
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            raise ResourceBoundedStudyRunnerError(
                "no existing ancestor is available for disk-capacity measurement"
            )
        probe = parent
    return probe if probe.is_dir() else probe.parent


def require_resource_capacity(
    target: str | Path,
    *,
    capacity_policy: Mapping[str, Any],
    phase: str,
    resource_input_workspace_plan: Mapping[str, Any] | None = None,
    disk_usage: Callable[[Path], _DiskUsage] = _default_disk_usage,
    clock: Callable[[], str] = _utc_now_iso,
) -> ResourceCapacityEvidence:
    """Require the exact C/v2 or phase-specific D/v3 disk threshold."""

    policy = _exact_capacity_policy(
        capacity_policy,
        resource_input_workspace_plan=resource_input_workspace_plan,
    )
    if not isinstance(phase, str) or not phase.strip():
        raise ValueError("capacity phase must be non-empty")
    probe = _nearest_existing_capacity_probe(Path(target))
    usage = disk_usage(probe)
    free = getattr(usage, "free", None)
    if type(free) is not int or free < 0:
        raise ResourceBoundedStudyRunnerError(
            "disk-capacity provider returned an invalid free-byte count"
        )
    minimum_field: str
    if resource_input_workspace_plan is None:
        minimum_field = "minimum_free_bytes_before_tracker"
    else:
        phase_fields = {
            "guarded_before_workspace_build": ("minimum_free_bytes_before_workspace_build"),
            "guarded_immediately_before_tracker": ("minimum_free_bytes_before_tracker"),
        }
        selected_minimum_field = phase_fields.get(phase)
        if selected_minimum_field is None:
            raise ValueError(
                "capacity-v3 phase must be guarded_before_workspace_build or "
                "guarded_immediately_before_tracker"
            )
        minimum_field = selected_minimum_field
    minimum = int(policy[minimum_field])
    evidence = ResourceCapacityEvidence(
        phase=phase,
        probe_path=probe,
        free_bytes=free,
        minimum_free_bytes=minimum,
        policy_sha256=canonical_sha256(policy),
        checked_at_utc=clock(),
        passed=free >= minimum,
    )
    if not evidence.passed:
        if resource_input_workspace_plan is None:
            raise ResourceBoundedStudyRunnerError(
                "resource-bounded run lacks the fixed 22-GiB free-space threshold: "
                f"free={free}, required={minimum}"
            )
        raise ResourceBoundedStudyRunnerError(
            "resource-bounded run lacks its fixed phase-specific disk capacity: "
            f"free={free}, required={minimum}"
        )
    return evidence


def require_resource_compute(
    *,
    phase: str,
    capacity_policy: Mapping[str, Any],
    resource_input_workspace_plan: Mapping[str, Any] | None = None,
    probe: Callable[[], Mapping[str, Any]] = _default_compute_probe,
    clock: Callable[[], str] = _utc_now_iso,
) -> ResourceComputeEvidence:
    """Require the exact C/v2 or D/v3 RAM/CUDA/AMP/smoke policy."""

    policy = _exact_capacity_policy(
        capacity_policy,
        resource_input_workspace_plan=resource_input_workspace_plan,
    )
    available_policy_field = _RESOURCE_COMPUTE_PHASES.get(phase)
    if available_policy_field is None:
        raise ValueError(
            "compute phase must be exactly guarded_before_data_loading or "
            "guarded_immediately_before_tracker"
        )
    observed = dict(probe())
    if set(observed) != _RESOURCE_COMPUTE_OBSERVATION_FIELDS:
        raise ResourceBoundedStudyRunnerError(
            "compute provider returned a non-contract observation"
        )
    integer_fields = (
        "total_host_ram_bytes",
        "available_host_ram_bytes",
        "cuda_device_count",
        "selected_cuda_device_index",
        "free_vram_bytes",
        "total_vram_bytes",
        "smoke_peak_allocated_bytes",
    )
    if any(
        type(observed.get(field)) is not int or cast(int, observed[field]) < 0
        for field in integer_fields
    ):
        raise ResourceBoundedStudyRunnerError(
            "compute provider returned invalid RAM/CUDA/VRAM/smoke integer evidence"
        )
    boolean_fields = (
        "cuda_available",
        "cudnn_available",
        "amp_available",
        "weight_present",
        "smoke_attempted",
        "smoke_completed",
        "smoke_forward_finite",
        "smoke_backward_finite",
    )
    if any(type(observed.get(field)) is not bool for field in boolean_fields):
        raise ResourceBoundedStudyRunnerError("compute provider returned invalid boolean evidence")
    device_name = observed.get("selected_cuda_device_name")
    if not isinstance(device_name, str) or not device_name.strip():
        raise ResourceBoundedStudyRunnerError(
            "compute provider returned an invalid selected CUDA device name"
        )
    weight_path_value = observed.get("weight_path")
    if not isinstance(weight_path_value, str) or not Path(weight_path_value).is_absolute():
        raise ResourceBoundedStudyRunnerError(
            "compute provider returned an invalid official-weight path"
        )
    weight_path = Path(weight_path_value)
    try:
        weight_stat = weight_path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ResourceBoundedStudyRunnerError(
            "official ResNet-18 weight file is unavailable"
        ) from exc
    attributes = int(getattr(weight_stat, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if (
        not stat.S_ISREG(weight_stat.st_mode)
        or weight_path.is_symlink()
        or bool(attributes & reparse_flag)
    ):
        raise ResourceBoundedStudyRunnerError(
            "official ResNet-18 weight must be a regular physical file"
        )
    minimum_available = cast(int, policy[available_policy_field])
    total_host_ram = cast(int, observed["total_host_ram_bytes"])
    available_host_ram = cast(int, observed["available_host_ram_bytes"])
    cuda_device_count = cast(int, observed["cuda_device_count"])
    total_vram = cast(int, observed["total_vram_bytes"])
    free_vram = cast(int, observed["free_vram_bytes"])
    smoke_peak = cast(int, observed["smoke_peak_allocated_bytes"])
    observation_sha256 = canonical_sha256(observed)
    passed = bool(
        total_host_ram >= policy["minimum_total_ram_bytes"]
        and available_host_ram >= minimum_available
        and observed.get("cuda_available") is policy["cuda_required"]
        and cuda_device_count >= 1
        and observed.get("selected_cuda_device_index") == policy["cuda_device_index"]
        and total_vram >= policy["minimum_total_vram_bytes"]
        and free_vram >= policy["minimum_free_vram_bytes"]
        and observed.get("cudnn_available") is policy["cudnn_required"]
        and observed.get("amp_available") is policy["amp_required"]
        and observed.get("amp_dtype") == policy["amp_dtype"]
        and observed.get("smoke_attempted") is True
        and observed.get("smoke_completed") is True
        and observed.get("smoke_input_shape") == policy["cuda_smoke_input_shape"]
        and observed.get("smoke_forward_finite") is True
        and observed.get("smoke_backward_finite") is True
        and smoke_peak <= policy["cuda_smoke_max_peak_allocated_bytes"]
        and observed.get("smoke_error") is None
        and observed.get("weight_identifier") == policy["official_weight_identifier"]
        and observed.get("weight_present") is True
        and observed.get("weight_sha256") == policy["official_weight_sha256"]
        and sha256_file(weight_path) == policy["official_weight_sha256"]
        and policy["implicit_weight_download_allowed"] is False
    )
    evidence = ResourceComputeEvidence(
        phase=phase,
        minimum_available_ram_bytes=minimum_available,
        policy_sha256=canonical_sha256(policy),
        observation=observed,
        observation_sha256=observation_sha256,
        checked_at_utc=clock(),
        passed=passed,
    )
    if not passed:
        raise ResourceBoundedStudyRunnerError(
            "resource compute preflight failed authority C's fixed "
            "RAM/CUDA/cuDNN/AMP/smoke/weight requirements"
        )
    return evidence


def _resource_compute_document_readback(
    document: Mapping[str, Any],
    *,
    capacity_policy: Mapping[str, Any],
    resource_input_workspace_plan: Mapping[str, Any] | None = None,
) -> str:
    """Verify the complete two-check carrier without probing compute again."""

    policy = _exact_capacity_policy(
        capacity_policy,
        resource_input_workspace_plan=resource_input_workspace_plan,
    )
    expected_policy_sha256 = canonical_sha256(policy)
    expected_top_fields = {
        "schema_version",
        "policy",
        "policy_sha256",
        "checks",
        "check_sha256s",
        "passed",
        "outcome_values_read",
        "prohibited_for_selection_tuning",
        "adaptive_execution_changes_allowed",
        "evidence_without_self_hash_sha256",
    }
    if (
        set(document) != expected_top_fields
        or document.get("schema_version") != 1
        or document.get("policy") != policy
        or document.get("policy_sha256") != expected_policy_sha256
        or document.get("passed") is not True
        or document.get("outcome_values_read") is not False
        or document.get("prohibited_for_selection_tuning") is not True
        or document.get("adaptive_execution_changes_allowed") is not False
    ):
        raise ResourceBoundedStudyRunnerError(
            "resource compute evidence document differs from authority C"
        )
    checks = document.get("checks")
    check_sha256s = document.get("check_sha256s")
    if (
        not isinstance(checks, list)
        or not isinstance(check_sha256s, list)
        or len(checks) != 2
        or len(check_sha256s) != 2
    ):
        raise ResourceBoundedStudyRunnerError(
            "resource compute evidence document lacks the exact two checks"
        )
    expected_check_fields = {
        "schema_version",
        "phase",
        "minimum_available_ram_bytes",
        "policy_sha256",
        "observation",
        "observation_sha256",
        "checked_at_utc",
        "passed",
        "outcome_values_read",
        "prohibited_for_selection_tuning",
        "adaptive_execution_changes_allowed",
    }
    for index, (raw_check, expected_phase) in enumerate(
        zip(checks, _RESOURCE_COMPUTE_PHASES, strict=True)
    ):
        if not isinstance(raw_check, Mapping):
            raise ResourceBoundedStudyRunnerError("resource compute check must be a mapping")
        check = dict(raw_check)
        observation = check.get("observation")
        minimum_field = _RESOURCE_COMPUTE_PHASES[expected_phase]
        if (
            set(check) != expected_check_fields
            or check.get("schema_version") != 1
            or check.get("phase") != expected_phase
            or check.get("minimum_available_ram_bytes") != policy[minimum_field]
            or check.get("policy_sha256") != expected_policy_sha256
            or check.get("passed") is not True
            or check.get("outcome_values_read") is not False
            or check.get("prohibited_for_selection_tuning") is not True
            or check.get("adaptive_execution_changes_allowed") is not False
            or not isinstance(check.get("checked_at_utc"), str)
            or not cast(str, check["checked_at_utc"]).strip()
            or not isinstance(observation, Mapping)
            or set(observation) != _RESOURCE_COMPUTE_OBSERVATION_FIELDS
            or check.get("observation_sha256")
            != canonical_sha256(dict(cast(Mapping[str, Any], observation)))
            or check_sha256s[index] != canonical_sha256(check)
        ):
            raise ResourceBoundedStudyRunnerError(
                "resource compute check failed exact carrier readback"
            )
    unsigned = dict(document)
    saved_sha256 = unsigned.pop("evidence_without_self_hash_sha256")
    if not _valid_sha(saved_sha256) or saved_sha256 != canonical_sha256(unsigned):
        raise ResourceBoundedStudyRunnerError(
            "resource compute evidence self-excluding hash is invalid"
        )
    return str(saved_sha256)


def _resource_compute_evidence_document(
    *,
    capacity_policy: Mapping[str, Any],
    resource_input_workspace_plan: Mapping[str, Any] | None = None,
    checks: Sequence[ResourceComputeEvidence],
) -> dict[str, Any]:
    """Validate and self-bind the exact two authority-bound compute observations."""

    policy = _exact_capacity_policy(
        capacity_policy,
        resource_input_workspace_plan=resource_input_workspace_plan,
    )
    expected_policy_sha256 = canonical_sha256(policy)
    expected_phases = tuple(_RESOURCE_COMPUTE_PHASES)
    if len(checks) != 2 or tuple(check.phase for check in checks) != expected_phases:
        raise ResourceBoundedStudyRunnerError(
            "resource compute evidence must contain the exact two ordered phases"
        )
    for check in checks:
        if not isinstance(check, ResourceComputeEvidence):
            raise ResourceBoundedStudyRunnerError("resource compute evidence carrier must be typed")
        payload = check.as_dict()
        if (
            check.passed is not True
            or check.policy_sha256 != expected_policy_sha256
            or check.observation_sha256 != canonical_sha256(dict(check.observation))
            or payload.get("outcome_values_read") is not False
            or payload.get("prohibited_for_selection_tuning") is not True
            or payload.get("adaptive_execution_changes_allowed") is not False
        ):
            raise ResourceBoundedStudyRunnerError(
                "resource compute evidence carrier failed exact typed readback"
            )
    document = {
        "schema_version": 1,
        "policy": policy,
        "policy_sha256": expected_policy_sha256,
        "checks": [check.as_dict() for check in checks],
        "check_sha256s": [check.evidence_sha256 for check in checks],
        "passed": True,
        "outcome_values_read": False,
        "prohibited_for_selection_tuning": True,
        "adaptive_execution_changes_allowed": False,
    }
    document["evidence_without_self_hash_sha256"] = canonical_sha256(document)
    _resource_compute_document_readback(
        document,
        capacity_policy=policy,
        resource_input_workspace_plan=resource_input_workspace_plan,
    )
    return document


def _validate_resource_gate(
    gate: ResourceBoundedExecutionGateEvidence,
    *,
    primary_run: Path,
    authority_directory: Path,
) -> None:
    if not isinstance(gate, ResourceBoundedExecutionGateEvidence):
        raise TypeError("gate_evidence must be typed ResourceBoundedExecutionGateEvidence")
    authority = gate.execution_authority
    if (
        gate.historical_primary.primary_run_directory.resolve() != primary_run
        or authority.authority_directory.resolve() != authority_directory
        or authority.resource_profile_id != RESOURCE_BOUNDED_CONFIRMATORY_PROFILE
        or authority.resource_confirmatory_config_semantic_sha256
        != RESOURCE_BOUNDED_CONFIRMATORY_CONFIG_SHA256
        or gate.analysis_disposition != RESOURCE_BOUNDED_ANALYSIS_DISPOSITION
        or gate.outcomes_inspected is not True
        or gate.original_confirmatory_claim_allowed is not False
        or gate.study_outcome_eligible is not False
        or gate.completion_stage is not None
        or gate.primary_rebinding_allowed is not False
        or gate.primary_mutation_allowed is not False
    ):
        raise ResourceBoundedStudyRunnerError(
            "resource-bounded gate lacks the exact permanent non-claiming P+C authority"
        )
    _exact_capacity_policy(
        authority.resource_capacity_policy,
        resource_input_workspace_plan=authority.resource_input_workspace_plan,
    )


def _plain_physical_directory(path: Path, *, role: str) -> Path:
    try:
        observed = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ResourceBoundedStudyRunnerError(f"{role} is unavailable") from exc
    attributes = int(getattr(observed, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if not stat.S_ISDIR(observed.st_mode) or path.is_symlink() or bool(attributes & reparse_flag):
        raise ResourceBoundedStudyRunnerError(f"{role} must be a plain physical directory")
    return path


def _read_strict_physical_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        observed = path.stat(follow_symlinks=False)
        attributes = int(getattr(observed, "st_file_attributes", 0))
        reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        if (
            not stat.S_ISREG(observed.st_mode)
            or path.is_symlink()
            or bool(attributes & reparse_flag)
        ):
            raise ValueError(f"{role} is not a regular physical file")

        def reject_constant(value: str) -> Any:
            raise ValueError(f"non-finite JSON constant: {value}")

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResourceBoundedStudyRunnerError(f"{role} is unavailable or invalid") from exc
    if not isinstance(payload, dict):
        raise ResourceBoundedStudyRunnerError(f"{role} must be one JSON object")
    return payload


def _require_gate_equality(
    expected: ResourceBoundedExecutionGateEvidence,
    observed: ResourceBoundedExecutionGateEvidence,
) -> None:
    if not isinstance(observed, ResourceBoundedExecutionGateEvidence) or observed != expected:
        raise ResourceBoundedStudyRunnerError(
            "resource-bounded P+C gate changed during mandatory live revalidation"
        )


def _resource_model_metadata(configuration: Mapping[str, Any]) -> dict[str, Any]:
    weight_identifier = configuration.get("weight_identifier")
    if weight_identifier != OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER:
        raise ResourceBoundedStudyRunnerError(
            "resource-bounded checkpoints require the official frozen ImageNet weight ID"
        )
    weight_path = official_resnet18_weight_cache_path(str(weight_identifier)).resolve()
    if not weight_path.is_file():
        raise ResourceBoundedStudyRunnerError(
            "official ResNet-18 weights are absent; implicit download is forbidden"
        )
    input_variant = configuration.get("input_variant")
    if input_variant not in {
        "context_rgb",
        "context_rgb_plus_binary_target_mask",
    }:
        raise ResourceBoundedStudyRunnerError("resource-bounded CNN input variant is unsupported")
    uses_target_mask = input_variant == "context_rgb_plus_binary_target_mask"
    return {
        "architecture": "torchvision.resnet18",
        "class_order": [0, 1, 2, 3, 4],
        "input_channels": 4 if uses_target_mask else 3,
        "weight_identifier": weight_identifier,
        "weight_path": str(weight_path),
        "weight_sha256": sha256_file(weight_path),
        "implicit_weight_download": False,
        "preprocessing": {
            "rgb_resize": "bilinear_antialias",
            "rgb_range_before_normalisation": [0.0, 1.0],
            "rgb_mean": [0.485, 0.456, 0.406],
            "rgb_standard_deviation": [0.229, 0.224, 0.225],
            "target_mask_resize": ("nearest_binary_unnormalised" if uses_target_mask else None),
        },
        "fourth_channel_initialisation": (
            configuration.get("fourth_channel_initialisation") if uses_target_mask else None
        ),
    }


def build_resource_checkpoint_allowlist(
    *,
    controls: ConfirmatoryExecutionControls,
    cnn_preflight_fingerprints: Mapping[
        str,
        Mapping[str, Mapping[str, str]],
    ],
) -> tuple[ResumeCheckpointExpectation, ...]:
    """Build the exact 30 canonical CNN-fold checkpoint expectations."""

    output: list[ResumeCheckpointExpectation] = []
    expected_fold_ids = {str(value) for value in range(controls.n_splits)}
    for cell in controls.plan.cells:
        scenario = controls.scenarios_by_id[cell.scenario_id]
        if scenario.family != "cnn":
            continue
        fingerprints = cnn_preflight_fingerprints.get(cell.cell_id)
        if fingerprints is None or set(fingerprints) != expected_fold_ids:
            raise ResourceBoundedStudyRunnerError(
                f"CNN preflight fingerprints are incomplete for {cell.cell_id}"
            )
        for fold_id in range(controls.n_splits):
            config = confirmatory_cnn_config_for_cell(
                scenario,
                controls,
                model_seed=cell.model_seed + fold_id,
                cpu_test_only=False,
            )
            configuration = asdict(config)
            metadata = _resource_model_metadata(configuration)
            output.append(
                ResumeCheckpointExpectation(
                    relative_path=(f"cells/{cell.cell_id}/checkpoints/fold_{fold_id:02d}.pt"),
                    cell_id=cell.cell_id,
                    fold_id=fold_id,
                    expected_configuration=configuration,
                    expected_model_metadata=metadata,
                    expected_data_and_split_sha256=dict(fingerprints[str(fold_id)]),
                )
            )
    expected_count = cast(
        int,
        RESOURCE_BOUNDED_CAPACITY_POLICY["planned_cnn_fold_checkpoints"],
    )
    if len(output) != expected_count:
        raise ResourceBoundedStudyRunnerError(
            f"resource checkpoint allowlist has {len(output)} entries, expected {expected_count}"
        )
    return tuple(output)


def _log_resource_progress(
    tracker: RunTracker,
    event: Mapping[str, Any],
) -> None:
    payload = dict(event)
    if (
        set(payload) != _RESOURCE_PROGRESS_FIELDS
        or payload.get("status") not in _RESOURCE_PROGRESS_STATUSES
        or payload.get("telemetry_contract") != "outcome_value_free_operational_telemetry"
        or payload.get("prohibited_for_selection_tuning") is not True
        or payload.get("adaptive_execution_changes_allowed") is not False
    ):
        raise ResourceBoundedStudyRunnerError(
            "resource progress callback received a non-contract event"
        )
    tracker.log_event("resource_bounded_cell_progress", **payload)


def _resource_completion_candidate(
    candidate: Mapping[str, Any],
    *,
    plan: ConfirmatoryMatrixPlan,
    readback: ConfirmatoryFilesystemReadback,
    core_completion: Mapping[str, Any],
    gate: ResourceBoundedExecutionGateEvidence,
    run_id: str,
    retry_of_run_id: str | None,
    statistics: ConfirmatoryStatisticsVerification,
    resume_evidence_sha256: str,
    predecessor_qualification_sha256: str | None,
    capacity_policy_sha256: str,
    workspace_plan_sha256: str,
    workspace_plan_file_sha256: str,
    workspace_receipt_sha256: str,
    workspace_receipt_file_sha256: str,
    workspace_artifact_root_sha256: str,
    workspace_cleanup_sha256: str,
    workspace_cleanup_file_sha256: str,
    compute_evidence_sha256: str,
) -> dict[str, Any]:
    """Bind a completed sensitivity while preserving permanent scientific ineligibility."""

    result = dict(candidate)
    technical_matrix_eligible = bool(
        core_completion.get("model_matrix_execution_eligible") is True
        and _valid_sha(core_completion.get("matrix_execution_telemetry_sha256"))
        and readback.passed
    )
    result.update(
        completion_stage=None,
        study_outcome_eligible=False,
        valid_completion_claim=False,
        artifact_scope=RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
        analysis_disposition=RESOURCE_BOUNDED_ANALYSIS_DISPOSITION,
        original_confirmatory_claim_allowed=False,
        m9_unlock_allowed=False,
        model_matrix_execution_eligible=technical_matrix_eligible,
        matrix_execution_telemetry_sha256=(
            core_completion.get("matrix_execution_telemetry_sha256")
        ),
        matrix_execution_eligibility_source=(
            "real_per_cell_execution_telemetry_reconciliation_and_filesystem_readback"
        ),
        run_id=run_id,
        retry_of_run_id=retry_of_run_id,
        outcomes_inspected=True,
        historical_primary_run_id=gate.historical_primary.primary_run_id,
        historical_primary_artifact_root_sha256=(
            gate.historical_primary.primary_artifact_root_sha256
        ),
        resource_authorization_sha256=gate.execution_authority.authorization_sha256,
        resource_execution_source_root_sha256=(
            gate.execution_authority.resource_execution_source_root_sha256
        ),
        confirmatory_storage_policy_sha256=(
            gate.execution_authority.confirmatory_storage_policy_sha256
        ),
        statistics_sha256=statistics.statistics_sha256,
        bootstrap_evidence_sha256=statistics.bootstrap_evidence_sha256,
        resource_resume_evidence_sha256=resume_evidence_sha256,
        resource_predecessor_qualification_sha256=(predecessor_qualification_sha256),
        resource_capacity_policy_sha256=capacity_policy_sha256,
        resource_input_workspace_plan_sha256=workspace_plan_sha256,
        resource_input_workspace_plan_file_sha256=workspace_plan_file_sha256,
        resource_input_workspace_receipt_sha256=workspace_receipt_sha256,
        resource_input_workspace_receipt_file_sha256=(workspace_receipt_file_sha256),
        resource_input_workspace_artifact_root_sha256=(workspace_artifact_root_sha256),
        resource_input_workspace_cleanup_sha256=workspace_cleanup_sha256,
        resource_input_workspace_cleanup_file_sha256=(workspace_cleanup_file_sha256),
        resource_input_workspace_removed=True,
        resource_compute_evidence_sha256=compute_evidence_sha256,
        filesystem_readback_status=readback.status,
        filesystem_checked_artifact_count=readback.checked_artifact_count,
        filesystem_matrix_plan_sha256=readback.matrix_plan_sha256,
        filesystem_cell_index_sha256=readback.cell_index_sha256,
        filesystem_root_artifact_manifest_sha256=(readback.root_artifact_manifest_sha256),
        filesystem_confirmatory_storage_policy_sha256=(readback.confirmatory_storage_policy_sha256),
        filesystem_readback_sha256=canonical_sha256(readback.as_dict()),
        post_seal_integrity_verification_required=True,
        post_seal_stage_attestation_forbidden=True,
        full_pc_gate_validation_count=2,
    )
    if (
        result.get("schema_version") != 1
        or result.get("matrix_config_sha256") != plan.config_sha256
        or result.get("planned_cell_count") != len(plan.cells)
        or result.get("required_cell_count") != plan.required_cell_count
        or result.get("completed_required_cell_count") != plan.required_cell_count
        or result.get("failed_required_cell_count") != 0
        or result.get("reconciliation_status") != "passed"
        or result.get("fold_rotation_complete") is not True
        or result.get("completion_stage") is not None
        or result.get("study_outcome_eligible") is not False
        or result.get("valid_completion_claim") is not False
        or result.get("analysis_disposition") != RESOURCE_BOUNDED_ANALYSIS_DISPOSITION
        or result.get("original_confirmatory_claim_allowed") is not False
        or result.get("m9_unlock_allowed") is not False
        or result.get("model_matrix_execution_eligible") is not True
        or result.get("resource_input_workspace_removed") is not True
        or result.get("full_pc_gate_validation_count") != 2
        or not readback.passed
    ):
        raise ResourceBoundedStudyRunnerError(
            "resource completion builder did not preserve the exact non-claiming contract"
        )
    return result


def _demote_resource_run(
    tracker: RunTracker,
    error: BaseException,
    *,
    core_completion: Mapping[str, Any] | None,
    retry_of_run_id: str | None,
) -> None:
    fallback = dict(core_completion or {})
    fallback.update(
        schema_version=1,
        completion_stage=None,
        study_outcome_eligible=False,
        valid_completion_claim=False,
        artifact_scope=RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
        analysis_disposition=RESOURCE_BOUNDED_ANALYSIS_DISPOSITION,
        original_confirmatory_claim_allowed=False,
        m9_unlock_allowed=False,
        retry_of_run_id=retry_of_run_id,
        runner_failure=f"{type(error).__name__}: {error}",
    )
    tracker.write_json("completion_evidence.json", fallback)
    tracker.write_metrics(
        {
            "schema_version": 1,
            "artifact_scope": RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
            "analysis_disposition": RESOURCE_BOUNDED_ANALYSIS_DISPOSITION,
            "completion_stage": None,
            "study_outcome_eligible": False,
            "valid_completion_claim": False,
            "original_confirmatory_claim_allowed": False,
            "m9_unlock_allowed": False,
            "run_id": tracker.run_id,
            "retry_of_run_id": retry_of_run_id,
            "status": "failed",
        }
    )


def _finish_failed_resource_run(
    tracker: RunTracker,
    error: BaseException,
    *,
    core_completion: Mapping[str, Any] | None,
    retry_of_run_id: str | None,
) -> None:
    if tracker.finalized:
        return
    try:
        _demote_resource_run(
            tracker,
            error,
            core_completion=core_completion,
            retry_of_run_id=retry_of_run_id,
        )
    except BaseException as demotion_error:
        error.add_note(
            "resource completion demotion failed; run deliberately remains unsealed: "
            f"{type(demotion_error).__name__}: {demotion_error}"
        )
        return
    try:
        tracker.fail(error)
    except BaseException as seal_error:
        error.add_note(
            "resource failed-run sealing failed after demotion: "
            f"{type(seal_error).__name__}: {seal_error}"
        )


def _require_zero_scientific_registry_records(
    run_directory: Path,
    *,
    run_id: str,
) -> None:
    run_root = run_directory.parent
    attestations = read_run_stage_attestations(run_root / RUN_STAGE_ATTESTATION_REGISTRY_FILENAME)
    dispositions = read_run_dispositions(run_root / RUN_DISPOSITION_REGISTRY_FILENAME)
    if any(record.get("run_id") == run_id for record in attestations):
        raise ResourceBoundedStudyIntegrityError(
            "resource-bounded sensitivity unexpectedly has a stage attestation"
        )
    if any(record.get("run_id") == run_id for record in dispositions):
        raise ResourceBoundedStudyIntegrityError(
            "resource-bounded sensitivity unexpectedly has a stage disposition record"
        )


def qualify_resource_checkpoint_predecessor(
    predecessor_directory: Path,
    *,
    run_root: Path,
    retry_of_run_id: str,
    gate: ResourceBoundedExecutionGateEvidence,
    crop_cache: Path,
    expected_crop_cache_sha256: str,
    expected_crop_metadata_sha256: str,
    expected_raw_inventory_sha256: str,
    resolved_specs: Sequence[Any],
    prepared: PanNukeConfirmatoryInputs,
    controls: ConfirmatoryExecutionControls,
    bridge: ConfirmatoryBridgeResult,
    cnn_fingerprints: Mapping[str, Mapping[str, Mapping[str, str]]],
    integrity_verifier: Callable[[str | Path], IntegrityVerification],
) -> dict[str, Any]:
    """Qualify one failed sealed sibling before semantic checkpoint read."""

    predecessor = predecessor_directory.resolve()
    canonical_root = run_root.resolve()
    if predecessor.name != retry_of_run_id or predecessor.parent != canonical_root:
        raise ResourceBoundedStudyRunnerError(
            "checkpoint predecessor must be the explicit canonical runs-root sibling"
        )
    _plain_physical_directory(predecessor, role="checkpoint predecessor")
    mutation_lock = canonical_root / f".{retry_of_run_id}.mutation.lock"
    if mutation_lock.exists():
        raise ResourceBoundedStudyRunnerError(
            "checkpoint predecessor has an active or stale mutation lock"
        )

    integrity = integrity_verifier(predecessor)
    if (
        not integrity.valid
        or not integrity.registry_record_present
        or integrity.run_id != retry_of_run_id
        or not _valid_sha(integrity.expected_root_sha256)
    ):
        raise ResourceBoundedStudyRunnerError(
            "checkpoint predecessor is not a valid registry-backed immutable run"
        )
    status = _read_strict_physical_json(
        predecessor / "status.json",
        role="checkpoint predecessor status",
    )
    if (
        status.get("status") != "failed"
        or status.get("run_id") != retry_of_run_id
        or status.get("experiment_name") != RESOURCE_BOUNDED_EXPERIMENT_NAME
    ):
        raise ResourceBoundedStudyRunnerError(
            "checkpoint predecessor is not a terminal failed resource-bounded run"
        )
    completion = _read_strict_physical_json(
        predecessor / "completion_evidence.json",
        role="checkpoint predecessor completion evidence",
    )
    if (
        completion.get("completion_stage") is not None
        or completion.get("study_outcome_eligible") is not False
        or completion.get("valid_completion_claim") is not False
        or completion.get("artifact_scope") != RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE
        or completion.get("analysis_disposition") != RESOURCE_BOUNDED_ANALYSIS_DISPOSITION
        or completion.get("original_confirmatory_claim_allowed") is not False
        or completion.get("m9_unlock_allowed") is not False
    ):
        raise ResourceBoundedStudyRunnerError(
            "checkpoint predecessor carries a forbidden or ambiguous scientific claim"
        )
    expected_gate = {
        **gate.as_dict(),
        "confirmatory_storage_policy_sha256": (
            gate.execution_authority.confirmatory_storage_policy_sha256
        ),
    }
    saved_gate = _read_strict_physical_json(
        predecessor / "confirmatory_execution_gate.json",
        role="checkpoint predecessor P+C gate",
    )
    if saved_gate != expected_gate:
        raise ResourceBoundedStudyRunnerError(
            "checkpoint predecessor is bound to a different live P+C authority"
        )
    saved_bindings = _read_strict_physical_json(
        predecessor / "confirmatory_input_bindings.json",
        role="checkpoint predecessor structural input bindings",
    )
    expected_bindings = {
        "artifact_scope": RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
        "analysis_disposition": RESOURCE_BOUNDED_ANALYSIS_DISPOSITION,
        "completion_stage": None,
        "study_outcome_eligible": False,
        "original_confirmatory_claim_allowed": False,
        "m9_unlock_allowed": False,
        "crop_cache_path": str(crop_cache),
        "crop_cache_sha256": expected_crop_cache_sha256,
        "crop_metadata_sha256": expected_crop_metadata_sha256,
        "raw_inventory_sha256": expected_raw_inventory_sha256,
        "manifest_sha256": prepared.manifest_sha256,
        "config_semantic_sha256": prepared.config_sha256,
        "execution_controls_binding_sha256": controls.binding_sha256,
        "confirmatory_storage_policy_sha256": (
            gate.execution_authority.confirmatory_storage_policy_sha256
        ),
        "historical_primary_run_id": gate.historical_primary.primary_run_id,
        "resource_authorization_sha256": (gate.execution_authority.authorization_sha256),
        "resource_capacity_policy_sha256": canonical_sha256(
            dict(gate.execution_authority.resource_capacity_policy)
        ),
        "resource_input_workspace_plan_sha256": (
            gate.execution_authority.resource_input_workspace_plan_sha256
        ),
        "bridge": bridge.as_dict(),
        "cnn_fold_data_and_split_sha256": cnn_fingerprints,
        "feature_caches": [
            {**asdict(spec), "cache_path": str(spec.cache_path)} for spec in resolved_specs
        ],
    }
    differing = [
        field
        for field, expected in expected_bindings.items()
        if saved_bindings.get(field) != expected
    ]
    if differing:
        raise ResourceBoundedStudyRunnerError(
            "checkpoint predecessor structural bindings differ from current "
            f"authority/data/cache/split inputs: {differing}"
        )
    _require_zero_scientific_registry_records(
        predecessor,
        run_id=retry_of_run_id,
    )
    if mutation_lock.exists():
        raise ResourceBoundedStudyRunnerError(
            "checkpoint predecessor mutation lock appeared during qualification"
        )
    payload = {
        "schema_version": 1,
        "policy": "failed_sealed_resource_checkpoint_predecessor_v1",
        "qualified": True,
        "predecessor_run_id": retry_of_run_id,
        "predecessor_directory": str(predecessor),
        "terminal_status": "failed",
        "experiment_name": RESOURCE_BOUNDED_EXPERIMENT_NAME,
        "artifact_root_sha256": integrity.expected_root_sha256,
        "registry_record_present": True,
        "resource_gate_sha256": canonical_sha256(saved_gate),
        "structural_input_bindings_sha256": canonical_sha256(saved_bindings),
        "completion_evidence_sha256": sha256_file(predecessor / "completion_evidence.json"),
        "stage_attestation_record_count": 0,
        "stage_disposition_record_count": 0,
        "mutation_lock_present": False,
        "oof_artifacts_read": False,
        "metrics_artifacts_read": False,
        "ranking_artifacts_read": False,
        "automatic_retry_allowed": False,
        "auto_discovery_used": False,
    }
    payload["qualification_without_self_hash_sha256"] = canonical_sha256(payload)
    return payload


def _run_resource_bounded_sensitivity(
    *,
    run_mode: Literal["fresh", "successor_resume"],
    primary_run_directory: str | Path,
    project_root: str | Path,
    resource_authority_directory: str | Path,
    dataset_path: str | Path,
    manifest_path: str | Path,
    duplicate_audit_path: str | Path,
    pathology_encoder_audit_path: str | Path,
    lifecycle_readiness_run_directory: str | Path | None = None,
    checkpoint_predecessor_run_directory: str | Path | None = None,
    retry_of_run_id: str | None = None,
    runs_root: str | Path | None = None,
    run_id: str | None = None,
    dependencies: ResourceBoundedRunnerDependencies | None = None,
    preflight_only: bool,
) -> dict[str, Any]:
    """Shared implementation for preflight and one amended sensitivity run.

    This function never requests a positive stage attestation and can only
    return ``completion_stage=None`` and ``study_outcome_eligible=False``.
    """

    mode = _validate_run_mode(
        run_mode=run_mode,
        checkpoint_predecessor_run_directory=(checkpoint_predecessor_run_directory),
        retry_of_run_id=retry_of_run_id,
        run_id=run_id,
    )
    if lifecycle_readiness_run_directory is None:
        raise ResourceBoundedStudyRunnerError(
            "resource-bounded lifecycle readiness is required before gate, data, "
            "predecessor, or tracker access"
        )
    deps = dependencies or ResourceBoundedRunnerDependencies()
    root = Path(project_root).resolve()
    primary_run = _resolve(root, primary_run_directory)
    authority = _resolve(root, resource_authority_directory)
    readiness = _resolve(root, lifecycle_readiness_run_directory)
    dataset = _resolve(root, dataset_path)
    manifest = _resolve(root, manifest_path)
    duplicate_audit = _resolve(root, duplicate_audit_path)
    pathology_audit = _resolve(root, pathology_encoder_audit_path)
    config_path = authority / "confirmatory_frozen.yaml"
    canonical_run_root = (root / "artifacts" / "runs").resolve()
    run_root = _resolve(root, runs_root) if runs_root is not None else canonical_run_root
    if run_root != canonical_run_root:
        raise ResourceBoundedStudyRunnerError(
            "resource-bounded runs_root must be the canonical project artifacts/runs root"
        )
    predecessor = (
        _resolve(root, checkpoint_predecessor_run_directory)
        if checkpoint_predecessor_run_directory is not None
        else None
    )
    # C readiness is checked before any data, predecessor, or RunTracker access.
    deps.lifecycle_validator(
        project_root=root,
        authority_directory=authority,
        readiness_run_directory=readiness,
    )
    gate_kwargs = {
        "primary_run_directory": primary_run,
        "project_root": root,
        "resource_authority_directory": authority,
        "dataset_path": dataset,
        "manifest_path": manifest,
        "duplicate_audit_path": duplicate_audit,
        "pathology_encoder_audit_path": pathology_audit,
        "resource_confirmatory_config_path": config_path,
    }
    tracker: RunTracker | None = None
    core_completion: Mapping[str, Any] | None = None
    artifacts: ConfirmatoryMatrixArtifacts | None = None
    readback: ConfirmatoryFilesystemReadback | None = None
    candidate: dict[str, Any] | None = None
    final_capacity: ResourceCapacityEvidence | None = None
    final_compute: ResourceComputeEvidence | None = None
    workspace: ConfirmatoryMemoryWorkspace | None = None
    workspace_array_specs: tuple[ConfirmatoryWorkspaceArraySpec, ...] = ()
    workspace_index_specs: tuple[ConfirmatoryWorkspaceIndexSpec, ...] = ()
    workspace_plan: dict[str, Any] | None = None
    workspace_cleaned = False
    workspace_cleanup_attempted = False
    workspace_cleanup_evidence: dict[str, Any] | None = None

    def expected_workspace_cleanup() -> dict[str, Any]:
        if workspace is None:
            raise ResourceBoundedStudyRunnerError(
                "input workspace is unavailable for cleanup binding"
            )
        return {
            "schema_version": 1,
            "status": "complete",
            "workspace_key": workspace.workspace_key,
            "workspace_receipt_sha256": workspace.receipt_sha256,
            "artifact_root_sha256": workspace.artifact_root_sha256,
            "resource_input_workspace_plan_sha256": workspace.resource_input_workspace_plan_sha256,
            "workspace_removed": True,
            "source_annotations_modified": False,
            "scientific_outcomes_read": False,
        }

    def cleanup_workspace() -> dict[str, Any] | None:
        nonlocal workspace_cleaned, workspace_cleanup_attempted
        nonlocal workspace_cleanup_evidence
        if workspace is None:
            return None
        if workspace_cleaned:
            return workspace_cleanup_evidence
        if workspace_cleanup_attempted:
            raise ResourceBoundedStudyRunnerError(
                "input-workspace cleanup was already attempted and cannot be retried"
            )
        if workspace_plan is None:
            raise ResourceBoundedStudyRunnerError(
                "builder-owned workspace lacks its authority-D plan during cleanup"
            )
        workspace_cleanup_attempted = True
        cleanup = deps.workspace_cleaner(
            root,
            workspace,
            workspace_array_specs,
            resource_input_workspace_plan=workspace_plan,
            index_specs=workspace_index_specs,
            minimum_free_bytes_after=int(workspace_plan["minimum_free_bytes_after"]),
            maximum_workspace_bytes=int(workspace_plan["maximum_workspace_bytes"]),
        )
        expected_cleanup = expected_workspace_cleanup()
        if dict(cleanup) != expected_cleanup:
            raise ResourceBoundedStudyRunnerError(
                "input-workspace cleanup did not return its exact outcome-blind receipt"
            )
        workspace_cleanup_evidence = dict(cleanup)
        workspace_cleaned = True
        return workspace_cleanup_evidence

    try:
        # Exactly one full P+C validation occurs before execution.  Its primary
        # guard remains active through input resolution, predecessor inspection,
        # the second fixed-capacity check, and unique RunTracker creation.
        with guard_run_stage_eligibility(primary_run) as receipt:
            if receipt is None:
                raise ResourceBoundedStudyRunnerError(
                    "historical primary lacks active stage authority before run creation"
                )
            live_gate = deps.gate_validator(
                **gate_kwargs,
                primary_stage_eligibility_receipt=receipt,
            )
            _validate_resource_gate(
                live_gate,
                primary_run=primary_run,
                authority_directory=authority,
            )
            capacity_policy, workspace_plan = _exact_capacity_v3_authority(live_gate)
            initial_capacity = require_resource_capacity(
                run_root,
                capacity_policy=capacity_policy,
                phase="guarded_before_workspace_build",
                resource_input_workspace_plan=workspace_plan,
                disk_usage=deps.disk_usage,
                clock=deps.clock,
            )
            initial_compute = require_resource_compute(
                phase="guarded_before_data_loading",
                capacity_policy=capacity_policy,
                resource_input_workspace_plan=workspace_plan,
                probe=deps.compute_probe,
                clock=deps.clock,
            )
            raw_config = deps.config_loader(config_path)
            config = validate_resource_bounded_confirmatory_config(raw_config)
            if (
                sha256_file(config_path)
                != live_gate.execution_authority.resource_confirmatory_config_file_sha256
                or config_sha256(config)
                != live_gate.execution_authority.resource_confirmatory_config_semantic_sha256
            ):
                raise ResourceBoundedStudyRunnerError(
                    "runner config is not effective authority D's exact schema-v3 snapshot"
                )
            plan = deps.plan_builder(config)
            controls = deps.controls_builder(config)
            controls.validate_for_plan(plan)
            if (
                plan.schema_version != 3
                or plan.config_sha256 != RESOURCE_BOUNDED_CONFIRMATORY_CONFIG_SHA256
                or controls.plan != plan
                or len(plan.cells) != cast(int, capacity_policy["planned_required_cells"])
                or plan.required_cell_count != len(plan.cells)
            ):
                raise ResourceBoundedStudyRunnerError(
                    "resource-bounded matrix differs from effective authority D's exact profile"
                )

            resolved_inputs = deps.input_resolver(
                gate_evidence=live_gate,
                primary_run_directory=primary_run,
                resource_confirmatory_config_path=config_path,
                manifest_path=manifest,
            )
            crop_cache = resolved_inputs.crop_cache_path.resolve()
            expected_crop_cache_sha256 = resolved_inputs.expected_crop_cache_sha256
            expected_crop_metadata_sha256 = resolved_inputs.expected_crop_metadata_sha256
            expected_raw_inventory_sha256 = resolved_inputs.expected_raw_inventory_sha256
            resolved_specs = tuple(resolved_inputs.frozen_feature_caches)
            observed_label_sets = tuple(resolved_inputs.observed_label_sets)
            _verify_cache_files(
                crop_cache,
                expected_crop_cache_sha256=expected_crop_cache_sha256,
                expected_crop_metadata_sha256=expected_crop_metadata_sha256,
                frozen_feature_caches=resolved_specs,
            )
            if any(
                value.configuration_sha256 != plan.config_sha256 for value in observed_label_sets
            ):
                raise ResourceBoundedStudyRunnerError(
                    "resolved observed labels differ from authority D's resource config"
                )
            workspace_array_specs = tuple(
                deps.workspace_array_spec_builder(
                    crop_cache,
                    expected_crop_cache_sha256=expected_crop_cache_sha256,
                    expected_crop_metadata_sha256=expected_crop_metadata_sha256,
                    frozen_feature_caches=resolved_specs,
                )
            )
            workspace_index_specs = tuple(
                deps.workspace_index_spec_builder(
                    crop_cache,
                    confirmatory_config=config,
                    expected_config_sha256=plan.config_sha256,
                    expected_crop_cache_sha256=expected_crop_cache_sha256,
                    expected_crop_metadata_sha256=expected_crop_metadata_sha256,
                    expected_manifest_sha256=(
                        live_gate.historical_primary.primary_gate.manifest_sha256
                    ),
                    expected_raw_inventory_sha256=expected_raw_inventory_sha256,
                )
            )
            if len(workspace_array_specs) != cast(
                int, capacity_policy["workspace_source_array_count"]
            ) or len(workspace_index_specs) != cast(
                int, capacity_policy["workspace_partition_count"]
            ):
                raise ResourceBoundedStudyRunnerError(
                    "derived workspace inputs differ from authority D's exact 12-array/"
                    "nine-index contract"
                )
            workspace = deps.workspace_builder(
                root,
                workspace_array_specs,
                minimum_free_bytes_after=cast(
                    int,
                    capacity_policy["minimum_free_bytes_before_tracker"],
                ),
                resource_input_workspace_plan=workspace_plan,
                index_specs=workspace_index_specs,
                maximum_workspace_bytes=cast(
                    int,
                    capacity_policy["maximum_workspace_bytes"],
                ),
            )
            if (
                workspace.resource_input_workspace_plan_sha256
                != live_gate.execution_authority.resource_input_workspace_plan_sha256
                or workspace.resource_input_workspace_plan_sha256
                != workspace_plan["plan_without_self_hash_sha256"]
                or not _valid_sha(workspace.receipt_sha256)
                or not _valid_sha(workspace.artifact_root_sha256)
            ):
                raise ResourceBoundedStudyRunnerError(
                    "fresh input workspace differs from authority D's exact plan"
                )
            prepared = deps.input_builder(
                crop_cache,
                confirmatory_config=config,
                expected_config_sha256=plan.config_sha256,
                expected_crop_cache_sha256=expected_crop_cache_sha256,
                expected_crop_metadata_sha256=expected_crop_metadata_sha256,
                expected_manifest_sha256=(
                    live_gate.historical_primary.primary_gate.manifest_sha256
                ),
                expected_raw_inventory_sha256=expected_raw_inventory_sha256,
                frozen_feature_caches=resolved_specs,
                observed_label_sets=observed_label_sets,
                memory_workspace=workspace,
            )
            if (
                prepared.memory_workspace_receipt_sha256 != workspace.receipt_sha256
                or prepared.memory_workspace_artifact_root_sha256 != workspace.artifact_root_sha256
                or prepared.memory_workspace_plan_sha256
                != workspace.resource_input_workspace_plan_sha256
            ):
                raise ResourceBoundedStudyRunnerError(
                    "prepared PanNuke inputs lost the exact workspace bindings"
                )
            bridge = deps.bridge_builder(
                prepared,
                controls,
                pathology_encoder_audit_sha256=(
                    live_gate.historical_primary.primary_gate.pathology_encoder_audit_sha256
                ),
            )
            cnn_fingerprints = _confirmatory_cnn_preflight_fingerprints(
                prepared,
                bridge,
                controls,
            )
            checkpoint_allowlist = build_resource_checkpoint_allowlist(
                controls=controls,
                cnn_preflight_fingerprints=cnn_fingerprints,
            )
            predecessor_snapshot: ReadOnlyPredecessorSnapshot | None = None
            predecessor_qualification: dict[str, Any] | None = None
            if mode == "successor_resume":
                assert predecessor is not None
                assert retry_of_run_id is not None
                predecessor_qualification = qualify_resource_checkpoint_predecessor(
                    predecessor,
                    run_root=run_root,
                    retry_of_run_id=retry_of_run_id,
                    gate=live_gate,
                    crop_cache=crop_cache,
                    expected_crop_cache_sha256=expected_crop_cache_sha256,
                    expected_crop_metadata_sha256=expected_crop_metadata_sha256,
                    expected_raw_inventory_sha256=expected_raw_inventory_sha256,
                    resolved_specs=resolved_specs,
                    prepared=prepared,
                    controls=controls,
                    bridge=bridge,
                    cnn_fingerprints=cnn_fingerprints,
                    integrity_verifier=deps.integrity_verifier,
                )
                predecessor_snapshot = deps.predecessor_inspector(
                    predecessor,
                    retry_of_run_id=retry_of_run_id,
                    checkpoint_allowlist=checkpoint_allowlist,
                    validator=deps.checkpoint_validator,
                )
                _require_reusable_successor_checkpoint(predecessor_snapshot)

            assert workspace is not None
            verified_workspace = deps.workspace_verifier(
                root,
                workspace.root,
                workspace_array_specs,
                resource_input_workspace_plan=workspace_plan,
                index_specs=workspace_index_specs,
                minimum_free_bytes_after=cast(
                    int,
                    capacity_policy["minimum_free_bytes_before_tracker"],
                ),
                maximum_workspace_bytes=cast(
                    int,
                    capacity_policy["maximum_workspace_bytes"],
                ),
            )
            try:
                if (
                    verified_workspace.cleanup_ownership_token is not None
                    or verified_workspace.workspace_key != workspace.workspace_key
                    or verified_workspace.receipt_sha256 != workspace.receipt_sha256
                    or verified_workspace.artifact_root_sha256 != workspace.artifact_root_sha256
                    or verified_workspace.resource_input_workspace_plan_sha256
                    != workspace.resource_input_workspace_plan_sha256
                ):
                    raise ResourceBoundedStudyRunnerError(
                        "cold input-workspace verification differs from its fresh build"
                    )
            finally:
                verified_workspace.close()
            deps.lifecycle_validator(
                project_root=root,
                authority_directory=authority,
                readiness_run_directory=readiness,
            )
            _verify_cache_files(
                crop_cache,
                expected_crop_cache_sha256=expected_crop_cache_sha256,
                expected_crop_metadata_sha256=expected_crop_metadata_sha256,
                frozen_feature_caches=resolved_specs,
            )
            final_capacity = require_resource_capacity(
                run_root,
                capacity_policy=capacity_policy,
                phase="guarded_immediately_before_tracker",
                resource_input_workspace_plan=workspace_plan,
                disk_usage=deps.disk_usage,
                clock=deps.clock,
            )
            final_compute = require_resource_compute(
                phase="guarded_immediately_before_tracker",
                capacity_policy=capacity_policy,
                resource_input_workspace_plan=workspace_plan,
                probe=deps.compute_probe,
                clock=deps.clock,
            )
            compute_evidence = _resource_compute_evidence_document(
                capacity_policy=capacity_policy,
                resource_input_workspace_plan=workspace_plan,
                checks=(initial_compute, final_compute),
            )
            if preflight_only:
                reusable_checkpoint_count = (
                    len(predecessor_snapshot.resume_records)
                    if predecessor_snapshot is not None
                    else 0
                )
                missing_checkpoint_count = (
                    len(predecessor_snapshot.fresh_records)
                    if predecessor_snapshot is not None
                    else len(checkpoint_allowlist)
                )
                cleanup_evidence = cleanup_workspace()
                if cleanup_evidence is None or workspace is None:
                    raise ResourceBoundedStudyRunnerError(
                        "preflight workspace was not closed and removed"
                    )
                cleanup_sha256 = canonical_sha256(cleanup_evidence)
                return {
                    "status": "passed",
                    "operation": "resource_bounded_sensitivity_preflight",
                    "tracker_created": False,
                    "scientific_run_created": False,
                    "completion_stage": None,
                    "study_outcome_eligible": False,
                    "valid_completion_claim": False,
                    "artifact_scope": RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
                    "analysis_disposition": RESOURCE_BOUNDED_ANALYSIS_DISPOSITION,
                    "original_confirmatory_claim_allowed": False,
                    "m9_unlock_allowed": False,
                    "outcomes_inspected": True,
                    "run_mode": mode,
                    "retry_of_run_id": retry_of_run_id,
                    "predecessor_directory": (
                        str(predecessor) if predecessor is not None else None
                    ),
                    "predecessor_read_performed": (predecessor_snapshot is not None),
                    "predecessor_qualified": (predecessor_qualification is not None),
                    "predecessor_qualification_sha256": (
                        predecessor_qualification.get("qualification_without_self_hash_sha256")
                        if predecessor_qualification is not None
                        else None
                    ),
                    "checkpoint_copy_performed": False,
                    "checkpoint_allowlist_count": len(checkpoint_allowlist),
                    "reusable_checkpoint_count": reusable_checkpoint_count,
                    "missing_checkpoint_count": missing_checkpoint_count,
                    "capacity_checks": [
                        initial_capacity.as_dict(),
                        final_capacity.as_dict(),
                    ],
                    "compute_checks": [
                        initial_compute.as_dict(),
                        final_compute.as_dict(),
                    ],
                    "resource_compute_evidence_sha256": compute_evidence[
                        "evidence_without_self_hash_sha256"
                    ],
                    "resource_capacity_policy": capacity_policy,
                    "resource_capacity_policy_sha256": canonical_sha256(capacity_policy),
                    "resource_input_workspace_plan_sha256": (
                        workspace.resource_input_workspace_plan_sha256
                    ),
                    "resource_input_workspace_receipt_sha256": (workspace.receipt_sha256),
                    "resource_input_workspace_artifact_root_sha256": (
                        workspace.artifact_root_sha256
                    ),
                    "resource_input_workspace_cleanup_sha256": cleanup_sha256,
                    "resource_input_workspace_removed": True,
                    "resource_gate_sha256": canonical_sha256(live_gate.as_dict()),
                    "historical_primary_run_id": (live_gate.historical_primary.primary_run_id),
                    "resource_authorization_sha256": (
                        live_gate.execution_authority.authorization_sha256
                    ),
                    "resource_config_semantic_sha256": plan.config_sha256,
                    "planned_cell_count": len(plan.cells),
                    "required_cell_count": plan.required_cell_count,
                    "data_and_cache_inputs_verified": True,
                    "lifecycle_readiness_verified": True,
                    "dual_authority_gate_verified": True,
                    "full_pc_gate_validation_count": 1,
                    "monitoring_contract": ("outcome_value_free_operational_telemetry"),
                    "monitoring_prohibited_for_selection_tuning": True,
                    "automatic_retry_allowed": False,
                    "auto_discovery_used": False,
                    "oof_artifacts_read": False,
                    "metrics_artifacts_read": False,
                    "ranking_artifacts_read": False,
                }
            tracker = deps.tracker_starter(
                experiment_name=RESOURCE_BOUNDED_EXPERIMENT_NAME,
                config=config,
                project_root=root,
                runs_root=run_root,
                run_id=run_id,
                dataset_path=dataset,
                manifest_path=manifest,
                duplicate_audit_status="passed",
            )
        if (
            tracker.source_tree.get("root_sha256")
            != live_gate.execution_authority.resource_execution_source_root_sha256
        ):
            raise ResourceBoundedStudyRunnerError(
                "source tree changed between authority D validation and tracker capture"
            )
        assert workspace is not None
        assert workspace_plan is not None
        live_workspace_receipt = _read_strict_physical_json(
            workspace.root / "workspace_receipt.json",
            role="fresh resource input-workspace receipt",
        )
        if (
            sha256_file(workspace.root / "workspace_receipt.json") != workspace.receipt_sha256
            or live_workspace_receipt.get("resource_input_workspace_plan_sha256")
            != workspace.resource_input_workspace_plan_sha256
        ):
            raise ResourceBoundedStudyRunnerError(
                "fresh input-workspace receipt differs before tracked execution"
            )
        workspace_plan_path = tracker.write_json(
            "resource_input_workspace_plan.json",
            workspace_plan,
        )
        workspace_receipt_path = tracker.write_json(
            "resource_input_workspace_receipt.json",
            live_workspace_receipt,
        )
        workspace_plan_file_sha256 = sha256_file(workspace_plan_path)
        workspace_receipt_file_sha256 = sha256_file(workspace_receipt_path)
        expected_workspace_cleanup_sha256 = canonical_sha256(expected_workspace_cleanup())

        predecessor_qualification_sha256: str | None = None
        if predecessor_qualification is not None:
            raw_qualification_sha = predecessor_qualification.get(
                "qualification_without_self_hash_sha256"
            )
            if not _valid_sha(raw_qualification_sha):
                raise ResourceBoundedStudyRunnerError(
                    "predecessor qualification lacks its exact semantic SHA-256"
                )
            predecessor_qualification_sha256 = str(raw_qualification_sha)
            tracker.write_json(
                "resource_predecessor_qualification.json",
                predecessor_qualification,
            )
        if predecessor_snapshot is None:
            checkpoint_preparation = deps.fresh_checkpoint_execution_preparer(checkpoint_allowlist)
        else:
            copy_receipt = deps.checkpoint_copier(
                predecessor_snapshot,
                tracker.run_directory,
                validator=deps.checkpoint_validator,
            )
            checkpoint_preparation = deps.successor_checkpoint_execution_preparer(
                predecessor_snapshot,
                copy_receipt,
            )
        if not isinstance(
            checkpoint_preparation,
            ResourceBoundedCheckpointExecutionPreparation,
        ):
            raise ResourceBoundedStudyRunnerError(
                "resource checkpoint preparation returned an untyped result"
            )
        resume_evidence = dict(checkpoint_preparation.resume_evidence)
        resume_sha = resume_evidence.get(RESOURCE_BOUNDED_RESUME_EVIDENCE_HASH_FIELD)
        if not _valid_sha(resume_sha):
            raise ResourceBoundedStudyRunnerError(
                "resource resume evidence lacks its exact semantic SHA-256"
            )
        resume_evidence_path = tracker.write_json(
            "resource_resume_evidence.json",
            resume_evidence,
        )
        if sha256_file(resume_evidence_path) == str(resume_sha):
            raise ResourceBoundedStudyRunnerError(
                "resume self-excluding semantic hash was confused with its file hash"
            )
        capacity_policy_sha = canonical_sha256(capacity_policy)
        assert final_capacity is not None
        tracker.write_json(
            "resource_capacity_evidence.json",
            {
                "schema_version": 1,
                "policy": capacity_policy,
                "policy_sha256": capacity_policy_sha,
                "checks": [
                    initial_capacity.as_dict(),
                    final_capacity.as_dict(),
                ],
                "passed": True,
            },
        )
        compute_evidence_sha = compute_evidence.get("evidence_without_self_hash_sha256")
        if not _valid_sha(compute_evidence_sha):
            raise ResourceBoundedStudyRunnerError(
                "resource compute evidence lacks its exact semantic SHA-256"
            )
        tracker.write_json(
            "resource_compute_evidence.json",
            compute_evidence,
        )
        gate_payload = {
            **live_gate.as_dict(),
            "confirmatory_storage_policy_sha256": (
                live_gate.execution_authority.confirmatory_storage_policy_sha256
            ),
        }
        tracker.write_json("confirmatory_execution_gate.json", gate_payload)
        bindings_path = tracker.write_json(
            "confirmatory_input_bindings.json",
            {
                "schema_version": 1,
                "artifact_scope": RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
                "analysis_disposition": RESOURCE_BOUNDED_ANALYSIS_DISPOSITION,
                "completion_stage": None,
                "study_outcome_eligible": False,
                "original_confirmatory_claim_allowed": False,
                "m9_unlock_allowed": False,
                "crop_cache_path": str(crop_cache),
                "crop_cache_sha256": expected_crop_cache_sha256,
                "crop_metadata_sha256": expected_crop_metadata_sha256,
                "raw_inventory_sha256": expected_raw_inventory_sha256,
                "manifest_sha256": prepared.manifest_sha256,
                "config_semantic_sha256": prepared.config_sha256,
                "execution_controls_binding_sha256": controls.binding_sha256,
                "confirmatory_storage_policy_sha256": (
                    live_gate.execution_authority.confirmatory_storage_policy_sha256
                ),
                "historical_primary_run_id": (live_gate.historical_primary.primary_run_id),
                "resource_authorization_sha256": (
                    live_gate.execution_authority.authorization_sha256
                ),
                "resource_capacity_policy_sha256": capacity_policy_sha,
                "resource_input_workspace_plan_sha256": (
                    workspace.resource_input_workspace_plan_sha256
                ),
                "resource_input_workspace_receipt_sha256": workspace.receipt_sha256,
                "resource_input_workspace_artifact_root_sha256": (workspace.artifact_root_sha256),
                "resource_input_workspace_plan_file_sha256": (workspace_plan_file_sha256),
                "resource_input_workspace_receipt_file_sha256": (workspace_receipt_file_sha256),
                "resource_input_workspace_cleanup_sha256": (expected_workspace_cleanup_sha256),
                "resource_compute_evidence_sha256": compute_evidence_sha,
                "resource_resume_evidence_sha256": resume_sha,
                "resource_predecessor_qualification_sha256": (predecessor_qualification_sha256),
                "bridge": bridge.as_dict(),
                "cnn_fold_data_and_split_sha256": cnn_fingerprints,
                "feature_caches": [
                    {**asdict(spec), "cache_path": str(spec.cache_path)} for spec in resolved_specs
                ],
                "run_mode": mode,
                "retry_of_run_id": retry_of_run_id,
                "resume_policy": (
                    "explicit predecessor checkpoint allowlist; no discovery, "
                    "automatic retry, OOF, metric, or ranking read"
                ),
            },
        )
        tracker.write_provenance(
            artifact_scope=RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
            analysis_disposition=RESOURCE_BOUNDED_ANALYSIS_DISPOSITION,
            completion_stage=None,
            study_outcome_eligible=False,
            original_confirmatory_claim_allowed=False,
            m9_unlock_allowed=False,
            outcomes_inspected=True,
            resource_gate=live_gate.as_dict(),
            confirmatory_input_bindings_sha256=sha256_file(bindings_path),
            confirmatory_storage_policy_sha256=(
                live_gate.execution_authority.confirmatory_storage_policy_sha256
            ),
            resource_capacity_policy_sha256=capacity_policy_sha,
            resource_input_workspace_plan_sha256=(workspace.resource_input_workspace_plan_sha256),
            resource_input_workspace_receipt_sha256=workspace.receipt_sha256,
            resource_input_workspace_artifact_root_sha256=(workspace.artifact_root_sha256),
            resource_input_workspace_cleanup_sha256=(expected_workspace_cleanup_sha256),
            resource_compute_evidence_sha256=compute_evidence_sha,
            resource_resume_evidence_sha256=resume_sha,
            resource_predecessor_qualification_sha256=(predecessor_qualification_sha256),
            matrix_plan_sha256=canonical_sha256(plan.as_dict()),
            execution_controls_binding_sha256=controls.binding_sha256,
            run_mode=mode,
            retry_of_run_id=retry_of_run_id,
            stage_attestation_forbidden=True,
        )

        artifacts = deps.matrix_executor(
            bridge.rotations,
            plan,
            controls,
            output_directory=tracker.run_directory,
            frozen_oof_runner=run_confirmatory_frozen_feature_oof,
            frozen_blockers=bridge.frozen_blockers,
            artifact_scope=RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
            cpu_test_only=False,
            checkpoint_execution_contract=(checkpoint_preparation.checkpoint_execution_contract),
            gate_evidence=live_gate,
            progress_callback=lambda event: _log_resource_progress(tracker, event),
        )
        core_completion = _validate_core_artifacts(
            artifacts,
            run_directory=tracker.run_directory,
            plan=plan,
            expected_artifact_scope=(RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE),
            expected_resource_gate=live_gate,
        )
        statistics = deps.statistics_aggregator(tracker.run_directory, controls)
        statistics_verification = deps.statistics_verifier(
            tracker.run_directory,
            controls,
        )
        if (
            statistics_verification.status != "passed"
            or Path(statistics.output_directory).resolve() != tracker.run_directory
            or Path(statistics_verification.output_directory).resolve() != tracker.run_directory
            or statistics.statistics_sha256 != statistics_verification.statistics_sha256
            or statistics.bootstrap_evidence_sha256
            != statistics_verification.bootstrap_evidence_sha256
            or statistics_verification.completed_comparison_count
            != statistics_verification.comparison_count
        ):
            raise ResourceBoundedStudyRunnerError(
                "resource-bounded paired statistics failed strict verification"
            )
        finalize_resource_bounded_analysis(
            run_directory=tracker.run_directory,
            matrix_artifacts=artifacts,
            statistics_artifacts=statistics,
            prepared_inputs=prepared,
            bridge=bridge,
            controls=controls,
            config_semantic_authority_sha256=(
                live_gate.execution_authority.resource_confirmatory_config_semantic_sha256
            ),
        )
        _validate_restoration_source_binding(
            tracker.run_directory,
            prepared_inputs=prepared,
            bridge=bridge,
            controls=controls,
        )
        cleanup_evidence = cleanup_workspace()
        if (
            cleanup_evidence is None
            or canonical_sha256(cleanup_evidence) != expected_workspace_cleanup_sha256
        ):
            raise ResourceBoundedStudyRunnerError(
                "input workspace cleanup differs from its pre-execution binding"
            )
        workspace_cleanup_path = tracker.write_json(
            "resource_input_workspace_cleanup.json",
            cleanup_evidence,
        )
        workspace_cleanup_file_sha256 = sha256_file(workspace_cleanup_path)
        readback = deps.filesystem_reader(
            plan,
            tracker.run_directory,
            frozen_confirmatory_config_path=config_path,
            expected_frozen_config_sha256=(
                live_gate.execution_authority.resource_confirmatory_config_file_sha256
            ),
            expected_confirmatory_storage_policy_sha256=(
                live_gate.execution_authority.confirmatory_storage_policy_sha256
            ),
            require_final_policy_bindings=False,
        )
        _validate_final_readback(
            readback,
            artifacts,
            tracker.run_directory,
            expected_confirmatory_storage_policy_sha256=(
                live_gate.execution_authority.confirmatory_storage_policy_sha256
            ),
        )
        raw_candidate = deps.completion_builder(
            plan=plan,
            reconciliation=artifacts.reconciliation,
            artifact_scope=RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
            study_outcome_eligible=False,
            gate_evidence=live_gate,
        )
        candidate = _resource_completion_candidate(
            raw_candidate,
            plan=plan,
            readback=readback,
            core_completion=core_completion,
            gate=live_gate,
            run_id=tracker.run_id,
            retry_of_run_id=retry_of_run_id,
            statistics=statistics_verification,
            resume_evidence_sha256=str(resume_sha),
            predecessor_qualification_sha256=(predecessor_qualification_sha256),
            capacity_policy_sha256=capacity_policy_sha,
            workspace_plan_sha256=str(workspace.resource_input_workspace_plan_sha256),
            workspace_plan_file_sha256=workspace_plan_file_sha256,
            workspace_receipt_sha256=workspace.receipt_sha256,
            workspace_receipt_file_sha256=workspace_receipt_file_sha256,
            workspace_artifact_root_sha256=workspace.artifact_root_sha256,
            workspace_cleanup_sha256=expected_workspace_cleanup_sha256,
            workspace_cleanup_file_sha256=workspace_cleanup_file_sha256,
            compute_evidence_sha256=str(compute_evidence_sha),
        )
        tracker.write_json("core_completion_evidence.json", core_completion)
        completion_path = tracker.write_json(
            "completion_evidence.json",
            candidate,
        )
        report = (tracker.run_directory / "report.md").read_text(encoding="utf-8")
        if (
            "resource-bounded" not in report
            or "sensitivity" not in report
            or "potentially inconsistent annotation" not in report
            or "recommended for expert review" not in report
            or "cannot unlock external-validation milestone M9" not in report
        ):
            raise ResourceBoundedStudyRunnerError(
                "resource sensitivity report lacks its mandatory non-claiming terminology"
            )
        tracker.write_metrics(
            {
                "schema_version": 1,
                "artifact_scope": RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
                "analysis_disposition": RESOURCE_BOUNDED_ANALYSIS_DISPOSITION,
                "completion_stage": None,
                "study_outcome_eligible": False,
                "valid_completion_claim": False,
                "original_confirmatory_claim_allowed": False,
                "m9_unlock_allowed": False,
                "run_id": tracker.run_id,
                "retry_of_run_id": retry_of_run_id,
                "matrix_config_sha256": plan.config_sha256,
                "planned_cell_count": len(plan.cells),
                "required_cell_count": plan.required_cell_count,
                "completed_required_cell_count": plan.required_cell_count,
                "filesystem_readback_status": readback.status,
                "filesystem_checked_artifact_count": readback.checked_artifact_count,
                "completion_evidence_sha256": sha256_file(completion_path),
                "statistics_sha256": statistics_verification.statistics_sha256,
                "bootstrap_evidence_sha256": (statistics_verification.bootstrap_evidence_sha256),
                "confirmatory_storage_policy_sha256": (
                    live_gate.execution_authority.confirmatory_storage_policy_sha256
                ),
                "resource_capacity_policy_sha256": capacity_policy_sha,
                "resource_input_workspace_plan_sha256": (
                    workspace.resource_input_workspace_plan_sha256
                ),
                "resource_input_workspace_receipt_sha256": (workspace.receipt_sha256),
                "resource_input_workspace_artifact_root_sha256": (workspace.artifact_root_sha256),
                "resource_input_workspace_cleanup_sha256": (expected_workspace_cleanup_sha256),
                "resource_input_workspace_cleanup_file_sha256": (workspace_cleanup_file_sha256),
                "resource_input_workspace_removed": True,
                "resource_compute_evidence_sha256": compute_evidence_sha,
                "resource_resume_evidence_sha256": resume_sha,
                "resource_predecessor_qualification_sha256": (predecessor_qualification_sha256),
                "full_pc_gate_validation_count": 2,
                "status": "completed_non_claiming_sensitivity",
            }
        )
        tracker.log_event(
            "resource_bounded_sensitivity_candidate_written",
            completion_stage=None,
            study_outcome_eligible=False,
            analysis_disposition=RESOURCE_BOUNDED_ANALYSIS_DISPOSITION,
            m9_unlock_allowed=False,
            completion_evidence_sha256=sha256_file(completion_path),
        )

        with guard_run_stage_eligibility(primary_run) as preseal_receipt:
            if preseal_receipt is None:
                raise ResourceBoundedStudyRunnerError(
                    "historical primary lost stage authority before sensitivity seal"
                )
            preseal_gate = deps.gate_validator(
                **gate_kwargs,
                primary_stage_eligibility_receipt=preseal_receipt,
            )
            _require_gate_equality(live_gate, preseal_gate)
            deps.lifecycle_validator(
                project_root=root,
                authority_directory=authority,
                readiness_run_directory=readiness,
            )
            tracker.complete()
    except BaseException as error:
        failure: BaseException = error
        if workspace is not None and not workspace_cleanup_attempted:
            try:
                cleanup_evidence = cleanup_workspace()
                if tracker is not None and cleanup_evidence is not None:
                    cleanup_path = tracker.run_directory / "resource_input_workspace_cleanup.json"
                    if not cleanup_path.exists():
                        tracker.write_json(
                            "resource_input_workspace_cleanup.json",
                            cleanup_evidence,
                        )
            except BaseException as cleanup_error:
                cleanup_error.add_note(
                    "input-workspace cleanup failed while handling the original "
                    f"{type(error).__name__}: {error}"
                )
                failure = cleanup_error
        if tracker is not None:
            _finish_failed_resource_run(
                tracker,
                failure,
                core_completion=core_completion,
                retry_of_run_id=retry_of_run_id,
            )
        if failure is not error:
            raise failure from error
        raise

    assert tracker is not None
    assert artifacts is not None
    assert readback is not None
    assert candidate is not None
    integrity = deps.integrity_verifier(tracker.run_directory)
    if (
        not integrity.valid
        or not integrity.registry_record_present
        or integrity.run_id != tracker.run_id
        or not _valid_sha(integrity.expected_root_sha256)
    ):
        raise ResourceBoundedStudyIntegrityError(
            "sealed resource sensitivity failed post-seal integrity verification: "
            f"{integrity.errors}"
        )
    sealed = cast(
        dict[str, Any],
        __import__("json").loads(
            (tracker.run_directory / "completion_evidence.json").read_text(encoding="utf-8")
        ),
    )
    if sealed != candidate:
        raise ResourceBoundedStudyIntegrityError(
            "sealed resource completion differs from the verified non-claiming candidate"
        )
    try:
        sealed_compute = _read_strict_physical_json(
            tracker.run_directory / "resource_compute_evidence.json",
            role="sealed resource compute evidence",
        )
        sealed_compute_sha = _resource_compute_document_readback(
            sealed_compute,
            capacity_policy=capacity_policy,
            resource_input_workspace_plan=workspace_plan,
        )
    except ResourceBoundedStudyRunnerError as exc:
        raise ResourceBoundedStudyIntegrityError(
            "sealed resource compute evidence failed exact readback"
        ) from exc
    if sealed_compute != compute_evidence or sealed_compute_sha != sealed.get(
        "resource_compute_evidence_sha256"
    ):
        raise ResourceBoundedStudyIntegrityError(
            "sealed resource compute evidence differs from the pre-tracker carrier"
        )
    sealed_workspace_plan = _read_strict_physical_json(
        tracker.run_directory / "resource_input_workspace_plan.json",
        role="sealed resource input-workspace plan",
    )
    sealed_workspace_receipt = _read_strict_physical_json(
        tracker.run_directory / "resource_input_workspace_receipt.json",
        role="sealed resource input-workspace receipt",
    )
    if (
        sealed_workspace_plan != workspace_plan
        or sealed_workspace_receipt != live_workspace_receipt
        or sha256_file(tracker.run_directory / "resource_input_workspace_plan.json")
        != sealed.get("resource_input_workspace_plan_file_sha256")
        or sha256_file(tracker.run_directory / "resource_input_workspace_receipt.json")
        != sealed.get("resource_input_workspace_receipt_file_sha256")
        or sealed_workspace_plan.get("plan_without_self_hash_sha256")
        != sealed.get("resource_input_workspace_plan_sha256")
        or sealed.get("resource_input_workspace_receipt_sha256") != workspace.receipt_sha256
        or sealed.get("resource_input_workspace_artifact_root_sha256")
        != workspace.artifact_root_sha256
    ):
        raise ResourceBoundedStudyIntegrityError(
            "sealed input-workspace plan/receipt differs from its authority-bound build"
        )
    sealed_workspace_cleanup = _read_strict_physical_json(
        tracker.run_directory / "resource_input_workspace_cleanup.json",
        role="sealed resource input-workspace cleanup",
    )
    if (
        sealed_workspace_cleanup != expected_workspace_cleanup()
        or canonical_sha256(sealed_workspace_cleanup)
        != sealed.get("resource_input_workspace_cleanup_sha256")
        or sha256_file(tracker.run_directory / "resource_input_workspace_cleanup.json")
        != sealed.get("resource_input_workspace_cleanup_file_sha256")
        or sealed_workspace_cleanup.get("workspace_removed") is not True
    ):
        raise ResourceBoundedStudyIntegrityError(
            "sealed input-workspace cleanup differs from its pre-execution binding"
        )
    postseal_readback = deps.filesystem_reader(
        plan,
        tracker.run_directory,
        frozen_confirmatory_config_path=config_path,
        expected_frozen_config_sha256=(
            live_gate.execution_authority.resource_confirmatory_config_file_sha256
        ),
        expected_confirmatory_storage_policy_sha256=(
            live_gate.execution_authority.confirmatory_storage_policy_sha256
        ),
        require_final_policy_bindings=True,
    )
    _validate_final_readback(
        postseal_readback,
        artifacts,
        tracker.run_directory,
        expected_confirmatory_storage_policy_sha256=(
            live_gate.execution_authority.confirmatory_storage_policy_sha256
        ),
    )
    if (
        postseal_readback.matrix_plan_sha256 != readback.matrix_plan_sha256
        or postseal_readback.cell_index_sha256 != readback.cell_index_sha256
        or postseal_readback.root_artifact_manifest_sha256 != readback.root_artifact_manifest_sha256
        or postseal_readback.reconciliation != readback.reconciliation
    ):
        raise ResourceBoundedStudyIntegrityError(
            "post-seal sensitivity readback differs from pre-seal scientific readback"
        )
    final_integrity = deps.integrity_verifier(tracker.run_directory)
    if (
        not final_integrity.valid
        or not final_integrity.registry_record_present
        or final_integrity.run_id != tracker.run_id
        or final_integrity.expected_root_sha256 != integrity.expected_root_sha256
        or final_integrity.actual_root_sha256 != integrity.actual_root_sha256
    ):
        raise ResourceBoundedStudyIntegrityError(
            "resource sensitivity integrity changed during post-seal readback"
        )
    _require_zero_scientific_registry_records(
        tracker.run_directory,
        run_id=tracker.run_id,
    )
    if (
        sealed.get("completion_stage") is not None
        or sealed.get("study_outcome_eligible") is not False
        or sealed.get("valid_completion_claim") is not False
        or sealed.get("analysis_disposition") != RESOURCE_BOUNDED_ANALYSIS_DISPOSITION
        or sealed.get("m9_unlock_allowed") is not False
    ):
        raise ResourceBoundedStudyIntegrityError(
            "sealed sensitivity acquired a forbidden scientific claim"
        )
    return {
        "status": "completed",
        "completion_stage": None,
        "study_outcome_eligible": False,
        "valid_completion_claim": False,
        "artifact_scope": RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
        "analysis_disposition": RESOURCE_BOUNDED_ANALYSIS_DISPOSITION,
        "original_confirmatory_claim_allowed": False,
        "m9_unlock_allowed": False,
        "run_mode": mode,
        "run_id": tracker.run_id,
        "run_directory": str(tracker.run_directory),
        "retry_of_run_id": retry_of_run_id,
        "artifact_root_sha256": final_integrity.expected_root_sha256,
        "registry_record_present": final_integrity.registry_record_present,
        "post_seal_filesystem_readback_status": postseal_readback.status,
        "stage_attestation_record_count": 0,
        "stage_disposition_record_count": 0,
        "confirmatory_storage_policy_sha256": (
            live_gate.execution_authority.confirmatory_storage_policy_sha256
        ),
        "resource_capacity_policy_sha256": canonical_sha256(capacity_policy),
        "resource_input_workspace_plan_sha256": sealed["resource_input_workspace_plan_sha256"],
        "resource_input_workspace_receipt_sha256": sealed[
            "resource_input_workspace_receipt_sha256"
        ],
        "resource_input_workspace_artifact_root_sha256": sealed[
            "resource_input_workspace_artifact_root_sha256"
        ],
        "resource_input_workspace_cleanup_sha256": sealed[
            "resource_input_workspace_cleanup_sha256"
        ],
        "resource_input_workspace_removed": True,
        "resource_compute_evidence_sha256": sealed["resource_compute_evidence_sha256"],
        "resource_resume_evidence_sha256": sealed["resource_resume_evidence_sha256"],
        "resource_predecessor_qualification_sha256": sealed.get(
            "resource_predecessor_qualification_sha256"
        ),
        "completion_evidence_path": str(tracker.run_directory / "completion_evidence.json"),
        "completion_evidence_sha256": sha256_file(
            tracker.run_directory / "completion_evidence.json"
        ),
        "planned_cell_count": len(plan.cells),
        "completed_required_cell_count": plan.required_cell_count,
        "full_pc_gate_validation_count": 2,
    }


def preflight_resource_bounded_sensitivity(
    *,
    run_mode: Literal["fresh", "successor_resume"],
    primary_run_directory: str | Path,
    project_root: str | Path,
    resource_authority_directory: str | Path,
    dataset_path: str | Path,
    manifest_path: str | Path,
    duplicate_audit_path: str | Path,
    pathology_encoder_audit_path: str | Path,
    lifecycle_readiness_run_directory: str | Path | None = None,
    checkpoint_predecessor_run_directory: str | Path | None = None,
    retry_of_run_id: str | None = None,
    runs_root: str | Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run the complete read-only sensitivity preflight without creating a tracker."""

    return _run_resource_bounded_sensitivity(
        run_mode=run_mode,
        primary_run_directory=primary_run_directory,
        project_root=project_root,
        resource_authority_directory=resource_authority_directory,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        duplicate_audit_path=duplicate_audit_path,
        pathology_encoder_audit_path=pathology_encoder_audit_path,
        lifecycle_readiness_run_directory=lifecycle_readiness_run_directory,
        checkpoint_predecessor_run_directory=(checkpoint_predecessor_run_directory),
        retry_of_run_id=retry_of_run_id,
        runs_root=runs_root,
        run_id=run_id,
        dependencies=None,
        preflight_only=True,
    )


def execute_resource_bounded_sensitivity(
    *,
    run_mode: Literal["fresh", "successor_resume"],
    primary_run_directory: str | Path,
    project_root: str | Path,
    resource_authority_directory: str | Path,
    dataset_path: str | Path,
    manifest_path: str | Path,
    duplicate_audit_path: str | Path,
    pathology_encoder_audit_path: str | Path,
    lifecycle_readiness_run_directory: str | Path | None = None,
    checkpoint_predecessor_run_directory: str | Path | None = None,
    retry_of_run_id: str | None = None,
    runs_root: str | Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Execute one fresh or explicit-successor amended sensitivity run."""

    return _run_resource_bounded_sensitivity(
        run_mode=run_mode,
        primary_run_directory=primary_run_directory,
        project_root=project_root,
        resource_authority_directory=resource_authority_directory,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        duplicate_audit_path=duplicate_audit_path,
        pathology_encoder_audit_path=pathology_encoder_audit_path,
        lifecycle_readiness_run_directory=lifecycle_readiness_run_directory,
        checkpoint_predecessor_run_directory=(checkpoint_predecessor_run_directory),
        retry_of_run_id=retry_of_run_id,
        runs_root=runs_root,
        run_id=run_id,
        dependencies=None,
        preflight_only=False,
    )


__all__ = [
    "RESOURCE_BOUNDED_ANALYSIS_DISPOSITION",
    "RESOURCE_BOUNDED_CAPACITY_POLICY",
    "RESOURCE_BOUNDED_EXPERIMENT_NAME",
    "ResourceBoundedStudyIntegrityError",
    "ResourceBoundedStudyRunnerError",
    "ResourceCapacityEvidence",
    "ResourceComputeEvidence",
    "build_resource_checkpoint_allowlist",
    "execute_resource_bounded_sensitivity",
    "preflight_resource_bounded_sensitivity",
    "require_resource_capacity",
    "require_resource_compute",
]
