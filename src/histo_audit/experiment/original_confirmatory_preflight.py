"""Fail-closed resource preflight for the unchanged original confirmatory study.

This module is deliberately separate from the resource-bounded sensitivity
runner and its authority receipts.  It binds the exact frozen 108-cell plan,
all caller-declared caches, the official ResNet-18 weight, CUDA/AMP execution,
host/GPU memory, and both physical copies required for every successful CNN
fold checkpoint: the canonical checkpoint and its distinct versioned O_EXCL
artifact.  The conservative bound is two 30-GiB copies plus a fixed 10-GiB
safety margin.  It reads no outcomes and creates no run or authority artifact.
"""

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast

from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.experiment.confirmatory_core import ConfirmatoryExecutionControls
from histo_audit.experiment.original_confirmatory_resume import (
    ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT,
    ORIGINAL_CONFIRMATORY_CNN_CELL_COUNT,
    ORIGINAL_CONFIRMATORY_CONFIG_SEMANTIC_SHA256,
    ORIGINAL_CONFIRMATORY_CONTROLS_BINDING_SHA256,
    ORIGINAL_CONFIRMATORY_OOF_FOLD_COUNT,
    ORIGINAL_CONFIRMATORY_OPTIONAL_CELL_COUNT,
    ORIGINAL_CONFIRMATORY_PLAN_SEMANTIC_SHA256,
    ORIGINAL_CONFIRMATORY_PLANNED_CELL_COUNT,
    ORIGINAL_CONFIRMATORY_REQUIRED_CELL_COUNT,
    ORIGINAL_CONFIRMATORY_WEIGHT_SHA256,
    OriginalConfirmatoryResumeError,
    _hash_runtime_file,
    _official_weight_binding,
    _plain_directory_identity,
    _require_exact_original_controls,
)
from histo_audit.models.cnn import OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER

_GIB = 1_073_741_824
ORIGINAL_CONFIRMATORY_CAPACITY_POLICY = MappingProxyType(
    {
        "schema_version": 2,
        "policy": "original_confirmatory_sealed_plan_capacity_v2",
        "capacity_basis": (
            "PLAN.md_2026-07-30_checkpoint_estimate_and_"
            "DECISIONS.md_2026-07-30_distinct_dual_copy_policy"
        ),
        "plan_semantic_sha256": ORIGINAL_CONFIRMATORY_PLAN_SEMANTIC_SHA256,
        "config_semantic_sha256": ORIGINAL_CONFIRMATORY_CONFIG_SEMANTIC_SHA256,
        "controls_binding_sha256": ORIGINAL_CONFIRMATORY_CONTROLS_BINDING_SHA256,
        "planned_cell_count": ORIGINAL_CONFIRMATORY_PLANNED_CELL_COUNT,
        "planned_required_cell_count": ORIGINAL_CONFIRMATORY_REQUIRED_CELL_COUNT,
        "planned_optional_cell_count": ORIGINAL_CONFIRMATORY_OPTIONAL_CELL_COUNT,
        "planned_cnn_cell_count": ORIGINAL_CONFIRMATORY_CNN_CELL_COUNT,
        "planned_cnn_fold_checkpoint_count": ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT,
        "oof_fold_count": ORIGINAL_CONFIRMATORY_OOF_FOLD_COUNT,
        "checkpoint_publication_policy": (
            "canonical_plus_distinct_versioned_o_excl_physical_copy_v1"
        ),
        "checkpoint_physical_copy_count": 2,
        "projected_checkpoint_bytes_per_physical_copy": 30 * _GIB,
        "projected_all_checkpoint_copies_bytes": 60 * _GIB,
        "fixed_safety_margin_bytes": 10 * _GIB,
        "minimum_free_bytes": 70 * _GIB,
        "minimum_total_ram_bytes": 30 * _GIB,
        "minimum_available_ram_bytes_before_data": 16 * _GIB,
        "minimum_available_ram_bytes_before_tracker": 12 * _GIB,
        "cuda_required": True,
        "cuda_device_index": 0,
        "minimum_total_vram_bytes": 10 * _GIB,
        "minimum_free_vram_bytes": 8 * _GIB,
        "cudnn_required": True,
        "amp_required": True,
        "amp_dtype": "float16",
        "grad_scaler_required": True,
        "cuda_smoke_input_shapes": ((1, 3, 224, 224), (1, 4, 224, 224)),
        "cuda_smoke_forward_backward_required": True,
        "cuda_smoke_finite_required": True,
        "cuda_smoke_max_peak_allocated_bytes": 1 * _GIB,
        "official_weight_identifier": OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER,
        "official_weight_sha256": ORIGINAL_CONFIRMATORY_WEIGHT_SHA256,
        "implicit_weight_download_allowed": False,
        "outcome_values_read": False,
        "adaptive_execution_changes_allowed": False,
    }
)
ORIGINAL_CONFIRMATORY_CAPACITY_POLICY_SHA256 = canonical_sha256(
    dict(ORIGINAL_CONFIRMATORY_CAPACITY_POLICY)
)
ORIGINAL_CONFIRMATORY_INITIAL_PREFLIGHT_FILENAME = "original_confirmatory_initial_preflight.json"
ORIGINAL_CONFIRMATORY_FINAL_PREFLIGHT_FILENAME = "original_confirmatory_final_preflight.json"
ORIGINAL_CONFIRMATORY_PREFLIGHT_CARRIER_CONTRACT = MappingProxyType(
    {
        "schema_version": 1,
        "policy": "original_confirmatory_preflight_carriers_v1",
        "receipt_schema_version": 1,
        "capacity_policy_sha256": ORIGINAL_CONFIRMATORY_CAPACITY_POLICY_SHA256,
        "initial_filename": ORIGINAL_CONFIRMATORY_INITIAL_PREFLIGHT_FILENAME,
        "initial_phase": "before_authority_or_data_loading",
        "final_filename": ORIGINAL_CONFIRMATORY_FINAL_PREFLIGHT_FILENAME,
        "final_phase": "guarded_immediately_before_tracker",
        "final_parent_field": "receipt_without_self_hash_sha256",
        "capacity_policy_binding_field": ("original_confirmatory_capacity_policy_sha256"),
        "initial_carrier_binding_field": ("original_confirmatory_initial_preflight_sha256"),
        "initial_receipt_binding_field": ("original_confirmatory_initial_preflight_receipt_sha256"),
        "final_carrier_binding_field": ("original_confirmatory_final_preflight_sha256"),
        "final_receipt_binding_field": ("original_confirmatory_final_preflight_receipt_sha256"),
        "carrier_hash_chain": ("carrier_file_sha256_and_inner_receipt_self_hash_v1"),
        "outcome_values_read": False,
        "adaptive_execution_changes_allowed": False,
    }
)
ORIGINAL_CONFIRMATORY_PREFLIGHT_CARRIER_CONTRACT_SHA256 = canonical_sha256(
    dict(ORIGINAL_CONFIRMATORY_PREFLIGHT_CARRIER_CONTRACT)
)
ORIGINAL_CONFIRMATORY_REQUIRED_FEATURE_CACHE_SCENARIOS = (
    "imagenet_frozen_context_morphometrics_logistic",
    "imagenet_frozen_logistic",
    "imagenet_frozen_target_highlighted_logistic",
)
ORIGINAL_CONFIRMATORY_REQUIRED_CACHE_ROLES = (
    "crop_cache",
    "crop_cache_metadata",
    *tuple(
        role
        for scenario_id in ORIGINAL_CONFIRMATORY_REQUIRED_FEATURE_CACHE_SCENARIOS
        for role in (
            f"frozen_feature_cache:{scenario_id}",
            f"frozen_feature_metadata:{scenario_id}",
        )
    ),
    "official_resnet18_weights",
)

type OriginalConfirmatoryPreflightPhase = Literal[
    "before_authority_or_data_loading",
    "guarded_immediately_before_tracker",
]

_HEX = frozenset("0123456789abcdef")
_COMPUTE_FIELDS = frozenset(
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
        "grad_scaler_available",
        "grad_scaler_enabled",
        "weight_identifier",
        "weight_path",
        "weight_present",
        "weight_sha256",
        "smoke_attempted",
        "smoke_completed",
        "smoke_input_shapes",
        "smoke_forward_finite",
        "smoke_backward_finite",
        "smoke_peak_allocated_bytes",
        "smoke_error",
    }
)


class OriginalConfirmatoryPreflightError(RuntimeError):
    """The original confirmatory resource contract failed closed."""


class _DiskUsage(Protocol):
    free: int


def _default_disk_usage(path: Path) -> _DiskUsage:
    return cast(_DiskUsage, shutil.disk_usage(path))


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryCacheBinding:
    """One explicit immutable input/cache file binding."""

    role: str
    path: Path
    expected_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": str(self.path),
            "expected_sha256": self.expected_sha256,
        }


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryCacheEvidence:
    """One physical, private cache file verified during a preflight phase."""

    role: str
    path: Path
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryComputeEvidence:
    """Closed CUDA/AMP/RAM observation for one preflight phase."""

    phase: OriginalConfirmatoryPreflightPhase
    minimum_available_ram_bytes: int
    observation: Mapping[str, Any]
    observation_sha256: str
    policy_sha256: str
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        observation = dict(self.observation)
        observation["smoke_input_shapes"] = [
            list(value)
            for value in cast(tuple[tuple[int, ...], ...], observation["smoke_input_shapes"])
        ]
        return {
            "phase": self.phase,
            "minimum_available_ram_bytes": self.minimum_available_ram_bytes,
            "observation": observation,
            "observation_sha256": self.observation_sha256,
            "policy_sha256": self.policy_sha256,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryDiskEvidence:
    """One disk-capacity observation against the sealed original plan."""

    phase: OriginalConfirmatoryPreflightPhase
    probe_path: Path
    free_bytes: int
    projected_active_bytes: int
    safety_margin_bytes: int
    minimum_free_bytes: int
    policy_sha256: str
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "probe_path": str(self.probe_path),
            "free_bytes": self.free_bytes,
            "projected_active_bytes": self.projected_active_bytes,
            "safety_margin_bytes": self.safety_margin_bytes,
            "minimum_free_bytes": self.minimum_free_bytes,
            "policy_sha256": self.policy_sha256,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryPreflightReceipt:
    """Self-hashed, outcome-blind original-confirmatory preflight receipt."""

    phase: OriginalConfirmatoryPreflightPhase
    policy_sha256: str
    plan_semantic_sha256: str
    config_semantic_sha256: str
    controls_binding_sha256: str
    planned_cell_count: int
    planned_required_cell_count: int
    planned_optional_cell_count: int
    planned_cnn_cell_count: int
    planned_cnn_fold_checkpoint_count: int
    cache_evidence: tuple[OriginalConfirmatoryCacheEvidence, ...]
    cache_manifest_sha256: str
    compute_evidence: OriginalConfirmatoryComputeEvidence
    disk_evidence: OriginalConfirmatoryDiskEvidence
    parent_receipt_sha256: str | None
    checked_at_utc: str
    outcome_values_read: Literal[False]
    adaptive_execution_changes_allowed: Literal[False]
    receipt_without_self_hash_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "phase": self.phase,
            "policy": dict(ORIGINAL_CONFIRMATORY_CAPACITY_POLICY),
            "policy_sha256": self.policy_sha256,
            "plan_semantic_sha256": self.plan_semantic_sha256,
            "config_semantic_sha256": self.config_semantic_sha256,
            "controls_binding_sha256": self.controls_binding_sha256,
            "planned_cell_count": self.planned_cell_count,
            "planned_required_cell_count": self.planned_required_cell_count,
            "planned_optional_cell_count": self.planned_optional_cell_count,
            "planned_cnn_cell_count": self.planned_cnn_cell_count,
            "planned_cnn_fold_checkpoint_count": self.planned_cnn_fold_checkpoint_count,
            "cache_evidence": [value.as_dict() for value in self.cache_evidence],
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "compute_evidence": self.compute_evidence.as_dict(),
            "disk_evidence": self.disk_evidence.as_dict(),
            "parent_receipt_sha256": self.parent_receipt_sha256,
            "checked_at_utc": self.checked_at_utc,
            "outcome_values_read": self.outcome_values_read,
            "adaptive_execution_changes_allowed": self.adaptive_execution_changes_allowed,
            "receipt_without_self_hash_sha256": self.receipt_without_self_hash_sha256,
        }


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value).issubset(_HEX)


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_nlink),
        stat.S_IFMT(value.st_mode),
    )


def _verify_cache_bindings(
    bindings: Sequence[OriginalConfirmatoryCacheBinding],
    *,
    official_weight_binding: Callable[[], tuple[Path, str]],
) -> tuple[tuple[OriginalConfirmatoryCacheEvidence, ...], Path]:
    if not bindings:
        raise OriginalConfirmatoryPreflightError(
            "original confirmatory preflight requires explicit cache bindings"
        )
    try:
        weight_path, weight_sha256 = official_weight_binding()
    except (OriginalConfirmatoryResumeError, OSError, ValueError, TypeError) as exc:
        raise OriginalConfirmatoryPreflightError(
            "official ResNet-18 weight binding failed closed"
        ) from exc
    if weight_sha256 != ORIGINAL_CONFIRMATORY_WEIGHT_SHA256:
        raise OriginalConfirmatoryPreflightError(
            "official ResNet-18 weight differs from the frozen original authority"
        )
    all_bindings = (
        *bindings,
        OriginalConfirmatoryCacheBinding(
            role="official_resnet18_weights",
            path=weight_path,
            expected_sha256=weight_sha256,
        ),
    )
    supplied_roles = tuple(
        binding.role
        for binding in all_bindings
        if isinstance(binding, OriginalConfirmatoryCacheBinding)
    )
    if len(supplied_roles) != len(all_bindings) or tuple(sorted(supplied_roles)) != tuple(
        sorted(ORIGINAL_CONFIRMATORY_REQUIRED_CACHE_ROLES)
    ):
        raise OriginalConfirmatoryPreflightError(
            "original confirmatory cache bindings do not cover the exact required "
            "crop/three-feature/weight universe"
        )
    bindings_by_role = {
        binding.role: binding
        for binding in all_bindings
        if isinstance(binding, OriginalConfirmatoryCacheBinding)
    }
    cache_sidecar_pairs = (
        ("crop_cache", "crop_cache_metadata"),
        *tuple(
            (
                f"frozen_feature_cache:{scenario_id}",
                f"frozen_feature_metadata:{scenario_id}",
            )
            for scenario_id in ORIGINAL_CONFIRMATORY_REQUIRED_FEATURE_CACHE_SCENARIOS
        ),
    )
    for cache_role, metadata_role in cache_sidecar_pairs:
        cache_path = bindings_by_role[cache_role].path
        expected_metadata = cache_path.with_suffix(f"{cache_path.suffix}.metadata.json")
        if os.path.normcase(str(bindings_by_role[metadata_role].path)) != os.path.normcase(
            str(expected_metadata)
        ):
            raise OriginalConfirmatoryPreflightError(
                f"original confirmatory cache sidecar path is noncanonical: {cache_role}"
            )
    roles: set[str] = set()
    lexical_paths: set[str] = set()
    file_identities: set[tuple[int, int]] = set()
    records: list[OriginalConfirmatoryCacheEvidence] = []
    for binding in all_bindings:
        if not isinstance(binding, OriginalConfirmatoryCacheBinding):
            raise TypeError("cache bindings must use OriginalConfirmatoryCacheBinding")
        if (
            not binding.role
            or binding.role != binding.role.strip()
            or binding.role.casefold() in roles
            or not binding.path.is_absolute()
            or not _valid_sha256(binding.expected_sha256)
        ):
            raise OriginalConfirmatoryPreflightError(
                "original confirmatory cache binding is incomplete or duplicated"
            )
        roles.add(binding.role.casefold())
        path = Path(os.path.abspath(binding.path))
        lexical_key = os.path.normcase(str(path))
        if lexical_key in lexical_paths:
            raise OriginalConfirmatoryPreflightError(
                "original confirmatory cache roles alias one lexical path"
            )
        lexical_paths.add(lexical_key)
        try:
            _plain_directory_identity(path.parent, role=f"{binding.role} parent directory")
            before = path.lstat()
            digest = _hash_runtime_file(path, role=binding.role)
            after = path.lstat()
        except (OriginalConfirmatoryResumeError, OSError) as exc:
            raise OriginalConfirmatoryPreflightError(
                f"original confirmatory cache is not one private physical file: {binding.role}"
            ) from exc
        if (
            _file_identity(before) != _file_identity(after)
            or digest != binding.expected_sha256
            or int(after.st_size) <= 0
        ):
            raise OriginalConfirmatoryPreflightError(
                f"original confirmatory cache changed or differs from its hash: {binding.role}"
            )
        physical_identity = (int(after.st_dev), int(after.st_ino))
        if physical_identity in file_identities:
            raise OriginalConfirmatoryPreflightError(
                "original confirmatory cache roles alias one physical file"
            )
        file_identities.add(physical_identity)
        records.append(
            OriginalConfirmatoryCacheEvidence(
                role=binding.role,
                path=path,
                sha256=digest,
                size_bytes=int(after.st_size),
            )
        )
    return tuple(sorted(records, key=lambda value: value.role)), Path(weight_path)


def _default_compute_probe(weight_path: Path) -> Mapping[str, Any]:
    import psutil  # type: ignore[import-untyped]
    import torch
    from torch import nn
    from torchvision.models import resnet18  # type: ignore[import-untyped]

    policy = ORIGINAL_CONFIRMATORY_CAPACITY_POLICY
    memory = psutil.virtual_memory()
    torch_runtime = cast(Any, torch)
    cuda_available = bool(torch.cuda.is_available())
    device_index = cast(int, policy["cuda_device_index"])
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    device_name: str | None = None
    free_vram = 0
    total_vram = 0
    cudnn_available = bool(
        cuda_available
        and torch_runtime.backends.cudnn.is_available()
        and torch_runtime.backends.cudnn.enabled
    )
    amp_available = bool(
        cuda_available
        and hasattr(torch, "autocast")
        and hasattr(torch, "amp")
        and hasattr(torch_runtime.amp, "GradScaler")
    )
    if cuda_available and device_count > device_index:
        free_value, total_value = torch.cuda.mem_get_info(device_index)
        free_vram = int(free_value)
        total_vram = int(total_value)
        device_name = str(torch.cuda.get_device_name(device_index))

    smoke_attempted = False
    smoke_completed = False
    smoke_forward_finite = False
    smoke_backward_finite = False
    smoke_peak_allocated_bytes = 0
    smoke_error: str | None = None
    grad_scaler_available = bool(amp_available)
    grad_scaler_enabled = False
    if (
        cuda_available
        and device_count > device_index
        and cudnn_available
        and amp_available
        and weight_path.is_file()
    ):
        smoke_attempted = True
        network: Any = None
        state_dict: Any = None
        optimiser: Any = None
        scaler: Any = None
        inputs: Any = None
        outputs: Any = None
        loss: Any = None
        gradients: Any = None
        try:
            device = torch.device("cuda", device_index)
            state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
            forward_results: list[bool] = []
            backward_results: list[bool] = []
            for raw_shape in cast(
                tuple[tuple[int, ...], ...],
                policy["cuda_smoke_input_shapes"],
            ):
                network = resnet18(weights=None)
                network.load_state_dict(state_dict, strict=True)
                network.fc = nn.Linear(network.fc.in_features, 5)
                if raw_shape[1] == 4:
                    original = network.conv1
                    expanded = nn.Conv2d(
                        4,
                        original.out_channels,
                        kernel_size=original.kernel_size,
                        stride=original.stride,
                        padding=original.padding,
                        bias=original.bias is not None,
                    )
                    with torch.no_grad():
                        expanded.weight[:, :3].copy_(original.weight)
                        expanded.weight[:, 3:4].zero_()
                    network.conv1 = expanded
                network.train()
                network.to(device)
                optimiser = torch.optim.AdamW(network.parameters(), lr=1e-4)
                scaler = torch_runtime.amp.GradScaler(
                    "cuda",
                    enabled=True,
                )
                grad_scaler_enabled = bool(scaler.is_enabled())
                torch.cuda.reset_peak_memory_stats(device)
                inputs = torch.ones(raw_shape, device=device, dtype=torch.float32)
                optimiser.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    outputs = network(inputs)
                    loss = outputs.float().square().mean()
                forward_results.append(
                    bool(torch.isfinite(outputs).all().item() and torch.isfinite(loss).item())
                )
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                gradients = [
                    parameter.grad
                    for parameter in network.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                backward_results.append(
                    bool(
                        gradients
                        and all(torch.isfinite(gradient).all().item() for gradient in gradients)
                    )
                )
                scaler.step(optimiser)
                scaler.update()
                torch.cuda.synchronize(device)
                smoke_peak_allocated_bytes = max(
                    smoke_peak_allocated_bytes,
                    int(torch.cuda.max_memory_allocated(device)),
                )
                gradients = None
                loss = None
                outputs = None
                inputs = None
                scaler = None
                optimiser = None
                network = None
                torch.cuda.empty_cache()
            smoke_forward_finite = bool(forward_results and all(forward_results))
            smoke_backward_finite = bool(backward_results and all(backward_results))
            smoke_completed = True
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            smoke_error = f"{type(exc).__name__}: {exc}"
        finally:
            gradients = None
            loss = None
            outputs = None
            inputs = None
            scaler = None
            optimiser = None
            network = None
            state_dict = None
            torch.cuda.empty_cache()
    weight_sha256: str | None = None
    if weight_path.is_file():
        try:
            weight_sha256 = _hash_runtime_file(
                weight_path,
                role="official ResNet-18 weight compute probe",
            )
        except OriginalConfirmatoryResumeError:
            weight_sha256 = None
    return {
        "total_host_ram_bytes": int(memory.total),
        "available_host_ram_bytes": int(memory.available),
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "selected_cuda_device_index": device_index,
        "selected_cuda_device_name": device_name,
        "total_vram_bytes": total_vram,
        "free_vram_bytes": free_vram,
        "cudnn_available": cudnn_available,
        "amp_available": amp_available,
        "amp_dtype": "float16",
        "grad_scaler_available": grad_scaler_available,
        "grad_scaler_enabled": grad_scaler_enabled,
        "weight_identifier": OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER,
        "weight_path": str(weight_path),
        "weight_present": weight_path.is_file(),
        "weight_sha256": weight_sha256,
        "smoke_attempted": smoke_attempted,
        "smoke_completed": smoke_completed,
        "smoke_input_shapes": [
            list(value)
            for value in cast(
                tuple[tuple[int, ...], ...],
                policy["cuda_smoke_input_shapes"],
            )
        ],
        "smoke_forward_finite": smoke_forward_finite,
        "smoke_backward_finite": smoke_backward_finite,
        "smoke_peak_allocated_bytes": smoke_peak_allocated_bytes,
        "smoke_error": smoke_error,
    }


def _normalise_and_evaluate_compute_observation(
    phase: OriginalConfirmatoryPreflightPhase,
    *,
    weight_path: Path,
    raw_observation: Mapping[str, Any],
) -> tuple[Mapping[str, Any], int, bool]:
    policy = ORIGINAL_CONFIRMATORY_CAPACITY_POLICY
    observed = dict(raw_observation)
    if set(observed) != _COMPUTE_FIELDS:
        raise OriginalConfirmatoryPreflightError(
            "original confirmatory compute evidence has a non-contract field set"
        )
    integer_fields = (
        "total_host_ram_bytes",
        "available_host_ram_bytes",
        "cuda_device_count",
        "selected_cuda_device_index",
        "total_vram_bytes",
        "free_vram_bytes",
        "smoke_peak_allocated_bytes",
    )
    boolean_fields = (
        "cuda_available",
        "cudnn_available",
        "amp_available",
        "grad_scaler_available",
        "grad_scaler_enabled",
        "weight_present",
        "smoke_attempted",
        "smoke_completed",
        "smoke_forward_finite",
        "smoke_backward_finite",
    )
    if any(
        type(observed.get(field)) is not int or cast(int, observed[field]) < 0
        for field in integer_fields
    ) or any(type(observed.get(field)) is not bool for field in boolean_fields):
        raise OriginalConfirmatoryPreflightError(
            "original confirmatory compute evidence has invalid field types"
        )
    raw_shapes = observed.get("smoke_input_shapes")
    if not isinstance(raw_shapes, (list, tuple)) or any(
        not isinstance(shape, (list, tuple)) or any(type(value) is not int for value in shape)
        for shape in raw_shapes
    ):
        raise OriginalConfirmatoryPreflightError(
            "original confirmatory compute smoke shapes are invalid"
        )
    shapes = tuple(tuple(cast(Sequence[int], shape)) for shape in raw_shapes)
    observed["smoke_input_shapes"] = shapes
    minimum_available_field = (
        "minimum_available_ram_bytes_before_data"
        if phase == "before_authority_or_data_loading"
        else "minimum_available_ram_bytes_before_tracker"
    )
    minimum_available = cast(int, policy[minimum_available_field])
    passed = bool(
        observed["total_host_ram_bytes"] >= policy["minimum_total_ram_bytes"]
        and observed["available_host_ram_bytes"] >= minimum_available
        and observed["cuda_available"] is policy["cuda_required"]
        and observed["cuda_device_count"] >= 1
        and observed["selected_cuda_device_index"] == policy["cuda_device_index"]
        and isinstance(observed["selected_cuda_device_name"], str)
        and bool(observed["selected_cuda_device_name"].strip())
        and observed["total_vram_bytes"] >= policy["minimum_total_vram_bytes"]
        and observed["free_vram_bytes"] >= policy["minimum_free_vram_bytes"]
        and observed["cudnn_available"] is policy["cudnn_required"]
        and observed["amp_available"] is policy["amp_required"]
        and observed["amp_dtype"] == policy["amp_dtype"]
        and observed["grad_scaler_available"] is policy["grad_scaler_required"]
        and observed["grad_scaler_enabled"] is policy["grad_scaler_required"]
        and observed["weight_identifier"] == policy["official_weight_identifier"]
        and observed["weight_path"] == str(weight_path)
        and observed["weight_present"] is True
        and observed["weight_sha256"] == policy["official_weight_sha256"]
        and observed["smoke_attempted"] is True
        and observed["smoke_completed"] is True
        and shapes == policy["cuda_smoke_input_shapes"]
        and observed["smoke_forward_finite"] is True
        and observed["smoke_backward_finite"] is True
        and 0
        < observed["smoke_peak_allocated_bytes"]
        <= policy["cuda_smoke_max_peak_allocated_bytes"]
        and observed["smoke_error"] is None
        and policy["implicit_weight_download_allowed"] is False
    )
    return MappingProxyType(observed), minimum_available, passed


def _compute_evidence(
    phase: OriginalConfirmatoryPreflightPhase,
    *,
    weight_path: Path,
    compute_probe: Callable[[Path], Mapping[str, Any]],
) -> OriginalConfirmatoryComputeEvidence:
    normalized, minimum_available, passed = _normalise_and_evaluate_compute_observation(
        phase,
        weight_path=weight_path,
        raw_observation=compute_probe(weight_path),
    )
    evidence = OriginalConfirmatoryComputeEvidence(
        phase=phase,
        minimum_available_ram_bytes=minimum_available,
        observation=normalized,
        observation_sha256=canonical_sha256(dict(normalized)),
        policy_sha256=ORIGINAL_CONFIRMATORY_CAPACITY_POLICY_SHA256,
        passed=passed,
    )
    if not passed:
        raise OriginalConfirmatoryPreflightError(
            "original confirmatory CUDA/AMP/cache/RAM smoke preflight failed closed"
        )
    return evidence


def _nearest_existing_probe(target: Path) -> Path:
    probe = Path(os.path.abspath(target))
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            raise OriginalConfirmatoryPreflightError(
                "original confirmatory capacity target has no existing ancestor"
            )
        probe = parent
    return probe if probe.is_dir() else probe.parent


def _disk_evidence(
    phase: OriginalConfirmatoryPreflightPhase,
    target: Path,
    *,
    disk_usage: Callable[[Path], _DiskUsage],
) -> OriginalConfirmatoryDiskEvidence:
    policy = ORIGINAL_CONFIRMATORY_CAPACITY_POLICY
    probe = _nearest_existing_probe(target)
    usage = disk_usage(probe)
    free = getattr(usage, "free", None)
    if type(free) is not int or free < 0:
        raise OriginalConfirmatoryPreflightError(
            "original confirmatory disk provider returned invalid free bytes"
        )
    copy_count = cast(int, policy["checkpoint_physical_copy_count"])
    per_copy = cast(int, policy["projected_checkpoint_bytes_per_physical_copy"])
    projected = cast(int, policy["projected_all_checkpoint_copies_bytes"])
    margin = cast(int, policy["fixed_safety_margin_bytes"])
    minimum = cast(int, policy["minimum_free_bytes"])
    if (
        policy["policy"] != "original_confirmatory_sealed_plan_capacity_v2"
        or policy["checkpoint_publication_policy"]
        != "canonical_plus_distinct_versioned_o_excl_physical_copy_v1"
        or policy["planned_cnn_fold_checkpoint_count"] != ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT
        or copy_count != 2
        or per_copy != 30 * _GIB
        or projected != copy_count * per_copy
        or projected + margin != minimum
        or margin != 10 * _GIB
    ):
        raise OriginalConfirmatoryPreflightError(
            "original confirmatory sealed capacity arithmetic is invalid"
        )
    evidence = OriginalConfirmatoryDiskEvidence(
        phase=phase,
        probe_path=probe,
        free_bytes=free,
        projected_active_bytes=projected,
        safety_margin_bytes=margin,
        minimum_free_bytes=minimum,
        policy_sha256=ORIGINAL_CONFIRMATORY_CAPACITY_POLICY_SHA256,
        passed=free >= minimum,
    )
    if not evidence.passed:
        raise OriginalConfirmatoryPreflightError(
            "original confirmatory disk capacity is below the sealed 60-GiB "
            "canonical-plus-versioned checkpoint plan plus 10-GiB margin: "
            f"free={free}, required={minimum}"
        )
    return evidence


def _build_receipt(
    *,
    phase: OriginalConfirmatoryPreflightPhase,
    controls: ConfirmatoryExecutionControls,
    target: Path,
    cache_bindings: Sequence[OriginalConfirmatoryCacheBinding],
    parent_receipt_sha256: str | None,
    compute_probe: Callable[[Path], Mapping[str, Any]],
    disk_usage: Callable[[Path], _DiskUsage],
    clock: Callable[[], str],
    official_weight_binding: Callable[[], tuple[Path, str]],
) -> OriginalConfirmatoryPreflightReceipt:
    try:
        _require_exact_original_controls(controls)
    except (OriginalConfirmatoryResumeError, TypeError, ValueError) as exc:
        raise OriginalConfirmatoryPreflightError(
            "preflight controls differ from the exact frozen 108/90/18 and 180-fit plan"
        ) from exc
    records, weight_path = _verify_cache_bindings(
        cache_bindings,
        official_weight_binding=official_weight_binding,
    )
    compute = _compute_evidence(
        phase,
        weight_path=weight_path,
        compute_probe=compute_probe,
    )
    disk = _disk_evidence(phase, target, disk_usage=disk_usage)
    checked_at = clock()
    if not isinstance(checked_at, str) or not checked_at.strip():
        raise OriginalConfirmatoryPreflightError("preflight clock returned an invalid timestamp")
    cache_manifest_sha256 = canonical_sha256([value.as_dict() for value in records])
    partial = OriginalConfirmatoryPreflightReceipt(
        phase=phase,
        policy_sha256=ORIGINAL_CONFIRMATORY_CAPACITY_POLICY_SHA256,
        plan_semantic_sha256=ORIGINAL_CONFIRMATORY_PLAN_SEMANTIC_SHA256,
        config_semantic_sha256=ORIGINAL_CONFIRMATORY_CONFIG_SEMANTIC_SHA256,
        controls_binding_sha256=ORIGINAL_CONFIRMATORY_CONTROLS_BINDING_SHA256,
        planned_cell_count=ORIGINAL_CONFIRMATORY_PLANNED_CELL_COUNT,
        planned_required_cell_count=ORIGINAL_CONFIRMATORY_REQUIRED_CELL_COUNT,
        planned_optional_cell_count=ORIGINAL_CONFIRMATORY_OPTIONAL_CELL_COUNT,
        planned_cnn_cell_count=ORIGINAL_CONFIRMATORY_CNN_CELL_COUNT,
        planned_cnn_fold_checkpoint_count=ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT,
        cache_evidence=records,
        cache_manifest_sha256=cache_manifest_sha256,
        compute_evidence=compute,
        disk_evidence=disk,
        parent_receipt_sha256=parent_receipt_sha256,
        checked_at_utc=checked_at,
        outcome_values_read=False,
        adaptive_execution_changes_allowed=False,
        receipt_without_self_hash_sha256="",
    )
    payload = partial.as_dict()
    payload.pop("receipt_without_self_hash_sha256")
    return replace(
        partial,
        receipt_without_self_hash_sha256=canonical_sha256(payload),
    )


def validate_original_confirmatory_preflight_receipt(
    receipt: OriginalConfirmatoryPreflightReceipt,
    controls: ConfirmatoryExecutionControls,
) -> None:
    """Validate a receipt structurally without reusing it as execution authority."""

    if not isinstance(receipt, OriginalConfirmatoryPreflightReceipt):
        raise TypeError("preflight receipt must use OriginalConfirmatoryPreflightReceipt")
    try:
        _require_exact_original_controls(controls)
    except (OriginalConfirmatoryResumeError, TypeError, ValueError) as exc:
        raise OriginalConfirmatoryPreflightError(
            "receipt controls differ from the exact frozen original plan"
        ) from exc
    expected_counts = (
        ORIGINAL_CONFIRMATORY_PLANNED_CELL_COUNT,
        ORIGINAL_CONFIRMATORY_REQUIRED_CELL_COUNT,
        ORIGINAL_CONFIRMATORY_OPTIONAL_CELL_COUNT,
        ORIGINAL_CONFIRMATORY_CNN_CELL_COUNT,
        ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT,
    )
    observed_counts = (
        receipt.planned_cell_count,
        receipt.planned_required_cell_count,
        receipt.planned_optional_cell_count,
        receipt.planned_cnn_cell_count,
        receipt.planned_cnn_fold_checkpoint_count,
    )
    cache_roles = tuple(value.role for value in receipt.cache_evidence)
    cache_records_by_role = {value.role: value for value in receipt.cache_evidence}
    cache_sidecar_pairs = (
        ("crop_cache", "crop_cache_metadata"),
        *tuple(
            (
                f"frozen_feature_cache:{scenario_id}",
                f"frozen_feature_metadata:{scenario_id}",
            )
            for scenario_id in ORIGINAL_CONFIRMATORY_REQUIRED_FEATURE_CACHE_SCENARIOS
        ),
    )
    sidecars_canonical = all(
        cache_role in cache_records_by_role
        and metadata_role in cache_records_by_role
        and os.path.normcase(str(cache_records_by_role[metadata_role].path))
        == os.path.normcase(
            str(
                cache_records_by_role[cache_role].path.with_suffix(
                    f"{cache_records_by_role[cache_role].path.suffix}.metadata.json"
                )
            )
        )
        for cache_role, metadata_role in cache_sidecar_pairs
    )
    weight_records = tuple(
        value for value in receipt.cache_evidence if value.role == "official_resnet18_weights"
    )
    if len(weight_records) != 1:
        raise OriginalConfirmatoryPreflightError(
            "original confirmatory preflight receipt lacks one exact weight record"
        )
    normalized_compute, minimum_available, compute_passed = (
        _normalise_and_evaluate_compute_observation(
            receipt.phase,
            weight_path=weight_records[0].path,
            raw_observation=receipt.compute_evidence.observation,
        )
    )
    expected_projected = cast(
        int,
        ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["projected_all_checkpoint_copies_bytes"],
    )
    expected_margin = cast(
        int,
        ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["fixed_safety_margin_bytes"],
    )
    expected_minimum = cast(
        int,
        ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["minimum_free_bytes"],
    )
    try:
        checked_at = datetime.fromisoformat(receipt.checked_at_utc)
    except ValueError as exc:
        raise OriginalConfirmatoryPreflightError(
            "original confirmatory preflight receipt timestamp is invalid"
        ) from exc
    if checked_at.utcoffset() is None:
        raise OriginalConfirmatoryPreflightError(
            "original confirmatory preflight timestamp must be timezone-aware"
        )
    if (
        receipt.phase
        not in {
            "before_authority_or_data_loading",
            "guarded_immediately_before_tracker",
        }
        or receipt.policy_sha256 != ORIGINAL_CONFIRMATORY_CAPACITY_POLICY_SHA256
        or receipt.plan_semantic_sha256 != ORIGINAL_CONFIRMATORY_PLAN_SEMANTIC_SHA256
        or receipt.config_semantic_sha256 != ORIGINAL_CONFIRMATORY_CONFIG_SEMANTIC_SHA256
        or receipt.controls_binding_sha256 != ORIGINAL_CONFIRMATORY_CONTROLS_BINDING_SHA256
        or observed_counts != expected_counts
        or tuple(sorted(cache_roles)) != tuple(sorted(ORIGINAL_CONFIRMATORY_REQUIRED_CACHE_ROLES))
        or not sidecars_canonical
        or tuple(sorted(receipt.cache_evidence, key=lambda value: value.role))
        != receipt.cache_evidence
        or len({value.role.casefold() for value in receipt.cache_evidence})
        != len(receipt.cache_evidence)
        or len({os.path.normcase(str(value.path)) for value in receipt.cache_evidence})
        != len(receipt.cache_evidence)
        or any(
            not isinstance(value, OriginalConfirmatoryCacheEvidence)
            or not value.path.is_absolute()
            or not _valid_sha256(value.sha256)
            or type(value.size_bytes) is not int
            or value.size_bytes <= 0
            for value in receipt.cache_evidence
        )
        or receipt.cache_manifest_sha256
        != canonical_sha256([value.as_dict() for value in receipt.cache_evidence])
        or receipt.compute_evidence.phase != receipt.phase
        or receipt.compute_evidence.policy_sha256 != ORIGINAL_CONFIRMATORY_CAPACITY_POLICY_SHA256
        or receipt.compute_evidence.minimum_available_ram_bytes != minimum_available
        or dict(receipt.compute_evidence.observation) != dict(normalized_compute)
        or receipt.compute_evidence.observation_sha256
        != canonical_sha256(dict(receipt.compute_evidence.observation))
        or receipt.compute_evidence.passed is not compute_passed
        or compute_passed is not True
        or receipt.disk_evidence.phase != receipt.phase
        or receipt.disk_evidence.policy_sha256 != ORIGINAL_CONFIRMATORY_CAPACITY_POLICY_SHA256
        or not receipt.disk_evidence.probe_path.is_absolute()
        or receipt.disk_evidence.projected_active_bytes != expected_projected
        or receipt.disk_evidence.safety_margin_bytes != expected_margin
        or receipt.disk_evidence.minimum_free_bytes != expected_minimum
        or type(receipt.disk_evidence.free_bytes) is not int
        or receipt.disk_evidence.free_bytes < 0
        or receipt.disk_evidence.passed
        is not (receipt.disk_evidence.free_bytes >= expected_minimum)
        or receipt.disk_evidence.passed is not True
        or receipt.outcome_values_read is not False
        or receipt.adaptive_execution_changes_allowed is not False
        or not _valid_sha256(receipt.receipt_without_self_hash_sha256)
    ):
        raise OriginalConfirmatoryPreflightError(
            "original confirmatory preflight receipt is incomplete or altered"
        )
    if (
        receipt.phase == "before_authority_or_data_loading"
        and receipt.parent_receipt_sha256 is not None
    ) or (
        receipt.phase == "guarded_immediately_before_tracker"
        and not _valid_sha256(receipt.parent_receipt_sha256)
    ):
        raise OriginalConfirmatoryPreflightError(
            "original confirmatory preflight receipt lineage is invalid"
        )
    payload = receipt.as_dict()
    observed_self_hash = cast(str, payload.pop("receipt_without_self_hash_sha256"))
    if canonical_sha256(payload) != observed_self_hash:
        raise OriginalConfirmatoryPreflightError(
            "original confirmatory preflight receipt self-hash differs"
        )


def validate_original_confirmatory_preflight_pair(
    initial_receipt: OriginalConfirmatoryPreflightReceipt,
    final_receipt: OriginalConfirmatoryPreflightReceipt,
    controls: ConfirmatoryExecutionControls,
) -> None:
    """Independently validate the exact initial-to-final receipt lineage."""

    validate_original_confirmatory_preflight_receipt(initial_receipt, controls)
    validate_original_confirmatory_preflight_receipt(final_receipt, controls)
    initial_time = datetime.fromisoformat(initial_receipt.checked_at_utc)
    final_time = datetime.fromisoformat(final_receipt.checked_at_utc)
    if (
        initial_receipt.phase != "before_authority_or_data_loading"
        or final_receipt.phase != "guarded_immediately_before_tracker"
        or final_receipt.parent_receipt_sha256 != initial_receipt.receipt_without_self_hash_sha256
        or final_receipt.cache_manifest_sha256 != initial_receipt.cache_manifest_sha256
        or final_receipt.cache_evidence != initial_receipt.cache_evidence
        or os.path.normcase(str(final_receipt.disk_evidence.probe_path))
        != os.path.normcase(str(initial_receipt.disk_evidence.probe_path))
        or final_time <= initial_time
    ):
        raise OriginalConfirmatoryPreflightError(
            "original confirmatory final preflight does not exactly descend from "
            "the initial cache/volume receipt"
        )


def require_original_confirmatory_preflight(
    controls: ConfirmatoryExecutionControls,
    *,
    target: str | Path,
    cache_bindings: Sequence[OriginalConfirmatoryCacheBinding],
    compute_probe: Callable[[Path], Mapping[str, Any]] = _default_compute_probe,
    disk_usage: Callable[[Path], _DiskUsage] = _default_disk_usage,
    clock: Callable[[], str] = _utc_now_iso,
    official_weight_binding: Callable[[], tuple[Path, str]] = _official_weight_binding,
) -> OriginalConfirmatoryPreflightReceipt:
    """Run the initial full preflight before authority, data loading, or RunTracker."""

    receipt = _build_receipt(
        phase="before_authority_or_data_loading",
        controls=controls,
        target=Path(target),
        cache_bindings=cache_bindings,
        parent_receipt_sha256=None,
        compute_probe=compute_probe,
        disk_usage=disk_usage,
        clock=clock,
        official_weight_binding=official_weight_binding,
    )
    validate_original_confirmatory_preflight_receipt(receipt, controls)
    return receipt


def recheck_original_confirmatory_capacity(
    initial_receipt: OriginalConfirmatoryPreflightReceipt,
    controls: ConfirmatoryExecutionControls,
    *,
    target: str | Path,
    cache_bindings: Sequence[OriginalConfirmatoryCacheBinding],
    compute_probe: Callable[[Path], Mapping[str, Any]] = _default_compute_probe,
    disk_usage: Callable[[Path], _DiskUsage] = _default_disk_usage,
    clock: Callable[[], str] = _utc_now_iso,
    official_weight_binding: Callable[[], tuple[Path, str]] = _official_weight_binding,
) -> OriginalConfirmatoryPreflightReceipt:
    """Recompute cache/compute/disk capacity immediately before RunTracker."""

    validate_original_confirmatory_preflight_receipt(initial_receipt, controls)
    if initial_receipt.phase != "before_authority_or_data_loading":
        raise OriginalConfirmatoryPreflightError(
            "final capacity recheck requires the initial pre-authority receipt"
        )
    receipt = _build_receipt(
        phase="guarded_immediately_before_tracker",
        controls=controls,
        target=Path(target),
        cache_bindings=cache_bindings,
        parent_receipt_sha256=initial_receipt.receipt_without_self_hash_sha256,
        compute_probe=compute_probe,
        disk_usage=disk_usage,
        clock=clock,
        official_weight_binding=official_weight_binding,
    )
    validate_original_confirmatory_preflight_pair(initial_receipt, receipt, controls)
    return receipt


__all__ = [
    "ORIGINAL_CONFIRMATORY_CAPACITY_POLICY",
    "ORIGINAL_CONFIRMATORY_CAPACITY_POLICY_SHA256",
    "ORIGINAL_CONFIRMATORY_FINAL_PREFLIGHT_FILENAME",
    "ORIGINAL_CONFIRMATORY_INITIAL_PREFLIGHT_FILENAME",
    "ORIGINAL_CONFIRMATORY_PREFLIGHT_CARRIER_CONTRACT",
    "ORIGINAL_CONFIRMATORY_PREFLIGHT_CARRIER_CONTRACT_SHA256",
    "OriginalConfirmatoryCacheBinding",
    "OriginalConfirmatoryCacheEvidence",
    "OriginalConfirmatoryComputeEvidence",
    "OriginalConfirmatoryDiskEvidence",
    "OriginalConfirmatoryPreflightError",
    "OriginalConfirmatoryPreflightReceipt",
    "recheck_original_confirmatory_capacity",
    "require_original_confirmatory_preflight",
    "validate_original_confirmatory_preflight_pair",
    "validate_original_confirmatory_preflight_receipt",
]
