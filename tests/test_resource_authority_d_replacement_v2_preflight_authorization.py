"""Hermetic security tests for replacement-v2 preflight authorization."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from histo_audit.workflows import (
    preregistration_amendment as amendment,
)
from histo_audit.workflows import (
    resource_authority_d_replacement_v2_controller as controller,
)

_PROPOSED_AT = datetime(2026, 7, 28, 20, 30, tzinfo=UTC)
_AUTHORIZED_AT = _PROPOSED_AT + timedelta(seconds=2)


def _timestamp(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _record(path: Path, label: str) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": len(label.encode("utf-8")) or 1,
        "sha256": _sha(label),
    }


def _set_path(payload: dict[str, Any], keys: tuple[str | int, ...], value: Any) -> None:
    current: Any = payload
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


def _refresh_preflight_fingerprint(payload: dict[str, Any]) -> None:
    payload["preflight"]["preflight_fingerprint_sha256"] = controller._compact_sha256(
        payload["preflight"]["contract"]
    )


def _refresh_technical_authorization(payload: dict[str, Any]) -> None:
    technical = payload["preflight"]["contract"]["technical_successor"]
    technical["authorization_sha256"] = controller._compact_sha256(technical["authorization"])
    _refresh_preflight_fingerprint(payload)


def _refresh_compute_probe(payload: dict[str, Any], index: int) -> None:
    observation = payload["preflight"]["compute_observations"][index]
    observation["observation_sha256"] = controller._compact_sha256(observation["observation"])


def _synthetic_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[controller.Namespace, Path, dict[str, Any]]:
    """Return one fully canonical auth receipt without live Q, I3, or science data."""

    project = (tmp_path / "project").resolve()
    control_root = project / "artifacts" / "resource_control"
    control_root.mkdir(parents=True)
    namespace = controller.Namespace.for_project(project)
    parent = (
        project / "artifacts" / "preregistration_amendments" / controller._AUTHORITY_C_COMPONENT
    )
    destination = parent.parent / _PROPOSED_AT.strftime("%Y%m%dT%H%M%S.%fZ")
    run_root = project / "artifacts" / "runs"

    run_hashes = {
        filename: _sha(f"run-state:{filename}") for filename in controller._RUN_STATE_FILENAMES
    }
    run_state_sha256 = controller._compact_sha256(run_hashes)
    terminal_receipt = {
        "run_state": {
            "root": str(run_root),
            "files": {
                filename: {
                    "path": str(run_root / filename),
                    "size_bytes": index + 1,
                    "sha256": run_hashes[filename],
                }
                for index, filename in enumerate(controller._RUN_STATE_FILENAMES)
            },
            "sha256": run_state_sha256,
        }
    }
    terminal_lineage = {
        "terminal_qualification_receipt_path": str(namespace.terminal_qualification),
        "terminal_qualification_receipt_sha256": hashlib.sha256(
            controller._canonical_bytes(terminal_receipt)
        ).hexdigest(),
        "terminal_qualification_receipt": terminal_receipt,
    }
    monkeypatch.setattr(
        controller,
        "_canonical_terminal_receipt",
        lambda value, **_kwargs: value,
    )

    frozen_files = {
        role: _record(namespace.input_v3 / filename, f"input-v3:{role}")
        for role, filename in controller.INPUT_V3_FILENAMES.items()
    }
    source = {
        "root_sha256": _sha("source-root"),
        "manifest_sha256": _sha("source-manifest"),
        "delta_sha256": _sha("source-delta"),
        "allowlisted_change_count": len(controller._EXPECTED_SOURCE_CHANGE_KINDS),
    }
    publication = {
        "amendment_timestamp_utc": _timestamp(_PROPOSED_AT),
        "intended_authority_directory": str(destination),
        "parent_authority_directory": str(parent),
        "amendment_schema_version": 5,
        "amendment_purpose": controller._TECHNICAL_SUCCESSOR_PURPOSE,
        "chain_depth": 4,
    }
    safety_margin = 10 * 1024**3
    projected_run = 1000
    maximum_workspace = 2000
    minimum_tracker = projected_run + safety_margin
    minimum_workspace = minimum_tracker + maximum_workspace
    required_before = minimum_workspace - 1
    planned_workspace = 1500
    workspace_plan_without_self_hash = _sha("workspace-plan-without-self")
    workspace_plan = {
        "plan_without_self_hash_sha256": workspace_plan_without_self_hash,
        "planned_workspace_bytes": planned_workspace,
        "required_free_bytes_before": required_before,
    }
    capacity_policy = {
        "projected_stable_run_bytes": projected_run,
        "fixed_safety_margin_bytes": safety_margin,
        "minimum_free_bytes_before_tracker": minimum_tracker,
        "maximum_workspace_bytes": maximum_workspace,
        "minimum_free_bytes_before_workspace_build": minimum_workspace,
        "minimum_total_ram_bytes": 32 * 1024**3,
        "minimum_available_ram_bytes_before_data": 16 * 1024**3,
        "cuda_device_index": 0,
        "minimum_total_vram_bytes": 10 * 1024**3,
        "minimum_free_vram_bytes": 8 * 1024**3,
        "cuda_smoke_max_peak_allocated_bytes": 512 * 1024**2,
        "cuda_required": True,
        "cudnn_required": True,
        "amp_required": True,
        "amp_dtype": "float16",
        "cuda_smoke_input_shape": [1, 3, 224, 224],
        "official_weight_identifier": "synthetic-official-resnet18",
        "official_weight_sha256": _sha("synthetic-official-resnet18"),
        "implicit_weight_download_allowed": False,
    }
    technical_authorization = {
        "schema_version": 3,
        "policy": controller.SCHEMA_V3_AUTHORIZATION_POLICY,
        "purpose": controller._TECHNICAL_SUCCESSOR_PURPOSE,
        "supersedes": {},
        "prior_publication_failure": {},
        "failed_preflight": {},
        "historical_primary": {},
        "resource_profile": {},
        "replacement_publication_failure_lineage": terminal_lineage,
        "execution_source_delta": {
            "resource_root_sha256": source["root_sha256"],
            "resource_manifest_sha256": source["manifest_sha256"],
            "delta_sha256": source["delta_sha256"],
        },
        "cnn_provenance_correction": {},
        "resource_capacity_policy": capacity_policy,
        "resource_input_workspace_plan": workspace_plan,
        "expected_successor_config_semantic_sha256": (controller._RESOURCE_CONFIG_SEMANTIC_SHA256),
        "resource_profile_shape": {
            "planned_required_cells": 24,
            "planned_cnn_cells": 6,
            "planned_cnn_fold_checkpoints": 30,
        },
        "outcomes_inspected": True,
        "analysis_disposition": "amended_or_exploratory",
        "outcome_use_policy": (
            "resource_constraints_only_no_outcome_value_selection_tuning_or_exclusion"
        ),
        "original_confirmatory_claim_allowed": False,
        "study_outcome_eligible": False,
        "completion_stage": None,
        "primary_rebinding_allowed": False,
        "primary_mutation_allowed": False,
        "automatic_retry_allowed": False,
        "scientific_profile_change_allowed": False,
    }
    technical_authorization_sha256 = controller._compact_sha256(technical_authorization)
    monkeypatch.setattr(
        amendment,
        "validate_resource_bounded_capacity_v3",
        lambda capacity_value, workspace_value: (
            capacity_value,
            workspace_value,
        ),
    )
    monkeypatch.setattr(
        amendment,
        "require_confirmatory_storage_policy",
        lambda _parent: amendment.ConfirmatoryStoragePolicy().as_dict(),
    )
    monkeypatch.setattr(
        amendment,
        "resource_bounded_technical_successor_intent_sha256",
        lambda **_kwargs: _sha("technical-intent"),
    )
    capacity = {
        "resource_capacity_policy_sha256": controller._compact_sha256(capacity_policy),
        "workspace_plan_sha256": frozen_files["workspace_plan"]["sha256"],
        "workspace_plan_without_self_hash_sha256": workspace_plan_without_self_hash,
        "projected_stable_run_bytes": projected_run,
        "fixed_safety_margin_bytes": safety_margin,
        "minimum_free_bytes_before_tracker": minimum_tracker,
        "maximum_workspace_bytes": maximum_workspace,
        "minimum_free_bytes_before_workspace_build": minimum_workspace,
        "planned_workspace_bytes": planned_workspace,
        "required_free_bytes_before": required_before,
        "required_free_bytes": minimum_workspace,
    }
    compute_probe = {
        "total_host_ram_bytes": 32 * 1024**3,
        "available_host_ram_bytes": 20 * 1024**3,
        "cuda_available": True,
        "cuda_device_count": 1,
        "selected_cuda_device_index": 0,
        "selected_cuda_device_name": "synthetic-cuda-device",
        "total_vram_bytes": 10 * 1024**3,
        "free_vram_bytes": 9 * 1024**3,
        "cudnn_available": True,
        "amp_available": True,
        "amp_dtype": "float16",
        "weight_identifier": capacity_policy["official_weight_identifier"],
        "weight_path": str(project / "weights" / "synthetic-resnet18.pth"),
        "weight_present": True,
        "weight_sha256": capacity_policy["official_weight_sha256"],
        "smoke_attempted": True,
        "smoke_completed": True,
        "smoke_input_shape": capacity_policy["cuda_smoke_input_shape"],
        "smoke_forward_finite": True,
        "smoke_backward_finite": True,
        "smoke_peak_allocated_bytes": 128 * 1024**2,
        "smoke_error": None,
    }

    def capacity_observation(moment: datetime) -> dict[str, Any]:
        return {
            "phase": "guarded_before_workspace_build",
            "probe_path": str(run_root),
            "free_bytes": minimum_workspace + 1024,
            "minimum_free_bytes": minimum_workspace,
            "policy_sha256": capacity["resource_capacity_policy_sha256"],
            "checked_at_utc": _timestamp(moment),
            "passed": True,
        }

    def compute_observation(moment: datetime) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "phase": "guarded_before_data_loading",
            "minimum_available_ram_bytes": capacity_policy[
                "minimum_available_ram_bytes_before_data"
            ],
            "policy_sha256": capacity["resource_capacity_policy_sha256"],
            "observation": copy.deepcopy(compute_probe),
            "observation_sha256": controller._compact_sha256(compute_probe),
            "checked_at_utc": _timestamp(moment),
            "passed": True,
            "outcome_values_read": False,
            "prohibited_for_selection_tuning": True,
            "adaptive_execution_changes_allowed": False,
        }

    contract = {
        "project_root": str(project),
        "parent_authority_directory": str(parent),
        "controller": controller._controller_identity(),
        "terminal_qualification": terminal_lineage,
        "frozen_input_bundle": {
            "directory": str(namespace.input_v3),
            "files": frozen_files,
            "records_sha256": controller._compact_sha256(frozen_files),
        },
        "source": source,
        "config": {
            "path": str(project / "configs" / "confirmatory_resource_bounded_amended.yaml"),
            "file_sha256": controller._RESOURCE_CONFIG_FILE_SHA256,
            "semantic_sha256": controller._RESOURCE_CONFIG_SEMANTIC_SHA256,
        },
        "manifest": {
            "path": str(
                project / "data" / "manifests" / "pannuke" / "pannuke_nucleus_manifest.parquet"
            ),
            "sha256": controller._PANNUKE_MANIFEST_SHA256,
        },
        "historical_lineage": {
            "failed_preflight_receipt_path": str(
                control_root / controller._HISTORICAL_FAILED_PREFLIGHT_FILENAME
            ),
            "failed_preflight_receipt_sha256": (
                controller.DEFAULT_HISTORICAL_PINS.failed_preflight.sha256
            ),
            "prior_failure_receipt_path": str(
                control_root / controller._HISTORICAL_PRIOR_FAILURE_FILENAME
            ),
            "prior_failure_receipt_sha256": (
                controller.DEFAULT_HISTORICAL_PINS.prior_failure.sha256
            ),
            "retired_input_invalidation_receipt_path": str(
                control_root / controller._HISTORICAL_INVALIDATION_FILENAME
            ),
            "retired_input_invalidation_receipt_sha256": (
                controller.DEFAULT_HISTORICAL_PINS.invalidation.sha256
            ),
        },
        "run_state": {
            "root": str(run_root),
            "files": run_hashes,
            "sha256": run_state_sha256,
        },
        "technical_successor": {
            "authorization": technical_authorization,
            "authorization_sha256": technical_authorization_sha256,
            "intent_sha256": _sha("technical-intent"),
            "storage_policy": amendment.ConfirmatoryStoragePolicy().as_dict(),
        },
        "publication": publication,
        "replacement_state": {
            "state": controller.State.AUTHORIZATION_REQUIRED.value,
            "candidate_count": 0,
            "attempt_marker_absent": True,
            "success_marker_absent": True,
            "failure_marker_absent": True,
            "intended_authority_absent": True,
        },
        "capacity_contract": capacity,
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }
    source_allowlist_payload = {
        "schema_version": 1,
        "policy": controller.INPUT_V3_POLICIES["source_allowlist"],
        "file_count": len(controller._EXPECTED_SOURCE_CHANGE_KINDS),
        "records": [
            (
                {
                    "path": logical,
                    "change_kind": change_kind,
                }
                if change_kind == "removed"
                else {
                    "path": logical,
                    "change_kind": change_kind,
                    "size_bytes": 1,
                    "sha256": _sha(f"allowlisted-source:{logical}"),
                }
            )
            for logical, change_kind in sorted(controller._EXPECTED_SOURCE_CHANGE_KINDS.items())
        ],
    }
    cnn_payload = {
        "schema_version": 1,
        "policy": controller.INPUT_V3_POLICIES["cnn_correction_receipt"],
        "correction": {"synthetic": True},
        "semantic_equivalence_evidence": {"synthetic": True},
    }
    frozen_source_payload = {
        "schema_version": 1,
        "policy": controller.INPUT_V3_POLICIES["frozen_source_receipt"],
        "file_count": len(controller._EXPECTED_SOURCE_CHANGE_KINDS),
        "source_allowlist_sha256": hashlib.sha256(
            controller._canonical_bytes(source_allowlist_payload)
        ).hexdigest(),
        "workspace_plan_sha256": hashlib.sha256(
            controller._canonical_bytes(workspace_plan)
        ).hexdigest(),
        "cnn_correction_receipt_sha256": hashlib.sha256(
            controller._canonical_bytes(cnn_payload)
        ).hexdigest(),
        "source_allowlist_semantic_sha256": controller._compact_sha256(source_allowlist_payload),
        "execution_source_root_sha256": source["root_sha256"],
        "execution_source_manifest_sha256": source["manifest_sha256"],
        "execution_source_artifact_count": 1,
        "execution_source_delta_count": source["allowlisted_change_count"],
        "execution_source_delta_sha256": source["delta_sha256"],
        "execution_source_change_kinds_sha256": controller._compact_sha256(
            controller._EXPECTED_SOURCE_CHANGE_KINDS
        ),
        "parent_execution_source_root_sha256": controller._AUTHORITY_C_SOURCE_PINS["root_sha256"],
        "parent_execution_source_manifest_sha256": (
            controller._AUTHORITY_C_SOURCE_PINS["manifest_sha256"]
        ),
        "config_path": contract["config"]["path"],
        "config_file_sha256": contract["config"]["file_sha256"],
        "config_semantic_sha256": contract["config"]["semantic_sha256"],
        "manifest_path": contract["manifest"]["path"],
        "manifest_sha256": contract["manifest"]["sha256"],
        **copy.deepcopy(contract["historical_lineage"]),
        "terminal_qualification_receipt_path": str(namespace.terminal_qualification),
        "terminal_qualification_receipt_sha256": terminal_lineage[
            "terminal_qualification_receipt_sha256"
        ],
        "controller_path": contract["controller"]["path"],
        "controller_size_bytes": contract["controller"]["size_bytes"],
        "controller_sha256": contract["controller"]["sha256"],
        "run_state_root": contract["run_state"]["root"],
        "run_state_files": copy.deepcopy(contract["run_state"]["files"]),
        "run_state_sha256": contract["run_state"]["sha256"],
        "authorization_sha256": technical_authorization_sha256,
        "workspace_plan_without_self_hash_sha256": (workspace_plan_without_self_hash),
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }
    static_input_payloads = {
        "source_allowlist": source_allowlist_payload,
        "workspace_plan": workspace_plan,
        "cnn_correction_receipt": cnn_payload,
        "frozen_source_receipt": frozen_source_payload,
    }
    for role, filename in controller.INPUT_V3_FILENAMES.items():
        encoded = controller._canonical_bytes(static_input_payloads[role])
        frozen_files[role] = {
            "path": str(namespace.input_v3 / filename),
            "size_bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    frozen_records_sha256 = controller._compact_sha256(frozen_files)
    contract["frozen_input_bundle"]["records_sha256"] = frozen_records_sha256
    capacity["workspace_plan_sha256"] = frozen_files["workspace_plan"]["sha256"]
    monkeypatch.setattr(
        controller,
        "_read_input_v3",
        lambda _namespace, *, verify_live=False: (
            copy.deepcopy(static_input_payloads),
            copy.deepcopy(frozen_files),
            frozen_records_sha256,
        ),
    )
    receipt = {
        "schema_version": 2,
        "policy": controller.PUBLICATION_AUTHORIZATION_V2_POLICY,
        "status": "authorized_for_one_attempt",
        "authorized_at_utc": _timestamp(_AUTHORIZED_AT),
        "authorized_attempt_id": _sha("authorized-attempt"),
        "max_attempt_count": 1,
        "automatic_retry_allowed": False,
        "publication": publication,
        "preflight": {
            "schema_version": 2,
            "policy": controller._LIVE_PREFLIGHT_V2_POLICY,
            "status": "passed_twice",
            "contract": contract,
            "preflight_fingerprint_sha256": controller._compact_sha256(contract),
            "capacity_observations": [
                capacity_observation(_PROPOSED_AT),
                capacity_observation(_PROPOSED_AT + timedelta(seconds=1)),
            ],
            "compute_observations": [
                compute_observation(_PROPOSED_AT + timedelta(milliseconds=500)),
                compute_observation(_PROPOSED_AT + timedelta(milliseconds=1500)),
            ],
        },
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }
    return namespace, parent, receipt


@dataclass
class _FlowState:
    namespace: controller.Namespace
    parent: Path
    lock_state: dict[str, bool] = field(
        default_factory=lambda: {"protocol": False, "parent": False}
    )
    authorized: bool = False
    preflight_calls: list[dict[str, Any]] = field(default_factory=list)
    presence_calls: list[tuple[bool, bool, bool]] = field(default_factory=list)
    candidate_calls: list[tuple[bool, bool, bool]] = field(default_factory=list)
    inventory_calls: list[bool] = field(default_factory=list)
    legacy_guard_calls: list[bool] = field(default_factory=list)
    parent_lock_arguments: list[tuple[tuple[Path, ...], str]] = field(default_factory=list)
    readback_calls: list[tuple[bool, bool, bool]] = field(default_factory=list)
    final_read_calls: list[tuple[Path, str]] = field(default_factory=list)
    rollback_calls: list[list[Any]] = field(default_factory=list)
    published_receipt: dict[str, Any] | None = None
    published_bytes: bytes = b""


def _install_flow_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    clocks: tuple[datetime, datetime] = (_PROPOSED_AT, _AUTHORIZED_AT),
    second_preflight_drift: bool = False,
    candidate_before_write: bool = False,
    candidate_after_write: bool = False,
    publish_error: bool = False,
    readback_mismatch: bool = False,
    parent_exit_error: bool = False,
    late_preflight_drift: str | None = None,
    final_presence_drift: str | None = None,
) -> tuple[_FlowState, Any]:
    project = (tmp_path / "project").resolve()
    parent = (
        project / "artifacts" / "preregistration_amendments" / controller._AUTHORITY_C_COMPONENT
    )
    namespace = controller.Namespace.for_project(project)
    state = _FlowState(namespace=namespace, parent=parent)

    class FakeLock:
        def __init__(self, name: str, *, fail_on_exit: bool = False) -> None:
            self.name = name
            self.fail_on_exit = fail_on_exit
            self.lock_paths = (tmp_path / f"{name}.lock",)

        def __enter__(self) -> FakeLock:
            assert state.lock_state[self.name] is False
            state.lock_state[self.name] = True
            return self

        def __exit__(self, *_args: object) -> None:
            state.lock_state[self.name] = False
            if self.fail_on_exit:
                raise RuntimeError(f"{self.name} lock exit failed")

        def assert_owned(self) -> None:
            assert state.lock_state[self.name] is True

    class FakePublished:
        def __init__(self, encoded: bytes) -> None:
            self.sha256 = hashlib.sha256(encoded).hexdigest()

        def still_owned(self) -> bool:
            return state.authorized

    protocol_lock = FakeLock("protocol")
    parent_lock = FakeLock("parent", fail_on_exit=parent_exit_error)
    clock_values = iter(clocks)

    monkeypatch.setattr(
        controller,
        "_require_parent",
        lambda *_args, **_kwargs: (project, parent),
    )

    def presence(_namespace: controller.Namespace) -> dict[str, bool]:
        state.presence_calls.append(
            (
                state.authorized,
                state.lock_state["protocol"],
                state.lock_state["parent"],
            )
        )
        after_late_readback = bool(state.readback_calls)
        return {
            "qualification": not (after_late_readback and final_presence_drift == "qualification"),
            "inputs": not (after_late_readback and final_presence_drift == "inputs"),
            "authorization": state.authorized
            and not (after_late_readback and final_presence_drift == "authorization"),
            "attempt": after_late_readback and final_presence_drift == "attempt",
            "success": after_late_readback and final_presence_drift == "success",
            "failure": after_late_readback and final_presence_drift == "failure",
        }

    monkeypatch.setattr(controller, "_reserved_family_presence", presence)
    monkeypatch.setattr(controller, "_legacy_scoped_lock_paths", lambda *_a, **_k: ())
    monkeypatch.setattr(
        controller,
        "_protocol_lock",
        lambda *_args, **_kwargs: protocol_lock,
    )

    def parent_lock_factory(paths: Any, *, role: str) -> FakeLock:
        materialized = tuple(Path(path) for path in paths)
        state.parent_lock_arguments.append((materialized, role))
        return parent_lock

    monkeypatch.setattr(controller, "ExclusiveBundlePublicationLock", parent_lock_factory)

    def require_owned(*, legacy_paths: Any, owned_locks: Any) -> None:
        assert tuple(legacy_paths) == ()
        assert tuple(owned_locks) == (protocol_lock, parent_lock)
        protocol_lock.assert_owned()
        parent_lock.assert_owned()
        after_late_readback = bool(state.readback_calls)
        state.legacy_guard_calls.append(after_late_readback)
        if after_late_readback and final_presence_drift == "legacy_lock":
            raise controller.ControlError("synthetic late legacy lock drift")

    monkeypatch.setattr(
        controller,
        "_require_legacy_lock_state_under_protocol_lock",
        require_owned,
    )

    def stable_inventory(*_args: Any, **_kwargs: Any) -> tuple[()]:
        after_late_readback = bool(state.readback_calls)
        state.inventory_calls.append(after_late_readback)
        if after_late_readback and final_presence_drift == "inventory":
            raise controller.ControlError("synthetic late Authority-C inventory drift")
        return ()

    monkeypatch.setattr(controller, "_stable_amendment_inventory", stable_inventory)

    def candidates(_parent: Path) -> tuple[Path, ...]:
        state.candidate_calls.append(
            (
                state.authorized,
                state.lock_state["protocol"],
                state.lock_state["parent"],
            )
        )
        if candidate_before_write and not state.authorized:
            return (parent.parent / "candidate-before-write",)
        if candidate_after_write and state.authorized:
            return (parent.parent / "candidate-after-write",)
        if state.readback_calls and final_presence_drift == "candidate":
            return (parent.parent / "candidate-after-final-live-readback",)
        return ()

    monkeypatch.setattr(controller, "discover_candidates", candidates)

    destination = parent.parent / _PROPOSED_AT.strftime("%Y%m%dT%H%M%S.%fZ")
    publication = {
        "amendment_timestamp_utc": _timestamp(_PROPOSED_AT),
        "intended_authority_directory": str(destination),
        "parent_authority_directory": str(parent),
        "amendment_schema_version": 5,
        "amendment_purpose": controller._TECHNICAL_SUCCESSOR_PURPOSE,
        "chain_depth": 4,
    }
    stable_contract = {
        "terminal_qualification": {"sha256": _sha("flow-q")},
        "frozen_input_bundle": {"records_sha256": _sha("flow-i3")},
        "source": {"root_sha256": _sha("flow-source")},
        "run_state": {"sha256": _sha("flow-run-state")},
        "publication": publication,
        "technical_successor": {
            "intent_sha256": _sha("flow-intent"),
            "storage_policy": {"sha256": _sha("flow-authority-c")},
        },
    }

    def preflight(**kwargs: Any) -> dict[str, Any]:
        call_index = len(state.preflight_calls)
        state.preflight_calls.append(
            {
                "lock_state": dict(state.lock_state),
                "authorization_present": kwargs.get("authorization_present", False),
                "amendment_timestamp": kwargs["amendment_timestamp"],
            }
        )
        contract = copy.deepcopy(stable_contract)
        if second_preflight_drift and call_index == 1:
            contract["drift"] = True
        if late_preflight_drift and call_index == 2:
            drift_paths = {
                "qualification": ("terminal_qualification", "sha256"),
                "inputs": ("frozen_input_bundle", "records_sha256"),
                "source": ("source", "root_sha256"),
                "run_state": ("run_state", "sha256"),
                "authority_c": ("technical_successor", "storage_policy", "sha256"),
            }
            drift_path = drift_paths[late_preflight_drift]
            _set_path(contract, drift_path, _sha(f"late-drift:{late_preflight_drift}"))
        drifted = (second_preflight_drift and call_index == 1) or (
            late_preflight_drift is not None and call_index == 2
        )
        return {
            "contract": contract,
            "preflight_fingerprint_sha256": _sha(
                "drifted-fingerprint" if drifted else "stable-fingerprint"
            ),
            "intent_sha256": _sha("flow-intent"),
            "destination": destination,
            "capacity_observation": {
                "checked_at_utc": _timestamp(_PROPOSED_AT + timedelta(microseconds=call_index)),
                "passed": True,
                "observation_id": f"capacity-{call_index + 1}",
            },
            "compute_observation": {
                "checked_at_utc": _timestamp(_PROPOSED_AT + timedelta(microseconds=call_index)),
                "passed": True,
                "observation_id": f"compute-{call_index + 1}",
            },
        }

    monkeypatch.setattr(controller, "_build_live_preflight_v2", preflight)
    monkeypatch.setattr(
        controller,
        "_canonical_publication_authorization_v2",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        controller,
        "_require_live_observation_cross_links",
        lambda **_kwargs: None,
    )

    def publish(encoded: bytes, destination_path: Path) -> FakePublished:
        assert state.lock_state == {"protocol": True, "parent": True}
        assert destination_path == namespace.authorization_v2
        if publish_error:
            raise FileExistsError("synthetic O_EXCL collision")
        state.published_bytes = encoded
        state.published_receipt = json.loads(encoded.decode("utf-8"))
        state.authorized = True
        return FakePublished(encoded)

    monkeypatch.setattr(controller, "publish_bytes_no_overwrite", publish)

    real_readback = controller._read_publication_authorization_v2

    def read_bytes(path: Path, role: str, **_kwargs: Any) -> bytes:
        assert state.authorized is True
        assert path == namespace.authorization_v2
        state.final_read_calls.append((path, role))
        if readback_mismatch and len(state.final_read_calls) == 1:
            tampered = json.loads(state.published_bytes.decode("utf-8"))
            tampered["status"] = "tampered"
            return controller._canonical_bytes(tampered)
        return state.published_bytes

    monkeypatch.setattr(controller, "_read_bytes", read_bytes)

    def readback(
        readback_namespace: controller.Namespace,
        *,
        verify_live: bool,
    ) -> tuple[dict[str, Any], str]:
        state.readback_calls.append(
            (
                verify_live,
                state.lock_state["protocol"],
                state.lock_state["parent"],
            )
        )
        return real_readback(readback_namespace, verify_live=verify_live)

    monkeypatch.setattr(controller, "_read_publication_authorization_v2", readback)

    def rollback(publications: Any) -> None:
        state.rollback_calls.append(list(publications))
        state.authorized = False
        state.published_receipt = None
        state.published_bytes = b""

    monkeypatch.setattr(controller, "rollback_owned_publications", rollback)

    def invoke() -> tuple[dict[str, Any], str]:
        return controller._authorize_publication_v2_once(
            namespace=namespace,
            parent_authority_directory=parent,
            clock=lambda: next(clock_values),
        )

    return state, invoke


def test_public_authorization_signature_is_sealed() -> None:
    signature = inspect.signature(controller.authorize_publication_v2_once)
    assert tuple(signature.parameters) == (
        "namespace",
        "parent_authority_directory",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert {"clock", "preflight", "authorization"}.isdisjoint(signature.parameters)


def test_authorization_runs_two_stable_preflights_and_full_live_readback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, invoke = _install_flow_harness(monkeypatch, tmp_path)

    receipt, digest = invoke()

    assert [call["lock_state"] for call in state.preflight_calls] == [
        {"protocol": False, "parent": False},
        {"protocol": True, "parent": True},
        {"protocol": True, "parent": True},
    ]
    assert [call["amendment_timestamp"] for call in state.preflight_calls] == [
        _PROPOSED_AT,
        _PROPOSED_AT,
        _PROPOSED_AT,
    ]
    assert [call["authorization_present"] for call in state.preflight_calls] == [False, False, True]
    assert receipt["authorized_at_utc"] == _timestamp(_AUTHORIZED_AT)
    assert receipt["preflight"]["status"] == "passed_twice"
    assert (
        receipt["preflight"]["capacity_observations"][0]
        != receipt["preflight"]["capacity_observations"][1]
    )
    assert (
        receipt["preflight"]["compute_observations"][0]
        != receipt["preflight"]["compute_observations"][1]
    )
    assert state.readback_calls == [(True, True, True)]
    assert state.presence_calls == [
        (False, False, False),
        (False, True, True),
        (True, True, True),
        (True, True, True),
    ]
    assert state.candidate_calls == [
        (False, True, True),
        (True, True, True),
        (True, True, True),
    ]
    assert state.inventory_calls == [False, False, True]
    assert state.legacy_guard_calls == [False, False, True]
    assert state.parent_lock_arguments == [
        (
            (state.parent,),
            "resource Authority-D replacement-v2 authorization Authority-C parent guard",
        )
    ]
    assert digest == hashlib.sha256(state.published_bytes).hexdigest()
    assert [role for _path, role in state.final_read_calls] == [
        "publication authorization-v2",
        "final publication authorization-v2",
    ]
    assert state.authorized is True
    assert state.rollback_calls == []
    assert state.lock_state == {"protocol": False, "parent": False}


def test_authorization_rejects_unstable_second_preflight_before_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, invoke = _install_flow_harness(
        monkeypatch,
        tmp_path,
        second_preflight_drift=True,
    )

    with pytest.raises(controller.ControlError, match="different stable contracts"):
        invoke()

    assert len(state.preflight_calls) == 2
    assert state.published_receipt is None
    assert state.authorized is False
    assert state.rollback_calls == []
    assert state.lock_state == {"protocol": False, "parent": False}


@pytest.mark.parametrize(
    "drift",
    ["qualification", "inputs", "source", "run_state", "authority_c"],
)
def test_late_full_live_preflight_drift_rolls_back_authorization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    drift: str,
) -> None:
    state, invoke = _install_flow_harness(
        monkeypatch,
        tmp_path,
        late_preflight_drift=drift,
    )

    with pytest.raises(controller.ControlError, match="live contract changed"):
        invoke()

    assert len(state.preflight_calls) == 3
    assert state.preflight_calls[-1]["authorization_present"] is True
    assert state.preflight_calls[-1]["lock_state"] == {
        "protocol": True,
        "parent": True,
    }
    assert [role for _path, role in state.final_read_calls] == ["publication authorization-v2"]
    assert len(state.rollback_calls) == 1
    assert len(state.rollback_calls[0]) == 1
    assert state.authorized is False
    assert state.published_receipt is None
    assert state.lock_state == {"protocol": False, "parent": False}


@pytest.mark.parametrize(
    "residual_mutation",
    [
        "qualification",
        "inputs",
        "authorization",
        "attempt",
        "success",
        "failure",
        "candidate",
        "inventory",
        "legacy_lock",
    ],
)
def test_residual_mutation_after_late_preflight_is_stopped_and_rolled_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    residual_mutation: str,
) -> None:
    state, invoke = _install_flow_harness(
        monkeypatch,
        tmp_path,
        final_presence_drift=residual_mutation,
    )

    with pytest.raises(controller.ControlError):
        invoke()

    assert len(state.preflight_calls) == 3
    assert state.preflight_calls[-1]["authorization_present"] is True
    assert state.presence_calls[-1] == (True, True, True)
    assert [role for _path, role in state.final_read_calls] == ["publication authorization-v2"]
    assert len(state.rollback_calls) == 1
    assert len(state.rollback_calls[0]) == 1
    assert state.authorized is False
    assert state.published_receipt is None
    assert state.lock_state == {"protocol": False, "parent": False}


@pytest.mark.parametrize(
    ("candidate_before", "candidate_after", "readback_mismatch"),
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
def test_authorization_candidate_and_readback_failures_are_rollback_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    candidate_before: bool,
    candidate_after: bool,
    readback_mismatch: bool,
) -> None:
    state, invoke = _install_flow_harness(
        monkeypatch,
        tmp_path,
        candidate_before_write=candidate_before,
        candidate_after_write=candidate_after,
        readback_mismatch=readback_mismatch,
    )

    with pytest.raises(controller.ControlError):
        invoke()

    if candidate_before:
        assert state.rollback_calls == []
    else:
        assert len(state.rollback_calls) == 1
        assert len(state.rollback_calls[0]) == 1
    assert state.authorized is False
    assert state.published_receipt is None
    assert state.lock_state == {"protocol": False, "parent": False}


def test_authorization_o_excl_collision_and_lock_exit_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    collision_state, collision_invoke = _install_flow_harness(
        monkeypatch,
        tmp_path / "collision",
        publish_error=True,
    )
    with pytest.raises(FileExistsError, match="O_EXCL"):
        collision_invoke()
    assert collision_state.rollback_calls == []
    assert collision_state.authorized is False
    assert collision_state.lock_state == {"protocol": False, "parent": False}

    monkeypatch.undo()
    exit_state, exit_invoke = _install_flow_harness(
        monkeypatch,
        tmp_path / "lock-exit",
        parent_exit_error=True,
    )
    with pytest.raises(
        controller.AmbiguousStateError,
        match="exclusion ended while an owned publication may remain",
    ):
        exit_invoke()
    assert exit_state.authorized is True
    assert exit_state.rollback_calls == []
    assert exit_state.lock_state == {"protocol": False, "parent": False}


@pytest.mark.parametrize(
    "clocks",
    [
        (datetime.now(UTC) + timedelta(days=1), datetime.now(UTC) + timedelta(days=1)),
        (_PROPOSED_AT, _PROPOSED_AT - timedelta(microseconds=1)),
    ],
)
def test_private_authorization_rejects_untrusted_clock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clocks: tuple[datetime, datetime],
) -> None:
    state, invoke = _install_flow_harness(monkeypatch, tmp_path, clocks=clocks)

    with pytest.raises(controller.ControlError, match="clock"):
        invoke()

    assert state.authorized is False
    assert state.published_receipt is None
    assert state.rollback_calls == []
    assert state.lock_state == {"protocol": False, "parent": False}


def test_live_runner_timestamp_is_normalized_but_stored_alias_is_rejected() -> None:
    runner_timestamp = "2026-07-28T20:30:00.000000+00:00"

    assert (
        controller._normalize_external_timestamp(
            runner_timestamp,
            "synthetic runner observation",
        )
        == "2026-07-28T20:30:00.000000Z"
    )
    with pytest.raises(controller.ControlError, match="not canonical"):
        controller._canonical_external_timestamp(
            runner_timestamp,
            "stored authorization observation",
        )


def test_live_authorization_readback_repeats_full_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    namespace, parent, receipt = _synthetic_authorization(tmp_path, monkeypatch)
    encoded = controller._canonical_bytes(receipt)
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(controller, "_read_bytes", lambda *_a, **_k: encoded)
    monkeypatch.setattr(
        controller,
        "_controller_identity",
        lambda: copy.deepcopy(receipt["preflight"]["contract"]["controller"]),
    )

    def repeated_preflight(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "contract": copy.deepcopy(receipt["preflight"]["contract"]),
            "preflight_fingerprint_sha256": receipt["preflight"]["preflight_fingerprint_sha256"],
            "intent_sha256": receipt["preflight"]["contract"]["technical_successor"][
                "intent_sha256"
            ],
            "destination": Path(receipt["publication"]["intended_authority_directory"]),
            "capacity_observation": copy.deepcopy(receipt["preflight"]["capacity_observations"][1]),
            "compute_observation": copy.deepcopy(receipt["preflight"]["compute_observations"][1]),
        }

    monkeypatch.setattr(controller, "_build_live_preflight_v2", repeated_preflight)
    readback, digest = controller._read_publication_authorization_v2(
        namespace,
        verify_live=True,
    )

    assert readback == receipt
    assert digest == hashlib.sha256(encoded).hexdigest()
    assert len(calls) == 1
    assert calls[0]["parent_authority_directory"] == str(parent)
    assert calls[0]["authorization_present"] is True
    assert calls[0]["amendment_timestamp"] == _PROPOSED_AT

    def drifted_preflight(**kwargs: Any) -> dict[str, Any]:
        repeated = repeated_preflight(**kwargs)
        repeated["contract"]["source"]["root_sha256"] = _sha("drifted-live-source")
        return repeated

    monkeypatch.setattr(controller, "_build_live_preflight_v2", drifted_preflight)
    with pytest.raises(controller.ControlError, match="live contract changed"):
        controller._read_publication_authorization_v2(namespace, verify_live=True)


def test_live_readback_rejects_forged_stable_compute_observation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    namespace, _parent, receipt = _synthetic_authorization(tmp_path, monkeypatch)
    forged = copy.deepcopy(receipt)
    forged["preflight"]["compute_observations"][1]["observation"]["selected_cuda_device_name"] = (
        "forged-different-device"
    )
    _refresh_compute_probe(forged, 1)
    encoded = controller._canonical_bytes(forged)

    monkeypatch.setattr(controller, "_read_bytes", lambda *_a, **_k: encoded)
    monkeypatch.setattr(
        controller,
        "_controller_identity",
        lambda: copy.deepcopy(forged["preflight"]["contract"]["controller"]),
    )
    monkeypatch.setattr(
        controller,
        "_build_live_preflight_v2",
        lambda **_kwargs: {
            "contract": copy.deepcopy(forged["preflight"]["contract"]),
            "preflight_fingerprint_sha256": forged["preflight"]["preflight_fingerprint_sha256"],
            "intent_sha256": forged["preflight"]["contract"]["technical_successor"][
                "intent_sha256"
            ],
            "destination": Path(forged["publication"]["intended_authority_directory"]),
            "capacity_observation": copy.deepcopy(forged["preflight"]["capacity_observations"][1]),
            "compute_observation": copy.deepcopy(forged["preflight"]["compute_observations"][0]),
        },
    )

    with pytest.raises(
        controller.ControlError,
        match="stored compute observations differ from live stable hardware",
    ):
        controller._read_publication_authorization_v2(namespace, verify_live=True)


@pytest.mark.parametrize(
    ("keys", "replacement"),
    [
        (("schema_version",), 2.0),
        (("max_attempt_count",), True),
        (("automatic_retry_allowed",), 0),
        (("publication", "amendment_schema_version"), 5.0),
        (("publication", "chain_depth"), 4.0),
        (("preflight", "schema_version"), True),
        (
            ("preflight", "contract", "source", "allowlisted_change_count"),
            float(len(controller._EXPECTED_SOURCE_CHANGE_KINDS)),
        ),
        (
            ("preflight", "contract", "replacement_state", "candidate_count"),
            False,
        ),
        (
            (
                "preflight",
                "contract",
                "capacity_contract",
                "projected_stable_run_bytes",
            ),
            1000.0,
        ),
        (("outcome_value_interpretation_performed",), 0),
    ],
)
def test_authorization_canonicalizer_rejects_int_and_bool_aliases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    keys: tuple[str | int, ...],
    replacement: Any,
) -> None:
    namespace, _parent, receipt = _synthetic_authorization(tmp_path, monkeypatch)
    tampered = copy.deepcopy(receipt)
    _set_path(tampered, keys, replacement)
    _refresh_preflight_fingerprint(tampered)

    with pytest.raises(controller.ControlError):
        controller._canonical_publication_authorization_v2(
            tampered,
            namespace=namespace,
            verify_live_controller=False,
        )


@pytest.mark.parametrize(
    ("parent_keys", "field_name", "operation", "replacement"),
    [
        ((), "unexpected", "add", "not-in-schema"),
        ((), "status", "delete", None),
        (("preflight",), "unexpected", "add", "not-in-schema"),
        (("preflight",), "policy", "delete", None),
        (("preflight", "contract"), "source", "delete", None),
        (
            ("preflight", "capacity_observations", 0),
            "unexpected",
            "add",
            "not-in-schema",
        ),
        (("preflight", "capacity_observations", 0), "free_bytes", "replace", True),
        (("preflight", "compute_observations", 0), "passed", "delete", None),
        (("preflight", "compute_observations", 0), "schema_version", "replace", 1.0),
        (
            ("preflight", "compute_observations", 0, "observation"),
            "smoke_error",
            "delete",
            None,
        ),
        (
            ("preflight", "compute_observations", 0, "observation"),
            "cuda_device_count",
            "replace",
            True,
        ),
        (
            ("preflight", "compute_observations", 0, "observation"),
            "cuda_available",
            "replace",
            1,
        ),
    ],
)
def test_auth_v2_rejects_extra_missing_and_wrong_type_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    parent_keys: tuple[str | int, ...],
    field_name: str,
    operation: str,
    replacement: Any,
) -> None:
    namespace, _parent, receipt = _synthetic_authorization(tmp_path, monkeypatch)
    tampered = copy.deepcopy(receipt)
    target: Any = tampered
    for key in parent_keys:
        target = target[key]
    if operation == "delete":
        del target[field_name]
    else:
        target[field_name] = replacement
    for index, observation in enumerate(tampered["preflight"]["compute_observations"]):
        if type(observation) is dict and type(observation.get("observation")) is dict:
            _refresh_compute_probe(tampered, index)
    _refresh_preflight_fingerprint(tampered)

    with pytest.raises(controller.ControlError):
        controller._canonical_publication_authorization_v2(
            tampered,
            namespace=namespace,
            verify_live_controller=False,
        )


@pytest.mark.parametrize(
    ("keys", "replacement", "refresh"),
    [
        (("policy",), "forged-authorization-policy", "none"),
        (("status",), "preflight_only", "none"),
        (("preflight", "policy"), "forged-preflight-policy", "none"),
        (("preflight", "status"), "passed_once", "none"),
        (
            ("preflight", "contract", "replacement_state", "state"),
            "publication_ready",
            "contract",
        ),
        (
            (
                "preflight",
                "contract",
                "technical_successor",
                "authorization",
                "policy",
            ),
            "forged-schema-v3-policy",
            "technical",
        ),
        (("authorized_attempt_id",), "not-a-sha256", "none"),
        (
            ("preflight", "preflight_fingerprint_sha256"),
            "0" * 64,
            "none",
        ),
        (
            (
                "preflight",
                "contract",
                "technical_successor",
                "authorization_sha256",
            ),
            "0" * 64,
            "contract",
        ),
        (
            ("preflight", "compute_observations", 0, "observation_sha256"),
            "0" * 64,
            "none",
        ),
        (("outcome_value_interpretation_performed",), True, "none"),
        (("scientific_execution_performed",), True, "none"),
        (("publication_performed",), True, "none"),
        (
            (
                "preflight",
                "contract",
                "outcome_value_interpretation_performed",
            ),
            True,
            "contract",
        ),
        (
            ("preflight", "contract", "scientific_execution_performed"),
            True,
            "contract",
        ),
        (
            ("preflight", "contract", "publication_performed"),
            True,
            "contract",
        ),
        (
            ("preflight", "compute_observations", 0, "outcome_values_read"),
            True,
            "none",
        ),
        (
            (
                "preflight",
                "compute_observations",
                0,
                "prohibited_for_selection_tuning",
            ),
            False,
            "none",
        ),
        (
            (
                "preflight",
                "compute_observations",
                0,
                "adaptive_execution_changes_allowed",
            ),
            True,
            "none",
        ),
        (
            ("preflight", "capacity_observations", 0, "phase"),
            "guarded_after_workspace_build",
            "none",
        ),
        (
            ("preflight", "compute_observations", 0, "phase"),
            "guarded_after_data_loading",
            "none",
        ),
    ],
)
def test_auth_v2_rejects_wrong_phase_policy_hash_and_no_outcome_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    keys: tuple[str | int, ...],
    replacement: Any,
    refresh: str,
) -> None:
    namespace, _parent, receipt = _synthetic_authorization(tmp_path, monkeypatch)
    tampered = copy.deepcopy(receipt)
    _set_path(tampered, keys, replacement)
    if refresh == "contract":
        _refresh_preflight_fingerprint(tampered)
    elif refresh == "technical":
        _refresh_technical_authorization(tampered)

    with pytest.raises(controller.ControlError):
        controller._canonical_publication_authorization_v2(
            tampered,
            namespace=namespace,
            verify_live_controller=False,
        )


@pytest.mark.parametrize(
    "forgery",
    [
        "capacity_below_minimum",
        "capacity_minimum_cross_link",
        "capacity_probe_path",
        "capacity_policy",
        "duplicate_capacity_observation",
        "compute_available_ram",
        "compute_weight_hash",
        "compute_policy",
        "compute_passed",
    ],
)
def test_auth_v2_rejects_forged_capacity_and_compute_observations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    forgery: str,
) -> None:
    namespace, _parent, receipt = _synthetic_authorization(tmp_path, monkeypatch)
    tampered = copy.deepcopy(receipt)
    capacities = tampered["preflight"]["capacity_observations"]
    computes = tampered["preflight"]["compute_observations"]

    if forgery == "capacity_below_minimum":
        capacities[0]["free_bytes"] = capacities[0]["minimum_free_bytes"] - 1
    elif forgery == "capacity_minimum_cross_link":
        capacities[0]["minimum_free_bytes"] += 1
    elif forgery == "capacity_probe_path":
        capacities[0]["probe_path"] = str(tmp_path.resolve())
    elif forgery == "capacity_policy":
        capacities[0]["policy_sha256"] = _sha("forged-capacity-policy")
    elif forgery == "duplicate_capacity_observation":
        capacities[1] = copy.deepcopy(capacities[0])
    elif forgery == "compute_available_ram":
        computes[0]["observation"]["available_host_ram_bytes"] = 0
        _refresh_compute_probe(tampered, 0)
    elif forgery == "compute_weight_hash":
        computes[0]["observation"]["weight_sha256"] = _sha("forged-weight")
        _refresh_compute_probe(tampered, 0)
    elif forgery == "compute_policy":
        computes[0]["policy_sha256"] = _sha("forged-compute-policy")
    elif forgery == "compute_passed":
        computes[0]["passed"] = False
    else:  # pragma: no cover - protects the test mutation table
        raise AssertionError(f"unknown forgery {forgery}")

    with pytest.raises(controller.ControlError):
        controller._canonical_publication_authorization_v2(
            tampered,
            namespace=namespace,
            verify_live_controller=False,
        )


@pytest.mark.parametrize(
    "chronology",
    [
        "capacity_after_compute_in_same_preflight",
        "second_capacity_before_first_compute",
        "ordered_but_before_amendment_1970",
        "observation_after_authorization",
        "noncanonical_offset_alias",
    ],
)
def test_auth_v2_rejects_reversed_and_pre_amendment_observation_chronology(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    chronology: str,
) -> None:
    namespace, _parent, receipt = _synthetic_authorization(tmp_path, monkeypatch)
    tampered = copy.deepcopy(receipt)
    capacities = tampered["preflight"]["capacity_observations"]
    computes = tampered["preflight"]["compute_observations"]

    if chronology == "capacity_after_compute_in_same_preflight":
        capacities[0]["checked_at_utc"] = _timestamp(_PROPOSED_AT + timedelta(milliseconds=750))
    elif chronology == "second_capacity_before_first_compute":
        capacities[1]["checked_at_utc"] = _timestamp(_PROPOSED_AT + timedelta(milliseconds=250))
    elif chronology == "ordered_but_before_amendment_1970":
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        capacities[0]["checked_at_utc"] = _timestamp(epoch)
        computes[0]["checked_at_utc"] = _timestamp(epoch + timedelta(seconds=1))
        capacities[1]["checked_at_utc"] = _timestamp(epoch + timedelta(seconds=2))
        computes[1]["checked_at_utc"] = _timestamp(epoch + timedelta(seconds=3))
    elif chronology == "observation_after_authorization":
        computes[1]["checked_at_utc"] = _timestamp(_AUTHORIZED_AT + timedelta(microseconds=1))
    elif chronology == "noncanonical_offset_alias":
        capacities[0]["checked_at_utc"] = _PROPOSED_AT.isoformat()
    else:  # pragma: no cover - protects the test mutation table
        raise AssertionError(f"unknown chronology {chronology}")

    with pytest.raises(controller.ControlError):
        controller._canonical_publication_authorization_v2(
            tampered,
            namespace=namespace,
            verify_live_controller=False,
        )


def test_authorization_canonicalizer_rejects_path_and_timestamp_aliases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    namespace, parent, receipt = _synthetic_authorization(tmp_path, monkeypatch)
    aliases: list[dict[str, Any]] = []

    parent_alias = copy.deepcopy(receipt)
    parent_alias["publication"]["parent_authority_directory"] = str(
        parent.parent / ".." / parent.parent.name / parent.name
    )
    aliases.append(parent_alias)

    config_alias = copy.deepcopy(receipt)
    config_path = Path(config_alias["preflight"]["contract"]["config"]["path"])
    config_alias["preflight"]["contract"]["config"]["path"] = str(
        config_path.parent / ".." / config_path.parent.name / config_path.name
    )
    _refresh_preflight_fingerprint(config_alias)
    aliases.append(config_alias)

    timestamp_alias = copy.deepcopy(receipt)
    timestamp_alias["authorized_at_utc"] = _AUTHORIZED_AT.isoformat()
    aliases.append(timestamp_alias)

    future = copy.deepcopy(receipt)
    future["authorized_at_utc"] = _timestamp(datetime.now(UTC) + timedelta(days=1))
    aliases.append(future)

    for tampered in aliases:
        with pytest.raises(controller.ControlError):
            controller._canonical_publication_authorization_v2(
                tampered,
                namespace=namespace,
                verify_live_controller=False,
            )


def test_authorization_canonicalizer_rejects_source_q_and_run_state_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    namespace, _parent, receipt = _synthetic_authorization(tmp_path, monkeypatch)
    drifts: list[dict[str, Any]] = []

    source_drift = copy.deepcopy(receipt)
    source_drift["preflight"]["contract"]["source"]["root_sha256"] = _sha("drifted-source-root")
    _refresh_preflight_fingerprint(source_drift)
    drifts.append(source_drift)

    q_drift = copy.deepcopy(receipt)
    q_drift["preflight"]["contract"]["terminal_qualification"] = copy.deepcopy(
        q_drift["preflight"]["contract"]["terminal_qualification"]
    )
    q_drift["preflight"]["contract"]["terminal_qualification"][
        "terminal_qualification_receipt_sha256"
    ] = "0" * 64
    _refresh_preflight_fingerprint(q_drift)
    drifts.append(q_drift)

    run_drift = copy.deepcopy(receipt)
    run_contract = run_drift["preflight"]["contract"]["run_state"]
    run_contract["files"]["registry.csv"] = _sha("drifted-run-registry")
    run_contract["sha256"] = controller._compact_sha256(run_contract["files"])
    _refresh_preflight_fingerprint(run_drift)
    drifts.append(run_drift)

    for tampered in drifts:
        with pytest.raises(controller.ControlError):
            controller._canonical_publication_authorization_v2(
                tampered,
                namespace=namespace,
                verify_live_controller=False,
            )


def test_synthetic_i3_fixture_exposes_all_full_role_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    namespace, _parent, _receipt = _synthetic_authorization(tmp_path, monkeypatch)

    payloads, records, root = controller._read_input_v3(
        namespace,
        verify_live=False,
    )

    assert set(payloads) == set(controller.INPUT_V3_FILENAMES)
    assert set(records) == set(controller.INPUT_V3_FILENAMES)
    assert root == controller._compact_sha256(records)
    assert set(payloads["frozen_source_receipt"]) >= controller._FROZEN_SOURCE_FIELDS
    assert {
        "plan_without_self_hash_sha256",
        "planned_workspace_bytes",
        "required_free_bytes_before",
    } <= set(payloads["workspace_plan"])


@pytest.mark.parametrize(
    "binding",
    [
        "terminal_qualification_receipt_sha256",
        "controller_path",
        "controller_size_bytes",
        "controller_sha256",
        "execution_source_root_sha256",
        "execution_source_manifest_sha256",
        "execution_source_delta_sha256",
        "execution_source_delta_count",
        "config_path",
        "config_file_sha256",
        "config_semantic_sha256",
        "manifest_path",
        "manifest_sha256",
        "failed_preflight_receipt_path",
        "failed_preflight_receipt_sha256",
        "prior_failure_receipt_path",
        "prior_failure_receipt_sha256",
        "retired_input_invalidation_receipt_path",
        "retired_input_invalidation_receipt_sha256",
        "run_state_root",
        "run_state_files",
        "run_state_sha256",
        "authorization_sha256",
        "workspace_plan_without_self_hash_sha256",
        "workspace.plan_without_self_hash_sha256",
        "workspace.planned_workspace_bytes",
        "workspace.required_free_bytes_before",
    ],
)
def test_auth_v2_rejects_semantic_tamper_in_sealed_i3_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    binding: str,
) -> None:
    namespace, _parent, receipt = _synthetic_authorization(tmp_path, monkeypatch)
    tampered = copy.deepcopy(receipt)
    payloads, records, _root = controller._read_input_v3(
        namespace,
        verify_live=False,
    )
    frozen_source = payloads["frozen_source_receipt"]
    workspace_plan = payloads["workspace_plan"]

    if binding.startswith("workspace."):
        field = binding.removeprefix("workspace.")
        current = workspace_plan[field]
        workspace_plan[field] = (
            current + 1 if type(current) is int else _sha(f"tampered-i3:{binding}")
        )
        frozen_source["workspace_plan_sha256"] = hashlib.sha256(
            controller._canonical_bytes(workspace_plan)
        ).hexdigest()
        if field == "plan_without_self_hash_sha256":
            frozen_source["workspace_plan_without_self_hash_sha256"] = workspace_plan[field]
    elif binding.endswith("_path") or binding.endswith("_root"):
        frozen_source[binding] = str((tmp_path / "tampered-i3" / binding).resolve())
    elif binding in {
        "controller_size_bytes",
        "execution_source_delta_count",
    }:
        frozen_source[binding] += 1
    elif binding == "run_state_files":
        frozen_source[binding] = copy.deepcopy(frozen_source[binding])
        first_filename = controller._RUN_STATE_FILENAMES[0]
        frozen_source[binding][first_filename] = _sha("tampered-i3:run-state-file")
    else:
        frozen_source[binding] = _sha(f"tampered-i3:{binding}")

    for role, filename in controller.INPUT_V3_FILENAMES.items():
        encoded = controller._canonical_bytes(payloads[role])
        records[role] = {
            "path": str(namespace.input_v3 / filename),
            "size_bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    records_root = controller._compact_sha256(records)
    contract = tampered["preflight"]["contract"]
    contract["frozen_input_bundle"]["files"] = copy.deepcopy(records)
    contract["frozen_input_bundle"]["records_sha256"] = records_root
    contract["capacity_contract"]["workspace_plan_sha256"] = records["workspace_plan"]["sha256"]
    _refresh_preflight_fingerprint(tampered)
    monkeypatch.setattr(
        controller,
        "_read_input_v3",
        lambda _namespace, *, verify_live=False: (
            copy.deepcopy(payloads),
            copy.deepcopy(records),
            records_root,
        ),
    )

    with pytest.raises(controller.ControlError):
        controller._canonical_publication_authorization_v2(
            tampered,
            namespace=namespace,
            verify_live_controller=False,
        )


@pytest.mark.parametrize(
    ("binding", "keys"),
    [
        (
            "terminal_qualification_receipt_sha256",
            (
                "terminal_qualification",
                "terminal_qualification_receipt_sha256",
            ),
        ),
        ("controller_path", ("controller", "path")),
        ("controller_size_bytes", ("controller", "size_bytes")),
        ("controller_sha256", ("controller", "sha256")),
        ("source_root_sha256", ("source", "root_sha256")),
        ("source_manifest_sha256", ("source", "manifest_sha256")),
        ("source_delta_sha256", ("source", "delta_sha256")),
        ("source_count", ("source", "allowlisted_change_count")),
        ("config_path", ("config", "path")),
        ("config_file_sha256", ("config", "file_sha256")),
        ("config_semantic_sha256", ("config", "semantic_sha256")),
        ("manifest_path", ("manifest", "path")),
        ("manifest_sha256", ("manifest", "sha256")),
        (
            "failed_preflight_receipt_path",
            ("historical_lineage", "failed_preflight_receipt_path"),
        ),
        (
            "failed_preflight_receipt_sha256",
            ("historical_lineage", "failed_preflight_receipt_sha256"),
        ),
        (
            "prior_failure_receipt_path",
            ("historical_lineage", "prior_failure_receipt_path"),
        ),
        (
            "prior_failure_receipt_sha256",
            ("historical_lineage", "prior_failure_receipt_sha256"),
        ),
        (
            "retired_input_invalidation_receipt_path",
            (
                "historical_lineage",
                "retired_input_invalidation_receipt_path",
            ),
        ),
        (
            "retired_input_invalidation_receipt_sha256",
            (
                "historical_lineage",
                "retired_input_invalidation_receipt_sha256",
            ),
        ),
        ("run_state_root", ("run_state", "root")),
        ("run_state_files", ("run_state", "files")),
        ("run_state_sha256", ("run_state", "sha256")),
        (
            "authorization_sha256",
            ("technical_successor", "authorization_sha256"),
        ),
        (
            "workspace_record_sha256",
            (
                "frozen_input_bundle",
                "files",
                "workspace_plan",
                "sha256",
            ),
        ),
        (
            "capacity_policy_sha256",
            ("capacity_contract", "resource_capacity_policy_sha256"),
        ),
        (
            "capacity_workspace_plan_sha256",
            ("capacity_contract", "workspace_plan_sha256"),
        ),
        (
            "capacity_workspace_without_self_sha256",
            (
                "capacity_contract",
                "workspace_plan_without_self_hash_sha256",
            ),
        ),
        (
            "capacity_projected_run",
            ("capacity_contract", "projected_stable_run_bytes"),
        ),
        (
            "capacity_safety_margin",
            ("capacity_contract", "fixed_safety_margin_bytes"),
        ),
        (
            "capacity_tracker_minimum",
            ("capacity_contract", "minimum_free_bytes_before_tracker"),
        ),
        (
            "capacity_maximum_workspace",
            ("capacity_contract", "maximum_workspace_bytes"),
        ),
        (
            "capacity_workspace_minimum",
            (
                "capacity_contract",
                "minimum_free_bytes_before_workspace_build",
            ),
        ),
        (
            "capacity_planned_workspace",
            ("capacity_contract", "planned_workspace_bytes"),
        ),
        (
            "capacity_required_before",
            ("capacity_contract", "required_free_bytes_before"),
        ),
        (
            "capacity_required",
            ("capacity_contract", "required_free_bytes"),
        ),
        ("storage_policy", ("technical_successor", "storage_policy")),
        ("intent_sha256", ("technical_successor", "intent_sha256")),
    ],
)
def test_auth_v2_rejects_tamper_in_every_u_full_contract_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    binding: str,
    keys: tuple[str, ...],
) -> None:
    namespace, _parent, receipt = _synthetic_authorization(tmp_path, monkeypatch)
    tampered = copy.deepcopy(receipt)
    contract = tampered["preflight"]["contract"]
    target: Any = contract
    for key in keys[:-1]:
        target = target[key]
    current = target[keys[-1]]

    if binding == "run_state_files":
        replacement = copy.deepcopy(current)
        replacement[controller._RUN_STATE_FILENAMES[0]] = _sha("tampered-u:run-state-file")
        target[keys[-1]] = replacement
        contract["run_state"]["sha256"] = controller._compact_sha256(replacement)
    elif binding == "storage_policy":
        target[keys[-1]] = {"policy": "forged-storage-policy"}
    elif binding == "workspace_record_sha256":
        target[keys[-1]] = _sha(f"tampered-u:{binding}")
        contract["frozen_input_bundle"]["records_sha256"] = controller._compact_sha256(
            contract["frozen_input_bundle"]["files"]
        )
    elif binding.endswith("_path") or binding.endswith("_root"):
        target[keys[-1]] = str((tmp_path / "tampered-u" / binding).resolve())
    elif type(current) is int:
        target[keys[-1]] = current + 1
    else:
        target[keys[-1]] = _sha(f"tampered-u:{binding}")
    _refresh_preflight_fingerprint(tampered)

    with pytest.raises(controller.ControlError):
        controller._canonical_publication_authorization_v2(
            tampered,
            namespace=namespace,
            verify_live_controller=False,
        )
