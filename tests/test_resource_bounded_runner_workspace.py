"""Workspace lifecycle regressions for the resource-bounded runner.

The tests use injected providers and tiny metadata-only workspace objects.  They
therefore exercise the runner's ownership, ordering, and provenance contract
without materialising the real 4.3-GB PanNuke workspace.
"""

from __future__ import annotations

import inspect
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_resource_bounded_runner_execution import (
    _assert_zero_scientific_records,
    _build_harness,
)

import histo_audit.experiment.resource_bounded_runner as runner
from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.experiment.confirmatory_memory_workspace import (
    ConfirmatoryMemoryWorkspace,
    ConfirmatoryMemoryWorkspaceError,
)
from histo_audit.experiment.resource_bounded_runner import (
    RESOURCE_BOUNDED_ANALYSIS_DISPOSITION,
    ResourceBoundedRunnerDependencies,
    ResourceBoundedStudyRunnerError,
    _run_resource_bounded_sensitivity,
    execute_resource_bounded_sensitivity,
    preflight_resource_bounded_sensitivity,
)
from histo_audit.utils.run_tracking import sha256_file, verify_run_integrity

_SHA = "a" * 64


@dataclass(slots=True)
class _WorkspaceProbe:
    """Captured injected-provider state for one runner invocation."""

    workspace: ConfirmatoryMemoryWorkspace
    read_only_workspace: ConfirmatoryMemoryWorkspace
    receipt: dict[str, Any]
    cleanup: dict[str, Any]
    array_specs: tuple[Any, ...]
    index_specs: tuple[Any, ...]
    plan: dict[str, Any]
    capacity: dict[str, Any]
    calls: dict[str, list[Any]]

    @property
    def cleanup_sha256(self) -> str:
        return canonical_sha256(self.cleanup)


def _write_workspace_receipt(
    workspace_root: Path,
    *,
    workspace_key: str,
    plan_sha256: str,
    artifact_root_sha256: str,
) -> tuple[dict[str, Any], str]:
    workspace_root.mkdir(parents=True)
    base = {
        "schema_version": 1,
        "recipe_id": "pannuke_confirmatory_shared_memmap_workspace_v1",
        "status": "complete",
        "workspace_key": workspace_key,
        "workspace_reuse_allowed": False,
        "resource_input_workspace_plan_sha256": plan_sha256,
        "artifact_root_sha256": artifact_root_sha256,
        "arrays": [],
        "indices": [],
        "capacity": {
            "minimum_free_bytes_after": 0,
            "maximum_workspace_bytes": 1,
            "planned_workspace_bytes": 0,
        },
        "source_annotations_modified": False,
        "scientific_outcomes_read": False,
    }
    receipt = {
        **base,
        "receipt_without_self_hash_sha256": canonical_sha256(base),
    }
    receipt_path = workspace_root / "workspace_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt, sha256_file(receipt_path)


def _install_workspace_probe(
    harness: Any,
    *,
    workspace_builder_error: BaseException | None = None,
    input_builder_error: BaseException | None = None,
    bridge_error: BaseException | None = None,
    cleanup_error: BaseException | None = None,
    workspace_reuse_error: bool = False,
) -> _WorkspaceProbe:
    """Install exact injected workspace providers on an existing runner harness."""

    authority = harness.gate.execution_authority
    plan = dict(authority.resource_input_workspace_plan)
    capacity = dict(authority.resource_capacity_policy)
    plan_sha = str(authority.resource_input_workspace_plan_sha256)
    workspace_key = str(plan["workspace_key"])
    artifact_root = "b" * 64
    workspace_root = (
        harness.project_root / "artifacts" / "resource_control" / "input_workspaces" / workspace_key
    )
    receipt, receipt_sha = _write_workspace_receipt(
        workspace_root,
        workspace_key=workspace_key,
        plan_sha256=plan_sha,
        artifact_root_sha256=artifact_root,
    )
    workspace = ConfirmatoryMemoryWorkspace(
        root=workspace_root,
        workspace_key=workspace_key,
        receipt_sha256=receipt_sha,
        artifact_root_sha256=artifact_root,
        arrays={},
        index_arrays={},
        resource_input_workspace_plan_sha256=plan_sha,
        cleanup_ownership_token="fixture-builder-ownership",
    )
    read_only_workspace = replace(workspace, cleanup_ownership_token=None)
    cleanup = {
        "schema_version": 1,
        "status": "complete",
        "workspace_key": workspace_key,
        "workspace_receipt_sha256": receipt_sha,
        "artifact_root_sha256": artifact_root,
        "resource_input_workspace_plan_sha256": plan_sha,
        "workspace_removed": True,
        "source_annotations_modified": False,
        "scientific_outcomes_read": False,
    }
    array_specs = tuple(
        SimpleNamespace(array_id=f"fixture_array_{index:02d}") for index in range(12)
    )
    roles = ("audit", "reference_validation", "final_reference")
    index_specs = tuple(
        SimpleNamespace(outer_fold=fold, role=role) for fold in (1, 2, 3) for role in roles
    )
    calls: dict[str, list[Any]] = {
        "array_specs": [],
        "index_specs": [],
        "build": [],
        "verify": [],
        "input": [],
        "clean": [],
        "disk": [],
    }
    base_input_builder = harness.dependencies.input_builder
    base_bridge_builder = harness.dependencies.bridge_builder
    disk_thresholds = (
        int(capacity["minimum_free_bytes_before_workspace_build"]),
        int(capacity["minimum_free_bytes_before_tracker"]),
    )

    def array_spec_builder(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        harness.events.append("workspace_array_specs")
        calls["array_specs"].append((args, kwargs))
        return array_specs

    def index_spec_builder(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        harness.events.append("workspace_index_specs")
        calls["index_specs"].append((args, kwargs))
        return index_specs

    def workspace_builder(*args: Any, **kwargs: Any) -> ConfirmatoryMemoryWorkspace:
        harness.events.append("workspace_build")
        calls["build"].append((args, kwargs))
        if workspace_builder_error is not None:
            # The public provider owns rollback until it returns the ownership
            # token.  Model that boundary explicitly: the runner must not try to
            # clean a workspace object it never received.
            shutil.rmtree(workspace.root)
            harness.events.append("workspace_builder_rollback")
            raise workspace_builder_error
        if workspace_reuse_error:
            raise ConfirmatoryMemoryWorkspaceError(
                "capacity-v3 forbids reuse of an existing workspace destination"
            )
        assert len(calls["build"]) == 1
        assert kwargs["resource_input_workspace_plan"] == plan
        assert kwargs["resource_input_workspace_plan"]["workspace_reuse_allowed"] is False
        assert kwargs["index_specs"] == index_specs
        assert kwargs["minimum_free_bytes_after"] == capacity["minimum_free_bytes_before_tracker"]
        assert kwargs["maximum_workspace_bytes"] == capacity["maximum_workspace_bytes"]
        return workspace

    def workspace_verifier(*args: Any, **kwargs: Any) -> ConfirmatoryMemoryWorkspace:
        harness.events.append("workspace_verify")
        calls["verify"].append((args, kwargs))
        assert args[0] == harness.project_root
        assert args[1] == workspace.root
        assert args[2] == array_specs
        assert kwargs["resource_input_workspace_plan"] == plan
        assert kwargs["index_specs"] == index_specs
        assert kwargs["minimum_free_bytes_after"] == capacity["minimum_free_bytes_before_tracker"]
        assert kwargs["maximum_workspace_bytes"] == capacity["maximum_workspace_bytes"]
        return read_only_workspace

    def input_builder(*args: Any, **kwargs: Any) -> Any:
        harness.events.append("input_builder")
        calls["input"].append((args, kwargs))
        assert kwargs.get("memory_workspace") is workspace
        if input_builder_error is not None:
            raise input_builder_error
        prepared = base_input_builder(*args, **kwargs)
        prepared.memory_workspace_path = str(workspace.root)
        prepared.memory_workspace_receipt_sha256 = workspace.receipt_sha256
        prepared.memory_workspace_artifact_root_sha256 = workspace.artifact_root_sha256
        prepared.memory_workspace_plan_sha256 = workspace.resource_input_workspace_plan_sha256
        return prepared

    def bridge_builder(*args: Any, **kwargs: Any) -> Any:
        harness.events.append("bridge")
        if bridge_error is not None:
            raise bridge_error
        return base_bridge_builder(*args, **kwargs)

    def workspace_cleaner(*args: Any, **kwargs: Any) -> dict[str, Any]:
        harness.events.append("workspace_cleanup")
        calls["clean"].append((args, kwargs))
        assert len(calls["clean"]) == 1
        assert args[0] == harness.project_root
        assert args[1] is workspace
        assert args[2] == array_specs
        assert kwargs["resource_input_workspace_plan"] == plan
        assert kwargs["index_specs"] == index_specs
        assert kwargs["minimum_free_bytes_after"] == capacity["minimum_free_bytes_before_tracker"]
        assert kwargs["maximum_workspace_bytes"] == capacity["maximum_workspace_bytes"]
        if cleanup_error is not None:
            raise cleanup_error
        shutil.rmtree(workspace.root)
        return cleanup

    def disk_usage(path: Path) -> Any:
        call_index = len(calls["disk"])
        harness.events.append(f"workspace_disk_{call_index + 1}")
        calls["disk"].append(path)
        threshold = disk_thresholds[min(call_index, len(disk_thresholds) - 1)]
        return SimpleNamespace(free=threshold)

    dependencies = replace(
        harness.dependencies,
        workspace_array_spec_builder=array_spec_builder,
        workspace_index_spec_builder=index_spec_builder,
        workspace_builder=workspace_builder,
        workspace_verifier=workspace_verifier,
        workspace_cleaner=workspace_cleaner,
        input_builder=input_builder,
        bridge_builder=bridge_builder,
        disk_usage=disk_usage,
    )
    harness.dependencies = dependencies
    harness.kwargs["dependencies"] = dependencies
    return _WorkspaceProbe(
        workspace=workspace,
        read_only_workspace=read_only_workspace,
        receipt=receipt,
        cleanup=cleanup,
        array_specs=array_specs,
        index_specs=index_specs,
        plan=plan,
        capacity=capacity,
        calls=calls,
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_preflight_uses_distinct_workspace_and_tracker_thresholds_then_cleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _build_harness(tmp_path, monkeypatch)
    probe = _install_workspace_probe(harness)
    harness.kwargs["preflight_only"] = True

    result = _run_resource_bounded_sensitivity(**harness.kwargs)

    assert result["status"] == "passed"
    assert result["tracker_created"] is False
    assert harness.trackers == []
    assert [item["phase"] for item in result["capacity_checks"]] == [
        "guarded_before_workspace_build",
        "guarded_immediately_before_tracker",
    ]
    assert [item["minimum_free_bytes"] for item in result["capacity_checks"]] == [
        probe.capacity["minimum_free_bytes_before_workspace_build"],
        probe.capacity["minimum_free_bytes_before_tracker"],
    ]
    assert harness.events.index("workspace_disk_1") < harness.events.index("workspace_build")
    assert harness.events.index("workspace_build") < harness.events.index("input_builder")
    assert harness.events.index("workspace_verify") < harness.events.index("workspace_disk_2")
    assert harness.events.index("workspace_disk_2") < harness.events.index("workspace_cleanup")
    assert len(probe.calls["clean"]) == 1
    assert not probe.workspace.root.exists()
    assert result["resource_input_workspace_plan_sha256"] == (
        probe.workspace.resource_input_workspace_plan_sha256
    )
    assert result["resource_input_workspace_receipt_sha256"] == (probe.workspace.receipt_sha256)
    assert result["resource_input_workspace_artifact_root_sha256"] == (
        probe.workspace.artifact_root_sha256
    )
    assert result["resource_input_workspace_cleanup_sha256"] == probe.cleanup_sha256
    assert result["resource_input_workspace_removed"] is True


def test_execution_binds_exact_workspace_evidence_and_cleans_before_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _build_harness(tmp_path, monkeypatch)
    probe = _install_workspace_probe(harness)

    result = _run_resource_bounded_sensitivity(**harness.kwargs)

    tracker = harness.trackers[0]
    run_directory = tracker.run_directory
    assert result["status"] == "completed"
    assert result["completion_stage"] is None
    assert result["study_outcome_eligible"] is False
    assert result["analysis_disposition"] == RESOURCE_BOUNDED_ANALYSIS_DISPOSITION
    assert result["resource_input_workspace_plan_sha256"] == (
        probe.workspace.resource_input_workspace_plan_sha256
    )
    assert result["resource_input_workspace_receipt_sha256"] == (probe.workspace.receipt_sha256)
    assert result["resource_input_workspace_artifact_root_sha256"] == (
        probe.workspace.artifact_root_sha256
    )
    assert result["resource_input_workspace_cleanup_sha256"] == probe.cleanup_sha256
    assert _load_json(run_directory / "resource_input_workspace_plan.json") == probe.plan
    assert _load_json(run_directory / "resource_input_workspace_receipt.json") == probe.receipt
    assert _load_json(run_directory / "resource_input_workspace_cleanup.json") == probe.cleanup
    completion = _load_json(run_directory / "completion_evidence.json")
    provenance = _load_json(run_directory / "run_provenance.json")
    for carrier in (completion, provenance):
        assert carrier["resource_input_workspace_plan_sha256"] == (
            probe.workspace.resource_input_workspace_plan_sha256
        )
        assert carrier["resource_input_workspace_receipt_sha256"] == (
            probe.workspace.receipt_sha256
        )
        assert carrier["resource_input_workspace_artifact_root_sha256"] == (
            probe.workspace.artifact_root_sha256
        )
        assert carrier["resource_input_workspace_cleanup_sha256"] == probe.cleanup_sha256
    assert len(probe.calls["input"]) == 1
    assert probe.calls["input"][0][1]["memory_workspace"] is probe.workspace
    assert len(probe.calls["verify"]) == 1
    assert len(probe.calls["clean"]) == 1
    assert harness.events.index("workspace_disk_2") < harness.events.index("tracker_start")
    assert harness.events.index("compute_2") < harness.events.index("tracker_start")
    assert harness.events.index("workspace_cleanup") < harness.events.index("gate_2")
    assert harness.events.index("workspace_cleanup") < harness.events.index("complete")
    assert not probe.workspace.root.exists()
    assert verify_run_integrity(run_directory).valid
    _assert_zero_scientific_records(harness.run_root, tracker.run_id)


@pytest.mark.parametrize(
    ("fault_phase", "expected_tracker_count"),
    [
        ("input_builder", 0),
        ("bridge", 0),
        ("matrix", 1),
        ("finalization", 1),
    ],
)
def test_workspace_cleanup_runs_once_on_pipeline_faults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_phase: str,
    expected_tracker_count: int,
) -> None:
    fault = RuntimeError(f"injected {fault_phase} fault")
    harness = _build_harness(
        tmp_path,
        monkeypatch,
        matrix_error=fault if fault_phase == "matrix" else None,
    )
    probe = _install_workspace_probe(
        harness,
        input_builder_error=fault if fault_phase == "input_builder" else None,
        bridge_error=fault if fault_phase == "bridge" else None,
    )
    if fault_phase == "finalization":
        monkeypatch.setattr(
            runner,
            "finalize_resource_bounded_analysis",
            lambda **_kwargs: (_ for _ in ()).throw(fault),
        )

    with pytest.raises(RuntimeError, match=f"injected {fault_phase} fault"):
        _run_resource_bounded_sensitivity(**harness.kwargs)

    assert len(probe.calls["clean"]) == 1
    assert not probe.workspace.root.exists()
    assert len(harness.trackers) == expected_tracker_count
    if harness.trackers:
        tracker = harness.trackers[0]
        status = _load_json(tracker.run_directory / "status.json")
        completion = _load_json(tracker.run_directory / "completion_evidence.json")
        assert status["status"] == "failed"
        assert completion["completion_stage"] is None
        assert completion["study_outcome_eligible"] is False
        assert completion["valid_completion_claim"] is False
        assert completion["analysis_disposition"] == RESOURCE_BOUNDED_ANALYSIS_DISPOSITION
        assert tracker.complete_calls == 0
        assert tracker.fail_calls == 1
        assert verify_run_integrity(tracker.run_directory).valid
        _assert_zero_scientific_records(harness.run_root, tracker.run_id)


def test_workspace_builder_failure_rolls_back_inside_provider_before_tracker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fault = RuntimeError("injected workspace builder fault")
    harness = _build_harness(tmp_path, monkeypatch)
    probe = _install_workspace_probe(harness, workspace_builder_error=fault)

    with pytest.raises(RuntimeError, match="injected workspace builder fault"):
        _run_resource_bounded_sensitivity(**harness.kwargs)

    assert harness.events.index("workspace_build") < harness.events.index(
        "workspace_builder_rollback"
    )
    assert not probe.workspace.root.exists()
    assert probe.calls["clean"] == []
    assert probe.calls["verify"] == []
    assert probe.calls["input"] == []
    assert harness.trackers == []


def test_cleanup_failure_demotes_to_one_failed_nonclaiming_seal_without_second_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_fault = RuntimeError("injected cleanup fault")
    harness = _build_harness(tmp_path, monkeypatch)
    probe = _install_workspace_probe(harness, cleanup_error=cleanup_fault)

    with pytest.raises(RuntimeError, match="injected cleanup fault"):
        _run_resource_bounded_sensitivity(**harness.kwargs)

    assert len(probe.calls["clean"]) == 1
    assert len(harness.trackers) == 1
    tracker = harness.trackers[0]
    assert tracker.complete_calls == 0
    assert tracker.fail_calls == 1
    status = _load_json(tracker.run_directory / "status.json")
    completion = _load_json(tracker.run_directory / "completion_evidence.json")
    assert status["status"] == "failed"
    assert completion["completion_stage"] is None
    assert completion["study_outcome_eligible"] is False
    assert completion["valid_completion_claim"] is False
    assert completion["analysis_disposition"] == RESOURCE_BOUNDED_ANALYSIS_DISPOSITION
    assert verify_run_integrity(tracker.run_directory).valid
    _assert_zero_scientific_records(harness.run_root, tracker.run_id)


def test_existing_workspace_is_not_adopted_or_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _build_harness(tmp_path, monkeypatch)
    probe = _install_workspace_probe(harness, workspace_reuse_error=True)
    harness.kwargs["preflight_only"] = True

    with pytest.raises(
        ConfirmatoryMemoryWorkspaceError,
        match="forbids reuse",
    ):
        _run_resource_bounded_sensitivity(**harness.kwargs)

    assert probe.plan["workspace_reuse_allowed"] is False
    assert len(probe.calls["build"]) == 1
    assert probe.calls["verify"] == []
    assert probe.calls["input"] == []
    assert probe.calls["clean"] == []
    assert harness.trackers == []


def test_public_preflight_and_execute_parameters_remain_identical() -> None:
    preflight = inspect.signature(preflight_resource_bounded_sensitivity)
    execute = inspect.signature(execute_resource_bounded_sensitivity)

    assert tuple(preflight.parameters) == tuple(execute.parameters)
    for name in preflight.parameters:
        before = preflight.parameters[name]
        after = execute.parameters[name]
        assert before.kind == after.kind
        assert before.default == after.default
        assert before.annotation == after.annotation
    assert preflight.return_annotation == execute.return_annotation


def test_capacity_v3_rejects_legacy_workspace_free_space_at_first_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _build_harness(tmp_path, monkeypatch)
    probe = _install_workspace_probe(harness)
    harness.kwargs["preflight_only"] = True
    legacy_only_free = int(probe.capacity["minimum_free_bytes_before_tracker"])
    assert legacy_only_free < int(probe.capacity["minimum_free_bytes_before_workspace_build"])
    dependencies: ResourceBoundedRunnerDependencies = replace(
        harness.dependencies,
        disk_usage=lambda _path: SimpleNamespace(free=legacy_only_free),
    )
    harness.kwargs["dependencies"] = dependencies

    with pytest.raises(ResourceBoundedStudyRunnerError, match="disk capacity"):
        _run_resource_bounded_sensitivity(**harness.kwargs)

    assert probe.calls["build"] == []
    assert probe.calls["clean"] == []
    assert harness.trackers == []
