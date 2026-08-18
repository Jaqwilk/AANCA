"""Integrated execution-boundary tests for the non-claiming resource runner."""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_resource_bounded_confirmatory_contract import _resource_gate

import histo_audit.experiment.resource_bounded_runner as runner
from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.experiment.confirmatory_completion import (
    ConfirmatoryFilesystemReadback,
)
from histo_audit.experiment.confirmatory_core import (
    ConfirmatoryMatrixArtifacts,
    ConfirmatoryMatrixReconciliation,
)
from histo_audit.experiment.confirmatory_memory_workspace import (
    ConfirmatoryMemoryWorkspace,
)
from histo_audit.experiment.confirmatory_statistics import (
    ConfirmatoryStatisticsArtifacts,
    ConfirmatoryStatisticsVerification,
)
from histo_audit.experiment.resource_bounded_resume import (
    RESOURCE_BOUNDED_RESUME_EVIDENCE_HASH_FIELD,
    ResumeCheckpointExpectation,
)
from histo_audit.experiment.resource_bounded_runner import (
    RESOURCE_BOUNDED_ANALYSIS_DISPOSITION,
    RESOURCE_BOUNDED_CAPACITY_POLICY,
    RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
    ResourceBoundedRunnerDependencies,
    ResourceBoundedStudyIntegrityError,
    _run_resource_bounded_sensitivity,
)
from histo_audit.utils.run_tracking import (
    RUN_DISPOSITION_REGISTRY_FILENAME,
    RUN_STAGE_ATTESTATION_REGISTRY_FILENAME,
    RunTracker,
    capture_source_tree,
    read_run_dispositions,
    read_run_stage_attestations,
    sha256_file,
    verify_run_integrity,
)

_ROOT = Path(__file__).resolve().parents[1]
_RESOURCE_CONFIG = _ROOT / "configs" / "confirmatory_resource_bounded_amended.yaml"
_SHA = "a" * 64


class _RecordingTracker:
    """Record terminal calls while retaining the production RunTracker lifecycle."""

    def __init__(self, inner: RunTracker, events: list[str]) -> None:
        self.inner = inner
        self.events = events
        self.complete_calls = 0
        self.fail_calls = 0
        self.completion_writes = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def write_json(self, relative_path: str | Path, value: Any) -> Path:
        if Path(relative_path).as_posix() == "completion_evidence.json":
            self.completion_writes += 1
            self.events.append("completion_write")
        return self.inner.write_json(relative_path, value)

    def complete(self) -> None:
        self.complete_calls += 1
        self.events.append("complete")
        self.inner.complete()

    def fail(self, error: BaseException) -> None:
        self.fail_calls += 1
        self.events.append("fail")
        self.inner.fail(error)


@dataclass(slots=True)
class _Harness:
    project_root: Path
    run_root: Path
    dependencies: ResourceBoundedRunnerDependencies
    kwargs: dict[str, Any]
    events: list[str]
    trackers: list[_RecordingTracker]
    gate: Any


class _Bridge:
    def __init__(self) -> None:
        self.rotations: tuple[Any, ...] = ()
        self.frozen_blockers: dict[str, Any] = {}

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "status": "bound"}


def _checkpoint_allowlist() -> tuple[ResumeCheckpointExpectation, ...]:
    fingerprints = {
        "training_data_sha256": "1" * 64,
        "reference_validation_data_sha256": "2" * 64,
        "training_split_sha256": "3" * 64,
        "reference_validation_split_sha256": "4" * 64,
    }
    return tuple(
        ResumeCheckpointExpectation(
            relative_path=f"cells/cnn_{cell_index}/checkpoints/fold_{fold_id:02d}.pt",
            cell_id=f"cnn_{cell_index}",
            fold_id=fold_id,
            expected_configuration={"epochs": 4},
            expected_model_metadata={"architecture": "test-resnet18"},
            expected_data_and_split_sha256=fingerprints,
        )
        for cell_index in range(6)
        for fold_id in range(5)
    )


def _reconciliation(plan: Any) -> ConfirmatoryMatrixReconciliation:
    outer_folds = tuple(sorted({cell.outer_fold for cell in plan.cells}))
    return ConfirmatoryMatrixReconciliation(
        status="passed",
        fold_rotation_complete=True,
        planned_cell_count=len(plan.cells),
        planned_required_cell_count=plan.required_cell_count,
        completed_cell_count=len(plan.cells),
        completed_required_cell_count=plan.required_cell_count,
        skipped_optional_cell_count=0,
        failed_cell_count=0,
        planned_outer_folds=outer_folds,
        completed_outer_folds=outer_folds,
        incomplete_outer_folds=(),
        missing_cell_ids=(),
        extra_cell_ids=(),
        duplicate_cell_ids=(),
        invalid_cell_ids=(),
        errors=(),
    )


def _matrix_artifacts(
    output_directory: Path,
    plan: Any,
) -> ConfirmatoryMatrixArtifacts:
    paths = {
        "matrix_plan_path": output_directory / "matrix_plan.json",
        "execution_controls_path": output_directory / "execution_controls.json",
        "frozen_feature_provenance_path": output_directory / "frozen_feature_provenance.json",
        "original_audit_selection_path": output_directory / "original_audit_selection.json",
        "cell_index_path": output_directory / "cell_index.csv",
        "ensemble_evidence_path": output_directory / "ensemble_evidence.json",
        "hybrid_ablations_path": output_directory / "fixed_hybrid_drop_one_ablations.json",
        "fold_aggregate_path": output_directory / "fold_aggregate.json",
        "reconciliation_path": output_directory / "reconciliation.json",
        "completion_evidence_path": output_directory / "matrix_completion.json",
        "analysis_gaps_path": output_directory / "analysis_gaps.json",
        "figure_manifest_path": output_directory / "figure_manifest.json",
        "report_path": output_directory / "report.md",
        "artifact_manifest_path": output_directory / "matrix_artifact_manifest.json",
    }
    for path in paths.values():
        path.write_text("{}\n", encoding="utf-8")
    outcomes = tuple(
        {
            "cell_id": cell.cell_id,
            "required": True,
            "status": "completed",
            "outer_fold": cell.outer_fold,
            "model_seed": cell.model_seed,
            "scenario_id": cell.scenario_id,
            "corruption_cell_id": cell.corruption_cell_id,
        }
        for cell in plan.cells
    )
    return ConfirmatoryMatrixArtifacts(
        output_directory=output_directory,
        outcomes=outcomes,
        reconciliation=_reconciliation(plan),
        study_outcome_eligible=False,
        **paths,
    )


def _registry_rows(run_root: Path, run_id: str) -> list[dict[str, str]]:
    with (run_root / "registry.csv").open(encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("run_id") == run_id]


def _assert_zero_scientific_records(run_root: Path, run_id: str) -> None:
    attestations = read_run_stage_attestations(run_root / RUN_STAGE_ATTESTATION_REGISTRY_FILENAME)
    dispositions = read_run_dispositions(run_root / RUN_DISPOSITION_REGISTRY_FILENAME)
    assert [row for row in attestations if row.get("run_id") == run_id] == []
    assert [row for row in dispositions if row.get("run_id") == run_id] == []


def _build_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    matrix_error: BaseException | None = None,
    postseal_integrity_fault: bool = False,
) -> _Harness:
    events: list[str] = []
    trackers: list[_RecordingTracker] = []
    project_root = tmp_path.resolve()
    run_root = project_root / "artifacts" / "runs"
    authority = project_root / "artifacts" / "authorities" / "C"
    authority.mkdir(parents=True)
    config_path = authority / "confirmatory_frozen.yaml"
    shutil.copyfile(_RESOURCE_CONFIG, config_path)
    primary = run_root / "primary"
    readiness = run_root / "readiness"
    dataset = project_root / "data" / "raw" / "pannuke"
    dataset.mkdir(parents=True)
    (dataset / "fixture.bin").write_bytes(b"dataset")
    manifest = project_root / "data" / "manifests" / "manifest.parquet"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"manifest")
    duplicate_audit = project_root / "artifacts" / "duplicate.json"
    duplicate_audit.write_text("{}\n", encoding="utf-8")
    pathology_audit = project_root / "reports" / "pathology.json"
    pathology_audit.parent.mkdir(parents=True)
    pathology_audit.write_text("{}\n", encoding="utf-8")
    crop_cache = project_root / "artifacts" / "crops.npz"
    crop_cache.write_bytes(b"crop")

    source_root = str(capture_source_tree(project_root)["root_sha256"])
    base_gate = _resource_gate()
    primary_gate = replace(
        base_gate.historical_primary.primary_gate,
        manifest_sha256=sha256_file(manifest),
    )
    historical = replace(
        base_gate.historical_primary,
        primary_gate=primary_gate,
        primary_run_directory=primary.resolve(),
    )
    execution_authority = replace(
        base_gate.execution_authority,
        authority_directory=authority.resolve(),
        resource_confirmatory_config_file_sha256=sha256_file(config_path),
        resource_execution_source_root_sha256=source_root,
    )
    gate = replace(
        base_gate,
        historical_primary=historical,
        execution_authority=execution_authority,
    )

    weight_path = project_root / "artifacts" / "weights" / "resnet18.pth"
    weight_path.parent.mkdir(parents=True)
    weight_path.write_bytes(b"test weight")
    real_runner_sha256 = runner.sha256_file

    def selective_sha256(path: str | Path) -> str:
        if Path(path).resolve() == weight_path.resolve():
            return str(RESOURCE_BOUNDED_CAPACITY_POLICY["official_weight_sha256"])
        return real_runner_sha256(path)

    monkeypatch.setattr(runner, "sha256_file", selective_sha256)
    monkeypatch.setattr(runner, "_verify_cache_files", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "_confirmatory_cnn_preflight_fingerprints",
        lambda *_args, **_kwargs: {"fixture": {"0": {"sha256": _SHA}}},
    )
    monkeypatch.setattr(
        runner,
        "build_resource_checkpoint_allowlist",
        lambda **_kwargs: _checkpoint_allowlist(),
    )
    monkeypatch.setattr(
        runner,
        "_validate_core_artifacts",
        lambda *_args, **_kwargs: {
            "model_matrix_execution_eligible": True,
            "matrix_execution_telemetry_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(
        runner,
        "_validate_restoration_source_binding",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runner,
        "_validate_final_readback",
        lambda *_args, **_kwargs: None,
    )

    @contextmanager
    def fake_primary_guard(path: Path) -> Iterator[Any]:
        assert path == primary.resolve()
        events.append("primary_guard")
        yield SimpleNamespace(valid=True)

    monkeypatch.setattr(runner, "guard_run_stage_eligibility", fake_primary_guard)

    gate_calls = 0

    def gate_validator(**kwargs: Any) -> Any:
        nonlocal gate_calls
        gate_calls += 1
        events.append(f"gate_{gate_calls}")
        assert kwargs["primary_stage_eligibility_receipt"] is not None
        return gate

    def tracker_starter(**kwargs: Any) -> _RecordingTracker:
        events.append("tracker_start")
        tracker = _RecordingTracker(RunTracker.start(**kwargs), events)
        trackers.append(tracker)
        return tracker

    def matrix_executor(
        _rotations: Any,
        plan: Any,
        _controls: Any,
        *,
        output_directory: Path,
        **kwargs: Any,
    ) -> ConfirmatoryMatrixArtifacts:
        events.append("matrix")
        assert kwargs["artifact_scope"] == RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE
        checkpoint_contract = kwargs["checkpoint_execution_contract"]
        assert checkpoint_contract.contract_profile == ("resource_bounded_confirmatory_exact_30")
        assert checkpoint_contract.execution_mode == "fresh"
        assert len(checkpoint_contract.directives) == 30
        assert kwargs["gate_evidence"] == gate
        if matrix_error is not None:
            raise matrix_error
        return _matrix_artifacts(output_directory, plan)

    statistics_state: dict[str, ConfirmatoryStatisticsArtifacts] = {}

    def statistics_aggregator(
        output_directory: str | Path,
        _controls: Any,
    ) -> ConfirmatoryStatisticsArtifacts:
        events.append("statistics")
        output = Path(output_directory)
        statistics_path = output / "paired_statistics.json"
        bootstrap_path = output / "paired_bootstrap_evidence.npz"
        statistics_path.write_text('{"status":"passed"}\n', encoding="utf-8")
        bootstrap_path.write_bytes(b"bootstrap")
        artifacts = ConfirmatoryStatisticsArtifacts(
            output_directory=output,
            statistics_path=statistics_path,
            bootstrap_evidence_path=bootstrap_path,
            comparison_count=1,
            completed_comparison_count=1,
            statistics_sha256=sha256_file(statistics_path),
            bootstrap_evidence_sha256=sha256_file(bootstrap_path),
        )
        statistics_state["value"] = artifacts
        return artifacts

    def statistics_verifier(
        output_directory: str | Path,
        _controls: Any,
    ) -> ConfirmatoryStatisticsVerification:
        events.append("statistics_verify")
        artifacts = statistics_state["value"]
        return ConfirmatoryStatisticsVerification(
            status="passed",
            output_directory=Path(output_directory),
            statistics_sha256=artifacts.statistics_sha256,
            bootstrap_evidence_sha256=artifacts.bootstrap_evidence_sha256,
            comparison_count=1,
            completed_comparison_count=1,
        )

    def finalize_analysis(*, run_directory: str | Path, **_kwargs: Any) -> None:
        events.append("finalize_analysis")
        Path(run_directory, "report.md").write_text(
            "resource-bounded sensitivity\n"
            "potentially inconsistent annotation\n"
            "recommended for expert review\n"
            "cannot unlock external-validation milestone M9\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(runner, "finalize_resource_bounded_analysis", finalize_analysis)

    def filesystem_reader(
        plan: Any,
        run_directory: str | Path,
        **kwargs: Any,
    ) -> ConfirmatoryFilesystemReadback:
        phase = (
            "readback_postseal" if kwargs["require_final_policy_bindings"] else "readback_preseal"
        )
        events.append(phase)
        return ConfirmatoryFilesystemReadback(
            status="passed",
            run_directory=Path(run_directory),
            matrix_plan_sha256="1" * 64,
            cell_index_sha256="2" * 64,
            root_artifact_manifest_sha256="3" * 64,
            confirmatory_storage_policy_sha256=(
                gate.execution_authority.confirmatory_storage_policy_sha256
            ),
            checked_artifact_count=42,
            cells=(),
            reconciliation=_reconciliation(plan),
            root_artifacts=(),
            errors=(),
        )

    def completion_builder(*, plan: Any, **_kwargs: Any) -> dict[str, Any]:
        events.append("completion_builder")
        return {
            "schema_version": 1,
            "matrix_config_sha256": plan.config_sha256,
            "planned_cell_count": len(plan.cells),
            "required_cell_count": plan.required_cell_count,
            "completed_required_cell_count": plan.required_cell_count,
            "failed_required_cell_count": 0,
            "reconciliation_status": "passed",
            "fold_rotation_complete": True,
        }

    integrity_calls = 0

    def integrity_verifier(path: str | Path) -> Any:
        nonlocal integrity_calls
        integrity_calls += 1
        events.append(f"integrity_{integrity_calls}")
        observed = verify_run_integrity(path)
        if postseal_integrity_fault and integrity_calls == 2:
            return replace(
                observed,
                valid=False,
                errors=("injected post-seal verification fault",),
            )
        return observed

    disk_calls = 0

    def disk_usage(_path: Path) -> Any:
        nonlocal disk_calls
        disk_calls += 1
        events.append(f"disk_{disk_calls}")
        capacity_policy = gate.execution_authority.resource_capacity_policy
        return SimpleNamespace(
            free=int(capacity_policy["minimum_free_bytes_before_workspace_build"])
        )

    compute_calls = 0

    def compute_probe() -> dict[str, Any]:
        nonlocal compute_calls
        compute_calls += 1
        events.append(f"compute_{compute_calls}")
        policy = RESOURCE_BOUNDED_CAPACITY_POLICY
        return {
            "total_host_ram_bytes": int(policy["minimum_total_ram_bytes"]),
            "available_host_ram_bytes": int(policy["minimum_total_ram_bytes"]),
            "cuda_available": True,
            "cuda_device_count": 1,
            "selected_cuda_device_index": int(policy["cuda_device_index"]),
            "selected_cuda_device_name": "fixture CUDA device",
            "total_vram_bytes": int(policy["minimum_total_vram_bytes"]),
            "free_vram_bytes": int(policy["minimum_free_vram_bytes"]),
            "cudnn_available": True,
            "amp_available": True,
            "amp_dtype": policy["amp_dtype"],
            "weight_identifier": policy["official_weight_identifier"],
            "weight_path": str(weight_path.resolve()),
            "weight_present": True,
            "weight_sha256": policy["official_weight_sha256"],
            "smoke_attempted": True,
            "smoke_completed": True,
            "smoke_input_shape": list(policy["cuda_smoke_input_shape"]),
            "smoke_forward_finite": True,
            "smoke_backward_finite": True,
            "smoke_peak_allocated_bytes": 1,
            "smoke_error": None,
        }

    workspace_plan = dict(execution_authority.resource_input_workspace_plan)
    workspace_plan_sha256 = str(execution_authority.resource_input_workspace_plan_sha256)
    workspace_key = str(workspace_plan["workspace_key"])
    workspace_artifact_root_sha256 = "b" * 64
    workspace_state: dict[str, ConfirmatoryMemoryWorkspace] = {}
    workspace_array_specs = tuple(
        SimpleNamespace(array_id=f"fixture_array_{index:02d}") for index in range(12)
    )
    workspace_index_specs = tuple(
        SimpleNamespace(outer_fold=fold, role=role)
        for fold in (1, 2, 3)
        for role in ("audit", "reference_validation", "final_reference")
    )

    def workspace_builder(*_args: Any, **_kwargs: Any) -> ConfirmatoryMemoryWorkspace:
        workspace_root = (
            project_root / "artifacts" / "resource_control" / "input_workspaces" / workspace_key
        )
        workspace_root.mkdir(parents=True)
        receipt_base = {
            "schema_version": 1,
            "status": "complete",
            "workspace_key": workspace_key,
            "resource_input_workspace_plan_sha256": workspace_plan_sha256,
            "artifact_root_sha256": workspace_artifact_root_sha256,
            "source_annotations_modified": False,
            "scientific_outcomes_read": False,
        }
        receipt = {
            **receipt_base,
            "receipt_without_self_hash_sha256": canonical_sha256(receipt_base),
        }
        receipt_path = workspace_root / "workspace_receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, allow_nan=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        workspace = ConfirmatoryMemoryWorkspace(
            root=workspace_root,
            workspace_key=workspace_key,
            receipt_sha256=sha256_file(receipt_path),
            artifact_root_sha256=workspace_artifact_root_sha256,
            arrays={},
            index_arrays={},
            resource_input_workspace_plan_sha256=workspace_plan_sha256,
            cleanup_ownership_token="execution-fixture-owner",
        )
        workspace_state["value"] = workspace
        return workspace

    def workspace_verifier(*_args: Any, **_kwargs: Any) -> ConfirmatoryMemoryWorkspace:
        return replace(workspace_state["value"], cleanup_ownership_token=None)

    def workspace_cleaner(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        workspace = workspace_state["value"]
        workspace.close()
        shutil.rmtree(workspace.root)
        return {
            "schema_version": 1,
            "status": "complete",
            "workspace_key": workspace.workspace_key,
            "workspace_receipt_sha256": workspace.receipt_sha256,
            "artifact_root_sha256": workspace.artifact_root_sha256,
            "resource_input_workspace_plan_sha256": (
                workspace.resource_input_workspace_plan_sha256
            ),
            "workspace_removed": True,
            "source_annotations_modified": False,
            "scientific_outcomes_read": False,
        }

    def input_builder(*_args: Any, **kwargs: Any) -> Any:
        workspace = kwargs["memory_workspace"]
        return SimpleNamespace(
            manifest_sha256=primary_gate.manifest_sha256,
            config_sha256=(execution_authority.resource_confirmatory_config_semantic_sha256),
            memory_workspace_receipt_sha256=workspace.receipt_sha256,
            memory_workspace_artifact_root_sha256=workspace.artifact_root_sha256,
            memory_workspace_plan_sha256=(workspace.resource_input_workspace_plan_sha256),
        )

    dependencies = replace(
        ResourceBoundedRunnerDependencies(),
        gate_validator=gate_validator,
        lifecycle_validator=lambda **_kwargs: events.append("lifecycle"),
        input_resolver=lambda **_kwargs: SimpleNamespace(
            crop_cache_path=crop_cache,
            expected_crop_cache_sha256="5" * 64,
            expected_crop_metadata_sha256="6" * 64,
            expected_raw_inventory_sha256="7" * 64,
            frozen_feature_caches=(),
            observed_label_sets=(),
        ),
        workspace_array_spec_builder=lambda *_args, **_kwargs: workspace_array_specs,
        workspace_index_spec_builder=lambda *_args, **_kwargs: workspace_index_specs,
        workspace_builder=workspace_builder,
        workspace_verifier=workspace_verifier,
        workspace_cleaner=workspace_cleaner,
        input_builder=input_builder,
        bridge_builder=lambda *_args, **_kwargs: _Bridge(),
        matrix_executor=matrix_executor,
        statistics_aggregator=statistics_aggregator,
        statistics_verifier=statistics_verifier,
        filesystem_reader=filesystem_reader,
        completion_builder=completion_builder,
        tracker_starter=tracker_starter,
        integrity_verifier=integrity_verifier,
        disk_usage=disk_usage,
        clock=lambda: "2026-07-27T00:00:00+00:00",
        compute_probe=compute_probe,
    )
    kwargs = {
        "run_mode": "fresh",
        "primary_run_directory": primary,
        "project_root": project_root,
        "resource_authority_directory": authority,
        "dataset_path": dataset,
        "manifest_path": manifest,
        "duplicate_audit_path": duplicate_audit,
        "pathology_encoder_audit_path": pathology_audit,
        "lifecycle_readiness_run_directory": readiness,
        "runs_root": run_root,
        "run_id": "resource-execution-fixture",
        "dependencies": dependencies,
        "preflight_only": False,
    }
    return _Harness(
        project_root=project_root,
        run_root=run_root,
        dependencies=dependencies,
        kwargs=kwargs,
        events=events,
        trackers=trackers,
        gate=gate,
    )


def test_full_execution_seals_nonclaiming_run_with_two_ordered_live_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _build_harness(tmp_path, monkeypatch)

    result = _run_resource_bounded_sensitivity(**harness.kwargs)

    tracker = harness.trackers[0]
    run_directory = tracker.run_directory
    assert result["status"] == "completed"
    assert result["completion_stage"] is None
    assert result["study_outcome_eligible"] is False
    assert result["valid_completion_claim"] is False
    assert result["analysis_disposition"] == RESOURCE_BOUNDED_ANALYSIS_DISPOSITION
    assert result["m9_unlock_allowed"] is False
    assert tracker.complete_calls == 1
    assert tracker.fail_calls == 0
    assert tracker.completion_writes == 1
    assert [event for event in harness.events if event.startswith("gate_")] == [
        "gate_1",
        "gate_2",
    ]
    assert harness.events.index("gate_1") < harness.events.index("tracker_start")
    assert harness.events.index("tracker_start") < harness.events.index("matrix")
    assert harness.events.index("matrix") < harness.events.index("statistics")
    assert harness.events.index("statistics") < harness.events.index("completion_builder")
    assert harness.events.index("completion_builder") < harness.events.index("gate_2")
    assert harness.events.index("gate_2") < harness.events.index("complete")
    assert harness.events.index("complete") < harness.events.index("integrity_1")
    assert harness.events.index("integrity_1") < harness.events.index("readback_postseal")
    assert harness.events.index("readback_postseal") < harness.events.index("integrity_2")
    assert [event for event in harness.events if event.startswith("disk_")] == [
        "disk_1",
        "disk_2",
    ]
    assert [event for event in harness.events if event.startswith("compute_")] == [
        "compute_1",
        "compute_2",
    ]

    completion = json.loads(
        (run_directory / "completion_evidence.json").read_text(encoding="utf-8")
    )
    capacity = json.loads(
        (run_directory / "resource_capacity_evidence.json").read_text(encoding="utf-8")
    )
    compute = json.loads(
        (run_directory / "resource_compute_evidence.json").read_text(encoding="utf-8")
    )
    resume = json.loads(
        (run_directory / "resource_resume_evidence.json").read_text(encoding="utf-8")
    )
    capacity_policy = dict(harness.gate.execution_authority.resource_capacity_policy)
    capacity_sha = canonical_sha256(capacity_policy)
    assert capacity["policy_sha256"] == capacity_sha
    assert [item["phase"] for item in capacity["checks"]] == [
        "guarded_before_workspace_build",
        "guarded_immediately_before_tracker",
    ]
    assert completion["resource_capacity_policy_sha256"] == capacity_sha
    assert result["resource_capacity_policy_sha256"] == capacity_sha
    assert (
        completion["resource_compute_evidence_sha256"]
        == compute["evidence_without_self_hash_sha256"]
        == result["resource_compute_evidence_sha256"]
    )
    assert (
        completion["resource_resume_evidence_sha256"]
        == resume[RESOURCE_BOUNDED_RESUME_EVIDENCE_HASH_FIELD]
        == result["resource_resume_evidence_sha256"]
    )
    assert resume["run_mode"] == "fresh"
    assert resume["predecessor_read_performed"] is False
    assert len(resume["fresh_checkpoint_paths"]) == 30
    assert verify_run_integrity(run_directory).valid
    assert _registry_rows(harness.run_root, tracker.run_id)[0]["status"] == "completed"
    _assert_zero_scientific_records(harness.run_root, tracker.run_id)


def test_preseal_failure_demotes_once_and_seals_failed_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fault = RuntimeError("injected matrix failure")
    harness = _build_harness(tmp_path, monkeypatch, matrix_error=fault)

    with pytest.raises(RuntimeError, match="injected matrix failure"):
        _run_resource_bounded_sensitivity(**harness.kwargs)

    assert len(harness.trackers) == 1
    tracker = harness.trackers[0]
    run_directory = tracker.run_directory
    assert tracker.complete_calls == 0
    assert tracker.fail_calls == 1
    assert tracker.completion_writes == 1
    assert harness.events.count("tracker_start") == 1
    assert harness.events.count("matrix") == 1
    assert [event for event in harness.events if event.startswith("gate_")] == ["gate_1"]
    assert "gate_2" not in harness.events
    assert "complete" not in harness.events
    assert (run_directory / ".immutable.json").is_file()
    status = json.loads((run_directory / "status.json").read_text(encoding="utf-8"))
    completion = json.loads(
        (run_directory / "completion_evidence.json").read_text(encoding="utf-8")
    )
    metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert completion["completion_stage"] is None
    assert completion["study_outcome_eligible"] is False
    assert completion["valid_completion_claim"] is False
    assert completion["analysis_disposition"] == RESOURCE_BOUNDED_ANALYSIS_DISPOSITION
    assert completion["retry_of_run_id"] is None
    assert metrics["status"] == "failed"
    assert verify_run_integrity(run_directory).valid
    assert _registry_rows(harness.run_root, tracker.run_id)[0]["status"] == "failed"
    _assert_zero_scientific_records(harness.run_root, tracker.run_id)


def test_postseal_fault_preserves_completed_immutable_run_without_failed_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _build_harness(
        tmp_path,
        monkeypatch,
        postseal_integrity_fault=True,
    )

    with pytest.raises(
        ResourceBoundedStudyIntegrityError,
        match="integrity changed during post-seal readback",
    ):
        _run_resource_bounded_sensitivity(**harness.kwargs)

    assert len(harness.trackers) == 1
    tracker = harness.trackers[0]
    run_directory = tracker.run_directory
    assert tracker.complete_calls == 1
    assert tracker.fail_calls == 0
    assert tracker.completion_writes == 1
    assert harness.events.count("tracker_start") == 1
    assert [event for event in harness.events if event.startswith("gate_")] == [
        "gate_1",
        "gate_2",
    ]
    assert [event for event in harness.events if event.startswith("integrity_")] == [
        "integrity_1",
        "integrity_2",
    ]
    assert "fail" not in harness.events
    assert (run_directory / ".immutable.json").is_file()
    status = json.loads((run_directory / "status.json").read_text(encoding="utf-8"))
    completion = json.loads(
        (run_directory / "completion_evidence.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "completed"
    assert completion["completion_stage"] is None
    assert completion["study_outcome_eligible"] is False
    assert completion["analysis_disposition"] == RESOURCE_BOUNDED_ANALYSIS_DISPOSITION
    assert completion["m9_unlock_allowed"] is False
    assert verify_run_integrity(run_directory).valid
    rows = _registry_rows(harness.run_root, tracker.run_id)
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    _assert_zero_scientific_records(harness.run_root, tracker.run_id)
