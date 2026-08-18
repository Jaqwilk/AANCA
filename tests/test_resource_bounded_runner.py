"""Structural and fail-closed tests for the non-claiming sensitivity runner."""

from __future__ import annotations

import ast
import inspect
import json
import shutil
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_resource_bounded_confirmatory_contract import _resource_gate

import histo_audit.experiment.resource_bounded_runner as runner
from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.experiment.confirmatory_core import (
    confirmatory_execution_controls_from_frozen_config,
)
from histo_audit.experiment.confirmatory_memory_workspace import (
    ConfirmatoryMemoryWorkspace,
)
from histo_audit.experiment.resource_bounded_runner import (
    RESOURCE_BOUNDED_CAPACITY_POLICY,
    ResourceBoundedRunnerDependencies,
    ResourceBoundedStudyRunnerError,
    ResourceComputeEvidence,
    _log_resource_progress,
    _require_gate_equality,
    _require_reusable_successor_checkpoint,
    _resource_compute_evidence_document,
    _validate_run_mode,
    build_resource_checkpoint_allowlist,
    execute_resource_bounded_sensitivity,
    preflight_resource_bounded_sensitivity,
    qualify_resource_checkpoint_predecessor,
    require_resource_capacity,
    require_resource_compute,
)
from histo_audit.utils.run_tracking import (
    _ensure_run_disposition_anchor,
    _ensure_run_stage_attestation_anchor,
    sha256_file,
)

_ROOT = Path(__file__).resolve().parents[1]
_RESOURCE_CONFIG = _ROOT / "configs" / "confirmatory_resource_bounded_amended.yaml"


class _ProgressTracker:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def log_event(self, event: str, **details: Any) -> None:
        self.events.append((event, details))


def _capacity_usage(free: int) -> Any:
    return SimpleNamespace(free=free)


def _passing_compute_observation(weight_path: Path) -> dict[str, Any]:
    policy = RESOURCE_BOUNDED_CAPACITY_POLICY
    return {
        "total_host_ram_bytes": policy["minimum_total_ram_bytes"],
        "available_host_ram_bytes": policy["minimum_available_ram_bytes_before_data"],
        "cuda_available": True,
        "cuda_device_count": 1,
        "selected_cuda_device_index": 0,
        "selected_cuda_device_name": "deterministic-test-cuda",
        "total_vram_bytes": policy["minimum_total_vram_bytes"],
        "free_vram_bytes": policy["minimum_free_vram_bytes"],
        "cudnn_available": True,
        "amp_available": True,
        "amp_dtype": "float16",
        "weight_identifier": policy["official_weight_identifier"],
        "weight_path": str(weight_path.resolve()),
        "weight_present": True,
        "weight_sha256": policy["official_weight_sha256"],
        "smoke_attempted": True,
        "smoke_completed": True,
        "smoke_input_shape": list(policy["cuda_smoke_input_shape"]),
        "smoke_forward_finite": True,
        "smoke_backward_finite": True,
        "smoke_peak_allocated_bytes": policy["cuda_smoke_max_peak_allocated_bytes"],
        "smoke_error": None,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), sort_keys=True),
        encoding="utf-8",
    )


def _predecessor_fixture(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_root = tmp_path / "artifacts" / "runs"
    predecessor = run_root / "failed-resource-run"
    _ensure_run_disposition_anchor(run_root)
    _ensure_run_stage_attestation_anchor(run_root)
    predecessor.mkdir()
    gate = _resource_gate()
    prepared = SimpleNamespace(
        manifest_sha256="a" * 64,
        config_sha256="b" * 64,
    )
    controls = SimpleNamespace(binding_sha256="c" * 64)
    bridge = SimpleNamespace(as_dict=lambda: {"bridge_sha256": "d" * 64})
    crop_cache = tmp_path / "crop-cache.npz"
    gate_payload = {
        **gate.as_dict(),
        "confirmatory_storage_policy_sha256": (
            gate.execution_authority.confirmatory_storage_policy_sha256
        ),
    }
    bindings = {
        "schema_version": 1,
        "artifact_scope": runner.RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
        "analysis_disposition": "amended_or_exploratory",
        "completion_stage": None,
        "study_outcome_eligible": False,
        "original_confirmatory_claim_allowed": False,
        "m9_unlock_allowed": False,
        "crop_cache_path": str(crop_cache),
        "crop_cache_sha256": "1" * 64,
        "crop_metadata_sha256": "2" * 64,
        "raw_inventory_sha256": "3" * 64,
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
        "cnn_fold_data_and_split_sha256": {},
        "feature_caches": [],
    }
    status = {
        "status": "failed",
        "run_id": predecessor.name,
        "experiment_name": runner.RESOURCE_BOUNDED_EXPERIMENT_NAME,
    }
    completion = {
        "completion_stage": None,
        "study_outcome_eligible": False,
        "valid_completion_claim": False,
        "artifact_scope": runner.RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
        "analysis_disposition": "amended_or_exploratory",
        "original_confirmatory_claim_allowed": False,
        "m9_unlock_allowed": False,
    }
    _write_json(predecessor / "status.json", status)
    _write_json(predecessor / "completion_evidence.json", completion)
    _write_json(predecessor / "confirmatory_execution_gate.json", gate_payload)
    _write_json(predecessor / "confirmatory_input_bindings.json", bindings)
    return {
        "predecessor": predecessor,
        "run_root": run_root,
        "retry_of_run_id": predecessor.name,
        "gate": gate,
        "crop_cache": crop_cache,
        "expected_crop_cache_sha256": "1" * 64,
        "expected_crop_metadata_sha256": "2" * 64,
        "expected_raw_inventory_sha256": "3" * 64,
        "resolved_specs": (),
        "prepared": prepared,
        "controls": controls,
        "bridge": bridge,
        "cnn_fingerprints": {},
    }, {
        "status": status,
        "completion": completion,
        "gate": gate_payload,
        "bindings": bindings,
    }


def test_capacity_gate_uses_exact_fixed_threshold_without_creating_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "not-created" / "runs"
    minimum = cast(
        int,
        RESOURCE_BOUNDED_CAPACITY_POLICY["minimum_free_bytes_before_tracker"],
    )

    evidence = require_resource_capacity(
        target,
        capacity_policy=RESOURCE_BOUNDED_CAPACITY_POLICY,
        phase="unit",
        disk_usage=lambda path: (
            _capacity_usage(minimum) if path == tmp_path else _capacity_usage(-1)
        ),
        clock=lambda: "2026-07-27T00:00:00+00:00",
    )

    assert evidence.passed
    assert evidence.free_bytes == minimum
    assert evidence.minimum_free_bytes == 23_622_320_128
    assert evidence.policy_sha256 == canonical_sha256(RESOURCE_BOUNDED_CAPACITY_POLICY)
    assert not target.exists()


def test_capacity_gate_rejects_low_space_and_policy_tamper(tmp_path: Path) -> None:
    minimum = cast(
        int,
        RESOURCE_BOUNDED_CAPACITY_POLICY["minimum_free_bytes_before_tracker"],
    )
    with pytest.raises(ResourceBoundedStudyRunnerError, match="22-GiB"):
        require_resource_capacity(
            tmp_path,
            capacity_policy=RESOURCE_BOUNDED_CAPACITY_POLICY,
            phase="low",
            disk_usage=lambda _: _capacity_usage(minimum - 1),
        )

    called = False

    def forbidden_usage(_: Path) -> Any:
        nonlocal called
        called = True
        return _capacity_usage(minimum)

    tampered = dict(RESOURCE_BOUNDED_CAPACITY_POLICY)
    tampered["minimum_free_bytes_before_tracker"] = minimum - 1
    with pytest.raises(ResourceBoundedStudyRunnerError, match="exact schema-v2"):
        require_resource_capacity(
            tmp_path,
            capacity_policy=tampered,
            phase="tampered",
            disk_usage=forbidden_usage,
        )
    assert called is False


def test_authority_capacity_v2_contains_fixed_compute_policy() -> None:
    policy = RESOURCE_BOUNDED_CAPACITY_POLICY

    assert policy["schema_version"] == 2
    assert policy["policy"] == "resource_bounded_confirmatory_capacity_v2"
    assert policy["minimum_total_ram_bytes"] == 30 * 1024**3
    assert policy["minimum_available_ram_bytes_before_data"] == 16 * 1024**3
    assert policy["minimum_available_ram_bytes_before_tracker"] == 12 * 1024**3
    assert policy["minimum_total_vram_bytes"] == 10 * 1024**3
    assert policy["minimum_free_vram_bytes"] == 8 * 1024**3
    assert policy["cuda_smoke_max_peak_allocated_bytes"] == 512 * 1024**2


def test_compute_gate_accepts_exact_boundaries_for_both_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = tmp_path / "resnet18-f37072fd.pth"
    weight.write_bytes(b"fixture")
    expected_weight_sha = cast(
        str,
        RESOURCE_BOUNDED_CAPACITY_POLICY["official_weight_sha256"],
    )
    monkeypatch.setattr(runner, "sha256_file", lambda path: expected_weight_sha)
    observation = _passing_compute_observation(weight)

    before_data = require_resource_compute(
        phase="guarded_before_data_loading",
        capacity_policy=RESOURCE_BOUNDED_CAPACITY_POLICY,
        probe=lambda: observation,
        clock=lambda: "2026-07-27T00:00:00+00:00",
    )
    before_tracker_observation = {
        **observation,
        "available_host_ram_bytes": RESOURCE_BOUNDED_CAPACITY_POLICY[
            "minimum_available_ram_bytes_before_tracker"
        ],
    }
    before_tracker = require_resource_compute(
        phase="guarded_immediately_before_tracker",
        capacity_policy=RESOURCE_BOUNDED_CAPACITY_POLICY,
        probe=lambda: before_tracker_observation,
        clock=lambda: "2026-07-27T00:00:01+00:00",
    )

    assert isinstance(before_data, ResourceComputeEvidence)
    assert before_data.passed is True
    assert before_data.minimum_available_ram_bytes == 16 * 1024**3
    assert before_tracker.passed is True
    assert before_tracker.minimum_available_ram_bytes == 12 * 1024**3
    document = _resource_compute_evidence_document(
        capacity_policy=RESOURCE_BOUNDED_CAPACITY_POLICY,
        checks=(before_data, before_tracker),
    )
    assert document["check_sha256s"] == [
        before_data.evidence_sha256,
        before_tracker.evidence_sha256,
    ]


def test_compute_gate_rejects_policy_tamper_before_probe() -> None:
    called = False

    def forbidden_probe() -> Mapping[str, Any]:
        nonlocal called
        called = True
        return {}

    tampered = dict(RESOURCE_BOUNDED_CAPACITY_POLICY)
    tampered["minimum_free_vram_bytes"] = cast(int, tampered["minimum_free_vram_bytes"]) - 1
    with pytest.raises(ResourceBoundedStudyRunnerError, match="schema-v2"):
        require_resource_compute(
            phase="guarded_before_data_loading",
            capacity_policy=tampered,
            probe=forbidden_probe,
        )

    assert called is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_host_ram_bytes", 30.0),
        ("cuda_available", 1),
        ("cudnn_available", False),
        ("amp_available", False),
        ("smoke_forward_finite", False),
        ("smoke_backward_finite", False),
        ("smoke_peak_allocated_bytes", 512 * 1024**2 + 1),
    ],
)
def test_compute_gate_rejects_malformed_or_insufficient_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    weight = tmp_path / "weight.pth"
    weight.write_bytes(b"fixture")
    monkeypatch.setattr(
        runner,
        "sha256_file",
        lambda path: RESOURCE_BOUNDED_CAPACITY_POLICY["official_weight_sha256"],
    )
    observation = {**_passing_compute_observation(weight), field: value}

    with pytest.raises(ResourceBoundedStudyRunnerError, match="compute"):
        require_resource_compute(
            phase="guarded_before_data_loading",
            capacity_policy=RESOURCE_BOUNDED_CAPACITY_POLICY,
            probe=lambda: observation,
        )


def test_compute_evidence_carrier_tamper_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = tmp_path / "weight.pth"
    weight.write_bytes(b"fixture")
    monkeypatch.setattr(
        runner,
        "sha256_file",
        lambda path: RESOURCE_BOUNDED_CAPACITY_POLICY["official_weight_sha256"],
    )
    first = require_resource_compute(
        phase="guarded_before_data_loading",
        capacity_policy=RESOURCE_BOUNDED_CAPACITY_POLICY,
        probe=lambda: _passing_compute_observation(weight),
    )
    second_observation = {
        **_passing_compute_observation(weight),
        "available_host_ram_bytes": RESOURCE_BOUNDED_CAPACITY_POLICY[
            "minimum_available_ram_bytes_before_tracker"
        ],
    }
    second = require_resource_compute(
        phase="guarded_immediately_before_tracker",
        capacity_policy=RESOURCE_BOUNDED_CAPACITY_POLICY,
        probe=lambda: second_observation,
    )
    tampered = replace(
        first,
        observation={
            **dict(first.observation),
            "available_host_ram_bytes": 0,
        },
    )

    with pytest.raises(ResourceBoundedStudyRunnerError, match="carrier"):
        _resource_compute_evidence_document(
            capacity_policy=RESOURCE_BOUNDED_CAPACITY_POLICY,
            checks=(tampered, second),
        )


@pytest.mark.parametrize(
    ("mode", "predecessor", "retry", "error"),
    [
        ("automatic", None, None, "explicitly"),
        ("fresh", Path("predecessor"), "predecessor", "forbids"),
        ("successor_resume", None, None, "requires"),
        ("successor_resume", Path("wrong"), "right", "exactly equal"),
    ],
)
def test_run_mode_has_no_automatic_discovery_or_retry(
    mode: str,
    predecessor: Path | None,
    retry: str | None,
    error: str,
) -> None:
    with pytest.raises(ResourceBoundedStudyRunnerError, match=error):
        _validate_run_mode(
            run_mode=mode,
            checkpoint_predecessor_run_directory=predecessor,
            retry_of_run_id=retry,
            run_id=None,
        )


@pytest.mark.parametrize(
    "invalid_run_root",
    [
        Path("artifacts/runs/primary"),
        Path("authority-C"),
        Path("artifacts/runs/predecessor"),
        Path("artifacts/runs/nested/child"),
    ],
)
def test_noncanonical_runs_root_rejected_before_lifecycle_or_write(
    tmp_path: Path,
    invalid_run_root: Path,
) -> None:
    lifecycle_calls = 0

    def forbidden_lifecycle(**_: Any) -> Any:
        nonlocal lifecycle_calls
        lifecycle_calls += 1
        raise AssertionError("lifecycle must not run for a noncanonical runs root")

    dependencies = replace(
        ResourceBoundedRunnerDependencies(),
        lifecycle_validator=forbidden_lifecycle,
    )
    with pytest.raises(ResourceBoundedStudyRunnerError, match="canonical"):
        runner._run_resource_bounded_sensitivity(
            run_mode="fresh",
            primary_run_directory=tmp_path / "primary",
            project_root=tmp_path,
            resource_authority_directory=tmp_path / "authority-C",
            dataset_path=tmp_path / "dataset",
            manifest_path=tmp_path / "manifest.parquet",
            duplicate_audit_path=tmp_path / "duplicates.json",
            pathology_encoder_audit_path=tmp_path / "pathology.json",
            lifecycle_readiness_run_directory=tmp_path / "readiness",
            runs_root=tmp_path / invalid_run_root,
            dependencies=dependencies,
            preflight_only=True,
        )

    assert lifecycle_calls == 0
    assert not any(tmp_path.iterdir())


def _qualify_fixture_predecessor(
    fixture: Mapping[str, Any],
    *,
    registry_record_present: bool = True,
) -> dict[str, Any]:
    return qualify_resource_checkpoint_predecessor(
        cast(Path, fixture["predecessor"]),
        run_root=cast(Path, fixture["run_root"]),
        retry_of_run_id=cast(str, fixture["retry_of_run_id"]),
        gate=fixture["gate"],
        crop_cache=cast(Path, fixture["crop_cache"]),
        expected_crop_cache_sha256=cast(
            str,
            fixture["expected_crop_cache_sha256"],
        ),
        expected_crop_metadata_sha256=cast(
            str,
            fixture["expected_crop_metadata_sha256"],
        ),
        expected_raw_inventory_sha256=cast(
            str,
            fixture["expected_raw_inventory_sha256"],
        ),
        resolved_specs=fixture["resolved_specs"],
        prepared=fixture["prepared"],
        controls=fixture["controls"],
        bridge=fixture["bridge"],
        cnn_fingerprints=fixture["cnn_fingerprints"],
        integrity_verifier=lambda _: SimpleNamespace(
            valid=True,
            registry_record_present=registry_record_present,
            run_id=fixture["retry_of_run_id"],
            expected_root_sha256="e" * 64,
            errors=(),
        ),
    )


def test_failed_sealed_resource_predecessor_qualification_passes(
    tmp_path: Path,
) -> None:
    fixture, _ = _predecessor_fixture(tmp_path)

    evidence = _qualify_fixture_predecessor(fixture)

    assert evidence["qualified"] is True
    assert evidence["terminal_status"] == "failed"
    assert evidence["oof_artifacts_read"] is False
    assert evidence["metrics_artifacts_read"] is False
    assert evidence["ranking_artifacts_read"] is False
    assert len(evidence["qualification_without_self_hash_sha256"]) == 64


@pytest.mark.parametrize(
    ("artifact", "field", "value", "error"),
    [
        ("status", "status", "running", "terminal failed"),
        ("status", "experiment_name", "wrong-experiment", "terminal failed"),
        ("completion", "study_outcome_eligible", True, "scientific claim"),
        ("completion", "completion_stage", "CONFIRMATORY_COMPLETE", "scientific claim"),
        ("gate", "completion_stage", "CONFIRMATORY_COMPLETE", "different live P\\+C"),
    ],
)
def test_predecessor_qualification_rejects_unsealed_wrong_authority_or_claim(
    tmp_path: Path,
    artifact: str,
    field: str,
    value: Any,
    error: str,
) -> None:
    fixture, payloads = _predecessor_fixture(tmp_path)
    payload = {**payloads[artifact], field: value}
    filename = {
        "status": "status.json",
        "completion": "completion_evidence.json",
        "gate": "confirmatory_execution_gate.json",
    }[artifact]
    _write_json(cast(Path, fixture["predecessor"]) / filename, payload)

    with pytest.raises(ResourceBoundedStudyRunnerError, match=error):
        _qualify_fixture_predecessor(fixture)


def test_predecessor_qualification_rejects_missing_registry_seal_and_lock(
    tmp_path: Path,
) -> None:
    fixture, _ = _predecessor_fixture(tmp_path)
    with pytest.raises(ResourceBoundedStudyRunnerError, match="registry-backed"):
        _qualify_fixture_predecessor(
            fixture,
            registry_record_present=False,
        )

    lock = cast(Path, fixture["run_root"]) / (f".{fixture['retry_of_run_id']}.mutation.lock")
    lock.write_text("locked", encoding="utf-8")
    with pytest.raises(ResourceBoundedStudyRunnerError, match="mutation lock"):
        _qualify_fixture_predecessor(fixture)


def test_successor_resume_rejects_zero_validated_reusable_checkpoints() -> None:
    with pytest.raises(ResourceBoundedStudyRunnerError, match="zero reuse"):
        _require_reusable_successor_checkpoint(
            cast(
                Any,
                SimpleNamespace(
                    resume_records=(),
                    fresh_records=tuple(range(30)),
                ),
            )
        )


def test_operational_progress_has_no_duration_and_forbids_adaptation() -> None:
    tracker = _ProgressTracker()
    event = {
        "cell_id": "cell",
        "scenario": "scenario",
        "fold": 1,
        "corruption": "clean",
        "seed": 303,
        "status": "started",
        "timestamp": "2026-07-27T00:00:00+00:00",
        "telemetry_contract": "outcome_value_free_operational_telemetry",
        "prohibited_for_selection_tuning": True,
        "adaptive_execution_changes_allowed": False,
    }

    _log_resource_progress(cast(Any, tracker), event)

    assert tracker.events == [("resource_bounded_cell_progress", event)]
    with pytest.raises(ResourceBoundedStudyRunnerError, match="non-contract"):
        _log_resource_progress(
            cast(Any, tracker),
            {**event, "duration_seconds": 1.0},
        )


def test_checkpoint_allowlist_is_exactly_six_cells_by_five_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.load_config(_RESOURCE_CONFIG)
    controls = confirmatory_execution_controls_from_frozen_config(config)
    cnn_cells = [
        cell
        for cell in controls.plan.cells
        if controls.scenarios_by_id[cell.scenario_id].family == "cnn"
    ]
    fingerprints = {
        cell.cell_id: {
            str(fold): {
                "training_data_sha256": "1" * 64,
                "reference_validation_data_sha256": "2" * 64,
                "training_split_sha256": "3" * 64,
                "reference_validation_split_sha256": "4" * 64,
            }
            for fold in range(5)
        }
        for cell in cnn_cells
    }
    monkeypatch.setattr(
        runner,
        "_resource_model_metadata",
        lambda _: {"model": "exact-test-double"},
    )

    allowlist = build_resource_checkpoint_allowlist(
        controls=controls,
        cnn_preflight_fingerprints=fingerprints,
    )

    assert len(cnn_cells) == 6
    assert len(allowlist) == 30
    assert len({item.relative_path for item in allowlist}) == 30
    assert all(
        item.relative_path == f"cells/{item.cell_id}/checkpoints/fold_{item.fold_id:02d}.pt"
        for item in allowlist
    )


def test_preseal_gate_equality_binds_storage_hash_without_extra_c_path_read() -> None:
    gate = _resource_gate()
    changed = replace(
        gate,
        execution_authority=replace(
            gate.execution_authority,
            confirmatory_storage_policy_sha256="f" * 64,
        ),
    )

    with pytest.raises(ResourceBoundedStudyRunnerError, match="changed"):
        _require_gate_equality(gate, changed)

    source = inspect.getsource(runner._run_resource_bounded_sensitivity)
    assert "_require_sealed_resource_storage_policy_hash" not in source
    assert "amendment_evidence.json" not in source


def test_public_preflight_and_execute_have_identical_parameters() -> None:
    preflight = inspect.signature(preflight_resource_bounded_sensitivity)
    execute = inspect.signature(execute_resource_bounded_sensitivity)
    expected = (
        "run_mode",
        "primary_run_directory",
        "project_root",
        "resource_authority_directory",
        "dataset_path",
        "manifest_path",
        "duplicate_audit_path",
        "pathology_encoder_audit_path",
        "lifecycle_readiness_run_directory",
        "checkpoint_predecessor_run_directory",
        "retry_of_run_id",
        "runs_root",
        "run_id",
    )

    assert tuple(preflight.parameters) == expected
    assert tuple(execute.parameters) == expected
    assert all(
        preflight.parameters[name].kind == execute.parameters[name].kind
        for name in preflight.parameters
    )


@pytest.mark.parametrize("second_compute_passes", [True, False])
def test_public_preflight_completes_before_tracker_creation_or_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    second_compute_passes: bool,
) -> None:
    authority = tmp_path / "C"
    authority.mkdir()
    config_path = authority / "confirmatory_frozen.yaml"
    shutil.copyfile(_RESOURCE_CONFIG, config_path)
    primary = (tmp_path / "runs" / "primary").resolve()
    gate = _resource_gate()
    gate = replace(
        gate,
        historical_primary=replace(
            gate.historical_primary,
            primary_run_directory=primary,
        ),
        execution_authority=replace(
            gate.execution_authority,
            authority_directory=authority.resolve(),
            resource_confirmatory_config_file_sha256=sha256_file(config_path),
        ),
    )
    tracker_calls = 0
    gate_calls = 0
    compute_calls = 0
    weight = tmp_path / "official-weight.pth"
    weight.write_bytes(b"fixture")
    real_sha256_file = runner.sha256_file

    def conditional_sha256(path: str | Path) -> str:
        if Path(path).resolve() == weight.resolve():
            return cast(
                str,
                RESOURCE_BOUNDED_CAPACITY_POLICY["official_weight_sha256"],
            )
        return real_sha256_file(path)

    def counted_compute_probe() -> Mapping[str, Any]:
        nonlocal compute_calls
        compute_calls += 1
        observation = _passing_compute_observation(weight)
        if compute_calls == 2 and not second_compute_passes:
            observation["available_host_ram_bytes"] = (
                cast(
                    int,
                    RESOURCE_BOUNDED_CAPACITY_POLICY["minimum_available_ram_bytes_before_tracker"],
                )
                - 1
            )
        return observation

    def forbidden_tracker(**_: Any) -> Any:
        nonlocal tracker_calls
        tracker_calls += 1
        raise AssertionError("preflight created a tracker")

    def counted_gate(**_: Any) -> Any:
        nonlocal gate_calls
        gate_calls += 1
        return gate

    @contextmanager
    def fake_primary_guard(_: Path) -> Any:
        yield SimpleNamespace(valid=True)

    workspace_plan = dict(gate.execution_authority.resource_input_workspace_plan)
    workspace_plan_sha256 = str(gate.execution_authority.resource_input_workspace_plan_sha256)
    workspace_key = str(workspace_plan["workspace_key"])
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
        workspace_root = tmp_path / "artifacts" / "resource_control" / workspace_key
        workspace_root.mkdir(parents=True)
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "resource_input_workspace_plan_sha256": workspace_plan_sha256,
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
            artifact_root_sha256="a" * 64,
            arrays={},
            index_arrays={},
            resource_input_workspace_plan_sha256=workspace_plan_sha256,
            cleanup_ownership_token="preflight-fixture-owner",
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
            memory_workspace_receipt_sha256=workspace.receipt_sha256,
            memory_workspace_artifact_root_sha256=workspace.artifact_root_sha256,
            memory_workspace_plan_sha256=(workspace.resource_input_workspace_plan_sha256),
        )

    dependencies = replace(
        ResourceBoundedRunnerDependencies(),
        gate_validator=counted_gate,
        lifecycle_validator=lambda **_: SimpleNamespace(valid=True),
        input_resolver=cast(
            Any,
            lambda **_: SimpleNamespace(
                crop_cache_path=tmp_path / "crops.npz",
                expected_crop_cache_sha256="1" * 64,
                expected_crop_metadata_sha256="2" * 64,
                expected_raw_inventory_sha256="3" * 64,
                frozen_feature_caches=(),
                observed_label_sets=(),
            ),
        ),
        workspace_array_spec_builder=lambda *_args, **_kwargs: workspace_array_specs,
        workspace_index_spec_builder=lambda *_args, **_kwargs: workspace_index_specs,
        workspace_builder=workspace_builder,
        workspace_verifier=workspace_verifier,
        workspace_cleaner=workspace_cleaner,
        input_builder=input_builder,
        bridge_builder=cast(
            Any,
            lambda *_args, **_kwargs: SimpleNamespace(),
        ),
        tracker_starter=forbidden_tracker,
        compute_probe=counted_compute_probe,
        disk_usage=lambda _: _capacity_usage(
            cast(
                int,
                cast(
                    Mapping[str, Any],
                    gate.execution_authority.resource_capacity_policy,
                )["minimum_free_bytes_before_workspace_build"],
            )
        ),
        clock=lambda: "2026-07-27T00:00:00+00:00",
    )
    monkeypatch.setattr(runner, "_verify_cache_files", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "sha256_file", conditional_sha256)
    monkeypatch.setattr(runner, "guard_run_stage_eligibility", fake_primary_guard)
    monkeypatch.setattr(
        runner,
        "_confirmatory_cnn_preflight_fingerprints",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        runner,
        "build_resource_checkpoint_allowlist",
        lambda **_: tuple(range(30)),
    )

    arguments = {
        "run_mode": "fresh",
        "primary_run_directory": primary,
        "project_root": tmp_path,
        "resource_authority_directory": authority,
        "dataset_path": tmp_path / "dataset",
        "manifest_path": tmp_path / "manifest.parquet",
        "duplicate_audit_path": tmp_path / "duplicates.json",
        "pathology_encoder_audit_path": tmp_path / "pathology.json",
        "lifecycle_readiness_run_directory": tmp_path / "readiness",
        "runs_root": tmp_path / "artifacts" / "runs",
        "dependencies": dependencies,
        "preflight_only": True,
    }
    if second_compute_passes:
        result = runner._run_resource_bounded_sensitivity(**cast(Any, arguments))
    else:
        with pytest.raises(ResourceBoundedStudyRunnerError, match="compute preflight"):
            runner._run_resource_bounded_sensitivity(**cast(Any, arguments))

    assert tracker_calls == 0
    assert gate_calls == 1
    assert compute_calls == 2
    assert not (tmp_path / "artifacts" / "runs").exists()
    if not second_compute_passes:
        return
    assert result["tracker_created"] is False
    assert result["scientific_run_created"] is False
    assert result["completion_stage"] is None
    assert result["study_outcome_eligible"] is False
    assert result["analysis_disposition"] == "amended_or_exploratory"
    assert result["checkpoint_allowlist_count"] == 30
    assert result["reusable_checkpoint_count"] == 0
    assert result["missing_checkpoint_count"] == 30
    assert result["full_pc_gate_validation_count"] == 1
    assert len(result["capacity_checks"]) == 2
    assert len(result["compute_checks"]) == 2
    assert len(result["resource_compute_evidence_sha256"]) == 64


def test_runner_source_has_no_positive_attestation_or_withdrawal_path() -> None:
    tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert "attest_run_stage_eligibility" not in names
    assert "withdraw_run_eligibility" not in names
