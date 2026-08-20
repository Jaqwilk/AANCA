"""Fast, deterministic coverage for interrupted-primary recovery primitives."""

from __future__ import annotations

import copy
import csv
import inspect
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import histo_audit.experiment.primary_completion as completion_module
import histo_audit.experiment.primary_core as primary_core_module
import histo_audit.experiment.primary_recovery as recovery_module
import histo_audit.experiment.primary_statistics as primary_statistics_module
from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.experiment.primary_completion import (
    PrimaryRestorationReadbackEvidence,
    read_primary_filesystem_evidence,
)
from histo_audit.experiment.primary_core import (
    PrimaryExecutionControls,
    _synthetic_fixture_controls,
)
from histo_audit.experiment.primary_recovery import (
    RECOVERY_COPY_POLICY,
    RECOVERY_POLICY,
    RECOVERY_REGISTRATION_STATUS,
    OrphanSourceSnapshot,
    PrimaryRecoveryError,
    RecoveryAuthorization,
    collect_orphan_source_snapshot,
    copy_authorized_orphan_artifacts,
    inspect_orphan_source,
    verify_recovery_destination,
)
from histo_audit.experiment.study_contracts import (
    PrimaryCell,
    PrimaryMatrixPlan,
    PrimaryScenario,
)
from histo_audit.utils.run_tracking import sha256_file

_CELL_ARTIFACTS = (
    "bootstrap_evidence.npz",
    "cleanlab_evidence.json",
    "cleanlab_evidence.npz",
    "corruption_manifest.json",
    "independence_evidence.json",
    "metrics.json",
    "neighbour_evidence.npz",
    "oof_predictions.npz",
    "oof_provenance.json",
    "ranking.csv",
    "risk_scores.npz",
)
_SOURCE_ROOT_FILES = {
    "cell_index.csv",
    "execution_controls.json",
    "matrix_plan.json",
    "primary_input_bindings.json",
    "reconciliation.json",
    "restoration_index.json",
}
_STATISTICS_FILES = {
    "primary_bootstrap_evidence.npz",
    "primary_statistics.json",
    "primary_statistics_manifest.json",
    "primary_subgroups.csv",
}


@dataclass(frozen=True, slots=True)
class _RecoveryFixture:
    runs_root: Path
    source: Path
    destination: Path
    receipt: Path
    plan: PrimaryMatrixPlan
    controls: PrimaryExecutionControls
    snapshot: OrphanSourceSnapshot
    authorization_mapping: dict[str, Any]
    authorization: RecoveryAuthorization


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def _one_cell_plan() -> PrimaryMatrixPlan:
    scenario = PrimaryScenario(
        scenario_id="tiny_symmetric",
        mechanism="symmetric_random_corruption",
        rate=0.1,
        corruption_seed=404,
    )
    cell = PrimaryCell(
        cell_id="tiny_required_cell",
        scenario_id=scenario.scenario_id,
        mechanism=scenario.mechanism,
        rate=scenario.rate,
        corruption_seed=scenario.corruption_seed,
        representation_id="tiny_representation",
        classifier_id="multinomial_logistic_regression",
        required=True,
    )
    return PrimaryMatrixPlan(
        schema_version=1,
        config_sha256="f" * 64,
        scenarios=(scenario,),
        cells=(cell,),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(field for field in row if field not in fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_cell_tree(
    root: Path,
    plan: PrimaryMatrixPlan,
    controls: PrimaryExecutionControls,
) -> None:
    cells = root / "cells"
    scenarios = root / "corruption_scenarios"
    cells.mkdir(parents=True)
    scenarios.mkdir()
    _write_json(root / "matrix_plan.json", plan.as_dict())
    _write_json(
        root / "execution_controls.json",
        {"schema_version": 1, "binding_sha256": controls.binding_sha256},
    )

    scenario = plan.scenarios[0]
    cell = plan.cells[0]
    identity = asdict(cell)
    scenario_payload = asdict(scenario)
    shared_hash = "a" * 64
    independence_hash = "b" * 64
    matrix_hash = "c" * 64
    cell_directory = cells / cell.cell_id
    cell_directory.mkdir()
    _write_json(
        cell_directory / "metrics.json",
        {
            "cell": identity,
            "scenario": scenario_payload,
            "corruption_configuration_hash": shared_hash,
            "independence_status": "verified_independent",
            "independence_evidence_sha256": independence_hash,
            "independence_matrix_artifact_sha256": matrix_hash,
            "circularity_risk": False,
            "primary_confirmatory_eligible": True,
            "paired_group_bootstrap": {
                "tiny_comparison": {"claim_status": "primary_confirmatory_eligible"}
            },
        },
    )
    _write_json(
        cell_directory / "corruption_manifest.json",
        {
            "schema_version": 1,
            "cell": identity,
            "scenario": scenario_payload,
            "configuration_hash": shared_hash,
            "shared_scenario_corruption_hash": shared_hash,
            "independence_status": "verified_independent",
            "circularity_risk": False,
        },
    )
    _write_json(
        cell_directory / "independence_evidence.json",
        {
            "schema_version": 1,
            "mechanism": cell.mechanism,
            "representation_id": cell.representation_id,
            "status": "verified_independent",
            "circularity_risk": False,
            "primary_confirmatory_eligible": True,
            "matrix_artifact_sha256": matrix_hash,
            "evidence_sha256": independence_hash,
        },
    )
    _write_json(
        cell_directory / "cleanlab_evidence.json",
        {
            "schema_version": 1,
            "available": True,
            "package_version": "fixture",
            "api_path": "fixture",
            "error": None,
            "blocker": None,
            "failure_policy": "missing_with_recorded_blocker",
        },
    )
    for name in _CELL_ARTIFACTS:
        path = cell_directory / name
        if not path.exists():
            path.write_bytes(f"tiny:{cell.cell_id}:{name}".encode())
    records = [
        {
            "path": name,
            "size_bytes": (cell_directory / name).stat().st_size,
            "sha256": sha256_file(cell_directory / name),
        }
        for name in _CELL_ARTIFACTS
    ]
    manifest = _write_json(
        cell_directory / "artifact_manifest.json",
        {"schema_version": 1, "artifacts": records},
    )
    _write_csv(
        root / "cell_index.csv",
        [
            {
                **identity,
                "status": "completed",
                "artifact_manifest_sha256": sha256_file(manifest),
                "metrics_sha256": sha256_file(cell_directory / "metrics.json"),
                "corruption_configuration_hash": shared_hash,
                "execution_controls_binding_sha256": controls.binding_sha256,
                "independence_status": "verified_independent",
                "independence_evidence_sha256": independence_hash,
                "independence_matrix_artifact_sha256": matrix_hash,
                "circularity_risk": False,
                "primary_confirmatory_eligible": True,
            }
        ],
    )
    _write_json(
        scenarios / f"{scenario.scenario_id}.json",
        {
            "scenario": scenario_payload,
            "shared_scenario_corruption_hash": shared_hash,
        },
    )


def _write_restoration_tree(root: Path, cell_id: str) -> None:
    directory = root / "restorations" / cell_id
    _write_json(directory / "restoration.json", {"cell_id": cell_id})
    (directory / "restoration_evidence.npz").write_bytes(b"tiny-restoration")
    _write_json(directory / "restoration_manifest.json", {"cell_id": cell_id})
    _write_json(
        root / "restoration_index.json",
        {
            "schema_version": 1,
            "restoration_cell_ids": [cell_id],
            "restoration_cell_count": 1,
        },
    )


def _restoration_reader(
    run_directory: str | Path,
    controls: PrimaryExecutionControls,
) -> PrimaryRestorationReadbackEvidence:
    run = Path(run_directory).resolve()
    matrix = read_primary_filesystem_evidence(controls.plan, run)
    cell_id = controls.restoration_cell_ids[0]
    directory = run / "restorations" / cell_id
    json_hash = sha256_file(directory / "restoration.json")
    evidence_hash = sha256_file(directory / "restoration_evidence.npz")
    manifest_hash = sha256_file(directory / "restoration_manifest.json")
    index_hash = sha256_file(run / "restoration_index.json")
    evidence = PrimaryRestorationReadbackEvidence(
        run_directory=run,
        status="passed",
        restoration_index_sha256=index_hash,
        readback_root_sha256=canonical_sha256(
            {
                "index": index_hash,
                "json": {cell_id: json_hash},
                "evidence": {cell_id: evidence_hash},
                "manifest": {cell_id: manifest_hash},
            }
        ),
        source_readback_root_sha256=matrix.readback_root_sha256,
        restoration_cell_count=1,
        downstream_comparison_count=len(controls.restoration_downstream_comparisons),
        cell_json_sha256=((cell_id, json_hash),),
        cell_evidence_sha256=((cell_id, evidence_hash),),
        cell_manifest_sha256=((cell_id, manifest_hash),),
    )
    object.__setattr__(
        evidence,
        "_attestation",
        completion_module._RESTORATION_READBACK_ATTESTATION,
    )
    return evidence


def _write_statistics(
    root: Path,
    controls: PrimaryExecutionControls,
    *,
    source_readback_root_sha256: str,
) -> None:
    comparison_ids = [
        *(value.comparison_id for value in controls.within_cell_comparisons),
        *(value.comparison_id for value in controls.method_vs_random_comparisons),
        *(value.comparison_id for value in controls.cross_cell_comparisons),
    ]
    statistics = {
        "schema_version": 1,
        "analysis_scope": "real_pannuke_primary_statistics",
        "execution_controls_binding_sha256": controls.binding_sha256,
        "matrix_plan_sha256": controls.plan_sha256,
        "source_filesystem_readback_root_sha256": source_readback_root_sha256,
        "comparisons": [{"comparison_id": value} for value in comparison_ids],
        "bootstrap": {
            "requested_iterations": controls.bootstrap_iterations,
            "saved_draw_count": controls.bootstrap_iterations,
            "seed": controls.bootstrap_seed,
        },
    }
    _write_json(root / "primary_statistics.json", statistics)
    arrays: dict[str, np.ndarray[Any, Any]] = {
        "comparison_ids": np.asarray(comparison_ids, dtype=np.str_),
        "comparison_kinds": np.asarray(["fixture"] * len(comparison_ids), dtype=np.str_),
        "draw_indices": np.arange(controls.bootstrap_iterations, dtype=np.int64),
        "draw_offsets": np.arange(controls.bootstrap_iterations + 1, dtype=np.int64),
        "group_ids": np.asarray(["tiny-group"], dtype=np.str_),
        "random_review_seeds": np.arange(controls.random_review_repeats, dtype=np.int64),
        "sample_ids": np.asarray(["tiny-sample"], dtype=np.str_),
        "tissue_types": np.asarray(["tiny-tissue"], dtype=np.str_),
    }
    for index, _ in enumerate(comparison_ids):
        prefix = f"comparison_{index:03d}"
        arrays[f"{prefix}_valid_draw_indices"] = np.asarray([], dtype=np.int64)
        arrays[f"{prefix}_metric_a"] = np.asarray([], dtype=np.float64)
        arrays[f"{prefix}_metric_b"] = np.asarray([], dtype=np.float64)
        arrays[f"{prefix}_differences"] = np.asarray([], dtype=np.float64)
    np.savez_compressed(root / "primary_bootstrap_evidence.npz", **arrays)
    (root / "primary_subgroups.csv").write_text(
        "cell_id,status\ntiny,reported\n",
        encoding="utf-8",
    )
    output_names = (
        "primary_bootstrap_evidence.npz",
        "primary_statistics.json",
        "primary_subgroups.csv",
    )
    _write_json(
        root / "primary_statistics_manifest.json",
        {
            "schema_version": 1,
            "execution_controls_binding_sha256": controls.binding_sha256,
            "source_filesystem_readback_root_sha256": source_readback_root_sha256,
            "statistics_payload_sha256": canonical_sha256(statistics),
            "artifacts": [
                {
                    "path": name,
                    "size_bytes": (root / name).stat().st_size,
                    "sha256": sha256_file(root / name),
                }
                for name in output_names
            ],
        },
    )


def _authorization_mapping(
    *,
    source: Path,
    receipt: Path,
    snapshot: OrphanSourceSnapshot,
) -> dict[str, Any]:
    source_tree = json.loads((source / "source_tree_manifest.json").read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "policy": RECOVERY_POLICY,
        "source_run_id": source.name,
        "source_run_directory": str(source.resolve()),
        "interruption_evidence": {
            "kind": "host_reboot",
            "observed_at_utc": "2026-07-27T12:40:00Z",
            "last_boot_at_utc": "2026-07-27T12:37:04Z",
            "event_id": 12,
            "source_process_id": 20792,
            "process_checked_at_utc": "2026-07-27T12:41:00Z",
            "process_active": False,
            "receipt_path": str(receipt.resolve()),
            "receipt_sha256": sha256_file(receipt),
        },
        "outcomes_inspected": True,
        "outcome_inspection_at_utc": "2026-07-27T12:42:00Z",
        "analysis_disposition": RECOVERY_REGISTRATION_STATUS,
        "scientific_method_changes": [],
        "expected_status_sha256": sha256_file(source / "status.json"),
        "expected_primary_execution_gate_sha256": sha256_file(
            source / "primary_execution_gate.json"
        ),
        "expected_source_tree_manifest_sha256": sha256_file(source / "source_tree_manifest.json"),
        "expected_source_tree_root_sha256": source_tree["root_sha256"],
        "expected_source_snapshot_root_sha256": snapshot.snapshot_root_sha256,
        "expected_source_filesystem_readback_root_sha256": (
            snapshot.filesystem_readback.readback_root_sha256
        ),
        "expected_restoration_readback_root_sha256": (
            snapshot.restoration_readback.readback_root_sha256
        ),
        "expected_statistics_manifest_sha256": snapshot.statistics_manifest_sha256,
        "trust_assumption": "local files were not concurrently modified during recovery",
        "limitation": "lightweight closure does not recompute numerical statistics",
        "reason": "the original primary process was interrupted by a documented host reboot",
    }


def _make_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _RecoveryFixture:
    monkeypatch.setattr(
        recovery_module,
        "read_primary_restoration_evidence",
        _restoration_reader,
    )
    runs_root = tmp_path / "runs"
    source = runs_root / "interrupted-primary"
    source.mkdir(parents=True)
    destination = runs_root / "recovered-primary"
    receipt = _write_json(tmp_path / "reboot_receipt.json", {"event_id": 12})
    plan = _one_cell_plan()
    controls = _synthetic_fixture_controls(plan)
    _write_cell_tree(source, plan, controls)
    readback = read_primary_filesystem_evidence(plan, source)
    _write_json(source / "reconciliation.json", readback.reconciliation.as_dict())
    _write_json(source / "primary_input_bindings.json", {"schema_version": 1})
    _write_restoration_tree(source, plan.cells[0].cell_id)
    _write_statistics(
        source,
        controls,
        source_readback_root_sha256=readback.readback_root_sha256,
    )
    _write_json(
        source / "status.json",
        {
            "status": "running",
            "run_id": source.name,
            "experiment_name": recovery_module.SOURCE_EXPERIMENT_NAME,
            "started_at_utc": "2026-07-19T01:27:36Z",
        },
    )
    _write_json(source / "primary_execution_gate.json", {"schema_version": 1})
    _write_json(
        source / "source_tree_manifest.json",
        {"schema_version": 1, "root_sha256": "d" * 64},
    )
    snapshot = collect_orphan_source_snapshot(source, plan=plan, controls=controls)
    mapping = _authorization_mapping(source=source, receipt=receipt, snapshot=snapshot)
    return _RecoveryFixture(
        runs_root=runs_root,
        source=source,
        destination=destination,
        receipt=receipt,
        plan=plan,
        controls=controls,
        snapshot=snapshot,
        authorization_mapping=mapping,
        authorization=RecoveryAuthorization.from_mapping(mapping),
    )


def _tree_state(root: Path) -> dict[str, tuple[int, int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            sha256_file(path),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _forbid_compute(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("primary recovery reached training or statistics recomputation")

    monkeypatch.setattr(primary_core_module, "execute_primary_matrix", forbidden)
    monkeypatch.setattr(primary_statistics_module, "aggregate_primary_statistics", forbidden)
    monkeypatch.setattr(primary_statistics_module, "_compute_statistics", forbidden)
    monkeypatch.setattr(recovery_module.importlib, "import_module", forbidden)


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_field",
        "missing_field",
        "outcomes_not_inspected",
        "wrong_disposition",
        "scientific_change",
        "active_receipt",
        "interruption_extra_field",
    ),
)
def test_recovery_authorization_schema_is_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    assert fixture.authorization.outcomes_inspected is True
    assert fixture.authorization.analysis_disposition == "amended_or_exploratory"
    payload = copy.deepcopy(fixture.authorization_mapping)
    if mutation == "extra_field":
        payload["unexpected"] = True
    elif mutation == "missing_field":
        payload.pop("reason")
    elif mutation == "outcomes_not_inspected":
        payload["outcomes_inspected"] = False
    elif mutation == "wrong_disposition":
        payload["analysis_disposition"] = "amended_before_outcome_inspection"
    elif mutation == "scientific_change":
        payload["scientific_method_changes"] = ["changed metric"]
    elif mutation == "active_receipt":
        payload["interruption_evidence"]["process_active"] = True
    else:
        payload["interruption_evidence"]["unexpected"] = True

    with pytest.raises(PrimaryRecoveryError):
        RecoveryAuthorization.from_mapping(payload)


@pytest.mark.parametrize("case", ("active_pid", "wrong_status", "sealed"))
def test_source_status_seal_and_pid_fail_before_snapshot_or_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    active = case == "active_pid"
    if case == "wrong_status":
        status = json.loads((fixture.source / "status.json").read_text(encoding="utf-8"))
        status["status"] = "failed"
        _write_json(fixture.source / "status.json", status)
    elif case == "sealed":
        _write_json(fixture.source / ".immutable.json", {"status": "completed"})

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("invalid orphan advanced to snapshot or physical copy")

    monkeypatch.setattr(recovery_module, "collect_orphan_source_snapshot", forbidden)
    monkeypatch.setattr(recovery_module, "anchored_physical_copy_session", forbidden)
    with pytest.raises(PrimaryRecoveryError):
        inspect_orphan_source(
            runs_root=fixture.runs_root,
            plan=fixture.plan,
            controls=fixture.controls,
            authorization=fixture.authorization,
            pid_probe=lambda _pid: active,
        )
    assert not fixture.destination.exists()


def test_exact_tiny_snapshot_is_read_only_and_allowlisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    before = _tree_state(fixture.source)
    inspection = inspect_orphan_source(
        runs_root=fixture.runs_root,
        plan=fixture.plan,
        controls=fixture.controls,
        authorization=fixture.authorization,
        pid_probe=lambda _pid: False,
    )
    snapshot = inspection.snapshot
    expected = {
        *_SOURCE_ROOT_FILES,
        *_STATISTICS_FILES,
        f"corruption_scenarios/{fixture.plan.scenarios[0].scenario_id}.json",
        *(
            f"cells/{fixture.plan.cells[0].cell_id}/{name}"
            for name in (*_CELL_ARTIFACTS, "artifact_manifest.json")
        ),
        *(
            f"restorations/{fixture.plan.cells[0].cell_id}/{name}"
            for name in (
                "restoration.json",
                "restoration_evidence.npz",
                "restoration_manifest.json",
            )
        ),
    }
    assert {record.path for record in snapshot.artifacts} == expected
    assert len(snapshot.artifacts) == 26
    assert snapshot.completed_required_cell_count == 1
    assert snapshot.skipped_optional_cell_count == 0
    assert snapshot.snapshot_root_sha256 == canonical_sha256(
        [record.as_dict() for record in snapshot.artifacts]
    )
    assert _tree_state(fixture.source) == before
    assert {"status.json", "primary_execution_gate.json", "source_tree_manifest.json"}.isdisjoint(
        expected
    )


@pytest.mark.skipif(os.name != "nt", reason="recovery copy requires Windows WOF/LZX")
def test_physical_copy_is_independent_and_destination_passes_light_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    _forbid_compute(monkeypatch)
    source_before = _tree_state(fixture.source)
    inspection = inspect_orphan_source(
        runs_root=fixture.runs_root,
        plan=fixture.plan,
        controls=fixture.controls,
        authorization=fixture.authorization,
        pid_probe=lambda _pid: False,
    )
    fixture.destination.mkdir()
    copy_calls = 0
    real_copy = recovery_module.anchored_physical_copy_session

    def counted_copy(*args: Any, **kwargs: Any) -> Any:
        nonlocal copy_calls
        copy_calls += 1
        if copy_calls > 1:
            raise AssertionError("primary recovery retried physical copy")
        return real_copy(*args, **kwargs)

    monkeypatch.setattr(recovery_module, "anchored_physical_copy_session", counted_copy)
    receipt = copy_authorized_orphan_artifacts(inspection, fixture.destination)
    verification = verify_recovery_destination(
        inspection,
        fixture.destination,
        plan=fixture.plan,
        controls=fixture.controls,
    )

    assert copy_calls == 1
    assert receipt.copy_policy == RECOVERY_COPY_POLICY
    assert receipt.artifact_count == 26
    assert verification.snapshot_root_sha256 == fixture.snapshot.snapshot_root_sha256
    assert verification.statistics_comparison_count == 0
    assert verification.bootstrap_saved_draw_count == fixture.controls.bootstrap_iterations
    assert _tree_state(fixture.source) == source_before
    for record in inspection.snapshot.artifacts:
        source = fixture.source / Path(record.path)
        copied = fixture.destination / Path(record.path)
        assert copied.is_file() and not copied.is_symlink()
        assert sha256_file(copied) == record.sha256
        assert copied.stat().st_nlink == 1
        assert not os.path.samefile(source, copied)


@pytest.mark.parametrize("mutation", ("missing_artifact", "same_size_tamper", "statistics_tamper"))
def test_missing_or_hash_drifted_source_is_rejected_without_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    if mutation == "missing_artifact":
        (fixture.source / "cells" / fixture.plan.cells[0].cell_id / "risk_scores.npz").unlink()
    elif mutation == "same_size_tamper":
        path = fixture.source / "cells" / fixture.plan.cells[0].cell_id / "neighbour_evidence.npz"
        payload = path.read_bytes()
        path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    else:
        path = fixture.source / "primary_statistics.json"
        path.write_bytes(path.read_bytes() + b" ")

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("tampered source reached physical copy")

    monkeypatch.setattr(recovery_module, "anchored_physical_copy_session", forbidden)
    with pytest.raises((PrimaryRecoveryError, ValueError)):
        inspect_orphan_source(
            runs_root=fixture.runs_root,
            plan=fixture.plan,
            controls=fixture.controls,
            authorization=fixture.authorization,
            pid_probe=lambda _pid: False,
        )
    assert not fixture.destination.exists()


@pytest.mark.skipif(os.name != "nt", reason="recovery copy requires Windows WOF/LZX")
def test_destination_light_closure_rejects_reauthorized_semantic_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    statistics_path = fixture.source / "primary_statistics.json"
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    statistics["analysis_scope"] = "tampered_scope"
    _write_json(statistics_path, statistics)
    manifest_path = fixture.source / "primary_statistics_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["statistics_payload_sha256"] = canonical_sha256(statistics)
    for record in manifest["artifacts"]:
        if record["path"] == "primary_statistics.json":
            record["size_bytes"] = statistics_path.stat().st_size
            record["sha256"] = sha256_file(statistics_path)
    _write_json(manifest_path, manifest)

    snapshot = collect_orphan_source_snapshot(
        fixture.source,
        plan=fixture.plan,
        controls=fixture.controls,
    )
    authorization = RecoveryAuthorization.from_mapping(
        _authorization_mapping(
            source=fixture.source,
            receipt=fixture.receipt,
            snapshot=snapshot,
        )
    )
    inspection = inspect_orphan_source(
        runs_root=fixture.runs_root,
        plan=fixture.plan,
        controls=fixture.controls,
        authorization=authorization,
        pid_probe=lambda _pid: False,
    )
    fixture.destination.mkdir()
    copy_authorized_orphan_artifacts(inspection, fixture.destination)

    with pytest.raises(PrimaryRecoveryError, match="lightweight semantic closure"):
        verify_recovery_destination(
            inspection,
            fixture.destination,
            plan=fixture.plan,
            controls=fixture.controls,
        )


def test_copy_failure_has_no_retry_training_or_statistics_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    _forbid_compute(monkeypatch)
    source_text = inspect.getsource(recovery_module)
    assert "execute_primary_matrix" not in source_text
    assert "aggregate_primary_statistics" not in source_text
    assert "_compute_statistics" not in source_text
    assert "histo_audit.experiment.primary_statistics" not in source_text
    inspection = inspect_orphan_source(
        runs_root=fixture.runs_root,
        plan=fixture.plan,
        controls=fixture.controls,
        authorization=fixture.authorization,
        pid_probe=lambda _pid: False,
    )
    fixture.destination.mkdir()
    factory_calls = 0
    file_calls = 0

    class FailingSession:
        def __enter__(self) -> FailingSession:
            return self

        def __exit__(self, *_args: Any) -> bool:
            return False

        def copy_file_no_overwrite(self, *_args: Any, **_kwargs: Any) -> Any:
            nonlocal file_calls
            file_calls += 1
            raise OSError("injected first-copy failure")

    def failing_factory(*_args: Any, **_kwargs: Any) -> FailingSession:
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls > 1:
            raise AssertionError("primary recovery retried after copy failure")
        return FailingSession()

    monkeypatch.setattr(
        recovery_module,
        "anchored_physical_copy_session",
        failing_factory,
    )
    with pytest.raises(OSError, match="injected first-copy failure"):
        copy_authorized_orphan_artifacts(inspection, fixture.destination)

    assert factory_calls == 1
    assert file_calls == 1
    assert not any(fixture.destination.rglob("*"))
