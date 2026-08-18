"""Completion-stage safety tests for the frozen primary matrix."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from test_study_contracts import complete_primary_config

from histo_audit.experiment import primary_completion as primary_completion_module
from histo_audit.experiment import primary_statistics as primary_statistics_module
from histo_audit.experiment.primary_completion import (
    REAL_PRIMARY_ARTIFACT_SCOPE,
    SYNTHETIC_PRIMARY_ARTIFACT_SCOPE,
    PrimaryRestorationReadbackEvidence,
    build_primary_completion_evidence,
    read_primary_filesystem_evidence,
    reconcile_primary_cell_outcomes,
)
from histo_audit.experiment.primary_statistics import PrimaryStatisticsVerification
from histo_audit.experiment.study_contracts import (
    PrimaryCell,
    PrimaryMatrixPlan,
    PrimaryScenario,
    build_primary_matrix_plan,
)
from histo_audit.utils.run_tracking import sha256_file
from histo_audit.workflows.study_gates import PrimaryExecutionGateEvidence

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


def _tiny_real_plan() -> PrimaryMatrixPlan:
    scenario = PrimaryScenario(
        scenario_id="scenario_shared",
        mechanism="symmetric_random_corruption",
        rate=0.1,
        corruption_seed=404,
    )
    cells = tuple(
        PrimaryCell(
            cell_id=f"primary_cell_{index}",
            scenario_id=scenario.scenario_id,
            mechanism=scenario.mechanism,
            rate=scenario.rate,
            corruption_seed=scenario.corruption_seed,
            representation_id=f"representation_{index}",
            classifier_id="multinomial_logistic_regression",
            required=True,
        )
        for index in range(2)
    )
    return PrimaryMatrixPlan(
        schema_version=2,
        config_sha256="f" * 64,
        scenarios=(scenario,),
        cells=cells,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_cell_artifact_manifest(cell_directory: Path) -> Path:
    records = [
        {
            "path": name,
            "size_bytes": (cell_directory / name).stat().st_size,
            "sha256": sha256_file(cell_directory / name),
        }
        for name in _CELL_ARTIFACTS
    ]
    path = cell_directory / "artifact_manifest.json"
    _write_json(path, {"schema_version": 1, "artifacts": records})
    return path


def _write_cell_index(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(field for field in row if field not in fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_primary_tree(
    root: Path,
    plan: PrimaryMatrixPlan,
    *,
    shared_hash_by_cell: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    root.mkdir()
    cells_root = root / "cells"
    cells_root.mkdir()
    scenarios_root = root / "corruption_scenarios"
    scenarios_root.mkdir()
    _write_json(root / "matrix_plan.json", plan.as_dict())
    _write_json(root / "execution_controls.json", {"binding_sha256": "e" * 64})
    scenario_by_id = {scenario.scenario_id: scenario for scenario in plan.scenarios}
    rows: list[dict[str, Any]] = []
    for cell in plan.cells:
        shared_hash = (
            shared_hash_by_cell[cell.cell_id] if shared_hash_by_cell is not None else "a" * 64
        )
        cell_directory = cells_root / cell.cell_id
        cell_directory.mkdir()
        identity = asdict(cell)
        scenario = asdict(scenario_by_id[cell.scenario_id])
        evidence_sha = "b" * 64
        matrix_sha = "c" * 64
        _write_json(
            cell_directory / "metrics.json",
            {
                "cell": identity,
                "scenario": scenario,
                "corruption_configuration_hash": shared_hash,
                "independence_status": "verified_independent",
                "independence_evidence_sha256": evidence_sha,
                "independence_matrix_artifact_sha256": matrix_sha,
                "circularity_risk": False,
                "primary_confirmatory_eligible": True,
                "paired_group_bootstrap": {
                    "fixed_hybrid_vs_self_confidence": {
                        "claim_status": "primary_confirmatory_eligible"
                    }
                },
            },
        )
        _write_json(
            cell_directory / "corruption_manifest.json",
            {
                "schema_version": 1,
                "cell": identity,
                "scenario": scenario,
                "configuration_hash": shared_hash,
                "shared_scenario_corruption_hash": shared_hash,
                "cell_corruption_provenance_sha256": "d" * 64,
                "rows": [],
                "independence_status": "verified_independent",
                "circularity_risk": False,
            },
        )
        _write_json(
            cell_directory / "cleanlab_evidence.json",
            {
                "schema_version": 1,
                "available": True,
                "package_version": "2.9.0",
                "api_path": "cleanlab.rank + cleanlab.filter",
                "error": None,
                "blocker": None,
                "failure_policy": "missing_with_recorded_blocker",
            },
        )
        _write_json(
            cell_directory / "independence_evidence.json",
            {
                "schema_version": 1,
                "mechanism": cell.mechanism,
                "representation_id": cell.representation_id,
                "status": "verified_independent",
                "reason": "separate generator and auditor representations",
                "circularity_risk": False,
                "primary_confirmatory_eligible": True,
                "matrix_artifact_sha256": matrix_sha,
                "evidence_sha256": evidence_sha,
                "evidence": {},
            },
        )
        for artifact in _CELL_ARTIFACTS:
            artifact_path = cell_directory / artifact
            if not artifact_path.exists():
                artifact_path.write_bytes(f"evidence:{cell.cell_id}:{artifact}".encode())
        manifest_path = _write_cell_artifact_manifest(cell_directory)
        rows.append(
            {
                **identity,
                "status": "completed",
                "artifact_manifest_sha256": sha256_file(manifest_path),
                "metrics_sha256": sha256_file(cell_directory / "metrics.json"),
                "corruption_configuration_hash": shared_hash,
                "execution_controls_binding_sha256": "e" * 64,
                "independence_status": "verified_independent",
                "independence_evidence_sha256": evidence_sha,
                "independence_matrix_artifact_sha256": matrix_sha,
                "circularity_risk": False,
                "primary_confirmatory_eligible": True,
            }
        )
    for scenario in plan.scenarios:
        scenario_cells = [cell for cell in plan.cells if cell.scenario_id == scenario.scenario_id]
        shared_hash = (
            shared_hash_by_cell[scenario_cells[0].cell_id]
            if shared_hash_by_cell is not None
            else "a" * 64
        )
        _write_json(
            scenarios_root / f"{scenario.scenario_id}.json",
            {
                "scenario": asdict(scenario),
                "shared_scenario_corruption_hash": shared_hash,
            },
        )
    _write_cell_index(root / "cell_index.csv", rows)
    return rows


def _real_gate(plan: PrimaryMatrixPlan, root: Path) -> PrimaryExecutionGateEvidence:
    return PrimaryExecutionGateEvidence(
        freeze_directory=root / "freeze",
        base_freeze_directory=root / "freeze",
        freeze_artifact_root_sha256="1" * 64,
        freeze_manifest_sha256="2" * 64,
        preregistration_sha256="3" * 64,
        frozen_primary_config_sha256="4" * 64,
        frozen_confirmatory_config_sha256="5" * 64,
        primary_config_semantic_sha256=plan.config_sha256,
        confirmatory_config_semantic_sha256="6" * 64,
        primary_matrix_cell_count=len(plan.cells),
        primary_required_cell_count=plan.required_cell_count,
        confirmatory_matrix_cell_count=1,
        pilot_run_id="real-pilot-run",
        pilot_artifact_root_sha256="7" * 64,
        dataset_sha256="8" * 64,
        manifest_sha256="9" * 64,
        duplicate_audit_sha256="a" * 64,
        pathology_encoder_audit_sha256="b" * 64,
        source_tree_root_sha256="c" * 64,
    )


def _successful_outcomes() -> tuple[Any, list[dict[str, Any]]]:
    plan = build_primary_matrix_plan(complete_primary_config())
    outcomes: list[dict[str, Any]] = []
    for cell in plan.cells:
        if cell.required:
            outcomes.append(
                {
                    "cell_id": cell.cell_id,
                    "required": True,
                    "status": "completed",
                    "artifact_manifest_sha256": "a" * 64,
                    "metrics_sha256": "b" * 64,
                }
            )
        else:
            outcomes.append(
                {
                    "cell_id": cell.cell_id,
                    "required": False,
                    "status": "skipped_with_frozen_blocker",
                    "frozen_unavailability": True,
                    "blocker": "pathology encoder unavailable under frozen priority rule",
                }
            )
    return plan, outcomes


def _gate_hashes() -> dict[str, str]:
    return {
        "freeze_artifact_root_sha256": "1" * 64,
        "frozen_primary_config_sha256": "2" * 64,
        "frozen_confirmatory_config_sha256": "3" * 64,
        "dataset_sha256": "4" * 64,
        "manifest_sha256": "5" * 64,
        "duplicate_audit_sha256": "6" * 64,
        "pathology_encoder_audit_sha256": "7" * 64,
        "source_tree_root_sha256": "8" * 64,
    }


def _attested_statistics_verification(
    run_directory: Path,
    *,
    source_readback_root_sha256: str,
) -> PrimaryStatisticsVerification:
    _write_json(run_directory / "primary_statistics.json", {"fixture": "statistics"})
    (run_directory / "primary_bootstrap_evidence.npz").write_bytes(b"fixture-bootstrap")
    (run_directory / "primary_subgroups.csv").write_text(
        "cell_id,average_precision\nfixture,0.5\n", encoding="utf-8"
    )
    _write_json(
        run_directory / "primary_statistics_manifest.json",
        {"fixture": "strict-verifier-output"},
    )
    verification = PrimaryStatisticsVerification(
        status="passed",
        output_directory=run_directory.resolve(),
        statistics_sha256=sha256_file(run_directory / "primary_statistics.json"),
        bootstrap_evidence_sha256=sha256_file(run_directory / "primary_bootstrap_evidence.npz"),
        subgroups_sha256=sha256_file(run_directory / "primary_subgroups.csv"),
        manifest_sha256=sha256_file(run_directory / "primary_statistics_manifest.json"),
        source_readback_root_sha256=source_readback_root_sha256,
        comparison_count=3,
    )
    object.__setattr__(
        verification,
        "_attestation",
        primary_statistics_module._STATISTICS_VERIFICATION_ATTESTATION,
    )
    return verification


def _attested_restoration_readback(
    run_directory: Path,
    plan: PrimaryMatrixPlan,
    *,
    source_readback_root_sha256: str,
) -> PrimaryRestorationReadbackEvidence:
    _write_json(run_directory / "restoration_index.json", {"fixture": "restoration"})
    json_hashes: list[tuple[str, str]] = []
    evidence_hashes: list[tuple[str, str]] = []
    manifest_hashes: list[tuple[str, str]] = []
    for cell in plan.cells:
        cell_directory = run_directory / "restorations" / cell.cell_id
        cell_directory.mkdir(parents=True)
        _write_json(cell_directory / "restoration.json", {"cell_id": cell.cell_id})
        (cell_directory / "restoration_evidence.npz").write_bytes(
            f"fixture:{cell.cell_id}".encode()
        )
        _write_json(cell_directory / "restoration_manifest.json", {"cell_id": cell.cell_id})
        json_hashes.append((cell.cell_id, sha256_file(cell_directory / "restoration.json")))
        evidence_hashes.append(
            (cell.cell_id, sha256_file(cell_directory / "restoration_evidence.npz"))
        )
        manifest_hashes.append(
            (cell.cell_id, sha256_file(cell_directory / "restoration_manifest.json"))
        )
    readback = PrimaryRestorationReadbackEvidence(
        run_directory=run_directory.resolve(),
        status="passed",
        restoration_index_sha256=sha256_file(run_directory / "restoration_index.json"),
        readback_root_sha256="e" * 64,
        source_readback_root_sha256=source_readback_root_sha256,
        restoration_cell_count=len(plan.cells),
        downstream_comparison_count=1,
        cell_json_sha256=tuple(json_hashes),
        cell_evidence_sha256=tuple(evidence_hashes),
        cell_manifest_sha256=tuple(manifest_hashes),
    )
    object.__setattr__(
        readback,
        "_attestation",
        primary_completion_module._RESTORATION_READBACK_ATTESTATION,
    )
    return readback


def test_filesystem_readback_can_enable_real_primary_with_typed_gate(tmp_path: Path) -> None:
    plan = _tiny_real_plan()
    run_directory = tmp_path / "real-primary"
    _build_primary_tree(run_directory, plan)
    readback = read_primary_filesystem_evidence(plan, run_directory)
    statistics_verification = _attested_statistics_verification(
        run_directory,
        source_readback_root_sha256=readback.readback_root_sha256,
    )
    restoration_readback = _attested_restoration_readback(
        run_directory,
        plan,
        source_readback_root_sha256=readback.readback_root_sha256,
    )
    evidence = build_primary_completion_evidence(
        plan=plan,
        reconciliation=readback.reconciliation,
        artifact_scope=REAL_PRIMARY_ARTIFACT_SCOPE,
        study_outcome_eligible=True,
        gate_evidence=_real_gate(plan, tmp_path),
        filesystem_readback=readback,
        statistics_verification=statistics_verification,
        restoration_readback=restoration_readback,
    )

    assert readback.passed
    assert readback.completed_required_cell_count == 2
    assert evidence["completion_stage"] == "PRIMARY_STUDY_COMPLETE"
    assert evidence["failed_required_cell_count"] == 0
    assert evidence["filesystem_readback_root_sha256"] == readback.readback_root_sha256
    assert evidence["primary_statistics_verification_status"] == "passed"
    assert evidence["primary_statistics_sha256"] == statistics_verification.statistics_sha256
    assert evidence["primary_restoration_verification_status"] == "passed"


def test_primary_completion_requires_attested_restoration_readback(tmp_path: Path) -> None:
    plan = _tiny_real_plan()
    run_directory = tmp_path / "missing-restoration"
    _build_primary_tree(run_directory, plan)
    readback = read_primary_filesystem_evidence(plan, run_directory)
    statistics_verification = _attested_statistics_verification(
        run_directory,
        source_readback_root_sha256=readback.readback_root_sha256,
    )

    with pytest.raises(ValueError, match="attested restoration readback"):
        build_primary_completion_evidence(
            plan=plan,
            reconciliation=readback.reconciliation,
            artifact_scope=REAL_PRIMARY_ARTIFACT_SCOPE,
            study_outcome_eligible=True,
            gate_evidence=_real_gate(plan, tmp_path),
            filesystem_readback=readback,
            statistics_verification=statistics_verification,
        )


def test_primary_completion_requires_attested_statistics_verification(tmp_path: Path) -> None:
    plan = _tiny_real_plan()
    run_directory = tmp_path / "missing-statistics"
    _build_primary_tree(run_directory, plan)
    readback = read_primary_filesystem_evidence(plan, run_directory)

    with pytest.raises(ValueError, match="attested primary statistics verification"):
        build_primary_completion_evidence(
            plan=plan,
            reconciliation=readback.reconciliation,
            artifact_scope=REAL_PRIMARY_ARTIFACT_SCOPE,
            study_outcome_eligible=True,
            gate_evidence=_real_gate(plan, tmp_path),
            filesystem_readback=readback,
        )


def test_primary_completion_rechecks_statistics_hashes_after_verification(
    tmp_path: Path,
) -> None:
    plan = _tiny_real_plan()
    run_directory = tmp_path / "statistics-toctou"
    _build_primary_tree(run_directory, plan)
    readback = read_primary_filesystem_evidence(plan, run_directory)
    verification = _attested_statistics_verification(
        run_directory,
        source_readback_root_sha256=readback.readback_root_sha256,
    )
    _write_json(run_directory / "primary_statistics.json", {"tampered": True})

    with pytest.raises(ValueError, match="changed after strict verification"):
        build_primary_completion_evidence(
            plan=plan,
            reconciliation=readback.reconciliation,
            artifact_scope=REAL_PRIMARY_ARTIFACT_SCOPE,
            study_outcome_eligible=True,
            gate_evidence=_real_gate(plan, tmp_path),
            filesystem_readback=readback,
            statistics_verification=verification,
        )


def test_sha_shaped_mapping_is_not_primary_gate_evidence(tmp_path: Path) -> None:
    plan = _tiny_real_plan()
    run_directory = tmp_path / "real-primary"
    _build_primary_tree(run_directory, plan)
    readback = read_primary_filesystem_evidence(plan, run_directory)

    with pytest.raises(TypeError, match="real PrimaryExecutionGateEvidence"):
        build_primary_completion_evidence(
            plan=plan,
            reconciliation=readback.reconciliation,
            artifact_scope=REAL_PRIMARY_ARTIFACT_SCOPE,
            study_outcome_eligible=True,
            gate_evidence=_gate_hashes(),  # type: ignore[arg-type]
            filesystem_readback=readback,
        )


def test_synthetic_matrix_fixture_never_enables_primary_completion() -> None:
    plan, outcomes = _successful_outcomes()
    reconciliation = reconcile_primary_cell_outcomes(plan, outcomes)

    evidence = build_primary_completion_evidence(
        plan=plan,
        reconciliation=reconciliation,
        artifact_scope=SYNTHETIC_PRIMARY_ARTIFACT_SCOPE,
        study_outcome_eligible=False,
    )

    assert evidence["completion_stage"] is None
    assert evidence["study_outcome_eligible"] is False
    with pytest.raises(ValueError, match="synthetic primary fixtures"):
        build_primary_completion_evidence(
            plan=plan,
            reconciliation=reconciliation,
            artifact_scope=SYNTHETIC_PRIMARY_ARTIFACT_SCOPE,
            study_outcome_eligible=True,
            gate_evidence=_gate_hashes(),
        )


def test_missing_required_cell_blocks_completion() -> None:
    plan, outcomes = _successful_outcomes()
    removed = next(index for index, item in enumerate(outcomes) if item["required"])
    outcomes.pop(removed)
    reconciliation = reconcile_primary_cell_outcomes(plan, outcomes)

    assert not reconciliation.passed
    assert len(reconciliation.missing_cell_ids) == 1
    with pytest.raises(ValueError, match="passed matrix reconciliation"):
        build_primary_completion_evidence(
            plan=plan,
            reconciliation=reconciliation,
            artifact_scope=REAL_PRIMARY_ARTIFACT_SCOPE,
            study_outcome_eligible=True,
            gate_evidence=_gate_hashes(),
        )


def test_required_skip_and_optional_execution_failure_both_block_completion() -> None:
    plan, outcomes = _successful_outcomes()
    required = next(item for item in outcomes if item["required"])
    required.update(
        status="skipped_with_frozen_blocker",
        frozen_unavailability=True,
        blocker="not permitted for a required cell",
    )
    optional = next(item for item in outcomes if not item["required"])
    optional.update(status="failed", error="CUDA OOM at minimum batch")
    reconciliation = reconcile_primary_cell_outcomes(plan, outcomes)

    assert not reconciliation.passed
    assert reconciliation.failed_cell_count == 1
    assert required["cell_id"] in reconciliation.invalid_cell_ids


def test_completed_cell_requires_hash_bound_metrics_and_artifact_manifest() -> None:
    plan, outcomes = _successful_outcomes()
    completed = next(item for item in outcomes if item["required"])
    completed["metrics_sha256"] = "not-a-hash"

    reconciliation = reconcile_primary_cell_outcomes(plan, outcomes)

    assert not reconciliation.passed
    assert completed["cell_id"] in reconciliation.invalid_cell_ids


def test_duplicate_or_unplanned_cell_outcome_fails_exact_set_reconciliation() -> None:
    plan, outcomes = _successful_outcomes()
    outcomes.append(dict(outcomes[0]))
    outcomes.append(
        {
            "cell_id": "primary_unplanned_cell",
            "required": True,
            "status": "completed",
            "artifact_manifest_sha256": "a" * 64,
            "metrics_sha256": "b" * 64,
        }
    )

    reconciliation = reconcile_primary_cell_outcomes(plan, outcomes)

    assert not reconciliation.passed
    assert reconciliation.duplicate_cell_ids == (outcomes[0]["cell_id"],)
    assert reconciliation.extra_cell_ids == ("primary_unplanned_cell",)


def test_filesystem_readback_rejects_missing_required_cell_artifact(tmp_path: Path) -> None:
    plan = _tiny_real_plan()
    run_directory = tmp_path / "missing-artifact"
    _build_primary_tree(run_directory, plan)
    (run_directory / "cells" / plan.cells[0].cell_id / "risk_scores.npz").unlink()

    with pytest.raises(ValueError, match="filesystem artifact set mismatch"):
        read_primary_filesystem_evidence(plan, run_directory)


def test_filesystem_readback_rejects_tamper_even_after_superficial_hash_updates(
    tmp_path: Path,
) -> None:
    plan = _tiny_real_plan()
    run_directory = tmp_path / "tampered-metrics"
    rows = _build_primary_tree(run_directory, plan)
    cell_directory = run_directory / "cells" / plan.cells[0].cell_id
    metrics_path = cell_directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["cell"]["representation_id"] = "tampered_representation"
    _write_json(metrics_path, metrics)
    manifest_path = _write_cell_artifact_manifest(cell_directory)
    rows[0]["metrics_sha256"] = sha256_file(metrics_path)
    rows[0]["artifact_manifest_sha256"] = sha256_file(manifest_path)
    _write_cell_index(run_directory / "cell_index.csv", rows)

    with pytest.raises(ValueError, match="metrics identity differs"):
        read_primary_filesystem_evidence(plan, run_directory)


def test_filesystem_readback_rejects_cell_identity_swap(tmp_path: Path) -> None:
    plan = _tiny_real_plan()
    run_directory = tmp_path / "identity-swap"
    rows = _build_primary_tree(run_directory, plan)
    rows[0]["representation_id"], rows[1]["representation_id"] = (
        rows[1]["representation_id"],
        rows[0]["representation_id"],
    )
    _write_cell_index(run_directory / "cell_index.csv", rows)

    with pytest.raises(ValueError, match="identity differs from the frozen plan"):
        read_primary_filesystem_evidence(plan, run_directory)


def test_filesystem_readback_rejects_scenario_corruption_hash_mismatch(
    tmp_path: Path,
) -> None:
    plan = _tiny_real_plan()
    run_directory = tmp_path / "scenario-mismatch"
    _build_primary_tree(
        run_directory,
        plan,
        shared_hash_by_cell={
            plan.cells[0].cell_id: "a" * 64,
            plan.cells[1].cell_id: "b" * 64,
        },
    )

    with pytest.raises(ValueError, match="different shared corruption hashes"):
        read_primary_filesystem_evidence(plan, run_directory)


def test_filesystem_readback_normalises_empty_optional_independence_hashes(
    tmp_path: Path,
) -> None:
    plan = _tiny_real_plan()
    run_directory = tmp_path / "optional-independence-hashes"
    rows = _build_primary_tree(run_directory, plan)
    cell = plan.cells[0]
    cell_directory = run_directory / "cells" / cell.cell_id
    metrics_path = cell_directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["independence_evidence_sha256"] = None
    metrics["independence_matrix_artifact_sha256"] = None
    _write_json(metrics_path, metrics)
    independence_path = cell_directory / "independence_evidence.json"
    independence = json.loads(independence_path.read_text(encoding="utf-8"))
    independence["evidence_sha256"] = None
    independence["matrix_artifact_sha256"] = None
    _write_json(independence_path, independence)
    manifest_path = _write_cell_artifact_manifest(cell_directory)
    rows[0]["independence_evidence_sha256"] = None
    rows[0]["independence_matrix_artifact_sha256"] = None
    rows[0]["metrics_sha256"] = sha256_file(metrics_path)
    rows[0]["artifact_manifest_sha256"] = sha256_file(manifest_path)
    _write_cell_index(run_directory / "cell_index.csv", rows)

    evidence = read_primary_filesystem_evidence(plan, run_directory)

    assert evidence.passed


def test_circularity_risk_cell_is_bound_and_excluded_from_confirmatory_claims(
    tmp_path: Path,
) -> None:
    plan = _tiny_real_plan()
    run_directory = tmp_path / "circularity-exclusion"
    rows = _build_primary_tree(run_directory, plan)
    cell = plan.cells[0]
    cell_directory = run_directory / "cells" / cell.cell_id
    metrics_path = cell_directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(
        independence_status="circularity_risk",
        circularity_risk=True,
        primary_confirmatory_eligible=False,
    )
    metrics["paired_group_bootstrap"]["fixed_hybrid_vs_self_confidence"]["claim_status"] = (
        "excluded_circularity_risk"
    )
    _write_json(metrics_path, metrics)
    corruption_path = cell_directory / "corruption_manifest.json"
    corruption = json.loads(corruption_path.read_text(encoding="utf-8"))
    corruption.update(independence_status="circularity_risk", circularity_risk=True)
    _write_json(corruption_path, corruption)
    independence_path = cell_directory / "independence_evidence.json"
    independence = json.loads(independence_path.read_text(encoding="utf-8"))
    independence.update(
        status="circularity_risk",
        circularity_risk=True,
        primary_confirmatory_eligible=False,
    )
    _write_json(independence_path, independence)
    manifest_path = _write_cell_artifact_manifest(cell_directory)
    rows[0].update(
        independence_status="circularity_risk",
        circularity_risk=True,
        primary_confirmatory_eligible=False,
        metrics_sha256=sha256_file(metrics_path),
        artifact_manifest_sha256=sha256_file(manifest_path),
    )
    _write_cell_index(run_directory / "cell_index.csv", rows)

    readback = read_primary_filesystem_evidence(plan, run_directory)
    statistics_verification = _attested_statistics_verification(
        run_directory,
        source_readback_root_sha256=readback.readback_root_sha256,
    )
    restoration_readback = _attested_restoration_readback(
        run_directory,
        plan,
        source_readback_root_sha256=readback.readback_root_sha256,
    )
    completion = build_primary_completion_evidence(
        plan=plan,
        reconciliation=readback.reconciliation,
        artifact_scope=REAL_PRIMARY_ARTIFACT_SCOPE,
        study_outcome_eligible=True,
        gate_evidence=_real_gate(plan, tmp_path),
        filesystem_readback=readback,
        statistics_verification=statistics_verification,
        restoration_readback=restoration_readback,
    )

    assert readback.circularity_excluded_cell_ids == (cell.cell_id,)
    assert completion["completion_stage"] == "PRIMARY_STUDY_COMPLETE"
    assert completion["circularity_excluded_cell_count"] == 1
    assert completion["circularity_excluded_cell_ids"] == [cell.cell_id]
    assert completion["primary_confirmatory_claims_require_exclusion_of_these_cells"] is True
