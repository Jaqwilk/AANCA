"""Fail-closed reconciliation and completion semantics for the primary matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.evaluation.restoration import classification_metrics, restore_reviewed_labels
from histo_audit.experiment.study_contracts import PrimaryMatrixPlan
from histo_audit.statistics.review import budget_count, rank_indices
from histo_audit.utils.run_tracking import sha256_file

if TYPE_CHECKING:
    from histo_audit.experiment.primary_core import PrimaryExecutionControls
    from histo_audit.experiment.primary_statistics import (
        AuthorizedOrphanNumericVerificationProof,
        AuthorizedPriorNumericVerificationProof,
        InheritedPrimaryStatisticsVerification,
        PrimaryStatisticsVerification,
    )
    from histo_audit.workflows.study_gates import PrimaryExecutionGateEvidence

REAL_PRIMARY_ARTIFACT_SCOPE = "real_pannuke_primary_study"
SYNTHETIC_PRIMARY_ARTIFACT_SCOPE = "synthetic_primary_orchestrator_integration_test"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATUSES = {"completed", "skipped_with_frozen_blocker", "failed"}
_REQUIRED_GATE_HASHES = (
    "freeze_artifact_root_sha256",
    "frozen_primary_config_sha256",
    "frozen_confirmatory_config_sha256",
    "dataset_sha256",
    "manifest_sha256",
    "duplicate_audit_sha256",
    "pathology_encoder_audit_sha256",
    "source_tree_root_sha256",
)
_REQUIRED_COMPLETED_CELL_ARTIFACTS = (
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
_FILESYSTEM_READBACK_ATTESTATION = object()
_RESTORATION_READBACK_ATTESTATION = object()
_RESTORATION_CONDITIONS = {
    "uncorrupted_reference_baseline",
    "corrupted_observed_baseline",
    "random_review_restoration",
    "audit_guided_restoration",
}
_RESTORATION_BASE_ARRAYS = {
    "class_order",
    "audit_sample_ids",
    "audit_group_ids",
    "audit_pre_corruption_labels",
    "audit_observed_labels",
    "audit_is_injected_corruption",
    "audit_risk_scores",
    "final_test_sample_ids",
    "final_test_group_ids",
    "final_test_labels",
    "audit_reviewed_indices",
    "guided_reviewed_mask",
    "guided_restored_mask",
    "guided_restored_labels",
    "random_reviewed_masks",
    "random_restored_masks",
    "random_restored_labels",
    "uncorrupted_final_probabilities",
    "uncorrupted_final_predicted_class",
    "corrupted_final_probabilities",
    "corrupted_final_predicted_class",
    "guided_final_probabilities",
    "guided_final_predicted_class",
    "random_final_probabilities",
    "random_final_predicted_class",
    "random_reviewed_indices",
    "downstream_comparison_ids",
    "random_review_seeds",
}


@dataclass(frozen=True, slots=True)
class PrimaryMatrixReconciliation:
    """Exact set/count reconciliation for one primary execution matrix."""

    status: str
    planned_cell_count: int
    planned_required_cell_count: int
    completed_cell_count: int
    completed_required_cell_count: int
    skipped_optional_cell_count: int
    failed_cell_count: int
    missing_cell_ids: tuple[str, ...]
    extra_cell_ids: tuple[str, ...]
    duplicate_cell_ids: tuple[str, ...]
    invalid_cell_ids: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PrimaryFilesystemReadbackEvidence:
    """Attested readback of one concrete primary matrix artifact tree."""

    run_directory: Path
    status: str
    matrix_plan_sha256: str
    execution_controls_sha256: str
    execution_controls_binding_sha256: str
    cell_index_sha256: str
    readback_root_sha256: str
    planned_cell_count: int
    completed_cell_count: int
    completed_required_cell_count: int
    skipped_optional_cell_count: int
    circularity_excluded_cell_ids: tuple[str, ...]
    cell_artifact_manifest_sha256: tuple[tuple[str, str], ...]
    scenario_artifact_sha256: tuple[tuple[str, str], ...]
    scenario_corruption_sha256: tuple[tuple[str, str], ...]
    reconciliation: PrimaryMatrixReconciliation
    _attestation: object | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def passed(self) -> bool:
        return (
            self.status == "passed"
            and self.reconciliation.passed
            and self._attestation is _FILESYSTEM_READBACK_ATTESTATION
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_directory": str(self.run_directory),
            "status": self.status,
            "matrix_plan_sha256": self.matrix_plan_sha256,
            "execution_controls_sha256": self.execution_controls_sha256,
            "execution_controls_binding_sha256": self.execution_controls_binding_sha256,
            "cell_index_sha256": self.cell_index_sha256,
            "readback_root_sha256": self.readback_root_sha256,
            "planned_cell_count": self.planned_cell_count,
            "completed_cell_count": self.completed_cell_count,
            "completed_required_cell_count": self.completed_required_cell_count,
            "skipped_optional_cell_count": self.skipped_optional_cell_count,
            "circularity_excluded_cell_count": len(self.circularity_excluded_cell_ids),
            "circularity_excluded_cell_ids": list(self.circularity_excluded_cell_ids),
            "cell_artifact_manifest_sha256": dict(self.cell_artifact_manifest_sha256),
            "scenario_artifact_sha256": dict(self.scenario_artifact_sha256),
            "scenario_corruption_sha256": dict(self.scenario_corruption_sha256),
            "reconciliation": self.reconciliation.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class PrimaryRestorationReadbackEvidence:
    """Strict semantic readback of every frozen downstream restoration cell."""

    run_directory: Path
    status: str
    restoration_index_sha256: str
    readback_root_sha256: str
    source_readback_root_sha256: str
    restoration_cell_count: int
    downstream_comparison_count: int
    cell_json_sha256: tuple[tuple[str, str], ...]
    cell_evidence_sha256: tuple[tuple[str, str], ...]
    cell_manifest_sha256: tuple[tuple[str, str], ...]
    _attestation: object | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def passed(self) -> bool:
        return self.status == "passed" and self._attestation is _RESTORATION_READBACK_ATTESTATION

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_directory": str(self.run_directory),
            "status": self.status,
            "restoration_index_sha256": self.restoration_index_sha256,
            "readback_root_sha256": self.readback_root_sha256,
            "source_readback_root_sha256": self.source_readback_root_sha256,
            "restoration_cell_count": self.restoration_cell_count,
            "downstream_comparison_count": self.downstream_comparison_count,
            "cell_json_sha256": dict(self.cell_json_sha256),
            "cell_evidence_sha256": dict(self.cell_evidence_sha256),
            "cell_manifest_sha256": dict(self.cell_manifest_sha256),
        }


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def reconcile_primary_cell_outcomes(
    plan: PrimaryMatrixPlan,
    outcomes: Sequence[Mapping[str, Any]],
) -> PrimaryMatrixReconciliation:
    """Reconcile saved per-cell outcomes against the immutable matrix plan.

    Required cells must complete.  Optional cells may only be skipped for an explicitly
    frozen availability blocker; execution failure is never treated as unavailability.
    """

    expected = {cell.cell_id: cell for cell in plan.cells}
    seen: dict[str, Mapping[str, Any]] = {}
    duplicates: set[str] = set()
    invalid: set[str] = set()
    errors: list[str] = []
    completed = 0
    completed_required = 0
    skipped_optional = 0
    failed = 0
    for index, outcome in enumerate(outcomes):
        cell_id = str(outcome.get("cell_id", ""))
        if not cell_id:
            errors.append(f"outcomes[{index}] lacks a non-empty cell_id")
            continue
        if cell_id in seen:
            duplicates.add(cell_id)
            continue
        seen[cell_id] = outcome
        planned = expected.get(cell_id)
        if planned is None:
            continue
        status = str(outcome.get("status", ""))
        if status not in _STATUSES:
            invalid.add(cell_id)
            errors.append(f"cell {cell_id} has unsupported status {status!r}")
            continue
        if outcome.get("required") is not planned.required:
            invalid.add(cell_id)
            errors.append(f"cell {cell_id} required flag differs from the frozen plan")
        if status == "completed":
            completed += 1
            completed_required += int(planned.required)
            if not _valid_sha(outcome.get("artifact_manifest_sha256")):
                invalid.add(cell_id)
                errors.append(f"completed cell {cell_id} lacks artifact_manifest_sha256")
            if not _valid_sha(outcome.get("metrics_sha256")):
                invalid.add(cell_id)
                errors.append(f"completed cell {cell_id} lacks metrics_sha256")
        elif status == "skipped_with_frozen_blocker":
            if planned.required:
                invalid.add(cell_id)
                errors.append(f"required cell {cell_id} cannot be skipped")
            elif (
                outcome.get("frozen_unavailability") is not True
                or not str(outcome.get("blocker", "")).strip()
            ):
                invalid.add(cell_id)
                errors.append(
                    f"optional skipped cell {cell_id} lacks a frozen unavailability blocker"
                )
            else:
                skipped_optional += 1
        else:
            failed += 1
            if not str(outcome.get("error", "")).strip():
                invalid.add(cell_id)
                errors.append(f"failed cell {cell_id} lacks error evidence")
    expected_ids = set(expected)
    seen_ids = set(seen)
    missing = tuple(sorted(expected_ids.difference(seen_ids)))
    extra = tuple(sorted(seen_ids.difference(expected_ids)))
    if missing:
        errors.append(f"matrix is missing {len(missing)} planned cells")
    if extra:
        errors.append(f"matrix contains {len(extra)} unplanned cells")
    if duplicates:
        errors.append(f"matrix contains duplicate outcomes for {len(duplicates)} cells")
    required_count = plan.required_cell_count
    if completed_required != required_count:
        errors.append(
            f"completed required-cell count {completed_required} differs from {required_count}"
        )
    optional_count = plan.optional_cell_count
    completed_optional = completed - completed_required
    if completed_optional + skipped_optional != optional_count:
        errors.append("optional cells are not all completed or skipped with a frozen blocker")
    if failed:
        errors.append(f"matrix contains {failed} execution failures")
    return PrimaryMatrixReconciliation(
        status="passed" if not errors and not invalid else "failed",
        planned_cell_count=len(plan.cells),
        planned_required_cell_count=required_count,
        completed_cell_count=completed,
        completed_required_cell_count=completed_required,
        skipped_optional_cell_count=skipped_optional,
        failed_cell_count=failed,
        missing_cell_ids=missing,
        extra_cell_ids=extra,
        duplicate_cell_ids=tuple(sorted(duplicates)),
        invalid_cell_ids=tuple(sorted(invalid)),
        errors=tuple(errors),
    )


def _read_json_object(path: Path, role: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is missing or invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{role} must be a JSON object: {path}")
    return payload


def _canonical_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _csv_boolean(value: Any, role: str) -> bool:
    normalised = str(value).strip().lower()
    if normalised == "true":
        return True
    if normalised == "false":
        return False
    raise ValueError(f"{role} must be exactly true or false")


def _cell_identity_from_csv(row: Mapping[str, Any], role: str) -> dict[str, Any]:
    try:
        return {
            "cell_id": str(row["cell_id"]),
            "scenario_id": str(row["scenario_id"]),
            "mechanism": str(row["mechanism"]),
            "rate": float(row["rate"]),
            "corruption_seed": int(row["corruption_seed"]),
            "representation_id": str(row["representation_id"]),
            "classifier_id": str(row["classifier_id"]),
            "required": _csv_boolean(row["required"], f"{role}.required"),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{role} lacks a valid exact cell identity: {exc}") from exc


def _validate_cell_artifact_manifest(
    cell_directory: Path,
    *,
    expected_manifest_sha256: Any,
) -> tuple[str, dict[str, str]]:
    manifest_path = cell_directory / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"completed cell lacks a real artifact_manifest.json: {cell_directory.name}"
        )
    actual_manifest_sha = sha256_file(manifest_path)
    if expected_manifest_sha256 != actual_manifest_sha:
        raise ValueError(
            f"cell {cell_directory.name} artifact manifest hash differs from cell_index"
        )
    manifest = _read_json_object(manifest_path, f"cell {cell_directory.name} artifact manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError(f"cell {cell_directory.name} artifact manifest schema is unsupported")
    raw_records = manifest.get("artifacts")
    if not isinstance(raw_records, list):
        raise ValueError(f"cell {cell_directory.name} artifact manifest lacks artifact records")
    records: dict[str, Mapping[str, Any]] = {}
    duplicates: set[str] = set()
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"cell {cell_directory.name} artifact manifest has a non-object row")
        artifact_name = str(raw_record.get("path", ""))
        if not artifact_name or Path(artifact_name).name != artifact_name:
            raise ValueError(f"cell {cell_directory.name} manifest contains an unsafe path")
        if artifact_name in records:
            duplicates.add(artifact_name)
        records[artifact_name] = raw_record
    if duplicates:
        raise ValueError(
            f"cell {cell_directory.name} artifact manifest has duplicate paths: "
            f"{sorted(duplicates)}"
        )
    expected_names = set(_REQUIRED_COMPLETED_CELL_ARTIFACTS)
    actual_names = set(records)
    missing = sorted(expected_names.difference(actual_names))
    extra = sorted(actual_names.difference(expected_names))
    if missing or extra:
        raise ValueError(
            f"cell {cell_directory.name} artifact manifest set mismatch: "
            f"missing={missing}, extra={extra}"
        )
    directory_names = {
        path.name
        for path in cell_directory.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    non_files = sorted(path.name for path in cell_directory.iterdir() if not path.is_file())
    if directory_names != expected_names or non_files:
        raise ValueError(
            f"cell {cell_directory.name} filesystem artifact set mismatch: "
            f"missing={sorted(expected_names.difference(directory_names))}, "
            f"extra={sorted(directory_names.difference(expected_names)) + non_files}"
        )
    recalculated: dict[str, str] = {}
    for artifact_name in _REQUIRED_COMPLETED_CELL_ARTIFACTS:
        path = cell_directory / artifact_name
        record = records[artifact_name]
        actual_sha = sha256_file(path)
        actual_size = path.stat().st_size
        if record.get("sha256") != actual_sha or record.get("size_bytes") != actual_size:
            raise ValueError(
                f"cell {cell_directory.name} artifact record differs from disk: {artifact_name}"
            )
        recalculated[artifact_name] = actual_sha
    return actual_manifest_sha, recalculated


def _validate_cleanlab_metadata(cell_id: str, payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 1 or payload.get("failure_policy") != (
        "missing_with_recorded_blocker"
    ):
        raise ValueError(f"cell {cell_id} Cleanlab evidence has an invalid schema/failure policy")
    available = payload.get("available")
    if available is True:
        if (
            not str(payload.get("package_version", "")).strip()
            or not str(payload.get("api_path", "")).strip()
            or payload.get("error") is not None
            or payload.get("blocker") is not None
        ):
            raise ValueError(f"cell {cell_id} available Cleanlab evidence lacks provenance")
    elif available is False:
        if not str(payload.get("error") or payload.get("blocker") or "").strip():
            raise ValueError(f"cell {cell_id} unavailable Cleanlab evidence lacks a blocker")
    else:
        raise ValueError(f"cell {cell_id} Cleanlab availability is not boolean")


def read_primary_filesystem_evidence(
    plan: PrimaryMatrixPlan,
    run_directory: str | Path,
) -> PrimaryFilesystemReadbackEvidence:
    """Read and cryptographically reconcile one concrete primary matrix directory."""

    run_path = Path(run_directory).resolve()
    matrix_plan_path = run_path / "matrix_plan.json"
    execution_controls_path = run_path / "execution_controls.json"
    cell_index_path = run_path / "cell_index.csv"
    cells_root = run_path / "cells"
    scenarios_root = run_path / "corruption_scenarios"
    if (
        not matrix_plan_path.is_file()
        or not execution_controls_path.is_file()
        or not cell_index_path.is_file()
        or not cells_root.is_dir()
        or not scenarios_root.is_dir()
    ):
        raise ValueError(
            "primary run lacks matrix_plan.json, execution_controls.json, cell_index.csv, "
            "cells/, or corruption_scenarios/"
        )
    saved_plan = _read_json_object(matrix_plan_path, "primary matrix plan")
    if saved_plan != plan.as_dict():
        raise ValueError("filesystem matrix_plan.json differs from the supplied frozen plan")
    execution_controls = _read_json_object(execution_controls_path, "primary execution controls")
    controls_binding_sha = execution_controls.get("binding_sha256")
    if not _valid_sha(controls_binding_sha):
        raise ValueError("execution_controls.json lacks a valid binding_sha256")

    with cell_index_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("cell_index.csv contains duplicate columns")
        rows = [dict(row) for row in reader]
    expected_by_id = {cell.cell_id: cell for cell in plan.cells}
    seen: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for row_index, row in enumerate(rows):
        cell_id = str(row.get("cell_id", ""))
        if not cell_id:
            raise ValueError(f"cell_index row {row_index} lacks cell_id")
        if cell_id in seen:
            duplicates.add(cell_id)
        seen[cell_id] = row
    if duplicates:
        raise ValueError(f"cell_index contains duplicate cells: {sorted(duplicates)}")
    expected_ids = set(expected_by_id)
    actual_ids = set(seen)
    missing_ids = sorted(expected_ids.difference(actual_ids))
    extra_ids = sorted(actual_ids.difference(expected_ids))
    if missing_ids or extra_ids:
        raise ValueError(f"cell_index exact set mismatch: missing={missing_ids}, extra={extra_ids}")
    filesystem_cell_ids = {path.name for path in cells_root.iterdir() if path.is_dir()}
    non_directories = sorted(path.name for path in cells_root.iterdir() if not path.is_dir())
    if filesystem_cell_ids != expected_ids or non_directories:
        raise ValueError(
            "cells/ exact set mismatch: "
            f"missing={sorted(expected_ids.difference(filesystem_cell_ids))}, "
            f"extra={sorted(filesystem_cell_ids.difference(expected_ids)) + non_directories}"
        )

    scenario_by_id = {scenario.scenario_id: scenario for scenario in plan.scenarios}
    expected_scenario_files = {f"{scenario_id}.json" for scenario_id in scenario_by_id}
    actual_scenario_files = {path.name for path in scenarios_root.iterdir() if path.is_file()}
    scenario_non_files = sorted(
        path.name for path in scenarios_root.iterdir() if not path.is_file()
    )
    if actual_scenario_files != expected_scenario_files or scenario_non_files:
        raise ValueError(
            "corruption_scenarios/ exact set mismatch: "
            f"missing={sorted(expected_scenario_files.difference(actual_scenario_files))}, "
            f"extra={sorted(actual_scenario_files.difference(expected_scenario_files)) + scenario_non_files}"
        )
    structural_outcomes: list[dict[str, Any]] = []
    manifest_hashes: list[tuple[str, str]] = []
    scenario_hashes: dict[str, str] = {}
    circularity_excluded_cell_ids: list[str] = []
    cell_readback_records: list[dict[str, Any]] = []
    for cell in plan.cells:
        row = seen[cell.cell_id]
        expected_identity = asdict(cell)
        actual_identity = _cell_identity_from_csv(row, f"cell_index[{cell.cell_id}]")
        if actual_identity != expected_identity:
            raise ValueError(f"cell {cell.cell_id} identity differs from the frozen plan")
        if row.get("execution_controls_binding_sha256") != controls_binding_sha:
            raise ValueError(
                f"cell {cell.cell_id} execution-controls binding differs from the root artifact"
            )
        scenario = scenario_by_id.get(cell.scenario_id)
        if scenario is None:
            raise ValueError(f"cell {cell.cell_id} references an unknown frozen scenario")
        status = str(row.get("status", ""))
        cell_directory = cells_root / cell.cell_id
        structural_outcome: dict[str, Any] = {
            **actual_identity,
            "status": status,
        }
        if status == "completed":
            manifest_sha, artifact_hashes = _validate_cell_artifact_manifest(
                cell_directory,
                expected_manifest_sha256=row.get("artifact_manifest_sha256"),
            )
            metrics_path = cell_directory / "metrics.json"
            if row.get("metrics_sha256") != artifact_hashes["metrics.json"]:
                raise ValueError(f"cell {cell.cell_id} metrics hash differs from cell_index")
            metrics = _read_json_object(metrics_path, f"cell {cell.cell_id} metrics")
            if metrics.get("cell") != expected_identity or metrics.get("scenario") != asdict(
                scenario
            ):
                raise ValueError(f"cell {cell.cell_id} metrics identity differs from frozen plan")
            corruption = _read_json_object(
                cell_directory / "corruption_manifest.json",
                f"cell {cell.cell_id} corruption manifest",
            )
            if (
                corruption.get("schema_version") != 1
                or corruption.get("cell") != expected_identity
                or corruption.get("scenario") != asdict(scenario)
            ):
                raise ValueError(
                    f"cell {cell.cell_id} corruption manifest identity differs from frozen plan"
                )
            shared_hash = corruption.get("shared_scenario_corruption_hash")
            if (
                not _valid_sha(shared_hash)
                or corruption.get("configuration_hash") != shared_hash
                or metrics.get("corruption_configuration_hash") != shared_hash
                or row.get("corruption_configuration_hash") != shared_hash
            ):
                raise ValueError(f"cell {cell.cell_id} corruption hash bindings are inconsistent")
            existing_scenario_hash = scenario_hashes.setdefault(cell.scenario_id, str(shared_hash))
            if existing_scenario_hash != shared_hash:
                raise ValueError(
                    f"scenario {cell.scenario_id} uses different shared corruption hashes"
                )
            cleanlab = _read_json_object(
                cell_directory / "cleanlab_evidence.json",
                f"cell {cell.cell_id} Cleanlab evidence",
            )
            _validate_cleanlab_metadata(cell.cell_id, cleanlab)
            independence = _read_json_object(
                cell_directory / "independence_evidence.json",
                f"cell {cell.cell_id} independence evidence",
            )
            expected_independence = {
                "mechanism": cell.mechanism,
                "representation_id": cell.representation_id,
                "status": row.get("independence_status"),
                "circularity_risk": _csv_boolean(
                    row.get("circularity_risk"), f"cell {cell.cell_id}.circularity_risk"
                ),
                "primary_confirmatory_eligible": _csv_boolean(
                    row.get("primary_confirmatory_eligible"),
                    f"cell {cell.cell_id}.primary_confirmatory_eligible",
                ),
                "matrix_artifact_sha256": (row.get("independence_matrix_artifact_sha256") or None),
                "evidence_sha256": row.get("independence_evidence_sha256") or None,
            }
            circularity_risk = bool(expected_independence["circularity_risk"])
            confirmatory_eligible = bool(expected_independence["primary_confirmatory_eligible"])
            if circularity_risk is confirmatory_eligible:
                raise ValueError(
                    f"cell {cell.cell_id} circularity and confirmatory eligibility are inconsistent"
                )
            independence_hash_values = (
                expected_independence["matrix_artifact_sha256"],
                expected_independence["evidence_sha256"],
            )
            if any(
                value is not None and not _valid_sha(value) for value in independence_hash_values
            ):
                raise ValueError(
                    f"cell {cell.cell_id} independence hashes are not valid SHA-256 values"
                )
            if (
                execution_controls.get("source") == "validated_frozen_primary_config_schema_v2"
                and cell.mechanism == "instance_dependent_corruption"
                and not all(_valid_sha(value) for value in independence_hash_values)
            ):
                raise ValueError(
                    f"cell {cell.cell_id} instance-dependent independence lacks frozen hashes"
                )
            if independence.get("schema_version") != 1 or any(
                independence.get(field) != expected
                for field, expected in expected_independence.items()
            ):
                raise ValueError(f"cell {cell.cell_id} independence bindings are inconsistent")
            if (
                corruption.get("independence_status") != expected_independence["status"]
                or corruption.get("circularity_risk")
                is not expected_independence["circularity_risk"]
            ):
                raise ValueError(f"cell {cell.cell_id} corruption/independence evidence disagrees")
            if (
                metrics.get("independence_status") != expected_independence["status"]
                or metrics.get("circularity_risk") is not circularity_risk
                or metrics.get("primary_confirmatory_eligible") is not confirmatory_eligible
                or metrics.get("independence_evidence_sha256")
                != expected_independence["evidence_sha256"]
                or metrics.get("independence_matrix_artifact_sha256")
                != expected_independence["matrix_artifact_sha256"]
            ):
                raise ValueError(
                    f"cell {cell.cell_id} metrics/independence eligibility evidence disagrees"
                )
            paired_metrics = metrics.get("paired_group_bootstrap")
            deferred_exact_comparisons = execution_controls.get(
                "source"
            ) == "validated_frozen_primary_config_schema_v2" and (
                execution_controls.get("comparison_execution_scope")
                == "deferred_exact_frozen_selectors"
                or metrics.get("comparison_execution_scope") == "deferred_exact_frozen_selectors"
            )
            if not isinstance(paired_metrics, Mapping) or (
                not paired_metrics and not deferred_exact_comparisons
            ):
                raise ValueError(
                    f"cell {cell.cell_id} lacks paired-statistic claim eligibility records"
                )
            expected_claim_status = (
                "excluded_circularity_risk" if circularity_risk else "primary_confirmatory_eligible"
            )
            for comparison_id, comparison in paired_metrics.items():
                if (
                    not isinstance(comparison, Mapping)
                    or comparison.get("claim_status") != expected_claim_status
                ):
                    raise ValueError(
                        f"cell {cell.cell_id} paired comparison {comparison_id!r} ignores "
                        "its circularity eligibility"
                    )
            if circularity_risk:
                circularity_excluded_cell_ids.append(cell.cell_id)
            structural_outcome.update(
                artifact_manifest_sha256=manifest_sha,
                metrics_sha256=artifact_hashes["metrics.json"],
            )
            manifest_hashes.append((cell.cell_id, manifest_sha))
            cell_readback_records.append(
                {
                    "cell_id": cell.cell_id,
                    "artifact_manifest_sha256": manifest_sha,
                    "artifacts": artifact_hashes,
                    "shared_scenario_corruption_hash": shared_hash,
                }
            )
        elif status == "skipped_with_frozen_blocker":
            if cell.required:
                raise ValueError(f"required cell {cell.cell_id} cannot be skipped")
            if (
                not _csv_boolean(
                    row.get("frozen_unavailability"),
                    f"cell {cell.cell_id}.frozen_unavailability",
                )
                or not str(row.get("blocker", "")).strip()
            ):
                raise ValueError(f"optional cell {cell.cell_id} lacks a frozen blocker")
            if any(cell_directory.iterdir()):
                raise ValueError(f"skipped cell {cell.cell_id} contains unmanifested artifacts")
            structural_outcome.update(
                frozen_unavailability=True,
                blocker=str(row["blocker"]),
            )
        elif status == "failed":
            structural_outcome["error"] = str(row.get("error", ""))
        structural_outcomes.append(structural_outcome)

    reconciliation = reconcile_primary_cell_outcomes(plan, structural_outcomes)
    if not reconciliation.passed:
        raise ValueError(
            "filesystem primary reconciliation failed: "
            f"{reconciliation.errors}; invalid={reconciliation.invalid_cell_ids}"
        )
    scenario_artifact_hashes: list[tuple[str, str]] = []
    for scenario_id, scenario in scenario_by_id.items():
        scenario_path = scenarios_root / f"{scenario_id}.json"
        scenario_payload = _read_json_object(
            scenario_path, f"primary corruption scenario {scenario_id}"
        )
        if scenario_payload.get("scenario") != asdict(scenario):
            raise ValueError(
                f"corruption scenario {scenario_id} identity differs from the frozen plan"
            )
        scenario_hash = scenario_hashes.get(scenario_id)
        if (
            scenario_hash is None
            or scenario_payload.get("shared_scenario_corruption_hash") != scenario_hash
        ):
            raise ValueError(
                f"corruption scenario {scenario_id} does not bind the shared cell hash"
            )
        scenario_artifact_hashes.append((scenario_id, sha256_file(scenario_path)))
    matrix_plan_sha = sha256_file(matrix_plan_path)
    execution_controls_sha = sha256_file(execution_controls_path)
    cell_index_sha = sha256_file(cell_index_path)
    readback_payload = {
        "matrix_plan_sha256": matrix_plan_sha,
        "execution_controls_sha256": execution_controls_sha,
        "execution_controls_binding_sha256": controls_binding_sha,
        "cell_index_sha256": cell_index_sha,
        "cells": cell_readback_records,
        "scenario_artifact_sha256": scenario_artifact_hashes,
        "scenario_corruption_sha256": sorted(scenario_hashes.items()),
        "circularity_excluded_cell_ids": circularity_excluded_cell_ids,
    }
    evidence = PrimaryFilesystemReadbackEvidence(
        run_directory=run_path,
        status="passed",
        matrix_plan_sha256=matrix_plan_sha,
        execution_controls_sha256=execution_controls_sha,
        execution_controls_binding_sha256=str(controls_binding_sha),
        cell_index_sha256=cell_index_sha,
        readback_root_sha256=_canonical_sha256(readback_payload),
        planned_cell_count=len(plan.cells),
        completed_cell_count=reconciliation.completed_cell_count,
        completed_required_cell_count=reconciliation.completed_required_cell_count,
        skipped_optional_cell_count=reconciliation.skipped_optional_cell_count,
        circularity_excluded_cell_ids=tuple(circularity_excluded_cell_ids),
        cell_artifact_manifest_sha256=tuple(manifest_hashes),
        scenario_artifact_sha256=tuple(scenario_artifact_hashes),
        scenario_corruption_sha256=tuple(sorted(scenario_hashes.items())),
        reconciliation=reconciliation,
    )
    object.__setattr__(evidence, "_attestation", _FILESYSTEM_READBACK_ATTESTATION)
    return evidence


def _strict_native_integer(values: np.ndarray[Any, Any], role: str) -> np.ndarray[Any, Any]:
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{role} must be a native integer array")
    return np.asarray(values, dtype=np.int64)


def _strict_native_float(values: np.ndarray[Any, Any], role: str) -> np.ndarray[Any, Any]:
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"{role} must be a native floating-point array")
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{role} contains non-finite values")
    return result


def _strict_native_bool(values: np.ndarray[Any, Any], role: str) -> np.ndarray[Any, Any]:
    if not np.issubdtype(values.dtype, np.bool_):
        raise ValueError(f"{role} must be a native boolean array")
    return np.asarray(values, dtype=np.bool_)


def _strict_string_vector(values: np.ndarray[Any, Any], role: str) -> np.ndarray[Any, Any]:
    if values.dtype.kind not in {"U", "S"} or values.ndim != 1:
        raise ValueError(f"{role} must be a pickle-free string vector")
    result = np.asarray(values, dtype=np.str_)
    if np.any(result == ""):
        raise ValueError(f"{role} contains an empty identifier")
    return result


def _valid_probability_matrix(values: np.ndarray[Any, Any], shape: tuple[int, ...]) -> bool:
    return bool(
        values.shape == shape
        and np.isfinite(values).all()
        and np.all(values >= 0.0)
        and np.all(values <= 1.0)
        and np.allclose(values.sum(axis=-1), 1.0, atol=1e-8)
    )


def _load_primary_crop_identity(
    run_directory: Path,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
]:
    bindings = _read_json_object(
        run_directory / "primary_input_bindings.json", "primary input bindings"
    )
    paths = bindings.get("cache_paths")
    expected = bindings.get("expected_hashes")
    verified = bindings.get("verified_hashes")
    if (
        not isinstance(paths, Mapping)
        or not isinstance(expected, Mapping)
        or not isinstance(verified, Mapping)
    ):
        raise ValueError("primary input bindings lack exact cache paths/hashes")
    raw_path = paths.get("crop_cache_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("primary input bindings lack crop_cache_path")
    crop_path = Path(raw_path).expanduser().resolve()
    if not crop_path.is_file():
        raise FileNotFoundError(f"primary crop cache is unavailable: {crop_path}")
    actual_sha = sha256_file(crop_path)
    if (
        expected.get("crop_cache_sha256") != actual_sha
        or verified.get("crop_cache_sha256") != actual_sha
    ):
        raise ValueError("primary crop cache differs from input bindings")
    try:
        with np.load(crop_path, allow_pickle=False) as archive:
            required = {
                "sample_ids",
                "group_ids",
                "pre_corruption_labels",
                "official_folds",
            }
            if not required.issubset(archive.files):
                raise ValueError("primary crop cache lacks restoration identity arrays")
            sample_ids = _strict_string_vector(np.asarray(archive["sample_ids"]), "crop sample IDs")
            group_ids = _strict_string_vector(np.asarray(archive["group_ids"]), "crop group IDs")
            labels = _strict_native_integer(
                np.asarray(archive["pre_corruption_labels"]), "crop labels"
            )
            folds = _strict_native_integer(
                np.asarray(archive["official_folds"]), "crop official folds"
            )
    except (OSError, KeyError, ValueError) as exc:
        if isinstance(exc, ValueError) and "crop" in str(exc):
            raise
        raise ValueError("primary crop cache is invalid or pickle-dependent") from exc
    if (
        sample_ids.ndim != 1
        or len(set(sample_ids.tolist())) != len(sample_ids)
        or group_ids.shape != sample_ids.shape
        or labels.shape != sample_ids.shape
        or folds.shape != sample_ids.shape
        or bindings.get("sample_order_sha256") != canonical_sha256(sample_ids.tolist())
    ):
        raise ValueError("primary crop identity/order differs from input bindings")
    return sample_ids, group_ids, labels, folds


def _metric_payload_matches(
    saved: Any,
    labels: np.ndarray[Any, Any],
    probabilities: np.ndarray[Any, Any],
    class_order: tuple[int, ...],
) -> bool:
    if not isinstance(saved, Mapping):
        return False
    expected = classification_metrics(labels, probabilities, class_order=class_order).as_dict()
    return canonical_sha256(saved) == canonical_sha256(expected)


def read_primary_restoration_evidence(
    run_directory: str | Path,
    controls: PrimaryExecutionControls,
) -> PrimaryRestorationReadbackEvidence:
    """Strictly recompute every frozen restoration selection, metric, and comparison."""

    from histo_audit.experiment.primary_core import PrimaryExecutionControls

    if not isinstance(controls, PrimaryExecutionControls):
        raise TypeError("primary restoration readback requires typed execution controls")
    if controls.frozen_config_schema_version != 2:
        raise ValueError("real primary restoration readback requires schema-v2 controls")
    controls.validate_for_plan(controls.plan)
    run_path = Path(run_directory).resolve()
    matrix_readback = read_primary_filesystem_evidence(controls.plan, run_path)
    if not matrix_readback.passed:
        raise ValueError("restoration readback requires passed primary matrix evidence")
    saved_controls = _read_json_object(
        run_path / "execution_controls.json", "primary execution controls"
    )
    if canonical_sha256(saved_controls) != canonical_sha256(controls.as_dict()):
        raise ValueError("restoration controls differ from the filesystem controls")
    index_path = run_path / "restoration_index.json"
    index = _read_json_object(index_path, "primary restoration index")
    expected_index_keys = {
        "schema_version",
        "execution_controls_binding_sha256",
        "restoration_cell_ids",
        "restoration_cell_count",
        "downstream_comparisons",
        "cells",
    }
    if set(index) != expected_index_keys:
        raise ValueError("restoration index schema has missing or extra fields")
    frozen_cell_ids = tuple(controls.restoration_cell_ids)
    if (
        index.get("schema_version") != 1
        or index.get("execution_controls_binding_sha256") != controls.binding_sha256
        or index.get("restoration_cell_ids") != list(frozen_cell_ids)
        or index.get("restoration_cell_count") != len(frozen_cell_ids)
        or index.get("downstream_comparisons")
        != [value.as_dict() for value in controls.restoration_downstream_comparisons]
    ):
        raise ValueError("restoration index differs from frozen controls")
    raw_rows = index.get("cells")
    if not isinstance(raw_rows, list):
        raise ValueError("restoration index lacks cell records")
    row_by_id = {
        str(row.get("cell", {}).get("cell_id", "")): row
        for row in raw_rows
        if isinstance(row, Mapping) and isinstance(row.get("cell"), Mapping)
    }
    if (
        len(row_by_id) != len(raw_rows)
        or set(row_by_id) != set(frozen_cell_ids)
        or tuple(str(row.get("cell", {}).get("cell_id")) for row in raw_rows) != frozen_cell_ids
    ):
        raise ValueError("restoration index cell set/order differs from frozen controls")
    restoration_root = run_path / "restorations"
    if not restoration_root.is_dir():
        raise FileNotFoundError("primary restorations directory is missing")
    actual_cell_entries = {path.name for path in restoration_root.iterdir() if path.is_dir()}
    extra_root_entries = sorted(
        path.name for path in restoration_root.iterdir() if not path.is_dir()
    )
    if actual_cell_entries != set(frozen_cell_ids) or extra_root_entries:
        raise ValueError("restorations directory has a missing or extra cell entry")
    crop_ids, crop_groups, crop_labels, crop_folds = _load_primary_crop_identity(run_path)
    from histo_audit.experiment.pannuke_primary_inputs import (
        select_stratified_reference_validation_groups,
    )

    development_members = np.isin(
        crop_folds, np.asarray(controls.development_official_folds, dtype=np.int64)
    )
    selected_reference_groups = select_stratified_reference_validation_groups(
        crop_labels[development_members],
        crop_groups[development_members],
        class_order=controls.class_order,
        fraction=controls.reference_validation_fraction_groups,
        seed=controls.split_seed,
    )
    reference_members = development_members & np.isin(
        crop_groups, np.asarray(selected_reference_groups, dtype=np.str_)
    )
    audit_members = development_members & ~reference_members
    expected_audit_ids = crop_ids[audit_members]
    expected_audit_groups = crop_groups[audit_members]
    expected_audit_labels = crop_labels[audit_members]
    final_members = crop_folds == controls.final_test_fold
    expected_final_ids = crop_ids[final_members]
    expected_final_groups = crop_groups[final_members]
    expected_final_labels = crop_labels[final_members]
    if not len(expected_final_ids):
        raise ValueError("frozen final-test fold is empty in the bound crop cache")
    if not len(expected_audit_ids) or not bool(reference_members.any()):
        raise ValueError("frozen audit/reference partition is empty in the bound crop cache")
    input_bindings = _read_json_object(
        run_path / "primary_input_bindings.json", "primary input bindings"
    )
    expected_partition_sha = canonical_sha256(
        {
            "assignment_label_source": "pre_corruption_label",
            "split_algorithm": "deterministic_group_greedy_class_distribution_v1",
            "split_seed": controls.split_seed,
            "audit_sample_ids": expected_audit_ids.tolist(),
            "reference_validation_sample_ids": crop_ids[reference_members].tolist(),
            "final_test_sample_ids": expected_final_ids.tolist(),
        }
    )
    if input_bindings.get("partition_assignment_sha256") != expected_partition_sha:
        raise ValueError("primary partition assignment differs from the bound crop cache")
    plan_by_id = {cell.cell_id: cell for cell in controls.plan.cells}
    comparison_definitions = tuple(controls.restoration_downstream_comparisons)
    comparison_ids = tuple(value.comparison_id for value in comparison_definitions)
    expected_array_keys = set(_RESTORATION_BASE_ARRAYS)
    for comparison_index in range(len(comparison_definitions)):
        prefix = f"downstream_comparison_{comparison_index:03d}"
        expected_array_keys.update(
            {
                f"{prefix}_metric_a",
                f"{prefix}_metric_b",
                f"{prefix}_differences",
            }
        )
    json_hashes: list[tuple[str, str]] = []
    evidence_hashes: list[tuple[str, str]] = []
    manifest_hashes: list[tuple[str, str]] = []
    semantic_records: list[dict[str, Any]] = []
    for cell_id in frozen_cell_ids:
        planned = plan_by_id.get(cell_id)
        if planned is None:
            raise ValueError(f"restoration cell {cell_id} is absent from the frozen plan")
        row = row_by_id[cell_id]
        expected_row_keys = {
            "schema_version",
            "cell",
            "ranking_method",
            "json_path",
            "json_sha256",
            "evidence_path",
            "evidence_sha256",
            "manifest_path",
            "manifest_sha256",
        }
        expected_directory = restoration_root / cell_id
        expected_paths = {
            "json_path": f"restorations/{cell_id}/restoration.json",
            "evidence_path": f"restorations/{cell_id}/restoration_evidence.npz",
            "manifest_path": f"restorations/{cell_id}/restoration_manifest.json",
        }
        if (
            set(row) != expected_row_keys
            or row.get("schema_version") != 1
            or row.get("cell") != asdict(planned)
            or row.get("ranking_method") != controls.restoration_ranking_method
            or any(row.get(field) != value for field, value in expected_paths.items())
        ):
            raise ValueError(f"restoration index identity/path mismatch for {cell_id}")
        actual_files = {path.name for path in expected_directory.iterdir() if path.is_file()}
        non_files = sorted(path.name for path in expected_directory.iterdir() if not path.is_file())
        if (
            actual_files
            != {
                "restoration.json",
                "restoration_evidence.npz",
                "restoration_manifest.json",
            }
            or non_files
        ):
            raise ValueError(f"restoration artifact set mismatch for {cell_id}")
        json_path = expected_directory / "restoration.json"
        evidence_path = expected_directory / "restoration_evidence.npz"
        manifest_path = expected_directory / "restoration_manifest.json"
        actual_hashes = {
            "json_sha256": sha256_file(json_path),
            "evidence_sha256": sha256_file(evidence_path),
            "manifest_sha256": sha256_file(manifest_path),
        }
        if any(row.get(field) != value for field, value in actual_hashes.items()):
            raise ValueError(f"restoration index SHA mismatch for {cell_id}")
        manifest = _read_json_object(manifest_path, f"{cell_id} restoration manifest")
        if set(manifest) != {
            "schema_version",
            "cell_id",
            "execution_controls_binding_sha256",
            "artifacts",
        } or (
            manifest.get("schema_version") != 1
            or manifest.get("cell_id") != cell_id
            or manifest.get("execution_controls_binding_sha256") != controls.binding_sha256
        ):
            raise ValueError(f"restoration manifest identity differs for {cell_id}")
        manifest_artifacts = manifest.get("artifacts")
        if not isinstance(manifest_artifacts, list) or manifest_artifacts != [
            {"path": "restoration.json", "sha256": actual_hashes["json_sha256"]},
            {
                "path": "restoration_evidence.npz",
                "sha256": actual_hashes["evidence_sha256"],
            },
        ]:
            raise ValueError(f"restoration manifest artifact hashes differ for {cell_id}")
        restoration = _read_json_object(json_path, f"{cell_id} restoration results")
        if set(restoration) != {
            "schema_version",
            "cell",
            "selected_cell_id",
            "execution_controls_binding_sha256",
            "shared_scenario_corruption_hash",
            "ranking_method",
            "review_budget",
            "required_experiments",
            "downstream_comparisons",
            "evaluation",
        } or (
            restoration.get("schema_version") != 1
            or restoration.get("cell") != asdict(planned)
            or restoration.get("selected_cell_id") != cell_id
            or restoration.get("execution_controls_binding_sha256") != controls.binding_sha256
            or restoration.get("ranking_method") != controls.restoration_ranking_method
            or restoration.get("review_budget") != controls.restoration_review_budget
            or restoration.get("required_experiments")
            != list(controls.restoration_required_experiments)
            or set(restoration.get("required_experiments", ())) != _RESTORATION_CONDITIONS
        ):
            raise ValueError(f"restoration JSON differs from frozen controls for {cell_id}")
        corruption = _read_json_object(
            run_path / "cells" / cell_id / "corruption_manifest.json",
            f"{cell_id} corruption manifest",
        )
        if restoration.get("shared_scenario_corruption_hash") != corruption.get(
            "shared_scenario_corruption_hash"
        ):
            raise ValueError(f"restoration corruption binding differs for {cell_id}")
        evaluation = restoration.get("evaluation")
        if not isinstance(evaluation, Mapping) or set(evaluation) != {
            "uncorrupted_reference_baseline",
            "corrupted_observed_baseline",
            "random_review_restoration",
            "audit_guided_restoration",
            "review_budget_fraction",
            "review_budget_count",
            "partition_evidence",
            "model_evidence",
        }:
            raise ValueError(f"restoration evaluation schema differs for {cell_id}")
        try:
            with np.load(evidence_path, allow_pickle=False) as archive:
                if set(archive.files) != expected_array_keys:
                    raise ValueError(f"restoration NPZ array set differs for {cell_id}")

                def array(name: str) -> np.ndarray[Any, Any]:
                    return np.asarray(archive[name])

                class_order = _strict_native_integer(array("class_order"), "class order")
                audit_ids = _strict_string_vector(array("audit_sample_ids"), "audit sample IDs")
                audit_groups = _strict_string_vector(array("audit_group_ids"), "audit groups")
                pre = _strict_native_integer(
                    array("audit_pre_corruption_labels"), "audit pre-corruption labels"
                )
                observed = _strict_native_integer(
                    array("audit_observed_labels"), "audit observed labels"
                )
                injected = _strict_native_bool(
                    array("audit_is_injected_corruption"), "audit corruption flags"
                )
                risks = _strict_native_float(array("audit_risk_scores"), "audit risks")
                final_ids = _strict_string_vector(
                    array("final_test_sample_ids"), "final-test sample IDs"
                )
                final_groups = _strict_string_vector(
                    array("final_test_group_ids"), "final-test groups"
                )
                final_labels = _strict_native_integer(
                    array("final_test_labels"), "final-test labels"
                )
                if not np.array_equal(class_order, np.asarray(controls.class_order)):
                    raise ValueError(f"restoration class order differs for {cell_id}")
                with np.load(
                    run_path / "cells" / cell_id / "oof_predictions.npz", allow_pickle=False
                ) as oof:
                    if (
                        not np.array_equal(audit_ids, np.asarray(oof["sample_ids"], dtype=np.str_))
                        or not np.array_equal(
                            audit_groups, np.asarray(oof["group_ids"], dtype=np.str_)
                        )
                        or not np.array_equal(
                            pre, np.asarray(oof["pre_corruption_label"], dtype=np.int64)
                        )
                        or not np.array_equal(
                            observed, np.asarray(oof["observed_label"], dtype=np.int64)
                        )
                        or not np.array_equal(
                            injected,
                            np.asarray(oof["is_injected_corruption"], dtype=np.bool_),
                        )
                    ):
                        raise ValueError(f"restoration audit identity differs for {cell_id}")
                with np.load(
                    run_path / "cells" / cell_id / "risk_scores.npz", allow_pickle=False
                ) as risk_archive:
                    expected_risk = np.asarray(
                        risk_archive[controls.restoration_ranking_method], dtype=np.float64
                    )
                if not np.array_equal(risks, expected_risk):
                    raise ValueError(f"restoration ranking scores differ for {cell_id}")
                if (
                    not np.array_equal(audit_ids, expected_audit_ids)
                    or not np.array_equal(audit_groups, expected_audit_groups)
                    or not np.array_equal(pre, expected_audit_labels)
                    or not np.array_equal(final_ids, expected_final_ids)
                    or not np.array_equal(final_groups, expected_final_groups)
                    or not np.array_equal(final_labels, expected_final_labels)
                    or set(audit_ids.tolist()).intersection(final_ids.tolist())
                    or set(audit_groups.tolist()).intersection(final_groups.tolist())
                    or any(int(value) not in controls.class_order for value in final_labels)
                ):
                    raise ValueError(f"restoration final partition differs for {cell_id}")
                n_audit = len(audit_ids)
                n_final = len(final_ids)
                budget = budget_count(n_audit, controls.restoration_review_budget)
                expected_guided_indices = rank_indices(risks, tie_break_ids=audit_ids.tolist())[
                    :budget
                ]
                guided_indices = _strict_native_integer(
                    array("audit_reviewed_indices"), "guided reviewed indices"
                )
                guided_reviewed = _strict_native_bool(
                    array("guided_reviewed_mask"), "guided reviewed mask"
                )
                guided_restored = _strict_native_bool(
                    array("guided_restored_mask"), "guided restored mask"
                )
                guided_labels = _strict_native_integer(
                    array("guided_restored_labels"), "guided restored labels"
                )
                expected_guided_mask = np.zeros(n_audit, dtype=np.bool_)
                expected_guided_mask[expected_guided_indices] = True
                expected_guided = restore_reviewed_labels(
                    pre, observed, injected, expected_guided_mask
                )
                if (
                    not np.array_equal(guided_indices, expected_guided_indices)
                    or not np.array_equal(guided_reviewed, expected_guided_mask)
                    or not np.array_equal(guided_restored, expected_guided.restored_mask)
                    or not np.array_equal(guided_labels, expected_guided.restored_labels)
                ):
                    raise ValueError(f"guided restoration semantics differ for {cell_id}")
                repeats = controls.restoration_random_repeats
                random_indices = _strict_native_integer(
                    array("random_reviewed_indices"), "random reviewed indices"
                )
                random_reviewed = _strict_native_bool(
                    array("random_reviewed_masks"), "random reviewed masks"
                )
                random_restored = _strict_native_bool(
                    array("random_restored_masks"), "random restored masks"
                )
                random_labels = _strict_native_integer(
                    array("random_restored_labels"), "random restored labels"
                )
                random_seeds = _strict_native_integer(
                    array("random_review_seeds"), "random review seeds"
                )
                if (
                    random_indices.shape != (repeats, budget)
                    or random_reviewed.shape != (repeats, n_audit)
                    or random_restored.shape != random_reviewed.shape
                    or random_labels.shape != random_reviewed.shape
                    or not np.array_equal(
                        random_seeds,
                        np.arange(
                            controls.restoration_random_seed,
                            controls.restoration_random_seed + repeats,
                            dtype=np.int64,
                        ),
                    )
                ):
                    raise ValueError(f"random restoration shapes/seeds differ for {cell_id}")
                for repeat, repeat_seed in enumerate(random_seeds.tolist()):
                    expected_indices = np.sort(
                        np.random.default_rng(int(repeat_seed)).choice(
                            n_audit, size=budget, replace=False
                        )
                    ).astype(np.int64)
                    expected_mask = np.zeros(n_audit, dtype=np.bool_)
                    expected_mask[expected_indices] = True
                    expected_restoration = restore_reviewed_labels(
                        pre, observed, injected, expected_mask
                    )
                    if (
                        not np.array_equal(random_indices[repeat], expected_indices)
                        or not np.array_equal(random_reviewed[repeat], expected_mask)
                        or not np.array_equal(
                            random_restored[repeat], expected_restoration.restored_mask
                        )
                        or not np.array_equal(
                            random_labels[repeat], expected_restoration.restored_labels
                        )
                    ):
                        raise ValueError(f"random restoration semantics differ for {cell_id}")
                probability_names = (
                    "uncorrupted_final_probabilities",
                    "corrupted_final_probabilities",
                    "guided_final_probabilities",
                )
                probabilities = {
                    name: _strict_native_float(array(name), name) for name in probability_names
                }
                random_probabilities = _strict_native_float(
                    array("random_final_probabilities"), "random final probabilities"
                )
                if not all(
                    _valid_probability_matrix(value, (n_final, len(controls.class_order)))
                    for value in probabilities.values()
                ) or not _valid_probability_matrix(
                    random_probabilities,
                    (repeats, n_final, len(controls.class_order)),
                ):
                    raise ValueError(f"restoration probability matrices differ for {cell_id}")
                predicted_pairs = (
                    ("uncorrupted_final_probabilities", "uncorrupted_final_predicted_class"),
                    ("corrupted_final_probabilities", "corrupted_final_predicted_class"),
                    ("guided_final_probabilities", "guided_final_predicted_class"),
                )
                class_values = np.asarray(controls.class_order, dtype=np.int64)
                for probability_name, predicted_name in predicted_pairs:
                    predicted = _strict_native_integer(array(predicted_name), predicted_name)
                    expected_predicted = class_values[
                        np.argmax(probabilities[probability_name], axis=1)
                    ]
                    if not np.array_equal(predicted, expected_predicted):
                        raise ValueError(f"restoration predicted classes differ for {cell_id}")
                random_predicted = _strict_native_integer(
                    array("random_final_predicted_class"), "random predicted classes"
                )
                expected_random_predicted = class_values[np.argmax(random_probabilities, axis=2)]
                if not np.array_equal(random_predicted, expected_random_predicted):
                    raise ValueError(f"random predicted classes differ for {cell_id}")
                condition_map = {
                    "uncorrupted_reference_baseline": probabilities[
                        "uncorrupted_final_probabilities"
                    ],
                    "corrupted_observed_baseline": probabilities["corrupted_final_probabilities"],
                    "audit_guided_restoration": probabilities["guided_final_probabilities"],
                }
                for condition, condition_probabilities in condition_map.items():
                    saved = evaluation.get(condition)
                    if not isinstance(saved, Mapping) or not _metric_payload_matches(
                        saved.get("metrics"),
                        final_labels,
                        condition_probabilities,
                        controls.class_order,
                    ):
                        raise ValueError(f"restoration {condition} metrics differ for {cell_id}")
                random_metrics = [
                    classification_metrics(
                        final_labels,
                        random_probabilities[repeat],
                        class_order=controls.class_order,
                    ).as_dict()
                    for repeat in range(repeats)
                ]
                random_saved = evaluation.get("random_review_restoration")
                if not isinstance(random_saved, Mapping) or not isinstance(
                    random_saved.get("runs"), list
                ):
                    raise ValueError(f"random restoration results are missing for {cell_id}")
                saved_runs = random_saved["runs"]
                if len(saved_runs) != repeats:
                    raise ValueError(f"random restoration repeat count differs for {cell_id}")
                for repeat, (saved, expected_metrics) in enumerate(
                    zip(saved_runs, random_metrics, strict=True)
                ):
                    if (
                        not isinstance(saved, Mapping)
                        or saved.get("review_seed") != int(random_seeds[repeat])
                        or saved.get("reviewed_count") != budget
                        or saved.get("restored_count") != int(random_restored[repeat].sum())
                        or canonical_sha256(saved.get("metrics"))
                        != canonical_sha256(expected_metrics)
                    ):
                        raise ValueError(f"random restoration metrics differ for {cell_id}")
                random_f1 = np.asarray(
                    [float(value["macro_f1"]) for value in random_metrics],
                    dtype=np.float64,
                )
                if (
                    random_saved.get("macro_f1_mean") != float(random_f1.mean())
                    or random_saved.get("macro_f1_interval_95")
                    != [
                        float(np.quantile(random_f1, 0.025)),
                        float(np.quantile(random_f1, 0.975)),
                    ]
                    or evaluation.get("review_budget_fraction")
                    != controls.restoration_review_budget
                    or evaluation.get("review_budget_count") != budget
                ):
                    raise ValueError(f"restoration aggregate metrics differ for {cell_id}")
                expected_counts = {
                    "uncorrupted_reference_baseline": (0, 0),
                    "corrupted_observed_baseline": (0, 0),
                    "audit_guided_restoration": (budget, int(guided_restored.sum())),
                }
                for condition, (reviewed_count, restored_count) in expected_counts.items():
                    saved = evaluation[condition]
                    if (
                        saved.get("reviewed_count") != reviewed_count
                        or saved.get("restored_count") != restored_count
                    ):
                        raise ValueError(f"restoration counts differ for {cell_id}/{condition}")
                partition = evaluation.get("partition_evidence")
                model = evaluation.get("model_evidence")
                expected_model_seed = (
                    controls.logistic_model_seed
                    if planned.classifier_id == "multinomial_logistic_regression"
                    else controls.mlp_config.seed
                )
                if (
                    not isinstance(partition, Mapping)
                    or partition.get("development_groups") != sorted(set(audit_groups.tolist()))
                    or partition.get("final_reference_groups") != sorted(set(final_groups.tolist()))
                    or partition.get("group_overlap_count") != 0
                    or partition.get("final_test_uncorrupted_verified") is not True
                    or not isinstance(model, Mapping)
                    or model.get("model_seed") != expected_model_seed
                    or model.get("same_factory_configuration_all_conditions") is not True
                ):
                    raise ValueError(f"restoration partition/model evidence differs for {cell_id}")
                saved_comparisons = restoration.get("downstream_comparisons")
                saved_comparison_ids = _strict_string_vector(
                    array("downstream_comparison_ids"), "downstream comparison IDs"
                )
                if (
                    not isinstance(saved_comparisons, list)
                    or not np.array_equal(
                        saved_comparison_ids, np.asarray(comparison_ids, dtype=np.str_)
                    )
                    or len(saved_comparisons) != len(comparison_definitions)
                ):
                    raise ValueError(f"downstream comparison set differs for {cell_id}")
                guided_macro_f1 = float(
                    classification_metrics(
                        final_labels,
                        probabilities["guided_final_probabilities"],
                        class_order=controls.class_order,
                    ).macro_f1
                )
                for comparison_index, (definition, saved) in enumerate(
                    zip(comparison_definitions, saved_comparisons, strict=True)
                ):
                    prefix = f"downstream_comparison_{comparison_index:03d}"
                    metric_a = _strict_native_float(array(f"{prefix}_metric_a"), "metric A")
                    metric_b = _strict_native_float(array(f"{prefix}_metric_b"), "metric B")
                    differences = _strict_native_float(
                        array(f"{prefix}_differences"), "metric differences"
                    )
                    expected_a = np.full(repeats, guided_macro_f1, dtype=np.float64)
                    if (
                        not np.array_equal(metric_a, expected_a)
                        or not np.array_equal(metric_b, random_f1)
                        or not np.array_equal(differences, expected_a - random_f1)
                    ):
                        raise ValueError(f"downstream comparison arrays differ for {cell_id}")
                    expected_comparison = {
                        **definition.as_dict(),
                        "status": "reported",
                        "pairing": (
                            "same_final_reference_set_across_frozen_random_review_repetitions"
                        ),
                        "random_repetitions": repeats,
                        "point_metric_a": guided_macro_f1,
                        "point_metric_b": float(random_f1.mean()),
                        "point_difference": float(guided_macro_f1 - random_f1.mean()),
                        "mean_difference": float((expected_a - random_f1).mean()),
                        "interval_95": [
                            float(np.quantile(expected_a - random_f1, 0.025)),
                            float(np.quantile(expected_a - random_f1, 0.975)),
                        ],
                        "probability_positive": float(
                            np.mean(expected_a - random_f1 > 0.0)
                            + 0.5 * np.mean(expected_a - random_f1 == 0.0)
                        ),
                    }
                    if canonical_sha256(saved) != canonical_sha256(expected_comparison):
                        raise ValueError(f"downstream comparison JSON differs for {cell_id}")
        except (OSError, KeyError, ValueError) as exc:
            if isinstance(exc, ValueError) and (
                "restoration" in str(exc)
                or "downstream" in str(exc)
                or "random" in str(exc)
                or "guided" in str(exc)
            ):
                raise
            raise ValueError(
                f"restoration evidence is invalid or pickle-dependent for {cell_id}"
            ) from exc
        json_hashes.append((cell_id, actual_hashes["json_sha256"]))
        evidence_hashes.append((cell_id, actual_hashes["evidence_sha256"]))
        manifest_hashes.append((cell_id, actual_hashes["manifest_sha256"]))
        semantic_records.append(
            {
                "cell_id": cell_id,
                "json_sha256": actual_hashes["json_sha256"],
                "evidence_sha256": actual_hashes["evidence_sha256"],
                "manifest_sha256": actual_hashes["manifest_sha256"],
            }
        )
    index_sha = sha256_file(index_path)
    readback_root = canonical_sha256(
        {
            "restoration_index_sha256": index_sha,
            "execution_controls_binding_sha256": controls.binding_sha256,
            "cells": semantic_records,
            "downstream_comparison_ids": comparison_ids,
        }
    )
    evidence = PrimaryRestorationReadbackEvidence(
        run_directory=run_path,
        status="passed",
        restoration_index_sha256=index_sha,
        readback_root_sha256=readback_root,
        source_readback_root_sha256=matrix_readback.readback_root_sha256,
        restoration_cell_count=len(frozen_cell_ids),
        downstream_comparison_count=len(comparison_definitions),
        cell_json_sha256=tuple(json_hashes),
        cell_evidence_sha256=tuple(evidence_hashes),
        cell_manifest_sha256=tuple(manifest_hashes),
    )
    object.__setattr__(evidence, "_attestation", _RESTORATION_READBACK_ATTESTATION)
    return evidence


def build_primary_completion_evidence(
    *,
    plan: PrimaryMatrixPlan,
    reconciliation: PrimaryMatrixReconciliation,
    artifact_scope: str,
    study_outcome_eligible: bool,
    gate_evidence: PrimaryExecutionGateEvidence | None = None,
    filesystem_readback: PrimaryFilesystemReadbackEvidence | None = None,
    statistics_verification: (
        PrimaryStatisticsVerification | InheritedPrimaryStatisticsVerification | None
    ) = None,
    restoration_readback: PrimaryRestorationReadbackEvidence | None = None,
    _inherited_authorization: (
        AuthorizedPriorNumericVerificationProof | AuthorizedOrphanNumericVerificationProof | None
    ) = None,
) -> dict[str, Any]:
    """Build stage evidence, never allowing synthetic fixtures to enable completion."""

    if artifact_scope not in {REAL_PRIMARY_ARTIFACT_SCOPE, SYNTHETIC_PRIMARY_ARTIFACT_SCOPE}:
        raise ValueError(f"unsupported primary artifact scope: {artifact_scope!r}")
    if artifact_scope == SYNTHETIC_PRIMARY_ARTIFACT_SCOPE and study_outcome_eligible:
        raise ValueError("synthetic primary fixtures can never be study-outcome eligible")
    if study_outcome_eligible and artifact_scope != REAL_PRIMARY_ARTIFACT_SCOPE:
        raise ValueError("only the real PanNuke primary scope can be outcome eligible")
    completion_stage: str | None = None
    bindings: dict[str, str] = {}
    filesystem_bindings: dict[str, Any] = {}
    if study_outcome_eligible:
        from histo_audit.experiment.primary_statistics import (
            AuthorizedOrphanNumericVerificationProof,
            AuthorizedPriorNumericVerificationProof,
            InheritedPrimaryStatisticsVerification,
            PrimaryStatisticsVerification,
        )
        from histo_audit.workflows.study_gates import PrimaryExecutionGateEvidence

        if not reconciliation.passed:
            raise ValueError(
                "PRIMARY_STUDY_COMPLETE requires passed matrix reconciliation: "
                f"{reconciliation.errors}"
            )
        if not isinstance(gate_evidence, PrimaryExecutionGateEvidence):
            raise TypeError(
                "eligible primary completion requires a real PrimaryExecutionGateEvidence "
                "instance; mappings and SHA-shaped strings are not execution evidence"
            )
        if filesystem_readback is None or not filesystem_readback.passed:
            raise ValueError(
                "eligible primary completion requires passed attested filesystem readback"
            )
        current_readback = read_primary_filesystem_evidence(plan, filesystem_readback.run_directory)
        if (
            current_readback.readback_root_sha256 != filesystem_readback.readback_root_sha256
            or current_readback.matrix_plan_sha256 != filesystem_readback.matrix_plan_sha256
            or current_readback.cell_index_sha256 != filesystem_readback.cell_index_sha256
        ):
            raise ValueError("primary filesystem evidence changed after its attested readback")
        if current_readback.reconciliation.as_dict() != reconciliation.as_dict():
            raise ValueError(
                "supplied primary reconciliation differs from filesystem-backed outcomes"
            )
        inherited_statistics = (
            statistics_verification
            if isinstance(
                statistics_verification,
                InheritedPrimaryStatisticsVerification,
            )
            else None
        )
        inherited_authorized = (
            inherited_statistics is not None
            and isinstance(
                _inherited_authorization,
                (
                    AuthorizedPriorNumericVerificationProof,
                    AuthorizedOrphanNumericVerificationProof,
                ),
            )
            and _inherited_authorization.valid
            and inherited_statistics.authorization_kind
            == _inherited_authorization.authorization_kind
            and inherited_statistics.prior_numeric_verification_proof_sha256
            == _inherited_authorization.prior_numeric_verification_proof_sha256
            and inherited_statistics.source_readback_root_sha256
            == _inherited_authorization.source_readback_root_sha256
            and inherited_statistics.comparison_count == _inherited_authorization.comparison_count
            and {
                "primary_statistics.json": inherited_statistics.statistics_sha256,
                "primary_bootstrap_evidence.npz": (inherited_statistics.bootstrap_evidence_sha256),
                "primary_subgroups.csv": inherited_statistics.subgroups_sha256,
                "primary_statistics_manifest.json": inherited_statistics.manifest_sha256,
            }
            == {
                name: digest
                for name, _size_bytes, digest in _inherited_authorization.statistics_quartet
            }
        )
        fresh_statistics = (
            isinstance(statistics_verification, PrimaryStatisticsVerification)
            and statistics_verification.valid
            and _inherited_authorization is None
        )
        if not (fresh_statistics or inherited_authorized):
            raise ValueError(
                "eligible primary completion requires passed attested primary statistics "
                "verification"
            )
        verified_statistics: PrimaryStatisticsVerification | InheritedPrimaryStatisticsVerification
        if fresh_statistics:
            assert isinstance(statistics_verification, PrimaryStatisticsVerification)
            verified_statistics = statistics_verification
        else:
            assert inherited_authorized
            assert isinstance(
                statistics_verification,
                InheritedPrimaryStatisticsVerification,
            )
            verified_statistics = statistics_verification
        if (
            verified_statistics.output_directory.resolve() != current_readback.run_directory
            or verified_statistics.source_readback_root_sha256
            != current_readback.readback_root_sha256
        ):
            raise ValueError(
                "primary statistics verification is bound to a different matrix readback"
            )
        statistics_hash_bindings = {
            "primary_statistics_sha256": (
                "primary_statistics.json",
                verified_statistics.statistics_sha256,
            ),
            "primary_bootstrap_evidence_sha256": (
                "primary_bootstrap_evidence.npz",
                verified_statistics.bootstrap_evidence_sha256,
            ),
            "primary_subgroups_sha256": (
                "primary_subgroups.csv",
                verified_statistics.subgroups_sha256,
            ),
            "primary_statistics_manifest_sha256": (
                "primary_statistics_manifest.json",
                verified_statistics.manifest_sha256,
            ),
        }
        for binding_name, (artifact_name, verified_sha) in statistics_hash_bindings.items():
            artifact_path = current_readback.run_directory / artifact_name
            if not artifact_path.is_file() or sha256_file(artifact_path) != verified_sha:
                raise ValueError(
                    f"primary statistics artifact changed after strict verification: "
                    f"{artifact_name}"
                )
            filesystem_bindings[binding_name] = verified_sha
        filesystem_bindings.update(
            primary_statistics_verification_status="passed",
            primary_statistics_comparison_count=verified_statistics.comparison_count,
            primary_statistics_source_readback_root_sha256=(
                verified_statistics.source_readback_root_sha256
            ),
        )
        if (
            not isinstance(restoration_readback, PrimaryRestorationReadbackEvidence)
            or not restoration_readback.passed
        ):
            raise ValueError(
                "eligible primary completion requires passed attested restoration readback"
            )
        if (
            restoration_readback.run_directory.resolve() != current_readback.run_directory
            or restoration_readback.source_readback_root_sha256
            != current_readback.readback_root_sha256
        ):
            raise ValueError("primary restoration readback is bound to a different matrix readback")
        restoration_index_path = current_readback.run_directory / "restoration_index.json"
        if (
            not restoration_index_path.is_file()
            or sha256_file(restoration_index_path) != restoration_readback.restoration_index_sha256
        ):
            raise ValueError(
                "primary restoration artifact changed after strict verification: "
                "restoration_index.json"
            )
        restoration_hash_groups = (
            ("json", "restoration.json", restoration_readback.cell_json_sha256),
            (
                "evidence",
                "restoration_evidence.npz",
                restoration_readback.cell_evidence_sha256,
            ),
            (
                "manifest",
                "restoration_manifest.json",
                restoration_readback.cell_manifest_sha256,
            ),
        )
        expected_restoration_ids = {cell_id for cell_id, _ in restoration_readback.cell_json_sha256}
        if len(expected_restoration_ids) != restoration_readback.restoration_cell_count or any(
            {cell_id for cell_id, _ in hashes} != expected_restoration_ids
            or len(hashes) != len(expected_restoration_ids)
            for _, _, hashes in restoration_hash_groups
        ):
            raise ValueError("primary restoration attestation has inconsistent cell hashes")
        restoration_hash_bindings: dict[str, str] = {}
        for role, filename, hashes in restoration_hash_groups:
            for cell_id, verified_sha in hashes:
                artifact_path = current_readback.run_directory / "restorations" / cell_id / filename
                if not artifact_path.is_file() or sha256_file(artifact_path) != verified_sha:
                    raise ValueError(
                        "primary restoration artifact changed after strict verification: "
                        f"{cell_id}/{filename}"
                    )
                restoration_hash_bindings[f"primary_restoration_{role}_sha256::{cell_id}"] = (
                    verified_sha
                )
        filesystem_bindings.update(
            restoration_hash_bindings,
            primary_restoration_verification_status="passed",
            primary_restoration_index_sha256=(restoration_readback.restoration_index_sha256),
            primary_restoration_readback_root_sha256=(restoration_readback.readback_root_sha256),
            primary_restoration_source_readback_root_sha256=(
                restoration_readback.source_readback_root_sha256
            ),
            primary_restoration_cell_count=restoration_readback.restoration_cell_count,
            primary_restoration_downstream_comparison_count=(
                restoration_readback.downstream_comparison_count
            ),
        )
        for field in _REQUIRED_GATE_HASHES:
            value = getattr(gate_evidence, field, None)
            if not _valid_sha(value):
                raise ValueError(f"primary gate evidence {field} is not a SHA-256")
            bindings[field] = str(value)
        filesystem_bindings = {
            **filesystem_bindings,
            "filesystem_run_directory": str(current_readback.run_directory),
            "filesystem_matrix_plan_sha256": current_readback.matrix_plan_sha256,
            "filesystem_execution_controls_sha256": (current_readback.execution_controls_sha256),
            "filesystem_execution_controls_binding_sha256": (
                current_readback.execution_controls_binding_sha256
            ),
            "filesystem_cell_index_sha256": current_readback.cell_index_sha256,
            "filesystem_readback_root_sha256": current_readback.readback_root_sha256,
            "filesystem_completed_cell_count": current_readback.completed_cell_count,
            "circularity_excluded_cell_count": len(current_readback.circularity_excluded_cell_ids),
            "circularity_excluded_cell_ids": list(current_readback.circularity_excluded_cell_ids),
            "primary_confirmatory_claims_require_exclusion_of_these_cells": bool(
                current_readback.circularity_excluded_cell_ids
            ),
            "filesystem_scenario_corruption_sha256": dict(
                current_readback.scenario_corruption_sha256
            ),
        }
        completion_stage = "PRIMARY_STUDY_COMPLETE"
    return {
        "schema_version": 2,
        "completion_stage": completion_stage,
        "study_outcome_eligible": study_outcome_eligible,
        "artifact_scope": artifact_scope,
        "matrix_config_sha256": plan.config_sha256,
        "planned_cell_count": len(plan.cells),
        "required_cell_count": plan.required_cell_count,
        "completed_required_cell_count": reconciliation.completed_required_cell_count,
        "skipped_optional_cell_count": reconciliation.skipped_optional_cell_count,
        "failed_required_cell_count": (
            plan.required_cell_count - reconciliation.completed_required_cell_count
        ),
        "reconciliation_status": reconciliation.status,
        "completion_stage_enabled_only_after_run_seal_and_integrity_verification": (
            study_outcome_eligible
        ),
        **bindings,
        **filesystem_bindings,
    }


__all__ = [
    "REAL_PRIMARY_ARTIFACT_SCOPE",
    "SYNTHETIC_PRIMARY_ARTIFACT_SCOPE",
    "PrimaryFilesystemReadbackEvidence",
    "PrimaryMatrixReconciliation",
    "PrimaryRestorationReadbackEvidence",
    "build_primary_completion_evidence",
    "read_primary_filesystem_evidence",
    "read_primary_restoration_evidence",
    "reconcile_primary_cell_outcomes",
]
