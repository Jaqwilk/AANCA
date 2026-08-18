"""Fail-closed reconciliation and completion semantics for confirmatory studies."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from histo_audit.auditing.ensemble import (
    ensemble_disagreement,
    predeclared_ensemble_risk,
)
from histo_audit.auditing.scores import (
    fixed_hybrid_drop_one_ablations,
    score_annotations,
)
from histo_audit.config import load_config
from histo_audit.corruption.controlled import array_artifact_sha256
from histo_audit.cross_validation.image_oof import (
    ConfirmatoryCheckpointContractError,
    ConfirmatoryCheckpointDirective,
    ConfirmatoryCheckpointFileIdentity,
    _checkpoint_file_identity_from_exact_manifest,
    _checkpoint_physical_identity_from_exact_manifest,
    _hold_private_checkpoint_snapshot,
    _require_exact_checkpoint_execution_manifest_payload,
)
from histo_audit.evaluation.restoration import classification_metrics, restore_reviewed_labels
from histo_audit.experiment.study_contracts import (
    CLASS_ORDER,
    RESOURCE_BOUNDED_CONFIRMATORY_CONFIG_SHA256,
    ConfirmatoryMatrixPlan,
    build_confirmatory_matrix_plan,
    validate_confirmatory_execution_config,
)
from histo_audit.models.cnn import validate_confirmatory_checkpoint_artifact
from histo_audit.statistics.review import average_precision, budget_count, rank_indices
from histo_audit.utils.run_tracking import sha256_file

if TYPE_CHECKING:
    from histo_audit.workflows.study_gates import (
        ConfirmatoryExecutionGateEvidence,
        ResourceBoundedExecutionGateEvidence,
    )

REAL_CONFIRMATORY_ARTIFACT_SCOPE = "real_pannuke_confirmatory_study"
SYNTHETIC_CONFIRMATORY_ARTIFACT_SCOPE = "synthetic_confirmatory_orchestrator_integration_test"
RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE = "resource_bounded_confirmatory_sensitivity"
RESOURCE_BOUNDED_CAPACITY_POLICY_V2 = {
    "schema_version": 2,
    "policy": "resource_bounded_confirmatory_capacity_v2",
    "planned_required_cells": 24,
    "planned_cnn_cells": 6,
    "planned_cnn_fold_checkpoints": 30,
    "max_epochs": 4,
    "projected_stable_run_bytes": 12_884_901_888,
    "fixed_safety_margin_bytes": 10_737_418_240,
    "minimum_free_bytes_before_tracker": 23_622_320_128,
    "max_active_atomic_temp_checkpoints": 1,
    "minimum_total_ram_bytes": 32_212_254_720,
    "minimum_available_ram_bytes_before_data": 17_179_869_184,
    "minimum_available_ram_bytes_before_tracker": 12_884_901_888,
    "cuda_required": True,
    "cuda_device_index": 0,
    "minimum_total_vram_bytes": 10_737_418_240,
    "minimum_free_vram_bytes": 8_589_934_592,
    "cudnn_required": True,
    "amp_required": True,
    "amp_dtype": "float16",
    "cuda_smoke_input_shape": [1, 3, 224, 224],
    "cuda_smoke_forward_backward_required": True,
    "cuda_smoke_finite_required": True,
    "cuda_smoke_max_peak_allocated_bytes": 536_870_912,
    "official_weight_identifier": "ResNet18_Weights.IMAGENET1K_V1",
    "official_weight_sha256": ("f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"),
    "implicit_weight_download_allowed": False,
}
# Backward-compatible public name retained for already sealed authority-C evidence.
RESOURCE_BOUNDED_CAPACITY_POLICY = RESOURCE_BOUNDED_CAPACITY_POLICY_V2
CONFIRMATORY_ARTIFACT_MANIFEST_FILENAME = "confirmatory_artifact_manifest.json"
CONFIRMATORY_REPORT_CONTRACT_START = "<!-- CONFIRMATORY_REPORT_CONTRACT_V1_START -->"
CONFIRMATORY_REPORT_CONTRACT_END = "<!-- CONFIRMATORY_REPORT_CONTRACT_V1_END -->"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATUSES = {"completed", "skipped_with_frozen_blocker", "failed"}
_PRIMARY_GATE_HASHES = (
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
_CONFIRMATORY_GATE_HASHES = (
    "primary_artifact_root_sha256",
    "primary_completion_evidence_sha256",
    "primary_reconciliation_sha256",
    "confirmatory_storage_policy_sha256",
)


@dataclass(frozen=True, slots=True)
class ConfirmatoryMatrixReconciliation:
    """Exact matrix and fold-rotation reconciliation for one confirmatory run."""

    status: str
    fold_rotation_complete: bool
    planned_cell_count: int
    planned_required_cell_count: int
    completed_cell_count: int
    completed_required_cell_count: int
    skipped_optional_cell_count: int
    failed_cell_count: int
    planned_outer_folds: tuple[int, ...]
    completed_outer_folds: tuple[int, ...]
    incomplete_outer_folds: tuple[int, ...]
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
class ConfirmatoryArtifactReadback:
    """One filesystem artifact whose bytes were independently re-hashed."""

    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ConfirmatoryCellReadback:
    """Typed filesystem identity and artifact evidence for one frozen cell."""

    cell_id: str
    status: str
    outer_fold: int
    corruption_cell_id: str
    corruption_mechanism: str
    corruption_rate: float
    corruption_seed: int
    scenario_id: str
    scenario_family: str
    representation_id: str
    cache_provenance_id: str
    model_seed: int
    required: bool
    artifact_manifest_sha256: str
    metrics_sha256: str | None
    oof_identity_sha256: str | None
    corruption_mapping_sha256: str | None
    fold_assignment_sha256: str | None
    checked_artifacts: tuple[ConfirmatoryArtifactReadback, ...]


@dataclass(frozen=True, slots=True)
class ConfirmatoryFilesystemReadback:
    """Independent readback of a complete confirmatory run directory."""

    status: str
    run_directory: Path
    matrix_plan_sha256: str | None
    cell_index_sha256: str | None
    root_artifact_manifest_sha256: str | None
    confirmatory_storage_policy_sha256: str | None
    checked_artifact_count: int
    cells: tuple[ConfirmatoryCellReadback, ...]
    reconciliation: ConfirmatoryMatrixReconciliation | None
    root_artifacts: tuple[ConfirmatoryArtifactReadback, ...]
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == "passed" and not self.errors

    def as_dict(self) -> dict[str, Any]:
        """Return stable JSON-compatible evidence for completion binding."""

        return {
            "status": self.status,
            "run_directory": str(self.run_directory),
            "matrix_plan_sha256": self.matrix_plan_sha256,
            "cell_index_sha256": self.cell_index_sha256,
            "root_artifact_manifest_sha256": self.root_artifact_manifest_sha256,
            "confirmatory_storage_policy_sha256": (self.confirmatory_storage_policy_sha256),
            "checked_artifact_count": self.checked_artifact_count,
            "cells": [asdict(value) for value in self.cells],
            "reconciliation": (
                self.reconciliation.as_dict() if self.reconciliation is not None else None
            ),
            "root_artifacts": [asdict(value) for value in self.root_artifacts],
            "errors": list(self.errors),
        }


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _exact_int(value: Any, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def reconcile_confirmatory_cell_outcomes(
    plan: ConfirmatoryMatrixPlan,
    outcomes: Sequence[Mapping[str, Any]],
) -> ConfirmatoryMatrixReconciliation:
    """Reconcile all saved outcomes against the immutable confirmatory plan.

    A completed outcome must repeat the frozen fold, model seed, scenario, and
    corruption-cell identity in addition to carrying hash-bound metrics and an artifact
    manifest. Required cells cannot be skipped. Optional cells can only be skipped for
    an explicit unavailability condition frozen before execution.
    """

    expected = {cell.cell_id: cell for cell in plan.cells}
    seen: dict[str, Mapping[str, Any]] = {}
    duplicates: set[str] = set()
    invalid: set[str] = set()
    valid_resolved: set[str] = set()
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

        cell_valid = True
        status = str(outcome.get("status", ""))
        if status not in _STATUSES:
            invalid.add(cell_id)
            errors.append(f"cell {cell_id} has unsupported status {status!r}")
            continue
        if outcome.get("required") is not planned.required:
            cell_valid = False
            invalid.add(cell_id)
            errors.append(f"cell {cell_id} required flag differs from the frozen plan")

        if status == "completed":
            completed += 1
            completed_required += int(planned.required)
            identity_checks = (
                (
                    "outer_fold",
                    _exact_int(outcome.get("outer_fold"), planned.outer_fold),
                ),
                (
                    "model_seed",
                    _exact_int(outcome.get("model_seed"), planned.model_seed),
                ),
                ("scenario_id", outcome.get("scenario_id") == planned.scenario_id),
                (
                    "corruption_cell_id",
                    outcome.get("corruption_cell_id") == planned.corruption_cell_id,
                ),
            )
            for field, matches in identity_checks:
                if not matches:
                    cell_valid = False
                    invalid.add(cell_id)
                    errors.append(f"completed cell {cell_id} {field} differs from the frozen plan")
            if not _valid_sha(outcome.get("artifact_manifest_sha256")):
                cell_valid = False
                invalid.add(cell_id)
                errors.append(f"completed cell {cell_id} lacks artifact_manifest_sha256")
            if not _valid_sha(outcome.get("metrics_sha256")):
                cell_valid = False
                invalid.add(cell_id)
                errors.append(f"completed cell {cell_id} lacks metrics_sha256")
        elif status == "skipped_with_frozen_blocker":
            if planned.required:
                cell_valid = False
                invalid.add(cell_id)
                errors.append(f"required cell {cell_id} cannot be skipped")
            elif (
                outcome.get("frozen_unavailability") is not True
                or not str(outcome.get("blocker", "")).strip()
            ):
                cell_valid = False
                invalid.add(cell_id)
                errors.append(
                    f"optional skipped cell {cell_id} lacks a frozen unavailability blocker"
                )
            else:
                skipped_optional += 1
        else:
            cell_valid = False
            failed += 1
            if not str(outcome.get("error", "")).strip():
                invalid.add(cell_id)
                errors.append(f"failed cell {cell_id} lacks error evidence")

        if cell_valid:
            valid_resolved.add(cell_id)

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
        valid_resolved.difference_update(duplicates)

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

    planned_folds = tuple(sorted({cell.outer_fold for cell in plan.cells}))
    completed_folds: list[int] = []
    for fold in planned_folds:
        fold_cell_ids = {cell.cell_id for cell in plan.cells if cell.outer_fold == fold}
        if fold_cell_ids and fold_cell_ids.issubset(valid_resolved):
            completed_folds.append(fold)
    completed_outer_folds = tuple(completed_folds)
    incomplete_outer_folds = tuple(
        fold for fold in planned_folds if fold not in completed_outer_folds
    )
    fold_rotation_complete = (
        len(planned_folds) == 3
        and completed_outer_folds == planned_folds
        and not incomplete_outer_folds
    )
    if len(planned_folds) != 3:
        errors.append("frozen confirmatory plan must contain exactly three outer-fold rotations")
    if not fold_rotation_complete:
        errors.append(
            "confirmatory fold rotation is incomplete: "
            f"completed={completed_outer_folds}, planned={planned_folds}"
        )

    return ConfirmatoryMatrixReconciliation(
        status="passed" if not errors and not invalid else "failed",
        fold_rotation_complete=fold_rotation_complete,
        planned_cell_count=len(plan.cells),
        planned_required_cell_count=required_count,
        completed_cell_count=completed,
        completed_required_cell_count=completed_required,
        skipped_optional_cell_count=skipped_optional,
        failed_cell_count=failed,
        planned_outer_folds=planned_folds,
        completed_outer_folds=completed_outer_folds,
        incomplete_outer_folds=incomplete_outer_folds,
        missing_cell_ids=missing,
        extra_cell_ids=extra,
        duplicate_cell_ids=tuple(sorted(duplicates)),
        invalid_cell_ids=tuple(sorted(invalid)),
        errors=tuple(errors),
    )


_COMPLETED_CELL_ARTIFACTS = (
    "cell_identity.json",
    "oof_evidence.npz",
    "checkpoint_manifest.json",
    "telemetry.json",
    "risk_scores.npz",
    "ranking.csv",
    "metrics.json",
)
_SKIPPED_CELL_ARTIFACTS = ("cell_identity.json", "blocker.json")
_ROOT_AGGREGATE_ARTIFACTS = (
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
_ROOT_BASE_ARTIFACTS = (
    "confirmatory_input_bindings.json",
    "matrix_plan.json",
    "execution_controls.json",
    "frozen_feature_provenance.json",
    "cell_index.csv",
    "reconciliation.json",
    *_ROOT_AGGREGATE_ARTIFACTS,
)
_ENSEMBLE_RISK_NAMES = (
    "predictive_entropy_of_mean",
    "mean_pairwise_js_divergence",
    "variation_ratio",
    "observed_label_probability_variance",
    "predicted_class_disagreement",
)
_RESTORATION_CONDITIONS = {
    "uncorrupted_reference_baseline",
    "corrupted_observed_baseline",
    "random_review_restoration",
    "audit_guided_restoration",
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def confirmatory_report_contract_payload(
    restoration_metrics: Mapping[str, Any],
    statistics_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact compact report payload required by semantic readback."""

    metric_fields = (
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "per_class_precision",
        "per_class_recall",
        "confusion_matrix",
        "multiclass_brier_score",
        "expected_calibration_error",
    )
    raw_rotations = restoration_metrics.get("rotations")
    if not isinstance(raw_rotations, list):
        raise ValueError("report contract lacks restoration rotations")
    rotations: list[dict[str, Any]] = []
    for raw in raw_rotations:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("conditions"), Mapping):
            raise ValueError("report contract restoration rotation is malformed")
        conditions = cast(Mapping[str, Any], raw["conditions"])
        condition_payload: dict[str, Any] = {}
        for condition in (
            "uncorrupted_reference_baseline",
            "corrupted_observed_baseline",
            "audit_guided_restoration",
        ):
            saved = conditions.get(condition)
            if not isinstance(saved, Mapping) or not isinstance(saved.get("metrics"), Mapping):
                raise ValueError(f"report contract lacks {condition} metrics")
            metrics = cast(Mapping[str, Any], saved["metrics"])
            if any(field not in metrics for field in metric_fields):
                raise ValueError(f"report contract {condition} metrics are incomplete")
            condition_payload[condition] = {
                "metrics": {field: metrics[field] for field in metric_fields},
                "reviewed_count": saved.get("reviewed_count"),
                "restored_count": saved.get("restored_count"),
            }
        random_saved = conditions.get("random_review_restoration")
        if not isinstance(random_saved, Mapping) or not isinstance(random_saved.get("runs"), list):
            raise ValueError("report contract lacks random-review runs")
        random_runs = cast(list[Any], random_saved["runs"])
        if not random_runs:
            raise ValueError("report contract random-review runs are empty")
        random_metrics: list[Mapping[str, Any]] = []
        for run in random_runs:
            if not isinstance(run, Mapping) or not isinstance(run.get("metrics"), Mapping):
                raise ValueError("report contract random-review metrics are malformed")
            metrics = cast(Mapping[str, Any], run["metrics"])
            if any(field not in metrics for field in metric_fields):
                raise ValueError("report contract random-review metrics are incomplete")
            random_metrics.append(metrics)
        condition_payload["random_review_restoration"] = {
            "macro_f1_mean": random_saved.get("macro_f1_mean"),
            "macro_f1_interval_95": random_saved.get("macro_f1_interval_95"),
            "metrics_mean": {
                field: np.asarray([metrics[field] for metrics in random_metrics], dtype=np.float64)
                .mean(axis=0)
                .tolist()
                for field in metric_fields
            },
        }
        rotations.append(
            {
                "outer_fold": raw.get("outer_fold"),
                "corruption_cell_id": raw.get("corruption_cell_id"),
                "conditions": condition_payload,
            }
        )
    raw_comparisons = statistics_payload.get("comparisons")
    if not isinstance(raw_comparisons, list):
        raise ValueError("report contract lacks paired comparisons")
    comparisons = []
    for raw in raw_comparisons:
        if not isinstance(raw, Mapping):
            raise ValueError("report contract paired comparison is malformed")
        comparisons.append(
            {
                field: raw.get(field)
                for field in (
                    "comparison_id",
                    "status",
                    "direction",
                    "observed_delta",
                    "ci_low",
                    "ci_high",
                    "probability_positive",
                    "raw_p",
                    "holm_adjusted_p",
                    "selected_injected_event_count",
                    "blocker",
                )
            }
        )
    return {
        "schema_version": 1,
        "restoration_rotations": rotations,
        "paired_comparisons": comparisons,
    }


def confirmatory_report_contract_block(
    restoration_metrics: Mapping[str, Any],
    statistics_payload: Mapping[str, Any],
) -> str:
    """Render the exact machine-readable report contract checked on readback."""

    payload = confirmatory_report_contract_payload(
        restoration_metrics,
        statistics_payload,
    )
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "\n".join(
        (
            CONFIRMATORY_REPORT_CONTRACT_START,
            "```json",
            encoded,
            "```",
            CONFIRMATORY_REPORT_CONTRACT_END,
        )
    )


@dataclass(frozen=True, slots=True)
class _OOFBinding:
    """Canonical hashes used to prove immutable cross-cell audit inputs."""

    sample_order_sha256: str
    group_order_sha256: str
    pre_corruption_label_sha256: str
    observed_label_sha256: str
    is_injected_corruption_sha256: str
    fold_id_sha256: str
    fold_assignment_label_sha256: str

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(
            {
                "sample_order_sha256": self.sample_order_sha256,
                "group_order_sha256": self.group_order_sha256,
                "pre_corruption_label_sha256": self.pre_corruption_label_sha256,
            }
        )

    @property
    def corruption_mapping_sha256(self) -> str:
        return _canonical_sha256(
            {
                "pre_corruption_label_sha256": self.pre_corruption_label_sha256,
                "observed_label_sha256": self.observed_label_sha256,
                "is_injected_corruption_sha256": self.is_injected_corruption_sha256,
            }
        )

    @property
    def fold_assignment_sha256(self) -> str:
        return _canonical_sha256(
            {
                "fold_id_sha256": self.fold_id_sha256,
                "fold_assignment_label_sha256": self.fold_assignment_label_sha256,
            }
        )


def _read_json_object(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{role} is missing or invalid strict JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object: {path}")
    return value


def _validate_confirmatory_storage_policy_bindings(
    run: Path,
    *,
    expected_sha256: str | None,
    require_final_bindings: bool,
) -> str | None:
    """Require one exact policy hash across every available execution carrier."""

    if expected_sha256 is not None and not _valid_sha(expected_sha256):
        raise ValueError("expected confirmatory storage-policy binding is not a SHA-256")
    input_bindings = _read_json_object(
        run / "confirmatory_input_bindings.json",
        "confirmatory input bindings",
    )
    observed = input_bindings.get("confirmatory_storage_policy_sha256")
    if not _valid_sha(observed):
        if expected_sha256 is None and not require_final_bindings:
            return None
        raise ValueError("confirmatory input bindings lack the storage-policy SHA-256")
    policy_sha256 = str(observed)
    if expected_sha256 is not None and policy_sha256 != expected_sha256:
        raise ValueError("confirmatory input storage-policy binding differs from the gated policy")

    execution_gate_path = run / "confirmatory_execution_gate.json"
    provenance_path = run / "run_provenance.json"
    require_core_bindings = (
        expected_sha256 is not None
        or require_final_bindings
        or execution_gate_path.exists()
        or provenance_path.exists()
    )
    if require_core_bindings:
        execution_gate = _read_json_object(
            execution_gate_path,
            "confirmatory execution gate",
        )
        provenance = _read_json_object(provenance_path, "confirmatory run provenance")
        if (
            execution_gate.get("confirmatory_storage_policy_sha256") != policy_sha256
            or provenance.get("confirmatory_storage_policy_sha256") != policy_sha256
        ):
            raise ValueError(
                "confirmatory gate/provenance storage-policy binding differs from inputs"
            )

    if require_final_bindings:
        completion = _read_json_object(
            run / "completion_evidence.json",
            "confirmatory completion evidence",
        )
        metrics = _read_json_object(run / "metrics.json", "confirmatory metrics")
        if (
            completion.get("confirmatory_storage_policy_sha256") != policy_sha256
            or metrics.get("confirmatory_storage_policy_sha256") != policy_sha256
        ):
            raise ValueError(
                "confirmatory completion/metrics storage-policy binding differs from inputs"
            )
    return policy_sha256


def _read_csv_rows(path: Path, role: str) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError(f"{role} has invalid CSV columns")
            rows = [dict(row) for row in reader]
    except (OSError, csv.Error) as error:
        raise ValueError(f"{role} is missing or invalid CSV: {path}") from error
    if not rows:
        raise ValueError(f"{role} cannot be empty")
    return rows


def _strict_bool(value: Any, role: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    raise ValueError(f"{role} must be exactly true or false")


def _strict_int(value: Any, role: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and re.fullmatch(r"-?(?:0|[1-9][0-9]*)", value):
        return int(value)
    raise ValueError(f"{role} must be an exact integer")


def _strict_float(value: Any, role: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{role} must be finite numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{role} must be finite numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{role} must be finite numeric")
    return result


def _artifact(path: Path, root: Path) -> ConfirmatoryArtifactReadback:
    if not path.is_file():
        raise FileNotFoundError(f"required confirmatory artifact is missing: {path}")
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"confirmatory artifact escapes its run directory: {path}")
    return ConfirmatoryArtifactReadback(
        relative_path=resolved.relative_to(root.resolve()).as_posix(),
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _read_hash_manifest(path: Path, role: str) -> dict[str, str]:
    raw = _read_json_object(path, role)
    if set(raw) == {"schema_version", "files"}:
        files = raw["files"]
        if raw["schema_version"] != 1 or not isinstance(files, Mapping) or not files:
            raise ValueError(f"{role} has an invalid files schema")
        file_output: dict[str, str] = {}
        for relative, digest in files.items():
            if not relative or not _valid_sha(digest):
                raise ValueError(f"{role} has an invalid path/SHA-256 record")
            file_output[str(relative)] = str(digest)
        if len(file_output) != len(files):
            raise ValueError(f"{role} contains duplicate normalised paths")
        return file_output
    if set(raw) == {"schema_version", "artifacts"}:
        if raw["schema_version"] != 1 or not isinstance(raw["artifacts"], list):
            raise ValueError(f"{role} has an invalid schema")
        output: dict[str, str] = {}
        for index, value in enumerate(raw["artifacts"]):
            if not isinstance(value, Mapping):
                raise ValueError(f"{role}.artifacts[{index}] must be a mapping")
            relative = str(value.get("path", ""))
            digest = value.get("sha256")
            if not relative or not _valid_sha(digest) or relative in output:
                raise ValueError(f"{role}.artifacts[{index}] is invalid or duplicate")
            output[relative] = str(digest)
        return output
    output = {}
    for relative, digest in raw.items():
        if not relative or not _valid_sha(digest):
            raise ValueError(f"{role} has an invalid path/SHA-256 record")
        output[str(relative)] = str(digest)
    if not output:
        raise ValueError(f"{role} cannot be empty")
    return output


def _verify_hash_manifest(
    directory: Path,
    manifest: Mapping[str, str],
    *,
    expected_paths: set[str],
    role: str,
    run_root: Path,
) -> tuple[ConfirmatoryArtifactReadback, ...]:
    if set(manifest) != expected_paths:
        missing = sorted(expected_paths.difference(manifest))
        extra = sorted(set(manifest).difference(expected_paths))
        raise ValueError(f"{role} path set differs: missing={missing}, extra={extra}")
    checked: list[ConfirmatoryArtifactReadback] = []
    for relative, expected_sha in sorted(manifest.items()):
        candidate = (directory / relative).resolve()
        if not candidate.is_relative_to(directory.resolve()):
            raise ValueError(f"{role} path escapes its directory: {relative}")
        record = _artifact(candidate, run_root)
        if record.sha256 != expected_sha:
            raise ValueError(f"{role} SHA-256 mismatch for {relative}")
        checked.append(record)
    return tuple(checked)


def _load_frozen_config(
    plan: ConfirmatoryMatrixPlan,
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError("frozen confirmatory config file SHA-256 differs from gate evidence")
    resolved = validate_confirmatory_execution_config(load_config(path))
    if build_confirmatory_matrix_plan(resolved) != plan:
        raise ValueError("filesystem readback frozen config differs from matrix plan")
    return resolved


def _validate_matrix_and_controls(
    run: Path,
    plan: ConfirmatoryMatrixPlan,
    config: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    matrix = _read_json_object(run / "matrix_plan.json", "confirmatory matrix plan")
    if matrix != plan.as_dict():
        raise ValueError("filesystem matrix_plan.json differs from the frozen plan")
    controls = _read_json_object(run / "execution_controls.json", "confirmatory execution controls")
    # Local import avoids the intentional core -> completion import at module load.
    from histo_audit.experiment.confirmatory_core import (
        confirmatory_execution_controls_from_frozen_config,
    )

    expected_controls = confirmatory_execution_controls_from_frozen_config(config).as_dict()
    if _canonical_sha256(controls) != _canonical_sha256(expected_controls):
        raise ValueError("execution controls differ from the complete frozen configuration")
    if controls.get("config_semantic_sha256") != plan.config_sha256:
        raise ValueError("execution controls config SHA differs from frozen plan")
    if controls.get("plan") != plan.as_dict():
        raise ValueError("execution controls embed a different matrix plan")
    binding = controls.get("binding_sha256")
    payload = {
        key: value
        for key, value in controls.items()
        if key not in {"schema_version", "source", "binding_sha256"}
    }
    if not _valid_sha(binding) or binding != _canonical_sha256(payload):
        raise ValueError("execution controls binding SHA-256 is invalid")

    config_corruptions = config["corruption"]
    config_scenarios = config["scenarios"]
    if not isinstance(config_corruptions, Mapping) or not isinstance(config_scenarios, Sequence):
        raise ValueError("frozen config lacks corruption/scenario records")
    raw_corruptions = config_corruptions.get("cells")
    if not isinstance(raw_corruptions, Sequence):
        raise ValueError("frozen config corruption cells are invalid")
    corruptions = {
        str(value["id"]): value for value in raw_corruptions if isinstance(value, Mapping)
    }
    scenarios = {
        str(value["id"]): value for value in config_scenarios if isinstance(value, Mapping)
    }
    if len(corruptions) != len(raw_corruptions) or len(scenarios) != len(config_scenarios):
        raise ValueError("frozen config has duplicate/invalid corruption or scenario IDs")

    control_corruptions = controls.get("corruption_specs")
    control_scenarios = controls.get("scenario_specs")
    if not isinstance(control_corruptions, list) or not isinstance(control_scenarios, list):
        raise ValueError("execution controls lack corruption/scenario specs")
    expected_corruption_specs = [
        {
            "corruption_cell_id": str(value["id"]),
            "mechanism": str(value["mechanism"]),
            "rate": float(value["rate"]),
            "seed": int(value["seed"]),
            "parameters": dict(value["parameters"]),
            "parameters_sha256": _canonical_sha256(dict(value["parameters"])),
        }
        for value in raw_corruptions
        if isinstance(value, Mapping)
    ]
    if control_corruptions != expected_corruption_specs:
        raise ValueError("execution-control corruption specs differ from frozen config")
    for control in control_scenarios:
        if not isinstance(control, Mapping):
            raise ValueError("execution-control scenario spec is invalid")
        scenario = scenarios.get(str(control.get("scenario_id", "")))
        if scenario is None:
            raise ValueError("execution controls contain an unknown scenario")
        expected = {
            "scenario_id": str(scenario["id"]),
            "family": str(scenario["family"]),
            "input_variant": str(scenario["input_variant"]),
            "encoder": str(scenario["encoder"]),
            "classifier": str(scenario["classifier"]),
            "representation_id": str(scenario["representation_id"]),
            "cache_provenance_id": str(scenario["cache_provenance_id"]),
            "required": bool(scenario["required"]),
            "availability_audit_sha256": scenario.get("availability_audit_sha256"),
        }
        if dict(control) != expected:
            raise ValueError("execution-control scenario spec differs from frozen config")
    if len(control_scenarios) != len(scenarios):
        raise ValueError("execution controls do not cover every frozen scenario")
    return corruptions, scenarios


def _cell_identity(
    raw: Mapping[str, Any],
    *,
    planned: Any,
    corruption: Mapping[str, Any],
    scenario: Mapping[str, Any],
    config_sha256: str,
    role: str,
) -> dict[str, Any]:
    expected = {
        "schema_version": 1,
        "cell_id": planned.cell_id,
        "outer_fold": planned.outer_fold,
        "corruption_cell_id": planned.corruption_cell_id,
        "corruption_mechanism": str(corruption["mechanism"]),
        "corruption_rate": float(corruption["rate"]),
        "corruption_seed": int(corruption["seed"]),
        "scenario_id": planned.scenario_id,
        "scenario_family": str(scenario["family"]),
        "representation_id": str(scenario["representation_id"]),
        "cache_provenance_id": str(scenario["cache_provenance_id"]),
        "model_seed": planned.model_seed,
        "required": planned.required,
        "config_semantic_sha256": config_sha256,
    }
    if dict(raw) != expected:
        raise ValueError(f"{role} differs from frozen cell identity")
    return expected


def _validate_oof_and_risks(
    cell_directory: Path,
    *,
    cell_id: str,
    corruption: Mapping[str, Any],
    hybrid_components: tuple[str, ...],
    hybrid_weights: tuple[float, ...],
    n_splits: int,
) -> tuple[
    tuple[str, ...],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    _OOFBinding,
]:
    path = cell_directory / "oof_evidence.npz"
    required_oof = {
        "sample_ids",
        "group_ids",
        "pre_corruption_label",
        "observed_label",
        "is_injected_corruption",
        "probabilities",
        "fold_id",
        "fold_assignment_labels",
        "coverage_count",
    }
    try:
        with np.load(path, allow_pickle=False) as payload:
            if missing := sorted(required_oof.difference(payload.files)):
                raise ValueError(f"{cell_id} OOF evidence lacks arrays: {missing}")
            sample_raw = payload["sample_ids"]
            group_raw = payload["group_ids"]
            if sample_raw.dtype.kind not in {"U", "S"} or group_raw.dtype.kind not in {
                "U",
                "S",
            }:
                raise ValueError(f"{cell_id} OOF identities require object/pickle")
            sample_ids = tuple(str(value) for value in sample_raw.tolist())
            groups = np.asarray(group_raw, dtype=np.str_)
            pre = np.asarray(payload["pre_corruption_label"])
            observed = np.asarray(payload["observed_label"])
            injected = np.asarray(payload["is_injected_corruption"])
            probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
            fold_id = np.asarray(payload["fold_id"])
            assignment = np.asarray(payload["fold_assignment_labels"])
            coverage = np.asarray(payload["coverage_count"])
    except (OSError, KeyError, ValueError) as error:
        if isinstance(error, ValueError) and cell_id in str(error):
            raise
        raise ValueError(f"{cell_id} OOF evidence is invalid or pickle-dependent") from error
    n = len(sample_ids)
    if not n or len(set(sample_ids)) != n or any(not value for value in sample_ids):
        raise ValueError(f"{cell_id} OOF sample IDs are invalid")
    if groups.shape != (n,) or any(not value for value in groups):
        raise ValueError(f"{cell_id} OOF groups are misaligned")
    for name, values in (
        ("pre labels", pre),
        ("observed labels", observed),
        ("fold assignment labels", assignment),
        ("fold IDs", fold_id),
        ("coverage", coverage),
    ):
        if values.shape != (n,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{cell_id} OOF {name} are not aligned integers")
    if injected.shape != (n,) or not np.issubdtype(injected.dtype, np.bool_):
        raise ValueError(f"{cell_id} corruption flags are not aligned booleans")
    if not np.array_equal(injected, observed != pre):
        raise ValueError(f"{cell_id} corruption flags differ from label changes")
    if not np.array_equal(assignment, pre):
        raise ValueError(f"{cell_id} OOF assignment is not frozen from pre labels")
    if set(int(value) for value in pre) != {0, 1, 2, 3, 4}:
        raise ValueError(f"{cell_id} OOF lacks the fixed five classes")
    if not np.isin(observed, (0, 1, 2, 3, 4)).all():
        raise ValueError(f"{cell_id} observed label lies outside fixed class order")
    expected_injected = math.floor(n * float(corruption["rate"]) + 0.5)
    if int(injected.sum()) != expected_injected:
        raise ValueError(f"{cell_id} injected count differs from frozen rate")
    if probabilities.shape != (n, 5) or not np.isfinite(probabilities).all():
        raise ValueError(f"{cell_id} OOF probabilities are malformed/non-finite")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError(f"{cell_id} OOF probabilities lie outside [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError(f"{cell_id} OOF probability rows do not sum to one")
    if not np.array_equal(coverage, np.ones(n, dtype=coverage.dtype)):
        raise ValueError(f"{cell_id} OOF coverage is not exactly once")
    if set(int(value) for value in fold_id) != set(range(n_splits)):
        raise ValueError(f"{cell_id} OOF fold IDs do not cover the frozen split count")
    for group in np.unique(groups):
        if len(np.unique(fold_id[groups == group])) != 1:
            raise ValueError(f"{cell_id} source group spans OOF holdout folds")

    risk_path = cell_directory / "risk_scores.npz"
    required_risks = {
        "sample_ids",
        "ensemble_mean_probabilities",
        "self_confidence",
        "ensemble_disagreement",
        "fixed_hybrid",
        *(f"ensemble_{name}" for name in _ENSEMBLE_RISK_NAMES),
        *(f"hybrid_drop_{component}" for component in hybrid_components),
    }
    try:
        with np.load(risk_path, allow_pickle=False) as payload:
            if missing := sorted(required_risks.difference(payload.files)):
                raise ValueError(f"{cell_id} risk evidence lacks arrays: {missing}")
            risk_ids_raw = payload["sample_ids"]
            if risk_ids_raw.dtype.kind not in {"U", "S"}:
                raise ValueError(f"{cell_id} risk sample IDs require object/pickle")
            risk_ids = tuple(str(value) for value in risk_ids_raw.tolist())
            ensemble_mean = np.asarray(payload["ensemble_mean_probabilities"], dtype=np.float64)
            risk_values = {
                name: np.asarray(payload[name], dtype=np.float64)
                for name in required_risks
                if name not in {"sample_ids", "ensemble_mean_probabilities"}
            }
    except (OSError, KeyError, ValueError) as error:
        if isinstance(error, ValueError) and cell_id in str(error):
            raise
        raise ValueError(f"{cell_id} risk evidence is invalid or pickle-dependent") from error
    if risk_ids != sample_ids:
        raise ValueError(f"{cell_id} risk sample order differs from OOF evidence")
    if (
        ensemble_mean.shape != probabilities.shape
        or not np.isfinite(ensemble_mean).all()
        or np.any(ensemble_mean < 0.0)
        or np.any(ensemble_mean > 1.0)
        or not np.allclose(ensemble_mean.sum(axis=1), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise ValueError(f"{cell_id} ensemble-mean probabilities are malformed")
    if any(
        values.shape != (n,) or not np.isfinite(values).all() for values in risk_values.values()
    ):
        raise ValueError(f"{cell_id} risk arrays are misaligned or non-finite")

    expected_self_confidence = score_annotations(
        observed,
        probabilities,
        method="self_confidence",
        class_order=CLASS_ORDER,
    )
    if not np.allclose(
        risk_values["self_confidence"],
        expected_self_confidence,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"{cell_id} self-confidence risk differs from OOF probabilities")
    hybrid = fixed_hybrid_drop_one_ablations(
        {
            "self_confidence": expected_self_confidence,
            "ensemble_disagreement": risk_values["ensemble_disagreement"],
        },
        components=hybrid_components,
        weights=hybrid_weights,
    )
    if not np.allclose(risk_values["fixed_hybrid"], hybrid.full_score, rtol=0.0, atol=1e-12):
        raise ValueError(f"{cell_id} fixed-hybrid risk differs from frozen components")
    for component, expected_values in hybrid.drop_one_scores.items():
        if not np.allclose(
            risk_values[f"hybrid_drop_{component}"],
            expected_values,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{cell_id} drop-one hybrid for {component} differs from frozen components"
            )

    binding = _OOFBinding(
        sample_order_sha256=_canonical_sha256(list(sample_ids)),
        group_order_sha256=_canonical_sha256([str(value) for value in groups.tolist()]),
        pre_corruption_label_sha256=_canonical_sha256([int(value) for value in pre.tolist()]),
        observed_label_sha256=_canonical_sha256([int(value) for value in observed.tolist()]),
        is_injected_corruption_sha256=_canonical_sha256(
            [bool(value) for value in injected.tolist()]
        ),
        fold_id_sha256=_canonical_sha256([int(value) for value in fold_id.tolist()]),
        fold_assignment_label_sha256=_canonical_sha256(
            [int(value) for value in assignment.tolist()]
        ),
    )
    return (
        sample_ids,
        injected.astype(bool, copy=False),
        risk_values["fixed_hybrid"],
        binding,
    )


def _validate_ranking_and_metrics(
    cell_directory: Path,
    *,
    cell_id: str,
    identity: Mapping[str, Any],
    sample_ids: tuple[str, ...],
    injected: np.ndarray[Any, Any],
    fixed_hybrid: np.ndarray[Any, Any],
) -> None:
    rows = _read_csv_rows(cell_directory / "ranking.csv", f"{cell_id} ranking")
    expected_columns = {"sample_id", "risk_method", "risk_score", "rank"}
    if set(rows[0]) != expected_columns or any(set(row) != expected_columns for row in rows):
        raise ValueError(f"{cell_id} ranking columns differ from the frozen schema")
    if len(rows) != len(sample_ids):
        raise ValueError(f"{cell_id} ranking does not cover every OOF sample")
    ordered = sorted(
        zip(sample_ids, fixed_hybrid, strict=True),
        key=lambda item: (-float(item[1]), item[0]),
    )
    for rank, (row, expected) in enumerate(zip(rows, ordered, strict=True), start=1):
        if (
            row["sample_id"] != expected[0]
            or row["risk_method"] != "fixed_hybrid"
            or _strict_int(row["rank"], f"{cell_id} ranking rank") != rank
            or not math.isclose(
                _strict_float(row["risk_score"], f"{cell_id} ranking score"),
                float(expected[1]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"{cell_id} ranking differs from saved fixed-hybrid scores")
    metrics = _read_json_object(cell_directory / "metrics.json", f"{cell_id} metrics")
    if metrics.get("cell_identity") != dict(identity):
        raise ValueError(f"{cell_id} metrics repeat a different cell identity")
    ranking = metrics.get("ranking")
    if not isinstance(ranking, Mapping) or not isinstance(ranking.get("fixed_hybrid"), Mapping):
        raise ValueError(f"{cell_id} metrics lack fixed-hybrid ranking metrics")
    saved_ap = ranking["fixed_hybrid"].get("average_precision")
    expected_ap = average_precision(injected, fixed_hybrid)
    if expected_ap is None:
        if saved_ap is not None:
            raise ValueError(f"{cell_id} zero-corruption AP must be null")
    elif not math.isclose(
        _strict_float(saved_ap, f"{cell_id} fixed-hybrid AP"),
        expected_ap,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{cell_id} fixed-hybrid AP differs from saved scores")


def _is_link_or_reparse(stat_result: os.stat_result) -> bool:
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    return stat.S_ISLNK(stat_result.st_mode) or bool(attributes & reparse_flag)


def _validate_local_private_checkpoint(
    candidate: Path,
    *,
    cell_directory: Path,
    run_root: Path,
    role: str,
) -> Path:
    """Require one regular canonical single-link file with no reparse component."""

    run_resolved = run_root.resolve(strict=True)
    cell_resolved = cell_directory.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(cell_resolved):
        raise ValueError(f"{role} escapes its final cell directory")
    try:
        lexical_relative = candidate.relative_to(run_resolved)
    except ValueError as error:
        raise ValueError(f"{role} is not lexically inside the confirmatory run") from error

    current = run_resolved
    final_stat: os.stat_result | None = None
    for part in lexical_relative.parts:
        current /= part
        try:
            current_stat = current.lstat()
        except OSError as error:
            raise FileNotFoundError(f"{role} path component is missing: {current}") from error
        if _is_link_or_reparse(current_stat):
            raise ValueError(f"{role} contains a symbolic-link or reparse-point component")
        if current != candidate and not stat.S_ISDIR(current_stat.st_mode):
            raise ValueError(f"{role} has a non-directory parent component")
        final_stat = current_stat

    if final_stat is None or not stat.S_ISREG(final_stat.st_mode):
        raise ValueError(f"{role} is not a regular file")
    if final_stat.st_nlink != 1:
        raise ValueError(f"{role} must have exactly one filesystem link")
    return resolved


def _strict_canonical_ascii_json_bytes(payload: bytes, *, role: str) -> dict[str, Any]:
    """Decode one duplicate-free canonical ASCII JSON object."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"{role} contains a duplicate key")
            output[key] = value
        return output

    try:
        decoded = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{role} contains non-finite JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{role} is not strict canonical ASCII JSON") from error
    expected = (
        json.dumps(
            decoded,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    if not isinstance(decoded, dict) or payload != expected:
        raise ValueError(f"{role} is not one canonical ASCII JSON object")
    return decoded


def _checkpoint_directive_from_exact_payload(
    payload: object,
    *,
    role: str,
) -> ConfirmatoryCheckpointDirective:
    """Reconstruct and validate one exact checkpoint directive."""

    expected_keys = set(ConfirmatoryCheckpointDirective.__dataclass_fields__)
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise ValueError(f"{role} directive schema is not exact")

    def checkpoint_identity(
        value: object,
        *,
        identity_role: str,
    ) -> ConfirmatoryCheckpointFileIdentity | None:
        if value is None:
            return None
        try:
            return _checkpoint_file_identity_from_exact_manifest(
                value,
                role=identity_role,
            )
        except ConfirmatoryCheckpointContractError as error:
            raise ValueError(f"{identity_role} is invalid") from error

    try:
        directive = ConfirmatoryCheckpointDirective(
            execution_mode=cast(Any, payload["execution_mode"]),
            cell_id=cast(Any, payload["cell_id"]),
            fold_id=cast(Any, payload["fold_id"]),
            action=cast(Any, payload["action"]),
            source_predecessor_checkpoint=checkpoint_identity(
                payload["source_predecessor_checkpoint"],
                identity_role=f"{role} source predecessor checkpoint",
            ),
            destination_imported_checkpoint=checkpoint_identity(
                payload["destination_imported_checkpoint"],
                identity_role=f"{role} destination imported checkpoint",
            ),
            versioned_checkpoint_output_directory_relative_path=cast(
                Any,
                payload["versioned_checkpoint_output_directory_relative_path"],
            ),
            checkpoint_execution_manifest_relative_path=cast(
                Any,
                payload["checkpoint_execution_manifest_relative_path"],
            ),
            checkpoint_sha256=cast(Any, payload["checkpoint_sha256"]),
            checkpoint_size_bytes=cast(Any, payload["checkpoint_size_bytes"]),
            completed_epochs_before_fit=cast(
                Any,
                payload["completed_epochs_before_fit"],
            ),
            stopped_early_before_fit=cast(Any, payload["stopped_early_before_fit"]),
            next_epoch_index=cast(Any, payload["next_epoch_index"]),
            maximum_epochs=cast(Any, payload["maximum_epochs"]),
            expected_configuration_json=cast(
                Any,
                payload["expected_configuration_json"],
            ),
            expected_configuration_sha256=cast(
                Any,
                payload["expected_configuration_sha256"],
            ),
            expected_model_metadata_json=cast(
                Any,
                payload["expected_model_metadata_json"],
            ),
            expected_model_metadata_sha256=cast(
                Any,
                payload["expected_model_metadata_sha256"],
            ),
            expected_data_and_split_json=cast(
                Any,
                payload["expected_data_and_split_json"],
            ),
            expected_data_and_split_sha256=cast(
                Any,
                payload["expected_data_and_split_sha256"],
            ),
        )
        directive.validate()
    except (ConfirmatoryCheckpointContractError, TypeError, ValueError) as error:
        raise ValueError(f"{role} directive is invalid") from error
    if directive.as_dict() != dict(payload):
        raise ValueError(f"{role} directive does not round-trip exactly")
    return directive


def _hold_exact_checkpoint_artifact(
    stack: ExitStack,
    candidate: Path,
    *,
    cell_directory: Path,
    run_root: Path,
    expected_sha256: object,
    expected_size_bytes: object,
    expected_physical_identity: object,
    role: str,
    require_read_only: bool = False,
) -> tuple[ConfirmatoryArtifactReadback, Any]:
    """Hold and bind one exact private checkpoint-tree file through readback."""

    if (
        not isinstance(expected_sha256, str)
        or not _valid_sha(expected_sha256)
        or type(expected_size_bytes) is not int
        or expected_size_bytes <= 0
    ):
        raise ValueError(f"{role} hash/size binding is invalid")
    try:
        physical_identity = _checkpoint_physical_identity_from_exact_manifest(
            expected_physical_identity,
            role=f"{role} physical identity",
        )
    except ConfirmatoryCheckpointContractError as error:
        raise ValueError(f"{role} physical identity is invalid") from error
    resolved = _validate_local_private_checkpoint(
        candidate,
        cell_directory=cell_directory,
        run_root=run_root,
        role=role,
    )
    try:
        held = stack.enter_context(
            _hold_private_checkpoint_snapshot(
                resolved,
                role=role,
            )
        )
    except ConfirmatoryCheckpointContractError as error:
        raise ValueError(f"{role} cannot be held as one exact private file") from error
    if (
        held.sha256 != expected_sha256
        or held.size_bytes != expected_size_bytes
        or held.identity != physical_identity
        or (
            require_read_only
            and stat.S_IMODE(held.identity.mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        )
    ):
        raise ValueError(f"{role} differs from its hash/size/physical identity")
    return (
        ConfirmatoryArtifactReadback(
            relative_path=resolved.relative_to(run_root.resolve(strict=True)).as_posix(),
            sha256=held.sha256,
            size_bytes=held.size_bytes,
        ),
        held,
    )


def _validate_checkpoint_and_telemetry(
    cell_directory: Path,
    *,
    run_root: Path,
    identity: Mapping[str, Any],
    n_splits: int,
    split_seed: int,
    scenario: Mapping[str, Any],
    cache_record: Mapping[str, Any],
    training: Mapping[str, Any],
    original_audit_selection: Mapping[str, Any],
    expected_cnn_data_hashes_by_fold: Mapping[str, Any] | None,
) -> tuple[ConfirmatoryArtifactReadback, ...]:
    cell_id = str(identity["cell_id"])
    telemetry = _read_json_object(cell_directory / "telemetry.json", f"{cell_id} telemetry")
    mode = str(telemetry.get("execution_mode", ""))
    family = str(identity["scenario_family"])
    expected_mode = "real_study_cuda" if family == "cnn" else "real_study_cpu"
    if (
        scenario.get("id") != identity.get("scenario_id")
        or scenario.get("family") != family
        or scenario.get("cache_provenance_id") != identity.get("cache_provenance_id")
        or cache_record.get("id") != identity.get("cache_provenance_id")
        or cache_record.get("representation_id") != identity.get("representation_id")
        or cache_record.get("status") != "available"
    ):
        raise ValueError(f"{cell_id} scenario/cache provenance identity is invalid")
    configuration_sha256 = telemetry.get("configuration_sha256")
    if (
        telemetry.get("schema_version") != 1
        or telemetry.get("cell_id") != cell_id
        or telemetry.get("study_outcome_eligible") is not True
        or mode != expected_mode
        or not _valid_sha(configuration_sha256)
        or not isinstance(telemetry.get("evidence"), Mapping)
    ):
        raise ValueError(f"{cell_id} telemetry is not real-study eligible")
    folds = telemetry.get("folds")
    if not isinstance(folds, list) or len(folds) != n_splits:
        raise ValueError(f"{cell_id} telemetry lacks every OOF fold")
    try:
        with np.load(cell_directory / "oof_evidence.npz", allow_pickle=False) as payload:
            sample_raw = payload["sample_ids"]
            group_raw = payload["group_ids"]
            fold_id = np.asarray(payload["fold_id"])
            if sample_raw.dtype.kind not in {"U", "S"} or group_raw.dtype.kind not in {
                "U",
                "S",
            }:
                raise ValueError("OOF identities require pickle")
            sample_ids = tuple(str(value) for value in sample_raw.tolist())
            group_ids = tuple(str(value) for value in group_raw.tolist())
            observed_labels = np.asarray(payload["observed_label"], dtype=np.int64)
            fold_assignment_labels = np.asarray(payload["fold_assignment_labels"], dtype=np.int64)
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f"{cell_id} telemetry cannot be bound to OOF evidence") from error
    all_groups = set(group_ids)
    seen_fold_ids: set[int] = set()
    for raw_fold in folds:
        if not isinstance(raw_fold, Mapping) or set(raw_fold) != {
            "fold_id",
            "training_groups",
            "held_out_groups",
            "held_out_sample_ids",
        }:
            raise ValueError(f"{cell_id} telemetry fold schema is invalid")
        current_fold = _strict_int(raw_fold["fold_id"], f"{cell_id} telemetry fold ID")
        if current_fold in seen_fold_ids:
            raise ValueError(f"{cell_id} telemetry repeats an OOF fold")
        seen_fold_ids.add(current_fold)
        held_indices = np.flatnonzero(fold_id == current_fold)
        expected_samples = [sample_ids[int(index)] for index in held_indices]
        expected_held_groups = {group_ids[int(index)] for index in held_indices}
        training_groups = raw_fold["training_groups"]
        held_groups = raw_fold["held_out_groups"]
        held_samples = raw_fold["held_out_sample_ids"]
        if (
            not isinstance(training_groups, list)
            or not isinstance(held_groups, list)
            or not isinstance(held_samples, list)
            or held_samples != expected_samples
            or set(str(value) for value in held_groups) != expected_held_groups
            or set(str(value) for value in training_groups)
            != all_groups.difference(expected_held_groups)
            or set(str(value) for value in training_groups).intersection(
                str(value) for value in held_groups
            )
        ):
            raise ValueError(f"{cell_id} telemetry does not prove group-safe OOF")
    if seen_fold_ids != set(range(n_splits)):
        raise ValueError(f"{cell_id} telemetry fold IDs differ from frozen split count")

    checkpoint = _read_json_object(
        cell_directory / "checkpoint_manifest.json", f"{cell_id} checkpoint manifest"
    )
    records = checkpoint.get("checkpoints")
    expected_checkpoint_schema = 3 if family == "cnn" else 1
    if (
        set(checkpoint) != {"schema_version", "cell_id", "status", "checkpoints"}
        or checkpoint.get("schema_version") != expected_checkpoint_schema
        or checkpoint.get("cell_id") != cell_id
        or not isinstance(records, list)
    ):
        raise ValueError(f"{cell_id} checkpoint manifest identity/schema is invalid")
    expected_status = "complete" if family == "cnn" else "not_applicable_frozen_feature"
    if checkpoint.get("status") != expected_status:
        raise ValueError(f"{cell_id} checkpoint status differs from scenario family")
    if (family == "cnn" and len(records) != n_splits) or (family != "cnn" and records):
        raise ValueError(f"{cell_id} checkpoint count differs from frozen execution")
    telemetry_evidence = telemetry["evidence"]
    provenance = {
        "cache_provenance_id": cache_record.get("id"),
        "representation_id": cache_record.get("representation_id"),
        "cache_file_sha256": cache_record.get("cache_file_sha256"),
        "sidecar_semantic_sha256": cache_record.get("sidecar_semantic_sha256"),
        "sample_order_sha256": cache_record.get("sample_order_sha256"),
        "manifest_sha256": cache_record.get("manifest_sha256"),
        "encoder_identifier": cache_record.get("encoder_identifier"),
        "encoder_metadata_sha256": cache_record.get("encoder_metadata_sha256"),
        "weight_identifier": cache_record.get("weight_identifier"),
        "weights_sha256": cache_record.get("weights_sha256"),
        "preprocessing_identifier": cache_record.get("preprocessing_identifier"),
        "preprocessing_sha256": cache_record.get("preprocessing_sha256"),
        "input_variant": cache_record.get("input_variant"),
        "audit_sample_order_sha256": _canonical_sha256(list(sample_ids)),
    }
    provenance_sha256 = _canonical_sha256(provenance)
    fold_execution_by_id: dict[int, Mapping[str, Any]] = {}
    cnn_configuration_by_fold: dict[int, Mapping[str, Any]] = {}
    cnn_metadata_by_fold: dict[int, Mapping[str, Any]] = {}
    cnn_data_hashes_by_fold: dict[int, Mapping[str, str]] = {}
    if family == "cnn":
        if not isinstance(expected_cnn_data_hashes_by_fold, Mapping) or set(
            expected_cnn_data_hashes_by_fold
        ) != {str(value) for value in range(n_splits)}:
            raise ValueError(f"{cell_id} lacks pre-execution CNN data/split bindings")
        fold_execution = telemetry_evidence.get("fold_evidence")
        if (
            set(telemetry_evidence)
            != {
                "fold_evidence",
                "execution_mode",
                "study_outcome_eligible",
                "scenario_cache_provenance_sha256",
            }
            or telemetry_evidence.get("execution_mode") != "real_study_cuda"
            or telemetry_evidence.get("study_outcome_eligible") is not True
            or telemetry_evidence.get("scenario_cache_provenance_sha256") != provenance_sha256
            or not isinstance(fold_execution, list)
            or len(fold_execution) != n_splits
        ):
            raise ValueError(f"{cell_id} CNN telemetry lacks fold execution evidence")
        uses_mask = scenario.get("input_variant") == "context_rgb_plus_binary_target_mask"
        expected_preprocessing = {
            "rgb_resize": "bilinear_antialias",
            "rgb_range_before_normalisation": [0.0, 1.0],
            "rgb_mean": [0.485, 0.456, 0.406],
            "rgb_standard_deviation": [0.229, 0.224, 0.225],
            "target_mask_resize": "nearest_binary_unnormalised" if uses_mask else None,
        }
        for raw_execution in fold_execution:
            if not isinstance(raw_execution, Mapping):
                raise ValueError(f"{cell_id} CNN fold execution evidence is invalid")
            fold = _strict_int(raw_execution.get("fold_id"), f"{cell_id} CNN fold ID")
            expected_configuration = {
                "input_variant": scenario.get("input_variant"),
                "weight_identifier": cache_record.get("weight_identifier"),
                "fourth_channel_initialisation": "zeros",
                "input_size": int(training["input_size"]),
                "epochs": int(training["max_epochs"]),
                "batch_size": int(training["initial_batch_size"]),
                "minimum_batch_size": int(training["minimum_batch_size"]),
                "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
                "learning_rate": float(training["learning_rate"]),
                "weight_decay": float(training["weight_decay"]),
                "early_stopping_patience": int(training["early_stopping_patience"]),
                "early_stopping_min_delta": float(training["early_stopping_min_delta"]),
                "amp_dtype": training["amp_dtype"],
                "class_weight_balanced": training["class_weight"] == "balanced",
                "seed": int(identity["model_seed"]) + fold,
            }
            expected_configuration_sha256 = _canonical_sha256(expected_configuration)
            fold_telemetry = raw_execution.get("telemetry")
            model_metadata = raw_execution.get("model_metadata")
            data_hashes = raw_execution.get("data_and_split_sha256")
            if (
                fold in fold_execution_by_id
                or not _valid_sha(raw_execution.get("checkpoint_sha256"))
                or raw_execution.get("configuration_sha256") != expected_configuration_sha256
                or raw_execution.get("model_seed") != expected_configuration["seed"]
                or raw_execution.get("execution_mode") != "real_study_cuda"
                or raw_execution.get("study_outcome_eligible") is not True
                or not isinstance(fold_telemetry, Mapping)
                or fold_telemetry.get("execution_mode") != "real_study_cuda"
                or fold_telemetry.get("study_outcome_eligible") is not True
                or not isinstance(model_metadata, Mapping)
                or not isinstance(data_hashes, Mapping)
                or set(data_hashes)
                != {
                    "training_data_sha256",
                    "reference_validation_data_sha256",
                    "training_split_sha256",
                    "reference_validation_split_sha256",
                }
                or any(not _valid_sha(value) for value in data_hashes.values())
                or dict(data_hashes) != expected_cnn_data_hashes_by_fold.get(str(fold))
                or model_metadata.get("weight_identifier") != cache_record.get("weight_identifier")
                or model_metadata.get("weight_sha256") != cache_record.get("weights_sha256")
                or model_metadata.get("architecture") != "torchvision.resnet18"
                or model_metadata.get("class_order") != list(CLASS_ORDER)
                or model_metadata.get("input_channels") != (4 if uses_mask else 3)
                or model_metadata.get("preprocessing") != expected_preprocessing
                or model_metadata.get("fourth_channel_initialisation")
                != ("zeros" if uses_mask else None)
            ):
                raise ValueError(
                    f"{cell_id} CNN fold execution differs from frozen mode, "
                    "configuration, weights, preprocessing, or pre-execution data/split "
                    "bindings"
                )
            fold_execution_by_id[fold] = raw_execution
            cnn_configuration_by_fold[fold] = expected_configuration
            cnn_metadata_by_fold[fold] = dict(model_metadata)
            cnn_data_hashes_by_fold[fold] = {
                str(key): str(value) for key, value in data_hashes.items()
            }
        configuration_hashes = [
            str(fold_execution_by_id[fold]["configuration_sha256"]) for fold in range(n_splits)
        ]
        if configuration_sha256 != _canonical_sha256(configuration_hashes):
            raise ValueError(f"{cell_id} CNN aggregate configuration SHA is invalid")
    else:
        classifier = original_audit_selection.get("classifier")
        if not isinstance(classifier, Mapping) or not isinstance(
            classifier.get("parameters"), Mapping
        ):
            raise ValueError("frozen original-audit classifier selection is malformed")
        parameters = cast(Mapping[str, Any], classifier["parameters"])
        configuration = {
            "schema_version": 1,
            "classifier": "multinomial_logistic_regression",
            "representation_id": identity["representation_id"],
            "model_seed": int(identity["model_seed"]),
            "split_seed": split_seed,
            "n_splits": n_splits,
            "l2": float(parameters["l2"]),
            "max_iter": int(parameters["max_iter"]),
            "class_weight": "balanced",
            "class_order": list(CLASS_ORDER),
            "fold_assignment_label_source": "pre_corruption_label",
            "frozen_feature_provenance_sha256": provenance_sha256,
        }
        expected_configuration_sha256 = _canonical_sha256(configuration)
        feature_sha256 = telemetry_evidence.get("feature_array_sha256")
        expected_evidence = {
            **configuration,
            "configuration_sha256": expected_configuration_sha256,
            "feature_array_sha256": feature_sha256,
            "observed_labels_sha256": array_artifact_sha256(observed_labels),
            "fold_assignment_labels_sha256": array_artifact_sha256(fold_assignment_labels),
            "estimator_device": "cpu",
            "cuda_execution_gate_required": False,
        }
        if (
            not _valid_sha(feature_sha256)
            or configuration_sha256 != expected_configuration_sha256
            or dict(telemetry_evidence) != expected_evidence
        ):
            raise ValueError(
                f"{cell_id} frozen execution differs from exact configuration, "
                "cache provenance, labels, or CPU mode"
            )
    checked_by_path: dict[str, ConfirmatoryArtifactReadback] = {}
    checkpoint_fold_ids: set[int] = set()
    checkpoint_record_keys = {
        "fold_id",
        "status",
        "path",
        "sha256",
        "physical_identity",
        "configuration_sha256",
        "execution_manifest_path",
        "execution_manifest_sha256",
        "execution_manifest_physical_identity",
        "directive",
        "directive_sha256",
        "canonical_working_checkpoint",
        "versioned_outputs",
    }
    canonical_record_keys = {
        "path",
        "sha256",
        "size_bytes",
        "file_id_128",
        "physical_identity",
        "read_only",
    }
    versioned_record_keys = {
        "publication_index",
        "completed_epochs",
        "checkpoint_relative_path",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "checkpoint_physical_identity",
        "commit_manifest_relative_path",
        "commit_manifest_sha256",
        "commit_manifest_size_bytes",
        "commit_manifest_physical_identity",
    }

    def retain(record: ConfirmatoryArtifactReadback) -> None:
        previous = checked_by_path.get(record.relative_path)
        if previous is not None and previous != record:
            raise ValueError(f"{cell_id} checkpoint-tree artifact repeats with different bytes")
        checked_by_path[record.relative_path] = record

    with ExitStack() as held_checkpoint_tree:
        for index, raw in enumerate(records):
            if not isinstance(raw, Mapping) or set(raw) != checkpoint_record_keys:
                raise ValueError(f"{cell_id} checkpoint record {index} is invalid")
            checkpoint_fold = _strict_int(raw["fold_id"], f"{cell_id} checkpoint fold")
            if checkpoint_fold in checkpoint_fold_ids:
                raise ValueError(f"{cell_id} checkpoint manifest repeats a fold")
            checkpoint_fold_ids.add(checkpoint_fold)
            execution_row = fold_execution_by_id.get(checkpoint_fold)
            if execution_row is None:
                raise ValueError(f"{cell_id} checkpoint record {index} lacks telemetry")

            directive = _checkpoint_directive_from_exact_payload(
                raw["directive"],
                role=f"{cell_id} checkpoint record {index}",
            )
            directive_sha256 = raw.get("directive_sha256")
            try:
                expected_configuration_from_directive = json.loads(
                    directive.expected_configuration_json
                )
                expected_metadata_from_directive = json.loads(
                    directive.expected_model_metadata_json
                )
                expected_data_from_directive = json.loads(directive.expected_data_and_split_json)
            except json.JSONDecodeError as error:  # pragma: no cover - validate() guards
                raise ValueError(
                    f"{cell_id} checkpoint directive JSON cannot be decoded"
                ) from error
            if (
                directive.cell_id != cell_id
                or directive.fold_id != checkpoint_fold
                or directive_sha256 != directive.directive_sha256
                or directive.execution_mode != execution_row.get("checkpoint_execution_mode")
                or directive.action != execution_row.get("checkpoint_action")
                or directive.completed_epochs_before_fit
                != execution_row.get("completed_epochs_before_fit")
                or directive.maximum_epochs != int(training["max_epochs"])
                or directive.expected_configuration_sha256 != raw.get("configuration_sha256")
                or directive.expected_configuration_sha256
                != execution_row.get("configuration_sha256")
                or expected_configuration_from_directive
                != dict(cnn_configuration_by_fold[checkpoint_fold])
                or expected_metadata_from_directive != dict(cnn_metadata_by_fold[checkpoint_fold])
                or expected_data_from_directive != dict(cnn_data_hashes_by_fold[checkpoint_fold])
            ):
                raise ValueError(
                    f"{cell_id} checkpoint directive differs from frozen fold execution"
                )

            completed_epochs = _strict_int(
                execution_row.get("completed_epochs"),
                f"{cell_id} completed checkpoint epochs",
            )
            trained_epochs = _strict_int(
                execution_row.get("trained_epochs_this_invocation"),
                f"{cell_id} trained checkpoint epochs",
            )
            relative = raw.get("path")
            execution_manifest_relative = raw.get("execution_manifest_path")
            canonical_working = raw.get("canonical_working_checkpoint")
            if (
                not isinstance(relative, str)
                or not isinstance(execution_manifest_relative, str)
                or not isinstance(canonical_working, Mapping)
                or set(canonical_working) != canonical_record_keys
                or "\\" in relative
                or "\\" in execution_manifest_relative
            ):
                raise ValueError(f"{cell_id} checkpoint record {index} paths are unsafe")
            pure_relative = PurePosixPath(relative)
            pure_execution_manifest = PurePosixPath(execution_manifest_relative)
            expected_execution_manifest_relative = (
                f"checkpoint_execution/fold_{checkpoint_fold:02d}.json"
            )
            expected_canonical_relative = (
                f"cells/{cell_id}/checkpoints/fold_{checkpoint_fold:02d}.pt"
            )
            expected_final_relative = (
                f"checkpoints/fold_{checkpoint_fold:02d}.pt"
                if directive.action == "restore_terminal_checkpoint_without_fit"
                else (
                    f"checkpoint_versions/fold_{checkpoint_fold:02d}/"
                    f"epoch_{completed_epochs:04d}.pt"
                )
            )
            expected_checkpoint_absolute = cell_directory.joinpath(
                *PurePosixPath(expected_final_relative).parts
            ).resolve()
            expected_execution_manifest_absolute = cell_directory.joinpath(
                *PurePosixPath(expected_execution_manifest_relative).parts
            ).resolve()
            if (
                raw.get("status") != "complete"
                or raw.get("configuration_sha256") != execution_row.get("configuration_sha256")
                or raw.get("sha256") != execution_row.get("checkpoint_sha256")
                or raw.get("physical_identity") != execution_row.get("checkpoint_physical_identity")
                or raw.get("execution_manifest_sha256")
                != execution_row.get("checkpoint_execution_manifest_sha256")
                or raw.get("execution_manifest_physical_identity")
                != execution_row.get("checkpoint_execution_manifest_physical_identity")
                or raw.get("versioned_outputs") != execution_row.get("checkpoint_versioned_outputs")
                or canonical_working != execution_row.get("checkpoint_canonical_working")
                or canonical_working.get("path") != expected_canonical_relative
                or canonical_working.get("read_only") is not True
                or pure_relative.as_posix() != relative
                or pure_execution_manifest.as_posix() != execution_manifest_relative
                or any(part in {".", ".."} for part in pure_relative.parts)
                or any(part in {".", ".."} for part in pure_execution_manifest.parts)
                or relative != expected_final_relative
                or execution_manifest_relative != expected_execution_manifest_relative
                or execution_row.get("checkpoint_path") != str(expected_checkpoint_absolute)
                or execution_row.get("checkpoint_execution_manifest_path")
                != str(expected_execution_manifest_absolute)
                or completed_epochs != directive.completed_epochs_before_fit + trained_epochs
            ):
                raise ValueError(f"{cell_id} checkpoint record {index} is unsafe")

            final_record, _held_final = _hold_exact_checkpoint_artifact(
                held_checkpoint_tree,
                cell_directory.joinpath(*pure_relative.parts),
                cell_directory=cell_directory,
                run_root=run_root,
                expected_sha256=raw.get("sha256"),
                expected_size_bytes=execution_row.get("checkpoint_size_bytes"),
                expected_physical_identity=raw.get("physical_identity"),
                role=f"{cell_id} final checkpoint {index}",
                require_read_only=True,
            )
            retain(final_record)
            validate_confirmatory_checkpoint_artifact(
                expected_checkpoint_absolute,
                expected_configuration=cnn_configuration_by_fold[checkpoint_fold],
                expected_model_metadata=cnn_metadata_by_fold[checkpoint_fold],
                expected_data_and_split_sha256=cnn_data_hashes_by_fold[checkpoint_fold],
            )

            canonical_identity = _checkpoint_physical_identity_from_exact_manifest(
                canonical_working.get("physical_identity"),
                role=f"{cell_id} canonical working checkpoint {index}",
            )
            if canonical_working.get("file_id_128") != canonical_identity.file_id_128:
                raise ValueError(f"{cell_id} canonical working checkpoint file ID is inconsistent")
            expected_canonical_absolute = run_root.joinpath(
                *PurePosixPath(expected_canonical_relative).parts
            )
            canonical_record, held_canonical = _hold_exact_checkpoint_artifact(
                held_checkpoint_tree,
                expected_canonical_absolute,
                cell_directory=cell_directory,
                run_root=run_root,
                expected_sha256=canonical_working.get("sha256"),
                expected_size_bytes=canonical_working.get("size_bytes"),
                expected_physical_identity=canonical_working.get("physical_identity"),
                role=f"{cell_id} canonical working checkpoint {index}",
                require_read_only=True,
            )
            if (
                held_canonical.sha256 != _held_final.sha256
                or held_canonical.size_bytes != _held_final.size_bytes
                or (
                    directive.action == "restore_terminal_checkpoint_without_fit"
                    and held_canonical.identity != _held_final.identity
                )
                or (
                    directive.action != "restore_terminal_checkpoint_without_fit"
                    and held_canonical.identity.file_id_128 == _held_final.identity.file_id_128
                )
            ):
                raise ValueError(f"{cell_id} canonical/versioned checkpoint copies differ or alias")
            retain(canonical_record)
            validate_confirmatory_checkpoint_artifact(
                expected_canonical_absolute,
                expected_configuration=cnn_configuration_by_fold[checkpoint_fold],
                expected_model_metadata=cnn_metadata_by_fold[checkpoint_fold],
                expected_data_and_split_sha256=cnn_data_hashes_by_fold[checkpoint_fold],
            )

            execution_manifest_record, held_execution_manifest = _hold_exact_checkpoint_artifact(
                held_checkpoint_tree,
                cell_directory.joinpath(*pure_execution_manifest.parts),
                cell_directory=cell_directory,
                run_root=run_root,
                expected_sha256=raw.get("execution_manifest_sha256"),
                expected_size_bytes=(
                    cast(Mapping[str, Any], raw["execution_manifest_physical_identity"]).get(
                        "size_bytes"
                    )
                    if isinstance(
                        raw.get("execution_manifest_physical_identity"),
                        Mapping,
                    )
                    else None
                ),
                expected_physical_identity=raw.get("execution_manifest_physical_identity"),
                role=f"{cell_id} checkpoint execution manifest {index}",
                require_read_only=True,
            )
            retain(execution_manifest_record)
            execution_manifest_payload = _strict_canonical_ascii_json_bytes(
                held_execution_manifest.payload,
                role=f"{cell_id} checkpoint execution manifest {index}",
            )
            try:
                _require_exact_checkpoint_execution_manifest_payload(
                    execution_manifest_payload,
                    directive,
                    run_directory=run_root.resolve(strict=True),
                )
            except ConfirmatoryCheckpointContractError as error:
                raise ValueError(
                    f"{cell_id} checkpoint execution manifest {index} is invalid"
                ) from error

            raw_versions = raw["versioned_outputs"]
            manifest_versions = execution_manifest_payload.get("versioned_outputs")
            if not isinstance(raw_versions, list) or not isinstance(
                manifest_versions,
                list,
            ):
                raise ValueError(f"{cell_id} checkpoint history is invalid")
            if directive.action == "restore_terminal_checkpoint_without_fit":
                if raw_versions or manifest_versions:
                    raise ValueError(f"{cell_id} restored terminal checkpoint has version history")
            elif len(raw_versions) != 1 or len(manifest_versions) != 1:
                raise ValueError(
                    f"{cell_id} checkpoint history lacks its one fold-boundary publication"
                )

            for version_index, raw_version in enumerate(raw_versions, start=1):
                manifest_version = manifest_versions[version_index - 1]
                if (
                    not isinstance(raw_version, Mapping)
                    or set(raw_version) != versioned_record_keys
                    or not isinstance(manifest_version, Mapping)
                ):
                    raise ValueError(
                        f"{cell_id} checkpoint history record {version_index} is invalid"
                    )
                expected_version_checkpoint = (
                    manifest_version.get("checkpoint")
                    if isinstance(manifest_version, Mapping)
                    else None
                )
                try:
                    version_checkpoint_identity = _checkpoint_file_identity_from_exact_manifest(
                        expected_version_checkpoint,
                        role=(f"{cell_id} versioned checkpoint {version_index}"),
                    )
                except ConfirmatoryCheckpointContractError as error:
                    raise ValueError(
                        f"{cell_id} versioned checkpoint {version_index} identity is invalid"
                    ) from error
                expected_version_projection = {
                    "publication_index": manifest_version.get("publication_index"),
                    "completed_epochs": manifest_version.get("completed_epochs"),
                    "checkpoint_relative_path": manifest_version.get("checkpoint_relative_path"),
                    "checkpoint_sha256": version_checkpoint_identity.sha256,
                    "checkpoint_size_bytes": version_checkpoint_identity.size_bytes,
                    "checkpoint_physical_identity": (
                        version_checkpoint_identity.physical_identity.as_dict()
                    ),
                    "commit_manifest_relative_path": manifest_version.get(
                        "commit_manifest_relative_path"
                    ),
                    "commit_manifest_sha256": manifest_version.get("commit_manifest_sha256"),
                    "commit_manifest_size_bytes": manifest_version.get(
                        "commit_manifest_size_bytes"
                    ),
                    "commit_manifest_physical_identity": manifest_version.get(
                        "commit_manifest_physical_identity"
                    ),
                }
                if dict(raw_version) != expected_version_projection:
                    raise ValueError(
                        f"{cell_id} checkpoint history record {version_index} "
                        "differs from execution manifest"
                    )
                version_relative = raw_version["checkpoint_relative_path"]
                commit_relative = raw_version["commit_manifest_relative_path"]
                if (
                    not isinstance(version_relative, str)
                    or not isinstance(commit_relative, str)
                    or not version_relative.startswith(f"cells/{cell_id}/checkpoint_versions/")
                    or not commit_relative.startswith(f"cells/{cell_id}/checkpoint_versions/")
                ):
                    raise ValueError(f"{cell_id} checkpoint history paths are noncanonical")
                version_path = run_root.joinpath(*PurePosixPath(version_relative).parts)
                commit_path = run_root.joinpath(*PurePosixPath(commit_relative).parts)
                version_record, _held_version = _hold_exact_checkpoint_artifact(
                    held_checkpoint_tree,
                    version_path,
                    cell_directory=cell_directory,
                    run_root=run_root,
                    expected_sha256=raw_version.get("checkpoint_sha256"),
                    expected_size_bytes=raw_version.get("checkpoint_size_bytes"),
                    expected_physical_identity=raw_version.get("checkpoint_physical_identity"),
                    role=f"{cell_id} versioned checkpoint {version_index}",
                    require_read_only=True,
                )
                commit_record, _held_commit = _hold_exact_checkpoint_artifact(
                    held_checkpoint_tree,
                    commit_path,
                    cell_directory=cell_directory,
                    run_root=run_root,
                    expected_sha256=raw_version.get("commit_manifest_sha256"),
                    expected_size_bytes=raw_version.get("commit_manifest_size_bytes"),
                    expected_physical_identity=raw_version.get("commit_manifest_physical_identity"),
                    role=f"{cell_id} checkpoint commit sidecar {version_index}",
                    require_read_only=True,
                )
                retain(version_record)
                retain(commit_record)
    if family == "cnn" and checkpoint_fold_ids != set(range(n_splits)):
        raise ValueError(f"{cell_id} checkpoints do not cover every OOF fold")
    return tuple(checked_by_path[key] for key in sorted(checked_by_path))


def _read_cell_index(
    run: Path,
    plan: ConfirmatoryMatrixPlan,
    *,
    config: Mapping[str, Any],
    corruptions: Mapping[str, Mapping[str, Any]],
    scenarios: Mapping[str, Mapping[str, Any]],
) -> tuple[
    tuple[ConfirmatoryCellReadback, ...],
    list[dict[str, Any]],
    tuple[ConfirmatoryArtifactReadback, ...],
]:
    rows = _read_csv_rows(run / "cell_index.csv", "confirmatory cell index")
    if len(rows) != len(plan.cells):
        raise ValueError("cell index row count differs from frozen matrix")
    by_id = {row.get("cell_id", ""): row for row in rows}
    if (
        "" in by_id
        or len(by_id) != len(rows)
        or set(by_id) != {cell.cell_id for cell in plan.cells}
    ):
        raise ValueError("cell index cell-ID set differs from frozen matrix")
    data = config.get("data")
    oof = config.get("oof")
    training = config.get("training")
    hybrid = config.get("fixed_hybrid")
    original_audit_selection = config.get("original_audit_selection")
    raw_cache_records = config.get("cache_provenance")
    if (
        not isinstance(data, Mapping)
        or not isinstance(oof, Mapping)
        or not isinstance(training, Mapping)
        or not isinstance(hybrid, Mapping)
        or not isinstance(original_audit_selection, Mapping)
        or not isinstance(raw_cache_records, list)
    ):
        raise ValueError(
            "frozen config lacks data/OOF/training/hybrid/cache/original-audit controls"
        )
    cache_records = {
        str(record["id"]): record
        for record in raw_cache_records
        if isinstance(record, Mapping) and isinstance(record.get("id"), str)
    }
    if len(cache_records) != len(raw_cache_records):
        raise ValueError("frozen cache-provenance records are malformed or duplicated")
    confirmatory_input_bindings = _read_json_object(
        run / "confirmatory_input_bindings.json",
        "confirmatory input bindings",
    )
    cnn_preflight = confirmatory_input_bindings.get("cnn_fold_data_and_split_sha256")
    expected_cnn_cell_ids = {
        cell.cell_id for cell in plan.cells if scenarios[cell.scenario_id].get("family") == "cnn"
    }
    if not isinstance(cnn_preflight, Mapping) or set(cnn_preflight) != expected_cnn_cell_ids:
        raise ValueError("confirmatory input bindings lack the exact pre-execution CNN cell set")
    n_splits = int(oof["n_splits"])
    split_seed = int(data["split_seed"])
    hybrid_components = tuple(str(value) for value in hybrid["components"])
    hybrid_weights = tuple(float(value) for value in hybrid["weights"])
    cells: list[ConfirmatoryCellReadback] = []
    outcomes: list[dict[str, Any]] = []
    checked: list[ConfirmatoryArtifactReadback] = []
    oof_bindings: dict[str, _OOFBinding] = {}
    for planned in plan.cells:
        row = by_id[planned.cell_id]
        corruption = corruptions[planned.corruption_cell_id]
        scenario = scenarios[planned.scenario_id]
        status = str(row.get("status", ""))
        if status not in {"completed", "skipped_with_frozen_blocker"}:
            raise ValueError(f"cell index {planned.cell_id} has non-completable status {status!r}")
        expected_values = {
            "outer_fold": planned.outer_fold,
            "corruption_cell_id": planned.corruption_cell_id,
            "corruption_mechanism": str(corruption["mechanism"]),
            "corruption_rate": float(corruption["rate"]),
            "corruption_seed": int(corruption["seed"]),
            "scenario_id": planned.scenario_id,
            "scenario_family": str(scenario["family"]),
            "representation_id": str(scenario["representation_id"]),
            "cache_provenance_id": str(scenario["cache_provenance_id"]),
            "model_seed": planned.model_seed,
            "required": planned.required,
        }
        if (
            _strict_int(row.get("outer_fold"), f"{planned.cell_id} outer_fold")
            != expected_values["outer_fold"]
            or row.get("corruption_cell_id") != expected_values["corruption_cell_id"]
            or row.get("corruption_mechanism") != expected_values["corruption_mechanism"]
            or not math.isclose(
                _strict_float(row.get("corruption_rate"), f"{planned.cell_id} rate"),
                _strict_float(
                    expected_values["corruption_rate"],
                    f"{planned.cell_id} expected rate",
                ),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or _strict_int(row.get("corruption_seed"), f"{planned.cell_id} corruption seed")
            != expected_values["corruption_seed"]
            or row.get("scenario_id") != expected_values["scenario_id"]
            or row.get("scenario_family") != expected_values["scenario_family"]
            or row.get("representation_id") != expected_values["representation_id"]
            or row.get("cache_provenance_id") != expected_values["cache_provenance_id"]
            or _strict_int(row.get("model_seed"), f"{planned.cell_id} model seed")
            != expected_values["model_seed"]
            or _strict_bool(row.get("required"), f"{planned.cell_id} required")
            is not expected_values["required"]
        ):
            raise ValueError(f"cell index identity swap/mismatch for {planned.cell_id}")
        cell_directory = run / "cells" / planned.cell_id
        if not cell_directory.is_dir():
            raise FileNotFoundError(f"cell directory is missing: {planned.cell_id}")
        identity_path = cell_directory / "cell_identity.json"
        identity = _cell_identity(
            _read_json_object(identity_path, f"{planned.cell_id} identity"),
            planned=planned,
            corruption=corruption,
            scenario=scenario,
            config_sha256=plan.config_sha256,
            role=f"cell {planned.cell_id} identity",
        )
        manifest_path = cell_directory / "artifact_manifest.json"
        manifest = _read_hash_manifest(manifest_path, f"{planned.cell_id} artifact manifest")
        expected_paths = set(
            _COMPLETED_CELL_ARTIFACTS if status == "completed" else _SKIPPED_CELL_ARTIFACTS
        )
        cell_checked = list(
            _verify_hash_manifest(
                cell_directory,
                manifest,
                expected_paths=expected_paths,
                role=f"{planned.cell_id} artifact manifest",
                run_root=run,
            )
        )
        manifest_record = _artifact(manifest_path, run)
        cell_checked.append(manifest_record)
        index_manifest_sha = row.get("artifact_manifest_sha256")
        if index_manifest_sha != manifest_record.sha256:
            raise ValueError(f"{planned.cell_id} cell-index artifact manifest SHA differs")
        metrics_sha: str | None = None
        oof_binding: _OOFBinding | None = None
        if status == "completed":
            sample_ids, injected, fixed_hybrid, oof_binding = _validate_oof_and_risks(
                cell_directory,
                cell_id=planned.cell_id,
                corruption=corruption,
                hybrid_components=hybrid_components,
                hybrid_weights=hybrid_weights,
                n_splits=n_splits,
            )
            oof_bindings[planned.cell_id] = oof_binding
            _validate_ranking_and_metrics(
                cell_directory,
                cell_id=planned.cell_id,
                identity=identity,
                sample_ids=sample_ids,
                injected=injected,
                fixed_hybrid=fixed_hybrid,
            )
            checkpoint_records = _validate_checkpoint_and_telemetry(
                cell_directory,
                run_root=run,
                identity=identity,
                n_splits=n_splits,
                split_seed=split_seed,
                scenario=scenario,
                cache_record=cache_records[str(identity["cache_provenance_id"])],
                training=training,
                original_audit_selection=original_audit_selection,
                expected_cnn_data_hashes_by_fold=(
                    cast(Mapping[str, Any], cnn_preflight).get(planned.cell_id)
                    if str(identity["scenario_family"]) == "cnn"
                    else None
                ),
            )
            cell_checked.extend(checkpoint_records)
            metrics_sha = sha256_file(cell_directory / "metrics.json")
            if row.get("metrics_sha256") != metrics_sha:
                raise ValueError(f"{planned.cell_id} cell-index metrics SHA differs")
            outcome = {
                "cell_id": planned.cell_id,
                "required": planned.required,
                "status": "completed",
                "outer_fold": planned.outer_fold,
                "model_seed": planned.model_seed,
                "scenario_id": planned.scenario_id,
                "corruption_cell_id": planned.corruption_cell_id,
                "artifact_manifest_sha256": manifest_record.sha256,
                "metrics_sha256": metrics_sha,
            }
        else:
            if planned.required:
                raise ValueError(f"required cell {planned.cell_id} cannot be skipped")
            blocker = _read_json_object(
                cell_directory / "blocker.json", f"{planned.cell_id} blocker"
            )
            if (
                blocker.get("cell_id") != planned.cell_id
                or blocker.get("frozen_unavailability") is not True
                or not str(blocker.get("blocker", "")).strip()
                or not _valid_sha(blocker.get("blocker_evidence_sha256"))
            ):
                raise ValueError(f"{planned.cell_id} optional blocker evidence is invalid")
            if (
                row.get("frozen_unavailability", "").casefold() != "true"
                or row.get("blocker") != blocker["blocker"]
            ):
                raise ValueError(f"{planned.cell_id} cell-index blocker differs from artifact")
            outcome = {
                "cell_id": planned.cell_id,
                "required": False,
                "status": "skipped_with_frozen_blocker",
                "frozen_unavailability": True,
                "blocker": str(blocker["blocker"]),
            }
        cells.append(
            ConfirmatoryCellReadback(
                cell_id=planned.cell_id,
                status=status,
                outer_fold=planned.outer_fold,
                corruption_cell_id=planned.corruption_cell_id,
                corruption_mechanism=str(corruption["mechanism"]),
                corruption_rate=float(corruption["rate"]),
                corruption_seed=int(corruption["seed"]),
                scenario_id=planned.scenario_id,
                scenario_family=str(scenario["family"]),
                representation_id=str(scenario["representation_id"]),
                cache_provenance_id=str(scenario["cache_provenance_id"]),
                model_seed=planned.model_seed,
                required=planned.required,
                artifact_manifest_sha256=manifest_record.sha256,
                metrics_sha256=metrics_sha,
                oof_identity_sha256=(
                    oof_binding.identity_sha256 if oof_binding is not None else None
                ),
                corruption_mapping_sha256=(
                    oof_binding.corruption_mapping_sha256 if oof_binding is not None else None
                ),
                fold_assignment_sha256=(
                    oof_binding.fold_assignment_sha256 if oof_binding is not None else None
                ),
                checked_artifacts=tuple(cell_checked),
            )
        )
        outcomes.append(outcome)
        checked.extend(cell_checked)
    _validate_cross_cell_oof_bindings(plan, oof_bindings)
    _validate_ensemble_risk_recomputation(run, plan, cells, config)
    return tuple(cells), outcomes, tuple(checked)


def _validate_dual_checkpoint_layout(
    run: Path,
    cells: Sequence[ConfirmatoryCellReadback],
    *,
    n_splits: int,
    require_canonical_names: bool,
) -> None:
    """Reconcile the complete schema-v3 dual-copy checkpoint tree with readback."""

    expected: set[str] = set()
    _ = require_canonical_names
    for cell in cells:
        checkpoint_tree_paths = [
            record.relative_path
            for record in cell.checked_artifacts
            if {
                "checkpoints",
                "checkpoint_versions",
                "checkpoint_execution",
            }.intersection(PurePosixPath(record.relative_path).parts)
        ]
        if cell.status == "completed" and cell.scenario_family == "cnn":
            fold_ids: set[int] = set()
            canonical_folds: set[int] = set()
            execution_folds: set[int] = set()
            for relative in checkpoint_tree_paths:
                final_match = re.fullmatch(
                    rf"cells/{re.escape(cell.cell_id)}/checkpoints/fold_([0-9]{{2}})\.pt",
                    relative,
                )
                version_match = re.fullmatch(
                    rf"cells/{re.escape(cell.cell_id)}/checkpoint_versions/"
                    r"fold_([0-9]{2})/epoch_[0-9]{4}\.(?:pt|commit\.json)",
                    relative,
                )
                execution_match = re.fullmatch(
                    rf"cells/{re.escape(cell.cell_id)}/checkpoint_execution/"
                    r"fold_([0-9]{2})\.json",
                    relative,
                )
                match = final_match or version_match or execution_match
                if match is None:
                    raise ValueError(
                        f"{cell.cell_id} checkpoint-tree path is noncanonical: {relative}"
                    )
                fold_ids.add(int(match.group(1)))
                if final_match is not None:
                    canonical_folds.add(int(final_match.group(1)))
                if execution_match is not None:
                    execution_folds.add(int(execution_match.group(1)))
            if fold_ids != set(range(n_splits)):
                raise ValueError(f"{cell.cell_id} checkpoint tree does not cover every OOF fold")
            if canonical_folds != set(range(n_splits)) or execution_folds != set(range(n_splits)):
                raise ValueError(
                    f"{cell.cell_id} checkpoint tree lacks canonical/execution evidence "
                    "for every OOF fold"
                )
            expected.update(checkpoint_tree_paths)
        elif checkpoint_tree_paths:
            raise ValueError(f"non-CNN cell {cell.cell_id} unexpectedly contains checkpoints")

    discovered: set[str] = set()
    for candidate in run.rglob("*"):
        try:
            candidate_stat = candidate.lstat()
        except OSError as error:
            raise FileNotFoundError(
                f"confirmatory artifact disappeared during checkpoint scan: {candidate}"
            ) from error
        if _is_link_or_reparse(candidate_stat):
            raise ValueError(
                "confirmatory artifact tree contains a symbolic link or reparse point: "
                f"{candidate.relative_to(run).as_posix()}"
            )
        relative_candidate = candidate.relative_to(run).as_posix()
        relative_parts = PurePosixPath(relative_candidate).parts
        if stat.S_ISREG(candidate_stat.st_mode) and {
            "checkpoints",
            "checkpoint_versions",
            "checkpoint_execution",
        }.intersection(relative_parts[:-1]):
            discovered.add(relative_candidate)

    if discovered != expected:
        missing = sorted(expected.difference(discovered))
        extra = sorted(discovered.difference(expected))
        raise ValueError(
            "checkpoint filesystem set differs from frozen canonical CNN folds: "
            f"missing={missing}, extra={extra}"
        )


def _validate_cross_cell_oof_bindings(
    plan: ConfirmatoryMatrixPlan,
    bindings: Mapping[str, _OOFBinding],
) -> None:
    """Prove that scenario/seed changes never change audit inputs or fold assignment."""

    fold_bindings: dict[int, tuple[str, str, str, str, str]] = {}
    corruption_bindings: dict[tuple[int, str], tuple[str, str, str]] = {}
    for cell in plan.cells:
        binding = bindings.get(cell.cell_id)
        if binding is None:
            continue
        fold_value = (
            binding.sample_order_sha256,
            binding.group_order_sha256,
            binding.pre_corruption_label_sha256,
            binding.fold_id_sha256,
            binding.fold_assignment_label_sha256,
        )
        previous_fold = fold_bindings.setdefault(cell.outer_fold, fold_value)
        if previous_fold != fold_value:
            raise ValueError(
                "OOF sample/group/pre-label/fold mapping changes between cells in outer "
                f"fold {cell.outer_fold}"
            )
        corruption_value = (
            binding.pre_corruption_label_sha256,
            binding.observed_label_sha256,
            binding.is_injected_corruption_sha256,
        )
        corruption_key = (cell.outer_fold, cell.corruption_cell_id)
        previous_corruption = corruption_bindings.setdefault(corruption_key, corruption_value)
        if previous_corruption != corruption_value:
            raise ValueError(
                "observed-label corruption mapping changes between scenarios/seeds for "
                f"outer_fold={cell.outer_fold}, corruption={cell.corruption_cell_id}"
            )


def _ensemble_risk_arrays(result: Any) -> dict[str, np.ndarray[Any, Any]]:
    return {
        "predictive_entropy_of_mean": result.entropy_of_mean,
        "mean_pairwise_js_divergence": result.mean_pairwise_js_divergence,
        "variation_ratio": result.variation_ratio,
        "observed_label_probability_variance": (result.observed_label_probability_variance),
        "predicted_class_disagreement": result.predicted_class_disagreement,
    }


def _validate_ensemble_risk_recomputation(
    run: Path,
    plan: ConfirmatoryMatrixPlan,
    cells: Sequence[ConfirmatoryCellReadback],
    config: Mapping[str, Any],
) -> None:
    """Recompute every saved ensemble vector from the frozen member OOF matrices."""

    ensemble_config = config.get("ensemble")
    corruption_config = config.get("corruption")
    if not isinstance(ensemble_config, Mapping) or not isinstance(corruption_config, Mapping):
        raise ValueError("frozen config lacks ensemble/corruption controls")
    raw_members = ensemble_config.get("members")
    raw_corruptions = corruption_config.get("cells")
    if not isinstance(raw_members, list) or not isinstance(raw_corruptions, list):
        raise ValueError("frozen ensemble members/corruption cells are malformed")
    member_keys: tuple[tuple[str, int], ...] = tuple(
        (str(member["scenario_id"]), _strict_int(member["model_seed"], "ensemble seed"))
        for member in raw_members
        if isinstance(member, Mapping)
    )
    if len(member_keys) != len(raw_members) or len(set(member_keys)) != len(member_keys):
        raise ValueError("frozen ensemble member identities are invalid or duplicate")
    if len(member_keys) < 2:
        raise ValueError("confirmatory ensemble requires at least two frozen members")
    corruption_ids = tuple(
        str(value["id"]) for value in raw_corruptions if isinstance(value, Mapping)
    )
    if len(corruption_ids) != len(raw_corruptions):
        raise ValueError("frozen corruption cells are malformed")
    primary_risk = str(ensemble_config.get("primary_risk", ""))
    status_by_id = {cell.cell_id: cell.status for cell in cells}

    for outer_fold in sorted({cell.outer_fold for cell in plan.cells}):
        for corruption_id in corruption_ids:
            group = [
                cell
                for cell in plan.cells
                if cell.outer_fold == outer_fold and cell.corruption_cell_id == corruption_id
            ]
            member_cells = [
                cell for cell in group if (cell.scenario_id, cell.model_seed) in set(member_keys)
            ]
            actual_member_keys = {(cell.scenario_id, cell.model_seed) for cell in member_cells}
            if actual_member_keys != set(member_keys) or len(member_cells) != len(member_keys):
                raise ValueError(
                    "matrix lacks the exact frozen ensemble members for "
                    f"outer_fold={outer_fold}, corruption={corruption_id}"
                )
            if any(status_by_id.get(cell.cell_id) != "completed" for cell in member_cells):
                raise ValueError(
                    "frozen ensemble member did not complete for "
                    f"outer_fold={outer_fold}, corruption={corruption_id}"
                )

            probabilities: list[np.ndarray[Any, Any]] = []
            expected_sample_ids: tuple[str, ...] | None = None
            expected_observed: np.ndarray[Any, Any] | None = None
            for member in member_cells:
                path = run / "cells" / member.cell_id / "oof_evidence.npz"
                try:
                    with np.load(path, allow_pickle=False) as payload:
                        raw_ids = payload["sample_ids"]
                        if raw_ids.dtype.kind not in {"U", "S"}:
                            raise ValueError("sample IDs require pickle")
                        sample_ids = tuple(str(value) for value in raw_ids.tolist())
                        observed = np.asarray(payload["observed_label"], dtype=np.int64)
                        member_probabilities = np.asarray(
                            payload["probabilities"], dtype=np.float64
                        )
                except (OSError, KeyError, ValueError) as error:
                    raise ValueError(
                        f"cannot re-read ensemble member OOF evidence: {member.cell_id}"
                    ) from error
                if expected_sample_ids is None:
                    expected_sample_ids = sample_ids
                    expected_observed = observed
                else:
                    if expected_observed is None:
                        raise RuntimeError("ensemble observed-label reference is missing")
                    if sample_ids != expected_sample_ids or not np.array_equal(
                        observed, expected_observed
                    ):
                        raise ValueError(
                            "ensemble member sample/observed-label mapping is inconsistent "
                            f"for outer_fold={outer_fold}, corruption={corruption_id}"
                        )
                probabilities.append(member_probabilities)
            if expected_observed is None or expected_sample_ids is None:
                raise ValueError("frozen ensemble has no readable members")

            result = ensemble_disagreement(
                probabilities,
                observed_labels=expected_observed,
                class_order=CLASS_ORDER,
            )
            expected_risks = _ensemble_risk_arrays(result)
            expected_primary = predeclared_ensemble_risk(
                result,
                primary_risk=cast(Any, primary_risk),
            )
            for cell in group:
                if status_by_id.get(cell.cell_id) != "completed":
                    continue
                risk_path = run / "cells" / cell.cell_id / "risk_scores.npz"
                try:
                    with np.load(risk_path, allow_pickle=False) as payload:
                        saved_mean = np.asarray(
                            payload["ensemble_mean_probabilities"], dtype=np.float64
                        )
                        saved_primary = np.asarray(
                            payload["ensemble_disagreement"], dtype=np.float64
                        )
                        saved_risks = {
                            name: np.asarray(payload[f"ensemble_{name}"], dtype=np.float64)
                            for name in _ENSEMBLE_RISK_NAMES
                        }
                except (OSError, KeyError, ValueError) as error:
                    raise ValueError(f"cannot re-read ensemble risks for {cell.cell_id}") from error
                if not np.allclose(
                    saved_mean,
                    result.averaged_probabilities,
                    rtol=0.0,
                    atol=1e-12,
                ) or not np.allclose(saved_primary, expected_primary, rtol=0.0, atol=1e-12):
                    raise ValueError(
                        f"{cell.cell_id} primary ensemble evidence differs from member OOF"
                    )
                for name, expected_values in expected_risks.items():
                    if not np.allclose(saved_risks[name], expected_values, rtol=0.0, atol=1e-12):
                        raise ValueError(f"{cell.cell_id} ensemble {name} differs from member OOF")


def _exact_string_set(value: Any, expected: set[str], role: str) -> None:
    if not isinstance(value, list) or {str(item) for item in value} != expected:
        raise ValueError(f"{role} differs from the frozen set")


def _exact_fold_list(value: Any, expected: tuple[int, ...], role: str) -> None:
    if not isinstance(value, list) or tuple(_strict_int(item, role) for item in value) != expected:
        raise ValueError(f"{role} differs from the frozen fold rotation")


def _validate_frozen_feature_provenance(
    run: Path,
    *,
    config: Mapping[str, Any],
    plan: ConfirmatoryMatrixPlan,
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Bind every rotation's cache provenance to config and OOF sample order."""

    raw_cache_records = config.get("cache_provenance")
    if not isinstance(raw_cache_records, list):
        raise ValueError("frozen config cache provenance is malformed")
    available_by_representation: dict[str, Mapping[str, Any]] = {}
    for raw in raw_cache_records:
        if not isinstance(raw, Mapping):
            raise ValueError("frozen cache provenance record is malformed")
        if raw.get("status") != "available":
            continue
        representation_id = str(raw.get("representation_id", ""))
        if not representation_id or representation_id in available_by_representation:
            raise ValueError("available cache provenance representations are invalid/duplicate")
        available_by_representation[representation_id] = raw

    folds = tuple(sorted({cell.outer_fold for cell in plan.cells}))
    sample_order_by_fold: dict[int, str] = {}
    for outer_fold in folds:
        representative = next(
            (cell for cell in plan.cells if cell.outer_fold == outer_fold and cell.required),
            None,
        )
        if representative is None:
            raise ValueError(f"outer fold {outer_fold} has no required representative cell")
        oof_path = run / "cells" / representative.cell_id / "oof_evidence.npz"
        try:
            with np.load(oof_path, allow_pickle=False) as payload:
                raw_ids = payload["sample_ids"]
                if raw_ids.dtype.kind not in {"U", "S"}:
                    raise ValueError("sample IDs require pickle")
                sample_ids = [str(value) for value in raw_ids.tolist()]
        except (OSError, KeyError, ValueError) as error:
            raise ValueError(
                f"cannot bind frozen cache provenance to OOF sample order for fold {outer_fold}"
            ) from error
        sample_order_by_fold[outer_fold] = _canonical_sha256(sample_ids)

    artifact = _read_json_object(
        run / "frozen_feature_provenance.json", "frozen feature provenance"
    )
    expected_top_keys = {
        "schema_version",
        "status",
        "confirmatory_config_semantic_sha256",
        "matrix_plan_config_sha256",
        "representations",
    }
    if set(artifact) != expected_top_keys or artifact.get("schema_version") != 1:
        raise ValueError("frozen feature provenance has an invalid top-level schema")
    if (
        artifact.get("status") != "completed"
        or artifact.get("confirmatory_config_semantic_sha256") != plan.config_sha256
        or artifact.get("matrix_plan_config_sha256") != plan.config_sha256
    ):
        raise ValueError("frozen feature provenance is bound to a different plan/config")
    raw_representations = artifact.get("representations")
    if not isinstance(raw_representations, Mapping) or set(raw_representations) != set(
        available_by_representation
    ):
        raise ValueError(
            "frozen feature provenance does not exactly cover available representations"
        )

    validated: dict[str, dict[str, Mapping[str, Any]]] = {}
    for representation_id, cache_record in available_by_representation.items():
        representation = raw_representations[representation_id]
        if not isinstance(representation, Mapping) or set(representation) != {"rotations"}:
            raise ValueError(
                f"frozen feature provenance representation {representation_id} is invalid"
            )
        rotations = representation["rotations"]
        expected_fold_keys = {str(value) for value in folds}
        if not isinstance(rotations, Mapping) or set(rotations) != expected_fold_keys:
            raise ValueError(f"frozen feature provenance rotations differ for {representation_id}")
        validated_rotations: dict[str, Mapping[str, Any]] = {}
        expected_static = {
            "cache_provenance_id": str(cache_record["id"]),
            "representation_id": representation_id,
            "cache_file_sha256": cache_record["cache_file_sha256"],
            "sidecar_semantic_sha256": cache_record["sidecar_semantic_sha256"],
            "sample_order_sha256": str(cache_record["sample_order_sha256"]),
            "manifest_sha256": str(cache_record["manifest_sha256"]),
            "encoder_identifier": str(cache_record["encoder_identifier"]),
            "encoder_metadata_sha256": str(cache_record["encoder_metadata_sha256"]),
            "weight_identifier": str(cache_record["weight_identifier"]),
            "weights_sha256": str(cache_record["weights_sha256"]),
            "preprocessing_identifier": str(cache_record["preprocessing_identifier"]),
            "preprocessing_sha256": str(cache_record["preprocessing_sha256"]),
            "input_variant": str(cache_record["input_variant"]),
        }
        for fold_key, raw_rotation in rotations.items():
            if not isinstance(raw_rotation, Mapping):
                raise ValueError(f"frozen feature provenance rotation {fold_key} is not a mapping")
            expected = {
                **expected_static,
                "audit_sample_order_sha256": sample_order_by_fold[int(fold_key)],
            }
            if dict(raw_rotation) != expected:
                raise ValueError(
                    "frozen cache provenance differs from config/OOF sample order for "
                    f"{representation_id}, fold {fold_key}"
                )
            validated_rotations[str(fold_key)] = raw_rotation
        validated[representation_id] = validated_rotations
    return validated


def _two_sided_bootstrap_probability(differences: np.ndarray[Any, Any]) -> float:
    non_positive = float(np.mean(differences <= 0.0))
    non_negative = float(np.mean(differences >= 0.0))
    return min(1.0, 2.0 * min(non_positive, non_negative))


def _holm_adjusted_p_values(
    records: Sequence[tuple[str, str, float]],
) -> dict[str, float]:
    """Return deterministic Holm-adjusted p-values within each frozen family."""

    result: dict[str, float] = {}
    families = sorted({family for _, family, _ in records})
    for family in families:
        family_records = sorted(
            (
                (comparison_id, raw_p)
                for comparison_id, record_family, raw_p in records
                if record_family == family
            ),
            key=lambda value: (value[1], value[0]),
        )
        running = 0.0
        count = len(family_records)
        for index, (comparison_id, raw_p) in enumerate(family_records):
            running = max(running, min(1.0, (count - index) * raw_p))
            result[comparison_id] = running
    return result


def _paired_selected_injected_event_count(
    run: Path,
    plan: ConfirmatoryMatrixPlan,
    definition: Mapping[str, Any],
    status_by_id: Mapping[str, str],
) -> int:
    operand = definition.get("operand_a")
    if not isinstance(operand, Mapping):
        raise ValueError("paired comparison operand_a is malformed")
    outer_selector = operand.get("outer_fold")
    corruption_selector = operand.get("corruption_cell")
    selected_folds = (
        sorted({cell.outer_fold for cell in plan.cells})
        if outer_selector == "all_matched"
        else [_strict_int(outer_selector, "paired operand outer fold")]
    )
    selected_corruptions = (
        sorted({cell.corruption_cell_id for cell in plan.cells})
        if corruption_selector == "all_matched"
        else [str(corruption_selector)]
    )
    total = 0
    for outer_fold in selected_folds:
        for corruption_id in selected_corruptions:
            representative = next(
                (
                    cell
                    for cell in plan.cells
                    if cell.outer_fold == outer_fold
                    and cell.corruption_cell_id == corruption_id
                    and cell.required
                    and status_by_id.get(cell.cell_id) == "completed"
                ),
                None,
            )
            if representative is None:
                raise ValueError(
                    "paired comparison cannot bind selected corruption events to a "
                    "completed required cell"
                )
            try:
                with np.load(
                    run / "cells" / representative.cell_id / "oof_evidence.npz",
                    allow_pickle=False,
                ) as payload:
                    injected = payload["is_injected_corruption"]
                    if not np.issubdtype(injected.dtype, np.bool_):
                        raise ValueError("corruption flags are not boolean")
                    total += int(np.asarray(injected, dtype=bool).sum())
            except (OSError, KeyError, ValueError) as error:
                raise ValueError(
                    "paired comparison cannot read selected injected-event evidence"
                ) from error
    return total


def _paired_operand_arrays(
    run: Path,
    plan: ConfirmatoryMatrixPlan,
    operand: Mapping[str, Any],
    status_by_id: Mapping[str, str],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Load one frozen operand as aligned event, score, and tagged-group vectors."""

    scenario_id = str(operand.get("scenario_id", ""))
    risk_id = str(operand.get("risk_id", ""))
    outer_selector = operand.get("outer_fold")
    corruption_selector = operand.get("corruption_cell")
    selected = [
        cell
        for cell in plan.cells
        if cell.scenario_id == scenario_id
        and (outer_selector == "all_matched" or cell.outer_fold == outer_selector)
        and (corruption_selector == "all_matched" or cell.corruption_cell_id == corruption_selector)
    ]
    if not selected or any(status_by_id.get(cell.cell_id) != "completed" for cell in selected):
        raise ValueError("paired operand does not resolve only to completed frozen cells")
    injected_parts: list[np.ndarray[Any, Any]] = []
    score_parts: list[np.ndarray[Any, Any]] = []
    group_parts: list[np.ndarray[Any, Any]] = []
    for cell in selected:
        cell_directory = run / "cells" / cell.cell_id
        try:
            with np.load(cell_directory / "oof_evidence.npz", allow_pickle=False) as oof:
                groups_raw = oof["group_ids"]
                injected_raw = oof["is_injected_corruption"]
                if groups_raw.dtype.kind not in {"U", "S"} or not np.issubdtype(
                    injected_raw.dtype, np.bool_
                ):
                    raise ValueError("paired OOF identities/events have unsafe dtypes")
                groups = np.asarray(
                    [f"fold_{cell.outer_fold}::{value}" for value in groups_raw.tolist()],
                    dtype=np.str_,
                )
                injected = np.asarray(injected_raw, dtype=bool)
            with np.load(cell_directory / "risk_scores.npz", allow_pickle=False) as risks:
                score_raw = risks[risk_id]
                if not np.issubdtype(score_raw.dtype, np.floating):
                    raise ValueError("paired risk scores are not floating point")
                scores = np.asarray(score_raw, dtype=np.float64)
        except (OSError, KeyError, ValueError) as error:
            raise ValueError(
                f"paired operand cannot read OOF/risk evidence for {cell.cell_id}"
            ) from error
        if groups.shape != injected.shape or scores.shape != injected.shape:
            raise ValueError("paired operand group/event/score arrays are misaligned")
        injected_parts.append(injected)
        score_parts.append(scores)
        group_parts.append(groups)
    return (
        np.concatenate(injected_parts),
        np.concatenate(score_parts),
        np.concatenate(group_parts),
    )


def _recompute_paired_bootstrap_metrics(
    group_draws: np.ndarray[Any, Any],
    injected: np.ndarray[Any, Any],
    scores_a: np.ndarray[Any, Any],
    scores_b: np.ndarray[Any, Any],
    group_tags: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    members = {str(group): np.flatnonzero(group_tags == group) for group in np.unique(group_tags)}
    valid_mask = np.zeros(len(group_draws), dtype=bool)
    metric_a: list[float] = []
    metric_b: list[float] = []
    for draw_index, draw in enumerate(group_draws):
        try:
            indices = np.concatenate([members[str(group)] for group in draw])
        except KeyError as error:
            raise ValueError(
                "paired group draw names a group absent from operand arrays"
            ) from error
        value_a = average_precision(injected[indices], scores_a[indices])
        value_b = average_precision(injected[indices], scores_b[indices])
        if value_a is None or value_b is None:
            continue
        if not math.isfinite(value_a) or not math.isfinite(value_b):
            continue
        valid_mask[draw_index] = True
        metric_a.append(value_a)
        metric_b.append(value_b)
    return (
        valid_mask,
        np.asarray(metric_a, dtype=np.float64),
        np.asarray(metric_b, dtype=np.float64),
    )


def _validate_paired_statistics(
    run: Path,
    *,
    config: Mapping[str, Any],
    plan: ConfirmatoryMatrixPlan,
    status_by_id: Mapping[str, str],
) -> None:
    """Recompute paired-bootstrap summaries and frozen Holm families from NPZ arrays."""

    statistics_config = config.get("statistics")
    data_config = config.get("data")
    raw_scenarios = config.get("scenarios")
    if (
        not isinstance(statistics_config, Mapping)
        or not isinstance(data_config, Mapping)
        or not isinstance(raw_scenarios, list)
    ):
        raise ValueError("frozen paired-statistics controls are malformed")
    definitions = statistics_config.get("preregistered_paired_comparisons")
    if not isinstance(definitions, list) or not definitions:
        raise ValueError("frozen paired comparisons are malformed")
    definition_by_id = {
        str(value["comparison_id"]): value for value in definitions if isinstance(value, Mapping)
    }
    if len(definition_by_id) != len(definitions):
        raise ValueError("frozen paired comparison IDs are invalid or duplicate")
    iterations = _strict_int(
        statistics_config.get("paired_group_bootstrap_iterations"),
        "frozen paired bootstrap iterations",
    )
    bootstrap_seed = _strict_int(statistics_config.get("bootstrap_seed"), "frozen bootstrap seed")
    bootstrap_path = run / "paired_bootstrap_evidence.npz"
    statistics = _read_json_object(run / "paired_statistics.json", "paired statistics")
    if (
        statistics.get("schema_version") != 1
        or statistics.get("config_semantic_sha256") != plan.config_sha256
        or statistics.get("outer_folds") != list(sorted({cell.outer_fold for cell in plan.cells}))
        or statistics.get("paired_unit") != data_config.get("group_unit")
        or statistics.get("bootstrap_iterations") != iterations
        or statistics.get("bootstrap_seed") != bootstrap_seed
        or statistics.get("bootstrap_evidence_path") != bootstrap_path.name
        or statistics.get("bootstrap_evidence_sha256") != sha256_file(bootstrap_path)
    ):
        raise ValueError("paired statistics top-level bindings are invalid")
    raw_results = statistics.get("comparisons")
    if not isinstance(raw_results, list):
        raise ValueError("paired statistics comparison results are missing")
    result_by_id = {
        str(value.get("comparison_id", "")): value
        for value in raw_results
        if isinstance(value, Mapping)
    }
    if len(result_by_id) != len(raw_results) or set(result_by_id) != set(definition_by_id):
        raise ValueError("paired statistics results differ from frozen comparison IDs")

    expected_npz_keys = {"bootstrap_group_universe", "bootstrap_group_draws"}
    for comparison_id in definition_by_id:
        expected_npz_keys.update(
            {
                f"valid_draw_mask__{comparison_id}",
                f"metric_a__{comparison_id}",
                f"metric_b__{comparison_id}",
                f"differences__{comparison_id}",
            }
        )
    try:
        with np.load(bootstrap_path, allow_pickle=False) as payload:
            if set(payload.files) != expected_npz_keys:
                raise ValueError("paired bootstrap NPZ keys differ from frozen comparisons")
            group_universe = payload["bootstrap_group_universe"]
            group_draws = payload["bootstrap_group_draws"]
            if (
                group_universe.dtype.kind not in {"U", "S"}
                or group_universe.ndim != 1
                or len(group_universe) < 2
                or len(set(str(value) for value in group_universe.tolist())) != len(group_universe)
                or group_draws.dtype.kind not in {"U", "S"}
                or group_draws.ndim != 2
                or group_draws.shape != (iterations, len(group_universe))
                or np.any(group_draws == "")
            ):
                raise ValueError("paired bootstrap shared source-group draws are invalid")
            selector_pairs = {
                (
                    value["operand_a"]["outer_fold"],
                    value["operand_a"]["corruption_cell"],
                )
                for value in definitions
                if isinstance(value, Mapping) and isinstance(value.get("operand_a"), Mapping)
            }
            if len(selector_pairs) != 1:
                raise ValueError(
                    "shared paired bootstrap draws require one matched fold/corruption selector"
                )
            outer_selector, _corruption_selector = next(iter(selector_pairs))
            selected_outer_folds = (
                sorted({cell.outer_fold for cell in plan.cells})
                if outer_selector == "all_matched"
                else [_strict_int(outer_selector, "paired shared outer fold")]
            )
            expected_group_universe: list[str] = []
            for outer_fold in selected_outer_folds:
                representative = next(
                    cell for cell in plan.cells if cell.outer_fold == outer_fold and cell.required
                )
                with np.load(
                    run / "cells" / representative.cell_id / "oof_evidence.npz",
                    allow_pickle=False,
                ) as oof:
                    groups = tuple(str(value) for value in oof["group_ids"].tolist())
                expected_group_universe.extend(
                    f"fold_{outer_fold}::{group}" for group in sorted(set(groups))
                )
            if tuple(str(value) for value in group_universe.tolist()) != tuple(
                expected_group_universe
            ):
                raise ValueError("paired bootstrap group universe differs from OOF groups")
            rng = np.random.default_rng(bootstrap_seed)
            expected_draws = np.stack(
                [
                    rng.choice(group_universe, size=len(group_universe), replace=True)
                    for _ in range(iterations)
                ]
            )
            if not np.array_equal(group_draws, expected_draws):
                raise ValueError("paired bootstrap group draws differ from frozen seed")
            arrays = {
                comparison_id: (
                    np.asarray(payload[f"valid_draw_mask__{comparison_id}"]),
                    np.asarray(payload[f"metric_a__{comparison_id}"], dtype=np.float64),
                    np.asarray(payload[f"metric_b__{comparison_id}"], dtype=np.float64),
                    np.asarray(payload[f"differences__{comparison_id}"], dtype=np.float64),
                )
                for comparison_id in definition_by_id
            }
    except (OSError, KeyError, ValueError) as error:
        if isinstance(error, ValueError) and "paired bootstrap" in str(error):
            raise
        raise ValueError("paired bootstrap NPZ is invalid or pickle-dependent") from error

    optional_scenarios = {
        str(value["id"]): value
        for value in raw_scenarios
        if isinstance(value, Mapping) and value.get("required") is False
    }
    holm_inputs: list[tuple[str, str, float]] = []
    completed_results: dict[str, Mapping[str, Any]] = {}
    for comparison_id, definition in definition_by_id.items():
        result = result_by_id[comparison_id]
        for field in ("metric", "operand_a", "operand_b", "direction", "holm_family"):
            if result.get(field) != definition.get(field):
                raise ValueError(f"paired comparison {comparison_id} differs from frozen {field}")
        valid_draw_mask, metric_a, metric_b, differences = arrays[comparison_id]
        if valid_draw_mask.shape != (iterations,) or not np.issubdtype(
            valid_draw_mask.dtype, np.bool_
        ):
            raise ValueError(f"paired comparison {comparison_id} valid-draw mask is invalid")
        valid_iterations = int(valid_draw_mask.sum())
        status = result.get("status")
        optional_method_ids = {
            str(value.get("scenario_id"))
            for value in (definition.get("operand_a"), definition.get("operand_b"))
            if isinstance(value, Mapping)
        }.intersection(optional_scenarios)
        optional_unavailable = bool(optional_method_ids) and all(
            status_by_id.get(cell.cell_id) == "skipped_with_frozen_blocker"
            for cell in plan.cells
            if cell.scenario_id in optional_method_ids
        )
        selected_event_count = _paired_selected_injected_event_count(
            run,
            plan,
            definition,
            status_by_id,
        )
        if result.get("selected_injected_event_count") != selected_event_count:
            raise ValueError(
                f"paired comparison {comparison_id} injected-event count differs from OOF"
            )
        if status == "not_estimable_frozen_optional_blocker":
            if (
                not optional_unavailable
                or result.get("paired_unit") != data_config.get("group_unit")
                or result.get("bootstrap_seed") != bootstrap_seed
                or valid_iterations != 0
                or any(values.size for values in (metric_a, metric_b, differences))
                or result.get("valid_iterations") != 0
                or result.get("requested_iterations") != iterations
                or result.get("observed_delta") is not None
                or result.get("ci_low") is not None
                or result.get("ci_high") is not None
                or result.get("probability_positive") is not None
                or result.get("raw_p") is not None
                or result.get("holm_adjusted_p") is not None
                or result.get("frozen_unavailability") is not True
                or not str(result.get("blocker", "")).strip()
            ):
                raise ValueError(
                    f"paired comparison {comparison_id} has invalid optional blocker evidence"
                )
            availability_hashes = {
                str(optional_scenarios[value].get("availability_audit_sha256"))
                for value in optional_method_ids
            }
            if result.get("availability_audit_sha256") not in availability_hashes:
                raise ValueError(
                    f"paired comparison {comparison_id} optional blocker hash is invalid"
                )
            continue
        if status == "not_applicable_zero_event":
            if (
                selected_event_count != 0
                or result.get("paired_unit") != data_config.get("group_unit")
                or result.get("bootstrap_seed") != bootstrap_seed
                or result.get("requested_iterations") != iterations
                or result.get("valid_iterations") != 0
                or result.get("observed_delta") is not None
                or result.get("ci_low") is not None
                or result.get("ci_high") is not None
                or result.get("probability_positive") is not None
                or result.get("raw_p") is not None
                or result.get("holm_adjusted_p") is not None
                or valid_iterations != 0
                or any(values.size for values in (metric_a, metric_b, differences))
            ):
                raise ValueError(
                    f"paired comparison {comparison_id} zero-event status is not evidenced"
                )
            continue
        if status != "completed" or optional_unavailable or selected_event_count == 0:
            raise ValueError(f"paired comparison {comparison_id} did not validly complete")
        operand_a = definition.get("operand_a")
        operand_b = definition.get("operand_b")
        if (
            definition.get("metric") != "average_precision"
            or not isinstance(operand_a, Mapping)
            or not isinstance(operand_b, Mapping)
        ):
            raise ValueError(f"paired comparison {comparison_id} lacks a recomputable AP operand")
        injected_a, scores_a, groups_a = _paired_operand_arrays(run, plan, operand_a, status_by_id)
        injected_b, scores_b, groups_b = _paired_operand_arrays(run, plan, operand_b, status_by_id)
        if not np.array_equal(injected_a, injected_b) or not np.array_equal(groups_a, groups_b):
            raise ValueError(f"paired comparison {comparison_id} operands are not exactly aligned")
        if set(str(value) for value in np.unique(groups_a)) != set(
            str(value) for value in group_universe.tolist()
        ):
            raise ValueError(
                f"paired comparison {comparison_id} group universe differs from operands"
            )
        recomputed_mask, recomputed_a, recomputed_b = _recompute_paired_bootstrap_metrics(
            group_draws,
            injected_a,
            scores_a,
            scores_b,
            groups_a,
        )
        if (
            not np.array_equal(valid_draw_mask, recomputed_mask)
            or not np.allclose(metric_a, recomputed_a, rtol=0.0, atol=1e-12)
            or not np.allclose(metric_b, recomputed_b, rtol=0.0, atol=1e-12)
        ):
            raise ValueError(
                f"paired comparison {comparison_id} metric arrays differ from OOF risks"
            )
        if valid_iterations < 1 or any(
            values.shape != (valid_iterations,) or not np.isfinite(values).all()
            for values in (metric_a, metric_b, differences)
        ):
            raise ValueError(f"paired comparison {comparison_id} arrays are invalid")
        if np.all(metric_a == 0.0) and np.all(metric_b == 0.0):
            raise ValueError(f"paired comparison {comparison_id} is an all-zero placeholder")
        if not np.allclose(differences, metric_a - metric_b, rtol=0.0, atol=1e-12):
            raise ValueError(f"paired comparison {comparison_id} is not paired A-minus-B")
        observed_delta = float(np.mean(differences))
        ci_low, ci_high = (float(value) for value in np.quantile(differences, (0.025, 0.975)))
        probability_positive = float(np.mean(differences > 0.0) + 0.5 * np.mean(differences == 0.0))
        raw_p = _two_sided_bootstrap_probability(differences)
        expected_scalars = {
            "observed_delta": observed_delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "probability_positive": probability_positive,
            "raw_p": raw_p,
        }
        if (
            result.get("paired_unit") != data_config.get("group_unit")
            or result.get("bootstrap_seed") != bootstrap_seed
            or result.get("requested_iterations") != iterations
            or result.get("valid_iterations") != valid_iterations
            or any(
                not math.isclose(
                    _strict_float(result.get(field), f"{comparison_id} {field}"),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for field, expected in expected_scalars.items()
            )
        ):
            raise ValueError(f"paired comparison {comparison_id} summary differs from NPZ")
        holm_inputs.append((comparison_id, str(definition["holm_family"]), raw_p))
        completed_results[comparison_id] = result

    adjusted = _holm_adjusted_p_values(holm_inputs)
    for comparison_id, expected in adjusted.items():
        result = completed_results[comparison_id]
        if not math.isclose(
            _strict_float(result.get("holm_adjusted_p"), f"{comparison_id} Holm-adjusted p"),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"paired comparison {comparison_id} Holm correction is invalid")


def _valid_probability_matrix(
    values: np.ndarray[Any, Any],
    *,
    expected_shape: tuple[int, ...],
) -> bool:
    return bool(
        values.shape == expected_shape
        and np.isfinite(values).all()
        and np.all(values >= 0.0)
        and np.all(values <= 1.0)
        and np.allclose(values.sum(axis=-1), 1.0, rtol=0.0, atol=1e-8)
    )


def _strict_bool_array(values: np.ndarray[Any, Any], role: str) -> np.ndarray[Any, Any]:
    if not np.issubdtype(values.dtype, np.bool_):
        raise ValueError(f"{role} must be a native boolean array")
    return np.asarray(values, dtype=bool)


def _strict_integer_array(values: np.ndarray[Any, Any], role: str) -> np.ndarray[Any, Any]:
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{role} must be a native integer array")
    return np.asarray(values, dtype=np.int64)


def _strict_float_array(values: np.ndarray[Any, Any], role: str) -> np.ndarray[Any, Any]:
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"{role} must be a native floating-point array")
    return np.asarray(values, dtype=np.float64)


def _restoration_key(outer_fold: int, corruption_cell_id: str) -> str:
    return f"fold_{outer_fold}__{corruption_cell_id}"


def _validate_restoration_evidence(
    run: Path,
    *,
    config: Mapping[str, Any],
    plan: ConfirmatoryMatrixPlan,
) -> None:
    """Recompute four-condition downstream evidence and exact equal-budget restoration."""

    restoration = config.get("restoration")
    corruption = config.get("corruption")
    if not isinstance(restoration, Mapping) or not isinstance(corruption, Mapping):
        raise ValueError("frozen restoration/corruption controls are malformed")
    raw_corruptions = corruption.get("cells")
    if not isinstance(raw_corruptions, list):
        raise ValueError("frozen restoration corruption cells are malformed")
    active_corruptions = [
        str(value["id"])
        for value in raw_corruptions
        if isinstance(value, Mapping) and float(value["rate"]) > 0.0
    ]
    if not active_corruptions:
        active_corruptions = [
            str(value["id"]) for value in raw_corruptions if isinstance(value, Mapping)
        ]
    folds = tuple(sorted({cell.outer_fold for cell in plan.cells}))
    scenario_id = str(restoration.get("scenario_id", ""))
    model_seed = _strict_int(restoration.get("model_seed"), "restoration model seed")
    review_budget = _strict_float(restoration.get("review_budget"), "restoration budget")
    random_repeats = _strict_int(restoration.get("random_repeats"), "restoration random repeats")
    random_seed = _strict_int(restoration.get("random_seed"), "restoration random seed")
    conditions = [str(value) for value in restoration.get("conditions", [])]
    if set(conditions) != _RESTORATION_CONDITIONS:
        raise ValueError("frozen restoration conditions are malformed")

    evidence_path = run / "restoration_evidence.npz"
    metrics = _read_json_object(run / "restoration_metrics.json", "restoration aggregate")
    expected_top = {
        "schema_version": 1,
        "status": "completed",
        "config_semantic_sha256": plan.config_sha256,
        "outer_folds": list(folds),
        "scenario_id": scenario_id,
        "model_seed": model_seed,
        "representation_id": restoration.get("representation_id"),
        "ranking_method": restoration.get("ranking_method"),
        "review_budget": review_budget,
        "random_repeats": random_repeats,
        "random_seed": random_seed,
        "conditions": conditions,
        "evidence_path": evidence_path.name,
        "evidence_sha256": sha256_file(evidence_path),
    }
    for field, expected in expected_top.items():
        if metrics.get(field) != expected:
            raise ValueError(f"restoration aggregate {field} differs from frozen evidence")
    raw_rotations = metrics.get("rotations")
    if not isinstance(raw_rotations, list):
        raise ValueError("restoration aggregate lacks per-rotation results")
    rotation_by_key = {
        (value.get("outer_fold"), value.get("corruption_cell_id")): value
        for value in raw_rotations
        if isinstance(value, Mapping)
    }
    expected_rotation_keys = {
        (outer_fold, corruption_id) for outer_fold in folds for corruption_id in active_corruptions
    }
    if len(rotation_by_key) != len(raw_rotations) or set(rotation_by_key) != (
        expected_rotation_keys
    ):
        raise ValueError("restoration rotations differ from frozen fold/corruption matrix")

    suffixes = {
        "audit_sample_ids",
        "audit_group_ids",
        "pre_corruption_label",
        "observed_label",
        "is_injected_corruption",
        "guided_reviewed_mask",
        "guided_restored_mask",
        "guided_restored_label",
        "random_reviewed_mask",
        "random_restored_mask",
        "random_restored_label",
        "final_sample_ids",
        "final_group_ids",
        "final_pre_corruption_label",
        "final_observed_label",
        "final_is_injected_corruption",
        "probabilities__uncorrupted_reference_baseline",
        "probabilities__corrupted_observed_baseline",
        "probabilities__random_review_restoration",
        "probabilities__audit_guided_restoration",
    }
    expected_npz_keys = {
        f"{_restoration_key(outer_fold, corruption_id)}__{suffix}"
        for outer_fold, corruption_id in expected_rotation_keys
        for suffix in suffixes
    }
    certificate = _read_json_object(
        run / "restoration_replay_certificate.json",
        "restoration replay certificate",
    )
    restoration_bindings = _read_json_object(
        run / "restoration_input_bindings.json",
        "restoration input bindings",
    )
    controls = _read_json_object(run / "execution_controls.json", "execution controls")
    confirmatory_bindings = _read_json_object(
        run / "confirmatory_input_bindings.json",
        "confirmatory input bindings",
    )
    bridge_binding = confirmatory_bindings.get("bridge")
    if (
        confirmatory_bindings.get("schema_version") != 1
        or confirmatory_bindings.get("config_semantic_sha256") != plan.config_sha256
        or confirmatory_bindings.get("execution_controls_binding_sha256")
        != controls.get("binding_sha256")
        or not isinstance(bridge_binding, Mapping)
        or not isinstance(bridge_binding.get("partition_bindings"), Mapping)
        or bridge_binding.get("partition_content_sha256")
        != _canonical_sha256(bridge_binding["partition_bindings"])
        or bridge_binding.get("partition_content_sha256")
        != certificate.get("bridge_partition_content_sha256")
        or bridge_binding.get("corruption_assignment_sha256")
        != certificate.get("bridge_corruption_assignment_sha256")
        or bridge_binding.get("provenance_binding_sha256")
        != certificate.get("bridge_provenance_binding_sha256")
    ):
        raise ValueError("restoration replay certificate differs from confirmatory input bindings")
    if (
        restoration_bindings.get("schema_version") != 1
        or restoration_bindings.get("policy") != "immutable_pre_replay_partition_bindings_v1"
        or restoration_bindings.get("confirmatory_config_semantic_sha256") != plan.config_sha256
        or restoration_bindings.get("bridge_partition_content_sha256")
        != certificate.get("bridge_partition_content_sha256")
        or restoration_bindings.get("bridge_corruption_assignment_sha256")
        != certificate.get("bridge_corruption_assignment_sha256")
        or restoration_bindings.get("bridge_provenance_binding_sha256")
        != certificate.get("bridge_provenance_binding_sha256")
        or restoration_bindings.get("partition_bindings")
        != bridge_binding.get("partition_bindings")
    ):
        raise ValueError("restoration replay certificate differs from immutable input bindings")
    original_selection = config.get("original_audit_selection")
    if not isinstance(original_selection, Mapping) or not isinstance(
        original_selection.get("classifier"), Mapping
    ):
        raise ValueError("restoration replay classifier selection is malformed")
    classifier = cast(Mapping[str, Any], original_selection["classifier"])
    parameters = classifier.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("restoration replay classifier parameters are malformed")
    for field in (
        "bridge_partition_content_sha256",
        "bridge_corruption_assignment_sha256",
        "bridge_provenance_binding_sha256",
    ):
        if not _valid_sha(certificate.get(field)):
            raise ValueError(f"restoration replay certificate {field} is invalid")
    expected_certificate_fields = {
        "schema_version": 1,
        "status": "passed",
        "policy": "deterministic_checksum_bound_restoration_replay_v1",
        "confirmatory_config_semantic_sha256": plan.config_sha256,
        "execution_controls_binding_sha256": controls.get("binding_sha256"),
        "scenario_id": scenario_id,
        "representation_id": restoration.get("representation_id"),
        "model_seed": model_seed,
        "ranking_method": restoration.get("ranking_method"),
        "review_budget": review_budget,
        "random_repeats": random_repeats,
        "random_seed": random_seed,
        "l2": float(parameters["l2"]),
        "max_iter": int(parameters["max_iter"]),
        "evidence_relative_path": evidence_path.name,
        "evidence_sha256": sha256_file(evidence_path),
    }
    for field, expected in expected_certificate_fields.items():
        if certificate.get(field) != expected:
            raise ValueError(f"restoration replay certificate {field} differs")
    saved_array_hashes = certificate.get("evidence_arrays")
    if not isinstance(saved_array_hashes, Mapping) or set(saved_array_hashes) != expected_npz_keys:
        raise ValueError("restoration replay certificate array schema differs")
    expected_risk_sources: dict[str, dict[str, str]] = {}
    for outer_fold, corruption_id in sorted(expected_rotation_keys):
        source_cells = [
            cell
            for cell in plan.cells
            if cell.outer_fold == outer_fold
            and cell.corruption_cell_id == corruption_id
            and cell.scenario_id == scenario_id
            and cell.model_seed == model_seed
        ]
        if len(source_cells) != 1:
            raise ValueError("restoration replay source cell is absent or ambiguous")
        cell_id = source_cells[0].cell_id
        relative = f"cells/{cell_id}/risk_scores.npz"
        expected_risk_sources[_restoration_key(outer_fold, corruption_id)] = {
            "cell_id": cell_id,
            "relative_path": relative,
            "sha256": sha256_file(run / relative),
        }
    if certificate.get("risk_sources") != expected_risk_sources:
        raise ValueError("restoration replay certificate risk sources differ")
    original = _read_json_object(run / "original_audit_selection.json", "original-audit selection")
    final_group_bindings = original.get("sealed_feature_cache_provenance_by_rotation")
    if not isinstance(final_group_bindings, Mapping):
        raise ValueError("original-audit selection lacks final-reference group bindings")
    final_binding_by_fold: dict[int, str] = {}
    final_identity_by_fold: dict[int, tuple[str, str, str]] = {}
    try:
        with np.load(evidence_path, allow_pickle=False) as payload:
            if set(payload.files) != expected_npz_keys:
                raise ValueError("restoration evidence NPZ keys differ from frozen rotations")
            for key in payload.files:
                if saved_array_hashes.get(key) != array_artifact_sha256(payload[key]):
                    raise ValueError(f"restoration replay certificate array hash differs: {key}")
            for outer_fold, corruption_id in sorted(expected_rotation_keys):
                prefix = _restoration_key(outer_fold, corruption_id)

                def array(name: str, current_prefix: str = prefix) -> np.ndarray[Any, Any]:
                    return np.asarray(payload[f"{current_prefix}__{name}"])

                audit_ids_raw = array("audit_sample_ids")
                audit_groups_raw = array("audit_group_ids")
                final_ids_raw = array("final_sample_ids")
                final_groups_raw = array("final_group_ids")
                if any(
                    value.dtype.kind not in {"U", "S"}
                    for value in (
                        audit_ids_raw,
                        audit_groups_raw,
                        final_ids_raw,
                        final_groups_raw,
                    )
                ):
                    raise ValueError("restoration sample/group identities require pickle")
                audit_ids = tuple(str(value) for value in audit_ids_raw.tolist())
                audit_groups = tuple(str(value) for value in audit_groups_raw.tolist())
                final_ids = tuple(str(value) for value in final_ids_raw.tolist())
                final_groups = tuple(str(value) for value in final_groups_raw.tolist())
                n_audit = len(audit_ids)
                n_final = len(final_ids)
                if (
                    not n_audit
                    or not n_final
                    or len(audit_groups) != n_audit
                    or len(final_groups) != n_final
                    or len(set(audit_ids)) != n_audit
                    or len(set(final_ids)) != n_final
                    or set(audit_ids).intersection(final_ids)
                    or set(audit_groups).intersection(final_groups)
                ):
                    raise ValueError("restoration audit/final identities leak or misalign")
                final_group_sha = _canonical_sha256(sorted(set(final_groups)))
                expected_final = final_group_bindings.get(str(outer_fold))
                if (
                    not isinstance(expected_final, Mapping)
                    or expected_final.get("final_reference_group_ids_sha256") != final_group_sha
                ):
                    raise ValueError("restoration final groups differ from sealed rotation")
                previous_final = final_binding_by_fold.setdefault(outer_fold, final_group_sha)
                if previous_final != final_group_sha:
                    raise ValueError("restoration final groups change across corruption cells")

                source_cell = next(
                    (
                        cell
                        for cell in plan.cells
                        if cell.outer_fold == outer_fold
                        and cell.corruption_cell_id == corruption_id
                        and cell.scenario_id == scenario_id
                        and cell.model_seed == model_seed
                    ),
                    None,
                )
                if source_cell is None:
                    raise ValueError("restoration source cell is absent from frozen matrix")
                with np.load(
                    run / "cells" / source_cell.cell_id / "oof_evidence.npz",
                    allow_pickle=False,
                ) as oof:
                    expected_audit_ids = tuple(str(value) for value in oof["sample_ids"].tolist())
                    expected_audit_groups = tuple(str(value) for value in oof["group_ids"].tolist())
                    expected_pre = np.asarray(oof["pre_corruption_label"], dtype=np.int64)
                    expected_observed = np.asarray(oof["observed_label"], dtype=np.int64)
                    expected_injected = np.asarray(oof["is_injected_corruption"], dtype=bool)
                pre = _strict_integer_array(
                    array("pre_corruption_label"), "restoration pre-corruption labels"
                )
                observed = _strict_integer_array(
                    array("observed_label"), "restoration observed labels"
                )
                injected = _strict_bool_array(
                    array("is_injected_corruption"), "restoration corruption flags"
                )
                if (
                    audit_ids != expected_audit_ids
                    or audit_groups != expected_audit_groups
                    or not np.array_equal(pre, expected_pre)
                    or not np.array_equal(observed, expected_observed)
                    or not np.array_equal(injected, expected_injected)
                ):
                    raise ValueError("restoration audit labels/identities differ from source OOF")

                final_pre = _strict_integer_array(
                    array("final_pre_corruption_label"), "final pre-corruption labels"
                )
                final_observed = _strict_integer_array(
                    array("final_observed_label"), "final observed labels"
                )
                final_injected = _strict_bool_array(
                    array("final_is_injected_corruption"), "final corruption flags"
                )
                if (
                    final_pre.shape != (n_final,)
                    or set(int(value) for value in final_pre) != set(CLASS_ORDER)
                    or not np.array_equal(final_pre, final_observed)
                    or final_injected.shape != (n_final,)
                    or final_injected.any()
                ):
                    raise ValueError("restoration final reference is not untouched/all-class")
                final_identity = (
                    _canonical_sha256(list(final_ids)),
                    _canonical_sha256(list(final_groups)),
                    _canonical_sha256([int(value) for value in final_pre.tolist()]),
                )
                previous_identity = final_identity_by_fold.setdefault(outer_fold, final_identity)
                if previous_identity != final_identity:
                    raise ValueError(
                        "restoration final sample/group/label identity changes across "
                        "corruption cells"
                    )

                budget = budget_count(n_audit, review_budget)
                guided_reviewed = _strict_bool_array(
                    array("guided_reviewed_mask"), "guided reviewed mask"
                )
                guided_restored = _strict_bool_array(
                    array("guided_restored_mask"), "guided restored mask"
                )
                guided_labels = _strict_integer_array(
                    array("guided_restored_label"), "guided restored labels"
                )
                random_reviewed = _strict_bool_array(
                    array("random_reviewed_mask"), "random reviewed masks"
                )
                random_restored = _strict_bool_array(
                    array("random_restored_mask"), "random restored masks"
                )
                random_labels = _strict_integer_array(
                    array("random_restored_label"), "random restored labels"
                )
                if (
                    guided_reviewed.shape != (n_audit,)
                    or random_reviewed.shape != (random_repeats, n_audit)
                    or random_restored.shape != random_reviewed.shape
                    or random_labels.shape != random_reviewed.shape
                    or int(guided_reviewed.sum()) != budget
                    or np.any(random_reviewed.sum(axis=1) != budget)
                ):
                    raise ValueError("restoration guided/random review budgets differ")
                with np.load(
                    run / "cells" / source_cell.cell_id / "risk_scores.npz",
                    allow_pickle=False,
                ) as risks:
                    ranking_scores = np.asarray(
                        risks[str(restoration["ranking_method"])], dtype=np.float64
                    )
                expected_guided = np.zeros(n_audit, dtype=bool)
                expected_guided[rank_indices(ranking_scores, tie_break_ids=audit_ids)[:budget]] = (
                    True
                )
                if not np.array_equal(guided_reviewed, expected_guided):
                    raise ValueError("restoration guided selection differs from frozen ranking")
                expected_guided_restoration = restore_reviewed_labels(
                    pre, observed, injected, expected_guided
                )
                if not np.array_equal(
                    guided_restored, expected_guided_restoration.restored_mask
                ) or not np.array_equal(guided_labels, expected_guided_restoration.restored_labels):
                    raise ValueError("guided restoration changes unreviewed/non-injected labels")

                expected_random_seeds = [random_seed + repeat for repeat in range(random_repeats)]
                for repeat, repeat_seed in enumerate(expected_random_seeds):
                    rng = np.random.default_rng(repeat_seed)
                    indices = np.sort(rng.choice(n_audit, size=budget, replace=False)).astype(
                        np.int64
                    )
                    expected_mask = np.zeros(n_audit, dtype=bool)
                    expected_mask[indices] = True
                    expected_restoration = restore_reviewed_labels(
                        pre, observed, injected, expected_mask
                    )
                    if (
                        not np.array_equal(random_reviewed[repeat], expected_mask)
                        or not np.array_equal(
                            random_restored[repeat], expected_restoration.restored_mask
                        )
                        or not np.array_equal(
                            random_labels[repeat], expected_restoration.restored_labels
                        )
                    ):
                        raise ValueError(
                            "random restoration selections/labels differ from frozen seeds"
                        )

                uncorrupted_probabilities = _strict_float_array(
                    array("probabilities__uncorrupted_reference_baseline"),
                    "uncorrupted-reference probabilities",
                )
                corrupted_probabilities = _strict_float_array(
                    array("probabilities__corrupted_observed_baseline"),
                    "corrupted-observed probabilities",
                )
                guided_probabilities = _strict_float_array(
                    array("probabilities__audit_guided_restoration"),
                    "audit-guided probabilities",
                )
                random_probabilities = _strict_float_array(
                    array("probabilities__random_review_restoration"),
                    "random-review probabilities",
                )
                if not all(
                    _valid_probability_matrix(value, expected_shape=(n_final, len(CLASS_ORDER)))
                    for value in (
                        uncorrupted_probabilities,
                        corrupted_probabilities,
                        guided_probabilities,
                    )
                ) or not _valid_probability_matrix(
                    random_probabilities,
                    expected_shape=(random_repeats, n_final, len(CLASS_ORDER)),
                ):
                    raise ValueError("restoration final-reference probabilities are invalid")

                rotation = rotation_by_key[(outer_fold, corruption_id)]
                condition_results = rotation.get("conditions")
                if (
                    not isinstance(condition_results, Mapping)
                    or set(condition_results) != _RESTORATION_CONDITIONS
                ):
                    raise ValueError("restoration rotation lacks four condition results")
                deterministic_probabilities = {
                    "uncorrupted_reference_baseline": uncorrupted_probabilities,
                    "corrupted_observed_baseline": corrupted_probabilities,
                    "audit_guided_restoration": guided_probabilities,
                }
                for condition, probabilities in deterministic_probabilities.items():
                    expected_metrics = classification_metrics(
                        final_pre, probabilities, class_order=CLASS_ORDER
                    ).as_dict()
                    saved = condition_results[condition]
                    if not isinstance(saved, Mapping) or _canonical_sha256(
                        saved.get("metrics")
                    ) != _canonical_sha256(expected_metrics):
                        raise ValueError(
                            f"restoration {condition} metrics differ from probabilities"
                        )
                random_metrics = [
                    classification_metrics(
                        final_pre, random_probabilities[repeat], class_order=CLASS_ORDER
                    ).as_dict()
                    for repeat in range(random_repeats)
                ]
                random_saved = condition_results["random_review_restoration"]
                if not isinstance(random_saved, Mapping) or not isinstance(
                    random_saved.get("runs"), list
                ):
                    raise ValueError("random restoration per-seed metrics are missing")
                saved_runs = random_saved["runs"]
                if len(saved_runs) != random_repeats:
                    raise ValueError("random restoration metric run count differs")
                for repeat, (saved, expected_metrics) in enumerate(
                    zip(saved_runs, random_metrics, strict=True)
                ):
                    if (
                        not isinstance(saved, Mapping)
                        or saved.get("review_seed") != expected_random_seeds[repeat]
                        or _canonical_sha256(saved.get("metrics"))
                        != _canonical_sha256(expected_metrics)
                        or saved.get("reviewed_count") != budget
                        or saved.get("restored_count") != int(random_restored[repeat].sum())
                    ):
                        raise ValueError("random restoration run metrics/counts differ")
                random_f1 = np.asarray(
                    [float(value["macro_f1"]) for value in random_metrics],
                    dtype=np.float64,
                )
                interval = [float(value) for value in np.quantile(random_f1, (0.025, 0.975))]
                if (
                    not math.isclose(
                        _strict_float(random_saved.get("macro_f1_mean"), "random macro-F1 mean"),
                        float(random_f1.mean()),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or random_saved.get("macro_f1_interval_95") != interval
                    or rotation.get("audit_sample_count") != n_audit
                    or rotation.get("final_sample_count") != n_final
                    or rotation.get("review_budget_count") != budget
                    or rotation.get("random_review_seeds") != expected_random_seeds
                    or rotation.get("final_reference_group_ids_sha256") != final_group_sha
                ):
                    raise ValueError("restoration rotation summary differs from evidence")
                deterministic_counts = {
                    "uncorrupted_reference_baseline": (0, 0),
                    "corrupted_observed_baseline": (0, 0),
                    "audit_guided_restoration": (
                        budget,
                        int(guided_restored.sum()),
                    ),
                }
                for condition, (reviewed_count, restored_count) in deterministic_counts.items():
                    saved = condition_results[condition]
                    if (
                        saved.get("reviewed_count") != reviewed_count
                        or saved.get("restored_count") != restored_count
                    ):
                        raise ValueError(f"restoration {condition} review/restored counts differ")
    except (OSError, KeyError, ValueError) as error:
        if isinstance(error, ValueError) and "restoration" in str(error):
            raise
        raise ValueError("restoration NPZ is invalid or pickle-dependent") from error


def _validate_root_aggregates(
    run: Path,
    *,
    config: Mapping[str, Any],
    plan: ConfirmatoryMatrixPlan,
    cell_manifest_paths: set[str],
) -> tuple[ConfirmatoryArtifactReadback, ...]:
    folds = tuple(sorted({cell.outer_fold for cell in plan.cells}))
    corruption = config["corruption"]
    ensemble_config = config["ensemble"]
    hybrid_config = config["fixed_hybrid"]
    rotation_config = config["fold_rotation"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            corruption,
            ensemble_config,
            hybrid_config,
            rotation_config,
        )
    ):
        raise ValueError("frozen config aggregate controls are malformed")
    corruption_cells = corruption["cells"]
    if not isinstance(corruption_cells, Sequence):
        raise ValueError("frozen corruption cells are malformed")
    corruption_ids = {str(value["id"]) for value in corruption_cells if isinstance(value, Mapping)}
    frozen_provenance = _validate_frozen_feature_provenance(
        run,
        config=config,
        plan=plan,
    )
    index_rows = _read_csv_rows(run / "cell_index.csv", "confirmatory cell index")
    status_by_id = {str(row["cell_id"]): str(row["status"]) for row in index_rows}

    ensemble = _read_json_object(run / "ensemble_evidence.json", "ensemble aggregate")
    _exact_fold_list(ensemble.get("outer_folds"), folds, "ensemble outer folds")
    _exact_string_set(
        ensemble.get("corruption_cell_ids"), corruption_ids, "ensemble corruption cells"
    )
    if ensemble.get("primary_risk") != ensemble_config.get("primary_risk"):
        raise ValueError("ensemble aggregate primary risk differs from frozen config")
    if ensemble.get("members") != ensemble_config.get("members"):
        raise ValueError("ensemble aggregate members differ from frozen config")
    if (
        ensemble.get("config_semantic_sha256") != plan.config_sha256
        or ensemble.get("secondary_risks") != ensemble_config.get("secondary_risks")
        or ensemble.get("risk_arrays_are_saved_per_cell") != "cells/<cell_id>/risk_scores.npz"
    ):
        raise ValueError("ensemble aggregate provenance/secondary-risk binding is invalid")
    raw_members = ensemble_config.get("members")
    if not isinstance(raw_members, list):
        raise ValueError("frozen ensemble members are malformed")
    member_keys = {
        (str(value["scenario_id"]), int(value["model_seed"]))
        for value in raw_members
        if isinstance(value, Mapping)
    }
    expected_groups = []
    for outer_fold in folds:
        for corruption_id in [
            str(value["id"]) for value in corruption_cells if isinstance(value, Mapping)
        ]:
            member_cells = [
                cell
                for cell in plan.cells
                if cell.outer_fold == outer_fold
                and cell.corruption_cell_id == corruption_id
                and (cell.scenario_id, cell.model_seed) in member_keys
            ]
            expected_groups.append(
                {
                    "outer_fold": outer_fold,
                    "corruption_cell_id": corruption_id,
                    "member_cell_ids": [cell.cell_id for cell in member_cells],
                    "all_members_completed": all(
                        status_by_id.get(cell.cell_id) == "completed" for cell in member_cells
                    ),
                }
            )
    if ensemble.get("groups") != expected_groups:
        raise ValueError("ensemble aggregate group/member evidence differs from filesystem")

    ablations = _read_json_object(
        run / "fixed_hybrid_drop_one_ablations.json", "hybrid ablation aggregate"
    )
    _exact_fold_list(ablations.get("outer_folds"), folds, "hybrid ablation outer folds")
    if ablations.get("components") != hybrid_config.get("components") or set(
        str(value) for value in ablations.get("drop_one_ablations", [])
    ) != set(str(value) for value in hybrid_config.get("drop_one_ablations", [])):
        raise ValueError("hybrid ablation aggregate differs from frozen components")
    if ablations.get("config_semantic_sha256") != plan.config_sha256 or ablations.get(
        "weights"
    ) != hybrid_config.get("weights"):
        raise ValueError("hybrid ablation aggregate has invalid config/weight binding")
    expected_cell_evidence = [
        {
            "cell_id": cell.cell_id,
            "risk_scores_path": f"cells/{cell.cell_id}/risk_scores.npz",
            "risk_scores_sha256": sha256_file(run / "cells" / cell.cell_id / "risk_scores.npz"),
        }
        for cell in plan.cells
        if status_by_id.get(cell.cell_id) == "completed"
    ]
    if ablations.get("cell_evidence") != expected_cell_evidence:
        raise ValueError("hybrid ablation cell evidence differs from filesystem risks")

    _validate_paired_statistics(
        run,
        config=config,
        plan=plan,
        status_by_id=status_by_id,
    )

    _validate_restoration_evidence(run, config=config, plan=plan)

    aggregate = _read_json_object(run / "fold_aggregate.json", "fold aggregate")
    _exact_fold_list(aggregate.get("outer_folds"), folds, "fold aggregate outer folds")
    if aggregate.get("aggregate_policy") != rotation_config.get("aggregate_policy"):
        raise ValueError("fold aggregate policy differs from frozen config")
    expected_fold_rows = []
    for outer_fold in folds:
        fold_rows = [
            row for row in index_rows if _strict_int(row["outer_fold"], "fold row") == outer_fold
        ]
        expected_fold_rows.append(
            {
                "outer_fold": outer_fold,
                "planned_cell_count": len(fold_rows),
                "completed_cell_count": sum(row["status"] == "completed" for row in fold_rows),
                "skipped_optional_cell_count": sum(
                    row["status"] == "skipped_with_frozen_blocker" for row in fold_rows
                ),
                "failed_cell_count": sum(row["status"] == "failed" for row in fold_rows),
                "reported_separately": True,
            }
        )
    if (
        aggregate.get("folds") != expected_fold_rows
        or aggregate.get("outcome_metrics_aggregation_status")
        != "completed_by_stage_statistics_runner"
    ):
        raise ValueError("fold aggregate completion evidence differs from filesystem")

    original = _read_json_object(run / "original_audit_selection.json", "original-audit selection")
    if original.get("status") != "completed" or original.get("selection") != config.get(
        "original_audit_selection"
    ):
        raise ValueError("original-audit selection differs from frozen config")
    selection = config.get("original_audit_selection")
    cache_records = config.get("cache_provenance")
    controls = _read_json_object(run / "execution_controls.json", "confirmatory execution controls")
    if not isinstance(selection, Mapping) or not isinstance(cache_records, list):
        raise ValueError("frozen original-audit/cache selection is malformed")
    selected_cache_id = str(selection.get("cache_provenance_id", ""))
    selected_representation = str(selection.get("representation_id", ""))
    selected_cache = next(
        (
            value
            for value in cache_records
            if isinstance(value, Mapping) and value.get("id") == selected_cache_id
        ),
        None,
    )
    if selected_cache is None or original.get("frozen_cache_provenance_record") != dict(
        selected_cache
    ):
        raise ValueError("original-audit cache record differs from frozen config")
    if original.get("selection_semantic_sha256") != _canonical_sha256(dict(selection)):
        raise ValueError("original-audit selection semantic SHA-256 is invalid")
    if (
        original.get("confirmatory_config_semantic_sha256") != plan.config_sha256
        or original.get("matrix_plan_config_sha256") != plan.config_sha256
        or original.get("execution_controls_binding_sha256") != controls.get("binding_sha256")
    ):
        raise ValueError("original-audit selection has invalid plan/control bindings")
    sealed_by_rotation = original.get("sealed_feature_cache_provenance_by_rotation")
    selected_rotations = frozen_provenance.get(selected_representation)
    if (
        not isinstance(sealed_by_rotation, Mapping)
        or selected_rotations is None
        or set(sealed_by_rotation) != set(selected_rotations)
    ):
        raise ValueError("original-audit cache provenance lacks exact fold rotations")
    for fold_key, raw in sealed_by_rotation.items():
        if not isinstance(raw, Mapping):
            raise ValueError("original-audit rotation provenance is malformed")
        final_groups_sha = raw.get("final_reference_group_ids_sha256")
        without_final = {
            key: value for key, value in raw.items() if key != "final_reference_group_ids_sha256"
        }
        if not _valid_sha(final_groups_sha) or without_final != dict(
            selected_rotations[str(fold_key)]
        ):
            raise ValueError("original-audit rotation cache/final-group binding is invalid")

    report_path = run / "report.md"
    try:
        report = report_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("confirmatory report is missing or unreadable") from error
    if not report.strip():
        raise ValueError("confirmatory report cannot be empty")
    report_fragments = (
        "potentially inconsistent annotation",
        "recommended for expert review",
        "Random mean macro F1 (95% interval)",
        "direction `",
        "probability delta > 0",
    )
    if any(fragment not in report for fragment in report_fragments):
        raise ValueError("confirmatory report lacks the required scientific result fields")
    restoration_payload = _read_json_object(
        run / "restoration_metrics.json", "report restoration aggregate"
    )
    paired_payload = _read_json_object(run / "paired_statistics.json", "report paired statistics")
    expected_report_contract = confirmatory_report_contract_block(
        restoration_payload,
        paired_payload,
    )
    if (
        report.count(CONFIRMATORY_REPORT_CONTRACT_START) != 1
        or report.count(CONFIRMATORY_REPORT_CONTRACT_END) != 1
        or expected_report_contract not in report
    ):
        raise ValueError(
            "confirmatory report machine-readable contract differs from saved evidence"
        )

    figures = _read_hash_manifest(run / "figure_manifest.json", "figure manifest")
    if not figures:
        raise ValueError("figure manifest cannot be empty")
    figure_records = _verify_hash_manifest(
        run,
        figures,
        expected_paths=set(figures),
        role="figure manifest",
        run_root=run,
    )

    root_manifest_path = run / CONFIRMATORY_ARTIFACT_MANIFEST_FILENAME
    root_manifest = _read_hash_manifest(root_manifest_path, "root artifact manifest")
    expected_root_paths = {*_ROOT_BASE_ARTIFACTS, *cell_manifest_paths}
    root_records = _verify_hash_manifest(
        run,
        root_manifest,
        expected_paths=expected_root_paths,
        role="root artifact manifest",
        run_root=run,
    )
    return (
        *root_records,
        *figure_records,
        _artifact(root_manifest_path, run),
    )


def read_confirmatory_run_directory(
    plan: ConfirmatoryMatrixPlan,
    run_directory: str | Path,
    *,
    frozen_confirmatory_config_path: str | Path,
    expected_frozen_config_sha256: str,
    expected_confirmatory_storage_policy_sha256: str | None = None,
    require_final_policy_bindings: bool | None = None,
) -> ConfirmatoryFilesystemReadback:
    """Independently re-read and hash every stage-eligible confirmatory artifact."""

    run = Path(run_directory).resolve()
    cells: tuple[ConfirmatoryCellReadback, ...] = ()
    reconciliation: ConfirmatoryMatrixReconciliation | None = None
    root_artifacts: tuple[ConfirmatoryArtifactReadback, ...] = ()
    matrix_sha: str | None = None
    index_sha: str | None = None
    root_manifest_sha: str | None = None
    storage_policy_sha: str | None = None
    errors: list[str] = []
    try:
        if not run.is_dir():
            raise FileNotFoundError(f"confirmatory run directory is missing: {run}")
        if os.path.lexists(run / "checkpoints"):
            raise ValueError("legacy top-level confirmatory checkpoints path is forbidden")
        final_policy_bindings_required = (
            (run / ".immutable.json").is_file()
            if require_final_policy_bindings is None
            else require_final_policy_bindings
        )
        storage_policy_sha = _validate_confirmatory_storage_policy_bindings(
            run,
            expected_sha256=expected_confirmatory_storage_policy_sha256,
            require_final_bindings=final_policy_bindings_required,
        )
        frozen_path = Path(frozen_confirmatory_config_path).resolve()
        config = _load_frozen_config(
            plan,
            frozen_path,
            expected_sha256=expected_frozen_config_sha256,
        )
        corruptions, scenarios = _validate_matrix_and_controls(run, plan, config)
        matrix_sha = sha256_file(run / "matrix_plan.json")
        index_sha = sha256_file(run / "cell_index.csv")
        cells, outcomes, _cell_artifacts = _read_cell_index(
            run,
            plan,
            config=config,
            corruptions=corruptions,
            scenarios=scenarios,
        )
        _validate_dual_checkpoint_layout(
            run,
            cells,
            n_splits=int(cast(Mapping[str, Any], config["oof"])["n_splits"]),
            require_canonical_names=storage_policy_sha is not None,
        )
        reconciliation = reconcile_confirmatory_cell_outcomes(plan, outcomes)
        if not reconciliation.passed or not reconciliation.fold_rotation_complete:
            raise ValueError(f"filesystem cell reconciliation failed: {reconciliation.errors}")
        saved_reconciliation = _read_json_object(
            run / "reconciliation.json", "saved confirmatory reconciliation"
        )
        if _canonical_sha256(saved_reconciliation) != _canonical_sha256(reconciliation.as_dict()):
            raise ValueError("saved reconciliation differs from filesystem readback")
        cell_manifest_paths = {f"cells/{cell.cell_id}/artifact_manifest.json" for cell in cells}
        root_artifacts = _validate_root_aggregates(
            run,
            config=config,
            plan=plan,
            cell_manifest_paths=cell_manifest_paths,
        )
        root_manifest_sha = sha256_file(run / CONFIRMATORY_ARTIFACT_MANIFEST_FILENAME)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    unique_checked_paths = {
        record.relative_path
        for record in (
            *(record for cell in cells for record in cell.checked_artifacts),
            *root_artifacts,
        )
    }
    checked_count = len(unique_checked_paths)
    return ConfirmatoryFilesystemReadback(
        status="passed" if not errors else "failed",
        run_directory=run,
        matrix_plan_sha256=matrix_sha,
        cell_index_sha256=index_sha,
        root_artifact_manifest_sha256=root_manifest_sha,
        confirmatory_storage_policy_sha256=storage_policy_sha,
        checked_artifact_count=checked_count,
        cells=cells,
        reconciliation=reconciliation,
        root_artifacts=root_artifacts,
        errors=tuple(errors),
    )


def _normalise_gate_evidence(
    gate_evidence: (
        ConfirmatoryExecutionGateEvidence | ResourceBoundedExecutionGateEvidence | Mapping[str, Any]
    ),
) -> tuple[Mapping[str, Any], bool, bool, str, str]:
    """Accept only the typed object returned by the real confirmatory gate."""

    # Kept local to avoid an experiment/workflow import cycle at module import time.
    from histo_audit.workflows.study_gates import ConfirmatoryExecutionGateEvidence

    if isinstance(gate_evidence, ConfirmatoryExecutionGateEvidence):
        return (
            gate_evidence.as_dict(),
            True,
            True,
            "PRIMARY_STUDY_COMPLETE",
            "passed",
        )
    raise ValueError(
        "outcome-eligible confirmatory completion requires a real typed "
        "ConfirmatoryExecutionGateEvidence; serialised mappings are rejected"
    )


def _normalise_resource_gate_evidence(
    gate_evidence: object,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Require typed P+(historical C or effective D) non-claiming authority."""

    from histo_audit.workflows.preregistration_amendment import (
        RESOURCE_BOUNDED_AUTHORITY_C_CONFIG_SEMANTIC_SHA256,
        validate_resource_bounded_capacity_v3,
    )
    from histo_audit.workflows.study_gates import (
        ResourceBoundedExecutionGateEvidence,
    )

    if not isinstance(gate_evidence, ResourceBoundedExecutionGateEvidence):
        raise ValueError(
            "resource-bounded completion requires typed ResourceBoundedExecutionGateEvidence"
        )
    authority = gate_evidence.execution_authority
    historical = gate_evidence.historical_primary
    capacity_policy = dict(authority.resource_capacity_policy)
    workspace_plan = authority.resource_input_workspace_plan
    workspace_plan_sha256 = authority.resource_input_workspace_plan_sha256
    historical_c_contract = (
        authority.resource_confirmatory_config_semantic_sha256
        == RESOURCE_BOUNDED_AUTHORITY_C_CONFIG_SEMANTIC_SHA256
        and capacity_policy == RESOURCE_BOUNDED_CAPACITY_POLICY_V2
        and workspace_plan is None
        and workspace_plan_sha256 is None
    )
    effective_d_contract = False
    canonical_workspace_plan: dict[str, Any] | None = None
    if isinstance(workspace_plan, Mapping):
        try:
            canonical_capacity, canonical_workspace_plan = validate_resource_bounded_capacity_v3(
                capacity_policy,
                workspace_plan,
            )
        except (TypeError, ValueError):
            pass
        else:
            effective_d_contract = (
                capacity_policy == canonical_capacity
                and authority.resource_confirmatory_config_semantic_sha256
                == RESOURCE_BOUNDED_CONFIRMATORY_CONFIG_SHA256
                and workspace_plan_sha256
                == canonical_workspace_plan["plan_without_self_hash_sha256"]
            )
    if (
        gate_evidence.outcomes_inspected is not True
        or gate_evidence.analysis_disposition != "amended_or_exploratory"
        or gate_evidence.original_confirmatory_claim_allowed is not False
        or gate_evidence.study_outcome_eligible is not False
        or gate_evidence.completion_stage is not None
        or gate_evidence.primary_rebinding_allowed is not False
        or gate_evidence.primary_mutation_allowed is not False
        or authority.resource_profile_id != "resource_bounded_confirmatory_v1"
        or not (historical_c_contract or effective_d_contract)
    ):
        raise ValueError(
            "resource-bounded gate lacks the exact permanent non-claiming "
            "P+(historical C or effective D) contract"
        )
    gate = gate_evidence.as_dict()
    disposition = {
        "outcomes_inspected": True,
        "analysis_disposition": "amended_or_exploratory",
        "original_confirmatory_claim_allowed": False,
        "study_outcome_eligible": False,
        "completion_stage": None,
        "m9_unlock_allowed": False,
    }
    bindings = {
        "resource_gate_sha256": _canonical_sha256(gate),
        "historical_primary_dependency_sha256": _canonical_sha256(historical.as_dict()),
        "resource_execution_authority_sha256": _canonical_sha256(authority.as_dict()),
        "resource_capacity_policy_sha256": _canonical_sha256(capacity_policy),
        "resource_input_workspace_plan_sha256": (
            None
            if canonical_workspace_plan is None
            else canonical_workspace_plan["plan_without_self_hash_sha256"]
        ),
        "resource_disposition_binding_sha256": _canonical_sha256(disposition),
        "resource_authorization_sha256": authority.authorization_sha256,
        "historical_primary_run_id": historical.primary_run_id,
        "historical_primary_artifact_root_sha256": (historical.primary_artifact_root_sha256),
        "resource_execution_source_root_sha256": (authority.resource_execution_source_root_sha256),
        "confirmatory_storage_policy_sha256": (authority.confirmatory_storage_policy_sha256),
        **disposition,
    }
    return gate, bindings


def _mapping(value: Any, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"confirmatory gate evidence {role} must be a mapping")
    return value


def build_confirmatory_completion_evidence(
    *,
    plan: ConfirmatoryMatrixPlan,
    reconciliation: ConfirmatoryMatrixReconciliation,
    artifact_scope: str,
    study_outcome_eligible: bool,
    gate_evidence: (
        ConfirmatoryExecutionGateEvidence
        | ResourceBoundedExecutionGateEvidence
        | Mapping[str, Any]
        | None
    ) = None,
    run_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Build completion evidence without allowing synthetic or unverified stage claims."""

    supported_scopes = {
        REAL_CONFIRMATORY_ARTIFACT_SCOPE,
        RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
        SYNTHETIC_CONFIRMATORY_ARTIFACT_SCOPE,
    }
    if artifact_scope not in supported_scopes:
        raise ValueError(f"unsupported confirmatory artifact scope: {artifact_scope!r}")
    if artifact_scope == SYNTHETIC_CONFIRMATORY_ARTIFACT_SCOPE and study_outcome_eligible:
        raise ValueError("synthetic confirmatory fixtures can never be study-outcome eligible")
    if artifact_scope == REAL_CONFIRMATORY_ARTIFACT_SCOPE and plan.schema_version != 2:
        raise ValueError("real confirmatory artifact scope requires the original schema-v2 plan")
    if artifact_scope == RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE and (
        plan.schema_version != 3
        or plan.config_sha256 != RESOURCE_BOUNDED_CONFIRMATORY_CONFIG_SHA256
    ):
        raise ValueError(
            "resource-bounded artifact scope requires the exact frozen schema-v3 profile"
        )
    if artifact_scope == RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE and study_outcome_eligible:
        raise ValueError(
            "resource-bounded confirmatory sensitivity artifacts can never be "
            "study-outcome eligible"
        )
    if study_outcome_eligible and artifact_scope != REAL_CONFIRMATORY_ARTIFACT_SCOPE:
        raise ValueError("only the real PanNuke confirmatory scope can be outcome eligible")

    completion_stage: str | None = None
    bindings: dict[str, Any] = {}
    primary_run_sealed = False
    primary_run_registry_backed = False
    primary_completion_stage: str | None = None
    confirmatory_execution_gate_status: str | None = None
    filesystem_readback: ConfirmatoryFilesystemReadback | None = None

    if artifact_scope == RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE:
        resource_gate, resource_bindings = _normalise_resource_gate_evidence(gate_evidence)
        resource_authority = _mapping(
            resource_gate.get("execution_authority"),
            "resource execution_authority",
        )
        if (
            resource_authority.get("resource_confirmatory_config_semantic_sha256")
            != plan.config_sha256
        ):
            raise ValueError(
                "resource-bounded matrix semantic hash differs from its effective "
                "execution authority"
            )
        bindings.update(resource_bindings)
        primary_run_sealed = True
        primary_run_registry_backed = True
        primary_completion_stage = "PRIMARY_STUDY_COMPLETE"
        confirmatory_execution_gate_status = "passed_resource_bounded_non_claiming"

    if study_outcome_eligible:
        if not reconciliation.passed:
            raise ValueError(
                "CONFIRMATORY_COMPLETE requires passed matrix reconciliation: "
                f"{reconciliation.errors}"
            )
        if not reconciliation.fold_rotation_complete:
            raise ValueError("CONFIRMATORY_COMPLETE requires all three frozen fold rotations")
        if gate_evidence is None:
            raise ValueError("eligible confirmatory completion requires immutable gate evidence")
        if run_directory is None:
            raise ValueError(
                "eligible confirmatory completion requires a filesystem-backed run directory"
            )
        (
            normalised_gate,
            primary_run_sealed,
            primary_run_registry_backed,
            primary_completion_stage,
            confirmatory_execution_gate_status,
        ) = _normalise_gate_evidence(gate_evidence)
        if confirmatory_execution_gate_status != "passed":
            raise ValueError("confirmatory execution gate did not pass")
        if not primary_run_sealed or not primary_run_registry_backed:
            raise ValueError(
                "confirmatory completion requires a sealed, registry-backed primary run"
            )
        if primary_completion_stage != "PRIMARY_STUDY_COMPLETE":
            raise ValueError("confirmatory completion requires PRIMARY_STUDY_COMPLETE")

        primary_gate = _mapping(normalised_gate.get("primary_gate"), "primary_gate")
        for field in _PRIMARY_GATE_HASHES:
            value = primary_gate.get(field)
            if not _valid_sha(value):
                raise ValueError(f"confirmatory primary-gate binding {field} is not a SHA-256")
            bindings[field] = str(value)
        for field in _CONFIRMATORY_GATE_HASHES:
            value = normalised_gate.get(field)
            if field == "confirmatory_storage_policy_sha256" and value is None:
                continue
            if not _valid_sha(value):
                raise ValueError(f"confirmatory gate binding {field} is not a SHA-256")
            bindings[field] = str(value)

        primary_run_id = normalised_gate.get("primary_run_id")
        if not isinstance(primary_run_id, str) or not primary_run_id.strip():
            raise ValueError("confirmatory gate evidence lacks a primary run ID")
        primary_required_count = primary_gate.get("primary_required_cell_count")
        completed_primary_count = normalised_gate.get("completed_required_cell_count")
        if (
            not isinstance(primary_required_count, int)
            or isinstance(primary_required_count, bool)
            or primary_required_count < 1
            or not isinstance(completed_primary_count, int)
            or isinstance(completed_primary_count, bool)
            or completed_primary_count != primary_required_count
        ):
            raise ValueError("confirmatory gate evidence does not prove all primary required cells")
        if primary_gate.get("confirmatory_matrix_cell_count") != len(plan.cells):
            raise ValueError("confirmatory matrix cell count differs from frozen gate evidence")
        if primary_gate.get("confirmatory_config_semantic_sha256") != plan.config_sha256:
            raise ValueError("confirmatory matrix semantic hash differs from frozen gate evidence")
        bindings.update(
            primary_run_id=primary_run_id,
            primary_required_cell_count=primary_required_count,
            completed_primary_required_cell_count=completed_primary_count,
        )

        # The typed gate binds the timestamped, immutable frozen config.  Re-read
        # that config and the entire confirmatory artifact tree rather than
        # trusting outcome mappings or caller-provided SHA strings.
        from histo_audit.workflows.study_gates import ConfirmatoryExecutionGateEvidence

        if not isinstance(gate_evidence, ConfirmatoryExecutionGateEvidence):
            raise ValueError("eligible confirmatory completion requires typed gate evidence")
        frozen_config_path = gate_evidence.primary_gate.freeze_directory / (
            "confirmatory_frozen.yaml"
        )
        filesystem_readback = read_confirmatory_run_directory(
            plan,
            run_directory,
            frozen_confirmatory_config_path=frozen_config_path,
            expected_frozen_config_sha256=(
                gate_evidence.primary_gate.frozen_confirmatory_config_sha256
            ),
            expected_confirmatory_storage_policy_sha256=(
                gate_evidence.confirmatory_storage_policy_sha256
            ),
            require_final_policy_bindings=False,
        )
        if not filesystem_readback.passed:
            raise ValueError(
                f"confirmatory filesystem readback did not pass: {filesystem_readback.errors}"
            )
        if (
            filesystem_readback.reconciliation is None
            or filesystem_readback.reconciliation.as_dict() != reconciliation.as_dict()
        ):
            raise ValueError("caller reconciliation differs from independent filesystem readback")
        if (
            filesystem_readback.confirmatory_storage_policy_sha256
            != gate_evidence.confirmatory_storage_policy_sha256
        ):
            raise ValueError("filesystem readback differs from gated storage-policy binding")
        completion_stage = "CONFIRMATORY_COMPLETE"

    return {
        "schema_version": 1,
        "completion_stage": completion_stage,
        "valid_completion_claim": bool(
            completion_stage == "CONFIRMATORY_COMPLETE"
            and artifact_scope == REAL_CONFIRMATORY_ARTIFACT_SCOPE
            and study_outcome_eligible
        ),
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
        "fold_rotation_complete": reconciliation.fold_rotation_complete,
        "planned_outer_folds": list(reconciliation.planned_outer_folds),
        "completed_outer_folds": list(reconciliation.completed_outer_folds),
        "primary_run_sealed": primary_run_sealed,
        "primary_run_registry_backed": primary_run_registry_backed,
        "primary_completion_stage": primary_completion_stage,
        "confirmatory_execution_gate_status": confirmatory_execution_gate_status,
        "completion_stage_enabled_only_after_run_seal_and_integrity_verification": (
            study_outcome_eligible
        ),
        "filesystem_readback_status": (
            filesystem_readback.status if filesystem_readback is not None else None
        ),
        "filesystem_checked_artifact_count": (
            filesystem_readback.checked_artifact_count if filesystem_readback is not None else 0
        ),
        "filesystem_matrix_plan_sha256": (
            filesystem_readback.matrix_plan_sha256 if filesystem_readback is not None else None
        ),
        "filesystem_cell_index_sha256": (
            filesystem_readback.cell_index_sha256 if filesystem_readback is not None else None
        ),
        "filesystem_root_artifact_manifest_sha256": (
            filesystem_readback.root_artifact_manifest_sha256
            if filesystem_readback is not None
            else None
        ),
        "filesystem_confirmatory_storage_policy_sha256": (
            filesystem_readback.confirmatory_storage_policy_sha256
            if filesystem_readback is not None
            else None
        ),
        "filesystem_readback_sha256": (
            _canonical_sha256(filesystem_readback.as_dict())
            if filesystem_readback is not None
            else None
        ),
        **bindings,
    }


__all__ = [
    "CONFIRMATORY_ARTIFACT_MANIFEST_FILENAME",
    "CONFIRMATORY_REPORT_CONTRACT_END",
    "CONFIRMATORY_REPORT_CONTRACT_START",
    "REAL_CONFIRMATORY_ARTIFACT_SCOPE",
    "RESOURCE_BOUNDED_CAPACITY_POLICY",
    "RESOURCE_BOUNDED_CAPACITY_POLICY_V2",
    "RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE",
    "SYNTHETIC_CONFIRMATORY_ARTIFACT_SCOPE",
    "ConfirmatoryArtifactReadback",
    "ConfirmatoryCellReadback",
    "ConfirmatoryFilesystemReadback",
    "ConfirmatoryMatrixReconciliation",
    "build_confirmatory_completion_evidence",
    "confirmatory_report_contract_block",
    "confirmatory_report_contract_payload",
    "read_confirmatory_run_directory",
    "reconcile_confirmatory_cell_outcomes",
]
