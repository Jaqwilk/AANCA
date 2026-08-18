"""Frozen, whole-group paired statistics for confirmatory cell artifacts.

Only preregistered, explicitly selector-scoped comparisons are evaluated.  The
module never reads final-reference labels and never infers comparisons from the
observed matrix.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from histo_audit.experiment.confirmatory_core import (
    ConfirmatoryComparisonOperand,
    ConfirmatoryExecutionControls,
    ConfirmatoryPairedComparison,
)
from histo_audit.experiment.study_contracts import ConfirmatoryCell
from histo_audit.statistics.review import average_precision
from histo_audit.utils.run_tracking import atomic_write_json, sha256_file

_STATISTICS_FILE = "paired_statistics.json"
_BOOTSTRAP_FILE = "paired_bootstrap_evidence.npz"
_COMPLETED_ARTIFACTS = {
    "cell_identity.json",
    "oof_evidence.npz",
    "checkpoint_manifest.json",
    "telemetry.json",
    "risk_scores.npz",
    "ranking.csv",
    "metrics.json",
}
_SKIPPED_ARTIFACTS = {"cell_identity.json", "blocker.json"}
_RISK_ARRAY_BY_ID = {
    "self_confidence": "self_confidence",
    "ensemble_disagreement": "ensemble_disagreement",
    "fixed_hybrid": "fixed_hybrid",
    "hybrid_drop_self_confidence": "hybrid_drop_self_confidence",
    "hybrid_drop_ensemble_disagreement": "hybrid_drop_ensemble_disagreement",
}


@dataclass(frozen=True, slots=True)
class ConfirmatoryStatisticsArtifacts:
    """The two atomically persisted confirmatory statistics artifacts."""

    output_directory: Path
    statistics_path: Path
    bootstrap_evidence_path: Path
    comparison_count: int
    completed_comparison_count: int
    statistics_sha256: str
    bootstrap_evidence_sha256: str


@dataclass(frozen=True, slots=True)
class ConfirmatoryStatisticsVerification:
    """Successful semantic reread and recomputation of both artifacts."""

    status: str
    output_directory: Path
    statistics_sha256: str
    bootstrap_evidence_sha256: str
    comparison_count: int
    completed_comparison_count: int


@dataclass(frozen=True, slots=True)
class _CellEvidence:
    cell: ConfirmatoryCell
    sample_ids: NDArray[np.str_]
    group_ids: NDArray[np.str_]
    group_tokens: NDArray[np.str_]
    pre_corruption_label: NDArray[np.int64]
    observed_label: NDArray[np.int64]
    injected: NDArray[np.bool_]
    fold_id: NDArray[np.int64]
    fold_assignment_labels: NDArray[np.int64]
    risks: Mapping[str, NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class _LoadedMatrix:
    completed: Mapping[str, _CellEvidence]
    status_by_id: Mapping[str, str]
    blocker_by_scenario: Mapping[str, Mapping[str, Any]]
    group_universe: NDArray[np.str_]


@dataclass(frozen=True, slots=True)
class _ComparisonArrays:
    valid_draw_mask: NDArray[np.bool_]
    metric_a: NDArray[np.float64]
    metric_b: NDArray[np.float64]

    @property
    def differences(self) -> NDArray[np.float64]:
        return self.metric_a - self.metric_b


@dataclass(frozen=True, slots=True)
class _ComputedStatistics:
    results: tuple[dict[str, Any], ...]
    arrays: Mapping[str, NDArray[np.generic]]


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, role: str) -> dict[str, Any]:
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
        raise ValueError(f"{role} must be a JSON object")
    return value


def _read_cell_index(path: Path) -> dict[str, dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError("confirmatory cell index has invalid columns")
            rows = [dict(row) for row in reader]
    except (OSError, csv.Error) as error:
        raise ValueError("confirmatory cell index is missing or malformed") from error
    by_id = {row.get("cell_id", ""): row for row in rows}
    if not rows or "" in by_id or len(by_id) != len(rows):
        raise ValueError("confirmatory cell index has empty or duplicate cell IDs")
    return by_id


def _read_hash_manifest(path: Path, role: str) -> dict[str, str]:
    raw = _read_json(path, role)
    if not raw or any(
        not isinstance(relative, str)
        or not relative
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for relative, digest in raw.items()
    ):
        raise ValueError(f"{role} is not a strict path-to-SHA mapping")
    return cast(dict[str, str], raw)


def _verify_cell_manifest(directory: Path, status: str) -> None:
    manifest_path = directory / "artifact_manifest.json"
    manifest = _read_hash_manifest(manifest_path, f"{directory.name} artifact manifest")
    expected = _COMPLETED_ARTIFACTS if status == "completed" else _SKIPPED_ARTIFACTS
    if set(manifest) != expected:
        raise ValueError(f"{directory.name} artifact manifest has missing or extra paths")
    root = directory.resolve()
    for relative, expected_sha in manifest.items():
        candidate = (directory / relative).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError(f"{directory.name} artifact manifest contains an unsafe path")
        if sha256_file(candidate) != expected_sha:
            raise ValueError(f"{directory.name} artifact SHA differs for {relative}")


def _load_npz(path: Path, role: str) -> dict[str, NDArray[np.generic]]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            return {name: np.asarray(payload[name]) for name in payload.files}
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f"{role} is invalid or pickle-dependent") from error


def _string_vector(value: NDArray[np.generic], role: str) -> NDArray[np.str_]:
    if value.ndim != 1 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{role} must be a one-dimensional non-pickle string array")
    result = np.asarray(value, dtype=np.str_)
    if not len(result) or np.any(result == ""):
        raise ValueError(f"{role} contains empty values")
    return result


def _integer_vector(value: NDArray[np.generic], n: int, role: str) -> NDArray[np.int64]:
    if value.shape != (n,) or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{role} must be an aligned integer vector")
    return np.asarray(value, dtype=np.int64)


def _load_completed_cell(
    run: Path,
    cell: ConfirmatoryCell,
    controls: ConfirmatoryExecutionControls,
) -> _CellEvidence:
    directory = run / "cells" / cell.cell_id
    _verify_cell_manifest(directory, "completed")
    identity = _read_json(directory / "cell_identity.json", f"{cell.cell_id} identity")
    scenario = controls.scenarios_by_id[cell.scenario_id]
    corruption = controls.corruptions_by_id[cell.corruption_cell_id]
    expected_identity = {
        "schema_version": 1,
        "cell_id": cell.cell_id,
        "outer_fold": cell.outer_fold,
        "corruption_cell_id": cell.corruption_cell_id,
        "corruption_mechanism": corruption.mechanism,
        "corruption_rate": corruption.rate,
        "corruption_seed": corruption.seed,
        "scenario_id": cell.scenario_id,
        "scenario_family": scenario.family,
        "representation_id": scenario.representation_id,
        "cache_provenance_id": scenario.cache_provenance_id,
        "model_seed": cell.model_seed,
        "required": cell.required,
        "config_semantic_sha256": controls.config_semantic_sha256,
    }
    if identity != expected_identity:
        raise ValueError(f"{cell.cell_id} identity differs from frozen controls")

    oof = _load_npz(directory / "oof_evidence.npz", f"{cell.cell_id} OOF evidence")
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
    if not required_oof.issubset(oof):
        raise ValueError(f"{cell.cell_id} OOF evidence lacks required arrays")
    sample_ids = _string_vector(oof["sample_ids"], f"{cell.cell_id} sample IDs")
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError(f"{cell.cell_id} sample IDs are not unique")
    groups = _string_vector(oof["group_ids"], f"{cell.cell_id} group IDs")
    n = len(sample_ids)
    if len(groups) != n:
        raise ValueError(f"{cell.cell_id} group IDs are misaligned")
    pre = _integer_vector(oof["pre_corruption_label"], n, f"{cell.cell_id} pre labels")
    observed = _integer_vector(oof["observed_label"], n, f"{cell.cell_id} observed labels")
    fold_id = _integer_vector(oof["fold_id"], n, f"{cell.cell_id} fold IDs")
    assignment = _integer_vector(
        oof["fold_assignment_labels"], n, f"{cell.cell_id} fold-assignment labels"
    )
    injected_raw = oof["is_injected_corruption"]
    if injected_raw.shape != (n,) or not np.issubdtype(injected_raw.dtype, np.bool_):
        raise ValueError(f"{cell.cell_id} corruption flags are not aligned booleans")
    injected = np.asarray(injected_raw, dtype=bool)
    coverage = _integer_vector(oof["coverage_count"], n, f"{cell.cell_id} OOF coverage")
    probabilities = np.asarray(oof["probabilities"], dtype=np.float64)
    if (
        not np.array_equal(injected, observed != pre)
        or not np.array_equal(assignment, pre)
        or not np.array_equal(coverage, np.ones(n, dtype=np.int64))
        or probabilities.shape != (n, 5)
        or not np.isfinite(probabilities).all()
        or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise ValueError(f"{cell.cell_id} OOF evidence violates frozen alignment")

    risk_archive = _load_npz(directory / "risk_scores.npz", f"{cell.cell_id} risk evidence")
    risk_ids = _string_vector(risk_archive.get("sample_ids", np.asarray([])), "risk sample IDs")
    if not np.array_equal(risk_ids, sample_ids):
        raise ValueError(f"{cell.cell_id} risk sample order differs from OOF evidence")
    risks: dict[str, NDArray[np.float64]] = {}
    for risk_id, array_name in _RISK_ARRAY_BY_ID.items():
        raw = risk_archive.get(array_name)
        if raw is None:
            raise ValueError(f"{cell.cell_id} lacks frozen risk {risk_id}")
        values = np.asarray(raw, dtype=np.float64)
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError(f"{cell.cell_id} risk {risk_id} is malformed")
        risks[risk_id] = values
    group_tokens = np.asarray(
        [f"fold_{cell.outer_fold}::{value}" for value in groups.tolist()], dtype=np.str_
    )
    return _CellEvidence(
        cell=cell,
        sample_ids=sample_ids,
        group_ids=groups,
        group_tokens=group_tokens,
        pre_corruption_label=pre,
        observed_label=observed,
        injected=injected,
        fold_id=fold_id,
        fold_assignment_labels=assignment,
        risks=risks,
    )


def _validate_skipped_cell(
    run: Path,
    cell: ConfirmatoryCell,
    controls: ConfirmatoryExecutionControls,
) -> Mapping[str, Any]:
    if cell.required:
        raise ValueError(f"required confirmatory cell cannot be skipped: {cell.cell_id}")
    directory = run / "cells" / cell.cell_id
    _verify_cell_manifest(directory, "skipped_with_frozen_blocker")
    scenario = controls.scenarios_by_id[cell.scenario_id]
    cache_record = controls.cache_provenance_by_id[scenario.cache_provenance_id]
    blocker = _read_json(directory / "blocker.json", f"{cell.cell_id} blocker")
    if (
        cache_record.get("status") != "unavailable_with_frozen_blocker"
        or blocker.get("cell_id") != cell.cell_id
        or blocker.get("frozen_unavailability") is not True
        or blocker.get("cache_provenance_id") != scenario.cache_provenance_id
        or blocker.get("config_semantic_sha256") != controls.config_semantic_sha256
        or blocker.get("blocker_evidence_sha256") != cache_record.get("blocker_evidence_sha256")
        or blocker.get("blocker_evidence_sha256") != scenario.availability_audit_sha256
        or not str(blocker.get("blocker", "")).strip()
    ):
        raise ValueError(f"{cell.cell_id} is not an exact frozen optional blocker")
    return blocker


def _same_array(left: NDArray[np.generic], right: NDArray[np.generic]) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and bool(np.array_equal(left, right))
    )


def _load_matrix(run: Path, controls: ConfirmatoryExecutionControls) -> _LoadedMatrix:
    controls.validate_for_plan(controls.plan)
    if not run.is_dir():
        raise FileNotFoundError(f"confirmatory run directory does not exist: {run}")
    if _read_json(run / "matrix_plan.json", "confirmatory matrix plan") != (
        controls.plan.as_dict()
    ):
        raise ValueError("filesystem matrix plan differs from frozen typed controls")
    saved_controls = _read_json(run / "execution_controls.json", "execution controls")
    if _canonical_sha256(saved_controls) != _canonical_sha256(controls.as_dict()):
        raise ValueError("filesystem execution controls differ from frozen typed controls")
    reconciliation = _read_json(run / "reconciliation.json", "matrix reconciliation")
    if reconciliation.get("status") != "passed":
        raise ValueError("confirmatory statistics require passed matrix reconciliation")

    rows = _read_cell_index(run / "cell_index.csv")
    planned_ids = {cell.cell_id for cell in controls.plan.cells}
    if set(rows) != planned_ids:
        raise ValueError("cell index differs from the exact frozen plan")
    completed: dict[str, _CellEvidence] = {}
    status_by_id: dict[str, str] = {}
    blockers_by_scenario: dict[str, Mapping[str, Any]] = {}
    for cell in controls.plan.cells:
        row = rows[cell.cell_id]
        status = str(row.get("status", ""))
        if status not in {"completed", "skipped_with_frozen_blocker"}:
            raise ValueError(f"cell {cell.cell_id} has non-statistical status {status!r}")
        manifest_path = run / "cells" / cell.cell_id / "artifact_manifest.json"
        if row.get("artifact_manifest_sha256") != sha256_file(manifest_path):
            raise ValueError(f"cell-index manifest SHA differs for {cell.cell_id}")
        status_by_id[cell.cell_id] = status
        if status == "completed":
            evidence = _load_completed_cell(run, cell, controls)
            completed[cell.cell_id] = evidence
            metrics_path = run / "cells" / cell.cell_id / "metrics.json"
            if row.get("metrics_sha256") != sha256_file(metrics_path):
                raise ValueError(f"cell-index metrics SHA differs for {cell.cell_id}")
        else:
            blocker = _validate_skipped_cell(run, cell, controls)
            previous = blockers_by_scenario.setdefault(cell.scenario_id, blocker)
            if previous.get("blocker") != blocker.get("blocker") or previous.get(
                "blocker_evidence_sha256"
            ) != blocker.get("blocker_evidence_sha256"):
                raise ValueError(f"scenario {cell.scenario_id} has inconsistent blockers")

    fold_binding: dict[int, tuple[NDArray[np.generic], ...]] = {}
    corruption_binding: dict[tuple[int, str], tuple[NDArray[np.generic], ...]] = {}
    for evidence in completed.values():
        fold_value = (
            evidence.sample_ids,
            evidence.group_ids,
            evidence.pre_corruption_label,
            evidence.fold_id,
            evidence.fold_assignment_labels,
        )
        previous_fold = fold_binding.setdefault(evidence.cell.outer_fold, fold_value)
        if any(
            not _same_array(left, right)
            for left, right in zip(previous_fold, fold_value, strict=True)
        ):
            raise ValueError("completed cells change sample/group/pre-label/fold mapping")
        corruption_value = (
            evidence.pre_corruption_label,
            evidence.observed_label,
            evidence.injected,
        )
        corruption_key = (evidence.cell.outer_fold, evidence.cell.corruption_cell_id)
        previous_corruption = corruption_binding.setdefault(corruption_key, corruption_value)
        if any(
            not _same_array(left, right)
            for left, right in zip(previous_corruption, corruption_value, strict=True)
        ):
            raise ValueError("completed cells change a frozen corruption mapping")

    group_universe: list[str] = []
    for outer_fold in sorted(controls.official_folds):
        representative = next(
            (
                evidence
                for evidence in completed.values()
                if evidence.cell.outer_fold == outer_fold and evidence.cell.required
            ),
            None,
        )
        if representative is None:
            raise ValueError(f"outer fold {outer_fold} has no completed required cell")
        group_universe.extend(
            f"fold_{outer_fold}::{group}"
            for group in sorted(set(representative.group_ids.tolist()))
        )
    if len(group_universe) < 2 or len(set(group_universe)) != len(group_universe):
        raise ValueError("confirmatory bootstrap group universe is invalid")
    return _LoadedMatrix(
        completed=completed,
        status_by_id=status_by_id,
        blocker_by_scenario=blockers_by_scenario,
        group_universe=np.asarray(group_universe, dtype=np.str_),
    )


def _cell_key(cell: ConfirmatoryCell) -> tuple[int, str, int]:
    return cell.outer_fold, cell.corruption_cell_id, cell.model_seed


def _selected_cells(
    operand: ConfirmatoryComparisonOperand,
    controls: ConfirmatoryExecutionControls,
) -> tuple[ConfirmatoryCell, ...]:
    if operand.model_seed != "matched":
        raise ValueError("confirmatory statistics require model_seed=matched")
    selected = tuple(
        cell
        for cell in controls.plan.cells
        if cell.scenario_id == operand.scenario_id
        and (operand.outer_fold == "all_matched" or cell.outer_fold == operand.outer_fold)
        and (
            operand.corruption_cell == "all_matched"
            or cell.corruption_cell_id == operand.corruption_cell
        )
    )
    if not selected:
        raise ValueError(f"operand {operand.scenario_id}/{operand.risk_id} selects no cells")
    return tuple(sorted(selected, key=_cell_key))


def _optional_blocker_result(
    comparison: ConfirmatoryPairedComparison,
    matrix: _LoadedMatrix,
    controls: ConfirmatoryExecutionControls,
) -> Mapping[str, Any] | None:
    selected_by_scenario: dict[str, tuple[ConfirmatoryCell, ...]] = {}
    for operand in (comparison.operand_a, comparison.operand_b):
        scenario = controls.scenarios_by_id[operand.scenario_id]
        if scenario.required:
            continue
        selected = selected_by_scenario.setdefault(
            operand.scenario_id, _selected_cells(operand, controls)
        )
        statuses = {matrix.status_by_id[cell.cell_id] for cell in selected}
        if statuses == {"skipped_with_frozen_blocker"}:
            blocker = matrix.blocker_by_scenario.get(operand.scenario_id)
            if blocker is None:
                raise ValueError("optional comparison lacks its frozen blocker artifact")
            return blocker
        if statuses != {"completed"}:
            raise ValueError("optional comparison is only partially available")
    return None


def _selected_injected_event_count(
    comparison: ConfirmatoryPairedComparison,
    matrix: _LoadedMatrix,
    controls: ConfirmatoryExecutionControls,
) -> int:
    """Count selected events once per frozen fold/corruption population."""

    operand = comparison.operand_a
    selected_folds = (
        sorted({cell.outer_fold for cell in controls.plan.cells})
        if operand.outer_fold == "all_matched"
        else [operand.outer_fold]
    )
    selected_corruptions = (
        sorted({cell.corruption_cell_id for cell in controls.plan.cells})
        if operand.corruption_cell == "all_matched"
        else [operand.corruption_cell]
    )
    total = 0
    for outer_fold in selected_folds:
        for corruption_id in selected_corruptions:
            representative = next(
                (
                    evidence
                    for evidence in matrix.completed.values()
                    if evidence.cell.outer_fold == outer_fold
                    and evidence.cell.corruption_cell_id == corruption_id
                    and evidence.cell.required
                ),
                None,
            )
            if representative is None:
                raise ValueError(
                    "paired comparison cannot bind selected corruption events to a "
                    "completed required cell"
                )
            total += int(representative.injected.sum())
    return total


def _paired_inputs(
    comparison: ConfirmatoryPairedComparison,
    matrix: _LoadedMatrix,
    controls: ConfirmatoryExecutionControls,
) -> tuple[NDArray[np.str_], NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    if comparison.metric != "average_precision":
        raise ValueError("cell-only confirmatory statistics cannot evaluate final-label macro_f1")
    selected_a = {_cell_key(cell): cell for cell in _selected_cells(comparison.operand_a, controls)}
    selected_b = {_cell_key(cell): cell for cell in _selected_cells(comparison.operand_b, controls)}
    if set(selected_a) != set(selected_b):
        raise ValueError(f"{comparison.comparison_id} operands do not select matched cells")
    groups: list[NDArray[np.str_]] = []
    injected: list[NDArray[np.bool_]] = []
    scores_a: list[NDArray[np.float64]] = []
    scores_b: list[NDArray[np.float64]] = []
    for key in sorted(selected_a):
        cell_a = selected_a[key]
        cell_b = selected_b[key]
        data_a = matrix.completed.get(cell_a.cell_id)
        data_b = matrix.completed.get(cell_b.cell_id)
        if data_a is None or data_b is None:
            raise ValueError(f"{comparison.comparison_id} lacks a completed required operand")
        for name in (
            "sample_ids",
            "group_ids",
            "pre_corruption_label",
            "observed_label",
            "injected",
            "fold_id",
            "fold_assignment_labels",
        ):
            if not _same_array(
                cast(NDArray[np.generic], getattr(data_a, name)),
                cast(NDArray[np.generic], getattr(data_b, name)),
            ):
                raise ValueError(
                    f"{comparison.comparison_id} operand cells are not exactly aligned"
                )
        groups.append(data_a.group_tokens)
        injected.append(data_a.injected)
        scores_a.append(data_a.risks[comparison.operand_a.risk_id])
        scores_b.append(data_b.risks[comparison.operand_b.risk_id])
    return (
        np.concatenate(groups),
        np.concatenate(injected),
        np.concatenate(scores_a),
        np.concatenate(scores_b),
    )


def _bootstrap_comparison(
    group_tokens: NDArray[np.str_],
    injected: NDArray[np.bool_],
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    group_draws: NDArray[np.str_],
) -> _ComparisonArrays:
    members = {
        group: np.flatnonzero(group_tokens == group) for group in sorted(set(group_tokens.tolist()))
    }
    valid_mask = np.zeros(len(group_draws), dtype=bool)
    metric_a: list[float] = []
    metric_b: list[float] = []
    empty = np.asarray([], dtype=np.int64)
    for draw_index, draw in enumerate(group_draws):
        selected = [members.get(str(group), empty) for group in draw]
        selected = [indices for indices in selected if len(indices)]
        if not selected:
            continue
        indices = np.concatenate(selected)
        value_a = average_precision(injected[indices], scores_a[indices])
        value_b = average_precision(injected[indices], scores_b[indices])
        if value_a is None or value_b is None:
            continue
        if not math.isfinite(value_a) or not math.isfinite(value_b):
            continue
        valid_mask[draw_index] = True
        metric_a.append(float(value_a))
        metric_b.append(float(value_b))
    return _ComparisonArrays(
        valid_draw_mask=valid_mask,
        metric_a=np.asarray(metric_a, dtype=np.float64),
        metric_b=np.asarray(metric_b, dtype=np.float64),
    )


def _two_sided_probability(differences: NDArray[np.float64]) -> float:
    non_positive = float(np.mean(differences <= 0.0))
    non_negative = float(np.mean(differences >= 0.0))
    return min(1.0, 2.0 * min(non_positive, non_negative))


def _holm_adjust(
    records: Sequence[tuple[str, str, float]],
) -> dict[str, float]:
    adjusted: dict[str, float] = {}
    for family in sorted({value[1] for value in records}):
        ordered = sorted(
            (
                (comparison_id, raw_p)
                for comparison_id, record_family, raw_p in records
                if record_family == family
            ),
            key=lambda value: (value[1], value[0]),
        )
        running = 0.0
        count = len(ordered)
        for index, (comparison_id, raw_p) in enumerate(ordered):
            running = max(running, min(1.0, (count - index) * raw_p))
            adjusted[comparison_id] = running
    return adjusted


def _empty_result(
    comparison: ConfirmatoryPairedComparison,
    controls: ConfirmatoryExecutionControls,
    *,
    status: str,
) -> dict[str, Any]:
    return {
        **comparison.as_dict(),
        "status": status,
        "paired_unit": controls.statistical_group_unit,
        "bootstrap_seed": controls.bootstrap_seed,
        "requested_iterations": controls.paired_group_bootstrap_iterations,
        "valid_iterations": 0,
        "observed_delta": None,
        "ci_low": None,
        "ci_high": None,
        "probability_positive": None,
        "raw_p": None,
        "holm_adjusted_p": None,
    }


def _compute_statistics(run: Path, controls: ConfirmatoryExecutionControls) -> _ComputedStatistics:
    matrix = _load_matrix(run, controls)
    iterations = controls.paired_group_bootstrap_iterations
    if iterations < 2_000:
        raise ValueError("confirmatory statistics require at least 2,000 frozen iterations")
    rng = np.random.default_rng(controls.bootstrap_seed)
    group_draws = np.stack(
        [
            rng.choice(
                matrix.group_universe,
                size=len(matrix.group_universe),
                replace=True,
            )
            for _ in range(iterations)
        ]
    )
    arrays: dict[str, NDArray[np.generic]] = {
        "bootstrap_group_universe": matrix.group_universe,
        "bootstrap_group_draws": group_draws,
    }
    results: list[dict[str, Any]] = []
    holm_inputs: list[tuple[str, str, float]] = []
    for comparison in controls.paired_comparisons:
        comparison_id = comparison.comparison_id
        selected_event_count = _selected_injected_event_count(comparison, matrix, controls)
        blocker = _optional_blocker_result(comparison, matrix, controls)
        if blocker is not None:
            result = _empty_result(
                comparison,
                controls,
                status="not_estimable_frozen_optional_blocker",
            )
            result.update(
                {
                    "frozen_unavailability": True,
                    "blocker": str(blocker["blocker"]),
                    "availability_audit_sha256": str(blocker["blocker_evidence_sha256"]),
                }
            )
            comparison_arrays = _ComparisonArrays(
                valid_draw_mask=np.zeros(iterations, dtype=bool),
                metric_a=np.asarray([], dtype=np.float64),
                metric_b=np.asarray([], dtype=np.float64),
            )
        else:
            groups, injected, scores_a, scores_b = _paired_inputs(comparison, matrix, controls)
            if not injected.any():
                result = _empty_result(
                    comparison,
                    controls,
                    status="not_applicable_zero_event",
                )
                comparison_arrays = _ComparisonArrays(
                    valid_draw_mask=np.zeros(iterations, dtype=bool),
                    metric_a=np.asarray([], dtype=np.float64),
                    metric_b=np.asarray([], dtype=np.float64),
                )
            else:
                comparison_arrays = _bootstrap_comparison(
                    groups,
                    injected,
                    scores_a,
                    scores_b,
                    group_draws,
                )
                differences = comparison_arrays.differences
                if not len(differences):
                    raise ValueError(f"{comparison_id} has no valid bootstrap iterations")
                if np.all(comparison_arrays.metric_a == 0.0) and np.all(
                    comparison_arrays.metric_b == 0.0
                ):
                    raise ValueError(f"{comparison_id} produced an all-zero placeholder")
                ci_low, ci_high = (
                    float(value) for value in np.quantile(differences, (0.025, 0.975))
                )
                raw_p = _two_sided_probability(differences)
                result = {
                    **comparison.as_dict(),
                    "status": "completed",
                    "paired_unit": controls.statistical_group_unit,
                    "bootstrap_seed": controls.bootstrap_seed,
                    "requested_iterations": iterations,
                    "valid_iterations": len(differences),
                    "observed_delta": float(differences.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "probability_positive": float(
                        np.mean(differences > 0.0) + 0.5 * np.mean(differences == 0.0)
                    ),
                    "raw_p": raw_p,
                    "holm_adjusted_p": None,
                }
                holm_inputs.append((comparison_id, comparison.holm_family, raw_p))
        result["selected_injected_event_count"] = selected_event_count
        arrays[f"valid_draw_mask__{comparison_id}"] = comparison_arrays.valid_draw_mask
        arrays[f"metric_a__{comparison_id}"] = comparison_arrays.metric_a
        arrays[f"metric_b__{comparison_id}"] = comparison_arrays.metric_b
        arrays[f"differences__{comparison_id}"] = comparison_arrays.differences
        results.append(result)

    adjusted = _holm_adjust(holm_inputs)
    for result in results:
        comparison_id = str(result["comparison_id"])
        if comparison_id in adjusted:
            result["holm_adjusted_p"] = adjusted[comparison_id]
    return _ComputedStatistics(results=tuple(results), arrays=arrays)


def _statistics_payload(
    computed: _ComputedStatistics,
    controls: ConfirmatoryExecutionControls,
    bootstrap_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_semantic_sha256": controls.config_semantic_sha256,
        "execution_controls_binding_sha256": controls.binding_sha256,
        "outer_folds": list(controls.official_folds),
        "paired_unit": controls.statistical_group_unit,
        "bootstrap_iterations": controls.paired_group_bootstrap_iterations,
        "bootstrap_seed": controls.bootstrap_seed,
        "bootstrap_evidence_path": _BOOTSTRAP_FILE,
        "bootstrap_evidence_sha256": bootstrap_sha256,
        "comparisons": list(computed.results),
    }


def _atomic_npz(path: Path, arrays: Mapping[str, NDArray[np.generic]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **cast(dict[str, Any], dict(arrays)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _arrays_equal(actual: NDArray[np.generic], expected: NDArray[np.generic]) -> bool:
    return bool(
        actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and np.array_equal(actual, expected)
    )


def aggregate_confirmatory_statistics(
    run_directory: str | Path,
    controls: ConfirmatoryExecutionControls,
) -> ConfirmatoryStatisticsArtifacts:
    """Compute and persist all and only frozen confirmatory comparisons."""

    run = Path(run_directory).resolve()
    for name in (_STATISTICS_FILE, _BOOTSTRAP_FILE):
        if (run / name).exists():
            raise FileExistsError(f"confirmatory statistics never overwrite {name}")
    computed = _compute_statistics(run, controls)
    bootstrap_path = _atomic_npz(run / _BOOTSTRAP_FILE, computed.arrays)
    statistics_path = atomic_write_json(
        run / _STATISTICS_FILE,
        _statistics_payload(computed, controls, sha256_file(bootstrap_path)),
    )
    verification = verify_confirmatory_statistics_artifacts(run, controls)
    return ConfirmatoryStatisticsArtifacts(
        output_directory=run,
        statistics_path=statistics_path,
        bootstrap_evidence_path=bootstrap_path,
        comparison_count=len(computed.results),
        completed_comparison_count=sum(
            value["status"] == "completed" for value in computed.results
        ),
        statistics_sha256=verification.statistics_sha256,
        bootstrap_evidence_sha256=verification.bootstrap_evidence_sha256,
    )


def verify_confirmatory_statistics_artifacts(
    run_directory: str | Path,
    controls: ConfirmatoryExecutionControls,
) -> ConfirmatoryStatisticsVerification:
    """Recompute from cell evidence and reject any semantic or byte-level tampering."""

    run = Path(run_directory).resolve()
    for name in (_STATISTICS_FILE, _BOOTSTRAP_FILE):
        if not (run / name).is_file():
            raise ValueError(f"confirmatory statistics artifact is missing: {name}")
    computed = _compute_statistics(run, controls)
    saved_arrays = _load_npz(run / _BOOTSTRAP_FILE, "paired bootstrap evidence")
    if set(saved_arrays) != set(computed.arrays):
        raise ValueError("paired bootstrap evidence has missing or extra arrays")
    for name, expected in computed.arrays.items():
        if not _arrays_equal(saved_arrays[name], expected):
            raise ValueError(f"paired bootstrap array differs from recomputation: {name}")
    expected_payload = _statistics_payload(
        computed,
        controls,
        sha256_file(run / _BOOTSTRAP_FILE),
    )
    saved_payload = _read_json(run / _STATISTICS_FILE, "paired statistics")
    if saved_payload != expected_payload:
        raise ValueError("paired_statistics.json differs from recomputed cell evidence")
    return ConfirmatoryStatisticsVerification(
        status="passed",
        output_directory=run,
        statistics_sha256=sha256_file(run / _STATISTICS_FILE),
        bootstrap_evidence_sha256=sha256_file(run / _BOOTSTRAP_FILE),
        comparison_count=len(computed.results),
        completed_comparison_count=sum(
            value["status"] == "completed" for value in computed.results
        ),
    )


__all__ = [
    "ConfirmatoryStatisticsArtifacts",
    "ConfirmatoryStatisticsVerification",
    "aggregate_confirmatory_statistics",
    "verify_confirmatory_statistics_artifacts",
]
