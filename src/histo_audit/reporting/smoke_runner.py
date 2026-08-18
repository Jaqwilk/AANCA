"""Tracked CLI integration for the deterministic synthetic core orchestrator."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from histo_audit.config import load_config, resolve_config
from histo_audit.reporting.builder import ReportArtifacts, build_synthetic_report
from histo_audit.reporting.reconciliation import reconcile_synthetic_smoke_artifacts
from histo_audit.reporting.synthetic_duplicates import audit_synthetic_duplicate_patches
from histo_audit.utils.run_tracking import (
    RunTracker,
    atomic_write_bytes,
    sha256_file,
)


@dataclass(frozen=True, slots=True)
class SmokeExecution:
    """Terminal paths for one tracked synthetic smoke execution."""

    success: bool
    status: str
    run_id: str
    run_directory: Path
    metrics_path: Path
    report: ReportArtifacts
    core_run_id: str | None
    core_run_directory: Path | None


def _result_value(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _load_core_runner() -> Callable[..., Any]:
    """Import the core only when smoke is invoked, keeping all CLI help functional."""

    try:
        module = importlib.import_module("histo_audit.experiment.smoke")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "synthetic smoke orchestrator is unavailable; expected "
            "histo_audit.experiment.smoke.run_synthetic_smoke"
        ) from exc
    runner = getattr(module, "run_synthetic_smoke", None) or getattr(module, "run_smoke", None)
    if not callable(runner):
        raise RuntimeError(
            "synthetic smoke orchestrator is not wired; expected callable "
            "histo_audit.experiment.smoke.run_synthetic_smoke"
        )
    return runner


def smoke_orchestrator_available() -> bool:
    """Return whether the confirmed public smoke entry point is importable."""

    try:
        _load_core_runner()
    except RuntimeError:
        return False
    return True


def _copy_core_artifact(
    tracker: RunTracker,
    result: Any,
    attribute: str,
    destination_name: str,
) -> dict[str, Any] | None:
    raw_path = _result_value(result, attribute)
    if raw_path is None:
        return None
    source = Path(raw_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"core result references missing {attribute}: {source}")
    destination = tracker.run_directory / destination_name
    atomic_write_bytes(destination, source.read_bytes())
    return {
        "source_path": str(source),
        "tracked_path": str(destination.resolve()),
        "sha256": sha256_file(destination),
    }


def _extract_metrics(result: Any) -> dict[str, Any]:
    raw_metrics = _result_value(result, "metrics")
    if isinstance(raw_metrics, Mapping):
        return dict(raw_metrics)
    metrics_path = _result_value(result, "metrics_path")
    if metrics_path is not None and Path(metrics_path).is_file():
        loaded = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        if isinstance(loaded, Mapping):
            return dict(loaded)
    raise RuntimeError("synthetic smoke result did not expose a metrics mapping or metrics JSON")


def execute_synthetic_smoke(
    *,
    project_root: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    runs_root: str | Path | None = None,
) -> SmokeExecution:
    """Run the core pipeline, track provenance, copy results, and build reports."""

    if config is not None and config_path is not None:
        raise ValueError("provide config or config_path, not both")
    root = Path(project_root or Path.cwd()).resolve()
    selected_config = (
        resolve_config(config)
        if config is not None
        else load_config(config_path or root / "configs" / "smoke.yaml")
    )
    experiment_name = str(selected_config.get("experiment_name", "synthetic_smoke"))
    tracker = RunTracker.start(
        experiment_name=experiment_name,
        config=selected_config,
        project_root=root,
        runs_root=runs_root,
    )
    try:
        runner = _load_core_runner()
        core_result = runner(project_root=root, config=selected_config)
        core_success = bool(_result_value(core_result, "success", False))
        core_status = str(_result_value(core_result, "status", "unknown"))
        if not core_success:
            raise RuntimeError(f"synthetic smoke core did not succeed (status={core_status})")
        metrics = _extract_metrics(core_result)
        metrics_path = tracker.write_metrics(metrics)
        copied: dict[str, Any] = {}
        artifact_names = {
            "predictions_path": "oof_predictions.npz",
            "rankings_path": "ranking.csv",
            "corruption_manifest_path": "corruption_manifest.json",
            "oof_provenance_path": "oof_provenance.json",
            "report_inputs_path": "report_inputs.json",
            "representation_example_path": "target_representation_example.npz",
            "neighbour_evidence_path": "neighbour_evidence.npz",
            "restoration_evidence_path": "restoration_evidence.npz",
            "bootstrap_evidence_path": "bootstrap_evidence.npz",
            "dataset_evidence_path": "synthetic_dataset_evidence.npz",
            "source_manifest_path": "synthetic_source_manifest.json",
            "source_manifest_csv_path": "synthetic_source_manifest.csv",
        }
        for attribute, destination in artifact_names.items():
            evidence = _copy_core_artifact(tracker, core_result, attribute, destination)
            if evidence is not None:
                copied[attribute] = evidence
        dataset_evidence_path = tracker.run_directory / "synthetic_dataset_evidence.npz"
        if not dataset_evidence_path.is_file():
            raise RuntimeError("tracked smoke run lacks synthetic_dataset_evidence.npz")
        duplicate_audit = audit_synthetic_duplicate_patches(
            dataset_evidence_path,
            tracker.run_directory,
        )
        tracked_report_inputs_path = tracker.run_directory / "report_inputs.json"
        if not tracked_report_inputs_path.is_file():
            raise RuntimeError("tracked smoke run lacks report_inputs.json")
        tracked_report_inputs = json.loads(tracked_report_inputs_path.read_text(encoding="utf-8"))
        if not isinstance(tracked_report_inputs, dict):
            raise RuntimeError("tracked report_inputs.json root must be an object")
        for attribute, destination_name in artifact_names.items():
            if attribute == "report_inputs_path":
                continue
            if attribute in copied:
                tracked_report_inputs[attribute] = str(
                    (tracker.run_directory / destination_name).resolve()
                )
        tracked_report_inputs["metrics_path"] = str(metrics_path.resolve())
        tracked_report_inputs["report_inputs_path"] = str(tracked_report_inputs_path.resolve())
        tracked_report_inputs["duplicate_audit_path"] = str(duplicate_audit.json_path.resolve())
        tracked_report_inputs["duplicate_candidates_csv_path"] = str(
            duplicate_audit.csv_path.resolve()
        )
        tracked_report_inputs["duplicate_candidates_figure_path"] = str(
            duplicate_audit.figure_path.resolve()
        )
        tracked_report_inputs["run_id"] = tracker.run_id
        tracker.write_json("report_inputs.json", tracked_report_inputs)
        if "report_inputs_path" in copied:
            copied["report_inputs_path"]["tracked_path"] = str(tracked_report_inputs_path.resolve())
            copied["report_inputs_path"]["sha256"] = sha256_file(tracked_report_inputs_path)
        core_metrics_path = _result_value(core_result, "metrics_path")
        if core_metrics_path is not None and Path(core_metrics_path).is_file():
            copied["metrics_path"] = {
                "source_path": str(Path(core_metrics_path).resolve()),
                "tracked_path": str(metrics_path.resolve()),
                "sha256": sha256_file(metrics_path),
            }
        core_run_directory_raw = _result_value(core_result, "run_dir")
        core_run_directory = (
            Path(core_run_directory_raw).resolve() if core_run_directory_raw is not None else None
        )
        core_run_id_raw = _result_value(core_result, "run_id")
        core_run_id = str(core_run_id_raw) if core_run_id_raw is not None else None
        tracker.write_json(
            "core_artifacts.json",
            {
                "core_run_id": core_run_id,
                "core_run_directory": str(core_run_directory) if core_run_directory else None,
                "core_status": core_status,
                "artifacts": copied,
                "tracked_synthetic_duplicate_audit": {
                    "json_path": str(duplicate_audit.json_path.resolve()),
                    "json_sha256": sha256_file(duplicate_audit.json_path),
                    "csv_path": str(duplicate_audit.csv_path.resolve()),
                    "csv_sha256": sha256_file(duplicate_audit.csv_path),
                    "figure_path": str(duplicate_audit.figure_path.resolve()),
                    "figure_sha256": sha256_file(duplicate_audit.figure_path),
                    "real_data_duplicate_gate_eligible": False,
                },
            },
        )
        reconciliation = reconcile_synthetic_smoke_artifacts(
            tracker.run_directory,
            metrics,
        )
        tracker.write_json("artifact_reconciliation.json", reconciliation)
        tracker.log_event(
            "artifact_reconciliation_passed",
            evidence_path="artifact_reconciliation.json",
            sample_count=reconciliation["sample_count"],
            group_count=reconciliation["group_count"],
            oof_fold_count=reconciliation["oof_fold_count"],
        )
        resolved_core_config = metrics.get("resolved_core_config")
        sample_counts = metrics.get("sample_counts")
        artifact_scope = metrics.get("artifact_scope")
        if not isinstance(resolved_core_config, Mapping) or not isinstance(sample_counts, Mapping):
            raise RuntimeError(
                "synthetic metrics lack resolved_core_config/sample_counts definition evidence"
            )
        tracker.write_json(
            "synthetic_dataset_definition.json",
            {
                "definition_type": "deterministic_synthetic_dataset_definition",
                "is_raw_real_dataset_checksum": False,
                "description": (
                    "Fingerprint input for deterministic synthetic generation; this is not a "
                    "checksum of PanNuke or any other raw real dataset."
                ),
                "artifact_scope": artifact_scope,
                "resolved_core_config": dict(resolved_core_config),
                "sample_counts": dict(sample_counts),
            },
        )
        corruption_manifest_path = tracker.run_directory / "corruption_manifest.json"
        source_manifest_path = tracker.run_directory / "synthetic_source_manifest.json"
        source_manifest_csv_path = tracker.run_directory / "synthetic_source_manifest.csv"
        for required_path in (
            corruption_manifest_path,
            dataset_evidence_path,
            source_manifest_path,
            source_manifest_csv_path,
        ):
            if not required_path.is_file():
                raise RuntimeError(f"tracked smoke run lacks {required_path.name}")
        tracker.checksums["dataset"] = {
            "path": str(dataset_evidence_path.resolve()),
            "sha256": sha256_file(dataset_evidence_path),
            "kind": "complete_deterministic_synthetic_arrays",
            "is_raw_real_dataset_checksum": False,
        }
        tracker.checksums["manifest"] = {
            "path": str(source_manifest_path.resolve()),
            "sha256": sha256_file(source_manifest_path),
            "kind": "complete_synthetic_source_manifest",
            "csv_path": str(source_manifest_csv_path.resolve()),
            "csv_sha256": sha256_file(source_manifest_csv_path),
        }
        tracker.checksums["corruption_manifest"] = {
            "path": str(corruption_manifest_path.resolve()),
            "sha256": sha256_file(corruption_manifest_path),
            "kind": "controlled_corruption_manifest",
        }
        tracker.checksums["machine_readable_evidence"] = {
            name: {
                "path": str((tracker.run_directory / name).resolve()),
                "sha256": sha256_file(tracker.run_directory / name),
            }
            for name in (
                "neighbour_evidence.npz",
                "restoration_evidence.npz",
                "bootstrap_evidence.npz",
            )
        }
        tracker.checksums["duplicate_audit_status"] = (
            "synthetic_patch_audit_complete_not_real_data_gate_eligible"
        )
        tracker.checksums["duplicate_audit"] = {
            "json": {
                "path": str(duplicate_audit.json_path.resolve()),
                "sha256": sha256_file(duplicate_audit.json_path),
            },
            "candidate_csv": {
                "path": str(duplicate_audit.csv_path.resolve()),
                "sha256": sha256_file(duplicate_audit.csv_path),
            },
            "candidate_figure": {
                "path": str(duplicate_audit.figure_path.resolve()),
                "sha256": sha256_file(duplicate_audit.figure_path),
            },
            "real_data_duplicate_gate_eligible": False,
        }
        report = build_synthetic_report(
            metrics_path,
            output_directory=tracker.run_directory,
            run_id=tracker.run_id,
            predictions_path=tracker.run_directory / "oof_predictions.npz",
        )
        expected_figures = {
            "average_precision_by_method.png",
            "class_distribution.png",
            "corruption_transition_matrix.png",
            "downstream_macro_f1.png",
            "duplicate_candidates.png",
            "false_high_and_low_risk_examples.png",
            "fold_safe_neighbour_explanation_grid.png",
            "lift_vs_review_budget.png",
            "oof_fold_distribution.png",
            "paired_bootstrap_interval.png",
            "paired_bootstrap_distribution.png",
            "paired_method_differences.png",
            "per_class_results_support.png",
            "per_tissue_results_support.png",
            "precision_recall_curves.png",
            "recall_vs_review_budget.png",
            "target_example_audit_evidence.png",
            "target_representation_example.png",
            "tissue_distribution.png",
            "top_suspicious_controlled_examples.png",
        }
        corruption_metrics = metrics.get("corruption")
        exact_count = (
            corruption_metrics.get("exact_count")
            if isinstance(corruption_metrics, Mapping)
            else None
        )
        if exact_count == 0:
            expected_figures.difference_update(
                {
                    "average_precision_by_method.png",
                    "false_high_and_low_risk_examples.png",
                    "lift_vs_review_budget.png",
                    "paired_bootstrap_interval.png",
                    "paired_bootstrap_distribution.png",
                    "paired_method_differences.png",
                    "precision_recall_curves.png",
                    "recall_vs_review_budget.png",
                }
            )
            expected_figures.update(
                {
                    "false_alerts_vs_review_budget.png",
                    "false_high_risk_examples.png",
                    "score_distribution_by_method.png",
                }
            )
        actual_figures = {path.name for path in report.figure_paths}
        missing_figures = expected_figures.difference(actual_figures)
        if missing_figures:
            raise RuntimeError(
                f"synthetic report lacks required figures: {sorted(missing_figures)}"
            )
        tracker.checksums["figures"] = {
            path.name: sha256_file(path) for path in report.figure_paths
        }
        tracker.checksums["reporting"] = {
            "markdown": {
                "path": str(report.markdown_path.resolve()),
                "sha256": sha256_file(report.markdown_path),
            },
            "html": {
                "path": str(report.html_path.resolve()),
                "sha256": sha256_file(report.html_path),
            },
            "figure_sources": (
                {
                    "path": str(report.figure_manifest_path.resolve()),
                    "sha256": sha256_file(report.figure_manifest_path),
                }
                if report.figure_manifest_path is not None
                else None
            ),
        }
        tracker.write_json("checksums.json", tracker.checksums)
        tracker.write_provenance(
            core_run_id=core_run_id,
            core_run_directory=str(core_run_directory) if core_run_directory else None,
            core_status=core_status,
            checksums=tracker.checksums,
            required_synthetic_figures=sorted(expected_figures),
        )
        tracker.complete()
        return SmokeExecution(
            success=True,
            status=core_status,
            run_id=tracker.run_id,
            run_directory=tracker.run_directory,
            metrics_path=metrics_path,
            report=report,
            core_run_id=core_run_id,
            core_run_directory=core_run_directory,
        )
    except BaseException as error:
        if not tracker.finalized:  # terminal evidence must survive every ordinary failure
            tracker.fail(error)
        raise


# Public wiring name used by the Typer command and tests.
run_tracked_smoke = execute_synthetic_smoke


__all__ = [
    "SmokeExecution",
    "execute_synthetic_smoke",
    "run_tracked_smoke",
    "smoke_orchestrator_available",
]
