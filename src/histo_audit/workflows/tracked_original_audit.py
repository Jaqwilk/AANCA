"""Registry-backed original-label audit for genuine external-review readiness."""

from __future__ import annotations

import shutil
from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from histo_audit.external_validation.eligibility import (
    ELIGIBILITY_FILENAME,
    ORIGINAL_AUDIT_CLASS_ORDER,
    ORIGINAL_AUDIT_EXPERIMENT_NAME,
    load_original_audit_feature_cache,
    validate_real_dataset_evidence,
    verify_original_audit_upstream,
)
from histo_audit.utils.run_tracking import (
    RunTracker,
    sha256_file,
    verify_run_integrity,
)

from .original_audit import OriginalLabelAuditResult, audit_original_labels


def _file_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def run_tracked_original_label_audit(
    manifest_path: str | Path,
    features: NDArray[np.generic],
    feature_sample_ids: Sequence[str],
    output_directory: str | Path,
    *,
    dataset_path: str | Path,
    dataset_validation_json: str | Path,
    duplicate_audit_json: str | Path,
    confirmatory_run_directory: str | Path,
    feature_cache_path: str | Path,
    feature_cache_provenance_json: str | Path,
    final_reference_groups_path: str | Path,
    final_reference_group_ids: Collection[str],
    project_root: str | Path,
) -> OriginalLabelAuditResult:
    """Execute, seal, and registry-bind one eligible real-data original-label audit."""

    root = Path(project_root).resolve()
    destination = Path(output_directory).resolve()
    manifest = Path(manifest_path).resolve()
    dataset = Path(dataset_path).resolve()
    validation = Path(dataset_validation_json).resolve()
    duplicate = Path(duplicate_audit_json).resolve()
    confirmatory_run = Path(confirmatory_run_directory).resolve()
    feature_cache = Path(feature_cache_path).resolve()
    feature_provenance = Path(feature_cache_provenance_json).resolve()
    final_groups = Path(final_reference_groups_path).resolve()
    if destination.exists():
        raise FileExistsError(f"original-label audit output already exists: {destination}")
    for role, source in (
        ("manifest", manifest),
        ("dataset validation", validation),
        ("duplicate audit", duplicate),
        ("feature cache", feature_cache),
        ("feature-cache provenance", feature_provenance),
        ("final-reference groups", final_groups),
    ):
        if not source.is_file():
            raise FileNotFoundError(f"{role} evidence does not exist: {source}")
    source_errors = validate_real_dataset_evidence(dataset, validation, duplicate)
    if source_errors:
        raise ValueError(f"real-dataset eligibility evidence failed: {source_errors}")

    cached_features, cached_sample_ids = load_original_audit_feature_cache(feature_cache)
    supplied_features = np.asarray(features)
    supplied_sample_ids = tuple(str(value) for value in feature_sample_ids)
    if supplied_sample_ids != cached_sample_ids or not np.array_equal(
        supplied_features, cached_features
    ):
        raise ValueError("in-memory features/sample order differ from the supplied feature cache")
    file_final_groups = tuple(
        line.strip()
        for line in final_groups.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    supplied_final_groups = tuple(str(value) for value in final_reference_group_ids)
    if (
        not file_final_groups
        or len(set(file_final_groups)) != len(file_final_groups)
        or set(file_final_groups) != set(supplied_final_groups)
    ):
        raise ValueError("in-memory final-reference groups differ from their frozen evidence file")
    upstream = verify_original_audit_upstream(
        confirmatory_run_directory=confirmatory_run,
        feature_cache_path=feature_cache,
        feature_cache_provenance_json=feature_provenance,
        manifest_path=manifest,
        final_reference_groups_path=final_groups,
    )
    if not upstream.eligible or upstream.selection is None:
        raise ValueError(f"original-audit upstream eligibility failed: {upstream.errors}")
    selected = upstream.selection

    top_count_overall = 100
    top_count_per_class = 20
    top_count_per_tissue = 20

    config = {
        "schema_version": 2,
        "experiment_name": ORIGINAL_AUDIT_EXPERIMENT_NAME,
        "data": {
            "source": "verified_pannuke_release",
            "class_order": list(ORIGINAL_AUDIT_CLASS_ORDER),
            "final_reference_groups_sha256": sha256_file(final_groups),
            "confirmatory_outer_fold": upstream.selected_outer_fold,
        },
        "model": {
            "scenario_id": selected.scenario_id,
            "representation": selected.representation_id,
            "method": selected.risk_method,
            "classifier": selected.classifier_id,
            "class_weight": selected.class_weight,
            "n_splits": selected.n_splits,
            "split_seed": selected.split_seed,
            "model_seed": selected.model_seed,
            "l2": selected.l2,
            "max_iter": selected.max_iter,
            "frozen_selection_sha256": selected.selection_sha256,
        },
        "selection": {
            "top_count_overall": top_count_overall,
            "top_count_per_class": top_count_per_class,
            "top_count_per_tissue": top_count_per_tissue,
        },
    }
    tracker = RunTracker.start(
        experiment_name=ORIGINAL_AUDIT_EXPERIMENT_NAME,
        config=config,
        project_root=root,
        runs_root=destination.parent,
        registry_path=destination.parent / "registry.csv",
        run_id=destination.name,
        dataset_path=dataset,
        manifest_path=manifest,
        duplicate_audit_status=f"complete_sha256:{sha256_file(duplicate)}",
    )
    build_directory = tracker.run_directory / "_audit_build"
    try:
        temporary = audit_original_labels(
            manifest,
            features,
            feature_sample_ids,
            build_directory,
            final_reference_group_ids=final_reference_group_ids,
            audit_sample_ids=upstream.audit_sample_ids,
            class_order=ORIGINAL_AUDIT_CLASS_ORDER,
            n_splits=selected.n_splits,
            split_seed=selected.split_seed,
            model_seed=selected.model_seed,
            representation=selected.representation_id,
            method=selected.risk_method,
            top_count_overall=top_count_overall,
            top_count_per_class=top_count_per_class,
            top_count_per_tissue=top_count_per_tissue,
            l2=selected.l2,
            max_iter=selected.max_iter,
        )
        for artifact in sorted(build_directory.iterdir(), key=lambda path: path.name):
            artifact.replace(tracker.run_directory / artifact.name)
        build_directory.rmdir()
        result = OriginalLabelAuditResult(
            output_directory=tracker.run_directory,
            ranking_all_path=tracker.run_directory / temporary.ranking_all_path.name,
            top_overall_path=tracker.run_directory / temporary.top_overall_path.name,
            top_per_class_path=tracker.run_directory / temporary.top_per_class_path.name,
            top_per_tissue_path=tracker.run_directory / temporary.top_per_tissue_path.name,
            oof_predictions_path=tracker.run_directory / temporary.oof_predictions_path.name,
            oof_provenance_path=tracker.run_directory / temporary.oof_provenance_path.name,
            metadata_path=tracker.run_directory / temporary.metadata_path.name,
            report_path=tracker.run_directory / temporary.report_path.name,
            sample_count=temporary.sample_count,
            group_count=temporary.group_count,
            top_overall_count=temporary.top_overall_count,
            top_per_class_count=temporary.top_per_class_count,
            top_per_tissue_count=temporary.top_per_tissue_count,
        )
        tracker.write_json(
            ELIGIBILITY_FILENAME,
            {
                "schema_version": 2,
                "workflow": "exploratory_original_label_audit",
                "study_outcome_eligible": True,
                "data_source": "verified_pannuke_release",
                "dataset": {
                    "path": str(dataset),
                    "sha256": tracker.checksums["dataset"]["sha256"],
                },
                "manifest": _file_record(manifest),
                "dataset_validation": _file_record(validation),
                "duplicate_audit": _file_record(duplicate),
                "feature_cache": _file_record(feature_cache),
                "feature_cache_provenance": _file_record(feature_provenance),
                "final_reference_groups": _file_record(final_groups),
                "ranking": _file_record(result.ranking_all_path),
                "confirmatory_run": {
                    "path": str(confirmatory_run),
                    "run_id": upstream.confirmatory_run_id,
                    "artifact_root_sha256": upstream.confirmatory_artifact_root_sha256,
                    "completion_evidence_sha256": upstream.completion_evidence_sha256,
                    "matrix_plan_sha256": upstream.confirmatory_plan_sha256,
                    "resolved_config_sha256": upstream.resolved_config_sha256,
                    "frozen_feature_provenance_sha256": (
                        upstream.feature_provenance_artifact_sha256
                    ),
                    "original_audit_selection_sha256": selected.selection_sha256,
                    "original_audit_selection_artifact_sha256": (
                        upstream.selection_artifact_sha256
                    ),
                    "selected_outer_fold": upstream.selected_outer_fold,
                },
            },
        )
        tracker.write_provenance(
            workflow="exploratory_original_label_audit",
            study_outcome_eligible=True,
            eligibility_evidence=ELIGIBILITY_FILENAME,
        )
        tracker.complete()
    except BaseException as exc:
        shutil.rmtree(build_directory, ignore_errors=True)
        if not tracker.finalized:
            tracker.fail(exc)
        raise

    integrity = verify_run_integrity(tracker.run_directory)
    if not integrity.valid or not integrity.registry_record_present:
        raise RuntimeError(
            f"sealed original-label audit failed integrity verification: {integrity.errors}"
        )
    return result


__all__ = ["run_tracked_original_label_audit"]
