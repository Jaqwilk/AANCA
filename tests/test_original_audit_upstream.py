"""Fail-closed provenance tests for stage-eligible original-label auditing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import histo_audit.external_validation.eligibility as eligibility_module
import histo_audit.workflows.tracked_original_audit as tracked_module
from histo_audit.cli import app
from histo_audit.config import config_sha256
from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.external_validation import (
    ORIGINAL_AUDIT_FEATURE_SCOPE,
    ordered_sample_ids_sha256,
    verify_original_audit_upstream,
)
from histo_audit.utils.run_tracking import (
    RunTracker,
    sha256_file,
    sha256_path,
    verify_run_integrity,
)
from histo_audit.workflows.original_audit import OriginalLabelAuditResult
from histo_audit.workflows.tracked_original_audit import run_tracked_original_label_audit


def _selection() -> dict[str, Any]:
    return {
        "scenario_id": "imagenet_frozen_logistic",
        "representation_id": "imagenet_resnet18_context_embeddings",
        "model_seed": 303,
        "risk_method": "self_confidence",
        "n_splits": 2,
        "cache_provenance_id": "imagenet-context-audit-cache",
        "classifier": {
            "id": "multinomial_logistic_regression",
            "parameters": {"l2": 1.0, "max_iter": 1000, "class_weight": "balanced"},
        },
    }


def _cache_record(cache: Path, manifest: Path, sample_ids: tuple[str, ...]) -> dict[str, Any]:
    return {
        "id": "imagenet-context-audit-cache",
        "representation_id": "imagenet_resnet18_context_embeddings",
        "status": "available",
        "cache_file_sha256": sha256_file(cache),
        "sidecar_semantic_sha256": None,
        "sample_order_sha256": ordered_sample_ids_sha256(sample_ids),
        "manifest_sha256": sha256_file(manifest),
        "encoder_identifier": "resnet18_imagenet1k_v1",
        "encoder_metadata_sha256": "c" * 64,
        "weight_identifier": "ResNet18_Weights.IMAGENET1K_V1",
        "weights_sha256": ("f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"),
        "preprocessing_identifier": "torchvision_resnet18_imagenet1k_v1_official",
        "preprocessing_sha256": "d" * 64,
        "input_variant": "context_rgb",
    }


def _write_sidecar(
    *,
    path: Path,
    run: Path,
    cache: Path,
    manifest: Path,
    final_groups: Path,
    sample_ids: tuple[str, ...],
    config: dict[str, Any],
) -> None:
    integrity = verify_run_integrity(run)
    selection = _selection()
    cache_record = config["cache_provenance"][0]
    encoder = {
        field: cache_record[field]
        for field in (
            "encoder_identifier",
            "encoder_metadata_sha256",
            "weight_identifier",
            "weights_sha256",
            "preprocessing_identifier",
            "preprocessing_sha256",
            "input_variant",
        )
    }
    payload = {
        "schema_version": 1,
        "artifact_scope": ORIGINAL_AUDIT_FEATURE_SCOPE,
        "study_outcome_eligible": True,
        "feature_cache": {"path": str(cache), "sha256": sha256_file(cache)},
        "representation_id": selection["representation_id"],
        "sample_count": len(sample_ids),
        "sample_order_sha256": ordered_sample_ids_sha256(sample_ids),
        "audit_sample_ids": list(sample_ids),
        "audit_sample_count": len(sample_ids),
        "audit_sample_order_sha256": ordered_sample_ids_sha256(sample_ids),
        "manifest_sha256": sha256_file(manifest),
        "final_reference_groups_sha256": sha256_file(final_groups),
        "final_reference_group_ids_sha256": canonical_sha256(["held-out-group"]),
        "confirmatory_outer_fold": 3,
        "cache_provenance_id": selection["cache_provenance_id"],
        "encoder_metadata": encoder,
        "frozen_bindings": {
            "confirmatory_run_id": integrity.run_id,
            "confirmatory_artifact_root_sha256": integrity.actual_root_sha256,
            "confirmatory_completion_sha256": sha256_file(run / "completion_evidence.json"),
            "confirmatory_plan_sha256": sha256_file(run / "matrix_plan.json"),
            "confirmatory_resolved_config_sha256": sha256_file(run / "resolved_config.yaml"),
            "frozen_primary_config_sha256": "a" * 64,
            "frozen_confirmatory_config_sha256": "b" * 64,
            "confirmatory_config_semantic_sha256": config_sha256(config),
            "original_audit_selection_sha256": canonical_sha256(selection),
            "frozen_feature_provenance_sha256": sha256_file(run / "frozen_feature_provenance.json"),
            "original_audit_selection_artifact_sha256": sha256_file(
                run / "original_audit_selection.json"
            ),
            "confirmatory_execution_controls_sha256": sha256_file(run / "execution_controls.json"),
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _upstream_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seal: bool = True,
    artifact_scope: str = "real_pannuke_confirmatory_study",
) -> dict[str, Path]:
    monkeypatch.setattr(
        eligibility_module,
        "validate_frozen_confirmatory_config",
        lambda value: dict(value),
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "source.bin").write_bytes(b"verified-fixture-source")
    sample_ids = tuple(f"sample-{index}" for index in range(6))
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "group_id": [f"group-{index}" for index in range(6)],
            "tissue_type": ["fixture"] * 6,
            "pre_corruption_label": [0, 1, 2, 3, 4, 0],
            "observed_label": [0, 1, 2, 3, 4, 0],
            "is_injected_corruption": [False] * 6,
        }
    ).to_csv(manifest, index=False)
    cache = tmp_path / "features.npz"
    np.savez(
        cache,
        features=np.arange(18, dtype=np.float64).reshape(6, 3),
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
    )
    final_groups = tmp_path / "final-groups.txt"
    final_groups.write_text("held-out-group\n", encoding="utf-8")
    selection = _selection()
    cache_record = _cache_record(cache, manifest, sample_ids)
    config: dict[str, Any] = {
        "schema_version": 2,
        "experiment_name": "confirmatory_study",
        "data": {"split_seed": 223},
        "cache_provenance": [cache_record],
        "original_audit_selection": selection,
    }
    tracker = RunTracker.start(
        experiment_name="confirmatory_study",
        config=config,
        project_root=tmp_path,
        runs_root=tmp_path / "confirmatory-runs",
        dataset_path=dataset,
        manifest_path=manifest,
    )
    semantic_config_sha = config_sha256(config)
    tracker.write_json(
        "execution_controls.json",
        {
            "schema_version": 1,
            "binding_sha256": "9" * 64,
            "original_audit_selection": selection,
        },
    )
    tracker.write_json(
        "matrix_plan.json",
        {"schema_version": 1, "config_sha256": semantic_config_sha, "cells": []},
    )
    provenance_record = {
        "cache_provenance_id": cache_record["id"],
        "representation_id": cache_record["representation_id"],
        "cache_file_sha256": cache_record["cache_file_sha256"],
        "sidecar_semantic_sha256": cache_record["sidecar_semantic_sha256"],
        "sample_order_sha256": cache_record["sample_order_sha256"],
        "manifest_sha256": cache_record["manifest_sha256"],
        "encoder_identifier": cache_record["encoder_identifier"],
        "encoder_metadata_sha256": cache_record["encoder_metadata_sha256"],
        "weight_identifier": cache_record["weight_identifier"],
        "weights_sha256": cache_record["weights_sha256"],
        "preprocessing_identifier": cache_record["preprocessing_identifier"],
        "preprocessing_sha256": cache_record["preprocessing_sha256"],
        "input_variant": cache_record["input_variant"],
        "audit_sample_order_sha256": ordered_sample_ids_sha256(sample_ids),
    }
    tracker.write_json(
        "frozen_feature_provenance.json",
        {
            "schema_version": 1,
            "status": "completed",
            "confirmatory_config_semantic_sha256": semantic_config_sha,
            "matrix_plan_config_sha256": semantic_config_sha,
            "representations": {
                selection["representation_id"]: {
                    "rotations": {"3": provenance_record},
                }
            },
        },
    )
    tracker.write_json(
        "original_audit_selection.json",
        {
            "schema_version": 1,
            "status": "completed",
            "confirmatory_config_semantic_sha256": semantic_config_sha,
            "matrix_plan_config_sha256": semantic_config_sha,
            "execution_controls_binding_sha256": "9" * 64,
            "selection_semantic_sha256": canonical_sha256(selection),
            "selection": selection,
            "frozen_cache_provenance_record": cache_record,
            "sealed_feature_cache_provenance_by_rotation": {
                "3": {
                    **provenance_record,
                    "final_reference_group_ids_sha256": canonical_sha256(["held-out-group"]),
                }
            },
        },
    )
    tracker.write_json(
        "reconciliation.json",
        {"status": "passed", "fold_rotation_complete": True},
    )
    tracker.write_json(
        "completion_evidence.json",
        {
            "schema_version": 1,
            **{
                field: "e" * 64
                for field in (
                    "freeze_artifact_root_sha256",
                    "freeze_manifest_sha256",
                    "preregistration_sha256",
                    "primary_config_semantic_sha256",
                    "pilot_artifact_root_sha256",
                    "duplicate_audit_sha256",
                    "pathology_encoder_audit_sha256",
                    "source_tree_root_sha256",
                    "primary_artifact_root_sha256",
                    "primary_completion_evidence_sha256",
                    "primary_reconciliation_sha256",
                )
            },
            "completion_stage": "CONFIRMATORY_COMPLETE",
            "study_outcome_eligible": True,
            "artifact_scope": artifact_scope,
            "matrix_config_sha256": semantic_config_sha,
            "required_cell_count": 1,
            "completed_required_cell_count": 1,
            "failed_required_cell_count": 0,
            "reconciliation_status": "passed",
            "fold_rotation_complete": True,
            "planned_outer_folds": [1, 2, 3],
            "completed_outer_folds": [1, 2, 3],
            "primary_run_sealed": True,
            "primary_run_registry_backed": True,
            "primary_completion_stage": "PRIMARY_STUDY_COMPLETE",
            "confirmatory_execution_gate_status": "passed",
            "primary_run_id": "primary-fixture-run",
            "dataset_sha256": sha256_path(dataset),
            "manifest_sha256": sha256_file(manifest),
            "frozen_primary_config_sha256": "a" * 64,
            "frozen_confirmatory_config_sha256": "b" * 64,
            "confirmatory_config_semantic_sha256": semantic_config_sha,
        },
    )
    if seal:
        tracker.complete()
    sidecar = tmp_path / "features.original-audit.provenance.json"
    if seal:
        _write_sidecar(
            path=sidecar,
            run=tracker.run_directory,
            cache=cache,
            manifest=manifest,
            final_groups=final_groups,
            sample_ids=sample_ids,
            config=config,
        )
    else:
        sidecar.write_text("{}", encoding="utf-8")
    return {
        "run": tracker.run_directory,
        "cache": cache,
        "sidecar": sidecar,
        "manifest": manifest,
        "final_groups": final_groups,
    }


def _verify(paths: dict[str, Path]):
    return verify_original_audit_upstream(
        confirmatory_run_directory=paths["run"],
        feature_cache_path=paths["cache"],
        feature_cache_provenance_json=paths["sidecar"],
        manifest_path=paths["manifest"],
        final_reference_groups_path=paths["final_groups"],
    )


def test_mocked_sealed_confirmatory_and_exact_cache_are_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _upstream_fixture(tmp_path, monkeypatch)

    result = _verify(paths)

    assert result.eligible, result.errors
    assert result.selection is not None
    assert result.selection.representation_id == "imagenet_resnet18_context_embeddings"
    assert result.selection.risk_method == "self_confidence"


def test_random_cache_is_rejected_even_with_valid_confirmatory_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _upstream_fixture(tmp_path, monkeypatch)
    random_cache = tmp_path / "random-features.npz"
    np.savez(
        random_cache,
        features=np.random.default_rng(9).normal(size=(6, 3)),
        sample_ids=np.asarray([f"sample-{index}" for index in range(6)], dtype=np.str_),
    )
    paths["cache"] = random_cache

    result = _verify(paths)

    assert not result.eligible
    assert any("differs" in error and "cache" in error for error in result.errors)


def test_unsealed_confirmatory_upstream_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _upstream_fixture(tmp_path, monkeypatch, seal=False)

    result = _verify(paths)

    assert not result.eligible
    assert any("integrity verification failed" in error for error in result.errors)


def test_synthetic_confirmatory_scope_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _upstream_fixture(
        tmp_path,
        monkeypatch,
        artifact_scope="synthetic_confirmatory_orchestrator_integration_test",
    )

    result = _verify(paths)

    assert not result.eligible
    assert any("artifact_scope" in error for error in result.errors)


def test_tampered_sealed_confirmatory_artifact_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _upstream_fixture(tmp_path, monkeypatch)
    completion = paths["run"] / "completion_evidence.json"
    payload = json.loads(completion.read_text(encoding="utf-8"))
    payload["completion_stage"] = None
    completion.write_text(json.dumps(payload), encoding="utf-8")

    result = _verify(paths)

    assert not result.eligible
    assert any("integrity verification failed" in error for error in result.errors)


def test_stage_cli_rejects_arbitrary_model_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _upstream_fixture(tmp_path, monkeypatch)
    validation = tmp_path / "validation.json"
    duplicate = tmp_path / "duplicate.json"
    validation.write_text("{}", encoding="utf-8")
    duplicate.write_text("{}", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "audit",
            "original",
            "--project-root",
            str(tmp_path),
            "--manifest",
            str(paths["manifest"]),
            "--feature-cache",
            str(paths["cache"]),
            "--final-reference-groups",
            str(paths["final_groups"]),
            "--output-dir",
            str(tmp_path / "refused-stage-audit"),
            "--dataset",
            str(tmp_path / "dataset"),
            "--dataset-validation-json",
            str(validation),
            "--duplicate-audit-json",
            str(duplicate),
            "--confirmatory-run-dir",
            str(paths["run"]),
            "--feature-cache-provenance-json",
            str(paths["sidecar"]),
            "--method",
            "prediction_margin",
        ],
    )

    assert result.exit_code == 1
    assert "rejects CLI model/selection overrides" in result.output
    assert not (tmp_path / "refused-stage-audit").exists()


def test_tracked_original_audit_uses_only_frozen_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _upstream_fixture(tmp_path, monkeypatch)
    validation = tmp_path / "validation.json"
    duplicate = tmp_path / "duplicate.json"
    validation.write_text("{}", encoding="utf-8")
    duplicate.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(tracked_module, "validate_real_dataset_evidence", lambda *_: ())

    def _fake_audit(
        manifest: Any,
        features: Any,
        feature_sample_ids: Any,
        output_directory: str | Path,
        **controls: Any,
    ) -> OriginalLabelAuditResult:
        del manifest, features, feature_sample_ids
        assert controls["representation"] == "imagenet_resnet18_context_embeddings"
        assert controls["method"] == "self_confidence"
        assert controls["model_seed"] == 303
        assert controls["split_seed"] == 223
        assert controls["l2"] == 1.0
        assert controls["max_iter"] == 1000
        destination = Path(output_directory)
        destination.mkdir(parents=True)
        names = (
            "ranking_all.csv",
            "top_overall.csv",
            "top_per_class.csv",
            "top_per_tissue.csv",
            "oof_predictions.npz",
            "oof_provenance.json",
            "audit_metadata.json",
            "report.md",
        )
        for name in names:
            (destination / name).write_text("fixture\n", encoding="utf-8")
        source_manifest = Path(manifest_path_for_evidence)
        (destination / "audit_metadata.json").write_text(
            json.dumps(
                {
                    "source_manifest": {
                        "path": str(source_manifest),
                        "sha256_before": sha256_file(source_manifest),
                        "sha256_after": sha256_file(source_manifest),
                        "input_mode": "read_only_file",
                    },
                    "observed_equals_pre_corruption": True,
                    "injected_corruption_count": 0,
                    "risk_method": "self_confidence",
                    "frozen_audit_sample_selection": {
                        "enabled": True,
                        "sample_count": 6,
                        "sample_order_sha256": ordered_sample_ids_sha256(
                            tuple(f"sample-{index}" for index in range(6))
                        ),
                    },
                    "group_safe_oof": {
                        "coverage_exactly_once": True,
                        "final_reference_groups_absent_from_audit": True,
                        "representation": "imagenet_resnet18_context_embeddings",
                        "model_seed": 303,
                        "split_seed": 223,
                        "class_order": [0, 1, 2, 3, 4],
                    },
                }
            ),
            encoding="utf-8",
        )
        return OriginalLabelAuditResult(
            output_directory=destination,
            ranking_all_path=destination / names[0],
            top_overall_path=destination / names[1],
            top_per_class_path=destination / names[2],
            top_per_tissue_path=destination / names[3],
            oof_predictions_path=destination / names[4],
            oof_provenance_path=destination / names[5],
            metadata_path=destination / names[6],
            report_path=destination / names[7],
            sample_count=6,
            group_count=6,
            top_overall_count=6,
            top_per_class_count=6,
            top_per_tissue_count=6,
        )

    manifest_path_for_evidence = paths["manifest"]
    monkeypatch.setattr(tracked_module, "audit_original_labels", _fake_audit)
    features, sample_ids = eligibility_module.load_original_audit_feature_cache(paths["cache"])
    output = tmp_path / "original-runs" / "audit-001"

    result = run_tracked_original_label_audit(
        paths["manifest"],
        features,
        sample_ids,
        output,
        dataset_path=tmp_path / "dataset",
        dataset_validation_json=validation,
        duplicate_audit_json=duplicate,
        confirmatory_run_directory=paths["run"],
        feature_cache_path=paths["cache"],
        feature_cache_provenance_json=paths["sidecar"],
        final_reference_groups_path=paths["final_groups"],
        final_reference_group_ids=("held-out-group",),
        project_root=tmp_path,
    )

    assert result.output_directory == output
    assert verify_run_integrity(output).valid
    evidence = json.loads(
        (output / "external_validation_eligibility.json").read_text(encoding="utf-8")
    )
    assert evidence["schema_version"] == 2
    assert evidence["confirmatory_run"]["selected_outer_fold"] == 3
    monkeypatch.setattr(eligibility_module, "validate_real_dataset_evidence", lambda *_: ())
    external = eligibility_module.verify_external_validation_eligibility(
        audit_run_directory=output,
        dataset_path=tmp_path / "dataset",
        manifest_path=paths["manifest"],
        dataset_validation_json=validation,
        duplicate_audit_json=duplicate,
        ranking_path=result.ranking_all_path,
        confirmatory_run_directory=paths["run"],
        feature_cache_provenance_json=paths["sidecar"],
    )
    assert external.eligible, external.errors
