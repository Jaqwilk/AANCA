"""Fail-closed external-readiness and transactional package tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from cli_contracts import cli_options
from PIL import Image
from typer.testing import CliRunner

import histo_audit.external_validation as external_validation
import histo_audit.external_validation.validation as validation_module
from histo_audit.cli import app
from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.external_validation import (
    ReviewPackageValidationResult,
    build_blinded_review_package,
    validate_real_dataset_evidence,
    verify_external_validation_eligibility,
)
from histo_audit.pannuke.models import OFFICIAL_METRICS_CLASS_MAPPING
from histo_audit.utils.run_tracking import RunTracker, sha256_file, sha256_path


def _review_inputs(root: Path) -> tuple[Path, Path]:
    asset = root / "asset.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(asset)
    sample_ids = [f"sample-{index}" for index in range(4)]
    manifest = root / "manifest.csv"
    ranking = root / "ranking.csv"
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "group_id": [f"group-{index}" for index in range(4)],
            "tissue": ["fixture"] * 4,
            "pre_corruption_label": [0, 1, 0, 1],
            "observed_label": [0, 1, 0, 1],
            "is_injected_corruption": [False] * 4,
            "full_patch_path": [str(asset)] * 4,
            "target_crop_path": [str(asset)] * 4,
            "target_contour_path": [str(asset)] * 4,
        }
    ).to_csv(manifest, index=False)
    pd.DataFrame({"sample_id": sample_ids, "risk_score": [0.9, 0.7, 0.4, 0.1]}).to_csv(
        ranking, index=False
    )
    return manifest, ranking


def _mask_qc_patch(fold_id: int, anomaly: str) -> dict[str, Any]:
    void = int(anomaly == "void")
    overlap = int(anomaly == "overlap")
    background = 3 - void
    affected_instances = (
        [{"overlap_pixel_count": 1, "positive_background_pixel_count": 0}] if overlap else []
    )
    return {
        "fold_id": fold_id,
        "patch_index": 0,
        "total_pixel_count": 4,
        "positive_any_pixel_count": 1,
        "background_pixel_count": background,
        "void_pixel_count": void,
        "cross_class_overlap_pixel_count": overlap,
        "positive_and_background_pixel_count": 0,
        "anomaly_union_pixel_count": int(bool(void or overlap)),
        "affected_instance_count": len(affected_instances),
        "affected_instances": affected_instances,
        "has_void": bool(void),
        "has_cross_class_overlap": bool(overlap),
        "has_positive_and_background": False,
        "mask_sha256_by_kind": {
            "positive_any": "a" * 64,
            "supplied_background": "b" * 64,
            "void_unlabelled": "c" * 64,
            "cross_class_overlap": "d" * 64,
            "positive_and_background": "e" * 64,
            "anomaly_union": "f" * 64,
        },
    }


def _fold_mask_qc(fold_id: int, anomaly: str) -> dict[str, Any]:
    patch = _mask_qc_patch(fold_id, anomaly)
    void_indices = [0] if anomaly == "void" else []
    overlap_indices = [0] if anomaly == "overlap" else []
    anomaly_indices = [0] if anomaly != "normal" else []
    return {
        "fold_id": fold_id,
        "patch_count": 1,
        "total_pixel_count": patch["total_pixel_count"],
        "positive_any_pixel_count": patch["positive_any_pixel_count"],
        "background_pixel_count": patch["background_pixel_count"],
        "void_pixel_count": patch["void_pixel_count"],
        "cross_class_overlap_pixel_count": patch["cross_class_overlap_pixel_count"],
        "positive_and_background_pixel_count": 0,
        "anomaly_union_pixel_count": patch["anomaly_union_pixel_count"],
        "void_patch_count": len(void_indices),
        "cross_class_overlap_patch_count": len(overlap_indices),
        "positive_and_background_patch_count": 0,
        "anomaly_union_patch_count": len(anomaly_indices),
        "normal_patch_count": int(anomaly == "normal"),
        "affected_instance_count": patch["affected_instance_count"],
        "overlap_touching_instance_count": patch["affected_instance_count"],
        "positive_background_touching_instance_count": 0,
        "void_patch_indices": void_indices,
        "cross_class_overlap_patch_indices": overlap_indices,
        "positive_and_background_patch_indices": [],
        "anomaly_union_patch_indices": anomaly_indices,
        "patches": [patch],
    }


def _real_evidence_fixture(root: Path) -> tuple[Path, Path, Path]:
    dataset = root / "data" / "raw" / "pannuke"
    dataset.mkdir(parents=True)
    raw_file = dataset / "release.bin"
    raw_file.write_bytes(b"immutable raw fixture")
    fold_qc = [
        _fold_mask_qc(1, "overlap"),
        _fold_mask_qc(2, "void"),
        _fold_mask_qc(3, "normal"),
    ]
    fold_validation = [
        {
            "fold_id": fold_id,
            "n_patches": 1,
            "height": 2,
            "width": 2,
            "positive_channel_indices": [0, 1, 2, 3, 4],
            "background_channel_index": 5,
            "validation_scope": "full_semantic_scan",
            "full_scan_patch_count": 1,
            "overlap_pixel_count_sampled": int(fold_id == 1),
            "malformed_instance_count_sampled": 0,
            "mask_qc": fold_qc[fold_id - 1],
        }
        for fold_id in (1, 2, 3)
    ]
    global_fields = (
        "patch_count",
        "total_pixel_count",
        "positive_any_pixel_count",
        "background_pixel_count",
        "void_pixel_count",
        "cross_class_overlap_pixel_count",
        "positive_and_background_pixel_count",
        "anomaly_union_pixel_count",
        "void_patch_count",
        "cross_class_overlap_patch_count",
        "positive_and_background_patch_count",
        "anomaly_union_patch_count",
        "normal_patch_count",
        "affected_instance_count",
        "overlap_touching_instance_count",
        "positive_background_touching_instance_count",
    )
    validation = {
        "status": "valid",
        "validation_scope": "full_semantic_scan",
        "release_complete": True,
        "expected_fold_ids": [1, 2, 3],
        "grouping_unit": "source_patch",
        "root": str(dataset.resolve()),
        "class_mapping": OFFICIAL_METRICS_CLASS_MAPPING.as_dict(),
        "fold_validation": fold_validation,
        "raw_file_inventory": [
            {
                "relative_path": raw_file.name,
                "sha256": sha256_file(raw_file),
                "size_bytes": raw_file.stat().st_size,
            }
        ],
        "automatic_source_annotation_modification": False,
        "qc_policy": {
            "policy_version": "pannuke-mask-qc-v2",
            "positive_channel_indices": [0, 1, 2, 3, 4],
            "background_channel_index_by_fold": {"1": 5, "2": 5, "3": 5},
            "supplied_background_is_exact_complement_required": False,
            "analysis_instance_exclusion_reason": "touches_cross_class_overlap",
            "applies_identically_to_primary_and_confirmatory": True,
            "no_class_arbitration": True,
            "source_masks_modified": False,
            "release_annotation_anomalies_are_fatal": False,
            "structural_invalidity_is_fatal": True,
        },
        "global_mask_qc": {
            "fold_ids": [1, 2, 3],
            "fold_count": 3,
            **{field: sum(int(qc[field]) for qc in fold_qc) for field in global_fields},
        },
    }
    validation_path = root / "validation.json"
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    raw_inventory_binding = canonical_sha256(
        {
            "schema_version": 1,
            "kind": "pannuke_validated_raw_inventory",
            "files": validation["raw_file_inventory"],
        }
    )
    patch_manifest_binding = "1" * 64
    rgb_input_binding = "2" * 64
    sample_order_binding = "3" * 64

    cache = root / "embeddings.npz"
    cache.write_bytes(b"frozen cache fixture")
    cache_metadata = {
        "encoder_name": "torchvision.resnet18",
        "encoder_frozen": True,
        "weight_identifier": "ResNet18_Weights.IMAGENET1K_V1",
        "weight_sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
        "preprocessing": {"api": "torchvision weight_enum.transforms(antialias=True)"},
        "input_variant": "context_rgb",
        "cache_npz_sha256": sha256_file(cache),
        "sample_count": 3,
        "manifest_sha256": patch_manifest_binding,
        "raw_inventory_sha256": raw_inventory_binding,
        "input_sha256": rgb_input_binding,
        "sample_order_sha256": sample_order_binding,
    }
    cache.with_suffix(".npz.metadata.json").write_text(json.dumps(cache_metadata), encoding="utf-8")
    ranking = root / "rankings.csv"
    ranking.write_text("rank,candidate_id\n", encoding="utf-8")
    duplicate = {
        "schema_version": 2,
        "status": "completed",
        "required_two_signal_near_duplicate_gate_complete": True,
        "policy": {
            "automatic_deletion": False,
            "candidate_action": "review_only",
            "cross_fold_only": True,
            "split_or_exclusion_change_applied": False,
            "final_reference_outcomes_used": False,
            "grouping_unit": "source_patch",
        },
        "coverage": {
            "total_source_patches": 3,
            "patches_with_full_hash_provenance": 3,
            "perceptual_comparison_patch_count": 3,
            "embedding_patch_count": 3,
            "fold_patch_counts": {"1": 1, "2": 1, "3": 1},
            "full_release_cross_fold_pair_counts_by_fold_pair": {
                "1-2": 1,
                "1-3": 1,
                "2-3": 1,
            },
            "perceptual_cross_fold_pair_counts_by_fold_pair": {
                "1-2": 1,
                "1-3": 1,
                "2-3": 1,
            },
            "embedding_cross_fold_pair_counts_by_fold_pair": {
                "1-2": 1,
                "1-3": 1,
                "2-3": 1,
            },
            "full_release_cross_fold_pair_count": 3,
            "perceptual_cross_fold_pair_count": 3,
            "embedding_cross_fold_pair_count": 3,
            "sample_order_sha256": sample_order_binding,
        },
        "provenance_bindings": {
            "patch_manifest_sha256": patch_manifest_binding,
            "raw_inventory_sha256": raw_inventory_binding,
            "canonical_rgb_embedding_input_sha256": rgb_input_binding,
        },
        "embedding_signal": {
            "status": "passed",
            "full_patch_coverage": True,
            "cache_path": str(cache.resolve()),
            "metadata": cache_metadata,
        },
        "artifacts": {
            "rankings_csv": str(ranking.resolve()),
            "rankings_csv_sha256": sha256_file(ranking),
        },
    }
    duplicate_path = root / "duplicates.json"
    duplicate_path.write_text(json.dumps(duplicate), encoding="utf-8")
    return dataset, validation_path, duplicate_path


def test_real_dataset_evidence_accepts_reconciled_overlap_void_qc_and_exact_sidecar(
    tmp_path: Path,
) -> None:
    dataset, validation, duplicate = _real_evidence_fixture(tmp_path)

    errors = validate_real_dataset_evidence(dataset, validation, duplicate)

    assert errors == ()


def test_real_dataset_evidence_rejects_qc_and_embedding_sidecar_tampering(
    tmp_path: Path,
) -> None:
    dataset, validation, duplicate = _real_evidence_fixture(tmp_path)
    duplicate_payload = json.loads(duplicate.read_text(encoding="utf-8"))
    duplicate_payload["embedding_signal"]["metadata"]["input_variant"] = "rgb"
    duplicate.write_text(json.dumps(duplicate_payload), encoding="utf-8")

    metadata_errors = validate_real_dataset_evidence(dataset, validation, duplicate)

    assert "duplicate audit embedding metadata differs from its cache sidecar" in metadata_errors

    duplicate_payload["embedding_signal"]["metadata"]["input_variant"] = "context_rgb"
    duplicate.write_text(json.dumps(duplicate_payload), encoding="utf-8")
    validation_payload = json.loads(validation.read_text(encoding="utf-8"))
    validation_payload["global_mask_qc"]["cross_class_overlap_pixel_count"] = 0
    validation.write_text(json.dumps(validation_payload), encoding="utf-8")

    qc_errors = validate_real_dataset_evidence(dataset, validation, duplicate)

    assert (
        "PanNuke release-wide mask-QC aggregate cross_class_overlap_pixel_count is inconsistent"
        in qc_errors
    )


def test_real_dataset_evidence_rejects_duplicate_policy_coverage_and_binding_tampering(
    tmp_path: Path,
) -> None:
    dataset, validation, duplicate = _real_evidence_fixture(tmp_path)
    canonical = json.loads(duplicate.read_text(encoding="utf-8"))

    policy_tamper = json.loads(json.dumps(canonical))
    policy_tamper["policy"]["automatic_deletion"] = True
    duplicate.write_text(json.dumps(policy_tamper), encoding="utf-8")
    policy_errors = validate_real_dataset_evidence(dataset, validation, duplicate)
    assert "duplicate audit policy automatic_deletion differs from the safe contract" in (
        policy_errors
    )

    coverage_tamper = json.loads(json.dumps(canonical))
    coverage_tamper["coverage"]["embedding_cross_fold_pair_count"] = 2
    duplicate.write_text(json.dumps(coverage_tamper), encoding="utf-8")
    coverage_errors = validate_real_dataset_evidence(dataset, validation, duplicate)
    assert "duplicate audit embedding total pair coverage is inconsistent" in coverage_errors

    binding_tamper = json.loads(json.dumps(canonical))
    binding_tamper["provenance_bindings"]["raw_inventory_sha256"] = "f" * 64
    duplicate.write_text(json.dumps(binding_tamper), encoding="utf-8")
    binding_errors = validate_real_dataset_evidence(dataset, validation, duplicate)
    assert "duplicate audit raw-inventory binding differs from validation" in binding_errors


def _invoke_package(
    root: Path,
    manifest: Path,
    ranking: Path,
    *,
    extra: list[str] | None = None,
) -> tuple[Any, Path, Path]:
    package = root / "review-package"
    private_key = root / "private" / "unblinding.csv"
    arguments = [
        "external",
        "build-review-package",
        "--project-root",
        str(root),
        "--manifest",
        str(manifest.relative_to(root)),
        "--ranking",
        str(ranking.relative_to(root)),
        "--output-dir",
        str(package.relative_to(root)),
        "--private-key",
        str(private_key.relative_to(root)),
        "--top",
        "1",
        "--random",
        "1",
        "--seed",
        "19",
        *(extra or []),
    ]
    return CliRunner().invoke(app, arguments), package, private_key


def test_structural_fixture_build_is_useful_but_never_emits_readiness(tmp_path: Path) -> None:
    manifest, ranking = _review_inputs(tmp_path)

    result, package, private_key = _invoke_package(tmp_path, manifest, ranking)

    assert result.exit_code == 0, result.output
    assert "EXTERNAL_VALIDATION_READY" not in result.output
    payload = json.loads(result.output)
    assert payload["workflow"] == "blinded_review_package_fixture_or_non_stage_build"
    assert payload["study_outcome_eligible"] is False
    metadata = json.loads((package / "package_metadata.json").read_text(encoding="utf-8"))
    assert metadata["study_outcome_eligible"] is False
    assert metadata["eligibility_evidence"] is None
    assert private_key.is_file()


def test_external_help_separates_stage_evidence_from_fixture_inputs() -> None:
    options = cli_options(app, ("external", "build-review-package"))
    assert {
        "--audit-run-dir",
        "--dataset",
        "--dataset-validation-json",
        "--duplicate-audit-json",
    }.issubset(options)


def test_partial_eligibility_request_fails_before_creating_outputs(tmp_path: Path) -> None:
    manifest, ranking = _review_inputs(tmp_path)

    result, package, private_key = _invoke_package(
        tmp_path,
        manifest,
        ranking,
        extra=["--audit-run-dir", "missing-audit"],
    )

    assert result.exit_code == 1
    assert "must be supplied together" in result.output
    assert "EXTERNAL_VALIDATION_READY" not in result.output
    assert not package.exists()
    assert not private_key.exists()


def test_staged_validation_failure_leaves_neither_package_nor_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, ranking = _review_inputs(tmp_path)
    package = tmp_path / "review-package"
    private_key = tmp_path / "private" / "unblinding.csv"

    def _invalid(
        package_directory: str | Path,
        *,
        private_unblinding_key_path: str | Path | None = None,
    ) -> ReviewPackageValidationResult:
        del private_unblinding_key_path
        return ReviewPackageValidationResult(
            valid=False,
            package_directory=Path(package_directory),
            item_count=2,
            asset_count=6,
            private_linkage_validated=False,
            errors=("forced staged failure",),
            warnings=(),
        )

    monkeypatch.setattr(validation_module, "validate_blinded_review_package", _invalid)
    with pytest.raises(RuntimeError, match="staged review package failed validation"):
        build_blinded_review_package(
            manifest,
            ranking,
            package,
            private_unblinding_key_path=private_key,
            top_count=1,
            random_count=1,
            seed=19,
        )

    assert not package.exists()
    assert not private_key.exists()


def test_cli_post_build_validation_failure_rolls_back_both_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, ranking = _review_inputs(tmp_path)

    def _invalid(
        package_directory: str | Path,
        *,
        private_unblinding_key_path: str | Path | None = None,
    ) -> ReviewPackageValidationResult:
        assert private_unblinding_key_path is not None
        return ReviewPackageValidationResult(
            valid=False,
            package_directory=Path(package_directory),
            item_count=2,
            asset_count=6,
            private_linkage_validated=False,
            errors=("forced independent failure",),
            warnings=(),
        )

    monkeypatch.setattr(external_validation, "validate_blinded_review_package", _invalid)
    result, package, private_key = _invoke_package(tmp_path, manifest, ranking)

    assert result.exit_code == 1
    assert "generated review package failed validation" in result.output
    assert "EXTERNAL_VALIDATION_READY" not in result.output
    assert not package.exists()
    assert not private_key.exists()


def test_sealed_structural_fixture_explicitly_ineligible_cannot_reach_stage(
    tmp_path: Path,
) -> None:
    manifest, _ = _review_inputs(tmp_path)
    dataset = tmp_path / "synthetic-fixture-dataset"
    dataset.mkdir()
    (dataset / "fixture.npy").write_bytes(b"not a real PanNuke release")
    validation_json = tmp_path / "validation.json"
    duplicate_json = tmp_path / "duplicates.json"
    final_groups = tmp_path / "final-groups.txt"
    feature_cache = tmp_path / "features.npz"
    validation_json.write_text("{}", encoding="utf-8")
    duplicate_json.write_text("{}", encoding="utf-8")
    final_groups.write_text("held-out-group\n", encoding="utf-8")
    np.savez(feature_cache, features=np.zeros((4, 2)), sample_ids=np.asarray(["a", "b", "c", "d"]))

    tracker = RunTracker.start(
        experiment_name="original_label_audit",
        config={"schema_version": 1, "experiment_name": "original_label_audit"},
        project_root=tmp_path,
        dataset_path=dataset,
        manifest_path=manifest,
        duplicate_audit_status=f"complete_sha256:{sha256_file(duplicate_json)}",
    )
    ranking = tracker.write_text(
        "ranking_all.csv",
        "sample_id,risk_score\nsample-0,0.9\nsample-1,0.7\nsample-2,0.4\nsample-3,0.1\n",
    )
    tracker.write_json(
        "audit_metadata.json",
        {
            "source_manifest": {
                "path": str(manifest),
                "sha256_before": sha256_file(manifest),
                "sha256_after": sha256_file(manifest),
                "input_mode": "read_only_file",
            },
            "observed_equals_pre_corruption": True,
            "injected_corruption_count": 0,
            "group_safe_oof": {
                "coverage_exactly_once": True,
                "final_reference_groups_absent_from_audit": True,
            },
        },
    )
    tracker.write_json(
        "external_validation_eligibility.json",
        {
            "schema_version": 1,
            "workflow": "exploratory_original_label_audit",
            "study_outcome_eligible": False,
            "data_source": "synthetic_fixture",
            "dataset": {"path": str(dataset), "sha256": sha256_path(dataset)},
            "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
            "dataset_validation": {
                "path": str(validation_json),
                "sha256": sha256_file(validation_json),
            },
            "duplicate_audit": {
                "path": str(duplicate_json),
                "sha256": sha256_file(duplicate_json),
            },
            "ranking": {"path": str(ranking), "sha256": sha256_file(ranking)},
            "feature_cache": {
                "path": str(feature_cache),
                "sha256": sha256_file(feature_cache),
            },
            "final_reference_groups": {
                "path": str(final_groups),
                "sha256": sha256_file(final_groups),
            },
        },
    )
    tracker.complete()

    evidence = verify_external_validation_eligibility(
        audit_run_directory=tracker.run_directory,
        dataset_path=dataset,
        manifest_path=manifest,
        dataset_validation_json=validation_json,
        duplicate_audit_json=duplicate_json,
        ranking_path=ranking,
    )

    assert not evidence.eligible
    assert any("explicitly not eligible" in error for error in evidence.errors)
    result, package, private_key = _invoke_package(
        tmp_path,
        manifest,
        ranking,
        extra=[
            "--audit-run-dir",
            str(tracker.run_directory.relative_to(tmp_path)),
            "--dataset",
            str(dataset.relative_to(tmp_path)),
            "--dataset-validation-json",
            validation_json.name,
            "--duplicate-audit-json",
            duplicate_json.name,
        ],
    )
    assert result.exit_code == 1
    assert "EXTERNAL_VALIDATION_READY" not in result.output
    assert not package.exists()
    assert not private_key.exists()
