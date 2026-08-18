from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from histo_audit.experiment.pannuke_confirmatory_inputs import (
    ConfirmatoryFrozenFeatureCacheSpec,
    ConfirmatoryObservedLabelSet,
    load_pannuke_confirmatory_inputs,
)
from histo_audit.experiment.study_contracts import build_confirmatory_matrix_plan
from histo_audit.representations.cache_provenance import (
    array_artifact_sha256,
    ordered_sample_ids_sha256,
)
from histo_audit.representations.eligibility import select_manifest_rows
from histo_audit.representations.imagenet import save_embedding_cache
from histo_audit.utils.run_tracking import sha256_file


def _confirmatory_config() -> dict[str, Any]:
    fixture_sample_ids = [
        f"f{fold}-g{group}-c{label}"
        for fold in (1, 2, 3)
        for group in range(4)
        for label in range(5)
    ]
    sample_order_sha256 = ordered_sample_ids_sha256(fixture_sample_ids)
    return {
        "schema_version": 2,
        "experiment_name": "confirmatory_study",
        "status": "ready_for_freeze",
        "data": {
            "source": "pannuke",
            "analysis_manifest_authority": {
                "canonical_manifest_sha256": "a" * 64,
                "analysis_eligible_sample_order_sha256": sample_order_sha256,
                "analysis_eligible_sample_count": len(fixture_sample_ids),
            },
            "official_folds": [1, 2, 3],
            "group_unit": "source_patch_id",
            "reference_validation_fraction_groups": 0.10,
            "reference_group_selection_algorithm": (
                "deterministic_group_greedy_class_distribution_v1"
            ),
            "split_seed": 223,
            "fold_assignment_labels": "pre_corruption_label",
        },
        "corruption": {
            "cells": [
                {
                    "id": "clean_reference_cell",
                    "mechanism": "symmetric_random_corruption",
                    "rate": 0.0,
                    "seed": 404,
                    "parameters": {},
                },
                {
                    "id": "symmetric_ten_percent",
                    "mechanism": "symmetric_random_corruption",
                    "rate": 0.10,
                    "seed": 404,
                    "parameters": {},
                },
            ]
        },
        "scenarios": [
            {
                "id": "cnn_context_rgb",
                "representation_id": "cnn_context_rgb_pixels",
                "cache_provenance_id": "cnn_context_rgb_cache",
                "family": "cnn",
                "input_variant": "context_rgb",
                "encoder": "resnet18_imagenet1k_v1",
                "classifier": "cnn_softmax_head",
                "required": True,
            },
            {
                "id": "cnn_context_target_mask",
                "representation_id": "cnn_context_rgb_mask_pixels",
                "cache_provenance_id": "cnn_context_target_mask_cache",
                "family": "cnn",
                "input_variant": "context_rgb_plus_binary_target_mask",
                "encoder": "resnet18_imagenet1k_v1_zero_init_fourth_channel",
                "classifier": "cnn_softmax_head",
                "required": True,
            },
            {
                "id": "imagenet_frozen_logistic",
                "representation_id": "imagenet_resnet18_context_embeddings",
                "cache_provenance_id": "imagenet_context_embedding_cache",
                "family": "imagenet_frozen",
                "input_variant": "context_rgb",
                "encoder": "resnet18_imagenet1k_v1",
                "classifier": "multinomial_logistic_regression",
                "required": True,
            },
            {
                "id": "imagenet_frozen_target_highlighted_logistic",
                "representation_id": "imagenet_target_highlighted_embeddings",
                "cache_provenance_id": "imagenet_target_highlighted_embedding_cache",
                "family": "imagenet_frozen",
                "input_variant": "target_highlighted_rgb",
                "encoder": "resnet18_imagenet1k_v1",
                "classifier": "multinomial_logistic_regression",
                "required": True,
            },
            {
                "id": "imagenet_frozen_context_morphometrics_logistic",
                "representation_id": "imagenet_context_embeddings_plus_target_morphometrics",
                "cache_provenance_id": "imagenet_context_morphometrics_cache",
                "family": "imagenet_frozen",
                "input_variant": "context_rgb_plus_target_morphometrics",
                "encoder": "resnet18_imagenet1k_v1_plus_target_morphometrics_v1",
                "classifier": "multinomial_logistic_regression",
                "required": True,
            },
            {
                "id": "pathology_frozen_logistic",
                "representation_id": "pathology_context_embeddings",
                "cache_provenance_id": "pathology_context_embedding_cache",
                "family": "pathology_frozen",
                "input_variant": "context_rgb",
                "encoder": "availability_selected_pathology_encoder",
                "classifier": "multinomial_logistic_regression",
                "required": False,
                "availability_audit_sha256": "b" * 64,
            },
        ],
        "cache_provenance": [
            {
                "id": "cnn_context_rgb_cache",
                "representation_id": "cnn_context_rgb_pixels",
                "status": "available",
                "cache_file_sha256": None,
                "sidecar_semantic_sha256": "1" * 64,
                "sample_order_sha256": sample_order_sha256,
                "manifest_sha256": "a" * 64,
                "encoder_identifier": "resnet18_imagenet1k_v1",
                "encoder_metadata_sha256": "4" * 64,
                "weight_identifier": "ResNet18_Weights.IMAGENET1K_V1",
                "weights_sha256": "5" * 64,
                "preprocessing_identifier": "torchvision_resnet18_imagenet1k_v1_official",
                "preprocessing_sha256": "6" * 64,
                "input_variant": "context_rgb",
            },
            {
                "id": "cnn_context_target_mask_cache",
                "representation_id": "cnn_context_rgb_mask_pixels",
                "status": "available",
                "cache_file_sha256": None,
                "sidecar_semantic_sha256": "7" * 64,
                "sample_order_sha256": sample_order_sha256,
                "manifest_sha256": "a" * 64,
                "encoder_identifier": "resnet18_imagenet1k_v1_zero_init_fourth_channel",
                "encoder_metadata_sha256": "8" * 64,
                "weight_identifier": "ResNet18_Weights.IMAGENET1K_V1",
                "weights_sha256": "5" * 64,
                "preprocessing_identifier": "torchvision_resnet18_imagenet1k_v1_official",
                "preprocessing_sha256": "9" * 64,
                "input_variant": "context_rgb_plus_binary_target_mask",
            },
            {
                "id": "imagenet_context_embedding_cache",
                "representation_id": "imagenet_resnet18_context_embeddings",
                "status": "available",
                "cache_file_sha256": None,
                "sidecar_semantic_sha256": "a" * 64,
                "sample_order_sha256": sample_order_sha256,
                "manifest_sha256": "a" * 64,
                "encoder_identifier": "resnet18_imagenet1k_v1",
                "encoder_metadata_sha256": "c" * 64,
                "weight_identifier": "ResNet18_Weights.IMAGENET1K_V1",
                "weights_sha256": "5" * 64,
                "preprocessing_identifier": "torchvision_resnet18_imagenet1k_v1_official",
                "preprocessing_sha256": "d" * 64,
                "input_variant": "context_rgb",
            },
            {
                "id": "imagenet_target_highlighted_embedding_cache",
                "representation_id": "imagenet_target_highlighted_embeddings",
                "status": "available",
                "cache_file_sha256": None,
                "sidecar_semantic_sha256": "e" * 64,
                "sample_order_sha256": sample_order_sha256,
                "manifest_sha256": "a" * 64,
                "encoder_identifier": "resnet18_imagenet1k_v1",
                "encoder_metadata_sha256": "f" * 64,
                "weight_identifier": "ResNet18_Weights.IMAGENET1K_V1",
                "weights_sha256": "5" * 64,
                "preprocessing_identifier": "target_highlighted_resnet18_imagenet1k_v1",
                "preprocessing_sha256": "1" * 64,
                "input_variant": "target_highlighted_rgb",
            },
            {
                "id": "imagenet_context_morphometrics_cache",
                "representation_id": "imagenet_context_embeddings_plus_target_morphometrics",
                "status": "available",
                "cache_file_sha256": None,
                "sidecar_semantic_sha256": "6" * 64,
                "sample_order_sha256": sample_order_sha256,
                "manifest_sha256": "a" * 64,
                "encoder_identifier": "resnet18_imagenet1k_v1_plus_target_morphometrics_v1",
                "encoder_metadata_sha256": "7" * 64,
                "weight_identifier": "ResNet18_Weights.IMAGENET1K_V1",
                "weights_sha256": "5" * 64,
                "preprocessing_identifier": "context_embeddings_target_morphometrics_concat_v1",
                "preprocessing_sha256": "8" * 64,
                "input_variant": "context_rgb_plus_target_morphometrics",
            },
            {
                "id": "pathology_context_embedding_cache",
                "representation_id": "pathology_context_embeddings",
                "status": "unavailable_with_frozen_blocker",
                "sample_order_sha256": sample_order_sha256,
                "manifest_sha256": "a" * 64,
                "encoder_identifier": "availability_selected_pathology_encoder",
                "input_variant": "context_rgb",
                "blocker_evidence_sha256": "b" * 64,
            },
        ],
        "model_seeds": [303, 304, 305],
        "training": {
            "optimizer": "adamw",
            "input_size": 64,
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "max_epochs": 100,
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 0.0001,
            "early_stopping_source": "reference_validation_only",
            "initial_batch_size": 128,
            "minimum_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "class_weight": "balanced",
            "oom_policy": "halve_batch_and_retry_same_samples",
            "amp": True,
            "amp_dtype": "float16",
            "checkpoint_resume": True,
            "cuda_required": True,
        },
        "oof": {
            "n_splits": 5,
            "split_kind": "stratified_group",
            "no_nucleus_level_fallback": True,
        },
        "original_audit_selection": {
            "scenario_id": "imagenet_frozen_logistic",
            "representation_id": "imagenet_resnet18_context_embeddings",
            "model_seed": 303,
            "risk_method": "self_confidence",
            "n_splits": 5,
            "cache_provenance_id": "imagenet_context_embedding_cache",
            "classifier": {
                "id": "multinomial_logistic_regression",
                "parameters": {
                    "l2": 1.0,
                    "max_iter": 1000,
                    "class_weight": "balanced",
                },
            },
        },
        "ensemble": {
            "members": [
                {"scenario_id": "cnn_context_rgb", "model_seed": 303},
                {"scenario_id": "cnn_context_target_mask", "model_seed": 303},
            ],
            "primary_risk": "mean_pairwise_js_divergence",
            "secondary_risks": ["predictive_entropy_of_mean", "variation_ratio"],
        },
        "fixed_hybrid": {
            "components": ["self_confidence", "ensemble_disagreement"],
            "weights": [0.5, 0.5],
            "drop_one_ablations": ["self_confidence", "ensemble_disagreement"],
        },
        "restoration": {
            "scenario_id": "cnn_context_target_mask",
            "model_seed": 303,
            "representation_id": "cnn_context_rgb_mask_pixels",
            "ranking_method": "fixed_hybrid",
            "review_budget": 0.05,
            "random_repeats": 100,
            "random_seed": 443,
            "conditions": [
                "uncorrupted_reference_baseline",
                "corrupted_observed_baseline",
                "random_review_restoration",
                "audit_guided_restoration",
            ],
        },
        "statistics": {
            "paired_group_bootstrap_iterations": 2000,
            "bootstrap_seed": 439,
            "holm_families": ["ranking_primary"],
            "preregistered_paired_comparisons": [
                {
                    "comparison_id": "hybrid_vs_self_confidence",
                    "metric": "average_precision",
                    "operand_a": {
                        "scenario_id": "imagenet_frozen_logistic",
                        "representation_id": "imagenet_resnet18_context_embeddings",
                        "classifier_id": "multinomial_logistic_regression",
                        "risk_id": "fixed_hybrid",
                        "model_seed": "matched",
                        "outer_fold": "all_matched",
                        "corruption_cell": "all_matched",
                    },
                    "operand_b": {
                        "scenario_id": "imagenet_frozen_logistic",
                        "representation_id": "imagenet_resnet18_context_embeddings",
                        "classifier_id": "multinomial_logistic_regression",
                        "risk_id": "self_confidence",
                        "model_seed": "matched",
                        "outer_fold": "all_matched",
                        "corruption_cell": "all_matched",
                    },
                    "direction": "method_a_minus_method_b",
                    "holm_family": "ranking_primary",
                }
            ],
        },
        "fold_rotation": {
            "enabled": True,
            "feasibility_rule": "all_five_classes_in_each_development_training_partition",
            "aggregate_policy": "report_each_rotation_and_descriptive_fold_mean",
        },
    }


def _arrays() -> dict[str, np.ndarray[Any, Any]]:
    sample_ids: list[str] = []
    groups: list[str] = []
    folds: list[int] = []
    labels: list[int] = []
    for fold in (1, 2, 3):
        for group in range(4):
            for label in range(5):
                sample_ids.append(f"f{fold}-g{group}-c{label}")
                groups.append(f"fold-{fold}-group-{group}")
                folds.append(fold)
                labels.append(label)
    n = len(sample_ids)
    rgb = np.arange(n * 6 * 6 * 3, dtype=np.uint8).reshape(n, 6, 6, 3)
    masks = np.zeros((n, 6, 6), dtype=bool)
    masks[:, 2:4, 2:4] = True
    return {
        "sample_ids": np.asarray(sample_ids, dtype=np.str_),
        "context_rgb": rgb,
        "target_masks": masks,
        "official_folds": np.asarray(folds, dtype=np.int16),
        "pre_corruption_labels": np.asarray(labels, dtype=np.int64),
        "group_ids": np.asarray(groups, dtype=np.str_),
        "identity_verified": np.ones(n, dtype=bool),
        "primary_eligible": np.ones(n, dtype=bool),
        "confirmatory_eligible": np.ones(n, dtype=bool),
    }


def _analysis_eligibility(sample_ids: np.ndarray[Any, Any]) -> dict[str, Any]:
    identifiers = np.asarray(sample_ids, dtype=np.str_)
    n = len(identifiers)
    table = pa.table(
        {
            "sample_id": identifiers,
            "primary_eligible": np.ones(n, dtype=bool),
            "confirmatory_eligible": np.ones(n, dtype=bool),
            "cross_class_overlap_touching": np.zeros(n, dtype=bool),
            "qc_exclusion_reason": pa.array([None] * n, type=pa.string()),
        }
    )
    return select_manifest_rows(table, sample_ids=None, scope="analysis").provenance


def _write_crop(
    root: Path,
    arrays: dict[str, np.ndarray[Any, Any]],
) -> tuple[Path, dict[str, str]]:
    path = root / "pannuke_crops.npz"
    np.savez_compressed(path, **arrays)
    metadata_path = path.with_suffix(".npz.metadata.json")
    metadata = {
        "schema_version": 1,
        "dataset": "PanNuke",
        "sample_count": len(arrays["sample_ids"]),
        "manifest_sha256": "a" * 64,
        "raw_inventory_sha256": "b" * 64,
        "cache_npz_sha256": sha256_file(path),
        "cache_array_sha256_by_name": {
            name: array_artifact_sha256(values) for name, values in arrays.items()
        },
        "source_annotations_modified": False,
        "analysis_eligibility": _analysis_eligibility(arrays["sample_ids"]),
    }
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    return path, {
        "expected_crop_cache_sha256": sha256_file(path),
        "expected_crop_metadata_sha256": sha256_file(metadata_path),
        "expected_manifest_sha256": "a" * 64,
        "expected_raw_inventory_sha256": "b" * 64,
    }


def _write_feature(
    root: Path,
    sample_ids: np.ndarray[Any, Any],
    *,
    scenario_id: str = "imagenet_frozen_logistic",
    filename: str = "imagenet.npz",
    input_variant: str = "rgb",
    feature_count: int = 512,
    values: np.ndarray[Any, Any] | None = None,
) -> ConfirmatoryFrozenFeatureCacheSpec:
    path = root / filename
    matrix = (
        np.arange(len(sample_ids) * feature_count, dtype=np.float32).reshape(
            len(sample_ids), feature_count
        )
        if values is None
        else np.asarray(values, dtype=np.float32)
    )
    identifiers = np.asarray(sample_ids, dtype=np.str_)
    metadata_path = path.with_suffix(f"{path.suffix}.metadata.json")
    if feature_count == 512 and np.isfinite(matrix).all():
        _, metadata_path, _ = save_embedding_cache(
            path,
            matrix,
            identifiers,
            {
                "schema_version": 1,
                "weight_sha256": "c" * 64,
                "input_variant": input_variant,
                "analysis_eligibility": _analysis_eligibility(identifiers),
            },
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        np.savez_compressed(path, values=matrix, sample_ids=identifiers)
        metadata = {
            "schema_version": 1,
            "cache_npz_sha256": sha256_file(path),
            "weight_sha256": "c" * 64,
            "input_variant": input_variant,
            "analysis_eligibility": _analysis_eligibility(identifiers),
        }
    metadata["crop_manifest_sha256"] = "a" * 64
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    return ConfirmatoryFrozenFeatureCacheSpec(
        scenario_id=scenario_id,
        cache_path=path,
        expected_cache_sha256=sha256_file(path),
        expected_metadata_sha256=sha256_file(metadata_path),
        expected_weight_sha256="c" * 64,
    )


def _bundle(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _confirmatory_config()
    arrays = _arrays()
    crop, bindings = _write_crop(tmp_path, arrays)
    features = (
        _write_feature(tmp_path, arrays["sample_ids"]),
        _write_feature(
            tmp_path,
            arrays["sample_ids"],
            scenario_id="imagenet_frozen_target_highlighted_logistic",
            filename="imagenet_target_highlighted.npz",
            input_variant="target_highlighted_rgb",
        ),
        _write_feature(
            tmp_path,
            arrays["sample_ids"],
            scenario_id="imagenet_frozen_context_morphometrics_logistic",
            filename="imagenet_context_morphometrics.npz",
            input_variant="context_rgb_plus_target_morphometrics",
            feature_count=520,
        ),
    )
    arguments: dict[str, Any] = {
        "crop_cache_path": crop,
        "confirmatory_config": config,
        "expected_config_sha256": build_confirmatory_matrix_plan(config).config_sha256,
        **bindings,
        "frozen_feature_caches": features,
        "synthetic_fixture": True,
    }
    return arguments, arrays, config


def test_loads_all_rotations_without_role_leakage_and_keeps_pathology_unavailable(
    tmp_path: Path,
) -> None:
    arguments, arrays, _ = _bundle(tmp_path)

    result = load_pannuke_confirmatory_inputs(**arguments)

    assert tuple(rotation.outer_fold for rotation in result.rotations) == (1, 2, 3)
    assert result.execution_mode == "synthetic_fixture"
    assert result.study_outcome_eligible is False
    assert result.ineligibility_reasons
    availability = {row.scenario_id: row for row in result.frozen_feature_availability}
    assert availability["imagenet_frozen_logistic"].available is True
    assert availability["imagenet_frozen_target_highlighted_logistic"].available is True
    assert availability["imagenet_frozen_context_morphometrics_logistic"].available is True
    assert availability["pathology_frozen_logistic"].available is False
    assert availability["pathology_frozen_logistic"].blocker
    for rotation in result.rotations:
        audit = rotation.audit
        reference = rotation.reference_validation
        final = rotation.final_reference
        assert set(audit.sample_ids).isdisjoint(reference.sample_ids)
        assert set(audit.sample_ids).isdisjoint(final.sample_ids)
        assert set(reference.sample_ids).isdisjoint(final.sample_ids)
        assert set(audit.group_ids).isdisjoint(reference.group_ids)
        assert set(audit.group_ids).isdisjoint(final.group_ids)
        assert set(reference.group_ids).isdisjoint(final.group_ids)
        assert np.all(arrays["official_folds"][final.source_indices] == rotation.outer_fold)
        assert not reference.is_injected_corruption.any()
        assert not final.is_injected_corruption.any()
        assert audit.context_rgb.flags.writeable is False
        assert audit.target_masks.flags.writeable is False
        assert len(audit.frozen_features) == 3
        assert all(feature.values.flags.writeable is False for feature in audit.frozen_features)
        assert set(audit.pre_corruption_labels) == {0, 1, 2, 3, 4}
        assert set(reference.pre_corruption_labels) == {0, 1, 2, 3, 4}
        assert set(final.pre_corruption_labels) == {0, 1, 2, 3, 4}


@pytest.mark.parametrize(
    ("flag", "mode"),
    [("cpu_test_only", "cpu_test_only"), ("synthetic_fixture", "synthetic_fixture")],
)
def test_cpu_and_synthetic_modes_are_permanently_ineligible(
    tmp_path: Path, flag: str, mode: str
) -> None:
    arguments, _, _ = _bundle(tmp_path)
    arguments["synthetic_fixture"] = False
    arguments[flag] = True

    result = load_pannuke_confirmatory_inputs(**arguments)

    assert result.execution_mode == mode
    assert result.study_outcome_eligible is False
    assert result.ineligibility_reasons


def test_rejects_feature_sample_order_misalignment(tmp_path: Path) -> None:
    arguments, arrays, _ = _bundle(tmp_path)
    bad_feature = _write_feature(
        tmp_path,
        arrays["sample_ids"][::-1],
        filename="misaligned_imagenet.npz",
    )
    arguments["frozen_feature_caches"] = (bad_feature,)

    with pytest.raises(ValueError, match="sample IDs/order"):
        load_pannuke_confirmatory_inputs(**arguments)


def test_rejects_feature_cache_without_shared_eligibility_binding(tmp_path: Path) -> None:
    arguments, _, _ = _bundle(tmp_path)
    specs = tuple(arguments["frozen_feature_caches"])
    original = specs[0]
    sidecar = original.cache_path.with_suffix(f"{original.cache_path.suffix}.metadata.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata.pop("analysis_eligibility")
    sidecar.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    altered = ConfirmatoryFrozenFeatureCacheSpec(
        scenario_id=original.scenario_id,
        cache_path=original.cache_path,
        expected_cache_sha256=original.expected_cache_sha256,
        expected_metadata_sha256=sha256_file(sidecar),
        expected_weight_sha256=original.expected_weight_sha256,
    )
    arguments["frozen_feature_caches"] = (altered, *specs[1:])

    with pytest.raises(ValueError, match="lacks analysis-eligibility provenance"):
        load_pannuke_confirmatory_inputs(**arguments)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("expected_config_sha256", "configuration differs"),
        ("expected_crop_cache_sha256", "crop cache SHA-256 differs"),
        ("expected_crop_metadata_sha256", "crop metadata SHA-256 differs"),
        ("expected_manifest_sha256", "different manifest"),
        ("expected_raw_inventory_sha256", "different raw inventory"),
    ],
)
def test_rejects_wrong_frozen_hash_bindings(tmp_path: Path, field: str, message: str) -> None:
    arguments, _, _ = _bundle(tmp_path)
    arguments[field] = "0" * 64

    with pytest.raises(ValueError, match=message):
        load_pannuke_confirmatory_inputs(**arguments)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("analysis_eligible_sample_count", 61, "sample count differs"),
        ("analysis_eligible_sample_order_sha256", "0" * 64, "sample order differs"),
    ],
)
def test_rejects_crop_cache_outside_confirmatory_analysis_authority(
    tmp_path: Path,
    field: str,
    value: str | int,
    message: str,
) -> None:
    arguments, _, config = _bundle(tmp_path)
    config["data"]["analysis_manifest_authority"][field] = value
    if field == "analysis_eligible_sample_order_sha256":
        for record in config["cache_provenance"]:
            record["sample_order_sha256"] = value
    arguments["expected_config_sha256"] = build_confirmatory_matrix_plan(config).config_sha256

    with pytest.raises(ValueError, match=message):
        load_pannuke_confirmatory_inputs(**arguments)


def test_rejects_missing_target_masks(tmp_path: Path) -> None:
    config = _confirmatory_config()
    arrays = _arrays()
    arrays.pop("target_masks")
    crop, bindings = _write_crop(tmp_path, arrays)
    feature = _write_feature(tmp_path, arrays["sample_ids"])

    with pytest.raises(ValueError, match=r"missing required arrays.*target_masks"):
        load_pannuke_confirmatory_inputs(
            crop,
            confirmatory_config=config,
            expected_config_sha256=build_confirmatory_matrix_plan(config).config_sha256,
            **bindings,
            frozen_feature_caches=(feature,),
            synthetic_fixture=True,
        )


def test_rejects_unknown_official_fold(tmp_path: Path) -> None:
    config = _confirmatory_config()
    arrays = _arrays()
    arrays["official_folds"][-5:] = 4
    crop, bindings = _write_crop(tmp_path, arrays)
    feature = _write_feature(tmp_path, arrays["sample_ids"])

    with pytest.raises(ValueError, match="official folds differ"):
        load_pannuke_confirmatory_inputs(
            crop,
            confirmatory_config=config,
            expected_config_sha256=build_confirmatory_matrix_plan(config).config_sha256,
            **bindings,
            frozen_feature_caches=(feature,),
            synthetic_fixture=True,
        )


def test_rejects_group_crossing_official_fold_boundary(tmp_path: Path) -> None:
    config = _confirmatory_config()
    arrays = _arrays()
    arrays["group_ids"][-1] = arrays["group_ids"][0]
    crop, bindings = _write_crop(tmp_path, arrays)
    feature = _write_feature(tmp_path, arrays["sample_ids"])

    with pytest.raises(ValueError, match="spans official folds"):
        load_pannuke_confirmatory_inputs(
            crop,
            confirmatory_config=config,
            expected_config_sha256=build_confirmatory_matrix_plan(config).config_sha256,
            **bindings,
            frozen_feature_caches=(feature,),
            synthetic_fixture=True,
        )


def test_rejects_corruption_in_final_reference_partition(tmp_path: Path) -> None:
    arguments, arrays, config = _bundle(tmp_path)
    config_sha = build_confirmatory_matrix_plan(config).config_sha256
    label_sets: list[ConfirmatoryObservedLabelSet] = []
    for outer_fold in (1, 2, 3):
        observed = arrays["pre_corruption_labels"].copy()
        injected = np.zeros(len(observed), dtype=bool)
        corruption_types = ["none"] * len(observed)
        if outer_fold == 1:
            final_index = int(np.flatnonzero(arrays["official_folds"] == 1)[0])
            observed[final_index] = (int(observed[final_index]) + 1) % 5
            injected[final_index] = True
            corruption_types[final_index] = "symmetric_random_corruption"
        label_sets.append(
            ConfirmatoryObservedLabelSet(
                outer_fold=outer_fold,
                sample_ids=tuple(str(value) for value in arrays["sample_ids"]),
                pre_corruption_labels=arrays["pre_corruption_labels"].copy(),
                observed_labels=observed,
                is_injected_corruption=injected,
                corruption_types=tuple(corruption_types),
                configuration_sha256=config_sha,
            )
        )
    arguments["observed_label_sets"] = tuple(label_sets)

    with pytest.raises(ValueError, match="final_reference must be uncorrupted"):
        load_pannuke_confirmatory_inputs(**arguments)


def test_rejects_nonfinite_required_feature_cache(tmp_path: Path) -> None:
    arguments, arrays, _ = _bundle(tmp_path)
    values = np.zeros((len(arrays["sample_ids"]), 512), dtype=np.float32)
    values[0, 0] = np.nan
    bad_feature = _write_feature(
        tmp_path,
        arrays["sample_ids"],
        filename="nonfinite_imagenet.npz",
        values=values,
    )
    arguments["frozen_feature_caches"] = (bad_feature,)

    with pytest.raises(ValueError, match="non-finite"):
        load_pannuke_confirmatory_inputs(**arguments)
