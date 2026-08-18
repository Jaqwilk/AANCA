"""Strict cache, freeze, partition, and independence tests for PanNuke primary inputs."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

import histo_audit.experiment.pannuke_primary_inputs as primary_inputs_module
from histo_audit.config import config_sha256, load_config
from histo_audit.corruption.controlled import (
    FeatureIndependenceEvidence,
    FeatureSpaceEvidence,
    canonical_sha256,
    semantic_sha256,
)
from histo_audit.experiment.pannuke_primary_inputs import (
    PanNukePrimaryCachePaths,
    PanNukePrimaryHashExpectations,
    PanNukePrimaryInputError,
    build_pannuke_primary_inputs,
    select_stratified_reference_validation_groups,
)
from histo_audit.experiment.study_contracts import build_primary_matrix_plan
from histo_audit.representations.cache_provenance import array_artifact_sha256
from histo_audit.representations.eligibility import select_manifest_rows
from histo_audit.representations.imagenet import save_embedding_cache
from histo_audit.utils.run_tracking import sha256_file
from histo_audit.workflows.preregistration import BASE_FREEZE_EVIDENCE_SCHEMA_VERSION


@dataclass(slots=True)
class TinyPrimaryStudy:
    root: Path
    config: dict[str, Any]
    paths: PanNukePrimaryCachePaths
    hashes: PanNukePrimaryHashExpectations
    sample_ids: np.ndarray[Any, np.dtype[np.str_]]
    group_ids: np.ndarray[Any, np.dtype[np.str_]]
    folds: np.ndarray[Any, np.dtype[np.int64]]
    labels: np.ndarray[Any, np.dtype[np.int64]]
    engineered: np.ndarray[Any, np.dtype[np.float64]]
    context_embeddings: np.ndarray[Any, np.dtype[np.float32]]
    highlighted_embeddings: np.ndarray[Any, np.dtype[np.float32]]


def _write_json(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def _write_npz_sidecar(path: Path, metadata: dict[str, Any], **arrays: Any) -> Path:
    np.savez_compressed(path, **arrays)
    payload = dict(metadata)
    array_hashes: dict[str, str] = {}
    for name, value in sorted(arrays.items()):
        try:
            array_hashes[name] = array_artifact_sha256(np.asarray(value))
        except ValueError:
            # Malicious object-array fixtures must still reach the fail-closed loader.
            array_hashes[name] = "0" * 64
    cache_sha256 = sha256_file(path)
    payload["cache_array_sha256_by_name"] = array_hashes
    payload["cache_content_sha256"] = canonical_sha256(array_hashes)
    payload["cache_file_sha256"] = cache_sha256
    payload["cache_npz_sha256"] = cache_sha256
    _write_json(path.with_suffix(f"{path.suffix}.metadata.json"), payload)
    return path


def _analysis_eligibility(sample_ids: np.ndarray[Any, np.dtype[np.str_]]) -> dict[str, Any]:
    n = len(sample_ids)
    table = pa.table(
        {
            "sample_id": sample_ids,
            "primary_eligible": np.ones(n, dtype=bool),
            "confirmatory_eligible": np.ones(n, dtype=bool),
            "cross_class_overlap_touching": np.zeros(n, dtype=bool),
            "qc_exclusion_reason": pa.array([None] * n, type=pa.string()),
        }
    )
    return select_manifest_rows(table, sample_ids=None, scope="analysis").provenance


def _attach_primary_cache_provenance(cache_path: Path, provenance: dict[str, Any]) -> None:
    sidecar_path = cache_path.with_suffix(f"{cache_path.suffix}.metadata.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["primary_cache_provenance"] = provenance
    _write_json(sidecar_path, sidecar)


def _primary_cache_provenance(
    *,
    encoder_id: str,
    encoder_implementation_sha256: str | None,
    weights_sha256: str | None,
    preprocessing_sha256: str | None,
    sample_order_sha256: str,
    dataset_manifest_sha256: str,
    cache_recipe_sha256: str,
    cache_file_sha256: str | None,
) -> dict[str, Any]:
    return {
        "status": "available" if cache_file_sha256 is not None else "unavailable_optional",
        "encoder_id": encoder_id,
        "encoder_implementation_sha256": encoder_implementation_sha256,
        "weights_sha256": weights_sha256,
        "preprocessing_sha256": preprocessing_sha256,
        "sample_order_sha256": sample_order_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "cache_recipe_sha256": cache_recipe_sha256,
        "cache_file_sha256": cache_file_sha256,
    }


def _resolved_primary_config(
    *,
    independence_sha: str,
    pathology_audit_sha: str,
    cache_provenance: dict[str, dict[str, Any]],
    analysis_eligible_sample_count: int,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "primary.yaml")
    config.pop("warning", None)
    config["status"] = "ready_for_freeze"
    data = config["data"]
    data.update(
        {
            "analysis_manifest_authority": {
                "canonical_manifest_sha256": cache_provenance["engineered_target_features"][
                    "dataset_manifest_sha256"
                ],
                "analysis_eligible_sample_order_sha256": cache_provenance[
                    "engineered_target_features"
                ]["sample_order_sha256"],
                "analysis_eligible_sample_count": analysis_eligible_sample_count,
            },
            "development_official_folds": [1, 2],
            "final_test_fold": 3,
            "group_unit": "source_patch_id",
        }
    )
    mechanisms = config["corruption"]["mechanisms"]
    mechanisms["confusion_targeted_corruption"]["transition_matrix"] = [
        [0.0 if row == column else 0.25 for column in range(5)] for row in range(5)
    ]
    mechanisms["group_conditional_corruption"].update(
        {
            "grouping_field": "tissue_type",
            "weights_by_value": {"tissue_a": 1.0, "tissue_b": 0.5},
            "default_weight": 0.5,
        }
    )
    mechanisms["instance_dependent_corruption"].update(
        {
            "independence_status": "per_representation_matrix",
            "independence_matrix_path": "independence.json",
            "independence_matrix_sha256": independence_sha,
        }
    )
    for representation in config["representations"]:
        identifier = str(representation["id"])
        representation["cache_provenance"] = deepcopy(cache_provenance[identifier])
        representation["generator_independence"] = {
            "status": (
                "circularity_risk"
                if representation["family"] == "engineered"
                else "unavailable_optional"
                if representation["family"] == "pathology"
                else "verified_independent"
            ),
            "independence_matrix_sha256": independence_sha,
        }
    config["representations"][-1]["availability_audit_sha256"] = pathology_audit_sha
    config["classifiers"]["multinomial_logistic_regression"].update({"l2": 1.0, "max_iter": 1000})
    config["classifiers"]["small_mlp"].update(
        {
            "hidden_dimensions": [16],
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_accumulation_steps": 1,
        }
    )
    config["calibration"].update(
        {
            "enabled": False,
            "method": "none",
            "seed": 233,
            "parameters": {},
        }
    )
    config["audit"]["primary_method"] = "self_confidence"
    config["audit"]["fixed_hybrid"].update(
        {
            "components": ["self_confidence", "nearest_neighbour_disagreement"],
            "weights": [0.5, 0.5],
        }
    )
    selector = {
        "mechanism": "symmetric_random_corruption",
        "rate": 0.10,
        "seed": 404,
        "representation_id": "imagenet_resnet18_highlighted",
        "classifier_id": "multinomial_logistic_regression",
    }
    config["statistics"].update(
        {
            "holm_families": [
                "primary_within_cell",
                "primary_cross_cell",
                "primary_method_vs_random",
            ],
            "within_cell_comparisons": [
                {
                    "comparison_id": "hybrid_vs_self_confidence",
                    "selector": selector,
                    "method_a": "fixed_hybrid",
                    "method_b": "self_confidence",
                    "metric": "average_precision",
                    "direction": "method_a_minus_method_b",
                    "holm_family": "primary_within_cell",
                }
            ],
            "cross_cell_comparisons": [
                {
                    "comparison_id": "highlighted_vs_context",
                    "selector_a": selector,
                    "selector_b": {
                        **selector,
                        "representation_id": "imagenet_resnet18_context",
                    },
                    "method_a": "self_confidence",
                    "method_b": "self_confidence",
                    "metric": "average_precision",
                    "direction": "method_a_minus_method_b",
                    "holm_family": "primary_cross_cell",
                }
            ],
            "method_vs_random_comparisons": [
                {
                    "comparison_id": "self_confidence_vs_random",
                    "selector": selector,
                    "method_a": "self_confidence",
                    "method_b": "random_review",
                    "metric": "precision_at_budget",
                    "review_budget": 0.05,
                    "direction": "method_a_minus_method_b",
                    "holm_family": "primary_method_vs_random",
                }
            ],
        }
    )
    config["restoration"].update(
        {
            "enabled_cells": ["primary_0027_8531672acd3c"],
            "ranking_method": "self_confidence",
        }
    )
    return config


def _set_independence_sha(config: dict[str, Any], digest: str) -> None:
    config["corruption"]["mechanisms"]["instance_dependent_corruption"][
        "independence_matrix_sha256"
    ] = digest
    for representation in config["representations"]:
        representation["generator_independence"]["independence_matrix_sha256"] = digest


def _bind_tiny_pilot_report(config: dict[str, Any], project_root: Path) -> Path:
    mechanisms = config["corruption"]["mechanisms"]
    run_id = "tiny-primary-pilot-001"
    artifact_root_sha256 = "7" * 64
    report = {
        "schema_version": 1,
        "producer_id": "pilot_derived_primary_parameters_v1",
        "source_pilot": {
            "run_id": run_id,
            "artifact_root_sha256": artifact_root_sha256,
        },
        "confusion_targeted_corruption": {
            "transition_matrix": mechanisms["confusion_targeted_corruption"]["transition_matrix"],
        },
        "group_conditional_corruption": {
            field_name: mechanisms["group_conditional_corruption"][field_name]
            for field_name in ("grouping_field", "weights_by_value", "default_weight")
        },
    }
    report_path = project_root / "reports" / "pilot_derived_primary_parameters.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, report)
    config["pilot_derived_parameters"].update(
        {
            "schema_version": 1,
            "producer_id": "pilot_derived_primary_parameters_v1",
            "path": "reports/pilot_derived_primary_parameters.json",
            "sha256": sha256_file(report_path),
            "source_pilot_run_id": run_id,
            "source_pilot_artifact_root_sha256": artifact_root_sha256,
        }
    )
    return report_path


def _tiny_study(tmp_path: Path) -> TinyPrimaryStudy:
    group_count = 15
    # The frozen 10%-by-group reference selector requires all five classes on
    # both sides.  One sample per class in every source group makes this tiny
    # fixture scientifically feasible without weakening the fail-closed rule.
    samples_per_group = 5
    n = group_count * samples_per_group
    sample_ids = np.asarray([f"sample-{index:03d}" for index in range(n)], dtype=np.str_)
    eligibility = _analysis_eligibility(sample_ids)
    group_ids = np.repeat(
        np.asarray([f"patch-{index:02d}" for index in range(group_count)], dtype=np.str_),
        samples_per_group,
    )
    folds = np.repeat(np.asarray([1] * 6 + [2] * 6 + [3] * 3, dtype=np.int64), samples_per_group)
    labels = np.asarray([index % 5 for index in range(n)], dtype=np.int64)
    rng = np.random.default_rng(51)
    context = rng.integers(0, 256, size=(n, 6, 6, 3), dtype=np.uint8)
    highlighted = context.copy()
    masks = np.zeros((n, 6, 6), dtype=bool)
    masks[:, 2:4, 2:4] = True
    contours = masks.copy()
    raw_component_counts = np.ones(n, dtype=np.int32)
    projected_component_pixel_counts = masks.reshape(n, -1).sum(axis=1).astype(np.int32)
    projection_policy: dict[str, Any] = {
        "schema_version": 1,
        "identifier": "nearest_per_component_with_forward_fallback_v1",
        "raw_component_connectivity": "4-connected",
        "quality_flag": "disconnected_instance_id",
        "raw_identity_action": ("retain_one_raw_identity_without_split_merge_repair_or_relabel"),
        "manifest_and_validation_must_agree_for_disconnected_identity": True,
        "model_facing_mask_definition": (
            "binary union of independently projected raw 4-connected components"
        ),
        "nearest_policy": "fixture valid nearest projection",
        "fallback_policy": "fixture deterministic forward fallback",
        "all_raw_components_must_contribute": True,
        "all_projected_component_footprints_are_4_connected": True,
        "projected_union_topology_is_exact": False,
        "projected_morphology_semantics": "fixture projected morphology",
        "sample_count": n,
        "raw_component_count": n,
        "zero_covered_component_count": 0,
        "disconnected_instance_count": 0,
        "fallback_component_count": 0,
        "fallback_instance_ids": [],
        "collision_instance_count": 0,
        "collision_instance_ids": [],
        "adjacency_instance_count": 0,
        "adjacency_instance_ids": [],
        "topology_changed_instance_count": 0,
        "topology_changed_instance_ids": [],
        "disconnected_instances": [],
        "source_annotations_modified": False,
    }
    projection_policy["semantic_sha256"] = canonical_sha256(projection_policy)
    source_xy = np.tile(np.asarray([[2, 2], [3, 2], [2, 3], [3, 3]], dtype=np.int32), (n, 1))
    offsets = np.arange(0, 4 * (n + 1), 4, dtype=np.int64)
    manifest = tmp_path / "manifest.parquet"
    manifest.write_bytes(b"tiny immutable manifest")
    manifest_sha = sha256_file(manifest)
    crop = tmp_path / "pannuke_crops.npz"
    _write_npz_sidecar(
        crop,
        {
            "schema_version": 1,
            "sample_count": n,
            "manifest_sha256": manifest_sha,
            "raw_inventory_sha256": "c" * 64,
            "crop_configuration": {
                "output_size": 6,
                "padding": 0,
                "context_brightness": 1.0,
            },
            "analysis_eligibility": eligibility,
            "target_mask_projection": projection_policy,
        },
        sample_ids=sample_ids,
        context_rgb=context,
        target_highlighted_rgb=highlighted,
        target_masks=masks,
        target_contour_masks=contours,
        raw_component_counts=raw_component_counts,
        disconnected_instance_flags=np.zeros(n, dtype=bool),
        projected_union_component_counts=np.ones(n, dtype=np.int32),
        projection_fallback_component_counts=np.zeros(n, dtype=np.int32),
        projection_collision_pixel_counts=np.zeros(n, dtype=np.int32),
        projection_collision_excess_counts=np.zeros(n, dtype=np.int32),
        projection_adjacency_pair_counts=np.zeros(n, dtype=np.int32),
        projection_topology_changed=np.zeros(n, dtype=bool),
        projected_component_pixel_counts=projected_component_pixel_counts,
        projected_component_unique_pixel_counts=projected_component_pixel_counts.copy(),
        baseline_projected_component_counts=np.ones(n, dtype=np.int32),
        projection_fallback_component_flags=np.zeros(n, dtype=bool),
        projected_component_offsets=np.arange(n + 1, dtype=np.int64),
        source_crop_boxes=np.tile(np.asarray([[0, 0, 6, 6]], dtype=np.int32), (n, 1)),
        source_target_boxes=np.tile(np.asarray([[2, 2, 4, 4]], dtype=np.int32), (n, 1)),
        official_folds=folds,
        source_patch_indices=np.repeat(np.arange(group_count, dtype=np.int32), samples_per_group),
        instance_channel_indices=labels.astype(np.int16),
        instance_ids=np.arange(1, n + 1, dtype=np.int64),
        pre_corruption_labels=labels,
        group_ids=group_ids,
        tissue_types=np.asarray(
            ["tissue_a" if (index // samples_per_group) % 2 else "tissue_b" for index in range(n)],
            dtype=np.str_,
        ),
        source_contour_xy=source_xy,
        source_contour_offsets=offsets,
        identity_verified=np.ones(n, dtype=bool),
        primary_eligible=np.ones(n, dtype=bool),
        confirmatory_eligible=np.ones(n, dtype=bool),
    )
    crop_sidecar = crop.with_suffix(f"{crop.suffix}.metadata.json")
    crop_metadata = json.loads(crop_sidecar.read_text(encoding="utf-8"))
    common_crop_binding = {
        "crop_cache_file_sha256": sha256_file(crop),
        "crop_cache_sidecar_file_sha256": sha256_file(crop_sidecar),
        "crop_cache_content_sha256": crop_metadata["cache_content_sha256"],
        "crop_manifest_sha256": manifest_sha,
        "raw_inventory_sha256": "c" * 64,
        "sample_order_sha256": canonical_sha256(sample_ids.tolist()),
        "target_mask_projection_semantic_sha256": projection_policy["semantic_sha256"],
    }
    context_crop_binding = {
        "schema_version": 1,
        "binding_type": "pannuke_component_covering_crop_cache_v1",
        **common_crop_binding,
        "input_variant": "context_rgb",
        "input_array_key": "context_rgb",
        "input_array_sha256": crop_metadata["cache_array_sha256_by_name"]["context_rgb"],
    }
    highlighted_crop_binding = {
        "schema_version": 1,
        "binding_type": "pannuke_component_covering_crop_cache_v1",
        **common_crop_binding,
        "input_variant": "target_highlighted_rgb",
        "input_array_key": "target_highlighted_rgb",
        "input_array_sha256": crop_metadata["cache_array_sha256_by_name"]["target_highlighted_rgb"],
    }
    engineered_crop_binding = {
        "schema_version": 1,
        "binding_type": "pannuke_component_covering_crop_cache_engineered_v1",
        **common_crop_binding,
        "input_variant": "context_rgb_plus_component_covering_target_masks",
        "input_array_sha256_by_name": {
            "context_rgb": crop_metadata["cache_array_sha256_by_name"]["context_rgb"],
            "target_masks": crop_metadata["cache_array_sha256_by_name"]["target_masks"],
        },
    }

    feature_names = np.asarray(
        [
            "morphology.area_fraction",
            "morphology.eccentricity",
            "morphology.solidity",
            "colour.target_red_histogram_bin_00",
            "texture.glcm_contrast_mean",
        ],
        dtype=np.str_,
    )
    engineered_values = rng.normal(size=(n, len(feature_names))).astype(np.float64)
    engineered = tmp_path / "pannuke_engineered_features.npz"
    _write_npz_sidecar(
        engineered,
        {
            "schema_version": 1,
            "sample_count": n,
            "feature_count": len(feature_names),
            "crop_manifest_sha256": manifest_sha,
            "analysis_eligibility": eligibility,
            "target_mask_projection_sha256": canonical_sha256(projection_policy),
            "target_mask_projection_semantic_sha256": projection_policy["semantic_sha256"],
            "source_crop_cache_binding": engineered_crop_binding,
            "source_crop_cache_binding_sha256": canonical_sha256(engineered_crop_binding),
        },
        values=engineered_values,
        names=feature_names,
        sample_ids=sample_ids,
    )

    weight_sha = "d" * 64
    context_embeddings = rng.normal(size=(n, 512)).astype(np.float32)
    highlighted_embeddings = rng.normal(size=(n, 512)).astype(np.float32)
    context_path = tmp_path / "context.npz"
    highlighted_path = tmp_path / "highlighted.npz"
    save_embedding_cache(
        context_path,
        context_embeddings,
        sample_ids,
        {
            "input_variant": "rgb",
            "weight_sha256": weight_sha,
            "manifest_sha256": manifest_sha,
            "raw_inventory_sha256": "c" * 64,
            "analysis_eligibility": eligibility,
            "source_crop_cache_binding": context_crop_binding,
        },
    )
    save_embedding_cache(
        highlighted_path,
        highlighted_embeddings,
        sample_ids,
        {
            "input_variant": "target_highlighted_rgb",
            "weight_sha256": weight_sha,
            "manifest_sha256": manifest_sha,
            "raw_inventory_sha256": "c" * 64,
            "analysis_eligibility": eligibility,
            "source_crop_cache_binding": highlighted_crop_binding,
        },
    )

    sample_order_sha = canonical_sha256(sample_ids.tolist())
    no_weights_sha = semantic_sha256("unlearned:no_weights")
    engineered_implementation_sha = semantic_sha256("full engineered features:v1")
    engineered_preprocessing_sha = semantic_sha256("target mask and context RGB:v1")
    imagenet_implementation_sha = semantic_sha256("torchvision frozen resnet18:v1")
    context_preprocessing_sha = semantic_sha256("official ImageNet transforms:v1")
    highlighted_preprocessing_sha = semantic_sha256("target highlighted ImageNet transforms:v1")
    cache_provenance = {
        "engineered_target_features": _primary_cache_provenance(
            encoder_id="engineered_target_features_v1",
            encoder_implementation_sha256=engineered_implementation_sha,
            weights_sha256=no_weights_sha,
            preprocessing_sha256=engineered_preprocessing_sha,
            sample_order_sha256=sample_order_sha,
            dataset_manifest_sha256=manifest_sha,
            cache_recipe_sha256=semantic_sha256("tiny engineered cache recipe:v1"),
            cache_file_sha256=sha256_file(engineered),
        ),
        "imagenet_resnet18_context": _primary_cache_provenance(
            encoder_id="resnet18_imagenet1k_v1",
            encoder_implementation_sha256=imagenet_implementation_sha,
            weights_sha256=weight_sha,
            preprocessing_sha256=context_preprocessing_sha,
            sample_order_sha256=sample_order_sha,
            dataset_manifest_sha256=manifest_sha,
            cache_recipe_sha256=semantic_sha256("tiny context embedding cache recipe:v1"),
            cache_file_sha256=sha256_file(context_path),
        ),
        "imagenet_resnet18_highlighted": _primary_cache_provenance(
            encoder_id="resnet18_imagenet1k_v1",
            encoder_implementation_sha256=imagenet_implementation_sha,
            weights_sha256=weight_sha,
            preprocessing_sha256=highlighted_preprocessing_sha,
            sample_order_sha256=sample_order_sha,
            dataset_manifest_sha256=manifest_sha,
            cache_recipe_sha256=semantic_sha256("tiny highlighted cache recipe:v1"),
            cache_file_sha256=sha256_file(highlighted_path),
        ),
        "pathology_encoder_optional": _primary_cache_provenance(
            encoder_id="availability_selected_pathology_encoder",
            encoder_implementation_sha256=None,
            weights_sha256=None,
            preprocessing_sha256=None,
            sample_order_sha256=sample_order_sha,
            dataset_manifest_sha256=manifest_sha,
            cache_recipe_sha256=semantic_sha256("tiny blocked pathology recipe:v1"),
            cache_file_sha256=None,
        ),
    }
    _attach_primary_cache_provenance(engineered, cache_provenance["engineered_target_features"])
    _attach_primary_cache_provenance(context_path, cache_provenance["imagenet_resnet18_context"])
    _attach_primary_cache_provenance(
        highlighted_path, cache_provenance["imagenet_resnet18_highlighted"]
    )

    pathology_audit = _write_json(
        tmp_path / "pathology_audit.json",
        {
            "status": "blocked",
            "blocker": "no encoder passed the frozen availability gates",
            "primary_cache_provenance": cache_provenance["pathology_encoder_optional"],
        },
    )
    provisional = _resolved_primary_config(
        independence_sha="0" * 64,
        pathology_audit_sha=sha256_file(pathology_audit),
        cache_provenance=cache_provenance,
        analysis_eligible_sample_count=n,
    )
    data = provisional["data"]
    development = folds != int(data["final_test_fold"])
    validation_groups = select_stratified_reference_validation_groups(
        labels[development],
        group_ids[development],
        class_order=(0, 1, 2, 3, 4),
        fraction=float(data["reference_validation_fraction_groups"]),
        seed=int(data["split_seed"]),
    )
    audit_indices = np.flatnonzero(development & ~np.isin(group_ids, validation_groups))
    morphology_audit = engineered_values[audit_indices, :3]
    context_audit = context_embeddings[audit_indices]
    fitted_hash = canonical_sha256(
        {
            "sample_ids": sample_ids[audit_indices].tolist(),
            "group_ids": group_ids[audit_indices].tolist(),
        }
    )
    generator = FeatureSpaceEvidence.from_array(
        morphology_audit,
        representation_name="morphology_only_v1",
        family="morphology",
        implementation_hash=engineered_implementation_sha,
        weights_hash=no_weights_sha,
        preprocessing_hash=engineered_preprocessing_sha,
        fitted_data_hash=fitted_hash,
    )
    auditor = FeatureSpaceEvidence.from_array(
        context_audit,
        representation_name="imagenet_resnet18_context",
        family="imagenet",
        implementation_hash=imagenet_implementation_sha,
        weights_hash=weight_sha,
        preprocessing_hash=context_preprocessing_sha,
        fitted_data_hash=fitted_hash,
    )
    highlighted_auditor = FeatureSpaceEvidence.from_array(
        highlighted_embeddings[audit_indices],
        representation_name="imagenet_resnet18_highlighted",
        family="imagenet",
        implementation_hash=imagenet_implementation_sha,
        weights_hash=weight_sha,
        preprocessing_hash=highlighted_preprocessing_sha,
        fitted_data_hash=fitted_hash,
    )
    engineered_auditor = FeatureSpaceEvidence.from_array(
        engineered_values[audit_indices],
        representation_name="engineered_target_features",
        family="engineered",
        implementation_hash=engineered_implementation_sha,
        weights_hash=no_weights_sha,
        preprocessing_hash=engineered_preprocessing_sha,
        fitted_data_hash=fitted_hash,
    )
    entries = {
        "engineered_target_features": FeatureIndependenceEvidence.create(
            matrix_version="tiny_primary_independence_v2",
            matrix_decision="not_independent",
            matrix_reason=(
                "The engineered auditor contains the exact morphology generator columns; "
                "its instance-dependent result is circularity_risk."
            ),
            generator=generator,
            auditor=engineered_auditor,
        ),
        "imagenet_resnet18_context": FeatureIndependenceEvidence.create(
            matrix_version="tiny_primary_independence_v2",
            matrix_decision="verified_independent",
            matrix_reason="Morphology-only generator and frozen context encoder are disjoint.",
            generator=generator,
            auditor=auditor,
        ),
        "imagenet_resnet18_highlighted": FeatureIndependenceEvidence.create(
            matrix_version="tiny_primary_independence_v2",
            matrix_decision="verified_independent",
            matrix_reason="Morphology-only generator and highlighted encoder are disjoint.",
            generator=generator,
            auditor=highlighted_auditor,
        ),
    }
    independence_path = _write_json(
        tmp_path / "independence.json",
        {
            "schema_version": 2,
            "entries": {identifier: evidence.as_dict() for identifier, evidence in entries.items()},
        },
    )
    config = _resolved_primary_config(
        independence_sha=sha256_file(independence_path),
        pathology_audit_sha=sha256_file(pathology_audit),
        cache_provenance=cache_provenance,
        analysis_eligible_sample_count=n,
    )
    _bind_tiny_pilot_report(config, tmp_path)
    dataset_evidence = _write_json(
        tmp_path / "dataset_validation.json",
        {"status": "valid", "release_complete": True, "raw_inventory_sha256": "c" * 64},
    )
    dataset_evidence_sha = sha256_file(dataset_evidence)
    freeze = tmp_path / "freeze.json"
    _write_json(
        freeze,
        {
            "schema_version": BASE_FREEZE_EVIDENCE_SCHEMA_VERSION,
            "completion_stage_enabled": "PRE_REGISTRATION_FROZEN",
            "primary_config": {"semantic_sha256": config_sha256(config)},
            "dataset": {"sha256": dataset_evidence_sha},
            "manifest": {"sha256": manifest_sha},
        },
    )
    paths = PanNukePrimaryCachePaths(
        crop_cache_path=crop,
        engineered_cache_path=engineered,
        context_embedding_cache_path=context_path,
        highlighted_embedding_cache_path=highlighted_path,
        pathology_embedding_cache_path=None,
        pathology_availability_audit_path=pathology_audit,
        dataset_evidence_path=dataset_evidence,
        dataset_manifest_path=manifest,
        freeze_record_path=freeze,
    )
    hashes = PanNukePrimaryHashExpectations(
        dataset_evidence_sha256=dataset_evidence_sha,
        dataset_manifest_sha256=manifest_sha,
        raw_inventory_sha256="c" * 64,
        crop_cache_sha256=sha256_file(crop),
        engineered_cache_sha256=sha256_file(engineered),
        context_embedding_cache_sha256=sha256_file(context_path),
        highlighted_embedding_cache_sha256=sha256_file(highlighted_path),
        freeze_record_sha256=sha256_file(freeze),
    )
    return TinyPrimaryStudy(
        tmp_path,
        config,
        paths,
        hashes,
        sample_ids,
        group_ids,
        folds,
        labels,
        engineered_values,
        context_embeddings,
        highlighted_embeddings,
    )


def _build(study: TinyPrimaryStudy, *, config: dict[str, Any] | None = None, **kwargs: Any):
    selected = study.config if config is None else config
    plan = build_primary_matrix_plan(selected)
    hashes = study.hashes
    if config is not None and config is not study.config:
        assert study.paths.freeze_record_path is not None
        freeze_payload = json.loads(study.paths.freeze_record_path.read_text(encoding="utf-8"))
        freeze_payload["primary_config"]["semantic_sha256"] = config_sha256(selected)
        _write_json(study.paths.freeze_record_path, freeze_payload)
        hashes = replace(
            study.hashes,
            freeze_record_sha256=sha256_file(study.paths.freeze_record_path),
        )
    defaults = {
        "expected_config_sha256": config_sha256(selected),
        "expected_plan_semantic_sha256": canonical_sha256(plan.as_dict()),
        "project_root": study.root,
        "expected_hashes": hashes,
    }
    defaults.update(kwargs)
    return build_pannuke_primary_inputs(selected, study.paths, **defaults)


def test_builds_strict_group_safe_inputs_and_records_optional_pathology(tmp_path: Path) -> None:
    study = _tiny_study(tmp_path)

    result = _build(study)

    inputs = result.inputs
    assert set(inputs.audit_group_ids).isdisjoint(inputs.reference_validation_group_ids)
    assert set(inputs.audit_group_ids).isdisjoint(inputs.final_test_group_ids)
    assert set(inputs.reference_validation_group_ids).isdisjoint(inputs.final_test_group_ids)
    final_by_id = {
        sample_id: int(fold) for sample_id, fold in zip(study.sample_ids, study.folds, strict=True)
    }
    assert {final_by_id[sample_id] for sample_id in inputs.final_test_sample_ids} == {3}
    assert np.array_equal(
        inputs.final_test_labels,
        np.asarray(
            [
                study.labels[list(study.sample_ids).index(sample_id)]
                for sample_id in inputs.final_test_sample_ids
            ]
        ),
    )
    assert result.morphology_feature_names == (
        "morphology.area_fraction",
        "morphology.eccentricity",
        "morphology.solidity",
    )
    assert inputs.corruption_generator_features.shape[1] == 3
    assert study.engineered.shape[1] == 5
    assert inputs.corruption_auditor_representation is None
    assert inputs.independence_evidence is None
    assert set(result.independence_evidence_by_representation) == {
        "engineered_target_features",
        "imagenet_resnet18_context",
        "imagenet_resnet18_highlighted",
    }
    assert (
        result.independence_evidence_by_representation["engineered_target_features"].matrix_decision
        == "not_independent"
    )
    pathology = result.representation_availability[-1]
    assert pathology.status == "unavailable_optional"
    assert "availability gates" in str(pathology.blocker)
    assert "pathology_encoder_optional" not in inputs.audit_features
    assert len(result.sample_order_sha256) == len(result.partition_assignment_sha256) == 64
    assert set(result.cache_provenance_by_representation) == {
        "engineered_target_features",
        "imagenet_resnet18_context",
        "imagenet_resnet18_highlighted",
        "pathology_encoder_optional",
    }
    assert (
        result.cache_provenance_by_representation["imagenet_resnet18_context"]["cache_file_sha256"]
        == study.hashes.context_embedding_cache_sha256
    )
    assert result.verified_hashes["freeze_record_sha256"] == study.hashes.freeze_record_sha256


def test_wrong_frozen_config_or_plan_digest_fails_closed(tmp_path: Path) -> None:
    study = _tiny_study(tmp_path)
    plan = build_primary_matrix_plan(study.config)

    with pytest.raises(PanNukePrimaryInputError, match="config semantic"):
        build_pannuke_primary_inputs(
            study.config,
            study.paths,
            expected_config_sha256="0" * 64,
            expected_plan_semantic_sha256=canonical_sha256(plan.as_dict()),
            project_root=study.root,
            expected_hashes=study.hashes,
        )
    with pytest.raises(PanNukePrimaryInputError, match="plan semantic"):
        build_pannuke_primary_inputs(
            study.config,
            study.paths,
            expected_config_sha256=config_sha256(study.config),
            expected_plan_semantic_sha256="0" * 64,
            project_root=study.root,
        )


def test_missing_pilot_derived_parameter_report_fails_before_cache_loading(
    tmp_path: Path,
) -> None:
    study = _tiny_study(tmp_path)
    report_path = study.root / str(study.config["pilot_derived_parameters"]["path"])
    report_path.unlink()

    with pytest.raises(PanNukePrimaryInputError, match="parameter report is missing"):
        _build(study)


def test_tampered_pilot_derived_parameter_report_fails_its_frozen_hash(
    tmp_path: Path,
) -> None:
    study = _tiny_study(tmp_path)
    report_path = study.root / str(study.config["pilot_derived_parameters"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["producer_id"] = "tampered_producer"
    _write_json(report_path, report)

    with pytest.raises(PanNukePrimaryInputError, match="report SHA-256 differs"):
        _build(study)


def test_rehashed_pilot_report_cannot_disagree_with_frozen_parameter_values(
    tmp_path: Path,
) -> None:
    study = _tiny_study(tmp_path)
    report_path = study.root / str(study.config["pilot_derived_parameters"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["group_conditional_corruption"]["default_weight"] = 0.25
    _write_json(report_path, report)
    config = deepcopy(study.config)
    config["pilot_derived_parameters"]["sha256"] = sha256_file(report_path)

    with pytest.raises(
        PanNukePrimaryInputError,
        match="group-conditional default_weight differs",
    ):
        _build(study, config=config)


def test_embedding_sample_order_mismatch_is_rejected(tmp_path: Path) -> None:
    study = _tiny_study(tmp_path)
    reordered = tmp_path / "highlighted_reordered.npz"
    permutation = np.arange(len(study.sample_ids))[::-1]
    save_embedding_cache(
        reordered,
        study.context_embeddings[permutation],
        study.sample_ids[permutation],
        {"input_variant": "target_highlighted_rgb", "weight_sha256": "d" * 64},
    )
    paths = replace(study.paths, highlighted_embedding_cache_path=reordered)
    plan = build_primary_matrix_plan(study.config)

    with pytest.raises(PanNukePrimaryInputError, match="sample IDs/order"):
        build_pannuke_primary_inputs(
            study.config,
            paths,
            expected_config_sha256=config_sha256(study.config),
            expected_plan_semantic_sha256=canonical_sha256(plan.as_dict()),
            project_root=study.root,
            expected_hashes=study.hashes,
        )


def test_modified_highlighted_crop_is_rejected_before_stale_embedding_use(
    tmp_path: Path,
) -> None:
    study = _tiny_study(tmp_path)
    crop = study.paths.crop_cache_path
    with np.load(crop, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays["target_highlighted_rgb"] = arrays["target_highlighted_rgb"].copy()
    arrays["target_highlighted_rgb"][0, 0, 0, 0] ^= np.uint8(1)
    metadata = json.loads(crop.with_suffix(".npz.metadata.json").read_text(encoding="utf-8"))
    _write_npz_sidecar(crop, metadata, **arrays)

    with pytest.raises(PanNukePrimaryInputError, match="highlighted crop content differs"):
        _build(
            study,
            expected_hashes=replace(
                study.hashes,
                crop_cache_sha256=sha256_file(crop),
            ),
        )


def test_embedding_rejects_exact_crop_artifact_mismatch_even_when_ids_match(
    tmp_path: Path,
) -> None:
    study = _tiny_study(tmp_path)
    crop_path = study.paths.crop_cache_path
    with np.load(crop_path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays["tissue_types"] = arrays["tissue_types"].copy()
    arrays["tissue_types"][0] = "tissue_z"
    metadata = json.loads(crop_path.with_suffix(".npz.metadata.json").read_text(encoding="utf-8"))
    _write_npz_sidecar(crop_path, metadata, **arrays)
    crop = primary_inputs_module._load_crop_cache(crop_path, (0, 1, 2, 3, 4))
    assert study.paths.highlighted_embedding_cache_path is not None

    with pytest.raises(PanNukePrimaryInputError, match="not bound to the exact crop cache"):
        primary_inputs_module._verified_imagenet_cache(
            study.paths.highlighted_embedding_cache_path,
            expected_ids=crop.sample_ids,
            expected_variant="target_highlighted_rgb",
            expected_crop=crop,
        )


def test_crop_identity_and_pickle_dependent_schema_fail_closed(tmp_path: Path) -> None:
    study = _tiny_study(tmp_path)
    crop = study.paths.crop_cache_path
    with np.load(crop, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays["identity_verified"] = np.zeros(len(study.sample_ids), dtype=bool)
    metadata = json.loads(crop.with_suffix(".npz.metadata.json").read_text(encoding="utf-8"))
    metadata.pop("cache_npz_sha256")
    _write_npz_sidecar(crop, metadata, **arrays)
    plan = build_primary_matrix_plan(study.config)

    with pytest.raises(PanNukePrimaryInputError, match="verified raw-instance identity"):
        build_pannuke_primary_inputs(
            study.config,
            study.paths,
            expected_config_sha256=config_sha256(study.config),
            expected_plan_semantic_sha256=canonical_sha256(plan.as_dict()),
            project_root=study.root,
        )


def test_crop_cache_rejects_any_false_or_divergent_analysis_eligibility_mask(
    tmp_path: Path,
) -> None:
    study = _tiny_study(tmp_path)
    crop = study.paths.crop_cache_path
    with np.load(crop, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays["primary_eligible"] = arrays["primary_eligible"].copy()
    arrays["primary_eligible"][0] = False
    metadata = json.loads(crop.with_suffix(".npz.metadata.json").read_text(encoding="utf-8"))
    metadata.pop("cache_npz_sha256")
    _write_npz_sidecar(crop, metadata, **arrays)
    plan = build_primary_matrix_plan(study.config)

    with pytest.raises(PanNukePrimaryInputError, match="eligibility arrays"):
        build_pannuke_primary_inputs(
            study.config,
            study.paths,
            expected_config_sha256=config_sha256(study.config),
            expected_plan_semantic_sha256=canonical_sha256(plan.as_dict()),
            project_root=tmp_path,
        )


def test_crop_cache_rejects_component_vector_without_disconnected_ledger(
    tmp_path: Path,
) -> None:
    study = _tiny_study(tmp_path)
    crop = study.paths.crop_cache_path
    with np.load(crop, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays["raw_component_counts"] = arrays["raw_component_counts"].copy()
    arrays["raw_component_counts"][0] = 2
    arrays["disconnected_instance_flags"] = arrays["disconnected_instance_flags"].copy()
    arrays["disconnected_instance_flags"][0] = True
    arrays["projected_union_component_counts"] = arrays["projected_union_component_counts"].copy()
    arrays["projected_union_component_counts"][0] = 1
    arrays["projection_topology_changed"] = arrays["projection_topology_changed"].copy()
    arrays["projection_topology_changed"][0] = True
    component_pixels = arrays["projected_component_pixel_counts"]
    arrays["projected_component_pixel_counts"] = np.insert(component_pixels, 1, 2)
    arrays["projected_component_pixel_counts"][0] = 2
    component_unique = arrays["projected_component_unique_pixel_counts"]
    arrays["projected_component_unique_pixel_counts"] = np.insert(component_unique, 1, 2)
    arrays["projected_component_unique_pixel_counts"][0] = 2
    arrays["baseline_projected_component_counts"] = np.insert(
        arrays["baseline_projected_component_counts"],
        1,
        1,
    )
    arrays["projection_fallback_component_flags"] = np.insert(
        arrays["projection_fallback_component_flags"],
        1,
        False,
    )
    offsets = arrays["projected_component_offsets"].copy()
    offsets[1:] += 1
    arrays["projected_component_offsets"] = offsets
    metadata = json.loads(crop.with_suffix(".npz.metadata.json").read_text(encoding="utf-8"))
    metadata.pop("cache_npz_sha256")
    _write_npz_sidecar(crop, metadata, **arrays)
    plan = build_primary_matrix_plan(study.config)

    with pytest.raises(
        PanNukePrimaryInputError,
        match="target-mask projection provenance is invalid",
    ):
        build_pannuke_primary_inputs(
            study.config,
            study.paths,
            expected_config_sha256=config_sha256(study.config),
            expected_plan_semantic_sha256=canonical_sha256(plan.as_dict()),
            project_root=study.root,
        )


def test_crop_cache_recomputes_union_topology_and_rejects_impossible_bounds(
    tmp_path: Path,
) -> None:
    study = _tiny_study(tmp_path)
    crop = study.paths.crop_cache_path
    with np.load(crop, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays["projected_union_component_counts"] = arrays["projected_union_component_counts"].copy()
    arrays["projected_union_component_counts"][0] = 2
    arrays["projection_topology_changed"] = arrays["projection_topology_changed"].copy()
    arrays["projection_topology_changed"][0] = True
    metadata = json.loads(crop.with_suffix(".npz.metadata.json").read_text(encoding="utf-8"))
    _write_npz_sidecar(crop, metadata, **arrays)

    with pytest.raises(PanNukePrimaryInputError, match="component-projection vectors"):
        primary_inputs_module._load_crop_cache(crop, (0, 1, 2, 3, 4))


def test_pickle_dependent_crop_array_is_never_loaded(tmp_path: Path) -> None:
    study = _tiny_study(tmp_path)
    crop = study.paths.crop_cache_path
    with np.load(crop, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays["group_ids"] = np.asarray(study.group_ids.tolist(), dtype=object)
    metadata = json.loads(crop.with_suffix(".npz.metadata.json").read_text(encoding="utf-8"))
    metadata.pop("cache_npz_sha256")
    _write_npz_sidecar(crop, metadata, **arrays)
    plan = build_primary_matrix_plan(study.config)

    with pytest.raises(PanNukePrimaryInputError, match=r"safely load|object"):
        build_pannuke_primary_inputs(
            study.config,
            study.paths,
            expected_config_sha256=config_sha256(study.config),
            expected_plan_semantic_sha256=canonical_sha256(plan.as_dict()),
            project_root=study.root,
        )


def test_required_pathology_cannot_be_silently_omitted(tmp_path: Path) -> None:
    study = _tiny_study(tmp_path)
    config = deepcopy(study.config)
    config["representations"][-1]["required"] = True

    with pytest.raises(ValueError, match="required and cannot use unavailable_optional"):
        _build(study, config=config)


def test_generator_must_match_only_named_morphology_columns(tmp_path: Path) -> None:
    study = _tiny_study(tmp_path)
    evidence_path = study.root / "independence.json"
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    raw_entry = payload["entries"]["imagenet_resnet18_context"]
    raw_entry["generator"]["feature_artifact_hash"] = "f" * 64
    # Recompute the internal decision hash so only the concrete artifact binding is wrong.
    generator = FeatureSpaceEvidence(**raw_entry["generator"])
    auditor = FeatureSpaceEvidence(**raw_entry["auditor"])
    changed = FeatureIndependenceEvidence.create(
        matrix_version=raw_entry["matrix_version"],
        matrix_decision=raw_entry["matrix_decision"],
        matrix_reason=raw_entry["matrix_reason"],
        generator=generator,
        auditor=auditor,
    )
    payload["entries"]["imagenet_resnet18_context"] = changed.as_dict()
    _write_json(evidence_path, payload)
    config = deepcopy(study.config)
    _set_independence_sha(config, sha256_file(evidence_path))

    with pytest.raises(PanNukePrimaryInputError, match=r"same concrete generator|morphology-only"):
        _build(study, config=config)


def test_every_available_representation_requires_exact_evidence(tmp_path: Path) -> None:
    study = _tiny_study(tmp_path)
    evidence_path = study.root / "independence.json"
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["entries"].pop("imagenet_resnet18_highlighted")
    _write_json(evidence_path, payload)
    config = deepcopy(study.config)
    _set_independence_sha(config, sha256_file(evidence_path))

    with pytest.raises(PanNukePrimaryInputError, match="exactly every available"):
        _build(study, config=config)


def test_engineered_morphology_overlap_is_always_circularity_risk(tmp_path: Path) -> None:
    study = _tiny_study(tmp_path)
    evidence_path = study.root / "independence.json"
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    raw = payload["entries"]["engineered_target_features"]
    false_claim = FeatureIndependenceEvidence.create(
        matrix_version=raw["matrix_version"],
        matrix_decision="verified_independent",
        matrix_reason="Incorrect independence claim used only to exercise the fail-closed gate.",
        generator=FeatureSpaceEvidence(**raw["generator"]),
        auditor=FeatureSpaceEvidence(**raw["auditor"]),
    )
    payload["entries"]["engineered_target_features"] = false_claim.as_dict()
    _write_json(evidence_path, payload)
    config = deepcopy(study.config)
    _set_independence_sha(config, sha256_file(evidence_path))

    with pytest.raises(PanNukePrimaryInputError, match="circularity_risk"):
        _build(study, config=config)


def test_cache_sidecar_semantics_must_equal_frozen_representation_provenance(
    tmp_path: Path,
) -> None:
    study = _tiny_study(tmp_path)
    assert study.paths.context_embedding_cache_path is not None
    sidecar_path = study.paths.context_embedding_cache_path.with_suffix(
        f"{study.paths.context_embedding_cache_path.suffix}.metadata.json"
    )
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["primary_cache_provenance"]["cache_recipe_sha256"] = "0" * 64
    _write_json(sidecar_path, payload)

    with pytest.raises(PanNukePrimaryInputError, match="sidecar provenance differs"):
        _build(study)


def test_frozen_cache_semantics_cannot_be_config_only_unverified_hashes(tmp_path: Path) -> None:
    study = _tiny_study(tmp_path)
    config = deepcopy(study.config)
    config["representations"][0]["cache_provenance"]["encoder_implementation_sha256"] = "0" * 64

    with pytest.raises(PanNukePrimaryInputError, match="sidecar provenance differs"):
        _build(study, config=config)


def test_outer_manifest_cache_and_freeze_hash_expectations_are_enforced(tmp_path: Path) -> None:
    study = _tiny_study(tmp_path)
    bad = replace(study.hashes, freeze_record_sha256="e" * 64)

    with pytest.raises(PanNukePrimaryInputError, match="freeze record"):
        _build(study, expected_hashes=bad)


def test_freeze_record_content_must_bind_config_and_manifest(tmp_path: Path) -> None:
    study = _tiny_study(tmp_path)
    assert study.paths.freeze_record_path is not None
    payload = json.loads(study.paths.freeze_record_path.read_text(encoding="utf-8"))
    payload["primary_config"]["semantic_sha256"] = "f" * 64
    _write_json(study.paths.freeze_record_path, payload)
    updated = replace(
        study.hashes,
        freeze_record_sha256=sha256_file(study.paths.freeze_record_path),
    )

    with pytest.raises(PanNukePrimaryInputError, match="not bound"):
        _build(study, expected_hashes=updated)


def test_freeze_record_rejects_legacy_schema_even_when_bindings_match(tmp_path: Path) -> None:
    study = _tiny_study(tmp_path)
    assert study.paths.freeze_record_path is not None
    payload = json.loads(study.paths.freeze_record_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    _write_json(study.paths.freeze_record_path, payload)
    updated = replace(
        study.hashes,
        freeze_record_sha256=sha256_file(study.paths.freeze_record_path),
    )

    with pytest.raises(PanNukePrimaryInputError, match="schema_version must be 3"):
        _build(study, expected_hashes=updated)
