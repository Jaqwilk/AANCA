"""Fail-closed tests for frozen primary and confirmatory study contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from inspect import signature
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from histo_audit.config import config_sha256
from histo_audit.corruption.controlled import (
    FeatureIndependenceEvidence,
    FeatureSpaceEvidence,
    canonical_sha256,
    semantic_sha256,
)
from histo_audit.experiment import primary_core as primary_core_module
from histo_audit.experiment.primary_core import (
    PrimaryExecutionControls,
    PrimaryMatrixInputs,
    execute_primary_matrix,
    primary_execution_controls_from_frozen_config,
)
from histo_audit.experiment.study_contracts import (
    StudyContractError,
    build_confirmatory_matrix_plan,
    build_primary_matrix_plan,
    validate_frozen_confirmatory_config,
    validate_frozen_primary_config,
    validate_primary_confirmatory_cross_config,
)


def _primary_cache_provenance(
    *,
    encoder_id: str,
    digest_character: str,
    available: bool = True,
) -> dict[str, Any]:
    return {
        "status": "available" if available else "unavailable_optional",
        "encoder_id": encoder_id,
        "encoder_implementation_sha256": digest_character * 64 if available else None,
        "weights_sha256": "c" * 64 if available else None,
        "preprocessing_sha256": "d" * 64 if available else None,
        "sample_order_sha256": "e" * 64,
        "dataset_manifest_sha256": "f" * 64,
        "cache_recipe_sha256": digest_character * 64,
        "cache_file_sha256": digest_character * 64 if available else None,
    }


def complete_primary_config() -> dict[str, Any]:
    """Return a small but scientifically complete schema-v2 primary plan."""

    return {
        "schema_version": 2,
        "experiment_name": "primary_frozen_feature_benchmark",
        "status": "ready_for_freeze",
        "pilot_derived_parameters": {
            "schema_version": 1,
            "producer_id": "pilot_derived_primary_parameters_v1",
            "path": "reports/pilot_derived_primary_parameters.json",
            "sha256": "8380b963a02b7ea4451039e9e5b37600809b22c689734202664522bdeda6113b",
            "source_pilot_run_id": "20260718T143216.354310Z_pannuke_pilot_c7797330e0",
            "source_pilot_artifact_root_sha256": (
                "37a9cdc4aab1eb74dc6e86555dfeb96f7682d8bc17bdb0e3a12ec6ab18254666"
            ),
        },
        "data": {
            "source": "pannuke",
            "class_order": [0, 1, 2, 3, 4],
            "analysis_manifest_authority": {
                "canonical_manifest_sha256": "f" * 64,
                "analysis_eligible_sample_order_sha256": "e" * 64,
                "analysis_eligible_sample_count": 100,
            },
            "development_official_folds": [1, 2],
            "final_test_fold": 3,
            "group_unit": "source_patch_id",
            "reference_validation_fraction_groups": 0.10,
            "reference_group_selection_algorithm": (
                "deterministic_group_greedy_class_distribution_v1"
            ),
            "split_seed": 223,
            "fold_assignment_labels": "pre_corruption_label",
            "exclusions": {
                "cross_class_overlap_touching": ("exclude_with_reason_touches_cross_class_overlap"),
                "positive_background_conflict_touching": (
                    "retain_with_qc_flag_no_class_arbitration"
                ),
                "disconnected_instance_id": "retain_with_flag",
                "border_instance": "retain_with_flag",
                "malformed_or_structurally_invalid_mask": "fail_closed_at_dataset_gate",
                "duplicates": "flag_without_automatic_deletion",
                "void_pixels": "retain_as_unlabeled_void",
                "missing_required_data": "fail_closed_at_dataset_gate",
            },
        },
        "corruption": {
            "rounding_policy": "round_half_up",
            "rates": [0.05, 0.10, 0.20],
            "seeds": [404, 405, 406],
            "clean_reference": {
                "id": "clean_reference_cell",
                "mechanism": "symmetric_random_corruption",
                "rate": 0.0,
                "seed": 404,
                "parameters": {},
            },
            "mechanisms": {
                "symmetric_random_corruption": {},
                "confusion_targeted_corruption": {
                    "transition_matrix": [
                        [0.0, 0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.0, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.0, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.0, 0.25],
                        [0.25, 0.25, 0.25, 0.25, 0.0],
                    ]
                },
                "group_conditional_corruption": {
                    "grouping_field": "tissue_type",
                    "weights_by_value": {"tissue_a": 1.0, "tissue_b": 0.5},
                    "default_weight": 0.5,
                },
                "instance_dependent_corruption": {
                    "generator_representation": "morphology_only_v1",
                    "auditor_representation_families": ["engineered", "imagenet"],
                    "independence_status": "per_representation_matrix",
                    "independence_matrix_path": "reports/representation_independence.json",
                    "independence_matrix_sha256": "a" * 64,
                },
            },
        },
        "representations": [
            {
                "id": "engineered_target_features",
                "family": "engineered",
                "input_variant": "context_rgb",
                "required": True,
                "classifiers": ["multinomial_logistic_regression"],
                "cache_provenance": _primary_cache_provenance(
                    encoder_id="engineered_target_features_v1",
                    digest_character="1",
                ),
                "generator_independence": {
                    "status": "circularity_risk",
                    "independence_matrix_sha256": "a" * 64,
                },
            },
            {
                "id": "imagenet_resnet18_context",
                "family": "imagenet",
                "input_variant": "context_rgb",
                "required": True,
                "classifiers": ["multinomial_logistic_regression", "small_mlp"],
                "cache_provenance": _primary_cache_provenance(
                    encoder_id="resnet18_imagenet1k_v1",
                    digest_character="2",
                ),
                "generator_independence": {
                    "status": "verified_independent",
                    "independence_matrix_sha256": "a" * 64,
                },
            },
            {
                "id": "imagenet_resnet18_highlighted",
                "family": "imagenet",
                "input_variant": "target_highlighted_rgb",
                "required": True,
                "classifiers": ["multinomial_logistic_regression", "small_mlp"],
                "cache_provenance": _primary_cache_provenance(
                    encoder_id="resnet18_imagenet1k_v1",
                    digest_character="3",
                ),
                "generator_independence": {
                    "status": "verified_independent",
                    "independence_matrix_sha256": "a" * 64,
                },
            },
            {
                "id": "pathology_encoder_optional",
                "family": "pathology",
                "input_variant": "context_rgb",
                "required": False,
                "availability_audit_sha256": "b" * 64,
                "classifiers": ["multinomial_logistic_regression"],
                "cache_provenance": _primary_cache_provenance(
                    encoder_id="availability_selected_pathology_encoder",
                    digest_character="4",
                    available=False,
                ),
                "generator_independence": {
                    "status": "unavailable_optional",
                    "independence_matrix_sha256": "a" * 64,
                },
            },
        ],
        "classifiers": {
            "multinomial_logistic_regression": {
                "l2": 1.0,
                "max_iter": 1000,
                "class_weight": "balanced",
                "class_weight_label_source": "observed_development_labels_only",
                "fit_label_source": "observed_development_labels_only",
                "model_seed": 227,
            },
            "small_mlp": {
                "hidden_dimensions": [128, 64],
                "dropout": 0.2,
                "epochs": 100,
                "batch_size": 128,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "model_seed": 229,
                "amp": True,
                "amp_dtype": "float16",
                "gradient_accumulation_steps": 1,
                "minimum_batch_size": 1,
                "early_stopping_patience": None,
                "early_stopping_min_delta": 0.0,
                "fit_label_source": "observed_development_labels_only",
            },
        },
        "calibration": {
            "enabled": False,
            "method": "none",
            "source": "reference_validation_only",
            "reporting": "calibrated_and_uncalibrated",
            "fit_labels_policy": "observed_reference_validation_labels_only",
            "seed": 233,
            "parameters": {},
        },
        "oof": {
            "n_splits": 5,
            "split_kind": "stratified_group",
            "no_nucleus_level_fallback": True,
        },
        "audit": {
            "methods": [
                "self_confidence",
                "negative_log_likelihood",
                "prediction_margin",
                "predictive_entropy",
                "cleanlab",
                "nearest_neighbour_disagreement",
                "fixed_hybrid",
            ],
            "primary_method": "self_confidence",
            "nearest_neighbour": {"k": 7, "metric": "cosine", "exclude_same_group": True},
            "fixed_hybrid": {
                "components": ["self_confidence", "nearest_neighbour_disagreement"],
                "weights": [0.5, 0.5],
            },
            "cleanlab_failure_policy": "missing_with_recorded_blocker",
        },
        "evaluation": {
            "primary_metric": "average_precision",
            "primary_review_budget": 0.05,
            "secondary_review_budgets": [0.01, 0.10, 0.20],
            "random_review_repeats": 100,
            "random_review_seed": 419,
            "subgroup_min_samples": 100,
            "subgroup_min_corruptions": 10,
        },
        "statistics": {
            "paired_group_bootstrap_iterations": 2000,
            "bootstrap_seed": 431,
            "holm_families": [
                "primary_within_cell",
                "primary_cross_cell",
                "primary_method_vs_random",
            ],
            "within_cell_comparisons": [
                {
                    "comparison_id": "hybrid_vs_self_confidence",
                    "selector": {
                        "mechanism": "symmetric_random_corruption",
                        "rate": 0.10,
                        "seed": 404,
                        "representation_id": "imagenet_resnet18_highlighted",
                        "classifier_id": "multinomial_logistic_regression",
                    },
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
                    "selector_a": {
                        "mechanism": "symmetric_random_corruption",
                        "rate": 0.10,
                        "seed": 404,
                        "representation_id": "imagenet_resnet18_highlighted",
                        "classifier_id": "multinomial_logistic_regression",
                    },
                    "selector_b": {
                        "mechanism": "symmetric_random_corruption",
                        "rate": 0.10,
                        "seed": 404,
                        "representation_id": "imagenet_resnet18_context",
                        "classifier_id": "multinomial_logistic_regression",
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
                    "selector": {
                        "mechanism": "symmetric_random_corruption",
                        "rate": 0.10,
                        "seed": 404,
                        "representation_id": "imagenet_resnet18_highlighted",
                        "classifier_id": "multinomial_logistic_regression",
                    },
                    "method_a": "self_confidence",
                    "method_b": "random_review",
                    "metric": "precision_at_budget",
                    "review_budget": 0.05,
                    "direction": "method_a_minus_method_b",
                    "holm_family": "primary_method_vs_random",
                }
            ],
            "exploratory_multiple_comparison_correction": "holm",
        },
        "restoration": {
            "enabled_cells": ["primary_0027_8531672acd3c"],
            "ranking_method": "self_confidence",
            "review_budget": 0.05,
            "random_repeats": 100,
            "random_seed": 433,
            "include_reference_validation_in_training": True,
            "required_experiments": [
                "uncorrupted_reference_baseline",
                "corrupted_observed_baseline",
                "random_review_restoration",
                "audit_guided_restoration",
            ],
            "downstream_comparisons": [
                {
                    "comparison_id": "audit_guided_minus_random_macro_f1",
                    "method_a": "audit_guided_restoration",
                    "method_b": "random_review_restoration",
                    "metric": "macro_f1",
                    "direction": "method_a_minus_method_b",
                }
            ],
        },
    }


def _available_confirmatory_cache_record(
    record_id: str,
    representation_id: str,
    encoder_identifier: str,
    input_variant: str,
    digest_character: str,
) -> dict[str, Any]:
    return {
        "id": record_id,
        "representation_id": representation_id,
        "status": "available",
        "cache_file_sha256": None,
        "sidecar_semantic_sha256": digest_character * 64,
        "sample_order_sha256": "e" * 64,
        "manifest_sha256": "f" * 64,
        "encoder_identifier": encoder_identifier,
        "encoder_metadata_sha256": digest_character * 64,
        "weight_identifier": "ResNet18_Weights.IMAGENET1K_V1",
        "weights_sha256": ("f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"),
        "preprocessing_identifier": "torchvision_resnet18_imagenet1k_v1_official",
        "preprocessing_sha256": "9" * 64,
        "input_variant": input_variant,
    }


def _confirmatory_comparison_operand(
    scenario_id: str,
    representation_id: str,
    classifier_id: str,
    risk_id: str,
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "representation_id": representation_id,
        "classifier_id": classifier_id,
        "risk_id": risk_id,
        "model_seed": "matched",
        "outer_fold": "all_matched",
        "corruption_cell": "all_matched",
    }


def complete_confirmatory_config() -> dict[str, Any]:
    """Return an exact confirmatory plan suitable for preregistration freezing."""

    return {
        "schema_version": 2,
        "experiment_name": "confirmatory_study",
        "status": "ready_for_freeze",
        "data": {
            "source": "pannuke",
            "analysis_manifest_authority": {
                "canonical_manifest_sha256": "f" * 64,
                "analysis_eligible_sample_order_sha256": "e" * 64,
                "analysis_eligible_sample_count": 100,
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
                "representation_id": "cnn_context_target_mask_pixels",
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
            _available_confirmatory_cache_record(
                "cnn_context_rgb_cache",
                "cnn_context_rgb_pixels",
                "resnet18_imagenet1k_v1",
                "context_rgb",
                "a",
            ),
            _available_confirmatory_cache_record(
                "cnn_context_target_mask_cache",
                "cnn_context_target_mask_pixels",
                "resnet18_imagenet1k_v1_zero_init_fourth_channel",
                "context_rgb_plus_binary_target_mask",
                "c",
            ),
            _available_confirmatory_cache_record(
                "imagenet_context_embedding_cache",
                "imagenet_resnet18_context_embeddings",
                "resnet18_imagenet1k_v1",
                "context_rgb",
                "d",
            ),
            _available_confirmatory_cache_record(
                "imagenet_target_highlighted_embedding_cache",
                "imagenet_target_highlighted_embeddings",
                "resnet18_imagenet1k_v1",
                "target_highlighted_rgb",
                "7",
            ),
            _available_confirmatory_cache_record(
                "imagenet_context_morphometrics_cache",
                "imagenet_context_embeddings_plus_target_morphometrics",
                "resnet18_imagenet1k_v1_plus_target_morphometrics_v1",
                "context_rgb_plus_target_morphometrics",
                "8",
            ),
            {
                "id": "pathology_context_embedding_cache",
                "representation_id": "pathology_context_embeddings",
                "status": "unavailable_with_frozen_blocker",
                "sample_order_sha256": "e" * 64,
                "manifest_sha256": "f" * 64,
                "encoder_identifier": "availability_selected_pathology_encoder",
                "input_variant": "context_rgb",
                "blocker_evidence_sha256": "b" * 64,
            },
        ],
        "model_seeds": [303, 304, 305],
        "training": {
            "optimizer": "adamw",
            "input_size": 224,
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "max_epochs": 100,
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 0.0001,
            "early_stopping_source": "reference_validation_only",
            "initial_batch_size": 128,
            "minimum_batch_size": 1,
            "gradient_accumulation_steps": 2,
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
                {"scenario_id": "cnn_context_rgb", "model_seed": 304},
                {"scenario_id": "cnn_context_rgb", "model_seed": 305},
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
            "scenario_id": "imagenet_frozen_logistic",
            "model_seed": 303,
            "representation_id": "imagenet_resnet18_context_embeddings",
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
            "preregistered_paired_comparisons": [
                {
                    "comparison_id": "target_mask_minus_context_rgb",
                    "metric": "average_precision",
                    "operand_a": _confirmatory_comparison_operand(
                        "cnn_context_target_mask",
                        "cnn_context_target_mask_pixels",
                        "cnn_softmax_head",
                        "self_confidence",
                    ),
                    "operand_b": _confirmatory_comparison_operand(
                        "cnn_context_rgb",
                        "cnn_context_rgb_pixels",
                        "cnn_softmax_head",
                        "self_confidence",
                    ),
                    "direction": "method_a_minus_method_b",
                    "holm_family": "confirmatory_ranking",
                },
                {
                    "comparison_id": "pathology_minus_imagenet",
                    "metric": "average_precision",
                    "operand_a": _confirmatory_comparison_operand(
                        "pathology_frozen_logistic",
                        "pathology_context_embeddings",
                        "multinomial_logistic_regression",
                        "self_confidence",
                    ),
                    "operand_b": _confirmatory_comparison_operand(
                        "imagenet_frozen_logistic",
                        "imagenet_resnet18_context_embeddings",
                        "multinomial_logistic_regression",
                        "self_confidence",
                    ),
                    "direction": "method_a_minus_method_b",
                    "holm_family": "confirmatory_ranking",
                },
                {
                    "comparison_id": "hybrid_minus_drop_ensemble",
                    "metric": "average_precision",
                    "operand_a": _confirmatory_comparison_operand(
                        "cnn_context_rgb",
                        "cnn_context_rgb_pixels",
                        "cnn_softmax_head",
                        "fixed_hybrid",
                    ),
                    "operand_b": _confirmatory_comparison_operand(
                        "cnn_context_rgb",
                        "cnn_context_rgb_pixels",
                        "cnn_softmax_head",
                        "hybrid_drop_ensemble_disagreement",
                    ),
                    "direction": "method_a_minus_method_b",
                    "holm_family": "confirmatory_ranking",
                },
                {
                    "comparison_id": "ensemble_minus_self_confidence",
                    "metric": "average_precision",
                    "operand_a": _confirmatory_comparison_operand(
                        "cnn_context_rgb",
                        "cnn_context_rgb_pixels",
                        "cnn_softmax_head",
                        "ensemble_disagreement",
                    ),
                    "operand_b": _confirmatory_comparison_operand(
                        "cnn_context_rgb",
                        "cnn_context_rgb_pixels",
                        "cnn_softmax_head",
                        "self_confidence",
                    ),
                    "direction": "method_a_minus_method_b",
                    "holm_family": "confirmatory_ranking",
                },
            ],
            "holm_families": ["confirmatory_ranking"],
        },
        "fold_rotation": {
            "enabled": True,
            "feasibility_rule": "all_five_classes_in_each_development_training_partition",
            "aggregate_policy": "report_each_rotation_and_descriptive_fold_mean",
        },
    }


def cross_compatible_confirmatory_config(
    primary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact confusion-targeted confirmatory cell selected from primary."""

    primary_config = complete_primary_config() if primary is None else primary
    confirmatory = complete_confirmatory_config()
    confirmatory["corruption"]["cells"][1] = {
        "id": "confusion_targeted_ten_percent",
        "mechanism": "confusion_targeted_corruption",
        "rate": 0.10,
        "seed": 404,
        "parameters": deepcopy(
            primary_config["corruption"]["mechanisms"]["confusion_targeted_corruption"]
        ),
    }
    return confirmatory


def test_primary_contract_expands_complete_deterministic_unique_matrix() -> None:
    config = complete_primary_config()
    first = build_primary_matrix_plan(config)
    second = build_primary_matrix_plan(deepcopy(config))

    assert first == second
    assert len(first.scenarios) == 37
    assert len(first.cells) == 222
    assert first.required_cell_count == 185
    assert first.optional_cell_count == 37
    assert len({cell.cell_id for cell in first.cells}) == len(first.cells)
    assert len({cell.scenario_id for cell in first.cells}) == len(first.scenarios)
    assert first.as_dict()["config_sha256"] == first.config_sha256


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda config: config.update(schema_version=1), "schema_version"),
        (lambda config: config["corruption"].update(rates=[0.1]), "rates"),
        (lambda config: config["corruption"].update(seeds=[404, 405]), "at least 3"),
        (
            lambda config: config["corruption"]["mechanisms"][
                "instance_dependent_corruption"
            ].update(independence_status="verified_independent"),
            "per_representation_matrix",
        ),
        (
            lambda config: config["statistics"].update(paired_group_bootstrap_iterations=1999),
            "at least 2000",
        ),
        (
            lambda config: config["data"]["analysis_manifest_authority"].update(
                analysis_eligible_sample_count=0
            ),
            "analysis_eligible_sample_count must be an integer >= 1",
        ),
        (
            lambda config: config["data"]["analysis_manifest_authority"].update(
                canonical_manifest_sha256="0" * 64
            ),
            "cache_provenance differs from data.analysis_manifest_authority",
        ),
    ],
)
def test_primary_contract_rejects_incomplete_scientific_choices(mutator: Any, message: str) -> None:
    config = complete_primary_config()
    mutator(config)

    with pytest.raises(StudyContractError, match=message):
        validate_frozen_primary_config(config)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda config: config.pop("calibration"), "missing=.*calibration"),
        (
            lambda config: config["calibration"].update(enabled=True),
            "enabled calibration cannot use method=none",
        ),
        (
            lambda config: config["calibration"].update(method="temperature_scaling"),
            "disabled calibration requires method=none",
        ),
        (
            lambda config: config["calibration"].update(source="final_test"),
            "reference_validation_only",
        ),
        (
            lambda config: config["calibration"].update(fit_labels_policy="pre_corruption_label"),
            "observed reference-validation labels only",
        ),
        (
            lambda config: config["classifiers"]["multinomial_logistic_regression"].update(
                class_weight_label_source="pre_corruption_label"
            ),
            "observed development labels only",
        ),
    ],
)
def test_primary_calibration_and_training_labels_are_exactly_frozen(
    mutator: Any,
    message: str,
) -> None:
    config = complete_primary_config()
    mutator(config)

    with pytest.raises(StudyContractError, match=message):
        validate_frozen_primary_config(config)


def test_primary_representation_provenance_and_independence_are_per_representation() -> None:
    invalid_cache = complete_primary_config()
    invalid_cache["representations"][1]["cache_provenance"]["cache_file_sha256"] = "invalid"
    with pytest.raises(StudyContractError, match="cache_file_sha256 must be a SHA-256"):
        validate_frozen_primary_config(invalid_cache)

    required_unavailable = complete_primary_config()
    provenance = required_unavailable["representations"][1]["cache_provenance"]
    provenance["status"] = "unavailable_optional"
    for field in (
        "encoder_implementation_sha256",
        "weights_sha256",
        "preprocessing_sha256",
        "cache_file_sha256",
    ):
        provenance[field] = None
    with pytest.raises(StudyContractError, match=r"required.*unavailable_optional"):
        validate_frozen_primary_config(required_unavailable)

    wrong_matrix = complete_primary_config()
    wrong_matrix["representations"][1]["generator_independence"]["independence_matrix_sha256"] = (
        "0" * 64
    )
    with pytest.raises(StudyContractError, match="matrix differs"):
        validate_frozen_primary_config(wrong_matrix)

    hidden_overlap = complete_primary_config()
    hidden_overlap["representations"][0]["generator_independence"]["status"] = (
        "verified_independent"
    )
    with pytest.raises(StudyContractError, match="circularity_risk"):
        validate_frozen_primary_config(hidden_overlap)

    fabricated_optional = complete_primary_config()
    fabricated_optional["representations"][-1]["cache_provenance"]["weights_sha256"] = "1" * 64
    with pytest.raises(StudyContractError, match="cannot claim unavailable artifact hashes"):
        validate_frozen_primary_config(fabricated_optional)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda config: config["statistics"].update(
                preregistered_comparisons=["broadcast_without_selector"]
            ),
            "statistics must contain exactly",
        ),
        (
            lambda config: config["statistics"]["within_cell_comparisons"][0].pop("selector"),
            "must contain exactly",
        ),
        (
            lambda config: config["statistics"]["cross_cell_comparisons"][0]["selector_b"].update(
                seed=405
            ),
            "must hold seed, representation, and classifier fixed",
        ),
        (
            lambda config: config["statistics"]["within_cell_comparisons"][0].update(
                metric="precision_at_budget"
            ),
            "metric is unsupported",
        ),
        (
            lambda config: config["statistics"]["method_vs_random_comparisons"][0].update(
                metric="precision_at_review_budget"
            ),
            "metric is unsupported",
        ),
        (
            lambda config: config["statistics"]["method_vs_random_comparisons"][0].update(
                comparison_id="hybrid_vs_self_confidence"
            ),
            "globally unique",
        ),
    ],
)
def test_primary_statistics_freeze_only_explicit_selector_scoped_comparisons(
    mutator: Any,
    message: str,
) -> None:
    config = complete_primary_config()
    mutator(config)

    with pytest.raises(StudyContractError, match=message):
        validate_frozen_primary_config(config)


def test_primary_downstream_macro_f1_comparison_is_separate_from_ranking_metrics() -> None:
    config = complete_primary_config()
    config["restoration"]["downstream_comparisons"][0]["metric"] = "average_precision"

    with pytest.raises(StudyContractError, match="audit-guided minus random macro_f1"):
        validate_frozen_primary_config(config)


def test_primary_execution_controls_bind_schema_v2_config_and_v1_plan_artifact() -> None:
    config = complete_primary_config()
    controls = primary_execution_controls_from_frozen_config(config)

    assert isinstance(controls, PrimaryExecutionControls)
    assert controls.frozen_config_schema_version == 2
    assert controls.plan.schema_version == 1
    assert controls.config_semantic_sha256 == config_sha256(config)
    assert controls.plan.config_sha256 == controls.config_semantic_sha256
    assert controls.plan_sha256 == canonical_sha256(controls.plan.as_dict())
    assert controls.statistical_group_unit == config["data"]["group_unit"]
    assert controls.development_official_folds == tuple(
        config["data"]["development_official_folds"]
    )
    assert controls.final_test_fold == config["data"]["final_test_fold"]
    assert (
        controls.reference_validation_fraction_groups
        == config["data"]["reference_validation_fraction_groups"]
    )
    assert controls.fold_assignment_label_source == "pre_corruption_label"
    assert controls.corruption_rounding_policy == "round_half_up"
    assert controls.audit_methods == (
        "self_confidence",
        "negative_log_likelihood",
        "prediction_margin",
        "predictive_entropy",
        "cleanlab",
        "nearest_neighbour_disagreement",
        "fixed_hybrid",
    )
    assert controls.fixed_hybrid_weights == (0.5, 0.5)
    assert controls.logistic_class_weight_label_source == "observed_development_labels_only"
    assert controls.logistic_fit_label_source == "observed_development_labels_only"
    assert controls.mlp_fit_label_source == "observed_development_labels_only"
    assert controls.calibration.as_dict() == config["calibration"]
    assert (controls.primary_review_budget, *controls.secondary_review_budgets) == (
        0.05,
        0.01,
        0.10,
        0.20,
    )
    assert controls.restoration_cell_ids == ("primary_0027_8531672acd3c",)
    planned = {cell.cell_id: cell for cell in controls.plan.cells}
    assert all(
        planned[cell_id].representation_id == "imagenet_resnet18_highlighted"
        and planned[cell_id].classifier_id == "multinomial_logistic_regression"
        for cell_id in controls.restoration_cell_ids
    )
    assert [
        comparison.as_dict() for comparison in controls.restoration_downstream_comparisons
    ] == config["restoration"]["downstream_comparisons"]
    controls.validate_for_plan(controls.plan)


def test_primary_execution_controls_preserve_exact_selector_scoped_comparisons() -> None:
    config = complete_primary_config()

    controls = primary_execution_controls_from_frozen_config(config)

    assert controls.paired_method_comparisons == ()
    assert [item.comparison_id for item in controls.within_cell_comparisons] == [
        "hybrid_vs_self_confidence"
    ]
    assert [item.comparison_id for item in controls.cross_cell_comparisons] == [
        "highlighted_vs_context"
    ]
    assert [item.comparison_id for item in controls.method_vs_random_comparisons] == [
        "self_confidence_vs_random"
    ]
    assert (
        controls.within_cell_comparisons[0].selector.as_dict()
        == config["statistics"]["within_cell_comparisons"][0]["selector"]
    )
    assert (
        controls.cross_cell_comparisons[0].selector_a.as_dict()
        == config["statistics"]["cross_cell_comparisons"][0]["selector_a"]
    )
    assert controls.method_vs_random_comparisons[0].method_b == "random_review"
    assert controls.holm_families == (
        "primary_within_cell",
        "primary_cross_cell",
        "primary_method_vs_random",
    )
    saved = controls.as_dict()
    assert saved["within_cell_comparisons"] == config["statistics"]["within_cell_comparisons"]
    assert saved["cross_cell_comparisons"] == config["statistics"]["cross_cell_comparisons"]
    assert (
        saved["method_vs_random_comparisons"]
        == config["statistics"]["method_vs_random_comparisons"]
    )


def test_primary_execution_controls_fail_closed_on_plan_binding_and_control_tamper() -> None:
    controls = primary_execution_controls_from_frozen_config(complete_primary_config())

    with pytest.raises(ValueError, match="binding SHA"):
        replace(controls, neighbour_k=controls.neighbour_k + 1).validate_for_plan(controls.plan)
    rehashed_tamper = replace(controls, neighbour_k=controls.neighbour_k + 1, binding_sha256="")
    rehashed_tamper = replace(
        rehashed_tamper,
        binding_sha256=canonical_sha256(rehashed_tamper._binding_payload()),
    )
    with pytest.raises(ValueError, match="exact derivation"):
        rehashed_tamper.validate_for_plan(controls.plan)
    tampered_plan = replace(controls.plan, config_sha256="f" * 64)
    with pytest.raises(ValueError, match="different primary plan"):
        controls.validate_for_plan(tampered_plan)
    with pytest.raises(ValueError, match="plan SHA"):
        replace(controls, plan_sha256="0" * 64, binding_sha256="0" * 64).validate_for_plan(
            controls.plan
        )


def test_primary_execution_api_exposes_no_loose_scientific_overrides() -> None:
    parameters = signature(execute_primary_matrix).parameters

    assert tuple(parameters) == (
        "inputs",
        "plan",
        "output_directory",
        "execution_controls",
    )
    for forbidden in (
        "n_splits",
        "model_seed",
        "review_budget",
        "bootstrap_iterations",
        "restoration_cell_id",
        "fixed_hybrid_weights",
    ):
        assert forbidden not in parameters


def test_primary_execution_controls_reject_unresolved_restoration_selector() -> None:
    config = complete_primary_config()
    config["restoration"]["enabled_cells"] = ["missing_representation_logistic"]

    with pytest.raises(StudyContractError, match="names a cell absent from the matrix"):
        primary_execution_controls_from_frozen_config(config)


def test_instance_independence_is_exact_per_auditor_and_shared_assignment_is_stable() -> None:
    controls = primary_execution_controls_from_frozen_config(complete_primary_config())
    generator = np.column_stack(
        (np.arange(10, dtype=np.float64), np.arange(10, dtype=np.float64) ** 2)
    )
    engineered = np.column_stack((generator[:, 0], np.ones(10, dtype=np.float64)))
    imagenet = np.column_stack((generator[:, 1], -generator[:, 0]))
    fitted_hash = "d" * 64
    generator_evidence = FeatureSpaceEvidence.from_array(
        generator,
        representation_name=controls.instance_generator_representation,
        family="morphology_generator",
        implementation_hash=semantic_sha256("test generator"),
        weights_hash=semantic_sha256("no weights"),
        preprocessing_hash=semantic_sha256("generator preprocessing"),
        fitted_data_hash=fitted_hash,
    )

    def evidence_for(
        representation_id: str,
        features: np.ndarray[Any, Any],
        *,
        decision: str,
    ) -> FeatureIndependenceEvidence:
        auditor = FeatureSpaceEvidence.from_array(
            features,
            representation_name=representation_id,
            family=f"auditor_{representation_id}",
            implementation_hash=semantic_sha256(f"implementation {representation_id}"),
            weights_hash=semantic_sha256(f"weights {representation_id}"),
            preprocessing_hash=semantic_sha256(f"preprocessing {representation_id}"),
            fitted_data_hash=fitted_hash,
        )
        return FeatureIndependenceEvidence.create(
            matrix_version="test_matrix_v1",
            matrix_decision=decision,
            matrix_reason=f"frozen test decision: {decision}",
            generator=generator_evidence,
            auditor=auditor,
        )

    engineered_id = "engineered_target_features"
    imagenet_id = "imagenet_resnet18_context"
    evidence = {
        engineered_id: evidence_for(engineered_id, engineered, decision="not_independent"),
        imagenet_id: evidence_for(imagenet_id, imagenet, decision="verified_independent"),
    }
    inputs = PrimaryMatrixInputs(
        audit_sample_ids=tuple(f"audit_{index}" for index in range(10)),
        audit_group_ids=tuple(f"audit_group_{index}" for index in range(10)),
        audit_pre_corruption_labels=np.asarray([0, 1, 2, 3, 4] * 2, dtype=np.int64),
        audit_features={engineered_id: engineered, imagenet_id: imagenet},
        reference_validation_sample_ids=tuple(f"ref_{index}" for index in range(5)),
        reference_validation_group_ids=tuple(f"ref_group_{index}" for index in range(5)),
        reference_validation_labels=np.arange(5, dtype=np.int64),
        reference_validation_features={
            engineered_id: engineered[:5],
            imagenet_id: imagenet[:5],
        },
        final_test_sample_ids=tuple(f"final_{index}" for index in range(5)),
        final_test_group_ids=tuple(f"final_group_{index}" for index in range(5)),
        final_test_labels=np.arange(5, dtype=np.int64),
        final_test_features={engineered_id: engineered[5:], imagenet_id: imagenet[5:]},
        corruption_generator_features=generator,
        corruption_generator_representation=controls.instance_generator_representation,
        corruption_auditor_representation=None,
        independence_evidence=None,
        dataset_seed=7,
        independence_evidence_by_representation=evidence,
        independence_matrix_artifact_sha256_by_representation={
            engineered_id: controls.instance_independence_matrix_sha256,
            imagenet_id: controls.instance_independence_matrix_sha256,
        },
        corruption_auditor_family_by_representation={
            engineered_id: "engineered",
            imagenet_id: "imagenet",
        },
    )
    inputs.validate()
    scenario = next(
        item
        for item in controls.plan.scenarios
        if item.mechanism == "instance_dependent_corruption"
    )
    circular, circular_evidence = primary_core_module._apply_cell_corruption(
        inputs,
        controls,
        scenario,
        upstream_manifest_hash="e" * 64,
        representation_id=engineered_id,
    )
    verified, verified_evidence = primary_core_module._apply_cell_corruption(
        inputs,
        controls,
        scenario,
        upstream_manifest_hash="e" * 64,
        representation_id=imagenet_id,
    )

    np.testing.assert_array_equal(circular.observed_labels, verified.observed_labels)
    assert circular.independence_status == "circularity_risk"
    assert circular.circularity_risk is True
    assert verified.independence_status == "verified_independent"
    assert verified.circularity_risk is False
    assert circular_evidence.evidence_sha256 != verified_evidence.evidence_sha256
    assert primary_core_module._shared_corruption_sha256(
        scenario, circular
    ) == primary_core_module._shared_corruption_sha256(scenario, verified)

    missing_inputs = replace(
        inputs,
        independence_evidence_by_representation={engineered_id: evidence[engineered_id]},
    )
    unverified, missing_evidence = primary_core_module._apply_cell_corruption(
        missing_inputs,
        controls,
        scenario,
        upstream_manifest_hash="e" * 64,
        representation_id=imagenet_id,
    )
    assert unverified.independence_status == "unverified"
    assert unverified.circularity_risk is True
    assert missing_evidence.evidence_sha256 is None


def test_confirmatory_contract_rejects_outcome_dependent_selection() -> None:
    config = complete_confirmatory_config()
    config["selection_note"] = "selected_after_primary"

    with pytest.raises(StudyContractError, match="outcome-dependent"):
        validate_frozen_confirmatory_config(config)


def test_primary_confirmatory_cross_contract_accepts_exact_shared_authority() -> None:
    primary, confirmatory = validate_primary_confirmatory_cross_config(
        complete_primary_config(),
        cross_compatible_confirmatory_config(),
    )

    assert (
        primary["data"]["analysis_manifest_authority"]
        == confirmatory["data"]["analysis_manifest_authority"]
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "canonical_manifest_sha256",
            "0" * 64,
            "differs from confirmatory.data.analysis_manifest_authority",
        ),
        (
            "analysis_eligible_sample_order_sha256",
            "0" * 64,
            "differs from confirmatory.data.analysis_manifest_authority",
        ),
        (
            "analysis_eligible_sample_count",
            101,
            "analysis_manifest_authority bindings differ",
        ),
    ],
)
def test_primary_confirmatory_cross_contract_rejects_authority_tampering(
    field: str,
    value: str | int,
    message: str,
) -> None:
    confirmatory = cross_compatible_confirmatory_config()
    confirmatory["data"]["analysis_manifest_authority"][field] = value

    with pytest.raises(StudyContractError, match=message):
        validate_primary_confirmatory_cross_config(
            complete_primary_config(),
            confirmatory,
        )


def test_confirmatory_contract_expands_every_rotation_scenario_and_seed() -> None:
    config = complete_confirmatory_config()
    plan = build_confirmatory_matrix_plan(config)

    assert len(plan.cells) == 108
    assert plan.required_cell_count == 90
    assert plan.optional_cell_count == 18
    assert {cell.outer_fold for cell in plan.cells} == {1, 2, 3}
    assert {cell.model_seed for cell in plan.cells} == {303, 304, 305}
    assert len({cell.cell_id for cell in plan.cells}) == 108
    assert plan.schema_version == 2


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda config: config["scenarios"].pop(3),
            "imagenet_frozen_target_highlighted_logistic",
        ),
        (
            lambda config: config["scenarios"][3].update(required=False),
            "must be required",
        ),
        (
            lambda config: config["scenarios"][4].update(
                representation_id="wrong_morphometric_concat"
            ),
            "must freeze representation_id",
        ),
        (
            lambda config: config["scenarios"][4].update(
                cache_provenance_id="imagenet_target_highlighted_embedding_cache"
            ),
            "must freeze cache_provenance_id",
        ),
    ],
)
def test_confirmatory_contract_requires_complete_target_representation_ablation(
    mutator: Any,
    message: str,
) -> None:
    config = complete_confirmatory_config()
    mutator(config)

    with pytest.raises(StudyContractError, match=message):
        validate_frozen_confirmatory_config(config)


def test_confirmatory_contract_requires_rgb_target_mask_and_cuda_controls() -> None:
    missing_target = complete_confirmatory_config()
    missing_target["scenarios"] = [missing_target["scenarios"][0]]
    with pytest.raises(StudyContractError, match=r"RGB\+target-mask"):
        validate_frozen_confirmatory_config(missing_target)

    cpu_fallback = complete_confirmatory_config()
    cpu_fallback["training"]["cuda_required"] = False
    with pytest.raises(StudyContractError, match="cuda_required"):
        validate_frozen_confirmatory_config(cpu_fallback)


def test_confirmatory_matrix_hash_changes_when_predeclared_scenario_changes() -> None:
    baseline = build_confirmatory_matrix_plan(complete_confirmatory_config())
    modified_config = complete_confirmatory_config()
    modified_config["training"]["learning_rate"] = 0.0002
    modified = build_confirmatory_matrix_plan(modified_config)

    assert modified.config_sha256 != baseline.config_sha256
    assert modified.cells == baseline.cells


def _confirmatory_transition_matrix() -> list[list[float]]:
    return [
        [0.0, 0.25, 0.25, 0.25, 0.25],
        [0.25, 0.0, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.0, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.0, 0.25],
        [0.25, 0.25, 0.25, 0.25, 0.0],
    ]


@pytest.mark.parametrize(
    ("mechanism", "parameters"),
    [
        (
            "confusion_targeted_corruption",
            {"transition_matrix": _confirmatory_transition_matrix()},
        ),
        (
            "group_conditional_corruption",
            {
                "grouping_field": "tissue_type",
                "weights_by_value": {"tissue_a": 1.0, "tissue_b": 0.5},
                "default_weight": 0.25,
            },
        ),
        (
            "instance_dependent_corruption",
            {
                "generator_representation": "morphology_only_v1",
                "auditor_representation_families": ["cnn", "imagenet_frozen"],
                "independence_status": "verified_independent",
                "independence_matrix_path": "reports/confirmatory_independence.json",
                "independence_matrix_sha256": "c" * 64,
            },
        ),
    ],
)
def test_confirmatory_contract_accepts_explicit_non_symmetric_corruption_parameters(
    mechanism: str,
    parameters: dict[str, Any],
) -> None:
    config = complete_confirmatory_config()
    config["corruption"]["cells"][1].update(
        mechanism=mechanism,
        parameters=parameters,
    )

    validate_frozen_confirmatory_config(config)


@pytest.mark.parametrize(
    ("mechanism", "parameters", "message"),
    [
        ("symmetric_random_corruption", {"unexpected": True}, "exactly empty"),
        ("confusion_targeted_corruption", {}, "transition_matrix"),
        (
            "group_conditional_corruption",
            {"grouping_field": "tissue_type"},
            "group-conditional",
        ),
        (
            "instance_dependent_corruption",
            {
                "generator_representation": "morphology_only_v1",
                "auditor_representation_families": ["cnn"],
                "independence_status": "verified_independent",
                "independence_matrix_path": "reports/confirmatory_independence.json",
            },
            "instance-dependent",
        ),
    ],
)
def test_confirmatory_contract_fails_closed_on_incomplete_corruption_parameters(
    mechanism: str,
    parameters: dict[str, Any],
    message: str,
) -> None:
    config = complete_confirmatory_config()
    config["corruption"]["cells"][1].update(
        mechanism=mechanism,
        parameters=parameters,
    )

    with pytest.raises(StudyContractError, match=message):
        validate_frozen_confirmatory_config(config)


@pytest.mark.parametrize(
    "field",
    ["input_size", "gradient_accumulation_steps", "class_weight"],
)
def test_confirmatory_contract_requires_all_cnn_execution_controls(field: str) -> None:
    config = complete_confirmatory_config()
    del config["training"][field]

    with pytest.raises(StudyContractError, match="lacks required fields"):
        validate_frozen_confirmatory_config(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("input_size", 31, "input_size"),
        ("gradient_accumulation_steps", 0, "gradient_accumulation_steps"),
        ("class_weight", "none", "class_weight"),
    ],
)
def test_confirmatory_contract_rejects_invalid_cnn_execution_controls(
    field: str,
    value: Any,
    message: str,
) -> None:
    config = complete_confirmatory_config()
    config["training"][field] = value

    with pytest.raises(StudyContractError, match=message):
        validate_frozen_confirmatory_config(config)


def test_confirmatory_cache_provenance_requires_exact_artifact_or_recipe_binding() -> None:
    both = complete_confirmatory_config()
    both["cache_provenance"][0]["cache_file_sha256"] = "1" * 64
    with pytest.raises(StudyContractError, match="exactly one"):
        validate_frozen_confirmatory_config(both)

    neither = complete_confirmatory_config()
    neither["cache_provenance"][0]["sidecar_semantic_sha256"] = None
    with pytest.raises(StudyContractError, match="exactly one"):
        validate_frozen_confirmatory_config(neither)

    invalid_sample_order = complete_confirmatory_config()
    invalid_sample_order["cache_provenance"][0]["sample_order_sha256"] = "invalid"
    with pytest.raises(StudyContractError, match="sample_order_sha256 must be a SHA-256"):
        validate_frozen_confirmatory_config(invalid_sample_order)


def test_confirmatory_scenarios_bind_exactly_one_eligible_cache_record() -> None:
    missing = complete_confirmatory_config()
    missing["scenarios"][0]["cache_provenance_id"] = "missing_cache_record"
    with pytest.raises(StudyContractError, match="references missing"):
        validate_frozen_confirmatory_config(missing)

    required_unavailable = complete_confirmatory_config()
    required_unavailable["cache_provenance"][0] = {
        "id": "cnn_context_rgb_cache",
        "representation_id": "cnn_context_rgb_pixels",
        "status": "unavailable_with_frozen_blocker",
        "sample_order_sha256": "e" * 64,
        "manifest_sha256": "f" * 64,
        "encoder_identifier": "resnet18_imagenet1k_v1",
        "input_variant": "context_rgb",
        "blocker_evidence_sha256": "a" * 64,
    }
    with pytest.raises(StudyContractError, match=r"required scenario.*lacks available"):
        validate_frozen_confirmatory_config(required_unavailable)

    wrong_optional_blocker = complete_confirmatory_config()
    wrong_optional_blocker["cache_provenance"][-1]["blocker_evidence_sha256"] = "c" * 64
    with pytest.raises(StudyContractError, match="blocker differs"):
        validate_frozen_confirmatory_config(wrong_optional_blocker)

    unreferenced = complete_confirmatory_config()
    extra_record = deepcopy(unreferenced["cache_provenance"][0])
    extra_record.update(
        id="unreferenced_cache",
        representation_id="unreferenced_representation",
    )
    unreferenced["cache_provenance"].append(extra_record)
    with pytest.raises(StudyContractError, match="exactly one referenced record"):
        validate_frozen_confirmatory_config(unreferenced)


def test_confirmatory_ensemble_members_are_exact_unique_and_required() -> None:
    unknown = complete_confirmatory_config()
    unknown["ensemble"]["members"][0]["scenario_id"] = "unknown_scenario"
    with pytest.raises(StudyContractError, match="not frozen"):
        validate_frozen_confirmatory_config(unknown)

    duplicate = complete_confirmatory_config()
    duplicate["ensemble"]["members"][1] = deepcopy(duplicate["ensemble"]["members"][0])
    with pytest.raises(StudyContractError, match="members must be unique"):
        validate_frozen_confirmatory_config(duplicate)

    optional = complete_confirmatory_config()
    optional["ensemble"]["members"][0]["scenario_id"] = "pathology_frozen_logistic"
    with pytest.raises(StudyContractError, match="required scenarios"):
        validate_frozen_confirmatory_config(optional)


def test_confirmatory_ensemble_risks_are_explicit_supported_and_non_overlapping() -> None:
    repeated = complete_confirmatory_config()
    repeated["ensemble"]["secondary_risks"].append("mean_pairwise_js_divergence")
    with pytest.raises(StudyContractError, match="must not be repeated"):
        validate_frozen_confirmatory_config(repeated)

    unsupported = complete_confirmatory_config()
    unsupported["ensemble"]["secondary_risks"] = ["outlier_score"]
    with pytest.raises(StudyContractError, match="unsupported risks"):
        validate_frozen_confirmatory_config(unsupported)


def test_confirmatory_restoration_is_bound_to_required_scenario_seed_and_representation() -> None:
    wrong_representation = complete_confirmatory_config()
    wrong_representation["restoration"]["representation_id"] = "cnn_context_rgb_pixels"
    with pytest.raises(StudyContractError, match="exactly match"):
        validate_frozen_confirmatory_config(wrong_representation)

    wrong_seed = complete_confirmatory_config()
    wrong_seed["restoration"]["model_seed"] = 999
    with pytest.raises(StudyContractError, match="frozen model seed"):
        validate_frozen_confirmatory_config(wrong_seed)

    optional = complete_confirmatory_config()
    optional["restoration"].update(
        scenario_id="pathology_frozen_logistic",
        representation_id="pathology_context_embeddings",
    )
    with pytest.raises(StudyContractError, match="optional scenario"):
        validate_frozen_confirmatory_config(optional)

    missing_random_seed = complete_confirmatory_config()
    missing_random_seed["restoration"].pop("random_seed")
    with pytest.raises(StudyContractError, match="random_seed"):
        validate_frozen_confirmatory_config(missing_random_seed)

    negative_random_seed = complete_confirmatory_config()
    negative_random_seed["restoration"]["random_seed"] = -1
    with pytest.raises(StudyContractError, match=r"integer >= 0"):
        validate_frozen_confirmatory_config(negative_random_seed)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda config: config.pop("original_audit_selection"),
            "missing=.*original_audit_selection",
        ),
        (
            lambda config: config["original_audit_selection"].update(scenario_id="cnn_context_rgb"),
            "frozen-feature scenario",
        ),
        (
            lambda config: config["original_audit_selection"].update(
                representation_id="cnn_context_rgb_pixels"
            ),
            "representation_id must match",
        ),
        (
            lambda config: config["original_audit_selection"].update(model_seed=999),
            "frozen model seed",
        ),
        (
            lambda config: config["original_audit_selection"].update(
                risk_method="selected_after_primary"
            ),
            "outcome-dependent",
        ),
        (
            lambda config: config["original_audit_selection"].update(n_splits=1),
            "n_splits",
        ),
        (
            lambda config: config["original_audit_selection"]["classifier"].update(
                id="cnn_softmax_head"
            ),
            "classifier must match",
        ),
        (
            lambda config: config["original_audit_selection"]["classifier"]["parameters"].pop("l2"),
            "logistic parameters",
        ),
        (
            lambda config: config["original_audit_selection"].update(
                cache_provenance_id="cnn_context_rgb_cache"
            ),
            "cache_provenance_id must match",
        ),
        (
            lambda config: config["cache_provenance"][2].update(
                encoder_identifier="different_encoder"
            ),
            "encoder differs",
        ),
        (
            lambda config: config["cache_provenance"][2].update(
                input_variant="target_highlighted_rgb"
            ),
            "input_variant differs",
        ),
        (
            lambda config: config["cache_provenance"][2].update(weights_sha256="not-a-sha"),
            "weights_sha256 must be a SHA-256",
        ),
    ],
)
def test_original_audit_selection_is_fully_frozen_and_scenario_bound(
    mutator: Any,
    message: str,
) -> None:
    config = complete_confirmatory_config()
    mutator(config)

    with pytest.raises(StudyContractError, match=message):
        validate_frozen_confirmatory_config(config)


def test_confirmatory_paired_comparisons_freeze_direction_and_holm_family() -> None:
    ambiguous_methods = complete_confirmatory_config()
    ambiguous = ambiguous_methods["statistics"]["preregistered_paired_comparisons"][0]
    ambiguous.pop("operand_a")
    ambiguous.pop("operand_b")
    ambiguous.update(method_a="cnn_context_target_mask", method_b="cnn_context_rgb")
    with pytest.raises(StudyContractError, match="operand_a, operand_b"):
        validate_frozen_confirmatory_config(ambiguous_methods)

    wrong_representation = complete_confirmatory_config()
    wrong_representation["statistics"]["preregistered_paired_comparisons"][0]["operand_a"][
        "representation_id"
    ] = "cnn_context_rgb_pixels"
    with pytest.raises(StudyContractError, match="must match its frozen scenario"):
        validate_frozen_confirmatory_config(wrong_representation)

    unmatched_seed = complete_confirmatory_config()
    unmatched_seed["statistics"]["preregistered_paired_comparisons"][0]["operand_a"][
        "model_seed"
    ] = 303
    with pytest.raises(StudyContractError, match="model_seed must be matched"):
        validate_frozen_confirmatory_config(unmatched_seed)

    unmatched_fold = complete_confirmatory_config()
    unmatched_fold["statistics"]["preregistered_paired_comparisons"][0]["operand_a"][
        "outer_fold"
    ] = 1
    with pytest.raises(StudyContractError, match="matched outer-fold"):
        validate_frozen_confirmatory_config(unmatched_fold)

    wrong_direction = complete_confirmatory_config()
    wrong_direction["statistics"]["preregistered_paired_comparisons"][0]["direction"] = (
        "method_b_minus_method_a"
    )
    with pytest.raises(StudyContractError, match="method_a_minus_method_b"):
        validate_frozen_confirmatory_config(wrong_direction)

    unknown_family = complete_confirmatory_config()
    unknown_family["statistics"]["preregistered_paired_comparisons"][0]["holm_family"] = (
        "undeclared_family"
    )
    with pytest.raises(StudyContractError, match="absent"):
        validate_frozen_confirmatory_config(unknown_family)

    duplicate_id = complete_confirmatory_config()
    duplicate_id["statistics"]["preregistered_paired_comparisons"][1]["comparison_id"] = (
        duplicate_id["statistics"]["preregistered_paired_comparisons"][0]["comparison_id"]
    )
    with pytest.raises(StudyContractError, match="IDs must be unique"):
        validate_frozen_confirmatory_config(duplicate_id)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda config: config["ensemble"]["members"][0].update(
            scenario_id="cnn_context_target_mask"
        ),
        lambda config: config["restoration"].update(model_seed=304),
        lambda config: config["statistics"]["preregistered_paired_comparisons"][0].update(
            metric="macro_f1"
        ),
        lambda config: config["corruption"]["cells"][1].update(rate=0.20),
        lambda config: config["original_audit_selection"]["classifier"]["parameters"].update(
            l2=0.5
        ),
        lambda config: config["cache_provenance"][2].update(sidecar_semantic_sha256="8" * 64),
    ],
)
def test_confirmatory_plan_hash_binds_non_cell_execution_decisions(mutator: Any) -> None:
    baseline = build_confirmatory_matrix_plan(complete_confirmatory_config())
    changed_config = complete_confirmatory_config()
    mutator(changed_config)
    changed = build_confirmatory_matrix_plan(changed_config)

    assert changed.config_sha256 != baseline.config_sha256


def test_confirmatory_project_config_records_finalized_cache_provenance() -> None:
    config = yaml.safe_load(Path("configs/confirmatory.yaml").read_text(encoding="utf-8"))

    assert config["schema_version"] == 2
    assert config["status"] == "READY_FOR_FREEZE"
    validated = validate_frozen_confirmatory_config(config)
    assert (
        config_sha256(validated)
        == "ff2ce8d5043813b08db23efe797abe444ba6b6bde292810a094d78757f74460b"
    )
    authority = {
        "analysis_eligible_sample_count": 188333,
        "analysis_eligible_sample_order_sha256": (
            "2b95c283b0a76d6eada176a7cd72b7fd322f2663a1d87b929cb1559687da8d26"
        ),
        "canonical_manifest_sha256": (
            "7bf0ed664da19103c0f1119623789bc9be3f23189dabef3920bc8bd1f8c49d9e"
        ),
    }
    assert validated["data"]["analysis_manifest_authority"] == authority
    records = {record["id"]: record for record in validated["cache_provenance"]}
    sidecar_by_id = {
        "cnn_context_rgb_cache": (
            "cf74379bab82d41be0df6cf047f8d365c5beb8854d34e7e69e22b2be403756b9"
        ),
        "cnn_context_target_mask_cache": (
            "cf74379bab82d41be0df6cf047f8d365c5beb8854d34e7e69e22b2be403756b9"
        ),
        "imagenet_context_embedding_cache": (
            "195cac1a2073d9d7974e71fbc485637f661c9db485ff53db78721021689ae618"
        ),
        "imagenet_context_morphometrics_cache": (
            "531a730ba54f6d1d6bd6447f3f8b336cd8b36184a73e32e2aaa7edec5d4421ee"
        ),
        "imagenet_target_highlighted_embedding_cache": (
            "efefa0cc0194f3b571cd82cce19fe3e61e963ded63ba172dcf2abce9c226f81e"
        ),
    }
    assert set(records) == {*sidecar_by_id, "pathology_context_embedding_cache"}
    for identifier, sidecar_sha256 in sidecar_by_id.items():
        record = records[identifier]
        assert record["status"] == "available"
        assert record["cache_file_sha256"] is None
        assert record["sidecar_semantic_sha256"] == sidecar_sha256
        assert record["manifest_sha256"] == authority["canonical_manifest_sha256"]
        assert record["sample_order_sha256"] == authority["analysis_eligible_sample_order_sha256"]
    pathology = records["pathology_context_embedding_cache"]
    assert pathology["status"] == "unavailable_with_frozen_blocker"
    assert (
        pathology["blocker_evidence_sha256"]
        == "5e568cf29e489d8948bfcd33feae5b292cb48837eb4c93754202a565778a6e4a"
    )
    plan = build_confirmatory_matrix_plan(validated)
    assert len(plan.cells) == 108
    assert plan.required_cell_count == 90
    assert plan.optional_cell_count == 18
    assert (
        canonical_sha256(plan.as_dict())
        == "c1993d4403982814a7259c524bbd21784537b7634e49a4f7150a9ca4de3c2c87"
    )


def test_primary_project_config_records_finalized_cache_provenance() -> None:
    config = yaml.safe_load(Path("configs/primary.yaml").read_text(encoding="utf-8"))
    confirmatory = yaml.safe_load(Path("configs/confirmatory.yaml").read_text(encoding="utf-8"))

    assert config["schema_version"] == 2
    assert config["status"] == "READY_FOR_FREEZE"
    assert config["calibration"]["enabled"] is False
    validated = validate_frozen_primary_config(config)
    validated, _ = validate_primary_confirmatory_cross_config(validated, confirmatory)
    assert (
        config_sha256(validated)
        == "c9949769ed8ab28514925ed2574958146b319d4ff848423559e0568c308cba15"
    )
    authority = {
        "analysis_eligible_sample_count": 188333,
        "analysis_eligible_sample_order_sha256": (
            "2b95c283b0a76d6eada176a7cd72b7fd322f2663a1d87b929cb1559687da8d26"
        ),
        "canonical_manifest_sha256": (
            "7bf0ed664da19103c0f1119623789bc9be3f23189dabef3920bc8bd1f8c49d9e"
        ),
    }
    assert validated["data"]["analysis_manifest_authority"] == authority
    representations = {item["id"]: item for item in validated["representations"]}
    cache_sha256_by_id = {
        "engineered_target_features": (
            "a4abe0cbb3d8ba4afc02f52e045af97a256c8e69ae70455294fc17aa5b752d32"
        ),
        "imagenet_resnet18_context": (
            "04cd2e0315f0a3c4270473af27aa1ca7d2a8df78268423186b3ad1bd5552e68b"
        ),
        "imagenet_resnet18_highlighted": (
            "585b6ccace8130c911d7300e86ffe74e7b28f71fcd81fd2e3e62da5c1def9d79"
        ),
    }
    for identifier, cache_sha256 in cache_sha256_by_id.items():
        provenance = representations[identifier]["cache_provenance"]
        assert provenance["status"] == "available"
        assert provenance["cache_file_sha256"] == cache_sha256
        assert provenance["dataset_manifest_sha256"] == authority["canonical_manifest_sha256"]
        assert (
            provenance["sample_order_sha256"] == authority["analysis_eligible_sample_order_sha256"]
        )
    pathology = representations["pathology_encoder_optional"]["cache_provenance"]
    assert pathology["status"] == "unavailable_optional"
    for field in (
        "cache_file_sha256",
        "encoder_implementation_sha256",
        "preprocessing_sha256",
        "weights_sha256",
    ):
        assert pathology[field] is None
    plan = build_primary_matrix_plan(validated)
    assert len(plan.scenarios) == 37
    assert len(plan.cells) == 222
    assert plan.required_cell_count == 185
    assert plan.optional_cell_count == 37
    assert (
        canonical_sha256(plan.as_dict())
        == "12a98f9dd40480927d94d8f25901392b0eb755194a0d44aebdbdb2ded26dee7f"
    )
