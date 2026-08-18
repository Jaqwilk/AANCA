"""Filesystem-backed completion tests for the frozen confirmatory study."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from test_study_contracts import complete_confirmatory_config
from torch import nn
from torchvision.models import resnet18  # type: ignore[import-untyped]

from histo_audit.auditing.ensemble import ensemble_disagreement, predeclared_ensemble_risk
from histo_audit.auditing.scores import (
    fixed_hybrid_drop_one_ablations,
    score_annotations,
)
from histo_audit.corruption.controlled import array_artifact_sha256
from histo_audit.cross_validation.image_oof import (
    ConfirmatoryCheckpointDirective,
    ConfirmatoryCheckpointPhysicalIdentity,
)
from histo_audit.evaluation.restoration import classification_metrics, restore_reviewed_labels
from histo_audit.experiment.confirmatory_completion import (
    REAL_CONFIRMATORY_ARTIFACT_SCOPE,
    ConfirmatoryCellReadback,
    ConfirmatoryFilesystemReadback,
    ConfirmatoryMatrixReconciliation,
    build_confirmatory_completion_evidence,
    confirmatory_report_contract_block,
    read_confirmatory_run_directory,
    reconcile_confirmatory_cell_outcomes,
)
from histo_audit.experiment.confirmatory_core import (
    ConfirmatoryExecutionControls,
    confirmatory_execution_controls_from_frozen_config,
)
from histo_audit.experiment.study_contracts import ConfirmatoryCell, ConfirmatoryMatrixPlan
from histo_audit.statistics.review import average_precision, budget_count, rank_indices
from histo_audit.utils.run_tracking import sha256_file
from histo_audit.workflows.study_gates import (
    ConfirmatoryExecutionGateEvidence,
    PrimaryExecutionGateEvidence,
)

_COMPLETED_ARTIFACTS = (
    "cell_identity.json",
    "oof_evidence.npz",
    "checkpoint_manifest.json",
    "telemetry.json",
    "risk_scores.npz",
    "ranking.csv",
    "metrics.json",
)
_ROOT_ARTIFACTS = (
    "confirmatory_input_bindings.json",
    "matrix_plan.json",
    "execution_controls.json",
    "frozen_feature_provenance.json",
    "cell_index.csv",
    "reconciliation.json",
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
_INDEX_FIELDS = (
    "cell_id",
    "status",
    "outer_fold",
    "corruption_cell_id",
    "corruption_mechanism",
    "corruption_rate",
    "corruption_seed",
    "scenario_id",
    "scenario_family",
    "representation_id",
    "cache_provenance_id",
    "model_seed",
    "required",
    "artifact_manifest_sha256",
    "metrics_sha256",
    "frozen_unavailability",
    "blocker",
)


@dataclass(frozen=True, slots=True)
class _ConfirmatoryTree:
    config: dict[str, Any]
    controls: ConfirmatoryExecutionControls
    plan: ConfirmatoryMatrixPlan
    run_directory: Path
    frozen_config_path: Path
    gate: ConfirmatoryExecutionGateEvidence
    reconciliation: ConfirmatoryMatrixReconciliation


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )


def _write_canonical_ascii_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_file_identity(path: Path) -> dict[str, Any]:
    physical = ConfirmatoryCheckpointPhysicalIdentity.from_stat(path.lstat())
    return {
        "path": str(path.resolve()),
        "file_id_128": physical.file_id_128,
        **physical.as_dict(),
        "sha256": sha256_file(path),
    }


@lru_cache(maxsize=2)
def _compact_resnet_state(input_channels: int) -> dict[str, torch.Tensor]:
    network = resnet18(weights=None)
    network.fc = nn.Linear(network.fc.in_features, 5)
    if input_channels == 4:
        original = network.conv1
        network.conv1 = nn.Conv2d(
            4,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=original.bias is not None,
        )
    compact: dict[str, torch.Tensor] = {}
    for key, tensor in network.state_dict().items():
        scalar = (
            torch.ones((), dtype=tensor.dtype)
            if key == "conv1.weight"
            else torch.zeros((), dtype=tensor.dtype)
        )
        compact[key] = scalar.expand(tensor.shape)
    return compact


@lru_cache(maxsize=2)
def _compact_adamw_moments(input_channels: int) -> dict[int, dict[str, torch.Tensor]]:
    network = resnet18(weights=None)
    network.fc = nn.Linear(network.fc.in_features, 5)
    if input_channels == 4:
        original = network.conv1
        network.conv1 = nn.Conv2d(
            4,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=original.bias is not None,
        )
    return {
        index: {
            "step": torch.tensor(1.0, dtype=torch.float32),
            "exp_avg": torch.zeros((), dtype=parameter.dtype).expand(parameter.shape),
            "exp_avg_sq": torch.zeros((), dtype=parameter.dtype).expand(parameter.shape),
        }
        for index, parameter in enumerate(network.parameters())
    }


def _write_real_checkpoint_fixture(
    path: Path,
    *,
    configuration: dict[str, Any],
    model_metadata: dict[str, Any],
    data_and_split_sha256: dict[str, str],
) -> None:
    resume_configuration = dict(configuration)
    resume_configuration.pop("epochs")
    input_channels = int(model_metadata["input_channels"])
    state = _compact_resnet_state(input_channels)
    optimiser_state = _compact_adamw_moments(input_channels)
    parameter_ids = list(range(len(optimiser_state)))
    numpy_state = np.random.RandomState(int(configuration["seed"])).get_state()
    torch.save(
        {
            "schema_version": 1,
            "model_kind": "confirmatory_resnet18_five_class",
            "execution_mode": "real_study_cuda",
            "study_outcome_eligible": True,
            "configuration": configuration,
            "configuration_sha256": _canonical_sha256(configuration),
            "resume_contract_sha256": _canonical_sha256(resume_configuration),
            "data_and_split_sha256": data_and_split_sha256,
            "model_metadata": model_metadata,
            "class_order": torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),
            "completed_epochs": 1,
            "network_state_dict": state,
            "optimiser_state_dict": {
                "state": optimiser_state,
                "param_groups": [
                    {
                        "lr": configuration["learning_rate"],
                        "betas": (0.9, 0.999),
                        "eps": 1e-8,
                        "weight_decay": configuration["weight_decay"],
                        "amsgrad": False,
                        "maximize": False,
                        "foreach": None,
                        "capturable": False,
                        "differentiable": False,
                        "fused": None,
                        "decoupled_weight_decay": True,
                        "params": parameter_ids,
                    }
                ],
            },
            "scaler_state_dict": {
                "scale": 65536.0,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 2000,
                "_growth_tracker": 1,
            },
            "history": [
                {
                    "epoch": 1,
                    "training_loss": 1.2,
                    "reference_validation_loss": 1.0,
                    "effective_batch_size": int(configuration["batch_size"]),
                    "optimiser_steps": 1,
                    "successful_optimiser_steps": 1,
                    "skipped_optimiser_steps": 0,
                    "early_stopping_improved": True,
                    "epochs_without_improvement": 0,
                    "stopped_early": False,
                }
            ],
            "effective_batch_size": int(configuration["batch_size"]),
            "rng_state": {
                "python": random.Random(int(configuration["seed"])).getstate(),
                "numpy": {
                    "bit_generator": str(numpy_state[0]),
                    "keys": torch.from_numpy(numpy_state[1].copy()),
                    "position": int(numpy_state[2]),
                    "has_gauss": int(numpy_state[3]),
                    "cached_gaussian": float(numpy_state[4]),
                },
                "torch_cpu": torch.arange(32, dtype=torch.uint8),
                "torch_cuda": [torch.arange(32, dtype=torch.uint8)],
            },
            "early_stopping_state": {
                "best_epoch": 1,
                "best_validation_loss": 1.0,
                "epochs_without_improvement": 0,
                "stopped_early": False,
                "best_network_state_dict": state,
            },
            "telemetry": {
                "schema_version": 1,
                "execution_mode": "real_study_cuda",
                "study_outcome_eligible": True,
                "device": "cuda:0",
                "amp_enabled": True,
                "amp_dtype": configuration["amp_dtype"],
                "grad_scaler_enabled": True,
                "gradient_accumulation_steps": configuration["gradient_accumulation_steps"],
                "initial_batch_size": configuration["batch_size"],
                "effective_batch_size": configuration["batch_size"],
                "minimum_batch_size": configuration["minimum_batch_size"],
                "batch_backoff_events": [],
                "current_fit_runtime_seconds": 1.0,
                "cumulative_runtime_seconds": 1.0,
                "cuda_peak_memory_allocated_bytes": 1,
                "cuda_peak_memory_reserved_bytes": 1,
                "completed_epochs": 1,
                "requested_epochs": configuration["epochs"],
                "successful_samples_processed": 10,
                "optimiser_steps": 1,
                "successful_optimiser_steps": 1,
                "skipped_optimiser_steps": 0,
                "early_stopping_source": "reference_validation_only",
                "best_epoch": 1,
                "best_validation_loss": 1.0,
                "epochs_without_improvement": 0,
                "stopped_early": False,
                "failure": None,
            },
        },
        path,
    )


def _write_hash_manifest(
    directory: Path,
    relative_paths: tuple[str, ...],
    *,
    filename: str = "artifact_manifest.json",
) -> Path:
    path = directory / filename
    _write_json(
        path,
        {relative: sha256_file(directory / relative) for relative in relative_paths},
    )
    return path


def _write_cell_index(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _read_cell_index(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _cell_identity(
    controls: ConfirmatoryExecutionControls,
    cell: ConfirmatoryCell,
) -> dict[str, Any]:
    corruption = controls.corruptions_by_id[cell.corruption_cell_id]
    scenario = controls.scenarios_by_id[cell.scenario_id]
    return {
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


def _write_completed_cell(
    directory: Path,
    *,
    controls: ConfirmatoryExecutionControls,
    cell: ConfirmatoryCell,
    identity: dict[str, Any],
) -> tuple[str, str]:
    directory.mkdir(parents=True)
    _write_json(directory / "cell_identity.json", identity)

    sample_ids = np.asarray([f"sample_{index:02d}" for index in range(10)], dtype=np.str_)
    group_ids = np.asarray([f"group_{index:02d}" for index in range(10)], dtype=np.str_)
    pre_labels = np.tile(np.arange(5, dtype=np.int64), 2)
    observed_labels = pre_labels.copy()
    corruption = controls.corruptions_by_id[cell.corruption_cell_id]
    injected_count = math.floor(len(sample_ids) * corruption.rate + 0.5)
    if injected_count:
        observed_labels[:injected_count] = (observed_labels[:injected_count] + 1) % 5
    injected = observed_labels != pre_labels
    probabilities = np.full((len(sample_ids), 5), 0.1, dtype=np.float64)
    probabilities[np.arange(len(sample_ids)), pre_labels] = 0.6
    fold_id = np.tile(np.arange(5, dtype=np.int64), 2)
    np.savez(
        directory / "oof_evidence.npz",
        sample_ids=sample_ids,
        group_ids=group_ids,
        pre_corruption_label=pre_labels,
        observed_label=observed_labels,
        is_injected_corruption=injected,
        probabilities=probabilities,
        fold_id=fold_id,
        fold_assignment_labels=pre_labels,
        coverage_count=np.ones(len(sample_ids), dtype=np.int64),
    )

    ensemble = ensemble_disagreement(
        [probabilities, probabilities, probabilities],
        observed_labels=observed_labels,
        class_order=(0, 1, 2, 3, 4),
    )
    ensemble_primary = predeclared_ensemble_risk(
        ensemble,
        primary_risk=controls.ensemble_primary_risk,
    )
    self_confidence = score_annotations(
        observed_labels,
        probabilities,
        method="self_confidence",
        class_order=(0, 1, 2, 3, 4),
    )
    hybrid = fixed_hybrid_drop_one_ablations(
        {
            "self_confidence": self_confidence,
            "ensemble_disagreement": ensemble_primary,
        },
        components=controls.hybrid_components,
        weights=controls.hybrid_weights,
    )
    fixed_hybrid = hybrid.full_score
    np.savez(
        directory / "risk_scores.npz",
        sample_ids=sample_ids,
        ensemble_mean_probabilities=ensemble.averaged_probabilities,
        self_confidence=self_confidence,
        ensemble_disagreement=ensemble_primary,
        fixed_hybrid=fixed_hybrid,
        ensemble_predictive_entropy_of_mean=ensemble.entropy_of_mean,
        ensemble_mean_pairwise_js_divergence=ensemble.mean_pairwise_js_divergence,
        ensemble_variation_ratio=ensemble.variation_ratio,
        ensemble_observed_label_probability_variance=(ensemble.observed_label_probability_variance),
        ensemble_predicted_class_disagreement=ensemble.predicted_class_disagreement,
        **{
            f"hybrid_drop_{component}": values
            for component, values in hybrid.drop_one_scores.items()
        },
    )
    ordered = sorted(
        zip(sample_ids.tolist(), fixed_hybrid.tolist(), strict=True),
        key=lambda item: (-item[1], item[0]),
    )
    with (directory / "ranking.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("sample_id", "risk_method", "risk_score", "rank"),
        )
        writer.writeheader()
        for rank, (sample_id, score) in enumerate(ordered, start=1):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "risk_method": "fixed_hybrid",
                    "risk_score": repr(score),
                    "rank": rank,
                }
            )

    _write_json(
        directory / "metrics.json",
        {
            "cell_identity": identity,
            "ranking": {
                "fixed_hybrid": {
                    "average_precision": average_precision(injected, fixed_hybrid),
                }
            },
        },
    )
    scenario = controls.scenarios_by_id[cell.scenario_id]
    checkpoint_records: list[dict[str, Any]] = []
    fold_execution: list[dict[str, Any]] = []
    configuration_hashes: list[str] = []
    cache_record = controls.cache_provenance_by_id[scenario.cache_provenance_id]
    provenance = {
        "cache_provenance_id": cache_record["id"],
        "representation_id": cache_record["representation_id"],
        "cache_file_sha256": cache_record.get("cache_file_sha256"),
        "sidecar_semantic_sha256": cache_record.get("sidecar_semantic_sha256"),
        "sample_order_sha256": cache_record["sample_order_sha256"],
        "manifest_sha256": cache_record["manifest_sha256"],
        "encoder_identifier": cache_record["encoder_identifier"],
        "encoder_metadata_sha256": cache_record["encoder_metadata_sha256"],
        "weight_identifier": cache_record["weight_identifier"],
        "weights_sha256": cache_record["weights_sha256"],
        "preprocessing_identifier": cache_record["preprocessing_identifier"],
        "preprocessing_sha256": cache_record["preprocessing_sha256"],
        "input_variant": cache_record["input_variant"],
        "audit_sample_order_sha256": _canonical_sha256(sample_ids.tolist()),
    }
    provenance_sha256 = _canonical_sha256(provenance)
    if scenario.family == "cnn":
        canonical_checkpoint_root = directory / "checkpoints"
        checkpoint_versions_root = directory / "checkpoint_versions"
        checkpoint_execution_root = directory / "checkpoint_execution"
        canonical_checkpoint_root.mkdir()
        checkpoint_versions_root.mkdir()
        checkpoint_execution_root.mkdir()
        for fold in range(controls.n_splits):
            checkpoint_output_directory = checkpoint_versions_root / f"fold_{fold:02d}"
            checkpoint_output_directory.mkdir()
            checkpoint = checkpoint_output_directory / "epoch_0001.pt"
            configuration = {
                "input_variant": scenario.input_variant,
                "weight_identifier": cache_record["weight_identifier"],
                "fourth_channel_initialisation": "zeros",
                "input_size": controls.input_size,
                "epochs": controls.max_epochs,
                "batch_size": controls.initial_batch_size,
                "minimum_batch_size": controls.minimum_batch_size,
                "gradient_accumulation_steps": controls.gradient_accumulation_steps,
                "learning_rate": controls.learning_rate,
                "weight_decay": controls.weight_decay,
                "early_stopping_patience": controls.early_stopping_patience,
                "early_stopping_min_delta": controls.early_stopping_min_delta,
                "amp_dtype": controls.amp_dtype,
                "class_weight_balanced": controls.class_weight == "balanced",
                "seed": cell.model_seed + fold,
            }
            model_metadata = {
                "architecture": "torchvision.resnet18",
                "class_order": [0, 1, 2, 3, 4],
                "input_channels": (
                    4 if scenario.input_variant == "context_rgb_plus_binary_target_mask" else 3
                ),
                "weight_identifier": cache_record["weight_identifier"],
                "weight_path": "C:/torch-cache/resnet18-f37072fd.pth",
                "weight_sha256": cache_record["weights_sha256"],
                "implicit_weight_download": False,
                "preprocessing": {
                    "rgb_resize": "bilinear_antialias",
                    "rgb_range_before_normalisation": [0.0, 1.0],
                    "rgb_mean": [0.485, 0.456, 0.406],
                    "rgb_standard_deviation": [0.229, 0.224, 0.225],
                    "target_mask_resize": (
                        "nearest_binary_unnormalised"
                        if scenario.input_variant == "context_rgb_plus_binary_target_mask"
                        else None
                    ),
                },
                "fourth_channel_initialisation": (
                    "zeros"
                    if scenario.input_variant == "context_rgb_plus_binary_target_mask"
                    else None
                ),
            }
            data_and_split_sha256 = {
                role: _canonical_sha256(
                    {
                        "fixture": "confirmatory_real_checkpoint",
                        "cell_id": cell.cell_id,
                        "fold_id": fold,
                        "role": role,
                    }
                )
                for role in (
                    "training_data_sha256",
                    "reference_validation_data_sha256",
                    "training_split_sha256",
                    "reference_validation_split_sha256",
                )
            }
            _write_real_checkpoint_fixture(
                checkpoint,
                configuration=configuration,
                model_metadata=model_metadata,
                data_and_split_sha256=data_and_split_sha256,
            )
            canonical_checkpoint = canonical_checkpoint_root / f"fold_{fold:02d}.pt"
            canonical_checkpoint.write_bytes(checkpoint.read_bytes())
            checkpoint.chmod(0o444)
            canonical_checkpoint.chmod(0o444)
            checkpoint_sha = sha256_file(checkpoint)
            assert sha256_file(canonical_checkpoint) == checkpoint_sha
            configuration_sha = _canonical_sha256(configuration)
            configuration_hashes.append(configuration_sha)
            versioned_output_relative = f"cells/{cell.cell_id}/checkpoint_versions/fold_{fold:02d}"
            execution_manifest_relative = (
                f"cells/{cell.cell_id}/checkpoint_execution/fold_{fold:02d}.json"
            )
            canonical_json = lambda value: json.dumps(  # noqa: E731
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            directive = ConfirmatoryCheckpointDirective(
                execution_mode="fresh",
                cell_id=cell.cell_id,
                fold_id=fold,
                action="fresh_fit",
                source_predecessor_checkpoint=None,
                destination_imported_checkpoint=None,
                versioned_checkpoint_output_directory_relative_path=(versioned_output_relative),
                checkpoint_execution_manifest_relative_path=(execution_manifest_relative),
                checkpoint_sha256=None,
                checkpoint_size_bytes=None,
                completed_epochs_before_fit=0,
                stopped_early_before_fit=None,
                next_epoch_index=0,
                maximum_epochs=controls.max_epochs,
                expected_configuration_json=canonical_json(configuration),
                expected_configuration_sha256=configuration_sha,
                expected_model_metadata_json=canonical_json(model_metadata),
                expected_model_metadata_sha256=_canonical_sha256(model_metadata),
                expected_data_and_split_json=canonical_json(data_and_split_sha256),
                expected_data_and_split_sha256=_canonical_sha256(data_and_split_sha256),
            )
            directive.validate()
            checkpoint_identity = _checkpoint_file_identity(checkpoint)
            canonical_checkpoint_identity = _checkpoint_file_identity(canonical_checkpoint)
            assert (
                canonical_checkpoint_identity["file_id_128"] != checkpoint_identity["file_id_128"]
            )
            commit_sidecar = checkpoint_output_directory / "epoch_0001.commit.json"
            _write_canonical_ascii_json(
                commit_sidecar,
                {
                    "schema_version": 2,
                    "policy": "aanca_fold_boundary_checkpoint_commit_v2",
                    "fit_id": f"{cell.cell_id}::fold_{fold:02d}",
                    "fit_attempt": 1,
                    "publication_index": 1,
                    "publication_boundary": "successful_fold_completion",
                    "completed_epochs": 1,
                    "trained_epochs": 1,
                    "versioned_checkpoint_output_directory_relative_path": (
                        versioned_output_relative
                    ),
                    "checkpoint_execution_manifest_relative_path": (execution_manifest_relative),
                    "directive_sha256": directive.directive_sha256,
                    "source_predecessor_checkpoint": None,
                    "destination_imported_checkpoint": None,
                    "previous_checkpoint": None,
                    "canonical_working_checkpoint": canonical_checkpoint_identity,
                    "versioned_checkpoint": checkpoint_identity,
                    "automatic_retry_allowed": False,
                    "hardlink_or_replace_used": False,
                    "canonical_working_checkpoint_read_only": True,
                    "mutable_latest_path_created": False,
                },
            )
            commit_sidecar.chmod(0o444)
            commit_identity = ConfirmatoryCheckpointPhysicalIdentity.from_stat(
                commit_sidecar.lstat()
            )
            publication = {
                "publication_index": 1,
                "completed_epochs": 1,
                "checkpoint_relative_path": (f"{versioned_output_relative}/epoch_0001.pt"),
                "checkpoint": checkpoint_identity,
                "commit_manifest_relative_path": (
                    f"{versioned_output_relative}/epoch_0001.commit.json"
                ),
                "commit_manifest_sha256": sha256_file(commit_sidecar),
                "commit_manifest_size_bytes": commit_sidecar.stat().st_size,
                "commit_manifest_physical_identity": commit_identity.as_dict(),
            }
            execution_manifest = checkpoint_execution_root / f"fold_{fold:02d}.json"
            _write_canonical_ascii_json(
                execution_manifest,
                {
                    "schema_version": 3,
                    "policy": "aanca_fold_boundary_checkpoint_execution_v3",
                    "fit_id": f"{cell.cell_id}::fold_{fold:02d}",
                    "fit_attempt": 1,
                    "action": "fresh_fit",
                    "directive_sha256": directive.directive_sha256,
                    "source_predecessor_checkpoint": None,
                    "destination_imported_checkpoint": None,
                    "imported_checkpoint_observed": None,
                    "canonical_working_checkpoint": canonical_checkpoint_identity,
                    "canonical_working_checkpoint_read_only": True,
                    "versioned_checkpoint_output_directory_relative_path": (
                        versioned_output_relative
                    ),
                    "checkpoint_execution_manifest_relative_path": (execution_manifest_relative),
                    "completed_epochs_before_fit": 0,
                    "completed_epochs_after_fit": 1,
                    "trained_epochs": 1,
                    "publication_boundary": "successful_fold_completion",
                    "versioned_outputs": [publication],
                    "final_checkpoint": checkpoint_identity,
                    "automatic_retry_allowed": False,
                    "imported_checkpoint_modified": False,
                    "hardlink_or_replace_used_for_immutable_publication": False,
                    "mutable_latest_path_created": False,
                },
            )
            execution_manifest.chmod(0o444)
            execution_manifest_identity = ConfirmatoryCheckpointPhysicalIdentity.from_stat(
                execution_manifest.lstat()
            )
            history_record = {
                "publication_index": 1,
                "completed_epochs": 1,
                "checkpoint_relative_path": publication["checkpoint_relative_path"],
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_physical_identity": (
                    ConfirmatoryCheckpointPhysicalIdentity.from_stat(checkpoint.lstat()).as_dict()
                ),
                "commit_manifest_relative_path": publication["commit_manifest_relative_path"],
                "commit_manifest_sha256": publication["commit_manifest_sha256"],
                "commit_manifest_size_bytes": publication["commit_manifest_size_bytes"],
                "commit_manifest_physical_identity": commit_identity.as_dict(),
            }
            canonical_record = {
                "path": canonical_checkpoint.relative_to(directory.parent.parent).as_posix(),
                "sha256": checkpoint_sha,
                "size_bytes": canonical_checkpoint.stat().st_size,
                "file_id_128": canonical_checkpoint_identity["file_id_128"],
                "physical_identity": (
                    ConfirmatoryCheckpointPhysicalIdentity.from_stat(
                        canonical_checkpoint.lstat()
                    ).as_dict()
                ),
                "read_only": True,
            }
            checkpoint_records.append(
                {
                    "fold_id": fold,
                    "status": "complete",
                    "path": checkpoint.relative_to(directory).as_posix(),
                    "sha256": checkpoint_sha,
                    "physical_identity": (
                        ConfirmatoryCheckpointPhysicalIdentity.from_stat(
                            checkpoint.lstat()
                        ).as_dict()
                    ),
                    "configuration_sha256": configuration_sha,
                    "execution_manifest_path": (
                        execution_manifest.relative_to(directory).as_posix()
                    ),
                    "execution_manifest_sha256": sha256_file(execution_manifest),
                    "execution_manifest_physical_identity": (execution_manifest_identity.as_dict()),
                    "directive": directive.as_dict(),
                    "directive_sha256": directive.directive_sha256,
                    "canonical_working_checkpoint": canonical_record,
                    "versioned_outputs": [history_record],
                }
            )
            fold_execution.append(
                {
                    "fold_id": fold,
                    "model_seed": cell.model_seed + fold,
                    "checkpoint_path": str(checkpoint.resolve()),
                    "checkpoint_sha256": checkpoint_sha,
                    "checkpoint_size_bytes": checkpoint.stat().st_size,
                    "checkpoint_physical_identity": (
                        ConfirmatoryCheckpointPhysicalIdentity.from_stat(
                            checkpoint.lstat()
                        ).as_dict()
                    ),
                    "checkpoint_execution_manifest_path": str(execution_manifest.resolve()),
                    "checkpoint_execution_manifest_sha256": sha256_file(execution_manifest),
                    "checkpoint_execution_manifest_physical_identity": (
                        execution_manifest_identity.as_dict()
                    ),
                    "checkpoint_execution_mode": "fresh",
                    "checkpoint_action": "fresh_fit",
                    "completed_epochs_before_fit": 0,
                    "trained_epochs_this_invocation": 1,
                    "completed_epochs": 1,
                    "checkpoint_canonical_working": canonical_record,
                    "checkpoint_versioned_outputs": [history_record],
                    "configuration_sha256": configuration_sha,
                    "execution_mode": "real_study_cuda",
                    "study_outcome_eligible": True,
                    "telemetry": {
                        "execution_mode": "real_study_cuda",
                        "study_outcome_eligible": True,
                    },
                    "model_metadata": model_metadata,
                    "data_and_split_sha256": data_and_split_sha256,
                }
            )
    _write_json(
        directory / "checkpoint_manifest.json",
        {
            "schema_version": 3 if scenario.family == "cnn" else 1,
            "cell_id": cell.cell_id,
            "status": ("complete" if scenario.family == "cnn" else "not_applicable_frozen_feature"),
            "checkpoints": checkpoint_records,
        },
    )
    telemetry_folds = []
    all_groups = set(group_ids.tolist())
    for fold in range(controls.n_splits):
        held_indices = np.flatnonzero(fold_id == fold)
        held_groups = {str(group_ids[index]) for index in held_indices}
        telemetry_folds.append(
            {
                "fold_id": fold,
                "training_groups": sorted(all_groups.difference(held_groups)),
                "held_out_groups": sorted(held_groups),
                "held_out_sample_ids": [str(sample_ids[index]) for index in held_indices],
            }
        )
    if scenario.family == "cnn":
        configuration_sha256 = _canonical_sha256(configuration_hashes)
        execution_evidence: dict[str, Any] = {
            "fold_evidence": fold_execution,
            "execution_mode": "real_study_cuda",
            "study_outcome_eligible": True,
            "scenario_cache_provenance_sha256": provenance_sha256,
        }
    else:
        classifier = controls.original_audit_selection["classifier"]
        parameters = classifier["parameters"]
        configuration = {
            "schema_version": 1,
            "classifier": "multinomial_logistic_regression",
            "representation_id": scenario.representation_id,
            "model_seed": cell.model_seed,
            "split_seed": controls.split_seed,
            "n_splits": controls.n_splits,
            "l2": float(parameters["l2"]),
            "max_iter": int(parameters["max_iter"]),
            "class_weight": "balanced",
            "class_order": [0, 1, 2, 3, 4],
            "fold_assignment_label_source": "pre_corruption_label",
            "frozen_feature_provenance_sha256": provenance_sha256,
        }
        configuration_sha256 = _canonical_sha256(configuration)
        execution_evidence = {
            **configuration,
            "configuration_sha256": configuration_sha256,
            "feature_array_sha256": "f" * 64,
            "observed_labels_sha256": array_artifact_sha256(observed_labels),
            "fold_assignment_labels_sha256": array_artifact_sha256(pre_labels),
            "estimator_device": "cpu",
            "cuda_execution_gate_required": False,
        }
    _write_json(
        directory / "telemetry.json",
        {
            "schema_version": 1,
            "cell_id": cell.cell_id,
            "execution_mode": ("real_study_cuda" if scenario.family == "cnn" else "real_study_cpu"),
            "study_outcome_eligible": True,
            "configuration_sha256": configuration_sha256,
            "folds": telemetry_folds,
            "evidence": execution_evidence,
        },
    )
    manifest = _write_hash_manifest(directory, _COMPLETED_ARTIFACTS)
    return sha256_file(manifest), sha256_file(directory / "metrics.json")


def _write_skipped_cell(
    directory: Path,
    *,
    cell: ConfirmatoryCell,
    identity: dict[str, Any],
) -> tuple[str, str]:
    directory.mkdir(parents=True)
    _write_json(directory / "cell_identity.json", identity)
    blocker = "pathology encoder unavailable under the frozen priority rule"
    _write_json(
        directory / "blocker.json",
        {
            "cell_id": cell.cell_id,
            "frozen_unavailability": True,
            "blocker": blocker,
            "blocker_evidence_sha256": "b" * 64,
        },
    )
    manifest = _write_hash_manifest(
        directory,
        ("cell_identity.json", "blocker.json"),
    )
    return sha256_file(manifest), blocker


def _two_sided_bootstrap_probability(differences: np.ndarray[Any, Any]) -> float:
    return min(
        1.0,
        2.0
        * min(
            float(np.mean(differences <= 0.0)),
            float(np.mean(differences >= 0.0)),
        ),
    )


def _holm_adjusted_p_values(records: list[tuple[str, str, float]]) -> dict[str, float]:
    adjusted: dict[str, float] = {}
    for family in sorted({value[1] for value in records}):
        members = sorted(
            (
                (comparison_id, raw_p)
                for comparison_id, record_family, raw_p in records
                if record_family == family
            ),
            key=lambda value: (value[1], value[0]),
        )
        running = 0.0
        for index, (comparison_id, raw_p) in enumerate(members):
            running = max(running, min(1.0, (len(members) - index) * raw_p))
            adjusted[comparison_id] = running
    return adjusted


def _selected_injected_event_count(
    run: Path,
    controls: ConfirmatoryExecutionControls,
    definition: dict[str, Any],
) -> int:
    operand = definition["operand_a"]
    selected_folds = (
        list(controls.official_folds)
        if operand["outer_fold"] == "all_matched"
        else [int(operand["outer_fold"])]
    )
    selected_corruptions = (
        sorted(cell.corruption_cell_id for cell in controls.corruption_specs)
        if operand["corruption_cell"] == "all_matched"
        else [str(operand["corruption_cell"])]
    )
    total = 0
    for outer_fold in selected_folds:
        for corruption_id in selected_corruptions:
            representative = next(
                cell
                for cell in controls.plan.cells
                if cell.outer_fold == outer_fold
                and cell.corruption_cell_id == corruption_id
                and cell.required
            )
            with np.load(
                run / "cells" / representative.cell_id / "oof_evidence.npz",
                allow_pickle=False,
            ) as payload:
                total += int(np.asarray(payload["is_injected_corruption"], dtype=bool).sum())
    return total


def _paired_operand_arrays(
    run: Path,
    controls: ConfirmatoryExecutionControls,
    operand: dict[str, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    selected = [
        cell
        for cell in controls.plan.cells
        if cell.scenario_id == operand["scenario_id"]
        and (operand["outer_fold"] == "all_matched" or cell.outer_fold == operand["outer_fold"])
        and (
            operand["corruption_cell"] == "all_matched"
            or cell.corruption_cell_id == operand["corruption_cell"]
        )
    ]
    injected_parts: list[np.ndarray[Any, Any]] = []
    score_parts: list[np.ndarray[Any, Any]] = []
    group_parts: list[np.ndarray[Any, Any]] = []
    for cell in selected:
        directory = run / "cells" / cell.cell_id
        with np.load(directory / "oof_evidence.npz", allow_pickle=False) as oof:
            injected_parts.append(np.asarray(oof["is_injected_corruption"], dtype=bool))
            group_parts.append(
                np.asarray(
                    [f"fold_{cell.outer_fold}::{group}" for group in oof["group_ids"].tolist()],
                    dtype=np.str_,
                )
            )
        with np.load(directory / "risk_scores.npz", allow_pickle=False) as risks:
            score_parts.append(np.asarray(risks[operand["risk_id"]], dtype=np.float64))
    return (
        np.concatenate(injected_parts),
        np.concatenate(score_parts),
        np.concatenate(group_parts),
    )


def _paired_bootstrap_metrics(
    group_draws: np.ndarray[Any, Any],
    injected: np.ndarray[Any, Any],
    scores_a: np.ndarray[Any, Any],
    scores_b: np.ndarray[Any, Any],
    group_tags: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    members = {str(group): np.flatnonzero(group_tags == group) for group in np.unique(group_tags)}
    valid = np.zeros(len(group_draws), dtype=bool)
    metric_a: list[float] = []
    metric_b: list[float] = []
    for index, draw in enumerate(group_draws):
        indices = np.concatenate([members[str(group)] for group in draw])
        value_a = average_precision(injected[indices], scores_a[indices])
        value_b = average_precision(injected[indices], scores_b[indices])
        if value_a is None or value_b is None:
            continue
        valid[index] = True
        metric_a.append(value_a)
        metric_b.append(value_b)
    return (
        valid,
        np.asarray(metric_a, dtype=np.float64),
        np.asarray(metric_b, dtype=np.float64),
    )


def _write_paired_statistics(
    run: Path,
    *,
    config: dict[str, Any],
    controls: ConfirmatoryExecutionControls,
) -> None:
    statistics = config["statistics"]
    definitions = statistics["preregistered_paired_comparisons"]
    iterations = int(statistics["paired_group_bootstrap_iterations"])
    optional_scenarios = {
        scenario["id"]: scenario
        for scenario in config["scenarios"]
        if scenario["required"] is False
    }
    group_universe = np.asarray(
        [
            f"fold_{outer_fold}::group_{index:02d}"
            for outer_fold in controls.official_folds
            for index in range(10)
        ],
        dtype=np.str_,
    )
    rng = np.random.default_rng(int(statistics["bootstrap_seed"]))
    arrays: dict[str, np.ndarray[Any, Any]] = {
        "bootstrap_group_universe": group_universe,
        "bootstrap_group_draws": np.stack(
            [
                rng.choice(group_universe, size=len(group_universe), replace=True)
                for _ in range(iterations)
            ]
        ),
    }
    results: list[dict[str, Any]] = []
    raw_p_values: list[tuple[str, str, float]] = []
    completed_by_id: dict[str, dict[str, Any]] = {}
    for definition in definitions:
        comparison_id = str(definition["comparison_id"])
        operand_a = definition["operand_a"]
        operand_b = definition["operand_b"]
        optional_ids = {
            str(operand_a["scenario_id"]),
            str(operand_b["scenario_id"]),
        }.intersection(optional_scenarios)
        if optional_ids:
            valid_draw_mask = np.zeros(iterations, dtype=bool)
            metric_a = np.empty(0, dtype=np.float64)
            metric_b = np.empty(0, dtype=np.float64)
            differences = np.empty(0, dtype=np.float64)
            optional_id = sorted(optional_ids)[0]
            result = {
                **definition,
                "status": "not_estimable_frozen_optional_blocker",
                "paired_unit": config["data"]["group_unit"],
                "bootstrap_seed": statistics["bootstrap_seed"],
                "requested_iterations": iterations,
                "valid_iterations": 0,
                "observed_delta": None,
                "ci_low": None,
                "ci_high": None,
                "probability_positive": None,
                "raw_p": None,
                "holm_adjusted_p": None,
                "frozen_unavailability": True,
                "blocker": "optional pathology encoder unavailable under frozen rule",
                "availability_audit_sha256": optional_scenarios[optional_id][
                    "availability_audit_sha256"
                ],
            }
        else:
            injected_a, scores_a, groups_a = _paired_operand_arrays(
                run,
                controls,
                operand_a,
            )
            injected_b, scores_b, groups_b = _paired_operand_arrays(
                run,
                controls,
                operand_b,
            )
            assert np.array_equal(injected_a, injected_b)
            assert np.array_equal(groups_a, groups_b)
            valid_draw_mask, metric_a, metric_b = _paired_bootstrap_metrics(
                arrays["bootstrap_group_draws"],
                injected_a,
                scores_a,
                scores_b,
                groups_a,
            )
            differences = metric_a - metric_b
            observed_delta = float(np.mean(differences))
            ci_low, ci_high = (float(value) for value in np.quantile(differences, (0.025, 0.975)))
            probability_positive = float(
                np.mean(differences > 0.0) + 0.5 * np.mean(differences == 0.0)
            )
            raw_p = _two_sided_bootstrap_probability(differences)
            result = {
                **definition,
                "status": "completed",
                "paired_unit": config["data"]["group_unit"],
                "bootstrap_seed": statistics["bootstrap_seed"],
                "requested_iterations": iterations,
                "valid_iterations": int(valid_draw_mask.sum()),
                "observed_delta": observed_delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "probability_positive": probability_positive,
                "raw_p": raw_p,
                "holm_adjusted_p": None,
            }
            raw_p_values.append((comparison_id, str(definition["holm_family"]), raw_p))
            completed_by_id[comparison_id] = result
        arrays[f"valid_draw_mask__{comparison_id}"] = valid_draw_mask
        arrays[f"metric_a__{comparison_id}"] = metric_a
        arrays[f"metric_b__{comparison_id}"] = metric_b
        arrays[f"differences__{comparison_id}"] = differences
        result["selected_injected_event_count"] = _selected_injected_event_count(
            run,
            controls,
            definition,
        )
        results.append(result)

    for comparison_id, adjusted in _holm_adjusted_p_values(raw_p_values).items():
        completed_by_id[comparison_id]["holm_adjusted_p"] = adjusted
    evidence_path = run / "paired_bootstrap_evidence.npz"
    np.savez(evidence_path, **arrays)
    _write_json(
        run / "paired_statistics.json",
        {
            "schema_version": 1,
            "config_semantic_sha256": controls.config_semantic_sha256,
            "outer_folds": list(controls.official_folds),
            "paired_unit": config["data"]["group_unit"],
            "bootstrap_iterations": iterations,
            "bootstrap_seed": statistics["bootstrap_seed"],
            "bootstrap_evidence_path": evidence_path.name,
            "bootstrap_evidence_sha256": sha256_file(evidence_path),
            "comparisons": results,
        },
    )


def _final_sample_ids(outer_fold: int) -> np.ndarray[Any, Any]:
    return np.asarray(
        [f"final_{outer_fold}_sample_{index:02d}" for index in range(10)],
        dtype=np.str_,
    )


def _final_group_ids(outer_fold: int) -> np.ndarray[Any, Any]:
    return np.asarray(
        [f"final_{outer_fold}_group_{index:02d}" for index in range(10)],
        dtype=np.str_,
    )


def _final_probabilities(
    labels: np.ndarray[Any, Any],
    *,
    wrong_indices: tuple[int, ...] = (),
) -> np.ndarray[Any, Any]:
    probabilities = np.full((len(labels), 5), 0.1, dtype=np.float64)
    probabilities[np.arange(len(labels)), labels] = 0.6
    for index in wrong_indices:
        probabilities[index] = 0.1
        probabilities[index, (int(labels[index]) + 1) % 5] = 0.6
    return probabilities


def _write_restoration_evidence(
    run: Path,
    *,
    config: dict[str, Any],
    controls: ConfirmatoryExecutionControls,
) -> None:
    restoration = config["restoration"]
    active_corruptions = [
        corruption.corruption_cell_id
        for corruption in controls.corruption_specs
        if corruption.rate > 0.0
    ] or [corruption.corruption_cell_id for corruption in controls.corruption_specs]
    repeats = int(restoration["random_repeats"])
    random_seed = int(restoration["random_seed"])
    evidence: dict[str, np.ndarray[Any, Any]] = {}
    rotations: list[dict[str, Any]] = []
    final_pre = np.tile(np.arange(5, dtype=np.int64), 2)
    final_observed = final_pre.copy()
    final_injected = np.zeros(len(final_pre), dtype=bool)
    for outer_fold in controls.official_folds:
        final_ids = _final_sample_ids(outer_fold)
        final_groups = _final_group_ids(outer_fold)
        final_group_sha = _canonical_sha256(sorted(set(final_groups.tolist())))
        for corruption_id in active_corruptions:
            source_cell = next(
                cell
                for cell in controls.plan.cells
                if cell.outer_fold == outer_fold
                and cell.corruption_cell_id == corruption_id
                and cell.scenario_id == restoration["scenario_id"]
                and cell.model_seed == restoration["model_seed"]
            )
            cell_directory = run / "cells" / source_cell.cell_id
            with np.load(cell_directory / "oof_evidence.npz", allow_pickle=False) as oof:
                audit_ids = np.asarray(oof["sample_ids"]).copy()
                audit_groups = np.asarray(oof["group_ids"]).copy()
                pre = np.asarray(oof["pre_corruption_label"], dtype=np.int64)
                observed = np.asarray(oof["observed_label"], dtype=np.int64)
                injected = np.asarray(oof["is_injected_corruption"], dtype=bool)
            with np.load(cell_directory / "risk_scores.npz", allow_pickle=False) as risks:
                ranking = np.asarray(risks[restoration["ranking_method"]], dtype=np.float64)
            budget = budget_count(len(audit_ids), float(restoration["review_budget"]))
            guided_reviewed = np.zeros(len(audit_ids), dtype=bool)
            guided_reviewed[rank_indices(ranking, tie_break_ids=audit_ids.tolist())[:budget]] = True
            guided = restore_reviewed_labels(pre, observed, injected, guided_reviewed)

            random_reviewed = np.zeros((repeats, len(audit_ids)), dtype=bool)
            random_restored = np.zeros_like(random_reviewed)
            random_labels = np.zeros((repeats, len(audit_ids)), dtype=np.int64)
            random_seeds = [random_seed + repeat for repeat in range(repeats)]
            for repeat, repeat_seed in enumerate(random_seeds):
                rng = np.random.default_rng(repeat_seed)
                selected = np.sort(rng.choice(len(audit_ids), size=budget, replace=False)).astype(
                    np.int64
                )
                random_reviewed[repeat, selected] = True
                restored = restore_reviewed_labels(
                    pre,
                    observed,
                    injected,
                    random_reviewed[repeat],
                )
                random_restored[repeat] = restored.restored_mask
                random_labels[repeat] = restored.restored_labels

            uncorrupted_probabilities = _final_probabilities(final_pre)
            corrupted_probabilities = _final_probabilities(final_pre, wrong_indices=(0, 1))
            guided_probabilities = _final_probabilities(final_pre, wrong_indices=(1,))
            random_probabilities = np.stack(
                [
                    _final_probabilities(
                        final_pre,
                        wrong_indices=(0, 1) if repeat % 2 == 0 else (),
                    )
                    for repeat in range(repeats)
                ]
            )
            prefix = f"fold_{outer_fold}__{corruption_id}"
            values = {
                "audit_sample_ids": audit_ids,
                "audit_group_ids": audit_groups,
                "pre_corruption_label": pre,
                "observed_label": observed,
                "is_injected_corruption": injected,
                "guided_reviewed_mask": guided_reviewed,
                "guided_restored_mask": guided.restored_mask,
                "guided_restored_label": guided.restored_labels,
                "random_reviewed_mask": random_reviewed,
                "random_restored_mask": random_restored,
                "random_restored_label": random_labels,
                "final_sample_ids": final_ids,
                "final_group_ids": final_groups,
                "final_pre_corruption_label": final_pre,
                "final_observed_label": final_observed,
                "final_is_injected_corruption": final_injected,
                "probabilities__uncorrupted_reference_baseline": (uncorrupted_probabilities),
                "probabilities__corrupted_observed_baseline": corrupted_probabilities,
                "probabilities__random_review_restoration": random_probabilities,
                "probabilities__audit_guided_restoration": guided_probabilities,
            }
            evidence.update({f"{prefix}__{name}": value for name, value in values.items()})

            deterministic = {
                "uncorrupted_reference_baseline": uncorrupted_probabilities,
                "corrupted_observed_baseline": corrupted_probabilities,
                "audit_guided_restoration": guided_probabilities,
            }
            condition_results: dict[str, Any] = {
                condition: {
                    "metrics": classification_metrics(
                        final_pre,
                        probabilities,
                        class_order=(0, 1, 2, 3, 4),
                    ).as_dict(),
                    "reviewed_count": budget if condition == "audit_guided_restoration" else 0,
                    "restored_count": int(guided.restored_mask.sum())
                    if condition == "audit_guided_restoration"
                    else 0,
                }
                for condition, probabilities in deterministic.items()
            }
            random_runs = [
                {
                    "review_seed": random_seeds[repeat],
                    "metrics": classification_metrics(
                        final_pre,
                        random_probabilities[repeat],
                        class_order=(0, 1, 2, 3, 4),
                    ).as_dict(),
                    "reviewed_count": budget,
                    "restored_count": int(random_restored[repeat].sum()),
                }
                for repeat in range(repeats)
            ]
            random_f1 = np.asarray(
                [float(value["metrics"]["macro_f1"]) for value in random_runs],
                dtype=np.float64,
            )
            condition_results["random_review_restoration"] = {
                "runs": random_runs,
                "macro_f1_mean": float(random_f1.mean()),
                "macro_f1_interval_95": [
                    float(value) for value in np.quantile(random_f1, (0.025, 0.975))
                ],
            }
            rotations.append(
                {
                    "outer_fold": outer_fold,
                    "corruption_cell_id": corruption_id,
                    "audit_sample_count": len(audit_ids),
                    "final_sample_count": len(final_ids),
                    "review_budget_count": budget,
                    "random_review_seeds": random_seeds,
                    "final_reference_group_ids_sha256": final_group_sha,
                    "conditions": condition_results,
                }
            )

    evidence_path = run / "restoration_evidence.npz"
    np.savez(evidence_path, **evidence)
    _write_json(
        run / "restoration_metrics.json",
        {
            "schema_version": 1,
            "status": "completed",
            "config_semantic_sha256": controls.config_semantic_sha256,
            "outer_folds": list(controls.official_folds),
            "scenario_id": restoration["scenario_id"],
            "model_seed": restoration["model_seed"],
            "representation_id": restoration["representation_id"],
            "ranking_method": restoration["ranking_method"],
            "review_budget": restoration["review_budget"],
            "random_repeats": repeats,
            "random_seed": random_seed,
            "conditions": restoration["conditions"],
            "evidence_path": evidence_path.name,
            "evidence_sha256": sha256_file(evidence_path),
            "rotations": rotations,
        },
    )
    classifier_parameters = controls.original_audit_selection["classifier"]["parameters"]
    partition_bindings = {"fixture": {"audit": {"sha256": "d" * 64}}}
    partition_content_sha256 = _canonical_sha256(partition_bindings)
    risk_sources: dict[str, dict[str, str]] = {}
    for outer_fold in controls.official_folds:
        for corruption_id in active_corruptions:
            source_cell = next(
                cell
                for cell in controls.plan.cells
                if cell.outer_fold == outer_fold
                and cell.corruption_cell_id == corruption_id
                and cell.scenario_id == restoration["scenario_id"]
                and cell.model_seed == restoration["model_seed"]
            )
            relative = f"cells/{source_cell.cell_id}/risk_scores.npz"
            risk_sources[f"fold_{outer_fold}__{corruption_id}"] = {
                "cell_id": source_cell.cell_id,
                "relative_path": relative,
                "sha256": sha256_file(run / relative),
            }
    _write_json(
        run / "restoration_replay_certificate.json",
        {
            "schema_version": 1,
            "status": "passed",
            "policy": "deterministic_checksum_bound_restoration_replay_v1",
            "confirmatory_config_semantic_sha256": controls.config_semantic_sha256,
            "execution_controls_binding_sha256": controls.binding_sha256,
            "bridge_partition_content_sha256": partition_content_sha256,
            "bridge_corruption_assignment_sha256": "b" * 64,
            "bridge_provenance_binding_sha256": "c" * 64,
            "scenario_id": restoration["scenario_id"],
            "representation_id": restoration["representation_id"],
            "model_seed": restoration["model_seed"],
            "ranking_method": restoration["ranking_method"],
            "review_budget": restoration["review_budget"],
            "random_repeats": repeats,
            "random_seed": random_seed,
            "l2": float(classifier_parameters["l2"]),
            "max_iter": int(classifier_parameters["max_iter"]),
            "evidence_relative_path": evidence_path.name,
            "evidence_sha256": sha256_file(evidence_path),
            "evidence_arrays": {
                key: array_artifact_sha256(value) for key, value in sorted(evidence.items())
            },
            "risk_sources": risk_sources,
        },
    )
    _write_json(
        run / "restoration_input_bindings.json",
        {
            "schema_version": 1,
            "policy": "immutable_pre_replay_partition_bindings_v1",
            "confirmatory_config_semantic_sha256": controls.config_semantic_sha256,
            "bridge_partition_content_sha256": partition_content_sha256,
            "bridge_corruption_assignment_sha256": "b" * 64,
            "bridge_provenance_binding_sha256": "c" * 64,
            "partition_bindings": partition_bindings,
        },
    )
    cnn_preflight = {
        cell.cell_id: {
            str(fold["fold_id"]): fold["data_and_split_sha256"]
            for fold in json.loads(
                (run / "cells" / cell.cell_id / "telemetry.json").read_text(encoding="utf-8")
            )["evidence"]["fold_evidence"]
        }
        for cell in controls.plan.cells
        if controls.scenarios_by_id[cell.scenario_id].family == "cnn"
    }
    _write_json(
        run / "confirmatory_input_bindings.json",
        {
            "schema_version": 1,
            "config_semantic_sha256": controls.config_semantic_sha256,
            "execution_controls_binding_sha256": controls.binding_sha256,
            "cnn_fold_data_and_split_sha256": cnn_preflight,
            "bridge": {
                "partition_content_sha256": partition_content_sha256,
                "corruption_assignment_sha256": "b" * 64,
                "provenance_binding_sha256": "c" * 64,
                "partition_bindings": partition_bindings,
            },
        },
    )


def _write_root_aggregates(
    run: Path,
    *,
    config: dict[str, Any],
    controls: ConfirmatoryExecutionControls,
) -> None:
    folds = list(controls.official_folds)
    member_keys = {
        (str(value["scenario_id"]), int(value["model_seed"]))
        for value in config["ensemble"]["members"]
    }
    ensemble_groups = [
        {
            "outer_fold": outer_fold,
            "corruption_cell_id": corruption.corruption_cell_id,
            "member_cell_ids": [
                cell.cell_id
                for cell in controls.plan.cells
                if cell.outer_fold == outer_fold
                and cell.corruption_cell_id == corruption.corruption_cell_id
                and (cell.scenario_id, cell.model_seed) in member_keys
            ],
            "all_members_completed": True,
        }
        for outer_fold in controls.official_folds
        for corruption in controls.corruption_specs
    ]
    _write_json(
        run / "ensemble_evidence.json",
        {
            "outer_folds": folds,
            "corruption_cell_ids": [item.corruption_cell_id for item in controls.corruption_specs],
            "primary_risk": config["ensemble"]["primary_risk"],
            "secondary_risks": config["ensemble"]["secondary_risks"],
            "members": config["ensemble"]["members"],
            "config_semantic_sha256": controls.config_semantic_sha256,
            "risk_arrays_are_saved_per_cell": "cells/<cell_id>/risk_scores.npz",
            "groups": ensemble_groups,
        },
    )
    completed_cells = [cell for cell in controls.plan.cells if cell.required]
    _write_json(
        run / "fixed_hybrid_drop_one_ablations.json",
        {
            "outer_folds": folds,
            "components": config["fixed_hybrid"]["components"],
            "weights": config["fixed_hybrid"]["weights"],
            "drop_one_ablations": config["fixed_hybrid"]["drop_one_ablations"],
            "config_semantic_sha256": controls.config_semantic_sha256,
            "cell_evidence": [
                {
                    "cell_id": cell.cell_id,
                    "risk_scores_path": f"cells/{cell.cell_id}/risk_scores.npz",
                    "risk_scores_sha256": sha256_file(
                        run / "cells" / cell.cell_id / "risk_scores.npz"
                    ),
                }
                for cell in completed_cells
            ],
        },
    )
    _write_json(
        run / "fold_aggregate.json",
        {
            "outer_folds": folds,
            "aggregate_policy": config["fold_rotation"]["aggregate_policy"],
            "folds": [
                {
                    "outer_fold": fold,
                    "planned_cell_count": sum(
                        cell.outer_fold == fold for cell in controls.plan.cells
                    ),
                    "completed_cell_count": sum(
                        cell.outer_fold == fold and cell.required for cell in controls.plan.cells
                    ),
                    "skipped_optional_cell_count": sum(
                        cell.outer_fold == fold and not cell.required
                        for cell in controls.plan.cells
                    ),
                    "failed_cell_count": 0,
                    "reported_separately": True,
                }
                for fold in controls.official_folds
            ],
            "outcome_metrics_aggregation_status": "completed_by_stage_statistics_runner",
        },
    )
    selection = config["original_audit_selection"]
    selected_cache = next(
        record
        for record in config["cache_provenance"]
        if record["id"] == selection["cache_provenance_id"]
    )
    provenance = json.loads((run / "frozen_feature_provenance.json").read_text(encoding="utf-8"))
    selected_rotations = provenance["representations"][selection["representation_id"]]["rotations"]
    _write_json(
        run / "original_audit_selection.json",
        {
            "schema_version": 1,
            "status": "completed",
            "confirmatory_config_semantic_sha256": controls.config_semantic_sha256,
            "matrix_plan_config_sha256": controls.plan.config_sha256,
            "execution_controls_binding_sha256": controls.binding_sha256,
            "selection": selection,
            "selection_semantic_sha256": _canonical_sha256(selection),
            "frozen_cache_provenance_record": selected_cache,
            "sealed_feature_cache_provenance_by_rotation": {
                fold: {
                    **rotation,
                    "final_reference_group_ids_sha256": _canonical_sha256(
                        sorted(set(_final_group_ids(int(fold)).tolist()))
                    ),
                }
                for fold, rotation in selected_rotations.items()
            },
        },
    )
    _write_paired_statistics(run, config=config, controls=controls)
    _write_restoration_evidence(run, config=config, controls=controls)
    restoration_metrics = json.loads((run / "restoration_metrics.json").read_text(encoding="utf-8"))
    paired_statistics = json.loads((run / "paired_statistics.json").read_text(encoding="utf-8"))
    (run / "report.md").write_text(
        "\n".join(
            (
                "# Confirmatory filesystem fixture",
                "",
                "This fixture ranks each potentially inconsistent annotation and marks it "
                "as recommended for expert review.",
                "",
                "Random mean macro F1 (95% interval)",
                "",
                "Paired comparison direction `method_a_minus_method_b`; probability delta > 0.",
                "",
                confirmatory_report_contract_block(
                    restoration_metrics,
                    paired_statistics,
                ),
                "",
            )
        ),
        encoding="utf-8",
    )
    figure = run / "figures" / "summary.png"
    figure.parent.mkdir()
    figure.write_bytes(b"fixture-figure")
    _write_json(run / "figure_manifest.json", {"figures/summary.png": sha256_file(figure)})


def _write_frozen_feature_provenance(
    run: Path,
    *,
    config: dict[str, Any],
    controls: ConfirmatoryExecutionControls,
) -> None:
    representations: dict[str, Any] = {}
    sample_order_sha = _canonical_sha256([f"sample_{index:02d}" for index in range(10)])
    for record in config["cache_provenance"]:
        if record["status"] != "available":
            continue
        provenance = {
            "cache_provenance_id": record["id"],
            "representation_id": record["representation_id"],
            "cache_file_sha256": record["cache_file_sha256"],
            "sidecar_semantic_sha256": record["sidecar_semantic_sha256"],
            "sample_order_sha256": record["sample_order_sha256"],
            "manifest_sha256": record["manifest_sha256"],
            "encoder_identifier": record["encoder_identifier"],
            "encoder_metadata_sha256": record["encoder_metadata_sha256"],
            "weight_identifier": record["weight_identifier"],
            "weights_sha256": record["weights_sha256"],
            "preprocessing_identifier": record["preprocessing_identifier"],
            "preprocessing_sha256": record["preprocessing_sha256"],
            "input_variant": record["input_variant"],
            "audit_sample_order_sha256": sample_order_sha,
        }
        representations[record["representation_id"]] = {
            "rotations": {str(fold): provenance for fold in controls.official_folds}
        }
    _write_json(
        run / "frozen_feature_provenance.json",
        {
            "schema_version": 1,
            "status": "completed",
            "confirmatory_config_semantic_sha256": controls.config_semantic_sha256,
            "matrix_plan_config_sha256": controls.plan.config_sha256,
            "representations": representations,
        },
    )


def _gate(
    freeze_directory: Path,
    plan: ConfirmatoryMatrixPlan,
    config_sha: str,
) -> ConfirmatoryExecutionGateEvidence:
    primary_gate = PrimaryExecutionGateEvidence(
        freeze_directory=freeze_directory,
        base_freeze_directory=freeze_directory,
        freeze_artifact_root_sha256="1" * 64,
        freeze_manifest_sha256="2" * 64,
        preregistration_sha256="3" * 64,
        frozen_primary_config_sha256="4" * 64,
        frozen_confirmatory_config_sha256=config_sha,
        primary_config_semantic_sha256="6" * 64,
        confirmatory_config_semantic_sha256=plan.config_sha256,
        primary_matrix_cell_count=216,
        primary_required_cell_count=180,
        confirmatory_matrix_cell_count=len(plan.cells),
        pilot_run_id="real-pannuke-pilot",
        pilot_artifact_root_sha256="7" * 64,
        dataset_sha256="8" * 64,
        manifest_sha256="9" * 64,
        duplicate_audit_sha256="a" * 64,
        pathology_encoder_audit_sha256="b" * 64,
        source_tree_root_sha256="c" * 64,
    )
    return ConfirmatoryExecutionGateEvidence(
        primary_gate=primary_gate,
        primary_run_directory=freeze_directory.parent / "primary-run",
        primary_run_id="real-pannuke-primary",
        primary_artifact_root_sha256="d" * 64,
        primary_completion_evidence_sha256="e" * 64,
        primary_reconciliation_sha256="f" * 64,
        completed_required_cell_count=180,
    )


def _build_confirmatory_tree(tmp_path: Path) -> _ConfirmatoryTree:
    config = complete_confirmatory_config()
    controls = confirmatory_execution_controls_from_frozen_config(config)
    plan = controls.plan
    assert len(plan.cells) == (
        len(config["data"]["official_folds"])
        * len(config["corruption"]["cells"])
        * len(config["scenarios"])
        * len(config["model_seeds"])
    )

    freeze_directory = tmp_path / "freeze"
    freeze_directory.mkdir()
    frozen_config_path = freeze_directory / "confirmatory_frozen.yaml"
    frozen_config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    run = tmp_path / "confirmatory-run"
    run.mkdir()
    _write_json(run / "matrix_plan.json", plan.as_dict())
    _write_json(run / "execution_controls.json", controls.as_dict())
    _write_frozen_feature_provenance(run, config=config, controls=controls)
    rows: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for cell in plan.cells:
        identity = _cell_identity(controls, cell)
        directory = run / "cells" / cell.cell_id
        corruption = controls.corruptions_by_id[cell.corruption_cell_id]
        scenario = controls.scenarios_by_id[cell.scenario_id]
        common: dict[str, Any] = {
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
        }
        if cell.required:
            manifest_sha, metrics_sha = _write_completed_cell(
                directory,
                controls=controls,
                cell=cell,
                identity=identity,
            )
            rows.append(
                {
                    **common,
                    "status": "completed",
                    "artifact_manifest_sha256": manifest_sha,
                    "metrics_sha256": metrics_sha,
                    "frozen_unavailability": "",
                    "blocker": "",
                }
            )
            outcomes.append(
                {
                    "cell_id": cell.cell_id,
                    "required": True,
                    "status": "completed",
                    "outer_fold": cell.outer_fold,
                    "model_seed": cell.model_seed,
                    "scenario_id": cell.scenario_id,
                    "corruption_cell_id": cell.corruption_cell_id,
                    "artifact_manifest_sha256": manifest_sha,
                    "metrics_sha256": metrics_sha,
                }
            )
        else:
            manifest_sha, blocker = _write_skipped_cell(
                directory,
                cell=cell,
                identity=identity,
            )
            rows.append(
                {
                    **common,
                    "status": "skipped_with_frozen_blocker",
                    "artifact_manifest_sha256": manifest_sha,
                    "metrics_sha256": "",
                    "frozen_unavailability": "true",
                    "blocker": blocker,
                }
            )
            outcomes.append(
                {
                    "cell_id": cell.cell_id,
                    "required": False,
                    "status": "skipped_with_frozen_blocker",
                    "frozen_unavailability": True,
                    "blocker": blocker,
                }
            )

    _write_cell_index(run / "cell_index.csv", rows)
    reconciliation = reconcile_confirmatory_cell_outcomes(plan, outcomes)
    assert reconciliation.passed
    _write_json(run / "reconciliation.json", reconciliation.as_dict())
    _write_root_aggregates(run, config=config, controls=controls)
    cell_manifests = tuple(f"cells/{cell.cell_id}/artifact_manifest.json" for cell in plan.cells)
    _write_hash_manifest(
        run,
        (*_ROOT_ARTIFACTS, *cell_manifests),
        filename="confirmatory_artifact_manifest.json",
    )
    gate = _gate(freeze_directory, plan, sha256_file(frozen_config_path))
    return _ConfirmatoryTree(
        config=config,
        controls=controls,
        plan=plan,
        run_directory=run,
        frozen_config_path=frozen_config_path,
        gate=gate,
        reconciliation=reconciliation,
    )


def _read_tree(tree: _ConfirmatoryTree) -> ConfirmatoryFilesystemReadback:
    return read_confirmatory_run_directory(
        tree.plan,
        tree.run_directory,
        frozen_confirmatory_config_path=tree.frozen_config_path,
        expected_frozen_config_sha256=sha256_file(tree.frozen_config_path),
    )


def _rehash_changed_cell(tree: _ConfirmatoryTree, cell_id: str) -> None:
    directory = tree.run_directory / "cells" / cell_id
    manifest = _write_hash_manifest(directory, _COMPLETED_ARTIFACTS)
    index = tree.run_directory / "cell_index.csv"
    rows = _read_cell_index(index)
    row = next(value for value in rows if value["cell_id"] == cell_id)
    row["artifact_manifest_sha256"] = sha256_file(manifest)
    row["metrics_sha256"] = sha256_file(directory / "metrics.json")
    _write_cell_index(index, rows)
    _rehash_root_manifest(tree)


def _rehash_root_manifest(tree: _ConfirmatoryTree) -> None:
    cell_manifests = tuple(
        f"cells/{cell.cell_id}/artifact_manifest.json" for cell in tree.plan.cells
    )
    _write_hash_manifest(
        tree.run_directory,
        (*_ROOT_ARTIFACTS, *cell_manifests),
        filename="confirmatory_artifact_manifest.json",
    )


def _npz_arrays(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]).copy() for name in payload.files}


def test_complete_tree_has_typed_readback_and_enables_stage(tmp_path: Path) -> None:
    tree = _build_confirmatory_tree(tmp_path)

    readback = _read_tree(tree)

    assert isinstance(readback, ConfirmatoryFilesystemReadback)
    assert readback.passed
    assert readback.errors == ()
    assert len(readback.cells) == len(tree.plan.cells)
    assert all(isinstance(cell, ConfirmatoryCellReadback) for cell in readback.cells)
    assert sum(cell.status == "completed" for cell in readback.cells) == (
        tree.plan.required_cell_count
    )
    assert sum(cell.status == "skipped_with_frozen_blocker" for cell in readback.cells) == (
        tree.plan.optional_cell_count
    )
    assert readback.checked_artifact_count > 600
    assert readback.reconciliation is not None
    assert readback.reconciliation.fold_rotation_complete
    assert readback.reconciliation.completed_outer_folds == (1, 2, 3)
    expected_cache_ids = {
        scenario["id"]: scenario["cache_provenance_id"] for scenario in tree.config["scenarios"]
    }
    assert all(
        cell.cache_provenance_id == expected_cache_ids[cell.scenario_id] for cell in readback.cells
    )

    evidence = build_confirmatory_completion_evidence(
        plan=tree.plan,
        reconciliation=tree.reconciliation,
        artifact_scope=REAL_CONFIRMATORY_ARTIFACT_SCOPE,
        study_outcome_eligible=True,
        gate_evidence=tree.gate,
        run_directory=tree.run_directory,
    )

    assert evidence["completion_stage"] == "CONFIRMATORY_COMPLETE"
    assert evidence["filesystem_readback_status"] == "passed"
    assert evidence["filesystem_checked_artifact_count"] == readback.checked_artifact_count
    assert evidence["filesystem_matrix_plan_sha256"] == sha256_file(
        tree.run_directory / "matrix_plan.json"
    )
    assert evidence["filesystem_root_artifact_manifest_sha256"] == sha256_file(
        tree.run_directory / "confirmatory_artifact_manifest.json"
    )


def test_generic_runtracker_manifest_does_not_shadow_scientific_manifest(
    tmp_path: Path,
) -> None:
    tree = _build_confirmatory_tree(tmp_path)
    before = _read_tree(tree)
    _write_json(tree.run_directory / "artifact_manifest.json", {"schema_version": 1})

    after = _read_tree(tree)

    assert before.passed and after.passed
    assert after.root_artifact_manifest_sha256 == before.root_artifact_manifest_sha256


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("metrics_sha256", "cell-index metrics SHA differs"),
        ("artifact_manifest_sha256", "cell-index artifact manifest SHA differs"),
    ],
)
def test_cell_index_rejects_well_formed_but_fake_sha(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    tree = _build_confirmatory_tree(tmp_path)
    path = tree.run_directory / "cell_index.csv"
    rows = _read_cell_index(path)
    completed = next(row for row in rows if row["status"] == "completed")
    completed[field] = "0" * 64
    _write_cell_index(path, rows)

    readback = _read_tree(tree)

    assert not readback.passed
    assert any(message in error for error in readback.errors)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "required confirmatory artifact is missing"),
        ("tampered", "SHA-256 mismatch for risk_scores.npz"),
        ("identity_swap", "differs from frozen cell identity"),
    ],
)
def test_cell_artifact_missing_tamper_and_identity_swap_fail_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    tree = _build_confirmatory_tree(tmp_path)
    completed = [cell for cell in tree.plan.cells if cell.required]
    first_directory = tree.run_directory / "cells" / completed[0].cell_id
    if mutation == "missing":
        (first_directory / "oof_evidence.npz").unlink()
    elif mutation == "tampered":
        with (first_directory / "risk_scores.npz").open("ab") as stream:
            stream.write(b"tampered")
    else:
        second_identity = tree.run_directory / "cells" / completed[1].cell_id / "cell_identity.json"
        (first_directory / "cell_identity.json").write_bytes(second_identity.read_bytes())

    readback = _read_tree(tree)

    assert not readback.passed
    assert any(message in error for error in readback.errors)


def test_cross_cell_observed_label_swap_fails_after_internal_rehash(tmp_path: Path) -> None:
    tree = _build_confirmatory_tree(tmp_path)
    cell = next(
        value
        for value in tree.plan.cells
        if value.required and tree.controls.corruptions_by_id[value.corruption_cell_id].rate > 0.0
    )
    path = tree.run_directory / "cells" / cell.cell_id / "oof_evidence.npz"
    arrays = _npz_arrays(path)
    injected_index = int(np.flatnonzero(arrays["is_injected_corruption"])[0])
    arrays["observed_label"][injected_index] = (
        int(arrays["observed_label"][injected_index]) + 1
    ) % 5
    assert (
        arrays["observed_label"][injected_index] != arrays["pre_corruption_label"][injected_index]
    )
    np.savez(path, **arrays)
    _rehash_changed_cell(tree, cell.cell_id)

    readback = _read_tree(tree)

    assert not readback.passed
    assert any(
        "observed-label corruption mapping changes between scenarios/seeds" in error
        for error in readback.errors
    )


@pytest.mark.parametrize(
    ("array_name", "message"),
    [
        ("self_confidence", "self-confidence risk differs from OOF probabilities"),
        ("ensemble_mean_probabilities", "primary ensemble evidence differs from member OOF"),
    ],
)
def test_semantic_risk_tamper_fails_after_internal_rehash(
    tmp_path: Path,
    array_name: str,
    message: str,
) -> None:
    tree = _build_confirmatory_tree(tmp_path)
    cell = next(value for value in tree.plan.cells if value.required)
    path = tree.run_directory / "cells" / cell.cell_id / "risk_scores.npz"
    arrays = _npz_arrays(path)
    if array_name == "self_confidence":
        arrays[array_name][0] += 0.01
    else:
        arrays[array_name][0, [0, 1]] = arrays[array_name][0, [1, 0]]
    np.savez(path, **arrays)
    _rehash_changed_cell(tree, cell.cell_id)

    readback = _read_tree(tree)

    assert not readback.passed
    assert any(message in error for error in readback.errors)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("difference", "is not paired A-minus-B"),
        ("summary", "summary differs from NPZ"),
        ("event_count", "injected-event count differs from OOF"),
    ],
)
def test_paired_evidence_tamper_fails_after_rehash(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    tree = _build_confirmatory_tree(tmp_path)
    statistics_path = tree.run_directory / "paired_statistics.json"
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    completed = next(value for value in statistics["comparisons"] if value["status"] == "completed")
    comparison_id = completed["comparison_id"]
    if mutation == "difference":
        evidence_path = tree.run_directory / "paired_bootstrap_evidence.npz"
        arrays = _npz_arrays(evidence_path)
        arrays[f"differences__{comparison_id}"][0] += 0.1
        np.savez(evidence_path, **arrays)
        statistics["bootstrap_evidence_sha256"] = sha256_file(evidence_path)
    elif mutation == "summary":
        completed["observed_delta"] += 0.1
    else:
        completed["selected_injected_event_count"] += 1
    _write_json(statistics_path, statistics)
    _rehash_root_manifest(tree)

    readback = _read_tree(tree)

    assert not readback.passed
    assert any(message in error for error in readback.errors)


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        ("telemetry", "does not prove group-safe OOF"),
        ("checkpoint", "checkpoint directive differs from frozen fold execution"),
        ("fold_mode", "CNN fold execution differs from frozen mode"),
        ("weight_sha", "CNN fold execution differs from frozen mode"),
    ],
)
def test_group_safe_telemetry_and_checkpoint_binding_tamper_fail_after_rehash(
    tmp_path: Path,
    artifact: str,
    message: str,
) -> None:
    tree = _build_confirmatory_tree(tmp_path)
    cell = next(
        value
        for value in tree.plan.cells
        if value.required and tree.controls.scenarios_by_id[value.scenario_id].family == "cnn"
    )
    directory = tree.run_directory / "cells" / cell.cell_id
    if artifact == "telemetry":
        path = directory / "telemetry.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["folds"][0]["training_groups"].append(payload["folds"][0]["held_out_groups"][0])
    elif artifact == "checkpoint":
        path = directory / "checkpoint_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["checkpoints"][0]["configuration_sha256"] = "0" * 64
    else:
        path = directory / "telemetry.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        fold_evidence = payload["evidence"]["fold_evidence"][0]
        if artifact == "fold_mode":
            fold_evidence["execution_mode"] = "cpu_test_only_non_evidence"
        else:
            fold_evidence["model_metadata"]["weight_sha256"] = "0" * 64
    _write_json(path, payload)
    _rehash_changed_cell(tree, cell.cell_id)

    readback = _read_tree(tree)

    assert not readback.passed
    assert any(message in error for error in readback.errors)


@pytest.mark.parametrize(
    "target",
    [
        "canonical_checkpoint",
        "final_checkpoint",
        "execution_manifest",
        "commit_sidecar",
    ],
)
def test_schema_v3_checkpoint_tree_deletion_fails_readback(
    tmp_path: Path,
    target: str,
) -> None:
    tree = _build_confirmatory_tree(tmp_path)
    cell = next(
        value
        for value in tree.plan.cells
        if value.required and tree.controls.scenarios_by_id[value.scenario_id].family == "cnn"
    )
    directory = tree.run_directory / "cells" / cell.cell_id
    checkpoint_manifest = json.loads(
        (directory / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    record = checkpoint_manifest["checkpoints"][0]
    if target == "canonical_checkpoint":
        victim = tree.run_directory.joinpath(
            *Path(record["canonical_working_checkpoint"]["path"]).parts
        )
    elif target == "final_checkpoint":
        victim = directory / record["path"]
    elif target == "execution_manifest":
        victim = directory / record["execution_manifest_path"]
    else:
        victim = tree.run_directory.joinpath(
            *Path(record["versioned_outputs"][0]["commit_manifest_relative_path"]).parts
        )
    victim.chmod(0o666)
    victim.unlink()

    readback = _read_tree(tree)

    assert not readback.passed
    assert any(
        "checkpoint" in error.casefold() or "manifest" in error.casefold()
        for error in readback.errors
    ), readback.errors


def test_schema_v3_same_byte_commit_sidecar_replacement_fails_physical_identity(
    tmp_path: Path,
) -> None:
    tree = _build_confirmatory_tree(tmp_path)
    cell = next(
        value
        for value in tree.plan.cells
        if value.required and tree.controls.scenarios_by_id[value.scenario_id].family == "cnn"
    )
    directory = tree.run_directory / "cells" / cell.cell_id
    checkpoint_manifest = json.loads(
        (directory / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    relative = checkpoint_manifest["checkpoints"][0]["versioned_outputs"][0][
        "commit_manifest_relative_path"
    ]
    sidecar = tree.run_directory.joinpath(*Path(relative).parts)
    same_bytes = sidecar.read_bytes()
    sidecar.chmod(0o666)
    sidecar.unlink()
    sidecar.write_bytes(same_bytes)
    sidecar.chmod(0o444)

    readback = _read_tree(tree)

    assert not readback.passed
    assert any("physical identity" in error for error in readback.errors), readback.errors


def test_schema_v3_rejects_hardlinked_canonical_and_versioned_copies(
    tmp_path: Path,
) -> None:
    tree = _build_confirmatory_tree(tmp_path)
    cell = next(
        value
        for value in tree.plan.cells
        if value.required and tree.controls.scenarios_by_id[value.scenario_id].family == "cnn"
    )
    directory = tree.run_directory / "cells" / cell.cell_id
    checkpoint_manifest = json.loads(
        (directory / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    record = checkpoint_manifest["checkpoints"][0]
    canonical = tree.run_directory.joinpath(
        *Path(record["canonical_working_checkpoint"]["path"]).parts
    )
    versioned = directory / record["path"]
    versioned.chmod(0o666)
    versioned.unlink()
    os.link(canonical, versioned)

    readback = _read_tree(tree)

    assert not readback.passed
    assert any(
        "exactly one filesystem link" in error or "copies differ or alias" in error
        for error in readback.errors
    ), readback.errors


def test_schema_v3_execution_manifest_policy_tamper_fails_after_rebinding(
    tmp_path: Path,
) -> None:
    tree = _build_confirmatory_tree(tmp_path)
    cell = next(
        value
        for value in tree.plan.cells
        if value.required and tree.controls.scenarios_by_id[value.scenario_id].family == "cnn"
    )
    directory = tree.run_directory / "cells" / cell.cell_id
    checkpoint_manifest_path = directory / "checkpoint_manifest.json"
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    record = checkpoint_manifest["checkpoints"][0]
    execution_manifest = directory / record["execution_manifest_path"]
    execution_payload = json.loads(execution_manifest.read_text(encoding="ascii"))
    execution_payload["automatic_retry_allowed"] = True
    execution_manifest.chmod(0o666)
    execution_manifest.unlink()
    _write_canonical_ascii_json(execution_manifest, execution_payload)
    execution_manifest.chmod(0o444)
    execution_identity = ConfirmatoryCheckpointPhysicalIdentity.from_stat(
        execution_manifest.lstat()
    )
    execution_sha256 = sha256_file(execution_manifest)
    record["execution_manifest_sha256"] = execution_sha256
    record["execution_manifest_physical_identity"] = execution_identity.as_dict()
    _write_json(checkpoint_manifest_path, checkpoint_manifest)

    telemetry_path = directory / "telemetry.json"
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    fold = next(
        value
        for value in telemetry["evidence"]["fold_evidence"]
        if value["fold_id"] == record["fold_id"]
    )
    fold["checkpoint_execution_manifest_sha256"] = execution_sha256
    fold["checkpoint_execution_manifest_physical_identity"] = execution_identity.as_dict()
    _write_json(telemetry_path, telemetry)
    _rehash_changed_cell(tree, cell.cell_id)

    readback = _read_tree(tree)

    assert not readback.passed
    assert any("checkpoint execution manifest" in error for error in readback.errors), (
        readback.errors
    )


def test_plaintext_checkpoint_cannot_pass_after_all_declared_hashes_are_rewritten(
    tmp_path: Path,
) -> None:
    tree = _build_confirmatory_tree(tmp_path)
    cell = next(
        value
        for value in tree.plan.cells
        if value.required and tree.controls.scenarios_by_id[value.scenario_id].family == "cnn"
    )
    directory = tree.run_directory / "cells" / cell.cell_id
    checkpoint_manifest_path = directory / "checkpoint_manifest.json"
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    record = checkpoint_manifest["checkpoints"][0]
    checkpoint_path = directory / record["path"]
    checkpoint_path.chmod(0o666)
    checkpoint_path.write_text(
        "not a torch checkpoint despite internally consistent JSON declarations\n",
        encoding="utf-8",
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    record["sha256"] = checkpoint_sha256
    _write_json(checkpoint_manifest_path, checkpoint_manifest)

    telemetry_path = directory / "telemetry.json"
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    fold = next(
        value
        for value in telemetry["evidence"]["fold_evidence"]
        if value["fold_id"] == record["fold_id"]
    )
    fold["checkpoint_sha256"] = checkpoint_sha256
    fold["checkpoint_size_bytes"] = checkpoint_path.stat().st_size
    _write_json(telemetry_path, telemetry)
    _rehash_changed_cell(tree, cell.cell_id)

    readback = _read_tree(tree)

    assert not readback.passed
    assert any(
        "differs from its hash/size/physical identity" in error for error in readback.errors
    ), readback.errors


def test_self_consistent_wrong_checkpoint_fingerprint_fails_preflight_binding(
    tmp_path: Path,
) -> None:
    tree = _build_confirmatory_tree(tmp_path)
    cell = next(
        value
        for value in tree.plan.cells
        if value.required and tree.controls.scenarios_by_id[value.scenario_id].family == "cnn"
    )
    directory = tree.run_directory / "cells" / cell.cell_id
    manifest_path = directory / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["checkpoints"][0]
    checkpoint_path = directory / record["path"]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["data_and_split_sha256"]["training_data_sha256"] = "0" * 64
    checkpoint_path.chmod(0o666)
    torch.save(payload, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    record["sha256"] = checkpoint_sha256
    _write_json(manifest_path, manifest)

    telemetry_path = directory / "telemetry.json"
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    fold = next(
        value
        for value in telemetry["evidence"]["fold_evidence"]
        if value["fold_id"] == record["fold_id"]
    )
    fold["data_and_split_sha256"]["training_data_sha256"] = "0" * 64
    fold["checkpoint_sha256"] = checkpoint_sha256
    fold["checkpoint_size_bytes"] = checkpoint_path.stat().st_size
    _write_json(telemetry_path, telemetry)
    _rehash_changed_cell(tree, cell.cell_id)

    readback = _read_tree(tree)

    assert not readback.passed
    assert any("pre-execution data/split bindings" in error for error in readback.errors)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("budget", "guided/random review budgets differ"),
        ("metric", "metrics differ from probabilities"),
    ],
)
def test_restoration_budget_and_metric_tamper_fail_after_rehash(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    tree = _build_confirmatory_tree(tmp_path)
    metrics_path = tree.run_directory / "restoration_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if mutation == "budget":
        evidence_path = tree.run_directory / "restoration_evidence.npz"
        arrays = _npz_arrays(evidence_path)
        key = next(name for name in arrays if name.endswith("__guided_reviewed_mask"))
        unreviewed = int(np.flatnonzero(~arrays[key])[0])
        arrays[key][unreviewed] = True
        np.savez(evidence_path, **arrays)
        metrics["evidence_sha256"] = sha256_file(evidence_path)
        certificate_path = tree.run_directory / "restoration_replay_certificate.json"
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        certificate["evidence_sha256"] = sha256_file(evidence_path)
        certificate["evidence_arrays"][key] = array_artifact_sha256(arrays[key])
        _write_json(certificate_path, certificate)
    else:
        guided = metrics["rotations"][0]["conditions"]["audit_guided_restoration"]
        guided["metrics"]["macro_f1"] += 0.1
    _write_json(metrics_path, metrics)
    _rehash_root_manifest(tree)

    readback = _read_tree(tree)

    assert not readback.passed
    assert any(message in error for error in readback.errors)


def test_restoration_replay_certificate_tamper_fails_after_root_rehash(
    tmp_path: Path,
) -> None:
    tree = _build_confirmatory_tree(tmp_path)
    path = tree.run_directory / "restoration_replay_certificate.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["bridge_partition_content_sha256"] = "0" * 64
    _write_json(path, payload)
    _rehash_root_manifest(tree)

    readback = _read_tree(tree)

    assert not readback.passed
    assert any("replay certificate" in error for error in readback.errors)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("machine_contract", "machine-readable contract differs"),
        ("random_interval", "required scientific result fields"),
    ],
)
def test_report_semantic_tamper_fails_after_root_rehash(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    tree = _build_confirmatory_tree(tmp_path)
    path = tree.run_directory / "report.md"
    report = path.read_text(encoding="utf-8")
    if mutation == "machine_contract":
        report = report.replace(
            '"probability_positive":',
            '"probability_positive_tampered":',
            1,
        )
    else:
        report = report.replace(
            "Random mean macro F1 (95% interval)",
            "Random mean macro F1",
            1,
        )
    path.write_text(report, encoding="utf-8")
    _rehash_root_manifest(tree)

    readback = _read_tree(tree)

    assert not readback.passed
    assert any(message in error for error in readback.errors)


def test_absent_root_aggregate_blocks_completion(tmp_path: Path) -> None:
    tree = _build_confirmatory_tree(tmp_path)
    (tree.run_directory / "fold_aggregate.json").unlink()

    readback = _read_tree(tree)

    assert not readback.passed
    assert any("fold aggregate is missing" in error for error in readback.errors)


def test_missing_filesystem_fold_rotation_cannot_enable_stage(tmp_path: Path) -> None:
    tree = _build_confirmatory_tree(tmp_path)
    index = tree.run_directory / "cell_index.csv"
    rows = _read_cell_index(index)
    _write_cell_index(index, [row for row in rows if row["outer_fold"] != "3"])

    readback = _read_tree(tree)

    assert not readback.passed
    assert any(
        "cell index row count differs from frozen matrix" in error for error in readback.errors
    )
    with pytest.raises(ValueError, match="filesystem readback did not pass"):
        build_confirmatory_completion_evidence(
            plan=tree.plan,
            reconciliation=tree.reconciliation,
            artifact_scope=REAL_CONFIRMATORY_ARTIFACT_SCOPE,
            study_outcome_eligible=True,
            gate_evidence=tree.gate,
            run_directory=tree.run_directory,
        )
