from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import shutil
import stat
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from histo_audit.cross_validation.image_oof import (
    ConfirmatoryCheckpointContractError,
    ConfirmatoryCheckpointDirective,
    ConfirmatoryCheckpointExecutionContract,
    ConfirmatoryCheckpointFileIdentity,
    ConfirmatoryCheckpointPhysicalIdentity,
    _OwnedCheckpointIO,
    _register_checkpoint_execution_contract,
    grouped_oof_confirmatory_cnn,
)
from histo_audit.cross_validation.oof import make_group_stratified_folds
from histo_audit.models.cnn import (
    CPU_TEST_ONLY_WEIGHT_IDENTIFIER,
    OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER,
    ConfirmatoryCNNConfig,
    confirmatory_cnn_data_and_split_sha256,
)

_DATA_AND_SPLIT_FINGERPRINT_KEYS = {
    "training_data_sha256",
    "reference_validation_data_sha256",
    "training_split_sha256",
    "reference_validation_split_sha256",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _select_rows(value: Any, indices: NDArray[np.int64]) -> Any:
    if value is None:
        return None
    if hasattr(value, "select_rows"):
        return value.select_rows(indices)
    return np.asarray(value)[indices]


def _physical_identity(path: Path) -> ConfirmatoryCheckpointPhysicalIdentity:
    return ConfirmatoryCheckpointPhysicalIdentity.from_stat(path.stat())


def _authorise_fresh(
    arguments: dict[str, Any],
    run_directory: Path,
    *,
    cell_id: str = "test_cnn_cell",
) -> None:
    checkpoint_directory = run_directory / "cells" / cell_id / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=False)
    config = arguments["base_config"]
    folds = make_group_stratified_folds(
        arguments["pre_corruption_labels"],
        tuple(str(value) for value in arguments["group_ids"]),
        n_splits=arguments["n_splits"],
        class_order=(0, 1, 2, 3, 4),
        seed=arguments["split_seed"],
    )
    directives: list[ConfirmatoryCheckpointDirective] = []
    for fold in folds:
        fold_config = replace(config, seed=config.seed + fold.fold_id)
        configuration = asdict(fold_config)
        data_and_split = confirmatory_cnn_data_and_split_sha256(
            _select_rows(arguments["audit_rgb"], fold.train_indices),
            np.asarray(arguments["observed_labels"])[fold.train_indices],
            training_sample_ids=np.asarray(arguments["sample_ids"])[fold.train_indices],
            training_group_ids=np.asarray(arguments["group_ids"])[fold.train_indices],
            training_target_masks=_select_rows(
                arguments["audit_target_masks"],
                fold.train_indices,
            ),
            reference_validation_images=arguments["reference_validation_rgb"],
            reference_validation_labels=arguments["reference_validation_labels"],
            reference_validation_sample_ids=arguments["reference_validation_sample_ids"],
            reference_validation_group_ids=arguments["reference_validation_group_ids"],
            reference_validation_target_masks=arguments["reference_validation_target_masks"],
            input_variant=config.input_variant,
        )
        directives.append(
            ConfirmatoryCheckpointDirective(
                execution_mode="fresh",
                cell_id=cell_id,
                fold_id=fold.fold_id,
                action="fresh_fit",
                source_predecessor_checkpoint=None,
                destination_imported_checkpoint=None,
                versioned_checkpoint_output_directory_relative_path=(
                    f"cells/{cell_id}/checkpoint_versions/fold_{fold.fold_id:02d}"
                ),
                checkpoint_execution_manifest_relative_path=(
                    f"cells/{cell_id}/checkpoint_execution/fold_{fold.fold_id:02d}.json"
                ),
                checkpoint_sha256=None,
                checkpoint_size_bytes=None,
                completed_epochs_before_fit=0,
                stopped_early_before_fit=None,
                next_epoch_index=0,
                maximum_epochs=config.epochs,
                expected_configuration_json=_canonical_json(configuration),
                expected_configuration_sha256=_canonical_sha256(configuration),
                expected_model_metadata_json=_canonical_json({}),
                expected_model_metadata_sha256=_canonical_sha256({}),
                expected_data_and_split_json=_canonical_json(data_and_split),
                expected_data_and_split_sha256=_canonical_sha256(data_and_split),
            )
        )
    contract = ConfirmatoryCheckpointExecutionContract(
        execution_mode="fresh",
        contract_profile="cpu_test_only",
        retry_of_run_id=None,
        directives=tuple(directives),
        directives_sha256=_canonical_sha256([directive.as_dict() for directive in directives]),
        predecessor_checkpoint_read_performed=False,
        predecessor_checkpoint_copy_performed=False,
        outcome_artifacts_read=False,
        automatic_retry_allowed=False,
    )
    arguments["cell_id"] = cell_id
    arguments["checkpoint_directory"] = checkpoint_directory
    arguments["checkpoint_execution_contract"] = _register_checkpoint_execution_contract(
        contract,
        expected_directive_count=len(directives),
    )


def _authorise_successor(
    arguments: dict[str, Any],
    run_directory: Path,
    *,
    completed_epochs: dict[int, int],
    stopped_early: dict[int, bool] | None = None,
    successful_steps: dict[int, int] | None = None,
    cell_id: str = "test_cnn_cell",
) -> dict[int, str]:
    _authorise_fresh(arguments, run_directory, cell_id=cell_id)
    fresh_contract = arguments["checkpoint_execution_contract"]
    stopped = stopped_early or {}
    steps = successful_steps or {}
    directives: list[ConfirmatoryCheckpointDirective] = []
    before_hashes: dict[int, str] = {}
    for fresh in fresh_contract.directives:
        completed = completed_epochs[fresh.fold_id]
        was_stopped = stopped.get(fresh.fold_id, False)
        configuration = json.loads(fresh.expected_configuration_json)
        data_and_split = json.loads(fresh.expected_data_and_split_json)
        payload = {
            "configuration": configuration,
            "configuration_sha256": _canonical_sha256(configuration),
            "data_and_split_sha256": data_and_split,
            "completed_epochs": completed,
            "early_stopping_state": {"stopped_early": was_stopped},
            "telemetry": {
                "successful_optimiser_steps": steps.get(
                    fresh.fold_id,
                    completed,
                ),
                "skipped_optimiser_steps": 0,
            },
            "history": [{"epoch": epoch_index} for epoch_index in range(1, completed + 1)],
        }
        source_checkpoint = (
            run_directory
            / "synthetic-predecessor"
            / f"cells/{cell_id}/checkpoints/fold_{fresh.fold_id:02d}.pt"
        )
        source_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, source_checkpoint)
        checkpoint = arguments["checkpoint_directory"] / (f"fold_{fresh.fold_id:02d}.pt")
        shutil.copyfile(source_checkpoint, checkpoint)
        source_checkpoint.chmod(0o444)
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        destination_identity = _physical_identity(checkpoint)
        source_identity = _physical_identity(source_checkpoint)
        assert source_identity.file_id_128 != destination_identity.file_id_128
        before_hashes[fresh.fold_id] = checkpoint_sha256
        terminal = was_stopped or completed == fresh.maximum_epochs
        directives.append(
            replace(
                fresh,
                execution_mode="successor_resume",
                action=(
                    "restore_terminal_checkpoint_without_fit"
                    if terminal
                    else "resume_incomplete_fit"
                ),
                checkpoint_sha256=checkpoint_sha256,
                checkpoint_size_bytes=checkpoint.stat().st_size,
                source_predecessor_checkpoint=ConfirmatoryCheckpointFileIdentity(
                    path=source_checkpoint.resolve(),
                    physical_identity=source_identity,
                    size_bytes=checkpoint.stat().st_size,
                    sha256=checkpoint_sha256,
                ),
                destination_imported_checkpoint=ConfirmatoryCheckpointFileIdentity(
                    path=checkpoint.resolve(),
                    physical_identity=destination_identity,
                    size_bytes=checkpoint.stat().st_size,
                    sha256=checkpoint_sha256,
                ),
                completed_epochs_before_fit=completed,
                stopped_early_before_fit=was_stopped,
                next_epoch_index=completed,
                versioned_checkpoint_output_directory_relative_path=(
                    None
                    if terminal
                    else (f"cells/{cell_id}/checkpoint_versions/fold_{fresh.fold_id:02d}")
                ),
            )
        )
    contract = ConfirmatoryCheckpointExecutionContract(
        execution_mode="successor_resume",
        contract_profile="cpu_test_only",
        retry_of_run_id="synthetic_predecessor",
        directives=tuple(directives),
        directives_sha256=_canonical_sha256([directive.as_dict() for directive in directives]),
        predecessor_checkpoint_read_performed=True,
        predecessor_checkpoint_copy_performed=True,
        outcome_artifacts_read=False,
        automatic_retry_allowed=False,
        predecessor_snapshot_sha256="a" * 64,
        predecessor_copy_receipt_sha256="b" * 64,
    )
    arguments["checkpoint_execution_contract"] = _register_checkpoint_execution_contract(
        contract,
        expected_directive_count=len(directives),
    )
    return before_hashes


def _inputs(*, input_variant: str = "context_rgb_plus_binary_target_mask") -> dict[str, Any]:
    rng = np.random.default_rng(901)
    labels = np.tile(np.arange(5, dtype=np.int64), 4)
    groups = np.repeat(np.asarray([f"audit_{index}" for index in range(4)]), 5)
    rgb = rng.integers(0, 256, size=(20, 16, 16, 3), dtype=np.uint8)
    masks = np.zeros((20, 16, 16), dtype=bool)
    masks[:, 5:11, 5:11] = True
    validation_rgb = rng.integers(0, 256, size=(5, 16, 16, 3), dtype=np.uint8)
    validation_masks = np.zeros((5, 16, 16), dtype=bool)
    validation_masks[:, 4:12, 4:12] = True
    uses_masks = input_variant == "context_rgb_plus_binary_target_mask"
    return {
        "audit_rgb": rgb,
        "observed_labels": labels.copy(),
        "pre_corruption_labels": labels.copy(),
        "group_ids": groups,
        "sample_ids": np.asarray([f"audit_sample_{index:02d}" for index in range(20)]),
        "audit_target_masks": masks if uses_masks else None,
        "reference_validation_rgb": validation_rgb,
        "reference_validation_labels": np.arange(5, dtype=np.int64),
        "reference_validation_sample_ids": np.asarray(
            [f"validation_sample_{index}" for index in range(5)]
        ),
        "reference_validation_group_ids": np.repeat("validation_group", 5),
        "reference_validation_target_masks": validation_masks if uses_masks else None,
        "final_reference_group_ids": {"final_group"},
        "base_config": ConfirmatoryCNNConfig(
            input_variant=input_variant,  # type: ignore[arg-type]
            weight_identifier=CPU_TEST_ONLY_WEIGHT_IDENTIFIER,
            input_size=32,
            epochs=1,
            batch_size=5,
            minimum_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            early_stopping_patience=3,
            seed=307,
        ),
        "cpu_test_only": True,
        "n_splits": 2,
        "split_seed": 41,
    }


class _FastCPUTestAdapter:
    """Small deterministic stand-in used only for split-contract tests."""

    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, config: ConfirmatoryCNNConfig) -> None:
        self.config = config
        self.classes_ = np.arange(5, dtype=np.int64)
        self.telemetry_: dict[str, Any] = {}
        self.model_metadata_: dict[str, Any] = {}
        self.data_and_split_sha256_: dict[str, str] = {}
        self.completed_epochs_ = 0
        self.best_epoch_: int | None = None
        self.best_validation_loss_: float | None = None

    def fit(
        self,
        training_images: NDArray[np.generic],
        observed_training_labels: NDArray[np.generic],
        **kwargs: Any,
    ) -> Self:
        checkpoint_path = Path(kwargs["checkpoint_path"])
        resumed = bool(kwargs["resume"])
        self.completed_epochs_ = self.config.epochs
        checkpoint_path.write_bytes(f"seed={self.config.seed};resume={resumed}".encode("ascii"))
        self.best_epoch_ = self.config.epochs
        self.best_validation_loss_ = 1.25
        self.telemetry_ = {
            "study_outcome_eligible": False,
            "execution_mode": "cpu_test_only_non_evidence",
            "successful_optimiser_steps": self.config.epochs,
        }
        self.model_metadata_ = {"input_variant": self.config.input_variant}
        self.data_and_split_sha256_ = confirmatory_cnn_data_and_split_sha256(
            training_images,
            observed_training_labels,
            training_sample_ids=kwargs["training_sample_ids"],
            training_group_ids=kwargs["training_group_ids"],
            training_target_masks=kwargs["training_target_masks"],
            reference_validation_images=kwargs["reference_validation_images"],
            reference_validation_labels=kwargs["reference_validation_labels"],
            reference_validation_sample_ids=kwargs["reference_validation_sample_ids"],
            reference_validation_group_ids=kwargs["reference_validation_group_ids"],
            reference_validation_target_masks=kwargs["reference_validation_target_masks"],
            input_variant=self.config.input_variant,
        )
        self.calls.append(
            {
                "seed": self.config.seed,
                "training_labels": np.asarray(observed_training_labels).copy(),
                "training_groups": tuple(str(value) for value in kwargs["training_group_ids"]),
                "resume": resumed,
                "image_count": len(training_images),
            }
        )
        return self

    def predict_proba(
        self,
        images: NDArray[np.generic],
        *,
        target_masks: NDArray[np.generic] | None = None,
    ) -> NDArray[np.float64]:
        del target_masks
        row = np.asarray([0.05, 0.10, 0.15, 0.20, 0.50], dtype=np.float64)
        return np.tile(row, (len(images), 1))


class _FastStudyAdapter(_FastCPUTestAdapter):
    def fit(
        self,
        training_images: NDArray[np.generic],
        observed_training_labels: NDArray[np.generic],
        **kwargs: Any,
    ) -> Self:
        super().fit(training_images, observed_training_labels, **kwargs)
        self.telemetry_["study_outcome_eligible"] = True
        self.telemetry_["execution_mode"] = "real_study_cuda"
        return self


class _CheckpointStateAdapter(_FastCPUTestAdapter):
    """Resume stand-in that never rewrites a terminal checkpoint."""

    def fit(
        self,
        training_images: NDArray[np.generic],
        observed_training_labels: NDArray[np.generic],
        **kwargs: Any,
    ) -> Self:
        checkpoint_path = Path(kwargs["checkpoint_path"])
        assert kwargs["resume"] is True
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        completed_before = int(payload["completed_epochs"])
        stopped_before = bool(payload["early_stopping_state"]["stopped_early"])
        steps_before = int(payload["telemetry"]["successful_optimiser_steps"])
        history = list(payload["history"])
        if not stopped_before and completed_before < self.config.epochs:
            for epoch_index in range(completed_before + 1, self.config.epochs + 1):
                history.append({"epoch": epoch_index})
            payload["completed_epochs"] = self.config.epochs
            payload["history"] = history
            payload["telemetry"]["successful_optimiser_steps"] = (
                steps_before + self.config.epochs - completed_before
            )
            torch.save(payload, checkpoint_path)
        self.completed_epochs_ = int(payload["completed_epochs"])
        self.stopped_early_ = stopped_before
        self.history_ = tuple(payload["history"])
        self.best_epoch_ = self.completed_epochs_
        self.best_validation_loss_ = 1.0
        self.data_and_split_sha256_ = confirmatory_cnn_data_and_split_sha256(
            training_images,
            observed_training_labels,
            training_sample_ids=kwargs["training_sample_ids"],
            training_group_ids=kwargs["training_group_ids"],
            training_target_masks=kwargs["training_target_masks"],
            reference_validation_images=kwargs["reference_validation_images"],
            reference_validation_labels=kwargs["reference_validation_labels"],
            reference_validation_sample_ids=kwargs["reference_validation_sample_ids"],
            reference_validation_group_ids=kwargs["reference_validation_group_ids"],
            reference_validation_target_masks=kwargs["reference_validation_target_masks"],
            input_variant=self.config.input_variant,
        )
        successful_steps = int(payload["telemetry"]["successful_optimiser_steps"])
        self.telemetry_ = {
            "study_outcome_eligible": False,
            "execution_mode": "cpu_test_only_non_evidence",
            "successful_optimiser_steps": successful_steps,
        }
        self.model_metadata_ = {}
        self.calls.append(
            {
                "seed": self.config.seed,
                "resume": True,
                "completed_before": completed_before,
                "completed_after": self.completed_epochs_,
            }
        )
        return self


class _FreshAppearanceAdapter(_FastCPUTestAdapter):
    def fit(
        self,
        training_images: NDArray[np.generic],
        observed_training_labels: NDArray[np.generic],
        **kwargs: Any,
    ) -> Self:
        Path(kwargs["checkpoint_path"]).write_bytes(b"foreign-racing-checkpoint")
        return super().fit(training_images, observed_training_labels, **kwargs)


class _SuccessorRaceAdapter(_CheckpointStateAdapter):
    mutation: ClassVar[str] = "swap"
    loaded_completed_epochs: ClassVar[list[int]] = []
    alias_path: ClassVar[Path | None] = None

    def fit(
        self,
        training_images: NDArray[np.generic],
        observed_training_labels: NDArray[np.generic],
        **kwargs: Any,
    ) -> Self:
        path = Path(kwargs["checkpoint_path"])
        assert kwargs["resume"] is True
        pinned_bytes = path.read_bytes()
        pinned = torch.load(
            io.BytesIO(pinned_bytes),
            map_location="cpu",
            weights_only=True,
        )
        self.loaded_completed_epochs.append(int(pinned["completed_epochs"]))
        if self.mutation == "swap":
            path.unlink()
            path.write_bytes(b"foreign-swapped-checkpoint")
        elif self.mutation == "same_bytes_swap":
            path.unlink()
            path.write_bytes(pinned_bytes)
        elif self.mutation == "delete":
            path.unlink()
        elif self.mutation == "alias":
            alias = path.parent / "foreign-hardlink-alias.pt"
            os.link(path, alias)
            type(self).alias_path = alias
        else:
            raise AssertionError(f"unsupported race mutation: {self.mutation}")
        return super().fit(training_images, observed_training_labels, **kwargs)


class _ResumeCheckpointFailureAdapter(_FastCPUTestAdapter):
    failure: ClassVar[str] = "value"

    def fit(
        self,
        training_images: NDArray[np.generic],
        observed_training_labels: NDArray[np.generic],
        **kwargs: Any,
    ) -> Self:
        del training_images, observed_training_labels
        assert kwargs["resume"] is True
        if self.failure == "missing":
            raise FileNotFoundError("synthetic checkpoint-originated disappearance")
        raise ValueError("synthetic checkpoint-originated restore failure")


class _FreshCheckpointFailureAdapter(_FastCPUTestAdapter):
    def fit(
        self,
        training_images: NDArray[np.generic],
        observed_training_labels: NDArray[np.generic],
        **kwargs: Any,
    ) -> Self:
        del training_images, observed_training_labels, kwargs
        raise ValueError("synthetic fresh model failure")


def test_api_cannot_accept_final_test_images_or_labels() -> None:
    names = set(inspect.signature(grouped_oof_confirmatory_cnn).parameters)
    assert "final_reference_group_ids" in names
    assert not any("final" in name and name != "final_reference_group_ids" for name in names)


def test_pre_corruption_labels_fix_fold_membership_when_observed_labels_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "histo_audit.cross_validation.image_oof.ConfirmatoryCNNCPUTestOnlyAdapter",
        _FastCPUTestAdapter,
    )
    first_arguments = _inputs(input_variant="context_rgb")
    _authorise_fresh(first_arguments, tmp_path / "first")
    first = grouped_oof_confirmatory_cnn(**first_arguments)

    second_arguments = _inputs(input_variant="context_rgb")
    observed = second_arguments["observed_labels"].copy()
    observed[:5] = (observed[:5] + 1) % 5
    second_arguments["observed_labels"] = observed
    _authorise_fresh(second_arguments, tmp_path / "second")
    second = grouped_oof_confirmatory_cnn(**second_arguments)

    np.testing.assert_array_equal(first.oof_result.fold_id, second.oof_result.fold_id)
    np.testing.assert_array_equal(
        first.oof_result.fold_assignment_labels,
        second.oof_result.fold_assignment_labels,
    )
    assert first.oof_result.fold_assignment_label_source == "pre_corruption_label"
    assert (
        first.oof_result.fold_assignment_labels_sha256
        == second.oof_result.fold_assignment_labels_sha256
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reference_validation_group_ids", np.repeat("audit_0", 5), "audit/reference"),
        ("final_reference_group_ids", {"audit_1"}, "final-reference groups present"),
        (
            "final_reference_group_ids",
            {"validation_group"},
            "reference-validation/final-reference",
        ),
    ],
)
def test_rejects_reference_validation_and_final_group_leakage(
    field: str,
    value: Any,
    message: str,
    tmp_path: Path,
) -> None:
    arguments = _inputs()
    arguments[field] = value
    _authorise_fresh(arguments, tmp_path / field)
    with pytest.raises(ValueError, match=message):
        grouped_oof_confirmatory_cnn(**arguments)


@pytest.mark.parametrize(
    "input_variant",
    ["context_rgb", "context_rgb_plus_binary_target_mask"],
)
def test_real_cpu_adapter_covers_rgb_variants_and_is_permanently_ineligible(
    input_variant: str,
    tmp_path: Path,
) -> None:
    arguments = _inputs(input_variant=input_variant)
    _authorise_fresh(arguments, tmp_path / input_variant)
    result = grouped_oof_confirmatory_cnn(**arguments)

    assert result.oof_result.probabilities.shape == (20, 5)
    np.testing.assert_array_equal(result.oof_result.coverage_count, np.ones(20, dtype=np.int64))
    assert result.oof_result.class_order == (0, 1, 2, 3, 4)
    assert result.execution_mode == "cpu_test_only_non_evidence"
    assert result.study_outcome_eligible is False
    assert {row.model_seed for row in result.fold_evidence} == {307, 308}
    assert all(row.study_outcome_eligible is False for row in result.fold_evidence)
    assert all(len(row.checkpoint_sha256) == 64 for row in result.fold_evidence)
    assert all(Path(row.checkpoint_path).is_file() for row in result.fold_evidence)
    for row in result.fold_evidence:
        assert set(row.data_and_split_sha256) == _DATA_AND_SPLIT_FINGERPRINT_KEYS
        assert all(len(value) == 64 for value in row.data_and_split_sha256.values())
        assert not set(row.training_groups).intersection(row.held_out_groups)
        assert not set(row.training_groups).intersection(row.reference_validation_groups)


def test_fresh_contract_rejects_existing_fold_instead_of_inferring_resume(
    tmp_path: Path,
) -> None:
    arguments = _inputs(input_variant="context_rgb")
    _authorise_fresh(arguments, tmp_path / "fresh")
    checkpoint = arguments["checkpoint_directory"] / "fold_00.pt"
    checkpoint.write_bytes(b"unauthorised-existing-checkpoint")

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="not absent",
    ):
        grouped_oof_confirmatory_cnn(**arguments)


def test_fresh_working_checkpoint_is_frozen_only_at_the_fold_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "histo_audit.cross_validation.image_oof.ConfirmatoryCNNCPUTestOnlyAdapter",
        _FreshAppearanceAdapter,
    )
    arguments = _inputs(input_variant="context_rgb")
    _authorise_fresh(arguments, tmp_path / "fresh-race")
    result = grouped_oof_confirmatory_cnn(**arguments)

    canonical = arguments["checkpoint_directory"] / "fold_00.pt"
    versioned = Path(result.fold_evidence[0].checkpoint_path)
    assert canonical.read_bytes() == versioned.read_bytes()
    assert _physical_identity(canonical).file_id_128 != _physical_identity(versioned).file_id_128
    assert stat.S_IMODE(canonical.stat().st_mode) & 0o222 == 0
    assert stat.S_IMODE(versioned.stat().st_mode) & 0o222 == 0
    assert not list(arguments["checkpoint_directory"].glob(".*.tmp"))
    assert not list(arguments["checkpoint_directory"].glob("*.aanca-owner.lock"))


def test_fresh_checkpoint_publication_is_versioned_append_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "histo_audit.cross_validation.image_oof.ConfirmatoryCNNCPUTestOnlyAdapter",
        _FastCPUTestAdapter,
    )
    arguments = _inputs(input_variant="context_rgb")
    _authorise_fresh(arguments, tmp_path / "append-only")

    result = grouped_oof_confirmatory_cnn(**arguments)

    for row in result.fold_evidence:
        checkpoint = Path(row.checkpoint_path)
        logical = arguments["checkpoint_directory"] / f"fold_{row.fold_id:02d}.pt"
        manifest = checkpoint.with_name(checkpoint.name.removesuffix(".pt") + ".commit.json")
        execution_manifest = Path(row.checkpoint_execution_manifest_path)
        assert checkpoint.name == "epoch_0001.pt"
        assert checkpoint.parent == (
            arguments["checkpoint_directory"].parent
            / "checkpoint_versions"
            / f"fold_{row.fold_id:02d}"
        )
        assert checkpoint.is_file()
        assert checkpoint.stat().st_nlink == 1
        assert logical.is_file()
        assert logical.read_bytes() == checkpoint.read_bytes()
        assert _physical_identity(logical).file_id_128 != _physical_identity(checkpoint).file_id_128
        assert stat.S_IMODE(logical.stat().st_mode) & 0o222 == 0
        assert stat.S_IMODE(checkpoint.stat().st_mode) & 0o222 == 0
        commit = json.loads(manifest.read_text(encoding="ascii"))
        assert commit["policy"] == "aanca_fold_boundary_checkpoint_commit_v2"
        assert commit["canonical_working_checkpoint"]["path"] == str(logical.resolve())
        assert commit["versioned_checkpoint"]["path"] == str(checkpoint.resolve())
        assert commit["versioned_checkpoint"]["file_id_128"] == (
            _physical_identity(checkpoint).file_id_128
        )
        assert commit["mutable_latest_path_created"] is False
        assert commit["automatic_retry_allowed"] is False
        execution = json.loads(execution_manifest.read_text(encoding="ascii"))
        assert execution["schema_version"] == 3
        assert execution["policy"] == "aanca_fold_boundary_checkpoint_execution_v3"
        assert execution["action"] == "fresh_fit"
        assert execution["imported_checkpoint_observed"] is None
        assert len(execution["versioned_outputs"]) == 1
        assert execution["canonical_working_checkpoint"]["path"] == str(logical.resolve())
        assert execution["final_checkpoint"]["path"] == str(checkpoint.resolve())


def test_unqualified_destination_race_is_not_overwritten(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "histo_audit.cross_validation.image_oof.ConfirmatoryCNNCPUTestOnlyAdapter",
        _FastCPUTestAdapter,
    )
    arguments = _inputs(input_variant="context_rgb")
    _authorise_fresh(arguments, tmp_path / "destination-race")
    from histo_audit.cross_validation import image_oof as module

    real_create = module._create_checkpoint_descriptor
    raced: list[Path] = []

    def race_create(destination: Path) -> Any:
        target = Path(destination)
        if not raced and target.name == "epoch_0001.pt":
            target.write_bytes(b"foreign-unqualified-writer")
            raced.append(target)
        return real_create(destination)

    monkeypatch.setattr(module, "_create_checkpoint_descriptor", race_create)
    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="immutable no-overwrite publication failed",
    ):
        grouped_oof_confirmatory_cnn(**arguments)

    assert len(raced) == 1
    assert raced[0].read_bytes() == b"foreign-unqualified-writer"
    assert not list(arguments["checkpoint_directory"].glob(".*.tmp"))


def test_fold_boundary_publication_never_uses_a_hardlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "histo_audit.cross_validation.image_oof.ConfirmatoryCNNCPUTestOnlyAdapter",
        _FastCPUTestAdapter,
    )
    arguments = _inputs(input_variant="context_rgb")
    _authorise_fresh(arguments, tmp_path / "no-hardlink")

    def forbidden_link(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("checkpoint publication attempted to use os.link")

    monkeypatch.setattr(os, "link", forbidden_link)
    result = grouped_oof_confirmatory_cnn(**arguments)

    for row in result.fold_evidence:
        canonical = arguments["checkpoint_directory"] / f"fold_{row.fold_id:02d}.pt"
        versioned = Path(row.checkpoint_path)
        assert canonical.read_bytes() == versioned.read_bytes()
        assert (
            _physical_identity(canonical).file_id_128 != _physical_identity(versioned).file_id_128
        )


def test_same_byte_swap_after_publication_is_not_adopted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "histo_audit.cross_validation.image_oof.ConfirmatoryCNNCPUTestOnlyAdapter",
        _FastCPUTestAdapter,
    )
    arguments = _inputs(input_variant="context_rgb")
    _authorise_fresh(arguments, tmp_path / "post-publication-race")
    from histo_audit.cross_validation import image_oof as module

    original_read = module._read_private_checkpoint_bytes
    swapped: list[Path] = []

    def swap_before_final_read(path: Path, *, role: str) -> Any:
        if role == "post-publication immutable checkpoint" and not swapped:
            payload = path.read_bytes()
            path.chmod(stat.S_IWRITE | stat.S_IREAD)
            path.unlink()
            path.write_bytes(payload)
            swapped.append(path)
        return original_read(path, role=role)

    monkeypatch.setattr(module, "_read_private_checkpoint_bytes", swap_before_final_read)
    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="copies changed",
    ):
        grouped_oof_confirmatory_cnn(**arguments)

    assert len(swapped) == 1
    assert swapped[0].is_file()


def test_fresh_model_failure_is_not_silently_retried(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "histo_audit.cross_validation.image_oof.ConfirmatoryCNNCPUTestOnlyAdapter",
        _FreshCheckpointFailureAdapter,
    )
    arguments = _inputs(input_variant="context_rgb")
    _authorise_fresh(arguments, tmp_path / "fresh-checkpoint-error")

    with pytest.raises(ValueError, match="synthetic fresh model failure"):
        grouped_oof_confirmatory_cnn(**arguments)


@pytest.mark.parametrize("mutation", ["swap", "delete", "alias"])
def test_successor_race_after_pin_fails_without_reopen_or_overwrite(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "histo_audit.cross_validation.image_oof.ConfirmatoryCNNCPUTestOnlyAdapter",
        _SuccessorRaceAdapter,
    )
    _SuccessorRaceAdapter.mutation = mutation
    _SuccessorRaceAdapter.loaded_completed_epochs.clear()
    _SuccessorRaceAdapter.alias_path = None
    arguments = _inputs(input_variant="context_rgb")
    arguments["base_config"] = replace(arguments["base_config"], epochs=2)
    _authorise_successor(
        arguments,
        tmp_path / f"resume-race-{mutation}",
        completed_epochs={0: 1, 1: 2},
    )
    checkpoint = arguments["checkpoint_directory"] / "fold_00.pt"

    with pytest.raises(ConfirmatoryCheckpointContractError):
        grouped_oof_confirmatory_cnn(**arguments)

    assert _SuccessorRaceAdapter.loaded_completed_epochs == [1]
    if mutation == "swap":
        assert checkpoint.read_bytes() == b"foreign-swapped-checkpoint"
    elif mutation == "same_bytes_swap":
        assert checkpoint.exists()
    elif mutation == "delete":
        assert not checkpoint.exists()
    else:
        assert checkpoint.exists()
        assert _SuccessorRaceAdapter.alias_path is not None
        assert _SuccessorRaceAdapter.alias_path.exists()
    assert not list(arguments["checkpoint_directory"].glob("*.aanca-owner.lock"))


def test_successor_same_byte_race_after_verify_preserves_foreign_predecessor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "histo_audit.cross_validation.image_oof.ConfirmatoryCNNCPUTestOnlyAdapter",
        _CheckpointStateAdapter,
    )
    arguments = _inputs(input_variant="context_rgb")
    arguments["base_config"] = replace(arguments["base_config"], epochs=2)
    _authorise_successor(
        arguments,
        tmp_path / "verify-link-race",
        completed_epochs={0: 1, 1: 2},
    )
    directive = arguments["checkpoint_execution_contract"].directives[0]
    assert directive.source_predecessor_checkpoint is not None
    predecessor = directive.source_predecessor_checkpoint.path
    expected_bytes = predecessor.read_bytes()
    original_identity = _physical_identity(predecessor)
    from histo_audit.cross_validation import image_oof as module

    original_read = module._read_private_checkpoint_bytes
    raced = False
    replacement_was_blocked = False
    replacement = tmp_path / "same-byte-predecessor-replacement.pt"
    replacement.write_bytes(expected_bytes)

    def race_before_source_recheck(path: Path, *, role: str) -> Any:
        nonlocal raced, replacement_was_blocked
        if (
            not raced
            and path == predecessor
            and role == "pre-publication immutable predecessor source checkpoint"
        ):
            try:
                os.replace(replacement, predecessor)
            except OSError:
                replacement_was_blocked = True
            raced = True
        return original_read(path, role=role)

    monkeypatch.setattr(
        module,
        "_read_private_checkpoint_bytes",
        race_before_source_recheck,
    )
    if os.name == "nt":
        grouped_oof_confirmatory_cnn(**arguments)
    else:
        with pytest.raises(ConfirmatoryCheckpointContractError):
            grouped_oof_confirmatory_cnn(**arguments)

    assert raced
    assert replacement_was_blocked is (os.name == "nt")
    assert predecessor.read_bytes() == expected_bytes
    if os.name == "nt":
        assert _physical_identity(predecessor) == original_identity
    else:
        assert _physical_identity(predecessor) != original_identity


def test_terminal_restore_detects_swap_even_without_checkpoint_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "histo_audit.cross_validation.image_oof.ConfirmatoryCNNCPUTestOnlyAdapter",
        _SuccessorRaceAdapter,
    )
    _SuccessorRaceAdapter.mutation = "swap"
    _SuccessorRaceAdapter.loaded_completed_epochs.clear()
    arguments = _inputs(input_variant="context_rgb")
    arguments["base_config"] = replace(arguments["base_config"], epochs=2)
    _authorise_successor(
        arguments,
        tmp_path / "terminal-swap",
        completed_epochs={0: 2, 1: 2},
    )
    checkpoint = arguments["checkpoint_directory"] / "fold_00.pt"

    with pytest.raises(ConfirmatoryCheckpointContractError):
        grouped_oof_confirmatory_cnn(**arguments)

    assert _SuccessorRaceAdapter.loaded_completed_epochs == [2]
    assert checkpoint.read_bytes() == b"foreign-swapped-checkpoint"


def _publish_synthetic_boundary(
    owner: _OwnedCheckpointIO,
    logical: Path,
    payload: bytes,
    *,
    completed_epochs: int,
    trained_epochs: int,
) -> None:
    logical.write_bytes(payload)
    working = owner.read_working_checkpoint()
    owner.publish_completed_working_checkpoint(
        working,
        completed_epochs=completed_epochs,
        trained_epochs=trained_epochs,
    )


def test_checkpoint_owner_lock_rejects_a_second_qualified_writer(tmp_path: Path) -> None:
    arguments = _inputs(input_variant="context_rgb")
    _authorise_fresh(arguments, tmp_path / "double-owner")
    directive = arguments["checkpoint_execution_contract"].directives[0]
    checkpoint = arguments["checkpoint_directory"] / "fold_00.pt"
    first = _OwnedCheckpointIO.acquire(checkpoint, directive)
    try:
        with pytest.raises(
            ConfirmatoryCheckpointContractError,
            match="already has an owner",
        ):
            _OwnedCheckpointIO.acquire(checkpoint, directive)
    finally:
        first.close()

    assert not list(arguments["checkpoint_directory"].glob("*.aanca-owner.lock"))


@pytest.mark.parametrize(
    ("observer_name", "share_names"),
    [
        ("_win32_native_identity_from_path", {"read"}),
        (
            "_win32_native_identity_from_live_writer_path",
            {"read", "write"},
        ),
        (
            "_win32_native_identity_from_owner_lock_path",
            {"read", "write", "delete"},
        ),
    ],
)
def test_win32_path_identity_observers_use_exact_role_bound_share_masks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    observer_name: str,
    share_names: set[str],
) -> None:
    from histo_audit.cross_validation import image_oof as module

    opened: list[dict[str, int]] = []
    closed: list[int] = []
    identity = module._NativeCheckpointIdentity(
        volume_serial_number=19,
        file_id_128="ab" * 16,
        file_attributes=0,
        reparse_tag=0,
    )

    def fake_open(_path: Path, **kwargs: int) -> int:
        opened.append(kwargs)
        return 71

    monkeypatch.setattr(module, "_win32_open_handle", fake_open)
    monkeypatch.setattr(module, "_win32_native_identity", lambda _handle: identity)
    monkeypatch.setattr(module, "_win32_close_handle", closed.append)

    result = getattr(module, observer_name)(tmp_path / "identity.pt")

    share_constants = {
        "read": module._WIN_FILE_SHARE_READ,
        "write": module._WIN_FILE_SHARE_WRITE,
        "delete": module._WIN_FILE_SHARE_DELETE,
    }
    expected_share = 0
    for name in share_names:
        expected_share |= share_constants[name]
    assert result == identity
    assert opened == [
        {
            "desired_access": module._WIN_GENERIC_READ,
            "creation_disposition": module._WIN_OPEN_EXISTING,
            "flags_and_attributes": 0,
            "share_mode": expected_share,
        }
    ]
    assert closed == [71]
    if observer_name != "_win32_native_identity_from_owner_lock_path":
        assert expected_share & module._WIN_FILE_SHARE_DELETE == 0


def test_win32_owner_lock_creation_has_exact_delete_on_close_masks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from histo_audit.cross_validation import image_oof as module

    opened: list[dict[str, int]] = []
    identity = module._NativeCheckpointIdentity(
        volume_serial_number=23,
        file_id_128="cd" * 16,
        file_attributes=0,
        reparse_tag=0,
    )

    def fake_open(_path: Path, **kwargs: int) -> int:
        opened.append(kwargs)
        return 73

    monkeypatch.setattr(module, "_win32_open_handle", fake_open)
    monkeypatch.setattr(module, "_win32_native_identity", lambda _handle: identity)
    monkeypatch.setattr(
        module,
        "_win32_descriptor_from_handle",
        lambda _handle, *, writable: 79 if writable else 78,
    )

    descriptor, observed_identity = module._win32_create_owner_lock_descriptor(
        tmp_path / "owner.lock"
    )

    assert descriptor == 79
    assert observed_identity == identity
    assert opened == [
        {
            "desired_access": (
                module._WIN_GENERIC_READ | module._WIN_GENERIC_WRITE | module._WIN_DELETE
            ),
            "creation_disposition": module._WIN_CREATE_NEW,
            "flags_and_attributes": (
                module._WIN_FILE_ATTRIBUTE_NORMAL
                | module._WIN_FILE_FLAG_WRITE_THROUGH
                | module._WIN_FILE_FLAG_DELETE_ON_CLOSE
            ),
            "share_mode": module._WIN_FILE_SHARE_READ,
        }
    ]


@pytest.mark.skipif(os.name != "nt", reason="Windows retained-directory handle canary")
@pytest.mark.parametrize(
    "directory_role",
    [
        "run",
        "cells",
        "cell",
        "checkpoints",
        "checkpoint_versions",
        "fold_versions",
        "checkpoint_execution",
    ],
)
def test_retained_ancestor_handle_denies_rename_until_owner_release(
    directory_role: str,
    tmp_path: Path,
) -> None:
    arguments = _inputs(input_variant="context_rgb")
    run_directory = tmp_path / f"ancestor-{directory_role}"
    _authorise_fresh(arguments, run_directory)
    directive = arguments["checkpoint_execution_contract"].directives[0]
    logical = arguments["checkpoint_directory"] / "fold_00.pt"
    owner = _OwnedCheckpointIO.acquire(logical, directive)
    targets = {
        "run": run_directory,
        "cells": run_directory / "cells",
        "cell": arguments["checkpoint_directory"].parent,
        "checkpoints": arguments["checkpoint_directory"],
        "checkpoint_versions": arguments["checkpoint_directory"].parent / "checkpoint_versions",
        "fold_versions": arguments["checkpoint_directory"].parent
        / "checkpoint_versions"
        / "fold_00",
        "checkpoint_execution": arguments["checkpoint_directory"].parent / "checkpoint_execution",
    }
    target = targets[directory_role]
    renamed = target.with_name(f"{target.name}.foreign-swap")
    try:
        with pytest.raises(OSError):
            target.rename(renamed)
        assert target.is_dir()
        assert not renamed.exists()
    finally:
        owner.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows native sharing canary")
def test_held_checkpoint_snapshot_denies_same_byte_aba_replacement(
    tmp_path: Path,
) -> None:
    from histo_audit.cross_validation import image_oof as module

    checkpoint = tmp_path / "held.pt"
    checkpoint.write_bytes(b"same-byte-evidence")
    with module._hold_private_checkpoint_snapshot(
        checkpoint,
        role="same-byte ABA canary",
    ) as held:
        assert held.payload == b"same-byte-evidence"
        with pytest.raises(OSError):
            checkpoint.unlink()
        assert checkpoint.read_bytes() == held.payload


@pytest.mark.skipif(os.name != "nt", reason="Windows predecessor custody canary")
@pytest.mark.parametrize(
    "directory_role",
    ["run", "cells", "cell", "checkpoints"],
)
def test_retained_predecessor_ancestor_denies_rename_until_owner_release(
    directory_role: str,
    tmp_path: Path,
) -> None:
    arguments = _inputs(input_variant="context_rgb")
    arguments["base_config"] = replace(arguments["base_config"], epochs=2)
    _authorise_successor(
        arguments,
        tmp_path / f"predecessor-ancestor-{directory_role}",
        completed_epochs={0: 1, 1: 2},
    )
    directive = arguments["checkpoint_execution_contract"].directives[0]
    assert directive.source_predecessor_checkpoint is not None
    source = directive.source_predecessor_checkpoint.path
    owner = _OwnedCheckpointIO.acquire(
        arguments["checkpoint_directory"] / "fold_00.pt",
        directive,
    )
    targets = {
        "run": source.parents[3],
        "cells": source.parents[2],
        "cell": source.parents[1],
        "checkpoints": source.parent,
    }
    target = targets[directory_role]
    renamed = target.with_name(f"{target.name}.foreign-swap")
    try:
        with pytest.raises(OSError):
            target.rename(renamed)
        assert target.is_dir()
        assert not renamed.exists()
    finally:
        owner.close()


@pytest.mark.skipif(os.name != "nt", reason="NTFS alternate-data-stream canary")
def test_successor_rejects_checkpoint_with_named_data_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "histo_audit.cross_validation.image_oof.ConfirmatoryCNNCPUTestOnlyAdapter",
        _CheckpointStateAdapter,
    )
    arguments = _inputs(input_variant="context_rgb")
    arguments["base_config"] = replace(arguments["base_config"], epochs=2)
    _authorise_successor(
        arguments,
        tmp_path / "ads",
        completed_epochs={0: 1, 1: 2},
    )
    checkpoint = arguments["checkpoint_directory"] / "fold_00.pt"
    Path(f"{checkpoint}:rogue").write_bytes(b"forbidden")

    with pytest.raises(ConfirmatoryCheckpointContractError, match="stream-free"):
        grouped_oof_confirmatory_cnn(**arguments)


def test_checkpoint_owner_publishes_exactly_one_successful_fold_boundary(
    tmp_path: Path,
) -> None:
    arguments = _inputs(input_variant="context_rgb")
    arguments["base_config"] = replace(arguments["base_config"], epochs=2)
    _authorise_fresh(arguments, tmp_path / "two-epochs")
    directive = arguments["checkpoint_execution_contract"].directives[0]
    logical = arguments["checkpoint_directory"] / "fold_00.pt"
    output_directory = arguments["checkpoint_directory"].parent / "checkpoint_versions" / "fold_00"
    final = output_directory / "epoch_0002.pt"
    owner = _OwnedCheckpointIO.acquire(logical, directive)
    try:
        _publish_synthetic_boundary(
            owner,
            logical,
            b"fold-boundary",
            completed_epochs=2,
            trained_epochs=2,
        )
        with pytest.raises(ConfirmatoryCheckpointContractError):
            owner.publish_completed_working_checkpoint(
                owner.read_working_checkpoint(),
                completed_epochs=2,
                trained_epochs=2,
            )
    finally:
        owner.close()

    assert final.read_bytes() == b"fold-boundary"
    assert logical.read_bytes() == final.read_bytes()
    assert _physical_identity(logical).file_id_128 != _physical_identity(final).file_id_128
    assert logical.stat().st_nlink == final.stat().st_nlink == 1
    assert not (output_directory / "epoch_0001.pt").exists()
    assert (output_directory / "epoch_0002.commit.json").is_file()


def test_checkpoint_owner_short_temp_supports_legal_near_max_path(
    tmp_path: Path,
) -> None:
    arguments = _inputs(input_variant="context_rgb")
    suffix = str(Path("cells") / "c" / "checkpoint_versions" / "fold_00" / "epoch_0001.pt")
    base = str(tmp_path.resolve())
    padding = 229 - len(base) - len(suffix) - 2
    if not 1 <= padding <= 240:
        pytest.skip("temporary root cannot construct the near-MAX_PATH fixture")
    run_directory = tmp_path / ("r" * padding)
    _authorise_fresh(arguments, run_directory, cell_id="c")
    directive = arguments["checkpoint_execution_contract"].directives[0]
    logical = arguments["checkpoint_directory"] / "fold_00.pt"
    checkpoint = run_directory / "cells" / "c" / "checkpoint_versions" / "fold_00" / "epoch_0001.pt"
    owner = _OwnedCheckpointIO.acquire(logical, directive)
    try:
        _publish_synthetic_boundary(
            owner,
            logical,
            b"near-max-path-checkpoint",
            completed_epochs=1,
            trained_epochs=1,
        )
        owner.record_execution_manifest(
            completed_epochs_after=1,
            trained_epochs=1,
        )
    finally:
        owner.close()

    assert len(str(checkpoint)) == 229
    assert checkpoint.is_file()
    assert not list(run_directory.rglob(".c*.tmp"))


@pytest.mark.parametrize(
    "source_kind",
    ["canonical_checkpoint", "versioned_checkpoint", "commit_sidecar"],
)
def test_execution_manifest_rejects_same_byte_replacement_of_every_held_source(
    source_kind: str,
    tmp_path: Path,
) -> None:
    arguments = _inputs(input_variant="context_rgb")
    _authorise_fresh(arguments, tmp_path / f"held-history-{source_kind}")
    directive = arguments["checkpoint_execution_contract"].directives[0]
    logical = arguments["checkpoint_directory"] / "fold_00.pt"
    output_directory = arguments["checkpoint_directory"].parent / "checkpoint_versions" / "fold_00"
    execution_manifest = (
        arguments["checkpoint_directory"].parent / "checkpoint_execution" / "fold_00.json"
    )
    owner = _OwnedCheckpointIO.acquire(logical, directive)
    try:
        _publish_synthetic_boundary(
            owner,
            logical,
            b"fold-boundary",
            completed_epochs=1,
            trained_epochs=1,
        )
        target = {
            "canonical_checkpoint": logical,
            "versioned_checkpoint": output_directory / "epoch_0001.pt",
            "commit_sidecar": output_directory / "epoch_0001.commit.json",
        }[source_kind]
        payload = target.read_bytes()
        target.chmod(stat.S_IWRITE | stat.S_IREAD)
        target.unlink()
        target.write_bytes(payload)
        with pytest.raises(
            ConfirmatoryCheckpointContractError,
        ):
            owner.record_execution_manifest(
                completed_epochs_after=1,
                trained_epochs=1,
            )
    finally:
        owner.close()

    assert not execution_manifest.exists()


def test_owned_execution_manifest_same_byte_replacement_is_not_adopted(
    tmp_path: Path,
) -> None:
    arguments = _inputs(input_variant="context_rgb")
    _authorise_fresh(arguments, tmp_path / "manifest-post-record-swap")
    directive = arguments["checkpoint_execution_contract"].directives[0]
    logical = arguments["checkpoint_directory"] / "fold_00.pt"
    execution_manifest = (
        arguments["checkpoint_directory"].parent / "checkpoint_execution" / "fold_00.json"
    )
    owner = _OwnedCheckpointIO.acquire(logical, directive)
    try:
        _publish_synthetic_boundary(
            owner,
            logical,
            b"epoch-one",
            completed_epochs=1,
            trained_epochs=1,
        )
        owner.record_execution_manifest(
            completed_epochs_after=1,
            trained_epochs=1,
        )
        manifest_bytes = execution_manifest.read_bytes()
        original_identity = _physical_identity(execution_manifest)
        replacement = execution_manifest.with_name("replacement-manifest.json")
        replacement.write_bytes(manifest_bytes)
        execution_manifest.chmod(stat.S_IWRITE | stat.S_IREAD)
        os.replace(replacement, execution_manifest)
        assert _physical_identity(execution_manifest) != original_identity

        with pytest.raises(
            ConfirmatoryCheckpointContractError,
            match="differs from the exact owned publication",
        ):
            owner.read_current_manifest_bytes()
    finally:
        owner.close()


def test_checkpoint_execution_manifest_requires_exact_published_epoch_count(
    tmp_path: Path,
) -> None:
    arguments = _inputs(input_variant="context_rgb")
    arguments["base_config"] = replace(arguments["base_config"], epochs=2)
    _authorise_fresh(arguments, tmp_path / "manifest-epoch-count")
    directive = arguments["checkpoint_execution_contract"].directives[0]
    logical = arguments["checkpoint_directory"] / "fold_00.pt"
    execution_manifest = (
        arguments["checkpoint_directory"].parent / "checkpoint_execution" / "fold_00.json"
    )
    owner = _OwnedCheckpointIO.acquire(logical, directive)
    try:
        _publish_synthetic_boundary(
            owner,
            logical,
            b"epoch-one",
            completed_epochs=1,
            trained_epochs=1,
        )
        with pytest.raises(
            ConfirmatoryCheckpointContractError,
            match="lacks its one fold-boundary publication",
        ):
            owner.record_execution_manifest(
                completed_epochs_after=2,
                trained_epochs=2,
            )
    finally:
        owner.close()

    assert not execution_manifest.exists()


@pytest.mark.parametrize("failure", ["missing", "value"])
def test_checkpoint_originated_model_errors_remain_run_level_contract_errors(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "histo_audit.cross_validation.image_oof.ConfirmatoryCNNCPUTestOnlyAdapter",
        _ResumeCheckpointFailureAdapter,
    )
    _ResumeCheckpointFailureAdapter.failure = failure
    arguments = _inputs(input_variant="context_rgb")
    arguments["base_config"] = replace(arguments["base_config"], epochs=2)
    _authorise_successor(
        arguments,
        tmp_path / f"model-failure-{failure}",
        completed_epochs={0: 1, 1: 2},
    )

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="exact pinned resume checkpoint",
    ):
        grouped_oof_confirmatory_cnn(**arguments)
    assert not list(arguments["checkpoint_directory"].glob("*.aanca-owner.lock"))


def test_successor_resumes_incomplete_and_restores_terminal_without_fit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "histo_audit.cross_validation.image_oof.ConfirmatoryCNNCPUTestOnlyAdapter",
        _CheckpointStateAdapter,
    )
    _CheckpointStateAdapter.calls.clear()
    arguments = _inputs(input_variant="context_rgb")
    arguments["base_config"] = replace(arguments["base_config"], epochs=2)
    before_hashes = _authorise_successor(
        arguments,
        tmp_path / "successor",
        completed_epochs={0: 1, 1: 2},
        successful_steps={0: 3, 1: 7},
    )
    imported_paths = {
        fold_id: arguments["checkpoint_directory"] / f"fold_{fold_id:02d}.pt" for fold_id in (0, 1)
    }
    imported_identities = {
        fold_id: _physical_identity(path) for fold_id, path in imported_paths.items()
    }
    source_paths = {
        directive.fold_id: directive.source_predecessor_checkpoint.path
        for directive in arguments["checkpoint_execution_contract"].directives
        if directive.source_predecessor_checkpoint is not None
    }
    source_identities = {
        fold_id: _physical_identity(path) for fold_id, path in source_paths.items()
    }

    result = grouped_oof_confirmatory_cnn(**arguments)
    evidence = {value.fold_id: value for value in result.fold_evidence}

    assert evidence[0].checkpoint_action == "resume_incomplete_fit"
    assert evidence[0].completed_epochs_before_fit == 1
    assert evidence[0].trained_epochs_this_invocation == 1
    assert evidence[0].successful_optimiser_steps_before_fit == 3
    assert evidence[0].successful_optimiser_steps_this_invocation == 1
    assert evidence[0].checkpoint_sha256 != before_hashes[0]
    assert evidence[1].checkpoint_action == ("restore_terminal_checkpoint_without_fit")
    assert evidence[1].completed_epochs_before_fit == 2
    assert evidence[1].trained_epochs_this_invocation == 0
    assert evidence[1].successful_optimiser_steps_before_fit == 7
    assert evidence[1].successful_optimiser_steps_this_invocation == 0
    assert evidence[1].checkpoint_sha256 == before_hashes[1]
    assert all(
        _physical_identity(source_paths[fold_id]) == source_identities[fold_id]
        for fold_id in source_paths
    )
    assert all(
        _physical_identity(imported_paths[fold_id]) != imported_identities[fold_id]
        for fold_id in imported_paths
    )
    assert all(stat.S_IMODE(path.stat().st_mode) & 0o222 == 0 for path in imported_paths.values())
    terminal_manifest_path = Path(evidence[1].checkpoint_execution_manifest_path)
    terminal_manifest = json.loads(terminal_manifest_path.read_text(encoding="ascii"))
    assert terminal_manifest_path == (
        arguments["checkpoint_directory"].parent / "checkpoint_execution" / "fold_01.json"
    )
    assert terminal_manifest["schema_version"] == 3
    assert terminal_manifest["policy"] == "aanca_fold_boundary_checkpoint_execution_v3"
    assert terminal_manifest["action"] == "restore_terminal_checkpoint_without_fit"
    assert terminal_manifest["trained_epochs"] == 0
    assert terminal_manifest["versioned_checkpoint_output_directory_relative_path"] is None
    assert terminal_manifest["versioned_outputs"] == []
    assert (
        terminal_manifest["imported_checkpoint_observed"]
        == terminal_manifest["destination_imported_checkpoint"]
    )
    assert (
        terminal_manifest["final_checkpoint"] == terminal_manifest["canonical_working_checkpoint"]
    )
    assert terminal_manifest["imported_checkpoint_modified"] is False
    assert terminal_manifest["final_checkpoint"]["path"] == str(imported_paths[1].resolve())
    assert terminal_manifest["final_checkpoint"]["sha256"] == before_hashes[1]


@pytest.mark.parametrize(
    "mutation",
    ["missing", "changed", "same-bytes-swap", "hardlink"],
)
def test_successor_checkpoint_violation_never_falls_back_to_fresh(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "histo_audit.cross_validation.image_oof.ConfirmatoryCNNCPUTestOnlyAdapter",
        _CheckpointStateAdapter,
    )
    _CheckpointStateAdapter.calls.clear()
    arguments = _inputs(input_variant="context_rgb")
    arguments["base_config"] = replace(arguments["base_config"], epochs=2)
    _authorise_successor(
        arguments,
        tmp_path / mutation,
        completed_epochs={0: 1, 1: 2},
    )
    checkpoint = arguments["checkpoint_directory"] / "fold_00.pt"
    if mutation == "missing":
        checkpoint.unlink()
    elif mutation == "changed":
        checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    elif mutation == "same-bytes-swap":
        payload = checkpoint.read_bytes()
        checkpoint.unlink()
        checkpoint.write_bytes(payload)
    else:
        os.link(checkpoint, tmp_path / "second-name-hardlink.pt")

    with pytest.raises(ConfirmatoryCheckpointContractError):
        grouped_oof_confirmatory_cnn(**arguments)
    assert not _CheckpointStateAdapter.calls


def test_successor_rejects_single_field_physical_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "histo_audit.cross_validation.image_oof.ConfirmatoryCNNCPUTestOnlyAdapter",
        _CheckpointStateAdapter,
    )
    arguments = _inputs(input_variant="context_rgb")
    arguments["base_config"] = replace(arguments["base_config"], epochs=2)
    _authorise_successor(
        arguments,
        tmp_path / "physical-identity-drift",
        completed_epochs={0: 1, 1: 2},
    )
    contract = arguments["checkpoint_execution_contract"]
    first = contract.directives[0]
    assert first.destination_imported_checkpoint is not None
    drifted_destination = replace(
        first.destination_imported_checkpoint,
        physical_identity=replace(
            first.destination_imported_checkpoint.physical_identity,
            changed_time_ns=(
                first.destination_imported_checkpoint.physical_identity.changed_time_ns + 1
            ),
        ),
    )
    directives = (
        replace(first, destination_imported_checkpoint=drifted_destination),
        *contract.directives[1:],
    )
    arguments["checkpoint_execution_contract"] = _register_checkpoint_execution_contract(
        replace(
            contract,
            directives=directives,
            directives_sha256=_canonical_sha256([directive.as_dict() for directive in directives]),
        ),
        expected_directive_count=len(directives),
    )
    _CheckpointStateAdapter.calls.clear()

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="disappeared or changed",
    ):
        grouped_oof_confirmatory_cnn(**arguments)
    assert not _CheckpointStateAdapter.calls


def test_production_path_rejects_nonproduction_partial_checkpoint_universe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "histo_audit.cross_validation.image_oof.ConfirmatoryResNet18Classifier",
        _FastStudyAdapter,
    )
    arguments = _inputs(input_variant="context_rgb")
    arguments["base_config"] = replace(
        arguments["base_config"],
        weight_identifier=OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER,
    )
    arguments["cpu_test_only"] = False
    _authorise_fresh(arguments, tmp_path / "real")

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="requires a production checkpoint profile",
    ):
        grouped_oof_confirmatory_cnn(**arguments)


def test_missing_observed_class_in_any_training_fold_fails_before_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "histo_audit.cross_validation.image_oof.ConfirmatoryCNNCPUTestOnlyAdapter",
        _FastCPUTestAdapter,
    )
    _FastCPUTestAdapter.calls.clear()
    arguments = _inputs(input_variant="context_rgb")
    observed = arguments["observed_labels"].copy()
    observed[observed == 4] = 3
    # Retain class 4 globally in only one group. Whichever fold holds that
    # group out necessarily leaves a training partition without class 4.
    observed[4] = 4
    arguments["observed_labels"] = observed
    _authorise_fresh(arguments, tmp_path / "missing-class")

    with pytest.raises(ValueError, match="training partition is missing fixed classes"):
        grouped_oof_confirmatory_cnn(**arguments)
    assert not _FastCPUTestAdapter.calls
