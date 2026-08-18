from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from histo_audit.models import FrozenEmbeddingMLPClassifier, FrozenEmbeddingMLPConfig
from histo_audit.models.mlp import _execute_epoch_with_backoff, _MinimumBatchOOMError


def _binary_embeddings() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(721)
    training = np.concatenate(
        (
            rng.normal(-0.8, 0.35, size=(20, 5)),
            rng.normal(0.8, 0.35, size=(20, 5)),
        )
    ).astype(np.float32)
    training_labels = np.repeat(np.asarray([0, 1], dtype=np.int64), 20)
    reference_validation = np.concatenate(
        (
            rng.normal(-0.8, 0.35, size=(6, 5)),
            rng.normal(0.8, 0.35, size=(6, 5)),
        )
    ).astype(np.float32)
    validation_labels = np.repeat(np.asarray([0, 1], dtype=np.int64), 6)
    return training, training_labels, reference_validation, validation_labels


def test_cuda_oom_backoff_rolls_back_and_retries_the_same_cursor() -> None:
    order = np.arange(10, dtype=np.int64)
    attempts: list[tuple[int, np.ndarray]] = []
    successful_rows: list[int] = []
    mutable_state = {"updates": 0}
    events: list[dict[str, float | int | str]] = []

    def execute(indices: np.ndarray, batch_size: int) -> float:
        attempts.append((batch_size, indices.copy()))
        mutable_state["updates"] += 1
        if batch_size == 8:
            mutable_state["updates"] += 100
            raise torch.OutOfMemoryError("CUDA out of memory: injected test failure")
        successful_rows.extend(int(index) for index in indices)
        return float(len(indices))

    def restore(snapshot: object) -> None:
        mutable_state["updates"] = int(snapshot)

    execution = _execute_epoch_with_backoff(
        order,
        epoch=1,
        phase="training",
        initial_batch_size=8,
        minimum_batch_size=2,
        accumulation_steps=1,
        cuda_oom_recovery=True,
        execute_window=execute,
        snapshot_state=lambda: mutable_state["updates"],
        restore_state=restore,
        clear_after_oom=lambda: None,
        backoff_events=events,
    )

    assert attempts[0][0] == 8
    assert attempts[1][0] == 4
    assert attempts[0][1][0] == attempts[1][1][0] == 0
    assert successful_rows == order.tolist()
    assert mutable_state["updates"] == execution.optimiser_steps == 3
    assert execution.successful_sample_count == len(order)
    assert execution.loss_sum == len(order)
    assert execution.final_batch_size == 4
    assert events == [
        {
            "epoch": 1,
            "phase": "training",
            "order_cursor": 0,
            "attempted_sample_count": 8,
            "attempted_first_row": 0,
            "attempted_last_row": 7,
            "failed_batch_size": 8,
            "minimum_batch_size": 2,
            "error_type": "OutOfMemoryError",
            "error_message": "CUDA out of memory: injected test failure",
            "retry_batch_size": 4,
            "outcome": "retry_same_order_cursor",
        }
    ]


def test_cuda_oom_at_minimum_batch_fails_with_rollback_evidence() -> None:
    mutable_state = {"updates": 0}
    events: list[dict[str, float | int | str]] = []

    def fail(_indices: np.ndarray, _batch_size: int) -> float:
        mutable_state["updates"] = 99
        raise torch.OutOfMemoryError("CUDA out of memory: minimum-batch test")

    def restore(snapshot: object) -> None:
        mutable_state["updates"] = int(snapshot)

    with pytest.raises(_MinimumBatchOOMError, match="minimum batch size") as raised:
        _execute_epoch_with_backoff(
            np.arange(2, dtype=np.int64),
            epoch=3,
            phase="training",
            initial_batch_size=1,
            minimum_batch_size=1,
            accumulation_steps=1,
            cuda_oom_recovery=True,
            execute_window=fail,
            snapshot_state=lambda: mutable_state["updates"],
            restore_state=restore,
            clear_after_oom=lambda: None,
            backoff_events=events,
        )

    assert mutable_state["updates"] == 0
    assert raised.value.evidence["outcome"] == "failed_at_minimum_batch_size"
    assert raised.value.evidence["order_cursor"] == 0
    assert events == [raised.value.evidence]


def test_early_stopping_is_reference_validation_only_and_resumable(tmp_path: Path) -> None:
    training, labels, reference_validation, validation_labels = _binary_embeddings()
    checkpoint = tmp_path / "early-stopping.pt"
    config = FrozenEmbeddingMLPConfig(
        hidden_dimensions=(8,),
        dropout=0.0,
        epochs=8,
        batch_size=8,
        learning_rate=3e-3,
        early_stopping_patience=2,
        early_stopping_min_delta=1_000_000.0,
        seed=47,
        device="cpu",
    )

    with pytest.raises(ValueError, match="explicitly marked as reference_validation"):
        FrozenEmbeddingMLPClassifier(config).fit(
            training,
            labels,
            validation_data=(reference_validation, validation_labels),
        )

    fitted = FrozenEmbeddingMLPClassifier(config).fit(
        training,
        labels,
        validation_data=(reference_validation, validation_labels),
        validation_role="reference_validation",
        checkpoint_path=checkpoint,
    )
    assert fitted.stopped_early_ is True
    assert fitted.completed_epochs_ == 3
    assert fitted.best_epoch_ == 1
    assert fitted.epochs_without_improvement_ == 2
    assert len(fitted.history_) == 3
    assert fitted.telemetry_["early_stopping_source"] == "reference_validation_only"
    assert fitted.telemetry_["cuda_peak_memory_allocated_bytes"] is None
    assert cast(float, fitted.telemetry_["current_fit_runtime_seconds"]) >= 0.0

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint_payload["schema_version"] == 2
    assert checkpoint_payload["scaler_state_dict"] is None
    assert checkpoint_payload["early_stopping_state"]["stopped_early"] is True
    assert checkpoint_payload["rng_state"]["torch_cpu"].dtype == torch.uint8

    resumed = FrozenEmbeddingMLPClassifier(config).fit(
        training,
        labels,
        validation_data=(reference_validation, validation_labels),
        validation_role="reference_validation",
        checkpoint_path=checkpoint,
        resume=True,
    )
    assert resumed.completed_epochs_ == fitted.completed_epochs_
    assert resumed.stopped_early_ is True
    np.testing.assert_array_equal(
        resumed.predict_proba(reference_validation),
        fitted.predict_proba(reference_validation),
    )


def test_gradient_accumulation_is_deterministic_and_telemetried() -> None:
    training, labels, _, _ = _binary_embeddings()
    config = FrozenEmbeddingMLPConfig(
        hidden_dimensions=(6,),
        dropout=0.0,
        epochs=2,
        batch_size=6,
        gradient_accumulation_steps=3,
        learning_rate=2e-3,
        seed=83,
        device="cpu",
    )
    first = FrozenEmbeddingMLPClassifier(config).fit(training, labels)
    second = FrozenEmbeddingMLPClassifier(config).fit(training, labels)

    np.testing.assert_array_equal(first.predict_proba(training), second.predict_proba(training))
    expected_steps_per_epoch = 3  # ceil(40 / (6 * 3))
    assert [row["optimiser_steps"] for row in first.history_] == [
        expected_steps_per_epoch,
        expected_steps_per_epoch,
    ]
    assert first.telemetry_["gradient_accumulation_steps"] == 3
    assert first.telemetry_["optimiser_steps"] == 2 * expected_steps_per_epoch
    assert first.telemetry_["successful_samples_processed"] == 2 * len(training)


def test_amp_request_on_cpu_fails_instead_of_using_cpu_autocast() -> None:
    training, labels, _, _ = _binary_embeddings()
    config = FrozenEmbeddingMLPConfig(
        hidden_dimensions=(4,),
        epochs=1,
        batch_size=8,
        amp=True,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="supported only on CUDA"):
        FrozenEmbeddingMLPClassifier(config).fit(training, labels)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP requires a CUDA runtime")
def test_cuda_amp_uses_grad_scaler_and_records_peak_memory(tmp_path: Path) -> None:
    training, labels, _, _ = _binary_embeddings()
    checkpoint = tmp_path / "cuda-amp.pt"
    config = FrozenEmbeddingMLPConfig(
        hidden_dimensions=(4,),
        dropout=0.0,
        epochs=1,
        batch_size=8,
        gradient_accumulation_steps=2,
        amp=True,
        amp_dtype="float16",
        seed=97,
        device="cuda",
    )
    fitted = FrozenEmbeddingMLPClassifier(config).fit(training, labels, checkpoint_path=checkpoint)
    probabilities = fitted.predict_proba(training)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert fitted.telemetry_["amp_enabled"] is True
    assert cast(int, fitted.telemetry_["cuda_peak_memory_allocated_bytes"]) > 0
    assert payload["scaler_state_dict"] is not None
    assert float(payload["scaler_state_dict"]["scale"]) > 0.0
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    extended_config = replace(config, epochs=2)
    uninterrupted = FrozenEmbeddingMLPClassifier(extended_config).fit(training, labels)
    resumed = FrozenEmbeddingMLPClassifier(extended_config).fit(
        training, labels, checkpoint_path=checkpoint, resume=True
    )
    assert resumed.completed_epochs_ == 2
    np.testing.assert_allclose(
        resumed.predict_proba(training),
        uninterrupted.predict_proba(training),
        rtol=0.0,
        atol=1e-7,
    )


def test_checkpoint_rejects_changed_reference_validation_data(tmp_path: Path) -> None:
    training, labels, reference_validation, validation_labels = _binary_embeddings()
    checkpoint = tmp_path / "reference-validation-fingerprint.pt"
    partial = FrozenEmbeddingMLPConfig(
        hidden_dimensions=(5,),
        dropout=0.0,
        epochs=1,
        batch_size=8,
        seed=101,
        device="cpu",
    )
    FrozenEmbeddingMLPClassifier(partial).fit(
        training,
        labels,
        validation_data=(reference_validation, validation_labels),
        validation_role="reference_validation",
        checkpoint_path=checkpoint,
    )
    changed = reference_validation.copy()
    changed[0, 0] += 0.1
    extended = FrozenEmbeddingMLPConfig(
        hidden_dimensions=(5,),
        dropout=0.0,
        epochs=2,
        batch_size=8,
        seed=101,
        device="cpu",
    )
    with pytest.raises(ValueError, match="reference-validation fingerprint"):
        FrozenEmbeddingMLPClassifier(extended).fit(
            training,
            labels,
            validation_data=(changed, validation_labels),
            validation_role="reference_validation",
            checkpoint_path=checkpoint,
            resume=True,
        )


def test_minimum_batch_and_early_stopping_configuration_validation() -> None:
    with pytest.raises(ValueError, match="minimum_batch_size cannot exceed batch_size"):
        FrozenEmbeddingMLPConfig(batch_size=2, minimum_batch_size=3).validate()
    with pytest.raises(ValueError, match="early_stopping_patience must be positive"):
        FrozenEmbeddingMLPConfig(early_stopping_patience=0).validate()
    with pytest.raises(ValueError, match="early_stopping_min_delta"):
        FrozenEmbeddingMLPConfig(early_stopping_min_delta=float("nan")).validate()


def test_checkpoint_payload_remains_weights_only_loadable(tmp_path: Path) -> None:
    training, labels, _, _ = _binary_embeddings()
    checkpoint = tmp_path / "safe-checkpoint.pt"
    config = FrozenEmbeddingMLPConfig(
        hidden_dimensions=(4,), epochs=1, batch_size=8, seed=131, device="cpu"
    )
    FrozenEmbeddingMLPClassifier(config).fit(training, labels, checkpoint_path=checkpoint)
    payload = cast(dict[str, Any], torch.load(checkpoint, map_location="cpu", weights_only=True))
    assert payload["model_kind"] == "frozen_embedding_mlp"
    assert payload["rng_state"]["numpy"]["keys"].dtype == torch.uint32
