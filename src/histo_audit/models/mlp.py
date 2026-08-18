"""A small, resumable probabilistic MLP for already-frozen embeddings."""

from __future__ import annotations

import copy
import hashlib
import os
import random
import tempfile
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Self, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

ClassWeightMode = Literal["balanced", "none"]
AmpDtype = Literal["float16", "bfloat16"]
ValidationRole = Literal["reference_validation"]
type HistoryRow = dict[str, float | int | bool | str | None]
type BackoffEvent = dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class FrozenEmbeddingMLPConfig:
    """Training settings for the compact frozen-feature classifier."""

    hidden_dimensions: tuple[int, ...] = (64,)
    dropout: float = 0.1
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    class_weight: ClassWeightMode = "balanced"
    seed: int = 0
    device: str = "auto"
    amp: bool = False
    amp_dtype: AmpDtype = "float16"
    gradient_accumulation_steps: int = 1
    minimum_batch_size: int = 1
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0

    def validate(self) -> None:
        """Reject settings that would make training ill-defined."""

        if not self.hidden_dimensions or any(width <= 0 for width in self.hidden_dimensions):
            raise ValueError("hidden_dimensions must contain positive widths")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.class_weight not in ("balanced", "none"):
            raise ValueError(f"unsupported class_weight mode: {self.class_weight!r}")
        if self.amp_dtype not in ("float16", "bfloat16"):
            raise ValueError(f"unsupported AMP dtype: {self.amp_dtype!r}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.minimum_batch_size <= 0:
            raise ValueError("minimum_batch_size must be positive")
        if self.minimum_batch_size > self.batch_size:
            raise ValueError("minimum_batch_size cannot exceed batch_size")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive when enabled")
        if not np.isfinite(self.early_stopping_min_delta) or self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be finite and non-negative")


class _Network(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: tuple[int, ...],
        output_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dimension
        for width in hidden_dimensions:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, output_dimension))
        self.layers = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


@dataclass(frozen=True, slots=True)
class _EpochExecution:
    loss_sum: float
    successful_sample_count: int
    optimiser_steps: int
    final_batch_size: int


class _MinimumBatchOOMError(RuntimeError):
    """Raised when CUDA OOM recovery has exhausted its declared batch floor."""

    def __init__(self, evidence: BackoffEvent) -> None:
        self.evidence = dict(evidence)
        super().__init__(
            "CUDA out of memory at the configured minimum batch size; "
            f"failure_evidence={self.evidence}"
        )


def _resolve_device(requested: str) -> torch.device:
    if requested.strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("frozen-embedding MLP supports CPU or CUDA devices")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _matrix(values: NDArray[np.generic], *, name: str) -> NDArray[np.float32]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty shape (n_samples, n_features)")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _labels(values: NDArray[np.generic] | list[int] | tuple[int, ...], n: int) -> NDArray[np.int64]:
    labels = np.asarray(values)
    if labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array aligned with embeddings")
    return labels.astype(np.int64)


def _dataset_sha256(features: NDArray[np.float32], labels: NDArray[np.int64]) -> str:
    digest = hashlib.sha256()
    for array in (features, labels):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _optimiser_to_device(optimiser: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimiser.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _model_state_to_cpu(network: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}


def _capture_rng_state() -> dict[str, Any]:
    numpy_state = cast(tuple[str, NDArray[np.uint32], int, int, float], np.random.get_state())
    cuda_states: list[torch.Tensor] = []
    if torch.cuda.is_available():
        cuda_states = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": cuda_states,
    }


def _restore_rng_state(payload: dict[str, Any], device: torch.device) -> None:
    python_state = cast(tuple[Any, ...], payload["python"])
    random.setstate(python_state)
    numpy_state = cast(dict[str, Any], payload["numpy"])
    numpy_keys = cast(torch.Tensor, numpy_state["keys"]).cpu().numpy().astype(np.uint32)
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_keys,
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(cast(torch.Tensor, payload["torch_cpu"]).cpu())
    cuda_states = cast(list[torch.Tensor], payload.get("torch_cuda", []))
    if device.type == "cuda" and cuda_states:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def _is_cuda_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return isinstance(error, torch.OutOfMemoryError) or "cuda out of memory" in message


def _execute_epoch_with_backoff(
    order: NDArray[np.int64],
    *,
    epoch: int,
    phase: str,
    initial_batch_size: int,
    minimum_batch_size: int,
    accumulation_steps: int,
    cuda_oom_recovery: bool,
    execute_window: Callable[[NDArray[np.int64], int], float],
    snapshot_state: Callable[[], Any],
    restore_state: Callable[[Any], None],
    clear_after_oom: Callable[[], None],
    backoff_events: list[BackoffEvent],
) -> _EpochExecution:
    """Execute every ordered row exactly once after transactional CUDA OOM retries."""

    batch_size = initial_batch_size
    cursor = 0
    loss_sum = 0.0
    successful_sample_count = 0
    optimiser_steps = 0
    while cursor < len(order):
        stop = min(len(order), cursor + batch_size * accumulation_steps)
        indices = order[cursor:stop]
        snapshot = snapshot_state() if cuda_oom_recovery else None
        try:
            window_loss = execute_window(indices, batch_size)
        except RuntimeError as error:
            if not cuda_oom_recovery or not _is_cuda_oom(error):
                raise
            restore_state(snapshot)
            clear_after_oom()
            event: BackoffEvent = {
                "epoch": epoch,
                "phase": phase,
                "order_cursor": cursor,
                "attempted_sample_count": len(indices),
                "attempted_first_row": int(indices[0]),
                "attempted_last_row": int(indices[-1]),
                "failed_batch_size": batch_size,
                "minimum_batch_size": minimum_batch_size,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
            if batch_size <= minimum_batch_size:
                event["outcome"] = "failed_at_minimum_batch_size"
                backoff_events.append(event)
                raise _MinimumBatchOOMError(event) from error
            next_batch_size = max(minimum_batch_size, batch_size // 2)
            event["retry_batch_size"] = next_batch_size
            event["outcome"] = "retry_same_order_cursor"
            backoff_events.append(event)
            batch_size = next_batch_size
            continue
        if not np.isfinite(window_loss):
            raise RuntimeError("MLP training produced a non-finite loss")
        loss_sum += window_loss
        successful_sample_count += len(indices)
        optimiser_steps += 1
        cursor = stop
    return _EpochExecution(
        loss_sum=loss_sum,
        successful_sample_count=successful_sample_count,
        optimiser_steps=optimiser_steps,
        final_batch_size=batch_size,
    )


class FrozenEmbeddingMLPClassifier:
    """Scikit-like MLP that consumes fixed embeddings and returns probabilities.

    The label argument is deliberately generic: callers must pass only observed
    development labels. No pre-corruption/reference-label argument exists.
    Early stopping additionally requires data explicitly marked as the clean
    ``reference_validation`` partition; no final-test validation role is accepted.
    """

    def __init__(self, config: FrozenEmbeddingMLPConfig | None = None) -> None:
        self.config = config or FrozenEmbeddingMLPConfig()
        self.config.validate()
        self.classes_: NDArray[np.int64] | None = None
        self.history_: tuple[HistoryRow, ...] = ()
        self.completed_epochs_: int = 0
        self.input_dimension_: int | None = None
        self.feature_mean_: NDArray[np.float32] | None = None
        self.feature_scale_: NDArray[np.float32] | None = None
        self.device_: str | None = None
        self.telemetry_: dict[str, Any] = {}
        self.effective_batch_size_: int = self.config.batch_size
        self.best_epoch_: int | None = None
        self.best_validation_loss_: float | None = None
        self.epochs_without_improvement_: int = 0
        self.stopped_early_: bool = False
        self._network: _Network | None = None
        self._best_network_state: dict[str, torch.Tensor] | None = None
        self._batch_backoff_events: list[BackoffEvent] = []
        self._prior_runtime_seconds: float = 0.0
        self._successful_samples_processed: int = 0
        self._optimiser_steps: int = 0

    def _build_network(
        self, input_dimension: int, class_count: int, device: torch.device
    ) -> _Network:
        _seed_everything(self.config.seed)
        network = _Network(
            input_dimension,
            self.config.hidden_dimensions,
            class_count,
            self.config.dropout,
        )
        return network.to(device)

    def _normalise(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError("classifier normalisation is not fitted")
        return ((features - self.feature_mean_) / self.feature_scale_).astype(np.float32)

    def _amp_torch_dtype(self) -> torch.dtype:
        return torch.float16 if self.config.amp_dtype == "float16" else torch.bfloat16

    def _training_state_snapshot(
        self,
        optimiser: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
    ) -> dict[str, Any]:
        if self._network is None:
            raise RuntimeError("cannot snapshot an uninitialised network")
        return {
            "network_state_dict": _model_state_to_cpu(self._network),
            "optimiser_state_dict": _clone_to_cpu(optimiser.state_dict()),
            "scaler_state_dict": _clone_to_cpu(scaler.state_dict()) if scaler else None,
            "rng_state": _capture_rng_state(),
        }

    def _restore_training_state(
        self,
        payload: dict[str, Any],
        optimiser: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
        device: torch.device,
    ) -> None:
        if self._network is None:
            raise RuntimeError("cannot restore an uninitialised network")
        self._network.load_state_dict(payload["network_state_dict"], strict=True)
        self._network.to(device)
        optimiser.load_state_dict(payload["optimiser_state_dict"])
        _optimiser_to_device(optimiser, device)
        scaler_state = payload.get("scaler_state_dict")
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        _restore_rng_state(cast(dict[str, Any], payload["rng_state"]), device)

    def _gpu_peak_telemetry(self, device: torch.device) -> tuple[int | None, int | None]:
        if device.type != "cuda":
            return None, None
        try:
            return (
                int(torch.cuda.max_memory_allocated(device)),
                int(torch.cuda.max_memory_reserved(device)),
            )
        except (RuntimeError, ValueError):
            return None, None

    def _telemetry_snapshot(
        self,
        device: torch.device,
        current_runtime_seconds: float,
        *,
        failure: str | None,
    ) -> dict[str, Any]:
        peak_allocated, peak_reserved = self._gpu_peak_telemetry(device)
        return {
            "schema_version": 1,
            "device": str(device),
            "amp_requested": self.config.amp,
            "amp_enabled": self.config.amp and device.type == "cuda",
            "amp_dtype": self.config.amp_dtype if self.config.amp else None,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "initial_batch_size": self.config.batch_size,
            "effective_batch_size": self.effective_batch_size_,
            "minimum_batch_size": self.config.minimum_batch_size,
            "batch_backoff_events": [dict(event) for event in self._batch_backoff_events],
            "current_fit_runtime_seconds": current_runtime_seconds,
            "cumulative_runtime_seconds": self._prior_runtime_seconds + current_runtime_seconds,
            "cuda_peak_memory_allocated_bytes": peak_allocated,
            "cuda_peak_memory_reserved_bytes": peak_reserved,
            "completed_epochs": self.completed_epochs_,
            "requested_epochs": self.config.epochs,
            "successful_samples_processed": self._successful_samples_processed,
            "optimiser_steps": self._optimiser_steps,
            "early_stopping_source": (
                "reference_validation_only"
                if self.config.early_stopping_patience is not None
                else None
            ),
            "early_stopping_patience": self.config.early_stopping_patience,
            "early_stopping_min_delta": self.config.early_stopping_min_delta,
            "best_epoch": self.best_epoch_,
            "best_validation_loss": self.best_validation_loss_,
            "epochs_without_improvement": self.epochs_without_improvement_,
            "stopped_early": self.stopped_early_,
            "failure": failure,
        }

    def _checkpoint_payload(
        self,
        optimiser: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
        training_sha256: str,
        reference_validation_sha256: str | None,
        device: torch.device,
        current_runtime_seconds: float,
    ) -> dict[str, Any]:
        if (
            self._network is None
            or self.classes_ is None
            or self.input_dimension_ is None
            or self.feature_mean_ is None
            or self.feature_scale_ is None
        ):
            raise RuntimeError("cannot checkpoint an unfitted classifier")
        return {
            "schema_version": 2,
            "model_kind": "frozen_embedding_mlp",
            "completed_epochs": self.completed_epochs_,
            "input_dimension": self.input_dimension_,
            "classes": torch.from_numpy(self.classes_.copy()),
            "feature_mean": torch.from_numpy(self.feature_mean_.copy()),
            "feature_scale": torch.from_numpy(self.feature_scale_.copy()),
            "network_state_dict": _model_state_to_cpu(self._network),
            "optimiser_state_dict": _clone_to_cpu(optimiser.state_dict()),
            "scaler_state_dict": _clone_to_cpu(scaler.state_dict()) if scaler else None,
            "history": list(self.history_),
            "configuration": asdict(self.config),
            "training_sha256": training_sha256,
            "reference_validation_sha256": reference_validation_sha256,
            "validation_role": (
                "reference_validation" if reference_validation_sha256 is not None else None
            ),
            "label_policy": "caller-supplied observed development labels only",
            "early_stopping_state": {
                "best_epoch": self.best_epoch_,
                "best_validation_loss": self.best_validation_loss_,
                "epochs_without_improvement": self.epochs_without_improvement_,
                "stopped_early": self.stopped_early_,
                "best_network_state_dict": self._best_network_state,
            },
            "effective_batch_size": self.effective_batch_size_,
            "rng_state": _capture_rng_state(),
            "telemetry": self._telemetry_snapshot(device, current_runtime_seconds, failure=None),
        }

    def _restore_training_checkpoint(
        self,
        checkpoint_path: Path,
        optimiser: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
        features: NDArray[np.float32],
        classes: NDArray[np.int64],
        training_sha256: str,
        reference_validation_sha256: str | None,
        device: torch.device,
    ) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        schema_version = payload.get("schema_version")
        if payload.get("model_kind") != "frozen_embedding_mlp" or schema_version not in {1, 2}:
            raise ValueError("checkpoint is not a supported frozen-embedding MLP checkpoint")
        checkpoint_config = cast(dict[str, Any], payload["configuration"])
        current_config = asdict(self.config)
        default_config = asdict(FrozenEmbeddingMLPConfig())
        for key in current_config:
            if key in {"epochs", "device"}:
                continue
            checkpoint_value = checkpoint_config.get(key, default_config[key])
            current_value = current_config[key]
            if key == "hidden_dimensions":
                checkpoint_value = tuple(checkpoint_value)
                current_value = tuple(current_value)
            if checkpoint_value != current_value:
                raise ValueError(f"checkpoint configuration mismatch for {key}")
        if int(payload["input_dimension"]) != features.shape[1]:
            raise ValueError("checkpoint input dimension differs from embeddings")
        checkpoint_classes = payload["classes"].cpu().numpy().astype(np.int64)
        if not np.array_equal(checkpoint_classes, classes):
            raise ValueError("checkpoint class order differs from supplied labels")
        if payload["training_sha256"] != training_sha256:
            raise ValueError("checkpoint training data fingerprint differs from supplied data")
        if schema_version == 2 and payload.get("reference_validation_sha256") != (
            reference_validation_sha256
        ):
            raise ValueError(
                "checkpoint reference-validation fingerprint differs from supplied data"
            )
        completed = int(payload["completed_epochs"])
        if completed > self.config.epochs:
            raise ValueError("checkpoint was trained beyond configured epochs")
        if self._network is None:
            raise RuntimeError("network must be initialised before checkpoint restoration")
        self._network.load_state_dict(payload["network_state_dict"], strict=True)
        self._network.to(device)
        optimiser.load_state_dict(payload["optimiser_state_dict"])
        _optimiser_to_device(optimiser, device)
        scaler_state = payload.get("scaler_state_dict")
        if scaler is not None:
            if scaler_state is None:
                raise ValueError("AMP checkpoint is missing GradScaler state")
            scaler.load_state_dict(scaler_state)
        self.classes_ = checkpoint_classes
        self.feature_mean_ = payload["feature_mean"].cpu().numpy().astype(np.float32)
        self.feature_scale_ = payload["feature_scale"].cpu().numpy().astype(np.float32)
        self.completed_epochs_ = completed
        self.history_ = tuple(cast(HistoryRow, dict(row)) for row in payload["history"])
        early_state = cast(dict[str, Any], payload.get("early_stopping_state", {}))
        self.best_epoch_ = (
            int(early_state["best_epoch"]) if early_state.get("best_epoch") is not None else None
        )
        self.best_validation_loss_ = (
            float(early_state["best_validation_loss"])
            if early_state.get("best_validation_loss") is not None
            else None
        )
        self.epochs_without_improvement_ = int(early_state.get("epochs_without_improvement", 0))
        self.stopped_early_ = bool(early_state.get("stopped_early", False))
        best_state = early_state.get("best_network_state_dict")
        self._best_network_state = (
            cast(dict[str, torch.Tensor], best_state) if best_state is not None else None
        )
        self.effective_batch_size_ = int(
            payload.get("effective_batch_size", self.config.batch_size)
        )
        checkpoint_telemetry = cast(dict[str, Any], payload.get("telemetry", {}))
        self._prior_runtime_seconds = float(
            checkpoint_telemetry.get("cumulative_runtime_seconds", 0.0)
        )
        self._batch_backoff_events = [
            cast(BackoffEvent, dict(event))
            for event in checkpoint_telemetry.get("batch_backoff_events", [])
        ]
        self._successful_samples_processed = int(
            checkpoint_telemetry.get("successful_samples_processed", 0)
        )
        self._optimiser_steps = int(checkpoint_telemetry.get("optimiser_steps", 0))
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            _restore_rng_state(cast(dict[str, Any], rng_state), device)

    def _execute_training_window(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        indices: NDArray[np.int64],
        micro_batch_size: int,
        optimiser: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
        criterion: nn.CrossEntropyLoss,
        device: torch.device,
    ) -> float:
        if self._network is None:
            raise RuntimeError("network is not initialised")
        optimiser.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for start in range(0, len(indices), micro_batch_size):
            micro_indices = indices[start : start + micro_batch_size]
            batch_x = train_x[micro_indices].to(device)
            batch_y = train_y[micro_indices].to(device)
            if self.config.amp:
                with torch.autocast(
                    device_type="cuda", dtype=self._amp_torch_dtype(), enabled=True
                ):
                    per_sample_loss = criterion(self._network(batch_x), batch_y)
            else:
                per_sample_loss = criterion(self._network(batch_x), batch_y)
            micro_loss_sum = per_sample_loss.sum()
            backward_loss = micro_loss_sum / len(indices)
            if scaler is not None:
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()
            loss_sum += float(micro_loss_sum.detach().float().cpu())
        if scaler is not None:
            scaler.step(optimiser)
            scaler.update()
        else:
            optimiser.step()
        return loss_sum

    def _reference_validation_loss(
        self,
        validation_x: torch.Tensor,
        validation_y: torch.Tensor,
        criterion: nn.CrossEntropyLoss,
        device: torch.device,
        epoch: int,
    ) -> float:
        if self._network is None:
            raise RuntimeError("network is not initialised")
        network = self._network
        network.eval()
        order = np.arange(len(validation_x), dtype=np.int64)

        def evaluate(indices: NDArray[np.int64], _batch_size: int) -> float:
            batch_x = validation_x[indices].to(device)
            batch_y = validation_y[indices].to(device)
            with torch.inference_mode():
                if self.config.amp:
                    with torch.autocast(
                        device_type="cuda", dtype=self._amp_torch_dtype(), enabled=True
                    ):
                        losses = criterion(network(batch_x), batch_y)
                else:
                    losses = criterion(network(batch_x), batch_y)
            return float(losses.sum().detach().float().cpu())

        result = _execute_epoch_with_backoff(
            order,
            epoch=epoch,
            phase="reference_validation",
            initial_batch_size=self.effective_batch_size_,
            minimum_batch_size=self.config.minimum_batch_size,
            accumulation_steps=1,
            cuda_oom_recovery=device.type == "cuda",
            execute_window=evaluate,
            snapshot_state=lambda: None,
            restore_state=lambda _snapshot: None,
            clear_after_oom=torch.cuda.empty_cache,
            backoff_events=self._batch_backoff_events,
        )
        self.effective_batch_size_ = result.final_batch_size
        return result.loss_sum / result.successful_sample_count

    def fit(
        self,
        embeddings: NDArray[np.generic],
        observed_labels: NDArray[np.generic] | list[int] | tuple[int, ...],
        *,
        validation_data: tuple[NDArray[np.generic], NDArray[np.generic]] | None = None,
        validation_role: ValidationRole | None = None,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
    ) -> Self:
        """Fit on observed labels, checkpointing after each completed epoch if requested.

        Early stopping is fail-closed: it is available only when ``validation_data``
        is explicitly identified with ``validation_role="reference_validation"``.
        """

        fit_started = time.perf_counter()
        features = _matrix(embeddings, name="embeddings")
        labels = _labels(observed_labels, len(features))
        classes, encoded = np.unique(labels, return_inverse=True)
        classes = classes.astype(np.int64)
        encoded = encoded.astype(np.int64)
        if len(classes) < 2:
            raise ValueError("MLP training requires at least two observed classes")
        path = Path(checkpoint_path) if checkpoint_path is not None else None
        if resume and path is None:
            raise ValueError("resume=True requires checkpoint_path")
        if resume and path is not None and not path.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
        if validation_role is not None and validation_role != "reference_validation":
            raise ValueError("validation_role must be 'reference_validation'")
        if validation_role is not None and validation_data is None:
            raise ValueError("validation_role requires validation_data")
        if self.config.early_stopping_patience is not None and (
            validation_data is None or validation_role != "reference_validation"
        ):
            raise ValueError(
                "early stopping requires validation_data explicitly marked as reference_validation"
            )

        validation_features: NDArray[np.float32] | None = None
        validation_positions: NDArray[np.int64] | None = None
        reference_validation_sha256: str | None = None
        if validation_data is not None:
            validation_features = _matrix(validation_data[0], name="validation embeddings")
            validation_labels = _labels(validation_data[1], len(validation_features))
            if validation_features.shape[1] != features.shape[1]:
                raise ValueError("validation embedding dimension differs from training")
            positions = np.searchsorted(classes, validation_labels)
            clipped = np.minimum(positions, len(classes) - 1)
            if np.any(positions == len(classes)) or not np.array_equal(
                classes[clipped], validation_labels
            ):
                raise ValueError("validation labels contain a class absent from training")
            validation_positions = positions.astype(np.int64)
            reference_validation_sha256 = _dataset_sha256(validation_features, validation_labels)

        device = _resolve_device(self.config.device)
        if self.config.amp and device.type != "cuda":
            raise RuntimeError("AMP was requested, but AMP/GradScaler is supported only on CUDA")
        _seed_everything(self.config.seed)
        self.classes_ = classes
        self.input_dimension_ = features.shape[1]
        self.feature_mean_ = features.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
        self.feature_scale_ = np.where(scale > 1e-7, scale, 1.0).astype(np.float32)
        self._network = self._build_network(features.shape[1], len(classes), device)
        self.device_ = str(device)
        optimiser = torch.optim.AdamW(
            self._network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True) if self.config.amp else None
        fingerprint = _dataset_sha256(features, labels)
        if resume and path is not None:
            self._restore_training_checkpoint(
                path,
                optimiser,
                scaler,
                features,
                classes,
                fingerprint,
                reference_validation_sha256,
                device,
            )
        else:
            self.completed_epochs_ = 0
            self.history_ = ()
            self.effective_batch_size_ = self.config.batch_size
            self.best_epoch_ = None
            self.best_validation_loss_ = None
            self.epochs_without_improvement_ = 0
            self.stopped_early_ = False
            self._best_network_state = None
            self._batch_backoff_events = []
            self._prior_runtime_seconds = 0.0
            self._successful_samples_processed = 0
            self._optimiser_steps = 0

        normalised = self._normalise(features)
        train_x = torch.from_numpy(normalised)
        train_y = torch.from_numpy(encoded)
        if self.config.class_weight == "balanced":
            counts = np.bincount(encoded, minlength=len(classes)).astype(np.float64)
            weights = len(encoded) / (len(classes) * counts)
            loss_weights = torch.from_numpy(weights.astype(np.float32)).to(device)
        else:
            loss_weights = None
        criterion = nn.CrossEntropyLoss(weight=loss_weights, reduction="none")

        validation_tensors: tuple[torch.Tensor, torch.Tensor] | None = None
        if validation_features is not None and validation_positions is not None:
            validation_tensors = (
                torch.from_numpy(self._normalise(validation_features)),
                torch.from_numpy(validation_positions),
            )

        failure: str | None = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            history = list(self.history_)
            if not self.stopped_early_:
                for epoch in range(self.completed_epochs_, self.config.epochs):
                    _seed_everything(self.config.seed + epoch)
                    order = (
                        np.random.default_rng(self.config.seed + epoch)
                        .permutation(len(features))
                        .astype(np.int64)
                    )
                    self._network.train()

                    def execute(indices: NDArray[np.int64], batch_size: int) -> float:
                        return self._execute_training_window(
                            train_x,
                            train_y,
                            indices,
                            batch_size,
                            optimiser,
                            scaler,
                            criterion,
                            device,
                        )

                    def clear_cuda_oom() -> None:
                        optimiser.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()

                    execution = _execute_epoch_with_backoff(
                        order,
                        epoch=epoch + 1,
                        phase="training",
                        initial_batch_size=self.effective_batch_size_,
                        minimum_batch_size=self.config.minimum_batch_size,
                        accumulation_steps=self.config.gradient_accumulation_steps,
                        cuda_oom_recovery=device.type == "cuda",
                        execute_window=execute,
                        snapshot_state=lambda: self._training_state_snapshot(optimiser, scaler),
                        restore_state=lambda snapshot: self._restore_training_state(
                            cast(dict[str, Any], snapshot), optimiser, scaler, device
                        ),
                        clear_after_oom=clear_cuda_oom,
                        backoff_events=self._batch_backoff_events,
                    )
                    self.effective_batch_size_ = execution.final_batch_size
                    self._successful_samples_processed += execution.successful_sample_count
                    self._optimiser_steps += execution.optimiser_steps

                    validation_loss: float | None = None
                    if validation_tensors is not None:
                        validation_loss = self._reference_validation_loss(
                            validation_tensors[0],
                            validation_tensors[1],
                            criterion,
                            device,
                            epoch + 1,
                        )
                    self.completed_epochs_ = epoch + 1
                    improved = False
                    if (
                        self.config.early_stopping_patience is not None
                        and validation_loss is not None
                    ):
                        improved = self.best_validation_loss_ is None or validation_loss < (
                            self.best_validation_loss_ - self.config.early_stopping_min_delta
                        )
                        if improved:
                            self.best_validation_loss_ = validation_loss
                            self.best_epoch_ = self.completed_epochs_
                            self.epochs_without_improvement_ = 0
                            self._best_network_state = _model_state_to_cpu(self._network)
                        else:
                            self.epochs_without_improvement_ += 1
                        self.stopped_early_ = (
                            self.epochs_without_improvement_ >= self.config.early_stopping_patience
                        )
                    history.append(
                        {
                            "epoch": self.completed_epochs_,
                            "training_loss": execution.loss_sum / len(features),
                            "validation_loss": validation_loss,
                            "effective_batch_size": self.effective_batch_size_,
                            "optimiser_steps": execution.optimiser_steps,
                            "early_stopping_improved": improved,
                            "epochs_without_improvement": self.epochs_without_improvement_,
                            "stopped_early": self.stopped_early_,
                        }
                    )
                    self.history_ = tuple(history)
                    if path is not None:
                        _atomic_torch_save(
                            path,
                            self._checkpoint_payload(
                                optimiser,
                                scaler,
                                fingerprint,
                                reference_validation_sha256,
                                device,
                                time.perf_counter() - fit_started,
                            ),
                        )
                    if self.stopped_early_:
                        break
            if self.config.early_stopping_patience is not None:
                if self._best_network_state is None:
                    raise RuntimeError("early stopping completed without a best model state")
                self._network.load_state_dict(self._best_network_state, strict=True)
                self._network.to(device)
            self._network.eval()
        except BaseException as error:
            failure = f"{type(error).__name__}: {error}"
            raise
        finally:
            if device.type == "cuda":
                with suppress(RuntimeError):
                    torch.cuda.synchronize(device)
            self.telemetry_ = self._telemetry_snapshot(
                device, time.perf_counter() - fit_started, failure=failure
            )
        return self

    def predict_proba(
        self,
        embeddings: NDArray[np.generic],
        *,
        batch_size: int | None = None,
    ) -> NDArray[np.float64]:
        """Return class probabilities in ``classes_`` order."""

        if self._network is None or self.classes_ is None or self.input_dimension_ is None:
            raise RuntimeError("classifier is not fitted")
        features = _matrix(embeddings, name="embeddings")
        if features.shape[1] != self.input_dimension_:
            raise ValueError("prediction embedding dimension differs from training")
        device = next(self._network.parameters()).device
        size = batch_size or self.effective_batch_size_
        if size <= 0:
            raise ValueError("prediction batch_size must be positive")
        normalised = torch.from_numpy(self._normalise(features))
        chunks: list[NDArray[np.float64]] = []
        self._network.eval()
        with torch.inference_mode():
            for start in range(0, len(features), size):
                batch = normalised[start : start + size].to(device)
                if self.config.amp:
                    with torch.autocast(
                        device_type="cuda", dtype=self._amp_torch_dtype(), enabled=True
                    ):
                        logits = self._network(batch)
                else:
                    logits = self._network(batch)
                chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy().astype(np.float64))
        probabilities = np.concatenate(chunks, axis=0)
        if not np.isfinite(probabilities).all() or not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=1e-6
        ):
            raise RuntimeError("MLP produced invalid class probabilities")
        return probabilities

    def predict(self, embeddings: NDArray[np.generic]) -> NDArray[np.int64]:
        """Return labels using the fitted stable class order."""

        if self.classes_ is None:
            raise RuntimeError("classifier is not fitted")
        return self.classes_[np.argmax(self.predict_proba(embeddings), axis=1)]


__all__ = ["FrozenEmbeddingMLPClassifier", "FrozenEmbeddingMLPConfig"]
