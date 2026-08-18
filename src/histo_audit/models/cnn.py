"""Leakage-guarded ResNet-18 trainer for the confirmatory image scenarios.

The study-facing classifier is deliberately CUDA-only and cannot download
weights.  A separately named CPU adapter exists solely for fast structural
tests; its checkpoints and telemetry are permanently marked ineligible for
study outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol, Self, cast, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as functional
from numpy.typing import NDArray
from torch import nn
from torchvision.models import resnet18  # type: ignore[import-untyped]

from histo_audit.models.mlp import (
    _atomic_torch_save,
    _capture_rng_state,
    _clone_to_cpu,
    _execute_epoch_with_backoff,
    _model_state_to_cpu,
    _optimiser_to_device,
    _restore_rng_state,
)
from histo_audit.representations.imagenet import (
    PretrainedWeightsUnavailableError,
    official_resnet18_weight_cache_path,
)
from histo_audit.utils.run_tracking import sha256_file

CLASS_ORDER: tuple[int, ...] = (0, 1, 2, 3, 4)
OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER = "ResNet18_Weights.IMAGENET1K_V1"
CPU_TEST_ONLY_WEIGHT_IDENTIFIER = "TEST_ONLY_RANDOM_SEEDED"

InputVariant = Literal["context_rgb", "context_rgb_plus_binary_target_mask"]
AmpDtype = Literal["float16", "bfloat16"]
ValidationRole = Literal["reference_validation"]
FourthChannelInitialisation = Literal["zeros"]
type HistoryRow = dict[str, float | int | bool | str | None]
type BackoffEvent = dict[str, float | int | str]


@runtime_checkable
class _IndexedRows(Protocol):
    """Structural boundary that avoids importing the experiment package here."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype[Any]: ...

    @property
    def ndim(self) -> int: ...

    def __len__(self) -> int: ...

    def select_rows(
        self,
        local_indices: Sequence[int] | NDArray[np.integer[Any]] | slice,
        *,
        allow_repeated_indices: bool = False,
    ) -> _IndexedRows: ...

    def gather_rows(
        self,
        local_indices: Sequence[int] | NDArray[np.integer[Any]] | slice | None = None,
        *,
        max_rows: int | None = None,
        allow_repeated_indices: bool = False,
    ) -> NDArray[Any]: ...

    def iter_chunks(self, max_chunk_bytes: int = 32 * 1024 * 1024) -> Iterator[NDArray[Any]]: ...


type ImageArray = NDArray[np.generic] | _IndexedRows


@dataclass(frozen=True, slots=True)
class ConfirmatoryCNNConfig:
    """Frozen training controls for a five-class confirmatory ResNet-18."""

    input_variant: InputVariant
    weight_identifier: str = OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER
    fourth_channel_initialisation: FourthChannelInitialisation = "zeros"
    input_size: int = 224
    epochs: int = 30
    batch_size: int = 32
    minimum_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    amp_dtype: AmpDtype = "float16"
    class_weight_balanced: bool = True
    seed: int = 0

    def validate(self) -> None:
        """Reject ambiguous architecture or training settings."""

        if self.input_variant not in (
            "context_rgb",
            "context_rgb_plus_binary_target_mask",
        ):
            raise ValueError(f"unsupported input_variant: {self.input_variant!r}")
        if self.weight_identifier not in {
            OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER,
            CPU_TEST_ONLY_WEIGHT_IDENTIFIER,
        }:
            raise ValueError(
                "weight_identifier must be the explicit official ImageNet identifier or "
                "the CPU test-only random identifier"
            )
        if self.fourth_channel_initialisation != "zeros":
            raise ValueError("fourth channel must use deterministic zero initialisation")
        if self.input_size < 32:
            raise ValueError("input_size must be at least 32 pixels")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0 or self.minimum_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.minimum_batch_size > self.batch_size:
            raise ValueError("minimum_batch_size cannot exceed batch_size")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if not np.isfinite(self.early_stopping_min_delta) or self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be finite and non-negative")
        if self.amp_dtype not in ("float16", "bfloat16"):
            raise ValueError(f"unsupported AMP dtype: {self.amp_dtype!r}")


@dataclass(frozen=True, slots=True)
class _PreparedImages:
    rgb: NDArray[np.uint8] | _IndexedRows
    target_masks: NDArray[np.bool_] | _IndexedRows | None
    input_variant: InputVariant

    @property
    def sample_count(self) -> int:
        return len(self.rgb)

    @property
    def channel_count(self) -> int:
        return 4 if self.target_masks is not None else 3


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _uint8_rgb(
    images: ImageArray,
    *,
    name: str,
) -> NDArray[np.uint8] | _IndexedRows:
    if isinstance(images, _IndexedRows):
        if images.ndim != 4 or images.shape[0] == 0 or images.shape[-1] != 3:
            raise ValueError(f"{name} must have non-empty shape (n, height, width, 3)")
        if images.shape[1] == 0 or images.shape[2] == 0:
            raise ValueError(f"{name} spatial dimensions must be non-empty")
        if images.dtype != np.dtype(np.uint8):
            raise ValueError(f"{name} indexed RGB cache must expose exact uint8 values")
        return images
    array = np.asarray(images)
    if array.ndim != 4 or array.shape[0] == 0 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have non-empty shape (n, height, width, 3)")
    if array.shape[1] == 0 or array.shape[2] == 0:
        raise ValueError(f"{name} spatial dimensions must be non-empty")
    if array.dtype == np.uint8:
        return cast(
            NDArray[np.uint8],
            array if array.flags.c_contiguous else np.ascontiguousarray(array),
        )
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric RGB values")
    converted = array.astype(np.float64)
    if float(converted.min()) < 0.0:
        raise ValueError(f"{name} RGB values must be non-negative")
    maximum = float(converted.max())
    if np.issubdtype(array.dtype, np.floating) and maximum <= 1.0:
        converted *= 255.0
    elif maximum > 255.0:
        raise ValueError(f"{name} RGB values must lie in [0, 255] or [0, 1]")
    return np.ascontiguousarray(np.rint(converted).astype(np.uint8))


def _binary_masks(
    masks: ImageArray,
    expected_shape: tuple[int, int, int],
    *,
    name: str,
) -> NDArray[np.bool_] | _IndexedRows:
    if isinstance(masks, _IndexedRows):
        if masks.shape != expected_shape:
            raise ValueError(f"{name} must align exactly with RGB images")
        if masks.dtype != np.dtype(np.bool_):
            raise ValueError(f"{name} indexed cache must expose exact boolean values")
        for chunk in masks.iter_chunks():
            if not chunk.reshape(len(chunk), -1).any(axis=1).all():
                raise ValueError(f"every {name} row must identify a non-empty target")
        return masks
    array = np.asarray(masks)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must align exactly with RGB images")
    if array.dtype == np.bool_:
        result = cast(
            NDArray[np.bool_],
            array if array.flags.c_contiguous else np.ascontiguousarray(array),
        )
    else:
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"{name} must be boolean or numeric binary values")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
        if not np.isin(array, (0, 1)).all():
            raise ValueError(f"{name} must be exactly binary (0/1)")
        result = np.ascontiguousarray(array.astype(bool))
    if not result.reshape(len(result), -1).any(axis=1).all():
        raise ValueError(f"every {name} row must identify a non-empty target")
    return result


def _prepare_images(
    images: ImageArray,
    target_masks: ImageArray | None,
    input_variant: InputVariant,
    *,
    name: str,
) -> _PreparedImages:
    rgb = _uint8_rgb(images, name=name)
    if input_variant == "context_rgb":
        if target_masks is not None:
            raise ValueError("context_rgb does not accept target_masks")
        masks = None
    else:
        if target_masks is None:
            raise ValueError("context_rgb_plus_binary_target_mask requires exact target_masks")
        masks = _binary_masks(
            target_masks,
            cast(tuple[int, int, int], rgb.shape[:3]),
            name=f"{name} target_masks",
        )
    return _PreparedImages(rgb=rgb, target_masks=masks, input_variant=input_variant)


def _labels(values: NDArray[np.generic] | Sequence[int], n: int, *, name: str) -> NDArray[np.int64]:
    labels = np.asarray(values)
    if labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array aligned with images")
    result = labels.astype(np.int64)
    if not np.isin(result, CLASS_ORDER).all():
        raise ValueError(f"{name} contains values outside fixed class order {CLASS_ORDER}")
    return result


def _identifiers(
    values: Sequence[str] | NDArray[np.str_], n: int, *, name: str
) -> NDArray[np.str_]:
    identifiers = np.asarray(values, dtype=np.str_)
    if identifiers.shape != (n,) or any(not item for item in identifiers.tolist()):
        raise ValueError(f"{name} must be non-empty strings aligned with images")
    if name.endswith("sample_ids") and len(set(identifiers.tolist())) != n:
        raise ValueError(f"{name} must be unique")
    return identifiers


def _update_array_hash(
    digest: Any,
    array: NDArray[np.generic] | _IndexedRows,
) -> None:
    if isinstance(array, _IndexedRows):
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        for chunk in array.iter_chunks():
            digest.update(memoryview(np.ascontiguousarray(chunk)).cast("B"))
        return
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes())


def _dataset_sha256(prepared: _PreparedImages, labels: NDArray[np.int64]) -> str:
    digest = hashlib.sha256()
    _update_array_hash(digest, prepared.rgb)
    if prepared.target_masks is not None:
        _update_array_hash(digest, prepared.target_masks)
    else:
        digest.update(b"no_target_mask")
    _update_array_hash(digest, labels)
    return digest.hexdigest()


def _split_sha256(sample_ids: NDArray[np.str_], group_ids: NDArray[np.str_]) -> str:
    digest = hashlib.sha256()
    _update_array_hash(digest, sample_ids)
    _update_array_hash(digest, group_ids)
    return digest.hexdigest()


def confirmatory_cnn_data_and_split_sha256(
    training_images: ImageArray,
    training_labels: NDArray[np.generic] | Sequence[int],
    *,
    training_sample_ids: Sequence[str] | NDArray[np.str_],
    training_group_ids: Sequence[str] | NDArray[np.str_],
    reference_validation_images: ImageArray,
    reference_validation_labels: NDArray[np.generic] | Sequence[int],
    reference_validation_sample_ids: Sequence[str] | NDArray[np.str_],
    reference_validation_group_ids: Sequence[str] | NDArray[np.str_],
    input_variant: InputVariant,
    training_target_masks: ImageArray | None = None,
    reference_validation_target_masks: ImageArray | None = None,
) -> dict[str, str]:
    """Precompute the exact checkpoint fingerprints from independently held inputs."""

    training = _prepare_images(
        training_images,
        training_target_masks,
        input_variant,
        name="training_images",
    )
    validation = _prepare_images(
        reference_validation_images,
        reference_validation_target_masks,
        input_variant,
        name="reference_validation_images",
    )
    train_labels = _labels(training_labels, training.sample_count, name="training_labels")
    validation_labels = _labels(
        reference_validation_labels,
        validation.sample_count,
        name="reference_validation_labels",
    )
    train_ids = _identifiers(
        training_sample_ids,
        training.sample_count,
        name="training_sample_ids",
    )
    train_groups = _identifiers(
        training_group_ids,
        training.sample_count,
        name="training_group_ids",
    )
    validation_ids = _identifiers(
        reference_validation_sample_ids,
        validation.sample_count,
        name="reference_validation_sample_ids",
    )
    validation_groups = _identifiers(
        reference_validation_group_ids,
        validation.sample_count,
        name="reference_validation_group_ids",
    )
    return {
        "training_data_sha256": _dataset_sha256(training, train_labels),
        "reference_validation_data_sha256": _dataset_sha256(
            validation,
            validation_labels,
        ),
        "training_split_sha256": _split_sha256(train_ids, train_groups),
        "reference_validation_split_sha256": _split_sha256(
            validation_ids,
            validation_groups,
        ),
    }


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _resume_contract_payload(config: ConfirmatoryCNNConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("epochs")
    return payload


def _load_official_weights_without_download(
    identifier: str,
) -> tuple[dict[str, torch.Tensor], Path]:
    if identifier != OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER:
        raise ValueError("study mode requires ResNet18_Weights.IMAGENET1K_V1")
    path = official_resnet18_weight_cache_path(identifier)
    if not path.is_file():
        raise PretrainedWeightsUnavailableError(
            "official ResNet-18 weights are not present in the local torch cache at "
            f"{path}; this trainer never downloads weights implicitly"
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise RuntimeError("cached official ResNet-18 checkpoint has an invalid state dictionary")
    return cast(dict[str, torch.Tensor], state), path


def _build_resnet18(
    config: ConfirmatoryCNNConfig,
    *,
    cpu_test_only: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build a deterministic 3/4-channel, five-output ResNet-18."""

    config.validate()
    if cpu_test_only:
        if config.weight_identifier != CPU_TEST_ONLY_WEIGHT_IDENTIFIER:
            raise ValueError("the CPU test-only adapter requires TEST_ONLY_RANDOM_SEEDED weights")
        state: dict[str, torch.Tensor] | None = None
        weight_path: Path | None = None
        weight_sha256: str | None = None
    else:
        if config.weight_identifier == CPU_TEST_ONLY_WEIGHT_IDENTIFIER:
            raise ValueError("TEST_ONLY_RANDOM_SEEDED weights are forbidden in study mode")
        state, weight_path = _load_official_weights_without_download(config.weight_identifier)
        weight_sha256 = sha256_file(weight_path)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        network = resnet18(weights=None)
        if state is not None:
            network.load_state_dict(state, strict=True)
        # Initialise the five-class head before creating the extra convolutional
        # channel so the RGB and target-mask scenarios share the same seeded head.
        network.fc = nn.Linear(network.fc.in_features, len(CLASS_ORDER))
        if config.input_variant == "context_rgb_plus_binary_target_mask":
            original = network.conv1
            expanded = nn.Conv2d(
                4,
                original.out_channels,
                kernel_size=original.kernel_size,
                stride=original.stride,
                padding=original.padding,
                bias=original.bias is not None,
            )
            with torch.no_grad():
                expanded.weight[:, :3].copy_(original.weight)
                expanded.weight[:, 3:4].zero_()
                if original.bias is not None and expanded.bias is not None:
                    expanded.bias.copy_(original.bias)
            network.conv1 = expanded

    metadata = {
        "architecture": "torchvision.resnet18",
        "class_order": list(CLASS_ORDER),
        "input_channels": 4 if config.input_variant.endswith("target_mask") else 3,
        "weight_identifier": config.weight_identifier,
        "weight_path": str(weight_path.resolve()) if weight_path is not None else None,
        "weight_sha256": weight_sha256,
        "implicit_weight_download": False,
        "preprocessing": {
            "rgb_resize": "bilinear_antialias",
            "rgb_range_before_normalisation": [0.0, 1.0],
            "rgb_mean": [0.485, 0.456, 0.406],
            "rgb_standard_deviation": [0.229, 0.224, 0.225],
            "target_mask_resize": (
                "nearest_binary_unnormalised"
                if config.input_variant == "context_rgb_plus_binary_target_mask"
                else None
            ),
        },
        "fourth_channel_initialisation": (
            config.fourth_channel_initialisation
            if config.input_variant == "context_rgb_plus_binary_target_mask"
            else None
        ),
    }
    return network, metadata


@lru_cache(maxsize=2)
def _confirmatory_state_schema(input_channels: int) -> dict[str, tuple[tuple[int, ...], str]]:
    network = resnet18(weights=None)
    network.fc = nn.Linear(network.fc.in_features, len(CLASS_ORDER))
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
    elif input_channels != 3:
        raise ValueError("confirmatory checkpoint input channels must be 3 or 4")
    return {
        key: (tuple(int(value) for value in tensor.shape), str(tensor.dtype))
        for key, tensor in network.state_dict().items()
    }


def _validate_checkpoint_state_dict(
    value: Any,
    *,
    input_channels: int,
    role: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"confirmatory checkpoint {role} must be a state dictionary")
    expected = _confirmatory_state_schema(input_channels)
    if set(value) != set(expected):
        raise ValueError(f"confirmatory checkpoint {role} keys differ from ResNet-18")
    for key, (shape, dtype) in expected.items():
        tensor = value[key]
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(int(item) for item in tensor.shape) != shape
            or str(tensor.dtype) != dtype
            or tensor.layout != torch.strided
        ):
            raise ValueError(f"confirmatory checkpoint {role} tensor schema differs for {key}")
        if tensor.is_floating_point() and not _checkpoint_tensor_all_finite(tensor):
            raise ValueError(f"confirmatory checkpoint {role} contains non-finite {key}")
    conv1 = value.get("conv1.weight")
    if not isinstance(conv1, torch.Tensor) or not _checkpoint_tensor_has_nonzero(conv1):
        raise ValueError(f"confirmatory checkpoint {role} has an empty learned state")


def _checkpoint_tensor_all_finite(tensor: torch.Tensor) -> bool:
    if tensor.numel() == 0:
        return True
    if tensor.ndim and all(stride == 0 for stride in tensor.stride()):
        return bool(torch.isfinite(tensor[(0,) * tensor.ndim]))
    return bool(torch.isfinite(tensor).all())


def _checkpoint_tensor_has_nonzero(tensor: torch.Tensor) -> bool:
    if tensor.numel() == 0:
        return False
    if tensor.ndim and all(stride == 0 for stride in tensor.stride()):
        return bool(tensor[(0,) * tensor.ndim] != 0)
    return bool(torch.count_nonzero(tensor))


@lru_cache(maxsize=2)
def _confirmatory_parameter_schema(
    input_channels: int,
) -> tuple[tuple[tuple[int, ...], str], ...]:
    network = resnet18(weights=None)
    network.fc = nn.Linear(network.fc.in_features, len(CLASS_ORDER))
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
    elif input_channels != 3:
        raise ValueError("confirmatory checkpoint input channels must be 3 or 4")
    return tuple(
        (tuple(int(value) for value in parameter.shape), str(parameter.dtype))
        for parameter in network.parameters()
    )


def _validate_adamw_state(
    value: Any,
    *,
    input_channels: int,
    configuration: Mapping[str, Any],
    successful_optimiser_steps: int,
    skipped_optimiser_steps: int,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"state", "param_groups"}:
        raise ValueError("confirmatory checkpoint AdamW state schema is invalid")
    state = value["state"]
    groups = value["param_groups"]
    parameter_schema = _confirmatory_parameter_schema(input_channels)
    parameter_ids = list(range(len(parameter_schema)))
    if (
        not isinstance(state, Mapping)
        or not isinstance(groups, list)
        or len(groups) != 1
        or not isinstance(groups[0], Mapping)
    ):
        raise ValueError("confirmatory checkpoint AdamW state is incomplete")
    group = groups[0]
    expected_group_fields = {
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
        "decoupled_weight_decay",
        "params",
    }
    if (
        set(group) != expected_group_fields
        or group.get("params") != parameter_ids
        or group.get("lr") != configuration["learning_rate"]
        or group.get("weight_decay") != configuration["weight_decay"]
        or tuple(group.get("betas", ())) != (0.9, 0.999)
        or group.get("eps") != 1e-8
        or group.get("amsgrad") is not False
        or group.get("maximize") is not False
        or group.get("foreach") is not None
        or group.get("capturable") is not False
        or group.get("differentiable") is not False
        or group.get("fused") is not None
        or group.get("decoupled_weight_decay") is not True
    ):
        raise ValueError("confirmatory checkpoint AdamW parameter group is invalid")
    if (
        not state
        or set(state) != set(parameter_ids)
        or successful_optimiser_steps <= 0
        or skipped_optimiser_steps < 0
    ):
        raise ValueError("confirmatory checkpoint AdamW state is incomplete")
    for parameter_id, (shape, dtype) in enumerate(parameter_schema):
        row = state[parameter_id]
        if not isinstance(row, Mapping) or set(row) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("confirmatory checkpoint AdamW moment state is invalid")
        step = row["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.shape != ()
            or not step.is_floating_point()
            or not bool(torch.isfinite(step))
            or float(step) < 1.0
        ):
            raise ValueError("confirmatory checkpoint AdamW step is invalid")
        for field_name in ("exp_avg", "exp_avg_sq"):
            tensor = row[field_name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(int(value) for value in tensor.shape) != shape
                or str(tensor.dtype) != dtype
                or not _checkpoint_tensor_all_finite(tensor)
            ):
                raise ValueError(f"confirmatory checkpoint AdamW {field_name} tensor is invalid")


def _validate_grad_scaler_state(value: Any) -> None:
    expected_fields = {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("confirmatory checkpoint CUDA GradScaler state is invalid")
    if (
        not math.isfinite(float(value["scale"]))
        or float(value["scale"]) <= 0.0
        or not math.isfinite(float(value["growth_factor"]))
        or float(value["growth_factor"]) <= 1.0
        or not math.isfinite(float(value["backoff_factor"]))
        or not 0.0 < float(value["backoff_factor"]) < 1.0
        or type(value["growth_interval"]) is not int
        or int(value["growth_interval"]) <= 0
        or type(value["_growth_tracker"]) is not int
        or int(value["_growth_tracker"]) < 0
    ):
        raise ValueError("confirmatory checkpoint CUDA GradScaler values are invalid")


def _validate_rng_state(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise ValueError("confirmatory checkpoint RNG state schema is invalid")
    python_state = value["python"]
    numpy_state = value["numpy"]
    torch_cpu = value["torch_cpu"]
    torch_cuda = value["torch_cuda"]
    if (
        not isinstance(python_state, tuple)
        or len(python_state) != 3
        or python_state[0] != 3
        or not isinstance(python_state[1], tuple)
        or len(python_state[1]) != 625
        or not isinstance(numpy_state, Mapping)
        or set(numpy_state) != {"bit_generator", "keys", "position", "has_gauss", "cached_gaussian"}
        or numpy_state.get("bit_generator") != "MT19937"
        or not isinstance(numpy_state.get("keys"), torch.Tensor)
        or numpy_state["keys"].dtype != torch.uint32
        or numpy_state["keys"].shape != (624,)
        or type(numpy_state.get("position")) is not int
        or not 0 <= int(numpy_state["position"]) <= 624
        or numpy_state.get("has_gauss") not in {0, 1}
        or not math.isfinite(float(numpy_state.get("cached_gaussian", math.nan)))
        or not isinstance(torch_cpu, torch.Tensor)
        or torch_cpu.dtype != torch.uint8
        or torch_cpu.ndim != 1
        or torch_cpu.numel() == 0
        or not isinstance(torch_cuda, list)
        or not torch_cuda
        or any(
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 1
            or state.numel() == 0
            for state in torch_cuda
        )
    ):
        raise ValueError("confirmatory checkpoint RNG state is invalid")


def _validate_history_and_telemetry(
    history: Any,
    early: Mapping[str, Any],
    telemetry: Any,
    *,
    completed_epochs: int,
    effective_batch_size: int,
    configuration: Mapping[str, Any],
) -> None:
    history_fields = {
        "epoch",
        "training_loss",
        "reference_validation_loss",
        "effective_batch_size",
        "optimiser_steps",
        "successful_optimiser_steps",
        "skipped_optimiser_steps",
        "early_stopping_improved",
        "epochs_without_improvement",
        "stopped_early",
    }
    if not isinstance(history, list) or len(history) != completed_epochs:
        raise ValueError("confirmatory checkpoint training history is invalid")
    for epoch, row in enumerate(history, start=1):
        if (
            not isinstance(row, Mapping)
            or set(row) != history_fields
            or row.get("epoch") != epoch
            or not math.isfinite(float(row.get("training_loss", math.nan)))
            or float(row["training_loss"]) < 0.0
            or not math.isfinite(float(row.get("reference_validation_loss", math.nan)))
            or float(row["reference_validation_loss"]) < 0.0
            or type(row.get("effective_batch_size")) is not int
            or not int(configuration["minimum_batch_size"])
            <= int(row["effective_batch_size"])
            <= int(configuration["batch_size"])
            or type(row.get("optimiser_steps")) is not int
            or int(row["optimiser_steps"]) <= 0
            or type(row.get("successful_optimiser_steps")) is not int
            or int(row["successful_optimiser_steps"]) < 0
            or type(row.get("skipped_optimiser_steps")) is not int
            or int(row["skipped_optimiser_steps"]) < 0
            or int(row["successful_optimiser_steps"]) + int(row["skipped_optimiser_steps"])
            != int(row["optimiser_steps"])
            or type(row.get("early_stopping_improved")) is not bool
            or type(row.get("epochs_without_improvement")) is not int
            or int(row["epochs_without_improvement"]) < 0
            or type(row.get("stopped_early")) is not bool
            or (row["early_stopping_improved"] and row["epochs_without_improvement"] != 0)
        ):
            raise ValueError("confirmatory checkpoint training history row is invalid")
    best_epoch = early.get("best_epoch")
    best_loss = early.get("best_validation_loss")
    if type(best_epoch) is not int or (
        not isinstance(best_loss, (int, float)) or isinstance(best_loss, bool)
    ):
        raise ValueError("confirmatory checkpoint early-stopping values are invalid")
    best_epoch_value = int(best_epoch)
    best_loss_value = float(best_loss)
    last = history[-1]
    if (
        not 1 <= best_epoch_value <= completed_epochs
        or not math.isfinite(best_loss_value)
        or best_loss_value != float(history[best_epoch_value - 1]["reference_validation_loss"])
        or early.get("epochs_without_improvement") != last["epochs_without_improvement"]
        or early.get("stopped_early") != last["stopped_early"]
        or last["effective_batch_size"] != effective_batch_size
    ):
        raise ValueError("confirmatory checkpoint early-stopping history is inconsistent")
    telemetry_fields = {
        "schema_version",
        "execution_mode",
        "study_outcome_eligible",
        "device",
        "amp_enabled",
        "amp_dtype",
        "grad_scaler_enabled",
        "gradient_accumulation_steps",
        "initial_batch_size",
        "effective_batch_size",
        "minimum_batch_size",
        "batch_backoff_events",
        "current_fit_runtime_seconds",
        "cumulative_runtime_seconds",
        "cuda_peak_memory_allocated_bytes",
        "cuda_peak_memory_reserved_bytes",
        "completed_epochs",
        "requested_epochs",
        "successful_samples_processed",
        "optimiser_steps",
        "successful_optimiser_steps",
        "skipped_optimiser_steps",
        "early_stopping_source",
        "best_epoch",
        "best_validation_loss",
        "epochs_without_improvement",
        "stopped_early",
        "failure",
    }
    if not isinstance(telemetry, Mapping) or set(telemetry) != telemetry_fields:
        raise ValueError("confirmatory checkpoint CUDA telemetry schema is invalid")
    if (
        telemetry.get("schema_version") != 1
        or telemetry.get("execution_mode") != "real_study_cuda"
        or telemetry.get("study_outcome_eligible") is not True
        or not str(telemetry.get("device", "")).startswith("cuda")
        or telemetry.get("amp_enabled") is not True
        or telemetry.get("amp_dtype") != configuration["amp_dtype"]
        or telemetry.get("grad_scaler_enabled") is not True
        or telemetry.get("gradient_accumulation_steps")
        != configuration["gradient_accumulation_steps"]
        or telemetry.get("initial_batch_size") != configuration["batch_size"]
        or telemetry.get("effective_batch_size") != effective_batch_size
        or telemetry.get("minimum_batch_size") != configuration["minimum_batch_size"]
        or not isinstance(telemetry.get("batch_backoff_events"), list)
        or not math.isfinite(float(telemetry.get("current_fit_runtime_seconds", math.nan)))
        or float(telemetry["current_fit_runtime_seconds"]) < 0.0
        or not math.isfinite(float(telemetry.get("cumulative_runtime_seconds", math.nan)))
        or float(telemetry["cumulative_runtime_seconds"])
        < float(telemetry["current_fit_runtime_seconds"])
        or telemetry.get("completed_epochs") != completed_epochs
        or telemetry.get("requested_epochs") != configuration["epochs"]
        or type(telemetry.get("successful_samples_processed")) is not int
        or int(telemetry["successful_samples_processed"]) <= 0
        or telemetry.get("optimiser_steps") != sum(int(row["optimiser_steps"]) for row in history)
        or type(telemetry.get("successful_optimiser_steps")) is not int
        or int(telemetry["successful_optimiser_steps"]) < 0
        or type(telemetry.get("skipped_optimiser_steps")) is not int
        or int(telemetry["skipped_optimiser_steps"]) < 0
        or int(telemetry["successful_optimiser_steps"]) + int(telemetry["skipped_optimiser_steps"])
        != int(telemetry["optimiser_steps"])
        or int(telemetry["successful_optimiser_steps"])
        != sum(int(row["successful_optimiser_steps"]) for row in history)
        or int(telemetry["skipped_optimiser_steps"])
        != sum(int(row["skipped_optimiser_steps"]) for row in history)
        or telemetry.get("early_stopping_source") != "reference_validation_only"
        or telemetry.get("best_epoch") != best_epoch
        or float(cast(float | int, telemetry.get("best_validation_loss", math.nan)))
        != best_loss_value
        or telemetry.get("epochs_without_improvement") != early.get("epochs_without_improvement")
        or telemetry.get("stopped_early") != early.get("stopped_early")
        or telemetry.get("failure") is not None
    ):
        raise ValueError("confirmatory checkpoint CUDA telemetry is inconsistent")
    for field_name in (
        "cuda_peak_memory_allocated_bytes",
        "cuda_peak_memory_reserved_bytes",
    ):
        value = telemetry[field_name]
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("confirmatory checkpoint CUDA memory telemetry is invalid")


def validate_confirmatory_checkpoint_artifact(
    path: str | Path,
    *,
    expected_configuration: Mapping[str, Any],
    expected_model_metadata: Mapping[str, Any],
    expected_data_and_split_sha256: Mapping[str, str],
) -> None:
    """Strictly validate a sealed real-CUDA checkpoint without executing pickle code."""

    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, pickle.UnpicklingError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("confirmatory checkpoint is not a safe Torch payload") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("confirmatory checkpoint payload must be a mapping")
    required_fields = {
        "schema_version",
        "model_kind",
        "execution_mode",
        "study_outcome_eligible",
        "configuration",
        "configuration_sha256",
        "resume_contract_sha256",
        "data_and_split_sha256",
        "model_metadata",
        "class_order",
        "completed_epochs",
        "network_state_dict",
        "optimiser_state_dict",
        "scaler_state_dict",
        "history",
        "effective_batch_size",
        "rng_state",
        "early_stopping_state",
        "telemetry",
    }
    if set(payload) != required_fields:
        raise ValueError("confirmatory checkpoint has an invalid exact schema")
    configuration = dict(expected_configuration)
    stored_configuration = payload.get("configuration")
    if (
        payload.get("schema_version") != 1
        or payload.get("model_kind") != "confirmatory_resnet18_five_class"
        or payload.get("execution_mode") != "real_study_cuda"
        or payload.get("study_outcome_eligible") is not True
        or stored_configuration != configuration
        or payload.get("configuration_sha256") != _json_sha256(configuration)
    ):
        raise ValueError("confirmatory checkpoint mode/configuration is invalid")
    resume_configuration = dict(configuration)
    resume_configuration.pop("epochs", None)
    if payload.get("resume_contract_sha256") != _json_sha256(resume_configuration):
        raise ValueError("confirmatory checkpoint resume contract is invalid")
    hashes = payload.get("data_and_split_sha256")
    expected_hashes = dict(expected_data_and_split_sha256)
    if (
        not isinstance(hashes, Mapping)
        or set(hashes)
        != {
            "training_data_sha256",
            "reference_validation_data_sha256",
            "training_split_sha256",
            "reference_validation_split_sha256",
        }
        or dict(hashes) != expected_hashes
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes.values()
        )
    ):
        raise ValueError("confirmatory checkpoint data/split fingerprints are invalid")
    metadata = payload.get("model_metadata")
    if not isinstance(metadata, Mapping) or dict(metadata) != dict(expected_model_metadata):
        raise ValueError("confirmatory checkpoint model metadata differs from fold evidence")
    class_order = payload.get("class_order")
    if (
        not isinstance(class_order, torch.Tensor)
        or class_order.dtype != torch.int64
        or class_order.shape != (len(CLASS_ORDER),)
        or not np.array_equal(class_order.cpu().numpy(), np.asarray(CLASS_ORDER, dtype=np.int64))
    ):
        raise ValueError("confirmatory checkpoint class order is invalid")
    input_channels = int(expected_model_metadata["input_channels"])
    _validate_checkpoint_state_dict(
        payload.get("network_state_dict"),
        input_channels=input_channels,
        role="network_state_dict",
    )
    early = payload.get("early_stopping_state")
    if not isinstance(early, Mapping) or set(early) != {
        "best_epoch",
        "best_validation_loss",
        "epochs_without_improvement",
        "stopped_early",
        "best_network_state_dict",
    }:
        raise ValueError("confirmatory checkpoint early-stopping state is invalid")
    _validate_checkpoint_state_dict(
        early.get("best_network_state_dict"),
        input_channels=input_channels,
        role="best_network_state_dict",
    )
    completed_epochs = payload.get("completed_epochs")
    effective_batch_size = payload.get("effective_batch_size")
    if (
        type(completed_epochs) is not int
        or not 1 <= completed_epochs <= int(configuration["epochs"])
        or type(effective_batch_size) is not int
        or not int(configuration["minimum_batch_size"])
        <= effective_batch_size
        <= int(configuration["batch_size"])
    ):
        raise ValueError("confirmatory checkpoint training state is invalid")
    telemetry = payload.get("telemetry")
    if (
        not isinstance(telemetry, Mapping)
        or type(telemetry.get("successful_optimiser_steps")) is not int
        or type(telemetry.get("skipped_optimiser_steps")) is not int
    ):
        raise ValueError("confirmatory checkpoint optimiser-step telemetry is invalid")
    _validate_adamw_state(
        payload.get("optimiser_state_dict"),
        input_channels=input_channels,
        configuration=configuration,
        successful_optimiser_steps=int(telemetry["successful_optimiser_steps"]),
        skipped_optimiser_steps=int(telemetry["skipped_optimiser_steps"]),
    )
    _validate_grad_scaler_state(payload.get("scaler_state_dict"))
    _validate_rng_state(payload.get("rng_state"))
    _validate_history_and_telemetry(
        payload.get("history"),
        early,
        telemetry,
        completed_epochs=completed_epochs,
        effective_batch_size=effective_batch_size,
        configuration=configuration,
    )


def _batch_tensor(
    prepared: _PreparedImages,
    indices: NDArray[np.int64],
    input_size: int,
) -> torch.Tensor:
    rgb_rows = (
        prepared.rgb.gather_rows(indices, max_rows=len(indices))
        if isinstance(prepared.rgb, _IndexedRows)
        else prepared.rgb[indices]
    )
    rgb = torch.from_numpy(np.array(rgb_rows, order="C", copy=True)).permute(0, 3, 1, 2)
    rgb_float = rgb.float().div_(255.0)
    if rgb_float.shape[-2:] != (input_size, input_size):
        rgb_float = functional.interpolate(
            rgb_float,
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
    standard_deviation = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)
    rgb_float = (rgb_float - mean) / standard_deviation
    if prepared.target_masks is None:
        return rgb_float
    mask_rows = (
        prepared.target_masks.gather_rows(indices, max_rows=len(indices))
        if isinstance(prepared.target_masks, _IndexedRows)
        else prepared.target_masks[indices]
    )
    masks = torch.from_numpy(np.array(mask_rows[:, None, ...], order="C", copy=True)).float()
    if masks.shape[-2:] != (input_size, input_size):
        masks = functional.interpolate(masks, size=(input_size, input_size), mode="nearest")
    if not torch.all((masks == 0) | (masks == 1)):
        raise RuntimeError("nearest-neighbour target-mask preprocessing produced non-binary values")
    return torch.cat((rgb_float, masks), dim=1)


class _ConfirmatoryCNNBase:
    _cpu_test_only: bool
    _execution_mode: str
    _study_outcome_eligible: bool

    def __init__(self, config: ConfirmatoryCNNConfig) -> None:
        config.validate()
        self.config = config
        self.classes_: NDArray[np.int64] = np.asarray(CLASS_ORDER, dtype=np.int64)
        self.history_: tuple[HistoryRow, ...] = ()
        self.completed_epochs_: int = 0
        self.effective_batch_size_: int = config.batch_size
        self.best_epoch_: int | None = None
        self.best_validation_loss_: float | None = None
        self.epochs_without_improvement_: int = 0
        self.stopped_early_: bool = False
        self.telemetry_: dict[str, Any] = {}
        self.model_metadata_: dict[str, Any] = {}
        self.data_and_split_sha256_: dict[str, str] = {}
        self._network: nn.Module | None = None
        self._best_network_state: dict[str, torch.Tensor] | None = None
        self._backoff_events: list[BackoffEvent] = []
        self._prior_runtime_seconds = 0.0
        self._successful_samples_processed = 0
        self._optimiser_steps = 0
        self._successful_optimiser_steps = 0
        self._skipped_optimiser_steps = 0

    def _device(self) -> torch.device:
        if self._cpu_test_only:
            return torch.device("cpu")
        if not torch.cuda.is_available():
            raise RuntimeError("confirmatory real-study CNN training is CUDA-only")
        return torch.device("cuda")

    def _amp_dtype(self) -> torch.dtype:
        return torch.float16 if self.config.amp_dtype == "float16" else torch.bfloat16

    def _training_snapshot(
        self,
        optimiser: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
    ) -> dict[str, Any]:
        if self._network is None:
            raise RuntimeError("network is not initialised")
        return {
            "network_state_dict": _model_state_to_cpu(self._network),
            "optimiser_state_dict": _clone_to_cpu(optimiser.state_dict()),
            "scaler_state_dict": _clone_to_cpu(scaler.state_dict()) if scaler else None,
            "rng_state": _capture_rng_state(),
            "successful_optimiser_steps": self._successful_optimiser_steps,
            "skipped_optimiser_steps": self._skipped_optimiser_steps,
        }

    def _restore_training_snapshot(
        self,
        payload: dict[str, Any],
        optimiser: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
        device: torch.device,
    ) -> None:
        if self._network is None:
            raise RuntimeError("network is not initialised")
        self._network.load_state_dict(payload["network_state_dict"], strict=True)
        self._network.to(device)
        optimiser.load_state_dict(payload["optimiser_state_dict"])
        _optimiser_to_device(optimiser, device)
        if scaler is not None:
            scaler_state = payload.get("scaler_state_dict")
            if scaler_state is None:
                raise ValueError("transaction snapshot is missing GradScaler state")
            scaler.load_state_dict(scaler_state)
        _restore_rng_state(cast(dict[str, Any], payload["rng_state"]), device)
        self._successful_optimiser_steps = int(payload["successful_optimiser_steps"])
        self._skipped_optimiser_steps = int(payload["skipped_optimiser_steps"])

    def _train_window(
        self,
        prepared: _PreparedImages,
        labels: NDArray[np.int64],
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
            batch = _batch_tensor(prepared, micro_indices, self.config.input_size).to(
                device, non_blocking=True
            )
            targets = torch.from_numpy(labels[micro_indices]).to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=self._amp_dtype() if device.type == "cuda" else torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                per_sample_loss = criterion(self._network(batch), targets)
            micro_loss_sum = per_sample_loss.sum()
            backward_loss = micro_loss_sum / len(indices)
            if scaler is not None:
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()
            loss_sum += float(micro_loss_sum.detach().float().cpu())
        if scaler is not None:
            previous_scale = float(scaler.get_scale())
            scaler.step(optimiser)
            scaler.update()
            if float(scaler.get_scale()) < previous_scale:
                self._skipped_optimiser_steps += 1
            else:
                self._successful_optimiser_steps += 1
        else:
            optimiser.step()
            self._successful_optimiser_steps += 1
        return loss_sum

    def _reference_validation_loss(
        self,
        prepared: _PreparedImages,
        labels: NDArray[np.int64],
        criterion: nn.CrossEntropyLoss,
        device: torch.device,
        epoch: int,
    ) -> float:
        if self._network is None:
            raise RuntimeError("network is not initialised")
        network = self._network
        network.eval()
        order = np.arange(prepared.sample_count, dtype=np.int64)

        def evaluate(indices: NDArray[np.int64], _batch_size: int) -> float:
            batch = _batch_tensor(prepared, indices, self.config.input_size).to(
                device, non_blocking=True
            )
            targets = torch.from_numpy(labels[indices]).to(device, non_blocking=True)
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=device.type,
                    dtype=self._amp_dtype() if device.type == "cuda" else torch.bfloat16,
                    enabled=device.type == "cuda",
                ),
            ):
                losses = criterion(network(batch), targets)
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
            backoff_events=self._backoff_events,
        )
        self.effective_batch_size_ = result.final_batch_size
        return result.loss_sum / result.successful_sample_count

    def _telemetry(
        self,
        device: torch.device,
        current_runtime_seconds: float,
        *,
        failure: str | None,
    ) -> dict[str, Any]:
        peak_allocated: int | None = None
        peak_reserved: int | None = None
        if device.type == "cuda":
            with suppress(RuntimeError, ValueError):
                peak_allocated = int(torch.cuda.max_memory_allocated(device))
                peak_reserved = int(torch.cuda.max_memory_reserved(device))
        return {
            "schema_version": 1,
            "execution_mode": self._execution_mode,
            "study_outcome_eligible": self._study_outcome_eligible,
            "device": str(device),
            "amp_enabled": device.type == "cuda",
            "amp_dtype": self.config.amp_dtype if device.type == "cuda" else None,
            "grad_scaler_enabled": device.type == "cuda",
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "initial_batch_size": self.config.batch_size,
            "effective_batch_size": self.effective_batch_size_,
            "minimum_batch_size": self.config.minimum_batch_size,
            "batch_backoff_events": [dict(event) for event in self._backoff_events],
            "current_fit_runtime_seconds": current_runtime_seconds,
            "cumulative_runtime_seconds": self._prior_runtime_seconds + current_runtime_seconds,
            "cuda_peak_memory_allocated_bytes": peak_allocated,
            "cuda_peak_memory_reserved_bytes": peak_reserved,
            "completed_epochs": self.completed_epochs_,
            "requested_epochs": self.config.epochs,
            "successful_samples_processed": self._successful_samples_processed,
            "optimiser_steps": self._optimiser_steps,
            "successful_optimiser_steps": self._successful_optimiser_steps,
            "skipped_optimiser_steps": self._skipped_optimiser_steps,
            "early_stopping_source": "reference_validation_only",
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
        hashes: dict[str, str],
        device: torch.device,
        current_runtime_seconds: float,
    ) -> dict[str, Any]:
        if self._network is None:
            raise RuntimeError("cannot checkpoint an uninitialised network")
        configuration = asdict(self.config)
        return {
            "schema_version": 1,
            "model_kind": "confirmatory_resnet18_five_class",
            "execution_mode": self._execution_mode,
            "study_outcome_eligible": self._study_outcome_eligible,
            "configuration": configuration,
            "configuration_sha256": _json_sha256(configuration),
            "resume_contract_sha256": _json_sha256(_resume_contract_payload(self.config)),
            "data_and_split_sha256": dict(hashes),
            "model_metadata": dict(self.model_metadata_),
            "class_order": torch.tensor(CLASS_ORDER, dtype=torch.int64),
            "completed_epochs": self.completed_epochs_,
            "network_state_dict": _model_state_to_cpu(self._network),
            "optimiser_state_dict": _clone_to_cpu(optimiser.state_dict()),
            "scaler_state_dict": _clone_to_cpu(scaler.state_dict()) if scaler else None,
            "history": list(self.history_),
            "effective_batch_size": self.effective_batch_size_,
            "rng_state": _capture_rng_state(),
            "early_stopping_state": {
                "best_epoch": self.best_epoch_,
                "best_validation_loss": self.best_validation_loss_,
                "epochs_without_improvement": self.epochs_without_improvement_,
                "stopped_early": self.stopped_early_,
                "best_network_state_dict": self._best_network_state,
            },
            "telemetry": self._telemetry(device, current_runtime_seconds, failure=None),
        }

    def _restore_checkpoint(
        self,
        path: Path,
        optimiser: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
        hashes: dict[str, str],
        device: torch.device,
    ) -> None:
        if self._network is None:
            raise RuntimeError("network must be initialised before resume")
        payload = cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=True))
        if payload.get("schema_version") != 1 or payload.get("model_kind") != (
            "confirmatory_resnet18_five_class"
        ):
            raise ValueError("checkpoint is not a supported confirmatory CNN checkpoint")
        if (
            payload.get("execution_mode") != self._execution_mode
            or bool(payload.get("study_outcome_eligible")) != self._study_outcome_eligible
        ):
            raise ValueError("checkpoint execution/evidence mode differs from this classifier")
        stored_configuration = cast(dict[str, Any], payload["configuration"])
        if payload.get("configuration_sha256") != _json_sha256(stored_configuration):
            raise ValueError("checkpoint configuration hash is invalid")
        if payload.get("resume_contract_sha256") != _json_sha256(
            _resume_contract_payload(self.config)
        ):
            raise ValueError("checkpoint configuration differs from the resume contract")
        if cast(dict[str, str], payload.get("data_and_split_sha256")) != hashes:
            raise ValueError("checkpoint data or split fingerprint differs from supplied inputs")
        checkpoint_classes = cast(torch.Tensor, payload["class_order"]).cpu().numpy()
        if not np.array_equal(checkpoint_classes, np.asarray(CLASS_ORDER)):
            raise ValueError("checkpoint class order is not the fixed five-class order")
        stored_metadata = cast(dict[str, Any], payload["model_metadata"])
        for key in ("weight_identifier", "weight_sha256", "input_channels"):
            if stored_metadata.get(key) != self.model_metadata_.get(key):
                raise ValueError(f"checkpoint model metadata differs for {key}")
        completed = int(payload["completed_epochs"])
        if completed > self.config.epochs:
            raise ValueError("checkpoint was trained beyond configured epochs")
        self._network.load_state_dict(payload["network_state_dict"], strict=True)
        self._network.to(device)
        optimiser.load_state_dict(payload["optimiser_state_dict"])
        _optimiser_to_device(optimiser, device)
        scaler_state = payload.get("scaler_state_dict")
        if scaler is not None:
            if scaler_state is None:
                raise ValueError("CUDA checkpoint is missing GradScaler state")
            scaler.load_state_dict(scaler_state)
        elif scaler_state is not None:
            raise ValueError("CPU test-only checkpoint unexpectedly contains GradScaler state")
        self.completed_epochs_ = completed
        self.history_ = tuple(cast(HistoryRow, dict(row)) for row in payload["history"])
        self.effective_batch_size_ = int(payload["effective_batch_size"])
        early = cast(dict[str, Any], payload["early_stopping_state"])
        self.best_epoch_ = int(early["best_epoch"]) if early.get("best_epoch") else None
        self.best_validation_loss_ = (
            float(early["best_validation_loss"])
            if early.get("best_validation_loss") is not None
            else None
        )
        self.epochs_without_improvement_ = int(early["epochs_without_improvement"])
        self.stopped_early_ = bool(early["stopped_early"])
        best_state = early.get("best_network_state_dict")
        self._best_network_state = (
            cast(dict[str, torch.Tensor], best_state) if best_state is not None else None
        )
        checkpoint_telemetry = cast(dict[str, Any], payload.get("telemetry", {}))
        self._prior_runtime_seconds = float(
            checkpoint_telemetry.get("cumulative_runtime_seconds", 0.0)
        )
        self._backoff_events = [
            cast(BackoffEvent, dict(event))
            for event in checkpoint_telemetry.get("batch_backoff_events", [])
        ]
        self._successful_samples_processed = int(
            checkpoint_telemetry.get("successful_samples_processed", 0)
        )
        self._optimiser_steps = int(checkpoint_telemetry.get("optimiser_steps", 0))
        self._successful_optimiser_steps = int(
            checkpoint_telemetry.get("successful_optimiser_steps", self._optimiser_steps)
        )
        self._skipped_optimiser_steps = int(checkpoint_telemetry.get("skipped_optimiser_steps", 0))
        _restore_rng_state(cast(dict[str, Any], payload["rng_state"]), device)

    def fit(
        self,
        training_images: ImageArray,
        observed_training_labels: NDArray[np.generic] | Sequence[int],
        *,
        training_sample_ids: Sequence[str] | NDArray[np.str_],
        training_group_ids: Sequence[str] | NDArray[np.str_],
        reference_validation_images: ImageArray,
        reference_validation_labels: NDArray[np.generic] | Sequence[int],
        reference_validation_sample_ids: Sequence[str] | NDArray[np.str_],
        reference_validation_group_ids: Sequence[str] | NDArray[np.str_],
        reference_validation_role: ValidationRole,
        training_target_masks: ImageArray | None = None,
        reference_validation_target_masks: ImageArray | None = None,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
    ) -> Self:
        """Fit only on development and explicitly identified reference-validation data.

        There is intentionally no final-test argument.  The final reference fold
        cannot influence fitting, early stopping, batch policy, or checkpoint state.
        """

        fit_started = time.perf_counter()
        if reference_validation_role != "reference_validation":
            raise ValueError("early stopping data must be explicitly marked reference_validation")
        training = _prepare_images(
            training_images,
            training_target_masks,
            self.config.input_variant,
            name="training_images",
        )
        reference_validation = _prepare_images(
            reference_validation_images,
            reference_validation_target_masks,
            self.config.input_variant,
            name="reference_validation_images",
        )
        training_labels = _labels(
            observed_training_labels,
            training.sample_count,
            name="observed_training_labels",
        )
        if set(training_labels.tolist()) != set(CLASS_ORDER):
            raise ValueError("observed training labels must contain every fixed class")
        validation_labels = _labels(
            reference_validation_labels,
            reference_validation.sample_count,
            name="reference_validation_labels",
        )
        train_sample_ids = _identifiers(
            training_sample_ids,
            training.sample_count,
            name="training_sample_ids",
        )
        validation_sample_ids = _identifiers(
            reference_validation_sample_ids,
            reference_validation.sample_count,
            name="reference_validation_sample_ids",
        )
        train_group_ids = _identifiers(
            training_group_ids,
            training.sample_count,
            name="training_group_ids",
        )
        validation_group_ids = _identifiers(
            reference_validation_group_ids,
            reference_validation.sample_count,
            name="reference_validation_group_ids",
        )
        sample_overlap = set(train_sample_ids.tolist()) & set(validation_sample_ids.tolist())
        if sample_overlap:
            raise ValueError("training/reference-validation sample IDs overlap")
        group_overlap = set(train_group_ids.tolist()) & set(validation_group_ids.tolist())
        if group_overlap:
            raise ValueError("training/reference-validation group leakage detected")

        path = Path(checkpoint_path) if checkpoint_path is not None else None
        if not self._cpu_test_only and path is None:
            raise ValueError("real-study CNN training requires an atomic checkpoint_path")
        if resume and path is None:
            raise ValueError("resume=True requires checkpoint_path")
        if resume and path is not None and not path.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {path}")

        hashes = {
            "training_data_sha256": _dataset_sha256(training, training_labels),
            "reference_validation_data_sha256": _dataset_sha256(
                reference_validation, validation_labels
            ),
            "training_split_sha256": _split_sha256(train_sample_ids, train_group_ids),
            "reference_validation_split_sha256": _split_sha256(
                validation_sample_ids, validation_group_ids
            ),
        }
        self.data_and_split_sha256_ = dict(hashes)
        device = self._device()
        _seed_everything(self.config.seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        self._network, self.model_metadata_ = _build_resnet18(
            self.config, cpu_test_only=self._cpu_test_only
        )
        self._network.to(device)
        optimiser = torch.optim.AdamW(
            self._network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True) if device.type == "cuda" else None
        if resume and path is not None:
            self._restore_checkpoint(path, optimiser, scaler, hashes, device)
        else:
            self.completed_epochs_ = 0
            self.history_ = ()
            self.effective_batch_size_ = self.config.batch_size
            self.best_epoch_ = None
            self.best_validation_loss_ = None
            self.epochs_without_improvement_ = 0
            self.stopped_early_ = False
            self._best_network_state = None
            self._backoff_events = []
            self._prior_runtime_seconds = 0.0
            self._successful_samples_processed = 0
            self._optimiser_steps = 0
            self._successful_optimiser_steps = 0
            self._skipped_optimiser_steps = 0

        weights: torch.Tensor | None = None
        if self.config.class_weight_balanced:
            counts = np.bincount(training_labels, minlength=len(CLASS_ORDER)).astype(np.float64)
            weights = torch.from_numpy(
                (len(training_labels) / (len(CLASS_ORDER) * counts)).astype(np.float32)
            ).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, reduction="none")
        failure: str | None = None
        try:
            history = list(self.history_)
            if not self.stopped_early_:
                for epoch_index in range(self.completed_epochs_, self.config.epochs):
                    _seed_everything(self.config.seed + epoch_index)
                    order = (
                        np.random.default_rng(self.config.seed + epoch_index)
                        .permutation(training.sample_count)
                        .astype(np.int64)
                    )
                    self._network.train()
                    successful_steps_before_epoch = self._successful_optimiser_steps
                    skipped_steps_before_epoch = self._skipped_optimiser_steps

                    def execute(indices: NDArray[np.int64], batch_size: int) -> float:
                        return self._train_window(
                            training,
                            training_labels,
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
                        epoch=epoch_index + 1,
                        phase="training",
                        initial_batch_size=self.effective_batch_size_,
                        minimum_batch_size=self.config.minimum_batch_size,
                        accumulation_steps=self.config.gradient_accumulation_steps,
                        cuda_oom_recovery=device.type == "cuda",
                        execute_window=execute,
                        snapshot_state=lambda: self._training_snapshot(optimiser, scaler),
                        restore_state=lambda snapshot: self._restore_training_snapshot(
                            cast(dict[str, Any], snapshot), optimiser, scaler, device
                        ),
                        clear_after_oom=clear_cuda_oom,
                        backoff_events=self._backoff_events,
                    )
                    self.effective_batch_size_ = execution.final_batch_size
                    self._successful_samples_processed += execution.successful_sample_count
                    self._optimiser_steps += execution.optimiser_steps
                    validation_loss = self._reference_validation_loss(
                        reference_validation,
                        validation_labels,
                        criterion,
                        device,
                        epoch_index + 1,
                    )
                    self.completed_epochs_ = epoch_index + 1
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
                            "training_loss": execution.loss_sum / training.sample_count,
                            "reference_validation_loss": validation_loss,
                            "effective_batch_size": self.effective_batch_size_,
                            "optimiser_steps": execution.optimiser_steps,
                            "successful_optimiser_steps": (
                                self._successful_optimiser_steps - successful_steps_before_epoch
                            ),
                            "skipped_optimiser_steps": (
                                self._skipped_optimiser_steps - skipped_steps_before_epoch
                            ),
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
                                hashes,
                                device,
                                time.perf_counter() - fit_started,
                            ),
                        )
                    if self.stopped_early_:
                        break
            if self._best_network_state is None:
                raise RuntimeError("reference-validation early stopping has no best state")
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
            self.telemetry_ = self._telemetry(
                device,
                time.perf_counter() - fit_started,
                failure=failure,
            )
        return self

    def predict_proba(
        self,
        images: ImageArray,
        *,
        target_masks: ImageArray | None = None,
        batch_size: int | None = None,
    ) -> NDArray[np.float64]:
        """Return probabilities in the immutable ``(0, 1, 2, 3, 4)`` order."""

        if self._network is None:
            raise RuntimeError("classifier is not fitted")
        network = self._network
        prepared = _prepare_images(
            images,
            target_masks,
            self.config.input_variant,
            name="prediction_images",
        )
        size = batch_size or self.effective_batch_size_
        if size <= 0:
            raise ValueError("batch_size must be positive")
        device = next(network.parameters()).device
        outputs: list[NDArray[np.float64]] = []
        order = np.arange(prepared.sample_count, dtype=np.int64)
        network.eval()

        def infer(indices: NDArray[np.int64], _batch_size: int) -> float:
            batch = _batch_tensor(prepared, indices, self.config.input_size).to(
                device, non_blocking=True
            )
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=device.type,
                    dtype=self._amp_dtype() if device.type == "cuda" else torch.bfloat16,
                    enabled=device.type == "cuda",
                ),
            ):
                probabilities = torch.softmax(network(batch), dim=1)
            outputs.append(probabilities.float().cpu().numpy().astype(np.float64))
            return 0.0

        execution = _execute_epoch_with_backoff(
            order,
            epoch=self.completed_epochs_,
            phase="prediction",
            initial_batch_size=min(size, prepared.sample_count),
            minimum_batch_size=min(self.config.minimum_batch_size, prepared.sample_count),
            accumulation_steps=1,
            cuda_oom_recovery=device.type == "cuda",
            execute_window=infer,
            snapshot_state=lambda: None,
            restore_state=lambda _snapshot: None,
            clear_after_oom=torch.cuda.empty_cache,
            backoff_events=self._backoff_events,
        )
        self.effective_batch_size_ = execution.final_batch_size
        result = np.concatenate(outputs, axis=0)
        if result.shape != (prepared.sample_count, len(CLASS_ORDER)):
            raise RuntimeError("unexpected confirmatory CNN probability shape")
        if not np.isfinite(result).all():
            raise RuntimeError("confirmatory CNN produced non-finite probabilities")
        return result


class ConfirmatoryResNet18Classifier(_ConfirmatoryCNNBase):
    """CUDA/AMP-only study classifier; checkpoints are eligible after outer gates."""

    _cpu_test_only = False
    _execution_mode = "real_study_cuda"
    _study_outcome_eligible = True


class ConfirmatoryCNNCPUTestOnlyAdapter(_ConfirmatoryCNNBase):
    """CPU-only structural-test adapter whose output is never study evidence."""

    _cpu_test_only = True
    _execution_mode = "cpu_test_only_non_evidence"
    _study_outcome_eligible = False


__all__ = [
    "CLASS_ORDER",
    "CPU_TEST_ONLY_WEIGHT_IDENTIFIER",
    "OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER",
    "ConfirmatoryCNNCPUTestOnlyAdapter",
    "ConfirmatoryCNNConfig",
    "ConfirmatoryResNet18Classifier",
    "confirmatory_cnn_data_and_split_sha256",
    "validate_confirmatory_checkpoint_artifact",
]
