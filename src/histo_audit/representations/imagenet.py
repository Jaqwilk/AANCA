"""Guarded official torchvision ResNet-18 frozen embedding extraction.

The default is intentionally offline: a missing official weight checkpoint is an
explicit blocker.  Network access occurs only when ``allow_weight_download=True``
is supplied by the caller.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18  # type: ignore[import-untyped]

from histo_audit.data.targets import highlight_target
from histo_audit.utils.run_tracking import sha256_file

from .cache_provenance import (
    atomic_save_npz_with_sidecar,
    build_frozen_cache_metadata,
    canonical_sha256,
    ordered_sample_ids_sha256,
    verify_frozen_cache_sidecar,
)

InputVariant = Literal["rgb", "target_highlighted_rgb"]
OutputDType = Literal["float32", "float16"]
EmbeddingArray = NDArray[np.float32] | NDArray[np.float16]


class PretrainedWeightsUnavailableError(RuntimeError):
    """Raised when official weights are absent and download was not authorised."""


@dataclass(frozen=True, slots=True)
class ResNet18EmbeddingConfig:
    """Frozen encoder and conservative extraction settings."""

    weight_identifier: str = "IMAGENET1K_V1"
    input_variant: InputVariant = "rgb"
    context_brightness: float = 0.45
    device: str = "auto"
    batch_size: int = 16
    minimum_batch_size: int = 1
    use_amp: bool = True
    output_dtype: OutputDType = "float32"
    allow_weight_download: bool = False
    download_progress: bool = False

    def validate(self) -> None:
        """Reject ambiguous or unsafe extraction settings."""

        if self.input_variant not in ("rgb", "target_highlighted_rgb"):
            raise ValueError(f"unsupported input_variant: {self.input_variant!r}")
        if not 0.0 <= self.context_brightness <= 1.0:
            raise ValueError("context_brightness must lie in [0, 1]")
        if self.batch_size <= 0 or self.minimum_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.minimum_batch_size > self.batch_size:
            raise ValueError("minimum_batch_size cannot exceed batch_size")
        if self.output_dtype not in ("float32", "float16"):
            raise ValueError(f"unsupported output dtype: {self.output_dtype!r}")


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """In-memory embeddings and complete extraction provenance."""

    embeddings: EmbeddingArray
    sample_ids: NDArray[np.str_]
    metadata: dict[str, Any]
    cache_path: Path | None = None
    metadata_path: Path | None = None

    def validate(self) -> None:
        """Validate sample alignment and finite 512-dimensional output."""

        if self.embeddings.ndim != 2 or self.embeddings.shape[1] != 512:
            raise ValueError("ResNet-18 embeddings must have shape (n_samples, 512)")
        if self.sample_ids.shape != (self.embeddings.shape[0],):
            raise ValueError("sample_ids do not align with embeddings")
        if len(set(self.sample_ids.tolist())) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique")
        if not np.isfinite(self.embeddings).all():
            raise ValueError("embeddings contain non-finite values")


@dataclass(frozen=True, slots=True)
class ResNet18SmokeResult:
    """Honest result from a tiny frozen-encoder execution attempt."""

    status: Literal["passed", "blocked", "failed"]
    blocker: str | None
    variants: dict[str, tuple[int, int]]
    cache_paths: tuple[str, ...]
    device: str | None
    weight_path: str | None
    weight_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible smoke evidence."""

        return asdict(self)


def _official_weights(identifier: str) -> ResNet18_Weights:
    normalised = identifier.strip()
    aliases = {
        "DEFAULT": ResNet18_Weights.DEFAULT,
        "IMAGENET1K_V1": ResNet18_Weights.IMAGENET1K_V1,
        "ResNet18_Weights.DEFAULT": ResNet18_Weights.DEFAULT,
        "ResNet18_Weights.IMAGENET1K_V1": ResNet18_Weights.IMAGENET1K_V1,
    }
    try:
        return aliases[normalised]
    except KeyError as exc:
        raise ValueError(
            "weight_identifier must name an official torchvision ResNet18_Weights member"
        ) from exc


def official_resnet18_weight_cache_path(
    weight_identifier: str = "IMAGENET1K_V1",
) -> Path:
    """Return the torch-hub path used by the selected official checkpoint."""

    weights = _official_weights(weight_identifier)
    filename = Path(urlparse(weights.url).path).name
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def _load_official_state_dict(
    weights: ResNet18_Weights,
    *,
    allow_download: bool,
    progress: bool,
) -> tuple[dict[str, torch.Tensor], Path]:
    checkpoint = official_resnet18_weight_cache_path(str(weights))
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    elif not allow_download:
        raise PretrainedWeightsUnavailableError(
            "official torchvision ResNet-18 weights are not present at "
            f"{checkpoint}; rerun only with allow_weight_download=True after network and "
            "licence/provenance approval"
        )
    else:
        state = torch.hub.load_state_dict_from_url(
            weights.url,
            model_dir=str(checkpoint.parent),
            map_location="cpu",
            progress=progress,
            check_hash=True,
            weights_only=True,
        )
    if not checkpoint.is_file():
        raise RuntimeError(f"official weight retrieval did not create expected file: {checkpoint}")
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise RuntimeError("official ResNet-18 checkpoint has an unexpected state-dict format")
    return cast(dict[str, torch.Tensor], state), checkpoint


def _frozen_encoder(
    config: ResNet18EmbeddingConfig,
) -> tuple[nn.Module, ResNet18_Weights, Path]:
    weights = _official_weights(config.weight_identifier)
    state, weight_path = _load_official_state_dict(
        weights,
        allow_download=config.allow_weight_download,
        progress=config.download_progress,
    )
    model = resnet18(weights=None)
    model.load_state_dict(state, strict=True)
    model.fc = nn.Identity()
    model.eval()
    model.requires_grad_(False)
    return model, weights, weight_path


def _resolve_device(requested: str) -> torch.device:
    normalised = requested.strip().lower()
    if normalised == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalised)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("ResNet-18 extraction supports CPU or CUDA devices")
    return device


def _uint8_rgb(images: NDArray[np.generic]) -> NDArray[np.uint8]:
    array = np.asarray(images)
    if array.ndim != 4 or array.shape[0] == 0 or array.shape[-1] != 3:
        raise ValueError("images must have non-empty shape (n, height, width, 3)")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("images must contain finite numeric RGB values")
    converted = array.astype(np.float64)
    if float(converted.min()) < 0.0:
        raise ValueError("RGB values must be non-negative")
    maximum = float(converted.max())
    if np.issubdtype(array.dtype, np.floating) and maximum <= 1.0:
        converted *= 255.0
    elif maximum > 255.0:
        raise ValueError("RGB values must lie in [0, 255] (or [0, 1] for floats)")
    return np.clip(np.rint(converted), 0, 255).astype(np.uint8)


def _prepare_variant(
    images: NDArray[np.uint8],
    target_masks: NDArray[np.generic] | None,
    config: ResNet18EmbeddingConfig,
) -> NDArray[np.uint8]:
    if config.input_variant == "rgb":
        if target_masks is not None and np.asarray(target_masks).shape != images.shape[:3]:
            raise ValueError("target_masks, when supplied, must align with images")
        return images
    if target_masks is None:
        raise ValueError("target_highlighted_rgb requires exact target_masks")
    masks = np.asarray(target_masks, dtype=bool)
    if masks.shape != images.shape[:3]:
        raise ValueError("target_masks must align with images")
    if not np.all(masks.reshape(len(masks), -1).any(axis=1)):
        raise ValueError("each highlighted input requires a non-empty target mask")
    return np.stack(
        [
            highlight_target(image, mask, context_brightness=config.context_brightness)
            for image, mask in zip(images, masks, strict=True)
        ]
    )


def _array_sha256(array: NDArray[np.generic]) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _preprocess_metadata(weights: ResNet18_Weights) -> dict[str, Any]:
    transform = weights.transforms(antialias=True)
    return {
        "api": "torchvision weight_enum.transforms(antialias=True)",
        "resize_size": list(transform.resize_size),
        "crop_size": list(transform.crop_size),
        "interpolation": str(transform.interpolation),
        "mean": [float(value) for value in transform.mean],
        "std": [float(value) for value in transform.std],
        "antialias": bool(transform.antialias),
    }


def _is_oom(error: RuntimeError) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def _metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(f"{cache_path.suffix}.metadata.json")


def _canonical_input_variant(value: object) -> str:
    variant = str(value or "rgb")
    if variant in {"rgb", "context_rgb"}:
        return "context_rgb"
    if variant == "target_highlighted_rgb":
        return variant
    raise ValueError(f"unsupported embedding provenance input_variant: {variant!r}")


_CROP_CACHE_BINDING_SHA256_FIELDS = (
    "crop_cache_file_sha256",
    "crop_cache_sidecar_file_sha256",
    "crop_cache_content_sha256",
    "crop_manifest_sha256",
    "raw_inventory_sha256",
    "sample_order_sha256",
    "target_mask_projection_semantic_sha256",
    "input_array_sha256",
)


def _normalise_source_crop_cache_binding(
    value: object,
    *,
    expected_input_variant: str,
) -> dict[str, Any] | None:
    """Validate the exact PanNuke crop artifact consumed by an embedding."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("source_crop_cache_binding must be a mapping")
    binding = {str(key): item for key, item in value.items()}
    expected_keys = {
        "schema_version",
        "binding_type",
        "crop_cache_file_sha256",
        "crop_cache_sidecar_file_sha256",
        "crop_cache_content_sha256",
        "crop_manifest_sha256",
        "raw_inventory_sha256",
        "sample_order_sha256",
        "target_mask_projection_semantic_sha256",
        "input_variant",
        "input_array_key",
        "input_array_sha256",
    }
    if set(binding) != expected_keys:
        raise ValueError("source_crop_cache_binding schema is invalid")
    canonical_variant = _canonical_input_variant(expected_input_variant)
    expected_array_key = (
        "context_rgb" if canonical_variant == "context_rgb" else "target_highlighted_rgb"
    )
    if (
        binding["schema_version"] != 1
        or binding["binding_type"] != "pannuke_component_covering_crop_cache_v1"
        or binding["input_variant"] != canonical_variant
        or binding["input_array_key"] != expected_array_key
    ):
        raise ValueError("source crop-cache binding semantics are invalid")
    for field in _CROP_CACHE_BINDING_SHA256_FIELDS:
        digest = str(binding[field]).casefold()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"source crop-cache binding {field} must be a SHA-256")
        binding[field] = digest
    return binding


def _runtime_embedding_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Expose the historical RGB alias to existing loaders without changing the sidecar."""

    runtime = dict(metadata)
    runtime["contract_input_variant"] = str(metadata["input_variant"])
    runtime["input_variant"] = str(
        metadata.get(
            "legacy_input_variant",
            "rgb" if metadata["input_variant"] == "context_rgb" else metadata["input_variant"],
        )
    )
    return runtime


def save_embedding_cache(
    cache_path: str | Path,
    embeddings: EmbeddingArray,
    sample_ids: NDArray[np.str_],
    metadata: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    """Atomically save embeddings plus a final-contract provenance sidecar."""

    destination = Path(cache_path).resolve()
    if destination.suffix.lower() != ".npz":
        raise ValueError("embedding cache path must end in .npz")
    values = np.asarray(embeddings)
    identifiers = np.asarray(sample_ids, dtype=np.str_)
    if values.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise ValueError("embedding cache dtype must be float16 or float32")
    candidate = EmbeddingResult(
        embeddings=cast(EmbeddingArray, values),
        sample_ids=identifiers,
        metadata={},
    )
    candidate.validate()
    embedded_metadata = {
        "schema_version": 1,
        "embeddings_sha256": _array_sha256(values),
        "sample_ids_sha256": _array_sha256(identifiers),
        "sample_order_sha256": ordered_sample_ids_sha256(identifiers),
    }
    embedded_json = np.asarray(
        json.dumps(
            embedded_metadata,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        dtype=np.str_,
    )

    supplied = dict(metadata)
    internal_variant = str(
        supplied.get("legacy_input_variant", supplied.get("input_variant", "rgb"))
    )
    contract_variant = _canonical_input_variant(internal_variant)
    source_crop_cache_binding = _normalise_source_crop_cache_binding(
        supplied.get("source_crop_cache_binding"),
        expected_input_variant=contract_variant,
    )
    representation_id = str(
        supplied.get(
            "representation_id",
            "imagenet_resnet18_context_embeddings"
            if contract_variant == "context_rgb"
            else "imagenet_target_highlighted_embeddings",
        )
    )
    embedding_sha = embedded_metadata["embeddings_sha256"]
    sample_order_sha = embedded_metadata["sample_order_sha256"]
    explicit_manifest = supplied.get(
        "manifest_sha256",
        supplied.get("crop_manifest_sha256", supplied.get("dataset_manifest_sha256")),
    )
    explicit_inventory = supplied.get("raw_inventory_sha256")
    manifest_sha = str(
        explicit_manifest
        or canonical_sha256(
            {
                "non_stage_fixture": True,
                "sample_order_sha256": sample_order_sha,
                "source": "in_memory_embedding_matrix",
            }
        )
    )
    inventory_sha = str(
        explicit_inventory
        or canonical_sha256(
            {
                "non_stage_fixture": True,
                "embedding_sha256": embedding_sha,
                "source": "in_memory_embedding_matrix",
            }
        )
    )
    supplied_weight_sha = supplied.get("weights_sha256", supplied.get("weight_sha256"))
    if supplied_weight_sha is None:
        weight_identifier = "non_stage_fixture:verified_weight_evidence_unavailable"
        weights_sha = canonical_sha256(
            {
                "non_stage_fixture": True,
                "verified_weight_evidence": False,
                "weight_identifier": str(supplied.get("weight_identifier", "unavailable")),
            }
        )
    else:
        weight_identifier = str(
            supplied.get("weight_identifier", "externally_supplied_verified_weight_sha256")
        )
        weights_sha = str(supplied_weight_sha)
    encoder_identifier = str(
        supplied.get("encoder_identifier", supplied.get("encoder_name", "frozen_embedding_matrix"))
    )
    preprocessing = supplied.get("preprocessing")
    if not isinstance(preprocessing, dict):
        preprocessing = {"api": "not_recorded_for_non_stage_embedding_fixture"}
    encoder_metadata = supplied.get("encoder_metadata")
    if not isinstance(encoder_metadata, dict):
        encoder_metadata = {
            "encoder_identifier": encoder_identifier,
            "embedding_sha256": embedding_sha,
            "frozen": bool(supplied.get("encoder_frozen", True)),
            "output_dimension": int(values.shape[1]),
        }
    else:
        encoder_metadata = dict(encoder_metadata)
    if source_crop_cache_binding is not None:
        source_binding_sha256 = canonical_sha256(source_crop_cache_binding)
        supplied["source_crop_cache_binding"] = source_crop_cache_binding
        supplied["source_crop_cache_binding_sha256"] = source_binding_sha256
        encoder_metadata["source_crop_cache_binding"] = source_crop_cache_binding
        encoder_metadata["source_crop_cache_binding_sha256"] = source_binding_sha256
    encoder_implementation = supplied.get("encoder_implementation")
    if not isinstance(encoder_implementation, dict):
        encoder_implementation = {
            "module": "histo_audit.representations.imagenet",
            "persistence_entrypoint": "save_embedding_cache",
            "source_file_sha256": sha256_file(Path(__file__)),
        }
    cache_recipe = supplied.get("cache_recipe")
    if not isinstance(cache_recipe, dict):
        cache_recipe = {
            "identifier": "frozen_embedding_npz_v2",
            "array_keys": ["embeddings", "metadata_json", "sample_ids"],
            "embedding_dtype": str(values.dtype),
            "embedding_dimension": int(values.shape[1]),
            "pickle_allowed": False,
        }
    else:
        cache_recipe = dict(cache_recipe)
    if source_crop_cache_binding is not None:
        cache_recipe["source_crop_cache_binding_sha256"] = canonical_sha256(
            source_crop_cache_binding
        )
    versions = supplied.get("package_versions", supplied.get("versions", {}))
    package_versions = (
        {str(key): str(value) for key, value in versions.items()}
        if isinstance(versions, dict)
        else {}
    )
    package_versions.setdefault("python", platform.python_version())
    package_versions.setdefault("numpy", _package_version("numpy"))
    package_versions.setdefault("torch", str(torch.__version__))
    package_versions.setdefault("torchvision", _package_version("torchvision"))
    stage_bindings_complete = (
        explicit_manifest is not None
        and explicit_inventory is not None
        and supplied_weight_sha is not None
    )
    provenance_scope = str(
        supplied.get(
            "provenance_scope",
            "stage_eligible" if stage_bindings_complete else "non_stage_fixture",
        )
    )
    if provenance_scope == "stage_eligible" and not stage_bindings_complete:
        raise ValueError(
            "stage-eligible embedding persistence requires manifest, raw inventory, and "
            "verified weight SHA-256 bindings"
        )
    supplied["legacy_input_variant"] = (
        "rgb" if contract_variant == "context_rgb" else contract_variant
    )
    supplied["embeddings_sha256"] = embedding_sha
    supplied["sample_ids_sha256"] = embedded_metadata["sample_ids_sha256"]
    frozen_metadata = build_frozen_cache_metadata(
        base_metadata=supplied,
        sample_ids=identifiers,
        manifest_sha256=manifest_sha,
        raw_inventory_sha256=inventory_sha,
        representation_id=representation_id,
        input_variant=contract_variant,
        encoder_identifier=encoder_identifier,
        encoder_metadata=encoder_metadata,
        encoder_implementation=encoder_implementation,
        weight_identifier=weight_identifier,
        weights_sha256=weights_sha,
        preprocessing_identifier=str(
            supplied.get(
                "preprocessing_identifier",
                "torchvision_resnet18_imagenet1k_v1_official"
                if encoder_identifier == "resnet18_imagenet1k_v1"
                else "non_stage_fixture_preprocessing_unverified",
            )
        ),
        preprocessing=preprocessing,
        cache_recipe=cache_recipe,
        dtype=str(values.dtype),
        feature_dimension=int(values.shape[1]),
        package_versions=package_versions,
        matrix_key="embeddings",
        provenance_scope=provenance_scope,
    )
    cache, sidecar, complete = atomic_save_npz_with_sidecar(
        destination,
        arrays={
            "embeddings": values,
            "sample_ids": identifiers,
            "metadata_json": embedded_json,
        },
        metadata=frozen_metadata,
    )
    return cache, sidecar, _runtime_embedding_metadata(complete)


def load_embedding_cache(cache_path: str | Path) -> EmbeddingResult:
    """Load and checksum-validate an embedding cache without pickle support."""

    verification = verify_frozen_cache_sidecar(cache_path)
    source = verification.cache_path
    sidecar = verification.sidecar_path
    metadata = verification.metadata
    with np.load(source, allow_pickle=False) as payload:
        if set(payload.files) != {"embeddings", "sample_ids", "metadata_json"}:
            raise ValueError("embedding cache contains unexpected arrays")
        embeddings = payload["embeddings"]
        sample_ids = payload["sample_ids"].astype(str)
        embedded = json.loads(str(payload["metadata_json"].item()))
    if embedded.get("embeddings_sha256") != _array_sha256(embeddings):
        raise ValueError("embedding matrix checksum does not match embedded metadata")
    if embedded.get("sample_ids_sha256") != _array_sha256(sample_ids):
        raise ValueError("sample ID checksum does not match embedded metadata")
    if embedded.get("sample_order_sha256") != ordered_sample_ids_sha256(sample_ids):
        raise ValueError("canonical sample order differs from embedded metadata")
    crop_binding = _normalise_source_crop_cache_binding(
        metadata.get("source_crop_cache_binding"),
        expected_input_variant=str(metadata.get("input_variant", "")),
    )
    if crop_binding is not None:
        binding_sha256 = canonical_sha256(crop_binding)
        encoder_metadata = metadata.get("encoder_metadata")
        cache_recipe = metadata.get("cache_recipe")
        if (
            metadata.get("source_crop_cache_binding_sha256") != binding_sha256
            or not isinstance(encoder_metadata, Mapping)
            or encoder_metadata.get("source_crop_cache_binding") != crop_binding
            or encoder_metadata.get("source_crop_cache_binding_sha256") != binding_sha256
            or not isinstance(cache_recipe, Mapping)
            or cache_recipe.get("source_crop_cache_binding_sha256") != binding_sha256
        ):
            raise ValueError("embedding source crop-cache binding is inconsistent")
    if (
        sha256_file(source) != verification.cache_file_sha256
        or sha256_file(sidecar) != verification.sidecar_file_sha256
    ):
        raise ValueError("embedding cache or sidecar changed during loading")
    if embeddings.dtype == np.float16:
        typed_embeddings: EmbeddingArray = embeddings.astype(np.float16, copy=False)
    elif embeddings.dtype == np.float32:
        typed_embeddings = embeddings.astype(np.float32, copy=False)
    else:
        raise ValueError(f"unsupported cached embedding dtype: {embeddings.dtype}")
    result = EmbeddingResult(
        embeddings=typed_embeddings,
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
        metadata=_runtime_embedding_metadata(metadata),
        cache_path=source,
        metadata_path=sidecar,
    )
    result.validate()
    return result


def extract_resnet18_embeddings(
    images: NDArray[np.generic],
    sample_ids: list[str] | tuple[str, ...] | NDArray[np.str_],
    *,
    target_masks: NDArray[np.generic] | None = None,
    config: ResNet18EmbeddingConfig | None = None,
    cache_path: str | Path | None = None,
    manifest_sha256: str | None = None,
    raw_inventory_sha256: str | None = None,
    representation_id: str | None = None,
    analysis_eligibility: Mapping[str, Any] | None = None,
    source_crop_cache_binding: Mapping[str, Any] | None = None,
) -> EmbeddingResult:
    """Extract frozen official ImageNet ResNet-18 embeddings in inference mode.

    Batching uses the calling process (zero DataLoader workers), with a
    conservative Windows cap and automatic CUDA OOM halving.  The final feature
    matrix is always moved to CPU before optional cache persistence.
    """

    settings = config or ResNet18EmbeddingConfig()
    settings.validate()
    rgb = _uint8_rgb(images)
    identifiers = np.asarray(sample_ids, dtype=np.str_)
    if identifiers.shape != (len(rgb),):
        raise ValueError("sample_ids must be one-dimensional and align with images")
    if any(not identifier for identifier in identifiers.tolist()):
        raise ValueError("sample_ids cannot be empty")
    if len(set(identifiers.tolist())) != len(identifiers):
        raise ValueError("sample_ids must be unique")
    prepared = _prepare_variant(rgb, target_masks, settings)
    model, weights, weight_path = _frozen_encoder(settings)
    device = _resolve_device(settings.device)
    model.to(device)
    preprocess = weights.transforms(antialias=True)
    requested_batch_size = settings.batch_size
    windows_cap = 32 if platform.system() == "Windows" else requested_batch_size
    current_batch_size = min(requested_batch_size, windows_cap, len(prepared))
    minimum_batch_size = min(settings.minimum_batch_size, current_batch_size)
    use_amp = settings.use_amp and device.type == "cuda"
    chunks: list[NDArray[np.float32]] = []
    backoffs: list[dict[str, int]] = []
    started = time.perf_counter()
    cursor = 0
    with torch.inference_mode():
        while cursor < len(prepared):
            take = min(current_batch_size, len(prepared) - cursor)
            try:
                batch = torch.from_numpy(
                    np.ascontiguousarray(prepared[cursor : cursor + take].transpose(0, 3, 1, 2))
                )
                batch = preprocess(batch).to(device, non_blocking=device.type == "cuda")
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                    enabled=use_amp,
                ):
                    output = model(batch)
                if output.shape != (take, 512):
                    raise RuntimeError(
                        f"unexpected frozen ResNet-18 output shape: {tuple(output.shape)}"
                    )
                chunks.append(output.float().cpu().numpy().astype(np.float32, copy=False))
                cursor += take
            except RuntimeError as exc:
                if not _is_oom(exc) or take <= minimum_batch_size:
                    raise
                reduced = max(minimum_batch_size, take // 2)
                if reduced >= take:
                    raise
                backoffs.append({"sample_offset": cursor, "from": take, "to": reduced})
                current_batch_size = reduced
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    float32_embeddings = np.concatenate(chunks, axis=0)
    output_embeddings: EmbeddingArray
    if settings.output_dtype == "float16":
        output_embeddings = float32_embeddings.astype(np.float16)
    else:
        output_embeddings = float32_embeddings
    preprocessing = _preprocess_metadata(weights)
    contract_input_variant = (
        "context_rgb" if settings.input_variant == "rgb" else "target_highlighted_rgb"
    )
    normalised_crop_binding = _normalise_source_crop_cache_binding(
        source_crop_cache_binding,
        expected_input_variant=contract_input_variant,
    )
    resolved_representation_id = representation_id or (
        "imagenet_resnet18_context_embeddings"
        if settings.input_variant == "rgb"
        else "imagenet_target_highlighted_embeddings"
    )
    encoder_metadata = {
        "architecture": "torchvision.models.resnet18",
        "classification_head": "removed (fc=Identity)",
        "encoder_frozen": True,
        "input_sha256": _array_sha256(prepared),
        "output_dimension": int(output_embeddings.shape[1]),
        "output_dtype": str(output_embeddings.dtype),
        "weight_identifier": str(weights),
        **(
            {
                "source_crop_cache_binding": normalised_crop_binding,
                "source_crop_cache_binding_sha256": canonical_sha256(normalised_crop_binding),
            }
            if normalised_crop_binding is not None
            else {}
        ),
    }
    encoder_implementation = {
        "module": "histo_audit.representations.imagenet",
        "entrypoint": "extract_resnet18_embeddings",
        "source_file_sha256": sha256_file(Path(__file__)),
        "torchvision_architecture": "resnet18",
        "torchvision_version": _package_version("torchvision"),
    }
    cache_recipe = {
        "identifier": "torchvision_resnet18_frozen_embedding_npz_v2",
        "array_keys": ["embeddings", "metadata_json", "sample_ids"],
        "context_brightness": (
            settings.context_brightness
            if settings.input_variant == "target_highlighted_rgb"
            else None
        ),
        "input_variant": contract_input_variant,
        "output_dtype": settings.output_dtype,
        "pickle_allowed": False,
        "preprocessing_sha256": canonical_sha256(preprocessing),
    }
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "sample_count": len(identifiers),
        "encoder_name": "torchvision.resnet18",
        "encoder_identifier": "resnet18_imagenet1k_v1",
        "encoder_frozen": True,
        "classification_head": "removed (fc=Identity)",
        "weight_identifier": str(weights),
        "weight_url": weights.url,
        "weight_path": str(weight_path.resolve()),
        "weight_sha256": sha256_file(weight_path),
        "weight_download_explicitly_allowed": settings.allow_weight_download,
        "weights_sha256": sha256_file(weight_path),
        "preprocessing_identifier": "torchvision_resnet18_imagenet1k_v1_official",
        "preprocessing": preprocessing,
        "input_variant": settings.input_variant,
        "legacy_input_variant": settings.input_variant,
        "contract_input_variant": contract_input_variant,
        "representation_id": resolved_representation_id,
        "context_brightness": (
            settings.context_brightness
            if settings.input_variant == "target_highlighted_rgb"
            else None
        ),
        "input_sha256": _array_sha256(prepared),
        "dtype": str(output_embeddings.dtype),
        "output_dimension": int(output_embeddings.shape[1]),
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp_enabled": use_amp,
        "torch_inference_mode": True,
        "batch_size_requested": requested_batch_size,
        "batch_size_initial_effective": min(requested_batch_size, windows_cap, len(prepared)),
        "batch_size_final": current_batch_size,
        "batch_oom_backoffs": backoffs,
        "data_loader_workers": 0,
        "batching_policy": "single-process batches with OOM halving",
        "extraction_seconds": elapsed,
        "extracted_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "versions": {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "torch": torch.__version__,
            "torchvision": _package_version("torchvision"),
        },
        "package_versions": {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "torch": str(torch.__version__),
            "torchvision": _package_version("torchvision"),
        },
        "encoder_metadata": encoder_metadata,
        "encoder_implementation": encoder_implementation,
        "cache_recipe": cache_recipe,
        "provenance_scope": (
            "stage_eligible"
            if manifest_sha256 is not None and raw_inventory_sha256 is not None
            else "non_stage_fixture"
        ),
        "configuration": asdict(settings),
        **(
            {
                "source_crop_cache_binding": normalised_crop_binding,
                "source_crop_cache_binding_sha256": canonical_sha256(normalised_crop_binding),
            }
            if normalised_crop_binding is not None
            else {}
        ),
    }
    if manifest_sha256 is not None:
        metadata["manifest_sha256"] = manifest_sha256
        metadata["crop_manifest_sha256"] = manifest_sha256
    if raw_inventory_sha256 is not None:
        metadata["raw_inventory_sha256"] = raw_inventory_sha256
    if analysis_eligibility is not None:
        metadata["analysis_eligibility"] = dict(analysis_eligibility)
    cache: Path | None = None
    sidecar: Path | None = None
    if cache_path is not None:
        cache, sidecar, metadata = save_embedding_cache(
            cache_path,
            output_embeddings,
            identifiers,
            metadata,
        )
    result = EmbeddingResult(
        embeddings=output_embeddings,
        sample_ids=identifiers,
        metadata=metadata,
        cache_path=cache,
        metadata_path=sidecar,
    )
    result.validate()
    return result


def run_resnet18_embedding_smoke(
    *,
    cache_dir: str | Path | None = None,
    device: str = "auto",
    allow_weight_download: bool = False,
    variants: tuple[InputVariant, ...] = ("rgb", "target_highlighted_rgb"),
) -> ResNet18SmokeResult:
    """Attempt a two-image RGB/highlighted smoke without implicit downloads."""

    images = np.zeros((2, 48, 48, 3), dtype=np.uint8)
    y, x = np.mgrid[:48, :48]
    images[0, ..., 0] = np.clip(40 + 3 * x, 0, 255).astype(np.uint8)
    images[0, ..., 1] = np.clip(30 + 2 * y, 0, 255).astype(np.uint8)
    images[1] = np.flip(images[0], axis=1)
    masks = np.stack(
        [
            (x - 20) ** 2 + (y - 24) ** 2 <= 8**2,
            (x - 28) ** 2 + (y - 24) ** 2 <= 7**2,
        ]
    )
    ids = np.asarray(["smoke-000", "smoke-001"], dtype=np.str_)
    output_shapes: dict[str, tuple[int, int]] = {}
    cache_paths: list[str] = []
    last_metadata: dict[str, Any] = {}
    try:
        for variant in variants:
            destination = (
                Path(cache_dir) / f"resnet18_{variant}_smoke.npz" if cache_dir is not None else None
            )
            result = extract_resnet18_embeddings(
                images,
                ids,
                target_masks=masks if variant == "target_highlighted_rgb" else None,
                config=ResNet18EmbeddingConfig(
                    input_variant=variant,
                    device=device,
                    batch_size=2,
                    allow_weight_download=allow_weight_download,
                ),
                cache_path=destination,
            )
            output_shapes[variant] = cast(tuple[int, int], result.embeddings.shape)
            if result.cache_path is not None:
                cache_paths.append(str(result.cache_path.resolve()))
            last_metadata = result.metadata
    except PretrainedWeightsUnavailableError as exc:
        return ResNet18SmokeResult(
            status="blocked",
            blocker=str(exc),
            variants={},
            cache_paths=(),
            device=None,
            weight_path=str(official_resnet18_weight_cache_path()),
            weight_sha256=None,
        )
    except Exception as exc:
        return ResNet18SmokeResult(
            status="failed",
            blocker=f"{type(exc).__name__}: {exc}",
            variants=output_shapes,
            cache_paths=tuple(cache_paths),
            device=cast(str | None, last_metadata.get("device")),
            weight_path=cast(str | None, last_metadata.get("weight_path")),
            weight_sha256=cast(str | None, last_metadata.get("weight_sha256")),
        )
    return ResNet18SmokeResult(
        status="passed",
        blocker=None,
        variants=output_shapes,
        cache_paths=tuple(cache_paths),
        device=cast(str, last_metadata["device"]),
        weight_path=cast(str, last_metadata["weight_path"]),
        weight_sha256=cast(str, last_metadata["weight_sha256"]),
    )


__all__ = [
    "EmbeddingResult",
    "PretrainedWeightsUnavailableError",
    "ResNet18EmbeddingConfig",
    "ResNet18SmokeResult",
    "extract_resnet18_embeddings",
    "load_embedding_cache",
    "official_resnet18_weight_cache_path",
    "run_resnet18_embedding_smoke",
    "save_embedding_cache",
]
