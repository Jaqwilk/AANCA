"""Memory-bounded, resumable publication of full PanNuke representation caches.

The public output directory remains absent until every cache/sidecar pair has
been produced and source freshness has been checked.  Intermediate arrays live
in a deterministic private workspace, are checkpointed by content digest, and
are retained under ``.chunked-state`` after publication so returned arrays can
remain memory mapped instead of being materialised in RAM.
"""

from __future__ import annotations

import atexit
import copy
import hashlib
import json
import os
import shutil
import threading
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from histo_audit.pannuke.exceptions import PanNukeSemanticsError
from histo_audit.pannuke.io import ensure_derived_output_outside_raw, sha256_file
from histo_audit.pannuke.models import PanNukeValidationResult
from histo_audit.pannuke.publication import (
    PublishedPath,
    assert_mutable_publication_destination,
    publish_flat_directory_no_overwrite,
    rollback_owned_publications,
)
from histo_audit.pannuke.validation import verify_raw_inventory_unchanged
from histo_audit.utils.run_tracking import atomic_write_npz

from . import cache_provenance as cache_provenance_module
from . import engineered as engineered_module
from . import imagenet as imagenet_module
from . import pannuke as pannuke_module
from .cache_provenance import (
    array_artifact_sha256,
    canonical_sha256,
    ordered_sample_ids_sha256,
    primary_cache_provenance_record,
    verify_frozen_cache_sidecar,
)
from .eligibility import _ordered_ids_sha256 as _eligibility_ordered_ids_sha256
from .eligibility import _semantic_sha256 as _eligibility_semantic_sha256
from .engineered import (
    EngineeredFeatureSet,
    build_engineered_feature_set,
    save_engineered_feature_cache,
)
from .imagenet import (
    EmbeddingResult,
    ResNet18EmbeddingConfig,
    extract_resnet18_embeddings,
    save_embedding_cache,
)
from .pannuke import (
    ContextMorphometricsCache,
    PanNukeCropBatch,
    PanNukeCropConfig,
    PanNukeRepresentationArtifacts,
    _contour_arrays,
    _embedding_crop_cache_binding,
    _engineered_cache_binding,
    _engineered_crop_cache_binding,
    _manifest_frame,
    _verify_validation_binding,
    extract_pannuke_crop_batch,
    save_context_morphometrics_cache,
    save_pannuke_crop_cache,
)

_CHECKPOINT_SCHEMA_VERSION = 1
_WORKSPACE_NAME = ".chunked-state"
_HASH_BLOCK_BYTES = 32 * 1024 * 1024
_WORKSPACE_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class _ChunkedWorkspaceLease:
    workspace: Path
    maps: tuple[np.memmap[Any, Any], ...]
    publications: tuple[PublishedPath, ...]


_LIVE_WORKSPACES: dict[Path, _ChunkedWorkspaceLease] = {}

_FIXED_CROP_FIELDS = (
    "sample_ids",
    "context_rgb",
    "target_highlighted_rgb",
    "target_masks",
    "target_contour_masks",
    "raw_component_counts",
    "disconnected_instance_flags",
    "projected_union_component_counts",
    "projection_fallback_component_counts",
    "projection_collision_pixel_counts",
    "projection_collision_excess_counts",
    "projection_adjacency_pair_counts",
    "projection_topology_changed",
    "source_crop_boxes",
    "source_target_boxes",
    "official_folds",
    "source_patch_indices",
    "instance_channel_indices",
    "instance_ids",
    "pre_corruption_labels",
    "group_ids",
    "tissue_types",
    "identity_verified",
    "primary_eligible",
    "confirmatory_eligible",
)

_COMPONENT_FIELDS = (
    "projected_component_pixel_counts",
    "projected_component_unique_pixel_counts",
    "baseline_projected_component_counts",
    "projection_fallback_component_flags",
)


def _strict_json_text(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
        sort_keys=True,
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"chunked checkpoint/state JSON is invalid: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"chunked checkpoint/state must be a JSON object: {path}")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(_strict_json_text(value, indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sidecar_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(f"{cache_path.suffix}.metadata.json")


def _raw_inventory_sha256(validation: PanNukeValidationResult) -> str:
    payload = json.dumps(
        [record.as_dict() for record in validation.inventory],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_file_hashes() -> dict[str, str]:
    modules = {
        "cache_provenance": cache_provenance_module,
        "engineered": engineered_module,
        "imagenet": imagenet_module,
        "pannuke": pannuke_module,
        "pannuke_chunked": cast(Any, __import__(__name__, fromlist=["__name__"])),
    }
    result: dict[str, str] = {}
    for name, module in modules.items():
        source = Path(str(module.__file__)).resolve()
        result[name] = sha256_file(source)
    return result


def _fixed_crop_specs(frame: Any, crop_size: int) -> dict[str, dict[str, Any]]:
    n_samples = len(frame)

    def spec(shape: tuple[int, ...], dtype: np.dtype[Any] | type[Any]) -> dict[str, Any]:
        return {"shape": list(shape), "dtype": np.dtype(dtype).str}

    sample_dtype = np.asarray(frame["sample_id"].astype(str).tolist(), dtype=np.str_).dtype
    group_dtype = np.asarray(frame["group_id"].astype(str).tolist(), dtype=np.str_).dtype
    tissue_dtype = np.asarray(frame["tissue_type"].astype(str).tolist(), dtype=np.str_).dtype
    return {
        "sample_ids": spec((n_samples,), sample_dtype),
        "context_rgb": spec((n_samples, crop_size, crop_size, 3), np.uint8),
        "target_highlighted_rgb": spec((n_samples, crop_size, crop_size, 3), np.uint8),
        "target_masks": spec((n_samples, crop_size, crop_size), np.bool_),
        "target_contour_masks": spec((n_samples, crop_size, crop_size), np.bool_),
        "raw_component_counts": spec((n_samples,), np.int32),
        "disconnected_instance_flags": spec((n_samples,), np.bool_),
        "projected_union_component_counts": spec((n_samples,), np.int32),
        "projection_fallback_component_counts": spec((n_samples,), np.int32),
        "projection_collision_pixel_counts": spec((n_samples,), np.int32),
        "projection_collision_excess_counts": spec((n_samples,), np.int32),
        "projection_adjacency_pair_counts": spec((n_samples,), np.int32),
        "projection_topology_changed": spec((n_samples,), np.bool_),
        "source_crop_boxes": spec((n_samples, 4), np.int32),
        "source_target_boxes": spec((n_samples, 4), np.int32),
        "official_folds": spec((n_samples,), np.int16),
        "source_patch_indices": spec((n_samples,), np.int32),
        "instance_channel_indices": spec((n_samples,), np.int16),
        "instance_ids": spec((n_samples,), np.int64),
        "pre_corruption_labels": spec((n_samples,), np.int64),
        "group_ids": spec((n_samples,), group_dtype),
        "tissue_types": spec((n_samples,), tissue_dtype),
        "identity_verified": spec((n_samples,), np.bool_),
        "primary_eligible": spec((n_samples,), np.bool_),
        "confirmatory_eligible": spec((n_samples,), np.bool_),
    }


def _array_path(array_directory: Path, name: str) -> Path:
    return array_directory / f"{name}.npy"


def _open_map(
    path: Path,
    spec: dict[str, Any],
    *,
    allow_create: bool,
) -> np.memmap[Any, Any]:
    shape = tuple(int(value) for value in spec["shape"])
    dtype = np.dtype(str(spec["dtype"]))
    if not path.exists():
        if not allow_create:
            raise RuntimeError(f"completed chunk workspace array is absent: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    try:
        mapped = np.load(path, mmap_mode="r+", allow_pickle=False)
    except (OSError, ValueError) as error:
        if not allow_create:
            raise RuntimeError(f"completed chunk workspace array is invalid: {path}") from error
        path.unlink(missing_ok=True)
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    if not isinstance(mapped, np.memmap) or mapped.shape != shape or mapped.dtype != dtype:
        if not allow_create:
            raise RuntimeError(f"completed chunk workspace array schema changed: {path}")
        del mapped
        path.unlink(missing_ok=True)
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    return mapped


def _open_fixed_maps(
    array_directory: Path,
    specs: dict[str, dict[str, Any]],
    *,
    allow_create: bool,
) -> dict[str, np.memmap[Any, Any]]:
    opened: dict[str, np.memmap[Any, Any]] = {}
    try:
        for name, spec in specs.items():
            opened[name] = _open_map(
                _array_path(array_directory, name),
                spec,
                allow_create=allow_create,
            )
    except BaseException:
        _close_maps(opened)
        raise
    return opened


def _flush_maps(maps: dict[str, np.memmap[Any, Any]]) -> None:
    for mapped in maps.values():
        mapped.flush()


def _close_maps(*collections: Mapping[str, NDArray[np.generic]]) -> None:
    seen: set[int] = set()
    for collection in collections:
        for value in collection.values():
            base: Any = value
            while isinstance(getattr(base, "base", None), np.ndarray):
                base = base.base
            if isinstance(base, np.memmap) and id(base) not in seen:
                seen.add(id(base))
                base.flush()
                mmap_handle = getattr(base, "_mmap", None)
                if mmap_handle is not None:
                    mmap_handle.close()


def cleanup_pannuke_chunked_workspace(
    artifacts_or_output: PanNukeRepresentationArtifacts | str | Path,
) -> None:
    """Close live memory maps and remove one private post-publication workspace.

    Array fields on a supplied ``PanNukeRepresentationArtifacts`` become invalid
    after this call.  Cache paths remain valid and the public output directory is
    never removed.  Normal CLI processes also invoke this cleanup through
    ``atexit`` so a successful extraction leaves only final NPZ/sidecar pairs.
    """

    output = (
        artifacts_or_output.crop_cache_path.parent
        if isinstance(artifacts_or_output, PanNukeRepresentationArtifacts)
        else Path(artifacts_or_output)
    ).resolve()
    with _WORKSPACE_LOCK:
        lease = _LIVE_WORKSPACES.pop(output, None)
    if lease is None:
        return
    _close_maps({str(index): mapped for index, mapped in enumerate(lease.maps)})
    try:
        shutil.rmtree(lease.workspace)
    except OSError:
        with _WORKSPACE_LOCK:
            _LIVE_WORKSPACES[output] = replace(lease, maps=())
        raise


def rollback_pannuke_chunked_publication(
    artifacts_or_output: PanNukeRepresentationArtifacts | str | Path,
) -> bool:
    """Retract one owned public bundle back into its resumable private workspace.

    This is intentionally narrower than cleanup: it is used when a downstream
    evidence artifact fails inside the same publication transaction.  Only the
    owned public hard links/directory are removed; the five expensive cache/sidecar
    pairs remain in ``resume/bundle`` while no public cache directory remains.  A
    missing live lease means the caller did not create the output through the
    chunked path and returns ``False`` without touching the filesystem.
    """

    output = (
        artifacts_or_output.crop_cache_path.parent
        if isinstance(artifacts_or_output, PanNukeRepresentationArtifacts)
        else Path(artifacts_or_output)
    ).resolve()
    with _WORKSPACE_LOCK:
        lease = _LIVE_WORKSPACES.pop(output, None)
    if lease is None:
        return False
    _close_maps({str(index): mapped for index, mapped in enumerate(lease.maps)})
    try:
        rollback_owned_publications(list(lease.publications))
    except BaseException:
        # Do not re-register a failed rollback: atexit cleanup must not erase the
        # resumable workspace while ownership requires manual investigation.
        raise
    return True


def _cleanup_all_chunked_workspaces() -> None:
    with _WORKSPACE_LOCK:
        outputs = tuple(_LIVE_WORKSPACES)
    for output in outputs:
        with suppress(OSError):
            cleanup_pannuke_chunked_workspace(output)


atexit.register(_cleanup_all_chunked_workspaces)


def _chunk_bounds(n_samples: int, chunk_size: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (start, min(start + chunk_size, n_samples)) for start in range(0, n_samples, chunk_size)
    )


def _chunk_eligibility_provenance(
    full: dict[str, Any], sample_ids: NDArray[np.str_]
) -> dict[str, Any]:
    result = copy.deepcopy(full)
    result.pop("semantic_sha256", None)
    identifiers = tuple(str(value) for value in sample_ids.tolist())
    result["output_sample_count"] = len(identifiers)
    result["output_sample_ids_sha256"] = _eligibility_ordered_ids_sha256(identifiers)
    result["all_output_primary_eligible"] = True
    result["all_output_confirmatory_eligible"] = True
    result["semantic_sha256"] = _eligibility_semantic_sha256(result)
    return result


def _fixed_slice_hashes(
    maps: dict[str, np.memmap[Any, Any]], start: int, stop: int
) -> dict[str, str]:
    return {
        name: array_artifact_sha256(np.asarray(mapped[start:stop]))
        for name, mapped in sorted(maps.items())
    }


def _crop_chunk_digest(
    maps: dict[str, np.memmap[Any, Any]],
    start: int,
    stop: int,
    variable_sha256: str,
) -> tuple[str, dict[str, str]]:
    fixed_hashes = _fixed_slice_hashes(maps, start, stop)
    digest = canonical_sha256(
        {
            "start": start,
            "stop": stop,
            "fixed_array_sha256_by_name": fixed_hashes,
            "variable_file_sha256": variable_sha256,
        }
    )
    return digest, fixed_hashes


def _checkpoint_update(path: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint["checkpoint_semantic_sha256"] = canonical_sha256(
        {key: value for key, value in checkpoint.items() if key != "checkpoint_semantic_sha256"}
    )
    _atomic_write_json(path, checkpoint)


def _validate_checkpoint(checkpoint: dict[str, Any], contract: dict[str, Any]) -> None:
    semantic = checkpoint.get("checkpoint_semantic_sha256")
    payload = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_semantic_sha256"
    }
    if semantic != canonical_sha256(payload):
        raise RuntimeError("chunked resume checkpoint checksum is invalid")
    if (
        checkpoint.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("contract_sha256") != canonical_sha256(contract)
        or checkpoint.get("contract") != contract
    ):
        raise RuntimeError(
            "chunked resume contract mismatch; source, manifest, sample order, or "
            "configuration changed"
        )
    completed = checkpoint.get("completed_chunks")
    publications = checkpoint.get("publications")
    if not isinstance(completed, dict) or not isinstance(publications, dict):
        raise RuntimeError("chunked resume checkpoint stage records are invalid")


def _load_crop_variable(path: Path) -> tuple[dict[str, NDArray[np.generic]], dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            expected = {
                *_COMPONENT_FIELDS,
                "projected_component_offsets",
                "source_contour_xy",
                "source_contour_offsets",
                "metadata_json",
            }
            if set(payload.files) != expected:
                raise RuntimeError(f"crop chunk variable schema is invalid: {path}")
            arrays = {
                name: np.asarray(payload[name]).copy()
                for name in expected
                if name != "metadata_json"
            }
            metadata = json.loads(str(payload["metadata_json"].item()))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"crop chunk variable payload is invalid: {path}") from error
    if not isinstance(metadata, dict):
        raise RuntimeError(f"crop chunk metadata is invalid: {path}")
    return arrays, metadata


def _write_crop_chunk(
    batch: PanNukeCropBatch,
    maps: dict[str, np.memmap[Any, Any]],
    start: int,
    stop: int,
    path: Path,
) -> tuple[str, dict[str, str], str]:
    if len(batch.sample_ids) != stop - start:
        raise RuntimeError("crop chunk length differs from its deterministic bounds")
    for name in _FIXED_CROP_FIELDS:
        maps[name][start:stop] = np.asarray(getattr(batch, name))
    _flush_maps(maps)
    contour_xy, contour_offsets = _contour_arrays(batch.source_contours_xy)
    arrays: dict[str, NDArray[np.generic]] = {
        name: np.asarray(getattr(batch, name)) for name in _COMPONENT_FIELDS
    }
    arrays.update(
        {
            "projected_component_offsets": np.asarray(batch.projected_component_offsets),
            "source_contour_xy": contour_xy,
            "source_contour_offsets": contour_offsets,
            "metadata_json": np.asarray(_strict_json_text(batch.metadata), dtype=np.str_),
        }
    )
    atomic_write_npz(path, arrays)
    variable_sha = sha256_file(path)
    digest, fixed_hashes = _crop_chunk_digest(maps, start, stop, variable_sha)
    return digest, fixed_hashes, variable_sha


def _verify_completed_crop_chunk(
    record: dict[str, Any],
    maps: dict[str, np.memmap[Any, Any]],
    start: int,
    stop: int,
    variable_path: Path,
) -> None:
    if record.get("start") != start or record.get("stop") != stop:
        raise RuntimeError("completed crop chunk bounds differ from deterministic order")
    if not variable_path.is_file():
        raise RuntimeError(f"completed crop chunk payload is absent: {variable_path}")
    variable_sha = sha256_file(variable_path)
    digest, fixed_hashes = _crop_chunk_digest(maps, start, stop, variable_sha)
    if (
        record.get("variable_file_sha256") != variable_sha
        or record.get("fixed_array_sha256_by_name") != fixed_hashes
        or record.get("digest") != digest
    ):
        raise RuntimeError(f"completed crop chunk content changed: {start}:{stop}")
    _load_crop_variable(variable_path)


def _merge_crop_metadata(
    chunk_metadata: list[dict[str, Any]],
    fixed_maps: dict[str, np.memmap[Any, Any]],
    eligibility: dict[str, Any],
) -> dict[str, Any]:
    if not chunk_metadata:
        raise RuntimeError("crop metadata cannot be merged from zero chunks")
    metadata = copy.deepcopy(chunk_metadata[0])
    first_projection = metadata.get("target_mask_projection")
    if not isinstance(first_projection, dict):
        raise RuntimeError("crop chunk lacks target-mask projection metadata")
    records: list[dict[str, Any]] = []
    for item in chunk_metadata:
        projection = item.get("target_mask_projection")
        if (
            item.get("manifest_sha256") != metadata.get("manifest_sha256")
            or item.get("raw_inventory_sha256") != metadata.get("raw_inventory_sha256")
            or item.get("crop_configuration") != metadata.get("crop_configuration")
            or not isinstance(projection, dict)
            or projection.get("identifier") != first_projection.get("identifier")
        ):
            raise RuntimeError("crop chunk provenance differs across deterministic chunks")
        chunk_records = projection.get("disconnected_instances")
        if not isinstance(chunk_records, list):
            raise RuntimeError("crop chunk disconnected-instance ledger is invalid")
        records.extend(copy.deepcopy(chunk_records))

    sample_ids = [str(value) for value in fixed_maps["sample_ids"].tolist()]
    fallback = np.asarray(fixed_maps["projection_fallback_component_counts"])
    collisions = np.asarray(fixed_maps["projection_collision_pixel_counts"])
    adjacency = np.asarray(fixed_maps["projection_adjacency_pair_counts"])
    topology = np.asarray(fixed_maps["projection_topology_changed"])
    projection = copy.deepcopy(first_projection)
    projection.pop("semantic_sha256", None)
    projection.update(
        {
            "sample_count": len(sample_ids),
            "raw_component_count": int(np.asarray(fixed_maps["raw_component_counts"]).sum()),
            "zero_covered_component_count": 0,
            "disconnected_instance_count": int(
                np.asarray(fixed_maps["disconnected_instance_flags"]).sum()
            ),
            "fallback_component_count": int(fallback.sum()),
            "fallback_instance_ids": [
                sample_id
                for sample_id, count in zip(sample_ids, fallback.tolist(), strict=True)
                if int(count) > 0
            ],
            "collision_instance_count": int(np.count_nonzero(collisions)),
            "collision_instance_ids": [
                sample_id
                for sample_id, count in zip(sample_ids, collisions.tolist(), strict=True)
                if int(count) > 0
            ],
            "adjacency_instance_count": int(np.count_nonzero(adjacency)),
            "adjacency_instance_ids": [
                sample_id
                for sample_id, count in zip(sample_ids, adjacency.tolist(), strict=True)
                if int(count) > 0
            ],
            "topology_changed_instance_count": int(np.count_nonzero(topology)),
            "topology_changed_instance_ids": [
                sample_id
                for sample_id, changed in zip(sample_ids, topology.tolist(), strict=True)
                if bool(changed)
            ],
            "disconnected_instances": records,
        }
    )
    projection["semantic_sha256"] = canonical_sha256(projection)
    metadata["target_mask_projection"] = projection
    metadata["analysis_eligibility"] = copy.deepcopy(eligibility)
    metadata["sample_count"] = len(sample_ids)
    metadata["chunked_extraction"] = {
        "schema_version": 1,
        "memory_bounded": True,
        "chunk_count": len(chunk_metadata),
        "source_annotations_modified": False,
    }
    return metadata


def _consolidate_crop_variable_arrays(
    state_directory: Path,
    chunk_paths: tuple[Path, ...],
    bounds: tuple[tuple[int, int], ...],
    fixed_maps: dict[str, np.memmap[Any, Any]],
    eligibility: dict[str, Any],
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
) -> tuple[dict[str, np.memmap[Any, Any]], dict[str, Any]]:
    chunk_metadata: list[dict[str, Any]] = []
    component_total = 0
    contour_total = 0
    for path in chunk_paths:
        payload_arrays, metadata = _load_crop_variable(path)
        component_total += int(np.asarray(payload_arrays["projected_component_offsets"])[-1])
        contour_total += int(np.asarray(payload_arrays["source_contour_offsets"])[-1])
        chunk_metadata.append(metadata)

    n_samples = len(fixed_maps["sample_ids"])
    specs = {
        "projected_component_pixel_counts": {
            "shape": [component_total],
            "dtype": np.dtype(np.int32).str,
        },
        "projected_component_unique_pixel_counts": {
            "shape": [component_total],
            "dtype": np.dtype(np.int32).str,
        },
        "baseline_projected_component_counts": {
            "shape": [component_total],
            "dtype": np.dtype(np.int32).str,
        },
        "projection_fallback_component_flags": {
            "shape": [component_total],
            "dtype": np.dtype(np.bool_).str,
        },
        "projected_component_offsets": {
            "shape": [n_samples + 1],
            "dtype": np.dtype(np.int64).str,
        },
        "source_contour_xy": {
            "shape": [contour_total, 2],
            "dtype": np.dtype(np.int32).str,
        },
        "source_contour_offsets": {
            "shape": [n_samples + 1],
            "dtype": np.dtype(np.int64).str,
        },
    }
    record = checkpoint.get("crop_consolidation")
    allow_create = record is None
    arrays = _open_fixed_maps(state_directory / "arrays", specs, allow_create=allow_create)
    metadata_path = state_directory / "crop_metadata.json"
    if record is None:
        component_cursor = 0
        contour_cursor = 0
        arrays["projected_component_offsets"][0] = 0
        arrays["source_contour_offsets"][0] = 0
        for path, (start, stop) in zip(chunk_paths, bounds, strict=True):
            local, _ = _load_crop_variable(path)
            local_component_offsets = np.asarray(local["projected_component_offsets"])
            local_contour_offsets = np.asarray(local["source_contour_offsets"])
            local_component_count = int(local_component_offsets[-1])
            local_contour_count = int(local_contour_offsets[-1])
            for name in _COMPONENT_FIELDS:
                arrays[name][component_cursor : component_cursor + local_component_count] = local[
                    name
                ]
            arrays["projected_component_offsets"][start : stop + 1] = (
                component_cursor + local_component_offsets
            )
            arrays["source_contour_xy"][contour_cursor : contour_cursor + local_contour_count] = (
                local["source_contour_xy"]
            )
            arrays["source_contour_offsets"][start : stop + 1] = (
                contour_cursor + local_contour_offsets
            )
            component_cursor += local_component_count
            contour_cursor += local_contour_count
        _flush_maps(arrays)
        metadata = _merge_crop_metadata(chunk_metadata, fixed_maps, eligibility)
        _atomic_write_json(metadata_path, metadata)
        array_hashes = {
            name: array_artifact_sha256(np.asarray(value)) for name, value in sorted(arrays.items())
        }
        record = {
            "component_count": component_total,
            "contour_point_count": contour_total,
            "array_sha256_by_name": array_hashes,
            "metadata_file_sha256": sha256_file(metadata_path),
            "digest": canonical_sha256(
                {
                    "array_sha256_by_name": array_hashes,
                    "metadata_file_sha256": sha256_file(metadata_path),
                }
            ),
        }
        checkpoint["crop_consolidation"] = record
        _checkpoint_update(checkpoint_path, checkpoint)
    else:
        if not isinstance(record, dict) or not metadata_path.is_file():
            raise RuntimeError("completed crop consolidation state is incomplete")
        array_hashes = {
            name: array_artifact_sha256(np.asarray(value)) for name, value in sorted(arrays.items())
        }
        metadata_sha = sha256_file(metadata_path)
        if (
            record.get("component_count") != component_total
            or record.get("contour_point_count") != contour_total
            or record.get("array_sha256_by_name") != array_hashes
            or record.get("metadata_file_sha256") != metadata_sha
            or record.get("digest")
            != canonical_sha256(
                {
                    "array_sha256_by_name": array_hashes,
                    "metadata_file_sha256": metadata_sha,
                }
            )
        ):
            raise RuntimeError("completed crop consolidation content changed")
        metadata = _read_json_object(metadata_path)
    return arrays, metadata


def _make_crop_batch(
    fixed: Mapping[str, NDArray[np.generic]],
    variable: Mapping[str, NDArray[np.generic]],
    metadata: dict[str, Any],
    validation: PanNukeValidationResult,
) -> PanNukeCropBatch:
    contour_xy = np.asarray(variable["source_contour_xy"])
    contour_offsets = np.asarray(variable["source_contour_offsets"])
    contours = tuple(
        contour_xy[int(contour_offsets[index]) : int(contour_offsets[index + 1])]
        for index in range(len(fixed["sample_ids"]))
    )
    return PanNukeCropBatch(
        sample_ids=cast(Any, fixed["sample_ids"]),
        context_rgb=cast(Any, fixed["context_rgb"]),
        target_highlighted_rgb=cast(Any, fixed["target_highlighted_rgb"]),
        target_masks=cast(Any, fixed["target_masks"]),
        target_contour_masks=cast(Any, fixed["target_contour_masks"]),
        raw_component_counts=cast(Any, fixed["raw_component_counts"]),
        disconnected_instance_flags=cast(Any, fixed["disconnected_instance_flags"]),
        projected_union_component_counts=cast(Any, fixed["projected_union_component_counts"]),
        projection_fallback_component_counts=cast(
            Any, fixed["projection_fallback_component_counts"]
        ),
        projection_collision_pixel_counts=cast(Any, fixed["projection_collision_pixel_counts"]),
        projection_collision_excess_counts=cast(Any, fixed["projection_collision_excess_counts"]),
        projection_adjacency_pair_counts=cast(Any, fixed["projection_adjacency_pair_counts"]),
        projection_topology_changed=cast(Any, fixed["projection_topology_changed"]),
        projected_component_pixel_counts=cast(Any, variable["projected_component_pixel_counts"]),
        projected_component_unique_pixel_counts=cast(
            Any, variable["projected_component_unique_pixel_counts"]
        ),
        baseline_projected_component_counts=cast(
            Any, variable["baseline_projected_component_counts"]
        ),
        projection_fallback_component_flags=cast(
            Any, variable["projection_fallback_component_flags"]
        ),
        projected_component_offsets=cast(Any, variable["projected_component_offsets"]),
        source_crop_boxes=cast(Any, fixed["source_crop_boxes"]),
        source_target_boxes=cast(Any, fixed["source_target_boxes"]),
        official_folds=cast(Any, fixed["official_folds"]),
        source_patch_indices=cast(Any, fixed["source_patch_indices"]),
        instance_channel_indices=cast(Any, fixed["instance_channel_indices"]),
        instance_ids=cast(Any, fixed["instance_ids"]),
        pre_corruption_labels=cast(Any, fixed["pre_corruption_labels"]),
        group_ids=cast(Any, fixed["group_ids"]),
        tissue_types=cast(Any, fixed["tissue_types"]),
        source_contours_xy=cast(Any, contours),
        identity_verified=cast(Any, fixed["identity_verified"]),
        primary_eligible=cast(Any, fixed["primary_eligible"]),
        confirmatory_eligible=cast(Any, fixed["confirmatory_eligible"]),
        metadata=metadata,
        validation_binding=validation,
    )


def _verify_stage_cache(
    path: Path,
    *,
    manifest_sha256: str,
    raw_inventory_sha256: str,
    sample_order_sha256: str,
    representation_id: str,
) -> dict[str, Any]:
    verification = verify_frozen_cache_sidecar(
        path,
        expected_manifest_sha256=manifest_sha256,
        expected_representation_id=representation_id,
    )
    metadata = verification.metadata
    if (
        metadata.get("raw_inventory_sha256") != raw_inventory_sha256
        or metadata.get("sample_order_sha256") != sample_order_sha256
        or metadata.get("provenance_scope") != "stage_eligible"
        or metadata.get("primary_cache_provenance") != primary_cache_provenance_record(metadata)
    ):
        raise RuntimeError(f"stage cache provenance differs from chunk contract: {path}")
    return {
        "cache_file_sha256": verification.cache_file_sha256,
        "sidecar_file_sha256": verification.sidecar_file_sha256,
        "sidecar_semantic_sha256": verification.sidecar_semantic_sha256,
    }


def _ensure_cache_publication(
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    stage: str,
    cache_path: Path,
    publisher: Callable[[], Any],
    *,
    manifest_sha256: str,
    raw_inventory_sha256: str,
    sample_order_sha256: str,
    representation_id: str,
) -> dict[str, Any]:
    sidecar = _sidecar_path(cache_path)
    if cache_path.exists() != sidecar.exists():
        raise RuntimeError(f"partial cache/sidecar pair exists in resume workspace: {cache_path}")
    if not cache_path.exists():
        publisher()
    actual = _verify_stage_cache(
        cache_path,
        manifest_sha256=manifest_sha256,
        raw_inventory_sha256=raw_inventory_sha256,
        sample_order_sha256=sample_order_sha256,
        representation_id=representation_id,
    )
    previous = checkpoint["publications"].get(stage)
    if previous is not None and previous != actual:
        raise RuntimeError(f"completed cache publication changed: {stage}")
    if previous is None:
        checkpoint["publications"][stage] = actual
        _checkpoint_update(checkpoint_path, checkpoint)
    return verify_frozen_cache_sidecar(cache_path).metadata


def _ensure_engineered_chunks(
    crops: PanNukeCropBatch,
    bounds: tuple[tuple[int, int], ...],
    state_directory: Path,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
) -> tuple[EngineeredFeatureSet, dict[str, NDArray[np.generic]]]:
    records = checkpoint["completed_chunks"].setdefault("engineered", {})
    if not isinstance(records, dict):
        raise RuntimeError("engineered chunk checkpoint is invalid")
    state_path = state_directory / "engineered_state.json"
    map_path = state_directory / "arrays" / "engineered_values.npy"
    state = _read_json_object(state_path) if state_path.is_file() else None
    if records and state is None:
        raise RuntimeError("completed engineered chunks lack their array schema")
    values: np.memmap[Any, Any] | None = None
    names: tuple[str, ...] | None = None
    if state is not None:
        names_raw = state.get("names")
        if not isinstance(names_raw, list) or not names_raw:
            raise RuntimeError("engineered chunk feature-name state is invalid")
        names = tuple(str(value) for value in names_raw)
        spec = {"shape": state.get("shape"), "dtype": state.get("dtype")}
        values = _open_map(map_path, spec, allow_create=not bool(records))

    for chunk_index, (start, stop) in enumerate(bounds):
        key = str(chunk_index)
        record = records.get(key)
        if record is not None:
            if values is None or not isinstance(record, dict):
                raise RuntimeError("completed engineered chunk state is incomplete")
            digest = array_artifact_sha256(np.asarray(values[start:stop]))
            if (
                record.get("start") != start
                or record.get("stop") != stop
                or record.get("values_sha256") != digest
            ):
                raise RuntimeError(f"completed engineered chunk content changed: {start}:{stop}")
            continue
        feature_chunk = build_engineered_feature_set(
            np.asarray(crops.context_rgb[start:stop]),
            np.asarray(crops.target_masks[start:stop]),
        )
        if values is None:
            names = feature_chunk.names
            spec = {
                "shape": [len(crops.sample_ids), feature_chunk.values.shape[1]],
                "dtype": feature_chunk.values.dtype.str,
            }
            values = _open_map(map_path, spec, allow_create=True)
            _atomic_write_json(
                state_path,
                {"schema_version": 1, "names": list(names), **spec},
            )
        if names != feature_chunk.names or values.shape[1] != feature_chunk.values.shape[1]:
            raise RuntimeError("engineered feature schema differs across chunks")
        values[start:stop] = feature_chunk.values
        values.flush()
        digest = array_artifact_sha256(np.asarray(values[start:stop]))
        records[key] = {"start": start, "stop": stop, "values_sha256": digest}
        _checkpoint_update(checkpoint_path, checkpoint)
    if values is None or names is None:
        raise RuntimeError("engineered chunk extraction produced no values")
    result = EngineeredFeatureSet(values=cast(Any, values), names=names)
    result.validate(expected_samples=len(crops.sample_ids))
    return result, {"engineered_values": values}


def _stream_imagenet_array_sha256(array: NDArray[np.generic]) -> str:
    value = np.asarray(array)
    if not value.flags.c_contiguous:
        raise ValueError("chunked ImageNet input hash requires C-contiguous storage")
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    bytes_per_row = max(1, int(np.prod(value.shape[1:], dtype=np.int64)) * value.dtype.itemsize)
    rows = max(1, _HASH_BLOCK_BYTES // bytes_per_row)
    for start in range(0, len(value), rows):
        digest.update(np.ascontiguousarray(value[start : start + rows]).tobytes(order="C"))
    return digest.hexdigest()


def _runtime_embedding_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(metadata)
    runtime["contract_input_variant"] = str(metadata["input_variant"])
    runtime["input_variant"] = str(
        metadata.get(
            "legacy_input_variant",
            "rgb" if metadata["input_variant"] == "context_rgb" else metadata["input_variant"],
        )
    )
    return runtime


def _aggregate_embedding_metadata(
    chunk_metadata: list[dict[str, Any]],
    bounds: tuple[tuple[int, int], ...],
    input_array: NDArray[np.generic],
    sample_count: int,
    chunk_size: int,
    manifest_sha256: str,
    raw_inventory_sha256: str,
    analysis_eligibility: dict[str, Any],
    source_crop_binding: dict[str, Any],
) -> dict[str, Any]:
    if len(chunk_metadata) != len(bounds) or not chunk_metadata:
        raise RuntimeError("embedding metadata does not cover every deterministic chunk")
    metadata = copy.deepcopy(chunk_metadata[0])
    stable_fields = (
        "encoder_identifier",
        "weight_identifier",
        "weights_sha256",
        "preprocessing_identifier",
        "preprocessing",
        "configuration",
        "dtype",
        "output_dimension",
    )
    for item in chunk_metadata[1:]:
        if any(item.get(field) != metadata.get(field) for field in stable_fields):
            raise RuntimeError("frozen encoder provenance differs across embedding chunks")
    input_sha256 = _stream_imagenet_array_sha256(input_array)
    encoder_metadata = dict(metadata.get("encoder_metadata", {}))
    encoder_metadata["input_sha256"] = input_sha256
    encoder_metadata["output_dimension"] = 512
    encoder_metadata["source_crop_cache_binding"] = source_crop_binding
    encoder_metadata["source_crop_cache_binding_sha256"] = canonical_sha256(source_crop_binding)
    backoffs: list[dict[str, int]] = []
    for item, (start, _) in zip(chunk_metadata, bounds, strict=True):
        raw_backoffs = item.get("batch_oom_backoffs", [])
        if not isinstance(raw_backoffs, list):
            raise RuntimeError("embedding chunk OOM-backoff ledger is invalid")
        for raw in raw_backoffs:
            if not isinstance(raw, dict):
                raise RuntimeError("embedding chunk OOM-backoff record is invalid")
            adjusted = {str(key): int(value) for key, value in raw.items()}
            adjusted["sample_offset"] = start + int(adjusted.get("sample_offset", 0))
            backoffs.append(adjusted)
    extraction_seconds = sum(float(item.get("extraction_seconds", 0.0)) for item in chunk_metadata)
    cache_recipe = dict(metadata.get("cache_recipe", {}))
    cache_recipe["chunked_extraction"] = {
        "schema_version": 1,
        "chunk_size": chunk_size,
        "chunk_count": len(bounds),
        "sample_order_preserved": True,
        "resume_digest_verified": True,
    }
    metadata.update(
        {
            "sample_count": sample_count,
            "input_sha256": input_sha256,
            "encoder_metadata": encoder_metadata,
            "cache_recipe": cache_recipe,
            "manifest_sha256": manifest_sha256,
            "crop_manifest_sha256": manifest_sha256,
            "raw_inventory_sha256": raw_inventory_sha256,
            "analysis_eligibility": copy.deepcopy(analysis_eligibility),
            "source_crop_cache_binding": source_crop_binding,
            "source_crop_cache_binding_sha256": canonical_sha256(source_crop_binding),
            "batch_oom_backoffs": backoffs,
            "batch_size_initial_effective": min(
                int(item.get("batch_size_initial_effective", sample_count))
                for item in chunk_metadata
            ),
            "batch_size_final": min(
                int(item.get("batch_size_final", sample_count)) for item in chunk_metadata
            ),
            "extraction_seconds": extraction_seconds,
            "provenance_scope": "stage_eligible",
            "chunked_extraction": {
                "schema_version": 1,
                "chunk_size": chunk_size,
                "chunk_count": len(bounds),
                "memory_bounded": True,
                "resumable": True,
                "sample_order_preserved": True,
            },
        }
    )
    return metadata


def _ensure_embedding_chunks(
    stage: str,
    crops: PanNukeCropBatch,
    bounds: tuple[tuple[int, int], ...],
    state_directory: Path,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    *,
    config: ResNet18EmbeddingConfig,
    manifest_sha256: str,
    raw_inventory_sha256: str,
    representation_id: str,
    source_crop_binding: dict[str, Any],
    chunk_size: int,
) -> tuple[EmbeddingResult, dict[str, NDArray[np.generic]]]:
    records = checkpoint["completed_chunks"].setdefault(stage, {})
    if not isinstance(records, dict):
        raise RuntimeError(f"embedding chunk checkpoint is invalid: {stage}")
    stage_directory = state_directory / "embedding_chunks" / stage
    state_path = stage_directory / "state.json"
    map_path = state_directory / "arrays" / f"{stage}_embeddings.npy"
    state = _read_json_object(state_path) if state_path.is_file() else None
    if records and state is None:
        raise RuntimeError(f"completed embedding chunks lack array schema: {stage}")
    values: np.memmap[Any, Any] | None = None
    if state is not None:
        values = _open_map(
            map_path,
            {"shape": state.get("shape"), "dtype": state.get("dtype")},
            allow_create=not bool(records),
        )

    metadata_by_chunk: list[dict[str, Any]] = []
    for chunk_index, (start, stop) in enumerate(bounds):
        key = str(chunk_index)
        metadata_path = stage_directory / f"chunk_{chunk_index:06d}.metadata.json"
        record = records.get(key)
        if record is not None:
            if values is None or not isinstance(record, dict) or not metadata_path.is_file():
                raise RuntimeError(f"completed embedding chunk state is incomplete: {stage}")
            values_sha = array_artifact_sha256(np.asarray(values[start:stop]))
            metadata_sha = sha256_file(metadata_path)
            if (
                record.get("start") != start
                or record.get("stop") != stop
                or record.get("values_sha256") != values_sha
                or record.get("metadata_file_sha256") != metadata_sha
                or record.get("digest")
                != canonical_sha256(
                    {"values_sha256": values_sha, "metadata_file_sha256": metadata_sha}
                )
            ):
                raise RuntimeError(
                    f"completed embedding chunk content changed: {stage} {start}:{stop}"
                )
            metadata_by_chunk.append(_read_json_object(metadata_path))
            continue
        masks = (
            np.asarray(crops.target_masks[start:stop])
            if config.input_variant == "target_highlighted_rgb"
            else None
        )
        result = extract_resnet18_embeddings(
            np.asarray(crops.context_rgb[start:stop]),
            np.asarray(crops.sample_ids[start:stop]),
            target_masks=masks,
            config=config,
            manifest_sha256=manifest_sha256,
            raw_inventory_sha256=raw_inventory_sha256,
            representation_id=representation_id,
            analysis_eligibility=crops.metadata["analysis_eligibility"],
            source_crop_cache_binding=source_crop_binding,
        )
        if values is None:
            spec = {
                "shape": [len(crops.sample_ids), 512],
                "dtype": result.embeddings.dtype.str,
            }
            values = _open_map(map_path, spec, allow_create=True)
            _atomic_write_json(state_path, {"schema_version": 1, **spec})
        if result.embeddings.shape != (stop - start, 512) or (
            result.embeddings.dtype != values.dtype
        ):
            raise RuntimeError(f"embedding chunk schema differs across chunks: {stage}")
        values[start:stop] = result.embeddings
        values.flush()
        _atomic_write_json(metadata_path, result.metadata)
        values_sha = array_artifact_sha256(np.asarray(values[start:stop]))
        metadata_sha = sha256_file(metadata_path)
        records[key] = {
            "start": start,
            "stop": stop,
            "values_sha256": values_sha,
            "metadata_file_sha256": metadata_sha,
            "digest": canonical_sha256(
                {"values_sha256": values_sha, "metadata_file_sha256": metadata_sha}
            ),
        }
        _checkpoint_update(checkpoint_path, checkpoint)
        metadata_by_chunk.append(result.metadata)
    if values is None:
        raise RuntimeError(f"embedding chunk extraction produced no values: {stage}")
    input_array = (
        crops.target_highlighted_rgb
        if config.input_variant == "target_highlighted_rgb"
        else crops.context_rgb
    )
    metadata = _aggregate_embedding_metadata(
        metadata_by_chunk,
        bounds,
        input_array,
        len(crops.sample_ids),
        chunk_size,
        manifest_sha256,
        raw_inventory_sha256,
        crops.metadata["analysis_eligibility"],
        source_crop_binding,
    )
    result = EmbeddingResult(
        embeddings=cast(Any, values),
        sample_ids=crops.sample_ids,
        metadata=metadata,
    )
    result.validate()
    return result, {f"{stage}_embeddings": values}


def _load_context_morphometrics(
    cache_path: Path, metadata: dict[str, Any]
) -> ContextMorphometricsCache:
    with np.load(cache_path, allow_pickle=False) as payload:
        if set(payload.files) != {"values", "names", "sample_ids"}:
            raise RuntimeError("context+morphometrics cache array schema is invalid")
        values = np.asarray(payload["values"], dtype=np.float32)
        names = tuple(str(value) for value in payload["names"].tolist())
        sample_ids = np.asarray(payload["sample_ids"], dtype=np.str_)
    result = ContextMorphometricsCache(
        values=values,
        names=names,
        sample_ids=sample_ids,
        metadata=metadata,
        cache_path=cache_path,
        metadata_path=_sidecar_path(cache_path),
    )
    result.validate()
    return result


def _prepare_checkpoint(
    resume: Path,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], Path, Path, Path]:
    checkpoint_path = resume / "checkpoint.json"
    bundle = resume / "bundle"
    state = resume / _WORKSPACE_NAME
    if resume.exists():
        if not checkpoint_path.is_file():
            raise RuntimeError(
                f"chunked resume directory lacks a valid checkpoint and was preserved: {resume}"
            )
        checkpoint = _read_json_object(checkpoint_path)
        _validate_checkpoint(checkpoint, contract)
        if not bundle.is_dir() or not state.is_dir():
            raise RuntimeError("chunked resume workspace is incomplete")
        return checkpoint, checkpoint_path, bundle, state
    resume.mkdir(parents=False, exist_ok=False)
    bundle.mkdir(parents=False, exist_ok=False)
    state.mkdir(parents=False, exist_ok=False)
    checkpoint = {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "contract": contract,
        "contract_sha256": canonical_sha256(contract),
        "completed_chunks": {
            "crops": {},
            "engineered": {},
            "highlighted_embeddings": {},
            "context_embeddings": {},
        },
        "publications": {},
        "state": "building",
    }
    _checkpoint_update(checkpoint_path, checkpoint)
    return checkpoint, checkpoint_path, bundle, state


def build_pannuke_representation_cache_chunked(
    validation: PanNukeValidationResult,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    sample_ids: tuple[str, ...] | None = None,
    crop_config: PanNukeCropConfig | None = None,
    resnet_config: ResNet18EmbeddingConfig | None = None,
    include_context_embeddings: bool = False,
    chunk_size: int = 4_096,
) -> PanNukeRepresentationArtifacts:
    """Build a full representation bundle with bounded transient memory and resume."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    crop_settings = crop_config or PanNukeCropConfig()
    crop_settings.validate()
    encoder_settings = resnet_config or ResNet18EmbeddingConfig(
        input_variant="target_highlighted_rgb",
        context_brightness=crop_settings.context_brightness,
    )
    encoder_settings.validate()
    if encoder_settings.input_variant != "target_highlighted_rgb":
        raise ValueError("the declared PanNuke pilot requires target_highlighted_rgb")
    if encoder_settings.context_brightness != crop_settings.context_brightness:
        raise ValueError("crop and encoder context-brightness policies differ")

    output = ensure_derived_output_outside_raw(
        output_dir,
        validation.root,
        purpose="PanNuke representation output directory",
    )
    resume_candidate = output.parent / f".{output.name}.chunked-resume"
    resume = ensure_derived_output_outside_raw(
        resume_candidate,
        validation.root,
        purpose="PanNuke chunked representation resume directory",
    )
    try:
        output = assert_mutable_publication_destination(
            output_dir,
            role="PanNuke representation output directory",
        )
        resume = assert_mutable_publication_destination(
            resume_candidate,
            role="PanNuke chunked representation resume directory",
        )
    except (NotADirectoryError, PermissionError, RuntimeError) as error:
        raise PanNukeSemanticsError(str(error)) from error
    pannuke_module.require_full_manifest_cache_disk_space(
        manifest_path,
        output,
        sample_ids=sample_ids,
    )
    if os.path.lexists(output):
        raise FileExistsError(f"representation output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    _verify_validation_binding(validation)
    source_manifest = Path(manifest_path).resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(f"PanNuke manifest does not exist: {source_manifest}")
    frame, eligibility, manifest_sha = _manifest_frame(
        source_manifest,
        sample_ids,
        eligibility_scope="analysis",
    )
    identifiers = np.asarray(frame["sample_id"].astype(str).tolist(), dtype=np.str_)
    sample_order_sha = ordered_sample_ids_sha256(identifiers)
    raw_inventory_sha = _raw_inventory_sha256(validation)
    fixed_specs = _fixed_crop_specs(frame, crop_settings.output_size)
    bounds = _chunk_bounds(len(identifiers), chunk_size)
    contract = {
        "schema_version": 1,
        "raw_root": str(validation.root.resolve()),
        "raw_inventory_sha256": raw_inventory_sha,
        "manifest_path": str(source_manifest),
        "manifest_sha256": manifest_sha,
        "sample_count": len(identifiers),
        "sample_order_sha256": sample_order_sha,
        "analysis_eligibility_semantic_sha256": eligibility["semantic_sha256"],
        "crop_configuration": asdict(crop_settings),
        "resnet_configuration": asdict(encoder_settings),
        "include_context_embeddings": include_context_embeddings,
        "chunk_size": chunk_size,
        "chunk_bounds": [list(value) for value in bounds],
        "fixed_crop_array_specs": fixed_specs,
        "producer_source_file_sha256": _source_file_hashes(),
        "output_directory": str(output),
    }
    checkpoint, checkpoint_path, bundle, state = _prepare_checkpoint(resume, contract)
    crop_records = checkpoint["completed_chunks"].get("crops")
    if not isinstance(crop_records, dict):
        raise RuntimeError("crop chunk checkpoint is invalid")
    fixed_maps = _open_fixed_maps(
        state / "arrays",
        fixed_specs,
        allow_create=not bool(crop_records),
    )
    chunk_directory = state / "crop_chunks"
    chunk_paths = tuple(chunk_directory / f"chunk_{index:06d}.npz" for index in range(len(bounds)))
    for chunk_index, ((start, stop), variable_path) in enumerate(
        zip(bounds, chunk_paths, strict=True)
    ):
        key = str(chunk_index)
        record = crop_records.get(key)
        if record is not None:
            if not isinstance(record, dict):
                raise RuntimeError("completed crop chunk checkpoint record is invalid")
            _verify_completed_crop_chunk(record, fixed_maps, start, stop, variable_path)
            continue
        chunk_frame = frame.iloc[start:stop].copy().reset_index(drop=True)
        chunk_ids = np.asarray(chunk_frame["sample_id"].astype(str).tolist(), dtype=np.str_)
        chunk_eligibility = _chunk_eligibility_provenance(eligibility, chunk_ids)
        batch = extract_pannuke_crop_batch(
            validation,
            source_manifest,
            config=crop_settings,
            _selection_override=(chunk_frame, chunk_eligibility, manifest_sha),
            _verify_raw_sources=False,
        )
        digest, fixed_hashes, variable_sha = _write_crop_chunk(
            batch, fixed_maps, start, stop, variable_path
        )
        crop_records[key] = {
            "start": start,
            "stop": stop,
            "fixed_array_sha256_by_name": fixed_hashes,
            "variable_file_sha256": variable_sha,
            "digest": digest,
        }
        _checkpoint_update(checkpoint_path, checkpoint)

    variable_maps, crop_metadata = _consolidate_crop_variable_arrays(
        state,
        chunk_paths,
        bounds,
        fixed_maps,
        eligibility,
        checkpoint,
        checkpoint_path,
    )
    crops = _make_crop_batch(fixed_maps, variable_maps, crop_metadata, validation)
    verify_raw_inventory_unchanged(validation)
    if sha256_file(source_manifest) != manifest_sha:
        raise PanNukeSemanticsError("PanNuke manifest changed during chunked crop extraction")

    crop_cache = bundle / "pannuke_crops.npz"
    _ensure_cache_publication(
        checkpoint,
        checkpoint_path,
        "crops",
        crop_cache,
        lambda: save_pannuke_crop_cache(crops, crop_cache, _verify_raw_sources=False),
        manifest_sha256=manifest_sha,
        raw_inventory_sha256=raw_inventory_sha,
        sample_order_sha256=sample_order_sha,
        representation_id="pannuke_component_covering_target_crops",
    )
    highlighted_crop_binding = _embedding_crop_cache_binding(
        crops, crop_cache, input_variant="target_highlighted_rgb"
    )
    engineered_crop_binding = _engineered_crop_cache_binding(crops, crop_cache)

    engineered, engineered_maps = _ensure_engineered_chunks(
        crops, bounds, state, checkpoint, checkpoint_path
    )
    engineered_cache = bundle / "pannuke_engineered_features.npz"
    _ensure_cache_publication(
        checkpoint,
        checkpoint_path,
        "engineered",
        engineered_cache,
        lambda: save_engineered_feature_cache(
            engineered,
            crops.sample_ids,
            engineered_cache,
            manifest_sha256=manifest_sha,
            raw_inventory_sha256=raw_inventory_sha,
            analysis_eligibility=crops.metadata["analysis_eligibility"],
            target_mask_projection=crops.metadata["target_mask_projection"],
            source_crop_cache_binding=engineered_crop_binding,
        ),
        manifest_sha256=manifest_sha,
        raw_inventory_sha256=raw_inventory_sha,
        sample_order_sha256=sample_order_sha,
        representation_id="engineered_target_features",
    )
    engineered_exact_binding = _engineered_cache_binding(engineered_cache)

    highlighted, highlighted_maps = _ensure_embedding_chunks(
        "highlighted_embeddings",
        crops,
        bounds,
        state,
        checkpoint,
        checkpoint_path,
        config=encoder_settings,
        manifest_sha256=manifest_sha,
        raw_inventory_sha256=raw_inventory_sha,
        representation_id="imagenet_target_highlighted_embeddings",
        source_crop_binding=highlighted_crop_binding,
        chunk_size=chunk_size,
    )
    highlighted_cache = bundle / "pannuke_resnet18_target_highlighted_embeddings.npz"
    highlighted_metadata = _ensure_cache_publication(
        checkpoint,
        checkpoint_path,
        "highlighted_embeddings",
        highlighted_cache,
        lambda: save_embedding_cache(
            highlighted_cache,
            highlighted.embeddings,
            highlighted.sample_ids,
            highlighted.metadata,
        ),
        manifest_sha256=manifest_sha,
        raw_inventory_sha256=raw_inventory_sha,
        sample_order_sha256=sample_order_sha,
        representation_id="imagenet_target_highlighted_embeddings",
    )
    highlighted = replace(
        highlighted,
        metadata=_runtime_embedding_metadata(highlighted_metadata),
        cache_path=highlighted_cache,
        metadata_path=_sidecar_path(highlighted_cache),
    )

    context: EmbeddingResult | None = None
    context_maps: dict[str, NDArray[np.generic]] = {}
    context_morphometrics: ContextMorphometricsCache | None = None
    if include_context_embeddings:
        context_crop_binding = _embedding_crop_cache_binding(
            crops, crop_cache, input_variant="context_rgb"
        )
        context_result, context_maps = _ensure_embedding_chunks(
            "context_embeddings",
            crops,
            bounds,
            state,
            checkpoint,
            checkpoint_path,
            config=replace(encoder_settings, input_variant="rgb"),
            manifest_sha256=manifest_sha,
            raw_inventory_sha256=raw_inventory_sha,
            representation_id="imagenet_resnet18_context_embeddings",
            source_crop_binding=context_crop_binding,
            chunk_size=chunk_size,
        )
        context_cache = bundle / "pannuke_resnet18_context_rgb_embeddings.npz"
        context_metadata = _ensure_cache_publication(
            checkpoint,
            checkpoint_path,
            "context_embeddings",
            context_cache,
            lambda: save_embedding_cache(
                context_cache,
                context_result.embeddings,
                context_result.sample_ids,
                context_result.metadata,
            ),
            manifest_sha256=manifest_sha,
            raw_inventory_sha256=raw_inventory_sha,
            sample_order_sha256=sample_order_sha,
            representation_id="imagenet_resnet18_context_embeddings",
        )
        context = replace(
            context_result,
            metadata=_runtime_embedding_metadata(context_metadata),
            cache_path=context_cache,
            metadata_path=_sidecar_path(context_cache),
        )
        if context.metadata.get("weight_sha256") != highlighted.metadata.get("weight_sha256"):
            raise RuntimeError("context and target-highlighted embeddings use different weights")
        morph_cache = bundle / "pannuke_resnet18_context_plus_target_morphometrics.npz"
        morph_sidecar = _sidecar_path(morph_cache)
        if morph_cache.exists() != morph_sidecar.exists():
            raise RuntimeError("partial context+morphometrics pair exists in resume workspace")
        if not morph_cache.exists():
            context_morphometrics = save_context_morphometrics_cache(
                context,
                engineered,
                crops.sample_ids,
                morph_cache,
                manifest_sha256=manifest_sha,
                raw_inventory_sha256=raw_inventory_sha,
                analysis_eligibility=crops.metadata["analysis_eligibility"],
                engineered_cache_binding=engineered_exact_binding,
            )
        morph_metadata = _ensure_cache_publication(
            checkpoint,
            checkpoint_path,
            "context_morphometrics",
            morph_cache,
            lambda: (_ for _ in ()).throw(
                RuntimeError("context+morphometrics publisher unexpectedly missing")
            ),
            manifest_sha256=manifest_sha,
            raw_inventory_sha256=raw_inventory_sha,
            sample_order_sha256=sample_order_sha,
            representation_id="imagenet_context_embeddings_plus_target_morphometrics",
        )
        if context_morphometrics is None:
            context_morphometrics = _load_context_morphometrics(morph_cache, morph_metadata)
        else:
            context_morphometrics = replace(
                context_morphometrics,
                metadata=morph_metadata,
            )

    expected_publications = {
        "crops",
        "engineered",
        "highlighted_embeddings",
    }
    if include_context_embeddings:
        expected_publications.update({"context_embeddings", "context_morphometrics"})
    if not expected_publications.issubset(checkpoint["publications"]):
        raise RuntimeError("chunked bundle is not complete enough for publication")
    checkpoint["state"] = "ready_for_atomic_publication"
    _checkpoint_update(checkpoint_path, checkpoint)

    if sha256_file(source_manifest) != manifest_sha:
        raise PanNukeSemanticsError("PanNuke manifest changed before bundle publication")
    verify_raw_inventory_unchanged(validation)
    ensure_derived_output_outside_raw(
        output,
        validation.root,
        purpose="PanNuke representation output directory",
    )
    if os.path.lexists(output):
        raise FileExistsError(f"representation output directory already exists: {output}")

    _close_maps(fixed_maps, variable_maps, engineered_maps, highlighted_maps, context_maps)
    publications: list[PublishedPath] = []
    try:
        publications = publish_flat_directory_no_overwrite(bundle, output)
        if sha256_file(source_manifest) != manifest_sha:
            raise PanNukeSemanticsError("PanNuke manifest changed after bundle publication")
        verify_raw_inventory_unchanged(validation)
        ensure_derived_output_outside_raw(
            output,
            validation.root,
            purpose="PanNuke representation output directory",
        )
    except BaseException as error:
        if publications:
            try:
                rollback_owned_publications(publications)
            except BaseException as rollback_error:
                raise RuntimeError(
                    "chunked publication failed and ownership-safe rollback was incomplete: "
                    f"{rollback_error}"
                ) from error
        raise

    published_state = state
    live_maps: list[np.memmap[Any, Any]] = []
    try:
        published_fixed = _open_fixed_maps(
            published_state / "arrays", fixed_specs, allow_create=False
        )
        live_maps.extend(published_fixed.values())
        consolidation = cast(dict[str, Any], checkpoint["crop_consolidation"])
        component_count = int(consolidation["component_count"])
        contour_count = int(consolidation["contour_point_count"])
        variable_specs = {
            "projected_component_pixel_counts": {
                "shape": [component_count],
                "dtype": np.dtype(np.int32).str,
            },
            "projected_component_unique_pixel_counts": {
                "shape": [component_count],
                "dtype": np.dtype(np.int32).str,
            },
            "baseline_projected_component_counts": {
                "shape": [component_count],
                "dtype": np.dtype(np.int32).str,
            },
            "projection_fallback_component_flags": {
                "shape": [component_count],
                "dtype": np.dtype(np.bool_).str,
            },
            "projected_component_offsets": {
                "shape": [len(identifiers) + 1],
                "dtype": np.dtype(np.int64).str,
            },
            "source_contour_xy": {
                "shape": [contour_count, 2],
                "dtype": np.dtype(np.int32).str,
            },
            "source_contour_offsets": {
                "shape": [len(identifiers) + 1],
                "dtype": np.dtype(np.int64).str,
            },
        }
        published_variable = _open_fixed_maps(
            published_state / "arrays", variable_specs, allow_create=False
        )
        live_maps.extend(published_variable.values())
        published_crop_metadata = _read_json_object(published_state / "crop_metadata.json")
        published_crops = _make_crop_batch(
            published_fixed, published_variable, published_crop_metadata, validation
        )
        engineered_state = _read_json_object(published_state / "engineered_state.json")
        published_engineered_map = _open_map(
            published_state / "arrays" / "engineered_values.npy",
            {"shape": engineered_state["shape"], "dtype": engineered_state["dtype"]},
            allow_create=False,
        )
        live_maps.append(published_engineered_map)
        published_engineered = EngineeredFeatureSet(
            values=cast(Any, published_engineered_map),
            names=tuple(str(value) for value in engineered_state["names"]),
        )

        def published_embedding(stage: str, cache_name: str) -> EmbeddingResult:
            embedding_state = _read_json_object(
                published_state / "embedding_chunks" / stage / "state.json"
            )
            mapped = _open_map(
                published_state / "arrays" / f"{stage}_embeddings.npy",
                {"shape": embedding_state["shape"], "dtype": embedding_state["dtype"]},
                allow_create=False,
            )
            live_maps.append(mapped)
            cache = output / cache_name
            metadata = verify_frozen_cache_sidecar(cache).metadata
            return EmbeddingResult(
                embeddings=cast(Any, mapped),
                sample_ids=published_crops.sample_ids,
                metadata=_runtime_embedding_metadata(metadata),
                cache_path=cache,
                metadata_path=_sidecar_path(cache),
            )

        published_highlighted = published_embedding(
            "highlighted_embeddings", "pannuke_resnet18_target_highlighted_embeddings.npz"
        )
        published_context = (
            published_embedding("context_embeddings", "pannuke_resnet18_context_rgb_embeddings.npz")
            if include_context_embeddings
            else None
        )
        if include_context_embeddings:
            published_morph_cache = (
                output / "pannuke_resnet18_context_plus_target_morphometrics.npz"
            )
            published_morph = _load_context_morphometrics(
                published_morph_cache,
                verify_frozen_cache_sidecar(published_morph_cache).metadata,
            )
        else:
            published_morph = None
        artifacts = PanNukeRepresentationArtifacts(
            crops=published_crops,
            engineered=published_engineered,
            embeddings=published_highlighted,
            crop_cache_path=output / "pannuke_crops.npz",
            crop_metadata_path=_sidecar_path(output / "pannuke_crops.npz"),
            engineered_cache_path=output / "pannuke_engineered_features.npz",
            engineered_metadata_path=_sidecar_path(output / "pannuke_engineered_features.npz"),
            context_embeddings=published_context,
            context_morphometrics=published_morph,
            publication_records=tuple(publications),
        )
        ensure_derived_output_outside_raw(
            output,
            validation.root,
            purpose="PanNuke representation output directory",
        )
        with _WORKSPACE_LOCK:
            if output.resolve() in _LIVE_WORKSPACES:
                raise RuntimeError(f"chunked workspace lease already exists for output: {output}")
            _LIVE_WORKSPACES[output.resolve()] = _ChunkedWorkspaceLease(
                workspace=resume,
                maps=tuple(live_maps),
                publications=tuple(publications),
            )
        return artifacts
    except BaseException as error:
        _close_maps({str(index): mapped for index, mapped in enumerate(live_maps)})
        try:
            rollback_owned_publications(publications)
        except BaseException as rollback_error:
            raise RuntimeError(
                "chunked post-publication readback failed and ownership-safe rollback was "
                f"incomplete: {rollback_error}"
            ) from error
        raise


__all__ = [
    "build_pannuke_representation_cache_chunked",
    "cleanup_pannuke_chunked_workspace",
    "rollback_pannuke_chunked_publication",
]
