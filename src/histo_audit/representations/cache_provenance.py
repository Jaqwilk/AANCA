"""Strict provenance and atomic persistence for frozen representation caches.

The confirmatory study may bind either the cache file itself or a semantic
sidecar digest.  The latter is useful only when it binds the exact arrays, so
the semantic projection below includes a canonical digest of every NPZ array.
Operational values such as wall-clock extraction time are deliberately outside
that projection; identical inputs and recipes therefore have identical
semantic provenance even when extracted in separate runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from histo_audit.utils.run_tracking import sha256_file

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SEMANTIC_FIELDS = (
    "representation_id",
    "status",
    "sample_order_sha256",
    "manifest_sha256",
    "raw_inventory_sha256",
    "encoder_identifier",
    "encoder_metadata_sha256",
    "encoder_implementation_sha256",
    "weight_identifier",
    "weights_sha256",
    "preprocessing_identifier",
    "preprocessing_sha256",
    "input_variant",
    "cache_recipe_sha256",
    "cache_content_sha256",
    "dtype",
    "feature_dimension",
    "sample_count",
    "package_versions",
)


@dataclass(frozen=True, slots=True)
class FrozenCacheVerification:
    """Verified cache and sidecar digests plus the parsed provenance."""

    cache_path: Path
    sidecar_path: Path
    cache_file_sha256: str
    sidecar_file_sha256: str
    sidecar_semantic_sha256: str
    metadata: dict[str, Any]


def canonical_sha256(payload: object) -> str:
    """Hash JSON-compatible data with the project's canonical JSON rule."""

    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ordered_sample_ids_sha256(sample_ids: NDArray[np.str_] | list[str] | tuple[str, ...]) -> str:
    """Hash an exact, non-empty, unique sample order canonically."""

    identifiers = [str(value) for value in np.asarray(sample_ids, dtype=np.str_).tolist()]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("sample IDs must be non-empty and unique")
    if any(not value.strip() for value in identifiers):
        raise ValueError("sample IDs cannot contain empty values")
    return canonical_sha256(identifiers)


def array_artifact_sha256(array: NDArray[np.generic]) -> str:
    """Hash one array's dtype, shape, and C-order bytes without coercion."""

    value = np.ascontiguousarray(np.asarray(array))
    if value.dtype.hasobject:
        raise ValueError("cache arrays cannot use object/pickle-dependent dtypes")
    header = json.dumps(
        {"dtype": value.dtype.str, "shape": list(value.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def explicit_unlearned_weights_sha256(weight_identifier: str) -> str:
    """Return a deterministic hash for an explicit no-learned-weights declaration."""

    if not weight_identifier.strip() or "unlearned" not in weight_identifier.casefold():
        raise ValueError("an explicit unlearned weight identifier is required")
    return canonical_sha256(
        {
            "learned_weights": False,
            "weight_identifier": weight_identifier,
        }
    )


def _require_sha256(value: object, name: str) -> str:
    normalised = str(value).casefold()
    if _SHA256.fullmatch(normalised) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return normalised


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def build_frozen_cache_metadata(
    *,
    base_metadata: Mapping[str, Any],
    sample_ids: NDArray[np.str_],
    manifest_sha256: str,
    raw_inventory_sha256: str,
    representation_id: str,
    input_variant: str,
    encoder_identifier: str,
    encoder_metadata: Mapping[str, Any],
    encoder_implementation: Mapping[str, Any],
    weight_identifier: str,
    weights_sha256: str,
    preprocessing_identifier: str,
    preprocessing: Mapping[str, Any],
    cache_recipe: Mapping[str, Any],
    dtype: str,
    feature_dimension: int | list[int],
    package_versions: Mapping[str, str],
    matrix_key: str,
    provenance_scope: str,
) -> dict[str, Any]:
    """Add the final frozen-cache contract fields to producer metadata."""

    identifiers = np.asarray(sample_ids, dtype=np.str_)
    sample_order_sha = ordered_sample_ids_sha256(identifiers)
    manifest_sha = _require_sha256(manifest_sha256, "manifest_sha256")
    inventory_sha = _require_sha256(raw_inventory_sha256, "raw_inventory_sha256")
    weights_sha = _require_sha256(weights_sha256, "weights_sha256")
    explicit_strings = {
        "representation_id": representation_id,
        "input_variant": input_variant,
        "encoder_identifier": encoder_identifier,
        "weight_identifier": weight_identifier,
        "preprocessing_identifier": preprocessing_identifier,
        "dtype": dtype,
        "matrix_key": matrix_key,
        "provenance_scope": provenance_scope,
    }
    if any(not str(value).strip() for value in explicit_strings.values()):
        raise ValueError("frozen cache provenance identifiers must be explicit")
    if isinstance(feature_dimension, int):
        if feature_dimension <= 0:
            raise ValueError("feature_dimension must be positive")
    elif not feature_dimension or any(int(value) <= 0 for value in feature_dimension):
        raise ValueError("feature_dimension axes must be positive")
    versions = {str(key): str(value) for key, value in sorted(package_versions.items())}
    if not versions:
        raise ValueError("package_versions cannot be empty")

    extraction_timestamp = str(base_metadata.get("extracted_at_utc") or _utc_now())
    metadata = dict(base_metadata)
    metadata.update(
        {
            "schema_version": int(metadata.get("schema_version", 1)),
            "provenance_schema_version": 1,
            "status": "available",
            "representation_id": representation_id,
            "sample_order_sha256": sample_order_sha,
            "manifest_sha256": manifest_sha,
            "dataset_manifest_sha256": manifest_sha,
            "raw_inventory_sha256": inventory_sha,
            "encoder_identifier": encoder_identifier,
            "encoder_id": encoder_identifier,
            "encoder_metadata": dict(encoder_metadata),
            "encoder_metadata_sha256": canonical_sha256(dict(encoder_metadata)),
            "encoder_implementation": dict(encoder_implementation),
            "encoder_implementation_sha256": canonical_sha256(dict(encoder_implementation)),
            "weight_identifier": weight_identifier,
            "weights_sha256": weights_sha,
            "weight_sha256": weights_sha,
            "preprocessing_identifier": preprocessing_identifier,
            "preprocessing": dict(preprocessing),
            "preprocessing_sha256": canonical_sha256(dict(preprocessing)),
            "input_variant": input_variant,
            "cache_recipe": dict(cache_recipe),
            "cache_recipe_sha256": canonical_sha256(dict(cache_recipe)),
            "dtype": dtype,
            "feature_dimension": feature_dimension,
            "dimension": feature_dimension,
            "sample_count": len(identifiers),
            "package_versions": versions,
            "versions": versions,
            "matrix_key": matrix_key,
            "provenance_scope": provenance_scope,
            "extracted_at_utc": extraction_timestamp,
            "extraction_timestamp_utc": extraction_timestamp,
        }
    )
    if isinstance(feature_dimension, int):
        metadata["output_dimension"] = feature_dimension
    return metadata


def _sidecar_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(f"{cache_path.suffix}.metadata.json")


def _semantic_payload(metadata: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in _SEMANTIC_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"frozen cache provenance fields are absent: {missing}")
    return {field: metadata[field] for field in _SEMANTIC_FIELDS}


def _strict_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _temporary_file(directory: Path, name: str, suffix: str) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}.", suffix=suffix, dir=directory)
    os.close(descriptor)
    return Path(temporary_name)


@dataclass(frozen=True, slots=True)
class _PublishedCacheFile:
    path: Path
    identity: tuple[int, int, int, int]
    sha256: str

    def still_owned(self) -> bool:
        try:
            value = self.path.stat(follow_symlinks=False)
            identity = (value.st_dev, value.st_ino, value.st_size, value.st_ctime_ns)
            return (
                self.path.is_file()
                and identity[:3] == self.identity[:3]
                and sha256_file(self.path) == self.sha256
            )
        except OSError:
            return False


def _commit_no_overwrite(temporary: Path, destination: Path) -> _PublishedCacheFile:
    """Atomically publish a staged file while refusing an existing destination."""

    anticipated = temporary.stat(follow_symlinks=False)
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite frozen cache artifact: {destination}"
        ) from error
    except OSError:
        # Windows rename is atomic and, unlike os.replace, refuses an existing target.
        if os.name != "nt":
            raise
        try:
            os.rename(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite frozen cache artifact: {destination}"
            ) from error
    else:
        temporary.unlink()
    try:
        value = destination.stat(follow_symlinks=False)
        published = _PublishedCacheFile(
            path=destination,
            identity=(value.st_dev, value.st_ino, value.st_size, value.st_ctime_ns),
            sha256=sha256_file(destination),
        )
        if not published.still_owned():
            raise OSError(f"frozen cache publication readback failed: {destination}")
        return published
    except BaseException as readback_error:
        if os.path.lexists(destination):
            try:
                current = destination.stat(follow_symlinks=False)
                same_file = (
                    current.st_dev,
                    current.st_ino,
                    current.st_size,
                ) == (anticipated.st_dev, anticipated.st_ino, anticipated.st_size)
            except OSError:
                same_file = False
            if same_file:
                destination.unlink()
            else:
                raise RuntimeError(
                    "frozen-cache readback failed after destination ownership changed; "
                    "foreign destination preserved"
                ) from readback_error
        raise


def atomic_save_npz_with_sidecar(
    cache_path: str | Path,
    *,
    arrays: Mapping[str, NDArray[np.generic]],
    metadata: Mapping[str, Any],
    pre_publish_check: Callable[[], None] | None = None,
    post_publish_check: Callable[[], None] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Stage and atomically publish one NPZ/sidecar pair without overwrite.

    Optional source-freshness checks run immediately before and after both files
    are committed.  A failing post-check rolls back only files whose exact
    publication identities are still owned by this call.
    """

    supplied = Path(cache_path).expanduser()
    if not supplied.is_absolute():
        supplied = Path.cwd() / supplied
    destination = supplied.parent.resolve() / supplied.name
    if destination.suffix.casefold() != ".npz":
        raise ValueError("frozen cache path must end in .npz")
    sidecar = _sidecar_path(destination)
    if os.path.lexists(destination) or os.path.lexists(sidecar):
        existing = destination if os.path.lexists(destination) else sidecar
        raise FileExistsError(f"refusing to overwrite frozen cache artifact: {existing}")
    if not arrays or "sample_ids" not in arrays:
        raise ValueError("frozen cache requires a sample_ids array")
    safe_arrays: dict[str, NDArray[np.generic]] = {}
    array_hashes: dict[str, str] = {}
    for name, raw in sorted(arrays.items()):
        value = np.asarray(raw)
        if value.dtype.hasobject:
            raise ValueError(f"cache array {name!r} uses an object/pickle-dependent dtype")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"cache array {name!r} contains non-finite values")
        safe_arrays[name] = value
        array_hashes[name] = array_artifact_sha256(value)

    destination.parent.mkdir(parents=True, exist_ok=True)
    cache_temporary = _temporary_file(destination.parent, destination.name, ".npz")
    sidecar_temporary = _temporary_file(destination.parent, sidecar.name, ".json")
    publications: list[_PublishedCacheFile] = []
    try:
        np.savez_compressed(cache_temporary, **cast(Any, safe_arrays))
        with cache_temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        cache_sha = sha256_file(cache_temporary)
        complete = dict(metadata)
        complete.update(
            {
                "cache_array_sha256_by_name": array_hashes,
                "cache_content_sha256": canonical_sha256(array_hashes),
                "cache_file_sha256": cache_sha,
                "cache_npz_sha256": cache_sha,
            }
        )
        if complete.get("provenance_scope") == "stage_eligible":
            complete["primary_cache_provenance"] = primary_cache_provenance_record(complete)
        complete["sidecar_semantic_sha256"] = canonical_sha256(_semantic_payload(complete))
        payload = _strict_json_bytes(complete)
        with sidecar_temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if pre_publish_check is not None:
            pre_publish_check()
        publications.append(_commit_no_overwrite(cache_temporary, destination))
        publications.append(_commit_no_overwrite(sidecar_temporary, sidecar))
        if post_publish_check is not None:
            post_publish_check()
    except BaseException as publish_error:
        cache_temporary.unlink(missing_ok=True)
        sidecar_temporary.unlink(missing_ok=True)
        for publication in reversed(publications):
            if not os.path.lexists(publication.path):
                continue
            if not publication.still_owned():
                raise RuntimeError(
                    "frozen-cache publication failed after destination ownership changed; "
                    "the foreign destination was preserved"
                ) from publish_error
            publication.path.unlink()
        raise
    return destination, sidecar, complete


def _read_strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cache sidecar is not strict finite JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("cache sidecar must contain a JSON object")
    return value


def verify_frozen_cache_sidecar(
    cache_path: str | Path,
    *,
    expected_weights_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_representation_id: str | None = None,
    expected_input_variant: str | None = None,
) -> FrozenCacheVerification:
    """Verify sidecar self-consistency, NPZ content, order, and frozen bindings."""

    source = Path(cache_path).resolve()
    sidecar = _sidecar_path(source)
    if not source.is_file() or not sidecar.is_file():
        raise FileNotFoundError("frozen NPZ cache and metadata sidecar must both exist")
    metadata = _read_strict_json(sidecar)
    cache_sha = sha256_file(source)
    if _require_sha256(metadata.get("cache_file_sha256"), "cache_file_sha256") != cache_sha:
        raise ValueError("cache file checksum differs from sidecar provenance")
    if _require_sha256(metadata.get("cache_npz_sha256"), "cache_npz_sha256") != cache_sha:
        raise ValueError("legacy cache NPZ checksum differs from final cache checksum")
    semantic_sha = canonical_sha256(_semantic_payload(metadata))
    if (
        _require_sha256(metadata.get("sidecar_semantic_sha256"), "sidecar_semantic_sha256")
        != semantic_sha
    ):
        raise ValueError("sidecar semantic checksum is not self-consistent")
    for field in (
        "sample_order_sha256",
        "manifest_sha256",
        "raw_inventory_sha256",
        "encoder_metadata_sha256",
        "encoder_implementation_sha256",
        "weights_sha256",
        "preprocessing_sha256",
        "cache_recipe_sha256",
        "cache_content_sha256",
    ):
        _require_sha256(metadata.get(field), field)
    for value_field, hash_field in (
        ("encoder_metadata", "encoder_metadata_sha256"),
        ("encoder_implementation", "encoder_implementation_sha256"),
        ("preprocessing", "preprocessing_sha256"),
        ("cache_recipe", "cache_recipe_sha256"),
    ):
        value = metadata.get(value_field)
        if not isinstance(value, dict) or canonical_sha256(value) != metadata[hash_field]:
            raise ValueError(f"{value_field} differs from its declared SHA-256")
    if metadata.get("weight_sha256") != metadata.get("weights_sha256"):
        raise ValueError("legacy and final weight SHA-256 fields differ")
    if metadata.get("dataset_manifest_sha256") != metadata.get("manifest_sha256"):
        raise ValueError("legacy and final manifest SHA-256 fields differ")
    if metadata.get("encoder_id") != metadata.get("encoder_identifier"):
        raise ValueError("legacy and final encoder identifier fields differ")
    if metadata.get("status") != "available" or metadata.get("provenance_schema_version") != 1:
        raise ValueError("frozen cache provenance status/schema is invalid")
    for field in (
        "representation_id",
        "input_variant",
        "encoder_identifier",
        "weight_identifier",
        "preprocessing_identifier",
        "dtype",
        "extracted_at_utc",
    ):
        if not str(metadata.get(field, "")).strip():
            raise ValueError(f"cache provenance field {field} must be explicit")
    if not isinstance(metadata.get("package_versions"), dict) or not metadata["package_versions"]:
        raise ValueError("cache package_versions must be a non-empty mapping")
    if (
        metadata.get("provenance_scope") == "stage_eligible"
        and "crop_manifest_sha256" in metadata
        and metadata["crop_manifest_sha256"] != metadata["manifest_sha256"]
    ):
        raise ValueError("stage-eligible crop/final manifest SHA-256 fields differ")

    actual_hashes: dict[str, str] = {}
    identifiers_raw: NDArray[np.generic] | None = None
    matrix_shape: tuple[int, ...] | None = None
    matrix_dtype: str | None = None
    matrix_key = str(metadata.get("matrix_key", ""))
    try:
        with np.load(source, allow_pickle=False) as payload:
            for name in sorted(payload.files):
                value = np.asarray(payload[name])
                if value.dtype.hasobject:
                    raise ValueError(f"cache array {name!r} uses object/pickle-dependent dtype")
                if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                    raise ValueError(f"cache array {name!r} contains non-finite values")
                actual_hashes[name] = array_artifact_sha256(value)
                if name == "sample_ids":
                    identifiers_raw = value.copy()
                if name == matrix_key:
                    matrix_shape = tuple(int(axis) for axis in value.shape)
                    matrix_dtype = str(value.dtype)
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, ValueError) and str(error).startswith("cache array"):
            raise
        raise ValueError("frozen cache cannot be opened without pickle") from error
    if actual_hashes != metadata.get("cache_array_sha256_by_name"):
        raise ValueError("cache arrays differ from sidecar content hashes")
    if canonical_sha256(actual_hashes) != metadata.get("cache_content_sha256"):
        raise ValueError("cache content checksum is not self-consistent")
    if (
        identifiers_raw is None
        or identifiers_raw.ndim != 1
        or identifiers_raw.dtype.kind
        not in {
            "U",
            "S",
        }
    ):
        raise ValueError("cache sample_ids must be a one-dimensional non-object string array")
    identifiers = np.asarray(identifiers_raw, dtype=np.str_)
    if ordered_sample_ids_sha256(identifiers) != metadata.get("sample_order_sha256"):
        raise ValueError("cache sample order differs from canonical sidecar hash")
    if int(metadata.get("sample_count", -1)) != len(identifiers):
        raise ValueError("cache sample count differs from sidecar")
    if matrix_shape is None or not matrix_shape or matrix_shape[0] != len(identifiers):
        raise ValueError("declared cache matrix is absent or misaligned")
    expected_dimension: int | list[int] = (
        matrix_shape[1] if len(matrix_shape) == 2 else list(matrix_shape[1:])
    )
    if metadata.get("feature_dimension") != expected_dimension:
        raise ValueError("cache feature dimension differs from sidecar")
    if matrix_dtype != metadata.get("dtype"):
        raise ValueError("cache matrix dtype differs from sidecar")
    if metadata.get("dimension") != expected_dimension:
        raise ValueError("cache dimension alias differs from final feature dimension")
    if len(matrix_shape) == 2 and metadata.get("output_dimension") != expected_dimension:
        raise ValueError("cache output_dimension differs from its matrix")
    if metadata.get("extraction_timestamp_utc") != metadata.get("extracted_at_utc"):
        raise ValueError("cache extraction timestamp aliases differ")

    external_expectations = (
        (expected_weights_sha256, "weights_sha256"),
        (expected_manifest_sha256, "manifest_sha256"),
    )
    for expected, field in external_expectations:
        if (
            expected is not None
            and _require_sha256(expected, f"expected {field}") != metadata[field]
        ):
            raise ValueError(f"cache {field} differs from the expected frozen binding")
    for expected, field in (
        (expected_representation_id, "representation_id"),
        (expected_input_variant, "input_variant"),
    ):
        if expected is not None and expected != metadata.get(field):
            raise ValueError(f"cache {field} differs from the expected frozen binding")
    return FrozenCacheVerification(
        cache_path=source,
        sidecar_path=sidecar,
        cache_file_sha256=cache_sha,
        sidecar_file_sha256=sha256_file(sidecar),
        sidecar_semantic_sha256=semantic_sha,
        metadata=metadata,
    )


def confirmatory_cache_provenance_record(
    metadata: Mapping[str, Any],
    *,
    record_id: str,
    bind_sidecar_semantics: bool = False,
) -> dict[str, Any]:
    """Project a producer sidecar into the exact frozen confirmatory schema."""

    if metadata.get("provenance_scope") != "stage_eligible":
        raise ValueError("only stage-eligible cache provenance may enter a frozen study config")
    cache_sha = _require_sha256(metadata.get("cache_file_sha256"), "cache_file_sha256")
    semantic_sha = _require_sha256(
        metadata.get("sidecar_semantic_sha256"), "sidecar_semantic_sha256"
    )
    return {
        "id": record_id,
        "representation_id": str(metadata["representation_id"]),
        "status": "available",
        "cache_file_sha256": None if bind_sidecar_semantics else cache_sha,
        "sidecar_semantic_sha256": semantic_sha if bind_sidecar_semantics else None,
        "sample_order_sha256": _require_sha256(
            metadata.get("sample_order_sha256"), "sample_order_sha256"
        ),
        "manifest_sha256": _require_sha256(metadata.get("manifest_sha256"), "manifest_sha256"),
        "encoder_identifier": str(metadata["encoder_identifier"]),
        "encoder_metadata_sha256": _require_sha256(
            metadata.get("encoder_metadata_sha256"), "encoder_metadata_sha256"
        ),
        "weight_identifier": str(metadata["weight_identifier"]),
        "weights_sha256": _require_sha256(metadata.get("weights_sha256"), "weights_sha256"),
        "preprocessing_identifier": str(metadata["preprocessing_identifier"]),
        "preprocessing_sha256": _require_sha256(
            metadata.get("preprocessing_sha256"), "preprocessing_sha256"
        ),
        "input_variant": str(metadata.get("contract_input_variant", metadata["input_variant"])),
    }


def primary_cache_provenance_record(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Project a producer sidecar into the exact frozen primary cache schema."""

    if metadata.get("provenance_scope") != "stage_eligible":
        raise ValueError("only stage-eligible cache provenance may enter a frozen study config")
    return {
        "status": "available",
        "encoder_id": str(metadata["encoder_identifier"]),
        "encoder_implementation_sha256": _require_sha256(
            metadata.get("encoder_implementation_sha256"),
            "encoder_implementation_sha256",
        ),
        "weights_sha256": _require_sha256(metadata.get("weights_sha256"), "weights_sha256"),
        "preprocessing_sha256": _require_sha256(
            metadata.get("preprocessing_sha256"), "preprocessing_sha256"
        ),
        "sample_order_sha256": _require_sha256(
            metadata.get("sample_order_sha256"), "sample_order_sha256"
        ),
        "dataset_manifest_sha256": _require_sha256(
            metadata.get("manifest_sha256"), "manifest_sha256"
        ),
        "cache_recipe_sha256": _require_sha256(
            metadata.get("cache_recipe_sha256"), "cache_recipe_sha256"
        ),
        "cache_file_sha256": _require_sha256(
            metadata.get("cache_file_sha256"), "cache_file_sha256"
        ),
    }


__all__ = [
    "FrozenCacheVerification",
    "array_artifact_sha256",
    "atomic_save_npz_with_sidecar",
    "build_frozen_cache_metadata",
    "canonical_sha256",
    "confirmatory_cache_provenance_record",
    "explicit_unlearned_weights_sha256",
    "ordered_sample_ids_sha256",
    "primary_cache_provenance_record",
    "verify_frozen_cache_sidecar",
]
