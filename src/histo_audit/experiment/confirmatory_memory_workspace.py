"""Bounded-memory, checksum-bound backing arrays for confirmatory execution.

The production PanNuke caches are compressed NPZ archives.  NumPy cannot memory-map
members of a compressed NPZ, so loading a member materialises the complete array.
This module creates one deterministic, sealed workspace containing exact extracted
NPY members and exposes row-index descriptors which never implement ``__array__``.
Callers must therefore request an explicit bounded gather instead of accidentally
materialising a complete confirmatory partition.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import uuid
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from histo_audit.utils.run_tracking import sha256_file

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARRAY_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_WORKSPACE_RECIPE_ID = "pannuke_confirmatory_shared_memmap_workspace_v1"
_WORKSPACE_RELATIVE_PARENT = Path("artifacts") / "resource_control" / "input_workspaces"
_METADATA_CAPACITY_ALLOWANCE_BYTES = 16 * 1024 * 1024
_DEFAULT_CHUNK_BYTES = 32 * 1024 * 1024
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


class ConfirmatoryMemoryWorkspaceError(RuntimeError):
    """The shared confirmatory backing workspace failed closed."""


@dataclass(frozen=True, slots=True)
class ConfirmatoryWorkspaceArraySpec:
    """One exact compressed-NPZ member authorized for extraction."""

    array_id: str
    source_npz_path: Path
    source_sidecar_path: Path
    expected_source_sha256: str
    expected_source_sidecar_sha256: str
    member_name: str
    expected_dtype: str
    expected_shape: tuple[int, ...]
    expected_array_sha256: str

    def semantic_dict(self) -> dict[str, Any]:
        """Return the path-independent identity used for the workspace key."""

        return {
            "array_id": self.array_id,
            "expected_source_sha256": self.expected_source_sha256,
            "expected_source_sidecar_sha256": self.expected_source_sidecar_sha256,
            "member_name": self.member_name,
            "expected_dtype": np.dtype(self.expected_dtype).str,
            "expected_shape": list(self.expected_shape),
            "expected_array_sha256": self.expected_array_sha256,
        }


@dataclass(frozen=True, slots=True)
class ConfirmatoryWorkspaceIndexSpec:
    """One exact role partition bound by its immutable source-index vector."""

    outer_fold: int
    role: Literal["audit", "reference_validation", "final_reference"]
    source_indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        raw = np.asarray(self.source_indices)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("workspace index vector must be one-dimensional integers")
        values = np.array(raw, dtype=np.int64, order="C", copy=True)
        if (
            not len(values)
            or int(values.min()) < 0
            or len(np.unique(values)) != len(values)
            or (len(values) > 1 and not np.all(values[1:] > values[:-1]))
        ):
            raise ValueError(
                "workspace index vector must be non-empty, unique, and strictly increasing"
            )
        immutable = np.frombuffer(values.tobytes(order="C"), dtype=np.int64)
        immutable.setflags(write=False)
        object.__setattr__(self, "source_indices", immutable)

    @property
    def row_count(self) -> int:
        return len(self.source_indices)

    @property
    def source_indices_sha256(self) -> str:
        return _array_artifact_sha256_chunks(
            self.source_indices.shape,
            self.source_indices.dtype,
            iter((self.source_indices,)),
        )

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "outer_fold": self.outer_fold,
            "role": self.role,
            "row_count": self.row_count,
            "source_indices_sha256": self.source_indices_sha256,
            "ordered_unique": True,
        }


@dataclass(frozen=True, slots=True)
class _ReadOnlyFlags:
    writeable: bool = False
    c_contiguous: bool = True
    f_contiguous: bool = False
    owndata: bool = False


@dataclass(frozen=True, slots=True)
class ReadOnlyBackingArray:
    """One verified read-only NPY memmap and its exact source binding."""

    array_id: str
    path: Path
    values: np.memmap
    file_sha256: str
    raw_array_sha256: str
    source_npz_sha256: str
    source_sidecar_sha256: str
    source_member_name: str
    verified_file_identity: tuple[int, int, int, int, int] | None = None
    all_rows_nonempty: bool = False
    all_finite: bool = False

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.values.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.values.dtype

    @property
    def ndim(self) -> int:
        return self.values.ndim

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes)

    @property
    def flags(self) -> _ReadOnlyFlags:
        return _ReadOnlyFlags()

    def assert_verified_file_identity(self) -> None:
        """Fail closed if the sealed NPY was replaced or normally modified."""

        if self.verified_file_identity is None:
            return
        if _physical_file_identity(self.path) != self.verified_file_identity:
            raise ConfirmatoryMemoryWorkspaceError(
                f"workspace backing {self.array_id!r} changed after verification"
            )


class RowIndexedArray:
    """Immutable logical rows backed by one shared read-only memmap.

    There is intentionally no ``__array__`` method.  The only materialisation API is
    :meth:`gather_rows`, which makes the allocation explicit and can be bounded by
    ``max_rows``.  Partition descriptors reject repeated source rows.  A caller that
    deliberately needs repeated rows in a transient batch must opt in through
    :meth:`select_rows`.
    """

    __slots__ = ("_backing", "_logical_dtype", "_source_indices")

    def __init__(
        self,
        backing: ReadOnlyBackingArray,
        source_indices: NDArray[np.integer[Any]] | Sequence[int],
        *,
        logical_dtype: Any | None = None,
        allow_repeated_indices: bool = False,
    ) -> None:
        raw_indices = np.asarray(source_indices)
        if raw_indices.ndim != 1 or not np.issubdtype(raw_indices.dtype, np.integer):
            raise ValueError("row-index descriptor requires a one-dimensional integer index")
        if (
            len(raw_indices)
            and raw_indices.dtype.kind == "u"
            and int(raw_indices.max()) > np.iinfo(np.int64).max
        ):
            raise OverflowError("row-index descriptor contains an integer above int64")
        exact = np.ascontiguousarray(raw_indices, dtype=np.int64)
        # A bytes owner cannot be made writeable again and disconnects the logical
        # descriptor from the physical index NPY after its typed/hash readback.
        indices = np.frombuffer(exact.tobytes(order="C"), dtype=np.int64)
        if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= backing.shape[0]):
            raise IndexError("row-index descriptor contains an out-of-bounds source row")
        if not allow_repeated_indices and len(np.unique(indices)) != len(indices):
            raise ValueError("partition row-index descriptor contains repeated source rows")
        indices.setflags(write=False)
        dtype = backing.dtype if logical_dtype is None else np.dtype(logical_dtype)
        if dtype.hasobject:
            raise ValueError("row-index descriptor cannot expose an object dtype")
        self._backing = backing
        self._source_indices = indices
        self._logical_dtype = dtype

    @property
    def backing_array_id(self) -> str:
        return self._backing.array_id

    @property
    def backing_file_sha256(self) -> str:
        return self._backing.file_sha256

    @property
    def all_rows_nonempty(self) -> bool:
        """Return the verifier/adapter semantic attestation without exposing backing."""

        return self._backing.all_rows_nonempty

    @property
    def all_finite(self) -> bool:
        """Return the verifier/adapter semantic attestation without exposing backing."""

        return self._backing.all_finite

    @property
    def source_indices(self) -> NDArray[np.int64]:
        view = self._source_indices.view()
        view.setflags(write=False)
        return view

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._source_indices), *self._backing.shape[1:])

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._logical_dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    @property
    def flags(self) -> _ReadOnlyFlags:
        return _ReadOnlyFlags()

    def __len__(self) -> int:
        return len(self._source_indices)

    def _normalise_local_indices(
        self,
        local_indices: slice | NDArray[np.integer[Any]] | Sequence[int],
        *,
        allow_repeated_indices: bool,
    ) -> NDArray[np.int64]:
        if isinstance(local_indices, slice):
            start, stop, step = local_indices.indices(len(self))
            selected = np.arange(start, stop, step, dtype=np.int64)
        else:
            raw = np.asarray(local_indices)
            if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
                raise ValueError("selected rows require a one-dimensional integer index")
            if len(raw) and raw.dtype.kind == "u" and int(raw.max()) > np.iinfo(np.int64).max:
                raise OverflowError("selected row exceeds the supported int64 range")
            selected = np.array(raw, dtype=np.int64, order="C", copy=True)
            selected[selected < 0] += len(self)
            if len(selected) and (int(selected.min()) < 0 or int(selected.max()) >= len(self)):
                raise IndexError("selected row lies outside the logical partition")
        if not allow_repeated_indices and len(np.unique(selected)) != len(selected):
            raise ValueError("repeated selected rows require explicit opt-in")
        selected.setflags(write=False)
        return selected

    def select_rows(
        self,
        local_indices: slice | NDArray[np.integer[Any]] | Sequence[int],
        *,
        allow_repeated_indices: bool = False,
    ) -> RowIndexedArray:
        """Compose a lightweight descriptor while preserving requested row order."""

        selected = self._normalise_local_indices(
            local_indices,
            allow_repeated_indices=allow_repeated_indices,
        )
        return RowIndexedArray(
            self._backing,
            self._source_indices[selected],
            logical_dtype=self._logical_dtype,
            allow_repeated_indices=allow_repeated_indices,
        )

    def gather_rows(
        self,
        local_indices: slice | NDArray[np.integer[Any]] | Sequence[int] | None = None,
        *,
        max_rows: int | None = None,
        allow_repeated_indices: bool = False,
    ) -> NDArray[Any]:
        """Materialise exactly the selected logical rows as one read-only C array."""

        if max_rows is not None and (type(max_rows) is not int or max_rows < 0):
            raise ValueError("explicit row gather maximum must be a non-negative integer")
        if local_indices is None:
            selected_count = len(self)
            if max_rows is not None and selected_count > max_rows:
                raise ValueError("explicit row gather exceeds its declared maximum")
            source_rows = self._source_indices
        else:
            if isinstance(local_indices, slice):
                start, stop, step = local_indices.indices(len(self))
                selected_count = len(range(start, stop, step))
            else:
                raw_local = np.asarray(local_indices)
                if raw_local.ndim != 1:
                    raise ValueError("selected rows require a one-dimensional integer index")
                selected_count = len(raw_local)
            if max_rows is not None and selected_count > max_rows:
                raise ValueError("explicit row gather exceeds its declared maximum")
            selected = self._normalise_local_indices(
                local_indices,
                allow_repeated_indices=allow_repeated_indices,
            )
            source_rows = self._source_indices[selected]
        self._backing.assert_verified_file_identity()
        mapped = np.load(self._backing.path, mmap_mode="r", allow_pickle=False)
        if not isinstance(mapped, np.memmap):
            raise ConfirmatoryMemoryWorkspaceError(
                "row-index backing cannot be reopened as a memmap"
            )
        try:
            self._backing.assert_verified_file_identity()
            gathered = np.asarray(mapped[source_rows])
            if gathered.dtype != self._logical_dtype:
                gathered = gathered.astype(self._logical_dtype, copy=False)
            output = np.ascontiguousarray(gathered)
            self._backing.assert_verified_file_identity()
            output.setflags(write=False)
            return output
        finally:
            mmap_handle = getattr(mapped, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()

    def iter_chunks(self, max_chunk_bytes: int = _DEFAULT_CHUNK_BYTES) -> Iterator[NDArray[Any]]:
        """Yield logical rows in order with a strict approximate byte bound."""

        if type(max_chunk_bytes) is not int or max_chunk_bytes <= 0:
            raise ValueError("chunk byte limit must be positive")
        row_elements = int(np.prod(self.shape[1:], dtype=np.int64))
        backing_row_bytes = max(1, row_elements * self._backing.dtype.itemsize)
        logical_row_bytes = max(1, row_elements * self.dtype.itemsize)
        # A normal ``for chunk in ...`` consumer still references the preceding
        # logical chunk while the generator allocates the next backing gather.
        peak_row_bytes = backing_row_bytes + logical_row_bytes
        if self.dtype != self._backing.dtype:
            # Advanced indexing first creates a backing-dtype copy; astype then
            # temporarily holds that copy alongside both the preceding and next
            # logical-dtype outputs.
            peak_row_bytes += logical_row_bytes
        if peak_row_bytes > max_chunk_bytes:
            raise ValueError("one logical row exceeds the strict chunk byte limit")
        rows_per_chunk = max_chunk_bytes // peak_row_bytes
        for start in range(0, len(self), rows_per_chunk):
            self._backing.assert_verified_file_identity()
            mapped = np.load(self._backing.path, mmap_mode="r", allow_pickle=False)
            if not isinstance(mapped, np.memmap):
                raise ConfirmatoryMemoryWorkspaceError(
                    "row-index backing cannot be reopened as a memmap"
                )
            try:
                stop = min(len(self), start + rows_per_chunk)
                source_rows = self._source_indices[start:stop]
                chunk = np.asarray(mapped[source_rows])
                if chunk.dtype != self._logical_dtype:
                    chunk = chunk.astype(self._logical_dtype, copy=False)
                output = np.ascontiguousarray(chunk)
                self._backing.assert_verified_file_identity()
                output.setflags(write=False)
            finally:
                mmap_handle = getattr(mapped, "_mmap", None)
                if mmap_handle is not None:
                    mmap_handle.close()
            yield output

    def logical_array_sha256(self, max_chunk_bytes: int = _DEFAULT_CHUNK_BYTES) -> str:
        """Hash logical dtype, shape and C-order bytes like cache provenance."""

        return _array_artifact_sha256_chunks(
            self.shape,
            self.dtype,
            self.iter_chunks(max_chunk_bytes),
        )


@dataclass(frozen=True, slots=True)
class ConfirmatoryMemoryWorkspace:
    """Verified immutable collection of shared backing arrays."""

    root: Path
    workspace_key: str
    receipt_sha256: str
    artifact_root_sha256: str
    arrays: Mapping[str, ReadOnlyBackingArray]
    index_arrays: Mapping[str, NDArray[np.int64]]
    resource_input_workspace_plan_sha256: str | None = None
    cleanup_ownership_token: str | None = None

    def close(self) -> None:
        """Deterministically close every backing and index mapping."""

        mappings = [
            *(value.values for value in self.arrays.values()),
            *self.index_arrays.values(),
        ]
        seen: set[int] = set()
        for mapping in mappings:
            mmap_handle = getattr(mapping, "_mmap", None)
            if mmap_handle is None or id(mmap_handle) in seen:
                continue
            seen.add(id(mmap_handle))
            mmap_handle.close()

    def __enter__(self) -> ConfirmatoryMemoryWorkspace:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha(value: str, role: str) -> str:
    normalised = str(value).casefold()
    if _SHA256.fullmatch(normalised) is None:
        raise ValueError(f"{role} must be a lowercase SHA-256")
    return normalised


def _array_artifact_sha256_chunks(
    shape: Sequence[int],
    dtype: np.dtype[Any],
    chunks: Iterator[NDArray[Any]],
) -> str:
    header = _canonical_json_bytes(
        {"dtype": np.dtype(dtype).str, "shape": [int(value) for value in shape]}
    )
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    for chunk in chunks:
        contiguous = np.ascontiguousarray(chunk, dtype=dtype)
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _memmap_array_sha256(
    values: np.memmap,
    *,
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
) -> str:
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("chunk byte limit must be positive")
    if not values.flags.c_contiguous:
        raise ValueError("workspace backing array must be C-contiguous")
    row_elements = int(np.prod(values.shape[1:], dtype=np.int64))
    row_bytes = max(1, row_elements * values.dtype.itemsize)
    if row_bytes > chunk_bytes:
        raise ValueError("one workspace array row exceeds the strict hash chunk limit")
    rows_per_chunk = chunk_bytes // row_bytes
    source_path = Path(str(values.filename))

    def chunks() -> Iterator[NDArray[Any]]:
        for start in range(0, len(values), rows_per_chunk):
            mapped = np.load(source_path, mmap_mode="r", allow_pickle=False)
            if not isinstance(mapped, np.memmap):
                raise ConfirmatoryMemoryWorkspaceError(
                    "workspace NPY cannot be reopened for bounded hashing"
                )
            try:
                yield np.asarray(mapped[start : start + rows_per_chunk])
            finally:
                mmap_handle = getattr(mapped, "_mmap", None)
                if mmap_handle is not None:
                    mmap_handle.close()

    return _array_artifact_sha256_chunks(values.shape, values.dtype, chunks())


def _strict_json_object(path: Path, role: str) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key {key}")
            output[key] = value
        return output

    try:
        if path.stat().st_size > _METADATA_CAPACITY_ALLOWANCE_BYTES:
            raise ConfirmatoryMemoryWorkspaceError(
                f"{role} exceeds the bounded workspace metadata allowance"
            )
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except ConfirmatoryMemoryWorkspaceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ConfirmatoryMemoryWorkspaceError(f"{role} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ConfirmatoryMemoryWorkspaceError(f"{role} must be a JSON object")
    return value


def _is_reparse(path: Path) -> bool:
    try:
        status = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(status.st_mode):
        return True
    if bool(getattr(status, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT):
        return True
    is_junction = getattr(os.path, "isjunction", None)
    return bool(is_junction(path)) if is_junction is not None else False


def _physical_file_identity(path: Path) -> tuple[int, int, int, int, int]:
    """Return stable physical identity/change fields for a verified plain file."""

    try:
        status = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ConfirmatoryMemoryWorkspaceError(
            "verified workspace file is no longer available"
        ) from error
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
    )


def _physical_directory_identity(path: Path) -> tuple[int, int, int]:
    try:
        status = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ConfirmatoryMemoryWorkspaceError(
            "owned workspace directory is no longer available"
        ) from error
    return (int(status.st_dev), int(status.st_ino), int(status.st_ctime_ns))


def _require_plain_existing_file(path: Path, role: str) -> Path:
    lexical = Path(os.path.abspath(path))
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise ConfirmatoryMemoryWorkspaceError(f"{role} must be a plain physical file") from error
    if (
        resolved != lexical
        or not lexical.is_file()
        or _is_reparse(lexical)
        or _is_reparse(resolved)
    ):
        raise ConfirmatoryMemoryWorkspaceError(f"{role} must be a plain physical file")
    for parent in lexical.parents:
        if not parent.is_dir() or _is_reparse(parent):
            raise ConfirmatoryMemoryWorkspaceError(
                f"{role} parent chain must contain only plain physical directories"
            )
    return lexical


def _plain_file_state(
    path: Path,
    role: str,
) -> tuple[
    tuple[int, int, int, int, int],
    tuple[tuple[str, int, int], ...],
]:
    """Capture one exact regular file and every lexical parent's physical identity."""

    def parent_state(exact_path: Path) -> tuple[tuple[str, int, int], ...]:
        output: list[tuple[str, int, int]] = []
        for parent in exact_path.parents:
            identity = _physical_directory_identity(parent)
            output.append((str(parent), identity[0], identity[1]))
        return tuple(output)

    exact = _require_plain_existing_file(path, role)
    parent_identities = parent_state(exact)
    file_identity = _physical_file_identity(exact)
    if _require_plain_existing_file(exact, role) != exact:
        raise ConfirmatoryMemoryWorkspaceError(
            f"{role} changed while its plain-file identity was captured"
        )
    if parent_identities != parent_state(exact):
        raise ConfirmatoryMemoryWorkspaceError(
            f"{role} parent chain changed while its identity was captured"
        )
    if _physical_file_identity(exact) != file_identity:
        raise ConfirmatoryMemoryWorkspaceError(
            f"{role} changed while its plain-file identity was captured"
        )
    return file_identity, parent_identities


def _require_plain_directory_chain(root: Path, target: Path) -> None:
    root_lexical = Path(os.path.abspath(root))
    target_lexical = Path(os.path.abspath(target))
    try:
        lexical_relative = target_lexical.relative_to(root_lexical)
    except ValueError as error:
        raise ConfirmatoryMemoryWorkspaceError(
            "workspace lexical path escapes the project root"
        ) from error
    current_lexical = root_lexical
    if _is_reparse(current_lexical):
        raise ConfirmatoryMemoryWorkspaceError("project root cannot be a reparse point")
    for component in lexical_relative.parts:
        current_lexical = current_lexical / component
        if current_lexical.exists() and (
            not current_lexical.is_dir() or _is_reparse(current_lexical)
        ):
            raise ConfirmatoryMemoryWorkspaceError(
                "workspace directory chain contains a non-directory or reparse point"
            )

    root_resolved = root_lexical.resolve()
    target_resolved = target_lexical.resolve(strict=False)
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ConfirmatoryMemoryWorkspaceError(
            "workspace path escapes the canonical project root"
        ) from error


def _normalise_specs(
    specs: Sequence[ConfirmatoryWorkspaceArraySpec],
) -> tuple[ConfirmatoryWorkspaceArraySpec, ...]:
    if not specs:
        raise ValueError("confirmatory memory workspace requires at least one array")
    output: list[ConfirmatoryWorkspaceArraySpec] = []
    array_ids: set[str] = set()
    source_members: set[tuple[Path, str]] = set()
    source_expectations: dict[Path, str] = {}
    sidecar_expectations: dict[Path, str] = {}
    for raw in specs:
        if _ARRAY_ID.fullmatch(raw.array_id) is None:
            raise ValueError(f"invalid workspace array ID: {raw.array_id!r}")
        if raw.array_id in array_ids:
            raise ValueError(f"duplicate workspace array ID: {raw.array_id}")
        array_ids.add(raw.array_id)
        source = _require_plain_existing_file(raw.source_npz_path, "source NPZ")
        sidecar = _require_plain_existing_file(raw.source_sidecar_path, "source sidecar")
        if source.suffix.casefold() != ".npz":
            raise ValueError("workspace source must be an NPZ file")
        if (
            not raw.member_name.endswith(".npy")
            or "/" in raw.member_name
            or "\\" in raw.member_name
            or raw.member_name in {".npy", "..npy"}
        ):
            raise ValueError("workspace member name must be one plain NPY basename")
        source_member = (source, raw.member_name)
        if source_member in source_members:
            raise ValueError("duplicate workspace source/member pair")
        source_members.add(source_member)
        dtype = np.dtype(raw.expected_dtype)
        if (
            dtype.hasobject
            or dtype.fields is not None
            or dtype.subdtype is not None
            or dtype.kind not in {"S", "U", "b", "i", "u", "f"}
        ):
            raise ValueError(
                "workspace arrays require a fixed-width string, boolean, integer, or floating dtype"
            )
        if not raw.expected_shape or any(
            type(value) is not int or value <= 0 for value in raw.expected_shape
        ):
            raise ValueError("workspace expected shape must contain positive integers")
        expected_source_sha = _require_sha(raw.expected_source_sha256, "source NPZ SHA-256")
        expected_sidecar_sha = _require_sha(
            raw.expected_source_sidecar_sha256,
            "source sidecar SHA-256",
        )
        if source in source_expectations and source_expectations[source] != expected_source_sha:
            raise ValueError("shared source NPZ has conflicting expected SHA-256 values")
        if sidecar in sidecar_expectations and (
            sidecar_expectations[sidecar] != expected_sidecar_sha
        ):
            raise ValueError("shared source sidecar has conflicting expected SHA-256 values")
        source_expectations[source] = expected_source_sha
        sidecar_expectations[sidecar] = expected_sidecar_sha
        output.append(
            ConfirmatoryWorkspaceArraySpec(
                array_id=raw.array_id,
                source_npz_path=source,
                source_sidecar_path=sidecar,
                expected_source_sha256=expected_source_sha,
                expected_source_sidecar_sha256=expected_sidecar_sha,
                member_name=raw.member_name,
                expected_dtype=dtype.str,
                expected_shape=tuple(raw.expected_shape),
                expected_array_sha256=_require_sha(
                    raw.expected_array_sha256, "expected logical array SHA-256"
                ),
            )
        )
    return tuple(sorted(output, key=lambda value: value.array_id))


def _workspace_key(specs: Sequence[ConfirmatoryWorkspaceArraySpec]) -> str:
    return _canonical_sha256(
        {
            "schema_version": 1,
            "recipe_id": _WORKSPACE_RECIPE_ID,
            "arrays": [spec.semantic_dict() for spec in specs],
        }
    )


def _array_filename(array_id: str) -> str:
    """Use a deterministic short basename to remain below Windows path limits."""

    return f"{hashlib.sha256(array_id.encode('utf-8')).hexdigest()[:20]}.npy"


def workspace_index_id(outer_fold: int, role: str) -> str:
    """Return the canonical public key for one physical partition-index NPY."""

    if outer_fold not in {1, 2, 3} or role not in {
        "audit",
        "reference_validation",
        "final_reference",
    }:
        raise ValueError("invalid workspace index fold/role")
    return f"fold_{outer_fold}__{role}"


def _index_filename(value: ConfirmatoryWorkspaceIndexSpec) -> str:
    return f"{workspace_index_id(value.outer_fold, value.role)}.npy"


def _npy_serialized_size(values: NDArray[Any]) -> int:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(values), allow_pickle=False)
    return buffer.tell()


def canonical_confirmatory_memory_workspace_parent(project_root: str | Path) -> Path:
    """Return the only permitted workspace parent below a canonical project root."""

    project = Path(project_root).resolve()
    if not project.is_dir() or _is_reparse(project):
        raise ConfirmatoryMemoryWorkspaceError("canonical project root is unavailable")
    parent = project / _WORKSPACE_RELATIVE_PARENT
    _require_plain_directory_chain(project, parent)
    return parent


def _zip_member_info(spec: ConfirmatoryWorkspaceArraySpec) -> zipfile.ZipInfo:
    try:
        with zipfile.ZipFile(spec.source_npz_path, "r") as archive:
            matches = [item for item in archive.infolist() if item.filename == spec.member_name]
    except (OSError, zipfile.BadZipFile) as error:
        raise ConfirmatoryMemoryWorkspaceError("source NPZ is unavailable or invalid") from error
    if len(matches) != 1:
        raise ConfirmatoryMemoryWorkspaceError(
            f"source NPZ must contain exactly one {spec.member_name!r} member"
        )
    info = matches[0]
    if info.flag_bits & 0x1:
        raise ConfirmatoryMemoryWorkspaceError("encrypted NPZ members are forbidden")
    if info.compress_type not in {zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED}:
        raise ConfirmatoryMemoryWorkspaceError("unsupported NPZ member compression")
    if info.file_size <= 0:
        raise ConfirmatoryMemoryWorkspaceError("NPZ member has an invalid uncompressed size")
    return info


def _normalise_index_specs(
    values: Sequence[ConfirmatoryWorkspaceIndexSpec],
    *,
    source_row_count: int,
) -> tuple[ConfirmatoryWorkspaceIndexSpec, ...]:
    roles = ("audit", "reference_validation", "final_reference")
    expected_keys = {(outer_fold, role) for outer_fold in (1, 2, 3) for role in roles}
    records: dict[tuple[int, str], ConfirmatoryWorkspaceIndexSpec] = {}
    for value in values:
        if (
            type(value.outer_fold) is not int
            or value.outer_fold not in {1, 2, 3}
            or value.role not in roles
        ):
            raise ValueError("workspace index specification has an invalid fold or role")
        key = (value.outer_fold, value.role)
        if key in records:
            raise ValueError("workspace index specifications contain a duplicate fold/role")
        if int(value.source_indices.max()) >= source_row_count:
            raise ValueError("workspace index specification exceeds the source row count")
        records[key] = ConfirmatoryWorkspaceIndexSpec(
            outer_fold=value.outer_fold,
            role=value.role,
            source_indices=value.source_indices,
        )
    if set(records) != expected_keys:
        raise ValueError("workspace plan requires exactly all nine fold/role index specifications")
    for outer_fold in (1, 2, 3):
        combined = np.concatenate([records[(outer_fold, role)].source_indices for role in roles])
        if len(combined) != source_row_count or not np.array_equal(
            np.sort(combined), np.arange(source_row_count)
        ):
            raise ValueError(
                "workspace fold/role indices do not exactly partition the source universe"
            )
    return tuple(records[key] for key in sorted(records))


def build_confirmatory_memory_workspace_plan(
    specs: Sequence[ConfirmatoryWorkspaceArraySpec],
    index_specs: Sequence[ConfirmatoryWorkspaceIndexSpec],
    *,
    minimum_free_bytes_after: int,
    maximum_workspace_bytes: int,
) -> dict[str, Any]:
    """Build the exact typed, outcome-blind carrier bound by authority D."""

    if type(minimum_free_bytes_after) is not int or minimum_free_bytes_after < 0:
        raise ValueError("minimum final free bytes must be a non-negative integer")
    if type(maximum_workspace_bytes) is not int or maximum_workspace_bytes <= 0:
        raise ValueError("maximum workspace bytes must be a positive integer")
    normalised = _normalise_specs(specs)
    source_counts = {spec.expected_shape[0] for spec in normalised}
    if len(source_counts) != 1:
        raise ValueError("workspace arrays disagree on the source row count")
    source_row_count = next(iter(source_counts))
    indices = _normalise_index_specs(index_specs, source_row_count=source_row_count)
    array_records: list[dict[str, Any]] = []
    expected_extracted_file_bytes = 0
    expected_raw_array_nbytes = 0
    for spec in normalised:
        info = _zip_member_info(spec)
        raw_nbytes = (
            int(np.prod(spec.expected_shape, dtype=np.int64))
            * np.dtype(spec.expected_dtype).itemsize
        )
        expected_extracted_file_bytes += int(info.file_size)
        expected_raw_array_nbytes += raw_nbytes
        array_records.append(
            {
                "array_id": spec.array_id,
                "source_npz_sha256": spec.expected_source_sha256,
                "source_sidecar_sha256": spec.expected_source_sidecar_sha256,
                "source_member_name": spec.member_name,
                "source_member_crc32": f"{info.CRC:08x}",
                "source_member_compression": (
                    "deflated" if info.compress_type == zipfile.ZIP_DEFLATED else "stored"
                ),
                "source_member_compressed_bytes": int(info.compress_size),
                "source_member_uncompressed_bytes": int(info.file_size),
                "dtype": np.dtype(spec.expected_dtype).str,
                "shape": list(spec.expected_shape),
                "raw_array_nbytes": raw_nbytes,
                "expected_array_sha256": spec.expected_array_sha256,
            }
        )
    index_file_bytes = sum(_npy_serialized_size(value.source_indices) for value in indices)
    index_raw_nbytes = sum(value.source_indices.nbytes for value in indices)
    planned_workspace_bytes = (
        expected_extracted_file_bytes + index_file_bytes + _METADATA_CAPACITY_ALLOWANCE_BYTES
    )
    if planned_workspace_bytes > maximum_workspace_bytes:
        raise ConfirmatoryMemoryWorkspaceError(
            "exact extracted workspace exceeds its authority-bound byte ceiling"
        )
    base = {
        "schema_version": 1,
        "recipe_id": _WORKSPACE_RECIPE_ID,
        "workspace_reuse_allowed": False,
        "workspace_key": _workspace_key(normalised),
        "source_row_count": source_row_count,
        "arrays": array_records,
        "partition_index_specs": [
            {
                **value.semantic_dict(),
                "relative_path": f"indices/{_index_filename(value)}",
                "raw_nbytes": value.source_indices.nbytes,
                "npy_file_bytes": _npy_serialized_size(value.source_indices),
            }
            for value in indices
        ],
        "expected_extracted_file_bytes": expected_extracted_file_bytes,
        "expected_raw_array_nbytes": expected_raw_array_nbytes,
        "expected_index_npy_file_bytes": index_file_bytes,
        "expected_index_raw_nbytes": index_raw_nbytes,
        "metadata_capacity_allowance_bytes": _METADATA_CAPACITY_ALLOWANCE_BYTES,
        "planned_workspace_bytes": planned_workspace_bytes,
        "minimum_free_bytes_after": minimum_free_bytes_after,
        "maximum_workspace_bytes": maximum_workspace_bytes,
        "required_free_bytes_before": minimum_free_bytes_after + planned_workspace_bytes,
    }
    return {**base, "plan_without_self_hash_sha256": _canonical_sha256(base)}


def validate_confirmatory_memory_workspace_plan(
    plan: Mapping[str, Any],
    specs: Sequence[ConfirmatoryWorkspaceArraySpec],
    index_specs: Sequence[ConfirmatoryWorkspaceIndexSpec],
    *,
    minimum_free_bytes_after: int,
    maximum_workspace_bytes: int,
) -> dict[str, Any]:
    """Rebuild and compare every canonical plan field before use."""

    expected = build_confirmatory_memory_workspace_plan(
        specs,
        index_specs,
        minimum_free_bytes_after=minimum_free_bytes_after,
        maximum_workspace_bytes=maximum_workspace_bytes,
    )
    observed = dict(plan)
    if observed != expected:
        raise ConfirmatoryMemoryWorkspaceError(
            "resource input workspace plan differs from its exact typed reconstruction"
        )
    return expected


def _write_json_fsync(path: Path, value: Any) -> None:
    encoded = _canonical_json_bytes(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _extract_member(
    spec: ConfirmatoryWorkspaceArraySpec,
    destination: Path,
    *,
    chunk_bytes: int,
    expected_info: zipfile.ZipInfo,
) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(spec.source_npz_path, "r") as archive:
            matches = [item for item in archive.infolist() if item.filename == spec.member_name]
            if len(matches) != 1:
                raise ConfirmatoryMemoryWorkspaceError(
                    f"source NPZ must contain exactly one {spec.member_name!r} member"
                )
            info = matches[0]
            if (
                info.filename != expected_info.filename
                or info.CRC != expected_info.CRC
                or info.compress_type != expected_info.compress_type
                or info.compress_size != expected_info.compress_size
                or info.file_size != expected_info.file_size
            ):
                raise ConfirmatoryMemoryWorkspaceError(
                    "NPZ member metadata changed after capacity planning"
                )
            if info.flag_bits & 0x1 or info.compress_type not in {
                zipfile.ZIP_DEFLATED,
                zipfile.ZIP_STORED,
            }:
                raise ConfirmatoryMemoryWorkspaceError("NPZ member security policy failed")
            digest = hashlib.sha256()
            written = 0
            with archive.open(info, "r") as source, destination.open("xb") as output:
                while True:
                    block = source.read(chunk_bytes)
                    if not block:
                        break
                    if written + len(block) > expected_info.file_size:
                        raise ConfirmatoryMemoryWorkspaceError(
                            "NPZ member exceeded its authority-bound extraction byte ceiling"
                        )
                    output.write(block)
                    digest.update(block)
                    written += len(block)
                output.flush()
                os.fsync(output.fileno())
    except ConfirmatoryMemoryWorkspaceError:
        raise
    except (OSError, EOFError, RuntimeError, zipfile.BadZipFile) as error:
        raise ConfirmatoryMemoryWorkspaceError(
            f"failed to stream verified NPZ member {spec.member_name!r}"
        ) from error
    if written != info.file_size:
        raise ConfirmatoryMemoryWorkspaceError("extracted NPY size differs from ZIP metadata")
    try:
        values = np.load(destination, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as error:
        raise ConfirmatoryMemoryWorkspaceError(
            "extracted member is not a safe NPY array"
        ) from error
    if not isinstance(values, np.memmap):
        raise ConfirmatoryMemoryWorkspaceError("extracted NPY did not open as a memmap")
    try:
        if (
            values.dtype.hasobject
            or values.dtype.str != np.dtype(spec.expected_dtype).str
            or tuple(values.shape) != spec.expected_shape
            or not values.flags.c_contiguous
            or values.flags.writeable
        ):
            raise ConfirmatoryMemoryWorkspaceError(
                "extracted NPY dtype/shape/order/writeability differs from its exact specification"
            )
        raw_array_sha = _memmap_array_sha256(values, chunk_bytes=chunk_bytes)
    finally:
        mmap_handle = getattr(values, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()
    if raw_array_sha != spec.expected_array_sha256:
        raise ConfirmatoryMemoryWorkspaceError(
            "extracted NPY logical content differs from its expected array SHA-256"
        )
    return {
        "array_id": spec.array_id,
        "relative_path": f"arrays/{_array_filename(spec.array_id)}",
        "source_npz_path": str(spec.source_npz_path),
        "source_sidecar_path": str(spec.source_sidecar_path),
        "source_npz_sha256": spec.expected_source_sha256,
        "source_sidecar_sha256": spec.expected_source_sidecar_sha256,
        "source_member_name": spec.member_name,
        "source_member_crc32": f"{info.CRC:08x}",
        "source_member_compression": (
            "deflated" if info.compress_type == zipfile.ZIP_DEFLATED else "stored"
        ),
        "source_member_compressed_bytes": int(info.compress_size),
        "source_member_uncompressed_bytes": int(info.file_size),
        "npy_file_sha256": digest.hexdigest(),
        "raw_array_sha256": raw_array_sha,
        "dtype": np.dtype(spec.expected_dtype).str,
        "shape": list(spec.expected_shape),
        "order": "C",
        "nbytes": int(np.prod(spec.expected_shape, dtype=np.int64))
        * np.dtype(spec.expected_dtype).itemsize,
    }


def _source_hashes(
    specs: Sequence[ConfirmatoryWorkspaceArraySpec],
) -> tuple[dict[Path, str], dict[Path, str]]:
    sources = {spec.source_npz_path: spec.expected_source_sha256 for spec in specs}
    sidecars = {spec.source_sidecar_path: spec.expected_source_sidecar_sha256 for spec in specs}
    roles = {
        **{path: "source NPZ" for path in sources},
        **{path: "source sidecar" for path in sidecars},
    }
    paths = tuple(sorted(roles))
    initial = {path: _plain_file_state(path, roles[path]) for path in paths}

    def hash_exact_plain_file(path: Path) -> str:
        before = _plain_file_state(path, roles[path])
        if before != initial[path]:
            raise ConfirmatoryMemoryWorkspaceError(
                "source cache or sidecar changed before its SHA-256 was read"
            )
        digest = sha256_file(path)
        after = _plain_file_state(path, roles[path])
        if after != before:
            raise ConfirmatoryMemoryWorkspaceError(
                "source cache or sidecar changed while its SHA-256 was read"
            )
        return digest

    actual_sources = {path: hash_exact_plain_file(path) for path in sorted(sources)}
    actual_sidecars = {path: hash_exact_plain_file(path) for path in sorted(sidecars)}
    final = {path: _plain_file_state(path, roles[path]) for path in paths}
    if initial != final:
        raise ConfirmatoryMemoryWorkspaceError(
            "source cache or sidecar changed while its SHA-256 was read"
        )
    if actual_sources != sources:
        raise ConfirmatoryMemoryWorkspaceError("source NPZ differs from its expected SHA-256")
    if actual_sidecars != sidecars:
        raise ConfirmatoryMemoryWorkspaceError("source sidecar differs from its expected SHA-256")
    return actual_sources, actual_sidecars


def _write_index_arrays(
    directory: Path,
    values: Sequence[ConfirmatoryWorkspaceIndexSpec],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for value in values:
        filename = _index_filename(value)
        path = directory / filename
        with path.open("xb") as stream:
            np.save(stream, value.source_indices, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        records.append(
            {
                **value.semantic_dict(),
                "relative_path": f"indices/{filename}",
                "raw_nbytes": value.source_indices.nbytes,
                "npy_file_bytes": path.stat().st_size,
                "npy_file_sha256": sha256_file(path),
            }
        )
    return records


def _receipt_without_self_hash(
    *,
    workspace_key: str,
    arrays: Sequence[Mapping[str, Any]],
    indices: Sequence[Mapping[str, Any]],
    capacity: Mapping[str, int],
    resource_input_workspace_plan_sha256: str | None,
    cleanup_ownership_token_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "complete",
        "recipe_id": _WORKSPACE_RECIPE_ID,
        "workspace_key": workspace_key,
        "workspace_reuse_allowed": False,
        "source_annotations_modified": False,
        "scientific_outcomes_read": False,
        "resource_input_workspace_plan_sha256": resource_input_workspace_plan_sha256,
        "arrays": [dict(value) for value in arrays],
        "partition_indices": [dict(value) for value in indices],
        "cleanup_ownership_token_sha256": cleanup_ownership_token_sha256,
        "capacity": dict(capacity),
    }


def _manifest_payload(stage: Path, relative_paths: Sequence[str]) -> dict[str, Any]:
    artifacts = []
    for relative in sorted(relative_paths):
        path = stage / relative
        artifacts.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"schema_version": 1, "artifacts": artifacts}


def _remove_owned_plain_tree(path: Path) -> None:
    """Delete one already-authorized physical tree without following links."""

    if not path.exists():
        return
    if not path.is_dir() or _is_reparse(path):
        raise ConfirmatoryMemoryWorkspaceError("owned cleanup target is not a plain directory")
    for child in list(path.iterdir()):
        if _is_reparse(child):
            raise ConfirmatoryMemoryWorkspaceError(
                "owned cleanup refuses a reparse point inside its target"
            )
        if child.is_dir():
            _remove_owned_plain_tree(child)
        elif child.is_file():
            child.unlink()
        else:
            raise ConfirmatoryMemoryWorkspaceError(
                "owned cleanup encountered a non-file/non-directory entry"
            )
    path.rmdir()


def _require_cleanup_ownership(
    path: Path,
    *,
    workspace_key: str,
    cleanup_ownership_token: str,
    expected_directory_identity: tuple[int, int, int] | None = None,
) -> tuple[int, int, int]:
    """Authenticate a published tree immediately before destructive cleanup."""

    if (
        not path.is_dir()
        or _is_reparse(path)
        or not cleanup_ownership_token
        or not cleanup_ownership_token.isascii()
    ):
        raise ConfirmatoryMemoryWorkspaceError(
            "published workspace cleanup lacks exact physical ownership"
        )
    identity = _physical_directory_identity(path)
    if expected_directory_identity is not None and identity != expected_directory_identity:
        raise ConfirmatoryMemoryWorkspaceError(
            "published workspace directory identity differs from its owned publication"
        )
    receipt_path = path / "workspace_receipt.json"
    resolved_receipt = _require_plain_existing_file(
        receipt_path,
        "workspace cleanup receipt",
    )
    if resolved_receipt != receipt_path.resolve(strict=True):
        raise ConfirmatoryMemoryWorkspaceError(
            "workspace cleanup receipt escapes its exact destination"
        )
    receipt = _strict_json_object(receipt_path, "workspace cleanup receipt")
    token_sha = hashlib.sha256(cleanup_ownership_token.encode("ascii")).hexdigest()
    if (
        receipt.get("status") != "complete"
        or receipt.get("workspace_key") != workspace_key
        or receipt.get("workspace_reuse_allowed") is not False
        or receipt.get("cleanup_ownership_token_sha256") != token_sha
    ):
        raise ConfirmatoryMemoryWorkspaceError(
            "published workspace cleanup token/key readback failed"
        )
    if _physical_directory_identity(path) != identity:
        raise ConfirmatoryMemoryWorkspaceError(
            "published workspace changed during cleanup authorization"
        )
    return identity


def _remove_authenticated_workspace_tree(
    path: Path,
    *,
    workspace_key: str,
    cleanup_ownership_token: str,
    expected_directory_identity: tuple[int, int, int] | None = None,
) -> None:
    identity = _require_cleanup_ownership(
        path,
        workspace_key=workspace_key,
        cleanup_ownership_token=cleanup_ownership_token,
        expected_directory_identity=expected_directory_identity,
    )
    if _physical_directory_identity(path) != identity:
        raise ConfirmatoryMemoryWorkspaceError(
            "published workspace identity changed before cleanup"
        )
    _remove_owned_plain_tree(path)


def build_confirmatory_memory_workspace(
    project_root: str | Path,
    specs: Sequence[ConfirmatoryWorkspaceArraySpec],
    *,
    minimum_free_bytes_after: int,
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
    resource_input_workspace_plan: Mapping[str, Any] | None = None,
    index_specs: Sequence[ConfirmatoryWorkspaceIndexSpec] = (),
    maximum_workspace_bytes: int | None = None,
) -> ConfirmatoryMemoryWorkspace:
    """Create one fresh canonical shared-backing workspace.

    Capacity-v3 forbids workspace reuse.  Any pre-existing destination, even one that
    verifies successfully, fails closed and must be handled by an explicit owned
    cleanup before a new preflight or execution attempt.
    """

    if type(minimum_free_bytes_after) is not int or minimum_free_bytes_after < 0:
        raise ValueError("minimum final free bytes must be a non-negative integer")
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("workspace chunk bytes must be a positive integer")
    normalised = _normalise_specs(specs)
    workspace_key = _workspace_key(normalised)
    source_row_count = normalised[0].expected_shape[0]
    normalised_indices: tuple[ConfirmatoryWorkspaceIndexSpec, ...] = ()
    plan_sha: str | None = None
    validated_plan: dict[str, Any] | None = None
    if resource_input_workspace_plan is None:
        if index_specs or maximum_workspace_bytes is not None:
            raise ValueError(
                "workspace index specifications and byte ceiling require an exact plan"
            )
    else:
        if maximum_workspace_bytes is None:
            raise ValueError("authority-bound workspace plan requires its byte ceiling")
        normalised_indices = _normalise_index_specs(
            index_specs,
            source_row_count=source_row_count,
        )
        validated_plan = validate_confirmatory_memory_workspace_plan(
            resource_input_workspace_plan,
            normalised,
            normalised_indices,
            minimum_free_bytes_after=minimum_free_bytes_after,
            maximum_workspace_bytes=maximum_workspace_bytes,
        )
        if validated_plan.get("workspace_reuse_allowed") is not False:
            raise ConfirmatoryMemoryWorkspaceError(
                "capacity-v3 requires workspace_reuse_allowed=false"
            )
        plan_sha = str(validated_plan["plan_without_self_hash_sha256"])

    parent = canonical_confirmatory_memory_workspace_parent(project_root)
    destination = parent / workspace_key
    if destination.exists():
        raise ConfirmatoryMemoryWorkspaceError(
            "capacity-v3 forbids reuse of an existing workspace destination"
        )
    parent.mkdir(parents=True, exist_ok=True)
    _require_plain_directory_chain(Path(project_root).resolve(), parent)
    planned_sources, planned_sidecars = _source_hashes(normalised)
    infos = [_zip_member_info(spec) for spec in normalised]
    extracted_array_file_bytes = sum(info.file_size for info in infos)
    index_npy_file_bytes = sum(
        _npy_serialized_size(value.source_indices) for value in normalised_indices
    )
    planned_workspace_bytes = (
        extracted_array_file_bytes + index_npy_file_bytes + _METADATA_CAPACITY_ALLOWANCE_BYTES
    )
    if validated_plan is not None and (
        validated_plan["expected_extracted_file_bytes"] != extracted_array_file_bytes
        or validated_plan["expected_index_npy_file_bytes"] != index_npy_file_bytes
        or validated_plan["planned_workspace_bytes"] != planned_workspace_bytes
    ):
        raise ConfirmatoryMemoryWorkspaceError(
            "workspace byte plan differs from exact physical NPY sizes"
        )
    required_before = minimum_free_bytes_after + planned_workspace_bytes
    capacity_before = shutil.disk_usage(parent).free
    if capacity_before < required_before:
        raise ConfirmatoryMemoryWorkspaceError(
            "insufficient disk capacity for the shared backing workspace while retaining "
            "the exact required final free-space boundary"
        )

    lock_path = parent / f".{workspace_key}.lock"
    stage = parent / f".stage-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    cleanup_token = uuid.uuid4().hex + uuid.uuid4().hex
    cleanup_token_sha = hashlib.sha256(cleanup_token.encode("ascii")).hexdigest()
    descriptor: int | None = None
    lock_owned = False
    published = False
    published_identity: tuple[int, int, int] | None = None
    try:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            lock_owned = True
        except FileExistsError as error:
            raise ConfirmatoryMemoryWorkspaceError(
                "workspace construction lease already exists"
            ) from error
        os.write(
            descriptor,
            _canonical_json_bytes(
                {
                    "schema_version": 1,
                    "workspace_key": workspace_key,
                    "pid": os.getpid(),
                }
            )
            + b"\n",
        )
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None

        if destination.exists():
            raise ConfirmatoryMemoryWorkspaceError(
                "workspace destination appeared after lease acquisition"
            )
        stage.mkdir()
        arrays_directory = stage / "arrays"
        arrays_directory.mkdir()
        indices_directory = stage / "indices"
        indices_directory.mkdir()
        before_sources, before_sidecars = _source_hashes(normalised)
        if before_sources != planned_sources or before_sidecars != planned_sidecars:
            raise ConfirmatoryMemoryWorkspaceError(
                "source cache or sidecar changed after capacity planning"
            )
        array_records = [
            _extract_member(
                spec,
                arrays_directory / _array_filename(spec.array_id),
                chunk_bytes=chunk_bytes,
                expected_info=info,
            )
            for spec, info in zip(normalised, infos, strict=True)
        ]
        index_records = _write_index_arrays(indices_directory, normalised_indices)
        after_sources, after_sidecars = _source_hashes(normalised)
        if before_sources != after_sources or before_sidecars != after_sidecars:
            raise ConfirmatoryMemoryWorkspaceError(
                "source cache or sidecar changed during workspace extraction"
            )
        capacity_after_staging = shutil.disk_usage(parent).free
        if capacity_after_staging < minimum_free_bytes_after:
            raise ConfirmatoryMemoryWorkspaceError(
                "workspace staging violated the exact final free-space boundary"
            )
        capacity = {
            "minimum_free_bytes_after": minimum_free_bytes_after,
            "maximum_workspace_bytes": (
                maximum_workspace_bytes
                if maximum_workspace_bytes is not None
                else planned_workspace_bytes
            ),
            "metadata_capacity_allowance_bytes": _METADATA_CAPACITY_ALLOWANCE_BYTES,
            "planned_extracted_array_file_bytes": extracted_array_file_bytes,
            "planned_index_npy_file_bytes": index_npy_file_bytes,
            "planned_workspace_bytes": planned_workspace_bytes,
            "required_free_bytes_before": required_before,
            "observed_free_bytes_before": capacity_before,
            "observed_free_bytes_after_staging": capacity_after_staging,
        }
        receipt_base = _receipt_without_self_hash(
            workspace_key=workspace_key,
            arrays=array_records,
            indices=index_records,
            capacity=capacity,
            resource_input_workspace_plan_sha256=plan_sha,
            cleanup_ownership_token_sha256=cleanup_token_sha,
        )
        receipt = {
            **receipt_base,
            "receipt_without_self_hash_sha256": _canonical_sha256(receipt_base),
        }
        receipt_path = stage / "workspace_receipt.json"
        _write_json_fsync(receipt_path, receipt)
        relative_artifacts = [
            *(f"arrays/{_array_filename(spec.array_id)}" for spec in normalised),
            *(f"indices/{_index_filename(value)}" for value in normalised_indices),
            "workspace_receipt.json",
        ]
        manifest = _manifest_payload(stage, relative_artifacts)
        manifest_path = stage / "artifact_manifest.json"
        _write_json_fsync(manifest_path, manifest)
        manifest_sha = sha256_file(manifest_path)
        artifact_root = _canonical_sha256(manifest["artifacts"])
        marker = {
            "schema_version": 1,
            "status": "immutable",
            "workspace_key": workspace_key,
            "artifact_manifest_sha256": manifest_sha,
            "artifact_root_sha256": artifact_root,
            "workspace_receipt_sha256": sha256_file(receipt_path),
        }
        _write_json_fsync(stage / ".immutable.json", marker)
        physical_staged_bytes = sum(
            path.stat().st_size for path in stage.rglob("*") if path.is_file()
        )
        effective_ceiling = (
            maximum_workspace_bytes
            if maximum_workspace_bytes is not None
            else planned_workspace_bytes
        )
        if physical_staged_bytes > effective_ceiling:
            raise ConfirmatoryMemoryWorkspaceError(
                "physical staged workspace exceeds its byte ceiling"
            )
        if destination.exists():
            raise ConfirmatoryMemoryWorkspaceError(
                "workspace destination appeared before atomic publication"
            )
        os.replace(stage, destination)
        published_identity = _physical_directory_identity(destination)
        published = True
        if shutil.disk_usage(parent).free < minimum_free_bytes_after:
            raise ConfirmatoryMemoryWorkspaceError(
                "published workspace violates the exact final free-space boundary"
            )
        verified = verify_confirmatory_memory_workspace(
            project_root,
            destination,
            normalised,
            resource_input_workspace_plan=resource_input_workspace_plan,
            index_specs=normalised_indices,
            minimum_free_bytes_after=minimum_free_bytes_after,
            maximum_workspace_bytes=maximum_workspace_bytes,
        )
        return ConfirmatoryMemoryWorkspace(
            root=verified.root,
            workspace_key=verified.workspace_key,
            receipt_sha256=verified.receipt_sha256,
            artifact_root_sha256=verified.artifact_root_sha256,
            arrays=verified.arrays,
            index_arrays=verified.index_arrays,
            resource_input_workspace_plan_sha256=(verified.resource_input_workspace_plan_sha256),
            cleanup_ownership_token=cleanup_token,
        )
    except BaseException:
        if stage.exists():
            _remove_owned_plain_tree(stage)
        if published and destination.exists():
            _remove_authenticated_workspace_tree(
                destination,
                workspace_key=workspace_key,
                cleanup_ownership_token=cleanup_token,
                expected_directory_identity=published_identity,
            )
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if lock_owned:
            lock_path.unlink(missing_ok=True)


def close_and_cleanup_confirmatory_memory_workspace(
    project_root: str | Path,
    workspace: ConfirmatoryMemoryWorkspace,
    specs: Sequence[ConfirmatoryWorkspaceArraySpec],
    *,
    resource_input_workspace_plan: Mapping[str, Any] | None = None,
    index_specs: Sequence[ConfirmatoryWorkspaceIndexSpec] = (),
    minimum_free_bytes_after: int | None = None,
    maximum_workspace_bytes: int | None = None,
) -> dict[str, Any]:
    """Verify, close, and remove exactly one fresh builder-owned workspace.

    A workspace returned by :func:`verify_confirmatory_memory_workspace` has no
    cleanup token and therefore cannot authorize deletion.  This keeps verification
    read-only while allowing the fresh-only capacity-v3 lifecycle to release its
    own disk allocation deterministically.
    """

    token = workspace.cleanup_ownership_token
    if token is None:
        raise ConfirmatoryMemoryWorkspaceError(
            "read-only workspace verification has no cleanup authority"
        )
    parent = canonical_confirmatory_memory_workspace_parent(project_root)
    expected_root = parent / workspace.workspace_key
    if workspace.root.resolve() != expected_root:
        raise ConfirmatoryMemoryWorkspaceError(
            "cleanup target is not the exact canonical builder-owned workspace"
        )
    lock_path = parent / f".{workspace.workspace_key}.lock"
    if lock_path.exists():
        raise ConfirmatoryMemoryWorkspaceError(
            "workspace cleanup is forbidden while a construction lease exists"
        )
    owned_identity = _physical_directory_identity(expected_root)

    verified = verify_confirmatory_memory_workspace(
        project_root,
        expected_root,
        specs,
        resource_input_workspace_plan=resource_input_workspace_plan,
        index_specs=index_specs,
        minimum_free_bytes_after=minimum_free_bytes_after,
        maximum_workspace_bytes=maximum_workspace_bytes,
    )
    try:
        if (
            verified.workspace_key != workspace.workspace_key
            or verified.receipt_sha256 != workspace.receipt_sha256
            or verified.artifact_root_sha256 != workspace.artifact_root_sha256
            or verified.resource_input_workspace_plan_sha256
            != workspace.resource_input_workspace_plan_sha256
        ):
            raise ConfirmatoryMemoryWorkspaceError(
                "cleanup readback differs from the builder-owned workspace"
            )
        receipt_sha = verified.receipt_sha256
        artifact_root = verified.artifact_root_sha256
        plan_sha = verified.resource_input_workspace_plan_sha256
    finally:
        verified.close()

    workspace.close()
    _remove_authenticated_workspace_tree(
        expected_root,
        workspace_key=workspace.workspace_key,
        cleanup_ownership_token=token,
        expected_directory_identity=owned_identity,
    )
    if expected_root.exists():
        raise ConfirmatoryMemoryWorkspaceError(
            "builder-owned workspace remains after authenticated cleanup"
        )
    return {
        "schema_version": 1,
        "status": "complete",
        "workspace_key": workspace.workspace_key,
        "workspace_receipt_sha256": receipt_sha,
        "artifact_root_sha256": artifact_root,
        "resource_input_workspace_plan_sha256": plan_sha,
        "workspace_removed": True,
        "source_annotations_modified": False,
        "scientific_outcomes_read": False,
    }


def verify_confirmatory_memory_workspace(
    project_root: str | Path,
    workspace_path: str | Path,
    specs: Sequence[ConfirmatoryWorkspaceArraySpec],
    *,
    resource_input_workspace_plan: Mapping[str, Any] | None = None,
    index_specs: Sequence[ConfirmatoryWorkspaceIndexSpec] = (),
    minimum_free_bytes_after: int | None = None,
    maximum_workspace_bytes: int | None = None,
) -> ConfirmatoryMemoryWorkspace:
    """Verify every sealed byte and return read-only memmap handles."""

    normalised = _normalise_specs(specs)
    expected_key = _workspace_key(normalised)
    expected_plan_sha: str | None = None
    normalised_indices: tuple[ConfirmatoryWorkspaceIndexSpec, ...] = ()
    validated_plan: dict[str, Any] | None = None
    if resource_input_workspace_plan is None:
        if index_specs or maximum_workspace_bytes is not None:
            raise ValueError("workspace plan validation arguments require an exact plan")
    else:
        if minimum_free_bytes_after is None or maximum_workspace_bytes is None:
            raise ValueError("workspace plan verification requires both capacity boundaries")
        normalised_indices = _normalise_index_specs(
            index_specs,
            source_row_count=normalised[0].expected_shape[0],
        )
        validated_plan = validate_confirmatory_memory_workspace_plan(
            resource_input_workspace_plan,
            normalised,
            normalised_indices,
            minimum_free_bytes_after=minimum_free_bytes_after,
            maximum_workspace_bytes=maximum_workspace_bytes,
        )
        expected_plan_sha = str(validated_plan["plan_without_self_hash_sha256"])
    parent = canonical_confirmatory_memory_workspace_parent(project_root)
    workspace = Path(workspace_path).resolve()
    if workspace != parent / expected_key:
        raise ConfirmatoryMemoryWorkspaceError(
            "workspace path is not the exact canonical path for its specification"
        )
    _require_plain_directory_chain(Path(project_root).resolve(), workspace)
    if not workspace.is_dir() or _is_reparse(workspace):
        raise ConfirmatoryMemoryWorkspaceError("workspace is unavailable or a reparse point")
    expected_names = {
        ".immutable.json",
        "artifact_manifest.json",
        "workspace_receipt.json",
        "arrays",
        "indices",
    }
    if {path.name for path in workspace.iterdir()} != expected_names:
        raise ConfirmatoryMemoryWorkspaceError("workspace root inventory is not exact")
    arrays_directory = workspace / "arrays"
    indices_directory = workspace / "indices"
    if not arrays_directory.is_dir() or _is_reparse(arrays_directory):
        raise ConfirmatoryMemoryWorkspaceError("workspace arrays directory is invalid")
    if not indices_directory.is_dir() or _is_reparse(indices_directory):
        raise ConfirmatoryMemoryWorkspaceError("workspace indices directory is invalid")
    expected_array_names = {_array_filename(spec.array_id) for spec in normalised}
    if {path.name for path in arrays_directory.iterdir()} != expected_array_names:
        raise ConfirmatoryMemoryWorkspaceError("workspace array inventory is not exact")
    for path in arrays_directory.iterdir():
        if not path.is_file() or _is_reparse(path):
            raise ConfirmatoryMemoryWorkspaceError("workspace array is not a plain physical file")
    expected_index_names = {_index_filename(value) for value in normalised_indices}
    if {path.name for path in indices_directory.iterdir()} != expected_index_names:
        raise ConfirmatoryMemoryWorkspaceError("workspace index inventory is not exact")
    for path in indices_directory.iterdir():
        if not path.is_file() or _is_reparse(path):
            raise ConfirmatoryMemoryWorkspaceError("workspace index is not a plain physical file")

    receipt_path = workspace / "workspace_receipt.json"
    manifest_path = workspace / "artifact_manifest.json"
    marker_path = workspace / ".immutable.json"
    for path, role in (
        (receipt_path, "workspace receipt"),
        (manifest_path, "workspace artifact manifest"),
        (marker_path, "workspace immutable marker"),
    ):
        resolved = _require_plain_existing_file(path, role)
        if resolved != path:
            raise ConfirmatoryMemoryWorkspaceError(f"{role} escapes its exact workspace path")
    if (
        receipt_path.stat().st_size + manifest_path.stat().st_size + marker_path.stat().st_size
        > _METADATA_CAPACITY_ALLOWANCE_BYTES
    ):
        raise ConfirmatoryMemoryWorkspaceError(
            "workspace metadata exceeds its aggregate bounded allowance"
        )
    receipt = _strict_json_object(receipt_path, "workspace receipt")
    manifest = _strict_json_object(manifest_path, "workspace artifact manifest")
    marker = _strict_json_object(marker_path, "workspace immutable marker")
    receipt_base = dict(receipt)
    receipt_self_hash = receipt_base.pop("receipt_without_self_hash_sha256", None)
    receipt_capacity = receipt.get("capacity")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "complete"
        or receipt.get("recipe_id") != _WORKSPACE_RECIPE_ID
        or receipt.get("workspace_key") != expected_key
        or receipt.get("workspace_reuse_allowed") is not False
        or receipt.get("source_annotations_modified") is not False
        or receipt.get("scientific_outcomes_read") is not False
        or receipt.get("resource_input_workspace_plan_sha256") != expected_plan_sha
        or _SHA256.fullmatch(str(receipt.get("cleanup_ownership_token_sha256"))) is None
        or not isinstance(receipt_capacity, dict)
        or (
            minimum_free_bytes_after is not None
            and receipt_capacity.get("minimum_free_bytes_after") != minimum_free_bytes_after
        )
        or receipt_self_hash != _canonical_sha256(receipt_base)
    ):
        raise ConfirmatoryMemoryWorkspaceError("workspace receipt failed exact readback")
    expected_manifest = _manifest_payload(
        workspace,
        [
            *(f"arrays/{_array_filename(spec.array_id)}" for spec in normalised),
            *(f"indices/{_index_filename(value)}" for value in normalised_indices),
            "workspace_receipt.json",
        ],
    )
    if manifest != expected_manifest:
        raise ConfirmatoryMemoryWorkspaceError("workspace manifest differs from physical files")
    manifest_sha = sha256_file(manifest_path)
    artifact_root = _canonical_sha256(manifest["artifacts"])
    if marker != {
        "schema_version": 1,
        "status": "immutable",
        "workspace_key": expected_key,
        "artifact_manifest_sha256": manifest_sha,
        "artifact_root_sha256": artifact_root,
        "workspace_receipt_sha256": sha256_file(receipt_path),
    }:
        raise ConfirmatoryMemoryWorkspaceError("workspace immutable marker failed readback")
    physical_workspace_bytes = sum(
        path.stat().st_size
        for path in (
            *(workspace / row["path"] for row in manifest["artifacts"]),
            manifest_path,
            marker_path,
        )
    )
    if maximum_workspace_bytes is not None and physical_workspace_bytes > maximum_workspace_bytes:
        raise ConfirmatoryMemoryWorkspaceError(
            "physical workspace exceeds its authority-bound byte ceiling"
        )
    if validated_plan is not None:
        expected_capacity = {
            "minimum_free_bytes_after": minimum_free_bytes_after,
            "maximum_workspace_bytes": maximum_workspace_bytes,
            "metadata_capacity_allowance_bytes": _METADATA_CAPACITY_ALLOWANCE_BYTES,
            "planned_extracted_array_file_bytes": validated_plan["expected_extracted_file_bytes"],
            "planned_index_npy_file_bytes": validated_plan["expected_index_npy_file_bytes"],
            "planned_workspace_bytes": validated_plan["planned_workspace_bytes"],
            "required_free_bytes_before": validated_plan["required_free_bytes_before"],
        }
        if any(receipt_capacity.get(key) != value for key, value in expected_capacity.items()):
            raise ConfirmatoryMemoryWorkspaceError(
                "workspace receipt capacity differs from authority-bound capacity-v3"
            )
    if (
        type(receipt_capacity.get("observed_free_bytes_before")) is not int
        or type(receipt_capacity.get("observed_free_bytes_after_staging")) is not int
        or receipt_capacity["observed_free_bytes_before"]
        < receipt_capacity["required_free_bytes_before"]
        or receipt_capacity["observed_free_bytes_after_staging"]
        < receipt_capacity["minimum_free_bytes_after"]
    ):
        raise ConfirmatoryMemoryWorkspaceError("workspace capacity observations are invalid")

    raw_records = receipt.get("arrays")
    if not isinstance(raw_records, list):
        raise ConfirmatoryMemoryWorkspaceError("workspace receipt lacks its array inventory")
    records = {
        str(value.get("array_id")): value for value in raw_records if isinstance(value, dict)
    }
    if len(records) != len(raw_records) or set(records) != {spec.array_id for spec in normalised}:
        raise ConfirmatoryMemoryWorkspaceError("workspace receipt array IDs are not exact")
    raw_index_records = receipt.get("partition_indices")
    if not isinstance(raw_index_records, list):
        raise ConfirmatoryMemoryWorkspaceError("workspace receipt lacks physical indices")
    index_records = {
        (value.get("outer_fold"), value.get("role")): value
        for value in raw_index_records
        if isinstance(value, dict)
    }
    expected_index_keys = {(value.outer_fold, value.role) for value in normalised_indices}
    if len(index_records) != len(raw_index_records) or set(index_records) != expected_index_keys:
        raise ConfirmatoryMemoryWorkspaceError("workspace receipt index bindings are not exact")
    arrays: dict[str, ReadOnlyBackingArray] = {}
    index_arrays: dict[str, NDArray[np.int64]] = {}
    opened: list[np.memmap] = []
    try:
        for spec in normalised:
            record = records[spec.array_id]
            expected_relative = f"arrays/{_array_filename(spec.array_id)}"
            path = workspace / expected_relative
            verified_identity = _physical_file_identity(path)
            file_sha = sha256_file(path)
            if _physical_file_identity(path) != verified_identity:
                raise ConfirmatoryMemoryWorkspaceError(
                    "workspace array changed while its file SHA-256 was read"
                )
            try:
                validation_map = np.load(path, allow_pickle=False, mmap_mode="r")
            except (OSError, ValueError) as error:
                raise ConfirmatoryMemoryWorkspaceError(
                    "workspace array cannot be opened safely"
                ) from error
            if not isinstance(validation_map, np.memmap):
                raise ConfirmatoryMemoryWorkspaceError("workspace array is not a memmap")
            try:
                raw_array_sha = _memmap_array_sha256(validation_map)
                typed = (
                    validation_map.dtype.str == np.dtype(spec.expected_dtype).str
                    and tuple(validation_map.shape) == spec.expected_shape
                    and not validation_map.dtype.hasobject
                    and validation_map.flags.c_contiguous
                    and not validation_map.flags.writeable
                )
            finally:
                mmap_handle = getattr(validation_map, "_mmap", None)
                if mmap_handle is not None:
                    mmap_handle.close()
            if _physical_file_identity(path) != verified_identity:
                raise ConfirmatoryMemoryWorkspaceError(
                    "workspace array changed during typed/hash validation"
                )
            info = _zip_member_info(spec)
            expected_record = {
                "array_id": spec.array_id,
                "relative_path": expected_relative,
                "source_npz_path": str(spec.source_npz_path),
                "source_sidecar_path": str(spec.source_sidecar_path),
                "source_npz_sha256": spec.expected_source_sha256,
                "source_sidecar_sha256": spec.expected_source_sidecar_sha256,
                "source_member_name": spec.member_name,
                "source_member_crc32": f"{info.CRC:08x}",
                "source_member_compression": (
                    "deflated" if info.compress_type == zipfile.ZIP_DEFLATED else "stored"
                ),
                "source_member_compressed_bytes": int(info.compress_size),
                "source_member_uncompressed_bytes": int(info.file_size),
                "npy_file_sha256": file_sha,
                "raw_array_sha256": raw_array_sha,
                "dtype": np.dtype(spec.expected_dtype).str,
                "shape": list(spec.expected_shape),
                "order": "C",
                "nbytes": int(np.prod(spec.expected_shape, dtype=np.int64))
                * np.dtype(spec.expected_dtype).itemsize,
            }
            if (
                record != expected_record
                or raw_array_sha != spec.expected_array_sha256
                or not typed
            ):
                raise ConfirmatoryMemoryWorkspaceError(
                    f"workspace array {spec.array_id!r} failed exact typed readback"
                )
            cold_map = np.load(path, allow_pickle=False, mmap_mode="r")
            if not isinstance(cold_map, np.memmap):
                raise ConfirmatoryMemoryWorkspaceError("workspace array failed cold reopen")
            cold_typed = (
                cold_map.dtype.str == np.dtype(spec.expected_dtype).str
                and tuple(cold_map.shape) == spec.expected_shape
                and cold_map.flags.c_contiguous
                and not cold_map.flags.writeable
            )
            if _physical_file_identity(path) != verified_identity or not cold_typed:
                mmap_handle = getattr(cold_map, "_mmap", None)
                if mmap_handle is not None:
                    mmap_handle.close()
                raise ConfirmatoryMemoryWorkspaceError(
                    "workspace array changed before its cold verified reopen"
                )
            opened.append(cold_map)
            arrays[spec.array_id] = ReadOnlyBackingArray(
                array_id=spec.array_id,
                path=path,
                values=cold_map,
                file_sha256=file_sha,
                raw_array_sha256=raw_array_sha,
                source_npz_sha256=spec.expected_source_sha256,
                source_sidecar_sha256=spec.expected_source_sidecar_sha256,
                source_member_name=spec.member_name,
                verified_file_identity=verified_identity,
            )
        for value in normalised_indices:
            key = workspace_index_id(value.outer_fold, value.role)
            record = index_records[(value.outer_fold, value.role)]
            expected_relative = f"indices/{_index_filename(value)}"
            path = workspace / expected_relative
            verified_identity = _physical_file_identity(path)
            file_sha = sha256_file(path)
            if _physical_file_identity(path) != verified_identity:
                raise ConfirmatoryMemoryWorkspaceError(
                    "workspace index changed while its file SHA-256 was read"
                )
            validation_map = np.load(path, allow_pickle=False, mmap_mode="r")
            if not isinstance(validation_map, np.memmap):
                raise ConfirmatoryMemoryWorkspaceError("workspace index is not a memmap")
            try:
                index_sha = _memmap_array_sha256(validation_map)
                index_matches = (
                    validation_map.dtype == np.dtype(np.int64)
                    and validation_map.shape == (value.row_count,)
                    and validation_map.flags.c_contiguous
                    and not validation_map.flags.writeable
                    and np.array_equal(validation_map, value.source_indices)
                )
            finally:
                mmap_handle = getattr(validation_map, "_mmap", None)
                if mmap_handle is not None:
                    mmap_handle.close()
            if _physical_file_identity(path) != verified_identity:
                raise ConfirmatoryMemoryWorkspaceError(
                    "workspace index changed during typed/hash validation"
                )
            expected_record = {
                **value.semantic_dict(),
                "relative_path": expected_relative,
                "raw_nbytes": value.source_indices.nbytes,
                "npy_file_bytes": path.stat().st_size,
                "npy_file_sha256": file_sha,
            }
            if (
                record != expected_record
                or index_sha != value.source_indices_sha256
                or not index_matches
            ):
                raise ConfirmatoryMemoryWorkspaceError(
                    f"workspace index {key!r} failed exact typed readback"
                )
            # Nine index vectors total only a few MiB.  Copy their verified bytes into
            # a bytes-owned array so later on-disk edits cannot change row authority.
            immutable_index = np.frombuffer(
                value.source_indices.tobytes(order="C"),
                dtype=np.int64,
            )
            immutable_index.setflags(write=False)
            index_arrays[key] = immutable_index
        _source_hashes(normalised)
        return ConfirmatoryMemoryWorkspace(
            root=workspace,
            workspace_key=expected_key,
            receipt_sha256=sha256_file(receipt_path),
            artifact_root_sha256=artifact_root,
            arrays=MappingProxyType(arrays),
            index_arrays=MappingProxyType(index_arrays),
            resource_input_workspace_plan_sha256=expected_plan_sha,
        )
    except BaseException:
        for mapping in opened:
            mmap_handle = getattr(mapping, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
        raise


__all__ = [
    "ConfirmatoryMemoryWorkspace",
    "ConfirmatoryMemoryWorkspaceError",
    "ConfirmatoryWorkspaceArraySpec",
    "ConfirmatoryWorkspaceIndexSpec",
    "ReadOnlyBackingArray",
    "RowIndexedArray",
    "build_confirmatory_memory_workspace",
    "build_confirmatory_memory_workspace_plan",
    "canonical_confirmatory_memory_workspace_parent",
    "close_and_cleanup_confirmatory_memory_workspace",
    "validate_confirmatory_memory_workspace_plan",
    "verify_confirmatory_memory_workspace",
    "workspace_index_id",
]
