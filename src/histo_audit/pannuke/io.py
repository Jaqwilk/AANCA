"""Path, hashing, bounded sampling, and atomic-I/O helpers."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from histo_audit.utils.run_tracking import atomic_write_json, atomic_write_text

from .exceptions import PanNukeNotFoundError, PanNukeSemanticsError
from .models import RawFileRecord
from .publication import assert_mutable_publication_destination

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".rar", ".7z")


def locate_pannuke_root(
    explicit_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve PanNuke using explicit path, ``PANNUKE_ROOT``, then the project default.

    An explicitly supplied but invalid path is an error rather than permission to
    silently use a different dataset.
    """

    environment = os.environ if environ is None else environ
    if explicit_path is not None:
        explicit = Path(explicit_path).expanduser().resolve()
        if not explicit.is_dir():
            raise PanNukeNotFoundError(
                f"explicit PanNuke root does not exist or is not a directory: {explicit}"
            )
        return explicit

    checked: list[Path] = []
    configured = environment.get("PANNUKE_ROOT", "").strip()
    if configured:
        candidate = Path(configured).expanduser().resolve()
        checked.append(candidate)
        if candidate.is_dir():
            return candidate

    base = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    default = base / "data" / "raw" / "pannuke"
    checked.append(default)
    if default.is_dir():
        return default
    rendered = ", ".join(str(path) for path in checked)
    raise PanNukeNotFoundError(
        "PanNuke was not found. Supply --root, set PANNUKE_ROOT, or place the "
        f"release at data/raw/pannuke. Checked: {rendered}"
    )


def sha256_file(path: str | Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Stream a file into SHA-256 without loading it into RAM."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def is_archive(path: Path) -> bool:
    """Recognise common release archive suffixes without extracting them."""

    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def inventory_raw_files(
    root: str | Path,
    *,
    fold_resolver: Any | None = None,
    exclude_paths: Iterable[str | Path] = (),
    include_temporary_files: bool = False,
) -> tuple[RawFileRecord, ...]:
    """Hash every raw file under *root* in stable relative-path order."""

    source = Path(root).resolve()
    excluded = {Path(path).resolve() for path in exclude_paths}

    def included(path: Path) -> bool:
        resolved = path.resolve()
        return not any(resolved == item or item in resolved.parents for item in excluded)

    records: list[RawFileRecord] = []
    paths = sorted((item for item in source.rglob("*") if item.is_file()), key=str)
    for path in paths:
        is_temporary = path.name.startswith(".") and path.name.endswith(".tmp")
        if not included(path) or (is_temporary and not include_temporary_files):
            continue
        try:
            stat_before = path.stat()
            digest = sha256_file(path)
            stat_after = path.stat()
        except OSError as error:
            raise PanNukeSemanticsError(
                f"raw file changed while inventory snapshot was being hashed: {path}"
            ) from error
        stable_identity_before = (
            stat_before.st_dev,
            stat_before.st_ino,
            stat_before.st_size,
            stat_before.st_mtime_ns,
            stat_before.st_ctime_ns,
        )
        stable_identity_after = (
            stat_after.st_dev,
            stat_after.st_ino,
            stat_after.st_size,
            stat_after.st_mtime_ns,
            stat_after.st_ctime_ns,
        )
        if stable_identity_before != stable_identity_after:
            raise PanNukeSemanticsError(
                f"raw file changed while inventory snapshot was being hashed: {path}"
            )
        relative = path.relative_to(source).as_posix()
        fold_id = fold_resolver(path, source) if fold_resolver is not None else None
        kind = "archive" if is_archive(path) else path.suffix.lower().lstrip(".") or "file"
        records.append(
            RawFileRecord(
                relative_path=relative,
                size_bytes=stat_after.st_size,
                sha256=digest,
                fold_id=fold_id,
                file_kind=kind,
            )
        )
    return tuple(records)


def open_npy_mmap(path: str | Path) -> NDArray[np.generic]:
    """Open an array read-only and memory-mapped, refusing pickle-dependent data."""

    source = Path(path)
    try:
        array = np.load(source, mmap_mode="r", allow_pickle=False)
    except ValueError as error:
        raise PanNukeSemanticsError(
            f"cannot safely memory-map {source}; object/pickled arrays are not accepted: {error}"
        ) from error
    if not isinstance(array, np.ndarray):
        raise PanNukeSemanticsError(f"expected a single .npy array: {source}")
    if array.dtype.hasobject:
        raise PanNukeSemanticsError(f"unsafe object dtype is not accepted: {source}")
    return array


def deterministic_sample_indices(
    count: int,
    *,
    max_samples: int,
    bytes_per_sample: int = 0,
    memory_budget_bytes: int = 256 * 1024 * 1024,
) -> tuple[int, ...]:
    """Return stable, evenly spread indices bounded by a memory-aware budget."""

    if count < 0 or max_samples <= 0 or memory_budget_bytes <= 0:
        raise ValueError("count must be non-negative and sampling budgets positive")
    if count == 0:
        return ()
    allowed = max_samples
    if bytes_per_sample > 0:
        allowed = min(allowed, max(1, memory_budget_bytes // bytes_per_sample))
    target = min(count, allowed)
    if target == count:
        return tuple(range(count))
    # Integer arithmetic avoids platform-dependent floating rounding.
    return (
        tuple((index * (count - 1)) // (target - 1) for index in range(target))
        if target > 1
        else (0,)
    )


def ensure_output_capacity(
    output_dir: str | Path,
    *,
    estimated_bytes: int,
    reserve_bytes: int = 64 * 1024 * 1024,
) -> Path:
    """Create an output directory and fail early when free space is inadequate."""

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination).free
    required = max(0, estimated_bytes) + max(0, reserve_bytes)
    if free < required:
        raise OSError(
            f"insufficient disk space at {destination}: need about {required} bytes, "
            f"found {free} bytes free"
        )
    return destination


def ensure_derived_output_outside_raw(
    output_path: str | Path,
    raw_root: str | Path,
    *,
    purpose: str = "derived output",
) -> Path:
    """Resolve an output path and reject writes inside the immutable raw tree.

    Resolution happens before callers create parents or staging files, so existing
    directory symlinks/junctions and ``..`` traversal cannot redirect a derived
    artifact into the validated raw release.
    """

    destination = Path(output_path).expanduser().resolve()
    source = Path(raw_root).expanduser().resolve()
    if destination == source or source in destination.parents:
        raise PanNukeSemanticsError(
            f"{purpose} must be outside the immutable PanNuke raw release: "
            f"destination={destination}, raw_root={source}"
        )
    try:
        return assert_mutable_publication_destination(destination, role=purpose)
    except (NotADirectoryError, PermissionError, RuntimeError) as error:
        raise PanNukeSemanticsError(str(error)) from error


def atomic_replace_via_temp(destination: str | Path, writer: Any) -> Path:
    """Call ``writer(temp_path)`` and atomically replace *destination*."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=target.suffix, dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


__all__ = [
    "ARCHIVE_SUFFIXES",
    "atomic_replace_via_temp",
    "atomic_write_json",
    "atomic_write_text",
    "deterministic_sample_indices",
    "ensure_derived_output_outside_raw",
    "ensure_output_capacity",
    "inventory_raw_files",
    "is_archive",
    "locate_pannuke_root",
    "open_npy_mmap",
    "sha256_file",
]
