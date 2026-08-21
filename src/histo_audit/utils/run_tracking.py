"""Atomic, append-only experiment provenance and run tracking.

The helpers in this module deliberately avoid advisory platform-specific file
locking.  Registry writers coordinate through an atomically-created lock file,
which works on Windows and POSIX filesystems.  Completed run directories receive
an immutable marker and this API will refuse all subsequent writes to them.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback as traceback_module
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import numpy as np
import yaml

from histo_audit.config import config_sha256, resolve_config

RunOutcome = Literal["completed", "failed"]

REGISTRY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "experiment_name",
    "status",
    "started_at",
    "completed_at",
    "config_sha256",
    "git_state",
    "dataset_sha256",
    "manifest_sha256",
    "split_seed",
    "model_seed",
    "corruption_seed",
    "run_path",
)
IMMUTABLE_MARKER = ".immutable.json"
STATUS_FILENAME = "status.json"
ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"
INTEGRITY_REGISTRY_FILENAME = "integrity_registry.jsonl"
RUN_DISPOSITION_ANCHOR_FILENAME = "run_dispositions.anchor.json"
RUN_DISPOSITION_REGISTRY_FILENAME = "run_dispositions.jsonl"
RUN_STAGE_ATTESTATION_ANCHOR_FILENAME = "run_stage_attestations.anchor.json"
RUN_STAGE_ATTESTATION_REGISTRY_FILENAME = "run_stage_attestations.jsonl"
_INTEGRITY_EXCLUSIONS = {ARTIFACT_MANIFEST_FILENAME, IMMUTABLE_MARKER}
SOURCE_TREE_MANIFEST_FILENAME = "source_tree_manifest.json"
EVENTS_FILENAME = "events.jsonl"
RUN_LOG_FILENAME = "run.log"

# These documents describe scientific governance, lifecycle state, and repository
# safeguards.  They are captured separately from the executable source identity so
# truthful post-freeze status and decision-log updates cannot change the code/config
# hash used by execution gates.
SOURCE_GOVERNANCE_FILENAMES: tuple[str, ...] = (
    ".gitignore",
    "AGENTS.md",
    "SPEC.md",
    "PLAN.md",
    "STATUS.md",
    "PRE_REGISTRATION.md",
    "DATASET_SETUP.md",
    "DECISIONS.md",
    "ETHICS_AND_LIMITATIONS.md",
)

_EXECUTION_SOURCE_DIRECTORIES: tuple[str, ...] = ("src", "configs")
_EXECUTION_SOURCE_STANDALONE_FILENAMES: tuple[str, ...] = ("pyproject.toml", "uv.lock")
# These are generated, byte-for-byte frozen publications of configs that remain
# independently authenticated by preregistration freeze/amendment bundles. Including
# them in the execution tree would make publishing the bundle invalidate the source
# root captured immediately before publication.
_EXECUTION_SOURCE_EXCLUDED_PATHS: frozenset[str] = frozenset(
    {"configs/confirmatory_frozen.yaml", "configs/primary_frozen.yaml"}
)
_SOURCE_TREE_IGNORED_PARTS: frozenset[str] = frozenset(
    {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
)


def utc_now() -> str:
    """Return a UTC timestamp in stable ISO-8601 form."""

    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def _fsync_directory(directory: Path) -> None:
    """Durably record directory-entry changes where directory fsync is supported."""

    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: str | Path, content: bytes) -> Path:
    """Durably replace *path* from a temporary file in the same directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_text(path: str | Path, content: str) -> Path:
    """Atomically write UTF-8 text with platform-independent newlines."""

    return atomic_write_bytes(path, content.replace("\r\n", "\n").encode("utf-8"))


def atomic_write_json(
    path: str | Path,
    value: Any,
    *,
    indent: int = 2,
) -> Path:
    """Atomically write strict JSON; NaN and infinity are rejected."""

    content = json.dumps(
        value,
        allow_nan=False,
        default=_json_default,
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
    )
    return atomic_write_text(path, f"{content}\n")


def atomic_write_npz(
    path: str | Path,
    arrays: Mapping[str, Any],
    *,
    compressed: bool = True,
) -> Path:
    """Durably replace one NumPy archive, compressed by default."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer = np.savez_compressed if compressed else np.savez
            writer(stream, **dict(arrays))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_yaml(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Atomically write a stable resolved YAML mapping."""

    content = yaml.safe_dump(
        resolve_config(value),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    )
    return atomic_write_text(path, content)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream one file into a SHA-256 digest."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"cannot checksum missing file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def windows_compatible_relative_path_sort_key(
    relative_path: str | PurePosixPath,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return one host-independent ordering key compatible with Windows paths."""

    parts = PurePosixPath(relative_path).parts
    return tuple(part.lower() for part in parts), tuple(parts)


def sha256_path(path: str | Path) -> str:
    """Hash a file or a directory tree including relative file names."""

    source = Path(path)
    if source.is_file():
        return sha256_file(source)
    if not source.is_dir():
        raise FileNotFoundError(f"cannot checksum missing path: {source}")
    digest = hashlib.sha256()
    files = sorted(
        (item for item in source.rglob("*") if item.is_file()),
        key=lambda item: windows_compatible_relative_path_sort_key(
            item.relative_to(source).as_posix()
        ),
    )
    for item in files:
        relative = item.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _artifact_records(
    run_directory: Path,
    *,
    extra_exclusions: set[str] | None = None,
) -> list[dict[str, Any]]:
    exclusions = _INTEGRITY_EXCLUSIONS | (extra_exclusions or set())
    records: list[dict[str, Any]] = []
    for artifact in sorted(run_directory.rglob("*"), key=lambda path: path.as_posix()):
        relative = artifact.relative_to(run_directory).as_posix()
        if relative in exclusions:
            continue
        if artifact.is_symlink():
            raise ValueError(f"run artifacts must not be symbolic links: {relative}")
        if artifact.is_file():
            records.append(
                {
                    "path": relative,
                    "size_bytes": artifact.stat().st_size,
                    "sha256": sha256_file(artifact),
                }
            )
    return records


def _artifact_root_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    canonical = json.dumps(
        list(records), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _tree_records(
    project_root: Path,
    candidates: Sequence[Path],
    *,
    role: str,
    excluded_relative_paths: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Return stable checksum records for one explicitly scoped project tree."""

    records: list[dict[str, Any]] = []
    for path in sorted(set(candidates), key=lambda item: item.relative_to(project_root).as_posix()):
        relative = path.relative_to(project_root)
        if relative.as_posix() in excluded_relative_paths:
            continue
        if any(part in _SOURCE_TREE_IGNORED_PARTS for part in relative.parts) or path.suffix in {
            ".pyc",
            ".pyo",
        }:
            continue
        if path.is_symlink():
            raise ValueError(f"{role} must not contain symlinks: {relative}")
        records.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def capture_source_tree(project_root: str | Path) -> dict[str, Any]:
    """Hash only executable code, configuration, and dependency definitions."""

    root = Path(project_root).resolve()
    candidates: list[Path] = []
    for directory_name in _EXECUTION_SOURCE_DIRECTORIES:
        directory = root / directory_name
        if directory.is_dir():
            candidates.extend(path for path in directory.rglob("*") if path.is_file())
    for filename in _EXECUTION_SOURCE_STANDALONE_FILENAMES:
        candidate = root / filename
        if candidate.is_file():
            candidates.append(candidate)
    records = _tree_records(
        root,
        candidates,
        role="execution source tree",
        excluded_relative_paths=_EXECUTION_SOURCE_EXCLUDED_PATHS,
    )
    return {
        "schema_version": 3,
        "scope_kind": "execution_source",
        "scope": [
            *(f"{directory}/**" for directory in _EXECUTION_SOURCE_DIRECTORIES),
            *_EXECUTION_SOURCE_STANDALONE_FILENAMES,
        ],
        "excluded_roots": [".git", ".venv", "artifacts", "data"],
        "excluded_paths": sorted(_EXECUTION_SOURCE_EXCLUDED_PATHS),
        "artifact_count": len(records),
        "root_sha256": _artifact_root_sha256(records),
        "artifacts": records,
    }


def capture_governance_tree(project_root: str | Path) -> dict[str, Any]:
    """Hash the exact, explicit set of live project-governance files."""

    root = Path(project_root).resolve()
    candidates = [
        candidate
        for filename in SOURCE_GOVERNANCE_FILENAMES
        if (candidate := root / filename).is_file()
    ]
    records = _tree_records(root, candidates, role="governance tree")
    return {
        "schema_version": 1,
        "scope_kind": "governance_snapshot",
        "scope": list(SOURCE_GOVERNANCE_FILENAMES),
        "governance_files": list(SOURCE_GOVERNANCE_FILENAMES),
        "artifact_count": len(records),
        "root_sha256": _artifact_root_sha256(records),
        "artifacts": records,
    }


checksum_file = sha256_file
checksum_path = sha256_path


def format_traceback(error: BaseException) -> str:
    """Format an exception and its attached traceback without fabricating one."""

    return "".join(traceback_module.format_exception(type(error), error, error.__traceback__))


def _subprocess_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0


def _git(
    project_root: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        creationflags=_subprocess_flags(),
    )


def capture_git_state(project_root: str | Path | None = None) -> dict[str, Any]:
    """Capture commit, branch, and exact porcelain status without changing Git."""

    root = Path(project_root or Path.cwd()).resolve()
    try:
        inside = _git(root, "rev-parse", "--is-inside-work-tree")
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": f"git unavailable: {exc}"}
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return {"available": False, "reason": "project root is not a Git work tree"}
    commit_result = _git(root, "rev-parse", "HEAD")
    branch_result = _git(root, "branch", "--show-current")
    status_result = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    status = status_result.stdout.rstrip("\r\n") if status_result.returncode == 0 else None
    return {
        "available": True,
        "commit": commit,
        "branch": branch_result.stdout.strip() if branch_result.returncode == 0 else None,
        "dirty": bool(status) if status is not None else None,
        "status_porcelain": status,
        "captured_at_utc": utc_now(),
    }


def _installed_distribution_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            versions[str(name)] = str(distribution.version)
    return dict(sorted(versions.items()))


def _nvidia_environment() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            creationflags=_subprocess_flags(),
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"available": False, "devices": [], "error": str(exc)}
    devices: list[dict[str, Any]] = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) != 4:
                continue
            try:
                index: int | str = int(fields[0])
            except ValueError:
                index = fields[0]
            try:
                vram_mib: float | str = float(fields[2])
            except ValueError:
                vram_mib = fields[2]
            devices.append(
                {
                    "index": index,
                    "name": fields[1],
                    "vram_total_mib": vram_mib,
                    "driver_version": fields[3],
                }
            )
    return {
        "available": result.returncode == 0,
        "devices": devices,
        "return_code": result.returncode,
        "error": result.stderr.strip() or None,
    }


def _torch_environment() -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "installed": False,
        "version": None,
        "torchvision_version": None,
        "cuda_build": None,
        "cuda_available": False,
        "cudnn_version": None,
        "devices": [],
        "functional_test_attempted": False,
    }
    with suppress(importlib.metadata.PackageNotFoundError):
        evidence["torchvision_version"] = importlib.metadata.version("torchvision")
    try:
        torch = importlib.import_module("torch")
        evidence["installed"] = True
        evidence["version"] = str(getattr(torch, "__version__", "unknown"))
        evidence["cuda_build"] = getattr(getattr(torch, "version", None), "cuda", None)
        cuda_available = bool(torch.cuda.is_available())
        evidence["cuda_available"] = cuda_available
        cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
        evidence["cudnn_version"] = cudnn.version() if cudnn is not None else None
        if cuda_available:
            devices: list[dict[str, Any]] = []
            for index in range(int(torch.cuda.device_count())):
                properties = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": str(properties.name),
                        "vram_total_bytes": int(properties.total_memory),
                        "vram_total_mib": round(int(properties.total_memory) / (1024**2), 2),
                        "compute_capability": list(torch.cuda.get_device_capability(index)),
                    }
                )
            evidence["devices"] = devices
    except Exception as exc:
        evidence["import_or_probe_error"] = f"{type(exc).__name__}: {exc}"
    return evidence


def _memory_environment() -> dict[str, Any]:
    try:
        psutil = importlib.import_module("psutil")
        memory = psutil.virtual_memory()
        return {
            "total_bytes": int(memory.total),
            "available_bytes": int(memory.available),
            "total_gib": round(int(memory.total) / (1024**3), 2),
            "available_gib": round(int(memory.available) / (1024**3), 2),
            "source": "psutil",
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "source": None}


def capture_environment(project_root: str | Path | None = None) -> dict[str, Any]:
    """Capture exact packages and non-mutating hardware/runtime evidence for a run."""

    root = Path(project_root or Path.cwd()).resolve()
    disk = shutil.disk_usage(root)
    torch_evidence = _torch_environment()
    nvidia_evidence = _nvidia_environment()
    return {
        "captured_at_utc": utc_now(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": _installed_distribution_versions(),
        "pytorch": torch_evidence,
        "cuda": {
            "available": torch_evidence["cuda_available"],
            "build_runtime": torch_evidence["cuda_build"],
            "cudnn_version": torch_evidence["cudnn_version"],
            "functional_test_attempted": False,
        },
        "gpu": {
            "pytorch_devices": torch_evidence["devices"],
            "nvidia_smi": nvidia_evidence,
        },
        "ram": _memory_environment(),
        "disk": {
            "path": str(root),
            "total_bytes": int(disk.total),
            "used_bytes": int(disk.used),
            "free_bytes": int(disk.free),
        },
        "process": {"pid": os.getpid(), "working_directory": str(Path.cwd().resolve())},
    }


def _normalise_name(value: str) -> str:
    normalised = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-._")
    if not normalised:
        raise ValueError("experiment name must contain a letter or digit")
    return normalised[:64]


def generate_run_id(experiment_name: str, *, timestamp: datetime | None = None) -> str:
    """Generate a sortable, collision-resistant run identifier."""

    moment = (timestamp or datetime.now(UTC)).astimezone(UTC)
    stamp = moment.strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{stamp}_{_normalise_name(experiment_name)}_{uuid.uuid4().hex[:10]}"


def create_run_directory(
    runs_root: str | Path,
    experiment_name: str = "run",
    *,
    run_id: str | None = None,
) -> Path:
    """Create a new run directory, refusing any existing identifier."""

    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    identifier = run_id or generate_run_id(experiment_name)
    if Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError("run_id must be a single safe path component")
    destination = root / identifier
    try:
        destination.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"run directory already exists and will not be overwritten: {destination}"
        ) from exc
    return destination


def is_run_immutable(run_directory: str | Path) -> bool:
    """Return whether the run carries a completed/failed immutable marker."""

    return (Path(run_directory) / IMMUTABLE_MARKER).is_file()


def sealed_run_ancestor(path: str | Path) -> Path | None:
    """Find the nearest immutable/sealed run containing *path*, if any."""

    resolved = Path(path).resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / IMMUTABLE_MARKER).is_file() or (
            candidate / ARTIFACT_MANIFEST_FILENAME
        ).is_file():
            return candidate
    return None


def assert_run_mutable(run_directory: str | Path) -> None:
    """Raise before any API write to a terminal run."""

    run_path = Path(run_directory).resolve()
    if is_run_immutable(run_path):
        raise PermissionError(f"run is immutable and cannot be modified: {run_path}")


def write_run_status(
    run_directory: str | Path,
    status: str,
    **details: Any,
) -> Path:
    """Atomically save the current run status while it remains mutable."""

    run_path = Path(run_directory)
    assert_run_mutable(run_path)
    payload = {"status": status, "updated_at_utc": utc_now(), **details}
    return atomic_write_json(run_path / STATUS_FILENAME, payload)


@contextmanager
def _registry_lock(
    registry_path: Path,
    *,
    timeout_seconds: float = 10.0,
) -> Iterator[None]:
    lock_path = registry_path.with_name(f".{registry_path.name}.lock")
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    owner_identity: tuple[int, int] | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"{os.getpid()} {utc_now()}\n".encode())
            os.fsync(descriptor)
            lock_stat = os.fstat(descriptor)
            owner_identity = (lock_stat.st_dev, lock_stat.st_ino)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for run registry lock: {lock_path}"
                ) from None
            time.sleep(0.05)
    try:
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        current_identity: tuple[int, int] | None
        try:
            current_stat = lock_path.stat()
            current_identity = (current_stat.st_dev, current_stat.st_ino)
        except FileNotFoundError:
            current_identity = None
        if owner_identity is not None and current_identity == owner_identity:
            lock_path.unlink(missing_ok=True)


def append_registry_row(
    registry_path: str | Path,
    row: Mapping[str, Any],
) -> Path:
    """Append exactly one CSV record without rewriting existing records."""

    destination = Path(registry_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    unknown = set(row).difference(REGISTRY_COLUMNS)
    if unknown:
        raise ValueError(f"unknown registry columns: {sorted(unknown)}")
    with _registry_lock(destination):
        needs_header = not destination.exists() or destination.stat().st_size == 0
        with destination.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS, extrasaction="raise")
            if needs_header:
                writer.writeheader()
            writer.writerow({column: row.get(column, "") for column in REGISTRY_COLUMNS})
            handle.flush()
            os.fsync(handle.fileno())
    return destination


append_run_registry = append_registry_row


def append_integrity_record(
    integrity_registry_path: str | Path,
    record: Mapping[str, Any],
) -> Path:
    """Append a strict JSON integrity record without rewriting prior records."""

    destination = Path(integrity_registry_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        dict(record),
        allow_nan=False,
        default=_json_default,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with (
        _registry_lock(destination),
        destination.open("a", encoding="utf-8", newline="\n") as handle,
    ):
        handle.write(f"{line}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return destination


def _config_seed(config: Mapping[str, Any], name: str) -> Any:
    seeds = config.get("seed")
    return seeds.get(name, "") if isinstance(seeds, Mapping) else ""


def _git_registry_value(state: Mapping[str, Any]) -> str:
    if not state.get("available"):
        return "unavailable"
    commit = state.get("commit") or "unborn"
    suffix = "+dirty" if state.get("dirty") else "+clean"
    return f"{commit}{suffix}"


@dataclass(slots=True)
class RuntimeTimer:
    """A monotonic runtime timer with an explicit wall-clock start timestamp."""

    started_at_utc: str = field(default_factory=utc_now)
    _started_monotonic: float = field(default_factory=time.perf_counter, repr=False)

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.perf_counter() - self._started_monotonic)


@dataclass(frozen=True, slots=True)
class IntegrityVerification:
    """Result of recomputing a sealed run's canonical artifact manifest."""

    valid: bool
    run_id: str | None
    expected_root_sha256: str | None
    actual_root_sha256: str | None
    missing_paths: tuple[str, ...]
    added_paths: tuple[str, ...]
    changed_paths: tuple[str, ...]
    registry_record_present: bool
    errors: tuple[str, ...]

    def __bool__(self) -> bool:
        return self.valid


def verify_run_integrity(run_directory: str | Path) -> IntegrityVerification:
    """Detect direct artifact edits, additions, deletions, or a broken seal."""

    run_path = Path(run_directory).resolve()
    manifest_path = run_path / ARTIFACT_MANIFEST_FILENAME
    marker_path = run_path / IMMUTABLE_MARKER
    errors: list[str] = []
    if not run_path.is_dir():
        return IntegrityVerification(
            valid=False,
            run_id=None,
            expected_root_sha256=None,
            actual_root_sha256=None,
            missing_paths=(),
            added_paths=(),
            changed_paths=(),
            registry_record_present=False,
            errors=(f"run directory does not exist: {run_path}",),
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return IntegrityVerification(
            valid=False,
            run_id=None,
            expected_root_sha256=None,
            actual_root_sha256=None,
            missing_paths=(),
            added_paths=(),
            changed_paths=(),
            registry_record_present=False,
            errors=(f"missing or invalid integrity manifest/marker: {exc}",),
        )
    if not isinstance(manifest, Mapping) or not isinstance(marker, Mapping):
        errors.append("integrity manifest and marker must be JSON objects")
        manifest = manifest if isinstance(manifest, Mapping) else {}
        marker = marker if isinstance(marker, Mapping) else {}
    run_id_value = manifest.get("run_id")
    run_id = str(run_id_value) if run_id_value is not None else None
    expected_root_value = manifest.get("artifact_root_sha256")
    expected_root = str(expected_root_value) if expected_root_value is not None else None
    expected_records_value = manifest.get("artifacts")
    expected_records = (
        [dict(record) for record in expected_records_value if isinstance(record, Mapping)]
        if isinstance(expected_records_value, list)
        else []
    )
    if len(expected_records) != (
        len(expected_records_value) if isinstance(expected_records_value, list) else -1
    ):
        errors.append("artifact manifest contains invalid records")
    if manifest.get("artifact_count") != len(expected_records):
        errors.append("artifact manifest count disagrees with its artifact records")
    try:
        actual_records = _artifact_records(run_path)
        actual_root = _artifact_root_sha256(actual_records)
    except (OSError, ValueError) as exc:
        actual_records = []
        actual_root = None
        errors.append(str(exc))
    expected_by_path = {str(record.get("path")): record for record in expected_records}
    actual_by_path = {str(record.get("path")): record for record in actual_records}
    missing = tuple(sorted(set(expected_by_path).difference(actual_by_path)))
    added = tuple(sorted(set(actual_by_path).difference(expected_by_path)))
    changed = tuple(
        sorted(
            path
            for path in set(expected_by_path).intersection(actual_by_path)
            if expected_by_path[path].get("size_bytes") != actual_by_path[path].get("size_bytes")
            or expected_by_path[path].get("sha256") != actual_by_path[path].get("sha256")
        )
    )
    manifest_sha256 = sha256_file(manifest_path) if manifest_path.is_file() else None
    manifest_status = manifest.get("status")
    if manifest_status not in {"completed", "failed"}:
        errors.append("artifact manifest has an invalid terminal status")
    if marker.get("run_id") != run_id:
        errors.append("immutable marker run ID disagrees with artifact manifest")
    if marker.get("status") != manifest_status:
        errors.append("immutable marker status disagrees with artifact manifest")
    marker_run_path = marker.get("run_path")
    if not isinstance(marker_run_path, str) or Path(marker_run_path).resolve() != run_path:
        errors.append("immutable marker does not bind the exact sealed run path")
    if marker.get("artifact_root_sha256") != expected_root:
        errors.append("immutable marker root digest disagrees with artifact manifest")
    if marker.get("artifact_manifest_sha256") != manifest_sha256:
        errors.append("immutable marker manifest digest is invalid")
    if marker.get("artifact_count") != len(expected_records):
        errors.append("immutable marker artifact count disagrees with artifact manifest")
    matching_registry_record_count = 0
    marker_integrity_registry = marker.get("integrity_registry")
    integrity_registry = (
        Path(marker_integrity_registry).resolve()
        if isinstance(marker_integrity_registry, str) and marker_integrity_registry
        else None
    )
    if integrity_registry is None:
        errors.append("immutable marker lacks its exact integrity registry path")
    elif integrity_registry.is_file():
        try:
            with _registry_lock(integrity_registry):
                content = integrity_registry.read_text(encoding="utf-8")
            lines = content.split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            for line in lines:
                record = json.loads(line)
                if not isinstance(record, Mapping):
                    raise ValueError("integrity registry records must be JSON objects")
                if (
                    record.get("run_id") == run_id
                    and record.get("status") == manifest_status
                    and isinstance(record.get("run_path"), str)
                    and Path(str(record["run_path"])).resolve() == run_path
                    and record.get("artifact_count") == len(expected_records)
                    and record.get("artifact_root_sha256") == expected_root
                    and record.get("artifact_manifest_sha256") == manifest_sha256
                ):
                    matching_registry_record_count += 1
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"integrity registry is invalid: {exc}")
    registry_record_present = matching_registry_record_count == 1
    if matching_registry_record_count > 1:
        errors.append("multiple matching append-only integrity registry records are present")
    if not registry_record_present:
        errors.append("matching append-only integrity registry record is absent")
    valid = (
        not errors and not missing and not added and not changed and actual_root == expected_root
    )
    return IntegrityVerification(
        valid=valid,
        run_id=run_id,
        expected_root_sha256=expected_root,
        actual_root_sha256=actual_root,
        missing_paths=missing,
        added_paths=added,
        changed_paths=changed,
        registry_record_present=registry_record_present,
        errors=tuple(errors),
    )


_RUN_DISPOSITION_FIELDS = {
    "schema_version",
    "sequence",
    "event_type",
    "recorded_at_utc",
    "run_id",
    "run_path",
    "terminal_status",
    "scientific_stage_eligible",
    "artifact_root_sha256",
    "artifact_manifest_sha256",
    "reason_code",
    "reason",
    "previous_record_sha256",
    "record_sha256",
}
_LOWER_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REASON_CODE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,127}$")
_RUN_DISPOSITION_ANCHOR_FIELDS = {
    "schema_version",
    "ledger_filename",
    "chain_algorithm",
    "record_count",
    "head_record_sha256",
    "ledger_sha256",
}


def _run_disposition_record_sha256(record: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    encoded = json.dumps(
        unsigned,
        allow_nan=False,
        default=_json_default,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_run_disposition_record(
    raw: Any,
    *,
    line_number: int,
    previous_record_sha256: str | None,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"run disposition line {line_number} must be a JSON object")
    record = dict(raw)
    if set(record) != _RUN_DISPOSITION_FIELDS:
        missing = sorted(_RUN_DISPOSITION_FIELDS.difference(record))
        unknown = sorted(set(record).difference(_RUN_DISPOSITION_FIELDS))
        raise ValueError(
            f"run disposition line {line_number} has invalid fields: "
            f"missing={missing}, unknown={unknown}"
        )
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValueError(f"run disposition line {line_number} has unsupported schema_version")
    if type(record["sequence"]) is not int or record["sequence"] != line_number:
        raise ValueError(f"run disposition line {line_number} has an invalid sequence")
    if record["event_type"] != "eligibility_withdrawn":
        raise ValueError(
            f"run disposition line {line_number} has an unsupported event_type; "
            "scientific eligibility cannot be reinstated"
        )
    if record["terminal_status"] != "completed":
        raise ValueError(
            f"run disposition line {line_number} must bind terminal_status='completed'"
        )
    if record["scientific_stage_eligible"] is not False:
        raise ValueError(
            f"run disposition line {line_number} must permanently withdraw stage eligibility"
        )
    recorded_at = record["recorded_at_utc"]
    if not isinstance(recorded_at, str) or not recorded_at.endswith("Z"):
        raise ValueError(f"run disposition line {line_number} has an invalid UTC timestamp")
    try:
        timestamp = datetime.fromisoformat(recorded_at.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError(
            f"run disposition line {line_number} has an invalid UTC timestamp"
        ) from exc
    if timestamp.utcoffset() != UTC.utcoffset(timestamp):
        raise ValueError(f"run disposition line {line_number} timestamp is not UTC")
    for field_name in ("run_id", "run_path", "reason"):
        value = record[field_name]
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"run disposition line {line_number} has an invalid {field_name}")
    if not Path(record["run_path"]).is_absolute():
        raise ValueError(f"run disposition line {line_number} run_path must be absolute")
    reason_code = record["reason_code"]
    if not isinstance(reason_code, str) or _REASON_CODE_PATTERN.fullmatch(reason_code) is None:
        raise ValueError(f"run disposition line {line_number} has an invalid reason_code")
    for field_name in (
        "artifact_root_sha256",
        "artifact_manifest_sha256",
        "record_sha256",
    ):
        value = record[field_name]
        if not isinstance(value, str) or _LOWER_SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"run disposition line {line_number} has an invalid {field_name}")
    if record["previous_record_sha256"] != previous_record_sha256:
        raise ValueError(f"run disposition line {line_number} breaks the append-only hash chain")
    if record["record_sha256"] != _run_disposition_record_sha256(record):
        raise ValueError(f"run disposition line {line_number} has an invalid record_sha256")
    return record


def _read_run_dispositions_unlocked(registry_path: Path) -> tuple[dict[str, Any], ...]:
    if not os.path.lexists(registry_path):
        return ()
    if not registry_path.is_file():
        raise ValueError(f"run disposition registry is not a file: {registry_path}")
    records: list[dict[str, Any]] = []
    previous_record_sha256: str | None = None
    try:
        content = registry_path.read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"run disposition registry is unreadable: {registry_path}: {exc}") from exc
    if content and not content.endswith("\n"):
        raise ValueError("run disposition registry has a truncated final record")
    lines = content[:-1].split("\n") if content else []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ValueError(f"run disposition line {line_number} is blank")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"run disposition line {line_number} is invalid JSON") from exc
        record = _validate_run_disposition_record(
            raw,
            line_number=line_number,
            previous_record_sha256=previous_record_sha256,
        )
        records.append(record)
        previous_record_sha256 = record["record_sha256"]
    run_ids = [record["run_id"] for record in records]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("run disposition registry contains more than one event for a run")
    return tuple(records)


def _run_disposition_anchor_path(registry_path: Path) -> Path:
    return registry_path.with_name(RUN_DISPOSITION_ANCHOR_FILENAME)


def _run_disposition_ledger_bytes(registry_path: Path) -> bytes:
    if not os.path.lexists(registry_path):
        return b""
    if not registry_path.is_file():
        raise ValueError(f"run disposition registry is not a file: {registry_path}")
    try:
        return registry_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"run disposition registry is unreadable: {registry_path}: {exc}") from exc


def _run_disposition_anchor_payload(
    records: Sequence[Mapping[str, Any]],
    ledger_bytes: bytes,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ledger_filename": RUN_DISPOSITION_REGISTRY_FILENAME,
        "chain_algorithm": "sha256(canonical-json-record-with-previous-head)",
        "record_count": len(records),
        "head_record_sha256": records[-1]["record_sha256"] if records else None,
        "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
    }


def _read_run_disposition_anchor_unlocked(anchor_path: Path) -> dict[str, Any]:
    if not os.path.lexists(anchor_path):
        raise ValueError(
            "run disposition anchor is missing; eligibility fails closed because ledger "
            "deletion cannot be distinguished from an empty history"
        )
    if not anchor_path.is_file():
        raise ValueError(f"run disposition anchor is not a file: {anchor_path}")
    try:
        raw = json.loads(anchor_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"run disposition anchor is unreadable or invalid: {exc}") from exc
    if not isinstance(raw, Mapping) or set(raw) != _RUN_DISPOSITION_ANCHOR_FIELDS:
        raise ValueError("run disposition anchor has an invalid schema")
    anchor = dict(raw)
    if type(anchor["schema_version"]) is not int or anchor["schema_version"] != 1:
        raise ValueError("run disposition anchor has an unsupported schema_version")
    if anchor["ledger_filename"] != RUN_DISPOSITION_REGISTRY_FILENAME:
        raise ValueError("run disposition anchor binds an unexpected ledger filename")
    if anchor["chain_algorithm"] != "sha256(canonical-json-record-with-previous-head)":
        raise ValueError("run disposition anchor binds an unexpected chain algorithm")
    if type(anchor["record_count"]) is not int or anchor["record_count"] < 0:
        raise ValueError("run disposition anchor has an invalid record_count")
    head = anchor["head_record_sha256"]
    if head is not None and (
        not isinstance(head, str) or _LOWER_SHA256_PATTERN.fullmatch(head) is None
    ):
        raise ValueError("run disposition anchor has an invalid head_record_sha256")
    ledger_sha = anchor["ledger_sha256"]
    if not isinstance(ledger_sha, str) or _LOWER_SHA256_PATTERN.fullmatch(ledger_sha) is None:
        raise ValueError("run disposition anchor has an invalid ledger_sha256")
    return anchor


def _require_run_disposition_anchor_matches_unlocked(
    registry_path: Path,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    anchor_path = _run_disposition_anchor_path(registry_path)
    anchor = _read_run_disposition_anchor_unlocked(anchor_path)
    expected = _run_disposition_anchor_payload(
        records,
        _run_disposition_ledger_bytes(registry_path),
    )
    if anchor != expected:
        raise ValueError(
            "run disposition anchor does not match the ledger head/count/content; "
            "eligibility fails closed"
        )
    return anchor


def _write_run_disposition_anchor_cas(
    anchor_path: Path,
    payload: Mapping[str, Any],
    *,
    expected_previous_sha256: str | None,
) -> None:
    if expected_previous_sha256 is None:
        if os.path.lexists(anchor_path):
            raise FileExistsError(f"run disposition anchor already exists: {anchor_path}")
    else:
        if not anchor_path.is_file() or sha256_file(anchor_path) != expected_previous_sha256:
            raise ValueError("run disposition anchor changed before compare-and-swap publication")
    atomic_write_json(anchor_path, payload)


def _ensure_run_disposition_anchor(runs_root: Path) -> Path:
    root = runs_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    registry_path = root / RUN_DISPOSITION_REGISTRY_FILENAME
    anchor_path = root / RUN_DISPOSITION_ANCHOR_FILENAME
    with _registry_lock(registry_path):
        records = _read_run_dispositions_unlocked(registry_path)
        if os.path.lexists(anchor_path):
            _require_run_disposition_anchor_matches_unlocked(registry_path, records)
        else:
            if records or _run_disposition_ledger_bytes(registry_path):
                raise ValueError(
                    "cannot initialize a missing anchor for a non-empty disposition ledger"
                )
            lock_path = registry_path.with_name(f".{registry_path.name}.lock")
            prior_run_evidence = [
                path for path in root.iterdir() if path != lock_path and path.name != ".gitkeep"
            ]
            if prior_run_evidence:
                raise ValueError(
                    "run disposition anchor is missing beside existing run evidence; "
                    "automatic reinitialization would permit eligibility reinstatement"
                )
            _write_run_disposition_anchor_cas(
                anchor_path,
                _run_disposition_anchor_payload((), b""),
                expected_previous_sha256=None,
            )
    return anchor_path


def read_run_dispositions(
    registry_path: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Read and verify the complete append-only scientific-disposition hash chain."""

    source = Path(registry_path)
    with _registry_lock(source):
        records = _read_run_dispositions_unlocked(source)
        _require_run_disposition_anchor_matches_unlocked(source, records)
        return records


def _completed_run_disposition_binding(
    run_directory: str | Path,
    *,
    integrity: IntegrityVerification | None = None,
) -> dict[str, str]:
    run_path = Path(run_directory).resolve()
    verification = integrity or verify_run_integrity(run_path)
    if not verification.valid or not verification.registry_record_present:
        raise ValueError(
            "run eligibility can be withdrawn only from an integrity-valid, "
            f"registry-backed sealed run: {verification.errors}"
        )
    manifest_path = run_path / ARTIFACT_MANIFEST_FILENAME
    marker_path = run_path / IMMUTABLE_MARKER
    status_path = run_path / STATUS_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"sealed run terminal evidence is missing or invalid: {exc}") from exc
    if not all(isinstance(value, Mapping) for value in (manifest, marker, status)):
        raise ValueError("sealed run manifest, marker, and status must be JSON objects")
    run_id = verification.run_id
    artifact_root = verification.expected_root_sha256
    if not isinstance(run_id, str) or not run_id or run_id != run_path.name:
        raise ValueError("sealed run ID does not exactly match its directory name")
    if not isinstance(artifact_root, str) or _LOWER_SHA256_PATTERN.fullmatch(artifact_root) is None:
        raise ValueError("sealed run lacks a valid artifact root SHA-256")
    manifest_sha256 = sha256_file(manifest_path)
    for role, payload in (("manifest", manifest), ("marker", marker), ("status", status)):
        if payload.get("run_id") != run_id:
            raise ValueError(f"sealed run {role} does not bind the exact run ID")
        if payload.get("status") != "completed":
            raise ValueError(
                "scientific eligibility can be withdrawn only from a terminal completed run"
            )
    if manifest.get("artifact_root_sha256") != artifact_root:
        raise ValueError("sealed run manifest does not bind the verified artifact root")
    if marker.get("artifact_root_sha256") != artifact_root:
        raise ValueError("sealed run marker does not bind the verified artifact root")
    if marker.get("artifact_manifest_sha256") != manifest_sha256:
        raise ValueError("sealed run marker does not bind the exact artifact manifest")
    marker_run_path = marker.get("run_path")
    if not isinstance(marker_run_path, str) or Path(marker_run_path).resolve() != run_path:
        raise ValueError("sealed run marker does not bind the exact run path")
    return {
        "run_id": run_id,
        "run_path": str(run_path),
        "terminal_status": "completed",
        "artifact_root_sha256": artifact_root,
        "artifact_manifest_sha256": manifest_sha256,
    }


def _run_mutation_lock_target(run_path: Path) -> Path:
    return run_path.parent / f"{run_path.name}.mutation"


def withdraw_run_eligibility(
    run_directory: str | Path,
    *,
    reason_code: str,
    reason: str,
) -> dict[str, Any]:
    """Permanently withdraw a completed sealed run from scientific stage eligibility."""

    clean_reason_code = reason_code.strip()
    clean_reason = reason.strip()
    if clean_reason_code != reason_code or _REASON_CODE_PATTERN.fullmatch(reason_code) is None:
        raise ValueError(
            "reason_code must contain 2-128 lowercase letters, digits, dots, underscores, or dashes"
        )
    if clean_reason != reason or not reason:
        raise ValueError("reason must be non-empty and must not have surrounding whitespace")
    run_path = Path(run_directory).resolve()
    registry_path = run_path.parent / RUN_DISPOSITION_REGISTRY_FILENAME
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        _registry_lock(_run_mutation_lock_target(run_path)),
        _registry_lock(registry_path),
    ):
        binding = _completed_run_disposition_binding(run_path)
        records = _read_run_dispositions_unlocked(registry_path)
        anchor = _require_run_disposition_anchor_matches_unlocked(registry_path, records)
        if any(record["run_id"] == binding["run_id"] for record in records):
            raise ValueError(
                f"scientific stage eligibility is already withdrawn for run {binding['run_id']}"
            )
        record: dict[str, Any] = {
            "schema_version": 1,
            "sequence": len(records) + 1,
            "event_type": "eligibility_withdrawn",
            "recorded_at_utc": utc_now(),
            **binding,
            "scientific_stage_eligible": False,
            "reason_code": reason_code,
            "reason": reason,
            "previous_record_sha256": records[-1]["record_sha256"] if records else None,
        }
        record["record_sha256"] = _run_disposition_record_sha256(record)
        _validate_run_disposition_record(
            record,
            line_number=len(records) + 1,
            previous_record_sha256=record["previous_record_sha256"],
        )
        encoded = json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        previous_ledger = _run_disposition_ledger_bytes(registry_path)
        next_ledger = previous_ledger + encoded + b"\n"
        previous_anchor_sha256 = sha256_file(_run_disposition_anchor_path(registry_path))
        atomic_write_bytes(registry_path, next_ledger)
        _write_run_disposition_anchor_cas(
            _run_disposition_anchor_path(registry_path),
            _run_disposition_anchor_payload((*records, record), next_ledger),
            expected_previous_sha256=previous_anchor_sha256,
        )
        committed = _read_run_dispositions_unlocked(registry_path)
        _require_run_disposition_anchor_matches_unlocked(registry_path, committed)
        if committed != (*records, record) or anchor["record_count"] + 1 != len(committed):
            raise RuntimeError("run disposition transaction failed exact readback verification")
    return record


_RUN_STAGE_ATTESTATION_FIELDS = {
    "schema_version",
    "sequence",
    "event_type",
    "recorded_at_utc",
    "run_id",
    "run_path",
    "terminal_status",
    "scientific_stage_eligible",
    "completion_stage",
    "artifact_root_sha256",
    "artifact_manifest_sha256",
    "completion_evidence_sha256",
    "verification",
    "verification_sha256",
    "previous_record_sha256",
    "record_sha256",
}

_PRIMARY_STAGE_EXPERIMENT_POLICIES = {
    "pannuke_primary_frozen_feature_benchmark": "primary_postseal_attestation_v2",
    "pannuke_primary_finalization_successor": (
        "primary_finalization_successor_postseal_attestation_v2"
    ),
    "pannuke_primary_orphan_recovery": ("primary_orphan_recovery_postseal_attestation_v1"),
}
_PRIMARY_STAGE_VERIFICATION_FIELDS = {
    "schema_version",
    "policy",
    "experiment_name",
    "run_id",
    "run_path",
    "completion_stage",
    "first_integrity_root_sha256",
    "final_integrity_root_sha256",
    "artifact_manifest_sha256",
    "completion_evidence_sha256",
    "matrix_plan_sha256",
    "execution_controls_sha256",
    "cell_index_sha256",
    "filesystem_readback_root_sha256",
    "primary_statistics_sha256",
    "primary_statistics_size_bytes",
    "primary_bootstrap_evidence_sha256",
    "primary_bootstrap_evidence_size_bytes",
    "primary_subgroups_sha256",
    "primary_subgroups_size_bytes",
    "primary_statistics_manifest_sha256",
    "primary_statistics_manifest_size_bytes",
    "primary_statistics_source_readback_root_sha256",
    "primary_statistics_comparison_count",
    "primary_restoration_index_sha256",
    "primary_restoration_readback_root_sha256",
    "retry_of_run_id",
    "lineage_binding_sha256",
    "authorization_binding_sha256",
    "semantic_verification_status",
}
_PRIMARY_STAGE_ATTESTATION_TOKEN = object()
_RUN_STAGE_ELIGIBILITY_RECEIPT_TOKEN = object()
_VALIDATED_RUN_STAGE_ATTESTATION_TOKEN = object()
_LIFECYCLE_QUALIFICATION_EXPERIMENT = "lifecycle_qualification_verification"
_LIFECYCLE_QUALIFICATION_POLICY = "lifecycle_readiness_postseal_qualification_v1"
_LIFECYCLE_QUALIFICATION_EVENT = "postseal_lifecycle_qualification_attested"
_LIFECYCLE_QUALIFICATION_VERIFICATION_FIELDS = {
    "schema_version",
    "policy",
    "experiment_name",
    "run_id",
    "run_path",
    "first_integrity_root_sha256",
    "final_integrity_root_sha256",
    "artifact_manifest_sha256",
    "readiness_evidence_sha256",
    "qualification_binding_sha256",
    "readiness_record_sha256",
    "decision",
    "scientific_outcome",
    "project_completion_status_changed",
    "semantic_verification_status",
}
_LIFECYCLE_QUALIFICATION_ATTESTATION_TOKEN = object()
_EXTERNAL_VALIDATION_READY_EXPERIMENT = "external_validation_package"
_EXTERNAL_VALIDATION_READY_CANDIDATE_POLICY = "tracked_external_validation_ready_v1"
_EXTERNAL_VALIDATION_READY_ATTESTATION_POLICY = "external_validation_ready_postseal_attestation_v1"
_EXTERNAL_VALIDATION_READY_GATE_FILENAME = "external_validation_execution_gate.json"
_EXTERNAL_VALIDATION_READY_COMPLETION_FILENAME = "completion_evidence.json"
_EXTERNAL_VALIDATION_READY_BUNDLE_RELATIVE_PATH = "review_bundle"
_EXTERNAL_VALIDATION_READY_TECHNICAL_INSPECTION_FILENAME = "technical_inspection_evidence.json"
_EXTERNAL_VALIDATION_READY_INSPECTION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_EXTERNAL_VALIDATION_READY_INSPECTION_UTC_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"
)
_EXTERNAL_VALIDATION_READY_INSPECTION_FIELDS = {
    "schema_version",
    "evidence_kind",
    "inspection_protocol",
    "inspection_id",
    "inspector_id",
    "inspected_at_utc",
    "decision",
    "bundle_root_sha256",
    "item_count",
    "asset_count",
    "expert_response_count",
    "annotation_judgment_data_present",
    "annotation_evaluations_performed",
    "outcomes_used_for_tuning",
    "checks",
}
_EXTERNAL_VALIDATION_READY_INSPECTION_CHECKS = {
    "review_html_opened",
    "item_count_200_confirmed",
    "asset_count_600_confirmed",
    "public_identifiers_blinded",
    "ranking_source_hidden",
    "model_suggestion_absent",
    "response_template_blank",
}
_EXTERNAL_VALIDATION_READY_ATTESTATION_TOKEN = object()
_EXTERNAL_VALIDATION_READY_SHARED_FIELDS = {
    "schema_version",
    "policy",
    "experiment_name",
    "run_id",
    "completion_stage",
    "study_outcome_eligible",
    "post_seal_attestation_required",
    "external_validation_complete_claimed",
    "item_count",
    "asset_count",
    "expert_response_count",
    "contract_sha256",
    "cohort_payload_sha256",
    "bundle_root_sha256",
    "public_tree_root_sha256",
    "private_tree_root_sha256",
    "raw_inventory_sha256",
    "canonical_pannuke_manifest_sha256",
    "original_ranking_sha256",
    "original_audit_run_directory",
    "original_audit_experiment_name",
    "original_audit_run_id",
    "original_audit_artifact_root_sha256",
    "original_audit_eligibility_evidence_sha256",
    "confirmatory_run_directory",
    "confirmatory_run_id",
    "confirmatory_artifact_root_sha256",
    "confirmatory_completion_evidence_sha256",
    "confirmatory_stage_attestation_record_sha256",
    "confirmatory_stage_attestation_verification_sha256",
    "technical_inspection_evidence_sha256",
}
_EXTERNAL_VALIDATION_READY_GATE_FIELDS = frozenset(_EXTERNAL_VALIDATION_READY_SHARED_FIELDS)
_EXTERNAL_VALIDATION_READY_COMPLETION_FIELDS = frozenset(
    {*_EXTERNAL_VALIDATION_READY_SHARED_FIELDS, "external_validation_execution_gate_sha256"}
)
_EXTERNAL_VALIDATION_READY_VERIFICATION_FIELDS = {
    "schema_version",
    "policy",
    "experiment_name",
    "run_id",
    "run_path",
    "completion_stage",
    "first_integrity_root_sha256",
    "final_integrity_root_sha256",
    "artifact_manifest_sha256",
    "completion_evidence_sha256",
    "external_validation_execution_gate_sha256",
    "item_count",
    "asset_count",
    "expert_response_count",
    "contract_sha256",
    "cohort_payload_sha256",
    "bundle_root_sha256",
    "public_tree_root_sha256",
    "private_tree_root_sha256",
    "raw_inventory_sha256",
    "canonical_pannuke_manifest_sha256",
    "original_ranking_sha256",
    "original_audit_run_directory",
    "original_audit_experiment_name",
    "original_audit_run_id",
    "original_audit_artifact_root_sha256",
    "original_audit_eligibility_evidence_sha256",
    "confirmatory_run_directory",
    "confirmatory_run_id",
    "confirmatory_artifact_root_sha256",
    "confirmatory_completion_evidence_sha256",
    "confirmatory_stage_attestation_record_sha256",
    "confirmatory_stage_attestation_verification_sha256",
    "technical_inspection_evidence_sha256",
    "semantic_verification_status",
}


@dataclass(frozen=True, slots=True)
class PrimaryStageAttestationVerification:
    """Non-forgeable in-process proof assembled from typed primary verifiers."""

    policy: str
    experiment_name: str
    run_id: str
    run_path: str
    completion_stage: str
    first_integrity_root_sha256: str
    final_integrity_root_sha256: str
    artifact_manifest_sha256: str
    completion_evidence_sha256: str
    matrix_plan_sha256: str
    execution_controls_sha256: str
    cell_index_sha256: str
    filesystem_readback_root_sha256: str
    primary_statistics_sha256: str
    primary_statistics_size_bytes: int
    primary_bootstrap_evidence_sha256: str
    primary_bootstrap_evidence_size_bytes: int
    primary_subgroups_sha256: str
    primary_subgroups_size_bytes: int
    primary_statistics_manifest_sha256: str
    primary_statistics_manifest_size_bytes: int
    primary_statistics_source_readback_root_sha256: str
    primary_statistics_comparison_count: int
    primary_restoration_index_sha256: str
    primary_restoration_readback_root_sha256: str
    retry_of_run_id: str | None
    lineage_binding_sha256: str | None
    authorization_binding_sha256: str | None
    semantic_verification_status: str = "passed"
    _attestation: object | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def valid(self) -> bool:
        return self._attestation is _PRIMARY_STAGE_ATTESTATION_TOKEN

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "policy": self.policy,
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "run_path": self.run_path,
            "completion_stage": self.completion_stage,
            "first_integrity_root_sha256": self.first_integrity_root_sha256,
            "final_integrity_root_sha256": self.final_integrity_root_sha256,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "completion_evidence_sha256": self.completion_evidence_sha256,
            "matrix_plan_sha256": self.matrix_plan_sha256,
            "execution_controls_sha256": self.execution_controls_sha256,
            "cell_index_sha256": self.cell_index_sha256,
            "filesystem_readback_root_sha256": self.filesystem_readback_root_sha256,
            "primary_statistics_sha256": self.primary_statistics_sha256,
            "primary_statistics_size_bytes": self.primary_statistics_size_bytes,
            "primary_bootstrap_evidence_sha256": self.primary_bootstrap_evidence_sha256,
            "primary_bootstrap_evidence_size_bytes": (self.primary_bootstrap_evidence_size_bytes),
            "primary_subgroups_sha256": self.primary_subgroups_sha256,
            "primary_subgroups_size_bytes": self.primary_subgroups_size_bytes,
            "primary_statistics_manifest_sha256": self.primary_statistics_manifest_sha256,
            "primary_statistics_manifest_size_bytes": (self.primary_statistics_manifest_size_bytes),
            "primary_statistics_source_readback_root_sha256": (
                self.primary_statistics_source_readback_root_sha256
            ),
            "primary_statistics_comparison_count": self.primary_statistics_comparison_count,
            "primary_restoration_index_sha256": self.primary_restoration_index_sha256,
            "primary_restoration_readback_root_sha256": (
                self.primary_restoration_readback_root_sha256
            ),
            "retry_of_run_id": self.retry_of_run_id,
            "lineage_binding_sha256": self.lineage_binding_sha256,
            "authorization_binding_sha256": self.authorization_binding_sha256,
            "semantic_verification_status": self.semantic_verification_status,
        }


@dataclass(frozen=True, slots=True)
class _RunStageEligibilityGuardState:
    """Private lease state invalidated when its mutation-lock guard exits."""

    owner_process_id: int
    owner_thread_id: int
    _active: bool = True

    @property
    def active(self) -> bool:
        return self._active

    @property
    def active_for_current_execution(self) -> bool:
        return (
            self._active
            and os.getpid() == self.owner_process_id
            and threading.get_ident() == self.owner_thread_id
        )

    def _revoke(self, token: object) -> None:
        if token is not _RUN_STAGE_ELIGIBILITY_RECEIPT_TOKEN:
            raise PermissionError("run-stage eligibility lease revocation token is invalid")
        object.__setattr__(self, "_active", False)


@dataclass(frozen=True, slots=True)
class RunStageEligibilityReceipt:
    """Opaque proof minted only from a freshly validated stage ledger record.

    ``valid`` proves that run tracking issued the receipt after the complete
    integrity, disposition, attestation-ledger, anchor, and payload checks.
    ``active_under_guard`` additionally proves that the issuing per-run mutation
    lock is still held.  A directly constructed or ``dataclasses.replace`` copy
    deliberately lacks both authorities.
    """

    run_directory: Path
    run_id: str
    completion_stage: str
    record_sha256: str
    verification_sha256: str
    _canonical_record_json: str = field(repr=False, compare=False)
    _guard_state: _RunStageEligibilityGuardState | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _attestation: object | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def valid(self) -> bool:
        return self._attestation is _RUN_STAGE_ELIGIBILITY_RECEIPT_TOKEN

    @property
    def active_under_guard(self) -> bool:
        return (
            self.valid
            and self._guard_state is not None
            and self._guard_state.active_for_current_execution
        )

    def require_active_authority(self) -> dict[str, Any]:
        """Return the record only to the process/thread owning the live guard."""

        if not self.active_under_guard:
            raise ValueError(
                "run-stage eligibility receipt is not active in its issuing process and thread"
            )
        return self.attestation_record()

    def attestation_record(self) -> dict[str, Any]:
        """Return an isolated copy of the exact validated ledger record."""

        if not self.valid:
            raise ValueError("run-stage eligibility receipt is not genuine")
        value = json.loads(self._canonical_record_json)
        if not isinstance(value, dict):  # pragma: no cover - issuer invariant
            raise RuntimeError("issued run-stage eligibility receipt is malformed")
        return value


@dataclass(frozen=True, slots=True)
class _ValidatedRunStageAttestation:
    """Private result of the complete mutation-locked eligibility validation."""

    run_directory: Path
    run_id: str
    completion_stage: str
    record_sha256: str
    verification_sha256: str
    canonical_record_json: str = field(repr=False, compare=False)
    _attestation: object | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def valid(self) -> bool:
        return self._attestation is _VALIDATED_RUN_STAGE_ATTESTATION_TOKEN

    def record(self) -> dict[str, Any]:
        if not self.valid:
            raise ValueError("validated run-stage attestation authority is not genuine")
        value = json.loads(self.canonical_record_json)
        if not isinstance(value, dict):  # pragma: no cover - issuer invariant
            raise RuntimeError("validated run-stage attestation authority is malformed")
        return value


@dataclass(frozen=True, slots=True)
class LifecycleQualificationAttestationVerification:
    """Non-forgeable proof of a freshly verified, non-scientific readiness run."""

    policy: str
    experiment_name: str
    run_id: str
    run_path: str
    first_integrity_root_sha256: str
    final_integrity_root_sha256: str
    artifact_manifest_sha256: str
    readiness_evidence_sha256: str
    qualification_binding_sha256: str
    readiness_record_sha256: str
    decision: str = "passed"
    scientific_outcome: bool = False
    project_completion_status_changed: bool = False
    semantic_verification_status: str = "passed"
    _attestation: object | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def valid(self) -> bool:
        return self._attestation is _LIFECYCLE_QUALIFICATION_ATTESTATION_TOKEN

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": self.policy,
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "run_path": self.run_path,
            "first_integrity_root_sha256": self.first_integrity_root_sha256,
            "final_integrity_root_sha256": self.final_integrity_root_sha256,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "readiness_evidence_sha256": self.readiness_evidence_sha256,
            "qualification_binding_sha256": self.qualification_binding_sha256,
            "readiness_record_sha256": self.readiness_record_sha256,
            "decision": self.decision,
            "scientific_outcome": self.scientific_outcome,
            "project_completion_status_changed": self.project_completion_status_changed,
            "semantic_verification_status": self.semantic_verification_status,
        }


@dataclass(frozen=True, slots=True)
class ExternalValidationReadyAttestationVerification:
    """Non-forgeable proof of one freshly verified, sealed M9 package run."""

    policy: str
    experiment_name: str
    run_id: str
    run_path: str
    completion_stage: str
    first_integrity_root_sha256: str
    final_integrity_root_sha256: str
    artifact_manifest_sha256: str
    completion_evidence_sha256: str
    external_validation_execution_gate_sha256: str
    item_count: int
    asset_count: int
    expert_response_count: int
    contract_sha256: str
    cohort_payload_sha256: str
    bundle_root_sha256: str
    public_tree_root_sha256: str
    private_tree_root_sha256: str
    raw_inventory_sha256: str
    canonical_pannuke_manifest_sha256: str
    original_ranking_sha256: str
    original_audit_run_directory: str
    original_audit_experiment_name: str
    original_audit_run_id: str
    original_audit_artifact_root_sha256: str
    original_audit_eligibility_evidence_sha256: str
    confirmatory_run_directory: str
    confirmatory_run_id: str
    confirmatory_artifact_root_sha256: str
    confirmatory_completion_evidence_sha256: str
    confirmatory_stage_attestation_record_sha256: str
    confirmatory_stage_attestation_verification_sha256: str
    technical_inspection_evidence_sha256: str
    semantic_verification_status: str = "passed"
    _confirmatory_receipt: RunStageEligibilityReceipt | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _material_refresh_authority: Any | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _ranking_path: Path | None = field(default=None, init=False, repr=False, compare=False)
    _attestation: object | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def valid(self) -> bool:
        return (
            self._attestation is _EXTERNAL_VALIDATION_READY_ATTESTATION_TOKEN
            and isinstance(self._confirmatory_receipt, RunStageEligibilityReceipt)
            and self._confirmatory_receipt.active_under_guard
            and self._material_refresh_authority is not None
            and isinstance(self._ranking_path, Path)
            and self._ranking_path.is_absolute()
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": self.policy,
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "run_path": self.run_path,
            "completion_stage": self.completion_stage,
            "first_integrity_root_sha256": self.first_integrity_root_sha256,
            "final_integrity_root_sha256": self.final_integrity_root_sha256,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "completion_evidence_sha256": self.completion_evidence_sha256,
            "external_validation_execution_gate_sha256": (
                self.external_validation_execution_gate_sha256
            ),
            "item_count": self.item_count,
            "asset_count": self.asset_count,
            "expert_response_count": self.expert_response_count,
            "contract_sha256": self.contract_sha256,
            "cohort_payload_sha256": self.cohort_payload_sha256,
            "bundle_root_sha256": self.bundle_root_sha256,
            "public_tree_root_sha256": self.public_tree_root_sha256,
            "private_tree_root_sha256": self.private_tree_root_sha256,
            "raw_inventory_sha256": self.raw_inventory_sha256,
            "canonical_pannuke_manifest_sha256": self.canonical_pannuke_manifest_sha256,
            "original_ranking_sha256": self.original_ranking_sha256,
            "original_audit_run_directory": self.original_audit_run_directory,
            "original_audit_experiment_name": self.original_audit_experiment_name,
            "original_audit_run_id": self.original_audit_run_id,
            "original_audit_artifact_root_sha256": self.original_audit_artifact_root_sha256,
            "original_audit_eligibility_evidence_sha256": (
                self.original_audit_eligibility_evidence_sha256
            ),
            "confirmatory_run_directory": self.confirmatory_run_directory,
            "confirmatory_run_id": self.confirmatory_run_id,
            "confirmatory_artifact_root_sha256": self.confirmatory_artifact_root_sha256,
            "confirmatory_completion_evidence_sha256": (
                self.confirmatory_completion_evidence_sha256
            ),
            "confirmatory_stage_attestation_record_sha256": (
                self.confirmatory_stage_attestation_record_sha256
            ),
            "confirmatory_stage_attestation_verification_sha256": (
                self.confirmatory_stage_attestation_verification_sha256
            ),
            "technical_inspection_evidence_sha256": self.technical_inspection_evidence_sha256,
            "semantic_verification_status": self.semantic_verification_status,
        }


def _validate_lifecycle_qualification_verification_payload(
    raw: Any,
    *,
    record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _LIFECYCLE_QUALIFICATION_VERIFICATION_FIELDS:
        raise ValueError("lifecycle qualification verification payload has invalid fields")
    payload = dict(raw)
    if (
        payload.get("schema_version") != 1
        or payload.get("policy") != _LIFECYCLE_QUALIFICATION_POLICY
        or payload.get("experiment_name") != _LIFECYCLE_QUALIFICATION_EXPERIMENT
        or payload.get("decision") != "passed"
        or payload.get("scientific_outcome") is not False
        or payload.get("project_completion_status_changed") is not False
        or payload.get("semantic_verification_status") != "passed"
    ):
        raise ValueError("lifecycle qualification verification payload has invalid semantics")
    for field_name in ("run_id", "run_path"):
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"lifecycle qualification verification has invalid {field_name}")
    if not Path(str(payload["run_path"])).is_absolute():
        raise ValueError("lifecycle qualification verification run_path is not absolute")
    for field_name in (
        "first_integrity_root_sha256",
        "final_integrity_root_sha256",
        "artifact_manifest_sha256",
        "readiness_evidence_sha256",
        "qualification_binding_sha256",
        "readiness_record_sha256",
    ):
        value = payload.get(field_name)
        if not isinstance(value, str) or _LOWER_SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"lifecycle qualification verification has invalid {field_name}")
    if payload["first_integrity_root_sha256"] != payload["final_integrity_root_sha256"]:
        raise ValueError("lifecycle qualification integrity roots differ")
    if record is not None:
        expected_record_bindings = {
            "run_id": payload["run_id"],
            "run_path": payload["run_path"],
            "completion_stage": None,
            "artifact_root_sha256": payload["final_integrity_root_sha256"],
            "artifact_manifest_sha256": payload["artifact_manifest_sha256"],
            "completion_evidence_sha256": payload["readiness_evidence_sha256"],
        }
        if any(
            record.get(field_name) != value
            for field_name, value in expected_record_bindings.items()
        ):
            raise ValueError("lifecycle qualification verification differs from its ledger record")
    return payload


def _validate_external_validation_ready_candidate_payload(
    raw: Any,
    *,
    completion: bool,
) -> dict[str, Any]:
    expected_fields = (
        _EXTERNAL_VALIDATION_READY_COMPLETION_FIELDS
        if completion
        else _EXTERNAL_VALIDATION_READY_GATE_FIELDS
    )
    role = "completion" if completion else "execution gate"
    if not isinstance(raw, Mapping) or set(raw) != expected_fields:
        raise ValueError(f"external-validation {role} has invalid fields")
    payload = dict(raw)
    exact = {
        "schema_version": 1,
        "policy": _EXTERNAL_VALIDATION_READY_CANDIDATE_POLICY,
        "experiment_name": _EXTERNAL_VALIDATION_READY_EXPERIMENT,
        "completion_stage": "EXTERNAL_VALIDATION_READY",
        "study_outcome_eligible": True,
        "post_seal_attestation_required": True,
        "external_validation_complete_claimed": False,
        "item_count": 200,
        "asset_count": 600,
        "expert_response_count": 0,
        "original_audit_experiment_name": "original_label_audit",
    }
    if any(
        type(payload.get(field)) is not type(value) or payload.get(field) != value
        for field, value in exact.items()
    ):
        raise ValueError(f"external-validation {role} has invalid readiness semantics")
    for field_name in ("run_id", "original_audit_run_id", "confirmatory_run_id"):
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"external-validation {role} has invalid {field_name}")
    for field_name in ("original_audit_run_directory", "confirmatory_run_directory"):
        value = payload.get(field_name)
        if (
            not isinstance(value, str)
            or not Path(value).is_absolute()
            or str(Path(value).resolve()) != value
        ):
            raise ValueError(f"external-validation {role} has invalid {field_name}")
    hash_fields = {
        "contract_sha256",
        "cohort_payload_sha256",
        "bundle_root_sha256",
        "public_tree_root_sha256",
        "private_tree_root_sha256",
        "raw_inventory_sha256",
        "canonical_pannuke_manifest_sha256",
        "original_ranking_sha256",
        "original_audit_artifact_root_sha256",
        "original_audit_eligibility_evidence_sha256",
        "confirmatory_artifact_root_sha256",
        "confirmatory_completion_evidence_sha256",
        "confirmatory_stage_attestation_record_sha256",
        "confirmatory_stage_attestation_verification_sha256",
        "technical_inspection_evidence_sha256",
    }
    if completion:
        hash_fields.add("external_validation_execution_gate_sha256")
    for field_name in hash_fields:
        value = payload.get(field_name)
        if not isinstance(value, str) or _LOWER_SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"external-validation {role} has invalid {field_name}")
    return payload


def _validate_external_validation_ready_verification_payload(
    raw: Any,
    *,
    record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _EXTERNAL_VALIDATION_READY_VERIFICATION_FIELDS:
        raise ValueError("external-validation-ready verification payload has invalid fields")
    payload = dict(raw)
    exact = {
        "schema_version": 1,
        "policy": _EXTERNAL_VALIDATION_READY_ATTESTATION_POLICY,
        "experiment_name": _EXTERNAL_VALIDATION_READY_EXPERIMENT,
        "completion_stage": "EXTERNAL_VALIDATION_READY",
        "item_count": 200,
        "asset_count": 600,
        "expert_response_count": 0,
        "original_audit_experiment_name": "original_label_audit",
        "semantic_verification_status": "passed",
    }
    if any(
        type(payload.get(field)) is not type(value) or payload.get(field) != value
        for field, value in exact.items()
    ):
        raise ValueError("external-validation-ready verification payload has invalid semantics")
    for field_name in ("run_id", "original_audit_run_id", "confirmatory_run_id"):
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"external-validation-ready verification has invalid {field_name}")
    for field_name in (
        "run_path",
        "original_audit_run_directory",
        "confirmatory_run_directory",
    ):
        value = payload.get(field_name)
        if (
            not isinstance(value, str)
            or not Path(value).is_absolute()
            or str(Path(value).resolve()) != value
        ):
            raise ValueError(f"external-validation-ready verification has invalid {field_name}")
    for field_name in (
        "first_integrity_root_sha256",
        "final_integrity_root_sha256",
        "artifact_manifest_sha256",
        "completion_evidence_sha256",
        "external_validation_execution_gate_sha256",
        "contract_sha256",
        "cohort_payload_sha256",
        "bundle_root_sha256",
        "public_tree_root_sha256",
        "private_tree_root_sha256",
        "raw_inventory_sha256",
        "canonical_pannuke_manifest_sha256",
        "original_ranking_sha256",
        "original_audit_artifact_root_sha256",
        "original_audit_eligibility_evidence_sha256",
        "confirmatory_artifact_root_sha256",
        "confirmatory_completion_evidence_sha256",
        "confirmatory_stage_attestation_record_sha256",
        "confirmatory_stage_attestation_verification_sha256",
        "technical_inspection_evidence_sha256",
    ):
        value = payload.get(field_name)
        if not isinstance(value, str) or _LOWER_SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"external-validation-ready verification has invalid {field_name}")
    if payload["first_integrity_root_sha256"] != payload["final_integrity_root_sha256"]:
        raise ValueError("external-validation-ready verification integrity roots differ")
    if record is not None:
        expected_record_bindings = {
            "run_id": payload["run_id"],
            "run_path": payload["run_path"],
            "completion_stage": payload["completion_stage"],
            "artifact_root_sha256": payload["final_integrity_root_sha256"],
            "artifact_manifest_sha256": payload["artifact_manifest_sha256"],
            "completion_evidence_sha256": payload["completion_evidence_sha256"],
        }
        if any(record.get(field) != value for field, value in expected_record_bindings.items()):
            raise ValueError(
                "external-validation-ready verification differs from its ledger record"
            )
    return payload


def _validate_primary_stage_verification_payload(
    raw: Any,
    *,
    record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _PRIMARY_STAGE_VERIFICATION_FIELDS:
        raise ValueError("primary post-seal verification payload has invalid fields")
    payload = dict(raw)
    experiment_name = payload.get("experiment_name")
    expected_policy = _PRIMARY_STAGE_EXPERIMENT_POLICIES.get(str(experiment_name))
    if (
        payload.get("schema_version") != 2
        or expected_policy is None
        or payload.get("policy") != expected_policy
        or payload.get("completion_stage") != "PRIMARY_STUDY_COMPLETE"
        or payload.get("semantic_verification_status") != "passed"
    ):
        raise ValueError("primary post-seal verification payload has invalid semantics")
    for field_name in ("run_id", "run_path"):
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"primary post-seal verification has invalid {field_name}")
    if not Path(str(payload["run_path"])).is_absolute():
        raise ValueError("primary post-seal verification run_path is not absolute")
    hash_fields = (
        "first_integrity_root_sha256",
        "final_integrity_root_sha256",
        "artifact_manifest_sha256",
        "completion_evidence_sha256",
        "matrix_plan_sha256",
        "execution_controls_sha256",
        "cell_index_sha256",
        "filesystem_readback_root_sha256",
        "primary_statistics_sha256",
        "primary_bootstrap_evidence_sha256",
        "primary_subgroups_sha256",
        "primary_statistics_manifest_sha256",
        "primary_statistics_source_readback_root_sha256",
        "primary_restoration_index_sha256",
        "primary_restoration_readback_root_sha256",
    )
    for field_name in hash_fields:
        value = payload.get(field_name)
        if not isinstance(value, str) or _LOWER_SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"primary post-seal verification has invalid {field_name}")
    for field_name in (
        "primary_statistics_size_bytes",
        "primary_bootstrap_evidence_size_bytes",
        "primary_subgroups_size_bytes",
        "primary_statistics_manifest_size_bytes",
    ):
        value = payload.get(field_name)
        if type(value) is not int or int(value) < 0:
            raise ValueError(f"primary post-seal verification has invalid {field_name}")
    comparison_count = payload.get("primary_statistics_comparison_count")
    if type(comparison_count) is not int or int(comparison_count) < 0:
        raise ValueError(
            "primary post-seal verification has invalid primary_statistics_comparison_count"
        )
    if (
        payload["primary_statistics_source_readback_root_sha256"]
        != payload["filesystem_readback_root_sha256"]
    ):
        raise ValueError("primary post-seal verification statistics source root differs")
    retry_of_run_id = payload.get("retry_of_run_id")
    if retry_of_run_id is not None and (
        not isinstance(retry_of_run_id, str)
        or not retry_of_run_id.strip()
        or retry_of_run_id != retry_of_run_id.strip()
    ):
        raise ValueError("primary post-seal verification has invalid retry_of_run_id")
    lineage_sha = payload.get("lineage_binding_sha256")
    if lineage_sha is not None and (
        not isinstance(lineage_sha, str) or _LOWER_SHA256_PATTERN.fullmatch(lineage_sha) is None
    ):
        raise ValueError("primary post-seal verification has invalid lineage binding")
    authorization_sha = payload.get("authorization_binding_sha256")
    if authorization_sha is not None and (
        not isinstance(authorization_sha, str)
        or _LOWER_SHA256_PATTERN.fullmatch(authorization_sha) is None
    ):
        raise ValueError("primary post-seal verification has invalid authorization binding")
    if (retry_of_run_id is None) is not (lineage_sha is None):
        raise ValueError("primary post-seal verification has incomplete retry lineage")
    if experiment_name == "pannuke_primary_finalization_successor" and (
        retry_of_run_id is None or lineage_sha is None or authorization_sha is None
    ):
        raise ValueError("primary successor post-seal verification lacks lineage")
    if experiment_name == "pannuke_primary_orphan_recovery" and (
        retry_of_run_id is None or lineage_sha is None or authorization_sha is None
    ):
        raise ValueError("primary orphan recovery post-seal verification lacks lineage")
    if experiment_name == "pannuke_primary_frozen_feature_benchmark" and (
        authorization_sha is not None
    ):
        raise ValueError("ordinary primary cannot carry successor authorization")
    if payload["first_integrity_root_sha256"] != payload["final_integrity_root_sha256"]:
        raise ValueError("primary post-seal verification integrity roots differ")
    if record is not None:
        expected_record_bindings = {
            "run_id": payload["run_id"],
            "run_path": payload["run_path"],
            "completion_stage": payload["completion_stage"],
            "artifact_root_sha256": payload["final_integrity_root_sha256"],
            "artifact_manifest_sha256": payload["artifact_manifest_sha256"],
            "completion_evidence_sha256": payload["completion_evidence_sha256"],
        }
        if any(record.get(field) != value for field, value in expected_record_bindings.items()):
            raise ValueError("primary post-seal verification differs from its ledger record")
    return payload


def _validate_run_stage_attestation_record(
    raw: Any,
    *,
    line_number: int,
    previous_record_sha256: str | None,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"run-stage attestation line {line_number} must be a JSON object")
    record = dict(raw)
    if set(record) != _RUN_STAGE_ATTESTATION_FIELDS:
        raise ValueError(f"run-stage attestation line {line_number} has invalid fields")
    if record.get("schema_version") != 1 or record.get("sequence") != line_number:
        raise ValueError(f"run-stage attestation line {line_number} has invalid schema/sequence")
    scientific_attestation = (
        record.get("event_type") == "postseal_stage_eligibility_attested"
        and record.get("terminal_status") == "completed"
        and record.get("scientific_stage_eligible") is True
        and record.get("completion_stage")
        in {
            "PRIMARY_STUDY_COMPLETE",
            "CONFIRMATORY_COMPLETE",
            "EXTERNAL_VALIDATION_READY",
        }
    )
    lifecycle_attestation = (
        record.get("event_type") == _LIFECYCLE_QUALIFICATION_EVENT
        and record.get("terminal_status") == "completed"
        and record.get("scientific_stage_eligible") is False
        and record.get("completion_stage") is None
    )
    if not scientific_attestation and not lifecycle_attestation:
        raise ValueError(f"run-stage attestation line {line_number} has invalid semantics")
    recorded_at = record.get("recorded_at_utc")
    if not isinstance(recorded_at, str) or not recorded_at.endswith("Z"):
        raise ValueError(f"run-stage attestation line {line_number} has invalid timestamp")
    for field_name in ("run_id", "run_path"):
        value = record.get(field_name)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"run-stage attestation line {line_number} has invalid {field_name}")
    if not Path(str(record["run_path"])).is_absolute():
        raise ValueError(f"run-stage attestation line {line_number} path is not absolute")
    for field_name in (
        "artifact_root_sha256",
        "artifact_manifest_sha256",
        "completion_evidence_sha256",
        "verification_sha256",
        "record_sha256",
    ):
        value = record.get(field_name)
        if not isinstance(value, str) or _LOWER_SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"run-stage attestation line {line_number} has invalid {field_name}")
    if record.get("previous_record_sha256") != previous_record_sha256:
        raise ValueError(f"run-stage attestation line {line_number} breaks the hash chain")
    if record.get("record_sha256") != _run_disposition_record_sha256(record):
        raise ValueError(f"run-stage attestation line {line_number} has invalid record hash")
    verification = record.get("verification")
    if not isinstance(verification, Mapping):
        raise ValueError(f"run-stage attestation line {line_number} checklist is invalid")
    if lifecycle_attestation:
        try:
            _validate_lifecycle_qualification_verification_payload(verification, record=record)
        except ValueError as exc:
            raise ValueError(
                f"run-stage attestation line {line_number} lifecycle verification is invalid: {exc}"
            ) from exc
    elif record["completion_stage"] == "PRIMARY_STUDY_COMPLETE":
        try:
            _validate_primary_stage_verification_payload(verification, record=record)
        except ValueError as exc:
            raise ValueError(
                f"run-stage attestation line {line_number} primary verification is invalid: {exc}"
            ) from exc
    elif record["completion_stage"] == "CONFIRMATORY_COMPLETE":
        expected_verification_fields = {
            "schema_version",
            "policy",
            "run_id",
            "completion_stage",
            "first_integrity_root_sha256",
            "final_integrity_root_sha256",
            "matrix_plan_sha256",
            "cell_index_sha256",
            "scientific_artifact_manifest_sha256",
            "reconciliation_sha256",
            "confirmatory_storage_policy_sha256",
            "semantic_readback_status",
            "semantic_checked_artifact_count",
        }
        if (
            set(verification) != expected_verification_fields
            or verification.get("schema_version") != 1
            or verification.get("policy") != "confirmatory_postseal_attestation_v1"
            or verification.get("run_id") != record["run_id"]
            or verification.get("completion_stage") != record["completion_stage"]
            or verification.get("semantic_readback_status") != "passed"
            or type(verification.get("semantic_checked_artifact_count")) is not int
            or int(verification["semantic_checked_artifact_count"]) <= 0
        ):
            raise ValueError(f"run-stage attestation line {line_number} checklist is invalid")
        for field_name in (
            "first_integrity_root_sha256",
            "final_integrity_root_sha256",
            "matrix_plan_sha256",
            "cell_index_sha256",
            "scientific_artifact_manifest_sha256",
            "reconciliation_sha256",
            "confirmatory_storage_policy_sha256",
        ):
            value = verification.get(field_name)
            if not isinstance(value, str) or _LOWER_SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(
                    f"run-stage attestation line {line_number} checklist {field_name} is invalid"
                )
        if (
            verification["first_integrity_root_sha256"] != record["artifact_root_sha256"]
            or verification["final_integrity_root_sha256"] != record["artifact_root_sha256"]
        ):
            raise ValueError(
                f"run-stage attestation line {line_number} checklist root differs from run"
            )
    else:
        try:
            _validate_external_validation_ready_verification_payload(
                verification,
                record=record,
            )
        except ValueError as exc:
            raise ValueError(
                f"run-stage attestation line {line_number} external-validation-ready "
                f"verification is invalid: {exc}"
            ) from exc
    encoded_verification = json.dumps(
        dict(verification),
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if record["verification_sha256"] != hashlib.sha256(encoded_verification).hexdigest():
        raise ValueError(f"run-stage attestation line {line_number} checklist hash is invalid")
    return record


def _read_run_stage_attestations_unlocked(
    registry_path: Path,
) -> tuple[dict[str, Any], ...]:
    if not os.path.lexists(registry_path):
        return ()
    if not registry_path.is_file():
        raise ValueError(f"run-stage attestation registry is not a file: {registry_path}")
    try:
        content = registry_path.read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"run-stage attestation registry is unreadable: {exc}") from exc
    if content and not content.endswith("\n"):
        raise ValueError("run-stage attestation registry has a truncated final record")
    lines = content[:-1].split("\n") if content else []
    records: list[dict[str, Any]] = []
    previous: str | None = None
    for line_number, line in enumerate(lines, start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"run-stage attestation line {line_number} is invalid JSON") from exc
        record = _validate_run_stage_attestation_record(
            raw,
            line_number=line_number,
            previous_record_sha256=previous,
        )
        records.append(record)
        previous = str(record["record_sha256"])
    if len({str(record["run_id"]) for record in records}) != len(records):
        raise ValueError("run-stage attestation registry repeats a run")
    return tuple(records)


def _run_stage_attestation_anchor_path(registry_path: Path) -> Path:
    return registry_path.with_name(RUN_STAGE_ATTESTATION_ANCHOR_FILENAME)


def _run_stage_attestation_ledger_bytes(registry_path: Path) -> bytes:
    if not os.path.lexists(registry_path):
        return b""
    if not registry_path.is_file():
        raise ValueError("run-stage attestation registry is not a regular file")
    return registry_path.read_bytes()


def _run_stage_attestation_anchor_payload(
    records: Sequence[Mapping[str, Any]], ledger_bytes: bytes
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ledger_filename": RUN_STAGE_ATTESTATION_REGISTRY_FILENAME,
        "chain_algorithm": "sha256(canonical-json-record-with-previous-head)",
        "record_count": len(records),
        "head_record_sha256": records[-1]["record_sha256"] if records else None,
        "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
    }


def _require_run_stage_attestation_anchor_unlocked(
    registry_path: Path,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    anchor_path = _run_stage_attestation_anchor_path(registry_path)
    if not anchor_path.is_file():
        raise ValueError("run-stage attestation anchor is missing; eligibility fails closed")
    try:
        raw = json.loads(anchor_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("run-stage attestation anchor is invalid") from exc
    expected = _run_stage_attestation_anchor_payload(
        records, _run_stage_attestation_ledger_bytes(registry_path)
    )
    if not isinstance(raw, Mapping) or dict(raw) != expected:
        raise ValueError("run-stage attestation anchor does not match its append-only ledger")
    return dict(raw)


def _ensure_run_stage_attestation_anchor(runs_root: Path) -> Path:
    root = runs_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    registry_path = root / RUN_STAGE_ATTESTATION_REGISTRY_FILENAME
    anchor_path = root / RUN_STAGE_ATTESTATION_ANCHOR_FILENAME
    with _registry_lock(registry_path):
        records = _read_run_stage_attestations_unlocked(registry_path)
        if anchor_path.exists():
            _require_run_stage_attestation_anchor_unlocked(registry_path, records)
        else:
            if records or _run_stage_attestation_ledger_bytes(registry_path):
                raise ValueError(
                    "cannot initialise a missing run-stage attestation anchor for a "
                    "non-empty ledger"
                )
            atomic_write_json(
                anchor_path,
                _run_stage_attestation_anchor_payload((), b""),
            )
    return anchor_path


def read_run_stage_attestations(
    registry_path: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Read and verify the positive post-seal stage-attestation hash chain."""

    source = Path(registry_path)
    with _registry_lock(source):
        records = _read_run_stage_attestations_unlocked(source)
        _require_run_stage_attestation_anchor_unlocked(source, records)
        return records


def _read_sealed_json_object(path: Path, role: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"sealed {role} is unavailable or invalid") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"sealed {role} must be a JSON object")
    return dict(raw)


def _external_validation_ready_inspection_sha256(
    run_path: Path,
    *,
    expected_bundle_root_sha256: str,
) -> str:
    """Strictly verify judgement-free technical inspection bytes from the sealed run."""

    path = run_path / _EXTERNAL_VALIDATION_READY_TECHNICAL_INSPECTION_FILENAME
    try:
        path_stat = path.lstat()
        attributes = int(getattr(path_stat, "st_file_attributes", 0))
        reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or path.is_symlink()
            or bool(attributes & reparse_flag)
        ):
            raise ValueError("technical inspection evidence is not a regular physical file")
        payload = path.read_bytes()

        def reject_constant(value: str) -> Any:
            raise ValueError(f"technical inspection contains non-finite constant {value}")

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"technical inspection repeats field {key!r}")
                result[key] = value
            return result

        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("sealed technical inspection evidence is invalid") from exc
    if not isinstance(value, Mapping) or set(value) != _EXTERNAL_VALIDATION_READY_INSPECTION_FIELDS:
        raise ValueError("technical inspection evidence has an invalid exact field set")
    exact: dict[str, Any] = {
        "schema_version": 1,
        "evidence_kind": "m9_blinded_package_technical_inspection",
        "inspection_protocol": "technical_blinded_package_inspection_v1",
        "decision": "passed",
        "bundle_root_sha256": expected_bundle_root_sha256,
        "item_count": 200,
        "asset_count": 600,
        "expert_response_count": 0,
        "annotation_judgment_data_present": False,
        "annotation_evaluations_performed": False,
        "outcomes_used_for_tuning": False,
    }
    if any(
        type(value.get(field)) is not type(expected) or value.get(field) != expected
        for field, expected in exact.items()
    ):
        raise ValueError("technical inspection evidence has invalid decision or semantics")
    for field_name in ("inspection_id", "inspector_id"):
        token = value.get(field_name)
        if (
            not isinstance(token, str)
            or _EXTERNAL_VALIDATION_READY_INSPECTION_ID_PATTERN.fullmatch(token) is None
        ):
            raise ValueError(f"technical inspection has invalid {field_name}")
    inspected_at = value.get("inspected_at_utc")
    if (
        not isinstance(inspected_at, str)
        or _EXTERNAL_VALIDATION_READY_INSPECTION_UTC_PATTERN.fullmatch(inspected_at) is None
    ):
        raise ValueError("technical inspection has an invalid UTC timestamp")
    checks = value.get("checks")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != _EXTERNAL_VALIDATION_READY_INSPECTION_CHECKS
        or any(
            checks.get(field) is not True for field in _EXTERNAL_VALIDATION_READY_INSPECTION_CHECKS
        )
    ):
        raise ValueError("technical inspection checklist is incomplete or non-positive")
    return hashlib.sha256(payload).hexdigest()


def _external_validation_ready_candidate_bindings(
    run_path: Path,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    """Re-read every sealed M9 candidate binding needed by the stage ledger."""

    status = _read_sealed_json_object(run_path / STATUS_FILENAME, "external-validation status")
    if (
        status.get("status") != "completed"
        or status.get("run_id") != run_path.name
        or status.get("experiment_name") != _EXTERNAL_VALIDATION_READY_EXPERIMENT
    ):
        raise ValueError("sealed run is not the exact completed external-validation package")

    gate_path = run_path / _EXTERNAL_VALIDATION_READY_GATE_FILENAME
    completion_path = run_path / _EXTERNAL_VALIDATION_READY_COMPLETION_FILENAME
    gate = _validate_external_validation_ready_candidate_payload(
        _read_sealed_json_object(gate_path, "external-validation execution gate"),
        completion=False,
    )
    completion = _validate_external_validation_ready_candidate_payload(
        _read_sealed_json_object(completion_path, "external-validation completion"),
        completion=True,
    )
    if gate["run_id"] != run_path.name or completion["run_id"] != run_path.name:
        raise ValueError("external-validation candidate run identity differs from its directory")
    shared_completion = {
        field: completion[field] for field in _EXTERNAL_VALIDATION_READY_SHARED_FIELDS
    }
    if shared_completion != gate:
        raise ValueError("external-validation completion differs from its sealed execution gate")
    gate_sha256 = sha256_file(gate_path)
    if completion["external_validation_execution_gate_sha256"] != gate_sha256:
        raise ValueError("external-validation completion does not bind its exact execution gate")

    bundle = run_path / _EXTERNAL_VALIDATION_READY_BUNDLE_RELATIVE_PATH
    package_evidence_path = bundle / "private" / "package_evidence.json"
    package_evidence = _read_sealed_json_object(
        package_evidence_path,
        "M9 review-bundle package evidence",
    )
    package_bindings = {
        "item_count": package_evidence.get("item_count"),
        "asset_count": package_evidence.get("asset_count"),
        "contract_sha256": package_evidence.get("contract_sha256"),
        "cohort_payload_sha256": package_evidence.get("cohort_payload_sha256"),
        "public_tree_root_sha256": package_evidence.get("public_tree_root_sha256"),
        "raw_inventory_sha256": package_evidence.get("raw_inventory_sha256"),
        "canonical_pannuke_manifest_sha256": package_evidence.get(
            "canonical_pannuke_manifest_sha256"
        ),
    }
    if any(completion.get(field) != value for field, value in package_bindings.items()):
        raise ValueError("external-validation completion differs from sealed bundle evidence")

    inspection_sha256 = _external_validation_ready_inspection_sha256(
        run_path,
        expected_bundle_root_sha256=str(completion["bundle_root_sha256"]),
    )
    if inspection_sha256 != completion["technical_inspection_evidence_sha256"]:
        raise ValueError("external-validation technical inspection evidence differs")

    original_path = Path(str(completion["original_audit_run_directory"])).resolve()
    if original_path == run_path or original_path.name != completion["original_audit_run_id"]:
        raise ValueError("external-validation original-audit identity is invalid")
    original_integrity = verify_run_integrity(original_path)
    if (
        not original_integrity.valid
        or not original_integrity.registry_record_present
        or original_integrity.run_id != completion["original_audit_run_id"]
        or original_integrity.expected_root_sha256
        != completion["original_audit_artifact_root_sha256"]
        or original_integrity.actual_root_sha256
        != completion["original_audit_artifact_root_sha256"]
    ):
        raise ValueError("external-validation original-audit seal differs")
    original_status = _read_sealed_json_object(
        original_path / STATUS_FILENAME, "original-audit status"
    )
    ranking_path = original_path / "ranking_all.csv"
    if (
        original_status.get("status") != "completed"
        or original_status.get("experiment_name") != completion["original_audit_experiment_name"]
        or ranking_path.is_symlink()
        or not ranking_path.is_file()
        or sha256_file(ranking_path) != completion["original_ranking_sha256"]
    ):
        raise ValueError("external-validation original-audit ranking differs")

    confirmatory_path = Path(str(completion["confirmatory_run_directory"])).resolve()
    if (
        confirmatory_path in {run_path, original_path}
        or confirmatory_path.name != completion["confirmatory_run_id"]
    ):
        raise ValueError("external-validation confirmatory identity is invalid")
    original_eligibility_path = original_path / "external_validation_eligibility.json"
    original_eligibility = _read_sealed_json_object(
        original_eligibility_path,
        "original-audit eligibility evidence",
    )
    ranking_record = original_eligibility.get("ranking")
    recorded_confirmatory = original_eligibility.get("confirmatory_run")
    if (
        original_eligibility.get("schema_version") != 2
        or original_eligibility.get("workflow") != "exploratory_original_label_audit"
        or original_eligibility.get("study_outcome_eligible") is not True
        or sha256_file(original_eligibility_path)
        != completion["original_audit_eligibility_evidence_sha256"]
        or not isinstance(ranking_record, Mapping)
        or Path(str(ranking_record.get("path", ""))).resolve() != ranking_path
        or ranking_record.get("sha256") != completion["original_ranking_sha256"]
        or not isinstance(recorded_confirmatory, Mapping)
    ):
        raise ValueError("external-validation original-audit eligibility evidence differs")
    expected_confirmatory = {
        "path": str(confirmatory_path),
        "run_id": completion["confirmatory_run_id"],
        "artifact_root_sha256": completion["confirmatory_artifact_root_sha256"],
        "completion_evidence_sha256": completion["confirmatory_completion_evidence_sha256"],
        "stage_attestation_record_sha256": completion[
            "confirmatory_stage_attestation_record_sha256"
        ],
        "stage_attestation_verification_sha256": completion[
            "confirmatory_stage_attestation_verification_sha256"
        ],
    }
    if any(
        recorded_confirmatory.get(field) != value for field, value in expected_confirmatory.items()
    ):
        raise ValueError("external-validation original-audit confirmatory lineage differs")
    return gate_path, gate, completion_path, completion


def _build_primary_stage_attestation_verification(
    run_directory: str | Path,
    *,
    integrity: IntegrityVerification,
    completion: Mapping[str, Any],
    filesystem_readback: Any,
    statistics_verification: Any,
    restoration_readback: Any,
    lineage_verification: Mapping[str, Any] | None = None,
) -> PrimaryStageAttestationVerification:
    """Build a typed proof without recomputing primary statistics after the seal."""

    from histo_audit.experiment.primary_completion import (
        PrimaryFilesystemReadbackEvidence,
        PrimaryRestorationReadbackEvidence,
    )
    from histo_audit.experiment.primary_statistics import (
        InheritedPrimaryStatisticsVerification,
        PrimaryStatisticsVerification,
    )

    run_path = Path(run_directory).resolve()
    inherited_statistics_verification = (
        statistics_verification
        if isinstance(
            statistics_verification,
            InheritedPrimaryStatisticsVerification,
        )
        else None
    )
    if (
        not integrity.valid
        or not integrity.registry_record_present
        or integrity.run_id != run_path.name
        or integrity.expected_root_sha256 != integrity.actual_root_sha256
        or not isinstance(integrity.expected_root_sha256, str)
        or _LOWER_SHA256_PATTERN.fullmatch(integrity.expected_root_sha256) is None
    ):
        raise ValueError("primary stage verification requires a fresh valid completed seal")
    if (
        not isinstance(filesystem_readback, PrimaryFilesystemReadbackEvidence)
        or not filesystem_readback.passed
        or filesystem_readback.run_directory.resolve() != run_path
        or not isinstance(
            statistics_verification,
            (
                PrimaryStatisticsVerification,
                InheritedPrimaryStatisticsVerification,
            ),
        )
        or not statistics_verification.valid
        or statistics_verification.output_directory.resolve() != run_path
        or not isinstance(restoration_readback, PrimaryRestorationReadbackEvidence)
        or not restoration_readback.passed
        or restoration_readback.run_directory.resolve() != run_path
    ):
        raise ValueError("primary stage verification requires genuine typed readback objects")
    if (
        statistics_verification.source_readback_root_sha256
        != filesystem_readback.readback_root_sha256
        or restoration_readback.source_readback_root_sha256
        != filesystem_readback.readback_root_sha256
    ):
        raise ValueError("primary typed verifications bind different filesystem readbacks")

    status = _read_sealed_json_object(run_path / STATUS_FILENAME, "primary status")
    experiment_name = status.get("experiment_name")
    policy = _PRIMARY_STAGE_EXPERIMENT_POLICIES.get(str(experiment_name))
    if status.get("status") != "completed" or policy is None:
        raise ValueError("sealed run is not an exact completed primary experiment")
    if (
        experiment_name == "pannuke_primary_orphan_recovery"
        and inherited_statistics_verification is None
    ):
        raise ValueError("primary orphan recovery requires inherited statistics verification only")
    if (
        experiment_name == "pannuke_primary_orphan_recovery"
        and inherited_statistics_verification is not None
        and inherited_statistics_verification.authorization_kind != "orphan_recovery"
    ):
        raise ValueError("primary orphan recovery requires an orphan-recovery numeric proof")
    sealed_completion = _read_sealed_json_object(
        run_path / "completion_evidence.json", "primary completion evidence"
    )
    if sealed_completion != dict(completion):
        raise ValueError("sealed primary completion differs from the verified candidate")

    expected_completion = {
        "run_id": run_path.name,
        "completion_stage": "PRIMARY_STUDY_COMPLETE",
        "study_outcome_eligible": True,
        "post_seal_attestation_required": True,
        "filesystem_matrix_plan_sha256": filesystem_readback.matrix_plan_sha256,
        "filesystem_execution_controls_sha256": filesystem_readback.execution_controls_sha256,
        "filesystem_execution_controls_binding_sha256": (
            filesystem_readback.execution_controls_binding_sha256
        ),
        "filesystem_cell_index_sha256": filesystem_readback.cell_index_sha256,
        "filesystem_readback_root_sha256": filesystem_readback.readback_root_sha256,
        "primary_statistics_verification_status": "passed",
        "primary_statistics_sha256": statistics_verification.statistics_sha256,
        "primary_bootstrap_evidence_sha256": (statistics_verification.bootstrap_evidence_sha256),
        "primary_subgroups_sha256": statistics_verification.subgroups_sha256,
        "primary_statistics_manifest_sha256": statistics_verification.manifest_sha256,
        "primary_statistics_source_readback_root_sha256": (
            statistics_verification.source_readback_root_sha256
        ),
        "primary_statistics_comparison_count": statistics_verification.comparison_count,
        "primary_restoration_verification_status": "passed",
        "primary_restoration_index_sha256": restoration_readback.restoration_index_sha256,
        "primary_restoration_readback_root_sha256": restoration_readback.readback_root_sha256,
        "primary_restoration_source_readback_root_sha256": (
            restoration_readback.source_readback_root_sha256
        ),
    }
    if any(sealed_completion.get(field) != value for field, value in expected_completion.items()):
        raise ValueError("sealed primary completion differs from typed verification evidence")
    expected_restoration_hashes: dict[str, str] = {}
    for role, hashes in (
        ("json", restoration_readback.cell_json_sha256),
        ("evidence", restoration_readback.cell_evidence_sha256),
        ("manifest", restoration_readback.cell_manifest_sha256),
    ):
        for cell_id, digest in hashes:
            expected_restoration_hashes[f"primary_restoration_{role}_sha256::{cell_id}"] = digest
    actual_restoration_keys = {
        key
        for key in sealed_completion
        if isinstance(key, str) and key.startswith("primary_restoration_") and "_sha256::" in key
    }
    if actual_restoration_keys != set(expected_restoration_hashes) or any(
        sealed_completion.get(field) != digest
        for field, digest in expected_restoration_hashes.items()
    ):
        raise ValueError("sealed primary completion lacks exact typed restoration hashes")

    retry_of_run_id = sealed_completion.get("retry_of_run_id")
    lineage_sha = sealed_completion.get("retry_predecessor_binding_sha256")
    if retry_of_run_id is not None and not isinstance(retry_of_run_id, str):
        raise ValueError("sealed primary retry identity is invalid")
    if lineage_sha is not None and (
        not isinstance(lineage_sha, str) or _LOWER_SHA256_PATTERN.fullmatch(lineage_sha) is None
    ):
        raise ValueError("sealed primary retry lineage hash is invalid")
    if (retry_of_run_id is None) is not (lineage_sha is None):
        raise ValueError("sealed primary retry lineage is incomplete")
    if experiment_name == "pannuke_primary_finalization_successor":
        evidence_path = run_path / "primary_finalization_successor_evidence.json"
        evidence = _read_sealed_json_object(evidence_path, "primary successor lineage evidence")
        authorization_sha = sealed_completion.get("finalization_successor_authorization_sha256")
        provenance = _read_sealed_json_object(
            run_path / "run_provenance.json", "primary successor provenance"
        )
        if (
            lineage_verification is None
            or dict(lineage_verification) != evidence
            or sealed_completion.get("finalization_only_successor") is not True
            or sealed_completion.get("finalization_successor_evidence_sha256")
            != sha256_file(evidence_path)
            or lineage_sha != sha256_file(evidence_path)
            or not isinstance(authorization_sha, str)
            or _LOWER_SHA256_PATTERN.fullmatch(authorization_sha) is None
            or provenance.get("finalization_successor_authorization_sha256") != authorization_sha
            or (
                inherited_statistics_verification is not None
                and (
                    inherited_statistics_verification.authorization_kind != "finalization_successor"
                    or evidence.get("schema_version") != 2
                    or sealed_completion.get("verification_mode")
                    != inherited_statistics_verification.verification_mode
                    or sealed_completion.get("prior_numeric_verification_proof_sha256")
                    != inherited_statistics_verification.prior_numeric_verification_proof_sha256
                    or evidence.get("verification_mode")
                    != inherited_statistics_verification.verification_mode
                    or evidence.get("prior_numeric_verification_proof_sha256")
                    != inherited_statistics_verification.prior_numeric_verification_proof_sha256
                    or provenance.get("verification_mode")
                    != inherited_statistics_verification.verification_mode
                    or provenance.get("prior_numeric_verification_proof_sha256")
                    != inherited_statistics_verification.prior_numeric_verification_proof_sha256
                )
            )
            or (
                inherited_statistics_verification is None
                and (
                    evidence.get("verification_mode") is not None
                    or sealed_completion.get("verification_mode") is not None
                )
            )
        ):
            raise ValueError("primary successor lacks its exact verified sealed lineage")
    elif experiment_name == "pannuke_primary_orphan_recovery":
        from histo_audit.experiment.primary_recovery import (
            RECOVERY_EVIDENCE_FILENAME,
            RECOVERY_POLICY,
            RECOVERY_REGISTRATION_STATUS,
        )

        assert inherited_statistics_verification is not None
        evidence_path = run_path / RECOVERY_EVIDENCE_FILENAME
        evidence = _read_sealed_json_object(
            evidence_path,
            "primary orphan-recovery lineage evidence",
        )
        authorization_sha = sealed_completion.get("recovery_authorization_sha256")
        source_snapshot_root = sealed_completion.get("recovery_source_snapshot_root_sha256")
        provenance = _read_sealed_json_object(
            run_path / "run_provenance.json",
            "primary orphan-recovery provenance",
        )
        if (
            lineage_verification is None
            or dict(lineage_verification) != evidence
            or retry_of_run_id is None
            or sealed_completion.get("recovery_only") is not True
            or sealed_completion.get("recovery_policy") != RECOVERY_POLICY
            or sealed_completion.get("analysis_disposition") != RECOVERY_REGISTRATION_STATUS
            or sealed_completion.get("outcomes_inspected") is not True
            or sealed_completion.get("reused_required_cell_count")
            != filesystem_readback.completed_required_cell_count
            or sealed_completion.get("skipped_optional_cell_count")
            != filesystem_readback.skipped_optional_cell_count
            or filesystem_readback.completed_required_cell_count != 185
            or filesystem_readback.skipped_optional_cell_count != 37
            or sealed_completion.get("retrained_cell_count") != 0
            or evidence.get("reused_required_cell_count")
            != filesystem_readback.completed_required_cell_count
            or evidence.get("skipped_optional_cell_count")
            != filesystem_readback.skipped_optional_cell_count
            or evidence.get("retrained_cell_count") != 0
            or sealed_completion.get("recovery_evidence_sha256") != sha256_file(evidence_path)
            or lineage_sha != sha256_file(evidence_path)
            or not isinstance(source_snapshot_root, str)
            or _LOWER_SHA256_PATTERN.fullmatch(source_snapshot_root) is None
            or evidence.get("source_snapshot_root_sha256") != source_snapshot_root
            or evidence.get("destination_snapshot_root_sha256") != source_snapshot_root
            or evidence.get("source_run_id") != retry_of_run_id
            or evidence.get("destination_run_id") != run_path.name
            or not isinstance(authorization_sha, str)
            or _LOWER_SHA256_PATTERN.fullmatch(authorization_sha) is None
            or evidence.get("recovery_authorization_sha256") != authorization_sha
            or provenance.get("recovery_authorization_sha256") != authorization_sha
            or provenance.get("recovery_source_snapshot_root_sha256") != source_snapshot_root
            or evidence.get("verification_mode")
            != inherited_statistics_verification.verification_mode
            or sealed_completion.get("verification_mode")
            != inherited_statistics_verification.verification_mode
            or provenance.get("verification_mode")
            != inherited_statistics_verification.verification_mode
            or evidence.get("prior_numeric_verification_proof_sha256")
            != inherited_statistics_verification.prior_numeric_verification_proof_sha256
            or sealed_completion.get("prior_numeric_verification_proof_sha256")
            != inherited_statistics_verification.prior_numeric_verification_proof_sha256
            or provenance.get("prior_numeric_verification_proof_sha256")
            != inherited_statistics_verification.prior_numeric_verification_proof_sha256
        ):
            raise ValueError("primary orphan recovery lacks its exact verified sealed lineage")
    elif (
        inherited_statistics_verification is not None
        or lineage_verification is not None
        or sealed_completion.get("finalization_only_successor")
    ):
        raise ValueError("ordinary primary cannot carry finalization-successor lineage")
    else:
        authorization_sha = None

    explicit_hashes = {
        "matrix_plan.json": filesystem_readback.matrix_plan_sha256,
        "execution_controls.json": filesystem_readback.execution_controls_sha256,
        "cell_index.csv": filesystem_readback.cell_index_sha256,
        "primary_statistics.json": statistics_verification.statistics_sha256,
        "primary_bootstrap_evidence.npz": statistics_verification.bootstrap_evidence_sha256,
        "primary_subgroups.csv": statistics_verification.subgroups_sha256,
        "primary_statistics_manifest.json": statistics_verification.manifest_sha256,
        "restoration_index.json": restoration_readback.restoration_index_sha256,
    }
    for relative_path, expected_sha in explicit_hashes.items():
        path = run_path / relative_path
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"sealed primary file differs from typed evidence: {relative_path}")

    statistics_paths = {
        "primary_statistics_size_bytes": run_path / "primary_statistics.json",
        "primary_bootstrap_evidence_size_bytes": run_path / "primary_bootstrap_evidence.npz",
        "primary_subgroups_size_bytes": run_path / "primary_subgroups.csv",
        "primary_statistics_manifest_size_bytes": (run_path / "primary_statistics_manifest.json"),
    }
    verification = PrimaryStageAttestationVerification(
        policy=policy,
        experiment_name=str(experiment_name),
        run_id=run_path.name,
        run_path=str(run_path),
        completion_stage="PRIMARY_STUDY_COMPLETE",
        first_integrity_root_sha256=str(integrity.expected_root_sha256),
        final_integrity_root_sha256=str(integrity.expected_root_sha256),
        artifact_manifest_sha256=sha256_file(run_path / ARTIFACT_MANIFEST_FILENAME),
        completion_evidence_sha256=sha256_file(run_path / "completion_evidence.json"),
        matrix_plan_sha256=filesystem_readback.matrix_plan_sha256,
        execution_controls_sha256=filesystem_readback.execution_controls_sha256,
        cell_index_sha256=filesystem_readback.cell_index_sha256,
        filesystem_readback_root_sha256=filesystem_readback.readback_root_sha256,
        primary_statistics_sha256=statistics_verification.statistics_sha256,
        primary_statistics_size_bytes=statistics_paths["primary_statistics_size_bytes"]
        .stat()
        .st_size,
        primary_bootstrap_evidence_sha256=(statistics_verification.bootstrap_evidence_sha256),
        primary_bootstrap_evidence_size_bytes=statistics_paths[
            "primary_bootstrap_evidence_size_bytes"
        ]
        .stat()
        .st_size,
        primary_subgroups_sha256=statistics_verification.subgroups_sha256,
        primary_subgroups_size_bytes=statistics_paths["primary_subgroups_size_bytes"]
        .stat()
        .st_size,
        primary_statistics_manifest_sha256=statistics_verification.manifest_sha256,
        primary_statistics_manifest_size_bytes=statistics_paths[
            "primary_statistics_manifest_size_bytes"
        ]
        .stat()
        .st_size,
        primary_statistics_source_readback_root_sha256=(
            statistics_verification.source_readback_root_sha256
        ),
        primary_statistics_comparison_count=statistics_verification.comparison_count,
        primary_restoration_index_sha256=restoration_readback.restoration_index_sha256,
        primary_restoration_readback_root_sha256=restoration_readback.readback_root_sha256,
        retry_of_run_id=retry_of_run_id,
        lineage_binding_sha256=lineage_sha,
        authorization_binding_sha256=authorization_sha,
    )
    object.__setattr__(verification, "_attestation", _PRIMARY_STAGE_ATTESTATION_TOKEN)
    payload = _validate_primary_stage_verification_payload(verification.as_dict())
    _require_primary_recovery_authorization_binding(
        run_path,
        completion=sealed_completion,
        payload=payload,
    )
    return verification


def _require_primary_verification_files_match(
    run_path: Path,
    payload: Mapping[str, Any],
) -> None:
    expected_files = {
        "matrix_plan.json": ("matrix_plan_sha256", None),
        "execution_controls.json": ("execution_controls_sha256", None),
        "cell_index.csv": ("cell_index_sha256", None),
        "primary_statistics.json": (
            "primary_statistics_sha256",
            "primary_statistics_size_bytes",
        ),
        "primary_bootstrap_evidence.npz": (
            "primary_bootstrap_evidence_sha256",
            "primary_bootstrap_evidence_size_bytes",
        ),
        "primary_subgroups.csv": (
            "primary_subgroups_sha256",
            "primary_subgroups_size_bytes",
        ),
        "primary_statistics_manifest.json": (
            "primary_statistics_manifest_sha256",
            "primary_statistics_manifest_size_bytes",
        ),
        "restoration_index.json": ("primary_restoration_index_sha256", None),
    }
    for relative_path, (sha_field, size_field) in expected_files.items():
        path = run_path / relative_path
        if path.is_symlink() or not path.is_file() or sha256_file(path) != payload[sha_field]:
            raise ValueError(f"primary attestation file binding changed: {relative_path}")
        if size_field is not None and path.stat().st_size != payload[size_field]:
            raise ValueError(f"primary attestation file size changed: {relative_path}")


def _independent_confirmatory_attestation_verification(
    run_path: Path,
    *,
    completion: Mapping[str, Any],
    artifact_root_sha256: str,
) -> dict[str, Any]:
    """Re-run the typed scientific reader instead of trusting caller assertions."""

    # These imports are intentionally local: completion validation uses run-tracking
    # primitives, while this final positive-commit path must independently invoke the
    # completed validator without creating an import cycle during module initialisation.
    from histo_audit.config import load_config
    from histo_audit.corruption.controlled import canonical_sha256
    from histo_audit.experiment.confirmatory_completion import (
        read_confirmatory_run_directory,
    )
    from histo_audit.experiment.study_contracts import build_confirmatory_matrix_plan
    from histo_audit.workflows.preregistration_amendment import (
        require_confirmatory_storage_policy,
    )

    gate_path = run_path / "confirmatory_execution_gate.json"
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("sealed confirmatory gate evidence is unavailable") from exc
    if not isinstance(gate, Mapping) or not isinstance(gate.get("primary_gate"), Mapping):
        raise ValueError("sealed confirmatory gate evidence is malformed")
    primary_gate = gate["primary_gate"]
    freeze_directory = primary_gate.get("freeze_directory")
    frozen_sha256 = primary_gate.get("frozen_confirmatory_config_sha256")
    if (
        not isinstance(freeze_directory, str)
        or not Path(freeze_directory).is_absolute()
        or not isinstance(frozen_sha256, str)
        or _LOWER_SHA256_PATTERN.fullmatch(frozen_sha256) is None
    ):
        raise ValueError("sealed confirmatory gate lacks an exact frozen config binding")
    frozen_path = Path(freeze_directory).resolve() / "confirmatory_frozen.yaml"
    if sha256_file(frozen_path) != frozen_sha256:
        raise ValueError("frozen confirmatory config differs before positive attestation")
    storage_policy_sha256 = canonical_sha256(
        require_confirmatory_storage_policy(Path(freeze_directory).resolve())
    )
    if (
        _LOWER_SHA256_PATTERN.fullmatch(storage_policy_sha256) is None
        or gate.get("confirmatory_storage_policy_sha256") != storage_policy_sha256
    ):
        raise ValueError("sealed confirmatory gate differs from the live storage-policy authority")
    plan = build_confirmatory_matrix_plan(load_config(frozen_path))
    readback = read_confirmatory_run_directory(
        plan,
        run_path,
        frozen_confirmatory_config_path=frozen_path,
        expected_frozen_config_sha256=frozen_sha256,
        expected_confirmatory_storage_policy_sha256=storage_policy_sha256,
        require_final_policy_bindings=True,
    )
    if not readback.passed or readback.reconciliation is None:
        raise ValueError(
            f"independent scientific readback failed before positive attestation: {readback.errors}"
        )
    reconciliation_sha256 = hashlib.sha256(
        json.dumps(
            readback.reconciliation.as_dict(),
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    readback_sha256 = hashlib.sha256(
        json.dumps(
            readback.as_dict(),
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    expected_completion = {
        "filesystem_readback_status": readback.status,
        "filesystem_checked_artifact_count": readback.checked_artifact_count,
        "filesystem_matrix_plan_sha256": readback.matrix_plan_sha256,
        "filesystem_cell_index_sha256": readback.cell_index_sha256,
        "filesystem_root_artifact_manifest_sha256": (readback.root_artifact_manifest_sha256),
        "filesystem_confirmatory_storage_policy_sha256": (
            readback.confirmatory_storage_policy_sha256
        ),
        "filesystem_readback_sha256": readback_sha256,
        "confirmatory_storage_policy_sha256": storage_policy_sha256,
    }
    for field_name, expected in expected_completion.items():
        if completion.get(field_name) != expected:
            raise ValueError(f"sealed completion {field_name} differs from independent readback")
    return {
        "schema_version": 1,
        "policy": "confirmatory_postseal_attestation_v1",
        "run_id": str(completion["run_id"]),
        "completion_stage": "CONFIRMATORY_COMPLETE",
        "first_integrity_root_sha256": artifact_root_sha256,
        "final_integrity_root_sha256": artifact_root_sha256,
        "matrix_plan_sha256": readback.matrix_plan_sha256,
        "cell_index_sha256": readback.cell_index_sha256,
        "scientific_artifact_manifest_sha256": (readback.root_artifact_manifest_sha256),
        "reconciliation_sha256": reconciliation_sha256,
        "confirmatory_storage_policy_sha256": storage_policy_sha256,
        "semantic_readback_status": readback.status,
        "semantic_checked_artifact_count": readback.checked_artifact_count,
    }


def _append_run_stage_attestation_unlocked(
    *,
    registry_path: Path,
    binding: Mapping[str, str],
    completion_stage: str | None,
    completion_path: Path,
    verification_payload: Mapping[str, Any],
    event_type: str = "postseal_stage_eligibility_attested",
    scientific_stage_eligible: bool = True,
) -> dict[str, Any]:
    records = _read_run_stage_attestations_unlocked(registry_path)
    _require_run_stage_attestation_anchor_unlocked(registry_path, records)
    if any(record["run_id"] == binding["run_id"] for record in records):
        raise ValueError("run already has a positive stage attestation")
    verification_sha256 = hashlib.sha256(
        json.dumps(
            dict(verification_payload),
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    record: dict[str, Any] = {
        "schema_version": 1,
        "sequence": len(records) + 1,
        "event_type": event_type,
        "recorded_at_utc": utc_now(),
        **binding,
        "scientific_stage_eligible": scientific_stage_eligible,
        "completion_stage": completion_stage,
        "completion_evidence_sha256": sha256_file(completion_path),
        "verification": dict(verification_payload),
        "verification_sha256": verification_sha256,
        "previous_record_sha256": records[-1]["record_sha256"] if records else None,
    }
    record["record_sha256"] = _run_disposition_record_sha256(record)
    _validate_run_stage_attestation_record(
        record,
        line_number=len(records) + 1,
        previous_record_sha256=record["previous_record_sha256"],
    )
    encoded = json.dumps(
        record,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    previous_ledger = _run_stage_attestation_ledger_bytes(registry_path)
    next_ledger = previous_ledger + encoded + b"\n"
    anchor_path = _run_stage_attestation_anchor_path(registry_path)
    previous_anchor_sha256 = sha256_file(anchor_path)
    try:
        atomic_write_bytes(registry_path, next_ledger)
        if sha256_file(anchor_path) != previous_anchor_sha256:
            raise ValueError("run-stage attestation anchor changed before CAS publication")
        atomic_write_json(
            anchor_path,
            _run_stage_attestation_anchor_payload((*records, record), next_ledger),
        )
        committed = _read_run_stage_attestations_unlocked(registry_path)
        _require_run_stage_attestation_anchor_unlocked(registry_path, committed)
        if committed != (*records, record):
            raise RuntimeError("run-stage attestation transaction failed exact readback")
    except BaseException as publication_error:
        # A replace can commit and only then report a durability error. Reconcile
        # under the same locks: an exact ledger+anchor commit is success, while
        # every partial or mismatched state stays fail-closed for all readers.
        try:
            committed = _read_run_stage_attestations_unlocked(registry_path)
            _require_run_stage_attestation_anchor_unlocked(registry_path, committed)
        except BaseException as reconciliation_error:
            raise publication_error from reconciliation_error
        if committed == (*records, record):
            return record
        raise publication_error
    return record


def _require_primary_retry_predecessor_binding(
    run_path: Path,
    *,
    completion: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    """Freshly revalidate one ordinary full-retry predecessor before attesting."""

    if payload.get("experiment_name") != "pannuke_primary_frozen_feature_benchmark":
        return
    retry_of_run_id = payload.get("retry_of_run_id")
    if retry_of_run_id is None:
        return
    if not isinstance(retry_of_run_id, str):
        raise ValueError("ordinary primary retry lacks an exact predecessor identity")

    retry_validator = getattr(
        importlib.import_module("histo_audit.experiment.primary_runner"),
        "_validate_retry_predecessor",
        None,
    )
    if not callable(retry_validator):
        raise ValueError(
            "ordinary primary retry lineage is unsupported by this bounded recovery build"
        )

    input_bindings = _read_sealed_json_object(
        run_path / "primary_input_bindings.json", "ordinary primary input bindings"
    )
    current_gate = _read_sealed_json_object(
        run_path / "primary_execution_gate.json", "ordinary primary execution gate"
    )
    provenance = _read_sealed_json_object(
        run_path / "run_provenance.json", "ordinary primary provenance"
    )
    sealed_predecessor = input_bindings.get("retry_predecessor")
    if not isinstance(sealed_predecessor, Mapping):
        raise ValueError("ordinary primary retry lacks its sealed predecessor binding")
    fresh_predecessor = retry_validator(
        run_root=run_path.parent,
        retry_of_run_id=retry_of_run_id,
        current_gate=current_gate,
    )
    binding_sha256 = fresh_predecessor.get("binding_sha256")
    if (
        dict(sealed_predecessor) != fresh_predecessor
        or not isinstance(binding_sha256, str)
        or _LOWER_SHA256_PATTERN.fullmatch(binding_sha256) is None
        or input_bindings.get("retry_of_run_id") != retry_of_run_id
        or input_bindings.get("retry_predecessor_binding_sha256") != binding_sha256
        or provenance.get("retry_of_run_id") != retry_of_run_id
        or provenance.get("retry_predecessor_binding_sha256") != binding_sha256
        or completion.get("retry_of_run_id") != retry_of_run_id
        or completion.get("retry_predecessor_binding_sha256") != binding_sha256
        or payload.get("lineage_binding_sha256") != binding_sha256
    ):
        raise ValueError("ordinary primary retry predecessor differs from its typed lineage")


def _require_primary_successor_authorization_binding(
    run_path: Path,
    *,
    completion: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    experiment_name = payload.get("experiment_name")
    if experiment_name == "pannuke_primary_frozen_feature_benchmark":
        if (
            completion.get("finalization_successor_authorization_sha256") is not None
            or completion.get("recovery_authorization_sha256") is not None
        ):
            raise ValueError("ordinary primary carries successor authorization")
        return
    if experiment_name == "pannuke_primary_orphan_recovery":
        if (
            completion.get("finalization_successor_authorization_sha256") is not None
            or completion.get("finalization_only_successor") is True
        ):
            raise ValueError("primary orphan recovery carries successor authorization")
        return
    if experiment_name != "pannuke_primary_finalization_successor":
        raise ValueError("primary attestation has an unsupported experiment")
    if (
        completion.get("recovery_authorization_sha256") is not None
        or completion.get("recovery_only") is True
    ):
        raise ValueError("primary successor carries orphan-recovery authorization")

    from histo_audit.corruption.controlled import canonical_sha256
    from histo_audit.workflows.preregistration_amendment import (
        _require_sealed_finalization_successor_authorization,
        require_authorized_prior_numeric_verification_proof,
    )

    require_bfast_authorization_claim = getattr(
        importlib.import_module("histo_audit.workflows.finalization_authorization_claims"),
        "require_bfast_authorization_claim",
        None,
    )
    if not callable(require_bfast_authorization_claim):
        raise ValueError("the retired finalization-successor authority is unavailable")

    gate = _read_sealed_json_object(
        run_path / "primary_execution_gate.json", "primary successor gate"
    )
    freeze_directory = gate.get("freeze_directory")
    if (
        not isinstance(freeze_directory, str)
        or not Path(freeze_directory).is_absolute()
        or str(Path(freeze_directory).resolve()) != freeze_directory
    ):
        raise ValueError("primary successor gate lacks an exact amendment directory")
    authorization = _require_sealed_finalization_successor_authorization(freeze_directory)
    authorization_sha = canonical_sha256(dict(authorization))
    provenance = _read_sealed_json_object(
        run_path / "run_provenance.json", "primary successor provenance"
    )
    environment = _read_sealed_json_object(
        run_path / "environment.json", "primary successor environment"
    )
    evidence = _read_sealed_json_object(
        run_path / "primary_finalization_successor_evidence.json",
        "primary successor lineage evidence",
    )
    predecessor = evidence.get("predecessor")
    authorized_predecessor = authorization.get("predecessor")
    authorized_numeric = authorization.get("numeric_verification")
    if not isinstance(predecessor, Mapping) or not isinstance(authorized_predecessor, Mapping):
        raise ValueError("primary successor authorization lacks predecessor bindings")
    predecessor_run_path = predecessor.get("run_path")
    if not isinstance(predecessor_run_path, str) or not Path(predecessor_run_path).is_absolute():
        raise ValueError("primary successor lineage lacks an absolute predecessor path")
    predecessor_source_manifest_sha = sha256_file(
        Path(predecessor_run_path).resolve() / SOURCE_TREE_MANIFEST_FILENAME
    )
    expected_predecessor = {
        "run_id": predecessor.get("run_id"),
        "run_directory": predecessor.get("run_path"),
        "terminal_status": "failed",
        "artifact_root_sha256": predecessor.get("artifact_root_sha256"),
        "artifact_manifest_sha256": predecessor.get("artifact_manifest_sha256"),
        "execution_source_root_sha256": predecessor.get("source_tree_root_sha256"),
        "execution_source_manifest_sha256": predecessor_source_manifest_sha,
    }
    if (
        completion.get("retry_of_run_id") != predecessor.get("run_id")
        or completion.get("predecessor_artifact_root_sha256")
        != predecessor.get("artifact_root_sha256")
        or completion.get("predecessor_artifact_manifest_sha256")
        != predecessor.get("artifact_manifest_sha256")
        or completion.get("predecessor_source_tree_root_sha256")
        != predecessor.get("source_tree_root_sha256")
        or any(
            authorized_predecessor.get(field) != value
            for field, value in expected_predecessor.items()
        )
        or completion.get("finalization_successor_authorization_sha256") != authorization_sha
        or provenance.get("finalization_successor_authorization_sha256") != authorization_sha
        or payload.get("authorization_binding_sha256") != authorization_sha
        or (
            authorized_numeric is not None
            and (
                not isinstance(authorized_numeric, Mapping)
                or authorized_numeric.get("mode") != completion.get("verification_mode")
                or authorized_numeric.get("prior_numeric_verification_proof_sha256")
                != completion.get("prior_numeric_verification_proof_sha256")
                or evidence.get("verification_mode") != completion.get("verification_mode")
                or evidence.get("prior_numeric_verification_proof_sha256")
                != completion.get("prior_numeric_verification_proof_sha256")
                or provenance.get("verification_mode") != completion.get("verification_mode")
                or provenance.get("prior_numeric_verification_proof_sha256")
                != completion.get("prior_numeric_verification_proof_sha256")
            )
        )
        or (
            authorized_numeric is None
            and (
                completion.get("verification_mode") is not None
                or evidence.get("verification_mode") is not None
            )
        )
    ):
        raise ValueError("primary successor authorization differs from sealed lineage")
    if authorized_numeric is not None:
        claim = evidence.get("authorization_claim")
        if not isinstance(claim, Mapping):
            raise ValueError("primary B-fast successor lacks its one-shot claim")
        claim_record_sha = claim.get("claim_record_sha256")
        if (
            not isinstance(claim_record_sha, str)
            or _LOWER_SHA256_PATTERN.fullmatch(claim_record_sha) is None
        ):
            raise ValueError("primary B-fast successor claim record SHA-256 is invalid")
        capability = require_authorized_prior_numeric_verification_proof(
            freeze_directory,
            canonical_authorization=authorization,
        )
        claim_readback = require_bfast_authorization_claim(
            authorization=capability,
            runs_root=run_path.parent,
            successor_run_id=run_path.name,
            claim_record_sha256=claim_record_sha,
        )
        if (
            not capability.valid
            or capability.authorization_sha256 != authorization_sha
            or claim_readback.as_dict() != dict(claim)
            or completion.get("authorization_claim") != dict(claim)
            or provenance.get("authorization_claim") != dict(claim)
            or environment.get("authorization_claim") != dict(claim)
            or environment.get("finalization_successor_authorization_sha256") != authorization_sha
            or environment.get("verification_mode") != completion.get("verification_mode")
            or environment.get("prior_numeric_verification_proof_sha256")
            != completion.get("prior_numeric_verification_proof_sha256")
        ):
            raise ValueError("primary B-fast successor claim differs from live one-shot authority")


def _require_primary_recovery_authorization_binding(
    run_path: Path,
    *,
    completion: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    """Revalidate one sealed orphan-recovery lineage without opening outcomes.

    The positive stage ledger binds the exact recovery evidence file through
    ``lineage_binding_sha256``.  This verifier independently reopens the immutable
    amendment and requires the source/destination snapshot root, authorization
    digest, retry identity, and no-execution declarations to agree everywhere.
    """

    experiment_name = payload.get("experiment_name")
    if experiment_name != "pannuke_primary_orphan_recovery":
        if completion.get("recovery_only") is True:
            raise ValueError("non-recovery primary carries orphan-recovery semantics")
        return

    from histo_audit.corruption.controlled import canonical_sha256
    from histo_audit.experiment.primary_recovery import (
        RECOVERY_COPY_POLICY,
        RECOVERY_EVIDENCE_FILENAME,
        RECOVERY_POLICY,
        RECOVERY_REGISTRATION_STATUS,
    )
    from histo_audit.experiment.primary_statistics import (
        INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE,
    )
    from histo_audit.workflows.preregistration_amendment import (
        require_primary_recovery_authorization,
    )

    gate = _read_sealed_json_object(
        run_path / "primary_execution_gate.json",
        "primary orphan-recovery gate",
    )
    freeze_directory = gate.get("freeze_directory")
    if (
        not isinstance(freeze_directory, str)
        or not Path(freeze_directory).is_absolute()
        or str(Path(freeze_directory).resolve()) != freeze_directory
    ):
        raise ValueError("primary orphan recovery gate lacks an exact amendment directory")
    authorization = require_primary_recovery_authorization(freeze_directory)
    authorization_sha256 = canonical_sha256(dict(authorization))
    evidence_path = run_path / RECOVERY_EVIDENCE_FILENAME
    evidence = _read_sealed_json_object(
        evidence_path,
        "primary orphan-recovery evidence",
    )
    evidence_sha256 = sha256_file(evidence_path)
    provenance = _read_sealed_json_object(
        run_path / "run_provenance.json",
        "primary orphan-recovery provenance",
    )
    forbidden_predecessor_fields = {
        "finalization_only_successor",
        "finalization_successor_authorization_sha256",
        "predecessor_artifact_root_sha256",
        "predecessor_artifact_manifest_sha256",
        "predecessor_source_tree_root_sha256",
    }
    if forbidden_predecessor_fields.intersection(completion) or (
        forbidden_predecessor_fields.intersection(provenance)
    ):
        raise ValueError("primary orphan recovery must not claim a sealed predecessor")

    source_run_id = authorization.get("source_run_id")
    source_snapshot_root = authorization.get("expected_source_snapshot_root_sha256")
    proof_sha256 = completion.get("prior_numeric_verification_proof_sha256")
    copy_policy = evidence.get("copy_policy")
    copied_artifact_count = evidence.get("copied_artifact_count")
    copied_total_bytes = evidence.get("copied_total_bytes")
    if (
        not isinstance(source_run_id, str)
        or not isinstance(source_snapshot_root, str)
        or _LOWER_SHA256_PATTERN.fullmatch(source_snapshot_root) is None
        or not isinstance(proof_sha256, str)
        or _LOWER_SHA256_PATTERN.fullmatch(proof_sha256) is None
        or copy_policy != RECOVERY_COPY_POLICY
        or type(copied_artifact_count) is not int
        or copied_artifact_count <= 0
        or type(copied_total_bytes) is not int
        or copied_total_bytes <= 0
    ):
        raise ValueError("primary orphan recovery authorization bindings are malformed")

    expected_common = {
        "recovery_policy": RECOVERY_POLICY,
        "retry_of_run_id": source_run_id,
        "primary_recovery_evidence_sha256": evidence_sha256,
        "recovery_evidence_sha256": evidence_sha256,
        "recovery_source_snapshot_root_sha256": source_snapshot_root,
        "recovery_authorization_sha256": authorization_sha256,
        "analysis_disposition": RECOVERY_REGISTRATION_STATUS,
        "outcomes_inspected": True,
        "verification_mode": INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE,
        "prior_numeric_verification_proof_sha256": proof_sha256,
        "reused_required_cell_count": 185,
        "skipped_optional_cell_count": 37,
        "retrained_cell_count": 0,
        "copy_policy": RECOVERY_COPY_POLICY,
        "copied_artifact_count": copied_artifact_count,
        "copied_total_bytes": copied_total_bytes,
        "training_invoked": False,
        "matrix_executor_invoked": False,
        "fallback_invoked": False,
        "automatic_retry_allowed": False,
    }
    if (
        completion.get("recovery_only") is not True
        or payload.get("retry_of_run_id") != source_run_id
        or payload.get("lineage_binding_sha256") != evidence_sha256
        or payload.get("authorization_binding_sha256") != authorization_sha256
        or completion.get("retry_predecessor_binding_sha256") != evidence_sha256
        or any(completion.get(field) != value for field, value in expected_common.items())
        or any(provenance.get(field) != value for field, value in expected_common.items())
    ):
        raise ValueError("primary orphan recovery differs from its sealed lineage")

    expected_evidence = {
        "schema_version": 1,
        "policy": RECOVERY_POLICY,
        "experiment_name": "pannuke_primary_orphan_recovery",
        "source_run_id": source_run_id,
        "destination_run_id": run_path.name,
        "recovery_authorization_sha256": authorization_sha256,
        "source_snapshot_root_sha256": source_snapshot_root,
        "destination_snapshot_root_sha256": source_snapshot_root,
        "analysis_disposition": RECOVERY_REGISTRATION_STATUS,
        "outcomes_inspected": True,
        "verification_mode": INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE,
        "prior_numeric_verification_proof_sha256": proof_sha256,
        "reused_required_cell_count": 185,
        "skipped_optional_cell_count": 37,
        "retrained_cell_count": 0,
        "copy_policy": RECOVERY_COPY_POLICY,
        "copied_artifact_count": copied_artifact_count,
        "copied_total_bytes": copied_total_bytes,
        "training_invoked": False,
        "matrix_executor_invoked": False,
        "fallback_invoked": False,
        "automatic_retry_allowed": False,
    }
    if dict(evidence) != expected_evidence:
        raise ValueError("primary orphan-recovery evidence differs from its authorization")


def _lifecycle_readiness_evidence_bindings(
    run_path: Path,
) -> tuple[Path, dict[str, Any], str, str]:
    """Validate the immutable non-scientific readiness record's self-bindings."""

    from histo_audit.corruption.controlled import canonical_sha256

    evidence_path = run_path / "lifecycle_readiness_evidence.json"
    evidence = _read_sealed_json_object(evidence_path, "lifecycle readiness evidence")
    unsigned = dict(evidence)
    readiness_record_sha256 = unsigned.pop("readiness_record_sha256", None)
    qualification_binding_sha256 = unsigned.get("qualification_binding_sha256")
    for field_name, value in (
        ("readiness_record_sha256", readiness_record_sha256),
        ("qualification_binding_sha256", qualification_binding_sha256),
    ):
        if not isinstance(value, str) or _LOWER_SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"lifecycle readiness evidence has invalid {field_name}")
    if canonical_sha256(unsigned) != readiness_record_sha256:
        raise ValueError("lifecycle readiness record checksum is not self-consistent")
    if (
        unsigned.get("schema_version") != 1
        or unsigned.get("policy") != "fresh_process_lifecycle_readiness_v1"
        or unsigned.get("decision") != "passed"
        or unsigned.get("scientific_outcome") is not False
        or unsigned.get("project_completion_status_changed") is not False
    ):
        raise ValueError("lifecycle readiness evidence is not a positive non-scientific record")
    return (
        evidence_path,
        evidence,
        str(qualification_binding_sha256),
        str(readiness_record_sha256),
    )


def build_lifecycle_qualification_attestation_verification(
    run_directory: str | Path,
    *,
    integrity: IntegrityVerification,
) -> LifecycleQualificationAttestationVerification:
    """Build a typed post-seal proof for the exact lifecycle-readiness experiment."""

    run_path = Path(run_directory).resolve()
    if (
        not isinstance(integrity, IntegrityVerification)
        or not integrity.valid
        or not integrity.registry_record_present
        or integrity.run_id != run_path.name
        or integrity.expected_root_sha256 != integrity.actual_root_sha256
    ):
        raise ValueError("lifecycle qualification requires a valid registry-backed seal")
    first_root = integrity.expected_root_sha256
    if not isinstance(first_root, str) or _LOWER_SHA256_PATTERN.fullmatch(first_root) is None:
        raise ValueError("lifecycle qualification integrity root is invalid")
    status = _read_sealed_json_object(run_path / STATUS_FILENAME, "lifecycle readiness status")
    if (
        status.get("status") != "completed"
        or status.get("run_id") != run_path.name
        or status.get("experiment_name") != _LIFECYCLE_QUALIFICATION_EXPERIMENT
    ):
        raise ValueError("sealed run is not the exact completed lifecycle-readiness experiment")
    evidence_path, _, qualification_sha, readiness_record_sha = (
        _lifecycle_readiness_evidence_bindings(run_path)
    )
    final_integrity = verify_run_integrity(run_path)
    if (
        not final_integrity.valid
        or not final_integrity.registry_record_present
        or final_integrity.run_id != run_path.name
        or final_integrity.expected_root_sha256 != first_root
        or final_integrity.actual_root_sha256 != first_root
    ):
        raise ValueError("lifecycle readiness changed during typed post-seal verification")
    verification = LifecycleQualificationAttestationVerification(
        policy=_LIFECYCLE_QUALIFICATION_POLICY,
        experiment_name=_LIFECYCLE_QUALIFICATION_EXPERIMENT,
        run_id=run_path.name,
        run_path=str(run_path),
        first_integrity_root_sha256=first_root,
        final_integrity_root_sha256=first_root,
        artifact_manifest_sha256=sha256_file(run_path / ARTIFACT_MANIFEST_FILENAME),
        readiness_evidence_sha256=sha256_file(evidence_path),
        qualification_binding_sha256=qualification_sha,
        readiness_record_sha256=readiness_record_sha,
    )
    _validate_lifecycle_qualification_verification_payload(verification.as_dict())
    object.__setattr__(
        verification,
        "_attestation",
        _LIFECYCLE_QUALIFICATION_ATTESTATION_TOKEN,
    )
    return verification


def _external_validation_ready_semantic_bundle_hashes(
    run_path: Path,
    semantic_verification: Any,
    *,
    expected_root_sha256: str,
    expected_integrity: IntegrityVerification | None = None,
) -> dict[str, str]:
    """Validate one result returned directly by the full sealed-run verifier."""

    bundle_verification_type = getattr(
        importlib.import_module("histo_audit.external_validation.m9_review_bundle"),
        "M9ReviewBundleVerification",
        None,
    )
    ready_verification_type = getattr(
        importlib.import_module("histo_audit.workflows.tracked_external_validation"),
        "ExternalValidationReadyRunVerification",
        None,
    )
    if not isinstance(bundle_verification_type, type) or not isinstance(
        ready_verification_type, type
    ):
        raise ValueError("external-validation-ready verifier types are unavailable")

    bundle_verification = getattr(semantic_verification, "bundle_verification", None)
    if (
        type(semantic_verification) is not ready_verification_type
        or type(bundle_verification) is not bundle_verification_type
    ):
        raise ValueError("external-validation-ready proof requires exact verifier result types")
    semantic_authority = cast(Any, semantic_verification)
    bundle_authority = cast(Any, bundle_verification)
    semantic_integrity = getattr(semantic_authority, "integrity", None)
    if (
        not semantic_authority.valid
        or semantic_authority.errors
        or semantic_authority.run_directory.resolve() != run_path
        or semantic_authority.run_id != run_path.name
        or semantic_authority.completion_stage != "EXTERNAL_VALIDATION_READY"
        or semantic_authority.stage_attested is not False
        or not isinstance(semantic_integrity, IntegrityVerification)
        or not semantic_integrity.valid
        or not semantic_integrity.registry_record_present
        or semantic_integrity.run_id != run_path.name
        or semantic_integrity.expected_root_sha256 != expected_root_sha256
        or semantic_integrity.actual_root_sha256 != expected_root_sha256
        or (expected_integrity is not None and semantic_integrity != expected_integrity)
    ):
        raise ValueError(
            "external-validation-ready proof requires fresh full post-seal semantic authority"
        )
    if (
        not bundle_authority.valid
        or bundle_authority.errors
        or bundle_authority.external_validation_ready_claimed is not False
        or bundle_authority.bundle_directory.resolve()
        != run_path / _EXTERNAL_VALIDATION_READY_BUNDLE_RELATIVE_PATH
        or type(bundle_authority.item_count) is not int
        or bundle_authority.item_count != 200
        or type(bundle_authority.asset_count) is not int
        or bundle_authority.asset_count != 600
    ):
        raise ValueError("full post-seal authority returned an invalid M9 bundle readback")
    bundle_hashes = {
        "contract_sha256": bundle_authority.contract_sha256,
        "cohort_payload_sha256": bundle_authority.cohort_payload_sha256,
        "public_tree_root_sha256": bundle_authority.public_tree_root_sha256,
        "private_tree_root_sha256": bundle_authority.private_tree_root_sha256,
        "bundle_root_sha256": bundle_authority.bundle_root_sha256,
    }
    if any(
        not isinstance(value, str) or _LOWER_SHA256_PATTERN.fullmatch(value) is None
        for value in bundle_hashes.values()
    ):
        raise ValueError("external-validation bundle readback has invalid hash bindings")
    return {field: str(value) for field, value in bundle_hashes.items()}


def build_external_validation_ready_attestation_verification(
    run_directory: str | Path,
    *,
    integrity: IntegrityVerification,
    material: Any,
    ranking_path: str | Path,
    confirmatory_eligibility_receipt: RunStageEligibilityReceipt,
) -> ExternalValidationReadyAttestationVerification:
    """Mint M9 proof only after the full post-seal semantic verifier passes freshly."""

    tracked_external = importlib.import_module("histo_audit.workflows.tracked_external_validation")
    refresh_active_m9_material = getattr(tracked_external, "refresh_active_m9_material", None)
    verify_external_validation_ready_run = getattr(
        tracked_external, "verify_external_validation_ready_run", None
    )
    if not callable(refresh_active_m9_material) or not callable(
        verify_external_validation_ready_run
    ):
        raise ValueError("external-validation-ready workflow is unavailable")

    run_path = Path(run_directory).resolve()
    if (
        not isinstance(integrity, IntegrityVerification)
        or not integrity.valid
        or not integrity.registry_record_present
        or integrity.run_id != run_path.name
        or integrity.expected_root_sha256 != integrity.actual_root_sha256
        or not isinstance(integrity.expected_root_sha256, str)
        or _LOWER_SHA256_PATTERN.fullmatch(integrity.expected_root_sha256) is None
    ):
        raise ValueError(
            "external-validation-ready verification requires a fresh valid completed seal"
        )
    if (
        not isinstance(confirmatory_eligibility_receipt, RunStageEligibilityReceipt)
        or confirmatory_eligibility_receipt.completion_stage != "CONFIRMATORY_COMPLETE"
    ):
        raise ValueError(
            "external-validation-ready proof requires active exact confirmatory-stage authority"
        )
    try:
        confirmatory_eligibility_receipt.require_active_authority()
    except ValueError as exc:
        raise ValueError(
            "external-validation-ready proof requires active exact confirmatory-stage authority"
        ) from exc
    _require_external_gate_matches_confirmatory_receipt(
        run_path,
        confirmatory_eligibility_receipt,
    )
    refreshed_material = refresh_active_m9_material(
        material,
        confirmatory_eligibility_receipt=confirmatory_eligibility_receipt,
    )
    resolved_ranking_path = Path(ranking_path).resolve()
    semantic_verification = verify_external_validation_ready_run(
        run_path,
        material=refreshed_material,
        ranking_path=resolved_ranking_path,
        require_stage_attestation=False,
    )
    bundle_hashes = _external_validation_ready_semantic_bundle_hashes(
        run_path,
        semantic_verification,
        expected_root_sha256=str(integrity.expected_root_sha256),
        expected_integrity=integrity,
    )

    gate_path, _, completion_path, completion = _external_validation_ready_candidate_bindings(
        run_path
    )
    if any(completion.get(field) != value for field, value in bundle_hashes.items()):
        raise ValueError("sealed M9 completion differs from typed bundle verification")
    _require_external_gate_matches_confirmatory_receipt(
        run_path,
        confirmatory_eligibility_receipt,
    )

    final_integrity = verify_run_integrity(run_path)
    first_root = integrity.expected_root_sha256
    if (
        not final_integrity.valid
        or not final_integrity.registry_record_present
        or final_integrity.run_id != run_path.name
        or final_integrity.expected_root_sha256 != first_root
        or final_integrity.actual_root_sha256 != first_root
    ):
        raise ValueError("external-validation-ready run changed during typed verification")

    verification = ExternalValidationReadyAttestationVerification(
        policy=_EXTERNAL_VALIDATION_READY_ATTESTATION_POLICY,
        experiment_name=_EXTERNAL_VALIDATION_READY_EXPERIMENT,
        run_id=run_path.name,
        run_path=str(run_path),
        completion_stage="EXTERNAL_VALIDATION_READY",
        first_integrity_root_sha256=first_root,
        final_integrity_root_sha256=first_root,
        artifact_manifest_sha256=sha256_file(run_path / ARTIFACT_MANIFEST_FILENAME),
        completion_evidence_sha256=sha256_file(completion_path),
        external_validation_execution_gate_sha256=sha256_file(gate_path),
        item_count=int(completion["item_count"]),
        asset_count=int(completion["asset_count"]),
        expert_response_count=int(completion["expert_response_count"]),
        contract_sha256=str(completion["contract_sha256"]),
        cohort_payload_sha256=str(completion["cohort_payload_sha256"]),
        bundle_root_sha256=str(completion["bundle_root_sha256"]),
        public_tree_root_sha256=str(completion["public_tree_root_sha256"]),
        private_tree_root_sha256=str(completion["private_tree_root_sha256"]),
        raw_inventory_sha256=str(completion["raw_inventory_sha256"]),
        canonical_pannuke_manifest_sha256=str(completion["canonical_pannuke_manifest_sha256"]),
        original_ranking_sha256=str(completion["original_ranking_sha256"]),
        original_audit_run_directory=str(completion["original_audit_run_directory"]),
        original_audit_experiment_name=str(completion["original_audit_experiment_name"]),
        original_audit_run_id=str(completion["original_audit_run_id"]),
        original_audit_artifact_root_sha256=str(completion["original_audit_artifact_root_sha256"]),
        original_audit_eligibility_evidence_sha256=str(
            completion["original_audit_eligibility_evidence_sha256"]
        ),
        confirmatory_run_directory=str(completion["confirmatory_run_directory"]),
        confirmatory_run_id=str(completion["confirmatory_run_id"]),
        confirmatory_artifact_root_sha256=str(completion["confirmatory_artifact_root_sha256"]),
        confirmatory_completion_evidence_sha256=str(
            completion["confirmatory_completion_evidence_sha256"]
        ),
        confirmatory_stage_attestation_record_sha256=str(
            completion["confirmatory_stage_attestation_record_sha256"]
        ),
        confirmatory_stage_attestation_verification_sha256=str(
            completion["confirmatory_stage_attestation_verification_sha256"]
        ),
        technical_inspection_evidence_sha256=str(
            completion["technical_inspection_evidence_sha256"]
        ),
    )
    _validate_external_validation_ready_verification_payload(verification.as_dict())
    object.__setattr__(
        verification,
        "_confirmatory_receipt",
        confirmatory_eligibility_receipt,
    )
    object.__setattr__(
        verification,
        "_material_refresh_authority",
        refreshed_material,
    )
    object.__setattr__(verification, "_ranking_path", resolved_ranking_path)
    object.__setattr__(
        verification,
        "_attestation",
        _EXTERNAL_VALIDATION_READY_ATTESTATION_TOKEN,
    )
    return verification


def attest_lifecycle_run_qualification(
    run_directory: str | Path,
    *,
    verification: LifecycleQualificationAttestationVerification,
) -> dict[str, Any]:
    """Append one anchored, explicitly non-scientific lifecycle qualification."""

    if (
        not isinstance(verification, LifecycleQualificationAttestationVerification)
        or not verification.valid
    ):
        raise ValueError("lifecycle qualification requires a genuine typed verification")
    payload = _validate_lifecycle_qualification_verification_payload(verification.as_dict())
    run_path = Path(run_directory).resolve()
    registry_path = run_path.parent / RUN_STAGE_ATTESTATION_REGISTRY_FILENAME
    disposition_path = run_path.parent / RUN_DISPOSITION_REGISTRY_FILENAME
    with (
        _registry_lock(_run_mutation_lock_target(run_path)),
        _registry_lock(disposition_path),
        _registry_lock(registry_path),
    ):
        binding = _completed_run_disposition_binding(run_path)
        status = _read_sealed_json_object(run_path / STATUS_FILENAME, "lifecycle readiness status")
        evidence_path, _, qualification_sha, readiness_record_sha = (
            _lifecycle_readiness_evidence_bindings(run_path)
        )
        current_bindings = {
            "experiment_name": status.get("experiment_name"),
            "run_id": binding["run_id"],
            "run_path": binding["run_path"],
            "first_integrity_root_sha256": binding["artifact_root_sha256"],
            "final_integrity_root_sha256": binding["artifact_root_sha256"],
            "artifact_manifest_sha256": binding["artifact_manifest_sha256"],
            "readiness_evidence_sha256": sha256_file(evidence_path),
            "qualification_binding_sha256": qualification_sha,
            "readiness_record_sha256": readiness_record_sha,
        }
        if (
            status.get("status") != "completed"
            or status.get("experiment_name") != _LIFECYCLE_QUALIFICATION_EXPERIMENT
            or any(
                payload.get(field_name) != value for field_name, value in current_bindings.items()
            )
        ):
            raise ValueError("typed lifecycle verification differs from the fresh sealed binding")
        dispositions = _read_run_dispositions_unlocked(disposition_path)
        _require_run_disposition_anchor_matches_unlocked(disposition_path, dispositions)
        if any(record["run_id"] == binding["run_id"] for record in dispositions):
            raise ValueError("withdrawn run cannot receive a lifecycle qualification")
        return _append_run_stage_attestation_unlocked(
            registry_path=registry_path,
            binding=binding,
            completion_stage=None,
            completion_path=evidence_path,
            verification_payload=payload,
            event_type=_LIFECYCLE_QUALIFICATION_EVENT,
            scientific_stage_eligible=False,
        )


def attest_primary_run_stage_eligibility(
    run_directory: str | Path,
    *,
    verification: PrimaryStageAttestationVerification,
) -> dict[str, Any]:
    """Commit PRIMARY_STUDY_COMPLETE only from a genuine typed verification."""

    if not isinstance(verification, PrimaryStageAttestationVerification) or not verification.valid:
        raise ValueError("primary stage attestation requires a genuine typed verification")
    payload = _validate_primary_stage_verification_payload(verification.as_dict())
    run_path = Path(run_directory).resolve()
    registry_path = run_path.parent / RUN_STAGE_ATTESTATION_REGISTRY_FILENAME
    disposition_path = run_path.parent / RUN_DISPOSITION_REGISTRY_FILENAME
    with (
        _registry_lock(_run_mutation_lock_target(run_path)),
        _registry_lock(disposition_path),
        _registry_lock(registry_path),
    ):
        binding = _completed_run_disposition_binding(run_path)
        status = _read_sealed_json_object(run_path / STATUS_FILENAME, "primary status")
        completion_path = run_path / "completion_evidence.json"
        completion = _read_sealed_json_object(completion_path, "primary completion evidence")
        expected_policy = _PRIMARY_STAGE_EXPERIMENT_POLICIES.get(str(status.get("experiment_name")))
        if (
            status.get("status") != "completed"
            or status.get("experiment_name") != payload["experiment_name"]
            or expected_policy != payload["policy"]
            or completion.get("run_id") != binding["run_id"]
            or completion.get("completion_stage") != "PRIMARY_STUDY_COMPLETE"
            or completion.get("study_outcome_eligible") is not True
            or completion.get("post_seal_attestation_required") is not True
            or completion.get("retry_of_run_id") != payload["retry_of_run_id"]
            or completion.get("retry_predecessor_binding_sha256")
            != payload["lineage_binding_sha256"]
        ):
            raise ValueError("sealed primary completion is not an attestable candidate")
        current_bindings = {
            "run_id": binding["run_id"],
            "run_path": binding["run_path"],
            "first_integrity_root_sha256": binding["artifact_root_sha256"],
            "final_integrity_root_sha256": binding["artifact_root_sha256"],
            "artifact_manifest_sha256": binding["artifact_manifest_sha256"],
            "completion_evidence_sha256": sha256_file(completion_path),
        }
        if any(payload.get(field) != value for field, value in current_bindings.items()):
            raise ValueError("typed primary verification differs from the fresh sealed binding")
        _require_primary_verification_files_match(run_path, payload)
        _require_primary_retry_predecessor_binding(
            run_path,
            completion=completion,
            payload=payload,
        )
        _require_primary_successor_authorization_binding(
            run_path,
            completion=completion,
            payload=payload,
        )
        _require_primary_recovery_authorization_binding(
            run_path,
            completion=completion,
            payload=payload,
        )
        dispositions = _read_run_dispositions_unlocked(disposition_path)
        _require_run_disposition_anchor_matches_unlocked(disposition_path, dispositions)
        if any(record["run_id"] == binding["run_id"] for record in dispositions):
            raise ValueError("withdrawn run cannot receive a positive stage attestation")
        return _append_run_stage_attestation_unlocked(
            registry_path=registry_path,
            binding=binding,
            completion_stage="PRIMARY_STUDY_COMPLETE",
            completion_path=completion_path,
            verification_payload=payload,
        )


def _external_original_audit_run_path(run_path: Path) -> Path:
    """Read the audit identity needed to acquire A before the M9 mutation lock."""

    gate = _validate_external_validation_ready_candidate_payload(
        _read_sealed_json_object(
            run_path / _EXTERNAL_VALIDATION_READY_GATE_FILENAME,
            "external-validation execution gate",
        ),
        completion=False,
    )
    audit_path = Path(str(gate["original_audit_run_directory"])).resolve()
    if (
        audit_path in {run_path, Path(str(gate["confirmatory_run_directory"])).resolve()}
        or audit_path.name != gate["original_audit_run_id"]
        or audit_path.parent != run_path.parent
    ):
        raise ValueError("external-validation gate has an invalid original-audit identity")
    return audit_path


def _require_external_original_audit_authority_under_mutation_lock(
    run_path: Path,
    audit_path: Path,
    *,
    dispositions: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Require the exact non-withdrawn A binding while its mutation lock is held."""

    gate = _validate_external_validation_ready_candidate_payload(
        _read_sealed_json_object(
            run_path / _EXTERNAL_VALIDATION_READY_GATE_FILENAME,
            "external-validation execution gate",
        ),
        completion=False,
    )
    if Path(str(gate["original_audit_run_directory"])).resolve() != audit_path:
        raise ValueError("external-validation original-audit path changed before authorization")
    binding = _completed_run_disposition_binding(audit_path)
    matching = [record for record in dispositions if record.get("run_id") == binding["run_id"]]
    if matching:
        record = matching[0]
        for field_name in (
            "run_path",
            "terminal_status",
            "artifact_root_sha256",
            "artifact_manifest_sha256",
        ):
            if record.get(field_name) != binding[field_name]:
                raise ValueError(
                    "original-audit disposition differs from its sealed run; M9 fails closed"
                )
        raise ValueError(
            "scientific stage eligibility was permanently withdrawn for sealed original-audit "
            f"run {binding['run_id']}: reason_code={record['reason_code']}; "
            f"reason={record['reason']}"
        )
    status = _read_sealed_json_object(audit_path / STATUS_FILENAME, "original-audit status")
    eligibility_path = audit_path / "external_validation_eligibility.json"
    eligibility = _read_sealed_json_object(
        eligibility_path,
        "original-audit eligibility evidence",
    )
    expected = {
        "original_audit_run_directory": binding["run_path"],
        "original_audit_experiment_name": status.get("experiment_name"),
        "original_audit_run_id": binding["run_id"],
        "original_audit_artifact_root_sha256": binding["artifact_root_sha256"],
        "original_audit_eligibility_evidence_sha256": sha256_file(eligibility_path),
    }
    if (
        status.get("status") != "completed"
        or status.get("experiment_name") != "original_label_audit"
        or eligibility.get("schema_version") != 2
        or eligibility.get("workflow") != "exploratory_original_label_audit"
        or eligibility.get("study_outcome_eligible") is not True
        or any(gate.get(field) != value for field, value in expected.items())
    ):
        raise ValueError(
            "external-validation gate differs from exact non-withdrawn original-audit authority"
        )
    return binding


def _require_external_gate_matches_confirmatory_receipt(
    run_path: Path,
    receipt: RunStageEligibilityReceipt,
) -> None:
    if not isinstance(receipt, RunStageEligibilityReceipt):
        raise ValueError(
            "external-validation attestation requires active exact confirmatory-stage authority"
        )
    try:
        confirmatory_record = receipt.require_active_authority()
    except ValueError as exc:
        raise ValueError(
            "external-validation attestation requires active exact confirmatory-stage authority"
        ) from exc
    if receipt.completion_stage != "CONFIRMATORY_COMPLETE":
        raise ValueError(
            "external-validation attestation requires active exact confirmatory-stage authority"
        )
    _, gate, _, _ = _external_validation_ready_candidate_bindings(run_path)
    expected = {
        "confirmatory_run_directory": str(receipt.run_directory),
        "confirmatory_run_id": receipt.run_id,
        "confirmatory_artifact_root_sha256": confirmatory_record.get("artifact_root_sha256"),
        "confirmatory_completion_evidence_sha256": confirmatory_record.get(
            "completion_evidence_sha256"
        ),
        "confirmatory_stage_attestation_record_sha256": receipt.record_sha256,
        "confirmatory_stage_attestation_verification_sha256": (receipt.verification_sha256),
    }
    if any(gate.get(field) != value for field, value in expected.items()):
        raise ValueError(
            "active confirmatory-stage authority differs from the sealed external-validation gate"
        )


def attest_external_validation_ready(
    run_directory: str | Path,
    *,
    verification: ExternalValidationReadyAttestationVerification,
    confirmatory_eligibility_receipt: RunStageEligibilityReceipt,
) -> dict[str, Any]:
    """Atomically append EXTERNAL_VALIDATION_READY under a live P->C->M9 lease."""

    if (
        not isinstance(verification, ExternalValidationReadyAttestationVerification)
        or not verification.valid
        or verification._confirmatory_receipt is not confirmatory_eligibility_receipt
    ):
        raise ValueError(
            "external-validation-ready attestation requires a genuine typed verification"
        )
    payload = _validate_external_validation_ready_verification_payload(verification.as_dict())
    if not isinstance(confirmatory_eligibility_receipt, RunStageEligibilityReceipt):
        raise ValueError(
            "external-validation attestation requires active exact confirmatory-stage authority"
        )
    try:
        confirmatory_eligibility_receipt.require_active_authority()
    except ValueError as exc:
        raise ValueError(
            "external-validation attestation requires active exact confirmatory-stage authority"
        ) from exc
    if confirmatory_eligibility_receipt.completion_stage != "CONFIRMATORY_COMPLETE":
        raise ValueError(
            "external-validation attestation requires active exact confirmatory-stage authority"
        )

    run_path = Path(run_directory).resolve()
    audit_path = _external_original_audit_run_path(run_path)
    registry_path = run_path.parent / RUN_STAGE_ATTESTATION_REGISTRY_FILENAME
    disposition_path = run_path.parent / RUN_DISPOSITION_REGISTRY_FILENAME
    with (
        _registry_lock(_run_mutation_lock_target(audit_path)),
        _registry_lock(_run_mutation_lock_target(run_path)),
        _registry_lock(disposition_path),
        _registry_lock(registry_path),
    ):
        binding = _completed_run_disposition_binding(run_path)
        gate_path, _, completion_path, completion = _external_validation_ready_candidate_bindings(
            run_path
        )
        _require_external_gate_matches_confirmatory_receipt(
            run_path,
            confirmatory_eligibility_receipt,
        )
        current_bindings: dict[str, Any] = {
            "run_id": binding["run_id"],
            "run_path": binding["run_path"],
            "first_integrity_root_sha256": binding["artifact_root_sha256"],
            "final_integrity_root_sha256": binding["artifact_root_sha256"],
            "artifact_manifest_sha256": binding["artifact_manifest_sha256"],
            "completion_evidence_sha256": sha256_file(completion_path),
            "external_validation_execution_gate_sha256": sha256_file(gate_path),
            "item_count": completion["item_count"],
            "asset_count": completion["asset_count"],
            "expert_response_count": completion["expert_response_count"],
            "contract_sha256": completion["contract_sha256"],
            "cohort_payload_sha256": completion["cohort_payload_sha256"],
            "bundle_root_sha256": completion["bundle_root_sha256"],
            "public_tree_root_sha256": completion["public_tree_root_sha256"],
            "private_tree_root_sha256": completion["private_tree_root_sha256"],
            "raw_inventory_sha256": completion["raw_inventory_sha256"],
            "canonical_pannuke_manifest_sha256": completion["canonical_pannuke_manifest_sha256"],
            "original_ranking_sha256": completion["original_ranking_sha256"],
            "original_audit_run_directory": completion["original_audit_run_directory"],
            "original_audit_experiment_name": completion["original_audit_experiment_name"],
            "original_audit_run_id": completion["original_audit_run_id"],
            "original_audit_artifact_root_sha256": completion[
                "original_audit_artifact_root_sha256"
            ],
            "original_audit_eligibility_evidence_sha256": completion[
                "original_audit_eligibility_evidence_sha256"
            ],
            "confirmatory_run_directory": completion["confirmatory_run_directory"],
            "confirmatory_run_id": completion["confirmatory_run_id"],
            "confirmatory_artifact_root_sha256": completion["confirmatory_artifact_root_sha256"],
            "confirmatory_completion_evidence_sha256": completion[
                "confirmatory_completion_evidence_sha256"
            ],
            "confirmatory_stage_attestation_record_sha256": completion[
                "confirmatory_stage_attestation_record_sha256"
            ],
            "confirmatory_stage_attestation_verification_sha256": completion[
                "confirmatory_stage_attestation_verification_sha256"
            ],
            "technical_inspection_evidence_sha256": completion[
                "technical_inspection_evidence_sha256"
            ],
        }
        if any(payload.get(field) != value for field, value in current_bindings.items()):
            raise ValueError(
                "typed external-validation verification differs from the fresh sealed binding"
            )
        dispositions = _read_run_dispositions_unlocked(disposition_path)
        _require_run_disposition_anchor_matches_unlocked(disposition_path, dispositions)
        _require_external_original_audit_authority_under_mutation_lock(
            run_path,
            audit_path,
            dispositions=dispositions,
        )
        if any(record["run_id"] == binding["run_id"] for record in dispositions):
            raise ValueError("withdrawn run cannot receive a positive stage attestation")
        try:
            confirmatory_eligibility_receipt.require_active_authority()
        except ValueError as exc:  # pragma: no cover - same-thread guard invariant
            raise ValueError(
                "external-validation attestation lost active exact confirmatory-stage authority"
            ) from exc
        _require_external_gate_matches_confirmatory_receipt(
            run_path,
            confirmatory_eligibility_receipt,
        )
        material_refresh_authority = verification._material_refresh_authority
        ranking_path = verification._ranking_path
        if material_refresh_authority is None or not isinstance(ranking_path, Path):
            raise ValueError(
                "external-validation-ready proof lacks live material refresh authority"
            )
        tracked_external = importlib.import_module(
            "histo_audit.workflows.tracked_external_validation"
        )
        refresh_active_m9_material = getattr(tracked_external, "refresh_active_m9_material", None)
        verify_external_validation_ready_run = getattr(
            tracked_external, "verify_external_validation_ready_run", None
        )
        if not callable(refresh_active_m9_material) or not callable(
            verify_external_validation_ready_run
        ):
            raise ValueError("external-validation-ready workflow is unavailable")

        refreshed_material = refresh_active_m9_material(
            material_refresh_authority,
            confirmatory_eligibility_receipt=confirmatory_eligibility_receipt,
        )
        semantic_verification = verify_external_validation_ready_run(
            run_path,
            material=refreshed_material,
            ranking_path=ranking_path,
            require_stage_attestation=False,
        )
        fresh_bundle_hashes = _external_validation_ready_semantic_bundle_hashes(
            run_path,
            semantic_verification,
            expected_root_sha256=str(binding["artifact_root_sha256"]),
        )
        if any(payload.get(field) != value for field, value in fresh_bundle_hashes.items()):
            raise ValueError(
                "external-validation-ready proof differs from fresh live material readback"
            )
        try:
            confirmatory_eligibility_receipt.require_active_authority()
        except ValueError as exc:  # pragma: no cover - same-thread guard invariant
            raise ValueError(
                "external-validation attestation lost active exact confirmatory-stage authority"
            ) from exc
        _require_external_gate_matches_confirmatory_receipt(
            run_path,
            confirmatory_eligibility_receipt,
        )
        _require_external_original_audit_authority_under_mutation_lock(
            run_path,
            audit_path,
            dispositions=dispositions,
        )
        final_integrity = verify_run_integrity(run_path)
        if (
            not final_integrity.valid
            or not final_integrity.registry_record_present
            or final_integrity.run_id != binding["run_id"]
            or final_integrity.expected_root_sha256 != binding["artifact_root_sha256"]
            or final_integrity.actual_root_sha256 != binding["artifact_root_sha256"]
            or sha256_file(run_path / ARTIFACT_MANIFEST_FILENAME)
            != binding["artifact_manifest_sha256"]
            or payload["first_integrity_root_sha256"] != final_integrity.actual_root_sha256
            or payload["final_integrity_root_sha256"] != final_integrity.actual_root_sha256
        ):
            raise ValueError(
                "external-validation seal changed before positive attestation; proof is stale"
            )
        return _append_run_stage_attestation_unlocked(
            registry_path=registry_path,
            binding=binding,
            completion_stage="EXTERNAL_VALIDATION_READY",
            completion_path=completion_path,
            verification_payload=payload,
        )


def attest_run_stage_eligibility(
    run_directory: str | Path,
    *,
    completion_stage: str,
    verification: Mapping[str, Any],
    primary_eligibility_receipt: RunStageEligibilityReceipt | None = None,
) -> dict[str, Any]:
    """Atomically commit the final positive post-seal confirmatory attestation."""

    if completion_stage != "CONFIRMATORY_COMPLETE":
        raise ValueError("positive post-seal attestation supports CONFIRMATORY_COMPLETE only")
    run_path = Path(run_directory).resolve()
    if not (run_path / "confirmatory_execution_gate.json").is_file():
        raise ValueError("sealed confirmatory gate evidence is unavailable")
    if not isinstance(primary_eligibility_receipt, RunStageEligibilityReceipt):
        raise ValueError(
            "confirmatory attestation requires active exact primary-stage eligibility authority"
        )
    try:
        primary_eligibility_receipt.require_active_authority()
    except ValueError as exc:
        raise ValueError(
            "confirmatory attestation requires active exact primary-stage eligibility authority"
        ) from exc
    if primary_eligibility_receipt.completion_stage != "PRIMARY_STUDY_COMPLETE":
        raise ValueError(
            "confirmatory attestation requires active exact primary-stage eligibility authority"
        )
    registry_path = run_path.parent / RUN_STAGE_ATTESTATION_REGISTRY_FILENAME
    disposition_path = run_path.parent / RUN_DISPOSITION_REGISTRY_FILENAME
    with (
        _registry_lock(_run_mutation_lock_target(run_path)),
        _registry_lock(disposition_path),
        _registry_lock(registry_path),
    ):
        binding = _completed_run_disposition_binding(run_path)
        try:
            status_payload = json.loads((run_path / STATUS_FILENAME).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("sealed status is unavailable for stage attestation") from exc
        if (
            not isinstance(status_payload, Mapping)
            or status_payload.get("experiment_name") != "pannuke_confirmatory_study"
        ):
            raise ValueError("positive confirmatory attestation requires the exact experiment")
        _require_confirmatory_gate_matches_primary_receipt(
            run_path,
            primary_eligibility_receipt,
        )
        completion_path = run_path / "completion_evidence.json"
        try:
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("sealed completion evidence is unavailable for attestation") from exc
        if (
            not isinstance(completion, Mapping)
            or completion.get("completion_stage") != completion_stage
            or completion.get("study_outcome_eligible") is not True
            or completion.get("post_seal_attestation_required") is not True
            or completion.get("run_id") != binding["run_id"]
        ):
            raise ValueError("sealed completion evidence is not an attestable candidate")
        verification_payload = _independent_confirmatory_attestation_verification(
            run_path,
            completion=completion,
            artifact_root_sha256=str(binding["artifact_root_sha256"]),
        )
        if dict(verification) != verification_payload:
            raise ValueError(
                "caller post-seal verification differs from independent scientific readback"
            )
        dispositions = _read_run_dispositions_unlocked(disposition_path)
        _require_run_disposition_anchor_matches_unlocked(disposition_path, dispositions)
        if any(record["run_id"] == binding["run_id"] for record in dispositions):
            raise ValueError("withdrawn run cannot receive a positive stage attestation")
        try:
            primary_eligibility_receipt.require_active_authority()
        except ValueError as exc:  # pragma: no cover - same-thread guard invariant
            raise ValueError(
                "confirmatory attestation lost active exact primary-stage eligibility authority"
            ) from exc
        return _append_run_stage_attestation_unlocked(
            registry_path=registry_path,
            binding=binding,
            completion_stage=completion_stage,
            completion_path=completion_path,
            verification_payload=verification_payload,
        )


def _require_run_stage_eligible_under_mutation_lock(
    run_directory: str | Path,
) -> _ValidatedRunStageAttestation | None:
    """Return private validated authority while holding the mutation lock.

    Ordinary completed runs that carry no scientific completion stage return
    ``None``.  Stage-bearing primary, confirmatory, and M9 package runs return the
    single ledger record validated under the same per-run mutation lock, allowing
    downstream consumers to bind its stable record and verification hashes.
    """

    run_path = Path(run_directory).resolve()
    registry_path = run_path.parent / RUN_DISPOSITION_REGISTRY_FILENAME
    with _registry_lock(registry_path):
        # Never trust a previously computed object at this authorization boundary:
        # artifacts may have changed between that verification and this call.  The
        # per-run mutation lock held by the wrapper makes this fresh re-hash atomic
        # with the withdrawal/attestation decision.
        binding = _completed_run_disposition_binding(run_path)
        records = _read_run_dispositions_unlocked(registry_path)
        _require_run_disposition_anchor_matches_unlocked(registry_path, records)
    matching = [record for record in records if record["run_id"] == binding["run_id"]]
    if matching:
        record = matching[0]
        for field_name in (
            "run_path",
            "terminal_status",
            "artifact_root_sha256",
            "artifact_manifest_sha256",
        ):
            if record[field_name] != binding[field_name]:
                raise ValueError(
                    f"run disposition {field_name} does not match the sealed run; stage "
                    "eligibility fails closed"
                )
        raise ValueError(
            "scientific stage eligibility was permanently withdrawn for sealed run "
            f"{binding['run_id']}: reason_code={record['reason_code']}; "
            f"reason={record['reason']}"
        )

    try:
        status_payload = json.loads((run_path / STATUS_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("sealed status is unreadable; eligibility fails closed") from exc
    if not isinstance(status_payload, Mapping):
        raise ValueError("sealed status is not an object; eligibility fails closed")
    experiment_name = status_payload.get("experiment_name")
    is_confirmatory = experiment_name == "pannuke_confirmatory_study"
    is_primary = experiment_name in _PRIMARY_STAGE_EXPERIMENT_POLICIES
    is_external_validation = experiment_name == _EXTERNAL_VALIDATION_READY_EXPERIMENT
    completion_path = run_path / "completion_evidence.json"
    if not completion_path.is_file():
        if is_confirmatory or is_primary or is_external_validation:
            raise ValueError(
                "sealed scientific run lacks completion evidence and positive attestation"
            )
        return None
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("completion evidence is unreadable; eligibility fails closed") from exc
    if not isinstance(completion, Mapping):
        raise ValueError("completion evidence is not an object; eligibility fails closed")
    if is_primary:
        expected_stage = "PRIMARY_STUDY_COMPLETE"
    elif is_confirmatory:
        expected_stage = "CONFIRMATORY_COMPLETE"
    else:
        expected_stage = "EXTERNAL_VALIDATION_READY"
    if completion.get("completion_stage") != expected_stage:
        if is_confirmatory or is_primary or is_external_validation:
            raise ValueError(f"sealed scientific run lacks a {expected_stage} candidate")
        if completion.get("completion_stage") in {
            "PRIMARY_STUDY_COMPLETE",
            "CONFIRMATORY_COMPLETE",
            "EXTERNAL_VALIDATION_READY",
        }:
            raise ValueError("stage-bearing completion belongs to an unrecognized experiment")
        return None
    if (
        completion.get("study_outcome_eligible") is not True
        or completion.get("post_seal_attestation_required") is not True
    ):
        raise ValueError(
            f"{expected_stage} is default-deny without the mandatory post-seal "
            "attestation requirement"
        )
    attestation_path = run_path.parent / RUN_STAGE_ATTESTATION_REGISTRY_FILENAME
    attestations = read_run_stage_attestations(attestation_path)
    matched = [record for record in attestations if record["run_id"] == binding["run_id"]]
    if len(matched) != 1:
        raise ValueError(f"{expected_stage} lacks one durable positive post-seal attestation")
    attestation = matched[0]
    expected = {
        **binding,
        "completion_stage": expected_stage,
        "completion_evidence_sha256": sha256_file(completion_path),
    }
    for field_name, value in expected.items():
        if attestation.get(field_name) != value:
            raise ValueError(f"post-seal attestation {field_name} differs from the sealed run")
    if is_primary:
        verification = _validate_primary_stage_verification_payload(
            attestation.get("verification"), record=attestation
        )
        expected_policy = _PRIMARY_STAGE_EXPERIMENT_POLICIES[str(experiment_name)]
        if (
            verification.get("experiment_name") != experiment_name
            or verification.get("policy") != expected_policy
        ):
            raise ValueError("primary post-seal attestation has the wrong experiment policy")
        _require_primary_successor_authorization_binding(
            run_path,
            completion=completion,
            payload=verification,
        )
        _require_primary_recovery_authorization_binding(
            run_path,
            completion=completion,
            payload=verification,
        )
    elif is_external_validation:
        _, _, current_completion_path, current_completion = (
            _external_validation_ready_candidate_bindings(run_path)
        )
        verification = _validate_external_validation_ready_verification_payload(
            attestation.get("verification"), record=attestation
        )
        current_bindings = {
            "run_id": binding["run_id"],
            "run_path": binding["run_path"],
            "first_integrity_root_sha256": binding["artifact_root_sha256"],
            "final_integrity_root_sha256": binding["artifact_root_sha256"],
            "artifact_manifest_sha256": binding["artifact_manifest_sha256"],
            "completion_evidence_sha256": sha256_file(current_completion_path),
            "external_validation_execution_gate_sha256": current_completion[
                "external_validation_execution_gate_sha256"
            ],
        }
        current_bindings.update(
            {
                field: current_completion[field]
                for field in (
                    "item_count",
                    "asset_count",
                    "expert_response_count",
                    "contract_sha256",
                    "cohort_payload_sha256",
                    "bundle_root_sha256",
                    "public_tree_root_sha256",
                    "private_tree_root_sha256",
                    "raw_inventory_sha256",
                    "canonical_pannuke_manifest_sha256",
                    "original_ranking_sha256",
                    "original_audit_run_directory",
                    "original_audit_experiment_name",
                    "original_audit_run_id",
                    "original_audit_artifact_root_sha256",
                    "original_audit_eligibility_evidence_sha256",
                    "confirmatory_run_directory",
                    "confirmatory_run_id",
                    "confirmatory_artifact_root_sha256",
                    "confirmatory_completion_evidence_sha256",
                    "confirmatory_stage_attestation_record_sha256",
                    "confirmatory_stage_attestation_verification_sha256",
                    "technical_inspection_evidence_sha256",
                )
            }
        )
        if any(verification.get(field) != value for field, value in current_bindings.items()):
            raise ValueError(
                "external-validation-ready attestation differs from current sealed bindings"
            )
    record_sha256 = attestation.get("record_sha256")
    verification_sha256 = attestation.get("verification_sha256")
    if (
        not isinstance(record_sha256, str)
        or _LOWER_SHA256_PATTERN.fullmatch(record_sha256) is None
        or not isinstance(verification_sha256, str)
        or _LOWER_SHA256_PATTERN.fullmatch(verification_sha256) is None
    ):
        raise ValueError("validated stage-attestation hashes are malformed")
    validated = _ValidatedRunStageAttestation(
        run_directory=run_path,
        run_id=str(binding["run_id"]),
        completion_stage=expected_stage,
        record_sha256=record_sha256,
        verification_sha256=verification_sha256,
        canonical_record_json=json.dumps(
            dict(attestation),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )
    object.__setattr__(
        validated,
        "_attestation",
        _VALIDATED_RUN_STAGE_ATTESTATION_TOKEN,
    )
    return validated


@contextmanager
def _guard_single_run_stage_eligibility(
    run_directory: str | Path,
) -> Iterator[RunStageEligibilityReceipt | None]:
    """Hold one per-run mutation lock while yielding fresh stage authority.

    The yielded receipt is the only form that may authorize an external action.
    It becomes inactive when the context exits, so a cached preliminary gate can
    never be reused as execution authority.
    """

    run_path = Path(run_directory).resolve()
    state = _RunStageEligibilityGuardState(
        owner_process_id=os.getpid(),
        owner_thread_id=threading.get_ident(),
    )
    try:
        with _registry_lock(_run_mutation_lock_target(run_path)):
            try:
                validated = _require_run_stage_eligible_under_mutation_lock(run_path)
                receipt: RunStageEligibilityReceipt | None = None
                if validated is not None:
                    if (
                        not isinstance(validated, _ValidatedRunStageAttestation)
                        or not validated.valid
                        or validated.run_directory != run_path
                        or validated.run_id != run_path.name
                        or validated.completion_stage
                        not in {
                            "PRIMARY_STUDY_COMPLETE",
                            "CONFIRMATORY_COMPLETE",
                            "EXTERNAL_VALIDATION_READY",
                        }
                    ):
                        raise ValueError(
                            "run-stage validation did not return genuine exact authority"
                        )
                    receipt = RunStageEligibilityReceipt(
                        run_directory=run_path,
                        run_id=validated.run_id,
                        completion_stage=validated.completion_stage,
                        record_sha256=validated.record_sha256,
                        verification_sha256=validated.verification_sha256,
                        _canonical_record_json=validated.canonical_record_json,
                    )
                    # This is the sole production mint.  It executes only after
                    # _require_run_stage_eligible_under_mutation_lock completed
                    # every integrity, withdrawal, ledger-chain, anchor, and
                    # verification-payload check while this lock remains held.
                    object.__setattr__(receipt, "_guard_state", state)
                    object.__setattr__(
                        receipt,
                        "_attestation",
                        _RUN_STAGE_ELIGIBILITY_RECEIPT_TOKEN,
                    )
                yield receipt
            finally:
                # Revoke the lease before the mutation lock becomes available.
                state._revoke(_RUN_STAGE_ELIGIBILITY_RECEIPT_TOKEN)
    finally:
        # Defensive idempotent revocation if lock acquisition/validation failed.
        state._revoke(_RUN_STAGE_ELIGIBILITY_RECEIPT_TOKEN)


def _confirmatory_primary_run_path(run_path: Path) -> Path | None:
    status = _read_sealed_json_object(run_path / STATUS_FILENAME, "run status")
    if status.get("experiment_name") != "pannuke_confirmatory_study":
        return None
    gate = _read_sealed_json_object(
        run_path / "confirmatory_execution_gate.json",
        "confirmatory execution gate",
    )
    raw_primary_path = gate.get("primary_run_directory")
    if not isinstance(raw_primary_path, str) or not Path(raw_primary_path).is_absolute():
        raise ValueError("sealed confirmatory gate lacks an absolute primary run binding")
    primary_path = Path(raw_primary_path).resolve()
    if primary_path == run_path:
        raise ValueError("sealed confirmatory gate cannot bind itself as upstream primary")
    return primary_path


def _external_confirmatory_run_path(run_path: Path) -> Path | None:
    status = _read_sealed_json_object(run_path / STATUS_FILENAME, "run status")
    if status.get("experiment_name") != _EXTERNAL_VALIDATION_READY_EXPERIMENT:
        return None
    _, gate, _, _ = _external_validation_ready_candidate_bindings(run_path)
    confirmatory_path = Path(str(gate["confirmatory_run_directory"])).resolve()
    if confirmatory_path == run_path:
        raise ValueError("sealed external-validation gate cannot bind itself as confirmatory")
    return confirmatory_path


def _require_confirmatory_gate_matches_primary_receipt(
    run_path: Path,
    receipt: RunStageEligibilityReceipt,
) -> None:
    if not isinstance(receipt, RunStageEligibilityReceipt):
        raise ValueError(
            "confirmatory lineage requires active exact primary-stage eligibility authority"
        )
    try:
        primary_record = receipt.require_active_authority()
    except ValueError as exc:
        raise ValueError(
            "confirmatory lineage requires active exact primary-stage eligibility authority"
        ) from exc
    if receipt.completion_stage != "PRIMARY_STUDY_COMPLETE":
        raise ValueError(
            "confirmatory lineage requires active exact primary-stage eligibility authority"
        )
    gate = _read_sealed_json_object(
        run_path / "confirmatory_execution_gate.json",
        "confirmatory execution gate",
    )
    expected = {
        "primary_run_directory": str(receipt.run_directory),
        "primary_run_id": receipt.run_id,
        "primary_artifact_root_sha256": primary_record.get("artifact_root_sha256"),
        "primary_completion_evidence_sha256": primary_record.get("completion_evidence_sha256"),
        "primary_stage_attestation_record_sha256": receipt.record_sha256,
        "primary_stage_attestation_verification_sha256": receipt.verification_sha256,
    }
    if any(gate.get(field) != value for field, value in expected.items()):
        raise ValueError(
            "active primary-stage eligibility authority differs from the sealed confirmatory gate"
        )


@contextmanager
def guard_run_stage_eligibility(
    run_directory: str | Path,
) -> Iterator[RunStageEligibilityReceipt | None]:
    """Yield stage authority with upstream locks in canonical order.

    A positively attested confirmatory run is probed under its own lock first,
    then revalidated while holding ``primary -> confirmatory`` mutation locks.
    An M9 package is then revalidated under
    ``primary -> confirmatory -> original-audit -> M9``.
    This makes later upstream withdrawal dynamically revoke downstream eligibility
    without ever acquiring an upstream lock from inside a downstream lock.
    """

    run_path = Path(run_directory).resolve()
    status = _read_sealed_json_object(run_path / STATUS_FILENAME, "run status")
    experiment_name = status.get("experiment_name")
    if experiment_name == _EXTERNAL_VALIDATION_READY_EXPERIMENT:
        # Preserve the exact default-deny error for an unattested/invalid M9 run,
        # then release its lock before acquiring primary/confirmatory upstream locks.
        with _guard_single_run_stage_eligibility(run_path) as probe:
            if probe is None:  # pragma: no cover - scientific M9 invariant
                raise ValueError("external-validation run lacks positive stage authority")
        confirmatory_path = _external_confirmatory_run_path(run_path)
        if confirmatory_path is None:  # pragma: no cover - status was just checked
            raise ValueError("external-validation run lost its confirmatory identity")
        with guard_run_stage_eligibility(confirmatory_path) as confirmatory_receipt:
            if confirmatory_receipt is None:
                raise ValueError("external-validation upstream confirmatory lacks authority")
            audit_path = _external_original_audit_run_path(run_path)
            with (
                _registry_lock(_run_mutation_lock_target(audit_path)),
                _guard_single_run_stage_eligibility(run_path) as receipt,
            ):
                _require_external_gate_matches_confirmatory_receipt(
                    run_path,
                    confirmatory_receipt,
                )
                disposition_path = run_path.parent / RUN_DISPOSITION_REGISTRY_FILENAME
                with _registry_lock(disposition_path):
                    dispositions = _read_run_dispositions_unlocked(disposition_path)
                    _require_run_disposition_anchor_matches_unlocked(
                        disposition_path,
                        dispositions,
                    )
                    _require_external_original_audit_authority_under_mutation_lock(
                        run_path,
                        audit_path,
                        dispositions=dispositions,
                    )
                yield receipt
        return

    if experiment_name != "pannuke_confirmatory_study":
        with _guard_single_run_stage_eligibility(run_path) as receipt:
            yield receipt
        return

    # Preserve the exact default-deny error for an unattested/invalid confirmatory
    # run, but release its lock before acquiring the upstream primary lock.
    with _guard_single_run_stage_eligibility(run_path) as probe:
        if probe is None:  # pragma: no cover - scientific confirmatory invariant
            raise ValueError("confirmatory run lacks positive stage authority")

    primary_path = _confirmatory_primary_run_path(run_path)
    if primary_path is None:  # pragma: no cover - status was just checked
        raise ValueError("confirmatory run lost its upstream-primary identity")
    with _guard_single_run_stage_eligibility(primary_path) as primary_receipt:
        if primary_receipt is None:
            raise ValueError("confirmatory upstream primary lacks stage authority")
        with _guard_single_run_stage_eligibility(run_path) as receipt:
            _require_confirmatory_gate_matches_primary_receipt(run_path, primary_receipt)
            yield receipt


def require_run_stage_eligibility_receipt(
    run_directory: str | Path,
    *,
    integrity: IntegrityVerification | None = None,
) -> RunStageEligibilityReceipt | None:
    """Return detached evidence issued under a fresh eligibility lock.

    This receipt remains genuine evidence after return but is not active
    execution authority; use :func:`guard_run_stage_eligibility` for that.
    """

    _ = integrity  # Retained for API compatibility; authorization always re-verifies.
    with guard_run_stage_eligibility(run_directory) as receipt:
        return receipt


def require_run_stage_eligible(
    run_directory: str | Path,
    *,
    integrity: IntegrityVerification | None = None,
) -> dict[str, Any] | None:
    """Require eligibility and return any exact positive post-seal attestation."""

    receipt = require_run_stage_eligibility_receipt(run_directory, integrity=integrity)
    return None if receipt is None else receipt.attestation_record()


def _require_lifecycle_run_qualified_under_mutation_lock(
    run_directory: str | Path,
) -> dict[str, Any]:
    run_path = Path(run_directory).resolve()
    disposition_path = run_path.parent / RUN_DISPOSITION_REGISTRY_FILENAME
    with _registry_lock(disposition_path):
        binding = _completed_run_disposition_binding(run_path)
        dispositions = _read_run_dispositions_unlocked(disposition_path)
        _require_run_disposition_anchor_matches_unlocked(disposition_path, dispositions)
    matching_dispositions = [
        record for record in dispositions if record["run_id"] == binding["run_id"]
    ]
    if matching_dispositions:
        disposition = matching_dispositions[0]
        for field_name in (
            "run_path",
            "terminal_status",
            "artifact_root_sha256",
            "artifact_manifest_sha256",
        ):
            if disposition[field_name] != binding[field_name]:
                raise ValueError(
                    f"run disposition {field_name} differs from the sealed lifecycle run"
                )
        raise ValueError(
            "lifecycle qualification was permanently withdrawn for sealed run "
            f"{binding['run_id']}: reason_code={disposition['reason_code']}; "
            f"reason={disposition['reason']}"
        )

    status = _read_sealed_json_object(run_path / STATUS_FILENAME, "lifecycle readiness status")
    if (
        status.get("status") != "completed"
        or status.get("run_id") != binding["run_id"]
        or status.get("experiment_name") != _LIFECYCLE_QUALIFICATION_EXPERIMENT
    ):
        raise ValueError("sealed run is not the exact lifecycle-readiness experiment")
    evidence_path, _, qualification_sha, readiness_record_sha = (
        _lifecycle_readiness_evidence_bindings(run_path)
    )
    attestation_path = run_path.parent / RUN_STAGE_ATTESTATION_REGISTRY_FILENAME
    attestations = read_run_stage_attestations(attestation_path)
    matched = [record for record in attestations if record["run_id"] == binding["run_id"]]
    if len(matched) != 1:
        raise ValueError("lifecycle readiness lacks one durable positive qualification")
    attestation = matched[0]
    expected = {
        **binding,
        "event_type": _LIFECYCLE_QUALIFICATION_EVENT,
        "scientific_stage_eligible": False,
        "completion_stage": None,
        "completion_evidence_sha256": sha256_file(evidence_path),
    }
    for field_name, value in expected.items():
        if attestation.get(field_name) != value:
            raise ValueError(f"lifecycle qualification {field_name} differs from the sealed run")
    verification = _validate_lifecycle_qualification_verification_payload(
        attestation.get("verification"), record=attestation
    )
    if (
        verification.get("qualification_binding_sha256") != qualification_sha
        or verification.get("readiness_record_sha256") != readiness_record_sha
    ):
        raise ValueError("lifecycle qualification differs from the sealed readiness record")
    return dict(attestation)


def require_lifecycle_run_qualified(
    run_directory: str | Path,
    *,
    integrity: IntegrityVerification | None = None,
) -> dict[str, Any]:
    """Require and return one exact anchored non-scientific qualification record."""

    _ = integrity  # Authorization always performs a fresh integrity verification.
    run_path = Path(run_directory).resolve()
    with _registry_lock(_run_mutation_lock_target(run_path)):
        return _require_lifecycle_run_qualified_under_mutation_lock(run_path)


@dataclass(slots=True)
class RunTracker:
    """Own one unique run directory from creation through terminal status."""

    run_id: str
    experiment_name: str
    run_directory: Path
    registry_path: Path
    project_root: Path
    config: dict[str, Any]
    config_hash: str
    git_state: dict[str, Any]
    source_tree: dict[str, Any]
    checksums: dict[str, Any]
    timer: RuntimeTimer
    _finalized: bool = field(default=False, init=False, repr=False)
    _mutation_guard: Any = field(default_factory=threading.RLock, init=False, repr=False)
    _mutation_depth: int = field(default=0, init=False, repr=False)

    @classmethod
    def start(
        cls,
        *,
        experiment_name: str,
        config: Mapping[str, Any],
        project_root: str | Path | None = None,
        runs_root: str | Path | None = None,
        registry_path: str | Path | None = None,
        run_id: str | None = None,
        environment: Mapping[str, Any] | None = None,
        dataset_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        duplicate_audit_status: str = "not_run",
    ) -> RunTracker:
        """Create a run and immediately persist configuration and provenance."""

        root = Path(project_root or Path.cwd()).resolve()
        run_root = (
            Path(runs_root).resolve() if runs_root is not None else root / "artifacts" / "runs"
        )
        _ensure_run_disposition_anchor(run_root)
        _ensure_run_stage_attestation_anchor(run_root)
        registry = Path(registry_path) if registry_path is not None else run_root / "registry.csv"
        resolved = resolve_config(config)
        run_directory = create_run_directory(
            run_root, experiment_name=experiment_name, run_id=run_id
        )
        identifier = run_directory.name
        git_state = capture_git_state(root)
        source_tree = capture_source_tree(root)
        checksums: dict[str, Any] = {
            "dataset": {
                "path": str(Path(dataset_path).resolve()) if dataset_path else None,
                "sha256": sha256_path(dataset_path) if dataset_path else None,
            },
            "manifest": {
                "path": str(Path(manifest_path).resolve()) if manifest_path else None,
                "sha256": sha256_path(manifest_path) if manifest_path else None,
            },
            "duplicate_audit_status": duplicate_audit_status,
        }
        timer = RuntimeTimer()
        tracker = cls(
            run_id=identifier,
            experiment_name=experiment_name,
            run_directory=run_directory,
            registry_path=registry,
            project_root=root,
            config=resolved,
            config_hash=config_sha256(resolved),
            git_state=git_state,
            source_tree=source_tree,
            checksums=checksums,
            timer=timer,
        )
        tracker.write_yaml("resolved_config.yaml", resolved)
        tracker.write_json("environment.json", dict(environment or capture_environment(root)))
        tracker.write_json("git_state.json", git_state)
        source_tree_path = tracker.write_json(SOURCE_TREE_MANIFEST_FILENAME, source_tree)
        tracker.checksums["source_tree"] = {
            "manifest_path": str(source_tree_path.resolve()),
            "manifest_sha256": sha256_file(source_tree_path),
            "root_sha256": source_tree["root_sha256"],
        }
        tracker.write_json("checksums.json", checksums)
        tracker.write_provenance()
        with tracker._mutation_lock():
            write_run_status(
                run_directory,
                "running",
                run_id=identifier,
                experiment_name=experiment_name,
                started_at_utc=timer.started_at_utc,
            )
        tracker.log_event("run_started", status="running")
        return tracker

    @property
    def path(self) -> Path:
        """Compatibility alias for the run directory."""

        return self.run_directory

    @property
    def finalized(self) -> bool:
        """Return whether this tracker has already written a terminal marker."""

        return self._finalized or is_run_immutable(self.run_directory)

    @contextmanager
    def _mutation_lock(self) -> Iterator[None]:
        """Serialize each write with sealing across threads and cooperating processes."""

        with self._mutation_guard:
            if self._mutation_depth > 0:
                self._mutation_depth += 1
                try:
                    yield
                finally:
                    self._mutation_depth -= 1
                return
            with _registry_lock(_run_mutation_lock_target(self.run_directory.resolve())):
                self._mutation_depth = 1
                try:
                    yield
                finally:
                    self._mutation_depth = 0

    def _destination(self, relative_path: str | Path) -> Path:
        assert_run_mutable(self.run_directory)
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("run artifact path must remain inside the run directory")
        return self.run_directory / relative

    def write_json(self, relative_path: str | Path, value: Any) -> Path:
        with self._mutation_lock():
            return atomic_write_json(self._destination(relative_path), value)

    def write_text(self, relative_path: str | Path, value: str) -> Path:
        with self._mutation_lock():
            return atomic_write_text(self._destination(relative_path), value)

    def write_yaml(self, relative_path: str | Path, value: Mapping[str, Any]) -> Path:
        with self._mutation_lock():
            return atomic_write_yaml(self._destination(relative_path), value)

    def write_metrics(self, metrics: Mapping[str, Any]) -> Path:
        """Persist strict machine-readable metrics under the standard name."""

        return self.write_json("metrics.json", dict(metrics))

    def log_event(self, event: str, **details: Any) -> None:
        """Append one strict structured event and one concise human log line."""

        with self._mutation_lock():
            if not event.strip():
                raise ValueError("run event name must not be empty")
            timestamp = utc_now()
            payload = {
                "timestamp_utc": timestamp,
                "run_id": self.run_id,
                "event": event,
                **details,
            }
            encoded = json.dumps(
                payload,
                allow_nan=False,
                default=_json_default,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            events_path = self._destination(EVENTS_FILENAME)
            run_log_path = self._destination(RUN_LOG_FILENAME)
            with events_path.open("a", encoding="utf-8", newline="\n") as events:
                events.write(f"{encoded}\n")
                events.flush()
                os.fsync(events.fileno())
            human_details = json.dumps(
                details,
                allow_nan=False,
                default=_json_default,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            with run_log_path.open("a", encoding="utf-8", newline="\n") as run_log:
                run_log.write(f"{timestamp} {event} {human_details}\n")
                run_log.flush()
                os.fsync(run_log.fileno())

    def write_provenance(self, **details: Any) -> Path:
        """Rewrite the mutable run's provenance snapshot with additional evidence."""

        payload = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "config_sha256": self.config_hash,
            "split_seed": _config_seed(self.config, "split"),
            "model_seed": _config_seed(self.config, "model"),
            "corruption_seed": _config_seed(self.config, "corruption"),
            "started_at_utc": self.timer.started_at_utc,
            "source_tree": {
                "manifest": SOURCE_TREE_MANIFEST_FILENAME,
                "manifest_sha256": self.checksums["source_tree"]["manifest_sha256"],
                "root_sha256": self.source_tree["root_sha256"],
            },
            **details,
        }
        return self.write_json("run_provenance.json", payload)

    def _registry_row(self, *, status: RunOutcome, completed_at: str) -> dict[str, Any]:
        dataset = self.checksums.get("dataset")
        manifest = self.checksums.get("manifest")
        dataset_hash = dataset.get("sha256") if isinstance(dataset, Mapping) else ""
        manifest_hash = manifest.get("sha256") if isinstance(manifest, Mapping) else ""
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "status": status,
            "started_at": self.timer.started_at_utc,
            "completed_at": completed_at,
            "config_sha256": self.config_hash,
            "git_state": _git_registry_value(self.git_state),
            "dataset_sha256": dataset_hash or "",
            "manifest_sha256": manifest_hash or "",
            "split_seed": _config_seed(self.config, "split"),
            "model_seed": _config_seed(self.config, "model"),
            "corruption_seed": _config_seed(self.config, "corruption"),
            "run_path": str(self.run_directory.resolve()),
        }

    def finalize(
        self,
        status: RunOutcome = "completed",
        *,
        error: BaseException | None = None,
    ) -> None:
        """Persist terminal runtime/status, append the registry, and seal the run."""

        with self._mutation_lock():
            self._finalize_locked(status, error=error)

    def _finalize_locked(
        self,
        status: RunOutcome,
        *,
        error: BaseException | None,
    ) -> None:
        """Finalize while holding the run mutation lock through marker publication."""

        if self._finalized or is_run_immutable(self.run_directory):
            raise RuntimeError(f"run has already been finalized: {self.run_id}")
        if status not in {"completed", "failed"}:
            raise ValueError("terminal run status must be 'completed' or 'failed'")
        if status == "completed" and error is not None:
            raise ValueError("a completed run cannot carry a failure exception")
        completed_at = utc_now()
        runtime = {
            "started_at_utc": self.timer.started_at_utc,
            "completed_at_utc": completed_at,
            "elapsed_seconds": self.timer.elapsed_seconds,
        }
        self.write_json("runtime.json", runtime)
        traceback_path: str | None = None
        if error is not None:
            traceback_path = "traceback.txt"
            self.write_text(traceback_path, format_traceback(error))
        write_run_status(
            self.run_directory,
            status,
            run_id=self.run_id,
            experiment_name=self.experiment_name,
            started_at_utc=self.timer.started_at_utc,
            completed_at_utc=completed_at,
            elapsed_seconds=runtime["elapsed_seconds"],
            traceback=traceback_path,
        )
        if error is not None:
            self.log_event(
                "run_failed",
                status=status,
                error_type=type(error).__name__,
                error_message=str(error),
                traceback=traceback_path,
            )
        self.log_event(
            "run_finalization_started",
            status=status,
            completed_at_utc=completed_at,
            elapsed_seconds=runtime["elapsed_seconds"],
        )
        payload_records = _artifact_records(
            self.run_directory,
            extra_exclusions={"checksums.json"},
        )
        payload_root = _artifact_root_sha256(payload_records)
        self.checksums["integrity"] = {
            "payload_root_sha256": payload_root,
            "payload_artifact_count": len(payload_records),
            "payload_scope_excludes": [
                ARTIFACT_MANIFEST_FILENAME,
                IMMUTABLE_MARKER,
                "checksums.json",
            ],
            "final_artifact_root_location": ARTIFACT_MANIFEST_FILENAME,
        }
        self.write_json("checksums.json", self.checksums)
        artifact_records = _artifact_records(self.run_directory)
        artifact_root = _artifact_root_sha256(artifact_records)
        artifact_manifest = {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": status,
            "created_at_utc": utc_now(),
            "artifact_count": len(artifact_records),
            "artifact_root_sha256": artifact_root,
            "root_digest_algorithm": "sha256(canonical-json-artifact-records)",
            "excluded_paths": sorted(_INTEGRITY_EXCLUSIONS),
            "artifacts": artifact_records,
        }
        artifact_manifest_path = atomic_write_json(
            self.run_directory / ARTIFACT_MANIFEST_FILENAME,
            artifact_manifest,
        )
        artifact_manifest_sha256 = sha256_file(artifact_manifest_path)
        integrity_record = {
            "run_id": self.run_id,
            "status": status,
            "sealed_at_utc": utc_now(),
            "run_path": str(self.run_directory.resolve()),
            "artifact_count": len(artifact_records),
            "artifact_root_sha256": artifact_root,
            "artifact_manifest_sha256": artifact_manifest_sha256,
        }
        append_integrity_record(
            self.registry_path.parent / INTEGRITY_REGISTRY_FILENAME,
            integrity_record,
        )
        append_registry_row(
            self.registry_path,
            self._registry_row(status=status, completed_at=completed_at),
        )
        atomic_write_json(
            self.run_directory / IMMUTABLE_MARKER,
            {
                **integrity_record,
                "integrity_registry": str(
                    (self.registry_path.parent / INTEGRITY_REGISTRY_FILENAME).resolve()
                ),
            },
        )
        self._finalized = True

    def complete(self) -> None:
        self.finalize("completed")

    def fail(self, error: BaseException) -> None:
        self.finalize("failed", error=error)

    def __enter__(self) -> RunTracker:
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: Any,
    ) -> Literal[False]:
        del error_type, traceback
        if self.finalized:
            return False
        if error is None:
            self.complete()
        else:
            self.fail(error)
        return False


def start_run(**kwargs: Any) -> RunTracker:
    """Convenience wrapper for :meth:`RunTracker.start`."""

    return RunTracker.start(**kwargs)


__all__ = [
    "ARTIFACT_MANIFEST_FILENAME",
    "EVENTS_FILENAME",
    "IMMUTABLE_MARKER",
    "INTEGRITY_REGISTRY_FILENAME",
    "REGISTRY_COLUMNS",
    "RUN_DISPOSITION_ANCHOR_FILENAME",
    "RUN_DISPOSITION_REGISTRY_FILENAME",
    "RUN_LOG_FILENAME",
    "RUN_STAGE_ATTESTATION_ANCHOR_FILENAME",
    "RUN_STAGE_ATTESTATION_REGISTRY_FILENAME",
    "SOURCE_GOVERNANCE_FILENAMES",
    "SOURCE_TREE_MANIFEST_FILENAME",
    "ExternalValidationReadyAttestationVerification",
    "IntegrityVerification",
    "LifecycleQualificationAttestationVerification",
    "PrimaryStageAttestationVerification",
    "RunStageEligibilityReceipt",
    "RunTracker",
    "RuntimeTimer",
    "append_integrity_record",
    "append_registry_row",
    "append_run_registry",
    "assert_run_mutable",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_npz",
    "atomic_write_text",
    "atomic_write_yaml",
    "attest_external_validation_ready",
    "attest_lifecycle_run_qualification",
    "attest_primary_run_stage_eligibility",
    "attest_run_stage_eligibility",
    "build_external_validation_ready_attestation_verification",
    "build_lifecycle_qualification_attestation_verification",
    "capture_environment",
    "capture_git_state",
    "capture_governance_tree",
    "capture_source_tree",
    "checksum_file",
    "checksum_path",
    "create_run_directory",
    "format_traceback",
    "generate_run_id",
    "guard_run_stage_eligibility",
    "is_run_immutable",
    "read_run_dispositions",
    "read_run_stage_attestations",
    "require_lifecycle_run_qualified",
    "require_run_stage_eligibility_receipt",
    "require_run_stage_eligible",
    "sealed_run_ancestor",
    "sha256_file",
    "sha256_path",
    "start_run",
    "utc_now",
    "verify_run_integrity",
    "windows_compatible_relative_path_sort_key",
    "withdraw_run_eligibility",
    "write_run_status",
]
