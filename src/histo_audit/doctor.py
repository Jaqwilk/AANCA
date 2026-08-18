"""Read-only environment diagnostics for reproducible experiment evidence."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from histo_audit.utils.run_tracking import atomic_write_json, capture_git_state, utc_now

PACKAGE_NAMES: tuple[str, ...] = (
    "histo-audit",
    "cleanlab",
    "jinja2",
    "matplotlib",
    "numpy",
    "pandas",
    "pillow",
    "psutil",
    "pyarrow",
    "pyyaml",
    "scikit-image",
    "scikit-learn",
    "scipy",
    "torch",
    "torchvision",
    "typer",
)


def _subprocess_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
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
                memory_mib: float | str = float(fields[2])
            except ValueError:
                memory_mib = fields[2]
            devices.append(
                {
                    "index": index,
                    "name": fields[1],
                    "vram_total_mib": memory_mib,
                    "driver_version": fields[3],
                }
            )
    return {
        "available": result.returncode == 0,
        "devices": devices,
        "return_code": result.returncode,
        "error": result.stderr.strip() or None,
    }


def _torch_evidence() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    pytorch: dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda_build": None,
    }
    cuda: dict[str, Any] = {
        "available": False,
        "device_count": 0,
        "cudnn_version": None,
        "functional_test": {
            "attempted": False,
            "success": False,
            "finite_gradient": None,
            "error": None,
        },
    }
    devices: list[dict[str, Any]] = []
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # optional dependency/import failures are evidence
        pytorch["import_error"] = f"{type(exc).__name__}: {exc}"
        return pytorch, cuda, devices

    pytorch["installed"] = True
    pytorch["version"] = str(getattr(torch, "__version__", "unknown"))
    version_module = getattr(torch, "version", None)
    pytorch["cuda_build"] = getattr(version_module, "cuda", None)
    try:
        cuda_module = torch.cuda
        cuda_available = bool(cuda_module.is_available())
        device_count = int(cuda_module.device_count()) if cuda_available else 0
        cuda["available"] = cuda_available
        cuda["device_count"] = device_count
        cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
        cuda["cudnn_version"] = cudnn.version() if cudnn is not None else None
        for index in range(device_count):
            properties = cuda_module.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": str(properties.name),
                    "vram_total_bytes": int(properties.total_memory),
                    "vram_total_mib": round(int(properties.total_memory) / (1024**2), 2),
                    "compute_capability": list(cuda_module.get_device_capability(index)),
                }
            )
        if cuda_available:
            functional_test = cuda["functional_test"]
            functional_test["attempted"] = True
            try:
                tensor = torch.randn((16, 16), device="cuda", requires_grad=True)
                loss = (tensor @ tensor.transpose(0, 1)).square().mean()
                loss.backward()
                finite_gradient = bool(torch.isfinite(tensor.grad).all().item())
                functional_test["finite_gradient"] = finite_gradient
                functional_test["success"] = finite_gradient
                # Release the diagnostic allocation promptly on memory-limited systems.
                del loss, tensor
            except Exception as exc:
                functional_test["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        cuda["probe_error"] = f"{type(exc).__name__}: {exc}"
    pytorch["cuda_available"] = cuda["available"]
    return pytorch, cuda, devices


def _ram_evidence() -> dict[str, Any]:
    try:
        psutil = importlib.import_module("psutil")
        memory = psutil.virtual_memory()
        return {
            "total_bytes": int(memory.total),
            "total_gib": round(int(memory.total) / (1024**3), 2),
            "available_bytes": int(memory.available),
            "available_gib": round(int(memory.available) / (1024**3), 2),
            "source": "psutil",
        }
    except Exception as exc:
        return {
            "total_bytes": None,
            "total_gib": None,
            "available_bytes": None,
            "available_gib": None,
            "source": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _disk_evidence(project_root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(project_root)
    return {
        "path": str(project_root),
        "total_bytes": int(usage.total),
        "total_gib": round(usage.total / (1024**3), 2),
        "free_bytes": int(usage.free),
        "free_gib": round(usage.free / (1024**3), 2),
        "used_bytes": int(usage.used),
        "used_gib": round(usage.used / (1024**3), 2),
    }


def _bounded_file_count(path: Path, *, limit: int = 10_000) -> tuple[int, bool]:
    count = 0
    for candidate in path.rglob("*"):
        if candidate.is_file() and candidate.name.lower() not in {
            ".gitkeep",
            ".keep",
            ".ds_store",
            "readme",
            "readme.md",
            "readme.txt",
        }:
            count += 1
            if count >= limit:
                return count, True
    return count, False


def inspect_dataset_status(project_root: str | Path) -> dict[str, Any]:
    """Inspect only configured and conventional PanNuke paths, without downloading."""

    root = Path(project_root).resolve()
    candidates: list[Path] = []
    configured = os.environ.get("PANNUKE_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(root / "data" / "raw" / "pannuke")
    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(str(candidate.resolve(strict=False)))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
    checks: list[dict[str, Any]] = []
    detected = False
    for candidate in unique_candidates:
        exists = candidate.is_dir()
        count, truncated = _bounded_file_count(candidate) if exists else (0, False)
        checks.append(
            {
                "path": str(candidate.resolve(strict=False)),
                "exists": exists,
                "file_count": count,
                "file_count_truncated": truncated,
            }
        )
        detected = detected or (exists and count > 0)
    return {
        "pannuke_detected": detected,
        "verified": False,
        "status": "present_unverified" if detected else "not_found",
        "paths_checked": checks,
        "setup_instructions": str((root / "DATASET_SETUP.md").resolve()),
    }


def inspect_write_access(project_root: str | Path) -> dict[str, Any]:
    """Attempt and remove a same-filesystem write probe."""

    root = Path(project_root).resolve()
    probe_path: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".histo-audit-write-probe-", dir=root)
        probe_path = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(b"write-access-probe\n")
            handle.flush()
            os.fsync(handle.fileno())
        probe_path.unlink()
        return {"path": str(root), "writable": True, "probe_removed": True, "error": None}
    except Exception as exc:
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)
        return {
            "path": str(root),
            "writable": False,
            "probe_removed": probe_path is None or not probe_path.exists(),
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_doctor_report(project_root: str | Path | None = None) -> dict[str, Any]:
    """Collect required OS, Python, package, accelerator, storage, and data evidence."""

    root = Path(project_root or Path.cwd()).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"project root does not exist: {root}")
    pytorch, cuda, torch_devices = _torch_evidence()
    smi = _nvidia_smi()
    devices = torch_devices or list(smi.get("devices", []))
    vram_values = [
        float(device["vram_total_mib"])
        for device in devices
        if isinstance(device, Mapping) and isinstance(device.get("vram_total_mib"), (int, float))
    ]
    return {
        "schema_version": 1,
        "captured_at_utc": utc_now(),
        "project_root": str(root),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "version_full": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": _package_versions(),
        "pytorch": pytorch,
        "cuda": cuda,
        "gpu": {"devices": devices, "nvidia_smi": smi},
        "vram": {
            "total_mib_by_device": vram_values,
            "maximum_total_mib": max(vram_values) if vram_values else None,
            "source": "pytorch" if torch_devices else "nvidia-smi" if devices else None,
        },
        "ram": _ram_evidence(),
        "disk": _disk_evidence(root),
        "dataset": inspect_dataset_status(root),
        "write_access": inspect_write_access(root),
        "git": capture_git_state(root),
    }


def save_doctor_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Atomically persist a doctor report as strict JSON."""

    return atomic_write_json(output_path, dict(report))


def run_doctor(
    *,
    project_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Collect and save diagnostics, returning the exact printed payload and path."""

    root = Path(project_root or Path.cwd()).resolve()
    report = collect_doctor_report(root)
    destination = Path(output_path) if output_path is not None else root / "reports" / "doctor.json"
    if not destination.is_absolute():
        destination = root / destination
    save_doctor_report(report, destination)
    return report, destination.resolve()


def format_doctor_report(report: Mapping[str, Any]) -> str:
    """Return a deterministic printable representation of saved evidence."""

    return json.dumps(report, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)


# Convenient name for direct Python callers.
doctor = run_doctor


__all__ = [
    "PACKAGE_NAMES",
    "collect_doctor_report",
    "doctor",
    "format_doctor_report",
    "inspect_dataset_status",
    "inspect_write_access",
    "run_doctor",
    "save_doctor_report",
]
