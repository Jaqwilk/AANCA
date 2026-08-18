"""Fail-closed dual-layout import guard for the external carrier tests."""

from __future__ import annotations

import importlib
import os
import stat
import sys
from pathlib import Path
from types import ModuleType

TESTS_ROOT = Path(__file__).resolve().parent


def _is_reparse_point(path_stat: os.stat_result) -> bool:
    attributes = getattr(path_stat, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & reparse_flag)


def require_ordinary_carrier_file(path: Path, *, carrier_root: Path) -> Path:
    root = carrier_root.resolve(strict=True)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        path_stat = path.lstat()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"carrier import target is missing or outside selected root: {str(path)!r}"
        ) from exc
    if path.is_symlink() or _is_reparse_point(path_stat) or not stat.S_ISREG(path_stat.st_mode):
        raise RuntimeError(f"carrier import target is not an ordinary file: {str(path)!r}")
    return resolved


def resolve_carrier_root(tests_root: Path) -> Path:
    if tests_root.name != "tests" or not tests_root.is_dir():
        raise RuntimeError("carrier import guard must be installed in an ordinary tests directory")
    carrier_root = tests_root.parent.resolve(strict=True)
    require_ordinary_carrier_file(
        carrier_root / "capsule_bootstrap.py",
        carrier_root=carrier_root,
    )
    if (carrier_root / "src" / "capsule_bootstrap.py").exists():
        raise RuntimeError("carrier bootstrap layout is ambiguous between root and src")
    return carrier_root


CARRIER_ROOT = resolve_carrier_root(TESTS_ROOT)


def resolve_package_import_root(carrier_root: Path) -> Path:
    candidates = [
        candidate
        for candidate in (carrier_root, carrier_root / "src")
        if (
            candidate / "histo_audit" / "workflows" / "original_confirmatory_capsule_authority.py"
        ).is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "carrier tests require exactly one external-root or repository-src package layout"
        )
    return candidates[0]


PACKAGE_IMPORT_ROOT = resolve_package_import_root(CARRIER_ROOT)

for _path in (CARRIER_ROOT, PACKAGE_IMPORT_ROOT):
    _text = str(_path)
    if _text in sys.path:
        sys.path.remove(_text)
    sys.path.insert(0, _text)


def import_exact(module_name: str, expected_path: Path) -> ModuleType:
    expected = require_ordinary_carrier_file(
        expected_path,
        carrier_root=CARRIER_ROOT,
    )
    module = importlib.import_module(module_name)
    file_origin = getattr(module, "__file__", None)
    spec = getattr(module, "__spec__", None)
    spec_origin = getattr(spec, "origin", None)
    if (
        type(file_origin) is not str
        or Path(file_origin).resolve() != expected
        or spec is None
        or type(spec_origin) is not str
        or spec_origin in ("built-in", "frozen")
        or Path(spec_origin).resolve() != expected
    ):
        raise RuntimeError(
            f"carrier test imported {module_name} with __file__={file_origin!r} "
            f"and __spec__.origin={spec_origin!r}; expected {str(expected)!r}"
        )
    return module
