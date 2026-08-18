"""Configuration loading, validation, and deterministic hashing.

Configuration hashes are based on a canonical JSON representation rather than
the spelling or key order of the input YAML file.  This means semantically
identical mappings receive the same hash while list order and scalar types stay
significant.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]


def _plain_value(value: Any, *, location: str = "config") -> Any:
    """Return a JSON/YAML-safe value with string mapping keys.

    PyYAML accepts several Python-specific values.  Refusing those values here
    keeps the resolved configuration portable and its hash unambiguous.
    """

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} contains a non-string mapping key: {key!r}")
            result[key] = _plain_value(nested, location=f"{location}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _plain_value(item, location=f"{location}[{index}]") for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (value != value or value in (float("inf"), -float("inf"))):
            raise ValueError(f"{location} contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    raise ValueError(f"{location} contains an unsupported value of type {type(value).__name__}")


def resolve_config(config: Mapping[str, Any]) -> Config:
    """Copy *config* into a portable, mutation-independent plain mapping."""

    resolved = _plain_value(config)
    if not isinstance(resolved, dict):  # defensive; Mapping above guarantees this
        raise TypeError("configuration must resolve to a mapping")
    return resolved


def load_config(path: str | Path) -> Config:
    """Load one YAML configuration and require a non-empty mapping root."""

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {config_path}")
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML configuration at {config_path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    if not loaded:
        raise ValueError(f"configuration must not be empty: {config_path}")
    return resolve_config(loaded)


def canonical_config_bytes(config: Mapping[str, Any]) -> bytes:
    """Serialize *config* deterministically for hashing and provenance."""

    resolved = resolve_config(config)
    return json.dumps(
        resolved,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def config_sha256(config: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of a resolved configuration mapping."""

    return hashlib.sha256(canonical_config_bytes(config)).hexdigest()


def hash_config(config_or_path: Mapping[str, Any] | str | Path) -> str:
    """Hash a mapping or a YAML configuration path.

    This convenience wrapper intentionally hashes parsed configuration meaning,
    not raw YAML bytes.
    """

    config = (
        load_config(config_or_path)
        if isinstance(config_or_path, (str, Path))
        else resolve_config(config_or_path)
    )
    return config_sha256(config)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def save_resolved_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically save a stable, human-readable resolved YAML configuration."""

    destination = Path(path)
    content = yaml.safe_dump(
        resolve_config(config),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    )
    _atomic_write_text(destination, content)
    return destination


# Descriptive aliases retained for callers that prefer explicit names.
load_yaml_config = load_config
configuration_sha256 = config_sha256


__all__ = [
    "Config",
    "canonical_config_bytes",
    "config_sha256",
    "configuration_sha256",
    "hash_config",
    "load_config",
    "load_yaml_config",
    "resolve_config",
    "save_resolved_config",
]
