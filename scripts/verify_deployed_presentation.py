#!/usr/bin/env python3
"""Read-only HTTP verification of the complete published presentation package."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_VERIFIER = runpy.run_path(str(PROJECT_ROOT / "scripts/present_demo.py"))
DEFAULT_OUTPUT = _LOCAL_VERIFIER["DEFAULT_OUTPUT"]
OUTPUT_FILES = frozenset(_LOCAL_VERIFIER["OUTPUT_FILES"])
MAX_FILE_BYTES = 25 * 1024 * 1024


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(data).hexdigest()


def _fetch(url: str, *, maximum_bytes: int, timeout: float) -> bytes:
    request = Request(
        url, headers={"Accept-Encoding": "identity", "User-Agent": "AANCA-publication-verifier/1.0"}
    )
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError(f"HTTP {response.status} for {url}")
        data = response.read(maximum_bytes + 1)
    if len(data) > maximum_bytes:
        raise ValueError(f"HTTP response exceeds the declared size limit: {url}")
    return data


def verify_remote_package(
    url: str,
    *,
    expected_manifest: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Check remote integrity and, when supplied, equality to a local release."""

    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("URL must identify an HTTP(S) package root without query or fragment")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL must not contain credentials")
    if not 0.0 < timeout <= 300.0:
        raise ValueError("timeout must lie in (0, 300] seconds")
    base = url.rstrip("/") + "/"
    manifest = json.loads(
        _fetch(urljoin(base, "manifest.json"), maximum_bytes=262144, timeout=timeout)
    )
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ValueError("remote manifest must contain a file list")
    records = manifest["files"]
    paths: list[str] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
            raise ValueError("remote manifest file record is malformed")
        path, digest, size = record["path"], record["sha256"], record["size_bytes"]
        if (
            not isinstance(path, str)
            or path not in OUTPUT_FILES
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 <= size <= MAX_FILE_BYTES
        ):
            raise ValueError("remote manifest contains an invalid or unexpected file")
        paths.append(path)
    if len(paths) != len(OUTPUT_FILES) or set(paths) != OUTPUT_FILES:
        raise ValueError("remote manifest does not match the closed presentation allowlist")
    root = _canonical_sha256(records)
    if manifest.get("manifest_root_sha256") != root:
        raise ValueError("remote manifest root differs from its file records")

    checks: list[dict[str, Any]] = []
    for record in records:
        path = record["path"]
        try:
            data = _fetch(urljoin(base, path), maximum_bytes=record["size_bytes"], timeout=timeout)
            digest = hashlib.sha256(data).hexdigest()
            valid = len(data) == record["size_bytes"] and digest == record["sha256"]
            checks.append({"path": path, "valid": valid, "size_bytes": len(data), "sha256": digest})
        except Exception as error:
            checks.append(
                {"path": path, "valid": False, "error": f"{type(error).__name__}: {error}"}
            )
    integrity_valid = all(check["valid"] for check in checks)
    matches_local = manifest == expected_manifest if expected_manifest is not None else None
    return {
        "status": "valid" if integrity_valid and matches_local is not False else "failed",
        "url": base,
        "remote_package_integrity_valid": integrity_valid,
        "matches_local_release": matches_local,
        "remote_manifest_root_sha256": root,
        "local_manifest_root_sha256": (
            expected_manifest.get("manifest_root_sha256") if expected_manifest is not None else None
        ),
        "file_count": len(checks) + 1,
        "files": checks,
        "verification_scope": "served HTTP bytes and manifest identities; no model execution",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://aancastudy.org/")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        _LOCAL_VERIFIER["verify_presentation"](args.output_dir)
        manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
        result = verify_remote_package(args.url, expected_manifest=manifest, timeout=args.timeout)
    except Exception as error:
        print(
            f"ERROR: deployed presentation verification failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
