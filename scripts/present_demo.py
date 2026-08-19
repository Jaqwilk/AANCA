#!/usr/bin/env python3
"""Verify and open the checked-in AANCA presentation without project dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "mvp_demo"
OUTPUT_FILES = ("README.md", "evidence.json", "index.html", "pannuke_mask_qc_overlays.png")

_PRESENTATION_HEADERS = (
    ("Cache-Control", "no-store"),
    ("Content-Security-Policy", "base-uri 'none'; frame-ancestors 'none'; object-src 'none'"),
    ("Permissions-Policy", "camera=(), geolocation=(), microphone=()"),
    ("Referrer-Policy", "no-referrer"),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
)


class _PresentationRequestHandler(SimpleHTTPRequestHandler):
    """Serve the verified package with presentation-safe response headers."""

    def version_string(self) -> str:
        """Avoid disclosing the Python runtime in the local server banner."""

        return "AANCA"

    def end_headers(self) -> None:
        """Attach deterministic hardening and no-cache headers to every response."""

        for name, value in _PRESENTATION_HEADERS:
            self.send_header(name, value)
        super().end_headers()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number rejected: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key rejected: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )
    if not isinstance(parsed, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return parsed


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_presentation(output_directory: str | Path) -> dict[str, Any]:
    """Verify the closed package using only Python's standard library."""

    directory = Path(output_directory).resolve(strict=True)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("presentation output is not a regular directory")
    expected_names = sorted((*OUTPUT_FILES, "manifest.json"))
    if sorted(path.name for path in directory.iterdir()) != expected_names:
        raise ValueError("presentation output allowlist differs")

    manifest = _load_json(directory / "manifest.json")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("policy") != "aanca_presentation_complete_h1_h7_selected_source_readback_v2"
    ):
        raise ValueError("presentation manifest schema or policy differs")
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != len(OUTPUT_FILES):
        raise ValueError("presentation manifest file list differs")
    record_paths = [record.get("path") if isinstance(record, dict) else None for record in records]
    if sorted(path for path in record_paths if isinstance(path, str)) != sorted(OUTPUT_FILES):
        raise ValueError("presentation manifest record allowlist differs")
    if manifest.get("manifest_root_sha256") != _canonical_sha256(records):
        raise ValueError("presentation manifest root differs")

    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
            raise ValueError("presentation manifest record is malformed")
        relative = record["path"]
        size = record["size_bytes"]
        digest = record["sha256"]
        if (
            not isinstance(relative, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError(f"presentation manifest record is invalid: {relative!r}")
        path = directory / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"presentation file is not regular: {relative}")
        if path.stat().st_size != size or _sha256(path) != digest:
            raise ValueError(f"presentation file differs from its seal: {relative}")

    evidence = _load_json(directory / "evidence.json")
    primary = evidence.get("primary")
    if (
        evidence.get("schema_version") != 2
        or evidence.get("presentation_status") != "DEMO_COMPLETE"
        or evidence.get("scientific_status") != "PRIMARY_STUDY_COMPLETE"
        or evidence.get("confirmatory_completed") is not False
        or evidence.get("external_validation_completed") is not False
        or not isinstance(primary, dict)
        or primary.get("h4_restoration", {}).get("directional_result")
        != "adverse_to_registered_hypothesis"
        or primary.get("h4_restoration", {}).get("registered_hypothesis_supported") is not False
        or primary.get("h2_subgroups", {}).get("reported_count", 0) <= 0
        or primary.get("instance_dependent_seed_audit", {}).get(
            "independent_corruption_realisations"
        )
        is not False
        or primary.get("instance_dependent_seed_audit", {}).get("disclosure_required") is not True
        or primary.get("inference", {}).get("p_value_sidedness") != "one_sided"
    ):
        raise ValueError("presentation evidence scope differs")

    return {
        "status": "valid",
        "presentation_status": "DEMO_COMPLETE",
        "scientific_status": "PRIMARY_STUDY_COMPLETE",
        "manifest_root_sha256": manifest["manifest_root_sha256"],
        "file_count": len(expected_names),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and locally serve the checked-in AANCA presentation using only "
            "Python's standard library."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify checksums and exit without starting a server.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.host.strip() or args.host != args.host.strip():
        print("ERROR: host must be non-empty and have no outer whitespace", file=sys.stderr)
        return 2
    if not 0 <= args.port <= 65_535:
        print("ERROR: port must be between 0 and 65535", file=sys.stderr)
        return 2
    try:
        verification = verify_presentation(args.output_dir)
    except Exception as exc:
        print(
            f"ERROR: presentation verification failed: {type(exc).__name__}: {exc}", file=sys.stderr
        )
        return 1

    if args.verify_only:
        print(json.dumps(verification, indent=2))
        return 0

    directory = Path(args.output_dir).resolve(strict=True)
    handler = partial(_PresentationRequestHandler, directory=str(directory))
    try:
        server = ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as exc:
        print(f"ERROR: local presentation server failed: {exc}", file=sys.stderr)
        return 1
    server.daemon_threads = True
    bound_port = int(server.server_address[1])
    browser_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{browser_host}:{bound_port}/"
    print(json.dumps({**verification, "status": "verified_and_serving", "url": url}, indent=2))
    if not args.no_open:
        try:
            if not webbrowser.open(url, new=2):
                print(f"Browser launch was not confirmed; open {url} manually.", file=sys.stderr)
        except Exception as exc:
            print(
                f"Browser launch failed ({type(exc).__name__}); open {url} manually.",
                file=sys.stderr,
            )
    print("Press Ctrl+C to stop the local presentation server.")
    try:
        with server:
            server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nPresentation server stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
