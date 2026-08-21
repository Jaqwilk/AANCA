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
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "mvp_demo"
HERO_NUCLEUS_FILES = (
    "assets/hero/nuclei/nucleus-compact.png",
    "assets/hero/nuclei/nucleus-elongated.png",
    "assets/hero/nuclei/nucleus-kidney.png",
    "assets/hero/nuclei/nucleus-bilobed.png",
    "assets/hero/nuclei/nucleus-irregular.png",
    "assets/hero/nuclei/nucleus-flattened.png",
)
OUTPUT_FILES = (
    "README.md",
    "evidence.json",
    "index.html",
    "pannuke_mask_qc_overlays.png",
    *HERO_NUCLEUS_FILES,
)

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

    def __init__(self, *args: Any, directory: str | None = None, **kwargs: Any) -> None:
        """Bind URL translation to the verified package root on every platform."""

        self._presentation_root = Path(directory or ".").resolve(strict=True)
        super().__init__(*args, directory=str(self._presentation_root), **kwargs)

    def translate_path(self, path: str) -> str:
        """Translate a request without relying on platform-specific slash handling."""

        requested = PurePosixPath(unquote(urlsplit(path).path))
        parts = [part for part in requested.parts if part not in {"/", ""}]
        if any(part in {".", ".."} or "\\" in part or ":" in part for part in parts):
            return str(self._presentation_root / "__not_found__")
        candidate = self._presentation_root.joinpath(*parts).resolve()
        try:
            candidate.relative_to(self._presentation_root)
        except ValueError:
            return str(self._presentation_root / "__not_found__")
        return str(candidate)

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
    package_entries = tuple(directory.rglob("*"))
    if any(path.is_symlink() for path in package_entries):
        raise ValueError("presentation output contains a symlink")
    actual_names = sorted(
        path.relative_to(directory).as_posix() for path in package_entries if path.is_file()
    )
    if actual_names != expected_names:
        raise ValueError("presentation output allowlist differs")

    manifest = _load_json(directory / "manifest.json")
    if (
        manifest.get("schema_version") != 4
        or manifest.get("policy") != "aanca_presentation_current_evidence_readback_v4"
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
    external_completed = evidence.get("external_validation_completed")
    external = evidence.get("external_validation")
    controlled = evidence.get("controlled_external_benchmark")
    puma = evidence.get("new_source_confirmation")
    stress = evidence.get("realism_stress")
    sensitivity = evidence.get("audit_time_label_sensitivity")
    natural_action = evidence.get("natural_data_action")
    next_phase = evidence.get("next_phase")
    external_scope_valid = (
        external_completed is True
        and isinstance(external, dict)
        and external.get("completion_stage") == "EXTERNAL_VALIDATION_COMPLETE"
        and external.get("overall_conclusion") == "not_supported"
        and external.get("primary_subset", {}).get("ranking_success_conditions_met") is False
        and external.get("primary_subset", {}).get("downstream_success_conditions_met") is False
        and isinstance(external.get("claim_boundary"), dict)
        and not any(external["claim_boundary"].values())
    )
    controlled_scope_valid = (
        isinstance(controlled, dict)
        and controlled.get("study_id") == "monusac_current_aanca_controlled_external_v1"
        and controlled.get("decision") == "not_supported"
        and controlled.get("all_success_conditions_met") is False
        and controlled.get("success_conditions", {}).get("primary_top_k_beats_exact_matched_random")
        is True
        and controlled.get("success_conditions", {}).get(
            "primary_downstream_beats_corrupted_uncorrected"
        )
        is False
        and controlled.get("success_conditions", {}).get(
            "primary_downstream_beats_mean_matched_random"
        )
        is False
        and controlled.get("success_conditions", {}).get("important_class_recall_non_degradation")
        is False
        and isinstance(controlled.get("claim_boundary"), dict)
        and not any(controlled["claim_boundary"].values())
    )
    puma_scope_valid = (
        isinstance(puma, dict)
        and puma.get("study_id") == "puma_new_data_confirmation_v1"
        and puma.get("decision") == "controlled_noise_transfer_supported"
        and puma.get("all_success_conditions_met") is True
        and puma.get("final_case_groups") == 62
        and puma.get("success_conditions", {}).get(
            "all_four_seed_directions_positive_against_both_controls"
        )
        is True
        and puma.get("success_conditions", {}).get(
            "every_primary_class_recall_lower_bound_gte_minus_0_01"
        )
        is True
        and puma.get("claim_boundary", {}).get("controlled_noise_transfer_if_positive") is True
        and puma.get("claim_boundary", {}).get("natural_error_detection_proven") is False
        and puma.get("claim_boundary", {}).get("pathologist_error_detection_proven") is False
        and puma.get("claim_boundary", {}).get("clinical_utility_proven") is False
        and puma.get("verification", {}).get("verified") is True
    )
    stress_scope_valid = (
        isinstance(stress, dict)
        and stress.get("scenario_count") == 9
        and stress.get("positive_aggregate_lower_bound_count") == 9
        and stress.get("all_class_safeguards_passed_count") == 1
        and stress.get("all_scenarios_passed") is False
        and stress.get("candidate_changed") is False
        and isinstance(stress.get("claim_boundary"), dict)
        and not any(stress["claim_boundary"].values())
    )
    sensitivity_scope_valid = (
        isinstance(sensitivity, dict)
        and sensitivity.get("all_sensitivity_gates_passed") is True
        and sensitivity.get("fold_assignment_label_source") == "observed_label"
        and sensitivity.get("candidate_changed") is False
        and isinstance(sensitivity.get("claim_boundary"), dict)
        and not any(sensitivity["claim_boundary"].values())
    )
    natural_action_scope_valid = (
        isinstance(natural_action, dict)
        and natural_action.get("status") == "unavailable"
        and natural_action.get("action") == "retain_uncorrected"
    )
    next_phase_scope_valid = (
        isinstance(next_phase, dict)
        and next_phase.get("stage") == "INITIALISED"
        and next_phase.get("working_name") == "AANCA v2 research phase"
        and next_phase.get("natural_auto_change_allowed") is False
        and isinstance(next_phase.get("required_gates"), list)
        and len(next_phase["required_gates"]) == 5
    )
    if (
        evidence.get("schema_version") != 3
        or evidence.get("presentation_status") != "DEMO_COMPLETE"
        or evidence.get("scientific_status") != "EXTERNAL_VALIDATION_COMPLETE"
        or evidence.get("primary_study_status") != "PRIMARY_STUDY_COMPLETE"
        or evidence.get("confirmatory_completed") is not False
        or not external_scope_valid
        or not controlled_scope_valid
        or not puma_scope_valid
        or not stress_scope_valid
        or not sensitivity_scope_valid
        or not natural_action_scope_valid
        or not next_phase_scope_valid
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
        "scientific_status": "EXTERNAL_VALIDATION_COMPLETE",
        "external_validation_completed": external_completed,
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
