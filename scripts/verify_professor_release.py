#!/usr/bin/env python3
"""Fail closed when professor-facing AANCA materials drift from sealed evidence."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PRESENTATION_VERIFIER = runpy.run_path(str(PROJECT_ROOT / "scripts" / "present_demo.py"))
DEFAULT_OUTPUT = _PRESENTATION_VERIFIER["DEFAULT_OUTPUT"]
verify_presentation = _PRESENTATION_VERIFIER["verify_presentation"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_records(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if {"path", "sha256", "size_bytes"}.issubset(value):
            yield value
            return
        for child in value.values():
            yield from _source_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _source_records(child)


def _read(relative: str) -> str:
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def _require(text: str, fragments: list[str], *, role: str) -> None:
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        raise ValueError(f"{role} is missing required evidence text: {missing!r}")


def _verify_source_record(record: Mapping[str, Any]) -> str:
    relative = record.get("path")
    size = record.get("size_bytes")
    digest = record.get("sha256")
    if (
        not isinstance(relative, str)
        or not relative
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        raise ValueError(f"malformed evidence source record: {record!r}")

    root = DEFAULT_OUTPUT if relative.startswith("assets/") else PROJECT_ROOT
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"evidence source escapes its authority root: {relative}") from exc
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"evidence source is not a regular file: {relative}")
    if path.stat().st_size != size or _sha256(path) != digest:
        raise ValueError(f"evidence source differs from its published identity: {relative}")
    return f"{root.name}:{relative}"


def verify_professor_release() -> dict[str, Any]:
    """Verify package, upstream authorities and the concise public narrative."""

    package = verify_presentation(DEFAULT_OUTPUT)
    evidence = json.loads((DEFAULT_OUTPUT / "evidence.json").read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise ValueError("presentation evidence root must be an object")

    source_paths = sorted({_verify_source_record(record) for record in _source_records(evidence)})
    if len(source_paths) < 10:
        raise ValueError("too few upstream evidence authorities were bound to the release")

    nucls = evidence["independent_pathologist_validation"]
    nucls_primary = nucls["primary"]
    riva = evidence["public_independent_pathologist_replication"]["datasets"]["riva"]
    midogpp = evidence["public_independent_pathologist_replication"]["datasets"]["midogpp"]
    puma = evidence["new_source_confirmation"]

    professor_brief = _read("PROFESSOR_BRIEF.md")
    public_evidence = _read("PUBLIC_EVIDENCE.md")
    ethics = _read("ETHICS_AND_LIMITATIONS.md")
    for role, text in (
        ("PROFESSOR_BRIEF.md", professor_brief),
        ("PUBLIC_EVIDENCE.md", public_evidence),
    ):
        _require(
            text,
            [
                f"`{nucls_primary['aanca_precision']:.6f}`",
                f"`{nucls_primary['mean_matched_random_precision']:.6f}`",
                f"`{nucls_primary['precision_difference']:+.6f}`",
                f"`{riva['aanca_precision']:.6f}`",
                f"`{riva['mean_matched_random_precision']:.6f}`",
                f"`{riva['precision_difference']:+.6f}`",
                f"`{midogpp['aanca_precision']:.6f}`",
                f"`{midogpp['mean_matched_random_precision']:.6f}`",
                f"`{midogpp['precision_difference']:+.6f}`",
                f"`{puma['retrieval']['candidate_precision']:.6f}`",
                f"`{puma['retrieval']['mean_matched_random_precision']:.6f}`",
                f"`{puma['downstream']['minus_uncorrected']:+.6f}`",
                "`retain_uncorrected`",
            ],
            role=role,
        )

    _require(
        ethics,
        [
            f"`{nucls_primary['aanca_precision']:.6f}`",
            f"`{nucls_primary['mean_matched_random_precision']:.6f}`",
            "RIVA and MIDOG++",
            "`retain_uncorrected`",
            "not a medical device",
        ],
        role="ETHICS_AND_LIMITATIONS.md",
    )

    presentation_html = (DEFAULT_OUTPUT / "index.html").read_text(encoding="utf-8")
    _require(
        presentation_html,
        [
            "Evidence at a glance",
            f">{100.0 * nucls_primary['precision_difference']:+.2f}</strong> pp",
            f">{100.0 * riva['precision_difference']:+.2f}</strong> pp",
            f">{100.0 * midogpp['precision_difference']:+.2f}</strong> pp",
            f">{100.0 * puma['retrieval']['difference']:+.2f}</strong> pp",
            "Neither permits automatic label changes or establishes clinical utility.",
        ],
        role="sealed presentation",
    )

    readme = _read("README.md")
    _require(
        readme,
        [
            '<h1 align="center">AANCA</h1>',
            "https://aancastudy.org",
            "potentially inconsistent annotations",
            "recommended for expert review",
            "never modifies source",
            "## Current conclusion",
            f"`{nucls_primary['aanca_precision']:.6f}`",
            f"`{nucls_primary['mean_matched_random_precision']:.6f}`",
            f"`{riva['aanca_precision']:.6f}`",
            f"`{riva['mean_matched_random_precision']:.6f}`",
            f"`{riva['precision_difference']:+.6f}`",
            f"`{midogpp['aanca_precision']:.6f}`",
            f"`{midogpp['mean_matched_random_precision']:.6f}`",
            f"`{midogpp['precision_difference']:+.6f}`",
            f"`{puma['retrieval']['candidate_precision']:.6f}`",
            f"`{puma['retrieval']['mean_matched_random_precision']:.6f}`",
            "`retain_uncorrected`",
            "`CONFIRMATORY_COMPLETE` has not been reached",
            "## Reproducibility levels",
            "## Validation gates",
        ],
        role="README.md",
    )

    _require(
        _read("CITATION.cff"),
        ["date-released: 2026-08-26", "independent-expert disagreement"],
        role="CITATION.cff",
    )
    _require(
        _read("REPRODUCIBILITY.md"),
        ["closed thirteen-file allowlist", "run_public_pathologist_replication.py verify"],
        role="REPRODUCIBILITY.md",
    )
    _require(
        _read("MVP_SCOPE.md"),
        ["Current-status addendum — 26 August 2026", "current thirteen-file package"],
        role="MVP_SCOPE.md",
    )
    _require(
        _read("STATUS.md"),
        ["Updated: 1 September 2026", "Final professor-readiness audit"],
        role="STATUS.md",
    )
    _require(
        professor_brief,
        ["FINAL_READINESS_REPORT.md", "not third-party clinical validation"],
        role="PROFESSOR_BRIEF.md",
    )
    _require(
        _read("FINAL_READINESS_REPORT.md"),
        [
            "# AANCA final readiness report",
            "EXTERNAL_VALIDATION_COMPLETE",
            "DEMO_COMPLETE",
            "No adjudicated natural-error claim",
        ],
        role="FINAL_READINESS_REPORT.md",
    )

    if len(professor_brief.encode("utf-8")) > 7_000:
        raise ValueError("PROFESSOR_BRIEF.md exceeded its concise handout size limit")

    return {
        "status": "valid",
        "package_manifest_root_sha256": package["manifest_root_sha256"],
        "package_file_count": package["file_count"],
        "upstream_authority_count": len(source_paths),
        "professor_brief_bytes": len(professor_brief.encode("utf-8")),
        "scientific_status": package["scientific_status"],
        "presentation_status": package["presentation_status"],
    }


def main() -> int:
    try:
        result = verify_professor_release()
    except Exception as exc:
        print(
            f"ERROR: professor release verification failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
