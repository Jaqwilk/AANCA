"""Analyse the frozen expanded-search runtime amendment without rerunning metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from histo_audit.research.runtime_amendment import (
    analyse_runtime_amendment,
    load_runtime_amendment,
)
from histo_audit.utils.run_tracking import atomic_write_json, atomic_write_text, sha256_file


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"ledger line {line_number} is not an object")
        records.append(value)
    return records


def _verify_parent_sources(authority: dict[str, Any], project_root: Path) -> None:
    direct = {
        "base_evaluator_source_sha256": "src/histo_audit/research/autoresearch.py",
        "expanded_evaluator_source_sha256": ("src/histo_audit/research/expanded_autoresearch.py"),
        "runner_source_sha256": "scripts/run_aanca_autoresearch_expanded.py",
        "protocol_sha256": "AANCA_AUTORESEARCH_EXPANDED_PROTOCOL.md",
        "program_sha256": "AANCA_AUTORESEARCH_PROGRAM.md",
    }
    for authority_name, relative_path in direct.items():
        expected = authority.get(authority_name)
        if not isinstance(expected, str) or sha256_file(project_root / relative_path) != expected:
            raise ValueError(f"parent source identity changed: {relative_path}")
    dependencies = authority.get("evaluation_dependency_sha256")
    if not isinstance(dependencies, dict):
        raise ValueError("parent authority has no dependency hashes")
    for relative_path, expected in dependencies.items():
        if sha256_file(project_root / str(relative_path)) != expected:
            raise ValueError(f"parent dependency identity changed: {relative_path}")


def _render_report(summary: dict[str, Any]) -> str:
    selected = summary.get("selected_full_result")
    lines = [
        "# Expanded autoresearch runtime-amendment result",
        "",
        f"- Frozen finalists: {summary['frozen_finalist_count']}",
        f"- Amended per-candidate runtime: {summary['amended_runtime_budget_seconds']:.0f} s",
        f"- Development disposition: `{summary['development_disposition']}`",
        f"- Executable action: `{summary['executable_action']}`",
        "",
    ]
    if not isinstance(selected, dict):
        lines.extend(
            [
                "No finalist passed every frozen full nested-development gate.",
                "",
            ]
        )
    else:
        downstream = selected["downstream"]
        retrieval = selected["retrieval"]
        candidate = selected["candidate"]
        lines.extend(
            [
                "## Selected development candidate",
                "",
                f"- SHA-256: `{selected['candidate_sha256']}`",
                f"- Feature view: `{candidate['feature_view']}`",
                f"- Queue: `{candidate['queue_preset']}`",
                f"- Review budget: {100.0 * float(candidate['review_budget']):.1f}%",
                f"- Intervention: `{candidate['intervention']}`",
                "- Candidate minus uncorrected macro-F1: "
                f"{float(downstream['candidate_minus_uncorrected_macro_f1']):+.6f} "
                f"(95% CI {downstream['candidate_minus_uncorrected_interval_95']})",
                "- Candidate minus exact matched-random macro-F1: "
                f"{float(downstream['candidate_minus_matched_random_macro_f1']):+.6f} "
                f"(95% CI {downstream['candidate_minus_matched_random_interval_95']})",
                "- Retrieval precision gain over exact matched random: "
                f"{float(retrieval['candidate_minus_matched_random_precision']):+.6f} "
                f"(95% CI {retrieval['interval_95']})",
                "",
            ]
        )
        if summary["selected_candidate_tcga_pretraining_overlap_limitation"]:
            lines.extend(
                [
                    "The selected Phikon-v2 representation has a TCGA pretraining-overlap ",
                    "limitation for MoNuSAC and is not an independent external confirmation.",
                    "",
                ]
            )

    overlap_free = summary.get("best_overlap_free_full_result")
    if isinstance(overlap_free, dict) and (
        not isinstance(selected, dict)
        or overlap_free["candidate_sha256"] != selected["candidate_sha256"]
    ):
        downstream = overlap_free["downstream"]
        lines.extend(
            [
                "## Best passing candidate without known TCGA encoder overlap",
                "",
                f"- SHA-256: `{overlap_free['candidate_sha256']}`",
                f"- Feature view: `{overlap_free['candidate']['feature_view']}`",
                "- Candidate minus uncorrected macro-F1: "
                f"{float(downstream['candidate_minus_uncorrected_macro_f1']):+.6f}",
                "- Candidate minus exact matched-random macro-F1: "
                f"{float(downstream['candidate_minus_matched_random_macro_f1']):+.6f}",
                "",
            ]
        )

    lines.extend(
        [
            "## All frozen finalists",
            "",
            "| Candidate | Features | Risk | Budget | Intervention | Δ F1 vs unchanged "
            "(lower) | Δ F1 vs matched random (lower) | Minimum class-recall lower | Decision |",
            "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for record in summary["amended_full_results"]:
        candidate = record["candidate"]
        downstream = record["downstream"]
        recall_intervals = downstream["candidate_minus_uncorrected_recall_intervals_95"]
        minimum_recall_lower = min(float(interval[0]) for interval in recall_intervals)
        lines.append(
            "| `"
            + str(record["candidate_sha256"])[:12]
            + "` | `"
            + str(candidate["feature_view"])
            + "` | `"
            + str(candidate["risk_method"])
            + "` | "
            + f"{100.0 * float(candidate['review_budget']):.1f}%"
            + " | `"
            + str(candidate["intervention"])
            + "` | "
            + f"{float(downstream['candidate_minus_uncorrected_macro_f1']):+.6f} "
            + f"({float(downstream['candidate_minus_uncorrected_interval_95'][0]):+.6f})"
            + " | "
            + f"{float(downstream['candidate_minus_matched_random_macro_f1']):+.6f} "
            + f"({float(downstream['candidate_minus_matched_random_interval_95'][0]):+.6f})"
            + " | "
            + f"{minimum_recall_lower:+.6f}"
            + " | `"
            + str(record["status"])
            + "` |"
        )
    lines.append("")
    lines.extend(
        [
            "## Boundary",
            "",
            "This is development evidence from controlled corruption on MoNuSAC training data. ",
            "It does not prove natural pathologist-error detection or real-workflow superiority. ",
            "Source annotations remain unchanged and the executable action remains ",
            "`retain_uncorrected` pending genuinely new external and blinded-expert validation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--amendment",
        default="configs/aanca_autoresearch_full_runtime_amendment.yaml",
    )
    parser.add_argument(
        "--output-root",
        default=("artifacts/autoresearch/monusac_aanca_expanded_full_runtime_amendment_v1"),
    )
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    amendment_path = (project_root / args.amendment).resolve()
    checksum_path = amendment_path.with_suffix(amendment_path.suffix + ".sha256")
    expected_amendment_sha256 = checksum_path.read_text(encoding="utf-8").split()[0]
    if sha256_file(amendment_path) != expected_amendment_sha256:
        raise ValueError("runtime amendment checksum verification failed")
    amendment = load_runtime_amendment(amendment_path)
    parent_root = (project_root / str(amendment["parent_output_root"])).resolve()
    authority = _load_json(parent_root / "run_authority.json")
    _verify_parent_sources(authority, project_root)
    parent_config_path = project_root / str(amendment["parent_config_path"])
    if sha256_file(parent_config_path) != amendment["parent_config_sha256"]:
        raise ValueError("parent config file hash differs from frozen amendment")
    records = _load_jsonl(parent_root / "results.jsonl")
    summary = analyse_runtime_amendment(amendment, authority, records)
    summary["amendment_config_sha256"] = sha256_file(amendment_path)
    summary["parent_authority_sha256"] = sha256_file(parent_root / "run_authority.json")
    summary["parent_ledger_sha256"] = sha256_file(parent_root / "results.jsonl")

    output_root = (project_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "summary.json", summary)
    atomic_write_text(output_root / "report.md", _render_report(summary))
    if summary["selected_candidate"] is not None:
        atomic_write_json(
            output_root / "selected_development_candidate.json",
            {
                "schema_version": 1,
                "amendment_id": amendment["amendment_id"],
                "candidate": summary["selected_candidate"],
                "candidate_sha256": summary["selected_candidate_sha256"],
                "role": "development_candidate_only",
                "external_confirmation_complete": False,
                "natural_error_detection_proven": False,
                "real_use_superiority_proven": False,
                "automatic_annotation_change_permitted": False,
                "executable_action_until_new_confirmation": "retain_uncorrected",
            },
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
