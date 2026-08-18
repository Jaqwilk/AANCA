"""Sourced Markdown and standalone HTML reporting for synthetic runs."""

from __future__ import annotations

import html
import json
import math
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from histo_audit.reporting.figures import build_synthetic_figures
from histo_audit.utils.run_tracking import (
    atomic_write_text,
    sealed_run_ancestor,
    sha256_file,
)

_PLACEHOLDER_PATTERN = re.compile(
    r"(?:\bTODO\b|\bTBD\b|\bPLACEHOLDER\b|REPLACE[ _-]?ME|DUMMY[ _-]?METRIC|COMING[ _-]?SOON)",
    re.IGNORECASE,
)
_MISSING_STATUSES = {
    "failed",
    "insufficient_support",
    "not_applicable",
    "not_computed",
    "unavailable",
}


@dataclass(frozen=True, slots=True)
class ReportArtifacts:
    """Paths and source digest for one generated report pair."""

    markdown_path: Path
    html_path: Path
    metrics_path: Path
    metrics_sha256: str
    figure_paths: tuple[Path, ...]
    figure_manifest_path: Path | None


def _missing_value_is_documented(
    parent: Mapping[str, Any] | None,
    *,
    field_name: str,
) -> bool:
    if parent is None:
        return False
    status = str(parent.get("status", "")).strip().lower()
    reason = parent.get("reason") or parent.get("blocker") or parent.get("error")
    if status in _MISSING_STATUSES and isinstance(reason, str) and bool(reason.strip()):
        return True
    available = parent.get("available")
    if available is False and isinstance(reason, str) and bool(reason.strip()):
        return True
    # A null diagnostic error is positive evidence when an optional method says
    # it succeeded; it is metadata, not a missing result value.
    if available is True and field_name in {"error", "reason", "blocker"}:
        return True
    return field_name in {"error", "reason", "blocker"} and status in {
        "available",
        "completed",
        "passed",
        "reported",
        "success",
    }


def validate_metrics_payload(metrics: Mapping[str, Any]) -> None:
    """Reject unsourced placeholder-like or non-finite metric values.

    A JSON ``null`` is allowed only for an explicitly failed, unavailable, or
    not-applicable result that includes a non-empty reason/blocker/error.  This
    supports honest optional-method failures without fabricating a replacement.
    """

    if not metrics:
        raise ValueError("metrics payload must not be empty")

    def visit(value: Any, path: str, parent: Mapping[str, Any] | None = None) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError(f"metrics contain an empty or non-string key at {path}")
                if _PLACEHOLDER_PATTERN.search(key):
                    raise ValueError(f"placeholder metric key rejected at {path}.{key}")
                visit(nested, f"{path}.{key}", value)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]", parent)
            return
        if value is None:
            field_name = path.rsplit(".", maxsplit=1)[-1]
            if not _missing_value_is_documented(parent, field_name=field_name):
                raise ValueError(f"undocumented missing metric rejected at {path}")
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"non-finite metric rejected at {path}")
            return
        if isinstance(value, str):
            if not value.strip():
                raise ValueError(f"empty metric value rejected at {path}")
            if _PLACEHOLDER_PATTERN.search(value):
                raise ValueError(f"placeholder metric value rejected at {path}: {value!r}")
            return
        if not isinstance(value, (bool, int)):
            raise ValueError(f"unsupported metric value at {path}: {type(value).__name__}")

    visit(metrics, "metrics")


def load_metrics(metrics_path: str | Path) -> dict[str, Any]:
    """Load a strict JSON object and validate that it contains no placeholders."""

    source = Path(metrics_path)
    if not source.is_file():
        raise FileNotFoundError(f"metrics JSON does not exist: {source}")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant rejected: {value}")

    try:
        parsed = json.loads(source.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid metrics JSON at {source}: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"metrics JSON root must be an object: {source}")
    metrics = dict(parsed)
    validate_metrics_payload(metrics)
    return metrics


def _flatten(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(value[key], path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            yield from _flatten(nested, f"{prefix}[{index}]")
    else:
        yield prefix, value


def _display_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    if isinstance(value, (int, str)):
        return str(value)
    raise TypeError(f"unsupported report value: {type(value).__name__}")


def _markdown_cell(value: Any) -> str:
    return _display_value(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _select_rows(
    rows: list[tuple[str, Any]],
    keywords: tuple[str, ...],
) -> list[tuple[str, Any]]:
    selected = [row for row in rows if any(keyword in row[0].lower() for keyword in keywords)]
    return selected


def _aggregate_rows(rows: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """Omit repeat-level arrays while retaining their saved aggregate summaries."""

    repeat_path = re.compile(r"(?:^|\.)(?:runs|seeds)\[")
    return [row for row in rows if repeat_path.search(row[0]) is None]


def _metric_table(rows: Sequence[tuple[str, Any]]) -> str:
    if not rows:
        return "No result in this category was present in the supplied machine-readable artifact."
    lines = ["| Artifact field | Saved value |", "|---|---:|"]
    lines.extend(f"| `{_markdown_cell(path)}` | {_markdown_cell(value)} |" for path, value in rows)
    return "\n".join(lines)


def _metric_or_status(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _display_value(value)
    if isinstance(value, Mapping):
        status = str(value.get("status", "unavailable"))
        reason = value.get("reason")
        return f"{status}: {reason}" if isinstance(reason, str) else status
    return _display_value(value)


def _ranking_summary_table(metrics: Mapping[str, Any]) -> str:
    ranking = metrics.get("ranking")
    if not isinstance(ranking, Mapping):
        return _metric_table(())
    lines = ["| Audit method | AUROC |", "|---|---:|"]
    for method, payload in sorted(ranking.items()):
        if not isinstance(payload, Mapping) or "auroc" not in payload:
            continue
        lines.append(
            f"| {_markdown_cell(str(method).replace('_', ' '))} | "
            f"{_markdown_cell(_metric_or_status(payload['auroc']))} |"
        )
    return "\n".join(lines) if len(lines) > 2 else _metric_table(())


def _subgroup_summary_table(metrics: Mapping[str, Any]) -> str:
    ranking = metrics.get("ranking")
    if not isinstance(ranking, Mapping):
        return _metric_table(())
    lines = [
        "| Audit method | Subgroup dimension | Subgroup | N | Injected | AP/status |",
        "|---|---|---|---:|---:|---|",
    ]
    for method, method_payload in sorted(ranking.items()):
        if not isinstance(method_payload, Mapping):
            continue
        subgroups = method_payload.get("subgroups")
        if not isinstance(subgroups, Mapping):
            continue
        for dimension, entries in sorted(subgroups.items()):
            if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        _markdown_cell(value)
                        for value in (
                            str(method).replace("_", " "),
                            str(dimension).replace("_", " "),
                            entry.get("subgroup", "unavailable"),
                            entry.get("total_examples", "unavailable"),
                            entry.get("injected_corruptions", "unavailable"),
                            _metric_or_status(entry.get("average_precision")),
                        )
                    )
                    + " |"
                )
    return "\n".join(lines) if len(lines) > 2 else _metric_table(())


def _figure_markdown(
    figure_links: Mapping[str, tuple[str, str]],
    key: str,
) -> str:
    figure = figure_links.get(key)
    if figure is None:
        return ""
    alt_text, relative_path = figure
    return f"![{alt_text}]({relative_path})"


def _duplicate_audit_markdown(
    payload: Mapping[str, Any] | None,
    *,
    json_link: str | None,
    json_sha256: str | None,
    csv_link: str | None,
) -> str:
    if payload is None:
        return "No tracked synthetic source-patch duplicate audit was supplied to this report."
    dataset = payload.get("dataset_evidence")
    counts = payload.get("candidate_counts")
    pairs = payload.get("pair_counts")
    thresholds = payload.get("thresholds")
    if not all(isinstance(value, Mapping) for value in (dataset, counts, pairs, thresholds)):
        raise ValueError("synthetic duplicate audit lacks reportable counts or thresholds")
    assert isinstance(dataset, Mapping)
    assert isinstance(counts, Mapping)
    assert isinstance(pairs, Mapping)
    assert isinstance(thresholds, Mapping)
    rows = [
        ("nucleus rows in dataset evidence", dataset.get("nucleus_sample_count")),
        ("unique source patches after patch_id deduplication", dataset.get("unique_patch_count")),
        ("evaluated cross-boundary patch pairs", pairs.get("evaluated_cross_boundary_pairs")),
        ("exact candidates", counts.get("exact")),
        ("perceptual candidates (including exact)", counts.get("perceptual_including_exact")),
        ("union candidates", counts.get("union")),
        ("perceptual hash algorithm", thresholds.get("perceptual_hash_algorithm")),
        ("perceptual hash bits", thresholds.get("perceptual_hash_bits")),
        (
            "maximum perceptual Hamming distance",
            thresholds.get("max_perceptual_hamming_distance"),
        ),
        ("automatic deletion", payload.get("automatic_deletion")),
        ("real-data duplicate gate eligible", payload.get("real_data_duplicate_gate_eligible")),
    ]
    source_parts: list[str] = []
    if json_link is not None:
        source_parts.append(f"[duplicate audit JSON]({json_link})")
    if csv_link is not None:
        source_parts.append(f"[complete candidate CSV]({csv_link})")
    source_text = ", ".join(source_parts) if source_parts else "saved duplicate artifacts"
    digest_text = f" JSON SHA-256: `{json_sha256}`." if json_sha256 else ""
    return "\n\n".join(
        (
            (
                "This read-only software-validation audit first collapses nucleus rows by "
                "`patch_id`, then compares unique synthetic source patches across different "
                "split partitions or official folds. Exact equality and deterministic average "
                "hash are derived from the same synthetic pixels; they are not independent "
                "PanNuke perceptual-plus-embedding evidence and cannot satisfy a real-data "
                "duplicate gate. No source item is deleted or modified."
            ),
            f"Machine-readable sources: {source_text}.{digest_text}",
            _metric_table(rows),
        )
    )


def render_synthetic_markdown(
    metrics: Mapping[str, Any],
    *,
    metrics_path: str | Path,
    metrics_sha256: str,
    run_id: str,
    metrics_link: str | None = None,
    figure_links: Mapping[str, tuple[str, str]] | None = None,
    duplicate_audit: Mapping[str, Any] | None = None,
    duplicate_audit_link: str | None = None,
    duplicate_audit_sha256: str | None = None,
    duplicate_candidates_csv_link: str | None = None,
) -> str:
    """Render a synthetic report whose result values all come from *metrics*."""

    validate_metrics_payload(metrics)
    links = figure_links or {}
    rows = list(_flatten(metrics))
    aggregate_rows = _aggregate_rows(rows)
    budget_rows = [
        row
        for row in _select_rows(aggregate_rows, ("budget", "review", "precision", "recall", "lift"))
        if ".subgroups." not in row[0]
    ]
    restoration_rows = _select_rows(
        aggregate_rows,
        ("restoration", "macro_f1", "macro f1", "balanced_accuracy", "downstream"),
    )
    split_rows = _select_rows(aggregate_rows, ("fold", "group", "split", "leak", "coverage"))
    corruption_rows = _select_rows(
        aggregate_rows, ("corrupt", "injected", "circularity", "independ")
    )
    ranking_rows = [
        row
        for row in _select_rows(
            aggregate_rows,
            ("average_precision", "auprc", "auroc", "self_confidence", "hybrid", "entropy"),
        )
        if ".subgroups." not in row[0]
        and ".review_budgets." not in row[0]
        and ".score_distribution." not in row[0]
    ]
    source = Path(metrics_path).resolve()
    sections = [
        "# Synthetic Annotation-Auditing Software-Validation Report",
        "",
        "## Executive summary",
        "",
        "This report describes a controlled synthetic software-validation run. It is not a diagnostic result and does not establish that any source annotation is medically wrong. Potentially inconsistent annotations are recommended for expert review only.",
        "",
        f"Run ID: `{run_id}`  ",
        f"Machine-readable source: `{source}`  ",
        f"Source SHA-256: `{metrics_sha256}`",
        "",
        "## Research question",
        "",
        "Can group-safe automated auditing rank intentionally injected nucleus-class corruptions more efficiently than random review, and can restoration of reviewed injected labels improve downstream classification?",
        "",
        "## Prior work and novelty",
        "",
        "This software report does not claim novelty for label-error detection or noisy-label learning; it documents a reproducible controlled auditing workflow.",
        "",
        "## Dataset provenance",
        "",
        "The run is synthetic software-validation evidence, not a PanNuke or clinical result. Dataset-related values below are copied from the saved artifact.",
        "",
        _metric_table(
            _select_rows(
                aggregate_rows,
                (
                    "artifact_scope",
                    "sample_counts",
                    "dataset_seed",
                    "n_groups",
                    "instances_per_group",
                    "patch_size",
                ),
            )
        ),
        "",
        _figure_markdown(links, "class_distribution"),
        "",
        "## Terminology",
        "",
        "`pre_corruption_label` is the source label before intentional project corruption; `observed_label` is exposed to the model; `is_injected_corruption` records intentional changes; `restored_label` is used only after simulated review.",
        "",
        "## Data validation",
        "",
        _metric_table(_select_rows(aggregate_rows, ("valid", "count", "shape", "sample"))),
        "",
        "## Duplicate analysis",
        "",
        _duplicate_audit_markdown(
            duplicate_audit,
            json_link=duplicate_audit_link,
            json_sha256=duplicate_audit_sha256,
            csv_link=duplicate_candidates_csv_link,
        ),
        "",
        _figure_markdown(links, "duplicate_candidates"),
        "",
        "## Splitting and leakage prevention",
        "",
        _metric_table(split_rows),
        "",
        _figure_markdown(links, "oof_fold_distribution"),
        "",
        "## Target-nucleus representation",
        "",
        _metric_table(_select_rows(aggregate_rows, ("representation", "target", "feature"))),
        "",
        _figure_markdown(links, "target_representation_example"),
        "",
        _figure_markdown(links, "target_example_audit_evidence"),
        "",
        "## Controlled corruption",
        "",
        _metric_table(corruption_rows),
        "",
        _figure_markdown(links, "corruption_transition_matrix"),
        "",
        "## Encoders and models",
        "",
        _metric_table(_select_rows(aggregate_rows, ("encoder", "model", "classifier"))),
        "",
        "## Auditing methods",
        "",
        _metric_table(
            _select_rows(aggregate_rows, ("audit", "risk", "method", "confidence", "neighbour"))
        ),
        "",
        "## Preregistration",
        "",
        "Synthetic smoke results are software-validation artifacts and do not freeze the primary study.",
        "",
        "## Primary metrics",
        "",
        _metric_table(ranking_rows),
        "",
        _ranking_summary_table(metrics),
        "",
        _figure_markdown(links, "average_precision_by_method"),
        "",
        _figure_markdown(links, "precision_recall_curves"),
        "",
        _figure_markdown(links, "score_distribution_by_method"),
        "",
        (
            "For the 0% edge case, score distributions retain each method's native raw scale. "
            "Positions are interpretable within a method only and must not be compared as "
            "absolute risk magnitudes across methods."
            if "score_distribution_by_method" in links
            else ""
        ),
        "",
        "## Statistical analysis",
        "",
        _metric_table(
            _select_rows(aggregate_rows, ("bootstrap", "interval", "random", "difference"))
        ),
        "",
        _figure_markdown(links, "paired_bootstrap_interval"),
        "",
        _figure_markdown(links, "paired_method_differences"),
        "",
        _figure_markdown(links, "paired_bootstrap_distribution"),
        "",
        "## Review-budget results",
        "",
        _metric_table(budget_rows),
        "",
        _figure_markdown(links, "recall_vs_budget"),
        "",
        _figure_markdown(links, "lift_vs_budget"),
        "",
        _figure_markdown(links, "false_alerts_vs_review_budget"),
        "",
        "## Downstream restoration",
        "",
        _metric_table(restoration_rows),
        "",
        _figure_markdown(links, "downstream_macro_f1"),
        "",
        "## Class and tissue analysis",
        "",
        _subgroup_summary_table(metrics),
        "",
        _figure_markdown(links, "per_class_results_support"),
        "",
        _figure_markdown(links, "per_tissue_results_support"),
        "",
        _figure_markdown(links, "tissue_distribution"),
        "",
        "## Representation ablations",
        "",
        _metric_table(_select_rows(aggregate_rows, ("ablation", "representation"))),
        "",
        "## Corruption/auditor independence",
        "",
        _metric_table(
            _select_rows(aggregate_rows, ("circularity", "independ", "generator", "auditor"))
        ),
        "",
        "## Controlled error analysis",
        "",
        _metric_table(_select_rows(aggregate_rows, ("false", "error", "miss"))),
        "",
        _figure_markdown(links, "top_suspicious_controlled_examples"),
        "",
        _figure_markdown(links, "fold_safe_neighbour_explanation_grid"),
        "",
        _figure_markdown(links, "false_high_and_low_risk_examples"),
        "",
        _figure_markdown(links, "false_high_risk_examples"),
        "",
        "## Exploratory original-label ranking",
        "",
        "Not performed by this synthetic controlled benchmark. No natural source-label error is inferred.",
        "",
        "## External-validation status",
        "",
        "No expert-reviewed or external multi-rater validation is represented by this synthetic report.",
        "",
        "## Aggregate machine-readable result table",
        "",
        (
            "Repeat-level run records, retained random seeds, and all aggregate fields remain "
            f"in the [machine-readable metrics JSON]({metrics_link}); its SHA-256 is recorded "
            "above. They are intentionally not duplicated into this human-readable report."
            if metrics_link
            else "Repeat-level run records, retained random seeds, and all aggregate fields remain in the machine-readable metrics JSON identified by the source path and SHA-256 above. They are intentionally not duplicated into this human-readable report."
        ),
        "",
        "## Limitations",
        "",
        "Injected corruptions test the injected process and may not represent naturally occurring annotation inconsistencies. Model disagreement alone does not prove annotation error.",
        "",
        "## Ethical disclaimer",
        "",
        "This university research prototype is not a medical device and is not intended for diagnosis, prognosis, treatment, or automatic source-annotation modification.",
        "",
        "## Reproduction instructions",
        "",
        "Re-run the tracked smoke command with the resolved configuration stored in the associated immutable run directory. Verify the metrics SHA-256 above before comparing values.",
        "",
        "## Future work",
        "",
        "Real-data pilot, preregistered primary analysis, confirmatory experiments, and genuine external validation remain separately gated.",
        "",
    ]
    return "\n".join(sections)


def _inline_markdown(value: str) -> str:
    escaped = html.escape(value, quote=True)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)
    return escaped


def _split_table_row(line: str) -> list[str]:
    raw = line.strip().strip("|")
    cells = re.split(r"(?<!\\)\|", raw)
    return [cell.strip().replace("\\|", "|") for cell in cells]


def markdown_to_static_html(markdown: str, *, title: str = "Histo Audit Report") -> str:
    """Convert the generated Markdown subset into a standalone HTML document."""

    lines = markdown.splitlines()
    body: list[str] = []
    index = 0
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            body.append(f"<p>{_inline_markdown(' '.join(paragraph))}</p>")
            paragraph.clear()

    while index < len(lines):
        line = lines[index]
        image_match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
        if image_match:
            flush_paragraph()
            alt_text = html.escape(image_match.group(1), quote=True)
            source = html.escape(image_match.group(2), quote=True)
            body.append(f'<figure><img src="{source}" alt="{alt_text}" loading="lazy"></figure>')
            index += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            body.append(f"<h{level}>{_inline_markdown(heading.group(2))}</h{level}>")
            index += 1
            continue
        if (
            line.lstrip().startswith("|")
            and index + 1 < len(lines)
            and re.match(r"^\s*\|?\s*:?-+", lines[index + 1])
        ):
            flush_paragraph()
            headers = _split_table_row(line)
            index += 2
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                rows.append(_split_table_row(lines[index]))
                index += 1
            table = ['<div class="table-wrap"><table><thead><tr>']
            table.extend(f"<th>{_inline_markdown(cell)}</th>" for cell in headers)
            table.append("</tr></thead><tbody>")
            for row in rows:
                table.append("<tr>")
                table.extend(f"<td>{_inline_markdown(cell)}</td>" for cell in row)
                table.append("</tr>")
            table.append("</tbody></table></div>")
            body.append("".join(table))
            continue
        if re.match(r"^\s*[-*]\s+", line):
            flush_paragraph()
            items: list[str] = []
            while index < len(lines):
                match = re.match(r"^\s*[-*]\s+(.+)$", lines[index])
                if match is None:
                    break
                items.append(f"<li>{_inline_markdown(match.group(1))}</li>")
                index += 1
            body.append(f"<ul>{''.join(items)}</ul>")
            continue
        if not line.strip():
            flush_paragraph()
        else:
            paragraph.append(line.strip().removesuffix("  "))
        index += 1
    flush_paragraph()
    css = """
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;line-height:1.55;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1d2630;background:#fff}
h1,h2{color:#243b53}h2{margin-top:2rem;border-bottom:1px solid #d9e2ec;padding-bottom:.3rem}
code{background:#f0f4f8;padding:.1rem .25rem;border-radius:3px}.table-wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.92rem}th,td{border:1px solid #bcccdc;padding:.45rem .6rem;text-align:left;vertical-align:top}th{background:#f0f4f8}td:last-child{font-variant-numeric:tabular-nums}
figure{margin:1.2rem 0}figure img{display:block;max-width:100%;height:auto;margin:auto;border:1px solid #d9e2ec;border-radius:4px}
""".strip()
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{css}</style></head><body>"
        f"{''.join(body)}</body></html>\n"
    )


def build_synthetic_report(
    metrics_path: str | Path,
    *,
    output_directory: str | Path | None = None,
    run_id: str | None = None,
    markdown_name: str = "report.md",
    html_name: str = "report.html",
    predictions_path: str | Path | None = None,
    representation_example_path: str | Path | None = None,
    rankings_path: str | Path | None = None,
    neighbour_evidence_path: str | Path | None = None,
    bootstrap_evidence_path: str | Path | None = None,
    dataset_evidence_path: str | Path | None = None,
    corruption_manifest_path: str | Path | None = None,
    duplicate_audit_path: str | Path | None = None,
    duplicate_candidates_csv_path: str | Path | None = None,
    duplicate_candidates_figure_path: str | Path | None = None,
    class_names: Sequence[str] | None = None,
    generate_figures: bool = True,
) -> ReportArtifacts:
    """Build sourced Markdown and HTML reports atomically from metrics JSON."""

    source = Path(metrics_path).resolve()
    metrics = load_metrics(source)
    digest = sha256_file(source)
    identifier_value = run_id or metrics.get("run_id") or source.parent.name
    identifier = str(identifier_value).strip()
    if not identifier:
        raise ValueError("a non-empty run_id is required for report provenance")
    destination = Path(output_directory).resolve() if output_directory else source.parent
    sealed_run = sealed_run_ancestor(destination)
    if sealed_run is not None:
        raise PermissionError(
            "report output would modify a sealed immutable run: "
            f"{sealed_run}. Choose a separate external --output-dir."
        )
    destination.mkdir(parents=True, exist_ok=True)
    markdown_path = destination / markdown_name
    html_path = destination / html_name
    prediction_source = Path(predictions_path).resolve() if predictions_path else None
    if prediction_source is None:
        candidate = source.parent / "oof_predictions.npz"
        if candidate.is_file():
            prediction_source = candidate.resolve()
    representation_source = (
        Path(representation_example_path).resolve()
        if representation_example_path is not None
        else None
    )
    if representation_source is None:
        representation_candidate = source.parent / "target_representation_example.npz"
        if representation_candidate.is_file():
            representation_source = representation_candidate.resolve()
    bootstrap_source = (
        Path(bootstrap_evidence_path).resolve() if bootstrap_evidence_path is not None else None
    )
    if bootstrap_source is None:
        bootstrap_candidate = source.parent / "bootstrap_evidence.npz"
        if bootstrap_candidate.is_file():
            bootstrap_source = bootstrap_candidate.resolve()
    explicit_evidence_sources = {
        "rankings_path": rankings_path,
        "neighbour_evidence_path": neighbour_evidence_path,
        "dataset_evidence_path": dataset_evidence_path,
        "corruption_manifest_path": corruption_manifest_path,
    }
    if any(value is not None for value in explicit_evidence_sources.values()):
        resolved_evidence_sources: dict[str, Path | None] = {
            name: Path(value).resolve() if value is not None else None
            for name, value in explicit_evidence_sources.items()
        }
    else:
        candidate_names = {
            "rankings_path": "ranking.csv",
            "neighbour_evidence_path": "neighbour_evidence.npz",
            "dataset_evidence_path": "synthetic_dataset_evidence.npz",
            "corruption_manifest_path": "corruption_manifest.json",
        }
        candidates = {name: source.parent / filename for name, filename in candidate_names.items()}
        resolved_evidence_sources = (
            {name: path.resolve() for name, path in candidates.items()}
            if all(path.is_file() for path in candidates.values())
            else {name: None for name in candidate_names}
        )
    resolved_class_names = list(class_names) if class_names is not None else None
    if resolved_class_names is None:
        report_inputs_path = source.parent / "report_inputs.json"
        if report_inputs_path.is_file():
            try:
                report_inputs = json.loads(report_inputs_path.read_text(encoding="utf-8"))
                candidate_names = report_inputs.get("class_names")
                if isinstance(candidate_names, list) and all(
                    isinstance(value, str) for value in candidate_names
                ):
                    resolved_class_names = candidate_names
            except (json.JSONDecodeError, OSError):
                resolved_class_names = None
    duplicate_sources = {
        "duplicate_audit_path": duplicate_audit_path,
        "duplicate_candidates_csv_path": duplicate_candidates_csv_path,
        "duplicate_candidates_figure_path": duplicate_candidates_figure_path,
    }
    if any(value is not None for value in duplicate_sources.values()):
        if not all(value is not None for value in duplicate_sources.values()):
            missing = sorted(name for name, value in duplicate_sources.items() if value is None)
            raise ValueError(
                f"synthetic duplicate report requires all audit inputs; missing {missing}"
            )
        resolved_duplicate_sources = {
            name: Path(value).resolve()
            for name, value in duplicate_sources.items()
            if value is not None
        }
    else:
        duplicate_candidates = {
            "duplicate_audit_path": source.parent / "duplicate_audit.json",
            "duplicate_candidates_csv_path": source.parent / "duplicate_candidates.csv",
            "duplicate_candidates_figure_path": (
                source.parent / "figures" / "duplicate_candidates.png"
            ),
        }
        present = {name: path.is_file() for name, path in duplicate_candidates.items()}
        if any(present.values()) and not all(present.values()):
            raise ValueError("synthetic duplicate report artifacts are only partially present")
        resolved_duplicate_sources = (
            {name: path.resolve() for name, path in duplicate_candidates.items()}
            if all(present.values())
            else {}
        )
    duplicate_payload: dict[str, Any] | None = None
    if resolved_duplicate_sources:
        from histo_audit.reporting.synthetic_duplicates import load_synthetic_duplicate_audit

        duplicate_payload = load_synthetic_duplicate_audit(
            resolved_duplicate_sources["duplicate_audit_path"]
        )
    figure_set = (
        build_synthetic_figures(
            metrics,
            metrics_path=source,
            output_directory=destination / "figures",
            predictions_path=prediction_source,
            representation_example_path=representation_source,
            bootstrap_evidence_path=bootstrap_source,
            rankings_path=resolved_evidence_sources["rankings_path"],
            neighbour_evidence_path=resolved_evidence_sources["neighbour_evidence_path"],
            dataset_evidence_path=resolved_evidence_sources["dataset_evidence_path"],
            corruption_manifest_path=resolved_evidence_sources["corruption_manifest_path"],
            duplicate_audit_path=resolved_duplicate_sources.get("duplicate_audit_path"),
            duplicate_candidates_csv_path=resolved_duplicate_sources.get(
                "duplicate_candidates_csv_path"
            ),
            duplicate_candidates_figure_path=resolved_duplicate_sources.get(
                "duplicate_candidates_figure_path"
            ),
            class_names=resolved_class_names,
        )
        if generate_figures
        else None
    )
    figure_links: dict[str, tuple[str, str]] = {}
    if figure_set is not None:
        for figure in figure_set.figures:
            relative = os.path.relpath(figure.path, destination).replace("\\", "/")
            figure_links[figure.key] = (figure.alt_text, relative)
    metrics_link = os.path.relpath(source, destination).replace("\\", "/")
    duplicate_audit_link = (
        os.path.relpath(resolved_duplicate_sources["duplicate_audit_path"], destination).replace(
            "\\", "/"
        )
        if resolved_duplicate_sources
        else None
    )
    duplicate_csv_link = (
        os.path.relpath(
            resolved_duplicate_sources["duplicate_candidates_csv_path"], destination
        ).replace("\\", "/")
        if resolved_duplicate_sources
        else None
    )
    markdown = render_synthetic_markdown(
        metrics,
        metrics_path=source,
        metrics_sha256=digest,
        run_id=identifier,
        metrics_link=metrics_link,
        figure_links=figure_links,
        duplicate_audit=duplicate_payload,
        duplicate_audit_link=duplicate_audit_link,
        duplicate_audit_sha256=(
            sha256_file(resolved_duplicate_sources["duplicate_audit_path"])
            if resolved_duplicate_sources
            else None
        ),
        duplicate_candidates_csv_link=duplicate_csv_link,
    )
    static_html = markdown_to_static_html(
        markdown, title=f"Synthetic Histo Audit Report - {identifier}"
    )
    atomic_write_text(markdown_path, markdown)
    atomic_write_text(html_path, static_html)
    return ReportArtifacts(
        markdown_path=markdown_path,
        html_path=html_path,
        metrics_path=source,
        metrics_sha256=digest,
        figure_paths=(
            tuple(figure.path for figure in figure_set.figures) if figure_set is not None else ()
        ),
        figure_manifest_path=(figure_set.manifest_path if figure_set is not None else None),
    )


# General name used by the CLI; currently only synthetic sourced reports are valid.
build_report = build_synthetic_report


__all__ = [
    "ReportArtifacts",
    "build_report",
    "build_synthetic_report",
    "load_metrics",
    "markdown_to_static_html",
    "render_synthetic_markdown",
    "validate_metrics_payload",
]
