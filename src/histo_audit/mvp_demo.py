"""Build a small, evidence-backed presentation package from accepted AANCA artifacts."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from histo_audit.utils.run_tracking import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)

CANONICAL_PRIMARY_RUN_ID = "20260727T133947.089370Z_pannuke_primary_orphan_recovery"
DEFAULT_PRIMARY_RUN = Path("artifacts") / "runs" / CANONICAL_PRIMARY_RUN_ID
DEFAULT_QC_BUNDLE = Path("reports") / "pannuke_qc"
DEFAULT_OUTPUT = Path("artifacts") / "mvp_demo"

_SELECTED_RUN_FILES = (
    "completion_evidence.json",
    "metrics.json",
    "primary_subgroups.csv",
    "primary_recovery_evidence.json",
    "primary_recovery_statistics_verification.json",
    "primary_statistics.json",
    "primary_statistics_manifest.json",
    "reconciliation.json",
    "report.md",
    "restoration_index.json",
    "status.json",
)
_OUTPUT_FILES = ("README.md", "evidence.json", "index.html", "pannuke_mask_qc_overlays.png")
_H4_COMPARISON_ID = "audit_guided_minus_random_macro_f1"
_INSTANCE_DEPENDENT_SEEDS = (404, 405, 406)
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


@dataclass(frozen=True, slots=True)
class MvpPresentationArtifacts:
    """Paths and evidence root for one generated MVP presentation."""

    output_directory: Path
    html_path: Path
    evidence_path: Path
    overlay_path: Path
    readme_path: Path
    manifest_path: Path
    manifest_root_sha256: str


def create_mvp_http_server(
    output_directory: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> tuple[ThreadingHTTPServer, dict[str, Any]]:
    """Verify and bind a read-only local HTTP server for the presentation package."""

    if not host.strip() or host != host.strip():
        raise ValueError("MVP server host must be a non-empty value without outer whitespace")
    if not 0 <= port <= 65_535:
        raise ValueError("MVP server port must be between 0 and 65535")

    directory = Path(output_directory).resolve(strict=True)
    verification = verify_mvp_presentation(directory)
    handler = partial(_PresentationRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server, verification


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
    if not path.is_file():
        raise FileNotFoundError(f"required JSON file is missing: {path}")
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


def _require_exact_int(value: Any, *, role: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{role} must be an exact JSON integer")
    return value


def _require_bool(value: Any, *, role: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{role} must be an exact JSON boolean")
    return value


def _require_real(value: Any, *, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{role} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{role} must be finite")
    return result


def _require_interval(value: Any, *, role: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{role} must contain exactly two endpoints")
    interval = [
        _require_real(value[0], role=f"{role} lower"),
        _require_real(value[1], role=f"{role} upper"),
    ]
    if interval[0] > interval[1]:
        raise ValueError(f"{role} endpoints are reversed")
    return interval


def _summarise_hypothesis_comparisons(
    comparisons: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for hypothesis in ("h1", "h3", "h5", "h6", "h7"):
        selected = [
            item
            for item in comparisons
            if str(item.get("comparison_id", "")).startswith(f"{hypothesis}_")
        ]
        if not selected:
            raise ValueError(f"primary statistics contain no {hypothesis.upper()} comparisons")
        status_counts = Counter(str(item.get("status")) for item in selected)
        reported = [item for item in selected if item.get("status") == "reported"]
        differences = [
            _require_real(item.get("point_difference"), role=f"{hypothesis} point difference")
            for item in reported
        ]
        intervals = [
            _require_interval(item.get("interval_95"), role=f"{hypothesis} interval")
            for item in reported
        ]
        holm_values = [
            _require_real(item.get("p_value_holm"), role=f"{hypothesis} Holm p")
            for item in reported
        ]
        summaries[hypothesis] = {
            "comparison_count": len(selected),
            "reported_count": len(reported),
            "status_counts": dict(sorted(status_counts.items())),
            "positive_point_difference_count": sum(value > 0.0 for value in differences),
            "holm_one_sided_below_0_05_count": sum(value < 0.05 for value in holm_values),
            "interval_excludes_zero_in_positive_direction_count": sum(
                interval[0] > 0.0 for interval in intervals
            ),
            "interval_crosses_zero_count": sum(
                interval[0] <= 0.0 <= interval[1] for interval in intervals
            ),
            "point_difference_range": (
                [min(differences), max(differences)] if differences else None
            ),
        }
    return summaries


def _summarise_subgroups(path: Path, statistics: dict[str, Any]) -> dict[str, Any]:
    required_fields = {
        "cell_id",
        "scenario_id",
        "method",
        "dimension",
        "value",
        "sample_count",
        "injected_corruption_count",
        "average_precision_status",
        "average_precision",
        "suppression_reason",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != required_fields:
            raise ValueError("primary subgroup CSV schema differs")
        rows = list(reader)
    if not rows:
        raise ValueError("primary subgroup CSV is empty")

    dimensions: dict[str, dict[str, Any]] = {}
    for dimension in ("class", "tissue", "mechanism", "rate"):
        selected = [row for row in rows if row["dimension"] == dimension]
        if not selected:
            raise ValueError(f"primary subgroup CSV contains no {dimension} rows")
        status_counts = Counter(row["average_precision_status"] for row in selected)
        reported_values = [
            _require_real(float(row["average_precision"]), role=f"{dimension} subgroup AP")
            for row in selected
            if row["average_precision_status"] == "reported"
        ]
        dimensions[dimension] = {
            "row_count": len(selected),
            "reported_count": len(reported_values),
            "status_counts": dict(sorted(status_counts.items())),
            "reported_average_precision_range": (
                [min(reported_values), max(reported_values)] if reported_values else None
            ),
        }

    declared = statistics.get("subgroups")
    if not isinstance(declared, dict):
        raise ValueError("primary statistics subgroup metadata are absent")
    if _require_exact_int(declared.get("row_count"), role="subgroup row_count") != len(rows):
        raise ValueError("primary subgroup row count differs from statistics metadata")
    return {
        "source_path": "primary_subgroups.csv",
        "source_sha256": sha256_file(path),
        "row_count": len(rows),
        "reported_count": sum(row["average_precision_status"] == "reported" for row in rows),
        "interpretation_scope": "descriptive_heterogeneity_not_an_omnibus_causal_test",
        "dimensions": dimensions,
    }


def _extract_h4_restoration(
    run_path: Path,
    restoration_index: dict[str, Any],
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cells = restoration_index.get("cells")
    if (
        _require_exact_int(
            restoration_index.get("restoration_cell_count"), role="restoration cell count"
        )
        != 1
        or not isinstance(cells, list)
        or len(cells) != 1
        or not isinstance(cells[0], dict)
    ):
        raise ValueError("accepted primary restoration index differs")
    index_cell = cells[0]
    relative = index_cell.get("json_path")
    if not isinstance(relative, str) or relative not in records:
        raise ValueError("restoration JSON is absent from the sealed run manifest")
    source_path = _verify_manifest_file(run_path, relative, records[relative])
    if records[relative].get("sha256") != index_cell.get("json_sha256"):
        raise ValueError("restoration index and run manifest hashes disagree")
    restoration = _load_json(source_path)
    comparisons = restoration.get("downstream_comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("restoration comparison list is absent")
    matches = [
        item
        for item in comparisons
        if isinstance(item, dict) and item.get("comparison_id") == _H4_COMPARISON_ID
    ]
    if len(matches) != 1:
        raise ValueError("accepted primary must contain exactly one registered H4 comparison")
    comparison = matches[0]
    difference = _require_real(comparison.get("point_difference"), role="H4 difference")
    interval = _require_interval(comparison.get("interval_95"), role="H4 interval")
    point_a = _require_real(comparison.get("point_metric_a"), role="H4 audit-guided macro F1")
    point_b = _require_real(comparison.get("point_metric_b"), role="H4 random macro F1")
    probability = _require_real(
        comparison.get("probability_positive"), role="H4 probability positive"
    )
    evaluation = restoration.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("restoration evaluation is absent")

    def macro_f1(condition: str) -> float:
        value = evaluation.get(condition)
        if not isinstance(value, dict) or not isinstance(value.get("metrics"), dict):
            raise ValueError(f"restoration condition is absent: {condition}")
        return _require_real(value["metrics"].get("macro_f1"), role=f"{condition} macro F1")

    if interval[1] < 0.0:
        directional_result = "adverse_to_registered_hypothesis"
    elif interval[0] > 0.0:
        directional_result = "favourable_to_registered_hypothesis"
    else:
        directional_result = "inconclusive"
    return {
        "comparison_id": _H4_COMPARISON_ID,
        "status": comparison.get("status"),
        "cell": restoration.get("cell"),
        "metric": comparison.get("metric"),
        "audit_guided_macro_f1": point_a,
        "random_review_macro_f1_mean": point_b,
        "corrupted_observed_baseline_macro_f1": macro_f1("corrupted_observed_baseline"),
        "uncorrupted_reference_baseline_macro_f1": macro_f1("uncorrupted_reference_baseline"),
        "point_difference": difference,
        "interval_95": interval,
        "probability_positive": probability,
        "random_repetitions": _require_exact_int(
            comparison.get("random_repetitions"), role="H4 random repetitions"
        ),
        "directional_result": directional_result,
        "registered_hypothesis_supported": directional_result
        == "favourable_to_registered_hypothesis",
        "source_path": relative,
        "source_sha256": records[relative]["sha256"],
    }


def _extract_instance_seed_audit(
    run_path: Path,
    statistics: dict[str, Any],
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cells = statistics.get("cells")
    if not isinstance(cells, list):
        raise ValueError("primary statistics cell list is absent")
    selected: list[dict[str, Any]] = []
    for entry in cells:
        if not isinstance(entry, dict) or not isinstance(entry.get("cell"), dict):
            continue
        cell = entry["cell"]
        if (
            cell.get("mechanism") == "instance_dependent_corruption"
            and cell.get("rate") == 0.1
            and cell.get("representation_id") == "imagenet_resnet18_context"
            and cell.get("classifier_id") == "multinomial_logistic_regression"
            and cell.get("corruption_seed") in _INSTANCE_DEPENDENT_SEEDS
        ):
            selected.append(cell)
    if len(selected) != len(_INSTANCE_DEPENDENT_SEEDS):
        raise ValueError("registered instance-dependent seed cells are incomplete")

    seed_records: list[dict[str, Any]] = []
    for cell in sorted(selected, key=lambda item: int(item["corruption_seed"])):
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("instance-dependent seed cell has no identifier")
        ranking_relative = f"cells/{cell_id}/ranking.csv"
        oof_relative = f"cells/{cell_id}/oof_predictions.npz"
        if ranking_relative not in records or oof_relative not in records:
            raise ValueError("instance-dependent seed evidence is absent from the run manifest")
        _verify_manifest_file(run_path, ranking_relative, records[ranking_relative])
        _verify_manifest_file(run_path, oof_relative, records[oof_relative])
        seed_records.append(
            {
                "seed": _require_exact_int(cell.get("corruption_seed"), role="corruption seed"),
                "cell_id": cell_id,
                "ranking_sha256": records[ranking_relative]["sha256"],
                "oof_predictions_sha256": records[oof_relative]["sha256"],
            }
        )
    ranking_hashes = {str(item["ranking_sha256"]) for item in seed_records}
    oof_hashes = {str(item["oof_predictions_sha256"]) for item in seed_records}
    byte_identical = len(ranking_hashes) == 1 and len(oof_hashes) == 1
    return {
        "seeds": list(_INSTANCE_DEPENDENT_SEEDS),
        "records": seed_records,
        "ranking_outputs_byte_identical": len(ranking_hashes) == 1,
        "oof_outputs_byte_identical": len(oof_hashes) == 1,
        "independent_corruption_realisations": not byte_identical,
        "disclosure_required": byte_identical,
        "reporting_policy": (
            "retain_frozen_rows_but_treat_as_one_deterministic_realisation"
            if byte_identical
            else "report_distinct_saved_realisations"
        ),
    }


def _manifest_records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("run artifact manifest has no artifact list")
    records: dict[str, dict[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size_bytes"}:
            raise ValueError("run artifact manifest contains a malformed record")
        relative = item.get("path")
        if not isinstance(relative, str) or not relative or relative in records:
            raise ValueError("run artifact manifest contains a duplicate or invalid path")
        records[relative] = item
    if _require_exact_int(manifest.get("artifact_count"), role="artifact_count") != len(records):
        raise ValueError("run artifact manifest count differs")
    return records


def _verify_manifest_file(root: Path, relative: str, record: dict[str, Any]) -> Path:
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"selected presentation source is not a regular file: {relative}")
    size = _require_exact_int(record.get("size_bytes"), role=f"{relative} size")
    digest = record.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"selected presentation source has an invalid hash: {relative}")
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise ValueError(f"selected presentation source differs from its seal: {relative}")
    return path


def _validate_stage_ledger(run_path: Path, immutable: dict[str, Any]) -> dict[str, Any]:
    ledger = run_path.parent / "run_stage_attestations.jsonl"
    anchor_path = run_path.parent / "run_stage_attestations.anchor.json"
    anchor = _load_json(anchor_path)
    raw = ledger.read_bytes()
    if hashlib.sha256(raw).hexdigest() != anchor.get("ledger_sha256"):
        raise ValueError("stage-attestation ledger hash differs from its anchor")
    lines = raw.splitlines()
    if _require_exact_int(anchor.get("record_count"), role="stage record_count") != len(lines):
        raise ValueError("stage-attestation record count differs")
    previous: str | None = None
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        value = json.loads(
            line,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
        if not isinstance(value, dict):
            raise ValueError(f"stage-attestation line {index} is not an object")
        if value.get("previous_record_sha256") != previous:
            raise ValueError(f"stage-attestation line {index} breaks the chain")
        unsigned = {key: item for key, item in value.items() if key != "record_sha256"}
        if value.get("record_sha256") != _canonical_sha256(unsigned):
            raise ValueError(f"stage-attestation line {index} has an invalid record hash")
        previous = value["record_sha256"]
        records.append(value)
    if previous != anchor.get("head_record_sha256"):
        raise ValueError("stage-attestation head differs from its anchor")
    matching = [record for record in records if record.get("run_id") == immutable.get("run_id")]
    if len(matching) != 1:
        raise ValueError("accepted primary must have exactly one stage-attestation record")
    record = matching[0]
    if (
        record.get("event_type") != "postseal_stage_eligibility_attested"
        or record.get("completion_stage") != "PRIMARY_STUDY_COMPLETE"
        or _require_bool(record.get("scientific_stage_eligible"), role="scientific_stage_eligible")
        is not True
        or record.get("artifact_root_sha256") != immutable.get("artifact_root_sha256")
        or record.get("artifact_manifest_sha256") != immutable.get("artifact_manifest_sha256")
    ):
        raise ValueError("accepted primary stage attestation differs")
    return record


def _validate_primary_sources(run_path: Path) -> dict[str, Any]:
    immutable = _load_json(run_path / ".immutable.json")
    manifest_path = run_path / "artifact_manifest.json"
    manifest = _load_json(manifest_path)
    if (
        immutable.get("run_id") != run_path.name
        or immutable.get("status") != "completed"
        or _require_exact_int(immutable.get("artifact_count"), role="sealed artifact_count")
        != _require_exact_int(manifest.get("artifact_count"), role="manifest artifact_count")
        or manifest.get("run_id") != run_path.name
        or manifest.get("status") != "completed"
        or manifest.get("artifact_root_sha256") != immutable.get("artifact_root_sha256")
        or sha256_file(manifest_path) != immutable.get("artifact_manifest_sha256")
    ):
        raise ValueError("primary immutable seal and artifact manifest differ")
    records = _manifest_records(manifest)
    selected = {
        relative: _verify_manifest_file(run_path, relative, records[relative])
        for relative in _SELECTED_RUN_FILES
        if relative in records
    }
    if set(selected) != set(_SELECTED_RUN_FILES):
        missing = sorted(set(_SELECTED_RUN_FILES) - set(selected))
        raise ValueError(f"primary manifest lacks presentation sources: {missing}")

    completion = _load_json(selected["completion_evidence.json"])
    metrics = _load_json(selected["metrics.json"])
    recovery = _load_json(selected["primary_recovery_evidence.json"])
    statistics_verification = _load_json(selected["primary_recovery_statistics_verification.json"])
    statistics_manifest = _load_json(selected["primary_statistics_manifest.json"])
    statistics = _load_json(selected["primary_statistics.json"])
    restoration_index = _load_json(selected["restoration_index.json"])
    status = _load_json(selected["status.json"])
    reconciliation = _load_json(selected["reconciliation.json"])

    if (
        completion.get("completion_stage") != "PRIMARY_STUDY_COMPLETE"
        or completion.get("analysis_disposition") != "amended_or_exploratory"
        or _require_exact_int(
            completion.get("completed_required_cell_count"), role="completed cell count"
        )
        != 185
        or _require_exact_int(completion.get("failed_required_cell_count"), role="failed cells")
        != 0
        or _require_exact_int(completion.get("skipped_optional_cell_count"), role="optional skips")
        != 37
        or _require_bool(completion.get("training_invoked"), role="training_invoked") is not False
        or _require_bool(completion.get("fallback_invoked"), role="fallback_invoked") is not False
        or _require_bool(completion.get("automatic_retry_allowed"), role="automatic_retry_allowed")
        is not False
        or completion.get("primary_statistics_verification_status") != "passed"
        or completion.get("primary_restoration_verification_status") != "passed"
        or metrics.get("completion_stage") != "PRIMARY_STUDY_COMPLETE"
        or metrics.get("analysis_disposition") != "amended_or_exploratory"
        or status.get("status") != "completed"
        or reconciliation.get("status") != "passed"
        or recovery.get("retrained_cell_count") != 0
        or statistics_verification.get("verification", {}).get("status") != "passed"
    ):
        raise ValueError("accepted primary presentation invariants differ")

    comparison_count = _require_exact_int(
        completion.get("primary_statistics_comparison_count"), role="comparison count"
    )
    comparisons = statistics.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != comparison_count:
        raise ValueError("primary comparison list differs from completion evidence")
    if comparison_count != 36:
        raise ValueError("MVP expects the accepted 36-comparison primary artifact")
    summary_comparisons: list[dict[str, Any]] = []
    fields = (
        "comparison_id",
        "status",
        "method_a",
        "method_b",
        "metric",
        "point_difference",
        "interval_95",
        "p_value_holm",
        "valid_bootstrap_iterations",
    )
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise ValueError("primary comparison entry is not an object")
        identifier = comparison.get("comparison_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("primary comparison identifier is invalid")
        summary_comparisons.append({field: comparison.get(field) for field in fields})
    summary_comparisons.sort(key=lambda item: item["comparison_id"])
    hypothesis_comparisons = _summarise_hypothesis_comparisons(comparisons)

    correction = statistics.get("multiple_comparison_correction")
    if (
        not isinstance(correction, dict)
        or correction.get("method") != "holm"
        or not isinstance(correction.get("one_sided_p_value_definition"), str)
    ):
        raise ValueError("primary multiple-comparison definition differs")
    subgroup_summary = _summarise_subgroups(selected["primary_subgroups.csv"], statistics)
    h4_restoration = _extract_h4_restoration(run_path, restoration_index, records)
    instance_seed_audit = _extract_instance_seed_audit(run_path, statistics, records)

    statistics_records = statistics_manifest.get("artifacts")
    if not isinstance(statistics_records, list):
        raise ValueError("primary statistics manifest is malformed")
    statistics_by_path = {
        item.get("path"): item
        for item in statistics_records
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    source_statistics_record = statistics_by_path.get("primary_statistics.json")
    if source_statistics_record != records["primary_statistics.json"]:
        raise ValueError("primary statistics manifests disagree")
    source_subgroups_record = statistics_by_path.get("primary_subgroups.csv")
    if source_subgroups_record != records["primary_subgroups.csv"]:
        raise ValueError("primary subgroup manifests disagree")

    stage = _validate_stage_ledger(run_path, immutable)
    return {
        "run_id": run_path.name,
        "artifact_count": immutable["artifact_count"],
        "artifact_root_sha256": immutable["artifact_root_sha256"],
        "artifact_manifest_sha256": immutable["artifact_manifest_sha256"],
        "stage_attestation_record_sha256": stage["record_sha256"],
        "stage_attestation_verification_sha256": stage["verification_sha256"],
        "completion_evidence_sha256": records["completion_evidence.json"]["sha256"],
        "statistics_sha256": records["primary_statistics.json"]["sha256"],
        "analysis_disposition": completion["analysis_disposition"],
        "outcomes_inspected": completion["outcomes_inspected"],
        "completed_required_cells": completion["completed_required_cell_count"],
        "failed_required_cells": completion["failed_required_cell_count"],
        "skipped_optional_cells": completion["skipped_optional_cell_count"],
        "comparison_count": comparison_count,
        "bootstrap_iterations_required": 2000,
        "inference": {
            "multiple_comparison_method": correction["method"],
            "p_value_sidedness": "one_sided",
            "p_value_definition": correction["one_sided_p_value_definition"],
            "interval_label": "95_percentile_bootstrap_interval",
        },
        "comparisons": summary_comparisons,
        "hypothesis_comparisons": hypothesis_comparisons,
        "h2_subgroups": subgroup_summary,
        "h4_restoration": h4_restoration,
        "instance_dependent_seed_audit": instance_seed_audit,
    }


def _validate_qc_sources(qc_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    manifest = _load_json(qc_path / "artifact_manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("PanNuke QC artifact manifest is malformed")
    for relative in ("pannuke_mask_qc.json", "pannuke_mask_qc_overlays.png"):
        record = files.get(relative)
        if not isinstance(record, dict):
            raise ValueError(f"PanNuke QC manifest lacks {relative}")
        _verify_manifest_file(qc_path, relative, record)
    qc = _load_json(qc_path / "pannuke_mask_qc.json")
    policy = qc.get("qc_policy")
    totals = qc.get("global_mask_qc")
    if not isinstance(policy, dict) or not isinstance(totals, dict):
        raise ValueError("PanNuke QC summary fields are absent")
    if (
        _require_bool(qc.get("source_masks_modified"), role="source_masks_modified") is not False
        or _require_bool(policy.get("no_class_arbitration"), role="no_class_arbitration")
        is not True
        or _require_bool(
            policy.get("supplied_background_is_exact_complement_required"),
            role="background complement policy",
        )
        is not False
    ):
        raise ValueError("PanNuke QC safety policy differs")
    required_counts = (
        "fold_count",
        "patch_count",
        "cross_class_overlap_pixel_count",
        "cross_class_overlap_patch_count",
        "void_pixel_count",
        "void_patch_count",
        "overlap_touching_instance_count",
    )
    summary: dict[str, Any] = {
        key: _require_exact_int(totals.get(key), role=f"QC {key}") for key in required_counts
    }
    summary["source_masks_modified"] = False
    summary["no_class_arbitration"] = True
    summary["selection_sha256"] = qc.get("selection_sha256")
    summary["overlay_sha256"] = manifest.get("overlay_sha256")
    return summary, qc_path / "pannuke_mask_qc_overlays.png", manifest


def _format_number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return " - ".join(_format_number(item) for item in value)
    return str(value)


def _comparison_hypothesis(comparison_id: str) -> str:
    candidate = comparison_id.split("_", 1)[0].upper()
    return candidate if candidate in {"H1", "H3", "H5", "H6", "H7"} else "OTHER"


def _comparison_seed(comparison_id: str) -> str:
    marker = "_seed_"
    if marker not in comparison_id:
        return ""
    return comparison_id.rsplit(marker, 1)[1].split("_", 1)[0]


def _position_percent(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 50.0
    return 100.0 * (value - lower) / (upper - lower)


def _render_comparison_rows(comparisons: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for item in comparisons:
        comparison_id = str(item["comparison_id"])
        cells = (
            comparison_id,
            item["status"],
            item["method_a"],
            item["method_b"],
            item["point_difference"],
            item["interval_95"],
            item["p_value_holm"],
            item["valid_bootstrap_iterations"],
        )
        rows.append(
            '<tr data-comparison-row data-hypothesis="'
            + html.escape(_comparison_hypothesis(comparison_id))
            + '" data-status="'
            + html.escape(str(item["status"]))
            + '" data-seed="'
            + html.escape(_comparison_seed(comparison_id))
            + '">'
            + "".join(f"<td>{html.escape(_format_number(value))}</td>" for value in cells)
            + "</tr>"
        )
    return "".join(rows)


def _render_forest_plot(comparisons: list[dict[str, Any]]) -> str:
    bounds = [0.0]
    for item in comparisons:
        if item.get("status") != "reported":
            continue
        bounds.append(_require_real(item.get("point_difference"), role="forest point difference"))
        bounds.extend(_require_interval(item.get("interval_95"), role="forest interval"))
    lower = min(bounds)
    upper = max(bounds)
    padding = max((upper - lower) * 0.07, 0.001)
    lower -= padding
    upper += padding
    zero_position = _position_percent(0.0, lower, upper)

    output: list[str] = [
        '<div class="forest-axis" aria-hidden="true"><span>',
        html.escape(f"{lower:.3f}"),
        "</span><span>różnica AP · zero = brak różnicy</span><span>",
        html.escape(f"{upper:.3f}"),
        "</span></div>",
    ]
    previous_group = ""
    for item in comparisons:
        comparison_id = str(item["comparison_id"])
        hypothesis = _comparison_hypothesis(comparison_id)
        if hypothesis != previous_group:
            output.append(
                f'<div class="forest-group"><span>{html.escape(hypothesis)}</span>'
                "<span>punkt i zapisany 95% przedział bootstrap</span></div>"
            )
            previous_group = hypothesis
        label = comparison_id.replace(f"{hypothesis.lower()}_", "", 1).replace("_", " ")
        if item.get("status") != "reported":
            output.append(
                '<div class="forest-row is-unavailable">'
                f'<span class="forest-label" title="{html.escape(comparison_id)}">'
                f"{html.escape(label)}</span>"
                '<span class="forest-track forest-track-empty" aria-label="Wynik niedostępny">'
                '<span>niedostępne</span></span><span class="forest-value">—</span></div>'
            )
            continue
        point = _require_real(item.get("point_difference"), role="forest point difference")
        interval = _require_interval(item.get("interval_95"), role="forest interval")
        point_position = _position_percent(point, lower, upper)
        low_position = _position_percent(interval[0], lower, upper)
        high_position = _position_percent(interval[1], lower, upper)
        p_value = _require_real(item.get("p_value_holm"), role="forest Holm p")
        aria = (
            f"{comparison_id}: różnica {point:.8g}; 95% CI {interval[0]:.8g} do "
            f"{interval[1]:.8g}; jednostronne Holm p {p_value:.8g}"
        )
        output.append(
            '<div class="forest-row">'
            f'<span class="forest-label" title="{html.escape(comparison_id)}">'
            f"{html.escape(label)}</span>"
            f'<span class="forest-track" style="--zero:{zero_position:.4f}%;'
            f"--low:{low_position:.4f}%;--high:{high_position:.4f}%;"
            f'--point:{point_position:.4f}%" role="img" aria-label="{html.escape(aria)}">'
            '<span class="forest-zero"></span><span class="forest-ci"></span>'
            '<span class="forest-point"></span></span>'
            f'<span class="forest-value">{html.escape(_format_number(point))}</span></div>'
        )
    return "".join(output)


def _render_h4_chart(h4: dict[str, Any]) -> str:
    metrics = (
        (
            "Uncorrupted reference",
            _require_real(
                h4["uncorrupted_reference_baseline_macro_f1"],
                role="H4 uncorrupted macro-F1",
            ),
        ),
        (
            "Corrupted observed",
            _require_real(
                h4["corrupted_observed_baseline_macro_f1"],
                role="H4 corrupted macro-F1",
            ),
        ),
        (
            "Random review · mean",
            _require_real(h4["random_review_macro_f1_mean"], role="H4 random macro-F1"),
        ),
        (
            "Audit-guided",
            _require_real(h4["audit_guided_macro_f1"], role="H4 audit macro-F1"),
        ),
    )
    values = [value for _, value in metrics]
    metric_padding = max((max(values) - min(values)) * 0.08, 0.0005)
    metric_lower = min(values) - metric_padding
    metric_upper = max(values) + metric_padding
    metric_rows = []
    for label, value in metrics:
        position = _position_percent(value, metric_lower, metric_upper)
        metric_rows.append(
            '<div class="metric-row"><span class="metric-name">'
            f"{html.escape(label)}</span>"
            f'<span class="metric-track" role="img" aria-label="{html.escape(label)}: '
            f'{value:.12g}"><span class="metric-dot" style="--x:{position:.4f}%">'
            "</span></span>"
            f"<strong>{value:.6f}</strong></div>"
        )

    difference = _require_real(h4["point_difference"], role="H4 point difference")
    interval = _require_interval(h4["interval_95"], role="H4 interval")
    diff_lower = min(interval[0], difference, 0.0)
    diff_upper = max(interval[1], difference, 0.0)
    diff_padding = max((diff_upper - diff_lower) * 0.12, 0.0002)
    diff_lower -= diff_padding
    diff_upper += diff_padding
    point_position = _position_percent(difference, diff_lower, diff_upper)
    low_position = _position_percent(interval[0], diff_lower, diff_upper)
    high_position = _position_percent(interval[1], diff_lower, diff_upper)
    zero_position = _position_percent(0.0, diff_lower, diff_upper)
    repetitions = _require_exact_int(h4["random_repetitions"], role="H4 repetitions")
    return (
        '<div class="h4-chart" aria-label="Pełny wynik downstream H4">'
        '<div class="metric-axis"><span>'
        + f"{metric_lower:.3f}"
        + "</span><span>macro-F1 · oś zawężona</span><span>"
        + f"{metric_upper:.3f}"
        + "</span></div>"
        + "".join(metric_rows)
        + '<div class="difference-panel"><div class="difference-heading">'
        + "<span>Audit-guided minus random</span>"
        + f"<strong>{difference:.6f}</strong></div>"
        + f'<div class="difference-track" style="--zero:{zero_position:.4f}%;'
        + f"--low:{low_position:.4f}%;--high:{high_position:.4f}%;"
        + f'--point:{point_position:.4f}%" role="img" aria-label="Różnica '
        + f'{difference:.12g}; 95% CI {interval[0]:.12g} do {interval[1]:.12g}">'
        + '<span class="difference-zero"></span><span class="difference-ci"></span>'
        + '<span class="difference-point"></span></div>'
        + f'<div class="difference-axis"><span>{diff_lower:.4f}</span><span>0</span>'
        + f"<span>{diff_upper:.4f}</span></div>"
        + f"<p>95% CI: <strong>[{interval[0]:.6f}; {interval[1]:.6f}]</strong> · "
        + f"{repetitions} random-review repetitions</p></div></div>"
    )


def _render_h2_chart(h2: dict[str, Any]) -> str:
    labels = {
        "class": "Klasa jądra",
        "tissue": "Typ tkanki",
        "mechanism": "Mechanizm korupcji",
        "rate": "Poziom korupcji",
    }
    rows: list[str] = []
    for dimension, label in labels.items():
        summary = h2["dimensions"][dimension]
        interval = _require_interval(
            summary["reported_average_precision_range"], role=f"H2 {dimension} AP range"
        )
        low_position = 100.0 * interval[0]
        high_position = 100.0 * interval[1]
        not_applicable = int(summary["status_counts"].get("not_applicable_zero_corruption", 0))
        aria = (
            f"{label}: {summary['reported_count']} raportowalnych estymat, zakres AP "
            f"{interval[0]:.6g} do {interval[1]:.6g}, {not_applicable} nie dotyczy"
        )
        rows.append(
            '<div class="range-row"><span class="range-name">'
            f"{html.escape(label)}</span>"
            f'<span class="range-track" style="--low:{low_position:.4f}%;'
            f'--high:{high_position:.4f}%" role="img" aria-label="{html.escape(aria)}">'
            '<span class="range-line"></span><span class="range-start"></span>'
            '<span class="range-end"></span></span>'
            f'<span class="range-count"><strong>{summary["reported_count"]:,}</strong>'
            "<small>reported</small></span>"
            f'<span class="range-values">{interval[0]:.4f}-{interval[1]:.4f}</span></div>'
        )
    return (
        '<div class="h2-chart"><div class="range-axis"><span>0 AP</span>'
        "<span>opisowy zakres zapisanych estymat</span><span>1 AP</span></div>"
        + "".join(rows)
        + f'<p class="chart-note">Łącznie <strong>{h2["reported_count"]:,}</strong> '
        + f"raportowalnych estymat z <strong>{h2['row_count']:,}</strong> zapisanych "
        + "wierszy. Zakres nie jest testem omnibus ani rankingiem biologicznym.</p></div>"
    )


def _render_hypothesis_ledger(
    hypotheses: dict[str, dict[str, Any]], h2: dict[str, Any], h4: dict[str, Any]
) -> str:
    h1, h3, h5, h6, h7 = (
        hypotheses["h1"],
        hypotheses["h3"],
        hypotheses["h5"],
        hypotheses["h6"],
        hypotheses["h7"],
    )
    entries = (
        (
            "H1",
            "Ranking przewyższa losowy przegląd",
            f"{h1['positive_point_difference_count']}/{h1['reported_count']} dodatnich różnic",
            "positive",
            f"Wszystkie {h1['interval_excludes_zero_in_positive_direction_count']} zapisane "
            "95% przedziały leżą powyżej zera.",
        ),
        (
            "H2",
            "Heterogeniczność podgrup",
            f"{h2['reported_count']:,} raportowalnych estymat",
            "descriptive",
            "Wynik opisowy dla klas, tkanek, mechanizmów i poziomów korupcji; "
            "bez testu omnibus i bez interpretacji przyczynowej.",
        ),
        (
            "H3",
            "Mechanizmy mają różną trudność",
            f"{h3['positive_point_difference_count']}/{h3['reported_count']} dodatnich różnic",
            "qualified",
            f"Jedna z {h3['reported_count']} zapisanych 95% CI przecina zero, mimo "
            "jednostronnego Holm p poniżej 0,05.",
        ),
        (
            "H4",
            "Restoration nie uzyskało przewagi",
            f"Δ macro-F1 {_format_number(h4['point_difference'])}",
            "adverse",
            "Wynik jest przeciwny do zarejestrowanej hipotezy i pozostaje pokazany "
            "bez strojenia po poznaniu rezultatu.",
        ),
        (
            "H5",
            "Fixed hybrid uzyskał wyższy AP",
            f"{h5['positive_point_difference_count']}/{h5['reported_count']} dodatnich różnic",
            "positive",
            f"Wszystkie {h5['interval_excludes_zero_in_positive_direction_count']} zapisane "
            "95% przedziały leżą powyżej zera.",
        ),
        (
            "H6",
            "Enkoder patologiczny niedostępny",
            f"{h6['reported_count']}/{h6['comparison_count']} wykonanych porównań",
            "unavailable",
            "Nie wybrano zastępczego enkodera po poznaniu wyników; trzy zamrożone "
            "komórki opcjonalne pozostają jawnie niedostępne.",
        ),
        (
            "H7",
            "Brak dowodu przewagi highlighting",
            f"{h7['interval_crosses_zero_count']}/{h7['reported_count']} CI przecinają zero",
            "neutral",
            f"Jednostronne Holm p poniżej 0,05: "
            f"{h7['holm_one_sided_below_0_05_count']}/{h7['reported_count']}.",
        ),
    )
    return "".join(
        '<details class="hypothesis-row" data-status="'
        + status
        + '"><summary><span class="hypothesis-id">'
        + identifier
        + '</span><span class="hypothesis-title">'
        + html.escape(title)
        + '</span><span class="hypothesis-result">'
        + html.escape(result)
        + '</span><span class="hypothesis-toggle" aria-hidden="true">+</span></summary><p>'
        + html.escape(detail)
        + "</p></details>"
        for identifier, title, result, status, detail in entries
    )


def _render_html_legacy(evidence: dict[str, Any]) -> str:
    primary = evidence["primary"]
    qc = evidence["pannuke_qc"]
    hypotheses = primary["hypothesis_comparisons"]
    h2 = primary["h2_subgroups"]
    h4 = primary["h4_restoration"]
    seed_audit = primary["instance_dependent_seed_audit"]
    rows = []
    for item in primary["comparisons"]:
        cells = (
            item["comparison_id"],
            item["status"],
            item["method_a"],
            item["method_b"],
            item["point_difference"],
            item["interval_95"],
            item["p_value_holm"],
            item["valid_bootstrap_iterations"],
        )
        rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(_format_number(value))}</td>" for value in cells)
            + "</tr>"
        )

    subgroup_rows = []
    dimension_labels = {
        "class": "klasa jądra",
        "tissue": "typ tkanki",
        "mechanism": "mechanizm korupcji",
        "rate": "poziom korupcji",
    }
    for dimension, label in dimension_labels.items():
        summary = h2["dimensions"][dimension]
        subgroup_rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{summary['reported_count']:,}</td>"
            f"<td>{html.escape(_format_number(summary['reported_average_precision_range']))}</td>"
            "</tr>"
        )

    seed_rows = []
    for record in seed_audit["records"]:
        ranking_hash = str(record["ranking_sha256"])
        oof_hash = str(record["oof_predictions_sha256"])
        seed_rows.append(
            "<tr>"
            f"<td>{record['seed']}</td>"
            f"<td><code>{html.escape(str(record['cell_id']))}</code></td>"
            f'<td><code title="{ranking_hash}">{ranking_hash[:16]}…</code></td>'
            f'<td><code title="{oof_hash}">{oof_hash[:16]}…</code></td>'
            "</tr>"
        )

    h1 = hypotheses["h1"]
    h3 = hypotheses["h3"]
    h5 = hypotheses["h5"]
    h6 = hypotheses["h6"]
    h7 = hypotheses["h7"]
    return f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="AANCA: uczciwe podsumowanie kontrolowanego audytu adnotacji jąder PanNuke.">
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%2312324a'/%3E%3Cpath d='M14 48 29 14h7l14 34h-9l-3-8H26l-3 8zm15-16h7l-3-9z' fill='white'/%3E%3C/svg%3E">
  <title>AANCA — wyniki i ograniczenia</title>
  <style>
    :root {{ color-scheme:light; --ink:#17232d; --muted:#5b6874; --navy:#12324a;
      --teal:#11756f; --amber:#a96512; --red:#a63b32; --paper:#f3f6f7; --line:#d9e1e5; }}
    * {{ box-sizing:border-box; }} html {{ scroll-behavior:smooth; }}
    body {{ margin:0; font:16px/1.58 Inter,ui-sans-serif,system-ui,sans-serif; color:var(--ink); background:var(--paper); }}
    a {{ color:#075f71; }} a:focus-visible,summary:focus-visible {{ outline:3px solid #f0a43c; outline-offset:3px; }}
    .skip {{ position:absolute; left:-9999px; }} .skip:focus {{ left:16px; top:12px; z-index:10; background:white; padding:10px; }}
    header {{ padding:62px max(24px,8vw) 52px; color:white; background:linear-gradient(130deg,var(--navy),#145f70); }}
    header p {{ max-width:820px; font-size:1.12rem; }} nav {{ display:flex; gap:16px; flex-wrap:wrap; margin-top:24px; }}
    nav a {{ color:#e9fbff; font-weight:750; text-decoration:none; }} main {{ max-width:1180px; margin:auto; padding:30px 24px 72px; }}
    section {{ scroll-margin-top:20px; }} h1 {{ font-size:clamp(2.4rem,6vw,4.8rem); letter-spacing:-.04em; margin:8px 0 10px; }}
    h2 {{ margin:44px 0 12px; font-size:clamp(1.45rem,3vw,2.1rem); }} h3 {{ margin:.1rem 0 .45rem; }}
    .eyebrow {{ text-transform:uppercase; letter-spacing:.12em; font-weight:800; font-size:.78rem; opacity:.86; }}
    .badge {{ display:inline-block; padding:6px 10px; margin:0 6px 6px 0; border-radius:999px; background:#dff4f1; color:#07554f; font-weight:800; }}
    .badge.scope {{ background:#fbe7c8; color:#6f4007; }}
    .warning,.result-banner,.seed-warning {{ border-left:5px solid var(--amber); padding:16px 19px; background:#fff6e7; border-radius:0 10px 10px 0; }}
    .result-banner {{ border-color:var(--red); background:#fff0ee; }} .seed-warning {{ border-color:#7956a8; background:#f6f0ff; }}
    .cards,.hypotheses {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; margin:22px 0; }}
    .hypotheses {{ grid-template-columns:repeat(auto-fit,minmax(270px,1fr)); }}
    .card {{ background:white; border:1px solid var(--line); border-radius:14px; padding:18px; box-shadow:0 4px 18px #12324a0d; }}
    .hypothesis {{ border-top:5px solid var(--teal); }} .hypothesis.adverse {{ border-color:var(--red); }}
    .hypothesis.neutral {{ border-color:var(--amber); }} .hypothesis.unavailable {{ border-color:#87939d; }}
    .status {{ font-size:.76rem; text-transform:uppercase; letter-spacing:.07em; font-weight:850; color:var(--muted); }}
    .value {{ font-size:1.8rem; font-weight:800; color:var(--navy); }} .label,.muted {{ color:var(--muted); }} .fine {{ font-size:.88rem; }}
    .table-wrap {{ overflow:auto; background:white; border:1px solid var(--line); border-radius:12px; }}
    table {{ border-collapse:collapse; width:100%; min-width:950px; font-size:.86rem; }} table.compact {{ min-width:620px; }}
    th,td {{ padding:9px 11px; text-align:left; border-bottom:1px solid #e8ecef; }} th {{ position:sticky; top:0; background:#edf2f4; }}
    code {{ overflow-wrap:anywhere; }} details {{ margin:16px 0; border:1px solid var(--line); border-radius:12px; background:white; padding:14px 16px; }}
    summary {{ cursor:pointer; font-weight:800; color:var(--navy); }} details[open] summary {{ margin-bottom:14px; }}
    img {{ width:100%; height:auto; border-radius:12px; border:1px solid var(--line); background:white; }}
    .method-flow {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:20px 0; }}
    .method-flow div {{ background:#e7f2f3; padding:14px; border-radius:10px; text-align:center; font-weight:750; }}
    footer {{ color:var(--muted); margin-top:48px; font-size:.9rem; }}
    @media (max-width:760px) {{ .method-flow {{ grid-template-columns:1fr; }} header {{ padding-top:44px; }} }}
    @media print {{ body {{ background:white; }} nav,.skip {{ display:none; }} }}
  </style>
</head>
<body>
<a class="skip" href="#main">Przejdź do treści</a>
<header>
  <div class="eyebrow">Automated Nucleus-Annotation Auditing</div><h1>AANCA</h1>
  <p>Reprodukowalny prototyp badawczy do priorytetyzacji przypadków oznaczonych jako
  <em>potentially inconsistent annotation</em> i <em>recommended for expert review</em>.</p>
  <span class="badge">DEMO_COMPLETE</span><span class="badge">PRIMARY_STUDY_COMPLETE</span><span class="badge scope">amended_or_exploratory</span>
  <nav aria-label="Sekcje"><a href="#wyniki">Wyniki H1-H7</a><a href="#metoda">Metoda</a><a href="#qc">QC</a><a href="#dowody">Dowody</a></nav>
</header>
<main id="main">
  <div class="warning"><strong>Granica wniosku:</strong> to nie jest system diagnostyczny,
  nie dowodzi błędu patologa i nigdy automatycznie nie zmienia adnotacji źródłowych.
  Wyniki dotyczą kontrolowanych zmian etykiet; confirmatory, walidacja ekspercka i zewnętrzna nie zostały wykonane.</div>

  <section id="wyniki"><h2>Najważniejszy wynik</h2>
  <div class="result-banner"><strong>Ranking odnajdywał wstrzyknięte zmiany, lecz przewaga nie przełożyła się na downstream macro-F1.</strong>
  H4 uzyskała wynik przeciwny do zarejestrowanej hipotezy: audit-guided minus random =
  <strong>{_format_number(h4["point_difference"])}</strong>, 95% CI <strong>{html.escape(_format_number(h4["interval_95"]))}</strong>.
  Wyniku nie ukryto i nie przeliczano po jego poznaniu.</div>

  <h2>Wyniki H1-H7</h2>
  <p class="muted">Analiza ma dyspozycję <code>amended_or_exploratory</code>. „Istotne” oznacza jednostronne p skorygowane metodą Holm.</p>
  <div class="hypotheses">
    <article class="card hypothesis"><div class="status">H1 • wzorzec dodatni</div><h3>Lepsze niż losowy przegląd</h3><p>{h1["positive_point_difference_count"]}/{h1["reported_count"]} różnic dodatnich; {h1["interval_excludes_zero_in_positive_direction_count"]}/{h1["reported_count"]} zapisanych 95% CI powyżej zera.</p></article>
    <article class="card hypothesis"><div class="status">H2 • opis heterogeniczności</div><h3>Wyniki zależą od podgrupy</h3><p>{h2["reported_count"]:,} raportowalnych estymat AP dla klas, tkanek, mechanizmów i poziomów korupcji; bez testu omnibus.</p></article>
    <article class="card hypothesis neutral"><div class="status">H3 • dodatni z zastrzeżeniem</div><h3>Mechanizmy mają różną trudność</h3><p>{h3["positive_point_difference_count"]}/{h3["reported_count"]} różnic dodatnich; {h3["interval_crosses_zero_count"]} zapisany 95% CI przecina zero.</p></article>
    <article class="card hypothesis adverse"><div class="status">H4 • wynik niekorzystny</div><h3>Brak przewagi restoration</h3><p>Audit-guided {_format_number(h4["audit_guided_macro_f1"])} vs random {_format_number(h4["random_review_macro_f1_mean"])} macro-F1.</p></article>
    <article class="card hypothesis"><div class="status">H5 • wzorzec dodatni</div><h3>Fixed hybrid wyżej</h3><p>{h5["positive_point_difference_count"]}/{h5["reported_count"]} różnic dodatnich; {h5["interval_excludes_zero_in_positive_direction_count"]}/{h5["reported_count"]} zapisanych 95% CI powyżej zera.</p></article>
    <article class="card hypothesis unavailable"><div class="status">H6 • niedostępne</div><h3>Brak enkodera patologicznego</h3><p>{h6["reported_count"]}/{h6["comparison_count"]} porównań wykonanych; nie wybrano zastępstwa po wynikach.</p></article>
    <article class="card hypothesis neutral"><div class="status">H7 • brak dowodu przewagi</div><h3>Highlighting bez rozstrzygnięcia</h3><p>{h7["holm_one_sided_below_0_05_count"]}/{h7["reported_count"]} jednostronnych Holm p &lt; 0,05; wszystkie {h7["interval_crosses_zero_count"]} zapisane 95% CI przecinają zero.</p></article>
  </div>

  <h2>H4: pełny wynik downstream</h2>
  <div class="table-wrap"><table class="compact"><thead><tr><th>Warunek</th><th>Macro-F1</th><th>Interpretacja</th></tr></thead><tbody>
    <tr><td>Uncorrupted reference baseline</td><td>{_format_number(h4["uncorrupted_reference_baseline_macro_f1"])}</td><td>punkt odniesienia bez korupcji projektu</td></tr>
    <tr><td>Corrupted observed baseline</td><td>{_format_number(h4["corrupted_observed_baseline_macro_f1"])}</td><td>bez symulowanego review</td></tr>
    <tr><td>Random review restoration</td><td>{_format_number(h4["random_review_macro_f1_mean"])}</td><td>średnia ze {h4["random_repetitions"]} powtórzeń</td></tr>
    <tr><td>Audit-guided restoration</td><td>{_format_number(h4["audit_guided_macro_f1"])}</td><td>niżej niż random w tym eksperymencie</td></tr>
  </tbody></table></div>

  <h2>Ważne ujawnienie: seedy instance-dependent</h2>
  <div class="seed-warning"><strong>Seedy 404, 405 i 406 nie są niezależnymi realizacjami.</strong>
  Zapisane rankingi i predykcje OOF są bajtowo identyczne. Trzy zamrożone wiersze pozostają dla kompletności,
  lecz są interpretowane jako jeden deterministyczny scenariusz, nie trzy replikacje.</div>
  <details><summary>Dowód hashy dla trzech seedów</summary><div class="table-wrap"><table class="compact">
    <thead><tr><th>Seed</th><th>Cell ID</th><th>Ranking SHA-256</th><th>OOF SHA-256</th></tr></thead><tbody>{"".join(seed_rows)}</tbody>
  </table></div><p class="fine muted">Pełne hashe są zapisane w <code>evidence.json</code>.</p></details>

  <h2>H2: kompletność analizy podgrup</h2>
  <p>Zakresy obejmują wszystkie raportowalne estymaty AP przez wiele komórek i metod. Pokazują heterogeniczność,
  lecz nie są rankingiem biologicznym ani testem przyczynowym pomiędzy kategoriami.</p>
  <div class="table-wrap"><table class="compact"><thead><tr><th>Wymiar</th><th>Raportowalne estymaty</th><th>Zakres AP</th></tr></thead><tbody>{"".join(subgroup_rows)}</tbody></table></div></section>

  <section id="metoda"><h2>Jak działa kontrolowany benchmark</h2>
  <div class="method-flow"><div>1. Niezmienione dane źródłowe</div><div>2. Kontrolowana zmiana etykiet</div><div>3. Group-safe OOF ranking</div><div>4. Stały budżet symulowanego review</div></div>
  <div class="cards">
    <div class="card"><div class="value">{primary["completed_required_cells"]}/185</div><div class="label">wymaganych komórek primary</div></div>
    <div class="card"><div class="value">{primary["failed_required_cells"]}</div><div class="label">nieudanych wymaganych komórek</div></div>
    <div class="card"><div class="value">{primary["comparison_count"]}</div><div class="label">porównań H1/H3/H5/H6/H7</div></div>
    <div class="card"><div class="value">{qc["patch_count"]:,}</div><div class="label">zwalidowanych patchy PanNuke</div></div>
  </div>
  <h2>Zasady chroniące przed przeciekiem</h2><ul>
    <li>podział wyłącznie po <code>group_id</code>, co najmniej na poziomie patcha;</li>
    <li>główne oceny audytora pochodzą z group-safe OOF;</li>
    <li>finalny reference fold pozostaje niedostępny dla strojenia;</li>
    <li><code>pre_corruption_label</code>, <code>observed_label</code> i <code>is_injected_corruption</code> są rozdzielone;</li>
    <li>cross-class overlap jest wykluczany według zamrożonej polityki, bez zmiany masek.</li>
  </ul></section>

  <section id="qc"><h2>Kontrola jakości PanNuke</h2>
  <p>Surowe maski pozostały niemodyfikowane. Background nie był traktowany jako dokładne dopełnienie,
  void zachowano jako nieoznaczony, a przy nakładaniu klas nie wybierano arbitralnie zwycięzcy.</p>
  <div class="cards">
    <div class="card"><div class="value">{qc["cross_class_overlap_pixel_count"]:,}</div><div class="label">pikseli cross-class overlap</div></div>
    <div class="card"><div class="value">{qc["cross_class_overlap_patch_count"]}</div><div class="label">patchy z overlap</div></div>
    <div class="card"><div class="value">{qc["void_pixel_count"]:,}</div><div class="label">pikseli unlabeled/void</div></div>
    <div class="card"><div class="value">{qc["overlap_touching_instance_count"]:,}</div><div class="label">flagowanych instancji</div></div>
  </div>
  <details><summary>Zobacz deterministyczny overlay QC</summary><img loading="lazy" src="pannuke_mask_qc_overlays.png" alt="Deterministyczny zestaw overlay QC PanNuke"></details></section>

  <section id="dowody"><h2>Pełna tabela H1/H3/H5/H6/H7</h2>
  <p>Tabela zachowuje wszystkie {primary["comparison_count"]} porównań bez wyboru według wyniku. H2 i H4 są pokazane osobno powyżej.</p>
  <details><summary>Otwórz pełną tabelę 36 porównań</summary><div class="table-wrap"><table>
    <thead><tr><th>Comparison ID</th><th>Status</th><th>Metoda A</th><th>Metoda B</th><th>Różnica</th><th>95% CI bootstrap</th><th>Holm p (jednostronne)</th><th>Bootstrap</th></tr></thead>
    <tbody>{"".join(rows)}</tbody></table></div></details>
  <p class="fine"><strong>Uwaga statystyczna:</strong> p są jednostronne i korygowane metodą Holm.
  95% CI jest zapisanym przedziałem percentylowym bootstrap; te dwa podsumowania nie muszą dawać identycznego skrótu decyzyjnego.</p>

  <h2>Reprodukowalność i pochodzenie</h2><p>Run: <code>{html.escape(primary["run_id"])}</code><br>
  Artifact root SHA-256: <code>{primary["artifact_root_sha256"]}</code><br>
  Stage-attestation SHA-256: <code>{primary["stage_attestation_record_sha256"]}</code><br>
  QC overlay SHA-256: <code>{qc["overlay_sha256"]}</code></p>

  <h2>Źródła i granice interpretacji</h2><ul>
    <li><a href="https://arxiv.org/abs/2003.10778">PanNuke extension</a> — dane i klasy jąder.</li>
    <li><a href="https://arxiv.org/abs/1911.00068">Confident Learning</a> — ogólna metodologia jakości etykiet.</li>
    <li><a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/fc20ea8d104cab737a5561096f9bde9b-Abstract.html">AQuA</a> — benchmark jakości etykiet, DOI 10.52202/075280-3494.</li>
    <li><a href="https://arxiv.org/abs/2102.09099">NuCLS</a> — możliwa przyszła walidacja wielooceniająca.</li>
  </ul>
  <p>Original confirmatory, M9, blinded expert review i external validation pozostają pracą przyszłą i nie są przedstawiane jako wykonane.</p></section>

  <footer>Lokalne dowody licencyjne przypisują CC BY-NC-SA 4.0 do katalogu <code>masks/</code> wydania PanNuke;
  projekt niezależnie ogranicza całe użycie danych do badań niekomercyjnych i wymaga obu cytowań PanNuke.
  Pełne dowody i hashe są w <code>evidence.json</code> i <code>manifest.json</code>.</footer>
</main>
</body>
</html>
"""


_MVP_CSS = r"""
:root {
  color-scheme: dark;
  --canvas: #010102;
  --surface-1: #0f1011;
  --surface-2: #141516;
  --surface-3: #18191a;
  --ink: #f7f8f8;
  --ink-muted: #d0d6e0;
  --ink-subtle: #8a8f98;
  --ink-tertiary: #62666d;
  --line: #23252a;
  --line-strong: #34343a;
  --accent: #5e6ad2;
  --accent-hover: #828fff;
  --serif: "Newsreader", "Iowan Old Style", "Palatino Linotype", Georgia, serif;
  --sans: "Inter", "Segoe UI", system-ui, sans-serif;
  --mono: "Cascadia Mono", Consolas, monospace;
  --prose: 680px;
  --figure: 1040px;
  --wide: 1280px;
  --page-pad: clamp(24px, 4.45vw, 64px);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; background: var(--canvas); }
body { margin: 0; overflow-x: hidden; color: var(--ink); background: var(--canvas); font: 400 16px/1.55 var(--sans); }
::selection { color: white; background: var(--accent); }
a { color: inherit; text-underline-offset: 4px; text-decoration-color: var(--ink-tertiary); }
a:hover { color: var(--accent-hover); text-decoration-color: currentColor; }
a:focus-visible, button:focus-visible, summary:focus-visible, input:focus-visible, select:focus-visible { outline: 2px solid var(--accent-hover); outline-offset: 4px; }
code { overflow-wrap: anywhere; font-family: var(--mono); }
.skip { position: fixed; left: 16px; top: -80px; z-index: 100; padding: 10px 14px; border-radius: 8px; color: var(--canvas); background: var(--ink); transition: top .2s ease; }
.skip:focus { top: 14px; }
.site-header { position: fixed; inset: 0 0 auto; z-index: 50; height: 64px; border-bottom: 1px solid transparent; background: rgba(1,1,2,.86); backdrop-filter: blur(16px); transition: transform .3s ease, border-color .3s ease; }
.site-header.is-scrolled { border-color: var(--line); }
.site-header.is-hidden { transform: translateY(-100%); }
.nav-shell { width: min(var(--wide), calc(100% - 2 * var(--page-pad))); height: 100%; margin: auto; display: flex; align-items: center; justify-content: space-between; gap: 28px; }
.brand { display: flex; align-items: center; gap: 11px; font-weight: 600; letter-spacing: -.02em; text-decoration: none; }
.brand-mark { width: 22px; height: 22px; display: grid; grid-template-columns: repeat(2,1fr); gap: 3px; transform: rotate(45deg); }
.brand-mark i { display: block; border-radius: 2px; background: var(--accent); }
.nav-links { display: flex; align-items: center; gap: 24px; }
.nav-links a { color: var(--ink-muted); font-size: 14px; font-weight: 500; text-decoration: none; }
.nav-links a:hover { color: var(--ink); }
.menu-button { display: none; min-width: 42px; min-height: 42px; border: 1px solid var(--line); border-radius: 8px; color: var(--ink); background: var(--surface-1); font: 500 13px var(--sans); }
.reading-progress { position: absolute; left: 0; bottom: -1px; width: 100%; height: 1px; transform: scaleX(0); transform-origin: left; background: var(--accent); }
.hero { position: relative; min-height: 100svh; padding: 64px var(--page-pad) 0; overflow: hidden; border-bottom: 1px solid var(--line); }
#hero-canvas { position: absolute; inset: 64px 0 0; width: 100%; height: calc(100% - 64px); }
.hero-shell { position: relative; z-index: 1; width: min(var(--wide),100%); min-height: calc(100svh - 64px); margin: auto; display: grid; grid-template-columns: minmax(0,.92fr) minmax(360px,1.08fr); align-items: center; pointer-events: none; }
.hero-copy { max-width: 650px; padding: 72px 0; pointer-events: auto; }
.eyebrow { margin: 0 0 30px; color: var(--ink-subtle); font-size: 12px; font-weight: 600; letter-spacing: .09em; text-transform: uppercase; }
h1, h2, .display { font-family: var(--serif); font-weight: 400; }
h1 { max-width: 680px; margin: 0; font-size: clamp(52px,6.5vw,94px); line-height: .96; letter-spacing: -.055em; text-wrap: balance; }
.hero-lead { max-width: 580px; margin: 30px 0 0; color: var(--ink-muted); font: 400 clamp(19px,1.7vw,24px)/1.42 var(--serif); letter-spacing: -.012em; }
.hero-meta { display: flex; flex-wrap: wrap; gap: 10px 20px; margin-top: 38px; color: var(--ink-subtle); font: 500 12px/1.4 var(--sans); }
.hero-meta span { display: flex; align-items: center; gap: 8px; }
.hero-meta i { width: 6px; height: 6px; border-radius: 2px; background: var(--accent); }
.prose, .section-heading { width: min(var(--prose), calc(100% - 2 * var(--page-pad))); margin-inline: auto; }
.figure-width { width: min(var(--figure), calc(100% - 2 * var(--page-pad))); margin-inline: auto; }
.wide-width { width: min(var(--wide), calc(100% - 2 * var(--page-pad))); margin-inline: auto; }
.intro { padding: clamp(104px,12vw,176px) 0; }
.intro p { margin: 0 0 25px; color: var(--ink-muted); font: 400 19px/1.58 var(--serif); letter-spacing: -.008em; }
.intro p:first-child { color: var(--ink); font-size: clamp(25px,2.4vw,34px); line-height: 1.38; }
.scope-note { margin-top: 54px; padding-top: 24px; border-top: 1px solid var(--line); color: var(--ink-subtle); font-size: 14px; }
.scope-note strong { color: var(--ink); font-weight: 500; }
.section { padding: clamp(104px,12vw,176px) 0; border-top: 1px solid var(--line); scroll-margin-top: 64px; }
.section-kicker { display: flex; align-items: center; gap: 12px; margin: 0 0 18px; color: var(--accent-hover); font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }
.section-kicker::before { content: ""; width: 28px; height: 1px; background: currentColor; }
h2 { margin: 0; font-size: clamp(42px,5.5vw,72px); line-height: 1.02; letter-spacing: -.045em; text-wrap: balance; }
.section-deck { margin: 26px 0 0; color: var(--ink-muted); font: 400 clamp(19px,1.9vw,25px)/1.46 var(--serif); }
.section-heading { margin-bottom: 64px; }
.story { position: relative; height: 520vh; border-top: 1px solid var(--line); scroll-margin-top: 64px; }
.story-frame { position: sticky; top: 0; min-height: 100vh; display: grid; place-items: center; overflow: hidden; }
.story-inner { width: min(var(--wide),calc(100% - 2 * var(--page-pad))); display: grid; grid-template-columns: minmax(0,1.35fr) minmax(340px,.78fr); gap: clamp(48px,7vw,112px); align-items: center; }
.workflow-panel { min-height: 680px; display: grid; place-items: center; border: 1px solid var(--line); border-radius: 16px; background: var(--surface-1); overflow: hidden; }
.workflow-panel svg { width: 100%; height: auto; }
.workflow-panel [data-stage] { opacity: .11; transition: opacity .45s ease, transform .45s ease; transform-origin: center; }
.workflow-panel [data-stage].is-visible { opacity: .48; }
.workflow-panel [data-stage].is-current { opacity: 1; transform: translateY(-2px); }
.workflow-line { fill: none; stroke: var(--ink-tertiary); stroke-width: 1.5; stroke-linecap: round; stroke-dasharray: 4 7; }
.workflow-accent { fill: none; stroke: var(--accent); stroke-width: 2; stroke-linecap: round; }
.workflow-node { fill: var(--surface-2); stroke: var(--line-strong); stroke-width: 1.5; }
.workflow-cell { fill: var(--accent); }
.workflow-label { fill: var(--ink-subtle); font: 500 12px var(--sans); letter-spacing: .06em; }
.story-steps { display: flex; flex-direction: column; gap: 24px; margin: 0; padding: 0; list-style: none; }
.story-step { position: relative; padding: 4px 0 4px 42px; opacity: .28; transition: opacity .25s ease; }
.story-step::before { content: attr(data-index); position: absolute; left: 0; top: 5px; color: var(--ink-tertiary); font: 500 12px var(--mono); }
.story-step.is-active { opacity: 1; }
.story-step.is-active::before { color: var(--accent-hover); }
.story-step h3 { margin: 0 0 7px; font-size: 19px; font-weight: 600; letter-spacing: -.02em; }
.story-step p { margin: 0; color: var(--ink-subtle); font: 400 17px/1.46 var(--serif); }
.story-step.is-active p { color: var(--ink-muted); }
.result-lead { padding: 34px; border: 1px solid var(--line-strong); border-radius: 16px; background: var(--surface-1); }
.result-label { display: flex; justify-content: space-between; gap: 20px; color: var(--ink-subtle); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
.result-number { display: block; margin: 22px 0 8px; font: 400 clamp(46px,7vw,84px)/1 var(--serif); letter-spacing: -.05em; }
.result-lead p { max-width: 760px; margin: 0; color: var(--ink-muted); font: 400 19px/1.52 var(--serif); }
.result-lead strong { color: var(--ink); font-weight: 500; }
.h4-chart, .h2-chart { margin-top: 28px; padding: clamp(24px,4vw,48px); border: 1px solid var(--line); border-radius: 16px; background: var(--surface-1); }
.metric-axis, .range-axis, .forest-axis { display: grid; grid-template-columns: 1fr auto 1fr; gap: 16px; align-items: center; margin-bottom: 22px; color: var(--ink-tertiary); font: 500 11px/1.4 var(--sans); text-transform: uppercase; letter-spacing: .06em; }
.metric-axis span:last-child, .range-axis span:last-child, .forest-axis span:last-child { text-align: right; }
.metric-row { display: grid; grid-template-columns: 190px 1fr 92px; gap: 18px; align-items: center; min-height: 52px; border-top: 1px solid var(--line); }
.metric-name { color: var(--ink-muted); font-size: 13px; }
.metric-track, .difference-track, .range-track { position: relative; display: block; height: 18px; }
.metric-track::before, .difference-track::before, .range-track::before { content: ""; position: absolute; left: 0; right: 0; top: 50%; height: 1px; background: var(--line-strong); }
.metric-dot { position: absolute; left: var(--x); top: 50%; width: 11px; height: 11px; border: 2px solid var(--surface-1); border-radius: 3px; background: var(--accent); transform: translate(-50%,-50%) rotate(45deg); }
.metric-row strong, .range-values, .forest-value { color: var(--ink-muted); font: 400 12px var(--mono); text-align: right; }
.difference-panel { margin-top: 32px; padding-top: 28px; border-top: 1px solid var(--line-strong); }
.difference-heading { display: flex; justify-content: space-between; gap: 24px; margin-bottom: 18px; font-size: 14px; }
.difference-heading strong { font-family: var(--mono); font-weight: 400; }
.difference-track { height: 28px; }
.difference-zero, .forest-zero { position: absolute; left: var(--zero); top: 0; bottom: 0; width: 1px; background: var(--ink-tertiary); }
.difference-ci, .forest-ci { position: absolute; left: var(--low); width: calc(var(--high) - var(--low)); top: 50%; height: 3px; background: var(--ink-muted); transform: translateY(-50%); }
.difference-point, .forest-point { position: absolute; left: var(--point); top: 50%; width: 12px; height: 12px; border-radius: 3px; background: var(--accent); transform: translate(-50%,-50%) rotate(45deg); }
.difference-axis { display: grid; grid-template-columns: repeat(3,1fr); color: var(--ink-tertiary); font: 400 11px var(--mono); }
.difference-axis span:nth-child(2) { text-align: center; }
.difference-axis span:last-child { text-align: right; }
.difference-panel p, .chart-note { margin: 20px 0 0; color: var(--ink-subtle); font-size: 13px; }
.hypothesis-ledger { border-top: 1px solid var(--line-strong); }
.hypothesis-row { margin: 0; border: 0; border-bottom: 1px solid var(--line); border-radius: 0; background: transparent; }
.hypothesis-row summary { min-height: 88px; display: grid; grid-template-columns: 70px 1fr minmax(180px,auto) 24px; gap: 20px; align-items: center; cursor: pointer; list-style: none; }
.hypothesis-row summary::-webkit-details-marker { display: none; }
.hypothesis-id { color: var(--accent-hover); font: 500 13px var(--mono); }
.hypothesis-title { font-size: 18px; font-weight: 500; letter-spacing: -.02em; }
.hypothesis-result { color: var(--ink-subtle); font: 400 13px var(--mono); text-align: right; }
.hypothesis-toggle { color: var(--ink-subtle); font-size: 22px; transition: transform .2s ease; }
.hypothesis-row[open] .hypothesis-toggle { transform: rotate(45deg); }
.hypothesis-row p { max-width: 720px; margin: 0 0 28px 90px; color: var(--ink-subtle); font: 400 17px/1.5 var(--serif); }
.forest-plot { padding: 28px; border: 1px solid var(--line); border-radius: 16px; background: var(--surface-1); }
.forest-group { display: flex; justify-content: space-between; gap: 20px; margin: 28px 0 8px; padding-top: 18px; border-top: 1px solid var(--line-strong); color: var(--ink-tertiary); font-size: 11px; letter-spacing: .06em; text-transform: uppercase; }
.forest-group:first-of-type { margin-top: 0; padding-top: 0; border-top: 0; }
.forest-group span:first-child { color: var(--accent-hover); font-family: var(--mono); }
.forest-row { min-height: 42px; display: grid; grid-template-columns: minmax(220px,1.15fr) minmax(240px,1fr) 64px; gap: 20px; align-items: center; border-top: 1px solid rgba(35,37,42,.68); }
.forest-label { overflow: hidden; color: var(--ink-subtle); font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
.forest-track { position: relative; height: 22px; }
.forest-track-empty { display: grid; place-items: center; border: 1px dashed var(--line-strong); color: var(--ink-tertiary); font-size: 10px; text-transform: uppercase; }
.forest-track-empty::before { display: none; }
.forest-ci { height: 2px; }
.forest-point { width: 9px; height: 9px; }
.forest-zero { opacity: .7; }
.range-row { min-height: 66px; display: grid; grid-template-columns: 170px 1fr 88px 118px; gap: 18px; align-items: center; border-top: 1px solid var(--line); }
.range-name { color: var(--ink-muted); font-size: 13px; }
.range-track { height: 22px; }
.range-line { position: absolute; left: var(--low); width: calc(var(--high) - var(--low)); top: 50%; height: 3px; background: var(--accent); transform: translateY(-50%); }
.range-start, .range-end { position: absolute; top: 50%; width: 9px; height: 9px; border-radius: 3px; background: var(--accent-hover); transform: translate(-50%,-50%) rotate(45deg); }
.range-start { left: var(--low); }
.range-end { left: var(--high); }
.range-count { display: flex; flex-direction: column; text-align: right; font: 400 12px var(--mono); }
.range-count small { color: var(--ink-tertiary); font: 400 10px var(--sans); }
.seed-panel { display: grid; grid-template-columns: minmax(0,1.2fr) minmax(300px,.8fr); gap: 32px; align-items: center; padding: clamp(24px,4vw,48px); border: 1px solid var(--line); border-radius: 16px; background: var(--surface-1); }
.seed-panel svg { width: 100%; height: auto; }
.seed-copy h3 { margin: 0 0 15px; font: 500 22px/1.25 var(--sans); }
.seed-copy p { margin: 0 0 14px; color: var(--ink-subtle); font: 400 17px/1.48 var(--serif); }
.seed-copy code { color: var(--ink-muted); font-size: 11px; }
.stats-grid { display: grid; grid-template-columns: repeat(4,1fr); border: 1px solid var(--line); border-radius: 16px; overflow: hidden; }
.stat { min-height: 150px; padding: 28px; border-right: 1px solid var(--line); background: var(--surface-1); }
.stat:last-child { border-right: 0; }
.stat strong { display: block; margin-bottom: 10px; font: 400 clamp(30px,3.5vw,48px)/1 var(--serif); letter-spacing: -.04em; }
.stat span { color: var(--ink-subtle); font-size: 12px; }
.rules { display: grid; grid-template-columns: repeat(2,1fr); gap: 1px; margin-top: 42px; border: 1px solid var(--line); background: var(--line); }
.rule { min-height: 160px; padding: 28px; background: var(--surface-1); }
.rule b { display: block; margin-bottom: 12px; font-weight: 500; }
.rule p { margin: 0; color: var(--ink-subtle); font: 400 16px/1.5 var(--serif); }
.qc-frame { padding: 18px; border: 1px solid var(--line); border-radius: 16px; background: var(--surface-1); }
.qc-toolbar { display: flex; justify-content: space-between; gap: 20px; margin-bottom: 16px; color: var(--ink-subtle); font-size: 12px; }
.qc-frame img { display: block; width: 100%; height: auto; border: 1px solid var(--line); border-radius: 10px; background: white; }
details.evidence-details { margin-top: 32px; border: 1px solid var(--line); border-radius: 12px; background: var(--surface-1); }
details.evidence-details > summary { padding: 18px 20px; cursor: pointer; color: var(--ink-muted); font-weight: 500; }
details.evidence-details[open] > summary { border-bottom: 1px solid var(--line); }
.table-tools { display: flex; flex-wrap: wrap; gap: 12px; padding: 16px; border-bottom: 1px solid var(--line); }
.table-tools label { display: flex; flex-direction: column; gap: 6px; color: var(--ink-tertiary); font-size: 10px; letter-spacing: .06em; text-transform: uppercase; }
input, select { min-height: 40px; padding: 8px 11px; border: 1px solid var(--line-strong); border-radius: 8px; color: var(--ink); background: var(--surface-2); font: 400 13px var(--sans); }
.table-count { align-self: end; margin-left: auto; padding: 10px 0; color: var(--ink-subtle); font-size: 12px; }
.table-wrap { max-width: 100%; overflow: auto; }
table { width: 100%; min-width: 1050px; border-collapse: collapse; font-size: 12px; }
table.compact { min-width: 760px; }
th, td { padding: 11px 13px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
th { position: sticky; top: 0; z-index: 2; color: var(--ink-muted); background: var(--surface-2); font-weight: 500; white-space: nowrap; }
td { color: var(--ink-subtle); }
td:first-child, td:nth-child(n+5) { font-family: var(--mono); font-size: 11px; }
tr:hover td { color: var(--ink-muted); background: var(--surface-2); }
.fine { color: var(--ink-subtle); font-size: 13px; }
.fine strong { color: var(--ink-muted); font-weight: 500; }
.provenance { display: grid; grid-template-columns: 1fr 1fr; gap: 48px; }
.provenance h3 { margin: 0 0 18px; font-size: 17px; font-weight: 500; }
.provenance p, .provenance li { color: var(--ink-subtle); font-size: 13px; }
.provenance ul { margin: 0; padding-left: 18px; }
.hash-list { display: grid; gap: 12px; margin: 0; }
.hash-list div { padding-bottom: 12px; border-bottom: 1px solid var(--line); }
.hash-list dt { color: var(--ink-tertiary); font-size: 10px; text-transform: uppercase; }
.hash-list dd { margin: 5px 0 0; color: var(--ink-subtle); font: 400 11px/1.45 var(--mono); overflow-wrap: anywhere; }
footer { padding: 64px var(--page-pad); border-top: 1px solid var(--line); color: var(--ink-tertiary); font-size: 12px; }
.footer-inner { width: min(var(--wide),100%); margin: auto; display: flex; justify-content: space-between; gap: 40px; }
.motion-ready .reveal { opacity: 0; transform: translateY(18px); transition: opacity .65s cubic-bezier(.2,.7,.2,1), transform .65s cubic-bezier(.2,.7,.2,1); }
.motion-ready .reveal.is-visible { opacity: 1; transform: none; }
@media (max-width: 900px) {
  .hero-shell { grid-template-columns: 1fr; align-items: start; }
  .hero-copy { max-width: 720px; padding-top: 18vh; }
  .story { height: auto; padding: 96px 0; }
  .story-frame { position: static; min-height: 0; overflow: visible; }
  .story-inner { grid-template-columns: 1fr; }
  .workflow-panel { min-height: 0; }
  .story-steps { gap: 0; }
  .story-step { padding: 24px 0 24px 42px; border-bottom: 1px solid var(--line); opacity: 1; }
  .story-step p { color: var(--ink-muted); }
  .workflow-panel [data-stage], .workflow-panel [data-stage].is-visible, .workflow-panel [data-stage].is-current { opacity: 1; transform: none; }
  .seed-panel { grid-template-columns: 1fr; }
  .stats-grid { grid-template-columns: repeat(2,1fr); }
  .stat:nth-child(2) { border-right: 0; }
  .stat:nth-child(-n+2) { border-bottom: 1px solid var(--line); }
  .metric-row { grid-template-columns: 150px 1fr 82px; }
  .range-row { grid-template-columns: 150px 1fr 82px; }
  .range-values { display: none; }
}
@media (max-width: 720px) {
  :root { --page-pad: 24px; }
  .nav-links { position: absolute; top: 64px; left: 0; right: 0; display: none; flex-direction: column; align-items: flex-start; gap: 0; padding: 12px 24px 20px; border-bottom: 1px solid var(--line); background: var(--canvas); }
  .nav-links.is-open { display: flex; }
  .nav-links a { width: 100%; padding: 12px 0; }
  .menu-button { display: block; }
  .hero-copy { padding-top: 14vh; }
  .hero-lead { max-width: 420px; }
  .hero-meta { max-width: 310px; }
  .intro p { font-size: 18px; }
  .section-heading { margin-bottom: 42px; }
  .result-lead { padding: 24px; }
  .metric-row { grid-template-columns: 1fr 78px; gap: 8px 14px; padding: 14px 0; }
  .metric-track { grid-column: 1/-1; grid-row: 2; }
  .hypothesis-row summary { grid-template-columns: 48px 1fr 20px; min-height: 92px; }
  .hypothesis-result { grid-column: 2; grid-row: 2; text-align: left; }
  .hypothesis-toggle { grid-column: 3; grid-row: 1/3; }
  .hypothesis-row p { margin-left: 48px; }
  .forest-plot { padding: 18px; }
  .forest-axis { display: none; }
  .forest-row { grid-template-columns: 1fr 52px; gap: 5px 12px; padding: 10px 0; }
  .forest-track { grid-column: 1/-1; grid-row: 2; }
  .forest-value { grid-column: 2; grid-row: 1; }
  .forest-group { margin-top: 22px; }
  .forest-group span:last-child { display: none; }
  .range-row { grid-template-columns: 1fr 80px; gap: 7px 12px; padding: 14px 0; }
  .range-track { grid-column: 1/-1; grid-row: 2; }
  .range-count { grid-column: 2; grid-row: 1; }
  .rules, .stats-grid, .provenance { grid-template-columns: 1fr; }
  .stat { border-right: 0; border-bottom: 1px solid var(--line); }
  .stat:last-child { border-bottom: 0; }
  .footer-inner { flex-direction: column; }
  .table-count { width: 100%; margin-left: 0; }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; }
  .site-header { transform: none !important; }
  .motion-ready .reveal { opacity: 1; transform: none; }
}
@media print {
  :root { color-scheme: light; --canvas: #fff; --surface-1: #fff; --surface-2: #f6f6f6; --surface-3: #eee; --ink: #111; --ink-muted: #333; --ink-subtle: #555; --ink-tertiary: #666; --line: #ccc; --line-strong: #999; --accent: #4b56bb; }
  .site-header, .menu-button, #hero-canvas, .reading-progress { display: none !important; }
  .hero { min-height: auto; padding-top: 64px; }
  .hero-shell { min-height: auto; grid-template-columns: 1fr; }
  .story { height: auto; }
  .story-frame { position: static; min-height: 0; }
  .story-inner { grid-template-columns: 1fr; }
  .workflow-panel [data-stage], .story-step { opacity: 1 !important; }
  .section { break-inside: avoid; }
}
"""


_MVP_SCRIPT = r"""
(() => {
  const root = document.documentElement;
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!reduced) root.classList.add('motion-ready');

  const header = document.getElementById('site-header');
  const progress = document.getElementById('reading-progress');
  let lastScroll = window.scrollY;
  const updateChrome = () => {
    const current = window.scrollY;
    header.classList.toggle('is-scrolled', current > 16);
    header.classList.toggle('is-hidden', current > 140 && current > lastScroll + 4);
    const range = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.transform = `scaleX(${range > 0 ? current / range : 0})`;
    lastScroll = current;
  };
  window.addEventListener('scroll', updateChrome, {passive: true});
  updateChrome();

  const menuButton = document.getElementById('menu-button');
  const navLinks = document.getElementById('nav-links');
  menuButton.addEventListener('click', () => {
    const open = !navLinks.classList.contains('is-open');
    navLinks.classList.toggle('is-open', open);
    menuButton.setAttribute('aria-expanded', String(open));
  });
  navLinks.addEventListener('click', event => {
    if (event.target instanceof HTMLAnchorElement) {
      navLinks.classList.remove('is-open');
      menuButton.setAttribute('aria-expanded', 'false');
    }
  });

  const canvas = document.getElementById('hero-canvas');
  const context = canvas.getContext('2d');
  const cells = [];
  let canvasWidth = 0;
  let canvasHeight = 0;
  let lastFrame = 0;
  let seed = 271828;
  const random = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  for (let index = 0; index < 112; index += 1) {
    cells.push({
      angle: random() * Math.PI * 2,
      radius: Math.sqrt(random()),
      phase: random() * Math.PI * 2,
      rank: index,
    });
  }
  const resizeCanvas = () => {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    canvasWidth = rect.width;
    canvasHeight = rect.height;
    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
  };
  const ease = value => value * value * (3 - 2 * value);
  const drawHero = timestamp => {
    if (!reduced && timestamp - lastFrame < 32) {
      requestAnimationFrame(drawHero);
      return;
    }
    lastFrame = timestamp;
    context.clearRect(0, 0, canvasWidth, canvasHeight);
    const grid = canvasWidth < 720 ? 18 : 24;
    context.fillStyle = 'rgba(98,102,109,.22)';
    for (let x = grid / 2; x < canvasWidth; x += grid) {
      for (let y = grid / 2; y < canvasHeight; y += grid) context.fillRect(x, y, 1, 1);
    }
    const mobile = canvasWidth < 900;
    const centerX = mobile ? canvasWidth * .57 : canvasWidth * .73;
    const centerY = mobile ? canvasHeight * .82 : canvasHeight * .52;
    const spanX = Math.min(canvasWidth * (mobile ? .27 : .18), 260);
    const spanY = Math.min(canvasHeight * (mobile ? .14 : .24), 220);
    const cycle = reduced ? 0 : (Math.sin(timestamp * .00024) + 1) / 2;
    const morph = ease(cycle);
    cells.forEach((cell, index) => {
      const clusterX = centerX + Math.cos(cell.angle) * cell.radius * spanX *
        (.72 + .28 * Math.sin(cell.angle * 3));
      const clusterY = centerY + Math.sin(cell.angle) * cell.radius * spanY;
      const columns = mobile ? 8 : 10;
      const size = mobile ? 7 : 10;
      const gap = mobile ? 5 : 8;
      const queueX = centerX - ((columns - 1) * (size + gap)) / 2 +
        (index % columns) * (size + gap);
      const queueY = centerY - (mobile ? 62 : 105) +
        Math.floor(index / columns) * (size + gap);
      const wobble = reduced ? 0 : Math.sin(timestamp * .001 + cell.phase) * 2.4;
      const x = clusterX * (1 - morph) + queueX * morph;
      const y = clusterY * (1 - morph) + queueY * morph + wobble;
      const alpha = .35 + .58 * (1 - cell.radius * .45) +
        Math.sin(timestamp * .0012 + cell.phase) * .06;
      context.fillStyle = `rgba(94,106,210,${Math.max(.18, Math.min(.94, alpha))})`;
      context.beginPath();
      context.roundRect(x - size / 2, y - size / 2, size, size, 2.5);
      context.fill();
    });
    if (!reduced) requestAnimationFrame(drawHero);
  };
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas, {passive: true});
  requestAnimationFrame(drawHero);

  const story = document.querySelector('.story');
  const storySteps = [...document.querySelectorAll('[data-story-step]')];
  const stageItems = [...document.querySelectorAll('#workflow-panel [data-stage]')];
  const setStoryStage = active => {
    storySteps.forEach((step, index) => step.classList.toggle('is-active', index === active));
    stageItems.forEach(item => {
      const stage = Number(item.dataset.stage);
      item.classList.toggle('is-visible', stage <= active);
      item.classList.toggle('is-current', stage === active);
    });
  };
  const updateStory = () => {
    if (window.innerWidth <= 900 || reduced) {
      setStoryStage(4);
      return;
    }
    const rect = story.getBoundingClientRect();
    const distance = Math.max(1, rect.height - window.innerHeight);
    const ratio = Math.max(0, Math.min(.9999, -rect.top / distance));
    setStoryStage(Math.min(4, Math.floor(ratio * 5)));
  };
  window.addEventListener('scroll', updateStory, {passive: true});
  window.addEventListener('resize', updateStory, {passive: true});
  updateStory();

  const revealItems = [...document.querySelectorAll('.reveal')];
  if (reduced || !('IntersectionObserver' in window)) {
    revealItems.forEach(item => item.classList.add('is-visible'));
  } else {
    const observer = new IntersectionObserver(entries => entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    }), {threshold: .12});
    revealItems.forEach(item => observer.observe(item));
  }

  const hypothesisFilter = document.getElementById('filter-hypothesis');
  const statusFilter = document.getElementById('filter-status');
  const searchFilter = document.getElementById('filter-search');
  const comparisonRows = [...document.querySelectorAll('[data-comparison-row]')];
  const tableCount = document.getElementById('table-count');
  const filterTable = () => {
    const hypothesis = hypothesisFilter.value;
    const status = statusFilter.value;
    const query = searchFilter.value.trim().toLowerCase();
    let visible = 0;
    comparisonRows.forEach(row => {
      const matches = (hypothesis === 'ALL' || row.dataset.hypothesis === hypothesis) &&
        (status === 'ALL' || row.dataset.status === status) &&
        (!query || row.textContent.toLowerCase().includes(query));
      row.hidden = !matches;
      if (matches) visible += 1;
    });
    tableCount.textContent = `${visible} / ${comparisonRows.length} wierszy`;
  };
  [hypothesisFilter, statusFilter, searchFilter].forEach(control =>
    control.addEventListener('input', filterTable));
})();
"""


def _render_html_v4(evidence: dict[str, Any]) -> str:
    primary = evidence["primary"]
    qc = evidence["pannuke_qc"]
    hypotheses = primary["hypothesis_comparisons"]
    h2 = primary["h2_subgroups"]
    h4 = primary["h4_restoration"]
    seed_audit = primary["instance_dependent_seed_audit"]

    comparison_rows = _render_comparison_rows_v3(primary["comparisons"])
    forest_plot = _render_forest_plot_v3(primary["comparisons"])
    h4_chart = _render_h4_chart_v3(h4)
    h2_chart = _render_h2_chart_v3(h2)
    hypothesis_ledger = _render_hypothesis_ledger_v4(hypotheses, h2, h4)

    seed_rows: list[str] = []
    for record in seed_audit["records"]:
        ranking_hash = str(record["ranking_sha256"])
        oof_hash = str(record["oof_predictions_sha256"])
        seed_rows.append(
            "<tr>"
            f"<td>{record['seed']}</td>"
            f"<td><code>{html.escape(str(record['cell_id']))}</code></td>"
            f'<td><code title="{ranking_hash}">{ranking_hash[:16]}…</code></td>'
            f'<td><code title="{oof_hash}">{oof_hash[:16]}…</code></td>'
            "</tr>"
        )
    seed_hash = html.escape(str(seed_audit["records"][0]["ranking_sha256"]))
    oof_hash = html.escape(str(seed_audit["records"][0]["oof_predictions_sha256"]))
    point_difference = _require_real(h4["point_difference"], role="H4 point difference")
    reported_comparisons = sum(record["status"] == "reported" for record in primary["comparisons"])
    unavailable_comparisons = primary["comparison_count"] - reported_comparisons

    template = """<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="AANCA: reprodukowalny, niediagnostyczny prototyp rankingu potencjalnie niespójnych adnotacji jąder do przeglądu eksperckiego.">
  <meta name="theme-color" content="#010102">
  <meta name="referrer" content="no-referrer">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&amp;family=Newsreader:opsz,wght@6..72,400;6..72,500&amp;display=swap" rel="stylesheet">
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%23010102'/%3E%3Crect x='14' y='14' width='12' height='12' rx='3' fill='%235e6ad2'/%3E%3Crect x='38' y='14' width='12' height='12' rx='3' fill='%23828fff'/%3E%3Crect x='26' y='26' width='12' height='12' rx='3' fill='%235e6ad2'/%3E%3Crect x='14' y='38' width='12' height='12' rx='3' fill='%23828fff'/%3E%3Crect x='38' y='38' width='12' height='12' rx='3' fill='%235e6ad2'/%3E%3C/svg%3E">
  <title>AANCA — gdy adnotacja może być niespójna</title>
  <style>__CSS__</style>
</head>
<body>
<a class="skip" href="#main">Przejdź do treści</a>
<header class="site-header" id="site-header">
  <div class="nav-shell">
    <a class="brand" href="#top" aria-label="AANCA — początek strony"><span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span>AANCA</a>
    <button class="menu-button" id="menu-button" type="button" aria-expanded="false" aria-controls="nav-links">Menu</button>
    <nav class="nav-links" id="nav-links" aria-label="Sekcje"><a href="#wynik">Wynik</a><a href="#benchmarki">Benchmarki</a><a href="#metoda">Metoda</a><a href="#qc">QC</a><a href="#dowody">Dowody</a></nav>
  </div>
  <div class="reading-progress" id="reading-progress" aria-hidden="true"></div>
</header>

<section class="hero" id="top" aria-labelledby="hero-title">
  <canvas id="hero-canvas" aria-hidden="true"></canvas>
  <div class="hero-shell"><div class="hero-copy">
    <p class="eyebrow">Automated Nucleus-Annotation Auditing · research prototype</p>
    <h1 id="hero-title">Gdy adnotacja może być niespójna.</h1>
    <p class="hero-lead">Reprodukowalny ranking <em>potentially inconsistent annotation</em> do <em>recommended for expert review</em> — bez automatycznej zmiany danych źródłowych.</p>
    <div class="hero-meta"><span><i></i>DEMO_COMPLETE</span><span><i></i>PRIMARY_STUDY_COMPLETE</span><span>amended_or_exploratory</span></div>
  </div></div>
</section>

<main id="main">
  <section class="intro"><div class="prose reveal">
    <p>Modele potrafią wskazywać przypadki warte ponownego spojrzenia. Nie oznacza to jednak, że model rozstrzyga, kto miał rację.</p>
    <p>AANCA bada kontrolowany problem: czy ranking może priorytetyzować wstrzyknięte niespójności etykiet lepiej niż losowy przegląd, bez naruszania finalnego reference fold i bez modyfikowania źródłowych adnotacji.</p>
    <p>Wynik jest użyteczny tylko jako rekomendacja kolejności pracy eksperta. Nie jest diagnozą, automatyczną korektą ani dowodem błędu patologa.</p>
    <div class="scope-note"><strong>Granica wniosku.</strong> Confirmatory, blinded expert review i external validation nie zostały wykonane. Formalny status naukowy pozostaje <code>PRIMARY_STUDY_COMPLETE</code>.</div>
  </div></section>

  <section class="story" id="metoda" aria-labelledby="story-title"><div class="story-frame"><div class="story-inner">
    <div class="workflow-panel" id="workflow-panel" data-active-stage="0">
      <svg viewBox="0 0 760 680" role="img" aria-labelledby="workflow-title workflow-desc">
        <title id="workflow-title">Pięć etapów kontrolowanego benchmarku AANCA</title>
        <desc id="workflow-desc">Źródłowe adnotacje, kontrolowana korupcja, predykcje group-safe OOF, ranking audytowy i rekomendacja do przeglądu eksperta.</desc>
        <g opacity=".16"><path class="workflow-line" d="M70 110H690M70 230H690M70 350H690M70 470H690M70 590H690"/></g>
        <g data-stage="0"><rect class="workflow-node" x="74" y="62" width="166" height="96" rx="16"/><g transform="translate(104 82)"><rect class="workflow-cell" width="20" height="20" rx="5"/><rect class="workflow-cell" x="28" y="8" width="20" height="20" rx="5" opacity=".72"/><rect class="workflow-cell" x="10" y="34" width="20" height="20" rx="5" opacity=".9"/><rect class="workflow-cell" x="42" y="38" width="20" height="20" rx="5" opacity=".55"/></g><text class="workflow-label" x="178" y="112">SOURCE LABELS</text></g>
        <path data-stage="1" class="workflow-accent" d="M242 110H326"/><g data-stage="1"><rect class="workflow-node" x="328" y="62" width="166" height="96" rx="16"/><text class="workflow-label" x="354" y="96">PRE → OBSERVED</text><rect x="354" y="112" width="56" height="12" rx="6" fill="#62666d"/><rect x="414" y="112" width="54" height="12" rx="6" class="workflow-cell"/></g>
        <path data-stage="2" class="workflow-accent" d="M496 110H572V230H492"/><g data-stage="2"><rect class="workflow-node" x="300" y="182" width="194" height="96" rx="16"/><text class="workflow-label" x="326" y="214">GROUP-SAFE OOF</text><rect x="326" y="231" width="36" height="24" rx="5" fill="#5e6ad2"/><rect x="369" y="231" width="36" height="24" rx="5" fill="#34343a"/><rect x="412" y="231" width="36" height="24" rx="5" fill="#34343a"/></g>
        <path data-stage="3" class="workflow-accent" d="M300 230H188V350H300"/><g data-stage="3"><rect class="workflow-node" x="300" y="302" width="194" height="96" rx="16"/><text class="workflow-label" x="326" y="332">AUDIT RANKING</text><rect x="326" y="349" width="126" height="8" rx="4" class="workflow-cell"/><rect x="326" y="364" width="92" height="8" rx="4" fill="#62666d"/><rect x="326" y="379" width="58" height="8" rx="4" fill="#34343a"/></g>
        <path data-stage="4" class="workflow-accent" d="M494 350H590V470H494"/><g data-stage="4"><rect class="workflow-node" x="300" y="422" width="194" height="96" rx="16"/><circle cx="348" cy="470" r="24" fill="none" stroke="#5e6ad2" stroke-width="2"/><path d="M338 470l8 8 14-18" fill="none" stroke="#828fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/><text class="workflow-label" x="386" y="464">EXPERT REVIEW</text><text class="workflow-label" x="386" y="484">RECOMMENDED</text></g>
        <g data-stage="4"><path class="workflow-line" d="M397 520V590H170"/><text class="workflow-label" x="170" y="620">NO AUTOMATIC ANNOTATION CHANGE</text></g>
      </svg>
    </div>
    <div>
      <p class="section-kicker">Jak działa benchmark</p>
      <h2 id="story-title" class="display" style="font-size:clamp(38px,4vw,58px);margin-bottom:38px">Od adnotacji do kolejki przeglądu.</h2>
      <ol class="story-steps">
        <li class="story-step is-active" data-story-step="0" data-index="01"><h3>Źródłowe adnotacje</h3><p>Każde jądro pozostaje powiązane z patchem i pierwotną klasą.</p></li>
        <li class="story-step" data-story-step="1" data-index="02"><h3>Kontrolowana korupcja</h3><p><code>pre_corruption_label</code>, <code>observed_label</code> i metadane pozostają osobne.</p></li>
        <li class="story-step" data-story-step="2" data-index="03"><h3>Group-safe OOF</h3><p>Podział odbywa się po <code>group_id</code>, nigdy po pojedynczym jądrze.</p></li>
        <li class="story-step" data-story-step="3" data-index="04"><h3>Ranking audytowy</h3><p>Model ustala priorytet potencjalnie niespójnych adnotacji.</p></li>
        <li class="story-step" data-story-step="4" data-index="05"><h3>Przegląd eksperta</h3><p>System rekomenduje przypadki; nie zmienia adnotacji automatycznie.</p></li>
      </ol>
    </div>
  </div></div></section>

  <section class="section" id="wynik">
    <div class="section-heading"><p class="section-kicker">Najważniejszy wynik · H4</p><h2>Dobry ranking nie zagwarantował lepszego downstream.</h2><p class="section-deck">Zarejestrowana hipoteza restoration nie została wsparta. Wynik przeciwny pozostaje na pierwszym planie.</p></div>
    <div class="figure-width reveal">
      <div class="result-lead"><div class="result-label"><span>Audit-guided minus random</span><span>adverse to registered hypothesis</span></div><span class="result-number">__H4_POINT__</span><p><strong>H4 uzyskała wynik przeciwny do zarejestrowanej hipotezy.</strong> Audit-guided macro-F1 <strong>__H4_AUDIT__</strong> było niższe niż random-review mean <strong>__H4_RANDOM__</strong>. Wyniku nie ukryto i nie przeliczano po jego poznaniu.</p></div>
      __H4_CHART__
    </div>
  </section>

  <section class="section" id="benchmarki">
    <div class="section-heading"><p class="section-kicker">Zarejestrowane pytania · H1-H7</p><h2>Pełny obraz, nie tylko najlepsze wyniki.</h2><p class="section-deck">Każdy wynik, brak wyniku i zastrzeżenie pozostaje widoczne. Holm p oznacza tu zapisane p jednostronne.</p></div>
    <div class="figure-width reveal"><div class="hypothesis-ledger">__HYPOTHESIS_LEDGER__</div></div>
    <div class="section-heading" style="margin-top:clamp(104px,12vw,176px)"><p class="section-kicker">Atlas porównań</p><h2 style="font-size:clamp(38px,4.8vw,64px)">36 zamrożonych porównań.</h2><p class="section-deck">Punkty pokazują różnicę average precision, linie — zapisane 95% przedziały percentylowe bootstrap.</p></div>
    <div class="figure-width reveal"><div class="forest-plot" aria-label="Forest plot porównań H1, H3, H5, H6 i H7">__FOREST_PLOT__</div><p class="fine"><strong>Uwaga:</strong> zapisany jednostronny test Holma i dwustronny skrót 95% CI nie muszą prowadzić do identycznej etykiety słownej. H6 pozostaje niedostępne, nie jest zerem.</p></div>
  </section>

  <section class="section" id="podgrupy">
    <div class="section-heading"><p class="section-kicker">H2 · opis heterogeniczności</p><h2>Wynik zależał od kontekstu.</h2><p class="section-deck">Zakresy obejmują raportowalne estymaty dla klas, tkanek, mechanizmów i poziomów korupcji. Nie są rankingiem biologicznym.</p></div>
    <div class="figure-width reveal">__H2_CHART__</div>
  </section>

  <section class="section" id="seedy">
    <div class="section-heading"><p class="section-kicker">Ujawnienie reprodukowalności</p><h2>Trzy seedy, jedna deterministyczna realizacja.</h2><p class="section-deck">Wiersze pozostają zamrożone, ale nie są interpretowane jako trzy niezależne replikacje.</p></div>
    <div class="figure-width reveal">
      <div class="seed-panel">
        <svg viewBox="0 0 680 390" role="img" aria-labelledby="seed-title seed-desc"><title id="seed-title">Seedy 404, 405 i 406 prowadzą do identycznych plików</title><desc id="seed-desc">Trzy ścieżki łączą się w jeden ranking i jeden wynik OOF, ponieważ pliki są bajtowo identyczne.</desc>
          <g font-family="Inter, sans-serif" font-size="13" fill="#d0d6e0"><rect x="34" y="42" width="126" height="54" rx="10" fill="#141516" stroke="#34343a"/><text x="72" y="75">seed 404</text><rect x="34" y="168" width="126" height="54" rx="10" fill="#141516" stroke="#34343a"/><text x="72" y="201">seed 405</text><rect x="34" y="294" width="126" height="54" rx="10" fill="#141516" stroke="#34343a"/><text x="72" y="327">seed 406</text></g>
          <g fill="none" stroke="#5e6ad2" stroke-width="2"><path d="M160 69H270Q310 69 310 109V168"/><path d="M160 195H310"/><path d="M160 321H270Q310 321 310 281V222"/></g><circle cx="310" cy="195" r="18" fill="#5e6ad2"/><path d="M328 195H404" stroke="#828fff" stroke-width="2"/>
          <g font-family="Inter, sans-serif"><rect x="404" y="130" width="238" height="58" rx="10" fill="#141516" stroke="#34343a"/><text x="428" y="154" fill="#8a8f98" font-size="11">IDENTICAL RANKING SHA-256</text><text x="428" y="174" fill="#d0d6e0" font-size="12">__SEED_HASH_SHORT__…</text><rect x="404" y="204" width="238" height="58" rx="10" fill="#141516" stroke="#34343a"/><text x="428" y="228" fill="#8a8f98" font-size="11">IDENTICAL OOF SHA-256</text><text x="428" y="248" fill="#d0d6e0" font-size="12">__OOF_HASH_SHORT__…</text></g>
        </svg>
        <div class="seed-copy"><h3>Seedy 404, 405 i 406 nie są niezależnymi realizacjami.</h3><p>Rankingi i predykcje OOF są bajtowo identyczne. Raport traktuje je jako jeden deterministyczny scenariusz.</p><p><code>ranking __SEED_HASH__</code><br><code>OOF __OOF_HASH__</code></p></div>
      </div>
      <details class="evidence-details"><summary>Pełne hashe i cell IDs</summary><div class="table-wrap"><table class="compact"><thead><tr><th>Seed</th><th>Cell ID</th><th>Ranking SHA-256</th><th>OOF SHA-256</th></tr></thead><tbody>__SEED_ROWS__</tbody></table></div></details>
    </div>
  </section>

  <section class="section" id="zasady">
    <div class="section-heading"><p class="section-kicker">Integralność eksperymentu</p><h2>Wynik nie może przeciekać do decyzji.</h2></div>
    <div class="wide-width reveal">
      <div class="stats-grid"><div class="stat"><strong>__COMPLETED__/185</strong><span>wymaganych komórek primary</span></div><div class="stat"><strong>__FAILED__</strong><span>nieudanych wymaganych komórek</span></div><div class="stat"><strong>__COMPARISONS__</strong><span>porównań H1/H3/H5/H6/H7</span></div><div class="stat"><strong>__PATCHES__</strong><span>zwalidowanych patchy PanNuke</span></div></div>
      <div class="rules"><div class="rule"><b>Group-safe split</b><p>Podział wyłącznie po <code>group_id</code>, co najmniej na poziomie źródłowego patcha.</p></div><div class="rule"><b>Out-of-fold ranking</b><p>Główne modelowe oceny audytora pochodzą z group-safe OOF.</p></div><div class="rule"><b>Final reference fold</b><p>Pozostaje nietknięty, niekorumpowany i niedostępny dla wyboru lub strojenia.</p></div><div class="rule"><b>Rozdzielone etykiety</b><p><code>pre_corruption_label</code>, <code>observed_label</code>, flaga i metadane korupcji są przechowywane osobno.</p></div></div>
    </div>
  </section>

  <section class="section" id="qc">
    <div class="section-heading"><p class="section-kicker">Kontrola jakości PanNuke</p><h2>Maski pozostały źródłowe.</h2><p class="section-deck">Bez arbitralnego wyboru klasy w overlap, bez wymuszania background jako dokładnego dopełnienia i bez zmiany danych wejściowych.</p></div>
    <div class="wide-width reveal">
      <div class="stats-grid"><div class="stat"><strong>__OVERLAP_PIXELS__</strong><span>pikseli cross-class overlap</span></div><div class="stat"><strong>__OVERLAP_PATCHES__</strong><span>patchy z overlap</span></div><div class="stat"><strong>__VOID_PIXELS__</strong><span>pikseli unlabeled/void</span></div><div class="stat"><strong>__FLAGGED_INSTANCES__</strong><span>flagowanych instancji</span></div></div>
      <details class="evidence-details"><summary>Otwórz deterministyczny overlay QC</summary><div class="qc-frame"><div class="qc-toolbar"><span>1512 x 3840 px · oryginalny plik</span><a href="pannuke_mask_qc_overlays.png" target="_blank" rel="noopener">Otwórz w pełnej rozdzielczości ↗</a></div><img loading="lazy" src="pannuke_mask_qc_overlays.png" alt="Deterministyczny zestaw overlay QC PanNuke pokazujący obrazy i granice masek"></div></details>
    </div>
  </section>

  <section class="section" id="dowody">
    <div class="section-heading"><p class="section-kicker">Dowody i pochodzenie</p><h2>Każdy prezentowany wynik pozostaje sprawdzalny.</h2><p class="section-deck">Tabela zachowuje komplet 36 porównań bez wyboru według wyniku.</p></div>
    <div class="wide-width reveal">
      <details class="evidence-details" open><summary>Pełna tabela H1/H3/H5/H6/H7</summary>
        <div class="table-tools"><label>Hipoteza<select id="filter-hypothesis"><option value="ALL">Wszystkie</option><option>H1</option><option>H3</option><option>H5</option><option>H6</option><option>H7</option></select></label><label>Status<select id="filter-status"><option value="ALL">Wszystkie</option><option value="reported">reported</option><option value="not_available_frozen_optional_cell">unavailable</option></select></label><label>Szukaj<input id="filter-search" type="search" placeholder="comparison ID"></label><span class="table-count" id="table-count">__COMPARISONS__ / __COMPARISONS__ wierszy</span></div>
        <div class="table-wrap"><table id="comparison-table"><thead><tr><th>Comparison ID</th><th>Status</th><th>Metoda A</th><th>Metoda B</th><th>Różnica</th><th>95% CI bootstrap</th><th>Holm p (jednostronne)</th><th>Bootstrap</th></tr></thead><tbody>__COMPARISON_ROWS__</tbody></table></div>
      </details>
      <p class="fine"><strong>Uwaga statystyczna:</strong> p są jednostronne i korygowane metodą Holm. 95% CI jest zapisanym przedziałem percentylowym bootstrap; oba podsumowania pozostają widoczne.</p>
      <div class="provenance" style="margin-top:72px">
        <div><h3>Źródła i granice</h3><ul><li><a href="https://arxiv.org/abs/2003.10778">PanNuke extension</a> — dane i klasy jąder.</li><li><a href="https://arxiv.org/abs/1911.00068">Confident Learning</a> — metodologia jakości etykiet.</li><li><a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/fc20ea8d104cab737a5561096f9bde9b-Abstract.html">AQuA</a> — DOI 10.52202/075280-3494.</li><li><a href="https://arxiv.org/abs/2102.09099">NuCLS</a> — możliwa przyszła walidacja.</li></ul><p>Confirmatory, M9, blinded expert review i external validation pozostają pracą przyszłą.</p></div>
        <div><h3>Reprodukowalność</h3><dl class="hash-list"><div><dt>Run</dt><dd>__RUN_ID__</dd></div><div><dt>Artifact root SHA-256</dt><dd>__ARTIFACT_ROOT__</dd></div><div><dt>Stage attestation SHA-256</dt><dd>__STAGE_HASH__</dd></div><div><dt>QC overlay SHA-256</dt><dd>__QC_HASH__</dd></div></dl></div>
      </div>
    </div>
  </section>
</main>

<footer><div class="footer-inner"><span>AANCA · non-diagnostic research prototype</span><span>Lokalne dowody CC BY-NC-SA 4.0 dotyczą katalogu <code>masks/</code>; użycie danych ograniczono do badań niekomercyjnych.</span></div></footer>
<script>__SCRIPT__</script>
</body>
</html>
"""

    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="AANCA is a reproducible, non-diagnostic framework for ranking potentially inconsistent nucleus annotations for expert review.">
  <meta name="author" content="Natan Smogór">
  <meta name="date" content="2026-08-18">
  <meta name="theme-color" content="#010102">
  <meta name="referrer" content="no-referrer">
  <link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&amp;family=JetBrains+Mono:wght@400;500&amp;display=swap" rel="stylesheet">
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%23010102'/%3E%3Crect x='14' y='14' width='12' height='12' rx='3' fill='%235e6ad2'/%3E%3Crect x='38' y='14' width='12' height='12' rx='3' fill='%23828fff'/%3E%3Crect x='26' y='26' width='12' height='12' rx='3' fill='%235e6ad2'/%3E%3Crect x='14' y='38' width='12' height='12' rx='3' fill='%23828fff'/%3E%3Crect x='38' y='38' width='12' height='12' rx='3' fill='%235e6ad2'/%3E%3C/svg%3E">
  <title>AANCA — annotation auditing for expert review</title>
  <style>__CSS__</style>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="site-header" id="site-header">
  <div class="nav-shell">
    <a class="brand" href="#top" aria-label="AANCA — back to the beginning"><span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span><span class="brand-word-clip"><span class="brand-label">AANCA</span></span></a>
    <button class="menu-button" id="menu-button" type="button" aria-expanded="false" aria-controls="nav-links">Menu</button>
    <nav class="nav-links" id="nav-links" aria-label="Presentation sections"><a href="#overview">Study</a><a href="#method">Method</a><a href="#results">Results</a><a href="#evidence">Evidence</a><a href="#use">Use</a></nav>
  </div>
</header>

<section class="hero" id="top" aria-labelledby="hero-title">
  <canvas id="hero-canvas" aria-hidden="true"></canvas>
  <div class="hero-shell">
    <div class="hero-copy">
      <p class="eyebrow hero-animate">Automated nucleus-annotation auditing</p>
      <h1 id="hero-title" class="hero-animate">Which annotations deserve a second look?</h1>
      <p class="hero-lead hero-animate">A reproducible, group-safe framework that ranks each <em>potentially inconsistent annotation</em> and recommends the highest-priority cases for expert review—without rewriting the source labels.</p>
      <div class="hero-byline hero-animate"><span>Research prototype by <strong>Natan Smogór</strong></span><span>Released <time datetime="2026-08-18">18 August 2026</time></span></div>
    </div>
    <span class="hero-visual-label" aria-hidden="true">Source annotations stay fixed → review evidence is ranked<br>Conceptual workflow · not benchmark data</span>
  </div>
</section>

<main id="main">
  <section class="narrative" id="overview">
    <div class="article-copy reveal">
      <p class="lede">Annotation auditing is a way to decide what a human should inspect first. It is not a way to replace the human decision.</p>
      <p>Large histopathology datasets contain many already segmented nucleus instances, each paired with a class label. Even careful annotation can include ambiguity, inconsistent conventions or ordinary data-entry noise. Reviewing every instance again is expensive, so the useful question is not “can a model declare a label wrong?” It is “can a model create a better review queue than random sampling?”</p>
      <p>AANCA evaluates that question under controlled conditions. It intentionally changes a known subset of class labels, hides that intervention from the auditor, and then measures whether the changed labels move toward the front of a fixed review queue. Because the benchmark records exactly what it changed, retrieval can be scored without pretending that model disagreement is biological truth.</p>
      <p>The <strong>pre-corruption label is an experimental reference label, not guaranteed biological truth</strong>. It is used only to define the controlled benchmark, simulated restoration and final evaluation. In a real audit, a high score means <em>recommended for expert review</em>—never “confirmed error.”</p>
      <div class="scope-note"><strong>Study boundary.</strong> This presentation reports the completed primary frozen-feature benchmark. Confirmatory CNN experiments, blinded expert review and external validation have not been performed. Model-selection rules were frozen before outcome interpretation; the accepted analysis remains exploratory because outcomes were subsequently exposed during technical recovery.</div>
    </div>
  </section>

  <section class="study-at-a-glance" aria-labelledby="study-title">
    <div class="section-heading reveal"><p class="section-kicker">Study at a glance</p><h2 id="study-title">The exact benchmark, in five facts.</h2><p class="section-deck">These details define what the results do—and do not—measure.</p></div>
    <div class="study-specs reveal">
      <article class="spec-card"><span>Data</span><strong>PanNuke</strong><small>Verified official release; five positive nucleus classes across 19 tissue types.</small></article>
      <article class="spec-card"><span>Unit</span><strong>Already segmented nuclei</strong><small>Class-label consistency only. Segmentation quality and diagnosis are outside scope.</small></article>
      <article class="spec-card"><span>Primary model</span><strong>Frozen ResNet-18 + logistic regression</strong><small>ImageNet context embeddings with balanced multinomial fitting.</small></article>
      <article class="spec-card"><span>Prediction design</span><strong>Five-fold group-safe OOF</strong><small>A scored nucleus and its whole source patch are absent from its training fold.</small></article>
      <article class="spec-card"><span>Review budget</span><strong>5% primary queue</strong><small>Guided and random review receive the same integer budget.</small></article>
    </div>
  </section>

  <div class="research-question reveal">
    <span>Research question</span>
    <p>Can source-group-safe out-of-fold models retrieve controlled label inconsistencies more efficiently than random review—and does restoring the highest-ranked injected corruptions improve downstream nucleus classification?</p>
  </div>

  <section class="journey story" id="method" aria-labelledby="journey-title" data-active-stage="0">
    <div class="journey-sticky"><div class="journey-grid">
      <div class="journey-visual">
        <svg viewBox="0 0 760 680" role="img" aria-labelledby="method-graphic-title method-graphic-desc">
          <title id="method-graphic-title">The five cumulative stages of the controlled AANCA benchmark</title>
          <desc id="method-graphic-desc">Five horizontal lanes unfold along an alternating serpentine path from immutable source labels to an expert-review recommendation.</desc>

          <path class="journey-connector" pathLength="1" d="M688 81H711Q732 81 732 102V188Q732 209 711 209H722"/>
          <path class="journey-connector" pathLength="1" d="M72 209H49Q28 209 28 230V316Q28 337 49 337H38"/>
          <path class="journey-connector" pathLength="1" d="M688 337H711Q732 337 732 358V444Q732 465 711 465H722"/>
          <path class="journey-connector" pathLength="1" d="M72 465H49Q28 465 28 486V572Q28 593 49 593H38"/>

          <g class="journey-stage-group" data-journey-stage="0">
            <rect class="journey-lane-bg" x="38" y="34" width="650" height="94" rx="28"/>
            <path class="journey-lane-edge" d="M66 35H660"/>
            <text class="journey-stage-no" x="66" y="65">01</text>
            <text class="journey-stage-title" x="111" y="66">Preserve the source evidence</text>
            <text class="journey-stage-meta" x="111" y="91">LABEL + PATCH GROUP + ORIGINAL CLASS</text>
            <path class="journey-rail" d="M430 82H590"/>
            <path class="journey-rail" d="M583 76L590 82L583 88"/>
            <g transform="translate(612 59)"><rect class="journey-node" width="16" height="16" rx="4"/><rect class="journey-node" x="21" y="7" width="16" height="16" rx="4" opacity=".7"/><rect class="journey-node" x="7" y="25" width="16" height="16" rx="4" opacity=".85"/></g>
          </g>

          <g class="journey-stage-group" data-journey-stage="1">
            <rect class="journey-lane-bg" x="72" y="162" width="650" height="94" rx="28"/>
            <path class="journey-lane-edge" d="M100 163H694"/>
            <text class="journey-stage-no" x="100" y="193">02</text>
            <text class="journey-stage-title" x="145" y="194">Create known inconsistencies</text>
            <text class="journey-stage-meta" x="145" y="219">PRE-CORRUPTION ≠ OBSERVED · METADATA KEPT SEPARATE</text>
            <g transform="translate(596 190)"><rect class="journey-node-muted" width="38" height="10" rx="5"/><path class="journey-rail-accent" d="M44 5H58"/><rect class="journey-node" x="64" width="38" height="10" rx="5"/></g>
          </g>

          <g class="journey-stage-group" data-journey-stage="2">
            <rect class="journey-lane-bg" x="38" y="290" width="650" height="94" rx="28"/>
            <path class="journey-lane-edge" d="M66 291H660"/>
            <text class="journey-stage-no" x="66" y="321">03</text>
            <text class="journey-stage-title" x="111" y="322">Predict without patch leakage</text>
            <text class="journey-stage-meta" x="111" y="347">FIVE GROUP-SAFE OUT-OF-FOLD MODELS</text>
            <g transform="translate(564 316)"><rect class="journey-node" width="24" height="24" rx="5"/><rect class="journey-node-muted" x="31" width="24" height="24" rx="5"/><rect class="journey-node-muted" x="62" width="24" height="24" rx="5"/><rect class="journey-node-muted" x="93" width="24" height="24" rx="5"/></g>
          </g>

          <g class="journey-stage-group" data-journey-stage="3">
            <rect class="journey-lane-bg" x="72" y="418" width="650" height="94" rx="28"/>
            <path class="journey-lane-edge" d="M100 419H694"/>
            <text class="journey-stage-no" x="100" y="449">04</text>
            <text class="journey-stage-title" x="145" y="450">Turn evidence into a fixed review queue</text>
            <text class="journey-stage-meta" x="145" y="475">HIGHER SCORE = EARLIER REVIEW · PRIMARY BUDGET 5%</text>
            <g transform="translate(594 442)"><rect class="journey-node" width="104" height="7" rx="3.5"/><rect class="journey-node-muted" y="15" width="73" height="7" rx="3.5"/><rect class="journey-node-muted" y="30" width="42" height="7" rx="3.5"/></g>
          </g>

          <g class="journey-stage-group" data-journey-stage="4">
            <rect class="journey-lane-bg" x="38" y="546" width="650" height="94" rx="28"/>
            <path class="journey-lane-edge" d="M66 547H660"/>
            <text class="journey-stage-no" x="66" y="577">05</text>
            <text class="journey-stage-title" x="111" y="578">Measure retrieval; leave judgement to an expert</text>
            <text class="journey-stage-meta" x="111" y="603">RECOMMENDATION ONLY · SOURCE LABELS REMAIN UNCHANGED</text>
            <g transform="translate(607 564)"><circle class="journey-icon-line" cx="24" cy="24" r="22"/><path class="journey-icon-line" d="M13 24L21 32L37 12" style="stroke:#828fff;stroke-width:3"/></g>
          </g>
        </svg>
      </div>

      <div class="journey-copy">
        <p class="section-kicker">How the benchmark works</p>
        <h2 id="journey-title" class="display">One controlled question, unfolded step by step.</h2>
        <p class="journey-intro">Each stage adds evidence while preserving the source record. Scroll to build the complete benchmark; the graphic remains fully visible on mobile and in reduced-motion mode.</p>
        <ol class="journey-steps">
          <li class="journey-step is-active" data-journey-copy="0" data-index="01"><small>Reference</small><h3>Keep label states immutable and separate</h3><p>Each nucleus retains its source patch, experimental reference class and observed class.</p></li>
          <li class="journey-step" data-journey-copy="1" data-index="02"><small>Intervention</small><h3>Inject a change the benchmark can verify</h3><p>Four frozen mechanisms create objective positives without changing the raw source files.</p></li>
          <li class="journey-step" data-journey-copy="2" data-index="03"><small>Prediction</small><h3>Score only with unseen source groups</h3><p>No model may train on the scored nucleus or any other nucleus from its source patch.</p></li>
          <li class="journey-step" data-journey-copy="3" data-index="04"><small>Triage</small><h3>Rank suspicion under an equal budget</h3><p>Guided and random review inspect the same number of cases; only their ordering differs.</p></li>
          <li class="journey-step" data-journey-copy="4" data-index="05"><small>Evaluation</small><h3>Measure retrieval, then defer the decision</h3><p>Average precision and fixed-budget metrics score the queue. An expert still decides what a case means.</p></li>
        </ol>
        <div aria-hidden="true" style="height:1px;background:#23252a;margin-top:22px;overflow:hidden"><span id="journey-indicator" style="display:block;width:100%;height:1px;background:#5e6ad2;transform:scaleX(.02);transform-origin:left"></span></div>
      </div>
    </div></div>
  </section>

  <section class="section" id="reading">
    <div class="section-heading reveal"><p class="section-kicker">How to read the evidence</p><h2>Four quantities answer four different questions.</h2><p class="section-deck">Separating them prevents a strong ranking result from being mistaken for clinical or downstream utility.</p></div>
    <div class="reading-grid reveal">
      <article class="reading-card"><span>01</span><h3>Average precision</h3><p>Asks whether injected label changes appear early across the entire ranked list. Higher is better; at 0% corruption it is not applicable.</p></article>
      <article class="reading-card"><span>02</span><h3>Recall at 5%</h3><p>Asks how many injected changes are found when an expert can review only the first 5% of the queue.</p></article>
      <article class="reading-card"><span>03</span><h3>Random-review baseline</h3><p>Uses the identical integer budget over 100 deterministic repetitions, making the operational comparison fair.</p></article>
      <article class="reading-card"><span>04</span><h3>Restoration utility</h3><p>Asks a separate downstream question: after simulated review, does restoring found injected changes improve final-fold macro-F1?</p></article>
    </div>
  </section>

  <section class="section" id="results">
    <div class="section-heading reveal"><p class="section-kicker">Primary downstream test · H4</p><h2>Better triage did not improve the downstream model.</h2><p class="section-deck">This is the most important negative result. Ranking injected inconsistencies and improving a later classifier are related, but they are not equivalent objectives.</p></div>
    <div class="chapter-copy reveal"><p>The H4 experiment restored exactly the same 5% review budget under audit-guided and random selection, then trained the same downstream model and evaluated it on the untouched final reference fold. Audit-guided restoration was not favoured.</p></div>
    <div class="figure-width reveal result-layout">
      <div class="result-lead"><div class="result-label"><span>Audit-guided minus random review</span><span>Adverse to the registered hypothesis</span></div><span class="result-number">__H4_POINT__</span><p>Audit-guided restoration produced macro-F1 <strong>__H4_AUDIT__</strong>, below the random-review mean of <strong>__H4_RANDOM__</strong>.</p><div class="result-interpretation"><b>Plain-language interpretation</b><p>The audit-guided result was <strong>__H4_POINT_PP__ percentage points lower</strong>. The saved 95% interval remained below zero, so this registered test did not support downstream benefit.</p></div></div>
      __H4_CHART__
    </div>
  </section>

  <section class="section" id="benchmarks">
    <div class="section-heading reveal"><p class="section-kicker">Preregistered questions · H1-H7</p><h2>What the study actually learned.</h2><p class="section-deck">All seven questions remain visible together: supportive, qualified, adverse, neutral and unavailable. Across the 36 preregistered comparison entries, <strong>__REPORTED__</strong> have numeric results and <strong>__UNAVAILABLE__</strong> remain explicitly unavailable.</p></div>
    <div class="figure-width reveal"><div class="hypothesis-ledger">__HYPOTHESIS_LEDGER__</div></div>
    <div class="section-heading reveal" style="margin-top:var(--section-space)"><p class="section-kicker">Comparison atlas</p><h2>Every preregistered comparison entry.</h2><p class="section-deck">Each point is an average-precision difference; each line is its saved two-sided 95% percentile-bootstrap interval. Missing H6 points are shown as unavailable rather than as zero.</p></div>
    <div class="wide-width reveal"><div class="forest-plot" role="group" aria-label="Forest plot of all H1, H3, H5, H6 and H7 comparisons">__FOREST_PLOT__</div><p class="fine"><strong>Statistical reading.</strong> Registered Holm-adjusted p-values are one-sided, while the displayed 95% intervals are two-sided summaries. Those two summaries need not produce identical verbal labels.</p></div>
  </section>

  <section class="section" id="subgroups">
    <div class="section-heading reveal"><p class="section-kicker">H2 · descriptive heterogeneity</p><h2>Saved performance estimates varied across contexts.</h2><p class="section-deck">The saved subgroup estimates span nucleus classes, tissues, corruption mechanisms and corruption rates. They describe where performance differed in this benchmark; they do not establish a causal biological dependency.</p></div>
    <div class="chapter-copy reveal"><p>Subgroup average precision is reported only when the preregistered support rule is met: at least 100 samples and at least 10 injected corruptions. Otherwise the study retains counts without inventing an unstable estimate.</p></div>
    <div class="figure-width reveal">__H2_CHART__</div>
  </section>

  <section class="section" id="seed-audit">
    <div class="section-heading reveal"><p class="section-kicker">Reproducibility disclosure</p><h2>Three registered seeds produced one realisation.</h2><p class="section-deck">The three rows are retained for auditability, but the byte-identical files are not independent realisations and must not be interpreted as three replications.</p></div>
    <div class="figure-width reveal">
      <div class="seed-panel">
        <svg viewBox="0 0 680 390" role="img" aria-labelledby="seed-title seed-desc"><title id="seed-title">Seeds 404, 405 and 406 produced byte-identical files</title><desc id="seed-desc">Three seed paths converge on one ranking hash and one out-of-fold prediction hash.</desc>
          <g font-family="Inter, sans-serif" font-size="13" fill="#d0d6e0"><rect x="34" y="42" width="126" height="54" rx="10" fill="#141516" stroke="#34343a"/><text x="72" y="75">Seed 404</text><rect x="34" y="168" width="126" height="54" rx="10" fill="#141516" stroke="#34343a"/><text x="72" y="201">Seed 405</text><rect x="34" y="294" width="126" height="54" rx="10" fill="#141516" stroke="#34343a"/><text x="72" y="327">Seed 406</text></g>
          <g fill="none" stroke="#5e6ad2" stroke-width="2"><path d="M160 69H270Q310 69 310 109V168"/><path d="M160 195H310"/><path d="M160 321H270Q310 321 310 281V222"/></g><circle cx="310" cy="195" r="18" fill="#5e6ad2"/><path d="M328 195H404" stroke="#828fff" stroke-width="2"/>
          <g font-family="Inter, sans-serif"><rect x="404" y="130" width="238" height="58" rx="10" fill="#141516" stroke="#34343a"/><text x="428" y="154" fill="#8a8f98" font-size="11">IDENTICAL RANKING SHA-256</text><text x="428" y="174" fill="#d0d6e0" font-size="12">__SEED_HASH_SHORT__…</text><rect x="404" y="204" width="238" height="58" rx="10" fill="#141516" stroke="#34343a"/><text x="428" y="228" fill="#8a8f98" font-size="11">IDENTICAL OOF SHA-256</text><text x="428" y="248" fill="#d0d6e0" font-size="12">__OOF_HASH_SHORT__…</text></g>
        </svg>
        <div class="seed-copy"><h3>One deterministic output, preserved three times.</h3><p>The complete ranking and out-of-fold prediction files match byte for byte. That limitation narrows how the instance-dependent results should be interpreted.</p><p><code>ranking __SEED_HASH__</code><br><code>OOF __OOF_HASH__</code></p></div>
      </div>
      <div class="always-visible-evidence"><div class="evidence-label-row"><span>Complete seed identities</span><span>Always visible · exact traceability</span></div><div class="table-wrap"><table class="compact"><thead><tr><th>Seed</th><th>Cell ID</th><th>Ranking SHA-256</th><th>OOF SHA-256</th></tr></thead><tbody>__SEED_ROWS__</tbody></table></div></div>
    </div>
  </section>

  <section class="section" id="integrity">
    <div class="section-heading reveal"><p class="section-kicker">Experimental integrity</p><h2>The design limits outcome-informed model selection.</h2><p class="section-deck">The model, split, feature and statistical rules were frozen before outcome interpretation. Because outcomes were later exposed during recovery, this accepted result is reported as exploratory rather than as an untouched confirmatory analysis.</p></div>
    <div class="wide-width reveal">
      <div class="stats-grid"><div class="stat"><strong>__COMPLETED__/185</strong><span>required primary cells completed</span></div><div class="stat"><strong>__FAILED__</strong><span>failed required cells</span></div><div class="stat"><strong>__COMPARISONS__</strong><span>preregistered comparison entries retained</span></div><div class="stat"><strong>__BOOTSTRAPS__</strong><span>paired group-bootstrap iterations</span></div></div>
      <div class="rules"><div class="rule"><b>Group-safe splitting</b><p>Every split uses <code>group_id</code>, at least at source-patch level—never individual nuclei.</p></div><div class="rule"><b>Out-of-fold ranking</b><p>Primary model-based audit scores come from predictions made for source groups excluded from model fitting.</p></div><div class="rule"><b>Untouched final reference</b><p>The final reference fold is uncorrupted and unavailable for model selection, calibration or review-budget tuning.</p></div><div class="rule"><b>Separate label states</b><p><code>pre_corruption_label</code>, <code>observed_label</code>, <code>is_injected_corruption</code> and corruption metadata remain distinct.</p></div></div>
    </div>
  </section>

  <section class="section" id="interpretation">
    <div class="section-heading reveal"><p class="section-kicker">Interpretation boundary</p><h2>What this benchmark supports—and what it does not.</h2><p class="section-deck">A technically strong audit can still be limited in scope. The boundary below is part of the result, not a disclaimer added afterward.</p></div>
    <div class="figure-width reveal claim-grid">
      <article class="claim-card"><h3>Supported by the controlled benchmark</h3><ul><li>Registered group-safe rankings retrieved injected class-label changes more efficiently than random review in H1.</li><li>The preregistered fixed hybrid exceeded self-confidence in all 12 H5 comparisons.</li><li>Every displayed number remains traceable to sealed machine-readable evidence.</li></ul></article>
      <article class="claim-card"><h3>Not established by this study</h3><ul><li>That a naturally occurring annotation is wrong or that a pathologist made an error.</li><li>That better ranking automatically improves downstream classification; H4 was adverse.</li><li>Patient/WSI independence, clinical validity, expert agreement or external generalisation.</li></ul></article>
    </div>
  </section>

  <section class="section" id="quality">
    <div class="section-heading reveal"><p class="section-kicker">PanNuke quality control</p><h2>The source masks remained untouched.</h2><p class="section-deck">The validator measured cross-class overlaps and unlabeled regions, retained them in provenance and applied one frozen eligibility policy. It never arbitrated an overlap class or reconstructed the supplied background.</p></div>
    <div class="wide-width reveal">
      <div class="stats-grid"><div class="stat"><strong>__PATCHES__</strong><span>validated PanNuke patches</span></div><div class="stat"><strong>__OVERLAP_PIXELS__</strong><span>cross-class overlap pixels</span></div><div class="stat"><strong>__VOID_PIXELS__</strong><span>unlabeled / void pixels</span></div><div class="stat"><strong>__FLAGGED_INSTANCES__</strong><span>overlap-touching instances flagged</span></div></div>
      <div class="always-visible-evidence"><div class="evidence-label-row"><span>Deterministic quality-control overlay</span><span>1512 x 3840 px · source preview</span></div><div class="qc-frame"><div class="qc-toolbar"><span>Cropped representative preview · normal, overlap, void and exclusion cases</span><a href="pannuke_mask_qc_overlays.png" target="_blank" rel="noopener">Open the original image ↗</a></div><div class="qc-preview"><img loading="lazy" decoding="async" fetchpriority="low" width="1512" height="3840" src="pannuke_mask_qc_overlays.png" alt="Deterministic PanNuke quality-control overlays showing images and mask boundaries"></div></div></div>
    </div>
  </section>

  <section class="section" id="evidence">
    <div class="section-heading reveal"><p class="section-kicker">Evidence and provenance</p><h2>Every displayed result remains inspectable.</h2><p class="section-deck">The complete table is visible immediately and retains readable comparison names, exact raw identifiers, fixed precision, intervals, multiplicity-adjusted p-values and bootstrap counts.</p></div>
    <div class="wide-width reveal">
      <div class="always-visible-evidence"><div class="evidence-label-row"><span>Complete H1 / H3 / H5 / H6 / H7 comparison table</span><span>__REPORTED__ reported · __UNAVAILABLE__ unavailable</span></div>
        <div class="table-tools"><label>Hypothesis<select id="filter-hypothesis"><option value="ALL">All hypotheses</option><option>H1</option><option>H3</option><option>H5</option><option>H6</option><option>H7</option></select></label><label>Status<select id="filter-status"><option value="ALL">All statuses</option><option value="reported">Reported</option><option value="not_available_frozen_optional_cell">Unavailable</option></select></label><label>Search<input id="filter-search" type="search" placeholder="Name, seed or comparison ID"></label><span class="table-count" id="table-count">__COMPARISONS__ / __COMPARISONS__ rows</span></div>
        <div class="table-wrap" role="region" aria-label="Complete comparison results" tabindex="0"><table id="comparison-table"><colgroup><col style="width:7%"><col style="width:35%"><col style="width:11%"><col style="width:11%"><col style="width:19%"><col style="width:10%"><col style="width:7%"></colgroup><thead><tr><th>Hypothesis</th><th>Comparison</th><th>Status</th><th style="text-align:right">Δ AP</th><th style="text-align:right">95% bootstrap CI</th><th style="text-align:right">Holm-adjusted p</th><th style="text-align:right">Iterations</th></tr></thead><tbody>__COMPARISON_ROWS__</tbody></table></div>
      </div>
      <p class="fine"><strong>Statistical note.</strong> Holm-adjusted p-values are the registered one-sided tests. The saved 95% confidence intervals are percentile-bootstrap summaries based on whole-group resampling.</p>
      <div class="provenance" style="margin-top:64px">
        <div><h3>Research sources</h3><ul><li><a href="https://link.springer.com/chapter/10.1007/978-3-030-23937-4_2">Gamper et al. (2019), PanNuke</a> · DOI 10.1007/978-3-030-23937-4_2.</li><li><a href="https://arxiv.org/abs/2003.10778">Gamper et al. (2020), PanNuke extension</a> · dataset insights and baselines.</li><li><a href="https://www.jair.org/index.php/jair/article/view/12125">Northcutt, Jiang &amp; Chuang (2021), Confident Learning</a> · DOI 10.1613/jair.1.12125.</li><li><a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/fc20ea8d104cab737a5561096f9bde9b-Abstract.html">Goswami et al. (2023), AQuA</a> · DOI 10.52202/075280-3494.</li><li><a href="https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giac037/6586817">Amgad et al. (2022), NuCLS</a> · DOI 10.1093/gigascience/giac037.</li></ul><p>No patient- or WSI-level independence is claimed because the released metadata do not support it. Separation is guaranteed at source-patch level.</p></div>
        <div><h3>Reproducibility identities</h3><dl class="hash-list"><div><dt>Accepted run</dt><dd>__RUN_ID__</dd></div><div><dt>Artifact root SHA-256</dt><dd>__ARTIFACT_ROOT__</dd></div><div><dt>Stage attestation SHA-256</dt><dd>__STAGE_HASH__</dd></div><div><dt>QC overlay SHA-256</dt><dd>__QC_HASH__</dd></div></dl></div>
      </div>
    </div>
  </section>

  <section class="section" id="use">
    <div class="section-heading reveal"><p class="section-kicker">Open implementation</p><h2>Inspect the evidence first. Run the software when you need to.</h2><p class="section-deck">The checked-in presentation opens without a dataset or GPU. Package verification and the deterministic synthetic smoke path provide progressively deeper checks without claiming a new real-data study.</p></div>
    <div class="repo-shell">
      <article class="repo-card reveal">
        <div class="repo-top">
          <svg class="repo-icon" viewBox="0 0 24 24" role="img" aria-label="GitHub"><path fill="currentColor" d="M12 .7A11.3 11.3 0 0 0 8.43 22.72c.56.1.77-.24.77-.54v-2.11c-3.13.68-3.79-1.33-3.79-1.33-.51-1.3-1.25-1.65-1.25-1.65-1.02-.7.08-.69.08-.69 1.13.08 1.72 1.16 1.72 1.16 1 1.72 2.63 1.22 3.27.93.1-.73.39-1.22.71-1.5-2.5-.28-5.13-1.25-5.13-5.58 0-1.23.44-2.24 1.16-3.03-.12-.29-.5-1.44.11-2.99 0 0 .95-.3 3.11 1.16A10.8 10.8 0 0 1 12 6.16c.96 0 1.92.13 2.82.38 2.16-1.46 3.1-1.16 3.1-1.16.62 1.55.23 2.7.12 2.99.72.79 1.16 1.8 1.16 3.03 0 4.34-2.64 5.29-5.15 5.57.4.35.77 1.04.77 2.1v3.11c0 .3.2.65.78.54A11.3 11.3 0 0 0 12 .7Z"/></svg>
          <div><span class="repo-path">github.com / Jaqwilk</span><span class="repo-title">AANCA</span></div>
          <a class="repo-link" href="https://github.com/Jaqwilk/AANCA" target="_blank" rel="noopener">View repository ↗</a>
        </div>
        <p class="repo-description">The repository contains the scientific specification, preregistration, source code, tests, reproducible command-line workflows and this checksum-verifiable presentation. Raw PanNuke binaries, full runs, embeddings and checkpoints are intentionally not distributed.</p>
        <div class="repo-tags"><span>Python 3.12</span><span>group-safe OOF</span><span>immutable evidence</span><span>research only</span></div>
      </article>

      <div class="usage-grid">
        <article class="repo-command"><span>01 · Present</span><h3>Verify and open locally</h3><p>The dependency-free launcher checks every package hash, serves on loopback and opens the article.</p><pre>git clone https://github.com/Jaqwilk/AANCA.git
cd AANCA
python scripts/present_demo.py</pre></article>
        <article class="repo-command"><span>02 · Verify</span><h3>Check without a browser</h3><p>Use the same standard-library validation in CI, scripts or an offline review.</p><pre>python scripts/present_demo.py `
  --verify-only</pre></article>
        <article class="repo-command"><span>03 · Exercise</span><h3>Run the synthetic smoke path</h3><p>Installs the governed ML environment, then tests software without calling it PanNuke evidence.</p><pre>uv sync --dev
uv run histo-audit doctor
uv run histo-audit data generate-synthetic `
  --config configs\smoke.yaml
uv run histo-audit experiment smoke</pre></article>
      </div>
      <p class="usage-note">Real PanNuke execution is intentionally gated: obtain a lawful local copy from the official source, preserve it unchanged and follow <a href="https://github.com/Jaqwilk/AANCA/blob/main/DATASET_SETUP.md">DATASET_SETUP.md</a>. The repository does not silently download the dataset or relax scientific gates.</p>
    </div>
  </section>
</main>

<footer id="footer"><div class="footer-inner"><div class="footer-divider"></div>
  <div class="footer-grid">
    <div><div class="footer-brand-row"><span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span><span class="footer-brand">AANCA</span></div><p class="footer-summary">Automated auditing of nucleus class annotations: a university research prototype for prioritising potentially inconsistent annotations for expert review.</p></div>
    <div class="footer-column"><h3>Study</h3><p>Author: Natan Smogór</p><p>Release: 18 August 2026</p><p>Dataset: PanNuke</p><p>Primary frozen-feature benchmark completed; confirmatory and external validation pending.</p></div>
    <div class="footer-column"><h3>Responsible use</h3><p>Non-diagnostic research only</p><p>No automatic annotation changes</p><p>No claim that an annotator was wrong</p><p>No patient/WSI independence claim</p></div>
    <div class="footer-column"><h3>Inspect</h3><a href="evidence.json">Machine-readable evidence</a><a href="README.md">Presentation package notes</a><a href="https://github.com/Jaqwilk/AANCA" target="_blank" rel="noopener">GitHub repository ↗</a><a href="https://github.com/Jaqwilk/AANCA/blob/main/ETHICS_AND_LIMITATIONS.md" target="_blank" rel="noopener">Ethics and limitations ↗</a><a href="https://github.com/Jaqwilk/AANCA/blob/main/references/references.bib" target="_blank" rel="noopener">Bibliography ↗</a></div>
  </div>
  <div class="footer-bottom"><span>© 2026 Natan Smogór. This repository has no standalone general-purpose open-source licence; dataset files and pretrained weights retain their own terms.</span><span>Local PanNuke evidence applies CC BY-NC-SA 4.0 specifically to the release <code>masks/</code> directory. Project use is restricted to non-commercial research.</span></div>
</div></footer>
<script src="https://cdn.jsdelivr.net/npm/gsap@3.15.0/dist/gsap.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/gsap@3.15.0/dist/ScrollTrigger.min.js"></script>
<script>__SCRIPT__</script>
<script type="module">__THREE_SCRIPT__</script>
</body>
</html>
"""

    replacements = {
        "__CSS__": _MVP_CSS_V4,
        "__SCRIPT__": _MVP_SCRIPT_V4,
        "__THREE_SCRIPT__": _MVP_THREE_SCRIPT_V3,
        "__H4_POINT__": html.escape(_format_metric_v3(point_difference)),
        "__H4_POINT_PP__": html.escape(f"{abs(point_difference) * 100:.3f}"),
        "__H4_AUDIT__": html.escape(_format_metric_v3(h4["audit_guided_macro_f1"])),
        "__H4_RANDOM__": html.escape(_format_metric_v3(h4["random_review_macro_f1_mean"])),
        "__H4_CHART__": h4_chart,
        "__HYPOTHESIS_LEDGER__": hypothesis_ledger,
        "__FOREST_PLOT__": forest_plot,
        "__H2_CHART__": h2_chart,
        "__SEED_HASH_SHORT__": seed_hash[:16],
        "__OOF_HASH_SHORT__": oof_hash[:16],
        "__SEED_HASH__": seed_hash,
        "__OOF_HASH__": oof_hash,
        "__SEED_ROWS__": "".join(seed_rows),
        "__COMPLETED__": str(primary["completed_required_cells"]),
        "__FAILED__": str(primary["failed_required_cells"]),
        "__COMPARISONS__": str(primary["comparison_count"]),
        "__REPORTED__": str(reported_comparisons),
        "__UNAVAILABLE__": str(unavailable_comparisons),
        "__BOOTSTRAPS__": f"{primary['bootstrap_iterations_required']:,}",
        "__PATCHES__": f"{qc['patch_count']:,}",
        "__OVERLAP_PIXELS__": f"{qc['cross_class_overlap_pixel_count']:,}",
        "__VOID_PIXELS__": f"{qc['void_pixel_count']:,}",
        "__FLAGGED_INSTANCES__": f"{qc['overlap_touching_instance_count']:,}",
        "__COMPARISON_ROWS__": comparison_rows,
        "__RUN_ID__": html.escape(str(primary["run_id"])),
        "__ARTIFACT_ROOT__": html.escape(str(primary["artifact_root_sha256"])),
        "__STAGE_HASH__": html.escape(str(primary["stage_attestation_record_sha256"])),
        "__QC_HASH__": html.escape(str(qc["overlay_sha256"])),
    }
    for marker, replacement in replacements.items():
        if marker not in template:
            raise AssertionError(f"MVP template marker is absent: {marker}")
        template = template.replace(marker, replacement)
    return template


def _format_metric_v3(value: Any) -> str:
    """Format presentation metrics consistently without scientific notation."""

    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        return "[" + ", ".join(_format_metric_v3(item) for item in value) + "]"
    return str(value)


def _comparison_display_label_v3(comparison_id: str) -> str:
    hypothesis = _comparison_hypothesis(comparison_id)
    body = comparison_id.removeprefix(f"{hypothesis.lower()}_")
    seed = _comparison_seed(comparison_id)
    if seed:
        body = body.rsplit("_seed_", 1)[0]

    prefix_labels = (
        ("self_confidence_minus_random", "Self-confidence vs random review"),
        ("hybrid_minus_self_confidence", "Fixed hybrid vs self-confidence"),
        ("symmetric_minus_instance_dependent", "Symmetric vs instance-dependent corruption"),
        ("symmetric_minus_confusion", "Symmetric vs confusion-targeted corruption"),
        ("pathology_minus_imagenet", "Pathology encoder vs ImageNet"),
        ("highlighted_minus_context", "Target-highlighted vs context"),
    )
    mechanism_labels = {
        "confusion": "Confusion-targeted",
        "group_conditional": "Group-conditional",
        "instance_dependent": "Instance-dependent",
        "symmetric": "Symmetric",
    }
    label = body.replace("_", " ").title()
    qualifier = ""
    for prefix, candidate in prefix_labels:
        if body == prefix or body.startswith(f"{prefix}_"):
            label = candidate
            tail = body.removeprefix(prefix).removeprefix("_")
            qualifier = mechanism_labels.get(tail, tail.replace("_", " ").title())
            break

    parts = [label]
    if qualifier:
        parts.append(qualifier)
    if seed:
        parts.append(f"Seed {seed}")
    return " · ".join(parts)


def _render_comparison_rows_v3(comparisons: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for item in comparisons:
        comparison_id = str(item["comparison_id"])
        hypothesis = _comparison_hypothesis(comparison_id)
        is_reported = item.get("status") == "reported"
        status_label = "Reported" if is_reported else "Unavailable"
        status_class = "is-reported" if is_reported else "is-unavailable"
        rows.append(
            '<tr data-comparison-row data-hypothesis="'
            + html.escape(hypothesis)
            + '" data-status="'
            + html.escape(str(item["status"]))
            + '" data-seed="'
            + html.escape(_comparison_seed(comparison_id))
            + '">'
            + '<td data-label="Hypothesis"><span class="hypothesis-chip">'
            + html.escape(hypothesis)
            + "</span></td>"
            + '<td data-label="Comparison"><span class="comparison-name">'
            + html.escape(_comparison_display_label_v3(comparison_id))
            + '</span><code class="comparison-id">'
            + html.escape(comparison_id)
            + "</code></td>"
            + '<td data-label="Status"><span class="status-pill '
            + status_class
            + '">'
            + status_label
            + "</span></td>"
            + '<td class="numeric" data-label="Δ AP">'
            + html.escape(_format_metric_v3(item["point_difference"]))
            + "</td>"
            + '<td class="numeric" data-label="95% bootstrap CI">'
            + html.escape(_format_metric_v3(item["interval_95"]))
            + "</td>"
            + '<td class="numeric" data-label="Holm-adjusted p">'
            + html.escape(_format_metric_v3(item["p_value_holm"]))
            + "</td>"
            + '<td class="numeric" data-label="Iterations">'
            + html.escape(_format_metric_v3(item["valid_bootstrap_iterations"]))
            + "</td></tr>"
        )
    return "".join(rows)


def _render_forest_plot_v3(comparisons: list[dict[str, Any]]) -> str:
    bounds = [0.0]
    for item in comparisons:
        if item.get("status") != "reported":
            continue
        bounds.append(_require_real(item.get("point_difference"), role="forest point difference"))
        bounds.extend(_require_interval(item.get("interval_95"), role="forest interval"))
    lower = min(bounds)
    upper = max(bounds)
    padding = max((upper - lower) * 0.07, 0.001)
    lower -= padding
    upper += padding
    zero_position = _position_percent(0.0, lower, upper)

    output: list[str] = [
        '<div class="forest-axis" aria-hidden="true"><span>',
        html.escape(f"{lower:.3f}"),
        "</span><span>Average-precision difference · zero means no difference</span><span>",
        html.escape(f"{upper:.3f}"),
        "</span></div>",
    ]
    previous_group = ""
    for item in comparisons:
        comparison_id = str(item["comparison_id"])
        hypothesis = _comparison_hypothesis(comparison_id)
        if hypothesis != previous_group:
            output.append(
                f'<div class="forest-group"><span>{html.escape(hypothesis)}</span>'
                "<span>Point estimate and saved 95% bootstrap interval</span></div>"
            )
            previous_group = hypothesis
        label = _comparison_display_label_v3(comparison_id)
        if item.get("status") != "reported":
            output.append(
                '<div class="forest-row is-unavailable">'
                f'<span class="forest-label" title="{html.escape(comparison_id)}">'
                f"{html.escape(label)}</span>"
                '<span class="forest-track forest-track-empty">'
                "<span>Unavailable by frozen design</span></span>"
                '<span class="forest-value">—</span></div>'
            )
            continue
        point = _require_real(item.get("point_difference"), role="forest point difference")
        interval = _require_interval(item.get("interval_95"), role="forest interval")
        point_position = _position_percent(point, lower, upper)
        low_position = _position_percent(interval[0], lower, upper)
        high_position = _position_percent(interval[1], lower, upper)
        p_value = _require_real(item.get("p_value_holm"), role="forest Holm p")
        aria = (
            f"{label}: difference {point:.8g}; 95% CI {interval[0]:.8g} to "
            f"{interval[1]:.8g}; one-sided Holm-adjusted p {p_value:.8g}"
        )
        output.append(
            '<div class="forest-row">'
            f'<span class="forest-label" title="{html.escape(comparison_id)}">'
            f"{html.escape(label)}</span>"
            f'<span class="forest-track" style="--zero:{zero_position:.4f}%;'
            f"--low:{low_position:.4f}%;--high:{high_position:.4f}%;"
            f'--point:{point_position:.4f}%" role="img" aria-label="{html.escape(aria)}">'
            '<span class="forest-zero"></span><span class="forest-ci"></span>'
            '<span class="forest-point"></span></span>'
            f'<span class="forest-value">{html.escape(_format_metric_v3(point))}</span></div>'
        )
    return "".join(output)


def _render_h4_chart_v3(h4: dict[str, Any]) -> str:
    metrics = (
        (
            "Reference · uncorrupted",
            _require_real(
                h4["uncorrupted_reference_baseline_macro_f1"],
                role="H4 uncorrupted macro-F1",
            ),
        ),
        (
            "Baseline · corrupted labels",
            _require_real(
                h4["corrupted_observed_baseline_macro_f1"],
                role="H4 corrupted macro-F1",
            ),
        ),
        (
            "Random review · mean",
            _require_real(h4["random_review_macro_f1_mean"], role="H4 random macro-F1"),
        ),
        (
            "Audit-guided review",
            _require_real(h4["audit_guided_macro_f1"], role="H4 audit macro-F1"),
        ),
    )
    values = [value for _, value in metrics]
    metric_padding = max((max(values) - min(values)) * 0.08, 0.0005)
    metric_lower = min(values) - metric_padding
    metric_upper = max(values) + metric_padding
    metric_rows: list[str] = []
    for label, value in metrics:
        position = _position_percent(value, metric_lower, metric_upper)
        metric_rows.append(
            '<div class="metric-row"><span class="metric-name">'
            f"{html.escape(label)}</span>"
            f'<span class="metric-track" role="img" aria-label="{html.escape(label)}: '
            f'{value:.12g}"><span class="metric-dot" style="--x:{position:.4f}%">'
            "</span></span>"
            f"<strong>{value:.6f}</strong></div>"
        )

    difference = _require_real(h4["point_difference"], role="H4 point difference")
    interval = _require_interval(h4["interval_95"], role="H4 interval")
    diff_lower = min(interval[0], difference, 0.0)
    diff_upper = max(interval[1], difference, 0.0)
    diff_padding = max((diff_upper - diff_lower) * 0.12, 0.0002)
    diff_lower -= diff_padding
    diff_upper += diff_padding
    point_position = _position_percent(difference, diff_lower, diff_upper)
    low_position = _position_percent(interval[0], diff_lower, diff_upper)
    high_position = _position_percent(interval[1], diff_lower, diff_upper)
    zero_position = _position_percent(0.0, diff_lower, diff_upper)
    repetitions = _require_exact_int(h4["random_repetitions"], role="H4 repetitions")
    return (
        '<div class="h4-chart" role="group" aria-label="Complete registered H4 downstream result">'
        '<div class="chart-title-row"><div><span class="chart-label">Downstream outcome</span>'
        "<h3>Macro-F1 after equal-budget label restoration</h3></div>"
        '<span class="axis-warning">Truncated axis</span></div>'
        '<div class="metric-axis"><span>'
        + f"{metric_lower:.3f}"
        + "</span><span>Macro-F1</span><span>"
        + f"{metric_upper:.3f}"
        + "</span></div>"
        + "".join(metric_rows)
        + '<div class="difference-panel"><div class="difference-heading">'
        + "<span>Audit-guided minus random review</span>"
        + f"<strong>{difference:.6f}</strong></div>"
        + f'<div class="difference-track" style="--zero:{zero_position:.4f}%;'
        + f"--low:{low_position:.4f}%;--high:{high_position:.4f}%;"
        + f'--point:{point_position:.4f}%" role="img" aria-label="Difference '
        + f'{difference:.12g}; 95% CI {interval[0]:.12g} to {interval[1]:.12g}">'
        + '<span class="difference-zero"></span><span class="difference-ci"></span>'
        + '<span class="difference-point"></span></div>'
        + f'<div class="difference-axis"><span>{diff_lower:.4f}</span><span>0</span>'
        + f"<span>{diff_upper:.4f}</span></div>"
        + f"<p><strong>95% CI [{interval[0]:.6f}, {interval[1]:.6f}]</strong> · "
        + f"{repetitions} random-review repetitions</p></div></div>"
    )


def _render_h2_chart_v3(h2: dict[str, Any]) -> str:
    labels = {
        "class": "Nucleus class",
        "tissue": "Tissue type",
        "mechanism": "Corruption mechanism",
        "rate": "Corruption rate",
    }
    rows: list[str] = []
    for dimension, label in labels.items():
        summary = h2["dimensions"][dimension]
        interval = _require_interval(
            summary["reported_average_precision_range"], role=f"H2 {dimension} AP range"
        )
        low_position = 100.0 * interval[0]
        high_position = 100.0 * interval[1]
        not_applicable = int(summary["status_counts"].get("not_applicable_zero_corruption", 0))
        aria = (
            f"{label}: {summary['reported_count']} reportable estimates, AP range "
            f"{interval[0]:.6g} to {interval[1]:.6g}, {not_applicable} not applicable"
        )
        rows.append(
            '<div class="range-row"><span class="range-name">'
            f"{html.escape(label)}</span>"
            f'<span class="range-track" style="--low:{low_position:.4f}%;'
            f'--high:{high_position:.4f}%" role="img" aria-label="{html.escape(aria)}">'
            '<span class="range-line"></span><span class="range-start"></span>'
            '<span class="range-end"></span></span>'
            f'<span class="range-count"><strong>{summary["reported_count"]:,}</strong>'
            "<small>reported</small></span>"
            f'<span class="range-values">{interval[0]:.4f}-{interval[1]:.4f}</span></div>'
        )
    return (
        '<div class="h2-chart"><div class="chart-title-row"><div>'
        '<span class="chart-label">Descriptive heterogeneity</span>'
        "<h3>Observed average-precision ranges</h3></div></div>"
        '<div class="range-axis"><span>0 AP</span>'
        "<span>Range across saved subgroup estimates</span><span>1 AP</span></div>"
        + "".join(rows)
        + f'<p class="chart-note"><strong>{h2["reported_count"]:,}</strong> reportable '
        + f"estimates from <strong>{h2['row_count']:,}</strong> saved rows. These ranges "
        + "are descriptive—not an omnibus test and not a biological ranking.</p></div>"
    )


def _render_hypothesis_ledger_v3(
    hypotheses: dict[str, dict[str, Any]], h2: dict[str, Any], h4: dict[str, Any]
) -> str:
    h1, h3, h5, h6, h7 = (
        hypotheses["h1"],
        hypotheses["h3"],
        hypotheses["h5"],
        hypotheses["h6"],
        hypotheses["h7"],
    )
    entries = (
        (
            "H1",
            "Can ranking beat random review?",
            f"{h1['positive_point_difference_count']} / {h1['reported_count']} positive differences",
            "positive",
            f"All {h1['interval_excludes_zero_in_positive_direction_count']} saved 95% "
            "bootstrap intervals lie above zero. The instance-dependent seed caveat is "
            "reported separately below.",
        ),
        (
            "H2",
            "Does performance vary by context?",
            f"{h2['reported_count']:,} reportable estimates",
            "descriptive",
            "Saved estimates vary across nucleus class, tissue, corruption mechanism and "
            "corruption rate. This is descriptive heterogeneity, not a causal claim.",
        ),
        (
            "H3",
            "Are some corruption mechanisms harder?",
            f"{h3['positive_point_difference_count']} / {h3['reported_count']} positive differences",
            "qualified",
            f"One of {h3['reported_count']} saved 95% intervals crosses zero, even though "
            "the registered one-sided Holm-adjusted p-value is below 0.05.",
        ),
        (
            "H4",
            "Does audit-guided restoration improve downstream classification?",
            f"Δ macro-F1 {_format_metric_v3(h4['point_difference'])}",
            "adverse",
            "No. Audit-guided restoration underperformed the mean random-review result. "
            "The adverse registered result is retained without post-result tuning.",
        ),
        (
            "H5",
            "Does the fixed hybrid improve ranking?",
            f"{h5['positive_point_difference_count']} / {h5['reported_count']} positive differences",
            "positive",
            f"All {h5['interval_excludes_zero_in_positive_direction_count']} saved 95% "
            "bootstrap intervals lie above zero for fixed hybrid minus self-confidence.",
        ),
        (
            "H6",
            "Does a pathology-specific encoder outperform ImageNet?",
            f"{h6['reported_count']} / {h6['comparison_count']} comparisons available",
            "unavailable",
            "No eligible pathology encoder passed the frozen availability rule. No "
            "replacement was selected after inspecting results.",
        ),
        (
            "H7",
            "Does explicit target highlighting help?",
            f"{h7['interval_crosses_zero_count']} / {h7['reported_count']} intervals cross zero",
            "neutral",
            "The saved comparisons do not provide clear evidence that target highlighting "
            "improves average precision over context alone.",
        ),
    )
    return "".join(
        '<details class="hypothesis-row" data-status="'
        + status
        + '"><summary><span class="hypothesis-id">'
        + identifier
        + '</span><span class="hypothesis-title">'
        + html.escape(title)
        + '</span><span class="hypothesis-result">'
        + html.escape(result)
        + '</span><span class="hypothesis-toggle" aria-hidden="true">+</span></summary><p>'
        + html.escape(detail)
        + "</p></details>"
        for identifier, title, result, status, detail in entries
    )


def _render_hypothesis_ledger_v4(
    hypotheses: dict[str, dict[str, Any]], h2: dict[str, Any], h4: dict[str, Any]
) -> str:
    """Render every preregistered question as visible, non-collapsible evidence."""

    h1, h3, h5, h6, h7 = (
        hypotheses["h1"],
        hypotheses["h3"],
        hypotheses["h5"],
        hypotheses["h6"],
        hypotheses["h7"],
    )
    entries = (
        (
            "H1",
            "Detection",
            "Can ranking beat random review?",
            f"{h1['positive_point_difference_count']} / {h1['reported_count']} positive differences",
            "supported",
            f"All {h1['interval_excludes_zero_in_positive_direction_count']} saved 95% "
            "bootstrap intervals were above zero. The byte-identical "
            "instance-dependent seed outputs are disclosed separately and are not "
            "counted as independent replications.",
        ),
        (
            "H2",
            "Heterogeneity",
            "Did saved performance estimates vary across contexts?",
            f"{h2['reported_count']:,} reportable estimates",
            "descriptive",
            "Yes, the saved estimates varied across nucleus class, tissue, corruption "
            "mechanism and rate. This is descriptive heterogeneity; no omnibus test "
            "or causal biological interpretation was registered.",
        ),
        (
            "H3",
            "Difficulty",
            "Were some corruption mechanisms harder?",
            f"{h3['positive_point_difference_count']} / {h3['reported_count']} positive differences",
            "qualified",
            f"The registered contrasts generally placed confusion-targeted and "
            f"instance-dependent corruption below symmetric corruption. One of "
            f"{h3['reported_count']} saved 95% intervals crossed zero.",
        ),
        (
            "H4",
            "Utility",
            "Did guided restoration improve downstream classification?",
            f"Δ macro-F1 {_format_metric_v3(h4['point_difference'])}",
            "adverse",
            "No. At the same 5% review budget, audit-guided restoration performed "
            "below the mean random-review result. The adverse registered outcome is "
            "retained without post-result tuning.",
        ),
        (
            "H5",
            "Combination",
            "Did the fixed hybrid improve ranking?",
            f"{h5['positive_point_difference_count']} / {h5['reported_count']} positive differences",
            "supported",
            f"All {h5['interval_excludes_zero_in_positive_direction_count']} saved 95% "
            "bootstrap intervals were above zero for the preregistered fixed hybrid "
            "minus self-confidence comparison.",
        ),
        (
            "H6",
            "Representation",
            "Did a pathology-specific encoder outperform ImageNet?",
            f"{h6['reported_count']} / {h6['comparison_count']} numeric results",
            "unavailable",
            "This question was not estimated. No pathology encoder satisfied every "
            "frozen access, licence, reproducibility, hardware and smoke-test gate, "
            "and no replacement was chosen after outcomes were visible.",
        ),
        (
            "H7",
            "Target cue",
            "Did explicit target highlighting help?",
            f"{h7['interval_crosses_zero_count']} / {h7['reported_count']} intervals crossed zero",
            "neutral",
            "The saved comparisons do not provide clear evidence that target "
            "highlighting improved average precision over the context representation.",
        ),
    )
    return "".join(
        '<article class="hypothesis-row" data-status="'
        + status
        + '"><div class="hypothesis-row-head"><span class="hypothesis-id">'
        + identifier
        + '</span><span class="hypothesis-theme">'
        + html.escape(theme)
        + '</span><span class="hypothesis-result">'
        + html.escape(result)
        + '</span></div><h3 class="hypothesis-title">'
        + html.escape(title)
        + "</h3><p>"
        + html.escape(detail)
        + "</p></article>"
        for identifier, theme, title, result, status, detail in entries
    )


_MVP_CSS_V3 = r"""
:root {
  color-scheme: dark;
  --canvas: #010102;
  --surface-1: #0f1011;
  --surface-2: #141516;
  --surface-3: #18191a;
  --ink: #f7f8f8;
  --ink-muted: #d0d6e0;
  --ink-subtle: #8a8f98;
  --ink-tertiary: #62666d;
  --line: #23252a;
  --line-strong: #34343a;
  --accent: #5e6ad2;
  --accent-hover: #828fff;
  --sans: Inter, "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
  --prose: 760px;
  --figure: 1180px;
  --wide: 1320px;
  --page-pad: clamp(24px, 5vw, 72px);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; background: var(--canvas); }
body { margin: 0; overflow-x: hidden; color: var(--ink); background: var(--canvas); font: 400 15px/1.6 var(--sans); -webkit-font-smoothing: antialiased; }
::selection { color: #fff; background: var(--accent); }
a { color: inherit; text-underline-offset: 4px; text-decoration-color: var(--ink-tertiary); }
a:hover { color: var(--accent-hover); text-decoration-color: currentColor; }
a:focus-visible, button:focus-visible, summary:focus-visible, input:focus-visible, select:focus-visible { outline: 2px solid var(--accent-hover); outline-offset: 4px; }
code { overflow-wrap: anywhere; font-family: var(--mono); }
.skip { position: fixed; left: 16px; top: -80px; z-index: 100; padding: 9px 13px; border-radius: 8px; color: var(--canvas); background: var(--ink); transition: top .2s ease; }
.skip:focus { top: 14px; }
.site-header { position: fixed; inset: 0 0 auto; z-index: 50; height: 56px; border-bottom: 1px solid transparent; background: rgba(1,1,2,.9); backdrop-filter: blur(16px); }
.site-header.is-scrolled { border-color: var(--line); }
.nav-shell { width: min(var(--wide), calc(100% - 2 * var(--page-pad))); height: 100%; margin: auto; display: flex; align-items: center; justify-content: space-between; gap: 28px; }
.brand { display: flex; align-items: center; flex: 0 0 auto; gap: 10px; font-size: 14px; font-weight: 600; letter-spacing: -.02em; text-decoration: none; }
.brand-mark { width: 20px; height: 20px; display: grid; grid-template-columns: repeat(2,1fr); gap: 3px; transform: rotate(45deg); }
.brand-mark i { display: block; border-radius: 2px; background: var(--accent); }
.nav-links { display: flex; align-items: center; gap: 22px; }
.nav-links a { position: relative; color: var(--ink-subtle); font-size: 13px; font-weight: 500; text-decoration: none; }
.nav-links a::after { content: ""; position: absolute; left: 0; right: 0; bottom: -9px; height: 1px; transform: scaleX(0); background: var(--accent); }
.nav-links a:hover, .nav-links a.is-active { color: var(--ink); }
.nav-links a.is-active::after { transform: scaleX(1); }
.menu-button { display: none; min-width: 42px; min-height: 42px; border: 1px solid var(--line); border-radius: 8px; color: var(--ink); background: var(--surface-1); font: 500 13px var(--sans); }
.reading-progress { position: absolute; left: 0; bottom: -1px; width: 100%; height: 1px; transform: scaleX(0); transform-origin: left; background: var(--accent); }
.hero { position: relative; min-height: min(920px,100svh); padding: 56px var(--page-pad) 0; overflow: hidden; border-bottom: 1px solid var(--line); }
#hero-canvas { position: absolute; inset: 56px 0 0; width: 100%; height: calc(100% - 56px); opacity: .92; }
.hero-shell { position: relative; z-index: 1; width: min(var(--wide),100%); min-height: calc(min(920px,100svh) - 56px); margin: auto; display: grid; grid-template-columns: minmax(0,1.02fr) minmax(360px,.98fr); align-items: center; pointer-events: none; }
.hero-copy { max-width: 720px; padding: 72px 0; pointer-events: auto; }
.eyebrow, .section-kicker, .chart-label { margin: 0; color: var(--accent-hover); font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }
.hero h1 { max-width: 720px; margin: 20px 0 0; font-size: clamp(42px,5.4vw,72px); font-weight: 600; line-height: 1.02; letter-spacing: -.052em; text-wrap: balance; }
.hero-lead { max-width: 650px; margin: 26px 0 0; color: var(--ink-muted); font-size: clamp(17px,1.45vw,20px); line-height: 1.52; letter-spacing: -.012em; }
.hero-byline { display: flex; flex-wrap: wrap; gap: 8px 20px; margin-top: 34px; color: var(--ink-subtle); font-size: 12px; }
.hero-byline span { display: flex; align-items: center; gap: 8px; }
.hero-byline span + span::before { content: ""; width: 4px; height: 4px; border-radius: 1px; background: var(--accent); }
.hero-visual-label { position: absolute; right: 0; bottom: 34px; color: var(--ink-tertiary); font: 400 10px/1.5 var(--mono); letter-spacing: .05em; text-align: right; text-transform: uppercase; }
.prose, .section-heading { width: min(var(--prose), calc(100% - 2 * var(--page-pad))); margin-inline: auto; }
.figure-width { width: min(var(--figure), calc(100% - 2 * var(--page-pad))); margin-inline: auto; }
.wide-width { width: min(var(--wide), calc(100% - 2 * var(--page-pad))); margin-inline: auto; }
.intro { padding: clamp(92px,10vw,136px) 0; }
.intro-lead { margin: 0; color: var(--ink); font-size: clamp(24px,2.4vw,32px); font-weight: 500; line-height: 1.32; letter-spacing: -.032em; text-wrap: balance; }
.intro-copy { display: grid; grid-template-columns: repeat(2,1fr); gap: 48px; margin-top: 46px; }
.intro-copy p { margin: 0; color: var(--ink-subtle); font-size: 16px; line-height: 1.65; }
.scope-note { margin-top: 52px; padding: 22px 0 0; border-top: 1px solid var(--line); color: var(--ink-subtle); font-size: 13px; }
.scope-note strong { color: var(--ink); font-weight: 500; }
.question-panel { width: min(var(--figure), calc(100% - 2 * var(--page-pad))); margin: 0 auto clamp(92px,10vw,136px); display: grid; grid-template-columns: 220px 1fr; gap: 48px; padding: 34px; border: 1px solid var(--line); border-radius: 16px; background: var(--surface-1); }
.question-panel span { color: var(--accent-hover); font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }
.question-panel p { margin: 0; color: var(--ink-muted); font-size: clamp(18px,2vw,24px); font-weight: 500; line-height: 1.42; letter-spacing: -.02em; }
.section { padding: clamp(92px,10vw,136px) 0; border-top: 1px solid var(--line); scroll-margin-top: 56px; }
.section-kicker { display: flex; align-items: center; gap: 11px; margin-bottom: 16px; }
.section-kicker::before { content: ""; width: 24px; height: 1px; background: currentColor; }
h2, .display { margin: 0; font-size: clamp(34px,4.2vw,52px); font-weight: 600; line-height: 1.08; letter-spacing: -.042em; text-wrap: balance; }
.section-deck { margin: 22px 0 0; color: var(--ink-subtle); font-size: clamp(16px,1.55vw,19px); line-height: 1.58; }
.section-heading { margin-bottom: 52px; }
.story { position: relative; height: 480vh; border-top: 1px solid var(--line); scroll-margin-top: 56px; }
.story-frame { position: sticky; top: 0; min-height: 100vh; display: grid; place-items: center; overflow: hidden; }
.story-inner { width: min(var(--wide),calc(100% - 2 * var(--page-pad))); display: grid; grid-template-columns: minmax(520px,1.12fr) minmax(360px,.88fr); gap: clamp(48px,6vw,88px); align-items: center; }
.workflow-panel { min-height: 650px; display: grid; place-items: center; border: 1px solid var(--line); border-radius: 16px; background: var(--surface-1); overflow: hidden; }
.workflow-panel svg { width: 100%; height: auto; }
.workflow-panel [data-stage] { opacity: .16; transform-origin: center; }
.workflow-panel [data-stage].is-visible { opacity: .48; }
.workflow-panel [data-stage].is-current { opacity: 1; }
.workflow-node { fill: var(--surface-2); stroke: var(--line-strong); stroke-width: 1.25; }
.workflow-connector { fill: none; stroke: var(--line-strong); stroke-width: 1.5; stroke-linecap: round; }
.workflow-connector.is-accent { stroke: var(--accent); }
.workflow-step-no { fill: var(--accent-hover); font: 500 11px var(--mono); }
.workflow-title { fill: var(--ink); font: 600 15px var(--sans); letter-spacing: -.01em; }
.workflow-copy { fill: var(--ink-subtle); font: 400 11px var(--sans); }
.workflow-icon { fill: var(--accent); }
.story-copy .display { max-width: 460px; font-size: clamp(32px,3.4vw,46px); }
.story-steps { display: flex; flex-direction: column; gap: 0; margin: 30px 0 0; padding: 0; list-style: none; border-top: 1px solid var(--line); }
.story-step { position: relative; padding: 18px 0 18px 38px; border-bottom: 1px solid var(--line); opacity: .28; }
.story-step::before { content: attr(data-index); position: absolute; left: 0; top: 20px; color: var(--ink-tertiary); font: 500 11px var(--mono); }
.story-step.is-active { opacity: 1; }
.story-step.is-active::before { color: var(--accent-hover); }
.story-step h3 { margin: 0 0 5px; font-size: 15px; font-weight: 600; letter-spacing: -.015em; }
.story-step p { margin: 0; color: var(--ink-subtle); font-size: 13px; line-height: 1.5; }
.result-layout { display: grid; grid-template-columns: minmax(280px,.72fr) minmax(0,1.55fr); gap: 24px; align-items: stretch; }
.result-lead { padding: 30px; border: 1px solid var(--line-strong); border-radius: 16px; background: var(--surface-1); }
.result-label { display: flex; flex-direction: column; gap: 6px; color: var(--ink-subtle); font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: .07em; }
.result-label span:last-child { color: var(--accent-hover); }
.result-number { display: block; margin: 28px 0 16px; font: 500 clamp(42px,5vw,64px)/1 var(--sans); letter-spacing: -.055em; white-space: nowrap; }
.result-lead p { margin: 0; color: var(--ink-subtle); font-size: 14px; line-height: 1.6; }
.result-lead strong { color: var(--ink); font-weight: 500; }
.result-interpretation { margin-top: 24px; padding-top: 20px; border-top: 1px solid var(--line); }
.result-interpretation b { display: block; margin-bottom: 7px; color: var(--ink); font-size: 12px; font-weight: 600; }
.h4-chart, .h2-chart { padding: clamp(24px,3vw,38px); border: 1px solid var(--line); border-radius: 16px; background: var(--surface-1); }
.chart-title-row { display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; margin-bottom: 28px; }
.chart-title-row h3 { margin: 8px 0 0; font-size: 17px; font-weight: 600; letter-spacing: -.02em; }
.axis-warning { padding: 4px 7px; border: 1px solid var(--line-strong); border-radius: 4px; color: var(--ink-tertiary); font-size: 9px; letter-spacing: .07em; text-transform: uppercase; white-space: nowrap; }
.metric-axis, .range-axis, .forest-axis { display: grid; grid-template-columns: 1fr auto 1fr; gap: 16px; align-items: center; margin-bottom: 14px; color: var(--ink-tertiary); font: 500 10px/1.4 var(--sans); text-transform: uppercase; letter-spacing: .055em; }
.metric-axis span:last-child, .range-axis span:last-child, .forest-axis span:last-child { text-align: right; }
.metric-row { display: grid; grid-template-columns: 180px 1fr 78px; gap: 16px; align-items: center; min-height: 48px; border-top: 1px solid var(--line); }
.metric-name { color: var(--ink-muted); font-size: 12px; }
.metric-track, .difference-track, .range-track { position: relative; display: block; height: 18px; }
.metric-track::before, .difference-track::before, .range-track::before { content: ""; position: absolute; left: 0; right: 0; top: 50%; height: 1px; background: var(--line-strong); }
.metric-dot { position: absolute; left: var(--x); top: 50%; width: 9px; height: 9px; border: 2px solid var(--surface-1); border-radius: 2px; background: var(--accent); transform: translate(-50%,-50%) rotate(45deg); }
.metric-row strong, .range-values, .forest-value { color: var(--ink-muted); font: 400 11px var(--mono); text-align: right; white-space: nowrap; }
.difference-panel { margin-top: 26px; padding-top: 24px; border-top: 1px solid var(--line-strong); }
.difference-heading { display: flex; justify-content: space-between; gap: 24px; margin-bottom: 14px; color: var(--ink-muted); font-size: 12px; }
.difference-heading strong { font-family: var(--mono); font-weight: 400; white-space: nowrap; }
.difference-track { height: 26px; }
.difference-zero, .forest-zero { position: absolute; left: var(--zero); top: 0; bottom: 0; width: 1px; background: var(--ink-tertiary); }
.difference-ci, .forest-ci { position: absolute; left: var(--low); width: calc(var(--high) - var(--low)); top: 50%; height: 3px; background: var(--ink-muted); transform: translateY(-50%); transform-origin: left center; }
.difference-point, .forest-point { position: absolute; left: var(--point); top: 50%; width: 10px; height: 10px; border-radius: 2px; background: var(--accent); transform: translate(-50%,-50%) rotate(45deg); }
.difference-axis { display: grid; grid-template-columns: repeat(3,1fr); color: var(--ink-tertiary); font: 400 10px var(--mono); }
.difference-axis span:nth-child(2) { text-align: center; }
.difference-axis span:last-child { text-align: right; }
.difference-panel p, .chart-note { margin: 17px 0 0; color: var(--ink-subtle); font-size: 11px; line-height: 1.55; }
.hypothesis-ledger { border-top: 1px solid var(--line-strong); }
.hypothesis-row { margin: 0; border: 0; border-bottom: 1px solid var(--line); border-radius: 0; background: transparent; }
.hypothesis-row summary { min-height: 78px; display: grid; grid-template-columns: 58px 1fr minmax(200px,auto) 22px; gap: 18px; align-items: center; cursor: pointer; list-style: none; }
.hypothesis-row summary::-webkit-details-marker { display: none; }
.hypothesis-id { color: var(--accent-hover); font: 500 12px var(--mono); }
.hypothesis-title { font-size: 15px; font-weight: 500; letter-spacing: -.015em; }
.hypothesis-result { color: var(--ink-subtle); font: 400 11px var(--mono); text-align: right; white-space: nowrap; }
.hypothesis-toggle { color: var(--ink-subtle); font-size: 19px; }
.hypothesis-row[open] .hypothesis-toggle { transform: rotate(45deg); }
.hypothesis-row p { max-width: 760px; margin: 0 0 24px 76px; color: var(--ink-subtle); font-size: 13px; line-height: 1.6; }
.forest-plot { padding: 30px; border: 1px solid var(--line); border-radius: 16px; background: var(--surface-1); }
.forest-group { display: flex; justify-content: space-between; gap: 20px; margin: 30px 0 8px; padding-top: 18px; border-top: 1px solid var(--line-strong); color: var(--ink-tertiary); font-size: 10px; letter-spacing: .055em; text-transform: uppercase; }
.forest-group:first-of-type { margin-top: 0; padding-top: 0; border-top: 0; }
.forest-group span:first-child { color: var(--accent-hover); font-family: var(--mono); }
.forest-row { min-height: 48px; display: grid; grid-template-columns: minmax(320px,1.2fr) minmax(300px,1fr) 84px; gap: 22px; align-items: center; border-top: 1px solid rgba(35,37,42,.72); }
.forest-label { color: var(--ink-subtle); font-size: 11px; line-height: 1.38; }
.forest-track { position: relative; height: 22px; }
.forest-track-empty { display: grid; place-items: center; border: 1px dashed var(--line-strong); color: var(--ink-tertiary); font-size: 9px; letter-spacing: .04em; text-transform: uppercase; }
.forest-track-empty::before { display: none; }
.forest-ci { height: 2px; }
.forest-point { width: 8px; height: 8px; }
.forest-zero { opacity: .7; }
.range-row { min-height: 62px; display: grid; grid-template-columns: 180px 1fr 88px 110px; gap: 18px; align-items: center; border-top: 1px solid var(--line); }
.range-name { color: var(--ink-muted); font-size: 12px; }
.range-track { height: 22px; }
.range-line { position: absolute; left: var(--low); width: calc(var(--high) - var(--low)); top: 50%; height: 3px; background: var(--accent); transform: translateY(-50%); transform-origin: left center; }
.range-start, .range-end { position: absolute; top: 50%; width: 8px; height: 8px; border-radius: 2px; background: var(--accent-hover); transform: translate(-50%,-50%) rotate(45deg); }
.range-start { left: var(--low); }
.range-end { left: var(--high); }
.range-count { display: flex; flex-direction: column; text-align: right; font: 400 11px var(--mono); }
.range-count small { color: var(--ink-tertiary); font: 400 9px var(--sans); text-transform: uppercase; }
.seed-panel { display: grid; grid-template-columns: minmax(0,1.12fr) minmax(300px,.88fr); gap: 38px; align-items: center; padding: clamp(24px,3vw,38px); border: 1px solid var(--line); border-radius: 16px; background: var(--surface-1); }
.seed-panel svg { width: 100%; height: auto; }
.seed-copy h3 { margin: 0 0 13px; font-size: 18px; font-weight: 600; line-height: 1.3; letter-spacing: -.02em; }
.seed-copy p { margin: 0 0 13px; color: var(--ink-subtle); font-size: 13px; line-height: 1.6; }
.seed-copy code { color: var(--ink-muted); font-size: 10px; }
.stats-grid { display: grid; grid-template-columns: repeat(4,1fr); border: 1px solid var(--line); border-radius: 16px; overflow: hidden; }
.stat { min-height: 132px; padding: 24px; border-right: 1px solid var(--line); background: var(--surface-1); }
.stat:last-child { border-right: 0; }
.stat strong { display: block; margin-bottom: 10px; font-size: clamp(28px,3vw,40px); font-weight: 500; line-height: 1; letter-spacing: -.045em; }
.stat span { color: var(--ink-subtle); font-size: 11px; line-height: 1.45; }
.rules { display: grid; grid-template-columns: repeat(2,1fr); gap: 1px; margin-top: 32px; border: 1px solid var(--line); background: var(--line); }
.rule { min-height: 145px; padding: 25px; background: var(--surface-1); }
.rule b { display: block; margin-bottom: 10px; font-size: 13px; font-weight: 600; }
.rule p { margin: 0; color: var(--ink-subtle); font-size: 12px; line-height: 1.6; }
.claim-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 1px; border: 1px solid var(--line); border-radius: 16px; overflow: hidden; background: var(--line); }
.claim-card { min-height: 230px; padding: 30px; background: var(--surface-1); }
.claim-card h3 { margin: 0 0 18px; font-size: 17px; font-weight: 600; }
.claim-card ul { margin: 0; padding: 0; list-style: none; }
.claim-card li { position: relative; margin: 0 0 12px; padding-left: 22px; color: var(--ink-subtle); font-size: 13px; line-height: 1.55; }
.claim-card li::before { content: ""; position: absolute; left: 0; top: .62em; width: 8px; height: 1px; background: var(--accent); }
.qc-frame { padding: 16px; border: 1px solid var(--line); border-radius: 16px; background: var(--surface-1); }
.qc-toolbar { display: flex; justify-content: space-between; gap: 20px; margin-bottom: 14px; color: var(--ink-subtle); font-size: 11px; }
.qc-frame img { display: block; width: 100%; height: auto; border: 1px solid var(--line); border-radius: 10px; background: #fff; }
details.evidence-details { margin-top: 28px; border: 1px solid var(--line); border-radius: 12px; background: var(--surface-1); overflow: hidden; }
details.evidence-details > summary { padding: 16px 18px; cursor: pointer; color: var(--ink-muted); font-size: 13px; font-weight: 500; }
details.evidence-details[open] > summary { border-bottom: 1px solid var(--line); }
.table-tools { display: grid; grid-template-columns: 160px 180px minmax(220px,1fr) auto; gap: 12px; padding: 16px; border-bottom: 1px solid var(--line); }
.table-tools label { display: flex; flex-direction: column; gap: 5px; color: var(--ink-tertiary); font-size: 9px; letter-spacing: .06em; text-transform: uppercase; }
input, select { min-height: 38px; padding: 7px 10px; border: 1px solid var(--line-strong); border-radius: 8px; color: var(--ink); background: var(--surface-2); font: 400 12px var(--sans); }
.table-count { align-self: end; padding: 9px 0; color: var(--ink-subtle); font-size: 11px; white-space: nowrap; }
.table-wrap { max-width: 100%; overflow: auto; }
table { width: 100%; min-width: 1120px; border-collapse: collapse; table-layout: fixed; font-size: 11px; }
table.compact { min-width: 820px; table-layout: auto; }
th, td { padding: 12px 13px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: middle; }
th { position: sticky; top: 0; z-index: 2; color: var(--ink-muted); background: var(--surface-2); font-size: 10px; font-weight: 500; letter-spacing: .02em; white-space: nowrap; }
td { color: var(--ink-subtle); }
td.numeric { font: 400 10px var(--mono); text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
tr:hover td { color: var(--ink-muted); background: var(--surface-2); }
.hypothesis-chip { color: var(--accent-hover); font: 500 11px var(--mono); }
.comparison-name { display: block; color: var(--ink-muted); font-size: 11px; line-height: 1.4; }
.comparison-id { display: block; margin-top: 4px; color: var(--ink-tertiary); font-size: 9px; line-height: 1.4; overflow-wrap: anywhere; }
.status-pill { display: inline-block; padding: 3px 6px; border: 1px solid var(--line-strong); border-radius: 4px; color: var(--ink-subtle); font-size: 9px; text-transform: uppercase; letter-spacing: .04em; white-space: nowrap; }
.status-pill.is-reported { border-color: rgba(94,106,210,.55); color: var(--ink-muted); }
.fine { color: var(--ink-subtle); font-size: 11px; line-height: 1.6; }
.fine strong { color: var(--ink-muted); font-weight: 500; }
.provenance { display: grid; grid-template-columns: 1fr 1fr; gap: 48px; }
.provenance h3 { margin: 0 0 16px; font-size: 15px; font-weight: 600; }
.provenance p, .provenance li { color: var(--ink-subtle); font-size: 11px; line-height: 1.6; }
.provenance ul { margin: 0; padding-left: 17px; }
.hash-list { display: grid; gap: 10px; margin: 0; }
.hash-list div { padding-bottom: 10px; border-bottom: 1px solid var(--line); }
.hash-list dt { color: var(--ink-tertiary); font-size: 9px; text-transform: uppercase; }
.hash-list dd { margin: 4px 0 0; color: var(--ink-subtle); font: 400 9px/1.5 var(--mono); overflow-wrap: anywhere; }
footer { padding: 56px var(--page-pad); border-top: 1px solid var(--line); color: var(--ink-tertiary); font-size: 10px; }
.footer-inner { width: min(var(--wide),100%); margin: auto; display: flex; justify-content: space-between; gap: 40px; }
.gsap-ready .reveal { visibility: hidden; }
@media (max-width: 1040px) {
  .hero-shell { grid-template-columns: minmax(0,1.2fr) minmax(300px,.8fr); }
  .story-inner { grid-template-columns: minmax(460px,1.05fr) minmax(330px,.95fr); gap: 42px; }
  .result-layout { grid-template-columns: 1fr; }
  .forest-row { grid-template-columns: minmax(270px,1fr) minmax(250px,1fr) 80px; }
  .table-tools { grid-template-columns: repeat(3,1fr); }
  .table-count { grid-column: 1/-1; }
}
@media (max-width: 900px) {
  .hero { min-height: 860px; }
  .hero-shell { min-height: 804px; grid-template-columns: 1fr; align-items: start; }
  .hero-copy { max-width: 680px; padding-top: 126px; }
  .hero-visual-label { display: none; }
  .story { height: auto; padding: 92px 0; }
  .story-frame { position: static; min-height: 0; overflow: visible; }
  .story-inner { grid-template-columns: 1fr; }
  .workflow-panel { min-height: 0; }
  .story-steps { margin-top: 28px; }
  .story-step { opacity: 1; }
  .workflow-panel [data-stage], .workflow-panel [data-stage].is-visible, .workflow-panel [data-stage].is-current { opacity: 1; transform: none; }
  .seed-panel { grid-template-columns: 1fr; }
  .forest-row { grid-template-columns: minmax(240px,1fr) minmax(220px,1fr) 80px; gap: 16px; }
  .stats-grid { grid-template-columns: repeat(2,1fr); }
  .stat:nth-child(2) { border-right: 0; }
  .stat:nth-child(-n+2) { border-bottom: 1px solid var(--line); }
}
@media (max-width: 720px) {
  :root { --page-pad: 22px; }
  .nav-links { position: absolute; top: 56px; left: 0; right: 0; display: none; flex-direction: column; align-items: flex-start; gap: 0; padding: 10px 22px 18px; border-bottom: 1px solid var(--line); background: var(--canvas); }
  .nav-links.is-open { display: flex; }
  .nav-links a { width: 100%; padding: 11px 0; }
  .nav-links a::after { display: none; }
  .menu-button { display: block; }
  .hero { min-height: 800px; }
  .hero-shell { min-height: 744px; }
  .hero-copy { padding-top: 112px; }
  .hero h1 { font-size: clamp(38px,12vw,52px); }
  .hero-lead { max-width: 470px; font-size: 16px; }
  .hero-byline { max-width: 300px; flex-direction: column; gap: 6px; }
  .hero-byline span + span::before { display: none; }
  .intro-copy { grid-template-columns: 1fr; gap: 20px; }
  .question-panel { grid-template-columns: 1fr; gap: 14px; padding: 24px; }
  .section-heading { margin-bottom: 38px; }
  .workflow-panel svg { min-width: 0; }
  .result-lead, .h4-chart, .h2-chart { padding: 22px; }
  .metric-row { grid-template-columns: 1fr 76px; gap: 7px 12px; padding: 12px 0; }
  .metric-track { grid-column: 1/-1; grid-row: 2; }
  .hypothesis-row summary { grid-template-columns: 42px 1fr 20px; gap: 12px; padding: 12px 0; }
  .hypothesis-result { grid-column: 2; grid-row: 2; text-align: left; white-space: normal; }
  .hypothesis-toggle { grid-column: 3; grid-row: 1/3; }
  .hypothesis-row p { margin-left: 54px; }
  .forest-plot { padding: 20px; }
  .forest-axis { display: none; }
  .forest-row { grid-template-columns: 1fr 78px; gap: 7px 12px; padding: 11px 0; }
  .forest-label { font-size: 10px; }
  .forest-track { grid-column: 1/-1; grid-row: 2; }
  .forest-value { grid-column: 2; grid-row: 1; }
  .forest-group { margin-top: 22px; }
  .forest-group span:last-child { display: none; }
  .range-row { grid-template-columns: 1fr 76px; gap: 7px 12px; padding: 12px 0; }
  .range-track { grid-column: 1/-1; grid-row: 2; }
  .range-count { grid-column: 2; grid-row: 1; }
  .range-values { display: none; }
  .rules, .stats-grid, .claim-grid, .provenance { grid-template-columns: 1fr; }
  .stat { border-right: 0; border-bottom: 1px solid var(--line); }
  .stat:last-child { border-bottom: 0; }
  .table-tools { grid-template-columns: 1fr; }
  .table-count { grid-column: auto; }
  .table-wrap { overflow: visible; }
  table, table.compact { min-width: 0; table-layout: auto; }
  table thead { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
  table, table tbody, table tr, table td { display: block; width: 100%; }
  table tr { padding: 13px 14px; border-bottom: 1px solid var(--line); }
  table tr[hidden] { display: none !important; }
  table td { display: grid; grid-template-columns: minmax(92px,32%) 1fr; gap: 12px; padding: 5px 0; border: 0; text-align: left !important; white-space: normal !important; }
  table td::before { content: attr(data-label); color: var(--ink-tertiary); font: 500 9px/1.5 var(--sans); letter-spacing: .04em; text-transform: uppercase; }
  .footer-inner { flex-direction: column; }
  .qc-toolbar { flex-direction: column; gap: 7px; }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; }
  .site-header { transform: none !important; }
  .gsap-ready .reveal { visibility: visible !important; opacity: 1 !important; transform: none !important; }
  .story-step { opacity: 1 !important; }
  .workflow-panel [data-stage] { opacity: 1 !important; transform: none !important; }
}
@media print {
  :root { color-scheme: light; --canvas: #fff; --surface-1: #fff; --surface-2: #f6f6f6; --surface-3: #eee; --ink: #111; --ink-muted: #333; --ink-subtle: #555; --ink-tertiary: #666; --line: #ccc; --line-strong: #999; --accent: #4b56bb; }
  .site-header, .menu-button, #hero-canvas, .reading-progress { display: none !important; }
  .hero { min-height: auto; padding-top: 56px; }
  .hero-shell { min-height: auto; grid-template-columns: 1fr; }
  .story { height: auto; }
  .story-frame { position: static; min-height: 0; }
  .story-inner { grid-template-columns: 1fr; }
  .workflow-panel [data-stage], .story-step, .reveal { visibility: visible !important; opacity: 1 !important; transform: none !important; }
  .section { break-inside: avoid; }
}
"""


_MVP_SCRIPT_V3 = r"""
(() => {
  const root = document.documentElement;
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const gsapEngine = window.gsap;
  const scrollEngine = window.ScrollTrigger;
  const motionAvailable = Boolean(gsapEngine && scrollEngine && !reduced);
  if (gsapEngine && scrollEngine) gsapEngine.registerPlugin(scrollEngine);

  const header = document.getElementById('site-header');
  const progress = document.getElementById('reading-progress');
  let lastScroll = window.scrollY;
  const updateChrome = () => {
    const current = window.scrollY;
    header.classList.toggle('is-scrolled', current > 16);
    const hide = !reduced && current > 160 && current > lastScroll + 5;
    if (gsapEngine) {
      gsapEngine.to(header, {yPercent: hide ? -100 : 0, duration: .28, overwrite: true});
    } else {
      header.style.transform = hide ? 'translateY(-100%)' : 'translateY(0)';
    }
    const range = document.documentElement.scrollHeight - window.innerHeight;
    const scale = range > 0 ? current / range : 0;
    if (gsapEngine) gsapEngine.set(progress, {scaleX: scale});
    else progress.style.transform = `scaleX(${scale})`;
    lastScroll = current;
  };
  window.addEventListener('scroll', updateChrome, {passive: true});
  updateChrome();

  const menuButton = document.getElementById('menu-button');
  const navLinks = document.getElementById('nav-links');
  menuButton.addEventListener('click', () => {
    const open = !navLinks.classList.contains('is-open');
    navLinks.classList.toggle('is-open', open);
    menuButton.setAttribute('aria-expanded', String(open));
  });
  navLinks.addEventListener('click', event => {
    if (event.target instanceof HTMLAnchorElement) {
      navLinks.classList.remove('is-open');
      menuButton.setAttribute('aria-expanded', 'false');
    }
  });

  const story = document.querySelector('.story');
  const storySteps = [...document.querySelectorAll('[data-story-step]')];
  const stageItems = [...document.querySelectorAll('#workflow-panel [data-stage]')];
  let activeStoryStage = -1;
  const setStoryStage = active => {
    if (active === activeStoryStage) return;
    activeStoryStage = active;
    storySteps.forEach((step, index) => step.classList.toggle('is-active', index === active));
    stageItems.forEach(item => {
      const stage = Number(item.dataset.stage);
      item.classList.toggle('is-visible', stage <= active);
      item.classList.toggle('is-current', stage === active);
      if (motionAvailable && window.innerWidth > 900) {
        gsapEngine.to(item, {
          opacity: stage === active ? 1 : stage < active ? .48 : .16,
          y: stage === active ? -2 : 0,
          duration: .32,
          ease: 'power2.out',
          overwrite: true,
        });
      }
    });
  };
  const nativeStoryUpdate = () => {
    if (window.innerWidth <= 900 || reduced) {
      setStoryStage(4);
      return;
    }
    const rect = story.getBoundingClientRect();
    const distance = Math.max(1, rect.height - window.innerHeight);
    const ratio = Math.max(0, Math.min(.9999, -rect.top / distance));
    setStoryStage(Math.min(4, Math.floor(ratio * 5)));
  };
  if (motionAvailable) {
    scrollEngine.create({
      trigger: story,
      start: 'top top',
      end: 'bottom bottom',
      onUpdate: self => setStoryStage(Math.min(4, Math.floor(self.progress * 4.9999))),
      onLeaveBack: () => setStoryStage(0),
    });
    const mobileStory = window.matchMedia('(max-width: 900px)');
    const syncStoryMode = () => mobileStory.matches && setStoryStage(4);
    mobileStory.addEventListener('change', syncStoryMode);
    syncStoryMode();
  } else {
    window.addEventListener('scroll', nativeStoryUpdate, {passive: true});
    window.addEventListener('resize', nativeStoryUpdate, {passive: true});
    nativeStoryUpdate();
  }

  const revealItems = [...document.querySelectorAll('.reveal')];
  if (motionAvailable) {
    root.classList.add('gsap-ready');
    gsapEngine.set(revealItems, {autoAlpha: 0, y: 20});
    revealItems.forEach(item => {
      gsapEngine.to(item, {
        autoAlpha: 1,
        y: 0,
        duration: .7,
        ease: 'power3.out',
        scrollTrigger: {trigger: item, start: 'top 88%', once: true},
      });
    });
    gsapEngine.from('.hero-animate', {
      autoAlpha: 0,
      y: 24,
      duration: .85,
      stagger: .1,
      ease: 'power3.out',
      delay: .12,
    });
    document.querySelectorAll('.h4-chart, .h2-chart, .forest-plot').forEach(chart => {
      const marks = chart.querySelectorAll(
        '.metric-dot, .difference-ci, .difference-point, .range-line, .range-start, ' +
        '.range-end, .forest-ci, .forest-point'
      );
      gsapEngine.from(marks, {
        scale: 0,
        duration: .45,
        stagger: .012,
        ease: 'back.out(1.6)',
        scrollTrigger: {trigger: chart, start: 'top 82%', once: true},
      });
    });
    document.querySelectorAll('main section[id]').forEach(section => {
      const link = document.querySelector(`.nav-links a[href="#${section.id}"]`);
      if (!link) return;
      scrollEngine.create({
        trigger: section,
        start: 'top 45%',
        end: 'bottom 45%',
        onToggle: self => link.classList.toggle('is-active', self.isActive),
      });
    });
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => scrollEngine.refresh());
    }
  } else if ('IntersectionObserver' in window && !reduced) {
    const observer = new IntersectionObserver(entries => entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.visibility = 'visible';
        observer.unobserve(entry.target);
      }
    }), {threshold: .08});
    revealItems.forEach(item => observer.observe(item));
  }

  const hypothesisFilter = document.getElementById('filter-hypothesis');
  const statusFilter = document.getElementById('filter-status');
  const searchFilter = document.getElementById('filter-search');
  const comparisonRows = [...document.querySelectorAll('[data-comparison-row]')];
  const tableCount = document.getElementById('table-count');
  const filterTable = () => {
    const hypothesis = hypothesisFilter.value;
    const status = statusFilter.value;
    const query = searchFilter.value.trim().toLowerCase();
    let visible = 0;
    comparisonRows.forEach(row => {
      const matches = (hypothesis === 'ALL' || row.dataset.hypothesis === hypothesis) &&
        (status === 'ALL' || row.dataset.status === status) &&
        (!query || row.textContent.toLowerCase().includes(query));
      row.hidden = !matches;
      if (matches) visible += 1;
    });
    tableCount.textContent = `${visible} / ${comparisonRows.length} rows`;
  };
  [hypothesisFilter, statusFilter, searchFilter].forEach(control =>
    control.addEventListener('input', filterTable));
  filterTable();
})();
"""


_MVP_THREE_SCRIPT_V3 = r"""
const canvas = document.getElementById('hero-canvas');
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const disableCanvas = () => {
  canvas.hidden = true;
  canvas.dataset.renderer = 'unavailable';
  canvas.dataset.animationState = 'unavailable';
};

import('https://cdn.jsdelivr.net/npm/three@0.185.1/build/three.module.js').then(THREE => {
try {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    alpha: true,
    antialias: true,
    powerPreference: 'low-power',
  });
  renderer.setClearColor(0x010102, 0);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, .1, 100);
  camera.position.set(0, 0, 7.2);

  let randomSeed = 314159265;
  const random = () => {
    randomSeed = (randomSeed * 1664525 + 1013904223) >>> 0;
    return randomSeed / 4294967296;
  };

  const clamp = value => Math.max(0, Math.min(1, value));
  const smooth = value => {
    const limited = clamp(value);
    return limited * limited * (3 - 2 * limited);
  };

  const auditField = new THREE.Group();
  scene.add(auditField);

  const makeLabelSprite = (text, width = .72, height = .16) => {
    const labelCanvas = document.createElement('canvas');
    labelCanvas.width = 512;
    labelCanvas.height = 128;
    const context = labelCanvas.getContext('2d');
    context.clearRect(0, 0, labelCanvas.width, labelCanvas.height);
    context.fillStyle = '#7e828d';
    context.font = '500 34px "JetBrains Mono", monospace';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(text, labelCanvas.width / 2, labelCanvas.height / 2);
    const texture = new THREE.CanvasTexture(labelCanvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.minFilter = THREE.LinearFilter;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      opacity: .72,
      depthWrite: false,
    }));
    sprite.scale.set(width, height, 1);
    return sprite;
  };

  const blobVariants = Array.from({length: 9}, (_, variant) => {
    const outlinePoints = [];
    const shapePoints = [];
    const segments = 32;
    for (let pointIndex = 0; pointIndex < segments; pointIndex += 1) {
      const angle = pointIndex / segments * Math.PI * 2;
      const firstWave = 3 + variant % 3;
      const secondWave = 5 + variant % 4;
      const radius = 1
        + Math.sin(angle * firstWave + variant * .71) * (.075 + (variant % 2) * .018)
        + Math.cos(angle * secondWave - variant * .43) * .045;
      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius;
      shapePoints.push(new THREE.Vector2(x, y));
      outlinePoints.push(new THREE.Vector3(x, y, .012));
    }
    return {
      fill: new THREE.ShapeGeometry(new THREE.Shape(shapePoints)),
      outline: new THREE.BufferGeometry().setFromPoints(outlinePoints),
    };
  });

  const makeNucleus = (variant, {queueCopy = false} = {}) => {
    const group = new THREE.Group();
    const fillMaterial = new THREE.MeshBasicMaterial({
      color: queueCopy ? 0x5e6ad2 : 0x323641,
      transparent: true,
      opacity: queueCopy ? .58 : .22,
      depthWrite: false,
    });
    const outlineMaterial = new THREE.LineBasicMaterial({
      color: queueCopy ? 0x97a2ff : 0x707786,
      transparent: true,
      opacity: queueCopy ? .92 : .42,
      depthWrite: false,
    });
    group.add(new THREE.Mesh(blobVariants[variant].fill, fillMaterial));
    group.add(new THREE.LineLoop(blobVariants[variant].outline, outlineMaterial));
    group.userData.fillMaterial = fillMaterial;
    group.userData.outlineMaterial = outlineMaterial;
    return group;
  };

  const patchFramePoints = [
    new THREE.Vector3(-1.91, -1.62, -.18),
    new THREE.Vector3(.51, -1.62, -.18),
    new THREE.Vector3(.51, 1.62, -.18),
    new THREE.Vector3(-1.91, 1.62, -.18),
  ];
  const patchFrame = new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints(patchFramePoints),
    new THREE.LineBasicMaterial({
      color: 0x707580,
      transparent: true,
      opacity: .18,
      depthWrite: false,
    }),
  );
  auditField.add(patchFrame);
  const sourceLabel = makeLabelSprite('SOURCE PATCH', .82, .15);
  sourceLabel.position.set(-1.50, 1.78, 0);
  auditField.add(sourceLabel);

  const nucleusCount = 72;
  const sourceNuclei = [];
  const sourcePositions = [];
  for (let index = 0; index < nucleusCount; index += 1) {
    const angle = random() * Math.PI * 2;
    const radius = Math.sqrt(random());
    const position = new THREE.Vector3(
      -.70 + Math.cos(angle) * radius * 1.08,
      Math.sin(angle) * radius * 1.42,
      -.05 + (random() - .5) * .16,
    );
    const nucleus = makeNucleus(index % blobVariants.length, {queueCopy: false});
    const baseScale = .068 + random() * .052;
    const stretch = .78 + random() * .58;
    nucleus.position.copy(position);
    nucleus.rotation.z = random() * Math.PI;
    nucleus.userData.baseScale = baseScale;
    nucleus.userData.stretch = stretch;
    nucleus.userData.phase = random() * Math.PI * 2;
    nucleus.scale.set(baseScale * stretch, baseScale, baseScale);
    sourceNuclei.push(nucleus);
    sourcePositions.push(position);
    auditField.add(nucleus);
  }

  const selectedIndices = [8, 47, 20, 60, 34, 67];
  const queuePositions = selectedIndices.map((_, index) => new THREE.Vector3(
    1.55,
    1.15 - index * .47,
    .08,
  ));

  const queueLabel = makeLabelSprite('REVIEW QUEUE', .92, .15);
  queueLabel.position.set(1.47, 1.78, 0);
  auditField.add(queueLabel);
  const queueCopies = [];
  const queueSlotMaterial = new THREE.LineBasicMaterial({
    color: 0x666b75,
    transparent: true,
    opacity: .24,
    depthWrite: false,
  });
  selectedIndices.forEach((sourceIndex, index) => {
    const y = queuePositions[index].y;
    const slot = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(1.08, y - .17, -.12),
        new THREE.Vector3(2.03, y - .17, -.12),
      ]),
      queueSlotMaterial,
    );
    auditField.add(slot);
    const number = makeLabelSprite(String(index + 1).padStart(2, '0'), .24, .12);
    number.position.set(.91, y, 0);
    auditField.add(number);

    const copy = makeNucleus(sourceIndex % blobVariants.length, {queueCopy: true});
    copy.rotation.z = sourceNuclei[sourceIndex].rotation.z;
    copy.scale.setScalar(.11);
    copy.visible = false;
    queueCopies.push(copy);
    auditField.add(copy);
  });

  const reticle = new THREE.Group();
  const reticleMaterial = new THREE.MeshBasicMaterial({
    color: 0x7f8cff,
    transparent: true,
    opacity: .9,
    depthWrite: false,
  });
  const reticleOffsets = [[-.18, .18], [.18, .18], [-.18, -.18], [.18, -.18]];
  reticleOffsets.forEach(([x, y]) => {
    const tile = new THREE.Mesh(new THREE.PlaneGeometry(.075, .075), reticleMaterial);
    tile.position.set(x, y, .24);
    tile.rotation.z = Math.PI / 4;
    reticle.add(tile);
  });
  reticle.visible = false;
  auditField.add(reticle);

  const classSignals = new THREE.Group();
  const signalMaterials = [];
  for (let index = 0; index < 5; index += 1) {
    const angle = -Math.PI / 2 + index * Math.PI * 2 / 5;
    const material = new THREE.MeshBasicMaterial({
      color: 0xa5adff,
      transparent: true,
      opacity: 0,
      depthWrite: false,
    });
    const signal = new THREE.Mesh(new THREE.CircleGeometry(.022, 12), material);
    signal.position.set(Math.cos(angle) * .29, Math.sin(angle) * .29, .25);
    signalMaterials.push(material);
    classSignals.add(signal);
  }
  classSignals.visible = false;
  auditField.add(classSignals);

  const trailMaterial = new THREE.LineBasicMaterial({
    color: 0x7d89f4,
    transparent: true,
    opacity: 0,
    depthWrite: false,
  });
  const trailGeometry = new THREE.BufferGeometry();
  const trail = new THREE.Line(trailGeometry, trailMaterial);
  trail.visible = false;
  auditField.add(trail);

  const reviewRingMaterial = new THREE.MeshBasicMaterial({
    color: 0xb9c0ff,
    transparent: true,
    opacity: 0,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const reviewRing = new THREE.Mesh(new THREE.RingGeometry(.145, .158, 48), reviewRingMaterial);
  reviewRing.position.copy(queuePositions[0]);
  auditField.add(reviewRing);

  const backgroundGeometry = new THREE.BufferGeometry();
  const backgroundPositions = [];
  for (let index = 0; index < 240; index += 1) {
    backgroundPositions.push(
      (random() - .5) * 12,
      (random() - .5) * 7,
      -1.5 - random() * 2,
    );
  }
  backgroundGeometry.setAttribute(
    'position',
    new THREE.Float32BufferAttribute(backgroundPositions, 3),
  );
  const background = new THREE.Points(
    backgroundGeometry,
    new THREE.PointsMaterial({color: 0x62666d, size: .014, transparent: true, opacity: .34}),
  );
  scene.add(background);

  const itemDuration = 1.55;
  const activeDuration = selectedIndices.length * itemDuration;
  const holdDuration = 2.2;
  const fadeDuration = 1.15;
  const cycleDuration = activeDuration + holdDuration + fadeDuration;
  const sourcePoint = new THREE.Vector3();
  const targetPoint = new THREE.Vector3();
  const controlPoint = new THREE.Vector3();
  const animatedPoint = new THREE.Vector3();

  const queueCurve = (source, target) => {
    controlPoint.set(
      (source.x + target.x) / 2,
      source.y + (target.y >= source.y ? .34 : -.34),
      .34,
    );
    return new THREE.QuadraticBezierCurve3(source, controlPoint.clone(), target);
  };

  const updateScene = elapsed => {
    const cycle = reduced ? activeDuration + .4 : elapsed % cycleDuration;
    const fadeStart = activeDuration + holdDuration;
    const cycleFade = cycle <= fadeStart ? 1 : 1 - smooth((cycle - fadeStart) / fadeDuration);
    const activeIndex = cycle < activeDuration
      ? Math.min(selectedIndices.length - 1, Math.floor(cycle / itemDuration))
      : -1;
    const activePhase = activeIndex >= 0
      ? (cycle - activeIndex * itemDuration) / itemDuration
      : 1;

    sourceNuclei.forEach((nucleus, index) => {
      const breath = reduced ? 1 : 1 + Math.sin(elapsed * .72 + nucleus.userData.phase) * .035;
      const selectedPosition = selectedIndices.indexOf(index);
      const alreadyQueued = selectedPosition >= 0 && cycle >= selectedPosition * itemDuration + 1.18;
      const isActive = activeIndex >= 0 && selectedIndices[activeIndex] === index;
      const baseScale = nucleus.userData.baseScale * breath;
      nucleus.scale.set(
        baseScale * nucleus.userData.stretch,
        baseScale,
        baseScale,
      );
      nucleus.userData.fillMaterial.color.setHex(isActive ? 0x5662c2 : 0x323641);
      nucleus.userData.fillMaterial.opacity = isActive ? .52 : alreadyQueued ? .25 : .19;
      nucleus.userData.outlineMaterial.color.setHex(
        isActive ? 0xaab2ff : alreadyQueued ? 0x7985df : 0x707786,
      );
      nucleus.userData.outlineMaterial.opacity = isActive ? .96 : alreadyQueued ? .55 : .38;
    });

    queueCopies.forEach((copy, index) => {
      const local = cycle - index * itemDuration;
      if (local < .42 || cycleFade <= 0) {
        copy.visible = false;
        return;
      }
      const movement = smooth((local - .48) / .67);
      sourcePoint.copy(sourcePositions[selectedIndices[index]]);
      targetPoint.copy(queuePositions[index]);
      const curve = queueCurve(sourcePoint, targetPoint);
      curve.getPoint(movement, animatedPoint);
      copy.position.copy(animatedPoint);
      copy.visible = true;
      const appearing = smooth((local - .42) / .18);
      copy.userData.fillMaterial.opacity = .58 * appearing * cycleFade;
      copy.userData.outlineMaterial.opacity = .94 * appearing * cycleFade;
      const scale = .10 + .018 * smooth((local - .52) / .55);
      copy.scale.setScalar(scale);
    });

    if (activeIndex >= 0) {
      const activeSource = sourcePositions[selectedIndices[activeIndex]];
      reticle.position.copy(activeSource);
      reticle.rotation.z = elapsed * .34;
      reticle.scale.setScalar(.88 + Math.sin(activePhase * Math.PI) * .18);
      reticleMaterial.opacity = (.38 + Math.sin(activePhase * Math.PI) * .62) * cycleFade;
      reticle.visible = activePhase < .82;

      classSignals.position.copy(activeSource);
      classSignals.rotation.z = -elapsed * .12;
      classSignals.visible = activePhase > .10 && activePhase < .72;
      signalMaterials.forEach((material, index) => {
        const stagger = smooth((activePhase - .12 - index * .035) / .18);
        const withdraw = 1 - smooth((activePhase - .55) / .17);
        material.opacity = .78 * stagger * withdraw * cycleFade;
      });

      if (activePhase > .30 && activePhase < .90) {
        const movement = smooth((activePhase - .31) / .47);
        const curve = queueCurve(activeSource, queuePositions[activeIndex]);
        const trailPoints = [];
        for (let pointIndex = 0; pointIndex <= 24; pointIndex += 1) {
          trailPoints.push(curve.getPoint(movement * pointIndex / 24));
        }
        trailGeometry.setFromPoints(trailPoints);
        trailMaterial.opacity = .46 * (1 - movement * .52) * cycleFade;
        trail.visible = true;
      } else {
        trail.visible = false;
      }
    } else {
      reticle.visible = false;
      classSignals.visible = false;
      trail.visible = false;
    }

    const firstArrived = cycle >= 1.18;
    reviewRing.visible = firstArrived && cycleFade > 0;
    reviewRingMaterial.opacity = firstArrived
      ? (.25 + (reduced ? .18 : Math.sin(elapsed * 2.1) * .10)) * cycleFade
      : 0;
    reviewRing.scale.setScalar(reduced ? 1 : 1 + Math.sin(elapsed * 2.1) * .045);
  };

  const resize = () => {
    const rect = canvas.getBoundingClientRect();
    const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
    const saveData = Boolean(connection && connection.saveData);
    const ratioCap = saveData ? 1 : rect.width < 720 ? 1.25 : 1.5;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, ratioCap));
    canvas.dataset.pixelRatioCap = String(ratioCap);
    renderer.setSize(Math.max(1, rect.width), Math.max(1, rect.height), false);
    camera.aspect = Math.max(.1, rect.width / Math.max(1, rect.height));
    camera.updateProjectionMatrix();
    const mobile = rect.width < 720;
    auditField.position.set(mobile ? .18 : 2.16, mobile ? -2.05 : -.02, 0);
    auditField.scale.setScalar(mobile ? .42 : rect.width < 1000 ? .73 : .92);
  };
  resize();
  window.addEventListener('resize', resize, {passive: true});

  let heroVisible = true;

  if (window.gsap && window.ScrollTrigger && !reduced) {
    window.addEventListener('pointermove', event => {
      if (!heroVisible) return;
      const x = event.clientX / Math.max(1, window.innerWidth) - .5;
      const y = event.clientY / Math.max(1, window.innerHeight) - .5;
      window.gsap.to(auditField.rotation, {
        x: -y * .08,
        y: x * .14,
        duration: 1.1,
        ease: 'power2.out',
        overwrite: true,
      });
    }, {passive: true});
  }

  const timer = new THREE.Timer();
  timer.connect(document);
  const render = () => {
    timer.update();
    const elapsed = timer.getElapsed();
    updateScene(elapsed);
    if (!reduced) {
      auditField.rotation.z = Math.sin(elapsed * .16) * .012;
      background.rotation.z = elapsed * .004;
    }
    renderer.render(scene, camera);
  };
  let frameRequest = 0;
  const shouldAnimate = () => !reduced && heroVisible && !document.hidden;
  const frame = () => {
    frameRequest = 0;
    if (!shouldAnimate()) {
      canvas.dataset.animationState = 'paused';
      return;
    }
    render();
    frameRequest = window.requestAnimationFrame(frame);
  };
  const syncRenderLoop = () => {
    if (shouldAnimate()) {
      canvas.dataset.animationState = 'running';
      if (!frameRequest) frameRequest = window.requestAnimationFrame(frame);
      return;
    }
    if (frameRequest) window.cancelAnimationFrame(frameRequest);
    frameRequest = 0;
    canvas.dataset.animationState = reduced ? 'static' : 'paused';
    if (reduced) render();
  };
  if ('IntersectionObserver' in window) {
    const visibilityObserver = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.target === canvas) heroVisible = entry.isIntersecting;
      });
      syncRenderLoop();
    }, {rootMargin: '160px 0px'});
    visibilityObserver.observe(canvas);
  }
  document.addEventListener('visibilitychange', syncRenderLoop);
  if (reduced) window.addEventListener('resize', render, {passive: true});
  syncRenderLoop();
  canvas.dataset.renderer = 'threejs-review-queue';
  canvas.dataset.story = 'immutable-source-ranked-review';
} catch (_error) {
  disableCanvas();
}
}).catch(disableCanvas);
"""


_MVP_CSS_V4 = (
    _MVP_CSS_V3
    + r"""
/* Professor-facing editorial release: centered reading rhythm + cumulative method story. */
:root {
  --ink-tertiary: #7b8089;
  --prose: 780px;
  --editorial: 860px;
  --figure: 1160px;
  --wide: 1320px;
  --section-space: clamp(104px, 11vw, 164px);
}
body { font-size: 16px; line-height: 1.68; }
.site-header { height: 60px; border: 0; background: rgba(1,1,2,.82); backdrop-filter: blur(20px) saturate(125%); will-change: transform; }
.site-header.is-scrolled { border: 0; }
.nav-shell { height: 60px; }
.brand { min-height: 44px; gap: 9px; font-size: 14px; }
.brand-mark { position: relative; z-index: 2; width: 19px; height: 19px; flex: 0 0 19px; }
.brand-word-clip { display: block; width: 51px; overflow: hidden; }
.brand-label { display: block; transform-origin: left center; }
.nav-links { gap: 25px; }
.nav-links a { min-height: 44px; display: inline-flex; align-items: center; font-size: 12px; letter-spacing: .01em; }
.nav-links a::after { display: none; }
.nav-links a.is-active { color: var(--accent-hover); }
.reading-progress { display: none !important; }
.menu-button { min-width: 44px; min-height: 44px; border-color: var(--line-strong); }

.hero { min-height: min(900px,100svh); padding-top: 60px; border: 0; }
.hero::after { content: ""; position: absolute; left: 50%; bottom: 0; width: min(640px,calc(100% - 48px)); height: 1px; background: var(--line); transform: translateX(-50%); }
#hero-canvas { inset: 60px 0 0; height: calc(100% - 60px); }
#hero-canvas[hidden] { display: none !important; }
.hero-shell { min-height: calc(min(900px,100svh) - 60px); grid-template-columns: minmax(0,1.03fr) minmax(360px,.97fr); }
.hero-copy { max-width: 680px; padding: 86px 0 100px; }
.hero h1 { max-width: 680px; margin-top: 18px; font-size: clamp(42px,5vw,64px); line-height: 1.035; letter-spacing: -.048em; }
.hero-lead { max-width: 630px; margin-top: 24px; font-size: clamp(16px,1.35vw,19px); line-height: 1.62; }
.hero-byline { margin-top: 30px; }
.hero-visual-label { bottom: 38px; }
.eyebrow, .section-kicker, .chart-label { font-size: 10px; letter-spacing: .09em; }

.article-copy { width: min(var(--prose),calc(100% - 2 * var(--page-pad))); margin-inline: auto; }
.narrative { padding: var(--section-space) 0; }
.narrative .lede { margin: 0 0 42px; color: var(--ink); font-size: clamp(23px,2.4vw,31px); font-weight: 500; line-height: 1.38; letter-spacing: -.03em; text-wrap: balance; }
.article-copy p { margin: 0 0 24px; color: var(--ink-muted); font-size: clamp(16px,1.35vw,18px); line-height: 1.72; letter-spacing: -.008em; }
.article-copy p:last-child { margin-bottom: 0; }
.article-copy strong { color: var(--ink); font-weight: 550; }
.scope-note { margin-top: 44px; padding-top: 24px; font-size: 13px; line-height: 1.65; }
.study-at-a-glance { padding: 0 0 var(--section-space); }
.study-at-a-glance .section-heading { margin-bottom: 42px; }
.study-specs { width: min(var(--figure),calc(100% - 2 * var(--page-pad))); margin: auto; display: grid; grid-template-columns: repeat(5,1fr); border: 1px solid var(--line); border-radius: 16px; overflow: hidden; background: var(--line); gap: 1px; }
.spec-card { min-height: 158px; padding: 24px 21px; background: var(--surface-1); }
.spec-card span { display: block; margin-bottom: 24px; color: var(--accent-hover); font: 500 10px/1.3 var(--mono); letter-spacing: .05em; text-transform: uppercase; }
.spec-card strong { display: block; color: var(--ink); font-size: 14px; font-weight: 550; line-height: 1.45; }
.spec-card small { display: block; margin-top: 7px; color: var(--ink-subtle); font-size: 11px; line-height: 1.5; }
.research-question { width: min(var(--editorial),calc(100% - 2 * var(--page-pad))); margin: 0 auto var(--section-space); padding: 34px 0; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
.research-question span { display: block; margin-bottom: 16px; color: var(--accent-hover); font-size: 10px; font-weight: 600; letter-spacing: .09em; text-transform: uppercase; }
.research-question p { margin: 0; color: var(--ink); font-size: clamp(20px,2.25vw,29px); font-weight: 500; line-height: 1.43; letter-spacing: -.026em; text-wrap: balance; }

.section { position: relative; padding: var(--section-space) 0; border: 0; scroll-margin-top: 60px; }
.section::before { content: ""; position: absolute; top: 0; left: 50%; width: min(640px,calc(100% - 2 * var(--page-pad))); height: 1px; background: var(--line); transform: translateX(-50%); }
.section-heading { width: min(var(--editorial),calc(100% - 2 * var(--page-pad))); margin-bottom: 54px; }
h2, .display { font-size: clamp(32px,3.7vw,48px); line-height: 1.12; letter-spacing: -.038em; }
.section-deck { max-width: 760px; margin-top: 20px; font-size: clamp(16px,1.35vw,18px); line-height: 1.68; }
.chapter-copy { width: min(var(--prose),calc(100% - 2 * var(--page-pad))); margin: -20px auto 56px; }
.chapter-copy p { margin: 0 0 20px; color: var(--ink-subtle); font-size: 16px; line-height: 1.7; }

.journey { position: relative; height: 540vh; scroll-margin-top: 60px; }
.journey::before { content: ""; position: absolute; top: 0; left: 50%; width: min(640px,calc(100% - 2 * var(--page-pad))); height: 1px; background: var(--line); transform: translateX(-50%); }
.journey-sticky { position: sticky; top: 0; min-height: 100svh; display: grid; place-items: center; overflow: hidden; padding: 72px 0 44px; }
html:not(.motion-enhanced) .journey { height: auto; padding: var(--section-space) 0; }
html:not(.motion-enhanced) .journey-sticky { position: static; min-height: 0; padding: 0; overflow: visible; }
.journey-grid { width: min(var(--wide),calc(100% - 2 * var(--page-pad))); display: grid; grid-template-columns: minmax(560px,1.12fr) minmax(360px,.88fr); gap: clamp(52px,6vw,92px); align-items: center; }
.journey-visual { min-width: 0; }
.journey-visual svg { display: block; width: 100%; height: auto; overflow: visible; }
.journey-lane-bg { fill: var(--surface-1); stroke: var(--line); stroke-width: 1.2; }
.journey-lane-edge { fill: none; stroke: rgba(255,255,255,.045); stroke-width: 1; }
.journey-stage-no { fill: var(--accent-hover); font: 500 11px var(--mono); letter-spacing: .04em; }
.journey-stage-title { fill: var(--ink); font: 600 15px var(--sans); letter-spacing: -.01em; }
.journey-stage-meta { fill: var(--ink-tertiary); font: 500 9px var(--mono); letter-spacing: .05em; text-transform: uppercase; }
.journey-rail { fill: none; stroke: var(--line-strong); stroke-width: 1.4; stroke-linecap: round; }
.journey-rail-accent { fill: none; stroke: var(--accent); stroke-width: 1.8; stroke-linecap: round; }
.journey-connector { fill: none; stroke: var(--accent); stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; stroke-dasharray: 1; stroke-dashoffset: 0; }
.journey-node { fill: var(--accent); }
.journey-node-muted { fill: var(--line-strong); }
.journey-icon-line { fill: none; stroke: var(--ink-muted); stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
.journey-stage-group { transform-origin: center; }
.journey-copy .section-kicker { margin-bottom: 15px; }
.journey-copy .display { max-width: 480px; font-size: clamp(32px,3.45vw,46px); }
.journey-intro { max-width: 500px; margin: 20px 0 0; color: var(--ink-subtle); font-size: 14px; line-height: 1.62; }
.journey-steps { margin: 30px 0 0; padding: 0; list-style: none; border-top: 1px solid var(--line); }
.journey-step { position: relative; padding: 17px 0 17px 46px; border-bottom: 1px solid var(--line); opacity: 1; }
.journey-step::before { content: attr(data-index); position: absolute; left: 0; top: 19px; color: var(--ink-tertiary); font: 500 10px var(--mono); }
.journey-step small { display: block; margin-bottom: 4px; color: var(--ink-tertiary); font: 500 9px var(--mono); letter-spacing: .06em; text-transform: uppercase; }
.journey-step h3 { margin: 0 0 5px; font-size: 14px; font-weight: 600; letter-spacing: -.012em; }
.journey-step p { margin: 0; color: var(--ink-subtle); font-size: 12px; line-height: 1.52; }
.motion-enhanced .journey-step { opacity: 1; }
.journey-step.is-active { opacity: 1 !important; }
.journey-step.is-active::before, .journey-step.is-active small { color: var(--accent-hover); }

.reading-grid { width: min(var(--figure),calc(100% - 2 * var(--page-pad))); margin: auto; display: grid; grid-template-columns: repeat(4,1fr); gap: 1px; border: 1px solid var(--line); border-radius: 16px; overflow: hidden; background: var(--line); }
.reading-card { min-height: 240px; padding: 28px; background: var(--surface-1); }
.reading-card span { color: var(--accent-hover); font: 500 10px var(--mono); }
.reading-card h3 { margin: 34px 0 12px; font-size: 16px; font-weight: 600; letter-spacing: -.02em; }
.reading-card p { margin: 0; color: var(--ink-subtle); font-size: 12px; line-height: 1.62; }

.hypothesis-ledger { display: grid; grid-template-columns: repeat(2,1fr); gap: 1px; border: 1px solid var(--line); border-radius: 16px; overflow: hidden; background: var(--line); }
.hypothesis-row { min-height: 236px; margin: 0; padding: 26px; border: 0; background: var(--surface-1); }
.hypothesis-row:last-child { grid-column: 1 / -1; min-height: 0; }
.hypothesis-row-head { display: grid; grid-template-columns: 42px 1fr auto; gap: 12px; align-items: center; }
.hypothesis-id { font-size: 11px; }
.hypothesis-theme { color: var(--ink-tertiary); font: 500 9px var(--mono); letter-spacing: .06em; text-transform: uppercase; }
.hypothesis-result { max-width: 220px; font-size: 10px; text-align: right; white-space: normal; }
.hypothesis-row .hypothesis-title { margin: 30px 0 10px; font-size: 17px; font-weight: 600; line-height: 1.38; letter-spacing: -.02em; }
.hypothesis-row p { max-width: 560px; margin: 0; color: var(--ink-subtle); font-size: 12px; line-height: 1.62; }

.always-visible-evidence { margin-top: 30px; border: 1px solid var(--line); border-radius: 16px; background: var(--surface-1); overflow: hidden; }
.evidence-label-row { display: flex; align-items: center; justify-content: space-between; gap: 24px; padding: 15px 18px; border-bottom: 1px solid var(--line); color: var(--ink-muted); font-size: 12px; font-weight: 500; }
.evidence-label-row span:last-child { color: var(--ink-tertiary); font: 400 9px var(--mono); letter-spacing: .05em; text-transform: uppercase; }
.qc-frame { border: 0; border-radius: 0; }
.qc-preview { position: relative; max-height: 760px; overflow: hidden; border: 1px solid var(--line); border-radius: 10px; background: #fff; }
.qc-preview::after { display: none; }
.qc-preview img { border: 0; border-radius: 0; }
.qc-toolbar a { min-height: 40px; display: inline-flex; align-items: center; }
.table-wrap { scrollbar-color: var(--line-strong) var(--surface-1); }
.table-wrap:focus-visible { outline: 2px solid var(--accent-hover); outline-offset: 4px; }
.table-tools { border-top: 0; }
input, select { min-height: 44px; }
.provenance a, .usage-note a { min-height: 28px; display: inline-flex; align-items: center; }

.repo-shell { width: min(var(--figure),calc(100% - 2 * var(--page-pad))); margin: auto; }
.repo-card { position: relative; padding: clamp(28px,4vw,48px); border: 1px solid var(--line-strong); border-radius: 16px; background: var(--surface-1); overflow: hidden; }
.repo-card::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 2px; background: var(--accent); }
.repo-top { display: grid; grid-template-columns: auto 1fr auto; gap: 20px; align-items: center; }
.repo-icon { width: 40px; height: 40px; color: var(--ink); }
.repo-path { display: block; color: var(--ink-tertiary); font: 500 10px var(--mono); text-transform: uppercase; letter-spacing: .06em; }
.repo-title { display: block; margin-top: 3px; color: var(--ink); font-size: 20px; font-weight: 600; letter-spacing: -.025em; }
.repo-link { min-height: 44px; display: inline-flex; align-items: center; padding: 0 14px; border: 1px solid var(--line-strong); border-radius: 8px; color: var(--ink); background: var(--surface-2); font-size: 12px; font-weight: 550; text-decoration: none; }
.repo-description { max-width: 760px; margin: 30px 0 0; color: var(--ink-subtle); font-size: 14px; line-height: 1.7; }
.repo-tags { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 24px; }
.repo-tags span { padding: 5px 8px; border: 1px solid var(--line); border-radius: 5px; color: var(--ink-subtle); font: 400 10px var(--mono); }
.usage-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 18px; margin-top: 28px; }
.repo-command { min-width: 0; padding: 24px; border: 1px solid var(--line); border-radius: 12px; background: var(--surface-1); }
.repo-command > span { color: var(--accent-hover); font: 500 10px var(--mono); }
.repo-command h3 { margin: 20px 0 8px; font-size: 15px; font-weight: 600; }
.repo-command p { min-height: 58px; margin: 0 0 18px; color: var(--ink-subtle); font-size: 11px; line-height: 1.58; }
.repo-command pre { min-height: 124px; margin: 0; padding: 14px; overflow: auto; border: 1px solid var(--line); border-radius: 8px; color: var(--ink-muted); background: var(--canvas); font: 400 10px/1.65 var(--mono); white-space: pre-wrap; }
.usage-note { margin: 24px 0 0; color: var(--ink-subtle); font-size: 12px; line-height: 1.62; }

footer { position: relative; padding: 0 var(--page-pad) 48px; border: 0; color: var(--ink-subtle); font-size: 11px; }
.footer-inner { width: min(var(--wide),100%); display: block; }
.footer-divider { width: min(640px,100%); height: 1px; margin: 0 auto 56px; background: var(--line); }
.footer-grid { display: grid; grid-template-columns: 1.35fr repeat(3,1fr); gap: 48px; }
.footer-brand { color: var(--ink); font-size: 15px; font-weight: 600; }
.footer-brand-row { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
.footer-summary { max-width: 320px; margin: 0; color: var(--ink-subtle); font-size: 12px; line-height: 1.62; }
.footer-column h3 { margin: 0 0 15px; color: var(--ink-muted); font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; }
.footer-column p, .footer-column a { display: block; margin: 0 0 8px; color: var(--ink-subtle); font-size: 11px; line-height: 1.58; }
.footer-column a { min-height: 32px; display: flex; align-items: center; margin-bottom: 0; }
.footer-column a:hover { color: var(--ink); }
.footer-bottom { display: flex; justify-content: space-between; gap: 32px; margin-top: 48px; padding-top: 20px; border-top: 1px solid var(--line); color: var(--ink-subtle); font-size: 11px; line-height: 1.58; }
.footer-bottom span { max-width: 560px; }

.gsap-ready .reveal { visibility: hidden; }
@media (max-width: 1120px) {
  .study-specs { grid-template-columns: repeat(3,1fr); }
  .spec-card:nth-child(n+4) { min-height: 142px; }
  .journey-grid { grid-template-columns: minmax(500px,1.05fr) minmax(330px,.95fr); gap: 44px; }
  .reading-grid { grid-template-columns: repeat(2,1fr); }
  .usage-grid { grid-template-columns: 1fr 1fr; }
  .repo-command:last-child { grid-column: 1 / -1; }
}
@media (max-width: 1200px) {
  .journey { height: auto; padding: var(--section-space) 0; }
  .journey-sticky { position: static; min-height: 0; padding: 0; overflow: visible; }
  .journey-grid { grid-template-columns: 1fr; }
  .journey-visual { order: 2; }
  .journey-copy { order: 1; }
  .journey-copy .display, .journey-intro { max-width: 680px; }
  .journey-step, .motion-enhanced .journey-step { opacity: 1 !important; }
  .journey-stage-group, .journey-connector { opacity: 1 !important; transform: none !important; stroke-dashoffset: 0 !important; }
}
@media (max-width: 900px) {
  .hero { min-height: 850px; }
  .hero-shell { min-height: 790px; grid-template-columns: 1fr; }
  .hero-copy { padding-top: 122px; }
  .hypothesis-ledger { grid-template-columns: 1fr; }
  .hypothesis-row:last-child { grid-column: auto; }
  .footer-grid { grid-template-columns: 1.25fr repeat(2,1fr); }
  .footer-column:last-child { grid-column: 2 / -1; }
}
@media (max-width: 720px) {
  :root { --page-pad: 22px; --section-space: 96px; }
  .brand, .menu-button { position: relative; z-index: 2; }
  .nav-links { z-index: 1; top: 66px; left: 12px; right: 12px; padding: 10px 18px 16px; border: 1px solid var(--line); border-radius: 12px; background: #0f1011; }
  .hero { min-height: 810px; }
  .hero-shell { min-height: 750px; }
  .hero-copy { padding-top: 104px; }
  .hero h1 { font-size: clamp(38px,12vw,50px); }
  .hero-lead { font-size: 15px; }
  .narrative .lede { font-size: 23px; }
  .article-copy p { font-size: 16px; }
  .study-specs { grid-template-columns: 1fr 1fr; }
  .spec-card { min-height: 148px; }
  .spec-card:last-child { grid-column: 1 / -1; }
  .research-question { padding: 28px 0; }
  h2, .display { font-size: clamp(31px,10vw,42px); }
  .section-heading { margin-bottom: 42px; }
  .journey-grid { gap: 44px; }
  .journey-visual svg { width: 112%; margin-left: -6%; }
  .journey-step { padding-left: 40px; }
  .reading-grid, .usage-grid { grid-template-columns: 1fr; }
  .reading-card { min-height: 0; }
  .repo-command:last-child { grid-column: auto; }
  .repo-top { grid-template-columns: auto 1fr; }
  .repo-link { grid-column: 1 / -1; justify-self: start; }
  .repo-command p { min-height: 0; }
  .hypothesis-row { min-height: 0; }
  .hypothesis-row-head { grid-template-columns: 38px 1fr; }
  .hypothesis-result { grid-column: 1 / -1; max-width: none; text-align: left; }
  .metric-axis, .range-axis { font-size: 10px; }
  .evidence-label-row { align-items: flex-start; flex-direction: column; gap: 5px; }
  .qc-preview { max-height: 620px; }
  .footer-grid { grid-template-columns: 1fr; gap: 34px; }
  .footer-column:last-child { grid-column: auto; }
  .footer-column a { min-height: 44px; }
  .footer-bottom { flex-direction: column; }
  #comparison-table td[data-label="Comparison"] .comparison-id { grid-column: 2; margin-top: -4px; }
}
@media (prefers-reduced-motion: reduce) {
  .site-header { transform: none !important; }
  .brand-label, .journey-stage-group, .journey-connector, .journey-step { opacity: 1 !important; transform: none !important; stroke-dashoffset: 0 !important; }
  .gsap-ready .reveal { visibility: visible !important; opacity: 1 !important; transform: none !important; }
}
@media print {
  .site-header, .menu-button, #hero-canvas { display: none !important; }
  .hero { min-height: auto; padding-top: 48px; }
  .hero-shell { min-height: auto; grid-template-columns: 1fr; }
  .journey { height: auto; padding: 72px 0; }
  .journey-sticky { position: static; min-height: 0; padding: 0; }
  .journey-grid { grid-template-columns: 1fr; }
  .journey-stage-group, .journey-connector, .journey-step, .reveal { visibility: visible !important; opacity: 1 !important; transform: none !important; stroke-dashoffset: 0 !important; }
  .section, .repo-card, .hypothesis-row { break-inside: avoid; }
}
"""
)


_MVP_SCRIPT_V4 = r"""
(() => {
  const root = document.documentElement;
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const gsapEngine = window.gsap;
  const scrollEngine = window.ScrollTrigger;
  const motionAvailable = Boolean(gsapEngine && scrollEngine && !reduced);
  if (gsapEngine && scrollEngine) gsapEngine.registerPlugin(scrollEngine);

  const header = document.getElementById('site-header');
  const brandLabel = document.querySelector('.brand-label');
  const menuButton = document.getElementById('menu-button');
  const navLinks = document.getElementById('nav-links');
  let lastScroll = window.scrollY;
  let headerVisible = true;

  const setHeaderVisible = (visible, immediate = false) => {
    if (visible === headerVisible && !immediate) return;
    headerVisible = visible;
    if (gsapEngine && !reduced) {
      const duration = immediate ? 0 : .34;
      gsapEngine.to(header, {
        yPercent: visible ? 0 : -112,
        duration,
        ease: visible ? 'power3.out' : 'power2.in',
        overwrite: true,
      });
      if (visible) {
        gsapEngine.fromTo(brandLabel, {x: -17, autoAlpha: 0}, {
          x: 0,
          autoAlpha: 1,
          duration: immediate ? .01 : .46,
          delay: immediate ? 0 : .09,
          ease: 'power3.out',
          overwrite: true,
        });
      } else {
        gsapEngine.to(brandLabel, {x: -14, autoAlpha: 0, duration: .16, overwrite: true});
      }
    } else {
      header.style.transform = visible ? 'translateY(0)' : 'translateY(-112%)';
      brandLabel.style.opacity = visible ? '1' : '0';
      brandLabel.style.transform = visible ? 'translateX(0)' : 'translateX(-14px)';
    }
  };

  const updateChrome = () => {
    const current = Math.max(0, window.scrollY);
    const delta = current - lastScroll;
    header.classList.toggle('is-scrolled', current > 18);
    if (navLinks.classList.contains('is-open') || current < 72 || reduced) {
      setHeaderVisible(true);
    } else if (delta > 1.5) {
      setHeaderVisible(false);
    } else if (delta < -.25) {
      setHeaderVisible(true);
    }
    lastScroll = current;
  };
  setHeaderVisible(true, true);
  window.addEventListener('scroll', updateChrome, {passive: true});

  menuButton.addEventListener('click', () => {
    const open = !navLinks.classList.contains('is-open');
    navLinks.classList.toggle('is-open', open);
    menuButton.setAttribute('aria-expanded', String(open));
    menuButton.textContent = open ? 'Close' : 'Menu';
    setHeaderVisible(true);
  });
  navLinks.addEventListener('click', event => {
    if (event.target instanceof HTMLAnchorElement) {
      navLinks.classList.remove('is-open');
      menuButton.setAttribute('aria-expanded', 'false');
      menuButton.textContent = 'Menu';
    }
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && navLinks.classList.contains('is-open')) {
      navLinks.classList.remove('is-open');
      menuButton.setAttribute('aria-expanded', 'false');
      menuButton.textContent = 'Menu';
      menuButton.focus();
    }
  });

  const journey = document.getElementById('method');
  const journeyGroups = [...document.querySelectorAll('.journey-stage-group')];
  const journeyConnectors = [...document.querySelectorAll('.journey-connector')];
  const journeySteps = [...document.querySelectorAll('[data-journey-copy]')];
  let activeJourneyStage = -1;
  const setJourneyStage = (active, progress = 1) => {
    const next = Math.max(0, Math.min(4, active));
    activeJourneyStage = next;
    journey.dataset.activeStage = String(next);
    journeyGroups.forEach((group, index) => {
      group.classList.toggle('is-active', index === next);
      group.classList.toggle('is-complete', index < next);
      if (motionAvailable && window.innerWidth > 1200) {
        gsapEngine.to(group, {
          autoAlpha: index === next ? 1 : index < next ? .56 : .035,
          x: index > next ? (index % 2 ? 30 : -30) : 0,
          scale: index === next ? 1 : .985,
          duration: .48,
          ease: 'power3.out',
          overwrite: true,
        });
      }
    });
    journeyConnectors.forEach((connector, index) => {
      const visible = index < next;
      if (motionAvailable && window.innerWidth > 1200) {
        gsapEngine.to(connector, {
          strokeDashoffset: visible ? 0 : 1,
          opacity: visible ? .9 : .08,
          duration: .5,
          ease: 'power2.out',
          overwrite: true,
        });
      }
    });
    journeySteps.forEach((step, index) => step.classList.toggle('is-active', index === next));
    const indicator = document.getElementById('journey-indicator');
    if (indicator) indicator.style.transform = `scaleX(${Math.max(.02, progress)})`;
  };

  if (motionAvailable) {
    root.classList.add('motion-enhanced', 'gsap-ready');
    journeyConnectors.forEach(connector => gsapEngine.set(connector, {strokeDashoffset: 1, opacity: .08}));
    setJourneyStage(0, 0);
    scrollEngine.create({
      trigger: journey,
      start: 'top top',
      end: 'bottom bottom',
      onUpdate: self => setJourneyStage(Math.min(4, Math.floor(self.progress * 4.9999)), self.progress),
      onLeaveBack: () => setJourneyStage(0, 0),
    });
    const mobileJourney = window.matchMedia('(max-width: 1200px)');
    const syncJourneyMode = () => {
      if (mobileJourney.matches) setJourneyStage(4, 1);
      else setJourneyStage(activeJourneyStage < 0 ? 0 : activeJourneyStage);
    };
    mobileJourney.addEventListener('change', syncJourneyMode);
    syncJourneyMode();
  } else {
    setJourneyStage(4, 1);
  }

  const revealItems = [...document.querySelectorAll('.reveal')];
  if (motionAvailable) {
    gsapEngine.set(revealItems, {autoAlpha: 0, y: 24});
    revealItems.forEach(item => {
      gsapEngine.to(item, {
        autoAlpha: 1,
        y: 0,
        duration: .82,
        ease: 'power3.out',
        scrollTrigger: {trigger: item, start: 'top 87%', once: true},
      });
    });
    gsapEngine.from('.hero-animate', {
      autoAlpha: 0,
      y: 26,
      duration: .9,
      stagger: .095,
      ease: 'power3.out',
      delay: .12,
    });
    gsapEngine.to('#hero-canvas', {
      yPercent: 9,
      opacity: .62,
      ease: 'none',
      scrollTrigger: {trigger: '.hero', start: 'top top', end: 'bottom top', scrub: .5},
    });
    document.querySelectorAll('.h4-chart, .h2-chart, .forest-plot').forEach(chart => {
      const marks = chart.querySelectorAll(
        '.metric-dot, .difference-ci, .difference-point, .range-line, .range-start, ' +
        '.range-end, .forest-ci, .forest-point'
      );
      gsapEngine.from(marks, {
        scale: 0,
        duration: .48,
        stagger: .012,
        ease: 'back.out(1.55)',
        scrollTrigger: {trigger: chart, start: 'top 84%', once: true},
      });
    });
    document.querySelectorAll('.spec-card, .reading-card, .repo-command, .rule').forEach((card, index) => {
      gsapEngine.from(card, {
        y: 18,
        autoAlpha: 0,
        duration: .62,
        delay: Math.min(index % 5, 4) * .035,
        ease: 'power2.out',
        scrollTrigger: {trigger: card, start: 'top 102%', once: true},
      });
    });
    document.querySelectorAll('main section[id]').forEach(section => {
      const link = document.querySelector(`.nav-links a[href="#${section.id}"]`);
      if (!link) return;
      scrollEngine.create({
        trigger: section,
        start: 'top 46%',
        end: 'bottom 46%',
        onToggle: self => link.classList.toggle('is-active', self.isActive),
      });
    });
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => scrollEngine.refresh());
    }
  }

  const hypothesisFilter = document.getElementById('filter-hypothesis');
  const statusFilter = document.getElementById('filter-status');
  const searchFilter = document.getElementById('filter-search');
  const comparisonRows = [...document.querySelectorAll('[data-comparison-row]')];
  const tableCount = document.getElementById('table-count');
  let tableRefreshRequest = 0;
  const scheduleTableRefresh = () => {
    if (!motionAvailable) return;
    if (tableRefreshRequest) cancelAnimationFrame(tableRefreshRequest);
    tableRefreshRequest = requestAnimationFrame(() => {
      tableRefreshRequest = 0;
      scrollEngine.refresh();
    });
  };
  const filterTable = () => {
    const hypothesis = hypothesisFilter.value;
    const status = statusFilter.value;
    const query = searchFilter.value.trim().toLowerCase();
    let visible = 0;
    comparisonRows.forEach(row => {
      const matches = (hypothesis === 'ALL' || row.dataset.hypothesis === hypothesis) &&
        (status === 'ALL' || row.dataset.status === status) &&
        (!query || row.textContent.toLowerCase().includes(query));
      row.hidden = !matches;
      if (matches) visible += 1;
    });
    tableCount.textContent = `${visible} / ${comparisonRows.length} rows`;
    scheduleTableRefresh();
  };
  [hypothesisFilter, statusFilter, searchFilter].forEach(control =>
    control.addEventListener('input', filterTable));
  filterTable();
})();
"""


def _render_html_v3_legacy(evidence: dict[str, Any]) -> str:
    primary = evidence["primary"]
    qc = evidence["pannuke_qc"]
    hypotheses = primary["hypothesis_comparisons"]
    h2 = primary["h2_subgroups"]
    h4 = primary["h4_restoration"]
    seed_audit = primary["instance_dependent_seed_audit"]

    comparison_rows = _render_comparison_rows_v3(primary["comparisons"])
    forest_plot = _render_forest_plot_v3(primary["comparisons"])
    h4_chart = _render_h4_chart_v3(h4)
    h2_chart = _render_h2_chart_v3(h2)
    hypothesis_ledger = _render_hypothesis_ledger_v4(hypotheses, h2, h4)

    seed_rows: list[str] = []
    for record in seed_audit["records"]:
        ranking_hash = str(record["ranking_sha256"])
        oof_hash = str(record["oof_predictions_sha256"])
        seed_rows.append(
            "<tr>"
            f'<td data-label="Seed">{record["seed"]}</td>'
            f'<td data-label="Cell ID"><code>{html.escape(str(record["cell_id"]))}</code></td>'
            f'<td data-label="Ranking SHA-256"><code title="{ranking_hash}">'
            f"{ranking_hash[:16]}…</code></td>"
            f'<td data-label="OOF SHA-256"><code title="{oof_hash}">'
            f"{oof_hash[:16]}…</code></td>"
            "</tr>"
        )
    seed_hash = html.escape(str(seed_audit["records"][0]["ranking_sha256"]))
    oof_hash = html.escape(str(seed_audit["records"][0]["oof_predictions_sha256"]))
    point_difference = _require_real(h4["point_difference"], role="H4 point difference")
    reported_comparisons = sum(record["status"] == "reported" for record in primary["comparisons"])
    unavailable_comparisons = primary["comparison_count"] - reported_comparisons

    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="AANCA is a reproducible, non-diagnostic framework for ranking potentially inconsistent nucleus annotations for expert review.">
  <meta name="author" content="Natan Smogór">
  <meta name="date" content="2026-08-18">
  <meta name="theme-color" content="#010102">
  <meta name="referrer" content="no-referrer">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&amp;family=JetBrains+Mono:wght@400;500&amp;display=swap" rel="stylesheet">
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%23010102'/%3E%3Crect x='14' y='14' width='12' height='12' rx='3' fill='%235e6ad2'/%3E%3Crect x='38' y='14' width='12' height='12' rx='3' fill='%23828fff'/%3E%3Crect x='26' y='26' width='12' height='12' rx='3' fill='%235e6ad2'/%3E%3Crect x='14' y='38' width='12' height='12' rx='3' fill='%23828fff'/%3E%3Crect x='38' y='38' width='12' height='12' rx='3' fill='%235e6ad2'/%3E%3C/svg%3E">
  <title>AANCA — annotation auditing for expert review</title>
  <style>__CSS__</style>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="site-header" id="site-header">
  <div class="nav-shell">
    <a class="brand" href="#top" aria-label="AANCA — back to the beginning"><span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span>AANCA</a>
    <button class="menu-button" id="menu-button" type="button" aria-expanded="false" aria-controls="nav-links">Menu</button>
    <nav class="nav-links" id="nav-links" aria-label="Presentation sections"><a href="#overview">Overview</a><a href="#method">Method</a><a href="#results">Results</a><a href="#benchmarks">Benchmarks</a><a href="#evidence">Evidence</a></nav>
  </div>
  <div class="reading-progress" id="reading-progress" aria-hidden="true"></div>
</header>

<section class="hero" id="top" aria-labelledby="hero-title">
  <canvas id="hero-canvas" aria-hidden="true"></canvas>
  <div class="hero-shell">
    <div class="hero-copy">
      <p class="eyebrow hero-animate">Automated nucleus-annotation auditing</p>
      <h1 id="hero-title" class="hero-animate">Which annotations deserve a second look?</h1>
      <p class="hero-lead hero-animate">A reproducible, group-safe framework that assigns review priority to each <em>potentially inconsistent annotation</em>. High-priority cases are <em>recommended for expert review</em>—without changing the source labels.</p>
      <div class="hero-byline hero-animate"><span>By <strong>Natan Smogór</strong></span><span>Released <time datetime="2026-08-18">18 August 2026</time></span></div>
    </div>
    <span class="hero-visual-label" aria-hidden="true">Source annotations stay fixed → review evidence is ranked<br>Conceptual workflow · not benchmark data</span>
  </div>
</section>

<main id="main">
  <section class="intro" id="overview">
    <div class="prose reveal">
      <p class="intro-lead">A model can identify labels that do not fit the patterns learned from the rest of a dataset. That is a triage signal—not a verdict.</p>
      <div class="intro-copy">
        <p>AANCA tests whether that signal can move intentionally corrupted nucleus labels toward the front of a fixed expert-review queue. Because the benchmark knows exactly which labels were changed, ranking quality can be measured objectively.</p>
        <p>The system never decides that an annotator was wrong. It preserves every source annotation and produces only a priority order for review by a qualified expert.</p>
      </div>
      <div class="scope-note"><strong>Scope.</strong> This presentation reports the completed frozen-feature study. Confirmatory CNN experiments, blinded expert review and external validation have not been performed. The accepted analysis is presented as exploratory because outcomes were inspected during recovery.</div>
    </div>
  </section>

  <div class="question-panel reveal">
    <span>Research question</span>
    <p>Can group-safe out-of-fold models find controlled label inconsistencies more efficiently than random review—and does correcting the highest-ranked labels improve downstream classification?</p>
  </div>

  <section class="story" id="method" aria-labelledby="story-title">
    <div class="story-frame"><div class="story-inner">
      <div class="workflow-panel" id="workflow-panel" data-active-stage="0">
        <svg viewBox="0 0 720 680" role="img" aria-labelledby="workflow-title workflow-desc">
          <title id="workflow-title">Five stages of the controlled AANCA benchmark</title>
          <desc id="workflow-desc">Source labels, controlled label corruption, group-safe out-of-fold predictions, risk ranking and an expert-review queue.</desc>
          <path data-stage="1" class="workflow-connector is-accent" d="M112 128V166"/>
          <path data-stage="2" class="workflow-connector is-accent" d="M112 258V296"/>
          <path data-stage="3" class="workflow-connector is-accent" d="M112 388V426"/>
          <path data-stage="4" class="workflow-connector is-accent" d="M112 518V556"/>
          <g data-stage="0"><rect class="workflow-node" x="62" y="36" width="596" height="92" rx="14"/><text class="workflow-step-no" x="88" y="67">01</text><text class="workflow-title" x="132" y="68">Immutable source labels</text><text class="workflow-copy" x="132" y="94">Each nucleus remains linked to its source patch and original class.</text><g transform="translate(568 58)"><rect class="workflow-icon" width="16" height="16" rx="4"/><rect class="workflow-icon" x="23" y="7" width="16" height="16" rx="4" opacity=".7"/><rect class="workflow-icon" x="7" y="27" width="16" height="16" rx="4" opacity=".85"/></g></g>
          <g data-stage="1"><rect class="workflow-node" x="62" y="166" width="596" height="92" rx="14"/><text class="workflow-step-no" x="88" y="197">02</text><text class="workflow-title" x="132" y="198">Controlled label corruption</text><text class="workflow-copy" x="132" y="224">Known interventions create objective targets for ranking evaluation.</text><g transform="translate(558 192)"><rect x="0" y="0" width="38" height="10" rx="5" fill="#62666d"/><path d="M43 5H58" stroke="#828fff"/><rect class="workflow-icon" x="63" y="0" width="38" height="10" rx="5"/></g></g>
          <g data-stage="2"><rect class="workflow-node" x="62" y="296" width="596" height="92" rx="14"/><text class="workflow-step-no" x="88" y="327">03</text><text class="workflow-title" x="132" y="328">Group-safe out-of-fold predictions</text><text class="workflow-copy" x="132" y="354">A nucleus is scored only by a model that never trained on its patch.</text><g transform="translate(563 322)"><rect class="workflow-icon" width="27" height="24" rx="5"/><rect x="34" width="27" height="24" rx="5" fill="#34343a"/><rect x="68" width="27" height="24" rx="5" fill="#34343a"/></g></g>
          <g data-stage="3"><rect class="workflow-node" x="62" y="426" width="596" height="92" rx="14"/><text class="workflow-step-no" x="88" y="457">04</text><text class="workflow-title" x="132" y="458">Risk ranking at a fixed review budget</text><text class="workflow-copy" x="132" y="484">Higher score means earlier review; it does not mean “confirmed error.”</text><g transform="translate(556 449)"><rect class="workflow-icon" width="104" height="7" rx="3.5"/><rect x="0" y="15" width="74" height="7" rx="3.5" fill="#62666d"/><rect x="0" y="30" width="42" height="7" rx="3.5" fill="#34343a"/></g></g>
          <g data-stage="4"><rect class="workflow-node" x="62" y="556" width="596" height="92" rx="14"/><text class="workflow-step-no" x="88" y="587">05</text><text class="workflow-title" x="132" y="588">Expert-review queue</text><text class="workflow-copy" x="132" y="614">Recommendation only. Source annotations are never changed automatically.</text><g transform="translate(584 578)"><circle cx="24" cy="24" r="23" fill="none" stroke="#5e6ad2" stroke-width="2"/><path d="M13 24l8 8 16-20" fill="none" stroke="#828fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></g></g>
        </svg>
      </div>
      <div class="story-copy">
        <p class="section-kicker">How the benchmark works</p>
        <h2 id="story-title" class="display">From source label to review priority.</h2>
        <ol class="story-steps">
          <li class="story-step is-active" data-story-step="0" data-index="01"><h3>Preserve the original evidence</h3><p>Every nucleus keeps its patch-level group and pre-corruption class.</p></li>
          <li class="story-step" data-story-step="1" data-index="02"><h3>Create known inconsistencies</h3><p>Labels are intentionally changed under four fixed mechanisms and saved separately.</p></li>
          <li class="story-step" data-story-step="2" data-index="03"><h3>Predict without group leakage</h3><p>Out-of-fold models never see the scored nucleus or any nucleus from its source patch.</p></li>
          <li class="story-step" data-story-step="3" data-index="04"><h3>Rank, then measure retrieval</h3><p>Suspicion scores order a fixed review queue; average precision measures how well it works.</p></li>
          <li class="story-step" data-story-step="4" data-index="05"><h3>Leave the decision to an expert</h3><p>The output recommends what to inspect first. It never edits or diagnoses.</p></li>
        </ol>
      </div>
    </div></div>
  </section>

  <section class="section" id="results">
    <div class="section-heading"><p class="section-kicker">Primary downstream test · H4</p><h2>Better triage did not improve the downstream model.</h2><p class="section-deck">This is the most important negative result: ranking quality and downstream utility are related questions, but they are not the same question.</p></div>
    <div class="figure-width reveal result-layout">
      <div class="result-lead"><div class="result-label"><span>Audit-guided minus random review</span><span>Adverse to the registered hypothesis</span></div><span class="result-number">__H4_POINT__</span><p>Audit-guided restoration produced macro-F1 <strong>__H4_AUDIT__</strong>, below the random-review mean of <strong>__H4_RANDOM__</strong>.</p><div class="result-interpretation"><b>Plain-language interpretation</b><p>The audit-guided result was <strong>__H4_POINT_PP__ percentage points lower</strong>. The saved 95% interval remained below zero, so this registered test did not support downstream benefit.</p></div></div>
      __H4_CHART__
    </div>
  </section>

  <section class="section" id="benchmarks">
    <div class="section-heading"><p class="section-kicker">Registered questions · H1-H7</p><h2>Seven questions. One complete answer set.</h2><p class="section-deck">Positive, neutral, adverse and unavailable outcomes are shown together. Nothing is selected because it looks favourable.</p></div>
    <div class="figure-width reveal"><div class="hypothesis-ledger">__HYPOTHESIS_LEDGER__</div></div>
    <div class="section-heading" style="margin-top:clamp(92px,10vw,136px)"><p class="section-kicker">Comparison atlas</p><h2>All 36 frozen comparisons.</h2><p class="section-deck">Each point is an average-precision difference; each line is the saved 95% percentile-bootstrap interval.</p></div>
    <div class="wide-width reveal"><div class="forest-plot" aria-label="Forest plot of all H1, H3, H5, H6 and H7 comparisons">__FOREST_PLOT__</div><p class="fine"><strong>Statistical reading.</strong> The registered Holm-adjusted p-values are one-sided, while the displayed 95% intervals are two-sided summaries. H6 is unavailable by design; it is not a zero result.</p></div>
  </section>

  <section class="section" id="subgroups">
    <div class="section-heading"><p class="section-kicker">H2 · descriptive heterogeneity</p><h2>Performance depends on context.</h2><p class="section-deck">The saved subgroup estimates span nucleus classes, tissues, corruption mechanisms and corruption rates. They describe variation; they do not establish biological causation.</p></div>
    <div class="figure-width reveal">__H2_CHART__</div>
  </section>

  <section class="section" id="seed-audit">
    <div class="section-heading"><p class="section-kicker">Reproducibility disclosure</p><h2>Three registered seeds produced one realisation.</h2><p class="section-deck">The rows remain in the frozen result set, but they must not be interpreted as three independent replications.</p></div>
    <div class="figure-width reveal">
      <div class="seed-panel">
        <svg viewBox="0 0 680 390" role="img" aria-labelledby="seed-title seed-desc"><title id="seed-title">Seeds 404, 405 and 406 produced byte-identical files</title><desc id="seed-desc">Three seed paths converge on one ranking hash and one out-of-fold prediction hash.</desc>
          <g font-family="Inter, sans-serif" font-size="13" fill="#d0d6e0"><rect x="34" y="42" width="126" height="54" rx="10" fill="#141516" stroke="#34343a"/><text x="72" y="75">Seed 404</text><rect x="34" y="168" width="126" height="54" rx="10" fill="#141516" stroke="#34343a"/><text x="72" y="201">Seed 405</text><rect x="34" y="294" width="126" height="54" rx="10" fill="#141516" stroke="#34343a"/><text x="72" y="327">Seed 406</text></g>
          <g fill="none" stroke="#5e6ad2" stroke-width="2"><path d="M160 69H270Q310 69 310 109V168"/><path d="M160 195H310"/><path d="M160 321H270Q310 321 310 281V222"/></g><circle cx="310" cy="195" r="18" fill="#5e6ad2"/><path d="M328 195H404" stroke="#828fff" stroke-width="2"/>
          <g font-family="Inter, sans-serif"><rect x="404" y="130" width="238" height="58" rx="10" fill="#141516" stroke="#34343a"/><text x="428" y="154" fill="#8a8f98" font-size="11">IDENTICAL RANKING SHA-256</text><text x="428" y="174" fill="#d0d6e0" font-size="12">__SEED_HASH_SHORT__…</text><rect x="404" y="204" width="238" height="58" rx="10" fill="#141516" stroke="#34343a"/><text x="428" y="228" fill="#8a8f98" font-size="11">IDENTICAL OOF SHA-256</text><text x="428" y="248" fill="#d0d6e0" font-size="12">__OOF_HASH_SHORT__…</text></g>
        </svg>
        <div class="seed-copy"><h3>Seeds 404, 405 and 406 are not independent realisations.</h3><p>The ranking files and out-of-fold predictions are byte-identical. The presentation therefore treats them as one deterministic scenario.</p><p><code>ranking __SEED_HASH__</code><br><code>OOF __OOF_HASH__</code></p></div>
      </div>
      <details class="evidence-details"><summary>Inspect complete hashes and cell IDs</summary><div class="table-wrap"><table class="compact"><thead><tr><th>Seed</th><th>Cell ID</th><th>Ranking SHA-256</th><th>OOF SHA-256</th></tr></thead><tbody>__SEED_ROWS__</tbody></table></div></details>
    </div>
  </section>

  <section class="section" id="integrity">
    <div class="section-heading"><p class="section-kicker">Experimental integrity</p><h2>Outcome information cannot leak into model choice.</h2><p class="section-deck">The design separates groups, labels and evaluation roles before any result is interpreted.</p></div>
    <div class="wide-width reveal">
      <div class="stats-grid"><div class="stat"><strong>__COMPLETED__/185</strong><span>required primary cells completed</span></div><div class="stat"><strong>__FAILED__</strong><span>failed required cells</span></div><div class="stat"><strong>__COMPARISONS__</strong><span>saved registered comparisons</span></div><div class="stat"><strong>__BOOTSTRAPS__</strong><span>paired group-bootstrap iterations</span></div></div>
      <div class="rules"><div class="rule"><b>Group-safe splitting</b><p>Every split uses <code>group_id</code>, at least at source-patch level—never individual nuclei.</p></div><div class="rule"><b>Out-of-fold ranking</b><p>Primary model-based audit scores come from group-safe predictions made on unseen groups.</p></div><div class="rule"><b>Untouched reference fold</b><p>The final reference fold remains uncorrupted and unavailable for model selection or tuning.</p></div><div class="rule"><b>Separate label states</b><p><code>pre_corruption_label</code>, <code>observed_label</code> and all corruption metadata remain distinct.</p></div></div>
    </div>
  </section>

  <section class="section" id="interpretation">
    <div class="section-heading"><p class="section-kicker">Interpretation boundary</p><h2>What this study supports—and what it does not.</h2></div>
    <div class="figure-width reveal claim-grid">
      <article class="claim-card"><h3>Supported by this benchmark</h3><ul><li>Group-safe ranking retrieved injected label corruptions more efficiently than random review in the registered H1 comparisons.</li><li>The fixed hybrid exceeded self-confidence in all 12 registered H5 comparisons.</li><li>The complete workflow is reproducible from sealed, machine-readable evidence.</li></ul></article>
      <article class="claim-card"><h3>Not established here</h3><ul><li>That naturally occurring labels are wrong, or that a pathologist made an error.</li><li>That ranking automatically improves downstream classification—the H4 result was adverse.</li><li>Clinical validity, expert agreement or external generalisation.</li></ul></article>
    </div>
  </section>

  <section class="section" id="quality">
    <div class="section-heading"><p class="section-kicker">PanNuke quality control</p><h2>Source masks remained untouched.</h2><p class="section-deck">Cross-class overlaps and unlabeled regions were measured and retained in provenance. The pipeline never arbitrated an overlap class or reconstructed background.</p></div>
    <div class="wide-width reveal">
      <div class="stats-grid"><div class="stat"><strong>__PATCHES__</strong><span>validated PanNuke patches</span></div><div class="stat"><strong>__OVERLAP_PIXELS__</strong><span>cross-class overlap pixels</span></div><div class="stat"><strong>__VOID_PIXELS__</strong><span>unlabeled / void pixels</span></div><div class="stat"><strong>__FLAGGED_INSTANCES__</strong><span>overlap-touching instances flagged</span></div></div>
      <details class="evidence-details"><summary>Open the deterministic QC overlay</summary><div class="qc-frame"><div class="qc-toolbar"><span>1512 x 3840 px · original file</span><a href="pannuke_mask_qc_overlays.png" target="_blank" rel="noopener">Open full resolution ↗</a></div><img loading="lazy" src="pannuke_mask_qc_overlays.png" alt="Deterministic PanNuke quality-control overlays showing images and mask boundaries"></div></details>
    </div>
  </section>

  <section class="section" id="evidence">
    <div class="section-heading"><p class="section-kicker">Evidence and provenance</p><h2>Every displayed result remains inspectable.</h2><p class="section-deck">The table retains all 36 comparisons, with consistent precision and readable names. Raw comparison IDs remain visible for exact traceability.</p></div>
    <div class="wide-width reveal">
      <details class="evidence-details" open><summary>Complete H1 / H3 / H5 / H6 / H7 comparison table</summary>
        <div class="table-tools"><label>Hypothesis<select id="filter-hypothesis"><option value="ALL">All hypotheses</option><option>H1</option><option>H3</option><option>H5</option><option>H6</option><option>H7</option></select></label><label>Status<select id="filter-status"><option value="ALL">All statuses</option><option value="reported">Reported</option><option value="not_available_frozen_optional_cell">Unavailable</option></select></label><label>Search<input id="filter-search" type="search" placeholder="Name, seed or comparison ID"></label><span class="table-count" id="table-count">__COMPARISONS__ / __COMPARISONS__ rows</span></div>
        <div class="table-wrap"><table id="comparison-table"><colgroup><col style="width:7%"><col style="width:35%"><col style="width:11%"><col style="width:11%"><col style="width:19%"><col style="width:10%"><col style="width:7%"></colgroup><thead><tr><th>Hypothesis</th><th>Comparison</th><th>Status</th><th style="text-align:right">Δ AP</th><th style="text-align:right">95% bootstrap CI</th><th style="text-align:right">Holm-adjusted p</th><th style="text-align:right">Iterations</th></tr></thead><tbody>__COMPARISON_ROWS__</tbody></table></div>
      </details>
      <p class="fine"><strong>Statistical note.</strong> Holm-adjusted p-values are the registered one-sided tests. The saved 95% confidence intervals are percentile-bootstrap summaries based on whole-group resampling.</p>
      <div class="provenance" style="margin-top:64px">
        <div><h3>Sources and limits</h3><ul><li><a href="https://arxiv.org/abs/2003.10778">PanNuke extension</a> — nucleus data and classes.</li><li><a href="https://arxiv.org/abs/1911.00068">Confident Learning</a> — label-quality methodology.</li><li><a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/fc20ea8d104cab737a5561096f9bde9b-Abstract.html">AQuA</a> — annotation-quality assessment.</li><li><a href="https://arxiv.org/abs/2102.09099">NuCLS</a> — candidate future validation setting.</li></ul><p>No patient- or WSI-level independence is claimed because the released metadata do not support it. Separation is guaranteed at source-patch level.</p></div>
        <div><h3>Reproducibility</h3><dl class="hash-list"><div><dt>Accepted run</dt><dd>__RUN_ID__</dd></div><div><dt>Artifact root SHA-256</dt><dd>__ARTIFACT_ROOT__</dd></div><div><dt>Stage attestation SHA-256</dt><dd>__STAGE_HASH__</dd></div><div><dt>QC overlay SHA-256</dt><dd>__QC_HASH__</dd></div></dl></div>
      </div>
    </div>
  </section>
</main>

<footer><div class="footer-inner"><span>© 2026 Natan Smogór · AANCA research prototype</span><span>Non-diagnostic · no automatic annotation changes · non-commercial research use</span></div></footer>
<script src="https://cdn.jsdelivr.net/npm/gsap@3.15.0/dist/gsap.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/gsap@3.15.0/dist/ScrollTrigger.min.js"></script>
<script>__SCRIPT__</script>
<script type="module">__THREE_SCRIPT__</script>
</body>
</html>
"""

    replacements = {
        "__CSS__": _MVP_CSS_V4,
        "__SCRIPT__": _MVP_SCRIPT_V4,
        "__THREE_SCRIPT__": _MVP_THREE_SCRIPT_V3,
        "__H4_POINT__": html.escape(_format_metric_v3(point_difference)),
        "__H4_POINT_PP__": html.escape(f"{abs(point_difference) * 100:.3f}"),
        "__H4_AUDIT__": html.escape(_format_metric_v3(h4["audit_guided_macro_f1"])),
        "__H4_RANDOM__": html.escape(_format_metric_v3(h4["random_review_macro_f1_mean"])),
        "__H4_CHART__": h4_chart,
        "__HYPOTHESIS_LEDGER__": hypothesis_ledger,
        "__FOREST_PLOT__": forest_plot,
        "__H2_CHART__": h2_chart,
        "__SEED_HASH_SHORT__": seed_hash[:16],
        "__OOF_HASH_SHORT__": oof_hash[:16],
        "__SEED_HASH__": seed_hash,
        "__OOF_HASH__": oof_hash,
        "__SEED_ROWS__": "".join(seed_rows),
        "__COMPLETED__": str(primary["completed_required_cells"]),
        "__FAILED__": str(primary["failed_required_cells"]),
        "__COMPARISONS__": str(primary["comparison_count"]),
        "__REPORTED__": str(reported_comparisons),
        "__UNAVAILABLE__": str(unavailable_comparisons),
        "__BOOTSTRAPS__": f"{primary['bootstrap_iterations_required']:,}",
        "__PATCHES__": f"{qc['patch_count']:,}",
        "__OVERLAP_PIXELS__": f"{qc['cross_class_overlap_pixel_count']:,}",
        "__VOID_PIXELS__": f"{qc['void_pixel_count']:,}",
        "__FLAGGED_INSTANCES__": f"{qc['overlap_touching_instance_count']:,}",
        "__COMPARISON_ROWS__": comparison_rows,
        "__RUN_ID__": html.escape(str(primary["run_id"])),
        "__ARTIFACT_ROOT__": html.escape(str(primary["artifact_root_sha256"])),
        "__STAGE_HASH__": html.escape(str(primary["stage_attestation_record_sha256"])),
        "__QC_HASH__": html.escape(str(qc["overlay_sha256"])),
    }
    for marker, replacement in replacements.items():
        if marker not in template:
            raise AssertionError(f"MVP template marker is absent: {marker}")
        template = template.replace(marker, replacement)
    return template


def _render_html(evidence: dict[str, Any]) -> str:
    """Render the current professor-facing presentation release."""

    return _render_html_v4(evidence)


def _render_readme(evidence: dict[str, Any]) -> str:
    primary = evidence["primary"]
    return f"""# AANCA presentation MVP

The package was generated from selected, checksum-verified sources in the accepted
run `{primary["run_id"]}`. From the repository root, the recommended presentation
command is:

```powershell
python scripts/present_demo.py
```

This standard-library launcher requires no project dependency installation. It
verifies the closed package before serving it on `127.0.0.1` and opens the article
in the default browser. No model run, dataset, or GPU is required. Use `--no-open`
in headless environments and `--port 0` to select a free port.

For verification without a browser or server, run:

```powershell
python scripts/present_demo.py --verify-only
```

After installing the full research environment, the equivalent commands are
`uv run histo-audit demo serve` and `uv run histo-audit demo verify`.

Author: Natan Smogór. Released: 18 August 2026.

The responsive presentation is written in English and uses pinned GSAP and
Three.js browser modules for progressive animation. Its evidence, navigation,
tables and scientific interpretation remain available when motion is reduced;
network access is only needed for the optional web fonts and animation libraries.
The WebGL loop pauses while the hero or browser tab is not visible, uses a capped
pixel ratio, and respects reduced-motion and data-saving preferences.

Scientific status: `PRIMARY_STUDY_COMPLETE`. Presentation status:
`DEMO_COMPLETE`. The primary analysis is permanently labelled
`amended_or_exploratory`; confirmatory and external validation were not run.

This is a non-diagnostic research prototype. It identifies a potentially
inconsistent annotation and recommends it for expert review; it never modifies
source annotations or claims that a pathologist was wrong.

`evidence.json` contains the sourced H1-H7 summary, the adverse H4 result,
the complete H2 subgroup summary, the byte-identical instance-dependent seed
disclosure, and all 36 saved H1/H3/H5/H6/H7 comparisons. P-values shown in the
HTML are explicitly labelled one-sided and Holm-adjusted. `manifest.json` binds
every other file in this package.

Of the 36 preregistered comparison entries, 33 contain numeric results and the
three H6 entries remain explicitly unavailable under the frozen encoder gate.
Source code, setup guidance, specifications, tests, and the complete documentation
map are available at <https://github.com/Jaqwilk/AANCA>.
"""


def _build_output_manifest(directory: Path) -> dict[str, Any]:
    records = [
        {
            "path": relative,
            "sha256": sha256_file(directory / relative),
            "size_bytes": (directory / relative).stat().st_size,
        }
        for relative in sorted(_OUTPUT_FILES)
    ]
    return {
        "schema_version": 2,
        "policy": "aanca_presentation_complete_h1_h7_selected_source_readback_v2",
        "files": records,
        "manifest_root_sha256": _canonical_sha256(records),
    }


def verify_mvp_presentation(output_directory: str | Path) -> dict[str, Any]:
    """Verify the small generated package without reading the scientific run again."""

    directory = Path(output_directory).resolve(strict=True)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("MVP output is not a regular directory")
    actual_names = sorted(path.name for path in directory.iterdir())
    expected_names = sorted((*_OUTPUT_FILES, "manifest.json"))
    if actual_names != expected_names:
        raise ValueError("MVP output allowlist differs")
    manifest = _load_json(directory / "manifest.json")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("policy") != "aanca_presentation_complete_h1_h7_selected_source_readback_v2"
    ):
        raise ValueError("MVP manifest schema or policy differs")
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != len(_OUTPUT_FILES):
        raise ValueError("MVP manifest file list differs")
    record_paths = [record.get("path") if isinstance(record, dict) else None for record in records]
    if sorted(path for path in record_paths if isinstance(path, str)) != sorted(_OUTPUT_FILES):
        raise ValueError("MVP manifest record allowlist differs")
    if manifest.get("manifest_root_sha256") != _canonical_sha256(records):
        raise ValueError("MVP manifest root differs")
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
            raise ValueError("MVP manifest record is malformed")
        _verify_manifest_file(directory, str(record["path"]), record)
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
        raise ValueError("MVP evidence scope differs")
    return {
        "status": "valid",
        "presentation_status": "DEMO_COMPLETE",
        "scientific_status": "PRIMARY_STUDY_COMPLETE",
        "manifest_root_sha256": manifest["manifest_root_sha256"],
        "file_count": len(expected_names),
    }


def build_mvp_presentation(
    *,
    project_root: str | Path,
    run_directory: str | Path,
    qc_bundle_directory: str | Path,
    output_directory: str | Path,
) -> MvpPresentationArtifacts:
    """Create one immutable-size, static package without running a model or changing data."""

    root = Path(project_root).resolve(strict=True)
    run_path = Path(run_directory)
    if not run_path.is_absolute():
        run_path = root / run_path
    run_path = run_path.resolve(strict=True)
    qc_path = Path(qc_bundle_directory)
    if not qc_path.is_absolute():
        qc_path = root / qc_path
    qc_path = qc_path.resolve(strict=True)
    output = Path(output_directory)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"MVP output already exists; refusing overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    primary = _validate_primary_sources(run_path)
    qc, overlay_source, qc_manifest = _validate_qc_sources(qc_path)
    evidence = {
        "schema_version": 2,
        "presentation_status": "DEMO_COMPLETE",
        "scientific_status": "PRIMARY_STUDY_COMPLETE",
        "analysis_disposition": "amended_or_exploratory",
        "confirmatory_completed": False,
        "external_validation_completed": False,
        "automatic_annotation_changes": False,
        "diagnostic_system": False,
        "primary": primary,
        "pannuke_qc": qc,
        "source_files": {
            "primary_run": str(run_path),
            "qc_bundle": str(qc_path),
            "qc_bundle_manifest_sha256": sha256_file(qc_path / "artifact_manifest.json"),
            "qc_overlay_sha256": qc_manifest["overlay_sha256"],
        },
    }

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        atomic_write_json(staging / "evidence.json", evidence)
        atomic_write_text(staging / "index.html", _render_html(evidence))
        atomic_write_text(staging / "README.md", _render_readme(evidence))
        atomic_write_bytes(staging / "pannuke_mask_qc_overlays.png", overlay_source.read_bytes())
        manifest = _build_output_manifest(staging)
        atomic_write_json(staging / "manifest.json", manifest)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    verification = verify_mvp_presentation(output)
    return MvpPresentationArtifacts(
        output_directory=output,
        html_path=output / "index.html",
        evidence_path=output / "evidence.json",
        overlay_path=output / "pannuke_mask_qc_overlays.png",
        readme_path=output / "README.md",
        manifest_path=output / "manifest.json",
        manifest_root_sha256=str(verification["manifest_root_sha256"]),
    )


__all__ = [
    "CANONICAL_PRIMARY_RUN_ID",
    "DEFAULT_OUTPUT",
    "DEFAULT_PRIMARY_RUN",
    "DEFAULT_QC_BUNDLE",
    "MvpPresentationArtifacts",
    "build_mvp_presentation",
    "create_mvp_http_server",
    "verify_mvp_presentation",
]
