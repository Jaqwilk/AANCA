"""Build a small, evidence-backed presentation package from accepted AANCA artifacts."""

from __future__ import annotations

import base64
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
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

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

_RELEASE_EVIDENCE_PATHS = {
    "nucls_external": Path("artifacts/nucls_external_validation/unbiased-v1/results.json"),
    "monusac_external": Path("artifacts/monusac_external_validation/results.json"),
    "puma_confirmation": Path("artifacts/puma_new_data_confirmation/results.json"),
    "puma_verification": Path("artifacts/puma_new_data_confirmation/verification.json"),
    "puma_realism_stress": Path("artifacts/puma_realism_stress/results.json"),
    "puma_label_sensitivity": Path("artifacts/puma_audit_time_label_sensitivity/results.json"),
    "nucls_qc_feasibility": Path("artifacts/nucls_supervised_qc_feasibility/results.json"),
}

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
_BENCHMARK_ICON_DIR = Path(__file__).resolve().parent / "assets" / "benchmark"
_BENCHMARK_FACTS: tuple[tuple[str, str, str, str], ...] = (
    (
        "Data",
        "pannuke.png",
        "PanNuke",
        "Verified official release; five positive nucleus classes across 19 tissue types.",
    ),
    (
        "Unit",
        "segmented-nuclei.png",
        "Already segmented nuclei",
        "Class-label consistency only. Segmentation quality and diagnosis are outside scope.",
    ),
    (
        "Primary model",
        "resnet-model.png",
        "Frozen ResNet-18 + logistic regression",
        "ImageNet context embeddings with balanced multinomial fitting.",
    ),
    (
        "Prediction design",
        "five-fold-oof.png",
        "Five-fold group-safe OOF",
        "A scored nucleus and its whole source patch are absent from its training fold.",
    ),
    (
        "Review budget",
        "review-budget.png",
        "5% primary queue",
        "Guided and random review receive the same integer budget.",
    ),
)
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


def _require_mapping(value: Any, *, role: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _project_relative_path(root: Path, path: Path, *, role: str) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"{role} must remain inside the project root") from error


def _load_release_evidence(root: Path) -> dict[str, Any]:
    """Read and strictly summarise the checked-in external evidence authorities."""

    paths = {
        key: (root / relative).resolve(strict=True)
        for key, relative in _RELEASE_EVIDENCE_PATHS.items()
    }
    sources = {
        key: {
            "path": _project_relative_path(root, path, role=key),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for key, path in paths.items()
    }

    nucls = _load_json(paths["nucls_external"])
    nucls_ranking = _require_mapping(nucls.get("ranking"), role="NuCLS ranking")
    nucls_downstream = _require_mapping(nucls.get("downstream"), role="NuCLS downstream")
    nucls_claims = _require_mapping(nucls.get("claim_boundary"), role="NuCLS claim boundary")
    if (
        nucls.get("study_id") != "nucls_natural_label_external_validation_v1"
        or nucls.get("status") != "completed"
        or nucls.get("subset") != "unbiased_control"
        or nucls_ranking.get("success_conditions_met") is not False
        or nucls_downstream.get("success_conditions_met") is not False
        or any(
            nucls_claims.get(key) is not False
            for key in (
                "automatic_source_changes_permitted",
                "biological_truth_proven",
                "clinical_utility_proven",
                "pathologist_error_proven",
            )
        )
    ):
        raise ValueError("NuCLS external evidence scope differs")
    nucls_guided = _require_mapping(
        nucls_downstream.get("guided_minus_uncorrected_macro_f1"),
        role="NuCLS guided-minus-uncorrected result",
    )
    nucls_external = {
        "study_id": nucls["study_id"],
        "completion_stage": "EXTERNAL_VALIDATION_COMPLETE",
        "overall_conclusion": "not_supported",
        "reference_interpretation": str(nucls["reference_interpretation"]),
        "primary_subset": {
            "subset": nucls["subset"],
            "sample_count": _require_exact_int(nucls["sample_count"], role="NuCLS samples"),
            "patient_group_count": _require_exact_int(
                nucls["patient_group_count"], role="NuCLS patient groups"
            ),
            "natural_disagreement_count": _require_exact_int(
                nucls_ranking["natural_disagreement_count"],
                role="NuCLS disagreements",
            ),
            "natural_disagreement_prevalence": _require_real(
                nucls_ranking["natural_disagreement_prevalence"],
                role="NuCLS disagreement prevalence",
            ),
            "average_precision": _require_real(
                nucls_ranking["average_precision"], role="NuCLS average precision"
            ),
            "ap_minus_prevalence_interval_95": _require_interval(
                nucls_ranking["ap_minus_prevalence"]["interval_95"],
                role="NuCLS AP interval",
            ),
            "precision_at_5_percent": _require_real(
                nucls_ranking["budgets"]["0.05"]["precision"],
                role="NuCLS precision at five percent",
            ),
            "precision_at_5_percent_minus_prevalence_interval_95": _require_interval(
                nucls_ranking["precision_at_5_percent_minus_prevalence"]["interval_95"],
                role="NuCLS precision interval",
            ),
            "ranking_success_conditions_met": False,
            "guided_minus_uncorrected_macro_f1": _require_real(
                nucls_guided["estimate"], role="NuCLS macro-F1 difference"
            ),
            "guided_minus_uncorrected_macro_f1_interval_95": _require_interval(
                nucls_guided["interval_95"], role="NuCLS macro-F1 interval"
            ),
            "downstream_success_conditions_met": False,
        },
        "claim_boundary": {
            "biological_truth_proven": False,
            "clinical_utility_proven": False,
            "pathologist_error_proven": False,
        },
        "source": sources["nucls_external"],
    }

    monusac = _load_json(paths["monusac_external"])
    monusac_dataset = _require_mapping(monusac.get("dataset"), role="MoNuSAC dataset")
    monusac_corruption = _require_mapping(
        monusac.get("controlled_corruption"), role="MoNuSAC corruption"
    )
    monusac_retrieval = _require_mapping(
        monusac.get("primary_matched_random_retrieval"), role="MoNuSAC retrieval"
    )
    monusac_downstream = _require_mapping(monusac.get("downstream"), role="MoNuSAC downstream")
    monusac_metrics = _require_mapping(
        monusac_downstream.get("metrics"), role="MoNuSAC downstream metrics"
    )
    monusac_guards = _require_mapping(
        monusac_downstream.get("adoption_guards"), role="MoNuSAC adoption guards"
    )
    monusac_primary_guard = _require_mapping(
        monusac_guards.get("nearest_neighbour_disagreement_balanced_review"),
        role="MoNuSAC primary adoption guard",
    )
    monusac_macro_guard = _require_mapping(
        monusac_primary_guard.get("macro_f1"), role="MoNuSAC macro-F1 guard"
    )
    monusac_random = _require_mapping(
        monusac_downstream.get("primary_minus_mean_matched_random"),
        role="MoNuSAC matched-random downstream",
    )
    monusac_conditions = _require_mapping(
        monusac.get("success_conditions"), role="MoNuSAC success conditions"
    )
    monusac_claims = _require_mapping(monusac.get("claim_boundary"), role="MoNuSAC claim boundary")
    if (
        monusac.get("study_id") != "monusac_current_aanca_controlled_external_v1"
        or monusac.get("analysis_disposition")
        != "prospectively_frozen_controlled_external_benchmark"
        or monusac.get("all_success_conditions_met") is not False
        or monusac_conditions.get("primary_top_k_beats_exact_matched_random_control") is not True
        or any(
            monusac_conditions.get(key) is not False
            for key in (
                "primary_intervention_macro_f1_ci95_lower_gt_corrupted_uncorrected",
                "primary_intervention_macro_f1_ci95_lower_gt_mean_matched_random",
                "no_important_class_recall_ci95_lower_below_minus_0_01",
            )
        )
        or monusac_primary_guard.get("action") != "retain_uncorrected"
        or monusac_corruption.get("source_annotations_modified") is not False
        or any(
            monusac_claims.get(key) is not False
            for key in (
                "automatic_annotation_change_permitted",
                "clinical_or_operational_utility_claim_permitted",
                "natural_pathology_error_detection_claim_permitted",
                "pathologist_error_claim_permitted",
            )
        )
    ):
        raise ValueError("MoNuSAC external evidence scope differs")
    monusac_primary_metric = _require_mapping(
        monusac_metrics.get("nearest_neighbour_disagreement_balanced_review"),
        role="MoNuSAC candidate metric",
    )
    monusac_baseline_metric = _require_mapping(
        monusac_metrics.get("corrupted_uncorrected"), role="MoNuSAC baseline metric"
    )
    controlled_external = {
        "study_id": monusac["study_id"],
        "dataset": monusac_dataset["name"],
        "analysis_disposition": monusac["analysis_disposition"],
        "decision": "not_supported",
        "all_success_conditions_met": False,
        "development": {
            "patient_group_count": _require_exact_int(
                monusac_dataset["train_patient_groups"], role="MoNuSAC development groups"
            ),
            "eligible_nuclei": _require_exact_int(
                monusac_dataset["train_eligible_nuclei"], role="MoNuSAC development nuclei"
            ),
            "controlled_corruption_count": _require_exact_int(
                monusac_corruption["exact_count"], role="MoNuSAC corruptions"
            ),
            "controlled_corruption_rate": _require_real(
                monusac_corruption["rate"], role="MoNuSAC corruption rate"
            ),
        },
        "test": {
            "patient_group_count": _require_exact_int(
                monusac_dataset["test_patient_groups"], role="MoNuSAC test groups"
            ),
            "eligible_nuclei": _require_exact_int(
                monusac_dataset["test_eligible_nuclei"], role="MoNuSAC test nuclei"
            ),
        },
        "primary_ranking": {
            "reviewed_count": _require_exact_int(
                monusac_retrieval["top_reviewed"], role="MoNuSAC reviewed count"
            ),
            "changes_found": _require_exact_int(
                monusac_retrieval["top_found"], role="MoNuSAC changes found"
            ),
            "precision": _require_real(
                monusac_retrieval["top_precision"], role="MoNuSAC precision"
            ),
            "mean_matched_random_precision": _require_real(
                monusac_retrieval["mean_matched_random_precision"],
                role="MoNuSAC random precision",
            ),
            "difference_interval_95": _require_interval(
                monusac_retrieval["interval_95"], role="MoNuSAC retrieval interval"
            ),
        },
        "primary_downstream": {
            "candidate_macro_f1": _require_real(
                monusac_primary_metric["macro_f1"], role="MoNuSAC candidate macro-F1"
            ),
            "corrupted_uncorrected_macro_f1": _require_real(
                monusac_baseline_metric["macro_f1"], role="MoNuSAC baseline macro-F1"
            ),
            "minus_corrupted_uncorrected_macro_f1": _require_real(
                monusac_macro_guard["candidate_minus_uncorrected_macro_f1"],
                role="MoNuSAC baseline difference",
            ),
            "minus_corrupted_uncorrected_interval_95": _require_interval(
                monusac_macro_guard["interval_95"], role="MoNuSAC baseline interval"
            ),
            "minus_mean_matched_random_macro_f1": _require_real(
                monusac_random["candidate_minus_mean_matched_random_macro_f1"],
                role="MoNuSAC random difference",
            ),
            "minus_mean_matched_random_interval_95": _require_interval(
                monusac_random["interval_95"], role="MoNuSAC random interval"
            ),
        },
        "success_conditions": {
            "primary_top_k_beats_exact_matched_random": True,
            "primary_downstream_beats_corrupted_uncorrected": False,
            "primary_downstream_beats_mean_matched_random": False,
            "important_class_recall_non_degradation": False,
        },
        "claim_boundary": {
            "clinical_or_operational_utility_proven": False,
            "natural_pathology_error_detection_proven": False,
            "pathologist_error_proven": False,
            "source_annotations_modified": False,
        },
        "source": sources["monusac_external"],
    }

    puma = _load_json(paths["puma_confirmation"])
    puma_verification = _load_json(paths["puma_verification"])
    puma_dataset = _require_mapping(puma.get("dataset"), role="PUMA dataset")
    puma_retrieval = _require_mapping(puma.get("retrieval"), role="PUMA retrieval")
    puma_downstream = _require_mapping(puma.get("downstream"), role="PUMA downstream")
    puma_conditions = _require_mapping(
        puma.get("success_conditions"), role="PUMA success conditions"
    )
    puma_claims = _require_mapping(puma.get("claim_boundary"), role="PUMA claim boundary")
    if (
        puma.get("study_id") != "puma_new_data_confirmation_v1"
        or puma.get("analysis_disposition")
        != "prospectively_frozen_new_source_controlled_confirmation"
        or puma.get("all_success_conditions_met") is not True
        or puma.get("all_models_converged") is not True
        or puma.get("natural_error_detection_evaluated") is not False
        or puma.get("pathologist_error_detection_proven") is not False
        or puma.get("replacement_project_or_v2") is not False
        or puma_dataset.get("source_annotations_modified") is not False
        or puma_claims.get("controlled_noise_transfer_if_positive") is not True
        or any(
            puma_claims.get(key) is not False
            for key in (
                "automatic_annotation_change_permitted",
                "clinical_utility_proven",
                "natural_error_detection_proven",
                "pathologist_error_detection_proven",
            )
        )
        or not all(puma_conditions.values())
        or puma_verification.get("verified") is not True
        or puma_verification.get("all_seven_frozen_success_gates_passed") is not True
        or puma_verification.get("all_44_models_converged") is not True
        or puma_verification.get("source_reference_labels_unchanged") is not True
    ):
        raise ValueError("PUMA confirmation evidence scope differs")
    puma_confirmation = {
        "study_id": puma["study_id"],
        "dataset": puma_dataset["name"],
        "analysis_disposition": puma["analysis_disposition"],
        "decision": "controlled_noise_transfer_supported",
        "all_success_conditions_met": True,
        "final_case_groups": _require_exact_int(
            puma_dataset["final_case_groups"], role="PUMA final groups"
        ),
        "final_nuclei": _require_exact_int(puma_dataset["final_nuclei"], role="PUMA final nuclei"),
        "retrieval": {
            "candidate_precision": _require_real(
                puma_retrieval["candidate_precision"], role="PUMA precision"
            ),
            "mean_matched_random_precision": _require_real(
                puma_retrieval["mean_matched_random_precision"],
                role="PUMA random precision",
            ),
            "difference": _require_real(
                puma_retrieval["candidate_minus_matched_random_precision"],
                role="PUMA retrieval difference",
            ),
            "interval_95": _require_interval(
                puma_retrieval["interval_95"], role="PUMA retrieval interval"
            ),
        },
        "downstream": {
            "candidate_macro_f1": _require_real(
                puma_downstream["candidate_macro_f1"], role="PUMA candidate macro-F1"
            ),
            "uncorrected_macro_f1": _require_real(
                puma_downstream["uncorrected_macro_f1"], role="PUMA baseline macro-F1"
            ),
            "mean_matched_random_macro_f1": _require_real(
                puma_downstream["mean_matched_random_macro_f1"],
                role="PUMA random macro-F1",
            ),
            "minus_uncorrected": _require_real(
                puma_downstream["candidate_minus_uncorrected_macro_f1"],
                role="PUMA baseline difference",
            ),
            "minus_uncorrected_interval_95": _require_interval(
                puma_downstream["candidate_minus_uncorrected_interval_95"],
                role="PUMA baseline interval",
            ),
            "minus_matched_random": _require_real(
                puma_downstream["candidate_minus_matched_random_macro_f1"],
                role="PUMA random difference",
            ),
            "minus_matched_random_interval_95": _require_interval(
                puma_downstream["candidate_minus_matched_random_interval_95"],
                role="PUMA random interval",
            ),
        },
        "success_conditions": dict(puma_conditions),
        "claim_boundary": dict(puma_claims),
        "verification": {
            "verified": True,
            "all_seven_frozen_success_gates_passed": True,
            "all_44_models_converged": True,
            "source_reference_labels_unchanged": True,
        },
        "sources": {
            "results": sources["puma_confirmation"],
            "verification": sources["puma_verification"],
        },
    }

    stress = _load_json(paths["puma_realism_stress"])
    stress_scenarios = stress.get("scenarios")
    if (
        stress.get("study_id") != "puma_realism_stress_v1"
        or stress.get("disposition") != "post_confirmation_exploratory_stress_only"
        or stress.get("all_scenarios_passed") is not False
        or stress.get("candidate_changed") is not False
        or stress.get("source_annotations_modified") is not False
        or stress.get("natural_error_detection_evaluated") is not False
        or stress.get("pathologist_error_detection_proven") is not False
        or not isinstance(stress_scenarios, list)
        or len(stress_scenarios)
        != _require_exact_int(stress.get("scenario_count"), role="PUMA stress scenario count")
    ):
        raise ValueError("PUMA stress evidence scope differs")
    all_class_safe_count = sum(
        _require_mapping(item, role="PUMA stress scenario").get("all_scenario_gates_passed") is True
        for item in stress_scenarios
    )
    positive_aggregate_count = sum(
        _require_interval(
            _require_mapping(item.get("downstream"), role="PUMA stress downstream").get(
                "candidate_minus_uncorrected_interval_95"
            ),
            role="PUMA stress downstream interval",
        )[0]
        > 0
        for item in stress_scenarios
    )
    realism_stress = {
        "study_id": stress["study_id"],
        "disposition": stress["disposition"],
        "scenario_count": len(stress_scenarios),
        "positive_aggregate_lower_bound_count": positive_aggregate_count,
        "all_class_safeguards_passed_count": all_class_safe_count,
        "all_scenarios_passed": False,
        "candidate_changed": False,
        "claim_boundary": dict(
            _require_mapping(stress.get("claim_boundary"), role="PUMA stress claim boundary")
        ),
        "source": sources["puma_realism_stress"],
    }

    sensitivity = _load_json(paths["puma_label_sensitivity"])
    sensitivity_claims = _require_mapping(
        sensitivity.get("claim_boundary"), role="PUMA sensitivity claim boundary"
    )
    if (
        sensitivity.get("study_id") != "puma_audit_time_label_sensitivity_v1"
        or sensitivity.get("disposition") != "post_confirmation_exploratory_sensitivity_only"
        or sensitivity.get("all_sensitivity_gates_passed") is not True
        or sensitivity.get("candidate_changed") is not False
        or sensitivity.get("fold_assignment_label_source") != "observed_label"
        or sensitivity.get("pre_corruption_label_used_for_fold_assignment") is not False
        or sensitivity.get("source_annotations_modified") is not False
        or sensitivity.get("natural_error_detection_evaluated") is not False
        or sensitivity.get("pathologist_error_detection_proven") is not False
        or not all(
            _require_mapping(
                sensitivity.get("success_conditions"),
                role="PUMA sensitivity success conditions",
            ).values()
        )
        or any(value is not False for value in sensitivity_claims.values())
    ):
        raise ValueError("PUMA label-sensitivity evidence scope differs")
    puma_sensitivity = {
        "study_id": sensitivity["study_id"],
        "disposition": sensitivity["disposition"],
        "all_sensitivity_gates_passed": True,
        "fold_assignment_label_source": "observed_label",
        "candidate_changed": False,
        "claim_boundary": dict(sensitivity_claims),
        "source": sources["puma_label_sensitivity"],
    }

    feasibility = _load_json(paths["nucls_qc_feasibility"])
    if (
        feasibility.get("study_id") != "nucls_supervised_qc_prospective_v1"
        or feasibility.get("prospective_evaluation_status") != "unavailable"
        or feasibility.get("failure_action") != "retain_uncorrected"
        or feasibility.get("paired_nucleus_pre_post_label_available") is not False
        or feasibility.get("natural_error_detection_evaluated") is not False
        or feasibility.get("pathologist_error_detection_proven") is not False
        or feasibility.get("source_annotations_modified") is not False
    ):
        raise ValueError("NuCLS supervised-QC feasibility evidence scope differs")
    natural_data_action = {
        "study_id": feasibility["study_id"],
        "status": "unavailable",
        "action": "retain_uncorrected",
        "reason": str(feasibility["unavailable_reason"]),
        "source": sources["nucls_qc_feasibility"],
    }

    return {
        "external_validation": nucls_external,
        "controlled_external_benchmark": controlled_external,
        "new_source_confirmation": puma_confirmation,
        "realism_stress": realism_stress,
        "audit_time_label_sensitivity": puma_sensitivity,
        "natural_data_action": natural_data_action,
        "source_records": sources,
    }


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


def _render_current_evidence(evidence: dict[str, Any]) -> str:
    """Render the current external evidence as a compact article ledger."""

    nucls = evidence["external_validation"]["primary_subset"]
    monusac = evidence["controlled_external_benchmark"]
    puma = evidence["new_source_confirmation"]
    stress = evidence["realism_stress"]
    sensitivity = evidence["audit_time_label_sensitivity"]
    natural_action = evidence["natural_data_action"]

    nucls_ci = nucls["guided_minus_uncorrected_macro_f1_interval_95"]
    monusac_retrieval_ci = monusac["primary_ranking"]["difference_interval_95"]
    monusac_downstream_ci = monusac["primary_downstream"]["minus_corrupted_uncorrected_interval_95"]
    puma_retrieval_ci = puma["retrieval"]["interval_95"]
    puma_baseline_ci = puma["downstream"]["minus_uncorrected_interval_95"]
    puma_random_ci = puma["downstream"]["minus_matched_random_interval_95"]

    return f"""
    <div class="figure-width reveal">
      <div class="rules evidence-ledger">
        <article class="rule">
          <b>NuCLS natural multi-rater disagreement</b>
          <p>On {nucls["patient_group_count"]} patient groups and {nucls["sample_count"]:,}
          nuclei, average precision was {nucls["average_precision"]:.6f}. Precision at
          the 5% review budget was {nucls["precision_at_5_percent"]:.6f}, but its
          difference-from-prevalence interval crossed zero. Guided correction reduced
          macro-F1 by {abs(nucls["guided_minus_uncorrected_macro_f1"]):.6f}, with a 95%
          interval [{nucls_ci[0]:.6f}, {nucls_ci[1]:.6f}]. The frozen natural-data
          claims were not supported.</p>
        </article>
        <article class="rule">
          <b>MoNuSAC controlled external benchmark</b>
          <p>The frozen queue found {monusac["primary_ranking"]["changes_found"]:,} of
          {monusac["development"]["controlled_corruption_count"]:,} injected changes:
          precision {monusac["primary_ranking"]["precision"]:.6f} versus
          {monusac["primary_ranking"]["mean_matched_random_precision"]:.6f} for matched
          random review, with a positive difference interval
          [{monusac_retrieval_ci[0]:.6f}, {monusac_retrieval_ci[1]:.6f}]. Downstream
          macro-F1 changed by {monusac["primary_downstream"]["minus_corrupted_uncorrected_macro_f1"]:+.6f}
          [{monusac_downstream_ci[0]:+.6f}, {monusac_downstream_ci[1]:+.6f}], so the
          benefit and class-safety gates failed. The prescribed action remained
          <code>retain_uncorrected</code>.</p>
        </article>
        <article class="rule">
          <b>PUMA internally frozen new-source controlled confirmation</b>
          <p>The candidate was recorded internally as frozen before PUMA outcomes;
          public Git history does not independently verify that timing. It was evaluated on
          {puma["final_case_groups"]} held-out case/ROI groups from an AANCA-defined
          144/62 split of the 206 public ROIs, not the official hidden PUMA challenge
          test set. Queue precision was
          {puma["retrieval"]["candidate_precision"]:.6f} versus
          {puma["retrieval"]["mean_matched_random_precision"]:.6f}; the difference was
          {puma["retrieval"]["difference"]:+.6f}
          [{puma_retrieval_ci[0]:+.6f}, {puma_retrieval_ci[1]:+.6f}]. Candidate macro-F1
          was {puma["downstream"]["candidate_macro_f1"]:.6f}, improving on unchanged
          labels by {puma["downstream"]["minus_uncorrected"]:+.6f}
          [{puma_baseline_ci[0]:+.6f}, {puma_baseline_ci[1]:+.6f}] and matched random by
          {puma["downstream"]["minus_matched_random"]:+.6f}
          [{puma_random_ci[0]:+.6f}, {puma_random_ci[1]:+.6f}]. All seven internally
          pre-specified gates passed. The <code>flag_exclude</code> arm omitted the
          highest-ranked 5% of controlled training rows; they were not reviewed,
          corrected or automatically relabelled by an expert. This supports
          controlled-noise transfer only.</p>
        </article>
        <article class="rule">
          <b>Robustness and the remaining safety boundary</b>
          <p>All {stress["positive_aggregate_lower_bound_count"]} of
          {stress["scenario_count"]} post-confirmation PUMA stress scenarios had a
          positive aggregate downstream lower bound, but only
          {stress["all_class_safeguards_passed_count"]} passed every class safeguard.
          The <code>{html.escape(str(sensitivity["study_id"]))}</code> observed-label
          fold-allocation sensitivity passed all seven gates, but it was run after
          PUMA outcomes were open. Natural paired pre/post NuCLS
          evidence was unavailable, so the binding action is still
          <code>{html.escape(str(natural_action["action"]))}</code>.</p>
        </article>
      </div>
    </div>
    """.lstrip("\n").rstrip()


def _benchmark_icon_data_uri(filename: str) -> str:
    """Return a data URI for one sealed benchmark fact icon."""

    path = _BENCHMARK_ICON_DIR / filename
    if path.parent.resolve() != _BENCHMARK_ICON_DIR.resolve() or not path.is_file():
        raise ValueError(f"benchmark icon is missing: {filename}")
    raw = path.read_bytes()
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif raw.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    else:
        raise ValueError(f"benchmark icon format is unsupported: {filename}")
    encoded = base64.standard_b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _render_study_specs() -> str:
    """Render the five exact-benchmark facts with decorative icons."""

    cards: list[str] = []
    for label, icon_name, title, description in _BENCHMARK_FACTS:
        src = html.escape(_benchmark_icon_data_uri(icon_name), quote=True)
        cards.append(
            '<article class="spec-card">'
            f"<span>{html.escape(label)}</span>"
            '<div class="spec-copy">'
            f'<img class="spec-icon" src="{src}" alt="" aria-hidden="true" '
            'width="60" height="60" decoding="async">'
            f"<strong>{html.escape(title)}</strong>"
            f"<small>{html.escape(description)}</small>"
            "</div>"
            "</article>"
        )
    return '<div class="study-specs reveal">\n      ' + "\n      ".join(cards) + "\n    </div>"


def _render_html(evidence: dict[str, Any], *, evidence_sha256: str) -> str:
    primary = evidence["primary"]
    qc = evidence["pannuke_qc"]
    hypotheses = primary["hypothesis_comparisons"]
    h2 = primary["h2_subgroups"]
    h4 = primary["h4_restoration"]
    seed_audit = primary["instance_dependent_seed_audit"]

    comparison_rows = _render_comparison_rows(primary["comparisons"])
    forest_plot = _render_forest_plot(primary["comparisons"])
    h4_chart = _render_h4_chart(h4)
    h2_chart = _render_h2_chart(h2)
    hypothesis_ledger = _render_hypothesis_ledger(hypotheses, h2, h4)
    current_evidence = _render_current_evidence(evidence)
    puma = evidence["new_source_confirmation"]
    publication_limits = evidence["publication_limits"]

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
    oof_hash = html.escape(str(seed_audit["records"][0]["oof_predictions_sha256"]))
    point_difference = _require_real(h4["point_difference"], role="H4 point difference")
    reported_comparisons = sum(record["status"] == "reported" for record in primary["comparisons"])
    unavailable_comparisons = primary["comparison_count"] - reported_comparisons

    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="AANCA is a reproducible, non-diagnostic framework for ranking potentially inconsistent nucleus annotations for expert review.">
  <meta name="author" content="Natan Smogór">
  <meta name="date" content="2026-08-21">
  <meta name="theme-color" content="#010102">
  <meta name="referrer" content="no-referrer">
  <link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&amp;family=JetBrains+Mono:wght@400;500&amp;display=swap" rel="stylesheet">
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%23010102'/%3E%3Crect x='14' y='14' width='12' height='12' rx='3' fill='%235e6ad2'/%3E%3Crect x='38' y='14' width='12' height='12' rx='3' fill='%23828fff'/%3E%3Crect x='26' y='26' width='12' height='12' rx='3' fill='%235e6ad2'/%3E%3Crect x='14' y='38' width='12' height='12' rx='3' fill='%23828fff'/%3E%3Crect x='38' y='38' width='12' height='12' rx='3' fill='%235e6ad2'/%3E%3C/svg%3E">
  <title>AANCA: Automated Auditing of Nucleus Class Annotations</title>
  <style>__CSS__</style>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="site-header" id="site-header">
  <div class="nav-shell">
    <a class="brand" href="#top" aria-label="AANCA: back to the beginning"><span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span><span class="brand-word-clip"><span class="brand-label">AANCA</span></span></a>
    <button class="menu-button" id="menu-button" type="button" aria-expanded="false" aria-controls="nav-links" aria-label="Open menu"><span class="menu-button-icon" aria-hidden="true"><span></span><span></span><span></span></span></button>
    <nav class="nav-links" id="nav-links" aria-label="Presentation sections"><a href="#overview">Study</a><a href="#method">Method</a><a href="#results">Findings</a><a href="#evidence">Evidence</a><a href="#current-stage">Status</a><a href="#use">Reproduce</a><a href="#author">Author</a></nav>
  </div>
</header>

<section class="hero" id="top" aria-labelledby="hero-title">
  <div class="hero-shell">
    <div class="hero-copy">
      <p class="eyebrow hero-animate">Automated Auditing of Nucleus Class Annotations</p>
      <h1 id="hero-title" class="hero-animate">Which annotations deserve a second look?</h1>
      <p class="hero-lead hero-animate">A reproducible, group-safe framework that ranks each <em>potentially inconsistent annotation</em> and recommends the highest-priority cases for expert review without rewriting the source labels.</p>
      <div class="hero-byline hero-animate"><span>Research prototype by <a class="author-jump" href="#author"><strong>Natan Smogór</strong></a></span><span>Updated <time datetime="2026-08-21">21 August 2026</time></span></div>
    </div>
  </div>
</section>

<main id="main">
  <section class="executive-summary" id="summary" aria-labelledby="summary-title">
    <div class="article-copy summary-box reveal">
      <p class="summary-label">90-second summary</p>
      <h2 id="summary-title">AANCA ranks existing nucleus annotations for human review and never automatically relabels them.</h2>
      <p class="summary-statement">On an AANCA-defined 62-ROI holdout from the 206 public PUMA ROIs, after 10% controlled corruption, its 5% review queue achieved precision <strong>__PUMA_PRECISION_4__</strong> versus <strong>__PUMA_RANDOM_PRECISION_4__</strong> for matched random selection. Excluding the flagged examples improved downstream macro-F1 by <strong>__PUMA_DELTA_F1_4__</strong> over unchanged corrupted training. This is controlled-noise evidence, not natural pathologist-error or clinical validation.</p>
      <p class="summary-detail"><strong>Exact design boundary.</strong> The 144/62 development/final partition is an AANCA-defined split of the 206 public PUMA ROIs. It is not the official hidden PUMA challenge test set. The downstream intervention was <code>flag_exclude</code>: the highest-ranked 5% of training instances were omitted from downstream training. They were not reviewed, corrected or automatically relabelled by an expert.</p>
    </div>
  </section>

  <section class="narrative" id="overview">
    <div class="article-copy reveal">
      <p class="lede">Annotation auditing is a way to decide what a human should inspect first. It is not a way to replace the human decision.</p>
      <p>Large histopathology datasets contain many already segmented nucleus instances, each paired with a class label. Even careful annotation can include ambiguity, inconsistent conventions or ordinary data-entry noise. Reviewing every instance again is expensive, so the useful question is not “can a model declare a label wrong?” It is “can a model create a better review queue than random sampling?”</p>
      <p>AANCA evaluates that question under controlled conditions. It intentionally changes a known subset of class labels, hides that intervention from the auditor, and then measures whether the changed labels move toward the front of a fixed review queue. Because the benchmark records exactly what it changed, retrieval can be scored without pretending that model disagreement is biological truth.</p>
      <p>The <strong>pre-corruption label is an experimental reference label, not guaranteed biological truth</strong>. It is used only to define the controlled benchmark, simulated restoration and final evaluation. In a real audit, a high score means <em>recommended for expert review</em>. It never means “confirmed error.”</p>
      <div class="scope-note"><strong>Study boundary.</strong> The PanNuke primary benchmark, a frozen NuCLS multi-rater evaluation, a frozen MoNuSAC controlled benchmark and a new-source PUMA controlled confirmation have been completed. PUMA supports transfer under controlled label noise; NuCLS does not support natural-error or downstream-improvement claims. Blinded natural-case expert review and prospective clinical workflow evaluation have not been performed. The PanNuke primary analysis remains exploratory because outcomes were exposed during technical recovery.</div>
    </div>
  </section>

  <section class="study-at-a-glance" id="study-facts" aria-labelledby="study-title">
    <div class="section-heading reveal"><h2 id="study-title">The exact benchmark, in five facts.</h2><p class="section-deck">These details define both what the results measure and what they do not measure.</p></div>
    __STUDY_SPECS__
  </section>

  <div class="research-question reveal" id="research-question">
    <p>Can source-group-safe out-of-fold models retrieve controlled label inconsistencies more efficiently than random review, and does restoring the highest-ranked injected corruptions improve downstream nucleus classification?</p>
  </div>

  <section class="journey story" id="method" aria-labelledby="journey-title" data-active-stage="0">
    <div class="article-copy method-context reveal">
      <p class="section-kicker">Method</p>
      <h2>A controlled audit begins by preserving what the model is allowed to know.</h2>
      <p>The benchmark does not ask a classifier to overwrite a pathologist's annotation. It creates a controlled setting in which a known subset of labels is changed intentionally, while the source record, reference label and corruption metadata remain separate and immutable.</p>
      <p>All development splits are made by <code>group_id</code>, at least at source-patch level. A nucleus can be scored only by a model that was trained without that nucleus and without every other nucleus from the same source patch. This produces one group-safe out-of-fold probability vector for every audited instance.</p>
      <p>Those probabilities are converted into review-priority scores. The primary self-confidence score asks how little probability the model assigns to the observed label; complementary methods add likelihood, margin, ambiguity and fold-safe neighbourhood evidence. Higher scores mean earlier review, never automatic correction.</p>
      <p>Finally, guided and random review receive the same integer budget. The benchmark measures whether injected changes appear earlier in the guided queue and then asks a separate question: whether restoring only the reviewed injected changes improves a downstream classifier on the untouched final reference fold.</p>
    </div>
    <figure class="method-queue-figure rail-figure reveal">
      <div class="method-queue-stage">
        <svg class="method-queue-diagram" viewBox="0 0 960 300" role="img" aria-labelledby="queue-figure-title queue-figure-desc" preserveAspectRatio="xMidYMid meet" focusable="false">
          <title id="queue-figure-title">Conceptual review-queue field</title>
          <desc id="queue-figure-desc">A source patch of nucleus instances feeds a ranked review queue of three slots. Illustration only, not benchmark data.</desc>
          <defs>
            <linearGradient id="queue-flow" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#7884df" stop-opacity=".12"/><stop offset="1" stop-color="#8d98f4" stop-opacity=".78"/></linearGradient>
          </defs>
          <rect class="fallback-panel" x="42" y="43" width="525" height="214" rx="10"/>
          <rect class="fallback-panel" x="647" y="43" width="270" height="214" rx="10"/>
          <text class="fallback-label" x="66" y="72">SOURCE PATCH</text>
          <text class="fallback-label" x="671" y="72">REVIEW QUEUE</text>
          <g class="fallback-nuclei">
            <g class="cell"><path class="cell-membrane" d="M88 128 C90 108 102 96 118 94 C136 92 150 104 152 120 C154 136 144 150 128 154 C112 158 98 150 92 138 C88 132 87 130 88 128 Z"/><ellipse class="cell-nucleus" cx="118" cy="122" rx="7.5" ry="6.2"/></g>
            <g class="cell"><path class="cell-membrane" d="M158 108 C162 88 178 78 196 82 C214 86 226 100 222 118 C218 136 202 146 184 142 C166 138 154 126 158 108 Z"/><ellipse class="cell-nucleus" cx="190" cy="110" rx="6.8" ry="8"/></g>
            <g class="cell"><path class="cell-membrane" d="M220 156 C224 134 242 122 262 128 C280 134 290 152 282 170 C274 186 254 192 236 184 C220 177 216 166 220 156 Z"/><ellipse class="cell-nucleus" cx="250" cy="154" rx="8" ry="6.5"/></g>
            <g class="cell"><path class="cell-membrane" d="M298 118 C300 98 314 88 332 90 C350 92 360 106 356 122 C352 138 338 148 320 144 C304 140 296 130 298 118 Z"/><ellipse class="cell-nucleus" cx="328" cy="116" rx="7" ry="7.2"/></g>
            <g class="cell is-focus"><path class="cell-membrane" d="M388 100 C390 84 404 76 418 80 C432 84 440 98 436 112 C432 126 416 132 402 128 C388 124 384 112 388 100 Z"/><ellipse class="cell-nucleus" cx="408" cy="102" rx="7.5" ry="6.5"/></g>
            <g class="cell"><path class="cell-membrane" d="M458 124 C460 104 474 94 492 98 C510 102 520 118 514 134 C508 150 492 156 474 150 C458 144 456 134 458 124 Z"/><ellipse class="cell-nucleus" cx="486" cy="124" rx="6.5" ry="7.5"/></g>
            <g class="cell"><path class="cell-membrane" d="M100 214 C102 194 116 184 134 188 C152 192 162 208 156 224 C150 240 134 246 116 240 C100 234 98 224 100 214 Z"/><ellipse class="cell-nucleus" cx="128" cy="214" rx="7.2" ry="6.4"/></g>
            <g class="cell"><path class="cell-membrane" d="M172 226 C176 206 194 196 214 202 C232 208 242 224 234 240 C226 254 206 258 188 250 C172 243 168 234 172 226 Z"/><ellipse class="cell-nucleus" cx="204" cy="226" rx="7.8" ry="6.8"/></g>
            <g class="cell"><path class="cell-membrane" d="M262 208 C264 188 278 178 296 182 C314 186 324 200 318 216 C312 232 296 238 278 232 C262 226 260 216 262 208 Z"/><ellipse class="cell-nucleus" cx="290" cy="208" rx="6.6" ry="7.4"/></g>
            <g class="cell"><path class="cell-membrane" d="M330 228 C334 208 350 198 368 204 C386 210 394 226 386 242 C378 256 358 260 342 252 C328 245 326 236 330 228 Z"/><ellipse class="cell-nucleus" cx="360" cy="228" rx="7.4" ry="6.2"/></g>
            <g class="cell"><path class="cell-membrane" d="M430 216 C434 196 450 186 468 192 C486 198 494 214 486 230 C478 244 458 248 442 240 C428 233 426 224 430 216 Z"/><ellipse class="cell-nucleus" cx="460" cy="216" rx="7" ry="7.6"/></g>
          </g>
          <circle class="fallback-selected" cx="408" cy="102" r="34" pathLength="100"/>
          <path class="fallback-path" d="M442 102 H674"/>
          <g class="fallback-queue"><rect x="674" y="87" width="205" height="30" rx="6"/><rect x="674" y="130" width="205" height="30" rx="6"/><rect x="674" y="173" width="205" height="30" rx="6"/></g>
          <g class="fallback-ranks"><text x="687" y="107">01</text><text x="687" y="150">02</text><text x="687" y="193">03</text></g>
        </svg>
      </div>
      <figcaption>Figure. Review-queue sketch (not study data).</figcaption>
    </figure>
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
        <p class="journey-intro" id="journey-title">The diagram shows the complete governed path: preserve the reference, create a known intervention, predict without source-patch leakage, rank an equal review budget and measure retrieval while leaving the final decision to an expert.</p>
      </div>
    </div></div>
  </section>

  <section class="section" id="reading">
    <div class="section-heading reveal"><h2>Four quantities answer four different questions.</h2><p class="section-deck">Separating them prevents a strong ranking result from being mistaken for clinical or downstream utility.</p></div>
    <div class="chapter-copy reveal">
      <p>Average precision describes the complete ordering of the review queue. Recall at the primary 5% budget describes the operational slice an expert would actually inspect. The random-review baseline tells us whether that prioritisation is better than spending the same review effort without a model.</p>
      <p>Restoration utility is deliberately separate. It asks what happens after simulated review when found injected changes are restored and the downstream classifier is retrained. A ranking can retrieve controlled inconsistencies efficiently while still failing to improve the later classifier.</p>
      <p>Intervals and adjusted tests also answer different questions. The displayed 95% intervals come from paired whole-group bootstrap resampling; registered Holm-adjusted values are one-sided. At 0% corruption, average precision is not applicable because the controlled positive event does not exist.</p>
    </div>
  </section>

  <section class="section" id="results">
    <div class="section-heading reveal"><h2>On the original PanNuke benchmark, better triage did not improve the downstream model.</h2><p class="section-deck">This is the most important negative PanNuke result. Ranking injected inconsistencies and improving a later classifier are related, but they are not equivalent objectives.</p></div>
    <div class="chapter-copy reveal"><p>The H4 experiment restored exactly the same 5% review budget under audit-guided and random selection, then trained the same downstream model and evaluated it on the untouched final reference fold. Audit-guided restoration was not favoured.</p></div>
    <div class="figure-width reveal result-layout">
      <div class="result-lead"><div class="result-label"><span>Audit-guided minus random review</span></div><span class="result-number">__H4_POINT__</span><p>Audit-guided restoration produced macro-F1 <strong>__H4_AUDIT__</strong>, below the random-review mean of <strong>__H4_RANDOM__</strong>.</p><div class="result-interpretation"><b>Plain-language interpretation</b><p>The audit-guided result was <strong>__H4_POINT_PP__ percentage points lower</strong>. The saved 95% interval remained below zero, so this registered test did not support downstream benefit.</p></div></div>
      __H4_CHART__
    </div>
    <div class="chapter-copy result-afterword reveal">
      <p>This result does not erase the ranking evidence. It shows that retrieving more injected label changes near the front of a queue and improving final-fold classification are not interchangeable objectives. The first evaluates triage efficiency; the second depends on how the reviewed cases affect a later training distribution.</p>
      <p>Only reviewed injected corruptions were restored; unreviewed observations remained unchanged, and all downstream conditions used the same representation, learner, seeds, review count and untouched final fold. The adverse direction is retained without selecting a post-result explanation.</p>
    </div>
  </section>

  <section class="section learned-section" id="benchmarks">
    <div class="chapter-copy findings-intro reveal">
      <p class="section-kicker">The remaining registered questions</p>
      <p>H1 through H7 were designed to separate retrieval, heterogeneity, corruption difficulty, restoration, score combination, representation availability and target highlighting. Reading them together prevents one favourable comparison from becoming the story of the entire study.</p>
      <p>The pattern is mixed. H1 supports better-than-random prioritisation inside the controlled benchmark, H5 favours the fixed hybrid, and H3 indicates that some injected mechanisms were harder to retrieve. H4 is adverse, H7 is unresolved, and H6 is unavailable because no pathology encoder passed the frozen eligibility gates.</p>
      <p>Each answer below states the conclusion, the exact saved evidence and the interpretation boundary. None of the positive ranking results establishes that a naturally occurring annotation is wrong.</p>
    </div>
    <div class="learned-story article-findings" id="learned-story">
      <div class="learned-sticky"><div class="learned-shell">
        <div class="section-heading learned-heading"><h2>What the study actually learned.</h2></div>
        <div class="learned-stage" role="region" aria-label="Seven preregistered research questions and their answers">
          <div class="hypothesis-ledger" aria-label="Preregistered findings">__HYPOTHESIS_LEDGER__</div>
        </div>
      </div></div>
    </div>
    <div class="section-heading reveal" style="margin-top:var(--section-space)"><h2>Every preregistered comparison entry.</h2><p class="section-deck">Each point is an average-precision difference; each line is its saved two-sided 95% percentile-bootstrap interval. Missing H6 points are shown as unavailable rather than as zero.</p></div>
    <div class="chapter-copy reveal"><p>The atlas is the transition from narrative findings to detailed evidence. Rows remain grouped by hypothesis so that effect direction, interval width, unavailable cells and repeated seed structure can be inspected together instead of reduced to a single headline number.</p><p>The plot is intentionally bounded inside the article. Scrolling within it reveals all 36 registered entries while keeping the surrounding explanation in view; the complete numeric record appears later in the evidence table.</p></div>
    <div class="wide-width reveal"><div class="forest-plot" role="region" aria-label="Scrollable forest plot of all H1, H3, H5, H6 and H7 comparisons" tabindex="0">__FOREST_PLOT__</div><p class="fine"><strong>Statistical reading.</strong> Registered Holm-adjusted p-values are one-sided, while the displayed 95% intervals are two-sided summaries. Those two summaries need not produce identical verbal labels.</p></div>
  </section>

  <section class="section" id="subgroups">
    <div class="section-heading reveal"><h2>Saved performance estimates varied across contexts.</h2><p class="section-deck">The saved subgroup estimates span nucleus classes, tissues, corruption mechanisms and corruption rates. They describe where performance differed in this benchmark; they do not establish a causal biological dependency.</p></div>
    <div class="chapter-copy reveal"><p>Subgroup average precision is reported only when the preregistered support rule is met: at least 100 samples and at least 10 injected corruptions. Otherwise the study retains counts without inventing an unstable estimate.</p><p>The ranges expose heterogeneity that would disappear inside one aggregate score. They are descriptive, not a biological ranking: the preregistration did not define an omnibus test that could support a causal explanation for tissue, class, mechanism or corruption-rate differences.</p></div>
    <div class="figure-width reveal">__H2_CHART__</div>
  </section>

  <section class="section" id="seed-audit">
    <div class="section-heading reveal"><h2>Three registered seeds produced one realisation.</h2><p class="section-deck">The three rows are retained for auditability, but the byte-identical files are not independent realisations and must not be interpreted as three replications.</p></div>
    <div class="figure-width reveal">
      <div class="seed-copy"><h3>One deterministic output, preserved through three registered paths.</h3><p>The ranking and out-of-fold prediction files match byte for byte. The disclosure changes interpretation, not the stored record: these rows document one realisation rather than three replications.</p></div>
      <details class="evidence-details seed-evidence"><summary><span class="seed-summary-icon" aria-hidden="true">#</span><span class="seed-summary-copy"><strong>Inspect exact seed identities and SHA-256 hashes</strong><small>Three registered paths · one byte-identical realisation</small></span><span class="seed-summary-action" aria-hidden="true">View evidence</span></summary><div class="table-wrap"><table class="compact"><thead><tr><th>Seed</th><th>Cell ID</th><th>Ranking SHA-256</th><th>OOF SHA-256</th></tr></thead><tbody>__SEED_ROWS__</tbody></table></div></details>
    </div>
  </section>

  <section class="section" id="evidence">
    <div class="section-heading reveal"><h2>Every displayed result remains inspectable.</h2><p class="section-deck">Machine-readable summaries, exact identifiers and the complete 36-entry table remain available for audit. Package verification checks the published files; it does not recompute the primary study.</p></div>
    <div class="chapter-copy reveal"><p>Every value in the article is read from the accepted sealed run and carried into <code>evidence.json</code>. The table below traces each statement to its hypothesis, raw identifier, seed, point difference, confidence interval, adjusted p-value and bootstrap count.</p><p>The timestamped July freeze records a null commit, a dirty tree and untracked files; the first public Git commit followed on 19 August 2026. Internal hashes preserve file identity, but the later public history is not independent evidence that outcomes were unseen.</p><p>The public <a href="https://github.com/Jaqwilk/AANCA/releases/tag/primary-evidence-v1" target="_blank" rel="noopener">primary evidence release</a> contains all completed-cell OOF predictions and rankings, the full group bootstrap, subgroup table and H4 restoration arrays. A standalone evidence recalculator that does not import the primary analysis package recomputes the saved H1-H7 comparison statistics; this is not third-party validation. Fold checkpoints were not retained, and a second image-to-result execution still requires a lawful PanNuke copy.</p></div>
    <div class="wide-width reveal">
      <details class="evidence-details comparison-details"><summary>Inspect the complete H1 / H3 / H5 / H6 / H7 table <span>__REPORTED__ reported · __UNAVAILABLE__ unavailable</span></summary>
        <div class="table-tools"><label>Hypothesis<select id="filter-hypothesis"><option value="ALL">All hypotheses</option><option>H1</option><option>H3</option><option>H5</option><option>H6</option><option>H7</option></select></label><label>Status<select id="filter-status"><option value="ALL">All statuses</option><option value="reported">Reported</option><option value="not_available_frozen_optional_cell">Unavailable</option></select></label><label>Search<input id="filter-search" type="search" placeholder="Name, seed or comparison ID"></label><span class="table-count" id="table-count">__COMPARISONS__ / __COMPARISONS__ rows</span></div>
        <div class="table-wrap" role="region" aria-label="Complete comparison results" tabindex="0"><table id="comparison-table"><colgroup><col style="width:7%"><col style="width:35%"><col style="width:11%"><col style="width:11%"><col style="width:19%"><col style="width:10%"><col style="width:7%"></colgroup><thead><tr><th>Hypothesis</th><th>Comparison</th><th>Status</th><th style="text-align:right">Δ AP</th><th style="text-align:right">95% bootstrap CI</th><th style="text-align:right">Holm-adjusted p</th><th style="text-align:right">Iterations</th></tr></thead><tbody>__COMPARISON_ROWS__</tbody></table></div>
      </details>
      <p class="fine"><strong>Statistical note.</strong> Holm-adjusted p-values are the registered one-sided tests. The saved 95% confidence intervals are percentile-bootstrap summaries based on whole-group resampling.</p>
      <div class="provenance" style="margin-top:64px">
        <div><h3>Research sources</h3><ul><li><a href="https://link.springer.com/chapter/10.1007/978-3-030-23937-4_2">Gamper et al. (2019), PanNuke</a> · DOI 10.1007/978-3-030-23937-4_2.</li><li><a href="https://arxiv.org/abs/2003.10778">Gamper et al. (2020), PanNuke extension</a> · dataset insights and baselines.</li><li><a href="https://www.jair.org/index.php/jair/article/view/12125">Northcutt, Jiang &amp; Chuang (2021), Confident Learning</a> · DOI 10.1613/jair.1.12125.</li><li><a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/fc20ea8d104cab737a5561096f9bde9b-Abstract.html">Goswami et al. (2023), AQuA</a> · DOI 10.52202/075280-3494.</li><li><a href="https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giac037/6586817">Amgad et al. (2022), NuCLS</a> · DOI 10.1093/gigascience/giac037.</li><li><a href="https://monusac-2020.grand-challenge.org/">MoNuSAC 2020</a> · controlled external benchmark source.</li><li><a href="https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giaf011/8024182">Schuiveling et al. (2025), PUMA</a> · DOI 10.1093/gigascience/giaf011.</li></ul><p>Grouping follows the strongest identifiers available per dataset. No result is promoted beyond the grouping and reference-label evidence that the source actually provides.</p></div>
        <div><h3>Reproducibility identities</h3><dl class="hash-list"><div><dt>Accepted run</dt><dd>__RUN_ID__</dd></div><div><dt>Artifact root SHA-256</dt><dd>__ARTIFACT_ROOT__</dd></div><div><dt>Stage attestation SHA-256</dt><dd>__STAGE_HASH__</dd></div><div><dt>QC overlay SHA-256</dt><dd>__QC_HASH__</dd></div></dl></div>
      </div>
    </div>
  </section>

  <section class="section" id="external-validation">
    <div class="section-heading reveal"><h2>The same system transferred under controlled noise, but natural-case proof is still missing.</h2><p class="section-deck">Later evidence changes the overall assessment without rewriting the adverse PanNuke H4 result. Each study answers a different question and keeps its own frozen boundary.</p></div>
    <div class="chapter-copy reveal"><p>NuCLS is the closest completed test of genuine multi-rater disagreement. Its five-patient result did not satisfy the frozen ranking gate and the guided intervention made downstream macro-F1 worse. MoNuSAC then showed strong controlled retrieval but no statistically supported downstream gain or complete class safety.</p><p>The selected AANCA candidate was next recorded internally as frozen before PUMA outcomes. On a previously unused histopathology source it passed all seven internally pre-specified retrieval, downstream, direction, convergence and class-safety gates under controlled corruption. The 144/62 development/final partition was created by AANCA from the 206 public PUMA ROIs; it was not the official hidden PUMA challenge test. The downstream <code>flag_exclude</code> arm omitted the highest-ranked 5% of training rows without expert review or relabelling. This is meaningful controlled-noise transfer evidence, but PUMA does not provide paired natural pre/post expert decisions and therefore cannot prove pathologist-error detection.</p><p><strong>Public-history limit.</strong> The PUMA protocol, configuration and result first appeared together in public commit <code>c5bd44193b2abd67bc7e7f1bd9384aa87435d500</code>, so public Git history does not independently verify the intended pre-outcome timing. The PUMA verifier is a project-coupled evidence-readback script that recomputes metrics from saved predictions but does not retrain all 44 models. It is not third-party validation or a second image-to-result replication.</p></div>
__CURRENT_EVIDENCE__
  </section>

  <section class="section" id="quality">
    <div class="section-heading reveal"><h2>The source masks remained untouched.</h2><p class="section-deck">The validator measured cross-class overlaps and unlabeled regions, retained them in provenance and applied one frozen eligibility policy. It never arbitrated an overlap class or reconstructed the supplied background.</p></div>
    <div class="chapter-copy reveal"><p>Before any primary split is frozen, the inspector verifies folds, shapes, channels, class order, instance identifiers and representative overlays. Cross-class overlaps, void pixels and affected instances are recorded rather than silently corrected.</p><p>The overlay is evidence about ingestion and representation, not improved segmentation. Eligibility flags may follow the frozen policy, but the raw masks and source annotations remain unchanged.</p></div>
    <div class="wide-width reveal">
      <div class="stats-grid"><div class="stat"><strong>__PATCHES__</strong><span>validated PanNuke patches</span></div><div class="stat"><strong>__OVERLAP_PIXELS__</strong><span>cross-class overlap pixels</span></div><div class="stat"><strong>__VOID_PIXELS__</strong><span>unlabeled / void pixels</span></div><div class="stat"><strong>__FLAGGED_INSTANCES__</strong><span>overlap-touching instances flagged</span></div></div>
      <div class="always-visible-evidence"><div class="evidence-label-row"><span>Deterministic quality-control overlay</span><span>1512 x 3840 px · source preview</span></div><div class="qc-frame"><div class="qc-toolbar"><span>Cropped representative preview · normal, overlap, void and exclusion cases</span><a href="pannuke_mask_qc_overlays.png" target="_blank" rel="noopener">Open the original image ↗</a></div><div class="qc-preview"><img loading="lazy" decoding="async" fetchpriority="low" width="1512" height="3840" src="pannuke_mask_qc_overlays.png" alt="Deterministic PanNuke quality-control overlays showing images and mask boundaries"></div></div></div>
    </div>
  </section>

  <section class="section" id="integrity">
    <div class="section-heading reveal"><h2>The design limits outcome-informed model selection.</h2><p class="section-deck">The model, split, feature and statistical rules were frozen before outcome interpretation. Because outcomes were later exposed during recovery, this accepted result is reported as exploratory rather than as an untouched confirmatory analysis.</p></div>
    <div class="chapter-copy reveal"><p>Source groups stay together, each audit score is out of fold, the final reference fold is withheld from selection, and every label state remains separate. The matrix retained all 36 comparison entries; unavailable optional pathology cells were not replaced with estimates.</p><p>Outcome exposure during technical recovery does not change the stored measurements, but it narrows what can responsibly be claimed. The accepted primary analysis is therefore permanently described as amended and exploratory.</p></div>
    <div class="wide-width reveal">
      <div class="stats-grid"><div class="stat"><strong>__COMPLETED__/185</strong><span>required primary cells completed</span></div><div class="stat"><strong>__FAILED__</strong><span>failed required cells</span></div><div class="stat"><strong>__COMPARISONS__</strong><span>preregistered comparison entries retained</span></div><div class="stat"><strong>__BOOTSTRAPS__</strong><span>paired group-bootstrap iterations</span></div></div>
      <div class="rules"><div class="rule"><b>Group-safe splitting</b><p>Every split uses <code>group_id</code>, at least at source-patch level, never individual nuclei.</p></div><div class="rule"><b>Out-of-fold ranking</b><p>Primary model-based audit scores come from predictions made for source groups excluded from model fitting.</p></div><div class="rule"><b>Untouched final reference</b><p>The final reference fold is uncorrupted and unavailable for model selection, calibration or review-budget tuning.</p></div><div class="rule"><b>Separate label states</b><p><code>pre_corruption_label</code>, <code>observed_label</code>, <code>is_injected_corruption</code> and corruption metadata remain distinct.</p></div></div>
    </div>
  </section>

  <section class="section" id="interpretation">
    <div class="section-heading reveal"><h2>What this benchmark supports, and what it does not.</h2><p class="section-deck">A technically strong audit can still be limited in scope. The boundary below is part of the result, not a disclaimer added afterward.</p></div>
    <div class="chapter-copy reveal"><p>The controlled positive event is an injected label change. Performance against it measures retrieval of that injected process, not naturally occurring annotation inconsistency, expert disagreement or biological truth. The positive PUMA result is a new-source controlled transfer result, while the later stress analysis and observed-label sensitivity are exploratory because PUMA outcomes were already open.</p><p>Grouping strength differs by source: the primary PanNuke study guarantees source-patch separation, NuCLS uses patient groups, MoNuSAC uses patient groups and PUMA uses one ROI per case. These safeguards reduce leakage but do not transform a final expert reference label into guaranteed biological truth.</p><p>Natural and operational validity still require newly recruited blinded reviewers, ambiguity and abstention labels, a frozen policy evaluated on untouched patients or whole slides, and a prospective multi-site comparison of work with and without AANCA.</p></div>
    <div class="figure-width reveal claim-grid">
      <article class="claim-card"><h3>Supported by current evidence</h3><ul><li>Group-safe rankings retrieve injected class-label changes more efficiently than matched random review.</li><li>The frozen current candidate transferred to new PUMA images and improved controlled-noise downstream macro-F1 with positive whole-group intervals.</li><li>Every displayed conclusion is read from checksum-bound machine-readable evidence.</li></ul></article>
      <article class="claim-card"><h3>Not established</h3><ul><li>That a naturally occurring annotation is wrong, that a pathologist made an error or that model disagreement is biological truth.</li><li>That the intervention is uniformly class-safe across realistic corruption patterns; only one of nine stress scenarios passed every class safeguard.</li><li>Prospective workflow benefit, multi-site generalisation, clinical utility or permission to alter natural source labels automatically.</li></ul></article>
    </div>
  </section>

  <section class="section" id="current-stage">
    <div class="section-heading reveal"><h2>AANCA is externally evaluated, but not yet confirmatory or ready for real-use claims.</h2><p class="section-deck">Completion labels describe which governed evaluations ran. They do not turn a mixed result into efficacy.</p></div>
    <div class="chapter-copy reveal"><p>The current project has reached <code>PRIMARY_STUDY_COMPLETE</code>, <code>EXTERNAL_VALIDATION_COMPLETE</code> and <code>DEMO_COMPLETE</code>. It has not reached <code>CONFIRMATORY_COMPLETE</code>. The binding natural-data action is <code>retain_uncorrected</code>: AANCA may prioritise cases for qualified review, but it may not silently exclude, relabel or overwrite them.</p><p>The next research phase is provisionally named <strong>AANCA v2</strong>. This is a prospective evidence programme for the same core auditing system, not a retroactive rename of the existing results. Opened PanNuke, NuCLS, MoNuSAC and PUMA outcomes may inform diagnosis of weaknesses, but they cannot serve as the untouched final confirmation for the next claim.</p></div>
    <div class="figure-width reveal rules phase-ledger">
      <article class="rule"><b>1 · Natural reference</b><p>Recruit independent blinded pathologists on new cases and preserve agreement, disagreement, ambiguity, abstention and insufficient-context outcomes instead of forcing one truth label.</p></article>
      <article class="rule"><b>2 · Measured utility</b><p>Develop the review queue inside nested patient- or WSI-group cross-fitting using both inconsistency probability and conservatively estimated downstream benefit.</p></article>
      <article class="rule"><b>3 · Prospective freeze</b><p>Freeze one representation, queue, intervention, review budget, class-safety rule and analysis plan before inspecting any new confirmation outcome.</p></article>
      <article class="rule"><b>4 · Untouched confirmation</b><p>Run one-shot external validation on new patient or WSI groups. Ranking, downstream confidence intervals, every-class safety and convergence must all pass together.</p></article>
      <article class="rule"><b>5 · Real workflow</b><p>Compare blinded multi-site review with and without AANCA, measuring time, agreement, accepted corrections, downstream performance and failure modes.</p></article>
      <article class="rule"><b>Promotion boundary</b><p>Only the full sequence can support a claim about realistic natural-case improvement. Until then, AANCA remains a non-diagnostic expert-review prioritisation prototype.</p></article>
    </div>
  </section>

  <section class="section" id="use">
    <div class="repo-shell">
      <div class="reproduction-intro reveal"><h2>Read the evidence first; run the software when a deeper check is needed.</h2><p>The checked-in article opens without a dataset, model run or GPU. <code>present_demo.py --verify-only</code> verifies the five-file presentation package and its current PanNuke, NuCLS, MoNuSAC and PUMA summaries; it does not retrain a model or recalculate a scientific result. The separate synthetic path exercises the portable software workflow.</p><p>The public repository retains frozen protocols, configs, compact results and arrays for scoped evidence verification. The primary, NuCLS and MoNuSAC scripts independently recalculate their stated saved evidence. The PUMA readback imports maintained helpers and checks stored predictions rather than independently retraining the 44 models. Full image-to-result replication still requires lawfully obtained source datasets and appropriate compute; the project never downloads protected data silently or relaxes scientific gates when an input is unavailable.</p></div>
      <article class="repo-card reveal">
        <div class="repo-top">
          <svg class="repo-icon" viewBox="0 0 24 24" role="img" aria-label="GitHub"><path fill="currentColor" d="M12 .7A11.3 11.3 0 0 0 8.43 22.72c.56.1.77-.24.77-.54v-2.11c-3.13.68-3.79-1.33-3.79-1.33-.51-1.3-1.25-1.65-1.25-1.65-1.02-.7.08-.69.08-.69 1.13.08 1.72 1.16 1.72 1.16 1 1.72 2.63 1.22 3.27.93.1-.73.39-1.22.71-1.5-2.5-.28-5.13-1.25-5.13-5.58 0-1.23.44-2.24 1.16-3.03-.12-.29-.5-1.44.11-2.99 0 0 .95-.3 3.11 1.16A10.8 10.8 0 0 1 12 6.16c.96 0 1.92.13 2.82.38 2.16-1.46 3.1-1.16 3.1-1.16.62 1.55.23 2.7.12 2.99.72.79 1.16 1.8 1.16 3.03 0 4.34-2.64 5.29-5.15 5.57.4.35.77 1.04.77 2.1v3.11c0 .3.2.65.78.54A11.3 11.3 0 0 0 12 .7Z"/></svg>
          <div><span class="repo-path">github.com / Jaqwilk</span><span class="repo-title">AANCA</span></div>
          <a class="repo-link" href="https://github.com/Jaqwilk/AANCA" target="_blank" rel="noopener">View repository ↗</a>
        </div>
        <p class="repo-description">The repository contains the scientific specification, maintained source code, frozen study configs, scoped verification scripts, compact evidence and this checksum-verifiable presentation. Licensed raw datasets, reusable local embeddings and unretained historical checkpoints are deliberately excluded.</p>
        <div class="repo-tags"><span>Python 3.12</span><span>group-safe OOF</span><span>immutable evidence</span><span>research only</span></div>
      </article>

      <div class="usage-grid">
        <article class="repo-command"><h3>Verify and open locally</h3><p>Check the package, serve it locally and open the article.</p><pre>git clone https://github.com/Jaqwilk/AANCA.git
cd AANCA
python scripts/present_demo.py</pre></article>
        <article class="repo-command"><h3>Check without a browser</h3><p>Validate the presentation package without recomputing the study.</p><pre>python scripts/present_demo.py --verify-only</pre></article>
        <article class="repo-command"><h3>Run the synthetic smoke path</h3><p>Exercise the software with deterministic synthetic data.</p><pre>uv sync --dev
uv run histo-audit doctor
uv run histo-audit data generate-synthetic --config configs/smoke.yaml
uv run histo-audit experiment smoke --runs-root artifacts/smoke_runs</pre></article>
      </div>
      <p class="usage-note">Real PanNuke runs require a lawful local dataset and the governed setup in <a href="https://github.com/Jaqwilk/AANCA/blob/main/DATASET_SETUP.md">DATASET_SETUP.md</a>.</p>
    </div>
  </section>

  <section class="section author-section" id="author" aria-labelledby="author-title">
    <div class="author-shell">
      <div class="author-profile reveal">
        <h2 id="author-title">Research and implementation by Natan Smogór.</h2>
        <p class="author-lede">Natan Smogór is the author and developer of AANCA, a non-diagnostic research prototype for prioritising potentially inconsistent nucleus annotations for expert review.</p>
        <p class="author-copy">The project combines a frozen scientific specification, controlled data interventions, group-safe evaluation, machine-readable provenance and an inspectable presentation layer. Its purpose is to demonstrate a reproducible research workflow while keeping the final interpretation with qualified experts.</p>
      </div>

      <aside class="author-credentials reveal" aria-label="Education and credentials">
        <article class="credential-record">
          <div class="credential-heading"><a class="credential-logo" href="https://www.kozminski.edu.pl/en" target="_blank" rel="noopener" aria-label="Visit Kozminski University"><img src="https://www.kozminski.edu.pl/themes/custom/leon_copy/images/kozminski_logo.svg" alt="Kozminski University logo"></a></div>
          <h3><a href="https://www.kozminski.edu.pl/en" target="_blank" rel="noopener">Kozminski University</a></h3>
          <p class="credential-course">Management and Artificial Intelligence</p>
          <dl class="credential-meta"><div><dt>Location</dt><dd>Warsaw, Poland</dd></div><div><dt>Status</dt><dd>Current student</dd></div></dl>
        </article>

        <article class="credential-record">
          <div class="credential-heading"><a class="credential-logo" href="https://uniw.edu.pl/" target="_blank" rel="noopener" aria-label="Visit Uniwersytet Młodzieżowy"><img src="https://uniw.edu.pl/wp-content/uploads/2025/08/Uniwersytet-Mlodziezowy-rebrand-BIALY-scaled.png" alt="Uniwersytet Młodzieżowy logo"></a></div>
          <h3><a href="https://uniw.edu.pl/" target="_blank" rel="noopener">Uniwersytet Młodzieżowy</a></h3>
          <p class="credential-course">Artificial Intelligence</p>
          <dl class="credential-meta"><div><dt>Duration</dt><dd>80 hours</dd></div><div><dt>Academic year</dt><dd>2024/2025</dd></div></dl>
          <div class="credential-proof" aria-label="Completion diploma for the 80-hour Artificial Intelligence programme">
            <svg viewBox="0 0 48 48" aria-hidden="true"><path d="M13 5.5h16l7 7v30H13z"/><path d="M29 5.5v8h7M18 24h13M18 30h13M18 36h8"/></svg>
            <div><strong>Completion diploma</strong><span>80-hour programme, completed in the 2024/2025 academic year</span></div>
          </div>
        </article>

        <p class="institution-note">Education references identify the author's background and do not imply institutional endorsement of AANCA.</p>
      </aside>
    </div>
  </section>
</main>

<footer id="footer"><div class="footer-inner"><div class="footer-divider"></div>
  <div class="footer-grid">
    <div><div class="footer-brand-row"><span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span><span class="footer-brand">AANCA</span></div><p class="footer-summary">Automated auditing of nucleus class annotations: a university research prototype for prioritising potentially inconsistent annotations for expert review. Non-diagnostic research; source annotations are never changed automatically.</p></div>
    <div class="footer-column"><h3>Study</h3><p>Author: Natan Smogór</p><p>Updated: 21 August 2026</p><p>Evidence: PanNuke, NuCLS, MoNuSAC and PUMA</p><p>External evaluation complete; controlled PUMA transfer supported; natural-case confirmation and clinical utility not established.</p></div>
    <div class="footer-column"><h3>Inspect</h3><a href="https://github.com/Jaqwilk/AANCA/blob/main/PROFESSOR_BRIEF.md" target="_blank" rel="noopener">One-page project brief ↗</a><a href="evidence.json">Machine-readable evidence</a><a href="https://github.com/Jaqwilk/AANCA/releases/tag/primary-evidence-v1" target="_blank" rel="noopener">Primary evidence release ↗</a><a href="README.md">Presentation package notes</a><a href="#evidence">Reproducibility boundary</a><a href="https://github.com/Jaqwilk/AANCA" target="_blank" rel="noopener">GitHub repository ↗</a><a href="https://github.com/Jaqwilk/AANCA/blob/main/CONTRIBUTIONS.md" target="_blank" rel="noopener">Contributions and AI use ↗</a><a href="https://github.com/Jaqwilk/AANCA/blob/main/ETHICS_AND_LIMITATIONS.md" target="_blank" rel="noopener">Ethics and limitations ↗</a><a href="https://github.com/Jaqwilk/AANCA/blob/main/references/references.bib" target="_blank" rel="noopener">Bibliography ↗</a></div>
  </div>
  <div class="footer-evidence"><span>PUMA public evidence commit <a href="https://github.com/Jaqwilk/AANCA/commit/__PUMA_PUBLIC_COMMIT__" target="_blank" rel="noopener"><code>__PUMA_PUBLIC_COMMIT_SHORT__</code></a></span><span><code>evidence.json</code> SHA-256 <code>__EVIDENCE_SHA256__</code></span></div>
  <div class="footer-bottom"><span>© 2026 Natan Smogór. Project code and original documentation are all rights reserved under the repository <a href="https://github.com/Jaqwilk/AANCA/blob/main/LICENSE" target="_blank" rel="noopener">LICENSE</a>.</span><span>Dataset files, pretrained weights, dependencies and third-party assets retain their own terms. No diagnostic or clinical use is established.</span></div>
</div></footer>
<script src="https://cdn.jsdelivr.net/npm/gsap@3.15.0/dist/gsap.min.js" integrity="sha384-XmJ9SoHtVOHoQUcKvFAzVXwdkKo1Ie3bhmSoIAkcdsHGaIrVJIkmozyq0FJeb/Ly" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/gsap@3.15.0/dist/ScrollTrigger.min.js" integrity="sha384-wl5TeDVvOWt30Pbf8aSo2ZrzsOjddu3avOBvHe+p+OhJt9gP6w9YXmDkN5DK2/dF" crossorigin="anonymous"></script>
<script>__SCRIPT__</script>
</body>
</html>
"""

    replacements = {
        "__CSS__": _MVP_CSS,
        "__SCRIPT__": _MVP_SCRIPT,
        "__STUDY_SPECS__": _render_study_specs(),
        "__H4_POINT__": html.escape(_format_metric(point_difference)),
        "__H4_POINT_PP__": html.escape(f"{abs(point_difference) * 100:.3f}"),
        "__H4_AUDIT__": html.escape(_format_metric(h4["audit_guided_macro_f1"])),
        "__H4_RANDOM__": html.escape(_format_metric(h4["random_review_macro_f1_mean"])),
        "__H4_CHART__": h4_chart,
        "__PUMA_PRECISION_4__": f"{puma['retrieval']['candidate_precision']:.4f}",
        "__PUMA_RANDOM_PRECISION_4__": (
            f"{puma['retrieval']['mean_matched_random_precision']:.4f}"
        ),
        "__PUMA_DELTA_F1_4__": f"{puma['downstream']['minus_uncorrected']:+.4f}",
        "__PUMA_PUBLIC_COMMIT__": html.escape(
            str(publication_limits["puma_first_public_combined_commit"])
        ),
        "__PUMA_PUBLIC_COMMIT_SHORT__": html.escape(
            str(publication_limits["puma_first_public_combined_commit"])[:12]
        ),
        "__EVIDENCE_SHA256__": html.escape(evidence_sha256),
        "__HYPOTHESIS_LEDGER__": hypothesis_ledger,
        "__CURRENT_EVIDENCE__": current_evidence,
        "__FOREST_PLOT__": forest_plot,
        "__H2_CHART__": h2_chart,
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


def _format_metric(value: Any) -> str:
    """Format presentation metrics consistently without scientific notation."""

    if value is None:
        return "Unavailable"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        return "[" + ", ".join(_format_metric(item) for item in value) + "]"
    return str(value)


def _comparison_display_label(comparison_id: str) -> str:
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


def _render_comparison_rows(comparisons: list[dict[str, Any]]) -> str:
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
            + html.escape(_comparison_display_label(comparison_id))
            + '</span><code class="comparison-id">'
            + html.escape(comparison_id)
            + "</code></td>"
            + '<td data-label="Status"><span class="status-pill '
            + status_class
            + '">'
            + status_label
            + "</span></td>"
            + '<td class="numeric" data-label="Δ AP">'
            + html.escape(_format_metric(item["point_difference"]))
            + "</td>"
            + '<td class="numeric" data-label="95% bootstrap CI">'
            + html.escape(_format_metric(item["interval_95"]))
            + "</td>"
            + '<td class="numeric" data-label="Holm-adjusted p">'
            + html.escape(_format_metric(item["p_value_holm"]))
            + "</td>"
            + '<td class="numeric" data-label="Iterations">'
            + html.escape(_format_metric(item["valid_bootstrap_iterations"]))
            + "</td></tr>"
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
        label = _comparison_display_label(comparison_id)
        if item.get("status") != "reported":
            output.append(
                '<div class="forest-row is-unavailable">'
                f'<span class="forest-label" title="{html.escape(comparison_id)}">'
                f"{html.escape(label)}</span>"
                '<span class="forest-track forest-track-empty">'
                "<span>Unavailable by frozen design</span></span>"
                '<span class="forest-value">Unavailable</span></div>'
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
            f'<span class="forest-value">{html.escape(_format_metric(point))}</span></div>'
        )
    return "".join(output)


def _render_h4_chart(h4: dict[str, Any]) -> str:
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
        '<div class="chart-title-row"><div>'
        "<h3>Macro-F1 after equal-budget label restoration</h3></div></div>"
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


def _render_h2_chart(h2: dict[str, Any]) -> str:
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
        "<h3>Observed average-precision ranges</h3></div></div>"
        '<div class="range-axis"><span>0 AP</span>'
        "<span>Range across saved subgroup estimates</span><span>1 AP</span></div>"
        + "".join(rows)
        + f'<p class="chart-note"><strong>{h2["reported_count"]:,}</strong> reportable '
        + f"estimates from <strong>{h2['row_count']:,}</strong> saved rows. These ranges "
        + "are descriptive. They are not an omnibus test or a biological ranking.</p></div>"
    )


def _render_hypothesis_ledger(
    hypotheses: dict[str, dict[str, Any]], h2: dict[str, Any], h4: dict[str, Any]
) -> str:
    """Render every preregistered question as an accessible story slide."""

    h1, h3, h5, h6, h7 = (
        hypotheses["h1"],
        hypotheses["h3"],
        hypotheses["h5"],
        hypotheses["h6"],
        hypotheses["h7"],
    )
    h1_range = _require_interval(h1["point_difference_range"], role="H1 AP range")
    h5_range = _require_interval(h5["point_difference_range"], role="H5 AP range")
    h7_range = _require_interval(h7["point_difference_range"], role="H7 AP range")
    entries = (
        (
            "H1",
            "Can the audit score find injected label changes earlier than random review?",
            "Yes, within this controlled benchmark. Self-confidence ranking beat "
            f"random review in all {h1['reported_count']} registered comparisons: "
            f"average-precision gains ranged from {_format_metric(h1_range[0])} to "
            f"{_format_metric(h1_range[1])}, and every saved 95% bootstrap interval "
            "was above zero. This means injected changes were prioritised more "
            "efficiently; it does not prove that a natural annotation is wrong. "
            "Byte-identical instance-dependent seeds are not independent replications.",
        ),
        (
            "H2",
            "Where did ranking performance vary?",
            f"Across {h2['reported_count']:,} reportable subgroup estimates, ranking "
            "performance varied with nucleus class, tissue, corruption mechanism and "
            "corruption rate. The practical lesson is that audit difficulty depends on "
            "context. These are descriptive summaries only: the study did not register "
            "an omnibus test and cannot interpret the variation as a causal biological "
            "effect.",
        ),
        (
            "H3",
            "Which injected changes were harder to retrieve?",
            "Confusion-targeted and instance-dependent changes were harder to rank "
            "than symmetric changes, meaning their average precision was lower. All "
            f"{h3['reported_count']} registered point differences favoured symmetric "
            "corruption; five saved 95% intervals were above zero and one crossed zero. "
            "This describes the controlled benchmark and does not establish that one "
            "natural error type is universally harder.",
        ),
        (
            "H4",
            "Did a better review queue improve the later classifier?",
            "No. The main result above reports the complete comparison. At the same "
            f"5% review budget, audit-guided restoration was lower by "
            f"{_format_metric(h4['point_difference'])} macro-F1, so better retrieval "
            "did not translate into better downstream classification here.",
        ),
        (
            "H5",
            "Did combining two ranking signals help?",
            "Yes, for review prioritisation. The equal-weight combination of "
            "self-confidence and fold-safe nearest-neighbour disagreement outperformed "
            f"self-confidence alone in all {h5['reported_count']} registered comparisons. "
            f"Average-precision gains ranged from {_format_metric(h5_range[0])} to "
            f"{_format_metric(h5_range[1])}, and every saved 95% bootstrap interval "
            "was above zero. This supports the fixed hybrid within this benchmark; it "
            "does not authorise automatic correction or diagnosis.",
        ),
        (
            "H6",
            "Could a pathology-specific encoder be evaluated fairly?",
            "No result is available. None of the candidate pathology encoders passed "
            "every frozen access, licence, reproducibility, hardware and smoke-test "
            f"gate, so all {h6['comparison_count']} registered H6 entries remain "
            "unavailable. They are not zero or negative results, and no substitute was "
            "selected after the other outcomes became visible.",
        ),
        (
            "H7",
            "Did highlighting the target nucleus add useful signal?",
            "No clear benefit was detected. Across three seeds, the average-precision "
            f"differences ranged from {_format_metric(h7_range[0])} to "
            f"{_format_metric(h7_range[1])}; all {h7['interval_crosses_zero_count']} "
            "saved 95% intervals crossed zero, and no Holm-adjusted one-sided p-value "
            "was below 0.05. In this benchmark, explicit highlighting did not improve "
            "ranking over the context representation.",
        ),
    )
    slides: list[str] = []
    for _, title, detail in entries:
        slides.append(
            '<article class="hypothesis-row" data-learned-slide>'
            + '<h3 class="hypothesis-title">'
            + html.escape(title)
            + '</h3><p class="learned-answer">'
            + html.escape(detail)
            + "</p></article>"
        )
    return "".join(slides)


_MVP_CSS = (Path(__file__).resolve().parent / "assets" / "mvp_presentation.css").read_text(
    encoding="utf-8"
)


_MVP_SCRIPT = r"""
(() => {
  const root = document.documentElement;
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const gsapEngine = window.gsap;
  const scrollEngine = window.ScrollTrigger;
  const motionAvailable = Boolean(gsapEngine && scrollEngine && !reduced);
  if (gsapEngine && scrollEngine) gsapEngine.registerPlugin(scrollEngine);

  const article = document.getElementById('main');
  const articleOrder = [
    'overview', 'study-facts', 'research-question', 'method', 'reading', 'results',
    'benchmarks', 'subgroups', 'seed-audit', 'evidence', 'external-validation',
    'new-data-test', 'quality', 'integrity', 'interpretation', 'current-stage', 'use', 'author',
  ];
  if (article) {
    articleOrder.forEach(id => {
      const section = document.getElementById(id);
      if (section) article.appendChild(section);
    });
  }

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
      if (brandLabel) {
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
      }
    } else {
      header.style.transform = visible ? 'translateY(0)' : 'translateY(-112%)';
      if (brandLabel) {
        brandLabel.style.opacity = visible ? '1' : '0';
        brandLabel.style.transform = visible ? 'translateX(0)' : 'translateX(-14px)';
      }
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

  const setMenuOpen = open => {
    navLinks.classList.toggle('is-open', open);
    menuButton.setAttribute('aria-expanded', String(open));
    menuButton.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
  };
  menuButton.addEventListener('click', () => {
    setMenuOpen(!navLinks.classList.contains('is-open'));
    setHeaderVisible(true);
  });
  navLinks.addEventListener('click', event => {
    if (event.target instanceof HTMLAnchorElement) {
      setMenuOpen(false);
    }
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && navLinks.classList.contains('is-open')) {
      setMenuOpen(false);
      menuButton.focus();
    }
  });

  document.querySelectorAll('[data-learned-slide]').forEach(slide => {
    slide.removeAttribute('aria-hidden');
  });

  if (motionAvailable) {
    root.classList.add('motion-enhanced', 'gsap-ready');
    const revealItems = [...document.querySelectorAll('.reveal')];
    gsapEngine.set(revealItems, {autoAlpha: 0, y: 18});
    revealItems.forEach(item => {
      gsapEngine.to(item, {
        autoAlpha: 1,
        y: 0,
        duration: .55,
        ease: 'power2.out',
        scrollTrigger: {trigger: item, start: 'top 88%', once: true},
        onComplete: () => item.classList.add('is-shown'),
      });
    });
    document.querySelectorAll('[data-learned-slide]').forEach(row => {
      const content = row.querySelectorAll('.hypothesis-title, .learned-answer');
      gsapEngine.set(content, {autoAlpha: 0, y: 20});
      gsapEngine.to(content, {
        autoAlpha: 1,
        y: 0,
        duration: .62,
        stagger: .08,
        ease: 'power2.out',
        scrollTrigger: {trigger: row, start: 'top 84%', once: true},
      });
    });
    gsapEngine.from('.hero-animate', {
      autoAlpha: 0,
      y: 18,
      duration: .7,
      stagger: .08,
      ease: 'power2.out',
      delay: .08,
    });
    document.querySelectorAll('.nav-links a[href^="#"]').forEach(link => {
      const id = link.getAttribute('href').slice(1);
      const target = document.getElementById(id);
      if (!target) return;
      scrollEngine.create({
        trigger: target,
        start: 'top 35%',
        end: 'bottom 35%',
        onToggle: self => link.classList.toggle('is-active', self.isActive),
      });
    });
  }

  const filterTable = () => {
    const hypothesis = document.getElementById('filter-hypothesis');
    const status = document.getElementById('filter-status');
    const search = document.getElementById('filter-search');
    const count = document.getElementById('table-count');
    const rows = [...document.querySelectorAll('[data-comparison-row]')];
    if (!hypothesis || !status || !search || !count) return;
    const query = search.value.trim().toLowerCase();
    let visible = 0;
    rows.forEach(row => {
      const hypOk = hypothesis.value === 'ALL' || row.dataset.hypothesis === hypothesis.value;
      const statusOk = status.value === 'ALL' || row.dataset.status === status.value;
      const hay = row.textContent.toLowerCase();
      const searchOk = !query || hay.includes(query);
      const show = hypOk && statusOk && searchOk;
      row.hidden = !show;
      if (show) visible += 1;
    });
    count.textContent = `${visible} / ${rows.length} rows`;
  };
  ['filter-hypothesis', 'filter-status', 'filter-search'].forEach(id => {
    const control = document.getElementById(id);
    if (control) control.addEventListener('input', filterTable);
  });
})();
"""


def _render_readme(evidence: dict[str, Any]) -> str:
    primary = evidence["primary"]
    return f"""# AANCA presentation MVP

This five-file, read-only article package was generated from checksum-verified
PanNuke primary evidence plus the checked-in NuCLS, MoNuSAC and PUMA result
authorities. The accepted PanNuke run is `{primary["run_id"]}`.

From the repository root, verify the complete package and open it locally:

```powershell
python scripts/present_demo.py
```

The launcher uses only the Python standard library. It verifies every packaged file
before serving `127.0.0.1`; it never runs a model or changes data. For verification
without a browser or server:

```powershell
python scripts/present_demo.py --verify-only
```

With the project environment installed, `uv run histo-audit demo serve` and
`uv run histo-audit demo verify` provide the equivalent CLI workflow.

## Current scientific position

- Primary study: `PRIMARY_STUDY_COMPLETE`; its accepted PanNuke analysis remains
  permanently `amended_or_exploratory` and H4 was adverse.
- External evaluation: `EXTERNAL_VALIDATION_COMPLETE`. NuCLS natural multi-rater
  claims were not supported; MoNuSAC controlled retrieval passed but downstream and
  class-safety gates failed; the frozen PUMA controlled confirmation passed all seven
  internally pre-specified gates. Its protocol and result entered public history
  together, so the public repository does not independently prove the pre-outcome
  timing.
- Presentation: `DEMO_COMPLETE`.
- Confirmatory stage: not reached. `CONFIRMATORY_COMPLETE` is not claimed.
- Natural-data action: `retain_uncorrected`.

The positive PUMA result supports transfer under controlled label noise. It does not
show that AANCA detects pathologist errors, discovers biological truth, improves a
real laboratory workflow or is clinically useful. The software never modifies
source annotations automatically; it ranks potentially inconsistent annotations
for qualified expert review.

The 144/62 development/final partition is an AANCA-defined split of the 206 public
PUMA ROIs. It is not the official hidden PUMA challenge test set. The downstream
intervention was `flag_exclude`: the highest-ranked 5% of controlled training
instances were omitted from downstream training. They were not reviewed, corrected
or automatically relabelled by an expert.

The PUMA verifier is a project-coupled evidence-readback script that recomputes
metrics from saved predictions but does not retrain all 44 models. It is not
third-party validation.

## Package contents

- `index.html` — responsive English article, including the retained “What the study
  actually learned.” sequence;
- `evidence.json` — sourced primary, external, controlled-confirmation, stress,
  sensitivity, current-action and next-phase summaries;
- `pannuke_mask_qc_overlays.png` — deterministic source-ingestion QC preview;
- `README.md` — this handoff;
- `manifest.json` — SHA-256 allowlist binding every other package file.

The primary evidence retains all 36 registered H1/H3/H5/H6/H7 entries: 33 numeric
results and three explicitly unavailable H6 cells. Displayed intervals, adjusted
p-values, source hashes and external summaries are read from machine-readable
authorities rather than retyped into the page.

## Next phase

The provisional AANCA v2 research phase requires: a new blinded multi-rater natural
reference; nested group-safe measured-utility development; one prospectively frozen
policy; one-shot untouched patient/WSI confirmation; and a prospective multi-site
AANCA-versus-control workflow study. Promotion requires ranking, downstream
confidence intervals, every-class safety and workflow utility to pass together.

Source code, frozen protocols, configs, scoped verification scripts, evidence and the
complete limitation statement are at <https://github.com/Jaqwilk/AANCA>.

Author: Natan Smogór. Updated: 21 August 2026.
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
        "schema_version": 3,
        "policy": "aanca_presentation_current_evidence_readback_v3",
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
        manifest.get("schema_version") != 3
        or manifest.get("policy") != "aanca_presentation_current_evidence_readback_v3"
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
    external_completed = evidence.get("external_validation_completed")
    external = evidence.get("external_validation")
    controlled = evidence.get("controlled_external_benchmark")
    puma = evidence.get("new_source_confirmation")
    stress = evidence.get("realism_stress")
    sensitivity = evidence.get("audit_time_label_sensitivity")
    natural_action = evidence.get("natural_data_action")
    next_phase = evidence.get("next_phase")
    publication_limits = evidence.get("publication_limits")
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
    publication_limits_valid = (
        isinstance(publication_limits, dict)
        and publication_limits.get("puma_public_preoutcome_timestamp_available") is False
        and publication_limits.get("puma_first_public_combined_commit")
        == "c5bd44193b2abd67bc7e7f1bd9384aa87435d500"
        and publication_limits.get("puma_partition")
        == "aanca_defined_144_development_62_final_of_206_public_rois"
        and publication_limits.get("puma_official_hidden_challenge_test_used") is False
        and publication_limits.get("puma_downstream_intervention")
        == "flag_exclude_top_5_percent_training_rows"
        and publication_limits.get("puma_flagged_rows_expert_reviewed_or_relabelled") is False
        and publication_limits.get("puma_verifier_retrains_models_from_images") is False
        and publication_limits.get("puma_verifier_scope")
        == "saved_evidence_readback_with_maintained_helpers"
        and publication_limits.get("puma_verifier_is_third_party_validation") is False
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
        or not publication_limits_valid
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
        "scientific_status": "EXTERNAL_VALIDATION_COMPLETE",
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
    release_evidence = _load_release_evidence(root)
    evidence = {
        "schema_version": 3,
        "presentation_status": "DEMO_COMPLETE",
        "scientific_status": "EXTERNAL_VALIDATION_COMPLETE",
        "primary_study_status": "PRIMARY_STUDY_COMPLETE",
        "analysis_disposition": "amended_or_exploratory",
        "confirmatory_completed": False,
        "external_validation_completed": True,
        "automatic_annotation_changes": False,
        "diagnostic_system": False,
        "primary": primary,
        "pannuke_qc": qc,
        "external_validation": release_evidence["external_validation"],
        "controlled_external_benchmark": release_evidence["controlled_external_benchmark"],
        "new_source_confirmation": release_evidence["new_source_confirmation"],
        "realism_stress": release_evidence["realism_stress"],
        "audit_time_label_sensitivity": release_evidence["audit_time_label_sensitivity"],
        "natural_data_action": release_evidence["natural_data_action"],
        "publication_limits": {
            "puma_public_preoutcome_timestamp_available": False,
            "puma_first_public_combined_commit": ("c5bd44193b2abd67bc7e7f1bd9384aa87435d500"),
            "puma_partition": "aanca_defined_144_development_62_final_of_206_public_rois",
            "puma_official_hidden_challenge_test_used": False,
            "puma_downstream_intervention": "flag_exclude_top_5_percent_training_rows",
            "puma_flagged_rows_expert_reviewed_or_relabelled": False,
            "puma_verifier_retrains_models_from_images": False,
            "puma_verifier_scope": "saved_evidence_readback_with_maintained_helpers",
            "puma_verifier_is_third_party_validation": False,
        },
        "next_phase": {
            "stage": "INITIALISED",
            "working_name": "AANCA v2 research phase",
            "natural_auto_change_allowed": False,
            "required_gates": [
                "blinded multi-rater natural-case reference with ambiguity and abstention",
                "nested group-safe measured-utility queue development",
                "one policy frozen before any new confirmation outcomes",
                "one-shot untouched patient or WSI external confirmation",
                "prospective multi-site AANCA-versus-control workflow evaluation",
            ],
            "promotion_rule": (
                "promote only if ranking, downstream confidence intervals, class safety "
                "and prospective workflow utility all pass"
            ),
        },
        "source_files": {
            "primary_run": _project_relative_path(root, run_path, role="primary run"),
            "qc_bundle": _project_relative_path(root, qc_path, role="QC bundle"),
            "qc_bundle_manifest_sha256": sha256_file(qc_path / "artifact_manifest.json"),
            "qc_overlay_sha256": qc_manifest["overlay_sha256"],
            "released_evidence": release_evidence["source_records"],
        },
    }

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        atomic_write_json(staging / "evidence.json", evidence)
        atomic_write_text(
            staging / "index.html",
            _render_html(evidence, evidence_sha256=sha256_file(staging / "evidence.json")),
        )
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
