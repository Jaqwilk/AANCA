"""Post-outcome method development on the preserved NuCLS evidence.

This module deliberately reads the immutable external-validation package instead of
overwriting it.  Its results are exploratory and cannot change the prospectively
frozen study decision.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from histo_audit.evaluation.retraining_guard import evaluate_retraining_guard
from histo_audit.statistics.review import (
    average_precision,
    budget_count,
    draw_group_bootstrap_indices,
    rank_indices,
)
from histo_audit.utils.run_tracking import sha256_file

from .nucls import CLASS_ORDER, _array_sha256, _nucls_audit_risk


def _verify_artifact_root(root: Path) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"NuCLS artifact manifest is absent: {manifest_path}")
    artifact_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = artifact_manifest.get("files")
    if not isinstance(records, Mapping):
        raise ValueError("NuCLS artifact manifest lacks file records")
    verified: dict[str, Any] = {}
    for name, raw_record in records.items():
        if not isinstance(name, str) or not isinstance(raw_record, Mapping):
            raise ValueError("NuCLS artifact manifest contains an invalid record")
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"NuCLS artifact is absent: {path}")
        expected_bytes = int(raw_record["bytes"])
        expected_sha256 = str(raw_record["sha256"])
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
            raise ValueError(f"NuCLS artifact identity mismatch: {path}")
        verified[name] = {"bytes": actual_bytes, "sha256": actual_sha256}
    return verified


def _load_saved_evidence(
    root: Path,
) -> tuple[pd.DataFrame, dict[str, NDArray[np.generic]], dict[str, Any]]:
    verified = _verify_artifact_root(root)
    manifest = pd.read_csv(root / "canonical_manifest.csv")
    with np.load(root / "numeric_evidence.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    expected_sample_ids = manifest["sample_id"].astype(str).to_numpy(dtype=np.str_)
    expected_groups = manifest["group_id"].astype(str).to_numpy(dtype=np.str_)
    expected_observed = manifest["observed_label"].to_numpy(dtype=np.int64)
    expected_reference = manifest["reference_label"].to_numpy(dtype=np.int64)
    checks = (
        ("sample_ids", expected_sample_ids),
        ("group_ids", expected_groups),
        ("observed_labels", expected_observed),
        ("reference_labels", expected_reference),
        ("natural_disagreement", expected_observed != expected_reference),
    )
    for name, expected in checks:
        if name not in arrays or not np.array_equal(arrays[name], expected):
            raise ValueError(f"saved NuCLS {name} does not match the canonical manifest")
    saved_self_confidence = (
        1.0 - arrays["oof_probabilities"][np.arange(len(expected_observed)), expected_observed]
    )
    if not np.array_equal(arrays["risk_scores"], saved_self_confidence):
        raise ValueError("saved NuCLS risk is not the frozen self-confidence score")
    return manifest, arrays, verified


def _interval(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        raise RuntimeError("exploratory bootstrap produced no finite values")
    return [float(value) for value in np.quantile(array, [0.025, 0.975])]


def _ranking_summary(
    events: NDArray[np.bool_],
    risk: NDArray[np.float64],
    sample_ids: Sequence[str],
    draws: Sequence[NDArray[np.int64]],
    *,
    primary_budget: float,
) -> dict[str, Any]:
    prevalence = float(events.mean())
    ap = average_precision(events, risk)
    if ap is None:
        raise RuntimeError("NuCLS development evidence contains no disagreement events")
    count = budget_count(len(events), primary_budget)
    selected = rank_indices(risk, tie_break_ids=sample_ids)[:count]
    found = int(events[selected].sum())
    precision = float(found / count)
    ap_differences: list[float] = []
    precision_differences: list[float] = []
    for indices in draws:
        sampled_events = events[indices]
        sampled_prevalence = float(sampled_events.mean())
        sampled_ap = average_precision(sampled_events, risk[indices])
        if sampled_ap is not None:
            ap_differences.append(float(sampled_ap - sampled_prevalence))
        sampled_count = budget_count(len(indices), primary_budget)
        sampled_selected = rank_indices(risk[indices])[:sampled_count]
        precision_differences.append(
            float(sampled_events[sampled_selected].mean() - sampled_prevalence)
        )
    ap_interval = _interval(ap_differences)
    precision_interval = _interval(precision_differences)
    return {
        "average_precision": float(ap),
        "ap_minus_prevalence": float(ap - prevalence),
        "ap_minus_prevalence_interval_95": ap_interval,
        "review_budget": primary_budget,
        "reviewed_count": count,
        "disagreements_found": found,
        "precision": precision,
        "precision_minus_prevalence": float(precision - prevalence),
        "precision_minus_prevalence_interval_95": precision_interval,
        "strict_success_conditions_met": ap_interval[0] > 0.0 and precision_interval[0] > 0.0,
        "risk_sha256": _array_sha256(risk),
    }


def analyze_saved_nucls_subset(
    artifact_root: str | Path,
    frozen_config: Mapping[str, Any],
    development_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate declared score candidates and fail-closed guards on one saved subset."""

    root = Path(artifact_root).resolve()
    manifest, arrays, verified = _load_saved_evidence(root)
    observed = np.asarray(arrays["observed_labels"], dtype=np.int64)
    reference = np.asarray(arrays["reference_labels"], dtype=np.int64)
    events = np.asarray(arrays["natural_disagreement"], dtype=bool)
    groups = manifest["group_id"].astype(str).tolist()
    sample_ids = manifest["sample_id"].astype(str).tolist()
    ranking = frozen_config["ranking"]
    candidates = development_config["ranking"]["candidates"]
    neighbour_k = int(development_config["ranking"]["neighbour_k"])
    neighbour_metric = str(development_config["ranking"]["neighbour_metric"])
    hybrid_weights = tuple(
        float(value) for value in development_config["ranking"]["hybrid_weights"]
    )
    draws = draw_group_bootstrap_indices(
        groups,
        n_iterations=int(ranking["bootstrap_iterations"]),
        seed=int(ranking["bootstrap_seed"]),
    )
    strategy_results: dict[str, Any] = {}
    for method in candidates:
        score = _nucls_audit_risk(
            arrays["embeddings"],
            observed,
            arrays["oof_probabilities"],
            groups,
            sample_ids,
            method=str(method),
            n_splits=int(ranking["folds"]),
            split_seed=int(ranking["split_seed"]),
            neighbour_k=neighbour_k,
            neighbour_metric=neighbour_metric,
            hybrid_weights=hybrid_weights,
        )
        strategy_results[score.method] = {
            "strategy": score.as_dict(),
            **_ranking_summary(
                events,
                score.risk_scores,
                sample_ids,
                draws,
                primary_budget=float(ranking["primary_budget"]),
            ),
        }

    guard_config = development_config["retraining_guard"]
    common_guard = {
        "class_order": tuple(range(len(CLASS_ORDER))),
        "evidence_role": str(guard_config["evidence_role"]),
        "n_iterations": int(guard_config["bootstrap_iterations"]),
        "seed": int(guard_config["bootstrap_seed"]),
        "minimum_effect": float(guard_config["minimum_macro_f1_effect"]),
    }
    frozen_guided_guard = evaluate_retraining_guard(
        reference,
        arrays["downstream_uncorrected_probabilities"],
        arrays["downstream_guided_probabilities"],
        groups,
        **common_guard,
    )
    full_reference_guard = evaluate_retraining_guard(
        reference,
        arrays["downstream_uncorrected_probabilities"],
        arrays["downstream_reference_ceiling_probabilities"],
        groups,
        **common_guard,
    )
    return {
        "artifact_root": root.as_posix(),
        "artifact_identities": verified,
        "sample_count": len(manifest),
        "patient_group_count": int(manifest["group_id"].nunique()),
        "natural_disagreement_count": int(events.sum()),
        "natural_disagreement_prevalence": float(events.mean()),
        "ranking_candidates": strategy_results,
        "retraining_guard": {
            "frozen_audit_guided_candidate": frozen_guided_guard.as_dict(),
            "full_consensus_label_candidate": full_reference_guard.as_dict(),
            "source_annotations_modified": False,
        },
    }


def analyze_current_aanca(
    repository_root: str | Path,
    frozen_config: Mapping[str, Any],
    development_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Analyze both frozen subsets without creating a replacement project or study."""

    root = Path(repository_root).resolve()
    roots = development_config["evidence_roots"]
    subsets = {
        name: analyze_saved_nucls_subset(root / path, frozen_config, development_config)
        for name, path in roots.items()
    }
    promoted_method = str(development_config["promotion_rule"]["candidate_method"])
    promoted_canonical = (
        "nearest_neighbour_disagreement"
        if promoted_method in {"nearest_neighbour_disagreement", "fold_safe_neighbour_disagreement"}
        else promoted_method
    )
    passes_all = all(
        bool(subset["ranking_candidates"][promoted_canonical]["strict_success_conditions_met"])
        for subset in subsets.values()
    )
    return {
        "schema_version": 1,
        "project": "AANCA",
        "change_type": "incremental_current_model_improvement",
        "new_version_or_replacement_project": False,
        "study_id": str(development_config["study_id"]),
        "parent_frozen_study_id": str(development_config["parent_frozen_study_id"]),
        "analysis_disposition": "post_outcome_exploratory",
        "outcomes_inspected_before_this_analysis": True,
        "frozen_external_result_changed": False,
        "subsets": subsets,
        "candidate_promotion": {
            "method": promoted_canonical,
            "requires_strict_success_in_every_declared_subset": True,
            "promoted_to_new_default": passes_all,
            "decision": (
                "eligible_for_a_fresh_prospective freeze"
                if passes_all
                else "not promoted; retain the existing default pending fresh independent data"
            ),
        },
        "runtime_change": {
            "additional_group_safe_ranking_available": True,
            "retraining_application_is_fail_closed": True,
            "automatic_source_annotation_modification": False,
        },
        "claim_boundary": {
            "natural_pathology_error_detection_proven": False,
            "pathologist_error_detection_proven": False,
            "clinical_utility_proven": False,
            "prospective_workflow_benefit_proven": False,
        },
    }


def render_current_aanca_report(result: Mapping[str, Any]) -> str:
    """Render the compact human-readable companion to the development JSON."""

    lines = [
        "# Incremental improvement of the current AANCA model",
        "",
        "**Disposition:** post-outcome exploratory method development  ",
        "**Frozen NuCLS decision:** unchanged  ",
        "**Replacement project or v2:** no",
        "",
        "The existing AANCA pipeline now supports a fold-safe neighbour score and a fixed",
        "self-confidence/neighbour hybrid. The analysis below reuses only the preserved",
        "NuCLS manifests, embeddings and OOF probabilities. It does not turn consensus",
        "disagreement into biological truth and cannot revise the frozen negative study.",
        "",
        "## Ranking candidates",
        "",
        "| Subset | Method | AP | Disagreements at 5% | AP-difference CI | Precision-difference CI | Strict result |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for subset_name, subset in result["subsets"].items():
        for method, candidate in subset["ranking_candidates"].items():
            ap_interval = candidate["ap_minus_prevalence_interval_95"]
            precision_interval = candidate["precision_minus_prevalence_interval_95"]
            lines.append(
                "| "
                f"{subset_name} | {method} | {candidate['average_precision']:.6f} | "
                f"{candidate['disagreements_found']}/{candidate['reviewed_count']} | "
                f"[{ap_interval[0]:.6f}, {ap_interval[1]:.6f}] | "
                f"[{precision_interval[0]:.6f}, {precision_interval[1]:.6f}] | "
                f"{'passed' if candidate['strict_success_conditions_met'] else 'failed'} |"
            )
    promotion = result["candidate_promotion"]
    lines.extend(
        [
            "",
            "The neighbour candidate passes both strict conditions in the primary Unbiased",
            "Control subset but not in the declared Evaluation sensitivity subset. It is",
            f"therefore **{promotion['decision']}**. This avoids post-hoc replacement of the",
            "frozen self-confidence result with whichever candidate looks best on one table.",
            "",
            "## Retraining safeguard",
            "",
            "| Subset | Candidate | Macro-F1 difference | 95% group interval | Action |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for subset_name, subset in result["subsets"].items():
        for label, guard in subset["retraining_guard"].items():
            if not isinstance(guard, Mapping):
                continue
            interval = guard["interval_95"]
            interval_text = (
                f"[{interval[0]:.6f}, {interval[1]:.6f}]" if interval is not None else "unavailable"
            )
            lines.append(
                f"| {subset_name} | {label} | "
                f"{guard['candidate_minus_uncorrected_macro_f1']:+.6f} | "
                f"{interval_text} | {guard['action']} |"
            )
    lines.extend(
        [
            "",
            "Both saved correction candidates are rejected where the lower confidence bound",
            "does not exceed zero. The runtime policy therefore retains the uncorrected model",
            "instead of applying a correction that has not demonstrated non-degradation.",
            "This is a safety improvement, not evidence that AANCA improves real deployment.",
            "",
            "## What remains genuinely open",
            "",
            "Natural-error detection still requires blinded adjudication by newly recruited",
            "qualified pathologists. Real-world benefit still requires a prospective, multi-site",
            "comparison of work with and without AANCA. Those outcomes cannot be manufactured",
            "from the existing retrospective evidence.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "analyze_current_aanca",
    "analyze_saved_nucls_subset",
    "render_current_aanca_report",
]
