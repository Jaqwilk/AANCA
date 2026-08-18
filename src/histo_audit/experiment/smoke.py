"""Deterministic CPU synthetic end-to-end scientific smoke gate."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from histo_audit.auditing.neighbours import fold_safe_neighbour_disagreement
from histo_audit.auditing.scores import cleanlab_scores, fixed_hybrid_score, score_annotations
from histo_audit.corruption.controlled import (
    FeatureIndependenceEvidence,
    FeatureSpaceEvidence,
    apply_controlled_corruption,
    canonical_sha256,
    semantic_sha256,
)
from histo_audit.cross_validation.oof import grouped_oof_logistic
from histo_audit.data.splitting import (
    make_fractional_outer_audit_split,
    make_outer_audit_split,
)
from histo_audit.data.synthetic import (
    CLASS_NAMES,
    SYNTHETIC_GENERATOR_SCHEMA_VERSION,
    generate_synthetic_dataset,
    synthetic_generator_code_sha256,
)
from histo_audit.data.targets import extract_target_crop, highlight_target
from histo_audit.evaluation.restoration import evaluate_downstream_restoration
from histo_audit.experiment.evidence import (
    write_neighbour_evidence,
    write_restoration_evidence,
)
from histo_audit.statistics.review import (
    PairedBootstrapResult,
    RandomReviewSummary,
    ReviewBudgetResult,
    SubgroupRankingResult,
    binary_auroc,
    draw_group_bootstrap_indices,
    evaluate_review_budget,
    paired_group_bootstrap,
    random_review_baseline,
    rank_indices,
    subgroup_average_precision,
)

_ZERO_CORRUPTION_REASON = (
    "No injected corruptions are present; controlled detection AP, recall, and lift are undefined."
)
_PAIRED_COMPARATOR = "self_confidence"
_PAIRED_METHOD_ORDER = (
    "fixed_hybrid",
    "negative_log_likelihood",
    "prediction_margin",
    "predictive_entropy",
    "neighbour_disagreement",
    "cleanlab",
)


@dataclass(frozen=True, slots=True)
class SyntheticSmokeResult:
    """Core smoke output designed for defensive CLI/report integration."""

    success: bool
    status: str
    run_id: str
    run_dir: Path | None
    metrics: Mapping[str, Any]
    metrics_path: Path | None
    predictions_path: Path | None
    rankings_path: Path | None
    corruption_manifest_path: Path | None
    oof_provenance_path: Path | None
    representation_example_path: Path | None
    neighbour_evidence_path: Path | None
    restoration_evidence_path: Path | None
    bootstrap_evidence_path: Path | None
    dataset_evidence_path: Path | None
    source_manifest_path: Path | None
    source_manifest_csv_path: Path | None
    report_inputs: Mapping[str, Any]
    report_inputs_path: Path | None


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=_json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a compressed NumPy archive without exposing a partial artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _csv_cell(value: object) -> object:
    """Return a deterministic, losslessly parseable CSV cell for manifest values."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (tuple, list, dict)):
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return value


def _atomic_csv_rows(
    path: Path, rows: tuple[Mapping[str, object], ...], fieldnames: tuple[str, ...]
) -> None:
    """Write manifest rows atomically with stable columns and line endings."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: _csv_cell(row[name]) for name in fieldnames})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved(config: Mapping[str, object] | None) -> dict[str, Any]:
    values: dict[str, Any] = {
        "dataset_seed": 2027,
        "n_groups": 30,
        "instances_per_group": 7,
        "patch_size": 64,
        "final_test_fold": 2,
        "final_test_fraction_groups": None,
        "reference_validation_fraction": 0.10,
        "outer_split_seed": 41,
        "corruption_rate": 0.10,
        "corruption_mechanism": "symmetric_random_corruption",
        "corruption_seed": 17,
        "primary_representation": "target_colour_statistics",
        "auditor_representation": "target_colour_statistics",
        "oof_splits": 5,
        "split_seed": 23,
        "model_seed": 29,
        "review_budget": 0.05,
        "review_budgets": (0.01, 0.05, 0.10, 0.20),
        "random_review_repeats": 100,
        "random_review_seed": 101,
        "downstream_random_repeats": 10,
        "bootstrap_iterations": 200,
        "bootstrap_seed": 211,
        "neighbour_k": 7,
        "subgroup_min_samples": 100,
        "subgroup_min_corruptions": 10,
    }
    if config:

        def section(name: str) -> Mapping[str, object]:
            candidate = config.get(name, {})
            return candidate if isinstance(candidate, Mapping) else {}

        seeds = section("seed")
        data = section("data")
        corruption = section("corruption")
        representation = section("representation")
        model = section("model")
        audit = section("audit")
        evaluation = section("evaluation")
        restoration = section("restoration")
        nested: dict[str, object] = {}
        mappings = (
            (seeds, "dataset", "dataset_seed"),
            (seeds, "split", "outer_split_seed"),
            (seeds, "split", "split_seed"),
            (seeds, "model", "model_seed"),
            (seeds, "corruption", "corruption_seed"),
            (seeds, "random_review", "random_review_seed"),
            (seeds, "bootstrap", "bootstrap_seed"),
            (data, "groups", "n_groups"),
            (data, "samples_per_group", "instances_per_group"),
            (data, "image_size", "patch_size"),
            (data, "final_test_fraction_groups", "final_test_fraction_groups"),
            (
                data,
                "reference_validation_fraction_groups",
                "reference_validation_fraction",
            ),
            (corruption, "rate", "corruption_rate"),
            (corruption, "mechanism", "corruption_mechanism"),
            (representation, "primary", "primary_representation"),
            (representation, "auditor", "auditor_representation"),
            (model, "oof_splits", "oof_splits"),
            (audit, "nearest_neighbours", "neighbour_k"),
            (evaluation, "review_budgets", "review_budgets"),
            (evaluation, "random_review_repeats", "random_review_repeats"),
            (evaluation, "random_review_repeats", "downstream_random_repeats"),
            (evaluation, "bootstrap_iterations", "bootstrap_iterations"),
            (evaluation, "subgroup_min_samples", "subgroup_min_samples"),
            (
                evaluation,
                "subgroup_min_corruptions",
                "subgroup_min_corruptions",
            ),
            (restoration, "review_budget", "review_budget"),
        )
        for source, source_key, target_key in mappings:
            if source_key in source:
                nested[target_key] = source[source_key]
        values.update(nested)
        # Already-flat values are supported and take precedence over nested values.
        values.update({key: value for key, value in config.items() if key in values})
        classes = data.get("classes")
        if classes is not None and int(str(classes)) != len(CLASS_NAMES):
            raise ValueError("synthetic smoke requires exactly five classes")
    supported_representation = "target_colour_statistics"
    for key in ("primary_representation", "auditor_representation"):
        if str(values[key]) != supported_representation:
            raise ValueError(
                "synthetic smoke currently supports only "
                f"{supported_representation!r}; {key}={values[key]!r}"
            )
    return values


def _not_applicable(reason: str) -> dict[str, object]:
    return {"status": "not_applicable", "value": None, "reason": reason}


def _subgroup_dict(result: SubgroupRankingResult) -> dict[str, Any]:
    payload = asdict(result)
    if payload.get("average_precision") is None:
        payload["average_precision"] = _not_applicable(
            str(payload.get("reason") or "Subgroup AP is not applicable.")
        )
    return payload


def _review_dict(result: ReviewBudgetResult) -> dict[str, Any]:
    values: dict[str, Any] = {
        field: getattr(result, field)
        for field in result.__dataclass_fields__
        if field != "reviewed_indices"
    }
    values["false_alert_count"] = result.reviewed_count - result.injected_reviewed
    if result.injected_total == 0:
        values["status"] = "not_applicable"
        values["reason"] = _ZERO_CORRUPTION_REASON
        for field in (
            "recall",
            "expected_random_recall",
            "lift_over_random",
            "average_precision",
        ):
            values[field] = _not_applicable(_ZERO_CORRUPTION_REASON)
    return values


def _score_distribution(scores: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(scores, dtype=np.float64)
    quantiles = np.quantile(values, (0.05, 0.25, 0.5, 0.75, 0.95))
    return {
        "count": len(values),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
    }


def _random_review_dict(summary: RandomReviewSummary, *, injected_total: int) -> dict[str, Any]:
    false_alert_counts = summary.reviewed_count - summary.injected_reviewed
    payload: dict[str, Any] = {
        "budget_fraction": summary.budget_fraction,
        "reviewed_count": summary.reviewed_count,
        "repeats": len(summary.seeds),
        "seeds": summary.seeds,
        "mean_precision": summary.mean_precision,
        "mean_recall": summary.mean_recall,
        "recall_interval_95": summary.recall_interval_95,
        "mean_false_alert_count": float(false_alert_counts.mean()),
        "false_alert_count_interval_95": (
            float(np.quantile(false_alert_counts, 0.025)),
            float(np.quantile(false_alert_counts, 0.975)),
        ),
    }
    if injected_total == 0:
        payload.update(status="not_applicable", reason=_ZERO_CORRUPTION_REASON)
        payload["mean_recall"] = _not_applicable(_ZERO_CORRUPTION_REASON)
        payload["recall_interval_95"] = _not_applicable(_ZERO_CORRUPTION_REASON)
    return payload


def _bootstrap_dict(result: PairedBootstrapResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "iterations": result.requested_iterations,
        "valid_iterations": result.valid_iterations,
        "mean_difference": result.mean_difference,
        "interval_95": result.interval_95,
        "probability_positive": result.probability_positive,
    }
    if result.valid_iterations == 0:
        payload.update(status="not_applicable", reason=_ZERO_CORRUPTION_REASON)
        for field in ("mean_difference", "interval_95", "probability_positive"):
            payload[field] = _not_applicable(_ZERO_CORRUPTION_REASON)
    return payload


def _empty_bootstrap_result(*, iterations: int, seed: int) -> PairedBootstrapResult:
    empty = np.empty(0, dtype=np.float64)
    return PairedBootstrapResult(
        metric_name="average_precision",
        metric_a=empty,
        metric_b=empty,
        differences=empty,
        mean_difference=None,
        interval_95=None,
        probability_positive=None,
        requested_iterations=iterations,
        valid_iterations=0,
        seed=seed,
    )


def _paired_comparison_dict(
    method: str,
    result: PairedBootstrapResult,
) -> dict[str, Any]:
    return {
        "method": method,
        "comparator": _PAIRED_COMPARATOR,
        "metric": result.metric_name,
        "difference_direction": "selected_method_minus_comparator",
        **_bootstrap_dict(result),
    }


def _make_run_dir(project_root: Path, config: Mapping[str, Any]) -> tuple[str, Path]:
    serialised = json.dumps(config, sort_keys=True, default=_json_default, separators=(",", ":"))
    short_hash = hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:10]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base_id = f"synthetic_smoke_{timestamp}_{short_hash}"
    base_dir = project_root / "artifacts" / "synthetic_core"
    run_id = base_id
    run_dir = base_dir / run_id
    suffix = 1
    while run_dir.exists():
        run_id = f"{base_id}_{suffix:02d}"
        run_dir = base_dir / run_id
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def run_synthetic_smoke(
    *,
    project_root: Path | None = None,
    config: Mapping[str, object] | None = None,
) -> SyntheticSmokeResult:
    """Run generation through restoration using only grouped CPU-safe evidence."""

    resolved = _resolved(config)
    dataset = generate_synthetic_dataset(
        n_groups=int(resolved["n_groups"]),
        instances_per_group=int(resolved["instances_per_group"]),
        patch_size=int(resolved["patch_size"]),
        seed=int(resolved["dataset_seed"]),
    )
    all_sample_ids = tuple(str(value) for value in dataset.sample_ids)
    all_group_ids = tuple(str(value) for value in dataset.group_ids)
    if resolved["final_test_fraction_groups"] is not None:
        split = make_fractional_outer_audit_split(
            all_group_ids,
            final_test_fraction=float(resolved["final_test_fraction_groups"]),
            reference_validation_fraction=float(resolved["reference_validation_fraction"]),
            seed=int(resolved["outer_split_seed"]),
        )
    else:
        split = make_outer_audit_split(
            dataset.official_folds,
            all_group_ids,
            final_test_fold=int(resolved["final_test_fold"]),
            reference_validation_fraction=float(resolved["reference_validation_fraction"]),
            seed=int(resolved["outer_split_seed"]),
        )
    audit = split.audit_indices
    final_test = split.final_test_indices
    audit_sample_ids = tuple(all_sample_ids[int(index)] for index in audit)
    audit_group_ids = tuple(all_group_ids[int(index)] for index in audit)
    final_test_group_ids = tuple(all_group_ids[int(index)] for index in final_test)
    final_test_sample_ids = tuple(all_sample_ids[int(index)] for index in final_test)
    mechanism = str(resolved["corruption_mechanism"])
    generator_feature_matrix = dataset.corruption_features[audit]
    auditor_feature_matrix = dataset.audit_features[audit]
    upstream_manifest_hash = canonical_sha256(
        {
            "sample_ids": audit_sample_ids,
            "group_ids": audit_group_ids,
            "pre_corruption_labels": dataset.pre_corruption_labels[audit].tolist(),
            "dataset_configuration_hash": dataset.records[0].configuration_hash,
        }
    )
    independence_evidence: FeatureIndependenceEvidence | None = None
    if mechanism in {"instance", "instance_dependent", "instance_dependent_corruption"}:
        fitted_data_hash = canonical_sha256(
            {"sample_ids": audit_sample_ids, "group_ids": audit_group_ids}
        )
        generator_evidence = FeatureSpaceEvidence.from_array(
            generator_feature_matrix,
            representation_name="target_morphology",
            family="engineered_target_morphology",
            implementation_hash=semantic_sha256(
                "histo_audit.data.targets.extract_morphology_features:v1"
            ),
            weights_hash=semantic_sha256("deterministic_unlearned_morphology:no_weights"),
            preprocessing_hash=semantic_sha256("binary_target_mask_geometry:v1"),
            fitted_data_hash=fitted_data_hash,
        )
        auditor_evidence = FeatureSpaceEvidence.from_array(
            auditor_feature_matrix,
            representation_name=str(resolved["auditor_representation"]),
            family="engineered_target_colour_only",
            implementation_hash=semantic_sha256(
                "histo_audit.data.targets.extract_target_colour_features:v1"
            ),
            weights_hash=semantic_sha256("deterministic_unlearned_colour_statistics:no_weights"),
            preprocessing_hash=semantic_sha256(
                "target_rgb_mean_std_context_contrast_without_geometry:v1"
            ),
            fitted_data_hash=fitted_data_hash,
        )
        independence_evidence = FeatureIndependenceEvidence.create(
            matrix_version="synthetic_feature_independence_matrix_v1",
            matrix_decision="verified_independent",
            matrix_reason=(
                "The corruption generator uses mask-only morphology while the auditor uses "
                "only RGB mean, standard deviation, and context contrast; no morphology "
                "features are reused, and artifacts/implementations are hash-bound."
            ),
            generator=generator_evidence,
            auditor=auditor_evidence,
        )
    corruption = apply_controlled_corruption(
        dataset.pre_corruption_labels[audit],
        sample_ids=audit_sample_ids,
        group_ids=audit_group_ids,
        rate=float(resolved["corruption_rate"]),
        mechanism=mechanism,
        seed=int(resolved["corruption_seed"]),
        n_classes=len(CLASS_NAMES),
        generator_features=(
            generator_feature_matrix
            if mechanism in {"instance", "instance_dependent", "instance_dependent_corruption"}
            else None
        ),
        generator_representation="target_morphology",
        auditor_representation=str(resolved["auditor_representation"]),
        independence_evidence=independence_evidence,
        upstream_manifest_hash=upstream_manifest_hash,
        dataset_seed=int(resolved["dataset_seed"]),
    )
    full_observed_label = dataset.pre_corruption_labels.copy()
    full_is_injected_corruption = np.zeros(len(dataset.records), dtype=bool)
    full_observed_label[audit] = corruption.observed_labels
    full_is_injected_corruption[audit] = corruption.is_injected_corruption
    oof = grouped_oof_logistic(
        dataset.audit_features[audit],
        corruption.observed_labels,
        audit_group_ids,
        final_reference_group_ids=split.final_test_groups,
        sample_ids=audit_sample_ids,
        n_splits=int(resolved["oof_splits"]),
        class_order=tuple(range(len(CLASS_NAMES))),
        split_seed=int(resolved["split_seed"]),
        model_seed=int(resolved["model_seed"]),
        representation=str(resolved["primary_representation"]),
    )
    self_confidence = score_annotations(
        corruption.observed_labels,
        oof.probabilities,
        method="self_confidence",
        class_order=oof.class_order,
    )
    negative_log_likelihood = score_annotations(
        corruption.observed_labels,
        oof.probabilities,
        method="negative_log_likelihood",
        class_order=oof.class_order,
    )
    margin = score_annotations(
        corruption.observed_labels,
        oof.probabilities,
        method="prediction_margin",
        class_order=oof.class_order,
    )
    entropy = score_annotations(
        corruption.observed_labels,
        oof.probabilities,
        method="predictive_entropy",
        class_order=oof.class_order,
    )
    neighbours = fold_safe_neighbour_disagreement(
        dataset.audit_features[audit],
        corruption.observed_labels,
        audit_group_ids,
        oof.fold_id,
        oof.training_groups_by_fold,
        sample_ids=audit_sample_ids,
        class_order=oof.class_order,
        k=int(resolved["neighbour_k"]),
    )
    cleanlab = cleanlab_scores(corruption.observed_labels, oof.probabilities)
    hybrid = fixed_hybrid_score(
        {
            "self_confidence": self_confidence,
            "prediction_margin": margin,
            "neighbour_disagreement": neighbours.risk_scores,
        }
    )
    risk_by_method = {
        "self_confidence": self_confidence,
        "negative_log_likelihood": negative_log_likelihood,
        "prediction_margin": margin,
        "predictive_entropy": entropy,
        "neighbour_disagreement": neighbours.risk_scores,
        "fixed_hybrid": hybrid,
    }
    if cleanlab.available:
        if cleanlab.risk_scores is None:
            raise RuntimeError("Cleanlab reported availability without risk scores")
        risk_by_method["cleanlab"] = cleanlab.risk_scores
    review_budgets = tuple(float(value) for value in resolved["review_budgets"])
    ranking_metrics: dict[str, Any] = {}
    pre_corruption_classes = tuple(
        CLASS_NAMES[int(value)] for value in corruption.pre_corruption_labels
    )
    tissue_types = tuple(dataset.records[int(index)].tissue_type for index in audit)
    for method_name, scores in risk_by_method.items():
        auroc = binary_auroc(corruption.is_injected_corruption, scores)
        ranking_metrics[method_name] = {
            "score_distribution": _score_distribution(scores),
            "auroc": (
                auroc
                if auroc is not None
                else _not_applicable(
                    "AUROC requires both injected-corruption and non-corruption examples."
                )
            ),
            "subgroups": {
                "pre_corruption_class": [
                    _subgroup_dict(value)
                    for value in subgroup_average_precision(
                        corruption.is_injected_corruption,
                        scores,
                        pre_corruption_classes,
                        min_samples=int(resolved["subgroup_min_samples"]),
                        min_injected_corruptions=int(resolved["subgroup_min_corruptions"]),
                    )
                ],
                "tissue_type": [
                    _subgroup_dict(value)
                    for value in subgroup_average_precision(
                        corruption.is_injected_corruption,
                        scores,
                        tissue_types,
                        min_samples=int(resolved["subgroup_min_samples"]),
                        min_injected_corruptions=int(resolved["subgroup_min_corruptions"]),
                    )
                ],
            },
            "review_budgets": {
                str(budget): _review_dict(
                    evaluate_review_budget(
                        corruption.is_injected_corruption,
                        scores,
                        budget=budget,
                        tie_break_ids=audit_sample_ids,
                    )
                )
                for budget in review_budgets
            },
        }
    primary_budget = float(resolved["review_budget"])
    random_summaries_by_budget = {
        str(budget): random_review_baseline(
            corruption.is_injected_corruption,
            budget=budget,
            repeats=int(resolved["random_review_repeats"]),
            seed=int(resolved["random_review_seed"]),
        )
        for budget in review_budgets
    }
    random_summary = random_summaries_by_budget.get(str(primary_budget))
    if random_summary is None:
        random_summary = random_review_baseline(
            corruption.is_injected_corruption,
            budget=primary_budget,
            repeats=int(resolved["random_review_repeats"]),
            seed=int(resolved["random_review_seed"]),
        )
    bootstrap_iterations = int(resolved["bootstrap_iterations"])
    bootstrap_seed = int(resolved["bootstrap_seed"])
    comparison_methods = tuple(
        method for method in _PAIRED_METHOD_ORDER if method in risk_by_method
    )
    if not comparison_methods or "fixed_hybrid" not in comparison_methods:
        raise RuntimeError("paired smoke comparisons require the fixed hybrid method")
    if corruption.exact_count:
        shared_bootstrap_draws = draw_group_bootstrap_indices(
            audit_group_ids,
            n_iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )
        paired_comparisons = {
            method: paired_group_bootstrap(
                corruption.is_injected_corruption,
                risk_by_method[method],
                risk_by_method[_PAIRED_COMPARATOR],
                audit_group_ids,
                n_iterations=bootstrap_iterations,
                seed=bootstrap_seed,
                bootstrap_indices=shared_bootstrap_draws,
            )
            for method in comparison_methods
        }
        valid_bootstrap_draw_indices = np.asarray(
            [
                draw_index
                for draw_index, indices in enumerate(shared_bootstrap_draws)
                if corruption.is_injected_corruption[indices].any()
            ],
            dtype=np.int64,
        )
    else:
        # AP is undefined at 0% corruption. Do not create pseudo-distributions by
        # assigning zeros or sampling source groups for an inapplicable estimand.
        shared_bootstrap_draws = ()
        valid_bootstrap_draw_indices = np.empty(0, dtype=np.int64)
        paired_comparisons = {
            method: _empty_bootstrap_result(
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            )
            for method in comparison_methods
        }
    valid_iteration_counts = {result.valid_iterations for result in paired_comparisons.values()}
    if valid_iteration_counts != {len(valid_bootstrap_draw_indices)}:
        raise RuntimeError("paired methods disagree on valid shared bootstrap draws")
    bootstrap = paired_comparisons["fixed_hybrid"]
    downstream = evaluate_downstream_restoration(
        dataset.audit_features[audit],
        corruption.pre_corruption_labels,
        corruption.observed_labels,
        corruption.is_injected_corruption,
        dataset.audit_features[final_test],
        dataset.pre_corruption_labels[final_test],
        hybrid,
        review_budget=primary_budget,
        sample_ids=audit_sample_ids,
        development_group_ids=audit_group_ids,
        final_test_group_ids=final_test_group_ids,
        final_test_is_injected_corruption=tuple(
            bool(value) for value in full_is_injected_corruption[final_test]
        ),
        class_order=tuple(range(len(CLASS_NAMES))),
        random_repeats=int(resolved["downstream_random_repeats"]),
        random_seed=int(resolved["random_review_seed"]),
        model_seed=int(resolved["model_seed"]),
    )
    if corruption.independence_status == "not_applicable":
        feature_space_independence: dict[str, Any] = {
            "status": "not_applicable",
            "independent": None,
            "reason": corruption.independence_reason,
        }
    elif corruption.independence_status == "verified_independent":
        if corruption.independence_evidence is None:
            raise RuntimeError("verified independence is missing frozen evidence")
        feature_space_independence = {
            "status": "verified_independent",
            "independent": True,
            "reason": corruption.independence_reason,
            "evidence": corruption.independence_evidence.as_dict(),
        }
    elif corruption.independence_status == "circularity_risk":
        feature_space_independence = {
            "status": "circularity_risk",
            "independent": False,
            "reason": corruption.independence_reason,
            "evidence": (
                corruption.independence_evidence.as_dict()
                if corruption.independence_evidence is not None
                else {"status": "not_supplied", "reason": corruption.independence_reason}
            ),
        }
    else:
        feature_space_independence = {
            "status": "unverified",
            "independent": {
                "status": "not_computed",
                "value": None,
                "reason": corruption.independence_reason,
            },
            "reason": corruption.independence_reason,
            "evidence": (
                corruption.independence_evidence.as_dict()
                if corruption.independence_evidence is not None
                else {"status": "not_supplied", "reason": corruption.independence_reason}
            ),
        }
    metrics: dict[str, Any] = {
        "artifact_scope": "synthetic_software_validation",
        "resolved_core_config": {
            **resolved,
            "final_test_fold": (
                resolved["final_test_fold"]
                if resolved["final_test_fraction_groups"] is None
                else {
                    "status": "not_applicable",
                    "value": None,
                    "reason": (
                        "The synthetic final-reference partition is selected by a source-group "
                        "fraction, not by an official fold identifier."
                    ),
                }
            ),
            "final_test_fraction_groups": (
                resolved["final_test_fraction_groups"]
                if resolved["final_test_fraction_groups"] is not None
                else {
                    "status": "not_applicable",
                    "value": None,
                    "reason": "An explicit synthetic official fold is used instead.",
                }
            ),
        },
        "sample_counts": {
            "total": len(dataset.records),
            "audit_pool": len(audit),
            "reference_validation": len(split.reference_validation_indices),
            "final_reference_test": len(final_test),
            "source_groups": len(set(all_group_ids)),
        },
        "corruption": {
            "mechanism": corruption.mechanism,
            "requested_rate": corruption.requested_rate,
            "exact_count": corruption.exact_count,
            "feature_space_independence": feature_space_independence,
            "circularity_risk": corruption.circularity_risk,
            "configuration_hash": corruption.configuration_hash,
        },
        "oof": {
            "folds": len(oof.folds),
            "complete_once_coverage": bool(np.all(oof.coverage_count == 1)),
            "maximum_probability_sum_error": float(
                np.max(np.abs(oof.probabilities.sum(axis=1) - 1.0))
            ),
            "group_overlap_count": 0,
        },
        "ranking": ranking_metrics,
        "random_review": _random_review_dict(random_summary, injected_total=corruption.exact_count),
        "random_review_by_budget": {
            budget: _random_review_dict(summary, injected_total=corruption.exact_count)
            for budget, summary in random_summaries_by_budget.items()
        },
        "paired_group_bootstrap_hybrid_minus_self_confidence": _bootstrap_dict(bootstrap),
        "paired_method_differences": {
            "status": "reported" if corruption.exact_count else "not_applicable",
            **({} if corruption.exact_count else {"reason": _ZERO_CORRUPTION_REASON}),
            "metric": "average_precision",
            "comparator": _PAIRED_COMPARATOR,
            "comparator_role": "predeclared_synthetic_smoke_comparator",
            "difference_direction": "selected_method_minus_comparator",
            "pairing_unit": "source_group",
            "iterations": bootstrap_iterations,
            "valid_iterations": len(valid_bootstrap_draw_indices),
            "bootstrap_seed": bootstrap_seed,
            "shared_draws": (
                True if corruption.exact_count else _not_applicable(_ZERO_CORRUPTION_REASON)
            ),
            "evidence_file": "bootstrap_evidence.npz",
            "comparison_order": comparison_methods,
            "comparisons": {
                method: _paired_comparison_dict(method, paired_comparisons[method])
                for method in comparison_methods
            },
        },
        "downstream_restoration": downstream.as_dict(),
        "cleanlab": {
            "available": cleanlab.available,
            "package_version": cleanlab.package_version,
            "api_path": cleanlab.api_path,
            "error": cleanlab.error,
            "issue_count": (
                int(cleanlab.issue_mask.sum()) if cleanlab.issue_mask is not None else None
            ),
        },
    }
    report_inputs: dict[str, Any] = {
        "artifact_scope": "synthetic_software_validation",
        "medical_disclaimer": "Software-validation data only; no diagnostic or clinical claim.",
        "class_names": CLASS_NAMES,
        "split": {
            "final_test_fold": split.final_test_fold,
            "audit_groups": split.audit_groups,
            "reference_validation_groups": split.reference_validation_groups,
            "final_test_groups": split.final_test_groups,
        },
        "risk_methods": tuple(risk_by_method),
        "primary_review_budget": primary_budget,
        "metrics": metrics,
    }

    run_id = "synthetic_smoke_in_memory"
    run_dir: Path | None = None
    metrics_path: Path | None = None
    predictions_path: Path | None = None
    rankings_path: Path | None = None
    corruption_manifest_path: Path | None = None
    oof_provenance_path: Path | None = None
    representation_example_path: Path | None = None
    neighbour_evidence_path: Path | None = None
    restoration_evidence_path: Path | None = None
    bootstrap_evidence_path: Path | None = None
    dataset_evidence_path: Path | None = None
    source_manifest_path: Path | None = None
    source_manifest_csv_path: Path | None = None
    report_inputs_path: Path | None = None
    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
        run_id, run_dir = _make_run_dir(root, resolved)
        metrics_path = run_dir / "metrics.json"
        predictions_path = run_dir / "oof_predictions.npz"
        rankings_path = run_dir / "ranking.csv"
        corruption_manifest_path = run_dir / "corruption_manifest.json"
        oof_provenance_path = run_dir / "oof_provenance.json"
        representation_example_path = run_dir / "target_representation_example.npz"
        neighbour_evidence_path = run_dir / "neighbour_evidence.npz"
        restoration_evidence_path = run_dir / "restoration_evidence.npz"
        bootstrap_evidence_path = run_dir / "bootstrap_evidence.npz"
        dataset_evidence_path = run_dir / "synthetic_dataset_evidence.npz"
        source_manifest_path = run_dir / "synthetic_source_manifest.json"
        source_manifest_csv_path = run_dir / "synthetic_source_manifest.csv"
        report_inputs_path = run_dir / "report_inputs.json"
        draw_offsets = np.zeros(len(shared_bootstrap_draws) + 1, dtype=np.int64)
        if shared_bootstrap_draws:
            draw_offsets[1:] = np.cumsum(
                [len(indices) for indices in shared_bootstrap_draws], dtype=np.int64
            )
            flattened_draw_indices = np.concatenate(shared_bootstrap_draws).astype(np.int64)
        else:
            flattened_draw_indices = np.empty(0, dtype=np.int64)
        _atomic_npz(
            bootstrap_evidence_path,
            {
                "schema_version": np.asarray(1, dtype=np.int64),
                "status": np.asarray(
                    "reported" if corruption.exact_count else "not_applicable",
                    dtype=np.str_,
                ),
                "reason": np.asarray(
                    "" if corruption.exact_count else _ZERO_CORRUPTION_REASON,
                    dtype=np.str_,
                ),
                "metric_name": np.asarray("average_precision", dtype=np.str_),
                "comparator_method": np.asarray(_PAIRED_COMPARATOR, dtype=np.str_),
                "comparison_methods": np.asarray(comparison_methods, dtype=np.str_),
                "difference_direction": np.asarray(
                    "selected_method_minus_comparator", dtype=np.str_
                ),
                "pairing_unit": np.asarray("source_group", dtype=np.str_),
                "requested_iterations": np.asarray(bootstrap_iterations, dtype=np.int64),
                "valid_iterations": np.asarray(len(valid_bootstrap_draw_indices), dtype=np.int64),
                "bootstrap_seed": np.asarray(bootstrap_seed, dtype=np.int64),
                "sample_ids": np.asarray(audit_sample_ids, dtype=np.str_),
                "group_ids": np.asarray(audit_group_ids, dtype=np.str_),
                "draw_indices": flattened_draw_indices,
                "draw_offsets": draw_offsets,
                "valid_draw_indices": valid_bootstrap_draw_indices,
                "metric_a": np.stack(
                    [paired_comparisons[method].metric_a for method in comparison_methods]
                ),
                "metric_b": np.stack(
                    [paired_comparisons[method].metric_b for method in comparison_methods]
                ),
                "differences": np.stack(
                    [paired_comparisons[method].differences for method in comparison_methods]
                ),
            },
        )
        _atomic_json(metrics_path, metrics)
        np.savez_compressed(
            predictions_path,
            sample_ids=np.asarray(audit_sample_ids, dtype=np.str_),
            group_ids=np.asarray(audit_group_ids, dtype=np.str_),
            tissue_type=np.asarray(tissue_types, dtype=np.str_),
            pre_corruption_label=corruption.pre_corruption_labels,
            observed_label=corruption.observed_labels,
            is_injected_corruption=corruption.is_injected_corruption,
            probabilities=oof.probabilities,
            predicted_class=oof.predicted_class,
            fold_id=oof.fold_id,
            class_order=np.asarray(oof.class_order, dtype=np.int64),
            cleanlab_available=np.asarray(cleanlab.available, dtype=bool),
            cleanlab_quality_score=(
                cleanlab.quality_scores
                if cleanlab.quality_scores is not None
                else np.empty(0, dtype=np.float64)
            ),
            cleanlab_risk_score=(
                cleanlab.risk_scores
                if cleanlab.risk_scores is not None
                else np.empty(0, dtype=np.float64)
            ),
            cleanlab_issue_flag=(
                cleanlab.issue_mask if cleanlab.issue_mask is not None else np.empty(0, dtype=bool)
            ),
            cleanlab_suggested_class=(
                cleanlab.suggested_class
                if cleanlab.suggested_class is not None
                else np.empty(0, dtype=np.int64)
            ),
        )
        order = rank_indices(hybrid, tie_break_ids=audit_sample_ids)
        with rankings_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "rank",
                    "sample_id",
                    "group_id",
                    "tissue_type",
                    "pre_corruption_label",
                    "observed_label",
                    "is_injected_corruption",
                    "predicted_class",
                    *risk_by_method,
                    "cleanlab_quality_score",
                    "cleanlab_risk_score",
                    "cleanlab_issue_flag",
                    "cleanlab_suggested_class",
                ]
            )
            for rank, index in enumerate(order, start=1):
                writer.writerow(
                    [
                        rank,
                        audit_sample_ids[index],
                        audit_group_ids[index],
                        tissue_types[index],
                        int(corruption.pre_corruption_labels[index]),
                        int(corruption.observed_labels[index]),
                        bool(corruption.is_injected_corruption[index]),
                        int(oof.predicted_class[index]),
                        *(float(scores[index]) for scores in risk_by_method.values()),
                        (
                            float(cleanlab.quality_scores[index])
                            if cleanlab.quality_scores is not None
                            else ""
                        ),
                        (
                            float(cleanlab.risk_scores[index])
                            if cleanlab.risk_scores is not None
                            else ""
                        ),
                        (
                            bool(cleanlab.issue_mask[index])
                            if cleanlab.issue_mask is not None
                            else ""
                        ),
                        (
                            int(cleanlab.suggested_class[index])
                            if cleanlab.suggested_class is not None
                            else ""
                        ),
                    ]
                )
        _atomic_json(
            corruption_manifest_path,
            {
                "configuration_hash": corruption.configuration_hash,
                "configuration_payload": json.loads(corruption.configuration_payload_json),
                "feature_space_independence": feature_space_independence,
                "rows": corruption.manifest_rows(audit_sample_ids, audit_group_ids),
            },
        )
        _atomic_json(
            oof_provenance_path,
            {
                "class_order": oof.class_order,
                "representation": oof.representation,
                "model_seed": oof.model_seed,
                "split_seed": oof.split_seed,
                "splitter_class_name": oof.splitter_class_name,
                "splitter_fallback_status": oof.splitter_fallback_status,
                "splitter_fallback_reason": oof.splitter_fallback_reason,
                "final_reference_groups": oof.final_reference_groups,
                "folds": [
                    {
                        "fold_id": fold.fold_id,
                        "training_groups": fold.training_groups,
                        "held_out_groups": fold.held_out_groups,
                        "held_out_sample_ids": fold.held_out_sample_ids,
                        "group_overlap": sorted(
                            set(fold.training_groups).intersection(fold.held_out_groups)
                        ),
                    }
                    for fold in oof.folds
                ],
            },
        )

        n_samples = len(dataset.records)
        partition_coverage = np.zeros(n_samples, dtype=np.int64)
        split_partition = np.full(n_samples, "unassigned", dtype="<U32")
        partition_indices = (
            ("audit_pool", audit),
            ("reference_validation", split.reference_validation_indices),
            ("final_reference_test", final_test),
        )
        for partition_name, indices in partition_indices:
            index_array = np.asarray(indices, dtype=np.int64)
            partition_coverage[index_array] += 1
            split_partition[index_array] = partition_name
        if not np.all(partition_coverage == 1):
            raise RuntimeError(
                "synthetic evidence requires every sample in exactly one outer partition"
            )

        full_original_class = dataset.pre_corruption_labels.copy()
        full_replacement_class = np.full(n_samples, -1, dtype=np.int64)
        full_corruption_type = np.full(n_samples, "none", dtype="<U64")
        full_oof_fold_id = np.full(n_samples, -1, dtype=np.int64)
        full_observed_label[audit] = corruption.observed_labels
        full_is_injected_corruption[audit] = corruption.is_injected_corruption
        full_original_class[audit] = corruption.original_class
        full_replacement_class[audit] = corruption.replacement_class
        full_corruption_type[audit] = np.where(
            corruption.is_injected_corruption, corruption.mechanism, "none"
        )
        full_oof_fold_id[audit] = oof.fold_id
        if np.any(full_is_injected_corruption[split.reference_validation_indices]) or np.any(
            full_is_injected_corruption[final_test]
        ):
            raise RuntimeError(
                "reference-validation and final-reference samples must be uncorrupted"
            )
        if not np.array_equal(
            full_observed_label[final_test], dataset.pre_corruption_labels[final_test]
        ):
            raise RuntimeError("final-reference observed labels must equal pre-corruption labels")

        dataset_configuration_hashes = {
            str(record.configuration_hash) for record in dataset.records
        }
        if len(dataset_configuration_hashes) != 1:
            raise RuntimeError("synthetic source records have inconsistent configuration hashes")
        dataset_configuration_hash = next(iter(dataset_configuration_hashes))
        evidence_arrays: dict[str, np.ndarray] = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "record_index": np.arange(n_samples, dtype=np.int64),
            "sample_ids": np.asarray(all_sample_ids, dtype=np.str_),
            "group_ids": np.asarray(all_group_ids, dtype=np.str_),
            "patch_ids": np.asarray([record.patch_id for record in dataset.records], dtype=np.str_),
            "instance_id": np.asarray(
                [record.instance_id for record in dataset.records], dtype=np.int64
            ),
            "tissue_type": np.asarray(
                [record.tissue_type for record in dataset.records], dtype=np.str_
            ),
            "official_fold": np.asarray(dataset.official_folds, dtype=np.int64),
            "split_partition": split_partition,
            "oof_fold_id": full_oof_fold_id,
            "images": np.asarray(dataset.images, dtype=np.uint8),
            "target_masks": np.asarray(dataset.target_masks, dtype=bool),
            "audit_features": np.asarray(dataset.audit_features, dtype=np.float64),
            "corruption_features": np.asarray(dataset.corruption_features, dtype=np.float64),
            "pre_corruption_label": np.asarray(dataset.pre_corruption_labels, dtype=np.int64),
            "observed_label": full_observed_label,
            "is_injected_corruption": full_is_injected_corruption,
            "original_class": full_original_class,
            "replacement_class": full_replacement_class,
            "corruption_type": full_corruption_type,
            "class_names": np.asarray(CLASS_NAMES, dtype=np.str_),
            "dataset_configuration_hash": np.asarray(dataset_configuration_hash, dtype=np.str_),
            "corruption_configuration_hash": np.asarray(
                corruption.configuration_hash, dtype=np.str_
            ),
        }
        _atomic_npz(dataset_evidence_path, evidence_arrays)

        corruption_rows = corruption.manifest_rows(audit_sample_ids, audit_group_ids)
        corruption_row_by_sample = {str(row["sample_id"]): row for row in corruption_rows}
        full_manifest_rows: list[dict[str, object]] = []
        for record_index, record in enumerate(dataset.records):
            row: dict[str, object] = dict(record.as_dict())
            source_configuration_hash = str(row["configuration_hash"])
            audit_row = corruption_row_by_sample.get(record.sample_id)
            if audit_row is not None:
                row.update(
                    {
                        "pre_corruption_label": audit_row["pre_corruption_label"],
                        "observed_label": audit_row["observed_label"],
                        "is_injected_corruption": audit_row["is_injected_corruption"],
                        "corruption_type": audit_row["corruption_type"],
                        "original_class": audit_row["original_class"],
                        "replacement_class": audit_row["replacement_class"],
                        "corruption_seed": audit_row["corruption_seed"],
                        "corruption_rate": audit_row["corruption_rate"],
                        "corruption_representation": audit_row["corruption_representation"],
                        "auditor_representation": audit_row["auditor_representation"],
                        "feature_space_independent": audit_row["feature_space_independent"],
                        "circularity_risk": audit_row["circularity_risk"],
                        "dataset_seed": audit_row["dataset_seed"],
                        "configuration_hash": audit_row["configuration_hash"],
                        # Execution time remains in corruption_manifest.json. Omitting it
                        # here keeps the generated dataset evidence byte-deterministic.
                        "corruption_timestamp_utc": None,
                    }
                )
            oof_fold_value = int(full_oof_fold_id[record_index])
            row.update(
                {
                    "record_index": record_index,
                    "split_partition": str(split_partition[record_index]),
                    "oof_fold_id": oof_fold_value if oof_fold_value >= 0 else None,
                    "dataset_configuration_hash": source_configuration_hash,
                    "corruption_configuration_hash": (
                        corruption.configuration_hash if audit_row is not None else None
                    ),
                    "corruption_upstream_manifest_hash": (
                        corruption.upstream_manifest_hash if audit_row is not None else None
                    ),
                    "corruption_independence_status": (
                        corruption.independence_status
                        if audit_row is not None
                        else "not_applicable"
                    ),
                }
            )
            full_manifest_rows.append(row)
        source_rows = tuple(full_manifest_rows)
        source_manifest_fields = tuple(source_rows[0])
        _atomic_csv_rows(source_manifest_csv_path, source_rows, source_manifest_fields)
        source_manifest_payload: dict[str, Any] = {
            "schema_version": 1,
            "artifact_scope": "synthetic_software_validation",
            "record_count": n_samples,
            "class_names": CLASS_NAMES,
            "generator": {
                "schema_version": SYNTHETIC_GENERATOR_SCHEMA_VERSION,
                "code_sha256": synthetic_generator_code_sha256(),
                "dataset_configuration_hash": dataset_configuration_hash,
                "dataset_seed": int(resolved["dataset_seed"]),
            },
            "split": {
                "partition_names": tuple(name for name, _ in partition_indices),
                "oof_fold_unassigned_value_in_npz": -1,
                "oof_fold_unassigned_value_in_manifest": None,
                "replacement_class_not_applicable_value_in_npz": -1,
            },
            "dataset_evidence": {
                "file": dataset_evidence_path.name,
                "sha256": _file_sha256(dataset_evidence_path),
                "arrays": {
                    name: {"dtype": str(value.dtype), "shape": list(value.shape)}
                    for name, value in evidence_arrays.items()
                },
            },
            "csv_manifest": {
                "file": source_manifest_csv_path.name,
                "sha256": _file_sha256(source_manifest_csv_path),
                "columns": source_manifest_fields,
                "structured_cell_encoding": "compact_json",
                "null_cell_encoding": "empty_string",
            },
            "records_sha256": canonical_sha256(source_rows),
            "records": source_rows,
        }
        _atomic_json(source_manifest_path, source_manifest_payload)

        write_neighbour_evidence(
            neighbour_evidence_path,
            neighbours,
            sample_ids=audit_sample_ids,
            group_ids=audit_group_ids,
        )
        write_restoration_evidence(
            restoration_evidence_path,
            downstream,
            development_sample_ids=audit_sample_ids,
            development_group_ids=audit_group_ids,
            pre_corruption_label=corruption.pre_corruption_labels,
            observed_label=corruption.observed_labels,
            is_injected_corruption=corruption.is_injected_corruption,
            final_test_sample_ids=final_test_sample_ids,
            final_test_group_ids=final_test_group_ids,
            final_test_reference_label=dataset.pre_corruption_labels[final_test],
            final_test_is_injected_corruption=full_is_injected_corruption[final_test],
            class_order=oof.class_order,
        )

        example_dataset_index = int(audit[0])
        example_record = dataset.records[example_dataset_index]
        example_image = dataset.images[example_dataset_index]
        example_mask = dataset.target_masks[example_dataset_index]
        example_crop = extract_target_crop(
            example_image,
            example_mask,
            output_size=48,
            padding=example_record.crop_padding,
        )
        np.savez_compressed(
            representation_example_path,
            sample_id=np.asarray(example_record.sample_id, dtype=np.str_),
            target_instance_id=np.asarray(example_record.instance_id, dtype=np.int64),
            full_patch=example_image,
            full_target_mask=example_mask,
            source_bbox=np.asarray(example_record.bbox, dtype=np.int64),
            crop_source_box=np.asarray(example_crop.source_box, dtype=np.int64),
            target_crop=example_crop.image,
            crop_target_mask=example_crop.target_mask,
            highlighted_full_patch=highlight_target(example_image, example_mask),
            highlighted_crop=highlight_target(example_crop.image, example_crop.target_mask),
        )
        report_inputs = {
            **report_inputs,
            "run_id": run_id,
            "metrics_path": str(metrics_path),
            "predictions_path": str(predictions_path),
            "rankings_path": str(rankings_path),
            "corruption_manifest_path": str(corruption_manifest_path),
            "oof_provenance_path": str(oof_provenance_path),
            "representation_example_path": str(representation_example_path),
            "neighbour_evidence_path": str(neighbour_evidence_path),
            "restoration_evidence_path": str(restoration_evidence_path),
            "bootstrap_evidence_path": str(bootstrap_evidence_path),
            "dataset_evidence_path": str(dataset_evidence_path),
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_csv_path": str(source_manifest_csv_path),
            "report_inputs_path": str(report_inputs_path),
        }
        _atomic_json(report_inputs_path, report_inputs)
    return SyntheticSmokeResult(
        success=True,
        status="completed",
        run_id=run_id,
        run_dir=run_dir,
        metrics=metrics,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        rankings_path=rankings_path,
        corruption_manifest_path=corruption_manifest_path,
        oof_provenance_path=oof_provenance_path,
        representation_example_path=representation_example_path,
        neighbour_evidence_path=neighbour_evidence_path,
        restoration_evidence_path=restoration_evidence_path,
        bootstrap_evidence_path=bootstrap_evidence_path,
        dataset_evidence_path=dataset_evidence_path,
        source_manifest_path=source_manifest_path,
        source_manifest_csv_path=source_manifest_csv_path,
        report_inputs=report_inputs,
        report_inputs_path=report_inputs_path,
    )


run_smoke = run_synthetic_smoke
