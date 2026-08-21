"""Run the frozen PUMA observed-label OOF-allocation sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from histo_audit.auditing.scores import fixed_hybrid_score, score_annotations
from histo_audit.config import load_pinned_config
from histo_audit.corruption.controlled import apply_controlled_corruption
from histo_audit.external_validation.puma import (
    CLASS_ORDER,
    _array_sha256,
    _downstream_summary,
    _exact_fold_neighbours,
    _fit_predict,
    _matched_random,
    _neighbour_risk,
    _oof_probabilities,
    _retrieval_group_counts,
    _retrieval_summary,
    _select_queue,
    build_puma_manifest,
    load_frozen_puma_config,
)
from histo_audit.representations.imagenet import load_embedding_cache
from histo_audit.statistics.review import average_precision
from histo_audit.utils.run_tracking import (
    atomic_write_json,
    atomic_write_npz,
    atomic_write_text,
    sha256_file,
)

SENSITIVITY_CONFIG_SHA256 = "ed6fd1e85d15604efc331b634a0d7604ca2675ba58345aa31386c266781e661f"


def _load_sensitivity_config(root: Path) -> tuple[dict[str, Any], str]:
    path = root / "configs" / "puma_audit_time_label_sensitivity.yaml"
    config, digest = load_pinned_config(
        path,
        SENSITIVITY_CONFIG_SHA256,
        role="PUMA audit-time-label sensitivity config",
    )
    if (
        not isinstance(config, dict)
        or config.get("schema_version") != 1
        or config.get("primary_puma_result_opened_before_freeze") is not True
        or config.get("candidate_selection_or_change_permitted") is not False
        or config.get("fold_assignment", {}).get("label_source") != "observed_label"
    ):
        raise RuntimeError("PUMA audit-time-label sensitivity boundary is invalid")
    return config, digest


def _load_features(root: Path, split: str, sample_ids: list[str]) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for size in (64, 128):
        path = root / f"puma_{split}_resnet18_context_{size}.npz"
        cache = load_embedding_cache(path)
        cache.validate()
        if cache.sample_ids.tolist() != sample_ids:
            raise RuntimeError(f"PUMA {split} {size}px cache differs from the manifest")
        arrays.append(np.asarray(cache.embeddings, dtype=np.float32))
    return np.concatenate(arrays, axis=1).astype(np.float32, copy=False)


def _render_report(result: dict[str, Any]) -> str:
    retrieval = result["retrieval"]
    downstream = result["downstream"]
    lines = [
        "# PUMA audit-time-label sensitivity",
        "",
        (
            "This post-confirmation sensitivity allocated every audit fold from the "
            "observed label available at audit time. It did not change the candidate."
        ),
        "",
        "## Result",
        "",
        (
            f"AANCA retrieval precision was `{retrieval['candidate_precision']:.6f}` "
            f"versus `{retrieval['mean_matched_random_precision']:.6f}` for exact "
            "matched random. The difference was "
            f"`{retrieval['candidate_minus_matched_random_precision']:+.6f}` with "
            f"95% interval `[{retrieval['interval_95'][0]:+.6f}, "
            f"{retrieval['interval_95'][1]:+.6f}]`."
        ),
        "",
        (
            "AANCA minus unchanged macro-F1 was "
            f"`{downstream['candidate_minus_uncorrected_macro_f1']:+.6f}` with "
            f"interval `[{downstream['candidate_minus_uncorrected_interval_95'][0]:+.6f}, "
            f"{downstream['candidate_minus_uncorrected_interval_95'][1]:+.6f}]`. "
            "AANCA minus exact matched random was "
            f"`{downstream['candidate_minus_matched_random_macro_f1']:+.6f}` with "
            f"interval `[{downstream['candidate_minus_matched_random_interval_95'][0]:+.6f}, "
            f"{downstream['candidate_minus_matched_random_interval_95'][1]:+.6f}]`."
        ),
        "",
        "## Frozen sensitivity gates",
        "",
    ]
    for name, passed in result["success_conditions"].items():
        lines.append(f"- `{name}`: **{'PASS' if passed else 'FAIL'}**")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "PUMA outcomes were open before this sensitivity was frozen. This is not "
                "independent confirmation and does not evaluate natural errors, pathologist "
                "errors, clinical utility or automatic annotation changes."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/puma"))
    parser.add_argument(
        "--primary-run", type=Path, default=Path("artifacts/puma_new_data_confirmation")
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=Path("artifacts/embeddings/puma_new_data_confirmation"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/puma_audit_time_label_sensitivity")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/puma_audit_time_label_sensitivity_results.md"),
    )
    parser.add_argument("--neighbour-device", default="auto")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config, config_sha256 = _load_sensitivity_config(root)
    parent, parent_sha256, amendment, amendment_sha256 = load_frozen_puma_config(root)
    if (
        config["parent"]["config_sha256"] != parent_sha256
        or config["parent"]["runtime_amendment_sha256"] != amendment_sha256
        or config["parent"]["candidate_sha256"] != parent["candidate"]["sha256"]
    ):
        raise RuntimeError("PUMA audit-time sensitivity is detached from its parent")
    primary_root = (root / args.primary_run).resolve()
    if sha256_file(primary_root / "results.json") != config["parent"]["primary_results_sha256"]:
        raise RuntimeError("PUMA primary results changed before sensitivity execution")
    primary_arrays_path = primary_root / "evidence_arrays.npz"
    if sha256_file(primary_arrays_path) != config["parent"]["primary_evidence_arrays_sha256"]:
        raise RuntimeError("PUMA primary evidence changed before sensitivity execution")
    with np.load(primary_arrays_path, allow_pickle=False) as payload:
        primary_fold_ids = np.asarray(payload["oof_fold_ids"], dtype=np.int64)
    authority = build_puma_manifest((root / args.data_root).resolve(), parent)
    development = (
        authority.frame.loc[authority.frame["partition"] == "development"]
        .copy()
        .reset_index(drop=True)
    )
    final = (
        authority.frame.loc[authority.frame["partition"] == "final"].copy().reset_index(drop=True)
    )
    embedding_root = (root / args.embedding_root).resolve()
    train_x = _load_features(
        embedding_root, "development", development["sample_id"].astype(str).tolist()
    )
    test_x = _load_features(embedding_root, "final", final["sample_id"].astype(str).tolist())
    reference = development["reference_label"].to_numpy(dtype=np.int64)
    final_reference = final["reference_label"].to_numpy(dtype=np.int64)
    groups = development["group_id"].astype(str).to_numpy(dtype=np.str_)
    final_groups = final["group_id"].astype(str).to_numpy(dtype=np.str_)
    unique_groups = tuple(sorted(set(str(value) for value in groups)))
    candidate = parent["candidate"]
    runtime = amendment["runtime"]
    evaluation = config["evaluation"]
    seeds = tuple(int(value) for value in config["corruption"]["seeds"])
    repetitions = int(evaluation["matched_random_repetitions"])
    candidate_counts = np.zeros((len(seeds), len(unique_groups), 2), dtype=np.int64)
    random_counts = np.zeros((len(seeds), repetitions, len(unique_groups), 2), dtype=np.int64)
    candidate_probabilities: list[np.ndarray] = []
    baseline_probabilities: list[np.ndarray] = []
    random_probabilities: list[np.ndarray] = []
    observed_values: list[np.ndarray] = []
    injected_values: list[np.ndarray] = []
    oof_probability_values: list[np.ndarray] = []
    fold_values: list[np.ndarray] = []
    neighbour_index_values: list[np.ndarray] = []
    neighbour_distance_values: list[np.ndarray] = []
    selected_values: list[np.ndarray] = []
    random_index_values: list[np.ndarray] = []
    per_seed: list[dict[str, Any]] = []
    all_converged = True
    for seed_index, seed in enumerate(seeds):
        corruption = apply_controlled_corruption(
            reference,
            sample_ids=development["sample_id"].astype(str).tolist(),
            group_ids=groups.tolist(),
            rate=float(config["corruption"]["fraction"]),
            mechanism=str(config["corruption"]["mechanism"]),
            seed=seed,
            n_classes=len(CLASS_ORDER),
            upstream_manifest_hash=authority.manifest_sha256,
        )
        observed = np.asarray(corruption.observed_labels, dtype=np.int64)
        injected = np.asarray(corruption.is_injected_corruption, dtype=bool)
        probabilities, fold_ids, training_groups, oof_evidence = _oof_probabilities(
            train_x,
            observed,
            observed,
            groups.tolist(),
            folds=int(config["fold_assignment"]["folds"]),
            split_seed=int(config["fold_assignment"]["split_seed"]),
            l2=float(candidate["audit_l2"]),
            class_weight_balanced=bool(candidate["audit_class_weight_balanced"]),
            max_iter=int(runtime["max_iter"]),
        )
        neighbours, distances, neighbour_evidence = _exact_fold_neighbours(
            train_x,
            fold_ids,
            training_groups,
            groups.tolist(),
            development["sample_id"].astype(str).tolist(),
            k=int(candidate["neighbour_k"]),
            device=str(args.neighbour_device),
            query_chunk_size=int(runtime["neighbour_query_chunk_size"]),
        )
        neighbour_risk = _neighbour_risk(observed, neighbours, distances)
        self_confidence = score_annotations(
            observed,
            probabilities,
            method="self_confidence",
            class_order=tuple(range(len(CLASS_ORDER))),
        )
        risk = fixed_hybrid_score(
            {
                "self_confidence": self_confidence,
                "nearest_neighbour_disagreement": neighbour_risk,
            },
            components=("self_confidence", "nearest_neighbour_disagreement"),
            weights=(
                float(candidate["hybrid_self_confidence_weight"]),
                1.0 - float(candidate["hybrid_self_confidence_weight"]),
            ),
        )
        selected, queue = _select_queue(
            development,
            train_x,
            observed,
            probabilities,
            risk,
            review_budget=float(evaluation["review_budget"]),
        )
        comparators = _matched_random(
            selected,
            development,
            observed,
            probabilities,
            repetitions=repetitions,
            seed_start=int(runtime["matched_random_seed_start"]),
            corruption_seed=seed,
        )
        candidate_counts[seed_index] = _retrieval_group_counts(
            selected, injected, groups, unique_groups
        )
        for repeat, indices in enumerate(comparators):
            random_counts[seed_index, repeat] = _retrieval_group_counts(
                indices, injected, groups, unique_groups
            )
        baseline, baseline_converged = _fit_predict(
            train_x,
            observed,
            test_x,
            l2=float(candidate["downstream_l2"]),
            class_weight_balanced=bool(candidate["downstream_class_weight_balanced"]),
            max_iter=int(runtime["max_iter"]),
        )
        weights = np.ones(len(observed), dtype=np.float64)
        weights[selected] = 0.0
        candidate_values, candidate_converged = _fit_predict(
            train_x,
            observed,
            test_x,
            l2=float(candidate["downstream_l2"]),
            class_weight_balanced=bool(candidate["downstream_class_weight_balanced"]),
            max_iter=int(runtime["max_iter"]),
            sample_weight=weights,
        )
        seed_random: list[np.ndarray] = []
        random_converged: list[bool] = []
        for indices in comparators:
            weights = np.ones(len(observed), dtype=np.float64)
            weights[indices] = 0.0
            values, converged = _fit_predict(
                train_x,
                observed,
                test_x,
                l2=float(candidate["downstream_l2"]),
                class_weight_balanced=bool(candidate["downstream_class_weight_balanced"]),
                max_iter=int(runtime["max_iter"]),
                sample_weight=weights,
            )
            seed_random.append(values)
            random_converged.append(converged)
        seed_converged = (
            baseline_converged
            and candidate_converged
            and all(random_converged)
            and all(bool(value["converged"]) for value in oof_evidence)
        )
        all_converged = all_converged and seed_converged
        candidate_probabilities.append(candidate_values)
        baseline_probabilities.append(baseline)
        random_probabilities.append(np.stack(seed_random))
        observed_values.append(observed)
        injected_values.append(injected)
        oof_probability_values.append(probabilities)
        fold_values.append(fold_ids)
        neighbour_index_values.append(neighbours)
        neighbour_distance_values.append(distances)
        selected_values.append(selected)
        random_index_values.append(np.stack(comparators))
        per_seed.append(
            {
                "seed": seed,
                "configuration_hash": corruption.configuration_hash,
                "injected_count": int(injected.sum()),
                "reviewed_count": len(selected),
                "injected_found": int(injected[selected].sum()),
                "precision": float(injected[selected].mean()),
                "average_precision": average_precision(injected, risk),
                "queue_underfilled": bool(queue["underfilled"]),
                "all_models_converged": seed_converged,
                "fold_assignment_label_source": "observed_label",
                "query_groups_excluded_from_neighbours": True,
                "fold_ids_sha256": _array_sha256(fold_ids),
                "fraction_matching_primary_reference_label_folds": float(
                    np.mean(fold_ids == primary_fold_ids)
                ),
                "neighbour_evidence": neighbour_evidence,
            }
        )
        print(
            json.dumps(
                {
                    "seed": seed,
                    "precision": float(injected[selected].mean()),
                    "fold_match_fraction": float(np.mean(fold_ids == primary_fold_ids)),
                },
                sort_keys=True,
            )
        )
    retrieval = _retrieval_summary(
        candidate_counts,
        random_counts,
        iterations=int(evaluation["bootstrap_iterations"]),
        seed=int(evaluation["bootstrap_seed"]),
    )
    downstream = _downstream_summary(
        final_reference,
        final_groups,
        np.stack(candidate_probabilities),
        np.stack(baseline_probabilities),
        np.stack(random_probabilities),
        iterations=int(evaluation["bootstrap_iterations"]),
        seed=int(evaluation["bootstrap_seed"]) + 1,
    )
    recall_intervals = downstream["candidate_minus_uncorrected_recall_intervals_95"]
    success = {
        "retrieval_precision_lower_bound_gt_matched_random": retrieval["interval_95"][0] > 0.0,
        "downstream_macro_f1_lower_bound_gt_unchanged": downstream[
            "candidate_minus_uncorrected_interval_95"
        ][0]
        > 0.0,
        "downstream_macro_f1_lower_bound_gt_matched_random": downstream[
            "candidate_minus_matched_random_interval_95"
        ][0]
        > 0.0,
        "all_four_seed_directions_positive_against_both_controls": all(
            value > 0.0
            for value in downstream["candidate_minus_uncorrected_by_corruption_seed"]
            + downstream["candidate_minus_matched_random_by_corruption_seed"]
        ),
        "every_primary_class_recall_lower_bound_gte_minus_0_01": all(
            recall_intervals[name][0] >= -0.01 for name in CLASS_ORDER
        ),
        "all_required_models_converged": all_converged,
        "all_folds_use_observed_labels_and_exclude_query_groups": all(
            record["fold_assignment_label_source"] == "observed_label"
            and record["query_groups_excluded_from_neighbours"] is True
            for record in per_seed
        ),
    }
    result = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "project": "AANCA",
        "replacement_project_or_v2": False,
        "disposition": config["disposition"],
        "config_sha256": config_sha256,
        "parent_config_sha256": parent_sha256,
        "parent_runtime_amendment_sha256": amendment_sha256,
        "candidate_sha256": config["parent"]["candidate_sha256"],
        "candidate_changed": False,
        "fold_assignment_label_source": "observed_label",
        "pre_corruption_label_used_for_fold_assignment": False,
        "per_seed": per_seed,
        "retrieval": retrieval,
        "downstream": downstream,
        "success_conditions": success,
        "all_sensitivity_gates_passed": all(success.values()),
        "claim_boundary": config["claim_boundary"],
        "natural_error_detection_evaluated": False,
        "pathologist_error_detection_proven": False,
        "source_annotations_modified": False,
    }
    arrays = {
        "development_sample_ids": development["sample_id"].astype(str).to_numpy(dtype=np.str_),
        "final_sample_ids": final["sample_id"].astype(str).to_numpy(dtype=np.str_),
        "development_group_ids": groups,
        "final_group_ids": final_groups,
        "development_reference_labels": reference,
        "final_reference_labels": final_reference,
        "observed_labels_by_seed": np.stack(observed_values),
        "is_injected_corruption_by_seed": np.stack(injected_values),
        "oof_probabilities_by_seed": np.stack(oof_probability_values),
        "oof_fold_ids_by_seed": np.stack(fold_values),
        "neighbour_indices_by_seed": np.stack(neighbour_index_values),
        "neighbour_distances_by_seed": np.stack(neighbour_distance_values),
        "selected_indices_by_seed": np.stack(selected_values),
        "matched_random_indices_by_seed": np.stack(random_index_values),
        "final_candidate_probabilities_by_seed": np.stack(candidate_probabilities),
        "final_uncorrected_probabilities_by_seed": np.stack(baseline_probabilities),
        "final_matched_random_probabilities_by_seed": np.stack(random_probabilities),
    }
    output_root = (root / args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "results.json", result)
    atomic_write_npz(output_root / "evidence_arrays.npz", arrays)
    atomic_write_text((root / args.report).resolve(), _render_report(result))
    print(
        json.dumps(
            {
                "all_sensitivity_gates_passed": result["all_sensitivity_gates_passed"],
                "candidate_minus_uncorrected_macro_f1": downstream[
                    "candidate_minus_uncorrected_macro_f1"
                ],
                "candidate_minus_matched_random_macro_f1": downstream[
                    "candidate_minus_matched_random_macro_f1"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
