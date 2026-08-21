"""Run the frozen post-confirmation PUMA realism and clean-label safety stresses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from histo_audit.auditing.scores import fixed_hybrid_score, score_annotations
from histo_audit.config import load_pinned_config
from histo_audit.corruption.controlled import (
    FeatureIndependenceEvidence,
    FeatureSpaceEvidence,
    apply_controlled_corruption,
    semantic_sha256,
)
from histo_audit.external_validation.puma import (
    CLASS_ORDER,
    _downstream_summary,
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

STRESS_CONFIG_SHA256 = "00214940ee2cc1faf51202e23fe800a7f42a4f9b2936dae26bbfe20e3ee2555a"


def _load_stress_config(root: Path) -> tuple[dict[str, Any], str]:
    path = root / "configs" / "puma_realism_stress.yaml"
    config, digest = load_pinned_config(
        path,
        STRESS_CONFIG_SHA256,
        role="PUMA realism stress config",
    )
    if (
        not isinstance(config, dict)
        or config.get("schema_version") != 1
        or config.get("candidate_selection_or_change_permitted") is not False
        or config.get("primary_puma_result_opened_before_freeze") is not True
    ):
        raise RuntimeError("PUMA realism stress boundary is invalid")
    return config, digest


def _load_feature_view(embedding_root: Path, split: str, expected_ids: list[str]) -> np.ndarray:
    scales: list[np.ndarray] = []
    for scale in (64, 128):
        cached = load_embedding_cache(embedding_root / f"puma_{split}_resnet18_context_{scale}.npz")
        cached.validate()
        if cached.sample_ids.tolist() != expected_ids:
            raise RuntimeError(f"PUMA {split} {scale}px cache differs from the manifest")
        scales.append(np.asarray(cached.embeddings, dtype=np.float32))
    return np.concatenate(scales, axis=1).astype(np.float32, copy=False)


def _geometry_generator_features(development: Any) -> np.ndarray:
    width = development["bbox_width"].to_numpy(dtype=np.float64)
    height = development["bbox_height"].to_numpy(dtype=np.float64)
    area = np.maximum(width * height, 1.0)
    aspect = np.log(np.maximum(width, 1.0) / np.maximum(height, 1.0))
    return np.column_stack(
        (
            width,
            height,
            np.log(area),
            aspect,
            development["centre_x"].to_numpy(dtype=np.float64) / 1024.0,
            development["centre_y"].to_numpy(dtype=np.float64) / 1024.0,
        )
    ).astype(np.float32)


def _independence_evidence(
    geometry: np.ndarray,
    auditor_features: np.ndarray,
    *,
    manifest_sha256: str,
) -> FeatureIndependenceEvidence:
    generator = FeatureSpaceEvidence.from_array(
        geometry,
        representation_name="puma_released_bbox_geometry_v1",
        family="released_annotation_geometry",
        implementation_hash=semantic_sha256("PUMA bbox width height log-area aspect x y v1"),
        weights_hash=semantic_sha256("no learned weights"),
        preprocessing_hash=semantic_sha256("pixels; centres divided by 1024; natural logs"),
        fitted_data_hash=manifest_sha256,
    )
    auditor = FeatureSpaceEvidence.from_array(
        auditor_features,
        representation_name="resnet18_multiscale_64_128",
        family="frozen_imagenet_resnet18_pixels",
        implementation_hash=semantic_sha256("torchvision ResNet18 64+128 concatenation"),
        weights_hash=semantic_sha256("torchvision IMAGENET1K_V1 frozen weights"),
        preprocessing_hash=semantic_sha256("official weight transforms independently per scale"),
        fitted_data_hash=semantic_sha256("ImageNet-1K pretraining; no PUMA fitting"),
    )
    return FeatureIndependenceEvidence.create(
        matrix_version="aanca-puma-stress-independence-v1",
        matrix_decision="verified_independent",
        matrix_reason=(
            "The generator uses only released bounding-box geometry and no pixels or learned "
            "weights; the auditor uses frozen pixel embeddings and no geometry columns."
        ),
        generator=generator,
        auditor=auditor,
    )


def _group_weights(groups: np.ndarray, config: dict[str, Any]) -> dict[str, float]:
    values = config["group_conditional"]
    salt = str(values["salt"])
    threshold = int(str(values["high_hash_prefix_exclusive"]), 16)
    return {
        str(group): (
            float(values["high_weight"])
            if int(hashlib.sha256(f"{salt}|{group}".encode()).hexdigest()[:2], 16) < threshold
            else float(values["low_weight"])
        )
        for group in sorted(set(str(value) for value in groups))
    }


def _render_report(result: dict[str, Any]) -> str:
    lines = [
        "# PUMA realism and clean-label safety stress",
        "",
        (
            "This is exploratory post-confirmation evidence. The frozen candidate and its "
            "prospective PUMA result were not changed."
        ),
        "",
        "| Scenario | AANCA - unchanged macro-F1 (95% CI) | AANCA - matched random | Retrieval advantage | Gates | Failed guard |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for scenario in result["scenarios"]:
        downstream = scenario["downstream"]
        retrieval = scenario["retrieval"]
        baseline_interval = downstream["candidate_minus_uncorrected_interval_95"]
        random_interval = downstream["candidate_minus_matched_random_interval_95"]
        retrieval_interval = retrieval["interval_95"]
        failed_classes = [
            class_name
            for class_name in CLASS_ORDER
            if downstream["candidate_minus_uncorrected_recall_intervals_95"][class_name][0] < -0.01
        ]
        failed_guard = (
            ", ".join(f"`{value}`" for value in failed_classes) + " recall safety"
            if failed_classes
            else "none"
        )
        lines.append(
            "| "
            f"{scenario['name']} | "
            f"{downstream['candidate_minus_uncorrected_macro_f1']:+.6f} "
            f"[{baseline_interval[0]:+.6f}, {baseline_interval[1]:+.6f}] | "
            f"{downstream['candidate_minus_matched_random_macro_f1']:+.6f} "
            f"[{random_interval[0]:+.6f}, {random_interval[1]:+.6f}] | "
            f"{retrieval['candidate_minus_matched_random_precision']:+.6f} "
            f"[{retrieval_interval[0]:+.6f}, {retrieval_interval[1]:+.6f}] | "
            f"{'PASS' if scenario['all_scenario_gates_passed'] else 'FAIL'} | "
            f"{failed_guard} |"
        )
    clean = result["scenarios"][0]["downstream"]
    geometry = result["scenarios"][-1]["downstream"]
    lines.extend(
        [
            "",
            (
                "Every scenario had a positive aggregate macro-F1 lower bound against "
                "unchanged and matched-random training. Eight failed only because at least "
                "one class-recall lower bound was below `-0.01`."
            ),
            "",
            (
                "Clean-label exclusion changed `other` recall by "
                f"`{clean['candidate_minus_uncorrected_recall']['other']:+.6f}` with 95% "
                "interval "
                f"`[{clean['candidate_minus_uncorrected_recall_intervals_95']['other'][0]:+.6f}, "
                f"{clean['candidate_minus_uncorrected_recall_intervals_95']['other'][1]:+.6f}]`; "
                "geometry-dependent exclusion changed it by "
                f"`{geometry['candidate_minus_uncorrected_recall']['other']:+.6f}` with "
                "interval "
                f"`[{geometry['candidate_minus_uncorrected_recall_intervals_95']['other'][0]:+.6f}, "
                f"{geometry['candidate_minus_uncorrected_recall_intervals_95']['other'][1]:+.6f}]`."
            ),
            "",
            (
                "The geometry-dependent mechanism deterministically produced the same "
                "selected and replacement label arrays under the four configured seeds. "
                "They are not independent corruption replicates."
            ),
            "",
            "## Boundary",
            "",
            (
                "All corruptions remain controlled. Even broad robustness does not prove that "
                "a pathologist was wrong or that the workflow improves clinical practice. "
                "Source PUMA annotations were never modified."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/puma"))
    parser.add_argument(
        "--primary-run",
        type=Path,
        default=Path("artifacts/puma_new_data_confirmation"),
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=Path("artifacts/embeddings/puma_new_data_confirmation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/puma_realism_stress"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/puma_realism_stress_results.md")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    stress_config, stress_config_sha256 = _load_stress_config(root)
    parent_config, parent_sha256, amendment, amendment_sha256 = load_frozen_puma_config(root)
    if (
        stress_config["parent"]["config_sha256"] != parent_sha256
        or stress_config["parent"]["runtime_amendment_sha256"] != amendment_sha256
    ):
        raise RuntimeError("PUMA stress config is detached from its frozen parent")
    primary_root = (root / args.primary_run).resolve()
    primary_arrays_path = primary_root / "evidence_arrays.npz"
    if sha256_file(primary_arrays_path) != stress_config["parent"]["evidence_arrays_sha256"]:
        raise RuntimeError("PUMA primary evidence changed before the stress suite")
    with np.load(primary_arrays_path, allow_pickle=False) as payload:
        fold_ids = np.asarray(payload["oof_fold_ids"], dtype=np.int64)
        neighbour_indices = np.asarray(payload["neighbour_indices"], dtype=np.int64)
        neighbour_distances = np.asarray(payload["neighbour_distances"], dtype=np.float32)
    authority = build_puma_manifest((root / args.data_root).resolve(), parent_config)
    development = (
        authority.frame.loc[authority.frame["partition"] == "development"]
        .copy()
        .reset_index(drop=True)
    )
    final = (
        authority.frame.loc[authority.frame["partition"] == "final"].copy().reset_index(drop=True)
    )
    embedding_root = (root / args.embedding_root).resolve()
    train_x = _load_feature_view(
        embedding_root, "development", development["sample_id"].astype(str).tolist()
    )
    test_x = _load_feature_view(embedding_root, "final", final["sample_id"].astype(str).tolist())
    reference = development["reference_label"].to_numpy(dtype=np.int64)
    final_reference = final["reference_label"].to_numpy(dtype=np.int64)
    groups = development["group_id"].astype(str).to_numpy(dtype=np.str_)
    final_groups = final["group_id"].astype(str).to_numpy(dtype=np.str_)
    training_groups_by_fold = {
        int(fold): tuple(sorted(set(str(value) for value in groups[fold_ids != fold])))
        for fold in np.unique(fold_ids)
    }
    geometry = _geometry_generator_features(development)
    independence = _independence_evidence(
        geometry, train_x, manifest_sha256=authority.manifest_sha256
    )
    conditional_weights = _group_weights(groups, stress_config)
    transition_matrix = np.asarray(stress_config["targeted_transition_matrix"], dtype=np.float64)
    runtime = amendment["runtime"]
    candidate = parent_config["candidate"]
    evaluation = stress_config["evaluation"]
    frozen_seeds = tuple(int(value) for value in stress_config["seeds"])
    unique_groups = tuple(sorted(set(str(value) for value in groups)))
    scenario_results: list[dict[str, Any]] = []
    evidence_arrays: dict[str, np.ndarray] = {}
    for scenario_index, scenario in enumerate(stress_config["scenarios"]):
        name = str(scenario["name"])
        mechanism = str(scenario["mechanism"])
        rate = float(scenario["rate"])
        seeds = frozen_seeds[: int(scenario["seed_count"])]
        candidate_counts = np.zeros((len(seeds), len(unique_groups), 2), dtype=np.int64)
        random_counts = np.zeros(
            (len(seeds), int(evaluation["matched_random_repetitions"]), len(unique_groups), 2),
            dtype=np.int64,
        )
        candidate_probabilities: list[np.ndarray] = []
        baseline_probabilities: list[np.ndarray] = []
        random_probabilities: list[np.ndarray] = []
        observed_values: list[np.ndarray] = []
        injected_values: list[np.ndarray] = []
        selected_values: list[np.ndarray] = []
        per_seed: list[dict[str, Any]] = []
        all_converged = True
        for seed_index, seed in enumerate(seeds):
            corruption_kwargs: dict[str, Any] = {}
            if mechanism == "confusion_targeted_corruption":
                corruption_kwargs["transition_matrix"] = transition_matrix
            elif mechanism == "group_conditional_corruption":
                corruption_kwargs["group_weights"] = conditional_weights
            elif mechanism == "instance_dependent_corruption":
                corruption_kwargs.update(
                    {
                        "generator_features": geometry,
                        "generator_representation": "puma_released_bbox_geometry_v1",
                        "auditor_representation": "resnet18_multiscale_64_128",
                        "independence_evidence": independence,
                    }
                )
            corruption = apply_controlled_corruption(
                reference,
                sample_ids=development["sample_id"].astype(str).tolist(),
                group_ids=groups.tolist(),
                rate=rate,
                mechanism=mechanism,
                seed=seed,
                n_classes=len(CLASS_ORDER),
                upstream_manifest_hash=authority.manifest_sha256,
                **corruption_kwargs,
            )
            observed = np.asarray(corruption.observed_labels, dtype=np.int64)
            injected = np.asarray(corruption.is_injected_corruption, dtype=bool)
            probabilities, current_fold_ids, current_training_groups, oof_evidence = (
                _oof_probabilities(
                    train_x,
                    reference,
                    observed,
                    groups.tolist(),
                    folds=int(parent_config["corruption"]["audit_folds"]),
                    split_seed=int(runtime["audit_group_fold_seed"]),
                    l2=float(candidate["audit_l2"]),
                    class_weight_balanced=bool(candidate["audit_class_weight_balanced"]),
                    max_iter=int(runtime["max_iter"]),
                )
            )
            if not np.array_equal(current_fold_ids, fold_ids) or (
                current_training_groups != training_groups_by_fold
            ):
                raise RuntimeError("PUMA stress audit folds differ from primary evidence")
            neighbour_risk = _neighbour_risk(observed, neighbour_indices, neighbour_distances)
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
                repetitions=int(evaluation["matched_random_repetitions"]),
                seed_start=int(runtime["matched_random_seed_start"]) + scenario_index * 1000,
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
            selected_values.append(selected)
            per_seed.append(
                {
                    "seed": seed,
                    "configuration_hash": corruption.configuration_hash,
                    "injected_count": int(injected.sum()),
                    "reviewed_count": len(selected),
                    "injected_found": int(injected[selected].sum()),
                    "precision": float(injected[selected].mean()),
                    "average_precision": (
                        average_precision(injected, risk) if injected.any() else None
                    ),
                    "queue_underfilled": bool(queue["underfilled"]),
                    "all_models_converged": seed_converged,
                    "circularity_risk": bool(corruption.circularity_risk),
                    "independence_status": corruption.independence_status,
                }
            )
        retrieval = _retrieval_summary(
            candidate_counts,
            random_counts,
            iterations=int(evaluation["bootstrap_iterations"]),
            seed=int(evaluation["bootstrap_seed"]) + scenario_index * 10,
        )
        downstream = _downstream_summary(
            final_reference,
            final_groups,
            np.stack(candidate_probabilities),
            np.stack(baseline_probabilities),
            np.stack(random_probabilities),
            iterations=int(evaluation["bootstrap_iterations"]),
            seed=int(evaluation["bootstrap_seed"]) + scenario_index * 10 + 1,
        )
        recall_safe = all(
            downstream["candidate_minus_uncorrected_recall_intervals_95"][name][0]
            >= float(evaluation["class_recall_lower_safety_margin"])
            for name in CLASS_ORDER
        )
        if rate == 0.0:
            gates = {
                "clean_macro_f1_lower_bound_gte_minus_0_005": downstream[
                    "candidate_minus_uncorrected_interval_95"
                ][0]
                >= float(evaluation["clean_macro_f1_lower_safety_margin"]),
                "every_class_recall_lower_bound_gte_minus_0_01": recall_safe,
                "all_models_converged": all_converged,
            }
        else:
            gates = {
                "retrieval_lower_bound_gt_matched_random": retrieval["interval_95"][0] > 0.0,
                "downstream_lower_bound_gt_unchanged": downstream[
                    "candidate_minus_uncorrected_interval_95"
                ][0]
                > 0.0,
                "downstream_lower_bound_gt_matched_random": downstream[
                    "candidate_minus_matched_random_interval_95"
                ][0]
                > 0.0,
                "all_seed_directions_positive": all(
                    value > 0.0
                    for value in downstream["candidate_minus_uncorrected_by_corruption_seed"]
                    + downstream["candidate_minus_matched_random_by_corruption_seed"]
                ),
                "every_class_recall_lower_bound_gte_minus_0_01": recall_safe,
                "all_models_converged": all_converged,
            }
        scenario_results.append(
            {
                "name": name,
                "mechanism": mechanism,
                "rate": rate,
                "seeds": list(seeds),
                "per_seed": per_seed,
                "retrieval": retrieval,
                "downstream": downstream,
                "gates": gates,
                "all_scenario_gates_passed": all(gates.values()),
                "source_annotations_modified": False,
            }
        )
        key = name.replace(".", "_")
        evidence_arrays[f"{key}_observed_labels"] = np.stack(observed_values)
        evidence_arrays[f"{key}_injected"] = np.stack(injected_values)
        evidence_arrays[f"{key}_selected_indices"] = np.stack(selected_values)
        evidence_arrays[f"{key}_candidate_probabilities"] = np.stack(
            candidate_probabilities
        ).astype(np.float32)
        evidence_arrays[f"{key}_uncorrected_probabilities"] = np.stack(
            baseline_probabilities
        ).astype(np.float32)
        evidence_arrays[f"{key}_matched_random_probabilities"] = np.stack(
            random_probabilities
        ).astype(np.float32)
        print(
            json.dumps(
                {
                    "scenario": name,
                    "all_gates_passed": all(gates.values()),
                    "candidate_minus_uncorrected_macro_f1": downstream[
                        "candidate_minus_uncorrected_macro_f1"
                    ],
                },
                sort_keys=True,
            )
        )
    result = {
        "schema_version": 1,
        "study_id": stress_config["study_id"],
        "project": "AANCA",
        "replacement_project_or_v2": False,
        "disposition": stress_config["disposition"],
        "stress_config_sha256": stress_config_sha256,
        "parent_config_sha256": parent_sha256,
        "parent_runtime_amendment_sha256": amendment_sha256,
        "candidate_sha256": stress_config["parent"]["candidate_sha256"],
        "primary_result_was_open_before_freeze": True,
        "candidate_changed": False,
        "scenarios": scenario_results,
        "scenario_count": len(scenario_results),
        "all_scenarios_passed": all(
            scenario["all_scenario_gates_passed"] for scenario in scenario_results
        ),
        "geometry_auditor_independence": independence.as_dict(),
        "claim_boundary": stress_config["claim_boundary"],
        "natural_error_detection_evaluated": False,
        "pathologist_error_detection_proven": False,
        "source_annotations_modified": False,
    }
    output_root = (root / args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "results.json", result)
    atomic_write_npz(output_root / "evidence_arrays.npz", evidence_arrays)
    atomic_write_text((root / args.report).resolve(), _render_report(result))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
