"""Independently verify the frozen PUMA evidence arrays and reported decision."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from histo_audit.external_validation.puma import (
    CLASS_ORDER,
    _downstream_summary,
    _retrieval_group_counts,
    _retrieval_summary,
    build_puma_manifest,
    load_frozen_puma_config,
)
from histo_audit.utils.run_tracking import atomic_write_json, sha256_file


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _assert_close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(actual, expected, atol=1.0e-12, rtol=0.0):
        raise RuntimeError(f"PUMA verification changed {name}: {actual} != {expected}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/puma"))
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/puma_new_data_confirmation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/puma_new_data_confirmation/verification.json"),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_root = (root / args.data_root).resolve()
    run_root = (root / args.run_root).resolve()
    results_path = run_root / "results.json"
    arrays_path = run_root / "evidence_arrays.npz"
    authority_path = run_root / "run_authority.json"
    result = _load_json(results_path)
    authority = _load_json(authority_path)
    config, config_sha256, amendment, amendment_sha256 = load_frozen_puma_config(root)
    manifest = build_puma_manifest(data_root, config)
    development = (
        manifest.frame.loc[manifest.frame["partition"] == "development"]
        .copy()
        .reset_index(drop=True)
    )
    final = manifest.frame.loc[manifest.frame["partition"] == "final"].copy().reset_index(drop=True)
    required = {
        "development_sample_ids",
        "final_sample_ids",
        "development_group_ids",
        "final_group_ids",
        "development_reference_labels",
        "final_reference_labels",
        "oof_fold_ids",
        "neighbour_indices",
        "neighbour_distances",
        "observed_labels_by_seed",
        "is_injected_corruption_by_seed",
        "risk_by_seed",
        "selected_indices_by_seed",
        "matched_random_indices_by_seed",
        "oof_probabilities_by_seed",
        "final_candidate_probabilities_by_seed",
        "final_uncorrected_probabilities_by_seed",
        "final_matched_random_probabilities_by_seed",
    }
    with np.load(arrays_path, allow_pickle=False) as payload:
        if set(payload.files) != required:
            raise RuntimeError("PUMA evidence array keys differ from the frozen verifier")
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    expected_development_ids = development["sample_id"].astype(str).tolist()
    expected_final_ids = final["sample_id"].astype(str).tolist()
    if arrays["development_sample_ids"].tolist() != expected_development_ids:
        raise RuntimeError("PUMA development evidence does not align to the official manifest")
    if arrays["final_sample_ids"].tolist() != expected_final_ids:
        raise RuntimeError("PUMA final evidence does not align to the official manifest")
    development_groups = development["group_id"].astype(str).to_numpy(dtype=np.str_)
    final_groups = final["group_id"].astype(str).to_numpy(dtype=np.str_)
    if (
        not np.array_equal(arrays["development_group_ids"], development_groups)
        or not np.array_equal(arrays["final_group_ids"], final_groups)
        or set(development_groups).intersection(final_groups)
    ):
        raise RuntimeError("PUMA evidence violates the frozen case partition")
    reference = development["reference_label"].to_numpy(dtype=np.int64)
    final_reference = final["reference_label"].to_numpy(dtype=np.int64)
    if not np.array_equal(arrays["development_reference_labels"], reference) or not np.array_equal(
        arrays["final_reference_labels"], final_reference
    ):
        raise RuntimeError("PUMA evidence changed an official source label")
    observed_by_seed = arrays["observed_labels_by_seed"].astype(np.int64)
    injected_by_seed = arrays["is_injected_corruption_by_seed"].astype(bool)
    if observed_by_seed.shape != (4, len(reference)) or injected_by_seed.shape != (
        4,
        len(reference),
    ):
        raise RuntimeError("PUMA corruption arrays have unexpected shapes")
    if not all(
        np.array_equal(observed != reference, injected)
        and int(injected.sum()) == 6703
        and np.array_equal(reference, arrays["development_reference_labels"])
        for observed, injected in zip(observed_by_seed, injected_by_seed, strict=True)
    ):
        raise RuntimeError("PUMA controlled corruption is not exactly separated from source labels")
    fold_ids = arrays["oof_fold_ids"].astype(np.int64)
    for group in np.unique(development_groups):
        if len(np.unique(fold_ids[development_groups == group])) != 1:
            raise RuntimeError("one PUMA case entered multiple audit holdout folds")
    neighbour_indices = arrays["neighbour_indices"].astype(np.int64)
    neighbour_distances = arrays["neighbour_distances"].astype(np.float32)
    if (
        neighbour_indices.shape != (len(reference), 31)
        or neighbour_distances.shape != neighbour_indices.shape
        or np.any(neighbour_indices < 0)
        or np.any(neighbour_indices >= len(reference))
        or not np.isfinite(neighbour_distances).all()
    ):
        raise RuntimeError("PUMA neighbour evidence is incomplete")
    if any(
        str(development_groups[row]) in set(str(value) for value in development_groups[indices])
        for row, indices in enumerate(neighbour_indices)
    ):
        raise RuntimeError("PUMA neighbour evidence contains its query case")
    selected_by_seed = arrays["selected_indices_by_seed"].astype(np.int64)
    matched_by_seed = arrays["matched_random_indices_by_seed"].astype(np.int64)
    oof_probabilities = arrays["oof_probabilities_by_seed"].astype(np.float64)
    if selected_by_seed.shape != (4, 3352) or matched_by_seed.shape != (4, 5, 3352):
        raise RuntimeError("PUMA review-budget arrays differ from the frozen 5% budget")
    unique_groups = tuple(sorted(set(str(value) for value in development_groups)))
    candidate_counts = np.zeros((4, len(unique_groups), 2), dtype=np.int64)
    random_counts = np.zeros((4, 5, len(unique_groups), 2), dtype=np.int64)
    strata = development["source_stratum"].astype(str).to_numpy(dtype=np.str_)
    for seed_index, (observed, injected) in enumerate(
        zip(observed_by_seed, injected_by_seed, strict=True)
    ):
        selected = selected_by_seed[seed_index]
        if len(np.unique(selected)) != len(selected):
            raise RuntimeError("PUMA AANCA queue contains duplicate indices")
        proposed = np.argmax(oof_probabilities[seed_index], axis=1).astype(np.int64)
        selected_keys = Counter(
            (int(observed[index]), str(strata[index]), int(proposed[index])) for index in selected
        )
        candidate_counts[seed_index] = _retrieval_group_counts(
            selected, injected, development_groups, unique_groups
        )
        for repeat, comparator in enumerate(matched_by_seed[seed_index]):
            if len(np.unique(comparator)) != len(comparator) or set(selected).intersection(
                comparator
            ):
                raise RuntimeError("PUMA matched control duplicates or reuses a selected nucleus")
            comparator_keys = Counter(
                (int(observed[index]), str(strata[index]), int(proposed[index]))
                for index in comparator
            )
            if comparator_keys != selected_keys:
                raise RuntimeError("PUMA matched control is not exact on every frozen field")
            random_counts[seed_index, repeat] = _retrieval_group_counts(
                comparator, injected, development_groups, unique_groups
            )
    recomputed_retrieval = _retrieval_summary(
        candidate_counts,
        random_counts,
        iterations=int(config["evaluation"]["bootstrap_iterations"]),
        seed=int(amendment["runtime"]["bootstrap_seed"]),
    )
    recomputed_downstream = _downstream_summary(
        final_reference,
        final_groups,
        arrays["final_candidate_probabilities_by_seed"].astype(np.float64),
        arrays["final_uncorrected_probabilities_by_seed"].astype(np.float64),
        arrays["final_matched_random_probabilities_by_seed"].astype(np.float64),
        iterations=int(config["evaluation"]["bootstrap_iterations"]),
        seed=int(amendment["runtime"]["bootstrap_seed"]) + 1,
    )
    for name in (
        "candidate_precision",
        "mean_matched_random_precision",
        "candidate_minus_matched_random_precision",
    ):
        _assert_close(float(recomputed_retrieval[name]), float(result["retrieval"][name]), name)
    for name in (
        "candidate_macro_f1",
        "uncorrected_macro_f1",
        "mean_matched_random_macro_f1",
        "candidate_minus_uncorrected_macro_f1",
        "candidate_minus_matched_random_macro_f1",
    ):
        _assert_close(float(recomputed_downstream[name]), float(result["downstream"][name]), name)
    for name in (
        "candidate_minus_uncorrected_interval_95",
        "candidate_minus_matched_random_interval_95",
    ):
        np.testing.assert_allclose(
            recomputed_downstream[name], result["downstream"][name], atol=1.0e-12, rtol=0.0
        )
    all_fits_converged = bool(result["fits"]) and all(
        bool(record["converged"]) for record in result["fits"]
    )
    if (
        result["config_sha256"] != config_sha256
        or result["runtime_amendment_sha256"] != amendment_sha256
        or authority["full_manifest_sha256"] != manifest.manifest_sha256
        or result["fit_count"] != 44
        or not all_fits_converged
        or not result["all_success_conditions_met"]
        or not all(result["success_conditions"].values())
    ):
        raise RuntimeError("PUMA reported authority, convergence or success gates are inconsistent")
    verification = {
        "schema_version": 1,
        "study_id": result["study_id"],
        "verified": True,
        "results_sha256": sha256_file(results_path),
        "evidence_arrays_sha256": sha256_file(arrays_path),
        "run_authority_sha256": sha256_file(authority_path),
        "config_sha256": config_sha256,
        "runtime_amendment_sha256": amendment_sha256,
        "candidate_sha256": result["candidate_sha256"],
        "official_manifest_rebuilt": True,
        "development_final_case_overlap": False,
        "source_reference_labels_unchanged": True,
        "controlled_corruption_fields_exact": True,
        "all_neighbours_exclude_complete_query_case": True,
        "all_matched_random_controls_exact_and_disjoint": True,
        "retrieval_metrics_recomputed": True,
        "downstream_metrics_and_bootstrap_recomputed": True,
        "all_44_models_converged": True,
        "all_seven_frozen_success_gates_passed": True,
        "natural_error_detection_evaluated": False,
        "pathologist_error_detection_proven": False,
        "class_order": list(CLASS_ORDER),
    }
    output = (root / args.output).resolve()
    atomic_write_json(output, verification)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
