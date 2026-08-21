from __future__ import annotations

from pathlib import Path

import numpy as np

from histo_audit.external_validation.puma import (
    CLASS_ORDER,
    RAW_TO_PRIMARY,
    _downstream_summary,
    _exact_fold_neighbours,
    frozen_puma_case_split,
    load_frozen_puma_config,
)


def _case_ids() -> list[str]:
    return [
        *(f"training_set_primary_roi_{index:03d}" for index in range(1, 104)),
        *(f"training_set_metastatic_roi_{index:03d}" for index in range(1, 104)),
    ]


def test_frozen_puma_case_split_is_deterministic_stratified_and_disjoint() -> None:
    config = {
        "split": {
            "salt": "AANCA-PUMA-FINAL-V1",
            "final_cases_per_stratum": 31,
        }
    }
    first = frozen_puma_case_split(_case_ids(), config)
    second = frozen_puma_case_split(list(reversed(_case_ids())), config)
    assert first == second
    development, final = first
    assert len(development) == 144
    assert len(final) == 62
    assert not set(development).intersection(final)
    assert sum("primary" in value for value in final) == 31
    assert sum("metastatic" in value for value in final) == 31


def test_puma_primary_mapping_matches_the_official_three_class_benchmark() -> None:
    assert CLASS_ORDER == ("tumor", "lymphocyte", "other")
    assert RAW_TO_PRIMARY["nuclei_tumor"] == "tumor"
    assert RAW_TO_PRIMARY["nuclei_lymphocyte"] == "lymphocyte"
    assert RAW_TO_PRIMARY["nuclei_plasma_cell"] == "lymphocyte"
    assert set(RAW_TO_PRIMARY.values()) == set(CLASS_ORDER)
    assert len(RAW_TO_PRIMARY) == 10


def test_puma_freeze_files_bind_the_existing_candidate() -> None:
    root = Path(__file__).resolve().parents[1]
    config, config_sha256, amendment, amendment_sha256 = load_frozen_puma_config(root)
    assert config_sha256 == "434452cf94dce9cf9ce88edd79761686821043c53d64ccb00c02a3a547bdef30"
    assert amendment_sha256 == "d93bb2a353e0cdc931edf432fc8b83a82a67857aa90fe95e473ccedd732d0d07"
    assert config["candidate"]["intervention"] == "flag_exclude"
    assert amendment["candidate_changed"] is False


def test_exact_puma_neighbours_exclude_complete_query_groups() -> None:
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [1.0, 1.0],
            [0.9, 0.9],
            [-1.0, 0.0],
            [-0.9, 0.1],
        ],
        dtype=np.float32,
    )
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    folds = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    training_groups = {0: ("b", "d"), 1: ("a", "c")}
    indices, distances, evidence = _exact_fold_neighbours(
        features,
        folds,
        training_groups,
        groups,
        [f"sample-{index}" for index in range(len(features))],
        k=1,
        device="cpu",
        query_chunk_size=2,
    )
    assert indices.shape == (8, 1)
    assert distances.shape == (8, 1)
    assert evidence["backend"] == "exact_fold_safe_torch_float32_cosine"
    for row, neighbour in enumerate(indices[:, 0]):
        assert groups[row] != groups[int(neighbour)]


def test_near_but_distinct_float32_neighbours_do_not_trigger_full_tie_scan() -> None:
    rng = np.random.default_rng(91)
    features = rng.normal(size=(18, 7)).astype(np.float32)
    groups = [f"group-{index // 3}" for index in range(18)]
    folds = np.asarray([index // 6 for index in range(18)], dtype=np.int64)
    training_groups = {
        fold: tuple(sorted(set(groups) - set(groups[fold * 6 : (fold + 1) * 6])))
        for fold in range(3)
    }
    _, _, evidence = _exact_fold_neighbours(
        features,
        folds,
        training_groups,
        groups,
        [f"unique-{index:02d}" for index in range(18)],
        k=2,
        device="cpu",
        query_chunk_size=3,
    )
    assert evidence["boundary_tie_full_scan_rows"] == 0


def test_puma_downstream_summary_uses_three_classes_and_case_bootstrap() -> None:
    groups = np.repeat(np.asarray([f"case-{index}" for index in range(6)]), 3)
    reference = np.tile(np.arange(3, dtype=np.int64), 6)
    perfect = np.eye(3, dtype=np.float64)[reference]
    baseline = np.tile(np.asarray([[0.8, 0.1, 0.1]], dtype=np.float64), (len(reference), 1))
    random = baseline.copy()
    summary = _downstream_summary(
        reference,
        groups.astype(np.str_),
        np.stack((perfect, perfect)),
        np.stack((baseline, baseline)),
        np.stack(((random,), (random,))),
        iterations=100,
        seed=17,
    )
    assert summary["candidate_minus_uncorrected_macro_f1"] > 0.0
    assert summary["candidate_minus_uncorrected_interval_95"][0] > 0.0
    assert set(summary["candidate_minus_uncorrected_recall"]) == set(CLASS_ORDER)
