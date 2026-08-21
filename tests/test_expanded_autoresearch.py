from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from histo_audit.external_validation.monusac import MoNuSACPreparedData
from histo_audit.research.autoresearch import AutoresearchCandidate
from histo_audit.research.expanded_autoresearch import (
    ExpandedAutoresearchEvaluator,
    build_all_development_partition,
    build_expanded_feature_views,
    load_expanded_autoresearch_config,
    validate_aligned_scales,
)


def _prepared(*, crop_size: int, groups: int = 10) -> MoNuSACPreparedData:
    records: list[dict[str, object]] = []
    crops: list[np.ndarray] = []
    for group_index in range(groups):
        patient = f"TCGA-AA-{group_index:04d}"
        for label in range(4):
            index = group_index * 4 + label
            records.append(
                {
                    "sample_id": f"monusac-train-{index:024d}",
                    "split": "train",
                    "patient_id": patient,
                    "group_id": patient,
                    "organ": ("Breast", "Kidney", "Lung", "Prostate")[group_index % 4],
                    "image_id": f"image-{group_index}",
                    "region_authority": str(label),
                    "reference_label": label,
                    "reference_label_name": str(label),
                    "xmin": 2.0,
                    "ymin": 3.0,
                    "xmax": 6.0 + label,
                    "ymax": 8.0 + group_index % 3,
                    "xml_file": f"{patient}/image.xml",
                    "image_file": f"{patient}/image.tif",
                }
            )
            crop = np.full((crop_size, crop_size, 3), 20 + label * 50, dtype=np.uint8)
            crop[:, :, group_index % 3] = np.uint8(40 + group_index * 3)
            crops.append(crop)
    return MoNuSACPreparedData(
        split="train",
        manifest=pd.DataFrame.from_records(records),
        crops=np.stack(crops),
        exclusions={},
        source_inventory=(),
        source_inventory_sha256="a" * 64,
        manifest_sha256="b" * 64,
        crops_sha256=("c" if crop_size == 64 else "d") * 64,
    )


def _config() -> tuple[dict[str, object], str]:
    config, digest = load_expanded_autoresearch_config(".")
    output = deepcopy(config)
    output["data"]["expected_patient_groups_after_exclusion"] = 10
    output["partition"]["discovery_outer_folds"] = 2
    output["partition"]["audit_inner_folds"] = 2
    output["controlled_corruption"]["discovery_seeds"] = [101]
    output["successive_halving"]["screen_corruption_seeds"] = 1
    output["successive_halving"]["screen_outer_folds"] = 2
    output["controls"]["matched_random_fields"] = ["observed_class"]
    output["controls"]["matched_random_repetitions_screen"] = 1
    output["controls"]["bootstrap_iterations_screen"] = 20
    return output, digest


def test_expanded_partition_uses_all_patients_only_as_development() -> None:
    prepared = _prepared(crop_size=64)
    config, _ = _config()
    partition = build_all_development_partition(prepared, config)
    assert len(partition.discovery_groups) == 10
    assert not len(partition.lockbox_groups)
    assert not len(partition.lockbox_indices)
    np.testing.assert_array_equal(
        partition.discovery_indices, np.arange(len(prepared.manifest), dtype=np.int64)
    )
    coverage = np.zeros(len(prepared.manifest), dtype=np.int64)
    for train, validation in partition.discovery_outer_folds:
        assert set(prepared.manifest.iloc[train]["group_id"]).isdisjoint(
            prepared.manifest.iloc[validation]["group_id"]
        )
        coverage[validation] += 1
    np.testing.assert_array_equal(coverage, np.ones(len(coverage), dtype=np.int64))


def test_expanded_views_are_multiscale_aligned_and_label_independent() -> None:
    prepared_64 = _prepared(crop_size=64)
    prepared_128 = _prepared(crop_size=128)
    validate_aligned_scales(prepared_64, prepared_128)
    count = len(prepared_64.manifest)
    r64 = np.arange(count * 512, dtype=np.float32).reshape(count, 512)
    r128 = (r64 + 1.0).astype(np.float32)
    phikon = np.arange(count * 1024, dtype=np.float32).reshape(count, 1024)
    views = build_expanded_feature_views(prepared_64, prepared_128, r64, r128, phikon)
    assert views["resnet18_context_64_plus_stats"].shape == (count, 532)
    assert views["resnet18_context_128_plus_stats"].shape == (count, 532)
    assert views["resnet18_multiscale_64_128"].shape == (count, 1024)
    assert views["resnet18_multiscale_64_128_plus_stats"].shape == (count, 1064)
    assert views["phikon_v2_context_128_plus_stats"].shape == (count, 1044)
    assert views["phikon_v2_resnet18_multiscale"].shape == (count, 2048)
    assert all(np.isfinite(values).all() for values in views.values())


def test_trusted_review_weighting_restores_only_exposed_injected_rows() -> None:
    observed = np.asarray([0, 1, 2, 3], dtype=np.int64)
    reference = np.asarray([1, 1, 3, 3], dtype=np.int64)
    injected = np.asarray([True, False, True, False])
    selected = np.asarray([0, 1], dtype=np.int64)
    source = observed.copy()
    labels, weights, restored = ExpandedAutoresearchEvaluator._derive_intervention(
        observed,
        reference,
        injected,
        selected,
        "controlled_restore_selected_weight_2",
    )
    np.testing.assert_array_equal(observed, source)
    np.testing.assert_array_equal(labels, np.asarray([1, 1, 2, 3]))
    np.testing.assert_array_equal(weights, np.asarray([2.0, 2.0, 1.0, 1.0]))
    assert restored == 1


def test_expanded_ranking_reserves_an_exact_disjoint_comparator() -> None:
    prepared = _prepared(crop_size=64)
    config, digest = _config()
    partition = build_all_development_partition(prepared, config)
    rng = np.random.default_rng(42)
    features = rng.normal(size=(len(prepared.manifest), 12)).astype(np.float64)
    for label in range(4):
        features[prepared.manifest["reference_label"].to_numpy() == label, label] += 3.0
    evaluator = ExpandedAutoresearchEvaluator(
        prepared,
        {"resnet18_context_64": features},
        partition,
        config,
        config_sha256=digest,
    )
    result = evaluator.evaluate_ranking(
        AutoresearchCandidate(
            risk_method="self_confidence",
            queue_preset="global",
            review_budget=0.10,
        )
    )
    assert result["stage"] == "ranking_screen"
    assert result["final_external_test_used"] is False
    assert result["queue_evidence"][0]["exact_comparator_capacity_enforced"] is True
    assert result["retrieval"]["matched_random_repetitions"] == 1
