from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from histo_audit.external_validation.monusac import MoNuSACPreparedData
from histo_audit.research.autoresearch import (
    AutoresearchCandidate,
    AutoresearchEvaluator,
    build_autoresearch_feature_views,
    build_autoresearch_partition,
    generate_ranking_candidates,
    load_autoresearch_config,
)


def _prepared(*, split: str = "train", groups: int = 10) -> MoNuSACPreparedData:
    records: list[dict[str, object]] = []
    crops: list[np.ndarray] = []
    for group_index in range(groups):
        patient = f"TCGA-AA-{group_index:04d}"
        for label in range(4):
            sample_index = group_index * 4 + label
            records.append(
                {
                    "sample_id": f"monusac-{split}-{sample_index:024d}",
                    "split": split,
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
            crop = np.full((8, 8, 3), 20 + label * 50, dtype=np.uint8)
            crop[:, :, group_index % 3] = np.uint8(40 + group_index * 3)
            crops.append(crop)
    return MoNuSACPreparedData(
        split=split,
        manifest=pd.DataFrame.from_records(records),
        crops=np.stack(crops),
        exclusions={},
        source_inventory=(),
        source_inventory_sha256="a" * 64,
        manifest_sha256="b" * 64,
        crops_sha256="c" * 64,
    )


def _config() -> tuple[dict[str, object], str]:
    config, config_sha256 = load_autoresearch_config(".")
    output = deepcopy(config)
    output["data"]["expected_patient_groups_after_exclusion"] = 10
    output["partition"]["folds"] = 5
    output["partition"]["discovery_outer_folds"] = 2
    output["partition"]["audit_inner_folds"] = 2
    output["controlled_corruption"]["discovery_seeds"] = [101]
    output["successive_halving"]["screen_corruption_seeds"] = 1
    output["successive_halving"]["screen_outer_folds"] = 2
    output["controls"]["matched_random_fields"] = ["observed_class"]
    output["controls"]["matched_random_repetitions_screen"] = 1
    output["controls"]["bootstrap_iterations_screen"] = 20
    return output, config_sha256


def test_autoresearch_partition_is_patient_safe_and_keeps_a_lockbox() -> None:
    prepared = _prepared()
    config, _ = _config()
    partition = build_autoresearch_partition(prepared, config)
    assert len(partition.discovery_groups) + len(partition.lockbox_groups) == 10
    assert set(partition.discovery_groups).isdisjoint(partition.lockbox_groups)
    coverage = np.zeros(len(prepared.manifest), dtype=np.int64)
    for train, validation in partition.discovery_outer_folds:
        assert set(prepared.manifest.iloc[train]["group_id"]).isdisjoint(
            prepared.manifest.iloc[validation]["group_id"]
        )
        coverage[validation] += 1
    np.testing.assert_array_equal(
        coverage[partition.discovery_indices], np.ones(len(partition.discovery_indices))
    )
    assert not coverage[partition.lockbox_indices].any()


def test_autoresearch_rejects_nontraining_samples() -> None:
    prepared = _prepared(split="test")
    config, _ = _config()
    try:
        build_autoresearch_partition(prepared, config)
    except ValueError as error:
        assert "non-training" in str(error)
    else:
        raise AssertionError("test samples entered the development-only search")


def test_autoresearch_feature_views_are_label_independent_and_aligned() -> None:
    prepared = _prepared()
    embeddings = np.arange(len(prepared.manifest) * 512, dtype=np.float32).reshape(-1, 512)
    views = build_autoresearch_feature_views(prepared, embeddings)
    assert views["resnet18_context_64"].shape == (40, 512)
    assert views["resnet18_context_64_plus_stats"].shape == (40, 532)
    assert np.isfinite(views["resnet18_context_64_plus_stats"]).all()
    np.testing.assert_array_equal(views["resnet18_context_64"][:, :4], embeddings[:, :4])


def test_candidate_generation_is_deterministic_unique_and_contains_baseline() -> None:
    config, _ = _config()
    config["successive_halving"]["ranking_screen_max_trials"] = 20
    first = generate_ranking_candidates(config, ("resnet18_context_64",))
    second = generate_ranking_candidates(config, ("resnet18_context_64",))
    assert [value.candidate_sha256 for value in first] == [
        value.candidate_sha256 for value in second
    ]
    assert len(first) == len({value.candidate_sha256 for value in first}) == 20
    assert AutoresearchCandidate().candidate_sha256 in {value.candidate_sha256 for value in first}


def test_nested_screen_never_uses_lockbox_or_modifies_source_labels() -> None:
    prepared = _prepared()
    config, config_sha256 = _config()
    partition = build_autoresearch_partition(prepared, config)
    rng = np.random.default_rng(44)
    features = rng.normal(size=(len(prepared.manifest), 12)).astype(np.float64)
    for label in range(4):
        features[prepared.manifest["reference_label"].to_numpy() == label, label] += 3.0
    evaluator = AutoresearchEvaluator(
        prepared,
        {"resnet18_context_64": features},
        partition,
        config,
        config_sha256=config_sha256,
    )
    original_labels = prepared.manifest["reference_label"].copy()
    candidate = AutoresearchCandidate(
        risk_method="self_confidence",
        queue_preset="global",
        review_budget=0.05,
    )
    result = evaluator.evaluate_downstream_screen(candidate)
    assert result["stage"] == "downstream_screen"
    assert result["final_external_test_used"] is False
    assert result["source_annotations_modified"] is False
    assert result["downstream"]["evaluated_patient_groups"] == len(partition.discovery_groups)
    pd.testing.assert_series_equal(prepared.manifest["reference_label"], original_labels)
