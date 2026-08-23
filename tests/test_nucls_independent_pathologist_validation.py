from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import histo_audit.external_validation.nucls_independent_pathologist as module
from histo_audit.external_validation.nucls_independent_pathologist import (
    InputPreparedData,
    ScoredInputData,
    attach_hidden_reference,
    construct_leave_one_out_consensus,
    evaluate_scored_reference,
    load_frozen_independent_pathologist_config,
    map_nucls_superclass,
    score_pathologist_input,
    select_exact_comparator_capable_queue,
    verify_independent_pathologist_artifacts,
)
from histo_audit.utils.run_tracking import atomic_write_json, atomic_write_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("tumor", (0, "tumor_any")),
        ("mitotic_figure", (0, "tumor_any")),
        ("vascular_endothelium", (1, "nonTIL_stromal")),
        ("macrophage", (1, "nonTIL_stromal")),
        ("plasma_cell", (2, "sTIL")),
        ("undetected", (None, None)),
        ("neutrophil", (None, None)),
        (None, (None, None)),
    ],
)
def test_frozen_nucls_class_mapping(raw, expected) -> None:
    assert map_nucls_superclass(raw) == expected


def test_leave_one_out_consensus_excludes_input_and_handles_all_statuses() -> None:
    pathologists = ("SP.1", "SP.2", "JP.1", "JP.2")
    common = {"JP.1": "tumor", "EM_inferred_label_Ps": "lymphocyte"}

    agree = construct_leave_one_out_consensus(
        {**common, "SP.1": "tumor", "SP.2": "mitotic_figure", "JP.2": "undetected"},
        input_annotator="JP.1",
        pathologist_columns=pathologists,
        minimum_votes=2,
    )
    disagree = construct_leave_one_out_consensus(
        {**common, "SP.1": "lymphocyte", "SP.2": "plasma_cell", "JP.2": "undetected"},
        input_annotator="JP.1",
        pathologist_columns=pathologists,
        minimum_votes=2,
    )
    ambiguous = construct_leave_one_out_consensus(
        {**common, "SP.1": "lymphocyte", "SP.2": "fibroblast", "JP.2": "undetected"},
        input_annotator="JP.1",
        pathologist_columns=pathologists,
        minimum_votes=2,
    )
    insufficient = construct_leave_one_out_consensus(
        {**common, "SP.1": "lymphocyte", "SP.2": "undetected", "JP.2": "unlabeled"},
        input_annotator="JP.1",
        pathologist_columns=pathologists,
        minimum_votes=2,
    )

    assert agree.status == "consensus_agree"
    assert disagree.status == "consensus_disagree"
    assert ambiguous.status == "ambiguous"
    assert insufficient.status == "insufficient_reference"
    for outcome in (agree, disagree, ambiguous, insufficient):
        assert "JP.1" not in outcome.voter_ids
    assert agree.consensus_label_name == "tumor_any"
    assert disagree.consensus_label_name == "sTIL"


def _synthetic_scored_frame(count: int = 240) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(count):
        patient = f"TCGA-AA-{index % 5:04d}"
        observed = module.CLASS_ORDER[index % 3]
        proposed = module.CLASS_ORDER[(index // 3) % 3]
        event = index % 11 == 0 or index % 17 == 0
        risk = 1.0 - index / count if event else 0.45 - index / (2 * count)
        rows.append(
            {
                "sample_id": f"sample-{index:04d}",
                "patient_id": patient,
                "observed_class": observed,
                "proposed_transition": f"{observed}->{proposed}",
                "risk_aanca": risk,
                "binary_reference_eligible": True,
                "independent_consensus_disagreement": event,
                "reference_status": "consensus_disagree" if event else "consensus_agree",
            }
        )
    return pd.DataFrame.from_records(rows)


def _fast_config(repository_root: Path) -> dict[str, object]:
    frozen, _ = load_frozen_independent_pathologist_config(repository_root)
    config = copy.deepcopy(frozen)
    config["evaluation"]["matched_random_repetitions"] = 5
    config["evaluation"]["bootstrap_iterations"] = 100
    return config


def test_exact_queue_and_matched_random_metrics_are_deterministic() -> None:
    frame = _synthetic_scored_frame()
    fields = ("patient_id", "observed_class", "proposed_transition")
    first = select_exact_comparator_capable_queue(frame, budget=0.05, match_fields=fields)
    second = select_exact_comparator_capable_queue(frame, budget=0.05, match_fields=fields)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 12

    root = Path(__file__).resolve().parents[1]
    config = _fast_config(root)
    metrics_a, queues_a, curve_a = evaluate_scored_reference(frame, config=config)
    metrics_b, queues_b, curve_b = evaluate_scored_reference(frame, config=config)
    assert metrics_a == metrics_b
    assert queues_a == queues_b
    pd.testing.assert_frame_equal(curve_a, curve_b)
    assert metrics_a["primary"]["reviewed_count"] == 12
    assert metrics_a["primary"]["exact_equal_budget"] is True
    assert metrics_a["primary"]["exact_strata_preserved"] is True
    assert metrics_a["primary"]["aanca_random_disjoint"] is True
    primary = queues_a["budgets"]["0.05"]
    selected = set(primary["aanca_sample_ids"])
    for comparator in primary["matched_random"]:
        assert len(comparator["sample_ids"]) == len(selected)
        assert not selected.intersection(comparator["sample_ids"])


def test_patient_oof_and_neighbours_never_use_query_group() -> None:
    rng = np.random.default_rng(7)
    records: list[dict[str, object]] = []
    embeddings = np.zeros((90, 1024), dtype=np.float64)
    for index in range(90):
        label = index % 3
        group = f"patient-{index // 18}"
        sample_id = f"input-{index:03d}"
        records.append(
            {
                "sample_id": sample_id,
                "group_id": group,
                "patient_id": group,
                "observed_label": label,
                "observed_class": module.CLASS_ORDER[label],
            }
        )
        embeddings[index, label] = 3.0
        embeddings[index, 3:9] = rng.normal(0.0, 0.05, size=6)
    manifest = pd.DataFrame.from_records(records)
    prepared = InputPreparedData(
        manifest=manifest,
        crops={},
        exclusions={},
        source_inventory=(),
        source_inventory_sha256="inventory",
        manifest_sha256="manifest",
        crop_sha256={},
    )
    config = {
        "candidate": {
            "oof_folds": 5,
            "split_seed": 9,
            "audit_l2": 0.1,
            "audit_max_iter": 400,
            "audit_class_weight_balanced": False,
            "risk_method": "fixed_hybrid",
            "neighbour_k": 3,
            "neighbour_metric": "cosine",
            "hybrid_self_confidence_weight": 0.6,
            "hybrid_neighbour_weight": 0.4,
        }
    }
    scored = score_pathologist_input(prepared, embeddings, config=config)
    assert scored.all_models_converged is True
    assert scored.risk_metadata["reference_loaded_during_scoring"] is False
    for fold in scored.fold_evidence:
        assert not set(fold["training_groups"]).intersection(fold["held_out_groups"])
    for _, row in scored.scored_manifest.iterrows():
        neighbour_groups = set(json.loads(row["neighbour_groups_json"]))
        assert row["group_id"] not in neighbour_groups


def test_hidden_reference_attachment_ignores_aggregate_p_truth_and_input_vote(
    tmp_path: Path,
) -> None:
    contour_root = tmp_path / "data" / "raw" / "nucls" / "unbiased" / "p_truth"
    (contour_root / "contours").mkdir(parents=True)
    anchor_ids = [
        "10,10,20,20_TCGA-AA-0001-01Z-00-DX1_left-0_top-0_bottom-100_right-100",
        "30,30,40,40_TCGA-AA-0001-01Z-00-DX1_left-0_top-0_bottom-100_right-100",
    ]
    master = pd.DataFrame(
        {
            "anchor_id": anchor_ids,
            "xmin": [10, 30],
            "ymin": [10, 30],
            "xmax": [20, 40],
            "ymax": [20, 40],
            "JP.1": ["tumor", "tumor"],
            "SP.1": ["lymphocyte", "tumor"],
            "SP.2": ["plasma_cell", "mitotic_figure"],
            "JP.2": ["undetected", "undetected"],
            "EM_inferred_label_Ps": ["tumor", "lymphocyte"],
        }
    )
    master_path = contour_root / "master.csv"
    master.to_csv(master_path, index=False)
    pd.DataFrame({"anchor_id": anchor_ids}).to_csv(
        contour_root / "contours" / "ANCHFOV-0_example.csv", index=False
    )
    scored_frame = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "input_annotator": ["JP.1", "JP.1"],
            "anchor_fov_id": ["ANCHFOV-0", "ANCHFOV-0"],
            "observed_label": [0, 0],
            "absolute_xmin": [10, 30],
            "absolute_ymin": [10, 30],
            "absolute_xmax": [20, 40],
            "absolute_ymax": [20, 40],
            "risk_aanca": [0.9, 0.1],
        }
    )
    scored = ScoredInputData(
        scored_manifest=scored_frame,
        probabilities=np.asarray([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]]),
        fold_ids=np.asarray([0, 1]),
        training_groups_by_fold={0: ("b",), 1: ("a",)},
        fold_evidence=(),
        risk_metadata={},
        all_models_converged=True,
    )
    config = {
        "data": {
            "hidden_reference_table": str(master_path.relative_to(tmp_path)),
            "anchor_iou_threshold": 0.25,
        },
        "labels": {"minimum_independent_mappable_votes": 2},
    }
    attached, evidence, _ = attach_hidden_reference(
        scored,
        tmp_path,
        annotator="JP.1",
        config=config,
    )
    assert attached["reference_status"].tolist() == [
        "consensus_disagree",
        "consensus_agree",
    ]
    assert evidence["aggregate_p_truth_fields_read"] is False
    assert evidence["input_annotator_removed_from_reference"] is True
    assert all("JP.1" not in json.loads(value) for value in attached["reference_voter_ids_json"])


def test_verifier_recomputes_metrics_without_retraining(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config, config_sha = load_frozen_independent_pathologist_config(root)
    frame = _synthetic_scored_frame(300)
    metrics, queues, curve = evaluate_scored_reference(frame, config=config)
    frame["selected_primary_aanca"] = frame["sample_id"].isin(
        queues["budgets"]["0.05"]["aanca_sample_ids"]
    )
    random_ids = {
        sample_id
        for record in queues["budgets"]["0.05"]["matched_random"]
        for sample_id in record["sample_ids"]
    }
    frame["selected_in_any_primary_random_repeat"] = frame["sample_id"].isin(random_ids)
    atomic_write_text(tmp_path / "scored_samples.csv", frame.to_csv(index=False))
    atomic_write_json(tmp_path / "matched_random_queues.json", queues)
    atomic_write_text(tmp_path / "enrichment_curve.csv", curve.to_csv(index=False))
    atomic_write_json(
        tmp_path / "results.json",
        {
            "study_id": config["study_id"],
            "config_sha256": config_sha,
            "metrics": metrics,
        },
    )
    atomic_write_json(tmp_path / "artifact_manifest.json", module._artifact_manifest(tmp_path))

    verification = verify_independent_pathologist_artifacts(root, output_directory=tmp_path)

    assert verification["verified"] is True
    assert verification["metrics_recomputed_without_retraining"] is True
