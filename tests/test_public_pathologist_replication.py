from __future__ import annotations

import copy
import hashlib
import io
import json
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import histo_audit.external_validation.public_pathologist_replication as module
from histo_audit.external_validation.nucls_independent_pathologist import InputPreparedData
from histo_audit.external_validation.public_pathologist_replication import (
    attach_midogpp_hidden_reference,
    attach_riva_hidden_reference,
    evaluate_attached_dataset,
    load_frozen_public_replication_config,
    map_midogpp_class,
    map_riva_class,
    riva_smear_group,
    score_public_input,
    select_midogpp_images,
)


def test_frozen_public_config_and_class_mappings() -> None:
    root = Path(__file__).resolve().parents[1]
    config, digest = load_frozen_public_replication_config(root)
    assert digest == module.EXPECTED_CONFIG_SHA256
    assert config["aanca_reference_association_inspected_at_freeze"] is False
    assert map_riva_class("INFL") == (0, "non_lesion")
    assert map_riva_class("LSIL") == (1, "low_grade_or_equivocal")
    assert map_riva_class("SCC") == (2, "high_grade_or_malignant")
    assert map_riva_class("CA") == (2, "high_grade_or_malignant")
    assert map_midogpp_class(1) == (0, "mitotic_figure")
    assert map_midogpp_class(2) == (1, "not_mitotic_figure")
    assert riva_smear_group("HSIL_LSIL_12_7.png") == "HSIL_LSIL_12"


def test_midogpp_hash_selection_is_label_independent_and_deterministic() -> None:
    metadata = [
        {"id": index, "file_name": f"{index:03d}.tiff", "tumor_type": tumor}
        for tumor in ("a", "b")
        for index in range(1 if tumor == "a" else 101, 11 if tumor == "a" else 111)
    ]
    available = {int(item["id"]) for item in metadata}
    first = select_midogpp_images(metadata, available, salt="frozen", per_tumor_type=3)
    second = select_midogpp_images(
        list(reversed(metadata)), available, salt="frozen", per_tumor_type=3
    )
    assert first == second
    assert len(first) == 6
    assert {item["tumor_type"] for item in first} == {"a", "b"}
    assert all("label" not in item and "category" not in item for item in first)


def test_figshare_authority_deduplicates_identical_pages_and_rejects_conflicts() -> None:
    authorities: dict[str, dict[str, object]] = {}
    first = {
        "article_id": 2,
        "file_id": 20,
        "file_name": "001.tiff",
        "size_bytes": 100,
        "download_url": "https://example.test/file/20",
        "supplied_md5": "abc",
    }
    duplicate = {**first, "article_id": 1, "file_id": 10}
    module._register_figshare_authority(authorities, first)
    module._register_figshare_authority(authorities, duplicate)
    assert authorities["001.tiff"]["article_id"] == 1
    with pytest.raises(RuntimeError, match="conflicting MIDOG"):
        module._register_figshare_authority(authorities, {**duplicate, "size_bytes": 101})


def test_figshare_json_fetch_retries_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 60
        if calls == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "rate limited",
                {"Retry-After": "0"},
                None,
            )
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    assert module._fetch_json("https://example.test/data") == {"ok": True}
    assert calls == 2


def test_dynamic_public_scoring_is_group_safe() -> None:
    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    embeddings = np.zeros((120, 1024), dtype=np.float64)
    class_order = ("a", "b", "c")
    for index in range(120):
        label = index % 3
        group = f"group-{index // 12:02d}"
        rows.append(
            {
                "sample_id": f"sample-{index:04d}",
                "group_id": group,
                "observed_label": label,
                "observed_class": class_order[label],
            }
        )
        embeddings[index, label] = 2.0
        embeddings[index, 3:12] = rng.normal(0.0, 0.1, size=9)
    prepared = InputPreparedData(
        manifest=pd.DataFrame.from_records(rows),
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
    scored = score_public_input(prepared, embeddings, config=config, class_order=class_order)
    assert scored.all_models_converged is True
    assert scored.risk_metadata["reference_loaded_during_scoring"] is False
    for fold in scored.fold_evidence:
        assert not set(fold["training_groups"]).intersection(fold["held_out_groups"])
    for _, row in scored.scored_manifest.iterrows():
        assert row["group_id"] not in set(json.loads(row["neighbour_groups_json"]))


def _base_scored(rotation: str, raw_labels: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"{rotation}-a", f"{rotation}-b"],
            "source_annotation_id": ["1", "2"],
            "input_annotator": [rotation, rotation],
            "image_id": ["A_1_1", "A_1_1"],
            "group_id": ["A_1", "A_1"],
            "observed_raw_label": list(raw_labels),
            "observed_label": [map_riva_class(value)[0] for value in raw_labels],
            "observed_class": [map_riva_class(value)[1] for value in raw_labels],
            "point_x_percent": [10.0, 20.0],
            "point_y_percent": [10.0, 20.0],
            "proposed_transition": ["x", "x"],
            "risk_aanca": [0.9, 0.1],
        }
    )


def test_riva_reference_uses_raw_rows_and_removes_input(tmp_path: Path) -> None:
    cluster = pd.DataFrame(
        [
            ("A_1_1.png", 10.0, 10.0, 13, "LSIL", 1.0),
            ("A_1_1.png", 10.1, 10.1, 12, "LSIL", 1.0),
            ("A_1_1.png", 9.9, 9.9, 11, "ASCUS", 1.0),
            ("A_1_1.png", 20.0, 20.0, 13, "HSIL", 2.0),
            ("A_1_1.png", 20.1, 20.1, 12, "NILM", 2.0),
            ("A_1_1.png", 19.9, 19.9, 11, "NILM", 2.0),
        ],
        columns=(
            "image_filename",
            "nucleus_x",
            "nucleus_y",
            "annotator_id",
            "class_bethesda",
            "cluster_idx",
        ),
    )
    cluster_path = tmp_path / "cluster.csv"
    cluster.to_csv(cluster_path, index=False)
    root = Path(__file__).resolve().parents[1]
    config, _ = load_frozen_public_replication_config(root)
    config = copy.deepcopy(config)
    config["datasets"]["riva"]["cluster_table"] = "cluster.csv"
    config["datasets"]["riva"]["cluster_table_sha256"] = module.sha256_file(cluster_path)
    frames = {rotation: _base_scored(rotation, ("LSIL", "HSIL")) for rotation in ("annotator_1",)}
    config["datasets"]["riva"]["rotations"] = ["annotator_1"]
    attached, evidence, _ = attach_riva_hidden_reference(frames, tmp_path, config=config)
    output = attached["annotator_1"]
    assert output["reference_status"].tolist() == ["consensus_agree", "consensus_disagree"]
    assert evidence["official_released_majority_label_read"] is False
    assert all(
        "official_13" not in json.loads(value) for value in output["reference_voter_ids_json"]
    )


def test_midogpp_pairwise_reference_ignores_final_and_adjudicator(tmp_path: Path) -> None:
    payload = {
        "annotations": [
            {"id": 1, "labels": [1, 2, 2], "category_id": 2},
            {"id": 2, "labels": [2, 2], "category_id": 2},
        ]
    }
    source = tmp_path / "MIDOG++.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    config, _ = load_frozen_public_replication_config(root)
    config = copy.deepcopy(config)
    settings = config["datasets"]["midogpp"]
    settings["annotations"] = "MIDOG++.json"
    settings["annotations_sha256"] = module.sha256_file(source)
    settings["annotations_md5"] = hashlib.md5(
        source.read_bytes(), usedforsecurity=False
    ).hexdigest()
    frames: dict[str, pd.DataFrame] = {}
    for rotation, labels in {"expert_1": (1, 2), "expert_2": (2, 2)}.items():
        frames[rotation] = pd.DataFrame(
            {
                "sample_id": [f"{rotation}-1", f"{rotation}-2"],
                "source_annotation_id": ["1", "2"],
                "input_annotator": [rotation, rotation],
                "observed_label": [map_midogpp_class(value)[0] for value in labels],
                "observed_class": [map_midogpp_class(value)[1] for value in labels],
            }
        )
    attached, evidence, _ = attach_midogpp_hidden_reference(frames, tmp_path, config=config)
    assert attached["expert_1"]["reference_status"].tolist() == [
        "consensus_disagree",
        "consensus_agree",
    ]
    assert attached["expert_2"]["reference_status"].tolist() == [
        "consensus_disagree",
        "consensus_agree",
    ]
    assert evidence["adjudicator_label_used"] is False
    assert evidence["final_category_used"] is False
    assert "category_id" not in attached["expert_1"]


def _synthetic_attached(rotation: str, count: int = 600) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(count):
        group = f"group-{index % 20:02d}"
        observed = module.MIDOGPP_CLASS_ORDER[index % 2]
        proposed = module.MIDOGPP_CLASS_ORDER[(index // 2) % 2]
        event = index % 13 == 0 or index % 19 == 0
        risk = 1.0 - index / count if event else 0.4 - index / (2 * count)
        rows.append(
            {
                "sample_id": f"{rotation}-{index:04d}",
                "input_annotator": rotation,
                "group_id": group,
                "observed_class": observed,
                "proposed_transition": f"{observed}->{proposed}",
                "risk_aanca": risk,
                "binary_reference_eligible": True,
                "independent_consensus_disagreement": event,
            }
        )
    return pd.DataFrame.from_records(rows)


def test_aggregate_evaluation_is_deterministic_and_group_bootstrapped() -> None:
    root = Path(__file__).resolve().parents[1]
    frozen, _ = load_frozen_public_replication_config(root)
    config = copy.deepcopy(frozen)
    config["evaluation"]["matched_random_repetitions"] = 5
    config["evaluation"]["bootstrap_iterations"] = 100
    attached = {rotation: _synthetic_attached(rotation) for rotation in ("expert_1", "expert_2")}
    first = evaluate_attached_dataset(attached, dataset="midogpp", config=config)
    second = evaluate_attached_dataset(attached, dataset="midogpp", config=config)
    assert first[0] == second[0]
    assert first[1] == second[1]
    pd.testing.assert_frame_equal(first[2], second[2])
    assert first[0]["primary"]["bootstrap"]["unique_group_count"] == 20
    assert first[0]["primary"]["reviewed_rotation_rows"] == 60
    assert first[0]["primary"]["exact_equal_budget"] is True
