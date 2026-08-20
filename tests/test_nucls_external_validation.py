from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from histo_audit.evaluation.restoration import classification_metrics
from histo_audit.external_validation.nucls import (
    CLASS_ORDER,
    _group_confusion_tensor,
    _macro_f1_from_confusions,
    load_frozen_nucls_config,
    prepare_nucls_subset,
)


def _write_fixture(root: Path, *, unknown_label: bool = False) -> tuple[Path, Path]:
    np_root = root / "np_truth"
    p_root = root / "p_truth"
    (np_root / "contours").mkdir(parents=True)
    (np_root / "rgbs").mkdir()
    p_root.mkdir()
    master_rows: list[dict[str, str]] = []
    raw_labels = ("tumor", "fibroblast", "lymphocyte")
    for patient_index in range(5):
        slide = f"TCGA-A{patient_index}-A{patient_index:03d}-01Z-00-DX1"
        stem = f"ANCHFOV-{patient_index}_{slide}_left-0_top-0_bottom-96_right-96"
        image = np.full((96, 96, 3), 80 + patient_index, dtype=np.uint8)
        Image.fromarray(image).save(np_root / "rgbs" / f"{stem}.png")
        contour_rows: list[dict[str, object]] = []
        for class_index, raw_label in enumerate(raw_labels):
            anchor_id = f"{20 + class_index},20,{30 + class_index},30_{slide}_left-0_top-0_bottom-96_right-96"
            exported_label = "mystery_label" if unknown_label and not master_rows else raw_label
            contour_rows.append(
                {
                    "anchor_id": anchor_id,
                    "group": exported_label,
                    "xmin": 20 + class_index,
                    "ymin": 20,
                    "xmax": 30 + class_index,
                    "ymax": 30,
                }
            )
            master_rows.append(
                {
                    "anchor_id": anchor_id,
                    "EM_inferred_label_NPs": exported_label,
                    "EM_inferred_label_Ps": (
                        "fibroblast" if patient_index == 0 and class_index == 0 else raw_label
                    ),
                }
            )
        pd.DataFrame(contour_rows).to_csv(np_root / "contours" / f"{stem}.csv", index=False)
    pd.DataFrame(master_rows).to_csv(
        p_root / "v3.1_final_anchors_U-control_Ps_AreTruth.csv", index=False
    )
    return np_root, p_root


def test_prepare_nucls_subset_pairs_exact_multirater_anchors(tmp_path: Path) -> None:
    np_root, p_root = _write_fixture(tmp_path)

    prepared = prepare_nucls_subset(
        np_root,
        p_root,
        subset_name="unbiased_control",
        crop_size=64,
    )

    assert len(prepared.manifest) == 15
    assert prepared.manifest["group_id"].nunique() == 5
    assert set(prepared.manifest["observed_label_name"]) == set(CLASS_ORDER)
    assert int(prepared.manifest["natural_disagreement"].sum()) == 1
    assert prepared.crops.shape == (15, 64, 64, 3)
    assert prepared.crops.dtype == np.uint8
    assert prepared.exclusions == {}
    assert len(prepared.source_inventory) == 11
    assert {record["role"] for record in prepared.source_inventory} == {
        "np_contours",
        "np_rgb",
        "p_truth_master",
    }
    assert all(":" not in record["relative_path"] for record in prepared.source_inventory)
    assert prepared.manifest["np_contour_file"].str.startswith("contours/").all()
    assert prepared.manifest["np_rgb_file"].str.startswith("rgbs/").all()
    assert set(prepared.manifest["p_truth_master_file"]) == {
        "v3.1_final_anchors_U-control_Ps_AreTruth.csv"
    }
    assert not prepared.manifest["np_contour_file"].str.contains(":").any()


def test_prepare_nucls_subset_rejects_unknown_label(tmp_path: Path) -> None:
    np_root, p_root = _write_fixture(tmp_path, unknown_label=True)

    with pytest.raises(ValueError, match="unknown NuCLS raw label"):
        prepare_nucls_subset(np_root, p_root, subset_name="unbiased_control")


def test_group_confusion_bootstrap_uses_same_macro_f1_definition() -> None:
    reference = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    predicted = np.asarray([0, 1, 1, 0, 2, 2], dtype=np.int64)
    probabilities = np.eye(3, dtype=np.float64)[predicted]
    groups = ["a", "a", "a", "b", "b", "b"]

    tensor = _group_confusion_tensor(reference, probabilities, groups, ("a", "b"))
    from_tensor = float(_macro_f1_from_confusions(tensor[0].sum(axis=0))[0])
    from_public_metric = classification_metrics(
        reference, probabilities, class_order=(0, 1, 2)
    ).macro_f1

    assert from_tensor == pytest.approx(from_public_metric)


def test_repository_nucls_freeze_is_unchanged() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config, freeze = load_frozen_nucls_config(repository_root)

    assert config["study_id"] == "nucls_natural_label_external_validation_v1"
    assert freeze["outcome_tables_inspected_before_freeze"] is False
