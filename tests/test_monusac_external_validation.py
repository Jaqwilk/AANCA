from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from histo_audit.external_validation.monusac import (
    CLASS_ORDER,
    MoNuSACPreparedData,
    _fixed_crop,
    load_frozen_monusac_config,
    prepare_monusac_split,
    run_monusac_controlled_external,
)


def test_local_monusac_crop_matches_full_reflection_padding() -> None:
    image = np.arange(70 * 80 * 3, dtype=np.uint16).reshape(70, 80, 3).astype(np.uint8)
    size = 64
    half = size // 2
    padded = np.pad(image, ((half, half), (half, half), (0, 0)), mode="reflect")
    for centre_x, centre_y in ((0, 0), (1, 68), (79, 69), (40, 35)):
        expected = padded[
            centre_y : centre_y + size,
            centre_x : centre_x + size,
        ]
        actual = _fixed_crop(
            image,
            centre_x=centre_x,
            centre_y=centre_y,
            size=size,
        )
        assert np.array_equal(actual, expected)


def _xml_text(*, include_ambiguous: bool = False) -> str:
    names = list(CLASS_ORDER)
    if include_ambiguous:
        names.append("Ambiguous")
    annotations = []
    for annotation_id, name in enumerate(names, start=1):
        offset = 10 + annotation_id * 8
        annotations.append(
            f"""
  <Annotation Id="{annotation_id}" Name="">
    <Attributes><Attribute Name="{name}" Id="0" Value=""/></Attributes>
    <Regions><Region Id="1"><Vertices>
      <Vertex X="{offset}" Y="{offset}" Z="0"/>
      <Vertex X="{offset + 4}" Y="{offset}" Z="0"/>
      <Vertex X="{offset + 4}" Y="{offset + 4}" Z="0"/>
      <Vertex X="{offset}" Y="{offset + 4}" Z="0"/>
    </Vertices></Region></Regions>
  </Annotation>"""
        )
    return "<Annotations>" + "".join(annotations) + "</Annotations>"


def test_prepare_monusac_split_uses_patient_groups_and_excludes_ambiguous(tmp_path: Path) -> None:
    patients = (
        "TCGA-55-1594",
        "TCGA-69-7760",
        "TCGA-69-A59K",
        "TCGA-73-4668",
        "TCGA-78-7220",
        "TCGA-86-7713",
    )
    root = tmp_path / "official"
    for patient in patients:
        folder = root / f"{patient}-01Z-00-DX1"
        folder.mkdir(parents=True)
        stem = f"{patient}-01Z-00-DX1_001"
        image = np.full((96, 96, 3), 127, dtype=np.uint8)
        tifffile.imwrite(folder / f"{stem}.tif", image)
        (folder / f"{stem}.xml").write_text(_xml_text(include_ambiguous=True), encoding="utf-8")

    prepared = prepare_monusac_split(
        root,
        split="train",
        crop_size=64,
        excluded_patients=(patients[-1],),
    )

    assert len(prepared.manifest) == 5 * len(CLASS_ORDER)
    assert prepared.manifest["group_id"].nunique() == 5
    assert prepared.crops.shape == (20, 64, 64, 3)
    assert prepared.exclusions["ambiguous_region_excluded"] == 5
    assert prepared.exclusions["overlap_patient_excluded_from_development"] == 5
    assert set(prepared.manifest["reference_label_name"]) == set(CLASS_ORDER)


def _prepared(
    split: str, *, group_prefix: str, group_count: int
) -> tuple[MoNuSACPreparedData, np.ndarray]:
    rng = np.random.default_rng(17 if split == "train" else 19)
    labels = np.tile(np.arange(4, dtype=np.int64), group_count * 2)
    groups = np.repeat([f"{group_prefix}-{index}" for index in range(group_count)], 8)
    sample_ids = [f"{split}-{index:04d}" for index in range(len(labels))]
    features = rng.normal(0.0, 0.05, size=(len(labels), 512))
    features[np.arange(len(labels)), labels] += 2.0
    manifest = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "split": split,
            "patient_id": groups,
            "group_id": groups,
            "organ": ["Lung"] * len(labels),
            "image_id": sample_ids,
            "region_authority": ["1:1"] * len(labels),
            "reference_label": labels,
            "reference_label_name": [CLASS_ORDER[value] for value in labels],
            "xmin": np.zeros(len(labels)),
            "ymin": np.zeros(len(labels)),
            "xmax": np.ones(len(labels)),
            "ymax": np.ones(len(labels)),
            "xml_file": ["fixture.xml"] * len(labels),
            "image_file": ["fixture.tif"] * len(labels),
        }
    )
    prepared = MoNuSACPreparedData(
        split=split,
        manifest=manifest,
        crops=np.empty((len(labels), 0, 0, 3), dtype=np.uint8),
        exclusions={},
        source_inventory=(),
        source_inventory_sha256="1" * 64,
        manifest_sha256="2" * 64 if split == "train" else "3" * 64,
        crops_sha256="4" * 64,
    )
    return prepared, features.astype(np.float32)


def test_frozen_monusac_runner_is_group_safe_and_preserves_source_labels() -> None:
    root = Path(__file__).resolve().parents[1]
    config, config_sha256 = load_frozen_monusac_config(root)
    local = copy.deepcopy(config)
    local["ranking"]["bootstrap_iterations"] = 20
    local["downstream"]["bootstrap_iterations"] = 20
    local["matched_random_control"]["repetitions"] = 3
    local["matched_random_control"]["fields"] = ["organ"]
    local["model"]["max_iter"] = 60
    local["balanced_quality_queue"].update(
        {
            "max_per_group_fraction_of_review_count": 1.0,
            "max_per_class_fraction_of_review_count": 1.0,
            "max_per_tissue_fraction_of_review_count": 1.0,
            "max_per_transition_fraction_of_review_count": 1.0,
            "minimum_cosine_distance": 0.0,
        }
    )
    train, train_features = _prepared("train", group_prefix="development", group_count=10)
    test, test_features = _prepared("test", group_prefix="final", group_count=5)
    source_labels = train.manifest["reference_label"].to_numpy(dtype=np.int64).copy()

    result, arrays = run_monusac_controlled_external(
        train,
        test,
        train_features,
        test_features,
        local,
        config_sha256=config_sha256,
    )

    assert result["replacement_project_or_v2"] is False
    assert result["controlled_corruption"]["source_annotations_modified"] is False
    assert result["dataset"]["train_patient_groups"] == 10
    assert result["dataset"]["test_patient_groups"] == 5
    assert np.array_equal(train.manifest["reference_label"], source_labels)
    assert arrays["oof_probabilities"].shape == (80, 4)
    assert result["queue_evidence"]["model_improvement_queue_available"] is False
    assert set(result["success_conditions"]) == set(local["success_requires_all"])
