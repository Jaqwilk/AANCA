from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from histo_audit.external_validation.review_package import build_blinded_review_package
from histo_audit.external_validation.validation import validate_blinded_review_package


def test_blinded_package_accepts_a_frozen_matched_selection_plan(tmp_path: Path) -> None:
    image = tmp_path / "asset.png"
    Image.new("RGB", (8, 8), color=(120, 40, 80)).save(image)
    manifest = pd.DataFrame(
        {
            "sample_id": [f"s{index}" for index in range(6)],
            "observed_label": [0, 0, 1, 1, 0, 1],
            "full_patch_path": [str(image)] * 6,
            "target_crop_path": [str(image)] * 6,
            "target_contour_path": [str(image)] * 6,
        }
    )
    ranking = pd.DataFrame(
        {
            "sample_id": [f"s{index}" for index in range(6)],
            "risk_score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        }
    )
    selection = pd.DataFrame(
        {
            "sample_id": ["s0", "s2", "s1", "s3"],
            "selection_source": ["top_ranked", "top_ranked", "random", "random"],
            "match_stratum": ["class-0", "class-1", "class-0", "class-1"],
        }
    )

    package = build_blinded_review_package(
        manifest,
        ranking,
        tmp_path / "package",
        top_count=2,
        random_count=2,
        seed=37,
        selection_plan=selection,
    )
    validation = validate_blinded_review_package(
        package.package_directory,
        private_unblinding_key_path=package.private_unblinding_key_csv,
    )
    metadata = json.loads(package.package_metadata_json.read_text(encoding="utf-8"))
    key = pd.read_csv(package.private_unblinding_key_csv)

    assert validation.valid is True
    assert validation.private_linkage_validated is True
    assert metadata["comparator_sampling"] == "predeclared_exact_matched_random"
    assert len(metadata["selection_plan_sha256"]) == 64
    assert set(key["sample_id"]) == {"s0", "s1", "s2", "s3"}
    assert set(key["selection_source"]) == {"top_ranked", "random"}
    assert set(key["matching_stratum"]) == {"class-0", "class-1"}


def test_blinded_package_rejects_an_unmatched_plan(tmp_path: Path) -> None:
    image = tmp_path / "asset.png"
    Image.new("RGB", (8, 8), color=(120, 40, 80)).save(image)
    manifest = pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2", "s3"],
            "observed_label": [0, 0, 1, 1],
            "full_patch_path": [str(image)] * 4,
            "target_crop_path": [str(image)] * 4,
            "target_contour_path": [str(image)] * 4,
        }
    )
    ranking = pd.DataFrame(
        {"sample_id": ["s0", "s1", "s2", "s3"], "risk_score": [0.9, 0.8, 0.7, 0.6]}
    )
    selection = pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2", "s3"],
            "selection_source": ["top_ranked", "top_ranked", "random", "random"],
            "match_stratum": ["a", "a", "a", "b"],
        }
    )

    with pytest.raises(ValueError, match="not one-to-one matched"):
        build_blinded_review_package(
            manifest,
            ranking,
            tmp_path / "package",
            top_count=2,
            random_count=2,
            seed=37,
            selection_plan=selection,
        )
