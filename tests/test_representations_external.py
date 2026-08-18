from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from histo_audit.external_validation import build_blinded_review_package
from histo_audit.models import FrozenEmbeddingMLPClassifier, FrozenEmbeddingMLPConfig
from histo_audit.representations import (
    PathologyEncoderCandidate,
    audit_pathology_encoder_availability,
    build_engineered_feature_set,
    load_embedding_cache,
    run_resnet18_embedding_smoke,
    save_embedding_cache,
    unavailable_optional_pathology_cache_provenance,
)


def _tiny_rgb_masks() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:32, :32]
    images = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    images[..., 0] = np.clip(20 + 4 * x, 0, 255)
    images[..., 1] = np.clip(30 + 3 * y, 0, 255)
    images[..., 2] = 90
    images[1] = np.flip(images[1], axis=1)
    images[2] = np.flip(images[2], axis=0)
    masks = np.stack(
        (
            (x - 12) ** 2 + (y - 15) ** 2 <= 6**2,
            ((x - 18) / 5) ** 2 + ((y - 15) / 8) ** 2 <= 1,
            (x >= 9) & (x < 22) & (y >= 10) & (y < 21),
        )
    )
    return images, masks


def test_engineered_features_are_deterministic_named_and_complete() -> None:
    images, masks = _tiny_rgb_masks()
    first = build_engineered_feature_set(images, masks)
    second = build_engineered_feature_set(images, masks)
    first.validate(expected_samples=3)
    np.testing.assert_array_equal(first.values, second.values)
    assert first.names == second.names
    assert first.values.shape[1] > 300
    for family in ("morphology.", "colour.", "intensity.", "texture.", "hog.", "mask."):
        assert any(name.startswith(family) for name in first.names)
    assert "morphology.equivalent_diameter_pixels" in first.names
    assert "morphology.normalised_equivalent_diameter" in first.names
    assert "morphology.solidity" in first.names
    assert "morphology.filled_solidity" not in first.names
    assert "intensity.boundary_mean" in first.names
    assert "intensity.boundary_minus_context_mean" in first.names
    histogram_indices = [
        index for index, name in enumerate(first.names) if "target_red_histogram" in name
    ]
    np.testing.assert_allclose(first.values[:, histogram_indices].sum(axis=1), 1.0)


def test_engineered_solidity_uses_convex_hull_area() -> None:
    images = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    masks = np.zeros((1, 8, 8), dtype=bool)
    masks[0, 1:6, 1] = True
    masks[0, 5, 1:6] = True

    features = build_engineered_feature_set(images, masks)
    solidity = features.values[0, features.names.index("morphology.solidity")]

    assert 0.0 < solidity < 1.0


def test_resnet_smoke_fails_closed_without_cached_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_hub = tmp_path / "empty-torch-hub"
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(empty_hub))
    result = run_resnet18_embedding_smoke(allow_weight_download=False, device="cpu")
    assert result.status == "blocked"
    assert result.blocker is not None
    assert "allow_weight_download=True" in result.blocker
    assert not empty_hub.exists()


def test_embedding_npz_metadata_and_checksum_round_trip(tmp_path: Path) -> None:
    embeddings = np.arange(3 * 512, dtype=np.float32).reshape(3, 512) / 100.0
    sample_ids = np.asarray(["a", "b", "c"], dtype=np.str_)
    cache, metadata_path, metadata = save_embedding_cache(
        tmp_path / "embeddings.npz",
        embeddings,
        sample_ids,
        {
            "encoder_name": "torchvision.resnet18",
            "weight_identifier": "ResNet18_Weights.IMAGENET1K_V1",
            "preprocessing": {"api": "official weight transform"},
            "dtype": "float32",
            "extraction_seconds": 0.01,
            "versions": {"torch": torch.__version__},
        },
    )
    assert cache.is_file() and metadata_path.is_file()
    assert len(metadata["cache_npz_sha256"]) == 64
    loaded = load_embedding_cache(cache)
    np.testing.assert_array_equal(loaded.embeddings, embeddings)
    np.testing.assert_array_equal(loaded.sample_ids, sample_ids)
    assert loaded.metadata["embeddings_sha256"] == metadata["embeddings_sha256"]


def test_pathology_priority_audit_selects_first_fully_verified_candidate(tmp_path: Path) -> None:
    weights = tmp_path / "verified-weights.bin"
    weights.write_bytes(b"test-only-local-weight-evidence")
    candidates = [
        PathologyEncoderCandidate(name="priority-blocked", priority=1),
        PathologyEncoderCandidate(
            name="priority-available",
            priority=2,
            original_source="https://example.invalid/original-source-record",
            source_verified=True,
            licence="test licence evidence",
            licence_verified=True,
            weight_identifier="test-weight-v1",
            weights_path=weights,
            authentication_status="not_required",
            preprocessing="test preprocessing record",
            preprocessing_verified=True,
            hardware_fit_status="verified_fit",
            intended_use="test representation use",
            intended_use_verified=True,
            embedding_smoke_passed=True,
        ),
    ]
    output = tmp_path / "pathology_audit.json"
    audit = audit_pathology_encoder_availability(candidates, output_path=output)
    assert audit.status == "available"
    assert audit.selected_encoder == "priority-available"
    assert audit.records[0].status == "blocked"
    assert audit.records[1].status == "selected"
    assert audit.records[1].weights_sha256 is not None
    assert (
        json.loads(output.read_text(encoding="utf-8"))["selected_encoder"] == audit.selected_encoder
    )


def test_blocked_pathology_audit_publishes_reproducible_unavailable_provenance(
    tmp_path: Path,
) -> None:
    output = tmp_path / "blocked-pathology.json"
    sample_sha = "a" * 64
    manifest_sha = "b" * 64
    audit = audit_pathology_encoder_availability(
        output_path=output,
        sample_order_sha256=sample_sha,
        dataset_manifest_sha256=manifest_sha,
    )
    expected = unavailable_optional_pathology_cache_provenance(
        sample_order_sha256=sample_sha,
        dataset_manifest_sha256=manifest_sha,
    )
    assert audit.status == "blocked"
    assert audit.primary_cache_provenance == expected
    assert json.loads(output.read_text(encoding="utf-8"))["primary_cache_provenance"] == expected
    assert expected["status"] == "unavailable_optional"
    assert expected["encoder_implementation_sha256"] is None
    assert expected["weights_sha256"] is None
    assert expected["preprocessing_sha256"] is None
    assert expected["cache_file_sha256"] is None
    assert len(expected["cache_recipe_sha256"]) == 64


def test_pathology_unavailable_binding_rejects_partial_or_available_evidence(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must be supplied together"):
        audit_pathology_encoder_availability(sample_order_sha256="a" * 64)

    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"verified")
    available = PathologyEncoderCandidate(
        name="available",
        priority=1,
        original_source="https://example.invalid/source",
        source_verified=True,
        licence="verified licence",
        licence_verified=True,
        weight_identifier="weights-v1",
        weights_path=weights,
        authentication_status="not_required",
        preprocessing="verified preprocessing",
        preprocessing_verified=True,
        hardware_fit_status="verified_fit",
        intended_use="research representation",
        intended_use_verified=True,
        embedding_smoke_passed=True,
    )
    with pytest.raises(ValueError, match="cannot describe an available"):
        audit_pathology_encoder_availability(
            [available],
            sample_order_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
        )


def test_frozen_embedding_mlp_is_deterministic_and_resumes(tmp_path: Path) -> None:
    rng = np.random.default_rng(22)
    embeddings = np.concatenate(
        (
            rng.normal(-1.0, 0.25, size=(18, 6)),
            rng.normal(1.0, 0.25, size=(18, 6)),
        )
    ).astype(np.float32)
    observed_labels = np.concatenate((np.zeros(18, dtype=np.int64), np.ones(18, dtype=np.int64)))
    full_config = FrozenEmbeddingMLPConfig(
        hidden_dimensions=(10,),
        dropout=0.0,
        epochs=4,
        batch_size=9,
        learning_rate=5e-3,
        seed=17,
        device="cpu",
    )
    uninterrupted = FrozenEmbeddingMLPClassifier(full_config).fit(embeddings, observed_labels)
    repeated = FrozenEmbeddingMLPClassifier(full_config).fit(embeddings, observed_labels)
    np.testing.assert_array_equal(
        uninterrupted.predict_proba(embeddings), repeated.predict_proba(embeddings)
    )

    checkpoint = tmp_path / "mlp-checkpoint.pt"
    partial_config = FrozenEmbeddingMLPConfig(
        hidden_dimensions=(10,),
        dropout=0.0,
        epochs=2,
        batch_size=9,
        learning_rate=5e-3,
        seed=17,
        device="cpu",
    )
    FrozenEmbeddingMLPClassifier(partial_config).fit(
        embeddings, observed_labels, checkpoint_path=checkpoint
    )
    resumed = FrozenEmbeddingMLPClassifier(full_config).fit(
        embeddings,
        observed_labels,
        checkpoint_path=checkpoint,
        resume=True,
    )
    assert resumed.completed_epochs_ == 4
    np.testing.assert_allclose(
        uninterrupted.predict_proba(embeddings),
        resumed.predict_proba(embeddings),
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(resumed.predict_proba(embeddings).sum(axis=1), 1.0)


def test_review_package_is_exact_disjoint_anonymised_and_response_blank(tmp_path: Path) -> None:
    manifest_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    for index in range(8):
        image_path = tmp_path / f"secret-source-{index}.png"
        Image.fromarray(np.full((12, 12, 3), 20 + index * 10, dtype=np.uint8)).save(image_path)
        manifest_rows.append(
            {
                "sample_id": f"secret-sample-{index}",
                "observed_label": index % 2,
                "pre_corruption_label": 99,
                "model_suggestion": "hidden",
                "source_patch_path": str(image_path),
                "target_crop_path": str(image_path),
                "target_contour_path": str(image_path),
            }
        )
        ranking_rows.append(
            {
                "sample_id": f"secret-sample-{index}",
                "risk_score": float(8 - index),
                "suggested_class": 4,
            }
        )

    result = build_blinded_review_package(
        pd.DataFrame(manifest_rows),
        pd.DataFrame(ranking_rows),
        tmp_path / "reviewer-package",
        top_count=3,
        random_count=3,
        seed=42,
    )
    items = pd.read_csv(result.review_items_csv)
    key = pd.read_csv(result.private_unblinding_key_csv)
    responses = pd.read_csv(result.response_template_csv, keep_default_na=False)
    assert len(items) == len(key) == len(responses) == 6
    assert (key["selection_source"] == "top_ranked").sum() == 3
    assert (key["selection_source"] == "random").sum() == 3
    assert set(key.loc[key["selection_source"] == "top_ranked", "sample_id"]) == {
        "secret-sample-0",
        "secret-sample-1",
        "secret-sample-2",
    }
    assert set(items.columns).isdisjoint(
        {"sample_id", "selection_source", "risk_score", "model_suggestion", "pre_corruption_label"}
    )
    assert set(responses["response"]) == {""}
    assert set(responses["reviewer_id"]) == {""}
    reviewer_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            result.review_items_csv,
            result.response_template_csv,
            result.review_html,
            result.response_schema_json,
            result.package_metadata_json,
        )
    )
    assert "secret-sample" not in reviewer_text
    assert "secret-source" not in reviewer_text
    assert "model_suggestion" not in reviewer_text
    assert "pre_corruption_label" not in reviewer_text
    assert " checked" not in result.review_html.read_text(encoding="utf-8")
    assert not result.private_unblinding_key_csv.is_relative_to(result.package_directory)


def test_review_package_requires_all_three_visual_asset_roles(tmp_path: Path) -> None:
    image_path = tmp_path / "patch.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "sample-1",
                "observed_label": 0,
                "source_patch_path": str(image_path),
                "target_crop_path": str(image_path),
            },
            {
                "sample_id": "sample-2",
                "observed_label": 1,
                "source_patch_path": str(image_path),
                "target_crop_path": str(image_path),
            },
        ]
    )
    ranking = pd.DataFrame(
        [
            {"sample_id": "sample-1", "risk_score": 1.0},
            {"sample_id": "sample-2", "risk_score": 0.5},
        ]
    )

    with pytest.raises(ValueError, match="full patch, target crop, and exact target contour"):
        build_blinded_review_package(
            manifest,
            ranking,
            tmp_path / "review-package",
            top_count=1,
            random_count=1,
            seed=1,
        )
