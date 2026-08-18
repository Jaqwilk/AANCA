"""Strict schema-v2 representation-independence producer tests."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from histo_audit.experiment import representation_independence as independence_module
from histo_audit.experiment.representation_independence import (
    IndependenceAuditorInput,
    build_pannuke_representation_independence_artifact,
    build_representation_independence_payload,
    publish_representation_independence_artifact,
    validate_representation_independence_file,
    validate_representation_independence_payload,
)
from histo_audit.representations import (
    EmbeddingResult,
    EngineeredFeatureSet,
    save_embedding_cache,
    save_engineered_feature_cache,
    verify_frozen_cache_sidecar,
)


def _provenance(seed: str) -> dict[str, Any]:
    def digest(field: str) -> str:
        return hashlib.sha256(f"{seed}:{field}".encode()).hexdigest()

    return {
        "status": "available",
        "encoder_id": f"encoder-{seed}",
        "encoder_implementation_sha256": digest("implementation"),
        "weights_sha256": digest("weights"),
        "preprocessing_sha256": digest("preprocessing"),
        "sample_order_sha256": digest("sample-order"),
        "dataset_manifest_sha256": digest("manifest"),
        "cache_recipe_sha256": digest("recipe"),
        "cache_file_sha256": digest("cache-file"),
    }


def test_generic_builder_is_deterministic_strict_and_no_overwrite(tmp_path: Path) -> None:
    sample_ids = np.asarray([f"sample-{index}" for index in range(8)], dtype=np.str_)
    group_ids = np.asarray([f"group-{index // 2}" for index in range(8)], dtype=np.str_)
    generator = np.arange(24, dtype=np.float64).reshape(8, 3)
    engineered = np.column_stack((generator, np.ones((8, 2), dtype=np.float64)))
    imagenet = np.arange(32, dtype=np.float32).reshape(8, 4)
    payload = build_representation_independence_payload(
        generator_features=generator,
        audit_sample_ids=sample_ids,
        audit_group_ids=group_ids,
        generator_cache_provenance=_provenance("a"),
        auditors=(
            IndependenceAuditorInput(
                representation_id="engineered_target_features",
                family="engineered",
                features=engineered,
                cache_provenance=_provenance("a"),
                matrix_decision="not_independent",
                matrix_reason="Contains the exact generator columns.",
            ),
            IndependenceAuditorInput(
                representation_id="imagenet_resnet18_context",
                family="imagenet",
                features=imagenet,
                cache_provenance=_provenance("b"),
                matrix_decision="verified_independent",
                matrix_reason="Does not consume the engineered generator vector.",
            ),
        ),
    )
    parsed = validate_representation_independence_payload(
        payload,
        expected_representation_ids=(
            "engineered_target_features",
            "imagenet_resnet18_context",
        ),
    )
    assert parsed["engineered_target_features"].matrix_decision == "not_independent"
    assert parsed["imagenet_resnet18_context"].matrix_decision == "verified_independent"
    first = publish_representation_independence_artifact(
        payload,
        tmp_path / "first.json",
        expected_representation_ids=tuple(parsed),
        audit_sample_count=8,
        audit_group_count=4,
    )
    second = publish_representation_independence_artifact(
        payload,
        tmp_path / "second.json",
        expected_representation_ids=tuple(parsed),
        audit_sample_count=8,
        audit_group_count=4,
    )
    assert first.path.read_bytes() == second.path.read_bytes()
    assert first.file_sha256 == second.file_sha256
    assert first.entry_hashes == second.entry_hashes
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_representation_independence_artifact(
            payload,
            first.path,
            audit_sample_count=8,
            audit_group_count=4,
        )
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    (sealed / ".immutable.json").write_text("{}\n", encoding="utf-8")
    sealed_output = sealed / "nested" / "independence.json"
    with pytest.raises(PermissionError, match="sealed/immutable ancestor"):
        publish_representation_independence_artifact(
            payload,
            sealed_output,
            audit_sample_count=8,
            audit_group_count=4,
        )
    assert not sealed_output.exists()
    assert not sealed_output.parent.exists()


def test_validator_rejects_false_engineered_independence_and_hash_tampering() -> None:
    sample_ids = np.asarray(["s0", "s1"], dtype=np.str_)
    groups = np.asarray(["g0", "g1"], dtype=np.str_)
    with pytest.raises(ValueError, match="cannot be certified independent"):
        build_representation_independence_payload(
            generator_features=np.ones((2, 1), dtype=np.float64),
            audit_sample_ids=sample_ids,
            audit_group_ids=groups,
            generator_cache_provenance=_provenance("a"),
            auditors=(
                IndependenceAuditorInput(
                    representation_id="engineered_target_features",
                    family="engineered",
                    features=np.ones((2, 2), dtype=np.float64),
                    cache_provenance=_provenance("a"),
                    matrix_decision="verified_independent",
                    matrix_reason="false declaration",
                ),
            ),
        )

    with pytest.raises(ValueError, match="identical feature-space signatures or bytes"):
        build_representation_independence_payload(
            generator_features=np.ones((2, 1), dtype=np.float64),
            audit_sample_ids=sample_ids,
            audit_group_ids=groups,
            generator_cache_provenance=_provenance("a"),
            auditors=(
                IndependenceAuditorInput(
                    representation_id="imagenet_resnet18_context",
                    family="imagenet",
                    features=np.ones((2, 1), dtype=np.float64),
                    cache_provenance=_provenance("b"),
                    matrix_decision="verified_independent",
                    matrix_reason="false byte-identical declaration",
                ),
            ),
        )

    valid = build_representation_independence_payload(
        generator_features=np.ones((2, 1), dtype=np.float64),
        audit_sample_ids=sample_ids,
        audit_group_ids=groups,
        generator_cache_provenance=_provenance("a"),
        auditors=(
            IndependenceAuditorInput(
                representation_id="imagenet_resnet18_context",
                family="imagenet",
                features=np.zeros((2, 2), dtype=np.float32),
                cache_provenance=_provenance("b"),
                matrix_decision="verified_independent",
                matrix_reason="explicitly distinct input transformation",
            ),
        ),
    )
    tampered = copy.deepcopy(valid)
    tampered["entries"]["imagenet_resnet18_context"]["matrix_reason"] = "tampered"
    with pytest.raises(ValueError, match="canonical payload"):
        validate_representation_independence_payload(tampered)


def test_final_sha_failure_retracts_published_independence_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = build_representation_independence_payload(
        generator_features=np.ones((2, 1), dtype=np.float64),
        audit_sample_ids=np.asarray(["s0", "s1"], dtype=np.str_),
        audit_group_ids=np.asarray(["g0", "g1"], dtype=np.str_),
        generator_cache_provenance=_provenance("a"),
        auditors=(
            IndependenceAuditorInput(
                representation_id="imagenet_resnet18_context",
                family="imagenet",
                features=np.zeros((2, 2), dtype=np.float32),
                cache_provenance=_provenance("b"),
                matrix_decision="verified_independent",
                matrix_reason="Uses an explicitly distinct input transformation.",
            ),
        ),
    )
    destination = (tmp_path / "independence.json").resolve()
    original_sha256 = independence_module.sha256_file
    destination_calls = 0

    def fail_final_sha(path: str | Path) -> str:
        nonlocal destination_calls
        if Path(path).resolve() == destination:
            destination_calls += 1
            if destination_calls == 3:
                raise OSError("injected final independence SHA failure")
        return original_sha256(path)

    monkeypatch.setattr(independence_module, "sha256_file", fail_final_sha)
    with pytest.raises(OSError, match="injected final independence SHA failure"):
        publish_representation_independence_artifact(
            payload,
            destination,
            audit_sample_count=2,
            audit_group_count=2,
        )

    assert destination_calls == 3
    assert not destination.exists()


def _stage_cache_fixture(tmp_path: Path) -> Any:
    rng = np.random.default_rng(17)
    labels = np.tile(np.arange(5, dtype=np.int64), 7)
    n = len(labels)
    sample_ids = np.asarray([f"sample-{index:03d}" for index in range(n)], dtype=np.str_)
    group_ids = np.asarray([f"group-{index:03d}" for index in range(n)], dtype=np.str_)
    folds = np.concatenate(
        (
            np.ones(15, dtype=np.int16),
            np.full(15, 2, dtype=np.int16),
            np.full(5, 3, dtype=np.int16),
        )
    )
    engineered = EngineeredFeatureSet(
        values=rng.normal(size=(n, 4)).astype(np.float64),
        names=(
            "morphology.area_fraction",
            "morphology.eccentricity",
            "intensity.target_mean",
            "texture.glcm_contrast_mean",
        ),
    )
    engineered_path, _, _ = save_engineered_feature_cache(
        engineered,
        sample_ids,
        tmp_path / "engineered.npz",
        manifest_sha256="a" * 64,
        raw_inventory_sha256="b" * 64,
    )
    context_values = rng.normal(size=(n, 512)).astype(np.float32)
    highlighted_values = rng.normal(size=(n, 512)).astype(np.float32)
    context_path, _, context_metadata = save_embedding_cache(
        tmp_path / "context.npz",
        context_values,
        sample_ids,
        {
            "input_variant": "rgb",
            "manifest_sha256": "a" * 64,
            "raw_inventory_sha256": "b" * 64,
            "weight_sha256": "c" * 64,
        },
    )
    highlighted_path, _, highlighted_metadata = save_embedding_cache(
        tmp_path / "highlighted.npz",
        highlighted_values,
        sample_ids,
        {
            "input_variant": "target_highlighted_rgb",
            "manifest_sha256": "a" * 64,
            "raw_inventory_sha256": "b" * 64,
            "weight_sha256": "c" * 64,
        },
    )
    context = EmbeddingResult(
        embeddings=context_values,
        sample_ids=sample_ids,
        metadata=context_metadata,
        cache_path=context_path,
        metadata_path=context_path.with_suffix(".npz.metadata.json"),
    )
    highlighted = EmbeddingResult(
        embeddings=highlighted_values,
        sample_ids=sample_ids,
        metadata=highlighted_metadata,
        cache_path=highlighted_path,
        metadata_path=highlighted_path.with_suffix(".npz.metadata.json"),
    )
    crops = SimpleNamespace(
        sample_ids=sample_ids,
        group_ids=group_ids,
        official_folds=folds,
        pre_corruption_labels=labels,
    )
    return SimpleNamespace(
        crops=crops,
        engineered=engineered,
        embeddings=highlighted,
        crop_cache_path=tmp_path / "unused-crop.npz",
        crop_metadata_path=tmp_path / "unused-crop.npz.metadata.json",
        engineered_cache_path=engineered_path,
        engineered_metadata_path=engineered_path.with_suffix(".npz.metadata.json"),
        context_embeddings=context,
        context_morphometrics=None,
    )


def test_pannuke_builder_excludes_final_fold_and_binds_real_sidecars(tmp_path: Path) -> None:
    artifacts = _stage_cache_fixture(tmp_path)
    first = build_pannuke_representation_independence_artifact(
        cast(Any, artifacts),
        tmp_path / "independence-first.json",
        class_order=(0, 1, 2, 3, 4),
        development_official_folds=(1, 2),
        final_test_fold=3,
        reference_validation_fraction_groups=0.2,
        split_seed=227,
    )
    parsed = validate_representation_independence_file(
        first.path,
        expected_representation_ids=(
            "engineered_target_features",
            "imagenet_resnet18_context",
            "imagenet_resnet18_highlighted",
        ),
    )
    assert parsed["engineered_target_features"].matrix_decision == "not_independent"
    assert parsed["imagenet_resnet18_context"].matrix_decision == "verified_independent"
    assert parsed["imagenet_resnet18_highlighted"].matrix_decision == "verified_independent"
    for path in (
        artifacts.engineered_cache_path,
        artifacts.context_embeddings.cache_path,
        artifacts.embeddings.cache_path,
    ):
        metadata = verify_frozen_cache_sidecar(path).metadata
        assert metadata["primary_cache_provenance"]

    final_mask = artifacts.crops.official_folds == 3
    artifacts.engineered.values[final_mask] += 1000.0
    artifacts.context_embeddings.embeddings[final_mask] += 1000.0
    artifacts.embeddings.embeddings[final_mask] += 1000.0
    second = build_pannuke_representation_independence_artifact(
        cast(Any, artifacts),
        tmp_path / "independence-second.json",
        class_order=(0, 1, 2, 3, 4),
        development_official_folds=(1, 2),
        final_test_fold=3,
        reference_validation_fraction_groups=0.2,
        split_seed=227,
    )
    assert first.path.read_bytes() == second.path.read_bytes()
    assert first.file_sha256 == second.file_sha256
