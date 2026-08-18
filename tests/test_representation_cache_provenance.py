"""Fail-closed tests for final frozen representation-cache provenance."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import histo_audit.representations.cache_provenance as cache_provenance_module
from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.representations import (
    EmbeddingResult,
    EngineeredFeatureSet,
    confirmatory_cache_provenance_record,
    load_embedding_cache,
    ordered_sample_ids_sha256,
    primary_cache_provenance_record,
    save_context_morphometrics_cache,
    save_embedding_cache,
    save_engineered_feature_cache,
    verify_frozen_cache_sidecar,
)

_WEIGHT_SHA = "c" * 64
_MANIFEST_SHA = "d" * 64
_INVENTORY_SHA = "e" * 64


def _embedding_metadata() -> dict[str, object]:
    return {
        "schema_version": 1,
        "representation_id": "imagenet_resnet18_context_embeddings",
        "input_variant": "rgb",
        "encoder_identifier": "resnet18_imagenet1k_v1",
        "encoder_name": "torchvision.resnet18",
        "encoder_frozen": True,
        "weight_identifier": "ResNet18_Weights.IMAGENET1K_V1",
        "weights_sha256": _WEIGHT_SHA,
        "weight_sha256": _WEIGHT_SHA,
        "manifest_sha256": _MANIFEST_SHA,
        "crop_manifest_sha256": _MANIFEST_SHA,
        "raw_inventory_sha256": _INVENTORY_SHA,
        "preprocessing_identifier": "torchvision_resnet18_imagenet1k_v1_official",
        "preprocessing": {
            "api": "official weight transform",
            "resize_size": [256],
            "crop_size": [224],
        },
        "encoder_metadata": {
            "architecture": "torchvision.models.resnet18",
            "frozen": True,
            "output_dimension": 512,
        },
        "encoder_implementation": {
            "module": "torchvision.models",
            "architecture": "resnet18",
            "source_file_sha256": "f" * 64,
        },
        "cache_recipe": {
            "identifier": "test_frozen_embedding_npz_v2",
            "output_dtype": "float32",
            "pickle_allowed": False,
        },
        "package_versions": {"numpy": np.__version__, "python": "test"},
        "provenance_scope": "stage_eligible",
        "extracted_at_utc": "2026-07-17T00:00:00.000Z",
    }


def _save_context(path: Path) -> tuple[EmbeddingResult, dict[str, object]]:
    values = np.arange(3 * 512, dtype=np.float32).reshape(3, 512) / 100.0
    sample_ids = np.asarray(["sample-c", "sample-a", "sample-b"], dtype=np.str_)
    cache, sidecar, metadata = save_embedding_cache(
        path,
        values,
        sample_ids,
        _embedding_metadata(),
    )
    result = EmbeddingResult(
        embeddings=values,
        sample_ids=sample_ids,
        metadata=metadata,
        cache_path=cache,
        metadata_path=sidecar,
    )
    result.validate()
    return result, metadata


def test_embedding_sidecar_projects_exact_contract_and_recomputes_order(tmp_path: Path) -> None:
    result, metadata = _save_context(tmp_path / "context.npz")
    verification = verify_frozen_cache_sidecar(result.cache_path)
    assert verification.metadata["sample_order_sha256"] == ordered_sample_ids_sha256(
        result.sample_ids
    )
    assert verification.metadata["sample_order_sha256"] == canonical_sha256(
        result.sample_ids.tolist()
    )
    assert verification.metadata["input_variant"] == "context_rgb"
    assert verification.metadata["cache_file_sha256"] == verification.cache_file_sha256
    assert verification.metadata["cache_npz_sha256"] == verification.cache_file_sha256
    assert verification.metadata["weight_sha256"] == verification.metadata["weights_sha256"]
    assert len(verification.sidecar_file_sha256) == 64

    record = confirmatory_cache_provenance_record(
        metadata,
        record_id="imagenet_context_embedding_cache",
        bind_sidecar_semantics=True,
    )
    assert set(record) == {
        "id",
        "representation_id",
        "status",
        "cache_file_sha256",
        "sidecar_semantic_sha256",
        "sample_order_sha256",
        "manifest_sha256",
        "encoder_identifier",
        "encoder_metadata_sha256",
        "weight_identifier",
        "weights_sha256",
        "preprocessing_identifier",
        "preprocessing_sha256",
        "input_variant",
    }
    assert record["cache_file_sha256"] is None
    assert record["sidecar_semantic_sha256"] == metadata["sidecar_semantic_sha256"]
    assert record["input_variant"] == "context_rgb"
    primary_record = primary_cache_provenance_record(metadata)
    assert set(primary_record) == {
        "status",
        "encoder_id",
        "encoder_implementation_sha256",
        "weights_sha256",
        "preprocessing_sha256",
        "sample_order_sha256",
        "dataset_manifest_sha256",
        "cache_recipe_sha256",
        "cache_file_sha256",
    }
    assert load_embedding_cache(result.cache_path).metadata["input_variant"] == "rgb"


def test_embedding_cache_rejects_overwrite_tamper_and_wrong_weights(tmp_path: Path) -> None:
    result, _ = _save_context(tmp_path / "context.npz")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _save_context(tmp_path / "context.npz")
    verify_frozen_cache_sidecar(result.cache_path, expected_weights_sha256=_WEIGHT_SHA)
    with pytest.raises(ValueError, match="weights_sha256 differs"):
        verify_frozen_cache_sidecar(result.cache_path, expected_weights_sha256="a" * 64)

    assert result.metadata_path is not None
    sidecar = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    sidecar["weight_identifier"] = "tampered-weight"
    result.metadata_path.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic checksum"):
        verify_frozen_cache_sidecar(result.cache_path)

    cache_tamper, _ = _save_context(tmp_path / "cache-tamper.npz")
    assert cache_tamper.cache_path is not None
    with cache_tamper.cache_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="cache file checksum"):
        verify_frozen_cache_sidecar(cache_tamper.cache_path)


def test_embedding_cache_rejects_nonfinite_before_writing(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.npz"
    values = np.zeros((2, 512), dtype=np.float32)
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        save_embedding_cache(
            path,
            values,
            np.asarray(["sample-a", "sample-b"], dtype=np.str_),
            _embedding_metadata(),
        )
    assert not path.exists()
    assert not path.with_suffix(".npz.metadata.json").exists()


def test_cache_rollback_preserves_foreign_replacement_after_sidecar_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "context.npz"
    sidecar = destination.with_suffix(".npz.metadata.json")
    foreign_content = b"foreign cache replacement\n"
    real_commit = cache_provenance_module._commit_no_overwrite
    commit_count = 0

    def replace_then_fail(
        temporary: Path,
        final_path: Path,
    ) -> cache_provenance_module._PublishedCacheFile:
        nonlocal commit_count
        commit_count += 1
        if commit_count == 1:
            publication = real_commit(temporary, final_path)
            final_path.unlink()
            final_path.write_bytes(foreign_content)
            return publication
        raise OSError("injected sidecar publication failure")

    monkeypatch.setattr(cache_provenance_module, "_commit_no_overwrite", replace_then_fail)

    with pytest.raises(
        RuntimeError,
        match=r"ownership changed.*foreign destination was preserved",
    ):
        _save_context(destination)

    assert commit_count == 2
    assert destination.read_bytes() == foreign_content
    assert not sidecar.exists()


def test_atomic_cache_publish_treats_broken_final_symlink_as_occupied(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "context.npz"
    missing_target = tmp_path / "missing-target.npz"
    try:
        destination.symlink_to(missing_target)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable in this Windows environment: {error}")
    link_target_before = destination.readlink()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cache_provenance_module.atomic_save_npz_with_sidecar(
            destination,
            arrays={},
            metadata={},
        )

    assert os.path.lexists(destination)
    assert destination.is_symlink()
    assert destination.readlink() == link_target_before
    assert not missing_target.exists()
    assert not destination.with_suffix(".npz.metadata.json").exists()


def test_engineered_cache_declares_explicit_unlearned_provenance(tmp_path: Path) -> None:
    identifiers = np.asarray(["sample-a", "sample-b"], dtype=np.str_)
    features = EngineeredFeatureSet(
        values=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        names=("morphology.area", "mask.count"),
    )
    cache, _, metadata = save_engineered_feature_cache(
        features,
        identifiers,
        tmp_path / "engineered.npz",
        manifest_sha256=_MANIFEST_SHA,
        raw_inventory_sha256=_INVENTORY_SHA,
    )
    verification = verify_frozen_cache_sidecar(cache)
    assert metadata["representation_id"] == "engineered_target_features"
    assert str(metadata["weight_identifier"]).startswith("unlearned:")
    assert metadata["weights_sha256"] == metadata["weight_sha256"]
    assert metadata["sample_order_sha256"] == ordered_sample_ids_sha256(identifiers)
    assert verification.metadata["feature_dimension"] == 2


def test_context_morphometrics_cache_is_aligned_separate_and_deterministic(
    tmp_path: Path,
) -> None:
    first_context, _ = _save_context(tmp_path / "first" / "context.npz")
    second_context, _ = _save_context(tmp_path / "second" / "context.npz")
    engineered = EngineeredFeatureSet(
        values=np.asarray(
            [
                [1.0, 101.0, 2.0],
                [3.0, 102.0, 4.0],
                [5.0, 103.0, 6.0],
            ],
            dtype=np.float64,
        ),
        names=("morphology.area", "mask.count", "morphology.solidity"),
    )
    first = save_context_morphometrics_cache(
        first_context,
        engineered,
        first_context.sample_ids,
        tmp_path / "first" / "context_plus_morphometrics.npz",
        manifest_sha256=_MANIFEST_SHA,
        raw_inventory_sha256=_INVENTORY_SHA,
    )
    second = save_context_morphometrics_cache(
        second_context,
        engineered,
        second_context.sample_ids,
        tmp_path / "second" / "context_plus_morphometrics.npz",
        manifest_sha256=_MANIFEST_SHA,
        raw_inventory_sha256=_INVENTORY_SHA,
    )
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.values[:, :512], first_context.embeddings)
    np.testing.assert_array_equal(
        first.values[:, 512:],
        engineered.values[:, [0, 2]].astype(np.float32),
    )
    assert first.names[-2:] == ("morphology.area", "morphology.solidity")
    assert first.metadata["representation_id"] == (
        "imagenet_context_embeddings_plus_target_morphometrics"
    )
    assert first.metadata["input_variant"] == "context_rgb_plus_target_morphometrics"
    assert first.metadata["cache_content_sha256"] == second.metadata["cache_content_sha256"]
    assert first.metadata["sidecar_semantic_sha256"] == second.metadata["sidecar_semantic_sha256"]
    assert first.metadata["cache_file_sha256"] == second.metadata["cache_file_sha256"]
    verify_frozen_cache_sidecar(first.cache_path)

    reversed_ids = first_context.sample_ids[::-1].copy()
    with pytest.raises(ValueError, match="sample order differs"):
        save_context_morphometrics_cache(
            first_context,
            engineered,
            reversed_ids,
            tmp_path / "misaligned.npz",
            manifest_sha256=_MANIFEST_SHA,
            raw_inventory_sha256=_INVENTORY_SHA,
        )
