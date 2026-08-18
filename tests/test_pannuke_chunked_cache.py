"""Memory-bounded PanNuke cache publication and resume contract tests."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import histo_audit.representations.pannuke as pannuke_module
import histo_audit.representations.pannuke_chunked as chunked_module
from histo_audit.data.targets import highlight_target
from histo_audit.experiment import representation_independence as independence_module
from histo_audit.experiment.representation_independence import (
    IndependenceAuditorInput,
    build_pannuke_representation_cache_with_independence,
    build_representation_independence_payload,
    publish_representation_independence_artifact,
    validate_representation_independence_file,
)
from histo_audit.pannuke import (
    PanNukeSemanticsError,
    PanNukeValidationResult,
    build_nucleus_manifest,
    validate_pannuke,
)
from histo_audit.representations import (
    EmbeddingResult,
    PanNukeCropConfig,
    ResNet18EmbeddingConfig,
    build_pannuke_representation_cache,
    save_embedding_cache,
)
from histo_audit.representations.cache_provenance import (
    canonical_sha256,
    primary_cache_provenance_record,
    verify_frozen_cache_sidecar,
)


def _tiny_release(root: Path, *, fold_ids: tuple[int, ...] = (1,)) -> Path:
    height = width = 24
    y, x = np.mgrid[:height, :width]
    for fold_id in fold_ids:
        fold = root / f"Fold {fold_id}" / "release_arrays"
        fold.mkdir(parents=True)
        images = np.zeros((3, height, width, 3), dtype=np.uint8)
        for patch in range(3):
            images[patch, ..., 0] = (20 + x * 5 + patch * 7 + fold_id) % 256
            images[patch, ..., 1] = (30 + y * 4 + patch * 11 + fold_id) % 256
            images[patch, ..., 2] = (40 + (x + y) * 3 + patch * 13 + fold_id) % 256
        masks = np.zeros((3, height, width, 6), dtype=np.int32)
        masks[0, 2:7, 3:9, 0] = 101
        masks[0, 13:20, 12:19, 1] = 202
        masks[1, 3:9, 13:20, 2] = 303
        masks[1, 14:20, 3:9, 3] = 404
        masks[2, 7:17, 8:16, 4] = 505
        masks[..., 5] = (~np.any(masks[..., :5] > 0, axis=-1)).astype(np.int32)
        np.save(fold / "pixels.npy", images)
        np.save(fold / "labels.npy", masks)
        np.save(
            fold / "organs.npy",
            np.asarray(["Breast", "Colon", "Lung"], dtype="<U16"),
        )
    return root


def _validated_manifest(
    tmp_path: Path,
    *,
    fold_ids: tuple[int, ...] = (1,),
) -> tuple[PanNukeValidationResult, Path]:
    validation = validate_pannuke(
        _tiny_release(tmp_path / "pannuke", fold_ids=fold_ids),
        tmp_path / "validation",
        expected_fold_ids=fold_ids,
        max_overlay_patches=1,
    )
    manifest = build_nucleus_manifest(validation, tmp_path / "manifest", batch_rows=2)
    return validation.result, manifest.parquet_path


def _manifest_authority_kwargs(manifest: Path) -> dict[str, Any]:
    _, provenance, manifest_sha256 = pannuke_module._manifest_frame(
        manifest.resolve(),
        None,
        eligibility_scope="analysis",
    )
    return {
        "expected_canonical_manifest_sha256": manifest_sha256,
        "expected_analysis_eligible_sample_order_sha256": provenance[
            "manifest_eligible_sample_ids_sha256"
        ],
        "expected_analysis_eligible_sample_count": provenance["manifest_eligible_instance_count"],
    }


def _fake_embeddings(
    images: np.ndarray,
    sample_ids: Any,
    *,
    target_masks: np.ndarray | None = None,
    config: ResNet18EmbeddingConfig | None = None,
    manifest_sha256: str | None = None,
    raw_inventory_sha256: str | None = None,
    representation_id: str | None = None,
    analysis_eligibility: Any = None,
    source_crop_cache_binding: Any = None,
    cache_path: str | Path | None = None,
    **_: Any,
) -> EmbeddingResult:
    settings = config or ResNet18EmbeddingConfig()
    identifiers = np.asarray(sample_ids, dtype=np.str_)
    rgb = np.asarray(images, dtype=np.uint8)
    if settings.input_variant == "target_highlighted_rgb":
        assert target_masks is not None
        prepared = np.stack(
            [
                highlight_target(
                    image,
                    mask,
                    context_brightness=settings.context_brightness,
                )
                for image, mask in zip(rgb, np.asarray(target_masks, dtype=bool), strict=True)
            ]
        )
        contract_variant = "target_highlighted_rgb"
    else:
        prepared = rgb
        contract_variant = "context_rgb"
    row_signal = prepared.astype(np.float64).mean(axis=(1, 2, 3))
    identifier_signal = np.asarray(
        [int(canonical_sha256(value)[:8], 16) % 10_000 for value in identifiers.tolist()],
        dtype=np.float64,
    )
    columns = np.arange(512, dtype=np.float64)[None, :]
    values = (
        row_signal[:, None] / 255.0 + identifier_signal[:, None] / 10_000.0 + columns / 512.0
    ).astype(np.float32)
    weights_sha = "d" * 64
    binding = dict(source_crop_cache_binding)
    metadata = {
        "schema_version": 1,
        "sample_count": len(identifiers),
        "encoder_name": "fake.resnet18",
        "encoder_identifier": "resnet18_imagenet1k_v1",
        "encoder_frozen": True,
        "weight_identifier": "fake:IMAGENET1K_V1",
        "weight_sha256": weights_sha,
        "weights_sha256": weights_sha,
        "preprocessing_identifier": "fake_resnet18_official_preprocessing",
        "preprocessing": {"identifier": "deterministic-test-preprocessing"},
        "input_variant": settings.input_variant,
        "legacy_input_variant": settings.input_variant,
        "contract_input_variant": contract_variant,
        "representation_id": representation_id,
        "context_brightness": (
            settings.context_brightness
            if settings.input_variant == "target_highlighted_rgb"
            else None
        ),
        "input_sha256": canonical_sha256(prepared.tolist()),
        "dtype": "float32",
        "output_dimension": 512,
        "device": "cpu",
        "amp_enabled": False,
        "batch_size_requested": settings.batch_size,
        "batch_size_initial_effective": min(settings.batch_size, len(identifiers)),
        "batch_size_final": min(settings.batch_size, len(identifiers)),
        "batch_oom_backoffs": [],
        "extraction_seconds": 0.01,
        "extracted_at_utc": "2026-07-18T00:00:00Z",
        "package_versions": {"fake": "1"},
        "versions": {"fake": "1"},
        "encoder_metadata": {
            "architecture": "fake.resnet18",
            "encoder_frozen": True,
            "input_sha256": canonical_sha256(prepared.tolist()),
            "output_dimension": 512,
            "output_dtype": "float32",
            "weight_identifier": "fake:IMAGENET1K_V1",
            "source_crop_cache_binding": binding,
            "source_crop_cache_binding_sha256": canonical_sha256(binding),
        },
        "encoder_implementation": {
            "module": "tests.test_pannuke_chunked_cache",
            "entrypoint": "_fake_embeddings",
            "source_file_sha256": "e" * 64,
        },
        "cache_recipe": {
            "identifier": "fake_embedding_npz_v1",
            "array_keys": ["embeddings", "metadata_json", "sample_ids"],
            "output_dtype": "float32",
            "pickle_allowed": False,
        },
        "configuration": asdict(settings),
        "provenance_scope": "stage_eligible",
        "manifest_sha256": manifest_sha256,
        "crop_manifest_sha256": manifest_sha256,
        "raw_inventory_sha256": raw_inventory_sha256,
        "analysis_eligibility": dict(analysis_eligibility),
        "source_crop_cache_binding": binding,
        "source_crop_cache_binding_sha256": canonical_sha256(binding),
    }
    published_cache = None
    published_sidecar = None
    if cache_path is not None:
        published_cache, published_sidecar, metadata = save_embedding_cache(
            cache_path, values, identifiers, metadata
        )
    result = EmbeddingResult(
        embeddings=values,
        sample_ids=identifiers,
        metadata=metadata,
        cache_path=published_cache,
        metadata_path=published_sidecar,
    )
    result.validate()
    return result


def _patch_fake_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pannuke_module, "extract_resnet18_embeddings", _fake_embeddings)
    monkeypatch.setattr(chunked_module, "extract_resnet18_embeddings", _fake_embeddings)


def _assert_npz_arrays_equal(left: Path, right: Path, *, excluded: set[str] | None = None) -> None:
    excluded = excluded or set()
    with (
        np.load(left, allow_pickle=False) as left_payload,
        np.load(right, allow_pickle=False) as right_payload,
    ):
        assert set(left_payload.files) == set(right_payload.files)
        for name in left_payload.files:
            if name not in excluded:
                assert np.array_equal(left_payload[name], right_payload[name]), name


def test_chunked_bundle_matches_small_path_and_publishes_primary_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation, manifest = _validated_manifest(tmp_path)
    _patch_fake_encoder(monkeypatch)
    crop = PanNukeCropConfig(output_size=32, padding=3, context_brightness=0.45)
    encoder = ResNet18EmbeddingConfig(
        input_variant="target_highlighted_rgb",
        context_brightness=0.45,
        device="cpu",
        batch_size=2,
    )
    baseline = build_pannuke_representation_cache(
        validation,
        manifest,
        tmp_path / "baseline",
        crop_config=crop,
        resnet_config=encoder,
        include_context_embeddings=True,
    )
    chunked = build_pannuke_representation_cache(
        validation,
        manifest,
        tmp_path / "chunked",
        crop_config=crop,
        resnet_config=encoder,
        include_context_embeddings=True,
        chunk_size=2,
    )

    assert isinstance(chunked.crops.context_rgb, np.memmap)
    assert isinstance(chunked.engineered.values, np.memmap)
    assert isinstance(chunked.embeddings.embeddings, np.memmap)
    assert tuple(chunked.crops.sample_ids.tolist()) == tuple(baseline.crops.sample_ids.tolist())
    _assert_npz_arrays_equal(baseline.crop_cache_path, chunked.crop_cache_path)
    _assert_npz_arrays_equal(baseline.engineered_cache_path, chunked.engineered_cache_path)
    assert baseline.embeddings.cache_path is not None
    assert chunked.embeddings.cache_path is not None
    _assert_npz_arrays_equal(
        baseline.embeddings.cache_path,
        chunked.embeddings.cache_path,
        excluded={"metadata_json"},
    )
    assert baseline.context_embeddings is not None
    assert chunked.context_embeddings is not None
    assert baseline.context_embeddings.cache_path is not None
    assert chunked.context_embeddings.cache_path is not None
    _assert_npz_arrays_equal(
        baseline.context_embeddings.cache_path,
        chunked.context_embeddings.cache_path,
        excluded={"metadata_json"},
    )
    assert baseline.context_morphometrics is not None
    assert chunked.context_morphometrics is not None
    _assert_npz_arrays_equal(
        baseline.context_morphometrics.cache_path,
        chunked.context_morphometrics.cache_path,
    )

    expected = (
        chunked.crop_cache_path,
        chunked.engineered_cache_path,
        chunked.embeddings.cache_path,
        chunked.context_embeddings.cache_path,
        chunked.context_morphometrics.cache_path,
    )
    for cache in expected:
        verification = verify_frozen_cache_sidecar(cache)
        assert verification.metadata["primary_cache_provenance"] == (
            primary_cache_provenance_record(verification.metadata)
        )
    expected_names = {
        "pannuke_crops.npz",
        "pannuke_crops.npz.metadata.json",
        "pannuke_engineered_features.npz",
        "pannuke_engineered_features.npz.metadata.json",
        "pannuke_resnet18_target_highlighted_embeddings.npz",
        "pannuke_resnet18_target_highlighted_embeddings.npz.metadata.json",
        "pannuke_resnet18_context_rgb_embeddings.npz",
        "pannuke_resnet18_context_rgb_embeddings.npz.metadata.json",
        "pannuke_resnet18_context_plus_target_morphometrics.npz",
        "pannuke_resnet18_context_plus_target_morphometrics.npz.metadata.json",
    }
    assert {path.name for path in (tmp_path / "chunked").iterdir()} == expected_names
    resume = tmp_path / ".chunked.chunked-resume"
    assert resume.is_dir()
    chunked_module.cleanup_pannuke_chunked_workspace(chunked)
    assert not resume.exists()
    assert {path.name for path in (tmp_path / "chunked").iterdir()} == expected_names


def test_chunked_resume_skips_verified_crop_chunks_and_keeps_output_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation, manifest = _validated_manifest(tmp_path)
    _patch_fake_encoder(monkeypatch)
    output = tmp_path / "resumable"
    original = chunked_module.extract_pannuke_crop_batch
    calls = 0
    first_chunk_ids: tuple[str, ...] = ()

    def fail_on_second_chunk(*args: Any, **kwargs: Any):
        nonlocal calls, first_chunk_ids
        calls += 1
        selection = kwargs["_selection_override"][0]
        identifiers = tuple(selection["sample_id"].astype(str).tolist())
        if calls == 1:
            first_chunk_ids = identifiers
        if calls == 2:
            raise RuntimeError("injected chunk interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(chunked_module, "extract_pannuke_crop_batch", fail_on_second_chunk)
    with pytest.raises(RuntimeError, match="injected chunk interruption"):
        build_pannuke_representation_cache(
            validation,
            manifest,
            output,
            chunk_size=2,
            resnet_config=ResNet18EmbeddingConfig(
                input_variant="target_highlighted_rgb", device="cpu", batch_size=2
            ),
        )
    assert not output.exists()
    resume = tmp_path / ".resumable.chunked-resume"
    assert (resume / "checkpoint.json").is_file()
    assert not list(tmp_path.glob("resumable/*.npz"))

    resumed_ids: list[str] = []

    def track_resumed_chunks(*args: Any, **kwargs: Any):
        selection = kwargs["_selection_override"][0]
        resumed_ids.extend(selection["sample_id"].astype(str).tolist())
        return original(*args, **kwargs)

    monkeypatch.setattr(chunked_module, "extract_pannuke_crop_batch", track_resumed_chunks)
    artifacts = build_pannuke_representation_cache(
        validation,
        manifest,
        output,
        chunk_size=2,
        resnet_config=ResNet18EmbeddingConfig(
            input_variant="target_highlighted_rgb", device="cpu", batch_size=2
        ),
    )
    assert output.is_dir()
    assert resume.is_dir()
    assert not set(first_chunk_ids).intersection(resumed_ids)
    assert len(artifacts.crops.sample_ids) == 5
    chunked_module.cleanup_pannuke_chunked_workspace(artifacts)
    assert not resume.exists()


def test_same_process_independence_failure_retracts_and_retry_reuses_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation, manifest = _validated_manifest(tmp_path)
    _patch_fake_encoder(monkeypatch)
    output = tmp_path / "transactional"
    independence_path = tmp_path / "representation_independence.json"
    original_extract = chunked_module.extract_pannuke_crop_batch
    extraction_calls = 0

    def track_extraction(*args: Any, **kwargs: Any):
        nonlocal extraction_calls
        extraction_calls += 1
        return original_extract(*args, **kwargs)

    def fail_with_live_maps(*args: Any, **kwargs: Any):
        artifacts = args[0]
        assert isinstance(artifacts.crops.context_rgb, np.memmap)
        assert isinstance(artifacts.engineered.values, np.memmap)
        assert isinstance(artifacts.embeddings.embeddings, np.memmap)
        assert artifacts.context_embeddings is not None
        assert isinstance(artifacts.context_embeddings.embeddings, np.memmap)
        assert artifacts.context_morphometrics is not None
        morph_sample_ids = artifacts.context_morphometrics.sample_ids
        assert not isinstance(morph_sample_ids, np.memmap)
        owner: Any = morph_sample_ids
        while getattr(owner, "base", None) is not None:
            owner = owner.base
        assert not isinstance(owner, np.memmap)
        np.testing.assert_array_equal(morph_sample_ids, artifacts.crops.sample_ids)
        raise RuntimeError("injected independence failure")

    monkeypatch.setattr(chunked_module, "extract_pannuke_crop_batch", track_extraction)
    monkeypatch.setattr(
        independence_module,
        "build_pannuke_representation_independence_artifact",
        fail_with_live_maps,
    )
    with pytest.raises(RuntimeError, match="injected independence failure"):
        build_pannuke_representation_cache_with_independence(
            validation,
            manifest,
            output,
            independence_path,
            class_order=(0, 1, 2, 3, 4),
            development_official_folds=(1,),
            final_test_fold=2,
            reference_validation_fraction_groups=0.1,
            split_seed=223,
            **_manifest_authority_kwargs(manifest),
            resnet_config=ResNet18EmbeddingConfig(
                input_variant="target_highlighted_rgb", device="cpu", batch_size=2
            ),
            chunk_size=2,
        )
    assert extraction_calls > 0
    assert not output.exists()
    assert not independence_path.exists()
    resume = tmp_path / ".transactional.chunked-resume"
    assert (resume / "bundle").is_dir()

    extraction_calls = 0

    def publish_strict_test_matrix(artifacts: Any, path: Path, **_: Any):
        engineered_provenance = verify_frozen_cache_sidecar(
            artifacts.engineered_cache_path
        ).metadata["primary_cache_provenance"]
        assert artifacts.context_embeddings is not None
        context_provenance = verify_frozen_cache_sidecar(
            artifacts.context_embeddings.cache_path
        ).metadata["primary_cache_provenance"]
        highlighted_provenance = verify_frozen_cache_sidecar(
            artifacts.embeddings.cache_path
        ).metadata["primary_cache_provenance"]
        morphology_columns = tuple(
            index
            for index, name in enumerate(artifacts.engineered.names)
            if name.startswith("morphology.")
        )
        morphology = artifacts.engineered.values[:, morphology_columns]
        payload = build_representation_independence_payload(
            generator_features=morphology,
            audit_sample_ids=artifacts.crops.sample_ids,
            audit_group_ids=artifacts.crops.group_ids,
            generator_cache_provenance=engineered_provenance,
            auditors=(
                IndependenceAuditorInput(
                    representation_id="engineered_target_features",
                    family="engineered",
                    features=artifacts.engineered.values,
                    cache_provenance=engineered_provenance,
                    matrix_decision="not_independent",
                    matrix_reason="Contains the generator columns.",
                ),
                IndependenceAuditorInput(
                    representation_id="imagenet_resnet18_context",
                    family="imagenet",
                    features=artifacts.context_embeddings.embeddings,
                    cache_provenance=context_provenance,
                    matrix_decision="verified_independent",
                    matrix_reason="Independent test transformation.",
                ),
                IndependenceAuditorInput(
                    representation_id="imagenet_resnet18_highlighted",
                    family="imagenet",
                    features=artifacts.embeddings.embeddings,
                    cache_provenance=highlighted_provenance,
                    matrix_decision="verified_independent",
                    matrix_reason="Independent test transformation.",
                ),
            ),
        )
        return publish_representation_independence_artifact(
            payload,
            path,
            audit_sample_count=len(artifacts.crops.sample_ids),
            audit_group_count=len(np.unique(artifacts.crops.group_ids)),
        )

    monkeypatch.setattr(
        independence_module,
        "build_pannuke_representation_independence_artifact",
        publish_strict_test_matrix,
    )
    result = build_pannuke_representation_cache_with_independence(
        validation,
        manifest,
        output,
        independence_path,
        class_order=(0, 1, 2, 3, 4),
        development_official_folds=(1,),
        final_test_fold=2,
        reference_validation_fraction_groups=0.1,
        split_seed=223,
        **_manifest_authority_kwargs(manifest),
        resnet_config=ResNet18EmbeddingConfig(
            input_variant="target_highlighted_rgb", device="cpu", batch_size=2
        ),
        chunk_size=2,
    )
    assert extraction_calls == 0
    assert output.is_dir()
    assert independence_path.is_file()
    assert result.independence.path == independence_path.resolve()
    validate_representation_independence_file(independence_path)
    chunked_module.cleanup_pannuke_chunked_workspace(result.representations)
    assert not resume.exists()


@pytest.mark.parametrize(
    "authority_field",
    (
        "expected_canonical_manifest_sha256",
        "expected_analysis_eligible_sample_order_sha256",
        "expected_analysis_eligible_sample_count",
    ),
)
def test_same_process_bundle_rejects_wrong_analysis_manifest_authority_before_build(
    tmp_path: Path,
    authority_field: str,
) -> None:
    validation, manifest = _validated_manifest(tmp_path)
    authority = _manifest_authority_kwargs(manifest)
    if authority_field == "expected_analysis_eligible_sample_count":
        authority[authority_field] = int(authority[authority_field]) + 1
    else:
        authority[authority_field] = "f" * 64
    output = tmp_path / "authority-cache"
    independence_path = tmp_path / "authority-independence.json"

    with pytest.raises(ValueError, match="analysis manifest authority mismatch"):
        build_pannuke_representation_cache_with_independence(
            validation,
            manifest,
            output,
            independence_path,
            class_order=(0, 1, 2, 3, 4),
            development_official_folds=(1,),
            final_test_fold=2,
            reference_validation_fraction_groups=0.1,
            split_seed=223,
            **authority,
        )

    assert not output.exists()
    assert not independence_path.exists()


def test_same_process_bundle_forbids_explicit_sample_subset_at_api_boundary(
    tmp_path: Path,
) -> None:
    validation, manifest = _validated_manifest(tmp_path)
    authority = _manifest_authority_kwargs(manifest)
    frame, _, _ = pannuke_module._manifest_frame(
        manifest.resolve(),
        None,
        eligibility_scope="analysis",
    )
    output = tmp_path / "subset-cache"
    independence_path = tmp_path / "subset-independence.json"

    with pytest.raises(ValueError, match="sample_ids subsets are forbidden"):
        build_pannuke_representation_cache_with_independence(
            validation,
            manifest,
            output,
            independence_path,
            class_order=(0, 1, 2, 3, 4),
            development_official_folds=(1,),
            final_test_fold=2,
            reference_validation_fraction_groups=0.1,
            split_seed=223,
            sample_ids=(str(frame.iloc[0]["sample_id"]),),
            **authority,
        )

    assert not output.exists()
    assert not independence_path.exists()


def test_same_process_bundle_rejects_schema_valid_three_fold_manifest_subset(
    tmp_path: Path,
) -> None:
    validation, manifest = _validated_manifest(tmp_path, fold_ids=(1, 2, 3))
    authority = _manifest_authority_kwargs(manifest)
    table = pq.read_table(manifest)
    first_group = table["group_id"][0].as_py()
    keep = [value.as_py() != first_group for value in table["group_id"]]
    subset = tmp_path / "three-fold-subset.parquet"
    pq.write_table(table.filter(pa.array(keep, type=pa.bool_())), subset)
    assert set(pq.read_table(subset, columns=["official_fold"])["official_fold"].to_pylist()) == {
        1,
        2,
        3,
    }
    output = tmp_path / "three-fold-subset-cache"
    independence_path = tmp_path / "three-fold-subset-independence.json"

    with pytest.raises(ValueError, match="analysis manifest authority mismatch"):
        build_pannuke_representation_cache_with_independence(
            validation,
            subset,
            output,
            independence_path,
            class_order=(0, 1, 2, 3, 4),
            development_official_folds=(1, 2),
            final_test_fold=3,
            reference_validation_fraction_groups=0.1,
            split_seed=223,
            **authority,
        )

    assert not output.exists()
    assert not independence_path.exists()


def test_same_process_bundle_retracts_cache_on_postbuild_authority_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation, manifest = _validated_manifest(tmp_path)
    _patch_fake_encoder(monkeypatch)
    authority = _manifest_authority_kwargs(manifest)
    output = tmp_path / "postbuild-authority-cache"
    independence_path = tmp_path / "postbuild-authority-independence.json"
    original_builder = independence_module.build_pannuke_representation_cache

    def inject_postbuild_drift(*args: Any, **kwargs: Any):
        artifacts = original_builder(*args, **kwargs)
        artifacts.crops.metadata["manifest_sha256"] = "0" * 64
        return artifacts

    monkeypatch.setattr(
        independence_module,
        "build_pannuke_representation_cache",
        inject_postbuild_drift,
    )
    with pytest.raises(RuntimeError, match="published crop metadata manifest differs"):
        build_pannuke_representation_cache_with_independence(
            validation,
            manifest,
            output,
            independence_path,
            class_order=(0, 1, 2, 3, 4),
            development_official_folds=(1,),
            final_test_fold=2,
            reference_validation_fraction_groups=0.1,
            split_seed=223,
            resnet_config=ResNet18EmbeddingConfig(
                input_variant="target_highlighted_rgb",
                device="cpu",
                batch_size=2,
            ),
            chunk_size=2,
            **authority,
        )

    assert not output.exists()
    assert not independence_path.exists()


@pytest.mark.parametrize("cache_role", ("crop", "context_morphometrics"))
def test_complete_authority_readback_detects_crop_and_morph_cache_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache_role: str,
) -> None:
    validation, manifest = _validated_manifest(tmp_path)
    _patch_fake_encoder(monkeypatch)
    authority_values = _manifest_authority_kwargs(manifest)
    artifacts = build_pannuke_representation_cache(
        validation,
        manifest,
        tmp_path / f"tamper-{cache_role}",
        include_context_embeddings=True,
        chunk_size=2,
        resnet_config=ResNet18EmbeddingConfig(
            input_variant="target_highlighted_rgb",
            device="cpu",
            batch_size=2,
        ),
    )
    assert artifacts.context_morphometrics is not None
    cache_path = (
        artifacts.crop_cache_path
        if cache_role == "crop"
        else artifacts.context_morphometrics.cache_path
    )
    with cache_path.open("ab") as handle:
        handle.write(b"injected-cache-tamper")
    authority = independence_module._AnalysisManifestAuthority.validated(
        canonical_manifest_sha256=authority_values["expected_canonical_manifest_sha256"],
        analysis_eligible_sample_order_sha256=authority_values[
            "expected_analysis_eligible_sample_order_sha256"
        ],
        analysis_eligible_sample_count=authority_values["expected_analysis_eligible_sample_count"],
    )

    with pytest.raises(ValueError, match="cache file checksum differs"):
        independence_module._verify_published_analysis_manifest_authority(
            artifacts,
            manifest.resolve(),
            authority,
        )

    chunked_module.cleanup_pannuke_chunked_workspace(artifacts)


def test_manifest_mutation_during_evidence_retracts_cache_and_independence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation, manifest = _validated_manifest(tmp_path, fold_ids=(1, 2, 3))
    _patch_fake_encoder(monkeypatch)
    authority = _manifest_authority_kwargs(manifest)
    output = tmp_path / "final-authority-cache"
    independence_path = tmp_path / "final-authority-independence.json"
    original_builder = independence_module.build_pannuke_representation_independence_artifact

    def publish_then_mutate_manifest(*args: Any, **kwargs: Any):
        result = original_builder(*args, **kwargs)
        table = pq.read_table(manifest)
        metadata = dict(table.schema.metadata or {})
        metadata[b"injected_post_evidence_drift"] = b"true"
        pq.write_table(table.replace_schema_metadata(metadata), manifest)
        return result

    monkeypatch.setattr(
        independence_module,
        "build_pannuke_representation_independence_artifact",
        publish_then_mutate_manifest,
    )
    with pytest.raises(ValueError, match="analysis manifest authority mismatch"):
        build_pannuke_representation_cache_with_independence(
            validation,
            manifest,
            output,
            independence_path,
            class_order=(0, 1, 2, 3, 4),
            development_official_folds=(1, 2),
            final_test_fold=3,
            reference_validation_fraction_groups=0.5,
            split_seed=223,
            resnet_config=ResNet18EmbeddingConfig(
                input_variant="target_highlighted_rgb",
                device="cpu",
                batch_size=2,
            ),
            chunk_size=2,
            **authority,
        )

    assert not output.exists()
    assert not independence_path.exists()


def test_chunked_resume_fails_closed_when_configuration_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation, manifest = _validated_manifest(tmp_path)
    _patch_fake_encoder(monkeypatch)
    output = tmp_path / "contract-change"
    original = chunked_module.extract_pannuke_crop_batch
    calls = 0

    def fail_after_one(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected stop")
        return original(*args, **kwargs)

    monkeypatch.setattr(chunked_module, "extract_pannuke_crop_batch", fail_after_one)
    with pytest.raises(RuntimeError, match="injected stop"):
        build_pannuke_representation_cache(
            validation,
            manifest,
            output,
            crop_config=PanNukeCropConfig(output_size=32, padding=3),
            resnet_config=ResNet18EmbeddingConfig(
                input_variant="target_highlighted_rgb", device="cpu", batch_size=2
            ),
            chunk_size=2,
        )
    monkeypatch.setattr(chunked_module, "extract_pannuke_crop_batch", original)
    with pytest.raises(RuntimeError, match="resume contract mismatch"):
        build_pannuke_representation_cache(
            validation,
            manifest,
            output,
            crop_config=PanNukeCropConfig(output_size=32, padding=4),
            resnet_config=ResNet18EmbeddingConfig(
                input_variant="target_highlighted_rgb", device="cpu", batch_size=2
            ),
            chunk_size=2,
        )
    assert not output.exists()


def test_chunked_resume_detects_completed_array_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation, manifest = _validated_manifest(tmp_path)
    _patch_fake_encoder(monkeypatch)
    output = tmp_path / "tampered"
    original = chunked_module.extract_pannuke_crop_batch
    calls = 0

    def fail_after_one(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected stop")
        return original(*args, **kwargs)

    monkeypatch.setattr(chunked_module, "extract_pannuke_crop_batch", fail_after_one)
    with pytest.raises(RuntimeError, match="injected stop"):
        build_pannuke_representation_cache(
            validation,
            manifest,
            output,
            resnet_config=ResNet18EmbeddingConfig(
                input_variant="target_highlighted_rgb", device="cpu", batch_size=2
            ),
            chunk_size=2,
        )
    context_path = (
        tmp_path / ".tampered.chunked-resume" / ".chunked-state" / "arrays" / "context_rgb.npy"
    )
    context = np.load(context_path, mmap_mode="r+", allow_pickle=False)
    context[0, 0, 0, 0] ^= np.uint8(1)
    context.flush()
    del context
    monkeypatch.setattr(chunked_module, "extract_pannuke_crop_batch", original)
    with pytest.raises(RuntimeError, match="completed crop chunk content changed"):
        build_pannuke_representation_cache(
            validation,
            manifest,
            output,
            resnet_config=ResNet18EmbeddingConfig(
                input_variant="target_highlighted_rgb", device="cpu", batch_size=2
            ),
            chunk_size=2,
        )
    assert not output.exists()


def test_post_publication_map_reopen_failure_retracts_public_bundle_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation, manifest = _validated_manifest(tmp_path)
    _patch_fake_encoder(monkeypatch)
    output = tmp_path / "post-publication-reopen"
    resume = tmp_path / ".post-publication-reopen.chunked-resume"
    original_open = chunked_module._open_fixed_maps
    injected = False

    def fail_first_reopen_after_publication(
        array_directory: Path,
        specs: dict[str, dict[str, Any]],
        *,
        allow_create: bool,
    ) -> dict[str, np.memmap[Any, Any]]:
        nonlocal injected
        if output.is_dir() and not injected:
            injected = True
            raise RuntimeError("injected post-publication map reopen failure")
        return original_open(array_directory, specs, allow_create=allow_create)

    monkeypatch.setattr(
        chunked_module,
        "_open_fixed_maps",
        fail_first_reopen_after_publication,
    )
    with pytest.raises(RuntimeError, match="post-publication map reopen failure"):
        build_pannuke_representation_cache(
            validation,
            manifest,
            output,
            include_context_embeddings=True,
            resnet_config=ResNet18EmbeddingConfig(
                input_variant="target_highlighted_rgb",
                device="cpu",
                batch_size=2,
            ),
            chunk_size=2,
        )

    assert injected is True
    assert not output.exists()
    bundle = resume / "bundle"
    assert bundle.is_dir()
    assert len(tuple(bundle.iterdir())) == 10

    monkeypatch.setattr(chunked_module, "_open_fixed_maps", original_open)
    artifacts = build_pannuke_representation_cache(
        validation,
        manifest,
        output,
        include_context_embeddings=True,
        resnet_config=ResNet18EmbeddingConfig(
            input_variant="target_highlighted_rgb",
            device="cpu",
            batch_size=2,
        ),
        chunk_size=2,
    )
    assert output.is_dir()
    assert len(tuple(output.iterdir())) == 10
    chunked_module.cleanup_pannuke_chunked_workspace(artifacts)
    assert not resume.exists()


def test_representation_cache_rejects_sealed_ancestor_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation, manifest = _validated_manifest(tmp_path)
    _patch_fake_encoder(monkeypatch)
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    marker = sealed / ".immutable.json"
    marker.write_text('{"state":"complete"}\n', encoding="utf-8")
    output = sealed / "cache"

    with pytest.raises(PanNukeSemanticsError, match="sealed/immutable ancestor"):
        build_pannuke_representation_cache(
            validation,
            manifest,
            output,
            resnet_config=ResNet18EmbeddingConfig(
                input_variant="target_highlighted_rgb",
                device="cpu",
                batch_size=2,
            ),
            chunk_size=2,
        )

    assert not output.exists()
    assert {path.name for path in sealed.iterdir()} == {marker.name}


def test_new_seal_before_chunked_publication_fails_closed_and_keeps_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation, manifest = _validated_manifest(tmp_path)
    _patch_fake_encoder(monkeypatch)
    derived = tmp_path / "derived"
    derived.mkdir()
    output = derived / "cache"
    resume = derived / ".cache.chunked-resume"
    marker = derived / ".immutable.json"
    original_guard = chunked_module.ensure_derived_output_outside_raw
    output_checks = 0

    def seal_before_publication(
        output_path: str | Path,
        raw_root: str | Path,
        *,
        purpose: str = "derived output",
    ) -> Path:
        nonlocal output_checks
        if Path(output_path).resolve() == output.resolve():
            output_checks += 1
            if output_checks == 2:
                marker.write_text('{"state":"complete"}\n', encoding="utf-8")
        return original_guard(output_path, raw_root, purpose=purpose)

    monkeypatch.setattr(
        chunked_module,
        "ensure_derived_output_outside_raw",
        seal_before_publication,
    )
    with pytest.raises(PanNukeSemanticsError, match="sealed/immutable ancestor"):
        build_pannuke_representation_cache(
            validation,
            manifest,
            output,
            resnet_config=ResNet18EmbeddingConfig(
                input_variant="target_highlighted_rgb",
                device="cpu",
                batch_size=2,
            ),
            chunk_size=2,
        )

    assert output_checks == 2
    assert not output.exists()
    assert (resume / "bundle").is_dir()
    marker.unlink()
    monkeypatch.setattr(
        chunked_module,
        "ensure_derived_output_outside_raw",
        original_guard,
    )
    artifacts = build_pannuke_representation_cache(
        validation,
        manifest,
        output,
        resnet_config=ResNet18EmbeddingConfig(
            input_variant="target_highlighted_rgb",
            device="cpu",
            batch_size=2,
        ),
        chunk_size=2,
    )
    assert output.is_dir()
    chunked_module.cleanup_pannuke_chunked_workspace(artifacts)


def test_small_cache_independence_failure_uses_exact_publication_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation, manifest = _validated_manifest(tmp_path)
    _patch_fake_encoder(monkeypatch)
    output = tmp_path / "small-transactional"
    independence_path = tmp_path / "small-independence.json"

    def fail_independence(*_: Any, **__: Any) -> None:
        raise RuntimeError("injected small-path independence failure")

    monkeypatch.setattr(
        independence_module,
        "build_pannuke_representation_independence_artifact",
        fail_independence,
    )
    with pytest.raises(RuntimeError, match="small-path independence failure"):
        build_pannuke_representation_cache_with_independence(
            validation,
            manifest,
            output,
            independence_path,
            class_order=(0, 1, 2, 3, 4),
            development_official_folds=(1,),
            final_test_fold=2,
            reference_validation_fraction_groups=0.1,
            split_seed=223,
            **_manifest_authority_kwargs(manifest),
            resnet_config=ResNet18EmbeddingConfig(
                input_variant="target_highlighted_rgb",
                device="cpu",
                batch_size=2,
            ),
        )

    assert not output.exists()
    assert not independence_path.exists()
