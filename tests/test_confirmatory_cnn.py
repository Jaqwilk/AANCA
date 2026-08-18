from __future__ import annotations

import inspect
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from histo_audit.models import (
    CPU_TEST_ONLY_WEIGHT_IDENTIFIER,
    OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER,
    ConfirmatoryCNNConfig,
    ConfirmatoryCNNCPUTestOnlyAdapter,
    ConfirmatoryResNet18Classifier,
)
from histo_audit.models.cnn import (
    _batch_tensor,
    _build_resnet18,
    _prepare_images,
    confirmatory_cnn_data_and_split_sha256,
    validate_confirmatory_checkpoint_artifact,
)
from histo_audit.representations.imagenet import (
    PretrainedWeightsUnavailableError,
    official_resnet18_weight_cache_path,
)


def _tiny_images() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(1927)
    training = rng.integers(0, 256, size=(10, 32, 32, 3), dtype=np.uint8)
    validation = rng.integers(0, 256, size=(5, 32, 32, 3), dtype=np.uint8)
    training_labels = np.tile(np.arange(5, dtype=np.int64), 2)
    validation_labels = np.arange(5, dtype=np.int64)
    training_masks = np.zeros((10, 32, 32), dtype=np.uint8)
    validation_masks = np.zeros((5, 32, 32), dtype=np.uint8)
    training_masks[:, 8:24, 9:23] = 1
    validation_masks[:, 10:22, 11:21] = 1
    return (
        training,
        training_labels,
        training_masks,
        validation,
        validation_labels,
        validation_masks,
    )


def _fit_arguments() -> dict[str, Any]:
    training, labels, masks, validation, validation_labels, validation_masks = _tiny_images()
    return {
        "training_images": training,
        "observed_training_labels": labels,
        "training_sample_ids": [f"train-{index}" for index in range(10)],
        "training_group_ids": [f"train-group-{index // 2}" for index in range(10)],
        "reference_validation_images": validation,
        "reference_validation_labels": validation_labels,
        "reference_validation_sample_ids": [f"val-{index}" for index in range(5)],
        "reference_validation_group_ids": [f"val-group-{index}" for index in range(5)],
        "reference_validation_role": "reference_validation",
        "training_target_masks": masks,
        "reference_validation_target_masks": validation_masks,
    }


def _cpu_config(*, epochs: int = 1) -> ConfirmatoryCNNConfig:
    return ConfirmatoryCNNConfig(
        input_variant="context_rgb_plus_binary_target_mask",
        weight_identifier=CPU_TEST_ONLY_WEIGHT_IDENTIFIER,
        input_size=32,
        epochs=epochs,
        batch_size=5,
        minimum_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        early_stopping_patience=3,
        seed=73,
    )


def test_rgb_and_exact_binary_target_mask_input_contracts() -> None:
    images, _, masks, _, _, _ = _tiny_images()
    rgb = _prepare_images(images, None, "context_rgb", name="images")
    rgb_tensor = _batch_tensor(rgb, np.arange(2, dtype=np.int64), 32)
    assert rgb_tensor.shape == (2, 3, 32, 32)

    masked = _prepare_images(
        images,
        masks,
        "context_rgb_plus_binary_target_mask",
        name="images",
    )
    masked_tensor = _batch_tensor(masked, np.arange(2, dtype=np.int64), 48)
    assert masked_tensor.shape == (2, 4, 48, 48)
    assert set(torch.unique(masked_tensor[:, 3]).tolist()) == {0.0, 1.0}

    with pytest.raises(ValueError, match="does not accept target_masks"):
        _prepare_images(images, masks, "context_rgb", name="images")
    with pytest.raises(ValueError, match="requires exact target_masks"):
        _prepare_images(images, None, "context_rgb_plus_binary_target_mask", name="images")
    non_binary = masks.astype(np.float32)
    non_binary[0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="exactly binary"):
        _prepare_images(
            images,
            non_binary,
            "context_rgb_plus_binary_target_mask",
            name="images",
        )
    empty = masks.copy()
    empty[0] = 0
    with pytest.raises(ValueError, match="non-empty target"):
        _prepare_images(
            images,
            empty,
            "context_rgb_plus_binary_target_mask",
            name="images",
        )


def test_contiguous_uint8_and_bool_inputs_use_semantically_exact_fast_paths() -> None:
    images, labels, masks, validation, validation_labels, validation_masks = _tiny_images()
    bool_masks = np.ascontiguousarray(masks.astype(np.bool_))
    bool_validation_masks = np.ascontiguousarray(validation_masks.astype(np.bool_))

    prepared = _prepare_images(
        images,
        bool_masks,
        "context_rgb_plus_binary_target_mask",
        name="images",
    )

    assert prepared.rgb is images
    assert prepared.target_masks is bool_masks
    assert prepared.rgb.flags.c_contiguous
    assert prepared.target_masks.flags.c_contiguous
    np.testing.assert_array_equal(
        prepared.rgb,
        np.ascontiguousarray(np.rint(images.astype(np.float64)).astype(np.uint8)),
    )
    np.testing.assert_array_equal(
        prepared.target_masks,
        np.ascontiguousarray(bool_masks.astype(bool)),
    )

    sample_ids = [f"train-{index}" for index in range(len(images))]
    group_ids = [f"train-group-{index // 2}" for index in range(len(images))]
    validation_sample_ids = [f"val-{index}" for index in range(len(validation))]
    validation_group_ids = [f"val-group-{index}" for index in range(len(validation))]
    fast_hashes = confirmatory_cnn_data_and_split_sha256(
        images,
        labels,
        training_sample_ids=sample_ids,
        training_group_ids=group_ids,
        reference_validation_images=validation,
        reference_validation_labels=validation_labels,
        reference_validation_sample_ids=validation_sample_ids,
        reference_validation_group_ids=validation_group_ids,
        input_variant="context_rgb_plus_binary_target_mask",
        training_target_masks=bool_masks,
        reference_validation_target_masks=bool_validation_masks,
    )
    converted_hashes = confirmatory_cnn_data_and_split_sha256(
        images.astype(np.float64),
        labels,
        training_sample_ids=sample_ids,
        training_group_ids=group_ids,
        reference_validation_images=validation.astype(np.float64),
        reference_validation_labels=validation_labels,
        reference_validation_sample_ids=validation_sample_ids,
        reference_validation_group_ids=validation_group_ids,
        input_variant="context_rgb_plus_binary_target_mask",
        training_target_masks=bool_masks.astype(np.uint8),
        reference_validation_target_masks=bool_validation_masks.astype(np.uint8),
    )
    assert fast_hashes == converted_hashes


def test_non_contiguous_uint8_and_bool_inputs_are_copied_to_contiguous_buffers() -> None:
    images, _, masks, _, _, _ = _tiny_images()
    image_view = images[:, :, ::-1, :]
    mask_view = masks.astype(np.bool_)[:, :, ::-1]
    assert not image_view.flags.c_contiguous
    assert not mask_view.flags.c_contiguous

    prepared = _prepare_images(
        image_view,
        mask_view,
        "context_rgb_plus_binary_target_mask",
        name="images",
    )

    assert prepared.rgb.flags.c_contiguous
    assert prepared.target_masks is not None
    assert prepared.target_masks.flags.c_contiguous
    assert not np.shares_memory(prepared.rgb, image_view)
    assert not np.shares_memory(prepared.target_masks, mask_view)
    np.testing.assert_array_equal(prepared.rgb, image_view)
    np.testing.assert_array_equal(prepared.target_masks, mask_view)


def test_resnet18_has_fixed_five_outputs_and_deterministic_fourth_channel() -> None:
    rgb_config = ConfirmatoryCNNConfig(
        input_variant="context_rgb",
        weight_identifier=CPU_TEST_ONLY_WEIGHT_IDENTIFIER,
        input_size=32,
        epochs=1,
        seed=101,
    )
    mask_config = replace(rgb_config, input_variant="context_rgb_plus_binary_target_mask")
    rgb_model, rgb_metadata = _build_resnet18(rgb_config, cpu_test_only=True)
    first_mask_model, mask_metadata = _build_resnet18(mask_config, cpu_test_only=True)
    second_mask_model, _ = _build_resnet18(mask_config, cpu_test_only=True)
    rgb_backbone = cast(Any, rgb_model)
    first_mask_backbone = cast(Any, first_mask_model)
    second_mask_backbone = cast(Any, second_mask_model)

    assert rgb_backbone.conv1.in_channels == 3
    assert first_mask_backbone.conv1.in_channels == 4
    assert rgb_backbone.fc.out_features == first_mask_backbone.fc.out_features == 5
    torch.testing.assert_close(first_mask_backbone.conv1.weight[:, :3], rgb_backbone.conv1.weight)
    torch.testing.assert_close(
        first_mask_backbone.conv1.weight[:, 3:4],
        torch.zeros_like(first_mask_backbone.conv1.weight[:, 3:4]),
    )
    torch.testing.assert_close(first_mask_backbone.conv1.weight, second_mask_backbone.conv1.weight)
    torch.testing.assert_close(first_mask_backbone.fc.weight, rgb_backbone.fc.weight)
    torch.testing.assert_close(first_mask_backbone.fc.bias, rgb_backbone.fc.bias)
    assert rgb_metadata["implicit_weight_download"] is False
    assert mask_metadata["fourth_channel_initialisation"] == "zeros"


def test_official_weight_path_is_offline_and_missing_cache_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "histo_audit.models.cnn.official_resnet18_weight_cache_path",
        lambda _identifier: tmp_path / "absent-weights.pth",
    )
    config = ConfirmatoryCNNConfig(
        input_variant="context_rgb",
        weight_identifier=OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER,
        epochs=1,
    )
    with pytest.raises(PretrainedWeightsUnavailableError, match="never downloads"):
        _build_resnet18(config, cpu_test_only=False)


def test_fit_api_has_no_final_test_and_rejects_validation_leakage() -> None:
    parameter_names = inspect.signature(ConfirmatoryResNet18Classifier.fit).parameters
    assert not any("final" in name for name in parameter_names)

    arguments = _fit_arguments()
    arguments["reference_validation_role"] = "final_test"
    with pytest.raises(ValueError, match="reference_validation"):
        ConfirmatoryCNNCPUTestOnlyAdapter(_cpu_config()).fit(**arguments)

    arguments = _fit_arguments()
    arguments["reference_validation_group_ids"][0] = arguments["training_group_ids"][0]
    with pytest.raises(ValueError, match="group leakage"):
        ConfirmatoryCNNCPUTestOnlyAdapter(_cpu_config()).fit(**arguments)

    arguments = _fit_arguments()
    arguments["reference_validation_sample_ids"][0] = arguments["training_sample_ids"][0]
    with pytest.raises(ValueError, match="sample IDs overlap"):
        ConfirmatoryCNNCPUTestOnlyAdapter(_cpu_config()).fit(**arguments)


def test_cpu_test_only_checkpoint_resume_hashes_and_non_evidence_marker(tmp_path: Path) -> None:
    checkpoint = tmp_path / "cpu-test-only.pt"
    arguments = _fit_arguments()
    fitted = ConfirmatoryCNNCPUTestOnlyAdapter(_cpu_config()).fit(
        **arguments,
        checkpoint_path=checkpoint,
    )
    payload = cast(dict[str, Any], torch.load(checkpoint, map_location="cpu", weights_only=True))

    assert fitted.completed_epochs_ == 1
    assert payload["model_kind"] == "confirmatory_resnet18_five_class"
    assert payload["execution_mode"] == "cpu_test_only_non_evidence"
    assert payload["study_outcome_eligible"] is False
    assert payload["scaler_state_dict"] is None
    assert payload["early_stopping_state"]["best_network_state_dict"] is not None
    assert len(payload["configuration_sha256"]) == 64
    assert len(payload["resume_contract_sha256"]) == 64
    assert all(len(value) == 64 for value in payload["data_and_split_sha256"].values())
    assert payload["rng_state"]["torch_cpu"].dtype == torch.uint8
    assert fitted.telemetry_["study_outcome_eligible"] is False
    assert fitted.telemetry_["gradient_accumulation_steps"] == 2
    assert fitted.history_[0]["optimiser_steps"] == 1
    assert fitted.telemetry_["cuda_peak_memory_allocated_bytes"] is None

    resumed = ConfirmatoryCNNCPUTestOnlyAdapter(_cpu_config(epochs=2)).fit(
        **arguments,
        checkpoint_path=checkpoint,
        resume=True,
    )
    probabilities = resumed.predict_proba(
        arguments["reference_validation_images"],
        target_masks=arguments["reference_validation_target_masks"],
    )
    assert resumed.completed_epochs_ == 2
    assert probabilities.shape == (5, 5)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert not list(tmp_path.glob(".*.tmp"))

    changed = _fit_arguments()
    changed["training_images"] = changed["training_images"].copy()
    changed["training_images"][0, 0, 0, 0] ^= np.uint8(1)
    with pytest.raises(ValueError, match="data or split fingerprint"):
        ConfirmatoryCNNCPUTestOnlyAdapter(_cpu_config(epochs=3)).fit(
            **changed,
            checkpoint_path=checkpoint,
            resume=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real study trainer is CUDA-only")
def test_real_cuda_amp_checkpoint_resume_smoke(tmp_path: Path) -> None:
    if not official_resnet18_weight_cache_path(OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER).is_file():
        pytest.skip("official ResNet-18 checkpoint is absent; downloads are intentionally disabled")
    checkpoint = tmp_path / "cuda-study.pt"
    arguments = _fit_arguments()
    base = ConfirmatoryCNNConfig(
        input_variant="context_rgb_plus_binary_target_mask",
        weight_identifier=OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER,
        input_size=32,
        epochs=1,
        batch_size=10,
        minimum_batch_size=2,
        learning_rate=1e-4,
        early_stopping_patience=3,
        amp_dtype="float16",
        seed=211,
    )
    fitted = ConfirmatoryResNet18Classifier(base).fit(
        **arguments,
        checkpoint_path=checkpoint,
    )
    resumed = ConfirmatoryResNet18Classifier(replace(base, epochs=2)).fit(
        **arguments,
        checkpoint_path=checkpoint,
        resume=True,
    )
    probabilities = resumed.predict_proba(
        arguments["reference_validation_images"],
        target_masks=arguments["reference_validation_target_masks"],
    )
    payload = cast(dict[str, Any], torch.load(checkpoint, map_location="cpu", weights_only=True))
    validate_confirmatory_checkpoint_artifact(
        checkpoint,
        expected_configuration=asdict(replace(base, epochs=2)),
        expected_model_metadata=resumed.model_metadata_,
        expected_data_and_split_sha256=resumed.data_and_split_sha256_,
    )

    assert fitted.telemetry_["amp_enabled"] is True
    assert fitted.telemetry_["grad_scaler_enabled"] is True
    assert cast(int, fitted.telemetry_["cuda_peak_memory_allocated_bytes"]) > 0
    assert payload["study_outcome_eligible"] is True
    assert payload["scaler_state_dict"] is not None
    assert payload["telemetry"]["successful_optimiser_steps"] >= 1
    assert resumed.completed_epochs_ == 2
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
