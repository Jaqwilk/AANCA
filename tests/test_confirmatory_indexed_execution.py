from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from test_confirmatory_image_oof import (
    _authorise_fresh,
    _FastCPUTestAdapter,
    _inputs,
)
from test_confirmatory_runner import _bundle as _runner_bundle

from histo_audit.corruption.controlled import array_artifact_sha256
from histo_audit.cross_validation import image_oof as image_oof_module
from histo_audit.cross_validation.image_oof import grouped_oof_confirmatory_cnn
from histo_audit.experiment.confirmatory_core import (
    ConfirmatoryCellRequest,
    ConfirmatoryRunnerInputs,
    confirmatory_execution_controls_from_frozen_config,
)
from histo_audit.experiment.confirmatory_core import (
    _controlled_array_sha256 as core_controlled_array_sha256,
)
from histo_audit.experiment.confirmatory_core import (
    run_confirmatory_frozen_feature_oof as run_core_confirmatory_frozen_feature_oof,
)
from histo_audit.experiment.confirmatory_memory_workspace import (
    ReadOnlyBackingArray,
    RowIndexedArray,
)
from histo_audit.experiment.confirmatory_runner import (
    _controlled_array_sha256 as runner_controlled_array_sha256,
)
from histo_audit.experiment.confirmatory_runner import (
    _partition_content_binding,
    bridge_pannuke_confirmatory_inputs,
    run_confirmatory_frozen_feature_oof,
)
from histo_audit.experiment.pannuke_confirmatory_inputs import (
    ConfirmatoryPartitionFeature,
    ConfirmatoryPartitionInputs,
)
from histo_audit.models.cnn import (
    CPU_TEST_ONLY_WEIGHT_IDENTIFIER,
    ConfirmatoryCNNConfig,
    ConfirmatoryCNNCPUTestOnlyAdapter,
    _batch_tensor,
    _prepare_images,
    confirmatory_cnn_data_and_split_sha256,
)


def _readonly(values: NDArray[Any]) -> NDArray[Any]:
    output = np.ascontiguousarray(values).copy()
    output.setflags(write=False)
    return output


def _indexed(
    tmp_path: Path,
    name: str,
    values: NDArray[Any],
    indices: NDArray[np.int64] | None = None,
    *,
    logical_dtype: Any | None = None,
) -> RowIndexedArray:
    path = tmp_path / f"{name}.npy"
    np.save(path, np.ascontiguousarray(values), allow_pickle=False)
    mapped = np.load(path, allow_pickle=False, mmap_mode="r")
    assert isinstance(mapped, np.memmap)
    backing = ReadOnlyBackingArray(
        array_id=name,
        path=path,
        values=mapped,
        file_sha256="0" * 64,
        raw_array_sha256="1" * 64,
        source_npz_sha256="2" * 64,
        source_sidecar_sha256="3" * 64,
        source_member_name=f"{name}.npy",
    )
    selected = (
        np.arange(len(values), dtype=np.int64)
        if indices is None
        else np.asarray(indices, dtype=np.int64)
    )
    return RowIndexedArray(backing, selected, logical_dtype=logical_dtype)


def _forbid_implicit_array(
    _self: RowIndexedArray,
    *_args: Any,
    **_kwargs: Any,
) -> NDArray[Any]:
    raise AssertionError("indexed confirmatory input was implicitly materialised")


def test_indexed_controlled_and_cnn_hashes_are_bit_identical_to_legacy_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(701)
    images = rng.integers(0, 256, size=(15, 12, 12, 3), dtype=np.uint8)
    masks = np.zeros((15, 12, 12), dtype=bool)
    masks[:, 3:9, 3:9] = True
    train_indices = np.asarray([7, 1, 9, 3, 5, 0, 8, 2, 6, 4], dtype=np.int64)
    validation_indices = np.arange(10, 15, dtype=np.int64)
    indexed_images = _indexed(tmp_path, "images", images, train_indices)
    indexed_masks = _indexed(tmp_path, "masks", masks, train_indices)
    indexed_validation_images = _indexed(
        tmp_path,
        "validation_images",
        images,
        validation_indices,
    )
    indexed_validation_masks = _indexed(
        tmp_path,
        "validation_masks",
        masks,
        validation_indices,
    )
    raw_features = rng.normal(size=(15, 7)).astype(np.float32)
    indexed_features = _indexed(
        tmp_path,
        "features",
        raw_features,
        train_indices,
        logical_dtype=np.float64,
    )
    expected_features = raw_features[train_indices].astype(np.float64)
    assert not hasattr(indexed_images, "backing")
    assert not hasattr(indexed_images, "backing_path")

    monkeypatch.setattr(RowIndexedArray, "__array__", _forbid_implicit_array, raising=False)

    expected_controlled = array_artifact_sha256(expected_features)
    assert runner_controlled_array_sha256(indexed_features) == expected_controlled
    assert core_controlled_array_sha256(indexed_features) == expected_controlled

    labels = np.tile(np.arange(5, dtype=np.int64), 2)
    validation_labels = np.arange(5, dtype=np.int64)
    keyword_arguments = {
        "training_sample_ids": [f"train-{index}" for index in range(10)],
        "training_group_ids": [f"train-group-{index // 2}" for index in range(10)],
        "reference_validation_labels": validation_labels,
        "reference_validation_sample_ids": [f"validation-{index}" for index in range(5)],
        "reference_validation_group_ids": ["validation-group"] * 5,
        "input_variant": "context_rgb_plus_binary_target_mask",
    }
    indexed_hashes = confirmatory_cnn_data_and_split_sha256(
        indexed_images,
        labels,
        training_target_masks=indexed_masks,
        reference_validation_images=indexed_validation_images,
        reference_validation_target_masks=indexed_validation_masks,
        **keyword_arguments,
    )
    legacy_hashes = confirmatory_cnn_data_and_split_sha256(
        images[train_indices],
        labels,
        training_target_masks=masks[train_indices],
        reference_validation_images=images[validation_indices],
        reference_validation_target_masks=masks[validation_indices],
        **keyword_arguments,
    )
    assert indexed_hashes == legacy_hashes


def test_partition_binding_is_unchanged_for_shared_indexed_backing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(702)
    order = np.asarray([5, 1, 7, 3, 9], dtype=np.int64)
    images = rng.integers(0, 256, size=(10, 8, 8, 3), dtype=np.uint8)
    masks = np.zeros((10, 8, 8), dtype=bool)
    masks[:, 2:6, 2:6] = True
    raw_features = rng.normal(size=(10, 6)).astype(np.float32)
    labels = _readonly(np.arange(5, dtype=np.int64))
    injected = _readonly(np.zeros(5, dtype=bool))
    source_indices = _readonly(order)
    common = {
        "role": "audit",
        "source_indices": source_indices,
        "sample_ids": tuple(f"sample-{index}" for index in range(5)),
        "group_ids": tuple(f"group-{index}" for index in range(5)),
        "pre_corruption_labels": labels,
        "observed_labels": labels,
        "is_injected_corruption": injected,
        "corruption_types": ("none",) * 5,
    }
    legacy = ConfirmatoryPartitionInputs(
        **common,
        context_rgb=_readonly(images[order]),
        target_masks=_readonly(masks[order]),
        frozen_features=(
            ConfirmatoryPartitionFeature(
                scenario_id="frozen",
                values=_readonly(raw_features[order].astype(np.float64)),
            ),
        ),
    )
    indexed = ConfirmatoryPartitionInputs(
        **common,
        context_rgb=_indexed(tmp_path, "partition_images", images, order),
        target_masks=_indexed(tmp_path, "partition_masks", masks, order),
        frozen_features=(
            ConfirmatoryPartitionFeature(
                scenario_id="frozen",
                values=_indexed(
                    tmp_path,
                    "partition_features",
                    raw_features,
                    order,
                    logical_dtype=np.float64,
                ),
            ),
        ),
    )
    monkeypatch.setattr(RowIndexedArray, "__array__", _forbid_implicit_array, raising=False)

    legacy.validate()
    indexed.validate()
    assert _partition_content_binding(indexed) == _partition_content_binding(legacy)


def test_image_oof_keeps_indexed_inputs_as_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _inputs(input_variant="context_rgb_plus_binary_target_mask")
    arguments["audit_rgb"] = _indexed(tmp_path, "oof_audit_rgb", arguments["audit_rgb"])
    arguments["audit_target_masks"] = _indexed(
        tmp_path,
        "oof_audit_masks",
        arguments["audit_target_masks"],
    )
    arguments["reference_validation_rgb"] = _indexed(
        tmp_path,
        "oof_validation_rgb",
        arguments["reference_validation_rgb"],
    )
    arguments["reference_validation_target_masks"] = _indexed(
        tmp_path,
        "oof_validation_masks",
        arguments["reference_validation_target_masks"],
    )
    _authorise_fresh(arguments, tmp_path / "indexed-oof")

    class IndexedAdapter(_FastCPUTestAdapter):
        def fit(
            self,
            training_images: NDArray[np.generic] | RowIndexedArray,
            observed_training_labels: NDArray[np.generic],
            **kwargs: Any,
        ) -> Self:
            assert isinstance(training_images, RowIndexedArray)
            assert isinstance(kwargs["training_target_masks"], RowIndexedArray)
            assert isinstance(kwargs["reference_validation_images"], RowIndexedArray)
            return super().fit(training_images, observed_training_labels, **kwargs)

        def predict_proba(
            self,
            images: NDArray[np.generic] | RowIndexedArray,
            *,
            target_masks: NDArray[np.generic] | RowIndexedArray | None = None,
        ) -> NDArray[np.float64]:
            assert isinstance(images, RowIndexedArray)
            assert isinstance(target_masks, RowIndexedArray)
            return super().predict_proba(images, target_masks=target_masks)

    IndexedAdapter.calls.clear()
    monkeypatch.setattr(
        image_oof_module,
        "ConfirmatoryCNNCPUTestOnlyAdapter",
        IndexedAdapter,
    )
    monkeypatch.setattr(RowIndexedArray, "__array__", _forbid_implicit_array, raising=False)

    result = grouped_oof_confirmatory_cnn(**arguments)

    result.validate()
    np.testing.assert_array_equal(
        result.oof_result.coverage_count,
        np.ones(20, dtype=np.int64),
    )
    assert len(IndexedAdapter.calls) == 2


def test_cnn_tensor_preparation_gathers_only_requested_batch_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(703)
    images = rng.integers(0, 256, size=(12, 16, 16, 3), dtype=np.uint8)
    masks = np.zeros((12, 16, 16), dtype=bool)
    masks[:, 4:12, 4:12] = True
    indexed_images = _indexed(tmp_path, "batch_images", images)
    indexed_masks = _indexed(tmp_path, "batch_masks", masks)
    prepared = _prepare_images(
        indexed_images,
        indexed_masks,
        "context_rgb_plus_binary_target_mask",
        name="batch",
    )
    original = RowIndexedArray.gather_rows
    calls: list[tuple[int, int | None]] = []

    def recorded_gather(
        self: RowIndexedArray,
        local_indices: Any = None,
        *,
        max_rows: int | None = None,
        allow_repeated_indices: bool = False,
    ) -> NDArray[Any]:
        assert local_indices is not None
        calls.append((len(local_indices), max_rows))
        return original(
            self,
            local_indices,
            max_rows=max_rows,
            allow_repeated_indices=allow_repeated_indices,
        )

    monkeypatch.setattr(RowIndexedArray, "gather_rows", recorded_gather)
    batch = _batch_tensor(
        prepared,
        np.asarray([9, 2, 7], dtype=np.int64),
        input_size=32,
    )

    assert isinstance(batch, torch.Tensor)
    assert tuple(batch.shape) == (3, 4, 32, 32)
    assert calls == [(3, 3), (3, 3)]


def test_cpu_test_only_cnn_fit_and_predict_accept_indexed_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(704)
    training_images = rng.integers(0, 256, size=(10, 32, 32, 3), dtype=np.uint8)
    validation_images = rng.integers(0, 256, size=(5, 32, 32, 3), dtype=np.uint8)
    training_masks = np.zeros((10, 32, 32), dtype=bool)
    validation_masks = np.zeros((5, 32, 32), dtype=bool)
    training_masks[:, 8:24, 8:24] = True
    validation_masks[:, 8:24, 8:24] = True
    indexed_training = _indexed(tmp_path, "fit_training_images", training_images)
    indexed_training_masks = _indexed(tmp_path, "fit_training_masks", training_masks)
    indexed_validation = _indexed(tmp_path, "fit_validation_images", validation_images)
    indexed_validation_masks = _indexed(
        tmp_path,
        "fit_validation_masks",
        validation_masks,
    )
    monkeypatch.setattr(RowIndexedArray, "__array__", _forbid_implicit_array, raising=False)
    classifier = ConfirmatoryCNNCPUTestOnlyAdapter(
        ConfirmatoryCNNConfig(
            input_variant="context_rgb_plus_binary_target_mask",
            weight_identifier=CPU_TEST_ONLY_WEIGHT_IDENTIFIER,
            input_size=32,
            epochs=1,
            batch_size=5,
            minimum_batch_size=5,
            gradient_accumulation_steps=1,
            early_stopping_patience=2,
            seed=705,
        )
    )
    classifier.fit(
        indexed_training,
        np.tile(np.arange(5, dtype=np.int64), 2),
        training_sample_ids=[f"train-{index}" for index in range(10)],
        training_group_ids=[f"train-group-{index}" for index in range(10)],
        training_target_masks=indexed_training_masks,
        reference_validation_images=indexed_validation,
        reference_validation_labels=np.arange(5, dtype=np.int64),
        reference_validation_sample_ids=[f"validation-{index}" for index in range(5)],
        reference_validation_group_ids=[f"validation-group-{index}" for index in range(5)],
        reference_validation_target_masks=indexed_validation_masks,
        reference_validation_role="reference_validation",
        checkpoint_path=tmp_path / "indexed-cpu.pt",
    )

    probabilities = classifier.predict_proba(
        indexed_validation,
        target_masks=indexed_validation_masks,
        batch_size=2,
    )

    assert probabilities.shape == (5, 5)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_frozen_feature_oof_materialises_only_the_selected_indexed_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _runner_bundle(tmp_path)
    controls = confirmatory_execution_controls_from_frozen_config(bundle.config)
    bridge = bridge_pannuke_confirmatory_inputs(
        bundle.prepared,
        controls,
        pathology_encoder_audit_sha256="b" * 64,
    )
    original_rotation = bridge.rotations[0]
    cell = next(
        value
        for value in controls.plan.cells
        if value.outer_fold == original_rotation.outer_fold
        and value.corruption_cell_id == "symmetric_ten_percent"
        and value.scenario_id == "imagenet_frozen_logistic"
        and value.model_seed == 303
    )
    scenario = controls.scenarios_by_id[cell.scenario_id]
    representation = scenario.representation_id
    legacy_features = original_rotation.frozen_audit_features[representation]
    assert isinstance(legacy_features, np.ndarray)
    indexed_features = _indexed(
        tmp_path,
        "frozen_oof_features",
        legacy_features,
        logical_dtype=np.float64,
    )
    replacement_features = dict(original_rotation.frozen_audit_features)
    replacement_features[representation] = indexed_features
    rotation = replace(
        original_rotation,
        frozen_audit_features=MappingProxyType(replacement_features),
    )
    rotation.validate(controls)
    request = ConfirmatoryCellRequest(
        cell=cell,
        scenario=scenario,
        corruption=rotation.corruptions[cell.corruption_cell_id],
        inputs=ConfirmatoryRunnerInputs.from_rotation(rotation),
        controls=controls,
        checkpoint_directory=tmp_path / "frozen-checkpoints",
        checkpoint_execution_contract=None,
        checkpoint_directives=(),
        cpu_test_only=True,
    )
    monkeypatch.setattr(RowIndexedArray, "__array__", _forbid_implicit_array, raising=False)

    execution = run_confirmatory_frozen_feature_oof(request)
    core_execution = run_core_confirmatory_frozen_feature_oof(request)

    execution.oof_result.validate()
    core_execution.oof_result.validate()
    assert execution.evidence["feature_array_sha256"] == array_artifact_sha256(legacy_features)
    assert core_execution.configuration_sha256 == execution.configuration_sha256
    assert dict(core_execution.evidence) == dict(execution.evidence)
    np.testing.assert_array_equal(
        core_execution.oof_result.fold_id,
        execution.oof_result.fold_id,
    )
    np.testing.assert_allclose(
        core_execution.oof_result.probabilities,
        execution.oof_result.probabilities,
        rtol=0.0,
        atol=0.0,
    )
