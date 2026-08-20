from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from test_pannuke_confirmatory_inputs import _bundle

from histo_audit.experiment import confirmatory_memory_workspace as workspace_module
from histo_audit.experiment import pannuke_confirmatory_inputs as inputs_module
from histo_audit.experiment.confirmatory_memory_workspace import (
    ConfirmatoryMemoryWorkspaceError,
    ConfirmatoryWorkspaceArraySpec,
    ConfirmatoryWorkspaceIndexSpec,
    RowIndexedArray,
    build_confirmatory_memory_workspace,
    build_confirmatory_memory_workspace_plan,
    canonical_confirmatory_memory_workspace_parent,
    close_and_cleanup_confirmatory_memory_workspace,
    validate_confirmatory_memory_workspace_plan,
    verify_confirmatory_memory_workspace,
    workspace_index_id,
)
from histo_audit.experiment.pannuke_confirmatory_inputs import (
    ConfirmatoryFrozenFeatureCacheSpec,
    derive_pannuke_confirmatory_workspace_array_specs,
    derive_pannuke_confirmatory_workspace_index_specs,
    load_pannuke_confirmatory_inputs,
)
from histo_audit.representations.cache_provenance import array_artifact_sha256
from histo_audit.utils.run_tracking import sha256_file


def _write_source(
    root: Path,
    name: str,
    member: str,
    values: np.ndarray[Any, Any],
) -> ConfirmatoryWorkspaceArraySpec:
    source = root / f"{name}.npz"
    np.savez_compressed(source, **{member: values})
    sidecar = source.with_suffix(".npz.metadata.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "array_id": name,
                "cache_npz_sha256": sha256_file(source),
                "cache_array_sha256_by_name": {
                    member: array_artifact_sha256(values),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return ConfirmatoryWorkspaceArraySpec(
        array_id=name,
        source_npz_path=source,
        source_sidecar_path=sidecar,
        expected_source_sha256=sha256_file(source),
        expected_source_sidecar_sha256=sha256_file(sidecar),
        member_name=f"{member}.npy",
        expected_dtype=values.dtype.str,
        expected_shape=tuple(values.shape),
        expected_array_sha256=array_artifact_sha256(values),
    )


def _spec_for_existing(
    *,
    array_id: str,
    source: Path,
    member: str,
) -> ConfirmatoryWorkspaceArraySpec:
    sidecar = source.with_suffix(f"{source.suffix}.metadata.json")
    with np.load(source, allow_pickle=False) as payload:
        values = np.asarray(payload[member])
    return ConfirmatoryWorkspaceArraySpec(
        array_id=array_id,
        source_npz_path=source,
        source_sidecar_path=sidecar,
        expected_source_sha256=sha256_file(source),
        expected_source_sidecar_sha256=sha256_file(sidecar),
        member_name=f"{member}.npy",
        expected_dtype=values.dtype.str,
        expected_shape=tuple(values.shape),
        expected_array_sha256=array_artifact_sha256(values),
    )


def _bundle_specs(arguments: dict[str, Any]) -> tuple[ConfirmatoryWorkspaceArraySpec, ...]:
    crop = Path(arguments["crop_cache_path"])
    specs = list(
        derive_pannuke_confirmatory_workspace_array_specs(
            crop,
            expected_crop_cache_sha256=arguments["expected_crop_cache_sha256"],
            expected_crop_metadata_sha256=arguments["expected_crop_metadata_sha256"],
            frozen_feature_caches=(),
        )
    )
    for feature in arguments["frozen_feature_caches"]:
        source = Path(feature.cache_path)
        with np.load(source, allow_pickle=False) as payload:
            member = "embeddings" if "embeddings" in payload.files else "values"
        specs.append(
            _spec_for_existing(
                array_id=f"feature__{feature.scenario_id}",
                source=source,
                member=member,
            )
        )
    return tuple(specs)


def test_builds_sealed_shared_workspace_and_row_descriptors_without_full_array_api(
    tmp_path: Path,
) -> None:
    source_values = np.arange(96 * 8, dtype=np.float32).reshape(96, 8)
    spec = _write_source(tmp_path, "feature__context", "values", source_values)
    source_before = spec.source_npz_path.read_bytes()
    sidecar_before = spec.source_sidecar_path.read_bytes()

    workspace = build_confirmatory_memory_workspace(
        tmp_path,
        (spec,),
        minimum_free_bytes_after=0,
        chunk_bytes=1024,
    )

    assert workspace.root.parent == canonical_confirmatory_memory_workspace_parent(tmp_path)
    assert workspace.root.name == workspace.workspace_key
    assert len(workspace.workspace_key) == 64
    assert (workspace.root / "artifact_manifest.json").is_file()
    assert (workspace.root / ".immutable.json").is_file()
    assert spec.source_npz_path.read_bytes() == source_before
    assert spec.source_sidecar_path.read_bytes() == sidecar_before
    backing = workspace.arrays[spec.array_id]
    assert isinstance(backing.values, np.memmap)
    assert backing.flags.writeable is False
    assert backing.values.flags.writeable is False
    assert np.array_equal(backing.values, source_values)
    with pytest.raises(ValueError, match="one workspace array row"):
        workspace_module._memmap_array_sha256(backing.values, chunk_bytes=1)

    indices = np.asarray([8, 2, 70, 1], dtype=np.int64)
    indexed = RowIndexedArray(backing, indices, logical_dtype=np.float64)
    assert "__array__" not in type(indexed).__dict__
    assert indexed.shape == (4, 8)
    assert indexed.dtype == np.dtype(np.float64)
    assert indexed.flags.writeable is False
    with pytest.raises(ValueError):
        indexed.source_indices.setflags(write=True)
    gathered = indexed.gather_rows(max_rows=4)
    assert gathered.dtype == np.float64
    assert gathered.flags.writeable is False
    assert np.array_equal(gathered, source_values[indices].astype(np.float64))
    assert indexed.logical_array_sha256() == array_artifact_sha256(gathered)
    bounded_chunks = list(indexed.iter_chunks(320))
    assert [len(value) for value in bounded_chunks] == [2, 2]

    reordered = indexed.select_rows([3, 0, 2])
    assert np.array_equal(reordered.gather_rows(), gathered[[3, 0, 2]])
    with pytest.raises(ValueError, match="repeated"):
        indexed.select_rows([0, 0])
    with pytest.raises(ValueError, match="non-negative integer"):
        indexed.gather_rows(max_rows=float("nan"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="declared maximum"):
        indexed.gather_rows(slice(None), max_rows=1)
    with pytest.raises(OverflowError, match="int64"):
        indexed.select_rows(np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64))
    with pytest.raises(ValueError, match="one logical row"):
        next(indexed.iter_chunks(1))
    repeated = indexed.select_rows([0, 0], allow_repeated_indices=True)
    assert np.array_equal(
        repeated.gather_rows(allow_repeated_indices=True),
        gathered[[0, 0]],
    )

    receipt = json.loads((workspace.root / "workspace_receipt.json").read_text("utf-8"))
    assert receipt["source_annotations_modified"] is False
    assert receipt["scientific_outcomes_read"] is False
    assert receipt["arrays"][0]["source_member_compression"] == "deflated"
    assert receipt["arrays"][0]["raw_array_sha256"] == spec.expected_array_sha256
    assert receipt["capacity"]["minimum_free_bytes_after"] == 0


def test_pannuke_loader_uses_one_shared_backing_and_lightweight_partition_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, source_arrays, _ = _bundle(tmp_path)
    specs = _bundle_specs(arguments)
    index_specs = derive_pannuke_confirmatory_workspace_index_specs(
        arguments["crop_cache_path"],
        confirmatory_config=arguments["confirmatory_config"],
        expected_config_sha256=arguments["expected_config_sha256"],
        expected_crop_cache_sha256=arguments["expected_crop_cache_sha256"],
        expected_crop_metadata_sha256=arguments["expected_crop_metadata_sha256"],
        expected_manifest_sha256=arguments["expected_manifest_sha256"],
        expected_raw_inventory_sha256=arguments["expected_raw_inventory_sha256"],
    )
    maximum_workspace_bytes = 64 * 1024 * 1024
    plan = build_confirmatory_memory_workspace_plan(
        specs,
        index_specs,
        minimum_free_bytes_after=0,
        maximum_workspace_bytes=maximum_workspace_bytes,
    )
    workspace = build_confirmatory_memory_workspace(
        tmp_path,
        specs,
        minimum_free_bytes_after=0,
        resource_input_workspace_plan=plan,
        index_specs=index_specs,
        maximum_workspace_bytes=maximum_workspace_bytes,
    )
    arguments["memory_workspace"] = workspace
    captured: dict[str, Any] = {}
    original_crop_loader = inputs_module._load_crop_cache
    original_np_load = np.load
    crop_path = Path(arguments["crop_cache_path"]).resolve()

    def capture_crop(*args: Any, **kwargs: Any) -> Any:
        cache = original_crop_loader(*args, **kwargs)
        captured["cache"] = cache
        return cache

    def forbid_crop_npz_reload(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)) and Path(file).resolve() == crop_path:
            raise AssertionError("workspace-backed input loading reopened the crop NPZ")
        return original_np_load(file, *args, **kwargs)

    monkeypatch.setattr(inputs_module, "_load_crop_cache", capture_crop)
    monkeypatch.setattr(np, "load", forbid_crop_npz_reload)

    result = load_pannuke_confirmatory_inputs(**arguments)

    assert len(specs) == 12
    assert result.memory_workspace_path == str(workspace.root)
    assert result.memory_workspace_receipt_sha256 == workspace.receipt_sha256
    assert result.memory_workspace_artifact_root_sha256 == workspace.artifact_root_sha256
    assert result.memory_workspace_plan_sha256 == plan["plan_without_self_hash_sha256"]
    assert workspace.resource_input_workspace_plan_sha256 == plan["plan_without_self_hash_sha256"]
    crop_cache = captured["cache"]
    for field in (
        "sample_ids",
        "group_ids",
        "official_folds",
        "pre_corruption_labels",
        "confirmatory_eligible",
        "identity_verified",
        "primary_eligible",
    ):
        values = getattr(crop_cache, field)
        assert isinstance(values, np.memmap)
        assert values.flags.writeable is False
        assert np.array_equal(values, source_arrays[field])
    expected_indices = {(value.outer_fold, value.role): value for value in index_specs}
    rgb_backings: set[str] = set()
    mask_backings: set[str] = set()
    for rotation in result.rotations:
        for partition in (
            rotation.audit,
            rotation.reference_validation,
            rotation.final_reference,
        ):
            assert isinstance(partition.context_rgb, RowIndexedArray)
            assert isinstance(partition.target_masks, RowIndexedArray)
            assert partition.context_rgb.flags.writeable is False
            assert partition.target_masks.flags.writeable is False
            assert not hasattr(partition.context_rgb, "backing")
            assert not hasattr(partition.context_rgb, "backing_path")
            assert not hasattr(partition.context_rgb, "values")
            rgb_backings.add(partition.context_rgb.backing_array_id)
            mask_backings.add(partition.target_masks.backing_array_id)
            expected_rgb = source_arrays["context_rgb"][partition.source_indices]
            expected_masks = source_arrays["target_masks"][partition.source_indices]
            assert np.array_equal(partition.context_rgb.gather_rows(), expected_rgb)
            assert np.array_equal(partition.target_masks.gather_rows(), expected_masks)
            binding = expected_indices[(rotation.outer_fold, partition.role)]
            assert binding.row_count == len(partition.source_indices)
            assert binding.source_indices_sha256 == array_artifact_sha256(partition.source_indices)
            workspace_indices = workspace.index_arrays[
                workspace_index_id(rotation.outer_fold, partition.role)
            ]
            assert workspace_indices.flags.writeable is False
            assert np.array_equal(workspace_indices, partition.source_indices)
            for feature in partition.frozen_features:
                assert isinstance(feature.values, RowIndexedArray)
                assert feature.values.dtype == np.dtype(np.float64)
    assert rgb_backings == {"context_rgb"}
    assert mask_backings == {"target_masks"}
    audit = result.rotations[0].audit
    reverse = np.arange(len(audit.source_indices) - 1, -1, -1, dtype=np.int64)
    duplicate_indices = np.array(audit.source_indices, dtype=np.int64, copy=True)
    duplicate_indices[0] = duplicate_indices[1]
    duplicate_indices.setflags(write=False)
    with pytest.raises(ValueError, match="source indices are misaligned"):
        replace(audit, source_indices=duplicate_indices).validate()
    assert isinstance(audit.context_rgb, RowIndexedArray)
    with pytest.raises(ValueError, match="authoritative source indices"):
        replace(audit, context_rgb=audit.context_rgb.select_rows(reverse)).validate()
    assert isinstance(audit.target_masks, RowIndexedArray)
    with pytest.raises(ValueError, match="authoritative source indices"):
        replace(audit, target_masks=audit.target_masks.select_rows(reverse)).validate()
    assert isinstance(audit.frozen_features[0].values, RowIndexedArray)
    with pytest.raises(ValueError, match="authoritative source indices"):
        replace(
            audit,
            frozen_features=(
                replace(
                    audit.frozen_features[0],
                    values=audit.frozen_features[0].values.select_rows(reverse),
                ),
                *audit.frozen_features[1:],
            ),
        ).validate()
    assert (
        validate_confirmatory_memory_workspace_plan(
            plan,
            specs,
            index_specs,
            minimum_free_bytes_after=0,
            maximum_workspace_bytes=maximum_workspace_bytes,
        )
        == plan
    )


def test_exact_twelve_workspace_normalises_fixed_width_byte_identifiers_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, source_arrays, _ = _bundle(tmp_path)
    crop_path = Path(arguments["crop_cache_path"]).resolve()
    metadata_path = crop_path.with_suffix(".npz.metadata.json")
    expected_sample_ids = np.asarray(source_arrays["sample_ids"], dtype=np.str_)
    expected_group_ids = np.asarray(source_arrays["group_ids"], dtype=np.str_)
    byte_crop_arrays = {
        **source_arrays,
        "sample_ids": expected_sample_ids.astype("S64"),
        "group_ids": expected_group_ids.astype("S64"),
    }
    np.savez_compressed(crop_path, **byte_crop_arrays)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["cache_npz_sha256"] = sha256_file(crop_path)
    metadata["cache_array_sha256_by_name"] = {
        name: array_artifact_sha256(values) for name, values in byte_crop_arrays.items()
    }
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    arguments["expected_crop_cache_sha256"] = sha256_file(crop_path)
    arguments["expected_crop_metadata_sha256"] = sha256_file(metadata_path)

    specs = _bundle_specs(arguments)
    index_specs = derive_pannuke_confirmatory_workspace_index_specs(
        crop_path,
        confirmatory_config=arguments["confirmatory_config"],
        expected_config_sha256=arguments["expected_config_sha256"],
        expected_crop_cache_sha256=arguments["expected_crop_cache_sha256"],
        expected_crop_metadata_sha256=arguments["expected_crop_metadata_sha256"],
        expected_manifest_sha256=arguments["expected_manifest_sha256"],
        expected_raw_inventory_sha256=arguments["expected_raw_inventory_sha256"],
    )
    maximum_workspace_bytes = 64 * 1024 * 1024
    plan = build_confirmatory_memory_workspace_plan(
        specs,
        index_specs,
        minimum_free_bytes_after=0,
        maximum_workspace_bytes=maximum_workspace_bytes,
    )
    workspace = build_confirmatory_memory_workspace(
        tmp_path,
        specs,
        minimum_free_bytes_after=0,
        resource_input_workspace_plan=plan,
        index_specs=index_specs,
        maximum_workspace_bytes=maximum_workspace_bytes,
    )
    arguments["memory_workspace"] = workspace
    captured: dict[str, Any] = {}
    original_crop_loader = inputs_module._load_crop_cache
    original_np_load = np.load

    def capture_crop(*args: Any, **kwargs: Any) -> Any:
        cache = original_crop_loader(*args, **kwargs)
        captured["cache"] = cache
        return cache

    def forbid_crop_npz_reload(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)) and Path(file).resolve() == crop_path:
            raise AssertionError("workspace-backed input loading reopened the crop NPZ")
        return original_np_load(file, *args, **kwargs)

    monkeypatch.setattr(inputs_module, "_load_crop_cache", capture_crop)
    monkeypatch.setattr(np, "load", forbid_crop_npz_reload)
    try:
        result = load_pannuke_confirmatory_inputs(**arguments)
        cache = captured["cache"]
        assert len(specs) == 12
        assert len(index_specs) == 9
        assert cache.sample_ids.dtype.kind == "U"
        assert cache.group_ids.dtype.kind == "U"
        assert not isinstance(cache.sample_ids, np.memmap)
        assert not isinstance(cache.group_ids, np.memmap)
        assert cache.sample_ids.flags.writeable is False
        assert cache.group_ids.flags.writeable is False
        assert np.array_equal(cache.sample_ids, expected_sample_ids)
        assert np.array_equal(cache.group_ids, expected_group_ids)
        for field in (
            "official_folds",
            "pre_corruption_labels",
            "confirmatory_eligible",
            "identity_verified",
            "primary_eligible",
        ):
            assert isinstance(getattr(cache, field), np.memmap)
        expected_feature_order = tuple(
            sorted(value.scenario_id for value in arguments["frozen_feature_caches"])
        )
        for rotation in result.rotations:
            for partition in (
                rotation.audit,
                rotation.reference_validation,
                rotation.final_reference,
            ):
                assert isinstance(partition.context_rgb, RowIndexedArray)
                assert isinstance(partition.target_masks, RowIndexedArray)
                assert (
                    tuple(value.scenario_id for value in partition.frozen_features)
                    == expected_feature_order
                )
                assert all(not value.startswith("b'") for value in partition.sample_ids)
                assert all(not value.startswith("b'") for value in partition.group_ids)
    finally:
        workspace.close()


def test_workspace_plan_requires_exact_nine_index_bindings_and_byte_ceiling(
    tmp_path: Path,
) -> None:
    values = np.arange(60, dtype=np.float32).reshape(15, 4)
    specs = tuple(
        _write_source(
            tmp_path,
            f"feature_{index:02d}",
            "values",
            values + index,
        )
        for index in range(12)
    )
    roles = ("audit", "reference_validation", "final_reference")
    source_indices = {
        "audit": np.arange(0, 9, dtype=np.int64),
        "reference_validation": np.arange(9, 10, dtype=np.int64),
        "final_reference": np.arange(10, 15, dtype=np.int64),
    }
    index_specs = tuple(
        ConfirmatoryWorkspaceIndexSpec(
            outer_fold=outer_fold,
            role=cast(Any, role),
            source_indices=source_indices[role],
        )
        for outer_fold in (1, 2, 3)
        for role in roles
    )
    with pytest.raises(ValueError):
        index_specs[0].source_indices.setflags(write=True)
    plan = build_confirmatory_memory_workspace_plan(
        specs,
        index_specs,
        minimum_free_bytes_after=100,
        maximum_workspace_bytes=32 * 1024 * 1024,
    )
    assert len(plan["arrays"]) == 12
    assert len(plan["partition_index_specs"]) == 9
    assert plan["minimum_free_bytes_after"] == 100
    assert plan["planned_workspace_bytes"] <= plan["maximum_workspace_bytes"]
    with pytest.raises(ValueError, match="exactly all nine"):
        build_confirmatory_memory_workspace_plan(
            specs,
            index_specs[:-1],
            minimum_free_bytes_after=100,
            maximum_workspace_bytes=32 * 1024 * 1024,
        )
    with pytest.raises(ConfirmatoryMemoryWorkspaceError, match="ceiling"):
        build_confirmatory_memory_workspace_plan(
            specs,
            index_specs,
            minimum_free_bytes_after=100,
            maximum_workspace_bytes=1,
        )


def test_workspace_is_fresh_only_and_cleanup_requires_builder_ownership(
    tmp_path: Path,
) -> None:
    values = np.arange(120, dtype=np.uint8).reshape(10, 4, 3)
    spec = _write_source(tmp_path, "context_rgb", "context_rgb", values)
    first = build_confirmatory_memory_workspace(
        tmp_path,
        (spec,),
        minimum_free_bytes_after=0,
    )
    with pytest.raises(ConfirmatoryMemoryWorkspaceError, match="forbids reuse"):
        build_confirmatory_memory_workspace(
            tmp_path,
            (spec,),
            minimum_free_bytes_after=0,
        )
    read_only = verify_confirmatory_memory_workspace(tmp_path, first.root, (spec,))
    assert read_only.cleanup_ownership_token is None
    with pytest.raises(ConfirmatoryMemoryWorkspaceError, match="no cleanup authority"):
        close_and_cleanup_confirmatory_memory_workspace(
            tmp_path,
            read_only,
            (spec,),
        )
    read_only.close()
    cleanup = close_and_cleanup_confirmatory_memory_workspace(tmp_path, first, (spec,))
    assert cleanup["workspace_removed"] is True
    assert not first.root.exists()

    second = build_confirmatory_memory_workspace(
        tmp_path,
        (spec,),
        minimum_free_bytes_after=0,
    )
    with second.arrays["context_rgb"].path.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        original = stream.read(1)
        stream.seek(-1, os.SEEK_END)
        stream.write(bytes([original[0] ^ 1]))
        stream.flush()
        os.fsync(stream.fileno())
    with pytest.raises(ConfirmatoryMemoryWorkspaceError, match="manifest"):
        verify_confirmatory_memory_workspace(tmp_path, second.root, (spec,))
    with pytest.raises(ConfirmatoryMemoryWorkspaceError):
        build_confirmatory_memory_workspace(
            tmp_path,
            (spec,),
            minimum_free_bytes_after=0,
        )
    assert second.root.is_dir()


def test_capacity_failure_happens_before_workspace_or_lease_publication(tmp_path: Path) -> None:
    spec = _write_source(
        tmp_path,
        "target_masks",
        "target_masks",
        np.ones((20, 8, 8), dtype=bool),
    )
    parent = canonical_confirmatory_memory_workspace_parent(tmp_path)

    with pytest.raises(ConfirmatoryMemoryWorkspaceError, match="capacity"):
        build_confirmatory_memory_workspace(
            tmp_path,
            (spec,),
            minimum_free_bytes_after=10**30,
        )

    assert not parent.exists() or list(parent.iterdir()) == []


def test_lease_write_failure_closes_and_removes_owned_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _write_source(
        tmp_path,
        "target_masks",
        "target_masks",
        np.ones((20, 8, 8), dtype=bool),
    )
    parent = canonical_confirmatory_memory_workspace_parent(tmp_path)

    def fail_write(*args: Any, **kwargs: Any) -> int:
        raise OSError("injected lease write failure")

    monkeypatch.setattr(workspace_module.os, "write", fail_write)
    with pytest.raises(OSError, match="injected lease"):
        build_confirmatory_memory_workspace(
            tmp_path,
            (spec,),
            minimum_free_bytes_after=0,
        )

    assert parent.is_dir()
    assert list(parent.iterdir()) == []


def test_owned_staging_is_cleaned_after_extraction_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _write_source(
        tmp_path,
        "context_rgb",
        "context_rgb",
        np.ones((20, 8, 8, 3), dtype=np.uint8),
    )
    parent = canonical_confirmatory_memory_workspace_parent(tmp_path)

    def fail_extract(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("injected extraction failure")

    monkeypatch.setattr(workspace_module, "_extract_member", fail_extract)
    with pytest.raises(RuntimeError, match="injected"):
        build_confirmatory_memory_workspace(
            tmp_path,
            (spec,),
            minimum_free_bytes_after=0,
        )

    assert parent.is_dir()
    assert list(parent.iterdir()) == []
    assert spec.source_npz_path.is_file()


def test_verifier_rejects_oversized_metadata_before_json_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _write_source(
        tmp_path,
        "context_rgb",
        "context_rgb",
        np.ones((20, 8, 8, 3), dtype=np.uint8),
    )
    workspace = build_confirmatory_memory_workspace(
        tmp_path,
        (spec,),
        minimum_free_bytes_after=0,
    )
    workspace.close()
    monkeypatch.setattr(workspace_module, "_METADATA_CAPACITY_ALLOWANCE_BYTES", 1)

    with pytest.raises(ConfirmatoryMemoryWorkspaceError, match="bounded allowance"):
        verify_confirmatory_memory_workspace(tmp_path, workspace.root, (spec,))


@pytest.mark.parametrize("bad_kind", ["object", "fortran", "wrong_hash"])
def test_unsafe_or_mismatched_npy_member_fails_closed(
    tmp_path: Path,
    bad_kind: str,
) -> None:
    if bad_kind == "object":
        values = np.asarray([{"unsafe": True}], dtype=object)
    elif bad_kind == "fortran":
        values = np.asfortranarray(np.arange(60, dtype=np.float32).reshape(10, 6))
    else:
        values = np.arange(60, dtype=np.float32).reshape(10, 6)
    if bad_kind == "object":
        source = tmp_path / "unsafe_object.npz"
        np.savez_compressed(source, values=values)
        sidecar = source.with_suffix(".npz.metadata.json")
        sidecar.write_text("{}", encoding="utf-8")
        spec = ConfirmatoryWorkspaceArraySpec(
            array_id="unsafe_object",
            source_npz_path=source,
            source_sidecar_path=sidecar,
            expected_source_sha256=sha256_file(source),
            expected_source_sidecar_sha256=sha256_file(sidecar),
            member_name="values.npy",
            expected_dtype=values.dtype.str,
            expected_shape=tuple(values.shape),
            expected_array_sha256="0" * 64,
        )
    else:
        spec = _write_source(tmp_path, f"unsafe_{bad_kind}", "values", values)
    if bad_kind == "wrong_hash":
        spec = replace(spec, expected_array_sha256="0" * 64)

    with pytest.raises((ConfirmatoryMemoryWorkspaceError, ValueError)):
        build_confirmatory_memory_workspace(
            tmp_path,
            (spec,),
            minimum_free_bytes_after=0,
        )

    parent = canonical_confirmatory_memory_workspace_parent(tmp_path)
    assert not parent.exists() or list(parent.iterdir()) == []


def test_duplicate_ids_and_source_members_are_rejected(tmp_path: Path) -> None:
    values = np.arange(20, dtype=np.float32).reshape(5, 4)
    first = _write_source(tmp_path, "feature_a", "values", values)
    duplicate_id = ConfirmatoryWorkspaceArraySpec(
        array_id=first.array_id,
        source_npz_path=first.source_npz_path,
        source_sidecar_path=first.source_sidecar_path,
        expected_source_sha256=first.expected_source_sha256,
        expected_source_sidecar_sha256=first.expected_source_sidecar_sha256,
        member_name=first.member_name,
        expected_dtype=first.expected_dtype,
        expected_shape=first.expected_shape,
        expected_array_sha256=first.expected_array_sha256,
    )
    with pytest.raises(ValueError, match="duplicate workspace array ID"):
        build_confirmatory_memory_workspace(
            tmp_path,
            (first, duplicate_id),
            minimum_free_bytes_after=0,
        )
    duplicate_member = ConfirmatoryWorkspaceArraySpec(
        array_id="feature_b",
        source_npz_path=first.source_npz_path,
        source_sidecar_path=first.source_sidecar_path,
        expected_source_sha256=first.expected_source_sha256,
        expected_source_sidecar_sha256=first.expected_source_sidecar_sha256,
        member_name=first.member_name,
        expected_dtype=first.expected_dtype,
        expected_shape=first.expected_shape,
        expected_array_sha256=first.expected_array_sha256,
    )
    with pytest.raises(ValueError, match="source/member"):
        build_confirmatory_memory_workspace(
            tmp_path,
            (first, duplicate_member),
            minimum_free_bytes_after=0,
        )


def test_shared_source_cannot_claim_conflicting_expected_sha256(tmp_path: Path) -> None:
    source = tmp_path / "shared.npz"
    values_a = np.arange(20, dtype=np.float32).reshape(5, 4)
    values_b = values_a + 1
    np.savez_compressed(source, first=values_a, second=values_b)
    sidecar = source.with_suffix(".npz.metadata.json")
    sidecar.write_text("{}", encoding="utf-8")
    common = {
        "source_npz_path": source,
        "source_sidecar_path": sidecar,
        "expected_source_sidecar_sha256": sha256_file(sidecar),
    }
    specs = (
        ConfirmatoryWorkspaceArraySpec(
            array_id="first",
            expected_source_sha256=sha256_file(source),
            member_name="first.npy",
            expected_dtype=values_a.dtype.str,
            expected_shape=values_a.shape,
            expected_array_sha256=array_artifact_sha256(values_a),
            **common,
        ),
        ConfirmatoryWorkspaceArraySpec(
            array_id="second",
            expected_source_sha256="0" * 64,
            member_name="second.npy",
            expected_dtype=values_b.dtype.str,
            expected_shape=values_b.shape,
            expected_array_sha256=array_artifact_sha256(values_b),
            **common,
        ),
    )
    with pytest.raises(ValueError, match="conflicting expected SHA"):
        build_confirmatory_memory_workspace(
            tmp_path,
            specs,
            minimum_free_bytes_after=0,
        )


def test_source_symlink_is_rejected_when_platform_can_create_it(tmp_path: Path) -> None:
    values = np.arange(20, dtype=np.float32).reshape(5, 4)
    original = _write_source(tmp_path, "original", "values", values)
    linked_source = tmp_path / "linked.npz"
    try:
        linked_source.symlink_to(original.source_npz_path)
    except OSError:
        pytest.skip("platform does not permit an unprivileged file symlink")
    linked = ConfirmatoryWorkspaceArraySpec(
        array_id="linked",
        source_npz_path=linked_source,
        source_sidecar_path=original.source_sidecar_path,
        expected_source_sha256=original.expected_source_sha256,
        expected_source_sidecar_sha256=original.expected_source_sidecar_sha256,
        member_name=original.member_name,
        expected_dtype=original.expected_dtype,
        expected_shape=original.expected_shape,
        expected_array_sha256=original.expected_array_sha256,
    )
    with pytest.raises(ConfirmatoryMemoryWorkspaceError, match="plain physical"):
        build_confirmatory_memory_workspace(
            tmp_path,
            (linked,),
            minimum_free_bytes_after=0,
        )


@pytest.mark.parametrize("victim_role", ["source", "sidecar"])
def test_final_workspace_verification_rejects_checksum_identical_reparse_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    victim_role: str,
) -> None:
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    spec = _write_source(tmp_path, "final_swap", "values", values)
    victim = spec.source_npz_path if victim_role == "source" else spec.source_sidecar_path
    replacement = tmp_path / f"{victim.name}.replacement"
    replacement.write_bytes(victim.read_bytes())
    probe = tmp_path / f"{victim.name}.symlink-probe"
    try:
        probe.symlink_to(replacement)
    except OSError:
        pytest.skip("platform does not permit an unprivileged file symlink")
    probe.unlink()

    real_source_hashes = workspace_module._source_hashes
    hash_passes = 0

    def swap_on_final_verification(
        specs: tuple[ConfirmatoryWorkspaceArraySpec, ...],
    ) -> tuple[dict[Path, str], dict[Path, str]]:
        nonlocal hash_passes
        hash_passes += 1
        if hash_passes == 4:
            victim.unlink()
            victim.symlink_to(replacement)
        return real_source_hashes(specs)

    monkeypatch.setattr(workspace_module, "_source_hashes", swap_on_final_verification)
    with pytest.raises(ConfirmatoryMemoryWorkspaceError, match="plain physical"):
        build_confirmatory_memory_workspace(
            tmp_path,
            (spec,),
            minimum_free_bytes_after=0,
        )

    assert hash_passes == 4
    assert victim.is_symlink()
    parent = canonical_confirmatory_memory_workspace_parent(tmp_path)
    assert list(parent.iterdir()) == []


@pytest.mark.parametrize("victim_role", ["source", "sidecar"])
def test_source_hashing_rejects_reparse_swap_immediately_after_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    victim_role: str,
) -> None:
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    spec = _write_source(tmp_path, "during_hash_swap", "values", values)
    victim = spec.source_npz_path if victim_role == "source" else spec.source_sidecar_path
    replacement = tmp_path / f"{victim.name}.replacement"
    replacement.write_bytes(victim.read_bytes())
    probe = tmp_path / f"{victim.name}.symlink-probe"
    try:
        probe.symlink_to(replacement)
    except OSError:
        pytest.skip("platform does not permit an unprivileged file symlink")
    probe.unlink()

    real_sha256_file = workspace_module.sha256_file
    swapped = False

    def swap_after_digest(path: str | Path) -> str:
        nonlocal swapped
        digest = real_sha256_file(path)
        if not swapped and Path(path) == victim:
            swapped = True
            victim.unlink()
            victim.symlink_to(replacement)
        return digest

    monkeypatch.setattr(workspace_module, "sha256_file", swap_after_digest)
    with pytest.raises(ConfirmatoryMemoryWorkspaceError, match="plain physical"):
        workspace_module._source_hashes((spec,))

    assert swapped is True
    assert victim.is_symlink()


def test_final_verification_rejects_checksum_identical_parent_reparse_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_directory = tmp_path / "bound_sources"
    source_directory.mkdir()
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    spec = _write_source(source_directory, "parent_swap", "values", values)
    replacement_directory = tmp_path / "replacement_sources"
    replacement_directory.mkdir()
    for source in (spec.source_npz_path, spec.source_sidecar_path):
        (replacement_directory / source.name).write_bytes(source.read_bytes())
    probe = tmp_path / "directory-symlink-probe"
    try:
        probe.symlink_to(replacement_directory, target_is_directory=True)
    except OSError:
        pytest.skip("platform does not permit an unprivileged directory symlink")
    probe.unlink()

    real_source_hashes = workspace_module._source_hashes
    hash_passes = 0

    def swap_parent_on_final_verification(
        specs: tuple[ConfirmatoryWorkspaceArraySpec, ...],
    ) -> tuple[dict[Path, str], dict[Path, str]]:
        nonlocal hash_passes
        hash_passes += 1
        if hash_passes == 4:
            source_directory.rename(tmp_path / "parked_sources")
            source_directory.symlink_to(
                replacement_directory,
                target_is_directory=True,
            )
        return real_source_hashes(specs)

    monkeypatch.setattr(workspace_module, "_source_hashes", swap_parent_on_final_verification)
    with pytest.raises(ConfirmatoryMemoryWorkspaceError, match="plain physical"):
        build_confirmatory_memory_workspace(
            tmp_path,
            (spec,),
            minimum_free_bytes_after=0,
        )

    assert hash_passes == 4
    assert source_directory.is_symlink()
    parent = canonical_confirmatory_memory_workspace_parent(tmp_path)
    assert list(parent.iterdir()) == []


def test_workspace_parent_rejects_in_root_directory_reparse_chain(tmp_path: Path) -> None:
    real_artifacts = tmp_path / "real_artifacts"
    real_artifacts.mkdir()
    linked_artifacts = tmp_path / "artifacts"
    try:
        linked_artifacts.symlink_to(real_artifacts, target_is_directory=True)
    except OSError:
        pytest.skip("platform does not permit an unprivileged directory symlink")

    with pytest.raises(ConfirmatoryMemoryWorkspaceError, match="reparse"):
        canonical_confirmatory_memory_workspace_parent(tmp_path)


def test_row_descriptor_detects_backing_tamper_after_verification(tmp_path: Path) -> None:
    values = np.arange(96, dtype=np.float32).reshape(12, 8)
    spec = _write_source(tmp_path, "feature", "values", values)
    workspace = build_confirmatory_memory_workspace(
        tmp_path,
        (spec,),
        minimum_free_bytes_after=0,
    )
    descriptor = RowIndexedArray(
        workspace.arrays["feature"],
        np.asarray([0, 3, 7], dtype=np.int64),
    )
    with workspace.arrays["feature"].path.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        original = stream.read(1)
        stream.seek(-1, os.SEEK_END)
        stream.write(bytes([original[0] ^ 1]))
        stream.flush()
        os.fsync(stream.fileno())

    with pytest.raises(ConfirmatoryMemoryWorkspaceError, match="changed after verification"):
        descriptor.gather_rows()


def test_derives_array_specs_from_sidecars_without_loading_heavy_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crop = tmp_path / "crop.npz"
    rgb = np.arange(12 * 4 * 4 * 3, dtype=np.uint8).reshape(12, 4, 4, 3)
    masks = np.ones((12, 4, 4), dtype=bool)
    sample_ids = np.asarray([f"sample-{index}" for index in range(12)], dtype=np.str_)
    group_ids = np.asarray([f"group-{index // 2}" for index in range(12)], dtype=np.str_)
    official_folds = np.tile(np.asarray([1, 2, 3], dtype=np.int16), 4)
    pre_corruption_labels = np.asarray([index % 5 for index in range(12)], dtype=np.int64)
    identity_verified = np.ones(12, dtype=bool)
    primary_eligible = np.ones(12, dtype=bool)
    confirmatory_eligible = np.ones(12, dtype=bool)
    crop_arrays = {
        "sample_ids": sample_ids,
        "group_ids": group_ids,
        "official_folds": official_folds,
        "pre_corruption_labels": pre_corruption_labels,
        "confirmatory_eligible": confirmatory_eligible,
        "context_rgb": rgb,
        "identity_verified": identity_verified,
        "primary_eligible": primary_eligible,
        "target_masks": masks,
    }
    np.savez_compressed(crop, **crop_arrays)
    crop_sidecar = crop.with_suffix(".npz.metadata.json")
    crop_sidecar.write_text(
        json.dumps(
            {
                "cache_npz_sha256": sha256_file(crop),
                "cache_array_sha256_by_name": {
                    name: array_artifact_sha256(values) for name, values in crop_arrays.items()
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    feature = tmp_path / "feature.npz"
    matrix = np.arange(12 * 6, dtype=np.float32).reshape(12, 6)
    sample_ids = np.asarray([f"s{index}" for index in range(12)], dtype=np.str_)
    np.savez_compressed(feature, values=matrix, sample_ids=sample_ids)
    feature_sidecar = feature.with_suffix(".npz.metadata.json")
    feature_sidecar.write_text(
        json.dumps(
            {
                "cache_npz_sha256": sha256_file(feature),
                "cache_array_sha256_by_name": {
                    "values": array_artifact_sha256(matrix),
                    "sample_ids": array_artifact_sha256(sample_ids),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    feature_spec = ConfirmatoryFrozenFeatureCacheSpec(
        scenario_id="frozen_example",
        cache_path=feature,
        expected_cache_sha256=sha256_file(feature),
        expected_metadata_sha256=sha256_file(feature_sidecar),
    )
    hash_calls: list[Path] = []
    original_sha256_file = inputs_module.sha256_file

    def count_sha256_file(path: str | Path) -> str:
        hash_calls.append(Path(path).resolve())
        return original_sha256_file(path)

    monkeypatch.setattr(inputs_module, "sha256_file", count_sha256_file)

    specs = derive_pannuke_confirmatory_workspace_array_specs(
        crop,
        expected_crop_cache_sha256=sha256_file(crop),
        expected_crop_metadata_sha256=sha256_file(crop_sidecar),
        frozen_feature_caches=(feature_spec,),
    )

    assert len(hash_calls) == 4
    assert sorted(hash_calls) == sorted(
        (
            crop.resolve(),
            crop_sidecar.resolve(),
            feature.resolve(),
            feature_sidecar.resolve(),
        )
    )
    assert [value.array_id for value in specs] == [
        "confirmatory_eligible",
        "context_rgb",
        "feature__frozen_example",
        "group_ids",
        "identity_verified",
        "official_folds",
        "pre_corruption_labels",
        "primary_eligible",
        "sample_ids",
        "target_masks",
    ]
    by_id = {value.array_id: value for value in specs}
    assert by_id["context_rgb"].expected_shape == rgb.shape
    assert by_id["target_masks"].expected_array_sha256 == array_artifact_sha256(masks)
    assert by_id["feature__frozen_example"].expected_dtype == matrix.dtype.str


def test_full_memmap_hash_and_semantic_scan_have_bounded_final_rss(
    tmp_path: Path,
) -> None:
    # The subprocess scans all 128 MiB twice through disposable mappings.  Closing
    # each mapping must release the file-backed working set.
    script = """
import json
import os
import tracemalloc
from pathlib import Path
import numpy as np
import psutil
from histo_audit.experiment.confirmatory_memory_workspace import ReadOnlyBackingArray, RowIndexedArray, _memmap_array_sha256
from histo_audit.experiment.pannuke_confirmatory_inputs import _all_finite

root = Path(os.environ["MEMORY_TEST_ROOT"])
path = root / "large.npy"
values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(16384, 2048))
values.flush()
del values
process = psutil.Process()
before = process.memory_info().rss
peak_before = getattr(process.memory_info(), "peak_wset", before)
mapped = np.load(path, mmap_mode="r", allow_pickle=False)
backing = ReadOnlyBackingArray(
    array_id="large",
    path=path,
    values=mapped,
    file_sha256="0" * 64,
    raw_array_sha256="1" * 64,
    source_npz_sha256="2" * 64,
    source_sidecar_sha256="3" * 64,
    source_member_name="large.npy",
)
descriptor = RowIndexedArray(backing, np.arange(len(mapped), dtype=np.int64))
verified_digest = _memmap_array_sha256(mapped, chunk_bytes=4 * 1024 * 1024)
digest = descriptor.logical_array_sha256(4 * 1024 * 1024)
logical_descriptor = RowIndexedArray(
    backing,
    np.arange(len(mapped), dtype=np.int64),
    logical_dtype=np.float64,
)
tracemalloc.start()
logical_digest = logical_descriptor.logical_array_sha256(32 * 1024 * 1024)
_, logical_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
finite = _all_finite(backing)
memory_after = process.memory_info()
after = memory_after.rss
peak_after = getattr(memory_after, "peak_wset", after)
print(json.dumps({"delta": after - before, "peak_delta": peak_after - peak_before, "shape": descriptor.shape, "digest": digest, "verified_digest": verified_digest, "logical_digest": logical_digest, "logical_peak": logical_peak, "finite": finite}))
"""
    environment = dict(os.environ)
    environment["MEMORY_TEST_ROOT"] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    evidence = json.loads(completed.stdout)
    assert evidence["shape"] == [16384, 2048]
    assert len(evidence["digest"]) == 64
    assert evidence["verified_digest"] == evidence["digest"]
    assert len(evidence["logical_digest"]) == 64
    assert evidence["logical_peak"] < 40 * 1024 * 1024
    assert evidence["finite"] is True
    assert evidence["delta"] < 32 * 1024 * 1024
    assert evidence["peak_delta"] < 64 * 1024 * 1024
