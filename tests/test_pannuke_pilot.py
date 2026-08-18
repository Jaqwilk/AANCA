"""Focused all-class real-pilot orchestration test with a fake official embedding cache."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import histo_audit.experiment.pilot as pilot_module
from histo_audit.config import load_config
from histo_audit.experiment import (
    build_pannuke_pilot_development_manifest_view,
    reconcile_pilot_audit_evidence,
    run_pannuke_pilot,
)
from histo_audit.pannuke import build_nucleus_manifest, sha256_file, validate_pannuke
from histo_audit.representations import EmbeddingResult
from histo_audit.utils.run_tracking import verify_run_integrity


def _complete_tiny_release(root: Path) -> Path:
    height = width = 28
    y, x = np.mgrid[:height, :width]
    patch_counts = {1: 8, 2: 8, 3: 4}
    for fold_id, patch_count in patch_counts.items():
        fold = root / f"Fold {fold_id}" / "release_arrays"
        fold.mkdir(parents=True)
        images = np.zeros((patch_count, height, width, 3), dtype=np.uint8)
        masks = np.zeros((patch_count, height, width, 6), dtype=np.int32)
        tissues: list[str] = []
        for patch_index in range(patch_count):
            images[patch_index, ..., 0] = (x * 3 + fold_id * 17 + patch_index) % 256
            images[patch_index, ..., 1] = (y * 5 + fold_id * 19 + patch_index * 2) % 256
            images[patch_index, ..., 2] = ((x + y) * 2 + fold_id * 23 + patch_index * 3) % 256
            boxes = (
                (2, 2, 6, 6),
                (9, 2, 13, 7),
                (17, 2, 22, 6),
                (5, 15, 10, 21),
                (16, 15, 22, 21),
            )
            for class_index, (x0, y0, x1, y1) in enumerate(boxes):
                masks[patch_index, y0:y1, x0:x1, class_index] = (
                    fold_id * 10000 + patch_index * 10 + class_index + 1
                )
            if fold_id == 3 and patch_index == 0:
                masks[patch_index, 3, 3, 1] = fold_id * 10000 + patch_index * 10 + 2
            tissues.append(("Breast", "Colon", "Lung")[patch_index % 3])
        masks[..., 5] = (~np.any(masks[..., :5] > 0, axis=-1)).astype(np.int32)
        np.save(fold / "pixels.npy", images)
        np.save(fold / "labels.npy", masks)
        np.save(fold / "organs.npy", np.asarray(tissues, dtype="<U16"))
    return root


def _fake_representation_builder(
    validation: Any,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    sample_ids: tuple[str, ...] | None,
    **_kwargs: Any,
) -> Any:
    assert sample_ids is not None
    assert {fold.fold_id for fold in validation.folds} == {1, 2}
    table = pq.read_table(manifest_path)
    selection = pilot_module.select_manifest_rows(
        table,
        sample_ids=sample_ids,
        scope="analysis",
    )
    assert selection.provenance["manifest_view"]["scope"] == ("development_official_folds_only")
    assert set(int(value) for value in table["official_fold"].to_pylist()) == {1, 2}
    frame = table.to_pandas().set_index("sample_id")
    extracted_folds = set(int(value) for value in frame.loc[list(sample_ids), "official_fold"])
    assert extracted_folds <= {1, 2}, "the pilot must not extract final-reference representations"
    labels = np.asarray(
        [int(frame.loc[sample_id, "pre_corruption_label"]) for sample_id in sample_ids]
    )
    embeddings = np.zeros((len(sample_ids), 512), dtype=np.float32)
    embeddings[np.arange(len(sample_ids)), labels] = 3.0
    embeddings[:, 5] = np.linspace(-1.0, 1.0, len(sample_ids), dtype=np.float32)
    result = EmbeddingResult(
        embeddings=embeddings,
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
        metadata={
            "encoder_name": "torchvision.resnet18",
            "encoder_frozen": True,
            "input_variant": "target_highlighted_rgb",
            "weight_identifier": "ResNet18_Weights.IMAGENET1K_V1",
            "weight_sha256": "a" * 64,
            "test_substitution": "deterministic fake embedding matrix; not a study outcome",
            "analysis_eligibility": selection.provenance,
        },
    )
    result.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "fake_representation.npz.metadata.json").write_text(
        json.dumps({"analysis_eligibility": selection.provenance}),
        encoding="utf-8",
    )
    return SimpleNamespace(embeddings=result)


def test_pilot_rejects_reference_final_group_overlap_explicitly() -> None:
    groups = np.asarray([f"group-{index}" for index in range(4)], dtype=np.str_)
    rng = np.random.default_rng(211)
    selected_groups = groups[rng.permutation(len(groups))]
    reference_group = str(rng.permutation(selected_groups)[0])
    rows = [
        {
            "official_fold": 1 + (group_index % 2),
            "group_id": str(group_id),
            "pre_corruption_label": class_id,
        }
        for group_index, group_id in enumerate(groups)
        for class_id in range(5)
    ]
    final_rows = [
        {
            "official_fold": 3,
            "group_id": reference_group,
        }
        for _ in range(5)
    ]

    with pytest.raises(RuntimeError, match="reference_validation/final_reference=1"):
        pilot_module._select_groups(
            pd.DataFrame(rows),
            pd.DataFrame(final_rows),
            development_folds=(1, 2),
            final_fold=3,
            development_group_limit=4,
            reference_fraction=0.25,
            selection_seed=211,
            oof_splits=2,
        )


def test_pilot_seed_ledger_must_match_nested_execution_seeds() -> None:
    config = copy.deepcopy(load_config(Path("configs/pilot.yaml")))
    assert config["configuration_role"] == "fixed_pilot_protocol"
    assert "status" not in config
    assert config["seed"] == {"split": 223, "model": 227, "corruption": 404}
    assert config["evaluation"]["random_review_repeats"] == 100
    assert config["evaluation"]["downstream_random_repeats"] == 20
    config["seed"]["split"] = 999

    with pytest.raises(ValueError, match="exactly match the nested execution seeds"):
        pilot_module._validate_seed_provenance(
            config,
            model_config=config["model"],
            corruption_config=config["corruption"],
        )


def test_pilot_requires_an_explicit_raw_dataset_root(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="expected_data_root"):
        run_pannuke_pilot(  # type: ignore[call-arg]
            tmp_path / "gate.json",
            tmp_path / "manifest.parquet",
            development_manifest_source=tmp_path / "development.parquet",
            project_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("section", "field", "bad_value"),
    (
        ("model", "split_seed", 223.9),
        ("model", "seed", True),
        ("corruption", "seeds", [404.9]),
        ("seed", "split", 223.0),
        ("seed", "model", False),
        ("model", "oof_splits", 5.0),
        ("evaluation", "random_review_repeats", True),
        ("evaluation", "downstream_random_repeats", 20.0),
    ),
)
def test_pilot_integer_controls_reject_float_and_bool_before_casting(
    section: str, field: str, bad_value: object
) -> None:
    config = copy.deepcopy(load_config(Path("configs/pilot.yaml")))
    config[section][field] = bad_value

    if field in {"oof_splits", "random_review_repeats", "downstream_random_repeats"}:
        with pytest.raises(ValueError, match="must be an integer"):
            pilot_module._require_exact_int(bad_value, f"{section}.{field}")
    else:
        with pytest.raises(ValueError, match="must be an integer"):
            pilot_module._validate_seed_provenance(
                config,
                model_config=config["model"],
                corruption_config=config["corruption"],
            )


def test_final_reference_binding_is_independent_of_sample_ids_and_class_labels() -> None:
    first = pd.DataFrame(
        {
            "group_id": ["pannuke-f3-p000001", "pannuke-f3-p000001"],
            "sample_id": ["pannuke-f3-p000001-c0-i1", "pannuke-f3-p000001-c1-i2"],
            "pre_corruption_label": [0, 1],
        }
    )
    changed_outcomes = first.copy()
    changed_outcomes["sample_id"] = [
        "pannuke-f3-p000001-c4-i100",
        "pannuke-f3-p000001-c3-i200",
    ]
    changed_outcomes["pre_corruption_label"] = [4, 3]

    assert pilot_module._final_reference_metadata_binding(
        first, final_fold=3
    ) == pilot_module._final_reference_metadata_binding(changed_outcomes, final_fold=3)


def _development_manifest_publication_kwargs(
    tmp_path: Path,
) -> tuple[dict[str, Any], Path, Path]:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw_file = raw_root / "source.bin"
    raw_file.write_bytes(b"immutable raw bytes\n")
    source_evidence = tmp_path / "validation.json"
    source_evidence.write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "derived" / "development.parquet"
    metadata_path = destination.with_suffix(f"{destination.suffix}.metadata.json")
    raw_inventory = [
        {
            "relative_path": raw_file.name,
            "size_bytes": raw_file.stat().st_size,
            "sha256": sha256_file(raw_file),
            "fold_id": None,
            "file_kind": "bin",
        }
    ]
    kwargs = {
        "destination": destination,
        "metadata_path": metadata_path,
        "parquet_bytes": b"complete staged parquet bytes",
        "sidecar": {"schema_version": 1, "status": "complete"},
        "raw_root": raw_root,
        "raw_inventory": raw_inventory,
        "source_hashes": {source_evidence: sha256_file(source_evidence)},
    }
    return kwargs, raw_file, source_evidence


def test_development_manifest_bundle_publication_is_atomic_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs, _, _ = _development_manifest_publication_kwargs(tmp_path)
    destination = kwargs["destination"]
    metadata_path = kwargs["metadata_path"]
    assert isinstance(destination, Path)
    assert isinstance(metadata_path, Path)

    pilot_module._publish_development_manifest_bundle(**kwargs)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (destination, metadata_path)
    }
    pilot_module._publish_development_manifest_bundle(**kwargs)
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (destination, metadata_path)
    } == before

    destination.unlink()
    metadata_path.unlink()
    real_publish = pilot_module._link_locked_staging_file

    def fail_certificate_publish(staged: Any, parent: Any, target_name: str) -> Any:
        if target_name == metadata_path.name:
            raise OSError("injected certificate publication failure")
        return real_publish(staged, parent, target_name)

    monkeypatch.setattr(
        pilot_module,
        "_link_locked_staging_file",
        fail_certificate_publish,
    )
    with pytest.raises(OSError, match="injected certificate publication failure"):
        pilot_module._publish_development_manifest_bundle(**kwargs)
    assert not destination.exists()
    assert not metadata_path.exists()


def test_development_manifest_bundle_rechecks_raw_containment_after_parent_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _, _ = _development_manifest_publication_kwargs(tmp_path)
    destination = kwargs["destination"]
    metadata_path = kwargs["metadata_path"]
    raw_root = kwargs["raw_root"]
    assert isinstance(destination, Path)
    assert isinstance(metadata_path, Path)
    assert isinstance(raw_root, Path)
    relocated_parent = tmp_path / "relocated-derived"
    original_parent = destination.parent
    real_publish = pilot_module._link_locked_staging_file
    swapped = False
    swap_completed = False
    raw_target_observations: list[tuple[bool, bool]] = []

    def observe_raw_targets() -> None:
        raw_target_observations.append(
            (
                (raw_root / destination.name).exists(),
                (raw_root / metadata_path.name).exists(),
            )
        )

    def swap_parent_then_publish(staged: Any, parent: Any, target_name: str) -> Any:
        nonlocal swap_completed, swapped
        observe_raw_targets()
        try:
            if not swapped:
                swapped = True
                original_parent.rename(relocated_parent)
                observe_raw_targets()
                try:
                    original_parent.symlink_to(raw_root, target_is_directory=True)
                except OSError as error:
                    relocated_parent.rename(original_parent)
                    pytest.skip(f"directory symlinks are unavailable: {error}")
                swap_completed = True
                observe_raw_targets()
            result = real_publish(staged, parent, target_name)
            observe_raw_targets()
            return result
        finally:
            observe_raw_targets()

    monkeypatch.setattr(
        pilot_module,
        "_link_locked_staging_file",
        swap_parent_then_publish,
    )

    with pytest.raises((OSError, RuntimeError, ValueError)):
        pilot_module._publish_development_manifest_bundle(**kwargs)
    assert swapped is True
    assert swap_completed is True
    assert raw_target_observations
    assert not any(any(observation) for observation in raw_target_observations)
    assert not (raw_root / destination.name).exists()
    assert not (raw_root / metadata_path.name).exists()


def test_development_manifest_bundle_rejects_concurrent_raw_inventory_addition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _, _ = _development_manifest_publication_kwargs(tmp_path)
    destination = kwargs["destination"]
    metadata_path = kwargs["metadata_path"]
    raw_root = kwargs["raw_root"]
    assert isinstance(destination, Path)
    assert isinstance(metadata_path, Path)
    assert isinstance(raw_root, Path)
    concurrent_addition = raw_root / "concurrent-added.raw"
    real_publish = pilot_module._link_locked_staging_file
    added = False

    def add_raw_file_then_publish(staged: Any, parent: Any, target_name: str) -> Any:
        nonlocal added
        if not added:
            added = True
            concurrent_addition.write_bytes(b"foreign concurrent raw file")
        return real_publish(staged, parent, target_name)

    monkeypatch.setattr(
        pilot_module,
        "_link_locked_staging_file",
        add_raw_file_then_publish,
    )

    with pytest.raises(RuntimeError, match="raw inventory path set changed"):
        pilot_module._publish_development_manifest_bundle(**kwargs)
    assert concurrent_addition.is_file()
    assert not destination.exists()
    assert not metadata_path.exists()


def test_development_manifest_bundle_idempotent_readback_rehashes_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _, source_evidence = _development_manifest_publication_kwargs(tmp_path)
    destination = kwargs["destination"]
    assert isinstance(destination, Path)
    pilot_module._publish_development_manifest_bundle(**kwargs)
    real_read = pilot_module._read_locked_path
    mutated = False

    def mutate_source_after_destination_read(path: Path, parents: Any) -> Any:
        nonlocal mutated
        observed = real_read(path, parents)
        if path == destination and not mutated:
            mutated = True
            source_evidence.write_text('{"changed":true}\n', encoding="utf-8")
        return observed

    monkeypatch.setattr(
        pilot_module,
        "_read_locked_path",
        mutate_source_after_destination_read,
    )

    with pytest.raises(RuntimeError, match="source evidence changed"):
        pilot_module._publish_development_manifest_bundle(**kwargs)


def test_development_manifest_bundle_idempotent_readback_rehashes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _, _ = _development_manifest_publication_kwargs(tmp_path)
    destination = kwargs["destination"]
    assert isinstance(destination, Path)
    pilot_module._publish_development_manifest_bundle(**kwargs)
    real_read = pilot_module._read_locked_path
    mutated = False

    def mutate_destination_after_read(path: Path, parents: Any) -> Any:
        nonlocal mutated
        observed = real_read(path, parents)
        if path == destination and not mutated:
            mutated = True
            destination.write_bytes(b"foreign-after-read")
        return observed

    monkeypatch.setattr(
        pilot_module,
        "_read_locked_path",
        mutate_destination_after_read,
    )

    with pytest.raises(RuntimeError, match="stable byte/hash readback"):
        pilot_module._publish_development_manifest_bundle(**kwargs)
    assert destination.read_bytes() == b"foreign-after-read"


def test_development_manifest_bundle_new_publish_rehashes_outputs_before_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _, _ = _development_manifest_publication_kwargs(tmp_path)
    destination = kwargs["destination"]
    metadata_path = kwargs["metadata_path"]
    assert isinstance(destination, Path)
    assert isinstance(metadata_path, Path)
    real_read = pilot_module._read_locked_path
    mutated = False

    def mutate_destination_after_read(path: Path, parents: Any) -> Any:
        nonlocal mutated
        observed = real_read(path, parents)
        if path == destination and not mutated:
            mutated = True
            destination.write_bytes(b"foreign-after-read")
        return observed

    monkeypatch.setattr(
        pilot_module,
        "_read_locked_path",
        mutate_destination_after_read,
    )

    with pytest.raises(RuntimeError, match="ownership-safe rollback was incomplete"):
        pilot_module._publish_development_manifest_bundle(**kwargs)
    assert destination.read_bytes() == b"foreign-after-read"
    assert not metadata_path.exists()


def test_pre_pilot_builder_rejects_manifest_inventory_not_bound_to_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _complete_tiny_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        expected_fold_ids=(1, 2, 3),
        max_overlay_patches=1,
    )
    manifest = build_nucleus_manifest(validation, tmp_path / "manifest", batch_rows=32)
    duplicate_path = tmp_path / "duplicate_audit.json"
    duplicate_path.write_text(
        json.dumps({"required_two_signal_near_duplicate_gate_complete": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pilot_module,
        "_validate_complete_real_dataset_evidence",
        lambda *_args: (),
    )
    table = pq.read_table(manifest.parquet_path)
    metadata = dict(table.schema.metadata or {})
    inventory = json.loads(metadata[b"raw_file_inventory"].decode("ascii"))
    inventory[0]["size_bytes"] += 1
    encoded_inventory = json.dumps(
        inventory,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    metadata[b"raw_file_inventory"] = encoded_inventory
    metadata[b"raw_file_inventory_sha256"] = (
        hashlib.sha256(encoded_inventory).hexdigest().encode("ascii")
    )
    mismatched_manifest = tmp_path / "mismatched_manifest.parquet"
    pq.write_table(table.replace_schema_metadata(metadata), mismatched_manifest)

    with pytest.raises(
        ValueError,
        match="raw-file inventory differs from validation evidence",
    ):
        build_pannuke_pilot_development_manifest_view(
            validation,
            mismatched_manifest,
            duplicate_path,
            tmp_path / "development.parquet",
        )


@pytest.mark.parametrize(
    "payload",
    (
        {"leak": "pannuke-f3-p000001-c2-i99"},
        {"final_reference": {"class_labels": [2]}},
        {"final_test_labels": [2]},
    ),
)
def test_pre_seal_privacy_reconciliation_rejects_sensitive_json(
    tmp_path: Path,
    payload: dict[str, Any],
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="privacy reconciliation"):
        pilot_module._reconcile_pilot_final_reference_privacy(run, final_fold=3)


def test_pre_seal_privacy_reconciliation_rejects_final_fold_binary_rows(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    representations = run / "representations"
    representations.mkdir(parents=True)
    (run / "safe.json").write_text('{"status":"safe"}\n', encoding="utf-8")
    np.savez_compressed(
        representations / "leaked.npz",
        official_fold=np.asarray([1, 3], dtype=np.int16),
    )
    with pytest.raises(RuntimeError, match="final-fold rows in a representation array"):
        pilot_module._reconcile_pilot_final_reference_privacy(run, final_fold=3)

    (representations / "leaked.npz").unlink()
    pq.write_table(
        pa.table(
            {
                "official_fold": pa.array([1, 3], type=pa.int16()),
                "sample_id": pa.array(["pannuke-f1-p000001-c0-i1", "class-free-final-row"]),
            }
        ),
        run / "leaked.parquet",
    )
    with pytest.raises(RuntimeError, match="final-fold rows in Parquet"):
        pilot_module._reconcile_pilot_final_reference_privacy(run, final_fold=3)


def test_pre_seal_privacy_reconciliation_rejects_numeric_sensitive_columns(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "safe.json").write_text('{"status":"safe"}\n', encoding="utf-8")
    np.savez_compressed(
        run / "sensitive.npz",
        final_reference_class_labels=np.asarray([4], dtype=np.int16),
    )
    with pytest.raises(RuntimeError, match="populated final-sensitive field"):
        pilot_module._reconcile_pilot_final_reference_privacy(run, final_fold=3)

    (run / "sensitive.npz").unlink()
    pq.write_table(
        pa.table(
            {
                "final_reference_centroid": pa.array([12.5], type=pa.float64()),
            }
        ),
        run / "sensitive.parquet",
    )
    with pytest.raises(RuntimeError, match="populated final-sensitive field"):
        pilot_module._reconcile_pilot_final_reference_privacy(run, final_fold=3)

    (run / "sensitive.parquet").unlink()
    (run / "sensitive.csv").write_text(
        "final_reference_observed_label\n4\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="forbidden final-sensitive CSV column"):
        pilot_module._reconcile_pilot_final_reference_privacy(run, final_fold=3)

    (run / "sensitive.csv").unlink()
    (run / "sensitive.yaml").write_text(
        "final_reference_tissue: Breast\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="populated final-sensitive field"):
        pilot_module._reconcile_pilot_final_reference_privacy(run, final_fold=3)


def test_pre_seal_privacy_reconciliation_rejects_ids_in_binary_names_and_metadata(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "safe.json").write_text('{"status":"safe"}\n', encoding="utf-8")
    final_sample_id = "pannuke-f3-p000001-c2-i99"

    np.savez_compressed(
        run / "named-leak.npz",
        **{final_sample_id: np.asarray([1], dtype=np.int16)},
    )
    with pytest.raises(RuntimeError, match="final-reference sample ID"):
        pilot_module._reconcile_pilot_final_reference_privacy(run, final_fold=3)

    (run / "named-leak.npz").unlink()
    pq.write_table(
        pa.table({final_sample_id: pa.array([1], type=pa.int16())}),
        run / "column-leak.parquet",
    )
    with pytest.raises(RuntimeError, match="final-reference sample ID"):
        pilot_module._reconcile_pilot_final_reference_privacy(run, final_fold=3)

    (run / "column-leak.parquet").unlink()
    table = pa.table({"safe": pa.array([1], type=pa.int16())})
    table = table.replace_schema_metadata(
        {b"provenance": json.dumps({"sample_id": final_sample_id}).encode("utf-8")}
    )
    pq.write_table(table, run / "metadata-leak.parquet")
    with pytest.raises(RuntimeError, match="final-reference sample ID"):
        pilot_module._reconcile_pilot_final_reference_privacy(run, final_fold=3)


def test_pre_seal_privacy_reconciliation_recurses_structured_npz_dtypes(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "safe.json").write_text('{"status":"safe"}\n', encoding="utf-8")
    final_sample_id = "pannuke-f3-p000001-c2-i99"
    nested_dtype = np.dtype([("note", "U80")])
    structured_dtype = np.dtype([("records", nested_dtype, (2,))])
    structured = np.zeros(1, dtype=structured_dtype)
    structured["records"]["note"][0, 1] = final_sample_id
    np.savez_compressed(run / "structured-leak.npz", evidence=structured)

    with pytest.raises(RuntimeError, match="final-reference sample ID"):
        pilot_module._reconcile_pilot_final_reference_privacy(run, final_fold=3)

    (run / "structured-leak.npz").unlink()
    titled_dtype = np.dtype([((final_sample_id, "note"), "U8")])
    titled = np.zeros(1, dtype=titled_dtype)
    titled["note"] = "safe"
    np.savez_compressed(run / "field-title-leak.npz", evidence=titled)
    with pytest.raises(RuntimeError, match="final-reference sample ID"):
        pilot_module._reconcile_pilot_final_reference_privacy(run, final_fold=3)

    (run / "field-title-leak.npz").unlink()
    np.savez_compressed(
        run / "object-leak.npz",
        evidence=np.asarray([{"note": "opaque"}], dtype=object),
    )
    with pytest.raises(RuntimeError, match="forbidden or unreadable dtype"):
        pilot_module._reconcile_pilot_final_reference_privacy(run, final_fold=3)


def test_pre_seal_privacy_reconciliation_scans_nested_arrow_field_metadata(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "safe.json").write_text('{"status":"safe"}\n', encoding="utf-8")
    final_sample_id = "pannuke-f3-p000001-c2-i99"
    nested_field = pa.field(
        "payload",
        pa.struct(
            [
                pa.field(
                    "note",
                    pa.string(),
                    metadata={b"provenance": final_sample_id.encode("utf-8")},
                )
            ]
        ),
    )
    table = pa.Table.from_arrays(
        [pa.array([{"note": "safe"}], type=nested_field.type)],
        schema=pa.schema([nested_field]),
    )
    pq.write_table(table, run / "field-metadata-leak.parquet")

    with pytest.raises(RuntimeError, match="final-reference sample ID"):
        pilot_module._reconcile_pilot_final_reference_privacy(run, final_fold=3)


def test_pre_seal_privacy_reconciliation_scans_arrow_type_metadata(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "safe.json").write_text('{"status":"safe"}\n', encoding="utf-8")
    final_sample_id = "pannuke-f3-p000001-c2-i99"

    class FinalIdExtensionType(pa.ExtensionType):
        def __init__(self) -> None:
            super().__init__(pa.string(), final_sample_id)

        def __arrow_ext_serialize__(self) -> bytes:
            return b"safe extension metadata"

        @classmethod
        def __arrow_ext_deserialize__(
            cls,
            storage_type: pa.DataType,
            serialized: bytes,
        ) -> FinalIdExtensionType:
            assert storage_type == pa.string()
            assert serialized == b"safe extension metadata"
            return cls()

    extension_type = FinalIdExtensionType()
    pa.register_extension_type(extension_type)
    try:
        extension_array = pa.ExtensionArray.from_storage(
            extension_type,
            pa.array(["safe"], type=pa.string()),
        )
        pq.write_table(
            pa.Table.from_arrays([extension_array], names=["safe"]),
            run / "extension-type-leak.parquet",
        )
        with pytest.raises(RuntimeError, match="final-reference sample ID"):
            pilot_module._reconcile_pilot_final_reference_privacy(run, final_fold=3)
    finally:
        pa.unregister_extension_type(final_sample_id)

    (run / "extension-type-leak.parquet").unlink()
    timestamp_type = pa.timestamp("ns", tz=final_sample_id)
    pq.write_table(
        pa.table({"safe": pa.array([None], type=timestamp_type)}),
        run / "timestamp-type-leak.parquet",
    )
    with pytest.raises(RuntimeError, match="final-reference sample ID"):
        pilot_module._reconcile_pilot_final_reference_privacy(run, final_fold=3)


def test_pre_seal_privacy_reconciliation_scans_relative_artifact_paths(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    leaked_directory = run / "nested"
    leaked_directory.mkdir(parents=True)
    leaked_path = leaked_directory / "pannuke-f3-p000001-c2-i99.json"
    leaked_path.write_text('{"status":"safe"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="final-reference sample ID"):
        pilot_module._reconcile_pilot_final_reference_privacy(run, final_fold=3)


def test_pre_seal_privacy_reconciliation_rejects_non_integer_fold_values(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "safe.json").write_text('{"status":"safe"}\n', encoding="utf-8")

    np.savez_compressed(
        run / "float-fold.npz",
        official_fold=np.asarray([1.0, 2.0], dtype=np.float64),
    )
    with pytest.raises(RuntimeError, match="exact integers"):
        pilot_module._reconcile_pilot_final_reference_privacy(run, final_fold=3)

    (run / "float-fold.npz").unlink()
    pq.write_table(
        pa.table(
            {
                "official_fold": pa.array([1.0, 2.0], type=pa.float64()),
                "sample_id": pa.array(["pannuke-f1-p000001-c0-i1", "pannuke-f2-p000001-c0-i1"]),
            }
        ),
        run / "float-fold.parquet",
    )
    with pytest.raises(RuntimeError, match="exact integers"):
        pilot_module._reconcile_pilot_final_reference_privacy(run, final_fold=3)


def test_pannuke_pilot_preserves_outer_fold_and_seals_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _complete_tiny_release(tmp_path / "pannuke")
    validation = validate_pannuke(
        root,
        tmp_path / "validation",
        expected_fold_ids=(1, 2, 3),
        max_overlay_patches=1,
    )
    manifest = build_nucleus_manifest(validation, tmp_path / "manifest", batch_rows=32)
    manifest_frame = pq.read_table(manifest.parquet_path).to_pandas()
    final_sample_ids = tuple(
        manifest_frame.loc[manifest_frame["official_fold"] == 3, "sample_id"].astype(str)
    )
    final_excluded_sample_ids = tuple(
        manifest_frame.loc[
            (manifest_frame["official_fold"] == 3)
            & ~manifest_frame["primary_eligible"].astype(bool),
            "sample_id",
        ].astype(str)
    )
    assert len(final_sample_ids) == 20
    assert len(final_excluded_sample_ids) == 2
    duplicate_path = tmp_path / "duplicate_audit.json"
    duplicate_path.write_text(
        json.dumps({"required_two_signal_near_duplicate_gate_complete": True}),
        encoding="utf-8",
    )
    evidence_gate_calls: list[tuple[Path, Path, Path]] = []

    def fake_complete_evidence_gate(
        dataset_path: str | Path,
        validation_path: str | Path,
        duplicate_audit_path: str | Path,
    ) -> tuple[str, ...]:
        evidence_gate_calls.append(
            (
                Path(dataset_path).resolve(),
                Path(validation_path).resolve(),
                Path(duplicate_audit_path).resolve(),
            )
        )
        return ()

    monkeypatch.setattr(
        pilot_module,
        "_validate_complete_real_dataset_evidence",
        fake_complete_evidence_gate,
    )
    development_view = build_pannuke_pilot_development_manifest_view(
        validation,
        manifest,
        duplicate_path,
        tmp_path / "manifest" / "pannuke_pilot_development_manifest.parquet",
    )
    assert evidence_gate_calls == [
        (root.resolve(), validation.json_path.resolve(), duplicate_path.resolve())
    ]
    development_frame = pq.read_table(development_view.parquet_path).to_pandas()
    assert set(development_frame["official_fold"].astype(int)) == {1, 2}
    assert not set(development_frame["sample_id"].astype(str)).intersection(final_sample_ids)
    gate_text = development_view.metadata_path.read_text(encoding="utf-8")
    assert all(sample_id not in gate_text for sample_id in final_sample_ids)
    config = copy.deepcopy(load_config(Path("configs/pilot.yaml")))
    config["data"]["development_group_limit"] = 12
    config["model"]["oof_splits"] = 3
    config["model"]["max_iter"] = 80
    config["audit"]["neighbour_k"] = 3
    config["evaluation"]["downstream_random_repeats"] = 2
    duplicate_status = f"complete_sha256:{sha256_file(duplicate_path)}"

    with pytest.raises(ValueError, match="explicitly supplied dataset"):
        run_pannuke_pilot(
            development_view.metadata_path,
            manifest,
            development_manifest_source=development_view.parquet_path,
            project_root=tmp_path,
            expected_data_root=tmp_path / "different-pannuke-root",
            config=config,
            device="cpu",
            duplicate_audit_status=duplicate_status,
        )

    def rehashed_gate(name: str, mutation: Any, *, view_changed: bool = False) -> Path:
        payload = json.loads(gate_text)
        mutation(payload)
        if view_changed:
            validation_view = payload["development_validation_view"]
            validation_view.pop("semantic_sha256")
            validation_view["semantic_sha256"] = pilot_module.canonical_sha256(validation_view)
        payload.pop("semantic_sha256")
        payload["semantic_sha256"] = pilot_module.canonical_sha256(payload)
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    bad_boundary = rehashed_gate(
        "bad-boundary.json",
        lambda payload: payload.__setitem__(
            "materialization_boundary",
            {"final_class_labels": [4]},
        ),
    )
    with pytest.raises(ValueError, match="boundary policy"):
        run_pannuke_pilot(
            bad_boundary,
            manifest,
            development_manifest_source=development_view.parquet_path,
            project_root=tmp_path,
            expected_data_root=root,
            config=config,
            device="cpu",
            duplicate_audit_status=duplicate_status,
        )

    bad_mapping = rehashed_gate(
        "bad-mapping.json",
        lambda payload: payload["development_validation_view"]["class_mapping"].__setitem__(
            "source_note",
            "untrusted payload",
        ),
        view_changed=True,
    )
    with pytest.raises(ValueError, match="class mapping differs from pinned evidence"):
        run_pannuke_pilot(
            bad_mapping,
            manifest,
            development_manifest_source=development_view.parquet_path,
            project_root=tmp_path,
            expected_data_root=root,
            config=config,
            device="cpu",
            duplicate_audit_status=duplicate_status,
        )

    full_validation_payload = json.loads(validation.json_path.read_text(encoding="utf-8"))
    fold_three_descriptor = next(
        item for item in full_validation_payload["folds"] if item["fold_id"] == 3
    )

    def swap_development_paths(payload: dict[str, Any]) -> None:
        development_descriptor = payload["development_validation_view"]["folds"][0]
        for field in ("image_path", "mask_path", "tissue_path"):
            development_descriptor[field] = fold_three_descriptor[field]

    swapped_paths = rehashed_gate(
        "swapped-fold-paths.json",
        swap_development_paths,
        view_changed=True,
    )
    import histo_audit.pannuke.validation as validation_module

    semantic_open_attempts: list[Path] = []

    def forbid_any_semantic_open(path: str | Path) -> Any:
        semantic_open_attempts.append(Path(path).resolve())
        raise AssertionError("certificate descriptor attack reached semantic array loading")

    with monkeypatch.context() as attack_guard:
        attack_guard.setattr(validation_module, "open_npy_mmap", forbid_any_semantic_open)
        with pytest.raises(ValueError, match="raw-inventory fold/role"):
            run_pannuke_pilot(
                swapped_paths,
                manifest,
                development_manifest_source=development_view.parquet_path,
                project_root=tmp_path,
                expected_data_root=root,
                config=config,
                device="cpu",
                duplicate_audit_status=duplicate_status,
            )
    assert semantic_open_attempts == []

    monkeypatch.setattr(
        pilot_module, "build_pannuke_representation_cache", _fake_representation_builder
    )
    canonical_path = manifest.parquet_path.resolve()
    real_read_table = pilot_module.pq.read_table
    canonical_read_columns: list[tuple[str, ...] | None] = []

    def guarded_read_table(source: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(source).resolve() == canonical_path:
            columns = kwargs.get("columns")
            canonical_read_columns.append(None if columns is None else tuple(columns))
            assert tuple(columns or ()) == pilot_module.CLASS_FREE_ELIGIBILITY_COLUMNS
        return real_read_table(source, *args, **kwargs)

    monkeypatch.setattr(pilot_module.pq, "read_table", guarded_read_table)
    real_open_npy_mmap = validation_module.open_npy_mmap
    semantically_opened_paths: list[Path] = []

    def guarded_open_npy_mmap(path: str | Path) -> Any:
        resolved = Path(path).resolve()
        semantically_opened_paths.append(resolved)
        assert "Fold 3" not in resolved.parts
        return real_open_npy_mmap(resolved)

    monkeypatch.setattr(validation_module, "open_npy_mmap", guarded_open_npy_mmap)

    result = run_pannuke_pilot(
        development_view.metadata_path,
        manifest,
        development_manifest_source=development_view.parquet_path,
        project_root=tmp_path,
        expected_data_root=root,
        config=config,
        device="cpu",
        duplicate_audit_status=duplicate_status,
    )
    assert canonical_read_columns == [pilot_module.CLASS_FREE_ELIGIBILITY_COLUMNS]
    assert semantically_opened_paths

    assert result.audit_sample_count == 50
    assert result.reference_validation_sample_count == 10
    assert result.final_reference_sample_count == 18
    assert result.exact_corruption_count == 5
    assert result.report_path.is_file() and result.metrics_path.is_file()
    integrity = verify_run_integrity(result.run_directory)
    assert integrity.valid, integrity.errors
    selected_text = result.selected_ids_path.read_text(encoding="utf-8")
    selected = json.loads(selected_text)
    assert selected["final_fold_complete"] is True
    assert selected["final_test_fold"] == 3
    assert selected["final_reference_outcomes_used"] is False
    assert selected["final_reference_representations_extracted"] is False
    assert selected["final_reference_sample_ids_read"] is False
    assert selected["final_reference_class_labels_read"] is False
    assert "final_reference_sample_ids" not in selected
    assert "manifest_excluded_sample_ids" not in selected["analysis_eligibility"]
    assert selected["analysis_eligibility"]["contains_sample_ids"] is False
    assert selected["analysis_eligibility"]["contains_class_labels"] is False
    assert selected["final_reference_sample_count"] == result.final_reference_sample_count
    assert selected["final_reference_metadata_binding"] == {
        "bound_fields": ["official_fold", "group_id", "group_sample_count"],
        "contains_class_labels": False,
        "contains_sample_ids": False,
        "group_count": 4,
        "sample_count": result.final_reference_sample_count,
        "schema_version": 1,
        "scope": "analysis_eligible_final_reference_metadata",
        "sha256": selected["final_reference_metadata_binding"]["sha256"],
    }
    assert len(selected["final_reference_metadata_binding"]["sha256"]) == 64
    assert all(
        "-c" not in group_id and "-i" not in group_id
        for group_id in selected["final_reference_groups"]
    )
    assert all(sample_id not in selected_text for sample_id in final_sample_ids)
    assert selected["pairwise_group_overlap_counts"] == {
        "audit/final_reference": 0,
        "audit/reference_validation": 0,
        "reference_validation/final_reference": 0,
    }
    report = result.report_path.read_text(encoding="utf-8")
    assert "not a diagnostic system" in report
    assert "potentially inconsistent annotation" in report
    assert "recommended for expert review" in report
    assert "Controlled restoration on reference validation" in report
    assert "no representations were extracted for it" in report
    assert "Ranking-only random-review baseline repeats: 100" in report
    assert "Downstream random-restoration model refits: 2" in report
    assert "Explicit pilot reduction" in report
    assert "2 refitted models versus 100 inexpensive ranking-only" in report
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert any(
        "potentially inconsistent annotation is recommended for expert review" in limitation
        for limitation in metrics["limitations"]
    )
    assert metrics["final_reference_access"] == {
        "class_labels_read": False,
        "official_fold": 3,
        "outcomes_used": False,
        "policy": "metadata_only_until_preregistration_freeze",
        "representations_extracted": False,
        "sample_ids_read": False,
    }
    partition = metrics["downstream_restoration"]["partition_evidence"]
    assert partition["evaluation_partition"] == "reference_validation"
    assert partition["evaluation_partition_sample_count"] == 10
    assert partition["final_reference_outcomes_used"] is False
    assert partition["final_reference_representations_extracted"] is False
    assert metrics["pilot_reductions"] == [
        {
            "component": "downstream_random_restoration",
            "confirmatory_evidence": False,
            "downstream_random_restoration_repeats": 2,
            "ranking_random_review_repeats": 100,
            "reason": (
                "each downstream random-restoration repeat refits the classifier; "
                "ranking-only random review does not"
            ),
            "status": "declared_pilot_reduction",
        }
    ]
    privacy_reconciliation = json.loads(
        (result.run_directory / "final_reference_privacy_reconciliation.json").read_text(
            encoding="utf-8"
        )
    )
    assert privacy_reconciliation["status"] == "passed"
    assert privacy_reconciliation["final_fold_identity_pattern_absent"] is True
    assert privacy_reconciliation["final_sensitive_fields_unpopulated"] is True
    assert privacy_reconciliation["final_fold_representation_rows_absent"] is True
    assert "development_manifest_view.parquet" in privacy_reconciliation["scanned_paths"]

    resolved = load_config(result.run_directory / "resolved_config.yaml")
    assert resolved["configuration_role"] == "fixed_pilot_protocol"
    assert "status" not in resolved
    assert resolved["seed"] == {"split": 223, "model": 227, "corruption": 404}
    oof_provenance = json.loads(
        (result.run_directory / "oof_provenance.json").read_text(encoding="utf-8")
    )
    assert oof_provenance["split_seed"] == 223
    assert oof_provenance["model_seed"] == 227
    assert oof_provenance["fold_assignment_label_source"] == "pre_corruption_label"
    assert oof_provenance["splitter_class_name"] == "sklearn.model_selection.StratifiedGroupKFold"
    assert oof_provenance["splitter_fallback_status"] == "not_used"
    assert oof_provenance["splitter_fallback_reason"] is None
    assert len(oof_provenance["fold_assignment_labels_sha256"]) == 64
    assert [fold["model_seed"] for fold in oof_provenance["folds"]] == [227, 228, 229]
    run_provenance = json.loads(
        (result.run_directory / "run_provenance.json").read_text(encoding="utf-8")
    )
    assert run_provenance["split_seed"] == 223
    assert run_provenance["model_seed"] == 227
    assert run_provenance["corruption_seed"] == 404
    assert run_provenance["seed_provenance"] == {
        "split": 223,
        "model": 227,
        "corruption": 404,
    }
    with (tmp_path / "artifacts" / "runs" / "registry.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        registry_rows = list(csv.DictReader(handle))
    assert len(registry_rows) == 1
    assert registry_rows[0]["split_seed"] == "223"
    assert registry_rows[0]["model_seed"] == "227"
    assert registry_rows[0]["corruption_seed"] == "404"

    public_text_suffixes = {".csv", ".json", ".jsonl", ".log", ".md", ".txt", ".yaml"}
    for path in result.run_directory.rglob("*"):
        if not path.is_file():
            continue
        encoded = path.read_bytes()
        assert all(sample_id.encode("ascii") not in encoded for sample_id in final_sample_ids), path
        if path.suffix.lower() in public_text_suffixes:
            contents = path.read_text(encoding="utf-8")
            assert all(sample_id not in contents for sample_id in final_sample_ids), path
        elif path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                for name in payload.files:
                    values = np.asarray(payload[name])
                    if values.dtype.kind in {"U", "S"}:
                        published = set(values.astype(str).ravel().tolist())
                        assert not published.intersection(final_sample_ids), (path, name)
        elif path.suffix.lower() == ".parquet":
            published = pq.read_table(path)
            assert set(int(value) for value in published["official_fold"].to_pylist()) == {
                1,
                2,
            }
            assert not set(published["sample_id"].to_pylist()).intersection(final_sample_ids)

    reconciliation = reconcile_pilot_audit_evidence(result.run_directory)
    assert reconciliation["status"] == "passed"
    assert reconciliation["sample_count"] == result.audit_sample_count
    assert reconciliation["cleanlab"]["suggested_labels_available"] is False
    assert reconciliation["neighbours"]["same_group_exclusion_verified"] is True
    cleanlab_metadata = json.loads(
        (result.run_directory / "cleanlab_evidence.json").read_text(encoding="utf-8")
    )
    assert cleanlab_metadata["available"] is True
    assert cleanlab_metadata["package_name"] == "cleanlab"
    assert cleanlab_metadata["package_version"]
    assert cleanlab_metadata["api_path"]
    assert cleanlab_metadata["error"] is None
    assert cleanlab_metadata["suggested_labels"]["available"] is False
    assert "no substitute was fabricated" in cleanlab_metadata["suggested_labels"]["reason"]
    with np.load(result.run_directory / "cleanlab_evidence.npz", allow_pickle=False) as payload:
        assert payload["quality_scores"].shape == (result.audit_sample_count,)
        assert payload["risk_scores"].shape == (result.audit_sample_count,)
        assert payload["issue_mask"].shape == (result.audit_sample_count,)
        assert "suggested_class" not in payload.files
        assert not bool(payload["suggested_labels_available"].item())
    with np.load(result.run_directory / "neighbour_evidence.npz", allow_pickle=False) as payload:
        counts = np.asarray(payload["neighbour_count"], dtype=np.int64)
        weights = np.asarray(payload["neighbour_weights"], dtype=np.float64)
        support = np.asarray(payload["class_support"], dtype=np.float64)
        assert payload["neighbour_ids"].shape == payload["neighbour_groups"].shape
        assert payload["neighbour_ids"].shape == payload["neighbour_distances"].shape
        assert payload["neighbour_ids"].shape == payload["neighbour_observed_labels"].shape
        assert support.shape == (result.audit_sample_count, 5)
        assert np.allclose(support.sum(axis=1), 1.0)
        assert bool(np.asarray(payload["same_group_exclusion_verified"]).all())
        for index, count in enumerate(counts):
            assert np.isclose(weights[index, :count].sum(), 1.0)
    artifact_manifest = json.loads(
        (result.run_directory / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    checksum_paths = {record["path"] for record in artifact_manifest["artifacts"]}
    assert {
        "cleanlab_evidence.npz",
        "cleanlab_evidence.csv",
        "cleanlab_evidence.json",
        "neighbour_evidence.npz",
        "neighbour_evidence.csv",
        "neighbour_evidence.json",
        "audit_evidence_reconciliation.json",
    } <= checksum_paths

    order_tamper = tmp_path / "pilot-order-tamper"
    shutil.copytree(result.run_directory, order_tamper)
    cleanlab_npz = order_tamper / "cleanlab_evidence.npz"
    with np.load(cleanlab_npz, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays["sample_ids"] = arrays["sample_ids"][::-1]
    np.savez_compressed(cleanlab_npz, **arrays)
    metadata_path = order_tamper / "cleanlab_evidence.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["npz"]["sha256"] = sha256_file(cleanlab_npz)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Cleanlab evidence is not aligned"):
        reconcile_pilot_audit_evidence(order_tamper, require_sealed_integrity=False)

    value_tamper = tmp_path / "pilot-value-tamper"
    shutil.copytree(result.run_directory, value_tamper)
    cleanlab_npz = value_tamper / "cleanlab_evidence.npz"
    with np.load(cleanlab_npz, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays["risk_scores"] = arrays["risk_scores"].copy()
    arrays["risk_scores"][0] = float(arrays["risk_scores"][0]) + 0.125
    np.savez_compressed(cleanlab_npz, **arrays)
    metadata_path = value_tamper / "cleanlab_evidence.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["npz"]["sha256"] = sha256_file(cleanlab_npz)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Cleanlab evidence is not aligned"):
        reconcile_pilot_audit_evidence(value_tamper, require_sealed_integrity=False)

    missing_evidence = tmp_path / "pilot-missing-evidence"
    shutil.copytree(result.run_directory, missing_evidence)
    (missing_evidence / "neighbour_evidence.csv").unlink()
    with pytest.raises(ValueError, match="required pilot audit evidence is missing"):
        reconcile_pilot_audit_evidence(missing_evidence, require_sealed_integrity=False)
