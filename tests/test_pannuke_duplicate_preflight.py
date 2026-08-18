from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from histo_audit.cli import app
from histo_audit.pannuke import audit_pannuke_duplicates, validate_pannuke
from histo_audit.pannuke import duplicates as duplicate_module
from histo_audit.pannuke.duplicates import (
    PatchReference,
    _run_embedding_signal,
    _streaming_embedding_input_sha256,
)
from histo_audit.representations.imagenet import EmbeddingResult


def _release_arrays(
    fold_id: int, duplicate_first: np.ndarray[Any, Any] | None = None
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    height, width = 13, 11
    y_grid, x_grid = np.mgrid[:height, :width]
    images = np.empty((2, height, width, 3), dtype=np.uint8)
    for patch_index in range(2):
        images[patch_index, ..., 0] = (x_grid * 13 + fold_id * 17 + patch_index) % 256
        images[patch_index, ..., 1] = (y_grid * 11 + fold_id * 19 + patch_index * 3) % 256
        images[patch_index, ..., 2] = ((x_grid + y_grid) * 7 + fold_id * 23 + patch_index * 5) % 256
    if duplicate_first is not None:
        images[0] = duplicate_first
    masks = np.zeros((2, height, width, 6), dtype=np.int32)
    masks[0, 1:4, 1:4, 0] = 11
    masks[0, 7:11, 5:9, 1] = 27
    masks[1, 2:6, 6:10, 2] = 103
    masks[1, 8:12, 0:3, 4] = 205
    occupied = np.any(masks[..., :5] > 0, axis=-1)
    masks[..., 5] = (~occupied).astype(np.int32)
    tissues = np.asarray(["Breast", "Colon"], dtype="<U16")
    return images, masks, tissues


def _write_tiny_release(root: Path) -> Path:
    duplicate: np.ndarray[Any, Any] | None = None
    for fold_id in (1, 2):
        directory = root / f"Fold {fold_id}" / "release_arrays"
        directory.mkdir(parents=True)
        images, masks, tissues = _release_arrays(
            fold_id, duplicate_first=duplicate if fold_id == 2 else None
        )
        if duplicate is None:
            duplicate = images[0].copy()
        np.save(directory / "pixels.npy", images)
        np.save(directory / "labels.npy", masks)
        np.save(directory / "organs.npy", tissues)
    return root


def _validated_tiny_release(tmp_path: Path) -> tuple[Path, Any]:
    raw = _write_tiny_release(tmp_path / "raw" / "pannuke")
    validation = validate_pannuke(
        raw,
        tmp_path / "validation",
        max_overlay_patches=2,
        expected_fold_ids=(1, 2),
    )
    return raw, validation


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int, bytes]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            path.read_bytes(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _report_paths(output: Path) -> tuple[Path, ...]:
    return (
        output / "pannuke_duplicate_audit.json",
        output / "cross_fold_duplicate_candidates.csv",
        output / "cross_fold_duplicates.md",
        output / "pannuke_patch_hash_provenance.csv",
        output / "cross_fold_duplicate_candidate_grid.png",
    )


def _array_sha256(value: np.ndarray[Any, Any]) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _references(tmp_path: Path) -> tuple[PatchReference, ...]:
    image_path = tmp_path / "images.npy"
    images = np.arange(4 * 8 * 8 * 3, dtype=np.uint8).reshape(4, 8, 8, 3)
    np.save(image_path, images)
    return tuple(
        PatchReference(
            sample_id=f"pannuke-fold-{1 + index // 2}-patch-{index:06d}",
            fold_id=1 + index // 2,
            patch_index=index,
            image_path=image_path,
            image_channel_axis=3,
        )
        for index in range(4)
    )


def _fake_extract(
    images: np.ndarray[Any, Any], sample_ids: tuple[str, ...], **_: Any
) -> EmbeddingResult:
    embeddings = np.zeros((len(sample_ids), 512), dtype=np.float32)
    for index in range(len(sample_ids)):
        embeddings[index, index] = 1.0
    return EmbeddingResult(
        embeddings=embeddings,
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
        metadata={
            "encoder_name": "torchvision.resnet18",
            "encoder_frozen": True,
            "classification_head": "removed (fc=Identity)",
            "weight_identifier": "ResNet18_Weights.IMAGENET1K_V1",
            "weight_sha256": "a" * 64,
            "preprocessing": {"api": "unit-test fixture"},
            "input_variant": "rgb",
            "input_sha256": _array_sha256(images),
            "output_dimension": 512,
        },
    )


def test_duplicate_embedding_extraction_reuses_final_and_resumable_atomic_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    references = _references(tmp_path)
    reference_by_id = {item.sample_id: item for item in references}
    input_sha256 = _streaming_embedding_input_sha256(tuple(reference_by_id), reference_by_id)
    output = tmp_path / "audit"
    output.mkdir()
    monkeypatch.setattr("histo_audit.pannuke.duplicates.extract_resnet18_embeddings", _fake_extract)

    arguments = {
        "embedding_cache_path": None,
        "max_embedding_patches": None,
        "min_cosine_similarity": 0.999,
        "memory_budget_bytes": 2_000,
        "device": "cpu",
        "batch_size": 2,
        "allow_weight_download": False,
        "requested": True,
        "patch_manifest_sha256": "b" * 64,
        "raw_inventory_sha256": "c" * 64,
        "canonical_rgb_input_sha256": input_sha256,
    }
    fresh = _run_embedding_signal(references, output, **arguments)
    assert fresh.status == "passed"
    assert fresh.sample_count == len(references)
    assert fresh.source == "fresh_official_frozen_resnet18_extraction"
    assert fresh.metadata["provenance_scope"] == "stage_eligible"
    assert fresh.metadata["coverage_mode"] == "full_release"

    def forbidden_extract(*_: Any, **__: Any) -> EmbeddingResult:
        raise AssertionError("a validated cache must prevent recomputation")

    monkeypatch.setattr(
        "histo_audit.pannuke.duplicates.extract_resnet18_embeddings", forbidden_extract
    )
    from_final = _run_embedding_signal(references, output, **arguments)
    assert from_final.status == "passed"
    assert from_final.source == fresh.source

    assert from_final.cache_path is not None
    final_cache = from_final.cache_path
    final_sidecar = final_cache.with_suffix(f"{final_cache.suffix}.metadata.json")
    final_cache.unlink()
    final_sidecar.unlink()
    from_chunks = _run_embedding_signal(references, output, **arguments)
    assert from_chunks.status == "passed"
    assert from_chunks.source == "resumed_atomic_chunk_caches"
    assert from_chunks.metadata["duplicate_audit_resumed_chunk_count"] == len(references)


def test_duplicate_cli_returns_failure_when_required_full_coverage_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import histo_audit.pannuke as pannuke

    source = tmp_path / "pannuke"
    source.mkdir()
    (source / "images.npy").touch()
    output = tmp_path / "outputs"
    artifacts = SimpleNamespace(
        exact_pair_count=0,
        perceptual_pair_count=0,
        embedding_pair_count=0,
        embedding_status="blocked",
        required_two_signal_gate_complete=False,
        sampled_patch_count=4,
        embedding_sampled_patch_count=0,
        json_path=output / "duplicates.json",
        csv_path=output / "duplicates.csv",
        markdown_path=output / "duplicates.md",
        visual_grid_path=output / "duplicates.png",
        hash_provenance_csv_path=output / "hashes.csv",
    )
    monkeypatch.setattr(
        pannuke,
        "locate_pannuke_root",
        lambda explicit_path=None, project_root=None: source,
    )
    monkeypatch.setattr(pannuke, "audit_pannuke_duplicates", lambda *args, **kwargs: artifacts)

    result = CliRunner().invoke(
        app,
        [
            "data",
            "audit-duplicates",
            "--project-root",
            str(tmp_path),
            "--root",
            str(source),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 1
    assert '"required_two_signal_near_duplicate_gate_complete": false' in result.output
    assert "M5 remains open" in result.output


def test_library_rejects_every_raw_destination_before_any_write(tmp_path: Path) -> None:
    raw = _write_tiny_release(tmp_path / "raw" / "pannuke")
    before = _tree_snapshot(raw)
    safe = tmp_path / "safe-output"
    cases: tuple[tuple[Path, dict[str, Path]], ...] = (
        (raw / "derived_duplicate_outputs", {}),
        (raw / ".." / raw.name / "traversal", {}),
        (safe, {"rankings_csv_path": raw / "rankings.csv"}),
        (safe, {"report_path": raw / "report.md"}),
        (safe, {"hash_provenance_csv_path": raw / "hashes.csv"}),
        (safe, {"visual_grid_path": raw / "grid.png"}),
        (safe, {"embedding_cache_path": raw / "embeddings.npz"}),
    )

    for output, custom in cases:
        with pytest.raises(ValueError, match="immutable raw PanNuke root"):
            audit_pannuke_duplicates(
                raw,
                output,
                run_embedding_signal=False,
                **custom,
            )
        assert _tree_snapshot(raw) == before
        assert not safe.exists()


def test_library_resolves_directory_symlink_before_raw_containment_check(
    tmp_path: Path,
) -> None:
    raw = _write_tiny_release(tmp_path / "raw" / "pannuke")
    alias = tmp_path / "raw-alias"
    try:
        os.symlink(raw, alias, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    before = _tree_snapshot(raw)

    with pytest.raises(ValueError, match="immutable raw PanNuke root"):
        audit_pannuke_duplicates(
            raw,
            alias / "derived",
            run_embedding_signal=False,
        )

    assert _tree_snapshot(raw) == before


def test_report_bundle_is_idempotent_and_refuses_differing_or_failed_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, validation = _validated_tiny_release(tmp_path)
    output = tmp_path / "duplicates"
    first = audit_pannuke_duplicates(
        validation,
        output,
        max_hamming_distance=4,
        run_embedding_signal=False,
    )
    paths = _report_paths(output)
    assert (
        first.json_path,
        first.csv_path,
        first.markdown_path,
        first.hash_provenance_csv_path,
        first.visual_grid_path,
    ) == paths
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}

    second = audit_pannuke_duplicates(
        validation,
        output,
        max_hamming_distance=4,
        run_embedding_signal=False,
    )
    assert second.json_path == first.json_path
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths} == before

    with pytest.raises(FileExistsError, match="differs"):
        audit_pannuke_duplicates(
            validation,
            output,
            max_hamming_distance=3,
            run_embedding_signal=False,
        )
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths} == before

    def grid_failure(*_: Any, **__: Any) -> Path:
        raise RuntimeError("injected grid failure")

    monkeypatch.setattr(duplicate_module, "_write_candidate_grid", grid_failure)
    with pytest.raises(RuntimeError, match="injected grid failure"):
        audit_pannuke_duplicates(
            validation,
            output,
            max_hamming_distance=2,
            run_embedding_signal=False,
        )
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths} == before


def test_report_bundle_partial_state_and_publish_failure_never_leave_mixed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, validation = _validated_tiny_release(tmp_path)
    partial_output = tmp_path / "partial"
    partial_output.mkdir()
    partial = partial_output / "cross_fold_duplicate_candidates.csv"
    partial.write_bytes(b"existing\n")
    partial_before = (partial.read_bytes(), partial.stat().st_mtime_ns)
    with pytest.raises(FileExistsError, match="partial duplicate-report bundle"):
        audit_pannuke_duplicates(
            validation,
            partial_output,
            run_embedding_signal=False,
        )
    assert (partial.read_bytes(), partial.stat().st_mtime_ns) == partial_before
    assert [path for path in _report_paths(partial_output) if path.exists()] == [partial]

    output = tmp_path / "publish-failure"
    original_replace = duplicate_module._replace_staged_report_file
    calls = 0

    def fail_third_replace(staged: Path, destination: Path) -> duplicate_module.PublishedPath:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected third replace failure")
        return original_replace(staged, destination)

    monkeypatch.setattr(
        duplicate_module,
        "_replace_staged_report_file",
        fail_third_replace,
    )
    with pytest.raises(OSError, match="injected third replace failure"):
        audit_pannuke_duplicates(
            validation,
            output,
            run_embedding_signal=False,
        )
    assert not any(path.exists() for path in _report_paths(output))
    assert not list(output.glob(".*.bundle.*"))


def test_destination_alias_and_suffix_fail_in_library_and_cli_before_write(
    tmp_path: Path,
) -> None:
    raw = _write_tiny_release(tmp_path / "raw" / "pannuke")
    output = tmp_path / "duplicates"
    json_destination = output / "pannuke_duplicate_audit.json"
    with pytest.raises(ValueError, match=r"must use the \.csv suffix"):
        audit_pannuke_duplicates(
            raw,
            output,
            rankings_csv_path=json_destination,
            run_embedding_signal=False,
        )
    assert not output.exists()

    result = CliRunner().invoke(
        app,
        [
            "data",
            "audit-duplicates",
            "--project-root",
            str(tmp_path),
            "--root",
            str(raw),
            "--output-dir",
            str(output),
            "--rankings-csv",
            str(json_destination),
            "--skip-embedding-signal",
        ],
    )
    assert result.exit_code == 1
    assert "must use the .csv suffix" in result.output
    assert not output.exists()


def test_distinct_csv_destination_spellings_that_resolve_to_one_file_fail_before_write(
    tmp_path: Path,
) -> None:
    raw = _write_tiny_release(tmp_path / "raw" / "pannuke")
    before = _tree_snapshot(raw)
    output = tmp_path / "duplicates"
    rankings = output / "shared.csv"
    hash_provenance = output / "nested" / ".." / "shared.csv"

    assert rankings.resolve() == hash_provenance.resolve()
    with pytest.raises(ValueError, match="destinations must be pairwise distinct"):
        audit_pannuke_duplicates(
            raw,
            output,
            rankings_csv_path=rankings,
            hash_provenance_csv_path=hash_provenance,
            run_embedding_signal=False,
        )

    assert not output.exists()
    assert _tree_snapshot(raw) == before


def test_final_inventory_rehash_detects_concurrent_raw_mutation_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, validation = _validated_tiny_release(tmp_path)
    output = tmp_path / "duplicates"
    image_path = raw / "Fold 1" / "release_arrays" / "pixels.npy"
    original_grid = duplicate_module._write_candidate_grid

    def mutate_after_grid(*args: Any, **kwargs: Any) -> Path:
        result = original_grid(*args, **kwargs)
        images = np.load(image_path, mmap_mode="r+")
        images[0, 0, 0, 0] = (int(images[0, 0, 0, 0]) + 1) % 256
        images.flush()
        return result

    monkeypatch.setattr(duplicate_module, "_write_candidate_grid", mutate_after_grid)
    with pytest.raises(ValueError, match="raw PanNuke inventory changed"):
        audit_pannuke_duplicates(
            validation,
            output,
            run_embedding_signal=False,
        )

    assert not any(path.exists() for path in _report_paths(output))
    assert not list(output.rglob("*.npz"))
    assert not list(output.glob(".*.bundle.*"))


@pytest.mark.parametrize("inventory_change", ("addition", "removal"))
def test_duplicate_audit_full_inventory_change_blocks_publication(
    tmp_path: Path,
    inventory_change: str,
) -> None:
    raw = _write_tiny_release(tmp_path / "raw" / "pannuke")
    sidecar = raw / "inventory-only.txt"
    if inventory_change == "removal":
        sidecar.write_text("bound before validation\n", encoding="utf-8")
    validation = validate_pannuke(
        raw,
        tmp_path / "validation",
        max_overlay_patches=2,
        expected_fold_ids=(1, 2),
    )
    if inventory_change == "addition":
        sidecar.write_text("added after validation\n", encoding="utf-8")
    else:
        sidecar.unlink()

    output = tmp_path / f"duplicates-{inventory_change}"
    with pytest.raises(ValueError, match="raw PanNuke inventory changed"):
        audit_pannuke_duplicates(validation, output, run_embedding_signal=False)

    assert not any(path.exists() for path in _report_paths(output))
    assert not list(output.rglob("*.npz"))
    assert not list(output.glob(".*.bundle.*"))


def test_duplicate_publish_time_raw_mutation_rolls_back_complete_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, validation = _validated_tiny_release(tmp_path)
    output = tmp_path / "duplicates-publish-race"
    image_path = raw / "Fold 1" / "release_arrays" / "pixels.npy"
    real_replace = duplicate_module._replace_staged_report_file
    mutation_injected = False

    def mutate_at_first_promotion(
        staged: Path, destination: Path
    ) -> duplicate_module.PublishedPath:
        nonlocal mutation_injected
        if not mutation_injected:
            mutation_injected = True
            images = np.load(image_path, mmap_mode="r+", allow_pickle=False)
            images[0, 0, 0, 0] = (int(images[0, 0, 0, 0]) + 1) % 256
            images.flush()
            del images
        return real_replace(staged, destination)

    monkeypatch.setattr(duplicate_module, "_replace_staged_report_file", mutate_at_first_promotion)
    with pytest.raises(ValueError, match="raw PanNuke inventory changed"):
        audit_pannuke_duplicates(validation, output, run_embedding_signal=False)

    assert mutation_injected
    assert not any(path.exists() for path in _report_paths(output))
    assert not list(output.rglob("*.npz"))
    assert not list(output.glob(".*.bundle.*"))


def test_duplicate_concurrent_final_is_never_overwritten_or_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, validation = _validated_tiny_release(tmp_path)
    output = tmp_path / "duplicates-concurrent-final"
    foreign = b"foreign concurrent rankings\n"
    real_promote = duplicate_module._replace_staged_report_file
    injected = False

    def insert_foreign_then_promote(
        staged: Path, destination: Path
    ) -> duplicate_module.PublishedPath:
        nonlocal injected
        if not injected:
            injected = True
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(foreign)
        return real_promote(staged, destination)

    monkeypatch.setattr(
        duplicate_module, "_replace_staged_report_file", insert_foreign_then_promote
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit_pannuke_duplicates(validation, output, run_embedding_signal=False)

    assert injected
    assert (output / "cross_fold_duplicate_candidates.csv").read_bytes() == foreign
    assert [path for path in _report_paths(output) if path.exists()] == [
        output / "cross_fold_duplicate_candidates.csv"
    ]


def test_duplicate_rollback_preserves_foreign_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, validation = _validated_tiny_release(tmp_path)
    output = tmp_path / "duplicates-foreign-replacement"
    foreign = b"foreign replacement\n"
    real_promote = duplicate_module._replace_staged_report_file
    calls = 0

    def replace_first_then_fail_second(
        staged: Path, destination: Path
    ) -> duplicate_module.PublishedPath:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second promotion failure")
        publication = real_promote(staged, destination)
        destination.unlink()
        destination.write_bytes(foreign)
        return publication

    monkeypatch.setattr(
        duplicate_module, "_replace_staged_report_file", replace_first_then_fail_second
    )
    with pytest.raises(RuntimeError, match="ownership-safe rollback was incomplete"):
        audit_pannuke_duplicates(validation, output, run_embedding_signal=False)

    assert (output / "cross_fold_duplicate_candidates.csv").read_bytes() == foreign
    assert [path for path in _report_paths(output) if path.exists()] == [
        output / "cross_fold_duplicate_candidates.csv"
    ]


def test_duplicate_whole_audit_lock_is_keyed_by_raw_release_across_outputs(
    tmp_path: Path,
) -> None:
    raw, validation = _validated_tiny_release(tmp_path)
    logical_lock = raw / ".histo-audit-duplicates.logical-lock"
    with (
        duplicate_module.ExclusivePublicationLock(logical_lock, role="PanNuke duplicate audit"),
        pytest.raises(FileExistsError, match="publication is active"),
    ):
        audit_pannuke_duplicates(
            validation,
            tmp_path / "different-output",
            embedding_cache_path=tmp_path / "shared" / "cache.npz",
            run_embedding_signal=False,
        )
    assert not (tmp_path / "different-output" / "pannuke_duplicate_audit.json").exists()
