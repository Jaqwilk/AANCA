from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from histo_audit.data.duplicates import (
    canonical_array_sha256,
    find_embedding_duplicate_candidates,
    find_exact_duplicate_pairs,
    find_perceptual_duplicate_candidates,
    perceptual_hash,
)
from histo_audit.pannuke import duplicates as pannuke_duplicates
from histo_audit.reporting.synthetic_duplicates import (
    SyntheticDuplicateAuditError,
    audit_synthetic_duplicate_patches,
    reconcile_synthetic_duplicate_audit,
)


def _write_synthetic_patch_evidence(
    path: Path,
    patches: list[tuple[str, np.ndarray, str, int, int]],
) -> None:
    sample_ids: list[str] = []
    patch_ids: list[str] = []
    partitions: list[str] = []
    folds: list[int] = []
    images: list[np.ndarray] = []
    for patch_id, image, partition, fold, nucleus_count in patches:
        for nucleus_index in range(nucleus_count):
            sample_ids.append(f"{patch_id}_nucleus_{nucleus_index}")
            patch_ids.append(patch_id)
            partitions.append(partition)
            folds.append(fold)
            images.append(image.copy())
    np.savez_compressed(
        path,
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
        patch_ids=np.asarray(patch_ids, dtype=np.str_),
        split_partition=np.asarray(partitions, dtype=np.str_),
        official_fold=np.asarray(folds, dtype=np.int64),
        images=np.stack(images).astype(np.uint8),
    )


def test_exact_duplicate_detection_and_cross_fold_reporting() -> None:
    first = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    second = first.copy()
    third = np.zeros_like(first)
    candidates = find_exact_duplicate_pairs(
        np.stack([first, second, third]),
        sample_ids=("a", "b", "c"),
        folds=(0, 1, 1),
        cross_fold_only=True,
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert (candidate.sample_id_a, candidate.sample_id_b) == ("a", "b")
    assert candidate.crosses_fold
    assert candidate.recommended_action == "review_only"
    assert candidate.exact_sha256 == canonical_array_sha256(first)


def test_perceptual_hash_is_deterministic_and_reports_without_deleting() -> None:
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    assert perceptual_hash(image) == perceptual_hash(image.copy())
    candidates = find_perceptual_duplicate_candidates(
        np.stack([image, image.copy()]),
        sample_ids=("left", "right"),
        folds=(0, 2),
        max_hamming_distance=0,
    )
    assert len(candidates) == 1
    assert candidates[0].recommended_action == "review_only"


def test_embedding_signal_is_independent_candidate_report() -> None:
    embeddings = np.array([[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]])
    candidates = find_embedding_duplicate_candidates(
        embeddings,
        sample_ids=("a", "b", "c"),
        folds=(0, 1, 1),
        min_cosine_similarity=0.999,
    )
    assert [(row.sample_id_a, row.sample_id_b) for row in candidates] == [("a", "b")]
    assert candidates[0].embedding_cosine_similarity is not None


def test_tracked_synthetic_duplicate_audit_deduplicates_nuclei_and_binds_candidate(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "synthetic_dataset_evidence.npz"
    rng = np.random.default_rng(91)
    duplicate = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    distinct = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    assert perceptual_hash(duplicate) != perceptual_hash(distinct)
    _write_synthetic_patch_evidence(
        dataset,
        [
            ("patch_a", duplicate, "audit_pool", 0, 2),
            ("patch_b", duplicate.copy(), "final_reference_test", 0, 3),
            ("patch_c", distinct, "audit_pool", 0, 1),
            ("patch_d", distinct.copy(), "audit_pool", 1, 1),
        ],
    )

    artifacts = audit_synthetic_duplicate_patches(
        dataset,
        tmp_path / "run",
        max_hamming_distance=0,
    )
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    assert payload["dataset_evidence"]["nucleus_sample_count"] == 7
    assert payload["dataset_evidence"]["unique_patch_count"] == 4
    assert payload["dataset_evidence"]["deduplicated_nucleus_row_count"] == 3
    assert payload["candidate_counts"] == {
        "cross_official_fold_union": 1,
        "cross_partition_union": 1,
        "exact": 2,
        "perceptual_including_exact": 2,
        "union": 2,
    }
    assert payload["real_data_duplicate_gate_eligible"] is False
    assert payload["required_two_signal_near_duplicate_gate_complete"] is False
    candidates = {
        (candidate["patch_id_a"], candidate["patch_id_b"]): candidate
        for candidate in payload["candidates"]
    }
    assert set(candidates) == {("patch_a", "patch_b"), ("patch_c", "patch_d")}
    assert candidates[("patch_a", "patch_b")]["cross_partition"] is True
    assert candidates[("patch_c", "patch_d")]["cross_official_fold"] is True
    assert all(candidate["exact_match"] is True for candidate in candidates.values())
    assert all(candidate["automatic_deletion"] is False for candidate in candidates.values())
    with artifacts.csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["recommended_action"] == "review_only"
    assert artifacts.figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    summary = reconcile_synthetic_duplicate_audit(
        dataset,
        artifacts.json_path,
        artifacts.csv_path,
        artifacts.figure_path,
    )
    assert summary["synthetic_duplicate_audit_status"] == "passed"
    assert summary["synthetic_duplicate_audit_candidate_count"] == 2

    payload["dataset_evidence"]["unique_patch_count"] = 5
    artifacts.json_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SyntheticDuplicateAuditError, match="disagree with recomputation"):
        reconcile_synthetic_duplicate_audit(
            dataset,
            artifacts.json_path,
            artifacts.csv_path,
            artifacts.figure_path,
        )


def test_tracked_synthetic_duplicate_audit_reports_honest_zero_candidate_panel(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "synthetic_dataset_evidence.npz"
    left_right = np.zeros((16, 16, 3), dtype=np.uint8)
    left_right[:, 8:] = 255
    top_bottom = np.zeros((16, 16, 3), dtype=np.uint8)
    top_bottom[8:, :] = 255
    assert perceptual_hash(left_right) != perceptual_hash(top_bottom)
    _write_synthetic_patch_evidence(
        dataset,
        [
            ("patch_left_right", left_right, "audit_pool", 0, 2),
            ("patch_top_bottom", top_bottom, "reference_validation", 1, 2),
        ],
    )

    artifacts = audit_synthetic_duplicate_patches(
        dataset,
        tmp_path / "run",
        max_hamming_distance=0,
    )
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    assert payload["candidate_counts"]["union"] == 0
    assert payload["pair_counts"]["evaluated_cross_boundary_pairs"] == 1
    with artifacts.csv_path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []
    assert artifacts.figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (
        reconcile_synthetic_duplicate_audit(
            dataset,
            artifacts.json_path,
            artifacts.csv_path,
            artifacts.figure_path,
        )["synthetic_duplicate_audit_candidate_count"]
        == 0
    )


def _report_bundle(root: Path) -> pannuke_duplicates._ReportBundlePaths:
    return pannuke_duplicates._ReportBundlePaths(
        json=root / "audit.json",
        rankings_csv=root / "rankings.csv",
        markdown=root / "report.md",
        hash_provenance_csv=root / "provenance.csv",
        visual_grid=root / "grid.png",
    )


def test_duplicate_bundle_lock_rejects_partially_overlapping_publishers(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.json"
    shared_path = tmp_path / "shared.csv"
    second_path = tmp_path / "second.md"
    first = pannuke_duplicates.ExclusiveBundlePublicationLock(
        (first_path, shared_path),
        role="duplicate test bundle",
    )
    second = pannuke_duplicates.ExclusiveBundlePublicationLock(
        (shared_path, second_path),
        role="duplicate test bundle",
    )

    with first:
        first.assert_owned()
        with pytest.raises(FileExistsError, match="active or requires stale-lock review"):
            second.__enter__()
        first.assert_owned()

    assert all(not os.path.lexists(path) for path in first.lock_paths)
    assert all(not os.path.lexists(path) for path in second.lock_paths)


def test_duplicate_report_readback_failure_rolls_back_every_owned_output(
    tmp_path: Path,
) -> None:
    bundle = _report_bundle(tmp_path / "bundle")
    staged = pannuke_duplicates._allocate_staged_report_paths(bundle)
    for index, (label, _) in enumerate(bundle.ordered_items()):
        staged[label].write_bytes(f"artifact-{index}\n".encode())
    verification_calls = 0

    def fail_after_publication() -> None:
        nonlocal verification_calls
        verification_calls += 1
        if verification_calls == 2:
            raise ValueError("injected final evidence change")

    try:
        with pytest.raises(ValueError, match="injected final evidence change"):
            pannuke_duplicates._publish_or_verify_report_bundle(
                bundle,
                staged,
                initial_state="absent",
                raw_inventory_verifier=fail_after_publication,
            )
        assert verification_calls == 2
        assert all(not os.path.lexists(path) for _, path in bundle.ordered_items())
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)


def test_duplicate_report_rejects_staged_mutation_during_final_evidence_rehash(
    tmp_path: Path,
) -> None:
    bundle = _report_bundle(tmp_path / "bundle-staged-race")
    staged = pannuke_duplicates._allocate_staged_report_paths(bundle)
    for index, (label, _) in enumerate(bundle.ordered_items()):
        staged[label].write_bytes(f"artifact-{index}\n".encode())
    frozen = {
        label: pannuke_duplicates._capture_cache_file(staged[label])
        for label, _ in bundle.ordered_items()
    }

    def mutate_staged_rankings() -> None:
        staged["rankings_csv"].write_bytes(b"foreign staged rankings\n")

    try:
        with pytest.raises(ValueError, match="changed after rendering: rankings_csv"):
            pannuke_duplicates._publish_or_verify_report_bundle(
                bundle,
                staged,
                initial_state="absent",
                raw_inventory_verifier=mutate_staged_rankings,
                staged_bindings=frozen,
            )
        assert all(not os.path.lexists(path) for _, path in bundle.ordered_items())
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)


def test_duplicate_final_readback_preserves_foreign_report_replacement(
    tmp_path: Path,
) -> None:
    bundle = _report_bundle(tmp_path / "bundle-foreign")
    staged = pannuke_duplicates._allocate_staged_report_paths(bundle)
    for index, (label, _) in enumerate(bundle.ordered_items()):
        staged[label].write_bytes(f"artifact-{index}\n".encode())
    foreign = b"foreign concurrent rankings\n"
    verification_calls = 0

    def replace_during_final_evidence_check() -> None:
        nonlocal verification_calls
        verification_calls += 1
        if verification_calls == 2:
            bundle.rankings_csv.unlink()
            bundle.rankings_csv.write_bytes(foreign)

    try:
        with pytest.raises(RuntimeError, match="ownership-safe rollback was incomplete"):
            pannuke_duplicates._publish_or_verify_report_bundle(
                bundle,
                staged,
                initial_state="absent",
                raw_inventory_verifier=replace_during_final_evidence_check,
            )
        assert bundle.rankings_csv.read_bytes() == foreign
        assert [path for _, path in bundle.ordered_items() if os.path.lexists(path)] == [
            bundle.rankings_csv
        ]
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)


def test_duplicate_embedding_cache_pair_is_tracked_and_rollback_owned(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "cache" / "embeddings.npz"
    tracker = pannuke_duplicates._EmbeddingPublicationTracker(publications=[])
    embeddings = np.zeros((2, 512), dtype=np.float32)
    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0
    sample_ids = np.asarray(["patch-a", "patch-b"], dtype=np.str_)

    cache, sidecar, _ = pannuke_duplicates._save_tracked_embedding_cache(
        destination,
        embeddings,
        sample_ids,
        {"input_variant": "rgb"},
        tracker,
    )

    assert cache.is_file()
    assert sidecar.is_file()
    assert len(tracker.publications) == 2
    pannuke_duplicates._rollback_tracked_embedding_publications(tracker)
    assert not os.path.lexists(cache)
    assert not os.path.lexists(sidecar)
    assert tracker.publications == []
    assert not list(destination.parent.glob(".*.cache-stage-*"))


def test_duplicate_embedding_cache_sidecar_collision_preserves_foreign_file(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "cache" / "embeddings.npz"
    destination.parent.mkdir(parents=True)
    sidecar = destination.with_suffix(f"{destination.suffix}.metadata.json")
    foreign = b"foreign-sidecar\n"
    sidecar.write_bytes(foreign)
    tracker = pannuke_duplicates._EmbeddingPublicationTracker(publications=[])

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        pannuke_duplicates._save_tracked_embedding_cache(
            destination,
            np.pad(
                np.asarray([[1.0]], dtype=np.float32),
                ((0, 0), (0, 511)),
            ),
            np.asarray(["patch-a"], dtype=np.str_),
            {"input_variant": "rgb"},
            tracker,
        )

    assert not os.path.lexists(destination)
    assert sidecar.read_bytes() == foreign
    assert tracker.publications == []


def test_duplicate_resume_directory_rejects_windows_junction_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resume = tmp_path / "resume"
    resume.mkdir()
    original_is_junction = getattr(Path, "is_junction", None)

    def fake_is_junction(path: Path) -> bool:
        if path == resume:
            return True
        return bool(original_is_junction is not None and original_is_junction(path))

    monkeypatch.setattr(Path, "is_junction", fake_is_junction, raising=False)
    tracker = pannuke_duplicates._EmbeddingPublicationTracker(publications=[])

    with pytest.raises(FileExistsError, match="not a real directory"):
        pannuke_duplicates._ensure_owned_resume_directory(resume, tracker)

    assert resume.is_dir()
    assert tracker.publications == []
