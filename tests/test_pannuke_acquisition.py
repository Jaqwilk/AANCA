"""Regression tests for PanNuke acquisition provenance and Git safety."""

from __future__ import annotations

import errno
import json
import os
import subprocess
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Barrier
from typing import Any
from zipfile import ZipFile

import pytest

from histo_audit.pannuke.acquisition import (
    DEFAULT_ARCHIVE_EXPECTATIONS,
    ArchiveExpectation,
    PanNukeAcquisitionError,
    build_pannuke_acquisition_manifest,
    git_ignore_evidence,
    validate_acquisition_manifest,
    validate_zip_member_path,
    write_acquisition_artifact_bundle,
    write_acquisition_manifest,
    write_json_compare_and_swap,
)
from histo_audit.pannuke.publication import (
    ExclusiveBundlePublicationLock,
    ExclusivePublicationLock,
    create_directory_no_overwrite,
    publish_file_no_overwrite,
    publish_success_marker_no_overwrite,
    rollback_owned_publications,
)
from histo_audit.utils.run_tracking import sha256_file

_LICENSE_MARKER = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International"


def _tiny_release(project: Path) -> tuple[Path, tuple[ArchiveExpectation, ...]]:
    raw = project / "data" / "raw" / "pannuke"
    raw.mkdir(parents=True)
    expectations: list[ArchiveExpectation] = []
    for fold in (1, 2, 3):
        payloads = {
            f"Fold {fold}/README.md": (
                f"fold {fold}\n{_LICENSE_MARKER}\ngamper2020pannuke\ngamper2019pannuke\n"
            ).encode(),
            f"Fold {fold}/masks/README.md": (
                f"masks fold {fold}\n{_LICENSE_MARKER}\ngamper2020pannuke\ngamper2019pannuke\n"
            ).encode(),
            f"Fold {fold}/masks/by-nc-sa.md": _LICENSE_MARKER.encode(),
            f"Fold {fold}/images/fold{fold}/images.npy": f"tiny-{fold}-images".encode(),
            f"Fold {fold}/images/fold{fold}/types.npy": f"tiny-{fold}-types".encode(),
            f"Fold {fold}/masks/fold{fold}/masks.npy": f"tiny-{fold}-masks".encode(),
        }
        for relative, content in payloads.items():
            path = raw / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        archive = raw / f"fold_{fold}.zip"
        with ZipFile(archive, "w") as payload:
            for relative, content in payloads.items():
                payload.writestr(relative, content)
        expectations.append(
            ArchiveExpectation(
                fold=fold,
                relative_path=archive.relative_to(project).as_posix(),
                size_bytes=archive.stat().st_size,
                sha256=sha256_file(archive),
            )
        )
    return raw, tuple(expectations)


def _tiny_manifest(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    raw, expectations = _tiny_release(tmp_path)
    manifest = build_pannuke_acquisition_manifest(
        tmp_path,
        raw,
        verification_timestamp_utc="2026-07-17T21:00:00Z",
        archive_expectations=expectations,
    )
    return raw, manifest


@pytest.mark.parametrize(
    "member",
    [
        "../escape.npy",
        "/absolute.npy",
        r"C:\absolute.npy",
        r"\\server\share\escape.npy",
        "safe/../../escape.npy",
        "safe/NUL.txt",
        "safe/trailing.",
        "safe/file:stream",
    ],
)
def test_zip_member_path_rejects_traversal_absolute_and_windows_aliases(member: str) -> None:
    with pytest.raises(PanNukeAcquisitionError):
        validate_zip_member_path(member)


def test_zip_member_path_normalises_safe_separator() -> None:
    assert validate_zip_member_path(r"Fold 1\images\fold1\images.npy") == (
        "Fold 1/images/fold1/images.npy"
    )


def test_builder_rejects_unsafe_archive_before_recording_pass(tmp_path: Path) -> None:
    raw, expectations = _tiny_release(tmp_path)
    archive = raw / "fold_1.zip"
    with ZipFile(archive, "w") as payload:
        payload.writestr("../escape.npy", b"unsafe")
    replacement = ArchiveExpectation(
        fold=1,
        relative_path=archive.relative_to(tmp_path).as_posix(),
        size_bytes=archive.stat().st_size,
        sha256=sha256_file(archive),
    )
    current = (replacement, expectations[1], expectations[2])

    with pytest.raises(PanNukeAcquisitionError, match=r"traverses|absolute"):
        build_pannuke_acquisition_manifest(
            tmp_path,
            raw,
            verification_timestamp_utc="2026-07-17T21:00:00Z",
            archive_expectations=current,
        )


def test_manifest_schema_binds_three_archives_and_nine_extracted_arrays(tmp_path: Path) -> None:
    _, manifest = _tiny_manifest(tmp_path)

    validate_acquisition_manifest(manifest)

    assert [value["fold"] for value in manifest["archives"]] == [1, 2, 3]
    assert len(manifest["extracted_npy_inventory"]) == 9
    assert len(manifest["extracted_document_inventory"]) == 9
    assert all(value["zip_crc_status"] == "passed" for value in manifest["archives"])
    assert all(value["path_safety_status"] == "passed" for value in manifest["archives"])
    assert all(value["rejected_unsafe_member_path_count"] == 0 for value in manifest["archives"])
    assert all(
        value["archive_member_crc32_match"]
        for value in [
            *manifest["extracted_npy_inventory"],
            *manifest["extracted_document_inventory"],
        ]
    )
    assert manifest["license"]["spdx_id"] == "CC-BY-NC-SA-4.0"
    assert manifest["license"]["project_use"] == "research_noncommercial"
    assert manifest["citation_requirement"]["required"] is True
    assert len(manifest["citation_requirement"]["local_readme_evidence"]) == 6
    assert manifest["immutable_raw_policy"]["git_tracking_for_raw_release_forbidden"] is True
    assert manifest["raw_release_read_only_verification"]["status"] == "passed"


def test_manifest_schema_rejects_rehashed_inventory_tamper(tmp_path: Path) -> None:
    _, manifest = _tiny_manifest(tmp_path)
    tampered = deepcopy(manifest)
    tampered["extracted_npy_inventory"][0]["size_bytes"] += 1

    with pytest.raises(PanNukeAcquisitionError, match="archive member"):
        validate_acquisition_manifest(tampered)


def test_builder_rejects_extracted_file_that_differs_from_zip_member(tmp_path: Path) -> None:
    raw, expectations = _tiny_release(tmp_path)
    (raw / "Fold 2" / "images" / "fold2" / "images.npy").write_bytes(b"changed-size")

    with pytest.raises(PanNukeAcquisitionError, match="differs from its ZIP member"):
        build_pannuke_acquisition_manifest(
            tmp_path,
            raw,
            verification_timestamp_utc="2026-07-17T21:00:00Z",
            archive_expectations=expectations,
        )


def test_builder_does_not_modify_any_raw_release_file(tmp_path: Path) -> None:
    raw, expectations = _tiny_release(tmp_path)
    before = {
        path.relative_to(raw).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in raw.rglob("*")
        if path.is_file()
    }

    build_pannuke_acquisition_manifest(
        tmp_path,
        raw,
        verification_timestamp_utc="2026-07-17T21:00:00Z",
        archive_expectations=expectations,
    )

    after = {
        path.relative_to(raw).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in raw.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_manifest_writer_is_idempotent_and_compare_and_swap_guarded(tmp_path: Path) -> None:
    _, manifest = _tiny_manifest(tmp_path)
    path = tmp_path / "data" / "manifests" / "pannuke_acquisition.json"
    write_acquisition_manifest(path, manifest)
    original_sha = sha256_file(path)

    assert write_acquisition_manifest(path, manifest) == path
    changed = deepcopy(manifest)
    changed["acquisition"]["verification_timestamp_utc"] = "2026-07-17T21:01:00Z"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_acquisition_manifest(path, changed)

    write_acquisition_manifest(path, changed, expected_previous_sha256=original_sha)
    assert json.loads(path.read_text(encoding="utf-8")) == changed


def test_active_destination_lock_blocks_json_compare_and_swap(tmp_path: Path) -> None:
    path = tmp_path / "provenance.json"

    with (
        ExclusivePublicationLock(path, role="test fixture"),
        pytest.raises(FileExistsError, match="publication is active"),
    ):
        write_json_compare_and_swap(path, {"version": 1})

    assert not path.exists()


def test_anchored_file_publication_fails_closed_across_volumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import histo_audit.pannuke.publication as publication_module

    staged = tmp_path / "staged.json"
    staged.write_text('{"complete":true}\n', encoding="utf-8")
    destination = tmp_path / "published" / "artifact.json"

    def reject_cross_volume(*_: Any, **__: Any) -> None:
        raise OSError(errno.EXDEV, "injected cross-volume hard-link failure")

    link_helper = "_windows_link_relative" if os.name == "nt" else "_posix_link_open_descriptor"
    monkeypatch.setattr(publication_module, link_helper, reject_cross_volume)
    with pytest.raises(OSError, match="fail-closed across volumes"):
        publish_file_no_overwrite(staged, destination)

    assert not destination.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows no-share parent-handle contract")
def test_missing_parent_is_created_and_pinned_without_name_reopen_adoption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import histo_audit.pannuke.publication as publication_module

    staged = tmp_path / "staged.bin"
    staged.write_bytes(b"owned parent-chain payload")
    parent = tmp_path / "new-parent"
    moved = tmp_path / "moved-parent"
    original_open = publication_module._windows_open_relative_descriptor
    rename_attempts: list[str] = []

    def attempt_parent_swap(handle: int, name: str, **kwargs: Any) -> int:
        descriptor = original_open(handle, name, **kwargs)
        if kwargs.get("create") and name == parent.name:
            try:
                os.rename(parent, moved)
            except PermissionError:
                rename_attempts.append("blocked")
            else:
                rename_attempts.append("swapped")
                parent.mkdir()
                (parent / "foreign.txt").write_text("foreign", encoding="utf-8")
        return descriptor

    monkeypatch.setattr(
        publication_module,
        "_windows_open_relative_descriptor",
        attempt_parent_swap,
    )
    published = publish_file_no_overwrite(staged, parent / "artifact.bin")

    assert rename_attempts == ["blocked"]
    assert published.path.read_bytes() == b"owned parent-chain payload"
    assert not moved.exists()
    assert sorted(path.name for path in parent.iterdir()) == ["artifact.bin"]


def test_standalone_success_marker_rejects_unowned_predecessor_seal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "owned-root"
    owned_root = create_directory_no_overwrite(root)
    predecessor = root / "artifact_manifest.json"
    predecessor.write_text('{"foreign":true}\n', encoding="utf-8")
    staged_marker = tmp_path / ".immutable.json"
    staged_marker.write_text('{"state":"complete"}\n', encoding="utf-8")

    with pytest.raises(PermissionError, match="sealed/immutable ancestor"):
        publish_success_marker_no_overwrite(
            staged_marker,
            root / ".immutable.json",
            owned_parent=owned_root,
        )

    assert predecessor.read_text(encoding="utf-8") == '{"foreign":true}\n'
    assert not (root / ".immutable.json").exists()
    predecessor.unlink()
    rollback_owned_publications([owned_root])


def test_active_bundle_target_lock_blocks_json_compare_and_swap(tmp_path: Path) -> None:
    path = tmp_path / "provenance.json"

    with (
        ExclusiveBundlePublicationLock((path,), role="test fixture"),
        pytest.raises(FileExistsError, match="active or requires stale-lock review"),
    ):
        write_json_compare_and_swap(path, {"version": 1})

    assert not os.path.lexists(path)


def test_bundle_o_excl_lock_is_order_independent_and_not_auto_reclaimed(
    tmp_path: Path,
) -> None:
    first = tmp_path / "manifest.json"
    second = tmp_path / "report.json"
    lock = ExclusiveBundlePublicationLock((first, second), role="test bundle")
    reverse = ExclusiveBundlePublicationLock((second, first), role="test bundle")

    assert lock.path == reverse.path
    with lock:
        assert os.path.lexists(lock.path)
        with pytest.raises(FileExistsError, match="active or requires stale-lock review"):
            reverse.__enter__()
    assert not os.path.lexists(lock.path)

    lock.path.parent.mkdir(parents=True, exist_ok=True)
    lock.path.write_text("stale lock requiring review\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="requires stale-lock review"):
        ExclusiveBundlePublicationLock((first, second), role="test bundle").__enter__()
    assert lock.path.read_text(encoding="utf-8") == "stale lock requiring review\n"
    lock.path.unlink()


def test_bundle_o_excl_lock_rejects_partial_destination_overlap(tmp_path: Path) -> None:
    first = tmp_path / "manifest.json"
    shared = tmp_path / "shared.json"
    third = tmp_path / "third.json"
    left = ExclusiveBundlePublicationLock((first, shared), role="left bundle")
    right = ExclusiveBundlePublicationLock((shared, third), role="right bundle")

    with left:
        with pytest.raises(FileExistsError, match="active or requires stale-lock review"):
            right.__enter__()
        assert not os.path.lexists(right.path)
        left.assert_owned()

    assert all(not os.path.lexists(path) for path in {*left.lock_paths, *right.lock_paths})


def test_two_concurrent_cas_callers_with_same_expected_hash_have_one_winner(
    tmp_path: Path,
) -> None:
    path = tmp_path / "provenance.json"
    initial = {"version": 0}
    candidates = ({"version": 1}, {"version": 2})
    write_json_compare_and_swap(path, initial)
    initial_sha256 = sha256_file(path)
    start = Barrier(2)

    def update(payload: dict[str, int]) -> BaseException | None:
        start.wait()
        try:
            write_json_compare_and_swap(
                path,
                payload,
                expected_previous_sha256=initial_sha256,
            )
        except BaseException as error:
            return error
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(update, candidates))

    assert sum(error is None for error in outcomes) == 1
    failures = [error for error in outcomes if error is not None]
    assert len(failures) == 1
    assert isinstance(failures[0], (FileExistsError, PanNukeAcquisitionError))
    assert json.loads(path.read_text(encoding="utf-8")) in candidates


def test_cas_does_not_claim_or_rollback_foreign_post_replace_file(tmp_path: Path) -> None:
    path = tmp_path / "provenance.json"
    write_json_compare_and_swap(path, {"version": 0})
    initial_sha256 = sha256_file(path)
    foreign = b'{"foreign": true}\n'
    from histo_audit.pannuke import acquisition

    real_replace = acquisition.os.replace

    def replace_then_inject_foreign(source: Path, destination: Path) -> None:
        real_replace(source, destination)
        destination.unlink()
        destination.write_bytes(foreign)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(acquisition.os, "replace", replace_then_inject_foreign)
        with pytest.raises(RuntimeError, match="ownership changed during CAS"):
            write_json_compare_and_swap(
                path,
                {"version": 1},
                expected_previous_sha256=initial_sha256,
            )

    assert path.read_bytes() == foreign


def test_broken_final_symlink_is_preserved_and_not_followed(tmp_path: Path) -> None:
    missing_target = tmp_path / "missing-target.json"
    destination = tmp_path / "provenance.json"
    try:
        destination.symlink_to(missing_target)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")
    link_target_before = destination.readlink()

    with pytest.raises(FileNotFoundError):
        write_json_compare_and_swap(destination, {"version": 1})

    assert destination.is_symlink()
    assert destination.readlink() == link_target_before
    assert not missing_target.exists()


def test_bundle_preflights_and_binds_manifest_and_report(tmp_path: Path) -> None:
    _, manifest = _tiny_manifest(tmp_path)
    manifest_path = tmp_path / "data" / "manifests" / "pannuke_acquisition.json"
    report_path = tmp_path / "reports" / "pannuke_acquisition_verification.json"

    # The actual Git-safety subprocess contract is covered against this repository.
    # Keep bundle semantics isolated from Git by substituting that one evidence source.
    from histo_audit.pannuke import acquisition

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            acquisition,
            "git_ignore_evidence",
            lambda _project_root: {"status": "passed", "fixture": True},
        )
        written_manifest, written_report = write_acquisition_artifact_bundle(
            manifest_path,
            report_path,
            manifest,
            project_root=tmp_path,
        )
        write_acquisition_artifact_bundle(
            manifest_path,
            report_path,
            manifest,
            project_root=tmp_path,
        )

    report = json.loads(written_report.read_text(encoding="utf-8"))
    assert written_manifest == manifest_path
    assert report["acquisition_manifest_sha256"] == sha256_file(written_manifest)


def test_bundle_o_excl_lock_spans_input_verification_through_final_readback(
    tmp_path: Path,
) -> None:
    _, manifest = _tiny_manifest(tmp_path)
    manifest_path = tmp_path / "data" / "manifests" / "pannuke_acquisition.json"
    report_path = tmp_path / "reports" / "pannuke_acquisition_verification.json"
    probe = ExclusiveBundlePublicationLock(
        (manifest_path, report_path), role="acquisition provenance bundle"
    )
    from histo_audit.pannuke import acquisition

    real_verify = acquisition.verify_acquisition_raw_metadata_unchanged
    verification_calls = 0

    def verify_while_locked(payload: Mapping[str, Any], project_root: Path) -> list[dict[str, Any]]:
        nonlocal verification_calls
        verification_calls += 1
        assert all(os.path.lexists(path) for path in probe.lock_paths)
        return real_verify(payload, project_root)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            acquisition,
            "git_ignore_evidence",
            lambda _project_root: {"status": "passed", "fixture": True},
        )
        monkeypatch.setattr(
            acquisition,
            "verify_acquisition_raw_metadata_unchanged",
            verify_while_locked,
        )
        write_acquisition_artifact_bundle(
            manifest_path,
            report_path,
            manifest,
            project_root=tmp_path,
        )

    assert verification_calls == 3
    assert all(not os.path.lexists(path) for path in probe.lock_paths)


def test_active_destination_lock_blocks_bundle_before_any_publish(tmp_path: Path) -> None:
    _, manifest = _tiny_manifest(tmp_path)
    manifest_path = tmp_path / "data" / "manifests" / "pannuke_acquisition.json"
    report_path = tmp_path / "reports" / "pannuke_acquisition_verification.json"
    from histo_audit.pannuke import acquisition

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            acquisition,
            "git_ignore_evidence",
            lambda _project_root: {"status": "passed", "fixture": True},
        )
        with (
            ExclusivePublicationLock(report_path, role="test fixture"),
            pytest.raises(FileExistsError, match="publication is active"),
        ):
            write_acquisition_artifact_bundle(
                manifest_path,
                report_path,
                manifest,
                project_root=tmp_path,
            )

    assert not manifest_path.exists()
    assert not report_path.exists()


def test_bundle_concurrent_foreign_final_is_never_overwritten_or_removed(
    tmp_path: Path,
) -> None:
    _, manifest = _tiny_manifest(tmp_path)
    manifest_path = tmp_path / "data" / "manifests" / "pannuke_acquisition.json"
    report_path = tmp_path / "reports" / "pannuke_acquisition_verification.json"
    foreign = b"foreign concurrent report\n"
    from histo_audit.pannuke import acquisition

    real_publish = acquisition._publish_json_cas_locked
    injected = False

    def insert_foreign_then_publish(
        destination: Path,
        payload: Mapping[str, Any],
        previous: bytes | None,
    ) -> acquisition.PublishedPath:
        nonlocal injected
        if not injected:
            injected = True
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(foreign)
        return real_publish(destination, payload, previous)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            acquisition,
            "git_ignore_evidence",
            lambda _project_root: {"status": "passed", "fixture": True},
        )
        monkeypatch.setattr(acquisition, "_publish_json_cas_locked", insert_foreign_then_publish)
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            write_acquisition_artifact_bundle(
                manifest_path,
                report_path,
                manifest,
                project_root=tmp_path,
            )

    assert injected
    assert report_path.read_bytes() == foreign
    assert not manifest_path.exists()


def test_bundle_rollback_preserves_foreign_report_replacement(tmp_path: Path) -> None:
    _, manifest = _tiny_manifest(tmp_path)
    manifest_path = tmp_path / "data" / "manifests" / "pannuke_acquisition.json"
    report_path = tmp_path / "reports" / "pannuke_acquisition_verification.json"
    foreign = b"foreign replacement report\n"
    from histo_audit.pannuke import acquisition

    real_publish = acquisition._publish_json_cas_locked
    calls = 0

    def replace_report_then_fail_manifest(
        destination: Path,
        payload: Mapping[str, Any],
        previous: bytes | None,
    ) -> acquisition.PublishedPath:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected manifest publication failure")
        publication = real_publish(destination, payload, previous)
        destination.unlink()
        destination.write_bytes(foreign)
        return publication

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            acquisition,
            "git_ignore_evidence",
            lambda _project_root: {"status": "passed", "fixture": True},
        )
        monkeypatch.setattr(
            acquisition,
            "_publish_json_cas_locked",
            replace_report_then_fail_manifest,
        )
        with pytest.raises(RuntimeError, match="rollback was incomplete"):
            write_acquisition_artifact_bundle(
                manifest_path,
                report_path,
                manifest,
                project_root=tmp_path,
            )

    assert calls == 2
    assert report_path.read_bytes() == foreign
    assert not manifest_path.exists()


@pytest.mark.parametrize("inventory_change", ("addition", "removal"))
def test_bundle_rejects_raw_inventory_change_after_manifest_build(
    tmp_path: Path,
    inventory_change: str,
) -> None:
    raw, manifest = _tiny_manifest(tmp_path)
    changed_path = raw / "inventory-only.txt"
    if inventory_change == "addition":
        changed_path.write_text("added after verification\n", encoding="utf-8")
    else:
        (raw / "Fold 1" / "images" / "fold1" / "images.npy").unlink()
    manifest_path = tmp_path / "data" / "manifests" / "pannuke_acquisition.json"
    report_path = tmp_path / "reports" / "pannuke_acquisition_verification.json"
    from histo_audit.pannuke import acquisition

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            acquisition,
            "git_ignore_evidence",
            lambda _project_root: {"status": "passed", "fixture": True},
        )
        with pytest.raises(PanNukeAcquisitionError, match="inventory changed"):
            write_acquisition_artifact_bundle(
                manifest_path,
                report_path,
                manifest,
                project_root=tmp_path,
            )

    assert not manifest_path.exists()
    assert not report_path.exists()


def test_bundle_publish_time_raw_mutation_rolls_back_both_artifacts(tmp_path: Path) -> None:
    raw, manifest = _tiny_manifest(tmp_path)
    manifest_path = tmp_path / "data" / "manifests" / "pannuke_acquisition.json"
    report_path = tmp_path / "reports" / "pannuke_acquisition_verification.json"
    image_path = raw / "Fold 1" / "images" / "fold1" / "images.npy"
    from histo_audit.pannuke import acquisition

    real_publish = acquisition._publish_json_cas_locked
    mutation_injected = False

    def publish_then_mutate(
        path: Path, payload: Mapping[str, Any], previous: bytes | None
    ) -> acquisition.PublishedPath:
        nonlocal mutation_injected
        published = real_publish(path, payload, previous)
        if not mutation_injected:
            mutation_injected = True
            image_path.write_bytes(image_path.read_bytes() + b"changed")
        return published

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            acquisition,
            "git_ignore_evidence",
            lambda _project_root: {"status": "passed", "fixture": True},
        )
        monkeypatch.setattr(acquisition, "_publish_json_cas_locked", publish_then_mutate)
        with pytest.raises(PanNukeAcquisitionError, match="inventory changed"):
            write_acquisition_artifact_bundle(
                manifest_path,
                report_path,
                manifest,
                project_root=tmp_path,
            )

    assert mutation_injected
    assert not manifest_path.exists()
    assert not report_path.exists()


def test_bundle_refuses_conflicting_report_before_creating_manifest(tmp_path: Path) -> None:
    _, manifest = _tiny_manifest(tmp_path)
    manifest_path = tmp_path / "data" / "manifests" / "pannuke_acquisition.json"
    report_path = tmp_path / "reports" / "pannuke_acquisition_verification.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text('{"stale": true}\n', encoding="utf-8")
    from histo_audit.pannuke import acquisition

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            acquisition,
            "git_ignore_evidence",
            lambda _project_root: {"status": "passed", "fixture": True},
        )
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            write_acquisition_artifact_bundle(
                manifest_path,
                report_path,
                manifest,
                project_root=tmp_path,
            )

    assert not manifest_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8")) == {"stale": True}


def test_gitignore_blocks_raw_zip_npy_and_leaves_provenance_trackable(
    tmp_path: Path,
) -> None:
    project = Path(__file__).resolve().parents[1]
    patterns = {
        value.strip() for value in (project / ".gitignore").read_text(encoding="utf-8").splitlines()
    }

    assert {
        "*.zip",
        "*.npy",
        "data/raw/**",
        "artifacts/duplicate_audit/*.npz",
        "artifacts/duplicate_audit/.frozen_resnet18_duplicate_embeddings.resume/**",
    }.issubset(patterns)

    # Prove the Git behavior against a self-contained repository. The public
    # checkout intentionally excludes the raw PanNuke release, so this test must
    # not depend on one researcher's local dataset tree.
    (tmp_path / ".gitignore").write_text(
        (project / ".gitignore").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    raw_root = tmp_path / "data" / "raw" / "pannuke"
    representative_files = (
        raw_root / "Fold 1" / "images" / "fold1" / "images.npy",
        raw_root / "Fold 1" / "README.md",
        raw_root / "Fold 1.zip",
    )
    for path in representative_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"representative ignored PanNuke fixture\n")
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    evidence = git_ignore_evidence(tmp_path)
    assert evidence["status"] == "passed"
    assert evidence["raw_release_file_count"] == evidence["ignored_raw_file_count"]
    assert evidence["tracked_raw_file_count"] == 0
    assert any(value["path"].endswith("README.md") for value in evidence["ignored_raw_paths"])


def test_checked_in_acquisition_artifacts_have_strict_schema_and_binding() -> None:
    project = Path(__file__).resolve().parents[1]
    manifest_path = project / "data" / "manifests" / "pannuke_acquisition.json"
    report_path = project / "reports" / "pannuke_acquisition_verification.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    validate_acquisition_manifest(manifest)
    assert [
        (value["fold"], value["size_bytes"], value["sha256"]) for value in manifest["archives"]
    ] == [(value.fold, value.size_bytes, value.sha256) for value in DEFAULT_ARCHIVE_EXPECTATIONS]
    assert report["status"] == "passed"
    assert report["scope"] == "acquisition_provenance_only"
    assert report["acquisition_manifest_sha256"] == sha256_file(manifest_path)
    assert report["archive_crc_failed_member_count"] == 0
    assert report["rejected_unsafe_archive_member_path_count"] == 0
    assert report["git_safety"]["ignored_raw_file_count"] == 21
    assert report["git_safety"]["tracked_raw_file_count"] == 0
    assert report["scientific_stage_advanced"] is False
