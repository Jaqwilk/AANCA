"""Synthetic fail-closed contracts for optional per-file WOF LZX copies."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import histo_audit.pannuke.publication as publication
from histo_audit.pannuke.publication import (
    ANCHORED_PHYSICAL_COPY_WOF_LZX_POLICY,
    WOF_LZX_MIN_FREE_MARGIN_BYTES,
    AnchoredPhysicalCopyBoundaryError,
    anchored_physical_copy_session,
)
from histo_audit.utils.run_tracking import sha256_file


def _roots(
    tmp_path: Path, payload: bytes = b"compressible payload" * 200
) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    source_root.mkdir()
    destination_root.mkdir()
    source = source_root / "artifact.bin"
    source.write_bytes(payload)
    return source_root, destination_root, source


def _ample_space(source_size: int) -> int:
    return WOF_LZX_MIN_FREE_MARGIN_BYTES + source_size + 4096


def test_wof_copy_compresses_once_and_preserves_source_and_logical_hash(
    tmp_path: Path,
) -> None:
    source_root, destination_root, source = _roots(tmp_path)
    source_sha = sha256_file(source)
    source_identity = source.stat()
    compressor_calls: list[Path] = []
    free_space_calls: list[Path] = []

    def compressor(path: Path) -> None:
        compressor_calls.append(path)
        assert sha256_file(path) == source_sha

    def free_space(path: Path) -> int:
        free_space_calls.append(path)
        return _ample_space(source.stat().st_size)

    with anchored_physical_copy_session(
        source_root,
        destination_root,
        compression_policy=ANCHORED_PHYSICAL_COPY_WOF_LZX_POLICY,
        compressor=compressor,
        free_space_probe=free_space,
    ) as session:
        assert session.compression_policy == ANCHORED_PHYSICAL_COPY_WOF_LZX_POLICY
        published = session.copy_file_no_overwrite(
            "artifact.bin",
            expected_size_bytes=source.stat().st_size,
            expected_sha256=source_sha,
        )

    destination = destination_root / "artifact.bin"
    assert compressor_calls == [destination]
    assert free_space_calls == [destination_root, destination_root]
    assert published.sha256 == source_sha
    assert sha256_file(destination) == source_sha
    assert sha256_file(source) == source_sha
    source_after = source.stat()
    assert (source_after.st_dev, source_after.st_ino, source_after.st_size) == (
        source_identity.st_dev,
        source_identity.st_ino,
        source_identity.st_size,
    )


def test_wof_copy_rejects_changed_content_and_rolls_back_owned_file(
    tmp_path: Path,
) -> None:
    source_root, destination_root, source = _roots(tmp_path, b"A" * 8192)
    source_sha = sha256_file(source)
    calls = 0

    def corrupt_destination(path: Path) -> None:
        nonlocal calls
        calls += 1
        path.write_bytes(b"B" * source.stat().st_size)

    with (
        pytest.raises(RuntimeError, match="post-compression readback"),
        anchored_physical_copy_session(
            source_root,
            destination_root,
            compression_policy=ANCHORED_PHYSICAL_COPY_WOF_LZX_POLICY,
            compressor=corrupt_destination,
            free_space_probe=lambda _path: _ample_space(source.stat().st_size),
        ) as session,
    ):
        session.copy_file_no_overwrite(
            "artifact.bin",
            expected_size_bytes=source.stat().st_size,
            expected_sha256=source_sha,
        )

    assert calls == 1
    assert not os.path.lexists(destination_root / "artifact.bin")
    assert sha256_file(source) == source_sha


def test_wof_copy_identity_replacement_is_never_adopted_or_deleted(
    tmp_path: Path,
) -> None:
    source_root, destination_root, source = _roots(tmp_path)
    source_sha = sha256_file(source)
    replacement = b"foreign replacement"
    replacement_succeeded = False

    def replace_destination(path: Path) -> None:
        nonlocal replacement_succeeded
        try:
            path.unlink()
            path.write_bytes(replacement)
        except PermissionError:
            raise OSError("guard blocked identity replacement") from None
        replacement_succeeded = True

    with (
        pytest.raises((OSError, RuntimeError)),
        anchored_physical_copy_session(
            source_root,
            destination_root,
            compression_policy=ANCHORED_PHYSICAL_COPY_WOF_LZX_POLICY,
            compressor=replace_destination,
            free_space_probe=lambda _path: _ample_space(source.stat().st_size),
        ) as session,
    ):
        session.copy_file_no_overwrite(
            "artifact.bin",
            expected_size_bytes=source.stat().st_size,
            expected_sha256=source_sha,
        )

    destination = destination_root / "artifact.bin"
    if replacement_succeeded:
        assert destination.read_bytes() == replacement
    else:
        assert not os.path.lexists(destination)
    assert sha256_file(source) == source_sha


def test_wof_copy_compressor_failure_is_not_retried_and_rolls_back(
    tmp_path: Path,
) -> None:
    source_root, destination_root, source = _roots(tmp_path)
    source_sha = sha256_file(source)
    calls = 0

    def compact_nonzero(_path: Path) -> None:
        nonlocal calls
        calls += 1
        raise OSError("compact.exe WOF LZX failed with exit code 7")

    with (
        pytest.raises(OSError, match="exit code 7"),
        anchored_physical_copy_session(
            source_root,
            destination_root,
            compression_policy=ANCHORED_PHYSICAL_COPY_WOF_LZX_POLICY,
            compressor=compact_nonzero,
            free_space_probe=lambda _path: _ample_space(source.stat().st_size),
        ) as session,
    ):
        session.copy_file_no_overwrite(
            "artifact.bin",
            expected_size_bytes=source.stat().st_size,
            expected_sha256=source_sha,
        )

    assert calls == 1
    assert not os.path.lexists(destination_root / "artifact.bin")
    assert sha256_file(source) == source_sha


def test_wof_copy_fails_before_copy_when_next_file_plus_margin_does_not_fit(
    tmp_path: Path,
) -> None:
    source_root, destination_root, source = _roots(tmp_path)
    source_sha = sha256_file(source)

    def forbidden_compressor(_path: Path) -> None:
        raise AssertionError("low-space preflight invoked the compressor")

    with (
        pytest.raises(OSError, match="before copy"),
        anchored_physical_copy_session(
            source_root,
            destination_root,
            compression_policy=ANCHORED_PHYSICAL_COPY_WOF_LZX_POLICY,
            compressor=forbidden_compressor,
            free_space_probe=lambda _path: (
                WOF_LZX_MIN_FREE_MARGIN_BYTES + source.stat().st_size - 1
            ),
        ) as session,
    ):
        session.copy_file_no_overwrite(
            "artifact.bin",
            expected_size_bytes=source.stat().st_size,
            expected_sha256=source_sha,
        )

    assert not os.path.lexists(destination_root / "artifact.bin")
    assert sha256_file(source) == source_sha


def test_wof_copy_fails_after_compression_below_margin_and_rolls_back(
    tmp_path: Path,
) -> None:
    source_root, destination_root, source = _roots(tmp_path)
    source_sha = sha256_file(source)
    observations = iter(
        (
            _ample_space(source.stat().st_size),
            WOF_LZX_MIN_FREE_MARGIN_BYTES - 1,
        )
    )
    compressor_calls = 0

    def compressor(_path: Path) -> None:
        nonlocal compressor_calls
        compressor_calls += 1

    with (
        pytest.raises(OSError, match="after compression"),
        anchored_physical_copy_session(
            source_root,
            destination_root,
            compression_policy=ANCHORED_PHYSICAL_COPY_WOF_LZX_POLICY,
            compressor=compressor,
            free_space_probe=lambda _path: next(observations),
        ) as session,
    ):
        session.copy_file_no_overwrite(
            "artifact.bin",
            expected_size_bytes=source.stat().st_size,
            expected_sha256=source_sha,
        )

    assert compressor_calls == 1
    assert not os.path.lexists(destination_root / "artifact.bin")
    assert sha256_file(source) == source_sha


def test_wof_copy_destination_boundary_failure_stays_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, destination_root, source = _roots(tmp_path)
    source_sha = sha256_file(source)
    compressor_finished = False
    original_assert_tree = publication.AnchoredPhysicalCopySession._assert_tree_current

    def compressor(_path: Path) -> None:
        nonlocal compressor_finished
        compressor_finished = True

    def fail_destination_boundary(
        session: Any,
        *,
        source: bool,
    ) -> None:
        if not source and compressor_finished:
            raise RuntimeError("synthetic destination boundary swap")
        original_assert_tree(session, source=source)

    monkeypatch.setattr(
        publication.AnchoredPhysicalCopySession,
        "_assert_tree_current",
        fail_destination_boundary,
    )
    with (
        pytest.raises(AnchoredPhysicalCopyBoundaryError) as captured,
        anchored_physical_copy_session(
            source_root,
            destination_root,
            compression_policy=ANCHORED_PHYSICAL_COPY_WOF_LZX_POLICY,
            compressor=compressor,
            free_space_probe=lambda _path: _ample_space(source.stat().st_size),
        ) as session,
    ):
        session.copy_file_no_overwrite(
            "artifact.bin",
            expected_size_bytes=source.stat().st_size,
            expected_sha256=source_sha,
        )

    assert captured.value.destination_tree_current is False
    assert captured.value.rollback_complete is True
    assert not os.path.lexists(destination_root / "artifact.bin")
    assert sha256_file(source) == source_sha


def test_copy_session_entry_root_swap_is_typed_and_never_uses_pathname_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, destination_root, _source = _roots(tmp_path)
    moved_destination_root = tmp_path / "moved-destination"
    body_entered = False

    @contextmanager
    def swap_during_entry(
        _paths: Any,
        **_kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        os.rename(destination_root, moved_destination_root)
        destination_root.mkdir()
        (destination_root / "foreign.txt").write_bytes(b"foreign replacement root")
        raise OSError("synthetic entry failure after destination root swap")
        yield {}

    monkeypatch.setattr(
        publication,
        "_locked_publication_parents",
        swap_during_entry,
    )

    with (
        pytest.raises(AnchoredPhysicalCopyBoundaryError) as captured,
        anchored_physical_copy_session(source_root, destination_root),
    ):
        body_entered = True

    error = captured.value
    assert body_entered is False
    assert error.source_tree_current is False
    assert error.destination_tree_current is False
    assert error.rollback_complete is False
    assert error.expected_destination_root == destination_root
    assert isinstance(error.__cause__, OSError)
    assert "before anchored session creation" in str(error)
    assert (destination_root / "foreign.txt").read_bytes() == b"foreign replacement root"
    assert not tuple(moved_destination_root.iterdir())


@pytest.mark.skipif(os.name != "nt", reason="real WOF LZX integration is Windows-only")
def test_real_windows_default_wof_copy_uses_native_handoff_and_preserves_hash(
    tmp_path: Path,
) -> None:
    source_root, destination_root, source = _roots(tmp_path, b"\0" * (2 * 1024 * 1024))
    source_sha = sha256_file(source)
    source_before = source.stat()
    free_space_calls: list[Path] = []

    def free_space(path: Path) -> int:
        free_space_calls.append(path)
        return _ample_space(source.stat().st_size)

    with anchored_physical_copy_session(
        source_root,
        destination_root,
        compression_policy=ANCHORED_PHYSICAL_COPY_WOF_LZX_POLICY,
        free_space_probe=free_space,
    ) as session:
        published = session.copy_file_no_overwrite(
            "artifact.bin",
            expected_size_bytes=source.stat().st_size,
            expected_sha256=source_sha,
        )

    destination = destination_root / "artifact.bin"
    assert published.sha256 == source_sha
    assert sha256_file(destination) == source_sha
    assert sha256_file(source) == source_sha
    source_after = source.stat()
    destination_after = destination.stat()
    assert (source_after.st_dev, source_after.st_ino, source_after.st_size) == (
        source_before.st_dev,
        source_before.st_ino,
        source_before.st_size,
    )
    assert destination_after.st_size == source_after.st_size
    assert destination_after.st_nlink == 1
    assert (destination_after.st_dev, destination_after.st_ino) != (
        source_after.st_dev,
        source_after.st_ino,
    )
    assert free_space_calls == [destination_root, destination_root]
    assert tuple(path.name for path in source_root.iterdir()) == ("artifact.bin",)
    assert tuple(path.name for path in destination_root.iterdir()) == ("artifact.bin",)


@pytest.mark.skipif(os.name != "nt", reason="trusted compact.exe path is Windows-only")
def test_default_wof_compressor_uses_exact_lzx_command_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_root = tmp_path / "Windows"
    compact = system_root / "System32" / "compact.exe"
    compact.parent.mkdir(parents=True)
    compact.write_bytes(b"synthetic executable")
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"payload")
    calls: list[list[str]] = []

    def nonzero(command: list[str], **kwargs: Any) -> Any:
        calls.append(command)
        assert kwargs["shell"] is False
        return SimpleNamespace(returncode=7, stdout="", stderr="synthetic compact failure")

    monkeypatch.setenv("SYSTEMROOT", str(system_root))
    monkeypatch.setattr(subprocess, "run", nonzero)

    with pytest.raises(OSError, match="exit code 7"):
        publication._windows_wof_lzx_compress_file(destination)

    assert calls == [
        [
            str(compact),
            "/c",
            "/exe:lzx",
            "/f",
            str(destination),
        ]
    ]
