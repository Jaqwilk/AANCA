"""Focused tests for descriptor-anchored publication reads."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import histo_audit.pannuke.publication as publication
from histo_audit.pannuke.publication import read_file_anchored


def test_read_file_anchored_returns_exact_bytes(tmp_path: Path) -> None:
    source = tmp_path / "evidence.json"
    payload = b'{"status":"complete"}\n\x00binary-tail'
    source.write_bytes(payload)

    assert read_file_anchored(source) == payload


def test_read_file_anchored_rejects_empty_by_default_and_allows_it_explicitly(
    tmp_path: Path,
) -> None:
    source = tmp_path / "empty.txt"
    source.write_bytes(b"")

    with pytest.raises(ValueError, match="must not be empty"):
        read_file_anchored(source)

    assert read_file_anchored(source, allow_empty=True) == b""


def test_read_file_anchored_enforces_limit_before_payload_accumulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "bounded.bin"
    source.write_bytes(b"0123456789")
    digest_calls = 0
    original_digest = publication._descriptor_sha256

    def counted_digest(*args: Any, **kwargs: Any) -> str:
        nonlocal digest_calls
        digest_calls += 1
        return original_digest(*args, **kwargs)

    monkeypatch.setattr(publication, "_descriptor_sha256", counted_digest)

    with pytest.raises(ValueError, match="bounded size limit"):
        read_file_anchored(source, max_bytes=9)

    assert digest_calls == 0
    assert read_file_anchored(source, max_bytes=10) == b"0123456789"


@pytest.mark.parametrize("invalid_limit", [True, -1, 1.0])
def test_read_file_anchored_rejects_non_exact_limit(
    tmp_path: Path,
    invalid_limit: object,
) -> None:
    source = tmp_path / "bounded.bin"
    source.write_bytes(b"x")

    with pytest.raises(ValueError, match="non-negative exact integer"):
        read_file_anchored(source, max_bytes=invalid_limit)  # type: ignore[arg-type]


def test_read_file_anchored_rejects_hardlinked_source(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    alias = tmp_path / "alias.bin"
    source.write_bytes(b"shared inode")
    try:
        os.link(source, alias)
    except OSError as error:
        pytest.skip(f"hard links are unavailable on this test filesystem: {error}")

    assert source.stat().st_nlink >= 2
    with pytest.raises(ValueError, match="single-link"):
        read_file_anchored(source)


def test_read_file_anchored_rejects_symlink_or_reparse_source(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    link = tmp_path / "source-link.bin"
    source.write_bytes(b"link target")
    try:
        link.symlink_to(source)
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable on this test system: {error}")

    with pytest.raises(ValueError, match=r"lexical regular file|non-reparse"):
        read_file_anchored(link)


def test_read_file_anchored_rejects_same_payload_name_replacement_after_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    displaced = tmp_path / "captured-original.bin"
    payload = b"identical bytes cannot substitute for object identity"
    source.write_bytes(payload)
    original_capture = publication._capture_physical_publication_source
    replacement_identities: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def replace_name_after_capture(
        path: Path,
        parents: Mapping[str, Any],
        *,
        max_bytes: int | None = None,
    ) -> Any:
        captured = original_capture(path, parents, max_bytes=max_bytes)
        if os.name == "nt":
            # Production denies delete sharing. Closing only the test-owned retained
            # descriptor lets this regression deterministically emulate a name race.
            captured.close_retained_descriptor()
        os.replace(path, displaced)
        path.write_bytes(payload)
        replacement = path.stat()
        replacement_identities.append(
            (captured.identity[:2], (replacement.st_dev, replacement.st_ino))
        )
        return captured

    monkeypatch.setattr(
        publication,
        "_capture_physical_publication_source",
        replace_name_after_capture,
    )

    with pytest.raises((RuntimeError, ValueError), match=r"changed|identity|capture"):
        read_file_anchored(source)

    assert len(replacement_identities) == 1
    assert replacement_identities[0][0] != replacement_identities[0][1]
    assert source.read_bytes() == displaced.read_bytes() == payload
