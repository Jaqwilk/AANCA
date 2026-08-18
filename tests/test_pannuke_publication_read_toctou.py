"""Race regressions for descriptor-anchored publication reads."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from histo_audit.pannuke import publication


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX permits renaming an open file; Windows denies delete sharing here",
)
def test_anchored_read_rejects_leaf_replacement_during_final_logical_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "evidence.json"
    displaced = tmp_path / "captured-evidence.json"
    replacement = tmp_path / "replacement.json"
    original_payload = b'{"status":"captured"}\n'
    replacement_payload = b'{"status":"replacement"}\n'
    source.write_bytes(original_payload)
    replacement.write_bytes(replacement_payload)
    real_descriptor_sha256 = publication._descriptor_sha256
    digest_call_count = 0

    def replace_during_second_descriptor_hash(
        descriptor: int,
        *,
        chunk_size_bytes: int,
    ) -> str:
        nonlocal digest_call_count
        digest_call_count += 1
        digest = real_descriptor_sha256(
            descriptor,
            chunk_size_bytes=chunk_size_bytes,
        )
        if digest_call_count == 2:
            source.rename(displaced)
            replacement.rename(source)
        return digest

    monkeypatch.setattr(
        publication,
        "_descriptor_sha256",
        replace_during_second_descriptor_hash,
    )

    with pytest.raises(RuntimeError, match="logical source changed during capture"):
        publication.read_file_anchored(source)

    assert digest_call_count == 2
    assert source.read_bytes() == replacement_payload
    assert displaced.read_bytes() == original_payload
