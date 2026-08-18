"""Lifecycle regression tests for the schema-v3 Authority-D Q boundary."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.test_preregistration_amendment import _local_replacement_terminal_contract


def _lineage(amendment: Any, receipt: dict[str, Any], receipt_path: Path) -> dict[str, Any]:
    return {
        "terminal_qualification_receipt_path": str(receipt_path),
        "terminal_qualification_receipt_sha256": amendment._atomic_json_sha256(receipt),
        "terminal_qualification_receipt": receipt,
    }


def test_public_q_verifier_accepts_the_canonical_project_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    amendment, parent_state, receipt, receipt_path = _local_replacement_terminal_contract()
    receipt_bytes = amendment._atomic_json_bytes(receipt)
    real_read = amendment._read_stable_single_link_file

    def read_with_unpublished_q(
        path: Path,
        *,
        role: str,
        allow_empty: bool = False,
        max_bytes: int | None = None,
    ) -> bytes:
        if Path(path) == receipt_path:
            return receipt_bytes
        return real_read(
            Path(path),
            role=role,
            allow_empty=allow_empty,
            max_bytes=max_bytes,
        )

    monkeypatch.setattr(amendment, "_read_stable_single_link_file", read_with_unpublished_q)

    project_root = parent_state.directory.parent.parent.parent
    observed = amendment.verify_resource_bounded_replacement_terminal_qualification_receipt(
        receipt_path,
        project_root=project_root,
        parent_authority_directory=parent_state.directory,
    )

    assert observed == _lineage(amendment, receipt, receipt_path)


def test_sealed_d_readbacks_ignore_later_run_state_but_not_q_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    amendment, parent_state, receipt, receipt_path = _local_replacement_terminal_contract()
    receipt_snapshot = copy.deepcopy(receipt)
    receipt_bytes = amendment._atomic_json_bytes(receipt_snapshot)
    receipt_sha256 = amendment._atomic_json_sha256(receipt_snapshot)
    lineage = _lineage(amendment, receipt_snapshot, receipt_path)
    lineage_snapshot = copy.deepcopy(lineage)
    run_state_paths = {
        Path(record["path"]) for record in receipt_snapshot["run_state"]["files"].values()
    }
    real_read = amendment._read_stable_single_link_file
    lifecycle_advanced = False
    q_tampered = False
    run_state_reads = 0

    def read_after_seal(
        path: Path,
        *,
        role: str,
        allow_empty: bool = False,
        max_bytes: int | None = None,
    ) -> bytes:
        nonlocal run_state_reads
        lexical = Path(path)
        if lexical == receipt_path:
            return receipt_bytes + b" " if q_tampered else receipt_bytes
        payload = real_read(
            lexical,
            role=role,
            allow_empty=allow_empty,
            max_bytes=max_bytes,
        )
        if lexical in run_state_paths:
            run_state_reads += 1
            if lifecycle_advanced:
                return payload + b"\n"
        return payload

    monkeypatch.setattr(amendment, "_read_stable_single_link_file", read_after_seal)

    assert (
        amendment._canonical_replacement_publication_failure_lineage(
            lineage,
            resource_parent=parent_state,
            verify_live_receipt=True,
            verify_live_run_state=True,
        )
        == lineage
    )

    lifecycle_advanced = True
    with pytest.raises(ValueError, match="live bytes changed"):
        amendment._canonical_replacement_publication_failure_lineage(
            lineage,
            resource_parent=parent_state,
            verify_live_receipt=True,
            verify_live_run_state=True,
        )
    reads_after_full_live_rejection = run_state_reads

    successor = tmp_path / "sealed-schema-v3-d"
    successor.mkdir()
    authorization = {
        "schema_version": 3,
        "policy": "post_outcome_resource_bounded_confirmatory_technical_successor_v3",
        "purpose": amendment.RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE,
        "replacement_publication_failure_lineage": lineage,
    }
    evidence = {
        "schema_version": 5,
        "amendment_purpose": amendment.RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE,
        "parent": {"authority_directory": str(parent_state.directory)},
        "resource_bounded_technical_successor_authorization": authorization,
    }
    (successor / amendment._EVIDENCE_FILENAME).write_text(
        json.dumps(evidence, sort_keys=True),
        encoding="utf-8",
    )
    (successor / amendment._SOURCE_TREE_SNAPSHOT).write_text("{}\n", encoding="utf-8")
    (successor / amendment._CONFIRMATORY_CONFIG_SNAPSHOT).write_text(
        "schema_version: 1\n",
        encoding="utf-8",
    )

    verification = SimpleNamespace(valid=True, errors=())
    monkeypatch.setattr(
        amendment,
        "verify_preregistration_amendment",
        lambda _directory: verification,
    )
    monkeypatch.setattr(amendment, "_authority_state", lambda *_args, **_kwargs: parent_state)
    monkeypatch.setattr(
        amendment,
        "_require_resource_bounded_confirmatory_authorization",
        lambda *_args, **_kwargs: {"fixture": "authority-c"},
    )
    monkeypatch.setattr(
        amendment,
        "_snapshot_hashes",
        lambda _directory: {
            "confirmatory_config": {
                "file_sha256": "a" * 64,
                "semantic_sha256": "b" * 64,
            },
            "execution_source": {"manifest_sha256": "c" * 64},
        },
    )
    monkeypatch.setattr(
        amendment,
        "_resource_technical_successor_candidate_directories",
        lambda _parent: (successor.resolve(),),
    )

    observed_modes: list[tuple[bool, bool]] = []

    def canonical_minimal_d(
        value: Mapping[str, Any],
        *,
        parent_state: Any,
        verify_live_receipt: bool,
        verify_live_replacement_run_state: bool = False,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        raw = dict(value)
        observed_modes.append((verify_live_receipt, verify_live_replacement_run_state))
        raw["replacement_publication_failure_lineage"] = (
            amendment._canonical_replacement_publication_failure_lineage(
                raw["replacement_publication_failure_lineage"],
                resource_parent=parent_state,
                verify_live_receipt=verify_live_receipt,
                verify_live_run_state=verify_live_replacement_run_state,
            )
        )
        return raw

    monkeypatch.setattr(
        amendment,
        "_canonical_resource_bounded_technical_successor_authorization",
        canonical_minimal_d,
    )

    assert (
        amendment.require_resource_bounded_technical_successor_authorization(successor)
        == authorization
    )
    assert (
        amendment.require_effective_resource_bounded_confirmatory_authorization(successor)
        == authorization
    )
    assert observed_modes == [(True, False), (True, False)]
    assert run_state_reads == reads_after_full_live_rejection
    assert amendment._atomic_json_bytes(receipt_snapshot) == receipt_bytes
    assert amendment._atomic_json_sha256(receipt_snapshot) == receipt_sha256
    assert lineage == lineage_snapshot

    q_tampered = True
    with pytest.raises(ValueError, match="live receipt changed"):
        amendment.require_resource_bounded_technical_successor_authorization(successor)
    with pytest.raises(ValueError, match="live receipt changed"):
        amendment.require_effective_resource_bounded_confirmatory_authorization(successor)

    assert run_state_reads == reads_after_full_live_rejection
    assert amendment._atomic_json_bytes(receipt_snapshot) == receipt_bytes
    assert amendment._atomic_json_sha256(receipt_snapshot) == receipt_sha256
    assert lineage == lineage_snapshot
