"""Typed action-map tests for resource-bounded checkpoint execution."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch

from histo_audit.experiment import resource_bounded_checkpoint_execution as adapter
from histo_audit.experiment.resource_bounded_checkpoint_execution import (
    prepare_fresh_resource_bounded_checkpoint_execution,
    prepare_successor_resource_bounded_checkpoint_execution,
)
from histo_audit.experiment.resource_bounded_resume import (
    ResumeCheckpointExpectation,
    copy_validated_resume_checkpoints,
    inspect_read_only_resume_predecessor,
)

_DATA_KEYS = (
    "training_data_sha256",
    "reference_validation_data_sha256",
    "training_split_sha256",
    "reference_validation_split_sha256",
)


def _allowlist() -> tuple[ResumeCheckpointExpectation, ...]:
    return tuple(
        ResumeCheckpointExpectation(
            relative_path=f"cells/resource_cnn_{cell}/checkpoints/fold_{fold:02d}.pt",
            cell_id=f"resource_cnn_{cell}",
            fold_id=fold,
            expected_configuration={
                "epochs": 2,
                "seed": 303 + fold,
                "input_variant": "context_rgb",
            },
            expected_model_metadata={
                "architecture": "test.resnet18",
                "input_channels": 3,
            },
            expected_data_and_split_sha256={
                key: f"{index + 1:x}" * 64 for index, key in enumerate(_DATA_KEYS)
            },
        )
        for cell in range(6)
        for fold in range(5)
    )


def _write_checkpoint(
    root: Path,
    expectation: ResumeCheckpointExpectation,
    *,
    completed_epochs: int,
    stopped_early: bool = False,
) -> Path:
    path = root / expectation.relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "configuration": dict(expectation.expected_configuration),
            "model_metadata": dict(expectation.expected_model_metadata),
            "data_and_split_sha256": dict(expectation.expected_data_and_split_sha256),
            "completed_epochs": completed_epochs,
            "early_stopping_state": {"stopped_early": stopped_early},
            "history": [{"epoch": epoch_index} for epoch_index in range(1, completed_epochs + 1)],
        },
        path,
    )
    return path


def _validator(
    path: str | Path,
    *,
    expected_configuration: Mapping[str, Any],
    expected_model_metadata: Mapping[str, Any],
    expected_data_and_split_sha256: Mapping[str, str],
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload["configuration"] != dict(expected_configuration)
        or payload["model_metadata"] != dict(expected_model_metadata)
        or payload["data_and_split_sha256"] != dict(expected_data_and_split_sha256)
    ):
        raise ValueError("checkpoint bindings differ")


def test_resource_fresh_preparation_is_exact_and_filesystem_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_read(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("fresh preparation touched the filesystem")

    monkeypatch.setattr(os.path, "lexists", forbidden_read)
    monkeypatch.setattr(Path, "lstat", forbidden_read)

    prepared = prepare_fresh_resource_bounded_checkpoint_execution(_allowlist())
    contract = prepared.checkpoint_execution_contract

    contract.validate(expected_directive_count=30)
    assert contract.contract_profile == "resource_bounded_confirmatory_exact_30"
    assert contract.execution_mode == "fresh"
    assert len(contract.directives) == 30
    assert all(value.action == "fresh_fit" for value in contract.directives)
    assert prepared.resume_evidence["predecessor_read_performed"] is False
    assert prepared.resume_evidence["automatic_retry_allowed"] is False


def test_resource_successor_derives_mixed_actions_from_verified_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predecessor = tmp_path / "old-run"
    successor = tmp_path / "new-run"
    predecessor.mkdir()
    successor.mkdir()
    expectations = _allowlist()
    incomplete = _write_checkpoint(
        predecessor,
        expectations[0],
        completed_epochs=1,
    )
    terminal = _write_checkpoint(
        predecessor,
        expectations[1],
        completed_epochs=2,
    )
    source_hashes = {
        incomplete: incomplete.read_bytes(),
        terminal: terminal.read_bytes(),
    }
    snapshot = inspect_read_only_resume_predecessor(
        predecessor,
        retry_of_run_id=predecessor.name,
        checkpoint_allowlist=expectations,
        validator=_validator,
    )
    receipt = copy_validated_resume_checkpoints(
        snapshot,
        successor,
        validator=_validator,
    )
    monkeypatch.setattr(
        adapter,
        "validate_confirmatory_checkpoint_artifact",
        _validator,
    )

    prepared = prepare_successor_resource_bounded_checkpoint_execution(
        snapshot,
        receipt,
    )
    contract = prepared.checkpoint_execution_contract
    by_path = {
        f"cells/{value.cell_id}/checkpoints/fold_{value.fold_id:02d}.pt": value
        for value in contract.directives
    }

    assert contract.execution_mode == "successor_resume"
    assert contract.retry_of_run_id == predecessor.name
    assert contract.predecessor_checkpoint_read_performed
    assert contract.predecessor_checkpoint_copy_performed
    assert not contract.outcome_artifacts_read
    assert by_path[expectations[0].relative_path].action == ("resume_incomplete_fit")
    assert by_path[expectations[0].relative_path].next_epoch_index == 1
    assert by_path[expectations[1].relative_path].action == (
        "restore_terminal_checkpoint_without_fit"
    )
    assert sum(value.action == "fresh_fit" for value in contract.directives) == 28
    assert all(path.read_bytes() == payload for path, payload in source_hashes.items())


def test_resource_successor_rejects_changed_copy_without_fresh_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predecessor = tmp_path / "old-run"
    successor = tmp_path / "new-run"
    predecessor.mkdir()
    successor.mkdir()
    expectations = _allowlist()
    _write_checkpoint(predecessor, expectations[0], completed_epochs=1)
    snapshot = inspect_read_only_resume_predecessor(
        predecessor,
        retry_of_run_id=predecessor.name,
        checkpoint_allowlist=expectations,
        validator=_validator,
    )
    receipt = copy_validated_resume_checkpoints(
        snapshot,
        successor,
        validator=_validator,
    )
    monkeypatch.setattr(
        adapter,
        "validate_confirmatory_checkpoint_artifact",
        _validator,
    )
    copied = successor / expectations[0].relative_path
    copied.write_bytes(copied.read_bytes() + b"changed")

    with pytest.raises(RuntimeError):
        prepare_successor_resource_bounded_checkpoint_execution(
            snapshot,
            receipt,
        )
