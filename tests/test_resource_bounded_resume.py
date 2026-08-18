from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.experiment import resource_bounded_resume as resume_module
from histo_audit.experiment.resource_bounded_resume import (
    RESOURCE_BOUNDED_READ_SCOPE,
    RESOURCE_BOUNDED_RESUME_EVIDENCE_HASH_FIELD,
    RESOURCE_BOUNDED_RESUME_POLICY,
    ResourceBoundedResumeError,
    ResumeCheckpointExpectation,
    build_fresh_resource_resume_evidence,
    build_resource_bounded_resume_evidence,
    copy_validated_resume_checkpoints,
    inspect_read_only_resume_predecessor,
)

_DATA_AND_SPLIT_KEYS = (
    "training_data_sha256",
    "reference_validation_data_sha256",
    "training_split_sha256",
    "reference_validation_split_sha256",
)


def _expectation(cell_id: str, fold_id: int, *, seed: int = 303) -> ResumeCheckpointExpectation:
    relative_path = f"cells/{cell_id}/checkpoints/fold_{fold_id:02d}.pt"
    return ResumeCheckpointExpectation(
        relative_path=relative_path,
        cell_id=cell_id,
        fold_id=fold_id,
        expected_configuration={
            "epochs": 100,
            "seed": seed,
            "input_variant": "context_rgb",
        },
        expected_model_metadata={
            "architecture": "test.resnet18",
            "input_channels": 3,
            "weight_sha256": "a" * 64,
        },
        expected_data_and_split_sha256={
            key: f"{index + 1:x}" * 64 for index, key in enumerate(_DATA_AND_SPLIT_KEYS)
        },
    )


def _checkpoint_payload(
    expectation: ResumeCheckpointExpectation,
    *,
    marker: str,
) -> dict[str, Any]:
    return {
        "configuration": dict(expectation.expected_configuration),
        "model_metadata": dict(expectation.expected_model_metadata),
        "data_and_split_sha256": dict(expectation.expected_data_and_split_sha256),
        "marker": marker,
    }


def _write_checkpoint(
    root: Path,
    expectation: ResumeCheckpointExpectation,
    *,
    marker: str,
) -> Path:
    path = root / Path(expectation.relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _checkpoint_payload(expectation, marker=marker),
            allow_nan=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _strict_json_validator(
    path: str | Path,
    *,
    expected_configuration: Mapping[str, Any],
    expected_model_metadata: Mapping[str, Any],
    expected_data_and_split_sha256: Mapping[str, str],
) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(payload) != {
        "configuration",
        "model_metadata",
        "data_and_split_sha256",
        "marker",
    }:
        raise ValueError("checkpoint schema differs")
    if (
        payload["configuration"] != dict(expected_configuration)
        or payload["model_metadata"] != dict(expected_model_metadata)
        or payload["data_and_split_sha256"] != dict(expected_data_and_split_sha256)
    ):
        raise ValueError("checkpoint bindings differ")


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    predecessor = tmp_path / "run-old"
    destination = tmp_path / "run-new"
    predecessor.mkdir()
    destination.mkdir()
    return predecessor, destination


def test_mixed_existing_and_missing_checkpoints_copy_only_allowlist_without_outcome_reads(
    tmp_path: Path,
) -> None:
    predecessor, destination = _roots(tmp_path)
    expectations = (
        _expectation("cell-a", 0),
        _expectation("cell-a", 1),
        _expectation("cell-b", 0, seed=304),
    )
    first = _write_checkpoint(predecessor, expectations[0], marker="first")
    third = _write_checkpoint(predecessor, expectations[2], marker="third")
    source_before = {
        path: (
            path.read_bytes(),
            path.stat().st_dev,
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
            path.stat().st_ctime_ns,
        )
        for path in (first, third)
    }

    forbidden = {
        predecessor / "metrics.json": b'{"must_not_be_read": true}',
        predecessor / "ranking.csv": b"must,not,be,read\n",
        predecessor / "oof_predictions.npz": b"not-an-npz",
        predecessor / "cells" / "cell-a" / "checkpoints" / "fold_99.pt": b"extra",
    }
    for path, payload in forbidden.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    validator_paths: list[Path] = []

    def validator(
        path: str | Path,
        *,
        expected_configuration: Mapping[str, Any],
        expected_model_metadata: Mapping[str, Any],
        expected_data_and_split_sha256: Mapping[str, str],
    ) -> None:
        validator_paths.append(Path(path))
        _strict_json_validator(
            path,
            expected_configuration=expected_configuration,
            expected_model_metadata=expected_model_metadata,
            expected_data_and_split_sha256=expected_data_and_split_sha256,
        )

    snapshot = inspect_read_only_resume_predecessor(
        predecessor,
        retry_of_run_id="run-old",
        checkpoint_allowlist=expectations,
        validator=validator,
    )

    assert [record.decision for record in snapshot.records] == [
        "resume",
        "missing_fresh",
        "resume",
    ]
    assert snapshot.total_copy_bytes == first.stat().st_size + third.stat().st_size
    assert {path for path in validator_paths} == {first, third}

    receipt = copy_validated_resume_checkpoints(
        snapshot,
        destination,
        validator=validator,
        chunk_size_bytes=17,
    )
    evidence = build_resource_bounded_resume_evidence(
        snapshot,
        receipt,
        validator=validator,
    )

    assert receipt.source_unchanged is True
    assert receipt.copied_checkpoint_count == 2
    assert receipt.fresh_relative_paths == (expectations[1].relative_path,)
    assert (destination / Path(expectations[0].relative_path)).read_bytes() == first.read_bytes()
    assert not (destination / Path(expectations[1].relative_path)).exists()
    assert (destination / Path(expectations[2].relative_path)).read_bytes() == third.read_bytes()
    assert source_before == {
        path: (
            path.read_bytes(),
            path.stat().st_dev,
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
            path.stat().st_ctime_ns,
        )
        for path in (first, third)
    }
    assert not any((destination / path.relative_to(predecessor)).exists() for path in forbidden)
    assert all(
        (record.source_identity.device, record.source_identity.inode)
        != (record.destination_identity.device, record.destination_identity.inode)
        and record.destination_identity.link_count == 1
        for record in receipt.copied_records
    )

    assert evidence["policy"] == RESOURCE_BOUNDED_RESUME_POLICY
    assert evidence["run_mode"] == "successor_resume"
    assert evidence["read_scope"] == RESOURCE_BOUNDED_READ_SCOPE
    assert evidence["expected_checkpoint_count"] == 3
    assert evidence["imported_checkpoint_count"] == 2
    assert evidence["fresh_checkpoint_count"] == 1
    assert evidence["auto_discovery_used"] is False
    assert evidence["automatic_retry_allowed"] is False
    assert evidence["oof_artifacts_read"] is False
    assert evidence["metrics_artifacts_read"] is False
    assert evidence["ranking_artifacts_read"] is False
    assert all(
        row["relative_path"] not in {str(path.relative_to(predecessor)) for path in forbidden}
        for row in evidence["source_inventory"]
    )
    hash_bound_payload = copy.deepcopy(evidence)
    evidence_hash = hash_bound_payload.pop(RESOURCE_BOUNDED_RESUME_EVIDENCE_HASH_FIELD)
    assert evidence_hash == canonical_sha256(hash_bound_payload)
    assert set(evidence) == set(build_fresh_resource_resume_evidence(expectations))


def test_retry_id_and_canonical_checkpoint_paths_are_exact(tmp_path: Path) -> None:
    predecessor, _ = _roots(tmp_path)
    expectation = _expectation("cell-a", 0)

    with pytest.raises(ResourceBoundedResumeError, match="exactly equal"):
        inspect_read_only_resume_predecessor(
            predecessor,
            retry_of_run_id="different-run",
            checkpoint_allowlist=(expectation,),
            validator=_strict_json_validator,
        )

    unsafe = ResumeCheckpointExpectation(
        relative_path="cells/cell-a/checkpoints/../metrics.json",
        cell_id="cell-a",
        fold_id=0,
        expected_configuration=expectation.expected_configuration,
        expected_model_metadata=expectation.expected_model_metadata,
        expected_data_and_split_sha256=expectation.expected_data_and_split_sha256,
    )
    with pytest.raises(ResourceBoundedResumeError, match="exact canonical"):
        inspect_read_only_resume_predecessor(
            predecessor,
            retry_of_run_id="run-old",
            checkpoint_allowlist=(unsafe,),
            validator=_strict_json_validator,
        )

    for unsafe_cell_id in ("cell.", "CON"):
        unsafe_alias = replace(
            expectation,
            cell_id=unsafe_cell_id,
            relative_path=f"cells/{unsafe_cell_id}/checkpoints/fold_00.pt",
        )
        with pytest.raises(ResourceBoundedResumeError, match="cell/fold identity is unsafe"):
            inspect_read_only_resume_predecessor(
                predecessor,
                retry_of_run_id="run-old",
                checkpoint_allowlist=(unsafe_alias,),
                validator=_strict_json_validator,
            )

    with pytest.raises(ResourceBoundedResumeError, match="duplicate canonical"):
        inspect_read_only_resume_predecessor(
            predecessor,
            retry_of_run_id="run-old",
            checkpoint_allowlist=(
                expectation,
                _expectation("CELL-A", 0),
            ),
            validator=_strict_json_validator,
        )


def test_fresh_evidence_binds_exactly_thirty_missing_paths_without_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expectations = tuple(
        _expectation(f"cell-{cell_id:02d}", fold_id, seed=303 + cell_id)
        for cell_id in range(6)
        for fold_id in range(5)
    )

    def forbidden_source_access(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("fresh evidence must not inspect a predecessor")

    monkeypatch.setattr(
        resume_module,
        "_plain_directory_identity",
        forbidden_source_access,
    )

    evidence = build_fresh_resource_resume_evidence(expectations)

    assert evidence["run_mode"] == "fresh"
    assert evidence["predecessor_directory"] is None
    assert evidence["destination_directory"] is None
    assert evidence["retry_of_run_id"] is None
    assert evidence["predecessor_read_performed"] is False
    assert evidence["strict_checkpoint_validator_invoked"] is False
    assert evidence["expected_checkpoint_count"] == 30
    assert evidence["imported_checkpoint_count"] == 0
    assert evidence["fresh_checkpoint_count"] == 30
    assert evidence["imported_inventory"] == []
    assert evidence["copied_bytes"] == 0
    assert len(set(evidence["fresh_checkpoint_paths"])) == 30
    assert all(
        row["decision"] == "missing_fresh"
        and row["size_bytes"] is None
        and row["sha256"] is None
        and row["identity"] is None
        for row in evidence["source_inventory"]
    )
    hash_bound_payload = copy.deepcopy(evidence)
    evidence_hash = hash_bound_payload.pop(RESOURCE_BOUNDED_RESUME_EVIDENCE_HASH_FIELD)
    assert evidence_hash == canonical_sha256(hash_bound_payload)


def test_fabricated_typed_snapshot_cannot_escape_canonical_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    predecessor, destination = _roots(tmp_path)
    expectation = _expectation("cell-a", 0)
    _write_checkpoint(predecessor, expectation, marker="valid")
    (predecessor / "metrics.json").write_text('{"must_not_be_read": true}', encoding="utf-8")
    snapshot = inspect_read_only_resume_predecessor(
        predecessor,
        retry_of_run_id="run-old",
        checkpoint_allowlist=(expectation,),
        validator=_strict_json_validator,
    )
    forged_expectation = replace(
        snapshot.expectations[0],
        relative_path="../metrics.json",
    )
    forged_snapshot = replace(snapshot, expectations=(forged_expectation,))

    def forbidden_hash(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("forged snapshot reached a file read")

    monkeypatch.setattr(resume_module, "_hash_plain_checkpoint", forbidden_hash)

    with pytest.raises(ResourceBoundedResumeError, match="not canonical"):
        copy_validated_resume_checkpoints(
            forged_snapshot,
            destination,
            validator=_strict_json_validator,
        )


def test_invalid_existing_checkpoint_fails_closed_while_absence_is_fresh(tmp_path: Path) -> None:
    predecessor, _ = _roots(tmp_path)
    invalid = _expectation("cell-a", 0)
    missing = _expectation("cell-a", 1)
    path = _write_checkpoint(predecessor, invalid, marker="invalid")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["configuration"]["epochs"] = 99
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ResourceBoundedResumeError, match="strict checkpoint validation"):
        inspect_read_only_resume_predecessor(
            predecessor,
            retry_of_run_id="run-old",
            checkpoint_allowlist=(invalid, missing),
            validator=_strict_json_validator,
        )

    path.unlink()
    snapshot = inspect_read_only_resume_predecessor(
        predecessor,
        retry_of_run_id="run-old",
        checkpoint_allowlist=(invalid, missing),
        validator=_strict_json_validator,
    )
    assert all(record.decision == "missing_fresh" for record in snapshot.records)


def test_source_tamper_after_snapshot_fails_before_copy(tmp_path: Path) -> None:
    predecessor, destination = _roots(tmp_path)
    expectation = _expectation("cell-a", 0)
    source = _write_checkpoint(predecessor, expectation, marker="before")
    snapshot = inspect_read_only_resume_predecessor(
        predecessor,
        retry_of_run_id="run-old",
        checkpoint_allowlist=(expectation,),
        validator=_strict_json_validator,
    )

    source.write_text(
        json.dumps(_checkpoint_payload(expectation, marker="tampered"), sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ResourceBoundedResumeError, match="changed before copy"):
        copy_validated_resume_checkpoints(
            snapshot,
            destination,
            validator=_strict_json_validator,
        )
    assert not (destination / Path(expectation.relative_path)).exists()


def test_hardlinked_source_checkpoint_is_rejected(tmp_path: Path) -> None:
    predecessor, _ = _roots(tmp_path)
    expectation = _expectation("cell-a", 0)
    source = _write_checkpoint(predecessor, expectation, marker="linked")
    alias = predecessor / "checkpoint-alias.pt"
    os.link(source, alias)

    with pytest.raises(ResourceBoundedResumeError, match="private regular file"):
        inspect_read_only_resume_predecessor(
            predecessor,
            retry_of_run_id="run-old",
            checkpoint_allowlist=(expectation,),
            validator=_strict_json_validator,
        )


def test_symlinked_source_checkpoint_is_rejected(tmp_path: Path) -> None:
    predecessor, _ = _roots(tmp_path)
    expectation = _expectation("cell-a", 0)
    target = tmp_path / "outside.pt"
    target.write_text(
        json.dumps(_checkpoint_payload(expectation, marker="outside"), sort_keys=True),
        encoding="utf-8",
    )
    source = predecessor / Path(expectation.relative_path)
    source.parent.mkdir(parents=True)
    try:
        source.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"host cannot create a test symlink: {exc}")

    with pytest.raises(ResourceBoundedResumeError, match="link or reparse"):
        inspect_read_only_resume_predecessor(
            predecessor,
            retry_of_run_id="run-old",
            checkpoint_allowlist=(expectation,),
            validator=_strict_json_validator,
        )


def test_symlinked_parent_chain_is_rejected_for_source_and_destination(tmp_path: Path) -> None:
    real_source_parent = tmp_path / "real-source"
    real_source_parent.mkdir()
    predecessor = real_source_parent / "run-old"
    predecessor.mkdir()
    destination = tmp_path / "run-new"
    destination.mkdir()
    expectation = _expectation("cell-a", 0)
    _write_checkpoint(predecessor, expectation, marker="valid")
    source_alias = tmp_path / "source-alias"
    try:
        source_alias.symlink_to(real_source_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"host cannot create a directory symlink: {exc}")

    with pytest.raises(ResourceBoundedResumeError, match="link or reparse component"):
        inspect_read_only_resume_predecessor(
            source_alias / "run-old",
            retry_of_run_id="run-old",
            checkpoint_allowlist=(expectation,),
            validator=_strict_json_validator,
        )

    snapshot = inspect_read_only_resume_predecessor(
        predecessor,
        retry_of_run_id="run-old",
        checkpoint_allowlist=(expectation,),
        validator=_strict_json_validator,
    )
    real_destination_parent = tmp_path / "real-destination"
    real_destination_parent.mkdir()
    (real_destination_parent / "run-new").mkdir()
    destination_alias = tmp_path / "destination-alias"
    destination_alias.symlink_to(real_destination_parent, target_is_directory=True)

    with pytest.raises(ResourceBoundedResumeError, match="link or reparse component"):
        copy_validated_resume_checkpoints(
            snapshot,
            destination_alias / "run-new",
            validator=_strict_json_validator,
        )


def test_source_change_during_copy_fails_without_a_receipt(tmp_path: Path) -> None:
    predecessor, destination = _roots(tmp_path)
    expectations = (
        _expectation("cell-a", 0),
        _expectation("cell-b", 0, seed=304),
    )
    first_source = _write_checkpoint(predecessor, expectations[0], marker="first")
    _write_checkpoint(predecessor, expectations[1], marker="second")
    snapshot = inspect_read_only_resume_predecessor(
        predecessor,
        retry_of_run_id="run-old",
        checkpoint_allowlist=expectations,
        validator=_strict_json_validator,
    )
    changed = False

    def mutating_validator(
        path: str | Path,
        *,
        expected_configuration: Mapping[str, Any],
        expected_model_metadata: Mapping[str, Any],
        expected_data_and_split_sha256: Mapping[str, str],
    ) -> None:
        nonlocal changed
        _strict_json_validator(
            path,
            expected_configuration=expected_configuration,
            expected_model_metadata=expected_model_metadata,
            expected_data_and_split_sha256=expected_data_and_split_sha256,
        )
        candidate = Path(path)
        if (
            destination in candidate.parents
            and candidate.name == "fold_00.pt"
            and not changed
            and "cell-b" in candidate.parts
        ):
            # This is the second destination validation, after the first source was copied.
            changed = True
            first_source.write_text(
                json.dumps(
                    _checkpoint_payload(expectations[0], marker="changed-during-copy"),
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

    with pytest.raises(ResourceBoundedResumeError, match="changed after copy"):
        copy_validated_resume_checkpoints(
            snapshot,
            destination,
            validator=mutating_validator,
            chunk_size_bytes=19,
        )
    assert changed is True
    # The failed destination cannot be silently retried because copying is no-overwrite.
    assert all(
        (destination / Path(expectation.relative_path)).exists() for expectation in expectations
    )


def test_destination_checkpoint_is_never_overwritten(tmp_path: Path) -> None:
    predecessor, destination = _roots(tmp_path)
    expectation = _expectation("cell-a", 0)
    _write_checkpoint(predecessor, expectation, marker="source")
    snapshot = inspect_read_only_resume_predecessor(
        predecessor,
        retry_of_run_id="run-old",
        checkpoint_allowlist=(expectation,),
        validator=_strict_json_validator,
    )
    existing = _write_checkpoint(destination, expectation, marker="destination-owned")
    before = existing.read_bytes()

    with pytest.raises(ResourceBoundedResumeError, match="already exists"):
        copy_validated_resume_checkpoints(
            snapshot,
            destination,
            validator=_strict_json_validator,
        )
    assert existing.read_bytes() == before


def test_fabricated_copy_receipt_is_recomputed_against_destination(tmp_path: Path) -> None:
    predecessor, destination = _roots(tmp_path)
    expectation = _expectation("cell-a", 0)
    _write_checkpoint(predecessor, expectation, marker="source")
    snapshot = inspect_read_only_resume_predecessor(
        predecessor,
        retry_of_run_id="run-old",
        checkpoint_allowlist=(expectation,),
        validator=_strict_json_validator,
    )
    receipt = copy_validated_resume_checkpoints(
        snapshot,
        destination,
        validator=_strict_json_validator,
    )

    wrong_policy = replace(receipt, copy_policy="forged-copy-policy")
    with pytest.raises(ResourceBoundedResumeError, match="lineage differs"):
        build_resource_bounded_resume_evidence(
            snapshot,
            wrong_policy,
            validator=_strict_json_validator,
        )

    real_record = receipt.copied_records[0]
    fake_identity = replace(
        real_record.destination_identity,
        inode=real_record.destination_identity.inode + 1,
    )
    fake_record = replace(real_record, destination_identity=fake_identity)
    fake_records = (fake_record,)
    self_consistent_forgery = replace(
        receipt,
        copied_records=fake_records,
        destination_inventory_sha256=canonical_sha256(
            [record.evidence_dict() for record in fake_records]
        ),
    )
    with pytest.raises(ResourceBoundedResumeError, match="differs from filesystem"):
        build_resource_bounded_resume_evidence(
            snapshot,
            self_consistent_forgery,
            validator=_strict_json_validator,
        )
