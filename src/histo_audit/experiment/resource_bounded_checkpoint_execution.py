"""Typed per-fit checkpoint contract for the resource-bounded sensitivity run.

The adapter has no outcome-artifact read surface and never derives a training
decision from checkpoint existence alone.  Fresh decisions come from the complete
predeclared allowlist.  Successor decisions come only after the existing strict
read-only snapshot, physical-copy receipt, and independent destination readback have
all agreed.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import torch

from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.cross_validation.image_oof import (
    ConfirmatoryCheckpointContractError,
    ConfirmatoryCheckpointDirective,
    ConfirmatoryCheckpointExecutionContract,
    ConfirmatoryCheckpointFileIdentity,
    ConfirmatoryCheckpointPhysicalIdentity,
    _hash_private_checkpoint,
    _register_checkpoint_execution_contract,
)
from histo_audit.experiment.resource_bounded_resume import (
    RESOURCE_BOUNDED_RESUME_EVIDENCE_HASH_FIELD,
    BoundResumeCheckpointExpectation,
    ReadOnlyPredecessorSnapshot,
    ResourceBoundedResumeCopyReceipt,
    ResumeCheckpointExpectation,
    _bind_allowlist,
    build_fresh_resource_resume_evidence,
    build_resource_bounded_resume_evidence,
)
from histo_audit.models.cnn import validate_confirmatory_checkpoint_artifact

RESOURCE_BOUNDED_CHECKPOINT_COUNT = 30
RESOURCE_BOUNDED_CNN_CELL_COUNT = 6
RESOURCE_BOUNDED_OOF_FOLD_COUNT = 5


@dataclass(frozen=True, slots=True)
class ResourceBoundedCheckpointExecutionPreparation:
    """Evidence and typed data-plane contract produced in one closed preparation."""

    resume_evidence: Mapping[str, Any]
    checkpoint_execution_contract: ConfirmatoryCheckpointExecutionContract


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_checkpoint_path(
    path: str | Path,
    *,
    expected_configuration: Mapping[str, Any],
    expected_model_metadata: Mapping[str, Any],
    expected_data_and_split_sha256: Mapping[str, str],
) -> None:
    validate_confirmatory_checkpoint_artifact(
        path,
        expected_configuration=expected_configuration,
        expected_model_metadata=expected_model_metadata,
        expected_data_and_split_sha256=expected_data_and_split_sha256,
    )


def _physical_identity(identity: Any) -> ConfirmatoryCheckpointPhysicalIdentity:
    return ConfirmatoryCheckpointPhysicalIdentity(
        device=int(identity.device),
        inode=int(identity.inode),
        size_bytes=int(identity.size_bytes),
        mode=int(identity.mode),
        link_count=int(identity.link_count),
        modified_time_ns=int(identity.modified_time_ns),
        changed_time_ns=int(identity.changed_time_ns),
    )


def _require_exact_bound_allowlist(
    values: tuple[BoundResumeCheckpointExpectation, ...],
) -> None:
    identities = {(value.cell_id, value.fold_id) for value in values}
    cell_ids = {value.cell_id for value in values}
    expected = {
        (cell_id, fold_id)
        for cell_id in cell_ids
        for fold_id in range(RESOURCE_BOUNDED_OOF_FOLD_COUNT)
    }
    if (
        len(values) != RESOURCE_BOUNDED_CHECKPOINT_COUNT
        or len(cell_ids) != RESOURCE_BOUNDED_CNN_CELL_COUNT
        or identities != expected
        or len({value.relative_path.casefold() for value in values})
        != RESOURCE_BOUNDED_CHECKPOINT_COUNT
    ):
        raise ConfirmatoryCheckpointContractError(
            "resource-bounded checkpoint allowlist is not exact 6x5 coverage"
        )


def _directive(
    expectation: BoundResumeCheckpointExpectation,
    *,
    execution_mode: Literal["fresh", "successor_resume"],
    action: Literal[
        "fresh_fit",
        "resume_incomplete_fit",
        "restore_terminal_checkpoint_without_fit",
    ],
    checkpoint_sha256: str | None,
    checkpoint_size_bytes: int | None,
    source_predecessor_checkpoint: ConfirmatoryCheckpointFileIdentity | None,
    destination_imported_checkpoint: ConfirmatoryCheckpointFileIdentity | None,
    completed_epochs_before_fit: int,
    stopped_early_before_fit: bool | None,
) -> ConfirmatoryCheckpointDirective:
    configuration, metadata, data_and_split = expectation.validator_arguments()
    configuration_json = _canonical_json(configuration)
    metadata_json = _canonical_json(metadata)
    data_and_split_json = _canonical_json(data_and_split)
    maximum_epochs = configuration.get("epochs")
    if type(maximum_epochs) is not int:
        raise ConfirmatoryCheckpointContractError(
            "resource checkpoint expectation lacks maximum epochs"
        )
    return ConfirmatoryCheckpointDirective(
        execution_mode=execution_mode,
        cell_id=expectation.cell_id,
        fold_id=expectation.fold_id,
        action=action,
        source_predecessor_checkpoint=source_predecessor_checkpoint,
        destination_imported_checkpoint=destination_imported_checkpoint,
        versioned_checkpoint_output_directory_relative_path=(
            None
            if action == "restore_terminal_checkpoint_without_fit"
            else (f"cells/{expectation.cell_id}/checkpoint_versions/fold_{expectation.fold_id:02d}")
        ),
        checkpoint_execution_manifest_relative_path=(
            f"cells/{expectation.cell_id}/checkpoint_execution/fold_{expectation.fold_id:02d}.json"
        ),
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_size_bytes=checkpoint_size_bytes,
        completed_epochs_before_fit=completed_epochs_before_fit,
        stopped_early_before_fit=stopped_early_before_fit,
        next_epoch_index=completed_epochs_before_fit,
        maximum_epochs=maximum_epochs,
        expected_configuration_json=configuration_json,
        expected_configuration_sha256=_sha256_text(configuration_json),
        expected_model_metadata_json=metadata_json,
        expected_model_metadata_sha256=_sha256_text(metadata_json),
        expected_data_and_split_json=data_and_split_json,
        expected_data_and_split_sha256=_sha256_text(data_and_split_json),
    )


def _execution_contract(
    *,
    execution_mode: Literal["fresh", "successor_resume"],
    retry_of_run_id: str | None,
    directives: Sequence[ConfirmatoryCheckpointDirective],
    predecessor_snapshot_sha256: str | None,
    predecessor_copy_receipt_sha256: str | None,
) -> ConfirmatoryCheckpointExecutionContract:
    ordered = tuple(sorted(directives, key=lambda value: (value.cell_id, value.fold_id)))
    contract = ConfirmatoryCheckpointExecutionContract(
        execution_mode=execution_mode,
        contract_profile="resource_bounded_confirmatory_exact_30",
        retry_of_run_id=retry_of_run_id,
        directives=ordered,
        directives_sha256=canonical_sha256([value.as_dict() for value in ordered]),
        predecessor_checkpoint_read_performed=execution_mode == "successor_resume",
        predecessor_checkpoint_copy_performed=execution_mode == "successor_resume",
        outcome_artifacts_read=False,
        automatic_retry_allowed=False,
        predecessor_snapshot_sha256=predecessor_snapshot_sha256,
        predecessor_copy_receipt_sha256=predecessor_copy_receipt_sha256,
    )
    return _register_checkpoint_execution_contract(
        contract,
        expected_directive_count=RESOURCE_BOUNDED_CHECKPOINT_COUNT,
    )


def prepare_fresh_resource_bounded_checkpoint_execution(
    checkpoint_allowlist: Sequence[ResumeCheckpointExpectation],
) -> ResourceBoundedCheckpointExecutionPreparation:
    """Build exact fresh decisions without reading a predecessor or filesystem."""

    try:
        bound = _bind_allowlist(checkpoint_allowlist)
        evidence = build_fresh_resource_resume_evidence(checkpoint_allowlist)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ConfirmatoryCheckpointContractError(
            "resource-bounded fresh checkpoint preparation failed"
        ) from exc
    _require_exact_bound_allowlist(bound)
    directives = tuple(
        _directive(
            expectation,
            execution_mode="fresh",
            action="fresh_fit",
            checkpoint_sha256=None,
            checkpoint_size_bytes=None,
            source_predecessor_checkpoint=None,
            destination_imported_checkpoint=None,
            completed_epochs_before_fit=0,
            stopped_early_before_fit=None,
        )
        for expectation in bound
    )
    contract = _execution_contract(
        execution_mode="fresh",
        retry_of_run_id=None,
        directives=directives,
        predecessor_snapshot_sha256=None,
        predecessor_copy_receipt_sha256=None,
    )
    return ResourceBoundedCheckpointExecutionPreparation(
        resume_evidence=MappingProxyType(dict(evidence)),
        checkpoint_execution_contract=contract,
    )


def _checkpoint_state(
    path: os.PathLike[str] | str,
    *,
    maximum_epochs: int,
) -> tuple[int, bool]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ConfirmatoryCheckpointContractError(
            "resource successor checkpoint is not a safe Torch payload"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ConfirmatoryCheckpointContractError(
            "resource successor checkpoint payload is not a mapping"
        )
    completed = payload.get("completed_epochs")
    early = payload.get("early_stopping_state")
    history = payload.get("history")
    if (
        type(completed) is not int
        or not 1 <= completed <= maximum_epochs
        or not isinstance(early, Mapping)
        or type(early.get("stopped_early")) is not bool
        or not isinstance(history, (list, tuple))
        or len(history) != completed
        or any(
            not isinstance(row, Mapping) or row.get("epoch") != index
            for index, row in enumerate(history, start=1)
        )
    ):
        raise ConfirmatoryCheckpointContractError(
            "resource successor checkpoint epoch state is invalid"
        )
    return completed, bool(early["stopped_early"])


def prepare_successor_resource_bounded_checkpoint_execution(
    snapshot: ReadOnlyPredecessorSnapshot,
    receipt: ResourceBoundedResumeCopyReceipt,
) -> ResourceBoundedCheckpointExecutionPreparation:
    """Reverify snapshot/copy and derive all actions from copied checkpoint state."""

    if not isinstance(snapshot, ReadOnlyPredecessorSnapshot) or not isinstance(
        receipt,
        ResourceBoundedResumeCopyReceipt,
    ):
        raise ConfirmatoryCheckpointContractError(
            "resource successor preparation requires typed snapshot/copy evidence"
        )
    _require_exact_bound_allowlist(snapshot.expectations)
    try:
        evidence = build_resource_bounded_resume_evidence(
            snapshot,
            receipt,
            validator=_validate_checkpoint_path,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ConfirmatoryCheckpointContractError(
            "resource successor snapshot/copy readback failed"
        ) from exc
    evidence_sha = evidence.get(RESOURCE_BOUNDED_RESUME_EVIDENCE_HASH_FIELD)
    if not isinstance(evidence_sha, str) or len(evidence_sha) != 64:
        raise ConfirmatoryCheckpointContractError(
            "resource successor evidence lacks its semantic hash"
        )
    records = {value.relative_path: value for value in snapshot.records}
    copied = {value.relative_path: value for value in receipt.copied_records}
    if set(records) != {value.relative_path for value in snapshot.expectations} or set(copied) != {
        value.relative_path for value in snapshot.records if value.decision == "resume"
    }:
        raise ConfirmatoryCheckpointContractError(
            "resource successor snapshot/copy universes differ"
        )
    directives: list[ConfirmatoryCheckpointDirective] = []
    for expectation in snapshot.expectations:
        record = records[expectation.relative_path]
        destination = receipt.destination_directory / expectation.relative_path
        if record.decision == "missing_fresh":
            if os.path.lexists(destination):
                raise ConfirmatoryCheckpointContractError(
                    "resource fresh destination unexpectedly exists"
                )
            directives.append(
                _directive(
                    expectation,
                    execution_mode="successor_resume",
                    action="fresh_fit",
                    checkpoint_sha256=None,
                    checkpoint_size_bytes=None,
                    source_predecessor_checkpoint=None,
                    destination_imported_checkpoint=None,
                    completed_epochs_before_fit=0,
                    stopped_early_before_fit=None,
                )
            )
            continue
        copied_record = copied[expectation.relative_path]
        observed_sha, observed_size = _hash_private_checkpoint(
            destination,
            role="resource successor fit checkpoint",
        )
        if (
            observed_sha != copied_record.sha256
            or observed_size != copied_record.size_bytes
            or record.sha256 != copied_record.sha256
            or record.size_bytes != copied_record.size_bytes
        ):
            raise ConfirmatoryCheckpointContractError(
                "resource copied checkpoint changed before action derivation"
            )
        configuration, _, _ = expectation.validator_arguments()
        maximum_epochs = configuration.get("epochs")
        if type(maximum_epochs) is not int:
            raise ConfirmatoryCheckpointContractError(
                "resource checkpoint expectation lacks exact epochs"
            )
        completed, stopped_early = _checkpoint_state(
            destination,
            maximum_epochs=maximum_epochs,
        )
        terminal = stopped_early or completed == maximum_epochs
        directives.append(
            _directive(
                expectation,
                execution_mode="successor_resume",
                action=(
                    "restore_terminal_checkpoint_without_fit"
                    if terminal
                    else "resume_incomplete_fit"
                ),
                checkpoint_sha256=observed_sha,
                checkpoint_size_bytes=observed_size,
                source_predecessor_checkpoint=ConfirmatoryCheckpointFileIdentity(
                    path=Path(snapshot.predecessor_directory / expectation.relative_path),
                    physical_identity=_physical_identity(copied_record.source_identity),
                    size_bytes=copied_record.size_bytes,
                    sha256=copied_record.sha256,
                ),
                destination_imported_checkpoint=ConfirmatoryCheckpointFileIdentity(
                    path=Path(destination),
                    physical_identity=_physical_identity(copied_record.destination_identity),
                    size_bytes=copied_record.size_bytes,
                    sha256=copied_record.sha256,
                ),
                completed_epochs_before_fit=completed,
                stopped_early_before_fit=stopped_early,
            )
        )
    receipt_sha = canonical_sha256(
        {
            "source_snapshot_before_sha256": receipt.source_snapshot_before_sha256,
            "source_snapshot_after_sha256": receipt.source_snapshot_after_sha256,
            "destination_inventory_sha256": receipt.destination_inventory_sha256,
            "resume_evidence_sha256": evidence_sha,
        }
    )
    contract = _execution_contract(
        execution_mode="successor_resume",
        retry_of_run_id=snapshot.retry_of_run_id,
        directives=directives,
        predecessor_snapshot_sha256=snapshot.snapshot_sha256,
        predecessor_copy_receipt_sha256=receipt_sha,
    )
    return ResourceBoundedCheckpointExecutionPreparation(
        resume_evidence=MappingProxyType(dict(evidence)),
        checkpoint_execution_contract=contract,
    )


__all__ = [
    "RESOURCE_BOUNDED_CHECKPOINT_COUNT",
    "ResourceBoundedCheckpointExecutionPreparation",
    "prepare_fresh_resource_bounded_checkpoint_execution",
    "prepare_successor_resource_bounded_checkpoint_execution",
]
